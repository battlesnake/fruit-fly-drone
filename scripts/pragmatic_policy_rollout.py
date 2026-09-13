"""Training-only native-policy collection for the complete varied gate course.

The actor sees RGB and roll/pitch only. Physical state, course roles and compact
critic features are stored for training, never fed back as actor observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as functional
import train_variable_height_hover as hover_train
from pragmatic_correlated_exploration import ar1_conditional, joint_log_prob
from pragmatic_recurrent_policy_gradient import reward_to_go

from flydrone.gate import AnnularGate, render_annular_gates_rgb
from flydrone.gate_course import classify_course_step
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, QuadState, StickState


def stick_values(state):
    return state.joint_position, state.joint_velocity, state.position, state.velocity


class CourseRewardTracker:
    """Incremental existing course fitness; ground/invalid makes a row absorbing."""

    def __init__(self, episodes, gate_count, device):
        self.gate_count = gate_count
        self.failed = torch.zeros(episodes, device=device, dtype=torch.bool)
        self.absorbed = torch.zeros_like(self.failed)
        self.ground = torch.zeros_like(self.failed)
        self.invalid = torch.zeros_like(self.failed)
        self.ring = torch.zeros_like(self.failed)
        self.wrong_order = torch.zeros_like(self.failed)
        self.wrong_direction = torch.zeros_like(self.failed)
        self.first = torch.zeros_like(self.failed)
        self.passed = torch.zeros(episodes, gate_count, device=device, dtype=torch.bool)
        self.prefix = torch.zeros(episodes, device=device)
        self.centering = torch.zeros_like(self.prefix)

    def step(self, events, proposed_state, gate_config):
        live = ~self.absorbed
        ground = live & (proposed_state.position[:, 2] <= 0.03)
        invalid = live & ~hover_train.state_is_valid(proposed_state)
        unsafe = ground | invalid
        failure = live & (events.failed | unsafe)
        reward = -2.0 * (failure & ~self.failed) - 25.0 * unsafe
        self.failed |= failure
        self.absorbed |= unsafe
        self.ground |= ground
        self.invalid |= invalid
        self.ring |= live & events.ring_collision.any(1)
        self.wrong_order |= live & events.wrong_order.any(1)
        self.wrong_direction |= live & events.wrong_direction.any(1)
        self.passed |= events.passed & live[:, None]
        clean_pass = events.passed & (live & ~self.failed)[:, None]
        self.first |= clean_pass[:, 0]
        prefix = clean_pass.sum(1)
        radius = torch.sqrt(events.crossing_lateral.square() + events.crossing_vertical.square())
        centered = (1 - radius / (gate_config.inner_radius - gate_config.drone_radius)).clamp(0, 1)
        centering = torch.where(clean_pass, centered, 0).sum(1)
        self.prefix += prefix
        self.centering += centering
        return reward + prefix + 0.2 * centering / self.gate_count

    def clean(self):
        return self.passed.all(1) & ~self.failed


def pre_action_critic_features(state, sticks, gates, current, failed, time_fraction,
                               previous_residual, native_mean):
    """Privileged critic only; no current innovation/action is an argument."""
    gate_positions = torch.stack([gate.center - state.position for gate in gates], dim=1)
    gate_headings = torch.stack([torch.stack((gate.yaw.sin(), gate.yaw.cos()), -1)
                                 for gate in gates], dim=1)
    role = functional.one_hot(current, num_classes=len(gates) + 1).to(state.position.dtype)
    return torch.cat((
        *state.as_tuple(), *stick_values(sticks), gate_positions.flatten(1),
        gate_headings.flatten(1), role, failed[:, None].float(),
        state.position.new_full((len(current), 1), time_fraction),
        previous_residual, native_mean,
    ), dim=1)


@dataclass
class PolicyRollout:
    states: tuple[torch.Tensor, ...]  # time, episode, physical component
    gates: tuple[AnnularGate, ...]
    current: torch.Tensor
    latents: torch.Tensor
    old_means: torch.Tensor
    old_log_prob: torch.Tensor  # UNSQUASHED latent density, jointly across four axes
    rewards: torch.Tensor
    returns: torch.Tensor
    valid: torch.Tensor
    critic_features: torch.Tensor
    stationary_std: tuple[float, ...] | None
    rho: float
    warmup_steps: int
    metrics: dict
    sink_features: torch.Tensor | None = None  # Includes neural warmup, then every command.
    sink_parent_nodes: torch.Tensor | None = None

    def select(self, rows, device):
        rows = torch.as_tensor(rows, device=self.valid.device, dtype=torch.long)
        def take(value):
            return value.index_select(1, rows).to(device)
        fields = {name: take(getattr(self, name)) for name in (
            "current", "latents", "old_means", "old_log_prob", "rewards", "returns",
            "valid", "critic_features",
        )}
        return replace(
            self, states=tuple(take(value) for value in self.states),
            sink_features=take(self.sink_features) if self.sink_features is not None else None,
            sink_parent_nodes=(self.sink_parent_nodes.to(device)
                               if self.sink_parent_nodes is not None else None),
            gates=tuple(AnnularGate(g.center.index_select(0, rows).to(device),
                                   g.yaw.index_select(0, rows).to(device)) for g in self.gates),
            metrics={"scope": "selected replay rows, not original bank metrics"}, **fields,
        )

    def observation(self, time, camera, gate_config):
        state = QuadState(*(value[time] for value in self.states))
        image = render_annular_gates_rgb(state, self.gates, current_gate_index=self.current[time],
                                         camera=camera, gate_config=gate_config)
        return image, state.euler[:, :2]


@torch.no_grad()
def collect_policy_rollout(controller, cases, gates, *, camera, config, gate_config,
                           stationary_std=(0.006, 0.002, 0.001, 0.0025),
                           tau=0.6, noise_seed=1, seconds=30, warmup_steps=10,
                           record_sink_parents=None):
    """Collect from zero neural state; fixed weights, complete clean-flight tails.

stationary_std=None is a deterministic diagnostic, not data for Gaussian-policy
learning. Ground/invalid ends only that row's rewards/actions, never resets it.
Finite pre-contact physical state pads absorbed rows; the causing command remains
valid. Ring/order failures continue, so later ground penalties are not erased.
"""
    if not math.isclose(config.dt, 0.01) or not math.isfinite(tau) or tau < 0:
        raise ValueError("need 100 Hz physics and nonnegative finite correlation time")
    steps = round(seconds * 50)
    if steps < 1 or warmup_steps < 0:
        raise ValueError("positive flight duration and nonnegative warmup required")
    device, count = cases.side.device, len(cases.side)
    state = QuadState(*(value.clone() for value in cases.state.as_tuple()))
    sticks = StickState(*(value.clone() for value in stick_values(cases.sticks)))
    neural = controller.initial_state(count, device=device, dtype=state.position.dtype)
    sink_features = []
    if record_sink_parents is not None:
        record_sink_parents = record_sink_parents.to(device=device)
        if (record_sink_parents.ndim != 1 or record_sink_parents.dtype != torch.long
                or not len(record_sink_parents) or int(record_sink_parents.min()) < 0
                or int(record_sink_parents.max()) >= neural.shape[1]):
            raise ValueError("invalid sink parent neuron indices")

    def record_parents():
        if record_sink_parents is not None:
            sink_features.append(torch.tanh(neural[:, record_sink_parents]).clone())
    current = torch.zeros(count, device=device, dtype=torch.long)
    tracker = CourseRewardTracker(count, len(gates), device)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    rho = math.exp(-0.02 / tau) if tau else 0.0
    initial_image = render_annular_gates_rgb(state, gates, current_gate_index=current,
                                             camera=camera, gate_config=gate_config)
    for _ in range(warmup_steps):
        record_parents()
        _, neural = controller(initial_image, state.euler[:, :2], neural)
    previous_mean = previous_latent = None
    histories = [[] for _ in state.as_tuple()]
    roles, latents, means, log_probs, rewards, valid, features = ([] for _ in range(7))
    for time in range(steps):
        for history, value in zip(histories, state.as_tuple(), strict=True):
            history.append(value.clone())
        roles.append(current.clone())
        valid.append((~tracker.absorbed).clone())
        image = render_annular_gates_rgb(state, gates, current_gate_index=current,
                                         camera=camera, gate_config=gate_config)
        record_parents()
        native, neural = controller(image, state.euler[:, :2], neural)
        mean = torch.atanh(native.clamp(-0.999999, 0.999999))
        if not bool(mean.isfinite().all() & neural.isfinite().all()):
            raise FloatingPointError("nonfinite actor; discard rollout")
        previous_residual = (torch.zeros_like(mean) if previous_mean is None
                             else previous_latent - previous_mean)
        # Construct the baseline features BEFORE drawing the current innovation.
        features.append(pre_action_critic_features(
            state, sticks, gates, current, tracker.failed, time / steps, previous_residual, mean,
        ))
        if stationary_std is not None:
            conditional, std = ar1_conditional(
                mean, mean if previous_mean is None else previous_mean,
                mean if previous_latent is None else previous_latent,
                stationary_std, rho,
                torch.full((count,), time == 0, device=device, dtype=torch.bool),
            )
            latent = conditional + std * torch.randn(mean.shape, device=device,
                                                     dtype=mean.dtype, generator=generator)
            motor = latent.tanh()
            log_prob = joint_log_prob(latent, conditional, std, squashed=False)
        else:
            latent, motor, log_prob = mean, native, mean.new_zeros(count)
        means.append(mean)
        latents.append(latent)
        log_probs.append(log_prob)
        reward = mean.new_zeros(count)
        for _ in range(2):
            live = ~tracker.absorbed
            rc, proposed_sticks = legs(motor, sticks)
            proposed_state = quad(rc, state, cases.mass_scale)
            event = classify_course_step(state.position, proposed_state.position,
                                          gates, current, gate_config)
            reward += tracker.step(event, proposed_state, gate_config)
            current = torch.where(live, event.next_gate_index, current)
            # Terminal/invalid proposed states are never replay padding or actor inputs.
            keep = (~tracker.absorbed)[:, None]
            state = QuadState(*(torch.where(keep, new, old) for new, old in
                                zip(proposed_state.as_tuple(), state.as_tuple(), strict=True)))
            sticks = StickState(*(torch.where(keep, new, old) for new, old in zip(
                stick_values(proposed_sticks), stick_values(sticks), strict=True,
            )))
        if time == steps - 1:
            reward += 5.0 * tracker.clean()
        rewards.append(reward)
        previous_mean, previous_latent = mean, latent
    reward_tensor, valid_tensor = torch.stack(rewards), torch.stack(valid)
    metrics = dict(
        scope="training rollout, absorbing only after ground/invalid", episodes=count,
        seconds=seconds, stochastic=stationary_std is not None,
        clean_completions=int(tracker.clean().sum()),
        clean_by_side=[int(tracker.clean()[cases.side < 0].sum()),
                       int(tracker.clean()[cases.side > 0].sum())],
        clean_first_gates=int(tracker.first.sum()), clean_prefix_gates=int(tracker.prefix.sum()),
        ground_contacts=int(tracker.ground.sum()), invalid_episodes=int(tracker.invalid.sum()),
        ring_contact_episodes=int(tracker.ring.sum()),
        wrong_order_episodes=int(tracker.wrong_order.sum()),
        wrong_direction_episodes=int(tracker.wrong_direction.sum()),
        search_fitness=float(reward_tensor.sum(0).mean()), valid_commands=int(valid_tensor.sum()),
        critic_is_privileged_training_only=True,
    )
    result = PolicyRollout(
        states=tuple(torch.stack(history).cpu() for history in histories),
        gates=tuple(AnnularGate(g.center.detach().cpu(), g.yaw.detach().cpu()) for g in gates),
        current=torch.stack(roles).cpu(), latents=torch.stack(latents).cpu(),
        old_means=torch.stack(means).cpu(), old_log_prob=torch.stack(log_probs).cpu(),
        rewards=reward_tensor.cpu(), returns=reward_to_go(reward_tensor, valid_tensor).cpu(),
        sink_features=torch.stack(sink_features).cpu() if sink_features else None,
        sink_parent_nodes=(record_sink_parents.cpu()
                           if record_sink_parents is not None else None),
        valid=valid_tensor.cpu(), critic_features=torch.stack(features).cpu(),
        stationary_std=None if stationary_std is None else tuple(stationary_std),
        rho=rho, warmup_steps=warmup_steps, metrics=metrics,
    )
    for value in (*result.states, result.latents, result.old_means, result.old_log_prob,
                  result.rewards, result.returns, result.critic_features):
        if not bool(value.isfinite().all()):
            raise FloatingPointError("nonfinite rollout or padding; discard rollout")
    return result
