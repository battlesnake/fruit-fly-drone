#!/usr/bin/env python3
"""Train native motor-input parameters with recurrent PPO and a privileged critic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
    sample_matched_cases,
)
from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    evaluate_policy_batch,
    load_controller,
    motor_interface_spec,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    gate_coordinates,
    render_annular_gate,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    rotation_matrix,
)


@dataclass(frozen=True)
class MotorSpec:
    edges: Tensor
    nodes: Tensor
    baseline_edges: Tensor
    baseline_biases: Tensor


class MotorParameters(nn.Module):
    """Independently trainable native parameters compiled into the graph after search."""

    def __init__(self, spec: MotorSpec) -> None:
        super().__init__()
        self.edge_magnitudes = nn.Parameter(spec.baseline_edges.clone())
        self.biases = nn.Parameter(spec.baseline_biases.clone())

    @torch.no_grad()
    def project(self) -> None:
        self.edge_magnitudes.clamp_(0.0, 8.0)
        self.biases.clamp_(-2.0, 2.0)

    def vector(self) -> Tensor:
        return torch.cat((self.edge_magnitudes, self.biases))


class PrivilegedCritic(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def privileged_critic_input(physical: Tensor, neural: Tensor) -> Tensor:
    return torch.cat((physical, neural / 5.0), dim=-1)


@dataclass
class PPORollout:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    stick_position: Tensor
    critic_features: Tensor
    latent_actions: Tensor
    old_latent_mean: Tensor
    old_log_prob: Tensor
    rewards: Tensor
    values: Tensor
    valid: Tensor
    done: Tensor
    neural_before: Tensor
    mass_scale: Tensor
    summary: dict[str, Any]
    outcomes: dict[str, Tensor]
    advantages: Tensor | None = None
    returns: Tensor | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "recurrent-ppo-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--rollout-seconds", type=float, default=8.0)
    parser.add_argument("--chunk-steps", type=int, default=32)
    parser.add_argument("--burn-in-steps", type=int, default=8)
    parser.add_argument("--optimization-epochs", type=int, default=2)
    parser.add_argument("--sequence-minibatch", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--critic-warmup-epochs", type=int, default=2)
    parser.add_argument("--ppo-clip", type=float, default=0.10)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.995)
    parser.add_argument("--exploration-sigma", type=float, default=0.03)
    parser.add_argument("--minimum-exploration-sigma", type=float, default=0.001)
    parser.add_argument("--maximum-exploration-success-drop", type=float, default=0.10)
    parser.add_argument("--light-weight-iterations", type=int, default=5)
    parser.add_argument("--light-advantage-weight", type=float, default=2.0)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--checkpoint-iteration", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--validation-seconds", type=float, default=8.0)
    parser.add_argument("--checkpoint-light-improvement", type=float, default=0.05)
    parser.add_argument("--checkpoint-heavy-margin", type=float, default=0.03)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--final-seconds", type=float, default=12.0)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--final-heavy-margin", type=float, default=0.02)
    parser.add_argument("--gradient-check-episodes", type=int, default=8)
    parser.add_argument("--gradient-check-steps", type=int, default=25)
    parser.add_argument(
        "--gradient-check-scales", type=float, nargs=3, default=(1.0e-3, 3.0e-4, 1.0e-4)
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    parser.add_argument("--rollout-seed", type=int, default=800_031)
    parser.add_argument("--validation-seed", type=int, default=810_031)
    parser.add_argument("--final-seed", type=int, default=820_031)
    parser.add_argument("--optimization-seed", type=int, default=83_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.iterations,
        args.episodes,
        args.rollout_seconds,
        args.chunk_steps,
        args.optimization_epochs,
        args.sequence_minibatch,
        args.actor_learning_rate,
        args.critic_learning_rate,
        args.critic_warmup_epochs,
        args.ppo_clip,
        args.target_kl,
        args.gradient_norm_cap,
        args.gamma,
        args.gae_lambda,
        args.exploration_sigma,
        args.minimum_exploration_sigma,
        args.maximum_exploration_success_drop,
        args.light_weight_iterations,
        args.light_advantage_weight,
        args.validation_interval,
        args.checkpoint_iteration,
        args.validation_episodes,
        args.validation_seconds,
        args.checkpoint_light_improvement,
        args.checkpoint_heavy_margin,
        args.final_episodes,
        args.final_seconds,
        args.final_light_improvement,
        args.final_heavy_margin,
        args.gradient_check_episodes,
        args.gradient_check_steps,
        args.gradient_check_tolerance,
        *args.gradient_check_scales,
    )
    if min(positive) <= 0.0:
        raise SystemExit("PPO sizes, rates, scales, and thresholds must be positive")
    if args.burn_in_steps < 0 or args.burn_in_steps >= args.chunk_steps:
        raise SystemExit("burn-in must be nonnegative and shorter than a training chunk")
    if args.checkpoint_iteration > args.iterations:
        raise SystemExit("checkpoint iteration must fit within the iteration budget")
    if args.checkpoint_iteration % args.validation_interval:
        raise SystemExit("checkpoint iteration must coincide with validation")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("gamma and GAE lambda must be in (0, 1]")
    steps = round(args.rollout_seconds / dt)
    if steps % args.chunk_steps:
        raise SystemExit("rollout steps must be divisible by chunk steps")
    for name in (
        "episodes",
        "validation_episodes",
        "final_episodes",
        "gradient_check_episodes",
    ):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")


def motor_spec(controller: ConnectomeController) -> MotorSpec:
    nodes = torch.unique(controller.pool_indices, sorted=True)
    incoming = torch.isin(controller.edge_post, nodes)
    edges = torch.nonzero(incoming, as_tuple=False).flatten()
    return MotorSpec(
        edges=edges,
        nodes=nodes,
        baseline_edges=controller.edge_magnitude[edges].detach().clone(),
        baseline_biases=controller.bias[nodes].detach().clone(),
    )


def parameter_sha256(actor: MotorParameters) -> str:
    values = actor.vector().detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def load_actor_vector(actor: MotorParameters, vector: Tensor) -> None:
    edge_count = actor.edge_magnitudes.numel()
    with torch.no_grad():
        actor.edge_magnitudes.copy_(vector[:edge_count])
        actor.biases.copy_(vector[edge_count:])
    actor.project()


def controller_step(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    image: Tensor,
    roll_pitch: Tensor,
    neural: Tensor,
    specific_force: Tensor,
    stick_position: Tensor,
) -> tuple[Tensor, Tensor]:
    activity = torch.tanh(neural)
    messages = activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude
    messages[:, spec.edges] = (
        activity[:, controller.edge_pre[spec.edges]]
        * controller.edge_sign[spec.edges]
        * actor.edge_magnitudes
    )
    recurrent = torch.zeros_like(neural).index_add(1, controller.edge_post, messages)
    drive = (
        recurrent
        + controller.bias
        + controller.sensory_drive(image, roll_pitch, specific_force, stick_position)
    )
    drive[:, spec.nodes] += actor.biases - spec.baseline_biases
    target = 5.0 * torch.tanh(drive / 5.0)
    alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant)
    next_neural = neural + alpha * (target - neural)
    return controller.motor_drive(next_neural), next_neural


def latent_mean(motor_mean: Tensor) -> Tensor:
    return torch.atanh(motor_mean.clamp(-0.999999, 0.999999))


def squashed_log_prob(mean: Tensor, latent: Tensor, action: Tensor, sigma: float) -> Tensor:
    variance = sigma * sigma
    gaussian = -0.5 * ((latent - mean).square() / variance + math.log(2.0 * math.pi * variance))
    jacobian = torch.log1p(-action.square() + 1.0e-6)
    return (gaussian - jacobian).sum(dim=-1)


def critic_features(
    state: QuadState,
    gate: AnnularGate,
    mass_scale: Tensor,
    stick_position: Tensor,
    stick_velocity: Tensor,
    passed: Tensor,
    maximum_progress: Tensor,
    maximum_approach: Tensor,
    saturation_fraction: Tensor,
    time_fraction: float,
    hover_config: HoverConfig,
) -> Tensor:
    relative_world = gate.center - state.position
    relative_body = torch.einsum(
        "bij,bj->bi", rotation_matrix(state.euler).transpose(1, 2), relative_world
    )
    relative_yaw = gate.yaw - state.euler[:, 2]
    time_value = torch.full_like(mass_scale, time_fraction)
    return torch.cat(
        (
            state.position / 5.0,
            state.velocity / 5.0,
            state.euler / math.pi,
            state.rates
            / state.rates.new_tensor(
                (
                    hover_config.max_roll_pitch_rate,
                    hover_config.max_roll_pitch_rate,
                    hover_config.max_yaw_rate,
                )
            ),
            state.actuator,
            state.specific_force / 9.81,
            stick_position,
            stick_velocity / 10.0,
            relative_body / 5.0,
            torch.sin(relative_yaw)[:, None],
            torch.cos(relative_yaw)[:, None],
            ((mass_scale - 1.0) / 0.08)[:, None],
            time_value[:, None],
            passed.float()[:, None],
            maximum_progress[:, None],
            maximum_approach[:, None],
            saturation_fraction[:, None],
        ),
        dim=1,
    )


def _mean(values: Tensor, mask: Tensor) -> float | None:
    return float(values[mask].mean()) if bool(mask.any()) else None


def outcome_summary(
    success: Tensor,
    passed: Tensor,
    collision: Tensor,
    missed: Tensor,
    crossing_radial: Tensor,
    maximum_progress: Tensor,
    maximum_approach: Tensor,
    saturation_fraction: Tensor,
    codes: Tensor,
    episode_reward: Tensor,
) -> dict[str, Any]:
    lower = ~codes.bitwise_and(1).bool()
    negative_side = ~codes.bitwise_and(2).bool()
    negative_obliquity = ~codes.bitwise_and(4).bool()
    crossed = ~crossing_radial.isnan()
    return {
        "success_rate": float(success.float().mean()),
        "light_success_rate": _mean(success.float(), lower),
        "heavy_success_rate": _mean(success.float(), ~lower),
        "negative_lateral_success_rate": _mean(success.float(), negative_side),
        "positive_lateral_success_rate": _mean(success.float(), ~negative_side),
        "negative_obliquity_success_rate": _mean(success.float(), negative_obliquity),
        "positive_obliquity_success_rate": _mean(success.float(), ~negative_obliquity),
        "clean_pass_rate": float((passed & ~collision & ~missed).float().mean()),
        "ring_collision_rate": float(collision.float().mean()),
        "miss_rate": float(missed.float().mean()),
        "plane_crossing_rate": float(crossed.float().mean()),
        "crossing_radial_mean_m": _mean(crossing_radial, crossed),
        "progress_mean": float(maximum_progress.mean()),
        "approach_score_mean": float(maximum_approach.mean()),
        "stick_saturation_fraction": float(saturation_fraction.mean()),
        "episode_reward_mean": float(episode_reward.mean()),
    }


def initialize_outcomes(cases: BalancedCases, gate_config: GateConfig) -> dict[str, Tensor]:
    episodes = len(cases.mass_scale)
    device = cases.mass_scale.device
    initial_signed, _, initial_vertical = gate_coordinates(cases.state.position, cases.gate)
    _, initial_lateral, _ = gate_coordinates(cases.state.position, cases.gate)
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    initial_approach = torch.exp(
        -0.5
        * (torch.sqrt(initial_lateral.square() + initial_vertical.square()) / clean_radius).square()
    )
    return {
        "passed": torch.zeros(episodes, dtype=torch.bool, device=device),
        "collision": torch.zeros(episodes, dtype=torch.bool, device=device),
        "missed": torch.zeros(episodes, dtype=torch.bool, device=device),
        "lifted": torch.zeros(episodes, dtype=torch.bool, device=device),
        "recontact": torch.zeros(episodes, dtype=torch.bool, device=device),
        "cleared": torch.zeros(episodes, dtype=torch.bool, device=device),
        "pass_step": torch.full((episodes,), -1, dtype=torch.long, device=device),
        "crossing_radial": torch.full((episodes,), float("nan"), device=device),
        "maximum_tilt": torch.zeros(episodes, device=device),
        "saturation_steps": torch.zeros(episodes, device=device),
        "maximum_progress": torch.zeros(episodes, device=device),
        "maximum_approach": initial_approach,
        "previous_potential": torch.zeros(episodes, device=device),
        "initial_signed": initial_signed,
        "clean_radius": torch.full((episodes,), clean_radius, device=device),
    }


def update_outcomes(
    outcomes: dict[str, Tensor],
    previous_position: Tensor,
    state: QuadState,
    gate: AnnularGate,
    stick_position: Tensor,
    step: int,
    gate_config: GateConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    pass_now, collision_now, miss_now = classify_gate_crossing(
        previous_position, state.position, gate, gate_config
    )
    new_crossing = (pass_now | collision_now | miss_now) & outcomes["crossing_radial"].isnan()
    if bool(new_crossing.any()):
        _, lateral_crossing, vertical_crossing = crossing_coordinates(
            previous_position, state.position, gate
        )
        radial = torch.sqrt(lateral_crossing.square() + vertical_crossing.square())
        outcomes["crossing_radial"][new_crossing] = radial[new_crossing]
    new_pass = pass_now & ~outcomes["passed"]
    new_collision = collision_now & ~outcomes["collision"]
    outcomes["pass_step"][new_pass] = step + 1
    signed, lateral, vertical = gate_coordinates(state.position, gate)
    progress = (
        (signed - outcomes["initial_signed"])
        / (-outcomes["initial_signed"] + 0.4).clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)
    approach = torch.exp(
        -0.5
        * (torch.sqrt(lateral.square() + vertical.square()) / outcomes["clean_radius"]).square()
    )
    old_potential = outcomes["previous_potential"]
    outcomes["maximum_progress"] = torch.maximum(outcomes["maximum_progress"], progress)
    before_crossing = outcomes["crossing_radial"].isnan() | new_crossing
    outcomes["maximum_approach"] = torch.where(
        before_crossing,
        torch.maximum(outcomes["maximum_approach"], approach),
        outcomes["maximum_approach"],
    )
    new_potential = outcomes["maximum_progress"] + outcomes["maximum_approach"]
    outcomes["previous_potential"] = new_potential
    outcomes["passed"] |= pass_now
    outcomes["collision"] |= collision_now
    outcomes["missed"] |= miss_now
    outcomes["cleared"] |= outcomes["passed"] & (signed >= 0.4)
    outcomes["lifted"] |= state.position[:, 2] > 0.15
    new_recontact = outcomes["lifted"] & ~outcomes["recontact"] & (state.position[:, 2] <= 0.01)
    outcomes["recontact"] |= new_recontact
    outcomes["maximum_tilt"] = torch.maximum(
        outcomes["maximum_tilt"], torch.linalg.vector_norm(state.euler[:, :2], dim=1)
    )
    saturated = (stick_position.abs() > 0.98).any(dim=1)
    outcomes["saturation_steps"] += saturated
    hazard_now = new_collision | new_recontact
    return new_potential - old_potential, new_pass, hazard_now


def final_success(
    outcomes: dict[str, Tensor], *, step_count: int, hover_config: HoverConfig
) -> tuple[Tensor, Tensor]:
    saturation_fraction = outcomes["saturation_steps"] / step_count
    has_post_pass_second = (outcomes["pass_step"] >= 0) & (
        outcomes["pass_step"] <= step_count - round(1.0 / hover_config.dt)
    )
    success = (
        outcomes["passed"]
        & outcomes["cleared"]
        & has_post_pass_second
        & ~outcomes["collision"]
        & ~outcomes["missed"]
        & ~outcomes["recontact"]
        & (outcomes["maximum_tilt"] <= math.radians(40.0))
        & ~(saturation_fraction > 0.25)
    )
    return success, saturation_fraction


@torch.no_grad()
def collect_rollout(
    controller: ConnectomeController,
    actor: MotorParameters,
    critic: PrivilegedCritic,
    spec: MotorSpec,
    cases: BalancedCases,
    *,
    seconds: float,
    sigma: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    seed: int,
) -> PPORollout:
    seed_everything(seed)
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    feature_count = critic_features(
        state,
        cases.gate,
        cases.mass_scale,
        stick_state.position,
        stick_state.velocity,
        outcomes["passed"],
        outcomes["maximum_progress"],
        outcomes["maximum_approach"],
        outcomes["saturation_steps"],
        0.0,
        hover_config,
    ).shape[1]
    images = torch.empty(steps, episodes, resolution, resolution, device=device)
    roll_pitch = torch.empty(steps, episodes, 2, device=device)
    specific_force = torch.empty(steps, episodes, 3, device=device)
    stick_position = torch.empty(steps, episodes, 4, device=device)
    features = torch.empty(steps, episodes, feature_count, device=device)
    latent_actions = torch.empty(steps, episodes, 4, device=device)
    old_latent_mean = torch.empty(steps, episodes, 4, device=device)
    old_log_prob = torch.empty(steps, episodes, device=device)
    rewards = torch.zeros(steps, episodes, device=device)
    values = torch.empty(steps, episodes, device=device)
    valid = torch.empty(steps, episodes, dtype=torch.bool, device=device)
    done = torch.zeros(steps, episodes, dtype=torch.bool, device=device)
    neural_before = torch.empty(steps, episodes, controller.n_nodes, device=device)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    for step in range(steps):
        valid[step] = active
        neural_before[step] = neural
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        images[step] = image
        roll_pitch[step] = state.euler[:, :2]
        specific_force[step] = state.specific_force
        stick_position[step] = stick_state.position
        features[step] = critic_features(
            state,
            cases.gate,
            cases.mass_scale,
            stick_state.position,
            stick_state.velocity,
            outcomes["passed"],
            outcomes["maximum_progress"],
            outcomes["maximum_approach"],
            outcomes["saturation_steps"] / max(step, 1),
            step / max(steps - 1, 1),
            hover_config,
        )
        values[step] = critic(privileged_critic_input(features[step], neural))
        motor_mean, neural = controller_step(
            controller,
            actor,
            spec,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        mean = latent_mean(motor_mean)
        latent = mean + sigma * torch.randn_like(mean)
        action = torch.tanh(latent)
        latent_actions[step] = latent
        old_latent_mean[step] = mean
        old_log_prob[step] = squashed_log_prob(mean, latent, action, sigma)
        rc, stick_state = sticks(action, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        potential_gain, new_pass, hazard_now = update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        step_reward = potential_gain + 2.0 * new_pass.float() - 2.0 * hazard_now.float()
        step_reward -= 0.1 / steps * (stick_state.position.abs() > 0.98).any(dim=1)
        rewards[step] = step_reward * active.float()
        irrecoverable = (
            outcomes["collision"]
            | outcomes["missed"]
            | outcomes["recontact"]
            | (outcomes["maximum_tilt"] > math.radians(40.0))
        )
        terminal_now = active & irrecoverable
        done[step] = terminal_now
        active &= ~irrecoverable
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    rewards[-1] += 10.0 * success.float()
    done[-1] = True
    episode_reward = rewards.sum(dim=0)
    summary = outcome_summary(
        success,
        outcomes["passed"],
        outcomes["collision"],
        outcomes["missed"],
        outcomes["crossing_radial"],
        outcomes["maximum_progress"],
        outcomes["maximum_approach"],
        saturation_fraction,
        cases.stratum_code,
        episode_reward,
    )
    return PPORollout(
        images=images,
        roll_pitch=roll_pitch,
        specific_force=specific_force,
        stick_position=stick_position,
        critic_features=features,
        latent_actions=latent_actions,
        old_latent_mean=old_latent_mean,
        old_log_prob=old_log_prob,
        rewards=rewards,
        values=values,
        valid=valid,
        done=done,
        neural_before=neural_before,
        mass_scale=cases.mass_scale,
        summary=summary,
        outcomes={
            "success": success.detach().cpu(),
            "codes": cases.stratum_code.detach().cpu(),
        },
    )


def discounted_returns(rewards: Tensor, done: Tensor, gamma: float) -> Tensor:
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in range(len(rewards) - 1, -1, -1):
        running = rewards[step] + gamma * running * (~done[step]).float()
        returns[step] = running
    return returns


def refresh_values(rollout: PPORollout, critic: PrivilegedCritic, batch: int = 65_536) -> None:
    physical = rollout.critic_features.reshape(-1, rollout.critic_features.shape[-1])
    neural = rollout.neural_before.reshape(-1, rollout.neural_before.shape[-1])
    output = torch.empty(len(physical), device=physical.device)
    with torch.no_grad():
        for begin in range(0, len(physical), batch):
            features = privileged_critic_input(
                physical[begin : begin + batch], neural[begin : begin + batch]
            )
            output[begin : begin + batch] = critic(features)
    rollout.values.copy_(output.reshape_as(rollout.values))


def compute_gae(rollout: PPORollout, gamma: float, gae_lambda: float) -> None:
    advantages = torch.zeros_like(rollout.rewards)
    gae = torch.zeros(rollout.rewards.shape[1], device=rollout.rewards.device)
    next_value = torch.zeros_like(gae)
    for step in range(len(rollout.rewards) - 1, -1, -1):
        nonterminal = (~rollout.done[step]).float()
        delta = rollout.rewards[step] + gamma * next_value * nonterminal - rollout.values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
        next_value = rollout.values[step]
    mask = rollout.valid
    mean = advantages[mask].mean()
    scale = advantages[mask].std(unbiased=False).clamp_min(1.0e-6)
    rollout.advantages = (advantages - mean) / scale
    rollout.returns = advantages + rollout.values


def warmup_critic(
    rollout: PPORollout,
    critic: PrivilegedCritic,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    gamma: float,
) -> list[float]:
    targets = discounted_returns(rollout.rewards, rollout.done, gamma)
    features = privileged_critic_input(
        rollout.critic_features[rollout.valid], rollout.neural_before[rollout.valid]
    )
    target = targets[rollout.valid]
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(target), device=target.device)
        for begin in range(0, len(order), 8192):
            selected = order[begin : begin + 8192]
            prediction = critic(features[selected])
            loss = 0.5 * (prediction - target[selected]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    refresh_values(rollout, critic)
    return losses


def replay_sequence(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    rollout: PPORollout,
    chunks: Tensor,
    environments: Tensor,
    *,
    chunk_steps: int,
    burn_in_steps: int,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    starts = chunks * chunk_steps
    has_burn_in = starts > 0
    incoming_time = (starts - burn_in_steps).clamp_min(0)
    neural = rollout.neural_before[incoming_time, environments].float()
    with torch.no_grad():
        for offset in range(-burn_in_steps, 0):
            time = (starts + offset).clamp_min(0)
            _, next_neural = controller_step(
                controller,
                actor,
                spec,
                rollout.images[time, environments].float(),
                rollout.roll_pitch[time, environments],
                neural,
                rollout.specific_force[time, environments],
                rollout.stick_position[time, environments],
            )
            neural = torch.where(has_burn_in[:, None], next_neural, neural)
    log_prob = []
    latent_means = []
    for offset in range(chunk_steps):
        time = starts + offset
        motor_mean, neural = controller_step(
            controller,
            actor,
            spec,
            rollout.images[time, environments].float(),
            rollout.roll_pitch[time, environments],
            neural,
            rollout.specific_force[time, environments],
            rollout.stick_position[time, environments],
        )
        mean = latent_mean(motor_mean)
        latent = rollout.latent_actions[time, environments]
        log_prob.append(squashed_log_prob(mean, latent, torch.tanh(latent), sigma))
        latent_means.append(mean)
    return torch.stack(log_prob), torch.stack(latent_means)


@torch.no_grad()
def replay_consistency_audit(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    rollout: PPORollout,
    *,
    chunk_steps: int,
    burn_in_steps: int,
    sigma: float,
    sequence_minibatch: int,
) -> dict[str, Any]:
    """Verify that recurrent replay reproduces rollout likelihoods before any update."""
    steps, episodes = rollout.valid.shape
    chunks_per_episode = steps // chunk_steps
    sequence_count = chunks_per_episode * episodes
    differences = []
    mean_differences = []
    for begin in range(0, sequence_count, sequence_minibatch):
        sequence = torch.arange(
            begin,
            min(begin + sequence_minibatch, sequence_count),
            device=rollout.rewards.device,
        )
        chunks = torch.div(sequence, episodes, rounding_mode="floor")
        environments = sequence.remainder(episodes)
        starts = chunks * chunk_steps
        offsets = torch.arange(chunk_steps, device=sequence.device)[:, None]
        times = starts[None, :] + offsets
        environment_grid = environments[None, :].expand_as(times)
        replayed, replayed_mean = replay_sequence(
            controller,
            actor,
            spec,
            rollout,
            chunks,
            environments,
            chunk_steps=chunk_steps,
            burn_in_steps=burn_in_steps,
            sigma=sigma,
        )
        selected = rollout.valid[times, environment_grid]
        differences.append((replayed - rollout.old_log_prob[times, environment_grid])[selected])
        mean_difference = replayed_mean - rollout.old_latent_mean[times, environment_grid]
        if not torch.isfinite(mean_difference[selected]).all():
            raise RuntimeError("non-finite latent mean during unchanged-policy replay")
        mean_differences.append(mean_difference[selected])
    log_ratio = torch.cat(differences)
    latent_mean_difference = torch.cat(mean_differences)
    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
    approximate_kl = (((ratio - 1.0) - log_ratio).mean()).clamp_min(0.0)
    exact_kl = 0.5 * (latent_mean_difference / sigma).square().sum(dim=-1).mean()
    maximum_absolute = log_ratio.abs().max()
    mean_absolute = log_ratio.abs().mean()
    maximum_latent_mean = latent_mean_difference.abs().max()
    passed = bool(
        maximum_absolute <= 5.0e-4 and maximum_latent_mean <= 1.0e-6 and exact_kl <= 1.0e-8
    )
    return {
        "valid_action_steps": len(log_ratio),
        "maximum_absolute_log_probability_difference": float(maximum_absolute),
        "mean_absolute_log_probability_difference": float(mean_absolute),
        "maximum_absolute_latent_mean_difference": float(maximum_latent_mean),
        "approximate_kl": float(approximate_kl),
        "exact_kl": float(exact_kl),
        "maximum_absolute_difference_threshold": 5.0e-4,
        "maximum_absolute_latent_mean_difference_threshold": 1.0e-6,
        "exact_kl_threshold": 1.0e-8,
        "passed": passed,
    }


@torch.no_grad()
def burn_in_approximation_audit(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    rollout: PPORollout,
    *,
    chunk_steps: int,
    burn_in_steps: int,
    sigma: float,
    sequence_minibatch: int,
) -> dict[str, Any]:
    """Compare truncated burn-in with a current-policy replay from episode reset."""
    steps, episodes = rollout.valid.shape
    neural = controller.initial_state(episodes, device=rollout.images.device, dtype=torch.float32)
    full_prefix_means = torch.empty_like(rollout.old_latent_mean)
    for step in range(steps):
        motor_mean, neural = controller_step(
            controller,
            actor,
            spec,
            rollout.images[step],
            rollout.roll_pitch[step],
            neural,
            rollout.specific_force[step],
            rollout.stick_position[step],
        )
        full_prefix_means[step] = latent_mean(motor_mean)
    chunks_per_episode = steps // chunk_steps
    sequence_count = chunks_per_episode * episodes
    differences = []
    for begin in range(0, sequence_count, sequence_minibatch):
        sequence = torch.arange(
            begin,
            min(begin + sequence_minibatch, sequence_count),
            device=rollout.rewards.device,
        )
        chunks = torch.div(sequence, episodes, rounding_mode="floor")
        environments = sequence.remainder(episodes)
        starts = chunks * chunk_steps
        offsets = torch.arange(chunk_steps, device=sequence.device)[:, None]
        times = starts[None, :] + offsets
        environment_grid = environments[None, :].expand_as(times)
        _, truncated_means = replay_sequence(
            controller,
            actor,
            spec,
            rollout,
            chunks,
            environments,
            chunk_steps=chunk_steps,
            burn_in_steps=burn_in_steps,
            sigma=sigma,
        )
        selected = rollout.valid[times, environment_grid]
        differences.append((truncated_means - full_prefix_means[times, environment_grid])[selected])
    difference = torch.cat(differences)
    per_step_kl = 0.5 * (difference / sigma).square().sum(dim=-1)
    maximum_absolute = difference.abs().max()
    mean_exact_kl = per_step_kl.mean()
    passed = bool(maximum_absolute <= 5.0e-3 and mean_exact_kl <= 1.0e-3)
    return {
        "valid_action_steps": len(difference),
        "burn_in_steps": burn_in_steps,
        "maximum_absolute_latent_mean_difference": float(maximum_absolute),
        "mean_exact_kl": float(mean_exact_kl),
        "maximum_absolute_difference_threshold": 5.0e-3,
        "mean_exact_kl_threshold": 1.0e-3,
        "passed": passed,
    }


def ppo_update(
    controller: ConnectomeController,
    actor: MotorParameters,
    critic: PrivilegedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    spec: MotorSpec,
    rollout: PPORollout,
    *,
    args: argparse.Namespace,
    iteration: int,
) -> dict[str, Any]:
    assert rollout.advantages is not None and rollout.returns is not None
    steps, episodes = rollout.valid.shape
    chunks_per_episode = steps // args.chunk_steps
    sequence_count = chunks_per_episode * episodes
    updates = []
    stopped_for_kl = False
    for epoch in range(args.optimization_epochs):
        order = torch.randperm(sequence_count, device=rollout.rewards.device)
        for begin in range(0, sequence_count, args.sequence_minibatch):
            sequence = order[begin : begin + args.sequence_minibatch]
            chunks = torch.div(sequence, episodes, rounding_mode="floor")
            environments = sequence.remainder(episodes)
            starts = chunks * args.chunk_steps
            offsets = torch.arange(args.chunk_steps, device=sequence.device)[:, None]
            times = starts[None, :] + offsets
            environment_grid = environments[None, :].expand_as(times)
            mask = rollout.valid[times, environment_grid]
            if not bool(mask.any()):
                continue
            new_log_prob, new_latent_mean = replay_sequence(
                controller,
                actor,
                spec,
                rollout,
                chunks,
                environments,
                chunk_steps=args.chunk_steps,
                burn_in_steps=args.burn_in_steps,
                sigma=args.exploration_sigma_selected,
            )
            old_log_prob = rollout.old_log_prob[times, environment_grid]
            old_latent_mean = rollout.old_latent_mean[times, environment_grid]
            selected_new_log_prob = new_log_prob[mask]
            selected_old_log_prob = old_log_prob[mask]
            selected_new_latent_mean = new_latent_mean[mask]
            selected_old_latent_mean = old_latent_mean[mask]
            if not all(
                bool(torch.isfinite(values).all())
                for values in (
                    selected_new_log_prob,
                    selected_old_log_prob,
                    selected_new_latent_mean,
                    selected_old_latent_mean,
                )
            ):
                raise RuntimeError("non-finite recurrent policy replay")
            pre_update_exact_kl = float(
                (
                    0.5
                    * (
                        (selected_new_latent_mean - selected_old_latent_mean)
                        / args.exploration_sigma_selected
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                ).detach()
            )
            if pre_update_exact_kl > args.target_kl:
                stopped_for_kl = True
                break
            advantage = rollout.advantages[times, environment_grid][mask]
            weights = torch.ones_like(rollout.advantages[times, environment_grid])
            if iteration <= args.light_weight_iterations:
                light = rollout.mass_scale[environments] < 1.0
                weights[:, light] = args.light_advantage_weight
            weights = weights[mask]
            ratio = torch.exp((selected_new_log_prob - selected_old_log_prob).clamp(-20.0, 20.0))
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantage
            policy_loss = -(torch.minimum(unclipped, clipped) * weights).sum() / weights.sum()
            critic_prediction = critic(
                privileged_critic_input(
                    rollout.critic_features[times, environment_grid][mask],
                    rollout.neural_before[times, environment_grid][mask],
                )
            )
            value_error = critic_prediction - rollout.returns[times, environment_grid][mask]
            value_loss = 0.5 * value_error.square().mean()
            if not bool(torch.isfinite(policy_loss)) or not bool(torch.isfinite(value_loss)):
                raise RuntimeError("non-finite PPO loss")
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            value_loss.backward()
            raw_actor_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in actor.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            actor_state_before = [parameter.detach().clone() for parameter in actor.parameters()]
            actor_optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
            nn.utils.clip_grad_norm_(
                actor.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
            )
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            actor_optimizer.step()
            actor.project()
            with torch.no_grad():
                _, post_update_latent_mean = replay_sequence(
                    controller,
                    actor,
                    spec,
                    rollout,
                    chunks,
                    environments,
                    chunk_steps=args.chunk_steps,
                    burn_in_steps=args.burn_in_steps,
                    sigma=args.exploration_sigma_selected,
                )
                post_update_exact_kl = float(
                    0.5
                    * (
                        (post_update_latent_mean[mask] - selected_old_latent_mean)
                        / args.exploration_sigma_selected
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                )
            accepted = math.isfinite(post_update_exact_kl) and (
                post_update_exact_kl <= args.target_kl
            )
            if not accepted:
                with torch.no_grad():
                    for parameter, previous in zip(
                        actor.parameters(), actor_state_before, strict=True
                    ):
                        parameter.copy_(previous)
                actor_optimizer.load_state_dict(actor_optimizer_before)
                stopped_for_kl = True
            critic_optimizer.step()
            clip_fraction = float(((ratio - 1.0).abs() > args.ppo_clip).float().mean())
            updates.append(
                {
                    "epoch": epoch + 1,
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "pre_update_exact_kl": pre_update_exact_kl,
                    "post_update_exact_kl": post_update_exact_kl,
                    "actor_update_accepted": accepted,
                    "clip_fraction": clip_fraction,
                    "raw_actor_gradient_norm": float(raw_actor_norm),
                }
            )
            if stopped_for_kl:
                break
        if stopped_for_kl:
            break
    return {
        "minibatch_updates": updates,
        "stopped_for_target_kl": stopped_for_kl,
        "maximum_pre_update_exact_kl": max(
            (item["pre_update_exact_kl"] for item in updates), default=0.0
        ),
        "maximum_post_update_exact_kl": max(
            (item["post_update_exact_kl"] for item in updates), default=0.0
        ),
        "mean_policy_loss": float(np.mean([item["policy_loss"] for item in updates])),
        "mean_value_loss": float(np.mean([item["value_loss"] for item in updates])),
        "maximum_raw_actor_gradient_norm": max(
            (item["raw_actor_gradient_norm"] for item in updates), default=0.0
        ),
    }


@torch.no_grad()
def evaluate_actor(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    cases: BalancedCases,
    *,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    sigma: float = 0.0,
    seed: int = 0,
    frozen_visual: bool = False,
    acceleration_control: str = "live",
    return_outcomes: bool = False,
) -> dict[str, Any]:
    if acceleration_control not in {"live", "constant_1g", "pair_swapped"}:
        raise ValueError(f"unknown acceleration control: {acceleration_control}")
    seed_everything(seed)
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    initial_image = render_annular_gate(
        state,
        cases.gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    pair_swap = torch.arange(episodes, device=device).bitwise_xor(1)
    episode_reward = torch.zeros(episodes, device=device)
    for step in range(steps):
        image = (
            initial_image
            if frozen_visual
            else render_annular_gate(
                state,
                cases.gate,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
        )
        sensed_force = state.specific_force
        if acceleration_control == "constant_1g":
            sensed_force = torch.zeros_like(sensed_force)
            sensed_force[:, 2] = 9.81
        elif acceleration_control == "pair_swapped":
            sensed_force = sensed_force[pair_swap]
        motor_mean, neural = controller_step(
            controller,
            actor,
            spec,
            image,
            state.euler[:, :2],
            neural,
            sensed_force,
            stick_state.position,
        )
        if sigma:
            motor = torch.tanh(latent_mean(motor_mean) + sigma * torch.randn_like(motor_mean))
        else:
            motor = motor_mean
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        potential_gain, new_pass, hazard_now = update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        episode_reward += potential_gain + 2.0 * new_pass.float() - 2.0 * hazard_now.float()
        episode_reward -= 0.1 / steps * (stick_state.position.abs() > 0.98).any(dim=1)
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    episode_reward += 10.0 * success.float()
    result: dict[str, Any] = {
        "summary": outcome_summary(
            success,
            outcomes["passed"],
            outcomes["collision"],
            outcomes["missed"],
            outcomes["crossing_radial"],
            outcomes["maximum_progress"],
            outcomes["maximum_approach"],
            saturation_fraction,
            cases.stratum_code,
            episode_reward,
        )
    }
    if return_outcomes:
        result["outcomes"] = {
            "success": success.detach().cpu(),
            "codes": cases.stratum_code.detach().cpu(),
        }
    return result


def evaluation_parity_audit(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    cases: BalancedCases,
    *,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    """Compare this evaluator with the previously validated batched evaluator."""
    interface_spec = motor_interface_spec(
        controller,
        bias_scale=0.1,
        log_gain_scale=0.1,
        maximum_bias_delta=0.5,
        maximum_gain_ratio=2.0,
    )
    zero = torch.zeros(1, len(interface_spec.labels), device=cases.mass_scale.device)
    expected = evaluate_policy_batch(
        controller,
        zero,
        interface_spec,
        cases,
        seconds=seconds,
        shaping_weight=1.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summaries"][0]
    actual = evaluate_actor(
        controller,
        actor,
        spec,
        cases,
        seconds=seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summary"]
    comparable = {
        "success_rate": (actual["success_rate"], expected["success_rate"]),
        "light_success_rate": (
            actual["light_success_rate"],
            expected["success_by_stratum"]["lower_mass"],
        ),
        "heavy_success_rate": (
            actual["heavy_success_rate"],
            expected["success_by_stratum"]["higher_mass"],
        ),
        "clean_pass_rate": (actual["clean_pass_rate"], expected["clean_pass_rate"]),
        "ring_collision_rate": (
            actual["ring_collision_rate"],
            expected["ring_collision_rate"],
        ),
        "miss_rate": (actual["miss_rate"], expected["miss_rate"]),
        "plane_crossing_rate": (
            actual["plane_crossing_rate"],
            expected["plane_crossing_rate"],
        ),
        "progress_mean": (actual["progress_mean"], expected["progress_mean"]),
        "approach_score_mean": (
            actual["approach_score_mean"],
            expected["approach_score_mean"],
        ),
        "stick_saturation_fraction": (
            actual["stick_saturation_fraction"],
            expected["stick_saturation_fraction"],
        ),
        "episode_reward_mean": (actual["episode_reward_mean"], expected["mean_reward"]),
    }
    differences = {name: abs(left - right) for name, (left, right) in comparable.items()}
    if actual["crossing_radial_mean_m"] is None and expected["crossing_radial_mean_m"] is None:
        radial_difference = 0.0
    elif actual["crossing_radial_mean_m"] is None or expected["crossing_radial_mean_m"] is None:
        radial_difference = float("inf")
    else:
        radial_difference = abs(
            actual["crossing_radial_mean_m"] - expected["crossing_radial_mean_m"]
        )
    differences["crossing_radial_mean_m"] = radial_difference
    discrete = {
        name: difference
        for name, difference in differences.items()
        if name
        in {
            "success_rate",
            "light_success_rate",
            "heavy_success_rate",
            "clean_pass_rate",
            "ring_collision_rate",
            "miss_rate",
            "plane_crossing_rate",
        }
    }
    continuous = {
        name: difference for name, difference in differences.items() if name not in discrete
    }
    return {
        "episodes": len(cases.mass_scale),
        "seconds": seconds,
        "differences": differences,
        "discrete_outcomes_exact": max(discrete.values()) == 0.0,
        "maximum_continuous_difference": max(continuous.values()),
        "passed": max(discrete.values()) == 0.0 and max(continuous.values()) <= 1.0e-3,
    }


def likelihood_window_loss(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    rollout: PPORollout,
    *,
    steps: int,
    sigma: float,
) -> Tensor:
    environments = torch.arange(rollout.images.shape[1], device=rollout.images.device)
    chunks = torch.zeros_like(environments)
    log_prob, _ = replay_sequence(
        controller,
        actor,
        spec,
        rollout,
        chunks,
        environments,
        chunk_steps=steps,
        burn_in_steps=0,
        sigma=sigma,
    )
    return -log_prob.mean()


def gradient_audit(
    controller: ConnectomeController,
    actor: MotorParameters,
    critic: PrivilegedCritic,
    spec: MotorSpec,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    cases = sample_matched_cases(
        args.gradient_check_episodes,
        seed=args.rollout_seed - 1,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    seconds = args.gradient_check_steps * hover_config.dt
    rollout = collect_rollout(
        controller,
        actor,
        critic,
        spec,
        cases,
        seconds=seconds,
        sigma=args.exploration_sigma,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=args.rollout_seed - 1,
    )
    repeated = []
    with torch.no_grad():
        for _ in range(3):
            repeated.append(
                float(
                    likelihood_window_loss(
                        controller,
                        actor,
                        spec,
                        rollout,
                        steps=args.gradient_check_steps,
                        sigma=args.exploration_sigma,
                    )
                )
            )
    loss = likelihood_window_loss(
        controller,
        actor,
        spec,
        rollout,
        steps=args.gradient_check_steps,
        sigma=args.exploration_sigma,
    )
    gradients = torch.autograd.grad(loss, tuple(actor.parameters()))
    generator = torch.Generator(device=device).manual_seed(args.optimization_seed - 1)
    directions = [
        torch.randn(parameter.shape, generator=generator, device=device)
        for parameter in actor.parameters()
    ]
    edge_allowed = (actor.edge_magnitudes.detach() > 0.01) & (actor.edge_magnitudes.detach() < 7.99)
    directions[0] *= edge_allowed
    maximum = max(float(direction.abs().max()) for direction in directions)
    directions = [direction / maximum for direction in directions]
    analytic = float(
        sum(
            (gradient * direction).sum()
            for gradient, direction in zip(gradients, directions, strict=True)
        )
    )
    baseline = [parameter.detach().clone() for parameter in actor.parameters()]
    finite = []
    for scale in args.gradient_check_scales:
        losses = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                for parameter, value, direction in zip(
                    actor.parameters(), baseline, directions, strict=True
                ):
                    parameter.copy_(value + sign * scale * direction)
            losses.append(
                float(
                    likelihood_window_loss(
                        controller,
                        actor,
                        spec,
                        rollout,
                        steps=args.gradient_check_steps,
                        sigma=args.exploration_sigma,
                    ).detach()
                )
            )
        derivative = (losses[0] - losses[1]) / (2.0 * scale)
        relative_error = abs(derivative - analytic) / max(abs(derivative), abs(analytic), 1.0e-12)
        finite.append(
            {
                "scale": scale,
                "plus_loss": losses[0],
                "minus_loss": losses[1],
                "directional_derivative": derivative,
                "relative_error": relative_error,
                "matching_sign": derivative * analytic > 0.0,
            }
        )
    with torch.no_grad():
        for parameter, value in zip(actor.parameters(), baseline, strict=True):
            parameter.copy_(value)
    repeat_noise = max(repeated) - min(repeated)
    adjacent_passes = []
    for left, right in zip(finite[:-1], finite[1:], strict=True):
        adjacent_passes.append(
            left["matching_sign"]
            and right["matching_sign"]
            and left["relative_error"] <= args.gradient_check_tolerance
            and right["relative_error"] <= args.gradient_check_tolerance
            and abs(left["plus_loss"] - left["minus_loss"]) > 10.0 * repeat_noise
            and abs(right["plus_loss"] - right["minus_loss"]) > 10.0 * repeat_noise
        )
    return {
        "window_steps": args.gradient_check_steps,
        "incoming_hidden_state_held_fixed": True,
        "unchanged_loss_repeats": repeated,
        "unchanged_loss_range": repeat_noise,
        "analytic_directional_derivative": analytic,
        "finite_differences": finite,
        "passed": any(adjacent_passes),
    }


def parity_audit(
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    *,
    resolution: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(800_003)
    batch = 4
    image = torch.rand(batch, resolution, resolution, generator=generator, device=device)
    attitude = 0.1 * torch.randn(batch, 2, generator=generator, device=device)
    neural = 0.2 * torch.randn(batch, controller.n_nodes, generator=generator, device=device)
    force = torch.randn(batch, 3, generator=generator, device=device)
    force[:, 2] += 9.81
    sticks = torch.rand(batch, 4, generator=generator, device=device) * 2.0 - 1.0
    with torch.no_grad():
        expected_motor, expected_state = controller(image, attitude, neural, force, sticks)
        actual_motor, actual_state = controller_step(
            controller, actor, spec, image, attitude, neural, force, sticks
        )
        round_trip = torch.tanh(latent_mean(expected_motor))
    differences = {
        "motor": float((actual_motor - expected_motor).abs().max()),
        "neural_state": float((actual_state - expected_state).abs().max()),
        "zero_noise_squash": float((round_trip - expected_motor).abs().max()),
    }
    return {"differences": differences, "passed": max(differences.values()) <= 1.0e-6}


def compile_actor(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    actor: MotorParameters,
    spec: MotorSpec,
) -> None:
    controller.load_state_dict(base_state)
    with torch.no_grad():
        controller.edge_magnitude[spec.edges] = actor.edge_magnitudes
        controller.bias[spec.nodes] = actor.biases
    controller.eval()


def save_checkpoint(
    path: Path,
    source: dict[str, Any],
    source_sha256: str,
    controller: ConnectomeController,
    actor: MotorParameters,
    spec: MotorSpec,
    promotion: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source)
    checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    checkpoint["source_checkpoint_sha256"] = source_sha256
    checkpoint["recurrent_ppo"] = {
        "method": "recurrent PPO with training-only privileged critic",
        "motor_input_edge_indices": spec.edges.detach().cpu().tolist(),
        "motor_neuron_indices": spec.nodes.detach().cpu().tolist(),
        "compiled_into_native_parameters": True,
        "parameter_vector_sha256": parameter_sha256(actor),
        "promotion": promotion,
    }
    torch.save(checkpoint, path)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    validate_args(args, hover_config.dt)
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    spec = motor_spec(controller)
    if len(spec.edges) != 198 or len(spec.nodes) != 26:
        raise SystemExit(
            "expected 198 motor-input edges and 26 motor neurons, "
            f"found {len(spec.edges)} and {len(spec.nodes)}"
        )
    actor = MotorParameters(spec).to(device)
    dummy_cases = sample_matched_cases(
        8,
        seed=args.rollout_seed - 2,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    dummy_stick = (
        ForelegStickPlant(hover_config)
        .to(device)
        .initial_state(8, device=device, dtype=torch.float32)
    )
    dummy_outcomes = initialize_outcomes(dummy_cases, gate_config)
    feature_count = (
        critic_features(
            dummy_cases.state,
            dummy_cases.gate,
            dummy_cases.mass_scale,
            dummy_stick.position,
            dummy_stick.velocity,
            dummy_outcomes["passed"],
            dummy_outcomes["maximum_progress"],
            dummy_outcomes["maximum_approach"],
            dummy_outcomes["saturation_steps"],
            0.0,
            hover_config,
        ).shape[1]
        + controller.n_nodes
    )
    critic = PrivilegedCritic(feature_count).to(device)
    parity = parity_audit(controller, actor, spec, resolution=resolution, device=device)
    if not parity["passed"]:
        raise SystemExit(f"native actor parity failed: {parity}")
    evaluation_parity = evaluation_parity_audit(
        controller,
        actor,
        spec,
        dummy_cases,
        seconds=min(args.rollout_seconds, args.validation_seconds),
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if not evaluation_parity["passed"]:
        raise SystemExit(f"batched evaluator parity failed: {evaluation_parity}")
    gradient_check = gradient_audit(
        controller,
        actor,
        critic,
        spec,
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if not gradient_check["passed"]:
        raise SystemExit(f"short likelihood gradient audit failed: {gradient_check}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    validation_cases = sample_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    baseline_validation = evaluate_actor(
        controller,
        actor,
        spec,
        validation_cases,
        seconds=args.validation_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summary"]
    sigma_candidates = []
    selected_sigma = args.exploration_sigma
    exploration_scales = [args.exploration_sigma]
    while exploration_scales[-1] > args.minimum_exploration_sigma:
        exploration_scales.append(max(args.minimum_exploration_sigma, exploration_scales[-1] / 2.0))
    for sigma in exploration_scales:
        exploratory = evaluate_actor(
            controller,
            actor,
            spec,
            validation_cases,
            seconds=args.validation_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            sigma=float(sigma),
            seed=args.rollout_seed - 3,
        )["summary"]
        acceptable_checks = {
            name: exploratory[name]
            >= baseline_validation[name] - args.maximum_exploration_success_drop
            for name in ("success_rate", "light_success_rate", "heavy_success_rate")
        }
        acceptable = all(acceptable_checks.values())
        sigma_candidates.append(
            {
                "sigma": float(sigma),
                "metrics": exploratory,
                "acceptable_checks": acceptable_checks,
                "acceptable": acceptable,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "exploration_calibration",
                    "sigma": float(sigma),
                    "deterministic_success": baseline_validation["success_rate"],
                    "exploratory_success": exploratory["success_rate"],
                    "exploratory_light_success": exploratory["light_success_rate"],
                    "exploratory_heavy_success": exploratory["heavy_success_rate"],
                    "acceptable": acceptable,
                }
            ),
            flush=True,
        )
        if acceptable:
            selected_sigma = float(sigma)
            break
    else:
        raise SystemExit("even minimum PPO exploration collapses deterministic performance")
    args.exploration_sigma_selected = selected_sigma
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    seed_everything(args.optimization_seed)
    archive: dict[str, dict[str, Any]] = {}
    iterations = []
    validations = []
    early_stop = False
    replay_audit = None
    for iteration in range(1, args.iterations + 1):
        cases = sample_matched_cases(
            args.episodes,
            seed=args.rollout_seed + iteration - 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        rollout = collect_rollout(
            controller,
            actor,
            critic,
            spec,
            cases,
            seconds=args.rollout_seconds,
            sigma=selected_sigma,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.rollout_seed + iteration - 1,
        )
        if iteration == 1:
            replay_audit = replay_consistency_audit(
                controller,
                actor,
                spec,
                rollout,
                chunk_steps=args.chunk_steps,
                burn_in_steps=args.burn_in_steps,
                sigma=selected_sigma,
                sequence_minibatch=args.sequence_minibatch,
            )
            if not replay_audit["passed"]:
                raise RuntimeError(f"unchanged-policy recurrent replay failed: {replay_audit}")
        warmup_losses = []
        if iteration == 1:
            warmup_losses = warmup_critic(
                rollout,
                critic,
                critic_optimizer,
                epochs=args.critic_warmup_epochs,
                gamma=args.gamma,
            )
        compute_gae(rollout, args.gamma, args.gae_lambda)
        update = ppo_update(
            controller,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            spec,
            rollout,
            args=args,
            iteration=iteration,
        )
        burn_in_audit = burn_in_approximation_audit(
            controller,
            actor,
            spec,
            rollout,
            chunk_steps=args.chunk_steps,
            burn_in_steps=args.burn_in_steps,
            sigma=selected_sigma,
            sequence_minibatch=args.sequence_minibatch,
        )
        if not burn_in_audit["passed"]:
            raise RuntimeError(f"truncated recurrent burn-in failed: {burn_in_audit}")
        iteration_entry = {
            "iteration": iteration,
            "rollout_seed": args.rollout_seed + iteration - 1,
            "exploratory_rollout": rollout.summary,
            "critic_warmup_loss_first": warmup_losses[0] if warmup_losses else None,
            "critic_warmup_loss_last": warmup_losses[-1] if warmup_losses else None,
            "ppo": update,
            "burn_in_approximation_audit": burn_in_audit,
            "parameter_vector_sha256": parameter_sha256(actor),
        }
        iterations.append(iteration_entry)
        print(
            json.dumps(
                {
                    "phase": "ppo",
                    "iteration": iteration,
                    "rollout_success": rollout.summary["success_rate"],
                    "light_success": rollout.summary["light_success_rate"],
                    "heavy_success": rollout.summary["heavy_success_rate"],
                    "maximum_kl": update["maximum_post_update_exact_kl"],
                    "maximum_raw_gradient_norm": update["maximum_raw_actor_gradient_norm"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "iteration": iteration,
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu(),
                "biases": actor.biases.detach().cpu(),
                "parameter_vector_sha256": parameter_sha256(actor),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic": critic.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            },
            args.output_dir / "training-state.pt",
        )
        if iteration % args.validation_interval == 0:
            validation = evaluate_actor(
                controller,
                actor,
                spec,
                validation_cases,
                seconds=args.validation_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )["summary"]
            key = parameter_sha256(actor)
            archive[key] = {
                "iteration": iteration,
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu().clone(),
                "biases": actor.biases.detach().cpu().clone(),
                "metrics": validation,
                "light_success_delta": (
                    validation["light_success_rate"] - baseline_validation["light_success_rate"]
                ),
                "heavy_success_delta": (
                    validation["heavy_success_rate"] - baseline_validation["heavy_success_rate"]
                ),
            }
            validations.append(
                {
                    "parameter_vector_sha256": key,
                    **{
                        name: value
                        for name, value in archive[key].items()
                        if name not in {"edge_magnitudes", "biases"}
                    },
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "validation",
                        "iteration": iteration,
                        "success": validation["success_rate"],
                        "light_delta": archive[key]["light_success_delta"],
                        "heavy_delta": archive[key]["heavy_success_delta"],
                    }
                ),
                flush=True,
            )
            torch.save(
                {
                    archive_key: {
                        name: value
                        for name, value in item.items()
                        if name in {"iteration", "edge_magnitudes", "biases"}
                    }
                    for archive_key, item in archive.items()
                },
                args.output_dir / "archive.pt",
            )
            if iteration == args.checkpoint_iteration and not any(
                item["light_success_delta"] >= args.checkpoint_light_improvement
                and item["heavy_success_delta"] >= -args.checkpoint_heavy_margin
                for item in archive.values()
            ):
                early_stop = True
                print(json.dumps({"phase": "early_stop", "iteration": iteration}), flush=True)
                break
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "iterations": iterations,
                    "validations": validations,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    eligible = [
        key
        for key, item in archive.items()
        if item["heavy_success_delta"] >= -args.final_heavy_margin
    ]
    selection_pool = eligible or list(archive)
    winner_key = max(
        selection_pool,
        key=lambda key: (
            archive[key]["metrics"]["light_success_rate"],
            archive[key]["metrics"]["success_rate"],
            -(
                archive[key]["metrics"]["crossing_radial_mean_m"]
                if archive[key]["metrics"]["crossing_radial_mean_m"] is not None
                else float("inf")
            ),
        ),
    )
    winner = archive[winner_key]
    with torch.no_grad():
        actor.edge_magnitudes.copy_(winner["edge_magnitudes"].to(device))
        actor.biases.copy_(winner["biases"].to(device))
    candidate_vector_path = args.output_dir / "candidate-vector.json"
    candidate_vector_path.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": file_sha256(args.checkpoint),
                "selected_iteration": winner["iteration"],
                "parameter_vector_sha256": parameter_sha256(actor),
                "motor_input_edge_indices": spec.edges.detach().cpu().tolist(),
                "motor_neuron_indices": spec.nodes.detach().cpu().tolist(),
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu().tolist(),
                "biases": actor.biases.detach().cpu().tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    final_cases = sample_matched_cases(
        args.final_episodes,
        seed=args.final_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    baseline_actor = MotorParameters(spec).to(device)
    final = {}
    for name, evaluated_actor, controls in (
        ("reference", baseline_actor, {}),
        ("candidate", actor, {}),
        ("candidate_frozen_first_frame", actor, {"frozen_visual": True}),
        ("candidate_constant_1g", actor, {"acceleration_control": "constant_1g"}),
        ("candidate_pair_swapped_acceleration", actor, {"acceleration_control": "pair_swapped"}),
    ):
        final[name] = evaluate_actor(
            controller,
            evaluated_actor,
            spec,
            final_cases,
            seconds=args.final_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            return_outcomes=True,
            **controls,
        )
        print(
            json.dumps({"phase": "final", "name": name, **final[name]["summary"]}),
            flush=True,
        )
    reference_success = final["reference"]["outcomes"]["success"]
    candidate_success = final["candidate"]["outcomes"]["success"]
    codes = final["reference"]["outcomes"]["codes"]
    light = ~codes.bitwise_and(1).bool()
    heavy = ~light
    paired_overall = paired_clustered_confidence_interval(reference_success, candidate_success)
    paired_light = paired_confidence_interval(reference_success[light], candidate_success[light])
    paired_heavy = paired_confidence_interval(reference_success[heavy], candidate_success[heavy])
    acceleration_controls = {
        "live_minus_constant_1g": paired_clustered_confidence_interval(
            final["candidate_constant_1g"]["outcomes"]["success"], candidate_success
        ),
        "live_minus_pair_swapped": paired_clustered_confidence_interval(
            final["candidate_pair_swapped_acceleration"]["outcomes"]["success"],
            candidate_success,
        ),
    }
    acceleration_dependence = all(
        control["confidence_95"][0] > 0.0 for control in acceleration_controls.values()
    )
    promotion_checks = {
        "light_improvement_at_least_threshold": (
            final["candidate"]["summary"]["light_success_rate"]
            >= final["reference"]["summary"]["light_success_rate"] + args.final_light_improvement
        ),
        "light_paired_confidence_interval_excludes_zero": paired_light["confidence_95"][0] > 0.0,
        "heavy_nondegradation_within_margin": (
            final["candidate"]["summary"]["heavy_success_rate"]
            >= final["reference"]["summary"]["heavy_success_rate"] - args.final_heavy_margin
        ),
        "heavy_paired_noninferiority_interval_within_margin": (
            paired_heavy["confidence_95"][0] >= -args.final_heavy_margin
        ),
        "overall_success_improves": (
            final["candidate"]["summary"]["success_rate"]
            > final["reference"]["summary"]["success_rate"]
        ),
        "overall_paired_confidence_interval_excludes_zero": (
            paired_overall["confidence_95"][0] > 0.0
        ),
        "frozen_first_frame_success_at_most_five_percent": (
            final["candidate_frozen_first_frame"]["summary"]["success_rate"] <= 0.05
        ),
    }
    promotion = {"checks": promotion_checks, "passed": all(promotion_checks.values())}
    candidate_path = args.output_dir / "candidate.pt"
    if promotion["passed"]:
        compile_actor(controller, base_state, actor, spec)
        save_checkpoint(
            candidate_path,
            checkpoint,
            file_sha256(args.checkpoint),
            controller,
            actor,
            spec,
            promotion,
        )
    goal_passed = bool(
        promotion["passed"]
        and final["candidate"]["summary"]["success_rate"] >= 0.90
        and final["candidate_frozen_first_frame"]["summary"]["success_rate"] <= 0.05
    )
    report = {
        "method": "recurrent PPO on independent native motor-input parameters",
        "claim_scope": (
            "The deployed candidate contains only compiled existing graph parameters and its "
            "persistent connectome state. Exploration, advantages, mass labels, physical state, "
            "gate geometry, and the critic are training-only."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{
                key: value
                for key, value in vars(args).items()
                if not isinstance(value, Path) and key != "exploration_sigma_selected"
            },
            "selected_exploration_sigma": selected_sigma,
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "non_motor_input_edges_frozen": True,
            "non_motor_neuron_biases_frozen": True,
            "time_constants_frozen": True,
            "physics_renderer_outside_autograd": True,
            "critic_training_only": True,
            "critic_inputs": (
                "physical state, gate pose, mass, foreleg state, reward bookkeeping, and the "
                "detached native connectome state"
            ),
            "deployed_actor_inputs": (
                "current FPV, roll/pitch, body-Z specific force, foreleg throttle position, "
                "and persistent native connectome state"
            ),
            "development_horizon_seconds": args.rollout_seconds,
            "final_stress_test_horizon_seconds": args.final_seconds,
            "matched_light_heavy_geometry": True,
        },
        "parameterization": {
            "independent_motor_input_edge_magnitudes": len(spec.edges),
            "independent_motor_neuron_biases": len(spec.nodes),
            "count": len(spec.edges) + len(spec.nodes),
            "edge_indices": spec.edges.detach().cpu().tolist(),
            "node_indices": spec.nodes.detach().cpu().tolist(),
        },
        "native_forward_parity": parity,
        "batched_evaluator_parity": evaluation_parity,
        "likelihood_gradient_audit": gradient_check,
        "unchanged_policy_replay_audit": replay_audit,
        "exploration_calibration": sigma_candidates,
        "baseline_validation": baseline_validation,
        "iterations_completed": len(iterations),
        "early_stopped_at_checkpoint": early_stop,
        "iterations": iterations,
        "validations": validations,
        "selected_candidate": winner_key,
        "selected_candidate_validation": {
            name: value
            for name, value in winner.items()
            if name not in {"edge_magnitudes", "biases"}
        },
        "selected_parameter_vector_sha256": parameter_sha256(actor),
        "selected_candidate_vector_file": stable_path(candidate_vector_path),
        "selected_candidate_vector_file_sha256": file_sha256(candidate_vector_path),
        "paired_final": {
            "overall_success_difference": paired_overall,
            "light_success_difference": paired_light,
            "heavy_success_difference": paired_heavy,
        },
        "acceleration_controls": acceleration_controls,
        "acceleration_dependence_demonstrated": acceleration_dependence,
        "final": {name: value["summary"] for name, value in final.items()},
        "promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": file_sha256(candidate_path) if promotion["passed"] else None,
        "goal_passed": goal_passed,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "iterations_completed": len(iterations),
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
