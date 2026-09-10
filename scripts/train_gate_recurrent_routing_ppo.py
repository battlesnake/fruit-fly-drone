#!/usr/bin/env python3
"""Train fixed-sign recurrent routing edges on assisted complete-flight outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_analytic_teachers import teacher_rc_for_mode  # noqa: E402
from audit_gate_throttle_routing import make_readout_spec  # noqa: E402
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    make_path_spec,
    paired_clustered_confidence_interval,
)
from search_gate_assisted_motor_es import (  # noqa: E402
    evaluate_assisted_policy_batch,
)
from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    load_controller,
    motor_interface_spec,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_recurrent_ppo import (  # noqa: E402
    PPORollout,
    PrivilegedCritic,
    compute_gae,
    critic_features,
    final_success,
    initialize_outcomes,
    latent_mean,
    outcome_summary,
    privileged_critic_input,
    squashed_log_prob,
    update_outcomes,
    warmup_critic,
)
from train_gate_recurrent_routing import make_recurrent_routing_spec  # noqa: E402

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    gate_coordinates,
    render_annular_gate,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)

MASS_LATERAL_KEYS = (
    "light_success_rate",
    "heavy_success_rate",
    "negative_lateral_success_rate",
    "positive_lateral_success_rate",
)


class RoutingActor(nn.Module):
    """The 125 existing fixed-sign edge magnitudes, independent during training."""

    def __init__(self, baseline: Tensor) -> None:
        super().__init__()
        self.edge_magnitudes = nn.Parameter(baseline.detach().clone())

    @torch.no_grad()
    def project(self) -> None:
        self.edge_magnitudes.clamp_(0.0, 8.0)

    def vector(self) -> Tensor:
        return self.edge_magnitudes


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
        "--flight-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-recurrent-routing-flight-diagnostic-v1" / "report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "recurrent-routing-ppo-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--rollout-seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--chunk-steps", type=int, default=40)
    parser.add_argument("--burn-in-steps", type=int, default=10)
    parser.add_argument("--optimization-epochs", type=int, default=2)
    parser.add_argument("--sequence-minibatch", type=int, default=64)
    parser.add_argument("--actor-learning-rate", type=float, default=1.0e-4)
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
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--checkpoint-iteration", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--checkpoint-light-improvement", type=float, default=0.05)
    parser.add_argument("--checkpoint-floor-improvement", type=float, default=0.05)
    parser.add_argument("--development-light-improvement", type=float, default=0.10)
    parser.add_argument("--development-floor-improvement", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--gradient-check-episodes", type=int, default=8)
    parser.add_argument("--gradient-check-steps", type=int, default=25)
    parser.add_argument(
        "--gradient-check-scales", type=float, nargs=3, default=(1.0e-3, 3.0e-4, 1.0e-4)
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    parser.add_argument("--rollout-seed", type=int, default=1_049_031)
    parser.add_argument("--validation-seed", type=int, default=1_050_031)
    parser.add_argument("--final-seed", type=int, default=1_051_031)
    parser.add_argument("--optimization-seed", type=int, default=1_052_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.checkpoint, args.flight_audit_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "iterations": (args.iterations, 20),
        "episodes": (args.episodes, 64),
        "rollout_seconds": (args.rollout_seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "chunk_steps": (args.chunk_steps, 40),
        "burn_in_steps": (args.burn_in_steps, 10),
        "optimization_epochs": (args.optimization_epochs, 2),
        "sequence_minibatch": (args.sequence_minibatch, 64),
        "actor_learning_rate": (args.actor_learning_rate, 1.0e-4),
        "critic_learning_rate": (args.critic_learning_rate, 3.0e-4),
        "critic_warmup_epochs": (args.critic_warmup_epochs, 2),
        "ppo_clip": (args.ppo_clip, 0.10),
        "target_kl": (args.target_kl, 0.01),
        "gradient_norm_cap": (args.gradient_norm_cap, 0.5),
        "gamma": (args.gamma, 0.999),
        "gae_lambda": (args.gae_lambda, 0.995),
        "exploration_sigma": (args.exploration_sigma, 0.03),
        "minimum_exploration_sigma": (args.minimum_exploration_sigma, 0.001),
        "maximum_exploration_success_drop": (
            args.maximum_exploration_success_drop,
            0.10,
        ),
        "validation_interval": (args.validation_interval, 5),
        "checkpoint_iteration": (args.checkpoint_iteration, 10),
        "validation_episodes": (args.validation_episodes, 256),
        "checkpoint_light_improvement": (args.checkpoint_light_improvement, 0.05),
        "checkpoint_floor_improvement": (args.checkpoint_floor_improvement, 0.05),
        "development_light_improvement": (args.development_light_improvement, 0.10),
        "development_floor_improvement": (args.development_floor_improvement, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
        "final_episodes": (args.final_episodes, 1024),
        "gradient_check_episodes": (args.gradient_check_episodes, 8),
        "gradient_check_steps": (args.gradient_check_steps, 25),
        "gradient_check_scales": (
            tuple(args.gradient_check_scales),
            (1.0e-3, 3.0e-4, 1.0e-4),
        ),
        "gradient_check_tolerance": (args.gradient_check_tolerance, 0.10),
        "rollout_seed": (args.rollout_seed, 1_049_031),
        "validation_seed": (args.validation_seed, 1_050_031),
        "final_seed": (args.final_seed, 1_051_031),
        "optimization_seed": (args.optimization_seed, 1_052_031),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered recurrent-routing PPO values changed: {', '.join(wrong)}")
    steps = round(args.rollout_seconds / dt)
    if steps % args.chunk_steps:
        raise SystemExit("rollout steps must be divisible by chunk steps")
    if args.burn_in_steps < 0 or args.burn_in_steps >= args.chunk_steps:
        raise SystemExit("burn-in must be nonnegative and shorter than a chunk")
    for name in ("episodes", "validation_episodes", "final_episodes", "gradient_check_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")


def parameter_sha256(actor: RoutingActor) -> str:
    values = actor.vector().detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def controller_step(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    image: Tensor,
    roll_pitch: Tensor,
    neural: Tensor,
    specific_force: Tensor,
    stick_position: Tensor,
) -> tuple[Tensor, Tensor]:
    activity = torch.tanh(neural)
    messages = (
        activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude.detach()
    )
    messages[:, selected_edges] = (
        activity[:, controller.edge_pre[selected_edges]]
        * controller.edge_sign[selected_edges]
        * actor.edge_magnitudes
    )
    recurrent = torch.zeros_like(neural).index_add(1, controller.edge_post, messages)
    drive = (
        recurrent
        + controller.bias
        + controller.sensory_drive(image, roll_pitch, specific_force, stick_position)
    )
    target = 5.0 * torch.tanh(drive / 5.0)
    alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant)
    next_neural = neural + alpha * (target - neural)
    return controller.motor_drive(next_neural), next_neural


def mass_lateral_floor(summary: dict[str, Any]) -> float:
    return min(float(summary[key]) for key in MASS_LATERAL_KEYS)


def mass_lateral_safe(
    source: dict[str, Any], candidate: dict[str, Any], *, maximum_drop: float
) -> bool:
    return all(
        float(candidate[key]) >= float(source[key]) - maximum_drop for key in MASS_LATERAL_KEYS
    )


def progress_gate(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    light_improvement: float,
    floor_improvement: float,
    maximum_drop: float,
) -> bool:
    return bool(
        candidate["light_success_rate"] >= source["light_success_rate"] + light_improvement
        and mass_lateral_floor(candidate) >= mass_lateral_floor(source) + floor_improvement
        and mass_lateral_safe(source, candidate, maximum_drop=maximum_drop)
    )


def initial_height_alignment(cases: BalancedCases, gate_config: GateConfig) -> Tensor:
    _, _, vertical = gate_coordinates(cases.state.position, cases.gate)
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    return torch.exp(-0.5 * (vertical / clean_radius).square())


def outcome_potential(maximum_progress: Tensor, current_height_alignment: Tensor) -> Tensor:
    """A bounded potential with continuing feedback from current vertical error."""

    return 0.5 * maximum_progress + 0.5 * current_height_alignment


def assisted_steering_motor(
    controller: ConnectomeController,
    source_motor: Tensor,
    state: Any,
    gate: AnnularGate,
    mass_scale: Tensor,
    *,
    use_reserve: bool,
    hover_config: HoverConfig,
) -> Tensor:
    if not use_reserve:
        return source_motor[:, :3]
    reserve_rc = teacher_rc_for_mode(
        "visual_accelerometer_reserve",
        controller,
        state,
        gate,
        mass_scale,
        hover_config,
    )
    return motor_target_for_rc(reserve_rc, hover_config)[:, :3]


@torch.no_grad()
def collect_rollout(
    controller: ConnectomeController,
    actor: RoutingActor,
    critic: PrivilegedCritic,
    selected_edges: Tensor,
    cases: BalancedCases,
    *,
    seconds: float,
    takeover_seconds: float,
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
    takeover_step = round(takeover_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    source_neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    height_alignment = initial_height_alignment(cases, gate_config)
    reported_height_alignment = height_alignment.clone()
    previous_potential = outcome_potential(outcomes["maximum_progress"], height_alignment)
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
    latent_actions = torch.empty(steps, episodes, 1, device=device)
    old_latent_mean = torch.empty_like(latent_actions)
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
        native_motor, neural = controller_step(
            controller,
            actor,
            selected_edges,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        if step < takeover_step:
            source_motor, source_neural = controller(
                image,
                state.euler[:, :2],
                source_neural,
                state.specific_force,
                stick_state.position,
            )
        else:
            source_motor = native_motor
        mean = latent_mean(native_motor[:, 3:4])
        latent = mean + sigma * torch.randn_like(mean)
        throttle_motor = torch.tanh(latent)
        steering_motor = assisted_steering_motor(
            controller,
            source_motor,
            state,
            cases.gate,
            cases.mass_scale,
            use_reserve=step >= takeover_step,
            hover_config=hover_config,
        )
        motor = torch.cat((steering_motor, throttle_motor), dim=1)
        latent_actions[step] = latent
        old_latent_mean[step] = mean
        old_log_prob[step] = squashed_log_prob(mean, latent, throttle_motor, sigma)
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        _, _, vertical = gate_coordinates(state.position, cases.gate)
        height_alignment = torch.exp(-0.5 * (vertical / outcomes["clean_radius"]).square())
        reported_height_alignment = torch.where(active, height_alignment, reported_height_alignment)
        potential = outcome_potential(outcomes["maximum_progress"], height_alignment)
        shaping = (potential - previous_potential) * active.float()
        previous_potential = potential
        irrecoverable = (
            outcomes["collision"]
            | outcomes["missed"]
            | outcomes["recontact"]
            | (outcomes["maximum_tilt"] > math.radians(40.0))
        )
        terminal_now = active & irrecoverable
        rewards[step] = shaping - 2.0 * terminal_now.float()
        done[step] = terminal_now
        active &= ~irrecoverable
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    rewards[-1] += 10.0 * success.float() - 2.0 * (active & ~success).float()
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
    summary["height_alignment_mean"] = float(reported_height_alignment.mean())
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


@torch.no_grad()
def evaluate_actor(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    cases: BalancedCases,
    *,
    seconds: float,
    takeover_seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    sigma: float = 0.0,
    seed: int = 0,
    return_outcomes: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    source_neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    height_alignment = initial_height_alignment(cases, gate_config)
    reported_height_alignment = height_alignment.clone()
    previous_potential = outcome_potential(outcomes["maximum_progress"], height_alignment)
    episode_reward = torch.zeros(episodes, device=device)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        native_motor, neural = controller_step(
            controller,
            actor,
            selected_edges,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        if step < takeover_step:
            source_motor, source_neural = controller(
                image,
                state.euler[:, :2],
                source_neural,
                state.specific_force,
                stick_state.position,
            )
        else:
            source_motor = native_motor
        throttle_motor = native_motor[:, 3:4]
        if sigma:
            throttle_motor = torch.tanh(
                latent_mean(throttle_motor) + sigma * torch.randn_like(throttle_motor)
            )
        steering_motor = assisted_steering_motor(
            controller,
            source_motor,
            state,
            cases.gate,
            cases.mass_scale,
            use_reserve=step >= takeover_step,
            hover_config=hover_config,
        )
        rc, stick_state = sticks(torch.cat((steering_motor, throttle_motor), dim=1), stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        _, _, vertical = gate_coordinates(state.position, cases.gate)
        height_alignment = torch.exp(-0.5 * (vertical / outcomes["clean_radius"]).square())
        reported_height_alignment = torch.where(active, height_alignment, reported_height_alignment)
        potential = outcome_potential(outcomes["maximum_progress"], height_alignment)
        episode_reward += (potential - previous_potential) * active.float()
        previous_potential = potential
        irrecoverable = (
            outcomes["collision"]
            | outcomes["missed"]
            | outcomes["recontact"]
            | (outcomes["maximum_tilt"] > math.radians(40.0))
        )
        terminal_now = active & irrecoverable
        episode_reward -= 2.0 * terminal_now.float()
        active &= ~irrecoverable
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    episode_reward += 10.0 * success.float() - 2.0 * (active & ~success).float()
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
    summary["height_alignment_mean"] = float(reported_height_alignment.mean())
    result: dict[str, Any] = {"summary": summary}
    if return_outcomes:
        result["outcomes"] = {
            "success": success.detach().cpu(),
            "codes": cases.stratum_code.detach().cpu(),
        }
    return result


def replay_sequence(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
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
                selected_edges,
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
            selected_edges,
            rollout.images[time, environments].float(),
            rollout.roll_pitch[time, environments],
            neural,
            rollout.specific_force[time, environments],
            rollout.stick_position[time, environments],
        )
        mean = latent_mean(motor_mean[:, 3:4])
        latent = rollout.latent_actions[time, environments]
        log_prob.append(squashed_log_prob(mean, latent, torch.tanh(latent), sigma))
        latent_means.append(mean)
    return torch.stack(log_prob), torch.stack(latent_means)


@torch.no_grad()
def replay_consistency_audit(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    rollout: PPORollout,
    *,
    chunk_steps: int,
    burn_in_steps: int,
    sigma: float,
    sequence_minibatch: int,
) -> dict[str, Any]:
    steps, episodes = rollout.valid.shape
    sequence_count = (steps // chunk_steps) * episodes
    log_differences = []
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
            selected_edges,
            rollout,
            chunks,
            environments,
            chunk_steps=chunk_steps,
            burn_in_steps=burn_in_steps,
            sigma=sigma,
        )
        valid = rollout.valid[times, environment_grid]
        log_differences.append((replayed - rollout.old_log_prob[times, environment_grid])[valid])
        mean_differences.append(
            (replayed_mean - rollout.old_latent_mean[times, environment_grid])[valid]
        )
    log_difference = torch.cat(log_differences)
    mean_difference = torch.cat(mean_differences)
    exact_kl = 0.5 * (mean_difference / sigma).square().sum(dim=-1).mean()
    maximum_log = log_difference.abs().max()
    maximum_mean = mean_difference.abs().max()
    passed = bool(maximum_log <= 5.0e-4 and maximum_mean <= 1.0e-6 and exact_kl <= 1.0e-8)
    return {
        "valid_action_steps": len(log_difference),
        "action_dimensions": 1,
        "maximum_absolute_log_probability_difference": float(maximum_log),
        "maximum_absolute_latent_mean_difference": float(maximum_mean),
        "exact_kl": float(exact_kl),
        "passed": passed,
    }


@torch.no_grad()
def burn_in_approximation_audit(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    rollout: PPORollout,
    *,
    chunk_steps: int,
    burn_in_steps: int,
    sigma: float,
    sequence_minibatch: int,
) -> dict[str, Any]:
    steps, episodes = rollout.valid.shape
    neural = controller.initial_state(episodes, device=rollout.images.device, dtype=torch.float32)
    full_prefix_means = torch.empty_like(rollout.old_latent_mean)
    for step in range(steps):
        motor_mean, neural = controller_step(
            controller,
            actor,
            selected_edges,
            rollout.images[step],
            rollout.roll_pitch[step],
            neural,
            rollout.specific_force[step],
            rollout.stick_position[step],
        )
        full_prefix_means[step] = latent_mean(motor_mean[:, 3:4])
    sequence_count = (steps // chunk_steps) * episodes
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
            selected_edges,
            rollout,
            chunks,
            environments,
            chunk_steps=chunk_steps,
            burn_in_steps=burn_in_steps,
            sigma=sigma,
        )
        valid = rollout.valid[times, environment_grid]
        differences.append((truncated_means - full_prefix_means[times, environment_grid])[valid])
    difference = torch.cat(differences)
    maximum = difference.abs().max()
    mean_kl = (0.5 * (difference / sigma).square().sum(dim=-1)).mean()
    passed = bool(maximum <= 5.0e-3 and mean_kl <= 1.0e-3)
    return {
        "valid_action_steps": len(difference),
        "burn_in_steps": burn_in_steps,
        "maximum_absolute_latent_mean_difference": float(maximum),
        "mean_exact_kl": float(mean_kl),
        "maximum_absolute_difference_threshold": 5.0e-3,
        "mean_exact_kl_threshold": 1.0e-3,
        "passed": passed,
    }


def ppo_update(
    controller: ConnectomeController,
    actor: RoutingActor,
    critic: PrivilegedCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    selected_edges: Tensor,
    rollout: PPORollout,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    assert rollout.advantages is not None and rollout.returns is not None
    steps, episodes = rollout.valid.shape
    sequence_count = (steps // args.chunk_steps) * episodes
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
            valid = rollout.valid[times, environment_grid]
            if not bool(valid.any()):
                continue
            new_log_prob, new_latent_mean = replay_sequence(
                controller,
                actor,
                selected_edges,
                rollout,
                chunks,
                environments,
                chunk_steps=args.chunk_steps,
                burn_in_steps=args.burn_in_steps,
                sigma=args.exploration_sigma_selected,
            )
            old_log_prob = rollout.old_log_prob[times, environment_grid]
            old_latent_mean = rollout.old_latent_mean[times, environment_grid]
            selected_new_log_prob = new_log_prob[valid]
            selected_old_log_prob = old_log_prob[valid]
            selected_new_mean = new_latent_mean[valid]
            selected_old_mean = old_latent_mean[valid]
            if not all(
                bool(torch.isfinite(values).all())
                for values in (
                    selected_new_log_prob,
                    selected_old_log_prob,
                    selected_new_mean,
                    selected_old_mean,
                )
            ):
                raise RuntimeError("non-finite throttle-policy replay")
            pre_update_kl = float(
                (
                    0.5
                    * ((selected_new_mean - selected_old_mean) / args.exploration_sigma_selected)
                    .square()
                    .sum(dim=-1)
                    .mean()
                ).detach()
            )
            if pre_update_kl > args.target_kl:
                stopped_for_kl = True
                break
            advantage = rollout.advantages[times, environment_grid][valid]
            ratio = torch.exp((selected_new_log_prob - selected_old_log_prob).clamp(-20.0, 20.0))
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            critic_prediction = critic(
                privileged_critic_input(
                    rollout.critic_features[times, environment_grid][valid],
                    rollout.neural_before[times, environment_grid][valid],
                )
            )
            value_error = critic_prediction - rollout.returns[times, environment_grid][valid]
            value_loss = 0.5 * value_error.square().mean()
            if not bool(torch.isfinite(policy_loss)) or not bool(torch.isfinite(value_loss)):
                raise RuntimeError("non-finite recurrent-routing PPO loss")
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            value_loss.backward()
            raw_actor_norm = actor.edge_magnitudes.grad.detach().norm()
            actor_before = actor.edge_magnitudes.detach().clone()
            optimizer_before = copy.deepcopy(actor_optimizer.state_dict())
            nn.utils.clip_grad_norm_(
                actor.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
            )
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            actor_optimizer.step()
            actor.project()
            with torch.no_grad():
                _, post_update_mean = replay_sequence(
                    controller,
                    actor,
                    selected_edges,
                    rollout,
                    chunks,
                    environments,
                    chunk_steps=args.chunk_steps,
                    burn_in_steps=args.burn_in_steps,
                    sigma=args.exploration_sigma_selected,
                )
                post_update_kl = float(
                    0.5
                    * (
                        (post_update_mean[valid] - selected_old_mean)
                        / args.exploration_sigma_selected
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                )
            accepted = math.isfinite(post_update_kl) and post_update_kl <= args.target_kl
            if not accepted:
                with torch.no_grad():
                    actor.edge_magnitudes.copy_(actor_before)
                actor_optimizer.load_state_dict(optimizer_before)
                stopped_for_kl = True
            critic_optimizer.step()
            updates.append(
                {
                    "epoch": epoch + 1,
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "pre_update_exact_kl": pre_update_kl,
                    "post_update_exact_kl": post_update_kl,
                    "actor_update_accepted": accepted,
                    "clip_fraction": float(((ratio - 1.0).abs() > args.ppo_clip).float().mean()),
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
        "mean_policy_loss": (
            float(np.mean([item["policy_loss"] for item in updates])) if updates else None
        ),
        "mean_value_loss": (
            float(np.mean([item["value_loss"] for item in updates])) if updates else None
        ),
        "maximum_raw_actor_gradient_norm": max(
            (item["raw_actor_gradient_norm"] for item in updates), default=0.0
        ),
    }


def likelihood_window_loss(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
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
        selected_edges,
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
    actor: RoutingActor,
    critic: PrivilegedCritic,
    selected_edges: Tensor,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    cases = diverse_matched_cases(
        args.gradient_check_episodes,
        seed=args.rollout_seed - 1,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    rollout = collect_rollout(
        controller,
        actor,
        critic,
        selected_edges,
        cases,
        seconds=args.gradient_check_steps * hover_config.dt,
        takeover_seconds=args.takeover_seconds,
        sigma=args.exploration_sigma,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=args.rollout_seed - 1,
    )
    repeated = [
        float(
            likelihood_window_loss(
                controller,
                actor,
                selected_edges,
                rollout,
                steps=args.gradient_check_steps,
                sigma=args.exploration_sigma,
            ).detach()
        )
        for _ in range(3)
    ]
    loss = likelihood_window_loss(
        controller,
        actor,
        selected_edges,
        rollout,
        steps=args.gradient_check_steps,
        sigma=args.exploration_sigma,
    )
    (gradient,) = torch.autograd.grad(loss, (actor.edge_magnitudes,))
    generator = torch.Generator(device=device).manual_seed(args.optimization_seed - 1)
    direction = torch.randn(actor.edge_magnitudes.shape, generator=generator, device=device)
    allowed = (actor.edge_magnitudes.detach() > 0.01) & (actor.edge_magnitudes.detach() < 7.99)
    direction *= allowed
    direction /= direction.abs().max()
    analytic = float((gradient * direction).sum())
    baseline = actor.edge_magnitudes.detach().clone()
    finite = []
    for scale in args.gradient_check_scales:
        losses = []
        for sign in (1.0, -1.0):
            with torch.no_grad():
                actor.edge_magnitudes.copy_(baseline + sign * scale * direction)
            losses.append(
                float(
                    likelihood_window_loss(
                        controller,
                        actor,
                        selected_edges,
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
        actor.edge_magnitudes.copy_(baseline)
    repeat_noise = max(repeated) - min(repeated)
    adjacent_passes = [
        left["matching_sign"]
        and right["matching_sign"]
        and left["relative_error"] <= args.gradient_check_tolerance
        and right["relative_error"] <= args.gradient_check_tolerance
        and abs(left["plus_loss"] - left["minus_loss"]) > 10.0 * repeat_noise
        and abs(right["plus_loss"] - right["minus_loss"]) > 10.0 * repeat_noise
        for left, right in zip(finite[:-1], finite[1:], strict=True)
    ]
    return {
        "window_steps": args.gradient_check_steps,
        "action_dimensions": 1,
        "unchanged_loss_repeats": repeated,
        "analytic_directional_derivative": analytic,
        "finite_differences": finite,
        "passed": any(adjacent_passes),
    }


@torch.no_grad()
def native_parity_audit(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    *,
    resolution: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(1_049_003)
    image = torch.rand(4, resolution, resolution, generator=generator, device=device)
    attitude = 0.1 * torch.randn(4, 2, generator=generator, device=device)
    neural = 0.2 * torch.randn(4, controller.n_nodes, generator=generator, device=device)
    force = torch.randn(4, 3, generator=generator, device=device)
    force[:, 2] += 9.81
    sticks = torch.rand(4, 4, generator=generator, device=device) * 2.0 - 1.0
    expected_motor, expected_neural = controller(image, attitude, neural, force, sticks)
    actual_motor, actual_neural = controller_step(
        controller,
        actor,
        selected_edges,
        image,
        attitude,
        neural,
        force,
        sticks,
    )
    differences = {
        "motor": float((actual_motor - expected_motor).abs().max()),
        "neural_state": float((actual_neural - expected_neural).abs().max()),
    }
    return {"differences": differences, "passed": max(differences.values()) <= 1.0e-6}


@torch.no_grad()
def assisted_evaluator_parity_audit(
    controller: ConnectomeController,
    actor: RoutingActor,
    selected_edges: Tensor,
    cases: BalancedCases,
    *,
    seconds: float,
    takeover_seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    actual = evaluate_actor(
        controller,
        actor,
        selected_edges,
        cases,
        seconds=seconds,
        takeover_seconds=takeover_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summary"]
    interface = motor_interface_spec(
        controller,
        bias_scale=0.01,
        log_gain_scale=0.05,
        maximum_bias_delta=0.15,
        maximum_gain_ratio=3.0,
    )
    expected = evaluate_assisted_policy_batch(
        controller,
        torch.zeros(1, len(interface.labels), device=cases.mass_scale.device),
        interface,
        cases,
        intervention="reserve_steering",
        takeover_seconds=takeover_seconds,
        seconds=seconds,
        shaping_weight_value=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        native_controller_forward=True,
    )["summaries"][0]
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
        "negative_lateral_success_rate": (
            actual["negative_lateral_success_rate"],
            expected["success_by_stratum"]["negative_lateral_offset"],
        ),
        "positive_lateral_success_rate": (
            actual["positive_lateral_success_rate"],
            expected["success_by_stratum"]["positive_lateral_offset"],
        ),
        "ring_collision_rate": (
            actual["ring_collision_rate"],
            expected["ring_collision_rate"],
        ),
        "miss_rate": (actual["miss_rate"], expected["miss_rate"]),
        "plane_crossing_rate": (
            actual["plane_crossing_rate"],
            expected["plane_crossing_rate"],
        ),
        "crossing_radial_mean_m": (
            actual["crossing_radial_mean_m"],
            expected["crossing_radial_mean_m"],
        ),
    }
    differences = {
        name: (
            0.0
            if left is None and right is None
            else float("inf")
            if left is None or right is None
            else abs(float(left) - float(right))
        )
        for name, (left, right) in comparable.items()
    }
    discrete_names = {
        "success_rate",
        "light_success_rate",
        "heavy_success_rate",
        "negative_lateral_success_rate",
        "positive_lateral_success_rate",
        "ring_collision_rate",
        "miss_rate",
        "plane_crossing_rate",
    }
    discrete = [difference for name, difference in differences.items() if name in discrete_names]
    continuous = [
        difference for name, difference in differences.items() if name not in discrete_names
    ]
    passed = max(discrete) == 0.0 and max(continuous) <= 1.0e-5
    return {"differences": differences, "passed": passed}


def compact_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": summary["success_rate"],
        "light_success_rate": summary["light_success_rate"],
        "heavy_success_rate": summary["heavy_success_rate"],
        "worst_mass_lateral_success_rate": mass_lateral_floor(summary),
        "negative_lateral_success_rate": summary["negative_lateral_success_rate"],
        "positive_lateral_success_rate": summary["positive_lateral_success_rate"],
        "ring_collision_rate": summary["ring_collision_rate"],
        "miss_rate": summary["miss_rate"],
        "crossing_radial_mean_m": summary["crossing_radial_mean_m"],
        "progress_mean": summary["progress_mean"],
        "height_alignment_mean": summary["height_alignment_mean"],
        "episode_reward_mean": summary["episode_reward_mean"],
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    validate_args(args, hover_config.dt)
    flight_audit = json.loads(args.flight_audit_report.read_text())
    prerequisite_checks = {
        "flight_audit_graph_hash_matches": flight_audit.get("graph_sha256")
        == file_sha256(args.graph),
        "flight_audit_checkpoint_hash_matches": flight_audit.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "flight_audit_positive_control_passed": flight_audit.get("positive_control_passed") is True,
        "flight_audit_candidates_failed": not any(
            flight_audit.get("candidate_progress", {}).values()
        ),
        "flight_audit_did_not_promote": flight_audit.get("compiled_or_promoted") is False,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"recurrent-routing PPO prerequisite failed: {prerequisite_checks}")
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    readout_spec = make_readout_spec(controller)
    routing_spec = make_recurrent_routing_spec(controller, path_spec, readout_spec)
    selected_edges = routing_spec.selected_edges
    if len(selected_edges) != 125:
        raise SystemExit(f"expected 125 selected edges, found {len(selected_edges)}")
    actor = RoutingActor(controller.edge_magnitude[selected_edges]).to(device)
    dummy_cases = diverse_matched_cases(
        8,
        seed=args.rollout_seed - 2,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
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
    seed_everything(args.optimization_seed)
    critic = PrivilegedCritic(feature_count).to(device)
    native_parity = native_parity_audit(
        controller, actor, selected_edges, resolution=resolution, device=device
    )
    if not native_parity["passed"]:
        raise SystemExit(f"source actor parity failed: {native_parity}")
    evaluator_parity = assisted_evaluator_parity_audit(
        controller,
        actor,
        selected_edges,
        dummy_cases,
        seconds=args.rollout_seconds,
        takeover_seconds=args.takeover_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if not evaluator_parity["passed"]:
        raise SystemExit(f"assisted evaluator parity failed: {evaluator_parity}")
    gradient_check = gradient_audit(
        controller,
        actor,
        critic,
        selected_edges,
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if not gradient_check["passed"]:
        raise SystemExit(f"throttle likelihood gradient audit failed: {gradient_check}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit("output directory contains a stale report")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    validation_cases = diverse_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    baseline_validation = evaluate_actor(
        controller,
        actor,
        selected_edges,
        validation_cases,
        seconds=args.rollout_seconds,
        takeover_seconds=args.takeover_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summary"]
    sigma_candidates = []
    exploration_scales = [args.exploration_sigma]
    while exploration_scales[-1] > args.minimum_exploration_sigma:
        exploration_scales.append(max(args.minimum_exploration_sigma, exploration_scales[-1] / 2.0))
    selected_sigma = None
    for sigma in exploration_scales:
        exploratory = evaluate_actor(
            controller,
            actor,
            selected_edges,
            validation_cases,
            seconds=args.rollout_seconds,
            takeover_seconds=args.takeover_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            sigma=float(sigma),
            seed=args.rollout_seed - 3,
        )["summary"]
        checks = {
            name: exploratory[name]
            >= baseline_validation[name] - args.maximum_exploration_success_drop
            for name in (
                "success_rate",
                "light_success_rate",
                "heavy_success_rate",
                "negative_lateral_success_rate",
                "positive_lateral_success_rate",
            )
        }
        acceptable = all(checks.values())
        sigma_candidates.append(
            {
                "sigma": float(sigma),
                "metrics": compact_metrics(exploratory),
                "acceptable_checks": checks,
                "acceptable": acceptable,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "exploration_calibration",
                    "sigma": float(sigma),
                    **compact_metrics(exploratory),
                    "acceptable": acceptable,
                }
            ),
            flush=True,
        )
        if acceptable:
            selected_sigma = float(sigma)
            break
    if selected_sigma is None:
        raise SystemExit("even minimum scalar-throttle exploration collapses performance")
    args.exploration_sigma_selected = selected_sigma
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
    archive: dict[str, dict[str, Any]] = {}
    iterations = []
    validations = []
    replay_audit = None
    stopped_at_checkpoint = False
    for iteration in range(1, args.iterations + 1):
        cases = diverse_matched_cases(
            args.episodes,
            seed=args.rollout_seed + iteration - 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        rollout = collect_rollout(
            controller,
            actor,
            critic,
            selected_edges,
            cases,
            seconds=args.rollout_seconds,
            takeover_seconds=args.takeover_seconds,
            sigma=selected_sigma,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.rollout_seed + iteration - 1,
        )
        seed_everything(args.optimization_seed + iteration)
        if iteration == 1:
            replay_audit = replay_consistency_audit(
                controller,
                actor,
                selected_edges,
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
            selected_edges,
            rollout,
            args=args,
        )
        burn_in_audit = burn_in_approximation_audit(
            controller,
            actor,
            selected_edges,
            rollout,
            chunk_steps=args.chunk_steps,
            burn_in_steps=args.burn_in_steps,
            sigma=selected_sigma,
            sequence_minibatch=args.sequence_minibatch,
        )
        if not burn_in_audit["passed"]:
            raise RuntimeError(f"truncated recurrent burn-in failed: {burn_in_audit}")
        entry = {
            "iteration": iteration,
            "rollout_seed": args.rollout_seed + iteration - 1,
            "exploratory_rollout": compact_metrics(rollout.summary),
            "critic_warmup_loss_first": warmup_losses[0] if warmup_losses else None,
            "critic_warmup_loss_last": warmup_losses[-1] if warmup_losses else None,
            "ppo": update,
            "burn_in_approximation_audit": burn_in_audit,
            "parameter_vector_sha256": parameter_sha256(actor),
        }
        iterations.append(entry)
        print(
            json.dumps(
                {
                    "phase": "ppo",
                    "iteration": iteration,
                    **compact_metrics(rollout.summary),
                    "maximum_kl": update["maximum_post_update_exact_kl"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "iteration": iteration,
                "selected_edge_indices": selected_edges.detach().cpu(),
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu(),
                "parameter_vector_sha256": parameter_sha256(actor),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic": critic.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            },
            args.output_dir / "training-state.pt",
        )
        if iteration % args.validation_interval == 0:
            metrics = evaluate_actor(
                controller,
                actor,
                selected_edges,
                validation_cases,
                seconds=args.rollout_seconds,
                takeover_seconds=args.takeover_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )["summary"]
            key = parameter_sha256(actor)
            archive[key] = {
                "iteration": iteration,
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu().clone(),
                "metrics": metrics,
                "checkpoint_progress": progress_gate(
                    baseline_validation,
                    metrics,
                    light_improvement=args.checkpoint_light_improvement,
                    floor_improvement=args.checkpoint_floor_improvement,
                    maximum_drop=args.maximum_stratum_drop,
                ),
                "development_progress": progress_gate(
                    baseline_validation,
                    metrics,
                    light_improvement=args.development_light_improvement,
                    floor_improvement=args.development_floor_improvement,
                    maximum_drop=args.maximum_stratum_drop,
                ),
            }
            validation_entry = {
                "parameter_vector_sha256": key,
                "iteration": iteration,
                "metrics": compact_metrics(metrics),
                "checkpoint_progress": archive[key]["checkpoint_progress"],
                "development_progress": archive[key]["development_progress"],
            }
            validations.append(validation_entry)
            print(json.dumps({"phase": "validation", **validation_entry}), flush=True)
            torch.save(
                {
                    archive_key: {
                        "iteration": item["iteration"],
                        "edge_magnitudes": item["edge_magnitudes"],
                    }
                    for archive_key, item in archive.items()
                },
                args.output_dir / "archive.pt",
            )
            if iteration == args.checkpoint_iteration and not any(
                item["checkpoint_progress"] for item in archive.values()
            ):
                stopped_at_checkpoint = True
                print(
                    json.dumps({"phase": "early_stop", "iteration": iteration}),
                    flush=True,
                )
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

    qualified = [key for key, item in archive.items() if item["development_progress"]]
    selection_pool = qualified or [
        key
        for key, item in archive.items()
        if mass_lateral_safe(
            baseline_validation, item["metrics"], maximum_drop=args.maximum_stratum_drop
        )
    ]
    selection_pool = selection_pool or list(archive)
    winner_key = max(
        selection_pool,
        key=lambda key: (
            mass_lateral_floor(archive[key]["metrics"]),
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
    candidate_vector_path = args.output_dir / "candidate-vector.json"
    candidate_vector_path.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": file_sha256(args.checkpoint),
                "selected_iteration": winner["iteration"],
                "parameter_vector_sha256": parameter_sha256(actor),
                "selected_edge_indices": selected_edges.detach().cpu().tolist(),
                "edge_magnitudes": actor.edge_magnitudes.detach().cpu().tolist(),
                "development_qualified": winner_key in qualified,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    development_passed = bool(qualified)
    final = None
    final_progress = False
    paired_final = None
    if development_passed:
        final_cases = diverse_matched_cases(
            args.final_episodes,
            seed=args.final_seed,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        baseline_actor = RoutingActor(controller.edge_magnitude[selected_edges]).to(device)
        reference = evaluate_actor(
            controller,
            baseline_actor,
            selected_edges,
            final_cases,
            seconds=args.rollout_seconds,
            takeover_seconds=args.takeover_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            return_outcomes=True,
        )
        candidate = evaluate_actor(
            controller,
            actor,
            selected_edges,
            final_cases,
            seconds=args.rollout_seconds,
            takeover_seconds=args.takeover_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            return_outcomes=True,
        )
        reference_success = reference["outcomes"]["success"]
        candidate_success = candidate["outcomes"]["success"]
        codes = reference["outcomes"]["codes"]
        light = ~codes.bitwise_and(1).bool()
        paired_final = {
            "overall": paired_clustered_confidence_interval(reference_success, candidate_success),
            "light": paired_confidence_interval(reference_success[light], candidate_success[light]),
            "heavy": paired_confidence_interval(
                reference_success[~light], candidate_success[~light]
            ),
        }
        final_progress = progress_gate(
            reference["summary"],
            candidate["summary"],
            light_improvement=args.development_light_improvement,
            floor_improvement=args.development_floor_improvement,
            maximum_drop=args.maximum_stratum_drop,
        )
        final = {
            "reference": compact_metrics(reference["summary"]),
            "candidate": compact_metrics(candidate["summary"]),
        }
        print(
            json.dumps(
                {
                    "phase": "fresh_final",
                    "reference": final["reference"],
                    "candidate": final["candidate"],
                    "progress_passed": final_progress,
                }
            ),
            flush=True,
        )

    report = {
        "method": "teacher-steering-assisted scalar-throttle recurrent PPO",
        "claim_scope": (
            "The candidate native recurrent state controls throttle throughout. A frozen "
            "source supplies steering for the first 0.5 seconds and a privileged reserve "
            "teacher supplies steering thereafter. This is not a native coupled-flight test."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "flight_audit_report": stable_path(args.flight_audit_report),
        "flight_audit_report_sha256": file_sha256(args.flight_audit_report),
        "prerequisite_checks": prerequisite_checks,
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
            "balanced_mass_sampling_and_rewards": True,
            "critic_initialization_seed": args.optimization_seed,
            "optimization_shuffle_seed_schedule": "optimization_seed + iteration",
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "selected_edge_magnitudes_only": True,
            "biases_frozen": True,
            "time_constants_frozen": True,
            "training_action": "scalar native throttle motor drive",
            "launch_steering": "parallel frozen-source controller for first 0.5 seconds",
            "later_steering": "privileged reserve teacher after 0.5 seconds",
            "reward": (
                "+10 terminal success, -2 terminal failure, and a total bounded unit "
                "potential split equally between forward progress and gate-height alignment"
            ),
            "critic_training_only": True,
            "critic_inputs": (
                "physical state, gate pose, mass, foreleg state, reward bookkeeping, and "
                "detached native connectome state"
            ),
            "actor_inputs": (
                "current FPV, roll/pitch, body-Z specific force, and persistent native "
                "connectome state"
                + (" plus foreleg position" if controller.uses_proprioception else "")
            ),
            "proprioception_input_active": controller.uses_proprioception,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
            "fresh_final_only_after_development_qualification": True,
            "automatic_merge_or_promotion": False,
        },
        "parameterization": {
            "fixed_sign_edge_magnitudes": len(selected_edges),
            "selected_edge_indices": selected_edges.detach().cpu().tolist(),
        },
        "native_forward_parity": native_parity,
        "assisted_evaluator_parity": evaluator_parity,
        "likelihood_gradient_audit": gradient_check,
        "unchanged_policy_replay_audit": replay_audit,
        "exploration_calibration": sigma_candidates,
        "baseline_validation": compact_metrics(baseline_validation),
        "iterations_completed": len(iterations),
        "stopped_at_iteration_10_gate": stopped_at_checkpoint,
        "iterations": iterations,
        "validations": validations,
        "selected_candidate": winner_key,
        "selected_candidate_validation": {
            "iteration": winner["iteration"],
            "metrics": compact_metrics(winner["metrics"]),
            "checkpoint_progress": winner["checkpoint_progress"],
            "development_progress": winner["development_progress"],
        },
        "selected_candidate_vector_file": stable_path(candidate_vector_path),
        "selected_candidate_vector_file_sha256": file_sha256(candidate_vector_path),
        "development_passed": development_passed,
        "fresh_final_consumed": development_passed,
        "final": final,
        "paired_final": paired_final,
        "final_progress_passed": final_progress,
        "compiled_or_promoted": False,
        "candidate_checkpoint": None,
        "goal_passed": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "iterations_completed": len(iterations),
                "development_passed": development_passed,
                "fresh_final_consumed": development_passed,
                "final_progress_passed": final_progress,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if final_progress else 2


if __name__ == "__main__":
    raise SystemExit(main())
