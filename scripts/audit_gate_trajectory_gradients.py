#!/usr/bin/env python3
"""Audit and briefly exercise full-horizon gradients through the recurrent gate actor."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    actor_teacher_gate_rc,
    evaluate_gate,
    file_sha256,
    initial_rollout,
    seed_everything,
)

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    render_annular_gate,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
)


@dataclass(frozen=True)
class TrajectoryCases:
    """Immutable initial conditions reused for exact same-case comparisons."""

    state: QuadState
    gate: AnnularGate
    mass_scale: Tensor


@dataclass
class TrajectoryObjective:
    """Differentiable objective and detached diagnostics from one rollout."""

    loss: Tensor
    late_loss: Tensor
    first_motor: Tensor | None
    position_loss: Tensor
    velocity_loss: Tensor
    excessive_tilt_loss: Tensor
    per_episode_loss: Tensor
    crossing_radial: Tensor
    crossed: Tensor
    passed: Tensor
    collision: Tensor
    missed: Tensor
    saturation: dict[str, float]


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
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "trajectory-gradient-probe-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--train-episodes", type=int, default=8)
    parser.add_argument("--dev-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--evaluation-seconds", type=float, default=12.0)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--edge-bias-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--maximum-backtracks", type=int, default=8)
    parser.add_argument(
        "--finite-difference-epsilon",
        type=float,
        nargs="+",
        default=(1.0e-4, 3.0e-5, 1.0e-5),
    )
    parser.add_argument("--finite-difference-relative-tolerance", type=float, default=0.10)
    parser.add_argument("--early-motor-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--minimum-train-improvement", type=float, default=0.05)
    parser.add_argument("--maximum-side-success-drop", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=560_031)
    parser.add_argument("--dev-seed", type=int, default=570_031)
    parser.add_argument("--final-seed", type=int, default=580_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.steps,
        args.train_episodes,
        args.dev_episodes,
        args.final_episodes,
        args.evaluation_seconds,
        args.updates,
        args.edge_bias_learning_rate,
        args.time_constant_learning_rate,
        args.gradient_clip,
        args.maximum_backtracks,
        args.finite_difference_relative_tolerance,
        args.early_motor_epsilon,
        args.minimum_train_improvement,
        args.maximum_side_success_drop,
        *args.finite_difference_epsilon,
    )
    if min(positive) <= 0.0:
        raise SystemExit("episode counts, steps, rates, and thresholds must be positive")
    if args.train_episodes % 2:
        raise SystemExit("--train-episodes must be even for mirrored cases")
    if args.dev_episodes % 8 or args.final_episodes % 8:
        raise SystemExit("--dev-episodes and --final-episodes must be divisible by eight")
    if args.updates > 10:
        raise SystemExit("this bounded probe permits at most ten updates")
    if sorted(args.finite_difference_epsilon, reverse=True) != list(args.finite_difference_epsilon):
        raise SystemExit("finite-difference epsilons must be given in descending order")


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def clone_state(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def make_cases(
    episodes: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> TrajectoryCases:
    """Make exact lateral mirrors with matched masses and both mass signs."""

    if episodes % 2:
        raise ValueError("mirrored cases require an even episode count")
    seed_everything(seed)
    state, sampled_gate, sampled_mass = initial_rollout(
        episodes // 2,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    half = episodes // 2
    magnitudes = (sampled_mass - 1.0).abs().clamp_min(1.0e-4)
    mass_sign = torch.where(
        torch.arange(half, device=device).remainder(2).bool(),
        torch.ones(half, device=device),
        -torch.ones(half, device=device),
    )
    paired_mass = 1.0 + mass_sign * magnitudes
    mirrored_center = sampled_gate.center.clone()
    mirrored_center[:, 1] *= -1.0
    gate = AnnularGate(
        center=torch.cat((sampled_gate.center, mirrored_center)),
        yaw=torch.cat((sampled_gate.yaw, -sampled_gate.yaw)),
    )
    full_state = QuadState(*(torch.cat((value, value.clone())) for value in state.as_tuple()))
    return TrajectoryCases(
        state=full_state,
        gate=gate,
        mass_scale=torch.cat((paired_mass, paired_mass)),
    )


def load_controller(
    graph: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ConnectomeController, dict[str, Any], HoverConfig, GateConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("graph_sha256") != file_sha256(graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match the evaluator")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    resolution = int(checkpoint["image_resolution"])
    controller = ConnectomeController(
        graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    incompatible = controller.load_state_dict(checkpoint["controller"], strict=False)
    allowed_missing = {"initial_raw_time_constant"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise SystemExit(
            "checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return controller, checkpoint, hover_config, gate_config, resolution


def trajectory_objective(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    steps: int,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    retain_first_motor: bool = False,
    first_motor_delta: Tensor | None = None,
) -> TrajectoryObjective:
    """Run one uninterrupted recurrent actor/leg/vehicle graph for the full horizon."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    teacher_state = clone_state(cases.state)
    student_state = clone_state(cases.state)
    teacher_sticks = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    student_sticks = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    position_scale = torch.tensor((0.5, 0.5, 0.25), device=device)
    per_episode_position = torch.zeros(episodes, device=device)
    per_episode_velocity = torch.zeros(episodes, device=device)
    per_episode_tilt = torch.zeros(episodes, device=device)
    per_episode_late = torch.zeros(episodes, device=device)
    late_steps = 0
    passed = torch.zeros(episodes, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    crossing_radial = torch.full((episodes,), float("nan"), device=device)
    tanh_saturation = torch.zeros((), device=device)
    membrane_bound_saturation = torch.zeros((), device=device)
    motor_sigmoid_saturation = torch.zeros((), device=device)
    motor_nodes = controller.pool_indices
    first_motor: Tensor | None = None
    controller.train()
    for step in range(steps):
        with torch.no_grad():
            teacher_motor = motor_target_for_rc(
                actor_teacher_gate_rc(controller, teacher_state, cases.gate, hover_config),
                hover_config,
            )
            teacher_rc, teacher_sticks = sticks(teacher_motor, teacher_sticks)
            teacher_state = quad(teacher_rc, teacher_state, cases.mass_scale)
        image = render_annular_gate(
            student_state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(
            image,
            student_state.euler[:, :2],
            neural,
            student_state.specific_force,
            student_sticks.position,
        )
        if step == 0:
            first_motor = motor
            if retain_first_motor:
                first_motor.retain_grad()
            if first_motor_delta is not None:
                motor = motor + first_motor_delta
        student_rc, student_sticks = sticks(motor, student_sticks)
        previous_position = student_state.position
        student_state = quad(student_rc, student_state, cases.mass_scale)
        position_error = (student_state.position - teacher_state.position) / position_scale
        position_term = position_error.square().mean(dim=1)
        velocity_term = (student_state.velocity - teacher_state.velocity).square().mean(dim=1)
        tilt = torch.linalg.vector_norm(student_state.euler[:, :2], dim=1)
        tilt_term = torch.relu(tilt - math.radians(35.0)).square()
        per_episode_position += position_term
        per_episode_velocity += velocity_term
        per_episode_tilt += tilt_term
        if step >= 3 * steps // 4:
            per_episode_late += position_term + 0.1 * velocity_term + 0.25 * tilt_term
            late_steps += 1
        with torch.no_grad():
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position.detach(),
                student_state.position.detach(),
                cases.gate,
                gate_config,
            )
            crossing_now = (pass_now | collision_now | miss_now) & crossing_radial.isnan()
            if bool(crossing_now.any()):
                _, lateral, vertical = crossing_coordinates(
                    previous_position.detach(), student_state.position.detach(), cases.gate
                )
                radial = torch.sqrt(lateral.square() + vertical.square())
                crossing_radial[crossing_now] = radial[crossing_now]
            passed |= pass_now
            collision |= collision_now
            missed |= miss_now
            activity = torch.tanh(neural.detach())
            tanh_saturation += (activity.abs() >= 0.98).float().mean()
            membrane_bound_saturation += (neural.detach().abs() >= 4.9).float().mean()
            motor_activity = torch.sigmoid(neural.detach()[:, motor_nodes])
            motor_sigmoid_saturation += (
                ((motor_activity <= 0.01) | (motor_activity >= 0.99)).float().mean()
            )
    per_episode_position /= steps
    per_episode_velocity /= steps
    per_episode_tilt /= steps
    per_episode_loss = per_episode_position + 0.1 * per_episode_velocity + 0.25 * per_episode_tilt
    return TrajectoryObjective(
        loss=per_episode_loss.mean(),
        late_loss=(per_episode_late / max(late_steps, 1)).mean(),
        first_motor=first_motor,
        position_loss=per_episode_position.mean(),
        velocity_loss=per_episode_velocity.mean(),
        excessive_tilt_loss=per_episode_tilt.mean(),
        per_episode_loss=per_episode_loss,
        crossing_radial=crossing_radial,
        crossed=~crossing_radial.isnan(),
        passed=passed,
        collision=collision,
        missed=missed,
        saturation={
            "tanh_activity_fraction": float(tanh_saturation / steps),
            "membrane_near_bound_fraction": float(membrane_bound_saturation / steps),
            "motor_sigmoid_fraction": float(motor_sigmoid_saturation / steps),
        },
    )


@torch.no_grad()
def teacher_identity_loss(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    steps: int,
    hover_config: HoverConfig,
) -> dict[str, float]:
    """Apply identical teacher actions to two cloned plants and compare trajectories."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    reference = clone_state(cases.state)
    identical = clone_state(cases.state)
    reference_sticks = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    identical_sticks = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    position_loss = torch.zeros((), device=device)
    velocity_loss = torch.zeros((), device=device)
    for _ in range(steps):
        motor = motor_target_for_rc(
            actor_teacher_gate_rc(controller, reference, cases.gate, hover_config),
            hover_config,
        )
        reference_rc, reference_sticks = sticks(motor, reference_sticks)
        identical_rc, identical_sticks = sticks(motor, identical_sticks)
        reference = quad(reference_rc, reference, cases.mass_scale)
        identical = quad(identical_rc, identical, cases.mass_scale)
        position_loss += (reference.position - identical.position).square().mean()
        velocity_loss += (reference.velocity - identical.velocity).square().mean()
    return {
        "position_loss": float(position_loss / steps),
        "velocity_loss": float(velocity_loss / steps),
    }


def first_forward_parity(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, float]:
    """Check grad-enabled actor inputs against the frozen deployment call signature."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    sticks = ForelegStickPlant(hover_config).to(device)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    image = render_annular_gate(
        cases.state,
        cases.gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    motor_grad, state_grad = controller(
        image,
        cases.state.euler[:, :2],
        neural,
        cases.state.specific_force,
        stick_state.position,
    )
    with torch.no_grad():
        motor_frozen, state_frozen = controller(
            image,
            cases.state.euler[:, :2],
            neural,
            cases.state.specific_force,
            stick_state.position,
        )
    return {
        "motor_max_absolute_difference": float((motor_grad - motor_frozen).detach().abs().max()),
        "state_max_absolute_difference": float((state_grad - state_frozen).detach().abs().max()),
    }


def trainable_parameters(controller: ConnectomeController) -> dict[str, Tensor]:
    expected = {"edge_magnitude", "bias", "raw_time_constant"}
    selected = {name: parameter for name, parameter in controller.named_parameters()}
    if set(selected) != expected:
        raise RuntimeError(f"unexpected controller parameters: {sorted(selected)}")
    return selected


def roll_bias_direction(controller: ConnectomeController) -> dict[str, Tensor]:
    direction = {
        name: torch.zeros_like(value) for name, value in trainable_parameters(controller).items()
    }
    positive_end = int(controller.pool_offsets[1].item())
    negative_end = int(controller.pool_offsets[2].item())
    positive = controller.pool_indices[:positive_end]
    negative = controller.pool_indices[positive_end:negative_end]
    direction["bias"][positive] = 1.0
    direction["bias"][negative] = -1.0
    return direction


def mixed_direction(controller: ConnectomeController, *, seed: int) -> dict[str, Tensor]:
    parameters = trainable_parameters(controller)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result: dict[str, Tensor] = {}
    for name, parameter in parameters.items():
        signs = torch.randint(
            0,
            2,
            parameter.shape,
            generator=generator,
            device="cpu",
            dtype=torch.int64,
        ).to(device=parameter.device, dtype=parameter.dtype)
        signs = 2.0 * signs - 1.0
        signs /= math.sqrt(max(parameter.numel(), 1))
        result[name] = signs
    edge = parameters["edge_magnitude"].detach()
    eligible = (edge > 1.0e-3) & (edge < 8.0 - 1.0e-3)
    result["edge_magnitude"] *= eligible
    maximum = max(float(value.abs().max()) for value in result.values())
    if maximum <= 0.0:
        raise RuntimeError("mixed direction is empty")
    return {name: value / maximum for name, value in result.items()}


def parameter_snapshot(controller: ConnectomeController) -> dict[str, Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in trainable_parameters(controller).items()
    }


def restore_parameters(controller: ConnectomeController, snapshot: dict[str, Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in trainable_parameters(controller).items():
            parameter.copy_(snapshot[name])


def perturb_parameters(
    controller: ConnectomeController,
    base: dict[str, Tensor],
    direction: dict[str, Tensor],
    scale: float,
) -> None:
    with torch.no_grad():
        for name, parameter in trainable_parameters(controller).items():
            parameter.copy_(base[name] + scale * direction[name])


def direction_dot_gradient(
    controller: ConnectomeController,
    direction: dict[str, Tensor],
) -> float:
    total = torch.zeros((), device=controller.bias.device)
    for name, parameter in trainable_parameters(controller).items():
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        total += (parameter.grad * direction[name]).sum()
    return float(total)


@torch.no_grad()
def objective_value(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    steps: int,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    late: bool = False,
    first_motor_delta: Tensor | None = None,
) -> float:
    objective = trajectory_objective(
        controller,
        cases,
        steps=steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        first_motor_delta=first_motor_delta,
    )
    return float(objective.late_loss if late else objective.loss)


def finite_difference_audit(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    directions: dict[str, dict[str, Tensor]],
    analytic: dict[str, float],
    *,
    epsilons: list[float],
    numerical_noise: float,
    relative_tolerance: float,
    steps: int,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    base = parameter_snapshot(controller)
    noise_threshold = max(10.0 * numerical_noise, 1.0e-10)
    result: dict[str, Any] = {}
    try:
        for direction_name, direction in directions.items():
            checks = []
            for epsilon in epsilons:
                perturb_parameters(controller, base, direction, epsilon)
                plus = objective_value(
                    controller,
                    cases,
                    steps=steps,
                    resolution=resolution,
                    hover_config=hover_config,
                    gate_config=gate_config,
                )
                perturb_parameters(controller, base, direction, -epsilon)
                minus = objective_value(
                    controller,
                    cases,
                    steps=steps,
                    resolution=resolution,
                    hover_config=hover_config,
                    gate_config=gate_config,
                )
                estimate = (plus - minus) / (2.0 * epsilon)
                denominator = max(abs(analytic[direction_name]), abs(estimate), noise_threshold)
                relative_error = abs(estimate - analytic[direction_name]) / denominator
                measurable = abs(plus - minus) > noise_threshold
                same_sign = estimate * analytic[direction_name] > 0.0
                checks.append(
                    {
                        "epsilon": epsilon,
                        "plus_loss": plus,
                        "minus_loss": minus,
                        "loss_difference": plus - minus,
                        "central_difference": estimate,
                        "analytic_directional_derivative": analytic[direction_name],
                        "relative_error": relative_error,
                        "measurable_above_noise": measurable,
                        "same_sign": same_sign,
                        "passed": bool(
                            measurable and same_sign and relative_error <= relative_tolerance
                        ),
                    }
                )
            adjacent_pair_passed = any(
                checks[index]["passed"] and checks[index + 1]["passed"]
                for index in range(len(checks) - 1)
            )
            result[direction_name] = {
                "checks": checks,
                "two_adjacent_scales_passed": adjacent_pair_passed,
            }
    finally:
        restore_parameters(controller, base)
    return result


def gradient_audit(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    args: argparse.Namespace,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    parity = first_forward_parity(
        controller,
        cases,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    identity = teacher_identity_loss(
        controller,
        cases,
        steps=args.steps,
        hover_config=hover_config,
    )
    repeated = [
        objective_value(
            controller,
            cases,
            steps=args.steps,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        for _ in range(3)
    ]
    numerical_noise = max(repeated) - min(repeated)
    controller.zero_grad(set_to_none=True)
    objective = trajectory_objective(
        controller,
        cases,
        steps=args.steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        retain_first_motor=True,
    )
    assert objective.first_motor is not None
    early_gradient = torch.autograd.grad(
        objective.late_loss,
        objective.first_motor,
        retain_graph=True,
    )[0]
    objective.loss.backward()
    parameter_gradients = {}
    all_finite = True
    for name, parameter in trainable_parameters(controller).items():
        if parameter.grad is None:
            parameter_gradients[name] = None
            all_finite = False
            continue
        finite = bool(torch.isfinite(parameter.grad).all())
        all_finite &= finite
        parameter_gradients[name] = {
            "finite": finite,
            "l2_norm": float(torch.linalg.vector_norm(parameter.grad)),
            "maximum_absolute": float(parameter.grad.abs().max()),
            "nonzero_fraction": float((parameter.grad != 0.0).float().mean()),
        }
    directions = {
        "roll_motor_bias_contrast": roll_bias_direction(controller),
        "mixed_all_parameters": mixed_direction(controller, seed=args.seed + 17),
    }
    analytic = {
        name: direction_dot_gradient(controller, direction)
        for name, direction in directions.items()
    }
    finite_difference = finite_difference_audit(
        controller,
        cases,
        directions,
        analytic,
        epsilons=list(args.finite_difference_epsilon),
        numerical_noise=numerical_noise,
        relative_tolerance=args.finite_difference_relative_tolerance,
        steps=args.steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    roll_delta = torch.zeros(
        len(cases.mass_scale), 4, device=cases.mass_scale.device, dtype=torch.float32
    )
    roll_delta[:, 0] = args.early_motor_epsilon
    late_plus = objective_value(
        controller,
        cases,
        steps=args.steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        late=True,
        first_motor_delta=roll_delta,
    )
    late_minus = objective_value(
        controller,
        cases,
        steps=args.steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        late=True,
        first_motor_delta=-roll_delta,
    )
    early_analytic = float(early_gradient[:, 0].sum())
    early_finite_difference = (late_plus - late_minus) / (2.0 * args.early_motor_epsilon)
    early_measurable = abs(late_plus - late_minus) > max(10.0 * numerical_noise, 1.0e-10)
    early_nonzero = abs(early_analytic) > 0.0
    early_same_sign = early_analytic * early_finite_difference > 0.0
    finite_difference_passed = all(
        direction["two_adjacent_scales_passed"] for direction in finite_difference.values()
    )
    passed = bool(
        identity["position_loss"] <= 1.0e-12
        and identity["velocity_loss"] <= 1.0e-12
        and parity["motor_max_absolute_difference"] <= 1.0e-7
        and parity["state_max_absolute_difference"] <= 1.0e-7
        and all_finite
        and finite_difference_passed
        and early_measurable
        and early_nonzero
        and early_same_sign
    )
    controller.zero_grad(set_to_none=True)
    return {
        "passed": passed,
        "teacher_identity": identity,
        "deployed_forward_parity": parity,
        "repeated_unchanged_losses": repeated,
        "observed_numerical_noise": numerical_noise,
        "baseline_objective": float(objective.loss.detach()),
        "baseline_late_objective": float(objective.late_loss.detach()),
        "parameter_gradients": parameter_gradients,
        "analytic_directional_derivatives": analytic,
        "finite_difference": finite_difference,
        "late_loss_to_first_roll_motor": {
            "analytic_directional_derivative": early_analytic,
            "central_difference": early_finite_difference,
            "plus_loss": late_plus,
            "minus_loss": late_minus,
            "measurable_above_noise": early_measurable,
            "analytic_nonzero": early_nonzero,
            "same_sign": early_same_sign,
        },
        "saturation": objective.saturation,
    }


def objective_summary(objective: TrajectoryObjective, cases: TrajectoryCases) -> dict[str, Any]:
    negative = cases.gate.center[:, 1] < 0.0
    result: dict[str, Any] = {
        "trajectory_loss": float(objective.loss),
        "position_loss": float(objective.position_loss),
        "velocity_loss": float(objective.velocity_loss),
        "excessive_tilt_loss": float(objective.excessive_tilt_loss),
        "clean_pass_rate": float(
            (objective.passed & ~objective.collision & ~objective.missed).float().mean()
        ),
        "plane_crossing_rate": float(objective.crossed.float().mean()),
        "saturation": objective.saturation,
        "trajectory_loss_by_lateral_side": {},
        "crossing_radial_mean_by_lateral_side_m": {},
    }
    for name, mask in (("negative", negative), ("positive", ~negative)):
        result["trajectory_loss_by_lateral_side"][name] = float(
            objective.per_episode_loss[mask].mean()
        )
        crossed = mask & objective.crossed
        result["crossing_radial_mean_by_lateral_side_m"][name] = (
            float(objective.crossing_radial[crossed].mean()) if bool(crossed.any()) else None
        )
    return result


@torch.no_grad()
def evaluate_trajectory_cases(
    controller: ConnectomeController,
    cases: TrajectoryCases,
    *,
    steps: int,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    return objective_summary(
        trajectory_objective(
            controller,
            cases,
            steps=steps,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        ),
        cases,
    )


def strict_development_evaluation(
    controller: ConnectomeController,
    *,
    episodes: int,
    seconds: float,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    return evaluate_gate(
        controller,
        episodes=episodes,
        seconds=seconds,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=seed,
        balanced_strata=True,
    )


def make_optimizer(controller: ConnectomeController, args: argparse.Namespace) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        (
            {
                "params": [controller.edge_magnitude, controller.bias],
                "lr": args.edge_bias_learning_rate,
            },
            {
                "params": [controller.raw_time_constant],
                "lr": args.time_constant_learning_rate,
            },
        ),
        weight_decay=0.0,
    )


def learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def halve_learning_rates(optimizer: torch.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) / 2.0


def train_one_accepted_update(
    controller: ConnectomeController,
    optimizer: torch.optim.AdamW,
    cases: TrajectoryCases,
    *,
    args: argparse.Namespace,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    base_parameters = parameter_snapshot(controller)
    base_optimizer = copy.deepcopy(optimizer.state_dict())
    with torch.no_grad():
        pre_loss = objective_value(
            controller,
            cases,
            steps=args.steps,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
    attempts = []
    for backtrack in range(args.maximum_backtracks + 1):
        restore_parameters(controller, base_parameters)
        optimizer.load_state_dict(base_optimizer)
        if backtrack:
            for _ in range(backtrack):
                halve_learning_rates(optimizer)
        optimizer.zero_grad(set_to_none=True)
        objective = trajectory_objective(
            controller,
            cases,
            steps=args.steps,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        objective.loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.gradient_clip)
        if not torch.isfinite(raw_norm):
            raise RuntimeError("non-finite full-horizon gradient")
        optimizer.step()
        controller.project_parameters()
        post_loss = objective_value(
            controller,
            cases,
            steps=args.steps,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        accepted = post_loss <= pre_loss
        attempts.append(
            {
                "backtrack": backtrack,
                "learning_rates": learning_rates(optimizer),
                "pre_loss": pre_loss,
                "post_loss": post_loss,
                "raw_gradient_norm": float(raw_norm),
                "accepted": accepted,
            }
        )
        if accepted:
            return {"accepted": True, "attempts": attempts}
    restore_parameters(controller, base_parameters)
    optimizer.load_state_dict(base_optimizer)
    for _ in range(args.maximum_backtracks + 1):
        halve_learning_rates(optimizer)
    return {"accepted": False, "attempts": attempts}


def evaluate_stage(
    controller: ConnectomeController,
    dev_cases: TrajectoryCases,
    *,
    update: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    trajectory = evaluate_trajectory_cases(
        controller,
        dev_cases,
        steps=args.steps,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    strict = strict_development_evaluation(
        controller,
        episodes=args.dev_episodes,
        seconds=args.evaluation_seconds,
        seed=args.dev_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    result = {"update": update, "trajectory": trajectory, "strict_flight": strict}
    print(
        json.dumps(
            {
                "phase": "development",
                "update": update,
                "trajectory_loss": trajectory["trajectory_loss"],
                "crossing_radial_by_side": strict["crossing_radial_mean_by_lateral_side_m"],
                "success_rate": strict["success_rate"],
                "success_by_side": {
                    "negative": strict["success_by_stratum"]["negative_lateral_offset"],
                    "positive": strict["success_by_stratum"]["positive_lateral_offset"],
                },
            }
        ),
        flush=True,
    )
    return result


def go_no_go(
    development: list[dict[str, Any]],
    training: list[dict[str, Any]],
    *,
    minimum_train_improvement: float,
    maximum_side_success_drop: float,
) -> dict[str, Any]:
    baseline = development[0]
    final = development[-1]
    initial_train = training[0]["loss"]
    final_train = training[-1]["loss"]
    train_improvement = (initial_train - final_train) / max(abs(initial_train), 1.0e-12)
    dev_improved = (
        final["trajectory"]["trajectory_loss"] < baseline["trajectory"]["trajectory_loss"]
    )
    radial_checks = {}
    success_checks = {}
    for side in ("negative", "positive"):
        before_radial = baseline["strict_flight"]["crossing_radial_mean_by_lateral_side_m"][side]
        after_radial = final["strict_flight"]["crossing_radial_mean_by_lateral_side_m"][side]
        radial_checks[side] = bool(
            before_radial is not None and after_radial is not None and after_radial < before_radial
        )
        before_success = baseline["strict_flight"]["success_by_stratum"][f"{side}_lateral_offset"]
        after_success = final["strict_flight"]["success_by_stratum"][f"{side}_lateral_offset"]
        success_checks[side] = bool(
            before_success is not None
            and after_success is not None
            and after_success >= before_success - maximum_side_success_drop
        )
    checks = {
        "minimum_training_loss_reduction": train_improvement >= minimum_train_improvement,
        "development_trajectory_loss_improved": dev_improved,
        "crossing_radial_error_improved_both_sides": all(radial_checks.values()),
        "strict_success_drop_at_most_five_points_both_sides": all(success_checks.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "training_loss_fractional_improvement": train_improvement,
        "radial_checks": radial_checks,
        "side_success_checks": success_checks,
    }


def save_candidate(
    path: Path,
    checkpoint: dict[str, Any],
    controller: ConnectomeController,
    report_summary: dict[str, Any],
) -> None:
    result = copy.deepcopy(checkpoint)
    result["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    result["trajectory_gradient_probe"] = report_summary
    torch.save(result, path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train_cases = make_cases(
        args.train_episodes,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    dev_cases = make_cases(
        args.dev_episodes,
        seed=args.dev_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    gradient = gradient_audit(
        controller,
        train_cases,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    print(
        json.dumps(
            {
                "phase": "gradient_audit",
                "passed": gradient["passed"],
                "baseline_objective": gradient["baseline_objective"],
                "finite_difference": {
                    name: result["two_adjacent_scales_passed"]
                    for name, result in gradient["finite_difference"].items()
                },
                "late_credit_nonzero": gradient["late_loss_to_first_roll_motor"][
                    "analytic_nonzero"
                ],
                "peak_cuda_memory_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
        ),
        flush=True,
    )
    development: list[dict[str, Any]] = []
    training = [{"update": 0, "loss": gradient["baseline_objective"]}]
    learning_probe_run = False
    if gradient["passed"]:
        learning_probe_run = True
        development.append(
            evaluate_stage(
                controller,
                dev_cases,
                update=0,
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
        )
        optimizer = make_optimizer(controller, args)
        for update in range(1, args.updates + 1):
            result = train_one_accepted_update(
                controller,
                optimizer,
                train_cases,
                args=args,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            loss = (
                result["attempts"][-1]["post_loss"] if result["accepted"] else training[-1]["loss"]
            )
            training.append({"update": update, "loss": loss, **result})
            print(
                json.dumps(
                    {
                        "phase": "training",
                        "update": update,
                        "accepted": result["accepted"],
                        "loss": loss,
                        "attempts": len(result["attempts"]),
                    }
                ),
                flush=True,
            )
            if not result["accepted"]:
                break
            if update in {5, args.updates}:
                development.append(
                    evaluate_stage(
                        controller,
                        dev_cases,
                        update=update,
                        args=args,
                        resolution=resolution,
                        device=device,
                        hover_config=hover_config,
                        gate_config=gate_config,
                    )
                )
    decision = (
        go_no_go(
            development,
            training,
            minimum_train_improvement=args.minimum_train_improvement,
            maximum_side_success_drop=args.maximum_side_success_drop,
        )
        if len(development) >= 2
        else {
            "passed": False,
            "checks": {"gradient_audit_passed": gradient["passed"]},
            "reason": "learning probe skipped or did not reach a development checkpoint",
        }
    )
    final: dict[str, Any] = {}
    candidate_path = args.output_dir / "candidate.pt"
    if decision["passed"]:
        final["live_sensors"] = strict_development_evaluation(
            controller,
            episodes=args.final_episodes,
            seconds=args.evaluation_seconds,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        final["frozen_first_frame"] = evaluate_gate(
            controller,
            episodes=args.final_episodes,
            seconds=args.evaluation_seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
            frozen_visual=True,
        )
        final["constant_1g_accelerometer"] = evaluate_gate(
            controller,
            episodes=args.final_episodes,
            seconds=args.evaluation_seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
            frozen_acceleration=True,
        )
        save_candidate(
            candidate_path,
            checkpoint,
            controller,
            {
                "kind": "bounded_full_horizon_trajectory_probe",
                "updates": training[-1]["update"],
                "decision": decision,
                "seed": args.seed,
                "dev_seed": args.dev_seed,
            },
        )
    goal_passed = bool(
        final.get("live_sensors", {}).get("goal_pass", False)
        and final.get("frozen_first_frame", {}).get("success_rate", 1.0) <= 0.05
    )
    report = {
        "experiment": "bounded full-horizon recurrent trajectory-gradient probe",
        "claim_scope": (
            "Teacher trajectories and complete-horizon autodiff exist during training only. "
            "The runtime actor receives FPV, roll/pitch, instantaneous body specific force, "
            "measured foreleg-stick position when mapped, and persistent connectome state. "
            "No external history, estimator, clock, planner, or state machine is added."
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
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "full_horizon_bptt": True,
            "mid_rollout_parameter_updates": False,
            "fixed_transmitter_signs": True,
            "fixed_topology": True,
            "fixed_mechanics": True,
            "trainable_parameters": [
                "all edge magnitudes",
                "all neuron biases",
                "all time constants",
            ],
            "weight_decay": 0.0,
        },
        "gradient_audit": gradient,
        "learning_probe_run": learning_probe_run,
        "training": training,
        "development": development,
        "go_no_go": decision,
        "candidate_checkpoint": stable_path(candidate_path) if decision["passed"] else None,
        "final": final,
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
                "gradient_audit_passed": gradient["passed"],
                "go_no_go_passed": decision["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if decision["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
