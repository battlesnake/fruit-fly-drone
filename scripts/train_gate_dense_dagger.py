#!/usr/bin/env python3
"""Train the native fly on dense successful trajectories, then refresh with DAgger."""

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

from audit_gate_analytic_teachers import teacher_rc_for_mode  # noqa: E402
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
)
from search_gate_motor_interface_es import (  # noqa: E402
    clone_state,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    _episode_summary,
    compact_flight,
    controller_parameter_sha256,
    controller_step,
    evaluate_controller,
    load_frozen_controller,
    parameter_change_summary,
    parameter_snapshot,
    restore_parameters,
)
from train_gate_recurrent_ppo import initialize_outcomes, update_outcomes  # noqa: E402

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)


@dataclass
class DenseTrajectories:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    targets: Tensor
    executed_motor: Tensor
    valid: Tensor
    mass_scale: Tensor
    codes: Tensor
    teacher_driven: bool
    behavior_parameter_sha256: str
    summary: dict[str, Any]


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
        "--teacher-spec",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-analytic-teacher-reserve-v1" / "candidate.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "dense-dagger-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--block-updates", type=int, default=100)
    parser.add_argument("--collection-episodes", type=int, default=64)
    parser.add_argument("--collection-seconds", type=float, default=12.0)
    parser.add_argument("--teacher-takeover-seconds", type=float, default=0.50)
    parser.add_argument("--window-steps", type=int, default=100)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument(
        "--gradient-check-scales",
        type=float,
        nargs=3,
        default=(1.0e-3, 3.0e-4, 1.0e-4),
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    parser.add_argument("--action-scale-floor", type=float, default=0.01)
    parser.add_argument("--regularization-weight", type=float, default=1.0e-6)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-seconds", type=float, default=12.0)
    parser.add_argument("--midpoint-update", type=int, default=200)
    parser.add_argument("--midpoint-minimum-improvement", type=float, default=0.05)
    parser.add_argument("--midpoint-maximum-mass-drop", type=float, default=0.05)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--final-heavy-margin", type=float, default=0.02)
    parser.add_argument("--expert-seed", type=int, default=1_021_031)
    parser.add_argument("--student-seed", type=int, default=1_022_031)
    parser.add_argument("--validation-seed", type=int, default=1_023_031)
    parser.add_argument("--final-seed", type=int, default=1_024_031)
    parser.add_argument("--optimization-seed", type=int, default=1_025_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> tuple[int, int]:
    for path in (args.graph, args.checkpoint, args.teacher_spec):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.updates != 400 or args.block_updates != 100:
        raise SystemExit("the bounded protocol requires four 100-update blocks")
    if args.midpoint_update != 200 or args.validation_interval != 50:
        raise SystemExit("the fixed midpoint and validation cadence are 200 and 50 updates")
    if args.collection_episodes != 64 or args.validation_episodes != 256:
        raise SystemExit("the fixed collection and development sizes are 64 and 256 episodes")
    if args.final_episodes != 1024:
        raise SystemExit("the fresh final protocol requires 1,024 episodes")
    if args.batch_pairs != 8 or args.window_steps != 100:
        raise SystemExit("the dense replay batch is eight pairs over a 100-step window")
    positive = (
        args.collection_seconds,
        args.teacher_takeover_seconds,
        args.learning_rate,
        args.time_constant_learning_rate,
        args.gradient_norm_cap,
        *args.gradient_check_scales,
        args.gradient_check_tolerance,
        args.action_scale_floor,
        args.regularization_weight,
        args.validation_seconds,
        args.midpoint_minimum_improvement,
        args.midpoint_maximum_mass_drop,
        args.final_light_improvement,
        args.final_heavy_margin,
    )
    if min(positive) <= 0.0:
        raise SystemExit("training times, rates, scales, and thresholds must be positive")
    if sorted(args.gradient_check_scales, reverse=True) != list(args.gradient_check_scales):
        raise SystemExit("gradient-check scales must be strictly ordered from largest to smallest")
    steps = round(args.collection_seconds / dt)
    takeover_step = round(args.teacher_takeover_seconds / dt)
    if steps != 1200 or takeover_step != 50:
        raise SystemExit("the frozen dense protocol requires 1,200 steps and takeover at step 50")
    if args.window_steps > steps - takeover_step:
        raise SystemExit("the dense window does not fit after teacher takeover")
    return steps, takeover_step


@torch.no_grad()
def collect_dense_trajectories(
    behavior: ConnectomeController,
    source: ConnectomeController,
    cases: Any,
    *,
    teacher_mode: str,
    teacher_driven: bool,
    steps: int,
    takeover_step: int,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> DenseTrajectories:
    """Record source→teacher successes or labelled current-student histories."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    behavior_neural = behavior.initial_state(episodes, device=device, dtype=torch.float32)
    source_neural = source.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    images = []
    roll_pitch = []
    specific_force = []
    targets = []
    executed_motor = []
    valid = []
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        behavior_motor, behavior_neural = controller_step(
            behavior,
            image,
            state.euler[:, :2],
            behavior_neural,
            state.specific_force,
            stick_state.position,
        )
        source_motor, source_neural = controller_step(
            source,
            image,
            state.euler[:, :2],
            source_neural,
            state.specific_force,
            stick_state.position,
        )
        teacher_rc = teacher_rc_for_mode(
            teacher_mode,
            source,
            state,
            cases.gate,
            cases.mass_scale,
            hover_config,
        )
        teacher_motor = motor_target_for_rc(teacher_rc, hover_config)
        target = source_motor if step < takeover_step else teacher_motor
        motor = target if teacher_driven else behavior_motor
        # Replay must see exactly the observations that drove collection.  FP16 storage
        # caused small but accumulating changes in the native recurrent state.
        images.append(image)
        roll_pitch.append(state.euler[:, :2])
        specific_force.append(state.specific_force)
        targets.append(target)
        executed_motor.append(motor)
        valid.append(active.clone())
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
        post_pass_complete = (outcomes["pass_step"] >= 0) & (
            step + 1 >= outcomes["pass_step"] + round(1.0 / hover_config.dt)
        )
        active &= ~(
            outcomes["collision"]
            | outcomes["missed"]
            | outcomes["recontact"]
            | post_pass_complete
            | (outcomes["maximum_tilt"] > math.radians(40.0))
        )
    summary, _ = _episode_summary(outcomes, cases, steps=steps, hover_config=hover_config)
    summary["teacher_driven"] = teacher_driven
    summary["teacher_takeover_seconds"] = takeover_step * hover_config.dt
    return DenseTrajectories(
        images=torch.stack(images),
        roll_pitch=torch.stack(roll_pitch),
        specific_force=torch.stack(specific_force),
        targets=torch.stack(targets),
        executed_motor=torch.stack(executed_motor),
        valid=torch.stack(valid),
        mass_scale=cases.mass_scale,
        codes=cases.stratum_code,
        teacher_driven=teacher_driven,
        behavior_parameter_sha256=controller_parameter_sha256(behavior),
        summary=summary,
    )


def fixed_axis_scales(expert: DenseTrajectories, *, floor: float) -> Tensor:
    values = expert.targets[expert.valid]
    if not values.numel():
        raise RuntimeError("expert collection has no valid target")
    scales = values.square().mean(dim=0).sqrt().clamp_min(floor)
    if not bool(torch.isfinite(scales).all()):
        raise RuntimeError("nonfinite dense-action scale")
    return scales


def choose_pair_episodes(
    trajectories: DenseTrajectories,
    *,
    pairs: int,
    generator: torch.Generator,
) -> Tensor:
    available = trajectories.mass_scale.numel() // 2
    order = torch.randperm(available, generator=generator)[:pairs]
    return torch.stack((2 * order, 2 * order + 1), dim=1).flatten().to(trajectories.images.device)


def choose_dense_batch(
    trajectories: DenseTrajectories,
    *,
    window_steps: int,
    region: str,
    pairs: int,
    generator: torch.Generator,
    attempts_per_region: int = 64,
) -> tuple[int, Tensor, str]:
    """Jointly select complete mass pairs and a window with usable samples.

    Student rollouts can fail before the requested late region.  In that case the
    fallback is explicit and deterministic in kind: crossing/post-crossing falls
    back through approach to launch, while approach falls back only to launch.
    """

    fallbacks = {
        "launch": ("launch",),
        "approach": ("approach", "launch"),
        "crossing_post": ("crossing_post", "approach", "launch"),
    }
    if region not in fallbacks:
        raise ValueError(f"unknown dense-window region: {region}")
    if trajectories.mass_scale.numel() // 2 < pairs:
        raise ValueError("dense replay does not contain enough complete mass pairs")
    maximum_start = trajectories.valid.shape[0] - window_steps
    nominal_bounds = {
        "launch": (0, 50),
        "approach": (50, 400),
        "crossing_post": (400, maximum_start),
    }
    for effective_region in fallbacks[region]:
        low, nominal_high = nominal_bounds[effective_region]
        high = min(nominal_high, maximum_start)
        if low > high:
            continue
        for _ in range(attempts_per_region):
            episodes = choose_pair_episodes(trajectories, pairs=pairs, generator=generator)
            selected_valid = trajectories.valid[:, episodes]
            window_valid_fraction = (
                selected_valid.unfold(0, window_steps, 1).float().mean(dim=(1, 2))
            )
            candidates = (
                torch.nonzero(
                    window_valid_fraction[low : high + 1] >= 0.10,
                    as_tuple=False,
                ).flatten()
                + low
            )
            if not len(candidates):
                continue
            choice = torch.randint(len(candidates), (1,), generator=generator).item()
            return int(candidates[choice]), episodes, effective_region
    raise RuntimeError(
        f"no {region} dense replay batch retains at least ten percent valid samples "
        f"after {attempts_per_region} pair draws per fallback region"
    )


@torch.no_grad()
def controller_replay_parity(
    controller: ConnectomeController,
    trajectories: DenseTrajectories,
    reference: Tensor,
    *,
    steps: int,
) -> dict[str, Any]:
    """Replay stored FP32 sensors and compare motor output with collection."""

    episodes = trajectories.images.shape[1]
    neural = controller.initial_state(
        episodes, device=trajectories.images.device, dtype=torch.float32
    )
    predictions = []
    for step in range(steps):
        motor, neural = controller_step(
            controller,
            trajectories.images[step],
            trajectories.roll_pitch[step],
            neural,
            trajectories.specific_force[step],
            torch.zeros(episodes, 4, device=neural.device),
        )
        predictions.append(motor)
    replayed = torch.stack(predictions)
    expected = reference[:steps]
    difference = (replayed - expected).abs()
    return {
        "steps": steps,
        "episodes": episodes,
        "stored_image_dtype": str(trajectories.images.dtype),
        "exact": torch.equal(replayed, expected),
        "maximum_absolute_error": float(difference.max()),
        "last_step_maximum_absolute_error": float(difference[-1].max()),
    }


@torch.no_grad()
def replay_burn_in_state(
    controller: ConnectomeController,
    trajectories: DenseTrajectories,
    episodes: Tensor,
    *,
    start: int,
) -> Tensor:
    """Recompute native state from the stored sensor prefix under current weights."""

    neural = controller.initial_state(
        len(episodes), device=trajectories.images.device, dtype=torch.float32
    )
    zero_sticks = torch.zeros(len(episodes), 4, device=neural.device)
    for step in range(start):
        _, neural = controller_step(
            controller,
            trajectories.images[step, episodes],
            trajectories.roll_pitch[step, episodes],
            neural,
            trajectories.specific_force[step, episodes],
            zero_sticks,
        )
    return neural.detach()


def controller_regularization(
    controller: ConnectomeController, source_parameters: dict[str, Tensor]
) -> Tensor:
    edge_scale = source_parameters["edge_magnitude"].clamp_min(0.05)
    return (
        ((controller.edge_magnitude - source_parameters["edge_magnitude"]) / edge_scale)
        .square()
        .mean()
        + (controller.bias - source_parameters["bias"]).square().mean()
        + (controller.raw_time_constant - source_parameters["raw_time_constant"]).square().mean()
    )


def dense_window_loss(
    controller: ConnectomeController,
    trajectories: DenseTrajectories,
    source_parameters: dict[str, Tensor],
    axis_scales: Tensor,
    *,
    start: int,
    window_steps: int,
    episodes: Tensor,
    regularization_weight: float,
    burn_in_state: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Recompute current native state, then differentiate only through one dense window."""

    if burn_in_state is None:
        neural = replay_burn_in_state(controller, trajectories, episodes, start=start)
    else:
        expected_shape = (len(episodes), controller.n_nodes)
        if burn_in_state.shape != expected_shape:
            raise ValueError(
                f"burn-in state has shape {tuple(burn_in_state.shape)}, expected {expected_shape}"
            )
        neural = burn_in_state.detach()
    zero_sticks = torch.zeros(len(episodes), 4, device=neural.device)
    predictions = []
    for step in range(start, start + window_steps):
        motor, neural = controller_step(
            controller,
            trajectories.images[step, episodes],
            trajectories.roll_pitch[step, episodes],
            neural,
            trajectories.specific_force[step, episodes],
            zero_sticks,
        )
        predictions.append(motor)
    prediction = torch.stack(predictions)
    target = trajectories.targets[start : start + window_steps, episodes]
    valid = trajectories.valid[start : start + window_steps, episodes]
    valid_weight = valid.to(dtype=prediction.dtype).unsqueeze(-1)
    valid_samples = valid_weight.sum()
    if not bool(valid_samples > 0):
        raise RuntimeError("selected dense replay window contains no valid samples")
    absolute_error = (prediction - target).abs()
    axis_normalized_mae = ((absolute_error / axis_scales) * valid_weight).sum(
        dim=(0, 1)
    ) / valid_samples
    axis_mae = (absolute_error * valid_weight).sum(dim=(0, 1)) / valid_samples
    imitation = axis_normalized_mae.mean()
    regularization = controller_regularization(controller, source_parameters)
    loss = imitation + regularization_weight * regularization
    return loss, {
        "total_loss": float(loss.detach()),
        "imitation_loss": float(imitation.detach()),
        "axis_normalized_mae": axis_normalized_mae.detach().cpu().tolist(),
        "axis_mae": axis_mae.detach().cpu().tolist(),
        "regularization": float(regularization.detach()),
        "valid_fraction": float(valid.float().mean()),
        "window_start": start,
        "window_end": start + window_steps,
        "teacher_driven_source": trajectories.teacher_driven,
    }


def dense_gradient_audit(
    controller: ConnectomeController,
    trajectories: DenseTrajectories,
    source_parameters: dict[str, Tensor],
    axis_scales: Tensor,
    *,
    start: int,
    window_steps: int,
    episodes: Tensor,
    regularization_weight: float,
    perturbation_scales: tuple[float, ...] | list[float],
    tolerance: float,
) -> dict[str, Any]:
    """Check the exact 100-step truncated gradient against central differences.

    The current-parameter burn-in state is computed once and deliberately held fixed
    for the analytic and every perturbed objective.  The audit therefore measures the
    same truncated window that the optimizer differentiates, not a moving-prefix loss.
    """

    parameters = {
        "edge_magnitude": controller.edge_magnitude,
        "bias": controller.bias,
        "raw_time_constant": controller.raw_time_constant,
    }
    baseline = parameter_snapshot(controller)
    fixed_burn_in = replay_burn_in_state(controller, trajectories, episodes, start=start)

    def objective() -> Tensor:
        return dense_window_loss(
            controller,
            trajectories,
            source_parameters,
            axis_scales,
            start=start,
            window_steps=window_steps,
            episodes=episodes,
            regularization_weight=regularization_weight,
            burn_in_state=fixed_burn_in,
        )[0]

    repeats = []
    with torch.no_grad():
        for _ in range(3):
            repeats.append(float(objective()))
    repeat_noise = max(repeats) - min(repeats)
    controller.zero_grad(set_to_none=True)
    objective().backward()
    families: dict[str, Any] = {}
    try:
        for name, parameter in parameters.items():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing or nonfinite dense-window gradient for {name}")
            direction = parameter.grad.detach().sign()
            if name == "edge_magnitude":
                direction[(parameter.detach() <= 0.01) | (parameter.detach() >= 7.99)] = 0.0
            analytic = float((parameter.grad * direction).sum())
            if not math.isfinite(analytic) or analytic <= 0.0:
                raise RuntimeError(f"zero or nonfinite dense-window direction for {name}")
            checks = []
            for epsilon in perturbation_scales:
                with torch.no_grad():
                    parameter.copy_(baseline[name] + epsilon * direction)
                    plus = float(objective())
                    parameter.copy_(baseline[name] - epsilon * direction)
                    minus = float(objective())
                    parameter.copy_(baseline[name])
                estimate = (plus - minus) / (2.0 * epsilon)
                relative_error = abs(estimate - analytic) / max(
                    abs(estimate), abs(analytic), 1.0e-10
                )
                checks.append(
                    {
                        "epsilon": epsilon,
                        "plus_loss": plus,
                        "minus_loss": minus,
                        "central_difference": estimate,
                        "relative_error": relative_error,
                        "matching_sign": estimate * analytic > 0.0,
                        "measurable_above_repeat_noise": abs(plus - minus)
                        > max(10.0 * repeat_noise, 1.0e-10),
                    }
                )
            adjacent = [
                left["matching_sign"]
                and right["matching_sign"]
                and left["measurable_above_repeat_noise"]
                and right["measurable_above_repeat_noise"]
                and left["relative_error"] <= tolerance
                and right["relative_error"] <= tolerance
                for left, right in zip(checks[:-1], checks[1:], strict=True)
            ]
            families[name] = {
                "gradient_l2_norm": float(parameter.grad.norm()),
                "analytic_directional_derivative": analytic,
                "checks": checks,
                "passed": any(adjacent),
            }
            if not families[name]["passed"]:
                raise RuntimeError(f"dense-window gradient audit failed for {name}: {checks}")
    finally:
        restore_parameters(controller, baseline)
        controller.zero_grad(set_to_none=True)
    return {
        "window_start": start,
        "window_steps": window_steps,
        "complete_window_in_autograd": True,
        "burn_in_recomputed_under_current_parameters": True,
        "burn_in_held_fixed_across_perturbations": True,
        "unchanged_loss_repeats": repeats,
        "unchanged_loss_range": repeat_noise,
        "families": families,
        "passed": all(item["passed"] for item in families.values()),
    }


def validation_safe_and_improved(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    improvement: float,
    mass_drop: float,
) -> bool:
    return bool(
        candidate["success_rate"] >= baseline["success_rate"] + improvement
        and candidate["light_success_rate"] >= baseline["light_success_rate"] - mass_drop
        and candidate["heavy_success_rate"] >= baseline["heavy_success_rate"] - mass_drop
    )


def save_candidate_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    teacher_spec_path: Path,
    controller: ConnectomeController,
    promotion: dict[str, Any],
) -> None:
    saved = copy.deepcopy(checkpoint)
    saved["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    saved["source_checkpoint_sha256"] = file_sha256(checkpoint_path)
    saved["dense_reserve_dagger"] = {
        "method": "dense source-to-reserve imitation with three DAgger refreshes",
        "teacher_spec_sha256": file_sha256(teacher_spec_path),
        "fixed_topology": True,
        "fixed_transmitter_signs": True,
        "all_native_parameter_families_trainable": True,
        "mass_actor_input": False,
        "engineered_history_features": False,
        "parameter_vector_sha256": controller_parameter_sha256(controller),
        "promotion": promotion,
    }
    torch.save(saved, path)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source, source_checkpoint, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.checkpoint, device
    )
    steps, takeover_step = validate_args(args, hover_config.dt)
    if not source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("unexpected deployed sensor contract")
    teacher_spec = json.loads(args.teacher_spec.read_text())
    if (
        teacher_spec.get("kind") != "privileged_analytic_gate_teacher"
        or teacher_spec.get("teacher_mode") != "visual_accelerometer_reserve"
        or teacher_spec.get("uses_exact_simulator_mass")
        or not teacher_spec.get("preflight_passed")
        or teacher_spec.get("checkpoint_sha256") != file_sha256(args.checkpoint)
        or teacher_spec.get("takeover_seconds") != args.teacher_takeover_seconds
    ):
        raise SystemExit("dense DAgger requires the validated mass-free reserve teacher")
    teacher_mode = teacher_spec["teacher_mode"]
    student = copy.deepcopy(source).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    source_parameters = parameter_snapshot(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate.pt"
    vector_path = args.output_dir / "selected-vector.json"
    if candidate_path.exists() or vector_path.exists():
        raise SystemExit("output directory contains a stale candidate or selected vector")
    seed_everything(args.optimization_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    expert_cases = diverse_matched_cases(
        args.collection_episodes,
        seed=args.expert_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    normal_teacher = teacher_rc_for_mode(
        teacher_mode,
        source,
        expert_cases.state,
        expert_cases.gate,
        expert_cases.mass_scale,
        hover_config,
    )
    swapped_teacher = teacher_rc_for_mode(
        teacher_mode,
        source,
        expert_cases.state,
        expert_cases.gate,
        expert_cases.mass_scale.flip(0),
        hover_config,
    )
    teacher_mass_argument_invariant = torch.equal(normal_teacher, swapped_teacher)
    if not teacher_mass_argument_invariant:
        raise RuntimeError("reserve teacher depends on the mass argument")
    expert = collect_dense_trajectories(
        source,
        source,
        expert_cases,
        teacher_mode=teacher_mode,
        teacher_driven=True,
        steps=steps,
        takeover_step=takeover_step,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    expert_rates = {
        key: expert.summary[key]
        for key in (
            "success_rate",
            "light_success_rate",
            "heavy_success_rate",
            "negative_lateral_success_rate",
            "positive_lateral_success_rate",
            "negative_obliquity_success_rate",
            "positive_obliquity_success_rate",
        )
    }
    expert_preflight_passed = min(expert_rates.values()) >= 0.90
    print(
        json.dumps(
            {
                "phase": "dense_expert_preflight",
                "passed": expert_preflight_passed,
                "success_rates": expert_rates,
            }
        ),
        flush=True,
    )
    if not expert_preflight_passed:
        raise RuntimeError("dense expert collection failed its fixed success gate")
    if not torch.equal(expert.targets, expert.executed_motor):
        raise RuntimeError("teacher-driven dense labels do not equal executed actions")
    expert_replay_parity = controller_replay_parity(
        source,
        expert,
        expert.targets,
        steps=takeover_step,
    )
    if not expert_replay_parity["exact"]:
        raise RuntimeError(f"expert source-prefix replay mismatch: {expert_replay_parity}")
    axis_scales = fixed_axis_scales(expert, floor=args.action_scale_floor)

    validation_cases = diverse_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    baseline_validation, baseline_validation_tensors = evaluate_controller(
        source,
        validation_cases,
        seconds=args.validation_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    print(
        json.dumps({"phase": "validation", "update": 0, **compact_flight(baseline_validation)}),
        flush=True,
    )

    generator = torch.Generator(device="cpu").manual_seed(args.optimization_seed)
    fixed_start = 50
    fixed_episodes = torch.arange(2 * args.batch_pairs, device=device)
    gradient_audit = dense_gradient_audit(
        student,
        expert,
        source_parameters,
        axis_scales,
        start=fixed_start,
        window_steps=args.window_steps,
        episodes=fixed_episodes,
        regularization_weight=args.regularization_weight,
        perturbation_scales=args.gradient_check_scales,
        tolerance=args.gradient_check_tolerance,
    )

    optimizer = torch.optim.Adam(
        (
            {"params": [student.edge_magnitude, student.bias], "lr": args.learning_rate},
            {"params": [student.raw_time_constant], "lr": args.time_constant_learning_rate},
        )
    )
    history = []
    collections = [
        {
            "block": 0,
            "kind": "expert",
            "summary": expert.summary,
            "source_prefix_replay_parity": expert_replay_parity,
        }
    ]
    archive: dict[int, dict[str, Any]] = {
        0: {
            "parameters": parameter_snapshot(source, cpu=True),
            "summary": baseline_validation,
            "tensors": baseline_validation_tensors,
            "safe": True,
            "midpoint_qualified": False,
        }
    }
    stopped_at_midpoint = False
    updates_completed = 0
    for block in range(args.updates // args.block_updates):
        student_replay = None
        if block:
            cases = diverse_matched_cases(
                args.collection_episodes,
                seed=args.student_seed + block,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                extreme_fraction=0.5,
            )
            student_replay = collect_dense_trajectories(
                student,
                source,
                cases,
                teacher_mode=teacher_mode,
                teacher_driven=False,
                steps=steps,
                takeover_step=takeover_step,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            student_replay_parity = controller_replay_parity(
                student,
                student_replay,
                student_replay.executed_motor,
                steps=steps,
            )
            if not student_replay_parity["exact"]:
                raise RuntimeError(
                    f"student collection replay mismatch in block {block}: {student_replay_parity}"
                )
            collections.append(
                {
                    "block": block,
                    "kind": "student",
                    "summary": student_replay.summary,
                    "behavior_replay_parity": student_replay_parity,
                }
            )
        source_counts = {"expert": 0, "student": 0}
        for local_update in range(1, args.block_updates + 1):
            update = block * args.block_updates + local_update
            replay = expert if block == 0 or local_update % 2 else student_replay
            if replay is None:
                raise RuntimeError("missing student replay in a mixed DAgger block")
            source_counts["expert" if replay.teacher_driven else "student"] += 1
            region = ("launch", "approach", "crossing_post")[(local_update - 1) % 3]
            start, episodes, effective_region = choose_dense_batch(
                replay,
                window_steps=args.window_steps,
                region=region,
                pairs=args.batch_pairs,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, details = dense_window_loss(
                student,
                replay,
                source_parameters,
                axis_scales,
                start=start,
                window_steps=args.window_steps,
                episodes=episodes,
                regularization_weight=args.regularization_weight,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite dense loss at update {update}")
            loss.backward()
            raw_gradient_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in student.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
            )
            optimizer.step()
            student.project_parameters()
            updates_completed = update
            history.append(
                {
                    "update": update,
                    "block": block,
                    "source": "expert" if replay.teacher_driven else "student",
                    "requested_window_region": region,
                    "effective_window_region": effective_region,
                    "window_region_fallback": effective_region != region,
                    "raw_gradient_norm": float(raw_gradient_norm),
                    **details,
                }
            )
            if update % args.validation_interval:
                continue
            summary, tensors = evaluate_controller(
                student,
                validation_cases,
                seconds=args.validation_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            safe = bool(
                summary["light_success_rate"]
                >= baseline_validation["light_success_rate"] - args.midpoint_maximum_mass_drop
                and summary["heavy_success_rate"]
                >= baseline_validation["heavy_success_rate"] - args.midpoint_maximum_mass_drop
            )
            qualified = validation_safe_and_improved(
                baseline_validation,
                summary,
                improvement=args.midpoint_minimum_improvement,
                mass_drop=args.midpoint_maximum_mass_drop,
            )
            archive[update] = {
                "parameters": parameter_snapshot(student, cpu=True),
                "summary": summary,
                "tensors": tensors,
                "safe": safe,
                "midpoint_qualified": qualified,
            }
            print(
                json.dumps(
                    {
                        "phase": "validation",
                        "update": update,
                        "safe": safe,
                        "midpoint_qualified": qualified,
                        **compact_flight(summary),
                    }
                ),
                flush=True,
            )
            if update == args.midpoint_update and not any(
                item["midpoint_qualified"] for item in archive.values()
            ):
                stopped_at_midpoint = True
                break
        if stopped_at_midpoint:
            break
        safe_updates = [update for update, item in archive.items() if item["safe"]]
        selected_update = max(
            safe_updates,
            key=lambda update: (
                archive[update]["summary"]["success_rate"],
                min(
                    archive[update]["summary"]["light_success_rate"],
                    archive[update]["summary"]["heavy_success_rate"],
                ),
                -archive[update]["summary"]["ring_collision_rate"],
            ),
        )
        restore_parameters(student, archive[selected_update]["parameters"])
        optimizer = torch.optim.Adam(
            (
                {"params": [student.edge_magnitude, student.bias], "lr": args.learning_rate},
                {
                    "params": [student.raw_time_constant],
                    "lr": args.time_constant_learning_rate,
                },
            )
        )

    safe_updates = [update for update, item in archive.items() if item["safe"]]
    selected_update = max(
        safe_updates,
        key=lambda update: (
            archive[update]["summary"]["success_rate"],
            min(
                archive[update]["summary"]["light_success_rate"],
                archive[update]["summary"]["heavy_success_rate"],
            ),
            -archive[update]["summary"]["ring_collision_rate"],
        ),
    )
    restore_parameters(student, archive[selected_update]["parameters"])
    final = None
    paired_final = None
    promotion = {
        "checks": {
            "completed_all_updates": updates_completed == args.updates,
            "midpoint_progress_gate": not stopped_at_midpoint,
        },
        "passed": False,
    }
    goal_checks = None
    goal_passed = False
    if updates_completed == args.updates:
        final_cases = diverse_matched_cases(
            args.final_episodes,
            seed=args.final_seed,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        final = {}
        outcomes = {}
        for name, controller, controls in (
            ("reference", source, {}),
            ("candidate", student, {}),
            ("candidate_frozen_first_frame", student, {"frozen_visual": True}),
            ("candidate_constant_1g", student, {"constant_acceleration": True}),
            (
                "candidate_pair_swapped_acceleration",
                student,
                {"pair_swapped_acceleration": True},
            ),
        ):
            summary, tensors = evaluate_controller(
                controller,
                final_cases,
                seconds=args.validation_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
                **controls,
            )
            final[name] = summary
            outcomes[name] = tensors
            print(
                json.dumps({"phase": "final", "name": name, **compact_flight(summary)}),
                flush=True,
            )
        reference_success = outcomes["reference"]["success"]
        candidate_success = outcomes["candidate"]["success"]
        codes = outcomes["reference"]["codes"]
        light = ~codes.bitwise_and(1).bool()
        heavy = ~light
        paired_overall = paired_clustered_confidence_interval(reference_success, candidate_success)
        paired_light = paired_confidence_interval(
            reference_success[light], candidate_success[light]
        )
        paired_heavy = paired_confidence_interval(
            reference_success[heavy], candidate_success[heavy]
        )
        paired_final = {
            "overall_success_difference": paired_overall,
            "light_success_difference": paired_light,
            "heavy_success_difference": paired_heavy,
        }
        promotion_checks = {
            "completed_all_updates": True,
            "midpoint_progress_gate": True,
            "light_improvement_at_least_threshold": (
                final["candidate"]["light_success_rate"]
                >= final["reference"]["light_success_rate"] + args.final_light_improvement
            ),
            "light_paired_confidence_interval_excludes_zero": (
                paired_light["confidence_95"][0] > 0.0
            ),
            "heavy_nondegradation_within_margin": (
                final["candidate"]["heavy_success_rate"]
                >= final["reference"]["heavy_success_rate"] - args.final_heavy_margin
            ),
            "heavy_paired_noninferiority_interval_within_margin": (
                paired_heavy["confidence_95"][0] >= -args.final_heavy_margin
            ),
            "overall_success_improves": (
                final["candidate"]["success_rate"] > final["reference"]["success_rate"]
            ),
            "overall_paired_confidence_interval_excludes_zero": (
                paired_overall["confidence_95"][0] > 0.0
            ),
            "frozen_first_frame_success_at_most_five_percent": (
                final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
            ),
        }
        promotion = {"checks": promotion_checks, "passed": all(promotion_checks.values())}
        if promotion["passed"]:
            save_candidate_checkpoint(
                candidate_path,
                source_checkpoint,
                args.checkpoint,
                args.teacher_spec,
                student,
                promotion,
            )
        goal_checks = {
            key: final["candidate"][key] >= 0.90
            for key in (
                "success_rate",
                "light_success_rate",
                "heavy_success_rate",
                "negative_lateral_success_rate",
                "positive_lateral_success_rate",
                "negative_obliquity_success_rate",
                "positive_obliquity_success_rate",
            )
        }
        goal_passed = bool(
            promotion["passed"]
            and all(goal_checks.values())
            and final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
        )

    vector = {
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_update": selected_update,
        "edge_magnitude": student.edge_magnitude.detach().cpu().tolist(),
        "bias": student.bias.detach().cpu().tolist(),
        "raw_time_constant": student.raw_time_constant.detach().cpu().tolist(),
    }
    vector_path.write_text(json.dumps(vector, indent=2, sort_keys=True) + "\n")
    archive_path = args.output_dir / "archive.pt"
    torch.save(
        {
            update: {
                "parameters": item["parameters"],
                "summary": item["summary"],
                "safe": item["safe"],
                "midpoint_qualified": item["midpoint_qualified"],
            }
            for update, item in archive.items()
        },
        archive_path,
    )
    report = {
        "method": "dense successful-trajectory imitation with three DAgger refreshes",
        "claim_scope": (
            "Training uses teacher labels, a fixed takeover schedule, replay timestamps, and "
            "privileged relative gate geometry. Deployment uses only current FPV, roll/pitch, "
            "body-Z specific force, and native connectome state."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "teacher_spec": stable_path(args.teacher_spec),
        "teacher_spec_sha256": file_sha256(args.teacher_spec),
        "teacher_mode": teacher_mode,
        "teacher_mass_argument_invariant": teacher_mass_argument_invariant,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "block_schedule": [
                "100% expert",
                "50/50 expert/student",
                "50/50 expert/student",
                "50/50 expert/student",
            ],
            "student_refreshes": 3,
            "current_native_state_recomputed_from_prefix": True,
            "stored_native_state_used": False,
            "gradient_window_seconds": args.window_steps * hover_config.dt,
            "imitation_objective": "equal-weight per-axis RMS-normalized MAE/L1",
            "teacher_drives_deployed_actor": False,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
            "all_native_parameter_families_trainable": True,
            "fixed_topology_and_transmitter_signs": True,
            "selection_uses_flight_not_action_fidelity": True,
        },
        "axis_action_rms_scales": axis_scales.detach().cpu().tolist(),
        "expert_collection": expert.summary,
        "expert_preflight_rates": expert_rates,
        "expert_preflight_passed": expert_preflight_passed,
        "collections": collections,
        "expert_source_prefix_replay_parity": expert_replay_parity,
        "gradient_audit": gradient_audit,
        "baseline_validation": baseline_validation,
        "history": history,
        "validation_selections": {
            str(update): {
                "summary": item["summary"],
                "safe": item["safe"],
                "midpoint_qualified": item["midpoint_qualified"],
            }
            for update, item in archive.items()
        },
        "midpoint_progress_passed": not stopped_at_midpoint,
        "stopped_at_midpoint": stopped_at_midpoint,
        "updates_completed": updates_completed,
        "selected_update": selected_update,
        "selected_parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_parameter_change_from_source": parameter_change_summary(
            student, source_parameters
        ),
        "selected_vector": stable_path(vector_path),
        "selected_vector_sha256": file_sha256(vector_path),
        "archive": stable_path(archive_path),
        "archive_sha256": file_sha256(archive_path),
        "final": final,
        "paired_final": paired_final,
        "promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": (
            file_sha256(candidate_path) if promotion["passed"] else None
        ),
        "goal_threshold_checks": goal_checks,
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
                "updates_completed": updates_completed,
                "midpoint_progress_passed": not stopped_at_midpoint,
                "selected_update": selected_update,
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
