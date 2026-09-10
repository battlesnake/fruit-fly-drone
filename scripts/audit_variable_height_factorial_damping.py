#!/usr/bin/env python3
"""Preflight damping descent after separating matched height and velocity components."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_damping_routes as shallow  # noqa: E402
import audit_variable_height_damping_routes_v2 as bounded  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import audit_variable_height_upstream_damping_route as upstream  # noqa: E402
import train_variable_height_common_anchor_ablation as ablation  # noqa: E402
import train_variable_height_damping_routes as native_train  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402
import train_variable_height_upstream_damping_route as upstream_train  # noqa: E402

from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    HoverConfig,
    motor_target_for_rc,
    teacher_rc,
)
from flydrone.visual_hover import (  # noqa: E402
    render_visual_hover_scene,
    sample_visual_scenes,
)

HEIGHT_ERROR_MAGNITUDES = (0.05, 0.10)
VERTICAL_SPEED_MAGNITUDES = (0.15, 0.30)
APPROACH_STEPS = (15, 18, 22, 25)
PREFIX_STEPS = (5, 25, 50, 75)
DAMPING_MOTOR_SCALE = 0.05
FINITE_DIFFERENCE_SCALE = 0.0625
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE = 1.0e-6
LINEARIZED_HEIGHT_NULL_RELATIVE_LIMIT = 1.0e-3
BRANCH_SIGNS = ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))


@dataclass
class FactorialForward:
    prediction: dict[str, Tensor]
    target: dict[str, Tensor]
    height_error_magnitude: Tensor
    vertical_speed_magnitude: Tensor
    approach_steps: Tensor
    prefix_steps: Tensor
    endpoint_velocity_image_difference_max: float


def parse_args() -> argparse.Namespace:
    parser = shallow.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=REPO_ROOT
        / "runs/variable-height-hover/factorial-damping-route-preflight-001",
        seed=300_947,
        batch_size=4,
    )
    return parser.parse_args()


def balanced_amplitude_grid(
    batch: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return shuffled repeats of the preregistered |height error| x |velocity| grid."""

    combinations = torch.tensor(
        list(itertools.product(HEIGHT_ERROR_MAGNITUDES, VERTICAL_SPEED_MAGNITUDES)),
        device=device,
    )
    repeated = combinations.repeat(math.ceil(batch / len(combinations)), 1)[:batch]
    repeated = repeated[torch.randperm(batch, device=device)]
    return repeated[:, 0], repeated[:, 1]


def factorial_components(values: Tensor) -> dict[str, Tensor]:
    """Hadamard-decompose branch-major values ordered by ``BRANCH_SIGNS``."""

    if values.ndim != 2 or values.shape[0] != len(BRANCH_SIGNS):
        raise ValueError("factorial values must have shape (4, batch)")
    device, dtype = values.device, values.dtype
    height_sign = torch.tensor([item[0] for item in BRANCH_SIGNS], device=device, dtype=dtype)
    velocity_sign = torch.tensor([item[1] for item in BRANCH_SIGNS], device=device, dtype=dtype)
    return {
        "common": values.mean(dim=0),
        "height": (height_sign[:, None] * values).mean(dim=0),
        "damping": (velocity_sign[:, None] * values).mean(dim=0),
        "interaction": (height_sign[:, None] * velocity_sign[:, None] * values).mean(dim=0),
    }


def smooth_return_offset(
    signed_speed: Tensor,
    *,
    step: int,
    total_steps: int,
    approach_steps: Tensor,
    policy_hz: int,
) -> Tensor:
    """A smooth pose-returning history whose endpoint derivative is ``signed_speed``."""

    local_step = (step + 1 - (total_steps - approach_steps)).clamp_min(0)
    active = local_step > 0
    unit = local_step / approach_steps
    duration = approach_steps / float(policy_hz)
    offset = signed_speed * duration * (unit**3 - unit**2)
    offset -= signed_speed.sign() * 0.055 * torch.sin(math.pi * unit).square()
    offset = torch.where(active, offset, torch.zeros_like(offset))
    if step + 1 == total_steps:
        offset = torch.zeros_like(offset)
    return offset


def _factorial_forward(
    student: ConnectomeController,
    prefix_controller: ConnectomeController,
    *,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> FactorialForward:
    responsibility.seed_everything(seed)
    device = student.edge_magnitude.device
    height_error, speed = balanced_amplitude_grid(batch, device=device)
    state, height = responsibility.base_state(batch, device=device, config=config)
    scene = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    prefix_choices = torch.tensor(PREFIX_STEPS, device=device)
    prefix_steps = prefix_choices[torch.arange(batch, device=device) % len(prefix_choices)]
    prefix_steps = prefix_steps[torch.randperm(batch, device=device)]
    with torch.no_grad():
        prefix = shallow.variable_prefix(prefix_controller, state, height, scene, prefix_steps)

    approach_choices = torch.tensor(APPROACH_STEPS, device=device)
    approach_steps = approach_choices[torch.arange(batch, device=device) % len(approach_choices)]
    approach_steps = approach_steps[torch.randperm(batch, device=device)]
    recurrent = torch.cat([prefix.clone() for _ in BRANCH_SIGNS])
    attitude = torch.cat([state.euler[:, :2] for _ in BRANCH_SIGNS])
    markers = [height + height_sign * height_error for height_sign, _ in BRANCH_SIGNS]

    motor = torch.zeros(len(BRANCH_SIGNS) * batch, 4, device=device)
    endpoint_images: list[Tensor] = []
    for step in range(history_steps):
        branch_states = []
        for _, velocity_sign in BRANCH_SIGNS:
            signed_speed = velocity_sign * speed
            offset = smooth_return_offset(
                signed_speed,
                step=step,
                total_steps=history_steps,
                approach_steps=approach_steps,
                policy_hz=policy_hz,
            )
            branch_states.append(responsibility.state_at_height(state, height + offset))
        images = [
            render_visual_hover_scene(branch_state, marker, scene=scene)
            for branch_state, marker in zip(branch_states, markers, strict=True)
        ]
        endpoint_images = images
        motor, recurrent = student(torch.cat(images), attitude, recurrent)

    output = motor.reshape(len(BRANCH_SIGNS), batch, 4)[:, :, 3]
    target_outputs = []
    for (_, velocity_sign), marker in zip(BRANCH_SIGNS, markers, strict=True):
        endpoint = responsibility.state_at_height(
            state,
            height,
            vertical_velocity=velocity_sign * speed,
        )
        target_outputs.append(
            motor_target_for_rc(teacher_rc(endpoint, marker, config), config)[:, 3]
        )
    targets = torch.stack(target_outputs)
    images = torch.stack(endpoint_images)
    velocity_image_difference = max(
        float((images[0] - images[1]).abs().max()),
        float((images[2] - images[3]).abs().max()),
    )
    return FactorialForward(
        prediction=factorial_components(output),
        target=factorial_components(targets),
        height_error_magnitude=height_error,
        vertical_speed_magnitude=speed,
        approach_steps=approach_steps,
        prefix_steps=prefix_steps,
        endpoint_velocity_image_difference_max=velocity_image_difference,
    )


def _component_report(prediction: Tensor, target: Tensor) -> dict[str, Any]:
    error = prediction - target
    target_power = target.square().sum().clamp_min(1.0e-12)
    return {
        "fixed_scale_nrmse": float((error.detach() / DAMPING_MOTOR_SCALE).square().mean().sqrt()),
        "relative_target_nrmse": float(
            error.detach().square().mean().sqrt()
            / target.detach().square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "teacher_aligned_gain": float((prediction.detach() * target.detach()).sum() / target_power),
        "correct_sign_fraction": float(
            ((prediction.detach() * target.detach()) > 1.0e-6).float().mean()
        ),
        "target_rms": float(target.detach().square().mean().sqrt()),
        "prediction_rms": float(prediction.detach().square().mean().sqrt()),
        "target": target.detach().cpu().tolist(),
        "prediction": prediction.detach().cpu().tolist(),
    }


def factorial_terms(
    student: ConnectomeController,
    prefix_controller: ConnectomeController,
    *,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> tuple[Tensor, dict[str, Any]]:
    forward = _factorial_forward(
        student,
        prefix_controller,
        batch=batch,
        seed=seed,
        history_steps=history_steps,
        policy_hz=policy_hz,
        config=config,
    )
    damping_error = forward.prediction["damping"] - forward.target["damping"]
    loss = (damping_error / DAMPING_MOTOR_SCALE).square().mean()
    return loss, factorial_forward_report(forward)


def factorial_forward_report(forward: FactorialForward) -> dict[str, Any]:
    return {
        "damping": _component_report(forward.prediction["damping"], forward.target["damping"]),
        "height": _component_report(forward.prediction["height"], forward.target["height"]),
        "common": _component_report(forward.prediction["common"], forward.target["common"]),
        "interaction": _component_report(
            forward.prediction["interaction"], forward.target["interaction"]
        ),
        "height_error_magnitude_metres": forward.height_error_magnitude.detach().cpu().tolist(),
        "vertical_speed_magnitude_metres_per_second": (
            forward.vertical_speed_magnitude.detach().cpu().tolist()
        ),
        "approach_steps": forward.approach_steps.detach().cpu().tolist(),
        "prefix_steps": forward.prefix_steps.detach().cpu().tolist(),
        "endpoint_opposite_velocity_image_max_absolute_difference": (
            forward.endpoint_velocity_image_difference_max
        ),
    }


def matched_height_rows(
    student: ConnectomeController,
    source: ConnectomeController,
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    forward = _factorial_forward(
        student,
        source,
        batch=batch,
        seed=seed,
        history_steps=history_steps,
        policy_hz=policy_hz,
        config=config,
    )
    rows = []
    for item in range(batch):
        rows.append(
            shallow.masked_gradient(
                forward.prediction["height"][item],
                student,
                edge_indices,
                bias_indices,
                retain_graph=item + 1 < batch,
            )
        )
    return rows, {
        "height": _component_report(forward.prediction["height"], forward.target["height"]),
        "height_error_magnitude_metres": forward.height_error_magnitude.detach().cpu().tolist(),
        "vertical_speed_magnitude_metres_per_second": (
            forward.vertical_speed_magnitude.detach().cpu().tolist()
        ),
        "approach_steps": forward.approach_steps.detach().cpu().tolist(),
        "prefix_steps": forward.prefix_steps.detach().cpu().tolist(),
        "endpoint_opposite_velocity_image_max_absolute_difference": (
            forward.endpoint_velocity_image_difference_max
        ),
    }


def matched_height_response(
    candidate: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    candidate_prediction = torch.tensor(candidate["height"]["prediction"])
    source_prediction = torch.tensor(source["height"]["prediction"])
    target = torch.tensor(source["height"]["target"])
    magnitudes = torch.tensor(source["height_error_magnitude_metres"])
    by_height_magnitude = {}
    for magnitude in HEIGHT_ERROR_MAGNITUDES:
        selected = torch.isclose(magnitudes, torch.tensor(magnitude))
        sign = target[selected].sign()
        source_aligned = (source_prediction[selected] * sign).mean()
        candidate_aligned = (candidate_prediction[selected] * sign).mean()
        by_height_magnitude[f"{magnitude:.2f}"] = {
            "source_aligned_component": float(source_aligned),
            "candidate_aligned_component": float(candidate_aligned),
            "candidate_to_source_ratio": float(
                candidate_aligned / source_aligned.abs().clamp_min(1.0e-8)
            ),
            "items": int(selected.sum()),
        }
    by_scene = []
    for item in range(len(target)):
        sign = target[item].sign()
        source_aligned = source_prediction[item] * sign
        candidate_aligned = candidate_prediction[item] * sign
        nondegenerate = bool(
            target[item].abs() >= MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE
            and source_aligned.abs() >= MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE
        )
        by_scene.append(
            {
                "scene": item,
                "height_error_magnitude_metres": float(magnitudes[item]),
                "nondegenerate": nondegenerate,
                "source_aligned_component": float(source_aligned),
                "candidate_aligned_component": float(candidate_aligned),
                "candidate_to_source_ratio": (
                    float(candidate_aligned / source_aligned.abs().clamp_min(1.0e-8))
                    if nondegenerate
                    else None
                ),
            }
        )
    return {
        "all_scenes_nondegenerate": all(item["nondegenerate"] for item in by_scene),
        "by_scene": by_scene,
        "by_height_magnitude_diagnostic": by_height_magnitude,
    }


def _linearized_height_change(
    direction: dict[str, Tensor], rows: list[dict[str, Tensor]]
) -> Tensor:
    return torch.stack(
        [
            sum((row[name] * direction[name]).sum() for name in shallow.MASK_FAMILIES)
            for row in rows
        ]
    )


def direction_report(
    direction: dict[str, Tensor],
    gradient: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    source_height_component: Tensor,
) -> dict[str, Any]:
    derivative = float(
        sum((gradient[name] * direction[name]).sum() for name in shallow.MASK_FAMILIES)
    )
    height_change = _linearized_height_change(direction, rows)
    relative_height_change = height_change.abs() / source_height_component.abs().clamp_min(
        MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE
    )
    finite = native_train.numeric_tree_is_finite(
        [direction, derivative, height_change, relative_height_change]
    )
    height_null = bool(
        (source_height_component.abs() >= MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE).all()
        and relative_height_change.max() <= LINEARIZED_HEIGHT_NULL_RELATIVE_LIMIT
    )
    return {
        "admissible": finite and derivative < 0.0 and height_null,
        "all_finite": finite,
        "height_null_pass": height_null,
        "first_order_damping_loss_derivative": derivative,
        "linearized_matched_height_change_rms": float(height_change.square().mean().sqrt()),
        "linearized_matched_height_change_max_absolute": float(height_change.abs().max()),
        "linearized_matched_height_change_max_relative_to_source": float(
            relative_height_change.max()
        ),
        "linearized_matched_height_change_relative_limit": (
            LINEARIZED_HEIGHT_NULL_RELATIVE_LIMIT
        ),
        "reference_family_rms": upstream.reference_family_rms(direction),
    }


def bound_aware_height_null_projection(
    raw: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    current_edges: Tensor,
    *,
    maximum_iterations: int = 8,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Project on free coordinates, freezing any edge that would cross its bounds."""

    frozen_edges = torch.zeros_like(current_edges, dtype=torch.bool)
    iterations = []
    direction = {name: value.clone() for name, value in raw.items()}
    final_scale = 1.0
    for iteration in range(1, maximum_iterations + 1):
        working = {name: value.clone() for name, value in raw.items()}
        working["edge_magnitude"][frozen_edges] = 0.0
        free_rows = []
        for row in rows:
            free_row = {name: value.clone() for name, value in row.items()}
            free_row["edge_magnitude"][frozen_edges] = 0.0
            free_rows.append(free_row)
        projected, projection = upstream.project_in_reference_metric(
            working,
            free_rows,
            torch.zeros(len(rows), device=current_edges.device),
        )
        direction, final_scale = upstream.scale_to_reference_cap(
            projected, shallow.MASK_STEP_FAMILY_RMS_CAP
        )
        candidate_edges = current_edges + direction["edge_magnitude"]
        violations = (candidate_edges < 0.0) | (candidate_edges > 8.0)
        newly_frozen = violations & ~frozen_edges
        iterations.append(
            {
                "iteration": iteration,
                "frozen_edges_before": int(frozen_edges.sum()),
                "newly_frozen_edges": int(newly_frozen.sum()),
                "projection": projection,
            }
        )
        if not bool(newly_frozen.any()):
            break
        frozen_edges |= newly_frozen
    bounded_direction, bounds = bounded.apply_edge_bounds(direction, current_edges)
    return bounded_direction, {
        "iterations": iterations,
        "converged_without_bound_violation": not bool(
            ((current_edges + direction["edge_magnitude"] < 0.0)
            | (current_edges + direction["edge_magnitude"] > 8.0)).any()
        ),
        "frozen_bound_edges": int(frozen_edges.sum()),
        "final_rescale_to_step_cap": final_scale,
        "final_parameter_bounds": bounds,
    }


def finite_difference_report(
    student: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    direction: dict[str, Tensor],
    *,
    derivative: float,
    baseline_loss: float,
    args: argparse.Namespace,
    config: HoverConfig,
) -> dict[str, Any]:
    shallow.set_masked_trial(
        student,
        source_parameters,
        edge_indices,
        bias_indices,
        direction,
        FINITE_DIFFERENCE_SCALE,
    )
    with torch.no_grad():
        loss, _ = factorial_terms(
            student,
            source,
            batch=args.batch_size,
            seed=args.seed + 10_000,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
    measured = (float(loss.detach()) - baseline_loss) / FINITE_DIFFERENCE_SCALE
    relative_error = abs(measured - derivative) / max(abs(derivative), 1.0e-12)
    finite = native_train.numeric_tree_is_finite([derivative, measured, relative_error])
    shallow.set_masked_trial(
        student,
        source_parameters,
        edge_indices,
        bias_indices,
        direction,
        0.0,
    )
    return {
        "pass": (
            finite
            and derivative < 0.0
            and measured < 0.0
            and relative_error <= FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "all_finite": finite,
        "scale": FINITE_DIFFERENCE_SCALE,
        "autograd_directional_derivative": derivative,
        "forward_finite_difference_derivative": measured,
        "relative_error": relative_error,
        "relative_error_limit": FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
    }


@torch.no_grad()
def trial_report(
    student: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    direction: dict[str, Tensor],
    *,
    scale: float,
    baseline_fixed: dict[str, Any],
    baseline_complete: dict[str, Any],
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    actual = shallow.set_masked_trial(
        student, source_parameters, edge_indices, bias_indices, direction, scale
    )
    _, fixed = factorial_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    _, complete = factorial_terms(
        student,
        student,
        batch=args.batch_size,
        seed=args.seed + 10_001,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    fixed_improvement = (
        baseline_fixed["damping"]["fixed_scale_nrmse"]
        - fixed["damping"]["fixed_scale_nrmse"]
    )
    complete_improvement = (
        baseline_complete["damping"]["fixed_scale_nrmse"]
        - complete["damping"]["fixed_scale_nrmse"]
    )
    fixed_height = matched_height_response(fixed, baseline_fixed)
    complete_height = matched_height_response(complete, baseline_complete)
    height = shallow.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = ablation.functional_ablation_checks(
        student,
        source,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    reference_norm = upstream.reference_metric_norm(actual)
    reference_family_rms = upstream.reference_family_rms(actual)
    matched_height_ok = all(
        group["all_scenes_nondegenerate"]
        and all(
            shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
            <= value["candidate_to_source_ratio"]
            <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
            for value in group["by_scene"]
        )
        for group in (fixed_height, complete_height)
    )
    existing_height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    finite = native_train.numeric_tree_is_finite(
        [actual, fixed, complete, fixed_height, complete_height, height, functional]
    )
    reasons = []
    if fixed_improvement < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT:
        reasons.append("fixed-prefix factorial damping NRMSE improvement below 1e-4")
    if complete_improvement < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT:
        reasons.append("complete-zero-state factorial damping NRMSE improvement below 1e-4")
    if not matched_height_ok:
        reasons.append("matched factorial height response left source-relative 10% band")
    if not existing_height_ok:
        reasons.append("existing visual height response left source-relative 10% band")
    if not functional["pass"]:
        reasons.append("non-throttle functional preservation checks failed")
    if max(reference_family_rms.values()) > shallow.MASK_STEP_FAMILY_RMS_CAP * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected-family step RMS cap")
    if reference_norm > shallow.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected source metric radius")
    if max(
        fixed["endpoint_opposite_velocity_image_max_absolute_difference"],
        complete["endpoint_opposite_velocity_image_max_absolute_difference"],
    ) != 0.0:
        reasons.append("opposite-velocity histories did not end on identical pixels")
    if not finite:
        reasons.append("nonfinite preflight replay or parameter displacement")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "scale": scale,
        "all_finite": finite,
        "fixed_prefix": fixed,
        "complete_zero_state": complete,
        "fixed_prefix_damping_nrmse_improvement": fixed_improvement,
        "complete_zero_state_damping_nrmse_improvement": complete_improvement,
        "matched_height_response": {
            "fixed_prefix": fixed_height,
            "complete_zero_state": complete_height,
        },
        "existing_height_response": height,
        "functional_ablation": functional,
        "reference_actual_family_rms": reference_family_rms,
        "reference_actual_metric_norm": reference_norm,
    }


def main() -> int:
    args = parse_args()
    shallow.validate_args(args)
    if args.batch_size % 4:
        raise SystemExit("--batch-size must be a multiple of four")
    if args.history_steps < max(APPROACH_STEPS):
        raise SystemExit("history must cover every preregistered approach duration")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy frequency must divide the 100 Hz physics rate")
    physics_steps = physics_hz // args.policy_hz
    edges_np, biases_np, mask = upstream.build_expanded_route_mask(args.graph, args.raw_dir)
    upstream.validate_frozen_mask(mask)
    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != graph_sha256:
        raise SystemExit("checkpoint and graph hashes do not match")
    if int(checkpoint.get("policy_hz", args.policy_hz)) != args.policy_hz:
        raise SystemExit("checkpoint and requested policy frequencies do not match")

    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source):
        controller.load_state_dict(checkpoint["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edge_indices = torch.from_numpy(edges_np).to(device)
    bias_indices = torch.from_numpy(biases_np).to(device)
    source_parameters = trust.clone_parameters(source)
    started = perf_counter()

    baseline_loss_tensor, baseline_fixed = factorial_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    baseline_loss = float(baseline_loss_tensor.detach())
    gradient = shallow.masked_gradient(
        baseline_loss_tensor, student, edge_indices, bias_indices
    )
    with torch.no_grad():
        _, baseline_complete = factorial_terms(
            source,
            source,
            batch=args.batch_size,
            seed=args.seed + 10_001,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )

    raw, raw_scale = upstream.scale_to_reference_cap(
        upstream_train.fixed_metric_riesz(gradient), shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    rows, height_projection_bank = matched_height_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        batch=args.batch_size,
        # Deterministically replay the objective bank so its P rows and D gradient
        # describe the same scenes, amplitudes, prefixes and trajectories.
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    case_fields = (
        "height_error_magnitude_metres",
        "vertical_speed_magnitude_metres_per_second",
        "approach_steps",
        "prefix_steps",
    )
    projection_cases_match = all(
        height_projection_bank[field] == baseline_fixed[field] for field in case_fields
    )
    projection_height_difference = float(
        (
            torch.tensor(height_projection_bank["height"]["prediction"])
            - torch.tensor(baseline_fixed["height"]["prediction"])
        )
        .abs()
        .max()
    )
    projection_bank_match = {
        "pass": projection_cases_match and projection_height_difference <= 1.0e-7,
        "case_fields_match_exactly": projection_cases_match,
        "height_component_max_absolute_difference": projection_height_difference,
        "height_component_max_absolute_difference_limit": 1.0e-7,
    }
    projected, projection = bound_aware_height_null_projection(
        raw,
        rows,
        student.edge_magnitude[edge_indices].detach(),
    )
    source_height_component = torch.tensor(
        height_projection_bank["height"]["prediction"], device=device
    )
    direction_reports = {
        "raw_unprotected_diagnostic": direction_report(
            raw, gradient, rows, source_height_component
        ),
        "matched_height_null": direction_report(
            projected, gradient, rows, source_height_component
        ),
    }
    selected = (
        projected
        if direction_reports["matched_height_null"]["admissible"]
        and projection["converged_without_bound_violation"]
        and projection_bank_match["pass"]
        else None
    )

    finite_difference = None
    trials = []
    accepted_scale = None
    if selected is not None:
        derivative = direction_reports["matched_height_null"][
            "first_order_damping_loss_derivative"
        ]
        finite_difference = finite_difference_report(
            student,
            source,
            source_parameters,
            edge_indices,
            bias_indices,
            selected,
            derivative=derivative,
            baseline_loss=baseline_loss,
            args=args,
            config=config,
        )
        for scale in shallow.BACKTRACK_SCALES:
            trial = trial_report(
                student,
                source,
                source_parameters,
                edge_indices,
                bias_indices,
                selected,
                scale=scale,
                baseline_fixed=baseline_fixed,
                baseline_complete=baseline_complete,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            trials.append(trial)
            if trial["pass"]:
                accepted_scale = scale
                break

    trust.load_parameters(student, source_parameters)
    restored_error = max(
        float((getattr(student, name).detach() - source_parameters[name]).abs().max())
        for name in trust.PARAMETER_FAMILIES
    )
    report = {
        "experiment": "variable-height-factorial-damping-route-preflight-v1",
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": (
            accepted_scale is not None
            and finite_difference is not None
            and finite_difference["pass"]
        ),
        "accepted_scale": accepted_scale,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        },
        "actor_contract_unchanged": True,
        "mask": mask,
        "frozen": {
            "retinal_mapping_and_input_gains": True,
            "optic_neuron_biases": True,
            "all_neuron_time_constants": True,
            "motor_biases": True,
            "topology": True,
            "transmitter_signs": True,
        },
        "protocol": {
            "seed": args.seed,
            "scenes_per_bank": args.batch_size,
            "histories_per_scene": len(BRANCH_SIGNS),
            "height_error_magnitudes_metres": list(HEIGHT_ERROR_MAGNITUDES),
            "vertical_speed_magnitudes_metres_per_second": list(
                VERTICAL_SPEED_MAGNITUDES
            ),
            "branch_sign_order_height_velocity": [list(item) for item in BRANCH_SIGNS],
            "approach_steps": list(APPROACH_STEPS),
            "prefix_steps": list(PREFIX_STEPS),
            "history_steps": args.history_steps,
            "policy_hz": args.policy_hz,
            "objective": "factorial damping component only",
            "matched_height_component_projection": True,
            "common_component": "unanchored diagnostic only",
            "interaction_component": "diagnostic only",
            "backtrack_scales": list(shallow.BACKTRACK_SCALES),
            "finite_difference_scale": FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_limit": (
                FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "reference_metric_denominators": dict(upstream.REFERENCE_DENOMINATORS),
        },
        "thresholds": {
            "minimum_fixed_prefix_damping_nrmse_improvement": (
                shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT
            ),
            "minimum_complete_zero_state_damping_nrmse_improvement": (
                shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT
            ),
            "matched_height_source_ratio_range": list(
                shallow.HEIGHT_CONTRAST_RATIO_RANGE
            ),
            "matched_height_nondegenerate_absolute_minimum": (
                MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE
            ),
            "post_bound_linearized_height_change_relative_maximum": (
                LINEARIZED_HEIGHT_NULL_RELATIVE_LIMIT
            ),
            "existing_height_source_ratio_range": list(
                shallow.HEIGHT_CONTRAST_RATIO_RANGE
            ),
            "selected_family_step_rms_cap": shallow.MASK_STEP_FAMILY_RMS_CAP,
            "selected_source_metric_radius": shallow.MASK_SOURCE_METRIC_RADIUS,
            "absolute_throttle_source_gate": False,
            "dynamic_rpy_each_nrmse_limit": trust.SOURCE_FUNCTION_NRMSE_LIMIT,
        },
        "baseline_fixed_prefix": baseline_fixed,
        "baseline_complete_zero_state": baseline_complete,
        "height_projection_bank_same_cases_as_fixed_prefix_objective": (
            height_projection_bank
        ),
        "height_projection_bank_match": projection_bank_match,
        "raw_direction_initial_cap_scale": raw_scale,
        "bound_aware_height_projection": projection,
        "direction_reports": direction_reports,
        "selected_candidate": "matched_height_null" if selected is not None else None,
        "finite_difference": finite_difference,
        "trials": trials,
        "parameters_restored_max_absolute_error": restored_error,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass establishes only a one-step damping direction separated from the matched "
            "height component. It does not promote a controller or authorize closed-loop flight."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": report["pass"],
                "selected_candidate": report["selected_candidate"],
                "accepted_scale": accepted_scale,
                "finite_difference_pass": (
                    finite_difference["pass"] if finite_difference is not None else False
                ),
                "parameters_restored_max_absolute_error": restored_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
