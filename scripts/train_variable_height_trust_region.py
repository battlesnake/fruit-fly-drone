#!/usr/bin/env python3
"""Test projected local plasticity without promoting a hover controller."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_bridge as bridge  # noqa: E402
import train_variable_height_hover as hover  # noqa: E402

from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    HoverConfig,
)
from flydrone.variable_hover import protocol_manifest, sample_marker_pairs  # noqa: E402
from flydrone.visual_hover import DEFAULT_VISUAL_CAMERA  # noqa: E402

PARAMETER_FAMILIES = ("edge_magnitude", "bias", "raw_time_constant")
BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
STEP_FAMILY_RMS_CAP = 2.0e-5
SOURCE_METRIC_RADIUS = 1.0e-4
PER_UPDATE_COMMON_NRMSE_LIMIT = 0.02
SOURCE_COMMON_NRMSE_LIMIT = 0.05
SOURCE_COMMON_MAX_ABSOLUTE_LIMIT = 0.005
SOURCE_FUNCTION_NRMSE_LIMIT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/trust-region-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--constraint-batch-size", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=25)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--constraint-prefix-steps", type=int, default=50)
    parser.add_argument("--interim-evaluation-episodes", type=int, default=64)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--response-seconds", type=float, default=6.0)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def clone_parameters(controller: ConnectomeController) -> dict[str, torch.Tensor]:
    return {
        name: getattr(controller, name).detach().clone() for name in PARAMETER_FAMILIES
    }


@torch.no_grad()
def load_parameters(
    controller: ConnectomeController, parameters: dict[str, torch.Tensor]
) -> None:
    for name in PARAMETER_FAMILIES:
        getattr(controller, name).copy_(parameters[name])


def parameter_delta(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: first[name] - second[name] for name in PARAMETER_FAMILIES}


def family_rms(values: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        name: float(torch.sqrt(values[name].detach().square().mean()))
        for name in PARAMETER_FAMILIES
    }


def family_metric_norm(values: dict[str, torch.Tensor]) -> float:
    rms = family_rms(values)
    return sum(value * value for value in rms.values()) ** 0.5


def cap_displacement(
    displacement: dict[str, torch.Tensor], cap: float
) -> tuple[dict[str, torch.Tensor], float]:
    """Apply one scalar cap, preserving the projected direction."""

    largest = max(family_rms(displacement).values())
    scale = min(1.0, cap / max(largest, 1.0e-30))
    return ({name: scale * value for name, value in displacement.items()}, scale)


def project_displacement(
    displacement: dict[str, torch.Tensor],
    jacobian_rows: list[dict[str, torch.Tensor]],
    residual: torch.Tensor,
    *,
    rtol: float = 1.0e-6,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Project in sum-of-family-mean-squares geometry.

    If ``D_f=sqrt(n_f)I``, equal-family RMS coordinates are ``q=D^-1 delta``.
    This implements ``q*=q-A^T (AA^T)^+ (Aq+r)`` with ``A=JD``.
    """

    if not jacobian_rows:
        raise ValueError("at least one Jacobian row is required")
    if residual.shape != (len(jacobian_rows),):
        raise ValueError("residual must have one value per Jacobian row")
    device = residual.device
    rows = len(jacobian_rows)
    gram = torch.empty(rows, rows, device=device, dtype=torch.float64)
    j_delta = torch.empty(rows, device=device, dtype=torch.float64)
    family_sizes = {name: displacement[name].numel() for name in PARAMETER_FAMILIES}
    for row_index, row in enumerate(jacobian_rows):
        derivative = sum(
            (row[name] * displacement[name]).sum() for name in PARAMETER_FAMILIES
        )
        j_delta[row_index] = derivative.double()
        for column_index in range(row_index + 1):
            other = jacobian_rows[column_index]
            value = sum(
                family_sizes[name] * (row[name] * other[name]).sum()
                for name in PARAMETER_FAMILIES
            )
            gram[row_index, column_index] = value.double()
            gram[column_index, row_index] = value.double()

    diagonal = gram.diagonal().clamp_min(0.0).sqrt()
    active = diagonal > 1.0e-12
    if not bool(active.any()):
        raise RuntimeError("all common-output Jacobian rows are numerically zero")
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    active_gram = gram[active][:, active]
    row_scale = diagonal[active]
    normalized_gram = active_gram / (row_scale[:, None] * row_scale[None, :])
    right_hand_side = (j_delta + residual.double())[active] / row_scale
    singular_values = torch.linalg.svdvals(normalized_gram)
    cutoff = rtol * singular_values.max()
    rank = int((singular_values > cutoff).sum())
    coefficients_scaled = torch.linalg.pinv(normalized_gram, rtol=rtol) @ right_hand_side
    coefficients = coefficients_scaled / row_scale

    projected = {name: value.detach().clone() for name, value in displacement.items()}
    for coefficient, row_index in zip(coefficients, active_indices, strict=True):
        row = jacobian_rows[int(row_index)]
        coefficient_value = float(coefficient)
        for name in PARAMETER_FAMILIES:
            projected[name].add_(
                row[name], alpha=-coefficient_value * family_sizes[name]
            )

    linearized_after = residual.double().clone()
    for row_index, row in enumerate(jacobian_rows):
        linearized_after[row_index] += sum(
            (row[name] * projected[name]).sum() for name in PARAMETER_FAMILIES
        ).double()
    return projected, {
        "rows": rows,
        "active_rows": int(active.sum()),
        "retained_rank": rank,
        "normalized_gram_singular_values": singular_values.detach().cpu().tolist(),
        "residual_before": residual.detach().cpu().tolist(),
        "unprojected_linearized_residual": (residual.double() + j_delta)
        .detach()
        .cpu()
        .tolist(),
        "projected_linearized_residual": linearized_after.detach().cpu().tolist(),
        "projected_linearized_residual_max_absolute": float(linearized_after.abs().max()),
    }


def fixed_pair_window_terms(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    half_step: float,
    batch: int,
    seed: int,
    prefix_steps: int,
    unroll: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Per-example common output after a fixed source-generated dynamic burn-in."""

    hover.seed_everything(seed)
    pairs = sample_marker_pairs(
        batch,
        device=device,
        held_out=False,
        half_step_metres=half_step,
    )
    state, stick_state, scene, prefix_marker = bridge._initial_pair_state(
        pairs,
        randomized_scene=True,
        all_style_combinations=True,
        device=device,
        config=config,
    )
    state, _, _, source_neural, valid = bridge._dual_prefix(
        source,
        source,
        state,
        stick_state,
        prefix_marker,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    loss_start = min(unroll - 1, 10)
    student_a, student_b = bridge._branch_outputs(
        student,
        state,
        source_neural,
        scene,
        pairs.marker_a,
        pairs.marker_b,
        response_steps=unroll,
        loss_start=loss_start,
    )
    with torch.no_grad():
        source_a, source_b = bridge._branch_outputs(
            source,
            state,
            source_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=unroll,
            loss_start=loss_start,
        )
        target_a = hover.teacher_motor(state, pairs.marker_a, config)
        target_b = hover.teacher_motor(state, pairs.marker_b, config)

    common_error = 0.5 * (
        student_a[:, :, 3] + student_b[:, :, 3] - source_a[:, :, 3] - source_b[:, :, 3]
    )
    constraints = common_error.mean(dim=0) / hover.MOTOR_CORRECTION_SCALES[3]
    desired = target_a[:, 3] - target_b[:, 3]
    predicted = student_a[:, :, 3] - student_b[:, :, 3]
    scale = max(0.01, 0.214 * (2.0 * half_step))
    valid_steps = valid[None].expand(predicted.shape[0], -1)
    contrast_loss = hover.masked_mean(
        ((predicted - desired[None]) / scale).square(), valid_steps
    )
    return constraints, contrast_loss, {
        "contrast_nrmse": float(torch.sqrt(contrast_loss.detach())),
        "common_constraint_rms": float(torch.sqrt(constraints.detach().square().mean())),
        "common_constraint_max_absolute": float(constraints.detach().abs().max()),
        "valid_fraction": float(valid.float().mean()),
        "style_slots": [
            f"wall_{int(scene.wall_style[item])}_floor_{int(scene.floor_style[item])}"
            for item in range(batch)
        ],
    }


def common_output_jacobian(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[
    list[dict[str, torch.Tensor]],
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    rows: list[dict[str, torch.Tensor]] = []
    contrast_gradient = {
        name: torch.zeros_like(getattr(student, name)) for name in PARAMETER_FAMILIES
    }
    residuals = []
    labels = []
    metrics = {}
    parameters = [getattr(student, name) for name in PARAMETER_FAMILIES]
    for amplitude_index, (amplitude, half_step) in enumerate(
        bridge.PAIR_HALF_STEPS.items()
    ):
        constraints, contrast_loss, amplitude_metrics = fixed_pair_window_terms(
            student,
            source,
            half_step=half_step,
            batch=args.constraint_batch_size,
            seed=args.seed + 20_000 + amplitude_index,
            prefix_steps=args.constraint_prefix_steps,
            unroll=args.unroll,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        metrics[amplitude] = amplitude_metrics
        amplitude_gradient = torch.autograd.grad(
            contrast_loss,
            parameters,
            retain_graph=True,
        )
        for name, gradient in zip(PARAMETER_FAMILIES, amplitude_gradient, strict=True):
            contrast_gradient[name].add_(0.5 * gradient.detach())
        for item in range(args.constraint_batch_size):
            gradients = torch.autograd.grad(
                constraints[item],
                parameters,
                retain_graph=item + 1 < args.constraint_batch_size,
            )
            rows.append(
                {
                    name: gradient.detach().clone()
                    for name, gradient in zip(PARAMETER_FAMILIES, gradients, strict=True)
                }
            )
            residuals.append(constraints[item].detach())
            labels.append(
                f"{amplitude}_{amplitude_metrics['style_slots'][item]}_slot_{item}"
            )
        del constraints
    return (
        rows,
        torch.stack(residuals),
        contrast_gradient,
        {"labels": labels, "amplitudes": metrics},
    )


@torch.no_grad()
def evaluate_fixed_pairs(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    amplitudes = {}
    for amplitude_index, (amplitude, half_step) in enumerate(
        bridge.PAIR_HALF_STEPS.items()
    ):
        _, _, metrics = fixed_pair_window_terms(
            student,
            source,
            half_step=half_step,
            batch=args.constraint_batch_size,
            seed=args.seed + 20_000 + amplitude_index,
            prefix_steps=args.constraint_prefix_steps,
            unroll=args.unroll,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        amplitudes[amplitude] = metrics
    return {
        "mean_contrast_nrmse": sum(
            value["contrast_nrmse"] for value in amplitudes.values()
        )
        / len(amplitudes),
        "maximum_common_constraint_absolute": max(
            value["common_constraint_max_absolute"] for value in amplitudes.values()
        ),
        "amplitudes": amplitudes,
    }


@torch.no_grad()
def functional_trust_checks(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    source_parameters: dict[str, torch.Tensor],
    previous_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source_pair = {}
    per_update_pair = {}
    for amplitude_index, (amplitude, half_step) in enumerate(
        bridge.PAIR_HALF_STEPS.items()
    ):
        seed = args.seed + 30_000 + amplitude_index
        hover.seed_everything(seed)
        _, source_pair[amplitude] = bridge._teacher_pair_loss(
            student,
            source,
            half_step=half_step,
            batch=args.constraint_batch_size,
            unroll=args.unroll,
            prefix_steps=args.constraint_prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )
        hover.seed_everything(seed)
        _, per_update_pair[amplitude] = bridge._teacher_pair_loss(
            student,
            previous,
            half_step=half_step,
            batch=args.constraint_batch_size,
            unroll=args.unroll,
            prefix_steps=args.constraint_prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )

    hover.seed_everything(args.seed + 30_010)
    _, legacy = bridge._source_cross_band_loss(
        student,
        source,
        batch=args.constraint_batch_size,
        unroll=args.unroll,
        prefix_steps=args.constraint_prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    hover.seed_everything(args.seed + 30_020)
    _, dynamic = bridge._dynamic_source_replay_loss(
        student,
        source,
        batch=args.constraint_batch_size,
        unroll=args.unroll,
        prefix_steps=args.constraint_prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
        all_style_combinations=True,
    )
    current = clone_parameters(student)
    step_delta = parameter_delta(current, previous_parameters)
    source_delta = parameter_delta(current, source_parameters)
    step_rms = family_rms(step_delta)
    source_rms = family_rms(source_delta)
    reasons = []
    if max(step_rms.values()) > STEP_FAMILY_RMS_CAP * (1.0 + 1.0e-5):
        reasons.append("per-family parameter RMS step cap")
    if family_metric_norm(source_delta) > SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("source-global family-metric radius")
    if any(
        value["common_nrmse"] > PER_UPDATE_COMMON_NRMSE_LIMIT
        for value in per_update_pair.values()
    ):
        reasons.append("per-update common-throttle RMS")
    if any(
        value["common_nrmse"] > SOURCE_COMMON_NRMSE_LIMIT
        for value in source_pair.values()
    ):
        reasons.append("source-global common-throttle RMS")
    if any(
        value["common_error_max_absolute"] > SOURCE_COMMON_MAX_ABSOLUTE_LIMIT
        for value in source_pair.values()
    ):
        reasons.append("source-global maximum common-throttle drift")
    if legacy["source_motor_nrmse"] > SOURCE_FUNCTION_NRMSE_LIMIT:
        reasons.append("legacy response source error")
    if any(
        dynamic[f"{axis}_source_nrmse"] > SOURCE_FUNCTION_NRMSE_LIMIT
        for axis in ("roll", "pitch", "yaw", "throttle")
    ):
        reasons.append("dynamic source replay error")
    valid_values = [
        *(value["valid_fraction"] for value in source_pair.values()),
        *(value["valid_fraction"] for value in per_update_pair.values()),
        legacy["valid_fraction"],
        dynamic["valid_fraction"],
    ]
    if min(valid_values) < 1.0:
        reasons.append("invalid complete replay")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "source_pair": source_pair,
        "per_update_pair": per_update_pair,
        "legacy": legacy,
        "dynamic": dynamic,
        "step_family_rms": step_rms,
        "source_family_rms": source_rms,
        "source_family_metric_norm": family_metric_norm(source_delta),
    }


def set_trial_parameters(
    student: ConnectomeController,
    base: dict[str, torch.Tensor],
    displacement: dict[str, torch.Tensor],
    scale: float,
) -> None:
    with torch.no_grad():
        for name in PARAMETER_FAMILIES:
            getattr(student, name).copy_(base[name] + scale * displacement[name])
    student.project_parameters()


def find_safe_trial(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    *,
    base_parameters: dict[str, torch.Tensor],
    source_parameters: dict[str, torch.Tensor],
    displacement: dict[str, torch.Tensor],
    baseline_fixed: dict[str, Any],
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[float | None, list[dict[str, Any]]]:
    trials = []
    for scale in BACKTRACK_SCALES:
        set_trial_parameters(student, base_parameters, displacement, scale)
        fixed = evaluate_fixed_pairs(
            student,
            source,
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        contrast_improved = (
            fixed["mean_contrast_nrmse"]
            < baseline_fixed["mean_contrast_nrmse"] - 1.0e-6
        )
        trust = functional_trust_checks(
            student,
            source,
            previous,
            args=args,
            physics_steps=physics_steps,
            device=device,
            config=config,
            source_parameters=source_parameters,
            previous_parameters=base_parameters,
        )
        entry = {
            "scale": scale,
            "contrast_improved": contrast_improved,
            "fixed_pair": fixed,
            "trust": trust,
        }
        trials.append(entry)
        if contrast_improved and trust["pass"]:
            return scale, trials
    load_parameters(student, base_parameters)
    return None, trials


def proposal_and_projection(
    student: ConnectomeController,
    source: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_parameters: dict[str, torch.Tensor],
    *,
    attempt: int,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, Any],
]:
    base = clone_parameters(student)
    proposal_metrics = bridge.accumulated_update(
        student,
        source,
        optimizer,
        source_parameters,
        update=attempt - 1,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    proposed = clone_parameters(student)
    displacement = parameter_delta(proposed, base)
    load_parameters(student, base)
    rows, residual, contrast_gradient, jacobian = common_output_jacobian(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    projected, projection = project_displacement(displacement, rows, residual)
    projected_before_cap_rms = family_rms(projected)
    projected, cap_scale = cap_displacement(projected, STEP_FAMILY_RMS_CAP)
    ordinary_capped, ordinary_cap_scale = cap_displacement(
        displacement, STEP_FAMILY_RMS_CAP
    )
    ordinary_derivative = float(
        sum(
            (contrast_gradient[name] * ordinary_capped[name]).sum()
            for name in PARAMETER_FAMILIES
        )
    )
    projected_derivative = float(
        sum(
            (contrast_gradient[name] * projected[name]).sum()
            for name in PARAMETER_FAMILIES
        )
    )
    del rows
    projection.update(
        {
            "raw_proposal_family_rms": family_rms(displacement),
            "projected_family_rms_before_cap": projected_before_cap_rms,
            "single_scalar_step_cap_scale": cap_scale,
            "projected_family_rms_after_cap": family_rms(projected),
            "fixed_contrast_first_order": {
                "ordinary_capped_derivative": ordinary_derivative,
                "projected_capped_derivative": projected_derivative,
                "descent_surviving_fraction": (
                    projected_derivative / ordinary_derivative
                    if ordinary_derivative < 0.0 and projected_derivative < 0.0
                    else None
                ),
                "ordinary_cap_scale": ordinary_cap_scale,
            },
            "jacobian_batch": jacobian,
        }
    )
    return base, displacement, projected, {
        "proposal": proposal_metrics,
        "projection": projection,
    }


def final_decision(
    candidate: dict[str, Any], baseline: dict[str, Any], *, reached_limit: bool
) -> dict[str, Any]:
    training_improvement = 1.0 - candidate["training_support_pair_matrix"][
        "mean_contrast_nrmse"
    ] / baseline["training_support_pair_matrix"]["mean_contrast_nrmse"]
    development_improvement = 1.0 - candidate["fresh_height_pair_matrix"][
        "mean_contrast_nrmse"
    ] / baseline["fresh_height_pair_matrix"]["mean_contrast_nrmse"]
    preservation_pass, preservation_reasons = bridge._bridge_guards(candidate, baseline)
    reasons = list(preservation_reasons)
    if not reached_limit:
        reasons.append("did not reach 25 attempted updates")
    if training_improvement < 0.10:
        reasons.append("training-support contrast improved by less than ten percent")
    if development_improvement < 0.05:
        reasons.append("development-height contrast improved by less than five percent")
    if not candidate["legacy_pair"]["pass"]:
        reasons.append("legacy paired response failed")
    return {
        "pass": not reasons,
        "non_promotional": True,
        "reasons": reasons,
        "training_support_improvement_fraction": training_improvement,
        "development_height_improvement_fraction": development_improvement,
        "preservation_gate_pass": preservation_pass,
    }


def zero_reproduction_error_max(
    fixed: dict[str, Any], trust: dict[str, Any]
) -> float:
    values = [
        fixed["maximum_common_constraint_absolute"],
        trust["legacy"]["source_motor_nrmse"],
        *(value["common_nrmse"] for value in trust["source_pair"].values()),
        *(value["axis_nrmse"] for value in trust["source_pair"].values()),
        *(
            trust["dynamic"][f"{axis}_source_nrmse"]
            for axis in ("roll", "pitch", "yaw", "throttle")
        ),
    ]
    return max(values)


def checkpoint_payload(
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
    *,
    source_sha256: str,
    attempts: int,
    accepted_updates: int,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-projected-trust-region-v1",
        "non_promotional": True,
        "controller": controller.state_dict(),
        "graph_sha256": hover.file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "attempts": attempts,
        "accepted_updates": accepted_updates,
        "policy_hz": args.policy_hz,
        "plant_model_version": PLANT_MODEL_VERSION,
        "hover_config": asdict(config),
        "camera": asdict(DEFAULT_VISUAL_CAMERA),
        "actor_inputs": {
            "rgb_camera": True,
            "estimated_roll_pitch": True,
            "accelerometer": False,
            "mass": False,
            "hover_thrust": False,
            "external_history": False,
        },
        "protocol": protocol_manifest(),
        "scene_protocol": bridge.bridge_scene_manifest(args),
    }


def main() -> int:
    args = parse_args()
    if not args.graph.is_file() or not args.checkpoint.is_file():
        raise SystemExit("the generated graph and source checkpoint are required")
    if not args.smoke_test and args.attempts != 25:
        raise SystemExit("v1 is preregistered for exactly 25 attempted updates")
    if args.batch_size % 4 or args.constraint_batch_size % 4:
        raise SystemExit("training and constraint batches must be divisible by four")
    if args.unroll < 11 or args.constraint_prefix_steps < 0:
        raise SystemExit("unroll must be at least 11 and prefix steps nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the physics rate")
    physics_steps = physics_hz // args.policy_hz
    hover.seed_everything(args.seed)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    graph_sha256 = hover.file_sha256(args.graph)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(loaded["controller"])
    source.eval()
    source.requires_grad_(False)
    previous.eval()
    previous.requires_grad_(False)
    source_parameters = clone_parameters(source)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
    )
    args.scene_coverage = "all"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    # Preflight is a transaction: it consumes neither optimizer state nor training RNG.
    preflight_parameters = clone_parameters(student)
    preflight_optimizer = copy.deepcopy(optimizer.state_dict())
    preflight_rng = hover.capture_rng_state()
    baseline_fixed = evaluate_fixed_pairs(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    zero_trust = functional_trust_checks(
        student,
        source,
        previous,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        source_parameters=source_parameters,
        previous_parameters=preflight_parameters,
    )
    base, ordinary, projected, proposal_report = proposal_and_projection(
        student,
        source,
        optimizer,
        source_parameters,
        attempt=1,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    ordinary, ordinary_cap_scale = cap_displacement(ordinary, STEP_FAMILY_RMS_CAP)
    set_trial_parameters(student, base, ordinary, 1.0)
    ordinary_fixed = evaluate_fixed_pairs(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    ordinary_trust = functional_trust_checks(
        student,
        source,
        previous,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        source_parameters=source_parameters,
        previous_parameters=base,
    )
    load_parameters(student, base)
    projected_scale, projected_trials = find_safe_trial(
        student,
        source,
        previous,
        base_parameters=base,
        source_parameters=source_parameters,
        displacement=projected,
        baseline_fixed=baseline_fixed,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    zero_error = zero_reproduction_error_max(baseline_fixed, zero_trust)
    projected_improvement = (
        baseline_fixed["mean_contrast_nrmse"]
        - projected_trials[-1]["fixed_pair"]["mean_contrast_nrmse"]
        if projected_scale is not None
        else None
    )
    ordinary_improvement = (
        baseline_fixed["mean_contrast_nrmse"] - ordinary_fixed["mean_contrast_nrmse"]
    )
    preflight = {
        "zero_step": {
            "fixed_pair": baseline_fixed,
            "trust": zero_trust,
            "numerical_error_max": zero_error,
            "numerical_tolerance": 1.0e-5,
            "reproduces_source": zero_error <= 1.0e-5,
        },
        "proposal": proposal_report,
        "ordinary_capped_adam": {
            "single_scalar_step_cap_scale": ordinary_cap_scale,
            "fixed_pair": ordinary_fixed,
            "trust": ordinary_trust,
        },
        "projected_backtracking": {
            "accepted_scale": projected_scale,
            "trials": projected_trials,
        },
        "fixed_contrast_descent": {
            "ordinary_capped_improvement": ordinary_improvement,
            "accepted_projected_improvement": projected_improvement,
            "surviving_fraction": (
                projected_improvement / ordinary_improvement
                if projected_improvement is not None and ordinary_improvement > 0.0
                else None
            ),
        },
        "pass": bool(
            zero_trust["pass"]
            and zero_error <= 1.0e-5
            and projected_scale is not None
        ),
    }
    load_parameters(student, preflight_parameters)
    optimizer.load_state_dict(preflight_optimizer)
    hover.restore_rng_state(preflight_rng)
    print(json.dumps({"progress": "preflight", **preflight}), flush=True)

    baseline = None
    history = []
    attempted = 0
    accepted = 0
    consecutive_rejections = 0
    stop_reason = None
    if preflight["pass"] and not args.smoke_test:
        student.eval()
        baseline = bridge.evaluate_bridge(
            student,
            episodes=args.interim_evaluation_episodes,
            args=args,
            device=device,
            config=config,
            physics_steps=physics_steps,
            seed=args.seed + 40_000,
        )
        for attempt in range(1, args.attempts + 1):
            attempted = attempt
            base_parameters = clone_parameters(student)
            load_parameters(previous, base_parameters)
            optimizer_before = copy.deepcopy(optimizer.state_dict())
            baseline_fixed = evaluate_fixed_pairs(
                student,
                source,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            base, _, projected, proposal_report = proposal_and_projection(
                student,
                source,
                optimizer,
                source_parameters,
                attempt=attempt,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            accepted_scale, trials = find_safe_trial(
                student,
                source,
                previous,
                base_parameters=base,
                source_parameters=source_parameters,
                displacement=projected,
                baseline_fixed=baseline_fixed,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            if accepted_scale is None:
                load_parameters(student, base_parameters)
                optimizer.load_state_dict(optimizer_before)
                consecutive_rejections += 1
            else:
                accepted += 1
                consecutive_rejections = 0
            entry = {
                "attempt": attempt,
                "accepted": accepted_scale is not None,
                "accepted_scale": accepted_scale,
                "consecutive_rejections": consecutive_rejections,
                **proposal_report,
                "backtracking_trials": trials,
            }
            history.append(entry)
            print(
                json.dumps(
                    {
                        "progress": "attempt",
                        "attempt": attempt,
                        "accepted": entry["accepted"],
                        "accepted_scale": accepted_scale,
                        "consecutive_rejections": consecutive_rejections,
                        "fixed_contrast_before": baseline_fixed["mean_contrast_nrmse"],
                        "fixed_contrast_after": trials[-1]["fixed_pair"][
                            "mean_contrast_nrmse"
                        ],
                    }
                ),
                flush=True,
            )
            if consecutive_rejections >= 5:
                stop_reason = "five consecutive unsafe or non-improving proposals"
                break
    elif not preflight["pass"]:
        stop_reason = "projection preflight failed"
    else:
        stop_reason = "smoke test ended after preflight"

    candidate = None
    decision = None
    reached_limit = attempted == 25
    if baseline is not None and reached_limit:
        student.eval()
        candidate = bridge.evaluate_bridge(
            student,
            episodes=args.interim_evaluation_episodes,
            args=args,
            device=device,
            config=config,
            physics_steps=physics_steps,
            seed=args.seed + 40_000,
        )
        decision = final_decision(candidate, baseline, reached_limit=True)
        torch.save(
            checkpoint_payload(
                student,
                args,
                config,
                source_sha256=hover.file_sha256(args.checkpoint),
                attempts=attempted,
                accepted_updates=accepted,
            ),
            args.output_dir / "nonpromotional-candidate.pt",
        )
        if not decision["pass"] and stop_reason is None:
            stop_reason = "; ".join(decision["reasons"])

    report = {
        "experiment": "variable-height-projected-trust-region-v1",
        "passed": bool(decision and decision["pass"]),
        "promoted": False,
        "claim": (
            "safe local visual-height plasticity only; this run cannot promote a hover "
            "controller and repeatedly used matrices are development evidence"
        ),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": hover.file_sha256(args.checkpoint),
        "graph": str(args.graph),
        "graph_sha256": graph_sha256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "seed": args.seed,
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "attempts_requested": args.attempts,
        "attempts_completed": attempted,
        "accepted_updates": accepted,
        "stop_reason": stop_reason,
        "thresholds": {
            "step_family_rms_cap": STEP_FAMILY_RMS_CAP,
            "source_family_metric_radius": SOURCE_METRIC_RADIUS,
            "per_update_common_nrmse_limit": PER_UPDATE_COMMON_NRMSE_LIMIT,
            "source_common_nrmse_limit": SOURCE_COMMON_NRMSE_LIMIT,
            "source_common_max_absolute_motor_limit": SOURCE_COMMON_MAX_ABSOLUTE_LIMIT,
            "source_function_nrmse_limit": SOURCE_FUNCTION_NRMSE_LIMIT,
            "backtrack_scales": list(BACKTRACK_SCALES),
            "consecutive_rejection_stop": 5,
        },
        "actor_inputs": checkpoint_payload(
            student,
            args,
            config,
            source_sha256=hover.file_sha256(args.checkpoint),
            attempts=attempted,
            accepted_updates=accepted,
        )["actor_inputs"],
        "preflight": preflight,
        "baseline": baseline,
        "history": history,
        "candidate": candidate,
        "decision": decision,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.smoke_test:
        return 0 if preflight["pass"] else 1
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
