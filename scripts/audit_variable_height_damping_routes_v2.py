#!/usr/bin/env python3
"""Retry the fixed damping route with bounded, rather than equality, common output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_damping_routes as v1  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

PER_UPDATE_COMMON_RMS = 0.001
SOURCE_COMMON_RMS = 0.0025
COMMON_MAX_ABSOLUTE = 0.005
BLEND_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = v1.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=(REPO_ROOT / "runs/variable-height-hover/damping-route-bounded-preflight-001"),
        seed=260_923,
    )
    return parser.parse_args()


def scale_to_final_cap(direction: dict[str, Tensor], cap: float) -> tuple[dict[str, Tensor], float]:
    largest = max(v1.masked_family_rms(direction).values())
    if largest <= 0.0:
        return {name: value.clone() for name, value in direction.items()}, 1.0
    scale = cap / largest
    return {name: scale * value for name, value in direction.items()}, scale


def apply_edge_bounds(
    direction: dict[str, Tensor], baseline_edge: Tensor
) -> tuple[dict[str, Tensor], dict[str, int]]:
    candidate = baseline_edge + direction["edge_magnitude"]
    below = candidate < 0.0
    above = candidate > 8.0
    bounded = {
        "edge_magnitude": candidate.clamp(0.0, 8.0) - baseline_edge,
        "bias": direction["bias"].clone(),
    }
    return bounded, {
        "edges_clipped_at_zero": int(below.sum()),
        "edges_clipped_at_eight": int(above.sum()),
    }


def linearized_common_native(
    direction: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    residual: Tensor,
) -> Tensor:
    predicted = residual.clone()
    for index, row in enumerate(rows):
        predicted[index] += sum((row[name] * direction[name]).sum() for name in v1.MASK_FAMILIES)
    # The v1 rows are normalized by the fixed 0.05 throttle correction scale.
    return 0.05 * predicted


def screen_candidates(
    candidates: dict[str, dict[str, Tensor]],
    gradient: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    residual: Tensor,
    *,
    per_update_rms: float = PER_UPDATE_COMMON_RMS,
    source_rms: float = SOURCE_COMMON_RMS,
    maximum_absolute: float = COMMON_MAX_ABSOLUTE,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    reports = {}
    for label, direction in candidates.items():
        predicted_source = linearized_common_native(direction, rows, residual)
        current_source = 0.05 * residual
        predicted_step = predicted_source - current_source
        step_rms = float(predicted_step.square().mean().sqrt())
        source_rms_value = float(predicted_source.square().mean().sqrt())
        maximum = float(predicted_source.abs().max())
        derivative = float(
            sum((gradient[name] * direction[name]).sum() for name in v1.MASK_FAMILIES)
        )
        admissible = (
            step_rms <= per_update_rms
            and source_rms_value <= source_rms
            and maximum <= maximum_absolute
            and derivative < 0.0
        )
        reports[label] = {
            "first_order_loss_derivative": derivative,
            "linearized_step_common_rms_native_units": step_rms,
            "linearized_source_common_rms_native_units": source_rms_value,
            "linearized_source_common_max_absolute_native_units": maximum,
            "family_rms": v1.masked_family_rms(direction),
            "admissible": admissible,
        }
    eligible = [label for label, report in reports.items() if report["admissible"]]
    selected = (
        min(eligible, key=lambda label: reports[label]["first_order_loss_derivative"])
        if eligible
        else None
    )
    return selected, reports


def bounded_trial_report(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    direction: dict[str, Tensor],
    *,
    label: str,
    scale: float,
    derivative: float,
    baseline_loss: float,
    baseline_nrmse: float,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    actual = v1.set_masked_trial(
        student,
        source_parameters,
        edge_indices,
        bias_indices,
        direction,
        scale,
    )
    report = v1.evaluate_trial(
        student,
        source,
        previous,
        source_parameters,
        edge_indices,
        bias_indices,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        baseline_objective_nrmse=baseline_nrmse,
    )
    per_update_common = max(
        value["common_error_rms"]
        for value in report["functional_trust"]["per_update_pair"].values()
    )
    if per_update_common > PER_UPDATE_COMMON_RMS:
        report["pass"] = False
        report["reasons"].append("per-update common throttle RMS")
    report.update(
        {
            "candidate": label,
            "scale": scale,
            "masked_actual_family_rms": v1.masked_family_rms(actual),
            "per_update_common_throttle_rms": per_update_common,
            "predicted_first_order_loss_change": scale * derivative,
            "actual_loss_change": report["objective_loss"] - baseline_loss,
        }
    )
    return report


def main() -> int:
    args = parse_args()
    v1.validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy frequency must divide the 100 Hz physics rate")
    physics_steps = physics_hz // args.policy_hz
    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != graph_sha256:
        raise SystemExit("checkpoint and graph hashes do not match")
    if int(checkpoint.get("policy_hz", args.policy_hz)) != args.policy_hz:
        raise SystemExit("checkpoint and requested policy frequencies do not match")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(checkpoint["controller"])
    source.eval().requires_grad_(False)
    previous.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edge_indices_np, bias_indices_np, mask_manifest = v1.build_route_mask(args.graph, args.raw_dir)
    edge_indices = torch.from_numpy(edge_indices_np).to(device)
    bias_indices = torch.from_numpy(bias_indices_np).to(device)
    source_parameters = trust.clone_parameters(source)
    started = perf_counter()

    baseline_loss_tensor, baseline_motion = v1.motion_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    baseline_loss = float(baseline_loss_tensor.detach())
    gradient = v1.masked_gradient(baseline_loss_tensor, student, edge_indices, bias_indices)
    riesz = {name: -gradient[name] * gradient[name].numel() for name in v1.MASK_FAMILIES}
    raw, raw_scale = v1.cap_masked_displacement(riesz, v1.MASK_STEP_FAMILY_RMS_CAP)
    rows, residual, common_report = v1.common_constraint_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    equality_small, equality_projection = v1.project_masked_displacement(raw, rows, residual)
    equality, equality_rescale = scale_to_final_cap(equality_small, v1.MASK_STEP_FAMILY_RMS_CAP)

    candidates = {}
    bound_reports = {}
    baseline_edges = source_parameters["edge_magnitude"][edge_indices]
    for fraction in BLEND_FRACTIONS:
        label = f"equality_to_raw_{fraction:.2f}"
        mixed = {
            name: (1.0 - fraction) * equality[name] + fraction * raw[name]
            for name in v1.MASK_FAMILIES
        }
        mixed, _ = scale_to_final_cap(mixed, v1.MASK_STEP_FAMILY_RMS_CAP)
        mixed, bounds = apply_edge_bounds(mixed, baseline_edges)
        candidates[label] = mixed
        bound_reports[label] = bounds
    selected, screening = screen_candidates(candidates, gradient, rows, residual)
    for label in screening:
        screening[label]["parameter_bounds"] = bound_reports[label]

    equality_label = "equality_to_raw_0.00"
    equality_sanity = bounded_trial_report(
        student,
        source,
        previous,
        source_parameters,
        edge_indices,
        bias_indices,
        candidates[equality_label],
        label=equality_label,
        scale=1.0,
        derivative=screening[equality_label]["first_order_loss_derivative"],
        baseline_loss=baseline_loss,
        baseline_nrmse=baseline_motion["fixed_scale_nrmse"],
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    trust.load_parameters(student, source_parameters)

    trials = []
    accepted_scale = None
    if selected is not None:
        derivative = screening[selected]["first_order_loss_derivative"]
        for scale in v1.BACKTRACK_SCALES:
            trial = bounded_trial_report(
                student,
                source,
                previous,
                source_parameters,
                edge_indices,
                bias_indices,
                candidates[selected],
                label=selected,
                scale=scale,
                derivative=derivative,
                baseline_loss=baseline_loss,
                baseline_nrmse=baseline_motion["fixed_scale_nrmse"],
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
        "experiment": "variable-height-native-damping-route-bounded-preflight-v1",
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": accepted_scale is not None,
        "accepted_scale": accepted_scale,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        },
        "actor_contract_unchanged": True,
        "mask": mask_manifest,
        "protocol": {
            "same_route_mask_as_rejected_v1": True,
            "trainable_parameter_families": list(v1.MASK_FAMILIES),
            "time_constants_trainable": False,
            "motor_biases_trainable": False,
            "blend_fractions_equality_to_raw": list(BLEND_FRACTIONS),
            "final_selected_family_rms_cap": v1.MASK_STEP_FAMILY_RMS_CAP,
            "backtrack_scales": list(v1.BACKTRACK_SCALES),
        },
        "thresholds": {
            "per_update_common_throttle_rms_native_units": PER_UPDATE_COMMON_RMS,
            "source_common_throttle_rms_native_units": SOURCE_COMMON_RMS,
            "common_throttle_max_absolute_native_units": COMMON_MAX_ABSOLUTE,
            "minimum_motion_nrmse_improvement": v1.MINIMUM_MOTION_NRMSE_IMPROVEMENT,
            "height_contrast_source_ratio_range": list(v1.HEIGHT_CONTRAST_RATIO_RANGE),
            "masked_source_metric_radius_for_later_training": (v1.MASK_SOURCE_METRIC_RADIUS),
        },
        "baseline_motion": baseline_motion,
        "raw_direction_initial_cap_scale": raw_scale,
        "equality_projection": equality_projection,
        "equality_direction_rescale_to_final_cap": equality_rescale,
        "linearized_candidate_screening": screening,
        "selected_candidate": selected,
        "equality_rescaled_full_replay_sanity": equality_sanity,
        "trials": trials,
        "parameters_restored_max_absolute_error": restored_error,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass establishes only measurable safe local credit assignment on the unchanged "
            "shallow route. It neither reverses damping by itself nor promotes a controller."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    best = min(
        (trial["objective_motion"]["fixed_scale_nrmse"] for trial in [equality_sanity, *trials]),
        default=baseline_motion["fixed_scale_nrmse"],
    )
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": report["pass"],
                "selected_candidate": selected,
                "accepted_scale": accepted_scale,
                "baseline_motion_nrmse": baseline_motion["fixed_scale_nrmse"],
                "best_motion_nrmse": best,
                "parameters_restored_max_absolute_error": restored_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
