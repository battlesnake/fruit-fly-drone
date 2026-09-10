#!/usr/bin/env python3
"""Preflight an upstream-expanded native route for visual vertical damping."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_damping_routes as shallow  # noqa: E402
import audit_variable_height_damping_routes_v2 as bounded  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_damping_routes as damping_train  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

REFERENCE_DENOMINATORS = {"edge_magnitude": 12_314, "bias": 303}
FINITE_DIFFERENCE_SCALE = 0.0625
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
EXPECTED_EDGE_INDICES_SHA256 = "a8d6118684cf64850b9e2d08f5f200c4501198fef6d4f5c52ec3aa349afa5aa3"
EXPECTED_BIAS_NODE_INDICES_SHA256 = (
    "f2cedb5b38e80beb9c29bba190ac9881c4cc473ea2849329b1f449c80402bd9b"
)

SHALLOW_REFERENCE = {
    "experiment": "variable-height-native-damping-route-bounded-preflight-v1",
    "full_report_sha256": "f4c79d1d561e2d63d1d02c40c4afc77300818fa3ada2451309308cf74379fc01",
    "seed": 260_923,
    "motion_nrmse_improvement": 0.00238204,
    "source_common_throttle_rms": 0.0004260292,
    "improvement_per_common_rms": 0.00238204 / 0.0004260292,
}


def parse_args() -> argparse.Namespace:
    parser = shallow.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=(REPO_ROOT / "runs/variable-height-hover/upstream-damping-route-preflight-001"),
        seed=260_923,
    )
    parser.add_argument("--mask-only", action="store_true")
    return parser.parse_args()


def expanded_route_mask_arrays(
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    superclass: np.ndarray,
    throttle_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Add all afferents and biases of descending neurons already feeding I or T."""

    base_edges, base_biases, base_counts = shallow.route_mask_arrays(
        edge_pre, edge_post, superclass, throttle_nodes
    )
    node_count = len(superclass)
    text = superclass.astype(str)
    descending = text == "descending_neuron"
    intermediate = np.zeros(node_count, dtype=bool)
    intermediate[base_biases] = True
    throttle = np.zeros(node_count, dtype=bool)
    throttle[throttle_nodes] = True
    feeds_route = descending[edge_pre] & (intermediate[edge_post] | throttle[edge_post])
    selected_descending = np.zeros(node_count, dtype=bool)
    selected_descending[edge_pre[feeds_route]] = True

    base_edge_mask = np.zeros(len(edge_pre), dtype=bool)
    base_edge_mask[base_edges] = True
    incoming = selected_descending[edge_post]
    added_edge_mask = incoming & ~base_edge_mask
    expanded_edges = np.flatnonzero(base_edge_mask | incoming)
    added_edges = np.flatnonzero(added_edge_mask)

    base_bias_mask = np.zeros(node_count, dtype=bool)
    base_bias_mask[base_biases] = True
    added_biases = np.flatnonzero(selected_descending & ~base_bias_mask)
    expanded_biases = np.flatnonzero(base_bias_mask | selected_descending)
    selected_descending_nodes = np.flatnonzero(selected_descending)
    return (
        expanded_edges,
        expanded_biases,
        added_edges,
        selected_descending_nodes,
        {
            "base": base_counts,
            "selected_descending_neurons": int(selected_descending.sum()),
            "selected_descending_already_in_base_intermediates": int(
                (selected_descending & base_bias_mask).sum()
            ),
            "added_incoming_edges": int(added_edge_mask.sum()),
            "added_descending_biases": int(len(added_biases)),
            "expanded_edges": int(len(expanded_edges)),
            "expanded_biases": int(len(expanded_biases)),
        },
    )


def build_expanded_route_mask(
    graph_path: Path, raw_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    graph = np.load(graph_path)
    annotations = _read_annotations(raw_dir / ANNOTATIONS_FILE)
    rows = {int(body): index for index, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray(
        [str(annotations["superclass"][rows[int(body)]]) for body in graph["node_ids"]]
    )
    offsets, pools = graph["output_pool_offsets"], graph["output_pool_indices"]
    throttle = pools[offsets[6] : offsets[8]]
    edges, biases, added_edges, descending, counts = expanded_route_mask_arrays(
        graph["edge_pre"], graph["edge_post"], superclass, throttle
    )
    added_names, added_counts = np.unique(
        superclass[graph["edge_pre"][added_edges]], return_counts=True
    )
    manifest = {
        "algorithm": (
            "start with the fixed shallow <=2-edge descending/VNC-intrinsic-to-throttle "
            "route; find every descending_neuron with an existing output to a shallow "
            "intermediate or throttle motor; add every existing edge ending at those "
            "descending neurons and add their biases; deduplicate all indices"
        ),
        "counts": counts,
        "reference_metric_denominators": REFERENCE_DENOMINATORS,
        "selected_edge_magnitudes": int(len(edges)),
        "selected_biases": int(len(biases)),
        "edge_indices_sha256": shallow.array_sha256(edges.astype(np.int64)),
        "bias_node_indices_sha256": shallow.array_sha256(biases.astype(np.int64)),
        "bias_body_ids_sha256": shallow.array_sha256(graph["node_ids"][biases].astype(np.int64)),
        "selected_descending_body_ids_sha256": shallow.array_sha256(
            graph["node_ids"][descending].astype(np.int64)
        ),
        "added_edge_presynaptic_superclass_counts": {
            str(name): int(count) for name, count in zip(added_names, added_counts, strict=True)
        },
        "edge_indices": edges.tolist(),
        "bias_node_indices": biases.tolist(),
        "bias_body_ids": graph["node_ids"][biases].tolist(),
        "selected_descending_body_ids": graph["node_ids"][descending].tolist(),
        "throttle_motor_body_ids": graph["node_ids"][throttle].tolist(),
    }
    return edges, biases, manifest


def validate_frozen_mask(manifest: dict[str, Any]) -> None:
    expected = (EXPECTED_EDGE_INDICES_SHA256, EXPECTED_BIAS_NODE_INDICES_SHA256)
    actual = (manifest["edge_indices_sha256"], manifest["bias_node_indices_sha256"])
    if actual != expected:
        raise SystemExit(f"expanded route mask changed: expected {expected}, got {actual}")


def reference_family_rms(values: dict[str, Tensor]) -> dict[str, float]:
    return {
        name: float((values[name].detach().square().sum() / REFERENCE_DENOMINATORS[name]).sqrt())
        for name in shallow.MASK_FAMILIES
    }


def reference_metric_norm(values: dict[str, Tensor]) -> float:
    rms = reference_family_rms(values)
    return sum(value * value for value in rms.values()) ** 0.5


def scale_to_reference_cap(
    displacement: dict[str, Tensor], cap: float
) -> tuple[dict[str, Tensor], float]:
    largest = max(reference_family_rms(displacement).values())
    if largest <= 0.0:
        return {name: value.clone() for name, value in displacement.items()}, 1.0
    scale = cap / largest
    return {name: scale * value for name, value in displacement.items()}, scale


def project_in_reference_metric(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    residual: Tensor,
    *,
    rtol: float = 1.0e-6,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if not rows or residual.shape != (len(rows),):
        raise ValueError("projection requires one residual per nonempty row")
    row_count = len(rows)
    gram = torch.empty(row_count, row_count, device=residual.device, dtype=torch.float64)
    j_delta = torch.empty(row_count, device=residual.device, dtype=torch.float64)
    for first, row in enumerate(rows):
        j_delta[first] = sum(
            (row[name] * displacement[name]).sum() for name in shallow.MASK_FAMILIES
        ).double()
        for second in range(first + 1):
            other = rows[second]
            value = sum(
                REFERENCE_DENOMINATORS[name] * (row[name] * other[name]).sum()
                for name in shallow.MASK_FAMILIES
            )
            gram[first, second] = value.double()
            gram[second, first] = value.double()
    diagonal = gram.diagonal().clamp_min(0.0).sqrt()
    active = diagonal > 1.0e-12
    if not bool(active.any()):
        raise RuntimeError("all expanded-mask common-output rows are numerically zero")
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    row_scale = diagonal[active]
    normalized = gram[active][:, active] / (row_scale[:, None] * row_scale[None])
    singular = torch.linalg.svdvals(normalized)
    rank = int((singular > rtol * singular.max()).sum())
    right_hand_side = (j_delta + residual.double())[active] / row_scale
    coefficient = torch.linalg.pinv(normalized, rtol=rtol) @ right_hand_side
    coefficient = coefficient / row_scale
    projected = {name: value.detach().clone() for name, value in displacement.items()}
    for value, row_index in zip(coefficient, active_indices, strict=True):
        row = rows[int(row_index)]
        for name in shallow.MASK_FAMILIES:
            projected[name].add_(row[name], alpha=-float(value) * REFERENCE_DENOMINATORS[name])
    after = residual.double().clone()
    for index, row in enumerate(rows):
        after[index] += sum(
            (row[name] * projected[name]).sum() for name in shallow.MASK_FAMILIES
        ).double()
    return projected, {
        "rows": row_count,
        "active_rows": int(active.sum()),
        "retained_rank": rank,
        "normalized_gram_singular_values": singular.detach().cpu().tolist(),
        "residual_before": residual.detach().cpu().tolist(),
        "projected_linearized_residual_max_absolute": float(after.abs().max()),
    }


def screen_candidates(
    candidates: dict[str, dict[str, Tensor]],
    gradient: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    residual: Tensor,
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    reports = {}
    current_source = 0.05 * residual
    for label, direction in candidates.items():
        predicted_source = bounded.linearized_common_native(direction, rows, residual)
        predicted_step = predicted_source - current_source
        derivative = float(
            sum((gradient[name] * direction[name]).sum() for name in shallow.MASK_FAMILIES)
        )
        step_rms = float(predicted_step.square().mean().sqrt())
        source_rms = float(predicted_source.square().mean().sqrt())
        source_max = float(predicted_source.abs().max())
        finite = damping_train.numeric_tree_is_finite(
            [derivative, step_rms, source_rms, source_max, direction]
        )
        admissible = (
            finite
            and derivative < 0.0
            and step_rms <= bounded.PER_UPDATE_COMMON_RMS
            and source_rms <= bounded.SOURCE_COMMON_RMS
            and source_max <= bounded.COMMON_MAX_ABSOLUTE
        )
        reports[label] = {
            "admissible": admissible,
            "all_finite": finite,
            "first_order_loss_derivative": derivative,
            "linearized_step_common_rms_native_units": step_rms,
            "linearized_source_common_rms_native_units": source_rms,
            "linearized_source_common_max_absolute_native_units": source_max,
            "reference_family_rms": reference_family_rms(direction),
        }
    eligible = [label for label, report in reports.items() if report["admissible"]]
    selected = (
        min(eligible, key=lambda label: reports[label]["first_order_loss_derivative"])
        if eligible
        else None
    )
    return selected, reports


def trial_report(
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
    baseline_complete_nrmse: float,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    actual = shallow.set_masked_trial(
        student, source_parameters, edge_indices, bias_indices, direction, scale
    )
    report = shallow.evaluate_trial(
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
    reference_norm = reference_metric_norm(actual)
    complete_improvement = (
        baseline_complete_nrmse - report["complete_zero_state_motion"]["fixed_scale_nrmse"]
    )
    finite = damping_train.numeric_tree_is_finite([report, actual, per_update_common])
    if complete_improvement < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT:
        report["pass"] = False
        report["reasons"].append(
            "complete zero-state motion NRMSE did not improve above the replay floor"
        )
    if per_update_common > bounded.PER_UPDATE_COMMON_RMS:
        report["pass"] = False
        report["reasons"].append("per-update common throttle RMS")
    if reference_norm > shallow.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        report["pass"] = False
        report["reasons"].append("fixed-denominator selected source metric radius")
    if not finite:
        report["pass"] = False
        report["reasons"].append("nonfinite preflight replay or parameter displacement")
    report.update(
        {
            "candidate": label,
            "scale": scale,
            "all_finite": finite,
            "reference_actual_family_rms": reference_family_rms(actual),
            "reference_actual_metric_norm": reference_norm,
            "complete_zero_state_motion_nrmse_improvement": complete_improvement,
            "per_update_common_throttle_rms": per_update_common,
            "predicted_first_order_loss_change": scale * derivative,
            "actual_loss_change": report["objective_loss"] - baseline_loss,
        }
    )
    return report


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
    loss, _ = shallow.motion_terms(
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
    finite = damping_train.numeric_tree_is_finite([measured, derivative, relative_error])
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


def main() -> int:
    args = parse_args()
    shallow.validate_args(args)
    edges_np, biases_np, manifest = build_expanded_route_mask(args.graph, args.raw_dir)
    validate_frozen_mask(manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_only:
        output = args.output_dir / "mask.json"
        output.write_text(f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": responsibility.stable_path(output),
                    "counts": manifest["counts"],
                    "edge_indices_sha256": manifest["edge_indices_sha256"],
                    "bias_node_indices_sha256": manifest["bias_node_indices_sha256"],
                    "added_edge_presynaptic_superclass_counts": manifest[
                        "added_edge_presynaptic_superclass_counts"
                    ],
                },
                indent=2,
            )
        )
        return 0

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
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(checkpoint["controller"])
    source.eval().requires_grad_(False)
    previous.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edge_indices = torch.from_numpy(edges_np).to(device)
    bias_indices = torch.from_numpy(biases_np).to(device)
    source_parameters = trust.clone_parameters(source)
    started = perf_counter()

    baseline_loss_tensor, baseline_motion = shallow.motion_terms(
        student,
        source,
        batch=args.batch_size,
        seed=args.seed + 10_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    baseline_loss = float(baseline_loss_tensor.detach())
    with torch.no_grad():
        _, baseline_complete_motion = shallow.motion_terms(
            source,
            source,
            batch=args.batch_size,
            seed=args.seed + 10_001,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
    gradient = shallow.masked_gradient(baseline_loss_tensor, student, edge_indices, bias_indices)
    riesz = {name: -gradient[name] * REFERENCE_DENOMINATORS[name] for name in shallow.MASK_FAMILIES}
    raw, raw_scale = scale_to_reference_cap(riesz, shallow.MASK_STEP_FAMILY_RMS_CAP)
    rows, residual, common_report = shallow.common_constraint_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    equality_small, projection = project_in_reference_metric(raw, rows, residual)
    equality, equality_scale = scale_to_reference_cap(
        equality_small, shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    candidates = {}
    bounds = {}
    current_edges = student.edge_magnitude[edge_indices].detach()
    for fraction in bounded.BLEND_FRACTIONS:
        label = f"equality_to_raw_{fraction:.2f}"
        mixed = {
            name: (1.0 - fraction) * equality[name] + fraction * raw[name]
            for name in shallow.MASK_FAMILIES
        }
        mixed, _ = scale_to_reference_cap(mixed, shallow.MASK_STEP_FAMILY_RMS_CAP)
        candidates[label], bounds[label] = bounded.apply_edge_bounds(mixed, current_edges)
    selected, screening = screen_candidates(candidates, gradient, rows, residual)
    for label in screening:
        screening[label]["parameter_bounds"] = bounds[label]

    finite_difference = None
    trials = []
    accepted_scale = None
    if selected is not None:
        derivative = screening[selected]["first_order_loss_derivative"]
        finite_difference = finite_difference_report(
            student,
            source,
            source_parameters,
            edge_indices,
            bias_indices,
            candidates[selected],
            derivative=derivative,
            baseline_loss=baseline_loss,
            args=args,
            config=config,
        )
        for scale in shallow.BACKTRACK_SCALES:
            trial = trial_report(
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
                baseline_complete_nrmse=baseline_complete_motion["fixed_scale_nrmse"],
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
    accepted_trial = next((trial for trial in trials if trial["pass"]), None)
    comparison = dict(SHALLOW_REFERENCE)
    if accepted_trial is not None:
        improvement = accepted_trial["objective_nrmse_improvement"]
        common_rms = accepted_trial["source_common_throttle_rms"]
        comparison["upstream_expanded"] = {
            "motion_nrmse_improvement": improvement,
            "source_common_throttle_rms": common_rms,
            "improvement_per_common_rms": improvement / max(common_rms, 1.0e-12),
        }
    report = {
        "experiment": "variable-height-native-upstream-damping-route-preflight-v1",
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
        "mask": manifest,
        "metric": {
            "definition": (
                "sqrt(sum(expanded edge delta^2)/12314 + sum(expanded bias delta^2)/303)"
            ),
            "reference_denominators": REFERENCE_DENOMINATORS,
            "step_family_rms_cap": shallow.MASK_STEP_FAMILY_RMS_CAP,
            "source_metric_radius_for_later_training": shallow.MASK_SOURCE_METRIC_RADIUS,
        },
        "frozen": {
            "retinal_mapping_and_input_gains": True,
            "optic_neuron_biases": True,
            "all_neuron_time_constants": True,
            "motor_biases": True,
            "topology": True,
            "transmitter_signs": True,
            "optic_to_descending_synaptic_magnitudes_may_change": True,
        },
        "protocol": {
            "seed": args.seed,
            "motion_pairs": args.batch_size,
            "constraint_pairs_per_height_amplitude": args.constraint_batch_size,
            "history_steps": args.history_steps,
            "policy_hz": args.policy_hz,
            "blend_fractions": list(bounded.BLEND_FRACTIONS),
            "backtrack_scales": list(shallow.BACKTRACK_SCALES),
            "finite_difference_scale": FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_limit": (FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT),
        },
        "thresholds": {
            "minimum_motion_nrmse_improvement": shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT,
            "per_update_common_throttle_rms": bounded.PER_UPDATE_COMMON_RMS,
            "source_common_throttle_rms": bounded.SOURCE_COMMON_RMS,
            "source_common_throttle_max_absolute": bounded.COMMON_MAX_ABSOLUTE,
            "height_contrast_source_ratio_range": list(shallow.HEIGHT_CONTRAST_RATIO_RANGE),
        },
        "baseline_motion": baseline_motion,
        "baseline_complete_zero_state_motion": baseline_complete_motion,
        "raw_direction_initial_cap_scale": raw_scale,
        "equality_projection": projection,
        "equality_direction_rescale_to_final_cap": equality_scale,
        "common_constraints": common_report,
        "linearized_candidate_screening": screening,
        "selected_candidate": selected,
        "finite_difference": finite_difference,
        "trials": trials,
        "shallow_route_comparison": comparison,
        "parameters_restored_max_absolute_error": restored_error,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass establishes only safe local credit assignment in the expanded native "
            "subspace. It does not identify a biological damping module or promote a controller."
        ),
    }
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": report["pass"],
                "selected_candidate": selected,
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
