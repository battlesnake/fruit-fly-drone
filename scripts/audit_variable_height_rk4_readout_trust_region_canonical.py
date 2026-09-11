#!/usr/bin/env python3
"""Canonicalize one reconstructed readout direction, then test its upward trust region."""

from __future__ import annotations

import copy
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

import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import audit_variable_height_rk4_readout_trust_region as trust  # noqa: E402
import audit_variable_height_rk4_throttle_readout_step as readout  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-rk4-readout-trust-region-canonical-v1"
PROTOCOL_COMMIT = "3d8d1d9"
EXPECTED_FAILED_TRUST_REPORT_SHA256 = (
    "03c4ee348d51c351611b425de184a3d64cf3211f1a75ddf1c6f9e50006dcfc5e"
)
FAILED_TRUST_REPORT = REPO_ROOT / (
    "runs/variable-height-hover/native-rk4-readout-trust-region-001/report.json"
)
REPRODUCTION_ABSOLUTE_TOLERANCE = 2.0e-5


def parse_args():
    args = trust.parse_args()
    if args.output_dir == REPO_ROOT / (
        "runs/variable-height-hover/native-rk4-readout-trust-region-001"
    ):
        args.output_dir = REPO_ROOT / (
            "runs/variable-height-hover/native-rk4-readout-trust-region-canonical-001"
        )
    return args


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_failed_trust_report_sha256": EXPECTED_FAILED_TRUST_REPORT_SHA256,
        "locked_original_readout_report_sha256": trust.EXPECTED_READOUT_REPORT_SHA256,
        "scalar_and_functional_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "direction_archive": {
            "persist_before_candidate_replay": True,
            "physical_and_semantic_hashes": True,
            "cpu_reload_bit_exact": True,
            "regenerate_after_archive": False,
        },
        "old_scale_one_is_reproduction_only": True,
        "new_scales_descending": list(trust.TRUST_REGION_SCALES),
        "candidate_gates": "unchanged from the original readout-step audit",
        "candidate_retained": False,
        "training_hover_gate_or_promotion_authorized": False,
    }


def _parameter_families(state: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: state[name].detach().cpu().clone() for name in readout.PARAMETER_FAMILIES}


def _archive_without_hash(payload: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != "semantic_sha256"}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: _cpu_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def archive_payload(
    source: dict[str, Tensor],
    pending: dict[str, Tensor],
    optimizer_before: dict[str, Any],
    optimizer_after: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "source_parameters": _parameter_families(source),
        "pending_parameters": _parameter_families(pending),
        "optimizer_before": _cpu_tree(optimizer_before),
        "optimizer_after": _cpu_tree(optimizer_after),
    }
    payload["semantic_sha256"] = readout.assisted.audit.semantic_sha256(
        _archive_without_hash(payload)
    )
    return payload


def validate_archive(payload: dict[str, Any]) -> None:
    if payload.get("experiment") != EXPERIMENT:
        raise SystemExit("canonical direction archive experiment mismatch")
    if payload.get("semantic_sha256") != readout.assisted.audit.semantic_sha256(
        _archive_without_hash(payload)
    ):
        raise SystemExit("canonical direction archive semantic hash mismatch")
    if set(payload.get("source_parameters", {})) != set(readout.PARAMETER_FAMILIES):
        raise SystemExit("canonical archive source-family set mismatch")
    if set(payload.get("pending_parameters", {})) != set(readout.PARAMETER_FAMILIES):
        raise SystemExit("canonical archive pending-family set mismatch")


def install_parameter_families(
    template: dict[str, Tensor], families: dict[str, Tensor]
) -> dict[str, Tensor]:
    result = readout._clone_state_dict(template)
    for name in readout.PARAMETER_FAMILIES:
        result[name] = families[name].detach().cpu().clone()
    return result


def scalar_reproduction_decision(
    prior: dict[str, Any],
    *,
    baseline_values: list[float],
    gradient_objective: float,
    directional_derivative: float,
    optimizer_counters_before: list[float],
    optimizer_counters_after: list[float],
) -> dict[str, Any]:
    differences = {
        "baseline_objectives": [
            abs(actual - expected)
            for actual, expected in zip(
                baseline_values, prior["baseline_fixed_objectives"], strict=True
            )
        ],
        "gradient_objective": abs(gradient_objective - prior["gradient_objective"]),
        "directional_derivative": abs(
            directional_derivative - prior["full_proposal_directional_derivative"]
        ),
    }
    maximum = max(
        *differences["baseline_objectives"],
        differences["gradient_objective"],
        differences["directional_derivative"],
    )
    counters_pass = bool(
        optimizer_counters_before == [] and optimizer_counters_after == [1.0, 1.0, 1.0]
    )
    return {
        "pass": maximum <= REPRODUCTION_ABSOLUTE_TOLERANCE and counters_pass,
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "absolute_differences": differences,
        "maximum_absolute_difference": maximum,
        "optimizer_counters_before": optimizer_counters_before,
        "optimizer_counters_after": optimizer_counters_after,
        "optimizer_counters_pass": counters_pass,
    }


def _tensor_max_abs(left: Any, right: Any) -> float:
    left_tensor = torch.as_tensor(left, dtype=torch.float64)
    right_tensor = torch.as_tensor(right, dtype=torch.float64)
    if left_tensor.shape != right_tensor.shape:
        return float("inf")
    return float((left_tensor - right_tensor).abs().max())


def scale_one_functional_reproduction(
    previous: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    scalar_paths = {
        "fixed_nrmse": (previous["fixed"]["nrmse"], current["fixed"]["nrmse"]),
        "fixed_gain": (
            previous["fixed"]["teacher_aligned_gain"],
            current["fixed"]["teacher_aligned_gain"],
        ),
        "full_nrmse": (previous["full"]["nrmse"], current["full"]["nrmse"]),
        "full_gain": (
            previous["full"]["teacher_aligned_gain"],
            current["full"]["teacher_aligned_gain"],
        ),
        "fixed_improvement": (
            previous["fixed_nrmse_improvement"],
            current["fixed_nrmse_improvement"],
        ),
        "full_improvement": (
            previous["full_nrmse_improvement"],
            current["full_nrmse_improvement"],
        ),
        "common_rms": (
            previous["preservation"]["pair_common_throttle_drift_rms"],
            current["preservation"]["pair_common_throttle_drift_rms"],
        ),
        "common_max": (
            previous["preservation"]["pair_common_throttle_drift_maximum_absolute"],
            current["preservation"]["pair_common_throttle_drift_maximum_absolute"],
        ),
    }
    for axis in ("roll", "pitch", "yaw"):
        for metric in ("rms", "maximum_absolute"):
            scalar_paths[f"{axis}_{metric}"] = (
                previous["preservation"]["rpy_drift"][axis][metric],
                current["preservation"]["rpy_drift"][axis][metric],
            )
    scalar_differences = {
        name: abs(float(left) - float(right)) for name, (left, right) in scalar_paths.items()
    }
    tensor_differences = {
        "fixed_prediction_contrasts": _tensor_max_abs(
            previous["fixed"]["prediction_contrasts"],
            current["fixed"]["prediction_contrasts"],
        ),
        "fixed_terminal_motor_outputs": _tensor_max_abs(
            previous["fixed"]["terminal_motor_outputs"],
            current["fixed"]["terminal_motor_outputs"],
        ),
        "full_prediction_contrasts": _tensor_max_abs(
            previous["full"]["prediction_contrasts"],
            current["full"]["prediction_contrasts"],
        ),
        "full_terminal_motor_outputs": _tensor_max_abs(
            previous["full"]["terminal_motor_outputs"],
            current["full"]["terminal_motor_outputs"],
        ),
    }
    maximum = max(*scalar_differences.values(), *tensor_differences.values())
    controls_pass = bool(
        current["outside_mask_parameters_exact"]
        and current["native_bounds_pass"]
        and current["candidate_state_loaded_and_restored"]
        and current["full"]["endpoint_image_difference_max"] == 0.0
        and current["fixed"]["all_recurrent_states_and_outputs_finite"]
        and current["full"]["all_recurrent_states_and_outputs_finite"]
        and current["fixed"]["all_metrics_finite"]
        and current["full"]["all_metrics_finite"]
    )
    original_failure_preserved = not current["decision"]["pass"]
    return {
        "pass": bool(
            maximum <= REPRODUCTION_ABSOLUTE_TOLERANCE
            and controls_pass
            and original_failure_preserved
        ),
        "absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "maximum_absolute_difference": maximum,
        "scalar_absolute_differences": scalar_differences,
        "tensor_maximum_absolute_differences": tensor_differences,
        "controls_pass": controls_pass,
        "original_scale_one_failure_preserved": original_failure_preserved,
    }


def _write_start_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise SystemExit("canonical trust-region start marker exists; replay is forbidden") from exc


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit("the canonical readout trust-region repeat already has a report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes, locked, original_report = trust.validate_inputs(args)
    if (
        not FAILED_TRUST_REPORT.is_file()
        or readout.assisted.responsibility.file_sha256(FAILED_TRUST_REPORT)
        != EXPECTED_FAILED_TRUST_REPORT_SHA256
    ):
        raise SystemExit("failed trust-region report is missing or changed")
    with FAILED_TRUST_REPORT.open() as stream:
        failed_report = json.load(stream)
    if (
        failed_report.get("classification") != "direction_reproduction_failed"
        or failed_report.get("trust_region_candidates") != []
    ):
        raise SystemExit("locked failed trust-region report has unexpected contents")
    input_hashes[readout.assisted.responsibility.stable_path(FAILED_TRUST_REPORT)] = (
        EXPECTED_FAILED_TRUST_REPORT_SHA256
    )
    cache = locked["cache"]
    source_full = locked["solver_report"]["conditions"]["rk4_m1"]
    mask = readout.build_readout_mask(args.graph)
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_state = readout._clone_state_dict(loaded["controller"])
    source_sha256 = readout.assisted.audit.semantic_sha256(source_state)
    _write_start_marker(
        args.output_dir / "start.json",
        {
            "experiment": EXPERIMENT,
            "protocol": protocol_manifest(),
            "input_file_sha256": input_hashes,
            "source_state_sha256": source_sha256,
            "readout_mask": mask["manifest"],
            "device": str(device),
        },
    )
    started = perf_counter()
    controller = ConnectomeController(args.graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(source_state, strict=True)
    prefix_states = readout.source_prefix_states(controller, cache, device=device)
    baseline_values = []
    source_fixed = None
    print(json.dumps({"stage": "reconstructing_and_archiving_direction"}), flush=True)
    with torch.no_grad():
        for _ in range(3):
            objective, summary = readout.fixed_prefix_objective(
                controller, cache, prefix_states, device=device, backward=False
            )
            baseline_values.append(float(objective))
            source_fixed = summary
    assert source_fixed is not None
    optimizer = readout.make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = None
    gradient_objective_tensor, _ = readout.fixed_prefix_objective(
        controller, cache, prefix_states, device=device, backward=True
    )
    clipped_gradient, gradient_controls = readout._masked_gradients(controller, mask)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    controller.project_parameters()
    pending_state = readout._clone_state_dict(controller.state_dict())
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    displacement = readout.parameter_displacement(source_state, pending_state)
    directional_derivative = sum(
        float((clipped_gradient[name].detach().cpu().double() * displacement[name].double()).sum())
        for name in readout.PARAMETER_FAMILIES
    )
    scalar_reproduction = scalar_reproduction_decision(
        original_report,
        baseline_values=baseline_values,
        gradient_objective=float(gradient_objective_tensor),
        directional_derivative=directional_derivative,
        optimizer_counters_before=readout.assisted.optimizer_step_counters(optimizer_before),
        optimizer_counters_after=readout.assisted.optimizer_step_counters(optimizer_after),
    )
    archive = archive_payload(source_state, pending_state, optimizer_before, optimizer_after)
    archive_path = args.output_dir / "canonical-direction.pt"
    readout.assisted._atomic_torch_save(archive, archive_path)
    archive_file_sha256 = readout.assisted.responsibility.file_sha256(archive_path)
    reloaded = torch.load(archive_path, map_location="cpu", weights_only=True)
    validate_archive(reloaded)
    archive_reload_exact = readout.assisted.audit.trees_equal(archive, reloaded)
    canonical_source = install_parameter_families(source_state, reloaded["source_parameters"])
    canonical_pending = install_parameter_families(source_state, reloaded["pending_parameters"])
    controller.load_state_dict(canonical_source, strict=True)

    print(json.dumps({"stage": "scale_one_functional_reproduction"}), flush=True)
    scale_one_state = readout.materialize_candidate(canonical_source, canonical_pending, scale=1.0)
    scale_one_record = readout.evaluate_candidate(
        graph=args.graph,
        source_state=canonical_source,
        candidate_state=scale_one_state,
        cache=cache,
        prefix_states=prefix_states,
        source_fixed=source_fixed,
        source_full=source_full,
        mask=mask,
        device=device,
        scale=1.0,
        families=readout.PARAMETER_FAMILIES,
    )
    functional_reproduction = scale_one_functional_reproduction(
        original_report["ordinary_candidates"][0], scale_one_record
    )
    controls_pass = bool(
        scalar_reproduction["pass"]
        and archive_reload_exact
        and functional_reproduction["pass"]
        and gradient_controls["outside_mask_gradients_exactly_zero"]
    )

    candidates = []
    selected = None
    if controls_pass:
        for scale in trust.TRUST_REGION_SCALES:
            print(
                json.dumps({"stage": "trust_region_candidate", "scale": scale}),
                flush=True,
            )
            candidate_state = readout.materialize_candidate(
                canonical_source, canonical_pending, scale=scale
            )
            record = readout.evaluate_candidate(
                graph=args.graph,
                source_state=canonical_source,
                candidate_state=candidate_state,
                cache=cache,
                prefix_states=prefix_states,
                source_fixed=source_fixed,
                source_full=source_full,
                mask=mask,
                device=device,
                scale=scale,
                families=readout.PARAMETER_FAMILIES,
            )
            record["masked_displacement"] = trust.masked_displacement_report(
                canonical_source, candidate_state, mask
            )
            candidates.append(record)
            if record["decision"]["pass"]:
                selected = record
                break

    controller.load_state_dict(canonical_source, strict=True)
    optimizer.load_state_dict(reloaded["optimizer_before"])
    terminal_source_restored = rate._source_state_sha256(controller) == source_sha256
    terminal_optimizer_restored = readout.assisted.audit.trees_equal(
        optimizer.state_dict(), reloaded["optimizer_before"]
    )
    passed = bool(
        controls_pass
        and selected is not None
        and terminal_source_restored
        and terminal_optimizer_restored
    )
    classification = (
        "canonical_direct_readout_trust_region_exists"
        if passed
        else (
            "canonical_reproduction_control_failed"
            if not controls_pass
            else "no_canonical_direct_readout_trust_region"
        )
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha256,
        "readout_mask": mask["manifest"],
        "scalar_reproduction": scalar_reproduction,
        "gradient_controls": gradient_controls,
        "direction_archive": {
            "path": readout.assisted.responsibility.stable_path(archive_path),
            "file_sha256": archive_file_sha256,
            "semantic_sha256": reloaded["semantic_sha256"],
            "cpu_reload_bit_exact": archive_reload_exact,
        },
        "scale_one_reproduction_record": scale_one_record,
        "scale_one_functional_reproduction": functional_reproduction,
        "all_reproduction_controls_pass": controls_pass,
        "trust_region_candidates": candidates,
        "selected_scale": None if selected is None else selected["scale"],
        "selected_candidate_retained": False,
        "terminal_source_restored": terminal_source_restored,
        "terminal_optimizer_restored": terminal_optimizer_restored,
        "bounded_last_hop_fitting_preregistration_authorized": passed,
        "training_execution_authorized": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    readout.assisted._atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": readout.assisted.responsibility.stable_path(
                    args.output_dir / "report.json"
                ),
                "classification": classification,
                "pass": passed,
                "all_reproduction_controls_pass": controls_pass,
                "selected_scale": report["selected_scale"],
                "bounded_last_hop_fitting_preregistration_authorized": passed,
                "training_execution_authorized": False,
                "hover_or_gate_flight_authorized": False,
                "promotion_authorized": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
