#!/usr/bin/env python3
"""Audit one nonlinear feasibility correction of the rejected D-first update."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_bound_aware as bounded  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-nonlinear-correction-audit-v1"
PROTOCOL_COMMIT = "236671c"
SOURCE_EXPERIMENT = canonical.EXPERIMENT
EXPECTED_RESUME_SHA256 = "eb30a7c1d6f75ee65a6588dbe09475b66c172227170300248625baea1b15b148"
EXPECTED_SOURCE_REPORT_SHA256 = "bd7d27595d0cec185542e0ede5f873cdc3057252138ae7b700b6b5faa8e22e43"
EXPECTED_ACCEPTED_UPDATES = 8
STARTING_TRIAL_SCALE = 0.0625
CORRECTION_SCALES = (1.0, 0.5, 0.25, 0.125)
ENDPOINT_COMMON_INTERIOR_MARGIN = 0.0199
ENDPOINT_DAMPING_RETENTION_IMPROVEMENT = 0.001
SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN = ENDPOINT_COMMON_INTERIOR_MARGIN
USE_DISTINCT_SOLVER_ACCEPTANCE_SPECS = False
USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW = False
DIRECTIONAL_FINITE_DIFFERENCE_REQUIRED = False
DIRECTIONAL_FINITE_DIFFERENCE_PROBE_FRACTION = 0.25
DIRECTIONAL_FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE = 1.0e-8
REPRODUCTION_ABSOLUTE_TOLERANCE = 5.0e-6
REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-5
GRADIENT_ABSOLUTE_TOLERANCE = 1.0e-7
GRADIENT_RELATIVE_TOLERANCE = 1.0e-5


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
        "--source-audit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--source-fit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-canonical-fitting-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-nonlinear-correction-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.source_fit_dir / "resume.pt",
        args.source_fit_dir / "report.json",
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training/development cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered correction audit has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_resume_sha256": EXPECTED_RESUME_SHA256,
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "actor_inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
        "privileged_actor_inputs": [],
        "actor_contract_unchanged": True,
        "starting_candidate": {
            "proposal": "reconstructed rejected update-9 Adam proposal",
            "scale": STARTING_TRIAL_SCALE,
            "parameters_persisted_and_hashed_before_correction": True,
            "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
            "reproduction_relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        },
        "correction": {
            "objective": "minimum learning-rate-scaled Euclidean norm from zero",
            "jacobian_point": "actual nonlinear update-9 scale-0.0625 candidate",
            "common_and_height_reference": "original source",
            "endpoint_common_nrmse_maximum_source_plus": (ENDPOINT_COMMON_INTERIOR_MARGIN),
            "endpoint_damping_reference": "update 8",
            "endpoint_damping_nrmse_improvement_retained": (ENDPOINT_DAMPING_RETENTION_IMPROVEMENT),
            "limits_converted_to_squared_error": True,
            "rpy_activation_nrmse": base.RPY_CONSTRAINT_ACTIVATION_NRMSE,
            "metric": "Euclidean in per-family learning-rate-scaled coordinates",
            "edge_bound_handling": bounded.protocol_manifest()["constraint_projection"][
                "edge_bound_handling"
            ],
            "single_solve": True,
            "relinearization": False,
            "scales_descending": list(CORRECTION_SCALES),
            "first_passing_scale_selected": True,
            "linear_constraint_tolerance": base.LINEAR_CONSTRAINT_TOLERANCE,
            "canonical_parameter_idempotence_tolerance": (
                canonical.PARAMETER_IDEMPOTENCE_TOLERANCE
            ),
        },
        "development_diagnostic": {
            "controller": "fixed update-8 checkpoint",
            "bank": "already-exposed development bank",
            "minimum_endpoint_damping_nrmse_improvement": (ENDPOINT_DAMPING_RETENTION_IMPROVEMENT),
            "all_original_preservation_gates": True,
        },
        "pass_requires_training_correction_and_development_transfer": True,
        "parameters_and_optimizer_restored_exactly": True,
        "corrected_candidate_retained": False,
        "promotion": False,
        "closed_loop_hover": False,
        "pass_authorizes": "separately registered corrected fitting protocol only",
    }


def make_optimizer(controller: ConnectomeController) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {
                "params": [controller.edge_magnitude, controller.bias],
                "lr": joint.EDGE_BIAS_LEARNING_RATE,
            },
            {
                "params": [controller.raw_time_constant],
                "lr": joint.TIME_CONSTANT_LEARNING_RATE,
            },
        ],
        weight_decay=0.0,
    )


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(struct.pack("<Q", tensor.ndim))
        for size in tensor.shape:
            digest.update(struct.pack("<Q", size))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_digest(digest, key)
            _update_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_digest(digest, item)
    elif isinstance(value, bool):
        digest.update(b"bool\0" + (b"1" if value else b"0"))
    elif isinstance(value, int):
        digest.update(b"int\0" + str(value).encode() + b"\0")
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack("<d", value))
    elif isinstance(value, str):
        digest.update(b"str\0" + value.encode() + b"\0")
    elif value is None:
        digest.update(b"none\0")
    else:
        raise TypeError(f"unsupported digest value: {type(value)!r}")


def semantic_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def trees_equal(first: Any, second: Any) -> bool:
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        return (
            first.dtype == second.dtype
            and first.shape == second.shape
            and torch.equal(first, second)
        )
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            trees_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, type(first)):
        return len(first) == len(second) and all(
            trees_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    return type(first) is type(second) and first == second


@contextmanager
def transactional_restoration(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
) -> Iterator[tuple[dict[str, Tensor], dict[str, Any]]]:
    parameters = joint._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    authoritative_base = canonical._AUTHORITATIVE_BASE
    authoritative_candidate = canonical._AUTHORITATIVE_CANDIDATE
    try:
        yield parameters, optimizer_state
    finally:
        joint._load_parameters(controller, parameters)
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        canonical._AUTHORITATIVE_BASE = authoritative_base
        canonical._AUTHORITATIVE_CANDIDATE = authoritative_candidate


def numeric_tree_comparison(
    actual: Any,
    expected: Any,
    *,
    absolute_tolerance: float = REPRODUCTION_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = REPRODUCTION_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0

    def visit(left: Any, right: Any, path: str) -> None:
        nonlocal checked
        if isinstance(right, dict):
            if not isinstance(left, dict):
                failures.append({"path": path, "reason": "type mismatch"})
                return
            for key, value in right.items():
                if key not in left:
                    failures.append({"path": f"{path}.{key}", "reason": "missing"})
                else:
                    visit(left[key], value, f"{path}.{key}")
            return
        if isinstance(right, list):
            if not isinstance(left, list) or len(left) != len(right):
                failures.append({"path": path, "reason": "list shape mismatch"})
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                visit(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(right, bool) or right is None or isinstance(right, str):
            checked += 1
            if left != right:
                failures.append({"path": path, "actual": left, "expected": right})
            return
        if isinstance(right, (int, float)):
            checked += 1
            left_value = float(left)
            right_value = float(right)
            difference = abs(left_value - right_value)
            limit = absolute_tolerance + relative_tolerance * abs(right_value)
            if not math.isfinite(left_value) or difference > limit:
                failures.append(
                    {
                        "path": path,
                        "actual": left_value,
                        "expected": right_value,
                        "absolute_difference": difference,
                        "limit": limit,
                    }
                )
            return
        raise TypeError(f"unsupported comparison value at {path}: {type(right)!r}")

    visit(actual, expected, "root")
    return {
        "pass": not failures,
        "values_checked": checked,
        "failure_count": len(failures),
        "failures": failures[:32],
    }


def proposal_signature(projection: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass_after_parameter_bounds": projection["pass_after_parameter_bounds"],
        "maximum_bounded_linearized_violation_after": projection[
            "maximum_bounded_linearized_violation_after"
        ],
        "raw_damping_directional_derivative": projection["raw_damping_directional_derivative"],
        "bounded_projected_damping_directional_derivative": projection[
            "bounded_projected_damping_directional_derivative"
        ],
        "damping_descent_fraction_retained": projection["damping_descent_fraction_retained"],
        "gradient_norm_before_clipping": projection["gradient_norm_before_clipping"],
        "raw_displacement_family_rms": projection["raw_displacement_family_rms"],
        "bounded_projected_displacement_family_rms": projection[
            "bounded_projected_displacement_family_rms"
        ],
        "authoritative_parameter_canonicalization": {
            "pass": projection["authoritative_parameter_canonicalization"]["pass"],
            "second_direct_projection_maximum_parameter_change": projection[
                "authoritative_parameter_canonicalization"
            ]["second_direct_projection_maximum_parameter_change"],
        },
    }


def find_trial(history_entry: dict[str, Any], scale: float) -> dict[str, Any]:
    matches = [trial for trial in history_entry["trials"] if float(trial["scale"]) == scale]
    if len(matches) != 1:
        raise SystemExit(f"source report does not contain exactly one scale-{scale} trial")
    return matches[0]


def correction_constraint_specs(
    source_metrics: dict[str, Any],
    update8_metrics: dict[str, Any],
    starting_metrics: dict[str, Any],
    *,
    endpoint_common_margin: float = ENDPOINT_COMMON_INTERIOR_MARGIN,
) -> list[dict[str, Any]]:
    specs = base.constraint_specs(source_metrics, starting_metrics)
    endpoint_common = next(spec for spec in specs if spec["name"] == "common.step_25")
    endpoint_common["limit_mse"] = (
        source_metrics["by_supervision_step_nrmse"]["25"]["common"] + endpoint_common_margin
    ) ** 2
    endpoint_common["remaining_mse_allowance"] = (
        endpoint_common["limit_mse"] - endpoint_common["current_mse"]
    )
    damping_limit = (
        update8_metrics["endpoint_damping_nrmse"] - ENDPOINT_DAMPING_RETENTION_IMPROVEMENT
    )
    damping_spec = {
        "name": "damping.step_25.retention",
        "kind": "factorial",
        "component": "damping",
        "horizon_index": len(joint.SUPERVISION_STEPS) - 1,
        "current_mse": starting_metrics["endpoint_damping_nrmse"] ** 2,
        "limit_mse": damping_limit**2,
    }
    damping_spec["remaining_mse_allowance"] = (
        damping_spec["limit_mse"] - damping_spec["current_mse"]
    )
    specs.append(damping_spec)
    return specs


def directional_finite_difference_report(
    gradient: dict[str, Tensor],
    displacement: dict[str, Tensor],
    *,
    starting_endpoint_damping_nrmse: float,
    probe_endpoint_damping_nrmse: float,
    canonicalization_pass: bool,
    parameter_bounds_pass: bool,
    starting_restored_exactly: bool,
) -> dict[str, Any]:
    derivative = canonical._dot_float64(gradient, displacement)
    actual_change = probe_endpoint_damping_nrmse**2 - starting_endpoint_damping_nrmse**2
    finite = math.isfinite(derivative) and math.isfinite(actual_change)
    measurable = abs(actual_change) >= DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
    relative_error = (
        abs(actual_change - derivative) / max(abs(actual_change), abs(derivative), 1.0e-30)
        if finite
        else math.inf
    )
    return {
        "pass": bool(
            finite
            and measurable
            and relative_error <= DIRECTIONAL_FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            and canonicalization_pass
            and parameter_bounds_pass
            and starting_restored_exactly
        ),
        "autograd_directional_derivative": derivative,
        "complete_replay_squared_error_change": actual_change,
        "all_values_finite": finite,
        "measurable_nonzero_change": measurable,
        "minimum_absolute_change": (DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE),
        "relative_error": relative_error,
        "relative_error_limit": DIRECTIONAL_FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
        "probe_fraction_toward_update_8": DIRECTIONAL_FINITE_DIFFERENCE_PROBE_FRACTION,
        "canonicalization_pass": canonicalization_pass,
        "parameter_bounds_pass": parameter_bounds_pass,
        "starting_candidate_restored_exactly": starting_restored_exactly,
    }


def gradient_agreement(row: dict[str, Tensor], reference: dict[str, Tensor]) -> dict[str, Any]:
    maximum_absolute = max(
        float((row[name] - reference[name]).abs().max()) for name in joint.PARAMETER_FAMILIES
    )
    difference_norm = math.sqrt(
        sum(
            float((row[name].double() - reference[name].double()).square().sum())
            for name in joint.PARAMETER_FAMILIES
        )
    )
    reference_norm = math.sqrt(
        sum(float(reference[name].double().square().sum()) for name in joint.PARAMETER_FAMILIES)
    )
    reference_maximum_absolute = max(
        float(reference[name].abs().max()) for name in joint.PARAMETER_FAMILIES
    )
    maximum_absolute_limit = (
        GRADIENT_ABSOLUTE_TOLERANCE + GRADIENT_RELATIVE_TOLERANCE * reference_maximum_absolute
    )
    relative = difference_norm / max(reference_norm, 1.0e-30)
    return {
        "pass": (
            maximum_absolute <= maximum_absolute_limit and relative <= GRADIENT_RELATIVE_TOLERANCE
        ),
        "maximum_absolute_difference": maximum_absolute,
        "maximum_absolute_difference_limit": maximum_absolute_limit,
        "reference_maximum_absolute": reference_maximum_absolute,
        "absolute_tolerance": GRADIENT_ABSOLUTE_TOLERANCE,
        "relative_tolerance": GRADIENT_RELATIVE_TOLERANCE,
        "relative_l2_difference": relative,
        "relative_l2_difference_limit": GRADIENT_RELATIVE_TOLERANCE,
    }


def linearized_constraint_violations(
    specs: list[dict[str, Any]],
    rows: list[dict[str, Tensor]],
    displacement: dict[str, Tensor],
) -> list[float]:
    return [
        spec["current_mse"] - spec["limit_mse"] + canonical._dot_float64(row, displacement)
        for spec, row in zip(specs, rows, strict=True)
    ]


def maximum_linearized_violation(violations: list[float]) -> tuple[float, bool]:
    finite = all(math.isfinite(value) for value in violations)
    return (max(violations) if finite else math.inf), finite


def correction_projection_preplay_pass(
    *,
    projection_pass: bool,
    canonicalization_pass: bool,
    solver_linearized_finite: bool,
    maximum_solver_linearized_violation: float,
    acceptance_linearized_finite: bool,
    parameter_bounds_pass: bool,
    damping_row_control_pass: bool,
    distinct_solver_acceptance_specs: bool,
) -> bool:
    return bool(
        projection_pass
        and canonicalization_pass
        and solver_linearized_finite
        and acceptance_linearized_finite
        and parameter_bounds_pass
        and damping_row_control_pass
        and (
            distinct_solver_acceptance_specs
            or maximum_solver_linearized_violation <= base.LINEAR_CONSTRAINT_TOLERANCE
        )
    )


@torch.no_grad()
def install_authoritative_trial(
    controller: ConnectomeController,
    starting_parameters: dict[str, Tensor],
    corrected_parameters: dict[str, Tensor],
    *,
    scale: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    for name in joint.PARAMETER_FAMILIES:
        target = (
            corrected_parameters[name]
            if scale == 1.0
            else starting_parameters[name].double()
            + scale * (corrected_parameters[name].double() - starting_parameters[name].double())
        )
        getattr(controller, name).copy_(target.to(starting_parameters[name].dtype))
    controller.project_parameters()
    before_second_projection = joint._copy_parameters(controller)
    controller.project_parameters()
    maximum_second_change = max(
        float((getattr(controller, name).detach() - before_second_projection[name]).abs().max())
        for name in joint.PARAMETER_FAMILIES
    )
    actual_parameters = joint._copy_parameters(controller)
    actual_displacement = {
        name: actual_parameters[name].double() - starting_parameters[name].double()
        for name in joint.PARAMETER_FAMILIES
    }
    return (
        actual_parameters,
        actual_displacement,
        {
            "pass": maximum_second_change <= canonical.PARAMETER_IDEMPOTENCE_TOLERANCE,
            "second_direct_projection_maximum_parameter_change": maximum_second_change,
            "second_direct_projection_maximum_parameter_change_limit": (
                canonical.PARAMETER_IDEMPOTENCE_TOLERANCE
            ),
        },
    )


def correction_decision(
    source_metrics: dict[str, Any],
    update8_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    *,
    maximum_linearized_violation: float,
    canonicalization_pass: bool,
    parameter_bounds_pass: bool,
) -> dict[str, Any]:
    preservation = base.preservation_decision(source_metrics, candidate_metrics)
    reasons = list(preservation["reasons"])
    endpoint_common_limit = (
        source_metrics["by_supervision_step_nrmse"]["25"]["common"]
        + ENDPOINT_COMMON_INTERIOR_MARGIN
    )
    endpoint_common = candidate_metrics["by_supervision_step_nrmse"]["25"]["common"]
    if endpoint_common > endpoint_common_limit:
        reasons.append("endpoint common NRMSE exceeded original source plus 0.0199")
    damping_limit = (
        update8_metrics["endpoint_damping_nrmse"] - ENDPOINT_DAMPING_RETENTION_IMPROVEMENT
    )
    if candidate_metrics["endpoint_damping_nrmse"] > damping_limit:
        reasons.append("endpoint damping retained less than 0.001 improvement from update 8")
    if not math.isfinite(maximum_linearized_violation):
        reasons.append("actual scaled correction had a nonfinite linearized constraint")
    if maximum_linearized_violation > base.LINEAR_CONSTRAINT_TOLERANCE:
        reasons.append("actual scaled correction failed fixed linearized constraints")
    if not canonicalization_pass:
        reasons.append("actual scaled correction failed canonical parameter idempotence")
    if not parameter_bounds_pass:
        reasons.append("actual scaled correction failed native parameter bounds")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "preservation": preservation,
        "endpoint_common_nrmse": endpoint_common,
        "endpoint_common_nrmse_limit": endpoint_common_limit,
        "endpoint_damping_nrmse": candidate_metrics["endpoint_damping_nrmse"],
        "endpoint_damping_nrmse_limit": damping_limit,
        "endpoint_damping_nrmse_improvement_from_update_8": (
            update8_metrics["endpoint_damping_nrmse"] - candidate_metrics["endpoint_damping_nrmse"]
        ),
        "maximum_linearized_violation": maximum_linearized_violation,
        "maximum_linearized_violation_limit": base.LINEAR_CONSTRAINT_TOLERANCE,
        "canonicalization_pass": canonicalization_pass,
        "parameter_bounds_pass": parameter_bounds_pass,
    }


def development_transfer_decision(
    source_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, Any]:
    preservation = base.preservation_decision(source_metrics, candidate_metrics)
    improvement = (
        source_metrics["endpoint_damping_nrmse"] - candidate_metrics["endpoint_damping_nrmse"]
    )
    reasons = list(preservation["reasons"])
    if improvement < ENDPOINT_DAMPING_RETENTION_IMPROVEMENT:
        reasons.append("development endpoint damping NRMSE improvement was below 0.001")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "preservation": preservation,
        "endpoint_damping_nrmse_improvement": improvement,
        "minimum_endpoint_damping_nrmse_improvement": (ENDPOINT_DAMPING_RETENTION_IMPROVEMENT),
    }


def parameter_bounds_report(parameters: dict[str, Tensor]) -> dict[str, Any]:
    edges = parameters["edge_magnitude"]
    finite = all(bool(torch.isfinite(value).all()) for value in parameters.values())
    minimum = float(edges.min())
    maximum = float(edges.max())
    return {
        "pass": finite and minimum >= 0.0 and maximum <= 8.0,
        "all_parameters_finite": finite,
        "edge_magnitude_minimum": minimum,
        "edge_magnitude_maximum": maximum,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    config = HoverConfig()
    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint_sha256 = responsibility.file_sha256(args.checkpoint)
    resume_path = args.source_fit_dir / "resume.pt"
    source_report_path = args.source_fit_dir / "report.json"
    resume_sha256_before = responsibility.file_sha256(resume_path)
    source_report_sha256 = responsibility.file_sha256(source_report_path)
    if resume_sha256_before != EXPECTED_RESUME_SHA256:
        raise SystemExit("canonical update-8 resume hash does not match the protocol")
    if source_report_sha256 != EXPECTED_SOURCE_REPORT_SHA256:
        raise SystemExit("canonical fitting report hash does not match the protocol")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    expected_report_identity = (
        SOURCE_EXPERIMENT,
        canonical.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "deterministic constrained proposal had no acceptable scale",
    )
    actual_report_identity = (
        source_report.get("experiment"),
        source_report.get("protocol", {}).get("protocol_commit"),
        int(source_report.get("accepted_updates", -1)),
        source_report.get("stop_reason"),
    )
    if actual_report_identity != expected_report_identity:
        raise SystemExit("canonical fitting report identity does not match the protocol")
    registered_source_training = source_report["source_training"]
    registered_source_development = source_report["source_development"]
    registered_update8_training = source_report["final_training"]
    source_history_entry = source_report["history"][-1]
    if source_history_entry.get("update") != 9 or source_history_entry.get("accepted"):
        raise SystemExit("canonical fitting report does not end at rejected update 9")
    expected_starting_trial = find_trial(source_history_entry, STARTING_TRIAL_SCALE)

    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    optimizer = make_optimizer(student)

    resume = torch.load(resume_path, map_location=device, weights_only=True)
    expected_resume_identity = (
        SOURCE_EXPERIMENT,
        canonical.PROTOCOL_COMMIT,
        checkpoint_sha256,
        EXPECTED_ACCEPTED_UPDATES,
    )
    actual_resume_identity = (
        resume.get("experiment"),
        resume.get("protocol_commit"),
        resume.get("source_checkpoint_sha256"),
        int(resume.get("accepted_updates", -1)),
    )
    if actual_resume_identity != expected_resume_identity:
        raise SystemExit("resume identity does not match the frozen update-8 state")
    student.load_state_dict(resume["controller"])
    optimizer.load_state_dict(resume["optimizer"])
    optimizer_loaded_exactly = trees_equal(optimizer.state_dict(), resume["optimizer"])
    optimizer_semantic_sha256 = semantic_sha256(resume["optimizer"])
    update8_parameters = joint._copy_parameters(student)
    update8_parameter_sha256 = semantic_sha256(update8_parameters)
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())

    print(json.dumps({"stage": "loading_authorized_immutable_caches"}), flush=True)
    (
        train_factorial,
        train_attitude,
        development_factorial,
        development_attitude,
        cache_integrity,
    ) = endpoint.load_or_create_caches(
        args.source_audit_dir / "immutable-cache-v1",
        source=source,
        device=device,
        config=config,
        graph_sha256=graph_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    if cache_integrity["created_this_run"]:
        raise SystemExit("authorized cache was regenerated, contrary to protocol")
    if cache_integrity["manifest_sha256"] != base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256:
        raise SystemExit("authorized cache manifest hash does not match the protocol")
    endpoint_scale = endpoint.endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )

    print(json.dumps({"stage": "replaying_source_and_update_8"}), flush=True)
    source_training, source_training_raw = endpoint.evaluate(
        source,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    _, source_training_replay = endpoint.evaluate(
        source,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    source_replay_difference = joint._max_prediction_difference(
        source_training_raw, source_training_replay
    )
    source_development, _ = endpoint.evaluate(
        source,
        development_factorial,
        development_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    update8_training, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    source_training_reproduction = numeric_tree_comparison(
        source_training, source_report["source_training"]
    )
    source_development_reproduction = numeric_tree_comparison(
        source_development, source_report["source_development"]
    )
    update8_reproduction = numeric_tree_comparison(
        update8_training, source_report["final_training"]
    )

    reconstructed_proposal: dict[str, Any]
    starting_archive: dict[str, Any]
    correction: dict[str, Any]
    correction_trials: list[dict[str, Any]]
    selected_scale: float | None
    selected_metrics: dict[str, Any] | None
    development_metrics: dict[str, Any]
    development_decision: dict[str, Any]
    with transactional_restoration(student, optimizer):
        print(json.dumps({"stage": "reconstructing_rejected_update_9"}), flush=True)
        displacement, _, proposal = canonical.make_projected_proposal(
            student,
            optimizer,
            registered_source_training,
            registered_update8_training,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale,
            device=device,
        )
        expected_proposal = source_history_entry["projection"]
        full_proposal_reproduction = numeric_tree_comparison(proposal, expected_proposal)
        proposal_reproduction = numeric_tree_comparison(
            proposal_signature(proposal), proposal_signature(expected_proposal)
        )
        canonical.install_trial(
            student,
            update8_parameters,
            displacement,
            scale=STARTING_TRIAL_SCALE,
        )
        canonical._AUTHORITATIVE_BASE = None
        canonical._AUTHORITATIVE_CANDIDATE = None
        starting_metrics, _ = endpoint.evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        starting_reproduction = numeric_tree_comparison(
            starting_metrics, expected_starting_trial["metrics"]
        )
        starting_parameters = joint._copy_parameters(student)
        starting_parameter_sha256 = semantic_sha256(starting_parameters)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        starting_path = args.output_dir / "starting-candidate.pt"
        base.atomic_torch_save(
            {
                "experiment": EXPERIMENT,
                "protocol_commit": PROTOCOL_COMMIT,
                "source_resume_sha256": EXPECTED_RESUME_SHA256,
                "starting_scale": STARTING_TRIAL_SCALE,
                "parameter_tensor_sha256": starting_parameter_sha256,
                "parameters": {
                    name: value.detach().cpu() for name, value in starting_parameters.items()
                },
            },
            starting_path,
        )
        starting_file_sha256 = responsibility.file_sha256(starting_path)
        reloaded_starting = torch.load(starting_path, map_location="cpu", weights_only=True)
        starting_archive_reload_matches = bool(
            reloaded_starting.get("parameter_tensor_sha256") == starting_parameter_sha256
            and semantic_sha256(reloaded_starting["parameters"]) == starting_parameter_sha256
        )
        starting_archive = {
            "path": responsibility.stable_path(starting_path),
            "file_sha256": starting_file_sha256,
            "parameter_tensor_sha256": starting_parameter_sha256,
            "scale": STARTING_TRIAL_SCALE,
            "persisted_before_correction": True,
            "reload_matches_parameter_tensor_sha256": starting_archive_reload_matches,
        }
        reconstructed_proposal = {
            "full_proposal_reproduction": full_proposal_reproduction,
            "proposal_reproduction": proposal_reproduction,
            "starting_metrics_reproduction": starting_reproduction,
            "proposal": proposal,
            "starting_metrics": starting_metrics,
        }

        reconstruction_controls_pass = bool(
            cache_integrity["all_persisted_file_and_tensor_hashes_match"]
            and teacher_identity["pass"]
            and teacher_positive["pass"]
            and source_replay_difference <= joint.REPLAY_TOLERANCE
            and train_factorial.endpoint_image_difference_max == 0.0
            and development_factorial.endpoint_image_difference_max == 0.0
            and optimizer_loaded_exactly
            and source_training_reproduction["pass"]
            and source_development_reproduction["pass"]
            and update8_reproduction["pass"]
            and proposal_reproduction["pass"]
            and starting_reproduction["pass"]
            and starting_archive_reload_matches
        )
        correction_trials = []
        selected_scale = None
        selected_metrics = None
        if reconstruction_controls_pass:
            print(
                json.dumps({"stage": "differentiating_correction_constraints"}),
                flush=True,
            )
            solver_specs = correction_constraint_specs(
                registered_source_training,
                registered_update8_training,
                starting_metrics,
                endpoint_common_margin=SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN,
            )
            acceptance_specs = correction_constraint_specs(
                registered_source_training,
                registered_update8_training,
                starting_metrics,
                endpoint_common_margin=ENDPOINT_COMMON_INTERIOR_MARGIN,
            )
            rows = base.constraint_gradient_rows(
                student,
                train_factorial,
                train_attitude,
                scales,
                solver_specs,
                device=device,
            )
            damping_index = next(
                index
                for index, spec in enumerate(solver_specs)
                if spec["name"] == "damping.step_25.retention"
            )
            optimizer.zero_grad(set_to_none=True)
            endpoint.accumulated_endpoint_damping_gradient(
                student,
                train_factorial,
                scale=endpoint_scale,
                prefix_steps=joint.PREFIX_STEPS,
                device=device,
            )
            reference_damping_row = {
                name: getattr(student, name).grad.detach().clone()
                for name in joint.PARAMETER_FAMILIES
            }
            damping_gradient_agreement = gradient_agreement(
                rows[damping_index], reference_damping_row
            )
            optimizer.zero_grad(set_to_none=True)
            if USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW:
                rows[damping_index] = reference_damping_row

            if DIRECTIONAL_FINITE_DIFFERENCE_REQUIRED:
                probe_parameters, probe_displacement, probe_idempotence = (
                    install_authoritative_trial(
                        student,
                        starting_parameters,
                        update8_parameters,
                        scale=DIRECTIONAL_FINITE_DIFFERENCE_PROBE_FRACTION,
                    )
                )
                probe_bounds = parameter_bounds_report(probe_parameters)
                probe_metrics, _ = endpoint.evaluate(
                    student,
                    train_factorial,
                    train_attitude,
                    scales,
                    endpoint_scale=endpoint_scale,
                    device=device,
                )
                joint._load_parameters(student, starting_parameters)
                probe_starting_restored = all(
                    torch.equal(getattr(student, name).detach(), starting_parameters[name])
                    for name in joint.PARAMETER_FAMILIES
                )
                damping_directional_finite_difference = directional_finite_difference_report(
                    reference_damping_row,
                    probe_displacement,
                    starting_endpoint_damping_nrmse=starting_metrics["endpoint_damping_nrmse"],
                    probe_endpoint_damping_nrmse=probe_metrics["endpoint_damping_nrmse"],
                    canonicalization_pass=probe_idempotence["pass"],
                    parameter_bounds_pass=probe_bounds["pass"],
                    starting_restored_exactly=probe_starting_restored,
                )
                damping_directional_finite_difference.update(
                    {
                        "probe_metrics": probe_metrics,
                        "probe_parameter_bounds": probe_bounds,
                        "probe_canonical_parameter_idempotence": probe_idempotence,
                    }
                )
                damping_row_control = damping_directional_finite_difference
            else:
                damping_directional_finite_difference = {
                    "required": False,
                    "pass": None,
                }
                damping_row_control = damping_gradient_agreement

            print(
                json.dumps({"stage": "solving_single_minimum_norm_correction"}),
                flush=True,
            )
            zero = {name: torch.zeros_like(value) for name, value in starting_parameters.items()}
            proposed_correction, projection = bounded.bound_aware_project_inequality_displacement(
                zero, rows, solver_specs, starting_parameters["edge_magnitude"]
            )
            corrected_parameters, effective_correction, correction_canonicalization = (
                canonical.materialize_authoritative_candidate(
                    student, starting_parameters, proposed_correction
                )
            )
            full_solver_linearized = linearized_constraint_violations(
                solver_specs, rows, effective_correction
            )
            maximum_full_solver_linearized, full_solver_linearized_finite = (
                maximum_linearized_violation(full_solver_linearized)
            )
            full_acceptance_linearized = linearized_constraint_violations(
                acceptance_specs, rows, effective_correction
            )
            maximum_full_acceptance_linearized, full_acceptance_linearized_finite = (
                maximum_linearized_violation(full_acceptance_linearized)
            )
            full_bounds = parameter_bounds_report(corrected_parameters)
            correction_projection_pass = correction_projection_preplay_pass(
                projection_pass=projection["pass"],
                canonicalization_pass=correction_canonicalization["pass"],
                solver_linearized_finite=full_solver_linearized_finite,
                maximum_solver_linearized_violation=(maximum_full_solver_linearized),
                acceptance_linearized_finite=full_acceptance_linearized_finite,
                parameter_bounds_pass=full_bounds["pass"],
                damping_row_control_pass=damping_row_control["pass"],
                distinct_solver_acceptance_specs=(USE_DISTINCT_SOLVER_ACCEPTANCE_SPECS),
            )
            correction = {
                "attempted": True,
                "skip_reason": None,
                "solver_constraint_specs": solver_specs,
                "acceptance_constraint_specs": acceptance_specs,
                "active_rpy_rows": [
                    spec["name"] for spec in solver_specs if spec["kind"] == "attitude"
                ],
                "endpoint_damping_gradient_agreement": damping_gradient_agreement,
                "endpoint_damping_gradient_agreement_is_diagnostic_only": (
                    USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW
                ),
                "endpoint_damping_row_source": (
                    "dedicated accumulated endpoint-damping objective"
                    if USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW
                    else "multi-loss constraint Jacobian"
                ),
                "endpoint_damping_directional_finite_difference": (
                    damping_directional_finite_difference
                ),
                "endpoint_damping_row_control": damping_row_control,
                "zero_reference_displacement": True,
                "projection": projection,
                "authoritative_parameter_canonicalization": (correction_canonicalization),
                "effective_full_correction_solver_linearized_violations": (full_solver_linearized),
                "effective_full_correction_solver_linearized_violations_finite": (
                    full_solver_linearized_finite
                ),
                "maximum_effective_full_correction_solver_linearized_violation": (
                    maximum_full_solver_linearized
                ),
                "effective_full_correction_acceptance_linearized_violations": (
                    full_acceptance_linearized
                ),
                "effective_full_correction_acceptance_linearized_violations_finite": (
                    full_acceptance_linearized_finite
                ),
                "maximum_effective_full_correction_acceptance_linearized_violation": (
                    maximum_full_acceptance_linearized
                ),
                "effective_full_correction_parameter_bounds": full_bounds,
                "pass_before_nonlinear_replay": correction_projection_pass,
                "single_solve_completed": True,
                "relinearized": False,
            }

            print(json.dumps({"stage": "correction_scale_replay"}), flush=True)
            for scale in CORRECTION_SCALES:
                actual_parameters, actual_displacement, idempotence = install_authoritative_trial(
                    student,
                    starting_parameters,
                    corrected_parameters,
                    scale=scale,
                )
                linearized = linearized_constraint_violations(
                    acceptance_specs, rows, actual_displacement
                )
                maximum_linearized, linearized_finite = maximum_linearized_violation(linearized)
                bounds = parameter_bounds_report(actual_parameters)
                metrics, _ = endpoint.evaluate(
                    student,
                    train_factorial,
                    train_attitude,
                    scales,
                    endpoint_scale=endpoint_scale,
                    device=device,
                )
                decision = correction_decision(
                    registered_source_training,
                    registered_update8_training,
                    metrics,
                    maximum_linearized_violation=maximum_linearized,
                    canonicalization_pass=idempotence["pass"],
                    parameter_bounds_pass=bounds["pass"],
                )
                correction_trials.append(
                    {
                        "scale": scale,
                        "metrics": metrics,
                        "decision": decision,
                        "linearized_constraint_violations": linearized,
                        "linearized_constraint_violations_finite": linearized_finite,
                        "canonical_parameter_idempotence": idempotence,
                        "parameter_bounds": bounds,
                        "actual_displacement_family_rms": {
                            name: float(value.square().mean().sqrt())
                            for name, value in actual_displacement.items()
                        },
                    }
                )
                if correction_projection_pass and damping_row_control["pass"] and decision["pass"]:
                    selected_scale = scale
                    selected_metrics = metrics
                    break
        else:
            correction = {
                "attempted": False,
                "skip_reason": "reconstructed update-9 controls failed",
                "endpoint_damping_gradient_agreement": {
                    "pass": False,
                    "reason": "correction was not differentiated",
                },
                "endpoint_damping_row_control": {
                    "pass": False,
                    "reason": "correction was not differentiated",
                },
                "pass_before_nonlinear_replay": False,
                "single_solve_completed": False,
                "relinearized": False,
            }

        joint._load_parameters(student, update8_parameters)
        optimizer.load_state_dict(optimizer_snapshot)
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"stage": "fixed_update_8_development_diagnostic"}), flush=True)
        development_metrics, _ = endpoint.evaluate(
            student,
            development_factorial,
            development_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        development_decision = development_transfer_decision(
            registered_source_development, development_metrics
        )

    parameters_restored = all(
        torch.equal(getattr(student, name).detach(), update8_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_restored = trees_equal(optimizer.state_dict(), optimizer_snapshot)
    resume_sha256_after = responsibility.file_sha256(resume_path)
    resume_unchanged = resume_sha256_after == resume_sha256_before
    controls_pass = bool(
        cache_integrity["all_persisted_file_and_tensor_hashes_match"]
        and teacher_identity["pass"]
        and teacher_positive["pass"]
        and source_replay_difference <= joint.REPLAY_TOLERANCE
        and train_factorial.endpoint_image_difference_max == 0.0
        and development_factorial.endpoint_image_difference_max == 0.0
        and optimizer_loaded_exactly
        and source_training_reproduction["pass"]
        and source_development_reproduction["pass"]
        and update8_reproduction["pass"]
        and reconstructed_proposal["proposal_reproduction"]["pass"]
        and reconstructed_proposal["starting_metrics_reproduction"]["pass"]
        and starting_archive["reload_matches_parameter_tensor_sha256"]
        and correction["endpoint_damping_row_control"]["pass"]
        and parameters_restored
        and optimizer_restored
        and resume_unchanged
    )
    training_correction_pass = bool(
        correction["pass_before_nonlinear_replay"] and selected_scale is not None
    )
    passed = bool(controls_pass and training_correction_pass and development_decision["pass"])
    if not controls_pass:
        classification = "audit_control_failure"
    elif not development_decision["pass"]:
        classification = "fixed_update_8_failed_development_transfer"
    elif not training_correction_pass:
        classification = "no_fixed_scale_satisfied_nonlinear_correction_gates"
    else:
        classification = "nonlinear_correction_feasible_and_update_8_transfers"

    report = {
        "experiment": EXPERIMENT,
        "status": "diagnostic_only_no_retained_corrected_candidate",
        "pass": passed,
        "classification": classification,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "canonical_fit_directory": responsibility.stable_path(args.source_fit_dir),
            "canonical_fit_report_sha256": source_report_sha256,
            "update_8_resume": responsibility.stable_path(resume_path),
            "update_8_resume_sha256_before": resume_sha256_before,
            "update_8_resume_sha256_after": resume_sha256_after,
            "update_8_resume_unchanged": resume_unchanged,
        },
        "actor_contract_unchanged": True,
        "protocol": protocol_manifest(),
        "cache_integrity": cache_integrity,
        "teacher_component_scales_motor_units": scales,
        "endpoint_damping_scale_motor_units": endpoint_scale,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "source_replay_max_absolute_difference": source_replay_difference,
        "source_training": source_training,
        "source_development": source_development,
        "registered_source_training": registered_source_training,
        "registered_source_development": registered_source_development,
        "source_training_reproduction": source_training_reproduction,
        "source_development_reproduction": source_development_reproduction,
        "update_8": {
            "accepted_updates": EXPECTED_ACCEPTED_UPDATES,
            "metrics": update8_training,
            "metrics_reproduction": update8_reproduction,
            "parameter_tensor_sha256": update8_parameter_sha256,
            "optimizer_semantic_sha256": optimizer_semantic_sha256,
            "optimizer_loaded_exactly": optimizer_loaded_exactly,
            "registered_metrics": registered_update8_training,
        },
        "reconstructed_update_9": reconstructed_proposal,
        "starting_candidate_archive": starting_archive,
        "correction": correction,
        "correction_trials": correction_trials,
        "selected_correction_scale": selected_scale,
        "selected_correction_metrics": selected_metrics,
        "training_correction_pass": training_correction_pass,
        "fixed_update_8_development": {
            "metrics": development_metrics,
            "decision": development_decision,
        },
        "controls_pass": controls_pass,
        "parameters_restored_exactly_to_update_8": parameters_restored,
        "optimizer_restored_exactly_to_update_8": optimizer_restored,
        "corrected_fitting_protocol_authorized": passed,
        "corrected_candidate_retained": False,
        "promoted": False,
        "closed_loop_hover_run": False,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass establishes only one training-bank feasibility correction and "
            "development transfer of the uncorrected update-8 state. It authorizes a "
            "separately registered corrected fitting protocol, not promotion, stable "
            "hover, or flight."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": classification,
                "controls_pass": controls_pass,
                "training_correction_pass": training_correction_pass,
                "selected_correction_scale": selected_scale,
                "development_transfer_pass": development_decision["pass"],
                "parameters_restored_exactly_to_update_8": parameters_restored,
                "optimizer_restored_exactly_to_update_8": optimizer_restored,
                "corrected_fitting_protocol_authorized": passed,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
