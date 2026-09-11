#!/usr/bin/env python3
"""Continue corrected fitting from update 24 with the qualified polished projector."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_update25_exhaustive as audit25  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_update25_slsqp_polished as polished  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402
import train_variable_height_full_native_d_first_fp64_corrected as continuation  # noqa: E402
import train_variable_height_full_native_d_first_fp64_snapshot_corrected as source_fit  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-polished-corrected-fitting-v1"
PROTOCOL_COMMIT = "d1e5c82"
EXPECTED_SOURCE_REPORT_SHA256 = audit25.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = audit25.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_FAILED_PRODUCER_REPORT_SHA256 = polished.EXPECTED_FAILED_PRODUCER_REPORT_SHA256
EXPECTED_FAILED_AUDIT_REPORT_SHA256 = polished.EXPECTED_FAILED_AUDIT_REPORT_SHA256
EXPECTED_FROZEN_ARCHIVE_SHA256 = polished.EXPECTED_FROZEN_ARCHIVE_SHA256
EXPECTED_POLISHED_REPORT_SHA256 = "9abe6a4b433927e6136395dbe7abec869bce8daccd5b03284fc91cb076f0e075"
EXPECTED_UPDATE24_PARAMETER_SHA256 = audit25.EXPECTED_SOURCE_PARAMETER_SHA256
EXPECTED_UPDATE24_OPTIMIZER_SHA256 = audit25.EXPECTED_SOURCE_OPTIMIZER_SHA256
EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256 = (
    "44646023930bf6734c2bfffd23c71a81c6a5e7ae089e9655cc1789d1a73575e0"
)
EXPECTED_UPDATE25_PROJECTED_DISPLACEMENT_SHA256 = (
    "56cf914132ffd85356d3c28fcccc1416e409ee9ed122e770820bffa4fcaeea44"
)
EXPECTED_UPDATE25_PROJECTION_OBJECTIVE = 18712.47294252837
EXPECTED_INITIAL_ACCEPTED_UPDATES = 24
FROZEN_UPDATE = 25
MAXIMUM_ACCEPTED_UPDATES = 50

_FROZEN_UPDATE25_CPU: dict[str, Any] | None = None
_FROZEN_REFERENCE: dict[str, Any] | None = None
_FROZEN_UPDATE25_CONTROL: dict[str, Any] | None = None
_UPDATE_NUMBER = EXPECTED_INITIAL_ACCEPTED_UPDATES

_ORIGINAL_BASE_VALUES = {
    "EXPERIMENT": base.EXPERIMENT,
    "PROTOCOL_COMMIT": base.PROTOCOL_COMMIT,
    "MAXIMUM_ACCEPTED_UPDATES": base.MAXIMUM_ACCEPTED_UPDATES,
    "DEVELOPMENT_PRESERVATION_REQUIRED": base.DEVELOPMENT_PRESERVATION_REQUIRED,
    "DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED": (base.DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED),
    "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": base.PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED,
    "parse_args": base.parse_args,
    "validate_args": base.validate_args,
    "protocol_manifest": base.protocol_manifest,
    "make_projected_proposal": base.make_projected_proposal,
    "install_trial": base.install_trial,
    "find_safe_trial": base.find_safe_trial,
}


def seed_provenance_manifest() -> dict[str, Any]:
    return {
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
        "frozen_update25_archive_sha256": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "polished_audit_report_sha256": EXPECTED_POLISHED_REPORT_SHA256,
        "rejected_update_25_removed_in_copy_only": True,
    }


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
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-snapshot-corrected-fitting-001"
        ),
    )
    parser.add_argument(
        "--update25-audit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update25-exhaustive-audit-001"
        ),
    )
    parser.add_argument(
        "--polished-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update25-slsqp-polished-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-polished-corrected-fitting-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def locked_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "graph": args.graph,
        "checkpoint": args.checkpoint,
        "cache_manifest": args.source_audit_dir / "immutable-cache-v1/manifest.json",
        "source_report": args.source_fit_dir / "report.json",
        "source_resume": args.source_fit_dir / "resume.pt",
        "failed_producer_report": args.update25_audit_dir / audit25.PRODUCER_REPORT_NAME,
        "failed_audit_report": args.update25_audit_dir / audit25.FINAL_REPORT_NAME,
        "frozen_update25_archive": args.update25_audit_dir / audit25.ARCHIVE_NAME,
        "polished_audit_report": args.polished_audit_report,
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": audit25.EXPECTED_GRAPH_SHA256,
        "checkpoint": audit25.EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "failed_producer_report": EXPECTED_FAILED_PRODUCER_REPORT_SHA256,
        "failed_audit_report": EXPECTED_FAILED_AUDIT_REPORT_SHA256,
        "frozen_update25_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "polished_audit_report": EXPECTED_POLISHED_REPORT_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in locked_input_paths(args).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered polished continuation has no smoke variant")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the polished continuation already has a final report")
    hashes = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if hashes != expected_locked_hashes():
        raise SystemExit("one or more hash-locked continuation inputs do not match")


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(source_fit.protocol_manifest())
    constraint_projection = copy.deepcopy(manifest["constraint_projection"])
    constraint_projection.update(
        {
            "solver": ("exhaustive active-support Lawson-Hanson nonnegative least squares"),
            "independent_reference": ("SLSQP-only support plus one full-rank FP64 SVD solve"),
            "lbfgsb_is_diagnostic_only": True,
            "old_projector_fallback": False,
        }
    )
    optimizer = copy.deepcopy(manifest["optimizer"])
    optimizer.update(
        {
            "adam_step_per_fresh_attempt_updates_26_plus": 1,
            "adam_step_executed_in_continuation_for_frozen_update_25": 0,
        }
    )
    optimizer.pop("adam_step_per_attempt", None)
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "starting_resume": {
                "source_experiment": source_fit.EXPERIMENT,
                "source_protocol_commit": source_fit.PROTOCOL_COMMIT,
                "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
                "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
                "accepted_updates": EXPECTED_INITIAL_ACCEPTED_UPDATES,
                "rejected_update_25_history_removed_in_copy_only": True,
                "source_overwritten": False,
            },
            "authorizing_polished_audit": {
                "report_sha256": EXPECTED_POLISHED_REPORT_SHA256,
                "frozen_archive_sha256": EXPECTED_FROZEN_ARCHIVE_SHA256,
                "required_pass": True,
            },
            "starting_accepted_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "constraint_projection": constraint_projection,
            "optimizer": optimizer,
            "update_25": {
                "proposal_inputs_loaded_from_frozen_archive": True,
                "constraint_or_gradient_regeneration": False,
                "adam_step_called_again": False,
                "optimizer_after_loaded_as_archived_pending_transaction": True,
                "optimizer_before_sha256": EXPECTED_UPDATE24_OPTIMIZER_SHA256,
                "optimizer_after_sha256": EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256,
                "projected_displacement_sha256": (EXPECTED_UPDATE25_PROJECTED_DISPLACEMENT_SHA256),
                "complete_projection_primal_objective": (EXPECTED_UPDATE25_PROJECTION_OBJECTIVE),
                "objective_relative_tolerance": (frozen.OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT),
                "canonical_and_post_materialization_gates_required": True,
                "directional_finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
                "ordinary_then_existing_eligible_repair": True,
                "cross_process_repaired_tensor_hash_required": False,
                "accepted_tensors_persisted_immediately": True,
            },
            "updates_26_through_50": {
                "fresh_native_gradient_and_one_adam_transaction": True,
                "primary_solver": (
                    "exhaustive active-support Lawson-Hanson nonnegative least squares"
                ),
                "independent_reference": ("SLSQP-only support plus one full-rank FP64 SVD solve"),
                "all_primary_reference_active_set_and_post_projection_gates_required": True,
                "old_projector_fallback": False,
            },
            "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
            "development_totals": [30, 40, 50],
            "development_preservation_failure_is_terminal": True,
            "development_preservation_failure_rolls_back_transaction": True,
            "mandatory_gate_total": 50,
            "large_archives_remain_ignored_not_git_or_lfs": True,
        }
    )
    for stale_key in (
        "authorizing_snapshot_qualification",
        "update_21",
        "before_update_22",
        "updates_22_through_50",
    ):
        manifest.pop(stale_key, None)
    return manifest


def validate_seed_history(payload: dict[str, Any]) -> None:
    preflight = payload.get("preflight", {})
    provenance = payload.get("seed_provenance")
    if not isinstance(provenance, dict) and isinstance(preflight, dict):
        provenance = preflight.get("polished_continuation_seed_provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    if (
        provenance.get("source_report_sha256"),
        provenance.get("source_resume_sha256"),
        provenance.get("frozen_update25_archive_sha256"),
        provenance.get("polished_audit_report_sha256"),
    ) != (
        EXPECTED_SOURCE_REPORT_SHA256,
        EXPECTED_SOURCE_RESUME_SHA256,
        EXPECTED_FROZEN_ARCHIVE_SHA256,
        EXPECTED_POLISHED_REPORT_SHA256,
    ):
        raise SystemExit("polished continuation seed provenance does not match")
    accepted_updates = int(payload.get("accepted_updates", -1))
    update25_entries = [
        entry for entry in payload.get("history", []) if int(entry.get("update", -1)) == 25
    ]
    if accepted_updates >= 25:
        if len(update25_entries) != 1:
            raise SystemExit("advanced resume lacks its unique update-25 record")
        control = update25_entries[0].get("projection", {}).get("frozen_update25_preflight")
        if (
            update25_entries[0].get("accepted") is not True
            or not isinstance(control, dict)
            or control.get("pass") is not True
        ):
            raise SystemExit("advanced resume lacks a passing frozen update-25 preflight")


def seed_resume(args: argparse.Namespace) -> int:
    output = args.output_dir / "resume.pt"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        if (payload.get("experiment"), payload.get("protocol_commit")) != (
            EXPERIMENT,
            PROTOCOL_COMMIT,
        ):
            raise SystemExit("existing polished continuation resume does not match")
        validate_seed_history(payload)
        return int(payload["accepted_updates"])
    source_path = args.source_fit_dir / "resume.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    if (
        source.get("experiment"),
        source.get("protocol_commit"),
        int(source.get("accepted_updates", -1)),
        source.get("run_state"),
    ) != (
        source_fit.EXPERIMENT,
        source_fit.PROTOCOL_COMMIT,
        EXPECTED_INITIAL_ACCEPTED_UPDATES,
        "stopped",
    ):
        raise SystemExit("source is not the registered stopped accepted-update-24 state")
    history = copy.deepcopy(source.get("history", []))
    if (
        not history
        or int(history[-1].get("update", -1)) != FROZEN_UPDATE
        or history[-1].get("accepted") is not False
    ):
        raise SystemExit("source resume lacks its rejected update-25 record")
    history.pop()
    if (
        not history
        or int(history[-1].get("update", -1)) != EXPECTED_INITIAL_ACCEPTED_UPDATES
        or history[-1].get("accepted") is not True
    ):
        raise SystemExit("source history does not end at accepted update 24")
    seeded = copy.deepcopy(source)
    seeded.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "accepted_updates": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "history": history,
            "development_history": [
                entry
                for entry in source.get("development_history", [])
                if int(entry.get("update", -1)) <= 20
            ],
            "run_state": "active",
            "qualification": None,
            "development_rollback": None,
            "seed_provenance": seed_provenance_manifest(),
        }
    )
    preflight = seeded.get("preflight")
    if not isinstance(preflight, dict) or not preflight.get("pass", False):
        raise SystemExit("source accepted-update-24 resume lacks a passing preflight")
    preflight["continued_from_qualified_polished_update25_reference"] = True
    preflight["polished_continuation_seed_provenance"] = seed_provenance_manifest()
    base.atomic_torch_save(seeded, output)
    if responsibility.file_sha256(source_path) != EXPECTED_SOURCE_RESUME_SHA256:
        raise SystemExit("source resume changed while polished continuation was seeded")
    return EXPECTED_INITIAL_ACCEPTED_UPDATES


def validate_authorizations(args: argparse.Namespace) -> None:
    global _FROZEN_UPDATE25_CPU, _FROZEN_REFERENCE

    polished_report = json.loads(args.polished_audit_report.read_text(encoding="utf-8"))
    if (
        polished_report.get("experiment"),
        polished_report.get("classification"),
        polished_report.get("pass"),
        polished_report.get("continuation_authorized"),
    ) != (
        polished.EXPERIMENT,
        "slsqp_support_polished_reference_qualified",
        True,
        True,
    ):
        raise SystemExit("polished reference audit does not authorize continuation")
    failed_report = json.loads(
        (args.update25_audit_dir / audit25.FINAL_REPORT_NAME).read_text(encoding="utf-8")
    )
    producer = json.loads(
        (args.update25_audit_dir / audit25.PRODUCER_REPORT_NAME).read_text(encoding="utf-8")
    )
    if (
        producer.get("experiment"),
        producer.get("classification"),
        producer.get("pass"),
        failed_report.get("experiment"),
        failed_report.get("classification"),
        failed_report.get("pass"),
    ) != (
        audit25.EXPERIMENT,
        "frozen_update_25_inputs_produced",
        True,
        audit25.EXPERIMENT,
        "one_or_more_exhaustive_primary_runs_failed",
        False,
    ):
        raise SystemExit("frozen update-25 producer or failed audit identity does not match")
    archive = torch.load(
        args.update25_audit_dir / audit25.ARCHIVE_NAME,
        map_location="cpu",
        weights_only=True,
    )
    semantic = audit25.archive_semantic_hashes(archive)
    if (
        archive.get("experiment"),
        archive.get("protocol_commit"),
        archive.get("accepted_updates_before"),
        archive.get("reconstructed_update"),
        semantic,
        audit.semantic_sha256(archive.get("current_parameters")),
        audit.semantic_sha256(archive.get("optimizer_before")),
        audit.semantic_sha256(archive.get("optimizer_after")),
    ) != (
        audit25.EXPERIMENT,
        audit25.PROTOCOL_COMMIT,
        EXPECTED_INITIAL_ACCEPTED_UPDATES,
        FROZEN_UPDATE,
        archive.get("tensor_semantic_sha256"),
        EXPECTED_UPDATE24_PARAMETER_SHA256,
        EXPECTED_UPDATE24_OPTIMIZER_SHA256,
        EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256,
    ):
        raise SystemExit("frozen update-25 archive identity or transaction does not match")
    _FROZEN_UPDATE25_CPU = archive
    _FROZEN_REFERENCE = {
        "source_training_metrics": copy.deepcopy(archive["source_training_metrics"]),
        "current_training_metrics": copy.deepcopy(archive["current_training_metrics"]),
    }


def frozen_update25_preflight(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    if _FROZEN_UPDATE25_CPU is None or _FROZEN_REFERENCE is None:
        raise RuntimeError("frozen update-25 authorization is unavailable")
    parameters = joint._copy_parameters(student)
    optimizer_state = optimizer.state_dict()
    parameters_cpu = audit25.to_cpu_tree(parameters)
    optimizer_state_cpu = audit25.to_cpu_tree(optimizer_state)
    archive_parameters = _FROZEN_UPDATE25_CPU["current_parameters"]
    archive_optimizer = _FROZEN_UPDATE25_CPU["optimizer_before"]
    source_replay = audit.numeric_tree_comparison(
        source_metrics, _FROZEN_REFERENCE["source_training_metrics"]
    )
    current_replay = audit.numeric_tree_comparison(
        current_metrics, _FROZEN_REFERENCE["current_training_metrics"]
    )
    report = {
        "pass": bool(
            audit.semantic_sha256(parameters) == EXPECTED_UPDATE24_PARAMETER_SHA256
            and audit.semantic_sha256(optimizer_state) == EXPECTED_UPDATE24_OPTIMIZER_SHA256
            and audit.trees_equal(parameters_cpu, archive_parameters)
            and audit.trees_equal(optimizer_state_cpu, archive_optimizer)
            and source_replay["pass"]
            and current_replay["pass"]
        ),
        "candidate_parameter_sha256": audit.semantic_sha256(parameters),
        "expected_candidate_parameter_sha256": EXPECTED_UPDATE24_PARAMETER_SHA256,
        "optimizer_before_sha256": audit.semantic_sha256(optimizer_state),
        "expected_optimizer_before_sha256": EXPECTED_UPDATE24_OPTIMIZER_SHA256,
        "archive_current_parameters_exact": audit.trees_equal(parameters_cpu, archive_parameters),
        "archive_optimizer_before_exact": audit.trees_equal(optimizer_state_cpu, archive_optimizer),
        "source_training_replay": source_replay,
        "current_training_replay": current_replay,
        "checked_before_update_25_projection": True,
        "fresh_proposal_inputs_generated": False,
        "adam_step_called": False,
    }
    return report


def _update25_directional_finite_difference(
    student: ConnectomeController,
    current_parameters: dict[str, Tensor],
    authoritative: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _, actual, idempotence = fp64.install_fp64_trial(
        student,
        current_parameters,
        authoritative,
        scale=joint.FINITE_DIFFERENCE_SCALE,
    )
    bounds = audit.parameter_bounds_report(joint._copy_parameters(student))
    metrics, _ = endpoint.evaluate(
        student,
        factorial_cache,
        attitude_cache,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    expected = canonical._dot_float64(raw_gradients, actual) / joint.FINITE_DIFFERENCE_SCALE
    actual_change = (
        metrics["endpoint_damping_nrmse"] ** 2 - current_metrics["endpoint_damping_nrmse"] ** 2
    ) / joint.FINITE_DIFFERENCE_SCALE
    relative = abs(actual_change - expected) / max(abs(actual_change), abs(expected), 1.0e-12)
    return {
        "pass": bool(
            math.isfinite(expected)
            and math.isfinite(actual_change)
            and abs(actual_change) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
            and expected < 0.0
            and actual_change < 0.0
            and relative <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            and idempotence["pass"]
            and bounds["pass"]
        ),
        "scale": joint.FINITE_DIFFERENCE_SCALE,
        "autograd_directional_derivative": expected,
        "complete_replay_finite_difference": actual_change,
        "relative_error": relative,
        "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
        "canonical_parameter_idempotence": idempotence,
        "parameter_bounds": bounds,
        "metrics": metrics,
        "nonlinear_scale_acceptance_required": False,
    }


def _projection_controls_pass(projection: dict[str, Any]) -> bool:
    return bool(
        projection.get("pass")
        and projection.get("rounds")
        and all(
            item.get("free_projection", {}).get("pass")
            and item.get("free_projection", {})
            .get("slsqp_support_polished_reference", {})
            .get("pass")
            for item in projection["rounds"]
        )
    )


def make_projected_proposal(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    global _FROZEN_UPDATE25_CONTROL, _UPDATE_NUMBER

    _UPDATE_NUMBER += 1
    corrected._UPDATE_NUMBER = _UPDATE_NUMBER
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    frozen_control: dict[str, Any] = {"required": False, "pass": True}
    if _UPDATE_NUMBER == FROZEN_UPDATE:
        frozen_control = frozen_update25_preflight(
            student, optimizer, source_metrics, current_metrics
        )
        frozen_control["required"] = True
        _FROZEN_UPDATE25_CONTROL = copy.deepcopy(frozen_control)
        if not frozen_control["pass"]:
            zero = {
                name: torch.zeros_like(getattr(student, name)) for name in joint.PARAMETER_FAMILIES
            }
            return (
                zero,
                zero,
                {
                    "pass": False,
                    "pass_after_parameter_bounds": False,
                    "reason": "frozen update-25 preflight failed",
                    "frozen_update25_preflight": frozen_control,
                    "optimizer_step_calls": 0,
                    "fresh_proposal_inputs_generated": False,
                },
            )
        if _FROZEN_UPDATE25_CPU is None:
            raise RuntimeError("frozen update-25 inputs disappeared after preflight")
        archived = audit25.to_device_tree(_FROZEN_UPDATE25_CPU, device)
        specs = archived["constraint_specs"]
        rows = archived["constraint_rows"]
        raw_gradients = archived["raw_damping_gradient"]
        raw_displacement = archived["raw_displacement"]
        gradient_norm = float(archived["gradient_norm_before_clipping"])
        optimizer.load_state_dict(archived["optimizer_after"])
    else:
        specs, rows, raw_gradients, gradient_norm = continuation._fresh_proposal_inputs(
            student,
            optimizer,
            source_metrics,
            current_metrics,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale,
            device=device,
        )
        raw_parameters = joint._copy_parameters(student)
        raw_displacement = {
            name: raw_parameters[name] - current_parameters[name]
            for name in joint.PARAMETER_FAMILIES
        }
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    optimizer_transaction = corrected.optimizer_step_transaction(optimizer_before, optimizer_after)
    proposed, projection = polished.polished_bound_aware_projection(
        raw_displacement, rows, specs, current_parameters["edge_magnitude"]
    )
    projection_control = _projection_controls_pass(projection)
    authoritative, effective64, canonicalization = canonical.materialize_authoritative_candidate(
        student, current_parameters, proposed
    )
    parameter_bounds = audit.parameter_bounds_report(authoritative)
    linearized = audit.linearized_constraint_violations(specs, rows, effective64)
    maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
    raw64 = {name: value.double() for name, value in raw_displacement.items()}
    raw_derivative = canonical._dot_float64(raw_gradients, raw64)
    effective_derivative = canonical._dot_float64(raw_gradients, effective64)
    update25_projection_reproduction: dict[str, Any] = {
        "required": False,
        "pass": True,
    }
    update25_fd: dict[str, Any] = {"required": False, "pass": True}
    if _UPDATE_NUMBER == FROZEN_UPDATE:
        displacement_sha256 = audit.semantic_sha256(proposed)
        objective = frozen.complete_projection_primal_objective(proposed, raw_displacement)
        objective_relative = abs(objective - EXPECTED_UPDATE25_PROJECTION_OBJECTIVE) / max(
            1.0, abs(EXPECTED_UPDATE25_PROJECTION_OBJECTIVE)
        )
        update25_projection_reproduction = {
            "required": True,
            "pass": bool(
                displacement_sha256 == EXPECTED_UPDATE25_PROJECTED_DISPLACEMENT_SHA256
                and objective_relative <= frozen.OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
            ),
            "projected_displacement_sha256": displacement_sha256,
            "expected_projected_displacement_sha256": (
                EXPECTED_UPDATE25_PROJECTED_DISPLACEMENT_SHA256
            ),
            "complete_projection_primal_objective": objective,
            "expected_complete_projection_primal_objective": (
                EXPECTED_UPDATE25_PROJECTION_OBJECTIVE
            ),
            "objective_relative_difference": objective_relative,
            "objective_relative_difference_limit": (
                frozen.OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
            ),
            "optimizer_after_sha256": audit.semantic_sha256(optimizer_after),
            "expected_optimizer_after_sha256": (EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256),
            "archived_pending_transaction": True,
            "adam_step_called_in_continuation": False,
        }
        update25_projection_reproduction["pass"] = bool(
            update25_projection_reproduction["pass"]
            and update25_projection_reproduction["optimizer_after_sha256"]
            == EXPECTED_UPDATE25_PENDING_OPTIMIZER_SHA256
        )
        update25_fd = _update25_directional_finite_difference(
            student,
            current_parameters,
            authoritative,
            raw_gradients,
            current_metrics,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale,
            device=device,
        )
        update25_fd["required"] = True
        frozen_control.update(
            {
                "projection_reproduction": copy.deepcopy(update25_projection_reproduction),
                "directional_finite_difference": copy.deepcopy(update25_fd),
            }
        )
        frozen_control["pass"] = bool(
            frozen_control["pass"]
            and update25_projection_reproduction["pass"]
            and update25_fd["pass"]
        )
        _FROZEN_UPDATE25_CONTROL = copy.deepcopy(frozen_control)

    projection.update(
        {
            "pass_after_parameter_bounds": bool(
                projection_control
                and canonicalization["pass"]
                and parameter_bounds["pass"]
                and linearized_finite
                and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
                and effective_derivative < 0.0
                and optimizer_transaction["pass"]
                and frozen_control["pass"]
                and update25_projection_reproduction["pass"]
                and update25_fd["pass"]
            ),
            "polished_projection_control_pass": projection_control,
            "authoritative_parameter_canonicalization": canonicalization,
            "authoritative_parameter_bounds": parameter_bounds,
            "bounded_linearized_violation_after": linearized,
            "maximum_bounded_linearized_violation_after": maximum_linearized,
            "raw_damping_directional_derivative": raw_derivative,
            "bounded_projected_damping_directional_derivative": effective_derivative,
            "damping_descent_fraction_retained": (
                effective_derivative / raw_derivative if raw_derivative < 0.0 else None
            ),
            "raw_displacement_family_rms": {
                name: float(value.double().square().mean().sqrt())
                for name, value in raw_displacement.items()
            },
            "bounded_projected_displacement_family_rms": {
                name: float(value.square().mean().sqrt()) for name, value in effective64.items()
            },
            "constraint_specs": specs,
            "gradient_norm_before_clipping": gradient_norm,
            "optimizer_transaction": optimizer_transaction,
            "frozen_update25_preflight": frozen_control,
            "update25_projection_reproduction": update25_projection_reproduction,
            "update25_directional_finite_difference": update25_fd,
            "effective_displacement_computed_in_float64": True,
        }
    )
    canonical._AUTHORITATIVE_BASE = current_parameters
    canonical._AUTHORITATIVE_CANDIDATE = authoritative
    corrected._PENDING_OPTIMIZER = optimizer
    corrected._OPTIMIZER_BEFORE_PROPOSAL = optimizer_before
    corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = audit.semantic_sha256(optimizer_after)
    joint._load_parameters(student, current_parameters)
    effective32 = {
        name: value.to(current_parameters[name].dtype) for name, value in effective64.items()
    }
    return effective32, raw_gradients, projection


def find_safe_trial(
    student: ConnectomeController,
    current_parameters: dict[str, Tensor],
    displacement: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
) -> tuple[float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    selected_scale, selected_metrics, trials = corrected.find_safe_trial(
        student,
        current_parameters,
        displacement,
        raw_gradients,
        source_metrics,
        current_metrics,
        factorial_cache,
        attitude_cache,
        scales,
        endpoint_scale,
        device=device,
    )
    if _UPDATE_NUMBER == FROZEN_UPDATE and trials:
        trials[-1]["frozen_update25_transaction"] = {
            "archived_pending_optimizer_loaded": True,
            "adam_step_called_in_continuation": False,
            "repair_tensor_cross_process_hash_required": False,
            "selected_tensors_persisted_by_outer_transaction": selected_metrics is not None,
        }
    return selected_scale, selected_metrics, trials


def main() -> int:
    global _FROZEN_REFERENCE, _FROZEN_UPDATE25_CONTROL, _FROZEN_UPDATE25_CPU, _UPDATE_NUMBER

    args = parse_args()
    validate_args(args)
    locked_before = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    globals_before = {
        "frozen_cpu": _FROZEN_UPDATE25_CPU,
        "frozen_reference": _FROZEN_REFERENCE,
        "frozen_control": _FROZEN_UPDATE25_CONTROL,
        "update_number": _UPDATE_NUMBER,
        "corrected_guard": corrected._GUARD_REPORT,
        "corrected_update": corrected._UPDATE_NUMBER,
        "corrected_optimizer": corrected._PENDING_OPTIMIZER,
        "corrected_optimizer_before": corrected._OPTIMIZER_BEFORE_PROPOSAL,
        "corrected_optimizer_after": corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
        "corrected_capture": corrected._REPAIR_TENSOR_CAPTURE,
        "canonical_base": canonical._AUTHORITATIVE_BASE,
        "canonical_candidate": canonical._AUTHORITATIVE_CANDIDATE,
    }
    run_frozen_control = None
    try:
        validate_authorizations(args)
        accepted_updates = seed_resume(args)
        _UPDATE_NUMBER = accepted_updates
        corrected._UPDATE_NUMBER = accepted_updates
        corrected._REPAIR_TENSOR_CAPTURE = None
        overrides = {
            "EXPERIMENT": EXPERIMENT,
            "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
            "MAXIMUM_ACCEPTED_UPDATES": MAXIMUM_ACCEPTED_UPDATES,
            "DEVELOPMENT_PRESERVATION_REQUIRED": True,
            "DEVELOPMENT_FAILURE_ROLLBACK_REQUIRED": True,
            "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": True,
            "parse_args": lambda: args,
            "validate_args": validate_args,
            "protocol_manifest": protocol_manifest,
            "make_projected_proposal": make_projected_proposal,
            "install_trial": canonical.install_trial,
            "find_safe_trial": find_safe_trial,
        }
        for name, value in overrides.items():
            setattr(base, name, value)
        result = base.main()
    finally:
        run_frozen_control = copy.deepcopy(_FROZEN_UPDATE25_CONTROL)
        for name, value in _ORIGINAL_BASE_VALUES.items():
            setattr(base, name, value)
        _FROZEN_UPDATE25_CPU = globals_before["frozen_cpu"]
        _FROZEN_REFERENCE = globals_before["frozen_reference"]
        _FROZEN_UPDATE25_CONTROL = globals_before["frozen_control"]
        _UPDATE_NUMBER = globals_before["update_number"]
        corrected._GUARD_REPORT = globals_before["corrected_guard"]
        corrected._UPDATE_NUMBER = globals_before["corrected_update"]
        corrected._PENDING_OPTIMIZER = globals_before["corrected_optimizer"]
        corrected._OPTIMIZER_BEFORE_PROPOSAL = globals_before["corrected_optimizer_before"]
        corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = globals_before["corrected_optimizer_after"]
        corrected._REPAIR_TENSOR_CAPTURE = globals_before["corrected_capture"]
        canonical._AUTHORITATIVE_BASE = globals_before["canonical_base"]
        canonical._AUTHORITATIVE_CANDIDATE = globals_before["canonical_candidate"]
    locked_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if locked_after != locked_before:
        raise RuntimeError("one or more locked source files changed during fitting")
    report_path = args.output_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if run_frozen_control is None:
            for entry in report.get("history", []):
                if int(entry.get("update", -1)) == FROZEN_UPDATE:
                    run_frozen_control = entry.get("projection", {}).get(
                        "frozen_update25_preflight"
                    )
                    break
        report["polished_continuation_provenance"] = {
            "locked_hashes_before": locked_before,
            "locked_hashes_after": locked_after,
            "all_locked_inputs_unchanged": True,
            "seeded_from_accepted_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "frozen_update25_preflight": run_frozen_control,
            "phase_maximum_total_update": MAXIMUM_ACCEPTED_UPDATES,
        }
        audit25.write_json(report_path, report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
