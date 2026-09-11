#!/usr/bin/env python3
"""Continue corrected D-first fitting with the qualified FP64 primary projection."""

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

import audit_variable_height_full_native_d_first_fp64_corrected_step as step_audit  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-corrected-fitting-v1"
PROTOCOL_COMMIT = "ba6fcc3"
EXPECTED_SOURCE_REPORT_SHA256 = step_audit.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = step_audit.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_QUALIFICATION_REPORT_SHA256 = step_audit.EXPECTED_QUALIFICATION_REPORT_SHA256
EXPECTED_FROZEN_ARCHIVE_SHA256 = step_audit.EXPECTED_FROZEN_ARCHIVE_SHA256
EXPECTED_STEP_AUDIT_REPORT_SHA256 = (
    "8f76e9b5bd9cb1cecc782b207c3694b332926fb9f34b5be1fea8fd31b2deb6dd"
)
EXPECTED_INITIAL_ACCEPTED_UPDATES = 20
MAXIMUM_ACCEPTED_UPDATES = 50
EXPECTED_UPDATE_21_PARAMETER_SHA256 = (
    "3d8d96f36944113ad942ebd1d30f6510998a3a337881b03e0ce3e6032e0f7308"
)

_FROZEN_ARCHIVE_CPU: dict[str, Any] | None = None
_STEP_AUDIT_REPORT: dict[str, Any] | None = None
_UPDATE_NUMBER = EXPECTED_INITIAL_ACCEPTED_UPDATES

_ORIGINAL_BASE_VALUES = {
    "EXPERIMENT": base.EXPERIMENT,
    "PROTOCOL_COMMIT": base.PROTOCOL_COMMIT,
    "MAXIMUM_ACCEPTED_UPDATES": base.MAXIMUM_ACCEPTED_UPDATES,
    "DEVELOPMENT_PRESERVATION_REQUIRED": base.DEVELOPMENT_PRESERVATION_REQUIRED,
    "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": (base.PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED),
    "parse_args": base.parse_args,
    "validate_args": base.validate_args,
    "protocol_manifest": base.protocol_manifest,
    "make_projected_proposal": base.make_projected_proposal,
    "install_trial": base.install_trial,
    "find_safe_trial": base.find_safe_trial,
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
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-corrected-fitting-001"
        ),
    )
    parser.add_argument(
        "--qualification-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fp64-frozen-input-audit-001"
        ),
    )
    parser.add_argument(
        "--step-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-corrected-step-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fp64-corrected-fitting-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _locked_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "source_report": args.source_fit_dir / "report.json",
        "source_resume": args.source_fit_dir / "resume.pt",
        "qualification_report": args.qualification_dir / "report.json",
        "frozen_input_archive": args.qualification_dir / "frozen-inputs.pt",
        "step_audit_report": args.step_audit_report,
    }


def _expected_locked_hashes() -> dict[str, str]:
    return {
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "qualification_report": EXPECTED_QUALIFICATION_REPORT_SHA256,
        "frozen_input_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "step_audit_report": EXPECTED_STEP_AUDIT_REPORT_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, *_locked_input_paths(args).values()):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training/development cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered fitting continuation has no smoke variant")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the FP64 corrected fitting continuation already has a final report")
    actual = {
        name: responsibility.file_sha256(path) for name, path in _locked_input_paths(args).items()
    }
    if actual != _expected_locked_hashes():
        raise SystemExit("one or more hash-locked continuation inputs do not match")


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(corrected.protocol_manifest())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "starting_resume": {
                "experiment": corrected.EXPERIMENT,
                "protocol_commit": corrected.PROTOCOL_COMMIT,
                "sha256": EXPECTED_SOURCE_RESUME_SHA256,
                "accepted_updates": EXPECTED_INITIAL_ACCEPTED_UPDATES,
                "copied_to_new_run": True,
                "source_overwritten": False,
                "terminal_rejected_update_removed_from_seed_history": 21,
            },
            "authorizing_fp64_qualification": {
                "report_sha256": EXPECTED_QUALIFICATION_REPORT_SHA256,
                "archive_sha256": EXPECTED_FROZEN_ARCHIVE_SHA256,
                "required_pass": True,
            },
            "authorizing_step_audit": {
                "report_sha256": EXPECTED_STEP_AUDIT_REPORT_SHA256,
                "required_pass": True,
                "selected_parameter_sha256": EXPECTED_UPDATE_21_PARAMETER_SHA256,
            },
            "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
            "accepted_update_budget_is_total_not_additional": True,
            "development_totals": [30, 40, 50],
            "development_preservation_failure_is_terminal": True,
            "scheduled_candidate_persisted_before_development": True,
            "numerical_exceptions_restore_and_persist_stopped_resume": True,
            "mandatory_gate_total": base.MANDATORY_GATE_UPDATE,
            "update_21": {
                "proposal_source": "qualified frozen archive",
                "jacobians_regenerated": False,
                "pending_adam_reconstructed_from_archived_gradient": True,
                "exact_raw_projection_and_candidate_reproduction_required": True,
            },
            "updates_22_through_50": {
                "proposal_inputs_generated_once_per_attempt": True,
                "primary_projection": "qualified FP64 active-set projection",
                "primary_solver_success_required": True,
                "original_unit_primal_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
                "normalized_kkt_residual_maximum": fp64.NORMALIZED_KKT_RESIDUAL_LIMIT,
                "independent_exhaustive_support_check_required": True,
                "maximum_active_set_rounds": fp64.MAXIMUM_ACTIVE_SET_ROUNDS,
                "old_projector_fallback": False,
            },
            "repair_arithmetic_unchanged": True,
            "phase_exit": {
                "mandatory_25_percent_training_and_development_improvement_at_50": True,
                "fresh_qualification_only_after_terminal_checkpoint": True,
                "closed_loop_without_fresh_qualification": False,
                "automatic_promotion": False,
            },
        }
    )
    return manifest


def seed_resume(args: argparse.Namespace) -> int:
    output = args.output_dir / "resume.pt"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        identity = (
            payload.get("experiment"),
            payload.get("protocol_commit"),
            int(payload.get("accepted_updates", -1)),
        )
        if identity[0:2] != (EXPERIMENT, PROTOCOL_COMMIT):
            raise SystemExit("existing continuation resume does not match the protocol")
        return identity[2]
    source_path = args.source_fit_dir / "resume.pt"
    payload = torch.load(source_path, map_location="cpu", weights_only=True)
    identity = (
        payload.get("experiment"),
        payload.get("protocol_commit"),
        int(payload.get("accepted_updates", -1)),
        payload.get("run_state"),
    )
    if identity != (
        corrected.EXPERIMENT,
        corrected.PROTOCOL_COMMIT,
        EXPECTED_INITIAL_ACCEPTED_UPDATES,
        "stopped",
    ):
        raise SystemExit("source resume is not the registered stopped update-20 state")
    seeded = copy.deepcopy(payload)
    seeded.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "history": [
                entry
                for entry in payload["history"]
                if bool(entry.get("accepted"))
                and int(entry.get("update", -1)) <= EXPECTED_INITIAL_ACCEPTED_UPDATES
            ],
            "development_history": [
                entry
                for entry in payload["development_history"]
                if int(entry.get("update", -1)) <= EXPECTED_INITIAL_ACCEPTED_UPDATES
            ],
            "run_state": "active",
            "qualification": None,
            "seed_provenance": {
                "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
                "qualification_report_sha256": EXPECTED_QUALIFICATION_REPORT_SHA256,
                "frozen_input_archive_sha256": EXPECTED_FROZEN_ARCHIVE_SHA256,
                "step_audit_report_sha256": EXPECTED_STEP_AUDIT_REPORT_SHA256,
            },
        }
    )
    preflight = seeded.get("preflight")
    if not isinstance(preflight, dict) or not preflight.get("pass", False):
        raise SystemExit("source update-20 resume lacks its passing preflight")
    preflight["continued_from_passing_corrected_fit_preflight"] = True
    base.atomic_torch_save(seeded, output)
    if responsibility.file_sha256(source_path) != EXPECTED_SOURCE_RESUME_SHA256:
        raise SystemExit("source resume changed while continuation was seeded")
    return EXPECTED_INITIAL_ACCEPTED_UPDATES


def _optimizer_step_from_gradient(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    gradients: dict[str, Tensor],
) -> tuple[dict[str, Tensor], float]:
    optimizer.zero_grad(set_to_none=True)
    for name in joint.PARAMETER_FAMILIES:
        getattr(student, name).grad = gradients[name].to(getattr(student, name).device).clone()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    )
    optimizer.step()
    student.project_parameters()
    return joint._copy_parameters(student), gradient_norm


def _fresh_proposal_inputs(
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
) -> tuple[list[dict[str, Any]], list[dict[str, Tensor]], dict[str, Tensor], float]:
    specs = base.constraint_specs(source_metrics, current_metrics)
    print(json.dumps({"stage": "fp64_constraint_jacobian", "rows": len(specs)}), flush=True)
    rows = base.constraint_gradient_rows(
        student, factorial_cache, attitude_cache, scales, specs, device=device
    )
    optimizer.zero_grad(set_to_none=True)
    endpoint.accumulated_endpoint_damping_gradient(
        student,
        factorial_cache,
        scale=endpoint_scale,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    _, gradient_norm = _optimizer_step_from_gradient(student, optimizer, gradients)
    return specs, rows, gradients, gradient_norm


def _update_21_inputs(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    current_parameters: dict[str, Tensor],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Tensor]],
    dict[str, Tensor],
    dict[str, Tensor],
    float,
    dict[str, Any],
]:
    if _FROZEN_ARCHIVE_CPU is None:
        raise RuntimeError("frozen update-21 archive is unavailable")
    archived = frozen._device_tensor_tree(_FROZEN_ARCHIVE_CPU, device)
    current_match = step_audit.parameter_tensor_comparison(
        current_parameters, archived["current_parameters"]
    )
    raw_parameters, gradient_norm = _optimizer_step_from_gradient(
        student, optimizer, archived["raw_damping_gradient"]
    )
    raw_displacement = {
        name: raw_parameters[name] - current_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    raw_match = step_audit.parameter_tensor_comparison(
        raw_displacement, archived["raw_displacement"]
    )
    report = {
        "pass": bool(
            current_match["pass"]
            and raw_match["pass"]
            and math.isclose(
                gradient_norm,
                float(archived["gradient_norm_before_clipping"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ),
        "current_parameters": current_match,
        "raw_displacement": raw_match,
        "gradient_norm_exact_match": math.isclose(
            gradient_norm,
            float(archived["gradient_norm_before_clipping"]),
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "jacobians_regenerated": False,
    }
    return (
        archived["constraint_specs"],
        archived["constraint_rows"],
        archived["raw_damping_gradient"],
        archived["raw_displacement"],
        gradient_norm,
        report,
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
    global _UPDATE_NUMBER

    _UPDATE_NUMBER += 1
    corrected._UPDATE_NUMBER = _UPDATE_NUMBER
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    update21_reproduction: dict[str, Any] = {"required": False, "pass": True}
    if _UPDATE_NUMBER == 21:
        (
            specs,
            rows,
            raw_gradients,
            raw_displacement,
            gradient_norm,
            update21_reproduction,
        ) = _update_21_inputs(student, optimizer, current_parameters, device)
        update21_reproduction["required"] = True
    else:
        specs, rows, raw_gradients, gradient_norm = _fresh_proposal_inputs(
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
    optimizer_after_sha256 = audit.semantic_sha256(optimizer_after)
    proposed, projection = fp64.fp64_bound_aware_projection(
        raw_displacement, rows, specs, current_parameters["edge_magnitude"]
    )
    fp64_control = frozen.fp64_run_control(projection)
    authoritative, effective64, canonicalization = canonical.materialize_authoritative_candidate(
        student, current_parameters, proposed
    )
    parameter_bounds = audit.parameter_bounds_report(authoritative)
    effective32 = {
        name: value.to(current_parameters[name].dtype) for name, value in effective64.items()
    }
    linearized = audit.linearized_constraint_violations(specs, rows, effective64)
    maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
    raw64 = {name: value.double() for name, value in raw_displacement.items()}
    raw_derivative = canonical._dot_float64(raw_gradients, raw64)
    effective_derivative = canonical._dot_float64(raw_gradients, effective64)

    if _UPDATE_NUMBER == 21:
        projected_sha256 = audit.semantic_sha256(proposed)
        objective = frozen.complete_projection_primal_objective(proposed, raw_displacement)
        update21_reproduction.update(
            {
                "qualified_projection_displacement_sha256": projected_sha256,
                "expected_qualified_projection_displacement_sha256": (
                    step_audit.EXPECTED_PROJECTED_DISPLACEMENT_SHA256
                ),
                "qualified_projection_objective": objective,
                "expected_qualified_projection_objective": (
                    step_audit.EXPECTED_COMPLETE_PRIMAL_OBJECTIVE
                ),
                "qualified_projection_objective_relative_difference": (
                    step_audit.relative_difference(
                        objective, step_audit.EXPECTED_COMPLETE_PRIMAL_OBJECTIVE
                    )
                ),
                "optimizer_after_sha256": optimizer_after_sha256,
                "expected_optimizer_after_sha256": (step_audit.EXPECTED_PENDING_OPTIMIZER_SHA256),
            }
        )
        update21_reproduction["pass"] = bool(
            update21_reproduction["pass"]
            and projected_sha256 == step_audit.EXPECTED_PROJECTED_DISPLACEMENT_SHA256
            and step_audit.relative_difference(
                objective, step_audit.EXPECTED_COMPLETE_PRIMAL_OBJECTIVE
            )
            <= step_audit.OBJECTIVE_RELATIVE_TOLERANCE
            and optimizer_after_sha256 == step_audit.EXPECTED_PENDING_OPTIMIZER_SHA256
        )

    projection.update(
        {
            "pass_after_parameter_bounds": bool(
                fp64_control["pass"]
                and canonicalization["pass"]
                and parameter_bounds["pass"]
                and linearized_finite
                and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
                and effective_derivative < 0.0
                and optimizer_transaction["pass"]
                and update21_reproduction["pass"]
            ),
            "fp64_control": fp64_control,
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
                name: float(value.square().mean().sqrt())
                for name, value in raw_displacement.items()
            },
            "bounded_projected_displacement_family_rms": {
                name: float(value.square().mean().sqrt()) for name, value in effective64.items()
            },
            "constraint_specs": specs,
            "gradient_norm_before_clipping": gradient_norm,
            "optimizer_transaction": optimizer_transaction,
            "update_21_frozen_reproduction": update21_reproduction,
            "effective_displacement_computed_in_float64": True,
        }
    )
    canonical._AUTHORITATIVE_BASE = current_parameters
    canonical._AUTHORITATIVE_CANDIDATE = authoritative
    corrected._PENDING_OPTIMIZER = optimizer
    corrected._OPTIMIZER_BEFORE_PROPOSAL = optimizer_before
    corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = optimizer_after_sha256
    joint._load_parameters(student, current_parameters)
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
    if _UPDATE_NUMBER != 21:
        return selected_scale, selected_metrics, trials
    if _STEP_AUDIT_REPORT is None:
        raise RuntimeError("passing update-21 step audit is unavailable")
    candidate_parameters = joint._copy_parameters(student) if selected_metrics is not None else None
    parameter_sha256 = (
        audit.semantic_sha256(candidate_parameters) if candidate_parameters is not None else None
    )
    metrics_reproduction = (
        audit.numeric_tree_comparison(
            selected_metrics, _STEP_AUDIT_REPORT["selected_training_metrics"]
        )
        if selected_metrics is not None
        else {"pass": False, "reason": "update 21 selected no candidate"}
    )
    reproduction = {
        "pass": bool(
            selected_scale == corrected.REPAIR_PROPOSAL_SCALE
            and parameter_sha256 == EXPECTED_UPDATE_21_PARAMETER_SHA256
            and metrics_reproduction["pass"]
        ),
        "selected_scale": selected_scale,
        "expected_selected_scale": corrected.REPAIR_PROPOSAL_SCALE,
        "parameter_sha256": parameter_sha256,
        "expected_parameter_sha256": EXPECTED_UPDATE_21_PARAMETER_SHA256,
        "training_metrics": metrics_reproduction,
    }
    if trials:
        trials[-1]["registered_update_21_reproduction"] = reproduction
    if reproduction["pass"]:
        return selected_scale, selected_metrics, trials
    if trials:
        trials[-1]["decision"] = {
            **trials[-1]["decision"],
            "pass": False,
            "reasons": [
                *trials[-1]["decision"].get("reasons", []),
                "registered update-21 candidate reproduction failed",
            ],
        }
    corrected._restore_rejected_transaction(student, current_parameters)
    return None, None, trials


def _validate_authorizations(args: argparse.Namespace) -> None:
    global _FROZEN_ARCHIVE_CPU, _STEP_AUDIT_REPORT

    qualification = json.loads((args.qualification_dir / "report.json").read_text(encoding="utf-8"))
    if (
        qualification.get("experiment"),
        qualification.get("classification"),
        qualification.get("pass"),
        qualification.get("fp64_projection_implementation_authorized"),
    ) != (
        frozen.EXPERIMENT,
        "identical_input_fp64_projection_qualified",
        True,
        True,
    ):
        raise SystemExit("frozen FP64 qualification does not authorize continuation")
    _STEP_AUDIT_REPORT = json.loads(args.step_audit_report.read_text(encoding="utf-8"))
    if (
        _STEP_AUDIT_REPORT.get("experiment"),
        _STEP_AUDIT_REPORT.get("classification"),
        _STEP_AUDIT_REPORT.get("pass"),
        _STEP_AUDIT_REPORT.get("fp64_corrected_continuation_authorized"),
        _STEP_AUDIT_REPORT.get("selected_parameter_sha256"),
    ) != (
        step_audit.EXPERIMENT,
        "fp64_repaired_update_21_transfers_to_development",
        True,
        True,
        EXPECTED_UPDATE_21_PARAMETER_SHA256,
    ):
        raise SystemExit("restored update-21 audit does not authorize continuation")
    _FROZEN_ARCHIVE_CPU = torch.load(
        args.qualification_dir / "frozen-inputs.pt",
        map_location="cpu",
        weights_only=True,
    )
    if frozen.archive_tensor_hashes(_FROZEN_ARCHIVE_CPU) != _FROZEN_ARCHIVE_CPU.get(
        "tensor_semantic_sha256"
    ):
        raise SystemExit("frozen update-21 archive failed its semantic hashes")


def main() -> int:
    global _UPDATE_NUMBER

    args = parse_args()
    validate_args(args)
    locked_before = {
        name: responsibility.file_sha256(path) for name, path in _locked_input_paths(args).items()
    }
    _validate_authorizations(args)
    corrected_globals = {
        "guard_report": corrected._GUARD_REPORT,
        "update_number": corrected._UPDATE_NUMBER,
        "pending_optimizer": corrected._PENDING_OPTIMIZER,
        "optimizer_before": corrected._OPTIMIZER_BEFORE_PROPOSAL,
        "optimizer_after_sha256": corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
    }
    _UPDATE_NUMBER = seed_resume(args)
    corrected._UPDATE_NUMBER = _UPDATE_NUMBER
    overrides = {
        "EXPERIMENT": EXPERIMENT,
        "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
        "MAXIMUM_ACCEPTED_UPDATES": MAXIMUM_ACCEPTED_UPDATES,
        "DEVELOPMENT_PRESERVATION_REQUIRED": True,
        "PERSIST_NUMERICAL_EXCEPTIONS_AS_STOPPED": True,
        "parse_args": lambda: args,
        "validate_args": validate_args,
        "protocol_manifest": protocol_manifest,
        "make_projected_proposal": make_projected_proposal,
        "install_trial": canonical.install_trial,
        "find_safe_trial": find_safe_trial,
    }
    canonical_base = canonical._AUTHORITATIVE_BASE
    canonical_candidate = canonical._AUTHORITATIVE_CANDIDATE
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        result = base.main()
    finally:
        for name, value in _ORIGINAL_BASE_VALUES.items():
            setattr(base, name, value)
        corrected._GUARD_REPORT = corrected_globals["guard_report"]
        corrected._UPDATE_NUMBER = corrected_globals["update_number"]
        corrected._PENDING_OPTIMIZER = corrected_globals["pending_optimizer"]
        corrected._OPTIMIZER_BEFORE_PROPOSAL = corrected_globals["optimizer_before"]
        corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = corrected_globals["optimizer_after_sha256"]
        canonical._AUTHORITATIVE_BASE = canonical_base
        canonical._AUTHORITATIVE_CANDIDATE = canonical_candidate
    locked_after = {
        name: responsibility.file_sha256(path) for name, path in _locked_input_paths(args).items()
    }
    if locked_after != locked_before:
        raise RuntimeError("one or more locked source files changed during fitting")
    report_path = args.output_dir / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fp64_continuation_provenance"] = {
            "locked_hashes_before": locked_before,
            "locked_hashes_after": locked_after,
            "all_locked_inputs_unchanged": True,
            "seeded_from_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "phase_maximum_total_update": MAXIMUM_ACCEPTED_UPDATES,
        }
        report_path.write_text(
            f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
