#!/usr/bin/env python3
"""Continue D-first fitting with the preregistered nonlinear C guard-band repair."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_canonical_guard_band as guard  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_bound_aware as bounded  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-corrected-fitting-v1"
PROTOCOL_COMMIT = "4e9f646"
EXPECTED_INITIAL_RESUME_SHA256 = audit.EXPECTED_RESUME_SHA256
EXPECTED_INITIAL_EXPERIMENT = canonical.EXPERIMENT
EXPECTED_INITIAL_PROTOCOL_COMMIT = canonical.PROTOCOL_COMMIT
EXPECTED_INITIAL_ACCEPTED_UPDATES = audit.EXPECTED_ACCEPTED_UPDATES
EXPECTED_GUARD_REPORT_SHA256 = "40c34d00e8ab5c69558a3d385197ab8ef0170c8a2bd076c09f75ed2ee22c019e"
EXPECTED_GUARD_CLASSIFICATION = "nonlinear_correction_feasible_and_update_8_transfers"
REPAIR_PROPOSAL_SCALE = 0.0625
CORRECTION_SCALES = (1.0, 0.5, 0.25, 0.125)
SOLVER_ENDPOINT_COMMON_MARGIN = 0.0198
ACCEPTANCE_ENDPOINT_COMMON_MARGIN = 0.0199
ENDPOINT_DAMPING_IMPROVEMENT = 0.001

_ORIGINAL_BASE_VALUES = {
    "EXPERIMENT": base.EXPERIMENT,
    "PROTOCOL_COMMIT": base.PROTOCOL_COMMIT,
    "parse_args": base.parse_args,
    "validate_args": base.validate_args,
    "protocol_manifest": base.protocol_manifest,
    "make_projected_proposal": base.make_projected_proposal,
    "install_trial": base.install_trial,
    "find_safe_trial": base.find_safe_trial,
}
_ORIGINAL_FIND_SAFE_TRIAL = base.find_safe_trial
_GUARD_REPORT: dict[str, Any] | None = None
_UPDATE_NUMBER = EXPECTED_INITIAL_ACCEPTED_UPDATES
_PENDING_OPTIMIZER: torch.optim.Optimizer | None = None
_OPTIMIZER_BEFORE_PROPOSAL: dict[str, Any] | None = None
_OPTIMIZER_AFTER_PROPOSAL_SHA256: str | None = None
_REPAIR_TENSOR_CAPTURE: dict[str, Any] | None = None


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
        "--initial-resume",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-d-first-canonical-fitting-001/resume.pt"
        ),
    )
    parser.add_argument(
        "--guard-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-canonical-guard-band-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-corrected-fitting-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _validated_guard_report(path: Path) -> dict[str, Any]:
    if responsibility.file_sha256(path) != EXPECTED_GUARD_REPORT_SHA256:
        raise SystemExit("canonical guard-band audit report hash does not match the protocol")
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = (
        report.get("experiment"),
        report.get("protocol", {}).get("protocol_commit"),
        report.get("classification"),
        report.get("pass"),
        report.get("corrected_fitting_protocol_authorized"),
        report.get("selected_correction_scale"),
        report.get("source", {}).get("update_8_resume_sha256_before"),
    )
    expected = (
        guard.EXPERIMENT,
        guard.PROTOCOL_COMMIT,
        EXPECTED_GUARD_CLASSIFICATION,
        True,
        True,
        1.0,
        EXPECTED_INITIAL_RESUME_SHA256,
    )
    if identity != expected:
        raise SystemExit("canonical guard-band audit identity does not match the protocol")
    return report


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.initial_resume,
        args.guard_audit_report,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training/development cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered fitting diagnostic has no smoke variant")
    if (args.output_dir / "resume.pt").resolve() == args.initial_resume.resolve():
        raise SystemExit("corrected fitting must not overwrite the canonical update-8 resume")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the corrected fitting run already has a final report")
    if responsibility.file_sha256(args.initial_resume) != EXPECTED_INITIAL_RESUME_SHA256:
        raise SystemExit("canonical update-8 resume hash does not match the protocol")
    _validated_guard_report(args.guard_audit_report)


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(canonical.protocol_manifest())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "starting_resume": {
                "experiment": EXPECTED_INITIAL_EXPERIMENT,
                "protocol_commit": EXPECTED_INITIAL_PROTOCOL_COMMIT,
                "sha256": EXPECTED_INITIAL_RESUME_SHA256,
                "accepted_updates": EXPECTED_INITIAL_ACCEPTED_UPDATES,
                "copied_to_new_run": True,
                "source_overwritten": False,
            },
            "authorizing_guard_band_audit": {
                "experiment": guard.EXPERIMENT,
                "protocol_commit": guard.PROTOCOL_COMMIT,
                "report_sha256": EXPECTED_GUARD_REPORT_SHA256,
                "required_pass": True,
            },
            "maximum_accepted_updates": base.MAXIMUM_ACCEPTED_UPDATES,
            "development_interval_accepted_updates": base.DEVELOPMENT_INTERVAL,
            "starting_accepted_update": EXPECTED_INITIAL_ACCEPTED_UPDATES,
            "accepted_update_budget_is_total_not_additional": True,
        }
    )
    manifest["optimizer"].update(
        {
            "adam_step_per_attempt": 1,
            "correction_optimizer_steps": 0,
            "pending_state_retained_for_ordinary_or_repaired_acceptance": True,
            "pre_attempt_state_restored_on_any_rejection": True,
        }
    )
    manifest["nonlinear_repair"] = {
        "ordinary_backtracks_attempted_first": list(base.BACKTRACK_SCALES),
        "only_eligible_proposal_scale": REPAIR_PROPOSAL_SCALE,
        "eligibility": {
            "minimum_actual_endpoint_damping_nrmse_improvement": (ENDPOINT_DAMPING_IMPROVEMENT),
            "negative_actual_damping_direction": True,
            "only_allowed_ordinary_failure": "common.step_25 source plus 0.02",
            "projection_and_numerical_controls_must_pass": True,
        },
        "endpoint_damping_reference": "current accepted controller",
        "endpoint_damping_row": "dedicated accumulated endpoint-D objective",
        "duplicate_multi_loss_row": "diagnostic only under prior limits",
        "finite_difference_probe": "one quarter from candidate toward current controller",
        "finite_difference_relative_error_maximum": (
            audit.DIRECTIONAL_FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "single_zero_reference_bound_aware_solve": True,
        "fixed_jacobian": True,
        "relinearized": False,
        "second_solve": False,
        "solver_endpoint_common_nrmse_maximum_source_plus": (SOLVER_ENDPOINT_COMMON_MARGIN),
        "acceptance_endpoint_common_nrmse_maximum_source_plus": (ACCEPTANCE_ENDPOINT_COMMON_MARGIN),
        "correction_scales_descending": list(CORRECTION_SCALES),
    }
    manifest["crash_safe_lifecycle"] = {
        "scheduled_development_persisted_before_next_update": True,
        "rejected_proposal_is_terminal_on_resume": True,
        "qualification_marked_started_before_fresh_cases": True,
        "interrupted_qualification_retried": False,
    }
    return manifest


def seed_corrected_resume(args: argparse.Namespace) -> int:
    """Fork the ignored update-8 state into the new run without touching its source."""
    output = args.output_dir / "resume.pt"
    if output.is_file():
        payload = torch.load(output, map_location="cpu", weights_only=True)
        identity = (
            payload.get("experiment"),
            payload.get("protocol_commit"),
            int(payload.get("accepted_updates", -1)),
        )
        if identity[0:2] != (EXPERIMENT, PROTOCOL_COMMIT):
            raise SystemExit("existing corrected resume does not match the frozen protocol")
        return identity[2]
    payload = torch.load(args.initial_resume, map_location="cpu", weights_only=True)
    identity = (
        payload.get("experiment"),
        payload.get("protocol_commit"),
        int(payload.get("accepted_updates", -1)),
    )
    if identity != (
        EXPECTED_INITIAL_EXPERIMENT,
        EXPECTED_INITIAL_PROTOCOL_COMMIT,
        EXPECTED_INITIAL_ACCEPTED_UPDATES,
    ):
        raise SystemExit("canonical update-8 resume identity does not match the protocol")
    if not payload.get("preflight", {}).get("pass", False):
        raise SystemExit("canonical update-8 resume lacks its passing preflight")
    seeded = copy.deepcopy(payload)
    seeded.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "run_state": "active",
            "qualification": None,
            "seed_provenance": {
                "source_resume_sha256": EXPECTED_INITIAL_RESUME_SHA256,
                "guard_audit_report_sha256": EXPECTED_GUARD_REPORT_SHA256,
            },
        }
    )
    base.atomic_torch_save(seeded, output)
    if responsibility.file_sha256(args.initial_resume) != EXPECTED_INITIAL_RESUME_SHA256:
        raise SystemExit("canonical update-8 resume changed while the new run was seeded")
    return EXPECTED_INITIAL_ACCEPTED_UPDATES


def _optimizer_step(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu())
    return float(value)


def optimizer_step_transaction(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    parameter_ids = [
        parameter_id for group in after["param_groups"] for parameter_id in group["params"]
    ]
    counters = []
    for parameter_id in parameter_ids:
        before_state = before["state"].get(parameter_id, {})
        after_state = after["state"].get(parameter_id, {})
        before_step = _optimizer_step(before_state.get("step", 0.0))
        after_step = _optimizer_step(after_state.get("step", 0.0))
        counters.append(
            {
                "parameter_id": int(parameter_id),
                "before": before_step,
                "after": after_step,
                "increment": after_step - before_step,
            }
        )
    before_sha = audit.semantic_sha256(before)
    after_sha = audit.semantic_sha256(after)
    return {
        "pass": bool(counters and all(item["increment"] == 1.0 for item in counters)),
        "state_before_sha256": before_sha,
        "state_after_sha256": after_sha,
        "state_changed": before_sha != after_sha,
        "parameter_step_counters": counters,
        "optimizer_step_calls_expected": 1,
    }


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
    global _PENDING_OPTIMIZER, _OPTIMIZER_BEFORE_PROPOSAL
    global _OPTIMIZER_AFTER_PROPOSAL_SHA256, _UPDATE_NUMBER

    _UPDATE_NUMBER += 1
    before = copy.deepcopy(optimizer.state_dict())
    displacement, raw_gradients, projection = canonical.make_projected_proposal(
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
    after = copy.deepcopy(optimizer.state_dict())
    transaction = optimizer_step_transaction(before, after)
    projection["optimizer_transaction"] = transaction
    projection["pass_after_parameter_bounds"] = bool(
        projection["pass_after_parameter_bounds"] and transaction["pass"]
    )
    if _UPDATE_NUMBER == EXPECTED_INITIAL_ACCEPTED_UPDATES + 1:
        if _GUARD_REPORT is None:
            raise RuntimeError("guard-band report was not loaded")
        expected = _GUARD_REPORT["reconstructed_update_9"]["proposal"]
        reproduction = audit.numeric_tree_comparison(
            audit.proposal_signature(projection), audit.proposal_signature(expected)
        )
        projection["guard_audit_update_9_proposal_reproduction"] = reproduction
        projection["pass_after_parameter_bounds"] = bool(
            projection["pass_after_parameter_bounds"] and reproduction["pass"]
        )
    _PENDING_OPTIMIZER = optimizer
    _OPTIMIZER_BEFORE_PROPOSAL = before
    _OPTIMIZER_AFTER_PROPOSAL_SHA256 = transaction["state_after_sha256"]
    return displacement, raw_gradients, projection


def repair_eligibility(
    source: dict[str, Any],
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    damping_directional_derivative: float,
    projection_controls_pass: bool,
    parameter_bounds_pass: bool,
    canonicalization_pass: bool,
    registered_reproduction_pass: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not projection_controls_pass:
        reasons.append("proposal projection or numerical control failed")
    if not joint._numeric_tree_is_finite(candidate):
        reasons.append("one or more candidate metrics were nonfinite")
    if not parameter_bounds_pass:
        reasons.append("candidate parameters violated native bounds")
    if not canonicalization_pass:
        reasons.append("candidate parameters failed canonical idempotence")
    if not registered_reproduction_pass:
        reasons.append("registered update-9 candidate reproduction failed")
    if candidate["motor_output_max_absolute"] > 1.0:
        reasons.append("motor output exceeded [-1, 1]")
    if not candidate["all_attitude_cache_states_valid"]:
        reasons.append("attitude cache contains an invalid physical state")
    for component in ("common", "height"):
        if (
            candidate["component_nrmse"][component]
            > source["component_nrmse"][component] + joint.COMPONENT_BASELINE_TOLERANCE
        ):
            reasons.append(f"aggregate {component} preservation failed")
        for horizon in map(str, joint.SUPERVISION_STEPS):
            if component == "common" and horizon == "25":
                continue
            if (
                candidate["by_supervision_step_nrmse"][horizon][component]
                > source["by_supervision_step_nrmse"][horizon][component]
                + joint.COMPONENT_BASELINE_TOLERANCE
            ):
                reasons.append(f"{component}.step_{horizon} preservation failed")
    for axis, value in candidate["rpy_source_nrmse"].items():
        if value > joint.RPY_NRMSE_LIMIT:
            reasons.append(f"rpy.{axis} preservation failed")
    endpoint_common = candidate["by_supervision_step_nrmse"]["25"]["common"]
    outer_limit = (
        source["by_supervision_step_nrmse"]["25"]["common"] + joint.COMPONENT_BASELINE_TOLERANCE
    )
    if endpoint_common <= outer_limit:
        reasons.append("common.step_25 did not uniquely require nonlinear repair")
    improvement = current["endpoint_damping_nrmse"] - candidate["endpoint_damping_nrmse"]
    if improvement < ENDPOINT_DAMPING_IMPROVEMENT:
        reasons.append("actual endpoint-D improvement was below 0.001")
    if damping_directional_derivative >= 0.0:
        reasons.append("actual candidate displacement was not a damping descent direction")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "proposal_scale": REPAIR_PROPOSAL_SCALE,
        "endpoint_common_nrmse": endpoint_common,
        "endpoint_common_outer_limit": outer_limit,
        "endpoint_damping_nrmse_improvement": improvement,
        "damping_directional_derivative": damping_directional_derivative,
    }


@torch.no_grad()
def _candidate_idempotence(
    student: ConnectomeController,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    before = joint._copy_parameters(student)
    student.project_parameters()
    maximum = max(
        float((getattr(student, name).detach() - before[name]).abs().max())
        for name in joint.PARAMETER_FAMILIES
    )
    return joint._copy_parameters(student), {
        "pass": maximum <= canonical.PARAMETER_IDEMPOTENCE_TOLERANCE,
        "second_direct_projection_maximum_parameter_change": maximum,
        "second_direct_projection_maximum_parameter_change_limit": (
            canonical.PARAMETER_IDEMPOTENCE_TOLERANCE
        ),
    }


def _optimizer_pending_report() -> dict[str, Any]:
    if _PENDING_OPTIMIZER is None or _OPTIMIZER_AFTER_PROPOSAL_SHA256 is None:
        return {"pass": False, "reason": "pending optimizer transaction is unavailable"}
    current = audit.semantic_sha256(_PENDING_OPTIMIZER.state_dict())
    return {
        "pass": current == _OPTIMIZER_AFTER_PROPOSAL_SHA256,
        "expected_pending_state_sha256": _OPTIMIZER_AFTER_PROPOSAL_SHA256,
        "actual_state_sha256": current,
        "additional_optimizer_steps_during_trials_or_correction": 0,
    }


def _restore_rejected_transaction(
    student: ConnectomeController, current_parameters: dict[str, Tensor]
) -> dict[str, Any]:
    joint._load_parameters(student, current_parameters)
    parameters_restored = all(
        torch.equal(getattr(student, name).detach(), current_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_restored = False
    if _PENDING_OPTIMIZER is not None and _OPTIMIZER_BEFORE_PROPOSAL is not None:
        _PENDING_OPTIMIZER.load_state_dict(_OPTIMIZER_BEFORE_PROPOSAL)
        _PENDING_OPTIMIZER.zero_grad(set_to_none=True)
        optimizer_restored = audit.trees_equal(
            _PENDING_OPTIMIZER.state_dict(), _OPTIMIZER_BEFORE_PROPOSAL
        )
    return {
        "pass": parameters_restored and optimizer_restored,
        "parameters_restored_exactly": parameters_restored,
        "optimizer_restored_exactly": optimizer_restored,
    }


def _attempt_repair(
    student: ConnectomeController,
    current_parameters: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    source_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    projection_controls_pass: bool,
    device: torch.device,
) -> tuple[float | None, dict[str, Any] | None, dict[str, Any]]:
    canonical.install_trial(
        student,
        current_parameters,
        {name: torch.zeros_like(value) for name, value in current_parameters.items()},
        scale=REPAIR_PROPOSAL_SCALE,
    )
    starting_parameters, candidate_idempotence = _candidate_idempotence(student)
    starting_metrics, _ = endpoint.evaluate(
        student,
        factorial_cache,
        attitude_cache,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    actual = {
        name: starting_parameters[name].double() - current_parameters[name].double()
        for name in joint.PARAMETER_FAMILIES
    }
    derivative = canonical._dot_float64(raw_gradients, actual)
    bounds = audit.parameter_bounds_report(starting_parameters)
    registered_reproduction: dict[str, Any] = {"required": False, "pass": True}
    if _UPDATE_NUMBER == EXPECTED_INITIAL_ACCEPTED_UPDATES + 1:
        if _GUARD_REPORT is None:
            raise RuntimeError("guard-band report was not loaded")
        expected = _GUARD_REPORT["reconstructed_update_9"]["starting_metrics"]
        registered_reproduction = audit.numeric_tree_comparison(starting_metrics, expected)
        registered_reproduction["required"] = True
    eligibility = repair_eligibility(
        source_metrics,
        current_metrics,
        starting_metrics,
        damping_directional_derivative=derivative,
        projection_controls_pass=projection_controls_pass,
        parameter_bounds_pass=bounds["pass"],
        canonicalization_pass=candidate_idempotence["pass"],
        registered_reproduction_pass=registered_reproduction["pass"],
    )
    report: dict[str, Any] = {
        "attempted": eligibility["pass"],
        "eligibility": eligibility,
        "starting_candidate": {
            "parameter_tensor_sha256": audit.semantic_sha256(starting_parameters),
            "metrics": starting_metrics,
            "parameter_bounds": bounds,
            "canonical_parameter_idempotence": candidate_idempotence,
            "guard_audit_reproduction": registered_reproduction,
        },
        "solver_constraint_specs": None,
        "acceptance_constraint_specs": None,
        "endpoint_damping_gradient_agreement": None,
        "endpoint_damping_directional_finite_difference": None,
        "projection": None,
        "correction_trials": [],
        "selected_correction_scale": None,
        "optimizer_pending_state": None,
        "relinearized": False,
        "second_solve": False,
    }
    if not eligibility["pass"]:
        return None, None, report

    solver_specs = audit.correction_constraint_specs(
        source_metrics,
        current_metrics,
        starting_metrics,
        endpoint_common_margin=SOLVER_ENDPOINT_COMMON_MARGIN,
    )
    acceptance_specs = audit.correction_constraint_specs(
        source_metrics,
        current_metrics,
        starting_metrics,
        endpoint_common_margin=ACCEPTANCE_ENDPOINT_COMMON_MARGIN,
    )
    rows = base.constraint_gradient_rows(
        student,
        factorial_cache,
        attitude_cache,
        scales,
        solver_specs,
        device=device,
    )
    damping_index = next(
        index
        for index, spec in enumerate(solver_specs)
        if spec["name"] == "damping.step_25.retention"
    )
    duplicate_damping_row = rows[damping_index]
    if _PENDING_OPTIMIZER is None:
        raise RuntimeError("pending optimizer is unavailable for correction differentiation")
    _PENDING_OPTIMIZER.zero_grad(set_to_none=True)
    endpoint.accumulated_endpoint_damping_gradient(
        student,
        factorial_cache,
        scale=endpoint_scale,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    authoritative_damping_row = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_agreement = audit.gradient_agreement(duplicate_damping_row, authoritative_damping_row)
    rows[damping_index] = authoritative_damping_row
    _PENDING_OPTIMIZER.zero_grad(set_to_none=True)

    probe_parameters, probe_displacement, probe_idempotence = audit.install_authoritative_trial(
        student,
        starting_parameters,
        current_parameters,
        scale=audit.DIRECTIONAL_FINITE_DIFFERENCE_PROBE_FRACTION,
    )
    probe_bounds = audit.parameter_bounds_report(probe_parameters)
    probe_metrics, _ = endpoint.evaluate(
        student,
        factorial_cache,
        attitude_cache,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    joint._load_parameters(student, starting_parameters)
    starting_restored = all(
        torch.equal(getattr(student, name).detach(), starting_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    finite_difference = audit.directional_finite_difference_report(
        authoritative_damping_row,
        probe_displacement,
        starting_endpoint_damping_nrmse=starting_metrics["endpoint_damping_nrmse"],
        probe_endpoint_damping_nrmse=probe_metrics["endpoint_damping_nrmse"],
        canonicalization_pass=probe_idempotence["pass"],
        parameter_bounds_pass=probe_bounds["pass"],
        starting_restored_exactly=starting_restored,
    )
    finite_difference.update(
        {
            "probe_reference": "current accepted controller",
            "probe_metrics": probe_metrics,
            "probe_parameter_bounds": probe_bounds,
            "probe_canonical_parameter_idempotence": probe_idempotence,
        }
    )

    zero = {name: torch.zeros_like(value) for name, value in starting_parameters.items()}
    proposed, projection = bounded.bound_aware_project_inequality_displacement(
        zero, rows, solver_specs, starting_parameters["edge_magnitude"]
    )
    corrected_parameters, effective, correction_idempotence = (
        canonical.materialize_authoritative_candidate(student, starting_parameters, proposed)
    )
    solver_linearized = audit.linearized_constraint_violations(solver_specs, rows, effective)
    maximum_solver, solver_finite = audit.maximum_linearized_violation(solver_linearized)
    acceptance_linearized = audit.linearized_constraint_violations(
        acceptance_specs, rows, effective
    )
    maximum_acceptance, acceptance_finite = audit.maximum_linearized_violation(
        acceptance_linearized
    )
    corrected_bounds = audit.parameter_bounds_report(corrected_parameters)
    preplay_pass = audit.correction_projection_preplay_pass(
        projection_pass=projection["pass"],
        canonicalization_pass=correction_idempotence["pass"],
        solver_linearized_finite=solver_finite,
        maximum_solver_linearized_violation=maximum_solver,
        acceptance_linearized_finite=acceptance_finite,
        parameter_bounds_pass=corrected_bounds["pass"],
        damping_row_control_pass=finite_difference["pass"],
        distinct_solver_acceptance_specs=True,
    )
    report.update(
        {
            "solver_constraint_specs": solver_specs,
            "acceptance_constraint_specs": acceptance_specs,
            "endpoint_damping_gradient_agreement": gradient_agreement,
            "endpoint_damping_gradient_agreement_is_diagnostic_only": True,
            "endpoint_damping_directional_finite_difference": finite_difference,
            "projection": projection,
            "authoritative_parameter_canonicalization": correction_idempotence,
            "effective_full_correction_parameter_bounds": corrected_bounds,
            "effective_full_correction_solver_linearized_violations": solver_linearized,
            "maximum_effective_full_correction_solver_linearized_violation": maximum_solver,
            "effective_full_correction_acceptance_linearized_violations": (acceptance_linearized),
            "maximum_effective_full_correction_acceptance_linearized_violation": (
                maximum_acceptance
            ),
            "pass_before_nonlinear_replay": preplay_pass,
        }
    )

    selected_scale = None
    selected_metrics = None
    for correction_scale in CORRECTION_SCALES:
        parameters, displacement, idempotence = audit.install_authoritative_trial(
            student,
            starting_parameters,
            corrected_parameters,
            scale=correction_scale,
        )
        linearized = audit.linearized_constraint_violations(acceptance_specs, rows, displacement)
        maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
        trial_bounds = audit.parameter_bounds_report(parameters)
        metrics, _ = endpoint.evaluate(
            student,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        decision = audit.correction_decision(
            source_metrics,
            current_metrics,
            metrics,
            maximum_linearized_violation=maximum_linearized,
            canonicalization_pass=idempotence["pass"],
            parameter_bounds_pass=trial_bounds["pass"],
        )
        trial = {
            "scale": correction_scale,
            "metrics": metrics,
            "decision": decision,
            "linearized_constraint_violations": linearized,
            "linearized_constraint_violations_finite": linearized_finite,
            "canonical_parameter_idempotence": idempotence,
            "parameter_bounds": trial_bounds,
            "actual_displacement_family_rms": {
                name: float(value.square().mean().sqrt()) for name, value in displacement.items()
            },
        }
        report["correction_trials"].append(trial)
        if preplay_pass and decision["pass"]:
            selected_scale = correction_scale
            selected_metrics = metrics
            break

    optimizer_pending = _optimizer_pending_report()
    report["optimizer_pending_state"] = optimizer_pending
    if not optimizer_pending["pass"]:
        selected_scale = None
        selected_metrics = None
    report["selected_correction_scale"] = selected_scale
    if selected_metrics is not None:
        common_limit = (
            source_metrics["by_supervision_step_nrmse"]["25"]["common"]
            + ACCEPTANCE_ENDPOINT_COMMON_MARGIN
        )
        report["accepted_endpoint_common_headroom"] = (
            common_limit - selected_metrics["by_supervision_step_nrmse"]["25"]["common"]
        )
        report["accepted_endpoint_damping_nrmse_improvement"] = (
            current_metrics["endpoint_damping_nrmse"] - selected_metrics["endpoint_damping_nrmse"]
        )
        if _REPAIR_TENSOR_CAPTURE is not None:
            _REPAIR_TENSOR_CAPTURE.update(
                {
                    "solver_constraint_rows": [
                        {name: value.detach().cpu().clone() for name, value in row.items()}
                        for row in rows
                    ],
                    "authoritative_endpoint_damping_row": {
                        name: value.detach().cpu().clone()
                        for name, value in authoritative_damping_row.items()
                    },
                    "full_correction_direction": {
                        name: value.detach().cpu().clone() for name, value in effective.items()
                    },
                    "starting_candidate_parameters": {
                        name: value.detach().cpu().clone()
                        for name, value in starting_parameters.items()
                    },
                    "full_correction_parameters": {
                        name: value.detach().cpu().clone()
                        for name, value in corrected_parameters.items()
                    },
                    "solver_constraint_specs": copy.deepcopy(solver_specs),
                    "acceptance_constraint_specs": copy.deepcopy(acceptance_specs),
                    "selected_correction_scale": selected_scale,
                }
            )
    return selected_scale, selected_metrics, report


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
    selected_scale, selected_metrics, trials = _ORIGINAL_FIND_SAFE_TRIAL(
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
    for trial in trials:
        trial.update(
            {
                "acceptance_kind": "ordinary" if trial["decision"]["pass"] else "ordinary_attempt",
                "proposal_scale": trial["scale"],
                "correction_scale": None,
            }
        )
    if selected_scale is not None and selected_metrics is not None:
        pending = _optimizer_pending_report()
        selected_trial = trials[-1]
        selected_trial["optimizer_pending_state"] = pending
        selected_trial["endpoint_common_headroom"] = (
            source_metrics["by_supervision_step_nrmse"]["25"]["common"]
            + joint.COMPONENT_BASELINE_TOLERANCE
            - selected_metrics["by_supervision_step_nrmse"]["25"]["common"]
        )
        selected_trial["endpoint_damping_nrmse_improvement"] = (
            current_metrics["endpoint_damping_nrmse"] - selected_metrics["endpoint_damping_nrmse"]
        )
        if pending["pass"]:
            return selected_scale, selected_metrics, trials
        selected_trial["decision"] = {
            **selected_trial["decision"],
            "pass": False,
            "reasons": [
                *selected_trial["decision"]["reasons"],
                "pending optimizer state changed during ordinary trial evaluation",
            ],
        }
        selected_trial["rejected_transaction_restoration"] = _restore_rejected_transaction(
            student, current_parameters
        )
        return None, None, trials

    try:
        correction_scale, corrected_metrics, repair = _attempt_repair(
            student,
            current_parameters,
            raw_gradients,
            source_metrics,
            current_metrics,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale,
            projection_controls_pass=True,
            device=device,
        )
    except Exception:
        _restore_rejected_transaction(student, current_parameters)
        raise
    repaired = correction_scale is not None and corrected_metrics is not None
    repair_trial = {
        "scale": REPAIR_PROPOSAL_SCALE,
        "acceptance_kind": "repaired" if repaired else "repair_attempt",
        "proposal_scale": REPAIR_PROPOSAL_SCALE,
        "correction_scale": correction_scale,
        "metrics": corrected_metrics or repair["starting_candidate"]["metrics"],
        "decision": {
            "pass": repaired,
            "reasons": []
            if repaired
            else (
                repair["eligibility"]["reasons"]
                if not repair["eligibility"]["pass"]
                else ["no fixed correction scale passed every repair gate"]
            ),
        },
        "repair": repair,
    }
    trials.append(repair_trial)
    if repaired:
        return REPAIR_PROPOSAL_SCALE, corrected_metrics, trials
    repair_trial["rejected_transaction_restoration"] = _restore_rejected_transaction(
        student, current_parameters
    )
    return None, None, trials


def main() -> int:
    global _GUARD_REPORT, _UPDATE_NUMBER

    args = parse_args()
    validate_args(args)
    _GUARD_REPORT = _validated_guard_report(args.guard_audit_report)
    _UPDATE_NUMBER = seed_corrected_resume(args)
    overrides = {
        "EXPERIMENT": EXPERIMENT,
        "PROTOCOL_COMMIT": PROTOCOL_COMMIT,
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
        return base.main()
    finally:
        for name, value in _ORIGINAL_BASE_VALUES.items():
            setattr(base, name, value)
        canonical._AUTHORITATIVE_BASE = canonical_base
        canonical._AUTHORITATIVE_CANDIDATE = canonical_candidate


if __name__ == "__main__":
    raise SystemExit(main())
