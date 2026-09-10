#!/usr/bin/env python3
"""Audit one restored update-21 FP64 proposal, repair, and development transfer."""

from __future__ import annotations

import argparse
import copy
import json
import math
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

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-corrected-step-audit-v1"
PROTOCOL_COMMIT = "d6fa381"
EXPECTED_SOURCE_REPORT_SHA256 = frozen.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = frozen.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_QUALIFICATION_REPORT_SHA256 = (
    "e1ead1779c568cacfcebda5df435175f505c9e7218592c37b1187faeaf929f25"
)
EXPECTED_FROZEN_ARCHIVE_SHA256 = "e228a3920c47f56a7eac6d1452f996d9709721f82536240d96dbe6e0c6e1830f"
EXPECTED_PROJECTED_DISPLACEMENT_SHA256 = (
    "0fb352c318ab18efafe0bc2ff68991709aa2f14b79774f7c8e5475d432135e3f"
)
EXPECTED_COMPLETE_PRIMAL_OBJECTIVE = 17920.27762329695
EXPECTED_PENDING_OPTIMIZER_SHA256 = (
    "6743cc01e41e01ccd7183b8981acff29eb5af22a72c3039bb9d234cfcbde42d6"
)
EXPECTED_ACCEPTED_UPDATES = 20
OBJECTIVE_RELATIVE_TOLERANCE = 1.0e-12
DEVELOPMENT_DAMPING_IMPROVEMENT = 0.001


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
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-d-first-fp64-corrected-step-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.source_fit_dir / "report.json",
        args.source_fit_dir / "resume.pt",
        args.qualification_dir / "report.json",
        args.qualification_dir / "frozen-inputs.pt",
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training/development cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered one-step audit has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": corrected.EXPERIMENT,
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
        "source_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "qualification_report_sha256": EXPECTED_QUALIFICATION_REPORT_SHA256,
        "frozen_input_archive_sha256": EXPECTED_FROZEN_ARCHIVE_SHA256,
        "archived_jacobians_regenerated": False,
        "primary_fp64_projection": {
            "qualified_displacement_sha256": EXPECTED_PROJECTED_DISPLACEMENT_SHA256,
            "qualified_complete_primal_objective": EXPECTED_COMPLETE_PRIMAL_OBJECTIVE,
            "objective_relative_tolerance": OBJECTIVE_RELATIVE_TOLERANCE,
            "all_qualified_projection_and_post_materialization_controls_required": True,
        },
        "optimizer_transaction": {
            "gradient_source": "frozen archive",
            "adam_steps": 1,
            "counter_increment": 1,
            "raw_displacement_must_match_archive": True,
            "pending_optimizer_sha256": EXPECTED_PENDING_OPTIMIZER_SHA256,
            "additional_steps_during_trials_or_repair": 0,
        },
        "ordinary_scales_descending": list(base.BACKTRACK_SCALES),
        "repair": {
            "implementation": corrected.EXPERIMENT,
            "proposal_scale": corrected.REPAIR_PROPOSAL_SCALE,
            "correction_scales_descending": list(corrected.CORRECTION_SCALES),
            "endpoint_common_solver_margin": corrected.SOLVER_ENDPOINT_COMMON_MARGIN,
            "endpoint_common_acceptance_margin": (corrected.ACCEPTANCE_ENDPOINT_COMMON_MARGIN),
            "endpoint_damping_improvement_from_update_20": (corrected.ENDPOINT_DAMPING_IMPROVEMENT),
            "fixed_jacobian": True,
            "second_solve": False,
            "arithmetic_unchanged": True,
        },
        "development": {
            "one_selected_candidate": True,
            "fixed_bank": True,
            "endpoint_damping_improvement_from_update_20": (DEVELOPMENT_DAMPING_IMPROVEMENT),
            "original_source_preservation": True,
            "alternative_after_failure": False,
            "fresh_data": False,
        },
        "actor_contract_unchanged": True,
        "all_state_and_sources_restored": True,
        "candidate_retained": False,
        "continuation_checkpoint": False,
        "closed_loop_hover": False,
        "promotion": False,
        "pass_authorizes": (
            "separately registered FP64-primary corrected fitting continuation from "
            "update 20 toward the existing total-update-50 milestone"
        ),
    }


def relative_difference(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(expected))


def parameter_tensor_comparison(
    actual: dict[str, Tensor], expected: dict[str, Tensor]
) -> dict[str, Any]:
    families = {}
    for name in joint.PARAMETER_FAMILIES:
        left = actual[name].detach()
        right = expected[name].detach().to(left.device)
        families[name] = {
            "pass": bool(
                left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
            ),
            "dtype_actual": str(left.dtype),
            "dtype_expected": str(right.dtype),
            "shape_actual": list(left.shape),
            "shape_expected": list(right.shape),
            "maximum_absolute_difference": (
                float((left.double() - right.double()).abs().max())
                if left.shape == right.shape
                else None
            ),
        }
    return {
        "pass": all(item["pass"] for item in families.values()),
        "families": families,
        "actual_semantic_sha256": audit.semantic_sha256(actual),
        "expected_semantic_sha256": audit.semantic_sha256(expected),
    }


def development_transfer_decision(
    source: dict[str, Any],
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    preservation = base.preservation_decision(source, candidate)
    improvement = current["endpoint_damping_nrmse"] - candidate["endpoint_damping_nrmse"]
    reasons = list(preservation["reasons"])
    if improvement < DEVELOPMENT_DAMPING_IMPROVEMENT:
        reasons.append("development endpoint-D NRMSE improvement from update 20 was below 0.001")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "preservation": preservation,
        "endpoint_damping_nrmse_improvement_from_update_20": improvement,
        "minimum_endpoint_damping_nrmse_improvement": DEVELOPMENT_DAMPING_IMPROVEMENT,
    }


def selection_preconditions_pass(
    *,
    input_controls_pass: bool,
    transaction_controls_pass: bool,
    projection_controls_pass: bool,
) -> bool:
    return bool(input_controls_pass and transaction_controls_pass and projection_controls_pass)


def repair_numerical_control_report(trials: list[dict[str, Any]]) -> dict[str, Any]:
    repair_trials = [trial for trial in trials if "repair" in trial]
    if not repair_trials:
        return {"pass": True, "repair_path_present": False, "checks": {}}
    repair = repair_trials[-1]["repair"]
    starting = repair["starting_candidate"]
    checks = {
        "starting_candidate_canonicalization": starting["canonical_parameter_idempotence"]["pass"],
        "starting_candidate_parameter_bounds": starting["parameter_bounds"]["pass"],
    }
    if repair["attempted"]:
        checks.update(
            {
                "preplay": repair["pass_before_nonlinear_replay"],
                "directional_finite_difference": repair[
                    "endpoint_damping_directional_finite_difference"
                ]["pass"],
                "projection": repair["projection"]["pass"],
                "full_correction_canonicalization": repair[
                    "authoritative_parameter_canonicalization"
                ]["pass"],
                "full_correction_parameter_bounds": repair[
                    "effective_full_correction_parameter_bounds"
                ]["pass"],
                "optimizer_pending_state": repair["optimizer_pending_state"]["pass"],
            }
        )
        for index, trial in enumerate(repair["correction_trials"]):
            checks[f"correction_trial_{index + 1}_linear_values_finite"] = trial[
                "linearized_constraint_violations_finite"
            ]
            checks[f"correction_trial_{index + 1}_canonicalization"] = trial[
                "canonical_parameter_idempotence"
            ]["pass"]
            checks[f"correction_trial_{index + 1}_parameter_bounds"] = trial["parameter_bounds"][
                "pass"
            ]
    return {
        "pass": all(checks.values()),
        "repair_path_present": True,
        "repair_attempted": repair["attempted"],
        "checks": checks,
    }


@contextmanager
def corrected_trial_runtime(
    optimizer: torch.optim.Optimizer,
    optimizer_before: dict[str, Any],
    optimizer_after_sha256: str,
) -> Iterator[None]:
    previous = {
        "base_install_trial": base.install_trial,
        "guard_report": corrected._GUARD_REPORT,
        "update_number": corrected._UPDATE_NUMBER,
        "pending_optimizer": corrected._PENDING_OPTIMIZER,
        "optimizer_before": corrected._OPTIMIZER_BEFORE_PROPOSAL,
        "optimizer_after_sha256": corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
    }
    base.install_trial = canonical.install_trial
    corrected._GUARD_REPORT = None
    corrected._UPDATE_NUMBER = EXPECTED_ACCEPTED_UPDATES + 1
    corrected._PENDING_OPTIMIZER = optimizer
    corrected._OPTIMIZER_BEFORE_PROPOSAL = optimizer_before
    corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = optimizer_after_sha256
    try:
        yield
    finally:
        base.install_trial = previous["base_install_trial"]
        corrected._GUARD_REPORT = previous["guard_report"]
        corrected._UPDATE_NUMBER = previous["update_number"]
        corrected._PENDING_OPTIMIZER = previous["pending_optimizer"]
        corrected._OPTIMIZER_BEFORE_PROPOSAL = previous["optimizer_before"]
        corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = previous["optimizer_after_sha256"]


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    source_report_path = args.source_fit_dir / "report.json"
    source_resume_path = args.source_fit_dir / "resume.pt"
    qualification_report_path = args.qualification_dir / "report.json"
    archive_path = args.qualification_dir / "frozen-inputs.pt"
    locked_paths = {
        "source_report": source_report_path,
        "source_resume": source_resume_path,
        "qualification_report": qualification_report_path,
        "frozen_input_archive": archive_path,
    }
    hashes_before = {name: responsibility.file_sha256(path) for name, path in locked_paths.items()}
    expected_hashes = {
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "qualification_report": EXPECTED_QUALIFICATION_REPORT_SHA256,
        "frozen_input_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
    }
    if hashes_before != expected_hashes:
        raise SystemExit("one or more hash-locked inputs do not match the protocol")

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    qualification_report = json.loads(qualification_report_path.read_text(encoding="utf-8"))
    qualification_identity = (
        qualification_report.get("experiment"),
        qualification_report.get("protocol", {}).get("protocol_commit"),
        qualification_report.get("classification"),
        qualification_report.get("pass"),
        qualification_report.get("fp64_projection_implementation_authorized"),
    )
    if qualification_identity != (
        frozen.EXPERIMENT,
        frozen.PROTOCOL_COMMIT,
        "identical_input_fp64_projection_qualified",
        True,
        True,
    ):
        raise SystemExit("FP64 qualification identity does not match the protocol")

    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint_sha256 = responsibility.file_sha256(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    source_resume = torch.load(source_resume_path, map_location=device, weights_only=True)
    resume_identity = (
        source_resume.get("experiment"),
        source_resume.get("protocol_commit"),
        int(source_resume.get("accepted_updates", -1)),
        source_resume.get("run_state"),
    )
    if resume_identity != (
        corrected.EXPERIMENT,
        corrected.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "stopped",
    ):
        raise SystemExit("stopped update-20 resume identity does not match the protocol")
    student.load_state_dict(source_resume["controller"])
    optimizer = fp64.make_optimizer(student)
    optimizer.load_state_dict(source_resume["optimizer"])
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    print(json.dumps({"stage": "loading_frozen_archive_and_caches"}), flush=True)
    frozen_cpu = torch.load(archive_path, map_location="cpu", weights_only=True)
    archive_identity = (
        frozen_cpu.get("experiment"),
        frozen_cpu.get("protocol_commit"),
        frozen_cpu.get("source_resume_sha256"),
    )
    if archive_identity != (
        frozen.EXPERIMENT,
        frozen.PROTOCOL_COMMIT,
        EXPECTED_SOURCE_RESUME_SHA256,
    ):
        raise SystemExit("frozen tensor archive identity does not match the protocol")
    archive_semantic_hashes = frozen.archive_tensor_hashes(frozen_cpu)
    if archive_semantic_hashes != frozen_cpu.get("tensor_semantic_sha256"):
        raise SystemExit("frozen tensor archive semantic hashes do not match")
    if archive_semantic_hashes != qualification_report.get("frozen_input_archive", {}).get(
        "tensor_semantic_sha256"
    ):
        raise SystemExit("qualification report and frozen archive tensor hashes disagree")
    frozen_device = frozen._device_tensor_tree(frozen_cpu, device)
    del frozen_cpu
    current_parameter_match = all(
        torch.equal(current_parameters[name], frozen_device["current_parameters"][name])
        for name in joint.PARAMETER_FAMILIES
    )

    config = HoverConfig()
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
    source_training, _ = endpoint.evaluate(
        source,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    current_training, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    source_development, _ = endpoint.evaluate(
        source,
        development_factorial,
        development_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    current_development, _ = endpoint.evaluate(
        student,
        development_factorial,
        development_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    update20_development = next(
        entry["metrics"]
        for entry in source_report["development_history"]
        if int(entry["update"]) == EXPECTED_ACCEPTED_UPDATES
    )
    reproductions = {
        "source_training": audit.numeric_tree_comparison(
            source_training, source_report["source_training"]
        ),
        "update_20_training": audit.numeric_tree_comparison(
            current_training, source_report["final_training"]
        ),
        "source_development": audit.numeric_tree_comparison(
            source_development, source_report["source_development"]
        ),
        "update_20_development": audit.numeric_tree_comparison(
            current_development, update20_development
        ),
    }
    reproduction_pass = all(item["pass"] for item in reproductions.values())
    input_controls_pass = bool(
        current_parameter_match
        and cache_integrity["all_persisted_file_and_tensor_hashes_match"]
        and reproduction_pass
    )

    selected_scale: float | None = None
    selected_training: dict[str, Any] | None = None
    selected_development: dict[str, Any] | None = None
    selected_parameters: dict[str, Tensor] | None = None
    trials: list[dict[str, Any]] = []
    development_decision: dict[str, Any] | None = None
    print(json.dumps({"stage": "reconstructing_one_pending_adam_transaction"}), flush=True)
    with audit.transactional_restoration(student, optimizer):
        optimizer.zero_grad(set_to_none=True)
        for name in joint.PARAMETER_FAMILIES:
            getattr(student, name).grad = frozen_device["raw_damping_gradient"][name].clone()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
        )
        optimizer.step()
        student.project_parameters()
        raw_parameters = joint._copy_parameters(student)
        regenerated_raw = {
            name: raw_parameters[name] - current_parameters[name]
            for name in joint.PARAMETER_FAMILIES
        }
        optimizer_after = copy.deepcopy(optimizer.state_dict())
        optimizer_transaction = corrected.optimizer_step_transaction(
            optimizer_before, optimizer_after
        )
        optimizer_after_sha256 = audit.semantic_sha256(optimizer_after)
        gradient_norm_match = math.isclose(
            gradient_norm,
            float(frozen_device["gradient_norm_before_clipping"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        raw_displacement_match = parameter_tensor_comparison(
            regenerated_raw, frozen_device["raw_displacement"]
        )
        raw_displacement_match["exact_hash_match"] = bool(
            raw_displacement_match["actual_semantic_sha256"]
            == archive_semantic_hashes["raw_displacement"]
        )
        pending_optimizer_match = optimizer_after_sha256 == EXPECTED_PENDING_OPTIMIZER_SHA256
        joint._load_parameters(student, current_parameters)

        print(json.dumps({"stage": "qualified_fp64_primary_projection"}), flush=True)
        projected, projection = fp64.fp64_bound_aware_projection(
            frozen_device["raw_displacement"],
            frozen_device["constraint_rows"],
            frozen_device["constraint_specs"],
            frozen_device["current_parameters"]["edge_magnitude"],
        )
        projection_control = frozen.fp64_run_control(projection)
        projected_sha256 = audit.semantic_sha256(projected)
        primal_objective = frozen.complete_projection_primal_objective(
            projected, frozen_device["raw_displacement"]
        )
        projection_agreement = {
            "pass": bool(
                projected_sha256 == EXPECTED_PROJECTED_DISPLACEMENT_SHA256
                and relative_difference(primal_objective, EXPECTED_COMPLETE_PRIMAL_OBJECTIVE)
                <= OBJECTIVE_RELATIVE_TOLERANCE
            ),
            "projected_displacement_sha256": projected_sha256,
            "expected_projected_displacement_sha256": (EXPECTED_PROJECTED_DISPLACEMENT_SHA256),
            "complete_projection_primal_objective": primal_objective,
            "expected_complete_projection_primal_objective": (EXPECTED_COMPLETE_PRIMAL_OBJECTIVE),
            "objective_relative_difference": relative_difference(
                primal_objective, EXPECTED_COMPLETE_PRIMAL_OBJECTIVE
            ),
            "objective_relative_tolerance": OBJECTIVE_RELATIVE_TOLERANCE,
        }
        authoritative, effective, idempotence = canonical.materialize_authoritative_candidate(
            student, current_parameters, projected
        )
        bounds = audit.parameter_bounds_report(authoritative)
        linearized = audit.linearized_constraint_violations(
            frozen_device["constraint_specs"],
            frozen_device["constraint_rows"],
            effective,
        )
        maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
        damping_derivative = canonical._dot_float64(
            frozen_device["raw_damping_gradient"], effective
        )
        _, fd_displacement, fd_idempotence = fp64.install_fp64_trial(
            student,
            current_parameters,
            authoritative,
            scale=joint.FINITE_DIFFERENCE_SCALE,
        )
        fd_metrics, _ = endpoint.evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        fd_predicted = (
            canonical._dot_float64(frozen_device["raw_damping_gradient"], fd_displacement)
            / joint.FINITE_DIFFERENCE_SCALE
        )
        fd_actual = (
            fd_metrics["endpoint_damping_nrmse"] ** 2
            - current_training["endpoint_damping_nrmse"] ** 2
        ) / joint.FINITE_DIFFERENCE_SCALE
        fd_relative = abs(fd_actual - fd_predicted) / max(
            abs(fd_actual), abs(fd_predicted), 1.0e-12
        )
        finite_difference = {
            "pass": bool(
                math.isfinite(fd_predicted)
                and math.isfinite(fd_actual)
                and abs(fd_actual) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
                and fd_predicted < 0.0
                and fd_actual < 0.0
                and fd_relative <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                and fd_idempotence["pass"]
            ),
            "scale": joint.FINITE_DIFFERENCE_SCALE,
            "autograd_directional_derivative": fd_predicted,
            "complete_replay_finite_difference": fd_actual,
            "relative_error": fd_relative,
            "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
            "canonical_parameter_idempotence": fd_idempotence,
        }
        projection_post_controls = {
            "pass": bool(
                projection_control["pass"]
                and projection_agreement["pass"]
                and idempotence["pass"]
                and bounds["pass"]
                and linearized_finite
                and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
                and damping_derivative < 0.0
                and finite_difference["pass"]
            ),
            "canonical_parameter_idempotence": idempotence,
            "parameter_bounds": bounds,
            "linearized_constraint_violations": linearized,
            "maximum_linearized_constraint_violation": maximum_linearized,
            "damping_directional_derivative": damping_derivative,
            "finite_difference": finite_difference,
        }
        joint._load_parameters(student, current_parameters)
        canonical._AUTHORITATIVE_BASE = current_parameters
        canonical._AUTHORITATIVE_CANDIDATE = authoritative

        transaction_controls_pass = bool(
            optimizer_transaction["pass"]
            and gradient_norm_match
            and raw_displacement_match["pass"]
            and raw_displacement_match["exact_hash_match"]
            and pending_optimizer_match
        )
        selection_preconditions = selection_preconditions_pass(
            input_controls_pass=input_controls_pass,
            transaction_controls_pass=transaction_controls_pass,
            projection_controls_pass=projection_post_controls["pass"],
        )
        if selection_preconditions:
            print(json.dumps({"stage": "ordinary_then_guard_band_training_selection"}), flush=True)
            with corrected_trial_runtime(optimizer, optimizer_before, optimizer_after_sha256):
                selected_scale, selected_training, trials = corrected.find_safe_trial(
                    student,
                    current_parameters,
                    {
                        name: value.to(current_parameters[name].dtype)
                        for name, value in effective.items()
                    },
                    frozen_device["raw_damping_gradient"],
                    source_training,
                    current_training,
                    train_factorial,
                    train_attitude,
                    scales,
                    endpoint_scale,
                    device=device,
                )
                if selected_training is not None:
                    selected_parameters = joint._copy_parameters(student)
                    print(json.dumps({"stage": "single_fixed_development_transfer"}), flush=True)
                    selected_development, _ = endpoint.evaluate(
                        student,
                        development_factorial,
                        development_attitude,
                        scales,
                        endpoint_scale=endpoint_scale,
                        device=device,
                    )
                    development_decision = development_transfer_decision(
                        source_development,
                        current_development,
                        selected_development,
                    )
        else:
            trials = []

    repair_numerical_controls = repair_numerical_control_report(trials)
    parameters_restored = all(
        torch.equal(getattr(student, name).detach(), current_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_restored = audit.trees_equal(optimizer.state_dict(), optimizer_before)
    hashes_after = {name: responsibility.file_sha256(path) for name, path in locked_paths.items()}
    sources_unchanged = hashes_after == hashes_before
    training_selection_pass = selected_training is not None
    development_pass = bool(development_decision is not None and development_decision["pass"])
    controls_pass = bool(
        input_controls_pass
        and transaction_controls_pass
        and projection_post_controls["pass"]
        and repair_numerical_controls["pass"]
        and parameters_restored
        and optimizer_restored
        and sources_unchanged
    )
    passed = bool(controls_pass and training_selection_pass and development_pass)
    if not controls_pass:
        classification = "audit_control_failure"
    elif not training_selection_pass:
        classification = "no_training_candidate_passed"
    elif not development_pass:
        classification = "selected_training_candidate_failed_development_transfer"
    else:
        selected_kind = next(
            trial.get("acceptance_kind", "ordinary")
            for trial in trials
            if trial.get("decision", {}).get("pass")
        )
        classification = f"fp64_{selected_kind}_update_21_transfers_to_development"
    report = {
        "experiment": EXPERIMENT,
        "status": "restored_one_step_audit_no_retained_candidate",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "all_locked_files_unchanged": sources_unchanged,
        },
        "cache_integrity": cache_integrity,
        "frozen_archive": {
            "tensor_semantic_sha256": archive_semantic_hashes,
            "current_parameters_match_update_20": current_parameter_match,
            "jacobians_regenerated": False,
        },
        "reproductions": reproductions,
        "input_controls_pass_before_selection": input_controls_pass,
        "source_training": source_training,
        "update_20_training": current_training,
        "source_development": source_development,
        "update_20_development": current_development,
        "optimizer_transaction": optimizer_transaction,
        "gradient_norm_before_clipping": gradient_norm,
        "archived_gradient_norm_exact_match": gradient_norm_match,
        "raw_displacement_reproduction": raw_displacement_match,
        "pending_optimizer_sha256": optimizer_after_sha256,
        "pending_optimizer_sha256_match": pending_optimizer_match,
        "transaction_controls_pass": transaction_controls_pass,
        "primary_fp64_projection": projection,
        "primary_fp64_projection_control": projection_control,
        "qualification_agreement": projection_agreement,
        "post_projection_controls": projection_post_controls,
        "selection_preconditions_pass": selection_preconditions,
        "training_trials": trials,
        "repair_numerical_controls": repair_numerical_controls,
        "selected_training_scale": selected_scale,
        "selected_training_metrics": selected_training,
        "selected_parameter_sha256": (
            audit.semantic_sha256(selected_parameters) if selected_parameters is not None else None
        ),
        "selected_development_metrics": selected_development,
        "development_transfer_decision": development_decision,
        "controls_pass": controls_pass,
        "parameters_restored_exactly_to_update_20": parameters_restored,
        "optimizer_restored_exactly_to_update_20": optimizer_restored,
        "candidate_retained": False,
        "continuation_checkpoint_written": False,
        "fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "fp64_corrected_continuation_authorized": passed,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass authorizes only a separately registered continuation toward total "
            "update 50; it does not establish useful damping, hover, or flight."
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
                "training_selection_pass": training_selection_pass,
                "development_transfer_pass": development_pass,
                "candidate_retained": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
