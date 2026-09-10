#!/usr/bin/env python3
"""Qualify FP64 projection determinism on one persisted update-21 tensor set."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-frozen-input-audit-v1"
PROTOCOL_COMMIT = "cb7c7aa"
EXPECTED_SOURCE_REPORT_SHA256 = fp64.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = fp64.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_FAILED_AUDIT_REPORT_SHA256 = (
    "8e92a28ba2c67f76ee8d08426ccc0651e915007111085772e3c81ecfabf0ca80"
)
EXPECTED_ACCEPTED_UPDATES = fp64.EXPECTED_ACCEPTED_UPDATES
FP64_REPEATS = 3
PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT = 1.0e-10
OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT = 1.0e-12


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
        "--failed-audit-report",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-projection-audit-001/report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fp64-frozen-input-audit-001"
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
        args.failed_audit_report,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered qualification audit has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": corrected.EXPERIMENT,
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
        "preserved_failed_audit_report_sha256": EXPECTED_FAILED_AUDIT_REPORT_SHA256,
        "source_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "actor_contract_unchanged": True,
        "tensor_generation": {
            "one_update_21_adam_transaction": True,
            "adam_counter_increment": 1,
            "actual_tensors_persisted_before_solver": True,
            "archive_device": "CPU",
            "archive_ignored": True,
            "archive_reloaded_for_each_projection": True,
            "contents": [
                "update-20 parameters",
                "raw update-21 displacement",
                "ordered constraint rows and specifications",
                "endpoint-D gradient",
                "gradient norm",
            ],
        },
        "fp64_repeats_from_empty_active_set": FP64_REPEATS,
        "fp64_projection_protocol": fp64.protocol_manifest()["fp64_projection"],
        "primary_repeat_agreement": {
            "learning_rate_scaled_primal_relative_distance_maximum": (
                PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT
            ),
            "complete_projection_primal_objective_relative_difference_maximum": (
                OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
            ),
            "dual_coefficients_compared": False,
            "iteration_counts_compared": False,
            "historical_status_compared": False,
        },
        "independent_solver": {
            "name": "exhaustive active-support Lawson-Hanson nonnegative least squares",
            "maximum_supports": 2**10,
            "stationarity_system": "G_SS * lambda_S = v_S",
            "full_original_qp_kkt_required": True,
            "relative_rank_cutoff": fp64.RANK_RELATIVE_EIGENVALUE_CUTOFF,
            "violation_outside_retained_eigenspace_reported": True,
            "violation_outside_retained_eigenspace_discarded": False,
            "gram_induced_primal_relative_distance_maximum": (
                fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
            ),
            "dual_objective_relative_difference_maximum": (
                fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
            ),
            "dual_coefficients_compared": False,
            "candidate_selected": False,
        },
        "production_projection": "diagnostic only on the frozen archive",
        "post_projection_gates": {
            "first_primary_fp64_result_only": True,
            "canonical_parameter_idempotence_maximum": (canonical.PARAMETER_IDEMPOTENCE_TOLERANCE),
            "native_parameter_bounds": True,
            "linear_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
            "negative_endpoint_damping_derivative": True,
            "finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_maximum": (
                joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "ordinary_nonlinear_acceptance_required": False,
            "nonlinear_repair": False,
        },
        "parameters_optimizer_and_all_source_files_restored_exactly": True,
        "development_or_fresh_data": False,
        "candidate_retained": False,
        "closed_loop_hover": False,
        "promotion": False,
        "pass_authorizes": "one separately registered FP64-projected-plus-repaired step audit",
    }


def _cpu_tensor_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tensor_tree(item) for item in value]
    return copy.deepcopy(value)


def _device_tensor_tree(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _device_tensor_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tensor_tree(item, device) for item in value]
    return copy.deepcopy(value)


def archive_tensor_hashes(archive: dict[str, Any]) -> dict[str, str]:
    return {
        key: audit.semantic_sha256(archive[key])
        for key in (
            "current_parameters",
            "raw_displacement",
            "constraint_rows",
            "constraint_specs",
            "raw_damping_gradient",
        )
    }


def learning_rate_scaled_norm(displacement: dict[str, Tensor]) -> float:
    return math.sqrt(
        sum(
            float(
                (displacement[name].double() / base.PARAMETER_LEARNING_RATES[name]).square().sum()
            )
            for name in joint.PARAMETER_FAMILIES
        )
    )


def complete_projection_primal_objective(
    projected: dict[str, Tensor], raw: dict[str, Tensor]
) -> float:
    correction = {
        name: projected[name].double() - raw[name].double() for name in joint.PARAMETER_FAMILIES
    }
    return 0.5 * learning_rate_scaled_norm(correction) ** 2


def primary_repeat_agreement(
    displacements: list[dict[str, Tensor]], objectives: list[float]
) -> dict[str, Any]:
    pairs = []
    for left, right in combinations(range(len(displacements)), 2):
        difference = {
            name: displacements[left][name].double() - displacements[right][name].double()
            for name in joint.PARAMETER_FAMILIES
        }
        relative_distance = learning_rate_scaled_norm(difference) / max(
            1.0, learning_rate_scaled_norm(displacements[left])
        )
        objective_difference = abs(objectives[left] - objectives[right]) / max(
            1.0, abs(objectives[left])
        )
        pairs.append(
            {
                "left_repeat": left + 1,
                "right_repeat": right + 1,
                "learning_rate_scaled_primal_relative_distance": relative_distance,
                "complete_projection_primal_objective_relative_difference": (objective_difference),
                "pass": bool(
                    relative_distance <= PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT
                    and objective_difference <= OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
                ),
            }
        )
    return {
        "pass": bool(pairs and all(pair["pass"] for pair in pairs)),
        "pairs": pairs,
        "learning_rate_scaled_primal_relative_distance_limit": (
            PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT
        ),
        "complete_projection_primal_objective_relative_difference_limit": (
            OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
        ),
    }


def fp64_run_control(projection: dict[str, Any]) -> dict[str, Any]:
    nnls_pass = all(
        round_report["free_projection"]["independent_nnls"]["pass"]
        for round_report in projection["rounds"]
    )
    return {
        "pass": bool(projection["pass"] and nnls_pass),
        "projection_pass": projection["pass"],
        "independent_nnls_every_round_pass": nnls_pass,
        "rounds": len(projection["rounds"]),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    source_report_path = args.source_fit_dir / "report.json"
    source_resume_path = args.source_fit_dir / "resume.pt"
    source_report_sha_before = responsibility.file_sha256(source_report_path)
    source_resume_sha_before = responsibility.file_sha256(source_resume_path)
    failed_audit_sha_before = responsibility.file_sha256(args.failed_audit_report)
    if source_report_sha_before != EXPECTED_SOURCE_REPORT_SHA256:
        raise SystemExit("corrected fitting report hash does not match the protocol")
    if source_resume_sha_before != EXPECTED_SOURCE_RESUME_SHA256:
        raise SystemExit("corrected fitting resume hash does not match the protocol")
    if failed_audit_sha_before != EXPECTED_FAILED_AUDIT_REPORT_SHA256:
        raise SystemExit("failed FP64 audit report hash does not match the protocol")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    failed_audit = json.loads(args.failed_audit_report.read_text(encoding="utf-8"))
    failed_identity = (
        failed_audit.get("experiment"),
        failed_audit.get("classification"),
        failed_audit.get("pass"),
    )
    if failed_identity != (fp64.EXPERIMENT, "audit_control_failure", False):
        raise SystemExit("failed FP64 audit identity does not match the protocol")
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
        raise SystemExit("update-20 resume identity does not match the protocol")

    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint_sha256 = responsibility.file_sha256(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    student.load_state_dict(source_resume["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    optimizer = fp64.make_optimizer(student)
    optimizer.load_state_dict(source_resume["optimizer"])
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    print(json.dumps({"stage": "loading_authorized_immutable_training_cache"}), flush=True)
    config = HoverConfig()
    (
        train_factorial,
        train_attitude,
        _,
        _,
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
    source_metrics, _ = endpoint.evaluate(
        source,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    current_metrics, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    source_reproduction = audit.numeric_tree_comparison(
        source_metrics, source_report["source_training"]
    )
    current_reproduction = audit.numeric_tree_comparison(
        current_metrics, source_report["final_training"]
    )

    print(json.dumps({"stage": "generating_and_persisting_frozen_inputs"}), flush=True)
    specs = base.constraint_specs(source_report["source_training"], source_report["final_training"])
    rows = base.constraint_gradient_rows(
        student,
        train_factorial,
        train_attitude,
        scales,
        specs,
        device=device,
    )
    with audit.transactional_restoration(student, optimizer):
        optimizer.zero_grad(set_to_none=True)
        endpoint.accumulated_endpoint_damping_gradient(
            student,
            train_factorial,
            scale=endpoint_scale,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        raw_gradient = {
            name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
        }
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
        )
        optimizer.step()
        student.project_parameters()
        raw_parameters = joint._copy_parameters(student)
        raw_displacement = {
            name: raw_parameters[name] - current_parameters[name]
            for name in joint.PARAMETER_FAMILIES
        }
        optimizer_after = copy.deepcopy(optimizer.state_dict())
        optimizer_transaction = corrected.optimizer_step_transaction(
            optimizer_before, optimizer_after
        )
    archive = _cpu_tensor_tree(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
            "current_parameters": current_parameters,
            "raw_displacement": raw_displacement,
            "constraint_rows": rows,
            "constraint_specs": specs,
            "raw_damping_gradient": raw_gradient,
            "gradient_norm_before_clipping": gradient_norm,
        }
    )
    tensor_hashes = archive_tensor_hashes(archive)
    archive["tensor_semantic_sha256"] = tensor_hashes
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "frozen-inputs.pt"
    base.atomic_torch_save(archive, archive_path)
    archive_file_sha256 = responsibility.file_sha256(archive_path)
    reloaded_cpu = torch.load(archive_path, map_location="cpu", weights_only=True)
    archive_reload = {
        "pass": bool(
            reloaded_cpu.get("experiment") == EXPERIMENT
            and reloaded_cpu.get("protocol_commit") == PROTOCOL_COMMIT
            and archive_tensor_hashes(reloaded_cpu) == tensor_hashes
        ),
        "file_sha256": archive_file_sha256,
        "tensor_semantic_sha256": archive_tensor_hashes(reloaded_cpu),
        "persisted_before_any_projection": True,
    }
    del reloaded_cpu, archive, rows, raw_displacement, raw_gradient

    print(json.dumps({"stage": "three_frozen_input_fp64_replays"}), flush=True)
    fp64_runs = []
    fp64_displacements = []
    objectives = []
    for repeat in range(1, FP64_REPEATS + 1):
        frozen = _device_tensor_tree(
            torch.load(archive_path, map_location="cpu", weights_only=True), device
        )
        displacement, projection = fp64.fp64_bound_aware_projection(
            frozen["raw_displacement"],
            frozen["constraint_rows"],
            frozen["constraint_specs"],
            frozen["current_parameters"]["edge_magnitude"],
        )
        control = fp64_run_control(projection)
        final_dual_objective = float(projection["rounds"][-1]["free_projection"]["dual_objective"])
        objective = complete_projection_primal_objective(displacement, frozen["raw_displacement"])
        fp64_runs.append(
            {
                "repeat": repeat,
                "control": control,
                "projection": projection,
                "final_displacement_sha256": audit.semantic_sha256(displacement),
                "learning_rate_scaled_displacement_norm": (learning_rate_scaled_norm(displacement)),
                "complete_projection_primal_objective": objective,
                "final_free_round_dual_objective_diagnostic": final_dual_objective,
            }
        )
        fp64_displacements.append(displacement)
        objectives.append(objective)
        del frozen
    repeat_agreement = primary_repeat_agreement(fp64_displacements, objectives)

    print(json.dumps({"stage": "frozen_input_production_diagnostic"}), flush=True)
    production_frozen = _device_tensor_tree(
        torch.load(archive_path, map_location="cpu", weights_only=True), device
    )
    with audit.transactional_restoration(student, optimizer):
        _, production = fp64.production_projection(
            student,
            production_frozen["current_parameters"],
            production_frozen["raw_displacement"],
            production_frozen["raw_damping_gradient"],
            production_frozen["constraint_rows"],
            production_frozen["constraint_specs"],
            gradient_norm=float(production_frozen["gradient_norm_before_clipping"]),
        )
    del production_frozen

    print(json.dumps({"stage": "first_fp64_post_projection_controls"}), flush=True)
    post_frozen = _device_tensor_tree(
        torch.load(archive_path, map_location="cpu", weights_only=True), device
    )
    with audit.transactional_restoration(student, optimizer):
        authoritative, effective, idempotence = canonical.materialize_authoritative_candidate(
            student,
            post_frozen["current_parameters"],
            fp64_displacements[0],
        )
        bounds = audit.parameter_bounds_report(authoritative)
        linearized = audit.linearized_constraint_violations(
            post_frozen["constraint_specs"], post_frozen["constraint_rows"], effective
        )
        maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
        derivative = canonical._dot_float64(post_frozen["raw_damping_gradient"], effective)
        _, actual_fd, fd_idempotence = fp64.install_fp64_trial(
            student,
            post_frozen["current_parameters"],
            authoritative,
            scale=joint.FINITE_DIFFERENCE_SCALE,
        )
        finite_difference_metrics, _ = endpoint.evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        fd_derivative = (
            canonical._dot_float64(post_frozen["raw_damping_gradient"], actual_fd)
            / joint.FINITE_DIFFERENCE_SCALE
        )
        fd_actual = (
            finite_difference_metrics["endpoint_damping_nrmse"] ** 2
            - current_metrics["endpoint_damping_nrmse"] ** 2
        ) / joint.FINITE_DIFFERENCE_SCALE
        fd_relative = abs(fd_actual - fd_derivative) / max(
            abs(fd_actual), abs(fd_derivative), 1.0e-12
        )
        finite_difference = {
            "pass": bool(
                math.isfinite(fd_derivative)
                and math.isfinite(fd_actual)
                and abs(fd_actual) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
                and fd_derivative < 0.0
                and fd_actual < 0.0
                and fd_relative <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                and fd_idempotence["pass"]
            ),
            "scale": joint.FINITE_DIFFERENCE_SCALE,
            "autograd_directional_derivative": fd_derivative,
            "complete_replay_finite_difference": fd_actual,
            "relative_error": fd_relative,
            "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
            "canonical_parameter_idempotence": fd_idempotence,
        }
    post_projection = {
        "pass": bool(
            idempotence["pass"]
            and bounds["pass"]
            and linearized_finite
            and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
            and derivative < 0.0
            and finite_difference["pass"]
        ),
        "canonical_parameter_idempotence": idempotence,
        "parameter_bounds": bounds,
        "linearized_constraint_violations": linearized,
        "maximum_linearized_constraint_violation": maximum_linearized,
        "damping_directional_derivative": derivative,
        "finite_difference": finite_difference,
        "ordinary_nonlinear_acceptance_required": False,
        "nonlinear_repair_attempted": False,
    }
    del post_frozen

    joint._load_parameters(student, current_parameters)
    optimizer.load_state_dict(optimizer_before)
    optimizer.zero_grad(set_to_none=True)
    parameters_restored = all(
        torch.equal(getattr(student, name).detach(), current_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_restored = audit.trees_equal(optimizer.state_dict(), optimizer_before)
    source_report_sha_after = responsibility.file_sha256(source_report_path)
    source_resume_sha_after = responsibility.file_sha256(source_resume_path)
    failed_audit_sha_after = responsibility.file_sha256(args.failed_audit_report)
    sources_unchanged = bool(
        source_report_sha_after == source_report_sha_before
        and source_resume_sha_after == source_resume_sha_before
        and failed_audit_sha_after == failed_audit_sha_before
    )
    controls_pass = bool(
        cache_integrity["all_persisted_file_and_tensor_hashes_match"]
        and source_reproduction["pass"]
        and current_reproduction["pass"]
        and optimizer_transaction["pass"]
        and archive_reload["pass"]
        and parameters_restored
        and optimizer_restored
        and sources_unchanged
    )
    all_fp64_runs_pass = all(run["control"]["pass"] for run in fp64_runs)
    passed = bool(
        controls_pass
        and all_fp64_runs_pass
        and repeat_agreement["pass"]
        and post_projection["pass"]
    )
    if not controls_pass:
        classification = "audit_control_failure"
    elif not all_fp64_runs_pass:
        classification = "one_or_more_identical_input_fp64_runs_failed"
    elif not repeat_agreement["pass"]:
        classification = "identical_input_fp64_primal_results_disagreed"
    elif not post_projection["pass"]:
        classification = "fp64_projection_failed_original_numerical_gates"
    else:
        classification = "identical_input_fp64_projection_qualified"
    report = {
        "experiment": EXPERIMENT,
        "status": "restored_numerical_qualification_no_retained_candidate",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "corrected_fit_report_sha256_before": source_report_sha_before,
            "corrected_fit_report_sha256_after": source_report_sha_after,
            "corrected_fit_resume_sha256_before": source_resume_sha_before,
            "corrected_fit_resume_sha256_after": source_resume_sha_after,
            "failed_fp64_audit_report_sha256_before": failed_audit_sha_before,
            "failed_fp64_audit_report_sha256_after": failed_audit_sha_after,
            "all_source_files_unchanged": sources_unchanged,
        },
        "cache_integrity": cache_integrity,
        "source_training": source_metrics,
        "update_20_training": current_metrics,
        "source_training_reproduction": source_reproduction,
        "update_20_training_reproduction": current_reproduction,
        "optimizer_one_step_transaction": optimizer_transaction,
        "frozen_input_archive": {
            "path": responsibility.stable_path(archive_path),
            **archive_reload,
        },
        "fp64_runs": fp64_runs,
        "primary_repeat_agreement": repeat_agreement,
        "production_projection_diagnostic": production,
        "post_projection": post_projection,
        "controls_pass": controls_pass,
        "parameters_restored_exactly_to_update_20": parameters_restored,
        "optimizer_restored_exactly_to_update_20": optimizer_restored,
        "candidate_retained": False,
        "development_or_fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "fp64_projection_implementation_authorized": passed,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass qualifies only identical-input FP64 projection numerics. It cannot "
            "accept update 21, bypass nonlinear preservation, or authorize hover."
        ),
    }
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": classification,
                "all_fp64_runs_pass": all_fp64_runs_pass,
                "repeat_agreement_pass": repeat_agreement["pass"],
                "post_projection_pass": post_projection["pass"],
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
