#!/usr/bin/env python3
"""Audit frozen update-25 projection with one SLSQP-support SVD polish."""

from __future__ import annotations

import argparse
import json
import math
import os
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

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_update25_exhaustive as audit25  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_fp64_snapshot_corrected as source_fit  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-update25-slsqp-polished-audit-v1"
PROTOCOL_COMMIT = "3340c27"
EXPECTED_GRAPH_SHA256 = audit25.EXPECTED_GRAPH_SHA256
EXPECTED_CHECKPOINT_SHA256 = audit25.EXPECTED_CHECKPOINT_SHA256
EXPECTED_SOURCE_REPORT_SHA256 = audit25.EXPECTED_SOURCE_REPORT_SHA256
EXPECTED_SOURCE_RESUME_SHA256 = audit25.EXPECTED_SOURCE_RESUME_SHA256
EXPECTED_FAILED_PRODUCER_REPORT_SHA256 = (
    "7022b5e7768279574166aca7186f8155e4904f6a5b3bd8a5d4521cf9a3e67995"
)
EXPECTED_FAILED_AUDIT_REPORT_SHA256 = (
    "7b9532798c7bb55ea8d27bd06dd026d6de4c54847a10c108120a22bfc0bfc47a"
)
EXPECTED_FROZEN_ARCHIVE_SHA256 = "5fc3b908962986c43f511565296950ca06f0f04d80a9d2b3266c9dbb2fcc4f61"
EXPECTED_ACCEPTED_UPDATES = audit25.EXPECTED_ACCEPTED_UPDATES
RECONSTRUCTED_UPDATE = audit25.RECONSTRUCTED_UPDATE
REPEATS = 3
SLSQP_SUPPORT_RELATIVE_THRESHOLD = 1.0e-10
SVD_RELATIVE_CUTOFF = 1.0e-12
STARTED_NAME = "verifier-started.json"
FINAL_REPORT_NAME = "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize-interrupted", action="store_true")
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
        "--failed-audit-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update25-exhaustive-audit-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/"
            "full-native-d-first-fp64-update25-slsqp-polished-audit-001"
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
        "failed_producer_report": args.failed_audit_dir / audit25.PRODUCER_REPORT_NAME,
        "failed_audit_report": args.failed_audit_dir / audit25.FINAL_REPORT_NAME,
        "frozen_archive": args.failed_audit_dir / audit25.ARCHIVE_NAME,
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": EXPECTED_GRAPH_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": base.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume": EXPECTED_SOURCE_RESUME_SHA256,
        "failed_producer_report": EXPECTED_FAILED_PRODUCER_REPORT_SHA256,
        "failed_audit_report": EXPECTED_FAILED_AUDIT_REPORT_SHA256,
        "frozen_archive": EXPECTED_FROZEN_ARCHIVE_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in locked_input_paths(args).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered polished-reference audit has no smoke variant")
    started = args.output_dir / STARTED_NAME
    final = args.output_dir / FINAL_REPORT_NAME
    if args.finalize_interrupted:
        if final.exists() or not started.is_file():
            raise SystemExit("no interrupted verifier is available to finalize")
    elif started.exists() or final.exists():
        raise SystemExit("verifier output already exists; this audit cannot be retried")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_input_sha256": expected_locked_hashes(),
        "frozen_update_25_inputs_reused_without_regeneration": True,
        "primary_solver_unchanged": (
            "exhaustive active-support Lawson-Hanson nonnegative least squares"
        ),
        "slsqp": {
            "run_once_per_round": True,
            "initial_dual": "zero",
            "analytic_gradient": True,
            "nonnegative_bounds": True,
            "ftol": fp64.INDEPENDENT_SLSQP_FTOL,
            "maximum_iterations": fp64.MAXIMUM_SOLVER_ITERATIONS,
            "unpolished_result_is_diagnostic_only": True,
        },
        "support_polish": {
            "support_source": "SLSQP coefficients only",
            "relative_threshold": SLSQP_SUPPORT_RELATIVE_THRESHOLD,
            "threshold_formula": "lambda_i > threshold * max(1, infinity_norm(lambda))",
            "primary_support_consulted": False,
            "solve": "one direct FP64 SVD solve of G_SS lambda_S = v_S",
            "relative_singular_value_cutoff": SVD_RELATIVE_CUTOFF,
            "full_support_rank_required": True,
            "retries_or_refinement": False,
            "alternate_threshold_or_regularization": False,
            "fallback": False,
        },
        "polished_reference_gates": {
            "finite_and_nonnegative_coefficients": True,
            "full_original_qp_normalized_kkt_maximum": fp64.NORMALIZED_KKT_RESIDUAL_LIMIT,
            "original_unit_primal_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
            "gram_induced_primal_relative_distance_maximum": (
                fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
            ),
            "dual_objective_relative_difference_maximum": (
                fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
            ),
            "raw_dual_coefficients_compared": False,
        },
        "lbfgsb_and_unpolished_slsqp_are_diagnostic_only": True,
        "active_set": {
            "monotonic_actual_boundary_heuristic_unchanged": True,
            "maximum_rounds": fp64.MAXIMUM_ACTIVE_SET_ROUNDS,
        },
        "repeats_from_fresh_cpu_archive_load": REPEATS,
        "active_and_fixed_set_paths_exact": True,
        "learning_rate_scaled_primal_relative_distance_maximum": (
            frozen.PRIMAL_REPEAT_RELATIVE_DISTANCE_LIMIT
        ),
        "objective_relative_difference_maximum": (
            frozen.OBJECTIVE_REPEAT_RELATIVE_DIFFERENCE_LIMIT
        ),
        "actor_instantiated_or_candidate_materialized": False,
        "gradients_metrics_or_adam_regenerated": False,
        "development_or_fresh_data": False,
        "state_retained": False,
        "closed_loop_hover": False,
        "promotion": False,
        "failure_action": "pause this exhaustive-primary integration route",
        "pass_authorizes": (
            "only a separately preregistered continuation from exact accepted update 24"
        ),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    with staging.open("w", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(value, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def write_started(args: argparse.Namespace) -> Path:
    marker = args.output_dir / STARTED_NAME
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "phase": "verify",
        "no_retry": True,
    }
    with marker.open("x", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(payload, indent=2, sort_keys=True)}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return marker


def finalize_interrupted(args: argparse.Namespace) -> int:
    marker = args.output_dir / STARTED_NAME
    report = {
        "experiment": EXPERIMENT,
        "status": "frozen_numerical_reference_audit",
        "pass": False,
        "classification": "verifier_interrupted_after_phase_start",
        "protocol": protocol_manifest(),
        "verifier_marker_sha256": responsibility.file_sha256(marker),
        "phase_retried": False,
        "continuation_authorized": False,
        "actor_candidate_materialized": False,
        "state_retained": False,
        "development_or_fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": False,
                "classification": report["classification"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def validate_locked_identities(args: argparse.Namespace) -> dict[str, Any]:
    hashes = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if hashes != expected_locked_hashes():
        raise SystemExit("one or more hash-locked inputs do not match the protocol")
    source_report = json.loads((args.source_fit_dir / "report.json").read_text(encoding="utf-8"))
    source_resume = torch.load(
        args.source_fit_dir / "resume.pt", map_location="cpu", weights_only=True
    )
    failed_producer = json.loads(
        (args.failed_audit_dir / audit25.PRODUCER_REPORT_NAME).read_text(encoding="utf-8")
    )
    failed_report = json.loads(
        (args.failed_audit_dir / audit25.FINAL_REPORT_NAME).read_text(encoding="utf-8")
    )
    last = source_report.get("history", [{}])[-1]
    source_identity = (
        source_report.get("experiment"),
        source_report.get("protocol", {}).get("protocol_commit"),
        source_report.get("accepted_updates"),
        source_report.get("stop_reason"),
        last.get("update"),
        last.get("accepted"),
        source_resume.get("experiment"),
        source_resume.get("protocol_commit"),
        source_resume.get("accepted_updates"),
        source_resume.get("run_state"),
    )
    if source_identity != (
        source_fit.EXPERIMENT,
        source_fit.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "constraint projection failed",
        RECONSTRUCTED_UPDATE,
        False,
        source_fit.EXPERIMENT,
        source_fit.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "stopped",
    ):
        raise SystemExit("stopped update-24 source identity does not match")
    if (
        failed_producer.get("experiment"),
        failed_producer.get("classification"),
        failed_producer.get("pass"),
        failed_report.get("experiment"),
        failed_report.get("classification"),
        failed_report.get("pass"),
        failed_report.get("continuation_authorized"),
    ) != (
        audit25.EXPERIMENT,
        "frozen_update_25_inputs_produced",
        True,
        audit25.EXPERIMENT,
        "one_or_more_exhaustive_primary_runs_failed",
        False,
        False,
    ):
        raise SystemExit("failed exhaustive-primary audit identity does not match")
    archive_path = args.failed_audit_dir / audit25.ARCHIVE_NAME
    archive = torch.load(archive_path, map_location="cpu", weights_only=True)
    archive_identity = bool(
        archive.get("experiment") == audit25.EXPERIMENT
        and archive.get("protocol_commit") == audit25.PROTOCOL_COMMIT
        and archive.get("accepted_updates_before") == EXPECTED_ACCEPTED_UPDATES
        and archive.get("reconstructed_update") == RECONSTRUCTED_UPDATE
        and audit25.archive_semantic_hashes(archive) == archive.get("tensor_semantic_sha256")
        and failed_producer.get("archive", {}).get("file_sha256") == EXPECTED_FROZEN_ARCHIVE_SHA256
        and failed_report.get("archive", {}).get("file_sha256_before")
        == EXPECTED_FROZEN_ARCHIVE_SHA256
    )
    if not archive_identity:
        raise SystemExit("frozen update-25 archive identity or semantic hashes failed")
    return {
        "hashes": hashes,
        "source_report": source_report,
        "source_resume": source_resume,
        "failed_producer": failed_producer,
        "failed_report": failed_report,
        "archive_semantic_sha256": audit25.archive_semantic_hashes(archive),
    }


def slsqp_support(dual: np.ndarray) -> tuple[np.ndarray, float]:
    threshold = SLSQP_SUPPORT_RELATIVE_THRESHOLD * max(1.0, float(np.linalg.norm(dual, ord=np.inf)))
    return np.flatnonzero(dual > threshold).astype(np.int64), threshold


def polish_slsqp_support(
    normalized_gram: np.ndarray,
    normalized_violation: np.ndarray,
    slsqp_dual: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    symmetric = 0.5 * (normalized_gram + normalized_gram.T)
    support, threshold = slsqp_support(slsqp_dual)
    polished = np.zeros_like(slsqp_dual, dtype=np.float64)
    singular_values = np.array([], dtype=np.float64)
    rank = 0
    if support.size:
        support_gram = symmetric[np.ix_(support, support)]
        support_violation = normalized_violation[support]
        left, singular_values, right_transpose = np.linalg.svd(support_gram, full_matrices=False)
        cutoff = SVD_RELATIVE_CUTOFF * float(singular_values.max(initial=0.0))
        retained = singular_values > cutoff
        rank = int(retained.sum())
        if rank == support.size:
            polished[support] = right_transpose.T @ ((left.T @ support_violation) / singular_values)
    else:
        cutoff = 0.0
    finite = bool(np.all(np.isfinite(polished)))
    nonnegative = bool(finite and np.all(polished >= 0.0))
    kkt = fp64._projected_gradient_report(symmetric, normalized_violation, polished)
    full_rank = rank == support.size
    return polished, {
        "pass": bool(full_rank and finite and nonnegative and kkt["pass"]),
        "support_source": "unpolished SLSQP coefficients only",
        "support_threshold": threshold,
        "support_indices": support.tolist(),
        "support_mask": int(sum(1 << int(index) for index in support)),
        "support_size": int(support.size),
        "svd_relative_cutoff": SVD_RELATIVE_CUTOFF,
        "svd_absolute_cutoff": cutoff,
        "singular_values": singular_values.tolist(),
        "rank": rank,
        "full_support_rank": full_rank,
        "coefficients_finite": finite,
        "coefficients_nonnegative": nonnegative,
        "minimum_coefficient": float(polished.min(initial=0.0)),
        "normalized_kkt": kkt,
        "dual_coefficients": polished.tolist(),
        "primary_support_consulted": False,
        "solve_count": 1,
        "iterative_refinement": False,
        "fallback": False,
        "used_as_independent_reference": True,
    }


def polished_free_projection(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    rows64 = [{name: value.detach().double() for name, value in row.items()} for row in rows]
    projected = {name: value.detach().double().clone() for name, value in displacement.items()}
    count = len(rows64)
    gram = torch.empty(
        count,
        count,
        dtype=torch.float64,
        device=next(iter(rows64[0].values())).device,
    )
    j_displacement = torch.empty(count, dtype=torch.float64, device=gram.device)
    for row_index, row in enumerate(rows64):
        j_displacement[row_index] = sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
        for column_index in range(row_index + 1):
            other = rows64[column_index]
            value = sum(
                (base.PARAMETER_LEARNING_RATES[name] ** 2) * (row[name] * other[name]).sum()
                for name in joint.PARAMETER_FAMILIES
            )
            gram[row_index, column_index] = value
            gram[column_index, row_index] = value
    residual = torch.tensor(
        [spec["current_mse"] - spec["limit_mse"] for spec in specs],
        dtype=torch.float64,
        device=gram.device,
    )
    violation_before = residual + j_displacement
    row_norm = gram.diagonal().clamp_min(0.0).sqrt()
    active = row_norm > 1.0e-15
    infeasible_zero = (~active) & (violation_before > base.LINEAR_CONSTRAINT_TOLERANCE)
    if bool(infeasible_zero.any()):
        return projected, {
            "pass": False,
            "reason": "a violated constraint has a numerically zero Jacobian row",
            "active_rows": int(active.sum()),
        }
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    normalized_gram = (
        (gram[active][:, active] / (row_norm[active, None] * row_norm[None, active])).cpu().numpy()
    )
    normalized_violation = (violation_before[active] / row_norm[active]).cpu().numpy()
    primary_dual, primary = audit25.exhaustive_primary_dual(normalized_gram, normalized_violation)
    slsqp_dual, slsqp = audit25._diagnostic_solver("SLSQP", normalized_gram, normalized_violation)
    polished_dual, polished = polish_slsqp_support(
        normalized_gram, normalized_violation, slsqp_dual
    )
    _, lbfgsb = audit25._diagnostic_solver("L-BFGS-B", normalized_gram, normalized_violation)

    primary_coefficients = torch.zeros_like(violation_before)
    primary_coefficients[active_indices] = (
        torch.from_numpy(primary_dual).to(gram.device) / row_norm[active]
    )
    for coefficient, row in zip(primary_coefficients, rows64, strict=True):
        for name in joint.PARAMETER_FAMILIES:
            projected[name].add_(
                row[name],
                alpha=-(base.PARAMETER_LEARNING_RATES[name] ** 2) * float(coefficient),
            )
    primary_linearized = residual.clone()
    for row_index, row in enumerate(rows64):
        primary_linearized[row_index] += sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
    primary_maximum = float(primary_linearized.max())
    primary_primal_pass = bool(
        math.isfinite(primary_maximum) and primary_maximum <= base.LINEAR_CONSTRAINT_TOLERANCE
    )

    polished_coefficients = torch.zeros_like(violation_before)
    polished_coefficients[active_indices] = (
        torch.from_numpy(polished_dual).to(gram.device) / row_norm[active]
    )
    polished_displacement = {
        name: value.detach().double().clone() for name, value in displacement.items()
    }
    for coefficient, row in zip(polished_coefficients, rows64, strict=True):
        for name in joint.PARAMETER_FAMILIES:
            polished_displacement[name].add_(
                row[name],
                alpha=-(base.PARAMETER_LEARNING_RATES[name] ** 2) * float(coefficient),
            )
    polished_linearized = residual.clone()
    for row_index, row in enumerate(rows64):
        polished_linearized[row_index] += sum(
            (row[name] * polished_displacement[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
    polished_maximum = float(polished_linearized.max())
    polished_primal_pass = bool(
        math.isfinite(polished_maximum) and polished_maximum <= base.LINEAR_CONSTRAINT_TOLERANCE
    )

    delta = polished_dual - primary_dual
    primary_norm_squared = max(0.0, float(primary_dual @ normalized_gram @ primary_dual))
    distance_squared = max(0.0, float(delta @ normalized_gram @ delta))
    relative_distance = math.sqrt(distance_squared) / max(math.sqrt(primary_norm_squared), 1.0e-30)
    primary_objective = audit25._dual_objective(normalized_gram, normalized_violation, primary_dual)
    polished_objective = audit25._dual_objective(
        normalized_gram, normalized_violation, polished_dual
    )
    objective_difference = abs(polished_objective - primary_objective) / max(
        1.0, abs(primary_objective)
    )
    polished.update(
        {
            "original_unit_linearized_violation_after": polished_linearized.tolist(),
            "maximum_original_unit_primal_violation": polished_maximum,
            "original_unit_primal_violation_pass": polished_primal_pass,
            "original_unit_primal_violation_limit": base.LINEAR_CONSTRAINT_TOLERANCE,
            "gram_induced_primal_relative_distance_from_primary": relative_distance,
            "gram_induced_primal_relative_distance_limit": (
                fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
            ),
            "dual_objective": polished_objective,
            "primary_dual_objective": primary_objective,
            "dual_objective_relative_difference": objective_difference,
            "dual_objective_relative_difference_limit": (
                fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
            ),
        }
    )
    polished["pass"] = bool(
        polished["pass"]
        and polished_primal_pass
        and relative_distance <= fp64.INDEPENDENT_PRIMAL_RELATIVE_DISTANCE_LIMIT
        and objective_difference <= fp64.INDEPENDENT_OBJECTIVE_RELATIVE_DIFFERENCE_LIMIT
    )
    report = {
        **primary,
        "pass": bool(primary["pass"] and primary_primal_pass and polished["pass"]),
        "constraint_names": [spec["name"] for spec in specs],
        "linearized_violation_before": violation_before.tolist(),
        "linearized_violation_after": primary_linearized.tolist(),
        "maximum_linearized_violation_after": primary_maximum,
        "original_unit_primal_violation_pass": primary_primal_pass,
        "original_unit_primal_violation_limit": base.LINEAR_CONSTRAINT_TOLERANCE,
        "slsqp_unpolished_diagnostic": slsqp,
        "slsqp_support_polished_reference": polished,
        "lbfgsb_diagnostic": lbfgsb,
        "arithmetic_dtype": "float64",
    }
    return projected, report


def polished_bound_aware_projection(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
    current_edges: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    displacement64 = {name: value.detach().double().clone() for name, value in displacement.items()}
    rows64 = [
        {name: value.detach().double().clone() for name, value in row.items()} for row in rows
    ]
    current_edges64 = current_edges.detach().double()
    fixed = torch.zeros_like(current_edges, dtype=torch.bool)
    fixed_displacement = torch.zeros_like(current_edges64)
    final = {name: value.clone() for name, value in displacement64.items()}
    rounds = []
    converged = False
    for round_index in range(1, fp64.MAXIMUM_ACTIVE_SET_ROUNDS + 1):
        working = {name: value.clone() for name, value in displacement64.items()}
        working["edge_magnitude"][fixed] = fixed_displacement[fixed]
        free_rows = []
        adjusted_specs = []
        for row, spec in zip(rows64, specs, strict=True):
            free_row = {name: value.clone() for name, value in row.items()}
            fixed_contribution = float(
                (free_row["edge_magnitude"][fixed] * fixed_displacement[fixed]).sum()
            )
            free_row["edge_magnitude"][fixed] = 0.0
            adjusted = dict(spec)
            adjusted["current_mse"] = spec["current_mse"] + fixed_contribution
            adjusted["remaining_mse_allowance"] = adjusted["limit_mse"] - adjusted["current_mse"]
            free_rows.append(free_row)
            adjusted_specs.append(adjusted)
        final, projection = polished_free_projection(working, free_rows, adjusted_specs)
        candidate_edges = current_edges64 + final["edge_magnitude"]
        below = candidate_edges < 0.0
        above = candidate_edges > 8.0
        violations = below | above
        newly_fixed = violations & ~fixed
        rounds.append(
            {
                "round": round_index,
                "fixed_edges_before": int(fixed.sum()),
                "fixed_set_sha256_before": audit.semantic_sha256(fixed),
                "newly_fixed_edges": int(newly_fixed.sum()),
                "newly_fixed_set_sha256": audit.semantic_sha256(newly_fixed),
                "free_projection": projection,
            }
        )
        if not projection["pass"]:
            return final, {
                "pass": False,
                "reason": "free-coordinate polished-reference solve failed",
                "rounds": rounds,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
                "arithmetic_dtype": "float64",
            }
        if not bool(violations.any()):
            converged = True
            break
        if not bool(newly_fixed.any()):
            return final, {
                "pass": False,
                "reason": "active set made no progress",
                "rounds": rounds,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
                "arithmetic_dtype": "float64",
            }
        fixed |= newly_fixed
        fixed_displacement[below] = -current_edges64[below]
        fixed_displacement[above] = 8.0 - current_edges64[above]
    final_edges = current_edges64 + final["edge_magnitude"]
    maximum_box_violation = max(
        float((-final_edges).clamp_min(0.0).max()),
        float((final_edges - 8.0).clamp_min(0.0).max()),
    )
    return final, {
        "pass": bool(converged and maximum_box_violation == 0.0),
        "reason": None if converged else "active-set round limit reached",
        "rounds": rounds,
        "fixed_edges": int(fixed.sum()),
        "final_fixed_set_sha256": audit.semantic_sha256(fixed),
        "converged_without_bound_violation": converged,
        "maximum_box_violation": maximum_box_violation,
        "arithmetic_dtype": "float64",
    }


def round_path_signature(projection: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "round": item["round"],
            "fixed_edges_before": item["fixed_edges_before"],
            "fixed_set_sha256_before": item["fixed_set_sha256_before"],
            "newly_fixed_edges": item["newly_fixed_edges"],
            "newly_fixed_set_sha256": item["newly_fixed_set_sha256"],
            "primary_support_mask": item["free_projection"].get("selected_support_mask"),
            "slsqp_polished_support_mask": item["free_projection"]
            .get("slsqp_support_polished_reference", {})
            .get("support_mask"),
        }
        for item in projection["rounds"]
    ]


def run_audit(args: argparse.Namespace, device: torch.device) -> int:
    started = perf_counter()
    identities = validate_locked_identities(args)
    archive_path = args.failed_audit_dir / audit25.ARCHIVE_NAME
    runs = []
    displacements: list[dict[str, Tensor]] = []
    objectives = []
    print(json.dumps({"stage": "three_slsqp_support_polished_reloads"}), flush=True)
    for repeat in range(1, REPEATS + 1):
        archive = torch.load(archive_path, map_location="cpu", weights_only=True)
        if audit25.archive_semantic_hashes(archive) != identities["archive_semantic_sha256"]:
            raise RuntimeError("frozen archive semantic hashes changed between repeats")
        frozen_input = audit25.to_device_tree(archive, device)
        displacement, projection = polished_bound_aware_projection(
            frozen_input["raw_displacement"],
            frozen_input["constraint_rows"],
            frozen_input["constraint_specs"],
            frozen_input["current_parameters"]["edge_magnitude"],
        )
        objective = frozen.complete_projection_primal_objective(
            displacement, frozen_input["raw_displacement"]
        )
        runs.append(
            {
                "repeat": repeat,
                "pass": projection["pass"],
                "projection": projection,
                "round_path_signature": round_path_signature(projection),
                "final_displacement_sha256": audit.semantic_sha256(displacement),
                "learning_rate_scaled_displacement_norm": (
                    frozen.learning_rate_scaled_norm(displacement)
                ),
                "complete_projection_primal_objective": objective,
            }
        )
        displacements.append(displacement)
        objectives.append(objective)
    all_runs_pass = all(run["pass"] for run in runs)
    repeat_agreement = frozen.primary_repeat_agreement(displacements, objectives)
    round_paths_exact = bool(
        runs
        and all(run["round_path_signature"] == runs[0]["round_path_signature"] for run in runs[1:])
    )
    hashes_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    inputs_unchanged = hashes_after == identities["hashes"]
    passed = bool(
        all_runs_pass and repeat_agreement["pass"] and round_paths_exact and inputs_unchanged
    )
    if not all_runs_pass:
        classification = "one_or_more_polished_reference_runs_failed"
    elif not repeat_agreement["pass"] or not round_paths_exact:
        classification = "polished_reference_repeats_disagreed"
    elif not inputs_unchanged:
        classification = "locked_input_changed_during_verification"
    else:
        classification = "slsqp_support_polished_reference_qualified"
    report = {
        "experiment": EXPERIMENT,
        "status": "frozen_numerical_reference_audit",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "locked_hashes_before": identities["hashes"],
        "locked_hashes_after": hashes_after,
        "all_locked_inputs_unchanged": inputs_unchanged,
        "archive_tensor_semantic_sha256": identities["archive_semantic_sha256"],
        "polished_reference_runs": runs,
        "all_polished_reference_runs_pass": all_runs_pass,
        "primary_repeat_agreement": repeat_agreement,
        "round_paths_exact": round_paths_exact,
        "actor_instantiated": False,
        "actor_candidate_materialized": False,
        "gradients_metrics_or_adam_regenerated": False,
        "state_retained": False,
        "development_or_fresh_data_used": False,
        "closed_loop_hover_run": False,
        "promoted": False,
        "polished_reference_implementation_authorized": passed,
        "continuation_authorized": passed,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass qualifies only this frozen-input polished numerical reference and "
            "permits a separately preregistered continuation from accepted update 24."
        ),
    }
    output = args.output_dir / FINAL_REPORT_NAME
    write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": classification,
                "all_runs_pass": all_runs_pass,
                "repeat_agreement_pass": repeat_agreement["pass"],
                "round_paths_exact": round_paths_exact,
                "actor_candidate_materialized": False,
                "state_retained": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.finalize_interrupted:
        return finalize_interrupted(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    write_started(args)
    return run_audit(args, device)


if __name__ == "__main__":
    raise SystemExit(main())
