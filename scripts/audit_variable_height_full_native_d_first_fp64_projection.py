#!/usr/bin/env python3
"""Audit the rejected update-21 projection with genuinely FP64 linear algebra."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_bound_aware as bounded  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fp64-projection-audit-v1"
PROTOCOL_COMMIT = "44448fe"
SOURCE_EXPERIMENT = corrected.EXPERIMENT
EXPECTED_SOURCE_REPORT_SHA256 = "9b236fc6a6744b1b06984958ebc2c3ff3681cc45fc1fa6da73d1f96ae56b458b"
EXPECTED_SOURCE_RESUME_SHA256 = "61d32fab3995599286f6eee3b30f24b3bea48042d2528392fdd1ca68e9bf60c4"
EXPECTED_ACCEPTED_UPDATES = 20
RECONSTRUCTED_UPDATE = 21
RANK_RELATIVE_EIGENVALUE_CUTOFF = 1.0e-12
NORMALIZED_KKT_RESIDUAL_LIMIT = 1.0e-8
INDEPENDENT_SLSQP_FTOL = 1.0e-12
MAXIMUM_SOLVER_ITERATIONS = 10_000
MAXIMUM_ACTIVE_SET_ROUNDS = bounded.MAXIMUM_ACTIVE_SET_ROUNDS


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
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fp64-projection-audit-001"
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
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered projection audit has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": SOURCE_EXPERIMENT,
        "source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
        "source_resume_sha256": EXPECTED_SOURCE_RESUME_SHA256,
        "source_accepted_updates": EXPECTED_ACCEPTED_UPDATES,
        "reconstructed_update": RECONSTRUCTED_UPDATE,
        "actor_contract_unchanged": True,
        "frozen_tensors": [
            "raw Adam displacement",
            "constraint Jacobian rows",
            "constraint specifications",
            "failed round-two fixed set",
        ],
        "gradients_generated_once": True,
        "variants_consume_clones": True,
        "production_projection_replayed_unchanged": True,
        "fp64_projection": {
            "float64_before_products": True,
            "float64_operations": [
                "Gram products and reductions",
                "raw and fixed-coordinate residual dot products",
                "projected displacement accumulation",
                "final residuals",
            ],
            "metric_and_constraints_unchanged": True,
            "dual_solver": "L-BFGS-B from zero with analytic gradient",
            "solver_success_required": True,
            "maximum_solver_iterations": MAXIMUM_SOLVER_ITERATIONS,
            "active_set": "unchanged monotonic actual-boundary heuristic",
            "maximum_active_set_rounds": MAXIMUM_ACTIVE_SET_ROUNDS,
            "rank_relative_eigenvalue_cutoff": RANK_RELATIVE_EIGENVALUE_CUTOFF,
            "original_unit_primal_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
            "normalized_projected_gradient_residual_maximum": (NORMALIZED_KKT_RESIDUAL_LIMIT),
        },
        "independent_solver": {
            "name": "SLSQP",
            "initial_dual": "zero",
            "analytic_gradient": True,
            "nonnegative_bounds": True,
            "ftol": INDEPENDENT_SLSQP_FTOL,
            "maximum_iterations": MAXIMUM_SOLVER_ITERATIONS,
            "diagnostic_only": True,
        },
        "post_projection_gates": {
            "canonical_parameter_idempotence_maximum": (canonical.PARAMETER_IDEMPOTENCE_TOLERANCE),
            "native_parameter_bounds": True,
            "linear_violation_maximum": base.LINEAR_CONSTRAINT_TOLERANCE,
            "negative_endpoint_damping_derivative": True,
            "finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_maximum": (
                joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "ordinary_scales_descending": list(base.BACKTRACK_SCALES),
            "nonlinear_repair": False,
        },
        "source_parameters_optimizer_resume_and_report_restored_exactly": True,
        "candidate_retained": False,
        "development_or_fresh_data": False,
        "closed_loop_hover": False,
        "promotion": False,
        "pass_authorizes": "separately registered genuine-FP64 projection implementation",
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


def _projected_gradient_report(
    normalized_gram: np.ndarray,
    normalized_violation: np.ndarray,
    dual: np.ndarray,
) -> dict[str, Any]:
    gradient = normalized_gram @ dual - normalized_violation
    positive = dual > 0.0
    projected = np.where(positive, np.abs(gradient), np.maximum(-gradient, 0.0))
    normalization = max(1.0, float(np.max(np.abs(normalized_violation), initial=0.0)))
    residual = float(np.max(projected, initial=0.0)) / normalization
    complementarity = float(np.max(np.abs(dual * gradient), initial=0.0)) / normalization
    return {
        "pass": bool(np.isfinite(residual) and residual <= NORMALIZED_KKT_RESIDUAL_LIMIT),
        "normalized_projected_gradient_residual": residual,
        "normalized_projected_gradient_residual_limit": NORMALIZED_KKT_RESIDUAL_LIMIT,
        "normalized_complementarity_residual": complementarity,
        "normalization": normalization,
        "positive_dual_coordinates": int(positive.sum()),
        "gradient_minimum": float(gradient.min(initial=0.0)),
        "gradient_maximum": float(gradient.max(initial=0.0)),
    }


def _rank_report(normalized_gram: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * (normalized_gram + normalized_gram.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    largest = float(np.max(np.abs(eigenvalues), initial=0.0))
    cutoff = RANK_RELATIVE_EIGENVALUE_CUTOFF * largest
    rank = int(np.sum(eigenvalues > cutoff))
    positive = eigenvalues[eigenvalues > cutoff]
    return {
        "eigenvalues": eigenvalues.tolist(),
        "relative_eigenvalue_cutoff": RANK_RELATIVE_EIGENVALUE_CUTOFF,
        "absolute_eigenvalue_cutoff": cutoff,
        "rank": rank,
        "dimension": int(normalized_gram.shape[0]),
        "nullity": int(normalized_gram.shape[0] - rank),
        "condition_number_on_retained_subspace": (
            float(positive.max() / positive.min()) if positive.size else None
        ),
    }


def solve_dual_fp64(
    gram: Tensor,
    violation: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    row_norm = gram.diagonal().clamp_min(0.0).sqrt()
    active = row_norm > 1.0e-15
    infeasible_zero = (~active) & (violation > base.LINEAR_CONSTRAINT_TOLERANCE)
    if bool(infeasible_zero.any()):
        return torch.zeros_like(violation), {
            "pass": False,
            "solver_success": False,
            "reason": "a violated constraint has a numerically zero Jacobian row",
            "active_rows": int(active.sum()),
        }
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    normalized_gram = (
        (gram[active][:, active] / (row_norm[active, None] * row_norm[None, active])).cpu().numpy()
    )
    normalized_violation = (violation[active] / row_norm[active]).cpu().numpy()

    def objective(value: np.ndarray) -> float:
        return float(0.5 * value @ normalized_gram @ value - normalized_violation @ value)

    def gradient(value: np.ndarray) -> np.ndarray:
        return normalized_gram @ value - normalized_violation

    initial = np.zeros(len(active_indices), dtype=np.float64)
    bounds_spec = [(0.0, None)] * len(active_indices)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds_spec,
        options={
            "ftol": 1.0e-15,
            "gtol": base.DUAL_GRADIENT_TOLERANCE,
            "maxiter": MAXIMUM_SOLVER_ITERATIONS,
        },
    )
    independent = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds_spec,
        options={
            "ftol": INDEPENDENT_SLSQP_FTOL,
            "maxiter": MAXIMUM_SOLVER_ITERATIONS,
        },
    )
    coefficients = torch.zeros_like(violation)
    coefficients[active_indices] = torch.from_numpy(result.x).to(gram.device) / row_norm[active]
    return coefficients, {
        "pass": bool(result.success),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "dual_objective": float(result.fun),
        "normalized_kkt": _projected_gradient_report(
            normalized_gram, normalized_violation, result.x
        ),
        "normalized_gram": _rank_report(normalized_gram),
        "independent_slsqp": {
            "solver_success": bool(independent.success),
            "solver_message": str(independent.message),
            "solver_iterations": int(independent.nit),
            "dual_objective": float(independent.fun),
            "normalized_kkt": _projected_gradient_report(
                normalized_gram, normalized_violation, independent.x
            ),
            "maximum_dual_difference_from_primary": float(
                np.max(np.abs(independent.x - result.x), initial=0.0)
            ),
            "used_for_candidate": False,
        },
    }


def fp64_project_inequality_displacement(
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
    coefficients, solver = solve_dual_fp64(gram, violation_before)
    for coefficient, row in zip(coefficients, rows64, strict=True):
        for name in joint.PARAMETER_FAMILIES:
            projected[name].add_(
                row[name],
                alpha=-(base.PARAMETER_LEARNING_RATES[name] ** 2) * float(coefficient),
            )
    linearized_after = residual.clone()
    for row_index, row in enumerate(rows64):
        linearized_after[row_index] += sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
    maximum = float(linearized_after.max())
    primal_pass = math.isfinite(maximum) and maximum <= base.LINEAR_CONSTRAINT_TOLERANCE
    solver.update(
        {
            "pass": bool(
                solver["solver_success"] and solver["normalized_kkt"]["pass"] and primal_pass
            ),
            "constraint_names": [spec["name"] for spec in specs],
            "linearized_violation_before": violation_before.tolist(),
            "linearized_violation_after": linearized_after.tolist(),
            "maximum_linearized_violation_after": maximum,
            "original_unit_primal_violation_pass": primal_pass,
            "original_unit_primal_violation_limit": base.LINEAR_CONSTRAINT_TOLERANCE,
            "dual_coefficients": coefficients.tolist(),
            "arithmetic_dtype": "float64",
        }
    )
    return projected, solver


def fp64_bound_aware_projection(
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
    for round_index in range(1, MAXIMUM_ACTIVE_SET_ROUNDS + 1):
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
        final, projection = fp64_project_inequality_displacement(working, free_rows, adjusted_specs)
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
                "reason": "free-coordinate inequality solve failed",
                "rounds": rounds,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
                "arithmetic_dtype": "float64",
            }
        if not bool(violations.any()):
            converged = True
            break
        fixed |= newly_fixed
        fixed_displacement[below] = -current_edges64[below]
        fixed_displacement[above] = 8.0 - current_edges64[above]
    final_edges = current_edges64 + final["edge_magnitude"]
    maximum_box_violation = max(
        float((-final_edges).clamp_min(0.0).max()),
        float((final_edges - 8.0).clamp_min(0.0).max()),
    )
    return final, {
        "pass": converged and maximum_box_violation == 0.0,
        "reason": None if converged else "active-set round limit reached",
        "rounds": rounds,
        "fixed_edges": int(fixed.sum()),
        "final_fixed_set_sha256": audit.semantic_sha256(fixed),
        "converged_without_bound_violation": converged,
        "maximum_box_violation": maximum_box_violation,
        "arithmetic_dtype": "float64",
    }


def frozen_round_two_fixed_set(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
    current_edges: Tensor,
) -> dict[str, Any]:
    first, projection = base.project_inequality_displacement(displacement, rows, specs)
    candidate_edges = current_edges + first["edge_magnitude"]
    fixed = (candidate_edges < 0.0) | (candidate_edges > 8.0)
    return {
        "first_round_projection": projection,
        "fixed_edges": int(fixed.sum()),
        "fixed_set_sha256": audit.semantic_sha256(fixed),
    }


def production_projection(
    student: ConnectomeController,
    current_parameters: dict[str, Tensor],
    raw_displacement: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
    *,
    gradient_norm: float,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    proposed, projection = bounded.bound_aware_project_inequality_displacement(
        raw_displacement, rows, specs, current_parameters["edge_magnitude"]
    )
    authoritative, effective, canonicalization = canonical.materialize_authoritative_candidate(
        student, current_parameters, proposed
    )
    linearized = audit.linearized_constraint_violations(specs, rows, effective)
    maximum, _ = audit.maximum_linearized_violation(linearized)
    raw64 = {name: value.double() for name, value in raw_displacement.items()}
    raw_derivative = canonical._dot_float64(raw_gradients, raw64)
    effective_derivative = canonical._dot_float64(raw_gradients, effective)
    projection.update(
        {
            "pass_after_parameter_bounds": bool(
                projection["pass"]
                and canonicalization["pass"]
                and maximum <= base.LINEAR_CONSTRAINT_TOLERANCE
                and effective_derivative < 0.0
            ),
            "authoritative_parameter_canonicalization": canonicalization,
            "bounded_linearized_violation_after": linearized,
            "maximum_bounded_linearized_violation_after": maximum,
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
                name: float(value.square().mean().sqrt()) for name, value in effective.items()
            },
            "constraint_specs": specs,
            "gradient_norm_before_clipping": gradient_norm,
            "effective_displacement_computed_in_float64": True,
        }
    )
    joint._load_parameters(student, current_parameters)
    return authoritative, projection


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    report_path = args.source_fit_dir / "report.json"
    resume_path = args.source_fit_dir / "resume.pt"
    report_sha_before = responsibility.file_sha256(report_path)
    resume_sha_before = responsibility.file_sha256(resume_path)
    if report_sha_before != EXPECTED_SOURCE_REPORT_SHA256:
        raise SystemExit("corrected fitting report hash does not match the protocol")
    if resume_sha_before != EXPECTED_SOURCE_RESUME_SHA256:
        raise SystemExit("corrected fitting resume hash does not match the protocol")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    source_resume = torch.load(resume_path, map_location=device, weights_only=True)
    identity = (
        source_report.get("experiment"),
        source_report.get("accepted_updates"),
        source_report.get("stop_reason"),
        source_resume.get("experiment"),
        source_resume.get("protocol_commit"),
        int(source_resume.get("accepted_updates", -1)),
        source_resume.get("run_state"),
    )
    expected_identity = (
        SOURCE_EXPERIMENT,
        EXPECTED_ACCEPTED_UPDATES,
        "constraint projection failed",
        SOURCE_EXPERIMENT,
        corrected.PROTOCOL_COMMIT,
        EXPECTED_ACCEPTED_UPDATES,
        "stopped",
    )
    if identity != expected_identity:
        raise SystemExit("corrected fitting source identity does not match the protocol")

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
    optimizer = make_optimizer(student)
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

    print(json.dumps({"stage": "reconstructing_frozen_update_21_tensors"}), flush=True)
    specs = base.constraint_specs(source_report["source_training"], source_report["final_training"])
    rows = base.constraint_gradient_rows(
        student,
        train_factorial,
        train_attitude,
        scales,
        specs,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    endpoint.accumulated_endpoint_damping_gradient(
        student,
        train_factorial,
        scale=endpoint_scale,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    )
    optimizer.step()
    student.project_parameters()
    raw_parameters = joint._copy_parameters(student)
    raw_displacement = {
        name: raw_parameters[name] - current_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    optimizer_transaction = corrected.optimizer_step_transaction(optimizer_before, optimizer_after)
    joint._load_parameters(student, current_parameters)
    frozen_hashes = {
        "raw_displacement_sha256": audit.semantic_sha256(raw_displacement),
        "constraint_rows_sha256": audit.semantic_sha256(rows),
        "constraint_specs_sha256": audit.semantic_sha256(specs),
    }
    failed_round_two_set = frozen_round_two_fixed_set(
        raw_displacement, rows, specs, current_parameters["edge_magnitude"]
    )
    frozen_hashes["failed_round_two_fixed_set_sha256"] = failed_round_two_set["fixed_set_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_manifest_path = args.output_dir / "frozen-tensor-hashes.json"
    frozen_manifest = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        **frozen_hashes,
    }
    frozen_manifest_path.write_text(
        f"{json.dumps(frozen_manifest, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    frozen_manifest_sha256 = responsibility.file_sha256(frozen_manifest_path)

    print(json.dumps({"stage": "production_projection_replay"}), flush=True)
    _, production = production_projection(
        student,
        current_parameters,
        {name: value.clone() for name, value in raw_displacement.items()},
        {name: value.clone() for name, value in raw_gradients.items()},
        [{name: value.clone() for name, value in row.items()} for row in rows],
        copy.deepcopy(specs),
        gradient_norm=gradient_norm,
    )
    registered_projection = source_report["history"][-1]["projection"]
    production_reproduction = audit.numeric_tree_comparison(
        audit.proposal_signature(production),
        audit.proposal_signature(registered_projection),
    )
    production_rounds_reproduction = audit.numeric_tree_comparison(
        production["rounds"], registered_projection["rounds"]
    )

    print(json.dumps({"stage": "genuine_fp64_projection_replay"}), flush=True)
    fp64_displacement, fp64_projection = fp64_bound_aware_projection(
        {name: value.clone() for name, value in raw_displacement.items()},
        [{name: value.clone() for name, value in row.items()} for row in rows],
        copy.deepcopy(specs),
        current_parameters["edge_magnitude"].clone(),
    )
    post_projection: dict[str, Any] = {
        "attempted": False,
        "pass": False,
        "skip_reason": "genuine-FP64 active-set projection did not pass",
        "ordinary_trials": [],
    }
    if fp64_projection["pass"]:
        post_projection["attempted"] = True
        corrected_parameters, effective, idempotence = (
            canonical.materialize_authoritative_candidate(
                student, current_parameters, fp64_displacement
            )
        )
        bounds = audit.parameter_bounds_report(corrected_parameters)
        linearized = audit.linearized_constraint_violations(specs, rows, effective)
        maximum_linearized, linearized_finite = audit.maximum_linearized_violation(linearized)
        derivative = canonical._dot_float64(raw_gradients, effective)
        canonical.install_trial(
            student,
            current_parameters,
            {name: value.to(current_parameters[name].dtype) for name, value in effective.items()},
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
        actual_fd = {
            name: getattr(student, name).detach().double() - current_parameters[name].double()
            for name in joint.PARAMETER_FAMILIES
        }
        fd_derivative = canonical._dot_float64(raw_gradients, actual_fd) / (
            joint.FINITE_DIFFERENCE_SCALE
        )
        fd_actual = (
            finite_difference_metrics["endpoint_damping_nrmse"] ** 2
            - current_metrics["endpoint_damping_nrmse"] ** 2
        ) / joint.FINITE_DIFFERENCE_SCALE
        fd_relative_error = abs(fd_actual - fd_derivative) / max(
            abs(fd_actual), abs(fd_derivative), 1.0e-12
        )
        finite_difference = {
            "pass": bool(
                math.isfinite(fd_derivative)
                and math.isfinite(fd_actual)
                and abs(fd_actual) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
                and fd_derivative < 0.0
                and fd_actual < 0.0
                and fd_relative_error <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "scale": joint.FINITE_DIFFERENCE_SCALE,
            "autograd_directional_derivative": fd_derivative,
            "complete_replay_finite_difference": fd_actual,
            "relative_error": fd_relative_error,
            "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
        }
        selected_scale = None
        selected_metrics = None
        trials = []
        for scale in base.BACKTRACK_SCALES:
            canonical.install_trial(
                student,
                current_parameters,
                {
                    name: value.to(current_parameters[name].dtype)
                    for name, value in effective.items()
                },
                scale=scale,
            )
            actual = {
                name: getattr(student, name).detach().double() - current_parameters[name].double()
                for name in joint.PARAMETER_FAMILIES
            }
            metrics, _ = endpoint.evaluate(
                student,
                train_factorial,
                train_attitude,
                scales,
                endpoint_scale=endpoint_scale,
                device=device,
            )
            trial_derivative = canonical._dot_float64(raw_gradients, actual)
            decision = base.candidate_decision(
                source_metrics,
                current_metrics,
                metrics,
                damping_directional_derivative=trial_derivative,
            )
            trials.append({"scale": scale, "metrics": metrics, "decision": decision})
            if decision["pass"]:
                selected_scale = scale
                selected_metrics = metrics
                break
        post_projection = {
            "attempted": True,
            "pass": bool(
                idempotence["pass"]
                and bounds["pass"]
                and linearized_finite
                and maximum_linearized <= base.LINEAR_CONSTRAINT_TOLERANCE
                and derivative < 0.0
                and finite_difference["pass"]
                and selected_scale is not None
            ),
            "skip_reason": None,
            "canonical_parameter_idempotence": idempotence,
            "parameter_bounds": bounds,
            "linearized_constraint_violations": linearized,
            "maximum_linearized_constraint_violation": maximum_linearized,
            "damping_directional_derivative": derivative,
            "finite_difference": finite_difference,
            "ordinary_trials": trials,
            "selected_ordinary_scale": selected_scale,
            "selected_ordinary_metrics": selected_metrics,
            "nonlinear_repair_attempted": False,
        }

    joint._load_parameters(student, current_parameters)
    optimizer.load_state_dict(optimizer_before)
    optimizer.zero_grad(set_to_none=True)
    parameters_restored = all(
        torch.equal(getattr(student, name).detach(), current_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    optimizer_restored = audit.trees_equal(optimizer.state_dict(), optimizer_before)
    report_sha_after = responsibility.file_sha256(report_path)
    resume_sha_after = responsibility.file_sha256(resume_path)
    sources_unchanged = bool(
        report_sha_after == report_sha_before and resume_sha_after == resume_sha_before
    )
    controls_pass = bool(
        cache_integrity["all_persisted_file_and_tensor_hashes_match"]
        and source_reproduction["pass"]
        and current_reproduction["pass"]
        and optimizer_transaction["pass"]
        and production_reproduction["pass"]
        and production_rounds_reproduction["pass"]
        and parameters_restored
        and optimizer_restored
        and sources_unchanged
    )
    passed = bool(controls_pass and fp64_projection["pass"] and post_projection["pass"])
    if not controls_pass:
        classification = "audit_control_failure"
    elif not fp64_projection["pass"]:
        classification = "genuine_fp64_projection_failed"
    elif not post_projection["pass"]:
        classification = "genuine_fp64_projection_failed_original_post_projection_gates"
    else:
        classification = "genuine_fp64_projection_and_ordinary_replay_passed"
    report = {
        "experiment": EXPERIMENT,
        "status": "restored_projection_audit_no_retained_candidate",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "corrected_fit_directory": responsibility.stable_path(args.source_fit_dir),
            "corrected_fit_report_sha256_before": report_sha_before,
            "corrected_fit_report_sha256_after": report_sha_after,
            "corrected_fit_resume_sha256_before": resume_sha_before,
            "corrected_fit_resume_sha256_after": resume_sha_after,
            "source_files_unchanged": sources_unchanged,
        },
        "cache_integrity": cache_integrity,
        "source_training": source_metrics,
        "update_20_training": current_metrics,
        "source_training_reproduction": source_reproduction,
        "update_20_training_reproduction": current_reproduction,
        "frozen_tensor_hashes": frozen_hashes,
        "frozen_tensor_hash_manifest": {
            "path": responsibility.stable_path(frozen_manifest_path),
            "sha256": frozen_manifest_sha256,
            "persisted_before_numerical_variant_comparison": True,
        },
        "failed_round_two_fixed_set": failed_round_two_set,
        "optimizer_one_step_transaction": optimizer_transaction,
        "production_projection": production,
        "production_projection_reproduction": production_reproduction,
        "production_rounds_reproduction": production_rounds_reproduction,
        "genuine_fp64_projection": fp64_projection,
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
            "This audit diagnoses one frozen projection and cannot retroactively accept "
            "update 21, establish useful damping, authorize hover, or prove infeasibility."
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
                "production_reproduced": bool(
                    production_reproduction["pass"] and production_rounds_reproduction["pass"]
                ),
                "genuine_fp64_projection_pass": fp64_projection["pass"],
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
