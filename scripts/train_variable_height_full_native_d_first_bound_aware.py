#!/usr/bin/env python3
"""Run D-first fitting with a bound-aware C/P inequality projection."""

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

import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_full_native_step_response as step_audit  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-bound-aware-fitting-v1"
PROTOCOL_COMMIT = "fe4401f"
MAXIMUM_ACTIVE_SET_ROUNDS = 8
SUBSEQUENT_CLAMP_MAXIMUM_DIFFERENCE = 1.0e-7
_ORIGINAL_PROTOCOL_MANIFEST = base.protocol_manifest
FAILED_PREFLIGHT_EXPERIMENT = base.EXPERIMENT


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
            REPO_ROOT
            / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-d-first-bound-aware-fitting-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    manifest = copy.deepcopy(_ORIGINAL_PROTOCOL_MANIFEST())
    manifest.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "preserves_failed_preflight": FAILED_PREFLIGHT_EXPERIMENT,
        }
    )
    manifest["constraint_projection"].update(
        {
            "edge_bound_handling": (
                "monotonic active set; fix crossing edges at actual boundary "
                "displacement, include J_bound*delta_bound in each residual, remove "
                "fixed coordinates from the free metric and Gram, and re-solve"
            ),
            "maximum_active_set_rounds": MAXIMUM_ACTIVE_SET_ROUNDS,
            "active_edges_released": False,
            "subsequent_clamp_maximum_difference": SUBSEQUENT_CLAMP_MAXIMUM_DIFFERENCE,
            "failure_claim_limit": (
                "round-limit failure rejects this feasibility heuristic only, not the "
                "existence of a feasible direction"
            ),
        }
    )
    return manifest


def bound_aware_project_inequality_displacement(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
    current_edges: Tensor,
    *,
    maximum_rounds: int = MAXIMUM_ACTIVE_SET_ROUNDS,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if maximum_rounds < 1:
        raise ValueError("maximum_rounds must be positive")
    fixed = torch.zeros_like(current_edges, dtype=torch.bool)
    fixed_displacement = torch.zeros_like(current_edges)
    rounds = []
    final = {name: value.detach().clone() for name, value in displacement.items()}
    converged = False
    for round_index in range(1, maximum_rounds + 1):
        working = {name: value.detach().clone() for name, value in displacement.items()}
        working["edge_magnitude"][fixed] = fixed_displacement[fixed]
        free_rows = []
        adjusted_specs = []
        for row, spec in zip(rows, specs, strict=True):
            free_row = {name: value.detach().clone() for name, value in row.items()}
            fixed_contribution = float(
                (free_row["edge_magnitude"][fixed] * fixed_displacement[fixed]).sum()
            )
            free_row["edge_magnitude"][fixed] = 0.0
            adjusted = dict(spec)
            adjusted["current_mse"] = spec["current_mse"] + fixed_contribution
            adjusted["remaining_mse_allowance"] = (
                adjusted["limit_mse"] - adjusted["current_mse"]
            )
            free_rows.append(free_row)
            adjusted_specs.append(adjusted)
        final, projection = base.project_inequality_displacement(
            working, free_rows, adjusted_specs
        )
        candidate_edges = current_edges + final["edge_magnitude"]
        below = candidate_edges < 0.0
        above = candidate_edges > 8.0
        violations = below | above
        newly_fixed = violations & ~fixed
        rounds.append(
            {
                "round": round_index,
                "fixed_edges_before": int(fixed.sum()),
                "newly_fixed_edges": int(newly_fixed.sum()),
                "free_projection_pass": projection["pass"],
                "free_projection_maximum_linearized_violation": projection.get(
                    "maximum_linearized_violation_after"
                ),
            }
        )
        if not projection["pass"]:
            return final, {
                "pass": False,
                "reason": "free-coordinate inequality solve failed",
                "rounds": rounds,
                "final_free_projection": projection,
                "fixed_edges": int(fixed.sum()),
                "converged_without_bound_violation": False,
            }
        if not bool(violations.any()):
            converged = True
            break
        fixed |= newly_fixed
        fixed_displacement[below] = -current_edges[below]
        fixed_displacement[above] = 8.0 - current_edges[above]
    final_edges = current_edges + final["edge_magnitude"]
    maximum_box_violation = max(
        float((-final_edges).clamp_min(0.0).max()),
        float((final_edges - 8.0).clamp_min(0.0).max()),
    )
    return final, {
        "pass": converged and maximum_box_violation == 0.0,
        "reason": None if converged else "active-set round limit reached",
        "rounds": rounds,
        "final_free_projection": projection,
        "fixed_edges": int(fixed.sum()),
        "converged_without_bound_violation": converged,
        "maximum_box_violation": maximum_box_violation,
        "fixed_displacement_minimum": (
            float(fixed_displacement[fixed].min()) if bool(fixed.any()) else None
        ),
        "fixed_displacement_maximum": (
            float(fixed_displacement[fixed].max()) if bool(fixed.any()) else None
        ),
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
    base_parameters = joint._copy_parameters(student)
    specs = base.constraint_specs(source_metrics, current_metrics)
    print(json.dumps({"stage": "bound_aware_constraint_jacobian", "rows": len(specs)}), flush=True)
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
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    optimizer.step()
    student.project_parameters()
    raw_parameters = joint._copy_parameters(student)
    raw_displacement = {
        name: raw_parameters[name] - base_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    projected, projection = bound_aware_project_inequality_displacement(
        raw_displacement, rows, specs, base_parameters["edge_magnitude"]
    )
    step_audit.set_displacement(student, base_parameters, projected, scale=1.0)
    bounded_parameters = joint._copy_parameters(student)
    bounded = {
        name: bounded_parameters[name] - base_parameters[name]
        for name in joint.PARAMETER_FAMILIES
    }
    clamp_difference = max(
        float((bounded[name] - projected[name]).abs().max())
        for name in joint.PARAMETER_FAMILIES
    )
    bounded_linearized = [
        spec["current_mse"]
        - spec["limit_mse"]
        + float(
            sum((row[name] * bounded[name]).sum() for name in joint.PARAMETER_FAMILIES)
        )
        for spec, row in zip(specs, rows, strict=True)
    ]
    maximum_bounded_violation = max(bounded_linearized)
    raw_derivative = float(
        sum(
            (raw_gradients[name] * raw_displacement[name]).sum()
            for name in joint.PARAMETER_FAMILIES
        )
    )
    bounded_derivative = float(
        sum(
            (raw_gradients[name] * bounded[name]).sum() for name in joint.PARAMETER_FAMILIES
        )
    )
    projection.update(
        {
            "pass_after_parameter_bounds": (
                projection["pass"]
                and maximum_bounded_violation <= base.LINEAR_CONSTRAINT_TOLERANCE
                and clamp_difference <= SUBSEQUENT_CLAMP_MAXIMUM_DIFFERENCE
                and bounded_derivative < 0.0
            ),
            "bounded_linearized_violation_after": bounded_linearized,
            "maximum_bounded_linearized_violation_after": maximum_bounded_violation,
            "subsequent_clamp_maximum_difference": clamp_difference,
            "subsequent_clamp_maximum_difference_limit": (
                SUBSEQUENT_CLAMP_MAXIMUM_DIFFERENCE
            ),
            "raw_damping_directional_derivative": raw_derivative,
            "bounded_projected_damping_directional_derivative": bounded_derivative,
            "damping_descent_fraction_retained": (
                bounded_derivative / raw_derivative if raw_derivative < 0.0 else None
            ),
            "raw_displacement_family_rms": {
                name: float(value.square().mean().sqrt())
                for name, value in raw_displacement.items()
            },
            "bounded_projected_displacement_family_rms": {
                name: float(value.square().mean().sqrt()) for name, value in bounded.items()
            },
            "constraint_specs": specs,
            "gradient_norm_before_clipping": float(gradient_norm),
        }
    )
    joint._load_parameters(student, base_parameters)
    return bounded, raw_gradients, projection


def main() -> int:
    base.EXPERIMENT = EXPERIMENT
    base.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    base.parse_args = parse_args
    base.protocol_manifest = protocol_manifest
    base.make_projected_proposal = make_projected_proposal
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
