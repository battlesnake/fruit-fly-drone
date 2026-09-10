#!/usr/bin/env python3
"""Run bounded, constraint-aware full-native endpoint-damping fitting."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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

import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_full_native_step_response as step_audit  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-d-first-fitting-v1"
PROTOCOL_COMMIT = "420df40"
BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
MAXIMUM_ACCEPTED_UPDATES = 200
DEVELOPMENT_INTERVAL = 10
MANDATORY_GATE_UPDATE = 50
MANDATORY_D_IMPROVEMENT_FRACTION = 0.25
TRAIN_TERMINAL_D_NRMSE = 0.20
DEVELOPMENT_TERMINAL_D_NRMSE = 0.30
TERMINAL_SIGN_FRACTION = 0.90
TERMINAL_GAIN_RANGE = (0.5, 1.5)
RPY_CONSTRAINT_ACTIVATION_NRMSE = 0.04
LINEAR_CONSTRAINT_TOLERANCE = 1.0e-6
DUAL_GRADIENT_TOLERANCE = 1.0e-11
QUALIFICATION_FACTORIAL_SEEDS = tuple(range(360_971, 360_979))
QUALIFICATION_ATTITUDE_SEEDS = tuple(range(370_971, 370_979))
QUALIFICATION_SCENES_PER_BANK = 8
EXPECTED_SOURCE_CACHE_MANIFEST_SHA256 = (
    "a3aa57d103158adc571734a49182df2fbf04852089532bf9f88631c9bac92060"
)
PARAMETER_LEARNING_RATES = {
    "edge_magnitude": joint.EDGE_BIAS_LEARNING_RATE,
    "bias": joint.EDGE_BIAS_LEARNING_RATE,
    "raw_time_constant": joint.TIME_CONSTANT_LEARNING_RATE,
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
            REPO_ROOT
            / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/full-native-d-first-fitting-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the authorized immutable training/development cache is required")
    if args.smoke_test:
        raise SystemExit("this preregistered fitting diagnostic has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": endpoint.EXPERIMENT,
        "source_cache_manifest_sha256": EXPECTED_SOURCE_CACHE_MANIFEST_SHA256,
        "training_scenes": joint.SCENES_PER_BANK,
        "development_scenes": joint.SCENES_PER_BANK,
        "training_seed": endpoint.TRAIN_SEED,
        "development_seed": endpoint.DEVELOPMENT_SEED,
        "development_is_already_exposed": True,
        "objective": "normalized endpoint damping MSE only",
        "full_bank_gradient": True,
        "native_state_initialization": "zero",
        "prefix_and_response_are_differentiated": True,
        "actor_inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
        "privileged_actor_inputs": [],
        "opened_parameter_families": list(joint.PARAMETER_FAMILIES),
        "optimizer": {
            "name": "Adam",
            "edge_and_bias_learning_rate": joint.EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": joint.TIME_CONSTANT_LEARNING_RATE,
            "weight_decay": 0.0,
            "global_gradient_norm_cap": joint.GRADIENT_NORM_CAP,
            "state_retained_only_on_accepted_update": True,
        },
        "constraint_projection": {
            "metric": "Euclidean in per-family learning-rate-scaled coordinates",
            "inequality_quantities": (
                "squared normalized C/P error, aggregate and at steps 15/20/25"
            ),
            "limits": "square of each original-source NRMSE plus 0.02",
            "solver": "nonnegative dual quadratic program via L-BFGS-B",
            "linear_constraint_tolerance": LINEAR_CONSTRAINT_TOLERANCE,
            "individual_output_equalities": False,
            "rpy_row_activation_nrmse": RPY_CONSTRAINT_ACTIVATION_NRMSE,
            "rpy_forward_checked_always": True,
            "post_parameter_bound_displacement_checked": True,
        },
        "preflight": {
            "required": True,
            "finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_limit": (
                joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "exact_cache_and_source_replay": True,
            "teacher_and_foreleg_stick_controls": True,
            "negative_endpoint_damping_derivative": True,
            "linearized_constraints_pass": True,
            "complete_replay_pass": True,
        },
        "backtrack_scales_descending": list(BACKTRACK_SCALES),
        "minimum_actual_endpoint_damping_nrmse_improvement_per_update": (
            endpoint.MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT
        ),
        "cumulative_preservation": endpoint.protocol_manifest()["preservation"],
        "stop_immediately_after_one_rejected_deterministic_proposal": True,
        "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
        "development_interval_accepted_updates": DEVELOPMENT_INTERVAL,
        "mandatory_update_50_gate": {
            "endpoint_damping_nrmse_improvement_fraction_both_banks": (
                MANDATORY_D_IMPROVEMENT_FRACTION
            ),
            "all_preservation_gates": True,
        },
        "terminal": {
            "first_scheduled_qualifying_checkpoint_only": True,
            "training_endpoint_damping_nrmse_maximum": TRAIN_TERMINAL_D_NRMSE,
            "development_endpoint_damping_nrmse_maximum": (
                DEVELOPMENT_TERMINAL_D_NRMSE
            ),
            "correct_sign_fraction_minimum_both_banks": TERMINAL_SIGN_FRACTION,
            "teacher_aligned_gain_range_both_banks": list(TERMINAL_GAIN_RANGE),
            "all_preservation_gates": True,
        },
        "qualification": {
            "factorial_seeds": list(QUALIFICATION_FACTORIAL_SEEDS),
            "attitude_seeds": list(QUALIFICATION_ATTITUDE_SEEDS),
            "scenes_per_streamed_bank": QUALIFICATION_SCENES_PER_BANK,
            "total_scenes": (
                len(QUALIFICATION_FACTORIAL_SEEDS) * QUALIFICATION_SCENES_PER_BANK
            ),
            "generated_only_after_terminal_development_pass": True,
            "candidate_fallback": False,
            "endpoint_damping_nrmse_maximum": DEVELOPMENT_TERMINAL_D_NRMSE,
            "correct_sign_fraction_minimum": TERMINAL_SIGN_FRACTION,
            "teacher_aligned_gain_range": list(TERMINAL_GAIN_RANGE),
            "independent_source_relative_preservation": True,
        },
        "checkpoint_is_nonpromotional_ignored_run_artifact": True,
        "fresh_pass_authorizes": (
            "nominal-mass native closed-loop hover comparison with frozen-vision controls"
        ),
    }


def constraint_specs(
    source_metrics: dict[str, Any], current_metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    specs = []
    for component in ("common", "height"):
        source_value = source_metrics["component_nrmse"][component]
        current_value = current_metrics["component_nrmse"][component]
        specs.append(
            {
                "name": f"{component}.aggregate",
                "kind": "factorial",
                "component": component,
                "horizon_index": None,
                "current_mse": current_value**2,
                "limit_mse": (source_value + joint.COMPONENT_BASELINE_TOLERANCE) ** 2,
            }
        )
        for horizon_index, horizon in enumerate(joint.SUPERVISION_STEPS):
            source_value = source_metrics["by_supervision_step_nrmse"][str(horizon)][component]
            current_value = current_metrics["by_supervision_step_nrmse"][str(horizon)][
                component
            ]
            specs.append(
                {
                    "name": f"{component}.step_{horizon}",
                    "kind": "factorial",
                    "component": component,
                    "horizon_index": horizon_index,
                    "current_mse": current_value**2,
                    "limit_mse": (source_value + joint.COMPONENT_BASELINE_TOLERANCE) ** 2,
                }
            )
    for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
        current_value = current_metrics["rpy_source_nrmse"][axis]
        if current_value >= RPY_CONSTRAINT_ACTIVATION_NRMSE:
            specs.append(
                {
                    "name": f"rpy.{axis}",
                    "kind": "attitude",
                    "axis_index": axis_index,
                    "current_mse": current_value**2,
                    "limit_mse": joint.RPY_NRMSE_LIMIT**2,
                }
            )
    for spec in specs:
        spec["remaining_mse_allowance"] = spec["limit_mse"] - spec["current_mse"]
    return specs


def _empty_gradient_rows(
    controller: ConnectomeController, count: int
) -> list[dict[str, Tensor]]:
    return [
        {
            name: torch.zeros_like(getattr(controller, name))
            for name in joint.PARAMETER_FAMILIES
        }
        for _ in range(count)
    ]


def _accumulate_gradients(
    rows: list[dict[str, Tensor]],
    indices: list[int],
    losses: list[Tensor],
    parameters: tuple[Tensor, ...],
) -> None:
    for offset, (row_index, loss) in enumerate(zip(indices, losses, strict=True)):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=offset + 1 < len(losses),
        )
        for name, gradient in zip(joint.PARAMETER_FAMILIES, gradients, strict=True):
            rows[row_index][name].add_(gradient.detach())


def constraint_gradient_rows(
    controller: ConnectomeController,
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    specs: list[dict[str, Any]],
    *,
    device: torch.device,
) -> list[dict[str, Tensor]]:
    rows = _empty_gradient_rows(controller, len(specs))
    parameters = tuple(getattr(controller, name) for name in joint.PARAMETER_FAMILIES)
    factorial_indices = [index for index, spec in enumerate(specs) if spec["kind"] == "factorial"]
    factorial_scenes = factorial_cache.prefix_images.shape[0]
    for scene_index in range(factorial_scenes):
        prediction, _ = joint._run_factorial_scene(
            controller,
            factorial_cache,
            scene_index,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        target = joint._factorial_targets(factorial_cache, scene_index, device)
        losses = []
        for row_index in factorial_indices:
            spec = specs[row_index]
            normalized_squared = (
                (prediction[spec["component"]] - target[spec["component"]])
                / scales[spec["component"]]
            ).square()
            horizon_index = spec["horizon_index"]
            contribution = (
                normalized_squared.mean()
                if horizon_index is None
                else normalized_squared[horizon_index]
            )
            losses.append(contribution / factorial_scenes)
        _accumulate_gradients(rows, factorial_indices, losses, parameters)

    attitude_indices = [index for index, spec in enumerate(specs) if spec["kind"] == "attitude"]
    attitude_scenes = attitude_cache.images.shape[0]
    label_indices = [joint.PREFIX_STEPS + step - 1 for step in joint.SUPERVISION_STEPS]
    rpy_scales = torch.tensor(joint.RPY_SCALES, device=device)
    for scene_index in range(attitude_scenes):
        if not attitude_indices:
            break
        prediction, _ = joint._run_attitude_scene(
            controller,
            attitude_cache,
            scene_index,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        target = attitude_cache.source_motor[scene_index, label_indices, :3].to(device)
        per_axis_mse = ((prediction - target) / rpy_scales).square().mean(dim=0)
        losses = [
            per_axis_mse[specs[index]["axis_index"]] / attitude_scenes
            for index in attitude_indices
        ]
        _accumulate_gradients(rows, attitude_indices, losses, parameters)
    return rows


def project_inequality_displacement(
    displacement: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    if len(rows) != len(specs) or not rows:
        raise ValueError("one Jacobian row is required for every nonempty constraint spec")
    count = len(rows)
    gram = torch.empty(
        count,
        count,
        dtype=torch.float64,
        device=next(iter(rows[0].values())).device,
    )
    j_displacement = torch.empty(count, dtype=torch.float64, device=gram.device)
    for row_index, row in enumerate(rows):
        j_displacement[row_index] = sum(
            (row[name] * displacement[name]).sum() for name in joint.PARAMETER_FAMILIES
        ).double()
        for column_index in range(row_index + 1):
            other = rows[column_index]
            value = sum(
                (PARAMETER_LEARNING_RATES[name] ** 2)
                * (row[name] * other[name]).sum()
                for name in joint.PARAMETER_FAMILIES
            )
            gram[row_index, column_index] = value.double()
            gram[column_index, row_index] = value.double()
    residual = torch.tensor(
        [spec["current_mse"] - spec["limit_mse"] for spec in specs],
        dtype=torch.float64,
        device=gram.device,
    )
    violation_before = residual + j_displacement
    row_norm = gram.diagonal().clamp_min(0.0).sqrt()
    active = row_norm > 1.0e-15
    infeasible_zero_rows = (~active) & (violation_before > LINEAR_CONSTRAINT_TOLERANCE)
    if bool(infeasible_zero_rows.any()):
        return displacement, {
            "pass": False,
            "solver_success": False,
            "reason": "a violated constraint has a numerically zero Jacobian row",
            "constraint_names": [spec["name"] for spec in specs],
            "linearized_violation_before": violation_before.tolist(),
        }
    active_indices = active.nonzero(as_tuple=False)[:, 0]
    normalized_gram = gram[active][:, active] / (
        row_norm[active, None] * row_norm[None, active]
    )
    normalized_violation = violation_before[active] / row_norm[active]
    gram_numpy = normalized_gram.detach().cpu().numpy()
    violation_numpy = normalized_violation.detach().cpu().numpy()

    def objective(value: np.ndarray) -> float:
        return float(0.5 * value @ gram_numpy @ value - violation_numpy @ value)

    def gradient(value: np.ndarray) -> np.ndarray:
        return gram_numpy @ value - violation_numpy

    result = minimize(
        objective,
        np.zeros(len(active_indices), dtype=np.float64),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * len(active_indices),
        options={"ftol": 1.0e-15, "gtol": DUAL_GRADIENT_TOLERANCE, "maxiter": 10_000},
    )
    coefficients = torch.zeros(count, dtype=torch.float64, device=gram.device)
    coefficients[active_indices] = torch.from_numpy(result.x).to(gram.device) / row_norm[active]
    projected = {name: value.detach().clone() for name, value in displacement.items()}
    for coefficient, row in zip(coefficients, rows, strict=True):
        coefficient_value = float(coefficient)
        for name in joint.PARAMETER_FAMILIES:
            projected[name].add_(
                row[name],
                alpha=-(PARAMETER_LEARNING_RATES[name] ** 2) * coefficient_value,
            )
    linearized_after = residual.clone()
    for row_index, row in enumerate(rows):
        linearized_after[row_index] += sum(
            (row[name] * projected[name]).sum() for name in joint.PARAMETER_FAMILIES
        ).double()
    maximum_violation = float(linearized_after.max())
    projected_distance = math.sqrt(
        sum(
            float(
                ((projected[name] - displacement[name]) / PARAMETER_LEARNING_RATES[name])
                .square()
                .sum()
            )
            for name in joint.PARAMETER_FAMILIES
        )
    )
    return projected, {
        "pass": bool(result.success and maximum_violation <= LINEAR_CONSTRAINT_TOLERANCE),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "solver_iterations": int(result.nit),
        "constraint_names": [spec["name"] for spec in specs],
        "active_rpy_rows": [
            spec["name"] for spec in specs if spec["kind"] == "attitude"
        ],
        "linearized_violation_before": violation_before.tolist(),
        "linearized_violation_after": linearized_after.tolist(),
        "maximum_linearized_violation_after": maximum_violation,
        "dual_coefficients": coefficients.tolist(),
        "learning_rate_scaled_projection_distance": projected_distance,
    }


def preservation_decision(
    source: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    reasons = []
    for component in ("common", "height"):
        if (
            candidate["component_nrmse"][component]
            > source["component_nrmse"][component] + joint.COMPONENT_BASELINE_TOLERANCE
        ):
            reasons.append(f"aggregate {component} NRMSE exceeded original source plus 0.02")
        for horizon in map(str, joint.SUPERVISION_STEPS):
            if (
                candidate["by_supervision_step_nrmse"][horizon][component]
                > source["by_supervision_step_nrmse"][horizon][component]
                + joint.COMPONENT_BASELINE_TOLERANCE
            ):
                reasons.append(
                    f"{component} NRMSE at step {horizon} exceeded original source plus 0.02"
                )
    for axis, value in candidate["rpy_source_nrmse"].items():
        if value > joint.RPY_NRMSE_LIMIT:
            reasons.append(f"{axis} source NRMSE exceeded 0.05")
    if candidate["motor_output_max_absolute"] > 1.0:
        reasons.append("motor output exceeded [-1, 1]")
    if not candidate["all_attitude_cache_states_valid"]:
        reasons.append("attitude cache contains an invalid physical state")
    finite = joint._numeric_tree_is_finite(candidate)
    if not finite:
        reasons.append("one or more candidate metrics were nonfinite")
    return {"pass": not reasons, "reasons": reasons, "all_metrics_finite": finite}


def candidate_decision(
    source: dict[str, Any],
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    damping_directional_derivative: float,
) -> dict[str, Any]:
    decision = preservation_decision(source, candidate)
    reasons = list(decision["reasons"])
    improvement = current["endpoint_damping_nrmse"] - candidate["endpoint_damping_nrmse"]
    if improvement < endpoint.MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT:
        reasons.append("actual endpoint damping NRMSE improvement was below 0.001")
    if damping_directional_derivative >= 0.0:
        reasons.append("actual post-bound displacement was not a damping descent direction")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "actual_endpoint_damping_nrmse_improvement": improvement,
        "damping_directional_derivative": damping_directional_derivative,
        "all_metrics_finite": decision["all_metrics_finite"],
    }


def damping_directional_derivative(
    gradients: dict[str, Tensor],
    displacement: dict[str, Tensor],
    *,
    scale: float = 1.0,
) -> float:
    return float(
        sum(
            (
                gradients[name].double()
                * (displacement[name].double() / scale)
            ).sum()
            for name in joint.PARAMETER_FAMILIES
        )
    )


def terminal_decision(
    source_training: dict[str, Any],
    training: dict[str, Any],
    source_development: dict[str, Any],
    development: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    for label, source, candidate in (
        ("training", source_training, training),
        ("development", source_development, development),
    ):
        preservation = preservation_decision(source, candidate)
        reasons.extend(f"{label}: {reason}" for reason in preservation["reasons"])
        threshold = TRAIN_TERMINAL_D_NRMSE if label == "training" else DEVELOPMENT_TERMINAL_D_NRMSE
        if candidate["endpoint_damping_nrmse"] > threshold:
            reasons.append(f"{label} endpoint damping NRMSE exceeded {threshold:.2f}")
        if candidate["endpoint_damping_correct_sign_fraction"] < TERMINAL_SIGN_FRACTION:
            reasons.append(f"{label} endpoint damping correct-sign fraction was below 0.90")
        gain = candidate["endpoint_damping_teacher_aligned_gain"]
        if not (TERMINAL_GAIN_RANGE[0] <= gain <= TERMINAL_GAIN_RANGE[1]):
            reasons.append(f"{label} endpoint damping gain was outside [0.5, 1.5]")
    return {"pass": not reasons, "reasons": reasons}


def mandatory_gate_decision(
    source_training: dict[str, Any],
    training: dict[str, Any],
    source_development: dict[str, Any],
    development: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    improvements = {}
    for label, source, candidate in (
        ("training", source_training, training),
        ("development", source_development, development),
    ):
        fraction = 1.0 - candidate["endpoint_damping_nrmse"] / source[
            "endpoint_damping_nrmse"
        ]
        improvements[label] = fraction
        if fraction < MANDATORY_D_IMPROVEMENT_FRACTION:
            reasons.append(f"{label} endpoint damping NRMSE improvement was below 25%")
        preservation = preservation_decision(source, candidate)
        reasons.extend(f"{label}: {reason}" for reason in preservation["reasons"])
    return {"pass": not reasons, "reasons": reasons, "improvement_fraction": improvements}


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
    base = joint._copy_parameters(student)
    specs = constraint_specs(source_metrics, current_metrics)
    print(json.dumps({"stage": "constraint_jacobian", "rows": len(specs)}), flush=True)
    rows = constraint_gradient_rows(
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
        name: raw_parameters[name] - base[name] for name in joint.PARAMETER_FAMILIES
    }
    projected, projection = project_inequality_displacement(raw_displacement, rows, specs)
    step_audit.set_displacement(student, base, projected, scale=1.0)
    bounded_parameters = joint._copy_parameters(student)
    bounded = {
        name: bounded_parameters[name] - base[name] for name in joint.PARAMETER_FAMILIES
    }
    bounded_linearized = [
        spec["current_mse"]
        - spec["limit_mse"]
        + float(sum((row[name] * bounded[name]).sum() for name in joint.PARAMETER_FAMILIES))
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
        sum((raw_gradients[name] * bounded[name]).sum() for name in joint.PARAMETER_FAMILIES)
    )
    projection.update(
        {
            "pass_after_parameter_bounds": (
                projection["pass"]
                and maximum_bounded_violation <= LINEAR_CONSTRAINT_TOLERANCE
                and bounded_derivative < 0.0
            ),
            "bounded_linearized_violation_after": bounded_linearized,
            "maximum_bounded_linearized_violation_after": maximum_bounded_violation,
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
    joint._load_parameters(student, base)
    return bounded, raw_gradients, projection


def find_safe_trial(
    student: ConnectomeController,
    base: dict[str, Tensor],
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
    trials = []
    selected_scale = None
    selected_metrics = None
    for scale in BACKTRACK_SCALES:
        actual_rms = install_trial(student, base, displacement, scale=scale)
        actual = {
            name: getattr(student, name).detach() - base[name] for name in joint.PARAMETER_FAMILIES
        }
        derivative = damping_directional_derivative(
            raw_gradients,
            actual,
        )
        metrics, _ = endpoint.evaluate(
            student,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        decision = candidate_decision(
            source_metrics,
            current_metrics,
            metrics,
            damping_directional_derivative=derivative,
        )
        trials.append(
            {
                "scale": scale,
                "metrics": metrics,
                "decision": decision,
                "actual_displacement_family_rms": actual_rms,
            }
        )
        if decision["pass"]:
            selected_scale = scale
            selected_metrics = metrics
            break
    if selected_scale is None:
        joint._load_parameters(student, base)
    return selected_scale, selected_metrics, trials


def install_trial(
    controller: ConnectomeController,
    source: dict[str, Tensor],
    displacement: dict[str, Tensor],
    *,
    scale: float,
) -> dict[str, float]:
    return step_audit.set_displacement(
        controller, source, displacement, scale=scale
    )


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(f"{path.suffix}.staging")
    torch.save(payload, staging)
    os.replace(staging, path)


def save_resume(
    path: Path,
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    source_checkpoint_sha256: str,
    accepted_updates: int,
    history: list[dict[str, Any]],
    development_history: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "accepted_updates": accepted_updates,
            "controller": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "development_history": development_history,
            "preflight": preflight,
        },
        path,
    )


def load_resume(
    path: Path,
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    source_checkpoint_sha256: str,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(path, map_location=student.edge_magnitude.device, weights_only=True)
    expected = (EXPERIMENT, PROTOCOL_COMMIT, source_checkpoint_sha256)
    actual = (
        payload.get("experiment"),
        payload.get("protocol_commit"),
        payload.get("source_checkpoint_sha256"),
    )
    if actual != expected:
        raise SystemExit("resume state does not match the frozen protocol or source")
    student.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["accepted_updates"]),
        list(payload["history"]),
        list(payload["development_history"]),
        dict(payload["preflight"]),
    )


def _aggregate_nrmse(metrics: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for item in metrics:
        value: Any = item
        for key in path:
            value = value[key]
        values.append(float(value))
    return math.sqrt(sum(value * value for value in values) / len(values))


def aggregate_streamed_metrics(
    metrics: list[dict[str, Any]],
    endpoint_predictions: list[Tensor],
    endpoint_targets: list[Tensor],
    *,
    endpoint_scale: float,
) -> dict[str, Any]:
    prediction = torch.cat(endpoint_predictions)
    target = torch.cat(endpoint_targets)
    return {
        "joint_normalized_mse": sum(item["joint_normalized_mse"] for item in metrics)
        / len(metrics),
        "component_nrmse": {
            component: _aggregate_nrmse(metrics, ("component_nrmse", component))
            for component in joint.COMPONENTS
        },
        "by_supervision_step_nrmse": {
            str(horizon): {
                component: _aggregate_nrmse(
                    metrics,
                    ("by_supervision_step_nrmse", str(horizon), component),
                )
                for component in joint.COMPONENTS
            }
            for horizon in joint.SUPERVISION_STEPS
        },
        "rpy_source_nrmse": {
            axis: _aggregate_nrmse(metrics, ("rpy_source_nrmse", axis))
            for axis in ("roll", "pitch", "yaw")
        },
        "endpoint_damping_nrmse": float(
            ((prediction - target) / endpoint_scale).square().mean().sqrt()
        ),
        "endpoint_damping_correct_sign_fraction": float(
            ((prediction * target) > 0.0).float().mean()
        ),
        "endpoint_damping_teacher_aligned_gain": float(
            (prediction * target).sum() / target.square().sum().clamp_min(1.0e-12)
        ),
        "motor_output_max_absolute": max(item["motor_output_max_absolute"] for item in metrics),
        "all_attitude_cache_states_valid": all(
            item["all_attitude_cache_states_valid"] for item in metrics
        ),
    }


def qualification_decision(
    source: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    preservation = preservation_decision(source, candidate)
    reasons = list(preservation["reasons"])
    if candidate["endpoint_damping_nrmse"] > DEVELOPMENT_TERMINAL_D_NRMSE:
        reasons.append("qualification endpoint damping NRMSE exceeded 0.30")
    if candidate["endpoint_damping_correct_sign_fraction"] < TERMINAL_SIGN_FRACTION:
        reasons.append("qualification endpoint damping correct-sign fraction was below 0.90")
    gain = candidate["endpoint_damping_teacher_aligned_gain"]
    if not (TERMINAL_GAIN_RANGE[0] <= gain <= TERMINAL_GAIN_RANGE[1]):
        reasons.append("qualification endpoint damping gain was outside [0.5, 1.5]")
    return {"pass": not reasons, "reasons": reasons}


@torch.no_grad()
def run_qualification(
    source: ConnectomeController,
    candidate: ConnectomeController,
    scales: dict[str, float],
    endpoint_scale: float,
    *,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    source_metrics = []
    candidate_metrics = []
    source_predictions = []
    candidate_predictions = []
    endpoint_targets = []
    cache_hashes = []
    for bank_index, (factorial_seed, attitude_seed) in enumerate(
        zip(QUALIFICATION_FACTORIAL_SEEDS, QUALIFICATION_ATTITUDE_SEEDS, strict=True), start=1
    ):
        print(
            json.dumps(
                {
                    "stage": "qualification_bank",
                    "bank": bank_index,
                    "banks": len(QUALIFICATION_FACTORIAL_SEEDS),
                }
            ),
            flush=True,
        )
        factorial_cache = joint.build_factorial_cache(
            scenes=QUALIFICATION_SCENES_PER_BANK,
            seed=factorial_seed,
            response_steps=joint.RESPONSE_STEPS,
            policy_hz=joint.POLICY_HZ,
            device=device,
            config=config,
        )
        attitude_cache = joint.build_attitude_cache(
            source,
            scenes=QUALIFICATION_SCENES_PER_BANK,
            seed=attitude_seed,
            prefix_steps=joint.PREFIX_STEPS,
            response_steps=joint.RESPONSE_STEPS,
            policy_hz=joint.POLICY_HZ,
            device=device,
            config=config,
        )
        source_bank, source_raw = endpoint.evaluate(
            source,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        candidate_bank, candidate_raw = endpoint.evaluate(
            candidate,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        target = torch.stack(
            [
                joint._factorial_targets(factorial_cache, scene_index, device)["damping"][-1]
                for scene_index in range(QUALIFICATION_SCENES_PER_BANK)
            ]
        ).cpu()
        source_metrics.append(source_bank)
        candidate_metrics.append(candidate_bank)
        source_predictions.append(source_raw["damping"][:, -1].cpu())
        candidate_predictions.append(candidate_raw["damping"][:, -1].cpu())
        endpoint_targets.append(target)
        cache_hashes.append(
            {
                "factorial_seed": factorial_seed,
                "factorial_tensor_sha256": factorial_cache.sha256,
                "attitude_seed": attitude_seed,
                "attitude_tensor_sha256": attitude_cache.sha256,
                "endpoint_opposite_motion_image_max_absolute_difference": (
                    factorial_cache.endpoint_image_difference_max
                ),
            }
        )
        del factorial_cache, attitude_cache, source_raw, candidate_raw
    source_aggregate = aggregate_streamed_metrics(
        source_metrics, source_predictions, endpoint_targets, endpoint_scale=endpoint_scale
    )
    candidate_aggregate = aggregate_streamed_metrics(
        candidate_metrics, candidate_predictions, endpoint_targets, endpoint_scale=endpoint_scale
    )
    decision = qualification_decision(source_aggregate, candidate_aggregate)
    if any(
        item["endpoint_opposite_motion_image_max_absolute_difference"] != 0.0
        for item in cache_hashes
    ):
        decision["pass"] = False
        decision["reasons"].append("one or more qualification endpoint image pairs differed")
    return {
        "source": source_aggregate,
        "candidate": candidate_aggregate,
        "decision": decision,
        "cache_hashes": cache_hashes,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    started = perf_counter()
    graph_sha256 = responsibility.file_sha256(args.graph)
    checkpoint_sha256 = responsibility.file_sha256(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    student.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    source_parameters = joint._copy_parameters(source)

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
    if cache_integrity["manifest_sha256"] != EXPECTED_SOURCE_CACHE_MANIFEST_SHA256:
        raise SystemExit("authorized cache manifest hash does not match the protocol")
    endpoint_scale = endpoint.endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )
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

    optimizer = torch.optim.Adam(
        [
            {
                "params": [student.edge_magnitude, student.bias],
                "lr": joint.EDGE_BIAS_LEARNING_RATE,
            },
            {
                "params": [student.raw_time_constant],
                "lr": joint.TIME_CONSTANT_LEARNING_RATE,
            },
        ],
        weight_decay=0.0,
    )
    resume_path = args.output_dir / "resume.pt"
    resumed = resume_path.is_file()
    accepted_updates = 0
    history: list[dict[str, Any]] = []
    development_history: list[dict[str, Any]] = []
    resumed_preflight = None
    if resumed:
        accepted_updates, history, development_history, resumed_preflight = load_resume(
            resume_path,
            student,
            optimizer,
            source_checkpoint_sha256=checkpoint_sha256,
        )
    current_training, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )

    preflight = None
    if not resumed:
        print(json.dumps({"stage": "constrained_proposal_preflight"}), flush=True)
        preflight_parameters = joint._copy_parameters(student)
        preflight_optimizer = copy.deepcopy(optimizer.state_dict())
        displacement, raw_gradients, projection = make_projected_proposal(
            student,
            optimizer,
            source_training,
            current_training,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale,
            device=device,
        )
        install_trial(
            student,
            preflight_parameters,
            displacement,
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
            name: getattr(student, name).detach() - preflight_parameters[name]
            for name in joint.PARAMETER_FAMILIES
        }
        derivative = damping_directional_derivative(
            raw_gradients,
            actual_fd,
            scale=joint.FINITE_DIFFERENCE_SCALE,
        )
        finite_difference = (
            finite_difference_metrics["endpoint_damping_nrmse"] ** 2
            - current_training["endpoint_damping_nrmse"] ** 2
        ) / joint.FINITE_DIFFERENCE_SCALE
        relative_error = abs(finite_difference - derivative) / max(
            abs(finite_difference), abs(derivative), 1.0e-12
        )
        replay_decision = candidate_decision(
            source_training,
            current_training,
            finite_difference_metrics,
            damping_directional_derivative=derivative,
        )
        preflight = {
            "pass": bool(
                cache_integrity["all_persisted_file_and_tensor_hashes_match"]
                and teacher_identity["pass"]
                and teacher_positive["pass"]
                and source_replay_difference <= joint.REPLAY_TOLERANCE
                and train_factorial.endpoint_image_difference_max == 0.0
                and development_factorial.endpoint_image_difference_max == 0.0
                and projection["pass_after_parameter_bounds"]
                and derivative < 0.0
                and finite_difference < 0.0
                and relative_error <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                and replay_decision["pass"]
            ),
            "projection": projection,
            "finite_difference": {
                "scale": joint.FINITE_DIFFERENCE_SCALE,
                "autograd_directional_derivative": derivative,
                "forward_finite_difference": finite_difference,
                "relative_error": relative_error,
                "pass": (
                    derivative < 0.0
                    and finite_difference < 0.0
                    and relative_error <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
                ),
            },
            "complete_replay_metrics": finite_difference_metrics,
            "complete_replay_decision": replay_decision,
        }
        joint._load_parameters(student, preflight_parameters)
        optimizer.load_state_dict(preflight_optimizer)
        current_training = source_training
    else:
        if resumed_preflight is None or not resumed_preflight["pass"]:
            raise SystemExit("resume state lacks a passing constrained-proposal preflight")
        preflight = resumed_preflight
        preflight["resumed_from_prior_passing_preflight"] = True

    stop_reason = None
    terminal_checkpoint = None
    if not preflight["pass"]:
        stop_reason = "constrained proposal preflight failed"
    while stop_reason is None and accepted_updates < MAXIMUM_ACCEPTED_UPDATES:
        update_number = accepted_updates + 1
        base = joint._copy_parameters(student)
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        displacement, raw_gradients, projection = make_projected_proposal(
            student,
            optimizer,
            source_training,
            current_training,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale,
            device=device,
        )
        if not projection["pass_after_parameter_bounds"]:
            joint._load_parameters(student, base)
            optimizer.load_state_dict(optimizer_before)
            stop_reason = "constraint projection failed"
            history.append(
                {
                    "update": update_number,
                    "accepted": False,
                    "projection": projection,
                    "trials": [],
                }
            )
            break
        selected_scale, selected_metrics, trials = find_safe_trial(
            student,
            base,
            displacement,
            raw_gradients,
            source_training,
            current_training,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale,
            device=device,
        )
        entry = {
            "update": update_number,
            "accepted": selected_scale is not None,
            "accepted_scale": selected_scale,
            "projection": projection,
            "trials": trials,
        }
        history.append(entry)
        if selected_scale is None or selected_metrics is None:
            optimizer.load_state_dict(optimizer_before)
            stop_reason = "deterministic constrained proposal had no acceptable scale"
            break
        accepted_updates = update_number
        current_training = selected_metrics
        save_resume(
            resume_path,
            student,
            optimizer,
            source_checkpoint_sha256=checkpoint_sha256,
            accepted_updates=accepted_updates,
            history=history,
            development_history=development_history,
            preflight=preflight,
        )
        print(
            json.dumps(
                {
                    "progress": "accepted_update",
                    "update": accepted_updates,
                    "scale": selected_scale,
                    "training_endpoint_damping_nrmse": current_training[
                        "endpoint_damping_nrmse"
                    ],
                    "training_endpoint_damping_sign": current_training[
                        "endpoint_damping_correct_sign_fraction"
                    ],
                    "training_endpoint_damping_gain": current_training[
                        "endpoint_damping_teacher_aligned_gain"
                    ],
                }
            ),
            flush=True,
        )
        if accepted_updates % DEVELOPMENT_INTERVAL != 0:
            continue
        development_metrics, _ = endpoint.evaluate(
            student,
            development_factorial,
            development_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        preservation = preservation_decision(source_development, development_metrics)
        mandatory = (
            mandatory_gate_decision(
                source_training,
                current_training,
                source_development,
                development_metrics,
            )
            if accepted_updates == MANDATORY_GATE_UPDATE
            else None
        )
        terminal = terminal_decision(
            source_training,
            current_training,
            source_development,
            development_metrics,
        )
        development_entry = {
            "update": accepted_updates,
            "metrics": development_metrics,
            "preservation": preservation,
            "mandatory_update_50_gate": mandatory,
            "terminal": terminal,
        }
        development_history.append(development_entry)
        save_resume(
            resume_path,
            student,
            optimizer,
            source_checkpoint_sha256=checkpoint_sha256,
            accepted_updates=accepted_updates,
            history=history,
            development_history=development_history,
            preflight=preflight,
        )
        print(
            json.dumps(
                {
                    "progress": "development",
                    "update": accepted_updates,
                    "endpoint_damping_nrmse": development_metrics["endpoint_damping_nrmse"],
                    "preservation_pass": preservation["pass"],
                    "mandatory_gate_pass": mandatory["pass"] if mandatory else None,
                    "terminal_pass": terminal["pass"],
                }
            ),
            flush=True,
        )
        if mandatory is not None and not mandatory["pass"]:
            stop_reason = "mandatory update-50 gate failed"
            break
        if terminal["pass"]:
            terminal_checkpoint = args.output_dir / "nonpromotional-terminal.pt"
            atomic_torch_save(
                {
                    "experiment": EXPERIMENT,
                    "protocol_commit": PROTOCOL_COMMIT,
                    "source_checkpoint_sha256": checkpoint_sha256,
                    "graph_sha256": graph_sha256,
                    "accepted_updates": accepted_updates,
                    "controller": student.state_dict(),
                },
                terminal_checkpoint,
            )
            stop_reason = "first scheduled terminal checkpoint qualified"
            break
    if stop_reason is None:
        stop_reason = "maximum accepted-update budget reached without terminal qualification"

    qualification = None
    if terminal_checkpoint is not None:
        qualification = run_qualification(
            source, student, scales, endpoint_scale, device=device, config=config
        )
    fresh_pass = bool(qualification and qualification["decision"]["pass"])
    if terminal_checkpoint is None:
        final_training = current_training
    else:
        final_training, _ = endpoint.evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
    final_source_distance = {
        name: float(
            (getattr(student, name).detach() - source_parameters[name]).square().mean().sqrt()
        )
        for name in joint.PARAMETER_FAMILIES
    }
    report = {
        "experiment": EXPERIMENT,
        "status": "nonpromotional_fitting_diagnostic",
        "pass": fresh_pass,
        "classification": (
            "fresh_qualification_passed"
            if fresh_pass
            else (
                "terminal_checkpoint_failed_fresh_qualification"
                if terminal_checkpoint is not None
                else "no_terminal_checkpoint"
            )
        ),
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "actor_contract_unchanged": True,
        "protocol": protocol_manifest(),
        "cache_integrity": cache_integrity,
        "teacher_component_scales_motor_units": scales,
        "endpoint_damping_scale_motor_units": endpoint_scale,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "source_replay_max_absolute_difference": source_replay_difference,
        "resumed": resumed,
        "resume_state": responsibility.stable_path(resume_path) if resume_path.is_file() else None,
        "resume_state_sha256": (
            responsibility.file_sha256(resume_path) if resume_path.is_file() else None
        ),
        "preflight": preflight,
        "accepted_updates": accepted_updates,
        "stop_reason": stop_reason,
        "source_training": source_training,
        "source_development": source_development,
        "final_training": final_training,
        "history": history,
        "development_history": development_history,
        "terminal_checkpoint": (
            responsibility.stable_path(terminal_checkpoint)
            if terminal_checkpoint is not None
            else None
        ),
        "terminal_checkpoint_sha256": (
            responsibility.file_sha256(terminal_checkpoint)
            if terminal_checkpoint is not None
            else None
        ),
        "qualification": qualification,
        "fresh_qualification_pass": fresh_pass,
        "closed_loop_hover_authorized": fresh_pass,
        "promoted": False,
        "closed_loop_hover_run": False,
        "final_parameter_family_rms_from_source": final_source_distance,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "Even a fresh pass establishes damping fitting only and cannot establish "
            "calibrated collective, stable hover, or flight."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": fresh_pass,
                "classification": report["classification"],
                "accepted_updates": accepted_updates,
                "stop_reason": stop_reason,
                "terminal_checkpoint": report["terminal_checkpoint"],
                "fresh_qualification_pass": fresh_pass,
                "closed_loop_hover_authorized": fresh_pass,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
