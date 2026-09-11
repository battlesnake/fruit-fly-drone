#!/usr/bin/env python3
"""Run the preregistered full-native joint C/P/D teacher-learning diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_full_native_d_first_fp64_update25_exhaustive as audit25  # noqa: E402
import audit_variable_height_full_native_d_first_fp64_update25_slsqp_polished as polished  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as d_first  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-joint-teacher-learning-v1"
PROTOCOL_COMMIT = "f0e5614"
EXPECTED_GRAPH_SHA256 = audit25.EXPECTED_GRAPH_SHA256
EXPECTED_CHECKPOINT_SHA256 = audit25.EXPECTED_CHECKPOINT_SHA256
EXPECTED_CACHE_MANIFEST_SHA256 = d_first.EXPECTED_SOURCE_CACHE_MANIFEST_SHA256
SOURCE_DENOMINATOR_FLOOR_MSE = 0.25**2
MINIMUM_J_IMPROVEMENT = 1.0e-4
MIDPOINT_UPDATE = 25
MIDPOINT_D_IMPROVEMENT_FRACTION = 0.15
FINAL_UPDATE = 50
FINAL_D_IMPROVEMENT_FRACTION = 0.25
FINAL_D_SIGN_FRACTION = 0.50


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
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "runs/variable-height-hover/full-native-joint-teacher-learning-001"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def locked_input_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "graph": args.graph,
        "checkpoint": args.checkpoint,
        "cache_manifest": args.source_audit_dir / "immutable-cache-v1/manifest.json",
    }


def expected_locked_hashes() -> dict[str, str]:
    return {
        "graph": EXPECTED_GRAPH_SHA256,
        "checkpoint": EXPECTED_CHECKPOINT_SHA256,
        "cache_manifest": EXPECTED_CACHE_MANIFEST_SHA256,
    }


def validate_args(args: argparse.Namespace) -> None:
    missing = [path for path in locked_input_paths(args).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input: {missing[0]}")
    if args.smoke_test:
        raise SystemExit("this preregistered joint teacher run has no smoke variant")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the joint teacher-learning experiment already has a final report")
    hashes = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    if hashes != expected_locked_hashes():
        raise SystemExit("one or more hash-locked joint teacher inputs do not match")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source": "original native visual-hover checkpoint, not a prior fit",
        "source_cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
        "actor_inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
        "privileged_actor_inputs": [],
        "native_state_initialization": "zero",
        "topology_signs_input_mapping_recurrence_and_foreleg_outputs_frozen": True,
        "opened_parameter_families": list(joint.PARAMETER_FAMILIES),
        "objective": {
            "name": "J",
            "components": ["common", "height", "endpoint_damping"],
            "component_weighting": "equal one third each",
            "common_and_height_weighting": "scenes and supervision horizons equally",
            "damping_horizon": joint.RESPONSE_STEPS,
            "source_denominator_floor_mse": SOURCE_DENOMINATOR_FLOOR_MSE,
            "denominators_frozen_from_training_source_before_updates": True,
            "interaction_is_reported_not_optimized": True,
            "rpy_is_constrained_not_optimized": True,
        },
        "optimizer": {
            "name": "Adam",
            "initialized_fresh": True,
            "initial_update_counter": 0,
            "edge_and_bias_learning_rate": joint.EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": joint.TIME_CONSTANT_LEARNING_RATE,
            "weight_decay": 0.0,
            "global_gradient_norm_cap": joint.GRADIENT_NORM_CAP,
            "one_gradient_and_adam_transaction_per_attempt": True,
            "pre_attempt_state_restored_on_rejection": True,
        },
        "projection": {
            "c_and_p_source_ceiling_rows": False,
            "activated_rpy_rows_only": True,
            "rpy_activation_nrmse": d_first.RPY_CONSTRAINT_ACTIVATION_NRMSE,
            "rpy_limit_nrmse": joint.RPY_NRMSE_LIMIT,
            "active_solver": (
                "exhaustive active-support Lawson-Hanson NNLS with SLSQP-support FP64 polish"
            ),
            "zero_active_rpy_rows": "identity raw proposal before native-bound materialization",
            "old_projector_fallback": False,
            "linear_constraint_tolerance": d_first.LINEAR_CONSTRAINT_TOLERANCE,
        },
        "proposal_controls": {
            "negative_authoritative_j_direction": True,
            "finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
            "finite_difference_minimum_absolute_change": (
                audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
            ),
            "finite_difference_relative_error_limit": (
                joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
            ),
            "native_bounds_and_canonical_idempotence": True,
            "finite_state_and_actuator_output": True,
        },
        "selection": {
            "fixed_training_bank_only": True,
            "backtrack_scales_descending": list(d_first.BACKTRACK_SCALES),
            "minimum_actual_j_improvement": MINIMUM_J_IMPROVEMENT,
            "per_step_endpoint_d_improvement_required": False,
            "d_may_temporarily_worsen_before_milestone": True,
            "nonlinear_c25_repair": False,
            "smaller_scales": False,
        },
        "midpoint_gate": {
            "update": MIDPOINT_UPDATE,
            "endpoint_d_source_relative_improvement_minimum": (MIDPOINT_D_IMPROVEMENT_FRACTION),
            "aggregate_common_and_height_no_worse_than_training_source": True,
            "failure_is_terminal": True,
        },
        "final_training_gate": {
            "update": FINAL_UPDATE,
            "endpoint_d_source_relative_improvement_minimum": (FINAL_D_IMPROVEMENT_FRACTION),
            "endpoint_d_correct_sign_fraction_minimum": FINAL_D_SIGN_FRACTION,
            "common_and_height_each_horizon_no_worse_than_training_source": True,
            "failure_is_terminal": True,
        },
        "development": {
            "candidate_evaluations_before_passing_final_training_gate": 0,
            "candidate_evaluations_after_passing_final_training_gate": 1,
            "same_final_bank_gates": True,
            "development_started_persisted_before_evaluation": True,
            "source_and_candidate_started_and_completed_counts_persisted": True,
            "interrupted_evaluation_retried": False,
        },
        "maximum_accepted_updates": FINAL_UPDATE,
        "failure_restores_current_controller_and_adam_and_persists_stopped_resume": True,
        "passing_authorizes": "a subsequent learning experiment only",
        "closed_loop_hover": False,
        "gate_flight": False,
        "promotion": False,
        "large_run_artifacts_remain_ignored": True,
    }


def objective_denominators(source_metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "common": max(
            source_metrics["component_nrmse"]["common"] ** 2,
            SOURCE_DENOMINATOR_FLOOR_MSE,
        ),
        "height": max(
            source_metrics["component_nrmse"]["height"] ** 2,
            SOURCE_DENOMINATOR_FLOOR_MSE,
        ),
        "endpoint_damping": max(
            source_metrics["endpoint_damping_nrmse"] ** 2,
            SOURCE_DENOMINATOR_FLOOR_MSE,
        ),
    }


def joint_teacher_objective(metrics: dict[str, Any], denominators: dict[str, float]) -> float:
    return (
        metrics["component_nrmse"]["common"] ** 2 / denominators["common"]
        + metrics["component_nrmse"]["height"] ** 2 / denominators["height"]
        + metrics["endpoint_damping_nrmse"] ** 2 / denominators["endpoint_damping"]
    ) / 3.0


def attach_joint_objective(
    metrics: dict[str, Any], denominators: dict[str, float]
) -> dict[str, Any]:
    result = copy.deepcopy(metrics)
    result["joint_teacher_objective"] = joint_teacher_objective(result, denominators)
    return result


def accumulated_joint_teacher_gradient(
    controller: ConnectomeController,
    cache: joint.FactorialCache,
    scales: dict[str, float],
    endpoint_scale: float,
    denominators: dict[str, float],
    *,
    device: torch.device,
) -> None:
    scenes = cache.prefix_images.shape[0]
    for scene_index in range(scenes):
        prediction, _ = joint._run_factorial_scene(
            controller,
            cache,
            scene_index,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        target = joint._factorial_targets(cache, scene_index, device)
        common_mse = ((prediction["common"] - target["common"]) / scales["common"]).square().mean()
        height_mse = ((prediction["height"] - target["height"]) / scales["height"]).square().mean()
        endpoint_damping_mse = (
            (prediction["damping"][-1] - target["damping"][-1]) / endpoint_scale
        ).square()
        loss = (
            common_mse / denominators["common"]
            + height_mse / denominators["height"]
            + endpoint_damping_mse / denominators["endpoint_damping"]
        ) / (3.0 * scenes)
        loss.backward()


def rpy_constraint_specs(current_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for axis_index, axis in enumerate(("roll", "pitch", "yaw")):
        current_value = current_metrics["rpy_source_nrmse"][axis]
        if current_value >= d_first.RPY_CONSTRAINT_ACTIVATION_NRMSE:
            specs.append(
                {
                    "name": f"rpy.{axis}",
                    "kind": "attitude",
                    "axis_index": axis_index,
                    "current_mse": current_value**2,
                    "limit_mse": joint.RPY_NRMSE_LIMIT**2,
                    "remaining_mse_allowance": (joint.RPY_NRMSE_LIMIT**2 - current_value**2),
                }
            )
    return specs


def rpy_constraint_gradient_rows(
    controller: ConnectomeController,
    cache: joint.AttitudeCache,
    specs: list[dict[str, Any]],
    *,
    device: torch.device,
) -> list[dict[str, Tensor]]:
    if not specs:
        return []
    rows = d_first._empty_gradient_rows(controller, len(specs))
    parameters = tuple(getattr(controller, name) for name in joint.PARAMETER_FAMILIES)
    scenes = cache.images.shape[0]
    label_indices = [joint.PREFIX_STEPS + step - 1 for step in joint.SUPERVISION_STEPS]
    rpy_scales = torch.tensor(joint.RPY_SCALES, device=device)
    indices = list(range(len(specs)))
    for scene_index in range(scenes):
        prediction, _ = joint._run_attitude_scene(
            controller,
            cache,
            scene_index,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        target = cache.source_motor[scene_index, label_indices, :3].to(device)
        per_axis_mse = ((prediction - target) / rpy_scales).square().mean(dim=0)
        losses = [per_axis_mse[spec["axis_index"]] / scenes for spec in specs]
        d_first._accumulate_gradients(rows, indices, losses, parameters)
    return rows


def projection_controls_pass(projection: dict[str, Any]) -> bool:
    if projection.get("mode") == "identity_no_active_rpy_rows":
        return projection.get("pass") is True
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
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    denominators: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    current_parameters = joint._copy_parameters(student)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    specs = rpy_constraint_specs(current_metrics)
    print(json.dumps({"stage": "joint_rpy_constraint_jacobian", "rows": len(specs)}), flush=True)
    rows = rpy_constraint_gradient_rows(student, attitude_cache, specs, device=device)
    optimizer.zero_grad(set_to_none=True)
    accumulated_joint_teacher_gradient(
        student,
        factorial_cache,
        scales,
        endpoint_scale,
        denominators,
        device=device,
    )
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [getattr(student, name) for name in joint.PARAMETER_FAMILIES],
        joint.GRADIENT_NORM_CAP,
    )
    optimizer.step()
    raw_parameters = joint._copy_parameters(student)
    raw_displacement = {
        name: raw_parameters[name] - current_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    optimizer_transaction = corrected.optimizer_step_transaction(optimizer_before, optimizer_after)
    if specs:
        proposed, projection = polished.polished_bound_aware_projection(
            raw_displacement, rows, specs, current_parameters["edge_magnitude"]
        )
        projection["mode"] = "active_rpy_polished_fp64"
    else:
        proposed = {
            name: value.detach().double().clone() for name, value in raw_displacement.items()
        }
        projection = {
            "pass": True,
            "mode": "identity_no_active_rpy_rows",
            "rounds": [],
            "constraint_names": [],
            "arithmetic_dtype": "float64",
        }
    projection_control = projection_controls_pass(projection)
    authoritative, effective64, canonicalization = canonical.materialize_authoritative_candidate(
        student, current_parameters, proposed
    )
    parameter_bounds = audit.parameter_bounds_report(authoritative)
    linearized = audit.linearized_constraint_violations(specs, rows, effective64)
    maximum_linearized, linearized_finite = (
        audit.maximum_linearized_violation(linearized) if linearized else (0.0, True)
    )
    raw64 = {name: value.double() for name, value in raw_displacement.items()}
    raw_derivative = canonical._dot_float64(raw_gradients, raw64)
    effective_derivative = canonical._dot_float64(raw_gradients, effective64)
    gradient_finite = math.isfinite(float(gradient_norm)) and all(
        bool(torch.isfinite(value).all()) for value in raw_gradients.values()
    )
    projection.update(
        {
            "pass_after_parameter_bounds": bool(
                projection_control
                and canonicalization["pass"]
                and parameter_bounds["pass"]
                and linearized_finite
                and maximum_linearized <= d_first.LINEAR_CONSTRAINT_TOLERANCE
                and gradient_finite
                and effective_derivative < 0.0
                and optimizer_transaction["pass"]
            ),
            "polished_projection_control_pass": projection_control,
            "authoritative_parameter_canonicalization": canonicalization,
            "authoritative_parameter_bounds": parameter_bounds,
            "bounded_linearized_violation_after": linearized,
            "maximum_bounded_linearized_violation_after": maximum_linearized,
            "raw_j_directional_derivative": raw_derivative,
            "bounded_projected_j_directional_derivative": effective_derivative,
            "j_descent_fraction_retained": (
                effective_derivative / raw_derivative if raw_derivative < 0.0 else None
            ),
            "gradient_norm_before_clipping": float(gradient_norm),
            "gradient_is_finite": gradient_finite,
            "optimizer_transaction": optimizer_transaction,
            "constraint_specs": specs,
            "c_or_p_constraint_rows": 0,
            "rpy_constraint_rows": len(specs),
            "effective_displacement_computed_in_float64": True,
        }
    )
    joint._load_parameters(student, current_parameters)
    return {
        "current_parameters": current_parameters,
        "optimizer_before": optimizer_before,
        "optimizer_after_sha256": audit.semantic_sha256(optimizer_after),
        "authoritative_parameters": authoritative,
        "effective_displacement": {
            name: value.to(current_parameters[name].dtype) for name, value in effective64.items()
        },
        "raw_gradients": raw_gradients,
        "projection": projection,
    }


@torch.no_grad()
def install_authoritative_trial(
    student: ConnectomeController,
    current_parameters: dict[str, Tensor],
    authoritative_parameters: dict[str, Tensor],
    *,
    scale: float,
) -> dict[str, Any]:
    for name in joint.PARAMETER_FAMILIES:
        target = current_parameters[name].double() + scale * (
            authoritative_parameters[name].double() - current_parameters[name].double()
        )
        getattr(student, name).copy_(target.to(current_parameters[name].dtype))
    student.project_parameters()
    first = joint._copy_parameters(student)
    student.project_parameters()
    second_difference = max(
        float((getattr(student, name) - first[name]).abs().max())
        for name in joint.PARAMETER_FAMILIES
    )
    parameters = joint._copy_parameters(student)
    displacement = {
        name: parameters[name].double() - current_parameters[name].double()
        for name in joint.PARAMETER_FAMILIES
    }
    bounds = audit.parameter_bounds_report(parameters)
    return {
        "pass": bool(
            second_difference <= canonical.PARAMETER_IDEMPOTENCE_TOLERANCE and bounds["pass"]
        ),
        "scale": scale,
        "parameters": parameters,
        "displacement": displacement,
        "canonical_parameter_idempotence": {
            "pass": second_difference <= canonical.PARAMETER_IDEMPOTENCE_TOLERANCE,
            "second_direct_projection_maximum_parameter_change": second_difference,
            "limit": canonical.PARAMETER_IDEMPOTENCE_TOLERANCE,
        },
        "parameter_bounds": bounds,
    }


def directional_finite_difference(
    student: ConnectomeController,
    proposal: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    denominators: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    installed = install_authoritative_trial(
        student,
        proposal["current_parameters"],
        proposal["authoritative_parameters"],
        scale=joint.FINITE_DIFFERENCE_SCALE,
    )
    if not installed["pass"]:
        joint._load_parameters(student, proposal["current_parameters"])
        return {
            "pass": False,
            "scale": joint.FINITE_DIFFERENCE_SCALE,
            "reason": "native-bound or canonical-idempotence control failed before replay",
            "canonical_parameter_idempotence": installed["canonical_parameter_idempotence"],
            "parameter_bounds": installed["parameter_bounds"],
            "metrics": None,
            "tests_joint_objective_not_endpoint_d_only": True,
        }
    try:
        metrics, _ = endpoint.evaluate(
            student,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        metrics = attach_joint_objective(metrics, denominators)
    finally:
        joint._load_parameters(student, proposal["current_parameters"])
    expected = (
        canonical._dot_float64(proposal["raw_gradients"], installed["displacement"])
        / joint.FINITE_DIFFERENCE_SCALE
    )
    actual = (
        metrics["joint_teacher_objective"] - current_metrics["joint_teacher_objective"]
    ) / joint.FINITE_DIFFERENCE_SCALE
    relative = abs(actual - expected) / max(abs(actual), abs(expected), 1.0e-12)
    return {
        "pass": bool(
            installed["pass"]
            and math.isfinite(expected)
            and math.isfinite(actual)
            and expected < 0.0
            and actual < 0.0
            and abs(actual) >= audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE
            and relative <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "scale": joint.FINITE_DIFFERENCE_SCALE,
        "autograd_directional_derivative": expected,
        "complete_replay_finite_difference": actual,
        "relative_error": relative,
        "relative_error_limit": joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
        "minimum_absolute_change": (audit.DIRECTIONAL_FINITE_DIFFERENCE_MINIMUM_ABSOLUTE_CHANGE),
        "canonical_parameter_idempotence": installed["canonical_parameter_idempotence"],
        "parameter_bounds": installed["parameter_bounds"],
        "metrics": metrics,
        "tests_joint_objective_not_endpoint_d_only": True,
    }


def candidate_decision(
    current_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    installed: dict[str, Any],
    *,
    optimizer_pending_exact: bool,
) -> dict[str, Any]:
    reasons = []
    improvement = (
        current_metrics["joint_teacher_objective"] - candidate_metrics["joint_teacher_objective"]
    )
    if improvement < MINIMUM_J_IMPROVEMENT:
        reasons.append("joint teacher objective did not improve by at least 1e-4")
    for axis, value in candidate_metrics["rpy_source_nrmse"].items():
        if value > joint.RPY_NRMSE_LIMIT:
            reasons.append(f"{axis} source NRMSE exceeded 0.05")
    if candidate_metrics["motor_output_max_absolute"] > 1.0:
        reasons.append("motor output exceeded [-1, 1]")
    if not candidate_metrics["all_attitude_cache_states_valid"]:
        reasons.append("attitude cache contains an invalid physical state")
    if not joint._numeric_tree_is_finite(candidate_metrics):
        reasons.append("one or more candidate metrics were nonfinite")
    if not installed["pass"]:
        reasons.append("candidate failed native-bound or canonical-idempotence controls")
    if not optimizer_pending_exact:
        reasons.append("pending Adam state changed during candidate evaluation")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "actual_joint_teacher_objective_improvement": improvement,
        "minimum_actual_joint_teacher_objective_improvement": MINIMUM_J_IMPROVEMENT,
        "endpoint_damping_nrmse_change": (
            candidate_metrics["endpoint_damping_nrmse"] - current_metrics["endpoint_damping_nrmse"]
        ),
        "endpoint_damping_may_temporarily_worsen": True,
        "all_metrics_finite": joint._numeric_tree_is_finite(candidate_metrics),
    }


def find_safe_trial(
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    proposal: dict[str, Any],
    current_metrics: dict[str, Any],
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    endpoint_scale: float,
    denominators: dict[str, float],
    *,
    device: torch.device,
) -> tuple[float | None, dict[str, Any] | None, list[dict[str, Any]]]:
    trials = []
    for scale in d_first.BACKTRACK_SCALES:
        installed = install_authoritative_trial(
            student,
            proposal["current_parameters"],
            proposal["authoritative_parameters"],
            scale=scale,
        )
        if not installed["pass"]:
            raise RuntimeError(
                f"candidate numerical control failure at scale {scale}: "
                "native-bound or canonical-idempotence control failed"
            )
        metrics, _ = endpoint.evaluate(
            student,
            factorial_cache,
            attitude_cache,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        metrics = attach_joint_objective(metrics, denominators)
        pending_exact = (
            audit.semantic_sha256(optimizer.state_dict()) == proposal["optimizer_after_sha256"]
        )
        fatal_reasons = []
        if not joint._numeric_tree_is_finite(metrics):
            fatal_reasons.append("candidate metrics were nonfinite")
        if not metrics["all_attitude_cache_states_valid"]:
            fatal_reasons.append("attitude cache contained an invalid physical state")
        if not pending_exact:
            fatal_reasons.append("pending Adam state changed during candidate evaluation")
        if fatal_reasons:
            raise RuntimeError(
                f"candidate numerical control failure at scale {scale}: " + "; ".join(fatal_reasons)
            )
        decision = candidate_decision(
            current_metrics,
            metrics,
            installed,
            optimizer_pending_exact=pending_exact,
        )
        trials.append(
            {
                "scale": scale,
                "acceptance_kind": "ordinary" if decision["pass"] else "ordinary_attempt",
                "metrics": metrics,
                "decision": decision,
                "canonical_parameter_idempotence": installed["canonical_parameter_idempotence"],
                "parameter_bounds": installed["parameter_bounds"],
                "optimizer_pending_state_exact": pending_exact,
                "nonlinear_repair_attempted": False,
            }
        )
        if decision["pass"]:
            return scale, metrics, trials
    joint._load_parameters(student, proposal["current_parameters"])
    return None, None, trials


def source_relative_d_improvement(
    source_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> float:
    source_value = float(source_metrics["endpoint_damping_nrmse"])
    candidate_value = float(candidate_metrics["endpoint_damping_nrmse"])
    if not math.isfinite(source_value) or source_value <= 0.0 or not math.isfinite(candidate_value):
        return math.nan
    return 1.0 - candidate_value / source_value


def midpoint_decision(
    source_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> dict[str, Any]:
    reasons = []
    source_finite = joint._numeric_tree_is_finite(source_metrics)
    candidate_finite = joint._numeric_tree_is_finite(candidate_metrics)
    source_d_positive = bool(
        math.isfinite(float(source_metrics["endpoint_damping_nrmse"]))
        and float(source_metrics["endpoint_damping_nrmse"]) > 0.0
    )
    if not source_finite:
        reasons.append("training source contains one or more nonfinite metrics")
    if not candidate_finite:
        reasons.append("midpoint candidate contains one or more nonfinite metrics")
    if not source_d_positive:
        reasons.append("training source endpoint damping NRMSE is not finite and positive")
    improvement = source_relative_d_improvement(source_metrics, candidate_metrics)
    if not math.isfinite(improvement) or improvement < MIDPOINT_D_IMPROVEMENT_FRACTION:
        reasons.append("endpoint damping source-relative improvement was below 15%")
    for component in ("common", "height"):
        if (
            candidate_metrics["component_nrmse"][component]
            > source_metrics["component_nrmse"][component]
        ):
            reasons.append(f"aggregate {component} NRMSE exceeded the training source")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "update": MIDPOINT_UPDATE,
        "endpoint_damping_source_relative_improvement": improvement,
        "minimum_endpoint_damping_source_relative_improvement": (MIDPOINT_D_IMPROVEMENT_FRACTION),
        "aggregate_common_and_height_compared_to_source": True,
        "source_metrics_finite": source_finite,
        "candidate_metrics_finite": candidate_finite,
        "source_endpoint_damping_nrmse_positive": source_d_positive,
    }


def final_bank_decision(
    source_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    *,
    bank: str,
) -> dict[str, Any]:
    reasons = []
    source_finite = joint._numeric_tree_is_finite(source_metrics)
    candidate_finite = joint._numeric_tree_is_finite(candidate_metrics)
    source_d_positive = bool(
        math.isfinite(float(source_metrics["endpoint_damping_nrmse"]))
        and float(source_metrics["endpoint_damping_nrmse"]) > 0.0
    )
    if not source_finite:
        reasons.append(f"{bank} source contains one or more nonfinite metrics")
    if not source_d_positive:
        reasons.append(f"{bank} source endpoint damping NRMSE is not finite and positive")
    improvement = source_relative_d_improvement(source_metrics, candidate_metrics)
    if not math.isfinite(improvement) or improvement < FINAL_D_IMPROVEMENT_FRACTION:
        reasons.append(f"{bank} endpoint damping source-relative improvement was below 25%")
    sign = candidate_metrics["endpoint_damping_correct_sign_fraction"]
    if sign < FINAL_D_SIGN_FRACTION:
        reasons.append(f"{bank} endpoint damping correct-sign fraction was below 50%")
    for component in ("common", "height"):
        for horizon in map(str, joint.SUPERVISION_STEPS):
            if (
                candidate_metrics["by_supervision_step_nrmse"][horizon][component]
                > source_metrics["by_supervision_step_nrmse"][horizon][component]
            ):
                reasons.append(f"{bank} {component} NRMSE at step {horizon} exceeded source")
    for axis, value in candidate_metrics["rpy_source_nrmse"].items():
        if value > joint.RPY_NRMSE_LIMIT:
            reasons.append(f"{bank} {axis} source NRMSE exceeded 0.05")
    if candidate_metrics["motor_output_max_absolute"] > 1.0:
        reasons.append(f"{bank} motor output exceeded [-1, 1]")
    if not candidate_metrics["all_attitude_cache_states_valid"]:
        reasons.append(f"{bank} attitude cache contains an invalid physical state")
    if not candidate_finite:
        reasons.append(f"{bank} contains one or more nonfinite metrics")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "bank": bank,
        "endpoint_damping_source_relative_improvement": improvement,
        "minimum_endpoint_damping_source_relative_improvement": (FINAL_D_IMPROVEMENT_FRACTION),
        "endpoint_damping_correct_sign_fraction": sign,
        "minimum_endpoint_damping_correct_sign_fraction": FINAL_D_SIGN_FRACTION,
        "per_horizon_common_and_height_compared_to_source": True,
        "source_metrics_finite": source_finite,
        "candidate_metrics_finite": candidate_finite,
        "source_endpoint_damping_nrmse_positive": source_d_positive,
        "all_metrics_finite": source_finite and candidate_finite,
    }


def optimizer_step_counters(optimizer_state: dict[str, Any]) -> list[float]:
    return sorted(
        {
            corrected._optimizer_step(value.get("step", 0.0))
            for value in optimizer_state.get("state", {}).values()
        }
    )


def interrupted_development_record(
    development: dict[str, Any] | None,
    *,
    reason: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    prior = development if isinstance(development, dict) else {}
    result = {
        **prior,
        "pass": False,
        "interrupted": True,
        "retried": False,
        "reason": reason,
        "candidate_evaluation_started_count": int(
            bool(prior.get("candidate_evaluation_started", False))
        ),
        "candidate_evaluation_completed_count": int(
            bool(prior.get("candidate_evaluation_completed", False))
        ),
    }
    if error is not None:
        result.update(
            {
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        )
    return result


def save_resume(
    path: Path,
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint_sha256: str,
    denominators: dict[str, float],
    accepted_updates: int,
    current_training: dict[str, Any],
    history: list[dict[str, Any]],
    milestones: dict[str, Any],
    preflight: dict[str, Any],
    run_state: str,
    stop_reason: str | None = None,
    development: dict[str, Any] | None = None,
) -> None:
    controller_state = student.state_dict()
    controller_parameters = {name: controller_state[name] for name in joint.PARAMETER_FAMILIES}
    optimizer_state = optimizer.state_dict()
    d_first.atomic_torch_save(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "checkpoint_sha256": checkpoint_sha256,
            "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
            "objective_denominators": denominators,
            "objective_denominators_sha256": audit.semantic_sha256(denominators),
            "accepted_updates": accepted_updates,
            "controller": controller_state,
            "controller_parameters_sha256": audit.semantic_sha256(controller_parameters),
            "optimizer": optimizer_state,
            "optimizer_sha256": audit.semantic_sha256(optimizer_state),
            "optimizer_step_counters": optimizer_step_counters(optimizer_state),
            "current_training": current_training,
            "current_training_sha256": audit.semantic_sha256(current_training),
            "history": history,
            "milestones": milestones,
            "preflight": preflight,
            "run_state": run_state,
            "stop_reason": stop_reason,
            "development": development,
        },
        path,
    )


def validate_resume_payload(
    payload: dict[str, Any],
    *,
    checkpoint_sha256: str,
    regenerated_denominators: dict[str, float],
    regenerated_source_metrics: dict[str, Any],
) -> None:
    if (
        payload.get("experiment"),
        payload.get("protocol_commit"),
        payload.get("checkpoint_sha256"),
        payload.get("cache_manifest_sha256"),
    ) != (
        EXPERIMENT,
        PROTOCOL_COMMIT,
        checkpoint_sha256,
        EXPECTED_CACHE_MANIFEST_SHA256,
    ):
        raise SystemExit("joint teacher resume identity does not match")
    persisted_denominators = payload.get("objective_denominators")
    if (
        not isinstance(persisted_denominators, dict)
        or set(persisted_denominators) != {"common", "height", "endpoint_damping"}
        or (
            payload.get("objective_denominators_sha256")
            != audit.semantic_sha256(persisted_denominators)
        )
    ):
        raise SystemExit("joint teacher resume objective denominator hash does not match")
    denominator_replay = audit.numeric_tree_comparison(
        regenerated_denominators, persisted_denominators
    )
    if not denominator_replay["pass"]:
        raise SystemExit("joint teacher resume objective denominators did not replay")
    if not payload.get("preflight", {}).get("pass", False):
        raise SystemExit("joint teacher resume lacks a passing preflight")
    persisted_source_metrics = payload["preflight"].get("source_training_metrics")
    persisted_source = payload["preflight"].get("source_training")
    if (
        not isinstance(persisted_source_metrics, dict)
        or payload["preflight"].get("source_training_metrics_sha256")
        != audit.semantic_sha256(persisted_source_metrics)
        or not isinstance(persisted_source, dict)
        or payload["preflight"].get("source_training_sha256")
        != audit.semantic_sha256(persisted_source)
    ):
        raise SystemExit("joint teacher resume source baseline hash does not match")
    reconstructed_source = attach_joint_objective(persisted_source_metrics, persisted_denominators)
    if not audit.numeric_tree_comparison(reconstructed_source, persisted_source)["pass"]:
        raise SystemExit("joint teacher resume source baseline is internally inconsistent")
    source_replay = audit.numeric_tree_comparison(
        regenerated_source_metrics,
        persisted_source_metrics,
    )
    if not source_replay["pass"]:
        raise SystemExit("joint teacher resume source metrics did not replay")
    accepted = int(payload.get("accepted_updates", -1))
    accepted_entries = [entry for entry in payload.get("history", []) if entry.get("accepted")]
    if [int(entry.get("update", -1)) for entry in accepted_entries] != list(range(1, accepted + 1)):
        raise SystemExit("joint teacher resume accepted history is inconsistent")
    counters = payload.get("optimizer_step_counters", [])
    expected_counters = [] if accepted == 0 else [float(accepted)]
    if counters != expected_counters:
        raise SystemExit("joint teacher resume Adam counters are inconsistent")
    if accepted >= MIDPOINT_UPDATE and not payload.get("milestones", {}).get(
        str(MIDPOINT_UPDATE), {}
    ).get("pass", False):
        if payload.get("run_state") == "active":
            raise SystemExit("active joint teacher resume lacks a passing midpoint gate")
    if accepted > FINAL_UPDATE:
        raise SystemExit("joint teacher resume exceeds its accepted-update budget")
    run_state = payload.get("run_state")
    if run_state not in {"active", "stopped", "development_started", "complete"}:
        raise SystemExit("joint teacher resume has an unknown run state")
    final_milestone = payload.get("milestones", {}).get(str(FINAL_UPDATE), {})
    if run_state in {"development_started", "complete"} and (
        accepted != FINAL_UPDATE or final_milestone.get("pass") is not True
    ):
        raise SystemExit("joint teacher development state lacks a passing final training gate")
    if run_state == "complete" and not payload.get("development", {}).get("pass", False):
        raise SystemExit("complete joint teacher resume lacks a passing development gate")
    controller_state = payload.get("controller")
    optimizer_state = payload.get("optimizer")
    current_training = payload.get("current_training")
    if not isinstance(controller_state, dict) or not all(
        name in controller_state for name in joint.PARAMETER_FAMILIES
    ):
        raise SystemExit("joint teacher resume lacks opened controller parameters")
    controller_parameters = {name: controller_state[name] for name in joint.PARAMETER_FAMILIES}
    if payload.get("controller_parameters_sha256") != audit.semantic_sha256(controller_parameters):
        raise SystemExit("joint teacher resume controller hash does not match")
    if not isinstance(optimizer_state, dict) or payload.get(
        "optimizer_sha256"
    ) != audit.semantic_sha256(optimizer_state):
        raise SystemExit("joint teacher resume optimizer hash does not match")
    if not isinstance(current_training, dict) or payload.get(
        "current_training_sha256"
    ) != audit.semantic_sha256(current_training):
        raise SystemExit("joint teacher resume training-metric hash does not match")


def load_resume(
    path: Path,
    student: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint_sha256: str,
    regenerated_denominators: dict[str, float],
    regenerated_source_metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location=student.edge_magnitude.device, weights_only=True)
    validate_resume_payload(
        payload,
        checkpoint_sha256=checkpoint_sha256,
        regenerated_denominators=regenerated_denominators,
        regenerated_source_metrics=regenerated_source_metrics,
    )
    student.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["optimizer"])
    replayed_steps = optimizer_step_counters(optimizer.state_dict())
    if replayed_steps != payload["optimizer_step_counters"]:
        raise SystemExit("joint teacher resume Adam state did not replay exactly")
    return payload


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    config = HoverConfig()
    responsibility.seed_everything(endpoint.TRAIN_SEED)
    locked_before = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    graph_sha256 = locked_before["graph"]
    checkpoint_sha256 = locked_before["checkpoint"]
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
    if cache_integrity["manifest_sha256"] != EXPECTED_CACHE_MANIFEST_SHA256:
        raise SystemExit("authorized cache manifest hash does not match")
    endpoint_scale = endpoint.endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )
    source_training_raw_metrics, source_training_raw = endpoint.evaluate(
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
    regenerated_denominators = objective_denominators(source_training_raw_metrics)
    resume_path = args.output_dir / "resume.pt"
    resumed = resume_path.is_file()
    if resumed:
        persisted = torch.load(resume_path, map_location="cpu", weights_only=True)
        persisted_denominators = persisted.get("objective_denominators")
        persisted_source_training = persisted.get("preflight", {}).get("source_training")
        if not isinstance(persisted_denominators, dict) or set(persisted_denominators) != {
            "common",
            "height",
            "endpoint_damping",
        }:
            raise SystemExit("joint teacher resume lacks persisted objective denominators")
        if not isinstance(persisted_source_training, dict):
            raise SystemExit("joint teacher resume lacks its persisted source baseline")
        denominators = {name: float(value) for name, value in persisted_denominators.items()}
        source_training = copy.deepcopy(persisted_source_training)
    else:
        denominators = regenerated_denominators
        source_training = attach_joint_objective(source_training_raw_metrics, denominators)
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
    fresh_adam_state_empty = not optimizer.state_dict()["state"]
    preflight = {
        "pass": bool(
            cache_integrity["all_persisted_file_and_tensor_hashes_match"]
            and teacher_identity["pass"]
            and teacher_positive["pass"]
            and source_replay_difference <= joint.REPLAY_TOLERANCE
            and train_factorial.endpoint_image_difference_max == 0.0
            and development_factorial.endpoint_image_difference_max == 0.0
            and source_training["all_attitude_cache_states_valid"]
            and fresh_adam_state_empty
            and joint._numeric_tree_is_finite(source_training)
            and source_training["endpoint_damping_nrmse"] > 0.0
            and all(math.isfinite(value) and value > 0.0 for value in denominators.values())
        ),
        "cache_integrity_pass": cache_integrity["all_persisted_file_and_tensor_hashes_match"],
        "teacher_factorial_identity_pass": teacher_identity["pass"],
        "teacher_foreleg_stick_positive_control_pass": teacher_positive["pass"],
        "source_replay_max_absolute_difference": source_replay_difference,
        "source_replay_tolerance": joint.REPLAY_TOLERANCE,
        "opposite_motion_endpoint_images_exact": (
            train_factorial.endpoint_image_difference_max == 0.0
            and development_factorial.endpoint_image_difference_max == 0.0
        ),
        "objective_denominators": denominators,
        "objective_denominators_sha256": audit.semantic_sha256(denominators),
        "source_training_metrics": source_training_raw_metrics,
        "source_training_metrics_sha256": audit.semantic_sha256(source_training_raw_metrics),
        "source_training": source_training,
        "source_training_sha256": audit.semantic_sha256(source_training),
        "fresh_adam_state_empty": fresh_adam_state_empty,
        "accepted_update_counter_initial": 0,
        "development_candidate_evaluated": False,
        "source_training_metrics_finite": joint._numeric_tree_is_finite(source_training),
        "source_endpoint_damping_nrmse_positive": (source_training["endpoint_damping_nrmse"] > 0.0),
        "objective_denominators_finite_and_positive": all(
            math.isfinite(value) and value > 0.0 for value in denominators.values()
        ),
    }
    if not preflight["pass"]:
        raise SystemExit("joint teacher-learning source preflight failed")
    accepted_updates = 0
    current_training = copy.deepcopy(source_training)
    history: list[dict[str, Any]] = []
    milestones: dict[str, Any] = {}
    development: dict[str, Any] | None = None
    stop_reason: str | None = None
    run_state = "active"
    if resumed:
        resume = load_resume(
            resume_path,
            student,
            optimizer,
            checkpoint_sha256=checkpoint_sha256,
            regenerated_denominators=regenerated_denominators,
            regenerated_source_metrics=source_training_raw_metrics,
        )
        preflight = copy.deepcopy(resume["preflight"])
        source_training = copy.deepcopy(preflight["source_training"])
        accepted_updates = int(resume["accepted_updates"])
        history = list(resume["history"])
        milestones = dict(resume["milestones"])
        development = resume.get("development")
        run_state = str(resume["run_state"])
        stop_reason = resume.get("stop_reason")
        replayed_training_raw, _ = endpoint.evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        replayed_training = attach_joint_objective(replayed_training_raw, denominators)
        replay = audit.numeric_tree_comparison(replayed_training, resume["current_training"])
        if not replay["pass"]:
            raise SystemExit("joint teacher resume training metrics did not replay")
        current_training = copy.deepcopy(resume["current_training"])
        if run_state == "development_started":
            stop_reason = "development evaluation was interrupted and cannot be retried"
            run_state = "stopped"
            development = interrupted_development_record(development, reason=stop_reason)
            save_resume(
                resume_path,
                student,
                optimizer,
                checkpoint_sha256=checkpoint_sha256,
                denominators=denominators,
                accepted_updates=accepted_updates,
                current_training=current_training,
                history=history,
                milestones=milestones,
                preflight=preflight,
                run_state=run_state,
                stop_reason=stop_reason,
                development=development,
            )
        elif run_state != "active":
            if stop_reason is None:
                raise SystemExit("terminal joint teacher resume lacks a stop reason")

    if not resumed:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_resume(
            resume_path,
            student,
            optimizer,
            checkpoint_sha256=checkpoint_sha256,
            denominators=denominators,
            accepted_updates=accepted_updates,
            current_training=current_training,
            history=history,
            milestones=milestones,
            preflight=preflight,
            run_state="active",
        )

    while run_state == "active" and accepted_updates < FINAL_UPDATE:
        update_number = accepted_updates + 1
        current_parameters = joint._copy_parameters(student)
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        proposal = None
        finite_difference = None
        try:
            proposal = make_projected_proposal(
                student,
                optimizer,
                current_training,
                train_factorial,
                train_attitude,
                scales,
                endpoint_scale,
                denominators,
                device=device,
            )
            projection = proposal["projection"]
            if not projection["pass_after_parameter_bounds"]:
                raise RuntimeError("joint J proposal projection controls failed")
            finite_difference = directional_finite_difference(
                student,
                proposal,
                current_training,
                train_factorial,
                train_attitude,
                scales,
                endpoint_scale,
                denominators,
                device=device,
            )
            if not finite_difference["pass"]:
                raise RuntimeError("joint J directional finite-difference controls failed")
            selected_scale, selected_metrics, trials = find_safe_trial(
                student,
                optimizer,
                proposal,
                current_training,
                train_factorial,
                train_attitude,
                scales,
                endpoint_scale,
                denominators,
                device=device,
            )
        except Exception as error:
            joint._load_parameters(student, current_parameters)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            stop_reason = f"numerical control failure during update {update_number}"
            run_state = "stopped"
            entry: dict[str, Any] = {
                "update": update_number,
                "accepted": False,
                "stop_reason": stop_reason,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "trials": [],
            }
            if proposal is not None:
                entry["projection"] = proposal.get("projection")
            if finite_difference is not None:
                entry["finite_difference"] = finite_difference
            history.append(entry)
            save_resume(
                resume_path,
                student,
                optimizer,
                checkpoint_sha256=checkpoint_sha256,
                denominators=denominators,
                accepted_updates=accepted_updates,
                current_training=current_training,
                history=history,
                milestones=milestones,
                preflight=preflight,
                run_state=run_state,
                stop_reason=stop_reason,
            )
            break
        entry = {
            "update": update_number,
            "accepted": selected_scale is not None,
            "accepted_scale": selected_scale,
            "acceptance_kind": "ordinary" if selected_scale is not None else None,
            "projection": projection,
            "finite_difference": finite_difference,
            "trials": trials,
        }
        history.append(entry)
        if selected_scale is None or selected_metrics is None:
            joint._load_parameters(student, current_parameters)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            stop_reason = "deterministic joint J proposal had no admissible scale"
            run_state = "stopped"
            entry["stop_reason"] = stop_reason
            save_resume(
                resume_path,
                student,
                optimizer,
                checkpoint_sha256=checkpoint_sha256,
                denominators=denominators,
                accepted_updates=accepted_updates,
                current_training=current_training,
                history=history,
                milestones=milestones,
                preflight=preflight,
                run_state=run_state,
                stop_reason=stop_reason,
            )
            break
        accepted_updates = update_number
        current_training = selected_metrics
        milestone = None
        if accepted_updates == MIDPOINT_UPDATE:
            milestone = midpoint_decision(source_training, current_training)
            milestones[str(MIDPOINT_UPDATE)] = milestone
            if not milestone["pass"]:
                stop_reason = "mandatory update-25 joint teacher gate failed"
                run_state = "stopped"
        if accepted_updates == FINAL_UPDATE:
            milestone = final_bank_decision(source_training, current_training, bank="training")
            milestones[str(FINAL_UPDATE)] = milestone
            if not milestone["pass"]:
                stop_reason = "mandatory update-50 training gate failed"
                run_state = "stopped"
        save_resume(
            resume_path,
            student,
            optimizer,
            checkpoint_sha256=checkpoint_sha256,
            denominators=denominators,
            accepted_updates=accepted_updates,
            current_training=current_training,
            history=history,
            milestones=milestones,
            preflight=preflight,
            run_state=run_state,
            stop_reason=stop_reason,
        )
        print(
            json.dumps(
                {
                    "progress": "accepted_update",
                    "update": accepted_updates,
                    "scale": selected_scale,
                    "joint_teacher_objective": current_training["joint_teacher_objective"],
                    "endpoint_damping_nrmse": current_training["endpoint_damping_nrmse"],
                    "endpoint_damping_sign": current_training[
                        "endpoint_damping_correct_sign_fraction"
                    ],
                    "milestone_pass": milestone["pass"] if milestone else None,
                }
            ),
            flush=True,
        )
        if run_state != "active":
            break

    if (
        run_state == "active"
        and accepted_updates == FINAL_UPDATE
        and milestones[str(FINAL_UPDATE)]["pass"]
    ):
        run_state = "development_started"
        development = {
            "pass": False,
            "source_evaluation_started": True,
            "source_evaluation_completed": False,
            "candidate_evaluation_started": False,
            "candidate_evaluation_completed": False,
            "candidate_evaluation_started_count": 0,
            "candidate_evaluation_completed_count": 0,
            "interrupted": False,
            "retried": False,
        }
        save_resume(
            resume_path,
            student,
            optimizer,
            checkpoint_sha256=checkpoint_sha256,
            denominators=denominators,
            accepted_updates=accepted_updates,
            current_training=current_training,
            history=history,
            milestones=milestones,
            preflight=preflight,
            run_state=run_state,
            development=development,
        )
        try:
            source_development_raw, _ = endpoint.evaluate(
                source,
                development_factorial,
                development_attitude,
                scales,
                endpoint_scale=endpoint_scale,
                device=device,
            )
            source_development = attach_joint_objective(source_development_raw, denominators)
            development.update(
                {
                    "source_evaluation_completed": True,
                    "source": source_development,
                    "candidate_evaluation_started": True,
                    "candidate_evaluation_started_count": 1,
                }
            )
            save_resume(
                resume_path,
                student,
                optimizer,
                checkpoint_sha256=checkpoint_sha256,
                denominators=denominators,
                accepted_updates=accepted_updates,
                current_training=current_training,
                history=history,
                milestones=milestones,
                preflight=preflight,
                run_state=run_state,
                development=development,
            )
            candidate_development_raw, _ = endpoint.evaluate(
                student,
                development_factorial,
                development_attitude,
                scales,
                endpoint_scale=endpoint_scale,
                device=device,
            )
            development.update(
                {
                    "candidate_evaluation_completed": True,
                    "candidate_evaluation_completed_count": 1,
                }
            )
            candidate_development = attach_joint_objective(candidate_development_raw, denominators)
            development_decision = final_bank_decision(
                source_development, candidate_development, bank="development"
            )
            development = {
                **development,
                "pass": development_decision["pass"],
                "candidate": candidate_development,
                "decision": development_decision,
                "interrupted": False,
                "retried": False,
            }
            if development_decision["pass"]:
                run_state = "complete"
                stop_reason = "joint teacher-learning training and development gates passed"
            else:
                run_state = "stopped"
                stop_reason = "mandatory update-50 development gate failed"
        except Exception as error:
            run_state = "stopped"
            stop_reason = "development evaluation failed and cannot be retried"
            development = interrupted_development_record(
                development, reason=stop_reason, error=error
            )
        save_resume(
            resume_path,
            student,
            optimizer,
            checkpoint_sha256=checkpoint_sha256,
            denominators=denominators,
            accepted_updates=accepted_updates,
            current_training=current_training,
            history=history,
            milestones=milestones,
            preflight=preflight,
            run_state=run_state,
            stop_reason=stop_reason,
            development=development,
        )

    locked_after = {
        name: responsibility.file_sha256(path) for name, path in locked_input_paths(args).items()
    }
    inputs_unchanged = locked_after == locked_before
    passed = bool(
        run_state == "complete"
        and milestones.get(str(FINAL_UPDATE), {}).get("pass", False)
        and development
        and development.get("pass", False)
        and inputs_unchanged
    )
    if not inputs_unchanged:
        stop_reason = "one or more locked inputs changed during the experiment"
    if passed:
        classification = "joint_teacher_learning_qualified"
    elif not inputs_unchanged:
        classification = "locked_input_changed"
    elif stop_reason == "mandatory update-25 joint teacher gate failed":
        classification = "midpoint_gate_failed"
    elif stop_reason == "mandatory update-50 training gate failed":
        classification = "final_training_gate_failed"
    elif stop_reason == "mandatory update-50 development gate failed":
        classification = "development_gate_failed"
    elif development and development.get("interrupted"):
        classification = "development_interrupted"
    else:
        classification = "no_qualified_joint_teacher_candidate"
    final_distance = {
        name: float(
            (getattr(student, name).detach() - source_parameters[name]).square().mean().sqrt()
        )
        for name in joint.PARAMETER_FAMILIES
    }
    report = {
        "experiment": EXPERIMENT,
        "status": "nonpromotional_joint_teacher_learning_diagnostic",
        "pass": passed,
        "classification": classification,
        "protocol": protocol_manifest(),
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "locked_hashes_before": locked_before,
        "locked_hashes_after": locked_after,
        "all_locked_inputs_unchanged": inputs_unchanged,
        "actor_contract_unchanged": True,
        "cache_integrity": cache_integrity,
        "teacher_component_scales_motor_units": scales,
        "endpoint_damping_scale_motor_units": endpoint_scale,
        "objective_denominators": denominators,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "preflight": preflight,
        "resumed": resumed,
        "resume_state": responsibility.stable_path(resume_path),
        "resume_state_sha256": responsibility.file_sha256(resume_path),
        "accepted_updates": accepted_updates,
        "optimizer_step_counters": optimizer_step_counters(optimizer.state_dict()),
        "run_state": run_state,
        "stop_reason": stop_reason,
        "source_training": source_training,
        "final_training": current_training,
        "history": history,
        "milestones": milestones,
        "development": development,
        "development_candidate_evaluations": (
            development.get("candidate_evaluation_completed_count", 0) if development else 0
        ),
        "development_candidate_evaluations_started": (
            development.get("candidate_evaluation_started_count", 0) if development else 0
        ),
        "subsequent_learning_experiment_authorized": passed,
        "closed_loop_hover_authorized": False,
        "closed_loop_hover_run": False,
        "gate_flight_authorized": False,
        "promoted": False,
        "final_parameter_family_rms_from_source": final_distance,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass qualifies only this joint teacher-learning configuration and does not "
            "establish hover, flight, or a causal attribution to prior source ceilings."
        ),
    }
    output = args.output_dir / "report.json"
    audit25.write_json(output, report)
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": passed,
                "classification": classification,
                "accepted_updates": accepted_updates,
                "stop_reason": stop_reason,
                "development_candidate_evaluations": report["development_candidate_evaluations"],
                "subsequent_learning_experiment_authorized": passed,
                "closed_loop_hover_authorized": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
