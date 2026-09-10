#!/usr/bin/env python3
"""Audit step-size and parameter-family responses of the failed full-native update."""

from __future__ import annotations

import argparse
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

import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-step-family-audit-v1"
FULL_SCALES = (1.0, 0.5, 0.25, 0.125)
FAMILY_PROBES = joint.PARAMETER_FAMILIES
MINIMUM_D_NRMSE_IMPROVEMENT = 1.0e-4
REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-3
REPRODUCTION_ABSOLUTE_TOLERANCE = 1.0e-7
EXPECTED_CACHE_HASHES = {
    "training_factorial": "36cfc6b263d2603d7db71e419a565a8332dfac562f276f2eab0fed98fcdd4d62",
    "training_attitude": "8c90e4e200d36968f21011c061323480a90b78203e9b95de365897c321608cbd",
    "development_factorial": "6a390429a054f60f83b36adf394adcf3a237f1c1275a0d57b0fb9d59a7f6d8ff",
    "development_attitude": "ba6e59873bd4ede816b7cf478a6921eca8f83149c382cec07731df87aabe5115",
}
EXPECTED_SCALARS = {
    "baseline_training_joint_mse": 1.0406124591827393,
    "full_training_joint_mse": 0.3528713285923004,
    "autograd_directional_derivative": -1.1403942108154297,
    "edge_magnitude_rms": 0.00009709275036584586,
    "bias_rms": 0.00009616908937459812,
    "raw_time_constant_rms": 0.0000008837100722303148,
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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/full-native-step-family-audit-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "reused_preflight_train_seed": joint.TRAIN_SEED,
        "reused_preflight_development_seed": joint.DEVELOPMENT_SEED,
        "scenes_per_bank": joint.SCENES_PER_BANK,
        "full_displacement_scales_descending": list(FULL_SCALES),
        "largest_training_feasible_full_scale_selected": True,
        "development_candidates_evaluated": 1,
        "no_smaller_scale_after_development_failure": True,
        "single_family_scale": 1.0,
        "single_family_probes": list(FAMILY_PROBES),
        "single_family_probes_are_diagnostic_only": True,
        "family_effects_assumed_additive": False,
        "minimum_measurable_d_nrmse_improvement": MINIMUM_D_NRMSE_IMPROVEMENT,
        "multi_step_fitting_requires_measurable_d_on_both_banks": True,
        "reproduction_relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "reuses_full_native_joint_protocol": joint.protocol_manifest(
            argparse.Namespace(
                train_seed=joint.TRAIN_SEED,
                development_seed=joint.DEVELOPMENT_SEED,
                scenes=joint.SCENES_PER_BANK,
                prefix_steps=joint.PREFIX_STEPS,
                response_steps=joint.RESPONSE_STEPS,
                policy_hz=joint.POLICY_HZ,
                smoke_test=False,
            )
        ),
        "smoke_test": args.smoke_test,
    }


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.smoke_test:
        raise SystemExit("this audit reuses exact eight-scene caches and has no smoke variant")


def scale_decision(
    baseline: dict[str, Any], candidate: dict[str, Any], *, development: bool
) -> dict[str, Any]:
    reasons = []
    improvement = baseline["joint_normalized_mse"] - candidate["joint_normalized_mse"]
    minimum = 0.0 if development else joint.MINIMUM_TRAIN_OBJECTIVE_IMPROVEMENT
    failed_improvement = improvement <= minimum if development else improvement < minimum
    if failed_improvement:
        suffix = "positive" if development else "at least 1e-4"
        reasons.append(f"joint normalized MSE improvement was not {suffix}")
    for name in ("common", "height", "damping"):
        if (
            candidate["component_nrmse"][name]
            > baseline["component_nrmse"][name] + joint.COMPONENT_BASELINE_TOLERANCE
        ):
            reasons.append(f"{name} NRMSE exceeded source plus 0.02")
    for name, value in candidate["rpy_source_nrmse"].items():
        if value > joint.RPY_NRMSE_LIMIT:
            reasons.append(f"{name} source NRMSE exceeded 0.05")
    if candidate["motor_output_max_absolute"] > 1.0:
        reasons.append("motor output exceeded [-1, 1]")
    if not candidate["all_attitude_cache_states_valid"]:
        reasons.append("attitude cache contains an invalid physical state")
    finite = joint._numeric_tree_is_finite(candidate)
    if not finite:
        reasons.append("one or more candidate metrics were nonfinite")
    d_improvement = baseline["component_nrmse"]["damping"] - candidate["component_nrmse"]["damping"]
    return {
        "pass": not reasons,
        "reasons": reasons,
        "joint_normalized_mse_improvement": improvement,
        "damping_nrmse_improvement": d_improvement,
        "measurable_damping_error_reduction": (d_improvement >= MINIMUM_D_NRMSE_IMPROVEMENT),
        "all_metrics_finite": finite,
    }


def largest_feasible_scale(trials: list[dict[str, Any]]) -> float | None:
    for trial in trials:
        if trial["decision"]["pass"]:
            return float(trial["scale"])
    return None


@torch.no_grad()
def set_displacement(
    controller: ConnectomeController,
    source: dict[str, Tensor],
    displacement: dict[str, Tensor],
    *,
    scale: float,
    families: tuple[str, ...] = joint.PARAMETER_FAMILIES,
) -> dict[str, float]:
    selected = set(families)
    for name in joint.PARAMETER_FAMILIES:
        value = source[name]
        if name in selected:
            value = value + scale * displacement[name]
        getattr(controller, name).copy_(value)
    controller.project_parameters()
    return {
        name: float((getattr(controller, name) - source[name]).square().mean().sqrt())
        for name in joint.PARAMETER_FAMILIES
    }


def _close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=REPRODUCTION_RELATIVE_TOLERANCE,
        abs_tol=REPRODUCTION_ABSOLUTE_TOLERANCE,
    )


def reproduction_report(
    *,
    cache_hashes: dict[str, str],
    baseline: dict[str, Any],
    full_candidate: dict[str, Any],
    derivative: float,
    displacement_rms: dict[str, float],
) -> dict[str, Any]:
    scalar_actual = {
        "baseline_training_joint_mse": baseline["joint_normalized_mse"],
        "full_training_joint_mse": full_candidate["joint_normalized_mse"],
        "autograd_directional_derivative": derivative,
        "edge_magnitude_rms": displacement_rms["edge_magnitude"],
        "bias_rms": displacement_rms["bias"],
        "raw_time_constant_rms": displacement_rms["raw_time_constant"],
    }
    hashes = {
        name: {
            "actual": value,
            "expected": EXPECTED_CACHE_HASHES[name],
            "pass": value == EXPECTED_CACHE_HASHES[name],
        }
        for name, value in cache_hashes.items()
    }
    scalars = {
        name: {
            "actual": value,
            "expected": EXPECTED_SCALARS[name],
            "pass": _close(value, EXPECTED_SCALARS[name]),
        }
        for name, value in scalar_actual.items()
    }
    return {
        "pass": all(item["pass"] for item in (*hashes.values(), *scalars.values())),
        "cache_hashes": hashes,
        "scalars": scalars,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    started = perf_counter()
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    graph_sha256 = responsibility.file_sha256(args.graph)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / joint.POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    student.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    source_parameters = joint._copy_parameters(student)

    print(json.dumps({"stage": "rebuilding_frozen_caches"}), flush=True)
    train_factorial = joint.build_factorial_cache(
        scenes=joint.SCENES_PER_BANK,
        seed=joint.TRAIN_SEED,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    train_attitude = joint.build_attitude_cache(
        source,
        scenes=joint.SCENES_PER_BANK,
        seed=joint.TRAIN_SEED + 1,
        prefix_steps=joint.PREFIX_STEPS,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    development_factorial = joint.build_factorial_cache(
        scenes=joint.SCENES_PER_BANK,
        seed=joint.DEVELOPMENT_SEED,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    development_attitude = joint.build_attitude_cache(
        source,
        scenes=joint.SCENES_PER_BANK,
        seed=joint.DEVELOPMENT_SEED + 1,
        prefix_steps=joint.PREFIX_STEPS,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    cache_hashes = {
        "training_factorial": train_factorial.sha256,
        "training_attitude": train_attitude.sha256,
        "development_factorial": development_factorial.sha256,
        "development_attitude": development_attitude.sha256,
    }
    scales = joint.training_teacher_scales(train_factorial)
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )

    print(json.dumps({"stage": "replaying_source"}), flush=True)
    baseline_train, baseline_raw = joint.evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    _, replay_raw = joint.evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    baseline_development, _ = joint.evaluate_bank(
        student,
        development_factorial,
        development_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    replay_difference = joint._max_prediction_difference(baseline_raw, replay_raw)

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
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"stage": "regenerating_full_displacement"}), flush=True)
    joint.accumulated_gradient(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in joint.PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    optimizer.step()
    student.project_parameters()
    full_parameters = joint._copy_parameters(student)
    displacement = {
        name: full_parameters[name] - source_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    displacement_rms = {
        name: float(value.square().mean().sqrt()) for name, value in displacement.items()
    }
    derivative = float(
        sum((raw_gradients[name] * displacement[name]).sum() for name in joint.PARAMETER_FAMILIES)
    )
    full_candidate_train, _ = joint.evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )

    set_displacement(
        student,
        source_parameters,
        displacement,
        scale=joint.FINITE_DIFFERENCE_SCALE,
    )
    finite_difference_train, _ = joint.evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    finite_difference = (
        finite_difference_train["joint_normalized_mse"] - baseline_train["joint_normalized_mse"]
    ) / joint.FINITE_DIFFERENCE_SCALE
    derivative_relative_error = abs(finite_difference - derivative) / max(
        abs(finite_difference), abs(derivative), 1.0e-12
    )
    directional = {
        "pass": (
            derivative < 0.0
            and finite_difference < 0.0
            and derivative_relative_error <= joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "autograd_directional_derivative": derivative,
        "forward_finite_difference": finite_difference,
        "relative_error": derivative_relative_error,
        "scale": joint.FINITE_DIFFERENCE_SCALE,
    }
    reproduction = reproduction_report(
        cache_hashes=cache_hashes,
        baseline=baseline_train,
        full_candidate=full_candidate_train,
        derivative=derivative,
        displacement_rms=displacement_rms,
    )

    print(json.dumps({"stage": "training_scale_sweep"}), flush=True)
    scale_trials = []
    for scale in FULL_SCALES:
        if scale == 1.0:
            metrics = full_candidate_train
            actual_rms = displacement_rms
        else:
            actual_rms = set_displacement(student, source_parameters, displacement, scale=scale)
            metrics, _ = joint.evaluate_bank(
                student,
                train_factorial,
                train_attitude,
                scales,
                prefix_steps=joint.PREFIX_STEPS,
                device=device,
            )
        scale_trials.append(
            {
                "scale": scale,
                "metrics": metrics,
                "actual_displacement_family_rms": actual_rms,
                "decision": scale_decision(baseline_train, metrics, development=False),
            }
        )
    selected_scale = largest_feasible_scale(scale_trials)

    print(json.dumps({"stage": "single_family_probes"}), flush=True)
    family_probes = {}
    for family in FAMILY_PROBES:
        actual_rms = set_displacement(
            student,
            source_parameters,
            displacement,
            scale=1.0,
            families=(family,),
        )
        metrics, _ = joint.evaluate_bank(
            student,
            train_factorial,
            train_attitude,
            scales,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        family_probes[family] = {
            "metrics": metrics,
            "actual_displacement_family_rms": actual_rms,
            "decision_diagnostic_only": scale_decision(baseline_train, metrics, development=False),
        }

    development_trial = None
    if selected_scale is not None:
        set_displacement(
            student,
            source_parameters,
            displacement,
            scale=selected_scale,
        )
        metrics, _ = joint.evaluate_bank(
            student,
            development_factorial,
            development_attitude,
            scales,
            prefix_steps=joint.PREFIX_STEPS,
            device=device,
        )
        development_trial = {
            "scale": selected_scale,
            "metrics": metrics,
            "decision": scale_decision(baseline_development, metrics, development=True),
        }

    joint._load_parameters(student, source_parameters)
    restored = all(
        torch.equal(getattr(student, name).detach(), source_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    controls_pass = (
        reproduction["pass"]
        and directional["pass"]
        and teacher_identity["pass"]
        and teacher_positive["pass"]
        and replay_difference <= joint.REPLAY_TOLERANCE
        and train_factorial.endpoint_image_difference_max == 0.0
        and development_factorial.endpoint_image_difference_max == 0.0
        and restored
    )
    passed = bool(
        controls_pass
        and selected_scale is not None
        and development_trial is not None
        and development_trial["decision"]["pass"]
    )
    selected_training = next(
        (trial for trial in scale_trials if trial["scale"] == selected_scale), None
    )
    measurable_damping = bool(
        passed
        and selected_training is not None
        and selected_training["decision"]["measurable_damping_error_reduction"]
        and development_trial is not None
        and development_trial["decision"]["measurable_damping_error_reduction"]
    )
    if passed:
        classification = (
            "safe_scaled_step_with_measurable_damping_error_reduction"
            if measurable_damping
            else "safe_scaled_step_collective_calibration_only"
        )
    elif selected_scale is None:
        classification = "no_training_feasible_full_scale"
    elif not controls_pass:
        classification = "audit_control_failure"
    else:
        classification = "training_selected_scale_failed_development"

    report = {
        "experiment": EXPERIMENT,
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": passed,
        "classification": classification,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        },
        "actor_contract_unchanged": True,
        "protocol": protocol_manifest(args),
        "cache_hashes": cache_hashes,
        "teacher_component_scales_motor_units": scales,
        "reproduction": reproduction,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "deterministic_source_replay_max_absolute_difference": replay_difference,
        "directional_derivative": directional,
        "gradient_norm_before_clipping": float(gradient_norm),
        "full_displacement_family_rms": displacement_rms,
        "baseline_training": baseline_train,
        "training_scale_trials": scale_trials,
        "selected_training_scale": selected_scale,
        "single_family_training_probes_diagnostic_only": family_probes,
        "baseline_development": baseline_development,
        "selected_development_trial": development_trial,
        "controls_pass": controls_pass,
        "measurable_damping_error_reduction_on_both_banks": measurable_damping,
        "multi_step_joint_fitting_authorized": measurable_damping,
        "parameters_restored_exactly": restored,
        "promoted": False,
        "closed_loop_hover_run": False,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass identifies step-size overshoot for one regenerated Adam update. "
            "Family probes are nonselective diagnostics, and no result establishes "
            "closed-loop hover or absence of longer-term task conflict."
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
                "selected_training_scale": selected_scale,
                "development_pass": (
                    development_trial["decision"]["pass"] if development_trial is not None else None
                ),
                "measurable_damping_error_reduction_on_both_banks": measurable_damping,
                "multi_step_joint_fitting_authorized": measurable_damping,
                "parameters_restored_exactly": restored,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
