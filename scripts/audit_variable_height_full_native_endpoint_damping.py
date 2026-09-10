#!/usr/bin/env python3
"""Audit one full-native Adam step trained only on literal endpoint damping."""

from __future__ import annotations

import argparse
import json
import os
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
import audit_variable_height_full_native_step_response as step_audit  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-endpoint-damping-step-audit-v1"
PROTOCOL_COMMIT = "87ef766"
TRAIN_SEED = 340_961
DEVELOPMENT_SEED = 350_961
FULL_SCALES = step_audit.FULL_SCALES
MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT = 0.001
CACHE_SCHEMA = 1
CACHE_NAMES = (
    "training_factorial",
    "training_attitude",
    "development_factorial",
    "development_attitude",
)
FACTORIAL_TENSOR_FIELDS = (
    "prefix_images",
    "response_images",
    "attitude",
    "teacher_motor",
    "teacher_rc",
    "height_error",
    "vertical_speed",
    "approach_steps",
    "camera_height",
)
ATTITUDE_TENSOR_FIELDS = (
    "images",
    "attitudes",
    "source_motor",
    "valid",
    "camera_height",
    "marker_height",
)


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
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-endpoint-damping-step-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.smoke_test:
        raise SystemExit("this preregistered audit has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "train_seed": TRAIN_SEED,
        "train_attitude_seed": TRAIN_SEED + 1,
        "development_seed": DEVELOPMENT_SEED,
        "development_attitude_seed": DEVELOPMENT_SEED + 1,
        "scenes_per_bank": joint.SCENES_PER_BANK,
        "prefix_steps": joint.PREFIX_STEPS,
        "response_steps": joint.RESPONSE_STEPS,
        "supervision_steps_one_indexed": list(joint.SUPERVISION_STEPS),
        "policy_hz": joint.POLICY_HZ,
        "cache_schema": CACHE_SCHEMA,
        "cache_policy": (
            "generate once, persist, hash, reload, and reuse the exact tensors for every "
            "source replay, gradient, finite difference, scale trial, and development trial"
        ),
        "independent_cuda_regeneration_requires_exact_hash_match": False,
        "source_replay_tolerance": joint.REPLAY_TOLERANCE,
        "native_state_initialization": "zero",
        "prefix_and_response_are_differentiated": True,
        "actor_inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
        "privileged_actor_inputs": [],
        "opened_parameter_families": list(joint.PARAMETER_FAMILIES),
        "frozen": [
            "topology",
            "transmitter signs",
            "retinal mapping",
            "attitude mapping",
            "sensory gains",
            "actor inputs",
            "foreleg output pools",
        ],
        "objective": "normalized endpoint damping MSE only",
        "endpoint_damping_horizon_one_indexed": joint.RESPONSE_STEPS,
        "endpoint_opposite_motion_pose_and_image_must_match_exactly": True,
        "endpoint_damping_scale": (
            "training endpoint teacher RMS floored at 0.01 motor units and frozen for "
            "development"
        ),
        "optimizer": {
            "name": "Adam",
            "edge_and_bias_learning_rate": joint.EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": joint.TIME_CONSTANT_LEARNING_RATE,
            "weight_decay": 0.0,
            "global_gradient_norm_cap": joint.GRADIENT_NORM_CAP,
        },
        "finite_difference_scale": joint.FINITE_DIFFERENCE_SCALE,
        "finite_difference_relative_error_limit": (
            joint.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "complete_displacement_scales_descending": list(FULL_SCALES),
        "largest_training_feasible_scale_selected": True,
        "development_candidates_evaluated": 1,
        "no_smaller_scale_after_development_failure": True,
        "minimum_endpoint_damping_nrmse_improvement": (
            MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT
        ),
        "preservation": {
            "aggregate_common_and_height_source_plus": joint.COMPONENT_BASELINE_TOLERANCE,
            "each_horizon_common_and_height_source_plus": (
                joint.COMPONENT_BASELINE_TOLERANCE
            ),
            "rpy_source_nrmse_maximum": joint.RPY_NRMSE_LIMIT,
            "motor_output_maximum_absolute": 1.0,
            "finite_and_valid": True,
        },
        "joint_loss_and_interaction_are_reporting_only": True,
        "parameters_restored_exactly": True,
        "candidate_retained": False,
    }


def endpoint_damping_scale(cache: joint.FactorialCache) -> float:
    indices = [step - 1 for step in joint.SUPERVISION_STEPS]
    throttle = cache.teacher_motor[:, indices, :, 3]
    branch_major = throttle.permute(2, 0, 1).reshape(4, -1)
    endpoint = joint.factorial.factorial_components(branch_major)["damping"].reshape(
        cache.teacher_motor.shape[0], len(indices)
    )[:, -1]
    return max(float(endpoint.square().mean().sqrt()), joint.TEACHER_SCALE_FLOOR)


def _factorial_payload(cache: joint.FactorialCache) -> dict[str, Any]:
    return {
        **{name: getattr(cache, name) for name in FACTORIAL_TENSOR_FIELDS},
        "endpoint_image_difference_max": cache.endpoint_image_difference_max,
        "sha256": cache.sha256,
    }


def _attitude_payload(cache: joint.AttitudeCache) -> dict[str, Any]:
    return {
        **{name: getattr(cache, name) for name in ATTITUDE_TENSOR_FIELDS},
        "sha256": cache.sha256,
    }


def _load_factorial(path: Path) -> joint.FactorialCache:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cache = joint.FactorialCache(**payload)
    actual = joint._hash_tensors({name: getattr(cache, name) for name in FACTORIAL_TENSOR_FIELDS})
    if actual != cache.sha256:
        raise SystemExit(f"factorial tensor hash mismatch in immutable cache {path}")
    return cache


def _load_attitude(path: Path) -> joint.AttitudeCache:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cache = joint.AttitudeCache(**payload)
    actual = joint._hash_tensors({name: getattr(cache, name) for name in ATTITUDE_TENSOR_FIELDS})
    if actual != cache.sha256:
        raise SystemExit(f"attitude tensor hash mismatch in immutable cache {path}")
    return cache


def _expected_cache_provenance(*, graph_sha256: str, checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "schema": CACHE_SCHEMA,
        "experiment": EXPERIMENT,
        "graph_sha256": graph_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "train_seed": TRAIN_SEED,
        "train_attitude_seed": TRAIN_SEED + 1,
        "development_seed": DEVELOPMENT_SEED,
        "development_attitude_seed": DEVELOPMENT_SEED + 1,
        "scenes_per_bank": joint.SCENES_PER_BANK,
        "prefix_steps": joint.PREFIX_STEPS,
        "response_steps": joint.RESPONSE_STEPS,
        "policy_hz": joint.POLICY_HZ,
    }


def _write_cache_bank(
    path: Path,
    *,
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
) -> dict[str, Any]:
    factorial_path = path / "factorial.pt"
    attitude_path = path / "attitude.pt"
    torch.save(_factorial_payload(factorial_cache), factorial_path)
    torch.save(_attitude_payload(attitude_cache), attitude_path)
    return {
        "factorial_file": factorial_path.name,
        "factorial_file_sha256": responsibility.file_sha256(factorial_path),
        "factorial_tensor_sha256": factorial_cache.sha256,
        "attitude_file": attitude_path.name,
        "attitude_file_sha256": responsibility.file_sha256(attitude_path),
        "attitude_tensor_sha256": attitude_cache.sha256,
    }


def _build_persisted_caches(
    cache_dir: Path,
    *,
    source: ConnectomeController,
    device: torch.device,
    config: HoverConfig,
    provenance: dict[str, Any],
) -> None:
    staging = cache_dir.with_name(f"{cache_dir.name}.staging")
    if staging.exists():
        raise SystemExit(f"incomplete immutable cache staging directory exists: {staging}")
    staging.mkdir(parents=True)
    training = staging / "training"
    development = staging / "development"
    training.mkdir()
    development.mkdir()

    train_factorial = joint.build_factorial_cache(
        scenes=joint.SCENES_PER_BANK,
        seed=TRAIN_SEED,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    train_attitude = joint.build_attitude_cache(
        source,
        scenes=joint.SCENES_PER_BANK,
        seed=TRAIN_SEED + 1,
        prefix_steps=joint.PREFIX_STEPS,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    development_factorial = joint.build_factorial_cache(
        scenes=joint.SCENES_PER_BANK,
        seed=DEVELOPMENT_SEED,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    development_attitude = joint.build_attitude_cache(
        source,
        scenes=joint.SCENES_PER_BANK,
        seed=DEVELOPMENT_SEED + 1,
        prefix_steps=joint.PREFIX_STEPS,
        response_steps=joint.RESPONSE_STEPS,
        policy_hz=joint.POLICY_HZ,
        device=device,
        config=config,
    )
    manifest = {
        **provenance,
        "banks": {
            "training": _write_cache_bank(
                training,
                factorial_cache=train_factorial,
                attitude_cache=train_attitude,
            ),
            "development": _write_cache_bank(
                development,
                factorial_cache=development_factorial,
                attitude_cache=development_attitude,
            ),
        },
    }
    (staging / "manifest.json").write_text(
        f"{json.dumps(manifest, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    os.rename(staging, cache_dir)


def load_or_create_caches(
    cache_dir: Path,
    *,
    source: ConnectomeController,
    device: torch.device,
    config: HoverConfig,
    graph_sha256: str,
    checkpoint_sha256: str,
) -> tuple[
    joint.FactorialCache,
    joint.AttitudeCache,
    joint.FactorialCache,
    joint.AttitudeCache,
    dict[str, Any],
]:
    provenance = _expected_cache_provenance(
        graph_sha256=graph_sha256, checkpoint_sha256=checkpoint_sha256
    )
    created = not cache_dir.exists()
    if created:
        _build_persisted_caches(
            cache_dir,
            source=source,
            device=device,
            config=config,
            provenance=provenance,
        )
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"immutable cache directory lacks manifest: {cache_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_provenance = {name: manifest.get(name) for name in provenance}
    if actual_provenance != provenance:
        raise SystemExit("immutable cache provenance does not match the frozen protocol")

    loaded: dict[str, tuple[joint.FactorialCache, joint.AttitudeCache]] = {}
    integrity: dict[str, Any] = {}
    for bank in ("training", "development"):
        bank_manifest = manifest["banks"][bank]
        bank_dir = cache_dir / bank
        factorial_path = bank_dir / bank_manifest["factorial_file"]
        attitude_path = bank_dir / bank_manifest["attitude_file"]
        factorial_cache = _load_factorial(factorial_path)
        attitude_cache = _load_attitude(attitude_path)
        checks = {
            "factorial_file_sha256": responsibility.file_sha256(factorial_path),
            "factorial_tensor_sha256": factorial_cache.sha256,
            "attitude_file_sha256": responsibility.file_sha256(attitude_path),
            "attitude_tensor_sha256": attitude_cache.sha256,
        }
        expected = {name: bank_manifest[name] for name in checks}
        if checks != expected:
            raise SystemExit(f"immutable {bank} cache hash mismatch")
        loaded[bank] = (factorial_cache, attitude_cache)
        integrity[bank] = checks
    integrity.update(
        {
            "created_this_run": created,
            "manifest": responsibility.stable_path(manifest_path),
            "manifest_sha256": responsibility.file_sha256(manifest_path),
            "all_persisted_file_and_tensor_hashes_match": True,
        }
    )
    train_factorial, train_attitude = loaded["training"]
    development_factorial, development_attitude = loaded["development"]
    return (
        train_factorial,
        train_attitude,
        development_factorial,
        development_attitude,
        integrity,
    )


def accumulated_endpoint_damping_gradient(
    controller: ConnectomeController,
    cache: joint.FactorialCache,
    *,
    scale: float,
    prefix_steps: int,
    device: torch.device,
) -> None:
    scenes = cache.prefix_images.shape[0]
    for scene_index in range(scenes):
        prediction, _ = joint._run_factorial_scene(
            controller, cache, scene_index, prefix_steps=prefix_steps, device=device
        )
        target = joint._factorial_targets(cache, scene_index, device)
        loss = ((prediction["damping"][-1] - target["damping"][-1]) / scale).square()
        (loss / scenes).backward()


def endpoint_damping_nrmse(
    raw: dict[str, Tensor],
    cache: joint.FactorialCache,
    *,
    scale: float,
    device: torch.device,
) -> float:
    target = torch.stack(
        [
            joint._factorial_targets(cache, scene_index, device)["damping"][-1]
            for scene_index in range(cache.prefix_images.shape[0])
        ]
    )
    prediction = raw["damping"][:, -1]
    return float(((prediction - target) / scale).square().mean().sqrt())


def evaluate(
    controller: ConnectomeController,
    factorial_cache: joint.FactorialCache,
    attitude_cache: joint.AttitudeCache,
    scales: dict[str, float],
    *,
    endpoint_scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    metrics, raw = joint.evaluate_bank(
        controller,
        factorial_cache,
        attitude_cache,
        scales,
        prefix_steps=joint.PREFIX_STEPS,
        device=device,
    )
    metrics["endpoint_damping_nrmse"] = endpoint_damping_nrmse(
        raw, factorial_cache, scale=endpoint_scale, device=device
    )
    return metrics, raw


def scale_decision(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    improvement = baseline["endpoint_damping_nrmse"] - candidate["endpoint_damping_nrmse"]
    if improvement < MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT:
        reasons.append("endpoint damping NRMSE improvement was below 0.001")
    for component in ("common", "height"):
        if (
            candidate["component_nrmse"][component]
            > baseline["component_nrmse"][component] + joint.COMPONENT_BASELINE_TOLERANCE
        ):
            reasons.append(f"aggregate {component} NRMSE exceeded source plus 0.02")
        for horizon in map(str, joint.SUPERVISION_STEPS):
            if (
                candidate["by_supervision_step_nrmse"][horizon][component]
                > baseline["by_supervision_step_nrmse"][horizon][component]
                + joint.COMPONENT_BASELINE_TOLERANCE
            ):
                reasons.append(f"{component} NRMSE at step {horizon} exceeded source plus 0.02")
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
    return {
        "pass": not reasons,
        "reasons": reasons,
        "endpoint_damping_nrmse_improvement": improvement,
        "all_metrics_finite": finite,
    }


def largest_feasible_scale(trials: list[dict[str, Any]]) -> float | None:
    for trial in trials:
        if trial["decision"]["pass"]:
            return float(trial["scale"])
    return None


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
    source_parameters = joint._copy_parameters(student)

    print(json.dumps({"stage": "persisting_or_loading_immutable_caches"}), flush=True)
    (
        train_factorial,
        train_attitude,
        development_factorial,
        development_attitude,
        cache_integrity,
    ) = load_or_create_caches(
        args.output_dir / "immutable-cache-v1",
        source=source,
        device=device,
        config=config,
        graph_sha256=graph_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    endpoint_scale = endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )

    print(json.dumps({"stage": "replaying_training_source"}), flush=True)
    baseline_train, baseline_raw = evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    _, replay_raw = evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
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
    print(json.dumps({"stage": "differentiating_endpoint_damping"}), flush=True)
    accumulated_endpoint_damping_gradient(
        student,
        train_factorial,
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

    print(json.dumps({"stage": "training_scale_sweep"}), flush=True)
    scale_trials = []
    for scale in FULL_SCALES:
        actual_rms = step_audit.set_displacement(
            student, source_parameters, displacement, scale=scale
        )
        metrics, _ = evaluate(
            student,
            train_factorial,
            train_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        scale_trials.append(
            {
                "scale": scale,
                "metrics": metrics,
                "actual_displacement_family_rms": actual_rms,
                "decision": scale_decision(baseline_train, metrics),
            }
        )
    selected_scale = largest_feasible_scale(scale_trials)

    step_audit.set_displacement(
        student,
        source_parameters,
        displacement,
        scale=joint.FINITE_DIFFERENCE_SCALE,
    )
    finite_difference_metrics, _ = evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    baseline_objective = baseline_train["endpoint_damping_nrmse"] ** 2
    finite_difference_objective = finite_difference_metrics["endpoint_damping_nrmse"] ** 2
    finite_difference = (
        finite_difference_objective - baseline_objective
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

    baseline_development = None
    development_trial = None
    if selected_scale is not None:
        print(json.dumps({"stage": "single_development_trial"}), flush=True)
        joint._load_parameters(student, source_parameters)
        baseline_development, _ = evaluate(
            student,
            development_factorial,
            development_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        actual_rms = step_audit.set_displacement(
            student, source_parameters, displacement, scale=selected_scale
        )
        metrics, _ = evaluate(
            student,
            development_factorial,
            development_attitude,
            scales,
            endpoint_scale=endpoint_scale,
            device=device,
        )
        development_trial = {
            "scale": selected_scale,
            "metrics": metrics,
            "actual_displacement_family_rms": actual_rms,
            "decision": scale_decision(baseline_development, metrics),
        }

    joint._load_parameters(student, source_parameters)
    restored = all(
        torch.equal(getattr(student, name).detach(), source_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    controls_pass = bool(
        cache_integrity["all_persisted_file_and_tensor_hashes_match"]
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
    if not controls_pass:
        classification = "audit_control_failure"
    elif selected_scale is None:
        classification = "no_training_feasible_endpoint_damping_scale"
    elif not development_trial["decision"]["pass"]:
        classification = "training_selected_scale_failed_development"
    else:
        classification = "safe_transferable_endpoint_damping_step"

    report = {
        "experiment": EXPERIMENT,
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": passed,
        "classification": classification,
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
        "deterministic_source_replay_max_absolute_difference": replay_difference,
        "directional_derivative": directional,
        "gradient_norm_before_clipping": float(gradient_norm),
        "full_displacement_family_rms": displacement_rms,
        "baseline_training": baseline_train,
        "training_scale_trials": scale_trials,
        "selected_training_scale": selected_scale,
        "baseline_development": baseline_development,
        "selected_development_trial": development_trial,
        "controls_pass": controls_pass,
        "bounded_d_first_fitting_authorized": passed,
        "parameters_restored_exactly": restored,
        "promoted": False,
        "closed_loop_hover_run": False,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass demonstrates only one safe, transferable endpoint-damping-directed "
            "step and authorizes a separately preregistered bounded D-first fitting "
            "diagnostic; it is not a damping-capacity, hover, or flight claim."
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
                    development_trial["decision"]["pass"]
                    if development_trial is not None
                    else None
                ),
                "bounded_d_first_fitting_authorized": passed,
                "parameters_restored_exactly": restored,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
