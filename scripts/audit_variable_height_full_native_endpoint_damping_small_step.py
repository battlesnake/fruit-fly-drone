#!/usr/bin/env python3
"""Audit 1/16 and 1/32 of the frozen full-native endpoint-damping step."""

from __future__ import annotations

import argparse
import json
import math
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

import audit_variable_height_full_native_endpoint_damping as endpoint  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_full_native_step_response as step_audit  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-full-native-endpoint-damping-small-step-audit-v1"
PROTOCOL_COMMIT = "f293999"
SCALES = (0.0625, 0.03125)
REPRODUCTION_RELATIVE_TOLERANCE = 1.0e-3
REPRODUCTION_ABSOLUTE_TOLERANCE = 1.0e-7
EXPECTED_CACHE = {
    "manifest_sha256": "a3aa57d103158adc571734a49182df2fbf04852089532bf9f88631c9bac92060",
    "training_factorial_tensor_sha256": (
        "55e4a3199b0a3da9b221f6a34619954a1cda91297344fa13854b454507760a78"
    ),
    "training_attitude_tensor_sha256": (
        "2293ba902703f48c602f8074fb3965efd7922f8571a3cb8913266970cdd3f072"
    ),
    "development_factorial_tensor_sha256": (
        "50cc64ff87baf720175ce483500cb4bf1c53d53f7cb5905333284698c706cb5e"
    ),
    "development_attitude_tensor_sha256": (
        "0f28df2e589cc2c081d05794a1bf962fce09a5895fd61e5b6ae87ddbd2e0b6ac"
    ),
}
EXPECTED_SCALARS = {
    "endpoint_damping_scale": 0.02182983234524727,
    "baseline_endpoint_damping_nrmse": 1.4584003686904907,
    "full_scale_endpoint_damping_nrmse": 1.4032435417175293,
    "autograd_directional_derivative": -0.15886689722537994,
    "edge_magnitude_rms": 0.00009662459342507645,
    "bias_rms": 0.00009603651415091008,
    "raw_time_constant_rms": 0.00000087546766280866,
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
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/full-native-endpoint-damping-small-step-audit-001"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not (args.source_audit_dir / "immutable-cache-v1/manifest.json").is_file():
        raise SystemExit("the original audit's immutable cache is required and may not regenerate")
    if args.smoke_test:
        raise SystemExit("this preregistered extension has no smoke variant")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_experiment": endpoint.EXPERIMENT,
        "source_train_seed": endpoint.TRAIN_SEED,
        "source_development_seed": endpoint.DEVELOPMENT_SEED,
        "source_immutable_cache_required": True,
        "source_cache_regeneration_allowed": False,
        "displacement": (
            "regenerate the source audit's complete bound-projected Adam displacement, "
            "verify its frozen scalar signature, persist it, hash it, reload it, and use "
            "the exact persisted tensors for all trials"
        ),
        "reproduction_relative_tolerance": REPRODUCTION_RELATIVE_TOLERANCE,
        "reproduction_absolute_tolerance": REPRODUCTION_ABSOLUTE_TOLERANCE,
        "training_scales_descending": list(SCALES),
        "largest_training_feasible_scale_selected": True,
        "development_candidates_evaluated": 1,
        "no_smaller_scale_after_development_failure": True,
        "objective": "normalized endpoint damping MSE only",
        "minimum_endpoint_damping_nrmse_improvement": (
            endpoint.MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT
        ),
        "preservation": endpoint.protocol_manifest()["preservation"],
        "joint_loss_and_interaction_are_reporting_only": True,
        "actor_contract_unchanged": True,
        "parameters_restored_exactly": True,
        "candidate_retained": False,
        "pass_authorizes": (
            "only a separately preregistered bounded D-first fitting diagnostic with "
            "cumulative C/P preservation measured against the original source"
        ),
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
    cache_integrity: dict[str, Any],
    endpoint_scale: float,
    baseline: dict[str, Any],
    full_candidate: dict[str, Any],
    derivative: float,
    displacement_rms: dict[str, float],
) -> dict[str, Any]:
    cache_actual = {
        "manifest_sha256": cache_integrity["manifest_sha256"],
        "training_factorial_tensor_sha256": cache_integrity["training"][
            "factorial_tensor_sha256"
        ],
        "training_attitude_tensor_sha256": cache_integrity["training"][
            "attitude_tensor_sha256"
        ],
        "development_factorial_tensor_sha256": cache_integrity["development"][
            "factorial_tensor_sha256"
        ],
        "development_attitude_tensor_sha256": cache_integrity["development"][
            "attitude_tensor_sha256"
        ],
    }
    scalar_actual = {
        "endpoint_damping_scale": endpoint_scale,
        "baseline_endpoint_damping_nrmse": baseline["endpoint_damping_nrmse"],
        "full_scale_endpoint_damping_nrmse": full_candidate["endpoint_damping_nrmse"],
        "autograd_directional_derivative": derivative,
        "edge_magnitude_rms": displacement_rms["edge_magnitude"],
        "bias_rms": displacement_rms["bias"],
        "raw_time_constant_rms": displacement_rms["raw_time_constant"],
    }
    cache = {
        name: {
            "actual": value,
            "expected": EXPECTED_CACHE[name],
            "pass": value == EXPECTED_CACHE[name],
        }
        for name, value in cache_actual.items()
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
        "pass": all(item["pass"] for item in (*cache.values(), *scalars.values())),
        "cache": cache,
        "scalars": scalars,
    }


def persist_or_load_displacement(
    path: Path,
    regenerated: dict[str, Tensor],
    *,
    checkpoint_sha256: str,
    cache_manifest_sha256: str,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    regenerated_cpu = {name: value.detach().cpu() for name, value in regenerated.items()}
    tensor_sha256 = joint._hash_tensors(regenerated_cpu)
    provenance = {
        "experiment": EXPERIMENT,
        "checkpoint_sha256": checkpoint_sha256,
        "cache_manifest_sha256": cache_manifest_sha256,
        "tensor_sha256": tensor_sha256,
    }
    created = not path.exists()
    if created:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_suffix(f"{path.suffix}.staging")
        if staging.exists():
            raise SystemExit(f"incomplete immutable displacement staging file exists: {staging}")
        torch.save({**provenance, "tensors": regenerated_cpu}, staging)
        os.rename(staging, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actual_provenance = {name: payload.get(name) for name in provenance}
    if actual_provenance != provenance:
        raise SystemExit("immutable displacement provenance or tensor hash mismatch")
    persisted_cpu = payload["tensors"]
    if joint._hash_tensors(persisted_cpu) != tensor_sha256:
        raise SystemExit("immutable displacement payload hash mismatch")
    if any(
        not torch.equal(persisted_cpu[name], regenerated_cpu[name])
        for name in joint.PARAMETER_FAMILIES
    ):
        raise SystemExit("regenerated displacement differs from immutable displacement")
    device = next(iter(regenerated.values())).device
    persisted = {name: value.to(device) for name, value in persisted_cpu.items()}
    return persisted, {
        "created_this_run": created,
        "file": responsibility.stable_path(path),
        "file_sha256": responsibility.file_sha256(path),
        "tensor_sha256": tensor_sha256,
        "regenerated_matches_persisted_exactly": True,
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
    source_parameters = joint._copy_parameters(student)

    print(json.dumps({"stage": "loading_original_immutable_caches"}), flush=True)
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
        raise SystemExit("source cache was regenerated, contrary to the frozen protocol")
    endpoint_scale = endpoint.endpoint_damping_scale(train_factorial)
    scales = joint.training_teacher_scales(train_factorial)
    scales["damping"] = endpoint_scale
    teacher_identity = joint.teacher_identity_report(train_factorial, scales)
    teacher_positive = joint.teacher_plant_positive_control(
        train_factorial, device=device, config=config
    )

    print(json.dumps({"stage": "reproducing_source_audit_direction"}), flush=True)
    baseline_train, baseline_raw = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    _, replay_raw = endpoint.evaluate(
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
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), joint.GRADIENT_NORM_CAP)
    optimizer.step()
    student.project_parameters()
    full_parameters = joint._copy_parameters(student)
    regenerated_displacement = {
        name: full_parameters[name] - source_parameters[name] for name in joint.PARAMETER_FAMILIES
    }
    displacement_rms = {
        name: float(value.square().mean().sqrt())
        for name, value in regenerated_displacement.items()
    }
    derivative = float(
        sum(
            (raw_gradients[name] * regenerated_displacement[name]).sum()
            for name in joint.PARAMETER_FAMILIES
        )
    )
    full_candidate, _ = endpoint.evaluate(
        student,
        train_factorial,
        train_attitude,
        scales,
        endpoint_scale=endpoint_scale,
        device=device,
    )
    reproduction = reproduction_report(
        cache_integrity=cache_integrity,
        endpoint_scale=endpoint_scale,
        baseline=baseline_train,
        full_candidate=full_candidate,
        derivative=derivative,
        displacement_rms=displacement_rms,
    )

    displacement = None
    displacement_integrity = None
    scale_trials = []
    selected_scale = None
    if reproduction["pass"]:
        displacement, displacement_integrity = persist_or_load_displacement(
            args.output_dir / "immutable-displacement-v1.pt",
            regenerated_displacement,
            checkpoint_sha256=checkpoint_sha256,
            cache_manifest_sha256=cache_integrity["manifest_sha256"],
        )
        print(json.dumps({"stage": "small_training_scale_sweep"}), flush=True)
        for scale in SCALES:
            actual_rms = step_audit.set_displacement(
                student, source_parameters, displacement, scale=scale
            )
            metrics, _ = endpoint.evaluate(
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
                    "decision": endpoint.scale_decision(baseline_train, metrics),
                }
            )
        selected_scale = endpoint.largest_feasible_scale(scale_trials)

    baseline_development = None
    development_trial = None
    if selected_scale is not None and displacement is not None:
        print(json.dumps({"stage": "single_development_trial"}), flush=True)
        joint._load_parameters(student, source_parameters)
        baseline_development, _ = endpoint.evaluate(
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
        metrics, _ = endpoint.evaluate(
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
            "decision": endpoint.scale_decision(baseline_development, metrics),
        }

    joint._load_parameters(student, source_parameters)
    restored = all(
        torch.equal(getattr(student, name).detach(), source_parameters[name])
        for name in joint.PARAMETER_FAMILIES
    )
    controls_pass = bool(
        reproduction["pass"]
        and cache_integrity["all_persisted_file_and_tensor_hashes_match"]
        and displacement_integrity is not None
        and displacement_integrity["regenerated_matches_persisted_exactly"]
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
        classification = "no_training_feasible_small_scale"
    elif not development_trial["decision"]["pass"]:
        classification = "training_selected_scale_failed_development"
    else:
        classification = "safe_transferable_small_endpoint_damping_step"

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
            "source_audit_dir": responsibility.stable_path(args.source_audit_dir),
        },
        "actor_contract_unchanged": True,
        "protocol": protocol_manifest(),
        "cache_integrity": cache_integrity,
        "displacement_integrity": displacement_integrity,
        "reproduction": reproduction,
        "teacher_component_scales_motor_units": scales,
        "endpoint_damping_scale_motor_units": endpoint_scale,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "deterministic_source_replay_max_absolute_difference": replay_difference,
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
            "A pass demonstrates only locally feasible damping improvement, not "
            "independent C/D control, sustained learnability, hover, or flight."
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
