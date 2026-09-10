#!/usr/bin/env python3
"""Audit joint seven-horizon teacher-action capacity on eight fixed gate pairs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_analytic_teachers import teacher_rc_for_mode  # noqa: E402
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_motor_interface_es import stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    controller_parameter_sha256,
    load_frozen_controller,
    parameter_change_summary,
    parameter_snapshot,
    restore_parameters,
)
from train_gate_multitime_distillation import (  # noqa: E402
    MultiTimeDataset,
    collect_analytic_trajectories,
    fidelity_margin_score,
    fidelity_metrics,
    fidelity_passed,
    joint_multitime_loss,
    load_warm_start,
    make_dataset,
    target_scales,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--teacher-spec",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-analytic-teacher-reserve-v1" / "candidate.json"),
    )
    parser.add_argument(
        "--warm-start-vector",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-conditional-overfit-diagnostic-v1"
            / "candidate-vector.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "joint-multitime-overfit-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--initialization",
        choices=("conditional-vector", "source"),
        default="conditional-vector",
    )
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--candidate-pairs", type=int, default=32)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument("--axis-action-weight", type=float, default=3.0)
    parser.add_argument("--axis-action-scale", type=float, default=0.01)
    parser.add_argument("--throttle-action-scale", type=float, default=0.01)
    parser.add_argument("--anchor-seconds", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--regularization-weight", type=float, default=1.0e-6)
    parser.add_argument("--prefix-seconds", type=float, default=5.0)
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        nargs="+",
        default=(0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00),
    )
    parser.add_argument("--horizon-weights", type=float, nargs="+", default=(2, 2, 1, 1, 1, 1, 1))
    parser.add_argument("--action-fidelity-threshold", type=float, default=0.25)
    parser.add_argument("--training-seed", type=int, default=1_011_031)
    parser.add_argument("--holdout-seed", type=int, default=1_012_031)
    parser.add_argument("--optimization-seed", type=int, default=1_013_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> tuple[tuple[int, ...], int]:
    for path in (
        args.graph,
        args.student_checkpoint,
        args.teacher_spec,
        args.warm_start_vector,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.pairs != 8 or args.candidate_pairs != 32:
        raise SystemExit("the bounded audit requires 8 selected pairs from 32 candidates")
    if args.updates != 150 or args.evaluation_interval != 25:
        raise SystemExit("the bounded audit requires 150 updates evaluated every 25 updates")
    if len(args.horizon_seconds) != 7 or len(args.horizon_weights) != 7:
        raise SystemExit("the audit requires seven fixed horizons and weights")
    if tuple(args.horizon_weights) != (2, 2, 1, 1, 1, 1, 1):
        raise SystemExit("unexpected joint-horizon weights")
    positive = (
        args.learning_rate,
        args.time_constant_learning_rate,
        args.gradient_norm_cap,
        args.axis_action_weight,
        args.axis_action_scale,
        args.throttle_action_scale,
        args.anchor_seconds,
        args.anchor_weight,
        args.regularization_weight,
        args.prefix_seconds,
        args.action_fidelity_threshold,
        *args.horizon_seconds,
    )
    if min(positive) <= 0.0:
        raise SystemExit("audit rates, scales, times, and thresholds must be positive")
    horizon_steps = tuple(round(value / dt) for value in args.horizon_seconds)
    anchor_step = round(args.anchor_seconds / dt)
    if horizon_steps != (50, 75, 100, 150, 200, 300, 500):
        raise SystemExit("unexpected recurrent endpoints")
    if anchor_step != 20 or horizon_steps[-1] != round(args.prefix_seconds / dt):
        raise SystemExit("unexpected anchor or complete-prefix length")
    return horizon_steps, anchor_step


def subset(dataset: MultiTimeDataset, episodes: torch.Tensor) -> MultiTimeDataset:
    return MultiTimeDataset(
        images=dataset.images[:, episodes],
        roll_pitch=dataset.roll_pitch[:, episodes],
        specific_force=dataset.specific_force[:, episodes],
        stick_position=dataset.stick_position[:, episodes],
        valid_at_horizons=dataset.valid_at_horizons[:, episodes],
        source_motor=dataset.source_motor[:, episodes],
        source_anchor=dataset.source_anchor[episodes],
        valid_anchor=dataset.valid_anchor[episodes],
        target_correction=dataset.target_correction[:, episodes],
        mass_scale=dataset.mass_scale[episodes],
        codes=dataset.codes[episodes],
        source_summary=dataset.source_summary,
        collection_policy_sha256=dataset.collection_policy_sha256,
    )


def balanced_all_horizon_survivors(dataset: MultiTimeDataset, *, pairs: int) -> MultiTimeDataset:
    pair_valid = (dataset.valid_at_horizons[:, 0::2] & dataset.valid_at_horizons[:, 1::2]).all(
        dim=0
    )
    pair_codes = dataset.codes[0::2].bitwise_and(6)
    chosen = []
    per_geometry = pairs // 4
    for code in (0, 2, 4, 6):
        available = torch.nonzero(pair_valid & (pair_codes == code), as_tuple=False).flatten()
        if len(available) < per_geometry:
            raise RuntimeError(f"not enough all-horizon survivor pairs for geometry code {code}")
        chosen.extend(available[:per_geometry].tolist())
    episode_indices = torch.tensor(
        [episode for pair in chosen for episode in (2 * pair, 2 * pair + 1)],
        device=dataset.images.device,
    )
    selected = subset(dataset, episode_indices)
    if not bool(selected.valid_at_horizons.all()):
        raise RuntimeError("capacity-audit selection contains an invalid endpoint")
    return selected


@torch.no_grad()
def collect_audit_dataset(
    source: Any,
    *,
    teacher_mode: str,
    seed: int,
    args: argparse.Namespace,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
    device: torch.device,
    hover_config: Any,
    gate_config: Any,
    resolution: int,
) -> tuple[MultiTimeDataset, dict[str, Any]]:
    cases = diverse_matched_cases(
        2 * args.candidate_pairs,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    normal = teacher_rc_for_mode(
        teacher_mode, source, cases.state, cases.gate, cases.mass_scale, hover_config
    )
    swapped = teacher_rc_for_mode(
        teacher_mode, source, cases.state, cases.gate, cases.mass_scale.flip(0), hover_config
    )
    if not torch.equal(normal, swapped):
        raise RuntimeError("mass-free audit teacher depends on its mass argument")
    trajectories = collect_analytic_trajectories(
        source,
        source,
        cases,
        teacher_mode=teacher_mode,
        seconds=args.prefix_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    full = make_dataset(
        trajectories,
        source,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        device=device,
    )
    selected = balanced_all_horizon_survivors(full, pairs=args.pairs)
    selection = {
        "candidate_pairs": args.candidate_pairs,
        "all_horizon_survivor_pairs": int(
            (full.valid_at_horizons[:, 0::2] & full.valid_at_horizons[:, 1::2]).all(dim=0).sum()
        ),
        "selected_pairs": args.pairs,
        "selected_geometry_codes": selected.codes[0::2].bitwise_and(6).cpu().tolist(),
        "selected_mass_scales": selected.mass_scale.cpu().tolist(),
        "selected_all_horizons_valid": bool(selected.valid_at_horizons.all()),
        "collection": trajectories.summary,
    }
    return selected, selection


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        time: {
            "contrast": item["contrast_normalized_rmse"],
            "mean": item["mean_normalized_rmse"],
            "axis_max": max(item["axis_teacher_action_normalized_rmse"].values()),
        }
        for time, item in metrics.items()
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source, _, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.student_checkpoint, device
    )
    horizon_steps, anchor_step = validate_args(args, hover_config.dt)
    if not source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("unexpected deployed sensor contract")
    teacher_spec = json.loads(args.teacher_spec.read_text())
    if (
        teacher_spec.get("kind") != "privileged_analytic_gate_teacher"
        or teacher_spec.get("uses_exact_simulator_mass")
        or not teacher_spec.get("preflight_passed")
        or teacher_spec.get("checkpoint_sha256") != file_sha256(args.student_checkpoint)
    ):
        raise SystemExit("audit requires the validated mass-free analytical teacher")
    teacher_mode = teacher_spec["teacher_mode"]
    if teacher_mode != "visual_accelerometer_reserve" or teacher_spec.get(
        "takeover_seconds"
    ) != min(args.horizon_seconds):
        raise SystemExit("unexpected teacher mode or validated takeover time")
    regularization_reference = copy.deepcopy(source).to(device)
    warm_start = load_warm_start(
        regularization_reference,
        args.warm_start_vector,
        source_checkpoint=args.student_checkpoint,
    )
    student = copy.deepcopy(
        regularization_reference if args.initialization == "conditional-vector" else source
    ).to(device)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    initialization_parameters = parameter_snapshot(student)
    regularization_parameters = parameter_snapshot(regularization_reference)
    initialization_parameter_sha256 = controller_parameter_sha256(student)
    regularization_parameter_sha256 = controller_parameter_sha256(regularization_reference)
    if regularization_parameter_sha256 != warm_start["parameter_vector_sha256"]:
        raise RuntimeError("regularization reference does not match the frozen warm-start vector")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / "selected-vector.json"
    if vector_path.exists():
        raise SystemExit("output directory contains a stale selected-vector.json")
    seed_everything(args.optimization_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    training, training_selection = collect_audit_dataset(
        source,
        teacher_mode=teacher_mode,
        seed=args.training_seed,
        args=args,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        resolution=resolution,
    )
    holdout, holdout_selection = collect_audit_dataset(
        source,
        teacher_mode=teacher_mode,
        seed=args.holdout_seed,
        args=args,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        resolution=resolution,
    )
    scales = target_scales(
        training,
        axis_action_floor=args.axis_action_scale,
        throttle_action_floor=args.throttle_action_scale,
    )
    weights = tuple(args.horizon_weights)
    with torch.no_grad():
        initial_training_metrics = fidelity_metrics(
            student, training, scales, horizon_steps=horizon_steps, dt=hover_config.dt
        )
        initial_holdout_metrics = fidelity_metrics(
            student, holdout, scales, horizon_steps=horizon_steps, dt=hover_config.dt
        )
        initial_training_loss = float(
            joint_multitime_loss(
                student,
                training,
                scales,
                regularization_parameters,
                horizon_steps=horizon_steps,
                horizon_weights=weights,
                anchor_step=anchor_step,
                axis_action_weight=args.axis_action_weight,
                anchor_weight=args.anchor_weight,
                regularization_weight=args.regularization_weight,
            )[0]
        )
    initial_holdout_margin = fidelity_margin_score(
        initial_holdout_metrics, threshold=args.action_fidelity_threshold
    )
    initial_training_margin = fidelity_margin_score(
        initial_training_metrics, threshold=args.action_fidelity_threshold
    )

    student.zero_grad(set_to_none=True)
    audit_loss, _ = joint_multitime_loss(
        student,
        training,
        scales,
        regularization_parameters,
        horizon_steps=horizon_steps,
        horizon_weights=weights,
        anchor_step=anchor_step,
        axis_action_weight=args.axis_action_weight,
        anchor_weight=args.anchor_weight,
        regularization_weight=args.regularization_weight,
    )
    audit_loss.backward()
    gradient_families = {
        name: {
            "l2_norm": float(parameter.grad.norm()) if parameter.grad is not None else None,
            "finite": bool(parameter.grad is not None and torch.isfinite(parameter.grad).all()),
            "nonzero": bool(parameter.grad is not None and parameter.grad.abs().max() > 0.0),
        }
        for name, parameter in (
            ("edge_magnitude", student.edge_magnitude),
            ("bias", student.bias),
            ("raw_time_constant", student.raw_time_constant),
        )
    }
    gradient_audit_passed = all(
        item["finite"] and item["nonzero"] for item in gradient_families.values()
    )
    student.zero_grad(set_to_none=True)
    if not gradient_audit_passed:
        raise RuntimeError(f"joint-loss gradient audit failed: {gradient_families}")

    optimizer = torch.optim.Adam(
        (
            {"params": [student.edge_magnitude, student.bias], "lr": args.learning_rate},
            {"params": [student.raw_time_constant], "lr": args.time_constant_learning_rate},
        )
    )
    history = []
    best: dict[str, Any] | None = None
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = joint_multitime_loss(
            student,
            training,
            scales,
            regularization_parameters,
            horizon_steps=horizon_steps,
            horizon_weights=weights,
            anchor_step=anchor_step,
            axis_action_weight=args.axis_action_weight,
            anchor_weight=args.anchor_weight,
            regularization_weight=args.regularization_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite joint loss at update {update}")
        loss.backward()
        raw_gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in student.parameters()
                    if parameter.grad is not None
                ]
            )
        )
        torch.nn.utils.clip_grad_norm_(
            student.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
        )
        optimizer.step()
        student.project_parameters()
        if update % args.evaluation_interval:
            continue
        with torch.no_grad():
            training_metrics = fidelity_metrics(
                student, training, scales, horizon_steps=horizon_steps, dt=hover_config.dt
            )
            holdout_metrics = fidelity_metrics(
                student, holdout, scales, horizon_steps=horizon_steps, dt=hover_config.dt
            )
        training_margin = fidelity_margin_score(
            training_metrics, threshold=args.action_fidelity_threshold
        )
        holdout_margin = fidelity_margin_score(
            holdout_metrics, threshold=args.action_fidelity_threshold
        )
        record = {
            "update": update,
            "training_margin_score": training_margin,
            "holdout_margin_score": holdout_margin,
            "training_fidelity_passed": fidelity_passed(
                training_metrics, threshold=args.action_fidelity_threshold
            ),
            "loss": details,
            "raw_gradient_norm": float(raw_gradient_norm),
            "training_metrics": training_metrics,
            "holdout_metrics": holdout_metrics,
        }
        history.append(record)
        key = (training_margin, details["total_loss"])
        if best is None or key < best["selection_key"]:
            best = {
                **record,
                "selection_key": key,
                "parameter_vector_sha256": controller_parameter_sha256(student),
                "parameters": parameter_snapshot(student, cpu=True),
            }
        print(
            json.dumps(
                {
                    "phase": "joint_multitime_selection",
                    "update": update,
                    "training_margin_score": training_margin,
                    "holdout_margin_score": holdout_margin,
                    "training_fidelity_passed": record["training_fidelity_passed"],
                    "training": compact_metrics(training_metrics),
                }
            ),
            flush=True,
        )
    if best is None:
        raise RuntimeError("audit produced no selection snapshot")
    restore_parameters(student, best["parameters"])
    selected_training_metrics = best["training_metrics"]
    selected_holdout_metrics = best["holdout_metrics"]
    training_fit_passed = fidelity_passed(
        selected_training_metrics, threshold=args.action_fidelity_threshold
    )
    selected_holdout_margin = fidelity_margin_score(
        selected_holdout_metrics, threshold=args.action_fidelity_threshold
    )
    holdout_improved = selected_holdout_margin < initial_holdout_margin
    audit_passed = training_fit_passed and holdout_improved

    vector = {
        "source_checkpoint_sha256": file_sha256(args.student_checkpoint),
        "warm_start_parameter_sha256": warm_start["parameter_vector_sha256"],
        "initialization": args.initialization,
        "initialization_parameter_sha256": initialization_parameter_sha256,
        "regularization_reference_parameter_sha256": regularization_parameter_sha256,
        "parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_update": best["update"],
        "edge_magnitude": student.edge_magnitude.detach().cpu().tolist(),
        "bias": student.bias.detach().cpu().tolist(),
        "raw_time_constant": student.raw_time_constant.detach().cpu().tolist(),
    }
    vector_path.write_text(json.dumps(vector, indent=2, sort_keys=True) + "\n")
    report = {
        "method": "joint seven-horizon native recurrent capacity audit",
        "claim_scope": (
            "Limited eight-pair representability audit; never a promoted flight result. "
            "The actor uses no mass, clock, engineered history, or added recurrent module."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "student_checkpoint": stable_path(args.student_checkpoint),
        "student_checkpoint_sha256": file_sha256(args.student_checkpoint),
        "teacher_spec": stable_path(args.teacher_spec),
        "teacher_spec_sha256": file_sha256(args.teacher_spec),
        "teacher_mode": teacher_mode,
        "warm_start_vector": stable_path(args.warm_start_vector),
        "warm_start_vector_sha256": file_sha256(args.warm_start_vector),
        "initialization": args.initialization,
        "initialization_parameter_sha256": initialization_parameter_sha256,
        "regularization_reference_parameter_sha256": regularization_parameter_sha256,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "complete_prefix_per_update": True,
            "all_endpoints_in_every_loss": True,
            "endpoint_aggregation": "fixed-weight normalized average",
            "anchor_and_regularization_applied_once": True,
            "selection": "minimum worst threshold-normalized training error, then joint loss",
            "training_selection_uses_all_horizon_survivors": True,
            "holdout_not_used_for_selection": True,
            "fixed_topology_and_transmitter_signs": True,
            "all_native_parameter_families_trainable": True,
            "mass_actor_input": False,
            "engineered_history_features": False,
        },
        "training_case_selection": training_selection,
        "holdout_case_selection": holdout_selection,
        "target_scales": {
            "contrast_rms": scales.contrast_squared.sqrt().detach().cpu().tolist(),
            "mean_rms": scales.mean_squared.sqrt().detach().cpu().tolist(),
            "axis_rms": scales.axis_squared.sqrt().detach().cpu().tolist(),
            "anchor_rms": scales.anchor_squared.sqrt().detach().cpu().tolist(),
        },
        "initial_training_loss": initial_training_loss,
        "initial_training_metrics": initial_training_metrics,
        "initial_training_margin_score": initial_training_margin,
        "initial_holdout_metrics": initial_holdout_metrics,
        "initial_holdout_margin_score": initial_holdout_margin,
        "gradient_audit": {
            "complete_prefix_steps": horizon_steps[-1],
            "families": gradient_families,
            "passed": gradient_audit_passed,
        },
        "history": history,
        "selected_update": best["update"],
        "selected_parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_training_metrics": selected_training_metrics,
        "selected_holdout_metrics": selected_holdout_metrics,
        "selected_training_margin_score": best["training_margin_score"],
        "selected_holdout_margin_score": selected_holdout_margin,
        "selected_parameter_change_from_initialization": parameter_change_summary(
            student, initialization_parameters
        ),
        "selected_parameter_change_from_regularization_reference": parameter_change_summary(
            student, regularization_parameters
        ),
        "selected_vector": stable_path(vector_path),
        "selected_vector_sha256": file_sha256(vector_path),
        "training_fit_passed": training_fit_passed,
        "holdout_improved_from_initialization": holdout_improved,
        "audit_passed": audit_passed,
        "checkpoint_promoted": False,
        "goal_passed": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "selected_update": best["update"],
                "training_fit_passed": training_fit_passed,
                "holdout_improved": holdout_improved,
                "audit_passed": audit_passed,
            }
        ),
        flush=True,
    )
    return 0 if audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
