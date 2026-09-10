#!/usr/bin/env python3
"""Audit sampling and generalization of the frozen recurrent-routing checkpoint."""

from __future__ import annotations

import argparse
import json
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

from audit_gate_throttle_routing import (  # noqa: E402
    make_readout_spec,
    metric_summary,
    target_scales,
)
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_motor_interface_es import load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_recurrent_routing import (  # noqa: E402
    balanced_pair_split,
    make_recurrent_routing_spec,
    metric_samples,
    pair_episodes,
    prepare_history_targets,
    recurrent_motor_replay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--routing-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "report.json",
    )
    parser.add_argument(
        "--routing-readouts",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "fitted-readouts.json"
        ),
    )
    parser.add_argument(
        "--recurrent-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-recurrent-routing-diagnostic-v1" / "report.json",
    )
    parser.add_argument(
        "--recurrent-mask",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-diagnostic-v1"
            / "preregistered-mask.json"
        ),
    )
    parser.add_argument(
        "--recurrent-magnitudes",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-diagnostic-v1"
            / "selected-magnitudes.json"
        ),
    )
    parser.add_argument(
        "--replay-histories",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-routing-audit-v1" / "replay-histories.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "recurrent-routing-sampling-audit-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--minibatch-pairs", type=int, default=8)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--generalization-ratio", type=float, default=1.25)
    parser.add_argument("--sampling-distortion-fraction", type=float, default=0.10)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--seconds", type=float, default=1.50)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--anchor-seconds", type=float, default=0.20)
    parser.add_argument("--optimizer-seed", type=int, default=1_047_131)
    parser.add_argument("--bootstrap-seed", type=int, default=1_049_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.routing_report,
        args.routing_readouts,
        args.recurrent_report,
        args.recurrent_mask,
        args.recurrent_magnitudes,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "updates": (args.updates, 100),
        "minibatch_pairs": (args.minibatch_pairs, 8),
        "bootstrap_replicates": (args.bootstrap_replicates, 2000),
        "generalization_ratio": (args.generalization_ratio, 1.25),
        "sampling_distortion_fraction": (args.sampling_distortion_fraction, 0.10),
        "normalization_floor": (args.normalization_floor, 0.01),
        "seconds": (args.seconds, 1.50),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "anchor_seconds": (args.anchor_seconds, 0.20),
        "optimizer_seed": (args.optimizer_seed, 1_047_131),
        "bootstrap_seed": (args.bootstrap_seed, 1_049_031),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered sampling-audit values changed: {', '.join(wrong)}")


def recover_pair_exposures(
    *, updates: int, minibatch_pairs: int, seed: int
) -> tuple[np.ndarray, list[list[int]]]:
    fit_pairs, _ = balanced_pair_split()
    first_half = fit_pairs[: len(fit_pairs) // 2]
    second_half = fit_pairs[len(fit_pairs) // 2 :]
    generator = np.random.default_rng(seed)
    counts = np.zeros(64, dtype=np.int64)
    batches = []
    for _ in range(updates):
        batch = np.concatenate(
            (
                generator.choice(first_half, minibatch_pairs // 2, replace=False),
                generator.choice(second_half, minibatch_pairs // 2, replace=False),
            )
        )
        generator.shuffle(batch)
        counts[batch] += 1
        batches.append(batch.tolist())
    return counts, batches


def per_pair_normalized_mse(
    candidate_outputs: Tensor,
    targets: dict[str, Tensor],
    source_outputs: Tensor,
    scales: dict[int, float],
    *,
    start_step: int,
) -> np.ndarray:
    correction_error = (
        candidate_outputs[start_step:, :, 3]
        - source_outputs[start_step:, :, 3]
        - targets["correction"]
    )
    groups = (2 * targets["window"][:, None] + targets["mass_half"][None]).expand_as(
        correction_error
    )
    losses = []
    for pair in range(candidate_outputs.shape[1] // 2):
        episodes = slice(2 * pair, 2 * pair + 2)
        pair_losses = [
            (correction_error[:, episodes][groups[:, episodes] == group] / scales[group])
            .square()
            .mean()
            for group in range(6)
        ]
        losses.append(torch.stack(pair_losses).mean())
    return torch.stack(losses).detach().cpu().numpy().astype(np.float64, copy=False)


def bootstrap_gap(
    training_losses: np.ndarray,
    validation_losses: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    training_draws = generator.choice(
        training_losses, size=(replicates, len(training_losses)), replace=True
    ).mean(axis=1)
    validation_draws = generator.choice(
        validation_losses, size=(replicates, len(validation_losses)), replace=True
    ).mean(axis=1)
    gaps = validation_draws - training_draws
    return {
        "replicates": replicates,
        "seed": seed,
        "observed_validation_minus_training_normalized_mse": float(
            validation_losses.mean() - training_losses.mean()
        ),
        "mean": float(gaps.mean()),
        "confidence_95": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }


def split_metrics(
    outputs: Tensor,
    source_outputs: Tensor,
    targets: dict[str, Tensor],
    episode_ids: Tensor,
    scales: dict[int, float],
    *,
    start_step: int,
    constant: float,
) -> dict[str, Any]:
    samples = metric_samples(targets, episode_ids)
    constant_metrics = metric_summary(
        np.full(len(samples["target"]), constant),
        samples,
        scales,
        constant_aggregate_rmse=None,
    )
    prediction = (
        outputs[start_step:, episode_ids, 3] - source_outputs[start_step:, episode_ids, 3]
    ).reshape(-1)
    metrics = metric_summary(
        prediction.detach().cpu().numpy().astype(np.float64, copy=False),
        samples,
        scales,
        constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
    )
    metrics["aggregate_equal_group_normalized_mse"] = metrics["aggregate_equal_group_nrmse"] ** 2
    return {"metrics": metrics, "constant_baseline": constant_metrics}


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    recurrent_report = json.loads(args.recurrent_report.read_text())
    recorded_mask = json.loads(args.recurrent_mask.read_text())
    recorded_selection = json.loads(args.recurrent_magnitudes.read_text())
    prerequisite_checks = {
        "graph_hash_matches": recurrent_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": recurrent_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "routing_report_hash_matches": recurrent_report.get("routing_report_sha256")
        == file_sha256(args.routing_report),
        "routing_readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": recurrent_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "mask_hash_matches": recurrent_report.get("preregistered_mask_sha256")
        == file_sha256(args.recurrent_mask),
        "selection_hash_matches": recurrent_report.get("selected_magnitudes_sha256")
        == file_sha256(args.recurrent_magnitudes),
        "midpoint_was_stopped": recurrent_report.get("midpoint_stopped") is True,
        "fresh_heldout_was_not_consumed": recurrent_report.get("final_fresh_replay") is None,
        "source_was_preserved": recurrent_report.get("source_preserved") is True,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"sampling audit prerequisite failed: {prerequisite_checks}")

    controller, _checkpoint, hover_config, _gate_config, _resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    readout_spec = make_readout_spec(controller)
    spec = make_recurrent_routing_spec(controller, path_spec, readout_spec)
    if recorded_mask.get("selected_edge_indices") != spec.selected_edges.cpu().tolist():
        raise SystemExit("recorded mask does not match reconstructed graph mask")
    if recorded_selection.get("selected_edge_indices") != spec.selected_edges.cpu().tolist():
        raise SystemExit("selected magnitude order does not match reconstructed graph mask")
    if recorded_selection.get("selected_update") != 100:
        raise SystemExit("sampling audit requires the retained update-100 vector")
    selected_magnitudes = torch.tensor(
        recorded_selection["selected_magnitudes"], device=device, dtype=torch.float32
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit("output directory contains a stale report")
    seed_everything(args.bootstrap_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    saved = torch.load(args.replay_histories, map_location="cpu", weights_only=False)
    history = saved["development"]
    del saved
    sensors = {name: value.to(device) for name, value in history["sensors"].items()}
    start_step = round(args.takeover_seconds / hover_config.dt)
    anchor_steps = round(args.anchor_seconds / hover_config.dt)
    targets = prepare_history_targets(history, start_step=start_step, device=device)
    all_episode_ids = torch.arange(history["episodes"], device=device)
    with torch.no_grad():
        source_outputs = recurrent_motor_replay(controller, sensors, all_episode_ids)
        candidate_outputs = recurrent_motor_replay(
            controller,
            sensors,
            all_episode_ids,
            selected_edges=spec.selected_edges,
            selected_magnitudes=selected_magnitudes,
        )
    source_difference = float(
        (source_outputs[start_step:, :, 3] - targets["source_throttle"]).abs().max()
    )
    if source_difference > 1.0e-6:
        raise RuntimeError("source replay does not match saved source throttle")
    scale_samples = metric_samples(targets, all_episode_ids)
    scales = target_scales(scale_samples, floor=args.normalization_floor)
    constant = float(routing_readouts["constant_residual"])
    fit_pairs, validation_pairs = balanced_pair_split()
    fit_episode_ids = pair_episodes(fit_pairs, device=device)
    validation_episode_ids = pair_episodes(validation_pairs, device=device)
    split_report = {}
    for split, episode_ids in (
        ("training", fit_episode_ids),
        ("validation", validation_episode_ids),
    ):
        split_report[split] = {
            "source": split_metrics(
                source_outputs,
                source_outputs,
                targets,
                episode_ids,
                scales,
                start_step=start_step,
                constant=constant,
            ),
            "candidate": split_metrics(
                candidate_outputs,
                source_outputs,
                targets,
                episode_ids,
                scales,
                start_step=start_step,
                constant=constant,
            ),
        }

    pair_losses = per_pair_normalized_mse(
        candidate_outputs,
        targets,
        source_outputs,
        scales,
        start_step=start_step,
    )
    exposure_counts, batches = recover_pair_exposures(
        updates=args.updates,
        minibatch_pairs=args.minibatch_pairs,
        seed=args.optimizer_seed,
    )
    uniform_training_mse = float(pair_losses[fit_pairs].mean())
    exposure_weighted_mse = float(np.sum(pair_losses * exposure_counts) / exposure_counts.sum())
    distortion = abs(exposure_weighted_mse - uniform_training_mse) / max(
        uniform_training_mse, 1.0e-12
    )
    bootstrap = bootstrap_gap(
        pair_losses[fit_pairs],
        pair_losses[validation_pairs],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    training_nrmse = split_report["training"]["candidate"]["metrics"]["aggregate_equal_group_nrmse"]
    validation_nrmse = split_report["validation"]["candidate"]["metrics"][
        "aggregate_equal_group_nrmse"
    ]
    generalization_gap = bool(
        validation_nrmse >= args.generalization_ratio * training_nrmse
        and bootstrap["confidence_95"][0] > 0.0
    )
    sampling_distortion = bool(distortion > args.sampling_distortion_fraction)
    if generalization_gap and sampling_distortion:
        interpretation = "generalization_gap_and_sampling_distortion"
    elif generalization_gap:
        interpretation = "generalization_gap_established"
    elif sampling_distortion:
        interpretation = "sampling_distortion_flagged_without_generalization_gate"
    else:
        interpretation = "unresolved_early_window_fitting_error"
    prefix_rmse = float(
        (candidate_outputs[:anchor_steps] - source_outputs[:anchor_steps]).square().mean().sqrt()
    )
    report = {
        "method": "frozen recurrent-routing sampling and generalization audit",
        "claim_scope": (
            "This audit evaluates source and retained update-100 weights without training, "
            "fresh held-out data, flight, topology expansion, compilation, or promotion."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "recurrent_report": stable_path(args.recurrent_report),
        "recurrent_report_sha256": file_sha256(args.recurrent_report),
        "replay_histories": stable_path(args.replay_histories),
        "replay_histories_sha256": file_sha256(args.replay_histories),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "training_pairs": fit_pairs.tolist(),
            "validation_pairs": validation_pairs.tolist(),
            "full_native_replay_from_initialization": True,
            "optimizer_updates_performed": 0,
            "bootstrap_unit": "geometry pair",
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "source_replay_parity": {
            "maximum_throttle_absolute_difference": source_difference,
            "threshold": 1.0e-6,
            "passed": True,
        },
        "prefix_motor_rmse": prefix_rmse,
        "split_metrics": split_report,
        "pair_sampling": {
            "exposure_counts": exposure_counts.tolist(),
            "batches": batches,
            "total_pair_exposures": int(exposure_counts.sum()),
            "uniform_full_training_normalized_mse": uniform_training_mse,
            "exposure_weighted_training_normalized_mse": exposure_weighted_mse,
            "absolute_relative_difference": distortion,
            "distortion_threshold": args.sampling_distortion_fraction,
            "sampling_distortion_flagged": sampling_distortion,
        },
        "per_pair_candidate_normalized_mse": pair_losses.tolist(),
        "bootstrap_validation_minus_training": bootstrap,
        "classification": {
            "generalization_nrmse_ratio": validation_nrmse / max(training_nrmse, 1.0e-12),
            "generalization_gap_established": generalization_gap,
            "sampling_distortion_flagged": sampling_distortion,
            "interpretation": interpretation,
            "next_step": "stop after diagnostic; choose any training intervention separately",
        },
        "fresh_heldout_consumed": False,
        "flight_run": False,
        "source_preserved": True,
        "compiled_or_promoted": False,
        "candidate_checkpoint": None,
        "goal_passed": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "training_aggregate_nrmse": training_nrmse,
                "validation_aggregate_nrmse": validation_nrmse,
                "bootstrap_gap_95": bootstrap["confidence_95"],
                "sampling_distortion_fraction": distortion,
                "interpretation": interpretation,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
