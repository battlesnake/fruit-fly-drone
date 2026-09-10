#!/usr/bin/env python3
"""Compare longer training with early-window weighting on native recurrent routing."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_throttle_readout_ceiling import (  # noqa: E402
    compact_flight,
    evaluate_assisted_flights,
    flight_progress,
)
from audit_gate_throttle_routing import (  # noqa: E402
    collect_history,
    make_readout_spec,
    target_scales,
)
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_motor_interface_es import load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_recurrent_routing import (  # noqa: E402
    balanced_pair_split,
    evaluate_recurrent_snapshot,
    make_recurrent_routing_spec,
    metric_samples,
    pair_episodes,
    prepare_history_targets,
    public_validation,
    recurrent_motor_replay,
    validation_selection,
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
        "--sampling-report",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-sampling-diagnostic-v1"
            / "report.json"
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
        default=REPO_ROOT / "runs" / "gate" / "recurrent-routing-continuation-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--additional-updates", type=int, default=200)
    parser.add_argument("--validation-updates", type=int, nargs=3, default=(50, 100, 200))
    parser.add_argument("--continuation-gate-update", type=int, default=100)
    parser.add_argument("--minibatch-pairs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--early-window-weight", type=float, default=4.0)
    parser.add_argument("--other-window-weight", type=float, default=1.0)
    parser.add_argument("--minimum-early-improvement", type=float, default=0.20)
    parser.add_argument("--maximum-other-group-regression", type=float, default=0.05)
    parser.add_argument("--maximum-prefix-rmse", type=float, default=0.01)
    parser.add_argument("--anchor-scale", type=float, default=0.01)
    parser.add_argument("--anchor-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--seconds", type=float, default=1.50)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--anchor-seconds", type=float, default=0.20)
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--flight-episodes", type=int, default=256)
    parser.add_argument("--flight-seconds", type=float, default=12.0)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--continuation-seed", type=int, default=1_050_031)
    parser.add_argument("--held-out-seed", type=int, default=1_047_031)
    parser.add_argument("--flight-seed", type=int, default=1_048_031)
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
        args.sampling_report,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "additional_updates": (args.additional_updates, 200),
        "validation_updates": (tuple(args.validation_updates), (50, 100, 200)),
        "continuation_gate_update": (args.continuation_gate_update, 100),
        "minibatch_pairs": (args.minibatch_pairs, 8),
        "learning_rate": (args.learning_rate, 0.003),
        "gradient_clip_norm": (args.gradient_clip_norm, 1.0),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 8.0),
        "early_window_weight": (args.early_window_weight, 4.0),
        "other_window_weight": (args.other_window_weight, 1.0),
        "minimum_early_improvement": (args.minimum_early_improvement, 0.20),
        "maximum_other_group_regression": (args.maximum_other_group_regression, 0.05),
        "maximum_prefix_rmse": (args.maximum_prefix_rmse, 0.01),
        "anchor_scale": (args.anchor_scale, 0.01),
        "anchor_loss_coefficient": (args.anchor_loss_coefficient, 1.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
        "seconds": (args.seconds, 1.50),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "anchor_seconds": (args.anchor_seconds, 0.20),
        "held_out_pairs": (args.held_out_pairs, 128),
        "flight_episodes": (args.flight_episodes, 256),
        "flight_seconds": (args.flight_seconds, 12.0),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered continuation values changed: {', '.join(wrong)}")


def make_batch_schedule(*, updates: int, pairs_per_batch: int, seed: int) -> list[np.ndarray]:
    fit_pairs, _ = balanced_pair_split()
    first_half = fit_pairs[: len(fit_pairs) // 2]
    second_half = fit_pairs[len(fit_pairs) // 2 :]
    generator = np.random.default_rng(seed)
    schedule = []
    for _ in range(updates):
        batch = np.concatenate(
            (
                generator.choice(first_half, pairs_per_batch // 2, replace=False),
                generator.choice(second_half, pairs_per_batch // 2, replace=False),
            )
        )
        generator.shuffle(batch)
        schedule.append(batch)
    return schedule


def weighted_window_loss(group_losses: Sequence[Tensor], weights: Sequence[float]) -> Tensor:
    if len(group_losses) != 6 or len(weights) != 3:
        raise ValueError("window loss requires six groups and three window weights")
    window_losses = torch.stack(
        (
            (group_losses[0] + group_losses[1]) / 2.0,
            (group_losses[2] + group_losses[3]) / 2.0,
            (group_losses[4] + group_losses[5]) / 2.0,
        )
    )
    normalized = torch.as_tensor(weights, device=window_losses.device, dtype=window_losses.dtype)
    normalized = normalized / normalized.sum()
    return torch.sum(window_losses * normalized)


def continuation_loss(
    controller: Any,
    sensors: dict[str, Tensor],
    targets: dict[str, Tensor],
    source_outputs: Tensor,
    episode_ids: Tensor,
    selected_edges: Tensor,
    selected_magnitudes: Tensor,
    scales: dict[int, float],
    *,
    start_step: int,
    anchor_steps: int,
    anchor_scale: float,
    anchor_loss_coefficient: float,
    window_weights: Sequence[float],
) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = recurrent_motor_replay(
        controller,
        sensors,
        episode_ids,
        selected_edges=selected_edges,
        selected_magnitudes=selected_magnitudes,
    )
    error = outputs[start_step:, :, 3] - targets["reserve_motor"][:, episode_ids]
    groups = (2 * targets["window"][:, None] + targets["mass_half"][episode_ids][None]).expand_as(
        error
    )
    group_losses = [(error[groups == group] / scales[group]).square().mean() for group in range(6)]
    throttle_loss = weighted_window_loss(group_losses, window_weights)
    prefix_error = outputs[:anchor_steps] - source_outputs[:anchor_steps, episode_ids]
    prefix_rmse = prefix_error.square().mean().sqrt()
    anchor_loss = (prefix_error / anchor_scale).square().mean()
    return throttle_loss + anchor_loss_coefficient * anchor_loss, {
        "throttle_normalized_mse": throttle_loss.detach(),
        "prefix_motor_rmse": prefix_rmse.detach(),
    }


def continuation_gate(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    *,
    minimum_early_improvement: float,
    maximum_other_regression: float,
    prefix_rmse: float,
    maximum_prefix_rmse: float,
) -> dict[str, Any]:
    early = ("0.50:0.75/light", "0.50:0.75/heavy")
    other = (
        "0.75:1.00/light",
        "0.75:1.00/heavy",
        "1.00:1.50/light",
        "1.00:1.50/heavy",
    )
    current_early = max(metrics["per_window_and_mass"][name]["nrmse"] for name in early)
    baseline_early = max(baseline_metrics["per_window_and_mass"][name]["nrmse"] for name in early)
    checks = {
        "worst_early_group_improves_at_least_twenty_percent": (
            current_early <= (1.0 - minimum_early_improvement) * baseline_early
        ),
        "no_middle_or_late_group_regresses_over_limit": all(
            metrics["per_window_and_mass"][name]["nrmse"]
            <= baseline_metrics["per_window_and_mass"][name]["nrmse"] + maximum_other_regression
            for name in other
        ),
        "prefix_rmse_within_limit": prefix_rmse <= maximum_prefix_rmse,
    }
    return {
        "baseline_worst_early_group_nrmse": baseline_early,
        "candidate_worst_early_group_nrmse": current_early,
        "checks": checks,
        "passed": all(checks.values()),
    }


def train_arm(
    name: str,
    window_weights: Sequence[float],
    initial_magnitudes: Tensor,
    schedule: list[np.ndarray],
    controller: Any,
    sensors: dict[str, Tensor],
    targets: dict[str, Tensor],
    source_outputs: Tensor,
    selected_edges: Tensor,
    validation_episode_ids: Tensor,
    scales: dict[int, float],
    constant: float,
    baseline_metrics: dict[str, Any],
    args: argparse.Namespace,
    *,
    start_step: int,
    anchor_steps: int,
) -> dict[str, Any]:
    magnitudes = torch.nn.Parameter(initial_magnitudes.detach().clone())
    optimizer = torch.optim.Adam((magnitudes,), lr=args.learning_rate)
    history = []
    validations = []
    stopped_at = None
    for update, batch_pairs in enumerate(schedule, start=1):
        episode_ids = pair_episodes(batch_pairs, device=magnitudes.device)
        loss, components = continuation_loss(
            controller,
            sensors,
            targets,
            source_outputs,
            episode_ids,
            selected_edges,
            magnitudes,
            scales,
            start_step=start_step,
            anchor_steps=anchor_steps,
            anchor_scale=args.anchor_scale,
            anchor_loss_coefficient=args.anchor_loss_coefficient,
            window_weights=window_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_((magnitudes,), args.gradient_clip_norm)
        optimizer.step()
        with torch.no_grad():
            magnitudes.clamp_(0.0, args.maximum_edge_magnitude)
        if update == 1 or update % 20 == 0:
            record = {
                "additional_update": update,
                "loss": float(loss.detach()),
                "throttle_normalized_mse": float(components["throttle_normalized_mse"]),
                "prefix_motor_rmse": float(components["prefix_motor_rmse"]),
                "gradient_norm_before_clip": float(gradient_norm),
                "magnitude_minimum": float(magnitudes.detach().min()),
                "magnitude_maximum": float(magnitudes.detach().max()),
            }
            history.append(record)
            print(json.dumps({"phase": "training", "arm": name, **record}), flush=True)
        if update in args.validation_updates:
            validation = evaluate_recurrent_snapshot(
                controller,
                sensors,
                targets,
                source_outputs,
                validation_episode_ids,
                selected_edges,
                magnitudes.detach(),
                scales,
                start_step=start_step,
                anchor_steps=anchor_steps,
                constant=constant,
                maximum_group_nrmse=args.maximum_group_nrmse,
                minimum_constant_improvement=args.minimum_constant_improvement,
            )
            validation.update(
                {
                    "arm": name,
                    "additional_update": update,
                    "update": 100 + update,
                    "_magnitudes": magnitudes.detach().cpu().clone(),
                }
            )
            if update == args.continuation_gate_update:
                validation["continuation_gate"] = continuation_gate(
                    validation["metrics"],
                    baseline_metrics,
                    minimum_early_improvement=args.minimum_early_improvement,
                    maximum_other_regression=args.maximum_other_group_regression,
                    prefix_rmse=validation["prefix_motor_rmse"],
                    maximum_prefix_rmse=args.maximum_prefix_rmse,
                )
            validations.append(validation)
            print(
                json.dumps(
                    {
                        "phase": "validation",
                        "arm": name,
                        "additional_update": update,
                        "maximum_group_nrmse": validation["metrics"]["maximum_group_nrmse"],
                        "constant_improvement": validation["metrics"][
                            "rmse_improvement_over_constant"
                        ],
                        "prefix_motor_rmse": validation["prefix_motor_rmse"],
                        "pass_gate": validation["pass_gate"]["passed"],
                        "continuation_gate": validation.get("continuation_gate"),
                    }
                ),
                flush=True,
            )
            if (
                update == args.continuation_gate_update
                and not validation["continuation_gate"]["passed"]
            ):
                stopped_at = update
                break
    return {
        "name": name,
        "window_weights": list(window_weights),
        "training_history": history,
        "validations": validations,
        "stopped_at_additional_update": stopped_at,
        "_final_magnitudes": magnitudes.detach().cpu().clone(),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    recurrent_report = json.loads(args.recurrent_report.read_text())
    recurrent_mask = json.loads(args.recurrent_mask.read_text())
    recurrent_selection = json.loads(args.recurrent_magnitudes.read_text())
    sampling_report = json.loads(args.sampling_report.read_text())
    prerequisite_checks = {
        "graph_hash_matches": recurrent_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": recurrent_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "routing_readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": recurrent_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "mask_hash_matches": recurrent_report.get("preregistered_mask_sha256")
        == file_sha256(args.recurrent_mask),
        "selection_hash_matches": recurrent_report.get("selected_magnitudes_sha256")
        == file_sha256(args.recurrent_magnitudes),
        "sampling_found_no_gap": sampling_report.get("classification", {}).get(
            "generalization_gap_established"
        )
        is False,
        "sampling_found_no_distortion": sampling_report.get("classification", {}).get(
            "sampling_distortion_flagged"
        )
        is False,
        "fresh_heldout_not_previously_consumed": recurrent_report.get("final_fresh_replay") is None,
        "source_was_preserved": recurrent_report.get("source_preserved") is True,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"continuation prerequisite failed: {prerequisite_checks}")

    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=args.maximum_edge_magnitude,
    )
    path_nodes = torch.unique(
        torch.cat((controller.edge_pre[path_spec.edges], controller.edge_post[path_spec.edges]))
    )
    readout_spec = make_readout_spec(controller)
    spec = make_recurrent_routing_spec(controller, path_spec, readout_spec)
    if recurrent_mask.get("selected_edge_indices") != spec.selected_edges.cpu().tolist():
        raise SystemExit("recorded mask differs from reconstructed graph mask")
    if recurrent_selection.get("selected_edge_indices") != spec.selected_edges.cpu().tolist():
        raise SystemExit("update-100 selection differs from reconstructed graph mask")
    initial_magnitudes = torch.tensor(
        recurrent_selection["selected_magnitudes"], device=device, dtype=torch.float32
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    arm_path = args.output_dir / "arm-magnitudes.json"
    selection_path = args.output_dir / "selected-magnitudes.json"
    if report_path.exists() or arm_path.exists() or selection_path.exists():
        raise SystemExit("output directory contains a stale report or selection")
    seed_everything(args.continuation_seed)
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
    source_difference = float(
        (source_outputs[start_step:, :, 3] - targets["source_throttle"]).abs().max()
    )
    if source_difference > 1.0e-6:
        raise RuntimeError("source replay does not match saved source throttle")
    scales = target_scales(metric_samples(targets, all_episode_ids), floor=args.normalization_floor)
    constant = float(routing_readouts["constant_residual"])
    _, validation_pairs = balanced_pair_split()
    validation_episode_ids = pair_episodes(validation_pairs, device=device)
    initial_validation = evaluate_recurrent_snapshot(
        controller,
        sensors,
        targets,
        source_outputs,
        validation_episode_ids,
        spec.selected_edges,
        initial_magnitudes,
        scales,
        start_step=start_step,
        anchor_steps=anchor_steps,
        constant=constant,
        maximum_group_nrmse=args.maximum_group_nrmse,
        minimum_constant_improvement=args.minimum_constant_improvement,
    )
    reported_initial = recurrent_report["selected_validation"]["metrics"]
    if (
        abs(
            initial_validation["metrics"]["maximum_group_nrmse"]
            - reported_initial["maximum_group_nrmse"]
        )
        > 1.0e-6
    ):
        raise RuntimeError("retained update-100 validation replay does not match its report")

    schedule = make_batch_schedule(
        updates=args.additional_updates,
        pairs_per_batch=args.minibatch_pairs,
        seed=args.continuation_seed,
    )
    arms = [
        train_arm(
            "control_equal_window",
            (1.0, 1.0, 1.0),
            initial_magnitudes,
            schedule,
            controller,
            sensors,
            targets,
            source_outputs,
            spec.selected_edges,
            validation_episode_ids,
            scales,
            constant,
            initial_validation["metrics"],
            args,
            start_step=start_step,
            anchor_steps=anchor_steps,
        ),
        train_arm(
            "treatment_early_fourfold",
            (
                args.early_window_weight,
                args.other_window_weight,
                args.other_window_weight,
            ),
            initial_magnitudes,
            schedule,
            controller,
            sensors,
            targets,
            source_outputs,
            spec.selected_edges,
            validation_episode_ids,
            scales,
            constant,
            initial_validation["metrics"],
            args,
            start_step=start_step,
            anchor_steps=anchor_steps,
        ),
    ]
    validations = [item for arm in arms for item in arm["validations"]]
    selected_validation, selection_rule = validation_selection(
        validations, maximum_prefix_rmse=args.maximum_prefix_rmse
    )
    selected_magnitudes = selected_validation["_magnitudes"].to(device)
    validation_passed = bool(selected_validation["pass_gate"]["passed"])
    final_replay = None
    replay_passed = False
    assisted_flight = None
    flight_passed = False
    held_source_parity = None
    if validation_passed:
        held_history = collect_history(
            controller,
            path_nodes,
            readout_spec,
            pairs=args.held_out_pairs,
            seed=args.held_out_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seconds=args.seconds,
            takeover_seconds=args.takeover_seconds,
        )
        held_sensors = {name: value.to(device) for name, value in held_history["sensors"].items()}
        held_targets = prepare_history_targets(held_history, start_step=start_step, device=device)
        held_episode_ids = torch.arange(held_history["episodes"], device=device)
        with torch.no_grad():
            held_source_outputs = recurrent_motor_replay(controller, held_sensors, held_episode_ids)
        difference = float(
            (held_source_outputs[start_step:, :, 3] - held_targets["source_throttle"]).abs().max()
        )
        held_source_parity = {
            "maximum_throttle_absolute_difference": difference,
            "threshold": 1.0e-6,
            "passed": difference <= 1.0e-6,
        }
        if not held_source_parity["passed"]:
            raise RuntimeError("fresh source replay does not match saved source throttle")
        final_replay = evaluate_recurrent_snapshot(
            controller,
            held_sensors,
            held_targets,
            held_source_outputs,
            held_episode_ids,
            spec.selected_edges,
            selected_magnitudes,
            scales,
            start_step=start_step,
            anchor_steps=anchor_steps,
            constant=constant,
            maximum_group_nrmse=args.maximum_group_nrmse,
            minimum_constant_improvement=args.minimum_constant_improvement,
        )
        replay_passed = bool(final_replay["pass_gate"]["passed"])
        print(
            json.dumps(
                {
                    "phase": "fresh_recurrent_replay",
                    "arm": selected_validation["arm"],
                    "maximum_group_nrmse": final_replay["metrics"]["maximum_group_nrmse"],
                    "constant_improvement": final_replay["metrics"][
                        "rmse_improvement_over_constant"
                    ],
                    "passed": replay_passed,
                }
            ),
            flush=True,
        )
        if replay_passed:
            candidate = copy.deepcopy(controller).to(device)
            with torch.no_grad():
                candidate.edge_magnitude[spec.selected_edges] = selected_magnitudes
            assisted_flight = evaluate_assisted_flights(
                controller,
                candidate,
                episodes=args.flight_episodes,
                seed=args.flight_seed,
                takeover_seconds=args.takeover_seconds,
                seconds=args.flight_seconds,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            flight_passed = flight_progress(
                assisted_flight["source"],
                assisted_flight["candidate"],
                minimum_light_gain=args.minimum_light_gain,
                minimum_floor_gain=args.minimum_floor_gain,
                maximum_drop=args.maximum_stratum_drop,
            )
            print(
                json.dumps(
                    {
                        "phase": "assisted_flight",
                        "source": compact_flight(assisted_flight["source"]),
                        "candidate": compact_flight(assisted_flight["candidate"]),
                        "passed": flight_passed,
                    }
                ),
                flush=True,
            )

    arm_validation_passes = {
        arm["name"]: any(item["pass_gate"]["passed"] for item in arm["validations"]) for arm in arms
    }
    if not validation_passed:
        interpretation = "neither_continuation_arm_passed_development"
    elif arm_validation_passes["control_equal_window"]:
        interpretation = "additional_training_sufficient_on_development"
    else:
        interpretation = "early_window_weighting_required_on_development"
    arm_path.write_text(
        json.dumps(
            {
                "selected_edge_indices": spec.selected_edges.detach().cpu().tolist(),
                "source_update": 100,
                "arms": {
                    arm["name"]: {
                        "additional_update": arm["stopped_at_additional_update"]
                        or args.additional_updates,
                        "magnitudes": arm["_final_magnitudes"].tolist(),
                    }
                    for arm in arms
                },
                "deterministic_reconstruction_only": True,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    selection_path.write_text(
        json.dumps(
            {
                "selected_arm": selected_validation["arm"],
                "selected_additional_update": selected_validation["additional_update"],
                "selected_edge_indices": spec.selected_edges.detach().cpu().tolist(),
                "selected_magnitudes": selected_magnitudes.detach().cpu().tolist(),
                "selection_rule": selection_rule,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "method": "paired native recurrent-routing continuation",
        "claim_scope": (
            "Control and early-weighted arms continue the same fixed 125-edge native circuit "
            "from update 100 with reset Adam state and identical batches. No result is merged "
            "or promoted."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "recurrent_report": stable_path(args.recurrent_report),
        "recurrent_report_sha256": file_sha256(args.recurrent_report),
        "sampling_report": stable_path(args.sampling_report),
        "sampling_report_sha256": file_sha256(args.sampling_report),
        "replay_histories": stable_path(args.replay_histories),
        "replay_histories_sha256": file_sha256(args.replay_histories),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "starts": "identical retained update-100 magnitudes; fresh Adam per arm",
            "batch_schedule_identical_between_arms": True,
            "control_window_weights": [1.0 / 3.0] * 3,
            "treatment_window_weights": [4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            "full_native_replay_from_initialization": True,
            "fresh_replay_only_after_development_pass": True,
            "flight_only_after_fresh_replay_pass": True,
            "edge_signs_fixed": True,
            "all_other_parameters_frozen": True,
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
        "initial_validation": initial_validation,
        "arms": [
            {
                **{
                    key: value
                    for key, value in arm.items()
                    if key not in {"validations", "_final_magnitudes"}
                },
                "validations": [public_validation(item) for item in arm["validations"]],
            }
            for arm in arms
        ],
        "arm_development_passes": arm_validation_passes,
        "selected_arm": selected_validation["arm"],
        "selected_additional_update": selected_validation["additional_update"],
        "selection_rule": selection_rule,
        "selected_validation": public_validation(selected_validation),
        "development_passed": validation_passed,
        "held_out_source_replay_parity": held_source_parity,
        "final_fresh_replay": final_replay,
        "fresh_replay_passed": replay_passed,
        "assisted_flight": assisted_flight,
        "assisted_flight_passed": flight_passed,
        "arm_magnitudes": stable_path(arm_path),
        "arm_magnitudes_sha256": file_sha256(arm_path),
        "selected_magnitudes": stable_path(selection_path),
        "selected_magnitudes_sha256": file_sha256(selection_path),
        "classification": {
            "interpretation": interpretation,
            "next_step": "stop both arms if neither passes; no topology widening or promotion",
        },
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
                "arm_development_passes": arm_validation_passes,
                "selected_arm": selected_validation["arm"],
                "development_passed": validation_passed,
                "fresh_replay_passed": replay_passed,
                "assisted_flight_passed": flight_passed,
                "interpretation": interpretation,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if replay_passed and flight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
