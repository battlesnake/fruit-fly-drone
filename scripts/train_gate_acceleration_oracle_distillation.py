#!/usr/bin/env python3
"""Distill a training-only mass oracle into the anatomical accelerometer pathway."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from search_gate_acceleration_es import (  # noqa: E402
    acceleration_path_edges,
    compact_metrics,
)
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    seed_everything,
)
from train_gate_mass_current_distillation import (  # noqa: E402
    label_mass,
    paired_launch,
    stable_path,
    target_bias,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--baseline-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "acceleration-oracle-distillation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--pair-loss-weight", type=float, default=2.0)
    parser.add_argument("--other-axis-weight", type=float, default=0.25)
    parser.add_argument("--delta-scale", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--selection-interval", type=int, default=10)
    parser.add_argument("--selection-horizon-seconds", type=float, default=0.75)
    parser.add_argument("--selection-episodes", type=int, default=512)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=400_031)
    parser.add_argument("--selection-seed", type=int, default=410_031)
    parser.add_argument("--final-seed", type=int, default=420_031)
    parser.add_argument(
        "--label-mode",
        choices=("true", "pair_swapped", "constant"),
        default="true",
        help="Training-only causal control for the oracle label.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.baseline_graph, args.checkpoint, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.updates,
        args.batch_size,
        args.learning_rate,
        args.time_constant_learning_rate,
        args.pair_loss_weight,
        args.other_axis_weight,
        args.delta_scale,
        args.gradient_clip,
        args.target_onset_seconds,
        args.prefix_seconds,
        args.selection_interval,
        args.selection_horizon_seconds,
        args.selection_episodes,
        args.final_episodes,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, times, rates, and weights must be positive")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even for matched-mass pairs")
    for name in ("selection_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")
    if args.target_onset_seconds >= args.prefix_seconds:
        raise SystemExit("target onset must occur inside the training prefix")
    if args.selection_horizon_seconds not in (0.5, 0.75, args.prefix_seconds):
        raise SystemExit("selection horizon must be 0.5, 0.75, or the prefix duration")
    if round(args.target_onset_seconds / dt) < 1:
        raise SystemExit("target onset must leave a non-oracle launch interval")


def register_parameter_masks(
    controller: ConnectomeController,
    edges: Tensor,
    nodes: Tensor,
) -> tuple[Tensor, Tensor]:
    controller.edge_magnitude.requires_grad_(True)
    controller.raw_time_constant.requires_grad_(True)
    edge_mask = torch.zeros_like(controller.edge_magnitude)
    edge_mask[edges] = 1.0
    node_mask = torch.zeros_like(controller.raw_time_constant)
    node_mask[nodes] = 1.0
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    controller.raw_time_constant.register_hook(lambda gradient: gradient * node_mask)
    controller.bias.requires_grad_(False)
    return edge_mask, node_mask


def delta_metrics(prediction: Tensor, target: Tensor) -> dict[str, float | None]:
    prediction = prediction.detach()
    target = target.detach()
    error = prediction - target
    target_rms = target.square().mean().sqrt()
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    target_variance = centered_target.square().mean()
    prediction_variance = centered_prediction.square().mean()
    if float(target_variance) <= 1.0e-12:
        slope = None
        correlation = None
    else:
        covariance = (centered_prediction * centered_target).mean()
        slope = float(covariance / target_variance)
        correlation = float(
            covariance / (prediction_variance.sqrt() * target_variance.sqrt()).clamp_min(1.0e-12)
        )
    return {
        "rmse": float(error.square().mean().sqrt()),
        "normalized_rmse": float(error.square().mean().sqrt() / target_rms.clamp_min(1.0e-6)),
        "signed_slope": slope,
        "pearson_correlation": correlation,
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "prediction_std": float(prediction.std()),
        "target_std": float(target.std()),
    }


def prefix_rollout(
    student: ConnectomeController,
    reference: ConnectomeController,
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    target_onset_seconds: float,
    delta_scale: float,
    pair_loss_weight: float,
    other_axis_weight: float,
    label_mode: str = "true",
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
) -> tuple[Tensor, dict[str, dict[str, float | None]]]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    student_neural = student.initial_state(episodes, device=device, dtype=torch.float32)
    reference_neural = reference.initial_state(episodes, device=device, dtype=torch.float32)
    oracle_neural = reference.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    oracle_mass = label_mass(normalized_mass, label_mode)
    oracle_bias = target_bias(oracle_mass, calibration)
    steps = round(prefix_seconds / hover_config.dt)
    onset_step = round(target_onset_seconds / hover_config.dt)
    requested_horizons = (0.10, 0.50, 0.75, prefix_seconds)
    horizons = tuple(sorted({value for value in requested_horizons if value <= prefix_seconds}))
    horizon_steps = {round(value / hover_config.dt): value for value in horizons}
    permutation = torch.cat(
        (
            torch.arange(episodes // 2, episodes, device=device),
            torch.arange(episodes // 2, device=device),
        )
    )
    losses: list[Tensor] = []
    results: dict[str, dict[str, float | None]] = {}
    for zero_step in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        specific_force = state.specific_force
        if frozen_acceleration:
            specific_force = torch.zeros_like(specific_force)
            specific_force[:, 2] = 9.81
        elif swapped_acceleration:
            specific_force = specific_force[permutation]
        student_motor, student_neural = student(
            image,
            state.euler[:, :2],
            student_neural,
            specific_force,
        )
        with torch.no_grad():
            reference_motor, reference_neural = reference(
                image,
                state.euler[:, :2],
                reference_neural,
                state.specific_force,
            )
            applied_bias = oracle_bias if zero_step >= onset_step else None
            oracle_motor, oracle_neural = reference(
                image,
                state.euler[:, :2],
                oracle_neural,
                state.specific_force,
                privileged_throttle_pool_bias=applied_bias,
            )
            rc, stick_state = sticks(reference_motor, stick_state)
            state = quad(rc, state, mass_scale)
        completed_step = zero_step + 1
        if completed_step not in horizon_steps:
            continue
        prediction = student_motor - reference_motor
        target = oracle_motor - reference_motor
        throttle_error = (prediction[:, 3] - target[:, 3]) / delta_scale
        half = episodes // 2
        pair_error = (
            (prediction[:half, 3] - prediction[half:, 3]) - (target[:half, 3] - target[half:, 3])
        ) / delta_scale
        other_axis_error = (student_motor[:, :3] - reference_motor[:, :3]) / delta_scale
        losses.append(
            throttle_error.square().mean()
            + pair_loss_weight * pair_error.square().mean()
            + other_axis_weight * other_axis_error.square().mean()
        )
        results[f"{horizon_steps[completed_step]:g}"] = {
            **delta_metrics(prediction[:, 3], target[:, 3]),
            "other_axis_rmse": float(
                (student_motor[:, :3] - reference_motor[:, :3]).square().mean().sqrt().detach()
            ),
        }
    return torch.stack(losses).mean(), results


def save_checkpoint(
    path: Path,
    student: ConnectomeController,
    source: dict[str, Any],
    args: argparse.Namespace,
    selected_update: int,
) -> None:
    torch.save(
        {
            **copy.deepcopy(source),
            "controller": {
                name: value.detach().cpu() for name, value in student.state_dict().items()
            },
            "source_checkpoint_sha256": file_sha256(args.checkpoint),
            "acceleration_oracle_distillation": {
                "label_mode": args.label_mode,
                "selected_update": selected_update,
                "target_onset_seconds": args.target_onset_seconds,
            },
        },
        path,
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    validate_args(args, hover_config.dt)
    resolution = int(checkpoint["image_resolution"])

    reference = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    reference.load_state_dict(checkpoint["controller"])
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    student = copy.deepcopy(reference)
    student.train()
    path_edges_np = acceleration_path_edges(args.graph, args.baseline_graph)
    path_edges = torch.from_numpy(path_edges_np).to(device=device)
    with np.load(args.graph) as graph, np.load(args.baseline_graph) as baseline_graph:
        added_node_np = ~np.isin(graph["node_ids"], baseline_graph["node_ids"])
    added_nodes = torch.from_numpy(np.flatnonzero(added_node_np)).to(device=device)
    edge_mask, node_mask = register_parameter_masks(student, path_edges, added_nodes)
    initial_edge = student.edge_magnitude.detach().clone()
    initial_time_constant = student.raw_time_constant.detach().clone()
    optimizer = torch.optim.Adam(
        (
            {"params": [student.edge_magnitude], "lr": args.learning_rate},
            {
                "params": [student.raw_time_constant],
                "lr": args.time_constant_learning_rate,
            },
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    progress_path = args.output_dir / "progress.json"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    started = perf_counter()
    print(
        json.dumps(
            {
                "phase": "setup",
                "trainable_acceleration_path_edges": len(path_edges),
                "trainable_acceleration_path_time_constants": len(added_nodes),
                "all_other_parameters_frozen": True,
            }
        ),
        flush=True,
    )

    for update in range(1, args.updates + 1):
        student.train()
        loss, training = prefix_rollout(
            student,
            reference,
            calibration,
            episodes=args.batch_size,
            seed=args.seed + update - 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            target_onset_seconds=args.target_onset_seconds,
            delta_scale=args.delta_scale,
            pair_loss_weight=args.pair_loss_weight,
            other_axis_weight=args.other_axis_weight,
            label_mode=args.label_mode,
        )
        regularization = (
            (
                (student.edge_magnitude[path_edges] - initial_edge[path_edges])
                / initial_edge[path_edges].clamp_min(0.05)
            )
            .square()
            .mean()
        )
        time_constant_regularization = (
            (student.raw_time_constant[added_nodes] - initial_time_constant[added_nodes])
            .square()
            .mean()
        )
        objective = loss + 1.0e-4 * regularization + 1.0e-5 * time_constant_regularization
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (student.edge_magnitude, student.raw_time_constant), args.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite distillation gradient")
        optimizer.step()
        student.project_parameters()
        should_select = update % args.selection_interval == 0 or update == args.updates
        if not should_select:
            if update == 1 or update % 5 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "update": update,
                            "loss": float(loss.detach()),
                            "gradient_norm": float(gradient_norm),
                            "horizons": training,
                        }
                    ),
                    flush=True,
                )
            continue
        student.eval()
        with torch.no_grad():
            _, probe = prefix_rollout(
                student,
                reference,
                calibration,
                episodes=args.selection_episodes,
                seed=args.selection_seed,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                prefix_seconds=args.prefix_seconds,
                target_onset_seconds=args.target_onset_seconds,
                delta_scale=args.delta_scale,
                pair_loss_weight=args.pair_loss_weight,
                other_axis_weight=args.other_axis_weight,
            )
        selected_horizon = probe[f"{args.selection_horizon_seconds:g}"]
        correlation = selected_horizon["pearson_correlation"]
        key = (
            -float(selected_horizon["normalized_rmse"]),
            float(correlation) if correlation is not None else -1.0,
            -float(probe[f"{args.prefix_seconds:g}"]["normalized_rmse"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(best_path, student, checkpoint, args, update)
        entry = {
            "update": update,
            "training_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "training": training,
            "held_out": probe,
            "best_key": best_key,
        }
        history.append(entry)
        print(json.dumps({"phase": "selection", **entry}), flush=True)
        progress_path.write_text(
            json.dumps({"history": history, "best_key": best_key}, indent=2, sort_keys=True) + "\n"
        )

    best = torch.load(best_path, map_location=device, weights_only=True)
    student.load_state_dict(best["controller"])
    student.eval()
    final_probe_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "pair_swapped_acceleration": {"swapped_acceleration": True},
    }
    final_probes: dict[str, Any] = {}
    with torch.no_grad():
        for name, controls in final_probe_specs.items():
            _, final_probes[name] = prefix_rollout(
                student,
                reference,
                calibration,
                episodes=args.selection_episodes,
                seed=args.final_seed + 1,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                prefix_seconds=args.prefix_seconds,
                target_onset_seconds=args.target_onset_seconds,
                delta_scale=args.delta_scale,
                pair_loss_weight=args.pair_loss_weight,
                other_axis_weight=args.other_axis_weight,
                **controls,
            )
    final_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "mass_rank_swapped_acceleration": {"swapped_acceleration": True},
        "frozen_first_frame": {"frozen_visual": True},
    }
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    final: dict[str, Any] = {}
    for name, controls in final_specs.items():
        final[name] = evaluate_gate(
            student,
            episodes=args.final_episodes,
            seconds=12.0,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
            **controls,
        )
        print(
            json.dumps(
                {"phase": "final", "name": name, **compact_metrics(final[name], clean_radius)}
            ),
            flush=True,
        )
    baseline = evaluate_gate(
        reference,
        episodes=args.final_episodes,
        seconds=12.0,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=args.final_seed,
        balanced_strata=True,
    )
    selected_horizon_key = f"{args.selection_horizon_seconds:g}"
    live_half = final_probes["live_sensors"][selected_horizon_key]
    control_half = (
        final_probes["constant_1g_acceleration"][selected_horizon_key],
        final_probes["pair_swapped_acceleration"][selected_horizon_key],
    )
    imitation_passed = (
        live_half["normalized_rmse"] <= 0.20
        and live_half["signed_slope"] is not None
        and abs(live_half["signed_slope"] - 1.0) <= 0.20
    )
    probe_causality = all(
        control["normalized_rmse"] >= live_half["normalized_rmse"] + 0.25
        for control in control_half
    )
    flight_causality = (
        final["live_sensors"]["success_rate"]
        >= max(
            final["constant_1g_acceleration"]["success_rate"],
            final["mass_rank_swapped_acceleration"]["success_rate"],
        )
        + 0.05
    )
    goal_passed = (
        final["live_sensors"]["goal_pass"]
        and imitation_passed
        and probe_causality
        and flight_causality
    )
    frozen_edge = ~edge_mask.bool()
    frozen_node = ~node_mask.bool()
    report = {
        "experiment": "oracle motor-effect distillation into anatomical acceleration paths",
        "claim_scope": (
            "The exact mass is a training label only. Deployment receives instantaneous FPV, "
            "roll/pitch, body-Z specific force, and recurrent connectome state."
        ),
        "counts_toward_direct_sensor_goal": bool(imitation_passed and probe_causality),
        "external_runtime_parameters_added": 0,
        "external_actor_state_machine": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "baseline_graph": stable_path(args.baseline_graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "candidate_checkpoint": stable_path(best_path),
        "candidate_checkpoint_sha256": file_sha256(best_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "training": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "matched_geometry_light_heavy_pairs": True,
            "oracle_corrected_training_physics": False,
            "trainable_acceleration_path_edges": len(path_edges),
            "trainable_acceleration_path_time_constants": len(added_nodes),
            "all_other_parameters_frozen": True,
        },
        "history": history,
        "selected_update": best["acceleration_oracle_distillation"]["selected_update"],
        "parameter_drift": {
            "frozen_edge_magnitude_max_absolute_change": float(
                (student.edge_magnitude[frozen_edge] - initial_edge[frozen_edge])
                .abs()
                .max()
                .detach()
            ),
            "trainable_edge_magnitude_max_absolute_change": float(
                (student.edge_magnitude[path_edges] - initial_edge[path_edges]).abs().max().detach()
            ),
            "frozen_time_constant_max_absolute_change": float(
                (student.raw_time_constant[frozen_node] - initial_time_constant[frozen_node])
                .abs()
                .max()
                .detach()
            ),
            "trainable_time_constant_max_absolute_change": float(
                (student.raw_time_constant[added_nodes] - initial_time_constant[added_nodes])
                .abs()
                .max()
                .detach()
            ),
        },
        "baseline": baseline,
        "final": final,
        "final_prefix_probes": final_probes,
        "causal_checks": {
            "trained_with_true_mass_labels": args.label_mode == "true",
            "held_out_oracle_imitation": imitation_passed,
            "prefix_sensor_causality": probe_causality,
            "flight_sensor_causality": flight_causality,
        },
        "goal_passed": goal_passed,
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "selected_update": report["selected_update"],
                "goal_passed": goal_passed,
                "baseline_success_rate": baseline["success_rate"],
                "live_success_rate": final["live_sensors"]["success_rate"],
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
