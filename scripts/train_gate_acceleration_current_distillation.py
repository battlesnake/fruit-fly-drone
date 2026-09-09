#!/usr/bin/env python3
"""Fit anatomical acceleration paths to the mass-oracle synaptic current."""

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_acceleration_es import (  # noqa: E402
    compact_metrics,
    edges_on_short_paths,
)
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    seed_everything,
)
from train_gate_acceleration_oracle_distillation import delta_metrics  # noqa: E402
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
        default=REPO_ROOT / "runs" / "gate" / "acceleration-current-distillation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix-seconds", type=float, default=1.50)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--edge-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--time-constant-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--direct-current-weight", type=float, default=1.0)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--other-axis-weight", type=float, default=0.25)
    parser.add_argument("--reset-loss-weight", type=float, default=0.25)
    parser.add_argument("--current-scale", type=float, default=0.10)
    parser.add_argument("--motor-scale", type=float, default=0.02)
    parser.add_argument("--sensor-gain", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--selection-interval", type=int, default=10)
    parser.add_argument("--selection-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=440_031)
    parser.add_argument("--selection-seed", type=int, default=450_031)
    parser.add_argument("--final-seed", type=int, default=460_031)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument(
        "--plasticity",
        choices=(
            "incident",
            "incident_return",
            "return_only",
            "boundary_return",
            "all_path",
        ),
        default="incident",
        help=(
            "Train edges incident to added sensor-path cells; optionally add only direct "
            "throttle-return edges; or train every edge on an <=max-hops path."
        ),
    )
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
        args.prefix_seconds,
        args.target_onset_seconds,
        args.edge_learning_rate,
        args.time_constant_learning_rate,
        args.direct_current_weight,
        args.pair_loss_weight,
        args.other_axis_weight,
        args.reset_loss_weight,
        args.current_scale,
        args.motor_scale,
        args.sensor_gain,
        args.gradient_clip,
        args.selection_interval,
        args.selection_episodes,
        args.final_episodes,
        args.max_hops,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, rates, scales, and times must be positive")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even for matched-mass pairs")
    for name in ("selection_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")
    if not 0.0 < args.target_onset_seconds < 0.50:
        raise SystemExit("target onset must be in (0, 0.5) for the fixed temporal buckets")
    if args.prefix_seconds < 1.0:
        raise SystemExit("prefix must reach at least 1 second")
    if round(args.target_onset_seconds / dt) < 1:
        raise SystemExit("target onset must leave a zero-current reset interval")


def path_plasticity(
    graph_path: Path,
    baseline_graph_path: Path,
    *,
    max_hops: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    with np.load(graph_path) as graph, np.load(baseline_graph_path) as baseline:
        node_ids = graph["node_ids"]
        edge_pre = graph["edge_pre"]
        edge_post = graph["edge_post"]
        pool_offsets = graph["output_pool_offsets"]
        pool_indices = graph["output_pool_indices"]
        throttle_nodes = pool_indices[pool_offsets[6] : pool_offsets[8]]
        path_mask = edges_on_short_paths(
            edge_pre,
            edge_post,
            len(node_ids),
            graph["acceleration_node_indices"],
            throttle_nodes,
            max_hops,
        )
        new_node_mask = ~np.isin(node_ids, baseline["node_ids"])
        direct_return_mask = np.isin(edge_post, throttle_nodes)
        boundary_nodes = np.unique(edge_post[new_node_mask[edge_pre] & ~new_node_mask[edge_post]])
        return_sources = np.unique(edge_pre[direct_return_mask])
        boundary_return_mask = np.isin(edge_pre, boundary_nodes) & np.isin(
            edge_post, return_sources
        )
        if mode in ("incident", "incident_return"):
            trainable_edge_mask = path_mask & (new_node_mask[edge_pre] | new_node_mask[edge_post])
            if mode == "incident_return":
                trainable_edge_mask |= direct_return_mask
            trainable_node_mask = new_node_mask
        elif mode == "return_only":
            trainable_edge_mask = direct_return_mask
            trainable_node_mask = np.zeros(len(node_ids), dtype=bool)
        elif mode == "boundary_return":
            trainable_edge_mask = boundary_return_mask | direct_return_mask
            trainable_node_mask = np.zeros(len(node_ids), dtype=bool)
            trainable_node_mask[return_sources] = True
        elif mode == "all_path":
            trainable_edge_mask = path_mask
            trainable_node_mask = np.zeros(len(node_ids), dtype=bool)
            trainable_node_mask[
                np.unique(np.concatenate((edge_pre[path_mask], edge_post[path_mask])))
            ] = True
        else:  # pragma: no cover - argparse owns the public choices
            raise ValueError(f"unknown plasticity mode: {mode}")
        counts = {
            "all_path_edges": int(path_mask.sum()),
            "all_path_nodes": int(
                len(np.unique(np.concatenate((edge_pre[path_mask], edge_post[path_mask]))))
            ),
            "added_nodes": int(new_node_mask.sum()),
            "new_presynaptic_path_edges": int((path_mask & new_node_mask[edge_pre]).sum()),
            "old_to_new_feedback_edges": int(
                (path_mask & ~new_node_mask[edge_pre] & new_node_mask[edge_post]).sum()
            ),
            "direct_throttle_return_edges": int(direct_return_mask.sum()),
            "boundary_to_return_edges": int(boundary_return_mask.sum()),
            "throttle_return_source_nodes": int(len(return_sources)),
        }
    return np.flatnonzero(trainable_edge_mask), np.flatnonzero(trainable_node_mask), counts


def register_parameter_masks(
    controller: ConnectomeController,
    edges: Tensor,
    nodes: Tensor,
) -> tuple[Tensor, Tensor]:
    controller.edge_magnitude.requires_grad_(True)
    controller.raw_time_constant.requires_grad_(True)
    controller.bias.requires_grad_(False)
    edge_mask = torch.zeros_like(controller.edge_magnitude)
    edge_mask[edges] = 1.0
    node_mask = torch.zeros_like(controller.raw_time_constant)
    node_mask[nodes] = 1.0
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    controller.raw_time_constant.register_hook(lambda gradient: gradient * node_mask)
    return edge_mask, node_mask


def preactivation_drive(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    state: Tensor,
    specific_force: Tensor,
) -> Tensor:
    activity = torch.tanh(state)
    messages = activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude
    recurrent = torch.zeros_like(state).index_add(1, controller.edge_post, messages)
    return (
        recurrent
        + controller.bias
        + controller.sensory_drive(
            image,
            roll_pitch,
            specific_force,
        )
    )


def throttle_spec(controller: ConnectomeController) -> tuple[Tensor, Tensor]:
    positive_begin = int(controller.pool_offsets[6].item())
    positive_end = int(controller.pool_offsets[7].item())
    negative_end = int(controller.pool_offsets[8].item())
    nodes = controller.pool_indices[positive_begin:negative_end]
    signs = torch.cat(
        (
            torch.ones(positive_end - positive_begin, device=nodes.device),
            -torch.ones(negative_end - positive_end, device=nodes.device),
        )
    )
    return nodes, signs


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float | None]:
    return delta_metrics(prediction, target)


def bucket_name(time_seconds: float, onset_seconds: float) -> str:
    if time_seconds < onset_seconds:
        return "reset"
    if time_seconds < 0.50:
        return "0.25:0.5"
    if time_seconds < 0.75:
        return "0.5:0.75"
    return "0.75:end"


def current_rollout(
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
    current_scale: float,
    motor_scale: float,
    direct_current_weight: float,
    pair_loss_weight: float,
    other_axis_weight: float,
    reset_loss_weight: float,
    sensor_gain: float,
    label_mode: str = "true",
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
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
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    oracle_mass = label_mass(normalized_mass, label_mode)
    desired_bias = target_bias(oracle_mass, calibration)
    throttle_nodes, motor_sign = throttle_spec(student)
    target_current = desired_bias[:, None] * motor_sign
    steps = round(prefix_seconds / hover_config.dt)
    sample_times = tuple(
        value for value in (0.25, 0.50, 0.75, 1.00, 1.50) if value <= prefix_seconds
    )
    sample_steps = {round(value / hover_config.dt): value for value in sample_times}
    permutation = torch.cat(
        (
            torch.arange(episodes // 2, episodes, device=device),
            torch.arange(episodes // 2, device=device),
        )
    )
    bucket_losses: dict[str, list[Tensor]] = {
        "reset": [],
        "0.25:0.5": [],
        "0.5:0.75": [],
        "0.75:end": [],
    }
    bucket_squared_errors: dict[str, list[Tensor]] = {key: [] for key in bucket_losses}
    bucket_pool_squared_errors: dict[str, list[Tensor]] = {key: [] for key in bucket_losses}
    samples: dict[str, Any] = {}
    for step in range(steps + 1):
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
        if sensor_gain != 1.0:
            specific_force = specific_force.clone()
            specific_force[:, 2] = 9.81 + sensor_gain * (specific_force[:, 2] - 9.81)
        student_drive = preactivation_drive(
            student,
            image,
            state.euler[:, :2],
            student_neural,
            specific_force,
        )
        with torch.no_grad():
            reference_drive = preactivation_drive(
                reference,
                image,
                state.euler[:, :2],
                reference_neural,
                state.specific_force,
            )
        delta_current = student_drive[:, throttle_nodes] - reference_drive[:, throttle_nodes]
        time_seconds = step * hover_config.dt
        desired_current = (
            target_current
            if time_seconds >= target_onset_seconds
            else torch.zeros_like(target_current)
        )
        current_error = (delta_current - desired_current) / current_scale
        direct_loss = current_error.square().mean()
        predicted_bias = (delta_current * motor_sign).mean(dim=1)
        desired_scalar = (
            desired_bias if time_seconds >= target_onset_seconds else torch.zeros_like(desired_bias)
        )
        half = episodes // 2
        pair_error = (
            (predicted_bias[:half] - predicted_bias[half:])
            - (desired_scalar[:half] - desired_scalar[half:])
        ) / current_scale
        pool_error = (predicted_bias - desired_scalar) / current_scale
        name = bucket_name(time_seconds, target_onset_seconds)
        bucket_losses[name].append(
            pool_error.square().mean()
            + direct_current_weight * direct_loss
            + pair_loss_weight * pair_error.square().mean()
        )
        bucket_squared_errors[name].append(
            (delta_current - desired_current).square().mean().detach()
        )
        bucket_pool_squared_errors[name].append(
            (predicted_bias - desired_scalar).square().mean().detach()
        )
        if step in sample_steps:
            samples[f"{sample_steps[step]:g}"] = {
                **regression_metrics(predicted_bias, desired_bias),
                "direct_current_rmse": float(
                    (delta_current - target_current).square().mean().sqrt().detach()
                ),
            }
        if step == steps:
            break
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
            rc, stick_state = sticks(reference_motor, stick_state)
            state = quad(rc, state, mass_scale)
        other_axis_loss = (
            ((student_motor[:, :3] - reference_motor[:, :3]) / motor_scale).square().mean()
        )
        bucket_losses[name][-1] = bucket_losses[name][-1] + other_axis_weight * other_axis_loss
    active_losses = {
        name: torch.stack(values).mean() for name, values in bucket_losses.items() if values
    }
    supervised = [
        active_losses[name]
        for name in ("0.25:0.5", "0.5:0.75", "0.75:end")
        if name in active_losses
    ]
    loss = torch.stack(supervised).mean()
    if "reset" in active_losses:
        loss = loss + reset_loss_weight * active_losses["reset"]
    interval_metrics = {
        name: {
            "normalized_loss": float(active_losses[name].detach()),
            "direct_current_rmse": float(torch.stack(bucket_squared_errors[name]).mean().sqrt()),
            "pool_bias_rmse": float(torch.stack(bucket_pool_squared_errors[name]).mean().sqrt()),
        }
        for name in active_losses
    }
    return loss, {"samples": samples, "intervals": interval_metrics}


def save_checkpoint(
    path: Path,
    student: ConnectomeController,
    source: dict[str, Any],
    args: argparse.Namespace,
    update: int,
) -> None:
    torch.save(
        {
            **copy.deepcopy(source),
            "controller": {
                name: value.detach().cpu() for name, value in student.state_dict().items()
            },
            "source_checkpoint_sha256": file_sha256(args.checkpoint),
            "acceleration_current_distillation": {
                "label_mode": args.label_mode,
                "plasticity": args.plasticity,
                "selected_update": update,
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
    source = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if source["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(source.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    hover_config = HoverConfig(**source["hover_config"])
    gate_config = GateConfig(**source["gate_config"])
    validate_args(args, hover_config.dt)
    resolution = int(source["image_resolution"])
    reference = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source["retinal_receptive_field"]),
    ).to(device)
    reference.load_state_dict(source["controller"])
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    student = copy.deepcopy(reference)
    student.train()
    edge_indices_np, node_indices_np, path_counts = path_plasticity(
        args.graph,
        args.baseline_graph,
        max_hops=args.max_hops,
        mode=args.plasticity,
    )
    edge_indices = torch.from_numpy(edge_indices_np).to(device=device)
    node_indices = torch.from_numpy(node_indices_np).to(device=device)
    edge_mask, node_mask = register_parameter_masks(student, edge_indices, node_indices)
    initial_edge = student.edge_magnitude.detach().clone()
    initial_tau = student.raw_time_constant.detach().clone()
    optimizer = torch.optim.Adam(
        (
            {"params": [student.edge_magnitude], "lr": args.edge_learning_rate},
            {"params": [student.raw_time_constant], "lr": args.time_constant_learning_rate},
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
                "plasticity": args.plasticity,
                "trainable_edges": len(edge_indices),
                "trainable_time_constants": len(node_indices),
                **path_counts,
            }
        ),
        flush=True,
    )
    for update in range(1, args.updates + 1):
        student.train()
        loss, training = current_rollout(
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
            current_scale=args.current_scale,
            motor_scale=args.motor_scale,
            direct_current_weight=args.direct_current_weight,
            pair_loss_weight=args.pair_loss_weight,
            other_axis_weight=args.other_axis_weight,
            reset_loss_weight=args.reset_loss_weight,
            sensor_gain=args.sensor_gain,
            label_mode=args.label_mode,
        )
        edge_regularization = (
            (
                (student.edge_magnitude[edge_indices] - initial_edge[edge_indices])
                / initial_edge[edge_indices].clamp_min(0.05)
            )
            .square()
            .mean()
        )
        tau_regularization = (
            (student.raw_time_constant[node_indices] - initial_tau[node_indices]).square().mean()
            if len(node_indices)
            else loss.new_zeros(())
        )
        objective = loss + 1.0e-4 * edge_regularization + 1.0e-5 * tau_regularization
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (student.edge_magnitude, student.raw_time_constant), args.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite current-distillation gradient")
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
                            **training,
                        }
                    ),
                    flush=True,
                )
            continue
        student.eval()
        with torch.no_grad():
            heldout_loss, heldout = current_rollout(
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
                current_scale=args.current_scale,
                motor_scale=args.motor_scale,
                direct_current_weight=args.direct_current_weight,
                pair_loss_weight=args.pair_loss_weight,
                other_axis_weight=args.other_axis_weight,
                reset_loss_weight=args.reset_loss_weight,
                sensor_gain=args.sensor_gain,
            )
        half = heldout["samples"]["0.5"]
        late = heldout["intervals"]["0.75:end"]
        correlation = half["pearson_correlation"]
        key = (
            -float(heldout_loss),
            float(correlation) if correlation is not None else -1.0,
            -float(late["pool_bias_rmse"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(best_path, student, source, args, update)
        entry = {
            "update": update,
            "training_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "training": training,
            "heldout_loss": float(heldout_loss),
            "heldout": heldout,
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
    probe_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "pair_swapped_acceleration": {"swapped_acceleration": True},
    }
    final_probes: dict[str, Any] = {}
    with torch.no_grad():
        for name, controls in probe_specs.items():
            probe_loss, probe = current_rollout(
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
                current_scale=args.current_scale,
                motor_scale=args.motor_scale,
                direct_current_weight=args.direct_current_weight,
                pair_loss_weight=args.pair_loss_weight,
                other_axis_weight=args.other_axis_weight,
                reset_loss_weight=args.reset_loss_weight,
                sensor_gain=args.sensor_gain,
                **controls,
            )
            final_probes[name] = {"loss": float(probe_loss), **probe}
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
            acceleration_gain=args.sensor_gain,
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
    live_half = final_probes["live_sensors"]["samples"]["0.5"]
    live_late = final_probes["live_sensors"]["intervals"]["0.75:end"]
    controls = (
        final_probes["constant_1g_acceleration"]["samples"]["0.5"],
        final_probes["pair_swapped_acceleration"]["samples"]["0.5"],
    )
    temporal_fit = (
        live_half["pearson_correlation"] is not None
        and live_half["pearson_correlation"] >= 0.95
        and live_half["signed_slope"] is not None
        and 0.8 <= live_half["signed_slope"] <= 1.2
        and live_late["pool_bias_rmse"] <= 0.02
    )
    live_correlation = live_half["pearson_correlation"]
    sensor_causality = live_correlation is not None and all(
        control["pearson_correlation"] is None
        or control["pearson_correlation"] <= live_correlation - 0.20
        for control in controls
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
        and temporal_fit
        and sensor_causality
        and flight_causality
    )
    frozen_edges = ~edge_mask.bool()
    frozen_nodes = ~node_mask.bool()
    report = {
        "experiment": "dense mass-oracle current distillation into acceleration paths",
        "claim_scope": (
            "Exact mass is a training label only. Deployment receives instantaneous FPV, "
            "roll/pitch, body-Z specific force, and persistent connectome state."
        ),
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
        "path_counts": path_counts,
        "training": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "trainable_edges": len(edge_indices),
            "trainable_time_constants": len(node_indices),
            "balanced_temporal_intervals": True,
            "oracle_corrected_training_physics": False,
        },
        "history": history,
        "selected_update": best["acceleration_current_distillation"]["selected_update"],
        "parameter_drift": {
            "frozen_edge_magnitude_max_absolute_change": float(
                (student.edge_magnitude[frozen_edges] - initial_edge[frozen_edges])
                .abs()
                .max()
                .detach()
            ),
            "trainable_edge_magnitude_max_absolute_change": float(
                (student.edge_magnitude[edge_indices] - initial_edge[edge_indices])
                .abs()
                .max()
                .detach()
            ),
            "frozen_time_constant_max_absolute_change": float(
                (student.raw_time_constant[frozen_nodes] - initial_tau[frozen_nodes])
                .abs()
                .max()
                .detach()
            ),
            "trainable_time_constant_max_absolute_change": (
                float(
                    (student.raw_time_constant[node_indices] - initial_tau[node_indices])
                    .abs()
                    .max()
                    .detach()
                )
                if len(node_indices)
                else 0.0
            ),
        },
        "baseline": baseline,
        "final": final,
        "final_current_probes": final_probes,
        "causal_checks": {
            "trained_with_true_mass_labels": args.label_mode == "true",
            "temporal_current_fit": temporal_fit,
            "current_sensor_causality": sensor_causality,
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
                "baseline_success_rate": baseline["success_rate"],
                "live_success_rate": final["live_sensors"]["success_rate"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
