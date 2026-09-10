#!/usr/bin/env python3
"""Train one native recurrent layer between acceleration paths and throttle motors."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
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
    ReadoutSpec,
    collect_history,
    make_readout_spec,
    metric_summary,
    model_passes,
    target_scales,
)
from search_gate_acceleration_path_es import PathSpec, make_path_spec  # noqa: E402
from search_gate_motor_interface_es import load_controller, stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402


@dataclass(frozen=True)
class RecurrentRoutingSpec:
    boundary_edges: Tensor
    readout_edges: Tensor
    selected_edges: Tensor
    internal_return_edges: Tensor


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
        "--bias-report",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-throttle-readout-bias-diagnostic-v1" / "report.json"
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
        default=REPO_ROOT / "runs" / "gate" / "recurrent-routing-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fit-pairs", type=int, default=48)
    parser.add_argument("--validation-pairs", type=int, default=16)
    parser.add_argument("--minibatch-pairs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--midpoint-update", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--seconds", type=float, default=1.50)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--anchor-seconds", type=float, default=0.20)
    parser.add_argument("--anchor-scale", type=float, default=0.01)
    parser.add_argument("--anchor-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--midpoint-maximum-group-nrmse", type=float, default=0.50)
    parser.add_argument("--maximum-prefix-rmse", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--directional-epsilon", type=float, default=0.001)
    parser.add_argument("--directional-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--directional-absolute-tolerance", type=float, default=0.0001)
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--flight-episodes", type=int, default=256)
    parser.add_argument("--flight-seconds", type=float, default=12.0)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--optimizer-seed", type=int, default=1_047_131)
    parser.add_argument("--held-out-seed", type=int, default=1_047_031)
    parser.add_argument("--flight-seed", type=int, default=1_048_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.routing_report,
        args.routing_readouts,
        args.bias_report,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "fit_pairs": (args.fit_pairs, 48),
        "validation_pairs": (args.validation_pairs, 16),
        "minibatch_pairs": (args.minibatch_pairs, 8),
        "updates": (args.updates, 300),
        "validation_interval": (args.validation_interval, 100),
        "midpoint_update": (args.midpoint_update, 100),
        "learning_rate": (args.learning_rate, 0.003),
        "gradient_clip_norm": (args.gradient_clip_norm, 1.0),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 8.0),
        "seconds": (args.seconds, 1.50),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "anchor_seconds": (args.anchor_seconds, 0.20),
        "anchor_scale": (args.anchor_scale, 0.01),
        "anchor_loss_coefficient": (args.anchor_loss_coefficient, 1.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "midpoint_maximum_group_nrmse": (args.midpoint_maximum_group_nrmse, 0.50),
        "maximum_prefix_rmse": (args.maximum_prefix_rmse, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
        "directional_epsilon": (args.directional_epsilon, 0.001),
        "directional_relative_tolerance": (args.directional_relative_tolerance, 0.02),
        "directional_absolute_tolerance": (args.directional_absolute_tolerance, 0.0001),
        "held_out_pairs": (args.held_out_pairs, 128),
        "flight_episodes": (args.flight_episodes, 256),
        "flight_seconds": (args.flight_seconds, 12.0),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered recurrent-routing values changed: {', '.join(wrong)}")
    if args.fit_pairs + args.validation_pairs != 64:
        raise SystemExit("recurrent-routing split must partition all 64 development pairs")


def make_recurrent_routing_spec(
    controller: ConnectomeController,
    path_spec: PathSpec,
    readout_spec: ReadoutSpec,
) -> RecurrentRoutingSpec:
    enters_return = torch.isin(controller.edge_post[path_spec.edges], readout_spec.return_sources)
    incoming_path_edges = path_spec.edges[enters_return]
    internal = torch.isin(controller.edge_pre[incoming_path_edges], readout_spec.return_sources)
    internal_return_edges = incoming_path_edges[internal]
    boundary_edges = incoming_path_edges[~internal]
    selected_edges = torch.cat((boundary_edges, readout_spec.edges))
    if (
        len(incoming_path_edges),
        len(boundary_edges),
        len(internal_return_edges),
        len(readout_spec.edges),
        len(torch.unique(selected_edges)),
    ) != (107, 88, 19, 37, 125):
        raise RuntimeError("unexpected one-layer recurrent-routing topology")
    if torch.isin(boundary_edges, readout_spec.edges).any():
        raise RuntimeError("boundary and readout edge masks overlap")
    return RecurrentRoutingSpec(
        boundary_edges=boundary_edges,
        readout_edges=readout_spec.edges,
        selected_edges=selected_edges,
        internal_return_edges=internal_return_edges,
    )


def balanced_pair_split(
    total_pairs: int = 64, fit_pairs: int = 48
) -> tuple[np.ndarray, np.ndarray]:
    if total_pairs != 64 or fit_pairs != 48:
        raise ValueError("the audited recurrent-routing split is fixed at 48/16 of 64 pairs")
    half = total_pairs // 2
    per_half = fit_pairs // 2
    fit = np.concatenate((np.arange(per_half), np.arange(half, half + per_half)))
    validation = np.concatenate(
        (np.arange(per_half, half), np.arange(half + per_half, total_pairs))
    )
    return fit, validation


def pair_episodes(pair_ids: np.ndarray, *, device: torch.device) -> Tensor:
    episodes = np.stack((2 * pair_ids, 2 * pair_ids + 1), axis=1).reshape(-1)
    return torch.as_tensor(episodes, device=device, dtype=torch.long)


def controller_step_with_selected_magnitudes(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    state: Tensor,
    acceleration: Tensor,
    stick_position: Tensor,
    selected_edges: Tensor,
    selected_magnitudes: Tensor,
) -> tuple[Tensor, Tensor]:
    edge_magnitude = (
        controller.edge_magnitude.detach()
        .clone()
        .index_copy(0, selected_edges, selected_magnitudes)
    )
    messages = torch.tanh(state)[:, controller.edge_pre] * controller.edge_sign * edge_magnitude
    recurrent = torch.zeros_like(state).index_add(1, controller.edge_post, messages)
    drive = (
        recurrent
        + controller.bias.detach()
        + controller.sensory_drive(image, roll_pitch, acceleration, stick_position)
    )
    target = 5.0 * torch.tanh(drive / 5.0)
    alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant.detach())
    next_state = state + alpha * (target - state)
    return controller.motor_drive(next_state), next_state


def recurrent_motor_replay(
    controller: ConnectomeController,
    sensors: dict[str, Tensor],
    episode_ids: Tensor,
    *,
    selected_edges: Tensor | None = None,
    selected_magnitudes: Tensor | None = None,
) -> Tensor:
    if (selected_edges is None) != (selected_magnitudes is None):
        raise ValueError("selected edges and magnitudes must be supplied together")
    batch = len(episode_ids)
    state = controller.initial_state(batch, device=episode_ids.device, dtype=torch.float32)
    outputs = []
    for step in range(sensors["image"].shape[0]):
        inputs = (
            sensors["image"][step, episode_ids],
            sensors["roll_pitch"][step, episode_ids],
            state,
            sensors["acceleration"][step, episode_ids],
            sensors["stick_position"][step, episode_ids],
        )
        if selected_edges is None:
            motor, state = controller(*inputs)
        else:
            motor, state = controller_step_with_selected_magnitudes(
                controller,
                *inputs,
                selected_edges,
                selected_magnitudes,
            )
        outputs.append(motor)
    return torch.stack(outputs)


def prepare_history_targets(
    history: dict[str, Any],
    *,
    start_step: int,
    device: torch.device,
) -> dict[str, Tensor]:
    episodes = int(history["episodes"])
    window_steps = int(history["steps"]) - start_step
    samples = history["samples"]
    result = {
        "reserve_motor": samples["reserve_motor"].reshape(window_steps, episodes).to(device),
        "correction": samples["target"].reshape(window_steps, episodes).to(device),
        "source_throttle": samples["source_motor"].reshape(window_steps, episodes).to(device),
        "window": samples["window"].reshape(window_steps, episodes)[:, 0].long().to(device),
        "mass_half": (history["mass_scale"] > 1.0).long().to(device),
    }
    return result


def metric_samples(targets: dict[str, Tensor], episode_ids: Tensor) -> dict[str, Tensor]:
    steps = targets["correction"].shape[0]
    mass = targets["mass_half"][episode_ids]
    return {
        "target": targets["correction"][:, episode_ids].reshape(-1).detach().cpu(),
        "window": targets["window"][:, None]
        .expand(-1, len(episode_ids))
        .reshape(-1)
        .detach()
        .cpu(),
        "mass_half": mass[None].expand(steps, -1).reshape(-1).detach().cpu(),
    }


def recurrent_training_loss(
    controller: ConnectomeController,
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
) -> tuple[Tensor, dict[str, Tensor]]:
    outputs = recurrent_motor_replay(
        controller,
        sensors,
        episode_ids,
        selected_edges=selected_edges,
        selected_magnitudes=selected_magnitudes,
    )
    throttle_error = outputs[start_step:, :, 3] - targets["reserve_motor"][:, episode_ids]
    groups = (2 * targets["window"][:, None] + targets["mass_half"][episode_ids][None]).expand_as(
        throttle_error
    )
    group_losses = [
        (throttle_error[groups == group] / scales[group]).square().mean() for group in range(6)
    ]
    throttle_loss = torch.stack(group_losses).mean()
    anchor_error = outputs[:anchor_steps] - source_outputs[:anchor_steps, episode_ids]
    anchor_rmse = anchor_error.square().mean().sqrt()
    anchor_loss = (anchor_error / anchor_scale).square().mean()
    total = throttle_loss + anchor_loss_coefficient * anchor_loss
    return total, {
        "throttle_normalized_equal_group_mse": throttle_loss.detach(),
        "prefix_motor_rmse": anchor_rmse.detach(),
        "normalized_prefix_motor_mse": anchor_loss.detach(),
    }


@torch.no_grad()
def evaluate_recurrent_snapshot(
    controller: ConnectomeController,
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
    constant: float,
    maximum_group_nrmse: float,
    minimum_constant_improvement: float,
) -> dict[str, Any]:
    outputs = recurrent_motor_replay(
        controller,
        sensors,
        episode_ids,
        selected_edges=selected_edges,
        selected_magnitudes=selected_magnitudes,
    )
    samples = metric_samples(targets, episode_ids)
    constant_metrics = metric_summary(
        np.full(len(samples["target"]), constant),
        samples,
        scales,
        constant_aggregate_rmse=None,
    )
    prediction = (outputs[start_step:, :, 3] - source_outputs[start_step:, episode_ids, 3]).reshape(
        -1
    )
    metrics = metric_summary(
        prediction.detach().cpu().numpy().astype(np.float64, copy=False),
        samples,
        scales,
        constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
    )
    gate = model_passes(
        metrics,
        maximum_group_nrmse=maximum_group_nrmse,
        minimum_constant_improvement=minimum_constant_improvement,
    )
    prefix_rmse = float(
        (outputs[:anchor_steps] - source_outputs[:anchor_steps, episode_ids]).square().mean().sqrt()
    )
    return {
        "metrics": metrics,
        "constant_baseline": constant_metrics,
        "pass_gate": gate,
        "prefix_motor_rmse": prefix_rmse,
    }


def directional_gradient_audit(
    controller: ConnectomeController,
    sensors: dict[str, Tensor],
    targets: dict[str, Tensor],
    source_outputs: Tensor,
    episode_ids: Tensor,
    selected_edges: Tensor,
    source_magnitudes: Tensor,
    scales: dict[int, float],
    *,
    start_step: int,
    anchor_steps: int,
    anchor_scale: float,
    anchor_loss_coefficient: float,
    epsilon: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    seed: int,
) -> dict[str, Any]:
    magnitudes = source_magnitudes.detach().clone().requires_grad_(True)
    loss, _ = recurrent_training_loss(
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
        anchor_scale=anchor_scale,
        anchor_loss_coefficient=anchor_loss_coefficient,
    )
    gradient = torch.autograd.grad(loss, magnitudes)[0]
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(len(magnitudes), generator=generator).to(magnitudes.device)
    direction /= direction.norm()
    analytic = float(torch.dot(gradient, direction))
    with torch.no_grad():
        plus, _ = recurrent_training_loss(
            controller,
            sensors,
            targets,
            source_outputs,
            episode_ids,
            selected_edges,
            magnitudes + epsilon * direction,
            scales,
            start_step=start_step,
            anchor_steps=anchor_steps,
            anchor_scale=anchor_scale,
            anchor_loss_coefficient=anchor_loss_coefficient,
        )
        minus, _ = recurrent_training_loss(
            controller,
            sensors,
            targets,
            source_outputs,
            episode_ids,
            selected_edges,
            magnitudes - epsilon * direction,
            scales,
            start_step=start_step,
            anchor_steps=anchor_steps,
            anchor_scale=anchor_scale,
            anchor_loss_coefficient=anchor_loss_coefficient,
        )
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    absolute_error = abs(analytic - finite_difference)
    tolerance = absolute_tolerance + relative_tolerance * max(abs(analytic), abs(finite_difference))
    return {
        "seed": seed,
        "epsilon": epsilon,
        "analytic_directional_derivative": analytic,
        "finite_difference_directional_derivative": finite_difference,
        "absolute_error": absolute_error,
        "tolerance": tolerance,
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance": absolute_tolerance,
        "passed": bool(absolute_error <= tolerance),
    }


def validation_selection(
    validations: list[dict[str, Any]], *, maximum_prefix_rmse: float
) -> tuple[dict[str, Any], str]:
    prefix_safe = [item for item in validations if item["prefix_motor_rmse"] <= maximum_prefix_rmse]
    if not prefix_safe:
        pool = validations
        rule = "no prefix-safe snapshot; diagnostic minimum worst-group then aggregate NRMSE"
    else:
        passing = [item for item in prefix_safe if item["pass_gate"]["passed"]]
        pool = passing or prefix_safe
        rule = (
            "passing and prefix-safe snapshots first; minimum worst-group then aggregate NRMSE"
            if passing
            else "prefix-safe snapshots; minimum worst-group then aggregate NRMSE"
        )
    selected = min(
        pool,
        key=lambda item: (
            item["metrics"]["maximum_group_nrmse"],
            item["metrics"]["aggregate_equal_group_nrmse"],
        ),
    )
    return selected, rule


def public_validation(validation: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in validation.items() if name != "_magnitudes"}


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    bias_report = json.loads(args.bias_report.read_text())
    prerequisite_checks = {
        "graph_hash_matches": routing_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": routing_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "routing_readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": routing_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "bias_family_closed": bias_report.get("classification", {}).get("interpretation")
        == "native_bias_readout_failed",
        "bias_flight_not_run": bias_report.get("assisted_flight") is None,
        "source_was_preserved": bias_report.get("source_preserved") is True,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"recurrent-routing prerequisite failed: {prerequisite_checks}")

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
    selected_edges = spec.selected_edges
    source_magnitudes = controller.edge_magnitude[selected_edges].detach().clone()
    if float(source_magnitudes.max()) > args.maximum_edge_magnitude:
        raise SystemExit("source contains a selected magnitude above the legal bound")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    mask_path = args.output_dir / "preregistered-mask.json"
    selection_path = args.output_dir / "selected-magnitudes.json"
    if report_path.exists() or mask_path.exists() or selection_path.exists():
        raise SystemExit("output directory contains a stale report or selection")
    mask_path.write_text(
        json.dumps(
            {
                "definition": {
                    "return_source_count": len(readout_spec.return_sources),
                    "boundary_edges": (
                        "audited four-hop-path edges with post in return sources and pre "
                        "outside return sources"
                    ),
                    "readout_edges": "all existing edges entering the seven throttle motors",
                    "internal_return_edges": "audited return-to-return edges, kept frozen",
                },
                "graph_sha256": file_sha256(args.graph),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "boundary_edge_indices": spec.boundary_edges.detach().cpu().tolist(),
                "readout_edge_indices": spec.readout_edges.detach().cpu().tolist(),
                "selected_edge_indices": selected_edges.detach().cpu().tolist(),
                "selected_edge_signs": controller.edge_sign[selected_edges].detach().cpu().tolist(),
                "frozen_internal_return_edge_indices": spec.internal_return_edges.detach()
                .cpu()
                .tolist(),
                "trainable_edge_count": len(selected_edges),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    seed_everything(args.optimizer_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    saved = torch.load(args.replay_histories, map_location="cpu", weights_only=False)
    development_history = saved["development"]
    del saved
    sensors = {name: value.to(device) for name, value in development_history["sensors"].items()}
    start_step = round(args.takeover_seconds / hover_config.dt)
    anchor_steps = round(args.anchor_seconds / hover_config.dt)
    targets = prepare_history_targets(development_history, start_step=start_step, device=device)
    all_episode_ids = torch.arange(development_history["episodes"], device=device)
    with torch.no_grad():
        source_outputs = recurrent_motor_replay(controller, sensors, all_episode_ids)
    source_replay_difference = float(
        (source_outputs[start_step:, :, 3] - targets["source_throttle"]).abs().max()
    )
    source_replay_parity = {
        "maximum_throttle_absolute_difference": source_replay_difference,
        "threshold": 1.0e-6,
        "passed": source_replay_difference <= 1.0e-6,
    }
    if not source_replay_parity["passed"]:
        raise RuntimeError(f"source recurrent replay parity failed: {source_replay_parity}")
    scale_samples = {
        "target": targets["correction"].reshape(-1).detach().cpu(),
        "window": targets["window"][:, None]
        .expand(-1, development_history["episodes"])
        .reshape(-1)
        .detach()
        .cpu(),
        "mass_half": targets["mass_half"][None]
        .expand(targets["correction"].shape[0], -1)
        .reshape(-1)
        .detach()
        .cpu(),
    }
    scales = target_scales(scale_samples, floor=args.normalization_floor)
    reported_scale_map = routing_report["development_target_scales"]
    reported_scales = [
        float(
            reported_scale_map[
                f"{('0.50:0.75', '0.75:1.00', '1.00:1.50')[group // 2]}/"
                f"{'light' if group % 2 == 0 else 'heavy'}"
            ]
        )
        for group in range(6)
    ]
    if max(abs(scales[index] - reported_scales[index]) for index in range(6)) > 1.0e-12:
        raise RuntimeError("development target scales differ from the routing audit")
    constant = float(routing_readouts["constant_residual"])
    fit_pairs, validation_pairs = balanced_pair_split()
    validation_episode_ids = pair_episodes(validation_pairs, device=device)

    gradient_pairs = np.concatenate((fit_pairs[:4], fit_pairs[24:28]))
    gradient_episode_ids = pair_episodes(gradient_pairs, device=device)
    gradient_audit = directional_gradient_audit(
        controller,
        sensors,
        targets,
        source_outputs,
        gradient_episode_ids,
        selected_edges,
        source_magnitudes,
        scales,
        start_step=start_step,
        anchor_steps=anchor_steps,
        anchor_scale=args.anchor_scale,
        anchor_loss_coefficient=args.anchor_loss_coefficient,
        epsilon=args.directional_epsilon,
        relative_tolerance=args.directional_relative_tolerance,
        absolute_tolerance=args.directional_absolute_tolerance,
        seed=args.optimizer_seed + 1,
    )
    print(json.dumps({"phase": "directional_gradient_audit", **gradient_audit}), flush=True)
    if not gradient_audit["passed"]:
        raise RuntimeError(f"directional gradient audit failed: {gradient_audit}")

    magnitudes = torch.nn.Parameter(source_magnitudes.clone())
    optimizer = torch.optim.Adam((magnitudes,), lr=args.learning_rate)
    generator = np.random.default_rng(args.optimizer_seed)
    first_half = fit_pairs[: len(fit_pairs) // 2]
    second_half = fit_pairs[len(fit_pairs) // 2 :]
    training_history = []
    validations: list[dict[str, Any]] = []
    midpoint_stopped = False
    for update in range(1, args.updates + 1):
        batch_pairs = np.concatenate(
            (
                generator.choice(first_half, args.minibatch_pairs // 2, replace=False),
                generator.choice(second_half, args.minibatch_pairs // 2, replace=False),
            )
        )
        generator.shuffle(batch_pairs)
        episode_ids = pair_episodes(batch_pairs, device=device)
        loss, components = recurrent_training_loss(
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
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_((magnitudes,), args.gradient_clip_norm)
        optimizer.step()
        with torch.no_grad():
            magnitudes.clamp_(0.0, args.maximum_edge_magnitude)
        if update == 1 or update % 20 == 0:
            record = {
                "update": update,
                "loss": float(loss.detach()),
                "throttle_normalized_equal_group_mse": float(
                    components["throttle_normalized_equal_group_mse"]
                ),
                "prefix_motor_rmse": float(components["prefix_motor_rmse"]),
                "gradient_norm_before_clip": float(gradient_norm),
                "magnitude_minimum": float(magnitudes.detach().min()),
                "magnitude_maximum": float(magnitudes.detach().max()),
            }
            training_history.append(record)
            print(json.dumps({"phase": "training", **record}), flush=True)
        if update % args.validation_interval == 0:
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
            validation["update"] = update
            validation["_magnitudes"] = magnitudes.detach().cpu().clone()
            validations.append(validation)
            print(
                json.dumps(
                    {
                        "phase": "validation",
                        "update": update,
                        "maximum_group_nrmse": validation["metrics"]["maximum_group_nrmse"],
                        "aggregate_nrmse": validation["metrics"]["aggregate_equal_group_nrmse"],
                        "constant_improvement": validation["metrics"][
                            "rmse_improvement_over_constant"
                        ],
                        "prefix_motor_rmse": validation["prefix_motor_rmse"],
                        "pass_gate": validation["pass_gate"]["passed"],
                    }
                ),
                flush=True,
            )
            if update == args.midpoint_update and (
                validation["metrics"]["maximum_group_nrmse"] > args.midpoint_maximum_group_nrmse
                or validation["prefix_motor_rmse"] > args.maximum_prefix_rmse
            ):
                midpoint_stopped = True
                break

    selected_validation, selection_rule = validation_selection(
        validations, maximum_prefix_rmse=args.maximum_prefix_rmse
    )
    selected_magnitudes = selected_validation["_magnitudes"].to(device)
    final_replay = None
    replay_passed = False
    assisted_flight = None
    flight_passed = False
    held_source_parity = None
    if not midpoint_stopped:
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
            raise RuntimeError(f"held-out source replay parity failed: {held_source_parity}")
        final_replay = evaluate_recurrent_snapshot(
            controller,
            held_sensors,
            held_targets,
            held_source_outputs,
            held_episode_ids,
            selected_edges,
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
                    "maximum_group_nrmse": final_replay["metrics"]["maximum_group_nrmse"],
                    "constant_improvement": final_replay["metrics"][
                        "rmse_improvement_over_constant"
                    ],
                    "prefix_motor_rmse": final_replay["prefix_motor_rmse"],
                    "passed": replay_passed,
                }
            ),
            flush=True,
        )
        del held_sensors, held_source_outputs, held_history
        if replay_passed:
            candidate = copy.deepcopy(controller).to(device)
            with torch.no_grad():
                candidate.edge_magnitude[selected_edges] = selected_magnitudes.to(
                    dtype=candidate.edge_magnitude.dtype
                )
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

    if midpoint_stopped:
        interpretation = "recurrent_routing_failed_midpoint_gate"
        next_step = "stop the one-layer recurrent-routing experiment"
    elif not replay_passed:
        interpretation = "recurrent_routing_failed_fresh_replay"
        next_step = "stop before closed-loop flight"
    elif not flight_passed:
        interpretation = "recurrent_routing_passed_replay_but_failed_assisted_flight"
        next_step = "stop before any native merge"
    else:
        interpretation = "recurrent_routing_passed_replay_and_assisted_flight"
        next_step = "design a separate coupled native-flight confirmation; no automatic merge"
    selection_path.write_text(
        json.dumps(
            {
                "selected_update": selected_validation["update"],
                "selected_edge_indices": selected_edges.detach().cpu().tolist(),
                "selected_magnitudes": selected_magnitudes.detach().cpu().tolist(),
                "boundary_edge_indices": spec.boundary_edges.detach().cpu().tolist(),
                "readout_edge_indices": spec.readout_edges.detach().cpu().tolist(),
                "selection_rule": selection_rule,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "method": "one-layer native anatomical routing with full recurrent replay",
        "claim_scope": (
            "This diagnostic trains magnitudes on 125 existing fixed-sign edges while replaying "
            "the complete native recurrent state from initialization. Saved sensory histories "
            "are training data only. No fitted parameter is merged or promoted."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "routing_report": stable_path(args.routing_report),
        "routing_report_sha256": file_sha256(args.routing_report),
        "bias_report": stable_path(args.bias_report),
        "bias_report_sha256": file_sha256(args.bias_report),
        "replay_histories": stable_path(args.replay_histories),
        "replay_histories_sha256": file_sha256(args.replay_histories),
        "preregistered_mask": stable_path(mask_path),
        "preregistered_mask_sha256": file_sha256(mask_path),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "development_pairs": 64,
            "balanced_fit_validation_split": [48, 16],
            "validation_updates": [100, 200, 300],
            "trainable_parameters": (
                "magnitudes on 88 boundary-to-return and 37 return-to-motor edges"
            ),
            "edge_signs_fixed": True,
            "all_biases_and_time_constants_frozen": True,
            "all_other_edge_magnitudes_frozen": True,
            "recurrent_state": "entire native controller replayed from zero for every prefix",
            "training_throttle_target": "saved absolute reserve motor drive, aligned at each step",
            "prefix_anchor": "all four candidate motor drives versus source for t<0.2 seconds",
            "selection": (
                "prefix-safe, both-gates-pass first; then minimum validation worst-group NRMSE "
                "and aggregate NRMSE"
            ),
            "fresh_replay_histories": "source throttle plus reserve steering after 0.5 seconds",
            "assisted_flight_runs_only_after_replay_pass": True,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "topology": {
            "path_edge_count": len(path_spec.edges),
            "path_node_count": len(path_nodes),
            "incoming_path_edges_to_return_sources": 107,
            "boundary_edge_count": len(spec.boundary_edges),
            "internal_return_edge_count_frozen": len(spec.internal_return_edges),
            "readout_edge_count": len(spec.readout_edges),
            "selected_edge_count": len(selected_edges),
            "selected_positive_sign_count": int((controller.edge_sign[selected_edges] > 0).sum()),
            "selected_negative_sign_count": int((controller.edge_sign[selected_edges] < 0).sum()),
        },
        "source_replay_parity": source_replay_parity,
        "held_out_source_replay_parity": held_source_parity,
        "directional_gradient_audit": gradient_audit,
        "development_target_scales": scales,
        "training_history": training_history,
        "validations": [public_validation(item) for item in validations],
        "midpoint_stopped": midpoint_stopped,
        "selected_update": selected_validation["update"],
        "selection_rule": selection_rule,
        "selected_validation": public_validation(selected_validation),
        "final_fresh_replay": final_replay,
        "fresh_replay_passed": replay_passed,
        "selected_magnitudes": stable_path(selection_path),
        "selected_magnitudes_sha256": file_sha256(selection_path),
        "assisted_flight": assisted_flight,
        "assisted_flight_passed": flight_passed,
        "classification": {"interpretation": interpretation, "next_step": next_step},
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
