#!/usr/bin/env python3
"""Teach a small native recurrent motif a persistent launch-correction code."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import lsq_linear
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_acceleration_es import compact_metrics  # noqa: E402
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    seed_everything,
)
from train_gate_acceleration_current_distillation import preactivation_drive  # noqa: E402
from train_gate_mass_current_distillation import (  # noqa: E402
    paired_launch,
    regression_metrics,
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
        default=REPO_ROOT / "runs" / "gate" / "native-latent-motif",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix-seconds", type=float, default=1.50)
    parser.add_argument("--ramp-start-seconds", type=float, default=0.50)
    parser.add_argument("--established-seconds", type=float, default=0.75)
    parser.add_argument("--activity-scale", type=float, default=0.40)
    parser.add_argument("--maximum-center-absolute", type=float, default=0.50)
    parser.add_argument("--edge-learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--bias-learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--time-constant-learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--pair-loss-weight", type=float, default=100.0)
    parser.add_argument("--temporal-loss-weight", type=float, default=0.10)
    parser.add_argument("--reset-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--selection-interval", type=int, default=20)
    parser.add_argument(
        "--save-selection-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain every held-out selection checkpoint for bounded readout audits.",
    )
    parser.add_argument(
        "--neutral-training-every",
        type=int,
        default=2,
        help="Train a neutral-input retention suffix every Nth update.",
    )
    parser.add_argument("--selection-episodes", type=int, default=256)
    parser.add_argument("--fit-episodes", type=int, default=2048)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=480_031)
    parser.add_argument("--selection-seed", type=int, default=490_031)
    parser.add_argument("--fit-seed", type=int, default=500_031)
    parser.add_argument("--final-seed", type=int, default=510_031)
    parser.add_argument("--active-acceleration-channel", type=int, default=0)
    parser.add_argument("--maximum-sensory-hops", type=int, default=3)
    parser.add_argument("--maximum-cycle-length", type=int, default=5)
    parser.add_argument("--maximum-sources", type=int, default=4)
    parser.add_argument("--fit-stride", type=int, default=2)
    parser.add_argument("--launch-throttle-offset", type=float, default=0.002)
    parser.add_argument("--acceleration-noise-mg", type=float, default=0.10)
    parser.add_argument("--activation-noise-std", type=float, default=0.005)
    parser.add_argument(
        "--label-mode",
        choices=("true", "shuffled", "constant"),
        default="true",
        help="Training-only causal control for the explicit latent target.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.checkpoint, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.updates,
        args.batch_size,
        args.prefix_seconds,
        args.ramp_start_seconds,
        args.established_seconds,
        args.activity_scale,
        args.maximum_center_absolute,
        args.edge_learning_rate,
        args.bias_learning_rate,
        args.time_constant_learning_rate,
        args.pair_loss_weight,
        args.temporal_loss_weight,
        args.reset_loss_weight,
        args.gradient_clip,
        args.selection_interval,
        args.neutral_training_every,
        args.selection_episodes,
        args.fit_episodes,
        args.final_episodes,
        args.maximum_sensory_hops,
        args.maximum_cycle_length,
        args.maximum_sources,
        args.fit_stride,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, rates, scales, and times must be positive")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even for matched-mass pairs")
    for name in ("selection_episodes", "fit_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")
    if not 0.0 < args.ramp_start_seconds < args.established_seconds:
        raise SystemExit("the target ramp must start after reset and before establishment")
    if args.established_seconds >= args.prefix_seconds:
        raise SystemExit("the code must be established before the prefix ends")
    if args.established_seconds + 0.25 > args.prefix_seconds:
        raise SystemExit("the prefix must leave two post-establishment loss windows")
    if round(args.ramp_start_seconds / dt) < 2:
        raise SystemExit("the target ramp must leave at least two reset steps")
    if args.active_acceleration_channel not in (0, 1):
        raise SystemExit("the active acceleration channel must be 0 or 1")
    if args.maximum_center_absolute >= 0.8:
        raise SystemExit("the activity operating point must remain below tanh saturation")
    if not 2 <= args.maximum_sources <= 4:
        raise SystemExit("--maximum-sources must be between two and four")
    if args.acceleration_noise_mg < 0.0 or args.activation_noise_std < 0.0:
        raise SystemExit("perturbation magnitudes must be nonnegative")
    if not 0.0 <= args.launch_throttle_offset <= 0.02:
        raise SystemExit("launch throttle offset must be in [0, 0.02]")


def edge_lookup(pre: np.ndarray, post: np.ndarray) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for edge, pair in enumerate(zip(pre, post, strict=True)):
        key = (int(pair[0]), int(pair[1]))
        if key in lookup:
            raise SystemExit(f"parallel anatomical edges are unsupported: {key}")
        lookup[key] = edge
    return lookup


def sorted_adjacency(
    pre: np.ndarray,
    post: np.ndarray,
    node_ids: np.ndarray,
) -> list[list[int]]:
    adjacency = [[] for _ in range(len(node_ids))]
    for source, target in zip(pre, post, strict=True):
        adjacency[int(source)].append(int(target))
    for targets in adjacency:
        targets.sort(key=lambda node: (int(node_ids[node]), node))
    return adjacency


def shortest_signed_path(
    adjacency: list[list[int]],
    signs: dict[tuple[int, int], int],
    starts: list[int],
    target: int,
    *,
    required_sign: int,
    maximum_hops: int,
    forbidden: set[int],
) -> list[int] | None:
    queue: deque[tuple[list[int], int]] = deque(
        [([start], 1) for start in starts if start not in forbidden or start == target]
    )
    visited = {(start, 1) for start in starts}
    while queue:
        path, product = queue.popleft()
        node = path[-1]
        if node == target and product == required_sign:
            return path
        if len(path) - 1 >= maximum_hops:
            continue
        for neighbor in adjacency[node]:
            if neighbor in forbidden and neighbor != target:
                continue
            next_product = product * signs[(node, neighbor)]
            state = (neighbor, next_product)
            if state in visited:
                continue
            visited.add(state)
            queue.append(([*path, neighbor], next_product))
    return None


def shortest_positive_cycle(
    adjacency: list[list[int]],
    signs: dict[tuple[int, int], int],
    source: int,
    *,
    maximum_length: int,
    forbidden: set[int],
) -> list[int] | None:
    """Return cycle nodes without repeating ``source`` at the end."""

    queue: deque[tuple[list[int], int]] = deque()
    for neighbor in adjacency[source]:
        if neighbor in forbidden or neighbor == source:
            continue
        queue.append(([source, neighbor], signs[(source, neighbor)]))
    while queue:
        path, product = queue.popleft()
        node = path[-1]
        if len(path) > maximum_length:
            continue
        for neighbor in adjacency[node]:
            next_product = product * signs[(node, neighbor)]
            if neighbor == source:
                if next_product > 0:
                    return path
                continue
            if neighbor in forbidden or neighbor in path:
                continue
            if len(path) < maximum_length:
                queue.append(([*path, neighbor], next_product))
    return None


@torch.no_grad()
def return_source_alignment(
    controller: ConnectomeController,
    graph_path: Path,
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[tuple[int, int], dict[str, float | None]]:
    """Measure the untrained mass direction at every throttle-return source."""

    with np.load(graph_path) as graph:
        offsets = graph["output_pool_offsets"]
        pools = graph["output_pool_indices"]
        throttle_motors = pools[offsets[6] : offsets[8]]
        sources_np = np.unique(graph["edge_pre"][np.isin(graph["edge_post"], throttle_motors)])
    sources = torch.from_numpy(sources_np).to(device=device)
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    horizon_steps = {round(value / hover_config.dt): value for value in (0.50, 0.75, 1.00, 1.50)}
    by_horizon: dict[str, Tensor] = {}
    for step in range(1, max(horizon_steps) + 1):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
        )
        if step in horizon_steps:
            by_horizon[f"{horizon_steps[step]:g}"] = torch.tanh(neural[:, sources])
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
    centered_mass = normalized_mass - normalized_mass.mean()
    mass_scale_denominator = centered_mass.square().mean().sqrt()
    results: dict[tuple[int, int], dict[str, float | None]] = {}
    for slot, source in enumerate(map(int, sources_np)):
        for polarity in (-1, 1):
            horizons: dict[str, float | None] = {}
            for name, activity in by_horizon.items():
                value = polarity * activity[:, slot]
                centered = value - value.mean()
                denominator = centered.square().mean().sqrt() * mass_scale_denominator
                horizons[name] = (
                    float((centered * centered_mass).mean() / denominator)
                    if float(denominator) > 1.0e-12
                    else None
                )
            results[(source, polarity)] = horizons
    return results


def motif_spec(
    controller: ConnectomeController,
    graph_path: Path,
    *,
    source_alignment: dict[tuple[int, int], dict[str, float | None]],
    active_acceleration_channel: int,
    maximum_sensory_hops: int,
    maximum_cycle_length: int,
    maximum_sources: int,
    activity_scale: float,
) -> dict[str, Any]:
    with np.load(graph_path) as graph:
        node_ids = graph["node_ids"]
        pre = graph["edge_pre"]
        post = graph["edge_post"]
        edge_sign = graph["edge_sign"].astype(np.int8)
        acceleration_nodes = graph["acceleration_node_indices"]
        acceleration_channels = graph["acceleration_channels"]
        offsets = graph["output_pool_offsets"]
        pools = graph["output_pool_indices"]
    positive_motors = pools[offsets[6] : offsets[7]]
    negative_motors = pools[offsets[7] : offsets[8]]
    throttle_motors = np.concatenate((positive_motors, negative_motors))
    motor_set = set(map(int, throttle_motors))
    positive_motor_set = set(map(int, positive_motors))
    return_edges = np.flatnonzero(np.isin(post, throttle_motors))
    return_sources = np.unique(pre[return_edges])
    active_inputs = sorted(
        map(int, acceleration_nodes[acceleration_channels == active_acceleration_channel]),
        key=lambda node: (int(node_ids[node]), node),
    )
    adjacency = sorted_adjacency(pre, post, node_ids)
    lookup = edge_lookup(pre, post)
    signs = {pair: int(edge_sign[edge]) for pair, edge in lookup.items()}

    source_info: dict[tuple[int, int], dict[str, Any]] = {}
    candidates: list[tuple[int, int, set[int]]] = []
    for source in map(int, return_sources):
        cycle = shortest_positive_cycle(
            adjacency,
            signs,
            source,
            maximum_length=maximum_cycle_length,
            forbidden=motor_set,
        )
        if cycle is None:
            continue
        source_returns = return_edges[pre[return_edges] == source]
        for polarity in (-1, 1):
            # During the launch, positive body-Z specific force is larger for light
            # vehicles.  The oracle correction increases with mass, so an activity
            # code with ``polarity`` requires the opposite sensory path sign.
            sensory_path = shortest_signed_path(
                adjacency,
                signs,
                active_inputs,
                source,
                required_sign=-polarity,
                maximum_hops=maximum_sensory_hops,
                forbidden=motor_set,
            )
            if sensory_path is None:
                continue
            coverage = {
                int(post[edge])
                for edge in source_returns
                if int(edge_sign[edge]) * (1 if int(post[edge]) in positive_motor_set else -1)
                == polarity
            }
            if coverage:
                source_info[(source, polarity)] = {
                    "sensory_path": sensory_path,
                    "cycle": cycle,
                }
                candidates.append((source, polarity, coverage))

    required = set(map(int, throttle_motors))
    feasible: list[tuple[tuple[Any, ...], tuple[tuple[int, int, set[int]], ...]]] = []
    for size in range(2, maximum_sources + 1):
        for combination in itertools.combinations(candidates, size):
            sources = [item[0] for item in combination]
            if len(set(sources)) != size:
                continue
            covered = set().union(*(item[2] for item in combination))
            if covered != required:
                continue
            path_lengths = [
                len(source_info[(source, polarity)]["sensory_path"]) - 1
                for source, polarity, _ in combination
            ]
            cycle_lengths = [
                len(source_info[(source, polarity)]["cycle"]) for source, polarity, _ in combination
            ]
            alignments = [
                source_alignment[(source, polarity)]["0.75"] or -1.0
                for source, polarity, _ in combination
            ]
            score = (
                size,
                -min(alignments),
                -sum(alignments),
                max(cycle_lengths),
                sum(cycle_lengths),
                max(path_lengths),
                sum(path_lengths),
                tuple((int(node_ids[source]), polarity) for source, polarity, _ in combination),
            )
            feasible.append((score, combination))
        if feasible:
            break
    if not feasible:
        raise SystemExit(
            "no <=4-source positive-feedback motif has sign-compatible returns to every "
            "throttle motor"
        )
    score, selected = min(feasible, key=lambda item: item[0])
    sources = np.asarray([item[0] for item in selected], dtype=np.int64)
    polarity = np.asarray([item[1] for item in selected], dtype=np.float32)
    polarity_by_source = dict(zip(map(int, sources), map(int, polarity), strict=True))

    selected_returns = np.asarray(
        [
            edge
            for edge in return_edges
            if int(pre[edge]) in polarity_by_source
            and int(edge_sign[edge]) * (1 if int(post[edge]) in positive_motor_set else -1)
            == polarity_by_source[int(pre[edge])]
        ],
        dtype=np.int64,
    )
    if set(map(int, post[selected_returns])) != required:
        raise SystemExit("internal error: selected motif no longer covers every throttle motor")

    motif_edges: set[int] = set()
    motif_nodes: set[int] = set()
    source_records: list[dict[str, Any]] = []
    for source, selected_polarity, coverage in selected:
        path = source_info[(source, selected_polarity)]["sensory_path"]
        cycle = source_info[(source, selected_polarity)]["cycle"]
        for first, second in zip(path[:-1], path[1:], strict=True):
            motif_edges.add(lookup[(first, second)])
        for first, second in zip(cycle, [*cycle[1:], cycle[0]], strict=True):
            motif_edges.add(lookup[(first, second)])
        motif_nodes.update(path)
        motif_nodes.update(cycle)
        source_records.append(
            {
                "source_index": source,
                "source_body_id": int(node_ids[source]),
                "polarity": selected_polarity,
                "covered_motor_body_ids": sorted(int(node_ids[node]) for node in coverage),
                "sensory_path_indices": path,
                "sensory_path_body_ids": [int(node_ids[node]) for node in path],
                "sensory_path_sign_product": int(
                    np.prod(
                        [
                            signs[(first, second)]
                            for first, second in zip(path[:-1], path[1:], strict=True)
                        ]
                    )
                ),
                "cycle_indices": cycle,
                "cycle_body_ids": [int(node_ids[node]) for node in cycle],
                "cycle_sign_product": int(
                    np.prod(
                        [
                            signs[(first, second)]
                            for first, second in zip(cycle, [*cycle[1:], cycle[0]], strict=True)
                        ]
                    )
                ),
                "untrained_alignment": source_alignment[(source, selected_polarity)],
            }
        )
    motif_edges_np = np.asarray(sorted(motif_edges), dtype=np.int64)
    motif_nodes_np = np.asarray(sorted(motif_nodes), dtype=np.int64)
    motor_slot = {int(node): slot for slot, node in enumerate(throttle_motors)}
    return_slots = np.asarray(
        [motor_slot[int(post[edge])] for edge in selected_returns], dtype=np.int64
    )
    device = controller.bias.device
    dtype = controller.bias.dtype
    return {
        "sources": torch.from_numpy(sources).to(device=device),
        "polarity": torch.from_numpy(polarity).to(device=device, dtype=dtype),
        "motif_edges": torch.from_numpy(motif_edges_np).to(device=device),
        "motif_nodes": torch.from_numpy(motif_nodes_np).to(device=device),
        "return_edges": torch.from_numpy(selected_returns).to(device=device),
        "return_slots": torch.from_numpy(return_slots).to(device=device),
        "throttle_motors": torch.from_numpy(throttle_motors).to(device=device),
        "motor_sign": torch.cat(
            (
                torch.ones(len(positive_motors), device=device, dtype=dtype),
                -torch.ones(len(negative_motors), device=device, dtype=dtype),
            )
        ),
        "activity_scale": activity_scale,
        "selection_score": list(score[:-1]),
        "source_records": source_records,
        "motif_node_body_ids": [int(node_ids[node]) for node in motif_nodes_np],
        "motif_edges_body_ids_and_signs": [
            [int(node_ids[pre[edge]]), int(node_ids[post[edge]]), int(edge_sign[edge])]
            for edge in motif_edges_np
        ],
        "return_edges_body_ids_and_signs": [
            [int(node_ids[pre[edge]]), int(node_ids[post[edge]]), int(edge_sign[edge])]
            for edge in selected_returns
        ],
        "motif_node_count": int(len(motif_nodes_np)),
        "motif_edge_count": int(len(motif_edges_np)),
        "return_edge_count": int(len(selected_returns)),
        "active_acceleration_channel": active_acceleration_channel,
    }


def tensor_spec(spec: dict[str, Any], key: str) -> Tensor:
    value = spec[key]
    if not isinstance(value, Tensor):
        raise TypeError(f"motif specification field is not a tensor: {key}")
    return value


def public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if not isinstance(value, Tensor)}


def register_gradient_masks(
    controller: ConnectomeController,
    spec: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    edge_mask = torch.zeros_like(controller.edge_magnitude)
    node_mask = torch.zeros_like(controller.bias)
    edge_mask[tensor_spec(spec, "motif_edges")] = 1.0
    node_mask[tensor_spec(spec, "motif_nodes")] = 1.0
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)
    controller.bias.register_hook(lambda gradient: gradient * node_mask)
    controller.raw_time_constant.register_hook(lambda gradient: gradient * node_mask)
    return edge_mask.bool(), node_mask.bool()


def training_mass(normalized_mass: Tensor, mode: str) -> Tensor:
    if mode == "true":
        return normalized_mass
    if mode == "shuffled":
        return normalized_mass[torch.randperm(len(normalized_mass), device=normalized_mass.device)]
    if mode == "constant":
        return torch.zeros_like(normalized_mass)
    raise ValueError(f"unknown label mode: {mode}")


def schedule_weight(
    time: Tensor,
    *,
    ramp_start_seconds: float,
    established_seconds: float,
) -> Tensor:
    return ((time - ramp_start_seconds) / (established_seconds - ramp_start_seconds)).clamp(
        0.0, 1.0
    )


def correlation(prediction: Tensor, target: Tensor) -> float | None:
    prediction = prediction.detach()
    target = target.detach()
    x = prediction - prediction.mean()
    y = target - target.mean()
    denominator = x.square().mean().sqrt() * y.square().mean().sqrt()
    if float(denominator) <= 1.0e-12:
        return None
    return float((x * y).mean() / denominator)


def activity_metrics(
    activity: Tensor,
    desired_bias: Tensor,
    spec: dict[str, Any],
) -> dict[str, Any]:
    activity = activity.detach()
    desired_bias = desired_bias.detach()
    polarity = tensor_spec(spec, "polarity")
    center = tensor_spec(spec, "activity_center")
    scale = float(spec["activity_scale"])
    intercept = float(spec["calibration_intercept"])
    desired = center + (desired_bias[:, None] - intercept) * polarity / scale
    prediction = intercept + ((activity - center) * polarity).mean(dim=1) * scale
    variance = (desired_bias - desired_bias.mean()).square().mean().clamp_min(1.0e-12)
    result: dict[str, Any] = {
        **regression_metrics(prediction, desired_bias),
        "r2": float(1.0 - (prediction - desired_bias).square().mean() / variance),
        "direct_activity_rmse": float((activity - desired).square().mean().sqrt()),
        "activity_minimum": float(activity.min()),
        "activity_maximum": float(activity.max()),
        "per_source": [],
    }
    for slot in range(activity.shape[1]):
        source_prediction = activity[:, slot] * polarity[slot] * scale
        midpoint = desired_bias.mean()
        light = desired_bias < midpoint
        heavy = ~light
        result["per_source"].append(
            {
                "body_id": spec["source_records"][slot]["source_body_id"],
                "correlation": correlation(source_prediction, desired_bias),
                "rmse": float((source_prediction - desired_bias).square().mean().sqrt()),
                "activity_mean": float(activity[:, slot].mean()),
                "activity_light_mean": float(activity[light, slot].mean()),
                "activity_heavy_mean": float(activity[heavy, slot].mean()),
            }
        )
    return result


def matched_launch_offset(batch: int, magnitude: float, device: torch.device) -> Tensor:
    half = batch // 2
    signs = torch.where(
        torch.arange(half, device=device).remainder(2).bool(),
        1.0,
        -1.0,
    )
    return magnitude * torch.cat((signs, signs))


@torch.no_grad()
def measure_operating_center(
    controller: ConnectomeController,
    spec: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    established_seconds: float,
    maximum_absolute: float,
) -> tuple[Tensor, list[float]]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    sources = tensor_spec(spec, "sources")
    activities: list[Tensor] = []
    for step in range(1, round(prefix_seconds / hover_config.dt) + 1):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
        )
        if step * hover_config.dt >= established_seconds:
            activities.append(torch.tanh(neural[:, sources]))
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
    natural = torch.cat(activities).mean(dim=0)
    center = natural.clamp(-maximum_absolute, maximum_absolute)
    return center, natural.cpu().tolist()


def encoder_rollout(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Any],
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    ramp_start_seconds: float,
    established_seconds: float,
    label_mode: str = "true",
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
    acceleration_noise_mg: float = 0.0,
    activation_noise_std: float = 0.0,
    launch_throttle_offset: float = 0.0,
    neutral_after_seconds: float | None = None,
    pair_loss_weight: float = 1.0,
    temporal_loss_weight: float = 0.10,
    reset_loss_weight: float = 0.25,
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
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    base_neural = base_controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    label = training_mass(normalized_mass, label_mode)
    label_bias = target_bias(label, calibration)
    true_bias = target_bias(normalized_mass, calibration)
    polarity = tensor_spec(spec, "polarity")
    center = tensor_spec(spec, "activity_center")
    sources = tensor_spec(spec, "sources")
    intercept = float(spec["calibration_intercept"])
    desired_full = center + (label_bias[:, None] - intercept) * polarity / float(
        spec["activity_scale"]
    )
    permutation = torch.cat(
        (
            torch.arange(episodes // 2, episodes, device=device),
            torch.arange(episodes // 2, device=device),
        )
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 79_919)
    throttle_offset = matched_launch_offset(episodes, launch_throttle_offset, device)
    times: list[float] = []
    activities: list[Tensor] = []
    base_activities: list[Tensor] = []
    step_count = round(prefix_seconds / hover_config.dt)
    horizon_steps = {
        round(value / hover_config.dt): value
        for value in (0.15, 0.25, 0.50, 0.75, 1.00, prefix_seconds)
        if value <= prefix_seconds
    }
    horizons: dict[str, Any] = {}
    for step in range(1, step_count + 1):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        sensed_image = image
        sensed_attitude = state.euler[:, :2]
        sensed_force = state.specific_force
        if neutral_after_seconds is not None and step > round(
            neutral_after_seconds / hover_config.dt
        ):
            sensed_image = torch.zeros_like(image)
            sensed_attitude = torch.zeros_like(sensed_attitude)
            sensed_force = torch.zeros_like(sensed_force)
            sensed_force[:, 2] = 9.81
        elif frozen_acceleration:
            sensed_force = torch.zeros_like(sensed_force)
            sensed_force[:, 2] = 9.81
        elif swapped_acceleration:
            sensed_force = sensed_force[permutation]
        if acceleration_noise_mg:
            sensed_force = sensed_force.clone()
            sensed_force[:, 2] += torch.randn(
                episodes,
                device=device,
                dtype=sensed_force.dtype,
                generator=generator,
            ) * (acceleration_noise_mg * 1.0e-3 * 9.81)
        _, neural = controller(
            sensed_image,
            sensed_attitude,
            neural,
            sensed_force,
        )
        if activation_noise_std:
            perturbation = (
                torch.randn(
                    episodes,
                    len(sources),
                    device=device,
                    dtype=neural.dtype,
                    generator=generator,
                )
                * activation_noise_std
            )
            neural = neural.index_add(1, sources, perturbation)
        activity = torch.tanh(neural[:, sources])
        with torch.no_grad():
            base_motor, base_neural = base_controller(
                image,
                state.euler[:, :2],
                base_neural,
                state.specific_force,
            )
            base_activity = torch.tanh(base_neural[:, sources])
        activities.append(activity)
        base_activities.append(base_activity)
        time = step * hover_config.dt
        times.append(time)
        if step in horizon_steps:
            horizons[f"{horizon_steps[step]:g}"] = activity_metrics(
                activity,
                true_bias,
                spec,
            )
        with torch.no_grad():
            base_motor = base_motor.clone()
            base_motor[:, 3] = (base_motor[:, 3] + throttle_offset).clamp(-1.0, 1.0)
            rc, stick_state = sticks(base_motor, stick_state)
            state = quad(rc, state, mass_scale)

    activity_history = torch.stack(activities)
    base_activity_history = torch.stack(base_activities)
    time_tensor = torch.tensor(times, device=device, dtype=activity_history.dtype)
    weight = schedule_weight(
        time_tensor,
        ramp_start_seconds=ramp_start_seconds,
        established_seconds=established_seconds,
    )
    desired_history = (1.0 - weight[:, None, None]) * base_activity_history + weight[
        :, None, None
    ] * desired_full[None]
    middle_end = min(established_seconds + 0.25, prefix_seconds)
    window_masks = (
        (time_tensor >= ramp_start_seconds) & (time_tensor < established_seconds),
        (time_tensor >= established_seconds) & (time_tensor < middle_end),
        time_tensor >= middle_end,
    )
    window_losses = [
        (activity_history[mask] - desired_history[mask]).square().mean() for mask in window_masks
    ]
    direct = torch.stack(window_losses).mean()
    half = episodes // 2
    pair_difference = activity_history[:, :half] - activity_history[:, half:]
    desired_pair_difference = desired_history[:, :half] - desired_history[:, half:]
    pair = torch.stack(
        [
            (pair_difference[mask] - desired_pair_difference[mask]).square().mean()
            for mask in window_masks
        ]
    ).mean()
    stable = activity_history[time_tensor >= established_seconds]
    temporal = (stable - stable.mean(dim=0, keepdim=True)).square().mean()
    reset = (
        (activity_history[time_tensor <= 0.15] - base_activity_history[time_tensor <= 0.15])
        .square()
        .mean()
    )
    loss = (
        direct
        + pair_loss_weight * pair
        + temporal_loss_weight * temporal
        + reset_loss_weight * reset
    )
    return loss, {
        "horizons": horizons,
        "losses": {
            "total": float(loss.detach()),
            "direct": float(direct.detach()),
            "matched_pair": float(pair.detach()),
            "temporal": float(temporal.detach()),
            "reset": float(reset.detach()),
            "window_direct": [float(value.detach()) for value in window_losses],
        },
    }


def fit_return_edges(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Any],
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    ramp_start_seconds: float,
    established_seconds: float,
    stride: int,
) -> dict[str, Any]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    base_neural = base_controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired_bias = target_bias(normalized_mass, calibration)
    edges = tensor_spec(spec, "return_edges")
    slots = tensor_spec(spec, "return_slots")
    motors = tensor_spec(spec, "throttle_motors")
    motor_sign = tensor_spec(spec, "motor_sign")
    features: list[np.ndarray] = []
    right_hand_sides: list[np.ndarray] = []
    target_currents: list[np.ndarray] = []
    candidate_without_deltas: list[np.ndarray] = []
    labels: list[float] = []
    step_count = round(prefix_seconds / hover_config.dt)
    for step in range(step_count):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        with torch.no_grad():
            candidate_drive = preactivation_drive(
                controller,
                image,
                state.euler[:, :2],
                neural,
                state.specific_force,
            )
            reference_drive = preactivation_drive(
                base_controller,
                image,
                state.euler[:, :2],
                base_neural,
                state.specific_force,
            )
            edge_features = (
                torch.tanh(neural[:, controller.edge_pre[edges]]) * controller.edge_sign[edges]
            )
            selected_current = torch.zeros(
                episodes,
                len(motors),
                device=device,
                dtype=neural.dtype,
            ).index_add(1, slots, edge_features * controller.edge_magnitude[edges])
            candidate_without = candidate_drive[:, motors] - selected_current
            time = (step + 1) * hover_config.dt
            schedule = schedule_weight(
                torch.tensor(time, device=device),
                ramp_start_seconds=ramp_start_seconds,
                established_seconds=established_seconds,
            )
            target_current = schedule * desired_bias[:, None] * motor_sign
            target_total = reference_drive[:, motors] + target_current
            if step % stride == 0:
                features.append(edge_features.cpu().double().numpy())
                right_hand_sides.append((target_total - candidate_without).cpu().double().numpy())
                target_currents.append(target_current.cpu().double().numpy())
                candidate_without_deltas.append(
                    (candidate_without - reference_drive[:, motors]).cpu().double().numpy()
                )
                labels.extend([time] * episodes)
            _, neural = controller(
                image,
                state.euler[:, :2],
                neural,
                state.specific_force,
            )
            base_motor, base_neural = base_controller(
                image,
                state.euler[:, :2],
                base_neural,
                state.specific_force,
            )
            rc, stick_state = sticks(base_motor, stick_state)
            state = quad(rc, state, mass_scale)

    design = np.concatenate(features)
    rhs = np.concatenate(right_hand_sides)
    target_current_np = np.concatenate(target_currents)
    candidate_without_np = np.concatenate(candidate_without_deltas)
    slots_np = slots.detach().cpu().numpy()
    fitted = np.zeros(len(edges), dtype=np.float64)
    per_motor: list[dict[str, Any]] = []
    for slot in range(len(motors)):
        selected = np.flatnonzero(slots_np == slot)
        matrix = design[:, selected]
        result = lsq_linear(matrix, rhs[:, slot], bounds=(0.0, 8.0), tol=1.0e-12)
        fitted[selected] = result.x
        singular = np.linalg.svd(matrix, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1.0e-12 else None
        per_motor.append(
            {
                "motor_body_id": int(controller.node_ids[motors[slot]].item()),
                "edge_count": int(len(selected)),
                "condition_number": condition,
                "coefficient_minimum": float(result.x.min()),
                "coefficient_maximum": float(result.x.max()),
                "solver_cost": float(result.cost),
                "solver_optimality": float(result.optimality),
                "solver_status": int(result.status),
            }
        )
    with torch.no_grad():
        controller.edge_magnitude[edges] = torch.from_numpy(fitted).to(
            device=device,
            dtype=controller.edge_magnitude.dtype,
        )
    predicted = candidate_without_np.copy()
    for edge_slot, motor_slot in enumerate(slots_np):
        predicted[:, motor_slot] += design[:, edge_slot] * fitted[edge_slot]
    labels_np = np.asarray(labels)
    return {
        "edge_magnitudes": fitted.tolist(),
        "per_motor": per_motor,
        "coefficient_minimum": float(fitted.min()),
        "coefficient_maximum": float(fitted.max()),
        "coefficient_l2_norm": float(np.linalg.norm(fitted)),
        "overall_per_motor_current_rmse": float(
            np.sqrt(np.mean((predicted - target_current_np) ** 2))
        ),
        "per_horizon": correction_metrics_by_horizon(
            predicted,
            target_current_np,
            labels_np,
            motor_sign.detach().cpu().numpy(),
            hover_config.dt,
        ),
    }


def correction_metrics_by_horizon(
    prediction: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    motor_sign: np.ndarray,
    dt: float,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for horizon in (0.25, 0.50, 0.75, 1.00, 1.50):
        selected = np.isclose(times, horizon, atol=0.51 * dt)
        if not selected.any():
            continue
        predicted_bias = (prediction[selected] * motor_sign).mean(axis=1)
        target_bias_at_horizon = (target[selected] * motor_sign).mean(axis=1)
        variance = np.mean((target_bias_at_horizon - target_bias_at_horizon.mean()) ** 2)
        metrics = regression_metrics(
            torch.from_numpy(predicted_bias),
            torch.from_numpy(target_bias_at_horizon),
        )
        metrics["r2"] = (
            float(1.0 - np.mean((predicted_bias - target_bias_at_horizon) ** 2) / variance)
            if variance > 1.0e-12
            else None
        )
        metrics["per_motor_current_rmse"] = float(
            np.sqrt(np.mean((prediction[selected] - target[selected]) ** 2))
        )
        results[f"{horizon:g}"] = metrics
    return results


@torch.no_grad()
def audit_return_fit(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Any],
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_seconds: float,
    ramp_start_seconds: float,
    established_seconds: float,
) -> dict[str, Any]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    base_neural = base_controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired_bias = target_bias(normalized_mass, calibration)
    motors = tensor_spec(spec, "throttle_motors")
    motor_sign = tensor_spec(spec, "motor_sign")
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    times: list[float] = []
    step_count = round(prefix_seconds / hover_config.dt)
    for step in range(step_count):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        candidate_drive = preactivation_drive(
            controller,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
        )
        reference_drive = preactivation_drive(
            base_controller,
            image,
            state.euler[:, :2],
            base_neural,
            state.specific_force,
        )
        time = (step + 1) * hover_config.dt
        schedule = schedule_weight(
            torch.tensor(time, device=device),
            ramp_start_seconds=ramp_start_seconds,
            established_seconds=established_seconds,
        )
        predictions.append((candidate_drive[:, motors] - reference_drive[:, motors]).cpu().numpy())
        targets.append((schedule * desired_bias[:, None] * motor_sign).cpu().numpy())
        times.extend([time] * episodes)
        _, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
        )
        base_motor, base_neural = base_controller(
            image,
            state.euler[:, :2],
            base_neural,
            state.specific_force,
        )
        rc, stick_state = sticks(base_motor, stick_state)
        state = quad(rc, state, mass_scale)
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    return {
        "overall_per_motor_current_rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "per_horizon": correction_metrics_by_horizon(
            prediction,
            target,
            np.asarray(times),
            motor_sign.cpu().numpy(),
            hover_config.dt,
        ),
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    source_checkpoint: dict[str, Any],
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    selected_update: int,
    kind: str,
) -> None:
    torch.save(
        {
            **copy.deepcopy(source_checkpoint),
            "controller": {
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            "graph_sha256": file_sha256(args.graph),
            "native_latent_motif": {
                "kind": kind,
                "selected_update": selected_update,
                "label_mode": args.label_mode,
                "motif": public_spec(spec),
            },
        },
        path,
    )


def stable_encoding_pass(
    probe: dict[str, Any],
    neutral: dict[str, Any],
    established_seconds: float,
) -> bool:
    established_key = f"{established_seconds:g}"
    later_keys = [key for key in ("0.75", "1", "1.5") if float(key) > established_seconds]
    live_horizons = [probe["horizons"][established_key]] + [
        probe["horizons"][key] for key in later_keys
    ]
    neutral_horizons = [neutral["horizons"][key] for key in later_keys]
    checks = [*live_horizons, *neutral_horizons]
    return all(
        item["r2"] >= 0.90
        and item["direct_activity_rmse"] <= 0.10
        and 0.75 <= item["signed_slope"] <= 1.25
        for item in checks
    )


def main() -> int:
    args = parse_args()
    source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    hover_config = HoverConfig(**source_checkpoint["hover_config"])
    validate_args(args, hover_config.dt)
    if source_checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(source_checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source_checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    gate_config = GateConfig(**source_checkpoint["gate_config"])
    resolution = int(source_checkpoint["image_resolution"])
    base_controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source_checkpoint["retinal_receptive_field"]),
    ).to(device)
    base_controller.load_state_dict(source_checkpoint["controller"])
    base_controller.eval()
    for parameter in base_controller.parameters():
        parameter.requires_grad_(False)
    controller = copy.deepcopy(base_controller)
    controller.train()
    for parameter in controller.parameters():
        parameter.requires_grad_(True)
    source_alignment = return_source_alignment(
        base_controller,
        args.graph,
        episodes=args.selection_episodes,
        seed=args.selection_seed - 2,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    spec = motif_spec(
        controller,
        args.graph,
        source_alignment=source_alignment,
        active_acceleration_channel=args.active_acceleration_channel,
        maximum_sensory_hops=args.maximum_sensory_hops,
        maximum_cycle_length=args.maximum_cycle_length,
        maximum_sources=args.maximum_sources,
        activity_scale=args.activity_scale,
    )
    center, natural_center = measure_operating_center(
        base_controller,
        spec,
        episodes=args.selection_episodes,
        seed=args.selection_seed - 1,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        prefix_seconds=args.prefix_seconds,
        established_seconds=args.established_seconds,
        maximum_absolute=args.maximum_center_absolute,
    )
    spec["activity_center"] = center
    spec["activity_center_values"] = center.cpu().tolist()
    spec["natural_activity_center_values"] = natural_center
    spec["calibration_intercept"] = float(calibration["intercept"])
    edge_mask, node_mask = register_gradient_masks(controller, spec)
    initial = {
        "edge_magnitude": controller.edge_magnitude.detach().clone(),
        "bias": controller.bias.detach().clone(),
        "raw_time_constant": controller.raw_time_constant.detach().clone(),
    }
    optimizer = torch.optim.Adam(
        (
            {"params": [controller.edge_magnitude], "lr": args.edge_learning_rate},
            {"params": [controller.bias], "lr": args.bias_learning_rate},
            {
                "params": [controller.raw_time_constant],
                "lr": args.time_constant_learning_rate,
            },
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_encoder_path = args.output_dir / "best-encoder.pt"
    selection_checkpoint_dir = args.output_dir / "selection-checkpoints"
    if args.save_selection_checkpoints:
        selection_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate.pt"
    progress_path = args.output_dir / "progress.json"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    started = perf_counter()
    print(json.dumps({"phase": "motif", **public_spec(spec)}), flush=True)

    for update in range(1, args.updates + 1):
        controller.train()
        loss, training = encoder_rollout(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.batch_size,
            seed=args.seed + update - 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            ramp_start_seconds=args.ramp_start_seconds,
            established_seconds=args.established_seconds,
            label_mode=args.label_mode,
            neutral_after_seconds=(
                args.established_seconds if update % args.neutral_training_every == 0 else None
            ),
            pair_loss_weight=args.pair_loss_weight,
            temporal_loss_weight=args.temporal_loss_weight,
            reset_loss_weight=args.reset_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite motif gradient")
        optimizer.step()
        controller.project_parameters()
        if update % args.selection_interval and update != args.updates:
            if update == 1 or update % 10 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "training",
                            "update": update,
                            "gradient_norm": float(gradient_norm),
                            **training,
                        }
                    ),
                    flush=True,
                )
            continue
        controller.eval()
        with torch.no_grad():
            _, probe = encoder_rollout(
                controller,
                base_controller,
                spec,
                calibration,
                episodes=args.selection_episodes,
                seed=args.selection_seed,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                prefix_seconds=args.prefix_seconds,
                ramp_start_seconds=args.ramp_start_seconds,
                established_seconds=args.established_seconds,
                pair_loss_weight=args.pair_loss_weight,
                temporal_loss_weight=args.temporal_loss_weight,
                reset_loss_weight=args.reset_loss_weight,
            )
            _, neutral = encoder_rollout(
                controller,
                base_controller,
                spec,
                calibration,
                episodes=args.selection_episodes,
                seed=args.selection_seed,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                prefix_seconds=args.prefix_seconds,
                ramp_start_seconds=args.ramp_start_seconds,
                established_seconds=args.established_seconds,
                neutral_after_seconds=args.established_seconds,
                pair_loss_weight=args.pair_loss_weight,
                temporal_loss_weight=args.temporal_loss_weight,
                reset_loss_weight=args.reset_loss_weight,
            )
        established_key = f"{args.established_seconds:g}"
        later_keys = [key for key in ("0.75", "1", "1.5") if float(key) > args.established_seconds]
        live_items = [probe["horizons"][established_key]] + [
            probe["horizons"][key] for key in later_keys
        ]
        neutral_items = [neutral["horizons"][key] for key in later_keys]
        key = (
            -max(item["direct_activity_rmse"] for item in [*live_items, *neutral_items]),
            min(item["r2"] for item in [*live_items, *neutral_items]),
            -float(probe["losses"]["temporal"]),
        )
        selection_checkpoint_path: Path | None = None
        if args.save_selection_checkpoints:
            selection_checkpoint_path = selection_checkpoint_dir / f"update-{update:04d}.pt"
            save_checkpoint(
                selection_checkpoint_path,
                controller,
                source_checkpoint,
                args,
                spec,
                selected_update=update,
                kind="readout_audit_selection_checkpoint",
            )
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(
                best_encoder_path,
                controller,
                source_checkpoint,
                args,
                spec,
                selected_update=update,
                kind="held_out_persistent_encoder",
            )
        entry = {
            "update": update,
            "gradient_norm": float(gradient_norm),
            "training": training,
            "held_out_live": probe,
            "held_out_neutral_after_establishment": neutral,
            "selection_checkpoint": (
                stable_path(selection_checkpoint_path)
                if selection_checkpoint_path is not None
                else None
            ),
            "selection_key": key,
            "best_key": best_key,
        }
        history.append(entry)
        print(json.dumps({"phase": "selection", **entry}), flush=True)
        progress_path.write_text(
            json.dumps({"history": history, "best_key": best_key}, indent=2, sort_keys=True) + "\n"
        )

    best = torch.load(best_encoder_path, map_location=device, weights_only=True)
    controller.load_state_dict(best["controller"])
    controller.eval()
    selected_update = int(best["native_latent_motif"]["selected_update"])
    with torch.no_grad():
        _, best_probe = encoder_rollout(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.selection_episodes,
            seed=args.selection_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            ramp_start_seconds=args.ramp_start_seconds,
            established_seconds=args.established_seconds,
            pair_loss_weight=args.pair_loss_weight,
            temporal_loss_weight=args.temporal_loss_weight,
            reset_loss_weight=args.reset_loss_weight,
        )
        _, best_neutral = encoder_rollout(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.selection_episodes,
            seed=args.selection_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            ramp_start_seconds=args.ramp_start_seconds,
            established_seconds=args.established_seconds,
            neutral_after_seconds=args.established_seconds,
            pair_loss_weight=args.pair_loss_weight,
            temporal_loss_weight=args.temporal_loss_weight,
            reset_loss_weight=args.reset_loss_weight,
        )
    encoding_passed = stable_encoding_pass(
        best_probe,
        best_neutral,
        args.established_seconds,
    )
    return_fit: dict[str, Any] | None = None
    post_fit: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    baseline: dict[str, Any] | None = None
    controls: dict[str, Any] = {}
    if encoding_passed and args.label_mode == "true":
        return_fit = fit_return_edges(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.fit_episodes,
            seed=args.fit_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            ramp_start_seconds=args.ramp_start_seconds,
            established_seconds=args.established_seconds,
            stride=args.fit_stride,
        )
        post_fit = audit_return_fit(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.selection_episodes,
            seed=args.selection_seed + 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            prefix_seconds=args.prefix_seconds,
            ramp_start_seconds=args.ramp_start_seconds,
            established_seconds=args.established_seconds,
        )
        print(json.dumps({"phase": "return_fit", "fit": return_fit, "post_fit": post_fit}))
        save_checkpoint(
            candidate_path,
            controller,
            source_checkpoint,
            args,
            spec,
            selected_update=selected_update,
            kind="bounded_anatomical_return_fit",
        )
        with torch.no_grad():
            control_specs = {
                "constant_1g": {"frozen_acceleration": True},
                "pair_swapped_acceleration": {"swapped_acceleration": True},
                "neutral_after_establishment": {"neutral_after_seconds": args.established_seconds},
                "acceleration_noise": {"acceleration_noise_mg": args.acceleration_noise_mg},
                "activation_noise": {"activation_noise_std": args.activation_noise_std},
                "launch_command_perturbation": {
                    "launch_throttle_offset": args.launch_throttle_offset
                },
            }
            for name, kwargs in control_specs.items():
                _, controls[name] = encoder_rollout(
                    controller,
                    base_controller,
                    spec,
                    calibration,
                    episodes=args.selection_episodes,
                    seed=args.selection_seed + 2,
                    resolution=resolution,
                    device=device,
                    hover_config=hover_config,
                    gate_config=gate_config,
                    prefix_seconds=args.prefix_seconds,
                    ramp_start_seconds=args.ramp_start_seconds,
                    established_seconds=args.established_seconds,
                    pair_loss_weight=args.pair_loss_weight,
                    temporal_loss_weight=args.temporal_loss_weight,
                    reset_loss_weight=args.reset_loss_weight,
                    **kwargs,
                )
        final_specs = {
            "live_sensors": {},
            "constant_1g_acceleration": {"frozen_acceleration": True},
            "mass_rank_swapped_acceleration": {"swapped_acceleration": True},
            "frozen_first_frame": {"frozen_visual": True},
        }
        clean_radius = gate_config.inner_radius - gate_config.drone_radius
        for name, kwargs in final_specs.items():
            final[name] = evaluate_gate(
                controller,
                episodes=args.final_episodes,
                seconds=12.0,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                seed=args.final_seed,
                balanced_strata=True,
                **kwargs,
            )
            print(
                json.dumps(
                    {"phase": "final", "name": name, **compact_metrics(final[name], clean_radius)}
                ),
                flush=True,
            )
        baseline = evaluate_gate(
            base_controller,
            episodes=args.final_episodes,
            seconds=12.0,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.final_seed,
            balanced_strata=True,
        )

    return_edges = tensor_spec(spec, "return_edges")
    frozen_edge_mask = ~edge_mask
    frozen_edge_mask[return_edges] = False
    drift = {
        "frozen_nonreturn_edge_max_absolute_change": float(
            (
                controller.edge_magnitude.detach()[frozen_edge_mask]
                - initial["edge_magnitude"][frozen_edge_mask]
            )
            .abs()
            .max()
        ),
        "frozen_bias_max_absolute_change": float(
            (controller.bias.detach()[~node_mask] - initial["bias"][~node_mask]).abs().max()
        ),
        "frozen_time_constant_max_absolute_change": float(
            (
                controller.raw_time_constant.detach()[~node_mask]
                - initial["raw_time_constant"][~node_mask]
            )
            .abs()
            .max()
        ),
    }
    return_fit_passed = bool(
        post_fit
        and all(
            post_fit["per_horizon"][key]["r2"] is not None
            and post_fit["per_horizon"][key]["r2"] >= 0.90
            and 0.75 <= post_fit["per_horizon"][key]["signed_slope"] <= 1.25
            for key in (
                f"{args.established_seconds:g}",
                *[item for item in ("0.75", "1", "1.5") if float(item) > args.established_seconds],
            )
        )
    )
    live_success = final.get("live_sensors", {}).get("success_rate", 0.0)
    causal_success = max(
        final.get("constant_1g_acceleration", {}).get("success_rate", 0.0),
        final.get("mass_rank_swapped_acceleration", {}).get("success_rate", 0.0),
    )
    flight_causality = live_success >= causal_success + 0.05
    goal_passed = bool(
        final.get("live_sensors", {}).get("goal_pass", False)
        and encoding_passed
        and return_fit_passed
        and flight_causality
    )
    report = {
        "experiment": "bounded native recurrent launch-correction motif",
        "claim_scope": (
            "Exact mass is a training label only. Runtime inputs remain FPV, roll/pitch, "
            "instantaneous body specific force, and persistent connectome state; no external "
            "history feature, clock, mass estimate, or actor state machine is added."
        ),
        "external_runtime_parameters_added": 0,
        "external_history_features_added": 0,
        "external_actor_state_machine": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "best_encoder_checkpoint": stable_path(best_encoder_path),
        "candidate_checkpoint": stable_path(candidate_path) if candidate_path.is_file() else None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "motif": public_spec(spec),
        "training": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "matched_geometry_light_heavy_pairs": True,
            "oracle_corrected_training_physics": False,
            "whole_episode_held_out_seed": args.selection_seed,
        },
        "history": history,
        "selected_update": selected_update,
        "best_encoder_live": best_probe,
        "best_encoder_neutral_after_establishment": best_neutral,
        "return_fit": return_fit,
        "post_fit_replay_audit": post_fit,
        "encoder_controls": controls,
        "proprioception_yoking": "not_applicable_graph_has_no_proprioception_input",
        "baseline": baseline,
        "final": final,
        "parameter_drift": drift,
        "causal_checks": {
            "trained_with_true_mass_labels": args.label_mode == "true",
            "stable_encoding_passed": encoding_passed,
            "return_fit_passed": return_fit_passed,
            "flight_acceleration_causality": flight_causality,
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
                "selected_update": selected_update,
                "encoding_passed": encoding_passed,
                "return_fit_passed": return_fit_passed,
                "live_success_rate": live_success,
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
