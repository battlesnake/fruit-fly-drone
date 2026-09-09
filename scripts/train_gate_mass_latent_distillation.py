#!/usr/bin/env python3
"""Distill launch mass into an early recurrent code, then fit anatomical returns."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from search_gate_acceleration_es import compact_metrics  # noqa: E402
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    edges_on_short_paths,
    evaluate_gate,
    file_sha256,
    remap_anatomical_parameters,
    seed_everything,
)
from train_gate_mass_current_distillation import (  # noqa: E402
    label_mass,
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
        default=REPO_ROOT / "artifacts" / "gate-proprio-diagnostic-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--source-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "mass-latent-distillation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--edge-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--bias-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--selection-interval", type=int, default=15)
    parser.add_argument("--selection-episodes", type=int, default=512)
    parser.add_argument("--fit-episodes", type=int, default=2048)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=300_031)
    parser.add_argument("--selection-seed", type=int, default=310_031)
    parser.add_argument("--fit-seed", type=int, default=320_031)
    parser.add_argument("--final-seed", type=int, default=330_031)
    parser.add_argument(
        "--label-mode",
        choices=("true", "pair_swapped", "constant"),
        default="true",
        help="Training-only causal control for the latent target.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.source_checkpoint, args.source_graph, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.updates,
        args.batch_size,
        args.edge_learning_rate,
        args.bias_learning_rate,
        args.time_constant_learning_rate,
        args.gradient_clip,
        args.selection_interval,
        args.selection_episodes,
        args.fit_episodes,
        args.final_episodes,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, rates, and intervals must be positive")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even for matched-mass pairs")
    for name in ("selection_episodes", "fit_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")


def _minimal_compatible_sources(
    pre: np.ndarray,
    post: np.ndarray,
    edge_sign: np.ndarray,
    return_edges: np.ndarray,
    positive_motors: np.ndarray,
    throttle_motors: np.ndarray,
    reachable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the smallest source/polarity set whose compatible edges cover all motors."""

    candidates: list[tuple[int, int, set[int]]] = []
    positive_set = set(map(int, positive_motors))
    for source in np.unique(pre[return_edges]):
        edges = return_edges[pre[return_edges] == source]
        if not reachable[source]:
            continue
        for polarity in (-1, 1):
            compatible_motors = {
                int(post[edge])
                for edge in edges
                if int(edge_sign[edge]) * (1 if int(post[edge]) in positive_set else -1) == polarity
            }
            if compatible_motors:
                candidates.append((int(source), polarity, compatible_motors))
    required = set(map(int, throttle_motors))
    for size in range(1, len(candidates) + 1):
        for combination in itertools.combinations(candidates, size):
            if len({source for source, _, _ in combination}) != size:
                continue
            covered: set[int] = set()
            for _, _, motors in combination:
                covered.update(motors)
            if covered == required:
                return (
                    np.asarray([source for source, _, _ in combination], dtype=np.int64),
                    np.asarray([polarity for _, polarity, _ in combination], dtype=np.float32),
                )
    raise SystemExit("no compatible anatomical return set covers every throttle motor")


def circuit_spec(
    controller: ConnectomeController,
    source_graph: Path,
    *,
    max_hops: int = 8,
) -> dict[str, Tensor | int | float | list[int]]:
    with np.load(source_graph) as source:
        source_ids = set(map(int, source["node_ids"]))
    node_ids = controller.node_ids.detach().cpu().numpy()
    edge_pre = controller.edge_pre.detach().cpu().numpy()
    edge_post = controller.edge_post.detach().cpu().numpy()
    edge_sign = controller.edge_sign.detach().cpu().numpy()
    new_node = np.asarray([int(body) not in source_ids for body in node_ids])
    pool_offsets = controller.pool_offsets.detach().cpu().numpy()
    pool_indices = controller.pool_indices.detach().cpu().numpy()
    positive_motors = pool_indices[pool_offsets[6] : pool_offsets[7]]
    negative_motors = pool_indices[pool_offsets[7] : pool_offsets[8]]
    throttle_motors = np.concatenate((positive_motors, negative_motors))
    all_returns = np.flatnonzero(new_node[edge_pre] & np.isin(edge_post, throttle_motors))

    adjacency = csr_matrix(
        (np.ones(len(edge_pre), dtype=np.int8), (edge_pre, edge_post)),
        shape=(len(node_ids), len(node_ids)),
    )
    acceleration_nodes = controller.acceleration_nodes.detach().cpu().numpy()
    distance = shortest_path(
        adjacency,
        directed=True,
        indices=acceleration_nodes,
        unweighted=True,
    )
    reachable = np.isfinite(distance).any(axis=0) & (distance.min(axis=0) <= max_hops)
    encoder_nodes, encoder_polarity = _minimal_compatible_sources(
        edge_pre,
        edge_post,
        edge_sign,
        all_returns,
        positive_motors,
        throttle_motors,
        reachable,
    )
    polarity_by_source = {
        int(source): int(polarity)
        for source, polarity in zip(encoder_nodes, encoder_polarity, strict=True)
    }
    positive_motor_set = set(map(int, positive_motors))
    throttle_motor_set = set(map(int, throttle_motors))
    return_edge_mask = np.asarray(
        [
            int(edge_pre[edge]) in polarity_by_source
            and int(edge_post[edge]) in throttle_motor_set
            and int(edge_sign[edge]) * (1 if int(edge_post[edge]) in positive_motor_set else -1)
            == polarity_by_source[int(edge_pre[edge])]
            for edge in range(len(edge_pre))
        ]
    )
    return_edges = np.flatnonzero(return_edge_mask)
    motor_slot = {int(node): index for index, node in enumerate(throttle_motors)}
    return_slots = np.asarray(
        [motor_slot[int(edge_post[edge])] for edge in return_edges], dtype=np.int64
    )
    if set(map(int, edge_post[return_edges])) != set(map(int, throttle_motors)):
        raise SystemExit("selected return sources do not cover every throttle motor")

    encoder_path = edges_on_short_paths(
        edge_pre,
        edge_post,
        len(node_ids),
        acceleration_nodes,
        encoder_nodes,
        max_hops=max_hops,
    )
    trainable_edge = encoder_path & new_node[edge_post] & ~return_edge_mask
    path_nodes = np.unique(np.concatenate((edge_pre[trainable_edge], edge_post[trainable_edge])))
    trainable_node = new_node & np.isin(np.arange(len(node_ids)), path_nodes)
    device = controller.bias.device
    dtype = controller.bias.dtype
    return {
        "new_node_mask": torch.from_numpy(new_node).to(device=device, dtype=dtype),
        "trainable_node_mask": torch.from_numpy(trainable_node).to(device=device, dtype=dtype),
        "trainable_edge_mask": torch.from_numpy(trainable_edge).to(device=device, dtype=dtype),
        "encoder_nodes": torch.from_numpy(encoder_nodes).to(device=device),
        "encoder_polarity": torch.from_numpy(encoder_polarity).to(device=device, dtype=dtype),
        "return_edges": torch.from_numpy(return_edges).to(device=device),
        "return_slots": torch.from_numpy(return_slots).to(device=device),
        "motor_sign": torch.cat(
            (
                torch.ones(len(positive_motors), device=device, dtype=dtype),
                -torch.ones(len(negative_motors), device=device, dtype=dtype),
            )
        ),
        "positive_motor_count": len(positive_motors),
        "encoder_body_ids": [int(node_ids[node]) for node in encoder_nodes],
        "trainable_node_count": int(trainable_node.sum()),
        "trainable_edge_count": int(trainable_edge.sum()),
        "return_edge_count": int(return_edge_mask.sum()),
        "minimum_acceleration_hops": int(distance[:, encoder_nodes].min()),
        "maximum_minimum_acceleration_hops": int(distance[:, encoder_nodes].min(axis=0).max()),
    }


def register_gradient_masks(
    controller: ConnectomeController,
    spec: dict[str, Tensor | int | float | list[int]],
) -> None:
    node_mask = spec["trainable_node_mask"]
    edge_mask = spec["trainable_edge_mask"]
    assert isinstance(node_mask, Tensor) and isinstance(edge_mask, Tensor)
    controller.bias.register_hook(lambda gradient: gradient * node_mask)
    controller.raw_time_constant.register_hook(lambda gradient: gradient * node_mask)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)


def code_scale(calibration: dict[str, Any]) -> float:
    endpoints = (
        calibration["intercept"] - calibration["mass_slope"],
        calibration["intercept"] + calibration["mass_slope"],
    )
    return max(map(abs, endpoints))


def encoded_mass(
    controller: ConnectomeController,
    neural: Tensor,
    spec: dict[str, Tensor | int | float | list[int]],
) -> tuple[Tensor, Tensor]:
    nodes = spec["encoder_nodes"]
    polarity = spec["encoder_polarity"]
    assert isinstance(nodes, Tensor) and isinstance(polarity, Tensor)
    activities = torch.tanh(neural[:, nodes])
    return (activities * polarity).mean(dim=1), activities


def latent_metrics(
    prediction: Tensor, target: Tensor, activities: Tensor, desired: Tensor
) -> dict[str, float]:
    metrics = regression_metrics(prediction, target)
    prediction = prediction.detach()
    target = target.detach()
    activities = activities.detach()
    desired = desired.detach()
    variance = (target - target.mean()).square().mean().clamp_min(1.0e-12)
    metrics.update(
        r2=float(1.0 - (prediction - target).square().mean() / variance),
        direct_activity_rmse=float((activities - desired).square().mean().sqrt()),
    )
    return metrics


def encoder_rollout(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Tensor | int | float | list[int]],
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    label_mode: str = "true",
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
    frozen_proprioception: bool = False,
    swapped_proprioception: bool = False,
) -> tuple[Tensor, dict[str, dict[str, float]]]:
    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    base_neural = base_controller.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    training_mass = label_mass(normalized_mass, label_mode)
    target = target_bias(training_mass, calibration) / code_scale(calibration)
    nodes = spec["encoder_nodes"]
    polarity = spec["encoder_polarity"]
    assert isinstance(nodes, Tensor) and isinstance(polarity, Tensor)
    desired = target[:, None] * polarity
    horizons = (0.10, 0.25, 0.50, 0.75, 1.00)
    horizon_steps = {round(value / hover_config.dt): value for value in horizons}
    losses: list[Tensor] = []
    results: dict[str, dict[str, float]] = {}
    permutation = torch.cat(
        (
            torch.arange(episodes // 2, episodes, device=device),
            torch.arange(episodes // 2, device=device),
        )
    )
    for step in range(1, max(horizon_steps) + 1):
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
        sensed_stick = stick_state.position
        if frozen_proprioception:
            sensed_stick = torch.zeros_like(sensed_stick)
            sensed_stick[:, 3] = -1.0
        elif swapped_proprioception:
            sensed_stick = sensed_stick[permutation]
        _, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            specific_force,
            sensed_stick,
        )
        with torch.no_grad():
            base_motor, base_neural = base_controller(
                image,
                state.euler[:, :2],
                base_neural,
                state.specific_force,
            )
            rc, stick_state = sticks(base_motor, stick_state)
            state = quad(rc, state, mass_scale)
        if step in horizon_steps:
            prediction, activities = encoded_mass(controller, neural, spec)
            anchor = step == min(horizon_steps)
            horizon_target = torch.zeros_like(target) if anchor else target
            horizon_desired = torch.zeros_like(desired) if anchor else desired
            losses.append((activities - horizon_desired).square().mean())
            results[f"{horizon_steps[step]:g}"] = latent_metrics(
                prediction,
                horizon_target,
                activities,
                horizon_desired,
            )
    return torch.stack(losses).mean(), results


def fit_return_edges(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Tensor | int | float | list[int]],
    calibration: dict[str, Any],
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
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
    base_neural = base_controller.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired_bias = target_bias(normalized_mass, calibration)
    edges = spec["return_edges"]
    slots = spec["return_slots"]
    motor_sign = spec["motor_sign"]
    assert (
        isinstance(edges, Tensor) and isinstance(slots, Tensor) and isinstance(motor_sign, Tensor)
    )
    horizons = (0.10, 0.25, 0.50, 0.75, 1.00)
    horizon_steps = {round(value / hover_config.dt): value for value in horizons}
    features: list[Tensor] = []
    targets: list[Tensor] = []
    labels: list[str] = []
    for step in range(1, max(horizon_steps) + 1):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        _, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        with torch.no_grad():
            base_motor, base_neural = base_controller(
                image,
                state.euler[:, :2],
                base_neural,
                state.specific_force,
            )
            rc, stick_state = sticks(base_motor, stick_state)
            state = quad(rc, state, mass_scale)
        if step in horizon_steps:
            activity = torch.tanh(neural[:, controller.edge_pre[edges]])
            features.append(activity * controller.edge_sign[edges])
            bias = torch.zeros_like(desired_bias) if step == min(horizon_steps) else desired_bias
            targets.append(bias[:, None] * motor_sign)
            labels.extend([f"{horizon_steps[step]:g}"] * episodes)
    design = torch.cat(features).detach().cpu().double().numpy()
    target_current = torch.cat(targets).detach().cpu().double().numpy()
    slots_np = slots.detach().cpu().numpy()
    fitted = np.zeros(len(edges), dtype=np.float64)
    for slot in range(len(motor_sign)):
        selected = np.flatnonzero(slots_np == slot)
        fitted[selected], _ = nnls(design[:, selected], target_current[:, slot])
    with torch.no_grad():
        controller.edge_magnitude[edges] = torch.from_numpy(fitted).to(
            device=device,
            dtype=controller.edge_magnitude.dtype,
        )
    predicted = np.zeros_like(target_current)
    for edge_index, slot in enumerate(slots_np):
        predicted[:, slot] += design[:, edge_index] * fitted[edge_index]
    errors = predicted - target_current
    per_horizon: dict[str, Any] = {}
    labels_np = np.asarray(labels)
    for horizon in horizons:
        selected = labels_np == f"{horizon:g}"
        predicted_bias = (predicted[selected] * motor_sign.detach().cpu().numpy()).mean(axis=1)
        target_at_horizon = (target_current[selected] * motor_sign.detach().cpu().numpy()).mean(
            axis=1
        )
        per_horizon[f"{horizon:g}"] = {
            **regression_metrics(
                torch.from_numpy(predicted_bias),
                torch.from_numpy(target_at_horizon),
            ),
            "per_motor_current_rmse": float(np.sqrt(np.mean(errors[selected] ** 2))),
        }
    return {
        "edge_magnitudes": fitted.tolist(),
        "edge_body_pairs": [
            [
                int(controller.node_ids[controller.edge_pre[edge]].item()),
                int(controller.node_ids[controller.edge_post[edge]].item()),
            ]
            for edge in edges
        ],
        "overall_per_motor_current_rmse": float(np.sqrt(np.mean(errors**2))),
        "per_horizon": per_horizon,
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    source_checkpoint: dict[str, Any],
    args: argparse.Namespace,
    remap: dict[str, Any],
    update: int,
    kind: str,
) -> None:
    torch.save(
        {
            **copy.deepcopy(source_checkpoint),
            "controller": {
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            "graph_sha256": file_sha256(args.graph),
            "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
            "source_graph_sha256": file_sha256(args.source_graph),
            "graph_remap": remap,
            "mass_latent_distillation": {
                "label_mode": args.label_mode,
                "selected_update": update,
                "kind": kind,
            },
        },
        path,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source_checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=True)
    if source_checkpoint["graph_sha256"] != file_sha256(args.source_graph):
        raise SystemExit("source checkpoint and graph hashes do not match")
    if bool(source_checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    hover_config = HoverConfig(**source_checkpoint["hover_config"])
    gate_config = GateConfig(**source_checkpoint["gate_config"])
    resolution = int(source_checkpoint["image_resolution"])

    base_controller = ConnectomeController(
        args.source_graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source_checkpoint["retinal_receptive_field"]),
    ).to(device)
    base_controller.load_state_dict(source_checkpoint["controller"])
    base_controller.eval()
    for parameter in base_controller.parameters():
        parameter.requires_grad_(False)
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(source_checkpoint["retinal_receptive_field"]),
    ).to(device)
    remap = remap_anatomical_parameters(
        controller,
        source_checkpoint,
        args.source_graph,
        args.graph,
        freeze_existing=False,
        new_visual_hemifields=False,
        keep_roll_biases_frozen=True,
        new_pathways_roll_only=False,
        new_acceleration_pathways_only=False,
    )
    spec = circuit_spec(controller, args.source_graph)
    register_gradient_masks(controller, spec)
    initial = {
        "edge_magnitude": controller.edge_magnitude.detach().clone(),
        "bias": controller.bias.detach().clone(),
        "raw_time_constant": controller.raw_time_constant.detach().clone(),
    }
    optimizer = torch.optim.Adam(
        (
            {"params": [controller.edge_magnitude], "lr": args.edge_learning_rate},
            {"params": [controller.bias], "lr": args.bias_learning_rate},
            {"params": [controller.raw_time_constant], "lr": args.time_constant_learning_rate},
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_encoder_path = args.output_dir / "best-encoder.pt"
    candidate_path = args.output_dir / "candidate.pt"
    progress_path = args.output_dir / "progress.json"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    started = perf_counter()
    print(
        json.dumps(
            {"phase": "feasibility", **{k: v for k, v in spec.items() if not isinstance(v, Tensor)}}
        ),
        flush=True,
    )

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
            label_mode=args.label_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite encoder gradient")
        optimizer.step()
        controller.project_parameters()
        if update % args.selection_interval and update != args.updates:
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
            )
        key = (
            min(probe["0.25"]["r2"], probe["0.5"]["r2"]),
            probe["0.25"]["r2"],
            -0.5 * (probe["0.25"]["rmse"] + probe["0.5"]["rmse"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            save_checkpoint(
                best_encoder_path,
                controller,
                source_checkpoint,
                args,
                remap,
                update,
                "held_out_early_latent",
            )
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

    best = torch.load(best_encoder_path, map_location=device, weights_only=True)
    controller.load_state_dict(best["controller"])
    controller.eval()
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
    )
    save_checkpoint(
        candidate_path,
        controller,
        source_checkpoint,
        args,
        remap,
        best["mass_latent_distillation"]["selected_update"],
        "nnls_anatomical_returns",
    )
    print(json.dumps({"phase": "return_fit", **return_fit}), flush=True)

    final_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "mass_rank_swapped_acceleration": {"swapped_acceleration": True},
        "constant_initial_stick_position": {"frozen_proprioception": True},
        "mass_rank_swapped_stick_position": {"swapped_proprioception": True},
        "frozen_first_frame": {"frozen_visual": True},
    }
    final: dict[str, Any] = {}
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    for name, controls in final_specs.items():
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
            **controls,
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
    probe_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "pair_swapped_acceleration": {"swapped_acceleration": True},
        "constant_initial_stick_position": {"frozen_proprioception": True},
        "pair_swapped_stick_position": {"swapped_proprioception": True},
    }
    final_probes: dict[str, Any] = {}
    with torch.no_grad():
        for name, controls in probe_specs.items():
            _, final_probes[name] = encoder_rollout(
                controller,
                base_controller,
                spec,
                calibration,
                episodes=args.selection_episodes,
                seed=args.final_seed + 1,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                **controls,
            )
    live = final["live_sensors"]
    acceleration_controls = (
        final["constant_1g_acceleration"]["success_rate"],
        final["mass_rank_swapped_acceleration"]["success_rate"],
    )
    early_probe = final_probes["live_sensors"]["0.5"]
    encoding_passed = early_probe["r2"] >= 0.95
    return_fit_passed = (
        return_fit["per_horizon"]["0.5"]["per_motor_current_rmse"] <= 0.01
        and abs(return_fit["per_horizon"]["0.5"]["signed_slope"] - 1.0) <= 0.1
    )
    flight_causality = live["success_rate"] >= max(acceleration_controls) + 0.05
    goal_passed = live["goal_pass"] and encoding_passed and return_fit_passed and flight_causality
    trainable_node_mask = spec["trainable_node_mask"]
    trainable_edge_mask = spec["trainable_edge_mask"]
    assert isinstance(trainable_node_mask, Tensor) and isinstance(trainable_edge_mask, Tensor)
    drift = {
        "frozen_edge_magnitude_max_absolute_change": float(
            (
                controller.edge_magnitude[~trainable_edge_mask.bool()]
                - initial["edge_magnitude"][~trainable_edge_mask.bool()]
            )
            .abs()
            .max()
        ),
        "frozen_bias_max_absolute_change": float(
            (
                controller.bias[~trainable_node_mask.bool()]
                - initial["bias"][~trainable_node_mask.bool()]
            )
            .abs()
            .max()
        ),
        "frozen_time_constant_max_absolute_change": float(
            (
                controller.raw_time_constant[~trainable_node_mask.bool()]
                - initial["raw_time_constant"][~trainable_node_mask.bool()]
            )
            .abs()
            .max()
        ),
    }
    # Fitted return edges are the only intentional change outside the encoder mask.
    return_edges = spec["return_edges"]
    assert isinstance(return_edges, Tensor)
    frozen_without_returns = ~trainable_edge_mask.bool()
    frozen_without_returns[return_edges] = False
    drift["frozen_nonreturn_edge_magnitude_max_absolute_change"] = float(
        (
            controller.edge_magnitude[frozen_without_returns]
            - initial["edge_magnitude"][frozen_without_returns]
        )
        .abs()
        .max()
    )
    report = {
        "experiment": "two-stage early recurrent mass-code distillation",
        "claim_scope": (
            "Mass is a training label only. Deployment receives instantaneous FPV, roll/pitch, "
            "body-Z specific force, throttle-stick position, and recurrent connectome state."
        ),
        "counts_toward_direct_sensor_goal": bool(encoding_passed and flight_causality),
        "external_runtime_parameters_added": 0,
        "external_actor_state_machine": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_graph": stable_path(args.source_graph),
        "source_checkpoint": stable_path(args.source_checkpoint),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "best_encoder_checkpoint": stable_path(best_encoder_path),
        "candidate_checkpoint": stable_path(candidate_path),
        "candidate_checkpoint_sha256": file_sha256(candidate_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "graph_remap": remap,
        "circuit": {k: v for k, v in spec.items() if not isinstance(v, Tensor)},
        "training": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "matched_geometry_light_heavy_pairs": True,
            "oracle_corrected_training_physics": False,
        },
        "history": history,
        "selected_update": best["mass_latent_distillation"]["selected_update"],
        "return_fit": return_fit,
        "old_parameter_drift": drift,
        "baseline": baseline,
        "final": final,
        "final_encoder_probes": final_probes,
        "causal_checks": {
            "trained_with_true_mass_labels": args.label_mode == "true",
            "early_encoding_passed": encoding_passed,
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
                "goal_passed": goal_passed,
                "baseline_success_rate": baseline["success_rate"],
                "live_success_rate": live["success_rate"],
                "selected_update": report["selected_update"],
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
