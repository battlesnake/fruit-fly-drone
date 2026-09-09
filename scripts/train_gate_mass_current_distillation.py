#!/usr/bin/env python3
"""Distill the privileged mass trim into causal connectome return currents."""

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

from search_gate_acceleration_es import compact_metrics  # noqa: E402
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    initial_rollout,
    remap_anatomical_parameters,
    seed_everything,
)

from flydrone.gate import AnnularGate, GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
)


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-proprio-diagnostic-v1" / "connectome.npz"),
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
        default=REPO_ROOT / "runs" / "gate" / "mass-current-distillation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix-seconds", type=float, default=1.0)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--edge-learning-rate", type=float, default=0.025)
    parser.add_argument("--bias-learning-rate", type=float, default=0.006)
    parser.add_argument("--time-constant-learning-rate", type=float, default=0.003)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--selection-interval", type=int, default=20)
    parser.add_argument("--selection-episodes", type=int, default=128)
    parser.add_argument("--probe-episodes", type=int, default=512)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=200_031)
    parser.add_argument("--selection-seed", type=int, default=210_031)
    parser.add_argument("--final-seed", type=int, default=220_031)
    parser.add_argument(
        "--label-mode",
        choices=("true", "pair_swapped", "constant"),
        default="true",
        help="Training-only causal control for the mass label.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (args.graph, args.source_checkpoint, args.source_graph, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.updates,
        args.batch_size,
        args.prefix_seconds,
        args.target_onset_seconds,
        args.edge_learning_rate,
        args.bias_learning_rate,
        args.time_constant_learning_rate,
        args.pair_loss_weight,
        args.gradient_clip,
        args.selection_interval,
        args.selection_episodes,
        args.probe_episodes,
        args.final_episodes,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, times, rates, and weights must be positive")
    if args.batch_size % 2:
        raise SystemExit("--batch-size must be even for matched-mass pairs")
    for name in ("selection_episodes", "probe_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")
    if args.target_onset_seconds >= args.prefix_seconds:
        raise SystemExit("target onset must occur inside the training prefix")
    if round(args.target_onset_seconds / dt) < 1:
        raise SystemExit("target onset must leave at least one zero-current interval")


def paired_launch(
    batch: int,
    *,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[QuadState, AnnularGate, Tensor]:
    half = batch // 2
    state, gate, sampled_mass = initial_rollout(
        half,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    state = QuadState(*(torch.cat((value, value.clone())) for value in state.as_tuple()))
    gate = AnnularGate(
        center=torch.cat((gate.center, gate.center.clone())),
        yaw=torch.cat((gate.yaw, gate.yaw.clone())),
    )
    magnitude = (sampled_mass - 1.0).abs().clamp_min(1.0e-4)
    mass_scale = torch.cat((1.0 - magnitude, 1.0 + magnitude))
    return state, gate, mass_scale


def label_mass(normalized_mass: Tensor, mode: str) -> Tensor:
    if mode == "true":
        return normalized_mass
    if mode == "pair_swapped":
        half = len(normalized_mass) // 2
        return torch.cat((normalized_mass[half:], normalized_mass[:half]))
    if mode == "constant":
        return torch.zeros_like(normalized_mass)
    raise ValueError(f"unknown label mode: {mode}")


def circuit_spec(
    controller: ConnectomeController,
    source_graph: Path,
) -> dict[str, Tensor | int]:
    with np.load(source_graph) as source:
        source_ids = set(map(int, source["node_ids"]))
    node_ids = controller.node_ids.detach().cpu().numpy()
    new_node_mask_np = np.asarray([int(body) not in source_ids for body in node_ids])
    edge_pre_np = controller.edge_pre.detach().cpu().numpy()
    edge_post_np = controller.edge_post.detach().cpu().numpy()
    pool_offsets = controller.pool_offsets.detach().cpu().numpy()
    pool_indices = controller.pool_indices.detach().cpu().numpy()
    positive_nodes = pool_indices[pool_offsets[6] : pool_offsets[7]]
    negative_nodes = pool_indices[pool_offsets[7] : pool_offsets[8]]
    throttle_nodes = np.concatenate((positive_nodes, negative_nodes))
    return_edge_mask_np = new_node_mask_np[edge_pre_np] & np.isin(edge_post_np, throttle_nodes)
    return_edges_np = np.flatnonzero(return_edge_mask_np)
    motor_slot = {int(node): index for index, node in enumerate(throttle_nodes)}
    return_slots_np = np.asarray(
        [motor_slot[int(edge_post_np[edge])] for edge in return_edges_np], dtype=np.int64
    )
    reached = set(map(int, edge_post_np[return_edges_np]))
    if reached != set(map(int, throttle_nodes)):
        raise SystemExit("added circuit does not directly reach every throttle motor neuron")
    trainable_edge_mask_np = new_node_mask_np[edge_post_np] | return_edge_mask_np
    device = controller.bias.device
    dtype = controller.bias.dtype
    return {
        "new_node_mask": torch.from_numpy(new_node_mask_np).to(device=device, dtype=dtype),
        "trainable_edge_mask": torch.from_numpy(trainable_edge_mask_np).to(
            device=device, dtype=dtype
        ),
        "return_edges": torch.from_numpy(return_edges_np).to(device=device),
        "return_slots": torch.from_numpy(return_slots_np).to(device=device),
        "motor_sign": torch.cat(
            (
                torch.ones(len(positive_nodes), device=device, dtype=dtype),
                -torch.ones(len(negative_nodes), device=device, dtype=dtype),
            )
        ),
        "positive_motor_count": len(positive_nodes),
        "new_node_count": int(new_node_mask_np.sum()),
        "trainable_edge_count": int(trainable_edge_mask_np.sum()),
        "return_edge_count": int(return_edge_mask_np.sum()),
    }


def register_circuit_gradient_masks(
    controller: ConnectomeController,
    spec: dict[str, Tensor | int],
) -> None:
    node_mask = spec["new_node_mask"]
    edge_mask = spec["trainable_edge_mask"]
    assert isinstance(node_mask, Tensor) and isinstance(edge_mask, Tensor)
    controller.bias.register_hook(lambda gradient: gradient * node_mask)
    controller.raw_time_constant.register_hook(lambda gradient: gradient * node_mask)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * edge_mask)


def return_current(
    controller: ConnectomeController,
    neural: Tensor,
    spec: dict[str, Tensor | int],
) -> Tensor:
    edges = spec["return_edges"]
    slots = spec["return_slots"]
    motor_sign = spec["motor_sign"]
    assert (
        isinstance(edges, Tensor) and isinstance(slots, Tensor) and isinstance(motor_sign, Tensor)
    )
    messages = (
        torch.tanh(neural)[:, controller.edge_pre[edges]]
        * controller.edge_sign[edges]
        * controller.edge_magnitude[edges]
    )
    current = torch.zeros(
        neural.shape[0], len(motor_sign), device=neural.device, dtype=neural.dtype
    )
    return current.index_add(1, slots, messages)


def estimated_bias(current: Tensor, spec: dict[str, Tensor | int]) -> Tensor:
    motor_sign = spec["motor_sign"]
    assert isinstance(motor_sign, Tensor)
    return (current * motor_sign).mean(dim=1)


def target_bias(normalized_mass: Tensor, calibration: dict[str, Any]) -> Tensor:
    return calibration["intercept"] + calibration["mass_slope"] * normalized_mass


def current_loss(
    current: Tensor,
    target: Tensor,
    spec: dict[str, Tensor | int],
    pair_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    motor_sign = spec["motor_sign"]
    positive_count = spec["positive_motor_count"]
    assert isinstance(motor_sign, Tensor) and isinstance(positive_count, int)
    target_current = target[:, None] * motor_sign
    positive = (current[:, :positive_count] - target_current[:, :positive_count]).square().mean()
    negative = (current[:, positive_count:] - target_current[:, positive_count:]).square().mean()
    direct = 0.5 * (positive + negative)
    prediction = estimated_bias(current, spec)
    half = len(prediction) // 2
    contrastive = (
        ((prediction[:half] - prediction[half:]) - (target[:half] - target[half:])).square().mean()
    )
    return direct + pair_weight * contrastive, direct, contrastive


def regression_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    prediction = prediction.detach()
    target = target.detach()
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    target_variance = centered_target.square().mean().clamp_min(1.0e-12)
    covariance = (centered_prediction * centered_target).mean()
    correlation = covariance / (
        centered_prediction.square().mean().sqrt() * target_variance.sqrt()
    ).clamp_min(1.0e-12)
    return {
        "rmse": float((prediction - target).square().mean().sqrt()),
        "signed_slope": float(covariance / target_variance),
        "pearson_correlation": float(correlation),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "prediction_light_mean": float(prediction[target < target.mean()].mean()),
        "prediction_heavy_mean": float(prediction[target >= target.mean()].mean()),
    }


def state_observability(
    neural: Tensor,
    normalized_mass: Tensor,
    spec: dict[str, Tensor | int],
) -> dict[str, float]:
    """Training-only linear audit of mass information in the nine added states."""

    new_node_mask = spec["new_node_mask"]
    assert isinstance(new_node_mask, Tensor)
    features = neural[:, new_node_mask.bool()]
    features = (features - features.mean(dim=0)) / features.std(dim=0).clamp_min(1.0e-5)
    design = torch.cat((features, torch.ones_like(features[:, :1])), dim=1)
    gram = design.T @ design / len(design)
    penalty = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * 1.0e-3
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(
        gram + penalty,
        design.T @ normalized_mass / len(design),
    )
    prediction = design @ weights
    residual = prediction - normalized_mass
    variance = (normalized_mass - normalized_mass.mean()).square().mean().clamp_min(1.0e-12)
    centered_features = neural[:, new_node_mask.bool()] - neural[:, new_node_mask.bool()].mean(
        dim=0
    )
    centered_mass = normalized_mass - normalized_mass.mean()
    correlations = (centered_features * centered_mass[:, None]).mean(dim=0) / (
        centered_features.square().mean(dim=0).sqrt() * centered_mass.square().mean().sqrt()
    ).clamp_min(1.0e-12)
    return {
        "new_state_linear_readout_r2": float(1.0 - residual.square().mean() / variance),
        "new_state_max_absolute_neuron_mass_correlation": float(correlations.abs().max()),
    }


def training_update(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Tensor | int],
    calibration: dict[str, Any],
    *,
    batch: int,
    steps: int,
    onset_step: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    label_mode: str,
    pair_loss_weight: float,
    gradient_clip: float,
) -> dict[str, float]:
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        batch,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    base_neural = base_controller.initial_state(batch, device=device, dtype=torch.float32)
    student_neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(batch, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    training_mass = label_mass(normalized_mass, label_mode)
    desired_bias = target_bias(training_mass, calibration)
    losses = []
    direct_losses = []
    contrastive_losses = []
    final_prediction = None
    for step in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        current = return_current(controller, student_neural, spec)
        step_target = desired_bias if step >= onset_step else torch.zeros_like(desired_bias)
        loss, direct, contrastive = current_loss(
            current,
            step_target,
            spec,
            pair_loss_weight,
        )
        losses.append(loss)
        direct_losses.append(direct.detach())
        contrastive_losses.append(contrastive.detach())
        final_prediction = estimated_bias(current, spec)
        _, student_neural = controller(
            image,
            state.euler[:, :2],
            student_neural,
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
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("non-finite distillation gradient")
    optimizer.step()
    controller.project_parameters()
    assert final_prediction is not None
    return {
        "loss": float(loss.detach()),
        "direct_current_mse": float(torch.stack(direct_losses).mean()),
        "paired_current_mse": float(torch.stack(contrastive_losses).mean()),
        "gradient_norm": float(gradient_norm),
        **regression_metrics(final_prediction, desired_bias),
    }


@torch.no_grad()
def current_probe(
    controller: ConnectomeController,
    base_controller: ConnectomeController,
    spec: dict[str, Tensor | int],
    calibration: dict[str, Any],
    *,
    episodes: int,
    horizons: tuple[float, ...],
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    **sensor_controls: bool,
) -> dict[str, dict[str, float]]:
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
    student_neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    desired = target_bias(normalized_mass, calibration)
    horizon_steps = {round(horizon / hover_config.dt): horizon for horizon in horizons}
    maximum_step = max(horizon_steps)
    results: dict[str, dict[str, float]] = {}
    acceleration_permutation = torch.arange(episodes, device=device)
    half = episodes // 2
    acceleration_permutation = torch.cat(
        (acceleration_permutation[half:], acceleration_permutation[:half])
    )
    for step in range(maximum_step + 1):
        if step in horizon_steps:
            prediction = estimated_bias(return_current(controller, student_neural, spec), spec)
            results[f"{horizon_steps[step]:g}"] = {
                **regression_metrics(prediction, desired),
                **state_observability(student_neural, normalized_mass, spec),
            }
        if step == maximum_step:
            break
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        specific_force = state.specific_force
        if sensor_controls.get("frozen_acceleration", False):
            specific_force = torch.zeros_like(specific_force)
            specific_force[:, 2] = 9.81
        elif sensor_controls.get("swapped_acceleration", False):
            specific_force = specific_force[acceleration_permutation]
        sensed_position = stick_state.position
        if sensor_controls.get("frozen_proprioception", False):
            sensed_position = torch.zeros_like(sensed_position)
            sensed_position[:, 3] = -1.0
        elif sensor_controls.get("swapped_proprioception", False):
            sensed_position = sensed_position[acceleration_permutation]
        _, student_neural = controller(
            image,
            state.euler[:, :2],
            student_neural,
            specific_force,
            sensed_position,
        )
        base_motor, base_neural = base_controller(
            image,
            state.euler[:, :2],
            base_neural,
            state.specific_force,
        )
        rc, stick_state = sticks(base_motor, stick_state)
        state = quad(rc, state, mass_scale)
    return results


def old_parameter_drift(
    controller: ConnectomeController,
    initial: dict[str, Tensor],
    spec: dict[str, Tensor | int],
) -> dict[str, float]:
    new_node_mask = spec["new_node_mask"]
    trainable_edge_mask = spec["trainable_edge_mask"]
    assert isinstance(new_node_mask, Tensor) and isinstance(trainable_edge_mask, Tensor)
    old_nodes = ~new_node_mask.bool()
    frozen_edges = ~trainable_edge_mask.bool()
    return {
        "frozen_edge_magnitude_max_absolute_change": float(
            (controller.edge_magnitude[frozen_edges] - initial["edge_magnitude"][frozen_edges])
            .abs()
            .max()
            .detach()
        ),
        "old_bias_max_absolute_change": float(
            (controller.bias[old_nodes] - initial["bias"][old_nodes]).abs().max().detach()
        ),
        "old_raw_time_constant_max_absolute_change": float(
            (controller.raw_time_constant[old_nodes] - initial["raw_time_constant"][old_nodes])
            .abs()
            .max()
            .detach()
        ),
    }


def main() -> int:
    args = parse_args()
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
        raise SystemExit("calibration is not the expected privileged mass oracle")
    hover_config = HoverConfig(**source_checkpoint["hover_config"])
    gate_config = GateConfig(**source_checkpoint["gate_config"])
    validate_args(args, hover_config.dt)
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
        freeze_existing=True,
        new_visual_hemifields=False,
        keep_roll_biases_frozen=True,
        new_pathways_roll_only=False,
        new_acceleration_pathways_only=False,
    )
    spec = circuit_spec(controller, args.source_graph)
    register_circuit_gradient_masks(controller, spec)
    initial = {
        name: value.detach().clone()
        for name, value in {
            "edge_magnitude": controller.edge_magnitude,
            "bias": controller.bias,
            "raw_time_constant": controller.raw_time_constant,
        }.items()
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
    steps = round(args.prefix_seconds / hover_config.dt)
    onset_step = round(args.target_onset_seconds / hover_config.dt)
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    progress_path = args.output_dir / "progress.json"
    started = perf_counter()
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None

    for update in range(args.updates):
        seed_everything(args.seed + update)
        controller.train()
        training = training_update(
            controller,
            base_controller,
            optimizer,
            spec,
            calibration,
            batch=args.batch_size,
            steps=steps,
            onset_step=onset_step,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            label_mode=args.label_mode,
            pair_loss_weight=args.pair_loss_weight,
            gradient_clip=args.gradient_clip,
        )
        should_select = (update + 1) % args.selection_interval == 0 or update + 1 == args.updates
        if not should_select:
            if update == 0 or (update + 1) % 5 == 0:
                print(
                    json.dumps({"phase": "training", "update": update + 1, **training}), flush=True
                )
            continue

        controller.eval()
        selection = evaluate_gate(
            controller,
            episodes=args.selection_episodes,
            seconds=12.0,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.selection_seed,
            balanced_strata=True,
        )
        probe = current_probe(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.probe_episodes,
            horizons=(0.10, 0.25, 0.50, 0.75, 1.00),
            seed=args.selection_seed + 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        entry = {
            "update": update + 1,
            "training": training,
            "selection": compact_metrics(selection, clean_radius),
            "current_probe": probe,
        }
        history.append(entry)
        key = (
            selection["success_rate"],
            min(
                selection["success_by_stratum"]["lower_mass"],
                selection["success_by_stratum"]["higher_mass"],
            ),
            -probe["0.5"]["rmse"],
        )
        if best_key is None or key > best_key:
            best_key = key
            torch.save(
                {
                    **copy.deepcopy(source_checkpoint),
                    "controller": {
                        name: value.detach().cpu()
                        for name, value in controller.state_dict().items()
                    },
                    "graph_sha256": file_sha256(args.graph),
                    "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
                    "source_graph_sha256": file_sha256(args.source_graph),
                    "graph_remap": remap,
                    "mass_current_distillation": {
                        "label_mode": args.label_mode,
                        "selected_update": update + 1,
                        "target_onset_seconds": args.target_onset_seconds,
                    },
                },
                best_path,
            )
        print(json.dumps({"phase": "selection", **entry, "best_key": best_key}), flush=True)
        progress_path.write_text(
            json.dumps(
                {
                    "history": history,
                    "best_key": best_key,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    best = torch.load(best_path, map_location=device, weights_only=True)
    controller.load_state_dict(best["controller"])
    controller.eval()
    final_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "mass_rank_swapped_acceleration": {"swapped_acceleration": True},
        "constant_initial_stick_position": {"frozen_proprioception": True},
        "mass_rank_swapped_stick_position": {"swapped_proprioception": True},
        "frozen_first_frame": {"frozen_visual": True},
    }
    final: dict[str, Any] = {}
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
    final_probe_specs = {
        "live_sensors": {},
        "constant_1g_acceleration": {"frozen_acceleration": True},
        "pair_swapped_acceleration": {"swapped_acceleration": True},
        "constant_initial_stick_position": {"frozen_proprioception": True},
        "pair_swapped_stick_position": {"swapped_proprioception": True},
    }
    final_probes = {
        name: current_probe(
            controller,
            base_controller,
            spec,
            calibration,
            episodes=args.probe_episodes,
            horizons=(0.10, 0.25, 0.50, 0.75, 1.00),
            seed=args.final_seed + 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            **controls,
        )
        for name, controls in final_probe_specs.items()
    }
    drift = old_parameter_drift(controller, initial, spec)
    live_probe = final_probes["live_sensors"]["0.5"]
    acceleration_controls = (
        final_probes["constant_1g_acceleration"]["0.5"],
        final_probes["pair_swapped_acceleration"]["0.5"],
    )
    current_sensor_causality = (
        live_probe["pearson_correlation"] >= 0.8
        and live_probe["signed_slope"] >= 0.5
        and max(control["pearson_correlation"] for control in acceleration_controls)
        <= live_probe["pearson_correlation"] - 0.2
    )
    flight_sensor_causality = (
        final["live_sensors"]["success_rate"]
        >= max(
            final["constant_1g_acceleration"]["success_rate"],
            final["mass_rank_swapped_acceleration"]["success_rate"],
        )
        + 0.05
    )
    sensor_causality_demonstrated = (
        args.label_mode == "true" and current_sensor_causality and flight_sensor_causality
    )
    goal_passed = final["live_sensors"]["goal_pass"] and sensor_causality_demonstrated
    report = {
        "experiment": "causal mass-current distillation inside added connectome circuit",
        "claim_scope": (
            "Mass is a training label only. Deployment receives FPV, roll/pitch, body-Z "
            "acceleration, throttle-stick position, and recurrent connectome state."
        ),
        "counts_toward_direct_sensor_goal": sensor_causality_demonstrated,
        "external_runtime_parameters_added": 0,
        "external_actor_state_machine": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_graph": stable_path(args.source_graph),
        "source_checkpoint": stable_path(args.source_checkpoint),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "candidate_checkpoint": stable_path(best_path),
        "candidate_checkpoint_sha256": file_sha256(best_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "graph_remap": remap,
        "circuit": {
            "new_nodes": spec["new_node_count"],
            "trainable_edges": spec["trainable_edge_count"],
            "direct_throttle_return_edges": spec["return_edge_count"],
            "old_parameters_frozen": True,
            "mass_label_available_at_runtime": False,
            "training_physics_received_oracle_correction": False,
        },
        "training": {
            "updates": args.updates,
            "batch_size": args.batch_size,
            "matched_geometry_light_heavy_pairs": True,
            "prefix_seconds": args.prefix_seconds,
            "target_onset_seconds": args.target_onset_seconds,
            "label_mode": args.label_mode,
            "edge_learning_rate": args.edge_learning_rate,
            "bias_learning_rate": args.bias_learning_rate,
            "time_constant_learning_rate": args.time_constant_learning_rate,
            "pair_loss_weight": args.pair_loss_weight,
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
            "selection_seed": args.selection_seed,
            "final_seed": args.final_seed,
        },
        "history": history,
        "selected_update": best["mass_current_distillation"]["selected_update"],
        "old_parameter_drift": drift,
        "baseline": baseline,
        "final": final,
        "final_current_probes": final_probes,
        "causal_checks": {
            "trained_with_true_mass_labels": args.label_mode == "true",
            "current_sensor_causality": current_sensor_causality,
            "flight_sensor_causality": flight_sensor_causality,
            "sensor_causality_demonstrated": sensor_causality_demonstrated,
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
                "candidate_checkpoint": stable_path(best_path),
                "selected_update": report["selected_update"],
                "goal_passed": report["goal_passed"],
                "old_parameter_drift": drift,
                "baseline_success_rate": baseline["success_rate"],
                "live_success_rate": final["live_sensors"]["success_rate"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
