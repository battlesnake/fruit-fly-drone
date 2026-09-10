#!/usr/bin/env python3
"""Audit native representations needed by a control-theory hover decomposition.

The assays are synthetic, controlled sensory replays.  They do not constitute closed-loop
flight and their privileged labels are used only by offline probes and reports.  The deployed
controller still receives only RGB pixels, roll/pitch, and its own MaleCNS recurrent state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
    teacher_rc,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    VisualScene,
    render_visual_hover_scene,
    sample_visual_scenes,
)

ASSAY_NAMES = ("height_error", "vertical_motion", "command_history")
RIDGE_STRENGTHS = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
REPRESENTATION_R2 = 0.50
REPRESENTATION_SIGN_ACCURACY = 0.90
USABLE_RESPONSE_FRACTION = 0.10
CAUSAL_EFFECT_FRACTION = 0.30
MAX_COMMON_MOTOR_CHANGE = 0.0025
REPLAY_NOISE_FLOOR = 1.0e-6


@dataclass
class AssayBatch:
    state_a: Tensor
    state_b: Tensor
    image_a: Tensor
    image_b: Tensor
    attitude: Tensor
    target_contrast: Tensor
    physical_contrast: Tensor
    native_motor_contrast: Tensor
    diagnostics: dict[str, Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/responsibility-audit-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--development-pairs", type=int, default=64)
    parser.add_argument("--test-pairs", type=int, default=128)
    parser.add_argument("--causal-pairs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prefix-steps", type=int, default=25)
    parser.add_argument("--history-steps", type=int, default=25)
    parser.add_argument("--neutral-steps", type=int, default=13)
    parser.add_argument("--causal-response-steps", type=int, default=10)
    parser.add_argument("--probe-features", type=int, default=26)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=240_911)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.raw_dir / ANNOTATIONS_FILE, args.checkpoint):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.policy_hz,
        args.development_pairs,
        args.test_pairs,
        args.causal_pairs,
        args.batch_size,
        args.prefix_steps,
        args.history_steps,
        args.neutral_steps,
        args.causal_response_steps,
        args.probe_features,
        args.bootstrap_samples,
    )
    if min(positive) < 1:
        raise SystemExit("all counts and frequencies must be positive")
    if args.development_pairs < 4:
        raise SystemExit("at least four development pairs are required")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def clone_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def duplicate_scene(scene: VisualScene) -> VisualScene:
    return VisualScene(*(torch.cat((value, value)) for value in scene.__dict__.values()))


def settled_sticks(batch: int, *, device: torch.device, config: HoverConfig) -> StickState:
    rc = torch.zeros(batch, 4, device=device)
    rc[:, 3] = 1.0 / config.thrust_to_weight
    normalized = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    sine_limit = math.sin(config.foreleg_joint_limit)
    joint = torch.asin((normalized * sine_limit).clamp(-1.0, 1.0))
    zeros = torch.zeros_like(normalized)
    return StickState(joint, zeros.clone(), normalized, zeros)


def balanced_signed_values(
    count: int, magnitudes: tuple[float, ...], *, device: torch.device
) -> Tensor:
    values = torch.tensor(
        [sign * magnitude for magnitude in magnitudes for sign in (-1.0, 1.0)],
        device=device,
    )
    repeated = values.repeat(math.ceil(count / len(values)))[:count]
    return repeated[torch.randperm(count, device=device)]


def base_state(
    batch: int, *, device: torch.device, config: HoverConfig
) -> tuple[QuadState, Tensor]:
    quad = DifferentiableQuad(config).to(device)
    height = torch.empty(batch, device=device).uniform_(0.78, 1.22)
    position = torch.empty(batch, 3, device=device).uniform_(-0.10, 0.10)
    position[:, 2] = height
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(2.0), math.radians(2.0)
    )
    euler[:, 2] = torch.empty(batch, device=device).uniform_(-math.radians(4.0), math.radians(4.0))
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    state.actuator[:, 0] = 1.0 / config.thrust_to_weight
    return state, height


def state_at_height(
    state: QuadState, height: Tensor, *, vertical_velocity: Tensor | None = None
) -> QuadState:
    result = clone_quad(state)
    result.position[:, 2] = height
    if vertical_velocity is not None:
        result.velocity[:, 2] = vertical_velocity
    return result


@torch.no_grad()
def common_prefix(
    controller: ConnectomeController,
    state: QuadState,
    marker: Tensor,
    scene: VisualScene,
    *,
    steps: int,
) -> Tensor:
    neural = controller.initial_state(
        state.position.shape[0], device=state.position.device, dtype=state.position.dtype
    )
    image = render_visual_hover_scene(state, marker, scene=scene)
    for _ in range(steps):
        _, neural = controller(image, state.euler[:, :2], neural)
    return neural


def teacher_throttle_contrast(
    state_a: QuadState,
    state_b: QuadState,
    marker_a: Tensor,
    marker_b: Tensor,
    config: HoverConfig,
) -> Tensor:
    motor_a = motor_target_for_rc(teacher_rc(state_a, marker_a, config), config)
    motor_b = motor_target_for_rc(teacher_rc(state_b, marker_b, config), config)
    return motor_a[:, 3] - motor_b[:, 3]


@torch.no_grad()
def make_height_error_batch(
    controller: ConnectomeController,
    signed_error: Tensor,
    *,
    held_out_scene: bool,
    prefix_steps: int,
    history_steps: int,
    config: HoverConfig,
) -> AssayBatch:
    batch = len(signed_error)
    device = signed_error.device
    state, height = base_state(batch, device=device, config=config)
    scene = sample_visual_scenes(batch, device=device, held_out_combinations=held_out_scene)
    neural = common_prefix(controller, state, height, scene, steps=prefix_steps)
    marker_a = height + signed_error
    marker_b = height - signed_error
    image_a = render_visual_hover_scene(state, marker_a, scene=scene)
    image_b = render_visual_hover_scene(state, marker_b, scene=scene)
    recurrent = torch.cat((neural.clone(), neural.clone()))
    images = torch.cat((image_a, image_b))
    attitude = state.euler[:, :2]
    motor = torch.zeros(2 * batch, 4, device=device)
    for _ in range(history_steps):
        motor, recurrent = controller(images, torch.cat((attitude, attitude)), recurrent)
    state_a, state_b = recurrent[:batch], recurrent[batch:]
    target = teacher_throttle_contrast(state, state, marker_a, marker_b, config)
    return AssayBatch(
        state_a=state_a,
        state_b=state_b,
        image_a=image_a,
        image_b=image_b,
        attitude=attitude,
        target_contrast=target,
        physical_contrast=2.0 * signed_error,
        native_motor_contrast=motor[:batch] - motor[batch:],
        diagnostics={},
    )


@torch.no_grad()
def make_vertical_motion_batch(
    controller: ConnectomeController,
    signed_speed: Tensor,
    final_marker_error: Tensor,
    *,
    held_out_scene: bool,
    prefix_steps: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> AssayBatch:
    batch = len(signed_speed)
    device = signed_speed.device
    state, height = base_state(batch, device=device, config=config)
    scene = sample_visual_scenes(batch, device=device, held_out_combinations=held_out_scene)
    marker = height + final_marker_error
    neural = common_prefix(controller, state, marker, scene, steps=prefix_steps)
    recurrent = torch.cat((neural.clone(), neural.clone()))
    attitude = state.euler[:, :2]
    duration = history_steps / policy_hz
    motor = torch.zeros(2 * batch, 4, device=device)
    image_a = image_b = render_visual_hover_scene(state, marker, scene=scene)
    for step in range(history_steps):
        unit = (step + 1) / history_steps
        # Both branches leave and return to the same pose.  Their endpoint derivatives
        # are +/- signed_speed; the zero-derivative term makes the visual excursion large
        # enough to resolve without changing the endpoint velocity.
        base = signed_speed * duration * (unit**3 - unit**2)
        excursion = -signed_speed.sign() * 0.055 * math.sin(math.pi * unit) ** 2
        offset = base + excursion
        if step + 1 == history_steps:
            offset = torch.zeros_like(offset)
        state_a = state_at_height(state, height + offset)
        state_b = state_at_height(state, height - offset)
        image_a = render_visual_hover_scene(state_a, marker, scene=scene)
        image_b = render_visual_hover_scene(state_b, marker, scene=scene)
        motor, recurrent = controller(
            torch.cat((image_a, image_b)), torch.cat((attitude, attitude)), recurrent
        )
    neural_a, neural_b = recurrent[:batch], recurrent[batch:]
    endpoint_a = state_at_height(state, height, vertical_velocity=signed_speed)
    endpoint_b = state_at_height(state, height, vertical_velocity=-signed_speed)
    target = teacher_throttle_contrast(endpoint_a, endpoint_b, marker, marker, config)
    return AssayBatch(
        state_a=neural_a,
        state_b=neural_b,
        image_a=image_a,
        image_b=image_b,
        attitude=attitude,
        target_contrast=target,
        physical_contrast=2.0 * signed_speed,
        native_motor_contrast=motor[:batch] - motor[batch:],
        diagnostics={"final_marker_error": final_marker_error},
    )


@torch.no_grad()
def make_command_history_batch(
    controller: ConnectomeController,
    signed_error: Tensor,
    *,
    held_out_scene: bool,
    prefix_steps: int,
    history_steps: int,
    neutral_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> AssayBatch:
    batch = len(signed_error)
    device = signed_error.device
    state, height = base_state(batch, device=device, config=config)
    scene = sample_visual_scenes(batch, device=device, held_out_combinations=held_out_scene)
    neural = common_prefix(controller, state, height, scene, steps=prefix_steps)
    recurrent = torch.cat((neural.clone(), neural.clone()))
    attitude = state.euler[:, :2]
    marker_a, marker_b = height + signed_error, height - signed_error
    pulse_a = render_visual_hover_scene(state, marker_a, scene=scene)
    pulse_b = render_visual_hover_scene(state, marker_b, scene=scene)
    pulse_images = torch.cat((pulse_a, pulse_b))
    duplicated_attitude = torch.cat((attitude, attitude))

    sticks = ForelegStickPlant(config).to(device)
    stick_state = settled_sticks(2 * batch, device=device, config=config)
    quad = DifferentiableQuad(config).to(device)
    shadow = QuadState(*(torch.cat((value, value)) for value in state.as_tuple()))
    physics_steps = round(100 / policy_hz)
    if physics_steps * policy_hz != 100:
        raise ValueError("policy frequency must divide the 100 Hz plant frequency")
    pulse_motor: list[Tensor] = []
    rc = torch.zeros(2 * batch, 4, device=device)
    for _ in range(history_steps):
        motor, recurrent = controller(pulse_images, duplicated_attitude, recurrent)
        pulse_motor.append(motor)
        for _ in range(physics_steps):
            rc, stick_state = sticks(motor, stick_state)
            shadow = quad(rc, shadow)
    pulse_end_stick = stick_state.position[:, 3].clone()
    pulse_end_actuator = shadow.actuator[:, 0].clone()

    neutral = render_visual_hover_scene(state, height, scene=scene)
    neutral_images = torch.cat((neutral, neutral))
    motor = pulse_motor[-1]
    for _ in range(neutral_steps):
        motor, recurrent = controller(neutral_images, duplicated_attitude, recurrent)
        for _ in range(physics_steps):
            rc, stick_state = sticks(motor, stick_state)
            shadow = quad(rc, shadow)
    tail = torch.stack(pulse_motor[-min(5, len(pulse_motor)) :])
    prior_native = tail[:, :batch, 3].mean(dim=0) - tail[:, batch:, 3].mean(dim=0)
    neural_a, neural_b = recurrent[:batch], recurrent[batch:]
    return AssayBatch(
        state_a=neural_a,
        state_b=neural_b,
        image_a=neutral,
        image_b=neutral.clone(),
        attitude=attitude,
        target_contrast=prior_native,
        physical_contrast=2.0 * signed_error,
        native_motor_contrast=motor[:batch] - motor[batch:],
        diagnostics={
            "pulse_end_stick_throttle_contrast": (
                pulse_end_stick[:batch] - pulse_end_stick[batch:]
            ),
            "pulse_end_actuator_contrast": (
                pulse_end_actuator[:batch] - pulse_end_actuator[batch:]
            ),
            "neutral_end_stick_throttle_contrast": (
                stick_state.position[:batch, 3] - stick_state.position[batch:, 3]
            ),
            "neutral_end_actuator_contrast": (
                shadow.actuator[:batch, 0] - shadow.actuator[batch:, 0]
            ),
        },
    )


def make_assay_batch(
    name: str,
    controller: ConnectomeController,
    primary_value: Tensor,
    secondary_value: Tensor,
    *,
    held_out_scene: bool,
    args: argparse.Namespace,
    config: HoverConfig,
) -> AssayBatch:
    if name == "height_error":
        return make_height_error_batch(
            controller,
            primary_value,
            held_out_scene=held_out_scene,
            prefix_steps=args.prefix_steps,
            history_steps=args.history_steps,
            config=config,
        )
    if name == "vertical_motion":
        return make_vertical_motion_batch(
            controller,
            primary_value,
            secondary_value,
            held_out_scene=held_out_scene,
            prefix_steps=args.prefix_steps,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
    if name == "command_history":
        return make_command_history_batch(
            controller,
            primary_value,
            held_out_scene=held_out_scene,
            prefix_steps=args.prefix_steps,
            history_steps=args.history_steps,
            neutral_steps=args.neutral_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
    raise ValueError(f"unknown assay: {name}")


def assay_design(name: str, count: int, *, device: torch.device) -> tuple[Tensor, Tensor]:
    if name == "vertical_motion":
        primary = balanced_signed_values(count, (0.15, 0.30), device=device)
        secondary = balanced_signed_values(count, (0.03, 0.06), device=device)
    else:
        primary = balanced_signed_values(count, (0.05, 0.10), device=device)
        secondary = torch.zeros(count, device=device)
    return primary, secondary


@torch.no_grad()
def collect_probe_bank(
    name: str,
    controller: ConnectomeController,
    *,
    pairs: int,
    held_out_scene: bool,
    seed: int,
    args: argparse.Namespace,
    config: HoverConfig,
) -> dict[str, Tensor]:
    seed_everything(seed)
    device = controller.edge_magnitude.device
    primary, secondary = assay_design(name, pairs, device=device)
    state_contrasts: list[Tensor] = []
    retina_contrasts: list[Tensor] = []
    targets: list[Tensor] = []
    physical: list[Tensor] = []
    native: list[Tensor] = []
    diagnostics: dict[str, list[Tensor]] = {}
    for begin in range(0, pairs, args.batch_size):
        end = min(begin + args.batch_size, pairs)
        batch = make_assay_batch(
            name,
            controller,
            primary[begin:end],
            secondary[begin:end],
            held_out_scene=held_out_scene,
            args=args,
            config=config,
        )
        state_contrasts.append((batch.state_a - batch.state_b).cpu())
        retina_contrasts.append(
            (
                controller.sample_retina(batch.image_a) - controller.sample_retina(batch.image_b)
            ).cpu()
        )
        targets.append(batch.target_contrast.cpu())
        physical.append(batch.physical_contrast.cpu())
        native.append(batch.native_motor_contrast.cpu())
        for key, value in batch.diagnostics.items():
            diagnostics.setdefault(key, []).append(value.cpu())
    result = {
        "state": torch.cat(state_contrasts),
        "retina": torch.cat(retina_contrasts),
        "target": torch.cat(targets),
        "physical": torch.cat(physical),
        "native_motor": torch.cat(native),
    }
    result.update({key: torch.cat(value) for key, value in diagnostics.items()})
    return result


def annotation_partitions(
    graph_path: Path, raw_dir: Path
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    graph = np.load(graph_path)
    node_ids = graph["node_ids"]
    annotations = _read_annotations(raw_dir / ANNOTATIONS_FILE)
    rows = {int(body): index for index, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray([str(annotations["superclass"][rows[int(body)]]) for body in node_ids])
    output = np.zeros(len(node_ids), dtype=bool)
    offsets = graph["output_pool_offsets"]
    pool_indices = graph["output_pool_indices"]
    output[pool_indices[offsets[0] : offsets[-1]]] = True
    unassigned = ~output

    def take(condition: np.ndarray) -> np.ndarray:
        nonlocal unassigned
        selected = condition & unassigned
        unassigned &= ~selected
        return selected

    text = superclass.astype(str)
    partitions = {
        "optic_lobe_and_visual": take(
            np.char.startswith(text, "ol_") | np.char.startswith(text, "visual_")
        ),
        "central_brain": take(np.char.startswith(text, "cb_")),
        "descending": take(np.char.find(text, "descending") >= 0),
        "ascending": take(np.char.find(text, "ascending") >= 0),
        "vnc_intrinsic": take(text == "vnc_intrinsic"),
        "vnc_sensory": take(np.char.startswith(text, "vnc_sensory")),
        "other_internal": unassigned.copy(),
        "output_motor_26": output,
    }
    covered = np.stack(list(partitions.values())).sum(axis=0)
    if not np.all(covered == 1):
        raise RuntimeError("anatomical partitions do not form an exact cover")
    return partitions, node_ids


def throttle_reachability(
    graph_path: Path, partitions: dict[str, np.ndarray], maximum_hops: int = 10
) -> dict[str, Any]:
    graph = np.load(graph_path)
    edge_pre, edge_post = graph["edge_pre"], graph["edge_post"]
    offsets, pools = graph["output_pool_offsets"], graph["output_pool_indices"]
    throttle = pools[offsets[6] : offsets[8]]
    distance = np.full(len(graph["node_ids"]), -1, dtype=np.int16)
    distance[throttle] = 0
    for hop in range(1, maximum_hops + 1):
        reached_edge = distance[edge_post] >= 0
        candidates = np.unique(edge_pre[reached_edge & (distance[edge_pre] < 0)])
        if not len(candidates):
            break
        distance[candidates] = hop
    result = {}
    for name, mask in partitions.items():
        values = distance[mask]
        result[name] = {
            "nodes": int(mask.sum()),
            "reaches_throttle_within_10_hops": int((values >= 0).sum()),
            "fraction_reaching_throttle_within_10_hops": float((values >= 0).mean()),
            "distance_counts": {
                str(hop): int((values == hop).sum()) for hop in range(maximum_hops + 1)
            },
        }
    return result


def select_features(features: Tensor, target: Tensor, maximum: int) -> Tensor:
    centered_x = features - features.mean(dim=0)
    centered_y = target - target.mean()
    numerator = (centered_x * centered_y[:, None]).sum(dim=0)
    denominator = (centered_x.square().sum(dim=0) * centered_y.square().sum()).sqrt()
    score = torch.where(denominator > 1.0e-12, numerator.abs() / denominator, 0.0)
    count = min(maximum, int((denominator > 1.0e-12).sum()))
    if count == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.topk(score, count).indices


def ridge_prediction(
    train_features: Tensor,
    train_target: Tensor,
    test_features: Tensor,
    strength: float,
) -> Tensor:
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0, unbiased=False).clamp_min(1.0e-7)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train = torch.cat((train, torch.ones_like(train[:, :1])), dim=1)
    test = torch.cat((test, torch.ones_like(test[:, :1])), dim=1)
    gram = train.T @ train / len(train)
    penalty = torch.eye(gram.shape[0], dtype=gram.dtype) * strength
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + penalty, train.T @ train_target / len(train))
    return test @ weights


def prediction_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    error = prediction - target
    target_variance = (target - target.mean()).square().mean().clamp_min(1.0e-12)
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    correlation = (centered_prediction * centered_target).mean() / (
        centered_prediction.square().mean().sqrt() * centered_target.square().mean().sqrt()
    ).clamp_min(1.0e-12)
    return {
        "r2": float(1.0 - error.square().mean() / target_variance),
        "normalized_rmse": float(
            error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "pearson_correlation": float(correlation),
        "sign_accuracy": float(((prediction >= 0.0) == (target >= 0.0)).float().mean()),
    }


def bootstrap_r2_improvement(
    prediction: Tensor,
    baseline: Tensor,
    target: Tensor,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    improvements = []
    for _ in range(samples):
        indices = torch.randint(0, len(target), (len(target),), generator=generator)
        selected_target = target[indices]
        denominator = (selected_target - selected_target.mean()).square().mean().clamp_min(1e-12)
        probe_error = (prediction[indices] - selected_target).square().mean()
        baseline_error = (baseline[indices] - selected_target).square().mean()
        improvements.append(float((baseline_error - probe_error) / denominator))
    low, median, high = np.quantile(improvements, (0.025, 0.5, 0.975))
    return {
        "lower_95": float(low),
        "median": float(median),
        "upper_95": float(high),
    }


def fit_sparse_probe(
    train_features: Tensor,
    train_target: Tensor,
    test_features: Tensor,
    test_target: Tensor,
    *,
    maximum_features: int,
    bootstrap_samples: int,
    seed: int,
    feature_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    fit_mask = torch.arange(len(train_target)) % 4 != 0
    validation_mask = ~fit_mask
    selected = select_features(train_features[fit_mask], train_target[fit_mask], maximum_features)
    baseline = torch.full_like(test_target, float(train_target.mean()))
    if not len(selected):
        metrics = prediction_metrics(baseline, test_target)
        return {
            "features": 0,
            "selected_ridge": None,
            "metrics": metrics,
            "bootstrap_r2_improvement_over_constant": {
                "lower_95": 0.0,
                "median": 0.0,
                "upper_95": 0.0,
            },
            "representation_gate_passed": False,
            "selected_feature_ids": [],
        }
    fit_x = train_features[:, selected]
    validation_errors = {}
    for strength in RIDGE_STRENGTHS:
        prediction = ridge_prediction(
            fit_x[fit_mask],
            train_target[fit_mask],
            fit_x[validation_mask],
            strength,
        )
        validation_errors[str(strength)] = float(
            (prediction - train_target[validation_mask]).square().mean()
        )
    strength = min(RIDGE_STRENGTHS, key=lambda value: validation_errors[str(value)])
    prediction = ridge_prediction(fit_x, train_target, test_features[:, selected], strength)
    metrics = prediction_metrics(prediction, test_target)
    interval = bootstrap_r2_improvement(
        prediction,
        baseline,
        test_target,
        samples=bootstrap_samples,
        seed=seed,
    )
    identifiers = (
        selected.tolist() if feature_ids is None else feature_ids[selected.numpy()].tolist()
    )
    gate = (
        metrics["r2"] >= REPRESENTATION_R2
        and metrics["sign_accuracy"] >= REPRESENTATION_SIGN_ACCURACY
        and interval["lower_95"] > 0.0
    )
    permutation = torch.randperm(
        len(train_target), generator=torch.Generator().manual_seed(seed + 1)
    )
    shuffled_target = train_target[permutation]
    shuffled_selected = select_features(
        train_features[fit_mask], shuffled_target[fit_mask], maximum_features
    )
    if len(shuffled_selected):
        shuffled_validation_errors = {}
        for candidate in RIDGE_STRENGTHS:
            candidate_prediction = ridge_prediction(
                train_features[fit_mask][:, shuffled_selected],
                shuffled_target[fit_mask],
                train_features[validation_mask][:, shuffled_selected],
                candidate,
            )
            shuffled_validation_errors[str(candidate)] = float(
                (candidate_prediction - shuffled_target[validation_mask]).square().mean()
            )
        shuffled_strength = min(
            RIDGE_STRENGTHS, key=lambda value: shuffled_validation_errors[str(value)]
        )
        shuffled_prediction = ridge_prediction(
            train_features[:, shuffled_selected],
            shuffled_target,
            test_features[:, shuffled_selected],
            shuffled_strength,
        )
    else:
        shuffled_prediction = baseline
        shuffled_strength = None
        shuffled_validation_errors = {}
    return {
        "features": int(len(selected)),
        "selection": "largest absolute development-fit correlation; held-out evaluation",
        "selected_feature_ids": identifiers,
        "selected_ridge": strength,
        "validation_mse_by_ridge": validation_errors,
        "metrics": metrics,
        "bootstrap_r2_improvement_over_constant": interval,
        "representation_gate_passed": gate,
        "shuffled_label_control": {
            "selected_ridge": shuffled_strength,
            "validation_mse_by_ridge": shuffled_validation_errors,
            "metrics": prediction_metrics(shuffled_prediction, test_target),
        },
    }


def scalar_summary(values: Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "rms": float(values.square().mean().sqrt()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def bank_summary(bank: dict[str, Tensor]) -> dict[str, Any]:
    target = bank["target"]
    native = bank["native_motor"][:, 3]
    aligned = native * target.sign()
    return {
        "pairs": len(target),
        "target_contrast": scalar_summary(target),
        "physical_contrast": scalar_summary(bank["physical"]),
        "native_throttle_contrast": scalar_summary(native),
        "native_correct_sign_fraction": float((aligned > REPLAY_NOISE_FLOOR).float().mean()),
        "native_to_target_aligned_fraction": float(
            aligned.mean() / target.abs().mean().clamp_min(REPLAY_NOISE_FLOOR)
        ),
        "endpoint_retina_max_absolute_pair_difference": float(bank["retina"].abs().max()),
        "diagnostics": {
            key: scalar_summary(value)
            for key, value in bank.items()
            if key not in {"state", "retina", "target", "physical", "native_motor"}
        },
    }


def paired_mean_replace(neural: Tensor, indices: Tensor, pairs: int) -> None:
    if not len(indices):
        return
    common = 0.5 * (neural[:pairs, indices] + neural[pairs:, indices])
    neural[:pairs, indices] = common
    neural[pairs:, indices] = common


@torch.no_grad()
def continued_motor_response(
    controller: ConnectomeController,
    batch: AssayBatch,
    *,
    steps: int,
    replacement: Tensor | None,
    identical_control: bool = False,
) -> tuple[Tensor, Tensor]:
    pairs = len(batch.target_contrast)
    if identical_control:
        neural = torch.cat((batch.state_a.clone(), batch.state_a.clone()))
        images = torch.cat((batch.image_a, batch.image_a))
    else:
        neural = torch.cat((batch.state_a.clone(), batch.state_b.clone()))
        images = torch.cat((batch.image_a, batch.image_b))
    attitude = torch.cat((batch.attitude, batch.attitude))
    if replacement is not None:
        paired_mean_replace(neural, replacement, pairs)
    contrast, common = [], []
    for _ in range(steps):
        motor, neural = controller(images, attitude, neural)
        contrast.append(motor[:pairs] - motor[pairs:])
        common.append(0.5 * (motor[:pairs] + motor[pairs:]))
    return torch.stack(contrast), torch.stack(common)


def bootstrap_effect_interval(effect: Tensor, *, samples: int, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(samples):
        indices = torch.randint(0, len(effect), (len(effect),), generator=generator)
        means.append(float(effect[indices].mean()))
    low, median, high = np.quantile(means, (0.025, 0.5, 0.975))
    return {"lower_95": float(low), "median": float(median), "upper_95": float(high)}


@torch.no_grad()
def causal_audit(
    name: str,
    controller: ConnectomeController,
    partitions: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    device = controller.edge_magnitude.device
    primary, secondary = assay_design(name, args.causal_pairs, device=device)
    accumulated: dict[str, dict[str, list[Tensor]]] = {}
    identical_max = 0.0
    whole_state_max = 0.0
    all_indices = torch.arange(controller.n_nodes, device=device)
    masks = {
        key: torch.from_numpy(np.flatnonzero(value)).to(device) for key, value in partitions.items()
    }
    masks["whole_state_control"] = all_indices
    labels = ["untouched", *masks]
    for begin in range(0, args.causal_pairs, args.batch_size):
        end = min(begin + args.batch_size, args.causal_pairs)
        batch = make_assay_batch(
            name,
            controller,
            primary[begin:end],
            secondary[begin:end],
            held_out_scene=True,
            args=args,
            config=config,
        )
        identical, _ = continued_motor_response(
            controller,
            batch,
            steps=args.causal_response_steps,
            replacement=None,
            identical_control=True,
        )
        identical_max = max(identical_max, float(identical.abs().max()))
        for label in labels:
            contrast, common = continued_motor_response(
                controller,
                batch,
                steps=args.causal_response_steps,
                replacement=None if label == "untouched" else masks[label],
            )
            store = accumulated.setdefault(label, {"contrast": [], "common": [], "target": []})
            store["contrast"].append(contrast.cpu())
            store["common"].append(common.cpu())
            store["target"].append(batch.target_contrast.cpu())
            if label == "whole_state_control" and name != "height_error":
                whole_state_max = max(whole_state_max, float(contrast.abs().max()))

    joined = {
        label: {
            "contrast": torch.cat(values["contrast"], dim=1),
            "common": torch.cat(values["common"], dim=1),
            "target": torch.cat(values["target"]),
        }
        for label, values in accumulated.items()
    }
    baseline = joined["untouched"]
    sign = baseline["target"].sign()
    baseline_pair = baseline["contrast"][:, :, 3].mean(dim=0) * sign
    target_magnitude = baseline["target"].abs().mean().clamp_min(REPLAY_NOISE_FLOOR)
    usable_fraction = float(baseline_pair.mean() / target_magnitude)
    baseline_usable = (
        usable_fraction >= USABLE_RESPONSE_FRACTION
        and float((baseline_pair > REPLAY_NOISE_FLOOR).float().mean())
        >= REPRESENTATION_SIGN_ACCURACY
    )
    interventions = {}
    for offset, (label, values) in enumerate(joined.items()):
        if label == "untouched":
            continue
        intervention_pair = values["contrast"][:, :, 3].mean(dim=0) * sign
        effect = baseline_pair - intervention_pair
        common_change = values["common"] - baseline["common"]
        effect_fraction = float(effect.mean() / baseline_pair.mean().abs().clamp_min(1.0e-8))
        interval = bootstrap_effect_interval(
            effect, samples=args.bootstrap_samples, seed=seed + offset + 1
        )
        selective = (
            baseline_usable
            and effect_fraction >= CAUSAL_EFFECT_FRACTION
            and interval["lower_95"] > 0.0
            and float(common_change.abs().max()) <= MAX_COMMON_MOTOR_CHANGE
        )
        interventions[label] = {
            "nodes": int(len(masks[label])),
            "aligned_throttle_response_mean": float(intervention_pair.mean()),
            "response_removed_fraction": effect_fraction,
            "paired_effect_native_units_95_interval": interval,
            "maximum_absolute_common_motor_change": float(common_change.abs().max()),
            "selective_causal_gate_passed": selective,
        }
    return {
        "method": (
            "Replace the selected population once by each pair's mean at the assay endpoint, "
            "then continue the unchanged native graph for the response window."
        ),
        "baseline": {
            "aligned_throttle_response_mean": float(baseline_pair.mean()),
            "correct_sign_fraction": float((baseline_pair > REPLAY_NOISE_FLOOR).float().mean()),
            "response_to_target_fraction": usable_fraction,
            "usable_response_gate_passed": baseline_usable,
        },
        "identical_history_control_max_absolute_motor_difference": identical_max,
        "whole_state_shared_input_control_max_absolute_motor_difference": (
            whole_state_max if name != "height_error" else None
        ),
        "interventions": interventions,
    }


def probe_assay(
    train: dict[str, Tensor],
    test: dict[str, Tensor],
    partitions: dict[str, np.ndarray],
    node_ids: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    probes = {
        "endpoint_retina": fit_sparse_probe(
            train["retina"],
            train["target"],
            test["retina"],
            test["target"],
            maximum_features=args.probe_features,
            bootstrap_samples=args.bootstrap_samples,
            seed=seed,
        )
    }
    for offset, (name, mask) in enumerate(partitions.items(), start=1):
        probes[name] = fit_sparse_probe(
            train["state"][:, mask],
            train["target"],
            test["state"][:, mask],
            test["target"],
            maximum_features=args.probe_features,
            bootstrap_samples=args.bootstrap_samples,
            seed=seed + offset,
            feature_ids=node_ids[mask],
        )
    return probes


def decision_summary(assays: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for name, result in assays.items():
        represented = [
            partition
            for partition, probe in result["probes"].items()
            if partition != "endpoint_retina" and probe["representation_gate_passed"]
        ]
        causal = [
            partition
            for partition, intervention in result["causal"]["interventions"].items()
            if intervention["selective_causal_gate_passed"]
        ]
        summary[name] = {
            "represented_in": represented,
            "selectively_causal_partitions": causal,
            "native_response_usable": result["causal"]["baseline"]["usable_response_gate_passed"],
        }
    error_motion_supported = all(
        summary[name]["represented_in"] and summary[name]["selectively_causal_partitions"]
        for name in ("height_error", "vertical_motion")
    )
    if error_motion_supported:
        route = "anatomy_restricted_progressive_teacher_handoff"
        reason = (
            "Both error and damping information are represented and selectively affect the "
            "native throttle response."
        )
    elif all(summary[name]["represented_in"] for name in ("height_error", "vertical_motion")):
        route = "native_routing_and_readout_training"
        reason = (
            "Control-relevant information is present, but selective downstream use is not yet "
            "established; do not rebuild the graph."
        )
    else:
        route = "verify_temporal_visual_interface_and_module_coverage"
        reason = (
            "At least one representation gate failed. Check retinal temporal signal, saturation "
            "and probe sensitivity before considering a module-preserving graph change."
        )
    command = summary["command_history"]
    if not command["represented_in"]:
        feco = (
            "No self-generated command-memory representation passed this linear audit. FeCO is a "
            "controlled later candidate for observing physical leg state, not a prerequisite yet."
        )
    else:
        feco = (
            "Self-generated command history is represented somewhere in the native state. This "
            "does not establish hover-thrust estimation or remove the value of a later FeCO test."
        )
    return {"by_assay": summary, "next_route": route, "reason": reason, "feco_note": feco}


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if int(checkpoint.get("policy_hz", args.policy_hz)) != args.policy_hz:
        raise SystemExit("checkpoint and requested policy frequencies do not match")
    config = HoverConfig()
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    if controller.uses_accelerometer or controller.uses_proprioception:
        raise SystemExit("this audit requires the RGB+roll/pitch-only source graph")

    partitions, node_ids = annotation_partitions(args.graph, args.raw_dir)
    started = perf_counter()
    assays: dict[str, Any] = {}
    for assay_index, name in enumerate(ASSAY_NAMES):
        development = collect_probe_bank(
            name,
            controller,
            pairs=args.development_pairs,
            held_out_scene=False,
            seed=args.seed + 100 * assay_index,
            args=args,
            config=config,
        )
        test = collect_probe_bank(
            name,
            controller,
            pairs=args.test_pairs,
            held_out_scene=True,
            seed=args.seed + 100 * assay_index + 1,
            args=args,
            config=config,
        )
        assays[name] = {
            "development": bank_summary(development),
            "test": bank_summary(test),
            "probes": probe_assay(
                development,
                test,
                partitions,
                node_ids,
                args=args,
                seed=args.seed + 100 * assay_index + 2,
            ),
            "causal": causal_audit(
                name,
                controller,
                partitions,
                args=args,
                config=config,
                seed=args.seed + 100 * assay_index + 3,
            ),
        }
        del development, test

    report = {
        "experiment": "variable-height-native-control-responsibility-audit-v1",
        "status": "diagnostic_only_no_parameter_changes",
        "source": {
            "graph": stable_path(args.graph),
            "graph_sha256": file_sha256(args.graph),
            "checkpoint": stable_path(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
        },
        "actor_contract": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll angle", "pitch angle"],
            "recurrent_state": "MaleCNS graph only",
            "outputs": ["roll", "pitch", "yaw", "throttle via 26 T1 foreleg motor neurons"],
            "accelerometer": False,
            "proprioception": False,
            "decoded_probe_values_used_by_actor": False,
        },
        "protocol": {
            "development_pairs": args.development_pairs,
            "test_pairs": args.test_pairs,
            "causal_pairs": args.causal_pairs,
            "policy_hz": args.policy_hz,
            "prefix_steps": args.prefix_steps,
            "history_steps": args.history_steps,
            "neutral_steps_for_command_history": args.neutral_steps,
            "causal_response_steps": args.causal_response_steps,
            "probe_feature_cap": args.probe_features,
            "camera": {
                "width": DEFAULT_VISUAL_CAMERA.width,
                "height": DEFAULT_VISUAL_CAMERA.height,
                "horizontal_fov_degrees": DEFAULT_VISUAL_CAMERA.horizontal_fov_degrees,
            },
            "assays": {
                "height_error": (
                    "Common prefix and identical pose; physical marker differs by +/-5 or 10 cm."
                ),
                "vertical_motion": (
                    "Mirrored smooth camera-height histories end at identical pose and pixels "
                    "with endpoint speeds +/-0.15 or 0.30 m/s and balanced final marker error."
                ),
                "command_history": (
                    "Marker pulses induce different native commands and shadow leg/actuator "
                    "states, followed by 0.26 s of identical RGB and attitude."
                ),
            },
            "interpretation_limit": (
                "These prescribed sensory replays test representation and local causal use, not "
                "closed-loop flight. Linear-probe failure does not prove information absence. "
                "Command history is not a hover-thrust estimate."
            ),
        },
        "preregistered_gates": {
            "representation": {
                "test_r2_at_least": REPRESENTATION_R2,
                "test_sign_accuracy_at_least": REPRESENTATION_SIGN_ACCURACY,
                "bootstrap_r2_improvement_over_constant_lower_95_above": 0.0,
                "history_assay_endpoint_retina_pair_difference_expected": 0.0,
            },
            "usable_native_response": {
                "aligned_response_fraction_at_least": USABLE_RESPONSE_FRACTION,
                "correct_sign_fraction_at_least": REPRESENTATION_SIGN_ACCURACY,
                "noise_floor_native_motor_units": REPLAY_NOISE_FLOOR,
            },
            "selective_causal_contribution": {
                "response_removed_fraction_at_least": CAUSAL_EFFECT_FRACTION,
                "paired_effect_lower_95_above": 0.0,
                "maximum_absolute_common_motor_change": MAX_COMMON_MOTOR_CHANGE,
            },
        },
        "partition_throttle_reachability": throttle_reachability(args.graph, partitions),
        "assays": assays,
        "decision": decision_summary(assays),
        "elapsed_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps({"output": stable_path(output), "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
