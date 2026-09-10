#!/usr/bin/env python3
"""Fit a paired signed visual-height response in the native MaleCNS graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
)
from flydrone.visual_hover import (  # noqa: E402
    HELD_OUT_MARKER_HEIGHT_BAND,
    TRAIN_MARKER_HEIGHT_BANDS,
    render_visual_hover_scene,
    sample_marker_heights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/pilot-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--response-steps", type=int, default=25)
    parser.add_argument("--prefix-min-steps", type=int, default=5)
    parser.add_argument("--prefix-max-steps", type=int, default=100)
    parser.add_argument("--marker-step-metres", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--evaluation-pairs", type=int, default=64)
    return parser.parse_args()


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


def controller_regularization(
    controller: ConnectomeController, source_parameters: dict[str, torch.Tensor]
) -> torch.Tensor:
    preservation_scale = 0.006
    return (
        0.03
        * ((controller.edge_magnitude - source_parameters["edge_magnitude"]) / preservation_scale)
        .square()
        .mean()
        + 0.03
        * ((controller.bias - source_parameters["bias"]) / preservation_scale).square().mean()
        + 0.01
        * (
            (controller.raw_time_constant - source_parameters["raw_time_constant"])
            / preservation_scale
        )
        .square()
        .mean()
    )


def paired_states(
    batch: int,
    device: torch.device,
    config: HoverConfig,
    marker_step_metres: float,
    *,
    held_out: bool,
) -> tuple[QuadState, torch.Tensor, torch.Tensor, torch.Tensor]:
    quad = DifferentiableQuad(config).to(device)
    height = sample_marker_heights(batch, device=device, held_out=held_out)
    position = torch.zeros(batch, 3, device=device)
    position[:, 2] = height
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(3.0), math.radians(3.0)
    )
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    state.actuator[:, 0] = 1.0 / config.thrust_to_weight
    state.velocity[:, 2] = torch.empty(batch, device=device).uniform_(-0.20, 0.20)
    state.rates[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(8.0), math.radians(8.0)
    )
    direction = torch.where(
        torch.rand(batch, device=device) < 0.5,
        -torch.ones(batch, device=device),
        torch.ones(batch, device=device),
    )
    target_a = height + direction * marker_step_metres
    target_b = height - direction * marker_step_metres
    return state, height, target_a, target_b


def target_motor(
    state: QuadState, target_height: torch.Tensor, config: HoverConfig
) -> torch.Tensor:
    roll_rate = -3.5 * state.euler[:, 0] - 0.45 * state.rates[:, 0]
    pitch_rate = -3.5 * state.euler[:, 1] - 0.45 * state.rates[:, 1]
    rc = torch.zeros(state.position.shape[0], 4, device=state.position.device)
    rc[:, 0] = (roll_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    rc[:, 1] = (pitch_rate / config.max_roll_pitch_rate).clamp(-1.0, 1.0)
    rc[:, 3] = (
        1.0 / config.thrust_to_weight + 0.42 * (target_height - state.position[:, 2])
    ).clamp(0.0, 1.0)
    return motor_target_for_rc(rc, config)


def fixed_target_scales(
    config: HoverConfig, marker_step_metres: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    quad = DifferentiableQuad(config).to(device)
    state = quad.initial_state(
        1,
        device=device,
        dtype=torch.float32,
        position=torch.tensor(((0.0, 0.0, 1.0),), device=device),
    )
    upper = target_motor(state, torch.tensor([1.0 + marker_step_metres], device=device), config)
    lower = target_motor(state, torch.tensor([1.0 - marker_step_metres], device=device), config)
    contrast_scale = (upper[0, 3] - lower[0, 3]).abs()
    mean_scale = ((upper[0, 3] + lower[0, 3]) / 2.0).abs()
    return contrast_scale, mean_scale


def settled_sticks(batch: int, device: torch.device, config: HoverConfig) -> StickState:
    rc = torch.zeros(batch, 4, device=device)
    rc[:, 3] = 1.0 / config.thrust_to_weight
    normalized = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    sine_limit = math.sin(config.foreleg_joint_limit)
    joint = torch.asin((normalized * sine_limit).clamp(-1.0, 1.0))
    zeros = torch.zeros_like(normalized)
    return StickState(joint, zeros.clone(), normalized, zeros)


@torch.no_grad()
def common_prefix(
    controller: ConnectomeController,
    state: QuadState,
    target_height: torch.Tensor,
    steps: int,
    config: HoverConfig,
) -> tuple[torch.Tensor, QuadState]:
    quad = DifferentiableQuad(config).to(state.position.device)
    sticks = ForelegStickPlant(config).to(state.position.device)
    stick_state = settled_sticks(state.position.shape[0], state.position.device, config)
    mass_scale = torch.ones(state.position.shape[0], device=state.position.device)
    neural = controller.initial_state(
        state.position.shape[0], device=state.position.device, dtype=torch.float32
    )
    for _ in range(steps):
        image = render_visual_hover_scene(state, target_height)
        _, neural = controller(image, state.euler[:, :2], neural)
        teacher_motor = target_motor(state, target_height, config)
        for _ in range(2):
            rc, stick_state = sticks(teacher_motor, stick_state)
            state = quad(rc, state, mass_scale)
    return neural, state


def branch_response(
    controller: ConnectomeController,
    state: QuadState,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    initial_neural: torch.Tensor,
    response_steps: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    combined_image = torch.cat(
        (
            render_visual_hover_scene(state, target_a),
            render_visual_hover_scene(state, target_b),
        ),
        dim=0,
    )
    combined_attitude = torch.cat((state.euler[:, :2], state.euler[:, :2]), dim=0)
    neural = torch.cat((initial_neural.clone(), initial_neural.clone()), dim=0)
    predictions_a = []
    predictions_b = []
    batch = state.position.shape[0]
    for _ in range(response_steps):
        motor, neural = controller(combined_image, combined_attitude, neural)
        predictions_a.append(motor[:batch])
        predictions_b.append(motor[batch:])
    return predictions_a, predictions_b


def paired_training_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    response_steps: int,
    prefix_min_steps: int,
    prefix_max_steps: int,
    marker_step_metres: float,
    contrast_scale: torch.Tensor,
    mean_scale: torch.Tensor,
    policy_hz: int,
    source_parameters: dict[str, torch.Tensor],
    device: torch.device,
    config: HoverConfig,
) -> dict[str, float]:
    state, prefix_target, target_a, target_b = paired_states(
        batch, device, config, marker_step_metres, held_out=False
    )
    prefix_steps = random.randint(prefix_min_steps, prefix_max_steps)
    initial_neural, state = common_prefix(controller, state, prefix_target, prefix_steps, config)
    predictions_a, predictions_b = branch_response(
        controller, state, target_a, target_b, initial_neural, response_steps
    )
    desired_a = target_motor(state, target_a, config)
    desired_b = target_motor(state, target_b, config)
    desired_contrast = desired_a[:, 3] - desired_b[:, 3]
    desired_mean = (desired_a[:, 3] + desired_b[:, 3]) / 2.0
    contrast_losses = []
    mean_losses = []
    axis_losses = []
    loss_start = min(response_steps - 1, round(0.2 * policy_hz))
    for prediction_a, prediction_b in zip(
        predictions_a[loss_start:], predictions_b[loss_start:], strict=True
    ):
        predicted_contrast = prediction_a[:, 3] - prediction_b[:, 3]
        predicted_mean = (prediction_a[:, 3] + prediction_b[:, 3]) / 2.0
        contrast_losses.append(
            ((predicted_contrast - desired_contrast) / contrast_scale).square().mean()
        )
        mean_losses.append(((predicted_mean - desired_mean) / mean_scale).square().mean())
        axis_scale = 0.05
        axis_losses.append(
            ((prediction_a[:, :3] - desired_a[:, :3]) / axis_scale).square().mean()
            + ((prediction_b[:, :3] - desired_b[:, :3]) / axis_scale).square().mean()
        )
    contrast_loss = torch.stack(contrast_losses).mean()
    mean_loss = torch.stack(mean_losses).mean()
    axis_loss = torch.stack(axis_losses).mean()
    loss = (
        contrast_loss
        + 0.75 * mean_loss
        + 0.5 * axis_loss
        + controller_regularization(controller, source_parameters)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.7)
    optimizer.step()
    controller.project_parameters()
    return {
        "loss": float(loss.detach()),
        "contrast_nrmse": float(torch.sqrt(contrast_loss.detach())),
        "mean_nrmse": float(torch.sqrt(mean_loss.detach())),
        "axis_nrmse": float(torch.sqrt(axis_loss.detach() / 2.0)),
        "gradient_norm": float(gradient_norm.detach()),
        "prefix_steps": float(prefix_steps),
    }


@torch.no_grad()
def evaluate_pairs(
    controller: ConnectomeController,
    *,
    pairs: int,
    batch_size: int,
    response_steps: int,
    prefix_min_steps: int,
    prefix_max_steps: int,
    marker_step_metres: float,
    contrast_scale: torch.Tensor,
    device: torch.device,
    config: HoverConfig,
    seed: int,
    policy_hz: int,
) -> dict[str, Any]:
    seed_everything(seed)
    predicted_contrasts = []
    desired_contrasts = []
    identical_contrasts = []
    swapped_errors = []
    remaining = pairs
    while remaining:
        batch = min(batch_size, remaining)
        state, prefix_target, target_a, target_b = paired_states(
            batch, device, config, marker_step_metres, held_out=True
        )
        prefix_steps = random.randint(prefix_min_steps, prefix_max_steps)
        initial_neural, state = common_prefix(
            controller, state, prefix_target, prefix_steps, config
        )
        prediction_a, prediction_b = branch_response(
            controller, state, target_a, target_b, initial_neural, response_steps
        )
        window = min(len(prediction_a), max(1, round(0.2 * policy_hz)))
        final_a = torch.stack(prediction_a[-window:]).mean(dim=0)
        final_b = torch.stack(prediction_b[-window:]).mean(dim=0)
        predicted_contrast = final_a[:, 3] - final_b[:, 3]
        desired_a = target_motor(state, target_a, config)
        desired_b = target_motor(state, target_b, config)
        desired_contrast = desired_a[:, 3] - desired_b[:, 3]
        predicted_contrasts.append(predicted_contrast)
        desired_contrasts.append(desired_contrast)

        identical_a, identical_b = branch_response(
            controller, state, target_a, target_a, initial_neural, response_steps
        )
        identical_contrasts.append(identical_a[-1][:, 3] - identical_b[-1][:, 3])
        swapped_a, swapped_b = branch_response(
            controller, state, target_b, target_a, initial_neural, response_steps
        )
        swapped = swapped_a[-1][:, 3] - swapped_b[-1][:, 3]
        original_final = prediction_a[-1][:, 3] - prediction_b[-1][:, 3]
        swapped_errors.append(swapped + original_final)
        remaining -= batch

    predicted = torch.cat(predicted_contrasts)
    desired = torch.cat(desired_contrasts)
    error = predicted - desired
    nrmse = torch.sqrt(error.square().mean()) / contrast_scale
    slope = (predicted * desired).sum() / desired.square().sum()
    correct_sign = (predicted * desired > 0.0).float().mean()
    identical = torch.cat(identical_contrasts)
    swapped_error = torch.cat(swapped_errors)
    passed = bool(
        float(nrmse) <= 0.25
        and float(correct_sign) >= 0.95
        and 0.5 <= float(slope) <= 1.5
        and float(identical.abs().max()) <= 1.0e-6
        and float(swapped_error.abs().max()) <= 1.0e-5
    )
    return {
        "pairs": pairs,
        "height_band_metres": list(HELD_OUT_MARKER_HEIGHT_BAND),
        "contrast_nrmse": float(nrmse),
        "correct_sign_rate": float(correct_sign),
        "response_slope": float(slope),
        "predicted_contrast_rms": float(torch.sqrt(predicted.square().mean())),
        "desired_contrast_rms": float(torch.sqrt(desired.square().mean())),
        "identical_image_contrast_max_absolute": float(identical.abs().max()),
        "swapped_image_reversal_error_max_absolute": float(swapped_error.abs().max()),
        "pass": passed,
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    args: argparse.Namespace,
    source_checkpoint_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_schema_version": 1,
            "experiment": "paired-dynamic-visual-height-response-v2",
            "controller": controller.state_dict(),
            "graph_sha256": file_sha256(args.graph),
            "source_checkpoint": str(args.checkpoint),
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "policy_hz": args.policy_hz,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed)
    config = HoverConfig()
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source_checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint graph hash does not match --graph")
    controller.load_state_dict(checkpoint["controller"])
    source_parameters = {
        name: checkpoint["controller"][name].detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }
    contrast_scale, mean_scale = fixed_target_scales(config, args.marker_step_metres, device)

    baseline = evaluate_pairs(
        controller,
        pairs=args.evaluation_pairs,
        batch_size=args.batch_size,
        response_steps=args.response_steps,
        prefix_min_steps=args.prefix_min_steps,
        prefix_max_steps=args.prefix_max_steps,
        marker_step_metres=args.marker_step_metres,
        contrast_scale=contrast_scale,
        device=device,
        config=config,
        seed=args.seed + 10_000,
        policy_hz=args.policy_hz,
    )
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
    )
    started = perf_counter()
    history = []
    for iteration in range(args.iterations):
        metrics = paired_training_step(
            controller,
            optimizer,
            batch=args.batch_size,
            response_steps=args.response_steps,
            prefix_min_steps=args.prefix_min_steps,
            prefix_max_steps=args.prefix_max_steps,
            marker_step_metres=args.marker_step_metres,
            contrast_scale=contrast_scale,
            mean_scale=mean_scale,
            policy_hz=args.policy_hz,
            source_parameters=source_parameters,
            device=device,
            config=config,
        )
        if iteration % 10 == 0 or iteration + 1 == args.iterations:
            entry = {"iteration": iteration + 1, **metrics}
            history.append(entry)
            print(json.dumps(entry), flush=True)

    held_out = evaluate_pairs(
        controller,
        pairs=args.evaluation_pairs,
        batch_size=args.batch_size,
        response_steps=args.response_steps,
        prefix_min_steps=args.prefix_min_steps,
        prefix_max_steps=args.prefix_max_steps,
        marker_step_metres=args.marker_step_metres,
        contrast_scale=contrast_scale,
        device=device,
        config=config,
        seed=args.seed + 10_000,
        policy_hz=args.policy_hz,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller.pt"
    save_checkpoint(checkpoint_path, controller, args, source_checkpoint_sha256)
    report = {
        "passed": held_out["pass"],
        "experiment": "paired-dynamic-visual-height-response-v2",
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "policy_hz": args.policy_hz,
        "marker_step_metres": args.marker_step_metres,
        "training_height_bands_metres": [list(band) for band in TRAIN_MARKER_HEIGHT_BANDS],
        "held_out_height_band_metres": list(HELD_OUT_MARKER_HEIGHT_BAND),
        "paired_invariant": (
            "pose, floor, wall texture, attitude, and initial neural state are identical; "
            "only physical marker height differs"
        ),
        "prefix": "teacher-flown dynamic visual and attitude history",
        "source_relative_parameter_regularization": True,
        "axis_protection": "normalized teacher targets after dynamic prefixes",
        "fixed_contrast_scale": float(contrast_scale),
        "fixed_mean_scale": float(mean_scale),
        "iterations": args.iterations,
        "elapsed_seconds": perf_counter() - started,
        "baseline": baseline,
        "history": history,
        "held_out": held_out,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
