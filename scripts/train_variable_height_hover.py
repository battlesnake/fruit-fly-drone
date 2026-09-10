#!/usr/bin/env python3
"""Run the bounded mixed-curriculum variable-height hover experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_visual_height_response as legacy_response  # noqa: E402

from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
    teacher_rc,
)
from flydrone.variable_hover import (  # noqa: E402
    CAMERA_HEIGHT_BANDS,
    MarkerPairConditions,
    protocol_manifest,
    sample_height_conditions,
    sample_marker_pairs,
    sample_marker_steps,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    VisualScene,
    render_visual_hover_scene,
    sample_visual_scenes,
    visual_scene_manifest,
)

LESSON_CYCLE = ("closed_loop", "paired_marker", "closed_loop", "attitude_recovery")
MOTOR_CORRECTION_SCALES = (0.05, 0.05, 0.04, 0.05)


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
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/bounded-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=113)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument("--evaluation-every", type=int, default=50)
    parser.add_argument("--interim-evaluation-episodes", type=int, default=64)
    parser.add_argument("--final-evaluation-episodes", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--response-seconds", type=float, default=6.0)
    parser.add_argument("--evaluate-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def clone_sticks(state: StickState) -> StickState:
    return StickState(
        state.joint_position.clone(),
        state.joint_velocity.clone(),
        state.position.clone(),
        state.velocity.clone(),
    )


def settled_sticks(batch: int, device: torch.device, config: HoverConfig) -> StickState:
    rc = torch.zeros(batch, 4, device=device)
    rc[:, 3] = 1.0 / config.thrust_to_weight
    normalized = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    sine_limit = math.sin(config.foreleg_joint_limit)
    joint = torch.asin((normalized * sine_limit).clamp(-1.0, 1.0))
    zeros = torch.zeros_like(normalized)
    return StickState(joint, zeros.clone(), normalized, zeros)


def nominal_initial_state(
    batch: int,
    *,
    camera_height: torch.Tensor,
    device: torch.device,
    config: HoverConfig,
    attitude_degrees: float,
    rate_degrees_per_second: float,
    vertical_speed: float,
) -> tuple[QuadState, StickState]:
    quad = DifferentiableQuad(config).to(device)
    position = torch.empty(batch, 3, device=device).uniform_(-0.12, 0.12)
    position[:, 2] = camera_height
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(attitude_degrees), math.radians(attitude_degrees)
    )
    euler[:, 2] = torch.empty(batch, device=device).uniform_(-math.radians(5.0), math.radians(5.0))
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    state.velocity[:, :2] = torch.empty(batch, 2, device=device).uniform_(-0.08, 0.08)
    state.velocity[:, 2] = torch.empty(batch, device=device).uniform_(
        -vertical_speed, vertical_speed
    )
    state.rates[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(rate_degrees_per_second), math.radians(rate_degrees_per_second)
    )
    state.actuator[:, 0] = 1.0 / config.thrust_to_weight
    return state, settled_sticks(batch, device, config)


def state_is_valid(state: QuadState) -> torch.Tensor:
    finite = torch.stack([torch.isfinite(value).all(dim=1) for value in state.as_tuple()]).all(
        dim=0
    )
    tilt = torch.linalg.vector_norm(state.euler[:, :2], dim=1)
    return finite & (state.position[:, 2] > 0.03) & (tilt < math.radians(70.0))


def teacher_motor(state: QuadState, marker: torch.Tensor, config: HoverConfig) -> torch.Tensor:
    detached = state.detach()
    return motor_target_for_rc(teacher_rc(detached, marker.detach(), config), config).detach()


def source_regularization(
    controller: ConnectomeController, source: dict[str, torch.Tensor]
) -> torch.Tensor:
    scale = 0.006
    return (
        0.015 * ((controller.edge_magnitude - source["edge_magnitude"]) / scale).square().mean()
        + 0.015 * ((controller.bias - source["bias"]) / scale).square().mean()
        + 0.005
        * ((controller.raw_time_constant - source["raw_time_constant"]) / scale).square().mean()
    )


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def advance_physics(
    quad: DifferentiableQuad,
    sticks: ForelegStickPlant,
    motor: torch.Tensor,
    state: QuadState,
    stick_state: StickState,
    physics_steps: int,
) -> tuple[QuadState, StickState, torch.Tensor]:
    mass = torch.ones(motor.shape[0], device=motor.device)
    rc = torch.zeros_like(motor)
    for _ in range(physics_steps):
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass)
    return state, stick_state, rc


@torch.no_grad()
def current_policy_prefix(
    controller: ConnectomeController,
    state: QuadState,
    stick_state: StickState,
    neural: torch.Tensor,
    marker: torch.Tensor,
    scene: VisualScene,
    *,
    steps: int,
    physics_steps: int,
    config: HoverConfig,
) -> tuple[QuadState, StickState, torch.Tensor, torch.Tensor]:
    quad = DifferentiableQuad(config).to(state.position.device)
    sticks = ForelegStickPlant(config).to(state.position.device)
    valid = state_is_valid(state)
    for _ in range(steps):
        image = render_visual_hover_scene(state, marker, config=config, scene=scene)
        motor, neural = controller(image, state.euler[:, :2], neural)
        state, stick_state, _ = advance_physics(
            quad, sticks, motor, state, stick_state, physics_steps
        )
        valid &= state_is_valid(state)
    return state.detach(), stick_state.detach(), neural.detach(), valid


def trajectory_lesson(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: dict[str, torch.Tensor],
    *,
    batch: int,
    unroll: int,
    prefix_steps: int,
    physics_steps: int,
    attitude_recovery: bool,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, float]:
    if attitude_recovery:
        conditions = sample_height_conditions(batch, device=device, split="train")
        initial_marker = conditions.marker_height
        final_marker = initial_marker
        camera_height = conditions.camera_height
        attitude_degrees = 13.0
        rates = 28.0
        vertical_speed = 0.20
    else:
        stepped = random.random() < 0.67
        if stepped:
            steps = sample_marker_steps(batch, device=device, held_out_final=False)
            initial_marker = steps.initial_marker_height
            final_marker = steps.final_marker_height
            camera_noise = torch.empty(batch, device=device).uniform_(-0.08, 0.08)
            camera_height = (initial_marker + camera_noise).clamp(0.45, 1.55)
        else:
            conditions = sample_height_conditions(batch, device=device, split="train")
            initial_marker = conditions.marker_height
            final_marker = initial_marker
            camera_height = conditions.camera_height
        attitude_degrees = 5.0
        rates = 12.0
        vertical_speed = 0.25

    state, stick_state = nominal_initial_state(
        batch,
        camera_height=camera_height,
        device=device,
        config=config,
        attitude_degrees=attitude_degrees,
        rate_degrees_per_second=rates,
        vertical_speed=vertical_speed,
    )
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    scene = sample_visual_scenes(batch, device=device, held_out_combinations=False)
    state, stick_state, neural, valid = current_policy_prefix(
        controller,
        state,
        stick_state,
        neural,
        initial_marker,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    # A small physical impulse makes the temporal label depend on visible motion. It is
    # simulator state, not an actor channel.
    state.velocity[:, 2] += torch.empty(batch, device=device).uniform_(-0.18, 0.18)

    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    scales = state.position.new_tensor(MOTOR_CORRECTION_SCALES)
    axis_weights = state.position.new_tensor(
        (2.0, 2.0, 0.75, 0.75) if attitude_recovery else (1.0, 1.0, 0.5, 2.0)
    )
    losses = []
    predicted_errors = []
    valid_fractions = []
    for _ in range(unroll):
        image = render_visual_hover_scene(state, final_marker, config=config, scene=scene)
        prediction, neural = controller(image, state.euler[:, :2], neural)
        target = teacher_motor(state, final_marker, config)
        per_axis = ((prediction - target) / scales).square() * axis_weights
        losses.append(masked_mean(per_axis.mean(dim=1), valid))
        predicted_errors.append((prediction.detach() - target).abs())
        valid_fractions.append(valid.float().mean())

        # DAgger timing: the student's action drives the next observation, but physical
        # and stick evolution are detached. Only native neural recurrence receives BPTT.
        state, stick_state, _ = advance_physics(
            quad,
            sticks,
            prediction.detach(),
            state,
            stick_state,
            physics_steps,
        )
        state = state.detach()
        stick_state = stick_state.detach()
        valid &= state_is_valid(state)

    imitation = torch.stack(losses).mean()
    regularization = source_regularization(controller, source)
    loss = imitation + regularization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.7)
    optimizer.step()
    controller.project_parameters()
    mean_errors = torch.stack(predicted_errors).mean(dim=(0, 1))
    return {
        "loss": float(loss.detach()),
        "imitation_loss": float(imitation.detach()),
        "regularization": float(regularization.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "valid_fraction": float(torch.stack(valid_fractions).mean()),
        "prefix_steps": float(prefix_steps),
        "final_height_error_mean_m": float((state.position[:, 2] - final_marker).abs().mean()),
        "roll_motor_mae": float(mean_errors[0]),
        "pitch_motor_mae": float(mean_errors[1]),
        "yaw_motor_mae": float(mean_errors[2]),
        "throttle_motor_mae": float(mean_errors[3]),
    }


def _paired_initial_state(
    pairs: MarkerPairConditions,
    *,
    device: torch.device,
    config: HoverConfig,
    held_out_scene: bool,
    fixed_wall_style: int | None = None,
    fixed_floor_style: int | None = None,
    all_style_combinations: bool = False,
) -> tuple[QuadState, StickState, VisualScene]:
    batch = pairs.centre_height.shape[0]
    if pairs.held_out:
        camera_family = torch.randint(0, 2, (batch,), device=device)
        bands = torch.tensor(CAMERA_HEIGHT_BANDS, device=device)
        unit = torch.rand(batch, device=device)
        camera_height = bands[camera_family, 0] + unit * (
            bands[camera_family, 1] - bands[camera_family, 0]
        )
    else:
        camera_noise = torch.empty(batch, device=device).uniform_(-0.10, 0.10)
        camera_height = (pairs.centre_height + camera_noise).clamp(0.45, 1.55)
    state, sticks = nominal_initial_state(
        batch,
        camera_height=camera_height,
        device=device,
        config=config,
        attitude_degrees=4.0,
        rate_degrees_per_second=10.0,
        vertical_speed=0.20,
    )
    scene = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=held_out_scene,
        fixed_wall_style=fixed_wall_style,
        fixed_floor_style=fixed_floor_style,
        all_style_combinations=all_style_combinations,
    )
    return state, sticks, scene


def paired_predictions(
    controller: ConnectomeController,
    state: QuadState,
    neural: torch.Tensor,
    scene: VisualScene,
    marker_a: torch.Tensor,
    marker_b: torch.Tensor,
    *,
    response_steps: int,
    loss_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = state.position.shape[0]
    image_a = render_visual_hover_scene(state, marker_a, scene=scene)
    image_b = render_visual_hover_scene(state, marker_b, scene=scene)
    images = torch.cat((image_a, image_b))
    attitude = torch.cat((state.euler[:, :2], state.euler[:, :2]))
    recurrent = torch.cat((neural.clone(), neural.clone()))
    outputs_a = []
    outputs_b = []
    for step in range(response_steps):
        motor, recurrent = controller(images, attitude, recurrent)
        if step >= loss_start:
            outputs_a.append(motor[:batch])
            outputs_b.append(motor[batch:])
    return torch.stack(outputs_a), torch.stack(outputs_b)


def paired_lesson(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source: dict[str, torch.Tensor],
    *,
    batch: int,
    unroll: int,
    prefix_steps: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, float]:
    pairs = sample_marker_pairs(batch, device=device, held_out=False)
    state, stick_state, scene = _paired_initial_state(
        pairs, device=device, config=config, held_out_scene=False
    )
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    state, _, neural, valid = current_policy_prefix(
        controller,
        state,
        stick_state,
        neural,
        pairs.centre_height,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    outputs_a, outputs_b = paired_predictions(
        controller,
        state,
        neural,
        scene,
        pairs.marker_a,
        pairs.marker_b,
        response_steps=unroll,
        loss_start=min(unroll - 1, 10),
    )
    target_a = teacher_motor(state, pairs.marker_a, config)
    target_b = teacher_motor(state, pairs.marker_b, config)
    desired_contrast = target_a[:, 3] - target_b[:, 3]
    predicted_contrast = outputs_a[:, :, 3] - outputs_b[:, :, 3]
    contrast_scale = torch.sqrt(desired_contrast.square().mean()).clamp_min(0.01)
    valid_steps = valid[None].expand(outputs_a.shape[0], -1)
    contrast = masked_mean(
        ((predicted_contrast - desired_contrast[None]) / contrast_scale).square(),
        valid_steps,
    )
    predicted_mean = 0.5 * (outputs_a[:, :, 3] + outputs_b[:, :, 3])
    desired_mean = 0.5 * (target_a[:, 3] + target_b[:, 3])
    mean = masked_mean(
        ((predicted_mean - desired_mean[None]) / MOTOR_CORRECTION_SCALES[3]).square(),
        valid_steps,
    )
    axes = 0.5 * (outputs_a[:, :, :3] + outputs_b[:, :, :3])
    desired_axes = 0.5 * (target_a[:, :3] + target_b[:, :3])
    axis_scales = outputs_a.new_tensor(MOTOR_CORRECTION_SCALES[:3])
    axis_error = ((axes - desired_axes[None]) / axis_scales).square().mean(dim=2)
    axis = masked_mean(axis_error, valid_steps)
    regularization = source_regularization(controller, source)
    loss = contrast + 0.75 * mean + 0.75 * axis + regularization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.7)
    optimizer.step()
    controller.project_parameters()
    return {
        "loss": float(loss.detach()),
        "contrast_nrmse": float(torch.sqrt(contrast.detach())),
        "mean_nrmse": float(torch.sqrt(mean.detach())),
        "axis_nrmse": float(torch.sqrt(axis.detach())),
        "regularization": float(regularization.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "valid_fraction": float(valid.float().mean()),
        "prefix_steps": float(prefix_steps),
    }


def prefix_stratum(iteration: int, policy_hz: int) -> tuple[str, int]:
    stratum = (iteration // len(LESSON_CYCLE)) % 3
    if stratum == 0:
        return "early", random.randint(0, round(0.5 * policy_hz))
    if stratum == 1:
        return "middle", random.randint(round(1.5 * policy_hz), round(2.5 * policy_hz))
    return "late", random.randint(round(4.0 * policy_hz), round(5.0 * policy_hz))


def _maximum_true_run(values: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(values.shape[1], dtype=torch.long, device=values.device)
    active = torch.zeros_like(result)
    for row in values:
        active = torch.where(row, active + 1, torch.zeros_like(active))
        result = torch.maximum(result, active)
    return result


def _settling_time(
    height: torch.Tensor,
    vertical_speed: torch.Tensor,
    target: torch.Tensor,
    policy_hz: int,
) -> torch.Tensor:
    inside = (height - target[None]).abs().le(0.15) & vertical_speed.abs().le(0.20)
    required = max(1, round(0.5 * policy_hz))
    result = torch.full((height.shape[1],), float("inf"), device=height.device, dtype=height.dtype)
    if inside.shape[0] < required:
        return result
    windows = inside.unfold(0, required, 1).all(dim=2)
    for episode in range(height.shape[1]):
        candidates = torch.nonzero(windows[:, episode], as_tuple=False).flatten()
        if candidates.numel():
            result[episode] = candidates[0].to(height.dtype) / policy_hz
    return result


@torch.no_grad()
def _rollout_response(
    controller: ConnectomeController,
    state: QuadState,
    stick_state: StickState,
    neural: torch.Tensor,
    marker: torch.Tensor,
    scene: VisualScene,
    *,
    steps: int,
    physics_steps: int,
    config: HoverConfig,
    frozen_image: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    quad = DifferentiableQuad(config).to(state.position.device)
    sticks = ForelegStickPlant(config).to(state.position.device)
    heights = []
    vertical_speeds = []
    tilts = []
    controls = []
    ground = torch.zeros(state.position.shape[0], dtype=torch.bool, device=state.position.device)
    invalid = torch.zeros_like(ground)
    for _ in range(steps):
        image = (
            frozen_image
            if frozen_image is not None
            else render_visual_hover_scene(state, marker, config=config, scene=scene)
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        state, stick_state, rc = advance_physics(
            quad, sticks, motor, state, stick_state, physics_steps
        )
        heights.append(state.position[:, 2])
        vertical_speeds.append(state.velocity[:, 2])
        tilts.append(torch.linalg.vector_norm(state.euler[:, :2], dim=1))
        controls.append(rc)
        ground |= state.position[:, 2] <= 0.005
        invalid |= ~state_is_valid(state)
    return {
        "height": torch.stack(heights),
        "vertical_speed": torch.stack(vertical_speeds),
        "tilt": torch.stack(tilts),
        "controls": torch.stack(controls),
        "ground": ground,
        "invalid": invalid,
    }


@torch.no_grad()
def evaluate_marker_steps(
    controller: ConnectomeController,
    *,
    episodes: int,
    batch_size: int,
    prefix_steps: int,
    response_steps: int,
    physics_steps: int,
    policy_hz: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    collected: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "rmse",
            "vertical_rms",
            "tilt_rms",
            "settling_time",
            "ground",
            "invalid",
            "sustained_saturation",
            "direction",
            "live_response_ratio",
            "frozen_response_ratio",
        )
    }
    remaining = episodes
    while remaining:
        batch = min(batch_size, remaining)
        steps = sample_marker_steps(batch, device=device, held_out_final=True)
        state, stick_state = nominal_initial_state(
            batch,
            camera_height=steps.initial_marker_height,
            device=device,
            config=config,
            attitude_degrees=2.0,
            rate_degrees_per_second=5.0,
            vertical_speed=0.0,
        )
        scene = sample_visual_scenes(batch, device=device, held_out_combinations=True)
        neural = controller.initial_state(batch, device=device, dtype=torch.float32)
        state, stick_state, neural, prefix_valid = current_policy_prefix(
            controller,
            state,
            stick_state,
            neural,
            steps.initial_marker_height,
            scene,
            steps=prefix_steps,
            physics_steps=physics_steps,
            config=config,
        )
        frozen_image = render_visual_hover_scene(
            state, steps.initial_marker_height, config=config, scene=scene
        )
        live = _rollout_response(
            controller,
            clone_quad(state),
            clone_sticks(stick_state),
            neural.clone(),
            steps.final_marker_height,
            scene,
            steps=response_steps,
            physics_steps=physics_steps,
            config=config,
        )
        frozen = _rollout_response(
            controller,
            clone_quad(state),
            clone_sticks(stick_state),
            neural.clone(),
            steps.final_marker_height,
            scene,
            steps=response_steps,
            physics_steps=physics_steps,
            config=config,
            frozen_image=frozen_image,
        )
        control = _rollout_response(
            controller,
            clone_quad(state),
            clone_sticks(stick_state),
            neural.clone(),
            steps.initial_marker_height,
            scene,
            steps=response_steps,
            physics_steps=physics_steps,
            config=config,
        )
        final_window = min(round(2.0 * policy_hz), response_steps)
        error = live["height"][-final_window:] - steps.final_marker_height[None]
        rmse = torch.sqrt(error.square().mean(dim=0))
        vertical_rms = torch.sqrt(live["vertical_speed"][-final_window:].square().mean(dim=0))
        tilt_rms = torch.sqrt(live["tilt"][-final_window:].square().mean(dim=0))
        settling = _settling_time(
            live["height"], live["vertical_speed"], steps.final_marker_height, policy_hz
        )
        rc = live["controls"]
        saturated = (
            (rc[:, :, :3].abs() >= 0.98).any(dim=2) | (rc[:, :, 3] <= 0.01) | (rc[:, :, 3] >= 0.99)
        )
        sustained = _maximum_true_run(saturated) >= round(0.5 * policy_hz)
        requested_delta = 0.20 * steps.direction
        control_final = control["height"][-final_window:].mean(dim=0)
        live_delta = live["height"][-final_window:].mean(dim=0) - control_final
        frozen_delta = frozen["height"][-final_window:].mean(dim=0) - control_final
        collected["rmse"].append(rmse.cpu())
        collected["vertical_rms"].append(vertical_rms.cpu())
        collected["tilt_rms"].append(tilt_rms.cpu())
        collected["settling_time"].append(settling.cpu())
        collected["ground"].append((live["ground"] | ~prefix_valid).cpu())
        collected["invalid"].append((live["invalid"] | ~prefix_valid).cpu())
        collected["sustained_saturation"].append(sustained.cpu())
        collected["direction"].append(steps.direction.cpu())
        collected["live_response_ratio"].append((live_delta / requested_delta).cpu())
        collected["frozen_response_ratio"].append((frozen_delta / requested_delta).cpu())
        remaining -= batch

    values = {name: torch.cat(parts) for name, parts in collected.items()}
    success = (
        (values["rmse"] <= 0.15)
        & (values["vertical_rms"] <= 0.20)
        & (values["tilt_rms"] <= math.radians(5.0))
        & (values["settling_time"] <= 3.0)
        & ~values["ground"]
        & ~values["invalid"]
        & ~values["sustained_saturation"]
    )
    upward = values["direction"] > 0.0
    downward = ~upward
    success_rate = float(success.float().mean())
    upward_success = float(success[upward].float().mean())
    downward_success = float(success[downward].float().mean())
    requested = 0.20 * values["direction"]
    live_effect = values["live_response_ratio"] * requested
    frozen_effect = values["frozen_response_ratio"] * requested
    live_slope = (live_effect * requested).sum() / requested.square().sum()
    frozen_slope = (frozen_effect * requested).sum() / requested.square().sum()
    frozen_removed = abs(float(frozen_slope)) <= 0.25
    criteria_pass = (
        success_rate >= 0.90
        and upward_success >= 0.90
        and downward_success >= 0.90
        and not bool(values["ground"].any())
        and not bool(values["invalid"].any())
        and not bool(values["sustained_saturation"].any())
        and frozen_removed
    )
    return {
        "episodes": episodes,
        "success_rate": success_rate,
        "upward_success_rate": upward_success,
        "downward_success_rate": downward_success,
        "altitude_rmse_mean_m": float(values["rmse"].mean()),
        "altitude_rmse_p95_m": float(torch.quantile(values["rmse"], 0.95)),
        "vertical_speed_rms_mean_mps": float(values["vertical_rms"].mean()),
        "tilt_rms_mean_degrees": float(torch.rad2deg(values["tilt_rms"]).mean()),
        "settling_time_median_seconds": float(torch.median(values["settling_time"])),
        "ground_contact_rate": float(values["ground"].float().mean()),
        "invalid_flight_rate": float(values["invalid"].float().mean()),
        "sustained_saturation_rate": float(values["sustained_saturation"].float().mean()),
        "live_response_ratio_mean": float(values["live_response_ratio"].mean()),
        "live_paired_response_slope": float(live_slope),
        "frozen_response_ratio_absolute_mean": float(values["frozen_response_ratio"].abs().mean()),
        "frozen_paired_response_slope": float(frozen_slope),
        "frozen_step_tracking_removed": frozen_removed,
        "criteria_pass": criteria_pass,
        "promotion_eligible_episode_count": episodes >= 256,
        "pass": criteria_pass and episodes >= 256,
    }


@torch.no_grad()
def evaluate_attitude_recovery(
    controller: ConnectomeController,
    *,
    episodes: int,
    batch_size: int,
    response_steps: int,
    physics_steps: int,
    policy_hz: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    successes = []
    tilt_values = []
    height_values = []
    ground_values = []
    invalid_values = []
    remaining = episodes
    while remaining:
        batch = min(batch_size, remaining)
        conditions = sample_height_conditions(batch, device=device, split="held_out_marker")
        state, stick_state = nominal_initial_state(
            batch,
            camera_height=conditions.marker_height,
            device=device,
            config=config,
            attitude_degrees=12.0,
            rate_degrees_per_second=22.0,
            vertical_speed=0.10,
        )
        scene = sample_visual_scenes(batch, device=device, held_out_combinations=True)
        neural = controller.initial_state(batch, device=device, dtype=torch.float32)
        result = _rollout_response(
            controller,
            state,
            stick_state,
            neural,
            conditions.marker_height,
            scene,
            steps=response_steps,
            physics_steps=physics_steps,
            config=config,
        )
        final_window = min(round(1.0 * policy_hz), response_steps)
        tilt = torch.sqrt(result["tilt"][-final_window:].square().mean(dim=0))
        height_error = torch.sqrt(
            (result["height"][-final_window:] - conditions.marker_height[None]).square().mean(dim=0)
        )
        success = (
            (tilt <= math.radians(5.0))
            & (height_error <= 0.25)
            & ~result["ground"]
            & ~result["invalid"]
        )
        successes.append(success.cpu())
        tilt_values.append(tilt.cpu())
        height_values.append(height_error.cpu())
        ground_values.append(result["ground"].cpu())
        invalid_values.append(result["invalid"].cpu())
        remaining -= batch
    success = torch.cat(successes)
    tilt = torch.cat(tilt_values)
    height = torch.cat(height_values)
    ground = torch.cat(ground_values)
    invalid = torch.cat(invalid_values)
    return {
        "episodes": episodes,
        "success_rate": float(success.float().mean()),
        "tilt_rms_mean_degrees": float(torch.rad2deg(tilt).mean()),
        "altitude_rmse_mean_m": float(height.mean()),
        "ground_contact_rate": float(ground.float().mean()),
        "invalid_flight_rate": float(invalid.float().mean()),
        "pass": (
            float(success.float().mean()) >= 0.90
            and not bool(ground.any())
            and not bool(invalid.any())
        ),
    }


@torch.no_grad()
def evaluate_held_out_initial_combinations(
    controller: ConnectomeController,
    *,
    episodes: int,
    batch_size: int,
    response_steps: int,
    physics_steps: int,
    policy_hz: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    """Report the checkerboard reset-height split separately from marker promotion."""

    seed_everything(seed)
    successes = []
    rmse_values = []
    vertical_values = []
    tilt_values = []
    ground_values = []
    invalid_values = []
    remaining = episodes
    while remaining:
        batch = min(batch_size, remaining)
        conditions = sample_height_conditions(batch, device=device, split="held_out_combination")
        state, stick_state = nominal_initial_state(
            batch,
            camera_height=conditions.camera_height,
            device=device,
            config=config,
            attitude_degrees=2.0,
            rate_degrees_per_second=5.0,
            vertical_speed=0.05,
        )
        scene = sample_visual_scenes(batch, device=device, held_out_combinations=True)
        neural = controller.initial_state(batch, device=device, dtype=torch.float32)
        result = _rollout_response(
            controller,
            state,
            stick_state,
            neural,
            conditions.marker_height,
            scene,
            steps=response_steps,
            physics_steps=physics_steps,
            config=config,
        )
        final_window = min(round(2.0 * policy_hz), response_steps)
        rmse = torch.sqrt(
            (result["height"][-final_window:] - conditions.marker_height[None]).square().mean(dim=0)
        )
        vertical = torch.sqrt(result["vertical_speed"][-final_window:].square().mean(dim=0))
        tilt = torch.sqrt(result["tilt"][-final_window:].square().mean(dim=0))
        success = (
            (rmse <= 0.15)
            & (vertical <= 0.20)
            & (tilt <= math.radians(5.0))
            & ~result["ground"]
            & ~result["invalid"]
        )
        successes.append(success.cpu())
        rmse_values.append(rmse.cpu())
        vertical_values.append(vertical.cpu())
        tilt_values.append(tilt.cpu())
        ground_values.append(result["ground"].cpu())
        invalid_values.append(result["invalid"].cpu())
        remaining -= batch
    success = torch.cat(successes)
    rmse = torch.cat(rmse_values)
    vertical = torch.cat(vertical_values)
    tilt = torch.cat(tilt_values)
    ground = torch.cat(ground_values)
    invalid = torch.cat(invalid_values)
    return {
        "episodes": episodes,
        "split_scope": "held-out initial marker/camera family pairs",
        "success_rate": float(success.float().mean()),
        "altitude_rmse_mean_m": float(rmse.mean()),
        "vertical_speed_rms_mean_mps": float(vertical.mean()),
        "tilt_rms_mean_degrees": float(torch.rad2deg(tilt).mean()),
        "ground_contact_rate": float(ground.float().mean()),
        "invalid_flight_rate": float(invalid.float().mean()),
        "diagnostic_pass": (
            float(success.float().mean()) >= 0.90
            and not bool(ground.any())
            and not bool(invalid.any())
        ),
    }


@torch.no_grad()
def evaluate_genuine_pairs(
    controller: ConnectomeController,
    *,
    pairs_count: int,
    batch_size: int,
    response_steps: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
    half_step_metres: float = 0.05,
    fixed_wall_style: int | None = None,
    fixed_floor_style: int | None = None,
    held_out_markers: bool = True,
    held_out_scene: bool = True,
    all_style_combinations: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    predicted_values = []
    desired_values = []
    identical_values = []
    reversal_values = []
    valid_values = []
    remaining = pairs_count
    while remaining:
        batch = min(batch_size, remaining)
        pairs = sample_marker_pairs(
            batch,
            device=device,
            held_out=held_out_markers,
            half_step_metres=half_step_metres,
        )
        state, stick_state, scene = _paired_initial_state(
            pairs,
            device=device,
            config=config,
            held_out_scene=held_out_scene,
            fixed_wall_style=fixed_wall_style,
            fixed_floor_style=fixed_floor_style,
            all_style_combinations=all_style_combinations,
        )
        neural = controller.initial_state(batch, device=device, dtype=torch.float32)
        state, _, neural, prefix_valid = current_policy_prefix(
            controller,
            state,
            stick_state,
            neural,
            pairs.centre_height,
            scene,
            steps=50,
            physics_steps=physics_steps,
            config=config,
        )
        output_a, output_b = paired_predictions(
            controller,
            state,
            neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=response_steps,
            loss_start=max(0, response_steps - 10),
        )
        predicted = output_a[:, :, 3].mean(dim=0) - output_b[:, :, 3].mean(dim=0)
        target_a = teacher_motor(state, pairs.marker_a, config)
        target_b = teacher_motor(state, pairs.marker_b, config)
        desired = target_a[:, 3] - target_b[:, 3]
        identical_a, identical_b = paired_predictions(
            controller,
            state,
            neural,
            scene,
            pairs.marker_a,
            pairs.marker_a,
            response_steps=response_steps,
            loss_start=response_steps - 1,
        )
        swapped_a, swapped_b = paired_predictions(
            controller,
            state,
            neural,
            scene,
            pairs.marker_b,
            pairs.marker_a,
            response_steps=response_steps,
            loss_start=response_steps - 1,
        )
        original_final = output_a[-1, :, 3] - output_b[-1, :, 3]
        swapped_final = swapped_a[-1, :, 3] - swapped_b[-1, :, 3]
        predicted_values.append(predicted.cpu())
        desired_values.append(desired.cpu())
        identical_values.append((identical_a[-1, :, 3] - identical_b[-1, :, 3]).cpu())
        reversal_values.append((swapped_final + original_final).cpu())
        valid_values.append(prefix_valid.cpu())
        remaining -= batch
    predicted = torch.cat(predicted_values)
    desired = torch.cat(desired_values)
    identical = torch.cat(identical_values)
    reversal = torch.cat(reversal_values)
    prefix_valid = torch.cat(valid_values)
    scale = torch.sqrt(desired.square().mean()).clamp_min(0.01)
    nrmse = torch.sqrt((predicted - desired).square().mean()) / scale
    sign = (predicted * desired > 0.0).float().mean()
    slope = (predicted * desired).sum() / desired.square().sum().clamp_min(1.0e-8)
    passed = bool(
        nrmse <= 0.25
        and sign >= 0.95
        and 0.5 <= slope <= 1.5
        and identical.abs().max() <= 1.0e-6
        and reversal.abs().max() <= 1.0e-5
        and prefix_valid.all()
    )
    return {
        "pairs": pairs_count,
        "marker_difference_metres": 2.0 * half_step_metres,
        "wall_style_family": fixed_wall_style,
        "floor_style_family": fixed_floor_style,
        "all_absolute_marker_heights_held_out": held_out_markers,
        "contrast_nrmse": float(nrmse),
        "correct_sign_rate": float(sign),
        "response_slope": float(slope),
        "predicted_contrast_rms": float(torch.sqrt(predicted.square().mean())),
        "desired_contrast_rms": float(torch.sqrt(desired.square().mean())),
        "identical_image_contrast_max_absolute": float(identical.abs().max()),
        "swapped_image_reversal_error_max_absolute": float(reversal.abs().max()),
        "valid_dynamic_prefix_rate": float(prefix_valid.float().mean()),
        "pass": passed,
    }


def safe_against_baseline(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons = []
    if not candidate["legacy_pair"]["pass"]:
        reasons.append("legacy paired-marker retention failed")
    minimum_attitude = max(0.0, baseline["attitude_recovery"]["success_rate"] - 0.05)
    if candidate["attitude_recovery"]["success_rate"] < minimum_attitude:
        reasons.append("attitude-recovery success regressed by more than five points")
    if (
        candidate["attitude_recovery"]["tilt_rms_mean_degrees"]
        > baseline["attitude_recovery"]["tilt_rms_mean_degrees"] + 1.0
    ):
        reasons.append("attitude-recovery tilt regressed by more than one degree")
    return not reasons, reasons


def marker_score(evaluation: dict[str, Any]) -> float:
    marker = evaluation["marker_steps"]
    return (
        marker["altitude_rmse_mean_m"]
        + 0.5 * (1.0 - marker["success_rate"])
        + marker["ground_contact_rate"]
    )


@torch.no_grad()
def evaluate_suite(
    controller: ConnectomeController,
    *,
    episodes: int,
    args: argparse.Namespace,
    device: torch.device,
    config: HoverConfig,
    physics_steps: int,
    seed: int,
) -> dict[str, Any]:
    marker = evaluate_marker_steps(
        controller,
        episodes=episodes,
        batch_size=args.evaluation_batch_size,
        prefix_steps=round(args.prefix_seconds * args.policy_hz),
        response_steps=round(args.response_seconds * args.policy_hz),
        physics_steps=physics_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
        seed=seed,
    )
    pair_count = min(64, episodes)
    genuine_pair = evaluate_genuine_pairs(
        controller,
        pairs_count=pair_count,
        batch_size=args.evaluation_batch_size,
        response_steps=args.unroll,
        physics_steps=physics_steps,
        device=device,
        config=config,
        seed=seed + 1,
    )
    contrast_scale, _ = legacy_response.fixed_target_scales(config, 0.20, device)
    legacy_pair = legacy_response.evaluate_pairs(
        controller,
        pairs=pair_count,
        batch_size=args.evaluation_batch_size,
        response_steps=args.unroll,
        prefix_min_steps=5,
        prefix_max_steps=100,
        marker_step_metres=0.20,
        contrast_scale=contrast_scale,
        device=device,
        config=config,
        seed=seed + 2,
        policy_hz=args.policy_hz,
    )
    attitude = evaluate_attitude_recovery(
        controller,
        episodes=pair_count,
        batch_size=args.evaluation_batch_size,
        response_steps=round(4.0 * args.policy_hz),
        physics_steps=physics_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
        seed=seed + 3,
    )
    held_out_combination = evaluate_held_out_initial_combinations(
        controller,
        episodes=pair_count,
        batch_size=args.evaluation_batch_size,
        response_steps=round(args.response_seconds * args.policy_hz),
        physics_steps=physics_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
        seed=seed + 4,
    )
    return {
        "marker_steps": marker,
        "genuine_held_out_pair": genuine_pair,
        "legacy_pair": legacy_pair,
        "attitude_recovery": attitude,
        "held_out_initial_combination_diagnostic": held_out_combination,
    }


def checkpoint_payload(
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
    *,
    source_sha256: str,
    selected_update: int,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-hover-bounded-v1",
        "controller": controller.state_dict(),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "selected_update": selected_update,
        "policy_hz": args.policy_hz,
        "plant_model_version": PLANT_MODEL_VERSION,
        "hover_config": asdict(config),
        "camera": asdict(DEFAULT_VISUAL_CAMERA),
        "actor_inputs": {
            "rgb_camera": True,
            "estimated_roll_pitch": True,
            "accelerometer": False,
            "mass": False,
            "hover_thrust": False,
            "external_history": False,
        },
        "protocol": protocol_manifest(),
        "scene_protocol": visual_scene_manifest(),
    }


def main() -> int:
    args = parse_args()
    if not args.graph.is_file() or not args.checkpoint.is_file():
        raise SystemExit("the generated graph and paired-dynamic source checkpoint are required")
    if args.iterations < 1 and not args.evaluate_only:
        raise SystemExit("iterations must be positive")
    if args.unroll < 11:
        raise SystemExit("unroll must be at least 11 recurrent steps")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the 100 Hz physics rate")
    physics_steps = physics_hz // args.policy_hz
    seed_everything(args.seed)

    source_sha256 = file_sha256(args.checkpoint)
    source_checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if source_checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("source checkpoint graph hash does not match --graph")
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    controller.load_state_dict(source_checkpoint["controller"])
    source_parameters = {
        name: source_checkpoint["controller"][name].detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    evaluations = []
    started = perf_counter()
    evaluation_seed = args.seed + 10_000
    controller.eval()
    baseline = evaluate_suite(
        controller,
        episodes=args.interim_evaluation_episodes,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=evaluation_seed,
    )
    baseline_entry = {"update": 0, "evaluation": baseline}
    evaluations.append(baseline_entry)
    print(json.dumps(baseline_entry), flush=True)
    best_score = marker_score(baseline)
    best_update = 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in controller.state_dict().items()
    }
    stop_reason = None

    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
        )
        controller.train()
        for update in range(1, args.iterations + 1):
            lesson = LESSON_CYCLE[(update - 1) % len(LESSON_CYCLE)]
            stratum, prefix_steps = prefix_stratum(update - 1, args.policy_hz)
            if lesson == "paired_marker":
                metrics = paired_lesson(
                    controller,
                    optimizer,
                    source_parameters,
                    batch=args.batch_size,
                    unroll=args.unroll,
                    prefix_steps=prefix_steps,
                    physics_steps=physics_steps,
                    device=device,
                    config=config,
                )
            else:
                metrics = trajectory_lesson(
                    controller,
                    optimizer,
                    source_parameters,
                    batch=args.batch_size,
                    unroll=args.unroll,
                    prefix_steps=prefix_steps,
                    physics_steps=physics_steps,
                    attitude_recovery=lesson == "attitude_recovery",
                    device=device,
                    config=config,
                )
            entry = {"update": update, "lesson": lesson, "prefix_stratum": stratum, **metrics}
            if update % 10 == 0 or update == 1:
                history.append(entry)
                print(json.dumps(entry), flush=True)

            if update % args.evaluation_every == 0 or update == args.iterations:
                controller.eval()
                training_rng_state = capture_rng_state()
                evaluation = evaluate_suite(
                    controller,
                    episodes=args.interim_evaluation_episodes,
                    args=args,
                    device=device,
                    config=config,
                    physics_steps=physics_steps,
                    seed=evaluation_seed,
                )
                restore_rng_state(training_rng_state)
                safe, safety_reasons = safe_against_baseline(evaluation, baseline)
                score = marker_score(evaluation)
                improved = score < best_score - 0.005
                evaluation_entry = {
                    "update": update,
                    "marker_score": score,
                    "safe_against_baseline": safe,
                    "safety_reasons": safety_reasons,
                    "improved": improved,
                    "evaluation": evaluation,
                }
                evaluations.append(evaluation_entry)
                print(json.dumps(evaluation_entry), flush=True)
                torch.save(
                    checkpoint_payload(
                        controller,
                        args,
                        config,
                        source_sha256=source_sha256,
                        selected_update=update,
                    ),
                    args.output_dir / f"update-{update:04d}.pt",
                )
                if not safe:
                    stop_reason = "; ".join(safety_reasons)
                    break
                if not improved:
                    stop_reason = "marker score did not improve by at least 0.005"
                    break
                best_score = score
                best_update = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in controller.state_dict().items()
                }
                controller.train()

    controller.load_state_dict(best_state)
    controller.eval()
    final_evaluation = evaluate_suite(
        controller,
        episodes=args.final_evaluation_episodes,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=evaluation_seed + 50_000,
    )
    safe_final, final_safety_reasons = safe_against_baseline(final_evaluation, baseline)
    passed = bool(
        final_evaluation["marker_steps"]["pass"]
        and final_evaluation["genuine_held_out_pair"]["pass"]
        and final_evaluation["legacy_pair"]["pass"]
        and final_evaluation["attitude_recovery"]["pass"]
        and safe_final
    )
    selected_path = args.output_dir / "selected-controller.pt"
    torch.save(
        checkpoint_payload(
            controller,
            args,
            config,
            source_sha256=source_sha256,
            selected_update=best_update,
        ),
        selected_path,
    )
    report = {
        "experiment": "variable-height-hover-bounded-v1",
        "passed": passed,
        "promoted": passed,
        "selected_update": best_update,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "lesson_cycle": list(LESSON_CYCLE),
        "protocol": protocol_manifest(),
        "scene_protocol": visual_scene_manifest(),
        "credit_assignment": (
            "current-policy student actions create detached next observations; gradients flow "
            "only through native neural recurrence inside each bounded window"
        ),
        "actor_inputs": checkpoint_payload(
            controller,
            args,
            config,
            source_sha256=source_sha256,
            selected_update=best_update,
        )["actor_inputs"],
        "iterations_requested": args.iterations,
        "updates_completed": evaluations[-1]["update"],
        "stop_reason": stop_reason,
        "elapsed_seconds": perf_counter() - started,
        "baseline": baseline,
        "history": history,
        "evaluations": evaluations,
        "final_evaluation": final_evaluation,
        "final_safe_against_baseline": safe_final,
        "final_safety_reasons": final_safety_reasons,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
