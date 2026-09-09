#!/usr/bin/env python3
"""Train and evaluate the first connectome-to-foreleg-to-quad hover loop."""

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

from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
    render_target_band,
    teacher_rc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data" / "derived" / "hover-connectome-v1.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "runs" / "hover" / "latest"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--pretrain-iterations", type=int, default=400)
    parser.add_argument("--pretrain-unroll", type=int, default=50)
    parser.add_argument("--sequence-iterations", type=int, default=80)
    parser.add_argument("--sequence-steps", type=int, default=600)
    parser.add_argument("--sequence-batch-size", type=int, default=16)
    parser.add_argument("--airborne-iterations", type=int, default=180)
    parser.add_argument("--takeoff-iterations", type=int, default=240)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--pretrain-learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--max-roll-pitch-rate-degrees", type=float, default=160.0)
    parser.add_argument("--thrust-to-weight", type=float, default=1.5)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seconds", type=float, default=20.0)
    parser.add_argument("--checkpoint", type=Path, help="Resume/evaluate this checkpoint.")
    parser.add_argument("--evaluate-only", action="store_true")
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


def checkpoint_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-safe training arguments embedded with every checkpoint."""

    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def restored_hover_config(
    checkpoint: dict[str, Any], requested: HoverConfig
) -> HoverConfig:
    """Restore and validate the physical configuration saved with an actor."""

    saved = checkpoint.get("hover_config")
    if not isinstance(saved, dict):
        raise SystemExit("checkpoint has no hover_config; frozen replay is not possible")
    known = set(HoverConfig.__dataclass_fields__)
    unknown = set(saved) - known
    if unknown:
        raise SystemExit(f"checkpoint has unknown hover_config fields: {sorted(unknown)}")
    restored = HoverConfig(**saved)
    if restored != requested:
        print("restoring physical configuration from checkpoint", flush=True)
    return restored


def random_training_state(
    batch: int, device: torch.device, config: HoverConfig
) -> tuple[QuadState, torch.Tensor]:
    target = torch.empty(batch, device=device).uniform_(0.8, 1.2)
    position = torch.zeros(batch, 3, device=device)
    position[:, 2] = torch.empty(batch, device=device).uniform_(0.0, 1.5)
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(18.0), math.radians(18.0)
    )
    quad = DifferentiableQuad(config).to(device)
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    return state, target


def controller_regularization(controller: ConnectomeController) -> torch.Tensor:
    return (
        2.0e-4
        * (controller.edge_magnitude - controller.initial_edge_magnitude).square().mean()
        + 5.0e-5 * controller.bias.square().mean()
        + 2.0e-5
        * (
            controller.raw_time_constant - controller.initial_raw_time_constant
        ).square().mean()
    )


def motor_imitation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # A few degrees of unintended roll/pitch quickly dominate long-horizon flight;
    # throttle has much more benign plant gain.  These weights are training-only.
    axis_weight = prediction.new_tensor((4.0, 4.0, 2.0, 4.0))
    return ((prediction - target).square() * axis_weight).mean()


def pretrain_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    unroll: int,
    resolution: int,
    device: torch.device,
    config: HoverConfig,
) -> float:
    state, target = random_training_state(batch, device, config)
    image = render_target_band(state, target, resolution=resolution, config=config)
    desired_motor = motor_target_for_rc(teacher_rc(state, target, config), config)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    losses = []
    for step in range(unroll):
        motor, neural = controller(image, state.euler[:, :2], neural)
        if step >= unroll // 2:
            losses.append(motor_imitation_loss(motor, desired_motor))
    predictions = torch.stack(losses)
    direct_loss = predictions.mean()
    # A constant mean throttle is a deceptively good minimizer for the small initial
    # task.  Explicitly weight across-sample variation so retinal and attitude pathways
    # must explain why two observations require different motor-pool activity.
    centered_motor = motor - motor.mean(dim=0, keepdim=True)
    centered_target = desired_motor - desired_motor.mean(dim=0, keepdim=True)
    contrastive_loss = motor_imitation_loss(centered_motor, centered_target)
    loss = (
        8.0 * direct_loss
        + 40.0 * contrastive_loss
        + controller_regularization(controller)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 2.0)
    optimizer.step()
    controller.project_parameters()
    return float(loss.detach())


def reset_rollout(
    batch: int,
    device: torch.device,
    config: HoverConfig,
    *,
    airborne: bool,
) -> tuple[QuadState, torch.Tensor, torch.Tensor]:
    target = torch.empty(batch, device=device).uniform_(0.8, 1.2)
    position = torch.zeros(batch, 3, device=device)
    if airborne:
        position[:, 2] = target + torch.empty(batch, device=device).uniform_(-0.35, 0.35)
        position[:, 2].clamp_(0.25, 1.5)
    euler = torch.zeros(batch, 3, device=device)
    euler[:, :2] = torch.empty(batch, 2, device=device).uniform_(
        -math.radians(8.0), math.radians(8.0)
    )
    quad = DifferentiableQuad(config).to(device)
    state = quad.initial_state(
        batch, device=device, dtype=torch.float32, position=position, euler=euler
    )
    mass_scale = torch.empty(batch, device=device).uniform_(0.92, 1.08)
    return state, target, mass_scale


def sequence_imitation_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    steps: int,
    resolution: int,
    device: torch.device,
    config: HoverConfig,
    airborne: bool,
) -> dict[str, float]:
    """Clone teacher-flown temporal observations into recurrent connectome state."""

    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    teacher_state, target, mass_scale = reset_rollout(
        batch, device, config, airborne=airborne
    )
    if airborne:
        teacher_state.velocity[:, 2] = torch.empty(batch, device=device).uniform_(-0.8, 0.8)
    teacher_sticks = sticks.initial_state(batch, device=device, dtype=torch.float32)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    losses = []
    final_prediction = None
    final_target = None
    for _ in range(steps):
        image = render_target_band(
            teacher_state, target, resolution=resolution, config=config
        )
        prediction, neural = controller(image, teacher_state.euler[:, :2], neural)
        desired_rc = teacher_rc(teacher_state, target, config)
        desired_motor = motor_target_for_rc(desired_rc, config)
        losses.append(motor_imitation_loss(prediction, desired_motor))
        teacher_rc_measured, teacher_sticks = sticks(desired_motor, teacher_sticks)
        teacher_state = quad(teacher_rc_measured, teacher_state, mass_scale)
        final_prediction, final_target = prediction, desired_motor
    assert final_prediction is not None and final_target is not None
    direct_loss = torch.stack(losses).mean()
    centered_prediction = final_prediction - final_prediction.mean(dim=0, keepdim=True)
    centered_target = final_target - final_target.mean(dim=0, keepdim=True)
    contrastive_loss = motor_imitation_loss(centered_prediction, centered_target)
    loss = 8.0 * direct_loss + 10.0 * contrastive_loss + controller_regularization(controller)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
    optimizer.step()
    controller.project_parameters()
    return {
        "loss": float(loss.detach()),
        "direct_imitation": float(direct_loss.detach()),
        "contrastive_imitation": float(contrastive_loss.detach()),
        "teacher_final_height_error": float(
            (teacher_state.position[:, 2] - target).abs().mean().detach()
        ),
    }


def rollout_training_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    batch: int,
    steps: int,
    resolution: int,
    device: torch.device,
    config: HoverConfig,
    airborne: bool,
) -> dict[str, float]:
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    quad_state, target, mass_scale = reset_rollout(
        batch, device, config, airborne=airborne
    )
    stick_state = sticks.initial_state(batch, device=device, dtype=torch.float32)
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    imitation_losses = []
    task_losses = []
    for step in range(steps):
        image = render_target_band(quad_state, target, resolution=resolution, config=config)
        motor, neural = controller(image, quad_state.euler[:, :2], neural)
        rc, stick_state = sticks(motor, stick_state)
        desired_motor = motor_target_for_rc(teacher_rc(quad_state, target, config), config)
        imitation_losses.append(motor_imitation_loss(motor, desired_motor))
        quad_state = quad(rc, quad_state, mass_scale)
        if step >= min(50, steps // 4):
            height_error = quad_state.position[:, 2] - target
            tilt = quad_state.euler[:, :2]
            task_losses.append(
                height_error.square().mean()
                + 0.12 * quad_state.velocity[:, 2].square().mean()
                + 20.0 * tilt.square().mean()
                + 0.10 * quad_state.rates[:, :2].square().mean()
                + 0.03 * quad_state.position[:, :2].square().mean()
            )
    imitation = torch.stack(imitation_losses).mean()
    task = torch.stack(task_losses).mean()
    loss = 8.0 * imitation + task + controller_regularization(controller)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
    optimizer.step()
    controller.project_parameters()
    return {
        "loss": float(loss.detach()),
        "imitation": float(imitation.detach()),
        "task": float(task.detach()),
        "height_error": float((quad_state.position[:, 2] - target).abs().mean().detach()),
        "tilt_degrees": float(torch.rad2deg(quad_state.euler[:, :2]).abs().mean().detach()),
    }


@torch.no_grad()
def summarize_evaluation(
    *,
    heights: list[torch.Tensor],
    tilts: list[torch.Tensor],
    drifts: list[torch.Tensor],
    stick_positions: list[torch.Tensor],
    target: torch.Tensor,
    lifted: torch.Tensor,
    recontact: torch.Tensor,
    seconds: float,
    config: HoverConfig,
    label: str,
) -> dict[str, Any]:
    episodes = target.shape[0]
    height = torch.stack(heights)
    tilt = torch.stack(tilts)
    drift = torch.stack(drifts)
    stick = torch.stack(stick_positions)
    error = height - target[None, :]
    step_count = height.shape[0]
    final_steps = min(round(10.0 / config.dt), step_count)
    final_error = error[-final_steps:]
    final_tilt = tilt[-final_steps:]
    in_band = error.abs() <= 0.15
    first_band = torch.full((episodes,), float("inf"), device=target.device)
    time_axis = torch.arange(1, step_count + 1, device=target.device) * config.dt
    for episode in range(episodes):
        indices = torch.nonzero(in_band[:, episode], as_tuple=False).flatten()
        if len(indices):
            first_band[episode] = time_axis[indices[0]]
    altitude_rmse = torch.sqrt(final_error.square().mean(dim=0))
    tilt_rms = torch.sqrt(final_tilt.square().mean(dim=0))
    tilt_p95 = torch.quantile(final_tilt, 0.95, dim=0)
    max_drift = drift.max(dim=0).values
    sustained_saturation = (stick.abs() > 0.98).float().mean(dim=0).max(dim=1).values > 0.25
    goal_success = (
        lifted
        & (first_band <= 6.0)
        & (altitude_rmse <= 0.30)
        & (tilt_rms <= math.radians(5.0))
        & (tilt_p95 <= math.radians(10.0))
        & ~recontact
        & ~sustained_saturation
    )
    research_success = goal_success & (altitude_rmse <= 0.10) & (max_drift <= 0.75)
    settled_height = height[-final_steps:].mean(dim=0)
    target_centered = target - target.mean()
    height_centered = settled_height - settled_height.mean()
    tracking_slope = (target_centered * height_centered).sum() / target_centered.square().sum()
    tracking_correlation = (target_centered * height_centered).sum() / torch.sqrt(
        target_centered.square().sum() * height_centered.square().sum()
    )
    goal_success_rate = float(goal_success.float().mean())
    research_success_rate = float(research_success.float().mean())
    return {
        "episodes": episodes,
        "seconds": seconds,
        "label": label,
        "goal_success_rate": goal_success_rate,
        "research_success_rate": research_success_rate,
        "lift_off_rate": float(lifted.float().mean()),
        "entered_band_by_6s_rate": float((first_band <= 6.0).float().mean()),
        "altitude_rmse_final_10s_mean_m": float(altitude_rmse.mean()),
        "altitude_rmse_final_10s_p95_m": float(torch.quantile(altitude_rmse, 0.95)),
        "tilt_rms_final_10s_mean_deg": float(torch.rad2deg(tilt_rms).mean()),
        "tilt_p95_final_10s_mean_deg": float(torch.rad2deg(tilt_p95).mean()),
        "max_horizontal_drift_mean_m": float(max_drift.mean()),
        "ground_recontact_rate": float(recontact.float().mean()),
        "sustained_stick_saturation_rate": float(sustained_saturation.float().mean()),
        "final_height_mean_m": float(height[-1].mean()),
        "settled_height_bias_mean_m": float((settled_height - target).mean()),
        "target_tracking_slope": float(tracking_slope),
        "target_tracking_correlation": float(tracking_correlation),
        "target_height_mean_m": float(target.mean()),
        "goal_pass": goal_success_rate >= 0.90,
        "research_pass": research_success_rate >= 0.90,
    }


def initial_evaluation_state(
    episodes: int, device: torch.device, config: HoverConfig, seed: int
) -> tuple[QuadState, torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    quad = DifferentiableQuad(config).to(device)
    target = torch.empty(episodes, device=device).uniform_(0.8, 1.2)
    position = torch.zeros(episodes, 3, device=device)
    euler = torch.zeros(episodes, 3, device=device)
    euler[:, :2] = torch.empty(episodes, 2, device=device).uniform_(
        -math.radians(5.0), math.radians(5.0)
    )
    state = quad.initial_state(
        episodes, device=device, dtype=torch.float32, position=position, euler=euler
    )
    mass_scale = torch.empty(episodes, device=device).uniform_(0.9, 1.1)
    return state, target, mass_scale


@torch.no_grad()
def evaluate(
    controller: ConnectomeController,
    *,
    episodes: int,
    seconds: float,
    resolution: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
    frozen_visual: bool = False,
) -> dict[str, Any]:
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    state, target, mass_scale = initial_evaluation_state(episodes, device, config, seed)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    step_count = round(seconds / config.dt)
    heights = []
    tilts = []
    drifts = []
    stick_positions = []
    frozen_image = render_target_band(state, target, resolution=resolution, config=config)
    lifted = torch.zeros(episodes, dtype=torch.bool, device=device)
    recontact = torch.zeros_like(lifted)
    for _ in range(step_count):
        image = frozen_image if frozen_visual else render_target_band(
            state, target, resolution=resolution, config=config
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
        lifted |= state.position[:, 2] > 0.10
        recontact |= lifted & (state.position[:, 2] <= 0.005)
        heights.append(state.position[:, 2])
        tilts.append(torch.linalg.vector_norm(state.euler[:, :2], dim=1))
        drifts.append(torch.linalg.vector_norm(state.position[:, :2], dim=1))
        stick_positions.append(stick_state.position)

    return summarize_evaluation(
        heights=heights,
        tilts=tilts,
        drifts=drifts,
        stick_positions=stick_positions,
        target=target,
        lifted=lifted,
        recontact=recontact,
        seconds=seconds,
        config=config,
        label="frozen_visual" if frozen_visual else "connectome",
    )


@torch.no_grad()
def evaluate_teacher(
    *,
    episodes: int,
    seconds: float,
    device: torch.device,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    """Prove the same foreleg/stick/quad plant is solvable before judging learning."""

    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    state, target, mass_scale = initial_evaluation_state(episodes, device, config, seed)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    step_count = round(seconds / config.dt)
    heights = []
    tilts = []
    drifts = []
    stick_positions = []
    lifted = torch.zeros(episodes, dtype=torch.bool, device=device)
    recontact = torch.zeros_like(lifted)
    for _ in range(step_count):
        desired_rc = teacher_rc(state, target, config)
        motor = motor_target_for_rc(desired_rc, config)
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
        lifted |= state.position[:, 2] > 0.10
        recontact |= lifted & (state.position[:, 2] <= 0.005)
        heights.append(state.position[:, 2])
        tilts.append(torch.linalg.vector_norm(state.euler[:, :2], dim=1))
        drifts.append(torch.linalg.vector_norm(state.position[:, :2], dim=1))
        stick_positions.append(stick_state.position)
    return summarize_evaluation(
        heights=heights,
        tilts=tilts,
        drifts=drifts,
        stick_positions=stick_positions,
        target=target,
        lifted=lifted,
        recontact=recontact,
        seconds=seconds,
        config=config,
        label="privileged_teacher_through_foreleg_sticks",
    )


@torch.no_grad()
def evaluate_target_step(
    controller: ConnectomeController,
    *,
    episodes: int,
    seconds: float,
    resolution: int,
    device: torch.device,
    config: HoverConfig,
    seed: int,
) -> dict[str, Any]:
    """Compare a band step with a paired no-step continuation from the same state."""

    seed_everything(seed)
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    episodes = max(2, episodes + episodes % 2)
    initial_target = torch.where(
        torch.arange(episodes, device=device) % 2 == 0,
        torch.full((episodes,), 0.8, device=device),
        torch.full((episodes,), 1.2, device=device),
    )
    final_target = 2.0 - initial_target
    position = torch.zeros(episodes, 3, device=device)
    euler = torch.zeros(episodes, 3, device=device)
    euler[:, :2] = torch.empty(episodes, 2, device=device).uniform_(
        -math.radians(5.0), math.radians(5.0)
    )
    state = quad.initial_state(
        episodes, device=device, dtype=torch.float32, position=position, euler=euler
    )
    mass_scale = torch.empty(episodes, device=device).uniform_(0.9, 1.1)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    step_count = round(seconds / config.dt)
    switch_step = step_count // 2
    pre_switch_heights = []
    for _ in range(switch_step):
        image = render_target_band(
            state, initial_target, resolution=resolution, config=config
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)
        pre_switch_heights.append(state.position[:, 2])

    # Both branches begin from the exact same quad, joint, and recurrent neural state.
    stepped_state, control_state = state, state
    stepped_sticks, control_sticks = stick_state, stick_state
    stepped_neural, control_neural = neural, neural
    stepped_heights = []
    control_heights = []
    for _ in range(step_count - switch_step):
        stepped_image = render_target_band(
            stepped_state, final_target, resolution=resolution, config=config
        )
        stepped_motor, stepped_neural = controller(
            stepped_image, stepped_state.euler[:, :2], stepped_neural
        )
        stepped_rc, stepped_sticks = sticks(stepped_motor, stepped_sticks)
        stepped_state = quad(stepped_rc, stepped_state, mass_scale)
        stepped_heights.append(stepped_state.position[:, 2])

        control_image = render_target_band(
            control_state, initial_target, resolution=resolution, config=config
        )
        control_motor, control_neural = controller(
            control_image, control_state.euler[:, :2], control_neural
        )
        control_rc, control_sticks = sticks(control_motor, control_sticks)
        control_state = quad(control_rc, control_state, mass_scale)
        control_heights.append(control_state.position[:, 2])

    before_height = torch.stack(pre_switch_heights)
    stepped_height = torch.stack(stepped_heights)
    control_height = torch.stack(control_heights)
    window = max(1, round(2.0 / config.dt))
    before = before_height[-window:].mean(dim=0)
    stepped_after = stepped_height[-window:].mean(dim=0)
    control_after = control_height[-window:].mean(dim=0)
    target_delta = final_target - initial_target
    observed_height_delta = stepped_after - before
    paired_effect = stepped_after - control_after
    response_ratio = paired_effect / target_delta
    direction_correct = paired_effect * target_delta > 0.0
    return {
        "episodes": episodes,
        "seconds": seconds,
        "switch_seconds": switch_step * config.dt,
        "actor_state_reset_at_switch": False,
        "scalar_target_given_to_actor": False,
        "paired_no_step_counterfactual": True,
        "direction_correct_rate": float(direction_correct.float().mean()),
        "response_ratio_mean": float(response_ratio.mean()),
        "response_ratio_p10": float(torch.quantile(response_ratio, 0.10)),
        "response_ratio_p90": float(torch.quantile(response_ratio, 0.90)),
        "observed_height_change_mean_m": float(observed_height_delta.mean()),
        "paired_height_effect_mean_m": float(paired_effect.mean()),
        "mean_absolute_final_target_error_m": float(
            (stepped_after - final_target).abs().mean()
        ),
        "pass": bool(
            float(direction_correct.float().mean()) >= 0.90
            and 0.5 <= float(response_ratio.mean()) <= 1.5
        ),
    }


def save_checkpoint(
    path: Path,
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
    inherited_training_metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    training_arguments = checkpoint_arguments(args)
    training_seed = args.seed
    training_resolution = args.resolution
    training_plant_model_version = PLANT_MODEL_VERSION
    if inherited_training_metadata is not None:
        training_arguments = inherited_training_metadata.get(
            "training_arguments", training_arguments
        )
        training_seed = inherited_training_metadata.get("seed", training_seed)
        training_resolution = inherited_training_metadata.get(
            "image_resolution", training_resolution
        )
        training_plant_model_version = inherited_training_metadata.get(
            "training_plant_model_version",
            inherited_training_metadata.get("plant_model_version", "legacy-unspecified"),
        )
    torch.save(
        {
            "checkpoint_schema_version": 2,
            "plant_model_version": PLANT_MODEL_VERSION,
            "training_plant_model_version": training_plant_model_version,
            "controller": controller.state_dict(),
            "graph_sha256": file_sha256(args.graph),
            "graph_path": str(args.graph),
            "hover_config": asdict(config),
            "seed": training_seed,
            "image_resolution": training_resolution,
            "training_arguments": training_arguments,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.evaluate_only and args.checkpoint is None:
        raise SystemExit("--evaluate-only requires --checkpoint for frozen replay")
    if not args.graph.is_file():
        raise SystemExit(
            f"derived graph missing: run scripts/build_hover_connectome.py ({args.graph})"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; use scripts/run_hover_training.sh under WSL2"
        )
    seed_everything(args.seed)
    requested_config = HoverConfig(
        max_roll_pitch_rate=math.radians(args.max_roll_pitch_rate_degrees),
        thrust_to_weight=args.thrust_to_weight,
    )
    config = requested_config
    checkpoint: dict[str, Any] | None = None
    loaded_checkpoint_sha256: str | None = None
    source_report_provenance: dict[str, Any] | None = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        loaded_checkpoint_sha256 = file_sha256(args.checkpoint)
        if checkpoint["graph_sha256"] != file_sha256(args.graph):
            raise SystemExit("checkpoint graph hash does not match --graph")
        config = restored_hover_config(checkpoint, requested_config)
        saved_resolution = checkpoint.get("image_resolution")
        if (
            args.evaluate_only
            and saved_resolution is not None
            and saved_resolution != args.resolution
        ):
            raise SystemExit(
                "--resolution does not match the checkpoint image_resolution "
                f"({args.resolution} != {saved_resolution})"
            )
        saved_plant_model = checkpoint.get("plant_model_version")
        if (
            args.evaluate_only
            and saved_plant_model is not None
            and saved_plant_model != PLANT_MODEL_VERSION
        ):
            raise SystemExit(
                "checkpoint plant_model_version does not match this evaluator "
                f"({saved_plant_model!r} != {PLANT_MODEL_VERSION!r})"
            )
        source_report_path = args.checkpoint.with_name("report.json")
        if source_report_path.is_file():
            source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
            source_report_provenance = {
                "path": str(source_report_path),
                "sha256": file_sha256(source_report_path),
                "plant_model_version": source_report.get(
                    "plant_model_version", "legacy-unspecified"
                ),
                "training": source_report.get("training"),
            }
    controller = ConnectomeController(args.graph, neural_dt=config.dt).to(device)
    if checkpoint is not None:
        incompatible = controller.load_state_dict(checkpoint["controller"], strict=False)
        allowed_missing = {"initial_raw_time_constant"}
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise SystemExit(
                "checkpoint is incompatible: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        if "initial_raw_time_constant" in incompatible.missing_keys:
            print("using the current regularization prior for this legacy checkpoint", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "controller.pt"
    started = perf_counter()
    history: list[dict[str, Any]] = []
    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.pretrain_learning_rate, weight_decay=1.0e-5
        )
        for iteration in range(args.pretrain_iterations):
            loss = pretrain_step(
                controller,
                optimizer,
                batch=args.batch_size,
                unroll=args.pretrain_unroll,
                resolution=args.resolution,
                device=device,
                config=config,
            )
            if iteration % 20 == 0 or iteration + 1 == args.pretrain_iterations:
                entry = {"stage": "pretrain", "iteration": iteration + 1, "loss": loss}
                history.append(entry)
                print(json.dumps(entry), flush=True)

        optimizer = torch.optim.AdamW(
            controller.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
        )
        best_sequence_loss = float("inf")
        sequence_best_path = args.output_dir / "sequence-best.pt"
        for iteration in range(args.sequence_iterations):
            metrics = sequence_imitation_step(
                controller,
                optimizer,
                batch=args.sequence_batch_size,
                steps=args.sequence_steps,
                resolution=args.resolution,
                device=device,
                config=config,
                airborne=iteration % 2 == 1,
            )
            if metrics["loss"] < best_sequence_loss:
                best_sequence_loss = metrics["loss"]
                save_checkpoint(sequence_best_path, controller, args, config)
            if iteration % 10 == 0 or iteration + 1 == args.sequence_iterations:
                entry = {
                    "stage": "sequence_imitation",
                    "iteration": iteration + 1,
                    "best_loss": best_sequence_loss,
                    **metrics,
                }
                history.append(entry)
                print(json.dumps(entry), flush=True)
        if args.sequence_iterations:
            sequence_checkpoint = torch.load(
                sequence_best_path, map_location=device, weights_only=True
            )
            controller.load_state_dict(sequence_checkpoint["controller"])
            save_checkpoint(checkpoint_path, controller, args, config)

        for stage, iterations, airborne in (
            ("airborne", args.airborne_iterations, True),
            ("takeoff", args.takeoff_iterations, False),
        ):
            optimizer = torch.optim.AdamW(
                controller.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
            )
            best_score = float("inf")
            best_path = args.output_dir / f"{stage}-best.pt"
            for iteration in range(iterations):
                metrics = rollout_training_step(
                    controller,
                    optimizer,
                    batch=args.batch_size,
                    steps=args.rollout_steps,
                    resolution=args.resolution,
                    device=device,
                    config=config,
                    airborne=airborne,
                )
                score = (
                    metrics["height_error"]
                    + metrics["tilt_degrees"] / 30.0
                    + metrics["imitation"]
                )
                if score < best_score:
                    best_score = score
                    save_checkpoint(best_path, controller, args, config)
                if iteration % 10 == 0 or iteration + 1 == iterations:
                    entry = {
                        "stage": stage,
                        "iteration": iteration + 1,
                        "selection_score": score,
                        "best_selection_score": best_score,
                        **metrics,
                    }
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            if iterations:
                best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
                controller.load_state_dict(best_checkpoint["controller"])
            save_checkpoint(checkpoint_path, controller, args, config)

    controller.eval()
    teacher = evaluate_teacher(
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        device=device,
        config=config,
        seed=args.seed + 10_000,
    )
    nominal = evaluate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        config=config,
        seed=args.seed + 10_000,
    )
    frozen = evaluate(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        config=config,
        seed=args.seed + 10_000,
        frozen_visual=True,
    )
    target_step = evaluate_target_step(
        controller,
        episodes=args.evaluation_episodes,
        seconds=args.evaluation_seconds,
        resolution=args.resolution,
        device=device,
        config=config,
        seed=args.seed + 20_000,
    )
    save_checkpoint(
        checkpoint_path,
        controller,
        args,
        config,
        inherited_training_metadata=checkpoint if args.evaluate_only else None,
    )
    checkpoint_sha256 = file_sha256(checkpoint_path)
    report = {
        "passed": nominal["goal_pass"] and target_step["pass"],
        "plant_model_version": PLANT_MODEL_VERSION,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "loaded_checkpoint_sha256": loaded_checkpoint_sha256,
        "connectome_nodes": controller.n_nodes,
        "connectome_edges": int(controller.edge_pre.numel()),
        "config": asdict(config),
        "evaluation_protocol": {
            "image_resolution": [args.resolution, args.resolution],
            "episodes": args.evaluation_episodes,
            "seconds": args.evaluation_seconds,
            "nominal_and_ablation_seed": args.seed + 10_000,
            "target_step_seed": args.seed + 20_000,
        },
        "training": {
            "performed_this_run": not args.evaluate_only,
            "source_checkpoint_metadata": {
                key: checkpoint.get(key)
                for key in (
                    "checkpoint_schema_version",
                    "plant_model_version",
                    "training_plant_model_version",
                    "seed",
                    "image_resolution",
                    "training_arguments",
                )
                if checkpoint is not None and key in checkpoint
            },
            "source_report_provenance": source_report_provenance,
            "pretrain_iterations": args.pretrain_iterations if not args.evaluate_only else 0,
            "sequence_iterations": args.sequence_iterations if not args.evaluate_only else 0,
            "sequence_steps": args.sequence_steps,
            "sequence_batch_size": args.sequence_batch_size,
            "airborne_iterations": args.airborne_iterations if not args.evaluate_only else 0,
            "takeoff_iterations": args.takeoff_iterations if not args.evaluate_only else 0,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "elapsed_seconds": perf_counter() - started,
            "history": history,
        },
        "evaluation": nominal,
        "teacher_baseline": teacher,
        "frozen_visual_ablation": frozen,
        "visual_target_step": target_step,
        "commands_from_measured_sticks_only": True,
        "external_actor_state_machine": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"wrote {checkpoint_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
