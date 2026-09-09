#!/usr/bin/env python3
"""Distill a verified native mass oracle into the full recurrent connectome."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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

from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
    sample_matched_cases,
)
from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    load_controller,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_acceleration_oracle_distillation import delta_metrics  # noqa: E402
from train_gate_recurrent_ppo import (  # noqa: E402
    final_success,
    initialize_outcomes,
    outcome_summary,
    update_outcomes,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


@dataclass
class Trajectories:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    stick_position: Tensor
    oracle_motor: Tensor
    reference_motor: Tensor
    valid: Tensor
    mass_scale: Tensor
    codes: Tensor
    collection_policy_sha256: str
    teacher_drives_physics: bool
    summary: dict[str, Any]


@dataclass
class SelectedWindow:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    stick_position: Tensor
    oracle_motor: Tensor
    reference_motor: Tensor
    valid: Tensor
    mass_scale: Tensor
    student_episode: Tensor
    loss_start: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--teacher-checkpoint",
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
        default=REPO_ROOT / "runs" / "gate" / "full-network-oracle-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--updates-per-round", type=int, default=20)
    parser.add_argument("--collection-episodes", type=int, default=64)
    parser.add_argument("--clean-episodes", type=int, default=64)
    parser.add_argument("--holdout-episodes", type=int, default=128)
    parser.add_argument("--trajectory-seconds", type=float, default=8.0)
    parser.add_argument("--window-steps", type=int, default=50)
    parser.add_argument("--window-pairs", type=int, default=8)
    parser.add_argument("--student-window-fraction", type=float, default=0.75)
    parser.add_argument("--early-window-end-seconds", type=float, default=1.5)
    parser.add_argument("--late-window-start-seconds", type=float, default=1.5)
    parser.add_argument("--late-window-end-seconds", type=float, default=5.0)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--edge-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--bias-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--pair-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument("--regularization-weight", type=float, default=1.0e-5)
    parser.add_argument("--selection-interval", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=128)
    parser.add_argument("--validation-seconds", type=float, default=8.0)
    parser.add_argument("--checkpoint-update", type=int, default=100)
    parser.add_argument("--checkpoint-light-improvement", type=float, default=0.03)
    parser.add_argument("--checkpoint-heavy-margin", type=float, default=0.03)
    parser.add_argument("--teacher-audit-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--final-seconds", type=float, default=12.0)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--final-heavy-margin", type=float, default=0.02)
    parser.add_argument("--collection-seed", type=int, default=900_031)
    parser.add_argument("--clean-seed", type=int, default=910_031)
    parser.add_argument("--holdout-seed", type=int, default=920_031)
    parser.add_argument("--validation-seed", type=int, default=930_031)
    parser.add_argument("--teacher-audit-seed", type=int, default=940_031)
    parser.add_argument("--final-seed", type=int, default=950_031)
    parser.add_argument("--optimization-seed", type=int, default=96_031)
    parser.add_argument(
        "--gradient-check-scales", type=float, nargs=3, default=(1.0e-3, 3.0e-4, 1.0e-4)
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (
        args.graph,
        args.student_checkpoint,
        args.teacher_checkpoint,
        args.calibration,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.rounds,
        args.updates_per_round,
        args.collection_episodes,
        args.clean_episodes,
        args.holdout_episodes,
        args.trajectory_seconds,
        args.window_steps,
        args.window_pairs,
        args.student_window_fraction,
        args.early_window_end_seconds,
        args.late_window_start_seconds,
        args.late_window_end_seconds,
        args.target_onset_seconds,
        args.edge_learning_rate,
        args.bias_learning_rate,
        args.time_constant_learning_rate,
        args.pair_loss_weight,
        args.gradient_norm_cap,
        args.regularization_weight,
        args.selection_interval,
        args.validation_episodes,
        args.validation_seconds,
        args.checkpoint_update,
        args.checkpoint_light_improvement,
        args.checkpoint_heavy_margin,
        args.teacher_audit_episodes,
        args.final_episodes,
        args.final_seconds,
        args.final_light_improvement,
        args.final_heavy_margin,
        *args.gradient_check_scales,
        args.gradient_check_tolerance,
    )
    if min(positive) <= 0.0:
        raise SystemExit("training sizes, times, rates, and thresholds must be positive")
    total_updates = args.rounds * args.updates_per_round
    if total_updates != 200:
        raise SystemExit("the bounded protocol requires exactly 200 updates")
    if args.checkpoint_update > total_updates:
        raise SystemExit("checkpoint update must fit within the update budget")
    if args.checkpoint_update % args.selection_interval:
        raise SystemExit("checkpoint update must coincide with selection")
    if not 0.0 < args.student_window_fraction < 1.0:
        raise SystemExit("student window fraction must be in (0, 1)")
    if args.late_window_start_seconds < args.early_window_end_seconds:
        raise SystemExit("late windows must begin after the early-window boundary")
    if args.late_window_end_seconds > args.trajectory_seconds:
        raise SystemExit("late windows must fit inside recorded trajectories")
    if args.window_steps * dt > args.early_window_end_seconds:
        raise SystemExit("a training window must fit in the early interval")
    for name in (
        "collection_episodes",
        "clean_episodes",
        "holdout_episodes",
        "validation_episodes",
        "teacher_audit_episodes",
        "final_episodes",
    ):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")
    if args.window_pairs < 2:
        raise SystemExit("window batches need at least two matched pairs")


def controller_parameter_sha256(controller: ConnectomeController) -> str:
    values = torch.cat(
        (
            controller.edge_magnitude.detach().flatten(),
            controller.bias.detach().flatten(),
            controller.raw_time_constant.detach().flatten(),
        )
    )
    array = values.cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_frozen_controller(
    graph: Path, checkpoint_path: Path, device: torch.device
) -> tuple[ConnectomeController, dict[str, Any], HoverConfig, GateConfig, int]:
    controller, checkpoint, hover_config, gate_config, resolution = load_controller(
        graph, checkpoint_path, device
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    controller.eval()
    return controller, checkpoint, hover_config, gate_config, resolution


def oracle_bias(mass_scale: Tensor, calibration: dict[str, Any]) -> Tensor:
    normalized_mass = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    return calibration["intercept"] + calibration["mass_slope"] * normalized_mass


def controller_step(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    neural: Tensor,
    specific_force: Tensor,
    stick_position: Tensor,
    *,
    privileged_bias: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    return controller(
        image,
        roll_pitch,
        neural,
        specific_force,
        stick_position,
        privileged_throttle_pool_bias=privileged_bias,
    )


def _cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous()


def _episode_summary(
    outcomes: dict[str, Tensor],
    cases: BalancedCases,
    *,
    steps: int,
    hover_config: HoverConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    summary = outcome_summary(
        success,
        outcomes["passed"],
        outcomes["collision"],
        outcomes["missed"],
        outcomes["crossing_radial"],
        outcomes["maximum_progress"],
        outcomes["maximum_approach"],
        saturation_fraction,
        cases.stratum_code,
        torch.zeros_like(cases.mass_scale),
    )
    tensors = {
        "success": _cpu(success),
        "codes": _cpu(cases.stratum_code),
        "mass_scale": _cpu(cases.mass_scale),
    }
    return summary, tensors


@torch.no_grad()
def collect_trajectories(
    behavior: ConnectomeController,
    teacher: ConnectomeController,
    cases: BalancedCases,
    calibration: dict[str, Any],
    *,
    seconds: float,
    target_onset_seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    teacher_drives_physics: bool,
    oracle_labels_enabled: bool = True,
) -> Trajectories:
    """Record oracle labels on histories generated by one declared policy."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    onset_step = round(target_onset_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    behavior_neural = behavior.initial_state(episodes, device=device, dtype=torch.float32)
    teacher_neural = teacher.initial_state(episodes, device=device, dtype=torch.float32)
    reference_neural = teacher.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    if teacher_drives_physics and not oracle_labels_enabled:
        raise ValueError("teacher-driven collection requires oracle labels")
    bias = oracle_bias(cases.mass_scale, calibration)

    images = torch.empty(steps, episodes, resolution, resolution, dtype=torch.float32)
    roll_pitch = torch.empty(steps, episodes, 2, dtype=torch.float32)
    specific_force = torch.empty(steps, episodes, 3, dtype=torch.float32)
    stick_position = torch.empty(steps, episodes, 4, dtype=torch.float32)
    oracle_motor = torch.empty(steps, episodes, 4, dtype=torch.float32)
    reference_motor = torch.empty(steps, episodes, 4, dtype=torch.float32)
    valid = torch.empty(steps, episodes, dtype=torch.bool)

    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        images[step].copy_(_cpu(image))
        roll_pitch[step].copy_(_cpu(state.euler[:, :2]))
        specific_force[step].copy_(_cpu(state.specific_force))
        stick_position[step].copy_(_cpu(stick_state.position))
        valid[step].copy_(_cpu(active))

        behavior_motor, behavior_neural = controller_step(
            behavior,
            image,
            state.euler[:, :2],
            behavior_neural,
            state.specific_force,
            stick_state.position,
        )
        if oracle_labels_enabled:
            reference_output, reference_neural = controller_step(
                teacher,
                image,
                state.euler[:, :2],
                reference_neural,
                state.specific_force,
                stick_state.position,
            )
            applied_bias = bias if step >= onset_step else None
            oracle_output, teacher_neural = controller_step(
                teacher,
                image,
                state.euler[:, :2],
                teacher_neural,
                state.specific_force,
                stick_state.position,
                privileged_bias=applied_bias,
            )
        else:
            reference_output = torch.zeros_like(behavior_motor)
            oracle_output = torch.zeros_like(behavior_motor)
        oracle_motor[step].copy_(_cpu(oracle_output))
        reference_motor[step].copy_(_cpu(reference_output))

        motor = oracle_output if teacher_drives_physics else behavior_motor
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        irrecoverable = (
            outcomes["collision"]
            | outcomes["missed"]
            | outcomes["recontact"]
            | (outcomes["maximum_tilt"] > math.radians(40.0))
        )
        active &= ~irrecoverable

    summary, _ = _episode_summary(outcomes, cases, steps=steps, hover_config=hover_config)
    return Trajectories(
        images=images,
        roll_pitch=roll_pitch,
        specific_force=specific_force,
        stick_position=stick_position,
        oracle_motor=oracle_motor,
        reference_motor=reference_motor,
        valid=valid,
        mass_scale=_cpu(cases.mass_scale),
        codes=_cpu(cases.stratum_code),
        collection_policy_sha256=controller_parameter_sha256(behavior),
        teacher_drives_physics=teacher_drives_physics,
        summary=summary,
    )


@torch.no_grad()
def evaluate_controller(
    controller: ConnectomeController,
    cases: BalancedCases,
    *,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    frozen_visual: bool = False,
    constant_acceleration: bool = False,
    pair_swapped_acceleration: bool = False,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    if sum((frozen_visual, constant_acceleration, pair_swapped_acceleration)) > 1:
        raise ValueError("only one sensor intervention may be active")
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    initial_image = render_annular_gate(
        state,
        cases.gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    pair_swap = torch.arange(episodes, device=device).bitwise_xor(1)
    for step in range(steps):
        image = (
            initial_image
            if frozen_visual
            else render_annular_gate(
                state,
                cases.gate,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
        )
        acceleration = state.specific_force
        if constant_acceleration:
            acceleration = torch.zeros_like(acceleration)
            acceleration[:, 2] = 9.81
        elif pair_swapped_acceleration:
            acceleration = acceleration[pair_swap]
        motor, neural = controller_step(
            controller,
            image,
            state.euler[:, :2],
            neural,
            acceleration,
            stick_state.position,
        )
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
    return _episode_summary(outcomes, cases, steps=steps, hover_config=hover_config)


@torch.no_grad()
def teacher_takeover_audit(
    behavior: ConnectomeController,
    teacher: ConnectomeController,
    cases: BalancedCases,
    calibration: dict[str, Any],
    *,
    takeover_seconds: float,
    seconds: float,
    target_onset_seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    """Test the native oracle after shadowing the actual student history."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    onset_step = round(target_onset_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    behavior_neural = behavior.initial_state(episodes, device=device, dtype=torch.float32)
    teacher_neural = teacher.initial_state(episodes, device=device, dtype=torch.float32)
    bias = oracle_bias(cases.mass_scale, calibration)
    outcomes = initialize_outcomes(cases, gate_config)
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        behavior_motor, behavior_neural = controller_step(
            behavior,
            image,
            state.euler[:, :2],
            behavior_neural,
            state.specific_force,
            stick_state.position,
        )
        teacher_motor, teacher_neural = controller_step(
            teacher,
            image,
            state.euler[:, :2],
            teacher_neural,
            state.specific_force,
            stick_state.position,
            privileged_bias=bias if step >= onset_step else None,
        )
        motor = teacher_motor if step >= takeover_step else behavior_motor
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
    summary, _ = _episode_summary(outcomes, cases, steps=steps, hover_config=hover_config)
    summary["takeover_seconds"] = takeover_seconds
    return summary


def _paired_episode_indices(pair_indices: Tensor) -> Tensor:
    return torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=1).flatten()


def select_window(
    student: Trajectories,
    clean: Trajectories,
    *,
    pairs: int,
    student_fraction: float,
    start: int,
    steps: int,
    generator: torch.Generator,
    device: torch.device,
) -> SelectedWindow:
    student_pairs = round(pairs * student_fraction)
    student_pairs = min(max(student_pairs, 1), pairs - 1)
    clean_pairs = pairs - student_pairs
    student_choices = torch.randperm(student.mass_scale.numel() // 2, generator=generator)[
        :student_pairs
    ]
    clean_choices = torch.randperm(clean.mass_scale.numel() // 2, generator=generator)[:clean_pairs]
    student_indices = _paired_episode_indices(student_choices)
    clean_indices = _paired_episode_indices(clean_choices)
    end = start + steps
    if end > student.images.shape[0] or end > clean.images.shape[0]:
        raise ValueError("selected window exceeds a recorded trajectory")

    def combined(name: str) -> Tensor:
        student_value = getattr(student, name)[:end, student_indices]
        clean_value = getattr(clean, name)[:end, clean_indices]
        return torch.cat((student_value, clean_value), dim=1).to(device)

    return SelectedWindow(
        images=combined("images"),
        roll_pitch=combined("roll_pitch"),
        specific_force=combined("specific_force"),
        stick_position=combined("stick_position"),
        oracle_motor=combined("oracle_motor"),
        reference_motor=combined("reference_motor"),
        valid=combined("valid"),
        mass_scale=torch.cat(
            (student.mass_scale[student_indices], clean.mass_scale[clean_indices])
        ).to(device),
        student_episode=torch.cat(
            (
                torch.ones(len(student_indices), dtype=torch.bool),
                torch.zeros(len(clean_indices), dtype=torch.bool),
            )
        ).to(device),
        loss_start=start,
    )


@torch.no_grad()
def reconstruct_prefix(controller: ConnectomeController, window: SelectedWindow) -> Tensor:
    episodes = window.images.shape[1]
    neural = controller.initial_state(
        episodes, device=window.images.device, dtype=window.images.dtype
    )
    for step in range(window.loss_start):
        _, neural = controller_step(
            controller,
            window.images[step],
            window.roll_pitch[step],
            neural,
            window.specific_force[step],
            window.stick_position[step],
        )
    return neural.detach()


def replay_window(
    controller: ConnectomeController,
    window: SelectedWindow,
    incoming_hidden: Tensor,
    *,
    steps: int | None = None,
) -> Tensor:
    neural = incoming_hidden
    outputs = []
    end = len(window.images) if steps is None else window.loss_start + steps
    for step in range(window.loss_start, end):
        motor, neural = controller_step(
            controller,
            window.images[step],
            window.roll_pitch[step],
            neural,
            window.specific_force[step],
            window.stick_position[step],
        )
        outputs.append(motor)
    return torch.stack(outputs)


def normalized_action_loss(
    prediction: Tensor,
    target: Tensor,
    valid: Tensor,
    axis_scales: Tensor,
    *,
    pair_loss_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    if not bool(valid.any()):
        raise RuntimeError("training window contains no valid student states")
    normalized_error = (prediction - target) / axis_scales
    absolute = normalized_error.square()[valid].mean()
    paired_valid = valid[:, 0::2] & valid[:, 1::2]
    prediction_pair = prediction[:, 0::2, 3] - prediction[:, 1::2, 3]
    target_pair = target[:, 0::2, 3] - target[:, 1::2, 3]
    if bool(paired_valid.any()):
        pair = ((prediction_pair - target_pair) / axis_scales[3]).square()[paired_valid].mean()
    else:
        pair = absolute.new_zeros(())
    loss = absolute + pair_loss_weight * pair
    return loss, {
        "absolute_normalized_mse": float(absolute.detach()),
        "matched_throttle_normalized_mse": float(pair.detach()),
        "valid_fraction": float(valid.float().mean()),
    }


def training_loss(
    controller: ConnectomeController,
    window: SelectedWindow,
    incoming_hidden: Tensor,
    axis_scales: Tensor,
    initial_parameters: dict[str, Tensor],
    *,
    pair_loss_weight: float,
    regularization_weight: float,
    steps: int | None = None,
) -> tuple[Tensor, dict[str, float]]:
    prediction = replay_window(controller, window, incoming_hidden, steps=steps)
    length = len(prediction)
    begin = window.loss_start
    target = window.oracle_motor[begin : begin + length]
    valid = window.valid[begin : begin + length]
    imitation, details = normalized_action_loss(
        prediction,
        target,
        valid,
        axis_scales,
        pair_loss_weight=pair_loss_weight,
    )
    edge_scale = initial_parameters["edge_magnitude"].clamp_min(0.05)
    edge_regularization = (
        ((controller.edge_magnitude - initial_parameters["edge_magnitude"]) / edge_scale)
        .square()
        .mean()
    )
    bias_regularization = (controller.bias - initial_parameters["bias"]).square().mean()
    tau_regularization = (
        (controller.raw_time_constant - initial_parameters["raw_time_constant"]).square().mean()
    )
    regularization = edge_regularization + bias_regularization + tau_regularization
    total = imitation + regularization_weight * regularization
    details.update(
        {
            "imitation_loss": float(imitation.detach()),
            "regularization": float(regularization.detach()),
            "total_loss": float(total.detach()),
        }
    )
    return total, details


def label_axis_scales(*datasets: Trajectories) -> Tensor:
    labels = []
    for dataset in datasets:
        selected = dataset.oracle_motor[dataset.valid]
        if len(selected):
            labels.append(selected)
    if not labels:
        raise RuntimeError("cannot estimate action scales without valid oracle labels")
    combined = torch.cat(labels)
    return combined.std(dim=0).clamp_min(0.01)


def _parameter_groups(controller: ConnectomeController) -> dict[str, Tensor]:
    return {
        "edge_magnitude": controller.edge_magnitude,
        "bias": controller.bias,
        "raw_time_constant": controller.raw_time_constant,
    }


def gradient_audit(
    controller: ConnectomeController,
    window: SelectedWindow,
    axis_scales: Tensor,
    initial_parameters: dict[str, Tensor],
    *,
    lengths: tuple[int, ...],
    scales: tuple[float, ...],
    tolerance: float,
    pair_loss_weight: float,
    regularization_weight: float,
) -> dict[str, Any]:
    incoming_hidden = reconstruct_prefix(controller, window)
    parameters = _parameter_groups(controller)
    report: dict[str, Any] = {}
    originals = {name: value.detach().clone() for name, value in parameters.items()}
    try:
        for length in lengths:
            if length > len(window.images) - window.loss_start:
                raise ValueError("gradient audit length exceeds selected window")
            length_report: dict[str, Any] = {}
            for name, parameter in parameters.items():
                with torch.no_grad():
                    repeated = [
                        float(
                            training_loss(
                                controller,
                                window,
                                incoming_hidden,
                                axis_scales,
                                initial_parameters,
                                pair_loss_weight=pair_loss_weight,
                                regularization_weight=regularization_weight,
                                steps=length,
                            )[0]
                        )
                        for _ in range(3)
                    ]
                repeat_noise = max(repeated) - min(repeated)
                controller.zero_grad(set_to_none=True)
                loss, _ = training_loss(
                    controller,
                    window,
                    incoming_hidden,
                    axis_scales,
                    initial_parameters,
                    pair_loss_weight=pair_loss_weight,
                    regularization_weight=regularization_weight,
                    steps=length,
                )
                loss.backward()
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"nonfinite or missing analytic gradient for {name}")
                direction = parameter.grad.detach().sign()
                if name == "edge_magnitude":
                    at_bound = (parameter.detach() <= 0.01) | (parameter.detach() >= 8.0 - 0.01)
                    direction[at_bound] = 0.0
                analytic = float((parameter.grad * direction).sum())
                checks = []
                baseline = originals[name]
                for scale in scales:
                    with torch.no_grad():
                        parameter.copy_(baseline + scale * direction)
                    plus, _ = training_loss(
                        controller,
                        window,
                        incoming_hidden,
                        axis_scales,
                        initial_parameters,
                        pair_loss_weight=pair_loss_weight,
                        regularization_weight=regularization_weight,
                        steps=length,
                    )
                    with torch.no_grad():
                        parameter.copy_(baseline - scale * direction)
                    minus, _ = training_loss(
                        controller,
                        window,
                        incoming_hidden,
                        axis_scales,
                        initial_parameters,
                        pair_loss_weight=pair_loss_weight,
                        regularization_weight=regularization_weight,
                        steps=length,
                    )
                    plus_value = float(plus.detach())
                    minus_value = float(minus.detach())
                    estimate = (plus_value - minus_value) / (2.0 * scale)
                    with torch.no_grad():
                        parameter.copy_(baseline)
                    denominator = max(abs(analytic), abs(estimate), 1.0e-10)
                    relative_error = abs(estimate - analytic) / denominator
                    measurable = abs(plus_value - minus_value) > max(10.0 * repeat_noise, 1.0e-10)
                    checks.append(
                        {
                            "scale": scale,
                            "plus_loss": plus_value,
                            "minus_loss": minus_value,
                            "central_difference": estimate,
                            "relative_error": relative_error,
                            "measurable_above_repeat_noise": measurable,
                            "matching_sign": estimate * analytic > 0.0,
                        }
                    )
                adjacent_passes = [
                    left["measurable_above_repeat_noise"]
                    and right["measurable_above_repeat_noise"]
                    and left["matching_sign"]
                    and right["matching_sign"]
                    and left["relative_error"] <= tolerance
                    and right["relative_error"] <= tolerance
                    for left, right in zip(checks[:-1], checks[1:], strict=True)
                ]
                passed = any(adjacent_passes)
                if not passed:
                    raise RuntimeError(
                        f"{length}-step {name} finite difference failed: "
                        f"analytic={analytic:.6g}, checks={checks}"
                    )
                length_report[name] = {
                    "analytic": analytic,
                    "unchanged_loss_repeats": repeated,
                    "unchanged_loss_range": repeat_noise,
                    "checks": checks,
                    "passed": passed,
                }
            report[str(length)] = length_report
    finally:
        with torch.no_grad():
            for name, parameter in parameters.items():
                parameter.copy_(originals[name])
        controller.zero_grad(set_to_none=True)
    return report


@torch.no_grad()
def replay_action_metrics(
    controller: ConnectomeController,
    dataset: Trajectories,
    axis_scales: Tensor,
    *,
    horizons: tuple[float, ...],
    dt: float,
    device: torch.device,
) -> dict[str, Any]:
    episodes = dataset.mass_scale.numel()
    horizon_indices = {max(round(value / dt) - 1, 0): value for value in horizons}
    final_step = max(horizon_indices)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    report: dict[str, Any] = {}
    scales = axis_scales.to(device)
    mass_scale = dataset.mass_scale.to(device)

    def safe_delta(prediction: Tensor, target: Tensor) -> dict[str, float | None]:
        if not prediction.numel():
            return {
                "rmse": None,
                "normalized_rmse": None,
                "signed_slope": None,
                "pearson_correlation": None,
                "prediction_mean": None,
                "target_mean": None,
                "prediction_std": None,
                "target_std": None,
            }
        return delta_metrics(prediction, target)

    def safe_mean(values: Tensor) -> float | None:
        return float(values.mean()) if values.numel() else None

    for step in range(final_step + 1):
        prediction, neural = controller_step(
            controller,
            dataset.images[step].to(device),
            dataset.roll_pitch[step].to(device),
            neural,
            dataset.specific_force[step].to(device),
            dataset.stick_position[step].to(device),
        )
        if step not in horizon_indices:
            continue
        horizon = horizon_indices[step]
        reference = dataset.reference_motor[step].to(device)
        oracle = dataset.oracle_motor[step].to(device)
        valid = dataset.valid[step].to(device)
        target_delta = oracle - reference
        prediction_delta = prediction - reference
        error = (prediction_delta - target_delta) / scales
        pair_valid = valid[0::2] & valid[1::2]
        prediction_pair = prediction_delta[0::2, 3] - prediction_delta[1::2, 3]
        target_pair = target_delta[0::2, 3] - target_delta[1::2, 3]
        lower = mass_scale < 1.0
        valid_count = int(valid.sum())
        valid_pair_count = int(pair_valid.sum())
        metrics: dict[str, Any] = {
            "valid_episode_count": valid_count,
            "valid_matched_pair_count": valid_pair_count,
            "valid_fraction": float(valid.float().mean()),
            "all_axis_scale_normalized_rmse": (
                float(error[valid].square().mean().sqrt()) if valid_count else None
            ),
            "axis_absolute_rmse": {
                name: (
                    float(
                        (prediction_delta[valid, index] - target_delta[valid, index])
                        .square()
                        .mean()
                        .sqrt()
                    )
                    if valid_count
                    else None
                )
                for index, name in enumerate(("roll", "pitch", "yaw", "throttle"))
            },
            "throttle_delta": safe_delta(prediction_delta[valid, 3], target_delta[valid, 3]),
            "matched_throttle_contrast": safe_delta(
                prediction_pair[pair_valid], target_pair[pair_valid]
            ),
            "light_throttle_delta_mean": safe_mean(prediction_delta[valid & lower, 3]),
            "light_throttle_target_mean": safe_mean(target_delta[valid & lower, 3]),
            "heavy_throttle_delta_mean": safe_mean(prediction_delta[valid & ~lower, 3]),
            "heavy_throttle_target_mean": safe_mean(target_delta[valid & ~lower, 3]),
        }
        report[f"{horizon:.2f}"] = metrics
    return report


def action_fidelity_passed(metrics: dict[str, Any]) -> bool:
    for horizon in ("0.75", "1.50", "3.00"):
        if metrics[horizon]["valid_episode_count"] == 0:
            return False
        if metrics[horizon]["valid_matched_pair_count"] == 0:
            return False
        throttle = metrics[horizon]["throttle_delta"]
        pair = metrics[horizon]["matched_throttle_contrast"]
        if throttle["signed_slope"] is None or pair["signed_slope"] is None:
            return False
        required = (
            throttle["signed_slope"],
            throttle["normalized_rmse"],
            pair["signed_slope"],
            pair["normalized_rmse"],
        )
        if any(value is None or not math.isfinite(value) for value in required):
            return False
        if not 0.5 <= throttle["signed_slope"] <= 1.5:
            return False
        if throttle["normalized_rmse"] > 0.5 or pair["normalized_rmse"] > 0.5:
            return False
    return True


def action_fidelity_score(metrics: dict[str, Any]) -> float:
    values = []
    for horizon in ("0.75", "1.50", "3.00"):
        values.append(metrics[horizon]["throttle_delta"]["normalized_rmse"])
        values.append(metrics[horizon]["matched_throttle_contrast"]["normalized_rmse"])
    if any(value is None or not math.isfinite(value) for value in values):
        return float("inf")
    return float(np.mean(values))


def parameter_snapshot(controller: ConnectomeController, *, cpu: bool = False) -> dict[str, Tensor]:
    result = {}
    for name, parameter in _parameter_groups(controller).items():
        value = parameter.detach().clone()
        result[name] = value.cpu() if cpu else value
    return result


def restore_parameters(controller: ConnectomeController, state: dict[str, Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in _parameter_groups(controller).items():
            parameter.copy_(state[name].to(parameter.device))
    controller.project_parameters()


def parameter_change_summary(
    controller: ConnectomeController, initial: dict[str, Tensor]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, parameter in _parameter_groups(controller).items():
        delta = parameter.detach() - initial[name]
        result[name] = {
            "count": parameter.numel(),
            "l2_change": float(delta.norm()),
            "maximum_absolute_change": float(delta.abs().max()),
            "changed_fraction_above_1e-7": float((delta.abs() > 1.0e-7).float().mean()),
        }
    result["time_constant_seconds"] = {
        "minimum": float(controller.time_constant.detach().min()),
        "mean": float(controller.time_constant.detach().mean()),
        "maximum": float(controller.time_constant.detach().max()),
    }
    return result


def trajectory_causality_audit(
    with_labels: Trajectories, without_labels: Trajectories
) -> dict[str, Any]:
    differences = {}
    for name in (
        "images",
        "roll_pitch",
        "specific_force",
        "stick_position",
        "valid",
        "mass_scale",
        "codes",
    ):
        left = getattr(with_labels, name)
        right = getattr(without_labels, name)
        if left.dtype == torch.bool or not left.dtype.is_floating_point:
            differences[name] = int((left != right).sum())
        else:
            differences[name] = float((left - right).abs().max())
    passed = all(value == 0 for value in differences.values()) and (
        with_labels.summary == without_labels.summary
    )
    return {
        "teacher_labels_enabled_vs_disabled_sensor_differences": differences,
        "outcome_summaries_identical": with_labels.summary == without_labels.summary,
        "passed": passed,
    }


def compact_flight(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": summary["success_rate"],
        "light_success_rate": summary["light_success_rate"],
        "heavy_success_rate": summary["heavy_success_rate"],
        "ring_collision_rate": summary["ring_collision_rate"],
        "miss_rate": summary["miss_rate"],
    }


def save_candidate_checkpoint(
    path: Path,
    source: dict[str, Any],
    source_path: Path,
    teacher_path: Path,
    controller: ConnectomeController,
    promotion: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source)
    checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    checkpoint["source_checkpoint_sha256"] = file_sha256(source_path)
    checkpoint["full_network_oracle_distillation"] = {
        "method": "student-history recurrent oracle-action distillation",
        "teacher_checkpoint_sha256": file_sha256(teacher_path),
        "all_native_edge_magnitudes_trainable": True,
        "all_native_biases_trainable": True,
        "all_native_time_constants_trainable": True,
        "fixed_topology": True,
        "fixed_transmitter_signs": True,
        "training_only_mass_oracle": True,
        "compiled_into_native_parameters": True,
        "parameter_vector_sha256": controller_parameter_sha256(controller),
        "promotion": promotion,
    }
    torch.save(checkpoint, path)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    student, student_checkpoint, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.student_checkpoint, device
    )
    teacher, teacher_checkpoint, teacher_hover, teacher_gate, teacher_resolution = (
        load_frozen_controller(args.graph, args.teacher_checkpoint, device)
    )
    validate_args(args, hover_config.dt)
    if (
        asdict(teacher_hover) != asdict(hover_config)
        or asdict(teacher_gate) != asdict(gate_config)
        or teacher_resolution != resolution
    ):
        raise SystemExit("student and teacher simulation contracts differ")
    if not student.uses_accelerometer or not teacher.uses_accelerometer:
        raise SystemExit("full-network distillation requires the accelerometer graph")
    if student.uses_proprioception or teacher.uses_proprioception:
        raise SystemExit("this protocol expects no active engineered proprioception channel")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("unexpected teacher calibration kind")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.optimization_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    initial_parameters = parameter_snapshot(student)
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        (
            {"params": [student.edge_magnitude], "lr": args.edge_learning_rate},
            {"params": [student.bias], "lr": args.bias_learning_rate},
            {
                "params": [student.raw_time_constant],
                "lr": args.time_constant_learning_rate,
            },
        )
    )

    teacher_cases = sample_matched_cases(
        args.teacher_audit_episodes,
        seed=args.teacher_audit_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    takeover_thresholds = {0.0: 0.95, 0.5: 0.85, 1.0: 0.80, 2.0: 0.60}
    takeover = {}
    for takeover_time, threshold in takeover_thresholds.items():
        result = teacher_takeover_audit(
            student,
            teacher,
            teacher_cases,
            calibration,
            takeover_seconds=takeover_time,
            seconds=args.final_seconds,
            target_onset_seconds=args.target_onset_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        takeover[f"{takeover_time:.2f}"] = {**result, "minimum_required": threshold}
        print(
            json.dumps(
                {
                    "phase": "teacher_takeover_audit",
                    "takeover_seconds": takeover_time,
                    "minimum_required": threshold,
                    **compact_flight(result),
                }
            ),
            flush=True,
        )
    takeover_passed = all(
        item["success_rate"] >= item["minimum_required"] for item in takeover.values()
    )
    if not takeover_passed:
        raise SystemExit(f"native oracle failed student-state takeover audit: {takeover}")

    causal_device = torch.device("cpu")
    causal_student = copy.deepcopy(student).to(causal_device)
    causal_teacher = copy.deepcopy(teacher).to(causal_device)
    causal_cases = sample_matched_cases(
        8,
        seed=args.collection_seed - 1,
        device=causal_device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    causal_with = collect_trajectories(
        causal_student,
        causal_teacher,
        causal_cases,
        calibration,
        seconds=min(args.trajectory_seconds, 2.0),
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=False,
    )
    causal_without = collect_trajectories(
        causal_student,
        causal_teacher,
        causal_cases,
        calibration,
        seconds=min(args.trajectory_seconds, 2.0),
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=False,
        oracle_labels_enabled=False,
    )
    causality = trajectory_causality_audit(causal_with, causal_without)
    causality["audit_device"] = "cpu for deterministic index reductions"
    if not causality["passed"]:
        raise SystemExit(f"oracle labels leaked into student trajectory: {causality}")
    del causal_student, causal_teacher, causal_with, causal_without

    clean_cases = sample_matched_cases(
        args.clean_episodes,
        seed=args.clean_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    clean = collect_trajectories(
        student,
        teacher,
        clean_cases,
        calibration,
        seconds=args.trajectory_seconds,
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=True,
    )
    holdout_cases = sample_matched_cases(
        args.holdout_episodes,
        seed=args.holdout_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    holdout = collect_trajectories(
        student,
        teacher,
        holdout_cases,
        calibration,
        seconds=args.trajectory_seconds,
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=False,
    )
    validation_cases = sample_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    baseline_validation, _ = evaluate_controller(
        student,
        validation_cases,
        seconds=args.validation_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )

    window_generator = torch.Generator(device="cpu")
    window_generator.manual_seed(args.optimization_seed)
    training_updates = []
    validations = []
    archive: dict[str, dict[str, Any]] = {}
    axis_scales: Tensor | None = None
    baseline_action: dict[str, Any] | None = None
    baseline_action_score: float | None = None
    audits: dict[str, Any] = {}
    early_stopped = False
    global_update = 0

    for round_index in range(args.rounds):
        collection_cases = sample_matched_cases(
            args.collection_episodes,
            seed=args.collection_seed + round_index,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        student_data = collect_trajectories(
            student,
            teacher,
            collection_cases,
            calibration,
            seconds=args.trajectory_seconds,
            target_onset_seconds=args.target_onset_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            teacher_drives_physics=False,
        )
        print(
            json.dumps(
                {
                    "phase": "student_collection",
                    "round": round_index + 1,
                    "policy_sha256": student_data.collection_policy_sha256,
                    **compact_flight(student_data.summary),
                }
            ),
            flush=True,
        )
        if axis_scales is None:
            axis_scales = label_axis_scales(clean, student_data).to(device)
            baseline_action = replay_action_metrics(
                student,
                holdout,
                axis_scales,
                horizons=(0.75, 1.5, 3.0),
                dt=hover_config.dt,
                device=device,
            )
            baseline_action_score = action_fidelity_score(baseline_action)
            early_start = max(
                0,
                (round(args.early_window_end_seconds / hover_config.dt) - args.window_steps) // 2,
            )
            late_start = round(args.late_window_start_seconds / hover_config.dt)
            audit_windows = {
                "early": select_window(
                    student_data,
                    clean,
                    pairs=args.window_pairs,
                    student_fraction=args.student_window_fraction,
                    start=early_start,
                    steps=args.window_steps,
                    generator=window_generator,
                    device=device,
                ),
                "late": select_window(
                    student_data,
                    clean,
                    pairs=args.window_pairs,
                    student_fraction=args.student_window_fraction,
                    start=late_start,
                    steps=args.window_steps,
                    generator=window_generator,
                    device=device,
                ),
            }
            for name, audit_window in audit_windows.items():
                audits[name] = gradient_audit(
                    student,
                    audit_window,
                    axis_scales,
                    initial_parameters,
                    lengths=(25, args.window_steps),
                    scales=tuple(args.gradient_check_scales),
                    tolerance=args.gradient_check_tolerance,
                    pair_loss_weight=args.pair_loss_weight,
                    regularization_weight=args.regularization_weight,
                )
            print(json.dumps({"phase": "gradient_audits", "passed": True}), flush=True)

        assert axis_scales is not None
        assert baseline_action_score is not None
        for _local_update in range(args.updates_per_round):
            global_update += 1
            early = global_update % 2 == 1
            if early:
                maximum_start = (
                    round(args.early_window_end_seconds / hover_config.dt) - args.window_steps
                )
                start = int(
                    torch.randint(maximum_start + 1, (1,), generator=window_generator).item()
                )
                window_kind = "early"
            else:
                minimum_start = round(args.late_window_start_seconds / hover_config.dt)
                maximum_start = (
                    round(args.late_window_end_seconds / hover_config.dt) - args.window_steps
                )
                start = int(
                    torch.randint(
                        minimum_start,
                        maximum_start + 1,
                        (1,),
                        generator=window_generator,
                    ).item()
                )
                window_kind = "late"
            window = select_window(
                student_data,
                clean,
                pairs=args.window_pairs,
                student_fraction=args.student_window_fraction,
                start=start,
                steps=args.window_steps,
                generator=window_generator,
                device=device,
            )
            incoming_hidden = reconstruct_prefix(student, window)
            optimizer.zero_grad(set_to_none=True)
            loss, details = training_loss(
                student,
                window,
                incoming_hidden,
                axis_scales,
                initial_parameters,
                pair_loss_weight=args.pair_loss_weight,
                regularization_weight=args.regularization_weight,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite full-network distillation loss")
            loss.backward()
            raw_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in student.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
            )
            optimizer.step()
            student.project_parameters()
            training_updates.append(
                {
                    "update": global_update,
                    "round": round_index + 1,
                    "window_kind": window_kind,
                    "window_start_step": start,
                    "window_start_seconds": start * hover_config.dt,
                    "student_episode_fraction": float(window.student_episode.float().mean()),
                    "raw_gradient_norm": float(raw_norm),
                    **details,
                }
            )

            if global_update % args.selection_interval:
                continue
            validation, _ = evaluate_controller(
                student,
                validation_cases,
                seconds=args.validation_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            action = replay_action_metrics(
                student,
                holdout,
                axis_scales,
                horizons=(0.75, 1.5, 3.0),
                dt=hover_config.dt,
                device=device,
            )
            action_score = action_fidelity_score(action)
            key = controller_parameter_sha256(student)
            archive[key] = {
                "update": global_update,
                "parameters": parameter_snapshot(student, cpu=True),
                "flight": validation,
                "action": action,
                "action_fidelity_score": action_score,
                "action_fidelity_passed": action_fidelity_passed(action),
                "light_success_delta": (
                    validation["light_success_rate"] - baseline_validation["light_success_rate"]
                ),
                "heavy_success_delta": (
                    validation["heavy_success_rate"] - baseline_validation["heavy_success_rate"]
                ),
            }
            validations.append(
                {name: value for name, value in archive[key].items() if name != "parameters"}
                | {"parameter_vector_sha256": key}
            )
            print(
                json.dumps(
                    {
                        "phase": "validation",
                        "update": global_update,
                        "parameter_vector_sha256": key,
                        "action_fidelity_score": action_score,
                        "action_fidelity_passed": archive[key]["action_fidelity_passed"],
                        "light_success_delta": archive[key]["light_success_delta"],
                        "heavy_success_delta": archive[key]["heavy_success_delta"],
                        **compact_flight(validation),
                    }
                ),
                flush=True,
            )
            torch.save(
                {
                    archive_key: {
                        "update": item["update"],
                        "parameters": item["parameters"],
                        "action_fidelity_score": item["action_fidelity_score"],
                    }
                    for archive_key, item in archive.items()
                },
                args.output_dir / "archive.pt",
            )
            (args.output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "updates": training_updates,
                        "validations": validations,
                        "elapsed_seconds": perf_counter() - started,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            if global_update == args.checkpoint_update:
                joint_signal = any(
                    item["light_success_delta"] >= args.checkpoint_light_improvement
                    and item["heavy_success_delta"] >= -args.checkpoint_heavy_margin
                    and item["action_fidelity_score"] <= 0.90 * baseline_action_score
                    for item in archive.values()
                )
                if not joint_signal:
                    early_stopped = True
                    print(
                        json.dumps(
                            {
                                "phase": "early_stop",
                                "update": global_update,
                                "reason": "no_joint_action_and_flight_signal",
                            }
                        ),
                        flush=True,
                    )
                    break
        if early_stopped:
            break

    if not archive:
        raise RuntimeError("training produced no selectable checkpoints")
    assert axis_scales is not None
    assert baseline_action is not None
    best_fidelity_key = min(archive, key=lambda key: archive[key]["action_fidelity_score"])
    flight_eligible = [
        key
        for key, item in archive.items()
        if item["heavy_success_delta"] >= -args.final_heavy_margin
    ]
    selection_pool = flight_eligible or list(archive)
    fidelity_eligible = [key for key in selection_pool if archive[key]["action_fidelity_passed"]]
    if fidelity_eligible:
        selection_pool = fidelity_eligible
    best_flight_key = max(
        selection_pool,
        key=lambda key: (
            archive[key]["flight"]["light_success_rate"],
            archive[key]["flight"]["success_rate"],
            -archive[key]["action_fidelity_score"],
        ),
    )
    selected_key = best_flight_key
    restore_parameters(student, archive[selected_key]["parameters"])

    vector_path = args.output_dir / "candidate-vector.json"
    vector_path.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": file_sha256(args.student_checkpoint),
                "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint),
                "selected_update": archive[selected_key]["update"],
                "parameter_vector_sha256": controller_parameter_sha256(student),
                "edge_magnitude": student.edge_magnitude.detach().cpu().tolist(),
                "bias": student.bias.detach().cpu().tolist(),
                "raw_time_constant": student.raw_time_constant.detach().cpu().tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    final_cases = sample_matched_cases(
        args.final_episodes,
        seed=args.final_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    reference = copy.deepcopy(student).to(device)
    restore_parameters(reference, initial_parameters)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    final: dict[str, dict[str, Any]] = {}
    final_outcomes: dict[str, dict[str, Tensor]] = {}
    for name, evaluated, controls in (
        ("reference", reference, {}),
        ("candidate", student, {}),
        ("candidate_frozen_first_frame", student, {"frozen_visual": True}),
        ("candidate_constant_1g", student, {"constant_acceleration": True}),
        (
            "candidate_pair_swapped_acceleration",
            student,
            {"pair_swapped_acceleration": True},
        ),
    ):
        summary, outcomes = evaluate_controller(
            evaluated,
            final_cases,
            seconds=args.final_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            **controls,
        )
        final[name] = summary
        final_outcomes[name] = outcomes
        print(json.dumps({"phase": "final", "name": name, **compact_flight(summary)}), flush=True)

    reference_success = final_outcomes["reference"]["success"]
    candidate_success = final_outcomes["candidate"]["success"]
    codes = final_outcomes["reference"]["codes"]
    light = ~codes.bitwise_and(1).bool()
    heavy = ~light
    paired_overall = paired_clustered_confidence_interval(reference_success, candidate_success)
    paired_light = paired_confidence_interval(reference_success[light], candidate_success[light])
    paired_heavy = paired_confidence_interval(reference_success[heavy], candidate_success[heavy])
    acceleration_controls = {
        "live_minus_constant_1g": paired_clustered_confidence_interval(
            final_outcomes["candidate_constant_1g"]["success"], candidate_success
        ),
        "live_minus_pair_swapped": paired_clustered_confidence_interval(
            final_outcomes["candidate_pair_swapped_acceleration"]["success"],
            candidate_success,
        ),
    }
    acceleration_dependence = all(
        item["confidence_95"][0] > 0.0 for item in acceleration_controls.values()
    )

    fresh_history_cases = sample_matched_cases(
        args.holdout_episodes,
        seed=args.final_seed + 1,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    fresh_student_history = collect_trajectories(
        student,
        teacher,
        fresh_history_cases,
        calibration,
        seconds=args.trajectory_seconds,
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives_physics=False,
    )
    final_action = replay_action_metrics(
        student,
        fresh_student_history,
        axis_scales,
        horizons=(0.75, 1.5, 3.0),
        dt=hover_config.dt,
        device=device,
    )
    final_action_passed = action_fidelity_passed(final_action)
    promotion_checks = {
        "student_history_action_fidelity": final_action_passed,
        "light_improvement_at_least_threshold": (
            final["candidate"]["light_success_rate"]
            >= final["reference"]["light_success_rate"] + args.final_light_improvement
        ),
        "light_paired_confidence_interval_excludes_zero": (paired_light["confidence_95"][0] > 0.0),
        "heavy_nondegradation_within_margin": (
            final["candidate"]["heavy_success_rate"]
            >= final["reference"]["heavy_success_rate"] - args.final_heavy_margin
        ),
        "heavy_paired_noninferiority_interval_within_margin": (
            paired_heavy["confidence_95"][0] >= -args.final_heavy_margin
        ),
        "overall_success_improves": (
            final["candidate"]["success_rate"] > final["reference"]["success_rate"]
        ),
        "overall_paired_confidence_interval_excludes_zero": (
            paired_overall["confidence_95"][0] > 0.0
        ),
        "frozen_first_frame_success_at_most_five_percent": (
            final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
        ),
        "causal_acceleration_dependence": acceleration_dependence,
    }
    promotion = {"checks": promotion_checks, "passed": all(promotion_checks.values())}
    candidate_path = args.output_dir / "candidate.pt"
    if promotion["passed"]:
        save_candidate_checkpoint(
            candidate_path,
            student_checkpoint,
            args.student_checkpoint,
            args.teacher_checkpoint,
            student,
            promotion,
        )
    goal_passed = bool(
        promotion["passed"]
        and final["candidate"]["success_rate"] >= 0.90
        and final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
    )
    report = {
        "method": "full-network recurrent oracle-action distillation on student histories",
        "claim_scope": (
            "The deployed actor receives only current FPV, roll/pitch, body-Z specific force, "
            "and its persistent native connectome state. Mass, oracle actions, clean-trajectory "
            "markers, timestamps, replay prefixes, and teacher state are training-only."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "student_checkpoint": stable_path(args.student_checkpoint),
        "student_checkpoint_sha256": file_sha256(args.student_checkpoint),
        "teacher_checkpoint": stable_path(args.teacher_checkpoint),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "all_native_edge_magnitudes_trainable": True,
            "all_native_biases_trainable": True,
            "all_native_time_constants_trainable": True,
            "teacher_and_mass_labels_training_only": True,
            "student_histories_drive_primary_supervision": True,
            "clean_teacher_histories_stabilization_only": True,
            "clean_window_fraction": 1.0 - args.student_window_fraction,
            "physics_and_renderer_outside_autograd": True,
            "incoming_hidden_reconstructed_from_reset_under_current_student": True,
            "incoming_hidden_held_fixed_during_truncated_gradient": True,
            "deployed_actor_inputs": (
                "current FPV, roll/pitch, body-Z specific force, and persistent native "
                "connectome state"
            ),
            "engineered_history_features": False,
            "proprioception_input_active": student.uses_proprioception,
        },
        "teacher_checkpoint_metadata_keys": sorted(teacher_checkpoint),
        "teacher_takeover_audit": {
            "results": takeover,
            "passed": takeover_passed,
        },
        "student_collection_causality_audit": causality,
        "gradient_audits": audits,
        "axis_scales": axis_scales.detach().cpu().tolist(),
        "clean_collection": clean.summary,
        "baseline_student_collection": holdout.summary,
        "baseline_validation": baseline_validation,
        "baseline_student_history_action": baseline_action,
        "baseline_action_fidelity_score": baseline_action_score,
        "updates_completed": len(training_updates),
        "early_stopped_at_checkpoint": early_stopped,
        "updates": training_updates,
        "validations": validations,
        "best_fidelity_candidate": best_fidelity_key,
        "best_fidelity_validation": {
            name: value
            for name, value in archive[best_fidelity_key].items()
            if name != "parameters"
        },
        "best_flight_candidate": best_flight_key,
        "selected_candidate": selected_key,
        "selected_validation": {
            name: value for name, value in archive[selected_key].items() if name != "parameters"
        },
        "selected_parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_candidate_vector_file": stable_path(vector_path),
        "selected_candidate_vector_file_sha256": file_sha256(vector_path),
        "parameter_change": parameter_change_summary(student, initial_parameters),
        "fresh_selected_student_collection": fresh_student_history.summary,
        "fresh_selected_student_history_action": final_action,
        "fresh_action_fidelity_passed": final_action_passed,
        "paired_final": {
            "overall_success_difference": paired_overall,
            "light_success_difference": paired_light,
            "heavy_success_difference": paired_heavy,
        },
        "acceleration_controls": acceleration_controls,
        "acceleration_dependence_demonstrated": acceleration_dependence,
        "final": final,
        "promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": (
            file_sha256(candidate_path) if promotion["passed"] else None
        ),
        "goal_passed": goal_passed,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "updates_completed": len(training_updates),
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
