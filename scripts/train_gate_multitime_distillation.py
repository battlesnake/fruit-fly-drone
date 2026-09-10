#!/usr/bin/env python3
"""Distill an analytical gate teacher into the native recurrent fly."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_analytic_teachers import (  # noqa: E402
    evaluate_teacher_takeover,
    teacher_rc_for_mode,
)
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
)
from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_acceleration_oracle_distillation import delta_metrics  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    Trajectories,
    _episode_summary,
    compact_flight,
    controller_parameter_sha256,
    controller_step,
    evaluate_controller,
    load_frozen_controller,
    parameter_change_summary,
    parameter_snapshot,
    restore_parameters,
)
from train_gate_recurrent_ppo import initialize_outcomes, update_outcomes  # noqa: E402

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)


@dataclass
class MultiTimeDataset:
    images: Tensor
    roll_pitch: Tensor
    specific_force: Tensor
    stick_position: Tensor
    valid_at_horizons: Tensor
    source_motor: Tensor
    source_anchor: Tensor
    valid_anchor: Tensor
    target_correction: Tensor
    mass_scale: Tensor
    codes: Tensor
    source_summary: dict[str, Any]
    collection_policy_sha256: str


@dataclass(frozen=True)
class HorizonScales:
    contrast_squared: Tensor
    mean_squared: Tensor
    axis_squared: Tensor
    anchor_squared: Tensor


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
        "--teacher-spec",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-analytic-teacher-reserve-v1" / "candidate.json"),
    )
    parser.add_argument(
        "--allow-exact-mass-teacher",
        action="store_true",
        help="allow an explicitly selected privileged-mass teacher for diagnostics",
    )
    parser.add_argument(
        "--warm-start-vector",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "gate-conditional-overfit-diagnostic-v1"
        / "candidate-vector.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "multitime-reserve-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--updates-per-round", type=int, default=200)
    parser.add_argument("--collection-episodes", type=int, default=64)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument("--prefix-seconds", type=float, default=5.0)
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        nargs="+",
        default=(0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00),
    )
    parser.add_argument("--anchor-seconds", type=float, default=0.20)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--preflight-takeover-seconds", type=float, default=0.50)
    parser.add_argument("--preflight-minimum-success", type=float, default=0.90)
    parser.add_argument("--edge-bias-learning-rates", type=float, nargs=2, default=(3.0e-4, 1.0e-4))
    parser.add_argument("--time-constant-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--axis-action-weight", type=float, default=0.25)
    parser.add_argument("--axis-action-scale", type=float, default=0.01)
    parser.add_argument("--throttle-action-scale", type=float, default=0.01)
    parser.add_argument("--regularization-weight", type=float, default=1.0e-6)
    parser.add_argument("--gradient-norm-cap", type=float, default=0.5)
    parser.add_argument("--selection-interval", type=int, default=25)
    parser.add_argument("--holdout-episodes", type=int, default=64)
    parser.add_argument("--action-fidelity-threshold", type=float, default=0.25)
    parser.add_argument("--checkpoint-update", type=int, default=200)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--final-seconds", type=float, default=12.0)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--final-heavy-margin", type=float, default=0.02)
    parser.add_argument(
        "--gradient-check-scales", type=float, nargs=3, default=(1.0e-3, 3.0e-4, 1.0e-4)
    )
    parser.add_argument("--gradient-check-tolerance", type=float, default=0.10)
    parser.add_argument("--collection-seed", type=int, default=990_031)
    parser.add_argument("--holdout-seed", type=int, default=991_031)
    parser.add_argument("--final-seed", type=int, default=993_031)
    parser.add_argument("--optimization-seed", type=int, default=994_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> tuple[int, ...]:
    for path in (
        args.graph,
        args.student_checkpoint,
        args.teacher_spec,
        args.warm_start_vector,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.rounds,
        args.updates_per_round,
        args.collection_episodes,
        args.batch_pairs,
        args.prefix_seconds,
        args.anchor_seconds,
        args.anchor_weight,
        args.preflight_takeover_seconds,
        args.preflight_minimum_success,
        args.time_constant_learning_rate,
        args.axis_action_weight,
        args.axis_action_scale,
        args.throttle_action_scale,
        args.regularization_weight,
        args.gradient_norm_cap,
        args.selection_interval,
        args.holdout_episodes,
        args.action_fidelity_threshold,
        args.checkpoint_update,
        args.final_episodes,
        args.final_seconds,
        args.final_light_improvement,
        args.final_heavy_margin,
        args.gradient_check_tolerance,
        *args.horizon_seconds,
        *args.edge_bias_learning_rates,
        *args.gradient_check_scales,
    )
    if min(positive) <= 0.0:
        raise SystemExit("training sizes, rates, times, and thresholds must be positive")
    total_updates = args.rounds * args.updates_per_round
    if total_updates != 400 or args.rounds != 2:
        raise SystemExit("the bounded protocol requires two rounds and exactly 400 updates")
    if args.checkpoint_update > total_updates:
        raise SystemExit("checkpoint update exceeds the training budget")
    if args.checkpoint_update != args.updates_per_round:
        raise SystemExit("the fidelity checkpoint must close the first round")
    if args.checkpoint_update % args.selection_interval:
        raise SystemExit("checkpoint update must coincide with selection")
    if args.prefix_seconds != max(args.horizon_seconds):
        raise SystemExit("the recurrent prefix must end at the last supervised horizon")
    if sorted(set(args.horizon_seconds)) != list(args.horizon_seconds):
        raise SystemExit("horizons must be strictly increasing")
    if args.anchor_seconds >= min(args.horizon_seconds):
        raise SystemExit("the reference-action anchor must precede supervised horizons")
    if args.preflight_takeover_seconds != min(args.horizon_seconds):
        raise SystemExit("the teacher preflight must begin at the first supervised horizon")
    if args.preflight_takeover_seconds >= args.final_seconds:
        raise SystemExit("the preflight takeover must precede the end of evaluation")
    if args.preflight_minimum_success > 1.0:
        raise SystemExit("the preflight success threshold cannot exceed one")
    for name in (
        "collection_episodes",
        "holdout_episodes",
        "final_episodes",
    ):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")
    if 2 * args.batch_pairs > args.collection_episodes:
        raise SystemExit("the batch cannot contain more matched pairs than collection")
    horizon_steps = tuple(round(value / dt) for value in args.horizon_seconds)
    if horizon_steps[-1] != round(args.prefix_seconds / dt):
        raise SystemExit("horizon rounding does not reach the complete prefix")
    return horizon_steps


@torch.no_grad()
def replay_source_motors(
    source: ConnectomeController,
    trajectories: Trajectories,
    *,
    requested_steps: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    episodes = trajectories.mass_scale.numel()
    neural = source.initial_state(episodes, device=device, dtype=torch.float32)
    requested = set(requested_steps)
    outputs = []
    for step in range(max(requested_steps)):
        motor, neural = controller_step(
            source,
            trajectories.images[step].to(device),
            trajectories.roll_pitch[step].to(device),
            neural,
            trajectories.specific_force[step].to(device),
            trajectories.stick_position[step].to(device),
        )
        if step + 1 in requested:
            outputs.append(motor)
    return torch.stack(outputs)


def make_dataset(
    trajectories: Trajectories,
    source: ConnectomeController,
    *,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
    device: torch.device,
) -> MultiTimeDataset:
    source_outputs = replay_source_motors(
        source,
        trajectories,
        requested_steps=(anchor_step, *horizon_steps),
        device=device,
    )
    indices = torch.tensor([step - 1 for step in horizon_steps], dtype=torch.long)
    oracle = trajectories.oracle_motor[indices].to(device)
    return MultiTimeDataset(
        images=trajectories.images[: horizon_steps[-1]].to(device),
        roll_pitch=trajectories.roll_pitch[: horizon_steps[-1]].to(device),
        specific_force=trajectories.specific_force[: horizon_steps[-1]].to(device),
        stick_position=trajectories.stick_position[: horizon_steps[-1]].to(device),
        valid_at_horizons=trajectories.valid[indices].to(device),
        source_motor=source_outputs[1:],
        source_anchor=source_outputs[0],
        valid_anchor=trajectories.valid[anchor_step - 1].to(device),
        target_correction=oracle - source_outputs[1:],
        mass_scale=trajectories.mass_scale.to(device),
        codes=trajectories.codes.to(device),
        source_summary=trajectories.summary,
        collection_policy_sha256=trajectories.collection_policy_sha256,
    )


@torch.no_grad()
def collect_analytic_trajectories(
    behavior: ConnectomeController,
    teacher_interface: ConnectomeController,
    cases: BalancedCases,
    *,
    teacher_mode: str,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> Trajectories:
    """Label one behavior policy's histories with exact analytical-teacher actions."""

    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    behavior_neural = behavior.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    active = torch.ones(episodes, dtype=torch.bool, device=device)
    images = torch.empty(steps, episodes, resolution, resolution, dtype=torch.float32)
    roll_pitch = torch.empty(steps, episodes, 2, dtype=torch.float32)
    specific_force = torch.empty(steps, episodes, 3, dtype=torch.float32)
    stick_position = torch.empty(steps, episodes, 4, dtype=torch.float32)
    oracle_motor = torch.empty(steps, episodes, 4, dtype=torch.float32)
    reference_motor = torch.empty(steps, episodes, 4, dtype=torch.float32)
    valid = torch.empty(steps, episodes, dtype=torch.bool)

    def cpu(value: Tensor) -> Tensor:
        return value.detach().cpu().contiguous()

    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        images[step].copy_(cpu(image))
        roll_pitch[step].copy_(cpu(state.euler[:, :2]))
        specific_force[step].copy_(cpu(state.specific_force))
        stick_position[step].copy_(cpu(stick_state.position))
        valid[step].copy_(cpu(active))
        behavior_motor, behavior_neural = controller_step(
            behavior,
            image,
            state.euler[:, :2],
            behavior_neural,
            state.specific_force,
            stick_state.position,
        )
        target_rc = teacher_rc_for_mode(
            teacher_mode,
            teacher_interface,
            state,
            cases.gate,
            cases.mass_scale,
            hover_config,
        )
        target_motor = motor_target_for_rc(target_rc, hover_config)
        oracle_motor[step].copy_(cpu(target_motor))
        reference_motor[step].copy_(cpu(behavior_motor))
        rc, stick_state = sticks(behavior_motor, stick_state)
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
    summary["teacher_mode"] = teacher_mode
    return Trajectories(
        images=images,
        roll_pitch=roll_pitch,
        specific_force=specific_force,
        stick_position=stick_position,
        oracle_motor=oracle_motor,
        reference_motor=reference_motor,
        valid=valid,
        mass_scale=cpu(cases.mass_scale),
        codes=cpu(cases.stratum_code),
        collection_policy_sha256=controller_parameter_sha256(behavior),
        teacher_drives_physics=False,
        summary=summary,
    )


def select_pairs(
    dataset: MultiTimeDataset,
    *,
    pairs: int,
    horizon_index: int,
    generator: torch.Generator,
) -> MultiTimeDataset:
    pair_valid = (
        dataset.valid_at_horizons[horizon_index, 0::2]
        & dataset.valid_at_horizons[horizon_index, 1::2]
    ).cpu()
    available = torch.nonzero(pair_valid, as_tuple=False).flatten()
    if not len(available):
        raise RuntimeError("selected horizon has no complete valid pair")
    order = torch.randperm(len(available), generator=generator)
    choices = available[order]
    if len(choices) < pairs:
        repeat = torch.randint(len(available), (pairs - len(choices),), generator=generator)
        choices = torch.cat((choices, available[repeat]))
    choices = choices[:pairs]
    episodes = (
        torch.stack((2 * choices, 2 * choices + 1), dim=1).flatten().to(dataset.images.device)
    )
    return MultiTimeDataset(
        images=dataset.images[:, episodes],
        roll_pitch=dataset.roll_pitch[:, episodes],
        specific_force=dataset.specific_force[:, episodes],
        stick_position=dataset.stick_position[:, episodes],
        valid_at_horizons=dataset.valid_at_horizons[:, episodes],
        source_motor=dataset.source_motor[:, episodes],
        source_anchor=dataset.source_anchor[episodes],
        valid_anchor=dataset.valid_anchor[episodes],
        target_correction=dataset.target_correction[:, episodes],
        mass_scale=dataset.mass_scale[episodes],
        codes=dataset.codes[episodes],
        source_summary=dataset.source_summary,
        collection_policy_sha256=dataset.collection_policy_sha256,
    )


def concatenate_pair_batches(first: MultiTimeDataset, second: MultiTimeDataset) -> MultiTimeDataset:
    def episodes(name: str, dimension: int) -> Tensor:
        return torch.cat((getattr(first, name), getattr(second, name)), dim=dimension)

    return MultiTimeDataset(
        images=episodes("images", 1),
        roll_pitch=episodes("roll_pitch", 1),
        specific_force=episodes("specific_force", 1),
        stick_position=episodes("stick_position", 1),
        valid_at_horizons=episodes("valid_at_horizons", 1),
        source_motor=episodes("source_motor", 1),
        source_anchor=episodes("source_anchor", 0),
        valid_anchor=episodes("valid_anchor", 0),
        target_correction=episodes("target_correction", 1),
        mass_scale=episodes("mass_scale", 0),
        codes=episodes("codes", 0),
        source_summary={"mixture": "equal original and current-student histories"},
        collection_policy_sha256=(
            f"{first.collection_policy_sha256}+{second.collection_policy_sha256}"
        ),
    )


def target_scales(
    dataset: MultiTimeDataset,
    *,
    axis_action_floor: float,
    throttle_action_floor: float,
) -> HorizonScales:
    target = dataset.target_correction[:, :, 3]
    contrast = target[:, 0::2] - target[:, 1::2]
    pair_mean = 0.5 * (target[:, 0::2] + target[:, 1::2])
    pair_valid = dataset.valid_at_horizons[:, 0::2] & dataset.valid_at_horizons[:, 1::2]
    contrast_squared = []
    mean_squared = []
    axis_squared = []
    for horizon in range(len(target)):
        mask = pair_valid[horizon]
        if not bool(mask.any()):
            raise RuntimeError("scale dataset has no complete valid pair at a horizon")
        contrast_squared.append(
            contrast[horizon, mask].square().mean().clamp_min(throttle_action_floor**2)
        )
        mean_squared.append(
            pair_mean[horizon, mask].square().mean().clamp_min(throttle_action_floor**2)
        )
        valid = dataset.valid_at_horizons[horizon]
        if not bool(valid.any()):
            raise RuntimeError("scale dataset has no valid episode at a horizon")
        oracle_action = (
            dataset.source_motor[horizon, valid, :3] + dataset.target_correction[horizon, valid, :3]
        )
        axis_squared.append(oracle_action.square().mean(dim=0).clamp_min(axis_action_floor**2))
    if not bool(dataset.valid_anchor.any()):
        raise RuntimeError("scale dataset has no valid episode at the anchor")
    anchor_squared = (
        dataset.source_anchor[dataset.valid_anchor]
        .std(dim=0, unbiased=False)
        .clamp_min(axis_action_floor)
        .square()
    )
    scales = HorizonScales(
        torch.stack(contrast_squared),
        torch.stack(mean_squared),
        torch.stack(axis_squared),
        anchor_squared,
    )
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            scales.contrast_squared,
            scales.mean_squared,
            scales.axis_squared,
            scales.anchor_squared,
        )
    ):
        raise RuntimeError("nonfinite fixed target scale")
    return scales


def replay_multitime(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    *,
    horizon_steps: tuple[int, ...],
) -> Tensor:
    neural = controller.initial_state(
        dataset.mass_scale.numel(),
        device=dataset.images.device,
        dtype=dataset.images.dtype,
    )
    requested = set(horizon_steps)
    outputs = []
    for step in range(horizon_steps[-1]):
        motor, neural = controller_step(
            controller,
            dataset.images[step],
            dataset.roll_pitch[step],
            neural,
            dataset.specific_force[step],
            dataset.stick_position[step],
        )
        if step + 1 in requested:
            outputs.append(motor)
    return torch.stack(outputs)


def replay_multitime_and_anchor(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    *,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
) -> tuple[Tensor, Tensor]:
    neural = controller.initial_state(
        dataset.mass_scale.numel(),
        device=dataset.images.device,
        dtype=dataset.images.dtype,
    )
    requested = set(horizon_steps)
    outputs = []
    anchor = None
    for step in range(horizon_steps[-1]):
        motor, neural = controller_step(
            controller,
            dataset.images[step],
            dataset.roll_pitch[step],
            neural,
            dataset.specific_force[step],
            dataset.stick_position[step],
        )
        completed = step + 1
        if completed == anchor_step:
            anchor = motor
        if completed in requested:
            outputs.append(motor)
    if anchor is None or len(outputs) != len(horizon_steps):
        raise RuntimeError("joint replay did not reach every endpoint and its anchor")
    return torch.stack(outputs), anchor


def replay_endpoint_and_anchor(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    *,
    endpoint_step: int,
    anchor_step: int,
) -> tuple[Tensor, Tensor]:
    neural = controller.initial_state(
        dataset.mass_scale.numel(),
        device=dataset.images.device,
        dtype=dataset.images.dtype,
    )
    anchor = None
    endpoint = None
    for step in range(endpoint_step):
        motor, neural = controller_step(
            controller,
            dataset.images[step],
            dataset.roll_pitch[step],
            neural,
            dataset.specific_force[step],
            dataset.stick_position[step],
        )
        if step + 1 == anchor_step:
            anchor = motor
        if step + 1 == endpoint_step:
            endpoint = motor
    if anchor is None or endpoint is None:
        raise RuntimeError("replay did not reach its anchor and endpoint")
    return endpoint, anchor


def multitime_loss(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    scales: HorizonScales,
    initial_parameters: dict[str, Tensor],
    *,
    horizon_index: int,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
    axis_action_weight: float,
    anchor_weight: float,
    regularization_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    motor, anchor_motor = replay_endpoint_and_anchor(
        controller,
        dataset,
        endpoint_step=horizon_steps[horizon_index],
        anchor_step=anchor_step,
    )
    valid = dataset.valid_at_horizons[horizon_index]
    pair_valid = valid[0::2] & valid[1::2]
    if not bool(pair_valid.any()):
        raise RuntimeError("training batch contains no valid pair")
    prediction = motor - dataset.source_motor[horizon_index]
    target = dataset.target_correction[horizon_index]
    prediction_contrast = prediction[0::2, 3] - prediction[1::2, 3]
    target_contrast = target[0::2, 3] - target[1::2, 3]
    prediction_mean = 0.5 * (prediction[0::2, 3] + prediction[1::2, 3])
    target_mean = 0.5 * (target[0::2, 3] + target[1::2, 3])
    contrast_loss = (
        prediction_contrast[pair_valid] - target_contrast[pair_valid]
    ).square().mean() / scales.contrast_squared[horizon_index]
    mean_loss = (
        prediction_mean[pair_valid] - target_mean[pair_valid]
    ).square().mean() / scales.mean_squared[horizon_index]
    axis_loss = (
        ((prediction[valid, :3] - target[valid, :3]) / scales.axis_squared[horizon_index].sqrt())
        .square()
        .mean()
    )
    anchor_valid = dataset.valid_anchor
    anchor_loss = (
        (
            (anchor_motor[anchor_valid] - dataset.source_anchor[anchor_valid])
            / scales.anchor_squared.sqrt()
        )
        .square()
        .mean()
    )
    edge_scale = initial_parameters["edge_magnitude"].clamp_min(0.05)
    regularization = (
        ((controller.edge_magnitude - initial_parameters["edge_magnitude"]) / edge_scale)
        .square()
        .mean()
        + (controller.bias - initial_parameters["bias"]).square().mean()
        + (controller.raw_time_constant - initial_parameters["raw_time_constant"]).square().mean()
    )
    total = (
        contrast_loss
        + mean_loss
        + axis_action_weight * axis_loss
        + anchor_weight * anchor_loss
        + regularization_weight * regularization
    )
    return total, {
        "total_loss": float(total.detach()),
        "contrast_normalized_mse": float(contrast_loss.detach()),
        "mean_normalized_mse": float(mean_loss.detach()),
        "axis_teacher_action_normalized_mse": float(axis_loss.detach()),
        "anchor_normalized_mse": float(anchor_loss.detach()),
        "regularization": float(regularization.detach()),
        "valid_pair_fraction": float(pair_valid.float().mean()),
    }


def joint_multitime_loss(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    scales: HorizonScales,
    initial_parameters: dict[str, Tensor],
    *,
    horizon_steps: tuple[int, ...],
    horizon_weights: tuple[float, ...],
    anchor_step: int,
    axis_action_weight: float,
    anchor_weight: float,
    regularization_weight: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Average every endpoint's loss from one complete recurrent replay."""

    if len(horizon_steps) != len(horizon_weights):
        raise ValueError("every horizon requires one fixed weight")
    motors, anchor_motor = replay_multitime_and_anchor(
        controller,
        dataset,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
    )
    weighted_endpoint = motors.sum() * 0.0
    included_weight = 0.0
    endpoint_details = []
    for index, (step, weight) in enumerate(zip(horizon_steps, horizon_weights, strict=True)):
        valid = dataset.valid_at_horizons[index]
        pair_valid = valid[0::2] & valid[1::2]
        if not bool(valid.any()) or not bool(pair_valid.any()):
            endpoint_details.append(
                {
                    "horizon_seconds": step * 0.01,
                    "included": False,
                    "valid_episode_count": int(valid.sum()),
                    "valid_pair_count": int(pair_valid.sum()),
                }
            )
            continue
        prediction = motors[index] - dataset.source_motor[index]
        target = dataset.target_correction[index]
        prediction_contrast = prediction[0::2, 3] - prediction[1::2, 3]
        target_contrast = target[0::2, 3] - target[1::2, 3]
        prediction_mean = 0.5 * (prediction[0::2, 3] + prediction[1::2, 3])
        target_mean = 0.5 * (target[0::2, 3] + target[1::2, 3])
        contrast_loss = (
            prediction_contrast[pair_valid] - target_contrast[pair_valid]
        ).square().mean() / scales.contrast_squared[index]
        mean_loss = (
            prediction_mean[pair_valid] - target_mean[pair_valid]
        ).square().mean() / scales.mean_squared[index]
        axis_loss = (
            ((prediction[valid, :3] - target[valid, :3]) / scales.axis_squared[index].sqrt())
            .square()
            .mean()
        )
        endpoint_loss = contrast_loss + mean_loss + axis_action_weight * axis_loss
        weighted_endpoint = weighted_endpoint + weight * endpoint_loss
        included_weight += weight
        endpoint_details.append(
            {
                "horizon_seconds": step * 0.01,
                "included": True,
                "weight": weight,
                "contrast_normalized_mse": float(contrast_loss.detach()),
                "mean_normalized_mse": float(mean_loss.detach()),
                "axis_teacher_action_normalized_mse": float(axis_loss.detach()),
                "valid_episode_count": int(valid.sum()),
                "valid_pair_count": int(pair_valid.sum()),
            }
        )
    if included_weight <= 0.0:
        raise RuntimeError("joint batch has no supervised endpoint")
    endpoint_average = weighted_endpoint / included_weight
    anchor_valid = dataset.valid_anchor
    if not bool(anchor_valid.any()):
        raise RuntimeError("joint batch has no valid anchor")
    anchor_loss = (
        (
            (anchor_motor[anchor_valid] - dataset.source_anchor[anchor_valid])
            / scales.anchor_squared.sqrt()
        )
        .square()
        .mean()
    )
    edge_scale = initial_parameters["edge_magnitude"].clamp_min(0.05)
    regularization = (
        ((controller.edge_magnitude - initial_parameters["edge_magnitude"]) / edge_scale)
        .square()
        .mean()
        + (controller.bias - initial_parameters["bias"]).square().mean()
        + (controller.raw_time_constant - initial_parameters["raw_time_constant"]).square().mean()
    )
    total = endpoint_average + anchor_weight * anchor_loss + regularization_weight * regularization
    return total, {
        "total_loss": float(total.detach()),
        "weighted_endpoint_loss": float(endpoint_average.detach()),
        "included_horizon_weight": included_weight,
        "anchor_normalized_mse": float(anchor_loss.detach()),
        "regularization": float(regularization.detach()),
        "endpoints": endpoint_details,
    }


@torch.no_grad()
def fidelity_metrics(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    scales: HorizonScales,
    *,
    horizon_steps: tuple[int, ...],
    dt: float,
) -> dict[str, Any]:
    def safe_delta(prediction: Tensor, target: Tensor) -> dict[str, Any] | None:
        if prediction.numel() == 0:
            return None
        return delta_metrics(prediction, target)

    def safe_mean(values: Tensor) -> float | None:
        if values.numel() == 0:
            return None
        result = float(values.mean())
        return result if math.isfinite(result) else None

    motors = replay_multitime(controller, dataset, horizon_steps=horizon_steps)
    report = {}
    for index, step in enumerate(horizon_steps):
        valid = dataset.valid_at_horizons[index]
        pair_valid = valid[0::2] & valid[1::2]
        prediction = motors[index] - dataset.source_motor[index]
        target = dataset.target_correction[index]
        prediction_contrast = prediction[0::2, 3] - prediction[1::2, 3]
        target_contrast = target[0::2, 3] - target[1::2, 3]
        prediction_mean = 0.5 * (prediction[0::2, 3] + prediction[1::2, 3])
        target_mean = 0.5 * (target[0::2, 3] + target[1::2, 3])
        pair_count = int(pair_valid.sum())
        valid_count = int(valid.sum())
        contrast_error = None
        mean_error = None
        if pair_count:
            contrast_error = float(
                (prediction_contrast[pair_valid] - target_contrast[pair_valid])
                .square()
                .mean()
                .sqrt()
                / scales.contrast_squared[index].sqrt()
            )
            mean_error = float(
                (prediction_mean[pair_valid] - target_mean[pair_valid]).square().mean().sqrt()
                / scales.mean_squared[index].sqrt()
            )
        axis_error: dict[str, float | None]
        axis_raw_error: dict[str, float | None]
        if valid_count:
            raw_values = (prediction[valid, :3] - target[valid, :3]).square().mean(dim=0).sqrt()
            values = raw_values / scales.axis_squared[index].sqrt()
            axis_error = {
                name: float(values[axis]) for axis, name in enumerate(("roll", "pitch", "yaw"))
            }
            axis_raw_error = {
                name: float(raw_values[axis]) for axis, name in enumerate(("roll", "pitch", "yaw"))
            }
        else:
            axis_error = {name: None for name in ("roll", "pitch", "yaw")}
            axis_raw_error = {name: None for name in ("roll", "pitch", "yaw")}
        lower = dataset.mass_scale < 1.0
        report[f"{step * dt:.2f}"] = {
            "valid_episode_count": valid_count,
            "valid_matched_pair_count": pair_count,
            "contrast_normalized_rmse": contrast_error,
            "mean_normalized_rmse": mean_error,
            "axis_teacher_action_normalized_rmse": axis_error,
            "axis_teacher_action_rmse": axis_raw_error,
            "contrast": safe_delta(prediction_contrast[pair_valid], target_contrast[pair_valid]),
            "pair_mean": safe_delta(prediction_mean[pair_valid], target_mean[pair_valid]),
            "light_throttle_correction_mean": safe_mean(prediction[valid & lower, 3]),
            "light_throttle_target_mean": safe_mean(target[valid & lower, 3]),
            "heavy_throttle_correction_mean": safe_mean(prediction[valid & ~lower, 3]),
            "heavy_throttle_target_mean": safe_mean(target[valid & ~lower, 3]),
        }
    return report


def fidelity_score(metrics: dict[str, Any]) -> float:
    values: list[float | None] = []
    for horizon in metrics.values():
        values.extend((horizon["contrast_normalized_rmse"], horizon["mean_normalized_rmse"]))
        values.extend(horizon["axis_teacher_action_normalized_rmse"].values())
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return float("inf")
    return max(value for value in values if value is not None)


def fidelity_margin_score(metrics: dict[str, Any], *, threshold: float) -> float:
    """Worst fidelity error divided by its preregistered acceptance limit."""

    values: list[float] = []
    for time, horizon in metrics.items():
        if horizon["valid_episode_count"] == 0 or horizon["valid_matched_pair_count"] == 0:
            return float("inf")
        contrast_limit = 0.20 if time in {"0.50", "0.75"} else threshold
        components = (
            (horizon["contrast_normalized_rmse"], contrast_limit),
            (horizon["mean_normalized_rmse"], threshold),
            *(
                (value, threshold)
                for value in horizon["axis_teacher_action_normalized_rmse"].values()
            ),
        )
        if any(value is None or not math.isfinite(value) for value, _ in components):
            return float("inf")
        values.extend(value / limit for value, limit in components if value is not None)
    return max(values, default=float("inf"))


def fidelity_passed(metrics: dict[str, Any], *, threshold: float) -> bool:
    for time, horizon in metrics.items():
        if horizon["valid_episode_count"] == 0 or horizon["valid_matched_pair_count"] == 0:
            return False
        contrast_limit = 0.20 if time in {"0.50", "0.75"} else threshold
        required = [
            horizon["contrast_normalized_rmse"],
            horizon["mean_normalized_rmse"],
            *horizon["axis_teacher_action_normalized_rmse"].values(),
        ]
        if any(value is None or not math.isfinite(value) for value in required):
            return False
        if horizon["contrast_normalized_rmse"] > contrast_limit:
            return False
        if horizon["mean_normalized_rmse"] > threshold:
            return False
        if max(horizon["axis_teacher_action_normalized_rmse"].values()) > threshold:
            return False
    return True


def load_warm_start(
    controller: ConnectomeController,
    vector_path: Path,
    *,
    source_checkpoint: Path,
) -> dict[str, Any]:
    vector = json.loads(vector_path.read_text())
    if vector.get("source_checkpoint_sha256") != file_sha256(source_checkpoint):
        raise SystemExit("warm-start vector has a different source checkpoint")
    state = {
        "edge_magnitude": torch.tensor(
            vector["edge_magnitude"], device=controller.edge_magnitude.device
        ),
        "bias": torch.tensor(vector["bias"], device=controller.bias.device),
        "raw_time_constant": torch.tensor(
            vector["raw_time_constant"], device=controller.raw_time_constant.device
        ),
    }
    if any(
        state[name].shape != parameter.shape
        for name, parameter in {
            "edge_magnitude": controller.edge_magnitude,
            "bias": controller.bias,
            "raw_time_constant": controller.raw_time_constant,
        }.items()
    ):
        raise SystemExit("warm-start vector shape does not match the graph")
    restore_parameters(controller, state)
    if controller_parameter_sha256(controller) != vector["parameter_vector_sha256"]:
        raise SystemExit("warm-start parameter hash does not round-trip")
    return vector


def gradient_audit(
    controller: ConnectomeController,
    dataset: MultiTimeDataset,
    scales: HorizonScales,
    initial_parameters: dict[str, Tensor],
    *,
    horizon_index: int,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parameters = {
        "edge_magnitude": controller.edge_magnitude,
        "bias": controller.bias,
        "raw_time_constant": controller.raw_time_constant,
    }
    baseline = parameter_snapshot(controller)
    repeats = []
    with torch.no_grad():
        for _ in range(3):
            repeats.append(
                float(
                    multitime_loss(
                        controller,
                        dataset,
                        scales,
                        initial_parameters,
                        horizon_index=horizon_index,
                        horizon_steps=horizon_steps,
                        anchor_step=anchor_step,
                        axis_action_weight=args.axis_action_weight,
                        anchor_weight=args.anchor_weight,
                        regularization_weight=args.regularization_weight,
                    )[0]
                )
            )
    repeat_noise = max(repeats) - min(repeats)
    controller.zero_grad(set_to_none=True)
    loss, _ = multitime_loss(
        controller,
        dataset,
        scales,
        initial_parameters,
        horizon_index=horizon_index,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        axis_action_weight=args.axis_action_weight,
        anchor_weight=args.anchor_weight,
        regularization_weight=args.regularization_weight,
    )
    loss.backward()
    report: dict[str, Any] = {}
    try:
        for name, parameter in parameters.items():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing or nonfinite longest-prefix gradient for {name}")
            direction = parameter.grad.detach().sign()
            if name == "edge_magnitude":
                direction[(parameter.detach() <= 0.01) | (parameter.detach() >= 7.99)] = 0.0
            analytic = float((parameter.grad * direction).sum())
            checks = []
            for epsilon in args.gradient_check_scales:
                with torch.no_grad():
                    parameter.copy_(baseline[name] + epsilon * direction)
                    plus = float(
                        multitime_loss(
                            controller,
                            dataset,
                            scales,
                            initial_parameters,
                            horizon_index=horizon_index,
                            horizon_steps=horizon_steps,
                            anchor_step=anchor_step,
                            axis_action_weight=args.axis_action_weight,
                            anchor_weight=args.anchor_weight,
                            regularization_weight=args.regularization_weight,
                        )[0]
                    )
                    parameter.copy_(baseline[name] - epsilon * direction)
                    minus = float(
                        multitime_loss(
                            controller,
                            dataset,
                            scales,
                            initial_parameters,
                            horizon_index=horizon_index,
                            horizon_steps=horizon_steps,
                            anchor_step=anchor_step,
                            axis_action_weight=args.axis_action_weight,
                            anchor_weight=args.anchor_weight,
                            regularization_weight=args.regularization_weight,
                        )[0]
                    )
                    parameter.copy_(baseline[name])
                derivative = (plus - minus) / (2.0 * epsilon)
                relative_error = abs(derivative - analytic) / max(
                    abs(derivative), abs(analytic), 1.0e-10
                )
                checks.append(
                    {
                        "epsilon": epsilon,
                        "plus_loss": plus,
                        "minus_loss": minus,
                        "central_difference": derivative,
                        "relative_error": relative_error,
                        "matching_sign": derivative * analytic > 0.0,
                        "measurable_above_repeat_noise": abs(plus - minus)
                        > max(10.0 * repeat_noise, 1.0e-10),
                    }
                )
            adjacent = [
                left["matching_sign"]
                and right["matching_sign"]
                and left["measurable_above_repeat_noise"]
                and right["measurable_above_repeat_noise"]
                and left["relative_error"] <= args.gradient_check_tolerance
                and right["relative_error"] <= args.gradient_check_tolerance
                for left, right in zip(checks[:-1], checks[1:], strict=True)
            ]
            report[name] = {
                "gradient_l2_norm": float(parameter.grad.norm()),
                "analytic_directional_derivative": analytic,
                "checks": checks,
                "passed": any(adjacent),
            }
            if not report[name]["passed"]:
                raise RuntimeError(f"longest-prefix gradient audit failed for {name}: {checks}")
    finally:
        restore_parameters(controller, baseline)
        controller.zero_grad(set_to_none=True)
    return {
        "horizon_seconds": horizon_steps[horizon_index] * 0.01,
        "full_prefix_steps": horizon_steps[horizon_index],
        "complete_prefix_in_autograd": True,
        "unchanged_loss_repeats": repeats,
        "unchanged_loss_range": repeat_noise,
        "families": report,
        "passed": all(item["passed"] for item in report.values()),
    }


def train_round(
    controller: ConnectomeController,
    original: MultiTimeDataset,
    current: MultiTimeDataset | None,
    holdout: MultiTimeDataset,
    scales: HorizonScales,
    initial_parameters: dict[str, Tensor],
    archive: dict[str, dict[str, Any]],
    *,
    round_index: int,
    learning_rate: float,
    horizon_steps: tuple[int, ...],
    anchor_step: int,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    optimizer = torch.optim.Adam(
        (
            {
                "params": [controller.edge_magnitude, controller.bias],
                "lr": learning_rate,
            },
            {
                "params": [controller.raw_time_constant],
                "lr": args.time_constant_learning_rate,
            },
        )
    )
    horizon_weights = torch.ones(len(horizon_steps), dtype=torch.float32)
    horizon_weights[:2] = 2.0
    updates = []
    selections = []
    round_offset = (round_index - 1) * args.updates_per_round
    for local_update in range(1, args.updates_per_round + 1):
        global_update = round_offset + local_update
        horizon_index = int(torch.multinomial(horizon_weights, 1, generator=generator).item())
        if current is None:
            batch = select_pairs(
                original,
                pairs=args.batch_pairs,
                horizon_index=horizon_index,
                generator=generator,
            )
        else:
            if args.batch_pairs % 2:
                raise RuntimeError("mixed DAgger batches require an even pair count")
            half = args.batch_pairs // 2
            batch = concatenate_pair_batches(
                select_pairs(
                    original,
                    pairs=half,
                    horizon_index=horizon_index,
                    generator=generator,
                ),
                select_pairs(
                    current,
                    pairs=half,
                    horizon_index=horizon_index,
                    generator=generator,
                ),
            )
        optimizer.zero_grad(set_to_none=True)
        loss, details = multitime_loss(
            controller,
            batch,
            scales,
            initial_parameters,
            horizon_index=horizon_index,
            horizon_steps=horizon_steps,
            anchor_step=anchor_step,
            axis_action_weight=args.axis_action_weight,
            anchor_weight=args.anchor_weight,
            regularization_weight=args.regularization_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite multi-time loss at update {global_update}")
        loss.backward()
        raw_gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    parameter.grad.detach().norm()
                    for parameter in controller.parameters()
                    if parameter.grad is not None
                ]
            )
        )
        torch.nn.utils.clip_grad_norm_(
            controller.parameters(), args.gradient_norm_cap, error_if_nonfinite=True
        )
        optimizer.step()
        controller.project_parameters()
        updates.append(
            {
                "update": global_update,
                "round": round_index,
                "horizon_index": horizon_index,
                "horizon_seconds": horizon_steps[horizon_index] * 0.01,
                "learning_rate": learning_rate,
                "raw_gradient_norm": float(raw_gradient_norm),
                **details,
            }
        )
        if local_update % args.selection_interval:
            continue
        metrics = fidelity_metrics(
            controller,
            holdout,
            scales,
            horizon_steps=horizon_steps,
            dt=0.01,
        )
        score = fidelity_score(metrics)
        key = controller_parameter_sha256(controller)
        archive[key] = {
            "update": global_update,
            "round": round_index,
            "score": score,
            "fidelity_passed": fidelity_passed(metrics, threshold=args.action_fidelity_threshold),
            "metrics": metrics,
            "parameters": parameter_snapshot(controller, cpu=True),
        }
        selections.append(
            {name: value for name, value in archive[key].items() if name != "parameters"}
            | {"parameter_vector_sha256": key}
        )
        print(
            json.dumps(
                {
                    "phase": "multi_time_selection",
                    "update": global_update,
                    "round": round_index,
                    "score": score,
                    "fidelity_passed": archive[key]["fidelity_passed"],
                    "early_contrast": {
                        time: metrics[time]["contrast_normalized_rmse"] for time in ("0.50", "0.75")
                    },
                }
            ),
            flush=True,
        )
    round_keys = [key for key, item in archive.items() if item["round"] == round_index]
    selected_key = min(
        round_keys,
        key=lambda key: (not archive[key]["fidelity_passed"], archive[key]["score"]),
    )
    restore_parameters(controller, archive[selected_key]["parameters"])
    return updates, selections, selected_key


def save_analytic_candidate_checkpoint(
    path: Path,
    source_checkpoint: dict[str, Any],
    source_path: Path,
    teacher_spec_path: Path,
    controller: ConnectomeController,
    promotion: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source_checkpoint)
    checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    checkpoint["source_checkpoint_sha256"] = file_sha256(source_path)
    checkpoint["multitime_analytic_teacher_distillation"] = {
        "method": "full-prefix multi-time analytical-teacher action distillation",
        "teacher_spec_sha256": file_sha256(teacher_spec_path),
        "all_native_edge_magnitudes_trainable": True,
        "all_native_biases_trainable": True,
        "all_native_time_constants_trainable": True,
        "fixed_topology": True,
        "fixed_transmitter_signs": True,
        "training_only_privileged_teacher": True,
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
    source, source_checkpoint, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.student_checkpoint, device
    )
    horizon_steps = validate_args(args, hover_config.dt)
    anchor_step = round(args.anchor_seconds / hover_config.dt)
    if not source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("unexpected deployed sensor contract")
    teacher_spec = json.loads(args.teacher_spec.read_text())
    if teacher_spec.get("kind") != "privileged_analytic_gate_teacher":
        raise SystemExit("unexpected analytical teacher specification")
    if not teacher_spec.get("preflight_passed"):
        raise SystemExit("analytical teacher did not pass its frozen preflight")
    if teacher_spec.get("checkpoint_sha256") != file_sha256(args.student_checkpoint):
        raise SystemExit("analytical teacher was validated against a different source checkpoint")
    if teacher_spec.get("takeover_seconds") != args.preflight_takeover_seconds:
        raise SystemExit("analytical teacher was validated at a different takeover time")
    teacher_uses_exact_mass = bool(teacher_spec.get("uses_exact_simulator_mass"))
    if teacher_uses_exact_mass and not args.allow_exact_mass_teacher:
        raise SystemExit(
            "exact-mass teachers require the explicit diagnostic flag --allow-exact-mass-teacher"
        )
    teacher_mode = teacher_spec["teacher_mode"]
    student = copy.deepcopy(source).to(device)
    warm_start = load_warm_start(
        student, args.warm_start_vector, source_checkpoint=args.student_checkpoint
    )
    for parameter in student.parameters():
        parameter.requires_grad_(True)
    initial_parameters = parameter_snapshot(student)
    source_parameters = parameter_snapshot(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate.pt"
    if candidate_path.exists():
        raise SystemExit("output directory contains a stale candidate.pt; use a fresh directory")
    seed_everything(args.optimization_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    original_cases = diverse_matched_cases(
        args.collection_episodes,
        seed=args.collection_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    initial_teacher_rc = teacher_rc_for_mode(
        teacher_mode,
        source,
        original_cases.state,
        original_cases.gate,
        original_cases.mass_scale,
        hover_config,
    )
    swapped_teacher_rc = teacher_rc_for_mode(
        teacher_mode,
        source,
        original_cases.state,
        original_cases.gate,
        original_cases.mass_scale.flip(0),
        hover_config,
    )
    teacher_mass_argument_invariant = torch.equal(initial_teacher_rc, swapped_teacher_rc)
    if not teacher_uses_exact_mass and not teacher_mass_argument_invariant:
        raise SystemExit("mass-free teacher actions changed when only the mass argument changed")
    preflight_target = evaluate_teacher_takeover(
        source,
        original_cases,
        mode=teacher_mode,
        takeover_seconds=args.preflight_takeover_seconds,
        seconds=args.final_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    preflight_rates = {
        "overall": preflight_target["success_rate"],
        "light": preflight_target["light_success_rate"],
        "heavy": preflight_target["heavy_success_rate"],
        "negative_lateral": preflight_target["negative_lateral_success_rate"],
        "positive_lateral": preflight_target["positive_lateral_success_rate"],
        "negative_obliquity": preflight_target["negative_obliquity_success_rate"],
        "positive_obliquity": preflight_target["positive_obliquity_success_rate"],
    }
    preflight_target_passed = all(
        rate >= args.preflight_minimum_success for rate in preflight_rates.values()
    )
    preflight_target["minimum_required_per_stratum"] = args.preflight_minimum_success
    preflight_target["passed"] = preflight_target_passed
    print(
        json.dumps(
            {
                "phase": "analytic_target_preflight",
                "takeover_seconds": args.preflight_takeover_seconds,
                "passed": preflight_target_passed,
                "success_rates": preflight_rates,
            }
        ),
        flush=True,
    )
    if not preflight_target_passed:
        report_path = args.output_dir / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "method": "full-prefix multi-time analytical-teacher distillation",
                    "claim_scope": "Target-policy preflight failed; no training was run.",
                    "graph": stable_path(args.graph),
                    "graph_sha256": file_sha256(args.graph),
                    "student_checkpoint": stable_path(args.student_checkpoint),
                    "student_checkpoint_sha256": file_sha256(args.student_checkpoint),
                    "teacher_spec": stable_path(args.teacher_spec),
                    "teacher_spec_sha256": file_sha256(args.teacher_spec),
                    "analytic_target_preflight": preflight_target,
                    "promotion": {
                        "checks": {"analytic_target_preflight": False},
                        "passed": False,
                    },
                    "goal_passed": False,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    original_trajectories = collect_analytic_trajectories(
        source,
        source,
        original_cases,
        teacher_mode=teacher_mode,
        seconds=args.prefix_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    original = make_dataset(
        original_trajectories,
        source,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        device=device,
    )
    scales = target_scales(
        original,
        axis_action_floor=args.axis_action_scale,
        throttle_action_floor=args.throttle_action_scale,
    )
    print(
        json.dumps(
            {
                "phase": "fixed_target_scales",
                "teacher_mass_argument_invariant": teacher_mass_argument_invariant,
                "contrast_rms": scales.contrast_squared.sqrt().detach().cpu().tolist(),
                "mean_rms": scales.mean_squared.sqrt().detach().cpu().tolist(),
                "axis_rms": scales.axis_squared.sqrt().detach().cpu().tolist(),
            }
        ),
        flush=True,
    )
    holdout_cases = diverse_matched_cases(
        args.holdout_episodes,
        seed=args.holdout_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.0,
    )
    holdout_trajectories = collect_analytic_trajectories(
        source,
        source,
        holdout_cases,
        teacher_mode=teacher_mode,
        seconds=args.prefix_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    holdout = make_dataset(
        holdout_trajectories,
        source,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        device=device,
    )
    initial_holdout_metrics = fidelity_metrics(
        student,
        holdout,
        scales,
        horizon_steps=horizon_steps,
        dt=hover_config.dt,
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(args.optimization_seed)
    audit_batch = select_pairs(
        original,
        pairs=args.batch_pairs,
        horizon_index=len(horizon_steps) - 1,
        generator=batch_generator,
    )
    audit = gradient_audit(
        student,
        audit_batch,
        scales,
        initial_parameters,
        horizon_index=len(horizon_steps) - 1,
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        args=args,
    )
    print(
        json.dumps(
            {
                "phase": "longest_prefix_gradient_audit",
                "horizon_seconds": args.prefix_seconds,
                "passed": audit["passed"],
            }
        ),
        flush=True,
    )

    archive: dict[str, dict[str, Any]] = {}
    all_updates = []
    all_selections = []
    round1_updates, round1_selections, round1_key = train_round(
        student,
        original,
        None,
        holdout,
        scales,
        initial_parameters,
        archive,
        round_index=1,
        learning_rate=args.edge_bias_learning_rates[0],
        horizon_steps=horizon_steps,
        anchor_step=anchor_step,
        generator=batch_generator,
        args=args,
    )
    all_updates.extend(round1_updates)
    all_selections.extend(round1_selections)
    round1_metrics = archive[round1_key]["metrics"]
    round1_gate_passed = fidelity_passed(round1_metrics, threshold=args.action_fidelity_threshold)
    dagger_summary = None
    takeover = None
    round2_key = None
    fresh_metrics = None

    if round1_gate_passed:
        dagger_cases = diverse_matched_cases(
            args.collection_episodes,
            seed=args.collection_seed + 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        dagger_trajectories = collect_analytic_trajectories(
            student,
            source,
            dagger_cases,
            teacher_mode=teacher_mode,
            seconds=args.prefix_seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        dagger_summary = dagger_trajectories.summary
        takeover = {}
        for takeover_time, threshold in ((2.0, 0.60), (3.0, 0.50)):
            result = evaluate_teacher_takeover(
                student,
                dagger_cases,
                mode=teacher_mode,
                takeover_seconds=takeover_time,
                seconds=args.final_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            takeover[f"{takeover_time:.2f}"] = {
                **result,
                "minimum_required": threshold,
                "passed": min(
                    result[key]
                    for key in (
                        "success_rate",
                        "light_success_rate",
                        "heavy_success_rate",
                        "negative_lateral_success_rate",
                        "positive_lateral_success_rate",
                        "negative_obliquity_success_rate",
                        "positive_obliquity_success_rate",
                    )
                )
                >= threshold,
            }
        if all(item["passed"] for item in takeover.values()):
            current = make_dataset(
                dagger_trajectories,
                source,
                horizon_steps=horizon_steps,
                anchor_step=anchor_step,
                device=device,
            )
            round2_updates, round2_selections, round2_key = train_round(
                student,
                original,
                current,
                holdout,
                scales,
                initial_parameters,
                archive,
                round_index=2,
                learning_rate=args.edge_bias_learning_rates[1],
                horizon_steps=horizon_steps,
                anchor_step=anchor_step,
                generator=batch_generator,
                args=args,
            )
            all_updates.extend(round2_updates)
            all_selections.extend(round2_selections)
            fresh_cases = diverse_matched_cases(
                args.holdout_episodes,
                seed=args.holdout_seed + 1,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                extreme_fraction=0.0,
            )
            fresh_trajectories = collect_analytic_trajectories(
                student,
                source,
                fresh_cases,
                teacher_mode=teacher_mode,
                seconds=args.prefix_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            fresh = make_dataset(
                fresh_trajectories,
                source,
                horizon_steps=horizon_steps,
                anchor_step=anchor_step,
                device=device,
            )
            fresh_metrics = fidelity_metrics(
                student,
                fresh,
                scales,
                horizon_steps=horizon_steps,
                dt=hover_config.dt,
            )

    fresh_fidelity_passed = bool(
        fresh_metrics is not None
        and fidelity_passed(fresh_metrics, threshold=args.action_fidelity_threshold)
    )
    final = None
    paired_final = None
    acceleration_controls = None
    acceleration_dependence = False
    promotion = {
        "checks": {"fresh_student_history_fidelity": fresh_fidelity_passed},
        "passed": False,
    }
    goal_threshold_checks = None
    goal_passed = False
    if fresh_fidelity_passed:
        final_cases = diverse_matched_cases(
            args.final_episodes,
            seed=args.final_seed,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        final = {}
        outcomes = {}
        for name, evaluated, controls in (
            ("reference", source, {}),
            ("candidate", student, {}),
            ("candidate_frozen_first_frame", student, {"frozen_visual": True}),
            ("candidate_constant_1g", student, {"constant_acceleration": True}),
            (
                "candidate_pair_swapped_acceleration",
                student,
                {"pair_swapped_acceleration": True},
            ),
        ):
            summary, tensors = evaluate_controller(
                evaluated,
                final_cases,
                seconds=args.final_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
                **controls,
            )
            final[name] = summary
            outcomes[name] = tensors
            print(
                json.dumps({"phase": "final", "name": name, **compact_flight(summary)}),
                flush=True,
            )
        reference_success = outcomes["reference"]["success"]
        candidate_success = outcomes["candidate"]["success"]
        codes = outcomes["reference"]["codes"]
        light = ~codes.bitwise_and(1).bool()
        heavy = ~light
        paired_overall = paired_clustered_confidence_interval(reference_success, candidate_success)
        paired_light = paired_confidence_interval(
            reference_success[light], candidate_success[light]
        )
        paired_heavy = paired_confidence_interval(
            reference_success[heavy], candidate_success[heavy]
        )
        paired_final = {
            "overall_success_difference": paired_overall,
            "light_success_difference": paired_light,
            "heavy_success_difference": paired_heavy,
        }
        acceleration_controls = {
            "live_minus_constant_1g": paired_clustered_confidence_interval(
                outcomes["candidate_constant_1g"]["success"], candidate_success
            ),
            "live_minus_pair_swapped": paired_clustered_confidence_interval(
                outcomes["candidate_pair_swapped_acceleration"]["success"],
                candidate_success,
            ),
        }
        acceleration_dependence = all(
            item["confidence_95"][0] > 0.0 for item in acceleration_controls.values()
        )
        promotion_checks = {
            "fresh_student_history_fidelity": fresh_fidelity_passed,
            "light_improvement_at_least_threshold": (
                final["candidate"]["light_success_rate"]
                >= final["reference"]["light_success_rate"] + args.final_light_improvement
            ),
            "light_paired_confidence_interval_excludes_zero": (
                paired_light["confidence_95"][0] > 0.0
            ),
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
        }
        promotion = {"checks": promotion_checks, "passed": all(promotion_checks.values())}
        if promotion["passed"]:
            save_analytic_candidate_checkpoint(
                candidate_path,
                source_checkpoint,
                args.student_checkpoint,
                args.teacher_spec,
                student,
                promotion,
            )
        goal_threshold_checks = {
            key: final["candidate"][key] >= 0.90
            for key in (
                "success_rate",
                "light_success_rate",
                "heavy_success_rate",
                "negative_lateral_success_rate",
                "positive_lateral_success_rate",
                "negative_obliquity_success_rate",
                "positive_obliquity_success_rate",
            )
        }
        goal_passed = bool(
            promotion["passed"]
            and all(goal_threshold_checks.values())
            and final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
        )

    vector_path = args.output_dir / "candidate-vector.json"
    vector_path.write_text(
        json.dumps(
            {
                "source_checkpoint_sha256": file_sha256(args.student_checkpoint),
                "warm_start_parameter_sha256": warm_start["parameter_vector_sha256"],
                "selected_round": 2 if round2_key is not None else 1,
                "selected_parameter_vector_sha256": controller_parameter_sha256(student),
                "edge_magnitude": student.edge_magnitude.detach().cpu().tolist(),
                "bias": student.bias.detach().cpu().tolist(),
                "raw_time_constant": student.raw_time_constant.detach().cpu().tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    torch.save(
        {
            key: {
                name: value
                for name, value in item.items()
                if name in {"update", "round", "score", "fidelity_passed", "parameters"}
            }
            for key, item in archive.items()
        },
        args.output_dir / "archive.pt",
    )
    report = {
        "method": "full-prefix multi-time analytical-teacher distillation",
        "claim_scope": (
            "The training schedule uses timestamps, privileged relative geometry, "
            f"{'exact-mass' if teacher_uses_exact_mass else 'mass-free'} analytical teacher "
            "actions, and replayed histories, but the deployed actor receives "
            "only current FPV, roll/pitch, body-Z specific force, and its persistent native "
            "connectome state."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "student_checkpoint": stable_path(args.student_checkpoint),
        "student_checkpoint_sha256": file_sha256(args.student_checkpoint),
        "teacher_spec": stable_path(args.teacher_spec),
        "teacher_spec_sha256": file_sha256(args.teacher_spec),
        "teacher_mode": teacher_mode,
        "teacher_uses_exact_simulator_mass": teacher_uses_exact_mass,
        "teacher_mass_argument_invariant": teacher_mass_argument_invariant,
        "warm_start_vector": stable_path(args.warm_start_vector),
        "warm_start_vector_sha256": file_sha256(args.warm_start_vector),
        "warm_start_parameter_sha256": warm_start["parameter_vector_sha256"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "horizon_sampling_weights": [2, 2, 1, 1, 1, 1, 1],
            "promoted_controller_immutable_reference": True,
            "student_driven_training_histories": True,
            "analytical_teacher_actions_training_only": True,
            "complete_prefix_per_sampled_endpoint": True,
            "physics_and_renderer_outside_autograd": True,
            "second_round_original_current_mixture": [0.5, 0.5],
            "final_evaluation_uses_diverse_geometry": True,
            "final_evaluation_extreme_mass_pair_fraction": 0.5,
            "distillation_target": "exact four-axis analytical-teacher action",
            "axis_loss": "absolute teacher-action fidelity on roll/pitch/yaw",
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "all_native_edge_magnitudes_trainable": True,
            "all_native_biases_trainable": True,
            "all_native_time_constants_trainable": True,
            "engineered_history_features": False,
            "clock_actor_input": False,
            "mass_actor_input": False,
            "teacher_actor_input": False,
            "proprioception_input_active": student.uses_proprioception,
        },
        "target_scales": {
            "contrast_rms": scales.contrast_squared.sqrt().detach().cpu().tolist(),
            "mean_rms": scales.mean_squared.sqrt().detach().cpu().tolist(),
            "axis_rms": scales.axis_squared.sqrt().detach().cpu().tolist(),
            "anchor_rms": scales.anchor_squared.sqrt().detach().cpu().tolist(),
        },
        "longest_prefix_gradient_audit": audit,
        "original_collection": original.source_summary,
        "analytic_target_preflight": preflight_target,
        "holdout_collection": holdout.source_summary,
        "initial_holdout_metrics": initial_holdout_metrics,
        "updates": all_updates,
        "selections": all_selections,
        "round1_selected": round1_key,
        "round1_selected_metrics": round1_metrics,
        "round1_fidelity_gate_passed": round1_gate_passed,
        "dagger_collection": dagger_summary,
        "dagger_analytic_teacher_takeover_audit": takeover,
        "round2_selected": round2_key,
        "fresh_student_history_metrics": fresh_metrics,
        "fresh_student_history_fidelity_passed": fresh_fidelity_passed,
        "selected_parameter_vector_sha256": controller_parameter_sha256(student),
        "selected_candidate_vector_file": stable_path(vector_path),
        "selected_candidate_vector_file_sha256": file_sha256(vector_path),
        "selected_parameter_change_from_promoted": parameter_change_summary(
            student, source_parameters
        ),
        "paired_final": paired_final,
        "acceleration_controls": acceleration_controls,
        "acceleration_dependence_demonstrated": acceleration_dependence,
        "final": final,
        "promotion": promotion,
        "goal_threshold_checks": goal_threshold_checks,
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
                "updates_completed": len(all_updates),
                "round1_fidelity_gate_passed": round1_gate_passed,
                "fresh_student_history_fidelity_passed": fresh_fidelity_passed,
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
