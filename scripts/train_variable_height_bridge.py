#!/usr/bin/env python3
"""Generalize visual height response before retrying closed-loop hover training."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover  # noqa: E402

from flydrone.hover import (  # noqa: E402
    PLANT_MODEL_VERSION,
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)
from flydrone.variable_hover import (  # noqa: E402
    MarkerPairConditions,
    protocol_manifest,
    sample_cross_band_marker_pairs,
    sample_height_conditions,
    sample_marker_pairs,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    VisualScene,
    default_visual_scene,
    render_visual_hover_scene,
    sample_visual_scenes,
    visual_scene_manifest,
)

BRIDGE_LESSONS = (
    "random_scene_small_pair",
    "random_scene_medium_pair",
    "legal_cross_band_source_pair",
    "dynamic_source_replay",
)
PAIR_HALF_STEPS = {"small": 0.025, "medium": 0.05}


def experiment_name(args: argparse.Namespace) -> str:
    version = "v2" if args.scene_coverage == "all" else "v1"
    return f"variable-height-visual-response-bridge-{version}"


def bridge_scene_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = visual_scene_manifest()
    if args.scene_coverage == "all":
        manifest["style_combination_split"] = {
            "training": "all four wall/floor style combinations, uniformly balanced",
            "evaluation": "all four combinations with fresh continuous realizations",
            "withheld_style_combinations": False,
        }
    return manifest


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
        default=REPO_ROOT / "runs/variable-height-hover/bridge-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--unroll", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=7.5e-5)
    parser.add_argument(
        "--scene-coverage",
        choices=("matched", "all"),
        default="all",
        help="v1 used matched styles; v2 balances all four wall/floor combinations",
    )
    parser.add_argument("--interim-evaluation-episodes", type=int, default=64)
    parser.add_argument("--confirmation-episodes", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--response-seconds", type=float, default=6.0)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _initial_pair_state(
    pairs: MarkerPairConditions,
    *,
    randomized_scene: bool,
    all_style_combinations: bool,
    device: torch.device,
    config: HoverConfig,
) -> tuple[Any, Any, VisualScene, torch.Tensor]:
    batch = pairs.marker_a.shape[0]
    # The prefix target is always one of the two legal training markers. It is never the
    # cross-band midpoint, which would fall inside the absolute-height holdout.
    prefix_marker = pairs.marker_a
    camera_height = (prefix_marker + torch.empty(batch, device=device).uniform_(-0.10, 0.10)).clamp(
        0.45, 1.55
    )
    state, sticks = hover.nominal_initial_state(
        batch,
        camera_height=camera_height,
        device=device,
        config=config,
        attitude_degrees=4.0,
        rate_degrees_per_second=10.0,
        vertical_speed=0.20,
    )
    scene = (
        sample_visual_scenes(
            batch,
            device=device,
            held_out_combinations=False,
            all_style_combinations=all_style_combinations,
        )
        if randomized_scene
        else default_visual_scene(state, config)
    )
    return state, sticks, scene, prefix_marker


@torch.no_grad()
def _dual_prefix(
    student: ConnectomeController,
    source: ConnectomeController,
    state: Any,
    stick_state: Any,
    marker: torch.Tensor,
    scene: VisualScene,
    *,
    steps: int,
    physics_steps: int,
    config: HoverConfig,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = marker.shape[0]
    student_neural = student.initial_state(batch, device=marker.device, dtype=torch.float32)
    source_neural = source.initial_state(batch, device=marker.device, dtype=torch.float32)
    quad = DifferentiableQuad(config).to(marker.device)
    sticks = ForelegStickPlant(config).to(marker.device)
    valid = hover.state_is_valid(state)
    for _ in range(steps):
        image = render_visual_hover_scene(state, marker, config=config, scene=scene)
        student_motor, student_neural = student(image, state.euler[:, :2], student_neural)
        _, source_neural = source(image, state.euler[:, :2], source_neural)
        state, stick_state, _ = hover.advance_physics(
            quad,
            sticks,
            student_motor,
            state,
            stick_state,
            physics_steps,
        )
        valid &= hover.state_is_valid(state)
    return (
        state.detach(),
        stick_state.detach(),
        student_neural.detach(),
        source_neural.detach(),
        valid,
    )


def _branch_outputs(
    controller: ConnectomeController,
    state: Any,
    neural: torch.Tensor,
    scene: VisualScene,
    marker_a: torch.Tensor,
    marker_b: torch.Tensor,
    *,
    response_steps: int,
    loss_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = marker_a.shape[0]
    images = torch.cat(
        (
            render_visual_hover_scene(state, marker_a, scene=scene),
            render_visual_hover_scene(state, marker_b, scene=scene),
        )
    )
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


def _teacher_pair_loss(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    half_step: float,
    batch: int,
    unroll: int,
    prefix_steps: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    all_style_combinations: bool,
    objective: str = "combined",
) -> tuple[torch.Tensor, dict[str, float]]:
    pairs = sample_marker_pairs(
        batch,
        device=device,
        held_out=False,
        half_step_metres=half_step,
    )
    state, stick_state, scene, prefix_marker = _initial_pair_state(
        pairs,
        randomized_scene=True,
        all_style_combinations=all_style_combinations,
        device=device,
        config=config,
    )
    state, _, student_neural, source_neural, valid = _dual_prefix(
        student,
        source,
        state,
        stick_state,
        prefix_marker,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    loss_start = min(unroll - 1, 10)
    student_a, student_b = _branch_outputs(
        student,
        state,
        student_neural,
        scene,
        pairs.marker_a,
        pairs.marker_b,
        response_steps=unroll,
        loss_start=loss_start,
    )
    with torch.no_grad():
        source_a, source_b = _branch_outputs(
            source,
            state,
            source_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=unroll,
            loss_start=loss_start,
        )
        target_a = hover.teacher_motor(state, pairs.marker_a, config)
        target_b = hover.teacher_motor(state, pairs.marker_b, config)

    desired_contrast = target_a[:, 3] - target_b[:, 3]
    predicted_contrast = student_a[:, :, 3] - student_b[:, :, 3]
    # These are fixed per amplitude bin: the teacher contrast is independent of the
    # sampled pose and vertical speed, so no tiny per-batch denominator is introduced.
    contrast_scale = max(0.01, 0.214 * (2.0 * half_step))
    valid_steps = valid[None].expand(student_a.shape[0], -1)
    contrast_error = ((predicted_contrast - desired_contrast[None]) / contrast_scale).square()
    contrast_loss = hover.masked_mean(contrast_error, valid_steps)

    student_common = 0.5 * (student_a[:, :, 3] + student_b[:, :, 3])
    source_common = 0.5 * (source_a[:, :, 3] + source_b[:, :, 3])
    common_error = student_common - source_common
    common_loss = hover.masked_mean((common_error / 0.05).square(), valid_steps)
    common_signed = hover.masked_mean(common_error / 0.05, valid_steps)
    axis_error = (
        ((student_a[:, :, :3] - source_a[:, :, :3]) / 0.05).square().mean(dim=2)
        + ((student_b[:, :, :3] - source_b[:, :, :3]) / 0.05).square().mean(dim=2)
    ) * 0.5
    axis_loss = hover.masked_mean(axis_error, valid_steps)
    absolute_teacher_throttle_loss = hover.masked_mean(
        0.5
        * (
            ((student_a[:, :, 3] - target_a[None, :, 3]) / 0.05).square()
            + ((student_b[:, :, 3] - target_b[None, :, 3]) / 0.05).square()
        ),
        valid_steps,
    )
    component_losses = {
        "contrast": contrast_loss,
        "common_throttle": common_loss,
        "common_signed": common_signed,
        "rpy": axis_loss,
        "combined": contrast_loss + 0.75 * common_loss + 0.5 * axis_loss,
    }
    if objective not in component_losses:
        raise ValueError(f"unknown paired objective: {objective}")
    loss = component_losses[objective]
    return loss, {
        "contrast_nrmse": float(torch.sqrt(contrast_loss.detach())),
        "common_nrmse": float(torch.sqrt(common_loss.detach())),
        "common_error_mean": float(hover.masked_mean(common_error.detach(), valid_steps).detach()),
        "common_error_rms": float(
            torch.sqrt(hover.masked_mean(common_error.detach().square(), valid_steps))
        ),
        "common_error_max_absolute": float(
            common_error.detach()[:, valid].abs().max() if bool(valid.any()) else 0.0
        ),
        "axis_nrmse": float(torch.sqrt(axis_loss.detach())),
        "absolute_teacher_throttle_nrmse": float(
            torch.sqrt(absolute_teacher_throttle_loss.detach())
        ),
        "student_motor_max_absolute": float(
            torch.cat((student_a, student_b), dim=1).detach().abs().max()
        ),
        "valid_fraction": float(valid.float().mean()),
    }


def _source_cross_band_loss(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    batch: int,
    unroll: int,
    prefix_steps: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    pairs = sample_cross_band_marker_pairs(batch, device=device)
    state, stick_state, scene, prefix_marker = _initial_pair_state(
        pairs,
        randomized_scene=False,
        all_style_combinations=False,
        device=device,
        config=config,
    )
    state, _, student_neural, source_neural, valid = _dual_prefix(
        student,
        source,
        state,
        stick_state,
        prefix_marker,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    loss_start = min(unroll - 1, 10)
    student_a, student_b = _branch_outputs(
        student,
        state,
        student_neural,
        scene,
        pairs.marker_a,
        pairs.marker_b,
        response_steps=unroll,
        loss_start=loss_start,
    )
    with torch.no_grad():
        source_a, source_b = _branch_outputs(
            source,
            state,
            source_neural,
            scene,
            pairs.marker_a,
            pairs.marker_b,
            response_steps=unroll,
            loss_start=loss_start,
        )
        target_a = hover.teacher_motor(state, pairs.marker_a, config)
        target_b = hover.teacher_motor(state, pairs.marker_b, config)
    scales = student_a.new_tensor(hover.MOTOR_CORRECTION_SCALES)
    valid_steps = valid[None].expand(student_a.shape[0], -1)
    per_axis_error = 0.5 * (
        ((student_a - source_a) / scales).square() + ((student_b - source_b) / scales).square()
    )
    error = per_axis_error.mean(dim=2)
    loss = hover.masked_mean(error, valid_steps)
    axis_nrmse = torch.stack(
        [
            torch.sqrt(hover.masked_mean(per_axis_error[:, :, axis], valid_steps))
            for axis in range(4)
        ]
    )
    absolute_teacher_throttle_loss = hover.masked_mean(
        0.5
        * (
            ((student_a[:, :, 3] - target_a[None, :, 3]) / 0.05).square()
            + ((student_b[:, :, 3] - target_b[None, :, 3]) / 0.05).square()
        ),
        valid_steps,
    )
    return loss, {
        "source_motor_nrmse": float(torch.sqrt(loss.detach())),
        "roll_source_nrmse": float(axis_nrmse[0].detach()),
        "pitch_source_nrmse": float(axis_nrmse[1].detach()),
        "yaw_source_nrmse": float(axis_nrmse[2].detach()),
        "throttle_source_nrmse": float(axis_nrmse[3].detach()),
        "absolute_teacher_throttle_nrmse": float(
            torch.sqrt(absolute_teacher_throttle_loss.detach())
        ),
        "student_motor_max_absolute": float(
            torch.cat((student_a, student_b), dim=1).detach().abs().max()
        ),
        "valid_fraction": float(valid.float().mean()),
    }


def _dynamic_source_replay_loss(
    student: ConnectomeController,
    source: ConnectomeController,
    *,
    batch: int,
    unroll: int,
    prefix_steps: int,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
    all_style_combinations: bool,
    axis_weights: tuple[float, float, float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    conditions = sample_height_conditions(batch, device=device, split="train")
    state, stick_state = hover.nominal_initial_state(
        batch,
        camera_height=conditions.camera_height,
        device=device,
        config=config,
        attitude_degrees=7.0,
        rate_degrees_per_second=18.0,
        vertical_speed=0.25,
    )
    scene = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=False,
        all_style_combinations=all_style_combinations,
    )
    state, stick_state, student_neural, source_neural, valid = _dual_prefix(
        student,
        source,
        state,
        stick_state,
        conditions.marker_height,
        scene,
        steps=prefix_steps,
        physics_steps=physics_steps,
        config=config,
    )
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    scales = conditions.marker_height.new_tensor(hover.MOTOR_CORRECTION_SCALES)
    losses = []
    per_axis_losses = []
    valid_fractions = []
    absolute_teacher_throttle_losses = []
    student_motor_max_absolute = state.position.new_zeros(())
    for _ in range(unroll):
        image = render_visual_hover_scene(
            state, conditions.marker_height, config=config, scene=scene
        )
        student_motor, student_neural = student(image, state.euler[:, :2], student_neural)
        with torch.no_grad():
            source_motor, source_neural = source(image, state.euler[:, :2], source_neural)
            teacher_motor = hover.teacher_motor(state, conditions.marker_height, config)
        student_motor_max_absolute = torch.maximum(
            student_motor_max_absolute, student_motor.detach().abs().max()
        )
        per_axis_error = ((student_motor - source_motor) / scales).square()
        if axis_weights is None:
            error = per_axis_error.mean(dim=1)
        else:
            weights = per_axis_error.new_tensor(axis_weights)
            error = (per_axis_error * weights).sum(dim=1) / weights.sum().clamp_min(1.0)
        losses.append(hover.masked_mean(error, valid))
        per_axis_losses.append(
            torch.stack(
                [
                    hover.masked_mean(
                        ((student_motor[:, axis] - source_motor[:, axis]) / scales[axis]).square(),
                        valid,
                    )
                    for axis in range(4)
                ]
            )
        )
        valid_fractions.append(valid.float().mean())
        absolute_teacher_throttle_losses.append(
            hover.masked_mean(((student_motor[:, 3] - teacher_motor[:, 3]) / 0.05).square(), valid)
        )
        state, stick_state, _ = hover.advance_physics(
            quad,
            sticks,
            student_motor.detach(),
            state,
            stick_state,
            physics_steps,
        )
        state = state.detach()
        stick_state = stick_state.detach()
        valid &= hover.state_is_valid(state)
    loss = torch.stack(losses).mean()
    axis_nrmse = torch.sqrt(torch.stack(per_axis_losses).mean(dim=0))
    return loss, {
        "source_motor_nrmse": float(torch.sqrt(loss.detach())),
        "roll_source_nrmse": float(axis_nrmse[0].detach()),
        "pitch_source_nrmse": float(axis_nrmse[1].detach()),
        "yaw_source_nrmse": float(axis_nrmse[2].detach()),
        "throttle_source_nrmse": float(axis_nrmse[3].detach()),
        "absolute_teacher_throttle_nrmse": float(
            torch.sqrt(torch.stack(absolute_teacher_throttle_losses).mean()).detach()
        ),
        "student_motor_max_absolute": float(student_motor_max_absolute),
        "valid_fraction": float(torch.stack(valid_fractions).mean()),
    }


def accumulated_update(
    student: ConnectomeController,
    source: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    source_parameters: dict[str, torch.Tensor],
    *,
    update: int,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    metrics: dict[str, Any] = {}
    losses = []
    for name, half_step in PAIR_HALF_STEPS.items():
        _, prefix_steps = hover.prefix_stratum(update * 4 + len(losses), args.policy_hz)
        loss, lesson_metrics = _teacher_pair_loss(
            student,
            source,
            half_step=half_step,
            batch=args.batch_size,
            unroll=args.unroll,
            prefix_steps=prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=args.scene_coverage == "all",
        )
        (loss / len(BRIDGE_LESSONS)).backward()
        losses.append(float(loss.detach()))
        metrics[f"random_scene_{name}_pair"] = lesson_metrics

    _, prefix_steps = hover.prefix_stratum(update * 4 + 2, args.policy_hz)
    loss, lesson_metrics = _source_cross_band_loss(
        student,
        source,
        batch=args.batch_size,
        unroll=args.unroll,
        prefix_steps=prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    (loss / len(BRIDGE_LESSONS)).backward()
    losses.append(float(loss.detach()))
    metrics["legal_cross_band_source_pair"] = lesson_metrics

    _, prefix_steps = hover.prefix_stratum(update * 4 + 3, args.policy_hz)
    loss, lesson_metrics = _dynamic_source_replay_loss(
        student,
        source,
        batch=args.batch_size,
        unroll=args.unroll,
        prefix_steps=prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
        all_style_combinations=args.scene_coverage == "all",
    )
    (loss / len(BRIDGE_LESSONS)).backward()
    losses.append(float(loss.detach()))
    metrics["dynamic_source_replay"] = lesson_metrics

    regularization = hover.source_regularization(student, source_parameters)
    regularization.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 0.7)
    optimizer.step()
    student.project_parameters()
    return {
        "mean_lesson_loss": sum(losses) / len(losses),
        "regularization": float(regularization.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "lessons": metrics,
    }


def _bridge_guards(evaluation: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    marker = evaluation["marker_steps"]
    baseline_marker = baseline["marker_steps"]
    if marker["altitude_rmse_mean_m"] > 1.05 * baseline_marker["altitude_rmse_mean_m"]:
        reasons.append("absolute-height RMSE regressed by more than five percent")
    if marker["ground_contact_rate"] > baseline_marker["ground_contact_rate"]:
        reasons.append("new marker-step ground contacts")
    if marker["invalid_flight_rate"] > baseline_marker["invalid_flight_rate"]:
        reasons.append("new invalid marker-step flights")
    attitude = evaluation["attitude_recovery"]
    baseline_attitude = baseline["attitude_recovery"]
    if attitude["success_rate"] < max(0.0, baseline_attitude["success_rate"] - 0.05):
        reasons.append("attitude recovery success regressed by more than five points")
    if attitude["tilt_rms_mean_degrees"] > baseline_attitude["tilt_rms_mean_degrees"] + 1.0:
        reasons.append("attitude recovery tilt regressed by more than one degree")
    if attitude["ground_contact_rate"] > baseline_attitude["ground_contact_rate"]:
        reasons.append("new attitude-recovery ground contacts")
    if attitude["invalid_flight_rate"] > baseline_attitude["invalid_flight_rate"]:
        reasons.append("new invalid attitude-recovery flights")
    if not evaluation["legacy_pair"]["pass"]:
        reasons.append("legacy paired-marker response failed")
    return not reasons, reasons


@torch.no_grad()
def evaluate_pair_matrix(
    controller: ConnectomeController,
    *,
    held_out_markers: bool,
    pairs_per_stratum: int,
    args: argparse.Namespace,
    device: torch.device,
    config: HoverConfig,
    physics_steps: int,
    seed: int,
) -> dict[str, Any]:
    strata = {}
    stratum_seed = seed
    for amplitude_name, half_step in PAIR_HALF_STEPS.items():
        for wall_style in (0, 1):
            for floor_style in (0, 1):
                key = f"{amplitude_name}_wall_{wall_style}_floor_{floor_style}"
                strata[key] = hover.evaluate_genuine_pairs(
                    controller,
                    pairs_count=pairs_per_stratum,
                    batch_size=args.evaluation_batch_size,
                    response_steps=args.unroll,
                    physics_steps=physics_steps,
                    device=device,
                    config=config,
                    seed=stratum_seed,
                    half_step_metres=half_step,
                    fixed_wall_style=wall_style,
                    fixed_floor_style=floor_style,
                    held_out_markers=held_out_markers,
                    held_out_scene=False,
                )
                stratum_seed += 1
    by_style = {}
    for wall_style in (0, 1):
        for floor_style in (0, 1):
            values = [
                result["contrast_nrmse"]
                for key, result in strata.items()
                if f"wall_{wall_style}_floor_{floor_style}" in key
            ]
            by_style[f"wall_{wall_style}_floor_{floor_style}"] = sum(values) / len(values)
    return {
        "marker_split": "held_out" if held_out_markers else "training_support",
        "pairs_per_stratum": pairs_per_stratum,
        "mean_contrast_nrmse": sum(result["contrast_nrmse"] for result in strata.values())
        / len(strata),
        "every_stratum_pass": all(result["pass"] for result in strata.values()),
        "mean_contrast_nrmse_by_style": by_style,
        "strata": strata,
    }


@torch.no_grad()
def evaluate_bridge(
    controller: ConnectomeController,
    *,
    episodes: int,
    args: argparse.Namespace,
    device: torch.device,
    config: HoverConfig,
    physics_steps: int,
    seed: int,
) -> dict[str, Any]:
    suite = hover.evaluate_suite(
        controller,
        episodes=episodes,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=seed,
    )
    stratum_pairs = 4 if args.smoke_test else max(16, episodes // 4)
    suite["training_support_pair_matrix"] = evaluate_pair_matrix(
        controller,
        held_out_markers=False,
        pairs_per_stratum=stratum_pairs,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=seed + 100,
    )
    suite["fresh_height_pair_matrix"] = evaluate_pair_matrix(
        controller,
        held_out_markers=True,
        pairs_per_stratum=stratum_pairs,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=seed + 200,
    )
    return suite


def _decision(evaluation: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    guards_pass, guard_reasons = _bridge_guards(evaluation, baseline)
    genuine = evaluation["genuine_held_out_pair"]
    improvement_fraction = (
        1.0 - genuine["contrast_nrmse"] / baseline["genuine_held_out_pair"]["contrast_nrmse"]
    )
    training_matrix = evaluation["training_support_pair_matrix"]
    baseline_training = baseline["training_support_pair_matrix"]
    training_improvement = (
        1.0 - training_matrix["mean_contrast_nrmse"] / baseline_training["mean_contrast_nrmse"]
    )
    style_improvements = {
        style: 1.0 - value / baseline_training["mean_contrast_nrmse_by_style"][style]
        for style, value in training_matrix["mean_contrast_nrmse_by_style"].items()
    }
    improvement_represented = all(value > 0.0 for value in style_improvements.values())
    fresh_matrix_pass = evaluation["fresh_height_pair_matrix"]["every_stratum_pass"]
    return {
        "guards_pass": guards_pass,
        "guard_reasons": guard_reasons,
        "genuine_pair_nrmse_improvement_fraction": improvement_fraction,
        "training_support_nrmse_improvement_fraction": training_improvement,
        "training_support_improvement_by_style": style_improvements,
        "training_support_improvement_represented_in_every_style": improvement_represented,
        "update_50_continuation_pass": bool(
            guards_pass and training_improvement >= 0.25 and improvement_represented
        ),
        "bridge_acceptance_pass": bool(
            guards_pass
            and genuine["pass"]
            and evaluation["legacy_pair"]["pass"]
            and fresh_matrix_pass
        ),
        "every_fresh_height_amplitude_scene_stratum_pass": fresh_matrix_pass,
    }


def _checkpoint_payload(
    controller: ConnectomeController,
    args: argparse.Namespace,
    config: HoverConfig,
    *,
    source_sha256: str,
    selected_update: int,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": experiment_name(args),
        "controller": controller.state_dict(),
        "graph_sha256": hover.file_sha256(args.graph),
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
        "scene_protocol": bridge_scene_manifest(args),
    }


def main() -> int:
    args = parse_args()
    if not args.graph.is_file() or not args.checkpoint.is_file():
        raise SystemExit("the generated graph and source checkpoint are required")
    if args.updates != 100 and not (args.evaluate_only or args.smoke_test):
        raise SystemExit("bridge v1 is preregistered for exactly 100 accumulated updates")
    if args.unroll < 11:
        raise SystemExit("unroll must be at least 11 recurrent steps")
    if args.scene_coverage == "all" and args.batch_size % 4:
        raise SystemExit("all-style scene coverage requires batch-size divisible by four")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the physics rate")
    physics_steps = physics_hz // args.policy_hz
    hover.seed_everything(args.seed)

    source_sha256 = hover.file_sha256(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    graph_sha256 = hover.file_sha256(args.graph)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    student.load_state_dict(loaded["controller"])
    source.load_state_dict(loaded["controller"])
    source.eval()
    source.requires_grad_(False)
    source_parameters = {
        name: loaded["controller"][name].detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    evaluation_seed = args.seed + 10_000
    student.eval()
    baseline = evaluate_bridge(
        student,
        episodes=args.interim_evaluation_episodes,
        args=args,
        device=device,
        config=config,
        physics_steps=physics_steps,
        seed=evaluation_seed,
    )
    evaluations = [{"update": 0, "decision": _decision(baseline, baseline), "evaluation": baseline}]
    print(json.dumps(evaluations[0]), flush=True)
    history = []
    accepted_snapshots: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    stop_reason = None
    updates_completed = 0

    if not args.evaluate_only:
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=args.learning_rate, weight_decay=1.0e-6
        )
        student.train()
        for update in range(1, args.updates + 1):
            metrics = accumulated_update(
                student,
                source,
                optimizer,
                source_parameters,
                update=update - 1,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            updates_completed = update
            if update == 1 or update % 10 == 0:
                entry = {"update": update, **metrics}
                history.append(entry)
                print(json.dumps(entry), flush=True)
            if update in (25, 50, 100):
                training_rng = hover.capture_rng_state()
                student.eval()
                evaluation = evaluate_bridge(
                    student,
                    episodes=args.interim_evaluation_episodes,
                    args=args,
                    device=device,
                    config=config,
                    physics_steps=physics_steps,
                    seed=evaluation_seed,
                )
                hover.restore_rng_state(training_rng)
                decision = _decision(evaluation, baseline)
                entry = {"update": update, "decision": decision, "evaluation": evaluation}
                evaluations.append(entry)
                print(json.dumps(entry), flush=True)
                torch.save(
                    _checkpoint_payload(
                        student,
                        args,
                        config,
                        source_sha256=source_sha256,
                        selected_update=update,
                    ),
                    args.output_dir / f"update-{update:04d}.pt",
                )
                if decision["bridge_acceptance_pass"]:
                    state = {
                        name: value.detach().cpu().clone()
                        for name, value in student.state_dict().items()
                    }
                    accepted_snapshots.append(
                        (evaluation["genuine_held_out_pair"]["contrast_nrmse"], update, state)
                    )
                if update == 50 and not decision["update_50_continuation_pass"]:
                    stop_reason = "update-50 continuation gate failed"
                    break
                if not decision["guards_pass"]:
                    stop_reason = "; ".join(decision["guard_reasons"])
                    break
                student.train()

    selected_update = 0
    selected_state = {
        name: value.detach().cpu().clone() for name, value in loaded["controller"].items()
    }
    candidate_confirmation = None
    passed = False
    if accepted_snapshots:
        _, selected_update, selected_state = min(accepted_snapshots, key=lambda item: item[0])
        student.load_state_dict(selected_state)
        student.eval()
        candidate_confirmation = evaluate_bridge(
            student,
            episodes=args.confirmation_episodes,
            args=args,
            device=device,
            config=config,
            physics_steps=physics_steps,
            seed=evaluation_seed + 50_000,
        )
        confirmation_decision = _decision(candidate_confirmation, baseline)
        candidate_confirmation["decision"] = confirmation_decision
        passed = confirmation_decision["bridge_acceptance_pass"]
        if not passed:
            selected_update = 0
            selected_state = {
                name: value.detach().cpu().clone() for name, value in loaded["controller"].items()
            }
            stop_reason = "fresh confirmation gate failed"

    student.load_state_dict(selected_state)
    student.eval()
    selected_path = args.output_dir / "selected-controller.pt"
    torch.save(
        _checkpoint_payload(
            student,
            args,
            config,
            source_sha256=source_sha256,
            selected_update=selected_update,
        ),
        selected_path,
    )
    report = {
        "experiment": experiment_name(args),
        "passed": passed,
        "promoted_to_mixed_curriculum_seed": passed,
        "selected_update": selected_update,
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": hover.file_sha256(selected_path),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "graph": str(args.graph),
        "graph_sha256": graph_sha256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "accumulated_lessons_per_update": list(BRIDGE_LESSONS),
        "updates_requested": args.updates,
        "updates_completed": updates_completed,
        "stop_reason": stop_reason,
        "actor_inputs": _checkpoint_payload(
            student,
            args,
            config,
            source_sha256=source_sha256,
            selected_update=selected_update,
        )["actor_inputs"],
        "protocol": protocol_manifest(),
        "scene_protocol": bridge_scene_manifest(args),
        "scene_coverage": args.scene_coverage,
        "closed_loop_safety_suite_scene_scope": (
            "the original fixed opposite-style diagnostic split; paired support and fresh-"
            "height matrices cover all four style combinations"
        ),
        "training_claim": (
            "all training and prefix markers stay in the two declared training bands; "
            "the held-out marker interval, texture realizations, and trajectories are "
            "evaluation-only; v2 balances all four wall/floor style combinations in training"
        ),
        "credit_assignment": (
            "four lesson gradients are accumulated at unchanged parameters; current-policy "
            "physical evolution is detached and neural recurrence remains native"
        ),
        "baseline": baseline,
        "history": history,
        "evaluations": evaluations,
        "candidate_confirmation": candidate_confirmation,
        "elapsed_seconds": perf_counter() - started,
    }
    (args.output_dir / "report.json").write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
