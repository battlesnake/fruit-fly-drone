#!/usr/bin/env python3
"""Preflight full-native joint visual-height and damping teacher fitting.

This is a restored one-step learnability diagnostic.  It never saves or promotes the
candidate parameters, and its cached observations contain no privileged actor inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_factorial_damping as factorial  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_hover as hover_train  # noqa: E402

import flydrone.variable_hover as variable_hover  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
    teacher_rc,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    render_visual_hover_scene,
    sample_visual_scenes,
    visual_scene_manifest,
)

EXPERIMENT = "variable-height-full-native-joint-preflight-v1"
TRAIN_SEED = 320_953
DEVELOPMENT_SEED = 330_953
SCENES_PER_BANK = 8
PREFIX_STEPS = 5
RESPONSE_STEPS = 25
SUPERVISION_STEPS = (15, 20, 25)
POLICY_HZ = 50
EDGE_BIAS_LEARNING_RATE = 1.0e-4
TIME_CONSTANT_LEARNING_RATE = 1.0e-6
GRADIENT_NORM_CAP = 1.0
FINITE_DIFFERENCE_SCALE = 0.0625
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
MINIMUM_TRAIN_OBJECTIVE_IMPROVEMENT = 1.0e-4
COMPONENT_BASELINE_TOLERANCE = 0.02
RPY_NRMSE_LIMIT = 0.05
REPLAY_TOLERANCE = 1.0e-6
COMMON_SCALE = 0.05
INTERACTION_SCALE = 0.01
TEACHER_SCALE_FLOOR = 0.01
RPY_SCALES = (0.05, 0.05, 0.04)
TEACHER_PLANT_SETTLING_STEPS = 100
COMPONENTS = ("common", "height", "damping", "interaction")
PARAMETER_FAMILIES = ("edge_magnitude", "bias", "raw_time_constant")


@dataclass
class FactorialCache:
    prefix_images: Tensor
    response_images: Tensor
    attitude: Tensor
    teacher_motor: Tensor
    teacher_rc: Tensor
    height_error: Tensor
    vertical_speed: Tensor
    approach_steps: Tensor
    camera_height: Tensor
    endpoint_image_difference_max: float
    sha256: str


@dataclass
class AttitudeCache:
    images: Tensor
    attitudes: Tensor
    source_motor: Tensor
    valid: Tensor
    camera_height: Tensor
    marker_height: Tensor
    sha256: str


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
        default=REPO_ROOT / "runs/variable-height-hover/full-native-joint-preflight-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--development-seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--scenes", type=int, default=SCENES_PER_BANK)
    parser.add_argument("--prefix-steps", type=int, default=PREFIX_STEPS)
    parser.add_argument("--response-steps", type=int, default=RESPONSE_STEPS)
    parser.add_argument("--policy-hz", type=int, default=POLICY_HZ)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.scenes < 4 or args.scenes % 4:
        raise SystemExit("--scenes must be a positive multiple of four")
    if args.response_steps < max(factorial.APPROACH_STEPS):
        raise SystemExit("response must cover every preregistered approach duration")
    if args.prefix_steps < 1 or args.policy_hz < 1:
        raise SystemExit("prefix steps and policy frequency must be positive")
    frozen = (
        args.train_seed,
        args.development_seed,
        args.scenes,
        args.prefix_steps,
        args.response_steps,
        args.policy_hz,
    )
    expected = (
        TRAIN_SEED,
        DEVELOPMENT_SEED,
        SCENES_PER_BANK,
        PREFIX_STEPS,
        RESPONSE_STEPS,
        POLICY_HZ,
    )
    if not args.smoke_test and frozen != expected:
        raise SystemExit(f"the full preflight is preregistered for {expected}")


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "train_seed": args.train_seed,
        "development_seed": args.development_seed,
        "scenes_per_bank": args.scenes,
        "factorial_histories_per_scene": len(factorial.BRANCH_SIGNS),
        "height_error_magnitudes_metres": list(factorial.HEIGHT_ERROR_MAGNITUDES),
        "vertical_speed_magnitudes_metres_per_second": list(factorial.VERTICAL_SPEED_MAGNITUDES),
        "approach_steps": list(factorial.APPROACH_STEPS),
        "prefix_steps": args.prefix_steps,
        "response_steps": args.response_steps,
        "supervision_steps_one_indexed": list(SUPERVISION_STEPS),
        "policy_hz": args.policy_hz,
        "native_state_initialization": "zero",
        "prefix_and_response_are_differentiated": True,
        "cached_observations_reused_for_all_parameter_points": True,
        "microbatch_unit": "one scene",
        "microbatch_gradient_accumulation_is_exact_objective_mean": True,
        "actor_inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
        "camera": {
            "width": DEFAULT_VISUAL_CAMERA.width,
            "height": DEFAULT_VISUAL_CAMERA.height,
            "horizontal_fov_degrees": DEFAULT_VISUAL_CAMERA.horizontal_fov_degrees,
        },
        "privileged_actor_inputs": [],
        "opened_parameter_families": list(PARAMETER_FAMILIES),
        "frozen": [
            "topology",
            "transmitter signs",
            "retinal mapping",
            "attitude mapping",
            "sensory gains",
            "actor inputs",
            "foreleg output pools",
        ],
        "source_distance_constraint": False,
        "height_null_projection": False,
        "component_scales": {
            "common_motor_units": COMMON_SCALE,
            "height": "fixed train-teacher RMS floored at 0.01 motor units",
            "damping": "fixed train-teacher RMS floored at 0.01 motor units",
            "interaction_motor_units": INTERACTION_SCALE,
            "roll_pitch_yaw_motor_units": list(RPY_SCALES),
        },
        "objective": "equal mean of normalized C/P/D/I/R/P/Y MSE",
        "supervision_horizon_weighting": "equal",
        "literal_damping_sign_and_gain_horizon": RESPONSE_STEPS,
        "optimizer": {
            "name": "Adam",
            "edge_and_bias_learning_rate": EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": TIME_CONSTANT_LEARNING_RATE,
            "weight_decay": 0.0,
            "global_gradient_norm_cap": GRADIENT_NORM_CAP,
        },
        "finite_difference_scale": FINITE_DIFFERENCE_SCALE,
        "finite_difference_relative_error_limit": (FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT),
        "teacher_positive_control_settling_steps_at_physics_rate": (TEACHER_PLANT_SETTLING_STEPS),
        "smoke_test": args.smoke_test,
    }


def smooth_return_velocity(
    signed_speed: Tensor,
    *,
    step: int,
    total_steps: int,
    approach_steps: Tensor,
    policy_hz: int,
) -> Tensor:
    """Exact time derivative of ``factorial.smooth_return_offset``."""

    local_step = (step + 1 - (total_steps - approach_steps)).clamp_min(0)
    active = local_step > 0
    unit = local_step / approach_steps
    duration = approach_steps / float(policy_hz)
    polynomial = signed_speed * (3.0 * unit.square() - 2.0 * unit)
    sinusoid = (
        signed_speed.sign()
        * 0.055
        * (2.0 * math.pi / duration)
        * torch.sin(math.pi * unit)
        * torch.cos(math.pi * unit)
    )
    return torch.where(active, polynomial - sinusoid, torch.zeros_like(signed_speed))


def _hash_tensors(named: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(named):
        value = named[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


@torch.no_grad()
def build_factorial_cache(
    *,
    scenes: int,
    seed: int,
    response_steps: int,
    policy_hz: int,
    device: torch.device,
    config: HoverConfig,
) -> FactorialCache:
    responsibility.seed_everything(seed)
    height_error, speed = factorial.balanced_amplitude_grid(scenes, device=device)
    state, height = responsibility.base_state(scenes, device=device, config=config)
    scene = sample_visual_scenes(
        scenes,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    choices = torch.tensor(factorial.APPROACH_STEPS, device=device)
    approach_steps = choices.repeat(math.ceil(scenes / len(choices)))[:scenes]
    approach_steps = approach_steps[torch.randperm(scenes, device=device)]
    prefix_images = render_visual_hover_scene(state, height, config=config, scene=scene)
    branch_markers = [
        height + height_sign * height_error for height_sign, _ in factorial.BRANCH_SIGNS
    ]
    images_by_step: list[Tensor] = []
    teacher_by_step: list[Tensor] = []
    teacher_rc_by_step: list[Tensor] = []
    endpoint_images: list[Tensor] = []
    for step in range(response_steps):
        images = []
        targets = []
        target_rc = []
        for (_, velocity_sign), marker in zip(factorial.BRANCH_SIGNS, branch_markers, strict=True):
            signed_speed = velocity_sign * speed
            offset = factorial.smooth_return_offset(
                signed_speed,
                step=step,
                total_steps=response_steps,
                approach_steps=approach_steps,
                policy_hz=policy_hz,
            )
            vertical_velocity = smooth_return_velocity(
                signed_speed,
                step=step,
                total_steps=response_steps,
                approach_steps=approach_steps,
                policy_hz=policy_hz,
            )
            branch_state = responsibility.state_at_height(
                state,
                height + offset,
                vertical_velocity=vertical_velocity,
            )
            images.append(
                render_visual_hover_scene(branch_state, marker, config=config, scene=scene)
            )
            rc = teacher_rc(branch_state, marker, config)
            target_rc.append(rc)
            targets.append(motor_target_for_rc(rc, config))
        endpoint_images = images
        images_by_step.append(torch.stack(images, dim=1).cpu())
        teacher_by_step.append(torch.stack(targets, dim=1).cpu())
        teacher_rc_by_step.append(torch.stack(target_rc, dim=1).cpu())
    endpoint_difference = max(
        float((endpoint_images[0] - endpoint_images[1]).abs().max()),
        float((endpoint_images[2] - endpoint_images[3]).abs().max()),
    )
    cache_tensors = {
        "prefix_images": prefix_images.cpu(),
        "response_images": torch.stack(images_by_step, dim=1),
        "attitude": state.euler[:, :2].cpu(),
        "teacher_motor": torch.stack(teacher_by_step, dim=1),
        "teacher_rc": torch.stack(teacher_rc_by_step, dim=1),
        "height_error": height_error.cpu(),
        "vertical_speed": speed.cpu(),
        "approach_steps": approach_steps.cpu(),
        "camera_height": height.cpu(),
    }
    return FactorialCache(
        **cache_tensors,
        endpoint_image_difference_max=endpoint_difference,
        sha256=_hash_tensors(cache_tensors),
    )


@torch.no_grad()
def build_attitude_cache(
    source: ConnectomeController,
    *,
    scenes: int,
    seed: int,
    prefix_steps: int,
    response_steps: int,
    policy_hz: int,
    device: torch.device,
    config: HoverConfig,
) -> AttitudeCache:
    responsibility.seed_everything(seed)
    conditions = variable_hover.sample_height_conditions(scenes, device=device, split="train")
    state, stick_state = hover_train.nominal_initial_state(
        scenes,
        camera_height=conditions.camera_height,
        device=device,
        config=config,
        attitude_degrees=13.0,
        rate_degrees_per_second=28.0,
        vertical_speed=0.20,
    )
    scene = sample_visual_scenes(
        scenes,
        device=device,
        held_out_combinations=False,
        all_style_combinations=True,
    )
    neural = source.initial_state(scenes, device=device, dtype=torch.float32)
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    physics_hz = round(1.0 / config.dt)
    physics_steps = physics_hz // policy_hz
    valid = hover_train.state_is_valid(state)
    images, attitudes, outputs, validity = [], [], [], []
    for _ in range(prefix_steps + response_steps):
        image = render_visual_hover_scene(
            state, conditions.marker_height, config=config, scene=scene
        )
        motor, neural = source(image, state.euler[:, :2], neural)
        images.append(image.cpu())
        attitudes.append(state.euler[:, :2].cpu())
        outputs.append(motor.cpu())
        validity.append(valid.cpu())
        state, stick_state, _ = hover_train.advance_physics(
            quad, sticks, motor, state, stick_state, physics_steps
        )
        valid &= hover_train.state_is_valid(state)
    cache_tensors = {
        "images": torch.stack(images, dim=1),
        "attitudes": torch.stack(attitudes, dim=1),
        "source_motor": torch.stack(outputs, dim=1),
        "valid": torch.stack(validity, dim=1),
        "camera_height": conditions.camera_height.cpu(),
        "marker_height": conditions.marker_height.cpu(),
    }
    return AttitudeCache(**cache_tensors, sha256=_hash_tensors(cache_tensors))


def training_teacher_scales(cache: FactorialCache) -> dict[str, float]:
    indices = [step - 1 for step in SUPERVISION_STEPS]
    throttle = cache.teacher_motor[:, indices, :, 3]
    branch_major = throttle.permute(2, 0, 1).reshape(4, -1)
    targets = factorial.factorial_components(branch_major)
    return {
        "common": COMMON_SCALE,
        "height": max(float(targets["height"].square().mean().sqrt()), TEACHER_SCALE_FLOOR),
        "damping": max(float(targets["damping"].square().mean().sqrt()), TEACHER_SCALE_FLOOR),
        "interaction": INTERACTION_SCALE,
    }


def teacher_identity_report(cache: FactorialCache, scales: dict[str, float]) -> dict[str, Any]:
    """Round-trip cached teacher labels through the trained factorial indexing path."""

    indices = [step - 1 for step in SUPERVISION_STEPS]
    original = cache.teacher_motor[:, indices, :, 3].permute(2, 0, 1).reshape(4, -1)
    components = factorial.factorial_components(original)
    reconstructed = torch.stack(
        [
            components["common"]
            + height_sign * components["height"]
            + velocity_sign * components["damping"]
            + height_sign * velocity_sign * components["interaction"]
            for height_sign, velocity_sign in factorial.BRANCH_SIGNS
        ]
    )
    substituted = factorial.factorial_components(reconstructed)
    nrmse = {
        name: float(((substituted[name] - components[name]) / scales[name]).square().mean().sqrt())
        for name in COMPONENTS
    }
    maximum = float((reconstructed - original).abs().max())
    return {
        "pass": maximum <= REPLAY_TOLERANCE and max(nrmse.values()) <= REPLAY_TOLERANCE,
        "branch_reconstruction_max_absolute_error": maximum,
        "component_nrmse": nrmse,
    }


def _run_factorial_scene(
    controller: ConnectomeController,
    cache: FactorialCache,
    scene_index: int,
    *,
    prefix_steps: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], float]:
    recurrent = controller.initial_state(4, device=device, dtype=torch.float32)
    prefix = cache.prefix_images[scene_index].to(device).unsqueeze(0).expand(4, -1, -1, -1)
    attitude = cache.attitude[scene_index].to(device).unsqueeze(0).expand(4, -1)
    motor_max = 0.0
    for _ in range(prefix_steps):
        motor, recurrent = controller(prefix, attitude, recurrent)
        motor_max = max(motor_max, float(motor.detach().abs().max()))
    outputs = []
    for step in range(cache.response_images.shape[1]):
        image = cache.response_images[scene_index, step].to(device)
        motor, recurrent = controller(image, attitude, recurrent)
        motor_max = max(motor_max, float(motor.detach().abs().max()))
        if step + 1 in SUPERVISION_STEPS:
            outputs.append(motor[:, 3])
    stacked = torch.stack(outputs, dim=1)
    components = factorial.factorial_components(stacked)
    return components, motor_max


def _run_attitude_scene(
    controller: ConnectomeController,
    cache: AttitudeCache,
    scene_index: int,
    *,
    prefix_steps: int,
    device: torch.device,
) -> tuple[Tensor, float]:
    recurrent = controller.initial_state(1, device=device, dtype=torch.float32)
    outputs = []
    motor_max = 0.0
    label_indices = {prefix_steps + step - 1 for step in SUPERVISION_STEPS}
    for step in range(cache.images.shape[1]):
        image = cache.images[scene_index, step].to(device).unsqueeze(0)
        attitude = cache.attitudes[scene_index, step].to(device).unsqueeze(0)
        motor, recurrent = controller(image, attitude, recurrent)
        motor_max = max(motor_max, float(motor.detach().abs().max()))
        if step in label_indices:
            outputs.append(motor[0, :3])
    return torch.stack(outputs), motor_max


def _factorial_targets(
    cache: FactorialCache, scene_index: int, device: torch.device
) -> dict[str, Tensor]:
    indices = [step - 1 for step in SUPERVISION_STEPS]
    throttle = cache.teacher_motor[scene_index, indices, :, 3].to(device).transpose(0, 1)
    return factorial.factorial_components(throttle)


def accumulated_gradient(
    controller: ConnectomeController,
    factorial_cache: FactorialCache,
    attitude_cache: AttitudeCache,
    scales: dict[str, float],
    *,
    prefix_steps: int,
    device: torch.device,
) -> None:
    scenes = factorial_cache.prefix_images.shape[0]
    for scene_index in range(scenes):
        prediction, _ = _run_factorial_scene(
            controller,
            factorial_cache,
            scene_index,
            prefix_steps=prefix_steps,
            device=device,
        )
        target = _factorial_targets(factorial_cache, scene_index, device)
        loss = sum(
            ((prediction[name] - target[name]) / scales[name]).square().mean()
            for name in COMPONENTS
        ) / (7.0 * scenes)
        loss.backward()
    rpy_scales = torch.tensor(RPY_SCALES, device=device)
    for scene_index in range(attitude_cache.images.shape[0]):
        prediction, _ = _run_attitude_scene(
            controller,
            attitude_cache,
            scene_index,
            prefix_steps=prefix_steps,
            device=device,
        )
        indices = [prefix_steps + step - 1 for step in SUPERVISION_STEPS]
        target = attitude_cache.source_motor[scene_index, indices, :3].to(device)
        loss = ((prediction - target) / rpy_scales).square().mean(dim=0).sum()
        (loss / (7.0 * attitude_cache.images.shape[0])).backward()


@torch.no_grad()
def evaluate_bank(
    controller: ConnectomeController,
    factorial_cache: FactorialCache,
    attitude_cache: AttitudeCache,
    scales: dict[str, float],
    *,
    prefix_steps: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    predictions: dict[str, list[Tensor]] = {name: [] for name in COMPONENTS}
    targets: dict[str, list[Tensor]] = {name: [] for name in COMPONENTS}
    rpy_predictions, rpy_targets = [], []
    motor_max = 0.0
    for scene_index in range(factorial_cache.prefix_images.shape[0]):
        prediction, maximum = _run_factorial_scene(
            controller,
            factorial_cache,
            scene_index,
            prefix_steps=prefix_steps,
            device=device,
        )
        target = _factorial_targets(factorial_cache, scene_index, device)
        motor_max = max(motor_max, maximum)
        for name in COMPONENTS:
            predictions[name].append(prediction[name])
            targets[name].append(target[name])
    indices = [prefix_steps + step - 1 for step in SUPERVISION_STEPS]
    for scene_index in range(attitude_cache.images.shape[0]):
        prediction, maximum = _run_attitude_scene(
            controller,
            attitude_cache,
            scene_index,
            prefix_steps=prefix_steps,
            device=device,
        )
        motor_max = max(motor_max, maximum)
        rpy_predictions.append(prediction)
        rpy_targets.append(attitude_cache.source_motor[scene_index, indices, :3].to(device))
    prediction_tensors = {name: torch.stack(values) for name, values in predictions.items()}
    target_tensors = {name: torch.stack(values) for name, values in targets.items()}
    rpy_prediction = torch.stack(rpy_predictions)
    rpy_target = torch.stack(rpy_targets)
    component_mse = {
        name: ((prediction_tensors[name] - target_tensors[name]) / scales[name]).square().mean()
        for name in COMPONENTS
    }
    rpy_scale = torch.tensor(RPY_SCALES, device=device)
    rpy_mse = ((rpy_prediction - rpy_target) / rpy_scale).square().mean(dim=(0, 1))
    objective = (sum(component_mse.values()) + rpy_mse.sum()) / 7.0
    horizon_reports = {}
    for horizon_index, horizon in enumerate(SUPERVISION_STEPS):
        horizon_reports[str(horizon)] = {
            name: float(
                (
                    (
                        prediction_tensors[name][:, horizon_index]
                        - target_tensors[name][:, horizon_index]
                    )
                    / scales[name]
                )
                .square()
                .mean()
                .sqrt()
            )
            for name in COMPONENTS
        }
    endpoint_prediction = prediction_tensors["damping"][:, -1]
    endpoint_target = target_tensors["damping"][:, -1]
    endpoint_gain = float(
        (endpoint_prediction * endpoint_target).sum()
        / endpoint_target.square().sum().clamp_min(1.0e-12)
    )
    endpoint_sign = float(((endpoint_prediction * endpoint_target) > 0.0).float().mean())
    metrics = {
        "joint_normalized_mse": float(objective),
        "component_nrmse": {name: float(value.sqrt()) for name, value in component_mse.items()},
        "rpy_source_nrmse": {
            name: float(rpy_mse[index].sqrt())
            for index, name in enumerate(("roll", "pitch", "yaw"))
        },
        "by_supervision_step_nrmse": horizon_reports,
        "endpoint_damping_correct_sign_fraction": endpoint_sign,
        "endpoint_damping_teacher_aligned_gain": endpoint_gain,
        "motor_output_max_absolute": motor_max,
        "all_attitude_cache_states_valid": bool(attitude_cache.valid.all()),
    }
    raw = {**prediction_tensors, "rpy": rpy_prediction}
    return metrics, raw


@torch.no_grad()
def teacher_plant_positive_control(
    cache: FactorialCache,
    *,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    sticks = ForelegStickPlant(config).to(device)
    measured_endpoints, desired_endpoints = [], []
    maximum_position = 0.0
    maximum_rc_error = 0.0
    horizon_indices = [step - 1 for step in SUPERVISION_STEPS]
    for scene_index in range(cache.prefix_images.shape[0]):
        endpoint_measured = None
        for step in horizon_indices:
            target_motor = cache.teacher_motor[scene_index, step].to(device)
            target_rc = cache.teacher_rc[scene_index, step].to(device)
            state = sticks.initial_state(4, device=device, dtype=torch.float32)
            measured = torch.zeros(4, 4, device=device)
            for _ in range(TEACHER_PLANT_SETTLING_STEPS):
                measured, state = sticks(target_motor, state)
            maximum_position = max(maximum_position, float(state.position.abs().max()))
            maximum_rc_error = max(maximum_rc_error, float((measured - target_rc).abs().max()))
            if step + 1 == RESPONSE_STEPS:
                endpoint_measured = measured[:, 3]
        if endpoint_measured is None:
            raise RuntimeError("teacher positive control did not visit the endpoint")
        measured_endpoints.append(endpoint_measured)
        desired_endpoints.append(cache.teacher_rc[scene_index, -1, :, 3].to(device))
    measured_components = factorial.factorial_components(torch.stack(measured_endpoints).T)
    desired_components = factorial.factorial_components(torch.stack(desired_endpoints).T)
    signs = {
        name: float(((measured_components[name] * desired_components[name]) > 0.0).float().mean())
        for name in ("common", "height", "damping")
    }
    return {
        "pass": (
            maximum_position <= 1.0 and maximum_rc_error <= 0.02 and min(signs.values()) == 1.0
        ),
        "constant_command_settling_steps_at_physics_rate": (TEACHER_PLANT_SETTLING_STEPS),
        "measured_stick_position_max_absolute": maximum_position,
        "teacher_rc_max_absolute_error_after_settling": maximum_rc_error,
        "endpoint_correct_sign_fraction": signs,
        "scope": (
            "Checks command units, signs and steady-state reachability only; it does not "
            "claim time-varying tracking or closed-loop stability."
        ),
    }


def _max_prediction_difference(first: dict[str, Tensor], second: dict[str, Tensor]) -> float:
    return max(float((first[name] - second[name]).abs().max()) for name in first)


def cache_distribution_report(
    factorial_cache: FactorialCache, attitude_cache: AttitudeCache
) -> dict[str, Any]:
    marker_low = factorial_cache.camera_height - factorial_cache.height_error
    marker_high = factorial_cache.camera_height + factorial_cache.height_error
    return {
        "factorial_camera_height_sampled_range_metres": [
            float(factorial_cache.camera_height.min()),
            float(factorial_cache.camera_height.max()),
        ],
        "factorial_marker_height_sampled_range_metres": [
            float(marker_low.min()),
            float(marker_high.max()),
        ],
        "factorial_height_error_magnitudes_metres": (factorial_cache.height_error.tolist()),
        "factorial_vertical_speed_magnitudes_metres_per_second": (
            factorial_cache.vertical_speed.tolist()
        ),
        "factorial_approach_steps": factorial_cache.approach_steps.tolist(),
        "attitude_camera_height_sampled_range_metres": [
            float(attitude_cache.camera_height.min()),
            float(attitude_cache.camera_height.max()),
        ],
        "attitude_marker_height_sampled_range_metres": [
            float(attitude_cache.marker_height.min()),
            float(attitude_cache.marker_height.max()),
        ],
        "visual_style_sampling": "all four wall/floor style combinations, balanced",
    }


def _numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree_is_finite(item) for item in value)
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def decision(
    baseline_train: dict[str, Any],
    candidate_train: dict[str, Any],
    baseline_development: dict[str, Any],
    candidate_development: dict[str, Any],
    *,
    directional: dict[str, Any],
    deterministic_replay_max_difference: float,
    teacher_identity: dict[str, Any],
    teacher_positive: dict[str, Any],
    endpoint_image_difference_max: float,
    parameters_restored_exactly: bool,
) -> dict[str, Any]:
    reasons = []
    train_improvement = (
        baseline_train["joint_normalized_mse"] - candidate_train["joint_normalized_mse"]
    )
    development_improvement = (
        baseline_development["joint_normalized_mse"] - candidate_development["joint_normalized_mse"]
    )
    if train_improvement < MINIMUM_TRAIN_OBJECTIVE_IMPROVEMENT:
        reasons.append("training joint normalized MSE did not improve by at least 1e-4")
    if development_improvement <= 0.0:
        reasons.append("development joint normalized MSE did not improve")
    for label, baseline, candidate in (
        ("training", baseline_train, candidate_train),
        ("development", baseline_development, candidate_development),
    ):
        for name in ("common", "height", "damping"):
            if (
                candidate["component_nrmse"][name]
                > baseline["component_nrmse"][name] + COMPONENT_BASELINE_TOLERANCE
            ):
                reasons.append(f"{label} {name} NRMSE exceeded baseline plus 0.02")
        if max(candidate["rpy_source_nrmse"].values()) > RPY_NRMSE_LIMIT:
            reasons.append(f"{label} dynamic RPY source NRMSE exceeded 0.05")
        if candidate["motor_output_max_absolute"] > 1.0:
            reasons.append(f"{label} motor output exceeded [-1, 1]")
        if not candidate["all_attitude_cache_states_valid"]:
            reasons.append(f"{label} attitude cache contains invalid physical state")
    if not directional["pass"]:
        reasons.append("actual Adam displacement failed the directional derivative audit")
    if deterministic_replay_max_difference > REPLAY_TOLERANCE:
        reasons.append("source replay was not deterministic to 1e-6")
    if not teacher_identity["pass"]:
        reasons.append("teacher labels failed factorial identity reconstruction")
    if not teacher_positive["pass"]:
        reasons.append("analytical teacher failed the foreleg/stick positive control")
    if endpoint_image_difference_max != 0.0:
        reasons.append("opposite-motion endpoint images were not exactly identical")
    if not parameters_restored_exactly:
        reasons.append("student parameters were not restored bit-exactly")
    finite = _numeric_tree_is_finite(
        [
            baseline_train,
            candidate_train,
            baseline_development,
            candidate_development,
            directional,
            teacher_identity,
            teacher_positive,
        ]
    )
    if not finite:
        reasons.append("one or more report metrics were nonfinite")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "training_joint_mse_improvement": train_improvement,
        "development_joint_mse_improvement": development_improvement,
        "all_metrics_finite": finite,
    }


def _copy_parameters(controller: ConnectomeController) -> dict[str, Tensor]:
    return {name: getattr(controller, name).detach().clone() for name in PARAMETER_FAMILIES}


@torch.no_grad()
def _load_parameters(controller: ConnectomeController, values: dict[str, Tensor]) -> None:
    for name, value in values.items():
        getattr(controller, name).copy_(value)


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    if round(1.0 / config.dt) % args.policy_hz:
        raise SystemExit("policy frequency must divide the physics frequency")
    started = perf_counter()
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    graph_sha256 = responsibility.file_sha256(args.graph)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("source checkpoint graph hash does not match --graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source.load_state_dict(loaded["controller"])
    student.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    student.eval()
    source_parameters = _copy_parameters(student)

    print(json.dumps({"stage": "building_train_cache"}), flush=True)
    train_factorial = build_factorial_cache(
        scenes=args.scenes,
        seed=args.train_seed,
        response_steps=args.response_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
    )
    train_attitude = build_attitude_cache(
        source,
        scenes=args.scenes,
        seed=args.train_seed + 1,
        prefix_steps=args.prefix_steps,
        response_steps=args.response_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
    )
    print(json.dumps({"stage": "building_development_cache"}), flush=True)
    development_factorial = build_factorial_cache(
        scenes=args.scenes,
        seed=args.development_seed,
        response_steps=args.response_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
    )
    development_attitude = build_attitude_cache(
        source,
        scenes=args.scenes,
        seed=args.development_seed + 1,
        prefix_steps=args.prefix_steps,
        response_steps=args.response_steps,
        policy_hz=args.policy_hz,
        device=device,
        config=config,
    )
    scales = training_teacher_scales(train_factorial)
    teacher_identity = teacher_identity_report(train_factorial, scales)
    teacher_positive = teacher_plant_positive_control(
        train_factorial,
        device=device,
        config=config,
    )

    print(json.dumps({"stage": "baseline_replay"}), flush=True)
    baseline_train, baseline_train_raw = evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    replay_train, replay_train_raw = evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    baseline_development, _ = evaluate_bank(
        student,
        development_factorial,
        development_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    replay_difference = _max_prediction_difference(baseline_train_raw, replay_train_raw)

    optimizer = torch.optim.Adam(
        [
            {
                "params": [student.edge_magnitude, student.bias],
                "lr": EDGE_BIAS_LEARNING_RATE,
            },
            {"params": [student.raw_time_constant], "lr": TIME_CONSTANT_LEARNING_RATE},
        ],
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    print(json.dumps({"stage": "accumulating_full_native_gradient"}), flush=True)
    accumulated_gradient(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    raw_gradients = {
        name: getattr(student, name).grad.detach().clone() for name in PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), GRADIENT_NORM_CAP)
    optimizer.step()
    student.project_parameters()
    candidate_parameters = _copy_parameters(student)
    displacement = {
        name: candidate_parameters[name] - source_parameters[name] for name in PARAMETER_FAMILIES
    }
    derivative = float(
        sum((raw_gradients[name] * displacement[name]).sum() for name in PARAMETER_FAMILIES)
    )

    print(json.dumps({"stage": "candidate_replay"}), flush=True)
    candidate_train, _ = evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    candidate_development, _ = evaluate_bank(
        student,
        development_factorial,
        development_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )

    finite_difference_parameters = {
        name: source_parameters[name] + FINITE_DIFFERENCE_SCALE * displacement[name]
        for name in PARAMETER_FAMILIES
    }
    _load_parameters(student, finite_difference_parameters)
    student.project_parameters()
    finite_difference_train, _ = evaluate_bank(
        student,
        train_factorial,
        train_attitude,
        scales,
        prefix_steps=args.prefix_steps,
        device=device,
    )
    finite_difference = (
        finite_difference_train["joint_normalized_mse"] - baseline_train["joint_normalized_mse"]
    ) / FINITE_DIFFERENCE_SCALE
    relative_error = abs(finite_difference - derivative) / max(
        abs(finite_difference), abs(derivative), 1.0e-12
    )
    directional = {
        "pass": (
            derivative < 0.0
            and finite_difference < 0.0
            and relative_error <= FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        ),
        "autograd_directional_derivative": derivative,
        "forward_finite_difference": finite_difference,
        "relative_error": relative_error,
        "scale": FINITE_DIFFERENCE_SCALE,
    }

    _load_parameters(student, source_parameters)
    parameters_restored_exactly = all(
        torch.equal(getattr(student, name).detach(), source_parameters[name])
        for name in PARAMETER_FAMILIES
    )
    endpoint_image_difference = max(
        train_factorial.endpoint_image_difference_max,
        development_factorial.endpoint_image_difference_max,
    )
    final_decision = decision(
        baseline_train,
        candidate_train,
        baseline_development,
        candidate_development,
        directional=directional,
        deterministic_replay_max_difference=replay_difference,
        teacher_identity=teacher_identity,
        teacher_positive=teacher_positive,
        endpoint_image_difference_max=endpoint_image_difference,
        parameters_restored_exactly=parameters_restored_exactly,
    )
    report = {
        "experiment": EXPERIMENT,
        "status": "diagnostic_only_no_retained_parameter_changes",
        "pass": final_decision["pass"],
        "decision": final_decision,
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
        },
        "actor_contract_unchanged": True,
        "protocol": protocol_manifest(args),
        "parameter_counts": {
            "edge_magnitudes": student.edge_magnitude.numel(),
            "biases": student.bias.numel(),
            "raw_time_constants": student.raw_time_constant.numel(),
        },
        "cache": {
            "training_factorial_sha256": train_factorial.sha256,
            "training_attitude_sha256": train_attitude.sha256,
            "development_factorial_sha256": development_factorial.sha256,
            "development_attitude_sha256": development_attitude.sha256,
            "attitude_seed_offsets": {"training": 1, "development": 1},
            "endpoint_opposite_motion_image_max_absolute_difference": (endpoint_image_difference),
            "training_distributions": cache_distribution_report(train_factorial, train_attitude),
            "development_distributions": cache_distribution_report(
                development_factorial, development_attitude
            ),
        },
        "scene_protocol": {
            "visual_randomization": visual_scene_manifest(),
            "height_randomization": variable_hover.protocol_manifest(),
            "preflight_override": (
                "Both fixed banks balance all four visual style combinations; their "
                "continuous nuisance values and seeds remain disjoint."
            ),
        },
        "teacher_component_scales_motor_units": scales,
        "teacher_factorial_identity": teacher_identity,
        "teacher_foreleg_stick_positive_control": teacher_positive,
        "deterministic_source_replay_max_absolute_difference": replay_difference,
        "baseline_training": baseline_train,
        "candidate_training": candidate_train,
        "baseline_development": baseline_development,
        "candidate_development": candidate_development,
        "directional_derivative": directional,
        "gradient_norm_before_clipping": float(gradient_norm),
        "actual_displacement_family_rms": {
            name: float(displacement[name].square().mean().sqrt()) for name in PARAMETER_FAMILIES
        },
        "parameters_restored_exactly": parameters_restored_exactly,
        "promoted": False,
        "closed_loop_hover_run": False,
        "elapsed_seconds": perf_counter() - started,
        "interpretation_limit": (
            "A pass establishes one-step full-native joint teacher-fitting learnability only. "
            "It neither retains a controller nor demonstrates closed-loop hover."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "pass": report["pass"],
                "reasons": final_decision["reasons"],
                "training_joint_mse_improvement": final_decision["training_joint_mse_improvement"],
                "development_joint_mse_improvement": final_decision[
                    "development_joint_mse_improvement"
                ],
                "directional_derivative": directional,
                "parameters_restored_exactly": parameters_restored_exactly,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
