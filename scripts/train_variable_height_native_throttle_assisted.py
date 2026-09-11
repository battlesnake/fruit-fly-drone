#!/usr/bin/env python3
"""Train native visual throttle under training-only analytical attitude assistance."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_factorial_damping as factorial  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import audit_variable_height_full_native_joint as joint  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_hover as hover_train  # noqa: E402

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
from flydrone.variable_hover import (  # noqa: E402
    CAMERA_HEIGHT_BANDS,
    HELD_OUT_ABSOLUTE_MARKER_BAND,
    TRAIN_ABSOLUTE_MARKER_BANDS,
)
from flydrone.visual_hover import (  # noqa: E402
    DEFAULT_VISUAL_CAMERA,
    VisualScene,
    render_visual_hover_scene,
    sample_visual_scenes,
    visual_scene_manifest,
)

EXPERIMENT = "variable-height-native-throttle-assisted-v1"
PROTOCOL_COMMIT = "6d2c8c2"
EXPECTED_GRAPH_SHA256 = "8c6ba28d149e9ac4a5223c5919657a1734f2c2cac9828c51114fd0174a383665"
EXPECTED_CHECKPOINT_SHA256 = "7238b0e3ca39dc1a8bfd35dcf9f8b6fc9e8c8881ff989f64cb135fa6fed4e572"

POLICY_HZ = 50
PHYSICS_HZ = 100
EPISODE_SECONDS = 6.0
EPISODE_STEPS = int(EPISODE_SECONDS * POLICY_HZ)
FINAL_WINDOW_STEPS = 2 * POLICY_HZ
TRAIN_WINDOW_STEPS = 25
DENSE_BANK_CASES = 16
MOTION_BANK_CASES = 24
DENSE_SAMPLES_PER_UPDATE = 8
MOTION_SAMPLES_PER_UPDATE = 8
MAX_ACCEPTED_UPDATES = 100
MAX_ATTEMPTED_UPDATES = 125
MAX_CONSECUTIVE_REJECTIONS = 5
MIDPOINT_UPDATE = 50
FINAL_UPDATE = 100
MINIMUM_OBJECTIVE_IMPROVEMENT = 1.0e-4
FINITE_DIFFERENCE_SCALE = 0.0625
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
DENOMINATOR_RMS_FLOOR = 0.01
GRADIENT_NORM_CAP = joint.GRADIENT_NORM_CAP
EDGE_BIAS_LEARNING_RATE = joint.EDGE_BIAS_LEARNING_RATE
TIME_CONSTANT_LEARNING_RATE = joint.TIME_CONSTANT_LEARNING_RATE
PARAMETER_FAMILIES = joint.PARAMETER_FAMILIES
OPTIMIZER_BETAS = (0.9, 0.999)

POSITIVE_CONTROL_SEED = 380_983
TEACHER_HISTORY_SEEDS = (381_983, 382_983, 383_983, 384_983)
STUDENT_HISTORY_SEEDS = (None, 392_983, 393_983, 394_983)
MOTION_HISTORY_SEEDS = (401_983, 402_983, 403_983, 404_983)
OPTIMIZER_SAMPLING_SEED = 410_983
MIDPOINT_HOVER_SEED = 420_983
MIDPOINT_MOTION_SEED = 420_984
FINAL_HOVER_SEED = 430_983
FINAL_MOTION_SEED = 430_984
BOOTSTRAP_SEED = 440_983
FORMAL_SEEDS = frozenset(
    {
        POSITIVE_CONTROL_SEED,
        *TEACHER_HISTORY_SEEDS,
        *(seed for seed in STUDENT_HISTORY_SEEDS if seed is not None),
        *MOTION_HISTORY_SEEDS,
        OPTIMIZER_SAMPLING_SEED,
        MIDPOINT_HOVER_SEED,
        MIDPOINT_MOTION_SEED,
        FINAL_HOVER_SEED,
        FINAL_MOTION_SEED,
        BOOTSTRAP_SEED,
    }
)

TRAIN_MOTION_HEIGHT_AMPLITUDES = (0.05, 0.10)
TRAIN_MOTION_SPEED_AMPLITUDES = (0.15, 0.30)
FINAL_MOTION_HEIGHT_AMPLITUDES = (0.075, 0.125)
FINAL_MOTION_SPEED_AMPLITUDES = (0.20, 0.35)
MOTION_HISTORY_LENGTHS = (15, 20, 25)
MOTION_PREFIX_STEPS = 25
MOTION_APPROACH_FRACTION = 0.60
MOTION_VELOCITY_SIGNS = (-1.0, 1.0)

POSITIVE_CONTROL_SUCCESS_MINIMUM = 0.95
MIDPOINT_SUCCESS_MINIMUM = 0.50
MIDPOINT_MOTION_SIGN_MINIMUM = 0.50
FINAL_SUCCESS_MINIMUM = 0.90
FINAL_MOTION_SIGN_MINIMUM = 0.90
FINAL_MOTION_GAIN_RANGE = (0.5, 1.5)
HEIGHT_RMSE_LIMIT_M = 0.10
VERTICAL_SPEED_RMS_LIMIT_MPS = 0.10
TILT_RMS_LIMIT_DEG = 5.0
BOOTSTRAP_SAMPLES = 10_000

STEP_TIME_RANGE = (1.0, 2.0)
IMPULSE_TIME_RANGE = (2.0, 3.0)
NO_STEP_FREEZE_SECONDS = 1.5
INITIAL_VERTICAL_SPEED_MAX = 0.20
MARKER_STEP_METRES = 0.20
IMPULSE_MAGNITUDES = (0.15, 0.30)


def require_nonformal_seeds(*seeds: int) -> None:
    collisions = sorted(FORMAL_SEEDS.intersection(seeds))
    if collisions:
        raise ValueError(f"disposable execution attempted to consume formal seeds: {collisions}")


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
        default=REPO_ROOT / "runs/variable-height-hover/native-throttle-assisted-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source": "original native visual-hover checkpoint",
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "privileged_inputs": [],
            "state": "native MaleCNS recurrence only",
            "outputs_computed": ["roll", "pitch", "yaw", "throttle"],
            "physical_training_ownership": {
                "teacher": ["roll", "pitch", "yaw"],
                "native_fly": ["throttle"],
            },
            "motor_merge_before_foreleg_stick_plant": True,
        },
        "plant": {
            "nominal_mass_only": True,
            "policy_hz": POLICY_HZ,
            "physics_hz": PHYSICS_HZ,
            "episode_seconds": EPISODE_SECONDS,
            "initialization": "airborne and hover-stick-settled",
        },
        "parameters": {
            "families": list(PARAMETER_FAMILIES),
            "edge_and_bias_learning_rate": EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": TIME_CONSTANT_LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "fresh_adam": True,
            "adam_betas": list(OPTIMIZER_BETAS),
            "topology_signs_and_interfaces_frozen": True,
            "rpy_source_constraints": False,
        },
        "curriculum": {
            "updates_1_through_25": "teacher histories only",
            "updates_26_through_100": "equal teacher and current-native-throttle histories",
            "block_boundaries": [1, 26, 51, 76],
            "teacher_bank_cases": DENSE_BANK_CASES,
            "student_bank_cases": DENSE_BANK_CASES,
            "teacher_history_seeds": list(TEACHER_HISTORY_SEEDS),
            "student_history_seeds": list(STUDENT_HISTORY_SEEDS),
            "motion_history_seeds": list(MOTION_HISTORY_SEEDS),
            "optimizer_sampling_seed": OPTIMIZER_SAMPLING_SEED,
            "long_plant_bptt": False,
            "recurrent_window_steps": TRAIN_WINDOW_STEPS,
            "zero_state_prefix_replay": True,
            "detached_burn_in_fixed_across_gradient_fd_and_trials": True,
        },
        "case_distribution": {
            "stationary_up_down_marker_fraction": [0.5, 0.25, 0.25],
            "none_positive_negative_impulse_fraction": [0.5, 0.25, 0.25],
            "marker_step_metres": MARKER_STEP_METRES,
            "training_up_step_metres": [[0.55, 0.65], [0.75, 0.85]],
            "training_down_step_metres": [[1.35, 1.45], [1.15, 1.25]],
            "initial_vertical_speed_absolute_range_mps": [0.0, INITIAL_VERTICAL_SPEED_MAX],
            "impulse_magnitudes_mps": list(IMPULSE_MAGNITUDES),
            "step_time_range_seconds": list(STEP_TIME_RANGE),
            "impulse_time_range_seconds": list(IMPULSE_TIME_RANGE),
            "post_impulse_unsupervised_frames": 2,
            "visual_scene": visual_scene_manifest(),
        },
        "objective": {
            "dense_and_motion_weight": [0.5, 0.5],
            "dense_examples_per_update": DENSE_SAMPLES_PER_UPDATE,
            "motion_examples_per_update": MOTION_SAMPLES_PER_UPDATE,
            "denominator_rms_floor_motor_units": DENOMINATOR_RMS_FLOOR,
            "denominators_frozen_from_first_teacher_and_motion_banks": True,
            "motion_history_lengths": list(MOTION_HISTORY_LENGTHS),
            "motion_prefix_steps": MOTION_PREFIX_STEPS,
            "training_motion_height_amplitudes_metres": list(TRAIN_MOTION_HEIGHT_AMPLITUDES),
            "training_motion_speed_amplitudes_mps": list(TRAIN_MOTION_SPEED_AMPLITUDES),
            "final_motion_height_amplitudes_metres": list(FINAL_MOTION_HEIGHT_AMPLITUDES),
            "final_motion_speed_amplitudes_mps": list(FINAL_MOTION_SPEED_AMPLITUDES),
            "endpoint_rgb_and_attitude_equal": True,
            "one_opposite_velocity_pair_per_example": True,
            "height_error_sign_balanced_across_examples": True,
        },
        "optimizer_transaction": {
            "one_gradient_and_adam_call_per_attempt": True,
            "finite_difference_scale": FINITE_DIFFERENCE_SCALE,
            "finite_difference_relative_error_limit": (FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT),
            "ordinary_scales_descending": list(BACKTRACK_SCALES),
            "minimum_fixed_burn_in_and_full_prefix_improvement": (MINIMUM_OBJECTIVE_IMPROVEMENT),
            "numerical_failure_is_terminal": True,
            "finite_no_scale_failure_advances_sampling_rng": True,
            "maximum_consecutive_ordinary_rejections": MAX_CONSECUTIVE_REJECTIONS,
            "projection_or_repair": False,
        },
        "positive_control": {
            "seed": POSITIVE_CONTROL_SEED,
            "cases": 64,
            "teacher_owns_all_axes_through_forelegs": True,
            "minimum_success_fraction": POSITIVE_CONTROL_SUCCESS_MINIMUM,
        },
        "midpoint": {
            "accepted_update": MIDPOINT_UPDATE,
            "hover_seed": MIDPOINT_HOVER_SEED,
            "motion_seed": MIDPOINT_MOTION_SEED,
            "cases_and_pairs": 64,
            "minimum_success_fraction": MIDPOINT_SUCCESS_MINIMUM,
            "minimum_motion_correct_sign_fraction": MIDPOINT_MOTION_SIGN_MINIMUM,
        },
        "final": {
            "accepted_update": FINAL_UPDATE,
            "hover_seed": FINAL_HOVER_SEED,
            "motion_seed": FINAL_MOTION_SEED,
            "cases_and_pairs": 128,
            "minimum_success_fraction": FINAL_SUCCESS_MINIMUM,
            "minimum_motion_correct_sign_fraction": FINAL_MOTION_SIGN_MINIMUM,
            "motion_gain_range": list(FINAL_MOTION_GAIN_RANGE),
            "paired_live_minus_frozen_bootstrap_seed": BOOTSTRAP_SEED,
            "paired_live_minus_frozen_lower_95_bound_positive": True,
            "height_rmse_limit_metres": HEIGHT_RMSE_LIMIT_M,
            "vertical_speed_rms_limit_mps": VERTICAL_SPEED_RMS_LIMIT_MPS,
            "tilt_rms_limit_degrees": TILT_RMS_LIMIT_DEG,
            "final_data_generated_only_at_update_100": True,
        },
        "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
        "maximum_attempted_updates": MAX_ATTEMPTED_UPDATES,
        "passing_authorizes": "native attitude reintegration experiment only",
        "full_native_hover": False,
        "promotion": False,
        "gate_flight": False,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test:
        raise SystemExit("the preregistered assisted-throttle run has no smoke variant")
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if responsibility.file_sha256(args.graph) != EXPECTED_GRAPH_SHA256:
        raise SystemExit("graph does not match the preregistered source")
    if responsibility.file_sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("checkpoint does not match the preregistered source")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the assisted-throttle experiment already has a terminal report")


def _uniform_by_family(family: Tensor, bands: tuple[tuple[float, float], ...]) -> Tensor:
    bounds = torch.tensor(bands, device=family.device, dtype=torch.float32)
    unit = torch.rand(family.shape, device=family.device)
    return bounds[family, 0] + unit * (bounds[family, 1] - bounds[family, 0])


def _balanced_codes(
    batch: int, values: tuple[int, ...], counts: tuple[int, ...], device: torch.device
) -> Tensor:
    if sum(counts) != batch or len(values) != len(counts):
        raise ValueError("balanced-code counts must cover the batch")
    result = torch.tensor(
        [value for value, count in zip(values, counts, strict=True) for _ in range(count)],
        device=device,
        dtype=torch.long,
    )
    return result[torch.randperm(batch, device=device)]


def _scene_dict(scene: VisualScene) -> dict[str, Tensor]:
    return {name: value.detach().cpu() for name, value in asdict(scene).items()}


def _scene_from_dict(
    values: dict[str, Tensor], *, device: torch.device, indices: Tensor | None = None
) -> VisualScene:
    selected: dict[str, Tensor] = {}
    for name, value in values.items():
        if indices is not None:
            value = value[indices]
        selected[name] = value.to(device)
    return VisualScene(**selected)


def _cat_scene_dicts(values: Iterable[dict[str, Tensor]]) -> dict[str, Tensor]:
    items = list(values)
    return {name: torch.cat([item[name] for item in items]) for name in items[0]}


def _state_dict(state: QuadState) -> dict[str, Tensor]:
    names = ("position", "velocity", "euler", "rates", "actuator", "specific_force")
    return {name: value.detach().cpu() for name, value in zip(names, state.as_tuple(), strict=True)}


def _state_from_dict(
    values: dict[str, Tensor], *, device: torch.device, indices: Tensor | None = None
) -> QuadState:
    fields = []
    for name in ("position", "velocity", "euler", "rates", "actuator", "specific_force"):
        value = values[name]
        if indices is not None:
            value = value[indices]
        fields.append(value.to(device))
    return QuadState(*fields)


def _stick_dict(state: StickState) -> dict[str, Tensor]:
    return {
        "joint_position": state.joint_position.detach().cpu(),
        "joint_velocity": state.joint_velocity.detach().cpu(),
        "position": state.position.detach().cpu(),
        "velocity": state.velocity.detach().cpu(),
    }


def _stick_from_dict(
    values: dict[str, Tensor], *, device: torch.device, indices: Tensor | None = None
) -> StickState:
    fields = []
    for name in ("joint_position", "joint_velocity", "position", "velocity"):
        value = values[name]
        if indices is not None:
            value = value[indices]
        fields.append(value.to(device))
    return StickState(*fields)


def _index_nested_tensors(values: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {name: value[indices] for name, value in values.items()}


def _concatenate_case_banks(banks: list[dict[str, Any]], *, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "initial_state": {
            name: torch.cat([bank["initial_state"][name] for bank in banks])
            for name in banks[0]["initial_state"]
        },
        "initial_stick": {
            name: torch.cat([bank["initial_stick"][name] for bank in banks])
            for name in banks[0]["initial_stick"]
        },
        "scene": _cat_scene_dicts([bank["scene"] for bank in banks]),
    }
    tensor_fields = (
        "initial_marker",
        "final_marker",
        "step_code",
        "step_index",
        "impulse",
        "impulse_index",
        "split_code",
        "camera_family",
        "marker_family",
    )
    for name in tensor_fields:
        result[name] = torch.cat([bank[name] for bank in banks])
    total = len(result["initial_marker"])
    permutation = torch.randperm(total, device=device).cpu()
    result["initial_state"] = _index_nested_tensors(result["initial_state"], permutation)
    result["initial_stick"] = _index_nested_tensors(result["initial_stick"], permutation)
    result["scene"] = _index_nested_tensors(result["scene"], permutation)
    for name in tensor_fields:
        result[name] = result[name][permutation]
    return result


def sample_hover_cases(
    batch: int,
    *,
    split: str,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    if batch % 16:
        raise ValueError("hover case batches must be divisible by 16")
    if split not in {"train", "held_out_marker", "held_out_combination"}:
        raise ValueError(f"unknown assisted-hover split: {split}")
    step_code = _balanced_codes(batch, (0, 1, -1), (batch // 2, batch // 4, batch // 4), device)
    impulse_code = _balanced_codes(batch, (0, 1, -1), (batch // 2, batch // 4, batch // 4), device)
    marker_family = torch.randint(0, 2, (batch,), device=device)
    camera_family = marker_family.clone()
    initial_marker = torch.empty(batch, device=device)
    final_marker = torch.empty(batch, device=device)

    stationary = step_code == 0
    up = step_code == 1
    down = step_code == -1
    unit = torch.rand(batch, device=device)
    if split == "held_out_marker":
        initial_marker[stationary] = HELD_OUT_ABSOLUTE_MARKER_BAND[0] + unit[stationary] * (
            HELD_OUT_ABSOLUTE_MARKER_BAND[1] - HELD_OUT_ABSOLUTE_MARKER_BAND[0]
        )
        initial_marker[up] = 0.70 + 0.15 * unit[up]
        initial_marker[down] = 1.15 + 0.15 * unit[down]
        marker_family[up] = 0
        marker_family[down] = 1
        camera_family = marker_family.clone()
    else:
        stationary_marker = _uniform_by_family(marker_family, TRAIN_ABSOLUTE_MARKER_BANDS)
        initial_marker[stationary] = stationary_marker[stationary]
        initial_marker[up] = 0.55 + 0.10 * unit[up]
        initial_marker[down] = 1.35 + 0.10 * unit[down]
        marker_family[up] = 0
        marker_family[down] = 1
        camera_family = marker_family.clone()
        if split == "held_out_combination":
            camera_family = 1 - marker_family
    final_marker.copy_(initial_marker)
    final_marker[up] += MARKER_STEP_METRES
    final_marker[down] -= MARKER_STEP_METRES

    camera_height = _uniform_by_family(camera_family, CAMERA_HEIGHT_BANDS)
    state, sticks = hover_train.nominal_initial_state(
        batch,
        camera_height=camera_height,
        device=device,
        config=config,
        attitude_degrees=5.0,
        rate_degrees_per_second=12.0,
        vertical_speed=0.0,
    )
    vertical_sign = _balanced_codes(batch, (-1, 1), (batch // 2, batch // 2), device)
    state.velocity[:, 2] = (
        vertical_sign * torch.rand(batch, device=device) * INITIAL_VERTICAL_SPEED_MAX
    )

    step_low = round(STEP_TIME_RANGE[0] * POLICY_HZ)
    step_high = round(STEP_TIME_RANGE[1] * POLICY_HZ)
    step_index = torch.randint(step_low, step_high, (batch,), device=device)
    step_index[stationary] = -1
    impulse_low = round(IMPULSE_TIME_RANGE[0] * POLICY_HZ)
    impulse_high = round(IMPULSE_TIME_RANGE[1] * POLICY_HZ)
    impulse_index = torch.randint(impulse_low, impulse_high, (batch,), device=device)
    impulse_index[impulse_code == 0] = -1
    nonzero_count = batch // 2
    impulse_magnitude = torch.tensor(
        list(IMPULSE_MAGNITUDES) * math.ceil(nonzero_count / len(IMPULSE_MAGNITUDES)),
        device=device,
    )[:nonzero_count]
    impulse_magnitude = impulse_magnitude[torch.randperm(nonzero_count, device=device)]
    impulse = torch.zeros(batch, device=device)
    impulse[impulse_code != 0] = impulse_code[impulse_code != 0].float() * impulse_magnitude
    scene = sample_visual_scenes(
        batch,
        device=device,
        held_out_combinations=split == "held_out_combination",
    )
    split_value = {"train": 0, "held_out_marker": 1, "held_out_combination": 2}[split]
    return {
        "initial_state": _state_dict(state),
        "initial_stick": _stick_dict(sticks),
        "scene": _scene_dict(scene),
        "initial_marker": initial_marker.cpu(),
        "final_marker": final_marker.cpu(),
        "step_code": step_code.cpu(),
        "step_index": step_index.cpu(),
        "impulse": impulse.cpu(),
        "impulse_index": impulse_index.cpu(),
        "split_code": torch.full((batch,), split_value, dtype=torch.long),
        "camera_family": camera_family.cpu(),
        "marker_family": marker_family.cpu(),
    }


def build_hover_case_bank(
    *,
    seed: int,
    counts: dict[str, int],
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    responsibility.seed_everything(seed)
    banks = [
        sample_hover_cases(count, split=split, device=device, config=config)
        for split, count in counts.items()
        if count
    ]
    result = _concatenate_case_banks(banks, device=device)
    result["seed"] = seed
    result["split_counts"] = dict(counts)
    result["sha256"] = audit.semantic_sha256(
        {name: value for name, value in result.items() if name != "sha256"}
    )
    return result


def hover_case_support_report(cases: dict[str, Any]) -> dict[str, Any]:
    return {
        "cases": len(cases["initial_marker"]),
        "seed": cases["seed"],
        "sha256": cases["sha256"],
        "split_counts": cases["split_counts"],
        "initial_marker_range_metres": [
            float(cases["initial_marker"].min()),
            float(cases["initial_marker"].max()),
        ],
        "final_marker_range_metres": [
            float(cases["final_marker"].min()),
            float(cases["final_marker"].max()),
        ],
        "camera_height_range_metres": [
            float(cases["initial_state"]["position"][:, 2].min()),
            float(cases["initial_state"]["position"][:, 2].max()),
        ],
        "stationary_up_down_counts": [
            int((cases["step_code"] == value).sum()) for value in (0, 1, -1)
        ],
        "none_positive_negative_impulse_counts": [
            int((cases["impulse"] == 0).sum()),
            int((cases["impulse"] > 0).sum()),
            int((cases["impulse"] < 0).sum()),
        ],
    }


def _clone_state(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def _clone_stick(state: StickState) -> StickState:
    return StickState(
        state.joint_position.clone(),
        state.joint_velocity.clone(),
        state.position.clone(),
        state.velocity.clone(),
    )


def _apply_case_events(
    state: QuadState,
    marker: Tensor,
    cases: dict[str, Any],
    *,
    step: int,
    device: torch.device,
) -> tuple[QuadState, Tensor]:
    step_index = cases["step_index"].to(device)
    final_marker = cases["final_marker"].to(device)
    marker = torch.where(step_index == step, final_marker, marker)
    impulse_index = cases["impulse_index"].to(device)
    impulse = cases["impulse"].to(device)
    if bool((impulse_index == step).any()):
        velocity = state.velocity.clone()
        velocity[:, 2] += torch.where(impulse_index == step, impulse, torch.zeros_like(impulse))
        state = QuadState(
            state.position,
            velocity,
            state.euler,
            state.rates,
            state.actuator,
            state.specific_force,
        )
    return state, marker


def merge_teacher_attitude_native_throttle(teacher_motor: Tensor, native_motor: Tensor) -> Tensor:
    if teacher_motor.shape != native_motor.shape or teacher_motor.shape[-1] != 4:
        raise ValueError("teacher and native motor tensors must have equal (..., 4) shape")
    return torch.cat((teacher_motor[..., :3], native_motor[..., 3:4]), dim=-1)


@torch.no_grad()
def collect_trajectory_bank(
    controller: ConnectomeController,
    cases: dict[str, Any],
    *,
    history_type: str,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    if history_type not in {"teacher", "student"}:
        raise ValueError("history_type must be teacher or student")
    batch = len(cases["initial_marker"])
    state = _state_from_dict(cases["initial_state"], device=device)
    sticks_state = _stick_from_dict(cases["initial_stick"], device=device)
    scene = _scene_from_dict(cases["scene"], device=device)
    marker = cases["initial_marker"].to(device).clone()
    neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    physics_steps = PHYSICS_HZ // POLICY_HZ
    stored_state: dict[str, list[Tensor]] = {
        name: []
        for name in ("position", "velocity", "euler", "rates", "actuator", "specific_force")
    }
    markers: list[Tensor] = []
    teacher_motors: list[Tensor] = []
    eligible: list[Tensor] = []
    valid = hover_train.state_is_valid(state)
    impulse_index = cases["impulse_index"].to(device)
    for step in range(EPISODE_STEPS):
        state, marker = _apply_case_events(state, marker, cases, step=step, device=device)
        valid &= hover_train.state_is_valid(state)
        for name, value in zip(stored_state, state.as_tuple(), strict=True):
            stored_state[name].append(value.cpu())
        markers.append(marker.cpu())
        target = hover_train.teacher_motor(state, marker, config)
        teacher_motors.append(target.cpu())
        observable = (impulse_index < 0) | (step < impulse_index) | (step >= impulse_index + 2)
        eligible.append((valid & observable).cpu())

        if history_type == "student":
            image = render_visual_hover_scene(state, marker, config=config, scene=scene)
            native_motor, neural = controller(image, state.euler[:, :2], neural)
            physical_motor = merge_teacher_attitude_native_throttle(target, native_motor)
        else:
            physical_motor = target
        state, sticks_state, _ = hover_train.advance_physics(
            quad,
            sticks,
            physical_motor,
            state,
            sticks_state,
            physics_steps,
        )
        state = state.detach()
        sticks_state = sticks_state.detach()
    payload = {
        "history_type": history_type,
        "case_seed": cases["seed"],
        "case_sha256": cases["sha256"],
        "states": {name: torch.stack(values, dim=1) for name, values in stored_state.items()},
        "marker": torch.stack(markers, dim=1),
        "teacher_motor": torch.stack(teacher_motors, dim=1),
        "eligible": torch.stack(eligible, dim=1),
        "scene": copy.deepcopy(cases["scene"]),
    }
    payload["sha256"] = audit.semantic_sha256(payload)
    return payload


def trajectory_bank_report(bank: dict[str, Any]) -> dict[str, Any]:
    return {
        "history_type": bank["history_type"],
        "case_seed": bank["case_seed"],
        "case_sha256": bank["case_sha256"],
        "sha256": bank["sha256"],
        "cases": int(bank["marker"].shape[0]),
        "policy_steps": int(bank["marker"].shape[1]),
        "eligible_fraction": float(bank["eligible"].float().mean()),
        "valid_final_fraction": float(bank["eligible"][:, -1].float().mean()),
        "height_range_metres": [
            float(bank["states"]["position"][..., 2].min()),
            float(bank["states"]["position"][..., 2].max()),
        ],
    }


def _slice_cases(cases: dict[str, Any], indices: Tensor) -> dict[str, Any]:
    result = {
        "initial_state": _index_nested_tensors(cases["initial_state"], indices),
        "initial_stick": _index_nested_tensors(cases["initial_stick"], indices),
        "scene": _index_nested_tensors(cases["scene"], indices),
    }
    for name in (
        "initial_marker",
        "final_marker",
        "step_code",
        "step_index",
        "impulse",
        "impulse_index",
        "split_code",
        "camera_family",
        "marker_family",
    ):
        result[name] = cases[name][indices]
    result["seed"] = cases["seed"]
    result["split_counts"] = cases["split_counts"]
    result["sha256"] = cases["sha256"]
    return result


@torch.no_grad()
def _evaluate_hover_slice(
    controller: ConnectomeController | None,
    cases: dict[str, Any],
    *,
    teacher_all_axes: bool,
    frozen_vision: bool,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Tensor]:
    batch = len(cases["initial_marker"])
    state = _state_from_dict(cases["initial_state"], device=device)
    stick_state = _stick_from_dict(cases["initial_stick"], device=device)
    scene = _scene_from_dict(cases["scene"], device=device)
    marker = cases["initial_marker"].to(device).clone()
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)
    neural = None
    if not teacher_all_axes:
        if controller is None:
            raise ValueError("assisted native evaluation requires a controller")
        neural = controller.initial_state(batch, device=device, dtype=torch.float32)
    freeze_index = torch.where(
        cases["step_index"].to(device) >= 0,
        cases["step_index"].to(device),
        torch.full((batch,), round(NO_STEP_FREEZE_SECONDS * POLICY_HZ), device=device),
    )
    frozen_image = torch.zeros(
        batch,
        3,
        DEFAULT_VISUAL_CAMERA.height,
        DEFAULT_VISUAL_CAMERA.width,
        device=device,
    )
    captured = torch.zeros(batch, device=device, dtype=torch.bool)
    height_values, speed_values, tilt_values = [], [], []
    valid = hover_train.state_is_valid(state)
    ground_contact = torch.zeros(batch, device=device, dtype=torch.bool)
    maximum_motor = torch.zeros(batch, device=device)
    physics_steps = PHYSICS_HZ // POLICY_HZ
    for step in range(EPISODE_STEPS):
        state, marker = _apply_case_events(state, marker, cases, step=step, device=device)
        teacher_motor = hover_train.teacher_motor(state, marker, config)
        if teacher_all_axes:
            physical_motor = teacher_motor
        else:
            assert controller is not None and neural is not None
            live_image = render_visual_hover_scene(state, marker, config=config, scene=scene)
            capture_now = (~captured) & (freeze_index == step)
            if bool(capture_now.any()):
                frozen_image[capture_now] = live_image[capture_now]
                captured |= capture_now
            image = (
                torch.where(captured[:, None, None, None], frozen_image, live_image)
                if frozen_vision
                else live_image
            )
            native_motor, neural = controller(image, state.euler[:, :2], neural)
            maximum_motor = torch.maximum(maximum_motor, native_motor.abs().amax(dim=1))
            physical_motor = merge_teacher_attitude_native_throttle(teacher_motor, native_motor)
        state, stick_state, _ = hover_train.advance_physics(
            quad,
            sticks,
            physical_motor,
            state,
            stick_state,
            physics_steps,
        )
        valid &= hover_train.state_is_valid(state)
        ground_contact |= state.position[:, 2] <= 0.03
        height_values.append(state.position[:, 2].cpu())
        speed_values.append(state.velocity[:, 2].cpu())
        tilt_values.append(torch.linalg.vector_norm(state.euler[:, :2], dim=1).cpu())
    height = torch.stack(height_values, dim=1)
    speed = torch.stack(speed_values, dim=1)
    tilt = torch.stack(tilt_values, dim=1)
    target = cases["final_marker"][:, None]
    final_height = height[:, -FINAL_WINDOW_STEPS:]
    final_speed = speed[:, -FINAL_WINDOW_STEPS:]
    final_tilt = tilt[:, -FINAL_WINDOW_STEPS:]
    height_rmse = ((final_height - target).square().mean(dim=1)).sqrt()
    speed_rms = final_speed.square().mean(dim=1).sqrt()
    tilt_rms_deg = torch.rad2deg(final_tilt.square().mean(dim=1).sqrt())
    success = (
        (~ground_contact.cpu())
        & valid.cpu()
        & (height_rmse <= HEIGHT_RMSE_LIMIT_M)
        & (speed_rms <= VERTICAL_SPEED_RMS_LIMIT_MPS)
        & (tilt_rms_deg <= TILT_RMS_LIMIT_DEG)
    )
    return {
        "success": success,
        "height_rmse": height_rmse,
        "vertical_speed_rms": speed_rms,
        "tilt_rms_degrees": tilt_rms_deg,
        "ground_contact": ground_contact.cpu(),
        "valid": valid.cpu(),
        "maximum_native_motor": maximum_motor.cpu(),
        "frozen_frame_captured": captured.cpu(),
    }


@torch.no_grad()
def evaluate_hover_cases(
    controller: ConnectomeController | None,
    cases: dict[str, Any],
    *,
    teacher_all_axes: bool,
    frozen_vision: bool,
    device: torch.device,
    config: HoverConfig,
    batch_size: int = 16,
) -> tuple[dict[str, Any], Tensor]:
    outputs: dict[str, list[Tensor]] = {}
    for begin in range(0, len(cases["initial_marker"]), batch_size):
        indices = torch.arange(begin, min(begin + batch_size, len(cases["initial_marker"])))
        result = _evaluate_hover_slice(
            controller,
            _slice_cases(cases, indices),
            teacher_all_axes=teacher_all_axes,
            frozen_vision=frozen_vision,
            device=device,
            config=config,
        )
        for name, value in result.items():
            outputs.setdefault(name, []).append(value)
    merged = {name: torch.cat(values) for name, values in outputs.items()}
    success = merged["success"]
    finite = all(
        torch.isfinite(value).all() for name, value in merged.items() if value.is_floating_point()
    )
    report = {
        "cases": int(len(success)),
        "success_fraction": float(success.float().mean()),
        "successes": int(success.sum()),
        "height_rmse_mean_metres": float(merged["height_rmse"].mean()),
        "height_rmse_p95_metres": float(torch.quantile(merged["height_rmse"], 0.95)),
        "vertical_speed_rms_mean_mps": float(merged["vertical_speed_rms"].mean()),
        "tilt_rms_mean_degrees": float(merged["tilt_rms_degrees"].mean()),
        "ground_contact_cases": int(merged["ground_contact"].sum()),
        "invalid_cases": int((~merged["valid"]).sum()),
        "maximum_native_motor_absolute": float(merged["maximum_native_motor"].max()),
        "all_metrics_finite": bool(finite),
        "teacher_all_axes": teacher_all_axes,
        "frozen_vision": frozen_vision,
        "all_frozen_frames_captured": bool(merged["frozen_frame_captured"].all())
        if frozen_vision
        else None,
    }
    return report, success


def paired_bootstrap_mean_interval(
    differences: Tensor,
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float]:
    values = differences.detach().cpu().float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means = []
    for _ in range(math.ceil(samples / 512)):
        count = min(512, samples - len(means) * 512)
        if count <= 0:
            break
        indices = torch.randint(0, len(values), (count, len(values)), generator=generator)
        means.append(values[indices].mean(dim=1))
    distribution = torch.cat(means)[:samples]
    return float(torch.quantile(distribution, 0.025)), float(torch.quantile(distribution, 0.975))


def _repeated_balanced(
    values: tuple[float | int, ...], count: int, *, device: torch.device
) -> Tensor:
    tensor = torch.tensor(values, device=device)
    repeated = tensor.repeat(math.ceil(count / len(values)))[:count]
    return repeated[torch.randperm(count, device=device)]


@torch.no_grad()
def build_motion_bank(
    *,
    seed: int,
    cases: int,
    height_amplitudes: tuple[float, ...],
    speed_amplitudes: tuple[float, ...],
    held_out_styles: bool,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    responsibility.seed_everything(seed)
    height_error = _repeated_balanced(height_amplitudes, cases, device=device).float()
    height_sign = _repeated_balanced((-1, 1), cases, device=device).float()
    speed = _repeated_balanced(speed_amplitudes, cases, device=device).float()
    horizon = _repeated_balanced(MOTION_HISTORY_LENGTHS, cases, device=device).long()
    family = _repeated_balanced((0, 1), cases, device=device).long()
    maximum_height_error = max(height_amplitudes)
    if held_out_styles and tuple(height_amplitudes) == FINAL_MOTION_HEIGHT_AMPLITUDES:
        centre_low, centre_high = 0.86, 1.14
        centre = centre_low + torch.rand(cases, device=device) * (centre_high - centre_low)
    else:
        centre_bands = (
            (
                TRAIN_ABSOLUTE_MARKER_BANDS[0][0] + maximum_height_error,
                TRAIN_ABSOLUTE_MARKER_BANDS[0][1] - maximum_height_error,
            ),
            (
                TRAIN_ABSOLUTE_MARKER_BANDS[1][0] + maximum_height_error,
                TRAIN_ABSOLUTE_MARKER_BANDS[1][1] - maximum_height_error,
            ),
        )
        centre = _uniform_by_family(family, centre_bands)
    state, _ = responsibility.base_state(cases, device=device, config=config)
    state = responsibility.state_at_height(state, centre)
    scene = sample_visual_scenes(
        cases,
        device=device,
        held_out_combinations=held_out_styles,
    )
    approach_steps = torch.maximum(
        torch.full_like(horizon, 5),
        torch.round(horizon.float() * MOTION_APPROACH_FRACTION).long(),
    )
    payload = {
        "seed": seed,
        "cases": cases,
        "height_error": height_error.cpu(),
        "height_sign": height_sign.cpu(),
        "speed": speed.cpu(),
        "horizon": horizon.cpu(),
        "approach_steps": approach_steps.cpu(),
        "state": _state_dict(state),
        "scene": _scene_dict(scene),
        "held_out_styles": held_out_styles,
        "height_amplitudes": list(height_amplitudes),
        "speed_amplitudes": list(speed_amplitudes),
    }
    payload["sha256"] = audit.semantic_sha256(payload)
    return payload


def motion_bank_report(bank: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": bank["seed"],
        "cases": bank["cases"],
        "sha256": bank["sha256"],
        "held_out_styles": bank["held_out_styles"],
        "height_amplitudes_metres": bank["height_amplitudes"],
        "speed_amplitudes_mps": bank["speed_amplitudes"],
        "history_length_counts": {
            str(length): int((bank["horizon"] == length).sum()) for length in MOTION_HISTORY_LENGTHS
        },
        "negative_positive_height_error_counts": [
            int((bank["height_sign"] == sign).sum()) for sign in (-1, 1)
        ],
    }


def _trajectory_state_at(sample: dict[str, Any], step: int, device: torch.device) -> QuadState:
    return QuadState(
        *(
            sample["states"][name][:, step].to(device)
            for name in ("position", "velocity", "euler", "rates", "actuator", "specific_force")
        )
    )


def materialize_dense_sample(
    teacher_bank: dict[str, Any],
    student_bank: dict[str, Any] | None,
    spec: dict[str, Any],
) -> dict[str, Any]:
    selections: list[tuple[dict[str, Any], Tensor]] = [
        (teacher_bank, torch.tensor(spec["teacher_indices"], dtype=torch.long))
    ]
    if spec["student_indices"]:
        if student_bank is None:
            raise ValueError("student sample indices require a student trajectory bank")
        selections.append((student_bank, torch.tensor(spec["student_indices"], dtype=torch.long)))
    result = {
        "states": {
            name: torch.cat([bank["states"][name][indices] for bank, indices in selections])
            for name in teacher_bank["states"]
        },
        "marker": torch.cat([bank["marker"][indices] for bank, indices in selections]),
        "teacher_motor": torch.cat(
            [bank["teacher_motor"][indices] for bank, indices in selections]
        ),
        "eligible": torch.cat([bank["eligible"][indices] for bank, indices in selections]),
        "scene": _cat_scene_dicts(
            [_index_nested_tensors(bank["scene"], indices) for bank, indices in selections]
        ),
        "window_start": int(spec["window_start"]),
    }
    if result["marker"].shape[0] != DENSE_SAMPLES_PER_UPDATE:
        raise ValueError("dense sample does not have the preregistered batch size")
    return result


def sample_update_spec(
    generator: torch.Generator,
    *,
    block_index: int,
    motion_cases: int,
) -> dict[str, Any]:
    if block_index == 0:
        teacher_count, student_count = 8, 0
    else:
        teacher_count, student_count = 4, 4
    teacher_indices = torch.randperm(DENSE_BANK_CASES, generator=generator)[:teacher_count]
    student_indices = torch.randperm(DENSE_BANK_CASES, generator=generator)[:student_count]
    motion_indices = torch.randperm(motion_cases, generator=generator)[:MOTION_SAMPLES_PER_UPDATE]
    maximum_start = EPISODE_STEPS - TRAIN_WINDOW_STEPS
    window_start = int(torch.randint(0, maximum_start + 1, (1,), generator=generator))
    return {
        "block_index": block_index,
        "teacher_indices": teacher_indices.tolist(),
        "student_indices": student_indices.tolist(),
        "motion_indices": motion_indices.tolist(),
        "window_start": window_start,
    }


@torch.no_grad()
def dense_burn_in(
    controller: ConnectomeController,
    sample: dict[str, Any],
    *,
    device: torch.device,
) -> Tensor:
    batch = sample["marker"].shape[0]
    recurrent = controller.initial_state(batch, device=device, dtype=torch.float32)
    scene = _scene_from_dict(sample["scene"], device=device)
    for step in range(sample["window_start"]):
        state = _trajectory_state_at(sample, step, device)
        image = render_visual_hover_scene(state, sample["marker"][:, step].to(device), scene=scene)
        _, recurrent = controller(image, state.euler[:, :2], recurrent)
    return recurrent.detach()


def dense_window_loss(
    controller: ConnectomeController,
    sample: dict[str, Any],
    burn_in: Tensor,
    *,
    dense_scale: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    recurrent = burn_in
    recurrent_state_finite = bool(torch.isfinite(recurrent).all())
    actor_output_finite = True
    scene = _scene_from_dict(sample["scene"], device=device)
    predictions, targets, masks = [], [], []
    begin = sample["window_start"]
    for step in range(begin, begin + TRAIN_WINDOW_STEPS):
        state = _trajectory_state_at(sample, step, device)
        image = render_visual_hover_scene(state, sample["marker"][:, step].to(device), scene=scene)
        motor, recurrent = controller(image, state.euler[:, :2], recurrent)
        recurrent_state_finite &= bool(torch.isfinite(recurrent).all())
        actor_output_finite &= bool(torch.isfinite(motor).all())
        predictions.append(motor[:, 3])
        targets.append(sample["teacher_motor"][:, step, 3].to(device))
        masks.append(sample["eligible"][:, step].to(device))
    prediction = torch.stack(predictions, dim=1)
    target = torch.stack(targets, dim=1)
    mask = torch.stack(masks, dim=1)
    squared = ((prediction - target) / dense_scale).square()
    loss = (squared * mask).sum() / mask.sum().clamp_min(1)
    report = {
        "eligible_frames": int(mask.sum().detach()),
        "total_frames": mask.numel(),
        "motor_mae": float(
            ((prediction.detach() - target.detach()).abs() * mask).sum() / mask.sum().clamp_min(1)
        ),
        "maximum_motor_absolute": float(prediction.detach().abs().max()),
        "recurrent_state_finite": recurrent_state_finite,
        "actor_output_finite": actor_output_finite,
    }
    return loss, report


@torch.no_grad()
def dense_full_prefix_loss(
    controller: ConnectomeController,
    sample: dict[str, Any],
    *,
    dense_scale: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    burn_in = dense_burn_in(controller, sample, device=device)
    loss, report = dense_window_loss(
        controller, sample, burn_in, dense_scale=dense_scale, device=device
    )
    return float(loss), report


def _motion_scene_slice(
    bank: dict[str, Any], index: int, device: torch.device
) -> tuple[QuadState, VisualScene]:
    indices = torch.tensor([index], dtype=torch.long)
    return (
        _state_from_dict(bank["state"], device=device, indices=indices),
        _scene_from_dict(bank["scene"], device=device, indices=indices),
    )


def velocity_pair_contrast(branch_values: Tensor) -> Tensor:
    """Return (+vertical velocity)-(-vertical velocity) for one equal-image pair."""

    if branch_values.shape[0] != len(MOTION_VELOCITY_SIGNS):
        raise ValueError("motion examples must contain exactly one opposite-velocity pair")
    return branch_values[1] - branch_values[0]


@torch.no_grad()
def motion_burn_in(
    controller: ConnectomeController,
    bank: dict[str, Any],
    index: int,
    *,
    device: torch.device,
) -> Tensor:
    state, scene = _motion_scene_slice(bank, index, device)
    marker = state.position[:, 2]
    image = render_visual_hover_scene(state, marker, scene=scene)
    recurrent = controller.initial_state(1, device=device, dtype=torch.float32)
    for _ in range(MOTION_PREFIX_STEPS):
        _, recurrent = controller(image, state.euler[:, :2], recurrent)
    return recurrent.detach()


def motion_example(
    controller: ConnectomeController,
    bank: dict[str, Any],
    index: int,
    burn_in: Tensor,
    *,
    motion_scale: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    state, scene = _motion_scene_slice(bank, index, device)
    height = state.position[:, 2]
    height_error = bank["height_error"][index].to(device).reshape(1)
    height_sign = bank["height_sign"][index].to(device).reshape(1)
    speed = bank["speed"][index].to(device).reshape(1)
    horizon = int(bank["horizon"][index])
    approach = bank["approach_steps"][index].to(device).reshape(1)
    marker = height + height_sign * height_error
    recurrent = burn_in.expand(len(MOTION_VELOCITY_SIGNS), -1).clone()
    attitude = state.euler[:, :2].expand(len(MOTION_VELOCITY_SIGNS), -1)
    endpoint_images: list[Tensor] = []
    output = torch.zeros(len(MOTION_VELOCITY_SIGNS), 4, device=device)
    recurrent_state_finite = bool(torch.isfinite(recurrent).all())
    actor_output_finite = True
    for step in range(horizon):
        branch_states = []
        endpoint_images = []
        for velocity_sign in MOTION_VELOCITY_SIGNS:
            signed_speed = velocity_sign * speed
            offset = factorial.smooth_return_offset(
                signed_speed,
                step=step,
                total_steps=horizon,
                approach_steps=approach,
                policy_hz=POLICY_HZ,
            )
            branch_state = responsibility.state_at_height(state, height + offset)
            branch_states.append(branch_state)
            endpoint_images.append(render_visual_hover_scene(branch_state, marker, scene=scene))
        output, recurrent = controller(
            torch.cat(endpoint_images),
            attitude,
            recurrent,
        )
        recurrent_state_finite &= bool(torch.isfinite(recurrent).all())
        actor_output_finite &= bool(torch.isfinite(output).all())
    target_outputs = []
    config = HoverConfig()
    for velocity_sign in MOTION_VELOCITY_SIGNS:
        endpoint_state = responsibility.state_at_height(
            state, height, vertical_velocity=velocity_sign * speed
        )
        target_outputs.append(
            motor_target_for_rc(teacher_rc(endpoint_state, marker, config), config)[:, 3]
        )
    predicted_contrast = velocity_pair_contrast(output[:, 3]).reshape(())
    target_contrast = velocity_pair_contrast(torch.stack(target_outputs).reshape(-1)).reshape(())
    loss = ((predicted_contrast - target_contrast) / motion_scale).square()
    endpoint_difference = float(
        (endpoint_images[0].detach() - endpoint_images[1].detach()).abs().max()
    )
    return loss, {
        "prediction": predicted_contrast.detach(),
        "target": target_contrast.detach(),
        "horizon": horizon,
        "endpoint_image_difference_max": endpoint_difference,
        "recurrent_state_finite": recurrent_state_finite,
        "actor_output_finite": actor_output_finite,
    }


@torch.no_grad()
def motion_full_prefix_example(
    controller: ConnectomeController,
    bank: dict[str, Any],
    index: int,
    *,
    motion_scale: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    burn_in = motion_burn_in(controller, bank, index, device=device)
    loss, report = motion_example(
        controller, bank, index, burn_in, motion_scale=motion_scale, device=device
    )
    return float(loss), report


def combined_sample_objective(
    controller: ConnectomeController,
    dense_sample: dict[str, Any],
    dense_burn: Tensor,
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    motion_burns: list[Tensor],
    *,
    dense_scale: float,
    motion_scale: float,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    dense_loss, dense_report = dense_window_loss(
        controller, dense_sample, dense_burn, dense_scale=dense_scale, device=device
    )
    motion_losses, motion_reports = [], []
    for index, burn in zip(motion_indices, motion_burns, strict=True):
        loss, report = motion_example(
            controller, motion_bank, index, burn, motion_scale=motion_scale, device=device
        )
        motion_losses.append(loss)
        motion_reports.append(report)
    motion_loss = torch.stack(motion_losses).mean()
    objective = 0.5 * (dense_loss + motion_loss)
    return objective, {
        "objective": float(objective.detach()),
        "dense_loss": float(dense_loss.detach()),
        "motion_loss": float(motion_loss.detach()),
        "dense": dense_report,
        "motion_correct_sign_fraction": float(
            torch.stack(
                [(report["prediction"] * report["target"] > 0).float() for report in motion_reports]
            ).mean()
        ),
        "motion_endpoint_image_difference_max": max(
            report["endpoint_image_difference_max"] for report in motion_reports
        ),
        "all_recurrent_states_and_outputs_finite": bool(
            dense_report["recurrent_state_finite"]
            and dense_report["actor_output_finite"]
            and all(
                report["recurrent_state_finite"] and report["actor_output_finite"]
                for report in motion_reports
            )
        ),
    }


@torch.no_grad()
def combined_full_prefix_objective(
    controller: ConnectomeController,
    dense_sample: dict[str, Any],
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    *,
    dense_scale: float,
    motion_scale: float,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    dense_loss, dense_report = dense_full_prefix_loss(
        controller, dense_sample, dense_scale=dense_scale, device=device
    )
    losses, reports = [], []
    for index in motion_indices:
        loss, report = motion_full_prefix_example(
            controller, motion_bank, index, motion_scale=motion_scale, device=device
        )
        losses.append(loss)
        reports.append(report)
    motion_loss = float(np.mean(losses))
    objective = 0.5 * (dense_loss + motion_loss)
    return objective, {
        "objective": objective,
        "dense_loss": dense_loss,
        "motion_loss": motion_loss,
        "dense": dense_report,
        "motion_endpoint_image_difference_max": max(
            report["endpoint_image_difference_max"] for report in reports
        ),
        "all_recurrent_states_and_outputs_finite": bool(
            dense_report["recurrent_state_finite"]
            and dense_report["actor_output_finite"]
            and all(
                report["recurrent_state_finite"] and report["actor_output_finite"]
                for report in reports
            )
        ),
    }


@torch.no_grad()
def evaluate_motion_bank(
    controller: ConnectomeController,
    bank: dict[str, Any],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    predictions, targets, image_differences, motion_reports = [], [], [], []
    by_horizon: dict[str, list[tuple[Tensor, Tensor]]] = {}
    for index in range(bank["cases"]):
        _, report = motion_full_prefix_example(
            controller, bank, index, motion_scale=motion_scale, device=device
        )
        predictions.append(report["prediction"])
        targets.append(report["target"])
        image_differences.append(report["endpoint_image_difference_max"])
        motion_reports.append(report)
        by_horizon.setdefault(str(report["horizon"]), []).append(
            (report["prediction"], report["target"])
        )
    prediction = torch.stack(predictions).cpu()
    target = torch.stack(targets).cpu()
    target_power = target.square().sum().clamp_min(1.0e-12)
    by_horizon_report = {}
    for horizon, values in by_horizon.items():
        horizon_prediction = torch.stack([prediction for prediction, _ in values])
        horizon_target = torch.stack([target for _, target in values])
        horizon_target_power = horizon_target.square().sum().clamp_min(1.0e-12)
        by_horizon_report[horizon] = {
            "pairs": len(values),
            "correct_sign_fraction": float(
                ((horizon_prediction * horizon_target) > 0).float().mean()
            ),
            "teacher_aligned_gain": float(
                (horizon_prediction * horizon_target).sum() / horizon_target_power
            ),
            "nrmse": float(
                ((horizon_prediction - horizon_target) / motion_scale).square().mean().sqrt()
            ),
        }
    report = {
        "pairs": bank["cases"],
        "correct_sign_fraction": float(((prediction * target) > 0).float().mean()),
        "teacher_aligned_gain": float((prediction * target).sum() / target_power),
        "nrmse": float(((prediction - target) / motion_scale).square().mean().sqrt()),
        "prediction_rms_motor_units": float(prediction.square().mean().sqrt()),
        "target_rms_motor_units": float(target.square().mean().sqrt()),
        "endpoint_image_difference_max": max(image_differences),
        "by_horizon": by_horizon_report,
        "all_recurrent_states_and_outputs_finite": bool(
            all(
                item["recurrent_state_finite"] and item["actor_output_finite"]
                for item in motion_reports
            )
        ),
    }
    report["all_metrics_finite"] = bool(_all_finite_nested(report))
    return report


def teacher_motion_contrasts(bank: dict[str, Any], *, config: HoverConfig) -> Tensor:
    values = []
    device = torch.device("cpu")
    for index in range(bank["cases"]):
        state = _state_from_dict(
            bank["state"], device=device, indices=torch.tensor([index], dtype=torch.long)
        )
        height = state.position[:, 2]
        height_error = bank["height_error"][index].reshape(1)
        height_sign = bank["height_sign"][index].reshape(1)
        speed = bank["speed"][index].reshape(1)
        outputs = []
        marker = height + height_sign * height_error
        for velocity_sign in MOTION_VELOCITY_SIGNS:
            endpoint_state = responsibility.state_at_height(
                state, height, vertical_velocity=velocity_sign * speed
            )
            outputs.append(
                motor_target_for_rc(teacher_rc(endpoint_state, marker, config), config)[:, 3]
            )
        values.append(velocity_pair_contrast(torch.stack(outputs).reshape(-1)))
    return torch.stack(values)


def frozen_objective_scales(
    teacher_bank: dict[str, Any], motion_bank: dict[str, Any], *, config: HoverConfig
) -> dict[str, float]:
    nominal_rc = torch.zeros(1, 4)
    nominal_rc[:, 3] = 1.0 / config.thrust_to_weight
    nominal_motor = float(motor_target_for_rc(nominal_rc, config)[0, 3])
    target = teacher_bank["teacher_motor"][..., 3]
    eligible = teacher_bank["eligible"]
    correction = target - nominal_motor
    dense_rms = float(((correction.square() * eligible).sum() / eligible.sum().clamp_min(1)).sqrt())
    motion_rms = float(teacher_motion_contrasts(motion_bank, config=config).square().mean().sqrt())
    return {
        "dense": max(dense_rms, DENOMINATOR_RMS_FLOOR),
        "motion": max(motion_rms, DENOMINATOR_RMS_FLOOR),
        "unfloored_dense": dense_rms,
        "unfloored_motion": motion_rms,
        "nominal_hover_motor_drive": nominal_motor,
    }


def _copy_parameters(controller: ConnectomeController) -> dict[str, Tensor]:
    return {name: getattr(controller, name).detach().clone() for name in PARAMETER_FAMILIES}


@torch.no_grad()
def _load_parameters(controller: ConnectomeController, values: dict[str, Tensor]) -> None:
    for name in PARAMETER_FAMILIES:
        getattr(controller, name).copy_(values[name])


def _parameter_bounds(controller: ConnectomeController) -> dict[str, Any]:
    finite = all(torch.isfinite(getattr(controller, name)).all() for name in PARAMETER_FAMILIES)
    edge = controller.edge_magnitude.detach()
    return {
        "pass": bool(finite and (edge >= 0).all() and (edge <= 8).all()),
        "all_parameters_finite": bool(finite),
        "edge_magnitude_minimum": float(edge.min()),
        "edge_magnitude_maximum": float(edge.max()),
    }


@torch.no_grad()
def _canonical_report(controller: ConnectomeController) -> dict[str, Any]:
    before = _copy_parameters(controller)
    controller.project_parameters()
    difference = max(
        float((getattr(controller, name) - before[name]).abs().max()) for name in PARAMETER_FAMILIES
    )
    return {
        "pass": difference <= 1.0e-7,
        "second_projection_maximum_parameter_change": difference,
        "limit": 1.0e-7,
    }


def optimizer_step_counters(state: dict[str, Any]) -> list[float]:
    counters = []
    for parameter_state in state.get("state", {}).values():
        if "step" in parameter_state:
            value = parameter_state["step"]
            counters.append(float(value.item() if torch.is_tensor(value) else value))
    return sorted(counters)


def _all_finite_nested(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, dict):
        return all(_all_finite_nested(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_nested(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _relative_error(measured: float, predicted: float) -> float:
    return abs(measured - predicted) / max(abs(measured), abs(predicted), 1.0e-12)


def trial_reports_are_finite(
    fixed: dict[str, Any], full: dict[str, Any], full_value: float
) -> bool:
    return bool(
        fixed["all_recurrent_states_and_outputs_finite"]
        and full["all_recurrent_states_and_outputs_finite"]
        and _all_finite_nested(fixed)
        and _all_finite_nested(full)
        and math.isfinite(full_value)
    )


def _build_fixed_burns(
    controller: ConnectomeController,
    dense_sample: dict[str, Any],
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    *,
    device: torch.device,
) -> tuple[Tensor, list[Tensor]]:
    dense = dense_burn_in(controller, dense_sample, device=device)
    motion = [
        motion_burn_in(controller, motion_bank, index, device=device) for index in motion_indices
    ]
    return dense, motion


def _sample_objective_report(
    controller: ConnectomeController,
    dense_sample: dict[str, Any],
    dense_burn: Tensor,
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    motion_burns: list[Tensor],
    scales: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        _, report = combined_sample_objective(
            controller,
            dense_sample,
            dense_burn,
            motion_bank,
            motion_indices,
            motion_burns,
            dense_scale=scales["dense"],
            motion_scale=scales["motion"],
            device=device,
        )
    return report


def clone_raw_gradients_and_clip(
    controller: ConnectomeController, *, maximum_norm: float = GRADIENT_NORM_CAP
) -> tuple[dict[str, Tensor], Tensor]:
    """Preserve objective derivatives, then clip only the gradients consumed by Adam."""

    raw_gradients = {
        name: (
            getattr(controller, name).grad.detach().clone()
            if getattr(controller, name).grad is not None
            else torch.full_like(getattr(controller, name), float("nan"))
        )
        for name in PARAMETER_FAMILIES
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [getattr(controller, name) for name in PARAMETER_FAMILIES], maximum_norm
    )
    return raw_gradients, gradient_norm


def accumulate_sample_gradient(
    controller: ConnectomeController,
    dense_sample: dict[str, Any],
    dense_burn: Tensor,
    motion_bank: dict[str, Any],
    motion_indices: list[int],
    motion_burns: list[Tensor],
    scales: dict[str, float],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    for parameter in controller.parameters():
        parameter.grad = None
    dense_loss, dense_report = dense_window_loss(
        controller,
        dense_sample,
        dense_burn,
        dense_scale=scales["dense"],
        device=device,
    )
    (0.5 * dense_loss).backward()
    motion_values, motion_reports = [], []
    for index, burn in zip(motion_indices, motion_burns, strict=True):
        loss, report = motion_example(
            controller,
            motion_bank,
            index,
            burn,
            motion_scale=scales["motion"],
            device=device,
        )
        (0.5 * loss / len(motion_indices)).backward()
        motion_values.append(float(loss.detach()))
        motion_reports.append(report)
    objective = 0.5 * (float(dense_loss.detach()) + float(np.mean(motion_values)))
    gradients_finite = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in (getattr(controller, name) for name in PARAMETER_FAMILIES)
    )
    raw_gradients, gradient_norm = clone_raw_gradients_and_clip(controller)
    report = {
        "objective": objective,
        "dense_loss": float(dense_loss.detach()),
        "motion_loss": float(np.mean(motion_values)),
        "dense": dense_report,
        "motion_correct_sign_fraction": float(
            torch.stack(
                [(report["prediction"] * report["target"] > 0).float() for report in motion_reports]
            ).mean()
        ),
        "motion_endpoint_image_difference_max": max(
            report["endpoint_image_difference_max"] for report in motion_reports
        ),
        "all_recurrent_states_and_outputs_finite": bool(
            dense_report["recurrent_state_finite"]
            and dense_report["actor_output_finite"]
            and all(
                item["recurrent_state_finite"] and item["actor_output_finite"]
                for item in motion_reports
            )
        ),
        "gradients_finite": bool(gradients_finite),
        "gradient_norm_before_clipping": float(gradient_norm),
    }
    return report, raw_gradients


def proposal_from_one_adam_transaction(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    current_parameters: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Any], dict[str, Tensor]]:
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    controller.project_parameters()
    raw_parameters = _copy_parameters(controller)
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    displacement = {
        name: raw_parameters[name] - current_parameters[name] for name in PARAMETER_FAMILIES
    }
    _load_parameters(controller, current_parameters)
    optimizer.load_state_dict(optimizer_before)
    return raw_parameters, optimizer_after, displacement


@torch.no_grad()
def materialize_scaled_proposal(
    controller: ConnectomeController,
    current: dict[str, Tensor],
    displacement: dict[str, Tensor],
    scale: float,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    for name in PARAMETER_FAMILIES:
        getattr(controller, name).copy_(current[name] + scale * displacement[name])
    controller.project_parameters()
    bounds = _parameter_bounds(controller)
    canonical = _canonical_report(controller)
    effective = {
        name: getattr(controller, name).detach().clone() - current[name]
        for name in PARAMETER_FAMILIES
    }
    return {
        "bounds": bounds,
        "canonical": canonical,
        "pass": bounds["pass"] and canonical["pass"],
    }, effective


def run_update_attempt(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    teacher_bank: dict[str, Any],
    student_bank: dict[str, Any] | None,
    motion_bank: dict[str, Any],
    spec: dict[str, Any],
    scales: dict[str, float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    current = _copy_parameters(controller)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    dense_sample = materialize_dense_sample(teacher_bank, student_bank, spec)
    motion_indices = [int(value) for value in spec["motion_indices"]]
    dense_burn, motion_burns = _build_fixed_burns(
        controller, dense_sample, motion_bank, motion_indices, device=device
    )
    current_fixed = _sample_objective_report(
        controller,
        dense_sample,
        dense_burn,
        motion_bank,
        motion_indices,
        motion_burns,
        scales,
        device=device,
    )
    current_full_value, current_full = combined_full_prefix_objective(
        controller,
        dense_sample,
        motion_bank,
        motion_indices,
        dense_scale=scales["dense"],
        motion_scale=scales["motion"],
        device=device,
    )
    gradient, raw_gradients = accumulate_sample_gradient(
        controller,
        dense_sample,
        dense_burn,
        motion_bank,
        motion_indices,
        motion_burns,
        scales,
        device=device,
    )
    numerical_reasons = []
    if (
        not current_fixed["all_recurrent_states_and_outputs_finite"]
        or not current_full["all_recurrent_states_and_outputs_finite"]
        or not _all_finite_nested(current_fixed)
        or not _all_finite_nested(current_full)
        or not math.isfinite(current_full_value)
    ):
        numerical_reasons.append(
            "current objective, recurrent state, output or report was nonfinite"
        )
    if (
        not gradient["gradients_finite"]
        or not gradient["all_recurrent_states_and_outputs_finite"]
        or not _all_finite_nested(gradient)
    ):
        numerical_reasons.append("gradient, recurrent state, output or report was nonfinite")
    raw, optimizer_after, displacement = proposal_from_one_adam_transaction(
        controller, optimizer, current
    )
    counter_before = optimizer_step_counters(optimizer_before)
    counter_after = optimizer_step_counters(optimizer_after)
    expected_before = [] if not counter_before else counter_before
    transaction_pass = bool(
        (not expected_before and counter_after == [1.0, 1.0, 1.0])
        or (
            len(counter_before) == len(counter_after) == 3
            and all(
                after == before + 1.0
                for before, after in zip(counter_before, counter_after, strict=True)
            )
        )
    )
    if not transaction_pass:
        numerical_reasons.append("Adam counters did not advance exactly once")
    proposal_controls, effective = materialize_scaled_proposal(
        controller, current, displacement, 1.0
    )
    if not proposal_controls["pass"]:
        numerical_reasons.append("raw proposal failed bounds or canonical control")
    authoritative_directional = float(
        sum(
            (raw_gradients[name].double() * effective[name].double()).sum()
            for name in PARAMETER_FAMILIES
        )
    )
    if not math.isfinite(authoritative_directional) or authoritative_directional >= 0.0:
        numerical_reasons.append("authoritative proposal was not a finite descent direction")

    finite_difference: dict[str, Any] = {
        "pass": False,
        "scale": FINITE_DIFFERENCE_SCALE,
        "not_run_due_to_prior_numerical_failure": bool(numerical_reasons),
    }
    if not numerical_reasons:
        fd_controls, fd_effective = materialize_scaled_proposal(
            controller, current, displacement, FINITE_DIFFERENCE_SCALE
        )
        fd_candidate = _sample_objective_report(
            controller,
            dense_sample,
            dense_burn,
            motion_bank,
            motion_indices,
            motion_burns,
            scales,
            device=device,
        )
        measured_direction = (
            fd_candidate["objective"] - current_fixed["objective"]
        ) / FINITE_DIFFERENCE_SCALE
        fd_directional = float(
            sum(
                (
                    raw_gradients[name].double()
                    * (fd_effective[name].double() / FINITE_DIFFERENCE_SCALE)
                ).sum()
                for name in PARAMETER_FAMILIES
            )
        )
        fd_error = _relative_error(measured_direction, fd_directional)
        fd_pass = bool(
            fd_controls["pass"]
            and fd_candidate["all_recurrent_states_and_outputs_finite"]
            and _all_finite_nested(fd_candidate)
            and math.isfinite(fd_directional)
            and fd_directional < -1.0e-8
            and math.isfinite(measured_direction)
            and measured_direction < -1.0e-8
            and fd_error <= FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        )
        if not fd_pass:
            numerical_reasons.append("fixed-burn-in directional finite difference failed")
        finite_difference = {
            "pass": fd_pass,
            "scale": FINITE_DIFFERENCE_SCALE,
            "autograd_directional_derivative": fd_directional,
            "authoritative_full_proposal_directional_derivative": authoritative_directional,
            "measured_directional_derivative": measured_direction,
            "relative_error": fd_error,
            "relative_error_limit": FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
            "uses_exact_frozen_sample_and_burn_in": True,
            "controls": fd_controls,
            "candidate": fd_candidate,
            "not_run_due_to_prior_numerical_failure": False,
        }
    _load_parameters(controller, current)

    trials = []
    selected: dict[str, Any] | None = None
    if not numerical_reasons:
        for scale in BACKTRACK_SCALES:
            controls, _ = materialize_scaled_proposal(controller, current, displacement, scale)
            if not controls["pass"]:
                numerical_reasons.append(
                    f"ordinary scale {scale:g} failed bounds or canonical control"
                )
                trials.append(
                    {
                        "scale": scale,
                        "pass": False,
                        "fixed_burn_in": None,
                        "full_prefix": None,
                        "fixed_burn_in_improvement": None,
                        "full_prefix_improvement": None,
                        "controls": controls,
                        "fatal_numerical_failure": True,
                    }
                )
                _load_parameters(controller, current)
                break
            fixed = _sample_objective_report(
                controller,
                dense_sample,
                dense_burn,
                motion_bank,
                motion_indices,
                motion_burns,
                scales,
                device=device,
            )
            full_value, full = combined_full_prefix_objective(
                controller,
                dense_sample,
                motion_bank,
                motion_indices,
                dense_scale=scales["dense"],
                motion_scale=scales["motion"],
                device=device,
            )
            finite_trial = trial_reports_are_finite(fixed, full, full_value)
            fixed_improvement = current_fixed["objective"] - fixed["objective"]
            full_improvement = current_full_value - full_value
            passed = bool(
                finite_trial
                and fixed_improvement >= MINIMUM_OBJECTIVE_IMPROVEMENT
                and full_improvement >= MINIMUM_OBJECTIVE_IMPROVEMENT
            )
            trial = {
                "scale": scale,
                "pass": passed,
                "fixed_burn_in": fixed,
                "full_prefix": full,
                "fixed_burn_in_improvement": fixed_improvement,
                "full_prefix_improvement": full_improvement,
                "controls": controls,
                "fatal_numerical_failure": not finite_trial,
            }
            trials.append(trial)
            if not finite_trial:
                numerical_reasons.append(
                    f"ordinary scale {scale:g} produced a nonfinite recurrent state, "
                    "output or metric"
                )
                _load_parameters(controller, current)
                break
            if passed:
                selected = trial
                break
            _load_parameters(controller, current)
    if selected is None:
        _load_parameters(controller, current)
        optimizer.load_state_dict(optimizer_before)
    else:
        optimizer.load_state_dict(optimizer_after)
    return {
        "accepted": selected is not None,
        "fatal_numerical_failure": bool(numerical_reasons),
        "numerical_failure_reasons": numerical_reasons,
        "sample_spec": spec,
        "current_fixed_burn_in": current_fixed,
        "current_full_prefix": current_full,
        "gradient": gradient,
        "optimizer_transaction": {
            "pass": transaction_pass,
            "counters_before": counter_before,
            "counters_after": counter_after,
            "state_before_sha256": audit.semantic_sha256(optimizer_before),
            "state_after_sha256": audit.semantic_sha256(optimizer_after),
        },
        "raw_parameter_family_rms": {
            name: float((raw[name] - current[name]).square().mean().sqrt())
            for name in PARAMETER_FAMILIES
        },
        "directional_derivative": authoritative_directional,
        "finite_difference": finite_difference,
        "trials": trials,
        "accepted_scale": None if selected is None else selected["scale"],
        "optimizer_pending_state_exact": selected is not None,
    }


def case_support_decision(cases: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    initial = cases["initial_marker"]
    final = cases["final_marker"]
    split = cases["split_code"]
    step = cases["step_code"]
    camera_family = cases["camera_family"]
    marker_family = cases["marker_family"]

    def in_train_bands(value: Tensor) -> Tensor:
        return (
            (value >= TRAIN_ABSOLUTE_MARKER_BANDS[0][0])
            & (value <= TRAIN_ABSOLUTE_MARKER_BANDS[0][1])
        ) | (
            (value >= TRAIN_ABSOLUTE_MARKER_BANDS[1][0])
            & (value <= TRAIN_ABSOLUTE_MARKER_BANDS[1][1])
        )

    training = split == 0
    if bool(training.any()):
        if not bool((in_train_bands(initial[training]) & in_train_bands(final[training])).all()):
            reasons.append("a training marker segment left the training bands")
        if not bool((camera_family[training] == marker_family[training]).all()):
            reasons.append("a training marker/camera family combination was held out")
    held_marker = split == 1
    if bool(held_marker.any()):
        stationary = held_marker & (step == 0)
        stepped = held_marker & (step != 0)
        if bool(stationary.any()) and not bool(
            (
                (final[stationary] >= HELD_OUT_ABSOLUTE_MARKER_BAND[0])
                & (final[stationary] <= HELD_OUT_ABSOLUTE_MARKER_BAND[1])
            ).all()
        ):
            reasons.append("a stationary held-out marker was outside its band")
        if bool(stepped.any()) and not bool(
            (
                in_train_bands(initial[stepped])
                & (final[stepped] >= HELD_OUT_ABSOLUTE_MARKER_BAND[0])
                & (final[stepped] <= HELD_OUT_ABSOLUTE_MARKER_BAND[1])
            ).all()
        ):
            reasons.append("a held-out marker step did not cross from train to held-out support")
    combination = split == 2
    if bool(combination.any()):
        if not bool((camera_family[combination] != marker_family[combination]).all()):
            reasons.append("a held-out combination did not oppose marker and camera families")
        if not bool(
            (in_train_bands(initial[combination]) & in_train_bands(final[combination])).all()
        ):
            reasons.append("a held-out combination marker segment left the training bands")
    finite = all(
        torch.isfinite(cases[name]).all() for name in ("initial_marker", "final_marker", "impulse")
    )
    if not finite:
        reasons.append("case values were nonfinite")
    return {"pass": not reasons, "reasons": reasons}


@torch.no_grad()
def teacher_motion_stick_positive_control(
    bank: dict[str, Any], *, device: torch.device, config: HoverConfig
) -> dict[str, Any]:
    desired_rc, motor = [], []
    for index in range(bank["cases"]):
        state, _ = _motion_scene_slice(bank, index, device)
        height = state.position[:, 2]
        height_error = bank["height_error"][index].to(device).reshape(1)
        height_sign = bank["height_sign"][index].to(device).reshape(1)
        speed = bank["speed"][index].to(device).reshape(1)
        marker = height + height_sign * height_error
        for velocity_sign in MOTION_VELOCITY_SIGNS:
            endpoint_state = responsibility.state_at_height(
                state, height, vertical_velocity=velocity_sign * speed
            )
            rc = teacher_rc(endpoint_state, marker, config)
            desired_rc.append(rc)
            motor.append(motor_target_for_rc(rc, config))
    desired = torch.cat(desired_rc)
    command = torch.cat(motor)
    sticks = ForelegStickPlant(config).to(device)
    state = hover_train.settled_sticks(len(command), device, config)
    measured = torch.zeros_like(command)
    for _ in range(PHYSICS_HZ):
        measured, state = sticks(command, state)
    desired_throttle = desired[:, 3].reshape(bank["cases"], len(MOTION_VELOCITY_SIGNS))
    measured_throttle = measured[:, 3].reshape(bank["cases"], len(MOTION_VELOCITY_SIGNS))
    desired_contrast = desired_throttle[:, 1] - desired_throttle[:, 0]
    measured_contrast = measured_throttle[:, 1] - measured_throttle[:, 0]
    maximum_error = float((measured - desired).abs().max())
    sign = float(((desired_contrast * measured_contrast) > 0).float().mean())
    return {
        "pass": maximum_error <= 0.01 and sign == 1.0,
        "constant_command_steps_at_physics_rate": PHYSICS_HZ,
        "maximum_rc_error": maximum_error,
        "damping_contrast_correct_sign_fraction": sign,
    }


def positive_control_decision(report: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if not support["pass"]:
        reasons.append("positive-control case support was invalid")
    if not report["all_metrics_finite"]:
        reasons.append("teacher positive-control metrics were nonfinite")
    if report["success_fraction"] < POSITIVE_CONTROL_SUCCESS_MINIMUM:
        reasons.append("teacher positive-control success was below 95%")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "minimum_success_fraction": POSITIVE_CONTROL_SUCCESS_MINIMUM,
        "measured_success_fraction": report["success_fraction"],
    }


def midpoint_decision(hover: dict[str, Any], motion: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if not hover["all_metrics_finite"]:
        reasons.append("midpoint hover metrics were nonfinite")
    if hover["success_fraction"] < MIDPOINT_SUCCESS_MINIMUM:
        reasons.append("midpoint assisted-hover success was below 50%")
    if not motion.get("all_metrics_finite", False) or not motion.get(
        "all_recurrent_states_and_outputs_finite", False
    ):
        reasons.append("midpoint motion metrics, recurrent state or outputs were nonfinite")
    if motion["correct_sign_fraction"] < MIDPOINT_MOTION_SIGN_MINIMUM:
        reasons.append("midpoint motion correct-sign fraction was below 50%")
    if motion["endpoint_image_difference_max"] != 0.0:
        reasons.append("midpoint paired-motion endpoint images were not exactly equal")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "minimum_success_fraction": MIDPOINT_SUCCESS_MINIMUM,
        "minimum_motion_correct_sign_fraction": MIDPOINT_MOTION_SIGN_MINIMUM,
    }


def final_decision(
    live: dict[str, Any],
    frozen: dict[str, Any],
    motion: dict[str, Any],
    bootstrap_interval: tuple[float, float],
) -> dict[str, Any]:
    reasons = []
    if not live["all_metrics_finite"] or not frozen["all_metrics_finite"]:
        reasons.append("final hover metrics were nonfinite")
    if live["success_fraction"] < FINAL_SUCCESS_MINIMUM:
        reasons.append("final assisted-hover success was below 90%")
    if not motion.get("all_metrics_finite", False) or not motion.get(
        "all_recurrent_states_and_outputs_finite", False
    ):
        reasons.append("final motion metrics, recurrent state or outputs were nonfinite")
    if motion["correct_sign_fraction"] < FINAL_MOTION_SIGN_MINIMUM:
        reasons.append("final motion correct-sign fraction was below 90%")
    low_gain, high_gain = FINAL_MOTION_GAIN_RANGE
    if not low_gain <= motion["teacher_aligned_gain"] <= high_gain:
        reasons.append("final motion teacher-aligned gain was outside 0.5-1.5")
    if motion["endpoint_image_difference_max"] != 0.0:
        reasons.append("final paired-motion endpoint images were not exactly equal")
    if not frozen["all_frozen_frames_captured"]:
        reasons.append("one or more target-aware frozen frames were not captured")
    if not all(math.isfinite(value) for value in bootstrap_interval):
        reasons.append("live-minus-frozen success bootstrap interval was nonfinite")
    elif bootstrap_interval[0] <= 0.0:
        reasons.append("live-minus-frozen success bootstrap lower bound was not positive")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "minimum_success_fraction": FINAL_SUCCESS_MINIMUM,
        "minimum_motion_correct_sign_fraction": FINAL_MOTION_SIGN_MINIMUM,
        "motion_gain_range": list(FINAL_MOTION_GAIN_RANGE),
        "live_minus_frozen_success_bootstrap_95_interval": list(bootstrap_interval),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _make_optimizer(controller: ConnectomeController) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": [controller.edge_magnitude, controller.bias],
                "lr": EDGE_BIAS_LEARNING_RATE,
            },
            {
                "params": [controller.raw_time_constant],
                "lr": TIME_CONSTANT_LEARNING_RATE,
            },
        ],
        betas=OPTIMIZER_BETAS,
        weight_decay=0.0,
    )


def _resume_payload(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    state: dict[str, Any],
) -> dict[str, Any]:
    controller_state = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    payload = {
        **state,
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "optimizer_betas": list(OPTIMIZER_BETAS),
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "controller": controller_state,
        "controller_parameter_sha256": audit.semantic_sha256(
            {name: controller_state[name] for name in PARAMETER_FAMILIES}
        ),
        "optimizer": optimizer_state,
        "optimizer_sha256": audit.semantic_sha256(optimizer_state),
        "optimizer_step_counters": optimizer_step_counters(optimizer_state),
        "optimizer_generator_state": generator.get_state(),
    }
    return payload


def _save_resume(
    path: Path,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    state: dict[str, Any],
) -> str:
    payload = _resume_payload(controller, optimizer, generator, state)
    _atomic_torch_save(payload, path)
    return responsibility.file_sha256(path)


def _payload_semantic_hash(payload: dict[str, Any]) -> str:
    return audit.semantic_sha256(
        {name: value for name, value in payload.items() if name != "sha256"}
    )


def training_block_identity(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(block["index"]),
        "teacher_sha256": block["teacher"]["sha256"],
        "student_sha256": (None if block["student"] is None else block["student"]["sha256"]),
        "motion_sha256": block["motion"]["sha256"],
    }


def _validate_resume_scientific_state(
    payload: dict[str, Any],
    *,
    teacher_history_seeds: tuple[int, ...] = TEACHER_HISTORY_SEEDS,
    student_history_seeds: tuple[int | None, ...] = STUDENT_HISTORY_SEEDS,
    motion_history_seeds: tuple[int, ...] = MOTION_HISTORY_SEEDS,
) -> None:
    if not (
        len(teacher_history_seeds) == len(student_history_seeds) == len(motion_history_seeds) == 4
    ):
        raise ValueError("resume seed registries must describe exactly four training blocks")
    accepted = int(payload.get("accepted_updates", -1))
    attempted = int(payload.get("attempted_updates", -1))
    if not (
        0 <= accepted <= MAX_ACCEPTED_UPDATES and accepted <= attempted <= MAX_ATTEMPTED_UPDATES
    ):
        raise SystemExit("resume accepted/attempted counters are invalid")
    pending_terminal = payload.get("pending_terminal")
    if pending_terminal is not None and (
        not isinstance(pending_terminal, dict)
        or set(pending_terminal) != {"classification", "stop_reason", "passed"}
        or not isinstance(pending_terminal["passed"], bool)
    ):
        raise SystemExit("resume pending terminal transition is invalid")
    if payload.get("run_state") == "stopped" and pending_terminal is None:
        raise SystemExit("stopped resume is missing its terminal transition")

    midpoint = payload.get("midpoint")
    if (
        accepted >= MIDPOINT_UPDATE
        and midpoint is None
        and not payload.get("development_started", False)
    ):
        raise SystemExit("resume could skip the mandatory midpoint evaluation")
    if accepted > MIDPOINT_UPDATE and (
        midpoint is None
        or not payload.get("development_completed", False)
        or not midpoint.get("decision", {}).get("pass", False)
    ):
        raise SystemExit("resume advanced beyond an unpassed midpoint gate")
    if (
        accepted == FINAL_UPDATE
        and payload.get("final") is None
        and not payload.get("final_started", False)
    ):
        raise SystemExit("resume could skip the mandatory final evaluation")

    actual_optimizer_counters = optimizer_step_counters(payload["optimizer"])
    if actual_optimizer_counters != payload.get("optimizer_step_counters"):
        raise SystemExit("resume stored Adam counters do not match optimizer state")
    expected_optimizer_counters = [] if accepted == 0 else [float(accepted)] * 3
    if actual_optimizer_counters != expected_optimizer_counters:
        raise SystemExit("resume Adam counters do not match accepted updates")
    if any(
        tuple(group.get("betas", ())) != OPTIMIZER_BETAS
        for group in payload["optimizer"]["param_groups"]
    ):
        raise SystemExit("resume Adam betas do not match the registered run")

    block = payload.get("block")
    scales = payload.get("objective_scales")
    if block is None or scales is None:
        if payload.get("run_state") == "active":
            raise SystemExit("active resume is missing its frozen training bank or scales")
        return
    block_index = int(block.get("index", -1))
    current_index = min(accepted // 25, len(teacher_history_seeds) - 1)
    previous_index = min(max(accepted - 1, 0) // 25, len(teacher_history_seeds) - 1)
    if block_index not in {current_index, previous_index}:
        raise SystemExit("resume training block does not match accepted-update phase")
    if block.get("identity") != training_block_identity(block):
        raise SystemExit("resume training-block identity mismatch")

    teacher_bank = block["teacher"]
    student_bank = block["student"]
    motion_bank = block["motion"]
    if teacher_bank.get("case_seed") != teacher_history_seeds[block_index]:
        raise SystemExit("resume teacher-history seed mismatch")
    if _payload_semantic_hash(teacher_bank) != teacher_bank.get("sha256"):
        raise SystemExit("resume teacher-history bank semantic hash mismatch")
    expected_student_seed = student_history_seeds[block_index]
    if expected_student_seed is None:
        if student_bank is not None:
            raise SystemExit("resume unexpectedly contains a block-one student bank")
    else:
        if student_bank is None or student_bank.get("case_seed") != expected_student_seed:
            raise SystemExit("resume student-history seed mismatch")
        if _payload_semantic_hash(student_bank) != student_bank.get("sha256"):
            raise SystemExit("resume student-history bank semantic hash mismatch")
    if motion_bank.get("seed") != motion_history_seeds[block_index]:
        raise SystemExit("resume motion-history seed mismatch")
    if _payload_semantic_hash(motion_bank) != motion_bank.get("sha256"):
        raise SystemExit("resume motion bank semantic hash mismatch")

    expected_reports = {
        "teacher": trajectory_bank_report(teacher_bank),
        "student": None if student_bank is None else trajectory_bank_report(student_bank),
        "motion": motion_bank_report(motion_bank),
    }
    if audit.semantic_sha256(block.get("reports")) != audit.semantic_sha256(expected_reports):
        raise SystemExit("resume training-block reports do not match frozen banks")
    reports = payload.get("training_block_reports", [])
    if len(reports) <= block_index or audit.semantic_sha256(
        reports[block_index]
    ) != audit.semantic_sha256(expected_reports):
        raise SystemExit("resume training-block report history mismatch")

    if not _all_finite_nested(scales) or scales["dense"] <= 0.0 or scales["motion"] <= 0.0:
        raise SystemExit("resume objective scales are invalid")
    if audit.semantic_sha256(scales) != payload.get("objective_scales_sha256"):
        raise SystemExit("resume frozen objective-scale identity mismatch")


def _load_resume(
    path: Path,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    teacher_history_seeds: tuple[int, ...] = TEACHER_HISTORY_SEEDS,
    student_history_seeds: tuple[int | None, ...] = STUDENT_HISTORY_SEEDS,
    motion_history_seeds: tuple[int, ...] = MOTION_HISTORY_SEEDS,
) -> dict[str, Any]:
    # Identity reports use exact CPU reductions; loading banks onto CUDA here could
    # introduce backend-dependent rounding before their hashes are checked.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "optimizer_betas": list(OPTIMIZER_BETAS),
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise SystemExit(f"resume {name} does not match the registered run")
    parameters = {name: payload["controller"][name] for name in PARAMETER_FAMILIES}
    if audit.semantic_sha256(parameters) != payload["controller_parameter_sha256"]:
        raise SystemExit("resume controller semantic hash mismatch")
    if audit.semantic_sha256(payload["optimizer"]) != payload["optimizer_sha256"]:
        raise SystemExit("resume optimizer semantic hash mismatch")
    _validate_resume_scientific_state(
        payload,
        teacher_history_seeds=teacher_history_seeds,
        student_history_seeds=student_history_seeds,
        motion_history_seeds=motion_history_seeds,
    )
    controller.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["optimizer"])
    generator.set_state(payload["optimizer_generator_state"])
    return {
        name: value
        for name, value in payload.items()
        if name
        not in {
            "experiment",
            "protocol_commit",
            "optimizer_betas",
            "graph_sha256",
            "checkpoint_sha256",
            "controller",
            "controller_parameter_sha256",
            "optimizer",
            "optimizer_sha256",
            "optimizer_generator_state",
        }
    }


def _new_training_block(
    controller: ConnectomeController,
    *,
    block_index: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    teacher_cases = build_hover_case_bank(
        seed=TEACHER_HISTORY_SEEDS[block_index],
        counts={"train": DENSE_BANK_CASES},
        device=device,
        config=config,
    )
    support = case_support_decision(teacher_cases)
    if not support["pass"]:
        raise RuntimeError(f"teacher training case support failed: {support['reasons']}")
    teacher_bank = collect_trajectory_bank(
        controller,
        teacher_cases,
        history_type="teacher",
        device=device,
        config=config,
    )
    student_bank = None
    student_seed = STUDENT_HISTORY_SEEDS[block_index]
    if student_seed is not None:
        student_cases = build_hover_case_bank(
            seed=student_seed,
            counts={"train": DENSE_BANK_CASES},
            device=device,
            config=config,
        )
        support = case_support_decision(student_cases)
        if not support["pass"]:
            raise RuntimeError(f"student training case support failed: {support['reasons']}")
        student_bank = collect_trajectory_bank(
            controller,
            student_cases,
            history_type="student",
            device=device,
            config=config,
        )
    motion_bank = build_motion_bank(
        seed=MOTION_HISTORY_SEEDS[block_index],
        cases=MOTION_BANK_CASES,
        height_amplitudes=TRAIN_MOTION_HEIGHT_AMPLITUDES,
        speed_amplitudes=TRAIN_MOTION_SPEED_AMPLITUDES,
        held_out_styles=False,
        device=device,
        config=config,
    )
    block = {
        "index": block_index,
        "teacher": teacher_bank,
        "student": student_bank,
        "motion": motion_bank,
        "reports": {
            "teacher": trajectory_bank_report(teacher_bank),
            "student": None if student_bank is None else trajectory_bank_report(student_bank),
            "motion": motion_bank_report(motion_bank),
        },
    }
    block["identity"] = training_block_identity(block)
    return block


@torch.no_grad()
def rpy_drift_on_training_bank(
    source: ConnectomeController,
    candidate: ConnectomeController,
    bank: dict[str, Any],
    *,
    device: torch.device,
    cases: int = 4,
) -> dict[str, float]:
    indices = torch.arange(min(cases, bank["marker"].shape[0]))
    sample = {
        "states": _index_nested_tensors(bank["states"], indices),
        "marker": bank["marker"][indices],
        "scene": _index_nested_tensors(bank["scene"], indices),
    }
    batch = len(indices)
    source_state = source.initial_state(batch, device=device, dtype=torch.float32)
    candidate_state = candidate.initial_state(batch, device=device, dtype=torch.float32)
    scene = _scene_from_dict(sample["scene"], device=device)
    errors = []
    scales = torch.tensor(joint.RPY_SCALES, device=device)
    for step in range(EPISODE_STEPS):
        state = _trajectory_state_at(sample, step, device)
        image = render_visual_hover_scene(state, sample["marker"][:, step].to(device), scene=scene)
        source_motor, source_state = source(image, state.euler[:, :2], source_state)
        candidate_motor, candidate_state = candidate(image, state.euler[:, :2], candidate_state)
        errors.append((candidate_motor[:, :3] - source_motor[:, :3]) / scales)
    nrmse = torch.stack(errors).square().mean(dim=(0, 1)).sqrt()
    return {name: float(nrmse[index]) for index, name in enumerate(("roll", "pitch", "yaw"))}


def _terminal_report(
    *,
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    classification: str,
    stop_reason: str,
    passed: bool,
    resume_path: Path | None,
    started: float,
) -> dict[str, Any]:
    source_parameters = _copy_parameters(source)
    current_parameters = _copy_parameters(controller)
    report = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "status": "nonpromotional_teacher_attitude_assisted_native_throttle",
        "classification": classification,
        "pass": passed,
        "stop_reason": stop_reason,
        "accepted_updates": state.get("accepted_updates", 0),
        "attempted_updates": state.get("attempted_updates", 0),
        "consecutive_ordinary_rejections": state.get("consecutive_rejections", 0),
        "source": {
            "graph": "data/derived/full-visual-connectome-v1.npz",
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "checkpoint": "runs/visual-hover/paired-dynamic-001/controller.pt",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "preflight": state.get("preflight"),
        "objective_scales": state.get("objective_scales"),
        "training_block_reports": state.get("training_block_reports", []),
        "history": state.get("history", []),
        "midpoint": state.get("midpoint"),
        "final": state.get("final"),
        "development_started": state.get("development_started", False),
        "development_completed": state.get("development_completed", False),
        "final_started": state.get("final_started", False),
        "final_completed": state.get("final_completed", False),
        "terminal_transition": state.get("pending_terminal"),
        "last_training_rpy_source_nrmse": state.get("last_training_rpy_source_nrmse"),
        "final_parameter_family_rms_from_source": {
            name: float((current_parameters[name] - source_parameters[name]).square().mean().sqrt())
            for name in PARAMETER_FAMILIES
        },
        "optimizer_step_counters": state.get("optimizer_step_counters", []),
        "run_state": "stopped",
        "resume_state_sha256": (
            responsibility.file_sha256(resume_path)
            if resume_path is not None and resume_path.is_file()
            else None
        ),
        "actor_contract_unchanged": True,
        "teacher_attitude_used_by_physical_training_and_evaluation_plant": True,
        "full_native_hover_authorized": False,
        "hover_controller_promoted": False,
        "gate_flight_authorized": False,
        "native_attitude_reintegration_authorized": passed,
        "interpretation_limit": (
            "A pass demonstrates assisted native vertical control only. It does not show "
            "that the fly can control attitude, take off unassisted, hover fully natively, "
            "or fly through a gate."
        ),
        "elapsed_seconds_this_process": perf_counter() - started,
    }
    return report


def _stop_and_write(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    classification: str,
    stop_reason: str,
    passed: bool,
    started: float,
) -> int:
    terminal = {
        "classification": classification,
        "stop_reason": stop_reason,
        "passed": passed,
    }
    existing_terminal = state.get("pending_terminal")
    if existing_terminal is not None and existing_terminal != terminal:
        raise RuntimeError("pending terminal transition does not match requested stop")
    state["pending_terminal"] = terminal
    state["run_state"] = "stopped"
    state["attempt_stage"] = "idle"
    state["pending_sample"] = None
    state["optimizer_step_counters"] = optimizer_step_counters(optimizer.state_dict())
    resume_path = args.output_dir / "resume.pt"
    _save_resume(resume_path, controller, optimizer, generator, state)
    report = _terminal_report(
        state=state,
        source=source,
        controller=controller,
        classification=classification,
        stop_reason=stop_reason,
        passed=passed,
        resume_path=resume_path,
        started=started,
    )
    _atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(args.output_dir / "report.json"),
                "pass": passed,
                "classification": classification,
                "accepted_updates": state.get("accepted_updates", 0),
                "attempted_updates": state.get("attempted_updates", 0),
                "native_attitude_reintegration_authorized": passed,
                "full_native_hover_authorized": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _write_assisted_checkpoint(
    args: argparse.Namespace, controller: ConnectomeController, state: dict[str, Any]
) -> None:
    checkpoint = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "controller": {
            name: value.detach().cpu() for name, value in controller.state_dict().items()
        },
        "accepted_updates": state["accepted_updates"],
        "claim": "teacher-attitude-assisted native-throttle only",
    }
    _atomic_torch_save(checkpoint, args.output_dir / "assisted-controller.pt")


def _resolve_pending_terminal(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    started: float,
) -> int:
    terminal = state.get("pending_terminal")
    if not isinstance(terminal, dict):
        raise RuntimeError("no pending terminal transition to resolve")
    if terminal.get("passed"):
        _write_assisted_checkpoint(args, controller, state)
    return _stop_and_write(
        args=args,
        state=state,
        source=source,
        controller=controller,
        optimizer=optimizer,
        generator=generator,
        classification=str(terminal["classification"]),
        stop_reason=str(terminal["stop_reason"]),
        passed=bool(terminal["passed"]),
        started=started,
    )


def record_attempt_outcome(
    state: dict[str, Any],
    result: dict[str, Any],
    *,
    attempted_number: int,
    target_update: int,
) -> None:
    """Record an attempt and every mandatory next transition before one atomic save."""

    state["attempted_updates"] = attempted_number
    result["attempt"] = attempted_number
    result["target_accepted_update"] = target_update
    if result["accepted"]:
        state["accepted_updates"] = target_update
        state["consecutive_rejections"] = 0
        result["accepted_update"] = target_update
    elif result["fatal_numerical_failure"]:
        result["accepted_update"] = None
    else:
        state["consecutive_rejections"] += 1
        result["accepted_update"] = None
    state["history"].append(result)
    state["attempt_stage"] = "idle"
    state["pending_sample"] = None

    if state["accepted_updates"] == MIDPOINT_UPDATE and state["midpoint"] is None:
        state["development_started"] = True
    if state["accepted_updates"] == FINAL_UPDATE and state["final"] is None:
        state["final_started"] = True
    if result["fatal_numerical_failure"]:
        state["pending_terminal"] = {
            "classification": "fatal_numerical_control_failure",
            "stop_reason": (
                "a mandatory gradient, transaction, direction or finite-difference control failed"
            ),
            "passed": False,
        }
    elif state["consecutive_rejections"] >= MAX_CONSECUTIVE_REJECTIONS:
        state["pending_terminal"] = {
            "classification": "consecutive_ordinary_rejections",
            "stop_reason": "five consecutive finite proposals had no admissible ordinary scale",
            "passed": False,
        }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    config = HoverConfig()
    if PHYSICS_HZ % POLICY_HZ:
        raise SystemExit("policy frequency must divide the physics frequency")
    started = perf_counter()
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded.get("graph_sha256") != EXPECTED_GRAPH_SHA256:
        raise SystemExit("checkpoint graph hash does not match the registered graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    controller.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    controller.eval()
    if source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("registered assisted-throttle source has an unexpected actor input")
    optimizer = _make_optimizer(controller)
    generator = torch.Generator(device="cpu").manual_seed(OPTIMIZER_SAMPLING_SEED)
    resume_path = args.output_dir / "resume.pt"

    if resume_path.is_file():
        state = _load_resume(resume_path, controller, optimizer, generator)
        print(
            json.dumps(
                {
                    "stage": "resume_loaded",
                    "accepted_updates": state["accepted_updates"],
                    "attempted_updates": state["attempted_updates"],
                    "attempt_stage": state["attempt_stage"],
                }
            ),
            flush=True,
        )
        if state.get("pending_terminal") is not None:
            return _resolve_pending_terminal(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                started=started,
            )
        if state.get("run_state") != "active":
            raise SystemExit("assisted-throttle resume is not active")
        if state.get("attempt_stage") != "idle":
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="interrupted_attempt_closed",
                stop_reason="an update was interrupted after its sampled minibatch was persisted",
                passed=False,
                started=started,
            )
        if state.get("development_started") and not state.get("development_completed"):
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="interrupted_midpoint_evaluation_closed",
                stop_reason="the once-only midpoint evaluation was interrupted",
                passed=False,
                started=started,
            )
        if state.get("final_started") and not state.get("final_completed"):
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="interrupted_final_evaluation_closed",
                stop_reason="the once-only final evaluation was interrupted",
                passed=False,
                started=started,
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "run_state": "active",
            "accepted_updates": 0,
            "attempted_updates": 0,
            "consecutive_rejections": 0,
            "history": [],
            "block": None,
            "training_block_reports": [],
            "objective_scales": None,
            "preflight": None,
            "midpoint": None,
            "final": None,
            "development_started": False,
            "development_completed": False,
            "final_started": False,
            "final_completed": False,
            "attempt_stage": "idle",
            "pending_sample": None,
            "pending_terminal": None,
            "objective_scales_sha256": None,
        }
        print(json.dumps({"stage": "teacher_dynamic_positive_control"}), flush=True)
        positive_cases = build_hover_case_bank(
            seed=POSITIVE_CONTROL_SEED,
            counts={"train": 32, "held_out_marker": 16, "held_out_combination": 16},
            device=device,
            config=config,
        )
        positive_support = case_support_decision(positive_cases)
        positive_report, _ = evaluate_hover_cases(
            None,
            positive_cases,
            teacher_all_axes=True,
            frozen_vision=False,
            device=device,
            config=config,
        )
        positive_decision = positive_control_decision(positive_report, positive_support)
        state["preflight"] = {
            "actor_has_accelerometer": source.uses_accelerometer,
            "actor_has_proprioception": source.uses_proprioception,
            "positive_case_support": positive_support,
            "positive_case_manifest": hover_case_support_report(positive_cases),
            "teacher_dynamic_hover": positive_report,
            "teacher_dynamic_hover_decision": positive_decision,
        }
        if not positive_decision["pass"]:
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="teacher_positive_control_failed",
                stop_reason="the end-to-end analytical teacher did not pass its fixed hover gate",
                passed=False,
                started=started,
            )
        print(json.dumps({"stage": "building_training_block", "block": 1}), flush=True)
        state["block"] = _new_training_block(
            controller, block_index=0, device=device, config=config
        )
        state["training_block_reports"].append(state["block"]["reports"])
        state["objective_scales"] = frozen_objective_scales(
            state["block"]["teacher"], state["block"]["motion"], config=config
        )
        state["objective_scales_sha256"] = audit.semantic_sha256(state["objective_scales"])
        motion_stick = teacher_motion_stick_positive_control(
            state["block"]["motion"], device=device, config=config
        )
        source_motion = evaluate_motion_bank(
            source,
            state["block"]["motion"],
            motion_scale=state["objective_scales"]["motion"],
            device=device,
        )
        state["preflight"].update(
            {
                "teacher_motion_stick_positive_control": motion_stick,
                "source_training_motion": source_motion,
                "objective_scales": state["objective_scales"],
                "motion_endpoint_images_exact": (
                    source_motion["endpoint_image_difference_max"] == 0.0
                ),
            }
        )
        state["preflight"]["pass"] = bool(
            positive_decision["pass"]
            and motion_stick["pass"]
            and state["preflight"]["motion_endpoint_images_exact"]
            and source_motion["all_metrics_finite"]
            and source_motion["all_recurrent_states_and_outputs_finite"]
            and _all_finite_nested(state["objective_scales"])
        )
        if not state["preflight"]["pass"]:
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="motion_preflight_failed",
                stop_reason="paired-motion identity, scaling or foreleg sign control failed",
                passed=False,
                started=started,
            )
        _save_resume(resume_path, controller, optimizer, generator, state)

    while state["accepted_updates"] < MAX_ACCEPTED_UPDATES:
        if state["attempted_updates"] >= MAX_ATTEMPTED_UPDATES:
            return _stop_and_write(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                classification="attempt_budget_exhausted",
                stop_reason="the fixed 125-attempt budget was exhausted",
                passed=False,
                started=started,
            )
        required_block = state["accepted_updates"] // 25
        if state["block"]["index"] != required_block:
            print(
                json.dumps({"stage": "building_training_block", "block": required_block + 1}),
                flush=True,
            )
            state["block"] = _new_training_block(
                controller, block_index=required_block, device=device, config=config
            )
            state["training_block_reports"].append(state["block"]["reports"])
            _save_resume(resume_path, controller, optimizer, generator, state)

        spec = sample_update_spec(
            generator,
            block_index=state["block"]["index"],
            motion_cases=state["block"]["motion"]["cases"],
        )
        state["attempt_stage"] = "sampled"
        state["pending_sample"] = spec
        _save_resume(resume_path, controller, optimizer, generator, state)
        attempted_number = state["attempted_updates"] + 1
        target_update = state["accepted_updates"] + 1
        print(
            json.dumps(
                {
                    "stage": "training_attempt",
                    "attempt": attempted_number,
                    "target_accepted_update": target_update,
                    "block": state["block"]["index"] + 1,
                }
            ),
            flush=True,
        )
        result = run_update_attempt(
            controller,
            optimizer,
            state["block"]["teacher"],
            state["block"]["student"],
            state["block"]["motion"],
            spec,
            state["objective_scales"],
            device=device,
        )
        record_attempt_outcome(
            state,
            result,
            attempted_number=attempted_number,
            target_update=target_update,
        )
        state["optimizer_step_counters"] = optimizer_step_counters(optimizer.state_dict())
        _save_resume(resume_path, controller, optimizer, generator, state)
        print(
            json.dumps(
                {
                    "progress": "accepted_update" if result["accepted"] else "rejected_attempt",
                    "attempt": attempted_number,
                    "accepted_updates": state["accepted_updates"],
                    "scale": result["accepted_scale"],
                    "fatal_numerical_failure": result["fatal_numerical_failure"],
                    "fixed_objective": (
                        result["trials"][-1]["fixed_burn_in"]["objective"]
                        if result["accepted"]
                        else result["current_fixed_burn_in"]["objective"]
                    ),
                }
            ),
            flush=True,
        )
        if state.get("pending_terminal") is not None:
            return _resolve_pending_terminal(
                args=args,
                state=state,
                source=source,
                controller=controller,
                optimizer=optimizer,
                generator=generator,
                started=started,
            )

        if state["accepted_updates"] == MIDPOINT_UPDATE and state["midpoint"] is None:
            print(json.dumps({"stage": "midpoint_evaluation"}), flush=True)
            midpoint_cases = build_hover_case_bank(
                seed=MIDPOINT_HOVER_SEED,
                counts={"train": 32, "held_out_marker": 16, "held_out_combination": 16},
                device=device,
                config=config,
            )
            midpoint_support = case_support_decision(midpoint_cases)
            midpoint_hover, _ = evaluate_hover_cases(
                controller,
                midpoint_cases,
                teacher_all_axes=False,
                frozen_vision=False,
                device=device,
                config=config,
            )
            midpoint_motion_bank = build_motion_bank(
                seed=MIDPOINT_MOTION_SEED,
                cases=64,
                height_amplitudes=TRAIN_MOTION_HEIGHT_AMPLITUDES,
                speed_amplitudes=TRAIN_MOTION_SPEED_AMPLITUDES,
                held_out_styles=True,
                device=device,
                config=config,
            )
            midpoint_motion = evaluate_motion_bank(
                controller,
                midpoint_motion_bank,
                motion_scale=state["objective_scales"]["motion"],
                device=device,
            )
            midpoint = midpoint_decision(midpoint_hover, midpoint_motion)
            if not midpoint_support["pass"]:
                midpoint["pass"] = False
                midpoint["reasons"].append("midpoint hover case support was invalid")
            state["midpoint"] = {
                "decision": midpoint,
                "hover": midpoint_hover,
                "hover_case_manifest": hover_case_support_report(midpoint_cases),
                "hover_case_support": midpoint_support,
                "motion": midpoint_motion,
                "motion_bank_manifest": motion_bank_report(midpoint_motion_bank),
            }
            state["development_completed"] = True
            if not midpoint["pass"]:
                state["last_training_rpy_source_nrmse"] = rpy_drift_on_training_bank(
                    source, controller, state["block"]["teacher"], device=device
                )
                state["pending_terminal"] = {
                    "classification": "midpoint_gate_failed",
                    "stop_reason": "the fixed update-50 assisted-hover or motion gate failed",
                    "passed": False,
                }
            _save_resume(resume_path, controller, optimizer, generator, state)
            if state.get("pending_terminal") is not None:
                return _resolve_pending_terminal(
                    args=args,
                    state=state,
                    source=source,
                    controller=controller,
                    optimizer=optimizer,
                    generator=generator,
                    started=started,
                )

    if not state["final_started"]:
        state["final_started"] = True
        _save_resume(resume_path, controller, optimizer, generator, state)
    print(json.dumps({"stage": "final_evaluation"}), flush=True)
    final_cases = build_hover_case_bank(
        seed=FINAL_HOVER_SEED,
        counts={"held_out_marker": 64, "held_out_combination": 64},
        device=device,
        config=config,
    )
    final_support = case_support_decision(final_cases)
    live_report, live_success = evaluate_hover_cases(
        controller,
        final_cases,
        teacher_all_axes=False,
        frozen_vision=False,
        device=device,
        config=config,
    )
    frozen_report, frozen_success = evaluate_hover_cases(
        controller,
        final_cases,
        teacher_all_axes=False,
        frozen_vision=True,
        device=device,
        config=config,
    )
    final_motion_bank = build_motion_bank(
        seed=FINAL_MOTION_SEED,
        cases=128,
        height_amplitudes=FINAL_MOTION_HEIGHT_AMPLITUDES,
        speed_amplitudes=FINAL_MOTION_SPEED_AMPLITUDES,
        held_out_styles=True,
        device=device,
        config=config,
    )
    final_motion = evaluate_motion_bank(
        controller,
        final_motion_bank,
        motion_scale=state["objective_scales"]["motion"],
        device=device,
    )
    interval = paired_bootstrap_mean_interval(
        live_success.float() - frozen_success.float(), seed=BOOTSTRAP_SEED
    )
    final = final_decision(live_report, frozen_report, final_motion, interval)
    if not final_support["pass"]:
        final["pass"] = False
        final["reasons"].append("final hover case support was invalid")
    state["final"] = {
        "decision": final,
        "live_hover": live_report,
        "frozen_hover": frozen_report,
        "hover_case_manifest": hover_case_support_report(final_cases),
        "hover_case_support": final_support,
        "motion": final_motion,
        "motion_bank_manifest": motion_bank_report(final_motion_bank),
    }
    state["final_completed"] = True
    state["last_training_rpy_source_nrmse"] = rpy_drift_on_training_bank(
        source, controller, state["block"]["teacher"], device=device
    )
    state["pending_terminal"] = {
        "classification": (
            "assisted_native_throttle_passed" if final["pass"] else "final_gate_failed"
        ),
        "stop_reason": (
            "all assisted native-throttle gates passed"
            if final["pass"]
            else "the fixed update-100 assisted native-throttle gate failed"
        ),
        "passed": bool(final["pass"]),
    }
    _save_resume(resume_path, controller, optimizer, generator, state)
    return _resolve_pending_terminal(
        args=args,
        state=state,
        source=source,
        controller=controller,
        optimizer=optimizer,
        generator=generator,
        started=started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
