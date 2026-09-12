#!/usr/bin/env python3
"""CPU preflight of a privileged continuous-path teacher through the real stick plant."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import active_gate, sample_two_gate_cases  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
    current_gate_roll_motor,
)
from flydrone.gate import GateConfig, wrap_angle  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import (  # noqa: E402
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    rotation_matrix,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402


def gate_center_frustum(state, centers, camera):
    """Center-only geometric proxy: not annulus visibility, pixels or occlusion."""
    relative = centers - state.position
    body = torch.einsum("bji,bj->bi", rotation_matrix(state.euler), relative)
    half_width = math.tan(math.radians(camera.horizontal_fov_degrees) / 2)
    half_height = half_width * camera.height / camera.width
    inside = (
        (body[:, 0] > 0)
        & (body[:, 1].abs() <= body[:, 0] * half_width)
        & (body[:, 2].abs() <= body[:, 0] * half_height)
    )
    return inside, torch.linalg.vector_norm(relative, dim=1)


def physical_teacher_motor(state, path, config, teacher_config, gates, index, roll_teacher):
    """Privileged physical preflight: optional local roll applies from the first frame."""
    motor = course_teacher_motor(state, path, config, teacher_config)
    if roll_teacher == "current-gate":
        motor = motor.clone()
        motor[:, 0] = current_gate_roll_motor(
            state, active_gate(gates, index), config, active=index < len(gates)
        )
    elif roll_teacher != "curved":
        raise ValueError("unknown physical roll teacher")
    return motor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1_030_983)
    parser.add_argument("--speed", type=float, default=0.55)
    parser.add_argument("--position-gain", type=float, default=1.5)
    parser.add_argument("--velocity-gain", type=float, default=2.5)
    parser.add_argument("--attitude-gain", type=float, default=3.0)
    parser.add_argument("--roll-teacher", choices=("curved", "current-gate"), default="curved")
    parser.add_argument(
        "--heading-mode", choices=("tangent", "world-x", "rate-damped"), default="tangent"
    )
    parser.add_argument("--initial-yaw-offset-degrees", type=float, default=0.0)
    parser.add_argument("--initial-yaw-rate-degrees-per-second", type=float, default=0.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a physical preflight report")
    torch.set_num_threads(4)
    started = perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = HoverConfig(**payload["hover_config"])
    gate_config = GateConfig(**payload["gate_config"])
    camera = CameraSpec(
        width=payload["image_resolution"][0],
        height=payload["image_resolution"][1],
        horizontal_fov_degrees=payload["camera_hfov_degrees"],
    )
    teacher_config = CourseTeacherConfig(
        forward_speed=args.speed,
        position_gain=args.position_gain,
        velocity_gain=args.velocity_gain,
        attitude_gain=args.attitude_gain,
        heading_mode=args.heading_mode,
    )
    cases, gates = sample_two_gate_cases(
        args.pairs,
        seed=args.seed,
        device=torch.device("cpu"),
        hover_config=config,
        layout="variable",
        gate_count=5,
        spacing_range=(0.9, 1.5),
        lateral_step_range=(0.0, 0.2),
        lateral_deviation_limit=0.5,
        height_step_range=(0.0, 0.08),
        height_range=(0.9, 1.3),
        yaw_jitter_degrees=15.0,
    )
    path = CoursePath.through_gates(cases.state.position, gates)
    count = 2 * args.pairs
    index = torch.zeros(count, dtype=torch.long)
    path_failed = torch.zeros(count, dtype=torch.bool)
    previous = path.knots[:, 0]
    for fraction in torch.linspace(0, 1, 401)[1:]:
        x = path.knots[:, 0, 0] + fraction * (path.knots[:, -1, 0] - path.knots[:, 0, 0])
        point, _, _ = path.sample(x)
        events = classify_course_step(previous, point, gates, index, gate_config)
        path_failed |= events.failed
        index = events.next_gate_index
        previous = point
    path_clean = (index == 5) & ~path_failed

    state, sticks = cases.state, cases.sticks
    signs = torch.tensor([-1.0, 1.0]).repeat(args.pairs)
    state.euler[:, 2] += signs * math.radians(args.initial_yaw_offset_degrees)
    state.rates[:, 2] += signs * math.radians(args.initial_yaw_rate_degrees_per_second)
    initial_yaw = state.euler[:, 2].clone()
    quad, legs = DifferentiableQuad(config), ForelegStickPlant(config)
    index.zero_()
    passed = torch.zeros(count, 5, dtype=torch.bool)
    failed = torch.zeros(count, dtype=torch.bool)
    radial = torch.full((count, 5), float("nan"))
    times = torch.full_like(radial, float("nan"))
    collisions = torch.zeros(count, dtype=torch.bool)
    violations = torch.zeros_like(collisions)
    ground = torch.zeros_like(collisions)
    invalid = torch.zeros_like(collisions)
    first = torch.zeros_like(collisions)
    prefix = torch.zeros(count, dtype=torch.long)
    max_tilt = torch.zeros(count)
    saturation = torch.zeros(count)
    center_counts = torch.zeros(2, dtype=torch.long)
    eligible_counts = torch.zeros_like(center_counts)
    max_yaw_excursion = torch.zeros(count)
    max_absolute_yaw = torch.zeros(count)
    centers = torch.stack([g.center for g in gates], dim=1)
    rows = torch.arange(count)
    for step in range(round(args.seconds * 50)):
        active = ~failed & (index < 5)
        for role in range(2):
            target_index = (index + role).clamp(max=4)
            inside, distance = gate_center_frustum(state, centers[rows, target_index], camera)
            eligible = active & (index + role < 5) & (distance >= 0.5)
            center_counts[role] += (inside & eligible).sum()
            eligible_counts[role] += eligible.sum()
        excursion = wrap_angle(state.euler[:, 2] - initial_yaw).abs()
        max_yaw_excursion = torch.maximum(max_yaw_excursion, torch.where(active, excursion, 0))
        max_absolute_yaw = torch.maximum(
            max_absolute_yaw, torch.where(active, wrap_angle(state.euler[:, 2]).abs(), 0)
        )
        motor = physical_teacher_motor(
            state, path, config, teacher_config, gates, index, args.roll_teacher
        )
        for substep in range(2):
            rc, sticks = legs(motor, sticks)
            previous = state.position
            state = quad(rc, state, cases.mass_scale)
            events = classify_course_step(previous, state.position, gates, index, gate_config)
            index = events.next_gate_index
            new = events.passed & ~passed
            passed |= events.passed
            errors = torch.sqrt(
                events.crossing_lateral.square() + events.crossing_vertical.square()
            )
            radial[new] = errors[new]
            times[new] = (2 * step + substep + 1) / 100
            collisions |= events.ring_collision.any(dim=1)
            violations |= events.illegal_traversal.any(dim=1)
            ground |= state.position[:, 2] <= 0.03
            invalid |= ~hover_train.state_is_valid(state)
            failed |= events.failed | ground | invalid
            first |= events.passed[:, 0] & ~failed
            prefix += (events.passed & ~failed[:, None]).sum(1)
            max_tilt = torch.maximum(max_tilt, torch.linalg.vector_norm(state.euler[:, :2], dim=1))
            saturation += (rc[:, :3].abs().amax(dim=1) > 0.98) | (rc[:, 3] > 0.98)
    clean = passed.all(dim=1) & ~failed
    result = dict(
        experiment="current-gate-roll-from-start-physical-preflight-v1"
        if args.roll_teacher == "current-gate"
        else "continuous-hermite-course-teacher-preflight-v1",
        checkpoint=str(args.checkpoint),
        roll_teacher=args.roll_teacher,
        roll_teacher_applies_from_first_frame=True,
        other_axes="privileged curved teacher, not native fly outputs",
        cpu_only=True,
        actor_loaded=False,
        seed=args.seed,
        episodes=count,
        seconds=args.seconds,
        hover_config=vars(config),
        teacher_config=vars(teacher_config),
        initial_yaw_offset_degrees=args.initial_yaw_offset_degrees,
        initial_yaw_rate_degrees_per_second=args.initial_yaw_rate_degrees_per_second,
        path_geometry_clean_rate=float(path_clean.float().mean()),
        clean_course_success_rate=float(clean.float().mean()),
        clean_successes=int(clean.sum()),
        clean_successes_by_side=[
            int(clean[cases.side < 0].sum()),
            int(clean[cases.side > 0].sum()),
        ],
        clean_first_passes=int(first.sum()),
        clean_first_by_side=[int(first[cases.side < 0].sum()), int(first[cases.side > 0].sum())],
        clean_prefix_gates=int(prefix.sum()),
        gate_pass_counts=passed.sum(dim=0).tolist(),
        ring_collision_rate=float(collisions.float().mean()),
        illegal_traversal_rate=float(violations.float().mean()),
        ground_contact_rate=float(ground.float().mean()),
        invalid_rate=float(invalid.float().mean()),
        mass_scale_range=[float(cases.mass_scale.min()), float(cases.mass_scale.max())],
        full_tail_scored=True,
        clean_completion_time_mean_seconds=float(times[clean, -1].mean()) if clean.any() else None,
        pass_radial_mean_metres=float(radial[passed].mean()) if passed.any() else None,
        pass_radial_max_metres=float(radial[passed].max()) if passed.any() else None,
        maximum_tilt_degrees=float(torch.rad2deg(max_tilt).max()),
        saturation_fraction=float(saturation.mean()) / (100 * args.seconds),
        heading_diagnostics=dict(
            maximum_excursion_from_launch_degrees=float(torch.rad2deg(max_yaw_excursion).max()),
            maximum_absolute_heading_degrees=float(torch.rad2deg(max_absolute_yaw).max()),
            period="before failure or fifth-gate passage",
        ),
        gate_center_frustum_diagnostics=dict(
            description="center-only frustum proxy, not rendered annulus visibility or occlusion",
            minimum_center_distance_metres=0.5,
            period="before failure or fifth-gate passage",
            roles=["current", "next"],
            eligible_frame_counts=eligible_counts.tolist(),
            center_in_frustum_counts=center_counts.tolist(),
            center_in_frustum_fractions=[
                int(inside) / int(total) if total else None
                for inside, total in zip(center_counts, eligible_counts, strict=True)
            ],
            camera=vars(camera),
        ),
        teacher_is_privileged=True,
        fly_training_or_fly_success=False,
        elapsed_seconds=perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
