#!/usr/bin/env python3
"""CPU preflight of a privileged continuous-path teacher through the real stick plant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import sample_two_gate_cases  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import GateConfig  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, HoverConfig  # noqa: E402


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
    parser.add_argument("--heading-mode", choices=("tangent", "world-x"), default="tangent")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    started = perf_counter()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = HoverConfig(**payload["hover_config"])
    gate_config = GateConfig(**payload["gate_config"])
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
    quad, legs = DifferentiableQuad(config), ForelegStickPlant(config)
    index.zero_()
    passed = torch.zeros(count, 5, dtype=torch.bool)
    failed = torch.zeros(count, dtype=torch.bool)
    radial = torch.full((count, 5), float("nan"))
    times = torch.full_like(radial, float("nan"))
    collisions = torch.zeros(count, dtype=torch.bool)
    violations = torch.zeros_like(collisions)
    ground = torch.zeros_like(collisions)
    max_tilt = torch.zeros(count)
    saturation = torch.zeros(count)
    for step in range(round(args.seconds * 50)):
        motor = course_teacher_motor(state, path, config, teacher_config)
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
            failed |= events.failed | ~hover_train.state_is_valid(state)
            max_tilt = torch.maximum(max_tilt, torch.linalg.vector_norm(state.euler[:, :2], dim=1))
            saturation += (rc[:, :3].abs().amax(dim=1) > 0.98) | (rc[:, 3] > 0.98)
    clean = passed.all(dim=1) & ~failed
    result = dict(
        experiment="continuous-hermite-course-teacher-preflight-v1",
        seed=args.seed,
        episodes=count,
        seconds=args.seconds,
        hover_config=vars(config),
        teacher_config=vars(teacher_config),
        path_geometry_clean_rate=float(path_clean.float().mean()),
        clean_course_success_rate=float(clean.float().mean()),
        clean_successes=int(clean.sum()),
        gate_pass_counts=passed.sum(dim=0).tolist(),
        ring_collision_rate=float(collisions.float().mean()),
        illegal_traversal_rate=float(violations.float().mean()),
        ground_contact_rate=float(ground.float().mean()),
        clean_completion_time_mean_seconds=float(times[clean, -1].mean()) if clean.any() else None,
        pass_radial_mean_metres=float(radial[passed].mean()) if passed.any() else None,
        pass_radial_max_metres=float(radial[passed].max()) if passed.any() else None,
        maximum_tilt_degrees=float(torch.rad2deg(max_tilt).max()),
        saturation_fraction=float(saturation.mean()) / (100 * args.seconds),
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
