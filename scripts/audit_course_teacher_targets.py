#!/usr/bin/env python3
"""Compare privileged teacher labels on identical, unmodified native fly flights."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_variable_height_hover as hover_train  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import sample_two_gate_cases  # noqa: E402
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import GateConfig, render_annular_gates_rgb  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def summarize_targets(counts, target_squared, error_squared, scales):
    result = []
    for phase in range(5):
        for side in range(2):
            count = int(counts[phase, side])
            if not count:
                continue
            target_rms = (target_squared[phase, side] / count).sqrt()
            error_rms = (error_squared[phase, side] / count).sqrt()
            result.append(
                dict(
                    gate=phase + 1,
                    side=("negative", "positive")[side],
                    frames=count,
                    target_rms=target_rms.tolist(),
                    source_error_rms=error_rms.tolist(),
                    normalized_target_rms=(target_rms / scales).tolist(),
                    normalized_source_error_rms=(error_rms / scales).tolist(),
                )
            )
    total = int(counts.sum())
    if total:
        target_rms = (target_squared.sum(dim=(0, 1)) / total).sqrt()
        error_rms = (error_squared.sum(dim=(0, 1)) / total).sqrt()
        overall = dict(
            frames=total,
            target_rms=target_rms.tolist(),
            source_error_rms=error_rms.tolist(),
            normalized_target_rms=(target_rms / scales).tolist(),
            normalized_source_error_rms=(error_rms / scales).tolist(),
        )
    else:
        overall = None
    return dict(overall=overall, by_gate_and_side=result)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=1160983)
    args = parser.parse_args()
    if args.pairs < 1 or args.seconds <= 0:
        raise SystemExit("pairs and seconds must be positive")
    started = perf_counter()
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    cases, gates = sample_two_gate_cases(
        args.pairs, seed=args.seed, device=device, hover_config=config, **GEOMETRY
    )
    state, sticks = cases.state, cases.sticks
    path = CoursePath.through_gates(state.position, gates)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    current = torch.zeros(2 * args.pairs, dtype=torch.long, device=device)
    failed = torch.zeros_like(current, dtype=torch.bool)
    neural = controller.initial_state(len(current), device=device, dtype=torch.float32)
    image = render_annular_gates_rgb(
        state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
    )
    for _ in range(10):
        _, neural = controller(image, state.euler[:, :2], neural)
    modes = ("world-x", "rate-damped")
    configs = [CourseTeacherConfig(heading_mode=mode) for mode in modes]
    counts = torch.zeros(5, 2, device=device)
    target_squared = torch.zeros(2, 5, 2, 4, device=device)
    error_squared = torch.zeros_like(target_squared)
    sides = torch.arange(len(current), device=device) % 2
    for _ in range(round(args.seconds * 50)):
        active = ~failed & (current < 5)
        if not bool(active.any()):
            break
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        prediction, neural = controller(image, state.euler[:, :2], neural)
        bucket = 2 * current[active] + sides[active]
        counts.view(-1).index_add_(0, bucket, torch.ones_like(bucket, dtype=counts.dtype))
        for mode_index, teacher in enumerate(configs):
            target = course_teacher_motor(state, path, config, teacher)
            target_squared[mode_index].view(-1, 4).index_add_(0, bucket, target[active].square())
            error_squared[mode_index].view(-1, 4).index_add_(
                0, bucket, (prediction[active] - target[active]).square()
            )
        # Only the unchanged fly acts; both teachers label those exact same states.
        for _ in range(2):
            rc, sticks = legs(prediction, sticks)
            previous = state.position
            state = quad(rc, state, cases.mass_scale)
            events = classify_course_step(previous, state.position, gates, current, gate_config)
            current = events.next_gate_index
            missed = (events.expected_forward_crossing & ~events.passed).any(dim=1)
            failed |= events.failed | missed | ~hover_train.state_is_valid(state)
    scales = torch.tensor([0.02, 0.02, 0.01, 0.025], device=device)
    result = dict(
        experiment="matched-native-history-teacher-target-audit-v1",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        episodes=2 * args.pairs,
        geometry=GEOMETRY,
        teacher_actions_applied=False,
        training_or_checkpoint_selection=False,
        period="native frames before lesson failure or fifth-gate passage",
        axes=["roll", "pitch", "yaw", "throttle"],
        unchanged_motor_normalization=scales.tolist(),
        modes={
            mode: summarize_targets(counts, target_squared[i], error_squared[i], scales)
            for i, mode in enumerate(modes)
        },
        elapsed_seconds=perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
