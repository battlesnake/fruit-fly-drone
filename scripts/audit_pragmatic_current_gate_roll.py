#!/usr/bin/env python3
"""Local roll-teacher diagnostic; never a native fly completion result."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_pragmatic_two_gate_zero_shot as evaluation  # noqa: E402
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.course_teacher import current_gate_roll_motor  # noqa: E402
from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--seed", type=int, default=1110983)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing audit")
    device = torch.device("cuda")
    controller, payload = load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = HoverConfig(**payload["hover_config"])
    gate_config = replace(GateConfig(**payload["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*payload["image_resolution"], payload["camera_hfov_degrees"])
    cases, gates = evaluation.sample_two_gate_cases(
        16, seed=args.seed, device=device, hover_config=config, **GEOMETRY
    )
    original_takeover = evaluation.diagnostic_axis_takeover
    captured = {}

    def capture_state(state, path, quad, teacher_config):
        # The evaluator constructs its ordinary path, but this target never uses it.
        captured["state"] = state
        return torch.zeros_like(state.actuator)

    def local_roll_takeover(native, unused_target, current, axes):
        target = torch.zeros_like(native)
        target[:, 0] = current_gate_roll_motor(
            captured["state"],
            evaluation.active_gate(gates, current),
            config,
            active=current < len(gates),
        )
        return original_takeover(native, target, current, axes)

    started = perf_counter()
    with (
        patch.object(evaluation, "course_teacher_motor", capture_state),
        patch.object(evaluation, "diagnostic_axis_takeover", local_roll_takeover),
    ):
        metrics = evaluation.evaluate(
            controller,
            cases,
            gates,
            seconds=30,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
            diagnostic_teacher_axes=(True, False, False, False),
        )
    conditional = {}
    for side in ("all", "negative", "positive"):
        suffix = "" if side == "all" else f"_{side}"
        first = metrics[f"clean_first_gate{suffix}_pass_rate"]
        clean = metrics[f"clean_course{suffix}_success_rate"]
        conditional[side] = clean / first if first else None
    result = dict(
        experiment="diagnostic-current-gate-velocity-damped-roll-v1",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        geometry=GEOMETRY,
        assisted_results_are_not_fly_success=True,
        teacher_inputs=["current gate relative position", "velocity", "roll", "roll rate"],
        absolute_heading_used_only_to_rotate_to_body_horizontal_frame=True,
        no_launch_line_or_path_used_by_target=True,
        native_axes=["pitch", "yaw", "throttle"],
        actor_inputs=["live RGB", "roll", "pitch"],
        continuous_neural_state=True,
        intervention="roll only after gate one, including full post-completion tail",
        clean_completion_given_clean_first=conditional,
        metrics=metrics,
        elapsed_seconds=perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                clean=metrics["clean_course_success_rate"],
                conditional=conditional,
                ground=metrics["ground_contact_rate"],
                invalid=metrics["invalid_rate"],
                elapsed_seconds=result["elapsed_seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
