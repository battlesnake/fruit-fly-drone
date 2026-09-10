#!/usr/bin/env python3
"""Audit retinal and recurrent visual sensitivity of a visual-hover checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import ConnectomeController, DifferentiableQuad, HoverConfig  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402


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
        default=REPO_ROOT / "runs/visual-hover/pilot-001/controller.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--settling-steps", type=int, default=50)
    parser.add_argument("--motion-steps", type=int, default=25)
    return parser.parse_args()


def physical_state_at_height(quad: DifferentiableQuad, height: float, *, device: torch.device):
    position = torch.tensor(((0.0, 0.0, height),), device=device)
    return quad.initial_state(1, device=device, dtype=torch.float32, position=position)


@torch.no_grad()
def settled_motor(
    controller: ConnectomeController,
    image: torch.Tensor,
    roll_pitch: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    neural = controller.initial_state(1, device=image.device, dtype=image.dtype)
    motor = torch.zeros(1, 4, device=image.device)
    for _ in range(steps):
        motor, neural = controller(image, roll_pitch, neural)
    return motor[0], neural


@torch.no_grad()
def motion_history_motor(
    controller: ConnectomeController,
    quad: DifferentiableQuad,
    heights: torch.Tensor,
    *,
    target_height: torch.Tensor,
) -> torch.Tensor:
    neural = controller.initial_state(1, device=heights.device, dtype=torch.float32)
    motor = torch.zeros(1, 4, device=heights.device)
    for height in heights:
        state = physical_state_at_height(quad, float(height), device=heights.device)
        image = render_visual_hover_scene(state, target_height)
        motor, neural = controller(image, state.euler[:, :2], neural)
    return motor[0]


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    quad = DifferentiableQuad(HoverConfig()).to(device)
    target = torch.tensor([1.0], device=device)

    states = {
        "below_20cm": physical_state_at_height(quad, 0.8, device=device),
        "on_target": physical_state_at_height(quad, 1.0, device=device),
        "above_20cm": physical_state_at_height(quad, 1.2, device=device),
    }
    images = {name: render_visual_hover_scene(state, target) for name, state in states.items()}
    retinae = {name: controller.sample_retina(image) for name, image in images.items()}
    motors = {
        name: settled_motor(controller, image, states[name].euler[:, :2], args.settling_steps)[0]
        for name, image in images.items()
    }

    graph = np.load(args.graph)
    spectral = graph["visual_channel_weights"]
    receptor_group_masks = {
        "R1-R6": np.count_nonzero(spectral, axis=1) == 3,
        "R8p": spectral[:, 2] == 1.0,
        "R8y": spectral[:, 1] == 1.0,
    }
    retinal_changes = {}
    for comparison in ("below_20cm", "above_20cm"):
        camera_delta = images[comparison] - images["on_target"]
        retina_delta = retinae[comparison] - retinae["on_target"]
        retinal_changes[comparison] = {
            "camera_rms": float(torch.sqrt(camera_delta.square().mean())),
            "camera_max_absolute": float(camera_delta.abs().max()),
            "retina_rms": float(torch.sqrt(retina_delta.square().mean())),
            "retina_max_absolute": float(retina_delta.abs().max()),
            "retina_fraction_changed_gt_1e-3": float((retina_delta.abs() > 1.0e-3).float().mean()),
            "retina_rms_by_type": {
                name: float(
                    torch.sqrt(retina_delta[:, torch.from_numpy(mask).to(device)].square().mean())
                )
                for name, mask in receptor_group_masks.items()
            },
        }

    upward_heights = torch.linspace(0.8, 1.0, args.motion_steps, device=device)
    downward_heights = torch.linspace(1.2, 1.0, args.motion_steps, device=device)
    upward_motor = motion_history_motor(controller, quad, upward_heights, target_height=target)
    downward_motor = motion_history_motor(controller, quad, downward_heights, target_height=target)
    report = {
        "checkpoint": str(args.checkpoint),
        "policy_hz": args.policy_hz,
        "settling_steps": args.settling_steps,
        "retinal_changes_from_on_target": retinal_changes,
        "settled_motor_drive": {name: motor.tolist() for name, motor in motors.items()},
        "height_feedback_motor_difference_below_minus_above": (
            motors["below_20cm"] - motors["above_20cm"]
        ).tolist(),
        "opposite_motion_histories_same_final_image": {
            "upward_final_motor": upward_motor.tolist(),
            "downward_final_motor": downward_motor.tolist(),
            "upward_minus_downward": (upward_motor - downward_motor).tolist(),
            "throttle_difference": float(upward_motor[3] - downward_motor[3]),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
