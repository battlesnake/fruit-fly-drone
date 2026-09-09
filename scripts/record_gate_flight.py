#!/usr/bin/env python3
"""Record a successful connectome-to-forelegs annular-gate flight headlessly."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.gate import (  # noqa: E402
    GateConfig,
    classify_gate_crossing,
    gate_coordinates,
    render_annular_gate,
    sample_annular_gates,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=10_031)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=20)
    return parser.parse_args()


def draw_disk(canvas: np.ndarray, x: int, y: int, radius: int, color: tuple[int, ...]) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    canvas[mask] = color


def draw_ellipse(
    canvas: np.ndarray,
    x: int,
    y: int,
    radius_x: int,
    radius_y: int,
    color: tuple[int, ...],
) -> None:
    yy, xx = np.ogrid[: canvas.shape[0], : canvas.shape[1]]
    mask = ((xx - x) / radius_x) ** 2 + ((yy - y) / radius_y) ** 2 <= 1.0
    canvas[mask] = color


def draw_line(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, ...],
    width: int = 2,
) -> None:
    samples = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1) + 1
    xs = np.rint(np.linspace(start[0], end[0], samples)).astype(int)
    ys = np.rint(np.linspace(start[1], end[1], samples)).astype(int)
    for offset_x in range(-width, width + 1):
        for offset_y in range(-width, width + 1):
            valid = (
                (xs + offset_x >= 0)
                & (xs + offset_x < canvas.shape[1])
                & (ys + offset_y >= 0)
                & (ys + offset_y < canvas.shape[0])
            )
            canvas[ys[valid] + offset_y, xs[valid] + offset_x] = color


def composite_frame(image: np.ndarray, sticks: np.ndarray, passed: bool) -> np.ndarray:
    canvas = np.full((400, 800, 3), 16, dtype=np.uint8)
    scale = 12
    fpv = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    fpv_rgb = np.repeat(fpv[..., None], 3, axis=2)
    canvas[8 : 8 + fpv.shape[0], 8 : 8 + fpv.shape[1]] = fpv_rgb
    border = (40, 220, 80) if passed else (210, 210, 210)
    for inset in range(3):
        canvas[5 + inset, 5:395] = border
        canvas[395 - inset, 5:395] = border
        canvas[5:396, 5 + inset] = border
        canvas[5:396, 395 - inset] = border

    # A deliberately schematic fly: its two front-leg endpoints are the two stick dots.
    draw_ellipse(canvas, 600, 75, 30, 54, (112, 82, 38))
    draw_disk(canvas, 600, 32, 22, (150, 108, 48))
    draw_ellipse(canvas, 552, 72, 44, 22, (72, 88, 98))
    draw_ellipse(canvas, 648, 72, 44, 22, (72, 88, 98))

    centers = ((515, 270), (685, 270))
    axes = ((sticks[0], sticks[1]), (sticks[2], sticks[3]))
    shoulders = ((580, 92), (620, 92))
    colors = ((255, 144, 45), (70, 180, 255))
    for center, axis, shoulder, color in zip(centers, axes, shoulders, colors, strict=True):
        draw_disk(canvas, center[0], center[1], 75, (42, 42, 48))
        draw_disk(canvas, center[0], center[1], 72, (20, 20, 24))
        draw_line(canvas, (center[0] - 65, center[1]), (center[0] + 65, center[1]), (75, 75, 82))
        draw_line(canvas, (center[0], center[1] - 65), (center[0], center[1] + 65), (75, 75, 82))
        endpoint = (round(center[0] + 62 * axis[0]), round(center[1] - 62 * axis[1]))
        draw_line(canvas, shoulder, endpoint, color, width=3)
        draw_disk(canvas, endpoint[0], endpoint[1], 9, color)
    return canvas


@torch.no_grad()
def main() -> int:
    args = parse_args()
    if args.episodes < 1 or args.video_fps < 1:
        raise SystemExit("--episodes and --video-fps must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=checkpoint["retinal_receptive_field"],
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    quad = DifferentiableQuad(hover_config).to(device)
    leg_plant = ForelegStickPlant(hover_config).to(device)
    state = quad.initial_state(args.episodes, device=device, dtype=torch.float32)
    gate = sample_annular_gates(
        args.episodes,
        device=device,
        dtype=torch.float32,
        config=gate_config,
        strict=True,
    )
    mass_scale = torch.empty(args.episodes, device=device).uniform_(0.92, 1.08)
    stick_state = leg_plant.initial_state(args.episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(args.episodes, device=device, dtype=torch.float32)

    steps = round(args.seconds / hover_config.dt)
    capture_stride = max(1, round(1.0 / (args.video_fps * hover_config.dt)))
    passed = torch.zeros(args.episodes, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    lifted = torch.zeros_like(passed)
    recontact = torch.zeros_like(passed)
    cleared = torch.zeros_like(passed)
    pass_step = torch.full((args.episodes,), -1, dtype=torch.long, device=device)
    max_tilt = torch.zeros(args.episodes, device=device)
    saturation_steps = torch.zeros(args.episodes, device=device)
    captured: dict[str, list[np.ndarray]] = {
        name: [] for name in ("image", "motor", "rc", "sticks", "position", "euler")
    }

    for step in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=args.resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        rc, stick_state = leg_plant(motor, stick_state)
        if step % capture_stride == 0:
            captured["image"].append((255.0 * image).byte().cpu().numpy())
            captured["motor"].append(motor.cpu().numpy())
            captured["rc"].append(rc.cpu().numpy())
            captured["sticks"].append(stick_state.position.cpu().numpy())
            captured["position"].append(state.position.cpu().numpy())
            captured["euler"].append(state.euler.cpu().numpy())

        previous_position = state.position
        state = quad(rc, state, mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, gate, gate_config
        )
        new_pass = pass_now & ~passed
        pass_step[new_pass] = step + 1
        passed |= pass_now
        collision |= collision_now
        missed |= miss_now
        signed, _, _ = gate_coordinates(state.position, gate)
        cleared |= passed & (signed >= 0.4)
        lifted |= state.position[:, 2] > 0.15
        recontact |= lifted & (state.position[:, 2] <= 0.01)
        max_tilt = torch.maximum(
            max_tilt, torch.linalg.vector_norm(state.euler[:, :2], dim=1)
        )
        saturation_steps += (stick_state.position.abs() > 0.98).any(dim=1)

    success = (
        passed
        & cleared
        & (pass_step <= steps - round(1.0 / hover_config.dt))
        & ~collision
        & ~missed
        & ~recontact
        & (max_tilt <= math.radians(40.0))
        & (saturation_steps / steps <= 0.25)
    )
    successful = torch.nonzero(success, as_tuple=False).squeeze(-1)
    if not successful.numel():
        raise SystemExit(f"no successful flight among {args.episodes} seeded episodes")
    chosen = int(successful[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = {name: np.stack(values)[:, chosen] for name, values in captured.items()}
    trajectory_path = args.output_dir / "showcase-trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        **selected,
        time_s=np.arange(len(selected["image"])) * capture_stride * hover_config.dt,
        gate_center=gate.center[chosen].cpu().numpy(),
        gate_yaw=gate.yaw[chosen].cpu().numpy(),
        mass_scale=mass_scale[chosen].cpu().numpy(),
    )

    video_path = args.output_dir / "showcase.mp4"
    encoder = subprocess.Popen(
        (
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "800x400",
            "-r",
            str(args.video_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ),
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    pass_frame = math.ceil(int(pass_step[chosen]) / capture_stride)
    for frame, (image, sticks) in enumerate(
        zip(selected["image"], selected["sticks"], strict=True)
    ):
        encoder.stdin.write(composite_frame(image, sticks, frame >= pass_frame).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise SystemExit("ffmpeg failed to encode the showcase")

    summary = {
        "seed": args.seed,
        "sampled_episodes": args.episodes,
        "selected_episode": chosen,
        "pass_time_seconds": float(pass_step[chosen] * hover_config.dt),
        "gate_center_m": gate.center[chosen].cpu().tolist(),
        "gate_yaw_degrees": float(torch.rad2deg(gate.yaw[chosen])),
        "mass_scale": float(mass_scale[chosen]),
        "max_tilt_degrees": float(torch.rad2deg(max_tilt[chosen])),
        "gate_config": asdict(gate_config),
        "actor_inputs": ["monochrome_fpv", "roll", "pitch", "recurrent_neural_state"],
        "actor_outputs": ["right_foreleg_roll_pitch", "left_foreleg_yaw_throttle"],
        "external_actor_state_machine": False,
        "privileged_gate_geometry_given_to_actor": False,
    }
    summary_path = args.output_dir / "showcase.json"
    summary_path.write_text(f"{json.dumps(summary, indent=2, sort_keys=True)}\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {trajectory_path}")
    print(f"wrote {video_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
