#!/usr/bin/env python3
"""Benchmark the full RGB-renderer-to-MaleCNS forward path headlessly."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import ConnectomeController, DifferentiableQuad  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--backward-unroll",
        type=int,
        default=0,
        help="benchmark one gradient step through this many recurrent steps",
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    if args.batch < 1 or args.warmup < 0 or args.steps < 1 or args.backward_unroll < 0:
        raise ValueError("batch and steps must be positive; warmup/unroll must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    controller = ConnectomeController(args.graph).to(device).eval()
    quad = DifferentiableQuad().to(device)
    position = torch.zeros(args.batch, 3, device=device)
    position[:, 2] = 1.0
    physical_state = quad.initial_state(
        args.batch, device=device, dtype=torch.float32, position=position
    )
    target_height = torch.full((args.batch,), 1.0, device=device)
    neural_state = controller.initial_state(args.batch, device=device, dtype=torch.float32)

    def step(neural: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = render_visual_hover_scene(physical_state, target_height)
        motor, following = controller(image, physical_state.euler[:, :2], neural)
        return motor, following

    with torch.inference_mode():
        for _ in range(args.warmup):
            _, neural_state = step(neural_state)
    neural_state = neural_state.detach()

    if args.backward_unroll:
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        controller.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss = torch.zeros((), device=device)
        for _ in range(args.backward_unroll):
            motor, neural_state = step(neural_state)
            loss = loss + motor.square().mean()
        loss.backward()
        synchronize(device)
        elapsed = time.perf_counter() - started
        measured_steps = args.backward_unroll
        mode = "forward_backward"
    else:
        with torch.inference_mode():
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(args.steps):
                _, neural_state = step(neural_state)
            synchronize(device)
            elapsed = time.perf_counter() - started
        measured_steps = args.steps
        mode = "inference"

    report = {
        "mode": mode,
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"),
        "batch": args.batch,
        "nodes": controller.n_nodes,
        "edges": int(controller.edge_pre.numel()),
        "camera": [320, 200],
        "steps": measured_steps,
        "elapsed_seconds": elapsed,
        "neural_steps_per_second": measured_steps / elapsed,
        "camera_frames_per_second": measured_steps * args.batch / elapsed,
        "realtime_factor_at_100_hz": measured_steps / elapsed / 100.0,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "final_state_finite": bool(torch.isfinite(neural_state).all().item()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
