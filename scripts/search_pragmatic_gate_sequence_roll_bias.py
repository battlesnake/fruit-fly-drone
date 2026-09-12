#!/usr/bin/env python3
"""Sweep an internal roll-motor bias for uninterrupted gate-sequence flight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_pragmatic_two_gate_zero_shot import (  # noqa: E402
    evaluate,
    sample_two_gate_cases,
)
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


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
        default=REPO_ROOT
        / "runs/gate/pragmatic-two-gate-aligned-roll-path-001/best-controller.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=22.0)
    parser.add_argument("--observation-warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=880_983)
    parser.add_argument("--spacing-min", type=float, default=3.8)
    parser.add_argument("--spacing-max", type=float, default=4.2)
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=(-0.01, -0.005, -0.0025, -0.00125, 0.0, 0.00125, 0.0025, 0.005, 0.01),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-checkpoint", type=Path)
    return parser.parse_args()


def selection_score(metrics: dict[str, object], first_gate_floor: float) -> tuple[float, ...]:
    retained = float(metrics["first_gate_pass_rate"]) >= first_gate_floor
    radial = metrics["second_gate_crossing_radial_mean_metres"]
    return (
        float(retained),
        float(metrics["both_gates_pass_rate"]),
        min(
            float(metrics["both_gates_negative_course_pass_rate"]),
            float(metrics["both_gates_positive_course_pass_rate"]),
        ),
        float(metrics["both_gates_paired_pass_rate"]),
        -float(radial) if radial is not None else float("-inf"),
        float(metrics["first_gate_pass_rate"]),
    )


def main() -> int:
    args = parse_args()
    if args.pairs < 1 or args.seconds <= 0 or not args.offsets:
        raise SystemExit("pairs, seconds and offsets must be positive/nonempty")
    device = torch.device(args.device)
    controller, payload = load_controller(args, device)
    controller.eval().requires_grad_(False)
    hover_config = HoverConfig(**payload["hover_config"])
    gate_config = GateConfig(**payload["gate_config"])
    camera = CameraSpec(
        width=payload["image_resolution"][0],
        height=payload["image_resolution"][1],
        horizontal_fov_degrees=payload["camera_hfov_degrees"],
    )
    cases, gates = sample_two_gate_cases(
        args.pairs,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
        spacing_range=(args.spacing_min, args.spacing_max),
    )
    positive = controller.pool_indices[
        int(controller.pool_offsets[0]) : int(controller.pool_offsets[1])
    ]
    negative = controller.pool_indices[
        int(controller.pool_offsets[1]) : int(controller.pool_offsets[2])
    ]
    source_bias = controller.bias.detach().clone()
    records = []
    baseline_metrics: dict[str, object] | None = None
    for offset in args.offsets:
        with torch.no_grad():
            controller.bias.copy_(source_bias)
            controller.bias[positive] += offset
            controller.bias[negative] -= offset
        metrics = evaluate(
            controller,
            cases,
            gates,
            seconds=args.seconds,
            warmup_steps=args.observation_warmup_steps,
            camera=camera,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        if offset == 0.0:
            baseline_metrics = metrics
        record = {"roll_motor_pool_bias_offset": offset, "metrics": metrics}
        records.append(record)
        print(json.dumps(record), flush=True)
    if baseline_metrics is None:
        with torch.no_grad():
            controller.bias.copy_(source_bias)
        baseline_metrics = evaluate(
            controller,
            cases,
            gates,
            seconds=args.seconds,
            warmup_steps=args.observation_warmup_steps,
            camera=camera,
            hover_config=hover_config,
            gate_config=gate_config,
        )
    floor = max(
        0.0,
        float(baseline_metrics["first_gate_pass_rate"]) - 1.0 / (2 * args.pairs),
    )
    best = max(records, key=lambda record: selection_score(record["metrics"], floor))
    result = {
        "experiment": "pragmatic-gate-sequence-roll-motor-bias-sweep-v1",
        "purpose": "rapid internal calibration probe; not a formal promotion run",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "episodes_per_offset": 2 * args.pairs,
        "roll_positive_pool_neurons": len(positive),
        "roll_negative_pool_neurons": len(negative),
        "first_gate_retention_floor": floor,
        "records": records,
        "selected": best,
    }
    if args.save_checkpoint is not None:
        offset = float(best["roll_motor_pool_bias_offset"])
        with torch.no_grad():
            controller.bias.copy_(source_bias)
            controller.bias[positive] += offset
            controller.bias[negative] -= offset
        saved = dict(payload)
        saved["experiment"] = result["experiment"]
        saved["controller"] = {
            name: value.detach().cpu() for name, value in controller.state_dict().items()
        }
        saved["roll_motor_pool_bias_offset"] = offset
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.save_checkpoint)
        result["saved_checkpoint"] = str(args.save_checkpoint)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"stage": "complete", **result}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
