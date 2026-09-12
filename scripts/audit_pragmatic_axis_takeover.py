#!/usr/bin/env python3
"""Privileged axis-takeover diagnosis; assisted flights never count toward the goal."""

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

from evaluate_pragmatic_two_gate_zero_shot import evaluate, sample_two_gate_cases  # noqa: E402
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1110983)
    parser.add_argument("--seconds", type=float, default=30)
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
    result = dict(
        experiment="diagnostic-post-first-gate-axis-takeover-v1",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        geometry=GEOMETRY,
        assisted_results_are_not_fly_success=True,
        actor_still_updates_native_neural_state=True,
        takeover_teacher_heading_mode="rate-damped",
        conditions={},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, axes in (
        ("native", None),
        ("teacher_roll", (True, False, False, False)),
        ("teacher_other_axes", (False, True, True, True)),
        ("teacher_all_axes", (True, True, True, True)),
    ):
        metrics = evaluate(
            controller,
            cases,
            gates,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
            diagnostic_teacher_axes=axes,
        )
        conditional = {}
        for side in ("all", "negative", "positive"):
            first_key = (
                "clean_first_gate_pass_rate"
                if side == "all"
                else f"clean_first_gate_{side}_pass_rate"
            )
            clean_key = (
                "clean_course_success_rate"
                if side == "all"
                else f"clean_course_{side}_success_rate"
            )
            conditional[side] = (
                metrics[clean_key] / metrics[first_key] if metrics[first_key] else None
            )
        result["conditions"][name] = dict(
            metrics=metrics, clean_completion_given_clean_first=conditional
        )
        result["elapsed_seconds"] = perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    condition=name,
                    clean=metrics["clean_course_success_rate"],
                    first=metrics["clean_first_gate_pass_rate"],
                    conditional=conditional,
                    elapsed_seconds=result["elapsed_seconds"],
                )
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
