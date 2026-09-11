#!/usr/bin/env python3
"""Disposable CUDA integration check for exponential-Euler and RK4 CNS solvers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_continuous_cns_solver as solver  # noqa: E402
import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import check_variable_height_neural_integration_rate_audit as check_rate  # noqa: E402


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
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    rate.validate_inputs(
        argparse.Namespace(
            graph=args.graph,
            checkpoint=args.checkpoint,
            motion_report=REPO_ROOT
            / "runs/variable-height-hover/native-throttle-motion-only-001/report.json",
            motion_resume=REPO_ROOT
            / "runs/variable-height-hover/native-throttle-motion-only-001/resume.pt",
        )
    )
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cache = check_rate.synthetic_cache()
    results = {}
    for method, steps in (("exponential_euler", 32), ("rk4", 1), ("rk4", 2)):
        name = f"{method}_{steps}"
        results[name] = solver.evaluate_condition(
            graph=args.graph,
            checkpoint_state=loaded["controller"],
            cache=cache,
            method=method,
            solver_steps=steps,
            device=device,
        )
    passed = all(
        value["source_loaded_exactly"]
        and value["source_restored"]
        and value["all_recurrent_states_and_outputs_finite"]
        and value["all_metrics_finite"]
        for value in results.values()
    )
    print(
        json.dumps(
            {
                "pass": passed,
                "device": str(device),
                "conditions": {
                    name: {
                        "graph_evaluations_per_camera_frame": value[
                            "graph_evaluations_per_camera_frame"
                        ],
                        "wall_time_seconds": value["wall_time_seconds"],
                        "prediction_contrasts": value["prediction_contrasts"],
                    }
                    for name, value in results.items()
                },
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
