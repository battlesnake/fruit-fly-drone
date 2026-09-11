#!/usr/bin/env python3
"""Disposable CUDA integration check for the neural-rate audit machinery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_neural_integration_rate as rate_audit  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402


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


def synthetic_cache() -> dict:
    cases = len(motion.HISTORY_LENGTHS)
    height, width = 200, 320
    prefix = torch.zeros(cases, 3, height, width)
    prefix[:, 0] = torch.linspace(0.0, 0.2, width).reshape(1, 1, width)
    groups = {}
    for index, horizon in enumerate(motion.HISTORY_LENGTHS):
        images = torch.zeros(horizon, 1, 2, 3, height, width)
        ramp = torch.linspace(0.0, 0.1, horizon).reshape(-1, 1, 1)
        images[:, 0, 0, 1] = ramp
        images[:, 0, 1, 1] = torch.flip(ramp, dims=(0,))
        # The final frame is deliberately identical across the two motion branches.
        images[-1, 0, 1] = images[-1, 0, 0]
        groups[str(horizon)] = {
            "case_indices": torch.tensor([index]),
            "response_images": images,
            "roll_pitch": torch.zeros(1, 2, 2),
        }
    return {
        "cases": cases,
        "prefix_images": prefix,
        "prefix_roll_pitch": torch.zeros(cases, 2),
        "prefix_repetitions": 2,
        "groups": groups,
        "teacher_targets": torch.tensor((0.02, 0.03, 0.04)),
        "teacher_scale": 0.03,
        "horizon": torch.tensor(motion.HISTORY_LENGTHS),
        "endpoint_image_difference": torch.zeros(cases),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if assisted.responsibility.file_sha256(args.graph) != assisted.EXPECTED_GRAPH_SHA256:
        raise SystemExit("graph hash mismatch")
    if assisted.responsibility.file_sha256(args.checkpoint) != assisted.EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("checkpoint hash mismatch")
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cache = synthetic_cache()
    results = {}
    for substeps in (1, 2):
        results[str(substeps)] = rate_audit.evaluate_condition(
            graph=args.graph,
            checkpoint_state=loaded["controller"],
            cache=cache,
            substeps=substeps,
            device=device,
        )
    passed = bool(
        all(
            result["source_loaded_exactly"]
            and result["source_restored"]
            and result["all_recurrent_states_and_outputs_finite"]
            and result["all_metrics_finite"]
            and len(result["prediction_contrasts"]) == len(motion.HISTORY_LENGTHS)
            and len(result["terminal_motor_outputs"]) == len(motion.HISTORY_LENGTHS)
            for result in results.values()
        )
        and results["1"]["neural_dt_seconds"] == 1.0 / 50
        and results["2"]["neural_dt_seconds"] == 1.0 / 100
        and results["2"]["native_branch_state_updates"]
        == 2 * results["1"]["native_branch_state_updates"]
    )
    print(
        json.dumps(
            {
                "pass": passed,
                "device": str(device),
                "k1_wall_time_seconds": results["1"]["wall_time_seconds"],
                "k2_wall_time_seconds": results["2"]["wall_time_seconds"],
                "k1_prediction_contrasts": results["1"]["prediction_contrasts"],
                "k2_prediction_contrasts": results["2"]["prediction_contrasts"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
