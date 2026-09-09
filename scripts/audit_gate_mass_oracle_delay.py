#!/usr/bin/env python3
"""Measure how long exact-mass throttle correction can be withheld."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from search_gate_acceleration_es import compact_metrics  # noqa: E402
from train_gate import RETINAL_FLIP_X, evaluate_gate, file_sha256  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "mass-oracle-delay",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=170_031)
    parser.add_argument(
        "--delays",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.calibration):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.episodes <= 0 or args.episodes % 8:
        raise SystemExit("--episodes must be a positive multiple of 8")
    if args.seconds <= 0.0:
        raise SystemExit("--seconds must be positive")
    if not args.delays or min(args.delays) < 0.0 or max(args.delays) >= args.seconds:
        raise SystemExit("delays must be nonnegative and shorter than the episode")


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected privileged mass oracle")

    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    resolution = int(checkpoint["image_resolution"])
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()

    started = perf_counter()
    results: dict[str, dict[str, Any]] = {}
    for delay in sorted(set(args.delays)):
        metrics = evaluate_gate(
            controller,
            episodes=args.episodes,
            seconds=args.seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            seed=args.seed,
            balanced_strata=True,
            privileged_mass_trim=(calibration["intercept"], calibration["mass_slope"]),
            privileged_mass_trim_after_seconds=delay,
        )
        results[f"{delay:g}"] = metrics
        print(
            json.dumps(
                {
                    "delay_seconds": delay,
                    **compact_metrics(metrics, clean_radius),
                    "goal_pass": metrics["goal_pass"],
                }
            ),
            flush=True,
        )

    passing_delays = [float(delay) for delay, result in results.items() if result["goal_pass"]]
    report = {
        "experiment": "privileged exact-mass correction onset-delay audit",
        "claim_scope": (
            "Timing upper bound only. Exact simulator mass still changes neuronal drive and is "
            "not a biological or onboard sensor."
        ),
        "counts_toward_direct_sensor_goal": False,
        "controller_parameters_changed": False,
        "per_episode_neuronal_drive_bias_changed_after_delay": True,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "mass_trim": {
            "intercept": calibration["intercept"],
            "mass_slope": calibration["mass_slope"],
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            "episodes_per_delay": args.episodes,
            "seconds": args.seconds,
            "seed": args.seed,
            "delays_seconds": sorted(set(args.delays)),
            "same_initial_conditions_for_every_delay": True,
            "exactly_balanced_mass_side_obliquity_strata": True,
            "abrupt_bias_onset": True,
        },
        "maximum_delay_meeting_90_percent_goal_seconds": (
            max(passing_delays) if passing_delays else None
        ),
        "results": results,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "maximum_delay_meeting_90_percent_goal_seconds": report[
                    "maximum_delay_meeting_90_percent_goal_seconds"
                ],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
