#!/usr/bin/env python3
"""Test whether exact simulator mass enables a useful throttle-pool trim."""

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

from search_gate_acceleration_es import compact_metrics, task_fitness  # noqa: E402
from train_gate import RETINAL_FLIP_X, evaluate_gate, file_sha256  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def stable_path(path: Path) -> str:
    """Prefer stable repository-relative paths in committed reports."""

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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "mass-trim-oracle",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--development-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--development-seed", type=int, default=140_031)
    parser.add_argument("--validation-seed", type=int, default=150_031)
    parser.add_argument("--final-seed", type=int, default=160_031)
    parser.add_argument("--top-mass-candidates", type=int, default=5)
    parser.add_argument("--top-constant-candidates", type=int, default=3)
    parser.add_argument(
        "--intercepts",
        type=float,
        nargs="+",
        default=(-0.05, -0.025, 0.0, 0.025, 0.05),
    )
    parser.add_argument(
        "--slopes",
        type=float,
        nargs="+",
        default=(-0.10, 0.0, 0.025, 0.05, 0.10, 0.20),
    )
    return parser.parse_args()


def evaluate(
    controller: ConnectomeController,
    *,
    intercept: float,
    slope: float,
    shuffled: bool,
    episodes: int,
    seed: int,
    seconds: float,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    return evaluate_gate(
        controller,
        episodes=episodes,
        seconds=seconds,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=seed,
        balanced_strata=True,
        privileged_mass_trim=(intercept, slope),
        shuffled_privileged_mass=shuffled,
    )


def score(metrics: dict[str, Any], clean_radius: float) -> tuple[float, float, float]:
    return (
        task_fitness(metrics, clean_radius),
        metrics["success_rate"],
        -abs(metrics["mass_vertical_error_correlation"] or 0.0),
    )


def summary(metrics: dict[str, Any], clean_radius: float) -> dict[str, Any]:
    return {
        **compact_metrics(metrics, clean_radius),
        "mass_vertical_error_correlation": metrics["mass_vertical_error_correlation"],
    }


def main() -> int:
    args = parse_args()
    for name in ("development_episodes", "validation_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match evaluator")

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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    development: list[dict[str, Any]] = []
    for intercept in args.intercepts:
        for slope in args.slopes:
            metrics = evaluate(
                controller,
                intercept=intercept,
                slope=slope,
                shuffled=False,
                episodes=args.development_episodes,
                seed=args.development_seed,
                seconds=args.seconds,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            entry = {
                "intercept": intercept,
                "mass_slope": slope,
                "metrics": metrics,
                "summary": summary(metrics, clean_radius),
            }
            development.append(entry)
            print(
                json.dumps(
                    {
                        "phase": "development",
                        "intercept": intercept,
                        "mass_slope": slope,
                        **entry["summary"],
                    }
                ),
                flush=True,
            )

    mass_ranked = sorted(
        (entry for entry in development if entry["mass_slope"] != 0.0),
        key=lambda entry: score(entry["metrics"], clean_radius),
        reverse=True,
    )[: args.top_mass_candidates]
    constant_ranked = sorted(
        (entry for entry in development if entry["mass_slope"] == 0.0),
        key=lambda entry: score(entry["metrics"], clean_radius),
        reverse=True,
    )[: args.top_constant_candidates]

    validation: list[dict[str, Any]] = []
    for source in (*mass_ranked, *constant_ranked):
        metrics = evaluate(
            controller,
            intercept=source["intercept"],
            slope=source["mass_slope"],
            shuffled=False,
            episodes=args.validation_episodes,
            seed=args.validation_seed,
            seconds=args.seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        entry = {
            "intercept": source["intercept"],
            "mass_slope": source["mass_slope"],
            "metrics": metrics,
            "summary": summary(metrics, clean_radius),
        }
        validation.append(entry)
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "intercept": entry["intercept"],
                    "mass_slope": entry["mass_slope"],
                    **entry["summary"],
                }
            ),
            flush=True,
        )

    mass_winner = max(
        (entry for entry in validation if entry["mass_slope"] != 0.0),
        key=lambda entry: score(entry["metrics"], clean_radius),
    )
    constant_winner = max(
        (entry for entry in validation if entry["mass_slope"] == 0.0),
        key=lambda entry: score(entry["metrics"], clean_radius),
    )
    final_specs = {
        "unchanged_controller": (0.0, 0.0, False),
        "validation_selected_constant_trim": (constant_winner["intercept"], 0.0, False),
        "mass_conditioned_trim": (
            mass_winner["intercept"],
            mass_winner["mass_slope"],
            False,
        ),
        "mass_labels_shuffled_within_geometry": (
            mass_winner["intercept"],
            mass_winner["mass_slope"],
            True,
        ),
    }
    final: dict[str, Any] = {}
    for name, (intercept, slope, shuffled) in final_specs.items():
        metrics = evaluate(
            controller,
            intercept=intercept,
            slope=slope,
            shuffled=shuffled,
            episodes=args.final_episodes,
            seed=args.final_seed,
            seconds=args.seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        final[name] = metrics
        print(
            json.dumps({"phase": "final", "name": name, **summary(metrics, clean_radius)}),
            flush=True,
        )

    conditioned = final["mass_conditioned_trim"]
    constant = final["validation_selected_constant_trim"]
    shuffled = final["mass_labels_shuffled_within_geometry"]
    causal_checks = {
        "beats_unchanged_controller": (
            conditioned["success_rate"] > final["unchanged_controller"]["success_rate"]
        ),
        "beats_validation_selected_constant_trim": (
            conditioned["success_rate"] > constant["success_rate"]
        ),
        "beats_shuffled_mass_labels": conditioned["success_rate"] > shuffled["success_rate"],
        "reduces_absolute_mass_vertical_error_correlation_vs_constant": abs(
            conditioned["mass_vertical_error_correlation"] or 0.0
        )
        < abs(constant["mass_vertical_error_correlation"] or 0.0),
    }
    causal_checks["oracle_useful_for_this_controller"] = all(causal_checks.values())

    candidate = {
        "kind": "privileged_non_biological_mass_oracle",
        "normalization": "z = (mass_scale - 1) / 0.08, clamped to [-1, 1]",
        "motor_pool_bias": "b = intercept + mass_slope * z",
        "intercept": mass_winner["intercept"],
        "mass_slope": mass_winner["mass_slope"],
        "counts_toward_direct_sensor_goal": False,
    }
    (args.output_dir / "candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n"
    )
    report = {
        "experiment": "privileged exact-mass conditioned throttle-pool trim",
        "claim_scope": (
            "Diagnostic upper bound only. Simulator mass directly changes actor motor-pool "
            "biases and is not a biological or onboard sensor."
        ),
        "counts_toward_direct_sensor_goal": False,
        "controller_parameters_changed": False,
        "per_episode_neuronal_drive_bias_changed": True,
        "synaptic_weights_changed": False,
        "connectome_edges_changed": False,
        "external_runtime_parameters": 2,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "search": {
            "intercepts": args.intercepts,
            "mass_slopes": args.slopes,
            "development_episodes": args.development_episodes,
            "validation_episodes": args.validation_episodes,
            "final_episodes": args.final_episodes,
            "development_seed": args.development_seed,
            "validation_seed": args.validation_seed,
            "final_seed": args.final_seed,
            "exactly_balanced_mass_side_obliquity_strata": True,
        },
        "development": development,
        "validation": validation,
        "selected_constant_trim": {
            "intercept": constant_winner["intercept"],
            "mass_slope": 0.0,
        },
        "selected_mass_conditioned_trim": candidate,
        "final": final,
        "causal_checks": causal_checks,
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(report_path), **causal_checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
