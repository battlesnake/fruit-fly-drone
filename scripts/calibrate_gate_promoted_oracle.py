#!/usr/bin/env python3
"""Calibrate a privileged mass oracle for the promoted native gate controller."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_motor_interface_es import stable_path  # noqa: E402
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    controller_parameter_sha256,
    load_frozen_controller,
    teacher_takeover_audit,
)

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


@dataclass(frozen=True)
class BiasSpec:
    light_bias: float
    heavy_bias: float

    @property
    def intercept(self) -> float:
        return 0.5 * (self.light_bias + self.heavy_bias)

    @property
    def mass_slope(self) -> float:
        return 0.5 * (self.heavy_bias - self.light_bias)


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
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "promoted-oracle-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--target-onset-seconds", type=float, default=0.25)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--development-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--preflight-episodes", type=int, default=1024)
    parser.add_argument("--development-seed", type=int, default=995_031)
    parser.add_argument("--validation-seed", type=int, default=996_031)
    parser.add_argument("--preflight-seed", type=int, default=997_031)
    parser.add_argument("--validation-candidates", type=int, default=5)
    parser.add_argument("--validation-constant-candidates", type=int, default=2)
    parser.add_argument("--minimum-preflight-success", type=float, default=0.90)
    parser.add_argument(
        "--light-endpoint-biases",
        type=float,
        nargs="+",
        default=(-0.20, -0.175, -0.15, -0.125, -0.10),
    )
    parser.add_argument(
        "--heavy-endpoint-biases",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.075, 0.10),
    )
    parser.add_argument(
        "--constant-biases",
        type=float,
        nargs="+",
        default=(-0.10, -0.075, -0.05, -0.025, 0.0),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    for name in ("development_episodes", "validation_episodes", "preflight_episodes"):
        value = getattr(args, name)
        if value <= 0 or value % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive and divisible by eight")
    if not 0.0 < args.target_onset_seconds < args.takeover_seconds < args.seconds:
        raise SystemExit("require 0 < target onset < takeover < evaluation duration")
    if not 0.0 < args.minimum_preflight_success <= 1.0:
        raise SystemExit("preflight threshold must be in (0, 1]")
    if args.validation_candidates <= 0 or args.validation_constant_candidates <= 0:
        raise SystemExit("validation candidate counts must be positive")
    if not args.light_endpoint_biases or not args.heavy_endpoint_biases:
        raise SystemExit("endpoint grids cannot be empty")


def calibration(spec: BiasSpec) -> dict[str, Any]:
    return {
        "kind": "privileged_promoted_native_mass_oracle",
        "normalization": "z = (mass_scale - 1) / 0.08, clamped to [-1, 1]",
        "motor_pool_bias": "b = intercept + mass_slope * z",
        "intercept": spec.intercept,
        "mass_slope": spec.mass_slope,
        "light_endpoint_bias": spec.light_bias,
        "heavy_endpoint_bias": spec.heavy_bias,
    }


def shuffled_calibration(spec: BiasSpec, mass_scale: Tensor) -> dict[str, Any]:
    normalized = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    shuffled = normalized[torch.arange(len(normalized), device=mass_scale.device).bitwise_xor(1)]
    # oracle_bias() will add slope * normalized. The per-episode intercept below
    # cancels it and substitutes the adjacent matched partner's label.
    per_episode_intercept = spec.intercept + spec.mass_slope * (shuffled - normalized)
    return calibration(spec) | {"intercept": per_episode_intercept}


def score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    worst_mass = min(metrics["light_success_rate"], metrics["heavy_success_rate"])
    return (worst_mass, metrics["success_rate"], -metrics["crossing_radial_mean_m"])


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "success_rate",
            "light_success_rate",
            "heavy_success_rate",
            "negative_lateral_success_rate",
            "positive_lateral_success_rate",
            "negative_obliquity_success_rate",
            "positive_obliquity_success_rate",
            "ring_collision_rate",
            "miss_rate",
            "crossing_radial_mean_m",
        )
    }


def evaluate(
    controller: ConnectomeController,
    cases: Any,
    spec: BiasSpec,
    *,
    shuffled: bool,
    args: argparse.Namespace,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: Any,
) -> dict[str, Any]:
    labels = shuffled_calibration(spec, cases.mass_scale) if shuffled else calibration(spec)
    return teacher_takeover_audit(
        controller,
        controller,
        cases,
        labels,
        takeover_seconds=args.takeover_seconds,
        seconds=args.seconds,
        target_onset_seconds=args.target_onset_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )


def evaluate_specs(
    phase: str,
    controller: ConnectomeController,
    cases: Any,
    specs: list[BiasSpec],
    *,
    args: argparse.Namespace,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: Any,
) -> list[dict[str, Any]]:
    results = []
    for spec in specs:
        metrics = evaluate(
            controller,
            cases,
            spec,
            shuffled=False,
            args=args,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        result = {"spec": asdict(spec), "calibration": calibration(spec), "metrics": metrics}
        results.append(result)
        print(
            json.dumps({"phase": phase, **asdict(spec), **compact(metrics)}),
            flush=True,
        )
    return results


def half_step_refinement(winner: BiasSpec, existing: set[BiasSpec]) -> list[BiasSpec]:
    half_step = 0.0125
    return [
        BiasSpec(winner.light_bias + light_delta, winner.heavy_bias + heavy_delta)
        for light_delta in (-half_step, 0.0, half_step)
        for heavy_delta in (-half_step, 0.0, half_step)
        if BiasSpec(winner.light_bias + light_delta, winner.heavy_bias + heavy_delta)
        not in existing
    ]


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, _, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.checkpoint, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.development_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    development_cases = diverse_matched_cases(
        args.development_episodes,
        seed=args.development_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    grid = [
        BiasSpec(light, heavy)
        for light in args.light_endpoint_biases
        for heavy in args.heavy_endpoint_biases
    ]
    development = evaluate_specs(
        "development_grid",
        controller,
        development_cases,
        grid,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    development_winner = max(development, key=lambda item: score(item["metrics"]))
    winner_spec = BiasSpec(**development_winner["spec"])
    refinement = half_step_refinement(winner_spec, set(grid))
    development_refinement = evaluate_specs(
        "development_refinement",
        controller,
        development_cases,
        refinement,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    development.extend(development_refinement)
    constant_specs = [BiasSpec(value, value) for value in args.constant_biases]
    development_constants = evaluate_specs(
        "development_constant",
        controller,
        development_cases,
        constant_specs,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )

    ranked = sorted(development, key=lambda item: score(item["metrics"]), reverse=True)
    ranked_constants = sorted(
        development_constants, key=lambda item: score(item["metrics"]), reverse=True
    )
    validation_specs = [BiasSpec(**item["spec"]) for item in ranked[: args.validation_candidates]]
    validation_constant_specs = [
        BiasSpec(**item["spec"]) for item in ranked_constants[: args.validation_constant_candidates]
    ]
    validation_cases = diverse_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    validation = evaluate_specs(
        "validation",
        controller,
        validation_cases,
        validation_specs,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    validation_constants = evaluate_specs(
        "validation_constant",
        controller,
        validation_cases,
        validation_constant_specs,
        args=args,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    selected = max(validation, key=lambda item: score(item["metrics"]))
    selected_spec = BiasSpec(**selected["spec"])
    selected_constant = max(validation_constants, key=lambda item: score(item["metrics"]))
    selected_constant_spec = BiasSpec(**selected_constant["spec"])

    preflight_cases = diverse_matched_cases(
        args.preflight_episodes,
        seed=args.preflight_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    zero = BiasSpec(0.0, 0.0)
    preflight = {
        "unchanged": evaluate(
            controller,
            preflight_cases,
            zero,
            shuffled=False,
            args=args,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        ),
        "validation_selected_constant": evaluate(
            controller,
            preflight_cases,
            selected_constant_spec,
            shuffled=False,
            args=args,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        ),
        "validation_selected_mass_conditioned": evaluate(
            controller,
            preflight_cases,
            selected_spec,
            shuffled=False,
            args=args,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        ),
        "matched_pair_mass_labels_swapped": evaluate(
            controller,
            preflight_cases,
            selected_spec,
            shuffled=True,
            args=args,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        ),
    }
    for name, metrics in preflight.items():
        print(json.dumps({"phase": "preflight", "name": name, **compact(metrics)}), flush=True)
    conditioned = preflight["validation_selected_mass_conditioned"]
    threshold_checks = {
        key: conditioned[key] >= args.minimum_preflight_success
        for key in ("success_rate", "light_success_rate", "heavy_success_rate")
    }
    causal_checks = {
        "beats_unchanged": (conditioned["success_rate"] > preflight["unchanged"]["success_rate"]),
        "beats_selected_constant": (
            conditioned["success_rate"] > preflight["validation_selected_constant"]["success_rate"]
        ),
        "beats_swapped_mass_labels": (
            conditioned["success_rate"]
            > preflight["matched_pair_mass_labels_swapped"]["success_rate"]
        ),
    }
    passed = all(threshold_checks.values())
    candidate = calibration(selected_spec) | {
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "controller_parameter_sha256": controller_parameter_sha256(controller),
        "target_onset_seconds": args.target_onset_seconds,
        "takeover_seconds": args.takeover_seconds,
        "preflight_episodes": args.preflight_episodes,
        "preflight_seed": args.preflight_seed,
        "minimum_success_per_mass_half": args.minimum_preflight_success,
        "preflight_passed": passed,
        "counts_toward_direct_sensor_goal": False,
    }
    candidate_path = args.output_dir / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    report = {
        "experiment": "promoted-controller privileged mass-oracle calibration",
        "claim_scope": (
            "Training-only teacher validation. Exact simulator mass changes native throttle-"
            "pool bias and does not count toward the direct-sensor flight goal."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "controller_parameter_sha256": controller_parameter_sha256(controller),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "endpoint_parameterization": "intercept=(light+heavy)/2; slope=(heavy-light)/2",
            "one_half_step_refinement": True,
            "common_balanced_cases_within_each_phase": True,
            "development_validation_preflight_seeds_disjoint": True,
            "preflight_threshold_not_tuned_after_observation": True,
        },
        "development": development,
        "development_constants": development_constants,
        "validation": validation,
        "validation_constants": validation_constants,
        "selected": selected,
        "selected_constant": selected_constant,
        "preflight": preflight,
        "preflight_threshold_checks": threshold_checks,
        "causal_controls": causal_checks,
        "preflight_passed": passed,
        "candidate": candidate,
        "candidate_file": stable_path(candidate_path),
        "candidate_file_sha256": file_sha256(candidate_path),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "selected": asdict(selected_spec),
                "preflight_passed": passed,
                "threshold_checks": threshold_checks,
                "causal_controls": causal_checks,
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
