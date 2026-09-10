#!/usr/bin/env python3
"""Audit whether correlated throttle exploration reaches useful light-mass flights."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    load_controller,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_recurrent_ppo import (  # noqa: E402
    final_success,
    initialize_outcomes,
    latent_mean,
    outcome_summary,
    update_outcomes,
)
from train_gate_recurrent_routing_ppo import assisted_steering_motor  # noqa: E402

from flydrone.gate import GateConfig, crossing_coordinates, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


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
        "--ppo-report",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-recurrent-routing-ppo-diagnostic-v1" / "report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-exploration-timescale-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--correlation-seconds", type=float, default=0.20)
    parser.add_argument("--minimum-correlated-light-success", type=float, default=0.10)
    parser.add_argument("--maximum-heavy-drop", type=float, default=0.10)
    parser.add_argument("--case-seed", type=int, default=1_050_031)
    parser.add_argument("--noise-seeds", type=int, nargs=2, default=(1_053_031, 1_054_031))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.ppo_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "episodes": (args.episodes, 256),
        "seconds": (args.seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "sigma": (args.sigma, 0.03),
        "correlation_seconds": (args.correlation_seconds, 0.20),
        "minimum_correlated_light_success": (
            args.minimum_correlated_light_success,
            0.10,
        ),
        "maximum_heavy_drop": (args.maximum_heavy_drop, 0.10),
        "case_seed": (args.case_seed, 1_050_031),
        "noise_seeds": (tuple(args.noise_seeds), (1_053_031, 1_054_031)),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered exploration-timescale values changed: {', '.join(wrong)}")


def ar1_coefficient(dt: float, correlation_seconds: float) -> float:
    if dt <= 0.0 or correlation_seconds <= 0.0:
        raise ValueError("AR(1) timing values must be positive")
    return math.exp(-dt / correlation_seconds)


def exploration_gate(
    independent: list[dict[str, Any]],
    correlated: list[dict[str, Any]],
    *,
    minimum_light_success: float,
    maximum_heavy_drop: float,
) -> dict[str, Any]:
    if len(independent) != len(correlated) or not independent:
        raise ValueError("exploration gate requires paired nonempty seed results")
    seed_checks = []
    for independent_result, correlated_result in zip(independent, correlated, strict=True):
        checks = {
            "correlated_light_success_at_least_minimum": (
                correlated_result["light_success_rate"] >= minimum_light_success
            ),
            "correlated_heavy_noninferior_to_independent": (
                correlated_result["heavy_success_rate"]
                >= independent_result["heavy_success_rate"] - maximum_heavy_drop
            ),
        }
        seed_checks.append({"checks": checks, "passed": all(checks.values())})
    return {"seed_checks": seed_checks, "passed": all(item["passed"] for item in seed_checks)}


def _conditional(values: Tensor, mask: Tensor, statistic: str) -> float | None:
    selected = values[mask]
    if not len(selected):
        return None
    if statistic == "mean":
        return float(selected.mean())
    if statistic == "mean_absolute":
        return float(selected.abs().mean())
    if statistic == "p90_absolute":
        return float(torch.quantile(selected.abs(), 0.9))
    raise ValueError(f"unknown conditional statistic: {statistic}")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "success_rate",
        "light_success_rate",
        "heavy_success_rate",
        "negative_lateral_success_rate",
        "positive_lateral_success_rate",
        "ring_collision_rate",
        "miss_rate",
        "plane_crossing_rate",
        "crossing_radial_mean_m",
        "signed_crossing_vertical_mean_m",
        "crossing_vertical_absolute_mean_m",
        "crossing_vertical_absolute_p90_m",
        "light_signed_crossing_vertical_mean_m",
        "heavy_signed_crossing_vertical_mean_m",
        "light_crossing_vertical_absolute_mean_m",
        "heavy_crossing_vertical_absolute_mean_m",
        "foreleg_throttle_range_mean",
        "foreleg_throttle_range_p90",
        "foreleg_throttle_temporal_std_mean",
        "throttle_motor_perturbation_rms",
        "foreleg_throttle_rms_difference_from_deterministic",
    )
    return {key: summary[key] for key in keys}


@torch.no_grad()
def evaluate_mode(
    controller: ConnectomeController,
    cases: BalancedCases,
    *,
    mode: str,
    seed: int,
    sigma: float,
    correlation_seconds: float,
    seconds: float,
    takeover_seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    deterministic_throttle_trace: Tensor | None = None,
) -> tuple[dict[str, Any], Tensor]:
    if mode not in {"deterministic", "independent", "correlated"}:
        raise ValueError(f"unknown exploration mode: {mode}")
    seed_everything(seed)
    device = cases.mass_scale.device
    generator = torch.Generator(device=device).manual_seed(seed)
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    rho = ar1_coefficient(hover_config.dt, correlation_seconds)
    innovation_scale = math.sqrt(1.0 - rho * rho)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    crossing_vertical = torch.full((episodes,), float("nan"), device=device)
    throttle_trace = torch.empty(steps, episodes, device=device)
    perturbation_square_sum = torch.zeros(episodes, device=device)
    correlated_noise = torch.zeros(episodes, 1, device=device)
    episode_reward = torch.zeros(episodes, device=device)
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        native_motor, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        throttle_motor = native_motor[:, 3:4]
        if mode != "deterministic":
            innovation = torch.randn(episodes, 1, generator=generator, device=device)
            if mode == "independent":
                latent_noise = sigma * innovation
            elif step == 0:
                correlated_noise = sigma * innovation
                latent_noise = correlated_noise
            else:
                correlated_noise = rho * correlated_noise + sigma * innovation_scale * innovation
                latent_noise = correlated_noise
            throttle_motor = torch.tanh(latent_mean(throttle_motor) + latent_noise)
        perturbation_square_sum += (throttle_motor - native_motor[:, 3:4]).square().squeeze(1)
        steering_motor = assisted_steering_motor(
            controller,
            native_motor,
            state,
            cases.gate,
            cases.mass_scale,
            use_reserve=step >= takeover_step,
            hover_config=hover_config,
        )
        rc, stick_state = sticks(torch.cat((steering_motor, throttle_motor), dim=1), stick_state)
        throttle_trace[step] = rc[:, 3]
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        not_crossed = outcomes["crossing_radial"].isnan()
        potential_gain, new_pass, hazard_now = update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
        newly_crossed = not_crossed & ~outcomes["crossing_radial"].isnan()
        if bool(newly_crossed.any()):
            _, _, vertical = crossing_coordinates(previous_position, state.position, cases.gate)
            crossing_vertical[newly_crossed] = vertical[newly_crossed]
        episode_reward += potential_gain + 2.0 * new_pass.float() - 2.0 * hazard_now.float()
    success, saturation_fraction = final_success(
        outcomes, step_count=steps, hover_config=hover_config
    )
    episode_reward += 10.0 * success.float()
    summary = outcome_summary(
        success,
        outcomes["passed"],
        outcomes["collision"],
        outcomes["missed"],
        outcomes["crossing_radial"],
        outcomes["maximum_progress"],
        outcomes["maximum_approach"],
        saturation_fraction,
        cases.stratum_code,
        episode_reward,
    )
    crossed = ~crossing_vertical.isnan()
    light = ~cases.stratum_code.bitwise_and(1).bool()
    throttle_range = throttle_trace.max(dim=0).values - throttle_trace.min(dim=0).values
    throttle_std = throttle_trace.std(dim=0, unbiased=False)
    summary.update(
        {
            "signed_crossing_vertical_mean_m": _conditional(crossing_vertical, crossed, "mean"),
            "crossing_vertical_absolute_mean_m": _conditional(
                crossing_vertical, crossed, "mean_absolute"
            ),
            "crossing_vertical_absolute_p90_m": _conditional(
                crossing_vertical, crossed, "p90_absolute"
            ),
            "light_signed_crossing_vertical_mean_m": _conditional(
                crossing_vertical, crossed & light, "mean"
            ),
            "heavy_signed_crossing_vertical_mean_m": _conditional(
                crossing_vertical, crossed & ~light, "mean"
            ),
            "light_crossing_vertical_absolute_mean_m": _conditional(
                crossing_vertical, crossed & light, "mean_absolute"
            ),
            "heavy_crossing_vertical_absolute_mean_m": _conditional(
                crossing_vertical, crossed & ~light, "mean_absolute"
            ),
            "foreleg_throttle_range_mean": float(throttle_range.mean()),
            "foreleg_throttle_range_p90": float(torch.quantile(throttle_range, 0.9)),
            "foreleg_throttle_temporal_std_mean": float(throttle_std.mean()),
            "throttle_motor_perturbation_rms": float(
                torch.sqrt(perturbation_square_sum.sum() / (steps * episodes))
            ),
            "foreleg_throttle_rms_difference_from_deterministic": (
                0.0
                if deterministic_throttle_trace is None
                else float(
                    torch.sqrt((throttle_trace - deterministic_throttle_trace).square().mean())
                )
            ),
        }
    )
    return compact_summary(summary), throttle_trace


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    ppo_report = json.loads(args.ppo_report.read_text())
    prerequisite_checks = {
        "ppo_graph_hash_matches": ppo_report.get("graph_sha256") == file_sha256(args.graph),
        "ppo_checkpoint_hash_matches": ppo_report.get("source_checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "ppo_stopped_at_iteration_10": ppo_report.get("stopped_at_iteration_10_gate") is True,
        "ppo_development_failed": ppo_report.get("development_passed") is False,
        "ppo_fresh_final_unused": ppo_report.get("fresh_final_consumed") is False,
        "ppo_did_not_promote": ppo_report.get("compiled_or_promoted") is False,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"exploration-audit prerequisite failed: {prerequisite_checks}")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit("output directory contains a stale report")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    cases = diverse_matched_cases(
        args.episodes,
        seed=args.case_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    deterministic, deterministic_trace = evaluate_mode(
        controller,
        cases,
        mode="deterministic",
        seed=0,
        sigma=args.sigma,
        correlation_seconds=args.correlation_seconds,
        seconds=args.seconds,
        takeover_seconds=args.takeover_seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    print(json.dumps({"mode": "deterministic", **deterministic}), flush=True)
    results: dict[str, list[dict[str, Any]]] = {"independent": [], "correlated": []}
    for mode in ("independent", "correlated"):
        for seed in args.noise_seeds:
            summary, _trace = evaluate_mode(
                controller,
                cases,
                mode=mode,
                seed=seed,
                sigma=args.sigma,
                correlation_seconds=args.correlation_seconds,
                seconds=args.seconds,
                takeover_seconds=args.takeover_seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
                deterministic_throttle_trace=deterministic_trace,
            )
            result = {"seed": seed, **summary}
            results[mode].append(result)
            print(json.dumps({"mode": mode, **result}), flush=True)
    gate = exploration_gate(
        results["independent"],
        results["correlated"],
        minimum_light_success=args.minimum_correlated_light_success,
        maximum_heavy_drop=args.maximum_heavy_drop,
    )
    report = {
        "method": "frozen-source scalar-throttle exploration-timescale audit",
        "claim_scope": (
            "This audit changes no actor parameter. Independent and correlated noise are "
            "diagnostic training-time interventions, not actor inputs or actor memory."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "ppo_report": stable_path(args.ppo_report),
        "ppo_report_sha256": file_sha256(args.ppo_report),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "ar1_coefficient": ar1_coefficient(hover_config.dt, args.correlation_seconds),
            "correlated_stationary_latent_standard_deviation": args.sigma,
            "same_cases_all_conditions": True,
            "same_innovations_for_matched_noisy_seeds": True,
            "source_steering_before_takeover": True,
            "reserve_steering_after_takeover": True,
            "native_source_throttle_throughout": True,
            "actor_parameter_updates": 0,
            "external_actor_memory": False,
            "measurement_note": (
                "Foreleg throttle RMS differences are complete closed-loop effects, not "
                "isolated actuator filtering; trajectories diverge and traces include "
                "post-classification simulation steps."
            ),
        },
        "deterministic": deterministic,
        "noisy_results": results,
        "continuation_gate": gate,
        "classification": (
            "correlated_exploration_prerequisite_passed"
            if gate["passed"]
            else "correlated_exploration_prerequisite_failed_pause_family"
        ),
        "source_preserved": True,
        "compiled_or_promoted": False,
        "candidate_checkpoint": None,
        "goal_passed": False,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "continuation_gate_passed": gate["passed"],
                "classification": report["classification"],
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
