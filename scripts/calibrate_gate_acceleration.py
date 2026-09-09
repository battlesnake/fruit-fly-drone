#!/usr/bin/env python3
"""Calibrate two anatomy-internal throttle parameters by complete gate flights."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
)

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v1" / "controller.pt",
    )
    parser.add_argument(
        "--baseline-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-v1" / "connectome.npz",
        help="Graph whose nodes define the frozen pre-accelerometer controller.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "acceleration-calibration",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bias-values", type=float, nargs="+", default=(-0.10, -0.05, -0.02, 0.0, 0.02)
    )
    parser.add_argument("--gain-values", type=float, nargs="+", default=(0.0, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--development-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--development-seed", type=int, default=20_031)
    parser.add_argument("--validation-seed", type=int, default=30_031)
    parser.add_argument("--final-seed", type=int, default=40_031)
    return parser.parse_args()


def candidate_key(bias: float, gain: float) -> str:
    return f"bias={bias:+.6g},gain={gain:.6g}"


def selection_key(metrics: dict[str, Any]) -> tuple[float, float]:
    strata = metrics["success_by_stratum"]
    balanced_success = metrics["success_rate"] + min(strata["lower_mass"], strata["higher_mass"])
    radial_p90 = metrics["crossing_radial_quantiles_m"]["90"]
    return balanced_success, -radial_p90


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    crossing = metrics["crossing_error_components_m"]
    strata = metrics["success_by_stratum"]
    return {
        "success_rate": metrics["success_rate"],
        "lower_mass_success_rate": strata["lower_mass"],
        "higher_mass_success_rate": strata["higher_mass"],
        "negative_offset_success_rate": strata["negative_lateral_offset"],
        "positive_offset_success_rate": strata["positive_lateral_offset"],
        "ring_collision_rate": metrics["ring_collision_rate"],
        "miss_rate": metrics["miss_rate"],
        "radial_error_p90_m": metrics["crossing_radial_quantiles_m"]["90"],
        "lateral_absolute_error_mean_m": crossing["lateral_absolute_mean"],
        "vertical_absolute_error_mean_m": crossing["vertical_absolute_mean"],
        "vertical_signed_error_mean_m": crossing["vertical_signed_mean"],
        "balanced_selection_score": selection_key(metrics)[0],
    }


def calibration_indices(
    controller: ConnectomeController,
    graph_path: Path,
    baseline_graph_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(graph_path) as graph, np.load(baseline_graph_path) as baseline_graph:
        node_ids = graph["node_ids"]
        baseline_ids = baseline_graph["node_ids"]
        edge_pre = graph["edge_pre"]
        edge_post = graph["edge_post"]
        pre_is_new = ~np.isin(node_ids[edge_pre], baseline_ids)
        post_is_old = np.isin(node_ids[edge_post], baseline_ids)
        boundary = np.nonzero(pre_is_new & post_is_old)[0]

    if not len(boundary):
        raise SystemExit("no new-to-old acceleration boundary edges found")
    if not controller.uses_accelerometer:
        raise SystemExit("the destination graph has no accelerometer interface")
    throttle_start = int(controller.pool_offsets[6].item())
    throttle_middle = int(controller.pool_offsets[7].item())
    throttle_end = int(controller.pool_offsets[8].item())
    positive = controller.pool_indices[throttle_start:throttle_middle]
    negative = controller.pool_indices[throttle_middle:throttle_end]
    return (
        positive,
        negative,
        torch.from_numpy(boundary).to(device=controller.edge_magnitude.device),
    )


def apply_candidate(
    controller: ConnectomeController,
    base_state: dict[str, torch.Tensor],
    positive_throttle_nodes: torch.Tensor,
    negative_throttle_nodes: torch.Tensor,
    acceleration_boundary_edges: torch.Tensor,
    *,
    bias: float,
    gain: float,
) -> None:
    controller.load_state_dict(base_state)
    with torch.no_grad():
        controller.bias[positive_throttle_nodes] += bias
        controller.bias[negative_throttle_nodes] -= bias
        controller.edge_magnitude[acceleration_boundary_edges] *= gain
    controller.eval()


def run_candidate(
    controller: ConnectomeController,
    base_state: dict[str, torch.Tensor],
    positive_throttle_nodes: torch.Tensor,
    negative_throttle_nodes: torch.Tensor,
    acceleration_boundary_edges: torch.Tensor,
    *,
    bias: float,
    gain: float,
    episodes: int,
    seconds: float,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    frozen_visual: bool = False,
    frozen_acceleration: bool = False,
    swapped_acceleration: bool = False,
) -> dict[str, Any]:
    apply_candidate(
        controller,
        base_state,
        positive_throttle_nodes,
        negative_throttle_nodes,
        acceleration_boundary_edges,
        bias=bias,
        gain=gain,
    )
    return evaluate_gate(
        controller,
        episodes=episodes,
        seconds=seconds,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=seed,
        frozen_visual=frozen_visual,
        frozen_acceleration=frozen_acceleration,
        swapped_acceleration=swapped_acceleration,
    )


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.baseline_graph):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not args.bias_values or not args.gain_values:
        raise SystemExit("the bias and gain grids must be nonempty")
    if any(gain < 0.0 for gain in args.gain_values):
        raise SystemExit("acceleration pathway gains must be nonnegative")
    if (
        min(
            args.development_episodes,
            args.validation_episodes,
            args.final_episodes,
            args.top_candidates,
        )
        <= 0
    ):
        raise SystemExit("episode counts and --top-candidates must be positive")


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
        raise SystemExit("checkpoint retinal orientation does not match the evaluator")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    resolution = int(checkpoint["image_resolution"])
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    positive, negative, boundary = calibration_indices(controller, args.graph, args.baseline_graph)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    development: dict[str, Any] = {}
    candidate_values = [
        (float(bias), float(gain)) for bias in args.bias_values for gain in args.gain_values
    ]
    for index, (bias, gain) in enumerate(candidate_values, start=1):
        metrics = run_candidate(
            controller,
            base_state,
            positive,
            negative,
            boundary,
            bias=bias,
            gain=gain,
            episodes=args.development_episodes,
            seconds=args.seconds,
            seed=args.development_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        key = candidate_key(bias, gain)
        development[key] = {
            "bias": bias,
            "acceleration_boundary_gain": gain,
            "metrics": compact_metrics(metrics),
        }
        print(
            json.dumps(
                {
                    "phase": "development",
                    "candidate": index,
                    "candidate_count": len(candidate_values),
                    "key": key,
                    **development[key]["metrics"],
                }
            ),
            flush=True,
        )

    ranked_development = sorted(
        development,
        key=lambda key: (
            development[key]["metrics"]["balanced_selection_score"],
            -development[key]["metrics"]["radial_error_p90_m"],
        ),
        reverse=True,
    )
    finalists = ranked_development[: min(args.top_candidates, len(ranked_development))]
    validation: dict[str, Any] = {}
    for key in finalists:
        candidate = development[key]
        metrics = run_candidate(
            controller,
            base_state,
            positive,
            negative,
            boundary,
            bias=candidate["bias"],
            gain=candidate["acceleration_boundary_gain"],
            episodes=args.validation_episodes,
            seconds=args.seconds,
            seed=args.validation_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        validation[key] = {
            **candidate,
            "metrics": compact_metrics(metrics),
        }
        print(
            json.dumps({"phase": "validation", "key": key, **validation[key]["metrics"]}),
            flush=True,
        )

    winner_key = max(
        validation,
        key=lambda key: (
            validation[key]["metrics"]["balanced_selection_score"],
            -validation[key]["metrics"]["radial_error_p90_m"],
        ),
    )
    winner = validation[winner_key]
    final_specs = {
        "baseline": (0.0, 1.0, {}),
        "winner": (winner["bias"], winner["acceleration_boundary_gain"], {}),
        "winner_bias_acceleration_disconnected": (winner["bias"], 0.0, {}),
        "winner_gain_without_bias": (0.0, winner["acceleration_boundary_gain"], {}),
        "winner_constant_1g": (
            winner["bias"],
            winner["acceleration_boundary_gain"],
            {"frozen_acceleration": True},
        ),
        "winner_mass_rank_swapped_trace": (
            winner["bias"],
            winner["acceleration_boundary_gain"],
            {"swapped_acceleration": True},
        ),
        "winner_frozen_first_frame": (
            winner["bias"],
            winner["acceleration_boundary_gain"],
            {"frozen_visual": True},
        ),
    }
    final: dict[str, Any] = {}
    for name, (bias, gain, controls) in final_specs.items():
        metrics = run_candidate(
            controller,
            base_state,
            positive,
            negative,
            boundary,
            bias=bias,
            gain=gain,
            episodes=args.final_episodes,
            seconds=args.seconds,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            **controls,
        )
        final[name] = {
            "bias": bias,
            "acceleration_boundary_gain": gain,
            "metrics": metrics,
        }
        print(json.dumps({"phase": "final", "name": name, **compact_metrics(metrics)}), flush=True)

    apply_candidate(
        controller,
        base_state,
        positive,
        negative,
        boundary,
        bias=winner["bias"],
        gain=winner["acceleration_boundary_gain"],
    )
    calibrated_checkpoint = copy.deepcopy(checkpoint)
    calibrated_checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    calibrated_checkpoint["source_checkpoint_sha256"] = file_sha256(args.checkpoint)
    calibrated_checkpoint["calibration"] = {
        "method": "complete-flight two-parameter grid search",
        "throttle_motor_bias": winner["bias"],
        "acceleration_boundary_gain": winner["acceleration_boundary_gain"],
        "new_to_old_boundary_edges": int(boundary.numel()),
        "selection": "success_rate + min(lower_mass_success, higher_mass_success)",
        "development_seed": args.development_seed,
        "validation_seed": args.validation_seed,
        "final_seed": args.final_seed,
    }
    checkpoint_path = args.output_dir / "controller.pt"
    torch.save(calibrated_checkpoint, checkpoint_path)
    report = {
        "method": "complete-flight two-parameter grid search",
        "deployed_runtime_parameters_added": 0,
        "all_changes_folded_into_connectome_parameters": True,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "search": {
            "bias_values": args.bias_values,
            "acceleration_boundary_gain_values": args.gain_values,
            "new_to_old_boundary_edges": int(boundary.numel()),
            "positive_throttle_motor_neurons": int(positive.numel()),
            "negative_throttle_motor_neurons": int(negative.numel()),
            "development_episodes": args.development_episodes,
            "validation_episodes": args.validation_episodes,
            "final_episodes": args.final_episodes,
            "seconds": args.seconds,
            "development_seed": args.development_seed,
            "validation_seed": args.validation_seed,
            "final_seed": args.final_seed,
        },
        "development": development,
        "development_ranking": ranked_development,
        "validation": validation,
        "winner": {"key": winner_key, **winner},
        "final": final,
        "goal_passed": final["winner"]["metrics"]["goal_pass"],
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"report": str(report_path), "winner": winner_key, "goal_passed": report["goal_passed"]}
        ),
        flush=True,
    )
    return 0 if report["goal_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
