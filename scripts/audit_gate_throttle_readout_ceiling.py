#!/usr/bin/env python3
"""Test one wider legal ceiling for the native throttle readout."""

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_throttle_readout_constraint import (  # noqa: E402
    arm_summary,
    feature_arrays,
    public_fit,
    run_fit,
)
from audit_gate_throttle_routing import (  # noqa: E402
    collect_history,
    make_readout_spec,
    metric_summary,
    replay_history,
    target_scales,
)
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import make_path_spec  # noqa: E402
from search_gate_assisted_motor_es import (  # noqa: E402
    evaluate_assisted_policy_batch,
    mass_lateral_floor,
    mass_lateral_safe,
    parameter_masks,
)
from search_gate_motor_interface_es import (  # noqa: E402
    load_controller,
    motor_interface_spec,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402


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
        "--routing-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "report.json",
    )
    parser.add_argument(
        "--routing-readouts",
        type=Path,
        default=(
            REPO_ROOT / "artifacts" / "gate-throttle-routing-diagnostic-v1" / "fitted-readouts.json"
        ),
    )
    parser.add_argument(
        "--constraint-report",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-throttle-readout-constraint-diagnostic-v1"
            / "report.json"
        ),
    )
    parser.add_argument(
        "--constraint-weights",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-throttle-readout-constraint-diagnostic-v1"
            / "selected-weights.json"
        ),
    )
    parser.add_argument(
        "--replay-histories",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-routing-audit-v1" / "replay-histories.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "throttle-readout-ceiling-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--held-out-pairs", type=int, default=128)
    parser.add_argument("--flight-episodes", type=int, default=256)
    parser.add_argument("--maximum-evaluations", type=int, default=300)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=32.0)
    parser.add_argument("--normalization-floor", type=float, default=0.01)
    parser.add_argument("--maximum-group-nrmse", type=float, default=0.25)
    parser.add_argument("--minimum-constant-improvement", type=float, default=0.50)
    parser.add_argument("--projected-gradient-threshold", type=float, default=1.0e-6)
    parser.add_argument("--flight-seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--held-out-seed", type=int, default=1_043_031)
    parser.add_argument("--flight-seed", type=int, default=1_044_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.routing_report,
        args.routing_readouts,
        args.constraint_report,
        args.constraint_weights,
        args.replay_histories,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "held_out_pairs": (args.held_out_pairs, 128),
        "flight_episodes": (args.flight_episodes, 256),
        "maximum_evaluations": (args.maximum_evaluations, 300),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 32.0),
        "normalization_floor": (args.normalization_floor, 0.01),
        "maximum_group_nrmse": (args.maximum_group_nrmse, 0.25),
        "minimum_constant_improvement": (args.minimum_constant_improvement, 0.50),
        "projected_gradient_threshold": (args.projected_gradient_threshold, 1.0e-6),
        "flight_seconds": (args.flight_seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered ceiling-audit values changed: {', '.join(wrong)}")


def flight_progress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_light_gain: float,
    minimum_floor_gain: float,
    maximum_drop: float,
) -> bool:
    return bool(
        candidate["success_by_stratum"]["lower_mass"]
        >= baseline["success_by_stratum"]["lower_mass"] + minimum_light_gain
        and mass_lateral_floor(candidate) >= mass_lateral_floor(baseline) + minimum_floor_gain
        and mass_lateral_safe(baseline, candidate, maximum_drop=maximum_drop)
    )


def compact_flight(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": summary["success_rate"],
        "worst_mass_lateral_success_rate": mass_lateral_floor(summary),
        "success_by_stratum": summary["success_by_stratum"],
        "ring_collision_rate": summary["ring_collision_rate"],
        "miss_rate": summary["miss_rate"],
        "crossing_radial_mean_m": summary["crossing_radial_mean_m"],
    }


def evaluate_assisted_flights(
    source: Any,
    candidate: Any,
    *,
    episodes: int,
    seed: int,
    takeover_seconds: float,
    seconds: float,
    resolution: int,
    device: torch.device,
    hover_config: Any,
    gate_config: Any,
) -> dict[str, Any]:
    cases = diverse_matched_cases(
        episodes,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    results = {}
    outcomes = {}
    for name, controller in (("source", source), ("candidate", candidate)):
        interface = motor_interface_spec(
            controller,
            bias_scale=0.01,
            log_gain_scale=0.05,
            maximum_bias_delta=0.15,
            maximum_gain_ratio=3.0,
        )
        masks = parameter_masks(interface)
        if (int(masks["steering"].sum()), int(masks["throttle"].sum())) != (18, 6):
            raise RuntimeError("unexpected motor-interface shape during ceiling flight audit")
        zero = torch.zeros(1, len(interface.labels), device=device)
        batch = evaluate_assisted_policy_batch(
            controller,
            zero,
            interface,
            cases,
            intervention="reserve_steering",
            takeover_seconds=takeover_seconds,
            seconds=seconds,
            shaping_weight_value=0.0,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            return_outcomes=True,
            native_controller_forward=True,
        )
        results[name] = batch["summaries"][0]
        outcomes[name] = batch["outcomes"]["success"][0]
    difference = outcomes["candidate"].float() - outcomes["source"].float()
    cluster = difference.reshape(-1, 2).mean(dim=1)
    mean = float(cluster.mean())
    standard_error = float(cluster.std(unbiased=True) / np.sqrt(len(cluster)))
    return {
        "seed": seed,
        "episodes": episodes,
        "source": results["source"],
        "candidate": results["candidate"],
        "paired_overall_success_difference": {
            "mean": mean,
            "standard_error": standard_error,
            "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
            "independent_geometry_clusters": len(cluster),
        },
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    routing_report = json.loads(args.routing_report.read_text())
    routing_readouts = json.loads(args.routing_readouts.read_text())
    constraint_report = json.loads(args.constraint_report.read_text())
    constraint_weights = json.loads(args.constraint_weights.read_text())
    prerequisite_checks = {
        "graph_hash_matches": routing_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": routing_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "routing_readout_hash_matches": routing_report.get("fitted_readouts_sha256")
        == file_sha256(args.routing_readouts),
        "history_hash_matches": routing_report.get("replay_histories_sha256")
        == file_sha256(args.replay_histories),
        "constraint_report_has_converged_legal_failure": (
            constraint_report.get("classification", {}).get("fixed_sign_converged") is True
            and constraint_report.get("classification", {}).get("fixed_sign_passed") is False
        ),
        "constraint_weight_hash_matches": constraint_report.get("selected_weights_sha256")
        == file_sha256(args.constraint_weights),
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"ceiling audit prerequisite failed: {prerequisite_checks}")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    path_spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    path_nodes = torch.unique(
        torch.cat((controller.edge_pre[path_spec.edges], controller.edge_post[path_spec.edges]))
    )
    spec = make_readout_spec(controller)
    if routing_readouts.get("native_edge_indices") != spec.edges.detach().cpu().tolist():
        raise SystemExit("routing readout edge order does not match the source controller")
    source_magnitudes = (
        controller.edge_magnitude[spec.edges].detach().cpu().numpy().astype(np.float64)
    )
    previous_magnitudes = np.asarray(constraint_weights["fixed_sign"]["weights"], dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    magnitudes_path = args.output_dir / "selected-magnitudes.json"
    if report_path.exists() or magnitudes_path.exists():
        raise SystemExit("output directory contains a stale report or magnitude selection")
    seed_everything(args.held_out_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    saved = torch.load(args.replay_histories, map_location="cpu", weights_only=False)
    development_history = saved["development"]
    del saved
    development = replay_history(
        controller,
        development_history,
        path_nodes,
        spec,
        acceleration_control="live",
        device=device,
    )
    scales = target_scales(development, floor=args.normalization_floor)
    constant = float(routing_readouts["constant_residual"])
    arrays = feature_arrays(development, controller, spec, arm="fixed_sign")
    fits = [
        run_fit(
            "fixed_sign",
            start_name,
            start,
            arrays,
            development,
            spec,
            scales,
            maximum_evaluations=args.maximum_evaluations,
            maximum_magnitude=args.maximum_edge_magnitude,
            projected_gradient_threshold=args.projected_gradient_threshold,
        )
        for start_name, start in (
            ("source", source_magnitudes),
            ("converged_bound_8", previous_magnitudes),
        )
    ]
    held_history = collect_history(
        controller,
        path_nodes,
        spec,
        pairs=args.held_out_pairs,
        seed=args.held_out_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seconds=1.50,
        takeover_seconds=0.50,
    )
    held_out = replay_history(
        controller,
        held_history,
        path_nodes,
        spec,
        acceleration_control="live",
        device=device,
    )
    constant_metrics = metric_summary(
        np.full(len(held_out["target"]), constant),
        held_out,
        scales,
        constant_aggregate_rmse=None,
    )
    result = arm_summary(
        fits,
        feature_arrays(held_out, controller, spec, arm="fixed_sign"),
        held_out,
        spec,
        scales,
        constant_aggregate_rmse=constant_metrics["aggregate_equal_group_rmse"],
        maximum_group_nrmse=args.maximum_group_nrmse,
        minimum_constant_improvement=args.minimum_constant_improvement,
    )
    selected_magnitudes = result.pop("_selected_weights")
    selected_fit = next(item for item in fits if item["start"] == result["selected_start"])
    readout_passed = bool(result["pass_gate"]["passed"])
    assisted_flight = None
    flight_passed = False
    if readout_passed:
        candidate = copy.deepcopy(controller).to(device)
        with torch.no_grad():
            candidate.edge_magnitude[spec.edges] = torch.from_numpy(selected_magnitudes).to(
                device=device,
                dtype=candidate.edge_magnitude.dtype,
            )
        assisted_flight = evaluate_assisted_flights(
            controller,
            candidate,
            episodes=args.flight_episodes,
            seed=args.flight_seed,
            takeover_seconds=args.takeover_seconds,
            seconds=args.flight_seconds,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        flight_passed = flight_progress(
            assisted_flight["source"],
            assisted_flight["candidate"],
            minimum_light_gain=args.minimum_light_gain,
            minimum_floor_gain=args.minimum_floor_gain,
            maximum_drop=args.maximum_stratum_drop,
        )
        print(
            json.dumps(
                {
                    "phase": "assisted_flight",
                    "source": compact_flight(assisted_flight["source"]),
                    "candidate": compact_flight(assisted_flight["candidate"]),
                    "passed": flight_passed,
                }
            ),
            flush=True,
        )
    upper_bound_count = int(np.isclose(selected_magnitudes, args.maximum_edge_magnitude).sum())
    lower_bound_count = int(np.isclose(selected_magnitudes, 0.0).sum())
    if not readout_passed:
        interpretation = "wider_legal_ceiling_failed"
        next_step = "stop without further ceiling escalation"
    elif flight_passed:
        interpretation = "wider_legal_readout_passed_replay_and_assisted_flight"
        next_step = "design a separate native coupled-flight test; no automatic merge"
    else:
        interpretation = "wider_legal_readout_passed_replay_but_failed_closed_loop_flight"
        next_step = "stop this readout before any native merge"
    magnitudes_path.write_text(
        json.dumps(
            {
                "edge_indices": spec.edges.detach().cpu().tolist(),
                "magnitudes": selected_magnitudes.tolist(),
                "selected_start": result["selected_start"],
                "upper_bound_count": upper_bound_count,
                "lower_bound_count": lower_bound_count,
                "compiled_or_promoted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    report = {
        "method": "single-ceiling legal fixed-sign throttle readout audit",
        "claim_scope": (
            "This diagnostic raises only the fitting ceiling from 8 to 32 on the same 37 "
            "fixed-sign magnitudes. The source remains unchanged. A fitted readout is not "
            "compiled, merged, promoted, or treated as a deployable flight controller."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "routing_report": stable_path(args.routing_report),
        "routing_report_sha256": file_sha256(args.routing_report),
        "constraint_report": stable_path(args.constraint_report),
        "constraint_report_sha256": file_sha256(args.constraint_report),
        "replay_histories": stable_path(args.replay_histories),
        "replay_histories_sha256": file_sha256(args.replay_histories),
        "prerequisite_checks": prerequisite_checks,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "development_pairs": 64,
            "development_source": "identical frozen histories from routing audit v1",
            "fresh_held_out_episodes": 2 * args.held_out_pairs,
            "starts": ["source", "converged legal bound-8 fit"],
            "solver": "scipy.optimize.least_squares method=trf tr_solver=exact, FP64",
            "analytic_jacobian": True,
            "selection": "minimum development loss only",
            "held_out_evaluated_after_selection": True,
            "edge_signs_fixed": True,
            "edge_magnitude_bounds": [0.0, args.maximum_edge_magnitude],
            "no_further_ceiling_escalation_if_failed": True,
            "assisted_flight_runs_only_after_replay_pass": True,
            "assisted_flight_axis_intervention": "reserve steering after 0.5 seconds",
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "constant_baseline": constant_metrics,
        "fits": [public_fit(item) for item in fits],
        "selected_start": result["selected_start"],
        "selected_fit_converged": selected_fit["converged_under_preregistered_rule"],
        "selected_held_out": result["held_out"],
        "readout_pass_gate": result["pass_gate"],
        "selected_upper_bound_count": upper_bound_count,
        "selected_lower_bound_count": lower_bound_count,
        "selected_magnitudes": stable_path(magnitudes_path),
        "selected_magnitudes_sha256": file_sha256(magnitudes_path),
        "assisted_flight": assisted_flight,
        "assisted_flight_passed": flight_passed,
        "classification": {"interpretation": interpretation, "next_step": next_step},
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
                "readout_passed": readout_passed,
                "assisted_flight_passed": flight_passed,
                "interpretation": interpretation,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if readout_passed and flight_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
