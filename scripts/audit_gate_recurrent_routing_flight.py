#!/usr/bin/env python3
"""Evaluate stopped recurrent-routing checkpoints in assisted complete flights."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_throttle_routing import make_readout_spec  # noqa: E402
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
from train_gate_recurrent_routing import make_recurrent_routing_spec  # noqa: E402


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
        "--continuation-report",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-continuation-diagnostic-v1"
            / "report.json"
        ),
    )
    parser.add_argument(
        "--selected-treatment",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-continuation-diagnostic-v1"
            / "selected-magnitudes.json"
        ),
    )
    parser.add_argument(
        "--arm-magnitudes",
        type=Path,
        default=(
            REPO_ROOT
            / "artifacts"
            / "gate-recurrent-routing-continuation-diagnostic-v1"
            / "arm-magnitudes.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "recurrent-routing-flight-audit-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--positive-control-minimum-mass-success", type=float, default=0.90)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--recovery-maximum-difference", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1_048_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.graph,
        args.checkpoint,
        args.continuation_report,
        args.selected_treatment,
        args.arm_magnitudes,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "episodes": (args.episodes, 256),
        "seconds": (args.seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "positive_control_minimum_mass_success": (
            args.positive_control_minimum_mass_success,
            0.90,
        ),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
        "recovery_maximum_difference": (args.recovery_maximum_difference, 1.0e-5),
        "seed": (args.seed, 1_048_031),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered flight-audit values changed: {', '.join(wrong)}")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": summary["success_rate"],
        "worst_mass_lateral_success_rate": mass_lateral_floor(summary),
        "success_by_stratum": summary["success_by_stratum"],
        "ring_collision_rate": summary["ring_collision_rate"],
        "miss_rate": summary["miss_rate"],
        "plane_crossing_rate": summary["plane_crossing_rate"],
        "crossing_radial_mean_m": summary["crossing_radial_mean_m"],
        "crossing_radial_p90_m": summary["crossing_radial_p90_m"],
    }


def paired_success_difference(candidate: Tensor, source: Tensor) -> dict[str, Any]:
    difference = candidate.float() - source.float()
    clusters = difference.reshape(-1, 2).mean(dim=1)
    mean = float(clusters.mean())
    standard_error = float(clusters.std(unbiased=True) / math.sqrt(len(clusters)))
    return {
        "mean": mean,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "independent_geometry_clusters": len(clusters),
        "episodes_per_cluster": 2,
    }


def positive_control_passes(summary: dict[str, Any], *, minimum: float) -> bool:
    strata = summary["success_by_stratum"]
    return bool(strata["lower_mass"] >= minimum and strata["higher_mass"] >= minimum)


def candidate_progress(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_light_gain: float,
    minimum_floor_gain: float,
    maximum_drop: float,
) -> bool:
    return bool(
        candidate["success_by_stratum"]["lower_mass"]
        >= source["success_by_stratum"]["lower_mass"] + minimum_light_gain
        and mass_lateral_floor(candidate) >= mass_lateral_floor(source) + minimum_floor_gain
        and mass_lateral_safe(source, candidate, maximum_drop=maximum_drop)
    )


@torch.no_grad()
def evaluate_controller(
    controller: Any,
    cases: Any,
    *,
    intervention: str,
    resolution: int,
    hover_config: Any,
    gate_config: Any,
    takeover_seconds: float,
    seconds: float,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor]:
    interface = motor_interface_spec(
        controller,
        bias_scale=0.01,
        log_gain_scale=0.05,
        maximum_bias_delta=0.15,
        maximum_gain_ratio=3.0,
    )
    masks = parameter_masks(interface)
    if (int(masks["steering"].sum()), int(masks["throttle"].sum())) != (18, 6):
        raise RuntimeError("unexpected motor interface in flight audit")
    result = evaluate_assisted_policy_batch(
        controller,
        torch.zeros(1, len(interface.labels), device=device),
        interface,
        cases,
        intervention=intervention,
        takeover_seconds=takeover_seconds,
        seconds=seconds,
        shaping_weight_value=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        return_outcomes=True,
        native_controller_forward=True,
    )
    return result["summaries"][0], result["outcomes"]["success"][0]


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    continuation_report = json.loads(args.continuation_report.read_text())
    selected_treatment = json.loads(args.selected_treatment.read_text())
    arm_magnitudes = json.loads(args.arm_magnitudes.read_text())
    prerequisite_checks = {
        "graph_hash_matches": continuation_report.get("graph_sha256") == file_sha256(args.graph),
        "checkpoint_hash_matches": continuation_report.get("checkpoint_sha256")
        == file_sha256(args.checkpoint),
        "both_development_arms_failed": all(
            value is False
            for value in continuation_report.get("arm_development_passes", {}).values()
        ),
        "fresh_replay_was_not_run": continuation_report.get("final_fresh_replay") is None,
        "assisted_flight_was_not_run": continuation_report.get("assisted_flight") is None,
        "source_was_preserved": continuation_report.get("source_preserved") is True,
    }
    if not all(prerequisite_checks.values()):
        raise SystemExit(f"flight audit prerequisite failed: {prerequisite_checks}")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    path_spec = make_path_spec(
        controller, args.graph, maximum_hops=4, floor_quantile=0.25, maximum_magnitude=8.0
    )
    readout_spec = make_readout_spec(controller)
    routing_spec = make_recurrent_routing_spec(controller, path_spec, readout_spec)
    edge_indices = routing_spec.selected_edges.detach().cpu().tolist()
    if arm_magnitudes.get("selected_edge_indices") != edge_indices:
        raise SystemExit("recovered arm edge order differs from reconstructed mask")
    if selected_treatment.get("selected_edge_indices") != edge_indices:
        raise SystemExit("selected treatment edge order differs from reconstructed mask")
    recovered_treatment = np.asarray(
        arm_magnitudes["arms"]["treatment_early_fourfold"]["magnitudes"]
    )
    original_treatment = np.asarray(selected_treatment["selected_magnitudes"])
    recovery_difference = float(np.max(np.abs(recovered_treatment - original_treatment)))
    if recovery_difference > args.recovery_maximum_difference:
        raise SystemExit("deterministically recovered treatment vector differs from original")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit("output directory contains a stale report")
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    cases = diverse_matched_cases(
        args.episodes,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    controllers = {"source": controller}
    for name in ("control_equal_window", "treatment_early_fourfold"):
        candidate = copy.deepcopy(controller).to(device)
        values = torch.tensor(
            arm_magnitudes["arms"][name]["magnitudes"],
            device=device,
            dtype=candidate.edge_magnitude.dtype,
        )
        with torch.no_grad():
            candidate.edge_magnitude[routing_spec.selected_edges] = values
        controllers[name] = candidate
    summaries = {}
    outcomes = {}
    for name, candidate in controllers.items():
        summary, success = evaluate_controller(
            candidate,
            cases,
            intervention="reserve_steering",
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            takeover_seconds=args.takeover_seconds,
            seconds=args.seconds,
            device=device,
        )
        summaries[name] = summary
        outcomes[name] = success
        print(
            json.dumps(
                {"phase": "assisted_flight", "controller": name, **compact_summary(summary)}
            ),
            flush=True,
        )
    reserve_summary, reserve_success = evaluate_controller(
        controller,
        cases,
        intervention="reserve_all",
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        takeover_seconds=args.takeover_seconds,
        seconds=args.seconds,
        device=device,
    )
    summaries["full_reserve_positive_control"] = reserve_summary
    outcomes["full_reserve_positive_control"] = reserve_success
    print(
        json.dumps(
            {
                "phase": "positive_control",
                "controller": "full_reserve_positive_control",
                **compact_summary(reserve_summary),
            }
        ),
        flush=True,
    )
    positive_passed = positive_control_passes(
        reserve_summary, minimum=args.positive_control_minimum_mass_success
    )
    progress = {
        name: candidate_progress(
            summaries["source"],
            summaries[name],
            minimum_light_gain=args.minimum_light_gain,
            minimum_floor_gain=args.minimum_floor_gain,
            maximum_drop=args.maximum_stratum_drop,
        )
        for name in ("control_equal_window", "treatment_early_fourfold")
    }
    paired = {
        name: paired_success_difference(outcomes[name], outcomes["source"])
        for name in (
            "control_equal_window",
            "treatment_early_fourfold",
            "full_reserve_positive_control",
        )
    }
    if not positive_passed:
        interpretation = "positive_control_failed_stop_interpretation"
    elif any(progress.values()):
        interpretation = "at_least_one_stopped_candidate_has_meaningful_flight_progress"
    else:
        interpretation = "no_stopped_candidate_has_meaningful_flight_progress"
    report = {
        "method": "read-only stopped-checkpoint assisted-flight diagnostic",
        "claim_scope": (
            "This diagnostic evaluates source, two stopped recurrent-routing vectors, and a "
            "full-reserve positive control on identical fresh cases. It performs no update, "
            "selection, interpolation, merge, compilation, or promotion."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "continuation_report": stable_path(args.continuation_report),
        "continuation_report_sha256": file_sha256(args.continuation_report),
        "arm_magnitudes": stable_path(args.arm_magnitudes),
        "arm_magnitudes_sha256": file_sha256(args.arm_magnitudes),
        "prerequisite_checks": prerequisite_checks,
        "recovered_treatment_maximum_absolute_difference": recovery_difference,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "case_structure": "128 distinct geometries with adjacent light/heavy episodes",
            "candidate_intervention": "reserve steering only after 0.5 seconds",
            "positive_control_intervention": "all reserve axes after source 0.5-second prefix",
            "native_candidate_forward_from_initialization": True,
            "optimizer_updates_performed": 0,
            "checkpoint_selection_performed": False,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "summaries": {name: compact_summary(value) for name, value in summaries.items()},
        "geometry_paired_success_differences_against_source": paired,
        "positive_control_passed": positive_passed,
        "candidate_progress": progress,
        "classification": {
            "interpretation": interpretation,
            "next_step": (
                "pause this training family if neither candidate progresses; redesign around "
                "closed-loop task outcomes and state-distribution mismatch"
            ),
        },
        "fresh_cases_consumed": True,
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
                "positive_control_passed": positive_passed,
                "candidate_progress": progress,
                "interpretation": interpretation,
                "compiled_or_promoted": False,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if positive_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
