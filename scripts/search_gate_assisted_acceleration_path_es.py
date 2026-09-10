#!/usr/bin/env python3
"""Search recurrent acceleration-to-throttle edges with reserve steering assistance."""

from __future__ import annotations

import argparse
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

from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    PathSpec,
    evaluate_policy_batch,
    make_path_spec,
    objective,
)
from search_gate_assisted_motor_es import parameter_masks  # noqa: E402
from search_gate_motor_interface_es import (  # noqa: E402
    load_controller,
    motor_interface_spec,
    shaping_weight,
    stable_path,
    vector_sha256,
)
from train_gate import file_sha256, seed_everything  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

MASS_LATERAL_KEYS = (
    "light_success_rate",
    "heavy_success_rate",
    "negative_lateral_success_rate",
    "positive_lateral_success_rate",
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
        "--assisted-motor-report",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-assisted-motor-es-diagnostic-v1" / "report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "assisted-acceleration-path-es-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-hops", type=int, default=4)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--checkpoint-generation", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--antithetic-directions", type=int, default=32)
    parser.add_argument("--development-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--maximum-update-sigma-fraction", type=float, default=0.25)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.25)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--heavy-reward-penalty", type=float, default=2.0)
    parser.add_argument("--minimum-light-gain", type=float, default=0.10)
    parser.add_argument("--minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--development-seed", type=int, default=1_036_031)
    parser.add_argument("--validation-seed", type=int, default=1_037_031)
    parser.add_argument("--final-seed", type=int, default=1_038_031)
    parser.add_argument("--search-seed", type=int, default=1_039_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.assisted_motor_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "maximum_hops": (args.maximum_hops, 4),
        "generations": (args.generations, 40),
        "checkpoint_generation": (args.checkpoint_generation, 20),
        "validation_interval": (args.validation_interval, 10),
        "antithetic_directions": (args.antithetic_directions, 32),
        "development_episodes": (args.development_episodes, 32),
        "validation_episodes": (args.validation_episodes, 256),
        "final_episodes": (args.final_episodes, 1024),
        "seconds": (args.seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
        "top_candidates": (args.top_candidates, 3),
        "sigma": (args.sigma, 0.03),
        "sigma_decay": (args.sigma_decay, 0.99),
        "learning_rate": (args.learning_rate, 0.012),
        "maximum_update_sigma_fraction": (args.maximum_update_sigma_fraction, 0.25),
        "scale_floor_quantile": (args.scale_floor_quantile, 0.25),
        "maximum_edge_magnitude": (args.maximum_edge_magnitude, 8.0),
        "heavy_reward_penalty": (args.heavy_reward_penalty, 2.0),
        "minimum_light_gain": (args.minimum_light_gain, 0.10),
        "minimum_floor_gain": (args.minimum_floor_gain, 0.10),
        "maximum_stratum_drop": (args.maximum_stratum_drop, 0.05),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered assisted-path values changed: {', '.join(wrong)}")
    positive = (
        args.top_candidates,
        args.sigma_decay,
        args.scale_floor_quantile,
        args.maximum_edge_magnitude,
        args.heavy_reward_penalty,
        args.minimum_light_gain,
        args.minimum_floor_gain,
        args.maximum_stratum_drop,
    )
    if min(positive) <= 0.0 or args.sigma_decay > 1.0:
        raise SystemExit("invalid assisted-path search thresholds")
    if args.scale_floor_quantile > 1.0:
        raise SystemExit("scale floor quantile must not exceed one")
    for name in ("development_episodes", "validation_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")


def mass_lateral_floor(summary: dict[str, Any]) -> float:
    return min(float(summary[key]) for key in MASS_LATERAL_KEYS)


def mass_lateral_safe(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_drop: float,
) -> bool:
    return all(
        float(candidate[key]) >= float(baseline[key]) - maximum_drop for key in MASS_LATERAL_KEYS
    )


def performance_qualified(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_light_gain: float,
    minimum_floor_gain: float,
    maximum_drop: float,
) -> bool:
    return bool(
        candidate["light_success_rate"] >= baseline["light_success_rate"] + minimum_light_gain
        and mass_lateral_floor(candidate) >= mass_lateral_floor(baseline) + minimum_floor_gain
        and mass_lateral_safe(baseline, candidate, maximum_drop=maximum_drop)
    )


def candidate_rank(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        mass_lateral_floor(summary),
        float(summary["light_success_rate"]),
        float(summary["success_rate"]),
        float(summary["heavy_success_rate"]),
    )


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: summary[key]
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
    } | {"worst_mass_lateral_success_rate": mass_lateral_floor(summary)}


def clustered_contrast_interval(contrast: Tensor, *, cluster_size: int = 2) -> dict[str, Any]:
    if contrast.ndim != 1 or len(contrast) % cluster_size:
        raise ValueError("contrast must contain complete one-dimensional geometry clusters")
    clusters = contrast.float().reshape(-1, cluster_size).mean(dim=1)
    mean = float(clusters.mean())
    standard_error = float(clusters.std(unbiased=True) / math.sqrt(len(clusters)))
    return {
        "method": "normal approximation clustered by matched geometry",
        "mean": mean,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "episodes": len(contrast),
        "independent_geometry_clusters": len(clusters),
        "episodes_per_cluster": cluster_size,
    }


def archive_rank(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (float(item["objective"]), *candidate_rank(item["development"])[:3])


def nominate_finalists(
    archive: dict[str, dict[str, Any]],
    *,
    required_keys: tuple[str, ...],
    limit: int,
) -> list[str]:
    if any(key not in archive for key in required_keys):
        raise KeyError("required validation nominee is absent from the archive")
    ranked = sorted(archive, key=lambda key: archive_rank(archive[key]), reverse=True)
    finalists: list[str] = []
    for key in (*required_keys, *ranked):
        if key not in finalists:
            finalists.append(key)
        if len(finalists) == min(limit, len(archive)):
            break
    return finalists


def validate_archive(
    controller: ConnectomeController,
    zero: Tensor,
    spec: PathSpec,
    archive: dict[str, dict[str, Any]],
    *,
    required_keys: tuple[str, ...],
    generation: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    finalists = nominate_finalists(
        archive,
        required_keys=required_keys,
        limit=args.top_candidates,
    )
    vectors = torch.stack([zero, *(archive[key]["vector"].to(device) for key in finalists)])
    cases = diverse_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    batch = evaluate_policy_batch(
        controller,
        vectors,
        spec,
        cases,
        seconds=args.seconds,
        reward_shaping_weight=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        axis_intervention="reserve_steering",
        takeover_seconds=args.takeover_seconds,
    )
    baseline = batch["summaries"][0]
    candidates: dict[str, Any] = {}
    for key, summary in zip(finalists, batch["summaries"][1:], strict=True):
        qualified = performance_qualified(
            baseline,
            summary,
            minimum_light_gain=args.minimum_light_gain,
            minimum_floor_gain=args.minimum_floor_gain,
            maximum_drop=args.maximum_stratum_drop,
        )
        candidates[key] = {
            "source": {name: value for name, value in archive[key].items() if name != "vector"},
            "metrics": summary,
            "mass_lateral_safe": mass_lateral_safe(
                baseline,
                summary,
                maximum_drop=args.maximum_stratum_drop,
            ),
            "performance_qualified": qualified,
        }
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "generation": generation,
                    "candidate": key,
                    "performance_qualified": qualified,
                    **compact(summary),
                }
            ),
            flush=True,
        )
    return {
        "generation": generation,
        "validation_seed": args.validation_seed,
        "baseline": baseline,
        "required_nominees": list(required_keys),
        "candidates": candidates,
    }


def select_validated_candidate(
    baseline: dict[str, Any],
    validated: dict[str, dict[str, Any]],
    *,
    maximum_drop: float,
) -> tuple[str | None, bool]:
    safe = {
        key: item
        for key, item in validated.items()
        if mass_lateral_safe(baseline, item["metrics"], maximum_drop=maximum_drop)
    }
    qualified = {key: item for key, item in safe.items() if item["performance_qualified"]}
    pool = qualified or safe
    selected = max(pool, key=lambda key: candidate_rank(pool[key]["metrics"])) if pool else None
    return selected, bool(selected and selected in qualified)


def verify_assisted_motor_report(
    report_path: Path,
    *,
    graph_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], Tensor]:
    report = json.loads(report_path.read_text())
    steering = report.get("steering_stage", {})
    checks = {
        "graph_hash_matches": report.get("graph_sha256") == file_sha256(graph_path),
        "checkpoint_hash_matches": report.get("checkpoint_sha256") == file_sha256(checkpoint_path),
        "steering_checkpoint_progress_passed": steering.get("checkpoint_progress_passed") is True,
        "steering_selection_qualified": steering.get("selected_progress_qualified") is True,
        "steering_search_completed": steering.get("generations_completed")
        == steering.get("generations_budget")
        == 30,
    }
    if not all(checks.values()):
        raise SystemExit(f"assisted motor diagnostic prerequisite failed: {checks}")
    vector = torch.tensor(steering["selected_vector"], dtype=torch.float32)
    if vector_sha256(vector) != steering.get("selected_vector_sha256"):
        raise SystemExit("assisted steering vector hash mismatch")
    return report, vector


def edge_overlap_audit(
    controller: ConnectomeController,
    path_spec: PathSpec,
    assisted_report: dict[str, Any],
) -> dict[str, Any]:
    protocol = assisted_report["protocol"]
    interface = motor_interface_spec(
        controller,
        bias_scale=float(protocol["bias_perturbation_scale"]),
        log_gain_scale=float(protocol["log_gain_perturbation_scale"]),
        maximum_bias_delta=float(protocol["maximum_bias_delta"]),
        maximum_gain_ratio=float(protocol["maximum_gain_ratio"]),
    )
    masks = parameter_masks(interface)
    steering_edges = interface.gain_edges[masks["steering"][interface.gain_parameter]]
    throttle_edges = interface.gain_edges[masks["throttle"][interface.gain_parameter]]
    path_edges = path_spec.edges
    steering_overlap = path_edges[torch.isin(path_edges, steering_edges)]
    throttle_overlap = path_edges[torch.isin(path_edges, throttle_edges)]
    involved_nodes = torch.unique(
        torch.cat((controller.edge_pre[path_edges], controller.edge_post[path_edges]))
    )
    return {
        "path_edge_count": len(path_edges),
        "path_involved_node_count": len(involved_nodes),
        "steering_output_gain_edge_count": len(steering_edges),
        "steering_output_gain_overlap_count": len(steering_overlap),
        "steering_output_gain_overlap_indices": steering_overlap.detach().cpu().tolist(),
        "throttle_output_gain_edge_count": len(throttle_edges),
        "throttle_output_gain_overlap_count": len(throttle_overlap),
        "throttle_output_gain_overlap_indices": throttle_overlap.detach().cpu().tolist(),
        "direct_parameter_disjointness_passed": len(steering_overlap) == 0,
        "caveat": (
            "Disjoint parameters do not imply disjoint behavior: recurrent path edges may "
            "alter activity that later reaches steering motor pools."
        ),
    }


def evaluate_final_controls(
    controller: ConnectomeController,
    zero: Tensor,
    winner: Tensor,
    spec: PathSpec,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    cases = diverse_matched_cases(
        args.final_episodes,
        seed=args.final_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    results: dict[str, Any] = {}
    for control in ("live", "constant_1g", "pair_swapped"):
        results[control] = evaluate_policy_batch(
            controller,
            torch.stack((zero, winner)),
            spec,
            cases,
            seconds=args.seconds,
            reward_shaping_weight=0.0,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            acceleration_control=control,
            axis_intervention="reserve_steering",
            takeover_seconds=args.takeover_seconds,
            return_outcomes=True,
        )
        print(
            json.dumps(
                {
                    "phase": "final_control",
                    "control": control,
                    "source": compact(results[control]["summaries"][0]),
                    "candidate": compact(results[control]["summaries"][1]),
                }
            ),
            flush=True,
        )
    comparisons: dict[str, Any] = {"candidate_vs_source": {}}
    for control, result in results.items():
        success = result["outcomes"]["success"]
        comparisons["candidate_vs_source"][control] = clustered_contrast_interval(
            success[1].float() - success[0].float()
        )
    live = results["live"]["outcomes"]["success"].float()
    comparisons["acceleration_controls"] = {}
    for control in ("constant_1g", "pair_swapped"):
        controlled = results[control]["outcomes"]["success"].float()
        comparisons["acceleration_controls"][control] = {
            "candidate_live_advantage": clustered_contrast_interval(live[1] - controlled[1]),
            "source_live_advantage": clustered_contrast_interval(live[0] - controlled[0]),
            "difference_in_differences": clustered_contrast_interval(
                (live[1] - live[0]) - (controlled[1] - controlled[0])
            ),
            "mean_absolute_early_throttle_stick_difference": float(
                (
                    results["live"]["outcomes"]["throttle_bins"][1]
                    - results[control]["outcomes"]["throttle_bins"][1]
                )
                .abs()
                .mean()
            ),
        }
    return {
        "seed": args.final_seed,
        "episodes": args.final_episodes,
        "summaries": {name: result["summaries"] for name, result in results.items()},
        "comparisons": comparisons,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, _checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    assisted_report, steering_vector = verify_assisted_motor_report(
        args.assisted_motor_report,
        graph_path=args.graph,
        checkpoint_path=args.checkpoint,
    )
    spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=args.maximum_hops,
        floor_quantile=args.scale_floor_quantile,
        maximum_magnitude=args.maximum_edge_magnitude,
    )
    if len(spec.edges) != 282:
        raise SystemExit(f"expected 282 four-hop path edges, found {len(spec.edges)}")
    overlap = edge_overlap_audit(controller, spec, assisted_report)
    if not overlap["direct_parameter_disjointness_passed"]:
        raise SystemExit("acceleration path directly overlaps the preserved steering readout")
    if len(steering_vector) != 24 or bool(steering_vector[18:].abs().max() > 0.0):
        raise SystemExit("preserved steering vector does not have the expected isolated mask")
    zero = torch.zeros(len(spec.edges), device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    vector_path = args.output_dir / "selected-theta.json"
    steering_path = args.output_dir / "preserved-steering-vector.json"
    archive_path = args.output_dir / "archive.pt"
    if any(path.exists() for path in (report_path, vector_path, steering_path, archive_path)):
        raise SystemExit("output directory contains a stale report, vector, or archive")
    seed_everything(args.search_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    rng = np.random.default_rng(args.search_seed)
    center = zero.clone()
    archive: dict[str, dict[str, Any]] = {}
    generations: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    validated: dict[str, dict[str, Any]] = {}
    checkpoint_progress_passed = False
    stopped_at_checkpoint = False
    for generation in range(1, args.generations + 1):
        evaluated_center = center.clone()
        sigma = args.sigma * args.sigma_decay ** (generation - 1)
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_directions, len(spec.edges))).astype(np.float32)
        ).to(device)
        plus = (center + sigma * epsilon).clamp(spec.lower, spec.upper)
        minus = (center - sigma * epsilon).clamp(spec.lower, spec.upper)
        candidates = torch.stack((plus, minus), dim=1).reshape(-1, len(spec.edges))
        policies = torch.cat((zero[None], evaluated_center[None], candidates), dim=0)
        cases = diverse_matched_cases(
            args.development_episodes,
            seed=args.development_seed + generation - 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        weight = shaping_weight(generation, args.generations)
        batch = evaluate_policy_batch(
            controller,
            policies,
            spec,
            cases,
            seconds=args.seconds,
            reward_shaping_weight=weight,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
            axis_intervention="reserve_steering",
            takeover_seconds=args.takeover_seconds,
        )
        reference, center_summary = batch["summaries"][:2]
        candidate_summaries = batch["summaries"][2:]
        objectives = np.asarray(
            [objective(item, reference, args.heavy_reward_penalty) for item in candidate_summaries],
            dtype=np.float32,
        )
        ranks = np.empty(len(objectives), dtype=np.float32)
        ranks[np.argsort(objectives)] = np.linspace(-0.5, 0.5, len(objectives))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        proposed_update = args.learning_rate * gradient
        proposed_rms = proposed_update.square().mean().sqrt()
        maximum_rms = args.maximum_update_sigma_fraction * sigma
        update_scale = min(1.0, maximum_rms / max(float(proposed_rms), 1.0e-12))
        update = proposed_update * update_scale
        center = (center + update).clamp(spec.lower, spec.upper)
        for index, vector in enumerate(candidates):
            key = vector_sha256(vector)
            archive[key] = {
                "vector": vector.detach().cpu().clone(),
                "generation": generation,
                "kind": f"direction_{index // 2 + 1}_{'plus' if index % 2 == 0 else 'minus'}",
                "objective": float(objectives[index]),
                "development": candidate_summaries[index],
            }
        center_key = vector_sha256(evaluated_center)
        center_objective = objective(center_summary, reference, args.heavy_reward_penalty)
        archive[center_key] = {
            "vector": evaluated_center.detach().cpu().clone(),
            "generation": generation,
            "kind": "evaluated_center",
            "objective": center_objective,
            "development": center_summary,
        }
        best_index = int(np.argmax(objectives))
        best_key = vector_sha256(candidates[best_index])
        global_key = max(archive, key=lambda key: archive_rank(archive[key]))
        generation_record = {
            "generation": generation,
            "development_seed": args.development_seed + generation - 1,
            "sigma": sigma,
            "shaping_weight": weight,
            "reference": reference,
            "center": center_summary,
            "center_objective": center_objective,
            "best_candidate": candidate_summaries[best_index],
            "best_candidate_objective": float(objectives[best_index]),
            "proposed_normalized_update_rms": float(proposed_rms),
            "applied_normalized_update_rms": float(update.square().mean().sqrt()),
            "update_cap_scale": update_scale,
            "next_center_sha256": vector_sha256(center),
        }
        generations.append(generation_record)
        print(
            json.dumps(
                {
                    "phase": "search",
                    "generation": generation,
                    "reference_success": reference["success_rate"],
                    "center_success": center_summary["success_rate"],
                    "best_success": candidate_summaries[best_index]["success_rate"],
                    "reference_floor": mass_lateral_floor(reference),
                    "best_floor": mass_lateral_floor(candidate_summaries[best_index]),
                    "best_objective": float(objectives[best_index]),
                    "update_rms": generation_record["applied_normalized_update_rms"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "center_sha256": vector_sha256(center),
                "selected_edge_indices": spec.edges.detach().cpu(),
            },
            args.output_dir / "search-state.pt",
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "experiment": "assisted-acceleration-path-es-v1",
                    "generation": generation,
                    "parameter_count": len(spec.edges),
                    "archive_size": len(archive),
                    "generations": generations,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if generation % args.validation_interval == 0:
            validation = validate_archive(
                controller,
                zero,
                spec,
                archive,
                required_keys=(center_key, best_key, global_key),
                generation=generation,
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            validations.append(validation)
            for key, item in validation["candidates"].items():
                previous = validated.get(key)
                if previous is None or candidate_rank(item["metrics"]) > candidate_rank(
                    previous["metrics"]
                ):
                    validated[key] = item
            if generation == args.checkpoint_generation:
                checkpoint_progress_passed = any(
                    item["performance_qualified"] for item in validated.values()
                )
                if not checkpoint_progress_passed:
                    stopped_at_checkpoint = True
                    print(
                        json.dumps({"phase": "early_stop", "generation": generation}),
                        flush=True,
                    )
                    break
    if not validations:
        raise RuntimeError("assisted acceleration path search produced no validation")
    validation_baseline = validations[-1]["baseline"]
    selected_key, selected_qualified = select_validated_candidate(
        validation_baseline,
        validated,
        maximum_drop=args.maximum_stratum_drop,
    )
    winner = archive[selected_key]["vector"].to(device) if selected_key else zero
    completed_search = len(generations) == args.generations and checkpoint_progress_passed
    final = (
        evaluate_final_controls(
            controller,
            zero,
            winner,
            spec,
            args=args,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        if completed_search and selected_qualified
        else None
    )
    causal_checks: dict[str, bool] | None = None
    final_performance_qualified = False
    if final is not None:
        live_source, live_candidate = final["summaries"]["live"]
        final_performance_qualified = performance_qualified(
            live_source,
            live_candidate,
            minimum_light_gain=args.minimum_light_gain,
            minimum_floor_gain=args.minimum_floor_gain,
            maximum_drop=args.maximum_stratum_drop,
        )
        causal_checks = {}
        for control in ("constant_1g", "pair_swapped"):
            audit = final["comparisons"]["acceleration_controls"][control]
            causal_checks[f"candidate_live_beats_{control}"] = (
                audit["candidate_live_advantage"]["confidence_95"][0] > 0.0
            )
            causal_checks[f"positive_difference_in_differences_vs_{control}"] = (
                audit["difference_in_differences"]["confidence_95"][0] > 0.0
            )
    acceleration_dependence_demonstrated = bool(
        causal_checks is not None and all(causal_checks.values())
    )
    merge_eligible = bool(
        completed_search
        and selected_qualified
        and final_performance_qualified
        and acceleration_dependence_demonstrated
    )
    selected_magnitudes = (spec.baseline + spec.scales * winner).clamp(0.0, spec.maximum)
    changed = (selected_magnitudes - spec.baseline).abs() > 1.0e-8
    vector = {
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "selected_normalized_vector_sha256": vector_sha256(winner),
        "selected_normalized_vector": winner.detach().cpu().tolist(),
        "compiled_or_promoted": False,
    }
    vector_path.write_text(json.dumps(vector, indent=2, sort_keys=True) + "\n")
    steering_path.write_text(
        json.dumps(
            {
                "source_diagnostic_sha256": file_sha256(args.assisted_motor_report),
                "vector_sha256": vector_sha256(steering_vector),
                "parameter_labels": assisted_report["parameterization"]["labels"],
                "vector": steering_vector.tolist(),
                "used_by_acceleration_path_search": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    torch.save(
        {
            key: {
                "vector": item["vector"],
                "generation": item["generation"],
                "kind": item["kind"],
            }
            for key, item in archive.items()
        },
        archive_path,
    )
    report = {
        "method": "teacher-steering-assisted edge-only acceleration-path ES",
        "claim_scope": (
            "This is a training-only causal diagnostic. Reserve steering is externally "
            "substituted after 0.5 seconds, so no candidate from this run is deployable. "
            "Only existing signed edge magnitudes on four-hop acceleration-to-throttle paths "
            "vary; topology, signs, biases, time constants, actor inputs, and recurrent state "
            "are unchanged."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "assisted_motor_report": stable_path(args.assisted_motor_report),
        "assisted_motor_report_sha256": file_sha256(args.assisted_motor_report),
        "preserved_assisted_steering_vector_sha256": vector_sha256(steering_vector),
        "preserved_assisted_steering_vector": stable_path(steering_path),
        "preserved_assisted_steering_file_sha256": file_sha256(steering_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "axis_intervention": "mass-free reserve roll/pitch/yaw after 0.5 seconds",
            "intervention_training_only": True,
            "fixed_topology_and_transmitter_signs": True,
            "biases_frozen": True,
            "time_constants_frozen": True,
            "nonselected_edges_frozen": True,
            "maximum_path_hops": 4,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
            "search_objective": "delta_light_mean_reward - 2*max(0,-delta_heavy_mean_reward)",
            "validation_nominees": "current center, current generation best, global archive best",
            "checkpoint_gate": (
                "one fixed-validation candidate must simultaneously gain >=10pp light and "
                ">=10pp worst mass/lateral, with no mass/lateral stratum dropping >5pp"
            ),
        },
        "parameterization": {
            "count": len(spec.edges),
            "maximum_hops": args.maximum_hops,
            "scale_floor": spec.floor,
            "scale_floor_quantile": args.scale_floor_quantile,
            "zero_edge_count": int(spec.zero_edges.sum()),
            "selected_edge_indices": spec.edges.detach().cpu().tolist(),
            "edge_overlap_audit": overlap,
        },
        "generations_budget": args.generations,
        "generations_completed": len(generations),
        "stopped_at_checkpoint": stopped_at_checkpoint,
        "checkpoint_progress_passed": checkpoint_progress_passed,
        "generations": generations,
        "validations": validations,
        "selected_candidate": selected_key,
        "selected_validation_qualified": selected_qualified,
        "selected_metrics": validated[selected_key]["metrics"] if selected_key else None,
        "selected_vector": stable_path(vector_path),
        "selected_vector_sha256": file_sha256(vector_path),
        "archive": stable_path(archive_path),
        "archive_sha256": file_sha256(archive_path),
        "compiled_edges_changed_if_applied": int(changed.sum()),
        "maximum_edge_magnitude_change_if_applied": float(
            (selected_magnitudes - spec.baseline).abs().max()
        ),
        "final": final,
        "final_performance_qualified": final_performance_qualified,
        "causal_checks": causal_checks,
        "acceleration_dependence_demonstrated": acceleration_dependence_demonstrated,
        "merge_eligible": merge_eligible,
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
                "generations_completed": len(generations),
                "checkpoint_progress_passed": checkpoint_progress_passed,
                "final_performance_qualified": final_performance_qualified,
                "acceleration_dependence_demonstrated": acceleration_dependence_demonstrated,
                "merge_eligible": merge_eligible,
                "goal_passed": False,
            }
        ),
        flush=True,
    )
    return 0 if merge_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
