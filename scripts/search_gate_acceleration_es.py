#!/usr/bin/env python3
"""Search existing acceleration-path edges using complete annular-gate flights."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    edges_on_short_paths,
    evaluate_gate,
    file_sha256,
    initial_rollout,
    seed_everything,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "acceleration-es",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--antithetic-pairs", type=int, default=8)
    parser.add_argument("--development-episodes", type=int, default=64)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=0.30)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--edge-scale-floor", type=float, default=0.05)
    parser.add_argument("--maximum-heavy-mass-drop", type=float, default=0.10)
    parser.add_argument("--development-seed", type=int, default=50_031)
    parser.add_argument("--validation-seed", type=int, default=60_031)
    parser.add_argument("--final-seed", type=int, default=70_031)
    parser.add_argument("--search-seed", type=int, default=31)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.baseline_graph):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if (
        min(
            args.generations,
            args.antithetic_pairs,
            args.development_episodes,
            args.validation_episodes,
            args.final_episodes,
            args.top_candidates,
        )
        <= 0
    ):
        raise SystemExit("search sizes and episode counts must be positive")
    if not 0.0 < args.sigma_decay <= 1.0:
        raise SystemExit("--sigma-decay must be in (0, 1]")
    if min(args.sigma, args.learning_rate, args.edge_scale_floor) <= 0.0:
        raise SystemExit("sigma, learning rate, and edge scale floor must be positive")


def acceleration_path_edges(
    graph_path: Path,
    baseline_graph_path: Path,
    *,
    max_hops: int = 8,
) -> np.ndarray:
    with np.load(graph_path) as graph, np.load(baseline_graph_path) as baseline:
        node_ids = graph["node_ids"]
        edge_pre = graph["edge_pre"]
        edge_post = graph["edge_post"]
        pool_offsets = graph["output_pool_offsets"]
        pool_indices = graph["output_pool_indices"]
        throttle_nodes = pool_indices[pool_offsets[6] : pool_offsets[8]]
        path_mask = edges_on_short_paths(
            edge_pre,
            edge_post,
            len(node_ids),
            graph["acceleration_node_indices"],
            throttle_nodes,
            max_hops,
        )
        presynaptic_is_new = ~np.isin(node_ids[edge_pre], baseline["node_ids"])
        return np.nonzero(path_mask & presynaptic_is_new)[0]


def vector_sha256(values: torch.Tensor) -> str:
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def apply_edge_vector(
    controller: ConnectomeController,
    base_state: dict[str, torch.Tensor],
    edge_indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    controller.load_state_dict(base_state)
    with torch.no_grad():
        controller.edge_magnitude[edge_indices] = values.to(
            device=controller.edge_magnitude.device,
            dtype=controller.edge_magnitude.dtype,
        )
    controller.eval()


def task_fitness(metrics: dict[str, Any], clean_radius: float) -> float:
    """Complete success dominates; dense crossing terms only break near ties."""

    strata = metrics["success_by_stratum"]
    worst_mass = min(strata["lower_mass"], strata["higher_mass"])
    radial = metrics["crossing_radial_mean_m"]
    postflight_failure = max(metrics["pass_rate"] - metrics["success_rate"], 0.0)
    return float(
        1000.0 * metrics["success_rate"]
        + 250.0 * worst_mass
        - radial / clean_radius
        - 5.0 * metrics["lateral_aperture_exceedance_rate"]
        - 20.0 * (1.0 - metrics["plane_crossing_rate"])
        - 10.0 * postflight_failure
    )


def compact_metrics(metrics: dict[str, Any], clean_radius: float) -> dict[str, Any]:
    crossing = metrics["crossing_error_components_m"]
    strata = metrics["success_by_stratum"]
    return {
        "fitness": task_fitness(metrics, clean_radius),
        "success_rate": metrics["success_rate"],
        "lower_mass_success_rate": strata["lower_mass"],
        "higher_mass_success_rate": strata["higher_mass"],
        "negative_offset_success_rate": strata["negative_lateral_offset"],
        "positive_offset_success_rate": strata["positive_lateral_offset"],
        "plane_crossing_rate": metrics["plane_crossing_rate"],
        "ring_collision_rate": metrics["ring_collision_rate"],
        "miss_rate": metrics["miss_rate"],
        "radial_error_mean_m": metrics["crossing_radial_mean_m"],
        "radial_error_p90_m": metrics["crossing_radial_quantiles_m"]["90"],
        "lateral_absolute_error_mean_m": crossing["lateral_absolute_mean"],
        "vertical_absolute_error_mean_m": crossing["vertical_absolute_mean"],
        "vertical_signed_error_mean_m": crossing["vertical_signed_mean"],
        "lateral_aperture_exceedance_rate": metrics["lateral_aperture_exceedance_rate"],
    }


def evaluate_vector(
    controller: ConnectomeController,
    base_state: dict[str, torch.Tensor],
    edge_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    episodes: int,
    seconds: float,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    **controls: Any,
) -> dict[str, Any]:
    apply_edge_vector(controller, base_state, edge_indices, values)
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
        **controls,
    )


@torch.no_grad()
def sensor_step_response(
    controller: ConnectomeController,
    *,
    episodes: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    prefix_steps: int = 150,
    intervention_steps: int = 20,
) -> dict[str, Any]:
    """Fork live recurrent/leg states and intervene on body-Z specific force."""

    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    for _ in range(prefix_steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        motor, neural = controller(image, state.euler[:, :2], neural, state.specific_force)
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)

    image = render_annular_gate(
        state,
        gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    levels: dict[str, torch.Tensor] = {}
    for label, multiplier in (("0.9g", 0.9), ("1.0g", 1.0), ("1.1g", 1.1)):
        forked_neural = neural.clone()
        forked_sticks = copy.deepcopy(stick_state)
        force = torch.zeros(episodes, 3, device=device)
        force[:, 2] = multiplier * 9.81
        throttle = []
        for _ in range(intervention_steps):
            motor, forked_neural = controller(
                image,
                state.euler[:, :2],
                forked_neural,
                force,
            )
            rc, forked_sticks = sticks(motor, forked_sticks)
            throttle.append(rc[:, 3])
        levels[label] = torch.stack(throttle[-10:]).mean(dim=0)

    masks = {
        "all": torch.ones(episodes, dtype=torch.bool, device=device),
        "lower_mass": mass_scale < 1.0,
        "higher_mass": mass_scale >= 1.0,
    }
    response: dict[str, Any] = {}
    for name, mask in masks.items():
        response[name] = {
            "throttle_0.9g": float(levels["0.9g"][mask].mean()),
            "throttle_1.0g": float(levels["1.0g"][mask].mean()),
            "throttle_1.1g": float(levels["1.1g"][mask].mean()),
            "below_minus_1g": float((levels["0.9g"][mask] - levels["1.0g"][mask]).mean()),
            "above_minus_1g": float((levels["1.1g"][mask] - levels["1.0g"][mask]).mean()),
        }
    response["intended_signs"] = {
        "below_1g_increases_throttle": response["all"]["below_minus_1g"] > 0.0,
        "above_1g_decreases_throttle": response["all"]["above_minus_1g"] < 0.0,
    }
    return response


def promotion_test(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    maximum_heavy_drop: float,
) -> dict[str, bool]:
    candidate_strata = candidate["success_by_stratum"]
    baseline_strata = baseline["success_by_stratum"]
    checks = {
        "overall_completion_improved": candidate["success_rate"] > baseline["success_rate"],
        "lower_mass_completion_improved": (
            candidate_strata["lower_mass"] > baseline_strata["lower_mass"]
        ),
        "higher_mass_drop_within_limit": (
            candidate_strata["higher_mass"] >= baseline_strata["higher_mass"] - maximum_heavy_drop
        ),
    }
    checks["eligible"] = all(checks.values())
    return checks


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
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    resolution = int(checkpoint["image_resolution"])
    controller = ConnectomeController(
        args.graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    edge_indices_np = acceleration_path_edges(args.graph, args.baseline_graph)
    edge_indices = torch.from_numpy(edge_indices_np).to(device=device)
    base_vector = controller.edge_magnitude[edge_indices].detach().clone()
    centre = base_vector.clone()
    edge_scale = base_vector.abs().clamp_min(args.edge_scale_floor)
    rng = np.random.default_rng(args.search_seed)
    archive: dict[str, dict[str, Any]] = {}
    generations: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    for generation in range(args.generations):
        seed = args.development_seed + generation
        baseline_metrics = evaluate_vector(
            controller,
            base_state,
            edge_indices,
            base_vector,
            episodes=args.development_episodes,
            seconds=args.seconds,
            seed=seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        baseline_fitness = task_fitness(baseline_metrics, clean_radius)
        centre_metrics = evaluate_vector(
            controller,
            base_state,
            edge_indices,
            centre,
            episodes=args.development_episodes,
            seconds=args.seconds,
            seed=seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        centre_key = vector_sha256(centre)
        archive[centre_key] = {
            "values": centre.detach().cpu().clone(),
            "development_metrics": compact_metrics(centre_metrics, clean_radius),
            "fitness_delta_from_baseline": (
                task_fitness(centre_metrics, clean_radius) - baseline_fitness
            ),
            "generation": generation + 1,
            "kind": "centre",
        }

        sigma = args.sigma * args.sigma_decay**generation
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_pairs, edge_indices.numel())).astype(np.float32)
        ).to(device)
        candidate_vectors = []
        for pair in range(args.antithetic_pairs):
            delta = sigma * edge_scale * epsilon[pair]
            candidate_vectors.extend(
                ((centre + delta).clamp(0.0, 8.0), (centre - delta).clamp(0.0, 8.0))
            )

        candidate_fitness = []
        candidate_summaries = []
        for index, values in enumerate(candidate_vectors):
            metrics = evaluate_vector(
                controller,
                base_state,
                edge_indices,
                values,
                episodes=args.development_episodes,
                seconds=args.seconds,
                seed=seed,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            fitness = task_fitness(metrics, clean_radius)
            summary = compact_metrics(metrics, clean_radius)
            key = vector_sha256(values)
            archive[key] = {
                "values": values.detach().cpu().clone(),
                "development_metrics": summary,
                "fitness_delta_from_baseline": fitness - baseline_fitness,
                "generation": generation + 1,
                "kind": f"pair_{index // 2 + 1}_{'plus' if index % 2 == 0 else 'minus'}",
            }
            candidate_fitness.append(fitness)
            candidate_summaries.append(summary)

        fitness_array = np.asarray(candidate_fitness)
        ranks = np.empty(len(fitness_array), dtype=np.float32)
        ranks[np.argsort(fitness_array)] = np.linspace(-0.5, 0.5, len(fitness_array))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        centre = (centre + args.learning_rate * edge_scale * gradient).clamp(0.0, 8.0)
        best_index = int(np.argmax(fitness_array))
        entry = {
            "generation": generation + 1,
            "seed": seed,
            "sigma": sigma,
            "baseline": compact_metrics(baseline_metrics, clean_radius),
            "centre": compact_metrics(centre_metrics, clean_radius),
            "best_candidate": candidate_summaries[best_index],
            "best_candidate_index": best_index + 1,
            "best_fitness_delta_from_baseline": (candidate_fitness[best_index] - baseline_fitness),
            "next_centre_sha256": vector_sha256(centre),
        }
        generations.append(entry)
        print(json.dumps({"phase": "search", **entry}), flush=True)
        progress = {
            "edge_count": int(edge_indices.numel()),
            "generations": generations,
            "elapsed_seconds": perf_counter() - started,
        }
        (args.output_dir / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n"
        )

    ranked_archive = sorted(
        archive,
        key=lambda key: (
            archive[key]["fitness_delta_from_baseline"],
            archive[key]["development_metrics"]["success_rate"],
            -archive[key]["development_metrics"]["radial_error_p90_m"],
        ),
        reverse=True,
    )
    finalists = ranked_archive[: min(args.top_candidates, len(ranked_archive))]
    validation_baseline = evaluate_vector(
        controller,
        base_state,
        edge_indices,
        base_vector,
        episodes=args.validation_episodes,
        seconds=args.seconds,
        seed=args.validation_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    validation: dict[str, Any] = {}
    for key in finalists:
        values = archive[key]["values"].to(device)
        metrics = evaluate_vector(
            controller,
            base_state,
            edge_indices,
            values,
            episodes=args.validation_episodes,
            seconds=args.seconds,
            seed=args.validation_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        validation[key] = {
            "archive": {name: value for name, value in archive[key].items() if name != "values"},
            "metrics": metrics,
            "compact_metrics": compact_metrics(metrics, clean_radius),
            "promotion_test": promotion_test(
                metrics,
                validation_baseline,
                maximum_heavy_drop=args.maximum_heavy_mass_drop,
            ),
        }
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "candidate": key,
                    **validation[key]["compact_metrics"],
                    **validation[key]["promotion_test"],
                }
            ),
            flush=True,
        )

    eligible = [key for key in finalists if validation[key]["promotion_test"]["eligible"]]
    selection_pool = eligible or finalists
    winner_key = max(
        selection_pool,
        key=lambda key: (
            task_fitness(validation[key]["metrics"], clean_radius),
            validation[key]["metrics"]["success_rate"],
        ),
    )
    winner_vector = archive[winner_key]["values"].to(device)

    final_specs = {
        "baseline": (base_vector, {}),
        "candidate": (winner_vector, {}),
        "candidate_constant_1g": (winner_vector, {"frozen_acceleration": True}),
        "candidate_above_1g_channel_disabled": (
            winner_vector,
            {"disabled_acceleration_channel": "above_1g"},
        ),
        "candidate_below_1g_channel_disabled": (
            winner_vector,
            {"disabled_acceleration_channel": "below_1g"},
        ),
        "candidate_mass_rank_swapped_trace": (
            winner_vector,
            {"swapped_acceleration": True},
        ),
        "candidate_frozen_first_frame": (winner_vector, {"frozen_visual": True}),
    }
    final: dict[str, Any] = {}
    for name, (values, controls) in final_specs.items():
        metrics = evaluate_vector(
            controller,
            base_state,
            edge_indices,
            values,
            episodes=args.final_episodes,
            seconds=args.seconds,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            **controls,
        )
        final[name] = metrics
        print(
            json.dumps({"phase": "final", "name": name, **compact_metrics(metrics, clean_radius)}),
            flush=True,
        )

    final_promotion_test = promotion_test(
        final["candidate"],
        final["baseline"],
        maximum_heavy_drop=args.maximum_heavy_mass_drop,
    )
    apply_edge_vector(controller, base_state, edge_indices, base_vector)
    signed_response = {
        "baseline": sensor_step_response(
            controller,
            episodes=64,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
    }
    apply_edge_vector(controller, base_state, edge_indices, winner_vector)
    signed_response["candidate"] = sensor_step_response(
        controller,
        episodes=64,
        seed=args.final_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )

    candidate_checkpoint = copy.deepcopy(checkpoint)
    candidate_checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    candidate_checkpoint["source_checkpoint_sha256"] = file_sha256(args.checkpoint)
    candidate_checkpoint["edge_search"] = {
        "method": "ranked antithetic evolution strategy on complete flights",
        "acceleration_path_edges": int(edge_indices.numel()),
        "edge_vector_sha256": vector_sha256(winner_vector),
        "validation_promotion_eligible": bool(eligible),
        "final_promotion_test": final_promotion_test,
    }
    checkpoint_path = args.output_dir / "candidate.pt"
    torch.save(candidate_checkpoint, checkpoint_path)

    report = {
        "method": "ranked antithetic evolution strategy on complete flights",
        "deployed_runtime_parameters_added": 0,
        "all_changes_folded_into_connectome_edges": True,
        "biases_changed": False,
        "time_constants_changed": False,
        "old_anatomy_changed": False,
        "graph": str(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": file_sha256(checkpoint_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "search": {
            "generations": args.generations,
            "antithetic_candidates_per_generation": 2 * args.antithetic_pairs,
            "unchanged_checkpoint_evaluated_each_generation": True,
            "exactly_balanced_mass_side_obliquity_strata": True,
            "acceleration_path_edges": int(edge_indices.numel()),
            "initial_zero_edges": int((base_vector == 0.0).sum()),
            "additive_perturbations": True,
            "sigma": args.sigma,
            "sigma_decay": args.sigma_decay,
            "learning_rate": args.learning_rate,
            "edge_scale_floor": args.edge_scale_floor,
            "development_episodes": args.development_episodes,
            "validation_episodes": args.validation_episodes,
            "final_episodes": args.final_episodes,
            "development_seed": args.development_seed,
            "validation_seed": args.validation_seed,
            "final_seed": args.final_seed,
            "fitness": (
                "1000*success + 250*min(mass success) - normalized radial mean "
                "- lateral/no-crossing/postflight penalties"
            ),
        },
        "generations": generations,
        "validation_baseline": validation_baseline,
        "validation": validation,
        "selected_candidate": winner_key,
        "validation_promotion_eligible": bool(eligible),
        "final_promotion_test": final_promotion_test,
        "signed_sensor_step_response": signed_response,
        "final": final,
        "goal_passed": final["candidate"]["goal_pass"],
        "elapsed_seconds": perf_counter() - started,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "selected_candidate": winner_key,
                "promotion_eligible": final_promotion_test["eligible"],
                "goal_passed": report["goal_passed"],
            }
        ),
        flush=True,
    )
    return 0 if report["goal_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
