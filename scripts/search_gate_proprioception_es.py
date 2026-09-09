#!/usr/bin/env python3
"""Search the added claw-proprioception circuit using complete gate flights."""

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

from search_gate_acceleration_es import (  # noqa: E402
    compact_metrics,
    promotion_test,
    task_fitness,
    vector_sha256,
)
from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
    evaluate_gate,
    file_sha256,
    initial_rollout,
    remap_anatomical_parameters,
    seed_everything,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


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
        default=REPO_ROOT / "runs" / "gate" / "proprio-v0" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--checkpoint-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "proprio-es",
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
    parser.add_argument("--development-seed", type=int, default=80_031)
    parser.add_argument("--validation-seed", type=int, default=90_031)
    parser.add_argument("--final-seed", type=int, default=100_031)
    parser.add_argument("--search-seed", type=int, default=13)
    parser.add_argument(
        "--frozen-proprioception-search",
        action="store_true",
        help="Same-capacity control: optimize added edges while the joint sensor is constant.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.generations < 1 or args.antithetic_pairs < 1:
        raise SystemExit("generations and antithetic pairs must be positive")
    for name in ("development_episodes", "validation_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by 8")


def added_circuit_edges(graph_path: Path, checkpoint_graph_path: Path) -> np.ndarray:
    """Return every anatomical edge with at least one newly selected endpoint."""

    with np.load(graph_path) as graph, np.load(checkpoint_graph_path) as checkpoint_graph:
        node_ids = graph["node_ids"]
        edge_pre = graph["edge_pre"]
        edge_post = graph["edge_post"]
        old = np.isin(node_ids, checkpoint_graph["node_ids"])
        selected = ~(old[edge_pre] & old[edge_post])
    result = np.flatnonzero(selected)
    if not len(result):
        raise SystemExit("destination graph has no added proprioception-circuit edges")
    return result


def apply_vector(
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
    frozen_search: bool = False,
    **controls: Any,
) -> dict[str, Any]:
    apply_vector(controller, base_state, edge_indices, values)
    if frozen_search:
        controls = {**controls, "frozen_proprioception": True}
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
def position_acceleration_probe(
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
    """Measure whether acceleration corrections depend on sensed throttle position."""

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
        motor, neural = controller(
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        rc, stick_state = sticks(motor, stick_state)
        state = quad(rc, state, mass_scale)

    image = render_annular_gate(
        state,
        gate,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    levels: dict[str, float] = {}
    for position_label, position in (("low", -0.5), ("high", 0.5)):
        for acceleration_label, multiplier in (("0.9g", 0.9), ("1.1g", 1.1)):
            forked_neural = neural.clone()
            forked_sticks = copy.deepcopy(stick_state)
            force = torch.zeros(episodes, 3, device=device)
            force[:, 2] = multiplier * 9.81
            sensed_position = stick_state.position.clone()
            sensed_position[:, 3] = position
            throttle = []
            for _ in range(intervention_steps):
                motor, forked_neural = controller(
                    image,
                    state.euler[:, :2],
                    forked_neural,
                    force,
                    sensed_position,
                )
                rc, forked_sticks = sticks(motor, forked_sticks)
                throttle.append(rc[:, 3])
            levels[f"{position_label}_{acceleration_label}"] = float(
                torch.stack(throttle[-10:]).mean()
            )
    low_effect = levels["low_1.1g"] - levels["low_0.9g"]
    high_effect = levels["high_1.1g"] - levels["high_0.9g"]
    return {
        "forced_normalized_throttle_joint_positions": {"low": -0.5, "high": 0.5},
        "mean_measured_throttle": levels,
        "acceleration_effect_at_low_position": low_effect,
        "acceleration_effect_at_high_position": high_effect,
        "position_x_acceleration_interaction": high_effect - low_effect,
        "intended_acceleration_sign_at_both_positions": low_effect < 0.0 and high_effect < 0.0,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["graph_sha256"] != file_sha256(args.checkpoint_graph):
        raise SystemExit("checkpoint and --checkpoint-graph hashes do not match")
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
    remap = remap_anatomical_parameters(
        controller,
        checkpoint,
        args.checkpoint_graph,
        args.graph,
        freeze_existing=False,
        new_visual_hemifields=False,
        keep_roll_biases_frozen=True,
        new_pathways_roll_only=False,
        new_acceleration_pathways_only=False,
    )
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    edge_indices_np = added_circuit_edges(args.graph, args.checkpoint_graph)
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
            frozen_search=args.frozen_proprioception_search,
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
            frozen_search=args.frozen_proprioception_search,
        )
        centre_key = vector_sha256(centre)
        archive[centre_key] = {
            "values": centre.detach().cpu().clone(),
            "development_metrics": compact_metrics(centre_metrics, clean_radius),
            "fitness_delta_from_baseline": task_fitness(centre_metrics, clean_radius)
            - baseline_fitness,
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
                frozen_search=args.frozen_proprioception_search,
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
            "best_fitness_delta_from_baseline": candidate_fitness[best_index] - baseline_fitness,
            "next_centre_sha256": vector_sha256(centre),
        }
        generations.append(entry)
        print(json.dumps({"phase": "search", **entry}), flush=True)
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "edge_count": int(edge_indices.numel()),
                    "generations": generations,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
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
        frozen_search=args.frozen_proprioception_search,
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
            frozen_search=args.frozen_proprioception_search,
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
    search_controls = {"frozen_proprioception": True} if args.frozen_proprioception_search else {}
    final_specs = {
        "baseline": (base_vector, search_controls),
        "candidate": (winner_vector, search_controls),
        "candidate_live_joint_position": (winner_vector, {}),
        "candidate_constant_proprioception": (
            winner_vector,
            {"frozen_proprioception": True},
        ),
        "candidate_mass_rank_swapped_proprioception": (
            winner_vector,
            {"swapped_proprioception": True},
        ),
        "candidate_constant_1g": (
            winner_vector,
            {**search_controls, "frozen_acceleration": True},
        ),
        "candidate_both_feedback_controls": (
            winner_vector,
            {"frozen_acceleration": True, "frozen_proprioception": True},
        ),
        "candidate_frozen_first_frame": (
            winner_vector,
            {**search_controls, "frozen_visual": True},
        ),
    }
    final: dict[str, Any] = {}
    for name, (values, controls) in final_specs.items():
        final[name] = evaluate_vector(
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
        print(
            json.dumps(
                {"phase": "final", "name": name, **compact_metrics(final[name], clean_radius)}
            ),
            flush=True,
        )

    apply_vector(controller, base_state, edge_indices, base_vector)
    probes = {
        "baseline": position_acceleration_probe(
            controller,
            episodes=64,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
    }
    apply_vector(controller, base_state, edge_indices, winner_vector)
    probes["candidate"] = position_acceleration_probe(
        controller,
        episodes=64,
        seed=args.final_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    sensor_dependence_checks = {
        "live_success_exceeds_constant_position": (
            final["candidate"]["success_rate"]
            > final["candidate_constant_proprioception"]["success_rate"]
        ),
        "live_success_differs_from_episode_swapped_position": (
            final["candidate"]["success_rate"]
            != final["candidate_mass_rank_swapped_proprioception"]["success_rate"]
        ),
        "position_acceleration_interaction_exceeds_1e_6": (
            abs(probes["candidate"]["position_x_acceleration_interaction"]) > 1.0e-6
        ),
    }
    sensor_dependence_demonstrated = all(sensor_dependence_checks.values())

    candidate_checkpoint = copy.deepcopy(checkpoint)
    candidate_checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    candidate_checkpoint["graph_sha256"] = file_sha256(args.graph)
    candidate_checkpoint["source_checkpoint_sha256"] = file_sha256(args.checkpoint)
    candidate_checkpoint["graph_remap"] = remap
    candidate_checkpoint["edge_search"] = {
        "method": "ranked antithetic evolution strategy on complete flights",
        "added_circuit_edges": int(edge_indices.numel()),
        "edge_vector_sha256": vector_sha256(winner_vector),
        "frozen_proprioception_search": args.frozen_proprioception_search,
        "validation_promotion_eligible": bool(eligible),
    }
    checkpoint_path = args.output_dir / "candidate.pt"
    torch.save(candidate_checkpoint, checkpoint_path)

    report = {
        "method": "ranked antithetic evolution strategy on complete flights",
        "claim_scope": "Added-circuit diagnostic; not promoted without sensor-causal controls.",
        "deployed_runtime_parameters_added": 0,
        "all_changes_folded_into_connectome_edges": True,
        "biases_changed": False,
        "time_constants_changed": False,
        "old_anatomy_changed": False,
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_graph": stable_path(args.checkpoint_graph),
        "graph_remap": remap,
        "candidate_checkpoint": stable_path(checkpoint_path),
        "candidate_checkpoint_sha256": file_sha256(checkpoint_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "search": {
            "generations": args.generations,
            "antithetic_candidates_per_generation": 2 * args.antithetic_pairs,
            "unchanged_warm_start_evaluated_each_generation": True,
            "exactly_balanced_mass_side_obliquity_strata": True,
            "frozen_proprioception_search": args.frozen_proprioception_search,
            "added_circuit_edges": int(edge_indices.numel()),
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
        },
        "generations": generations,
        "validation_baseline": validation_baseline,
        "validation": validation,
        "selected_candidate": winner_key,
        "validation_promotion_eligible": bool(eligible),
        "final_promotion_test": promotion_test(
            final["candidate"],
            final["baseline"],
            maximum_heavy_drop=args.maximum_heavy_mass_drop,
        ),
        "position_acceleration_probe": probes,
        "sensor_dependence_checks": sensor_dependence_checks,
        "sensor_dependence_demonstrated": sensor_dependence_demonstrated,
        "promoted_as_controller": bool(final["candidate"]["goal_pass"])
        and sensor_dependence_demonstrated,
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
                "candidate_checkpoint": str(checkpoint_path),
                "selected_candidate": winner_key,
                "validation_promotion_eligible": bool(eligible),
                "goal_passed": report["goal_passed"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
