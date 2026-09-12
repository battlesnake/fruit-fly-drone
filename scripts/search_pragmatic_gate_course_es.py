#!/usr/bin/env python3
"""Optimize complete varied-course flight through existing native motor pathways.

No teacher, decoded visual features, new edges or external actor state. Candidate
coordinates compile into native pool biases and signed incoming synapse gains.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import search_gate_motor_interface_es as motor_es  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import evaluate, sample_two_gate_cases  # noqa: E402
from pragmatic_course_batch import ParameterBatchController, repeat_course_bank  # noqa: E402
from search_pragmatic_full_native_gate_es import apply_vector  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--candidate-batch", type=int, default=1)
    parser.add_argument("--training-pairs", type=int, default=4)
    parser.add_argument("--development-pairs", type=int, default=32)
    parser.add_argument("--development-interval", type=int, default=5)
    parser.add_argument("--course-refresh-generations", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--maximum-minutes", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1_010_983)
    parser.add_argument("--development-seed", type=int, default=1_020_983)
    parser.add_argument("--bias-scale", type=float, default=0.0025)
    parser.add_argument("--gain-scale", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--sigma-decay", type=float, default=0.98)
    parser.add_argument("--spacing-min", type=float, default=0.9)
    parser.add_argument("--spacing-max", type=float, default=1.5)
    parser.add_argument("--lateral-step-max", type=float, default=0.2)
    parser.add_argument("--lateral-deviation-limit", type=float, default=0.5)
    parser.add_argument("--height-step-max", type=float, default=0.08)
    parser.add_argument("--height-min", type=float, default=0.9)
    parser.add_argument("--height-max", type=float, default=1.3)
    parser.add_argument("--yaw-jitter-degrees", type=float, default=15.0)
    return parser.parse_args()


def selection_score(metrics: dict, first_gate_floor: float) -> tuple[float, ...]:
    return (
        metrics["clean_course_success_rate"],
        min(
            metrics["clean_course_negative_success_rate"],
            metrics["clean_course_positive_success_rate"],
        ),
        float(metrics["first_gate_pass_rate"] >= first_gate_floor),
        metrics["gates_before_failure_mean"],
        metrics["course_race_fitness"],
    )


def centered_ranks(scores: np.ndarray) -> np.ndarray:
    """Tie-aware ranks: identical outcomes must not manufacture a search direction."""
    if len(scores) < 2:
        return np.zeros_like(scores, dtype=np.float32)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - 0.5 * (counts + 1.0))[inverse]
    return (ranks / max(len(scores) - 1, 1) - 0.5).astype(np.float32)


def training_course_seed(base_seed, generation, refresh_generations):
    if generation < 1 or refresh_generations < 1:
        raise ValueError("generation and course refresh interval must be positive")
    return base_seed + (generation - 1) // refresh_generations


def main() -> int:
    args = parse_args()
    if (
        min(
            args.generations,
            args.directions,
            args.candidate_batch,
            args.training_pairs,
            args.development_pairs,
            args.development_interval,
            args.course_refresh_generations,
            args.development_interval,
            args.seconds,
            args.maximum_minutes,
            args.bias_scale,
            args.gain_scale,
            args.learning_rate,
            args.sigma_decay,
        )
        <= 0
    ):
        raise SystemExit("search sizes, duration and scales must be positive")
    training_seeds = {
        training_course_seed(args.seed, generation, args.course_refresh_generations)
        for generation in range(1, args.generations + 1)
    }
    if args.development_seed in training_seeds:
        raise SystemExit("training and development course seeds must be separate")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing course-search experiment")
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    controller.eval().requires_grad_(False)
    hover = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(
        width=source["image_resolution"][0],
        height=source["image_resolution"][1],
        horizontal_fov_degrees=source["camera_hfov_degrees"],
    )
    spec = motor_es.motor_interface_spec(
        controller,
        bias_scale=args.bias_scale,
        log_gain_scale=args.gain_scale,
        maximum_bias_delta=0.02,
        maximum_gain_ratio=2.0,
    )
    base_bias = controller.bias.detach().clone()
    base_edge = controller.edge_magnitude.detach().clone()
    center = torch.zeros(len(spec.labels), device=device)
    rng = np.random.default_rng(args.seed)
    geometry = dict(
        layout="variable",
        gate_count=5,
        spacing_range=(args.spacing_min, args.spacing_max),
        lateral_step_range=(0.0, args.lateral_step_max),
        lateral_deviation_limit=args.lateral_deviation_limit,
        height_step_range=(0.0, args.height_step_max),
        height_range=(args.height_min, args.height_max),
        yaw_jitter_degrees=args.yaw_jitter_degrees,
    )
    started = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development = sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=hover,
        **geometry,
    )

    def assess(vector, bank):
        apply_vector(controller, base_bias, base_edge, vector, spec)
        return evaluate(
            controller,
            *bank,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=hover,
            gate_config=gate_config,
        )

    def assess_batch(vectors, bank):
        # The batched evaluator applies candidate deltas to the unchanged source.
        # No candidate sees another candidate's state or trajectory.
        apply_vector(controller, base_bias, base_edge, torch.zeros_like(center), spec)
        actor = ParameterBatchController(controller, vectors, spec, len(bank[0].side))
        return evaluate(
            actor,
            *repeat_course_bank(bank, len(vectors)),
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=hover,
            gate_config=gate_config,
            compact_policies=len(vectors),
        )

    def log(stage, metrics, **extra):
        print(
            json.dumps(
                dict(
                    stage=stage,
                    clean=metrics["clean_course_success_rate"],
                    first=metrics["first_gate_pass_rate"],
                    prefix=metrics["gates_before_failure_mean"],
                    fitness=metrics["course_race_fitness"],
                    elapsed_seconds=round(perf_counter() - started, 2),
                    **extra,
                )
            ),
            flush=True,
        )

    source_metrics = assess(center, development)
    best_metrics = source_metrics
    best_vector = center.clone()
    first_gate_floor = max(0.0, source_metrics["first_gate_pass_rate"] - 0.05)
    log("development-source", source_metrics)
    history = []

    def save():
        apply_vector(controller, base_bias, base_edge, best_vector, spec)
        payload = dict(source)
        payload.update(
            experiment="native-varied-five-gate-motor-es-v1",
            controller={
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            gate_config=vars(gate_config),
            course_geometry=geometry,
            course_rules="ordered-directed-all-annuli-v1",
            selection_metrics=best_metrics,
            native_search_vector=best_vector.detach().cpu(),
            native_search_labels=list(spec.labels),
        )
        torch.save(payload, args.output_dir / "best-controller.pt")
        report = dict(
            experiment=payload["experiment"],
            source_checkpoint=str(args.checkpoint),
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            geometry=geometry,
            parameterization=list(spec.description),
            actor_extra_state_or_privileged_inputs=False,
            source_development=source_metrics,
            selected_development=best_metrics,
            first_gate_floor=first_gate_floor,
            center=center.tolist(),
            best_vector=best_vector.tolist(),
            history=history,
            elapsed_seconds=perf_counter() - started,
        )
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for generation in range(1, args.generations + 1):
        if perf_counter() - started > args.maximum_minutes * 60:
            break
        case_seed = training_course_seed(args.seed, generation, args.course_refresh_generations)
        bank = sample_two_gate_cases(
            args.training_pairs,
            seed=case_seed,
            device=device,
            hover_config=hover,
            **geometry,
        )
        sigma = args.sigma_decay ** (generation - 1)
        epsilon = torch.tensor(
            rng.standard_normal((args.directions, len(spec.labels))),
            dtype=torch.float32,
            device=device,
        )
        delta = sigma * spec.scales * epsilon
        candidates = torch.stack((center + delta, center - delta), dim=1).reshape(
            -1, len(spec.labels)
        )
        candidates = candidates.clamp(spec.lower, spec.upper)
        metrics = []
        for offset in range(0, len(candidates), args.candidate_batch):
            chunk = candidates[offset:offset + args.candidate_batch]
            results = (
                [assess(chunk[0], bank)] if args.candidate_batch == 1
                else assess_batch(chunk, bank)
            )
            for local, result in enumerate(results):
                metrics.append(result)
                log(
                    "candidate", result, generation=generation,
                    candidate=offset + local, course_seed=case_seed,
                )
        values = np.asarray([result["course_race_fitness"] for result in metrics])
        utilities = torch.tensor(centered_ranks(values), device=device)
        gradient = ((utilities[0::2] - utilities[1::2])[:, None] * epsilon).mean(dim=0) / sigma
        center = (center + args.learning_rate * spec.scales * gradient).clamp(
            spec.lower, spec.upper
        )
        center_metrics = assess(center, bank)
        log("center", center_metrics, generation=generation)
        winner = int(values.argmax())
        record = dict(
            generation=generation,
            course_seed=case_seed,
            sigma=sigma,
            candidates=metrics,
            candidate_vectors=candidates.tolist(),
            center_metrics=center_metrics,
            center_vector=center.tolist(),
        )
        if generation % args.development_interval == 0 or generation == args.generations:
            checks = []
            for label, vector in (("center", center), ("generation-best", candidates[winner])):
                result = assess(vector, development)
                log("development", result, generation=generation, candidate=label)
                checks.append(dict(candidate=label, metrics=result))
                if selection_score(result, first_gate_floor) > selection_score(
                    best_metrics, first_gate_floor
                ):
                    best_metrics, best_vector = result, vector.clone()
            record["development"] = checks
        history.append(record)
        save()
        if generation == 10 and (
            best_metrics["clean_course_success_rate"]
            < source_metrics["clean_course_success_rate"] + 0.05
            and best_metrics["gates_before_failure_mean"]
            < source_metrics["gates_before_failure_mean"] + 0.5
        ):
            print(
                json.dumps(
                    {
                        "stage": "stop",
                        "reason": "no useful development improvement by generation 10",
                    }
                ),
                flush=True,
            )
            break
    log("complete", best_metrics, generations=len(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
