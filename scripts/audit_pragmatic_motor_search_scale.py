#!/usr/bin/env python3
"""One bounded, source-centred native motor-parameter scale screen.

Four matched antithetic directions at 0.1x and 0.25x; no centre updates,
teacher, action noise or new actor inputs. At most one nominee is confirmed.
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
from search_pragmatic_gate_course_es import (  # noqa: E402
    outcome_search_fitness,
    safe_development_candidate,
    selection_score,
)
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def scale_screen_vectors(spec, seed, *, directions=4):
    if directions < 1:
        raise ValueError("positive direction count required")
    epsilon = torch.as_tensor(
        np.random.default_rng(seed).standard_normal((directions, len(spec.scales))),
        device=spec.scales.device, dtype=spec.scales.dtype,
    )
    vectors, labels = [], []
    for scale in (0.1, 0.25):
        for direction in range(directions):
            for sign in (1, -1):
                raw = sign * scale * spec.scales * epsilon[direction]
                vectors.append(raw.clamp(spec.lower, spec.upper))
                labels.append(dict(scale=scale, direction=direction, sign=sign))
    return torch.stack(vectors), labels


def meaningful_gain(candidate, source, *, extra_clean=4):
    if candidate["episodes"] != source["episodes"]:
        raise ValueError("source and candidate must use matched episode counts")
    return (
        safe_development_candidate(candidate)
        and candidate["clean_course_success_rate"] - source["clean_course_success_rate"]
        >= extra_clean / source["episodes"]
    )


def nominate_once(candidates, source):
    eligible = [index for index, result in enumerate(candidates) if meaningful_gain(result, source)]
    return (
        max(eligible, key=lambda index: selection_score(candidates[index], 0)) if eligible else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026091393)
    parser.add_argument("--direction-seed", type=int, default=2026091394)
    parser.add_argument("--development-seed", type=int, default=1110983)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite a scale-screen experiment")
    if args.seed == args.development_seed:
        raise SystemExit("training and development courses must be separate")
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    spec = motor_es.motor_interface_spec(
        controller, bias_scale=0.0025, log_gain_scale=0.05,
        maximum_bias_delta=0.02, maximum_gain_ratio=2.0,
    )
    base_bias = controller.bias.detach().clone()
    base_edge = controller.edge_magnitude.detach().clone()
    zero = torch.zeros_like(spec.scales)
    vectors, labels = scale_screen_vectors(spec, args.direction_seed)
    training = sample_two_gate_cases(16, seed=args.seed, device=device,
                                     hover_config=config, **GEOMETRY)
    options = dict(seconds=30, warmup_steps=10, camera=camera,
                   hover_config=config, gate_config=gate_config)
    started = perf_counter()
    args.output_dir.mkdir(parents=True)
    report = dict(
        experiment="source-centred-native-motor-scale-screen-v1", status="running",
        checkpoint=str(args.checkpoint), seed=args.seed, direction_seed=args.direction_seed,
        development_seed=args.development_seed, geometry=GEOMETRY, episodes=32,
        seconds=30, directions=4, scales=[0.1, 0.25], centre_is_always_source=True,
        original_bias_scale=0.0025, original_log_gain_scale=0.05,
        parameter_labels=list(spec.labels), actor_extra_inputs_or_state=False,
        candidates=[], nominee=None, selected_controller=None,
        minimum_extra_clean=4, followup_authorized=False,
    )

    def save():
        report["elapsed_seconds"] = perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def log(stage, metrics, **extra):
        print(json.dumps(dict(
            stage=stage, clean=metrics["clean_course_success_rate"],
            first=metrics["first_gate_pass_rate"], prefix=metrics["gates_before_failure_mean"],
            ground=metrics["ground_contact_rate"], invalid=metrics["invalid_rate"],
            search_fitness=outcome_search_fitness(metrics, 25),
            elapsed_seconds=perf_counter() - started, **extra,
        )), flush=True)

    def assess(vector, bank):
        apply_vector(controller, base_bias, base_edge, vector, spec)
        return evaluate(controller, *bank, **options)

    baseline = assess(zero, training)
    report["source_training"] = baseline
    log("source-training", baseline)
    save()
    for offset in range(0, len(vectors), 2):
        apply_vector(controller, base_bias, base_edge, zero, spec)
        actor = ParameterBatchController(controller, vectors[offset:offset + 2], spec, episodes=32)
        metrics = evaluate(actor, *repeat_course_bank(training, 2), compact_policies=2, **options)
        for local, result in enumerate(metrics):
            index = offset + local
            report["candidates"].append(dict(
                index=index, **labels[index], vector=vectors[index].tolist(), metrics=result,
                search_fitness=outcome_search_fitness(result, 25),
                screen_eligible=meaningful_gain(result, baseline),
            ))
            log("candidate", result, index=index, **labels[index])
        save()
    nominee = nominate_once([c["metrics"] for c in report["candidates"]], baseline)
    report["nominee"] = nominee
    report["decision"] = "close-limited-motor-search-no-training-nominee"
    if nominee is not None:
        confirmed = assess(vectors[nominee], training)
        report["nominee_standalone_training"] = confirmed
        log("nominee-standalone-training", confirmed, index=nominee)
        report["decision"] = "close-limited-motor-search-confirmation-failed"
        if meaningful_gain(confirmed, baseline):
            development = sample_two_gate_cases(16, seed=args.development_seed, device=device,
                                                 hover_config=config, **GEOMETRY)
            source_dev = assess(zero, development)
            candidate_dev = assess(vectors[nominee], development)
            report.update(source_development=source_dev, nominee_development=candidate_dev)
            log("source-development", source_dev)
            log("nominee-development", candidate_dev, index=nominee)
            # The same coarse gain is required here before authorizing more ES.
            report["followup_authorized"] = meaningful_gain(candidate_dev, source_dev)
            report["decision"] = "close-limited-motor-search-no-meaningful-development-gain"
            if report["followup_authorized"]:
                report["decision"] = "brief-small-scale-followup-supported-not-goal-proof"
                name = "candidate-controller.pt"
                payload = dict(source)
                payload.update(
                    controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
                    experiment=report["experiment"], gate_config=vars(gate_config),
                    course_geometry=GEOMETRY, selection_metrics=candidate_dev,
                    native_search_vector=vectors[nominee].detach().cpu(),
                    native_search_labels=list(spec.labels), selection_scope="development-only",
                )
                torch.save(payload, args.output_dir / name)
                report["selected_controller"] = name
    apply_vector(controller, base_bias, base_edge, zero, spec)
    report["status"] = "complete"
    save()
    print(json.dumps(dict(
        stage="complete", decision=report["decision"], nominee=nominee,
        followup_authorized=report["followup_authorized"],
    )), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
