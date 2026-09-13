#!/usr/bin/env python3
"""Bounded on-policy roll imitation with refreshed, whole-approach coverage.

Privileged local roll labels and frozen-source PYT are training-only. The actor
and autonomous evaluator retain the original visual/attitude -> foreleg interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pragmatic_course_replay as replay  # noqa: E402
from audit_pragmatic_fixed_replay_fit import edge_anchor_loss  # noqa: E402


def coverage_plan(update):
    stage = ("launch", "middle", "late-approach", "early-approach", "middle", "late-approach")[
        (update - 1) % 6
    ]
    phase = 1 + (update - 1) % 4
    return [
        ("native", 0, stage),
        ("roll-assisted", 0, stage),
        ("native", phase, None),
        ("roll-assisted", phase, None),
    ]


def older_collection_updates(updates_per_round):
    """Retain fresh examples of every phase; avoid aliasing the four-phase cycle."""
    selected = set()
    for candidate in (4, 9):
        if candidate > updates_per_round:
            continue
        if any(
            other not in selected and other != candidate and (other - candidate) % 4 == 0
            for other in range(1, updates_per_round + 1)
        ):
            selected.add(candidate)
    return selected


def choose_lessons(banks, plan, unroll, rng, device):
    lessons = []
    for kind, phase, stage in plan:
        window, record = replay.select_balanced_window(
            banks[kind],
            banks["roll-assisted"],
            phase,
            unroll,
            rng,
            "approach",
            device,
            early_stage=stage,
        )
        record.update(requested_source=kind, roll_supervision="local-current-gate-from-start")
        lessons.append((window, record))
    return lessons


@torch.no_grad()
def fit_summary(controller, lessons, unroll, camera, gate_config, contrast_weight):
    """Measure by actual source/phase/side; never mislabel fallback as native."""
    records = []
    for start in range(0, len(lessons), 4):
        group = lessons[start : start + 4]
        window = replay.combine_replay_columns([w for w, _ in group])
        _, _, residual = replay.replay_window_loss(
            controller, window, unroll, camera, gate_config, contrast_weight, diagnostics=True
        )
        rmse = residual.square().mean(0).sqrt().cpu().tolist()
        for index, (_, record) in enumerate(group):
            records.append(dict(**record, motor_rmse_by_side=rmse[2 * index : 2 * index + 2]))
    return dict(scope="fixed per-round training examples, not held-out flight", windows=records)


def backward_coverage_lessons(
    controller, lessons, initial, mask, *, unroll, camera, gate_config,
    contrast_weight, anchor_reference_count, full_prefix_gradient=False, pairs_per_batch=4,
):
    """Accumulate equal-pair lessons and one anchor before one optimizer step.

    Pair adjacency preserves the contrast loss. Microbatch boundaries change
    activation memory, not the selected examples, objective weights or update count.
    The caller owns zeroing, clipping, stepping and native weight projection.
    """
    if not lessons or pairs_per_batch < 1:
        raise ValueError("positive pair microbatch size and nonempty lessons required")
    losses, axes = [], []
    for start in range(0, len(lessons), pairs_per_batch):
        group = lessons[start : start + pairs_per_batch]
        window = replay.combine_replay_columns([w for w, _ in group])
        loss, axis = replay.replay_window_loss(
            controller, window, unroll, camera, gate_config, contrast_weight,
            full_prefix_gradient=full_prefix_gradient,
        )
        if not bool(loss.isfinite()):
            raise RuntimeError("nonfinite coverage objective; no update applied")
        weight = len(group) / len(lessons)
        (weight * loss).backward()
        losses.append(weight * loss.detach())
        axes.append(weight * axis.detach())
    anchor = edge_anchor_loss(controller.edge_magnitude, initial, mask, anchor_reference_count)
    if not bool(anchor.isfinite()):
        raise RuntimeError("nonfinite coverage anchor; no update applied")
    anchor.backward()
    return torch.stack(losses).sum(), torch.stack(axes).sum(0), anchor.detach()


def native_selection_score(metrics):
    """Overall clean success first; balance breaks ties, not a per-side veto."""
    return (
        metrics["clean_course_success_rate"],
        min(
            metrics["clean_course_negative_success_rate"],
            metrics["clean_course_positive_success_rate"],
        ),
        metrics["gates_before_failure_mean"],
        metrics["course_race_fitness"],
    )


def select_native_candidate(metrics, best, revision, last_assessed_revision):
    """Only new, ground/invalid-free native candidates can replace the selection."""
    return (
        revision > last_assessed_revision
        and metrics["ground_contact_rate"] == 0
        and metrics["invalid_rate"] == 0
        and native_selection_score(metrics) > native_selection_score(best)
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--updates-per-round", type=int, default=10)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--native-pairs", type=int, default=8)
    parser.add_argument("--assisted-pairs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--full-prefix-gradient", action="store_true")
    parser.add_argument(
        "--replay-pairs-per-batch", type=int, default=4,
        help="Accumulate paired lesson microbatches before each single optimizer step.",
    )
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--hop-budget", type=int, default=7)
    parser.add_argument("--anchor-reference-count", type=int, default=19286)
    parser.add_argument("--seed", type=int, default=2026091310)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = (
        args.rounds,
        args.updates_per_round,
        args.unroll,
        args.native_pairs,
        args.assisted_pairs,
        args.learning_rate,
        args.hop_budget,
        args.anchor_reference_count,
        args.development_pairs,
        args.seconds,
        args.replay_pairs_per_batch,
    )
    if not all(math.isfinite(value) and value > 0 for value in sizes):
        raise SystemExit("sizes and rates must be finite and positive")
    if not math.isfinite(args.contrast_weight) or args.contrast_weight < 0:
        raise SystemExit("contrast weight must be finite and nonnegative")
    collection_seeds = [args.seed + 20 * r + k for r in range(args.rounds) for k in (0, 1)]
    if args.development_seed in collection_seeds:
        raise SystemExit("training and development seeds must be separate")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing coverage experiment")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    reference = copy.deepcopy(controller).requires_grad_(False)
    mask, manifest = replay.roll_preservation_mask(args.graph, device, hop_budget=args.hop_budget)
    manifest["supervision"] = "current-gate roll throughout plus frozen-source PYT"
    controller.edge_magnitude.register_hook(lambda grad: grad * mask)
    initial = controller.edge_magnitude.detach().clone()
    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    development = replay.sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    rng = np.random.default_rng(args.seed)

    def assess():
        return replay.evaluate(
            controller,
            *development,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
        )

    baseline = assess()
    best, best_update = baseline, 0
    result = dict(
        experiment="refreshed-whole-approach-roll-imitation-v1",
        status="running",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_development=baseline,
        geometry=replay.GEOMETRY,
        native_path_manifest=manifest,
        roll_labels="unified",
        history=[],
        rounds=[],
        actor_inputs=["320x200 RGB", "roll", "pitch"],
        actor_outputs="native foreleg pools -> bilateral gimbals -> sticks",
        teacher_or_history_deployed=False,
        neural_hz=50,
        physics_hz=100,
        autonomous_goal_established=False,
        replay_gradient=dict(
            full_prefix=args.full_prefix_gradient,
            warmup_differentiated=args.full_prefix_gradient,
            activation_chunk_steps=20 if args.full_prefix_gradient else None,
            pairs_per_microbatch=args.replay_pairs_per_batch,
            observations_are_fixed_training_data=True,
        ),
        policy_revision=0,
    )

    def report():
        result.update(
            selected_development=best,
            selected_update=best_update,
            elapsed_seconds=perf_counter() - started,
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    def save(name, update, metrics):
        payload = dict(source)
        payload.update(
            controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
            experiment=result["experiment"],
            training_update=update,
            native_path_manifest=manifest,
            gate_config=vars(gate_config),
            course_geometry=replay.GEOMETRY,
            selection_metrics=metrics,
            preservation_source_checkpoint=str(args.checkpoint),
            roll_teacher="current-gate",
            roll_from_start=True,
            supervision="unified-roll",
            teacher_inputs_are_actor_inputs=False,
            diagnostic_only=True,
            replay_gradient=result["replay_gradient"],
            policy_revision=result["policy_revision"],
        )
        torch.save(payload, args.output_dir / name)

    save("best-controller.pt", 0, baseline)
    print(
        json.dumps(dict(stage="baseline", clean=baseline["clean_course_success_rate"])), flush=True
    )
    report()
    bank_history = []
    last_assessed_revision = 0
    for round_index in range(args.rounds):
        round_record = dict(round=round_index + 1, collections=[], status="collecting")
        result["rounds"].append(round_record)
        report()
        banks = {}
        for kind, pairs, offset in (
            ("native", args.native_pairs, 0),
            ("roll-assisted", args.assisted_pairs, 1),
        ):
            bank = replay.collect_bank(
                controller,
                pairs,
                args.seed + 20 * round_index + offset,
                kind,
                args.seconds,
                camera,
                config,
                gate_config,
                "rate-damped",
                reference,
                roll_teacher="current-gate",
                roll_from_start=True,
            )
            banks[kind] = bank
            round_record["collections"].append(bank.manifest())
            report()
        bank_history.append(banks)
        # Keep old exposures, but prefer the most recent native distribution.
        # On updates 4 and 9, sample an older round if one exists, without
        # removing the only fresh occurrence of a gate phase in a short round.
        probe_plan = [
            (kind, 0, stage)
            for stage in ("launch", "early-approach", "middle", "late-approach")
            for kind in ("native", "roll-assisted")
        ]
        probe_plan += [
            (kind, phase, None) for phase in range(1, 5) for kind in ("native", "roll-assisted")
        ]
        probes = choose_lessons(
            banks,
            probe_plan,
            args.unroll,
            np.random.default_rng(args.seed + 1000 + round_index),
            device,
        )
        round_record["fit_before"] = fit_summary(
            controller, probes, args.unroll, camera, gate_config, args.contrast_weight
        )
        round_record["status"] = "training"
        report()
        for local_update in range(1, args.updates_per_round + 1):
            update = round_index * args.updates_per_round + local_update
            bank_index = round_index
            if round_index and local_update in older_collection_updates(args.updates_per_round):
                bank_index = int(rng.integers(round_index))
            lessons = choose_lessons(
                bank_history[bank_index], coverage_plan(update), args.unroll, rng, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss, axes, anchor = backward_coverage_lessons(
                controller, lessons, initial, mask, unroll=args.unroll, camera=camera,
                gate_config=gate_config, contrast_weight=args.contrast_weight,
                anchor_reference_count=args.anchor_reference_count,
                full_prefix_gradient=args.full_prefix_gradient,
                pairs_per_batch=args.replay_pairs_per_batch,
            )
            gradient = torch.nn.utils.clip_grad_norm_(
                controller.parameters(), 1.0, error_if_nonfinite=True
            )
            before = controller.edge_magnitude.detach()[mask].clone()
            optimizer.step()
            controller.project_parameters()
            delta = controller.edge_magnitude.detach()[mask] - before
            changed = int((delta != 0).sum())
            result["policy_revision"] += int(changed > 0)
            entry = dict(
                update=update,
                collection_round=bank_index + 1,
                loss=float(loss.detach()),
                anchor=float(anchor.detach()),
                gradient_norm=float(gradient),
                changed_edges=changed,
                edge_delta_l2=float(delta.norm()),
                axis_losses=axes.tolist(),
                windows=[dict(**record, loss_weight=0.25) for _, record in lessons],
                elapsed_seconds=perf_counter() - started,
            )
            result["history"].append(entry)
            print(json.dumps(dict(stage="update", **entry)), flush=True)
            del loss, anchor, lessons
            report()
        round_record["fit_after"] = fit_summary(
            controller, probes, args.unroll, camera, gate_config, args.contrast_weight
        )
        del probes
        metrics = assess()
        round_record["development"] = metrics
        save(f"update-{update}.pt", update, metrics)
        if select_native_candidate(
            metrics, best, result["policy_revision"], last_assessed_revision
        ):
            best, best_update = metrics, update
            save("best-controller.pt", update, metrics)
        last_assessed_revision = result["policy_revision"]
        round_record["status"] = "complete"
        report()
        print(
            json.dumps(
                dict(
                    stage="development",
                    update=update,
                    clean=metrics["clean_course_success_rate"],
                    first=metrics["clean_first_gate_pass_rate"],
                    ground=metrics["ground_contact_rate"],
                )
            ),
            flush=True,
        )
    result["status"] = "complete"
    gain = (best["clean_course_success_rate"] - baseline["clean_course_success_rate"]) * (
        2 * args.development_pairs
    )
    result.update(
        development_extra_clean=round(gain),
        meaningful_development_nominee=(
            gain >= 4 - 1e-6 and best["ground_contact_rate"] == 0 and best["invalid_rate"] == 0
        ),
    )
    report()
    print(
        json.dumps(
            dict(
                stage="complete",
                selected_update=best_update,
                clean=best["clean_course_success_rate"],
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
