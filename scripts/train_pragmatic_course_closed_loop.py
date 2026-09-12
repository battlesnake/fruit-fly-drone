#!/usr/bin/env python3
"""Train existing fly pathways through short closed-loop physical course segments."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pragmatic_closed_loop as physical  # noqa: E402
import train_pragmatic_course_replay as replay  # noqa: E402
from audit_pragmatic_anticipation_step import restored_step  # noqa: E402
from audit_pragmatic_closed_loop_step import (  # noqa: E402
    load_context,
    verify_saved_action_boundary,
)
from pragmatic_anticipation_lessons import early_training_banks  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1420983)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing physical-training run")
    if (
        min(
            args.steps,
            args.updates,
            args.interval,
            args.learning_rate,
            args.development_pairs,
            args.seconds,
        )
        <= 0
    ):
        raise SystemExit("sizes, intervals, learning rate and flight duration must be positive")
    controller, source, bank, cases, config, gates, camera, mask, manifest, digest = load_context(
        args
    )
    device = controller.bias.device
    rng = np.random.default_rng(args.seed)
    lessons, boundaries = [], []
    for phase in (1, 2, 3):
        rows, starts = physical.select_physical_window(bank, phase, args.steps, rng)
        lesson = physical.make_physical_lesson(
            bank, cases, rows, starts, args.steps, config, device
        )
        boundaries.append(verify_saved_action_boundary(lesson, args.steps, config))
        lessons.append(lesson)
    early_bank = early_training_banks([bank])[0]
    development = replay.sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    initial = controller.edge_magnitude.detach().clone()
    history = []
    started = perf_counter()
    args.output_dir.mkdir(parents=True)

    def assess():
        return replay.evaluate(
            controller,
            *development,
            seconds=args.seconds,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gates,
        )

    baseline = assess()
    best, best_update = baseline, 0
    first_floor = max(0, baseline["first_gate_pass_rate"] - 0.05)

    def save(name, update, metrics):
        payload = dict(source)
        payload.update(
            controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
            experiment="native-short-closed-loop-course-v1",
            training_update=update,
            native_path_manifest=manifest,
            course_geometry=replay.GEOMETRY,
            gate_config=vars(gates),
            selection_metrics=metrics,
            physical_training_source_sha256=digest,
            training_only_physical_lookahead_seconds=args.steps / 50,
            actor_privileged_inputs=False,
            deployed_extra_state=False,
            training_lessons=[item.record for item in lessons],
        )
        torch.save(payload, args.output_dir / name)

    def report():
        data = dict(
            experiment="native-short-closed-loop-course-v1",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            source_sha256=digest,
            native_path_manifest=manifest,
            geometry=replay.GEOMETRY,
            physical_boundary_checks=boundaries,
            lessons=[item.record for item in lessons],
            source_development=baseline,
            selected_development=best,
            selected_update=best_update,
            history=history,
            elapsed_seconds=perf_counter() - started,
            actor_inputs=["320x200 RGB", "roll", "pitch"],
            actor_outputs="native foreleg pools -> physical forelegs -> sticks",
            physical_lookahead_or_ground_truth_deployed=False,
        )
        (args.output_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")

    def anchor():
        return 1e-3 * ((controller.edge_magnitude[mask] - initial[mask]) / 0.02).square().mean()

    save("best-controller.pt", 0, baseline)
    report()
    print(
        json.dumps(
            dict(
                stage="baseline",
                clean=baseline["clean_course_success_rate"],
                first=baseline["first_gate_pass_rate"],
            )
        ),
        flush=True,
    )
    for update in range(1, args.updates + 1):
        lesson = lessons[(update - 1) % len(lessons)]
        _, rows, starts, early_record = replay.select_pair_window(
            early_bank, early_bank, 0, 20, rng, "approach"
        )
        early = replay.prepare_window(early_bank, rows, starts, 20, device)

        def early_loss(early_window=early):
            return replay.replay_window_loss(controller, early_window, 20, camera, gates, 1.0)[0]

        optimizer.zero_grad(set_to_none=True)
        preservation = early_loss()
        early_before = float(preservation.detach())
        (0.5 * preservation).backward()
        del preservation
        tracking, before_metrics = physical.physical_rollout_loss(
            controller, lesson, args.steps, camera, config, gates
        )
        objective_before = 0.5 * (early_before + float(tracking.detach())) + float(
            anchor().detach()
        )
        (0.5 * tracking).backward()
        del tracking
        anchor().backward()
        raw_gradient = controller.edge_magnitude.grad.detach().clone()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0, error_if_nonfinite=True)
        )
        before = controller.edge_magnitude.detach().clone()
        old_optimizer = copy.deepcopy(optimizer.state_dict())
        optimizer.step()
        displacement = controller.edge_magnitude.detach() - before
        with torch.no_grad():
            controller.edge_magnitude.copy_(before)
        trials, accepted, accepted_scale = [], None, 0.0
        for scale in (1.0, 0.3, 0.1):
            with restored_step(
                controller.edge_magnitude, before, displacement, scale
            ) as projection:
                with torch.no_grad():
                    trial_early = float(early_loss())
                    _, metrics = physical.physical_rollout_loss(
                        controller, lesson, args.steps, camera, config, gates
                    )
                    objective = 0.5 * (trial_early + metrics["loss"]) + float(anchor())
                    admissible = physical.admissible_tracking_trial(metrics)
                    keep = (
                        admissible
                        and objective < objective_before - 1e-7
                        and metrics["tracking_loss"] < before_metrics["tracking_loss"] - 1e-7
                    )
                    trials.append(
                        dict(
                            scale=scale,
                            objective=objective,
                            early_loss=trial_early,
                            physical=metrics,
                            admissible=admissible,
                            accepted=keep,
                            **projection,
                        )
                    )
                    if keep:
                        accepted = controller.edge_magnitude.detach().clone()
                        accepted_scale = scale
            if accepted is not None:
                break
        if accepted is not None:
            with torch.no_grad():
                controller.edge_magnitude.copy_(accepted)
        else:
            optimizer.load_state_dict(old_optimizer)
        entry = dict(
            update=update,
            lesson=lesson.record,
            early_window=early_record,
            objective_before=objective_before,
            early_loss_before=early_before,
            physical_before=before_metrics,
            raw_gradient_norm=grad_norm,
            accepted_scale=accepted_scale,
            trials=trials,
            raw_gradient_dot_step=float(
                (raw_gradient * (controller.edge_magnitude.detach() - before)).sum()
            ),
        )
        if update % args.interval == 0 or update == args.updates:
            metrics = assess()
            entry["development"] = metrics
            save("latest-controller.pt", update, metrics)
            if metrics["first_gate_pass_rate"] >= first_floor and replay.selection_score(
                metrics, first_floor
            ) > replay.selection_score(best, first_floor):
                best, best_update = metrics, update
                save("best-controller.pt", update, metrics)
            print(
                json.dumps(
                    dict(
                        stage="development",
                        update=update,
                        clean=metrics["clean_course_success_rate"],
                        first=metrics["first_gate_pass_rate"],
                    )
                ),
                flush=True,
            )
        history.append(entry)
        report()
        print(
            json.dumps(
                dict(
                    stage="update",
                    update=update,
                    accepted_scale=accepted_scale,
                    objective_before=objective_before,
                    objective_after=trials[-1]["objective"]
                    if accepted is not None
                    else objective_before,
                    physical_loss_before=before_metrics["tracking_loss"],
                    physical_loss_after=(
                        trials[-1]["physical"]["tracking_loss"]
                        if accepted is not None
                        else before_metrics["tracking_loss"]
                    ),
                )
            ),
            flush=True,
        )
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
