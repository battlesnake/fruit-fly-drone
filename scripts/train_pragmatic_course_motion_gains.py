#!/usr/bin/env python3
"""Test 32 shared visual motion-path gains with native course-flight selection."""

from __future__ import annotations

import argparse
import copy
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

import train_pragmatic_course_replay as replay  # noqa: E402
from search_pragmatic_gate_course_es import selection_score  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.motion_gains import (  # noqa: E402
    attach_motion_gains,
    compiled_controller_state,
    load_motion_edge_groups,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402


def continuation_check(baseline, candidate, early_limit=0.008):
    """A small diagnostic screen, not a course-completion or capability claim."""
    before = np.asarray(baseline["late_roll_rmse_by_side"])
    after = np.asarray(candidate["late_roll_rmse_by_side"])
    early = np.asarray(candidate["early_roll_rmse_by_side"])
    return bool(
        np.isfinite(after).all()
        and np.isfinite(early).all()
        and (after < before).all()
        and (early <= early_limit).all()
    )


def improvement_allowed(candidate, retained, first_floor):
    return candidate["first_gate_pass_rate"] >= first_floor and selection_score(
        candidate, first_floor
    ) > selection_score(retained, first_floor)


def motor_residual_summary(residuals, roles):
    """Split by each frame's actual gate phase, including transition windows."""
    errors, phases = torch.cat(residuals), torch.cat(roles)
    if errors.shape[:2] != phases.shape or not bool(torch.isfinite(errors).all()):
        raise ValueError("invalid diagnostic residuals / phases")
    early_rmse, late_rmse, early_counts, late_counts = [], [], [], []
    for row in range(errors.shape[1]):
        early, late = phases[:, row] == 0, phases[:, row] >= 1
        if not bool(early.any()) or not bool(late.any()):
            raise ValueError("diagnostics need both early and late frames on each side")
        early_counts.append(int(early.sum()))
        late_counts.append(int(late.sum()))
        early_rmse.append(float(errors[early, row, 0].square().mean().sqrt()))
        late_rmse.append(float(errors[late, row, 0].square().mean().sqrt()))
    return dict(
        side_order=["negative", "positive"],
        early_roll_rmse_by_side=early_rmse,
        late_roll_rmse_by_side=late_rmse,
        nonroll_rmse_by_side=errors[:, :, 1:].square().mean(dim=(0, 2)).sqrt().tolist(),
        early_frames_by_side=early_counts,
        late_frames_by_side=late_counts,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=REPO_ROOT
        / "data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--native-pairs", type=int, default=8)
    parser.add_argument("--teacher-pairs", type=int, default=4)
    parser.add_argument("--validation-pairs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1180983)
    parser.add_argument("--validation-seed", type=int, default=1210983)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--development-pairs", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    sizes = (
        args.updates,
        args.interval,
        args.unroll,
        args.native_pairs,
        args.teacher_pairs,
        args.validation_pairs,
        args.learning_rate,
        args.seconds,
        args.development_pairs,
    )
    if not all(np.isfinite(x) and x > 0 for x in sizes):
        raise SystemExit("sizes and learning rate must be finite and positive")
    if not np.isfinite(args.contrast_weight) or args.contrast_weight < 0:
        raise SystemExit("contrast weight must be finite and nonnegative")
    train_seeds = {args.seed, args.seed + 100000}
    validation_seeds = {args.validation_seed, args.validation_seed + 100000}
    if train_seeds & validation_seeds or args.development_seed in train_seeds | validation_seeds:
        raise SystemExit("collection, validation and development seeds must be disjoint")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing experiment directory")
    device = torch.device(args.device)
    controller, source = load_controller(args, device)
    reference = copy.deepcopy(controller).requires_grad_(False)
    indices, groups, manifest = load_motion_edge_groups(args.graph, args.annotations, device)
    module = attach_motion_gains(controller, indices, groups)
    optimizer = torch.optim.Adam([module.gains], lr=args.learning_rate)
    config = HoverConfig(**source["hover_config"])
    gate_config = replace(GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(
        width=source["image_resolution"][0],
        height=source["image_resolution"][1],
        horizontal_fov_degrees=source["camera_hfov_degrees"],
    )
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
    history, collections = [], []

    def collect(kind, pairs, seed, learner=controller):
        bank = replay.collect_bank(
            learner,
            pairs,
            seed,
            kind,
            args.seconds,
            camera,
            config,
            gate_config,
            "rate-damped",
            reference,
        )
        collections.append(bank.manifest())
        return bank

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

    def save(name, update, metrics):
        state = compiled_controller_state(controller)
        if set(state) != set(source["controller"]):
            raise RuntimeError("compiled checkpoint schema changed")
        for key, original in source["controller"].items():
            if key != "edge_magnitude" and not torch.equal(state[key], original):
                raise RuntimeError(f"frozen controller tensor changed: {key}")
        frozen_mask = torch.ones_like(state["edge_magnitude"], dtype=torch.bool)
        frozen_mask[indices.cpu()] = False
        if not torch.equal(
            state["edge_magnitude"][frozen_mask],
            source["controller"]["edge_magnitude"][frozen_mask],
        ):
            raise RuntimeError("unselected synapses changed")
        payload = dict(source)
        payload.update(
            controller=state,
            experiment="native-course-motion-shared-gains-v1",
            training_update=update,
            selection_metrics=metrics,
            motion_gain_manifest=manifest,
            motion_gains=module.gains.detach().cpu().tolist(),
            preservation_source_checkpoint=str(args.checkpoint),
            teacher_inputs_are_actor_inputs=False,
            gate_config=vars(gate_config),
            course_geometry=replay.GEOMETRY,
        )
        torch.save(payload, args.output_dir / name)

    baseline = assess()
    best, best_update = baseline, 0
    first_floor = max(0.0, baseline["first_gate_pass_rate"] - 0.05)
    save("best-controller.pt", 0, baseline)
    print(json.dumps(dict(stage="baseline", metrics=baseline, motion_groups=manifest)), flush=True)
    teacher_bank = collect("roll-assisted", args.teacher_pairs, args.seed + 100000)
    native_bank = collect("native", args.native_pairs, args.seed)
    validation_native = collect("native", args.validation_pairs, args.validation_seed, reference)
    validation_assisted = collect(
        "roll-assisted", args.validation_pairs, args.validation_seed + 100000, reference
    )
    validation_rng = np.random.default_rng(args.validation_seed)
    validation_windows, validation_records = [], []
    for phase in range(5):
        bank, rows, starts, record = replay.select_pair_window(
            validation_native,
            validation_assisted,
            phase,
            args.unroll,
            validation_rng,
            "approach",
        )
        validation_windows.append(replay.prepare_window(bank, rows, starts, args.unroll, device))
        validation_records.append(record)

    @torch.no_grad()
    def diagnostics():
        residuals, phases = [], []
        for window in validation_windows:
            _, _, residual = replay.replay_window_loss(
                controller,
                window,
                args.unroll,
                camera,
                gate_config,
                args.contrast_weight,
                diagnostics=True,
            )
            if not bool(torch.isfinite(residual).all()):
                raise RuntimeError("nonfinite validation motor residual")
            residuals.append(residual)
            starts = window[-1]
            times = starts[None] + torch.arange(args.unroll, device=device)[:, None]
            phases.append(window[2][times, torch.arange(len(starts), device=device)[None]])
        return motor_residual_summary(residuals, phases)

    initial_diagnostics = diagnostics()
    stopping_reason = "requested update budget"

    def report():
        result = dict(
            experiment="native-course-motion-shared-gains-v1",
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            native_path_manifest=manifest,
            source_development=baseline,
            selected_development=best,
            selected_update=best_update,
            history=history,
            source_validation=initial_diagnostics,
            validation_windows=validation_records,
            collections=collections,
            stopping_reason=stopping_reason,
            elapsed_seconds=perf_counter() - started,
            geometry=replay.GEOMETRY,
            actor_inputs=["320x200 RGB", "roll", "pitch"],
            actor_outputs="native foreleg pools -> physical forelegs -> sticks",
            replay_teacher_or_parameterization_deployed=False,
        )
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(dict(stage="source_validation", **initial_diagnostics)), flush=True)
    report()
    for update in range(1, args.updates + 1):
        optimizer.zero_grad(set_to_none=True)
        records, losses = [], []
        kind = ("approach", "transition", "pre-failure")[(update - 1) % 3]
        for phase in (0, 1 + (update - 1) % 4):
            bank, rows, starts, record = replay.select_pair_window(
                native_bank, teacher_bank, phase, args.unroll, rng, kind
            )
            window = replay.prepare_window(bank, rows, starts, args.unroll, device)
            # No parametrization cache crosses the no-grad prefix into this window.
            loss, axes = replay.replay_window_loss(
                controller, window, args.unroll, camera, gate_config, args.contrast_weight
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite motion gain loss")
            (loss * 0.5).backward()
            records.append(record)
            losses.append(dict(total=float(loss.detach()), axes=axes.tolist()))
            del loss, window
        anchor = 1e-3 * (module.gains - 1).square().mean()
        anchor.backward()
        gradient = torch.nn.utils.clip_grad_norm_([module.gains], 1.0, error_if_nonfinite=True)
        if update == 1 and float(gradient) == 0:
            raise RuntimeError("shared motion gains have no learning gradient")
        optimizer.step()
        module.project_parameters()
        entry = dict(
            update=update,
            windows=records,
            losses=losses,
            gradient_norm=float(gradient),
            gains=module.gains.detach().cpu().tolist(),
            elapsed_seconds=perf_counter() - started,
        )
        stop = False
        if update % args.interval == 0 or update == args.updates:
            validation = diagnostics()
            metrics = assess()
            entry.update(validation=validation, development=metrics)
            save("latest-controller.pt", update, metrics)
            if improvement_allowed(metrics, best, first_floor):
                best, best_update = metrics, update
                save("best-controller.pt", update, metrics)
            stop = not continuation_check(initial_diagnostics, validation)
            if stop:
                stopping_reason = "paired late-roll improvement / early preservation screen failed"
            print(
                json.dumps(
                    dict(
                        stage="development",
                        update=update,
                        clean=metrics["clean_course_success_rate"],
                        first=metrics["first_gate_pass_rate"],
                        validation=validation,
                        continue_screen_passed=not stop,
                    )
                ),
                flush=True,
            )
        history.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "development"}), flush=True)
        report()
        if stop:
            break
        if update % args.interval == 0 and update < args.updates:
            native_bank = collect("native", args.native_pairs, args.seed + update)
    print(
        json.dumps(dict(stage="complete", selected_update=best_update, reason=stopping_reason)),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
