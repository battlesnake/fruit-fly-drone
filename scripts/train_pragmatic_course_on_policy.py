#!/usr/bin/env python3
"""Bounded whole-flight native training with truncated physical gradients."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pragmatic_on_policy as online  # noqa: E402
import train_pragmatic_course_replay as replay  # noqa: E402
from audit_pragmatic_anticipation_step import restored_step  # noqa: E402
from pragmatic_closed_loop import stick_fields  # noqa: E402

from flydrone.gate import AnnularGate  # noqa: E402
from flydrone.hover import QuadState, StickState  # noqa: E402


def split_pairs(cases, gates):
    result = []
    for start in range(0, len(cases.side), 2):
        take = slice(start, start + 2)
        selected = tuple(AnnularGate(g.center[take], g.yaw[take]) for g in gates)
        result.append(
            (
                replace(
                    cases,
                    state=QuadState(*(v[take] for v in cases.state.as_tuple())),
                    sticks=StickState(*(v[take] for v in stick_fields(cases.sticks))),
                    gate=selected[0],
                    mass_scale=cases.mass_scale[take],
                    side=cases.side[take],
                    pair=cases.pair[take],
                ),
                selected,
            )
        )
    return result


def training_bank_seed(base, update, banks=0):
    """Predetermined course rotation, independent of source success or development."""
    if update < 1 or banks < 0:
        raise ValueError("positive update and nonnegative bank count required")
    return base + ((update - 1) % banks if banks else update - 1)


def should_select_development(
    metrics, best, first_floor, *, controller_change_update, last_evaluated_change_update
):
    """Do not present a better repeat of unchanged weights as a learned checkpoint."""
    return (
        controller_change_update > last_evaluated_change_update
        and metrics["clean_first_gate_pass_rate"] >= first_floor
        and replay.selection_score(metrics, first_floor) > replay.selection_score(best, first_floor)
    )


def combine_metrics(metrics):
    """Equal-size mirrored microbatches; counts add and per-episode losses average."""
    if not metrics or any(m["episodes"] != 2 for m in metrics):
        raise ValueError("expected nonempty two-episode microbatches")
    summed = (
        "episodes",
        "clean_completions",
        "clean_prefix_gates",
        "clean_first_passes",
        "ground_contacts",
        "invalid_episodes",
        "failed_episodes",
        "ring_contacts",
        "wrong_order",
        "wrong_direction",
    )
    averaged = (
        "objective",
        "continuous_objective",
        "tracking_loss",
        "nonroll_preservation_loss",
        "ground_clearance_loss",
        "discrete_objective",
    )
    result = {key: sum(m[key] for m in metrics) for key in summed}
    result.update({key: sum(m[key] for m in metrics) / len(metrics) for key in averaged})
    result["clean_first_by_side"] = [
        sum(m["clean_first_by_side"][side] for m in metrics) for side in (0, 1)
    ]
    result["pairs"] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--acceptance-mode", choices=("continuous", "flight-first"), default="continuous"
    )
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument(
        "--training-banks",
        type=int,
        default=0,
        help="zero draws new bank each update; otherwise rotate fixed seeds",
    )
    parser.add_argument("--development-updates", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--save-trial-proposals", action="store_true")
    parser.add_argument("--seed", type=int, default=1520983)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing on-policy run")
    if (
        min(
            args.updates,
            args.chunk_steps,
            args.learning_rate,
            args.training_pairs,
            args.seconds,
            args.development_pairs,
        )
        <= 0
        or args.updates > 10
        or args.training_banks < 0
        or min(args.development_updates) < 1
    ):
        raise SystemExit(
            "positive sizes/rates required; this bounded trial allows at most ten updates"
        )
    device = torch.device("cuda")
    controller, source = replay.load_controller(args, device)
    mask, manifest = replay.roll_preservation_mask(args.graph, device)
    manifest["supervision"] = "whole-flight native physical tracking with frozen nominal masks"
    controller.edge_magnitude.register_hook(lambda gradient: gradient * mask)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    development = replay.sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    options = dict(
        seconds=args.seconds,
        camera=camera,
        config=config,
        gate_config=gate_config,
        chunk_steps=args.chunk_steps,
    )

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

    started = perf_counter()
    args.output_dir.mkdir(parents=True)
    baseline = assess()
    best, best_update, rejections = baseline, 0, 0
    controller_change_update = last_evaluated_change_update = 0
    first_floor = max(0, baseline["clean_first_gate_pass_rate"] - 0.05)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    history = []

    def save(name, update, metrics, *, proposal_scale=None):
        payload = dict(source)
        payload.update(
            controller={
                key: value.detach().cpu() for key, value in controller.state_dict().items()
            },
            experiment="native-whole-flight-tbptt-v2",
            training_update=update,
            native_path_manifest=manifest,
            course_geometry=replay.GEOMETRY,
            gate_config=vars(gate_config),
            selection_metrics=metrics,
            training_source_sha256=digest,
            actor_privileged_inputs=False,
            deployed_extra_state=False,
            training_only_gradient_chunk_steps=args.chunk_steps,
            training_acceptance_mode=args.acceptance_mode,
            selection_scope="development"
            if proposal_scale is None
            else "training-only proposal, not promoted",
            training_proposal_scale=proposal_scale,
            nominal_tracking_rule="retain-first-unclean-expected-crossing-frame-stop-after",
        )
        torch.save(payload, args.output_dir / name)

    def report():
        data = dict(
            experiment="native-whole-flight-tbptt-v2",
            arguments={
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            source_sha256=digest,
            native_path_manifest=manifest,
            geometry=replay.GEOMETRY,
            source_development=baseline,
            selected_development=best,
            selected_update=best_update,
            controller_change_update=controller_change_update,
            last_evaluated_change_update=last_evaluated_change_update,
            history=history,
            elapsed_seconds=perf_counter() - started,
            actor_inputs=["320x200 RGB", "roll", "pitch"],
            actor_outputs="native foreleg pools -> physical forelegs -> sticks",
            optimizer_updates_during_flight=False,
            actor_privileged_inputs=False,
            deployed_extra_state=False,
            fixed_nominal_tracking_masks=True,
            nominal_tracking_rule="retain-first-unclean-expected-crossing-frame-stop-after",
        )
        (args.output_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")

    save("best-controller.pt", 0, baseline)
    report()
    print(
        json.dumps(
            dict(
                stage="baseline",
                clean=baseline["clean_course_success_rate"],
                first=baseline["clean_first_gate_pass_rate"],
            )
        ),
        flush=True,
    )
    for update in range(1, args.updates + 1):
        seed = training_bank_seed(args.seed, update, args.training_banks)
        bank = replay.sample_two_gate_cases(
            args.training_pairs, seed=seed, device=device, hover_config=config, **replay.GEOMETRY
        )
        pairs = split_pairs(*bank)
        nominal, references = [], []
        for cases, gates in pairs:
            metrics, trace, _ = online.fly_course(
                controller, cases, gates, **options, record_trace=True
            )
            nominal.append(metrics)
            references.append({key: value.to(device) for key, value in trace.items()})
        before_metrics = combine_metrics(nominal)
        (args.output_dir / f"nominal-u{update:03d}.json").write_text(
            json.dumps(dict(update=update, seed=seed, metrics=before_metrics), indent=2) + "\n"
        )
        print(
            json.dumps(dict(stage="nominal", update=update, seed=seed, metrics=before_metrics)),
            flush=True,
        )
        optimizer.zero_grad(set_to_none=True)
        gradient_runs = []
        for index, ((cases, gates), reference) in enumerate(zip(pairs, references, strict=True)):
            metrics, _, _ = online.fly_course(
                controller,
                cases,
                gates,
                **options,
                backward=True,
                gradient_scale=1 / len(pairs),
                reference_motors=reference["motors"],
                reference_active=reference["active"],
            )
            gradient_runs.append(metrics)
            print(
                json.dumps(
                    dict(
                        stage="gradient_pair",
                        update=update,
                        pair=index,
                        objective=metrics["objective"],
                    )
                ),
                flush=True,
            )
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
                trial_pairs = []
                for (cases, gates), reference in zip(pairs, references, strict=True):
                    metrics, _, _ = online.fly_course(
                        controller,
                        cases,
                        gates,
                        **options,
                        reference_motors=reference["motors"],
                        reference_active=reference["active"],
                    )
                    trial_pairs.append(metrics)
                metrics = combine_metrics(trial_pairs)
                keep = online.whole_flight_trial_admissible(
                    metrics, before_metrics, mode=args.acceptance_mode
                )
                trial_record = dict(scale=scale, metrics=metrics, accepted=keep, **projection)
                if args.save_trial_proposals:
                    name = f"trial-u{update:03d}-s{scale:g}.pt"
                    save(name, update, metrics, proposal_scale=scale)
                    trial_record["saved_actual_proposal"] = name
                trials.append(trial_record)
                if keep:
                    accepted = controller.edge_magnitude.detach().clone()
                    accepted_scale = scale
                print(
                    json.dumps(
                        dict(
                            stage="trial",
                            update=update,
                            scale=scale,
                            accepted=keep,
                            continuous=metrics["continuous_objective"],
                            clean=metrics["clean_completions"],
                            prefix=metrics["clean_prefix_gates"],
                        )
                    ),
                    flush=True,
                )
            if accepted is not None:
                break
        if accepted is not None:
            with torch.no_grad():
                controller.edge_magnitude.copy_(accepted)
            if not torch.equal(accepted, before):
                controller_change_update = update
            rejections = 0
        else:
            optimizer.load_state_dict(old_optimizer)
            rejections += 1
        entry = dict(
            update=update,
            seed=seed,
            nominal=before_metrics,
            gradient_flight=combine_metrics(gradient_runs),
            raw_gradient_norm=grad_norm,
            accepted_scale=accepted_scale,
            controller_change_update=controller_change_update,
            trials=trials,
            raw_gradient_dot_step=float(
                (raw_gradient * (controller.edge_magnitude.detach() - before)).sum()
            ),
        )
        stop = rejections >= 3
        if update in args.development_updates or update == args.updates or stop:
            metrics = assess()
            entry["development"] = metrics
            entry["development_repeats_unchanged_controller"] = (
                controller_change_update == last_evaluated_change_update
            )
            save("latest-controller.pt", update, metrics)
            if should_select_development(
                metrics,
                best,
                first_floor,
                controller_change_update=controller_change_update,
                last_evaluated_change_update=last_evaluated_change_update,
            ):
                best, best_update = metrics, update
                save("best-controller.pt", update, metrics)
            last_evaluated_change_update = controller_change_update
            if metrics["clean_first_gate_pass_rate"] < first_floor:
                stop = True
                entry["stop_reason"] = "material clean-first development regression"
            print(
                json.dumps(
                    dict(
                        stage="development",
                        update=update,
                        clean=metrics["clean_course_success_rate"],
                        first=metrics["clean_first_gate_pass_rate"],
                        controller_change_update=controller_change_update,
                        repeats_unchanged_controller=entry[
                            "development_repeats_unchanged_controller"
                        ],
                    )
                ),
                flush=True,
            )
        if rejections >= 3:
            entry["stop_reason"] = "three consecutive rejected updates"
        history.append(entry)
        report()
        if stop:
            break
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
