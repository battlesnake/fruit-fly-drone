#!/usr/bin/env python3
"""Alternate native premotor bearing learning with bounded native roll-sink PPO.

Privileged labels, frozen auxiliary decoder, teacher-assisted collection,
exploration and critic are TRAINING ONLY. Exports are ordinary full fly networks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pragmatic_premotor_bearing as representation  # noqa: E402
import train_pragmatic_course_ppo as ppo  # noqa: E402
import train_pragmatic_course_replay as replay  # noqa: E402
from pragmatic_centered_premotor import (  # noqa: E402
    MeanCompensatedPremotor,
    balanced_input_mean,
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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--representation-pairs", type=int, default=4)
    parser.add_argument("--representation-updates", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--center-premotor-inputs",
        action="store_true",
        help="Tie existing premotor biases to preserve reference mean input drive.",
    )
    parser.add_argument(
        "--reuse-head-run",
        type=Path,
        help="Reuse a completed run's frozen head/normalization; never refit.",
    )
    parser.add_argument("--training-pairs", type=int, default=16)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--proposals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026091600)
    parser.add_argument("--noise-seed", type=int, default=2026091630)
    parser.add_argument("--development-seed", type=int, default=1110983)
    return parser.parse_args()


def reuse_bearing_head(run, *, checkpoint, representation_manifest, device):
    """Controlled comparison: same source and same ordered anatomical readout."""
    previous = json.loads((run / "report.json").read_text())
    if previous["status"] != "complete":
        raise ValueError("head reuse requires a completed reference run")
    if Path(previous["arguments"]["checkpoint"]).resolve() != checkpoint.resolve():
        raise ValueError("head source checkpoint differs")
    previous_manifest = previous["native_path_manifest"]["representation_stage"]
    if previous_manifest["selected_body_ids"] != representation_manifest["selected_body_ids"]:
        raise ValueError("frozen head neuron identity/order differs")
    state = torch.load(
        run / "training-only-bearing-head.pt", map_location=device, weights_only=True
    )
    head = representation.FrozenBearingHead(**state)
    n = len(representation_manifest["selected_body_ids"])
    if (
        head.center.shape != (n,)
        or head.scale.shape != (n,)
        or head.weight.shape != (n, 2)
        or head.offset.shape != (2,)
        or head.source_mse.ndim != 0
        or not all(bool(v.isfinite().all()) for v in head.buffers())
        or not bool((head.scale > 0).all())
        or not bool(head.source_mse > 0)
    ):
        raise ValueError("invalid frozen bearing head")
    return head, dict(
        previous["bearing_head_fit"],
        reused_from_run=str(run),
        fit_metrics_are_from_original_run=True,
        refitted=False,
    )


def main():
    args = parse_args()
    if (
        min(
            args.rounds,
            args.representation_updates,
            args.training_pairs,
            args.development_pairs,
            args.proposals,
        )
        <= 0
        or args.representation_pairs < 2
        or not 0 < args.learning_rate < float("inf")
    ):
        raise SystemExit("positive sizes/rate and at least two representation pairs required")
    training_seeds = {args.seed + 3 * r + k for r in range(args.rounds) for k in range(3)}
    if args.development_seed in training_seeds:
        raise SystemExit("development seed must not be used for training")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite representation experiment")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    original, _ = replay.load_controller(args, device)
    controller.eval().requires_grad_(False)
    original.eval().requires_grad_(False)
    mask, nodes, representation_manifest = representation.premotor_mask(
        args.graph, args.annotations, device
    )
    source_input_nodes = (
        torch.unique(controller.edge_pre[mask]) if args.center_premotor_inputs else None
    )
    if args.center_premotor_inputs:
        representation_manifest.pop("outgoing_edges_biases_and_time_constants_frozen", None)
        representation_manifest.update(
            outgoing_edges_and_time_constants_frozen=True,
            bias_plasticity="existing selected biases tied to source-mean input compensation",
            biases_independently_optimized=False,
            centering_wrapper_is_deployed=False,
        )
    manifest = dict(
        representation_stage=representation_manifest,
        outcome_stage=dict(scope="all existing roll-sink incoming edges; not yet constructed"),
        total_plasticity="union of representation incoming mask and roll-sink incoming edges",
        outgoing_freeze_applies_only_during_representation=True,
    )
    if args.center_premotor_inputs:
        manifest["total_plasticity"] += "; plus constrained selected-premotor native biases"
    config = replay.HoverConfig(**source["hover_config"])
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    development = replay.sample_two_gate_cases(
        args.development_pairs,
        seed=args.development_seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    result = dict(
        experiment="premotor-bearing-native-outcome-v1",
        status="running",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        native_path_manifest=manifest,
        geometry=replay.GEOMETRY,
        seconds=30,
        actor_memory="existing full connectome recurrence only",
        auxiliary_head_is_deployed=False,
        privileged_labels_are_deployed=False,
        roll_consistency="round-entry native actor on same histories",
        pitch_yaw_throttle_consistency="original source on same histories",
        policy_revision=0,
        selected_round=0,
        rounds=[],
        goal_verified=False,
    )

    def report(stage, **event):
        result["elapsed_seconds"] = perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(dict(stage=stage, elapsed_seconds=result["elapsed_seconds"], **event)),
            flush=True,
        )

    def assess():
        return replay.evaluate(
            controller,
            *development,
            seconds=30,
            warmup_steps=10,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
        )

    def save_native(name, round_number, metrics):
        payload = dict(source)
        payload.update(
            controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
            experiment=result["experiment"],
            parent_checkpoint=str(args.checkpoint),
            training_round=round_number,
            native_path_manifest=manifest,
            gate_config=vars(gate_config),
            course_geometry=replay.GEOMETRY,
            selection_metrics=metrics,
            selection_metrics_round=round_number if metrics is not None else None,
            supervision="training-only bearing head then native sink outcome PPO",
            teacher_inputs_are_actor_inputs=False,
            auxiliary_head_is_deployed=False,
            critic_is_training_only=True,
            exploration_is_training_only=True,
            teacher_config=None,
            roll_teacher=None,
            preservation_source_checkpoint=None,
        )
        torch.save(payload, args.output_dir / name)

    report("initialized")
    try:
        baseline = best = assess()
        result.update(
            source_development=baseline,
            best_development=baseline,
            selected_controller=str(args.checkpoint),
        )
        report("source-development", metrics=baseline)
        head = critic = critic_optimizer = centered = None
        if args.reuse_head_run is not None:
            head, result["bearing_head_fit"] = reuse_bearing_head(
                args.reuse_head_run,
                checkpoint=args.checkpoint,
                representation_manifest=representation_manifest,
                device=device,
            )
            torch.save(head.state_dict(), args.output_dir / "training-only-bearing-head.pt")
            report("bearing-head-reused", metrics=result["bearing_head_fit"])
        last_revision = 0
        for number in range(1, args.rounds + 1):
            base_seed = args.seed + 3 * (number - 1)
            entry = dict(
                round=number,
                representation_seeds=[base_seed, base_seed + 1],
                outcome_seed=base_seed + 2,
                noise_seed=args.noise_seed + number - 1,
                representation_updates=[],
                proposals=[],
            )
            result["rounds"].append(entry)
            controller.requires_grad_(False)
            banks = []
            for kind, seed in zip(
                ("native", "roll-assisted"), entry["representation_seeds"], strict=True
            ):
                banks.append(
                    representation.collect_bearing_bank(
                        controller,
                        original,
                        nodes,
                        pairs=args.representation_pairs,
                        seed=seed,
                        kind=kind,
                        camera=camera,
                        config=config,
                        gate_config=gate_config,
                        presynaptic_nodes=source_input_nodes if centered is None else None,
                    )
                )
                entry["representation_collection"] = [b.metrics for b in banks]
                report("representation-collection", round=number, metrics=banks[-1].metrics)
            if head is None:
                head, result["bearing_head_fit"] = representation.fit_bearing_head(
                    banks,
                    device=device,
                    seed=args.seed,
                )
                torch.save(head.state_dict(), args.output_dir / "training-only-bearing-head.pt")
                report("bearing-head-fit", metrics=result["bearing_head_fit"])
            if args.center_premotor_inputs and centered is None:
                mean, result["source_input_mean"] = balanced_input_mean(banks, source_input_nodes)
                centered = MeanCompensatedPremotor(controller, mask, source_input_nodes, mean)
                torch.save(
                    dict(
                        nodes=source_input_nodes.cpu(), mean=mean.cpu(), **centered.training_state()
                    ),
                    args.output_dir / "training-only-centering.pt",
                )
                report("source-input-mean-frozen", metrics=result["source_input_mean"])
            training_actor = centered if centered is not None else controller
            rng = torch.Generator().manual_seed(base_seed)
            holdout_rng = torch.Generator().manual_seed(base_seed + 10000)
            first_holdout = representation.choose_windows(banks, holdout_rng, 0, heldout=True)
            heldout_windows = [first_holdout] + [
                representation.choose_windows(
                    banks,
                    holdout_rng,
                    i,
                    heldout=True,
                )
                for i in range(1, first_holdout[3]["available_paired_groups"])
            ]

            @torch.no_grad()
            def heldout_losses(head=head, windows=heldout_windows, actor=training_actor):
                return [
                    dict(
                        selection=selection,
                        **representation.bearing_window_loss(
                            actor,
                            head,
                            nodes,
                            bank,
                            rows,
                            starts,
                            camera=camera,
                            gate_config=gate_config,
                        )[1],
                        **representation.source_window_bearing(head, bank, rows, starts),
                    )
                    for bank, rows, starts, selection in windows
                ]

            entry["heldout_before"] = heldout_losses()
            controller.edge_magnitude.requires_grad_(True)
            optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
            for update in range(args.representation_updates):
                bank, rows, starts, selection = representation.choose_windows(banks, rng, update)
                stats = representation.representation_update(
                    controller,
                    optimizer,
                    mask,
                    lambda head=head, bank=bank, rows=rows, starts=starts, actor=training_actor: (
                        representation.bearing_window_loss(
                            actor,
                            head,
                            nodes,
                            bank,
                            rows,
                            starts,
                            camera=camera,
                            gate_config=gate_config,
                        )
                    ),
                )
                if centered is not None:
                    stats.update(centered.compile_bias())
                stats.update(update=update + 1, selection=selection)
                entry["representation_updates"].append(stats)
                result["policy_revision"] += int(stats["edge_delta_l2"] > 0)
                report("representation-update", round=number, **stats)
            controller.requires_grad_(False)
            entry["heldout_after"] = heldout_losses()
            save_native("post-representation-controller.pt", number, None)
            del optimizer, banks, heldout_windows, heldout_losses, bank, first_holdout
            report("representation-complete", round=number, heldout=entry["heldout_after"])

            # New slice and fresh full-native cache AFTER upstream changes.
            sink = ppo.NativeRollSinkPolicy(controller)
            manifest["outcome_stage"] = dict(
                sink.sink.manifest(),
                scope="all existing roll-sink incoming edges; upstream frozen in this phase",
            )
            cases = replay.sample_two_gate_cases(
                args.training_pairs,
                seed=entry["outcome_seed"],
                device=device,
                hover_config=config,
                **replay.GEOMETRY,
            )
            data = ppo.collect_policy_rollout(
                controller,
                *cases,
                camera=camera,
                config=config,
                gate_config=gate_config,
                noise_seed=entry["noise_seed"],
                record_sink_parents=sink.sink.parents,
            )
            entry["outcome_collection"] = data.metrics
            if critic is None:
                torch.manual_seed(args.seed)
                critic = ppo.OutcomeCritic(data.critic_features.shape[-1]).to(device)
                critic.initialize_normalization(data.critic_features[data.valid].to(device))
                critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
            advantages, entry["advantages"] = ppo.round_advantages(
                data, critic, zero_baseline=number == 1, failure_aware=True
            )
            entry["critic_fit"] = ppo.fit_critic(
                critic, critic_optimizer, data, seed=entry["outcome_seed"]
            )
            data = data.select(range(data.valid.shape[1]), device)
            advantages = advantages.to(device)
            entry["sink_recording_check"] = ppo.verify_sink_recording(sink, data)
            report("outcome-collection", round=number, metrics=entry["outcome_collection"])
            sink_mask = torch.ones_like(sink.edge_magnitude, dtype=torch.bool)

            def replay_fn(backward, sink=sink, data=data, advantages=advantages):
                return ppo.replay_sink_policy_gradient(sink, data, advantages, backward=backward)

            for proposal in range(args.proposals):
                stats = ppo.box_natural_actor_proposal(sink, data, sink_mask, replay_fn)
                entry["proposals"].append(stats)
                result["policy_revision"] += int(
                    stats["accepted"] and stats["proposed_edge_delta_l2"] > 0
                )
                if stats["accepted"] and "sink_compile_check" not in result:
                    sink.compile_into(controller)
                    result["sink_compile_check"] = ppo.verify_compiled_sink_policy(
                        controller, sink, data, camera=camera, gate_config=gate_config
                    )
                report("outcome-proposal", round=number, proposal=proposal + 1, **stats)
                if stats["stop_round"]:
                    break
            sink.compile_into(controller)
            del sink, data, advantages, replay_fn
            metrics = assess()
            entry["native_development"] = metrics
            if ppo.select_new_native(metrics, best, result["policy_revision"], last_revision):
                best = metrics
                result.update(
                    best_development=best,
                    selected_round=number,
                    selected_controller=str(args.output_dir / "best-controller.pt"),
                )
                save_native("best-controller.pt", number, metrics)
            last_revision = result["policy_revision"]
            save_native("last-controller.pt", number, metrics)
            torch.save(
                dict(
                    round=number,
                    bearing_head=head.state_dict(),
                    critic=critic.state_dict(),
                    critic_optimizer=critic_optimizer.state_dict(),
                    centering=centered.training_state() if centered is not None else None,
                ),
                args.output_dir / "training-state.pt",
            )
            report("native-development", round=number, metrics=metrics)
        gain = (
            (best["clean_course_success_rate"] - baseline["clean_course_success_rate"])
            * 2
            * args.development_pairs
        )
        result.update(
            status="complete",
            development_extra_clean=round(gain),
            meaningful_development_nominee=gain >= 4 - 1e-6 and ppo.eligible_native(best),
        )
        report("complete", meaningful_development_nominee=result["meaningful_development_nominee"])
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        report("failed", error=result["error"])
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
