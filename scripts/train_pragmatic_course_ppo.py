#!/usr/bin/env python3
"""Bounded native-connectome PPO pilot: outcomes, not teacher actions.

Training-only exploration and privileged critic are absent from native evaluation
and export. The existing fly recurrence is the actor's only persistent state.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pragmatic_course_replay as replay  # noqa: E402
from pragmatic_box_natural import box_natural_actor_proposal  # noqa: E402
from pragmatic_policy_optimization import (  # noqa: E402
    OutcomeCritic,
    actor_proposal,
    eligible_native,
    fit_critic,
    replay_round,
    round_advantages,
    select_new_native,
    steepest_actor_proposal,
)
from pragmatic_policy_rollout import collect_policy_rollout  # noqa: E402
from pragmatic_sink_fisher import natural_sink_actor_proposal  # noqa: E402
from pragmatic_sink_policy import (  # noqa: E402
    NativeRollSinkPolicy,
    replay_sink_policy_gradient,
    verify_compiled_sink_policy,
    verify_sink_recording,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026091400)
    parser.add_argument("--noise-seed", type=int, default=2026091410)
    parser.add_argument("--development-seed", type=int, default=1110983)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--proposals", type=int, default=2)
    parser.add_argument("--training-pairs", type=int, default=16)
    parser.add_argument("--development-pairs", type=int, default=16)
    parser.add_argument("--development-every", type=int, default=1)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--actor-update", choices=("adam", "steepest", "natural", "box-natural"),
                        default="adam")
    parser.add_argument("--full-history", action="store_true",
                        help="Differentiate warmup and all recurrent history; checkpoint chunks.")
    parser.add_argument("--predicted-decrease", type=float, default=5e-5,
                        help="Loss target for steepest/natural; unused by KL-budgeted box-natural.")
    parser.add_argument("--actor-scope", choices=("full", "roll-sinks"), default="full")
    parser.add_argument("--failure-aware-advantages", action="store_true",
                        help="Zero-baseline uncentered safety returns on already-failed tails.")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.rounds, args.proposals, args.training_pairs, args.development_pairs,
           args.development_every, args.microbatch, args.chunk_steps, args.learning_rate) <= 0:
        raise SystemExit("positive sizes and learning rate required")
    if not math.isfinite(args.predicted_decrease) or args.predicted_decrease <= 0:
        raise SystemExit("finite positive predicted decrease required")
    if args.actor_update in ("steepest", "natural", "box-natural") and not args.full_history:
        raise SystemExit("steepest/natural/box-natural updates require --full-history")
    if args.actor_scope == "roll-sinks" and args.actor_update not in (
        "steepest", "natural", "box-natural",
    ):
        raise SystemExit("roll-sinks requires steepest/natural/box-natural and --full-history")
    if args.actor_update in ("natural", "box-natural") and args.actor_scope != "roll-sinks":
        raise SystemExit("natural updates require --actor-scope roll-sinks")
    if args.development_seed in range(args.seed, args.seed + args.rounds):
        raise SystemExit("training and development seeds must differ")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite an existing PPO experiment")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    controller.eval().requires_grad_(False)
    sink = NativeRollSinkPolicy(controller) if args.actor_scope == "roll-sinks" else None
    actor = sink if sink is not None else controller
    if sink is None:
        controller.edge_magnitude.requires_grad_(True)
        mask, manifest = replay.roll_preservation_mask(args.graph, device, hop_budget=7)
    else:
        mask = torch.ones_like(sink.edge_magnitude, dtype=torch.bool)
        manifest = dict(sink.sink.manifest(), scope="all existing roll-sink incoming edges",
                        recurrent_motor_update="direct, no filter-tail truncation")
    manifest["supervision"] = "joint correlated-action outcome PPO; no teacher"
    optimizer = (torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
                 if args.actor_update == "adam" else None)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    development = replay.sample_two_gate_cases(
        args.development_pairs, seed=args.development_seed, device=device,
        hover_config=config, **replay.GEOMETRY,
    )
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    result = dict(experiment="native-correlated-course-ppo-v1", status="running",
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
                  geometry=replay.GEOMETRY, seconds=30, native_path_manifest=manifest,
                  actor_inputs="RGB 320x200/125deg and roll/pitch only",
                  actor_memory="existing connectome recurrence only",
                  critic_is_training_only=True, exploration_is_training_only=True,
                  actor_update=args.actor_update, full_history=args.full_history,
                  actor_scope=args.actor_scope,
                  failure_aware_advantages=args.failure_aware_advantages,
                  local_kl_budget=.002 if args.actor_update == "box-natural" else None,
                  effective_microbatch=args.microbatch, oom_fallbacks=0,
                  selected_round=0, accepted_steps=0, policy_revision=0,
                  rounds=[], goal_verified=False)

    def report():
        result["elapsed_seconds"] = perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    def progress(event):
        print(json.dumps(dict(elapsed_seconds=perf_counter() - started, **event)), flush=True)

    def assess():
        if sink is not None:
            sink.compile_into(controller)
        return replay.evaluate(controller, *development, seconds=30, warmup_steps=10,
                               camera=camera, hover_config=config, gate_config=gate_config)

    def save_native(name, round_number, metrics):
        if sink is not None:
            sink.compile_into(controller)
        payload = dict(source)
        payload.update(controller={k: v.detach().cpu() for k, v in controller.state_dict().items()},
                       experiment=result["experiment"], training_round=round_number,
                       native_path_manifest=manifest, gate_config=vars(gate_config),
                       course_geometry=replay.GEOMETRY, selection_metrics=metrics,
                       selection_metrics_round=round_number if metrics is not None else None,
                       supervision="outcome PPO; no teacher", teacher_inputs_are_actor_inputs=False,
                       teacher_config=None, roll_teacher=None, preservation_source_checkpoint=None,
                       parent_checkpoint=str(args.checkpoint),
                       critic_is_training_only=True, exploration_is_training_only=True)
        payload.update(actor_update=args.actor_update, full_history=args.full_history,
                       actor_scope=args.actor_scope,
                       failure_aware_advantages=args.failure_aware_advantages)
        torch.save(payload, args.output_dir / name)

    report()
    try:
        baseline = assess()
        best = baseline
        result.update(source_development=baseline, best_development=best,
                      selected_controller=str(args.checkpoint))
        report()
        progress(dict(stage="source-development", metrics=baseline))
        critic = critic_optimizer = None
        last_evaluated_revision = 0
        for round_number in range(1, args.rounds + 1):
            entry = dict(round=round_number, seed=args.seed + round_number - 1,
                         noise_seed=args.noise_seed + round_number - 1, proposals=[])
            result["rounds"].append(entry)
            report()
            bank = replay.sample_two_gate_cases(
                args.training_pairs, seed=entry["seed"], device=device,
                hover_config=config, **replay.GEOMETRY,
            )
            collected_at = perf_counter()
            if sink is not None:
                sink.compile_into(controller)
            data = collect_policy_rollout(controller, *bank, camera=camera, config=config,
                                           gate_config=gate_config, noise_seed=entry["noise_seed"],
                                           record_sink_parents=sink.sink.parents if sink else None)
            entry.update(collection=data.metrics,
                         collection_wall_seconds=perf_counter()-collected_at)
            if critic is None:
                torch.manual_seed(args.seed)
                critic = OutcomeCritic(data.critic_features.shape[-1]).to(device)
                critic.initialize_normalization(data.critic_features[data.valid].to(device))
                critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
            advantages, entry["advantages"] = round_advantages(
                data, critic, zero_baseline=round_number == 1,
                failure_aware=args.failure_aware_advantages,
            )
            # Critic is fitted only AFTER the round's baseline and advantages freeze.
            entry["critic_fit"] = fit_critic(critic, critic_optimizer, data, seed=entry["seed"])
            entry.update(stationary_std=data.stationary_std, rho=data.rho)
            report()
            progress(dict(stage="collection-complete", round=round_number, metrics=data.metrics))
            if sink is not None:
                data = data.select(range(data.valid.shape[1]), device)
                advantages = advantages.to(device)
                entry["sink_recording_check"] = verify_sink_recording(sink, data)
                report()

            def replay_fn(backward, data=data, advantages=advantages):
                if sink is not None:
                    return replay_sink_policy_gradient(sink, data, advantages, backward=backward)
                try:
                    return replay_round(controller, data, advantages, camera=camera,
                                        gate_config=gate_config,
                                        microbatch=result["effective_microbatch"],
                                        chunk_steps=args.chunk_steps, backward=backward,
                                        progress=progress, full_history=args.full_history)
                except torch.cuda.OutOfMemoryError:
                    # One fallback, only while computing a gradient before learning
                    # has accepted any parameter update. Discard partial gradients.
                    if (not backward or result["accepted_steps"] != 0
                            or result["effective_microbatch"] <= 4):
                        raise
                    result["effective_microbatch"] = 4
                    result["oom_fallbacks"] += 1
                # Leave the exception handler before freeing the failed graph.
                controller.edge_magnitude.grad = None
                gc.collect()
                torch.cuda.empty_cache()
                report()
                progress(dict(stage="oom-fallback", microbatch=4))
                return replay_round(controller, data, advantages, camera=camera,
                                    gate_config=gate_config, microbatch=4,
                                    chunk_steps=args.chunk_steps, backward=backward,
                                    progress=progress, full_history=args.full_history)

            for proposal in range(1, args.proposals + 1):
                proposed_at = perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                if args.actor_update == "box-natural":
                    stats = box_natural_actor_proposal(sink, data, mask, replay_fn)
                elif args.actor_update == "natural":
                    stats = natural_sink_actor_proposal(
                        sink, data, mask, replay_fn, predicted_decrease=args.predicted_decrease,
                    )
                elif args.actor_update == "steepest":
                    stats = steepest_actor_proposal(
                        actor, mask, replay_fn, data.stationary_std,
                        predicted_decrease=args.predicted_decrease,
                    )
                else:
                    stats = actor_proposal(controller, optimizer, mask, replay_fn,
                                           data.stationary_std)
                stats.update(proposal=proposal, wall_seconds=perf_counter()-proposed_at)
                entry["proposals"].append(stats)
                result["accepted_steps"] += int(stats["accepted"])
                result["policy_revision"] += int(
                    stats["accepted"] and stats["proposed_edge_delta_l2"] > 0)
                if sink is not None and stats["accepted"] and "sink_compile_check" not in result:
                    sink.compile_into(controller)
                    result["sink_compile_check"] = verify_compiled_sink_policy(
                        controller, sink, data, camera=camera, gate_config=gate_config,
                    )
                report()
                progress(dict(stage="proposal-complete", round=round_number, **stats))
                if stats["stop_round"]:
                    break
            metrics = None
            entry["policy_revision"] = result["policy_revision"]
            if round_number % args.development_every == 0 or round_number == args.rounds:
                metrics = assess()
                entry["native_development"] = metrics
                # Development chooses an export, never resets continuing training.
                if select_new_native(metrics, best, result["policy_revision"],
                                     last_evaluated_revision):
                    best = metrics
                    result.update(best_development=best, selected_round=round_number,
                                  selected_controller=str(args.output_dir / "best-controller.pt"))
                    save_native("best-controller.pt", round_number, metrics)
                last_evaluated_revision = result["policy_revision"]
            # A recovery checkpoint without assessment must not inherit stale scores.
            save_native("last-controller.pt", round_number, metrics)
            torch.save(dict(actor_optimizer=optimizer.state_dict() if optimizer else None,
                            actor_update=args.actor_update, critic=critic.state_dict(),
                            failure_aware_advantages=args.failure_aware_advantages,
                            critic_optimizer=critic_optimizer.state_dict(), round=round_number),
                       args.output_dir / "training-state.pt")
            report()
            progress(dict(stage="native-development" if metrics is not None else
                          "round-complete-without-native-check",
                          round=round_number, metrics=metrics,
                          selected_round=result["selected_round"]))
        gain = best["clean_course_success_rate"] - baseline["clean_course_success_rate"]
        result.update(status="complete",
                      development_extra_clean=round(gain * 2*args.development_pairs),
                      meaningful_development_nominee=(gain * 2*args.development_pairs >= 4 - 1e-6
                                                       and eligible_native(best)))
        report()
        progress(dict(stage="complete", accepted_steps=result["accepted_steps"],
                      meaningful_development_nominee=result["meaningful_development_nominee"]))
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        report()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
