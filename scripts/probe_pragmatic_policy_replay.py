#!/usr/bin/env python3
"""Measure real native rollout collection and a bounded GPU gradient replay.

No optimizer, weight changes, candidate export or goal-validation claim. A short
replay prefix measures practical cost; it is not a full-flight learning result.
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

import train_pragmatic_course_replay as replay  # noqa: E402
from pragmatic_policy_rollout import collect_policy_rollout  # noqa: E402
from pragmatic_recurrent_policy_gradient import replay_joint_policy_gradient  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--replay-frames", type=int, default=100)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026091395)
    parser.add_argument("--noise-seed", type=int, default=2026091396)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite a policy replay probe")
    if not 1 <= args.replay_frames <= 1500 or args.pairs < 1 or args.chunk_steps < 1:
        raise SystemExit("need positive sizes and at most one 30-second flight prefix")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    bank = replay.sample_two_gate_cases(args.pairs, seed=args.seed, device=device,
                                         hover_config=config, **replay.GEOMETRY)
    result = dict(
        experiment="native-correlated-policy-replay-cost-probe-v1", status="running",
        checkpoint=str(args.checkpoint), seed=args.seed, noise_seed=args.noise_seed,
        geometry=replay.GEOMETRY, episodes=2 * args.pairs, collection_seconds=30,
        replay_frames=args.replay_frames, chunk_steps=args.chunk_steps,
        optimizer_steps=0, actor_weights_changed=False,
        scope="training implementation/cost probe, not learned flight or goal validation",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    save()
    synchronize()
    started = perf_counter()
    data = collect_policy_rollout(controller, *bank, camera=camera, config=config,
                                   gate_config=gate_config, noise_seed=args.noise_seed)
    synchronize()
    result.update(collection_wall_seconds=perf_counter() - started,
                  collection_metrics=data.metrics,
                  critic_feature_count=data.critic_features.shape[-1],
                  collection_status="complete")
    save()
    print(json.dumps(dict(stage="collection-complete", seconds=result["collection_wall_seconds"],
                          metrics=data.metrics)), flush=True)

    mask, manifest = replay.roll_preservation_mask(args.graph, device, hop_budget=7)
    manifest["supervision"] = "joint correlated-action outcome policy gradient; no teacher"
    controller.edge_magnitude.requires_grad_(True)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * mask)
    controller.zero_grad(set_to_none=True)
    microbatch = data.select(range(2 * args.pairs), device)
    advantage = microbatch.returns.clone()
    values = advantage[microbatch.valid]
    advantage = (advantage - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

    def observe(time):
        return microbatch.observation(time, camera, gate_config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    started = perf_counter()
    try:
        frames = args.replay_frames
        stats = replay_joint_policy_gradient(
            controller, observe, microbatch.latents[:frames], microbatch.old_means[:frames],
            microbatch.old_log_prob[:frames], advantage[:frames], microbatch.valid[:frames],
            stationary_std=microbatch.stationary_std, rho=microbatch.rho,
            chunk_steps=args.chunk_steps, warmup_steps=microbatch.warmup_steps,
        )
        synchronize()
        seconds = perf_counter() - started
        gradient = controller.edge_magnitude.grad
        result.update(
            status="complete", replay_wall_seconds=seconds,
            replay_command_samples_per_second=stats["valid_commands"] / seconds,
            replay_diagnostics=stats, native_path_manifest=manifest,
            edge_gradient_l2=float(gradient.norm()),
            nonzero_selected_edge_gradients=int((gradient[mask] != 0).sum()),
            cuda_peak_allocated_mib=(torch.cuda.max_memory_allocated(device) / 2 ** 20
                                     if device.type == "cuda" else None),
            cuda_peak_reserved_mib=(torch.cuda.max_memory_reserved(device) / 2 ** 20
                                    if device.type == "cuda" else None),
        )
    except Exception as error:
        result.update(status="failed", replay_error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        controller.zero_grad(set_to_none=True)
    save()
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
