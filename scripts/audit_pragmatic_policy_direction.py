#!/usr/bin/env python3
"""Compare a TBPTT/Adam prediction with fixed-data full-history PPO loss changes.

This is a training diagnostic, not controller selection or fresh goal validation.
All temporary parameter changes are restored; the fixed rollout is saved as plain
tensor/primitive data for a possible longer-gradient comparison.
"""

from __future__ import annotations

import argparse
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
from pragmatic_policy_optimization import replay_round  # noqa: E402
from pragmatic_policy_rollout import collect_policy_rollout  # noqa: E402

SCALES = (0., 0., 1., .5, .25, .125, -.125, 0.)


def scaled_parameters(base, delta, scale, mask):
    """Always start at the source; include FP32 rounding and native projection."""
    result = base.clone()
    proposed = base[mask] + scale * delta[mask]
    result[mask] = proposed.clamp(0, 8)
    return result, int((proposed != result[mask]).sum())


def summarize_direction(entries):
    zero = [item["diagnostics"]["loss"] for item in entries if item["scale"] == 0]
    if len(zero) < 2 or not all(math.isfinite(value) for value in zero):
        raise ValueError("need finite repeated zero-step losses")
    baseline = sum(zero) / len(zero)
    tolerance = max(5 * (max(zero) - min(zero)), 1e-6)
    positive = [item for item in entries if item["scale"] > 0]
    for item in entries:
        item["observed_loss_change"] = item["diagnostics"]["loss"] - baseline
    descending = [p["scale"] for p in positive if p["predicted_loss_change"] < 0
                  and p["observed_loss_change"] < -tolerance]
    increasing = [p["scale"] for p in positive if p["predicted_loss_change"] < 0
                  and p["observed_loss_change"] > tolerance]
    if any(p["predicted_loss_change"] >= 0 for p in positive):
        interpretation = "non-descent-materialized-optimizer-direction-present"
    elif descending:
        interpretation = "descent-resolved-at-some-tested-scales"
    elif increasing:
        interpretation = "predicted-descent-but-observed-increase-not-yet-attributed"
    else:
        interpretation = "inconclusive-at-replay-variability-level"
    return dict(zero_step_losses=zero, baseline_loss_mean=baseline,
                zero_step_loss_range=max(zero)-min(zero), interpretation_tolerance=tolerance,
                observed_descent_scales=descending, observed_increase_scales=increasing,
                interpretation=interpretation)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026091420)
    parser.add_argument("--noise-seed", type=int, default=2026091421)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    if (min(args.pairs, args.microbatch, args.chunk_steps, args.learning_rate) <= 0
            or not math.isfinite(args.learning_rate)):
        raise SystemExit("need positive sizes and finite positive learning rate")
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite a direction diagnostic")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    mask, manifest = replay.roll_preservation_mask(args.graph, device, hop_budget=7)
    manifest["supervision"] = "fixed-data PPO directional diagnostic; no teacher"
    base = controller.edge_magnitude.detach().clone()
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    result = dict(experiment="native-ppo-direction-diagnostic-v1", status="running",
                  checkpoint=str(args.checkpoint), graph_sha256=source["graph_sha256"],
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()},
                  geometry=replay.GEOMETRY, seconds=30, scales=SCALES, entries=[],
                  actor_weights_changed=False, optimizer="fresh Adam; no inherited momentum",
                  scope="fixed training rollout, no selection or goal validation",
                  native_path_manifest=manifest)

    def report():
        result["elapsed_seconds"] = perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    def progress(event):
        print(json.dumps(dict(elapsed_seconds=perf_counter()-started, **event)), flush=True)

    report()
    try:
        bank = replay.sample_two_gate_cases(args.pairs, seed=args.seed, device=device,
                                            hover_config=config, **replay.GEOMETRY)
        data = collect_policy_rollout(controller, *bank, camera=camera, config=config,
                                      gate_config=gate_config, noise_seed=args.noise_seed)
        returns = data.returns[data.valid]
        advantages = torch.where(data.valid, (data.returns - returns.mean()) /
                                  returns.std(unbiased=False).clamp_min(1e-6), 0).detach()
        result["collection"] = data.metrics
        report()
        progress(dict(stage="collection-complete", metrics=data.metrics))
        controller.edge_magnitude.requires_grad_(True)
        optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)

        def assess(backward):
            return replay_round(controller, data, advantages, camera=camera,
                                gate_config=gate_config,
                                microbatch=args.microbatch, chunk_steps=args.chunk_steps,
                                backward=backward, progress=progress)

        optimizer.zero_grad(set_to_none=True)
        gradient_stats = assess(True)
        controller.edge_magnitude.grad.mul_(mask)
        gradient = controller.edge_magnitude.grad.detach().clone()
        norm = torch.nn.utils.clip_grad_norm_([controller.edge_magnitude], 1.,
                                              error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            controller.edge_magnitude[mask] = controller.edge_magnitude[mask].clamp(0, 8)
            controller.edge_magnitude[~mask] = base[~mask]
            delta = controller.edge_magnitude.detach().clone() - base
            controller.edge_magnitude.copy_(base)
        optimizer.zero_grad(set_to_none=True)
        if not bool(delta.isfinite().all()):
            raise FloatingPointError("nonfinite materialized Adam displacement")
        result.update(gradient_replay=gradient_stats, gradient_norm=float(norm),
                      adam_displacement_l2=float(delta.norm()),
                      predicted_full_step_loss_change=float((gradient * delta).sum()))
        payload = dict(vars(data))
        payload["gates"] = [dict(center=g.center, yaw=g.yaw) for g in data.gates]
        torch.save(dict(schema="plain-native-policy-direction-v1", rollout=payload,
                        advantages=advantages, gradient=gradient.cpu(), displacement=delta.cpu(),
                        mask=mask.cpu(), source_checkpoint=str(args.checkpoint),
                        graph_sha256=source["graph_sha256"]),
                   args.output_dir / "fixed-training-data.pt")
        report()
        progress(dict(stage="gradient-complete", gradient_norm=float(norm),
                      predicted_change=result["predicted_full_step_loss_change"]))
        for scale in SCALES:
            proposed, projections = scaled_parameters(base, delta, scale, mask)
            with torch.no_grad():
                controller.edge_magnitude.copy_(proposed)
            actual = proposed - base
            entry = dict(scale=scale, predicted_loss_change=float((gradient * actual).sum()),
                         actual_displacement_l2=float(actual.norm()),
                         nonzero_edge_changes=int((actual != 0).sum()),
                         projected_magnitudes=projections, diagnostics=assess(False))
            result["entries"].append(entry)
            report()
            progress(dict(stage="scale-complete", **entry))
        result.update(summary=summarize_direction(result["entries"]), status="complete")
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        with torch.no_grad():
            controller.edge_magnitude.copy_(base)
        controller.zero_grad(set_to_none=True)
        result["source_parameters_restored"] = torch.equal(controller.edge_magnitude, base)
        report()
    progress(dict(stage="complete", summary=result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
