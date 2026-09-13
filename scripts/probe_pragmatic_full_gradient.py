#!/usr/bin/env python3
"""Untruncated-autograd comparison on an archived fixed PPO training rollout.

Two small full-gradient Adam probes follow the gradient/cost comparison. They are
not flight evaluations or controller candidates, and source weights are restored.
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
from audit_pragmatic_policy_direction import scaled_parameters, summarize_direction  # noqa: E402
from pragmatic_policy_optimization import replay_round  # noqa: E402
from pragmatic_policy_rollout import PolicyRollout  # noqa: E402

SCALES = (0., .125, .03125, 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() or min(args.microbatch, args.chunk_steps) < 1:
        raise SystemExit("need a new output directory and positive batch/chunk sizes")
    archive = torch.load(args.archive, map_location="cpu", weights_only=True)
    if (archive.get("schema") != "plain-native-policy-direction-v1"
            or Path(archive["source_checkpoint"]).resolve() != args.checkpoint.resolve()):
        raise SystemExit("archive schema/source does not match this probe")
    device = torch.device(args.device)
    controller, source = replay.load_controller(args, device)
    if archive["graph_sha256"] != source["graph_sha256"]:
        raise SystemExit("archive and controller graphs differ")
    controller.eval().requires_grad_(False)
    controller.edge_magnitude.requires_grad_(True)
    base = controller.edge_magnitude.detach().clone()
    mask = archive["mask"].to(device)
    old_gradient, old_delta = archive["gradient"].to(device), archive["displacement"].to(device)
    fields = dict(archive["rollout"])
    fields["gates"] = tuple(replay.AnnularGate(**g) for g in fields["gates"])
    data, advantages = PolicyRollout(**fields), archive["advantages"]
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    result = dict(experiment="native-full-history-gradient-probe-v1", status="running",
                  archive=str(args.archive), checkpoint=str(args.checkpoint),
                  episodes=data.valid.shape[1], frames=data.valid.shape[0],
                  activation_chunk_steps=args.chunk_steps, microbatch=args.microbatch,
                  scales=SCALES, entries=[], new_flights_collected=0, controller_exported=False,
                  scope="untruncated autograd on fixed data; not gradient through physics")

    def report():
        result["elapsed_seconds"] = perf_counter()-started
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    def progress(event):
        print(json.dumps({**event, "elapsed_seconds": perf_counter()-started}), flush=True)

    def assess(backward):
        return replay_round(controller, data, advantages, camera=camera, gate_config=gate_config,
                            microbatch=args.microbatch, chunk_steps=args.chunk_steps,
                            backward=backward, progress=progress, full_history=True)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    report()
    try:
        optimizer = torch.optim.Adam([controller.edge_magnitude], lr=1e-6)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        timed = perf_counter()
        result["gradient_replay"] = assess(True)
        sync()
        result["full_gradient_wall_seconds"] = perf_counter()-timed
        controller.edge_magnitude.grad.mul_(mask)
        gradient = controller.edge_magnitude.grad.detach().clone()
        result.update(
            full_gradient_norm=float(gradient.norm()),
            truncated_gradient_norm=float(old_gradient.norm()),
            raw_gradient_cosine=float(torch.nn.functional.cosine_similarity(
                gradient, old_gradient, dim=0)),
            full_gradient_dot_original_adam=float((gradient*old_delta).sum()),
            truncated_gradient_dot_original_adam=float((old_gradient*old_delta).sum()),
            cuda_peak_allocated_mib=(torch.cuda.max_memory_allocated(device)/2**20
                                     if device.type == "cuda" else None),
            cuda_peak_reserved_mib=(torch.cuda.max_memory_reserved(device)/2**20
                                    if device.type == "cuda" else None),
        )
        report()
        progress(dict(stage="full-gradient-complete", **result))
        torch.nn.utils.clip_grad_norm_([controller.edge_magnitude], 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            controller.edge_magnitude[mask] = controller.edge_magnitude[mask].clamp(0, 8)
            controller.edge_magnitude[~mask] = base[~mask]
            delta = controller.edge_magnitude.detach().clone()-base
            controller.edge_magnitude.copy_(base)
        optimizer.zero_grad(set_to_none=True)
        if not bool(delta.isfinite().all()):
            raise FloatingPointError("nonfinite full-gradient Adam displacement")
        torch.save(dict(gradient=gradient.cpu(), displacement=delta.cpu(), mask=mask.cpu()),
                   args.output_dir / "full-gradient-data.pt")
        for scale in SCALES:
            proposed, projections = scaled_parameters(base, delta, scale, mask)
            with torch.no_grad():
                controller.edge_magnitude.copy_(proposed)
            actual = proposed-base
            entry = dict(scale=scale, predicted_loss_change=float((gradient*actual).sum()),
                         actual_displacement_l2=float(actual.norm()),
                         nonzero_edge_changes=int((actual != 0).sum()),
                         projected_magnitudes=projections, diagnostics=assess(False))
            result["entries"].append(entry)
            report()
            progress(dict(stage="scale-complete", **entry))
        result.update(status="complete", summary=summarize_direction(result["entries"]))
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
