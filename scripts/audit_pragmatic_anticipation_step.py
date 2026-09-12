#!/usr/bin/env python3
"""Restored finite-step audit of one training-only native anticipation lesson.

No controller is exported or deployed. Compare the deliberately frozen source
prefix with complete current-weight replay to test truncated-gradient usefulness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pragmatic_anticipation_lessons as lessons  # noqa: E402
import train_pragmatic_course_replay as replay  # noqa: E402


@contextmanager
def restored_step(parameter, source, displacement, scale):
    """Project each scaled proposal independently, restoring even on failure."""
    proposal = source + scale * displacement
    with torch.no_grad():
        parameter.copy_(proposal.clamp(0, 8))
    try:
        yield dict(
            clipped_edges=int(((proposal < 0) | (proposal > 8)).sum()),
            actual_delta_l2=float((parameter.detach() - source).norm()),
            actual_delta_max=float((parameter.detach() - source).abs().max()),
        )
    finally:
        with torch.no_grad():
            parameter.copy_(source)


def residual_metrics(loss, residual, desired):
    contrast = residual[:, 1, 0] - residual[:, 0, 0]
    predicted = desired + contrast
    return dict(
        loss=float(loss),
        contrast_error_rmse=float(contrast.square().mean().sqrt()),
        zero_contrast_error_rmse=float(desired.square().mean().sqrt()),
        teacher_alignment_cosine=float(
            (predicted * desired).sum() / (predicted.norm() * desired.norm()).clamp_min(1e-12)
        ),
        source_pair_mean_drift_rmse=float(residual[:, :, 0].mean(1).square().mean().sqrt()),
        nonroll_source_rmse=float(residual[:, :, 1:].square().mean().sqrt()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank-seed", type=int, default=1190983)
    parser.add_argument("--row", type=int, default=10)
    parser.add_argument("--start", type=int, default=275)
    parser.add_argument("--unroll", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 1, 10, 100, 1000])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing step audit")
    if args.unroll < 1 or args.learning_rate <= 0 or min(args.scales) < 0:
        raise SystemExit("invalid unroll, learning rate or step scale")
    device = torch.device("cuda")
    controller, source = replay.load_controller(args, device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if cache["source_sha256"] != digest:
        raise SystemExit("source checkpoint does not match the physical-history cache")
    bank = lessons.bank_from_motor_cache(
        next(bank for bank in cache["banks"] if bank["seed"] == args.bank_seed)
    )
    if not 0 <= args.row < bank.current.shape[1] - 4:
        raise SystemExit("the step audit must use a training row, not the held-out last two pairs")
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(
        width=source["image_resolution"][0],
        height=source["image_resolution"][1],
        horizontal_fov_degrees=source["camera_hfov_degrees"],
    )
    window, record = lessons.make_anticipation_lesson(
        bank,
        args.row,
        args.start,
        args.unroll,
        controller,
        camera,
        gate_config,
        config,
        mean_target="source",
    )
    mask, manifest = replay.roll_preservation_mask(args.graph, device)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * mask)
    initial = controller.edge_magnitude.detach().clone()
    frozen_prefix = replay.replay_prefix_state(controller, window, camera, gate_config).clone()
    desired = lessons.window_target_contrast(window, args.unroll)

    def measure(fixed):
        loss, _, residual = replay.replay_window_loss(
            controller,
            window,
            args.unroll,
            camera,
            gate_config,
            1.0,
            diagnostics=True,
            roll_contrast_weight=16.0,
            diagnostic_fixed_prefix=frozen_prefix if fixed else None,
        )
        return loss, residual

    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    loss, residual = measure(True)
    baseline = residual_metrics(loss.detach(), residual, desired)
    (0.5 * loss).backward()  # same first update as the paired-preservation trainer
    raw_gradient = controller.edge_magnitude.grad.detach().clone()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0))
    optimizer.step()
    displacement = controller.edge_magnitude.detach() - initial
    with torch.no_grad():
        controller.edge_magnitude.copy_(initial)
    del loss
    result = dict(
        experiment="restored-native-anticipation-step-audit-v1",
        source_sha256=digest,
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        lesson=record,
        native_path_manifest=manifest,
        baseline=baseline,
        raw_gradient_norm=gradient_norm,
        records=[],
        saved_or_deployed_controller=False,
        heldout_or_autonomous_flight_evidence=False,
    )
    for scale in args.scales:
        with restored_step(controller.edge_magnitude, initial, displacement, scale) as projection:
            with torch.no_grad():
                fixed_loss, fixed_residual = measure(True)
                fresh_loss, fresh_residual = measure(False)
                if scale == 0 and not torch.allclose(
                    fixed_residual, fresh_residual, atol=1e-6, rtol=1e-4
                ):
                    raise RuntimeError("source fixed-prefix and full-prefix residuals disagree")
                entry = dict(
                    scale=scale,
                    equivalent_first_adam_lr=args.learning_rate * scale,
                    **projection,
                    raw_gradient_dot_actual_delta=float(
                        (raw_gradient * (controller.edge_magnitude - initial)).sum()
                    ),
                    predicted_unweighted_loss_change=float(
                        2 * (raw_gradient * (controller.edge_magnitude - initial)).sum()
                    ),
                    fixed_prefix=residual_metrics(fixed_loss, fixed_residual, desired),
                    complete_prefix=residual_metrics(fresh_loss, fresh_residual, desired),
                )
                result["records"].append(entry)
                print(json.dumps(entry), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not torch.equal(controller.edge_magnitude.detach(), initial):
        raise RuntimeError("audit failed to restore the source parameters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
