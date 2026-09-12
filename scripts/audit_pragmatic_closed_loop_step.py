#!/usr/bin/env python3
"""Check actual short-flight gradients before training native course tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pragmatic_closed_loop as physical  # noqa: E402
import train_pragmatic_course_replay as replay  # noqa: E402
from audit_pragmatic_anticipation_step import restored_step  # noqa: E402
from pragmatic_anticipation_lessons import bank_from_motor_cache  # noqa: E402


def load_context(args):
    device = torch.device("cuda")
    controller, source = replay.load_controller(args, device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if cache["source_sha256"] != digest:
        raise ValueError("physical cache and source checkpoint do not match")
    raw = next(bank for bank in cache["banks"] if not bank["assisted"])
    bank = bank_from_motor_cache(raw)
    config = replay.HoverConfig(**source["hover_config"])
    gate_config = replace(replay.GateConfig(**source["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*source["image_resolution"], source["camera_hfov_degrees"])
    cases, sampled_gates = replay.sample_two_gate_cases(
        bank.current.shape[1] // 2,
        seed=bank.seed,
        device=device,
        hover_config=config,
        **replay.GEOMETRY,
    )
    if not torch.allclose(cases.state.position, bank.states[0][0].to(device), atol=1e-5):
        raise ValueError("sampler does not reconstruct cached launch conditions")
    for sampled, saved in zip(sampled_gates, bank.gates, strict=True):
        if not (
            torch.allclose(sampled.center, saved.center.to(device), atol=1e-5)
            and torch.allclose(sampled.yaw, saved.yaw.to(device), atol=1e-5)
        ):
            raise ValueError("sampler and cached gate geometry disagree")
    mask, manifest = replay.roll_preservation_mask(args.graph, device)
    manifest["supervision"] = "closed-loop physical lateral tracking with fixed native preservation"
    controller.edge_magnitude.register_hook(lambda gradient: gradient * mask)
    return controller, source, bank, cases, config, gate_config, camera, mask, manifest, digest


@torch.no_grad()
def verify_saved_action_boundary(lesson, steps, config):
    fields, _, _, motors, starts = lesson.window
    rows = torch.arange(len(starts), device=starts.device)
    state = physical.QuadState(*(value[starts, rows].clone() for value in fields))
    sticks = physical.StickState(*(value.clone() for value in physical.stick_fields(lesson.sticks)))
    quad = physical.DifferentiableQuad(config).to(starts.device)
    legs = physical.ForelegStickPlant(config).to(starts.device)
    maximum = torch.zeros(len(fields), device=starts.device)
    for frame in range(steps):
        for _ in range(2):
            rc, sticks = legs(motors[starts + frame, rows], sticks)
            state = quad(rc, state, lesson.mass_scale)
        error = torch.stack(
            [
                (actual - saved[starts + frame + 1, rows]).abs().max()
                for actual, saved in zip(state.as_tuple(), fields, strict=True)
            ]
        )
        maximum = torch.maximum(maximum, error)
    if not bool(torch.isfinite(maximum).all()) or float(maximum.max()) > 0.002:
        raise ValueError(f"saved-action physical boundary mismatch: {maximum.tolist()}")
    return dict(
        maximum_absolute_component_errors=maximum.tolist(),
        fields=["position", "velocity", "euler", "rates", "actuator", "specific_force"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1420983)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 0.1, 1, 3, 10])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing physical audit")
    if args.steps < 1 or args.learning_rate <= 0 or min(args.scales) < 0:
        raise SystemExit("invalid steps, rate or scales")
    controller, _, bank, cases, config, gates, camera, mask, manifest, digest = load_context(args)
    rng = np.random.default_rng(args.seed)
    lessons, boundaries = [], []
    for phase in (1, 2, 3):
        rows, starts = physical.select_physical_window(bank, phase, args.steps, rng)
        lesson = physical.make_physical_lesson(
            bank, cases, rows, starts, args.steps, config, controller.bias.device
        )
        boundaries.append(verify_saved_action_boundary(lesson, args.steps, config))
        lessons.append(lesson)
    lesson = lessons[0]
    fixed = replay.replay_prefix_state(controller, lesson.window, camera, gates)
    initial = controller.edge_magnitude.detach().clone()
    loss, baseline = physical.physical_rollout_loss(
        controller, lesson, args.steps, camera, config, gates, fixed_neural=fixed
    )
    if not physical.admissible_tracking_trial(baseline):
        raise RuntimeError(f"source physical lesson is inadmissible: {baseline}")
    loss.backward()
    raw_gradient = controller.edge_magnitude.grad.detach().clone()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0, error_if_nonfinite=True)
    )
    optimizer = torch.optim.Adam([controller.edge_magnitude], lr=args.learning_rate)
    optimizer.step()
    delta = controller.edge_magnitude.detach() - initial
    with torch.no_grad():
        controller.edge_magnitude.copy_(initial)
    del loss
    result = dict(
        experiment="native-closed-loop-physical-step-audit-v1",
        source_sha256=digest,
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        native_path_manifest=manifest,
        lessons=[item.record for item in lessons],
        physical_boundary_checks=boundaries,
        baseline=baseline,
        raw_gradient_norm=gradient_norm,
        records=[],
        actor_privileged_inputs=False,
        deployed_extra_state=False,
        exported_controller=False,
    )
    print(json.dumps(dict(stage="baseline", **result)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for scale in args.scales:
        with restored_step(controller.edge_magnitude, initial, delta, scale) as projection:
            with torch.no_grad():
                _, fixed_metrics = physical.physical_rollout_loss(
                    controller, lesson, args.steps, camera, config, gates, fixed_neural=fixed
                )
                _, fresh_metrics = physical.physical_rollout_loss(
                    controller, lesson, args.steps, camera, config, gates
                )
                entry = dict(
                    scale=scale,
                    equivalent_first_adam_lr=args.learning_rate * scale,
                    **projection,
                    predicted_loss_change=float(
                        (raw_gradient * (controller.edge_magnitude - initial)).sum()
                    ),
                    fixed_prefix=fixed_metrics,
                    current_prefix=fresh_metrics,
                    admissible=physical.admissible_tracking_trial(fresh_metrics),
                    tracking_improved=fresh_metrics["tracking_loss"]
                    < 0.999 * baseline["tracking_loss"],
                )
                if scale == 0 and abs(fixed_metrics["loss"] - fresh_metrics["loss"]) > 1e-4:
                    raise RuntimeError("zero-step fixed/current prefix objectives disagree")
        result["records"].append(entry)
        print(json.dumps(entry), flush=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert torch.equal(controller.edge_magnitude.detach(), initial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
