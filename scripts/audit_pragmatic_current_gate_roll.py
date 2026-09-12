#!/usr/bin/env python3
"""Local roll-teacher diagnostic; never a native fly completion result."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_pragmatic_two_gate_zero_shot as evaluation  # noqa: E402
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.course_teacher import current_gate_roll_motor  # noqa: E402
from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def local_roll_takeover(native, local_roll, current, *, from_start):
    """Replace roll only; the from-start condition includes the first physical command."""
    motor = native.clone()
    motor[:, 0] = (
        local_roll if from_start else torch.where(current >= 1, local_roll, native[:, 0])
    )
    return motor


def evaluate_condition(
    controller, cases, gates, config, camera, gate_config, *, intervention,
    seconds=30, warmup_steps=10,
):
    """Matched initial conditions, fresh neural state, native PYT throughout each rollout."""
    if intervention not in ("native", "after-first", "from-start"):
        raise ValueError("unknown intervention")
    kwargs = dict(
        seconds=seconds, warmup_steps=warmup_steps, camera=camera,
        hover_config=config, gate_config=gate_config,
    )
    if intervention == "native":
        return evaluation.evaluate(controller, cases, gates, **kwargs)
    captured = {}

    def capture_state(state, path, quad, teacher_config):
        # The evaluator constructs its ordinary path, but this target never uses it.
        captured["state"] = state
        return torch.zeros_like(state.actuator)

    def takeover(native, unused_target, current, axes):
        local = current_gate_roll_motor(
            captured["state"], evaluation.active_gate(gates, current), config,
            active=current < len(gates),
        )
        return local_roll_takeover(native, local, current, from_start=intervention == "from-start")

    with (
        patch.object(evaluation, "course_teacher_motor", capture_state),
        patch.object(evaluation, "diagnostic_axis_takeover", takeover),
    ):
        metrics = evaluation.evaluate(
            controller, cases, gates, **kwargs,
            diagnostic_teacher_axes=(True, False, False, False),
        )
    # The generic evaluator's original metadata describes its default after-first
    # hook. Do not mislabel the overridden from-start condition.
    if intervention == "from-start":
        metrics.pop("privileged_axis_takeover_after_first_gate", None)
        metrics["privileged_roll_from_first_physical_command"] = True
    return metrics


def comparison_screen(conditions):
    """Conservative teacher-rescue screen on 32 development cases, not a goal test."""
    after = conditions["after-first"]
    start = conditions["from-start"]
    after_ids = set(after["clean_course_success_episode_indices"])
    start_ids = set(start["clean_course_success_episode_indices"])
    lost, gained = sorted(after_ids - start_ids), sorted(start_ids - after_ids)
    scalar_keys = (
        "clean_course_success_rate", "clean_course_negative_success_rate",
        "clean_course_positive_success_rate", "clean_first_gate_pass_rate",
        "ground_contact_rate", "invalid_rate",
    )
    if any(not math.isfinite(start[key]) for key in scalar_keys):
        raise ValueError("nonfinite diagnostic metrics")
    if len(start_ids) != round(32 * start["clean_course_success_rate"]):
        raise ValueError("clean indices disagree with 32-case success rate")
    criteria = dict(
        at_least_26_clean=len(start_ids) >= 26,
        at_least_12_negative=start["clean_course_negative_success_rate"] >= 12 / 16,
        at_least_12_positive=start["clean_course_positive_success_rate"] >= 12 / 16,
        at_least_28_clean_first=start["clean_first_gate_pass_rate"] >= 28 / 32,
        zero_ground=start["ground_contact_rate"] == 0,
        zero_invalid=start["invalid_rate"] == 0,
        at_most_one_paired_clean_loss=len(lost) <= 1,
    )
    return dict(
        eligible_for_unified_label_learning=all(criteria.values()),
        criteria=criteria,
        clean_after_first=len(after_ids), clean_from_start=len(start_ids),
        net_clean_change=len(start_ids) - len(after_ids),
        paired_clean_losses=lost, paired_clean_gains=gained,
        assisted_results_are_not_fly_success=True,
    )


def conditional_success(metrics):
    conditional = {}
    for side in ("all", "negative", "positive"):
        suffix = "" if side == "all" else f"_{side}"
        first = metrics[f"clean_first_gate{suffix}_pass_rate"]
        clean = metrics[f"clean_course{suffix}_success_rate"]
        conditional[side] = clean / first if first else None
    return conditional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--graph", type=Path, default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz"
    )
    parser.add_argument("--seed", type=int, default=1110983)
    parser.add_argument(
        "--compare-starts", action="store_true",
        help="compare native, after-first and from-start roll on the same 32 initial cases",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing audit")
    device = torch.device("cuda")
    controller, payload = load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = HoverConfig(**payload["hover_config"])
    gate_config = replace(GateConfig(**payload["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*payload["image_resolution"], payload["camera_hfov_degrees"])
    cases, gates = evaluation.sample_two_gate_cases(
        16, seed=args.seed, device=device, hover_config=config, **GEOMETRY
    )
    started = perf_counter()
    result = dict(
        experiment="diagnostic-current-gate-roll-start-comparison-v1" if args.compare_starts
        else "diagnostic-current-gate-velocity-damped-roll-v1",
        checkpoint=str(args.checkpoint),
        seed=args.seed,
        geometry=GEOMETRY,
        assisted_results_are_not_fly_success=True,
        teacher_inputs=["current gate relative position", "velocity", "roll", "roll rate"],
        absolute_heading_used_only_to_rotate_to_body_horizontal_frame=True,
        no_launch_line_or_path_used_by_target=True,
        native_axes=["pitch", "yaw", "throttle"],
        actor_inputs=["live RGB", "roll", "pitch"],
        continuous_neural_state=True,
        seconds=30,
        warmup_steps=10,
        episodes=32,
        full_post_completion_tail_scored=True,
        same_initial_cases_across_conditions=True,
        mass_scale_range=[float(cases.mass_scale.min()), float(cases.mass_scale.max())],
        status="running",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conditions = {}
    names = ("native", "after-first", "from-start") if args.compare_starts else ("after-first",)
    for name in names:
        metrics = evaluate_condition(
            controller, cases, gates, config, camera, gate_config, intervention=name,
        )
        conditional = conditional_success(metrics)
        conditions[name] = metrics
        if args.compare_starts:
            result["conditions"] = conditions
        else:
            result.update(
                intervention="roll only after gate one, including full post-completion tail",
                clean_completion_given_clean_first=conditional, metrics=metrics,
            )
        result["elapsed_seconds"] = perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(
            intervention=name, clean=metrics["clean_course_success_rate"],
            conditional=conditional, ground=metrics["ground_contact_rate"],
            invalid=metrics["invalid_rate"], elapsed_seconds=result["elapsed_seconds"],
        )), flush=True)
    if args.compare_starts:
        result["screen"] = comparison_screen(conditions)
        print(json.dumps(result["screen"]), flush=True)
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
