#!/usr/bin/env python3
"""Select a strong training-only analytical teacher on diverse annular gates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_motor_interface_es import clone_state, stable_path  # noqa: E402
from train_gate import actor_teacher_gate_rc, file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    _episode_summary,
    controller_step,
    load_frozen_controller,
)
from train_gate_recurrent_ppo import initialize_outcomes, update_outcomes  # noqa: E402

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    render_annular_gate,
    teacher_gate_rc_state_feedback,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
)

TEACHER_MODES = (
    "visual_accelerometer_reserve",
    "visual_accelerometer_exact_mass",
    "state_feedback_nominal_mass",
    "state_feedback_exact_mass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "analytic-teacher-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--selection-episodes", type=int, default=256)
    parser.add_argument("--preflight-episodes", type=int, default=1024)
    parser.add_argument("--selection-seed", type=int, default=998_031)
    parser.add_argument("--preflight-seed", type=int, default=999_031)
    parser.add_argument("--minimum-preflight-success", type=float, default=0.90)
    parser.add_argument("--teacher-modes", nargs="+", choices=TEACHER_MODES, default=TEACHER_MODES)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    for name in ("selection_episodes", "preflight_episodes"):
        value = getattr(args, name)
        if value <= 0 or value % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive and divisible by eight")
    if not 0.0 <= args.takeover_seconds < args.seconds:
        raise SystemExit("takeover must be nonnegative and precede the evaluation end")
    if not 0.0 < args.minimum_preflight_success <= 1.0:
        raise SystemExit("preflight threshold must be in (0, 1]")


def mass_hover_correction(
    state: QuadState,
    mass_scale: Tensor,
    *,
    nominal_scale: float,
    hover_config: HoverConfig,
) -> Tensor:
    tilt_compensation = 1.0 / (
        torch.cos(state.euler[:, 0]) * torch.cos(state.euler[:, 1])
    ).clamp_min(0.75)
    return (mass_scale - nominal_scale) / hover_config.thrust_to_weight * tilt_compensation


def teacher_rc_for_mode(
    mode: str,
    source: ConnectomeController,
    state: QuadState,
    gate: AnnularGate,
    mass_scale: Tensor,
    hover_config: HoverConfig,
) -> Tensor:
    if mode.startswith("visual_accelerometer"):
        rc = actor_teacher_gate_rc(source, state, gate, hover_config)
        nominal_mass = 1.08
    elif mode.startswith("state_feedback"):
        rc = teacher_gate_rc_state_feedback(state, gate, hover_config)
        nominal_mass = 1.0
    else:
        raise ValueError(f"unknown teacher mode: {mode}")
    if mode.endswith("exact_mass"):
        rc = rc.clone()
        rc[:, 3] = (
            rc[:, 3]
            + mass_hover_correction(
                state,
                mass_scale,
                nominal_scale=nominal_mass,
                hover_config=hover_config,
            )
        ).clamp(0.0, 1.0)
    return rc


@torch.no_grad()
def evaluate_teacher_takeover(
    source: ConnectomeController,
    cases: Any,
    *,
    mode: str,
    takeover_seconds: float,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = source.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        source_motor, neural = controller_step(
            source,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        target_rc = teacher_rc_for_mode(
            mode,
            source,
            state,
            cases.gate,
            cases.mass_scale,
            hover_config,
        )
        target_motor = motor_target_for_rc(target_rc, hover_config)
        motor = target_motor if step >= takeover_step else source_motor
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, cases.mass_scale)
        update_outcomes(
            outcomes,
            previous_position,
            state,
            cases.gate,
            stick_state.position,
            step,
            gate_config,
        )
    summary, _ = _episode_summary(outcomes, cases, steps=steps, hover_config=hover_config)
    summary["takeover_seconds"] = takeover_seconds
    summary["teacher_mode"] = mode
    return summary


def score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        min(metrics["light_success_rate"], metrics["heavy_success_rate"]),
        metrics["success_rate"],
        -metrics["crossing_radial_mean_m"],
    )


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "success_rate",
            "light_success_rate",
            "heavy_success_rate",
            "negative_lateral_success_rate",
            "positive_lateral_success_rate",
            "negative_obliquity_success_rate",
            "positive_obliquity_success_rate",
            "ring_collision_rate",
            "miss_rate",
            "crossing_radial_mean_m",
        )
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source, _, hover_config, gate_config, resolution = load_frozen_controller(
        args.graph, args.checkpoint, device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.selection_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()

    selection_cases = diverse_matched_cases(
        args.selection_episodes,
        seed=args.selection_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    selection = {}
    for mode in args.teacher_modes:
        metrics = evaluate_teacher_takeover(
            source,
            selection_cases,
            mode=mode,
            takeover_seconds=args.takeover_seconds,
            seconds=args.seconds,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        selection[mode] = metrics
        print(json.dumps({"phase": "selection", "mode": mode, **compact(metrics)}), flush=True)
    selected_mode = max(args.teacher_modes, key=lambda mode: score(selection[mode]))

    preflight_cases = diverse_matched_cases(
        args.preflight_episodes,
        seed=args.preflight_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    preflight = evaluate_teacher_takeover(
        source,
        preflight_cases,
        mode=selected_mode,
        takeover_seconds=args.takeover_seconds,
        seconds=args.seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    threshold_checks = {
        key: preflight[key] >= args.minimum_preflight_success
        for key in ("success_rate", "light_success_rate", "heavy_success_rate")
    }
    passed = all(threshold_checks.values())
    print(
        json.dumps(
            {
                "phase": "preflight",
                "mode": selected_mode,
                "passed": passed,
                **compact(preflight),
            }
        ),
        flush=True,
    )
    candidate = {
        "kind": "privileged_analytic_gate_teacher",
        "teacher_mode": selected_mode,
        "uses_exact_simulator_mass": selected_mode.endswith("exact_mass"),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "takeover_seconds": args.takeover_seconds,
        "preflight_episodes": args.preflight_episodes,
        "preflight_seed": args.preflight_seed,
        "minimum_success_per_mass_half": args.minimum_preflight_success,
        "preflight_passed": passed,
        "counts_toward_direct_sensor_goal": False,
    }
    candidate_path = args.output_dir / "candidate.json"
    candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    report = {
        "experiment": "diverse-gate analytical-teacher selection",
        "claim_scope": (
            "Training-only teacher validation. Privileged relative geometry, state, or exact "
            "mass may be used by the teacher and never enters the deployed actor."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "common_balanced_selection_cases": True,
            "selection_preflight_seeds_disjoint": True,
            "preflight_threshold_not_tuned_after_observation": True,
            "source_controller_prefix_before_takeover": True,
        },
        "selection": selection,
        "selected_mode": selected_mode,
        "preflight": preflight,
        "preflight_threshold_checks": threshold_checks,
        "preflight_passed": passed,
        "candidate": candidate,
        "candidate_file": stable_path(candidate_path),
        "candidate_file_sha256": file_sha256(candidate_path),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "selected_mode": selected_mode,
                "preflight_passed": passed,
                "threshold_checks": threshold_checks,
            }
        ),
        flush=True,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
