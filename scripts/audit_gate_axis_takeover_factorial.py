#!/usr/bin/env python3
"""Localize gate-flight failures with frozen throttle/steering teacher takeovers."""

from __future__ import annotations

import argparse
import copy
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

from audit_gate_analytic_teachers import compact, teacher_rc_for_mode  # noqa: E402
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
)
from search_gate_motor_interface_es import (  # noqa: E402
    clone_state,
    paired_confidence_interval,
    stable_path,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    _episode_summary,
    controller_parameter_sha256,
    controller_step,
    load_frozen_controller,
    restore_parameters,
)
from train_gate_recurrent_ppo import initialize_outcomes, update_outcomes  # noqa: E402

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)

INTERVENTIONS = ("native", "reserve_throttle", "reserve_steering", "full_reserve")
STRATUM_RATE_KEYS = (
    "success_rate",
    "light_success_rate",
    "heavy_success_rate",
    "negative_lateral_success_rate",
    "positive_lateral_success_rate",
    "negative_obliquity_success_rate",
    "positive_obliquity_success_rate",
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
        "--failed-archive",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-dense-dagger-diagnostic-v2" / "archive.pt"),
    )
    parser.add_argument(
        "--failed-report",
        type=Path,
        default=(REPO_ROOT / "artifacts" / "gate-dense-dagger-diagnostic-v2" / "report.json"),
    )
    parser.add_argument("--failed-update", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "axis-takeover-factorial-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--minimum-control-success", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1_026_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.failed_archive, args.failed_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.episodes != 512:
        raise SystemExit("the preregistered factorial audit requires 512 episodes")
    if args.seconds != 12.0 or args.takeover_seconds != 0.50:
        raise SystemExit("the preregistered audit uses 12 seconds and takeover at 0.5 seconds")
    if not 0.0 < args.minimum_control_success <= 1.0:
        raise SystemExit("minimum control success must be in (0, 1]")


def compose_axis_takeover_motor(
    native_motor: Tensor,
    reserve_motor: Tensor,
    intervention: str,
) -> Tensor:
    if intervention == "native":
        return native_motor
    if intervention == "reserve_throttle":
        motor = native_motor.clone()
        motor[:, 3] = reserve_motor[:, 3]
        return motor
    if intervention == "reserve_steering":
        motor = native_motor.clone()
        motor[:, :3] = reserve_motor[:, :3]
        return motor
    if intervention == "full_reserve":
        return reserve_motor
    raise ValueError(f"unknown axis-takeover intervention: {intervention}")


@torch.no_grad()
def evaluate_axis_takeover(
    controller: ConnectomeController,
    cases: Any,
    *,
    intervention: str,
    takeover_seconds: float,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    device = cases.mass_scale.device
    episodes = len(cases.mass_scale)
    steps = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(cases.state)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(episodes, device=device, dtype=torch.float32)
    outcomes = initialize_outcomes(cases, gate_config)
    for step in range(steps):
        image = render_annular_gate(
            state,
            cases.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        native_motor, neural = controller_step(
            controller,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
        )
        reserve_rc = teacher_rc_for_mode(
            "visual_accelerometer_reserve",
            controller,
            state,
            cases.gate,
            cases.mass_scale,
            hover_config,
        )
        reserve_motor = motor_target_for_rc(reserve_rc, hover_config)
        motor = (
            native_motor
            if step < takeover_step
            else compose_axis_takeover_motor(native_motor, reserve_motor, intervention)
        )
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
    summary, tensors = _episode_summary(
        outcomes,
        cases,
        steps=steps,
        hover_config=hover_config,
    )
    summary["intervention"] = intervention
    summary["takeover_seconds"] = takeover_seconds
    return summary, tensors


def paired_effects(native: dict[str, Tensor], candidate: dict[str, Tensor]) -> dict[str, Any]:
    baseline = native["success"]
    intervention = candidate["success"]
    codes = native["codes"]
    light = native["mass_scale"] < 1.0
    groups = {
        "overall": torch.ones_like(light),
        "light": light,
        "heavy": ~light,
        "negative_lateral": ~codes.bitwise_and(2).bool(),
        "positive_lateral": codes.bitwise_and(2).bool(),
        "negative_obliquity": ~codes.bitwise_and(4).bool(),
        "positive_obliquity": codes.bitwise_and(4).bool(),
    }
    result = {}
    for name, selected in groups.items():
        if name == "overall":
            result[name] = paired_clustered_confidence_interval(
                baseline[selected], intervention[selected]
            )
        else:
            result[name] = paired_confidence_interval(baseline[selected], intervention[selected])
    return result


def rescue_classification(
    results: dict[str, dict[str, dict[str, Any]]],
    *,
    minimum_success: float,
) -> dict[str, Any]:
    def passes(controller: str, intervention: str) -> bool:
        return min(results[controller][intervention][key] for key in STRATUM_RATE_KEYS) >= (
            minimum_success
        )

    full_control = passes("source", "full_reserve")
    throttle = passes("source", "reserve_throttle")
    steering = passes("source", "reserve_steering")
    failed_full = passes("failed_update", "full_reserve")
    if not full_control:
        interpretation = "invalid_full_reserve_control"
    elif throttle and not steering:
        interpretation = "throttle_takeover_is_sufficient"
    elif steering and not throttle:
        interpretation = "steering_takeover_is_sufficient"
    elif not throttle and not steering:
        interpretation = "coupled_throttle_and_steering_takeover_required"
    else:
        interpretation = "either_axis_takeover_is_sufficient"
    return {
        "minimum_success_per_declared_stratum": minimum_success,
        "source_full_reserve_control_passed": full_control,
        "source_throttle_only_rescue_passed": throttle,
        "source_steering_only_rescue_passed": steering,
        "failed_update_full_reserve_control_passed": failed_full,
        "failed_prefix_damage_detected": full_control and not failed_full,
        "interpretation": interpretation,
        "passed": full_control,
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
    failed_report = json.loads(args.failed_report.read_text())
    if failed_report.get("checkpoint_sha256") != file_sha256(args.checkpoint):
        raise SystemExit("failed-run report does not derive from the requested source checkpoint")
    if failed_report.get("archive_sha256") != file_sha256(args.failed_archive):
        raise SystemExit("failed archive does not match its report")
    archive = torch.load(args.failed_archive, map_location=device, weights_only=True)
    if args.failed_update not in archive:
        raise SystemExit(f"failed archive has no update {args.failed_update}")
    failed = copy.deepcopy(source).to(device)
    restore_parameters(failed, archive[args.failed_update]["parameters"])
    if controller_parameter_sha256(failed) == controller_parameter_sha256(source):
        raise SystemExit("requested failed update is identical to the source controller")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite existing report: {report_path}")
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    cases = diverse_matched_cases(
        args.episodes,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    controllers = {"source": source, "failed_update": failed}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    outcome_tensors: dict[str, dict[str, dict[str, Tensor]]] = {}
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for controller_name, controller in controllers.items():
        results[controller_name] = {}
        outcome_tensors[controller_name] = {}
        for intervention in INTERVENTIONS:
            summary, tensors = evaluate_axis_takeover(
                controller,
                cases,
                intervention=intervention,
                takeover_seconds=args.takeover_seconds,
                seconds=args.seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            results[controller_name][intervention] = summary
            outcome_tensors[controller_name][intervention] = tensors
            print(
                json.dumps(
                    {
                        "controller": controller_name,
                        "intervention": intervention,
                        **compact(summary),
                    }
                ),
                flush=True,
            )
        paired[controller_name] = {
            intervention: paired_effects(
                outcome_tensors[controller_name]["native"],
                outcome_tensors[controller_name][intervention],
            )
            for intervention in INTERVENTIONS[1:]
        }
    classification = rescue_classification(
        results,
        minimum_success=args.minimum_control_success,
    )
    report = {
        "experiment": "frozen-policy axis-takeover factorial diagnostic",
        "claim_scope": (
            "Training-only causal intervention. The analytical reserve commands and failed "
            "snapshot are never eligible for promotion."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "source_parameter_sha256": controller_parameter_sha256(source),
        "failed_archive": stable_path(args.failed_archive),
        "failed_archive_sha256": file_sha256(args.failed_archive),
        "failed_report": stable_path(args.failed_report),
        "failed_report_sha256": file_sha256(args.failed_report),
        "failed_update": args.failed_update,
        "failed_parameter_sha256": controller_parameter_sha256(failed),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "teacher_mode": "visual_accelerometer_reserve",
            "uses_exact_simulator_mass": False,
            "common_cases_across_all_cells": True,
            "controller_owns_first_half_second": True,
            "native_state_continues_without_reset": True,
            "intervention_applied_before_physical_stick_plant": True,
            "diagnostic_only_no_promotion": True,
        },
        "interventions": list(INTERVENTIONS),
        "results": results,
        "paired_success_effects_vs_native": paired,
        "classification": classification,
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "report": stable_path(report_path),
                "control_passed": classification["passed"],
                "interpretation": classification["interpretation"],
                "failed_prefix_damage_detected": classification["failed_prefix_damage_detected"],
            }
        ),
        flush=True,
    )
    return 0 if classification["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
