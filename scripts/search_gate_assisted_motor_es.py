#!/usr/bin/env python3
"""Search native steering and throttle readouts with training-only axis assistance."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_gate_analytic_teachers import teacher_rc_for_mode  # noqa: E402
from audit_gate_axis_takeover_factorial import (  # noqa: E402
    compose_axis_takeover_motor,
)
from gate_diverse_cases import diverse_matched_cases  # noqa: E402
from search_gate_acceleration_path_es import (  # noqa: E402
    paired_clustered_confidence_interval,
)
from search_gate_motor_interface_es import (  # noqa: E402
    MotorInterfaceSpec,
    clone_state,
    compile_vector,
    controller_step_with_interface,
    load_controller,
    motor_interface_spec,
    paired_confidence_interval,
    repeat_cases,
    shaping_weight,
    stable_path,
    summarize_policy_batch,
    vector_sha256,
)
from train_gate import file_sha256, seed_everything  # noqa: E402
from train_gate_full_network_oracle import (  # noqa: E402
    compact_flight,
    controller_parameter_sha256,
    evaluate_controller,
)

from flydrone.gate import (  # noqa: E402
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    gate_coordinates,
    render_annular_gate,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
)

MASS_LATERAL_KEYS = (
    "lower_mass",
    "higher_mass",
    "negative_lateral_offset",
    "positive_lateral_offset",
)
GOAL_RATE_KEYS = (
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
        "--factorial-report",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-axis-takeover-factorial-v1" / "report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "assisted-motor-es-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--assisted-generations", type=int, default=30)
    parser.add_argument("--joint-generations", type=int, default=30)
    parser.add_argument("--checkpoint-generation", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--antithetic-directions", type=int, default=16)
    parser.add_argument("--development-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--takeover-seconds", type=float, default=0.50)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sigma-decay", type=float, default=0.985)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--bias-perturbation-scale", type=float, default=0.01)
    parser.add_argument("--log-gain-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--maximum-bias-delta", type=float, default=0.15)
    parser.add_argument("--maximum-gain-ratio", type=float, default=3.0)
    parser.add_argument("--assisted-minimum-floor-gain", type=float, default=0.10)
    parser.add_argument("--joint-minimum-overall-gain", type=float, default=0.05)
    parser.add_argument("--joint-minimum-floor-gain", type=float, default=0.05)
    parser.add_argument("--maximum-stratum-drop", type=float, default=0.05)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--final-heavy-margin", type=float, default=0.02)
    parser.add_argument("--steering-development-seed", type=int, default=1_027_031)
    parser.add_argument("--steering-validation-seed", type=int, default=1_028_031)
    parser.add_argument("--throttle-development-seed", type=int, default=1_029_031)
    parser.add_argument("--throttle-validation-seed", type=int, default=1_030_031)
    parser.add_argument("--merge-validation-seed", type=int, default=1_031_031)
    parser.add_argument("--joint-development-seed", type=int, default=1_032_031)
    parser.add_argument("--joint-validation-seed", type=int, default=1_033_031)
    parser.add_argument("--final-seed", type=int, default=1_034_031)
    parser.add_argument("--search-seed", type=int, default=1_035_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint, args.factorial_report):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    fixed = {
        "assisted_generations": (args.assisted_generations, 30),
        "joint_generations": (args.joint_generations, 30),
        "checkpoint_generation": (args.checkpoint_generation, 20),
        "validation_interval": (args.validation_interval, 10),
        "antithetic_directions": (args.antithetic_directions, 16),
        "development_episodes": (args.development_episodes, 32),
        "validation_episodes": (args.validation_episodes, 256),
        "final_episodes": (args.final_episodes, 1024),
        "seconds": (args.seconds, 12.0),
        "takeover_seconds": (args.takeover_seconds, 0.50),
    }
    wrong = [name for name, (actual, expected) in fixed.items() if actual != expected]
    if wrong:
        raise SystemExit(f"preregistered assisted-ES values changed: {', '.join(wrong)}")
    positive = (
        args.top_candidates,
        args.sigma,
        args.sigma_decay,
        args.learning_rate,
        args.bias_perturbation_scale,
        args.log_gain_perturbation_scale,
        args.maximum_bias_delta,
        args.maximum_gain_ratio,
        args.assisted_minimum_floor_gain,
        args.joint_minimum_overall_gain,
        args.joint_minimum_floor_gain,
        args.maximum_stratum_drop,
        args.final_light_improvement,
        args.final_heavy_margin,
    )
    if min(positive) <= 0.0 or args.sigma_decay > 1.0 or args.maximum_gain_ratio <= 1.0:
        raise SystemExit("invalid assisted-ES scales, rates, or thresholds")


def parameter_masks(spec: MotorInterfaceSpec) -> dict[str, Tensor]:
    steering = torch.tensor(
        [not label.startswith("throttle_") for label in spec.labels],
        dtype=torch.bool,
        device=spec.scales.device,
    )
    throttle = ~steering
    if int(steering.sum()) != 18 or int(throttle.sum()) != 6:
        raise RuntimeError("expected 18 steering and 6 throttle motor-interface parameters")
    return {"steering": steering, "throttle": throttle, "joint": steering | throttle}


def mass_lateral_floor(summary: dict[str, Any]) -> float:
    strata = summary["success_by_stratum"]
    return min(float(strata[key]) for key in MASS_LATERAL_KEYS)


def mass_lateral_safe(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    maximum_drop: float,
) -> bool:
    before = baseline["success_by_stratum"]
    after = candidate["success_by_stratum"]
    return all(float(after[key]) >= float(before[key]) - maximum_drop for key in MASS_LATERAL_KEYS)


def assisted_progress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_floor_gain: float,
    maximum_drop: float,
) -> bool:
    return bool(
        mass_lateral_floor(candidate) >= mass_lateral_floor(baseline) + minimum_floor_gain
        and mass_lateral_safe(baseline, candidate, maximum_drop=maximum_drop)
    )


def joint_progress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_overall_gain: float,
    minimum_floor_gain: float,
    maximum_drop: float,
) -> bool:
    return bool(
        candidate["success_rate"] >= baseline["success_rate"] + minimum_overall_gain
        and mass_lateral_floor(candidate) >= mass_lateral_floor(baseline) + minimum_floor_gain
        and mass_lateral_safe(baseline, candidate, maximum_drop=maximum_drop)
    )


def candidate_rank(summary: dict[str, Any]) -> tuple[float, float, float]:
    radial = summary["crossing_radial_mean_m"]
    return (
        mass_lateral_floor(summary),
        float(summary["success_rate"]),
        -float(radial) if radial is not None else -float("inf"),
    )


def assisted_controller_step(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    neural: Tensor,
    body_specific_force: Tensor,
    stick_position: Tensor,
    interface: Tensor,
    spec: MotorInterfaceSpec,
    *,
    native_controller_forward: bool,
) -> tuple[Tensor, Tensor]:
    if native_controller_forward:
        return controller(
            image,
            roll_pitch,
            neural,
            body_specific_force,
            stick_position,
        )
    return controller_step_with_interface(
        controller,
        image,
        roll_pitch,
        neural,
        body_specific_force,
        stick_position,
        interface,
        spec,
    )


@torch.no_grad()
def evaluate_assisted_policy_batch(
    controller: ConnectomeController,
    vectors: Tensor,
    spec: MotorInterfaceSpec,
    cases: Any,
    *,
    intervention: str,
    takeover_seconds: float,
    seconds: float,
    shaping_weight_value: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    return_outcomes: bool = False,
    native_controller_forward: bool = False,
) -> dict[str, Any]:
    """Evaluate complete flights, optionally substituting the complementary teacher axes."""

    if intervention not in {"native", "reserve_throttle", "reserve_steering"}:
        raise ValueError(f"unsupported assisted-ES intervention: {intervention}")
    policies = len(vectors)
    episodes = len(cases.mass_scale)
    expanded = repeat_cases(cases, policies)
    interface = vectors.repeat_interleave(episodes, dim=0)
    device = vectors.device
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state = clone_state(expanded.state)
    stick_state = sticks.initial_state(policies * episodes, device=device, dtype=torch.float32)
    neural = controller.initial_state(policies * episodes, device=device, dtype=torch.float32)
    step_count = round(seconds / hover_config.dt)
    takeover_step = round(takeover_seconds / hover_config.dt)
    passed = torch.zeros(policies * episodes, dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    lifted = torch.zeros_like(passed)
    recontact = torch.zeros_like(passed)
    cleared = torch.zeros_like(passed)
    pass_step = torch.full((policies * episodes,), -1, dtype=torch.long, device=device)
    crossing_radial = torch.full((policies * episodes,), float("nan"), device=device)
    maximum_tilt = torch.zeros(policies * episodes, device=device)
    saturation_steps = torch.zeros(policies * episodes, device=device)
    initial_signed, initial_lateral, initial_vertical = gate_coordinates(
        state.position, expanded.gate
    )
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    maximum_progress = torch.zeros(policies * episodes, device=device)
    maximum_approach = torch.exp(
        -0.5
        * (torch.sqrt(initial_lateral.square() + initial_vertical.square()) / clean_radius).square()
    )
    for step in range(step_count):
        image = render_annular_gate(
            state,
            expanded.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        native_motor, neural = assisted_controller_step(
            controller,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
            interface,
            spec,
            native_controller_forward=native_controller_forward,
        )
        if intervention == "native" or step < takeover_step:
            motor = native_motor
        else:
            reserve_rc = teacher_rc_for_mode(
                "visual_accelerometer_reserve",
                controller,
                state,
                expanded.gate,
                expanded.mass_scale,
                hover_config,
            )
            reserve_motor = motor_target_for_rc(reserve_rc, hover_config)
            motor = compose_axis_takeover_motor(native_motor, reserve_motor, intervention)
        rc, stick_state = sticks(motor, stick_state)
        previous_position = state.position
        state = quad(rc, state, expanded.mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, expanded.gate, gate_config
        )
        new_crossing = (pass_now | collision_now | miss_now) & crossing_radial.isnan()
        new_pass = pass_now & ~passed
        if bool(new_crossing.any()):
            _, lateral, vertical = crossing_coordinates(
                previous_position, state.position, expanded.gate
            )
            radial = torch.sqrt(lateral.square() + vertical.square())
            crossing_radial[new_crossing] = radial[new_crossing]
        pass_step[new_pass] = step + 1
        not_previously_crossed = crossing_radial.isnan() | new_crossing
        signed, lateral, vertical = gate_coordinates(state.position, expanded.gate)
        progress = ((signed - initial_signed) / (-initial_signed + 0.4).clamp_min(1.0e-6)).clamp(
            0.0, 1.0
        )
        maximum_progress = torch.maximum(maximum_progress, progress)
        approach = torch.exp(
            -0.5 * (torch.sqrt(lateral.square() + vertical.square()) / clean_radius).square()
        )
        maximum_approach = torch.where(
            not_previously_crossed,
            torch.maximum(maximum_approach, approach),
            maximum_approach,
        )
        passed |= pass_now
        collision |= collision_now
        missed |= miss_now
        cleared |= passed & (signed >= 0.4)
        lifted |= state.position[:, 2] > 0.15
        recontact |= lifted & (state.position[:, 2] <= 0.01)
        maximum_tilt = torch.maximum(
            maximum_tilt,
            torch.linalg.vector_norm(state.euler[:, :2], dim=1),
        )
        saturation_steps += (stick_state.position.abs() > 0.98).any(dim=1)
    has_post_pass_second = (pass_step >= 0) & (
        pass_step <= step_count - round(1.0 / hover_config.dt)
    )
    saturation_fraction = saturation_steps / step_count
    sustained_saturation = saturation_fraction > 0.25
    success = (
        passed
        & cleared
        & has_post_pass_second
        & ~collision
        & ~missed
        & ~recontact
        & (maximum_tilt <= math.radians(40.0))
        & ~sustained_saturation
    )
    clean_pass = passed & ~collision & ~missed
    hazard = collision | recontact
    reward = (
        10.0 * success.float()
        + 2.0 * clean_pass.float()
        + shaping_weight_value * (maximum_progress + maximum_approach)
        - 2.0 * hazard.float()
        - 0.1 * saturation_fraction
    )
    shape = (policies, episodes)
    reshaped = {
        "success": success.reshape(shape),
        "passed": clean_pass.reshape(shape),
        "collision": collision.reshape(shape),
        "missed": missed.reshape(shape),
        "crossing_radial": crossing_radial.reshape(shape),
        "progress": maximum_progress.reshape(shape),
        "approach": maximum_approach.reshape(shape),
        "hazard": hazard.reshape(shape),
        "saturation_fraction": saturation_fraction.reshape(shape),
        "reward": reward.reshape(shape),
        "codes": expanded.stratum_code.reshape(shape),
    }
    result: dict[str, Any] = {"summaries": summarize_policy_batch(**reshaped)}
    if return_outcomes:
        result["outcomes"] = {
            name: value.detach().cpu()
            for name, value in reshaped.items()
            if name in {"success", "reward", "codes"}
        }
    return result


def compact_search(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": summary["success_rate"],
        "worst_mass_lateral_success_rate": mass_lateral_floor(summary),
        "ring_collision_rate": summary["ring_collision_rate"],
        "miss_rate": summary["miss_rate"],
        "crossing_radial_mean_m": summary["crossing_radial_mean_m"],
        "success_by_stratum": summary["success_by_stratum"],
        "fitness": summary["fitness"],
    }


def native_evaluator_parity(
    controller: ConnectomeController,
    vectors: Tensor,
    spec: MotorInterfaceSpec,
    cases: Any,
    *,
    seconds: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    zero = vectors[:1]
    batched = evaluate_assisted_policy_batch(
        controller,
        zero,
        spec,
        cases,
        intervention="native",
        takeover_seconds=0.5,
        seconds=seconds,
        shaping_weight_value=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summaries"][0]
    standard, _ = evaluate_controller(
        controller,
        cases,
        seconds=seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    checks = {
        "success_rate_exact": batched["success_rate"] == standard["success_rate"],
        "ring_collision_rate_exact": (
            batched["ring_collision_rate"] == standard["ring_collision_rate"]
        ),
        "miss_rate_exact": batched["miss_rate"] == standard["miss_rate"],
        "crossing_radial_within_1e-5": abs(
            float(batched["crossing_radial_mean_m"]) - float(standard["crossing_radial_mean_m"])
        )
        <= 1.0e-5,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "batched": compact_search(batched),
        "standard": compact_flight(standard),
    }


def development_rank(item: dict[str, Any]) -> tuple[float, float, float]:
    summary = item["development"]
    return float(summary["fitness"]), *candidate_rank(summary)[:2]


def nominate_finalists(
    archive: dict[str, dict[str, Any]],
    *,
    required_keys: tuple[str, ...],
    limit: int,
) -> list[str]:
    if any(key not in archive for key in required_keys):
        raise KeyError("required validation nominee is absent from the archive")
    ranked = sorted(archive, key=lambda key: development_rank(archive[key]), reverse=True)
    finalists = []
    for key in (*required_keys, *ranked):
        if key not in finalists:
            finalists.append(key)
        if len(finalists) == min(limit, len(archive)):
            break
    return finalists


def select_validated_candidate(
    stage: str,
    baseline: dict[str, Any],
    validated: dict[str, dict[str, Any]],
    *,
    assisted_minimum_floor_gain: float,
    joint_minimum_overall_gain: float,
    joint_minimum_floor_gain: float,
    maximum_drop: float,
) -> tuple[str | None, bool]:
    safe = {
        key: item
        for key, item in validated.items()
        if mass_lateral_safe(baseline, item["metrics"], maximum_drop=maximum_drop)
    }
    if stage == "joint":
        qualified = {
            key: item
            for key, item in safe.items()
            if joint_progress(
                baseline,
                item["metrics"],
                minimum_overall_gain=joint_minimum_overall_gain,
                minimum_floor_gain=joint_minimum_floor_gain,
                maximum_drop=maximum_drop,
            )
        }
    else:
        qualified = {
            key: item
            for key, item in safe.items()
            if assisted_progress(
                baseline,
                item["metrics"],
                minimum_floor_gain=assisted_minimum_floor_gain,
                maximum_drop=maximum_drop,
            )
        }
    selection_pool = qualified or safe
    selected = (
        max(selection_pool, key=lambda key: candidate_rank(selection_pool[key]["metrics"]))
        if selection_pool
        else None
    )
    return selected, bool(selected and selected in qualified)


def validate_search_archive(
    controller: ConnectomeController,
    spec: MotorInterfaceSpec,
    archive: dict[str, dict[str, Any]],
    zero: Tensor,
    *,
    required_keys: tuple[str, ...],
    stage: str,
    generation: int,
    intervention: str,
    validation_seed: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    finalist_keys = nominate_finalists(
        archive,
        required_keys=required_keys,
        limit=args.top_candidates,
    )
    vectors = torch.stack([zero, *(archive[key]["vector"].to(device) for key in finalist_keys)])
    cases = diverse_matched_cases(
        args.validation_episodes,
        seed=validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    batch = evaluate_assisted_policy_batch(
        controller,
        vectors,
        spec,
        cases,
        intervention=intervention,
        takeover_seconds=args.takeover_seconds,
        seconds=args.seconds,
        shaping_weight_value=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    baseline = batch["summaries"][0]
    candidates = {}
    for key, summary in zip(finalist_keys, batch["summaries"][1:], strict=True):
        candidates[key] = {
            "source": {name: value for name, value in archive[key].items() if name != "vector"},
            "metrics": summary,
            "mass_lateral_safe": mass_lateral_safe(
                baseline,
                summary,
                maximum_drop=args.maximum_stratum_drop,
            ),
        }
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "stage": stage,
                    "generation": generation,
                    "candidate": key,
                    **compact_search(summary),
                }
            ),
            flush=True,
        )
    return {
        "stage": stage,
        "generation": generation,
        "validation_seed": validation_seed,
        "baseline": baseline,
        "candidates": candidates,
    }


def run_search(
    controller: ConnectomeController,
    spec: MotorInterfaceSpec,
    *,
    stage: str,
    intervention: str,
    active_mask: Tensor,
    initial_center: Tensor,
    generations_budget: int,
    development_seed: int,
    validation_seed: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    output_dir: Path,
) -> dict[str, Any]:
    zero = torch.zeros(len(spec.labels), device=device)
    center = initial_center.clone()
    rng = np.random.default_rng(args.search_seed + sum(ord(character) for character in stage))
    archive: dict[str, dict[str, Any]] = {}
    generation_records = []
    validations = []
    validated: dict[str, dict[str, Any]] = {}
    checkpoint_progress_passed = False
    stopped_at_checkpoint = False
    for generation in range(1, generations_budget + 1):
        evaluated_center = center.clone()
        sigma = args.sigma * args.sigma_decay ** (generation - 1)
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_directions, len(spec.labels))).astype(np.float32)
        ).to(device)
        epsilon[:, ~active_mask] = 0.0
        delta = sigma * spec.scales * epsilon
        plus = (center + delta).clamp(spec.lower, spec.upper)
        minus = (center - delta).clamp(spec.lower, spec.upper)
        candidates = torch.stack((plus, minus), dim=1).reshape(-1, len(spec.labels))
        policies = torch.cat((zero[None], evaluated_center[None], candidates), dim=0)
        cases = diverse_matched_cases(
            args.development_episodes,
            seed=development_seed + generation - 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        weight = shaping_weight(generation, generations_budget)
        batch = evaluate_assisted_policy_batch(
            controller,
            policies,
            spec,
            cases,
            intervention=intervention,
            takeover_seconds=args.takeover_seconds,
            seconds=args.seconds,
            shaping_weight_value=weight,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        baseline_summary, center_summary = batch["summaries"][:2]
        candidate_summaries = batch["summaries"][2:]
        fitness = np.asarray(
            [summary["fitness"] for summary in candidate_summaries], dtype=np.float32
        )
        ranks = np.empty(len(fitness), dtype=np.float32)
        ranks[np.argsort(fitness)] = np.linspace(-0.5, 0.5, len(fitness))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        center = (center + args.learning_rate * spec.scales * gradient).clamp(
            spec.lower, spec.upper
        )
        center[~active_mask] = initial_center[~active_mask]
        for index, vector in enumerate(candidates):
            key = vector_sha256(vector)
            archive[key] = {
                "vector": vector.detach().cpu().clone(),
                "generation": generation,
                "kind": f"direction_{index // 2 + 1}_{'plus' if index % 2 == 0 else 'minus'}",
                "development": candidate_summaries[index],
            }
        center_key = vector_sha256(evaluated_center)
        archive[center_key] = {
            "vector": evaluated_center.detach().cpu().clone(),
            "generation": generation,
            "kind": "evaluated_center",
            "development": center_summary,
        }
        best_index = int(np.argmax(fitness))
        best_key = vector_sha256(candidates[best_index])
        record = {
            "generation": generation,
            "development_seed": development_seed + generation - 1,
            "sigma": sigma,
            "shaping_weight": weight,
            "baseline": baseline_summary,
            "center": center_summary,
            "best_candidate": candidate_summaries[best_index],
            "best_candidate_index": best_index + 1,
            "next_center_sha256": vector_sha256(center),
        }
        generation_records.append(record)
        print(
            json.dumps(
                {
                    "phase": "search",
                    "stage": stage,
                    "generation": generation,
                    "baseline_success": baseline_summary["success_rate"],
                    "center_success": center_summary["success_rate"],
                    "best_success": candidate_summaries[best_index]["success_rate"],
                    "baseline_floor": mass_lateral_floor(baseline_summary),
                    "best_floor": mass_lateral_floor(candidate_summaries[best_index]),
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "center_sha256": vector_sha256(center),
                "parameter_labels": spec.labels,
            },
            output_dir / f"{stage}-search-state.pt",
        )
        if generation % args.validation_interval == 0:
            validation = validate_search_archive(
                controller,
                spec,
                archive,
                zero,
                required_keys=(center_key, best_key),
                stage=stage,
                generation=generation,
                intervention=intervention,
                validation_seed=validation_seed,
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            validations.append(validation)
            for key, item in validation["candidates"].items():
                previous = validated.get(key)
                if previous is None or candidate_rank(item["metrics"]) > candidate_rank(
                    previous["metrics"]
                ):
                    validated[key] = item
            if generation == args.checkpoint_generation:
                if stage == "joint":
                    checkpoint_progress_passed = any(
                        joint_progress(
                            validation["baseline"],
                            item["metrics"],
                            minimum_overall_gain=args.joint_minimum_overall_gain,
                            minimum_floor_gain=args.joint_minimum_floor_gain,
                            maximum_drop=args.maximum_stratum_drop,
                        )
                        for item in validated.values()
                    )
                else:
                    checkpoint_progress_passed = any(
                        assisted_progress(
                            validation["baseline"],
                            item["metrics"],
                            minimum_floor_gain=args.assisted_minimum_floor_gain,
                            maximum_drop=args.maximum_stratum_drop,
                        )
                        for item in validated.values()
                    )
                if not checkpoint_progress_passed:
                    stopped_at_checkpoint = True
                    break
    if not validations:
        raise RuntimeError(f"{stage} search produced no fixed validation")
    validation_baseline = validations[-1]["baseline"]
    selected_key, selected_progress_qualified = select_validated_candidate(
        stage,
        validation_baseline,
        validated,
        assisted_minimum_floor_gain=args.assisted_minimum_floor_gain,
        joint_minimum_overall_gain=args.joint_minimum_overall_gain,
        joint_minimum_floor_gain=args.joint_minimum_floor_gain,
        maximum_drop=args.maximum_stratum_drop,
    )
    selected_vector = archive[selected_key]["vector"].to(device) if selected_key else zero
    ranked_archive = sorted(archive, key=lambda key: development_rank(archive[key]), reverse=True)
    return {
        "stage": stage,
        "intervention": intervention,
        "active_parameter_count": int(active_mask.sum()),
        "generations_budget": generations_budget,
        "generations_completed": len(generation_records),
        "stopped_at_checkpoint": stopped_at_checkpoint,
        "checkpoint_progress_passed": checkpoint_progress_passed,
        "generations": generation_records,
        "validations": validations,
        "validation_baseline": validation_baseline,
        "selected_key": selected_key,
        "selected_progress_qualified": selected_progress_qualified,
        "selected_metrics": (
            validated[selected_key]["metrics"] if selected_key else validation_baseline
        ),
        "selected_vector_sha256": vector_sha256(selected_vector),
        "selected_vector": selected_vector.detach().cpu().tolist(),
        "top_archive": [
            {
                "sha256": key,
                **{name: value for name, value in archive[key].items() if name != "vector"},
            }
            for key in ranked_archive[:20]
        ],
        "_selected_vector": selected_vector,
        "_archive": archive,
    }


def evaluate_fixed_vectors(
    controller: ConnectomeController,
    spec: MotorInterfaceSpec,
    vectors: dict[str, Tensor],
    *,
    seed: int,
    intervention: str,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, dict[str, Any]]:
    labels = list(vectors)
    cases = diverse_matched_cases(
        args.validation_episodes,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    batch = evaluate_assisted_policy_batch(
        controller,
        torch.stack([vectors[label] for label in labels]),
        spec,
        cases,
        intervention=intervention,
        takeover_seconds=args.takeover_seconds,
        seconds=args.seconds,
        shaping_weight_value=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    return dict(zip(labels, batch["summaries"], strict=True))


def choose_merge_entry(
    controller: ConnectomeController,
    spec: MotorInterfaceSpec,
    merged: Tensor,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[dict[str, Any], Tensor | None]:
    zero = torch.zeros_like(merged)
    initial = evaluate_fixed_vectors(
        controller,
        spec,
        {"source": zero, "full_merge": merged},
        seed=args.merge_validation_seed,
        intervention="native",
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    source = initial["source"]
    full_safe = mass_lateral_safe(
        source,
        initial["full_merge"],
        maximum_drop=args.maximum_stratum_drop,
    )
    tested = initial
    if full_safe:
        selected_label = "full_merge"
    else:
        interpolated = evaluate_fixed_vectors(
            controller,
            spec,
            {
                "source": zero,
                "merge_25_percent": 0.25 * merged,
                "merge_50_percent": 0.50 * merged,
                "merge_75_percent": 0.75 * merged,
            },
            seed=args.merge_validation_seed,
            intervention="native",
            args=args,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        tested = {"full_merge": initial["full_merge"], **interpolated}
        fractions = {
            "merge_25_percent": 0.25,
            "merge_50_percent": 0.50,
            "merge_75_percent": 0.75,
        }
        safe_labels = [
            label
            for label in fractions
            if mass_lateral_safe(
                source,
                tested[label],
                maximum_drop=args.maximum_stratum_drop,
            )
        ]
        selected_label = (
            max(safe_labels, key=lambda label: candidate_rank(tested[label]))
            if safe_labels
            else None
        )
    if selected_label == "full_merge":
        selected = merged
    elif selected_label:
        selected = float(selected_label.split("_")[1]) / 100.0 * merged
    else:
        selected = None
    report = {
        "validation_seed": args.merge_validation_seed,
        "full_merge_safe": full_safe,
        "interpolations_tested": not full_safe,
        "tested": tested,
        "selected_label": selected_label,
        "selected_vector_sha256": vector_sha256(selected) if selected is not None else None,
        "joint_entry_allowed": selected is not None and bool(selected.abs().max() > 0.0),
    }
    return report, selected


def save_candidate_checkpoint(
    path: Path,
    source_checkpoint: dict[str, Any],
    checkpoint_path: Path,
    factorial_report_path: Path,
    controller: ConnectomeController,
    vector: Tensor,
    spec: MotorInterfaceSpec,
    promotion: dict[str, Any],
) -> None:
    saved = copy.deepcopy(source_checkpoint)
    saved["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    saved["source_checkpoint_sha256"] = file_sha256(checkpoint_path)
    saved["assisted_motor_es"] = {
        "method": "teacher-assisted staged complete-flight ES with native joint polish",
        "factorial_report_sha256": file_sha256(factorial_report_path),
        "parameter_labels": list(spec.labels),
        "parameter_vector": vector.detach().cpu().tolist(),
        "parameter_vector_sha256": vector_sha256(vector),
        "compiled_into_native_parameters": True,
        "teacher_used_at_deployment": False,
        "promotion": promotion,
    }
    torch.save(saved, path)


def public_search_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {key: value for key, value in result.items() if not key.startswith("_")}


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    factorial = json.loads(args.factorial_report.read_text())
    if (
        factorial.get("checkpoint_sha256") != file_sha256(args.checkpoint)
        or not factorial.get("classification", {}).get("passed")
        or factorial.get("classification", {}).get("interpretation")
        != "coupled_throttle_and_steering_takeover_required"
    ):
        raise SystemExit("assisted ES requires the passed coupled axis-takeover diagnostic")
    source_parameter_sha256 = controller_parameter_sha256(controller)
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    spec = motor_interface_spec(
        controller,
        bias_scale=args.bias_perturbation_scale,
        log_gain_scale=args.log_gain_perturbation_scale,
        maximum_bias_delta=args.maximum_bias_delta,
        maximum_gain_ratio=args.maximum_gain_ratio,
    )
    masks = parameter_masks(spec)
    zero = torch.zeros(len(spec.labels), device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    vector_path = args.output_dir / "selected-vector.json"
    candidate_path = args.output_dir / "candidate.pt"
    if any(path.exists() for path in (report_path, vector_path, candidate_path)):
        raise SystemExit("output directory contains a stale report, vector, or candidate")
    seed_everything(args.search_seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    parity_cases = diverse_matched_cases(
        args.development_episodes,
        seed=args.merge_validation_seed - 1,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    parity = native_evaluator_parity(
        controller,
        zero[None],
        spec,
        parity_cases,
        seconds=args.seconds,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if not parity["passed"]:
        raise RuntimeError(f"assisted evaluator native parity failed: {parity['checks']}")

    steering = run_search(
        controller,
        spec,
        stage="steering",
        intervention="reserve_throttle",
        active_mask=masks["steering"],
        initial_center=zero,
        generations_budget=args.assisted_generations,
        development_seed=args.steering_development_seed,
        validation_seed=args.steering_validation_seed,
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        output_dir=args.output_dir,
    )
    throttle = run_search(
        controller,
        spec,
        stage="throttle",
        intervention="reserve_steering",
        active_mask=masks["throttle"],
        initial_center=zero,
        generations_budget=args.assisted_generations,
        development_seed=args.throttle_development_seed,
        validation_seed=args.throttle_validation_seed,
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        output_dir=args.output_dir,
    )
    assisted_stages_passed = bool(
        steering["checkpoint_progress_passed"] and throttle["checkpoint_progress_passed"]
    )
    merge = None
    merge_entry = None
    joint = None
    if assisted_stages_passed:
        steering_vector = steering["_selected_vector"]
        throttle_vector = throttle["_selected_vector"]
        if bool(steering_vector[masks["throttle"]].abs().max() > 0.0) or bool(
            throttle_vector[masks["steering"]].abs().max() > 0.0
        ):
            raise RuntimeError("assisted stage changed parameters outside its declared mask")
        merged = steering_vector + throttle_vector
        merge, merge_entry = choose_merge_entry(
            controller,
            spec,
            merged,
            args=args,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        if merge["joint_entry_allowed"] and merge_entry is not None:
            joint = run_search(
                controller,
                spec,
                stage="joint",
                intervention="native",
                active_mask=masks["joint"],
                initial_center=merge_entry,
                generations_budget=args.joint_generations,
                development_seed=args.joint_development_seed,
                validation_seed=args.joint_validation_seed,
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
                output_dir=args.output_dir,
            )

    completed_joint = bool(
        joint is not None
        and joint["checkpoint_progress_passed"]
        and joint["generations_completed"] == args.joint_generations
    )
    selected_vector = joint["_selected_vector"] if completed_joint else zero
    final = None
    paired_final = None
    promotion = {
        "checks": {
            "assisted_stages_passed": assisted_stages_passed,
            "safe_nonzero_merge_entry": bool(merge and merge["joint_entry_allowed"]),
            "completed_joint_polish": completed_joint,
        },
        "passed": False,
    }
    goal_checks = None
    goal_passed = False
    if completed_joint:
        candidate = copy.deepcopy(controller).to(device)
        compile_vector(candidate, base_state, selected_vector, spec)
        final_cases = diverse_matched_cases(
            args.final_episodes,
            seed=args.final_seed,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            extreme_fraction=0.5,
        )
        final = {}
        outcomes = {}
        for name, policy, controls in (
            ("reference", controller, {}),
            ("candidate", candidate, {}),
            ("candidate_frozen_first_frame", candidate, {"frozen_visual": True}),
            ("candidate_constant_1g", candidate, {"constant_acceleration": True}),
            (
                "candidate_pair_swapped_acceleration",
                candidate,
                {"pair_swapped_acceleration": True},
            ),
        ):
            summary, tensors = evaluate_controller(
                policy,
                final_cases,
                seconds=args.seconds,
                resolution=resolution,
                hover_config=hover_config,
                gate_config=gate_config,
                **controls,
            )
            final[name] = summary
            outcomes[name] = tensors
            print(
                json.dumps({"phase": "final", "name": name, **compact_flight(summary)}), flush=True
            )
        reference_success = outcomes["reference"]["success"]
        candidate_success = outcomes["candidate"]["success"]
        light = outcomes["reference"]["mass_scale"] < 1.0
        paired_final = {
            "overall_success_difference": paired_clustered_confidence_interval(
                reference_success, candidate_success
            ),
            "light_success_difference": paired_confidence_interval(
                reference_success[light], candidate_success[light]
            ),
            "heavy_success_difference": paired_confidence_interval(
                reference_success[~light], candidate_success[~light]
            ),
        }
        promotion_checks = {
            "assisted_stages_passed": True,
            "safe_nonzero_merge_entry": True,
            "completed_joint_polish": True,
            "light_improvement_at_least_threshold": (
                final["candidate"]["light_success_rate"]
                >= final["reference"]["light_success_rate"] + args.final_light_improvement
            ),
            "light_paired_confidence_interval_excludes_zero": (
                paired_final["light_success_difference"]["confidence_95"][0] > 0.0
            ),
            "heavy_nondegradation_within_margin": (
                final["candidate"]["heavy_success_rate"]
                >= final["reference"]["heavy_success_rate"] - args.final_heavy_margin
            ),
            "heavy_paired_noninferiority_interval_within_margin": (
                paired_final["heavy_success_difference"]["confidence_95"][0]
                >= -args.final_heavy_margin
            ),
            "overall_success_improves": (
                final["candidate"]["success_rate"] > final["reference"]["success_rate"]
            ),
            "overall_paired_confidence_interval_excludes_zero": (
                paired_final["overall_success_difference"]["confidence_95"][0] > 0.0
            ),
            "frozen_first_frame_success_at_most_five_percent": (
                final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
            ),
        }
        promotion = {"checks": promotion_checks, "passed": all(promotion_checks.values())}
        if promotion["passed"]:
            save_candidate_checkpoint(
                candidate_path,
                checkpoint,
                args.checkpoint,
                args.factorial_report,
                candidate,
                selected_vector,
                spec,
                promotion,
            )
        goal_checks = {key: final["candidate"][key] >= 0.90 for key in GOAL_RATE_KEYS}
        goal_passed = bool(promotion["passed"] and all(goal_checks.values()))

    vector = {
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "selected_vector_sha256": vector_sha256(selected_vector),
        "parameter_labels": list(spec.labels),
        "selected_vector": selected_vector.detach().cpu().tolist(),
        "compiled_parameter_sha256": (
            controller_parameter_sha256(candidate) if completed_joint else source_parameter_sha256
        ),
    }
    vector_path.write_text(json.dumps(vector, indent=2, sort_keys=True) + "\n")
    archive_path = args.output_dir / "archive.pt"
    torch.save(
        {
            "steering": steering["_archive"],
            "throttle": throttle["_archive"],
            "joint": joint["_archive"] if joint is not None else None,
        },
        archive_path,
    )
    report = {
        "method": "staged teacher-assisted complete-flight ES with native joint polish",
        "claim_scope": (
            "Reserve-axis assistance is training-only. Any evaluated or promoted candidate "
            "uses current FPV, roll/pitch, body-Z specific force, and native connectome state."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "checkpoint": stable_path(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "factorial_report": stable_path(args.factorial_report),
        "factorial_report_sha256": file_sha256(args.factorial_report),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "steering_intervention": "mass-free reserve throttle after 0.5 seconds",
            "throttle_intervention": "mass-free reserve roll/pitch/yaw after 0.5 seconds",
            "assisted_stages_start_independently_from_source": True,
            "core_weights_outside_motor_interface_frozen": True,
            "time_constants_frozen": True,
            "fixed_topology_and_transmitter_signs": True,
            "assisted_candidates_ineligible_for_promotion": True,
            "native_merge_required": True,
            "native_joint_polish_required": True,
            "mass_actor_input": False,
            "clock_actor_input": False,
            "engineered_history_features": False,
            "added_recurrent_module": False,
        },
        "parameterization": {
            "count": len(spec.labels),
            "steering_count": int(masks["steering"].sum()),
            "throttle_count": int(masks["throttle"].sum()),
            "labels": list(spec.labels),
            "description": list(spec.description),
        },
        "native_evaluator_parity": parity,
        "steering_stage": public_search_result(steering),
        "throttle_stage": public_search_result(throttle),
        "assisted_stages_passed": assisted_stages_passed,
        "merge": merge,
        "joint_stage": public_search_result(joint),
        "completed_joint_polish": completed_joint,
        "selected_vector": stable_path(vector_path),
        "selected_vector_sha256": file_sha256(vector_path),
        "archive": stable_path(archive_path),
        "archive_sha256": file_sha256(archive_path),
        "final": final,
        "paired_final": paired_final,
        "promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": (
            file_sha256(candidate_path) if promotion["passed"] else None
        ),
        "goal_threshold_checks": goal_checks,
        "goal_passed": goal_passed,
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
                "assisted_stages_passed": assisted_stages_passed,
                "joint_entry_allowed": bool(merge and merge["joint_entry_allowed"]),
                "completed_joint_polish": completed_joint,
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
