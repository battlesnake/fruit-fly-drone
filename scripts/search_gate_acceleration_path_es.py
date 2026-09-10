#!/usr/bin/env python3
"""Search complete short acceleration-to-throttle paths with matched-mass ES."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_gate_motor_interface_es import (  # noqa: E402
    BalancedCases,
    clone_state,
    load_controller,
    paired_confidence_interval,
    repeat_cases,
    shaping_weight,
    stable_path,
    standard_compact,
    vector_sha256,
)
from train_gate import (  # noqa: E402
    edges_on_short_paths,
    evaluate_gate,
    file_sha256,
    initial_rollout,
    seed_everything,
)

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    gate_coordinates,
    render_annular_gate,
    wrap_angle,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
)


@dataclass(frozen=True)
class PathSpec:
    edges: Tensor
    baseline: Tensor
    scales: Tensor
    lower: Tensor
    upper: Tensor
    zero_edges: Tensor
    floor: float
    maximum: float


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
        default=REPO_ROOT / "runs" / "gate" / "acceleration-path-es-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-hops", type=int, default=4)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--checkpoint-generation", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--antithetic-directions", type=int, default=32)
    parser.add_argument("--development-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--search-seconds", type=float, default=8.0)
    parser.add_argument("--final-seconds", type=float, default=12.0)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--sigma-decay", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=0.012)
    parser.add_argument("--maximum-update-sigma-fraction", type=float, default=0.25)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.25)
    parser.add_argument("--maximum-edge-magnitude", type=float, default=8.0)
    parser.add_argument("--heavy-reward-penalty", type=float, default=2.0)
    parser.add_argument("--checkpoint-light-improvement", type=float, default=0.05)
    parser.add_argument("--heavy-noninferiority-margin", type=float, default=0.02)
    parser.add_argument("--final-light-improvement", type=float, default=0.10)
    parser.add_argument("--development-seed", type=int, default=700_031)
    parser.add_argument("--validation-seed", type=int, default=710_031)
    parser.add_argument("--final-seed", type=int, default=720_031)
    parser.add_argument("--search-seed", type=int, default=73_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.maximum_hops,
        args.generations,
        args.checkpoint_generation,
        args.validation_interval,
        args.antithetic_directions,
        args.development_episodes,
        args.validation_episodes,
        args.final_episodes,
        args.search_seconds,
        args.final_seconds,
        args.top_candidates,
        args.sigma,
        args.sigma_decay,
        args.learning_rate,
        args.maximum_update_sigma_fraction,
        args.maximum_edge_magnitude,
        args.heavy_reward_penalty,
        args.checkpoint_light_improvement,
        args.heavy_noninferiority_margin,
        args.final_light_improvement,
    )
    if min(positive) <= 0.0:
        raise SystemExit("search sizes, scales, and thresholds must be positive")
    if args.generations < args.checkpoint_generation:
        raise SystemExit("--generations must reach --checkpoint-generation")
    if args.checkpoint_generation % args.validation_interval:
        raise SystemExit("checkpoint generation must coincide with validation")
    if not 0.0 < args.sigma_decay <= 1.0:
        raise SystemExit("--sigma-decay must be in (0, 1]")
    if not 0.0 <= args.scale_floor_quantile <= 1.0:
        raise SystemExit("--scale-floor-quantile must be in [0, 1]")
    for name in ("development_episodes", "validation_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")


def make_path_spec(
    controller: ConnectomeController,
    graph_path: Path,
    *,
    maximum_hops: int,
    floor_quantile: float,
    maximum_magnitude: float,
) -> PathSpec:
    graph = np.load(graph_path)
    pool_offsets = graph["output_pool_offsets"]
    pool_indices = graph["output_pool_indices"]
    throttle = pool_indices[pool_offsets[6] : pool_offsets[8]]
    mask = edges_on_short_paths(
        graph["edge_pre"],
        graph["edge_post"],
        len(graph["node_ids"]),
        graph["acceleration_node_indices"],
        throttle,
        maximum_hops,
    )
    selected = np.flatnonzero(mask)
    edges = torch.from_numpy(selected).to(controller.edge_pre.device, dtype=torch.long)
    baseline = controller.edge_magnitude[edges].detach().clone()
    nonzero = baseline[baseline > 0.0]
    if not len(nonzero):
        raise SystemExit("selected path contains no nonzero edge magnitudes")
    floor = float(torch.quantile(nonzero, floor_quantile))
    scales = baseline.clamp_min(floor)
    lower = -baseline / scales
    upper = (maximum_magnitude - baseline) / scales
    return PathSpec(
        edges=edges,
        baseline=baseline,
        scales=scales,
        lower=lower,
        upper=upper,
        zero_edges=baseline == 0.0,
        floor=floor,
        maximum=maximum_magnitude,
    )


def sample_matched_cases(
    episodes: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> BalancedCases:
    """Return adjacent light/heavy cases with exactly matched gate geometry."""

    if episodes % 8:
        raise ValueError("matched cases require episodes divisible by eight")
    seed_everything(seed)
    pairs = episodes // 2
    source_state, source_gate, sampled_mass = initial_rollout(
        pairs,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    geometry = torch.arange(pairs, device=device) % 4
    lateral_sign = torch.where(geometry.bitwise_and(1).bool(), 1.0, -1.0)
    obliquity_sign = torch.where(geometry.bitwise_and(2).bool(), 1.0, -1.0)
    center = source_gate.center.clone()
    center[:, 1] = center[:, 1].abs() * lateral_sign
    bearing = torch.atan2(center[:, 1], center[:, 0])
    sampled_bearing = torch.atan2(source_gate.center[:, 1], source_gate.center[:, 0])
    obliquity = wrap_angle(source_gate.yaw - sampled_bearing).abs()
    gate = AnnularGate(center=center, yaw=bearing + obliquity_sign * obliquity)
    mass_delta = (sampled_mass - 1.0).abs()
    mass_sign = torch.tensor((-1.0, 1.0), device=device).repeat(pairs)
    repeated_geometry = geometry.repeat_interleave(2)
    mass_bit = torch.arange(episodes, device=device).bitwise_and(1)
    codes = mass_bit + 2 * repeated_geometry.bitwise_and(1) + 2 * repeated_geometry.bitwise_and(2)
    return BalancedCases(
        state=QuadState(*(value.repeat_interleave(2, dim=0) for value in source_state.as_tuple())),
        gate=AnnularGate(
            center=gate.center.repeat_interleave(2, dim=0),
            yaw=gate.yaw.repeat_interleave(2),
        ),
        mass_scale=1.0 + mass_sign * mass_delta.repeat_interleave(2),
        stratum_code=codes,
    )


def controller_step_with_paths(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    state: Tensor,
    body_specific_force: Tensor,
    stick_position: Tensor,
    theta: Tensor,
    spec: PathSpec,
) -> tuple[Tensor, Tensor]:
    activity = torch.tanh(state)
    messages = activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude
    magnitudes = (spec.baseline + spec.scales * theta).clamp(0.0, spec.maximum)
    messages[:, spec.edges] = (
        activity[:, controller.edge_pre[spec.edges]] * controller.edge_sign[spec.edges] * magnitudes
    )
    recurrent = torch.zeros_like(state).index_add(1, controller.edge_post, messages)
    drive = (
        recurrent
        + controller.bias
        + controller.sensory_drive(image, roll_pitch, body_specific_force, stick_position)
    )
    target = 5.0 * torch.tanh(drive / 5.0)
    alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant)
    next_state = state + alpha * (target - state)
    return controller.motor_drive(next_state), next_state


def _mean(values: Tensor, mask: Tensor) -> float | None:
    return float(values[mask].mean()) if bool(mask.any()) else None


def summarize(
    *,
    success: Tensor,
    passed: Tensor,
    collision: Tensor,
    missed: Tensor,
    crossing_radial: Tensor,
    progress: Tensor,
    approach: Tensor,
    hazard: Tensor,
    saturation_fraction: Tensor,
    reward: Tensor,
    codes: Tensor,
    throttle_bins: Tensor,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for policy in range(success.shape[0]):
        lower = ~codes[policy].bitwise_and(1).bool()
        negative_side = ~codes[policy].bitwise_and(2).bool()
        negative_obliquity = ~codes[policy].bitwise_and(4).bool()
        crossed = ~crossing_radial[policy].isnan()
        mean_reward = float(reward[policy].mean())
        summaries.append(
            {
                "mean_reward": mean_reward,
                "light_mean_reward": _mean(reward[policy], lower),
                "heavy_mean_reward": _mean(reward[policy], ~lower),
                "success_rate": float(success[policy].float().mean()),
                "light_success_rate": _mean(success[policy].float(), lower),
                "heavy_success_rate": _mean(success[policy].float(), ~lower),
                "negative_lateral_success_rate": _mean(success[policy].float(), negative_side),
                "positive_lateral_success_rate": _mean(success[policy].float(), ~negative_side),
                "negative_obliquity_success_rate": _mean(
                    success[policy].float(), negative_obliquity
                ),
                "positive_obliquity_success_rate": _mean(
                    success[policy].float(), ~negative_obliquity
                ),
                "clean_pass_rate": float(passed[policy].float().mean()),
                "ring_collision_rate": float(collision[policy].float().mean()),
                "miss_rate": float(missed[policy].float().mean()),
                "plane_crossing_rate": float(crossed.float().mean()),
                "progress_mean": float(progress[policy].mean()),
                "approach_score_mean": float(approach[policy].mean()),
                "hazard_rate": float(hazard[policy].float().mean()),
                "stick_saturation_fraction": float(saturation_fraction[policy].mean()),
                "crossing_radial_mean_m": _mean(crossing_radial[policy], crossed),
                "early_throttle_stick_mean_by_mass": {
                    "bins_seconds": [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0], [1.0, 1.5]],
                    "light": [_mean(throttle_bins[policy, :, index], lower) for index in range(4)],
                    "heavy": [_mean(throttle_bins[policy, :, index], ~lower) for index in range(4)],
                },
            }
        )
    return summaries


@torch.no_grad()
def evaluate_policy_batch(
    controller: ConnectomeController,
    theta: Tensor,
    spec: PathSpec,
    cases: BalancedCases,
    *,
    seconds: float,
    reward_shaping_weight: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    acceleration_control: str = "live",
    axis_intervention: str = "native",
    takeover_seconds: float = 0.5,
    return_outcomes: bool = False,
) -> dict[str, Any]:
    if acceleration_control not in {"live", "constant_1g", "pair_swapped"}:
        raise ValueError(f"unknown acceleration control: {acceleration_control}")
    if axis_intervention not in {"native", "reserve_steering"}:
        raise ValueError(f"unknown axis intervention: {axis_intervention}")
    teacher_rc_for_mode = None
    if axis_intervention == "reserve_steering":
        # Imported lazily because the teacher module routes its case generation
        # through this module.
        from audit_gate_analytic_teachers import teacher_rc_for_mode as teacher

        teacher_rc_for_mode = teacher
    policies, episodes = len(theta), len(cases.mass_scale)
    expanded = repeat_cases(cases, policies)
    policy_theta = theta.repeat_interleave(episodes, dim=0)
    device = theta.device
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
    del initial_lateral, initial_vertical
    clean_radius = gate_config.inner_radius - gate_config.drone_radius
    maximum_progress = torch.zeros(policies * episodes, device=device)
    signed, lateral, vertical = gate_coordinates(state.position, expanded.gate)
    del signed
    maximum_approach = torch.exp(
        -0.5 * (torch.sqrt(lateral.square() + vertical.square()) / clean_radius).square()
    )
    throttle_sum = torch.zeros(policies * episodes, 4, device=device)
    throttle_count = torch.zeros(4, device=device)
    bin_limits = ((0, 25), (25, 50), (50, 100), (100, 150))
    pair_swap = torch.arange(policies * episodes, device=device).bitwise_xor(1)
    for step in range(step_count):
        image = render_annular_gate(
            state,
            expanded.gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        specific_force = state.specific_force
        if acceleration_control == "constant_1g":
            specific_force = torch.zeros_like(specific_force)
            specific_force[:, 2] = 9.81
        elif acceleration_control == "pair_swapped":
            specific_force = specific_force[pair_swap]
        motor, neural = controller_step_with_paths(
            controller,
            image,
            state.euler[:, :2],
            neural,
            specific_force,
            stick_state.position,
            policy_theta,
            spec,
        )
        if axis_intervention == "reserve_steering" and step >= takeover_step:
            assert teacher_rc_for_mode is not None
            reserve_rc = teacher_rc_for_mode(
                "visual_accelerometer_reserve",
                controller,
                state,
                expanded.gate,
                expanded.mass_scale,
                hover_config,
            )
            reserve_motor = motor_target_for_rc(reserve_rc, hover_config)
            motor = motor.clone()
            motor[:, :3] = reserve_motor[:, :3]
        rc, stick_state = sticks(motor, stick_state)
        for index, (begin, end) in enumerate(bin_limits):
            if begin <= step < end:
                throttle_sum[:, index] += rc[:, 3]
                throttle_count[index] += 1
        previous_position = state.position
        state = quad(rc, state, expanded.mass_scale)
        pass_now, collision_now, miss_now = classify_gate_crossing(
            previous_position, state.position, expanded.gate, gate_config
        )
        new_crossing = (pass_now | collision_now | miss_now) & crossing_radial.isnan()
        new_pass = pass_now & ~passed
        if bool(new_crossing.any()):
            _, crossing_lateral, crossing_vertical = crossing_coordinates(
                previous_position, state.position, expanded.gate
            )
            radial = torch.sqrt(crossing_lateral.square() + crossing_vertical.square())
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
            maximum_tilt, torch.linalg.vector_norm(state.euler[:, :2], dim=1)
        )
        saturation_steps += (stick_state.position.abs() > 0.98).any(dim=1)
    has_post_pass_second = (pass_step >= 0) & (
        pass_step <= step_count - round(1.0 / hover_config.dt)
    )
    saturation_fraction = saturation_steps / step_count
    success = (
        passed
        & cleared
        & has_post_pass_second
        & ~collision
        & ~missed
        & ~recontact
        & (maximum_tilt <= math.radians(40.0))
        & ~(saturation_fraction > 0.25)
    )
    clean_pass = passed & ~collision & ~missed
    hazard = collision | recontact
    reward = (
        10.0 * success.float()
        + 2.0 * clean_pass.float()
        + reward_shaping_weight * (maximum_progress + maximum_approach)
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
        "throttle_bins": (throttle_sum / throttle_count.clamp_min(1.0)).reshape(
            policies, episodes, 4
        ),
    }
    result: dict[str, Any] = {"summaries": summarize(**reshaped)}
    if return_outcomes:
        result["outcomes"] = {
            name: value.detach().cpu()
            for name, value in reshaped.items()
            if name in {"success", "reward", "codes", "throttle_bins"}
        }
    return result


def objective(candidate: dict[str, Any], reference: dict[str, Any], penalty: float) -> float:
    light_delta = candidate["light_mean_reward"] - reference["light_mean_reward"]
    heavy_delta = candidate["heavy_mean_reward"] - reference["heavy_mean_reward"]
    return light_delta - penalty * max(0.0, -heavy_delta)


def compile_theta(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    theta: Tensor,
    spec: PathSpec,
) -> None:
    controller.load_state_dict(base_state)
    with torch.no_grad():
        controller.edge_magnitude[spec.edges] = (spec.baseline + spec.scales * theta).clamp(
            0.0, spec.maximum
        )
    controller.eval()


def standard_evaluation(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    theta: Tensor,
    spec: PathSpec,
    *,
    episodes: int,
    seconds: float,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    **controls: Any,
) -> dict[str, Any]:
    compile_theta(controller, base_state, theta, spec)
    return evaluate_gate(
        controller,
        episodes=episodes,
        seconds=seconds,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        seed=seed,
        balanced_strata=True,
        **controls,
    )


def matched_evaluation(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    theta: Tensor,
    reference_theta: Tensor,
    spec: PathSpec,
    *,
    episodes: int,
    seconds: float,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    acceleration_control: str = "live",
    return_outcomes: bool = False,
) -> dict[str, Any]:
    cases = sample_matched_cases(
        episodes,
        seed=seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    controller.load_state_dict(base_state)
    controller.eval()
    return evaluate_policy_batch(
        controller,
        torch.stack((reference_theta, theta)),
        spec,
        cases,
        seconds=seconds,
        reward_shaping_weight=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        acceleration_control=acceleration_control,
        return_outcomes=return_outcomes,
    )


def archive_rank(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item["development"]
    return item["objective"], metrics["light_success_rate"], metrics["success_rate"]


def validate_archive(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    zero: Tensor,
    spec: PathSpec,
    archive: dict[str, dict[str, Any]],
    *,
    generation: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    ranked = sorted(archive, key=lambda key: archive_rank(archive[key]), reverse=True)
    finalists = ranked[: min(args.top_candidates, len(ranked))]
    vectors = torch.stack([zero] + [archive[key]["vector"].to(device) for key in finalists])
    cases = sample_matched_cases(
        args.validation_episodes,
        seed=args.validation_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    controller.load_state_dict(base_state)
    controller.eval()
    batch = evaluate_policy_batch(
        controller,
        vectors,
        spec,
        cases,
        seconds=args.search_seconds,
        reward_shaping_weight=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    reference = batch["summaries"][0]
    candidates: dict[str, Any] = {}
    for key, metrics in zip(finalists, batch["summaries"][1:], strict=True):
        light_delta = metrics["light_success_rate"] - reference["light_success_rate"]
        heavy_delta = metrics["heavy_success_rate"] - reference["heavy_success_rate"]
        eligible = (
            light_delta >= args.checkpoint_light_improvement
            and heavy_delta >= -args.heavy_noninferiority_margin
        )
        candidates[key] = {
            "source": {name: value for name, value in archive[key].items() if name != "vector"},
            "metrics": metrics,
            "light_success_delta": light_delta,
            "heavy_success_delta": heavy_delta,
            "continuation_eligible": eligible,
        }
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "generation": generation,
                    "candidate": key,
                    "light_success_delta": light_delta,
                    "heavy_success_delta": heavy_delta,
                    "continuation_eligible": eligible,
                }
            ),
            flush=True,
        )
    return {
        "generation": generation,
        "seed": args.validation_seed,
        "reference": reference,
        "candidates": candidates,
    }


def zero_edge_sensitivity_audit(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    zero: Tensor,
    spec: PathSpec,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    activated = zero.clone()
    activated[spec.zero_edges] = args.sigma
    batch = matched_evaluation(
        controller,
        base_state,
        activated,
        zero,
        spec,
        episodes=min(32, args.development_episodes),
        seconds=min(2.0, args.search_seconds),
        seed=args.development_seed - 1,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        return_outcomes=True,
    )
    baseline_actions = batch["outcomes"]["throttle_bins"][0]
    activated_actions = batch["outcomes"]["throttle_bins"][1]
    maximum_action_difference = float((activated_actions - baseline_actions).abs().max())
    return {
        "zero_edge_count": int(spec.zero_edges.sum()),
        "activated_magnitude": spec.floor * args.sigma,
        "maximum_early_throttle_stick_difference": maximum_action_difference,
        "measurable_above_float_noise": maximum_action_difference > 1.0e-7,
    }


def paired_clustered_confidence_interval(
    baseline: Tensor, candidate: Tensor, *, cluster_size: int = 2
) -> dict[str, Any]:
    """Paired success interval with matched nuisance cases as the sampling units."""

    if len(baseline) != len(candidate) or len(baseline) % cluster_size:
        raise ValueError("paired outcomes must contain complete, equally sized clusters")
    episode_difference = candidate.float() - baseline.float()
    cluster_difference = episode_difference.reshape(-1, cluster_size).mean(dim=1)
    mean = float(cluster_difference.mean())
    standard_error = float(
        cluster_difference.std(unbiased=True) / math.sqrt(len(cluster_difference))
    )
    return {
        "method": "paired normal approximation clustered by matched geometry",
        "mean_success_difference": mean,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "baseline_only_successes": int(((baseline == 1) & (candidate == 0)).sum()),
        "candidate_only_successes": int(((baseline == 0) & (candidate == 1)).sum()),
        "paired_episodes": len(episode_difference),
        "independent_geometry_clusters": len(cluster_difference),
        "episodes_per_cluster": cluster_size,
    }


@torch.no_grad()
def controller_step_parity_audit(
    controller: ConnectomeController,
    zero: Tensor,
    spec: PathSpec,
    *,
    resolution: int,
    device: torch.device,
) -> dict[str, Any]:
    """Check that a zero path vector is the unchanged promoted controller."""

    generator = torch.Generator(device=device).manual_seed(700_003)
    batch = 4
    image = torch.rand(batch, resolution, resolution, generator=generator, device=device)
    attitude = 0.1 * torch.randn(batch, 2, generator=generator, device=device)
    neural = 0.2 * torch.randn(batch, controller.n_nodes, generator=generator, device=device)
    specific_force = torch.randn(batch, 3, generator=generator, device=device)
    specific_force[:, 2] += 9.81
    stick_position = torch.rand(batch, 4, generator=generator, device=device) * 2.0 - 1.0
    expected_motor, expected_state = controller(
        image, attitude, neural, specific_force, stick_position
    )
    actual_motor, actual_state = controller_step_with_paths(
        controller,
        image,
        attitude,
        neural,
        specific_force,
        stick_position,
        zero.expand(batch, -1),
        spec,
    )
    motor_difference = float((actual_motor - expected_motor).abs().max())
    state_difference = float((actual_state - expected_state).abs().max())
    return {
        "maximum_motor_absolute_difference": motor_difference,
        "maximum_state_absolute_difference": state_difference,
        "passed": motor_difference <= 1.0e-6 and state_difference <= 1.0e-6,
    }


def save_checkpoint(
    path: Path,
    source: dict[str, Any],
    source_checkpoint_sha256: str,
    controller: ConnectomeController,
    theta: Tensor,
    spec: PathSpec,
    promotion: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source)
    checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    checkpoint["source_checkpoint_sha256"] = source_checkpoint_sha256
    checkpoint["acceleration_path_es"] = {
        "method": "matched-mass ranked antithetic ES on complete flights",
        "maximum_hops": 4,
        "selected_edge_indices": spec.edges.detach().cpu().tolist(),
        "normalized_parameter_vector": theta.detach().cpu().tolist(),
        "normalized_parameter_vector_sha256": vector_sha256(theta),
        "compiled_into_native_edge_magnitudes": True,
        "promotion": promotion,
    }
    torch.save(checkpoint, path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    controller, checkpoint, hover_config, gate_config, resolution = load_controller(
        args.graph, args.checkpoint, device
    )
    base_state = {name: value.detach().clone() for name, value in controller.state_dict().items()}
    spec = make_path_spec(
        controller,
        args.graph,
        maximum_hops=args.maximum_hops,
        floor_quantile=args.scale_floor_quantile,
        maximum_magnitude=args.maximum_edge_magnitude,
    )
    if len(spec.edges) != 282 and args.maximum_hops == 4:
        raise SystemExit(f"expected 282 four-hop path edges, found {len(spec.edges)}")
    zero = torch.zeros(len(spec.edges), device=device)
    step_parity = controller_step_parity_audit(
        controller,
        zero,
        spec,
        resolution=resolution,
        device=device,
    )
    if not step_parity["passed"]:
        raise SystemExit(f"zero-vector controller parity failed: {step_parity}")
    zero_sensitivity = zero_edge_sensitivity_audit(
        controller,
        base_state,
        zero,
        spec,
        args=args,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    if int(spec.zero_edges.sum()) and not zero_sensitivity["measurable_above_float_noise"]:
        raise SystemExit("zero-edge perturbations do not measurably affect early actions")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rng = np.random.default_rng(args.search_seed)
    center = zero.clone()
    archive: dict[str, dict[str, Any]] = {}
    generations: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    early_stop = False
    for generation in range(1, args.generations + 1):
        evaluated_center = center.clone()
        sigma = args.sigma * args.sigma_decay ** (generation - 1)
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_directions, len(spec.edges))).astype(np.float32)
        ).to(device)
        plus = (center + sigma * epsilon).clamp(spec.lower, spec.upper)
        minus = (center - sigma * epsilon).clamp(spec.lower, spec.upper)
        candidates = torch.stack((plus, minus), dim=1).reshape(-1, len(spec.edges))
        policies = torch.cat((zero[None], evaluated_center[None], candidates), dim=0)
        cases = sample_matched_cases(
            args.development_episodes,
            seed=args.development_seed + generation - 1,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        weight = shaping_weight(generation, args.generations)
        controller.load_state_dict(base_state)
        controller.eval()
        batch = evaluate_policy_batch(
            controller,
            policies,
            spec,
            cases,
            seconds=args.search_seconds,
            reward_shaping_weight=weight,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        reference, center_summary = batch["summaries"][:2]
        candidate_summaries = batch["summaries"][2:]
        objectives = np.asarray(
            [objective(item, reference, args.heavy_reward_penalty) for item in candidate_summaries],
            dtype=np.float32,
        )
        ranks = np.empty(len(objectives), dtype=np.float32)
        ranks[np.argsort(objectives)] = np.linspace(-0.5, 0.5, len(objectives))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        proposed_update = args.learning_rate * gradient
        proposed_rms = proposed_update.square().mean().sqrt()
        maximum_rms = args.maximum_update_sigma_fraction * sigma
        update_scale = min(1.0, maximum_rms / max(float(proposed_rms), 1.0e-12))
        update = proposed_update * update_scale
        center = (center + update).clamp(spec.lower, spec.upper)
        for index, vector in enumerate(candidates):
            key = vector_sha256(vector)
            archive[key] = {
                "vector": vector.detach().cpu().clone(),
                "generation": generation,
                "kind": f"direction_{index // 2 + 1}_{'plus' if index % 2 == 0 else 'minus'}",
                "objective": float(objectives[index]),
                "development": candidate_summaries[index],
            }
        center_key = vector_sha256(evaluated_center)
        center_objective = objective(center_summary, reference, args.heavy_reward_penalty)
        archive[center_key] = {
            "vector": evaluated_center.detach().cpu().clone(),
            "generation": generation,
            "kind": "evaluated_center",
            "objective": center_objective,
            "development": center_summary,
        }
        best_index = int(np.argmax(objectives))
        entry = {
            "generation": generation,
            "development_seed": args.development_seed + generation - 1,
            "sigma": sigma,
            "shaping_weight": weight,
            "reference": reference,
            "center": center_summary,
            "center_objective": center_objective,
            "best_candidate": candidate_summaries[best_index],
            "best_candidate_objective": float(objectives[best_index]),
            "proposed_normalized_update_rms": float(proposed_rms),
            "applied_normalized_update_rms": float(update.square().mean().sqrt()),
            "update_cap_scale": update_scale,
            "next_center_sha256": vector_sha256(center),
        }
        generations.append(entry)
        print(
            json.dumps(
                {
                    "phase": "search",
                    "generation": generation,
                    "reference_light_success": reference["light_success_rate"],
                    "center_light_success": center_summary["light_success_rate"],
                    "best_light_success": candidate_summaries[best_index]["light_success_rate"],
                    "best_objective": float(objectives[best_index]),
                    "update_rms": entry["applied_normalized_update_rms"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "center_sha256": vector_sha256(center),
                "selected_edge_indices": spec.edges.detach().cpu(),
            },
            args.output_dir / "search-state.pt",
        )
        torch.save(
            {
                key: {
                    "vector": item["vector"],
                    "generation": item["generation"],
                    "kind": item["kind"],
                }
                for key, item in archive.items()
            },
            args.output_dir / "archive-vectors.pt",
        )
        (args.output_dir / "progress.json").write_text(
            json.dumps(
                {
                    "experiment": "acceleration-path-es-v1",
                    "generation": generation,
                    "parameter_count": len(spec.edges),
                    "archive_size": len(archive),
                    "generations": generations,
                    "elapsed_seconds": perf_counter() - started,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if generation % args.validation_interval == 0:
            validation = validate_archive(
                controller,
                base_state,
                zero,
                spec,
                archive,
                generation=generation,
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            validations.append(validation)
            if generation == args.checkpoint_generation and not any(
                item["continuation_eligible"] for item in validation["candidates"].values()
            ):
                early_stop = True
                print(json.dumps({"phase": "early_stop", "generation": generation}), flush=True)
                break
    if not validations or validations[-1]["generation"] != generations[-1]["generation"]:
        validations.append(
            validate_archive(
                controller,
                base_state,
                zero,
                spec,
                archive,
                generation=generations[-1]["generation"],
                args=args,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
        )
    validated: dict[str, dict[str, Any]] = {}
    for validation in validations:
        validated.update(validation["candidates"])
    eligible = [key for key, item in validated.items() if item["continuation_eligible"]]
    selection_pool = eligible or list(validated)
    winner_key = max(
        selection_pool,
        key=lambda key: (
            validated[key]["metrics"]["light_success_rate"],
            validated[key]["metrics"]["heavy_success_rate"],
            validated[key]["metrics"]["success_rate"],
        ),
    )
    winner = archive[winner_key]["vector"].to(device)
    standard_final: dict[str, Any] = {}
    for name, vector, controls in (
        ("reference", zero, {}),
        ("candidate", winner, {}),
        ("candidate_frozen_first_frame", winner, {"frozen_visual": True}),
        ("candidate_constant_1g", winner, {"frozen_acceleration": True}),
    ):
        standard_final[name] = standard_evaluation(
            controller,
            base_state,
            vector,
            spec,
            episodes=args.final_episodes,
            seconds=args.final_seconds,
            seed=args.final_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            **controls,
        )
        print(
            json.dumps(
                {"phase": "final_standard", "name": name, **standard_compact(standard_final[name])}
            ),
            flush=True,
        )
    paired_live = matched_evaluation(
        controller,
        base_state,
        winner,
        zero,
        spec,
        episodes=args.final_episodes,
        seconds=args.final_seconds,
        seed=args.final_seed + 1,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        return_outcomes=True,
    )
    paired_controls = {}
    for control in ("constant_1g", "pair_swapped"):
        paired_controls[control] = matched_evaluation(
            controller,
            base_state,
            winner,
            zero,
            spec,
            episodes=args.final_episodes,
            seconds=args.final_seconds,
            seed=args.final_seed + 1,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
            acceleration_control=control,
            return_outcomes=True,
        )
    light = ~paired_live["outcomes"]["codes"][0].bitwise_and(1).bool()
    live_reference_success = paired_live["outcomes"]["success"][0]
    live_candidate_success = paired_live["outcomes"]["success"][1]
    light_paired = paired_confidence_interval(
        live_reference_success[light], live_candidate_success[light]
    )
    overall_paired = paired_clustered_confidence_interval(
        live_reference_success, live_candidate_success
    )
    matched_reference, matched_candidate = paired_live["summaries"]
    acceleration_control_audit: dict[str, Any] = {}
    for control, result in paired_controls.items():
        control_candidate_success = result["outcomes"]["success"][1]
        live_advantage = paired_clustered_confidence_interval(
            control_candidate_success, live_candidate_success
        )
        action_difference = (
            paired_live["outcomes"]["throttle_bins"][1] - result["outcomes"]["throttle_bins"][1]
        ).abs()
        acceleration_control_audit[control] = {
            "summaries": result["summaries"],
            "live_success_advantage": live_advantage,
            "mean_absolute_early_throttle_stick_difference": float(action_difference.mean()),
            "maximum_absolute_early_throttle_stick_difference": float(action_difference.max()),
        }
    promotion_checks = {
        "light_improvement_at_least_threshold": (
            matched_candidate["light_success_rate"]
            >= matched_reference["light_success_rate"] + args.final_light_improvement
        ),
        "light_paired_confidence_interval_excludes_zero": light_paired["confidence_95"][0] > 0.0,
        "heavy_nondegradation_within_margin": (
            matched_candidate["heavy_success_rate"]
            >= matched_reference["heavy_success_rate"] - args.heavy_noninferiority_margin
        ),
        "matched_overall_success_improves": (
            matched_candidate["success_rate"] > matched_reference["success_rate"]
        ),
        "standard_overall_success_improves": (
            standard_final["candidate"]["success_rate"]
            > standard_final["reference"]["success_rate"]
        ),
    }
    promotion = {"passed": all(promotion_checks.values()), "checks": promotion_checks}
    candidate_path = args.output_dir / "candidate.pt"
    if promotion["passed"]:
        compile_theta(controller, base_state, winner, spec)
        save_checkpoint(
            candidate_path,
            checkpoint,
            file_sha256(args.checkpoint),
            controller,
            winner,
            spec,
            promotion,
        )
    selected_magnitudes = (spec.baseline + spec.scales * winner).clamp(0.0, 8.0)
    changed = (selected_magnitudes - spec.baseline).abs() > 1.0e-8
    goal_passed = bool(
        promotion["passed"]
        and standard_final["candidate"]["goal_pass"]
        and standard_final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
    )
    report = {
        "method": "matched-mass ranked mirrored ES on complete flights",
        "claim_scope": (
            "Only existing signed edge magnitudes on anatomical acceleration-to-throttle paths "
            "are changed. Deployment adds no mass input, history feature, scaler, estimator, "
            "clock, planner, or actor state beyond the persistent connectome state."
        ),
        "graph": stable_path(args.graph),
        "graph_sha256": file_sha256(args.graph),
        "source_checkpoint": stable_path(args.checkpoint),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "protocol": {
            **{key: value for key, value in vars(args).items() if not isinstance(value, Path)},
            "fixed_topology": True,
            "fixed_transmitter_signs": True,
            "biases_frozen": True,
            "time_constants_frozen": True,
            "nonselected_edges_frozen": True,
            "full_flight_backpropagation": False,
            "matched_light_heavy_geometry": True,
            "common_random_numbers_within_generation": True,
            "episode_reward": "10*S + 2*C + shaping_weight*(P+Q) - 2*H - 0.1*U",
            "search_objective": "delta_light_mean_reward - 2*max(0,-delta_heavy_mean_reward)",
        },
        "parameterization": {
            "count": len(spec.edges),
            "maximum_hops": args.maximum_hops,
            "scale_floor": spec.floor,
            "scale_floor_quantile": args.scale_floor_quantile,
            "zero_edge_count": int(spec.zero_edges.sum()),
            "zero_vector_controller_step_parity": step_parity,
            "zero_edge_sensitivity_audit": zero_sensitivity,
            "selected_edge_indices": spec.edges.detach().cpu().tolist(),
        },
        "generations_completed": len(generations),
        "early_stopped_at_checkpoint": early_stop,
        "generations": generations,
        "validations": validations,
        "eligible_validated_candidates": eligible,
        "selected_candidate": winner_key,
        "selected_normalized_vector": winner.tolist(),
        "selected_normalized_vector_sha256": vector_sha256(winner),
        "compiled_edges_changed": int(changed.sum()),
        "maximum_edge_magnitude_change": float((selected_magnitudes - spec.baseline).abs().max()),
        "final": {
            "standard": standard_final,
            "matched_live": paired_live["summaries"],
            "matched_controls": acceleration_control_audit,
            "paired_overall_success_difference": overall_paired,
            "paired_light_success_difference": light_paired,
        },
        "promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": file_sha256(candidate_path) if promotion["passed"] else None,
        "acceleration_dependence_demonstrated": (
            acceleration_control_audit["constant_1g"]["live_success_advantage"]["confidence_95"][0]
            > 0.0
            and acceleration_control_audit["pair_swapped"]["live_success_advantage"][
                "confidence_95"
            ][0]
            > 0.0
        ),
        "goal_passed": goal_passed,
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
                "generations_completed": len(generations),
                "promotion_passed": promotion["passed"],
                "goal_passed": goal_passed,
            }
        ),
        flush=True,
    )
    return 0 if goal_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
