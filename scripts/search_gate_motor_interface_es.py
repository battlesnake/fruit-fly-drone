#!/usr/bin/env python3
"""Search a 24-parameter native motor interface with complete-flight mirrored ES."""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from train_gate import (  # noqa: E402
    RETINAL_FLIP_X,
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
)

POOL_LABELS = (
    "roll_positive",
    "roll_negative",
    "pitch_positive",
    "pitch_negative",
    "yaw_positive",
    "yaw_negative",
    "throttle_positive",
    "throttle_negative",
)


@dataclass(frozen=True)
class BalancedCases:
    state: QuadState
    gate: AnnularGate
    mass_scale: Tensor
    stratum_code: Tensor


@dataclass(frozen=True)
class MotorInterfaceSpec:
    labels: tuple[str, ...]
    kinds: tuple[str, ...]
    scales: Tensor
    lower: Tensor
    upper: Tensor
    bias_nodes: Tensor
    bias_parameter: Tensor
    gain_edges: Tensor
    gain_parameter: Tensor
    description: tuple[dict[str, Any], ...]


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
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "motor-interface-es-v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--checkpoint-generation", type=int, default=20)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--antithetic-directions", type=int, default=16)
    parser.add_argument("--development-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=256)
    parser.add_argument("--final-episodes", type=int, default=1024)
    parser.add_argument("--search-seconds", type=float, default=8.0)
    parser.add_argument("--final-seconds", type=float, default=12.0)
    parser.add_argument("--top-candidates", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sigma-decay", type=float, default=0.985)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--bias-perturbation-scale", type=float, default=0.01)
    parser.add_argument("--log-gain-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--maximum-bias-delta", type=float, default=0.15)
    parser.add_argument("--maximum-gain-ratio", type=float, default=3.0)
    parser.add_argument("--minimum-validation-improvement", type=float, default=0.05)
    parser.add_argument("--maximum-mass-stratum-drop", type=float, default=0.05)
    parser.add_argument("--training-success-signal", type=float, default=0.03)
    parser.add_argument("--training-fitness-signal", type=float, default=0.10)
    parser.add_argument("--development-seed", type=int, default=600_031)
    parser.add_argument("--validation-seed", type=int, default=610_031)
    parser.add_argument("--final-seed", type=int, default=620_031)
    parser.add_argument("--search-seed", type=int, default=63_031)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
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
        args.bias_perturbation_scale,
        args.log_gain_perturbation_scale,
        args.maximum_bias_delta,
        args.maximum_gain_ratio,
        args.minimum_validation_improvement,
        args.maximum_mass_stratum_drop,
        args.training_success_signal,
        args.training_fitness_signal,
    )
    if min(positive) <= 0.0:
        raise SystemExit("search sizes, scales, and thresholds must be positive")
    if args.generations < args.checkpoint_generation:
        raise SystemExit("--generations must reach --checkpoint-generation")
    if args.checkpoint_generation % args.validation_interval:
        raise SystemExit("checkpoint generation must coincide with validation")
    if not 0.0 < args.sigma_decay <= 1.0:
        raise SystemExit("--sigma-decay must be in (0, 1]")
    if args.maximum_gain_ratio <= 1.0:
        raise SystemExit("--maximum-gain-ratio must exceed one")
    for name in ("development_episodes", "validation_episodes", "final_episodes"):
        if getattr(args, name) % 8:
            raise SystemExit(f"--{name.replace('_', '-')} must be divisible by eight")


def stable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def vector_sha256(values: Tensor) -> str:
    array = values.detach().cpu().numpy().astype("<f4", copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def clone_state(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def load_controller(
    graph: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ConnectomeController, dict[str, Any], HoverConfig, GateConfig, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("graph_sha256") != file_sha256(graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
        raise SystemExit("checkpoint retinal orientation does not match the evaluator")
    hover_config = HoverConfig(**checkpoint["hover_config"])
    gate_config = GateConfig(**checkpoint["gate_config"])
    resolution = int(checkpoint["image_resolution"])
    controller = ConnectomeController(
        graph,
        neural_dt=hover_config.dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    incompatible = controller.load_state_dict(checkpoint["controller"], strict=False)
    allowed_missing = {"initial_raw_time_constant"}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise SystemExit(
            "checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    controller.eval()
    return controller, checkpoint, hover_config, gate_config, resolution


def motor_interface_spec(
    controller: ConnectomeController,
    *,
    bias_scale: float,
    log_gain_scale: float,
    maximum_bias_delta: float,
    maximum_gain_ratio: float,
) -> MotorInterfaceSpec:
    labels: list[str] = []
    kinds: list[str] = []
    scales: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    bias_nodes: list[int] = []
    bias_parameter: list[int] = []
    gain_edges: list[int] = []
    gain_parameter: list[int] = []
    description: list[dict[str, Any]] = []
    log_bound = math.log(maximum_gain_ratio)
    for pool, pool_label in enumerate(POOL_LABELS):
        begin = int(controller.pool_offsets[pool].item())
        end = int(controller.pool_offsets[pool + 1].item())
        nodes = controller.pool_indices[begin:end]
        parameter = len(labels)
        labels.append(f"{pool_label}.intrinsic_bias")
        kinds.append("intrinsic_bias_delta")
        scales.append(bias_scale)
        lower.append(-maximum_bias_delta)
        upper.append(maximum_bias_delta)
        bias_nodes.extend(int(node) for node in nodes.tolist())
        bias_parameter.extend([parameter] * len(nodes))
        description.append(
            {
                "index": parameter,
                "label": labels[-1],
                "kind": kinds[-1],
                "pool_neurons": len(nodes),
                "affected_edges": 0,
            }
        )
        incoming = torch.isin(controller.edge_post, nodes)
        for sign, sign_label in ((1.0, "excitatory"), (-1.0, "inhibitory")):
            edges = torch.nonzero(
                incoming & (controller.edge_sign == sign), as_tuple=False
            ).flatten()
            if not len(edges):
                continue
            parameter = len(labels)
            labels.append(f"{pool_label}.{sign_label}_incoming_log_gain")
            kinds.append("incoming_log_gain")
            scales.append(log_gain_scale)
            lower.append(-log_bound)
            upper.append(log_bound)
            gain_edges.extend(int(edge) for edge in edges.tolist())
            gain_parameter.extend([parameter] * len(edges))
            description.append(
                {
                    "index": parameter,
                    "label": labels[-1],
                    "kind": kinds[-1],
                    "pool_neurons": len(nodes),
                    "affected_edges": len(edges),
                }
            )
    if len(labels) > 24:
        raise RuntimeError(
            f"motor-interface parameterization unexpectedly has {len(labels)} values"
        )
    device = controller.bias.device
    return MotorInterfaceSpec(
        labels=tuple(labels),
        kinds=tuple(kinds),
        scales=torch.tensor(scales, device=device),
        lower=torch.tensor(lower, device=device),
        upper=torch.tensor(upper, device=device),
        bias_nodes=torch.tensor(bias_nodes, device=device, dtype=torch.long),
        bias_parameter=torch.tensor(bias_parameter, device=device, dtype=torch.long),
        gain_edges=torch.tensor(gain_edges, device=device, dtype=torch.long),
        gain_parameter=torch.tensor(gain_parameter, device=device, dtype=torch.long),
        description=tuple(description),
    )


def sample_balanced_cases(
    episodes: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> BalancedCases:
    seed_everything(seed)
    state, gate, mass_scale = initial_rollout(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        strict=True,
    )
    code = torch.arange(episodes, device=device) % 8
    mass_sign = torch.where(code.bitwise_and(1).bool(), 1.0, -1.0)
    lateral_sign = torch.where(code.bitwise_and(2).bool(), 1.0, -1.0)
    obliquity_sign = torch.where(code.bitwise_and(4).bool(), 1.0, -1.0)
    center = gate.center.clone()
    center[:, 1] = center[:, 1].abs() * lateral_sign
    bearing = torch.atan2(center[:, 1], center[:, 0])
    sampled_bearing = torch.atan2(gate.center[:, 1], gate.center[:, 0])
    obliquity_magnitude = wrap_angle(gate.yaw - sampled_bearing).abs()
    balanced_gate = AnnularGate(
        center=center,
        yaw=bearing + obliquity_sign * obliquity_magnitude,
    )
    balanced_mass = 1.0 + mass_sign * (mass_scale - 1.0).abs()
    return BalancedCases(state, balanced_gate, balanced_mass, code)


def repeat_cases(cases: BalancedCases, policies: int) -> BalancedCases:
    return BalancedCases(
        state=QuadState(*(value.repeat((policies, 1)) for value in cases.state.as_tuple())),
        gate=AnnularGate(
            center=cases.gate.center.repeat((policies, 1)),
            yaw=cases.gate.yaw.repeat(policies),
        ),
        mass_scale=cases.mass_scale.repeat(policies),
        stratum_code=cases.stratum_code.repeat(policies),
    )


def controller_step_with_interface(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    state: Tensor,
    body_specific_force: Tensor,
    stick_position: Tensor,
    interface: Tensor,
    spec: MotorInterfaceSpec,
) -> tuple[Tensor, Tensor]:
    activity = torch.tanh(state)
    messages = activity[:, controller.edge_pre] * controller.edge_sign * controller.edge_magnitude
    selected_gain = interface[:, spec.gain_parameter]
    selected_magnitude = (
        controller.edge_magnitude[spec.gain_edges] * torch.exp(selected_gain)
    ).clamp(max=8.0)
    messages[:, spec.gain_edges] = (
        activity[:, controller.edge_pre[spec.gain_edges]]
        * controller.edge_sign[spec.gain_edges]
        * selected_magnitude
    )
    recurrent = torch.zeros_like(state).index_add(1, controller.edge_post, messages)
    drive = (
        recurrent
        + controller.bias
        + controller.sensory_drive(
            image,
            roll_pitch,
            body_specific_force,
            stick_position,
        )
    )
    drive[:, spec.bias_nodes] += interface[:, spec.bias_parameter]
    target = 5.0 * torch.tanh(drive / 5.0)
    alpha = 1.0 - torch.exp(-controller.neural_dt / controller.time_constant)
    next_state = state + alpha * (target - state)
    return controller.motor_drive(next_state), next_state


def _mean(values: Tensor, mask: Tensor) -> float | None:
    return float(values[mask].mean()) if bool(mask.any()) else None


def summarize_policy_batch(
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
) -> list[dict[str, Any]]:
    policy_count, episodes = success.shape
    summaries = []
    for policy in range(policy_count):
        policy_codes = codes[policy]
        negative_side = ~policy_codes.bitwise_and(2).bool()
        lower_mass = ~policy_codes.bitwise_and(1).bool()
        negative_obliquity = ~policy_codes.bitwise_and(4).bool()
        crossed = ~crossing_radial[policy].isnan()
        stratum_reward = [float(reward[policy][policy_codes == code].mean()) for code in range(8)]
        stratum_success = [
            float(success[policy][policy_codes == code].float().mean()) for code in range(8)
        ]
        mean_reward = float(reward[policy].mean())
        worst_reward = min(stratum_reward)
        summaries.append(
            {
                "fitness": 0.75 * mean_reward + 0.25 * worst_reward,
                "mean_reward": mean_reward,
                "worst_stratum_reward": worst_reward,
                "success_rate": float(success[policy].float().mean()),
                "worst_stratum_success_rate": min(stratum_success),
                "clean_pass_rate": float(passed[policy].float().mean()),
                "ring_collision_rate": float(collision[policy].float().mean()),
                "miss_rate": float(missed[policy].float().mean()),
                "plane_crossing_rate": float(crossed.float().mean()),
                "progress_mean": float(progress[policy].mean()),
                "approach_score_mean": float(approach[policy].mean()),
                "hazard_rate": float(hazard[policy].float().mean()),
                "stick_saturation_fraction": float(saturation_fraction[policy].mean()),
                "success_by_stratum": {
                    "lower_mass": _mean(success[policy].float(), lower_mass),
                    "higher_mass": _mean(success[policy].float(), ~lower_mass),
                    "negative_lateral_offset": _mean(success[policy].float(), negative_side),
                    "positive_lateral_offset": _mean(success[policy].float(), ~negative_side),
                    "negative_obliquity": _mean(success[policy].float(), negative_obliquity),
                    "positive_obliquity": _mean(success[policy].float(), ~negative_obliquity),
                },
                "crossing_radial_mean_m": _mean(crossing_radial[policy], crossed),
                "crossing_radial_mean_by_lateral_side_m": {
                    "negative": _mean(crossing_radial[policy], crossed & negative_side),
                    "positive": _mean(crossing_radial[policy], crossed & ~negative_side),
                },
            }
        )
    return summaries


@torch.no_grad()
def evaluate_policy_batch(
    controller: ConnectomeController,
    vectors: Tensor,
    spec: MotorInterfaceSpec,
    cases: BalancedCases,
    *,
    seconds: float,
    shaping_weight: float,
    resolution: int,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    return_outcomes: bool = False,
) -> dict[str, Any]:
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
        motor, neural = controller_step_with_interface(
            controller,
            image,
            state.euler[:, :2],
            neural,
            state.specific_force,
            stick_state.position,
            interface,
            spec,
        )
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
        + shaping_weight * (maximum_progress + maximum_approach)
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
    result: dict[str, Any] = {
        "summaries": summarize_policy_batch(**reshaped),
    }
    if return_outcomes:
        result["outcomes"] = {
            name: value.detach().cpu()
            for name, value in reshaped.items()
            if name in {"success", "reward", "codes"}
        }
    return result


def compile_vector(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    vector: Tensor,
    spec: MotorInterfaceSpec,
) -> None:
    controller.load_state_dict(base_state)
    with torch.no_grad():
        controller.bias[spec.bias_nodes] += vector[spec.bias_parameter]
        controller.edge_magnitude[spec.gain_edges] = (
            controller.edge_magnitude[spec.gain_edges] * torch.exp(vector[spec.gain_parameter])
        ).clamp(max=8.0)
    controller.eval()


def standard_evaluation(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    vector: Tensor,
    spec: MotorInterfaceSpec,
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
    compile_vector(controller, base_state, vector, spec)
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


def standard_compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate": metrics["success_rate"],
        "pass_rate": metrics["pass_rate"],
        "plane_crossing_rate": metrics["plane_crossing_rate"],
        "ring_collision_rate": metrics["ring_collision_rate"],
        "miss_rate": metrics["miss_rate"],
        "crossing_radial_mean_m": metrics["crossing_radial_mean_m"],
        "crossing_radial_mean_by_lateral_side_m": metrics["crossing_radial_mean_by_lateral_side_m"],
        "success_by_stratum": metrics["success_by_stratum"],
    }


def parity_audit(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    zero: Tensor,
    spec: MotorInterfaceSpec,
    *,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    controller.load_state_dict(base_state)
    controller.eval()
    episodes = min(args.development_episodes, 32)
    cases = sample_balanced_cases(
        episodes,
        seed=args.development_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    batched = evaluate_policy_batch(
        controller,
        zero[None],
        spec,
        cases,
        seconds=args.search_seconds,
        shaping_weight=1.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
    )["summaries"][0]
    standard = standard_evaluation(
        controller,
        base_state,
        zero,
        spec,
        episodes=episodes,
        seconds=args.search_seconds,
        seed=args.development_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )

    def optional_difference(left: float | None, right: float | None) -> float:
        if left is None and right is None:
            return 0.0
        if left is None or right is None:
            return float("inf")
        return abs(left - right)

    differences = {
        "success_rate": abs(batched["success_rate"] - standard["success_rate"]),
        "clean_pass_rate": abs(batched["clean_pass_rate"] - standard["pass_rate"]),
        "plane_crossing_rate": abs(
            batched["plane_crossing_rate"] - standard["plane_crossing_rate"]
        ),
        "crossing_radial_mean_m": optional_difference(
            batched["crossing_radial_mean_m"], standard["crossing_radial_mean_m"]
        ),
    }
    tolerances = {
        "success_rate": 0.0,
        "clean_pass_rate": 0.0,
        "plane_crossing_rate": 0.0,
        "crossing_radial_mean_m": 1.0e-4,
    }
    return {
        "passed": all(differences[name] <= tolerances[name] for name in differences),
        "maximum_allowed_difference": tolerances,
        "differences": differences,
        "batched": batched,
        "standard": standard_compact(standard),
    }


def shaping_weight(generation: int, total_generations: int) -> float:
    halfway = total_generations / 2.0
    if generation <= halfway:
        return 1.0
    fraction = (generation - halfway) / max(total_generations - halfway, 1.0)
    return 1.0 - 0.8 * fraction


def archive_rank(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item["development"]
    return (
        metrics["fitness"],
        metrics["worst_stratum_success_rate"],
        metrics["success_rate"],
    )


def validation_rank(item: dict[str, Any]) -> tuple[float, float, float]:
    metrics = item["metrics"]
    strata = metrics["success_by_stratum"]
    worst = min(
        strata["lower_mass"],
        strata["higher_mass"],
        strata["negative_lateral_offset"],
        strata["positive_lateral_offset"],
    )
    radial = metrics["crossing_radial_mean_m"]
    if radial is None:
        radial = float("inf")
    return metrics["success_rate"], worst, -radial


def validation_mass_balance_eligible(
    item: dict[str, Any],
    baseline: dict[str, Any],
    *,
    maximum_mass_drop: float,
) -> bool:
    candidate = item["metrics"]
    return bool(
        candidate["success_rate"] > baseline["success_rate"]
        and candidate["success_by_stratum"]["lower_mass"]
        >= baseline["success_by_stratum"]["lower_mass"] - maximum_mass_drop
        and candidate["success_by_stratum"]["higher_mass"]
        >= baseline["success_by_stratum"]["higher_mass"] - maximum_mass_drop
    )


def validate_archive(
    controller: ConnectomeController,
    base_state: dict[str, Tensor],
    zero: Tensor,
    spec: MotorInterfaceSpec,
    archive: dict[str, dict[str, Any]],
    *,
    generation: int,
    args: argparse.Namespace,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> dict[str, Any]:
    baseline = standard_evaluation(
        controller,
        base_state,
        zero,
        spec,
        episodes=args.validation_episodes,
        seconds=args.search_seconds,
        seed=args.validation_seed,
        resolution=resolution,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    ranked = sorted(archive, key=lambda key: archive_rank(archive[key]), reverse=True)
    finalists = ranked[: min(args.top_candidates, len(ranked))]
    candidates: dict[str, Any] = {}
    for key in finalists:
        metrics = standard_evaluation(
            controller,
            base_state,
            archive[key]["vector"].to(device),
            spec,
            episodes=args.validation_episodes,
            seconds=args.search_seconds,
            seed=args.validation_seed,
            resolution=resolution,
            device=device,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        candidates[key] = {
            "source": {name: value for name, value in archive[key].items() if name != "vector"},
            "metrics": metrics,
            "success_delta_from_baseline": metrics["success_rate"] - baseline["success_rate"],
        }
        print(
            json.dumps(
                {
                    "phase": "validation",
                    "generation": generation,
                    "candidate": key,
                    "success_rate": metrics["success_rate"],
                    "success_delta_from_baseline": candidates[key]["success_delta_from_baseline"],
                }
            ),
            flush=True,
        )
    return {
        "generation": generation,
        "seed": args.validation_seed,
        "baseline": baseline,
        "candidates": candidates,
    }


def paired_confidence_interval(baseline: Tensor, candidate: Tensor) -> dict[str, Any]:
    difference = candidate.float() - baseline.float()
    mean = float(difference.mean())
    standard_error = float(difference.std(unbiased=True) / math.sqrt(len(difference)))
    return {
        "method": "paired normal approximation",
        "mean_success_difference": mean,
        "standard_error": standard_error,
        "confidence_95": [mean - 1.96 * standard_error, mean + 1.96 * standard_error],
        "baseline_only_successes": int(((baseline == 1) & (candidate == 0)).sum()),
        "candidate_only_successes": int(((baseline == 0) & (candidate == 1)).sum()),
        "paired_episodes": len(difference),
    }


def final_promotion(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    paired: dict[str, Any],
    *,
    minimum_improvement: float,
    maximum_mass_drop: float,
) -> dict[str, Any]:
    checks = {
        "overall_improvement_at_least_threshold": (
            candidate["success_rate"] >= baseline["success_rate"] + minimum_improvement
        ),
        "paired_confidence_interval_excludes_zero": paired["confidence_95"][0] > 0.0,
        "lower_mass_drop_within_limit": (
            candidate["success_by_stratum"]["lower_mass"]
            >= baseline["success_by_stratum"]["lower_mass"] - maximum_mass_drop
        ),
        "higher_mass_drop_within_limit": (
            candidate["success_by_stratum"]["higher_mass"]
            >= baseline["success_by_stratum"]["higher_mass"] - maximum_mass_drop
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def save_checkpoint(
    path: Path,
    source: dict[str, Any],
    source_checkpoint_sha256: str,
    controller: ConnectomeController,
    vector: Tensor,
    spec: MotorInterfaceSpec,
    promotion: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source)
    checkpoint["controller"] = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    checkpoint["source_checkpoint_sha256"] = source_checkpoint_sha256
    checkpoint["motor_interface_es"] = {
        "method": "ranked antithetic ES on complete flights",
        "parameter_labels": list(spec.labels),
        "parameter_vector": vector.detach().cpu().tolist(),
        "parameter_vector_sha256": vector_sha256(vector),
        "compiled_into_native_parameters": True,
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
    spec = motor_interface_spec(
        controller,
        bias_scale=args.bias_perturbation_scale,
        log_gain_scale=args.log_gain_perturbation_scale,
        maximum_bias_delta=args.maximum_bias_delta,
        maximum_gain_ratio=args.maximum_gain_ratio,
    )
    zero = torch.zeros(len(spec.labels), device=device)
    parity = parity_audit(
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
    if not parity["passed"]:
        raise SystemExit(
            f"batched motor-interface evaluator parity failed: {parity['differences']}"
        )
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
            rng.standard_normal((args.antithetic_directions, len(spec.labels))).astype(np.float32)
        ).to(device)
        delta = sigma * spec.scales * epsilon
        plus = (center + delta).clamp(spec.lower, spec.upper)
        minus = (center - delta).clamp(spec.lower, spec.upper)
        candidates = torch.stack((plus, minus), dim=1).reshape(-1, len(spec.labels))
        policies = torch.cat((zero[None], evaluated_center[None], candidates), dim=0)
        cases = sample_balanced_cases(
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
            shaping_weight=weight,
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
        for index, vector in enumerate(candidates):
            key = vector_sha256(vector)
            archive[key] = {
                "vector": vector.detach().cpu().clone(),
                "generation": generation,
                "kind": f"direction_{index // 2 + 1}_{'plus' if index % 2 == 0 else 'minus'}",
                "development": candidate_summaries[index],
            }
        evaluated_center_key = vector_sha256(evaluated_center)
        next_center_key = vector_sha256(center)
        archive[evaluated_center_key] = {
            "vector": evaluated_center.detach().cpu().clone(),
            "generation": generation,
            "kind": "evaluated_center",
            "development": center_summary,
        }
        best_index = int(np.argmax(fitness))
        best_summary = candidate_summaries[best_index]
        entry = {
            "generation": generation,
            "development_seed": args.development_seed + generation - 1,
            "sigma": sigma,
            "shaping_weight": weight,
            "baseline": baseline_summary,
            "center": center_summary,
            "best_candidate": best_summary,
            "best_candidate_index": best_index + 1,
            "next_center_sha256": next_center_key,
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        }
        generations.append(entry)
        print(
            json.dumps(
                {
                    "phase": "search",
                    "generation": generation,
                    "baseline_success": baseline_summary["success_rate"],
                    "center_success": center_summary["success_rate"],
                    "best_success": best_summary["success_rate"],
                    "baseline_fitness": baseline_summary["fitness"],
                    "best_fitness": best_summary["fitness"],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "center_sha256": next_center_key,
                "parameter_labels": spec.labels,
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
                    "experiment": "motor-interface-es-v1",
                    "generation": generation,
                    "parameter_count": len(spec.labels),
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
            if generation == args.checkpoint_generation:
                validation_delta = max(
                    candidate["success_delta_from_baseline"]
                    for candidate in validation["candidates"].values()
                )
                training_success_delta = max(
                    item["development"]["success_rate"]
                    - generations[item["generation"] - 1]["baseline"]["success_rate"]
                    for item in archive.values()
                )
                training_fitness_delta = max(
                    item["development"]["fitness"]
                    - generations[item["generation"] - 1]["baseline"]["fitness"]
                    for item in archive.values()
                )
                training_flat = (
                    training_success_delta < args.training_success_signal
                    and training_fitness_delta < args.training_fitness_signal
                )
                if validation_delta < args.minimum_validation_improvement and training_flat:
                    early_stop = True
                    print(
                        json.dumps(
                            {
                                "phase": "early_stop",
                                "generation": generation,
                                "validation_success_delta": validation_delta,
                                "training_success_delta": training_success_delta,
                                "training_fitness_delta": training_fitness_delta,
                            }
                        ),
                        flush=True,
                    )
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
        for key, candidate in validation["candidates"].items():
            previous = validated.get(key)
            if previous is None or validation_rank(candidate) > validation_rank(previous):
                validated[key] = candidate
    validation_baseline = validations[-1]["baseline"]
    balanced_eligible = [
        key
        for key, candidate in validated.items()
        if validation_mass_balance_eligible(
            candidate,
            validation_baseline,
            maximum_mass_drop=args.maximum_mass_stratum_drop,
        )
    ]
    selection_pool = balanced_eligible or list(validated)
    winner_key = max(selection_pool, key=lambda key: validation_rank(validated[key]))
    winner = archive[winner_key]["vector"].to(device)
    final_specs = {
        "baseline": (zero, {}),
        "candidate": (winner, {}),
        "candidate_frozen_first_frame": (winner, {"frozen_visual": True}),
        "candidate_constant_1g": (winner, {"frozen_acceleration": True}),
    }
    final: dict[str, Any] = {}
    for name, (vector, controls) in final_specs.items():
        final[name] = standard_evaluation(
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
                {
                    "phase": "final",
                    "name": name,
                    **standard_compact(final[name]),
                }
            ),
            flush=True,
        )
    paired_cases = sample_balanced_cases(
        args.final_episodes,
        seed=args.final_seed,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    controller.load_state_dict(base_state)
    controller.eval()
    paired_batch = evaluate_policy_batch(
        controller,
        torch.stack((zero, winner)),
        spec,
        paired_cases,
        seconds=args.final_seconds,
        shaping_weight=0.0,
        resolution=resolution,
        hover_config=hover_config,
        gate_config=gate_config,
        return_outcomes=True,
    )
    paired = paired_confidence_interval(
        paired_batch["outcomes"]["success"][0],
        paired_batch["outcomes"]["success"][1],
    )
    promotion = final_promotion(
        final["baseline"],
        final["candidate"],
        paired,
        minimum_improvement=args.minimum_validation_improvement,
        maximum_mass_drop=args.maximum_mass_stratum_drop,
    )
    candidate_path = args.output_dir / "candidate.pt"
    if promotion["passed"]:
        compile_vector(controller, base_state, winner, spec)
        save_checkpoint(
            candidate_path,
            checkpoint,
            file_sha256(args.checkpoint),
            controller,
            winner,
            spec,
            promotion,
        )
    goal_passed = bool(
        final["candidate"]["goal_pass"]
        and final["candidate_frozen_first_frame"]["success_rate"] <= 0.05
    )
    ranked_archive = sorted(
        archive,
        key=lambda key: archive_rank(archive[key]),
        reverse=True,
    )
    report = {
        "method": "ranked mirrored antithetic ES on complete flights",
        "claim_scope": (
            "Search-time policy vectors are compiled into existing neuron biases and signed "
            "edge magnitudes. Deployment adds no scaler, memory, estimator, clock, planner, "
            "or actor state beyond the persistent connectome state."
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
            "zeros_preserved": True,
            "time_constants_frozen": True,
            "full_flight_backpropagation": False,
            "exactly_balanced_mass_side_obliquity_strata": True,
            "common_random_numbers_within_generation": True,
            "reward": "10*S + 2*C + shaping_weight*(P+Q) - 2*H - 0.1*U",
            "fitness": "0.75*mean_reward + 0.25*worst_intersection_stratum_reward",
        },
        "parameterization": {
            "count": len(spec.labels),
            "labels": list(spec.labels),
            "description": list(spec.description),
            "bias_delta_bound": args.maximum_bias_delta,
            "gain_ratio_bound": [
                1.0 / args.maximum_gain_ratio,
                args.maximum_gain_ratio,
            ],
        },
        "batched_evaluator_parity": parity,
        "generations_completed": len(generations),
        "early_stopped_at_checkpoint": early_stop,
        "generations": generations,
        "validations": validations,
        "selected_candidate": winner_key,
        "validation_mass_balance_eligible_candidates": balanced_eligible,
        "selected_candidate_validation_mass_balance_eligible": bool(balanced_eligible),
        "selected_parameter_vector": winner.tolist(),
        "selected_parameter_vector_sha256": vector_sha256(winner),
        "top_archive": [
            {
                "sha256": key,
                **{name: value for name, value in archive[key].items() if name != "vector"},
            }
            for key in ranked_archive[:20]
        ],
        "paired_final_success_difference": paired,
        "final_promotion": promotion,
        "candidate_checkpoint": stable_path(candidate_path) if promotion["passed"] else None,
        "candidate_checkpoint_sha256": (
            file_sha256(candidate_path) if promotion["passed"] else None
        ),
        "final": final,
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
