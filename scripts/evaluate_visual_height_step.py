#!/usr/bin/env python3
"""Evaluate paired nominal-mass closed-loop marker steps after visual response fitting."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
)
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--prefix-seconds", type=float, default=2.0)
    parser.add_argument("--response-seconds", type=float, default=4.0)
    parser.add_argument("--marker-step-metres", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def clone_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def clone_sticks(state: StickState) -> StickState:
    return StickState(
        state.joint_position.clone(),
        state.joint_velocity.clone(),
        state.position.clone(),
        state.velocity.clone(),
    )


def settled_sticks(batch: int, device: torch.device, config: HoverConfig) -> StickState:
    rc = torch.zeros(batch, 4, device=device)
    rc[:, 3] = 1.0 / config.thrust_to_weight
    normalized = torch.cat((rc[:, :3], 2.0 * rc[:, 3:4] - 1.0), dim=1)
    sine_limit = math.sin(config.foreleg_joint_limit)
    joint = torch.asin((normalized * sine_limit).clamp(-1.0, 1.0))
    zeros = torch.zeros_like(normalized)
    return StickState(joint, zeros.clone(), normalized, zeros)


def advance_physics(
    quad: DifferentiableQuad,
    sticks: ForelegStickPlant,
    motor: torch.Tensor,
    physical: QuadState,
    stick_state: StickState,
    physics_steps: int,
) -> tuple[QuadState, StickState, torch.Tensor]:
    mass_scale = torch.ones(motor.shape[0], device=motor.device)
    rc = torch.zeros_like(motor)
    for _ in range(physics_steps):
        rc, stick_state = sticks(motor, stick_state)
        physical = quad(rc, physical, mass_scale)
    return physical, stick_state, rc


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy-hz must evenly divide the 100 Hz physics rate")
    physics_steps = physics_hz // args.policy_hz
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    quad = DifferentiableQuad(config).to(device)
    sticks = ForelegStickPlant(config).to(device)

    initial_target = torch.empty(args.episodes, device=device).uniform_(0.90, 1.10)
    position = torch.zeros(args.episodes, 3, device=device)
    position[:, 2] = initial_target
    euler = torch.zeros(args.episodes, 3, device=device)
    euler[:, :2] = torch.empty(args.episodes, 2, device=device).uniform_(
        -math.radians(2.0), math.radians(2.0)
    )
    physical = quad.initial_state(
        args.episodes,
        device=device,
        dtype=torch.float32,
        position=position,
        euler=euler,
    )
    physical.actuator[:, 0] = 1.0 / config.thrust_to_weight
    stick_state = settled_sticks(args.episodes, device, config)
    neural = controller.initial_state(args.episodes, device=device, dtype=torch.float32)
    motor = torch.zeros(args.episodes, 4, device=device)
    prefix_heights = []
    prefix_ground = torch.zeros(args.episodes, dtype=torch.bool, device=device)
    for _ in range(round(args.prefix_seconds * args.policy_hz)):
        image = render_visual_hover_scene(physical, initial_target, config=config)
        motor, neural = controller(image, physical.euler[:, :2], neural)
        physical, stick_state, _ = advance_physics(
            quad, sticks, motor, physical, stick_state, physics_steps
        )
        prefix_heights.append(physical.position[:, 2])
        prefix_ground |= physical.position[:, 2] <= 0.005

    direction = torch.where(
        torch.arange(args.episodes, device=device) % 2 == 0,
        torch.ones(args.episodes, device=device),
        -torch.ones(args.episodes, device=device),
    )
    stepped_target = initial_target + direction * args.marker_step_metres
    frozen_image = render_visual_hover_scene(physical, initial_target, config=config)
    live_physical = clone_quad(physical)
    control_physical = clone_quad(physical)
    frozen_physical = clone_quad(physical)
    live_sticks = clone_sticks(stick_state)
    control_sticks = clone_sticks(stick_state)
    frozen_sticks = clone_sticks(stick_state)
    live_neural = neural.clone()
    control_neural = neural.clone()
    frozen_neural = neural.clone()
    live_heights = []
    control_heights = []
    frozen_heights = []
    live_tilts = []
    signed_motor_effects = []
    signed_stick_effects = []
    signed_actuator_effects = []
    signed_velocity_effects = []
    signed_height_effects = []
    live_ground = prefix_ground.clone()
    control_ground = prefix_ground.clone()
    frozen_ground = prefix_ground.clone()
    for _ in range(round(args.response_seconds * args.policy_hz)):
        live_image = render_visual_hover_scene(live_physical, stepped_target, config=config)
        live_motor, live_neural = controller(live_image, live_physical.euler[:, :2], live_neural)
        control_image = render_visual_hover_scene(control_physical, initial_target, config=config)
        control_motor, control_neural = controller(
            control_image, control_physical.euler[:, :2], control_neural
        )
        frozen_motor, frozen_neural = controller(
            frozen_image, frozen_physical.euler[:, :2], frozen_neural
        )
        live_physical, live_sticks, live_rc = advance_physics(
            quad, sticks, live_motor, live_physical, live_sticks, physics_steps
        )
        control_physical, control_sticks, control_rc = advance_physics(
            quad, sticks, control_motor, control_physical, control_sticks, physics_steps
        )
        frozen_physical, frozen_sticks, _ = advance_physics(
            quad, sticks, frozen_motor, frozen_physical, frozen_sticks, physics_steps
        )
        signed_motor_effects.append((live_motor[:, 3] - control_motor[:, 3]) * direction)
        signed_stick_effects.append((live_rc[:, 3] - control_rc[:, 3]) * direction)
        signed_actuator_effects.append(
            (live_physical.actuator[:, 0] - control_physical.actuator[:, 0]) * direction
        )
        signed_velocity_effects.append(
            (live_physical.velocity[:, 2] - control_physical.velocity[:, 2]) * direction
        )
        signed_height_effects.append(
            (live_physical.position[:, 2] - control_physical.position[:, 2]) * direction
        )
        live_heights.append(live_physical.position[:, 2])
        control_heights.append(control_physical.position[:, 2])
        frozen_heights.append(frozen_physical.position[:, 2])
        live_tilts.append(torch.linalg.vector_norm(live_physical.euler[:, :2], dim=1))
        live_ground |= live_physical.position[:, 2] <= 0.005
        control_ground |= control_physical.position[:, 2] <= 0.005
        frozen_ground |= frozen_physical.position[:, 2] <= 0.005

    prefix_height = torch.stack(prefix_heights)
    live_height = torch.stack(live_heights)
    control_height = torch.stack(control_heights)
    frozen_height = torch.stack(frozen_heights)
    live_tilt = torch.stack(live_tilts)
    signed_motor = torch.stack(signed_motor_effects)
    signed_stick = torch.stack(signed_stick_effects)
    signed_actuator = torch.stack(signed_actuator_effects)
    signed_velocity = torch.stack(signed_velocity_effects)
    signed_height = torch.stack(signed_height_effects)
    prefix_window = min(round(1.0 * args.policy_hz), prefix_height.shape[0])
    response_window = min(round(1.0 * args.policy_hz), live_height.shape[0])
    before = prefix_height[-prefix_window:].mean(dim=0)
    live_after = live_height[-response_window:].mean(dim=0)
    control_after = control_height[-response_window:].mean(dim=0)
    frozen_after = frozen_height[-response_window:].mean(dim=0)
    target_delta = stepped_target - initial_target
    live_effect = live_after - control_after
    frozen_effect = frozen_after - control_after
    response_ratio = live_effect / target_delta
    correct_direction = live_effect * target_delta > 0.0
    live_target_error = live_after - stepped_target
    control_target_error = control_after - initial_target
    frozen_target_error = frozen_after - initial_target
    prefix_error = before - initial_target
    tilt_rms = torch.sqrt(live_tilt.square().mean(dim=0))
    success = (
        correct_direction
        & (response_ratio >= 0.5)
        & (response_ratio <= 1.5)
        & (live_target_error.abs() <= 0.15)
        & (tilt_rms <= math.radians(6.0))
        & ~live_ground
    )
    early_steps = min(round(0.5 * args.policy_hz), signed_motor.shape[0])
    return {
        "experiment": "nominal-mass-closed-loop-marker-step-v1",
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "policy_hz": args.policy_hz,
        "physics_hz": physics_hz,
        "prefix_seconds": args.prefix_seconds,
        "response_seconds": args.response_seconds,
        "marker_step_metres": args.marker_step_metres,
        "mass_scale": 1.0,
        "initial_marker_height_range_metres": [0.90, 1.10],
        "paired_state_at_marker_step": True,
        "prefix_height_rmse_m": float(torch.sqrt(prefix_error.square().mean())),
        "direction_correct_rate": float(correct_direction.float().mean()),
        "response_ratio_mean": float(response_ratio.mean()),
        "response_ratio_p10": float(torch.quantile(response_ratio, 0.10)),
        "response_ratio_p90": float(torch.quantile(response_ratio, 0.90)),
        "live_paired_height_effect_mean_m": float(live_effect.mean()),
        "live_final_target_error_mean_absolute_m": float(live_target_error.abs().mean()),
        "control_final_target_error_mean_absolute_m": float(control_target_error.abs().mean()),
        "frozen_final_original_target_error_mean_absolute_m": float(
            frozen_target_error.abs().mean()
        ),
        "frozen_before_step_effect_mean_absolute_m": float(frozen_effect.abs().mean()),
        "causal_chain_first_500ms": {
            "signed_motor_drive_mean": float(signed_motor[:early_steps].mean()),
            "signed_motor_drive_peak": float(signed_motor[:early_steps].max()),
            "signed_throttle_stick_mean": float(signed_stick[:early_steps].mean()),
            "signed_throttle_stick_peak": float(signed_stick[:early_steps].max()),
            "signed_collective_actuator_mean": float(signed_actuator[:early_steps].mean()),
            "signed_vertical_velocity_mean_mps": float(signed_velocity[:early_steps].mean()),
            "signed_height_mean_m": float(signed_height[:early_steps].mean()),
        },
        "causal_chain_full_response": {
            "signed_motor_drive_mean": float(signed_motor.mean()),
            "signed_throttle_stick_mean": float(signed_stick.mean()),
            "signed_collective_actuator_mean": float(signed_actuator.mean()),
            "signed_vertical_velocity_mean_mps": float(signed_velocity.mean()),
            "signed_height_mean_m": float(signed_height.mean()),
        },
        "live_tilt_rms_mean_degrees": float(torch.rad2deg(tilt_rms).mean()),
        "live_ground_contact_rate": float(live_ground.float().mean()),
        "control_ground_contact_rate": float(control_ground.float().mean()),
        "frozen_ground_contact_rate": float(frozen_ground.float().mean()),
        "success_rate": float(success.float().mean()),
        "pass": float(success.float().mean()) >= 0.90,
    }


def main() -> int:
    args = parse_args()
    report = evaluate(args)
    rendered = f"{json.dumps(report, indent=2, sort_keys=True)}\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
