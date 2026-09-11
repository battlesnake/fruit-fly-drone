#!/usr/bin/env python3
"""Evaluate the full-MaleCNS single-gate actor on two uninterrupted gates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_hover as hover_train  # noqa: E402
from search_pragmatic_full_native_gate_es import (  # noqa: E402
    MirroredGateCases,
    sample_mirrored_cases,
)
from train_pragmatic_full_native_gate_imitation import teacher_motor  # noqa: E402

from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    gate_coordinates,
    render_annular_gates_rgb,
    wrap_angle,
)
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402

POLICY_HZ = 50
PHYSICS_HZ = 100


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
        default=REPO_ROOT / "runs/gate/pragmatic-visual-roll-path-001/best-controller.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--observation-warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=630_983)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--layout", choices=("aligned", "s-turn"), default="aligned")
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="drive the training-only staged teacher instead of the connectome actor",
    )
    return parser.parse_args()


def _clone_quad(state: QuadState) -> QuadState:
    return QuadState(*(value.clone() for value in state.as_tuple()))


def _clone_sticks(state: StickState) -> StickState:
    return StickState(
        state.joint_position.clone(),
        state.joint_velocity.clone(),
        state.position.clone(),
        state.velocity.clone(),
    )


def sample_two_gate_cases(
    pairs: int,
    *,
    seed: int,
    device: torch.device,
    hover_config: HoverConfig,
    layout: str = "aligned",
) -> tuple[MirroredGateCases, tuple[AnnularGate, AnnularGate]]:
    """Sample mirrored courses whose first gate matches the learned task."""

    cases = sample_mirrored_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=hover_config,
    )
    side = cases.side
    spacing = torch.empty(pairs, device=device).uniform_(3.8, 4.2).repeat_interleave(2)
    centre = cases.gate.center.clone()
    centre[:, 0] += spacing
    centre[:, 2] = 1.10
    if layout == "aligned":
        first_displacement = cases.gate.center - cases.state.position
        centre[:, 1] = cases.gate.center[:, 1] + spacing * (
            first_displacement[:, 1] / first_displacement[:, 0]
        )
    elif layout == "s-turn":
        lateral = torch.empty(pairs, device=device).uniform_(0.03, 0.08).repeat_interleave(2)
        centre[:, 1] = -side * lateral
    else:
        raise ValueError(f"unknown two-gate layout: {layout}")
    displacement = centre - cases.gate.center
    bearing = torch.atan2(displacement[:, 1], displacement[:, 0])
    if layout == "aligned":
        first_bearing = torch.atan2(first_displacement[:, 1], first_displacement[:, 0])
        obliquity = wrap_angle(cases.gate.yaw - first_bearing)
        yaw = bearing + obliquity
    else:
        obliquity = (
            torch.empty(pairs, device=device)
            .uniform_(math.radians(2.0), math.radians(7.0))
            .repeat_interleave(2)
        )
        yaw = bearing - side * obliquity
    second = AnnularGate(center=centre, yaw=yaw)
    return cases, (cases.gate, second)


def active_gate(
    gates: tuple[AnnularGate, ...],
    current: Tensor,
) -> AnnularGate:
    centres = torch.stack([gate.center for gate in gates], dim=1)
    yaws = torch.stack([gate.yaw for gate in gates], dim=1)
    row = torch.arange(len(current), device=current.device)
    index = current.clamp_max(len(gates) - 1)
    return AnnularGate(center=centres[row, index], yaw=yaws[row, index])


def side_rate(values: Tensor, side: Tensor, negative: bool) -> float:
    mask = side < 0.0 if negative else side > 0.0
    return float(values[mask].float().mean())


@torch.no_grad()
def evaluate(
    controller: ConnectomeController,
    cases: MirroredGateCases,
    gates: tuple[AnnularGate, AnnularGate],
    *,
    seconds: float,
    warmup_steps: int,
    camera: CameraSpec,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    teacher_drives: bool = False,
) -> dict[str, object]:
    device = cases.side.device
    count = len(cases.side)
    row = torch.arange(count, device=device)
    quad = DifferentiableQuad(hover_config).to(device)
    stick_plant = ForelegStickPlant(hover_config).to(device)
    state = _clone_quad(cases.state)
    sticks = _clone_sticks(cases.sticks)
    neural = controller.initial_state(count, device=device, dtype=torch.float32)
    current = torch.zeros(count, dtype=torch.long, device=device)
    passed = torch.zeros(count, len(gates), dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    missed = torch.zeros_like(passed)
    pass_step = torch.full_like(current[:, None].expand(-1, len(gates)), -1)
    ground = torch.zeros(count, dtype=torch.bool, device=device)
    valid = torch.ones_like(ground)
    maximum_tilt = torch.zeros(count, device=device)
    saturation_steps = torch.zeros(count, device=device)

    initial_image = render_annular_gates_rgb(
        state,
        gates,
        current_gate_index=current,
        camera=camera,
        gate_config=gate_config,
    )
    for _ in range(warmup_steps):
        _, neural = controller(initial_image, state.euler[:, :2], neural)

    policy_steps = round(seconds * POLICY_HZ)
    physics_steps = PHYSICS_HZ // POLICY_HZ
    for policy_step in range(policy_steps):
        image = render_annular_gates_rgb(
            state,
            gates,
            current_gate_index=current,
            camera=camera,
            gate_config=gate_config,
        )
        if teacher_drives:
            motor = teacher_motor(
                state,
                active_gate(gates, current),
                hover_config,
                mode="staged",
            )
        else:
            motor, neural = controller(image, state.euler[:, :2], neural)
        saturation_steps += (sticks.position.abs() > 0.98).any(dim=1)
        for _ in range(physics_steps):
            rc, sticks = stick_plant(motor, sticks)
            previous_position = state.position
            state = quad(rc, state, cases.mass_scale)
            active = current < len(gates)
            selected = active_gate(gates, current)
            pass_now, collision_now, miss_now = classify_gate_crossing(
                previous_position,
                state.position,
                selected,
                gate_config,
            )
            pass_now &= active
            collision_now &= active
            miss_now &= active
            index = current.clamp_max(len(gates) - 1)
            new_pass = pass_now & ~passed[row, index]
            passed[row[new_pass], index[new_pass]] = True
            pass_step[row[new_pass], index[new_pass]] = policy_step + 1
            collision[row[collision_now], index[collision_now]] = True
            missed[row[miss_now], index[miss_now]] = True
            current = current + new_pass.long()

        ground |= state.position[:, 2] <= 0.03
        valid &= hover_train.state_is_valid(state)
        maximum_tilt = torch.maximum(
            maximum_tilt,
            torch.linalg.vector_norm(state.euler[:, :2], dim=1),
        )

    final_gate_signed, _, _ = gate_coordinates(state.position, gates[-1])
    first = passed[:, 0]
    both = passed.all(dim=1)
    strict = (
        both
        & (final_gate_signed >= 0.30)
        & ~collision.any(dim=1)
        & ~missed.any(dim=1)
        & ~ground
        & valid
        & (maximum_tilt <= math.radians(45.0))
        & (saturation_steps / policy_steps <= 0.40)
    )
    pair_first = first.reshape(-1, 2).all(dim=1)
    pair_both = both.reshape(-1, 2).all(dim=1)
    pair_strict = strict.reshape(-1, 2).all(dim=1)
    first_times = pass_step[:, 0][first].float() / POLICY_HZ
    second_times = pass_step[:, 1][both].float() / POLICY_HZ
    return {
        "episodes": count,
        "mirrored_pairs": count // 2,
        "seconds": seconds,
        "observation_warmup_seconds": warmup_steps / POLICY_HZ,
        "first_gate_pass_rate": float(first.float().mean()),
        "first_gate_paired_pass_rate": float(pair_first.float().mean()),
        "first_gate_negative_offset_pass_rate": side_rate(first, cases.side, True),
        "first_gate_positive_offset_pass_rate": side_rate(first, cases.side, False),
        "both_gates_pass_rate": float(both.float().mean()),
        "both_gates_paired_pass_rate": float(pair_both.float().mean()),
        "both_gates_negative_course_pass_rate": side_rate(both, cases.side, True),
        "both_gates_positive_course_pass_rate": side_rate(both, cases.side, False),
        "strict_two_gate_success_rate": float(strict.float().mean()),
        "strict_two_gate_paired_success_rate": float(pair_strict.float().mean()),
        "first_gate_ring_collision_rate": float(collision[:, 0].float().mean()),
        "second_gate_ring_collision_rate": float(collision[:, 1].float().mean()),
        "first_gate_miss_rate": float(missed[:, 0].float().mean()),
        "second_gate_miss_rate": float(missed[:, 1].float().mean()),
        "ground_contact_rate": float(ground.float().mean()),
        "invalid_rate": float((~valid).float().mean()),
        "maximum_tilt_mean_degrees": float(torch.rad2deg(maximum_tilt).mean()),
        "stick_saturation_fraction_mean": float((saturation_steps / policy_steps).mean()),
        "first_pass_time_mean_seconds": (
            float(first_times.mean()) if bool(first_times.numel()) else None
        ),
        "second_pass_time_mean_seconds": (
            float(second_times.mean()) if bool(second_times.numel()) else None
        ),
    }


def main() -> int:
    args = parse_args()
    if args.pairs < 1 or args.seconds <= 0 or args.observation_warmup_steps < 0:
        raise SystemExit("pairs/seconds must be positive and warmup must be nonnegative")
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("graph_sha256") != responsibility.file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.eval().requires_grad_(False)
    hover_config = HoverConfig(**payload["hover_config"])
    gate_config = GateConfig(**payload["gate_config"])
    camera = CameraSpec(
        width=payload["image_resolution"][0],
        height=payload["image_resolution"][1],
        horizontal_fov_degrees=payload["camera_hfov_degrees"],
    )
    cases, gates = sample_two_gate_cases(
        args.pairs,
        seed=args.seed,
        device=device,
        hover_config=hover_config,
        layout=args.layout,
    )
    metrics = evaluate(
        controller,
        cases,
        gates,
        seconds=args.seconds,
        warmup_steps=args.observation_warmup_steps,
        camera=camera,
        hover_config=hover_config,
        gate_config=gate_config,
        teacher_drives=args.teacher,
    )
    result = {
        "experiment": "pragmatic-full-native-two-gate-zero-shot-v1",
        "purpose": (
            "training-only teacher preflight"
            if args.teacher
            else "behavior-first zero-shot feasibility test"
        ),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "continuous_state": ["MaleCNS recurrence", "forelegs", "sticks", "aircraft"],
        "actor_gate_index_or_pass_input": False,
        "gate_roles": ["current green", "next red", "passed black"],
        "geometry": {
            "layout": args.layout,
            "first_gate_matches_single_gate_training_distribution": True,
            "second_gate_spacing_metres": [3.8, 4.2],
            "second_gate_absolute_lateral_offset_metres": (
                [0.03, 0.08] if args.layout == "s-turn" else None
            ),
            "mirrored_courses": True,
        },
        "metrics": metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
