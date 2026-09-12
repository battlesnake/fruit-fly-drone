#!/usr/bin/env python3
"""Evaluate the full-MaleCNS single-gate actor on uninterrupted gate sequences."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
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
    crossing_coordinates,
    gate_coordinates,
    render_annular_gates_rgb,
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
    parser.add_argument("--gates", type=int, default=2)
    parser.add_argument("--spacing-min", type=float, default=3.8)
    parser.add_argument("--spacing-max", type=float, default=4.2)
    parser.add_argument(
        "--inner-radius",
        type=float,
        help="optional beginner-course aperture radius override",
    )
    parser.add_argument(
        "--outer-radius",
        type=float,
        help="optional beginner-course ring outer radius override",
    )
    parser.add_argument(
        "--frozen-vision",
        action="store_true",
        help="freeze the RGB frame after 0.5 s while the rest of the actor stays live",
    )
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
    spacing_range: tuple[float, float] = (3.8, 4.2),
    gate_count: int = 2,
) -> tuple[MirroredGateCases, tuple[AnnularGate, ...]]:
    """Sample mirrored courses whose first gate matches the learned task.

    Aligned courses may contain any positive number of equally spaced gates.  The
    two-gate S-turn remains the only intentionally non-collinear layout.
    """

    if gate_count < 1:
        raise ValueError("gate_count must be positive")
    if layout == "s-turn" and gate_count != 2:
        raise ValueError("s-turn currently supports exactly two gates")

    cases = sample_mirrored_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=hover_config,
    )
    if gate_count == 1:
        return cases, (cases.gate,)
    side = cases.side
    spacing = torch.empty(pairs, device=device).uniform_(*spacing_range).repeat_interleave(2)
    first_displacement = cases.gate.center - cases.state.position
    gates: list[AnnularGate] = [cases.gate]
    for gate_number in range(1, gate_count):
        centre = cases.gate.center.clone()
        centre[:, 0] += gate_number * spacing
        centre[:, 2] = 1.10
        if layout == "aligned":
            centre[:, 1] = cases.gate.center[:, 1] + gate_number * spacing * (
                first_displacement[:, 1] / first_displacement[:, 0]
            )
            yaw = cases.gate.yaw.clone()
        elif layout == "s-turn":
            lateral = (
                torch.empty(pairs, device=device)
                .uniform_(0.03, 0.08)
                .repeat_interleave(2)
            )
            centre[:, 1] = -side * lateral
            displacement = centre - cases.gate.center
            bearing = torch.atan2(displacement[:, 1], displacement[:, 0])
            obliquity = (
                torch.empty(pairs, device=device)
                .uniform_(math.radians(2.0), math.radians(7.0))
                .repeat_interleave(2)
            )
            yaw = bearing - side * obliquity
        else:
            raise ValueError(f"unknown gate layout: {layout}")
        gates.append(AnnularGate(center=centre, yaw=yaw))
    return cases, tuple(gates)


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
    gates: tuple[AnnularGate, ...],
    *,
    seconds: float,
    warmup_steps: int,
    camera: CameraSpec,
    hover_config: HoverConfig,
    gate_config: GateConfig,
    teacher_drives: bool = False,
    frozen_vision: bool = False,
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
    crossing_step = torch.full_like(pass_step, -1)
    crossing_lateral = torch.full(
        (count, len(gates)), float("nan"), device=device
    )
    crossing_vertical = torch.full_like(crossing_lateral, float("nan"))
    ground = torch.zeros(count, dtype=torch.bool, device=device)
    valid = torch.ones_like(ground)
    maximum_tilt = torch.zeros(count, device=device)
    saturation_steps = torch.zeros(count, device=device)
    freeze_step = round(0.5 * POLICY_HZ)
    frozen_image: Tensor | None = None

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
        if policy_step == freeze_step:
            frozen_image = image.clone()
        actor_image = frozen_image if frozen_vision and frozen_image is not None else image
        if teacher_drives:
            motor = teacher_motor(
                state,
                active_gate(gates, current),
                hover_config,
                mode="staged",
            )
        else:
            motor, neural = controller(actor_image, state.euler[:, :2], neural)
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
            crossing_now = pass_now | collision_now | miss_now
            new_crossing = crossing_now & crossing_lateral[row, index].isnan()
            if bool(new_crossing.any()):
                _, lateral, vertical = crossing_coordinates(
                    previous_position,
                    state.position,
                    selected,
                )
                crossing_step[row[new_crossing], index[new_crossing]] = policy_step + 1
                crossing_lateral[row[new_crossing], index[new_crossing]] = lateral[new_crossing]
                crossing_vertical[row[new_crossing], index[new_crossing]] = vertical[new_crossing]
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
    gate_pass_rates = passed.float().mean(dim=0)
    gate_crossed = crossing_lateral.isfinite()
    gate_radial = torch.sqrt(crossing_lateral.square() + crossing_vertical.square())

    def finite_gate_mean(
        values: Tensor,
        gate_number: int,
        *,
        absolute: bool = False,
    ) -> float | None:
        selected = values[:, gate_number][gate_crossed[:, gate_number]]
        if not bool(selected.numel()):
            return None
        return float((selected.abs() if absolute else selected).mean())

    per_gate = []
    for gate_number in range(len(gates)):
        reached = gate_crossed[:, gate_number]
        times = crossing_step[:, gate_number][reached].float() / POLICY_HZ
        per_gate.append(
            {
                "gate_number": gate_number + 1,
                "pass_rate": float(gate_pass_rates[gate_number]),
                "plane_crossing_rate": float(reached.float().mean()),
                "crossing_lateral_mean_metres": finite_gate_mean(
                    crossing_lateral, gate_number
                ),
                "crossing_lateral_absolute_mean_metres": finite_gate_mean(
                    crossing_lateral, gate_number, absolute=True
                ),
                "crossing_vertical_absolute_mean_metres": finite_gate_mean(
                    crossing_vertical, gate_number, absolute=True
                ),
                "crossing_radial_mean_metres": finite_gate_mean(
                    gate_radial, gate_number
                ),
                "plane_crossing_time_mean_seconds": (
                    float(times.mean()) if bool(times.numel()) else None
                ),
            }
        )
    first_times = pass_step[:, 0][first].float() / POLICY_HZ
    second_times = pass_step[:, 1][both].float() / POLICY_HZ
    second_crossed = crossing_lateral[:, 1].isfinite()
    second_crossing_times = crossing_step[:, 1][second_crossed].float() / POLICY_HZ
    second_radial = torch.sqrt(
        crossing_lateral[:, 1].square() + crossing_vertical[:, 1].square()
    )
    return {
        "episodes": count,
        "mirrored_pairs": count // 2,
        "seconds": seconds,
        "observation_warmup_seconds": warmup_steps / POLICY_HZ,
        "frozen_vision_after_seconds": 0.5 if frozen_vision else None,
        "gate_count": len(gates),
        "per_gate": per_gate,
        "first_gate_pass_rate": float(first.float().mean()),
        "first_gate_paired_pass_rate": float(pair_first.float().mean()),
        "first_gate_negative_offset_pass_rate": side_rate(first, cases.side, True),
        "first_gate_positive_offset_pass_rate": side_rate(first, cases.side, False),
        "both_gates_pass_rate": float(both.float().mean()),
        "all_gates_pass_rate": float(both.float().mean()),
        "second_gate_pass_given_first_rate": (
            float(both.sum() / first.sum()) if bool(first.any()) else None
        ),
        "all_gates_pass_given_first_rate": (
            float(both.sum() / first.sum()) if bool(first.any()) else None
        ),
        "both_gates_paired_pass_rate": float(pair_both.float().mean()),
        "all_gates_paired_pass_rate": float(pair_both.float().mean()),
        "both_gates_negative_course_pass_rate": side_rate(both, cases.side, True),
        "both_gates_positive_course_pass_rate": side_rate(both, cases.side, False),
        "strict_two_gate_success_rate": float(strict.float().mean()),
        "strict_all_gate_success_rate": float(strict.float().mean()),
        "strict_two_gate_paired_success_rate": float(pair_strict.float().mean()),
        "first_gate_ring_collision_rate": float(collision[:, 0].float().mean()),
        "second_gate_ring_collision_rate": float(collision[:, 1].float().mean()),
        "first_gate_miss_rate": float(missed[:, 0].float().mean()),
        "second_gate_miss_rate": float(missed[:, 1].float().mean()),
        "second_gate_plane_crossing_rate": float(second_crossed.float().mean()),
        "second_gate_crossing_lateral_mean_metres": (
            float(crossing_lateral[:, 1][second_crossed].mean())
            if bool(second_crossed.any())
            else None
        ),
        "second_gate_crossing_lateral_absolute_mean_metres": (
            float(crossing_lateral[:, 1][second_crossed].abs().mean())
            if bool(second_crossed.any())
            else None
        ),
        "second_gate_crossing_vertical_mean_metres": (
            float(crossing_vertical[:, 1][second_crossed].mean())
            if bool(second_crossed.any())
            else None
        ),
        "second_gate_crossing_vertical_absolute_mean_metres": (
            float(crossing_vertical[:, 1][second_crossed].abs().mean())
            if bool(second_crossed.any())
            else None
        ),
        "second_gate_crossing_radial_mean_metres": (
            float(second_radial[second_crossed].mean()) if bool(second_crossed.any()) else None
        ),
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
        "second_plane_crossing_time_mean_seconds": (
            float(second_crossing_times.mean())
            if bool(second_crossing_times.numel())
            else None
        ),
        "ended_before_first_gate": int((current == 0).sum()),
        "ended_between_gates": int((current == 1).sum()),
        "ended_after_second_gate": int((current == 2).sum()),
        "ended_at_gate_index_counts": [
            int((current == gate_number).sum()) for gate_number in range(len(gates) + 1)
        ],
        "all_gate_success_episode_indices": torch.nonzero(both, as_tuple=False)
        .squeeze(1)
        .cpu()
        .tolist(),
    }


def main() -> int:
    args = parse_args()
    if args.pairs < 1 or args.seconds <= 0 or args.observation_warmup_steps < 0:
        raise SystemExit("pairs/seconds must be positive and warmup must be nonnegative")
    if args.spacing_min <= 0 or args.spacing_max < args.spacing_min:
        raise SystemExit("spacing range must be positive and ordered")
    if args.gates < 1 or (args.layout == "s-turn" and args.gates != 2):
        raise SystemExit("gates must be positive; s-turn supports exactly two")
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
    if args.inner_radius is not None or args.outer_radius is not None:
        gate_config = replace(
            gate_config,
            inner_radius=(
                args.inner_radius
                if args.inner_radius is not None
                else gate_config.inner_radius
            ),
            outer_radius=(
                args.outer_radius
                if args.outer_radius is not None
                else gate_config.outer_radius
            ),
        )
    if not gate_config.drone_radius < gate_config.inner_radius < gate_config.outer_radius:
        raise SystemExit("gate radii must satisfy drone < inner < outer")
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
        spacing_range=(args.spacing_min, args.spacing_max),
        gate_count=args.gates,
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
        frozen_vision=args.frozen_vision,
    )
    result = {
        "experiment": "pragmatic-full-native-gate-sequence-zero-shot-v1",
        "purpose": (
            "training-only teacher preflight"
            if args.teacher
            else "behavior-first zero-shot feasibility test"
        ),
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "continuous_state": ["MaleCNS recurrence", "forelegs", "sticks", "aircraft"],
        "actor_gate_index_or_pass_input": False,
        "gate_roles": [
            "current green",
            "next red",
            "later blue",
            "passed black",
        ],
        "geometry": {
            "layout": args.layout,
            "first_gate_matches_single_gate_training_distribution": True,
            "second_gate_spacing_metres": [args.spacing_min, args.spacing_max],
            "gate_count": args.gates,
            "uniform_inter_gate_spacing_metres": [args.spacing_min, args.spacing_max],
            "inner_radius_metres": gate_config.inner_radius,
            "outer_radius_metres": gate_config.outer_radius,
            "clean_aperture_radius_metres": (
                gate_config.inner_radius - gate_config.drone_radius
            ),
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
