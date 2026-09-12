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

from flydrone.course_teacher import (  # noqa: E402
    CoursePath,
    CourseTeacherConfig,
    course_teacher_motor,
)
from flydrone.gate import (  # noqa: E402
    AnnularGate,
    GateConfig,
    gate_coordinates,
    render_annular_gates_rgb,
)
from flydrone.gate_course import classify_course_step  # noqa: E402
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
    parser.add_argument(
        "--layout",
        choices=("aligned", "s-turn", "variable"),
        default="aligned",
    )
    parser.add_argument("--gates", type=int, default=2)
    parser.add_argument("--gate-back-pattern", choices=("solid", "checkerboard"))
    parser.add_argument("--yaw-jitter-degrees", type=float, default=0.0)
    parser.add_argument("--spacing-min", type=float, default=3.8)
    parser.add_argument("--spacing-max", type=float, default=4.2)
    parser.add_argument(
        "--lateral-step-min",
        type=float,
        default=0.0,
        help="minimum absolute per-gate lateral perturbation for variable courses",
    )
    parser.add_argument(
        "--lateral-step-max",
        type=float,
        default=0.10,
        help="maximum absolute per-gate lateral perturbation for variable courses",
    )
    parser.add_argument(
        "--lateral-deviation-limit",
        type=float,
        default=0.25,
        help="maximum deviation from the first-gate centreline for variable courses",
    )
    parser.add_argument(
        "--height-step-min",
        type=float,
        default=0.0,
        help="minimum absolute per-gate height change for variable courses",
    )
    parser.add_argument(
        "--height-step-max",
        type=float,
        default=0.05,
        help="maximum absolute per-gate height change for variable courses",
    )
    parser.add_argument("--height-min", type=float, default=0.95)
    parser.add_argument("--height-max", type=float, default=1.25)
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
    lateral_step_range: tuple[float, float] = (0.0, 0.10),
    lateral_deviation_limit: float = 0.25,
    height_step_range: tuple[float, float] = (0.0, 0.05),
    height_range: tuple[float, float] = (0.95, 1.25),
    yaw_jitter_degrees: float = 0.0,
) -> tuple[MirroredGateCases, tuple[AnnularGate, ...]]:
    """Sample mirrored courses whose first gate matches the learned task.

    Aligned courses may contain any positive number of equally spaced gates. Variable
    courses perturb every later gate relative to that first-gate centreline. Paired
    episodes share the height path and mirror the lateral path; this makes left/right
    comparisons useful without exposing course state to the actor.
    """

    if gate_count < 1:
        raise ValueError("gate_count must be positive")
    if layout == "s-turn" and gate_count != 2:
        raise ValueError("s-turn currently supports exactly two gates")
    for name, bounds in (
        ("spacing_range", spacing_range),
        ("lateral_step_range", lateral_step_range),
        ("height_step_range", height_step_range),
        ("height_range", height_range),
    ):
        if bounds[0] < 0.0 or bounds[1] < bounds[0]:
            raise ValueError(f"{name} must be nonnegative and ordered")
    if spacing_range[0] <= 0.0:
        raise ValueError("spacing_range must be positive")
    if lateral_deviation_limit < 0.0:
        raise ValueError("lateral_deviation_limit must be nonnegative")
    if not math.isfinite(yaw_jitter_degrees) or yaw_jitter_degrees < 0.0:
        raise ValueError("yaw_jitter_degrees must be finite and nonnegative")

    cases = sample_mirrored_cases(
        pairs,
        seed=seed,
        device=device,
        hover_config=hover_config,
    )
    if gate_count == 1:
        return cases, (cases.gate,)
    side = cases.side
    first_displacement = cases.gate.center - cases.state.position
    gates: list[AnnularGate] = [cases.gate]
    if layout == "aligned":
        spacing = (
            torch.empty(pairs, device=device)
            .uniform_(*spacing_range)
            .repeat_interleave(2)
        )
    cumulative_distance = torch.zeros(2 * pairs, device=device)
    lateral_deviation = torch.zeros(pairs, device=device)
    height = torch.full((pairs,), 1.10, device=device)
    for gate_number in range(1, gate_count):
        centre = cases.gate.center.clone()
        if layout == "aligned":
            cumulative_distance = gate_number * spacing
            centre[:, 0] += cumulative_distance
            centre[:, 1] = cases.gate.center[:, 1] + cumulative_distance * (
                first_displacement[:, 1] / first_displacement[:, 0]
            )
            centre[:, 2] = 1.10
            yaw = cases.gate.yaw.clone()
        elif layout == "s-turn":
            spacing = (
                torch.empty(pairs, device=device)
                .uniform_(*spacing_range)
                .repeat_interleave(2)
            )
            centre[:, 0] += spacing
            centre[:, 2] = 1.10
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
        elif layout == "variable":
            spacing = (
                torch.empty(pairs, device=device)
                .uniform_(*spacing_range)
                .repeat_interleave(2)
            )
            cumulative_distance += spacing
            lateral_sign = torch.where(
                torch.rand(pairs, device=device) < 0.5,
                -torch.ones(pairs, device=device),
                torch.ones(pairs, device=device),
            )
            lateral_step = (
                torch.empty(pairs, device=device).uniform_(*lateral_step_range)
                * lateral_sign
            )
            lateral_deviation = (lateral_deviation + lateral_step).clamp(
                -lateral_deviation_limit,
                lateral_deviation_limit,
            )
            height_sign = torch.where(
                torch.rand(pairs, device=device) < 0.5,
                -torch.ones(pairs, device=device),
                torch.ones(pairs, device=device),
            )
            height_step = (
                torch.empty(pairs, device=device).uniform_(*height_step_range)
                * height_sign
            )
            height = (height + height_step).clamp(*height_range)
            centre[:, 0] += cumulative_distance
            centre[:, 1] = (
                cases.gate.center[:, 1]
                + cumulative_distance
                * (first_displacement[:, 1] / first_displacement[:, 0])
                + side * lateral_deviation.repeat_interleave(2)
            )
            centre[:, 2] = height.repeat_interleave(2)
            yaw = cases.gate.yaw.clone()
        else:
            raise ValueError(f"unknown gate layout: {layout}")
        gates.append(AnnularGate(center=centre, yaw=yaw))
    if yaw_jitter_degrees:
        # A separate stream preserves matched positions when auditing angle variation.
        rng = torch.Generator(device=device).manual_seed(seed + 104_729)
        limit = math.radians(yaw_jitter_degrees)
        for index in range(1, len(gates)):
            jitter = torch.empty(pairs, device=device).uniform_(-limit, limit, generator=rng)
            gates[index] = AnnularGate(
                center=gates[index].center,
                yaw=gates[index].yaw + side * jitter.repeat_interleave(2),
            )
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


def diagnostic_axis_takeover(native: Tensor, teacher: Tensor, current: Tensor, axes: Tensor):
    """Privileged diagnostic only: replace selected motor axes after gate one."""
    return torch.where((current >= 1)[:, None] & axes[None], teacher, native)


def compact_policy_metrics(
    first: Tensor,
    clean: Tensor,
    failed: Tensor,
    prefix: Tensor,
    centering: Tensor,
    side: Tensor,
    gate_count: int,
    policy_count: int,
) -> list[dict[str, float | int]]:
    """Keep independent candidate scores when several policies share a GPU batch."""
    if policy_count < 1 or len(first) % (2 * policy_count):
        raise ValueError("each policy requires whole mirrored pairs")
    rows = len(first) // policy_count
    summaries = []
    fitness = prefix + 5.0 * clean - 2.0 * failed + 0.2 * centering / gate_count
    for index in range(policy_count):
        take = slice(index * rows, (index + 1) * rows)
        summaries.append(dict(
            episodes=rows,
            first_gate_pass_rate=float(first[take].float().mean()),
            clean_course_success_rate=float(clean[take].float().mean()),
            clean_course_negative_success_rate=side_rate(clean[take], side[take], True),
            clean_course_positive_success_rate=side_rate(clean[take], side[take], False),
            course_failure_rate=float(failed[take].float().mean()),
            gates_before_failure_mean=float(prefix[take].mean()),
            course_race_fitness=float(fitness[take].mean()),
        ))
    return summaries


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
    compact_policies: int | None = None,
    diagnostic_teacher_axes: tuple[bool, bool, bool, bool] | None = None,
) -> dict[str, object] | list[dict[str, float | int]]:
    device = cases.side.device
    count = len(cases.side)
    if compact_policies is not None and (
        compact_policies < 1 or count % (2 * compact_policies)
    ):
        raise ValueError("each policy requires whole mirrored pairs")
    quad = DifferentiableQuad(hover_config).to(device)
    stick_plant = ForelegStickPlant(hover_config).to(device)
    state = _clone_quad(cases.state)
    takeover_path = None
    takeover_axes = None
    if diagnostic_teacher_axes is not None:
        if len(diagnostic_teacher_axes) != 4 or teacher_drives or compact_policies is not None:
            raise ValueError("axis takeover requires four axes and ordinary native evaluation")
        takeover_path = CoursePath.through_gates(state.position, gates)
        takeover_axes = torch.tensor(diagnostic_teacher_axes, device=device, dtype=torch.bool)
    sticks = _clone_sticks(cases.sticks)
    neural = controller.initial_state(count, device=device, dtype=torch.float32)
    current = torch.zeros(count, dtype=torch.long, device=device)
    passed = torch.zeros(count, len(gates), dtype=torch.bool, device=device)
    collision = torch.zeros_like(passed)
    wrong_order_count = torch.zeros_like(passed, dtype=torch.long)
    wrong_direction_count = torch.zeros_like(wrong_order_count)
    course_penalty = torch.zeros(count, device=device)
    prefix_passes = torch.zeros(count, device=device)
    failed_prefix = torch.zeros(count, dtype=torch.bool, device=device)
    clean_first = torch.zeros_like(failed_prefix)
    prefix_centering = torch.zeros(count, device=device)
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
        if takeover_path is not None:
            # The fly still receives only its ordinary sensors, and its native
            # recurrence continues even when privileged teacher muscles act.
            target = course_teacher_motor(
                state, takeover_path, hover_config, CourseTeacherConfig(heading_mode="rate-damped")
            )
            motor = diagnostic_axis_takeover(motor, target, current, takeover_axes)
        saturation_steps += (sticks.position.abs() > 0.98).any(dim=1)
        for _ in range(physics_steps):
            rc, sticks = stick_plant(motor, sticks)
            previous_position = state.position
            state = quad(rc, state, cases.mass_scale)
            events = classify_course_step(
                previous_position, state.position, gates, current, gate_config
            )
            new_crossing = events.expected_forward_crossing & crossing_lateral.isnan()
            crossing_step[new_crossing] = policy_step + 1
            crossing_lateral[new_crossing] = events.crossing_lateral[new_crossing]
            crossing_vertical[new_crossing] = events.crossing_vertical[new_crossing]
            new_pass = events.passed & ~passed
            passed |= events.passed
            pass_step[new_pass] = policy_step + 1
            collision |= events.ring_collision
            wrong_order_count += events.wrong_order.long()
            wrong_direction_count += events.wrong_direction.long()
            course_penalty += events.penalty()
            missed |= events.expected_forward_crossing & ~events.passed & ~events.ring_collision
            current = events.next_gate_index
            ground |= state.position[:, 2] <= 0.03
            valid &= hover_train.state_is_valid(state)
            failed_prefix |= events.failed | ground | ~valid
            clean_first |= events.passed[:, 0] & ~failed_prefix
            # Conservatively omit passes in the same physics step as a failure.
            # Subsequent recovery passes remain in the raw diagnostic only.
            prefix_passes += (events.passed & ~failed_prefix[:, None]).sum(dim=1)
            radial = torch.sqrt(
                events.crossing_lateral.square() + events.crossing_vertical.square()
            )
            clean_radius = gate_config.inner_radius - gate_config.drone_radius
            centering = (1.0 - radial / clean_radius).clamp(0.0, 1.0)
            prefix_centering += torch.where(
                events.passed & ~failed_prefix[:, None], centering, 0.0
            ).sum(dim=1)

        maximum_tilt = torch.maximum(
            maximum_tilt,
            torch.linalg.vector_norm(state.euler[:, :2], dim=1),
        )

    final_gate_signed, _, _ = gate_coordinates(state.position, gates[-1])
    first = passed[:, 0]
    both = passed.all(dim=1)
    clean_course = (
        both
        & ~collision.any(dim=1)
        & ~(wrong_order_count > 0).any(dim=1)
        & ~(wrong_direction_count > 0).any(dim=1)
        & ~ground
        & valid
    )
    if compact_policies is not None:
        return compact_policy_metrics(
            first, clean_course, failed_prefix, prefix_passes, prefix_centering,
            cases.side, len(gates), compact_policies,
        )
    strict = (
        clean_course
        & (final_gate_signed >= 0.30)
        & ~missed.any(dim=1)
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
        negative_side: bool | None = None,
    ) -> float | None:
        mask = gate_crossed[:, gate_number]
        if negative_side is not None:
            mask = mask & ((cases.side < 0) if negative_side else (cases.side > 0))
        selected = values[:, gate_number][mask]
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
        per_gate[-1]["crossing_lateral_absolute_mean_by_side_metres"] = {
            side: finite_gate_mean(
                crossing_lateral, gate_number, absolute=True, negative_side=negative
            )
            for side, negative in (("negative", True), ("positive", False))
        }
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
        "privileged_axis_takeover_after_first_gate": diagnostic_teacher_axes,
        "clean_first_gate_pass_rate": float(clean_first.float().mean()),
        "clean_first_gate_negative_pass_rate": side_rate(clean_first, cases.side, True),
        "clean_first_gate_positive_pass_rate": side_rate(clean_first, cases.side, False),
        "first_gate_pass_rate": float(first.float().mean()),
        "first_gate_paired_pass_rate": float(pair_first.float().mean()),
        "first_gate_negative_offset_pass_rate": side_rate(first, cases.side, True),
        "first_gate_positive_offset_pass_rate": side_rate(first, cases.side, False),
        "both_gates_pass_rate": float(both.float().mean()),
        "all_gates_pass_rate": float(both.float().mean()),
        "course_rules": "ordered-directed-all-annuli-v1",
        "clean_course_success_rate": float(clean_course.float().mean()),
        "clean_course_negative_success_rate": side_rate(clean_course, cases.side, True),
        "clean_course_positive_success_rate": side_rate(clean_course, cases.side, False),
        "course_failure_rate": float(failed_prefix.float().mean()),
        "course_race_fitness": float(
            (prefix_passes + 5.0 * clean_course - 2.0 * failed_prefix
             + 0.2 * prefix_centering / len(gates)).mean()
        ),
        "clean_course_paired_success_rate": float(
            clean_course.reshape(-1, 2).all(dim=1).float().mean()
        ),
        "wrong_order_episode_rate": float((wrong_order_count > 0).any(dim=1).float().mean()),
        "wrong_direction_episode_rate": float(
            (wrong_direction_count > 0).any(dim=1).float().mean()
        ),
        "wrong_order_events_mean": float(wrong_order_count.sum(dim=1).float().mean()),
        "wrong_direction_events_mean": float(wrong_direction_count.sum(dim=1).float().mean()),
        "all_gate_ring_collision_rate": float(collision.any(dim=1).float().mean()),
        "course_event_penalty_mean": float(course_penalty.mean()),
        "gates_before_failure_mean": float(prefix_passes.mean()),
        "course_progress_score_mean": float(
            (prefix_passes + course_penalty - 2.0 * (ground | ~valid)).mean()
        ),
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
        "all_gates_negative_course_pass_rate": side_rate(both, cases.side, True),
        "all_gates_positive_course_pass_rate": side_rate(both, cases.side, False),
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
    ordered_nonnegative = (
        (args.lateral_step_min, args.lateral_step_max),
        (args.height_step_min, args.height_step_max),
    )
    if any(low < 0.0 or high < low for low, high in ordered_nonnegative):
        raise SystemExit("lateral and height step ranges must be nonnegative and ordered")
    if args.lateral_deviation_limit < 0.0:
        raise SystemExit("lateral deviation limit must be nonnegative")
    if args.height_min < 0.0 or args.height_max < args.height_min:
        raise SystemExit("height range must be nonnegative and ordered")
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
    if args.gate_back_pattern is not None:
        gate_config = replace(gate_config, back_pattern=args.gate_back_pattern)
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
        lateral_step_range=(args.lateral_step_min, args.lateral_step_max),
        lateral_deviation_limit=args.lateral_deviation_limit,
        height_step_range=(args.height_step_min, args.height_step_max),
        height_range=(args.height_min, args.height_max),
        yaw_jitter_degrees=args.yaw_jitter_degrees,
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
        "gate_back_pattern": gate_config.back_pattern,
        "gate_roles": [
            "current green",
            "next red",
            "later blue",
            "passed black",
        ],
        "geometry": {
            "independent_later_gate_yaw_jitter_degrees": args.yaw_jitter_degrees,
            "layout": args.layout,
            "first_gate_matches_single_gate_training_distribution": True,
            "second_gate_spacing_metres": [args.spacing_min, args.spacing_max],
            "gate_count": args.gates,
            "uniform_inter_gate_spacing_metres": [args.spacing_min, args.spacing_max],
            "lateral_step_absolute_metres": (
                [args.lateral_step_min, args.lateral_step_max]
                if args.layout == "variable"
                else None
            ),
            "lateral_deviation_limit_metres": (
                args.lateral_deviation_limit if args.layout == "variable" else None
            ),
            "height_step_absolute_metres": (
                [args.height_step_min, args.height_step_max]
                if args.layout == "variable"
                else None
            ),
            "gate_height_range_metres": (
                [args.height_min, args.height_max]
                if args.layout == "variable"
                else [1.10, 1.10]
            ),
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
