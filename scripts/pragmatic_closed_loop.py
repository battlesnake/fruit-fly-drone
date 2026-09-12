"""Training-only short physical rollouts from native sensory-history lessons."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import train_pragmatic_course_replay as replay

from flydrone.course_teacher import CoursePath
from flydrone.gate import render_annular_gates_rgb
from flydrone.gate_course import classify_course_step
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, QuadState, StickState


@dataclass
class PhysicalLesson:
    window: tuple
    sticks: StickState
    mass_scale: torch.Tensor
    source_progress: torch.Tensor
    record: dict


def ground_clearance_loss(heights):
    """Give useful gradients before contact; contact itself is a latched hard failure."""
    return 10.0 * ((0.30 - heights).clamp_min(0.0) / 0.30).square().mean()


def admissible_tracking_trial(metrics):
    return (
        math.isfinite(metrics["loss"])
        and metrics["failed_episodes"] == 0
        and metrics["ground_contact_episodes"] == 0
        and not metrics["crossed_gate"]
        and metrics["progress_floor_met"]
        and metrics["terminal_forward_speed_floor_met"]
        and metrics["altitude_preservation_met"]
        and metrics["nonroll_preservation_loss"] <= 0.01
    )


def stick_fields(sticks):
    return (sticks.joint_position, sticks.joint_velocity, sticks.position, sticks.velocity)


@torch.no_grad()
def reconstruct_sticks(initial, motors, starts, config):
    """Integrate recorded commands, excluding neural warmup, without moving waiting rows."""
    sticks = StickState(*(value.clone() for value in stick_fields(initial)))
    legs = ForelegStickPlant(config).to(motors.device)
    for time in range(int(starts.max())):
        for _ in range(2):
            _, advanced = legs(motors[time], sticks)
            sticks = StickState(
                *(
                    torch.where((time < starts)[:, None], new, old)
                    for new, old in zip(stick_fields(advanced), stick_fields(sticks), strict=True)
                )
            )
    return sticks


def select_physical_window(bank, phase, steps, rng, *, heldout_pairs=2):
    """Pick both sides from training courses, remaining safely before a gate crossing."""
    if bank.kind != "native":
        raise ValueError("physical lessons require native, not assisted, histories")
    count = bank.current.shape[1] - 2 * heldout_pairs
    rows, starts = [], []
    for side in (0, 1):
        choices = []
        for row in range(side, count, 2):
            valid = bank.active[:, row] & (bank.current[:, row] == phase)
            if len(valid) <= steps:
                continue
            valid = valid.unfold(0, steps + 1, 1).all(-1)
            endpoint = bank.states[0][steps:, row]
            gate = bank.gates[phase]
            normal = torch.stack((gate.yaw[row].cos(), gate.yaw[row].sin(), gate.yaw[row] * 0))
            ahead = -((endpoint - gate.center[row]) * normal).sum(-1)
            options = torch.nonzero(valid & (ahead > 0.15)).flatten()
            preferred = options[ahead[options] < 0.45]
            options = preferred if len(preferred) else options
            if len(options):
                choices.append((row, options.tolist()))
        if not choices:
            raise ValueError(f"no clean native training approach for gate {phase + 1}, side {side}")
        row, options = choices[int(rng.integers(len(choices)))]
        rows.append(row)
        starts.append(int(rng.choice(options)))
    return tuple(rows), starts


@torch.no_grad()
def make_physical_lesson(bank, initial_cases, rows, starts, steps, config, device):
    if bank.reference_outputs is None or bank.kind != "native":
        raise ValueError("need recorded native source commands for stick reconstruction")
    # Shared replay banks contain teacher roll labels; boundary replay needs the
    # actual recorded native commands on every axis instead.
    window = replay.prepare_window(
        replace(bank, target=bank.reference_outputs), rows, starts, steps + 1, device
    )
    indices = torch.arange(len(rows), device=device)
    times = window[4]
    initial = StickState(
        *(value[list(rows)].to(device) for value in stick_fields(initial_cases.sticks))
    )
    motors = bank.reference_outputs[: max(starts), list(rows)].to(device)
    sticks = reconstruct_sticks(initial, motors, times, config)
    progress = window[0][0][times + steps, indices, 0] - window[0][0][times, indices, 0]
    if not bool((progress > 0.05).all()):
        raise ValueError("source lesson must make positive forward progress")
    return PhysicalLesson(
        window,
        sticks,
        initial_cases.mass_scale[list(rows)].to(device),
        progress,
        dict(
            bank_seed=bank.seed,
            rows=list(rows),
            starts=starts,
            gates=(window[2][times, indices] + 1).tolist(),
            base_sides=["negative" if row % 2 == 0 else "positive" for row in rows],
            seconds=steps / 50,
            source_forward_progress_metres=progress.tolist(),
            source_sticks_reconstructed=True,
            physical_prefix="fixed native source history",
            learner_neural_prefix="current weights, replayed from zero",
        ),
    )


def physical_rollout_loss(
    controller,
    lesson,
    steps,
    camera,
    config,
    gate_config,
    *,
    fixed_neural=None,
):
    """All four native actions and the complete one-second plant retain gradients."""
    window = lesson.window
    fields, gates, roles, fixed_source_motors, starts = window
    device = starts.device
    rows = torch.arange(len(starts), device=device)
    state = QuadState(*(value[starts, rows].clone() for value in fields))
    sticks = StickState(*(value.clone() for value in stick_fields(lesson.sticks)))
    neural = (
        replay.replay_prefix_state(controller, window, camera, gate_config)
        if fixed_neural is None
        else fixed_neural
    ).detach()
    current = roles[starts, rows].clone()
    initial_role = current.clone()
    centers = torch.stack([gate.center for gate in gates], dim=1)[rows, initial_role]
    normals = torch.stack([gate.normal for gate in gates], dim=1)[rows, initial_role]
    crossed_plane = torch.zeros(len(rows), dtype=torch.bool, device=device)
    launch = fields[0][0]
    path = CoursePath.through_gates(launch, gates)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    start_x = state.position[:, 0].clone()
    failed = torch.zeros(len(rows), dtype=torch.bool, device=device)
    ground = torch.zeros_like(failed)
    lowest_height = state.position[:, 2].clone()
    lateral_errors, preservation, heights = [], [], []
    for frame in range(steps):
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        motor, neural = controller(image, state.euler[:, :2], neural)
        # Fixed nominal native labels, not a moving no-grad target on learner images.
        source_motor = fixed_source_motors[starts + frame, rows]
        preservation.append(
            ((motor[:, 1:] - source_motor[:, 1:]) / motor.new_tensor((0.02, 0.01, 0.025)))
            .square()
            .mean()
        )
        for _ in range(2):
            rc, sticks = legs(motor, sticks)
            previous = state.position
            state = quad(rc, state, lesson.mass_scale)
            lowest_height = torch.minimum(lowest_height, state.position[:, 2])
            heights.append(state.position[:, 2])
            with torch.no_grad():
                previous_side = ((previous.detach() - centers) * normals).sum(1) < 0
                current_side = ((state.position.detach() - centers) * normals).sum(1) < 0
                crossed_plane |= previous_side != current_side
                event = classify_course_step(
                    previous.detach(), state.position.detach(), gates, current, gate_config
                )
                current = event.next_gate_index
                ground |= state.position[:, 2] <= 0.03
                failed |= event.failed | ground | ~replay.hover_train.state_is_valid(state)
        target, _, _ = path.sample(state.position[:, 0])
        lateral_errors.append(state.position[:, 1] - target[:, 1])
    _, tangent, _ = path.sample(state.position[:, 0])
    velocity_error = state.velocity[:, 1] - tangent[:, 1] * state.velocity[:, 0]
    lateral_errors = torch.stack(lateral_errors)
    position_loss = (lateral_errors / 0.5).square().mean()
    velocity_loss = (velocity_error / 0.5).square().mean()
    progress = state.position[:, 0] - start_x
    progress_loss = ((0.95 * lesson.source_progress - progress).clamp_min(0) / 0.1).square().mean()
    preservation_loss = torch.stack(preservation).mean()
    tracking = position_loss + velocity_loss
    near_ground_loss = ground_clearance_loss(torch.stack(heights))
    failure_penalty = 25.0 * failed.float().mean()
    loss = tracking + progress_loss + preservation_loss + near_ground_loss + failure_penalty
    with torch.no_grad():
        source_end_velocity = fields[1][starts + steps, rows, 0]
        source_end_height = fields[0][starts + steps, rows, 2]
        source_heights = fields[0][
            starts[None] + torch.arange(steps + 1, device=device)[:, None], rows, 2
        ]
        speed_ok = state.velocity[:, 0] >= 0.95 * source_end_velocity
        altitude_ok = (state.position[:, 2] >= source_end_height - 0.05) & (
            lowest_height >= source_heights.min(0).values - 0.05
        )
        metrics = dict(
            loss=float(loss),
            tracking_loss=float(tracking),
            position_loss=float(position_loss),
            terminal_velocity_loss=float(velocity_loss),
            progress_loss=float(progress_loss),
            nonroll_preservation_loss=float(preservation_loss),
            near_ground_loss=float(near_ground_loss),
            failure_penalty=float(failure_penalty),
            ground_contact_episodes=int(ground.sum()),
            lateral_rmse_metres=float(lateral_errors.square().mean().sqrt()),
            lateral_rmse_by_side_metres=lateral_errors.square().mean(0).sqrt().tolist(),
            terminal_lateral_velocity_error_m_s=velocity_error.tolist(),
            progress_metres=progress.tolist(),
            progress_fraction_of_source=(progress / lesson.source_progress).tolist(),
            progress_floor_met=bool((progress >= 0.95 * lesson.source_progress).all()),
            terminal_forward_speed_m_s=state.velocity[:, 0].tolist(),
            terminal_forward_speed_floor_met=bool(speed_ok.all()),
            terminal_height_metres=state.position[:, 2].tolist(),
            lowest_height_metres=lowest_height.tolist(),
            altitude_preservation_met=bool(altitude_ok.all()),
            failed_episodes=int(failed.sum()),
            crossed_gate=bool(crossed_plane.any() | (current != initial_role).any()),
            crossed_expected_plane=bool(crossed_plane.any()),
        )
    return loss, metrics
