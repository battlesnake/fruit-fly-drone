"""Training-only physical gradients along uninterrupted native course flights."""

from __future__ import annotations

import math

import torch
import train_pragmatic_course_replay as replay
from pragmatic_closed_loop import ground_clearance_loss, stick_fields

from flydrone.course_teacher import CoursePath
from flydrone.gate import render_annular_gates_rgb
from flydrone.gate_course import classify_course_step
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, QuadState, StickState


def whole_flight_trial_admissible(candidate, baseline, *, mode="continuous"):
    """A lower smooth loss cannot buy away clean flight performance on this bank."""
    if mode not in ("continuous", "flight-first"):
        raise ValueError("unknown whole-flight acceptance mode")
    improves = candidate["continuous_objective"] < baseline["continuous_objective"] - 1e-6
    if mode == "flight-first":
        outcome = (candidate["clean_completions"], candidate["clean_prefix_gates"])
        previous = (baseline["clean_completions"], baseline["clean_prefix_gates"])
        improves = outcome > previous or (outcome == previous and improves)
    return (
        math.isfinite(candidate["objective"])
        and improves
        and candidate["clean_completions"] >= baseline["clean_completions"]
        and candidate["clean_prefix_gates"] >= baseline["clean_prefix_gates"]
        and all(
            new >= old
            for new, old in zip(
                candidate["clean_first_by_side"], baseline["clean_first_by_side"], strict=True
            )
        )
        and candidate["ground_contacts"] <= baseline["ground_contacts"]
        and candidate["invalid_episodes"] <= baseline["invalid_episodes"]
        and all(
            candidate[key] <= baseline[key]
            for key in ("failed_episodes", "ring_contacts", "wrong_order", "wrong_direction")
        )
    )


def fly_course(
    controller,
    cases,
    gates,
    *,
    seconds,
    camera,
    config,
    gate_config,
    chunk_steps=50,
    backward=False,
    gradient_scale=1.0,
    reference_motors=None,
    reference_active=None,
    record_trace=False,
    record_chunks=False,
):
    """Fixed weights throughout; truncation detaches gradients, never numerical state.

    No teacher, replay states, privileged actor input, or phase-dependent actor path.
    Failed/completed flights keep running to score the full tail, just like evaluation.
    The caller alone performs optimizer updates, after all flight chunks are complete.
    """
    steps = round(seconds * 50)
    if steps < 1 or chunk_steps < 1:
        raise ValueError("positive flight duration and chunk length required")
    device = cases.side.device
    count = len(cases.side)
    if reference_motors is not None and reference_motors.shape != (steps, count, 4):
        raise ValueError("fixed executed-motor reference must match the complete flight")
    if reference_active is not None and (
        reference_active.shape != (steps, count) or reference_active.dtype != torch.bool
    ):
        raise ValueError("fixed tracking mask must match the complete flight")
    state = QuadState(*(value.clone() for value in cases.state.as_tuple()))
    sticks = StickState(*(value.clone() for value in stick_fields(cases.sticks)))
    neural = controller.initial_state(count, device=device, dtype=state.position.dtype)
    current = torch.zeros(count, dtype=torch.long, device=device)
    failed = torch.zeros(count, dtype=torch.bool, device=device)
    ground, invalid, first = (torch.zeros_like(failed) for _ in range(3))
    ring, wrong_order, wrong_direction = (torch.zeros_like(failed) for _ in range(3))
    passed = torch.zeros(count, len(gates), dtype=torch.bool, device=device)
    prefix = torch.zeros(count, device=device)
    failure_steps = torch.full_like(current, -1)
    phase_frames = torch.zeros(len(gates), dtype=torch.long, device=device)
    phase_frames_by_episode = torch.zeros(count, len(gates), dtype=torch.long, device=device)
    phase_indices = torch.arange(len(gates), device=device)
    phase_tracking = torch.zeros(len(gates), device=device)
    minimum_height = state.position[:, 2].clone()
    path = CoursePath.through_gates(state.position, gates)
    quad, legs = DifferentiableQuad(config).to(device), ForelegStickPlant(config).to(device)
    with torch.no_grad():
        image = render_annular_gates_rgb(
            state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
        )
        for _ in range(10):
            _, neural = controller(image, state.euler[:, :2], neural)
    totals = torch.zeros(3, device=device)
    motors, active_frames, chunks = [], [], []
    chunk_loss = None
    with torch.set_grad_enabled(backward):
        for frame in range(steps):
            active = ~failed & (current < len(gates))
            phase = current.clone()
            image = render_annular_gates_rgb(
                state, gates, current_gate_index=current, camera=camera, gate_config=gate_config
            )
            motor, neural = controller(image, state.euler[:, :2], neural)
            if record_trace:
                motors.append(motor.detach().cpu())
                active_frames.append(active.cpu())
            preservation = motor.sum() * 0.0
            if reference_motors is not None:
                preservation = (
                    (
                        (motor[:, 1:] - reference_motors[frame, :, 1:])
                        / motor.new_tensor((0.02, 0.01, 0.025))
                    )
                    .square()
                    .mean()
                )
            clearance = motor.sum() * 0.0
            for _ in range(2):
                rc, sticks = legs(motor, sticks)
                previous = state.position
                state = quad(rc, state, cases.mass_scale)
                clearance = clearance + 0.5 * ground_clearance_loss(state.position[:, 2])
                with torch.no_grad():
                    event = classify_course_step(
                        previous.detach(), state.position.detach(), gates, current, gate_config
                    )
                    current = event.next_gate_index
                    passed |= event.passed
                    ring |= event.ring_collision.any(1)
                    wrong_order |= event.wrong_order.any(1)
                    wrong_direction |= event.wrong_direction.any(1)
                    ground |= state.position[:, 2] <= 0.03
                    invalid |= ~replay.hover_train.state_is_valid(state)
                    new_failure = event.failed | ground | invalid
                    failure_steps[(failure_steps < 0) & new_failure] = frame + 1
                    failed |= new_failure
                    first |= event.passed[:, 0] & ~failed
                    prefix += (event.passed & ~failed[:, None]).sum(1)
                    minimum_height = torch.minimum(minimum_height, state.position[:, 2])
            target, tangent, _ = path.sample(state.position[:, 0])
            lateral = ((state.position[:, 1] - target[:, 1]) / 0.5).square()
            velocity = (
                (state.velocity[:, 1] - tangent[:, 1] * state.velocity[:, 0]) / 0.5
            ).square()
            tracking_each = lateral + 0.25 * velocity
            # A candidate failure must not erase expensive nominal tracking samples.
            tracking_mask = active if reference_active is None else reference_active[frame]
            tracking = (tracking_each * tracking_mask).mean()
            terms = torch.stack((tracking, 0.05 * preservation, clearance))
            frame_loss = terms.sum() / steps
            chunk_loss = frame_loss if chunk_loss is None else chunk_loss + frame_loss
            with torch.no_grad():
                totals += terms.detach() / steps
                selected = active[:, None] & (phase[:, None] == phase_indices[None])
                phase_frames_by_episode += selected.long()
                phase_frames += selected.sum(0)
                phase_tracking += (tracking_each.detach()[:, None] * selected).sum(0)
            if (frame + 1) % chunk_steps == 0 or frame + 1 == steps:
                if backward:
                    (gradient_scale * chunk_loss).backward()
                if record_chunks:
                    chunks.append(
                        dict(
                            step=frame + 1,
                            position=state.position.detach().cpu(),
                            velocity=state.velocity.detach().cpu(),
                            role=current.detach().cpu().clone(),
                            failed=failed.detach().cpu().clone(),
                        )
                    )
                chunk_loss = None
                state, sticks, neural = state.detach(), sticks.detach(), neural.detach()
    clean = passed.all(1) & ~failed
    discrete = 25.0 * failed.float().mean() - 2.0 * prefix.mean() - 5.0 * clean.float().mean()
    metrics = dict(
        objective=float(totals.sum() + discrete),
        continuous_objective=float(totals.sum()),
        tracking_loss=float(totals[0]),
        nonroll_preservation_loss=float(totals[1]),
        ground_clearance_loss=float(totals[2]),
        discrete_objective=float(discrete),
        episodes=count,
        clean_completions=int(clean.sum()),
        clean_prefix_gates=int(prefix.sum()),
        clean_first_passes=int(first.sum()),
        clean_first_by_side=[int(first[cases.side < 0].sum()), int(first[cases.side > 0].sum())],
        clean_completions_by_side=[
            int(clean[cases.side < 0].sum()),
            int(clean[cases.side > 0].sum()),
        ],
        failed_episodes=int(failed.sum()),
        ground_contacts=int(ground.sum()),
        invalid_episodes=int(invalid.sum()),
        ring_contacts=int(ring.sum()),
        wrong_order=int(wrong_order.sum()),
        wrong_direction=int(wrong_direction.sum()),
        phase_frames=phase_frames.tolist(),
        phase_frames_by_side=[
            phase_frames_by_episode[cases.side < 0].sum(0).tolist(),
            phase_frames_by_episode[cases.side > 0].sum(0).tolist(),
        ],
        clean_prefix_by_side=[int(prefix[cases.side < 0].sum()), int(prefix[cases.side > 0].sum())],
        phase_tracking_mean=(phase_tracking / phase_frames.clamp_min(1)).tolist(),
        failure_steps=failure_steps.tolist(),
        minimum_height_metres=minimum_height.tolist(),
        clean_episode_indices=torch.nonzero(clean).flatten().tolist(),
        final_positions=state.position.tolist(),
    )
    trace = (
        dict(motors=torch.stack(motors), active=torch.stack(active_frames))
        if record_trace
        else None
    )
    return metrics, trace, chunks
