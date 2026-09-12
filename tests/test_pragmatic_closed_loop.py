from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pragmatic_closed_loop as physical  # noqa: E402

from flydrone.gate import AnnularGate, GateConfig  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    StickState,
    motor_target_for_rc,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402


def test_stick_reconstruction_uses_two_ticks_and_excludes_selected_frame():
    config = HoverConfig()
    legs = ForelegStickPlant(config)
    initial = legs.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    motors = torch.full((4, 2, 4), 0.1)
    motors[2:] = -0.7  # selected frame 2 and later must not be applied
    actual = physical.reconstruct_sticks(initial, motors, torch.tensor([0, 2]), config)
    expected = StickState(*(value[1:].clone() for value in physical.stick_fields(initial)))
    for frame in range(2):
        for _ in range(2):
            _, expected = legs(motors[frame, 1:], expected)
    for got, unchanged, advanced in zip(
        physical.stick_fields(actual),
        physical.stick_fields(initial),
        physical.stick_fields(expected),
        strict=True,
    ):
        assert torch.equal(got[:1], unchanged[:1])
        assert torch.equal(got[1:], advanced)


def test_ground_penalty_is_zero_at_safe_height_and_pushes_up_before_contact():
    assert physical.ground_clearance_loss(torch.tensor([0.3, 1.1])) == 0
    low = torch.tensor([0.15], requires_grad=True)
    loss = physical.ground_clearance_loss(low)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert low.grad.item() < 0  # descent increases clearance
    assert physical.ground_clearance_loss(torch.tensor([0.0])) == 10


def test_contact_or_stalling_cannot_be_accepted_despite_a_better_tracking_loss():
    metrics = dict(
        loss=0.0,
        failed_episodes=0,
        ground_contact_episodes=0,
        crossed_gate=False,
        progress_floor_met=True,
        terminal_forward_speed_floor_met=True,
        altitude_preservation_met=True,
        nonroll_preservation_loss=0,
    )
    assert physical.admissible_tracking_trial(metrics)
    for key, bad in (
        ("failed_episodes", 1),
        ("ground_contact_episodes", 1),
        ("terminal_forward_speed_floor_met", False),
        ("altitude_preservation_met", False),
        ("progress_floor_met", False),
        ("crossed_gate", True),
        ("loss", float("nan")),
    ):
        assert not physical.admissible_tracking_trial(dict(metrics, **{key: bad}))


def tiny_lesson(steps):
    config = HoverConfig()
    position = torch.zeros(steps + 1, 2, 3)
    position[:, :, 0] = 0.02 * torch.arange(steps + 1)[:, None]
    position[:, :, 2] = 1.1
    velocity = torch.zeros_like(position)
    velocity[:, :, 0] = 1.0
    actuator = torch.zeros(steps + 1, 2, 4)
    actuator[:, :, 0] = 1 / config.thrust_to_weight
    fields = (
        position,
        velocity,
        torch.zeros_like(position),
        torch.zeros_like(position),
        actuator,
        torch.zeros_like(position),
    )
    gates = tuple(
        AnnularGate(torch.tensor([[5.0 + i, 0.3, 1.1]]).expand(2, -1), torch.zeros(2))
        for i in range(5)
    )
    rc = torch.tensor([[0.0, 0.0, 0.0, 1 / config.thrust_to_weight]]).expand(2, -1)
    commands = motor_target_for_rc(rc, config).expand(steps + 1, -1, -1).clone()
    window = (
        fields,
        gates,
        torch.zeros(steps + 1, 2, dtype=torch.long),
        commands,
        torch.zeros(2, dtype=torch.long),
    )
    positions = torch.cat((rc[:, :3], 2 * rc[:, 3:] - 1), dim=1)
    joints = torch.asin(positions * math.sin(config.foreleg_joint_limit))
    sticks = StickState(joints, torch.zeros_like(joints), positions, torch.zeros_like(positions))
    return physical.PhysicalLesson(
        window, sticks, torch.ones(2), torch.full((2,), 0.02 * steps), {}
    )


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roll = torch.nn.Parameter(torch.tensor(0.01))
        self.register_buffer(
            "nominal",
            motor_target_for_rc(
                torch.tensor([[0.0, 0.0, 0.0, 1 / HoverConfig().thrust_to_weight]])
            )[0],
        )

    def initial_state(self, count, **kwargs):
        return torch.zeros(count, 1, **kwargs)

    def forward(self, image, attitude, neural):
        roll = self.roll.expand(len(neural))
        return torch.cat((roll[:, None], self.nominal[1:].expand(len(roll), -1)), dim=1), neural


def tiny_render(state, *args, **kwargs):
    return state.position[:, :1, None, None].expand(-1, 3, 2, 2)


def test_physical_boundary_uses_recorded_commands_not_teacher_roll_labels():
    lesson = tiny_lesson(8)
    fields, gates, roles, source, _ = lesson.window
    teacher = source.clone()
    teacher[:, :, 0] = 0.5
    bank = physical.replay.ReplayBank(
        fields,
        gates,
        roles,
        torch.ones_like(roles, dtype=torch.bool),
        teacher,
        "native",
        1,
        reference_outputs=source,
    )
    cases = SimpleNamespace(sticks=lesson.sticks, mass_scale=lesson.mass_scale)
    actual = physical.make_physical_lesson(
        bank, cases, (0, 1), [0, 0], 8, HoverConfig(), torch.device("cpu")
    )
    assert torch.equal(actual.window[3], source)
    assert bool((bank.target[:, :, 0] == 0.5).all())


def test_tracking_gradient_reaches_motor_parameter_through_real_legs_and_aircraft(monkeypatch):
    monkeypatch.setattr(physical, "render_annular_gates_rgb", tiny_render)
    actor = TinyActor()
    loss, metrics = physical.physical_rollout_loss(
        actor,
        tiny_lesson(8),
        8,
        CameraSpec(),
        HoverConfig(),
        GateConfig(),
        fixed_neural=torch.zeros(2, 1),
    )
    loss.backward()
    assert torch.isfinite(actor.roll.grad) and actor.roll.grad.abs() > 1e-9
    assert metrics["ground_contact_episodes"] == 0


def test_ground_touch_is_latched_even_if_aircraft_recovers_before_end(monkeypatch):
    class TouchAndRecover(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            self.calls += 1
            position = state.position.clone()
            position[:, 0] += 0.01
            position[:, 2] = 0.02 if self.calls == 1 else 1.1
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(physical, "DifferentiableQuad", TouchAndRecover)
    monkeypatch.setattr(physical, "render_annular_gates_rgb", tiny_render)
    _, metrics = physical.physical_rollout_loss(
        TinyActor(),
        tiny_lesson(2),
        2,
        CameraSpec(),
        HoverConfig(),
        GateConfig(),
        fixed_neural=torch.zeros(2, 1),
    )
    assert metrics["terminal_height_metres"] == pytest.approx([1.1, 1.1])
    assert metrics["ground_contact_episodes"] == 2
    assert metrics["failed_episodes"] == 2
    assert metrics["failure_penalty"] == 25.0
    assert not physical.admissible_tracking_trial(metrics)


def test_exterior_plane_touch_and_return_is_rejected_without_gate_role_change(monkeypatch):
    class MissAndReturn(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            self.calls += 1
            position = state.position.clone()
            position[:, 0] = 5.0 if self.calls == 1 else 4.8
            position[:, 1] = 3.0  # outside the annulus: legal, but not a smooth pre-crossing lesson
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(physical, "DifferentiableQuad", MissAndReturn)
    monkeypatch.setattr(physical, "render_annular_gates_rgb", tiny_render)
    lesson = tiny_lesson(2)
    lesson.window[0][0][0, :, 0] = 4.8
    lesson.window[0][0][0, :, 1] = 3.0
    _, metrics = physical.physical_rollout_loss(
        TinyActor(),
        lesson,
        2,
        CameraSpec(),
        HoverConfig(),
        GateConfig(),
        fixed_neural=torch.zeros(2, 1),
    )
    assert metrics["failed_episodes"] == 0  # do not change the actual course's exterior-plane rules
    assert metrics["crossed_expected_plane"]
    assert metrics["crossed_gate"]
    assert not physical.admissible_tracking_trial(metrics)
