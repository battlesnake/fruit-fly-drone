from __future__ import annotations

import pytest
import torch

from flydrone.course_teacher import CoursePath, CourseTeacherConfig, course_teacher_motor
from flydrone.gate import AnnularGate
from flydrone.hover import DifferentiableQuad, HoverConfig


def paired_path():
    launch = torch.tensor([[0.0, 0.0, 1.1]]).expand(2, -1)
    gates = (
        AnnularGate(torch.tensor([[3.0, 0.0, 1.1]]).expand(2, -1), torch.zeros(2)),
        AnnularGate(torch.tensor([[4.2, -0.4, 1.2], [4.2, 0.4, 1.2]]), torch.zeros(2)),
    )
    return CoursePath.through_gates(launch, gates), gates


def test_reference_interpolates_each_gate_with_shared_tangent():
    path, gates = paired_path()
    for gate in gates:
        position, derivative, _ = path.sample(gate.center[:, 0])
        assert torch.allclose(position, gate.center, atol=1.0e-6)
        _, before, _ = path.sample(gate.center[:, 0] - 1.0e-5)
        _, after, _ = path.sample(gate.center[:, 0] + 1.0e-5)
        assert torch.allclose(before, after, atol=1.0e-4)
        assert torch.allclose(derivative[:, 0], torch.ones(2))


def test_next_gate_changes_exit_velocity_and_pre_crossing_actions():
    path, gates = paired_path()
    _, tangent, _ = path.sample(gates[0].center[:, 0])
    assert tangent[0, 1] < 0 < tangent[1, 1]
    state = DifferentiableQuad().initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:] = torch.tensor([2.6, 0.0, 1.1])
    state.velocity[:, 0] = 0.55
    motor = course_teacher_motor(state, path, HoverConfig())
    assert torch.isfinite(motor).all()
    # Same present pose and current gate: only the future visible gate differs.
    assert not torch.allclose(motor[0], motor[1], atol=1.0e-5)
    assert torch.allclose(motor[0, 0], -motor[1, 0], atol=1.0e-5)
    assert torch.allclose(motor[0, 2], -motor[1, 2], atol=1.0e-5)


def test_teacher_rejects_course_outside_its_monotonic_x_scope():
    launch = torch.tensor([[0.0, 0.0, 1.0]])
    gate = AnnularGate(torch.tensor([[-1.0, 0.0, 1.0]]), torch.zeros(1))
    with pytest.raises(ValueError, match="increasing-X"):
        CoursePath.through_gates(launch, (gate,))


def test_fixed_heading_teacher_keeps_yaw_feedback_and_next_gate_anticipation():
    path, _ = paired_path()
    state = DifferentiableQuad().initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:] = torch.tensor([2.6, 0.0, 1.1])
    state.velocity[:, 0] = 0.55
    config = CourseTeacherConfig(heading_mode="world-x")
    centered = course_teacher_motor(state, path, HoverConfig(), config)
    assert centered[0, 0] * centered[1, 0] < 0
    assert torch.allclose(centered[:, 2], torch.zeros(2), atol=1.0e-6)
    state.euler[:, 2] = torch.tensor([-0.1, 0.1])
    recovery = course_teacher_motor(state, path, HoverConfig(), config)
    assert recovery[0, 2] > 0 > recovery[1, 2]


def test_teacher_rejects_unknown_heading_mode():
    with pytest.raises(ValueError, match="heading_mode"):
        CourseTeacherConfig(heading_mode="constant-zero-output")


def test_rate_damped_teacher_has_no_absolute_heading_target_but_opposes_yaw_rate():
    path, _ = paired_path()
    state = DifferentiableQuad().initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:] = torch.tensor([2.6, 0.0, 1.1])
    state.euler[:, 2] = torch.tensor([-0.4, 0.4])
    config = CourseTeacherConfig(heading_mode="rate-damped")
    motor = course_teacher_motor(state, path, HoverConfig(), config)
    assert torch.allclose(motor[:, 2], torch.zeros(2), atol=1e-6)
    state.rates[:, 2] = torch.tensor([-0.2, 0.2])
    motor = course_teacher_motor(state, path, HoverConfig(), config)
    assert motor[0, 2] > 0 > motor[1, 2]


def test_rate_damped_teacher_rotates_horizontal_force_into_current_body_heading():
    launch = torch.tensor([[0.0, 0.0, 1.1]]).expand(2, -1)
    gate = AnnularGate(torch.tensor([[3.0, 0.0, 1.1]]).expand(2, -1), torch.zeros(2))
    path = CoursePath.through_gates(launch, (gate,))
    state = DifferentiableQuad().initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:] = launch
    state.euler[1, 2] = torch.pi / 2
    motor = course_teacher_motor(
        state, path, HoverConfig(), CourseTeacherConfig(heading_mode="rate-damped")
    )
    # Both need +world-X acceleration: pitch at yaw 0, roll at yaw +90 degrees.
    assert motor[0, 1] > 0 and abs(float(motor[0, 0])) < 1e-6
    assert motor[1, 0] > 0 and abs(float(motor[1, 1])) < 1e-6
