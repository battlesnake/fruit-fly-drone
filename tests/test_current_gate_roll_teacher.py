"""Small physical/sign checks for the diagnostic, not evidence of learned flight."""

import math

import torch

from flydrone.course_teacher import current_gate_roll_motor
from flydrone.gate import AnnularGate
from flydrone.hover import DifferentiableQuad, ForelegStickPlant, HoverConfig, motor_target_for_rc


def scene():
    state = DifferentiableQuad().initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:, 2] = 1.1
    state.velocity[:, 0] = 0.55
    gate = AnnularGate(torch.tensor([[2.0, -0.4, 1.1], [2.0, 0.4, 1.1]]), torch.zeros(2))
    return state, gate


def target(state, gate, active=None):
    if active is None:
        active = torch.ones(2, dtype=torch.bool)
    return current_gate_roll_motor(state, gate, HoverConfig(), active=active)


def test_mirrored_gate_signs_and_global_heading_invariance():
    state, gate = scene()
    original = target(state, gate)
    assert original[0] > 0 > original[1]
    assert torch.allclose(original[0], -original[1])
    # Rotate the entire world by 90 degrees: body-relative task is unchanged.
    for field in (state.position, state.velocity, gate.center):
        x, y = field[:, 0].clone(), field[:, 1].clone()
        field[:, 0], field[:, 1] = -y, x
    state.euler[:, 2] += torch.pi / 2
    assert torch.allclose(target(state, gate), original, atol=1e-6)


def test_completed_or_behind_target_does_not_keep_chasing_gate():
    state, gate = scene()
    assert torch.equal(target(state, gate, torch.zeros(2, dtype=torch.bool)), torch.zeros(2))
    gate.center[:, 0] = -0.01
    assert torch.equal(target(state, gate), torch.zeros(2))


def test_near_plane_and_large_velocity_produce_finite_bounded_commands():
    state, gate = scene()
    gate.center[:, 0] = torch.tensor([0.0, 1e-9])
    gate.center[:, 1] *= 1e5
    state.velocity[:, 1] = torch.tensor([-100.0, 100.0])
    motor = target(state, gate)
    assert torch.isfinite(motor).all()
    state.euler[:, 0] = torch.tensor([100.0, -100.0])
    saturated = target(state, gate)
    assert torch.all(motor.abs() <= saturated.abs())


def test_behind_gate_brakes_lateral_drift_through_real_foreleg_and_quad_dynamics():
    state, gate = scene()
    config = HoverConfig()
    quad, legs = DifferentiableQuad(config), ForelegStickPlant(config)
    sticks = legs.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    sticks.position[:, 3] = 2.0 / config.thrust_to_weight - 1.0
    sticks.joint_position = torch.asin(sticks.position * math.sin(config.foreleg_joint_limit))
    state.actuator[:, 0] = 1.0 / config.thrust_to_weight
    state.velocity[:, 0] = 0.0
    state.velocity[:, 1] = torch.tensor([-0.4, 0.4])
    gate.center[:, 0] = -1.0
    rc = torch.zeros(2, 4)
    rc[:, 3] = 1.0 / config.thrust_to_weight
    hold = motor_target_for_rc(rc, config)
    for _ in range(100):
        motor = hold.clone()
        motor[:, 0] = target(state, gate)
        for _ in range(2):
            measured_rc, sticks = legs(motor, sticks)
            state = quad(measured_rc, state)
    # Drag compensation may command force along velocity. Net physical deceleration,
    # not a naive motor-sign assertion, is the meaningful test of braking here.
    assert torch.all(state.velocity[:, 1].abs() < 0.1)
    assert torch.all(state.position[:, 2] > 0.9)
    assert torch.allclose(state.velocity[0, 1], -state.velocity[1, 1], atol=1e-6)
