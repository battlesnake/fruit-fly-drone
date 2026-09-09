from __future__ import annotations

import torch

from flydrone.hover import (
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
    motor_target_for_rc,
    render_target_band,
)


def test_foreleg_motor_to_measured_rc_jacobian_has_rank_four() -> None:
    plant = ForelegStickPlant()
    state = plant.initial_state(1, device=torch.device("cpu"), dtype=torch.float64)
    motor = torch.zeros(1, 4, dtype=torch.float64, requires_grad=True)

    jacobian = torch.autograd.functional.jacobian(lambda value: plant(value, state)[0], motor)
    local = jacobian[0, :, 0, :]

    assert torch.linalg.matrix_rank(local) == 4
    assert torch.all(torch.diag(local) > 0)


def test_zero_motor_drive_leaves_throttle_at_minimum() -> None:
    plant = ForelegStickPlant()
    state = plant.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    measured, next_state = plant(torch.zeros(2, 4), state)

    assert torch.equal(measured[:, 3], torch.zeros(2))
    assert torch.equal(next_state.position[:, 3], -torch.ones(2))


def test_foreleg_tarsus_positions_are_unit_length_forward_kinematics() -> None:
    plant = ForelegStickPlant()
    state = plant.initial_state(1, device=torch.device("cpu"), dtype=torch.float64)
    initial_tarsi = plant.tarsus_positions(state.joint_position)
    _, moved = plant(torch.ones(1, 4, dtype=torch.float64), state)
    moved_tarsi = plant.tarsus_positions(moved.joint_position)

    assert initial_tarsi.shape == (1, 2, 3)
    assert torch.allclose(
        torch.linalg.vector_norm(initial_tarsi, dim=-1), torch.ones(1, 2, dtype=torch.float64)
    )
    assert not torch.allclose(initial_tarsi, moved_tarsi)


def test_target_band_moves_down_image_as_camera_rises() -> None:
    quad = DifferentiableQuad()
    low = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    high_position = torch.tensor([[0.0, 0.0, 0.6]])
    high = quad.initial_state(
        1, device=torch.device("cpu"), dtype=torch.float32, position=high_position
    )
    target = torch.tensor([1.0])
    low_image = render_target_band(low, target, resolution=33)[0]
    high_image = render_target_band(high, target, resolution=33)[0]
    rows = torch.arange(33, dtype=torch.float32)
    low_row = (low_image.sum(dim=1) * rows).sum() / low_image.sum()
    high_row = (high_image.sum(dim=1) * rows).sum() / high_image.sum()

    assert high_row > low_row


def test_teacher_motor_target_can_raise_throttle_stick() -> None:
    config = HoverConfig()
    desired = torch.tensor([[0.0, 0.0, 0.0, 1.0 / config.thrust_to_weight]])
    motor = motor_target_for_rc(desired, config)

    assert motor[0, 3] > 0.0
    assert torch.equal(motor[0, :3], torch.zeros(3))


def test_zero_collective_cannot_create_fresh_roll_torque() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float64)
    rc = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)

    next_state = quad(rc, state)

    assert torch.equal(next_state.rates, torch.zeros_like(next_state.rates))


def test_mass_randomization_changes_thrust_acceleration() -> None:
    config = HoverConfig()
    quad = DifferentiableQuad(config)
    position = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    state = quad.initial_state(
        2, device=torch.device("cpu"), dtype=torch.float64, position=position
    )
    rc = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
    )

    next_state = quad(rc, state, torch.tensor([0.9, 1.1], dtype=torch.float64))

    assert next_state.velocity[0, 2] > next_state.velocity[1, 2]
