from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flydrone.hover import (
    ConnectomeController,
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
    rc = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)

    next_state = quad(rc, state, torch.tensor([0.9, 1.1], dtype=torch.float64))

    assert next_state.velocity[0, 2] > next_state.velocity[1, 2]


def test_completed_interval_specific_force_includes_mass_and_ground_constraint() -> None:
    config = HoverConfig(linear_drag=0.0)
    quad = DifferentiableQuad(config)
    position = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    state = quad.initial_state(
        2, device=torch.device("cpu"), dtype=torch.float32, position=position
    )
    state.actuator[:, 0] = 0.7
    rc = torch.tensor([[0.0, 0.0, 0.0, 0.7], [0.0, 0.0, 0.0, 0.7]])

    next_state = quad(rc, state, torch.tensor([0.92, 1.08]))
    grounded = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    grounded = quad(torch.zeros(1, 4), grounded)

    assert torch.allclose(next_state.specific_force[:, :2], torch.zeros(2, 2))
    assert next_state.specific_force[0, 2] > next_state.specific_force[1, 2]
    assert torch.allclose(grounded.specific_force, torch.tensor([[0.0, 0.0, 9.81]]))


def test_acceleration_interface_requires_sensor_and_injects_push_pull() -> None:
    controller = ConnectomeController(
        Path(__file__).resolve().parents[1] / "artifacts" / "gate-v1" / "connectome.npz"
    )
    controller.acceleration_nodes = controller.attitude_nodes[:2]
    controller.acceleration_channels = torch.tensor([0, 1])
    image = torch.zeros(2, 32, 32)
    attitude = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="requires body specific force"):
        controller.sensory_drive(image, attitude)
    force = torch.tensor([[0.0, 0.0, 1.25 * 9.81], [0.0, 0.0, 0.75 * 9.81]])
    drive = controller.sensory_drive(image, attitude, force)

    assert drive[0, controller.acceleration_nodes[0]] > 0.0
    assert drive[0, controller.acceleration_nodes[1]] == 0.0
    assert drive[1, controller.acceleration_nodes[0]] == 0.0
    assert drive[1, controller.acceleration_nodes[1]] > 0.0


def test_proprioception_interface_requires_sensor_and_injects_complementary_position() -> None:
    controller = ConnectomeController(
        Path(__file__).resolve().parents[1] / "artifacts" / "gate-v1" / "connectome.npz"
    )
    controller.proprioception_nodes = controller.attitude_nodes[:2]
    controller.proprioception_channels = torch.tensor([0, 1])
    image = torch.zeros(2, 32, 32)
    attitude = torch.zeros(2, 2)
    force = torch.tensor([[0.0, 0.0, 9.81], [0.0, 0.0, 9.81]])

    with pytest.raises(ValueError, match="requires foreleg stick position"):
        controller.sensory_drive(image, attitude, force)
    stick_position = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
    drive = controller.sensory_drive(image, attitude, force, stick_position)

    assert drive[0, controller.proprioception_nodes[0]] > 0.0
    assert drive[0, controller.proprioception_nodes[1]] == 0.0
    assert drive[1, controller.proprioception_nodes[0]] == 0.0
    assert drive[1, controller.proprioception_nodes[1]] > 0.0


def test_privileged_throttle_pool_bias_is_batch_specific() -> None:
    controller = ConnectomeController(
        Path(__file__).resolve().parents[1] / "artifacts" / "gate-v1" / "connectome.npz"
    )
    image = torch.zeros(2, 32, 32)
    attitude = torch.zeros(2, 2)
    state = controller.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    motor, _ = controller(
        image,
        attitude,
        state,
        privileged_throttle_pool_bias=torch.tensor([0.2, -0.2]),
    )

    assert motor[0, 3] > motor[1, 3]


def test_fixed_retinal_spectral_weights_map_rgb_without_a_learned_adapter() -> None:
    controller = ConnectomeController(
        Path(__file__).resolve().parents[1] / "artifacts" / "gate-v1" / "connectome.npz"
    )
    weights = torch.zeros(len(controller.visual_nodes), 3)
    weights[0::3, 0] = 1.0
    weights[1::3, 1] = 1.0
    weights[2::3, 2] = 1.0
    controller.visual_channel_weights = weights
    image = torch.zeros(1, 3, 19, 23)
    image[:, 0] = 0.2
    image[:, 1] = 0.5
    image[:, 2] = 0.8

    retina = controller.sample_retina(image)

    assert torch.allclose(retina[:, 0::3], torch.full_like(retina[:, 0::3], 0.2))
    assert torch.allclose(retina[:, 1::3], torch.full_like(retina[:, 1::3], 0.5))
    assert torch.allclose(retina[:, 2::3], torch.full_like(retina[:, 2::3], 0.8))


def _controller_with_unequal_motor_pools() -> ConnectomeController:
    controller = ConnectomeController(
        Path(__file__).resolve().parents[1] / "artifacts" / "gate-v1" / "connectome.npz"
    )
    controller.pool_indices = torch.arange(20)
    controller.pool_offsets = torch.tensor([0, 1, 4, 6, 10, 11, 14, 16, 20])
    controller.refresh_motor_pool_ranges()
    return controller


def _uncached_motor_drive(controller, state):
    """Original eight-pool reduction, including checkpoint-buffer scalar reads."""
    activity = torch.sigmoid(state)
    means = []
    for pool in range(8):
        begin = int(controller.pool_offsets[pool].item())
        end = int(controller.pool_offsets[pool + 1].item())
        means.append(activity[:, controller.pool_indices[begin:end]].mean(dim=1))
    pools = torch.stack(means, dim=-1)
    return torch.stack(tuple(pools[:, i] - pools[:, i + 1] for i in range(0, 8, 2)), dim=-1)


def test_motor_pool_cache_preserves_outputs_and_gradients_without_scalar_reads(monkeypatch):
    controller = _controller_with_unequal_motor_pools()
    state = torch.randn(2, controller.n_nodes, generator=torch.Generator().manual_seed(913))
    state.requires_grad_()
    expected = _uncached_motor_drive(controller, state)
    expected_gradient = torch.autograd.grad(expected.square().sum(), state)[0]

    def forbidden_scalar_read(*args, **kwargs):
        raise AssertionError("fixed motor wiring must not read device scalars each frame")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", forbidden_scalar_read)
        actual = controller.motor_drive(state)
        actual_gradient = torch.autograd.grad(actual.square().sum(), state)[0]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("nested", [False, True])
def test_motor_pool_cache_refreshes_after_direct_or_parent_checkpoint_load(nested):
    controller = _controller_with_unequal_motor_pools()
    model = torch.nn.ModuleDict({"actor": controller}) if nested else controller
    payload = model.state_dict()
    key = "actor.pool_offsets" if nested else "pool_offsets"
    payload[key] = payload[key].clone()
    payload[key][7] += 1  # move one neuron between the throttle antagonist pools
    original_ranges = controller._motor_pool_ranges

    model.load_state_dict(payload)

    assert controller._motor_pool_ranges != original_ranges
    assert controller._motor_pool_ranges[6:] == ((14, 17), (17, 20))
    assert list(model.state_dict()) == list(payload)  # no checkpoint-format change
    state = torch.randn(2, controller.n_nodes, generator=torch.Generator().manual_seed(914))
    torch.testing.assert_close(
        controller.motor_drive(state), _uncached_motor_drive(controller, state)
    )


def test_explicit_motor_pool_edit_can_refresh_cached_wiring():
    controller = _controller_with_unequal_motor_pools()
    controller.pool_offsets[1] += 1
    controller.refresh_motor_pool_ranges()

    assert controller._motor_pool_ranges[:2] == ((0, 2), (2, 4))
    state = torch.randn(2, controller.n_nodes, generator=torch.Generator().manual_seed(915))
    torch.testing.assert_close(
        controller.motor_drive(state), _uncached_motor_drive(controller, state)
    )
