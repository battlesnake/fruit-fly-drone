from __future__ import annotations

import math
from pathlib import Path

import torch

from flydrone.gate import (
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    crossing_coordinates,
    initial_gate_geometry,
    render_annular_gate,
    render_annular_gate_rgb,
    render_annular_gates_rgb,
    sample_annular_gates,
    teacher_gate_rc,
)
from flydrone.hover import ConnectomeController, DifferentiableQuad, QuadState
from flydrone.visual_hover import CameraSpec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_strict_gate_samples_are_off_axis_and_oblique() -> None:
    config = GateConfig()
    torch.manual_seed(4)
    gate = sample_annular_gates(
        128, device=torch.device("cpu"), dtype=torch.float64, config=config, strict=True
    )
    centre_offset, obliquity = initial_gate_geometry(gate)

    assert torch.all(centre_offset > math.radians(5.0))
    assert torch.all(obliquity >= math.radians(config.minimum_obliquity_degrees))


def test_gate_crossing_requires_whole_drone_inside_aperture() -> None:
    config = GateConfig(inner_radius=0.6, outer_radius=0.75, drone_radius=0.1)
    center = torch.tensor([[2.0, 0.0, 1.0]]).expand(3, -1)
    gate = AnnularGate(center=center, yaw=torch.zeros(3))
    previous = torch.tensor([[1.9, 0.0, 1.0], [1.9, 0.55, 1.0], [1.9, 1.0, 1.0]])
    current = previous + torch.tensor((0.2, 0.0, 0.0))

    passed, collision, missed = classify_gate_crossing(previous, current, gate, config)

    assert passed.tolist() == [True, False, False]
    assert collision.tolist() == [False, True, False]
    assert missed.tolist() == [False, False, True]


def test_crossing_coordinates_preserve_signed_lateral_and_vertical_error() -> None:
    gate = AnnularGate(
        center=torch.tensor([[2.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        yaw=torch.zeros(2),
    )
    previous = torch.tensor([[1.8, -0.3, 1.2], [2.1, 0.4, 0.7]])
    current = torch.tensor([[2.2, -0.3, 1.2], [2.3, 0.4, 0.7]])

    directed, lateral, vertical = crossing_coordinates(previous, current, gate)

    assert directed.tolist() == [True, False]
    assert torch.allclose(lateral, torch.tensor([-0.3, 0.4]))
    assert torch.allclose(vertical, torch.tensor([0.2, -0.3]))


def test_oblique_annulus_renders_off_centre_with_dark_aperture() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    gate = AnnularGate(
        center=torch.tensor([[4.5, 0.7, 1.0]]),
        yaw=torch.tensor([math.radians(22.0)]),
    )
    image = render_annular_gate(state, gate, resolution=65)[0]
    bright = (image > 0.6).float()
    columns = torch.arange(65, dtype=image.dtype)
    bright_column = (bright.sum(dim=0) * columns).sum() / bright.sum()

    assert image.max() > 0.9
    assert image[32, 32] < 0.5
    assert bright_column < 32  # positive world/body Y is left in the rendered image
    assert not torch.allclose(image, torch.flip(image, dims=(1,)))


def test_rgb_gate_camera_contains_coloured_gate_and_textured_floor() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    state = QuadState(
        position=torch.tensor([[0.0, 0.0, 1.1]]),
        velocity=state.velocity,
        euler=state.euler,
        rates=state.rates,
        actuator=state.actuator,
        specific_force=state.specific_force,
    )
    gate = AnnularGate(center=torch.tensor([[3.0, 0.7, 1.1]]), yaw=torch.tensor([0.2]))
    camera = CameraSpec(width=80, height=50, horizontal_fov_degrees=125.0)

    image = render_annular_gate_rgb(state, gate, camera=camera)[0]

    assert image.shape == (3, 50, 80)
    assert image.min() >= 0.0
    assert image.max() <= 1.0
    assert image[1].max() > 0.7  # green gate
    assert image[:, -1].std() > 0.002  # textured grey floor
    assert image[:, 0].max() < 0.01  # black background above the horizon


def test_multi_gate_rgb_roles_recolour_after_a_pass() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    state.position[:, 2] = 1.1
    gates = (
        AnnularGate(center=torch.tensor([[3.0, -0.8, 1.1]]), yaw=torch.zeros(1)),
        AnnularGate(center=torch.tensor([[4.0, 0.8, 1.1]]), yaw=torch.zeros(1)),
    )
    camera = CameraSpec(width=100, height=60, horizontal_fov_degrees=125.0)

    before = render_annular_gates_rgb(
        state,
        gates,
        current_gate_index=torch.tensor([0]),
        camera=camera,
    )[0]
    after = render_annular_gates_rgb(
        state,
        gates,
        current_gate_index=torch.tensor([1]),
        camera=camera,
    )[0]

    assert before[0].max() > 0.7  # red next-gate role is visible
    assert after[0].max() < 0.4  # passed gate is dark; current gate is green
    assert after[1].max() > 0.7
    assert not torch.allclose(before, after)


def test_fixed_l1_receptive_fields_see_every_strict_launch_gate() -> None:
    torch.manual_seed(19)
    batch = 128
    quad = DifferentiableQuad()
    state = quad.initial_state(batch, device=torch.device("cpu"), dtype=torch.float32)
    gate = sample_annular_gates(batch, device=torch.device("cpu"), dtype=torch.float32, strict=True)
    image = render_annular_gate(state, gate, resolution=32)
    controller = ConnectomeController(
        REPO_ROOT / "artifacts" / "hover-v1" / "connectome.npz",
        retinal_receptive_field=3,
    )
    response = controller.sample_retina(image).amax(dim=1)

    assert torch.all(response > 0.02)


def test_distillation_teacher_does_not_use_hidden_velocity_or_rates() -> None:
    quad = DifferentiableQuad()
    state = quad.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    gate = AnnularGate(
        center=torch.tensor([[4.6, -0.8, 1.1], [4.6, 0.8, 1.1]]),
        yaw=torch.tensor([math.radians(-30.0), math.radians(30.0)]),
    )
    perturbed = QuadState(
        position=state.position,
        velocity=torch.tensor([[3.0, -2.0, 1.0], [-1.0, 2.0, -3.0]]),
        euler=state.euler,
        rates=torch.tensor([[5.0, -4.0, 3.0], [-2.0, 3.0, -4.0]]),
        actuator=state.actuator,
        specific_force=state.specific_force,
    )

    assert torch.equal(teacher_gate_rc(state, gate), teacher_gate_rc(perturbed, gate))
