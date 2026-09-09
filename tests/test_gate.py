from __future__ import annotations

import math
from pathlib import Path

import torch

from flydrone.gate import (
    AnnularGate,
    GateConfig,
    classify_gate_crossing,
    initial_gate_geometry,
    render_annular_gate,
    sample_annular_gates,
    teacher_gate_rc,
)
from flydrone.hover import ConnectomeController, DifferentiableQuad, QuadState

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
    previous = torch.tensor(
        [[1.9, 0.0, 1.0], [1.9, 0.55, 1.0], [1.9, 1.0, 1.0]]
    )
    current = previous + torch.tensor((0.2, 0.0, 0.0))

    passed, collision, missed = classify_gate_crossing(previous, current, gate, config)

    assert passed.tolist() == [True, False, False]
    assert collision.tolist() == [False, True, False]
    assert missed.tolist() == [False, False, True]


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


def test_fixed_l1_receptive_fields_see_every_strict_launch_gate() -> None:
    torch.manual_seed(19)
    batch = 128
    quad = DifferentiableQuad()
    state = quad.initial_state(batch, device=torch.device("cpu"), dtype=torch.float32)
    gate = sample_annular_gates(
        batch, device=torch.device("cpu"), dtype=torch.float32, strict=True
    )
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
    )

    assert torch.equal(teacher_gate_rc(state, gate), teacher_gate_rc(perturbed, gate))
