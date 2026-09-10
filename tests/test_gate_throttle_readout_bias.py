from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from scripts.audit_gate_throttle_readout_bias import bias_prediction_and_jacobian
from scripts.audit_gate_throttle_routing import (
    make_readout_spec,
    recurrent_drive,
    selected_edge_features,
)
from scripts.search_gate_motor_interface_es import load_controller


def random_arrays(samples: int, edges: int, motors: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(51)
    return {
        "edge_features": generator.normal(scale=0.2, size=(samples, edges)),
        "drive_without_selected": generator.normal(scale=0.3, size=(samples, motors)),
        "previous_motor_state": generator.normal(scale=0.2, size=(samples, motors)),
        "motor_alpha": np.full((samples, motors), 0.2),
        "source_motor": generator.normal(scale=0.05, size=samples),
        "target": generator.normal(scale=0.05, size=samples),
    }


def test_bias_readout_jacobian_matches_central_difference() -> None:
    controller, _, _, _, _ = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = make_readout_spec(controller)
    arrays = random_arrays(9, len(spec.edges), len(spec.motors))
    generator = np.random.default_rng(52)
    parameters = np.concatenate(
        (
            controller.edge_magnitude[spec.edges].detach().double().numpy(),
            generator.normal(scale=0.1, size=len(spec.motors)),
        )
    )
    direction = generator.normal(size=len(parameters))
    direction /= np.linalg.norm(direction)
    prediction, jacobian = bias_prediction_and_jacobian(parameters, arrays, spec)
    epsilon = 1.0e-6
    plus = bias_prediction_and_jacobian(parameters + epsilon * direction, arrays, spec)[0]
    minus = bias_prediction_and_jacobian(parameters - epsilon * direction, arrays, spec)[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert prediction.shape == (9,)
    np.testing.assert_allclose(finite_difference, jacobian @ direction, atol=1.0e-9, rtol=1.0e-6)


def test_bias_readout_matches_native_one_step_forward() -> None:
    controller, _, _, _, resolution = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = make_readout_spec(controller)
    candidate = copy.deepcopy(controller)
    generator = torch.Generator().manual_seed(53)
    batch = 5
    image = torch.rand(batch, resolution, resolution, generator=generator)
    attitude = torch.randn(batch, 2, generator=generator) * 0.1
    neural = torch.randn(batch, controller.n_nodes, generator=generator) * 0.2
    acceleration = torch.randn(batch, 3, generator=generator)
    acceleration[:, 2] += 9.81
    stick = torch.rand(batch, 4, generator=generator) * 2.0 - 1.0
    drive = recurrent_drive(controller, image, attitude, neural, acceleration, stick)
    source_motor, _ = controller(image, attitude, neural, acceleration, stick)
    selected = selected_edge_features(controller, neural, spec)
    source_magnitudes = controller.edge_magnitude[spec.edges]
    selected_current = torch.zeros(batch, len(spec.motors)).index_add(
        1, spec.edge_slots, selected * source_magnitudes
    )
    magnitude_delta = torch.linspace(0.01, 0.37, len(spec.edges))
    bias_delta = torch.linspace(-0.3, 0.3, len(spec.motors))
    with torch.no_grad():
        candidate.edge_magnitude[spec.edges] += magnitude_delta
        candidate.bias[spec.motors] += bias_delta
    actual_motor, _ = candidate(image, attitude, neural, acceleration, stick)
    arrays = {
        "edge_features": selected.detach().double().numpy(),
        "drive_without_selected": (drive[:, spec.motors] - selected_current)
        .detach()
        .double()
        .numpy(),
        "previous_motor_state": neural[:, spec.motors].detach().double().numpy(),
        "motor_alpha": (
            1.0 - torch.exp(-controller.neural_dt / controller.time_constant[spec.motors])
        )[None]
        .expand(batch, -1)
        .detach()
        .double()
        .numpy(),
        "source_motor": source_motor[:, 3].detach().double().numpy(),
        "target": np.zeros(batch),
    }
    parameters = np.concatenate(
        (
            candidate.edge_magnitude[spec.edges].detach().double().numpy(),
            bias_delta.double().numpy(),
        )
    )
    prediction, _ = bias_prediction_and_jacobian(parameters, arrays, spec)
    expected = (actual_motor[:, 3] - source_motor[:, 3]).detach().double().numpy()
    np.testing.assert_allclose(prediction, expected, atol=2.0e-7, rtol=2.0e-6)
