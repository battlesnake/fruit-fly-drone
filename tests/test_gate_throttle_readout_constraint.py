from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.audit_gate_throttle_readout_constraint import (
    exact_prediction_and_jacobian,
    feature_arrays,
    projected_gradient,
)
from scripts.audit_gate_throttle_routing import make_readout_spec
from scripts.search_gate_motor_interface_es import load_controller


def test_exact_fp64_readout_jacobian_matches_central_difference() -> None:
    controller, _, _, _, _ = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = make_readout_spec(controller)
    generator = np.random.default_rng(5)
    samples = 7
    arrays = {
        "edge_features": generator.normal(scale=0.2, size=(samples, len(spec.edges))),
        "drive_without_selected": generator.normal(scale=0.3, size=(samples, len(spec.motors))),
        "previous_motor_state": generator.normal(scale=0.2, size=(samples, len(spec.motors))),
        "motor_alpha": np.full((samples, len(spec.motors)), 0.2),
        "source_motor": generator.normal(scale=0.05, size=samples),
        "target": generator.normal(scale=0.05, size=samples),
    }
    weights = controller.edge_magnitude[spec.edges].detach().double().numpy()
    direction = generator.normal(size=len(weights))
    direction /= np.linalg.norm(direction)
    prediction, jacobian = exact_prediction_and_jacobian(weights, arrays, spec)
    epsilon = 1.0e-6
    plus = exact_prediction_and_jacobian(weights + epsilon * direction, arrays, spec)[0]
    minus = exact_prediction_and_jacobian(weights - epsilon * direction, arrays, spec)[0]
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert prediction.shape == (samples,)
    np.testing.assert_allclose(finite_difference, jacobian @ direction, atol=1.0e-9, rtol=1.0e-6)


def test_relaxed_arm_removes_original_edge_sign_from_recorded_features() -> None:
    controller, _, _, _, _ = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = make_readout_spec(controller)
    recorded = torch.arange(2 * len(spec.edges), dtype=torch.float32).reshape(2, -1) / 100.0
    samples = {
        "edge_features": recorded,
        "drive_without_selected": torch.zeros(2, len(spec.motors)),
        "previous_motor_state": torch.zeros(2, len(spec.motors)),
        "motor_alpha": torch.ones(2, len(spec.motors)),
        "source_motor": torch.zeros(2),
        "target": torch.zeros(2),
    }
    fixed = feature_arrays(samples, controller, spec, arm="fixed_sign")
    relaxed = feature_arrays(samples, controller, spec, arm="relaxed_sign")
    sign = controller.edge_sign[spec.edges].numpy()
    np.testing.assert_allclose(fixed["edge_features"], recorded.numpy())
    np.testing.assert_allclose(relaxed["edge_features"], recorded.numpy() * sign)


def test_projected_gradient_zeros_only_outward_bound_components() -> None:
    weights = np.asarray([0.0, 0.0, 8.0, 8.0, 4.0])
    residual = np.asarray([1.0])
    jacobian = np.asarray([[1.0, -1.0, -1.0, 1.0, 2.0]])
    result = projected_gradient(
        weights,
        residual,
        jacobian,
        np.zeros(5),
        np.full(5, 8.0),
    )
    assert result["maximum_absolute_projected_gradient"] == 2.0
