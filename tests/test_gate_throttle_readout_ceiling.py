from __future__ import annotations

from pathlib import Path

import torch

from scripts.audit_gate_throttle_readout_ceiling import flight_progress
from scripts.search_gate_assisted_motor_es import assisted_controller_step
from scripts.search_gate_motor_interface_es import load_controller, motor_interface_spec


def summary(light: float, heavy: float, negative: float, positive: float):
    return {
        "success_by_stratum": {
            "lower_mass": light,
            "higher_mass": heavy,
            "negative_lateral_offset": negative,
            "positive_lateral_offset": positive,
        }
    }


def test_ceiling_flight_gate_requires_simultaneous_light_floor_and_safety() -> None:
    baseline = summary(0.0, 0.80, 0.38, 0.42)
    qualified = summary(0.12, 0.77, 0.50, 0.53)
    light_only = summary(0.12, 0.82, 0.08, 0.48)
    unsafe = summary(0.12, 0.74, 0.52, 0.55)
    kwargs = {
        "minimum_light_gain": 0.10,
        "minimum_floor_gain": 0.10,
        "maximum_drop": 0.05,
    }
    assert flight_progress(baseline, qualified, **kwargs)
    assert not flight_progress(baseline, light_only, **kwargs)
    assert not flight_progress(baseline, unsafe, **kwargs)


def test_native_assisted_step_preserves_compiled_magnitudes_above_eight() -> None:
    controller, _, _, _, resolution = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = motor_interface_spec(
        controller,
        bias_scale=0.01,
        log_gain_scale=0.05,
        maximum_bias_delta=0.15,
        maximum_gain_ratio=3.0,
    )
    with torch.no_grad():
        controller.edge_magnitude[spec.gain_edges[0]] = 16.0
    generator = torch.Generator().manual_seed(77)
    batch = 3
    image = torch.rand(batch, resolution, resolution, generator=generator)
    attitude = torch.randn(batch, 2, generator=generator) * 0.1
    neural = torch.randn(batch, controller.n_nodes, generator=generator) * 0.2
    acceleration = torch.randn(batch, 3, generator=generator)
    acceleration[:, 2] += 9.81
    stick = torch.rand(batch, 4, generator=generator) * 2.0 - 1.0
    expected_motor, expected_state = controller(image, attitude, neural, acceleration, stick)
    actual_motor, actual_state = assisted_controller_step(
        controller,
        image,
        attitude,
        neural,
        acceleration,
        stick,
        torch.zeros(batch, len(spec.labels)),
        spec,
        native_controller_forward=True,
    )
    torch.testing.assert_close(actual_motor, expected_motor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0.0, atol=0.0)
