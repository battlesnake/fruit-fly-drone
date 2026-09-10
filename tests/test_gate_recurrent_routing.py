from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.audit_gate_throttle_routing import make_readout_spec
from scripts.search_gate_acceleration_path_es import make_path_spec
from scripts.search_gate_motor_interface_es import load_controller
from scripts.train_gate_recurrent_routing import (
    balanced_pair_split,
    controller_step_with_selected_magnitudes,
    make_recurrent_routing_spec,
    validation_selection,
)


def controller_and_spec():
    graph = Path("artifacts/gate-accel-v2/connectome.npz")
    controller, _, _, _, resolution = load_controller(
        graph,
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    path = make_path_spec(
        controller, graph, maximum_hops=4, floor_quantile=0.25, maximum_magnitude=8.0
    )
    readout = make_readout_spec(controller)
    return controller, make_recurrent_routing_spec(controller, path, readout), resolution


def test_recurrent_routing_topology_is_the_preregistered_native_mask() -> None:
    controller, spec, _ = controller_and_spec()
    assert len(spec.boundary_edges) == 88
    assert len(spec.internal_return_edges) == 19
    assert len(spec.readout_edges) == 37
    assert len(spec.selected_edges) == len(torch.unique(spec.selected_edges)) == 125
    assert not torch.isin(spec.boundary_edges, spec.readout_edges).any()
    assert torch.all(controller.edge_sign[spec.selected_edges].abs() == 1)


def test_selected_source_magnitudes_match_native_controller_step() -> None:
    controller, spec, resolution = controller_and_spec()
    generator = torch.Generator().manual_seed(61)
    batch = 4
    image = torch.rand(batch, resolution, resolution, generator=generator)
    attitude = torch.randn(batch, 2, generator=generator) * 0.1
    neural = torch.randn(batch, controller.n_nodes, generator=generator) * 0.2
    acceleration = torch.randn(batch, 3, generator=generator)
    acceleration[:, 2] += 9.81
    stick = torch.rand(batch, 4, generator=generator) * 2.0 - 1.0
    expected_motor, expected_state = controller(image, attitude, neural, acceleration, stick)
    actual_motor, actual_state = controller_step_with_selected_magnitudes(
        controller,
        image,
        attitude,
        neural,
        acceleration,
        stick,
        spec.selected_edges,
        controller.edge_magnitude[spec.selected_edges].detach(),
    )
    torch.testing.assert_close(actual_motor, expected_motor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0.0, atol=0.0)


def test_balanced_pair_split_matches_extreme_and_nonextreme_halves() -> None:
    fit, validation = balanced_pair_split()
    np.testing.assert_array_equal(fit, np.r_[0:24, 32:56])
    np.testing.assert_array_equal(validation, np.r_[24:32, 56:64])
    assert not np.intersect1d(fit, validation).size
    np.testing.assert_array_equal(np.sort(np.r_[fit, validation]), np.arange(64))


def validation(update: int, worst: float, aggregate: float, passed: bool, prefix: float):
    return {
        "update": update,
        "metrics": {
            "maximum_group_nrmse": worst,
            "aggregate_equal_group_nrmse": aggregate,
        },
        "pass_gate": {"passed": passed},
        "prefix_motor_rmse": prefix,
    }


def test_validation_selection_prefers_gate_pass_then_worst_group() -> None:
    items = [
        validation(100, 0.22, 0.15, False, 0.005),
        validation(200, 0.24, 0.18, True, 0.006),
        validation(300, 0.20, 0.14, True, 0.011),
    ]
    selected, rule = validation_selection(items, maximum_prefix_rmse=0.01)
    assert selected["update"] == 200
    assert "passing" in rule
