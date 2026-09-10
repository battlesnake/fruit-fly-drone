from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_gate_throttle_routing import (
    balanced_probe_split,
    exact_residual,
    fit_constant,
    make_readout_spec,
    metric_summary,
    model_passes,
    recurrent_drive,
    selected_edge_features,
    window_index,
)
from scripts.search_gate_acceleration_path_es import make_path_spec
from scripts.search_gate_motor_interface_es import load_controller


def test_routing_dimensions_and_exact_zero_change_parity() -> None:
    graph = Path("artifacts/gate-accel-v2/connectome.npz")
    controller, _, _, _, resolution = load_controller(
        graph,
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    path = make_path_spec(
        controller,
        graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    path_nodes = torch.unique(
        torch.cat((controller.edge_pre[path.edges], controller.edge_post[path.edges]))
    )
    spec = make_readout_spec(controller)
    assert (len(path.edges), len(path_nodes)) == (282, 93)
    assert (len(spec.edges), len(spec.return_sources), len(spec.motors)) == (37, 19, 7)

    generator = torch.Generator().manual_seed(42)
    batch = 4
    image = torch.rand(batch, resolution, resolution, generator=generator)
    roll_pitch = 0.1 * torch.randn(batch, 2, generator=generator)
    neural = 0.2 * torch.randn(batch, controller.n_nodes, generator=generator)
    acceleration = torch.randn(batch, 3, generator=generator)
    acceleration[:, 2] += 9.81
    stick = torch.rand(batch, 4, generator=generator) * 2.0 - 1.0
    drive = recurrent_drive(controller, image, roll_pitch, neural, acceleration, stick)
    source_motor, _ = controller(image, roll_pitch, neural, acceleration, stick)
    edge_features = selected_edge_features(controller, neural, spec)
    baseline = controller.edge_magnitude[spec.edges]
    selected_current = torch.zeros(batch, len(spec.motors)).index_add(
        1, spec.edge_slots, edge_features * baseline
    )
    features = {
        "edge_features": edge_features,
        "drive_without_selected": drive[:, spec.motors] - selected_current,
        "previous_motor_state": neural[:, spec.motors],
        "motor_alpha": (
            1.0 - torch.exp(-controller.neural_dt / controller.time_constant[spec.motors])
        )[None].expand(batch, -1),
        "source_motor": source_motor[:, 3],
    }
    assert float(exact_residual(baseline, features, spec).detach().abs().max()) <= 1.0e-6


def test_routing_windows_are_disjoint_and_stop_at_one_point_five_seconds() -> None:
    assert window_index(49, dt=0.01) is None
    assert window_index(50, dt=0.01) == 0
    assert window_index(74, dt=0.01) == 0
    assert window_index(75, dt=0.01) == 1
    assert window_index(100, dt=0.01) == 2
    assert window_index(149, dt=0.01) == 2
    assert window_index(150, dt=0.01) is None


def test_probe_split_balances_extreme_and_nonextreme_geometry_pairs() -> None:
    pair = np.repeat(np.arange(64), 100)
    fit, validation = balanced_probe_split(pair, total_pairs=64, fit_pairs=48)
    fit_ids = np.unique(pair[fit])
    validation_ids = np.unique(pair[validation])
    assert len(fit_ids[fit_ids < 32]) == len(fit_ids[fit_ids >= 32]) == 24
    assert (
        len(validation_ids[validation_ids < 32]) == len(validation_ids[validation_ids >= 32]) == 8
    )
    assert not np.intersect1d(fit_ids, validation_ids).size


def test_constant_fit_and_pass_gate_equal_weight_all_six_groups() -> None:
    samples = {
        "target": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 6.0]),
        "window": torch.tensor([0, 0, 1, 1, 2, 2]),
        "mass_half": torch.tensor([0, 1, 0, 1, 0, 1]),
    }
    assert fit_constant(samples) == pytest.approx(1.0)
    scales = {group: 1.0 for group in range(6)}
    metrics = metric_summary(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),
        samples,
        scales,
        constant_aggregate_rmse=1.0,
    )
    assert metrics["maximum_group_nrmse"] == pytest.approx(5.8)
    assert not model_passes(
        metrics,
        maximum_group_nrmse=0.25,
        minimum_constant_improvement=0.50,
    )["passed"]
