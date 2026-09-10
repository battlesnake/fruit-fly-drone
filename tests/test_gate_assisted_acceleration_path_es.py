from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.search_gate_acceleration_path_es import make_path_spec
from scripts.search_gate_assisted_acceleration_path_es import (
    clustered_contrast_interval,
    edge_overlap_audit,
    nominate_finalists,
    performance_qualified,
    verify_assisted_motor_report,
)
from scripts.search_gate_motor_interface_es import load_controller


def summary(light: float, heavy: float, negative: float, positive: float):
    return {
        "success_rate": (light + heavy) / 2.0,
        "light_success_rate": light,
        "heavy_success_rate": heavy,
        "negative_lateral_success_rate": negative,
        "positive_lateral_success_rate": positive,
    }


def test_performance_gate_requires_light_floor_and_stratum_safety() -> None:
    baseline = summary(0.20, 0.50, 0.40, 0.30)
    qualified = summary(0.31, 0.48, 0.50, 0.41)
    no_light_gain = summary(0.29, 0.62, 0.55, 0.45)
    unsafe = summary(0.31, 0.44, 0.55, 0.45)
    kwargs = {
        "minimum_light_gain": 0.10,
        "minimum_floor_gain": 0.10,
        "maximum_drop": 0.05,
    }
    assert performance_qualified(baseline, qualified, **kwargs)
    assert not performance_qualified(baseline, no_light_gain, **kwargs)
    assert not performance_qualified(baseline, unsafe, **kwargs)


def test_validation_nominees_force_center_best_and_global_archive() -> None:
    def item(score: float):
        metrics = summary(0.20, 0.50, 0.40, 0.30)
        return {"objective": score, "development": metrics}

    archive = {
        "global": item(10.0),
        "center": item(1.0),
        "current_best": item(2.0),
        "fourth": item(9.0),
    }
    assert nominate_finalists(
        archive,
        required_keys=("center", "current_best", "global"),
        limit=3,
    ) == ["center", "current_best", "global"]


def test_clustered_contrast_interval_uses_geometry_pair_means() -> None:
    result = clustered_contrast_interval(torch.tensor([1.0, 1.0, -1.0, 1.0]))
    assert result["mean"] == pytest.approx(0.5)
    assert result["independent_geometry_clusters"] == 2
    assert result["episodes_per_cluster"] == 2


def test_four_hop_path_is_disjoint_from_preserved_steering_readout() -> None:
    graph = Path("artifacts/gate-accel-v2/connectome.npz")
    checkpoint = Path("artifacts/gate-motor-interface-es-v1/controller.pt")
    assisted = Path("artifacts/gate-assisted-motor-es-diagnostic-v1/report.json")
    controller, _, _, _, _ = load_controller(graph, checkpoint, torch.device("cpu"))
    path_spec = make_path_spec(
        controller,
        graph,
        maximum_hops=4,
        floor_quantile=0.25,
        maximum_magnitude=8.0,
    )
    report, steering = verify_assisted_motor_report(
        assisted,
        graph_path=graph,
        checkpoint_path=checkpoint,
    )
    overlap = edge_overlap_audit(controller, path_spec, report)
    assert len(path_spec.edges) == 282
    assert overlap["path_involved_node_count"] == 93
    assert overlap["steering_output_gain_overlap_count"] == 0
    assert overlap["throttle_output_gain_overlap_count"] == 36
    assert len(steering) == 24
    assert not bool(steering[18:].any())
