from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first_bound_aware as bounded  # noqa: E402


def test_bound_aware_projection_fixes_crossing_edge_at_actual_boundary() -> None:
    displacement = {
        "edge_magnitude": torch.tensor([-1.0]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor([-1.0]),
            "bias": torch.tensor([1.0]),
            "raw_time_constant": torch.tensor([0.0]),
        }
    ]
    specs = [
        {
            "name": "toy",
            "kind": "factorial",
            "current_mse": 0.0,
            "limit_mse": 0.4,
        }
    ]

    projected, report = bounded.bound_aware_project_inequality_displacement(
        displacement, rows, specs, torch.tensor([0.5])
    )

    assert report["pass"] is True
    assert report["fixed_edges"] == 1
    assert projected["edge_magnitude"] == pytest.approx(torch.tensor([-0.5]))
    # With J_edge * delta_edge = +0.5 and a -0.4 residual, free bias must supply -0.1.
    assert projected["bias"] == pytest.approx(torch.tensor([-0.1]), abs=2.0e-5)


def test_bound_aware_projection_reports_round_limit_as_solver_failure() -> None:
    displacement = {
        "edge_magnitude": torch.tensor([-1.0]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor([0.0]),
            "bias": torch.tensor([1.0]),
            "raw_time_constant": torch.tensor([0.0]),
        }
    ]
    specs = [
        {
            "name": "toy",
            "kind": "factorial",
            "current_mse": 0.0,
            "limit_mse": 1.0,
        }
    ]

    _, report = bounded.bound_aware_project_inequality_displacement(
        displacement,
        rows,
        specs,
        torch.tensor([0.5]),
        maximum_rounds=1,
    )

    assert report["pass"] is False
    assert report["reason"] == "active-set round limit reached"


def test_protocol_changes_only_bound_handling_contract() -> None:
    protocol = bounded.protocol_manifest()

    assert protocol["experiment"] == bounded.EXPERIMENT
    assert protocol["constraint_projection"]["maximum_active_set_rounds"] == 8
    assert protocol["constraint_projection"]["active_edges_released"] is False
    assert protocol["backtrack_scales_descending"] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert protocol["qualification"]["total_scenes"] == 64
