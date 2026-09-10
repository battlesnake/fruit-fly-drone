from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_projection as fp64  # noqa: E402


def test_projected_gradient_kkt_accepts_bound_and_positive_coordinates() -> None:
    gram = np.eye(2, dtype=np.float64)
    violation = np.array([-1.0, 2.0], dtype=np.float64)
    dual = np.array([0.0, 2.0], dtype=np.float64)

    report = fp64._projected_gradient_report(gram, violation, dual)

    assert report["pass"] is True
    assert report["normalized_projected_gradient_residual"] == pytest.approx(0.0)
    assert report["normalized_complementarity_residual"] == pytest.approx(0.0)


def test_projected_gradient_kkt_rejects_negative_gradient_at_zero_bound() -> None:
    report = fp64._projected_gradient_report(
        np.eye(1, dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )

    assert report["pass"] is False
    assert report["normalized_projected_gradient_residual"] == pytest.approx(1.0)


def test_rank_report_detects_dependent_constraint_rows() -> None:
    gram = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    report = fp64._rank_report(gram)

    assert report["rank"] == 1
    assert report["nullity"] == 1
    assert report["dimension"] == 2


def test_fp64_dual_solver_reports_primary_and_independent_solutions() -> None:
    coefficients, report = fp64.solve_dual_fp64(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
    )

    assert coefficients == pytest.approx(torch.tensor([1.0], dtype=torch.float64))
    assert report["pass"] is True
    assert report["normalized_kkt"]["pass"] is True
    assert report["independent_slsqp"]["used_for_candidate"] is False


def test_fp64_projector_solves_toy_halfspace_in_double_precision(monkeypatch) -> None:
    monkeypatch.setattr(
        fp64.base,
        "PARAMETER_LEARNING_RATES",
        {"edge_magnitude": 1.0, "bias": 1.0, "raw_time_constant": 1.0},
    )
    displacement = {
        "edge_magnitude": torch.tensor([2.0], dtype=torch.float32),
        "bias": torch.tensor([0.0], dtype=torch.float32),
        "raw_time_constant": torch.tensor([0.0], dtype=torch.float32),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor([1.0], dtype=torch.float32),
            "bias": torch.tensor([0.0], dtype=torch.float32),
            "raw_time_constant": torch.tensor([0.0], dtype=torch.float32),
        }
    ]
    specs = [{"name": "toy", "current_mse": 0.0, "limit_mse": 1.0}]

    projected, report = fp64.fp64_project_inequality_displacement(displacement, rows, specs)

    assert projected["edge_magnitude"].dtype == torch.float64
    assert projected["edge_magnitude"] == pytest.approx(torch.tensor([1.0], dtype=torch.float64))
    assert report["pass"] is True
    assert report["maximum_linearized_violation_after"] <= 1.0e-6


def test_protocol_changes_only_projection_arithmetic_and_retains_original_gates() -> None:
    protocol = fp64.protocol_manifest()
    projection = protocol["fp64_projection"]
    post = protocol["post_projection_gates"]

    assert protocol["protocol_commit"] == "44448fe"
    assert protocol["gradients_generated_once"] is True
    assert protocol["variants_consume_clones"] is True
    assert projection["float64_before_products"] is True
    assert projection["original_unit_primal_violation_maximum"] == pytest.approx(1e-6)
    assert projection["normalized_projected_gradient_residual_maximum"] == pytest.approx(1e-8)
    assert post["ordinary_scales_descending"] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert post["nonlinear_repair"] is False
    assert protocol["candidate_retained"] is False
