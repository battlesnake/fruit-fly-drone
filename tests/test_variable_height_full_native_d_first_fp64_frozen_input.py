from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_frozen_input as frozen  # noqa: E402


def _displacement(value: float) -> dict[str, torch.Tensor]:
    return {
        "edge_magnitude": torch.tensor([value], dtype=torch.float64),
        "bias": torch.tensor([value], dtype=torch.float64),
        "raw_time_constant": torch.tensor([value], dtype=torch.float64),
    }


def test_primary_repeat_agreement_accepts_identical_primal_results() -> None:
    report = frozen.primary_repeat_agreement(
        [_displacement(1.0), _displacement(1.0), _displacement(1.0)],
        [-2.0, -2.0, -2.0],
    )

    assert report["pass"] is True
    assert all(pair["pass"] for pair in report["pairs"])


def test_primary_repeat_agreement_rejects_primal_or_objective_drift() -> None:
    report = frozen.primary_repeat_agreement(
        [_displacement(1.0), _displacement(1.01), _displacement(1.0)],
        [-2.0, -2.0, -2.1],
    )

    assert report["pass"] is False
    assert any(not pair["pass"] for pair in report["pairs"])


def test_complete_projection_objective_includes_all_parameter_families() -> None:
    raw = _displacement(2.0)
    projected = _displacement(1.0)

    objective = frozen.complete_projection_primal_objective(projected, raw)

    expected = 0.5 * sum(
        1.0 / frozen.base.PARAMETER_LEARNING_RATES[name] ** 2
        for name in frozen.joint.PARAMETER_FAMILIES
    )
    assert objective == pytest.approx(expected)


def test_archive_hashes_cover_every_frozen_numerical_input() -> None:
    tensor = torch.tensor([1.0])
    archive = {
        "current_parameters": {"edge_magnitude": tensor},
        "raw_displacement": {"edge_magnitude": tensor},
        "constraint_rows": [{"edge_magnitude": tensor}],
        "constraint_specs": [{"name": "row", "current_mse": 1.0, "limit_mse": 2.0}],
        "raw_damping_gradient": {"edge_magnitude": tensor},
    }

    hashes = frozen.archive_tensor_hashes(archive)

    assert set(hashes) == {
        "current_parameters",
        "raw_displacement",
        "constraint_rows",
        "constraint_specs",
        "raw_damping_gradient",
    }
    assert all(len(value) == 64 for value in hashes.values())


def test_run_control_requires_primary_and_independent_primal_checks() -> None:
    passing = {
        "pass": True,
        "rounds": [
            {"free_projection": {"independent_nnls": {"pass": True}}},
            {"free_projection": {"independent_nnls": {"pass": True}}},
        ],
    }
    failing = {
        **passing,
        "rounds": [
            {"free_projection": {"independent_nnls": {"pass": False}}},
        ],
    }

    assert frozen.fp64_run_control(passing)["pass"] is True
    assert frozen.fp64_run_control(failing)["pass"] is False


def test_protocol_qualifies_numerics_without_accepting_a_candidate() -> None:
    protocol = frozen.protocol_manifest()

    assert protocol["protocol_commit"] == "cb7c7aa"
    assert protocol["fp64_repeats_from_empty_active_set"] == 3
    assert protocol["tensor_generation"]["actual_tensors_persisted_before_solver"] is True
    assert protocol["primary_repeat_agreement"][
        "learning_rate_scaled_primal_relative_distance_maximum"
    ] == pytest.approx(1e-10)
    assert protocol["independent_solver"][
        "gram_induced_primal_relative_distance_maximum"
    ] == pytest.approx(1e-6)
    assert protocol["post_projection_gates"]["ordinary_nonlinear_acceptance_required"] is False
    assert protocol["candidate_retained"] is False
