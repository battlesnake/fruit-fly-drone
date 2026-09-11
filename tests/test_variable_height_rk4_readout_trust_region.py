from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_rk4_readout_trust_region as audit  # noqa: E402


def _prior() -> dict:
    return {
        "gradient_objective": 1.5,
        "full_proposal_directional_derivative": -0.01,
        "baseline_fixed_objectives": [1.5, 1.5, 1.5],
    }


def test_protocol_uses_only_new_upward_scales() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["trust_region_scales_descending"] == [16.0, 8.0, 4.0, 2.0]
    assert protocol["scales_at_or_below_one_replayed"] is False
    assert protocol["actor_parameter_mask_solver_and_data_unchanged"] is True
    assert protocol["candidate_retained"] is False


def test_reproduction_requires_exact_tensor_hashes_and_counters() -> None:
    passing = audit.reproduction_decision(
        _prior(),
        baseline_values=[1.5, 1.5, 1.5],
        gradient_objective=1.5,
        directional_derivative=-0.01,
        pending_parameter_sha256=audit.EXPECTED_PENDING_PARAMETER_SHA256,
        optimizer_after_sha256=audit.EXPECTED_OPTIMIZER_AFTER_SHA256,
        optimizer_counters_before=[],
        optimizer_counters_after=[1.0, 1.0, 1.0],
    )
    wrong_hash = audit.reproduction_decision(
        _prior(),
        baseline_values=[1.5, 1.5, 1.5],
        gradient_objective=1.5,
        directional_derivative=-0.01,
        pending_parameter_sha256="wrong",
        optimizer_after_sha256=audit.EXPECTED_OPTIMIZER_AFTER_SHA256,
        optimizer_counters_before=[],
        optimizer_counters_after=[1.0, 1.0, 1.0],
    )

    assert passing["pass"] is True
    assert wrong_hash["pass"] is False


def test_reproduction_scalar_tolerance_is_gated() -> None:
    result = audit.reproduction_decision(
        _prior(),
        baseline_values=[1.5, 1.5, 1.5],
        gradient_objective=1.50003,
        directional_derivative=-0.01,
        pending_parameter_sha256=audit.EXPECTED_PENDING_PARAMETER_SHA256,
        optimizer_after_sha256=audit.EXPECTED_OPTIMIZER_AFTER_SHA256,
        optimizer_counters_before=[],
        optimizer_counters_after=[1.0, 1.0, 1.0],
    )

    assert result["pass"] is False
