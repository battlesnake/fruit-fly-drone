from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_canonical_guard_band as guard  # noqa: E402
import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402


def _metrics(damping: float, common_endpoint: float, *, pitch: float = 0.045) -> dict:
    return {
        "component_nrmse": {
            "common": 1.0,
            "height": 1.0,
            "damping": damping,
            "interaction": 0.0,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": common_endpoint if step == 25 else 1.0,
                "height": 1.0,
                "damping": damping,
                "interaction": 0.0,
            }
            for step in (15, 20, 25)
        },
        "rpy_source_nrmse": {"roll": 0.01, "pitch": pitch, "yaw": 0.01},
        "endpoint_damping_nrmse": damping,
    }


def test_solver_guard_band_is_stricter_than_unchanged_acceptance() -> None:
    source = _metrics(1.45, 1.66)
    update8 = _metrics(1.376, 1.679)
    starting = _metrics(1.374, 1.684)

    solver = audit.correction_constraint_specs(
        source,
        update8,
        starting,
        endpoint_common_margin=guard.SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN,
    )
    acceptance = audit.correction_constraint_specs(
        source,
        update8,
        starting,
        endpoint_common_margin=guard.ACCEPTANCE_ENDPOINT_COMMON_INTERIOR_MARGIN,
    )
    solver_common = next(item for item in solver if item["name"] == "common.step_25")
    acceptance_common = next(item for item in acceptance if item["name"] == "common.step_25")

    assert solver_common["limit_mse"] == pytest.approx((1.66 + 0.0198) ** 2)
    assert acceptance_common["limit_mse"] == pytest.approx((1.66 + 0.0199) ** 2)
    assert solver_common["limit_mse"] < acceptance_common["limit_mse"]


def test_directional_finite_difference_accepts_matching_measurable_probe() -> None:
    gradient = {
        "edge_magnitude": torch.tensor([1.0]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    displacement = {
        "edge_magnitude": torch.tensor([0.1]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }

    result = audit.directional_finite_difference_report(
        gradient,
        displacement,
        starting_endpoint_damping_nrmse=1.0,
        probe_endpoint_damping_nrmse=math.sqrt(1.1),
        canonicalization_pass=True,
        parameter_bounds_pass=True,
        starting_restored_exactly=True,
    )

    assert result["pass"] is True
    assert result["relative_error"] == pytest.approx(0.0, abs=1.0e-6)


def test_directional_finite_difference_fails_nonfinite_or_unrestored_probe() -> None:
    gradient = {
        "edge_magnitude": torch.tensor([float("nan")]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    displacement = {name: torch.zeros_like(value) for name, value in gradient.items()}

    result = audit.directional_finite_difference_report(
        gradient,
        displacement,
        starting_endpoint_damping_nrmse=1.0,
        probe_endpoint_damping_nrmse=1.1,
        canonicalization_pass=True,
        parameter_bounds_pass=True,
        starting_restored_exactly=False,
    )

    assert result["pass"] is False
    assert result["all_values_finite"] is False
    assert result["starting_candidate_restored_exactly"] is False


def test_protocol_preserves_failure_and_does_not_relax_acceptance() -> None:
    protocol = guard.protocol_manifest()

    assert protocol["protocol_commit"] == "4a51b39"
    assert protocol["preserves_failed_audit"]["report_sha256"] == (
        guard.EXPECTED_FAILED_AUDIT_REPORT_SHA256
    )
    assert protocol["preserves_failed_audit"]["thresholds_relaxed"] is False
    assert protocol["correction"][
        "solver_endpoint_common_nrmse_maximum_source_plus"
    ] == pytest.approx(0.0198)
    assert protocol["correction"][
        "acceptance_endpoint_common_nrmse_maximum_source_plus"
    ] == pytest.approx(0.0199)
    assert protocol["correction"]["duplicate_multi_loss_row"].endswith("diagnostic only")
    assert protocol["correction"]["single_solve"] is True


def test_only_distinct_spec_mode_defers_canonical_linear_gate_to_scale_replay() -> None:
    shared = {
        "projection_pass": True,
        "canonicalization_pass": True,
        "solver_linearized_finite": True,
        "maximum_solver_linearized_violation": 1.1e-5,
        "acceptance_linearized_finite": True,
        "parameter_bounds_pass": True,
        "damping_row_control_pass": True,
    }

    assert (
        audit.correction_projection_preplay_pass(**shared, distinct_solver_acceptance_specs=False)
        is False
    )
    assert (
        audit.correction_projection_preplay_pass(**shared, distinct_solver_acceptance_specs=True)
        is True
    )


def test_wrapper_selects_authoritative_row_and_directional_control(monkeypatch) -> None:
    original_experiment = guard.audit.EXPERIMENT
    original_protocol_manifest = guard.audit.protocol_manifest
    monkeypatch.setattr(
        guard.audit,
        "main",
        lambda: (
            guard.audit.EXPERIMENT,
            guard.audit.SOLVER_ENDPOINT_COMMON_INTERIOR_MARGIN,
            guard.audit.ENDPOINT_COMMON_INTERIOR_MARGIN,
            guard.audit.USE_DISTINCT_SOLVER_ACCEPTANCE_SPECS,
            guard.audit.USE_AUTHORITATIVE_ENDPOINT_DAMPING_ROW,
            guard.audit.DIRECTIONAL_FINITE_DIFFERENCE_REQUIRED,
        ),
    )

    result = guard.main()

    assert result == (guard.EXPERIMENT, 0.0198, 0.0199, True, True, True)
    assert guard.audit.EXPERIMENT == original_experiment
    assert guard.audit.protocol_manifest is original_protocol_manifest
