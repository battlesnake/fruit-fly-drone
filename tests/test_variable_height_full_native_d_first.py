from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_d_first as fit  # noqa: E402


def _metrics(
    damping: float = 1.0,
    *,
    component: float = 1.0,
    rpy: float = 0.0,
    sign: float = 0.0,
    gain: float = 0.0,
) -> dict:
    return {
        "joint_normalized_mse": 1.0,
        "component_nrmse": {
            "common": component,
            "height": component,
            "damping": damping,
            "interaction": 0.0,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": component,
                "height": component,
                "damping": damping,
                "interaction": 0.0,
            }
            for step in (15, 20, 25)
        },
        "rpy_source_nrmse": {"roll": rpy, "pitch": rpy, "yaw": rpy},
        "endpoint_damping_nrmse": damping,
        "endpoint_damping_correct_sign_fraction": sign,
        "endpoint_damping_teacher_aligned_gain": gain,
        "motor_output_max_absolute": 0.7,
        "all_attitude_cache_states_valid": True,
    }


def test_constraint_specs_use_cumulative_source_limits_and_activate_rpy_late() -> None:
    source = _metrics(component=1.0)
    current = _metrics(component=1.01, rpy=0.039)

    specs = fit.constraint_specs(source, current)

    assert len(specs) == 8
    aggregate = next(item for item in specs if item["name"] == "common.aggregate")
    assert aggregate["current_mse"] == pytest.approx(1.01**2)
    assert aggregate["limit_mse"] == pytest.approx(1.02**2)

    current["rpy_source_nrmse"]["pitch"] = 0.04
    specs = fit.constraint_specs(source, current)
    assert [item["name"] for item in specs if item["kind"] == "attitude"] == ["rpy.pitch"]


def test_learning_rate_scaled_projection_satisfies_toy_halfspace() -> None:
    displacement = {
        "edge_magnitude": torch.tensor([2.0]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor([1.0]),
            "bias": torch.tensor([0.0]),
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

    projected, report = fit.project_inequality_displacement(displacement, rows, specs)

    assert report["pass"] is True
    assert projected["edge_magnitude"] == pytest.approx(torch.tensor([1.0]), abs=1.0e-5)
    assert report["maximum_linearized_violation_after"] <= fit.LINEAR_CONSTRAINT_TOLERANCE


def test_candidate_decision_uses_current_damping_but_original_preservation() -> None:
    source = _metrics(damping=1.5, component=1.0)
    current = _metrics(damping=1.0, component=1.019)
    candidate = _metrics(damping=0.998, component=1.021)

    result = fit.candidate_decision(
        source, current, candidate, damping_directional_derivative=-1.0
    )

    assert result["actual_endpoint_damping_nrmse_improvement"] == pytest.approx(0.002)
    assert result["pass"] is False
    assert any("original source" in reason for reason in result["reasons"])


def test_update_50_and_terminal_gates_are_independent() -> None:
    source = _metrics(damping=1.0, component=1.0)
    quarter_better = _metrics(damping=0.75, component=1.0)
    mandatory = fit.mandatory_gate_decision(source, quarter_better, source, quarter_better)
    assert mandatory["pass"] is True

    terminal_candidate = _metrics(damping=0.19, component=1.0, sign=1.0, gain=1.0)
    terminal = fit.terminal_decision(
        source, terminal_candidate, source, terminal_candidate
    )
    assert terminal["pass"] is True


def test_protocol_freezes_budget_and_streamed_fresh_cohort() -> None:
    protocol = fit.protocol_manifest()

    assert protocol["maximum_accepted_updates"] == 200
    assert protocol["development_interval_accepted_updates"] == 10
    assert protocol["mandatory_update_50_gate"][
        "endpoint_damping_nrmse_improvement_fraction_both_banks"
    ] == pytest.approx(0.25)
    assert protocol["qualification"]["total_scenes"] == 64
    assert protocol["qualification"]["factorial_seeds"] == list(range(360_971, 360_979))
    assert protocol["qualification"]["attitude_seeds"] == list(range(370_971, 370_979))
