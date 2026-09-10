from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_full_native_step_response import (  # noqa: E402
    largest_feasible_scale,
    scale_decision,
)


def _metrics(
    objective: float,
    *,
    component: float = 1.0,
    rpy: float = 0.0,
) -> dict:
    return {
        "joint_normalized_mse": objective,
        "component_nrmse": {
            "common": component,
            "height": component,
            "damping": component,
            "interaction": component,
        },
        "rpy_source_nrmse": {"roll": rpy, "pitch": rpy, "yaw": rpy},
        "motor_output_max_absolute": 0.7,
        "all_attitude_cache_states_valid": True,
    }


def test_scale_decision_applies_training_and_development_improvement_floors() -> None:
    baseline = _metrics(1.0)
    same = _metrics(1.0)
    tiny = _metrics(1.0 - 5.0e-5)

    assert scale_decision(baseline, same, development=False)["pass"] is False
    assert scale_decision(baseline, tiny, development=False)["pass"] is False
    assert scale_decision(baseline, tiny, development=True)["pass"] is True


def test_scale_decision_keeps_component_and_rpy_guards() -> None:
    baseline = _metrics(1.0)
    component_failure = _metrics(0.8, component=1.021)
    rpy_failure = _metrics(0.8, rpy=0.051)

    first = scale_decision(baseline, component_failure, development=False)
    second = scale_decision(baseline, rpy_failure, development=False)

    assert first["pass"] is False
    assert any("source plus" in reason for reason in first["reasons"])
    assert second["pass"] is False
    assert any("source NRMSE" in reason for reason in second["reasons"])


def test_scale_decision_reports_measurable_damping_reduction() -> None:
    baseline = _metrics(1.0)
    candidate = _metrics(0.8)
    candidate["component_nrmse"]["damping"] = 0.9998

    result = scale_decision(baseline, candidate, development=False)

    assert result["pass"] is True
    assert result["damping_nrmse_improvement"] == pytest.approx(0.0002)
    assert result["measurable_damping_error_reduction"] is True


def test_largest_feasible_scale_uses_declared_descending_order() -> None:
    trials = [
        {"scale": 1.0, "decision": {"pass": False}},
        {"scale": 0.5, "decision": {"pass": True}},
        {"scale": 0.25, "decision": {"pass": True}},
    ]

    assert largest_feasible_scale(trials) == 0.5
