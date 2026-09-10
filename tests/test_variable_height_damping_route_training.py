from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_damping_routes import (  # noqa: E402
    evaluation_decision,
    guard_bank_improvements,
    summarize_motion_metrics,
)


def test_motion_summary_uses_all_balanced_banks() -> None:
    reports = [
        {"target_contrast": [-0.05, 0.05], "predicted_contrast": [-0.025, 0.025]},
        {"target_contrast": [-0.10, 0.10], "predicted_contrast": [-0.05, 0.05]},
    ]

    summary = summarize_motion_metrics(reports)

    assert summary["pairs"] == 4
    assert summary["teacher_aligned_gain"] == pytest.approx(0.5)
    assert summary["correct_sign_fraction"] == 1.0
    assert summary["fixed_scale_nrmse"] == pytest.approx((0.625) ** 0.5)


def test_attempt_25_gate_requires_meaningful_progress_and_replay() -> None:
    baseline = {"fixed_scale_nrmse": 1.0}
    preservation = {"pass": True}
    insufficient = {
        "fixed_scale_nrmse": 0.80,
        "correct_sign_fraction": 1.0,
        "teacher_aligned_gain": 1.0,
    }
    sufficient = {**insufficient, "fixed_scale_nrmse": 0.74}

    rejected = evaluation_decision(insufficient, baseline, preservation, milestone=True)
    accepted = evaluation_decision(sufficient, baseline, preservation, milestone=True)

    assert rejected["meaningful_progress_gate_passed"] is False
    assert rejected["pass"] is False
    assert accepted["pass"] is True


def test_guard_bank_improvements_remain_independent() -> None:
    baseline = {
        "banks": [
            {"fixed_scale_nrmse": 1.0},
            {"fixed_scale_nrmse": 2.0},
        ]
    }
    candidate = {
        "banks": [
            {"fixed_scale_nrmse": 0.9998},
            {"fixed_scale_nrmse": 2.0001},
        ]
    }

    improvements = guard_bank_improvements(baseline, candidate)

    assert improvements[0] == pytest.approx(0.0002)
    assert improvements[1] == pytest.approx(-0.0001)


def test_final_replay_rejects_wrong_sign_gain() -> None:
    decision = evaluation_decision(
        {
            "fixed_scale_nrmse": 0.70,
            "correct_sign_fraction": 0.0,
            "teacher_aligned_gain": -0.5,
        },
        {"fixed_scale_nrmse": 1.0},
        {"pass": True},
        milestone=False,
    )

    assert decision["fresh_motion_replay_gate_passed"] is False
    assert decision["pass"] is False


def test_final_replay_fails_closed_on_nonfinite_metrics() -> None:
    decision = evaluation_decision(
        {
            "fixed_scale_nrmse": float("nan"),
            "correct_sign_fraction": 1.0,
            "teacher_aligned_gain": 1.0,
        },
        {"fixed_scale_nrmse": 1.0},
        {"pass": True},
        milestone=False,
    )

    assert decision["all_finite"] is False
    assert decision["fresh_motion_replay_gate_passed"] is False
    assert decision["pass"] is False
