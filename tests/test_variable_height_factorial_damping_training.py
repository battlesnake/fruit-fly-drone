from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_factorial_damping import (  # noqa: E402
    fresh_qualification_decision,
    guard_bank_improvements,
    matched_height_bank_report,
    protocol_manifest,
)


def _bank(height_prediction: list[float], damping_nrmse: float) -> dict:
    return {
        "height": {
            "prediction": height_prediction,
            "target": [1.0, 2.0, 1.0, 2.0],
        },
        "height_error_magnitude_metres": [0.05, 0.10, 0.05, 0.10],
        "damping": {"fixed_scale_nrmse": damping_nrmse},
    }


def test_matched_height_bank_report_gates_each_scene() -> None:
    source = {"banks": [_bank([1.0, 2.0, 1.0, 2.0], 1.0)]}
    safe = {"banks": [_bank([0.91, 2.18, 0.91, 2.18], 0.9)]}
    unsafe_cancelling = {"banks": [_bank([0.7, 2.6, 1.1, 1.8], 0.9)]}

    safe_report = matched_height_bank_report(safe, source)
    unsafe_report = matched_height_bank_report(unsafe_cancelling, source)

    assert safe_report["pass"] is True
    assert safe_report["ratio_minimum"] == pytest.approx(0.91)
    assert safe_report["ratio_maximum"] == pytest.approx(1.09)
    assert unsafe_report["pass"] is False


def test_guard_improvement_is_checked_per_bank() -> None:
    baseline = {"banks": [_bank([1.0] * 4, 1.0), _bank([1.0] * 4, 2.0)]}
    candidate = {"banks": [_bank([1.0] * 4, 0.8), _bank([1.0] * 4, 1.9)]}

    assert guard_bank_improvements(baseline, candidate) == pytest.approx([0.2, 0.1])


def test_fresh_qualification_requires_sign_gain_height_and_preservation() -> None:
    candidate = {
        "all_finite": True,
        "damping": {"correct_sign_fraction": 0.95, "teacher_aligned_gain": 0.8},
    }

    passed = fresh_qualification_decision(candidate, {"pass": True}, {"pass": True})
    failed = fresh_qualification_decision(
        {**candidate, "damping": {"correct_sign_fraction": 0.8, "teacher_aligned_gain": 0.8}},
        {"pass": True},
        {"pass": True},
    )

    assert passed["pass"] is True
    assert failed["pass"] is False


def test_protocol_freezes_factorial_training_counts_and_unanchored_common() -> None:
    args = SimpleNamespace(
        attempts=25,
        gradient_banks=2,
        guard_banks=2,
        evaluation_banks=16,
        batch_size=4,
        constraint_batch_size=8,
        unroll=25,
        constraint_prefix_steps=50,
        history_steps=25,
        policy_hz=50,
        device="cuda",
        smoke_test=False,
        seed=310_949,
    )

    protocol = protocol_manifest(args)

    assert protocol["evaluation_scenes"] == 64
    assert protocol["evaluation_histories"] == 256
    assert protocol["matched_height_component_projection"].startswith("same cases")
    assert protocol["absolute_throttle_source_gate"] is False
    assert protocol["useful_progress_damping_nrmse_improvement_fraction"] == 0.25
