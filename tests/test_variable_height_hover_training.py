from __future__ import annotations

import math

import pytest
import torch

from scripts.train_variable_height_hover import (
    LESSON_CYCLE,
    _settling_time,
    marker_score,
    safe_against_baseline,
)


def test_lesson_cycle_is_exactly_fifty_twenty_five_twenty_five() -> None:
    lessons = [LESSON_CYCLE[index % len(LESSON_CYCLE)] for index in range(200)]

    assert lessons.count("closed_loop") == 100
    assert lessons.count("paired_marker") == 50
    assert lessons.count("attitude_recovery") == 50


def test_settling_time_requires_a_continuous_half_second_window() -> None:
    height = torch.full((50, 2), 1.4)
    vertical_speed = torch.zeros_like(height)
    target = torch.ones(2)
    height[15:, 0] = 1.05
    height[15:30, 1] = 1.05
    height[30:, 1] = 1.4

    settling = _settling_time(height, vertical_speed, target, policy_hz=50)

    assert float(settling[0]) == pytest.approx(0.3)
    assert math.isinf(float(settling[1]))


def test_safety_gate_preserves_legacy_pair_and_attitude() -> None:
    baseline = {"attitude_recovery": {"success_rate": 0.90, "tilt_rms_mean_degrees": 4.0}}
    candidate = {
        "legacy_pair": {"pass": True},
        "attitude_recovery": {"success_rate": 0.86, "tilt_rms_mean_degrees": 4.8},
    }

    safe, reasons = safe_against_baseline(candidate, baseline)
    assert safe
    assert not reasons

    candidate["legacy_pair"]["pass"] = False
    safe, reasons = safe_against_baseline(candidate, baseline)
    assert not safe
    assert "legacy paired-marker retention failed" in reasons


def test_marker_score_penalizes_failure_and_ground_contact() -> None:
    good = {
        "marker_steps": {
            "altitude_rmse_mean_m": 0.10,
            "success_rate": 0.90,
            "ground_contact_rate": 0.0,
        }
    }
    bad = {
        "marker_steps": {
            "altitude_rmse_mean_m": 0.20,
            "success_rate": 0.50,
            "ground_contact_rate": 0.25,
        }
    }

    assert marker_score(good) < marker_score(bad)
