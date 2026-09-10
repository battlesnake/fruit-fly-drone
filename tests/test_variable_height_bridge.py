from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_bridge import (  # noqa: E402
    BRIDGE_LESSONS,
    _bridge_guards,
    _decision,
)


def _evaluation() -> dict:
    return {
        "marker_steps": {
            "altitude_rmse_mean_m": 0.47,
            "ground_contact_rate": 0.0,
            "invalid_flight_rate": 0.0,
        },
        "attitude_recovery": {
            "success_rate": 0.20,
            "tilt_rms_mean_degrees": 4.0,
            "ground_contact_rate": 0.0,
            "invalid_flight_rate": 0.0,
        },
        "legacy_pair": {"pass": True},
        "genuine_held_out_pair": {"contrast_nrmse": 0.60, "pass": False},
        "training_support_pair_matrix": {
            "mean_contrast_nrmse": 0.60,
            "mean_contrast_nrmse_by_style": {
                "wall_0_floor_0": 0.60,
                "wall_0_floor_1": 0.60,
                "wall_1_floor_0": 0.60,
                "wall_1_floor_1": 0.60,
            },
        },
        "fresh_height_pair_matrix": {
            "every_stratum_pass": True,
        },
    }


def test_bridge_accumulates_exact_advisor_lesson_mix() -> None:
    assert BRIDGE_LESSONS == (
        "random_scene_small_pair",
        "random_scene_medium_pair",
        "legal_cross_band_source_pair",
        "dynamic_source_replay",
    )


def test_bridge_guard_rejects_absolute_height_regression() -> None:
    baseline = _evaluation()
    candidate = deepcopy(baseline)
    candidate["marker_steps"]["altitude_rmse_mean_m"] = 0.50

    passed, reasons = _bridge_guards(candidate, baseline)

    assert not passed
    assert "absolute-height RMSE" in reasons[0]


def test_update_50_requires_25_percent_genuine_pair_improvement() -> None:
    baseline = _evaluation()
    baseline["training_support_pair_matrix"]["mean_contrast_nrmse"] = 0.80
    baseline["training_support_pair_matrix"]["mean_contrast_nrmse_by_style"] = {
        key: 0.80
        for key in baseline["training_support_pair_matrix"][
            "mean_contrast_nrmse_by_style"
        ]
    }
    candidate = deepcopy(baseline)
    candidate["training_support_pair_matrix"]["mean_contrast_nrmse"] = 0.60
    candidate["training_support_pair_matrix"]["mean_contrast_nrmse_by_style"] = {
        key: 0.60
        for key in candidate["training_support_pair_matrix"][
            "mean_contrast_nrmse_by_style"
        ]
    }

    decision = _decision(candidate, baseline)

    assert decision["update_50_continuation_pass"]


def test_bridge_acceptance_requires_every_fresh_height_stratum() -> None:
    baseline = _evaluation()
    baseline["genuine_held_out_pair"]["contrast_nrmse"] = 0.80
    candidate = deepcopy(baseline)
    candidate["genuine_held_out_pair"] = {"contrast_nrmse": 0.20, "pass": True}
    candidate["fresh_height_pair_matrix"]["every_stratum_pass"] = False

    decision = _decision(candidate, baseline)

    assert not decision["bridge_acceptance_pass"]
    assert not decision["every_fresh_height_amplitude_scene_stratum_pass"]
