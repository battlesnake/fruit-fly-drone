from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import HoverConfig  # noqa: E402


def _final_candidate() -> dict:
    return {
        "all_metrics_finite": True,
        "all_recurrent_states_and_outputs_finite": True,
        "source_relative_nrmse_improvement": 0.6,
        "correct_sign_fraction": 1.0,
        "teacher_aligned_gain": 1.0,
        "endpoint_image_difference_max": 0.0,
        "maximum_motor_absolute": 0.7,
        "by_horizon": {
            str(horizon): {
                "correct_sign_fraction": 1.0,
                "teacher_aligned_gain": 1.0,
            }
            for horizon in motion.HISTORY_LENGTHS
        },
    }


def test_protocol_has_native_memory_and_motion_only_objective() -> None:
    protocol = motion.protocol_manifest()

    assert protocol["actor"]["privileged_inputs"] == []
    assert protocol["actor"]["state"] == "native MaleCNS recurrence only"
    assert protocol["objective"]["terms"] == [
        "normalized paired equal-endpoint throttle contrast MSE"
    ]
    assert protocol["objective"]["dense_or_source_anchor"] is False
    assert protocol["optimizer"]["first_finite_no_scale_result_is_terminal"] is True
    assert protocol["assisted_hover"] is False
    assert protocol["gate_flight"] is False


def test_exact_motion_bank_contains_each_registered_combination_once() -> None:
    bank = motion.build_exact_factorial_motion_bank(
        seed=991_001,
        held_out_styles=False,
        device=torch.device("cpu"),
        config=HoverConfig(),
    )

    manifest = motion.motion_bank_manifest(bank)

    assert manifest["cases"] == 24
    assert manifest["unique_combinations"] == 24
    assert manifest["exact_full_factorial"] is True
    assert manifest["history_length_counts"] == {"15": 8, "20": 8, "25": 8}
    assert manifest["height_sign_counts"] == {"-1.0": 12, "1.0": 12}
    assert motion._payload_hash(bank) == bank["sha256"]


def test_sampled_endpoint_motion_keeps_sign_and_reports_magnitude_mismatch() -> None:
    bank = motion.build_exact_factorial_motion_bank(
        seed=991_002,
        held_out_styles=False,
        device=torch.device("cpu"),
        config=HoverConfig(),
    )

    report = motion.sampled_endpoint_velocity_report(bank)

    assert report["pass"] is True
    assert report["all_nonzero_signs_match_label"] is True
    assert report["minimum_ratio"] > 0.0
    assert report["magnitude_match_required"] is False
    assert any(abs(value - 1.0) > 1.0e-3 for value in report[
        "ratio_sampled_last_frame_to_continuous_endpoint_label"
    ])


def test_milestone_requires_both_progress_and_sign() -> None:
    passing = {
        "all_metrics_finite": True,
        "all_recurrent_states_and_outputs_finite": True,
        "source_relative_nrmse_improvement": 0.25,
        "correct_sign_fraction": 0.5,
    }

    assert motion.milestone_decision(passing)["pass"] is True
    assert motion.milestone_decision(
        {**passing, "source_relative_nrmse_improvement": 0.249}
    )["pass"] is False
    assert motion.milestone_decision(
        {**passing, "correct_sign_fraction": 0.499}
    )["pass"] is False


def test_final_gate_applies_sign_and_gain_to_every_horizon() -> None:
    passing = _final_candidate()

    assert motion.final_motion_decision(passing, stage="training")["pass"] is True
    wrong_horizon = _final_candidate()
    wrong_horizon["by_horizon"]["20"]["teacher_aligned_gain"] = 0.49
    decision = motion.final_motion_decision(wrong_horizon, stage="development")
    assert decision["pass"] is False
    assert any("horizon 20 gain" in reason for reason in decision["reasons"])


def test_first_finite_no_scale_result_has_terminal_classification() -> None:
    classification, reason = motion._attempt_terminal(
        {"fatal_numerical_failure": False, "finite_difference": {}}
    )

    assert classification == "first_finite_no_scale_rejection"
    assert "no admissible ordinary scale" in reason


@pytest.mark.parametrize(
    "classification",
    [
        "derivative_probe_numerical_failure",
        "derivative_probe_noise_limited_inconclusive",
        "derivative_probe_above_noise_nonconvergence",
    ],
)
def test_derivative_terminal_classification_is_preserved(classification: str) -> None:
    actual, _ = motion._attempt_terminal(
        {
            "fatal_numerical_failure": True,
            "finite_difference": {"classification": classification},
        }
    )

    assert actual == classification


def test_formal_seeds_are_rejected_by_disposable_runs() -> None:
    with pytest.raises(ValueError, match="formal seeds"):
        motion.require_nonformal_seeds(motion.TRAINING_SEED)
