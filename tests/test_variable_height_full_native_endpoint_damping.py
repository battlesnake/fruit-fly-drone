from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_full_native_endpoint_damping import (  # noqa: E402
    DEVELOPMENT_SEED,
    FULL_SCALES,
    MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT,
    TRAIN_SEED,
    endpoint_damping_scale,
    largest_feasible_scale,
    protocol_manifest,
    scale_decision,
)
from audit_variable_height_full_native_joint import FactorialCache  # noqa: E402


def _metrics(endpoint: float, *, component: float = 1.0, rpy: float = 0.0) -> dict:
    return {
        "joint_normalized_mse": 100.0,
        "component_nrmse": {
            "common": component,
            "height": component,
            "damping": 5.0,
            "interaction": 5.0,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": component,
                "height": component,
                "damping": 5.0,
                "interaction": 5.0,
            }
            for step in (15, 20, 25)
        },
        "rpy_source_nrmse": {"roll": rpy, "pitch": rpy, "yaw": rpy},
        "endpoint_damping_nrmse": endpoint,
        "motor_output_max_absolute": 0.7,
        "all_attitude_cache_states_valid": True,
    }


def test_endpoint_scale_uses_only_endpoint_teacher_rms() -> None:
    motor = torch.zeros(2, 25, 4, 4)
    # Earlier damping is deliberately much larger than endpoint damping.
    motor[:, 14, :, 3] = torch.tensor((-0.4, 0.4, -0.4, 0.4))
    motor[:, 19, :, 3] = torch.tensor((-0.2, 0.2, -0.2, 0.2))
    motor[:, 24, :, 3] = torch.tensor((-0.02, 0.02, -0.02, 0.02))
    cache = FactorialCache(
        prefix_images=torch.empty(2, 3, 2, 2),
        response_images=torch.empty(2, 25, 4, 3, 2, 2),
        attitude=torch.empty(2, 2),
        teacher_motor=motor,
        teacher_rc=torch.empty(2, 25, 4, 4),
        height_error=torch.tensor((0.05, 0.1)),
        vertical_speed=torch.tensor((0.15, 0.3)),
        approach_steps=torch.tensor((15, 25)),
        camera_height=torch.tensor((1.0, 1.2)),
        endpoint_image_difference_max=0.0,
        sha256="test",
    )

    assert endpoint_damping_scale(cache) == pytest.approx(0.02)


def test_scale_decision_is_damping_specific_and_preserves_c_p_rpy() -> None:
    baseline = _metrics(1.0)
    candidate = _metrics(1.0 - MINIMUM_ENDPOINT_D_NRMSE_IMPROVEMENT)

    assert scale_decision(baseline, candidate)["pass"] is True

    no_damping = _metrics(0.9991)
    assert scale_decision(baseline, no_damping)["pass"] is False

    bad_horizon = _metrics(0.9)
    bad_horizon["by_supervision_step_nrmse"]["20"]["height"] = 1.021
    result = scale_decision(baseline, bad_horizon)
    assert result["pass"] is False
    assert any("step 20" in reason for reason in result["reasons"])

    bad_rpy = _metrics(0.9, rpy=0.051)
    assert scale_decision(baseline, bad_rpy)["pass"] is False


def test_scale_decision_does_not_gate_joint_or_interaction() -> None:
    baseline = _metrics(1.0)
    candidate = _metrics(0.9)
    candidate["joint_normalized_mse"] = 1_000_000.0
    candidate["component_nrmse"]["interaction"] = 1_000_000.0
    for horizon in candidate["by_supervision_step_nrmse"].values():
        horizon["interaction"] = 1_000_000.0

    assert scale_decision(baseline, candidate)["pass"] is True


def test_largest_feasible_scale_uses_frozen_descending_grid() -> None:
    trials = [
        {"scale": 1.0, "decision": {"pass": False}},
        {"scale": 0.5, "decision": {"pass": True}},
        {"scale": 0.25, "decision": {"pass": True}},
    ]

    assert FULL_SCALES == (1.0, 0.5, 0.25, 0.125)
    assert largest_feasible_scale(trials) == 0.5


def test_protocol_freezes_fresh_banks_and_no_development_retry() -> None:
    protocol = protocol_manifest()

    assert protocol["train_seed"] == TRAIN_SEED == 340_961
    assert protocol["development_seed"] == DEVELOPMENT_SEED == 350_961
    assert protocol["objective"] == "normalized endpoint damping MSE only"
    assert protocol["complete_displacement_scales_descending"] == [1.0, 0.5, 0.25, 0.125]
    assert protocol["development_candidates_evaluated"] == 1
    assert protocol["no_smaller_scale_after_development_failure"] is True
    assert protocol["joint_loss_and_interaction_are_reporting_only"] is True
