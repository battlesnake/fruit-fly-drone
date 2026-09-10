from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_full_native_joint import (  # noqa: E402
    COMPONENT_BASELINE_TOLERANCE,
    DEVELOPMENT_SEED,
    SCENES_PER_BANK,
    TRAIN_SEED,
    FactorialCache,
    decision,
    protocol_manifest,
    smooth_return_velocity,
    teacher_identity_report,
    training_teacher_scales,
)


def test_smooth_return_velocity_is_endpoint_speed_and_zero_while_inactive() -> None:
    speed = torch.tensor((0.15, -0.30))
    approach = torch.tensor((15, 25))

    inactive = smooth_return_velocity(
        speed, step=0, total_steps=25, approach_steps=approach, policy_hz=50
    )
    endpoint = smooth_return_velocity(
        speed, step=24, total_steps=25, approach_steps=approach, policy_hz=50
    )

    assert inactive[0] == 0.0
    assert endpoint == pytest.approx(speed)


def test_training_teacher_scales_use_rms_with_floor() -> None:
    # Branch-major target values give height RMS 0.02 and damping RMS 0.005.
    branch = torch.tensor((-0.015, -0.025, 0.025, 0.015)).reshape(1, 1, 4, 1)
    motor = torch.zeros(1, 25, 4, 4)
    for step in (14, 19, 24):
        motor[:, step, :, 3] = branch[0, 0, :, 0]
    cache = FactorialCache(
        prefix_images=torch.empty(1, 3, 2, 2),
        response_images=torch.empty(1, 25, 4, 3, 2, 2),
        attitude=torch.empty(1, 2),
        teacher_motor=motor,
        teacher_rc=torch.empty(1, 25, 4, 4),
        height_error=torch.tensor((0.05,)),
        vertical_speed=torch.tensor((0.15,)),
        approach_steps=torch.tensor((25,)),
        camera_height=torch.tensor((1.0,)),
        endpoint_image_difference_max=0.0,
        sha256="test",
    )

    scales = training_teacher_scales(cache)

    assert scales["height"] == pytest.approx(0.02)
    assert scales["damping"] == pytest.approx(0.01)
    identity = teacher_identity_report(cache, scales)
    assert identity["pass"] is True
    assert identity["branch_reconstruction_max_absolute_error"] < 1.0e-7


def _metrics(objective: float, component: float = 1.0, rpy: float = 0.0) -> dict:
    return {
        "joint_normalized_mse": objective,
        "component_nrmse": {
            "common": component,
            "height": component,
            "damping": component,
            "interaction": component,
        },
        "rpy_source_nrmse": {"roll": rpy, "pitch": rpy, "yaw": rpy},
        "motor_output_max_absolute": 0.5,
        "all_attitude_cache_states_valid": True,
    }


def test_decision_allows_small_first_step_component_trade() -> None:
    baseline = _metrics(1.0)
    candidate = _metrics(0.9, component=1.0 + COMPONENT_BASELINE_TOLERANCE)

    result = decision(
        baseline,
        candidate,
        baseline,
        candidate,
        directional={"pass": True},
        deterministic_replay_max_difference=0.0,
        teacher_identity={"pass": True},
        teacher_positive={"pass": True},
        endpoint_image_difference_max=0.0,
        parameters_restored_exactly=True,
    )

    assert result["pass"] is True


def test_decision_rejects_development_regression_and_rpy_breakage() -> None:
    baseline = _metrics(1.0)
    candidate = _metrics(0.9)
    bad_development = _metrics(1.1, rpy=0.051)

    result = decision(
        baseline,
        candidate,
        baseline,
        bad_development,
        directional={"pass": True},
        deterministic_replay_max_difference=0.0,
        teacher_identity={"pass": True},
        teacher_positive={"pass": True},
        endpoint_image_difference_max=0.0,
        parameters_restored_exactly=True,
    )

    assert result["pass"] is False
    assert any("development joint" in reason for reason in result["reasons"])
    assert any("dynamic RPY" in reason for reason in result["reasons"])


def test_protocol_freezes_full_native_preflight_contract() -> None:
    args = SimpleNamespace(
        train_seed=TRAIN_SEED,
        development_seed=DEVELOPMENT_SEED,
        scenes=SCENES_PER_BANK,
        prefix_steps=5,
        response_steps=25,
        policy_hz=50,
        smoke_test=False,
    )

    protocol = protocol_manifest(args)

    assert protocol["opened_parameter_families"] == [
        "edge_magnitude",
        "bias",
        "raw_time_constant",
    ]
    assert protocol["prefix_and_response_are_differentiated"] is True
    assert protocol["privileged_actor_inputs"] == []
    assert protocol["literal_damping_sign_and_gain_horizon"] == 25
