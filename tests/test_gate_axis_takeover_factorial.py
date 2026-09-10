from __future__ import annotations

import pytest
import torch

from scripts.audit_gate_axis_takeover_factorial import (
    compose_axis_takeover_motor,
    rescue_classification,
)


def test_axis_takeover_composes_motor_channels_before_stick_plant() -> None:
    native = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    reserve = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    assert torch.equal(compose_axis_takeover_motor(native, reserve, "native"), native)
    assert torch.equal(
        compose_axis_takeover_motor(native, reserve, "reserve_throttle"),
        torch.tensor([[1.0, 2.0, 3.0, 40.0]]),
    )
    assert torch.equal(
        compose_axis_takeover_motor(native, reserve, "reserve_steering"),
        torch.tensor([[10.0, 20.0, 30.0, 4.0]]),
    )
    assert torch.equal(compose_axis_takeover_motor(native, reserve, "full_reserve"), reserve)
    with pytest.raises(ValueError, match="unknown axis-takeover"):
        compose_axis_takeover_motor(native, reserve, "invalid")


def test_rescue_classification_requires_every_declared_stratum() -> None:
    passing = {
        key: 0.95
        for key in (
            "success_rate",
            "light_success_rate",
            "heavy_success_rate",
            "negative_lateral_success_rate",
            "positive_lateral_success_rate",
            "negative_obliquity_success_rate",
            "positive_obliquity_success_rate",
        )
    }
    failing = {**passing, "light_success_rate": 0.89}
    results = {
        "source": {
            "native": failing,
            "reserve_throttle": failing,
            "reserve_steering": failing,
            "full_reserve": passing,
        },
        "failed_update": {
            "native": failing,
            "reserve_throttle": failing,
            "reserve_steering": failing,
            "full_reserve": failing,
        },
    }
    classification = rescue_classification(results, minimum_success=0.90)
    assert classification["passed"]
    assert classification["interpretation"] == "coupled_throttle_and_steering_takeover_required"
    assert classification["failed_prefix_damage_detected"]
