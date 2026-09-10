from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_common_anchor_ablation import (  # noqa: E402
    legacy_rpy_nrmse_with_original_denominator,
    protocol_manifest,
    raw_direction,
)


def test_legacy_rpy_removes_throttle_without_changing_denominator() -> None:
    legacy = {
        "roll_source_nrmse": 0.03,
        "pitch_source_nrmse": 0.04,
        "yaw_source_nrmse": 0.0,
        "throttle_source_nrmse": 99.0,
    }

    value = legacy_rpy_nrmse_with_original_denominator(legacy)

    assert value == pytest.approx(0.025)


def test_raw_direction_has_no_common_projection_or_screen() -> None:
    student = SimpleNamespace(edge_magnitude=torch.tensor((0.5, 0.5)))
    gradient = {
        "edge_magnitude": torch.tensor((1.0, -1.0)),
        "bias": torch.tensor((1.0,)),
    }

    direction, report = raw_direction(student, gradient, torch.tensor((0, 1)))

    assert direction is not None
    assert report["candidate"] == "raw_no_common_anchor"
    assert report["first_order_loss_derivative"] < 0.0


def test_protocol_explicitly_removes_only_absolute_throttle_anchors() -> None:
    args = SimpleNamespace(
        attempts=25,
        gradient_banks=2,
        guard_banks=2,
        evaluation_banks=8,
        batch_size=8,
        constraint_batch_size=8,
        unroll=25,
        constraint_prefix_steps=50,
        history_steps=25,
        policy_hz=50,
        device="cuda",
        smoke_test=False,
        seed=290_941,
    )

    protocol = protocol_manifest(args)

    assert protocol["common_output_projection_or_screening"] is False
    assert protocol["absolute_throttle_source_gates"] is False
    assert protocol["absolute_teacher_throttle_objective_or_gate"] is False
    assert protocol["dynamic_rpy_each_nrmse_limit"] == 0.05
    assert protocol["selected_source_metric_radius"] == 0.0005
    assert protocol["reused_benchmark_seed_offset"] == 70_000
    assert protocol["fresh_qualification_seed_offset"] == 80_000
