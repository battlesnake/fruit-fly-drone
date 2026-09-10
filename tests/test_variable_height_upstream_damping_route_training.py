from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_upstream_damping_route import (  # noqa: E402
    fixed_metric_riesz,
    training_protocol,
)


def test_riesz_direction_uses_fixed_shallow_denominators() -> None:
    gradient = {
        "edge_magnitude": torch.tensor((2.0,)),
        "bias": torch.tensor((-3.0,)),
    }

    direction = fixed_metric_riesz(gradient)

    assert direction["edge_magnitude"].item() == pytest.approx(-2.0 * 12_314)
    assert direction["bias"].item() == pytest.approx(3.0 * 303)


def test_protocol_freezes_metric_and_unconsumed_evaluation_seed_offsets() -> None:
    args = SimpleNamespace(
        attempts=50,
        milestone_attempt=25,
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

    protocol = training_protocol(args)

    assert protocol["reference_metric_denominators"] == {
        "edge_magnitude": 12_314,
        "bias": 303,
    }
    assert "Riesz descent direction" in protocol["reference_metric_governs"]
    assert "Jacobian equality projection" in protocol["reference_metric_governs"]
    assert args.seed + protocol["milestone_seed_offset"] == 350_941
    assert args.seed + protocol["terminal_seed_offset"] == 360_941
