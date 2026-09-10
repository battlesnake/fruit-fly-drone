from __future__ import annotations

import torch

from scripts.train_gate_recurrent_routing_continuation import (
    continuation_gate,
    make_batch_schedule,
    weighted_window_loss,
)


def metrics(early_light, early_heavy, middle_light, middle_heavy, late_light, late_heavy):
    values = (early_light, early_heavy, middle_light, middle_heavy, late_light, late_heavy)
    names = (
        "0.50:0.75/light",
        "0.50:0.75/heavy",
        "0.75:1.00/light",
        "0.75:1.00/heavy",
        "1.00:1.50/light",
        "1.00:1.50/heavy",
    )
    return {
        "per_window_and_mass": {
            name: {"nrmse": value} for name, value in zip(names, values, strict=True)
        }
    }


def test_weighted_window_loss_matches_preregistered_treatment() -> None:
    groups = [torch.tensor(float(value)) for value in (1, 3, 5, 7, 9, 11)]
    assert float(weighted_window_loss(groups, (1, 1, 1))) == 6.0
    assert float(weighted_window_loss(groups, (4, 1, 1))) == 4.0


def test_continuation_gate_requires_early_gain_other_safety_and_prefix() -> None:
    baseline = metrics(0.48, 0.608, 0.30, 0.25, 0.29, 0.21)
    passing = metrics(0.44, 0.48, 0.34, 0.28, 0.30, 0.22)
    early_fail = metrics(0.44, 0.50, 0.30, 0.25, 0.29, 0.21)
    regression = metrics(0.44, 0.48, 0.36, 0.25, 0.29, 0.21)
    kwargs = {
        "minimum_early_improvement": 0.20,
        "maximum_other_regression": 0.05,
        "prefix_rmse": 0.005,
        "maximum_prefix_rmse": 0.01,
    }
    assert continuation_gate(passing, baseline, **kwargs)["passed"]
    assert not continuation_gate(early_fail, baseline, **kwargs)["passed"]
    assert not continuation_gate(regression, baseline, **kwargs)["passed"]


def test_paired_schedule_is_deterministic() -> None:
    first = make_batch_schedule(updates=200, pairs_per_batch=8, seed=1_050_031)
    second = make_batch_schedule(updates=200, pairs_per_batch=8, seed=1_050_031)
    assert len(first) == 200
    assert all((left == right).all() for left, right in zip(first, second, strict=True))
    assert first[0].tolist() == [38, 6, 53, 55, 5, 8, 14, 41]
