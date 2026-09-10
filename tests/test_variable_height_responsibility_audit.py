from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_responsibilities import (  # noqa: E402
    balanced_signed_values,
    bootstrap_r2_improvement,
    fit_sparse_probe,
    paired_mean_replace,
)


def test_balanced_signed_values_contains_both_signs_and_magnitudes() -> None:
    torch.manual_seed(3)
    values = balanced_signed_values(8, (0.05, 0.10), device=torch.device("cpu"))

    assert sorted(values.tolist()) == pytest.approx(
        [-0.10, -0.10, -0.05, -0.05, 0.05, 0.05, 0.10, 0.10]
    )


def test_paired_mean_replace_removes_only_selected_contrast() -> None:
    neural = torch.tensor(((1.0, 2.0), (3.0, 4.0), (5.0, 8.0), (7.0, 10.0)))

    paired_mean_replace(neural, torch.tensor([1]), pairs=2)

    assert neural[:, 0].tolist() == [1.0, 3.0, 5.0, 7.0]
    assert neural[:, 1].tolist() == [5.0, 7.0, 5.0, 7.0]


def test_sparse_probe_generalizes_signal_and_rejects_noise() -> None:
    generator = torch.Generator().manual_seed(9)
    train_target = torch.randn(64, generator=generator)
    test_target = torch.randn(128, generator=generator)
    train = torch.randn(64, 80, generator=generator) * 0.05
    test = torch.randn(128, 80, generator=generator) * 0.05
    train[:, 17] = 2.0 * train_target
    test[:, 17] = 2.0 * test_target

    report = fit_sparse_probe(
        train,
        train_target,
        test,
        test_target,
        maximum_features=4,
        bootstrap_samples=100,
        seed=11,
        feature_ids=torch.arange(1000, 1080).numpy(),
    )

    assert report["representation_gate_passed"] is True
    assert 1017 in report["selected_feature_ids"]
    assert report["metrics"]["r2"] > 0.99
    assert report["shuffled_label_control"]["metrics"]["r2"] < 0.2


def test_bootstrap_improvement_is_positive_for_exact_prediction() -> None:
    target = torch.linspace(-1.0, 1.0, 32)
    baseline = torch.zeros_like(target)

    interval = bootstrap_r2_improvement(target.clone(), baseline, target, samples=100, seed=12)

    assert interval["lower_95"] > 0.9
