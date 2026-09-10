from __future__ import annotations

import math

import torch

from scripts.calibrate_gate_promoted_oracle import (
    BiasSpec,
    calibration,
    half_step_refinement,
    score,
    shuffled_calibration,
)


def test_endpoint_parameterization_and_half_step_refinement() -> None:
    transferred = BiasSpec(-0.15, 0.05)
    values = calibration(transferred)
    assert math.isclose(values["intercept"], -0.05)
    assert math.isclose(values["mass_slope"], 0.10)

    grid = {transferred}
    refinement = half_step_refinement(transferred, grid)
    assert len(refinement) == 8
    assert transferred not in refinement
    assert any(
        math.isclose(item.light_bias, -0.1625) and math.isclose(item.heavy_bias, 0.0375)
        for item in refinement
    )


def test_shuffled_calibration_substitutes_matched_partner_labels() -> None:
    spec = BiasSpec(-0.15, 0.05)
    mass = torch.tensor([0.92, 1.08, 0.96, 1.04])
    values = shuffled_calibration(spec, mass)
    normalized = ((mass - 1.0) / 0.08).clamp(-1.0, 1.0)
    applied = values["intercept"] + values["mass_slope"] * normalized
    expected = spec.intercept + spec.mass_slope * normalized[[1, 0, 3, 2]]
    assert torch.allclose(applied, expected)


def test_score_prioritizes_worst_mass_half() -> None:
    balanced = {
        "light_success_rate": 0.90,
        "heavy_success_rate": 0.90,
        "success_rate": 0.90,
        "crossing_radial_mean_m": 0.3,
    }
    imbalanced = {
        "light_success_rate": 1.0,
        "heavy_success_rate": 0.89,
        "success_rate": 0.945,
        "crossing_radial_mean_m": 0.2,
    }
    assert score(balanced) > score(imbalanced)
