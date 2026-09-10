from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.audit_gate_analytic_teachers import (
    mass_hover_correction,
    score,
    teacher_rc_for_mode,
)
from scripts.train_gate_full_network_oracle import load_frozen_controller
from scripts.train_gate_multitime_distillation import diverse_matched_cases


def test_exact_mass_hover_correction_has_declared_zero_point() -> None:
    device = torch.device("cpu")
    source, _, hover, gate_config, _ = load_frozen_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        device,
    )
    cases = diverse_matched_cases(
        8,
        seed=91,
        device=device,
        hover_config=hover,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    correction = mass_hover_correction(
        cases.state,
        torch.tensor([1.08] * 8),
        nominal_scale=1.08,
        hover_config=hover,
    )
    assert torch.equal(correction, torch.zeros_like(correction))

    reserve = teacher_rc_for_mode(
        "visual_accelerometer_reserve",
        source,
        cases.state,
        cases.gate,
        cases.mass_scale,
        hover,
    )
    exact = teacher_rc_for_mode(
        "visual_accelerometer_exact_mass",
        source,
        cases.state,
        cases.gate,
        cases.mass_scale,
        hover,
    )
    assert torch.equal(reserve[:, :3], exact[:, :3])
    with pytest.raises(ValueError, match="unknown teacher mode"):
        teacher_rc_for_mode(
            "invalid",
            source,
            cases.state,
            cases.gate,
            cases.mass_scale,
            hover,
        )


def test_teacher_score_prioritizes_worst_mass_half() -> None:
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
