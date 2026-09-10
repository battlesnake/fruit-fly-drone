from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from train_variable_height_constrained_descent import (  # noqa: E402
    project_to_source_ball,
)
from train_variable_height_trust_region import (  # noqa: E402
    cap_displacement,
    family_rms,
    project_displacement,
)


def test_projection_satisfies_linearized_rows_in_equal_family_rms_metric() -> None:
    displacement = {
        "edge_magnitude": torch.tensor([0.3, -0.2, 0.1, 0.5]),
        "bias": torch.tensor([0.4, -0.1]),
        "raw_time_constant": torch.tensor([0.2]),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "bias": torch.tensor([0.5, 0.0]),
            "raw_time_constant": torch.tensor([0.0]),
        },
        {
            "edge_magnitude": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "bias": torch.tensor([0.0, -0.25]),
            "raw_time_constant": torch.tensor([0.5]),
        },
    ]
    residual = torch.tensor([0.2, -0.15])

    projected, report = project_displacement(displacement, rows, residual)

    actual = torch.stack(
        [
            residual[index]
            + sum((row[name] * projected[name]).sum() for name in displacement)
            for index, row in enumerate(rows)
        ]
    )
    assert torch.allclose(actual, torch.zeros_like(actual), atol=1.0e-6)
    assert report["retained_rank"] == 2


def test_single_scalar_cap_preserves_family_ratios() -> None:
    displacement = {
        "edge_magnitude": torch.full((4,), 4.0e-5),
        "bias": torch.full((2,), 2.0e-5),
        "raw_time_constant": torch.full((3,), 1.0e-5),
    }

    capped, scale = cap_displacement(displacement, 2.0e-5)

    assert scale == pytest.approx(0.5)
    assert max(family_rms(capped).values()) == pytest.approx(2.0e-5)
    assert capped["bias"][0] / capped["raw_time_constant"][0] == pytest.approx(2.0)


def test_source_ball_projection_enforces_metric_radius() -> None:
    source = {
        "edge_magnitude": torch.zeros(4),
        "bias": torch.zeros(2),
        "raw_time_constant": torch.zeros(3),
    }
    base = {
        "edge_magnitude": torch.full((4,), 6.0e-5),
        "bias": torch.full((2,), 6.0e-5),
        "raw_time_constant": torch.full((3,), 4.0e-5),
    }
    outward = {name: torch.full_like(value, 2.0e-5) for name, value in base.items()}

    step, report = project_to_source_ball(base, source, outward, 1.0e-4)
    candidate = {name: base[name] + step[name] for name in base}

    metric_norm = sum(float(value.square().mean()) for value in candidate.values()) ** 0.5
    assert metric_norm <= 1.0e-4 * (1.0 + 1.0e-6)
    assert report["source_ball_projection_active"] is True
