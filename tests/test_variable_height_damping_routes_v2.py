from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_damping_routes import masked_family_rms  # noqa: E402
from audit_variable_height_damping_routes_v2 import (  # noqa: E402
    apply_edge_bounds,
    scale_to_final_cap,
    screen_candidates,
)


def test_scale_to_final_cap_can_expand_projected_direction() -> None:
    direction = {
        "edge_magnitude": torch.full((4,), 2.5e-6),
        "bias": torch.full((2,), 1.0e-7),
    }

    scaled, factor = scale_to_final_cap(direction, 2.0e-5)

    assert factor == pytest.approx(8.0)
    assert masked_family_rms(scaled)["edge_magnitude"] == pytest.approx(2.0e-5)


def test_edge_bounds_are_applied_before_screening() -> None:
    direction = {
        "edge_magnitude": torch.tensor((-0.2, 0.3, 0.1)),
        "bias": torch.tensor((0.4,)),
    }

    bounded, report = apply_edge_bounds(direction, torch.tensor((0.1, 7.9, 2.0)))

    assert bounded["edge_magnitude"].tolist() == pytest.approx([-0.1, 0.1, 0.1])
    assert report == {"edges_clipped_at_zero": 1, "edges_clipped_at_eight": 1}


def test_screen_selects_best_admissible_descent() -> None:
    gradient = {"edge_magnitude": torch.tensor((1.0,)), "bias": torch.tensor((1.0,))}
    candidates = {
        "safe_slow": {
            "edge_magnitude": torch.tensor((-0.1,)),
            "bias": torch.tensor((0.0,)),
        },
        "safe_fast": {
            "edge_magnitude": torch.tensor((-0.2,)),
            "bias": torch.tensor((0.0,)),
        },
        "unsafe": {
            "edge_magnitude": torch.tensor((-0.3,)),
            "bias": torch.tensor((0.0,)),
        },
    }
    rows = [{"edge_magnitude": torch.tensor((1.0,)), "bias": torch.tensor((0.0,))}]

    selected, reports = screen_candidates(
        candidates,
        gradient,
        rows,
        torch.zeros(1),
        per_update_rms=0.011,
        source_rms=0.02,
        maximum_absolute=0.02,
    )

    assert selected == "safe_fast"
    assert reports["safe_fast"]["admissible"] is True
    assert reports["unsafe"]["admissible"] is False
