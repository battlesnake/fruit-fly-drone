from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_variable_height_factorial_damping import (  # noqa: E402
    balanced_amplitude_grid,
    bound_aware_height_null_projection,
    factorial_components,
    matched_height_response,
    smooth_return_offset,
)


def test_balanced_amplitude_grid_contains_full_factorial() -> None:
    torch.manual_seed(5)

    height, speed = balanced_amplitude_grid(4, device=torch.device("cpu"))

    observed = sorted(zip(height.tolist(), speed.tolist(), strict=True))
    expected = [(0.05, 0.15), (0.05, 0.30), (0.10, 0.15), (0.10, 0.30)]
    for actual, wanted in zip(observed, expected, strict=True):
        assert actual == pytest.approx(wanted)


def test_factorial_components_separate_all_four_terms() -> None:
    common = torch.tensor((0.30, 0.40))
    height = torch.tensor((0.10, 0.20))
    damping = torch.tensor((-0.05, -0.07))
    interaction = torch.tensor((0.01, -0.02))
    values = torch.stack(
        [
            common + se * height + sv * damping + se * sv * interaction
            for se, sv in ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
        ]
    )

    result = factorial_components(values)

    assert result["common"] == pytest.approx(common)
    assert result["height"] == pytest.approx(height)
    assert result["damping"] == pytest.approx(damping)
    assert result["interaction"] == pytest.approx(interaction)


def test_smooth_history_returns_to_same_endpoint() -> None:
    velocity = torch.tensor((0.15, -0.30))
    duration = torch.tensor((15, 25))

    endpoint = smooth_return_offset(
        velocity,
        step=24,
        total_steps=25,
        approach_steps=duration,
        policy_hz=50,
    )

    assert endpoint.tolist() == [0.0, 0.0]


def test_height_null_projection_freezes_edge_that_would_cross_zero() -> None:
    raw = {
        "edge_magnitude": torch.tensor((-2.0e-5, 1.0e-5)),
        "bias": torch.tensor((1.0e-5,)),
    }
    rows = [
        {
            "edge_magnitude": torch.tensor((1.0, 1.0)),
            "bias": torch.tensor((1.0,)),
        }
    ]

    direction, report = bound_aware_height_null_projection(
        raw, rows, torch.tensor((0.0, 1.0))
    )
    residual = sum((rows[0][name] * direction[name]).sum() for name in direction)

    assert direction["edge_magnitude"][0] == 0.0
    assert report["frozen_bound_edges"] == 1
    assert report["converged_without_bound_violation"] is True
    assert float(residual) == pytest.approx(0.0, abs=1e-7)


def test_matched_height_response_is_grouped_by_height_amplitude() -> None:
    source = {
        "height": {"prediction": [1.0, 2.0, 1.0, 2.0], "target": [1.0, 2.0, 1.0, 2.0]},
        "height_error_magnitude_metres": [0.05, 0.10, 0.05, 0.10],
    }
    candidate = {
        "height": {"prediction": [0.9, 2.2, 0.9, 2.2]},
        "height_error_magnitude_metres": source["height_error_magnitude_metres"],
    }

    result = matched_height_response(candidate, source)

    assert result["all_scenes_nondegenerate"] is True
    grouped = result["by_height_magnitude_diagnostic"]
    assert grouped["0.05"]["candidate_to_source_ratio"] == pytest.approx(0.9)
    assert grouped["0.10"]["candidate_to_source_ratio"] == pytest.approx(1.1)
    assert [item["candidate_to_source_ratio"] for item in result["by_scene"]] == pytest.approx(
        [0.9, 1.1, 0.9, 1.1]
    )
