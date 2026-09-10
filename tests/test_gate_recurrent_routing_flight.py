from __future__ import annotations

import torch

from scripts.audit_gate_recurrent_routing_flight import (
    candidate_progress,
    paired_success_difference,
    positive_control_passes,
)


def test_positive_control_requires_both_mass_halves() -> None:
    passing = {"success_by_stratum": {"lower_mass": 0.91, "higher_mass": 0.95}}
    failing = {"success_by_stratum": {"lower_mass": 0.89, "higher_mass": 1.0}}
    assert positive_control_passes(passing, minimum=0.90)
    assert not positive_control_passes(failing, minimum=0.90)


def test_paired_difference_clusters_adjacent_mass_cases() -> None:
    source = torch.tensor([False, True, False, False])
    candidate = torch.tensor([True, True, False, True])
    result = paired_success_difference(candidate, source)
    assert result["mean"] == 0.5
    assert result["independent_geometry_clusters"] == 2
    assert result["episodes_per_cluster"] == 2


def test_candidate_progress_requires_light_and_floor_gains_without_regression() -> None:
    source = {
        "success_by_stratum": {
            "lower_mass": 0.40,
            "higher_mass": 0.35,
            "negative_lateral_offset": 0.30,
            "positive_lateral_offset": 0.45,
        }
    }
    candidate = {
        "success_by_stratum": {
            "lower_mass": 0.52,
            "higher_mass": 0.45,
            "negative_lateral_offset": 0.40,
            "positive_lateral_offset": 0.48,
        }
    }
    assert candidate_progress(
        source,
        candidate,
        minimum_light_gain=0.10,
        minimum_floor_gain=0.10,
        maximum_drop=0.05,
    )
    candidate["success_by_stratum"]["lower_mass"] = 0.49
    assert not candidate_progress(
        source,
        candidate,
        minimum_light_gain=0.10,
        minimum_floor_gain=0.10,
        maximum_drop=0.05,
    )
