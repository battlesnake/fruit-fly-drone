from __future__ import annotations

from pathlib import Path

import torch

from scripts.search_gate_assisted_motor_es import (
    assisted_progress,
    joint_progress,
    mass_lateral_safe,
    nominate_finalists,
    parameter_masks,
    select_validated_candidate,
)
from scripts.search_gate_motor_interface_es import load_controller, motor_interface_spec


def summary(overall: float, lower: float, higher: float, negative: float, positive: float):
    return {
        "success_rate": overall,
        "success_by_stratum": {
            "lower_mass": lower,
            "higher_mass": higher,
            "negative_lateral_offset": negative,
            "positive_lateral_offset": positive,
        },
    }


def test_assisted_motor_parameter_masks_are_disjoint_and_complete() -> None:
    controller, _, _, _, _ = load_controller(
        Path("artifacts/gate-accel-v2/connectome.npz"),
        Path("artifacts/gate-motor-interface-es-v1/controller.pt"),
        torch.device("cpu"),
    )
    spec = motor_interface_spec(
        controller,
        bias_scale=0.01,
        log_gain_scale=0.05,
        maximum_bias_delta=0.15,
        maximum_gain_ratio=3.0,
    )
    masks = parameter_masks(spec)
    assert int(masks["steering"].sum()) == 18
    assert int(masks["throttle"].sum()) == 6
    assert not bool((masks["steering"] & masks["throttle"]).any())
    assert bool(masks["joint"].all())


def test_assisted_and_joint_progress_require_floor_gain_and_safety() -> None:
    baseline = summary(0.20, 0.10, 0.30, 0.25, 0.15)
    assisted = summary(0.35, 0.22, 0.36, 0.35, 0.25)
    floor_flat = summary(0.40, 0.18, 0.55, 0.50, 0.24)
    unsafe = summary(0.50, 0.25, 0.60, 0.18, 0.40)
    assert mass_lateral_safe(baseline, assisted, maximum_drop=0.05)
    assert assisted_progress(
        baseline,
        assisted,
        minimum_floor_gain=0.10,
        maximum_drop=0.05,
    )
    assert not assisted_progress(
        baseline,
        floor_flat,
        minimum_floor_gain=0.10,
        maximum_drop=0.05,
    )
    assert not mass_lateral_safe(baseline, unsafe, maximum_drop=0.05)
    assert joint_progress(
        baseline,
        assisted,
        minimum_overall_gain=0.05,
        minimum_floor_gain=0.05,
        maximum_drop=0.05,
    )
    assert not joint_progress(
        baseline,
        unsafe,
        minimum_overall_gain=0.05,
        minimum_floor_gain=0.05,
        maximum_drop=0.05,
    )


def test_validation_nomination_keeps_current_progress_over_lucky_archive() -> None:
    def item(fitness: float):
        metrics = summary(0.20, 0.10, 0.30, 0.25, 0.15)
        metrics.update({"fitness": fitness, "crossing_radial_mean_m": 1.0})
        return {"development": metrics}

    archive = {"lucky": item(10.0), "current_center": item(1.0), "current_best": item(2.0)}
    nominated = nominate_finalists(
        archive,
        required_keys=("current_center", "current_best"),
        limit=3,
    )
    assert nominated == ["current_center", "current_best", "lucky"]


def test_joint_selection_prefers_gate_qualified_over_higher_floor_only() -> None:
    baseline = summary(0.20, 0.10, 0.30, 0.25, 0.15)
    qualified = summary(0.27, 0.16, 0.31, 0.27, 0.21)
    higher_floor_but_flat = summary(0.24, 0.30, 0.30, 0.30, 0.30)
    for metrics in (qualified, higher_floor_but_flat):
        metrics.update({"fitness": 0.0, "crossing_radial_mean_m": 1.0})
    selected, passed = select_validated_candidate(
        "joint",
        baseline,
        {
            "qualified": {"metrics": qualified},
            "higher_floor": {"metrics": higher_floor_but_flat},
        },
        assisted_minimum_floor_gain=0.10,
        joint_minimum_overall_gain=0.05,
        joint_minimum_floor_gain=0.05,
        maximum_drop=0.05,
    )
    assert selected == "qualified"
    assert passed
