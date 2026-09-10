from __future__ import annotations

import math
from pathlib import Path

import torch

from scripts.audit_gate_throttle_exploration_timescale import (
    ar1_coefficient,
    evaluate_mode,
    exploration_gate,
)
from scripts.gate_diverse_cases import diverse_matched_cases
from scripts.search_gate_motor_interface_es import load_controller

REPO_ROOT = Path(__file__).resolve().parents[1]


def result(*, light: float, heavy: float) -> dict[str, float]:
    return {"light_success_rate": light, "heavy_success_rate": heavy}


def test_ar1_coefficient_matches_declared_correlation_time() -> None:
    rho = ar1_coefficient(0.01, 0.20)
    assert math.isclose(rho, math.exp(-0.05))
    assert math.isclose(rho**20, math.exp(-1.0))


def test_exploration_gate_requires_both_correlated_seeds() -> None:
    independent = [result(light=0.0, heavy=0.70), result(light=0.0, heavy=0.75)]
    correlated = [result(light=0.12, heavy=0.62), result(light=0.10, heavy=0.66)]
    gate = exploration_gate(
        independent,
        correlated,
        minimum_light_success=0.10,
        maximum_heavy_drop=0.10,
    )
    assert gate["passed"]
    correlated[1]["light_success_rate"] = 0.09
    assert not exploration_gate(
        independent,
        correlated,
        minimum_light_success=0.10,
        maximum_heavy_drop=0.10,
    )["passed"]


def test_exploration_gate_rejects_heavy_regression() -> None:
    gate = exploration_gate(
        [result(light=0.0, heavy=0.75)],
        [result(light=0.20, heavy=0.64)],
        minimum_light_success=0.10,
        maximum_heavy_drop=0.10,
    )
    assert not gate["passed"]


def test_short_noisy_modes_measure_foreleg_excursions() -> None:
    device = torch.device("cpu")
    controller, _, hover_config, gate_config, resolution = load_controller(
        REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
        REPO_ROOT / "artifacts" / "gate-motor-interface-es-v1" / "controller.pt",
        device,
    )
    cases = diverse_matched_cases(
        8,
        seed=93_001,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
        extreme_fraction=0.5,
    )
    common = {
        "controller": controller,
        "cases": cases,
        "sigma": 0.03,
        "correlation_seconds": 0.20,
        "seconds": 0.08,
        "takeover_seconds": 0.04,
        "resolution": resolution,
        "hover_config": hover_config,
        "gate_config": gate_config,
    }
    deterministic, deterministic_trace = evaluate_mode(**common, mode="deterministic", seed=0)
    independent, independent_trace = evaluate_mode(
        **common,
        mode="independent",
        seed=93_002,
        deterministic_throttle_trace=deterministic_trace,
    )
    correlated, correlated_trace = evaluate_mode(
        **common,
        mode="correlated",
        seed=93_002,
        deterministic_throttle_trace=deterministic_trace,
    )
    assert deterministic_trace.shape == independent_trace.shape == correlated_trace.shape
    assert deterministic["throttle_motor_perturbation_rms"] == 0.0
    assert independent["throttle_motor_perturbation_rms"] > 0.0
    assert correlated["throttle_motor_perturbation_rms"] > 0.0
    assert independent["foreleg_throttle_rms_difference_from_deterministic"] > 0.0
    assert correlated["foreleg_throttle_rms_difference_from_deterministic"] > 0.0
