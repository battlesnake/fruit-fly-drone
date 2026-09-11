from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_rk4_readout_capacity as fit  # noqa: E402


def _motor_outputs(*, common: float = 0.0, contrast: float = 0.0) -> list:
    output = torch.zeros(24, 2, 4)
    output[:, 0, 3] = common - contrast / 2
    output[:, 1, 3] = common + contrast / 2
    return output.tolist()


def _block(*, common: float = 0.0, contrast: float = 0.0) -> dict:
    return {
        "terminal_motor_outputs": _motor_outputs(common=common, contrast=contrast),
        "all_recurrent_states_and_outputs_finite": True,
        "all_metrics_finite": True,
        "endpoint_image_difference_max": 0.0,
    }


def test_protocol_freezes_data_ladder_mask_and_actor_contract() -> None:
    protocol = fit.protocol_manifest()

    assert protocol["protocol_commit"] == "3cda916"
    assert protocol["blocks"]["training_seeds"] == [450991, 450992, 450993, 450994]
    assert protocol["blocks"]["development_seeds"] == [460991, 460992]
    assert protocol["blocks"]["qualification_seeds"] == list(range(470991, 470999))
    assert protocol["ordinary_scales_descending"] == [
        16.0,
        8.0,
        4.0,
        2.0,
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
    ]
    assert protocol["readout_mask"]["edge_count"] == 493
    assert protocol["readout_mask"]["node_count"] == 7
    assert protocol["actor"]["external_or_engineered_state"] is False
    assert protocol["candidate_projection_or_repair"] is False


def test_formal_seed_guard_and_block_splits() -> None:
    fit.require_nonformal_seeds(1, 2, 3)
    with pytest.raises(ValueError):
        fit.require_nonformal_seeds(450992)

    assert len(fit.block_specs("training")) == 4
    assert all(not item["held_out_styles"] for item in fit.block_specs("training"))
    assert all(item["held_out_styles"] for item in fit.block_specs("development"))
    assert all(item["held_out_styles"] for item in fit.block_specs("qualification"))


def test_per_block_preservation_cannot_be_diluted() -> None:
    source = _block()
    passing = fit.block_preservation_decision(source, _block(common=0.004))
    failing = fit.block_preservation_decision(source, _block(common=0.006))

    assert passing["pass"] is True
    assert failing["pass"] is False
    assert "pair-common throttle RMS drift exceeded 0.005" in failing["reasons"]


def test_candidate_requires_current_progress_and_every_block_preserved() -> None:
    source_state = {
        "edge_magnitude": torch.tensor([1.0, 2.0, 3.0]),
        "bias": torch.tensor([0.0, 0.0]),
        "raw_time_constant": torch.tensor([0.0, 0.0]),
    }
    candidate_state = {name: value.clone() for name, value in source_state.items()}
    candidate_state["edge_magnitude"][0] += 0.1
    mask = {
        "edge_indices": np.array([0], dtype=np.int64),
        "node_indices": np.array([0], dtype=np.int64),
    }
    replay = {
        "nrmse": 0.998,
        "all_recurrent_states_and_outputs_finite": True,
        "all_metrics_finite": True,
    }
    decision = fit.candidate_decision(
        fixed=replay,
        full=replay,
        current_fixed={"nrmse": 1.0},
        current_full={"nrmse": 1.0},
        source_blocks={1: _block(), 2: _block()},
        candidate_blocks=[_block(common=0.004), _block(common=0.006)],
        manifests=[{"seed": 1}, {"seed": 2}],
        source_state=source_state,
        candidate_state=candidate_state,
        mask=mask,
    )

    assert decision["pass"] is False
    assert decision["fixed_nrmse_improvement_from_current"] == pytest.approx(0.002)
    assert decision["outside_mask_parameters_exact"] is True
    assert decision["block_decisions"][0]["pass"] is True
    assert decision["block_decisions"][1]["pass"] is False


def test_derivative_probe_requires_adjacent_passing_scales() -> None:
    no_window = fit.derivative_decision(
        [
            {"scale": 1 / 16, "pass": True, "numerical_failure": False},
            {"scale": 1 / 32, "pass": False, "numerical_failure": False},
            {"scale": 1 / 64, "pass": True, "numerical_failure": False},
        ]
    )
    passing = fit.derivative_decision(
        [
            {"scale": 1 / 16, "pass": False, "numerical_failure": False},
            {"scale": 1 / 32, "pass": True, "numerical_failure": False},
            {"scale": 1 / 64, "pass": True, "numerical_failure": False},
        ]
    )

    assert no_window["pass"] is False
    assert passing["pass"] is True
    assert passing["passing_adjacent_scale_windows"] == [[1 / 32, 1 / 64]]


def test_milestone_and_terminal_behavior_gates() -> None:
    horizons = {
        str(value): {
            "nrmse": 0.4,
            "correct_sign_fraction": 1.0,
            "teacher_aligned_gain": 1.0,
        }
        for value in fit.motion.HISTORY_LENGTHS
    }
    source_horizons = {
        key: {**value, "nrmse": 0.9} for key, value in horizons.items()
    }
    training = {
        "combined": {
            "source_absolute_nrmse_improvement": 0.03,
            "nrmse": 0.4,
            "correct_sign_fraction": 1.0,
            "teacher_aligned_gain": 1.0,
            "by_horizon": horizons,
            "all_recurrent_states_and_outputs_finite": True,
            "all_metrics_finite": True,
            "endpoint_image_difference_max": 0.0,
        },
        "all_preservation_pass": True,
    }
    development = {
        "combined": {
            **training["combined"],
            "source_absolute_nrmse_improvement": 0.02,
        },
        "source_combined": {"by_horizon": source_horizons},
        "all_preservation_pass": True,
    }

    assert fit.milestone_decision(training, development)["pass"] is True
    assert fit.terminal_behavior_decision(training, cohort="training")["pass"] is True

    development["combined"]["by_horizon"]["15"]["teacher_aligned_gain"] = -0.1
    assert fit.terminal_behavior_decision(development, cohort="development")["pass"] is False
