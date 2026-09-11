from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_rk4_throttle_readout_step as audit  # noqa: E402


def _record() -> dict:
    return {
        "fixed_nrmse_improvement": 0.002,
        "full_nrmse_improvement": 0.002,
        "preservation": {
            "pair_common_throttle_drift_rms": 0.001,
            "pair_common_throttle_drift_maximum_absolute": 0.002,
            "rpy_drift": {
                name: {"rms": 0.001, "maximum_absolute": 0.002} for name in ("roll", "pitch", "yaw")
            },
        },
        "fixed": {
            "all_recurrent_states_and_outputs_finite": True,
            "all_metrics_finite": True,
        },
        "full": {
            "all_recurrent_states_and_outputs_finite": True,
            "all_metrics_finite": True,
            "endpoint_image_difference_max": 0.0,
        },
        "maximum_motor_absolute": 0.5,
        "native_bounds_pass": True,
        "outside_mask_parameters_exact": True,
        "candidate_state_loaded_and_restored": True,
    }


def test_protocol_is_native_last_hop_only() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["parameter_mask"]["incoming_throttle_motor_edge_magnitudes"] == 493
    assert protocol["parameter_mask"]["throttle_motor_biases_and_time_constants"] == 7
    assert protocol["external_or_engineered_state"] is False
    assert protocol["objective"] == "frozen-scale paired endpoint throttle-contrast MSE only"
    assert protocol["candidate_retained"] is False


def test_real_graph_mask_matches_preregistered_identity() -> None:
    mask = audit.build_readout_mask(
        Path(__file__).resolve().parents[1] / "data/derived/full-visual-connectome-v1.npz"
    )

    assert mask["manifest"]["edge_count"] == 493
    assert mask["manifest"]["node_count"] == 7
    assert mask["manifest"]["positive_edges"] == 314
    assert mask["manifest"]["negative_edges"] == 179


def test_candidate_gate_requires_motion_and_preservation() -> None:
    passing = _record()

    assert audit.candidate_decision(passing)["pass"] is True
    common_failure = _record()
    common_failure["preservation"]["pair_common_throttle_drift_rms"] = 0.006
    assert audit.candidate_decision(common_failure)["pass"] is False
    motion_failure = _record()
    motion_failure["full_nrmse_improvement"] = 0.0009
    assert audit.candidate_decision(motion_failure)["pass"] is False


def test_materialization_changes_only_selected_family() -> None:
    source = {
        "edge_magnitude": torch.tensor((0.0, 1.0)),
        "bias": torch.tensor((0.0, 1.0)),
        "raw_time_constant": torch.tensor((0.0, 1.0)),
        "buffer": torch.tensor((3.0,)),
    }
    pending = {
        **source,
        "edge_magnitude": torch.tensor((-1.0, 3.0)),
        "bias": torch.tensor((2.0, 3.0)),
        "raw_time_constant": torch.tensor((2.0, 3.0)),
    }

    candidate = audit.materialize_candidate(
        source, pending, scale=0.5, families=("edge_magnitude",)
    )

    assert torch.equal(candidate["edge_magnitude"], torch.tensor((0.0, 2.0)))
    assert torch.equal(candidate["bias"], source["bias"])
    assert torch.equal(candidate["raw_time_constant"], source["raw_time_constant"])
    assert torch.equal(candidate["buffer"], source["buffer"])
