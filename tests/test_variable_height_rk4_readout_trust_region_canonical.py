from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_rk4_readout_trust_region_canonical as audit  # noqa: E402


def test_protocol_canonicalizes_before_new_scales() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["direction_archive"]["persist_before_candidate_replay"] is True
    assert protocol["direction_archive"]["cpu_reload_bit_exact"] is True
    assert protocol["old_scale_one_is_reproduction_only"] is True
    assert protocol["new_scales_descending"] == [16.0, 8.0, 4.0, 2.0]


def test_archive_round_trip_identity_and_family_installation() -> None:
    source = {
        "edge_magnitude": torch.tensor((1.0, 2.0)),
        "bias": torch.tensor((3.0,)),
        "raw_time_constant": torch.tensor((4.0,)),
        "buffer": torch.tensor((5.0,)),
    }
    pending = {
        **source,
        "edge_magnitude": torch.tensor((1.1, 2.1)),
        "bias": torch.tensor((3.1,)),
        "raw_time_constant": torch.tensor((4.1,)),
    }
    archive = audit.archive_payload(
        source,
        pending,
        {"state": {}},
        {"state": {0: {"moment": torch.tensor((1.0, 2.0))}}},
    )

    audit.validate_archive(archive)
    installed = audit.install_parameter_families(source, archive["pending_parameters"])

    assert torch.equal(installed["edge_magnitude"], pending["edge_magnitude"])
    assert torch.equal(installed["buffer"], source["buffer"])
    assert archive["optimizer_after"]["state"][0]["moment"].device.type == "cpu"


def test_scalar_reproduction_uses_tolerance_and_counters() -> None:
    prior = {
        "baseline_fixed_objectives": [1.0, 1.0, 1.0],
        "gradient_objective": 1.0,
        "full_proposal_directional_derivative": -0.01,
    }
    passing = audit.scalar_reproduction_decision(
        prior,
        baseline_values=[1.0, 1.0, 1.0],
        gradient_objective=1.0,
        directional_derivative=-0.01,
        optimizer_counters_before=[],
        optimizer_counters_after=[1.0, 1.0, 1.0],
    )
    bad_counters = audit.scalar_reproduction_decision(
        prior,
        baseline_values=[1.0, 1.0, 1.0],
        gradient_objective=1.0,
        directional_derivative=-0.01,
        optimizer_counters_before=[],
        optimizer_counters_after=[2.0, 2.0, 2.0],
    )

    assert passing["pass"] is True
    assert bad_counters["pass"] is False
