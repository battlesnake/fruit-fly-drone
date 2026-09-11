from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_rk4_premotor_tau as audit  # noqa: E402


def test_protocol_freezes_tau_mask_solver_and_held_out_order() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["protocol_commit"] == "c318a1f"
    assert protocol["corrected_replay"]["failed_report_sha256"] == (
        audit.EXPECTED_FAILED_REPORT_SHA256
    )
    assert protocol["mask"]["raw_time_constants_only"] is True
    assert protocol["mask"]["selected_nonmotor_nodes"] == 883
    assert protocol["physical_tau_direction"]["source_ratio_range"] == [0.8, 1.25]
    assert protocol["development_generated_only_after_training_and_solver_pass"] is True
    assert protocol["external_or_engineered_state"] is False
    assert protocol["candidate_retained"] is False


def test_capped_direction_has_registered_rms_and_component_cap() -> None:
    gradient = torch.ones(audit.SELECTED_NODES)
    gradient[:50] = 10.0
    direction, controls = audit.capped_unit_direction(-gradient)

    assert float(direction.square().mean().sqrt()) == pytest.approx(0.001)
    assert float(direction.abs().max()) <= 0.002
    assert controls["components_at_cap"] > 0


def test_tau_candidate_obeys_per_cell_ratio_and_freezes_other_values() -> None:
    source = {
        "edge_magnitude": torch.tensor([1.0, 2.0]),
        "bias": torch.tensor([3.0, 4.0, 5.0]),
        "raw_time_constant": audit.raw_from_tau(
            torch.tensor([0.011, 0.020, 0.249], dtype=torch.float64)
        ).float(),
    }
    nodes = np.array([0, 2], dtype=np.int64)
    direction = torch.tensor([-0.002, 0.002], dtype=torch.float64)
    candidate, controls = audit.tau_candidate(
        source, nodes, direction, scale=4.0
    )

    assert controls["ratio_bounds_pass"] is True
    assert controls["native_bounds_pass"] is True
    assert controls["components_clamped_lower"] == 1
    assert controls["components_clamped_upper"] == 1
    assert audit.frozen_parameter_controls(source, candidate, nodes)["pass"] is True


def test_directional_agreement_uses_actual_raw_displacement() -> None:
    source = {
        "raw_time_constant": torch.tensor([0.0, 1.0, 2.0]),
    }
    probe = {"raw_time_constant": torch.tensor([-0.1, 1.0, 1.8])}
    result = audit.directional_agreement(
        baseline_objectives=[1.0, 1.0, 1.0],
        probe_objective=0.95,
        raw_gradient_selected=torch.tensor([0.1, 0.2]),
        source_state=source,
        probe_state=probe,
        node_indices=np.array([0, 2], dtype=np.int64),
    )

    assert result["predicted_objective_change"] == pytest.approx(-0.05)
    assert result["pass"] is True


def test_combined_motor_bound_is_derived_from_terminal_outputs() -> None:
    assert audit.maximum_motor_absolute(
        {"terminal_motor_outputs": [[[0.0, -0.25, 0.5, -0.75]]]}
    ) == pytest.approx(0.75)


def test_development_rejects_tiny_over_tolerance_horizon_regression() -> None:
    source_horizons = {
        str(value): {"nrmse": 1.0} for value in audit.motion.HISTORY_LENGTHS
    }
    candidate_horizons = {
        key: {"nrmse": 0.9} for key in source_horizons
    }
    evaluation = {
        "combined": {
            "source_absolute_nrmse_improvement": 0.001,
            "by_horizon": candidate_horizons,
            "all_recurrent_states_and_outputs_finite": True,
            "all_metrics_finite": True,
            "endpoint_image_difference_max": 0.0,
        },
        "source_combined": {"by_horizon": source_horizons},
        "blocks": [
            {
                "source_loaded_exactly": True,
                "source_restored": True,
                "all_recurrent_states_and_outputs_finite": True,
                "all_metrics_finite": True,
                "endpoint_image_difference_max": 0.0,
            }
        ],
        "all_preservation_pass": True,
    }
    assert audit.development_decision(evaluation)["pass"] is True

    candidate_horizons[str(audit.motion.HISTORY_LENGTHS[0])]["nrmse"] = 1.00002
    assert audit.development_decision(evaluation)["pass"] is False


def test_invalid_source_reference_and_interrupted_development_fail_closed(
    tmp_path: Path,
) -> None:
    invalid = {
        "combined": {
            "nrmse": float("nan"),
            "all_recurrent_states_and_outputs_finite": True,
            "all_metrics_finite": True,
            "endpoint_image_difference_max": 0.0,
        },
        "blocks": [],
    }
    assert audit.source_references_valid(invalid) is False

    marker = tmp_path / "development-started.json"
    marker.write_text("{}")
    with pytest.raises(RuntimeError, match="refusing seed reuse"):
        audit.run_development(
            Namespace(output_dir=tmp_path),
            {},
            {},
            candidate_sha256="candidate",
            device=torch.device("cpu"),
        )
