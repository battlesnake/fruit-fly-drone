from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_neural_integration_rate as audit  # noqa: E402


def _condition(substeps: int, contrast: list[float], output_value: float) -> dict:
    outputs = torch.full((len(contrast), 2, 4), output_value).tolist()
    return {
        "substeps": substeps,
        "prediction_contrasts": contrast,
        "terminal_motor_outputs": outputs,
        "all_recurrent_states_and_outputs_finite": True,
        "all_metrics_finite": True,
        "endpoint_image_difference_max": 0.0,
        "source_loaded_exactly": True,
        "source_restored": True,
    }


def test_protocol_changes_only_native_integration_rate() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["camera_and_action_hz"] == 50
    assert protocol["internal_substeps"] == [1, 2, 4, 8, 16]
    assert protocol["privileged_inputs"] == []
    assert protocol["external_history_or_state"] is False
    assert protocol["learning"] is False
    assert protocol["aircraft_advanced_during_substeps"] is False
    assert protocol["condition_selection"]["behavior_metrics_used_for_selection"] is False


def test_summary_preserves_case_order_and_pair_contrast_sign() -> None:
    outputs = torch.zeros(3, 2, 4)
    outputs[:, 0, 3] = torch.tensor((0.2, 0.4, 0.1))
    outputs[:, 1, 3] = torch.tensor((0.3, 0.1, 0.5))
    targets = torch.tensor((0.1, -0.3, 0.4))
    horizons = torch.tensor((15, 20, 25))

    summary = audit.summarize_outputs(
        outputs,
        targets,
        horizons,
        scale=0.2,
        recurrence_finite=True,
        endpoint_difference_max=0.0,
        wall_time_seconds=1.0,
        recurrent_calls=10,
        branch_state_updates=20,
    )

    assert torch.allclose(
        torch.tensor(summary["prediction_contrasts"]), torch.tensor((0.1, -0.3, 0.4))
    )
    assert summary["correct_sign_fraction"] == 1.0
    assert summary["nrmse"] < 1.0e-6
    assert summary["all_metrics_finite"] is True


def test_adjacent_refinement_selects_first_numerically_converged_rate() -> None:
    k1 = _condition(1, [0.0, 0.0], 0.0)
    k2 = _condition(2, [0.02, 0.02], 0.01)
    k4 = _condition(4, [0.0201, 0.0201], 0.0101)
    k8 = _condition(8, [0.02011, 0.02011], 0.01011)

    refinements = [
        audit.adjacent_refinement(k1, k2, scale=0.04),
        audit.adjacent_refinement(k2, k4, scale=0.04),
        audit.adjacent_refinement(k4, k8, scale=0.04),
    ]

    assert refinements[0]["pass"] is False
    assert refinements[1]["pass"] is True
    assert audit.select_numerically_adequate_rate(refinements) == 2


def test_adjacent_refinement_rejects_nonfinite_or_changed_source() -> None:
    coarse = _condition(1, [0.0], 0.0)
    fine = _condition(2, [0.0], 0.0)
    fine["source_restored"] = False

    result = audit.adjacent_refinement(coarse, fine, scale=0.04)

    assert result["finite_exact_and_source_restored"] is False
    assert result["pass"] is False


def test_k1_reproduction_checks_registered_metrics_and_endpoint() -> None:
    result = {
        **audit.EXPECTED_SOURCE,
        "endpoint_image_difference_max": 0.0,
        "all_recurrent_states_and_outputs_finite": True,
        "all_metrics_finite": True,
        "source_loaded_exactly": True,
        "source_restored": True,
    }

    assert audit.k1_reproduction_decision(result)["pass"] is True
    result["nrmse"] += 3.0e-5
    assert audit.k1_reproduction_decision(result)["pass"] is False
