from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_continuous_cns_solver as solver  # noqa: E402


def _result(value: float) -> dict:
    outputs = torch.full((2, 2, 4), value).tolist()
    return {
        "prediction_contrasts": [value, value],
        "terminal_motor_outputs": outputs,
        "source_loaded_exactly": True,
        "source_restored": True,
        "all_recurrent_states_and_outputs_finite": True,
        "all_metrics_finite": True,
        "endpoint_image_difference_max": 0.0,
    }


def test_protocol_never_selects_on_behavior() -> None:
    protocol = solver.protocol_manifest()

    assert protocol["learning"] is False
    assert protocol["new_actor_inputs_or_external_state"] is False
    assert protocol["selection"]["behavior_sign_score_or_runtime_used"] is False
    assert protocol["training_execution_authorized"] is False


def test_classical_rk4_matches_scalar_exponential() -> None:
    state = torch.tensor([1.0], dtype=torch.float64)

    actual = solver.classical_rk4_step(lambda value: -value, state, 0.2)

    assert abs(float(actual) - math.exp(-0.2)) < 3.0e-6


def test_reference_comparison_applies_both_registered_limits() -> None:
    reference = _result(0.0)
    passing = solver.reference_comparison(
        "candidate", _result(0.0002), "reference", reference, scale=0.04
    )
    contrast_failure = solver.reference_comparison(
        "candidate", _result(0.0005), "reference", reference, scale=0.04
    )
    motor_failure = solver.reference_comparison(
        "candidate", _result(0.006), "reference", reference, scale=1.0
    )

    assert passing["pass"] is True
    assert contrast_failure["pass"] is False
    assert motor_failure["pass"] is False


def test_candidate_selection_uses_graph_evaluations_and_exponential_tie_break() -> None:
    candidates = [
        {
            "name": "rk4_m2",
            "method": "rk4",
            "graph_evaluations_per_camera_frame": 8,
            "comparison": {"pass": True},
        },
        {
            "name": "rk4_m4",
            "method": "rk4",
            "graph_evaluations_per_camera_frame": 16,
            "comparison": {"pass": True},
        },
        {
            "name": "exponential_euler_k8",
            "method": "exponential_euler",
            "graph_evaluations_per_camera_frame": 8,
            "comparison": {"pass": True},
        },
    ]

    selected = solver.select_candidate(candidates)

    assert selected is not None
    assert selected["name"] == "exponential_euler_k8"
