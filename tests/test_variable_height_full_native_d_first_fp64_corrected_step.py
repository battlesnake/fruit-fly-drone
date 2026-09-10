from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_corrected_step as step  # noqa: E402


def _metrics(damping: float, *, preserved: bool = True) -> dict[str, object]:
    offset = 0.0 if preserved else 1.0
    return {
        "component_nrmse": {"common": offset, "height": offset},
        "by_supervision_step_nrmse": {
            str(horizon): {"common": offset, "height": offset}
            for horizon in step.joint.SUPERVISION_STEPS
        },
        "rpy_source_nrmse": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "motor_output_max_absolute": 0.5,
        "all_attitude_cache_states_valid": True,
        "endpoint_damping_nrmse": damping,
    }


def test_development_transfer_requires_update20_improvement_and_source_preservation() -> None:
    source = _metrics(2.0)
    current = _metrics(1.5)

    passing = step.development_transfer_decision(source, current, _metrics(1.498))
    no_improvement = step.development_transfer_decision(source, current, _metrics(1.5))
    preservation_failure = step.development_transfer_decision(
        source, current, _metrics(1.498, preserved=False)
    )

    assert passing["pass"] is True
    assert no_improvement["pass"] is False
    assert preservation_failure["pass"] is False


def test_parameter_tensor_comparison_requires_exact_values_dtype_and_shape() -> None:
    expected = {
        name: torch.tensor([1.0], dtype=torch.float32) for name in step.joint.PARAMETER_FAMILIES
    }
    same = {name: value.clone() for name, value in expected.items()}
    different = {name: value.clone() for name, value in expected.items()}
    different["bias"][0] = 2.0

    assert step.parameter_tensor_comparison(same, expected)["pass"] is True
    assert step.parameter_tensor_comparison(different, expected)["pass"] is False


def test_selection_preconditions_fail_closed_on_input_controls() -> None:
    assert (
        step.selection_preconditions_pass(
            input_controls_pass=True,
            transaction_controls_pass=True,
            projection_controls_pass=True,
        )
        is True
    )
    assert (
        step.selection_preconditions_pass(
            input_controls_pass=False,
            transaction_controls_pass=True,
            projection_controls_pass=True,
        )
        is False
    )


def test_repair_numerical_failure_is_a_control_failure() -> None:
    repair = {
        "attempted": True,
        "starting_candidate": {
            "canonical_parameter_idempotence": {"pass": True},
            "parameter_bounds": {"pass": True},
        },
        "pass_before_nonlinear_replay": False,
        "endpoint_damping_directional_finite_difference": {"pass": False},
        "projection": {"pass": True},
        "authoritative_parameter_canonicalization": {"pass": True},
        "effective_full_correction_parameter_bounds": {"pass": True},
        "optimizer_pending_state": {"pass": True},
        "correction_trials": [],
    }

    report = step.repair_numerical_control_report([{"repair": repair}])

    assert report["repair_path_present"] is True
    assert report["pass"] is False
    assert report["checks"]["directional_finite_difference"] is False


def test_valid_nonlinear_repair_rejection_is_not_a_control_failure() -> None:
    repair = {
        "attempted": True,
        "starting_candidate": {
            "canonical_parameter_idempotence": {"pass": True},
            "parameter_bounds": {"pass": True},
        },
        "pass_before_nonlinear_replay": True,
        "endpoint_damping_directional_finite_difference": {"pass": True},
        "projection": {"pass": True},
        "authoritative_parameter_canonicalization": {"pass": True},
        "effective_full_correction_parameter_bounds": {"pass": True},
        "optimizer_pending_state": {"pass": True},
        "correction_trials": [
            {
                "linearized_constraint_violations_finite": True,
                "canonical_parameter_idempotence": {"pass": True},
                "parameter_bounds": {"pass": True},
                "decision": {"pass": False, "reasons": ["nonlinear C limit"]},
            }
        ],
    }

    assert step.repair_numerical_control_report([{"repair": repair}])["pass"] is True


def test_corrected_trial_runtime_restores_module_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    originals = (
        step.base.install_trial,
        step.corrected._GUARD_REPORT,
        step.corrected._UPDATE_NUMBER,
        step.corrected._PENDING_OPTIMIZER,
        step.corrected._OPTIMIZER_BEFORE_PROPOSAL,
        step.corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
    )

    with pytest.raises(RuntimeError, match="injected"):
        with step.corrected_trial_runtime(optimizer, optimizer_before, "after"):
            assert step.base.install_trial is step.canonical.install_trial
            assert step.corrected._UPDATE_NUMBER == 21
            assert step.corrected._PENDING_OPTIMIZER is optimizer
            raise RuntimeError("injected")

    restored = (
        step.base.install_trial,
        step.corrected._GUARD_REPORT,
        step.corrected._UPDATE_NUMBER,
        step.corrected._PENDING_OPTIMIZER,
        step.corrected._OPTIMIZER_BEFORE_PROPOSAL,
        step.corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256,
    )
    assert restored == originals


def test_protocol_is_one_restored_nonpromotional_update() -> None:
    protocol = step.protocol_manifest()

    assert protocol["protocol_commit"] == "d6fa381"
    assert protocol["archived_jacobians_regenerated"] is False
    assert protocol["optimizer_transaction"]["adam_steps"] == 1
    assert protocol["ordinary_scales_descending"] == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    assert protocol["repair"]["arithmetic_unchanged"] is True
    assert protocol["development"]["fresh_data"] is False
    assert protocol["candidate_retained"] is False
    assert protocol["promotion"] is False
