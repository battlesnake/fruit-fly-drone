from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_nonlinear_correction as audit  # noqa: E402
import train_variable_height_full_native_d_first_canonical as canonical  # noqa: E402


class TinyController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = nn.Parameter(torch.tensor([0.5, 8.0]))
        self.bias = nn.Parameter(torch.tensor([1.0]))
        self.raw_time_constant = nn.Parameter(torch.tensor([-3.0]))

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.edge_magnitude.clamp_(0.0, 8.0)


def _parameters(controller: TinyController) -> dict[str, torch.Tensor]:
    return {
        name: getattr(controller, name).detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }


def _metrics(
    damping: float,
    *,
    common_endpoint: float = 1.0,
    component: float = 1.0,
    pitch: float = 0.0,
) -> dict:
    return {
        "joint_normalized_mse": 1.0,
        "component_nrmse": {
            "common": component,
            "height": component,
            "damping": damping,
            "interaction": 0.0,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": common_endpoint if step == 25 else component,
                "height": component,
                "damping": damping,
                "interaction": 0.0,
            }
            for step in (15, 20, 25)
        },
        "rpy_source_nrmse": {"roll": 0.01, "pitch": pitch, "yaw": 0.01},
        "endpoint_damping_nrmse": damping,
        "endpoint_damping_correct_sign_fraction": 0.0,
        "endpoint_damping_teacher_aligned_gain": 0.0,
        "motor_output_max_absolute": 0.7,
        "all_attitude_cache_states_valid": True,
    }


def test_correction_specs_use_three_distinct_references_and_squared_limits() -> None:
    source = _metrics(1.45, common_endpoint=1.66)
    update8 = _metrics(1.376, common_endpoint=1.679)
    starting = _metrics(1.374, common_endpoint=1.684, pitch=0.045)

    specs = audit.correction_constraint_specs(source, update8, starting)
    common = next(spec for spec in specs if spec["name"] == "common.step_25")
    damping = next(spec for spec in specs if spec["name"] == "damping.step_25.retention")

    assert common["current_mse"] == pytest.approx(1.684**2)
    assert common["limit_mse"] == pytest.approx((1.66 + 0.0199) ** 2)
    assert damping["current_mse"] == pytest.approx(1.374**2)
    assert damping["limit_mse"] == pytest.approx((1.376 - 0.001) ** 2)
    assert [spec["name"] for spec in specs if spec["kind"] == "attitude"] == ["rpy.pitch"]


def test_correction_may_worsen_starting_damping_while_retaining_update8_gain() -> None:
    source = _metrics(1.45, common_endpoint=1.66)
    update8 = _metrics(1.376, common_endpoint=1.679)
    corrected = _metrics(1.3749, common_endpoint=1.6798)

    decision = audit.correction_decision(
        source,
        update8,
        corrected,
        maximum_linearized_violation=0.0,
        canonicalization_pass=True,
        parameter_bounds_pass=True,
    )

    assert corrected["endpoint_damping_nrmse"] > 1.374
    assert decision["pass"] is True
    assert decision["endpoint_damping_nrmse_improvement_from_update_8"] == pytest.approx(0.0011)


def test_scaled_correction_linear_feasibility_is_checked_independently() -> None:
    specs = [{"current_mse": 1.0, "limit_mse": 0.8}]
    rows = [
        {
            "edge_magnitude": torch.tensor([1.0]),
            "bias": torch.tensor([0.0]),
            "raw_time_constant": torch.tensor([0.0]),
        }
    ]
    full = {
        "edge_magnitude": torch.tensor([-0.2]),
        "bias": torch.tensor([0.0]),
        "raw_time_constant": torch.tensor([0.0]),
    }
    half = {name: 0.5 * value for name, value in full.items()}

    assert max(audit.linearized_constraint_violations(specs, rows, full)) <= 1.0e-6
    assert max(audit.linearized_constraint_violations(specs, rows, half)) > 1.0e-6


@pytest.mark.parametrize("violations", [[float("nan"), -1.0], [-1.0, float("nan")]])
def test_nonfinite_linearized_violation_fails_closed(violations: list[float]) -> None:
    maximum, finite = audit.maximum_linearized_violation(violations)

    assert finite is False
    assert maximum == float("inf")


def test_explicit_trial_installation_does_not_use_stale_canonical_globals() -> None:
    controller = TinyController()
    starting = _parameters(controller)
    corrected = {
        "edge_magnitude": torch.tensor([0.25, 7.5]),
        "bias": torch.tensor([1.5]),
        "raw_time_constant": torch.tensor([-2.5]),
    }
    canonical._AUTHORITATIVE_BASE = starting
    canonical._AUTHORITATIVE_CANDIDATE = {
        name: torch.full_like(value, 7.0) for name, value in starting.items()
    }

    actual, displacement, idempotence = audit.install_authoritative_trial(
        controller, starting, corrected, scale=0.5
    )

    assert actual["edge_magnitude"] == pytest.approx(torch.tensor([0.375, 7.75]))
    assert displacement["bias"] == pytest.approx(torch.tensor([0.25]).double())
    assert idempotence["pass"] is True


def test_transaction_restores_parameters_and_optimizer_after_exception() -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    sum(parameter.sum() for parameter in controller.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameters_before = _parameters(controller)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    starting_global_base = canonical._AUTHORITATIVE_BASE
    starting_global_candidate = canonical._AUTHORITATIVE_CANDIDATE

    with pytest.raises(RuntimeError, match="injected"):
        with audit.transactional_restoration(controller, optimizer):
            with torch.no_grad():
                controller.bias.add_(10.0)
            sum(parameter.sum() for parameter in controller.parameters()).backward()
            optimizer.step()
            canonical._AUTHORITATIVE_BASE = {"poison": torch.tensor(1.0)}
            canonical._AUTHORITATIVE_CANDIDATE = {"poison": torch.tensor(2.0)}
            raise RuntimeError("injected")

    assert all(
        torch.equal(getattr(controller, name).detach(), value)
        for name, value in parameters_before.items()
    )
    assert audit.trees_equal(optimizer.state_dict(), optimizer_before)
    assert canonical._AUTHORITATIVE_BASE is starting_global_base
    assert canonical._AUTHORITATIVE_CANDIDATE is starting_global_candidate


def test_gradient_agreement_detects_matching_and_mismatching_endpoint_rows() -> None:
    reference = {
        "edge_magnitude": torch.tensor([1.0, 2.0]),
        "bias": torch.tensor([3.0]),
        "raw_time_constant": torch.tensor([4.0]),
    }
    matching = {name: value.clone() for name, value in reference.items()}
    mismatching = {name: value.clone() for name, value in reference.items()}
    mismatching["bias"].add_(1.0e-3)

    assert audit.gradient_agreement(matching, reference)["pass"] is True
    assert audit.gradient_agreement(mismatching, reference)["pass"] is False


def test_protocol_freezes_one_solve_no_relinearization_and_no_retention() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["protocol_commit"] == "236671c"
    assert protocol["starting_candidate"]["scale"] == pytest.approx(0.0625)
    assert protocol["correction"]["single_solve"] is True
    assert protocol["correction"]["relinearization"] is False
    assert protocol["correction"]["scales_descending"] == [1.0, 0.5, 0.25, 0.125]
    assert protocol["corrected_candidate_retained"] is False
