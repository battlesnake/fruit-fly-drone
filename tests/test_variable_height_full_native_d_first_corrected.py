from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_full_native_d_first as base  # noqa: E402
import train_variable_height_full_native_d_first_corrected as corrected  # noqa: E402


class TinyController(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = nn.Parameter(torch.tensor([0.5, 1.0]))
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.raw_time_constant = nn.Parameter(torch.tensor([-3.0]))

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.edge_magnitude.clamp_(0.0, 8.0)


def _metrics(
    damping: float,
    *,
    common: float = 1.0,
    common_endpoint: float | None = None,
    pitch: float = 0.01,
) -> dict:
    endpoint_common = common if common_endpoint is None else common_endpoint
    return {
        "joint_normalized_mse": 1.0,
        "component_nrmse": {
            "common": common,
            "height": common,
            "damping": damping,
            "interaction": 0.0,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": endpoint_common if step == 25 else common,
                "height": common,
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


def _optimizer_with_one_step() -> tuple[TinyController, torch.optim.Optimizer]:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)
    sum(parameter.sum() for parameter in controller.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return controller, optimizer


def test_protocol_freezes_total_budget_current_relative_repair_and_crash_safety() -> None:
    protocol = corrected.protocol_manifest()
    repair = protocol["nonlinear_repair"]

    assert protocol["protocol_commit"] == "4e9f646"
    assert protocol["starting_accepted_update"] == 8
    assert protocol["maximum_accepted_updates"] == 200
    assert protocol["development_interval_accepted_updates"] == 10
    assert repair["only_eligible_proposal_scale"] == pytest.approx(0.0625)
    assert repair["endpoint_damping_reference"] == "current accepted controller"
    assert repair["correction_scales_descending"] == [1.0, 0.5, 0.25, 0.125]
    assert repair["second_solve"] is False
    assert protocol["crash_safe_lifecycle"]["interrupted_qualification_retried"] is False


def test_repair_eligibility_allows_only_endpoint_common_step_25_failure() -> None:
    source = _metrics(1.45, common=1.0)
    current = _metrics(1.376, common=1.019)
    candidate = _metrics(1.374, common=1.019, common_endpoint=1.021)

    eligible = corrected.repair_eligibility(
        source,
        current,
        candidate,
        damping_directional_derivative=-1.0,
        projection_controls_pass=True,
        parameter_bounds_pass=True,
        canonicalization_pass=True,
        registered_reproduction_pass=True,
    )
    assert eligible["pass"] is True

    candidate["rpy_source_nrmse"]["pitch"] = 0.051
    rejected = corrected.repair_eligibility(
        source,
        current,
        candidate,
        damping_directional_derivative=-1.0,
        projection_controls_pass=True,
        parameter_bounds_pass=True,
        canonicalization_pass=True,
        registered_reproduction_pass=True,
    )
    assert rejected["pass"] is False
    assert "rpy.pitch preservation failed" in rejected["reasons"]


def test_repair_eligibility_rejects_numerical_or_gradient_failures() -> None:
    source = _metrics(1.45)
    current = _metrics(1.376)
    candidate = _metrics(1.374, common_endpoint=1.021)

    result = corrected.repair_eligibility(
        source,
        current,
        candidate,
        damping_directional_derivative=0.0,
        projection_controls_pass=False,
        parameter_bounds_pass=True,
        canonicalization_pass=True,
        registered_reproduction_pass=True,
    )

    assert result["pass"] is False
    assert "proposal projection or numerical control failed" in result["reasons"]
    assert any("descent direction" in reason for reason in result["reasons"])


def test_optimizer_transaction_requires_every_adam_counter_to_advance_once() -> None:
    controller, optimizer = _optimizer_with_one_step()
    before = copy.deepcopy(optimizer.state_dict())
    sum(parameter.sum() for parameter in controller.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = copy.deepcopy(optimizer.state_dict())

    passed = corrected.optimizer_step_transaction(before, after)
    unchanged = corrected.optimizer_step_transaction(before, before)

    assert passed["pass"] is True
    assert all(item["increment"] == 1.0 for item in passed["parameter_step_counters"])
    assert unchanged["pass"] is False


def test_rejected_transaction_restores_parameters_and_optimizer_exactly() -> None:
    controller, optimizer = _optimizer_with_one_step()
    current = {
        name: getattr(controller, name).detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    corrected._PENDING_OPTIMIZER = optimizer
    corrected._OPTIMIZER_BEFORE_PROPOSAL = optimizer_before
    with torch.no_grad():
        controller.bias.add_(2.0)
    sum(parameter.sum() for parameter in controller.parameters()).backward()
    optimizer.step()

    report = corrected._restore_rejected_transaction(controller, current)

    assert report["pass"] is True
    assert all(
        torch.equal(getattr(controller, name).detach(), value) for name, value in current.items()
    )


def test_ordinary_acceptance_bypasses_repair_and_retains_pending_adam(
    monkeypatch,
) -> None:
    controller, optimizer = _optimizer_with_one_step()
    parameters = {
        name: getattr(controller, name).detach().clone()
        for name in ("edge_magnitude", "bias", "raw_time_constant")
    }
    metrics = _metrics(1.0)
    trial = {"scale": 0.5, "metrics": metrics, "decision": {"pass": True, "reasons": []}}
    monkeypatch.setattr(
        corrected,
        "_ORIGINAL_FIND_SAFE_TRIAL",
        lambda *args, **kwargs: (0.5, metrics, [trial]),
    )
    corrected._PENDING_OPTIMIZER = optimizer
    corrected._OPTIMIZER_AFTER_PROPOSAL_SHA256 = corrected.audit.semantic_sha256(
        optimizer.state_dict()
    )

    scale, selected, trials = corrected.find_safe_trial(
        controller,
        parameters,
        {name: torch.zeros_like(value) for name, value in parameters.items()},
        {name: torch.zeros_like(value) for name, value in parameters.items()},
        metrics,
        metrics,
        None,
        None,
        {},
        1.0,
        device=torch.device("cpu"),
    )

    assert scale == pytest.approx(0.5)
    assert selected is metrics
    assert trials[0]["acceptance_kind"] == "ordinary"
    assert trials[0]["optimizer_pending_state"]["pass"] is True


def test_seed_forks_update8_without_changing_source(tmp_path: Path, monkeypatch) -> None:
    initial = tmp_path / "canonical" / "resume.pt"
    output = tmp_path / "corrected"
    initial.parent.mkdir()
    payload = {
        "experiment": corrected.EXPECTED_INITIAL_EXPERIMENT,
        "protocol_commit": corrected.EXPECTED_INITIAL_PROTOCOL_COMMIT,
        "accepted_updates": 8,
        "preflight": {"pass": True},
        "controller": {},
        "optimizer": {},
        "history": [],
        "development_history": [],
    }
    base.atomic_torch_save(payload, initial)
    source_hash = responsibility.file_sha256(initial)
    monkeypatch.setattr(corrected, "EXPECTED_INITIAL_RESUME_SHA256", source_hash)
    args = type("Args", (), {"initial_resume": initial, "output_dir": output})()

    accepted = corrected.seed_corrected_resume(args)
    seeded = torch.load(output / "resume.pt", map_location="cpu", weights_only=True)

    assert accepted == 8
    assert responsibility.file_sha256(initial) == source_hash
    assert seeded["experiment"] == corrected.EXPERIMENT
    assert seeded["protocol_commit"] == corrected.PROTOCOL_COMMIT
    assert seeded["run_state"] == "active"


def test_resume_state_prevents_rejected_retry_and_recovers_scheduled_terminal() -> None:
    rejected = [{"update": 9, "accepted": False, "stop_reason": "rejected"}]
    assert base.resumed_stop_state(8, rejected, []) == ("rejected", False)

    development = [
        {
            "update": 10,
            "mandatory_update_50_gate": None,
            "terminal": {"pass": True},
        }
    ]
    assert base.resumed_stop_state(10, [], development) == (
        "first scheduled terminal checkpoint qualified",
        True,
    )


def test_acceptance_metadata_keeps_proposal_and_correction_scales_distinct() -> None:
    trials = [
        {
            "decision": {"pass": True},
            "acceptance_kind": "repaired",
            "proposal_scale": 0.0625,
            "correction_scale": 0.5,
        }
    ]

    result = base.accepted_trial_metadata(0.0625, trials)

    assert result == {
        "acceptance_kind": "repaired",
        "accepted_proposal_scale": 0.0625,
        "accepted_correction_scale": 0.5,
    }
