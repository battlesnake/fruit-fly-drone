from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_variable_height_full_native_joint_teacher as train  # noqa: E402


def _metrics(
    *,
    common: float = 1.0,
    height: float = 0.5,
    damping: float = 1.0,
    sign: float = 0.0,
    rpy: float = 0.01,
) -> dict[str, object]:
    return {
        "component_nrmse": {
            "common": common,
            "height": height,
            "damping": damping,
            "interaction": 0.1,
        },
        "by_supervision_step_nrmse": {
            str(step): {
                "common": common,
                "height": height,
                "damping": damping,
                "interaction": 0.1,
            }
            for step in train.joint.SUPERVISION_STEPS
        },
        "rpy_source_nrmse": {"roll": rpy, "pitch": rpy, "yaw": rpy},
        "endpoint_damping_nrmse": damping,
        "endpoint_damping_correct_sign_fraction": sign,
        "endpoint_damping_teacher_aligned_gain": 0.0,
        "motor_output_max_absolute": 0.7,
        "all_attitude_cache_states_valid": True,
    }


def test_objective_freezes_source_denominators_and_applies_floor() -> None:
    source = _metrics(common=2.0, height=0.1, damping=0.5)
    denominators = train.objective_denominators(source)

    assert denominators == {
        "common": pytest.approx(4.0),
        "height": pytest.approx(0.25**2),
        "endpoint_damping": pytest.approx(0.25),
    }
    assert train.joint_teacher_objective(source, denominators) == pytest.approx(
        (1.0 + 0.1**2 / 0.25**2 + 1.0) / 3.0
    )


def test_candidate_acceptance_is_j_based_and_allows_temporary_d_worsening() -> None:
    current = train.attach_joint_objective(
        _metrics(common=1.0, height=0.5, damping=1.0),
        {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0},
    )
    candidate = copy.deepcopy(current)
    candidate["joint_teacher_objective"] = current["joint_teacher_objective"] - 0.001
    candidate["endpoint_damping_nrmse"] = 1.1
    installed = {"pass": True}

    decision = train.candidate_decision(current, candidate, installed, optimizer_pending_exact=True)

    assert decision["pass"] is True
    assert decision["endpoint_damping_nrmse_change"] == pytest.approx(0.1)
    assert decision["endpoint_damping_may_temporarily_worsen"] is True


def test_rpy_projection_specs_have_no_c_or_p_rows() -> None:
    metrics = _metrics(rpy=0.039)
    metrics["rpy_source_nrmse"] = {"roll": 0.039, "pitch": 0.04, "yaw": 0.051}

    specs = train.rpy_constraint_specs(metrics)

    assert [spec["name"] for spec in specs] == ["rpy.pitch", "rpy.yaw"]
    assert all(spec["kind"] == "attitude" for spec in specs)
    assert all(spec["limit_mse"] == pytest.approx(0.05**2) for spec in specs)


def test_midpoint_requires_d_gain_and_aggregate_c_p_preservation() -> None:
    source = _metrics(common=1.0, height=0.5, damping=1.0)
    passing = _metrics(common=1.0, height=0.49, damping=0.85)
    failed = _metrics(common=1.001, height=0.49, damping=0.84)

    assert train.midpoint_decision(source, passing)["pass"] is True
    decision = train.midpoint_decision(source, failed)
    assert decision["pass"] is False
    assert decision["reasons"] == ["aggregate common NRMSE exceeded the training source"]


def test_final_bank_gate_requires_d_sign_and_each_c_p_horizon() -> None:
    source = _metrics(common=1.0, height=0.5, damping=1.0)
    passing = _metrics(common=0.99, height=0.49, damping=0.75, sign=0.5)

    assert train.final_bank_decision(source, passing, bank="training")["pass"] is True
    failing = copy.deepcopy(passing)
    failing["by_supervision_step_nrmse"]["20"]["height"] = 0.501
    decision = train.final_bank_decision(source, failing, bank="training")
    assert decision["pass"] is False
    assert "training height NRMSE at step 20 exceeded source" in decision["reasons"]


def test_milestones_reject_nonfinite_source_reference() -> None:
    source = _metrics(common=float("nan"), height=0.5, damping=float("nan"))
    candidate = _metrics(common=0.9, height=0.4, damping=0.7, sign=1.0)

    midpoint = train.midpoint_decision(source, candidate)
    final = train.final_bank_decision(source, candidate, bank="development")

    assert midpoint["pass"] is False
    assert midpoint["source_metrics_finite"] is False
    assert midpoint["source_endpoint_damping_nrmse_positive"] is False
    assert final["pass"] is False
    assert final["source_metrics_finite"] is False
    assert final["source_endpoint_damping_nrmse_positive"] is False


def test_resume_validation_requires_exact_counter_and_passing_midpoint() -> None:
    denominators = {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0}
    source_metrics = _metrics()
    source = train.attach_joint_objective(source_metrics, denominators)
    controller = {
        "edge_magnitude": torch.tensor([1.0]),
        "bias": torch.tensor([2.0]),
        "raw_time_constant": torch.tensor([3.0]),
    }
    optimizer = {"state": {}, "param_groups": []}
    payload = {
        "experiment": train.EXPERIMENT,
        "protocol_commit": train.PROTOCOL_COMMIT,
        "checkpoint_sha256": "checkpoint",
        "cache_manifest_sha256": train.EXPECTED_CACHE_MANIFEST_SHA256,
        "objective_denominators": denominators,
        "objective_denominators_sha256": train.audit.semantic_sha256(denominators),
        "preflight": {
            "pass": True,
            "source_training_metrics": source_metrics,
            "source_training_metrics_sha256": train.audit.semantic_sha256(source_metrics),
            "source_training": source,
            "source_training_sha256": train.audit.semantic_sha256(source),
        },
        "accepted_updates": 25,
        "optimizer_step_counters": [25.0],
        "history": [{"update": update, "accepted": True} for update in range(1, 26)],
        "milestones": {"25": {"pass": True}},
        "run_state": "active",
        "controller": controller,
        "controller_parameters_sha256": train.audit.semantic_sha256(controller),
        "optimizer": optimizer,
        "optimizer_sha256": train.audit.semantic_sha256(optimizer),
        "current_training": source,
        "current_training_sha256": train.audit.semantic_sha256(source),
    }
    train.validate_resume_payload(
        payload,
        checkpoint_sha256="checkpoint",
        regenerated_denominators={**denominators, "common": 1.0 + 1e-9},
        regenerated_source_metrics=source_metrics,
    )

    payload["optimizer_step_counters"] = [24.0]
    with pytest.raises(SystemExit, match="Adam counters"):
        train.validate_resume_payload(
            payload,
            checkpoint_sha256="checkpoint",
            regenerated_denominators=denominators,
            regenerated_source_metrics=source_metrics,
        )


def test_protocol_disables_d_only_acceptance_and_nonlinear_repair() -> None:
    protocol = train.protocol_manifest()

    assert protocol["protocol_commit"] == "f0e5614"
    assert protocol["optimizer"]["initialized_fresh"] is True
    assert protocol["optimizer"]["initial_update_counter"] == 0
    assert protocol["projection"]["c_and_p_source_ceiling_rows"] is False
    assert protocol["projection"]["activated_rpy_rows_only"] is True
    assert protocol["selection"]["per_step_endpoint_d_improvement_required"] is False
    assert protocol["selection"]["nonlinear_c25_repair"] is False
    assert protocol["selection"]["minimum_actual_j_improvement"] == pytest.approx(1e-4)
    assert protocol["development"]["candidate_evaluations_after_passing_final_training_gate"] == 1
    assert protocol["closed_loop_hover"] is False
    assert protocol["promotion"] is False


def test_optimizer_counter_helper_is_empty_before_first_adam_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)

    assert train.optimizer_step_counters(optimizer.state_dict()) == []


class TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([1.0]))
        self.bias = torch.nn.Parameter(torch.tensor([2.0]))
        self.raw_time_constant = torch.nn.Parameter(torch.tensor([3.0]))

    @torch.no_grad()
    def project_parameters(self) -> None:
        self.edge_magnitude.clamp_(0.0, 8.0)


def test_accumulated_gradient_matches_equal_c_p_endpoint_d_objective(monkeypatch) -> None:
    controller = TinyController()
    cache = SimpleNamespace(prefix_images=torch.empty(2, 1))

    def prediction(*args, **kwargs):
        return (
            {
                "common": controller.edge_magnitude.expand(3),
                "height": controller.bias.expand(3),
                "damping": controller.raw_time_constant.expand(3),
                "interaction": torch.zeros(3),
            },
            0.0,
        )

    monkeypatch.setattr(train.joint, "_run_factorial_scene", prediction)
    monkeypatch.setattr(
        train.joint,
        "_factorial_targets",
        lambda *args, **kwargs: {
            "common": torch.zeros(3),
            "height": torch.zeros(3),
            "damping": torch.zeros(3),
            "interaction": torch.zeros(3),
        },
    )

    train.accumulated_joint_teacher_gradient(
        controller,
        cache,
        {"common": 1.0, "height": 1.0},
        1.0,
        {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0},
        device=torch.device("cpu"),
    )

    assert controller.edge_magnitude.grad.item() == pytest.approx(2.0 / 3.0)
    assert controller.bias.grad.item() == pytest.approx(4.0 / 3.0)
    assert controller.raw_time_constant.grad.item() == pytest.approx(2.0)


def test_zero_active_rpy_rows_use_identity_without_polished_solver(monkeypatch) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(
        [
            {"params": [controller.edge_magnitude, controller.bias], "lr": 1e-4},
            {"params": [controller.raw_time_constant], "lr": 1e-6},
        ]
    )

    def fixed_gradient(student, *args, **kwargs):
        for name in train.joint.PARAMETER_FAMILIES:
            getattr(student, name).grad = torch.ones_like(getattr(student, name))

    def unexpected_polished_solver(*args, **kwargs):
        raise AssertionError("zero-row proposal called the polished solver")

    monkeypatch.setattr(train, "accumulated_joint_teacher_gradient", fixed_gradient)
    monkeypatch.setattr(
        train.polished, "polished_bound_aware_projection", unexpected_polished_solver
    )
    metrics = _metrics(rpy=0.01)

    proposal = train.make_projected_proposal(
        controller,
        optimizer,
        metrics,
        None,
        None,
        {},
        1.0,
        {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0},
        device=torch.device("cpu"),
    )

    projection = proposal["projection"]
    assert projection["mode"] == "identity_no_active_rpy_rows"
    assert projection["c_or_p_constraint_rows"] == 0
    assert projection["rpy_constraint_rows"] == 0
    assert projection["pass_after_parameter_bounds"] is True
    assert projection["optimizer_transaction"]["pass"] is True
    assert train.optimizer_step_counters(optimizer.state_dict()) == [1.0]


def test_failed_canonical_control_aborts_without_backtracking_or_evaluation(
    monkeypatch,
) -> None:
    controller = TinyController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)
    parameters = train.joint._copy_parameters(controller)
    proposal = {
        "current_parameters": parameters,
        "authoritative_parameters": parameters,
        "optimizer_after_sha256": train.audit.semantic_sha256(optimizer.state_dict()),
    }
    calls = {"install": 0}

    def failed_install(*args, **kwargs):
        calls["install"] += 1
        return {"pass": False}

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("failed canonical control reached candidate evaluation")

    monkeypatch.setattr(train, "install_authoritative_trial", failed_install)
    monkeypatch.setattr(train.endpoint, "evaluate", unexpected_evaluation)

    with pytest.raises(RuntimeError, match="canonical-idempotence"):
        train.find_safe_trial(
            controller,
            optimizer,
            proposal,
            _metrics(),
            None,
            None,
            {},
            1.0,
            {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0},
            device=torch.device("cpu"),
        )
    assert calls["install"] == 1


def test_failed_finite_difference_canonical_control_skips_replay(monkeypatch) -> None:
    controller = TinyController()
    parameters = train.joint._copy_parameters(controller)
    proposal = {
        "current_parameters": parameters,
        "authoritative_parameters": parameters,
        "raw_gradients": {name: torch.ones_like(value) for name, value in parameters.items()},
    }
    monkeypatch.setattr(
        train,
        "install_authoritative_trial",
        lambda *args, **kwargs: {
            "pass": False,
            "canonical_parameter_idempotence": {"pass": False},
            "parameter_bounds": {"pass": True},
        },
    )
    monkeypatch.setattr(
        train.endpoint,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failed finite-difference control reached replay")
        ),
    )

    result = train.directional_finite_difference(
        controller,
        proposal,
        {"joint_teacher_objective": 1.0},
        None,
        None,
        {},
        1.0,
        {"common": 1.0, "height": 1.0, "endpoint_damping": 1.0},
        device=torch.device("cpu"),
    )

    assert result["pass"] is False
    assert result["metrics"] is None


def test_interrupted_development_reports_started_not_completed() -> None:
    record = train.interrupted_development_record(
        {
            "candidate_evaluation_started": True,
            "candidate_evaluation_completed": False,
        },
        reason="interrupted",
    )

    assert record["pass"] is False
    assert record["candidate_evaluation_started_count"] == 1
    assert record["candidate_evaluation_completed_count"] == 0
    assert record["retried"] is False
