from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_vertical_motion_frozen_witness as audit  # noqa: E402
import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import check_vertical_motion_frozen_witness as checker  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402


class _TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        self.bias_offset = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.tau_ratio = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone() for name in attribution.PARAMETER_NAMES
        }

    def load_parameter_values(self, values: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, value in values.items():
                getattr(self, name).copy_(value)

    def project_parameters(self) -> None:
        with torch.no_grad():
            self.gain.clamp_(0.25, 4.0)
            self.bias_offset.clamp_(-0.25, 0.25)
            self.tau_ratio.clamp_(0.5, 2.0)


def _advance_optimizer(controller: _TinyController) -> torch.optim.Adam:
    optimizer = training.make_optimizer(controller)
    for _ in range(attribution.ARCHIVED_PROPOSALS):
        for parameter in controller.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return optimizer


def _unit_direction() -> torch.Tensor:
    direction = -torch.ones(24, dtype=torch.float64)
    return direction / torch.linalg.vector_norm(direction)


def test_protocol_freezes_one_witness_and_never_authorizes_hover() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["protocol_commit"] == audit.PROTOCOL_COMMIT
    assert protocol["frozen_witness"]["use_verbatim_without_renormalization"] is True
    assert protocol["frozen_witness"]["rerun_solver_or_optimize_direction"] is False
    assert protocol["finite_step_ladder"]["multipliers"] == list(audit.MULTIPLIERS)
    assert protocol["finite_step_ladder"]["optimizer_stepped_or_modified"] is False
    assert protocol["controls"]["all_values_finite_or_entire_audit_invalid"] is True
    assert protocol["authority"]["closed_training_run_may_resume"] is False
    assert protocol["authority"]["motion_routing_hover_gate_or_promotion_authorized"] is False


def test_witness_control_recomputes_normalized_derivatives() -> None:
    direction = _unit_direction()
    gradient = -direction
    gradients = {name: gradient.clone() for name in attribution.STRATUM_NAMES}
    summary = {"full_gradient": gradient.tolist()}
    locked_gradient = {
        "full_gradient": gradient.tolist(),
        "stratum_gradients": {name: gradient.tolist() for name in gradients},
        "gradient_semantic_sha256": "frozen",
    }
    full_report = {"result": {"gradient_attribution": locked_gradient}}
    compact = {"gradient_attribution": {"common_descent": {"unit_direction": direction.tolist()}}}
    parameters = _TinyController().parameter_values()

    result, observed_direction, observed_gradient = audit.witness_control(
        summary, gradients, parameters, full_report, compact
    )

    assert result["pass"] is True
    assert result["direction_norm"] == pytest.approx(1.0)
    assert result["raw_directional_derivatives"]["full"] == pytest.approx(-1.0)
    assert all(
        value == pytest.approx(-1.0)
        for value in result["normalized_directional_derivatives"].values()
    )
    assert torch.equal(observed_direction, direction)
    assert torch.equal(observed_gradient, gradient)


def test_witness_control_rejects_renormalization_need() -> None:
    direction = _unit_direction() * 0.99
    gradient = -_unit_direction()
    gradients = {name: gradient.clone() for name in attribution.STRATUM_NAMES}
    full_report = {
        "result": {
            "gradient_attribution": {
                "full_gradient": gradient.tolist(),
                "stratum_gradients": {name: gradient.tolist() for name in gradients},
                "gradient_semantic_sha256": "frozen",
            }
        }
    }
    compact = {"gradient_attribution": {"common_descent": {"unit_direction": direction.tolist()}}}

    with pytest.raises(audit.ControlFailure, match="witness qualification"):
        audit.witness_control(
            {"full_gradient": gradient.tolist()},
            gradients,
            _TinyController().parameter_values(),
            full_report,
            compact,
        )


def test_finite_step_trials_never_step_optimizer_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _TinyController()
    optimizer = _advance_optimizer(controller)
    archived_parameters = controller.parameter_values()
    archived_optimizer = copy.deepcopy(optimizer.state_dict())
    archived_fingerprint = audit._optimizer_fingerprint(archived_parameters, archived_optimizer)
    direction = _unit_direction()
    gradient = -direction
    gradients = {name: gradient.clone() for name in attribution.STRATUM_NAMES}
    baseline_strata = {
        name: value
        for name, value in zip(attribution.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True)
    }
    passing_strata = dict(baseline_strata)
    passing_strata["ON_down"] = 0.197
    monkeypatch.setattr(training, "_candidate_bank", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        training,
        "bank_loss",
        lambda *args, **kwargs: {"loss": 0.998, "components": {}},
    )
    monkeypatch.setattr(training, "edge_direction_strata", lambda *args, **kwargs: passing_strata)

    trials, first = audit.run_trials(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        direction,
        gradient,
        gradients,
        [],
        (),
        {},
        {},
        {},
        {"loss": 1.0},
        baseline_strata,
        device=torch.device("cpu"),
    )

    assert len(trials) == len(audit.MULTIPLIERS)
    assert first == 1.0
    assert all(trial["optimizer_unchanged"] for trial in trials)
    assert all(
        set(trial["adam_steps_after"].values()) == {attribution.ARCHIVED_PROPOSALS}
        for trial in trials
    )
    assert all(trial["all_signed_directional_derivatives_negative"] for trial in trials)
    assert (
        audit._optimizer_fingerprint(controller.parameter_values(), optimizer.state_dict())
        == archived_fingerprint
    )


def test_later_nonfinite_response_invalidates_ladder_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _TinyController()
    optimizer = _advance_optimizer(controller)
    archived_parameters = controller.parameter_values()
    archived_optimizer = copy.deepcopy(optimizer.state_dict())
    archived_fingerprint = audit._optimizer_fingerprint(archived_parameters, archived_optimizer)
    direction = _unit_direction()
    gradient = -direction
    gradients = {name: gradient.clone() for name in attribution.STRATUM_NAMES}
    baseline_strata = {
        name: value
        for name, value in zip(attribution.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True)
    }
    candidate_strata = dict(baseline_strata)
    candidate_strata["ON_down"] = 0.197
    calls = 0

    def candidate_bank(*args: object, **kwargs: object) -> dict:
        nonlocal calls
        calls += 1
        return {} if calls == 1 else {"response": torch.tensor(float("nan"))}

    monkeypatch.setattr(training, "_candidate_bank", candidate_bank)
    monkeypatch.setattr(
        training,
        "bank_loss",
        lambda *args, **kwargs: {"loss": 0.998, "components": {}},
    )
    monkeypatch.setattr(training, "edge_direction_strata", lambda *args, **kwargs: candidate_strata)

    with pytest.raises(audit.ControlFailure, match="numerical controls"):
        audit.run_trials(
            controller,
            optimizer,
            archived_parameters,
            archived_optimizer,
            direction,
            gradient,
            gradients,
            [],
            (),
            {},
            {},
            {},
            {"loss": 1.0},
            baseline_strata,
            device=torch.device("cpu"),
        )

    assert calls == 2
    assert (
        audit._optimizer_fingerprint(controller.parameter_values(), optimizer.state_dict())
        == archived_fingerprint
    )


def _terminal_report() -> dict:
    with audit.COMPACT_REPORT.open() as stream:
        direction = json.load(stream)["gradient_attribution"]["common_descent"]["unit_direction"]
    baseline_strata = {
        name: value
        for name, value in zip(attribution.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True)
    }
    candidate_strata = dict(baseline_strata)
    candidate_strata["ON_down"] = 0.197
    trials = []
    for multiplier in audit.MULTIPLIERS:
        intended = [value * audit.REFERENCE_DISPLACEMENT_NORM * multiplier for value in direction]
        trials.append(
            {
                "multiplier": multiplier,
                "reference_displacement_norm": audit.REFERENCE_DISPLACEMENT_NORM,
                "intended_displacement": intended,
                "intended_displacement_norm": sum(value * value for value in intended) ** 0.5,
                "actual_displacement": intended,
                "actual_displacement_norm": sum(value * value for value in intended) ** 0.5,
                "projection_difference": [0.0] * 24,
                "projection_difference_norm": 0.0,
                "signed_directional_derivatives": {
                    "full": -0.01,
                    "strata": {name: -0.01 for name in attribution.STRATUM_NAMES},
                },
                "all_signed_directional_derivatives_negative": True,
                "optimizer_semantic_sha256_before": "a" * 64,
                "optimizer_semantic_sha256_after": "a" * 64,
                "optimizer_unchanged": True,
                "adam_steps_before": {
                    name: attribution.ARCHIVED_PROPOSALS for name in attribution.PARAMETER_NAMES
                },
                "adam_steps_after": {
                    name: attribution.ARCHIVED_PROPOSALS for name in attribution.PARAMETER_NAMES
                },
                "loss": {"loss": 0.998, "components": {}},
                "strata": candidate_strata,
                "decision": attribution.trial_decision(
                    1.0,
                    baseline_strata,
                    0.998,
                    candidate_strata,
                    finite=True,
                ),
            }
        )
    result = {
        "reproduction_control": {"pass": True},
        "proposal25_loss": {"loss": 1.0},
        "proposal25_strata": baseline_strata,
        "witness_control": {
            "pass": True,
            "finite": True,
            "direction": direction,
            "direction_norm_error": 1.2e-10,
            "active_bound_tangent_pass": True,
            "normalized_directional_derivatives": {
                "full": -0.4,
                **{name: -0.1 for name in attribution.STRATUM_NAMES},
            },
            "raw_directional_derivatives": {
                "full": -0.1,
                **{name: -0.1 for name in attribution.STRATUM_NAMES},
            },
            "gradient_norms": {
                "full": 0.25,
                **{name: 1.0 for name in attribution.STRATUM_NAMES},
            },
            "gradient_reproduction_maximum_absolute_differences": {
                "full": 1.0e-7,
                "strata": {name: 1.0e-7 for name in attribution.STRATUM_NAMES},
            },
        },
        "trials": trials,
        "all_four_multipliers_evaluated": True,
        "first_passing_multiplier": 1.0,
        "finite_step_supported": True,
        "archived_state_fingerprint_before": "b" * 64,
        "archived_state_fingerprint_after": "b" * 64,
        "archived_state_restored": True,
        "source_state_sha256_before": "c" * 64,
        "source_state_sha256_after": "c" * 64,
        "source_restored": True,
    }
    return {
        "experiment": audit.EXPERIMENT,
        "protocol_commit": audit.PROTOCOL_COMMIT,
        "protocol": audit.protocol_manifest(),
        "input_file_sha256": {
            "runs/optic-motion/vertical-motion-gradient-attribution-001/report.json": (
                audit.ATTRIBUTION_REPORT_SHA256
            ),
            "runs/optic-motion/vertical-motion-gradient-attribution-001/started.json": (
                audit.ATTRIBUTION_STARTED_SHA256
            ),
            "artifacts/vertical-motion-gradient-attribution-v1/report.json": (
                audit.COMPACT_REPORT_SHA256
            ),
        },
        "implementation_file_sha256": registration.file_sha256(Path(audit.__file__)),
        "audit_completed": True,
        "classification": "frozen_witness_finite_step_supported",
        "result": result,
        "exception": None,
        "finite_step_supported": True,
        "constrained_training_preregistration_authorized": True,
        "closed_training_run_may_resume": False,
        "development_or_acceptance_opened": False,
        "candidate_or_optimizer_retained": False,
        "motion_routing_hover_gate_or_promotion_authorized": False,
    }


def test_terminal_checker_accepts_complete_report_and_rejects_tampering() -> None:
    report = _terminal_report()

    checker.validate_report(report)
    report["result"]["trials"][2]["optimizer_semantic_sha256_after"] = "d" * 64
    with pytest.raises(SystemExit, match="trial 0.25 is invalid"):
        checker.validate_report(report)
