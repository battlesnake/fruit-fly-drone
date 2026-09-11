from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_vertical_motion_gradient_attribution as audit  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402


class _TinyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(16, dtype=torch.float32))
        self.bias_offset = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.tau_ratio = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))

    def parameter_values(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone() for name in audit.PARAMETER_NAMES
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


def _response(values: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "integrated_response": values,
        "terminal_response": values,
    }


def _bound_activity(status: list[str] | None = None) -> dict:
    return {"flattened_status": status or ["free"] * 24}


def _gradient(*values: float) -> torch.Tensor:
    result = torch.zeros(24, dtype=torch.float64)
    result[: len(values)] = torch.tensor(values, dtype=torch.float64)
    return result


def test_protocol_is_disposable_training_only_and_tests_complete_ladder() -> None:
    protocol = audit.protocol_manifest()

    assert protocol["protocol_commit"] == audit.PROTOCOL_COMMIT
    assert protocol["scope"]["training_pairs"] == 96
    assert protocol["scope"]["development_specs_or_pixels_accessed"] is False
    assert protocol["scope"]["acceptance_specs_or_pixels_accessed"] is False
    assert protocol["scope"]["candidate_or_optimizer_retained"] is False
    assert protocol["gradients"]["clip_accumulated_total_once"] is True
    assert protocol["full_bank_adam_trial"]["multipliers"] == list(audit.MULTIPLIERS)
    assert protocol["full_bank_adam_trial"]["all_multipliers_evaluated"] is True
    assert protocol["authority"]["closed_training_run_may_resume"] is False
    assert protocol["authority"]["motion_routing_hover_gate_or_promotion_authorized"] is False


def test_differentiable_strata_are_per_edge_and_not_texture_diluted() -> None:
    anatomy = {
        "target_indices": np.arange(4, dtype=np.int64),
        "target_subtypes": np.arange(4, dtype=np.int64),
    }
    # T4 opponent is column 0 minus 1; T5 is column 2 minus 3.
    values = torch.tensor(
        [
            [-1.0, 0.0, -4.0, 0.0],
            [1.0, 0.0, 4.0, 0.0],
        ],
        requires_grad=True,
    )
    texture = torch.full_like(values, 1_000.0)
    normal = [_response(values), _response(values), _response(texture), _response(texture)]
    references = {
        "population": {
            window: [
                {"T4": {"normal_scale": torch.tensor(2.0)}},
                {"T5": {"normal_scale": torch.tensor(4.0)}},
                {},
                {},
            ]
            for window in ("integrated", "terminal")
        }
    }

    strata = audit.differentiable_edge_strata(normal, references, anatomy)

    assert float(strata["ON_down"].detach()) == pytest.approx(0.0)
    assert float(strata["ON_up"].detach()) == pytest.approx(0.0)
    assert float(strata["OFF_down"].detach()) == pytest.approx(0.0)
    assert float(strata["OFF_up"].detach()) == pytest.approx(0.0)
    assert set(strata) == set(audit.STRATUM_NAMES)


def test_batched_objective_gradients_returns_one_exact_row_per_objective() -> None:
    first = torch.tensor([2.0, -1.0], requires_grad=True)
    second = torch.tensor([3.0], requires_grad=True)
    objectives = [first.square().sum() + second.square().sum(), first.sum() * second[0]]

    gradients = audit.batched_objective_gradients(objectives, (first, second))

    assert gradients.dtype == torch.float64
    assert gradients.tolist() == [[4.0, -2.0, 6.0], [3.0, 3.0, 1.0]]


def test_trial_decision_requires_all_three_preregistered_conditions() -> None:
    baseline_strata = {
        name: value
        for name, value in zip(
            audit.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True
        )
    }
    passing_strata = dict(baseline_strata)
    passing_strata["ON_down"] = 0.197

    passing = audit.trial_decision(1.0, baseline_strata, 0.998, passing_strata, finite=True)
    regressing = dict(passing_strata)
    regressing["ON_up"] += audit.STRATUM_ABSOLUTE_INCREASE_MAXIMUM * 1.01

    assert passing["pass"] is True
    assert audit.trial_decision(1.0, baseline_strata, 0.9995, passing_strata, finite=True)[
        "pass"
    ] is False
    assert audit.trial_decision(1.0, baseline_strata, 0.998, regressing, finite=True)[
        "pass"
    ] is False
    assert audit.trial_decision(1.0, baseline_strata, 0.998, passing_strata, finite=False)[
        "pass"
    ] is False


def test_common_descent_reports_compatible_and_opposed_geometry() -> None:
    compatible = {
        "ON_down": _gradient(1.0, 0.0),
        "ON_up": _gradient(0.0, 1.0),
        "OFF_down": _gradient(1.0, 1.0),
        "OFF_up": _gradient(2.0, 1.0),
    }
    opposed = {
        "ON_down": _gradient(1.0),
        "ON_up": _gradient(-1.0),
        "OFF_down": _gradient(1.0),
        "OFF_up": _gradient(-1.0),
    }

    compatible_result = audit.common_descent_geometry(compatible, _bound_activity())
    opposed_result = audit.common_descent_geometry(opposed, _bound_activity())

    assert compatible_result["solver_success"] is True
    assert compatible_result["certified_feasible_solution"] is True
    assert compatible_result["strict_common_descent"] is True
    assert compatible_result["observed_minimum_descent_margin"] > 0.0
    assert opposed_result["solver_success"] is True
    assert opposed_result["certified_feasible_solution"] is True
    assert opposed_result["strict_common_descent"] is False
    assert (
        opposed_result["observed_minimum_descent_margin"]
        <= audit.COMMON_DESCENT_MARGIN_TOLERANCE
    )


def test_common_descent_respects_an_active_lower_bound() -> None:
    gradients = {name: _gradient(1.0) for name in audit.STRATUM_NAMES}
    status = ["lower", *(["free"] * 23)]

    result = audit.common_descent_geometry(gradients, _bound_activity(status))

    assert result["solver_success"] is True
    assert result["certified_feasible_solution"] is True
    assert result["strict_common_descent"] is False
    assert result["unit_direction"][0] >= 0.0


def test_parameter_bound_activity_reports_original_group_bounds() -> None:
    values = {
        "gain": torch.tensor([0.25, 4.0, *([1.0] * 14)]),
        "bias_offset": torch.tensor([-0.25, 0.25, 0.0, 0.0]),
        "tau_ratio": torch.tensor([0.5, 2.0, 1.0, 1.0]),
    }

    result = audit.parameter_bound_activity(values)

    assert result["groups"]["gain"]["lower_indices"] == [0]
    assert result["groups"]["gain"]["upper_indices"] == [1]
    assert result["groups"]["bias_offset"]["lower_indices"] == [0]
    assert result["groups"]["bias_offset"]["upper_indices"] == [1]
    assert result["groups"]["tau_ratio"]["lower_indices"] == [0]
    assert result["groups"]["tau_ratio"]["upper_indices"] == [1]


def test_trials_cast_host_fp64_gradient_to_fp32_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _TinyController()
    optimizer = training.make_optimizer(controller)
    for _ in range(audit.ARCHIVED_PROPOSALS):
        for parameter in controller.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    archived_parameters = controller.parameter_values()
    archived_optimizer = optimizer.state_dict()
    baseline_strata = {
        name: value
        for name, value in zip(
            audit.STRATUM_NAMES, (0.2, 0.1, 0.1, 0.1), strict=True
        )
    }
    candidate_strata = dict(baseline_strata)
    candidate_strata["ON_down"] = 0.197
    monkeypatch.setattr(training, "_candidate_bank", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        training,
        "bank_loss",
        lambda *args, **kwargs: {"loss": 0.998, "components": {}},
    )
    monkeypatch.setattr(
        training, "edge_direction_strata", lambda *args, **kwargs: candidate_strata
    )
    gradient = torch.ones(24, dtype=torch.float64) * 0.01
    stratum_gradients = {name: gradient.clone() for name in audit.STRATUM_NAMES}

    trials, first = audit.run_trials(
        controller,
        optimizer,
        archived_parameters,
        archived_optimizer,
        gradient,
        gradient,
        stratum_gradients,
        [gradient],
        [],
        (),
        {},
        {},
        {},
        {"loss": 1.0},
        baseline_strata,
        device=torch.device("cpu"),
    )

    assert len(trials) == 4
    assert first == 1.0
    assert all(trial["adam_counter_transition_pass"] for trial in trials)
    assert controller.gain.dtype == torch.float32
    assert audit._tree_semantic_sha256(
        controller.parameter_values()
    ) == audit._tree_semantic_sha256(archived_parameters)
    assert audit._tree_semantic_sha256(optimizer.state_dict()) == audit._tree_semantic_sha256(
        archived_optimizer
    )
