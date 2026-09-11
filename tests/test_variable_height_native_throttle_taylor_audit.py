from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_native_throttle_taylor as audit  # noqa: E402


def _record(*, scale: float, error: float, change: float, passing_controls: bool = True):
    return {
        "scale": scale,
        "controls": {"pass": passing_controls},
        "all_metrics_finite": True,
        "predicted_directional_derivative": -1.0,
        "measured_directional_derivative": -1.0,
        "relative_error": error,
        "objective_change": change,
    }


def test_replay_noise_uses_three_replays_and_first_is_not_averaged() -> None:
    assert audit.replay_noise([1.0, 1.0 + 2e-7, 1.0 - 3e-7]) == pytest.approx(5e-7)
    with pytest.raises(ValueError, match="3 finite baseline"):
        audit.replay_noise([1.0, 1.0])


def test_start_marker_is_exclusive_and_cannot_be_replayed(tmp_path: Path) -> None:
    path = tmp_path / "audit-started.json"
    payload = {"status": "started", "replay_permitted": False}

    digest = audit.write_exclusive_start_marker(path, payload)

    assert digest == audit.train.responsibility.file_sha256(path)
    before = path.read_bytes()
    with pytest.raises(SystemExit, match="already started"):
        audit.write_exclusive_start_marker(path, payload)
    assert path.read_bytes() == before


class _ToyController(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([0.2, 0.3]))
        self.bias = torch.nn.Parameter(torch.tensor([0.0, 0.1]))
        self.raw_time_constant = torch.nn.Parameter(torch.tensor([-0.1, 0.1]))


def test_outer_guard_restores_controller_and_optimizer_after_injected_exception() -> None:
    controller = _ToyController()
    optimizer = audit.train._make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    current = audit.train._copy_parameters(controller)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    controller_hash = audit.train.audit.semantic_sha256(current)
    optimizer_hash = audit.train.audit.semantic_sha256(optimizer_state)

    with pytest.raises(RuntimeError, match="injected"):
        with audit.restoring_controller_and_optimizer(
            controller, optimizer, current, optimizer_state
        ):
            with torch.no_grad():
                controller.bias.add_(3.0)
            for parameter in controller.parameters():
                parameter.grad = -torch.ones_like(parameter)
            optimizer.step()
            raise RuntimeError("injected")

    assert audit.train.audit.semantic_sha256(
        audit.train._copy_parameters(controller)
    ) == controller_hash
    assert audit.train.audit.semantic_sha256(optimizer.state_dict()) == optimizer_hash


def test_local_derivative_agreement_keeps_error_and_noise_gates() -> None:
    passing = _record(scale=1 / 64, error=0.2, change=-1.1e-5)
    assert audit.local_derivative_agreement(passing, noise_threshold=1e-5)
    assert not audit.local_derivative_agreement(
        {**passing, "relative_error": 0.20001}, noise_threshold=1e-5
    )
    assert not audit.local_derivative_agreement(
        {**passing, "objective_change": -1e-5}, noise_threshold=1e-5
    )


def test_adjacent_agreement_excludes_registered_one_sixteenth_scale() -> None:
    records = [
        {**_record(scale=1 / 16, error=0.0, change=-1.0), "local_derivative_agreement": True},
        {**_record(scale=1 / 32, error=0.1, change=-1.0), "local_derivative_agreement": True},
        {**_record(scale=1 / 64, error=0.1, change=-1.0), "local_derivative_agreement": True},
        {**_record(scale=1 / 128, error=0.3, change=-1.0), "local_derivative_agreement": False},
    ]
    assert audit.adjacent_passing_pairs(records) == [[1 / 32, 1 / 64]]


@pytest.mark.parametrize(
    ("controls", "registered", "pairs", "above_noise", "expected_pass", "classification"),
    [
        (False, True, [[1 / 32, 1 / 64]], [], False, "taylor_audit_control_failure"),
        (True, False, [[1 / 32, 1 / 64]], [], False, "taylor_audit_control_failure"),
        (
            True,
            True,
            [[1 / 32, 1 / 64]],
            [[1 / 32, 1 / 64]],
            True,
            "local_derivative_convergence_consistent_with_finite_step_curvature",
        ),
        (True, True, [], [], False, "noise_limited_inconclusive"),
        (
            True,
            True,
            [],
            [[1 / 32, 1 / 64]],
            False,
            "above_noise_local_derivative_nonconvergence",
        ),
    ],
)
def test_audit_classification_is_conservative(
    controls: bool,
    registered: bool,
    pairs: list[list[float]],
    above_noise: list[list[float]],
    expected_pass: bool,
    classification: str,
) -> None:
    assert audit.audit_classification(
        controls_pass=controls,
        registered_failure_reproduced=registered,
        adjacent_pairs=pairs,
        adjacent_above_noise_pairs=above_noise,
    ) == (expected_pass, classification)
