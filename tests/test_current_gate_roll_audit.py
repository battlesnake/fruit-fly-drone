from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import scripts.audit_pragmatic_current_gate_roll as audit
from flydrone.gate import AnnularGate, GateConfig
from flydrone.hover import HoverConfig
from flydrone.visual_hover import CameraSpec


@pytest.mark.parametrize("from_start", [False, True])
def test_local_takeover_changes_only_roll_including_tail_without_mutating_native(from_start):
    native = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = native.clone()
    local = torch.tensor([-0.1, 0.2, -0.3])
    motor = audit.local_roll_takeover(native, local, torch.tensor([0, 1, 5]), from_start=from_start)
    assert torch.equal(motor[:, 1:], original[:, 1:])
    assert torch.equal(motor[1:, 0], local[1:])
    assert motor[0, 0] == (local[0] if from_start else original[0, 0])
    assert torch.equal(native, original)


def comparison_metrics(clean_ids=None):
    clean_ids = range(26) if clean_ids is None else clean_ids
    return dict(
        clean_course_success_episode_indices=list(clean_ids),
        all_gate_success_episode_indices=list(range(32)),  # Raw passages are not clean flights.
        clean_course_success_rate=len(clean_ids) / 32,
        clean_course_negative_success_rate=13 / 16,
        clean_course_positive_success_rate=13 / 16,
        clean_first_gate_pass_rate=28 / 32,
        ground_contact_rate=0.0,
        invalid_rate=0.0,
    )


def test_screen_uses_clean_paired_losses_not_raw_passages_or_only_total_success():
    after = comparison_metrics()
    start = comparison_metrics(range(1, 27))
    result = audit.comparison_screen({"after-first": after, "from-start": start})
    assert result["eligible_for_unified_label_learning"]
    assert result["paired_clean_losses"] == [0]
    assert result["paired_clean_gains"] == [26]
    assert result["net_clean_change"] == 0
    assert result["clean_from_start"] == 26
    start = comparison_metrics(range(2, 28))
    result = audit.comparison_screen({"after-first": after, "from-start": start})
    assert not result["eligible_for_unified_label_learning"]
    assert result["paired_clean_losses"] == [0, 1]
    assert result["net_clean_change"] == 0


@pytest.mark.parametrize("key,value", [
    ("clean_course_negative_success_rate", 11 / 16),
    ("clean_course_positive_success_rate", 11 / 16),
    ("clean_first_gate_pass_rate", 27 / 32),
    ("ground_contact_rate", 1 / 32),
    ("invalid_rate", 1 / 32),
])
def test_screen_rejects_first_gate_side_or_safety_regression(key, value):
    after, start = comparison_metrics(), comparison_metrics()
    start[key] = value
    result = audit.comparison_screen({"after-first": after, "from-start": start})
    assert not result["eligible_for_unified_label_learning"]


def test_screen_rejects_fewer_than_26_and_nonfinite_metrics():
    after = comparison_metrics()
    start = comparison_metrics(range(25))
    assert not audit.comparison_screen({"after-first": after, "from-start": start})[
        "eligible_for_unified_label_learning"
    ]
    start["ground_contact_rate"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit.comparison_screen({"after-first": after, "from-start": start})


def test_matched_evaluations_keep_warmup_neural_state_native_pyt_and_physical_initial_state(
    monkeypatch,
):
    sequence = audit.evaluation
    config = HoverConfig()
    cases, _ = sequence.sample_two_gate_cases(
        1, seed=19, device=torch.device("cpu"), hover_config=config,
    )
    cases.state.position[:] = torch.tensor([0.0, 0.0, 1.0])
    cases.state.euler.zero_()
    initial_position = cases.state.position.clone()
    initial_sticks = cases.sticks.position.clone()
    gates = tuple(
        AnnularGate(torch.tensor([[x, 0.0, 1.0]]).expand(2, -1), torch.zeros(2))
        for x in (1.0, 2.0)
    )
    applied, neural_seen, local_calls, stick_starts = [], [], [], []

    class ScriptedQuad:
        def __init__(self, config):
            self.positions = iter([1.2, 1.4, 1.7, 2.2, 2.4, 2.6])

        def to(self, device):
            return self

        def __call__(self, rc, state, mass):
            position = state.position.clone()
            position[:, 0] = next(self.positions)
            return replace(state, position=position)

    class CapturingLegs:
        def __init__(self, config):
            self.first = True

        def to(self, device):
            return self

        def __call__(self, motor, sticks):
            if self.first:
                stick_starts.append(sticks.position.clone())
                self.first = False
            applied.append(motor.clone())
            return motor, sticks

    class Actor:
        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def __call__(self, image, attitude, neural):
            neural_seen.append(neural.clone())
            return torch.full((2, 4), 0.25), neural + 1

    def local(state, gate, config, *, active):
        local_calls.append((state.position.clone(), active.clone()))
        return torch.full((2,), -0.5)

    monkeypatch.setattr(sequence, "DifferentiableQuad", ScriptedQuad)
    monkeypatch.setattr(sequence, "ForelegStickPlant", CapturingLegs)
    monkeypatch.setattr(audit, "current_gate_roll_motor", local)
    original_hook = sequence.diagnostic_axis_takeover
    original_teacher = sequence.course_teacher_motor
    for condition in ("native", "after-first", "from-start"):
        applied.clear()
        neural_seen.clear()
        local_calls.clear()
        metrics = audit.evaluate_condition(
            Actor(), cases, gates, config, CameraSpec(width=32, height=20), GateConfig(),
            intervention=condition, seconds=0.06, warmup_steps=2,
        )
        assert [int(state[0, 0]) for state in neural_seen] == [0, 1, 2, 3, 4]
        assert len(applied) == 6  # Warmup has no physics; then two substeps per command.
        assert all(torch.all(motor[:, 1:] == 0.25) for motor in applied)
        assert torch.all(applied[0][:, 0] == (-0.5 if condition == "from-start" else 0.25))
        assert torch.all(applied[-1][:, 0] == (0.25 if condition == "native" else -0.5))
        if condition == "native":
            assert not local_calls
        else:
            assert torch.equal(local_calls[0][0], initial_position)
            assert local_calls[-1][1].tolist() == [False, False]  # Complete-course tail.
        if condition == "from-start":
            assert metrics["privileged_roll_from_first_physical_command"]
            assert "privileged_axis_takeover_after_first_gate" not in metrics
        assert sequence.diagnostic_axis_takeover is original_hook
        assert sequence.course_teacher_motor is original_teacher
        assert metrics["clean_course_success_episode_indices"] == [0, 1]
        assert torch.equal(cases.state.position, initial_position)
    assert all(torch.equal(position, initial_sticks) for position in stick_starts)
