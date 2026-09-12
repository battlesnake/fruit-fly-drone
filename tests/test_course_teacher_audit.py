from __future__ import annotations

from dataclasses import replace

import torch

import scripts.audit_pragmatic_course_teacher as physical_audit
import scripts.evaluate_pragmatic_two_gate_zero_shot as sequence
from flydrone.gate import AnnularGate, GateConfig
from flydrone.hover import DifferentiableQuad, HoverConfig
from flydrone.visual_hover import CameraSpec
from scripts.audit_course_teacher_targets import summarize_targets
from scripts.audit_pragmatic_course_teacher import gate_center_frustum
from scripts.evaluate_pragmatic_two_gate_zero_shot import diagnostic_axis_takeover


def test_local_physical_roll_teacher_applies_from_start_and_preserves_other_axes(monkeypatch):
    curved = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = curved.clone()
    index = torch.tensor([0, 1, 2])
    gates = tuple(AnnularGate(torch.full((3, 3), float(i)), torch.zeros(3)) for i in range(2))
    monkeypatch.setattr(physical_audit, "course_teacher_motor", lambda *args: curved)
    calls = []

    def local(state, gate, config, *, active):
        calls.append((gate.center.clone(), active.clone()))
        return torch.tensor([-0.1, 0.2, -0.3])

    monkeypatch.setattr(physical_audit, "current_gate_roll_motor", local)
    unchanged = physical_audit.physical_teacher_motor(
        None, None, None, None, gates, index, "curved"
    )
    assert torch.equal(unchanged, original) and not calls
    local_motor = physical_audit.physical_teacher_motor(
        None, None, None, None, gates, index, "current-gate"
    )
    assert torch.equal(local_motor[:, 0], torch.tensor([-0.1, 0.2, -0.3]))
    assert torch.equal(local_motor[:, 1:], original[:, 1:])
    assert calls[0][0][:, 0].tolist() == [0.0, 1.0, 1.0]
    assert calls[0][1].tolist() == [True, True, False]
    assert torch.equal(curved, original)


def test_center_frustum_uses_body_heading_and_excludes_behind_camera():
    state = DifferentiableQuad().initial_state(4, device=torch.device("cpu"), dtype=torch.float32)
    centers = state.position + torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 5.0]]
    )
    state.euler[2, 2] = torch.pi / 2
    inside, distance = gate_center_frustum(state, centers, CameraSpec())
    assert inside.tolist() == [True, False, True, False]
    assert torch.allclose(distance[:3], torch.ones(3))


def test_target_summary_weights_frames_and_keeps_missing_phases_explicit():
    counts = torch.zeros(5, 2)
    counts[0, 0], counts[1, 1] = 1, 3
    squared = torch.zeros(5, 2, 4)
    squared[0, 0] = 4
    squared[1, 1] = 12
    summary = summarize_targets(counts, squared, 4 * squared, torch.ones(4) * 2)
    assert summary["overall"]["frames"] == 4
    assert summary["overall"]["target_rms"] == [2.0] * 4
    assert summary["overall"]["normalized_source_error_rms"] == [2.0] * 4
    assert len(summary["by_gate_and_side"]) == 2
    assert summary["by_gate_and_side"][1]["gate"] == 2
    assert summary["by_gate_and_side"][1]["side"] == "positive"


def test_axis_takeover_is_inactive_before_first_gate_and_never_changes_other_axes():
    native = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    teacher = -torch.ones_like(native)
    current = torch.tensor([0, 1, 5])
    axes = torch.tensor([True, False, False, False])
    mixed = diagnostic_axis_takeover(native, teacher, current, axes)
    assert torch.equal(mixed[0], native[0])
    assert torch.equal(mixed[:, 1:], native[:, 1:])
    assert torch.equal(mixed[1:, 0], teacher[1:, 0])


def test_takeover_evaluation_keeps_native_recurrence_running(monkeypatch):
    cases, _ = sequence.sample_two_gate_cases(
        1, seed=19, device=torch.device("cpu"), hover_config=HoverConfig()
    )
    cases.state.position[:] = torch.tensor([0.0, 0.0, 1.0])
    cases.state.euler.zero_()
    gates = tuple(
        AnnularGate(torch.tensor([[x, 0.0, 1.0]]).expand(2, -1), torch.zeros(2)) for x in (1.0, 2.0)
    )
    applied = []
    neural_seen = []

    class ScriptedQuad:
        def __init__(self, config):
            self.positions = iter([1.2, 1.4, 1.7, 2.2])

        def to(self, device):
            return self

        def __call__(self, rc, state, mass):
            position = state.position.clone()
            position[:, 0] = next(self.positions)
            return replace(state, position=position)

    class CapturingLegs:
        def __init__(self, config):
            pass

        def to(self, device):
            return self

        def __call__(self, motor, sticks):
            applied.append(motor.clone())
            return motor, sticks

    class Actor:
        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def __call__(self, image, attitude, neural):
            neural_seen.append(neural.clone())
            return torch.full((2, 4), 0.25), neural + 1

    monkeypatch.setattr(sequence, "DifferentiableQuad", ScriptedQuad)
    monkeypatch.setattr(sequence, "ForelegStickPlant", CapturingLegs)
    monkeypatch.setattr(sequence, "course_teacher_motor", lambda *args: torch.full((2, 4), -0.5))
    metrics = sequence.evaluate(
        Actor(),
        cases,
        gates,
        seconds=0.04,
        warmup_steps=0,
        camera=CameraSpec(width=32, height=20),
        hover_config=HoverConfig(),
        gate_config=GateConfig(),
        diagnostic_teacher_axes=(True, False, False, False),
    )
    assert len(neural_seen) == 2
    assert torch.equal(neural_seen[1], neural_seen[0] + 1)
    assert torch.equal(applied[0], applied[1])
    assert torch.all(applied[0] == 0.25)
    assert torch.all(applied[2][:, 0] == -0.5)
    assert torch.all(applied[2][:, 1:] == 0.25)
    assert metrics["clean_first_gate_pass_rate"] == 1.0
    assert metrics["privileged_axis_takeover_after_first_gate"] == (True, False, False, False)
