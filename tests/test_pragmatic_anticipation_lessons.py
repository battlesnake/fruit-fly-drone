from __future__ import annotations

import numpy as np
import pytest
import torch

import scripts.pragmatic_anticipation_lessons as lessons
from flydrone.gate import AnnularGate, GateConfig
from flydrone.hover import HoverConfig
from flydrone.visual_hover import CameraSpec
from scripts.train_pragmatic_course_replay import ReplayBank
from scripts.train_pragmatic_course_teacher import action_imitation_loss


def small_bank():
    fields = [torch.zeros(12, 4, 3) for _ in range(6)]
    fields[0][:, :, 0] = 1.0
    fields[0][:, :, 1] = 0.6
    fields[0][:, :, 2] = 1.1
    gates = tuple(
        AnnularGate(torch.tensor([[3 + index, 0.6, 1.1]]).repeat(4, 1), torch.zeros(4))
        for index in range(5)
    )
    roles = torch.zeros(12, 4, dtype=torch.long)
    roles[5:] = 1
    return ReplayBank(
        tuple(fields),
        gates,
        roles,
        torch.ones_like(roles, dtype=torch.bool),
        torch.zeros(12, 4, 4),
        "native",
        1,
    )


def test_changed_next_pair_preserves_all_other_geometry_and_course_limits():
    bank = small_bank()
    paired = lessons.changed_next_gate_pair(bank.gates, 0, 1, bank.states[0][0, 0])
    assert paired[2].center[:, 1].tolist() == pytest.approx([0.45, 0.75])
    for index, gate in enumerate(paired):
        assert torch.equal(gate.center[:, (0, 2)], bank.gates[index].center[:2, (0, 2)])
        assert torch.equal(gate.yaw, bank.gates[index].yaw[:2])
        if index != 2:
            assert torch.equal(gate.center, bank.gates[index].center[:2])
    y = torch.stack([g.center[:, 1] for g in paired])
    assert bool((y.diff(dim=0).abs() <= 0.2 + 1e-6).all())
    assert bool(((y - y[:1]).abs() <= 0.5).all())
    bank.gates[3].center[:, 1] = 0.95
    assert lessons.changed_next_gate_pair(bank.gates, 0, 1, bank.states[0][0, 0]) is None


def test_course_limits_apply_to_jitter_around_the_sloping_initial_course_line():
    bank = small_bank()
    launch = torch.tensor([0.0, 0.0, 1.1])
    for index, gate in enumerate(bank.gates):
        gate.center[:, 1] = 0.6 + index * 0.2
    paired = lessons.changed_next_gate_pair(bank.gates, 0, 1, launch)
    assert paired is not None
    assert paired[2].center[:, 1].tolist() == pytest.approx([0.85, 1.15])
    y = torch.stack([g.center[:, 1] for g in paired])
    nominal = torch.tensor([0.6, 0.8, 1.0, 1.2, 1.4])[:, None]
    assert bool(((y - nominal).diff(dim=0).abs() <= 0.2 + 1e-6).all())
    assert bool(((y - nominal).abs() <= 0.5).all())
    assert bool((y.diff(dim=0).abs() > 0.2).any())  # raw world-Y is not the jitter bound


def test_selection_keeps_whole_pair_split_and_does_not_straddle_changed_gate():
    bank = small_bank()
    bank.current[10:] = 2
    for validation in (False, True):
        _, row, start = lessons.select_anticipation_window(
            [bank], 1, 1, 3, np.random.default_rng(4), heldout_pairs=1, validation=validation
        )
        assert row == (3 if validation else 1)
        assert bool((bank.current[start : start + 3, row] == 1).all())


def test_lesson_replays_changed_geometry_from_start_and_recomputes_source_targets(monkeypatch):
    bank = small_bank()
    observed = []

    class Reference:
        bias = torch.zeros(1)

        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def __call__(self, image, attitude, neural):
            observed.append(image.clone())
            neural = 0.9 * neural + image[:, 0, 0, 0, None]
            return neural.expand(-1, 4), neural

    def render(state, gates, **kwargs):
        return gates[2].center[:, 1, None, None, None].expand(-1, 3, 2, 2)

    def teacher(state, path, *args):
        return -path.knots[:, 3, 1, None].expand(-1, 4)

    monkeypatch.setattr(lessons, "render_annular_gates_rgb", render)
    monkeypatch.setattr(lessons, "course_teacher_motor", teacher)
    window, record = lessons.make_anticipation_lesson(
        bank, 0, 5, 3, Reference(), CameraSpec(), GateConfig(), HoverConfig()
    )
    fields, _, _, target, starts = window
    assert len(observed) == 10 + 5 + 3
    assert all(torch.equal(image, observed[0]) for image in observed)
    assert all(torch.equal(field[:, 0], field[:, 1]) for field in fields)
    assert starts.tolist() == [5, 5]
    assert target[5:, :, 0].tolist() == [[pytest.approx(-0.45), pytest.approx(-0.75)]] * 3
    assert bool((target[5:, 1, 1:] > target[5:, 0, 1:]).all())
    assert record["changed_geometry_fixed_for_entire_prefix"]


def test_lesson_rejects_copied_role_history_after_the_modified_gate_plane():
    bank = small_bank()
    bank.states[0][7, 0, 0] = 7.0
    reference = type("Reference", (), {"bias": torch.zeros(1)})()
    with pytest.raises(ValueError, match="remain ahead"):
        lessons.make_anticipation_lesson(
            bank, 0, 5, 3, reference, CameraSpec(), GateConfig(), HoverConfig()
        )


def test_early_preservation_excludes_heldout_courses_and_gate_transition_frames():
    bank = small_bank()
    bank.reference_outputs = torch.ones_like(bank.target)
    result = lessons.early_training_banks([bank], heldout_pairs=1)[0]
    assert result.current.shape[1] == 2
    assert not bool(result.active[5:].any())
    assert result.starts(0, 3) == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(result.target, bank.reference_outputs[:, :2])


def test_roll_only_contrast_emphasis_leaves_nonroll_and_pair_mean_penalties_unchanged():
    prediction = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, -0.1, -0.2, -0.3]])
    active, scale = torch.ones(2, dtype=torch.bool), torch.ones(4)
    target = torch.zeros_like(prediction)
    baseline, axes = action_imitation_loss(prediction, target, active, scale, 1)
    stronger, new_axes = action_imitation_loss(
        prediction, target, active, scale, 1, roll_contrast_weight=16
    )
    assert torch.equal(axes, new_axes)
    assert float(stronger - baseline) == pytest.approx(15 * 0.1**2 / 4, abs=1e-7)
    prediction[:, 0] = 0.3  # equal roll in both branches: no contrast penalty change
    ordinary, _ = action_imitation_loss(prediction, target, active, scale, 1)
    emphasized, _ = action_imitation_loss(
        prediction, target, active, scale, 1, roll_contrast_weight=16
    )
    assert torch.equal(ordinary, emphasized)


def test_zero_response_cannot_masquerade_as_learned_anticipation():
    desired = torch.tensor([0.01, -0.02])
    residual = torch.zeros(2, 2, 4)
    residual[:, 1, 0] = -desired  # prediction contrast is exactly zero
    records = [dict(base_side="negative"), dict(base_side="positive")]
    result = lessons.contrast_error_summary([residual, residual], records, [desired, desired])
    assert not result["negative"]["beats_zero_contrast"]
    assert result["positive"]["teacher_alignment_cosine"] == 0
    assert (
        result["positive"]["contrast_error_rmse"] == result["positive"]["zero_contrast_error_rmse"]
    )
    residual.zero_()  # prediction now equals the teacher
    result = lessons.contrast_error_summary([residual, residual], records, [desired, desired])
    assert result["negative"]["beats_zero_contrast"]
    assert result["negative"]["teacher_alignment_cosine"] == pytest.approx(1.0)


def test_centered_targets_preserve_source_pair_mean_and_true_teacher_contrast():
    teacher = torch.tensor([0.5, 0.1])
    source = torch.tensor([0.1, 0.3])
    target = lessons.anticipation_roll_targets(teacher, source, "source")
    assert torch.allclose(target.mean(), source.mean())
    assert torch.allclose(target.diff(), teacher.diff())
    assert torch.equal(lessons.anticipation_roll_targets(teacher, source, "teacher"), teacher)
    with pytest.raises(ValueError, match="output range"):
        lessons.anticipation_roll_targets(
            torch.tensor([0.5, -0.5]), torch.tensor([0.95, 0.95]), "source"
        )
