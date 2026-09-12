from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_pragmatic_course_replay as replay  # noqa: E402

from flydrone.gate import AnnularGate, GateConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


def bank(kind="native", steps=10):
    fields = [torch.zeros(steps, 2, 3) for _ in range(6)]
    fields[0][:, :, 0] = torch.arange(steps)[:, None] / 10 + torch.tensor([0.1, 0.2])
    gates = tuple(AnnularGate(torch.zeros(2, 3), torch.zeros(2)) for _ in range(5))
    return replay.ReplayBank(
        tuple(fields),
        gates,
        torch.zeros(steps, 2, dtype=torch.long),
        torch.ones(steps, 2, dtype=torch.bool),
        torch.zeros(steps, 2, 4),
        kind,
        1,
    )


def test_window_eligibility_excludes_any_post_failure_frame():
    data = bank()
    data.current[4:, 0] = 1
    data.current[7:, 1] = 1
    data.active[9:, 0] = False
    assert data.starts(1, 2) == [[4, 5, 6, 7], [7, 8]]
    assert data.starts(1, 20) == [[], []]


def test_late_roll_labels_preserve_all_source_axes_until_gate_one_and_other_axes_afterward():
    reference = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    teacher = -torch.ones_like(reference)
    target = replay.late_roll_preservation_targets(teacher, reference, torch.tensor([0, 1, 4]))
    assert torch.equal(target[0], reference[0])
    assert torch.equal(target[:, 1:], reference[:, 1:])
    assert torch.equal(target[1:, 0], teacher[1:, 0])
    assert reference[1, 0] == 4  # constructing labels does not mutate source records


@pytest.mark.parametrize("roll_teacher", ["curved", "neutral", "current-gate"])
def test_roll_teacher_variants_change_only_roll_labels_and_keep_source_preservation(
    monkeypatch, roll_teacher
):
    curved = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = curved.clone()
    reference = -torch.ones_like(curved)
    current = torch.tensor([0, 1, 2])
    gates = tuple(
        AnnularGate(torch.full((3, 3), float(index)), torch.zeros(3)) for index in range(2)
    )
    monkeypatch.setattr(replay, "course_teacher_motor", lambda *args: curved)
    calls = []

    def local(state, gate, config, *, active):
        calls.append((gate.center.clone(), active.clone()))
        return torch.tensor([0.1, 0.2, 0.3])

    monkeypatch.setattr(replay, "current_gate_roll_motor", local)
    target = replay.replay_teacher_motor(None, None, gates, current, None, None, roll_teacher)
    assert torch.equal(curved, original)
    assert torch.equal(target[:, 1:], original[:, 1:])
    expected = {
        "curved": original[:, 0],
        "neutral": torch.zeros(3),
        "current-gate": torch.tensor([0.1, 0.2, 0.3]),
    }[roll_teacher]
    assert torch.equal(target[:, 0], expected)
    preserved = replay.late_roll_preservation_targets(target, reference, current)
    assert torch.equal(preserved[0], reference[0])
    assert torch.equal(preserved[:, 1:], reference[:, 1:])
    assert torch.equal(preserved[1:, 0], expected[1:])
    if roll_teacher == "current-gate":
        assert calls[0][0][:, 0].tolist() == [0.0, 1.0, 1.0]
        assert calls[0][1].tolist() == [True, True, False]
    else:
        assert not calls


def test_noncurved_collection_cannot_silently_drive_other_teacher_axes():
    arguments = (None, 1, 1, "native", 1, None, None, None)
    with pytest.raises(ValueError, match="frozen-source"):
        replay.collect_bank(*arguments, roll_teacher="neutral")
    with pytest.raises(ValueError, match="unknown roll teacher"):
        replay.collect_bank(*arguments, roll_teacher="unrecognized")


def test_fixed_fit_summary_preserves_phase_side_axis_meaning(monkeypatch):
    residuals = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]]).repeat(3, 1, 1)
    monkeypatch.setattr(replay, "replay_window_loss", lambda *a, **k: (None, None, residuals))
    result = replay.replay_fit_summary(None, [(None, {"phase": 3, "seed": 10})], 3, None, None, 1.0)
    record = result["windows"][0]
    assert result["scope"] == "fixed examples from training collections, not held-out"
    assert record["phase"] == 3
    assert record["side_order"] == ["negative", "positive"]
    assert record["axis_order"] == ["roll", "pitch", "yaw", "throttle"]
    assert record["motor_rmse_by_side"] == [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]


def test_roll_mask_excludes_edges_entering_other_motor_pools(monkeypatch):
    graph = dict(
        output_pool_indices=np.array([10, 11, 12, 13]),
        output_pool_offsets=np.array([0, 1, 2, 3, 4]),
        edge_post=np.array([1, 10, 11, 12, 13]),
    )
    monkeypatch.setattr(replay.np, "load", lambda _: nullcontext(graph))
    monkeypatch.setattr(
        replay, "visual_roll_path_mask", lambda *a, **k: (torch.ones(5, dtype=torch.bool), {})
    )
    mask, manifest = replay.roll_preservation_mask(None, torch.device("cpu"))
    assert mask.tolist() == [True, True, True, False, False]
    assert manifest["selected_edges"] == 3
    assert manifest["excluded_edges_entering_other_motor_pools"] == 2


def test_pair_selection_aligns_phase_and_falls_back_only_when_native_pair_is_missing():
    native, teacher = bank(), bank("teacher")
    native.current[4:, 0] = 1
    teacher.current[3:, 0] = 1
    teacher.current[6:, 1] = 1
    chosen, rows, starts, info = replay.select_pair_window(
        native, teacher, 1, 2, np.random.default_rng(3), "approach"
    )
    assert chosen is teacher
    assert rows == (0, 1)
    assert info["source"] == "teacher"
    assert all(teacher.current[t, row] == 1 for t, row in zip(starts, rows, strict=True))
    native.current[7:, 1] = 1
    chosen, _, _, _ = replay.select_pair_window(
        native, teacher, 1, 2, np.random.default_rng(3), "transition"
    )
    assert chosen is native
    with pytest.raises(RuntimeError, match="gate 5"):
        replay.select_pair_window(native, teacher, 4, 2, np.random.default_rng(3), "approach")


def test_replay_matches_each_original_prefix_and_recomputes_after_weight_change(monkeypatch):
    class SmallActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def forward(self, image, attitude, state):
            state = 0.5 * state + self.weight * image[:, 0, 0, 0, None]
            return state.expand(-1, 4), state

    def rendered(state, gates, **kwargs):
        return state.position[:, :1, None, None].expand(-1, 3, 1, 1)

    monkeypatch.setattr(replay, "render_annular_gates_rgb", rendered)
    data = bank()
    actor = SmallActor()
    starts = [0, 4]
    window = replay.prepare_window(data, (0, 1), starts, 3, torch.device("cpu"))
    kwargs = dict(unroll=3, camera=CameraSpec(), gate_config=GateConfig(), contrast_weight=1)
    prefix = replay.replay_prefix_state(actor, window, kwargs["camera"], kwargs["gate_config"])
    loss, _ = replay.replay_window_loss(actor, window, **kwargs)
    fixed, _ = replay.replay_window_loss(actor, window, **kwargs, diagnostic_fixed_prefix=prefix)
    assert torch.equal(loss, fixed)
    expected = []
    for row, start in enumerate(starts):
        state = torch.zeros(1, 1)
        for step in [0] * 10 + list(range(start)):
            image = rendered(
                type("Pose", (), {"position": data.states[0][step, row : row + 1]})(), ()
            )
            _, state = actor(image, None, state)
        predictions = []
        for step in range(start, start + 3):
            image = rendered(
                type("Pose", (), {"position": data.states[0][step, row : row + 1]})(), ()
            )
            prediction, state = actor(image, None, state)
            predictions.append(prediction[0])
        expected.append(torch.stack(predictions))
    predictions = torch.stack(expected, dim=1)
    expected_loss = torch.stack(
        [
            replay.action_imitation_loss(
                p,
                torch.zeros_like(p),
                torch.ones(2, dtype=torch.bool),
                torch.tensor([0.02, 0.02, 0.01, 0.025]),
                1,
            )[0]
            for p in predictions
        ]
    ).mean()
    assert torch.allclose(loss, expected_loss, atol=1e-5)
    loss.backward()
    assert actor.weight.grad is not None and torch.isfinite(actor.weight.grad)
    with torch.no_grad():
        actor.weight.mul_(2)
    changed, _ = replay.replay_window_loss(actor, window, **kwargs)
    assert torch.allclose(changed.detach(), 4 * loss.detach(), atol=1e-5)
    stale, _ = replay.replay_window_loss(actor, window, **kwargs, diagnostic_fixed_prefix=prefix)
    assert not torch.allclose(changed, stale)
