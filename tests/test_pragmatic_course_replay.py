from __future__ import annotations

import sys
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
    loss, _ = replay.replay_window_loss(actor, window, **kwargs)
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
