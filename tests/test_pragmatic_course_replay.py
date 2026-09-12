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


def test_from_start_labels_change_only_roll_including_launch():
    reference = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = reference.clone()
    teacher = -torch.ones_like(reference)
    target = replay.late_roll_preservation_targets(
        teacher, reference, torch.tensor([0, 1, 4]), roll_from_start=True
    )
    assert torch.equal(target[:, 0], teacher[:, 0])
    assert torch.equal(target[:, 1:], reference[:, 1:])
    assert torch.equal(reference, original)


@pytest.mark.parametrize("kind", ["native", "roll-assisted"])
@pytest.mark.parametrize("from_start", [False, True])
def test_collection_timing_keeps_native_actions_and_source_other_axes(
    monkeypatch, kind, from_start
):
    class Actor(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.output = torch.tensor(output)
            self.calls = 0

        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def forward(self, image, attitude, neural):
            assert image.shape == (2, 3, 1, 1)
            assert attitude.shape == (2, 2)
            self.calls += 1
            return self.output.expand(2, -1), neural + 1

    motors = []

    class Legs:
        def __init__(self, config):
            pass

        def to(self, device):
            return self

        def __call__(self, motor, sticks):
            motors.append(motor.clone())
            return motor, sticks

    class Quad(Legs):
        def __call__(self, rc, state, mass):
            return state

    monkeypatch.setattr(replay, "render_annular_gates_rgb", lambda *a, **k: torch.zeros(2, 3, 1, 1))
    monkeypatch.setattr(
        replay,
        "replay_teacher_motor",
        lambda *a: torch.tensor([[0.2, 0.3, 0.4, 0.5]]).expand(2, -1),
    )
    monkeypatch.setattr(replay, "ForelegStickPlant", Legs)
    monkeypatch.setattr(replay, "DifferentiableQuad", Quad)
    learner, reference = Actor([-0.1, -0.2, -0.3, -0.4]), Actor([0.11, 0.12, 0.13, 0.14])
    result = replay.collect_bank(
        learner,
        1,
        5,
        kind,
        0.04,
        CameraSpec(),
        replay.HoverConfig(),
        GateConfig(),
        reference_controller=reference,
        roll_teacher="current-gate",
        roll_from_start=from_start,
    )
    expected = reference.output.clone()
    if from_start:
        expected[0] = 0.2
    assert torch.equal(result.target, expected.expand(2, 2, -1))
    action = learner.output if kind == "native" else expected
    assert all(torch.equal(motor, action.expand(2, -1)) for motor in motors)
    assert reference.calls == 12
    assert learner.calls == (12 if kind == "native" else 0)
    assert result.manifest()["roll_from_start"] is from_start


def test_from_start_collection_requires_source_preservation():
    with pytest.raises(ValueError, match="from-start roll requires"):
        replay.collect_bank(None, 1, 1, "native", 1, None, None, None, roll_from_start=True)


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


def test_balanced_replay_plan_preserves_half_early_and_half_late_weight():
    assert replay.replay_update_plan(3, True) == [
        (0, 0.5, "early-source"),
        (3, 0.25, "native-late"),
        (3, 0.25, "assisted-late"),
    ]
    assert replay.replay_update_plan(3, False) == [
        (0, 0.5, "native-first"),
        (3, 0.5, "native-first"),
    ]


def test_balanced_early_windows_never_cross_into_teacher_roll_labels():
    native, assisted = bank(), bank("roll-assisted")
    native.current[4:] = 1
    for seed in range(8):
        window, record = replay.select_balanced_window(
            native, assisted, 0, 3, np.random.default_rng(seed), "transition", torch.device("cpu")
        )
        assert record["matched_mirrored_reset"]
        assert record["entirely_pre_first_gate"]
        assert record["kinds"] == ["approach", "approach"]
        for side, start in enumerate(record["starts"]):
            assert torch.equal(window[2][start : start + 3, side], torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("stage", ["launch", "early-approach", "middle", "late-approach"])
def test_early_stage_sampling_covers_distinct_approach_parts(stage):
    native = bank(steps=30)
    native.current[15:, 0] = 1
    native.current[27:, 1] = 1
    window, record = replay.select_balanced_window(
        native,
        native,
        0,
        3,
        np.random.default_rng(1),
        "approach",
        torch.device("cpu"),
        early_stage=stage,
    )
    assert record["early_stage"] == stage
    for side, stop in enumerate((15, 27)):
        options = list(range(stop - 2))
        eligible = replay.early_stage_options(options, stage)
        start = record["starts"][side]
        assert start in eligible
        if stage == "launch":
            assert start == 0
        assert bool((window[2][start : start + 3, side] == 0).all())


def test_early_stage_does_not_invent_launch_or_relabel_short_failed_flight():
    assert replay.early_stage_options([1, 2, 3], "launch") == []
    assert replay.early_stage_options([], "middle") == []
    assert replay.early_stage_options([0], "late-approach") == []
    with pytest.raises(ValueError, match="unknown early replay stage"):
        replay.early_stage_options([0], "pre-crossing")


def test_approach_thirds_cover_every_eligible_start():
    options = list(range(300))
    parts = [
        replay.early_stage_options(options, stage)
        for stage in ("early-approach", "middle", "late-approach")
    ]
    assert [start for part in parts for start in part] == options


def test_missing_native_side_uses_only_that_assisted_side_and_preserves_physical_history():
    native, assisted = bank(), bank("roll-assisted")
    native.current[:, 0] = 1
    native.active[6:, 0] = False
    assisted.current[7:, 1] = 1
    assisted.states[0][:, :, 0] += 100
    for gate in assisted.gates:
        gate.center[:, 0] = 100
    window, record = replay.select_balanced_window(
        native, assisted, 1, 2, np.random.default_rng(4), "approach", torch.device("cpu")
    )
    assert not record["matched_mirrored_reset"]
    assert record["source_by_side"] == ["native", "roll-assisted"]
    assert record["substituted_sides"] == [1]
    assert not record["difference_loss_is_controlled_geometry_contrast"]
    assert window[1][0].center[:, 0].tolist() == [0.0, 100.0]
    assert len(window[2]) == max(record["starts"]) + 2
    for side, original in enumerate((native, assisted)):
        start, row = record["starts"][side], record["rows"][side]
        for field in range(6):
            assert torch.equal(
                window[0][field][: start + 2, side], original.states[field][: start + 2, row]
            )
        assert torch.equal(window[3][: start + 2, side], original.target[: start + 2, row])


def test_independent_native_courses_are_preferred_to_assisted_substitution():
    native, assisted = bank(), bank("roll-assisted")
    native.states = tuple(v.repeat(1, 2, 1) for v in native.states)
    native.current = native.current.repeat(1, 2)
    native.active = native.active.repeat(1, 2)
    native.target = native.target.repeat(1, 2, 1)
    native.gates = tuple(AnnularGate(g.center.repeat(2, 1), g.yaw.repeat(2)) for g in native.gates)
    native.current[:, (0, 3)] = 1
    assisted.current[:] = 1
    _, record = replay.select_balanced_window(
        native, assisted, 1, 2, np.random.default_rng(5), "approach", torch.device("cpu")
    )
    assert record["rows"] == [0, 3]
    assert record["source_by_side"] == ["native", "native"]
    assert record["substituted_sides"] == []
    assert not record["matched_mirrored_reset"]


def test_balanced_replay_fails_explicitly_when_a_side_has_no_valid_examples():
    native, assisted = bank(), bank("roll-assisted")
    native.current[:, 0] = 1
    with pytest.raises(RuntimeError, match="side 1.*gate 2"):
        replay.select_balanced_window(
            native, assisted, 1, 2, np.random.default_rng(5), "approach", torch.device("cpu")
        )


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
