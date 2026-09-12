from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_pragmatic_two_gate_zero_shot as sequence  # noqa: E402
from evaluate_pragmatic_two_gate_zero_shot import sample_two_gate_cases  # noqa: E402
from train_pragmatic_two_gate_visual_roll_path import new_rollout, score  # noqa: E402

from flydrone.gate import AnnularGate, GateConfig, gate_coordinates  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


@pytest.mark.parametrize("backtrack", [False, True])
@pytest.mark.parametrize("ground_contact", [False, True])
def test_evaluator_distinguishes_clean_completion_from_eventual_passes(
    monkeypatch, backtrack, ground_contact
):
    cases, _ = sample_two_gate_cases(
        1, seed=19, device=torch.device("cpu"), hover_config=HoverConfig(),
        layout="aligned", spacing_range=(1.0, 1.0),
    )
    cases.state.position[:] = torch.tensor([0.0, 0.0, 1.0])
    cases.state.euler.zero_()
    gates = tuple(
        AnnularGate(torch.tensor([[x, 0.0, 1.0]]).expand(2, -1), torch.zeros(2))
        for x in (1.0, 2.0)
    )

    class ScriptedQuad:
        def __init__(self, config):
            self.positions = iter([1.5, 0.5 if backtrack else 1.6, 1.8, 2.5])

        def to(self, device):
            return self

        def __call__(self, rc, state, mass):
            position = state.position.clone()
            position[:, 0] = next(self.positions)
            # A transient physics-substep failure must stay latched even after
            # recovery and eventual successful crossings.
            position[:, 2] = 0.01 if ground_contact and position[0, 0] < 1.7 else 1.0
            return replace(state, position=position)

    class StubActor:
        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

        def __call__(self, image, attitude, neural):
            return torch.zeros(2, 4), neural

    monkeypatch.setattr(sequence, "DifferentiableQuad", ScriptedQuad)
    metrics = sequence.evaluate(
        StubActor(), cases, gates, seconds=0.04, warmup_steps=0,
        camera=CameraSpec(width=32, height=20, horizontal_fov_degrees=125.0),
        hover_config=HoverConfig(), gate_config=GateConfig(back_pattern="checkerboard"),
    )
    compact = sequence.evaluate(
        StubActor(), cases, gates, seconds=0.04, warmup_steps=0,
        camera=CameraSpec(width=32, height=20, horizontal_fov_degrees=125.0),
        hover_config=HoverConfig(), gate_config=GateConfig(back_pattern="checkerboard"),
        compact_policies=1,
    )[0]
    for key, value in compact.items():
        assert value == pytest.approx(metrics[key])
    assert metrics["clean_course_success_episode_indices"] == (
        [] if ground_contact or backtrack else [0, 1]
    )
    if ground_contact:
        assert metrics["clean_course_success_rate"] == 0.0
        assert metrics["gates_before_failure_mean"] == 0.0
        assert metrics["course_progress_score_mean"] <= -2.0
    else:
        assert metrics["all_gates_pass_rate"] == 1.0
        assert metrics["all_gate_success_episode_indices"] == [0, 1]
        assert metrics["clean_course_success_rate"] == (0.0 if backtrack else 1.0)
        assert metrics["wrong_direction_episode_rate"] == float(backtrack)
        assert metrics["course_event_penalty_mean"] == (-2.0 if backtrack else 0.0)
        assert metrics["gates_before_failure_mean"] == (1.0 if backtrack else 2.0)


def test_selection_prefers_clean_flights_and_penalizes_illegal_partial_progress():
    common = dict.fromkeys((
        "first_gate_pass_rate", "first_gate_paired_pass_rate", "all_gates_paired_pass_rate",
        "all_gates_negative_course_pass_rate", "all_gates_positive_course_pass_rate",
        "all_gates_pass_rate",
    ), 1.0)
    clean = {
        **common, "clean_course_paired_success_rate": 1.0,
        "clean_course_success_rate": 1.0, "course_progress_score_mean": 2.0,
    }
    illegal = {
        **common, "clean_course_paired_success_rate": 0.0,
        "clean_course_success_rate": 0.0, "course_progress_score_mean": 0.0,
    }
    assert score(clean) > score(illegal)
    assert score({**illegal, "course_progress_score_mean": 1.0}) > score(illegal)


@pytest.mark.parametrize("start_gate", [1, 2, 3, 4])
def test_short_spacing_late_lessons_start_after_the_previous_gate(start_gate):
    class StubActor:
        def initial_state(self, count, **kwargs):
            return torch.zeros(count, 1, **kwargs)

    rollout = new_rollout(
        StubActor(), StubActor(), pairs=3, seed=234, device=torch.device("cpu"),
        config=HoverConfig(), camera=CameraSpec(width=32, height=20, horizontal_fov_degrees=125.0),
        gate_config=GateConfig(), warmup_steps=0, layout="variable", gate_count=5,
        spacing_range=(0.9, 1.1), lateral_step_range=(0.0, 0.1), lateral_deviation_limit=0.25,
        height_step_range=(0.0, 0.05), height_range=(0.95, 1.25), start_gate=start_gate,
    )
    previous_signed, _, _ = gate_coordinates(rollout.state.position, rollout.gates[start_gate - 1])
    selected_signed, _, _ = gate_coordinates(rollout.state.position, rollout.gates[start_gate])
    assert (previous_signed > GateConfig().drone_radius).all()
    assert (selected_signed < -GateConfig().drone_radius).all()
    events = classify_course_step(
        rollout.state.position,
        rollout.gates[start_gate].center + 0.1 * rollout.gates[start_gate].normal,
        rollout.gates,
        rollout.current,
    )
    assert not events.failed.any()
    assert events.passed[:, start_gate].all()


def test_gate_yaw_jitter_varies_independently_without_changing_course_positions():
    kwargs = dict(
        pairs=8, seed=789, device=torch.device("cpu"), hover_config=HoverConfig(),
        layout="variable", gate_count=5, spacing_range=(0.9, 1.5),
    )
    _, old = sample_two_gate_cases(**kwargs)
    _, varied = sample_two_gate_cases(**kwargs, yaw_jitter_degrees=15.0)
    assert torch.equal(varied[0].yaw, old[0].yaw)
    differences = []
    for original, gate in zip(old[1:], varied[1:], strict=True):
        assert torch.equal(original.center, gate.center)
        delta = gate.yaw - original.yaw
        assert (delta.abs() <= torch.deg2rad(torch.tensor(15.0))).all()
        assert delta.abs().max() > 0.1
        assert torch.allclose(gate.yaw.reshape(-1, 2).sum(dim=1), torch.zeros(8))
        differences.append(delta)
    assert not torch.equal(differences[0], differences[1])
    with pytest.raises(ValueError, match="yaw_jitter_degrees"):
        sample_two_gate_cases(**kwargs, yaw_jitter_degrees=-1.0)


def test_aligned_sequence_inserts_equally_spaced_collinear_gates() -> None:
    cases, gates = sample_two_gate_cases(
        3,
        seed=1234,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="aligned",
        spacing_range=(2.0, 2.0),
        gate_count=3,
    )

    assert len(gates) == 3
    first_displacement = cases.gate.center - cases.state.position
    slope = first_displacement[:, 1] / first_displacement[:, 0]
    for number, gate in enumerate(gates[1:], start=1):
        assert torch.allclose(
            gate.center[:, 0] - gates[0].center[:, 0],
            torch.full((6,), 2.0 * number),
        )
        assert torch.allclose(
            gate.center[:, 1] - gates[0].center[:, 1],
            2.0 * number * slope,
        )
        assert torch.allclose(gate.center[:, 2], torch.full((6,), 1.10))
        assert torch.equal(gate.yaw, gates[0].yaw)


def test_s_turn_rejects_more_than_two_gates() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        sample_two_gate_cases(
            1,
            seed=1234,
            device=torch.device("cpu"),
            hover_config=HoverConfig(),
            layout="s-turn",
            gate_count=3,
        )


def test_variable_sequence_mirrors_lateral_path_and_shares_height_path() -> None:
    cases, gates = sample_two_gate_cases(
        4,
        seed=4321,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="variable",
        spacing_range=(1.0, 1.0),
        gate_count=5,
        lateral_step_range=(0.10, 0.10),
        lateral_deviation_limit=0.25,
        height_step_range=(0.05, 0.05),
        height_range=(0.95, 1.25),
    )

    assert len(gates) == 5
    for gate in gates:
        relative_lateral = gate.center[:, 1] - cases.state.position[:, 1]
        assert torch.allclose(
            relative_lateral.reshape(-1, 2).sum(dim=1),
            torch.zeros(4),
            atol=1.0e-6,
        )
        heights = gate.center[:, 2].reshape(-1, 2)
        assert torch.equal(heights[:, 0], heights[:, 1])

    first_displacement = gates[0].center - cases.state.position
    slope = first_displacement[:, 1] / first_displacement[:, 0]
    deviations = []
    for gate in gates[1:]:
        distance = gate.center[:, 0] - gates[0].center[:, 0]
        centreline = gates[0].center[:, 1] + distance * slope
        deviations.append((gate.center[:, 1] - centreline).abs())
        assert bool((gate.center[:, 2] >= 0.95).all())
        assert bool((gate.center[:, 2] <= 1.25).all())
    assert float(torch.stack(deviations).max()) <= 0.250001
    assert bool((torch.stack(deviations) > 0.0).any())
    assert bool(
        (
            torch.stack([gate.center[:, 2] for gate in gates[1:]])
            != 1.10
        ).any()
    )


def test_variable_sequence_is_seed_deterministic() -> None:
    kwargs = dict(
        pairs=2,
        seed=777,
        device=torch.device("cpu"),
        hover_config=HoverConfig(),
        layout="variable",
        spacing_range=(0.9, 1.1),
        gate_count=5,
    )
    first_cases, first_gates = sample_two_gate_cases(**kwargs)
    second_cases, second_gates = sample_two_gate_cases(**kwargs)

    assert torch.equal(first_cases.state.position, second_cases.state.position)
    for first, second in zip(first_gates, second_gates, strict=True):
        assert torch.equal(first.center, second.center)
        assert torch.equal(first.yaw, second.yaw)
