from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pragmatic_on_policy as online  # noqa: E402
from train_pragmatic_course_on_policy import (  # noqa: E402
    combine_metrics,
    should_select_development,
    stop_for_development_first_regression,
    training_bank_seed,
)

from flydrone.gate import AnnularGate, GateConfig  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ForelegStickPlant,
    HoverConfig,
    QuadState,
    motor_target_for_rc,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.roll = torch.nn.Parameter(torch.tensor(0.01))
        self.register_buffer(
            "nominal",
            motor_target_for_rc(
                torch.tensor([[0.0, 0.0, 0.0, 1 / HoverConfig().thrust_to_weight]])
            )[0],
        )
        self.calls = 0

    def initial_state(self, count, **kwargs):
        return torch.zeros(count, 1, **kwargs)

    def forward(self, image, attitude, neural):
        self.calls += 1
        neural = 0.8 * neural + image[:, :1, 0, 0]
        roll = self.roll + 0.001 * neural[:, 0]
        return torch.cat((roll[:, None], self.nominal[1:].expand(len(roll), -1)), 1), neural


def tiny_setup(monkeypatch, steps=12):
    monkeypatch.setattr(
        online,
        "render_annular_gates_rgb",
        lambda state, *a, **k: state.position[:, :1, None, None].expand(-1, 3, 2, 2),
    )
    config = HoverConfig()
    position = torch.tensor([[0.0, 0.0, 1.1]]).repeat(2, 1)
    velocity = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)
    actuator = torch.zeros(2, 4)
    actuator[:, 0] = 1 / config.thrust_to_weight
    state = QuadState(
        position,
        velocity,
        torch.zeros_like(position),
        torch.zeros_like(position),
        actuator,
        torch.zeros_like(position),
    )
    cases = SimpleNamespace(
        state=state,
        side=torch.tensor([-1.0, 1.0]),
        mass_scale=torch.ones(2),
        sticks=ForelegStickPlant(config).initial_state(
            2, device=torch.device("cpu"), dtype=torch.float32
        ),
    )
    gates = tuple(
        AnnularGate(torch.tensor([[5.0 + i, 0.3, 1.1]]).repeat(2, 1), torch.zeros(2))
        for i in range(5)
    )
    options = dict(seconds=steps / 50, camera=CameraSpec(), config=config, gate_config=GateConfig())
    return cases, gates, options


def test_chunks_detach_gradients_without_resetting_native_flight(monkeypatch):
    cases, gates, options = tiny_setup(monkeypatch)
    actor = TinyActor()
    weights = copy.deepcopy(actor.state_dict())
    nominal, trace, _ = online.fly_course(
        actor, cases, gates, **options, chunk_steps=12, record_trace=True
    )
    assert actor.calls == 22  # ten static warmup observations, then twelve native frames
    chunked, actual, chunks = online.fly_course(
        actor,
        cases,
        gates,
        **options,
        chunk_steps=4,
        backward=True,
        record_trace=True,
        record_chunks=True,
        reference_motors=trace["motors"],
        reference_active=trace["active"],
    )
    assert chunked["final_positions"] == nominal["final_positions"]
    assert torch.equal(actual["motors"], trace["motors"])
    assert chunked["tracking_loss"] == nominal["tracking_loss"]
    assert [item["step"] for item in chunks] == [4, 8, 12]
    assert torch.isfinite(actor.roll.grad) and actor.roll.grad.abs() > 1e-9
    for key, value in weights.items():
        assert torch.equal(value, actor.state_dict()[key])


def test_frozen_tracking_mask_prevents_early_crash_from_erasing_loss(monkeypatch):
    cases, gates, options = tiny_setup(monkeypatch, 3)

    class TouchThenRecover(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            self.calls += 1
            position = state.position.clone()
            position[:, 1] += 0.2
            position[:, 2] = 0.02 if self.calls == 1 else 1.1
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(online, "DifferentiableQuad", TouchThenRecover)
    dynamic, _, _ = online.fly_course(TinyActor(), cases, gates, **options)
    fixed, _, _ = online.fly_course(
        TinyActor(), cases, gates, **options, reference_active=torch.ones(3, 2, dtype=torch.bool)
    )
    assert fixed["ground_contacts"] == fixed["failed_episodes"] == 2
    assert fixed["final_positions"][0][2] == pytest.approx(1.1)
    assert fixed["tracking_loss"] > 5 * dynamic["tracking_loss"]


def test_contacts_after_last_gate_still_fail_the_complete_flight(monkeypatch):
    import evaluate_pragmatic_two_gate_zero_shot as primary

    cases, _, options = tiny_setup(monkeypatch, 4)
    gates = tuple(
        AnnularGate(torch.tensor([[1.0 + i, 0.0, 1.1]]).repeat(2, 1), torch.zeros(2))
        for i in range(5)
    )

    class CompleteThenTouch(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            self.calls += 1
            position = state.position.clone()
            position[:, 0] += 1
            if self.calls == 7:
                position[:, 2] = 0.02
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(online, "DifferentiableQuad", CompleteThenTouch)
    metrics, _, _ = online.fly_course(TinyActor(), cases, gates, **options)
    assert metrics["clean_prefix_gates"] == 10
    assert metrics["clean_first_passes"] == 2
    assert metrics["clean_completions"] == 0
    assert metrics["ground_contacts"] == 2
    # Even all five passes cannot keep the completion bonus after a tail crash.
    assert online.whole_flight_outcome_score(metrics) == 6
    monkeypatch.setattr(primary, "DifferentiableQuad", CompleteThenTouch)
    monkeypatch.setattr(primary, "render_annular_gates_rgb", online.render_annular_gates_rgb)
    verified = primary.evaluate(
        TinyActor(),
        cases,
        gates,
        seconds=options["seconds"],
        warmup_steps=10,
        camera=options["camera"],
        hover_config=options["config"],
        gate_config=options["gate_config"],
    )
    assert metrics["clean_completions"] == 2 * verified["clean_course_success_rate"]
    assert metrics["ground_contacts"] == 2 * verified["ground_contact_rate"]
    assert metrics["clean_first_passes"] == 2 * verified["clean_first_gate_pass_rate"]
    assert metrics["clean_prefix_gates"] == 2 * verified["gates_before_failure_mean"]


def test_exterior_plane_misses_are_not_turned_into_training_failures(monkeypatch):
    cases, _, options = tiny_setup(monkeypatch, 2)
    cases.state.position[:, 1] = 3.0
    gates = tuple(
        AnnularGate(torch.tensor([[0.01 + i, 0.0, 1.1]]).repeat(2, 1), torch.zeros(2))
        for i in range(5)
    )
    # CoursePath requires >.05 m spacing from launch, but the plane can be approached
    # in Y without changing its normal. Put launch just behind the first plane.
    gates = tuple(AnnularGate(g.center + torch.tensor([0.05, 0.0, 0.0]), g.yaw) for g in gates)
    cases.state.velocity[:, 0] = 3.0
    metrics, _, _ = online.fly_course(TinyActor(), cases, gates, **options)
    assert metrics["final_positions"][0][0] > 0.06
    assert metrics["failed_episodes"] == 0
    assert metrics["clean_prefix_gates"] == 0


@pytest.mark.parametrize("fault", ["wrong_order", "wrong_direction", "ring_contacts"])
def test_illegal_or_contact_first_then_recovery_never_earns_clean_outcome(monkeypatch, fault):
    cases, _, options = tiny_setup(monkeypatch, 3)
    first_y = 0.7 if fault == "ring_contacts" else 2.0
    gates = tuple(
        AnnularGate(
            torch.tensor([[1.0 + i, first_y if i == 0 else 0.0, 1.1]]).repeat(2, 1),
            torch.zeros(2),
        )
        for i in range(5)
    )
    paths = {
        "wrong_order": [(2.5, 0), (2.5, 2), (0, 2), (1.5, 2), (1.5, 0), (6, 0)],
        "wrong_direction": [(1.5, 0), (1.5, 2), (0, 2), (1.5, 2), (1.5, 0), (6, 0)],
        "ring_contacts": [(1.5, 0), (0, 0), (0, 0.7), (1.5, 0.7), (1.5, 0), (6, 0)],
    }

    class FailThenClearAll(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            position = state.position.clone()
            position[:, :2] = position.new_tensor(paths[fault][self.calls])
            self.calls += 1
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(online, "DifferentiableQuad", FailThenClearAll)
    metrics, _, chunks = online.fly_course(
        TinyActor(), cases, gates, **options, chunk_steps=1, record_chunks=True
    )
    # All five physical gates are eventually passed in sequence, but only after
    # the initial violation. Neither recovery nor role changes reset the failure.
    assert chunks[-1]["role"].tolist() == [5, 5]
    assert metrics[fault] == metrics["failed_episodes"] == 2
    assert metrics["clean_completions"] == metrics["clean_prefix_gates"] == 0
    assert metrics["ground_contacts"] == metrics["invalid_episodes"] == 0
    assert online.whole_flight_outcome_score(metrics) == -4


def admissible_metrics():
    return dict(
        objective=1.0,
        continuous_objective=1.0,
        clean_completions=1,
        clean_prefix_gates=7,
        clean_first_by_side=[1, 1],
        ground_contacts=0,
        invalid_episodes=0,
        failed_episodes=1,
        ring_contacts=1,
        wrong_order=0,
        wrong_direction=0,
    )


def test_full_flight_guard_preserves_every_failure_category_and_clean_prefix():
    base = admissible_metrics()
    candidate = dict(base, objective=0.5, continuous_objective=0.5)
    assert online.whole_flight_trial_admissible(candidate, base)
    for key in (
        "ground_contacts",
        "invalid_episodes",
        "failed_episodes",
        "ring_contacts",
        "wrong_order",
        "wrong_direction",
    ):
        assert not online.whole_flight_trial_admissible(
            dict(candidate, **{key: base[key] + 1}), base
        )
    for key in ("clean_completions", "clean_prefix_gates"):
        assert not online.whole_flight_trial_admissible(
            dict(candidate, **{key: base[key] - 1}), base
        )
    assert not online.whole_flight_trial_admissible(
        dict(candidate, clean_first_by_side=[0, 2]), base
    )
    assert not online.whole_flight_trial_admissible(dict(candidate, continuous_objective=1.1), base)


def test_microbatch_metrics_add_counts_but_average_objectives(monkeypatch):
    cases, gates, options = tiny_setup(monkeypatch, 2)
    metrics, _, _ = online.fly_course(TinyActor(), cases, gates, **options)
    result = combine_metrics([metrics, metrics])
    assert result["episodes"] == 4
    assert result["objective"] == metrics["objective"]
    assert result["failed_episodes"] == 2 * metrics["failed_episodes"]
    assert metrics["phase_frames_by_side"] == [[2, 0, 0, 0, 0], [2, 0, 0, 0, 0]]
    with pytest.raises(ValueError):
        combine_metrics([dict(metrics, episodes=4)])


def test_flight_first_mode_can_prioritize_real_gate_gains_over_tracking_surrogate():
    base = admissible_metrics()
    candidate = dict(base, continuous_objective=1.1, clean_completions=2, clean_prefix_gates=9)
    assert not online.whole_flight_trial_admissible(candidate, base)
    assert online.whole_flight_trial_admissible(candidate, base, mode="flight-first")
    assert not online.whole_flight_trial_admissible(
        dict(candidate, ground_contacts=1), base, mode="flight-first"
    )
    assert not online.whole_flight_trial_admissible(
        dict(base, continuous_objective=1.1), base, mode="flight-first"
    )
    with pytest.raises(ValueError):
        online.whole_flight_trial_admissible(candidate, base, mode="unknown")


def test_outcome_mode_allows_course_tradeoffs_without_redefining_clean_flights():
    base = dict(
        admissible_metrics(),
        clean_completions=2,
        clean_prefix_gates=14,
        failed_episodes=2,
        ring_contacts=2,
        clean_first_by_side=[3, 3],
    )
    candidate = dict(
        base,
        continuous_objective=2.0,
        clean_completions=3,
        clean_prefix_gates=18,
        failed_episodes=3,
        ring_contacts=3,
        wrong_order=2,
        wrong_direction=1,
        clean_first_by_side=[2, 4],
    )
    assert online.whole_flight_outcome_score(base) == 20
    assert online.whole_flight_outcome_score(candidate) == 27
    assert online.whole_flight_trial_admissible(candidate, base, mode="outcome")
    assert not online.whole_flight_trial_admissible(candidate, base, mode="flight-first")
    # A training-score gain still cannot buy a single ground/invalid episode,
    # even when the nominal source already had the same safety count.
    for key in ("ground_contacts", "invalid_episodes"):
        unsafe = dict(candidate, **{key: 1})
        assert not online.whole_flight_trial_admissible(unsafe, base, mode="outcome")
        assert not online.whole_flight_trial_admissible(
            unsafe, dict(base, **{key: 1}), mode="outcome"
        )


def test_outcome_is_a_proxy_that_can_trade_one_completion_for_several_prefix_gains():
    base = dict(admissible_metrics(), clean_completions=2, clean_prefix_gates=13, failed_episodes=2)
    candidate = dict(base, clean_completions=1, clean_prefix_gates=21, failed_episodes=3)
    assert online.whole_flight_outcome_score(candidate) == 20
    assert online.whole_flight_outcome_score(base) == 19
    assert online.whole_flight_trial_admissible(candidate, base, mode="outcome")
    # Avoiding every gate can beat a crash-at-launch score, but not this source.
    idle = dict(base, clean_completions=0, clean_prefix_gates=0, failed_episodes=0)
    assert not online.whole_flight_trial_admissible(idle, base, mode="outcome")


def test_outcome_mode_uses_tracking_only_for_ties_and_requires_finite_loss():
    base = admissible_metrics()
    better_tracking = dict(base, objective=0.5, continuous_objective=0.5)
    assert online.whole_flight_trial_admissible(better_tracking, base, mode="outcome")
    assert not online.whole_flight_trial_admissible(base, base, mode="outcome")
    assert not online.whole_flight_trial_admissible(
        dict(better_tracking, clean_prefix_gates=6), base, mode="outcome"
    )
    for key in ("objective", "continuous_objective"):
        for value in (float("nan"), float("inf"), -float("inf")):
            assert not online.whole_flight_trial_admissible(
                dict(better_tracking, clean_prefix_gates=8, **{key: value}), base, mode="outcome"
            )


def test_training_course_rotation_is_predetermined_and_supports_new_banks():
    assert [training_bank_seed(100, update, 4) for update in range(1, 7)] == [
        100,
        101,
        102,
        103,
        100,
        101,
    ]
    assert [training_bank_seed(100, update) for update in range(1, 7)] == [
        100,
        101,
        102,
        103,
        104,
        105,
    ]
    with pytest.raises(ValueError):
        training_bank_seed(100, 0, 4)


@pytest.mark.parametrize(
    ("changed", "evaluated", "expected"), [(0, 0, False), (2, 0, True), (2, 2, False), (3, 2, True)]
)
def test_development_selection_requires_a_new_controller_not_a_better_repeat(
    changed, evaluated, expected
):
    source = dict(
        clean_course_success_rate=11 / 32,
        clean_course_negative_success_rate=8 / 16,
        clean_course_positive_success_rate=3 / 16,
        first_gate_pass_rate=28 / 32,
        clean_first_gate_pass_rate=28 / 32,
        gates_before_failure_mean=3.0,
        course_race_fitness=0.1,
    )
    improved = dict(source, clean_course_success_rate=13 / 32)
    options = dict(controller_change_update=changed, last_evaluated_change_update=evaluated)
    assert should_select_development(improved, source, 0.825, **options) is expected
    # A new weight set still has to pass the existing performance and first-gate rules.
    assert not should_select_development(source, source, 0.825, **options)
    assert not should_select_development(
        dict(improved, clean_first_gate_pass_rate=0.5), source, 0.825, **options
    )


def test_outcome_development_is_completion_first_despite_training_proxy_tradeoffs():
    source = dict(
        clean_course_success_rate=12 / 32,
        clean_course_negative_success_rate=9 / 16,
        clean_course_positive_success_rate=3 / 16,
        first_gate_pass_rate=28 / 32,
        clean_first_gate_pass_rate=28 / 32,
        gates_before_failure_mean=3.0,
        course_race_fitness=3.0,
    )
    improved = dict(
        source,
        clean_course_success_rate=18 / 32,
        clean_course_positive_success_rate=9 / 16,
        first_gate_pass_rate=25 / 32,
        clean_first_gate_pass_rate=25 / 32,
    )
    options = dict(
        controller_change_update=2, last_evaluated_change_update=0, acceptance_mode="outcome"
    )
    assert should_select_development(improved, source, 0.825, **options)
    assert not should_select_development(
        dict(improved, clean_course_success_rate=11 / 32, course_race_fitness=100),
        source,
        0.825,
        **options,
    )
    assert not should_select_development(
        improved, source, 0.825, **dict(options, controller_change_update=0)
    )
    assert not stop_for_development_first_regression(improved, 0.825, acceptance_mode="outcome")
    for mode in ("continuous", "flight-first"):
        assert stop_for_development_first_regression(improved, 0.825, acceptance_mode=mode)
        assert not should_select_development(
            improved, source, 0.825, **dict(options, acceptance_mode=mode)
        )


def test_tracking_validity_ends_after_miss_but_flight_and_frozen_candidate_mask_continue(
    monkeypatch,
):
    cases, _, options = tiny_setup(monkeypatch, 3)
    cases.state.position[:, 1] = 3.0
    gates = tuple(
        AnnularGate(torch.tensor([[0.15 + i, 0.0, 1.1]]).repeat(2, 1), torch.zeros(2))
        for i in range(5)
    )

    class ExteriorTouchAndReturn(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.calls = 0

        def forward(self, rc, state, mass):
            self.calls += 1
            position = state.position.clone()
            position[:, 0] = 0.2 if self.calls == 1 else 0.1
            return QuadState(position, *state.as_tuple()[1:])

    monkeypatch.setattr(online, "DifferentiableQuad", ExteriorTouchAndReturn)
    nominal, trace, _ = online.fly_course(TinyActor(), cases, gates, **options, record_trace=True)
    assert trace["active"].tolist() == [[True, True], [False, False], [False, False]]
    assert nominal["tracking_reference_end_steps"] == [1, 1]
    assert nominal["failed_episodes"] == 0  # reference validity must not alter course rules
    assert nominal["phase_frames"] == [6, 0, 0, 0, 0]  # all six flight observations still occur
    assert nominal["tracking_phase_frames_by_side"] == [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    assert nominal["tracking_loss"] > 0  # retain the crossing frame itself
    assert nominal["current_trajectory_post_miss_tracking_loss"] > 0
    assert nominal["tracking_mask_source"] == "current nominal eligibility"
    # If the proposal misses before its nominal reference did, do not remove any
    # additional samples from the already frozen training objective.
    candidate, _, _ = online.fly_course(
        TinyActor(), cases, gates, **options, reference_active=torch.ones(3, 2, dtype=torch.bool)
    )
    assert candidate["tracking_loss"] == pytest.approx(3 * nominal["tracking_loss"])
    assert candidate["tracking_mask_source"] == "frozen nominal reference"
    assert candidate["failed_episodes"] == 0
