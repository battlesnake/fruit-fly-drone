from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pragmatic_on_policy as online  # noqa: E402
from train_pragmatic_course_on_policy import combine_metrics  # noqa: E402

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
