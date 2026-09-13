from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_pragmatic_two_gate_zero_shot as evaluation  # noqa: E402
import pragmatic_policy_rollout as collection  # noqa: E402
from pragmatic_recurrent_policy_gradient import replay_joint_policy_gradient  # noqa: E402

from flydrone.gate import AnnularGate, GateConfig  # noqa: E402
from flydrone.gate_course import classify_course_step  # noqa: E402
from flydrone.hover import HoverConfig, QuadState  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402


class StubActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0., 0., 0., 0.1]))

    def initial_state(self, count, **kwargs):
        self.calls = 0
        return torch.zeros(count, 1, **kwargs)

    def forward(self, image, attitude, neural):
        assert image.ndim == 4 and image.shape[1] == 3
        assert attitude.shape == (len(image), 2)  # No critic or physical-state actor inputs.
        self.calls += 1
        return self.bias.tanh().expand(len(image), -1), neural + 0.01


class ScriptedQuad:
    def __init__(self, config):
        self.step = 0

    def to(self, device):
        return self

    def __call__(self, rc, state, mass):
        positions = [0.5, 1.1, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5]
        position = state.position.clone()
        position[:, 0] = positions[self.step]
        position[:, 2] = 1
        if self.step == 5:
            position[0, 2] = 0.01  # Contact and recovery in the baseline; absorb in collection.
        self.step += 1
        return replace(state, position=position)


def setup_case(monkeypatch):
    cases, _ = evaluation.sample_two_gate_cases(
        1, seed=91, device=torch.device("cpu"), hover_config=HoverConfig(),
    )
    cases.state.position[:] = torch.tensor([0., 0., 1.])
    cases.state.euler.zero_()
    gates = tuple(AnnularGate(torch.tensor([[x, 0., 1.]]).expand(2, -1), torch.zeros(2))
                  for x in (1., 2.))
    monkeypatch.setattr(collection, "DifferentiableQuad", ScriptedQuad)
    monkeypatch.setattr(evaluation, "DifferentiableQuad", ScriptedQuad)
    return cases, gates, dict(camera=CameraSpec(32, 20, 125), config=HoverConfig(),
                             gate_config=GateConfig(back_pattern="checkerboard"),
                             seconds=0.08, warmup_steps=2)


def test_absorbing_collection_matches_full_evaluator_return_and_keeps_clean_tail(monkeypatch):
    cases, gates, options = setup_case(monkeypatch)
    actor = StubActor()
    data = collection.collect_policy_rollout(actor, cases, gates, stationary_std=None, **options)
    assert actor.calls == 6  # Two warmup frames, four physical commands.
    metrics = evaluation.evaluate(
        actor, cases, gates, hover_config=options["config"],
        **{k: v for k, v in options.items() if k != "config"},
    )
    expected = metrics["course_race_fitness"] - 25 * metrics["ground_or_invalid_rate"]
    assert data.metrics["search_fitness"] == pytest.approx(expected, abs=1e-5)
    assert data.metrics["clean_completions"] == 1
    assert data.metrics["ground_contacts"] == data.metrics["invalid_episodes"] == 1
    assert data.valid[:, 0].tolist() == [True, True, True, False]
    assert data.valid[:, 1].tolist() == [True] * 4
    assert data.failed_before_command[:, 0].tolist() == [False, False, False, True]
    # Passing every gate is not failure; the full clean-flight tail still earns +5.
    assert not bool(data.failed_before_command[:, 1].any())
    assert data.rewards[2, 0] == -27  # One generic failure + one ground/invalid union cost.
    assert data.rewards[3, 0] == 0 and data.rewards[3, 1] == 5
    assert data.states[0][-1, 0, 2] == 1  # Finite pre-contact padding, not the invalid proposal.
    assert data.states[0][-1, 0, 0] == pytest.approx(2.2)


def test_critic_features_precede_innovation_and_replay_rows_keep_sensor_interface(monkeypatch):
    cases, gates, options = setup_case(monkeypatch)
    actor = StubActor()
    a = collection.collect_policy_rollout(actor, cases, gates, noise_seed=1, **options)
    b = collection.collect_policy_rollout(actor, cases, gates, noise_seed=2, **options)
    assert torch.equal(a.critic_features[0], b.critic_features[0])
    assert not torch.equal(a.latents[0], b.latents[0])
    assert a.critic_features.shape[:2] == a.valid.shape
    selected = a.select([1], torch.device("cpu"))
    assert selected.valid.shape == (4, 1)
    assert torch.equal(selected.latents[:, 0], a.latents[:, 1])
    assert torch.equal(selected.failed_before_command[:, 0], a.failed_before_command[:, 1])
    old_archive = replace(a, failed_before_command=None).select([1], torch.device("cpu"))
    assert old_archive.failed_before_command is None
    image, attitude = selected.observation(1, options["camera"], options["gate_config"])
    assert image.shape == (1, 3, 20, 32) and attitude.shape == (1, 2)


def test_sink_cache_includes_warmup_and_records_presynaptic_activity_before_each_tick(monkeypatch):
    cases, gates, options = setup_case(monkeypatch)
    actor = StubActor()
    data = collection.collect_policy_rollout(
        actor, cases, gates, record_sink_parents=torch.tensor([0]), **options,
    )
    assert actor.calls == 6
    assert data.sink_features.shape == (6, 2, 1)
    assert torch.allclose(data.sink_features[:, 0, 0], torch.tanh(torch.arange(6)*.01))
    selected = data.select([1], torch.device("cpu"))
    assert torch.equal(selected.sink_features[:, 0], data.sink_features[:, 1])
    assert selected.sink_parent_nodes.tolist() == [0]


def test_collected_unsquashed_densities_replay_with_correct_joint_policy_gradient(monkeypatch):
    cases, gates, options = setup_case(monkeypatch)
    actor = StubActor()
    data = collection.collect_policy_rollout(actor, cases, gates, noise_seed=9, **options)
    def observe(time):
        return data.observation(time, options["camera"], options["gate_config"])
    advantages = data.returns - data.returns[data.valid].mean()
    stats = replay_joint_policy_gradient(
        actor, observe, data.latents, data.old_means, data.old_log_prob, advantages, data.valid,
        stationary_std=data.stationary_std, rho=data.rho, chunk_steps=2,
        warmup_steps=data.warmup_steps,
    )
    assert stats["joint_kl_max"] < 1e-9
    assert stats["joint_ratio_mean"] == pytest.approx(1)
    assert torch.isfinite(actor.bias.grad).all() and actor.bias.grad.norm() > 0


def test_ring_failure_does_not_erase_later_ground_cost_or_allow_same_tick_pass():
    config = GateConfig()
    gate = AnnularGate(torch.tensor([[1., 0., 1.]]), torch.zeros(1))
    tracker = collection.CourseRewardTracker(1, 1, torch.device("cpu"))
    # Clip the annulus, rather than pass its aperture.
    previous = torch.tensor([[0., 0.7, 1.]])
    position = torch.tensor([[1.1, 0.7, 1.]])
    event = classify_course_step(
        previous, position, (gate,), torch.zeros(1, dtype=torch.long), config,
    )
    def state_at(position):
        return QuadState(position, torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
                         torch.zeros(1, 4), torch.zeros(1, 3))
    reward = tracker.step(event, state_at(position), config)
    assert reward.item() == -2 and tracker.failed.item() and not tracker.absorbed.item()
    ground = position.clone()
    ground[:, 2] = 0.01
    event = classify_course_step(
        position, ground, (gate,), torch.zeros(1, dtype=torch.long), config,
    )
    reward = tracker.step(event, state_at(ground), config)
    assert reward.item() == -25 and tracker.absorbed.item()
    assert tracker.step(event, state_at(ground), config).item() == 0

    tracker = collection.CourseRewardTracker(1, 1, torch.device("cpu"))
    passed = classify_course_step(torch.tensor([[0., 0., 1.]]), torch.tensor([[1.1, 0., 1.]]),
                                  (gate,), torch.zeros(1, dtype=torch.long), config)
    assert passed.passed.item()
    assert tracker.step(passed, state_at(ground), config).item() == -27
    assert tracker.prefix.item() == 0 and not tracker.clean().item()


@pytest.mark.parametrize("failure_tick", [0, 1])
@pytest.mark.parametrize("later_ground", [False, True])
def test_recorded_failure_precedes_both_ticks_and_keeps_full_tail_safety_credit(
    monkeypatch, failure_tick, later_ground,
):
    from pragmatic_policy_optimization import OutcomeCritic, round_advantages

    cases, gates, options = setup_case(monkeypatch)
    cases.state.position[0, 1] = .7  # First episode clips the annulus.

    class RingThenGroundQuad(ScriptedQuad):
        def __call__(self, rc, state, mass):
            time = self.step
            result = super().__call__(rc, state, mass)
            if time == 0 and failure_tick == 0:
                result.position[:, 0] = 1.01
            if time == 5 and not later_ground:
                result.position[0, 2] = 1.
            return result

    monkeypatch.setattr(collection, "DifferentiableQuad", RingThenGroundQuad)
    data = collection.collect_policy_rollout(StubActor(), cases, gates, **options)
    assert data.failed_before_command[:, 0].tolist() == [False, True, True, True]
    assert data.failed_before_command[:, 1].tolist() == [False]*4
    assert data.rewards[0, 0] == -2
    assert data.rewards[2, 0] == (-25 if later_ground else 0)
    assert data.rewards[-1, 1] == 5  # Still-healthy completed episode earns clean bonus.
    assert data.valid[:, 0].tolist() == [True, True, True, not later_ground]
    critic = OutcomeCritic(data.critic_features.shape[-1])
    original, old_stats = round_advantages(data, critic, zero_baseline=True)
    revised, _ = round_advantages(data, critic, zero_baseline=True, failure_aware=True)
    assert revised[0, 0] == original[0, 0]  # No hindsight reclassification of causing action.
    assert torch.equal(revised[:, 1], original[:, 1])
    if later_ground:
        assert revised[1:3, 0].tolist() == pytest.approx([-25 / old_stats["raw_std"]]*2)
        assert revised[3, 0] == 0
    else:
        assert torch.equal(revised[1:, 0], torch.zeros(3))
