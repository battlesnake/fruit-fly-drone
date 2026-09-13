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
    image, attitude = selected.observation(1, options["camera"], options["gate_config"])
    assert image.shape == (1, 3, 20, 32) and attitude.shape == (1, 2)


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
