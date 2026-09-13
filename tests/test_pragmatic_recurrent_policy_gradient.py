from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pragmatic_correlated_exploration import ar1_conditional, joint_log_prob  # noqa: E402
from pragmatic_recurrent_policy_gradient import (  # noqa: E402
    replay_joint_policy_gradient,
    reward_to_go,
)


class ToyActor(torch.nn.Module):
    def __init__(self, recurrent=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.02, 0.01, -0.01, 0.005]))
        self.recurrent = recurrent
        self.calls = 0

    def initial_state(self, count, **kwargs):
        self.calls = 0
        return torch.zeros(count, 4, **kwargs)

    def forward(self, image, attitude, neural):
        self.calls += 1
        mean = image * self.weight + (0.6 * neural if self.recurrent else 0)
        return mean.tanh(), mean


def example(actor, *, warmup=2):
    times, episodes = 7, 2
    image = torch.arange(times * episodes * 4).reshape(times, episodes, 4).float() / 50
    def observe(time):
        return image[time], None
    with torch.no_grad():
        neural = actor.initial_state(episodes, device=torch.device("cpu"), dtype=torch.float32)
        for _ in range(warmup):
            _, neural = actor(*observe(0), neural)
        means = []
        for time in range(times):
            motor, neural = actor(*observe(time), neural)
            means.append(torch.atanh(motor))
        means = torch.stack(means)
    generator = torch.Generator().manual_seed(7)
    latent = means + torch.randn(means.shape, generator=generator) * 0.01
    first = torch.zeros(times, episodes, dtype=torch.bool)
    first[0] = True
    conditional, std = ar1_conditional(
        means, torch.cat((means[:1], means[:-1])), torch.cat((latent[:1], latent[:-1])),
        [0.02] * 4, 0.8, first,
    )
    log_prob = joint_log_prob(latent, conditional, std, squashed=False)
    advantage = torch.linspace(-1, 1, times * episodes).reshape(times, episodes)
    valid = torch.ones(times, episodes, dtype=torch.bool)
    valid[-2:, 0] = False
    return observe, latent, means, log_prob, advantage, valid


def test_chunk_overlap_preserves_previous_mean_gradient_for_feedforward_actor():
    actor = ToyActor()
    args = example(actor)
    gradients = []
    for chunk in (7, 2, 1):
        actor.zero_grad()
        metrics = replay_joint_policy_gradient(
            actor, *args, stationary_std=[0.02] * 4, rho=0.8,
            chunk_steps=chunk, warmup_steps=2,
        )
        gradients.append(actor.weight.grad.clone())
        assert metrics["joint_kl_max"] < 1e-9
        assert metrics["joint_ratio_mean"] == pytest.approx(1)
        assert metrics["replay_forward_calls"] == actor.calls
        assert metrics["valid_commands"] == 12
    assert gradients[0].norm() > 0
    assert torch.allclose(gradients[0], gradients[1], atol=1e-5, rtol=1e-5)
    assert torch.allclose(gradients[0], gradients[2], atol=1e-5, rtol=1e-5)


def test_recurrent_values_stay_continuous_across_chunks_and_rebuild_after_weight_change():
    actor = ToyActor(recurrent=True)
    args = example(actor)
    original = actor.weight.detach().clone()
    metrics = replay_joint_policy_gradient(
        actor, *args, stationary_std=[0.02] * 4, rho=0.8, chunk_steps=2, warmup_steps=2,
    )
    assert metrics["joint_kl_max"] < 1e-9
    assert torch.equal(actor.weight, original)  # Helper takes no optimizer step.
    assert torch.isfinite(actor.weight.grad).all()
    with torch.no_grad():
        actor.weight[0] += 0.001
    changed = []
    for chunk in (2, 7):
        changed.append(replay_joint_policy_gradient(
            actor, *args, stationary_std=[0.02] * 4, rho=0.8,
            chunk_steps=chunk, warmup_steps=2, backward=False,
        ))
    assert changed[0]["joint_kl_mean"] > 0
    assert changed[0]["joint_kl_mean"] == pytest.approx(changed[1]["joint_kl_mean"], rel=1e-5)
    assert changed[0]["loss"] == pytest.approx(changed[1]["loss"], abs=1e-6)


def test_returns_include_later_ground_penalty_after_ring_and_full_clean_tail_bonus():
    # Row0: pass, ring failure, then ground; row1: five passes then final bonus.
    rewards = torch.tensor([[1., 1.], [-2., 1.], [-25., 1.], [0., 1.], [0., 1.], [0., 5.]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    valid[3:, 0] = False
    returns = reward_to_go(rewards, valid)
    assert returns[:, 0].tolist() == [-26, -27, -25, 0, 0, 0]
    assert returns[:, 1].tolist() == [10, 9, 8, 7, 6, 5]


def test_absorbing_rows_cannot_revive_or_hide_tail_rewards():
    with pytest.raises(ValueError, match="revive"):
        reward_to_go(torch.zeros(3, 1), torch.tensor([[True], [False], [True]]))
    with pytest.raises(ValueError, match="zero rewards"):
        reward_to_go(torch.ones(3, 1), torch.tensor([[True], [False], [False]]))


def test_poisoned_inactive_padding_is_rejected_before_gradient_arithmetic():
    actor = ToyActor()
    args = example(actor)
    args[1][-1, 0, 0] = float("nan")  # This command is masked, but still cannot contain NaN.
    with pytest.raises(ValueError, match="padding.*finite"):
        replay_joint_policy_gradient(actor, *args, stationary_std=[0.02] * 4, rho=0.8)
    assert actor.weight.grad is None


def test_poisoned_inactive_observation_rejects_replay_without_changing_weights():
    actor = ToyActor()
    args = list(example(actor))
    original_observe = args[0]
    def poisoned(time):
        image, attitude = original_observe(time)
        image = image.clone()
        if time == 6:
            image[0, 0] = float("nan")
        return image, attitude
    args[0] = poisoned
    initial = actor.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="discard accumulated"):
        replay_joint_policy_gradient(actor, *args, stationary_std=[0.02] * 4, rho=0.8,
                                     chunk_steps=2, warmup_steps=2)
    assert torch.equal(actor.weight, initial)
    actor.zero_grad()  # Required caller cleanup after a rejected replay.


def test_wrong_behavior_density_is_rejected():
    actor = ToyActor()
    args = list(example(actor))
    args[3] = args[3] + 0.01
    with pytest.raises(ValueError, match="unsquashed latent"):
        replay_joint_policy_gradient(actor, *args, stationary_std=[0.02] * 4, rho=0.8)


def test_gradient_scale_supports_weighted_microbatch_accumulation():
    actor = ToyActor()
    args = example(actor)
    gradients = []
    for scale in (1, 0.25):
        actor.zero_grad()
        replay_joint_policy_gradient(actor, *args, stationary_std=[0.02] * 4, rho=0.8,
                                     chunk_steps=2, warmup_steps=2, gradient_scale=scale)
        gradients.append(actor.weight.grad.clone())
    assert torch.allclose(gradients[1], 0.25 * gradients[0], rtol=1e-5, atol=1e-5)


def test_optional_kl_samples_include_only_valid_commands():
    actor = ToyActor()
    args = example(actor)
    result = replay_joint_policy_gradient(actor, *args, stationary_std=[0.02]*4, rho=.8,
                                          warmup_steps=2, backward=False, include_kl_samples=True)
    assert len(result["joint_kl_samples"]) == int(args[-1].sum())
    assert sum(result["joint_kl_samples"]) / 12 == pytest.approx(result["joint_kl_mean"])
