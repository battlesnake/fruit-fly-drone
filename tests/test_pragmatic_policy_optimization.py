from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pragmatic_policy_optimization as ppo  # noqa: E402


def diagnostics(samples=(0., 0.), rms=(0., 0., 0., 0.)):
    return dict(valid_commands=len(samples), joint_kl_samples=list(samples),
                loss=0.2, joint_ratio_mean=1., clipped_command_fraction=0.,
                native_latent_mean_displacement_rms=list(rms), native_clamped_fraction=[0.] * 4)


def test_aggregate_uses_global_quantile_and_count_weighted_squared_displacements():
    parts = [diagnostics([0.] * 198, [0.01]*4), diagnostics([0.2, 0.2], [0.03]*4)]
    stats = ppo.aggregate_replays(parts)
    assert stats["joint_kl_p99"] == pytest.approx(
        float(torch.quantile(torch.tensor([0.] * 198 + [0.2] * 2), .99)))
    assert stats["joint_kl_p99"] < .01  # Not the mean (.1) of the two batch p99s.
    expected = (.99 * .01**2 + .01 * .03**2)**.5
    assert stats["native_latent_mean_displacement_rms"] == pytest.approx([expected]*4)
    assert stats["joint_kl_mean"] == pytest.approx(.002)


@pytest.mark.parametrize("mean,p99,shift,accepted,stop", [
    (0.004, .09, .49, True, False), (.006, .09, .49, True, True),
    (.011, .09, .49, False, True), (.004, .11, .49, False, True),
    (.004, .09, .51, False, True),
])
def test_trust_bounds(mean, p99, shift, accepted, stop):
    decision = ppo.trust_decision(dict(joint_kl_mean=mean, joint_kl_p99=p99,
                                     native_latent_mean_displacement_rms=[shift*.01]*4), [.01]*4)
    assert decision["accepted"] is accepted and decision["stop_round"] is stop


class ParameterActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([2., .2, .3]))


def test_proposal_masks_gradients_and_restores_adam_on_rejection_or_error():
    actor = ParameterActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    mask = torch.tensor([True, False, True])
    unsafe = False
    raises = False
    calls = []

    def replay(backward):
        calls.append(backward)
        if backward:
            actor.edge_magnitude.sum().backward()
        elif raises:
            raise FloatingPointError("deliberate replay failure")
        return dict(joint_kl_mean=.02 if unsafe else .001, joint_kl_p99=.002,
                    native_latent_mean_displacement_rms=[.001]*4)

    initial = actor.edge_magnitude.detach().clone()
    stats = ppo.actor_proposal(actor, optimizer, mask, replay, [.01]*4)
    assert stats["accepted"] and calls == [True, False]
    assert actor.edge_magnitude[0] > 1  # Preserve existing [0,8] magnitude bound.
    assert actor.edge_magnitude[1] == initial[1]
    assert not torch.equal(actor.edge_magnitude, initial)
    before = actor.edge_magnitude.detach().clone()
    adam_before = copy.deepcopy(optimizer.state_dict())
    unsafe = True
    assert not ppo.actor_proposal(actor, optimizer, mask, replay, [.01]*4)["accepted"]
    assert torch.equal(actor.edge_magnitude, before)
    for key, value in adam_before["state"][0].items():
        assert torch.equal(optimizer.state_dict()["state"][0][key], value)
    raises = True
    with pytest.raises(FloatingPointError, match="deliberate"):
        ppo.actor_proposal(actor, optimizer, mask, replay, [.01]*4)
    assert torch.equal(actor.edge_magnitude, before) and actor.edge_magnitude.grad is None
    for key, value in adam_before["state"][0].items():
        assert torch.equal(optimizer.state_dict()["state"][0][key], value)


def test_advantages_global_and_frozen_before_independent_critic_fit():
    torch.manual_seed(5)
    data = SimpleNamespace(critic_features=torch.randn(8, 2, 3),
                           returns=torch.arange(16).reshape(8, 2).float(),
                           valid=torch.ones(8, 2, dtype=torch.bool))
    data.valid[-2:, 0] = False
    critic = ppo.OutcomeCritic(3)
    critic.initialize_normalization(data.critic_features[data.valid])
    normalized = (critic.center.clone(), critic.scale.clone())
    advantages, stats = ppo.round_advantages(data, critic, zero_baseline=True)
    frozen = advantages.clone()
    assert stats["zero_baseline"]
    assert advantages[data.valid].mean().item() == pytest.approx(0, abs=1e-6)
    assert advantages[data.valid].std(unbiased=False).item() == pytest.approx(1)
    assert not advantages.requires_grad and (advantages[~data.valid] == 0).all()
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    ppo.fit_critic(critic, optimizer, data, epochs=2, batch_size=5)
    assert torch.equal(advantages, frozen)
    assert torch.equal(normalized[0], critic.center) and torch.equal(normalized[1], critic.scale)
    _, later = ppo.round_advantages(data, critic)
    assert not later["zero_baseline"]


@pytest.mark.parametrize("full_history", [False, True])
def test_microbatches_are_count_weighted_without_optimizer_steps(monkeypatch, full_history):
    actor = ParameterActor()
    calls = []

    class Data:
        valid = torch.tensor([[True, True, True], [True, False, True]])

        def select(self, rows, device):
            v = self.valid[:, rows]
            return SimpleNamespace(valid=v, latents=None, old_means=None, old_log_prob=None,
                                   stationary_std=[.01]*4, rho=.8, warmup_steps=2,
                                   observation=lambda *a: None)

    def replay(controller, observe, latents, means, log_prob, advantage, valid, **kw):
        calls.append(kw)
        assert advantage.shape == valid.shape
        return diagnostics([0.] * int(valid.sum()))

    monkeypatch.setattr(ppo, "replay_full_policy_gradient" if full_history else
                        "replay_joint_policy_gradient", replay)
    ppo.replay_round(actor, Data(), torch.ones(2, 3), camera=None, gate_config=None, microbatch=2,
                     full_history=full_history)
    assert [c["gradient_scale"] for c in calls] == pytest.approx([3/5, 2/5])
    assert all(c["include_kl_samples"] for c in calls)
    assert actor.edge_magnitude.grad is None


def test_repeated_unchanged_native_evaluation_cannot_select_or_unsafe_policy_win():
    baseline = dict(clean_course_success_rate=.25, clean_course_negative_success_rate=.3,
                    clean_course_positive_success_rate=.2, gates_before_failure_mean=2.,
                    ground_contact_rate=0., invalid_rate=0.)
    improved = dict(baseline, clean_course_success_rate=.5)
    assert not ppo.select_new_native(improved, baseline, 0, 0)
    assert not ppo.select_new_native(improved, baseline, 1, 1)
    assert ppo.select_new_native(improved, baseline, 1, 0)
    assert not ppo.select_new_native(dict(improved, ground_contact_rate=.01), baseline, 1, 0)


@pytest.mark.parametrize("accepted,delta,selected", [(False, .001, False),
                                                   (True, 0., False), (True, .001, True)])
@pytest.mark.parametrize("actor_update", ["adam", "steepest"])
@pytest.mark.parametrize("oom", [False, True])
def test_pilot_selects_only_changed_accepted_policy(
    monkeypatch, tmp_path, accepted, delta, selected, actor_update, oom,
):
    import train_pragmatic_course_ppo as driver

    from flydrone.gate import GateConfig
    from flydrone.hover import HoverConfig

    actor = ParameterActor()
    args = SimpleNamespace(checkpoint=tmp_path / "source.pt", output_dir=tmp_path / "run",
                           graph=tmp_path / "graph.npz", device="cpu", seed=10, noise_seed=20,
                           development_seed=30, rounds=1, proposals=1, training_pairs=1,
                           development_pairs=16, microbatch=8, chunk_steps=2, learning_rate=1e-6,
                           actor_update=actor_update, full_history=actor_update == "steepest",
                           predicted_decrease=5e-5)
    source = dict(hover_config=vars(HoverConfig()), gate_config=vars(GateConfig()),
                  image_resolution=[32, 20], camera_hfov_degrees=125)
    base = dict(clean_course_success_rate=.25, clean_course_negative_success_rate=.3,
                clean_course_positive_success_rate=.2, gates_before_failure_mean=2.,
                ground_contact_rate=0., invalid_rate=0.)
    results = iter([base, dict(base, clean_course_success_rate=.5)])
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver.replay, "load_controller", lambda *a: (actor, source))
    monkeypatch.setattr(driver.replay, "roll_preservation_mask",
                        lambda *a, **kw: (torch.ones(3, dtype=torch.bool), {}))
    monkeypatch.setattr(driver.replay, "sample_two_gate_cases", lambda *a, **kw: (None, None))
    monkeypatch.setattr(driver.replay, "evaluate", lambda *a, **kw: next(results))
    data = SimpleNamespace(metrics={}, critic_features=torch.zeros(2, 2, 3),
                           returns=torch.ones(2, 2), valid=torch.ones(2, 2, dtype=torch.bool),
                           stationary_std=[.01]*4, rho=.8)
    monkeypatch.setattr(driver, "collect_policy_rollout", lambda *a, **kw: data)
    monkeypatch.setattr(driver, "fit_critic", lambda *a, **kw: {})
    replay_calls = []

    def round_replay(*a, **kw):
        replay_calls.append(kw)
        assert kw["full_history"] is args.full_history
        if oom and len(replay_calls) == 1:
            actor.edge_magnitude.grad = torch.ones_like(actor.edge_magnitude)
            raise torch.cuda.OutOfMemoryError("test partial microbatch failure")
        assert actor.edge_magnitude.grad is None
        return {}

    monkeypatch.setattr(driver, "replay_round", round_replay)
    monkeypatch.setattr(driver.torch.cuda, "empty_cache", lambda: None)

    def proposal(*a, **kw):
        a[2 if actor_update == "steepest" else 3](True)
        if accepted:
            with torch.no_grad():
                actor.edge_magnitude.add_(delta)
        return dict(accepted=accepted, stop_round=not accepted, proposed_edge_delta_l2=delta)

    monkeypatch.setattr(driver, "actor_proposal", proposal)
    monkeypatch.setattr(driver, "steepest_actor_proposal", proposal)
    assert driver.main() == 0
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "complete"
    assert report["selected_round"] == int(selected)
    assert report["meaningful_development_nominee"] is selected
    assert report["oom_fallbacks"] == int(oom)
    assert report["effective_microbatch"] == (4 if oom else 8)
    assert [c["microbatch"] for c in replay_calls] == ([8, 4] if oom else [8])
    assert (args.output_dir / "best-controller.pt").exists() is selected
    saved = torch.load(args.output_dir / "training-state.pt", weights_only=True)
    assert (saved["actor_optimizer"] is None) is (actor_update == "steepest")


@pytest.mark.parametrize("first_failure", [None, "loss", "trust", "both"])
def test_steepest_acceptance_half_retry_uses_same_base_gradient_and_native_mask(first_failure):
    actor = ParameterActor()
    base = actor.edge_magnitude.detach().clone()
    mask = torch.tensor([True, False, True])
    calls, deltas = [], []

    def replay(backward):
        calls.append(backward)
        loss = actor.edge_magnitude.square().sum()
        if backward:
            loss.backward()
        else:
            deltas.append(actor.edge_magnitude.detach().clone() - base)
            assert actor.edge_magnitude[1] == base[1]
        first_bad = not backward and len(deltas) == 1
        value = float(loss.detach())
        if first_bad and first_failure in ("loss", "both"):
            value = float(base.square().sum()) + .001
        return dict(loss=value, joint_kl_mean=(.02 if first_bad and first_failure in
                                              ("trust", "both") else .001),
                    joint_kl_p99=.002, native_latent_mean_displacement_rms=[.001]*4)

    stats = ppo.steepest_actor_proposal(actor, mask, replay, [.01]*4)
    assert stats["accepted"] and not stats["stop_round"]
    assert calls == ([True, False] if first_failure is None else [True, False, False])
    assert len(stats["attempts"]) == (1 if first_failure is None else 2)
    assert actor.edge_magnitude.grad is None
    for i, attempt in enumerate(stats["attempts"]):
        predicted = float((2*base*mask*deltas[i]).sum())
        assert attempt["predicted_loss_change"] == pytest.approx(predicted)
        assert attempt["target_predicted_decrease"] == pytest.approx(5e-5 * .5**i)
    if first_failure:
        assert deltas[1].norm() < deltas[0].norm()  # Not a cumulative second step.


@pytest.mark.parametrize("failure", ["loss", "trust", "nan_loss", "nan_gradient", "exception"])
def test_steepest_rejection_or_nonfinite_error_restores_parameters_and_clears_gradient(failure):
    actor = ParameterActor()
    base = actor.edge_magnitude.detach().clone()
    calls = []

    def replay(backward):
        calls.append(backward)
        if backward:
            actor.edge_magnitude.square().sum().backward()
            if failure == "nan_gradient":
                actor.edge_magnitude.grad[0] = float("nan")
        elif failure == "exception":
            raise FloatingPointError("deliberate failure")
        return dict(loss=float("nan") if failure == "nan_loss" and not backward else
                    (0.2 if backward or failure == "loss" else 0.1),
                    joint_kl_mean=.02 if failure == "trust" else .001,
                    joint_kl_p99=.002, native_latent_mean_displacement_rms=[.001]*4)

    if failure in ("loss", "trust"):
        stats = ppo.steepest_actor_proposal(actor, torch.ones(3, dtype=torch.bool), replay, [.01]*4)
        assert not stats["accepted"] and stats["stop_round"]
        assert calls == [True, False, False]
    else:
        with pytest.raises(FloatingPointError):
            ppo.steepest_actor_proposal(actor, torch.ones(3, dtype=torch.bool), replay, [.01]*4)
    assert torch.equal(actor.edge_magnitude, base) and actor.edge_magnitude.grad is None


def test_steepest_mode_requires_full_history_before_loading_any_controller(monkeypatch, tmp_path):
    import train_pragmatic_course_ppo as driver

    monkeypatch.setattr(sys, "argv", ["pilot", "--checkpoint", str(tmp_path / "source.pt"),
                                     "--output-dir", str(tmp_path / "out"),
                                     "--actor-update", "steepest"])
    with pytest.raises(SystemExit, match="require --full-history"):
        driver.main()
    assert not (tmp_path / "out").exists()
