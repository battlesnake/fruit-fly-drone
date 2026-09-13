from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pragmatic_correlated_exploration import ar1_conditional, joint_log_prob  # noqa: E402
from pragmatic_sink_policy import (  # noqa: E402
    NativeRollSinkPolicy,
    replay_sink_policy_gradient,
    verify_compiled_sink_policy,
    verify_sink_recording,
)


class SmallBrain(torch.nn.Module):
    """Independent full-network reference: two sensory parents, eight motor sinks."""
    def __init__(self):
        super().__init__()
        self.edge_magnitude = torch.nn.Parameter(torch.tensor([.2, .1] * 8))
        self.bias = torch.zeros(10)
        self.pool_indices = torch.arange(2, 10)
        self.pool_offsets = torch.arange(9)
        self.edge_pre = torch.tensor([0, 1] * 8)
        self.edge_post = torch.arange(2, 10).repeat_interleave(2)
        self.edge_sign = torch.tensor([1., -1.] * 8)
        self.time_constant = torch.linspace(.02, .03, 10)
        self.neural_dt = .02

    def initial_state(self, count, **kw):
        return torch.zeros(count, 10, **kw)

    def forward(self, observation, attitude, state):
        activity = state.tanh()
        messages = activity[:, self.edge_pre] * self.edge_sign * self.edge_magnitude
        drive = torch.zeros_like(state).index_add(1, self.edge_post, messages)
        drive = drive + torch.cat((observation, observation.new_zeros(len(observation), 8)), 1)
        alpha = 1 - torch.exp(-self.neural_dt / self.time_constant)
        state = state + alpha * (5*(drive/5).tanh()-state)
        motor = state[:, 2:].sigmoid().reshape(-1, 4, 2)
        return motor[..., 0]-motor[..., 1], state

    def rollout(self, observations):
        state = observations.new_zeros(observations.shape[1], 10)
        features, means = [], []
        for observation in observations:
            features.append(state[:, :2].tanh())
            motor, state = self(observation, None, state)
            means.append(torch.atanh(motor.clamp(-.999999, .999999)))
        return torch.stack(features), torch.stack(means)


def fixture():
    torch.manual_seed(12)
    brain = SmallBrain()
    model = NativeRollSinkPolicy(brain)
    observations = torch.randn(13, 3, 2)
    observations[:3] = observations[3]  # The collector's static warmup observation.
    with torch.no_grad():
        features, all_means = brain.rollout(observations)
    old_means = all_means[3:]
    latents = old_means + .01*torch.randn_like(old_means)
    first = torch.zeros(10, 3, dtype=torch.bool)
    first[0] = True
    conditional, std = ar1_conditional(
        old_means, torch.cat((old_means[:1], old_means[:-1])),
        torch.cat((latents[:1], latents[:-1])), [.02]*4, .8, first,
    )
    valid = torch.ones(10, 3, dtype=torch.bool)
    valid[-2:, 0] = False
    data = SimpleNamespace(sink_features=features, sink_parent_nodes=model.sink.parents.clone(),
                           old_means=old_means, latents=latents,
                           old_log_prob=joint_log_prob(latents, conditional, std, squashed=False),
                           warmup_steps=3, stationary_std=[.02]*4, rho=.8, valid=valid)
    return brain, model, data, observations, torch.randn(10, 3), std


def test_complete_sink_replay_matches_independent_full_network_values_and_policy_gradients():
    brain, model, data, observations, advantages, std = fixture()
    verify_sink_recording(model, data)
    with torch.no_grad():
        model.edge_magnitude.add_(torch.tensor([.004, -.001, .002, .003]))
    base = brain.edge_magnitude.detach().clone()
    model.compile_into(brain)
    assert torch.equal(brain.edge_magnitude[4:], base[4:])
    _, full_means = brain.rollout(observations)
    full_means = full_means[3:]
    sliced, _ = model.means(data)
    assert torch.allclose(sliced, full_means, atol=1e-7)
    assert torch.equal(sliced[..., 1:], data.old_means[..., 1:])
    stats = replay_sink_policy_gradient(model, data, advantages)
    conditional = torch.cat((full_means[:1], full_means[1:] + data.rho *
                             (data.latents[:-1] - full_means[:-1])))
    ratio = (joint_log_prob(data.latents, conditional, std, squashed=False)-data.old_log_prob).exp()
    loss = -torch.minimum(ratio*advantages, ratio.clamp(.9, 1.1)*advantages)[data.valid].mean()
    loss.backward()
    # Dense vs sparse FP32 sums differ slightly; the Gaussian density magnifies it.
    assert stats["loss"] == pytest.approx(float(loss.detach()), abs=1e-6)
    assert torch.allclose(model.edge_magnitude.grad, brain.edge_magnitude.grad[:4],
                          atol=2e-6, rtol=2e-5)
    assert stats["native_latent_mean_displacement_rms"][1:] == [0., 0., 0.]
    assert not stats["recurrent_gradient_is_truncated"] and stats["warmup_is_differentiated"]


@pytest.mark.parametrize("fault", ["missing", "parents", "warmup", "nan", "grad"])
def test_sink_replay_rejects_bad_cache(fault):
    _, model, data, _, advantages, _ = fixture()
    if fault == "missing":
        data.sink_features = None
    elif fault == "parents":
        data.sink_parent_nodes = data.sink_parent_nodes.flip(0)
    elif fault == "warmup":
        data.sink_features = data.sink_features[3:]
    elif fault == "nan":
        data.sink_features[-1, 0, 0] = float("nan")  # Padding must also remain finite.
    else:
        data.sink_features.requires_grad_(True)
    with pytest.raises((ValueError, FloatingPointError)):
        replay_sink_policy_gradient(model, data, advantages)


def test_sink_structure_rejects_overlap_and_source_recording_mismatch():
    brain, model, data, _, _, _ = fixture()
    brain.pool_indices[2] = brain.pool_indices[0]
    with pytest.raises(ValueError, match="disjoint"):
        NativeRollSinkPolicy(brain)
    data.old_means[0, 0, 0] += .01
    with pytest.raises(ValueError, match="reconstruct"):
        verify_sink_recording(model, data)


def test_compiled_check_replays_full_history_and_catches_nonroll_changes():
    brain, model, data, observations, _, _ = fixture()

    def select(rows, device):
        rows = list(rows)
        fields = {k: v[:, rows] if isinstance(v, torch.Tensor) and v.ndim >= 2 else v
                  for k, v in vars(data).items()}
        batch = SimpleNamespace(**fields)
        batch.observation = lambda time, *a: (observations[time+3, rows], None)
        return batch

    data.select = select
    with torch.no_grad():
        model.edge_magnitude.add_(.01)
    model.compile_into(brain)
    stats = verify_compiled_sink_policy(brain, model, data, camera=None, gate_config=None)
    assert stats["training_episodes"] == 2 and stats["new_flights_collected"] == 0
    assert max(stats["latent_mean_max_error"]) < 1e-6
    with torch.no_grad():
        brain.edge_magnitude[4] += 1.
    with pytest.raises(ValueError, match="differs materially"):
        verify_compiled_sink_policy(brain, model, data, camera=None, gate_config=None)


def test_sink_mode_refuses_adam_before_loading_controller(monkeypatch, tmp_path):
    import train_pragmatic_course_ppo as driver
    monkeypatch.setattr(sys, "argv", ["pilot", "--checkpoint", str(tmp_path / "source.pt"),
                                     "--output-dir", str(tmp_path / "run"),
                                     "--actor-scope", "roll-sinks"])
    with pytest.raises(SystemExit, match="roll-sinks requires"):
        driver.main()


@pytest.mark.parametrize("development_every", [1, 2, 4])
@pytest.mark.parametrize("actor_update", ["steepest", "natural"])
def test_sink_pilot_compiles_before_fresh_collection_and_assessment(
    monkeypatch, tmp_path, development_every, actor_update,
):
    import train_pragmatic_course_ppo as driver

    from flydrone.gate import GateConfig
    from flydrone.hover import HoverConfig

    brain, model, original, observations, advantages, _ = fixture()
    source_edges = brain.edge_magnitude.detach().clone()
    args = SimpleNamespace(checkpoint=tmp_path / "source.pt", output_dir=tmp_path / "run",
                           graph=tmp_path / "graph.npz", device="cpu", seed=10, noise_seed=20,
                           development_seed=30, rounds=2, proposals=2, training_pairs=2,
                           development_pairs=16, development_every=development_every,
                           microbatch=4, chunk_steps=20, learning_rate=1e-6,
                           actor_update=actor_update, full_history=True, predicted_decrease=5e-5,
                           actor_scope="roll-sinks")
    source = dict(hover_config=vars(HoverConfig()), gate_config=vars(GateConfig()),
                  image_resolution=[32, 20], camera_hfov_degrees=125)
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver.replay, "load_controller", lambda *a: (brain, source))
    monkeypatch.setattr(driver, "NativeRollSinkPolicy", lambda *a: model)
    monkeypatch.setattr(driver.replay, "sample_two_gate_cases", lambda *a, **kw: (None, None))
    collections, assessments, checks = [], [], []

    def check_compiled():
        assert torch.equal(brain.edge_magnitude[:4], model.edge_magnitude)
        assert torch.equal(brain.edge_magnitude[4:], source_edges[4:])
        assert not brain.edge_magnitude.requires_grad

    def collect(*a, **kw):
        check_compiled()
        if collections:
            saved = torch.load(args.output_dir / "last-controller.pt", weights_only=True)
            assert torch.equal(saved["controller"]["edge_magnitude"], brain.edge_magnitude)
            if development_every > 1:
                assert saved["selection_metrics"] is None
                assert saved["selection_metrics_round"] is None
        assert torch.equal(kw["record_sink_parents"], model.sink.parents)
        with torch.no_grad():
            features, means = brain.rollout(observations)
        means = means[3:]
        latents = original.latents + means - original.old_means
        first = torch.zeros_like(original.valid)
        first[0] = True
        conditional, std = ar1_conditional(
            means, torch.cat((means[:1], means[:-1])), torch.cat((latents[:1], latents[:-1])),
            original.stationary_std, original.rho, first,
        )
        data = SimpleNamespace(**{**vars(original), "sink_features": features, "old_means": means,
                                 "latents": latents, "old_log_prob": joint_log_prob(
                                     latents, conditional, std, squashed=False)},
                               metrics={}, critic_features=torch.zeros(10, 3, 3),
                               returns=torch.ones(10, 3))
        data.select = lambda *a: data
        collections.append(data)
        return data

    def evaluate(*a, **kw):
        check_compiled()
        assessments.append(1)
        rate = len(assessments) * .125
        return dict(clean_course_success_rate=rate, clean_course_negative_success_rate=rate,
                    clean_course_positive_success_rate=rate, gates_before_failure_mean=rate,
                    ground_contact_rate=0., invalid_rate=0.)

    def proposal(actor, mask, replay_fn, *a, **kw):
        assert actor is model and bool(mask.all()) and len(mask) == 4
        replay_fn(True)
        assert bool(model.edge_magnitude.grad.isfinite().all())
        model.edge_magnitude.grad = None
        with torch.no_grad():
            model.edge_magnitude.add_(.0001)
        return dict(accepted=True, stop_round=False, proposed_edge_delta_l2=.0002)

    def verify(*a, **kw):
        check_compiled()
        checks.append(1)
        return {}

    monkeypatch.setattr(driver, "collect_policy_rollout", collect)
    monkeypatch.setattr(driver.replay, "evaluate", evaluate)
    monkeypatch.setattr(driver, "round_advantages", lambda *a, **kw: (advantages, {}))
    monkeypatch.setattr(driver, "fit_critic", lambda *a, **kw: {})
    monkeypatch.setattr(driver, "steepest_actor_proposal", proposal)
    monkeypatch.setattr(driver, "natural_sink_actor_proposal",
                        lambda actor, data, mask, replay_fn, **kw:
                        proposal(actor, mask, replay_fn, **kw))
    monkeypatch.setattr(driver, "verify_compiled_sink_policy", verify)
    assert driver.main() == 0
    assert len(collections) == 2 and collections[0] is not collections[1]
    assert len(assessments) == (3 if development_every == 1 else 2) and len(checks) == 1
    result = json.loads((args.output_dir / "report.json").read_text())
    assert result["accepted_steps"] == 4 and result["actor_scope"] == "roll-sinks"
    saved = torch.load(args.output_dir / "last-controller.pt", weights_only=True)
    assert torch.equal(saved["controller"]["edge_magnitude"], brain.edge_magnitude)
    assert all("sink" not in key for key in saved["controller"])
