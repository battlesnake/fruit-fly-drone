from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pragmatic_correlated_exploration import ar1_conditional, joint_log_prob  # noqa: E402
from pragmatic_full_policy_gradient import replay_full_policy_gradient  # noqa: E402
from pragmatic_recurrent_policy_gradient import replay_joint_policy_gradient  # noqa: E402


class MemoryActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.1, .2, .15, .05]))

    def initial_state(self, count, **kw):
        return torch.zeros(count, 4, **kw)

    def forward(self, image, attitude, neural):
        advanced = .85 * neural + image * self.weight
        return (.4 * advanced).tanh(), advanced


def fixture():
    actor = MemoryActor()
    images = torch.linspace(.05, 1, 17*2*4).reshape(17, 2, 4)
    observe = lambda t: (images[t], None)  # noqa: E731
    def means():
        neural = actor.initial_state(2)
        for _ in range(3):
            _, neural = actor(*observe(0), neural)
        values = []
        for t in range(17):
            motor, neural = actor(*observe(t), neural)
            values.append(torch.atanh(motor))
        return torch.stack(values)
    with torch.no_grad():
        old = means()
    u = old + torch.randn(old.shape, generator=torch.Generator().manual_seed(3))*.01
    first = torch.zeros(17, 2, dtype=torch.bool)
    first[0] = True
    prev_u = torch.cat((u[:1], u[:-1]))
    def density(mu):
        conditional, std = ar1_conditional(mu, torch.cat((mu[:1], mu[:-1])), prev_u,
                                           [.02]*4, .8, first)
        return joint_log_prob(u, conditional, std, squashed=False)
    old_logp = density(old)
    adv = torch.linspace(-1, 1, 34).reshape(17, 2)
    valid = torch.ones(17, 2, dtype=torch.bool)
    valid[-3:, 0] = False
    ratio = (density(means()) - old_logp).exp()
    loss = -torch.minimum(ratio * adv, ratio.clamp(.9, 1.1)*adv)[valid].mean()
    loss.backward()
    reference = actor.weight.grad.clone()
    actor.zero_grad(set_to_none=True)
    return actor, (observe, u, old, old_logp, adv, valid), reference


@pytest.mark.parametrize("chunk,checkpointed", [(1, True), (3, True), (17, True), (17, False)])
def test_untruncated_gradient_matches_independent_reference_including_warmup(chunk, checkpointed):
    actor, args, reference = fixture()
    result = replay_full_policy_gradient(actor, *args, stationary_std=[.02]*4, rho=.8,
                                         chunk_steps=chunk, warmup_steps=3,
                                         checkpointed=checkpointed, include_kl_samples=True)
    assert torch.allclose(actor.weight.grad, reference, rtol=2e-5, atol=2e-6)
    assert not result["recurrent_gradient_is_truncated"]
    assert result["warmup_is_differentiated"] and len(result["joint_kl_samples"]) == 31
    assert result["joint_ratio_mean"] == pytest.approx(1)


def test_warmup_gradient_is_not_silently_detached_and_values_stay_same():
    actor, args, reference = fixture()
    short = replay_joint_policy_gradient(actor, *args, stationary_std=[.02]*4, rho=.8,
                                         chunk_steps=17, warmup_steps=3)
    assert not torch.allclose(actor.weight.grad, reference, rtol=1e-3, atol=1e-4)
    actor.zero_grad(set_to_none=True)
    full = replay_full_policy_gradient(actor, *args, stationary_std=[.02]*4, rho=.8,
                                        chunk_steps=3, warmup_steps=3, gradient_scale=.25)
    assert full["loss"] == pytest.approx(short["loss"], abs=1e-7)
    assert torch.allclose(actor.weight.grad, .25*reference, rtol=2e-5, atol=2e-6)


def test_inactive_nonfinite_padding_or_observation_rejected():
    actor, args, _ = fixture()
    args[1][-1, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="padding.*finite"):
        replay_full_policy_gradient(actor, *args, stationary_std=[.02]*4, rho=.8)
    actor, args, _ = fixture()
    original = args[0]
    def observe(t):
        image, attitude = original(t)
        if t == 16:
            image = image.clone()
            image[0, 0] = float("nan")
        return image, attitude
    with pytest.raises(FloatingPointError, match="nonfinite"):
        replay_full_policy_gradient(actor, observe, *args[1:], stationary_std=[.02]*4,
                                    rho=.8, warmup_steps=3, chunk_steps=3)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("steepest", [False, True])
def test_full_gradient_probe_uses_archived_data_and_restores_source(
    monkeypatch, tmp_path, fail, steepest,
):
    import probe_pragmatic_full_gradient as driver

    from flydrone.gate import GateConfig

    actor = torch.nn.Module()
    actor.edge_magnitude = torch.nn.Parameter(torch.tensor([2., .2, .3]))
    initial = actor.edge_magnitude.detach().clone()
    args = SimpleNamespace(archive=tmp_path / "archive.pt", checkpoint=tmp_path / "source.pt",
                           output_dir=tmp_path / "run", graph=tmp_path / "graph.npz",
                           device="cpu", microbatch=2, chunk_steps=2,
                           reuse_gradient=None, steepest=steepest)
    if steepest:
        args.reuse_gradient = tmp_path / "previous/full-gradient-data.pt"
        args.reuse_gradient.parent.mkdir()
        torch.save(dict(gradient=torch.tensor([4., 0., .6]),
                        mask=torch.tensor([True, False, True])), args.reuse_gradient)
        (args.reuse_gradient.parent / "report.json").write_text(json.dumps(
            dict(archive=str(args.archive), checkpoint=str(args.checkpoint))))
    fields = dict(states=(), gates=[], current=torch.zeros(2, 2, dtype=torch.long),
                  latents=torch.zeros(2, 2, 4), old_means=torch.zeros(2, 2, 4),
                  old_log_prob=torch.zeros(2, 2), rewards=torch.zeros(2, 2),
                  returns=torch.ones(2, 2), valid=torch.ones(2, 2, dtype=torch.bool),
                  critic_features=torch.zeros(2, 2, 76), stationary_std=(.02,)*4, rho=.8,
                  warmup_steps=3, metrics={})
    torch.save(dict(schema="plain-native-policy-direction-v1",
                    source_checkpoint=str(args.checkpoint),
                    graph_sha256="test", mask=torch.tensor([True, False, True]),
                    gradient=torch.tensor([4., 0., .6]),
                    displacement=torch.tensor([-1e-6, 0., -1e-6]),
                    rollout=fields, advantages=torch.ones(2, 2)), args.archive)
    source = dict(graph_sha256="test", gate_config=vars(GateConfig()),
                  image_resolution=[32, 20], camera_hfov_degrees=125)
    monkeypatch.setattr(driver.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(driver.replay, "load_controller", lambda *a: (actor, source))

    def replay(*a, backward, full_history, **kw):
        assert full_history
        loss = actor.edge_magnitude.square().sum()
        if backward:
            loss.backward()
        elif fail and not torch.equal(actor.edge_magnitude, initial):
            raise FloatingPointError("deliberate probe replay failure")
        return dict(loss=float(loss.detach()), joint_kl_mean=0., joint_kl_p99=0.,
                    native_latent_mean_displacement_rms=[0.]*4)

    monkeypatch.setattr(driver, "replay_round", replay)
    if fail:
        with pytest.raises(FloatingPointError, match="deliberate"):
            driver.main()
    else:
        assert driver.main() == 0
    assert torch.equal(actor.edge_magnitude, initial) and actor.edge_magnitude.grad is None
    saved = torch.load(args.reuse_gradient if steepest else
                       args.output_dir / "full-gradient-data.pt", weights_only=True)
    assert saved["gradient"].tolist() == pytest.approx([4., 0., .6])
    if not steepest:
        assert saved["displacement"][1] == 0


@pytest.mark.parametrize("target", [1e-4, 5e-5])
def test_projected_steepest_sizing_uses_actual_contraction_and_mask(target):
    from probe_pragmatic_full_gradient import steepest_parameters

    base = torch.tensor([0., 2., 8., 3.])
    gradient = torch.tensor([100., 4., -100., 1e5])
    mask = torch.tensor([True, True, True, False])
    original = base.clone()
    result, stats = steepest_parameters(base, gradient, mask, target)
    assert torch.equal(base, original)  # Sizing is read-only with respect to the actor.
    assert result[0] == 0 and result[2] == 8 and result[3] == 3
    actual = float((gradient * (result-base)).sum())
    assert stats["predicted_loss_change"] == pytest.approx(actual)
    assert .9*target <= -actual <= 1.1*target
