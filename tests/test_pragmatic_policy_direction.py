from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_pragmatic_policy_direction import scaled_parameters, summarize_direction  # noqa: E402


def test_scales_start_at_base_and_respect_mask_rounding_and_projection():
    base = torch.tensor([0., 2., 7.99999, 3.])
    delta = torch.tensor([-.1, .02, .1, 10.])
    mask = torch.tensor([True, True, True, False])
    full, projections = scaled_parameters(base, delta, 1, mask)
    assert projections == 2 and full[0] == 0 and full[2] == 8 and full[3] == 3
    half, _ = scaled_parameters(base, delta, .5, mask)
    assert half[1].item() == pytest.approx(2.01)
    zero, _ = scaled_parameters(base, delta, 0, mask)
    assert torch.equal(zero, base)


def entries(losses):
    return [dict(scale=s, predicted_loss_change=-s*.01, diagnostics=dict(loss=loss))
            for s, loss in losses]


def test_direction_summary_distinguishes_descent_increase_and_noise():
    data = entries([(0, 0), (0, 1e-7), (1, .01), (.125, -.001), (0, -1e-7)])
    result = summarize_direction(data)
    assert result["observed_descent_scales"] == [.125]
    assert result["observed_increase_scales"] == [1]
    assert result["interpretation_tolerance"] == pytest.approx(1e-6)
    assert result["interpretation"] == "descent-resolved-at-some-tested-scales"
    increased = summarize_direction(entries([(0, 0), (1, .01), (0, 0)]))
    assert increased["interpretation"].startswith("predicted-descent-but-observed-increase")
    noisy = summarize_direction(entries([(0, -.001), (1, .001), (0, .001)]))
    assert noisy["interpretation"] == "inconclusive-at-replay-variability-level"


def test_non_descent_adam_direction_not_mislabelled_as_gradient_truncation():
    data = entries([(0, 0), (1, .01), (0, 0)])
    data[1]["predicted_loss_change"] = .001
    assert summarize_direction(data)["interpretation"].startswith("non-descent-materialized")


@pytest.mark.parametrize("fail", [False, True])
def test_driver_restores_source_and_saves_weights_only_readable_data(monkeypatch, tmp_path, fail):
    import audit_pragmatic_policy_direction as driver

    from flydrone.gate import GateConfig
    from flydrone.hover import HoverConfig

    args = SimpleNamespace(checkpoint=tmp_path / "source.pt", output_dir=tmp_path / "run",
                           graph=tmp_path / "graph.npz", device="cpu", seed=10, noise_seed=20,
                           pairs=1, microbatch=2, chunk_steps=2, learning_rate=1e-6)
    actor = torch.nn.Module()
    actor.edge_magnitude = torch.nn.Parameter(torch.tensor([2., .2, .3]))
    initial = actor.edge_magnitude.detach().clone()
    source = dict(hover_config=vars(HoverConfig()), gate_config=vars(GateConfig()),
                  image_resolution=[32, 20], camera_hfov_degrees=125, graph_sha256="test")
    monkeypatch.setattr(driver, "parse_args", lambda: args)
    monkeypatch.setattr(driver.replay, "load_controller", lambda *a: (actor, source))
    monkeypatch.setattr(driver.replay, "roll_preservation_mask",
                        lambda *a, **kw: (torch.tensor([True, False, True]), {}))
    monkeypatch.setattr(driver.replay, "sample_two_gate_cases", lambda *a, **kw: (None, None))
    data = SimpleNamespace(metrics={}, returns=torch.arange(4).reshape(2, 2).float(),
                           valid=torch.ones(2, 2, dtype=torch.bool), gates=())
    monkeypatch.setattr(driver, "collect_policy_rollout", lambda *a, **kw: data)

    def replay(*a, backward, **kw):
        loss = actor.edge_magnitude.square().sum()
        if backward:
            loss.backward()
        elif fail and not torch.equal(actor.edge_magnitude, initial):
            raise FloatingPointError("deliberate changed-policy replay failure")
        return dict(loss=float(loss.detach()))

    monkeypatch.setattr(driver, "replay_round", replay)
    if fail:
        with pytest.raises(FloatingPointError, match="deliberate"):
            driver.main()
    else:
        assert driver.main() == 0
    assert torch.equal(actor.edge_magnitude, initial)
    assert actor.edge_magnitude.grad is None
    saved = torch.load(args.output_dir / "fixed-training-data.pt", weights_only=True)
    assert saved["gradient"].tolist() == pytest.approx([4., 0., .6])
    assert saved["displacement"][1] == 0
    assert not saved["gradient"].requires_grad
