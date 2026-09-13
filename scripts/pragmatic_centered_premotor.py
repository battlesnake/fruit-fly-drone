"""Training-only mean-compensated plasticity, compiled to native weights/biases."""

from __future__ import annotations

import torch
from torch.func import functional_call

PHASES = ("launch", "first-approach", "later-gates")


class PresynapticMoments:
    """Running pre-tick source statistics; never archive full-brain histories."""

    def __init__(self, nodes):
        self.nodes = nodes.detach().clone()
        self.sums = torch.zeros(6, len(nodes), device=nodes.device)
        self.counts = torch.zeros(6, device=nodes.device)

    @torch.no_grad()
    def observe(self, source_neural, current, active_visible, frame):
        rows = torch.arange(len(current), device=current.device)
        eligible = active_visible & (rows < len(current) - 2)  # Last whole pair held out.
        phase = torch.where(current > 0, 2, torch.full_like(current, int(frame >= 50)))
        group = 2 * phase + rows % 2
        weights = torch.nn.functional.one_hot(group, 6).float() * eligible[:, None]
        self.sums += weights.T @ source_neural[:, self.nodes].tanh()
        self.counts += weights.sum(0)

    def archive(self):
        return dict(nodes=self.nodes.cpu(), sums=self.sums.cpu(), counts=self.counts.cpu())


def balanced_input_mean(banks, nodes):
    means, coverage = [], {}
    for bank in banks:
        moments = bank.input_moments
        if moments is None or not torch.equal(moments["nodes"], nodes.cpu()):
            raise ValueError("missing or mismatched pre-tick source input statistics")
        if not bool(moments["sums"].isfinite().all() & moments["counts"].isfinite().all()):
            raise FloatingPointError("nonfinite source input statistics")
        for group, count in enumerate(moments["counts"].tolist()):
            coverage[f"{bank.replay.kind}/{PHASES[group // 2]}/{group % 2}"] = int(count)
            if count:
                means.append(moments["sums"][group] / count)
    if not means:
        raise ValueError("no eligible training frames for centering")
    mean = torch.stack(means).mean(0).to(nodes.device)
    return mean, dict(
        coverage=coverage,
        presynaptic_neurons=len(nodes),
        contributing_strata=len(means),
        source="original source pre-tick tanh",
        heldout_episodes_excluded=True,
        weighting="equal nonempty kind/side/phase",
    )


class MeanCompensatedPremotor(torch.nn.Module):
    """Functional training wrapper; export ONLY the separately held native controller.

    Bias is differentiably tied to selected weights, not independently optimized.
    This preserves reference mean drive on fixed histories, not closed-loop trim.
    """

    def __init__(self, controller, mask, presynaptic_nodes, mean):
        super().__init__()
        self.controller = controller
        edges = torch.nonzero(mask).flatten()
        expected = torch.unique(controller.edge_pre[edges])
        if not torch.equal(expected, presynaptic_nodes) or mean.shape != expected.shape:
            raise ValueError("centering mean must match the ordered external presynaptic neurons")
        if not bool(mean.isfinite().all()):
            raise FloatingPointError("nonfinite centering mean")
        if controller.bias.requires_grad:
            raise ValueError("native biases must not be independently trainable")
        lookup = torch.searchsorted(expected, controller.edge_pre[edges])
        self.register_buffer("edges", edges)
        self.register_buffer("posts", controller.edge_post[edges].clone())
        self.register_buffer("coefficients", controller.edge_sign[edges] * mean[lookup].detach())
        self.register_buffer("reference_weights", controller.edge_magnitude[edges].detach().clone())
        self.register_buffer("reference_bias", controller.bias.detach().clone())

    @property
    def bias(self):
        return self.controller.bias

    def initial_state(self, *args, **kwargs):
        return self.controller.initial_state(*args, **kwargs)

    def tied_bias(self):
        delta = self.controller.edge_magnitude[self.edges] - self.reference_weights
        return self.reference_bias.index_add(0, self.posts, -self.coefficients * delta)

    def forward(self, image, attitude, neural):
        return functional_call(
            self.controller, {"bias": self.tied_bias()}, (image, attitude, neural)
        )

    @torch.no_grad()
    def compile_bias(self):
        bias = self.tied_bias()
        if not bool(bias.isfinite().all()):
            raise FloatingPointError("nonfinite compensated native bias")
        # No clipping: that would silently change the affine constraint.
        self.controller.bias.copy_(bias)
        delta = bias - self.reference_bias
        return dict(
            changed_biases=int((delta != 0).sum()),
            bias_displacement_l2=float(delta.norm()),
            bias_displacement_max=float(delta.abs().max()),
        )

    def training_state(self):
        return {
            name: value.detach().cpu()
            for name, value in self.named_buffers()
            if not name.startswith("controller.")
        }
