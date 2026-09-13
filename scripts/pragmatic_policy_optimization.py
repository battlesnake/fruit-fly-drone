"""Training-only critic and transactional, full-round recurrent PPO updates."""

from __future__ import annotations

import copy
import math

import torch
from pragmatic_recurrent_policy_gradient import replay_joint_policy_gradient


class OutcomeCritic(torch.nn.Module):
    """Separate privileged baseline; never exported as part of the actor."""

    def __init__(self, features):
        super().__init__()
        self.register_buffer("center", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.network = torch.nn.Sequential(
            torch.nn.Linear(features, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1),
        )

    @torch.no_grad()
    def initialize_normalization(self, values):
        self.center.copy_(values.mean(0))
        self.scale.copy_(values.std(0, unbiased=False).clamp_min(0.01))

    def forward(self, features):
        return self.network((features - self.center) / self.scale).squeeze(-1)


@torch.no_grad()
def round_advantages(data, critic, *, zero_baseline=False):
    """Freeze advantages once across ALL valid samples, before critic fitting."""
    device = next(critic.parameters()).device
    baseline = (torch.zeros_like(data.returns) if zero_baseline else
                critic(data.critic_features.to(device)).cpu())
    raw = data.returns - baseline
    values = raw[data.valid]
    if not bool(values.isfinite().all()):
        raise FloatingPointError("nonfinite critic baseline or returns")
    mean, scale = values.mean(), values.std(unbiased=False).clamp_min(1e-6)
    result = torch.where(data.valid, (raw - mean) / scale, 0).detach()
    return result, dict(zero_baseline=zero_baseline, raw_mean=float(mean), raw_std=float(scale),
                        baseline_mse=float((baseline - data.returns)[data.valid].square().mean()))


def fit_critic(critic, optimizer, data, *, epochs=5, batch_size=1024, seed=1):
    device = next(critic.parameters()).device
    x, y = data.critic_features[data.valid].to(device), data.returns[data.valid].to(device)
    rng = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(y), device=device, generator=rng)
        for rows in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = (critic(x[rows]) - y[rows]).square().mean()
            if not bool(loss.isfinite()):
                raise FloatingPointError("nonfinite critic loss")
            loss.backward()
            if any(p.grad is not None and not bool(p.grad.isfinite().all())
                   for p in critic.parameters()):
                raise FloatingPointError("nonfinite critic gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    with torch.no_grad():
        mse = float((critic(x) - y).square().mean())
    if not math.isfinite(mse):
        raise FloatingPointError("nonfinite fitted critic")
    return dict(epochs=epochs, minibatches=len(losses), final_mse=mse)


def aggregate_replays(parts):
    """Count-weight means/RMS and compute p99 from ALL individual valid KLs."""
    count = sum(part["valid_commands"] for part in parts)
    if not count:
        raise ValueError("no valid replay commands")
    kl = torch.tensor([value for part in parts for value in part["joint_kl_samples"]])
    if len(kl) != count or not bool(kl.isfinite().all()):
        raise FloatingPointError("invalid per-command KL diagnostics")
    result = dict(valid_commands=count, joint_kl_mean=float(kl.mean()),
                  joint_kl_p99=float(torch.quantile(kl, 0.99)), joint_kl_max=float(kl.max()))
    for name in ("loss", "joint_ratio_mean", "clipped_command_fraction"):
        result[name] = sum(p[name] * p["valid_commands"] for p in parts) / count
    for name in ("native_latent_mean_displacement_rms", "native_clamped_fraction"):
        values = torch.tensor([p[name] for p in parts])
        weights = torch.tensor([p["valid_commands"] / count for p in parts])[:, None]
        result[name] = (((values.square() * weights).sum(0).sqrt() if name.endswith("rms")
                         else (values * weights).sum(0)).tolist())
    return result


def replay_round(controller, data, advantages, *, camera, gate_config, microbatch=4,
                 chunk_steps=20, backward=True, progress=None):
    device = next(controller.parameters()).device
    count = int(data.valid.sum())
    parts = []
    for start in range(0, data.valid.shape[1], microbatch):
        rows = range(start, min(start + microbatch, data.valid.shape[1]))
        batch = data.select(rows, device)
        n = int(batch.valid.sum())
        if not n:
            continue
        stats = replay_joint_policy_gradient(
            controller, lambda t, b=batch: b.observation(t, camera, gate_config), batch.latents,
            batch.old_means, batch.old_log_prob, advantages[:, rows].to(device), batch.valid,
            stationary_std=batch.stationary_std, rho=batch.rho, chunk_steps=chunk_steps,
            warmup_steps=batch.warmup_steps, backward=backward, gradient_scale=n / count,
            include_kl_samples=True,
        )
        parts.append(stats)
        if progress:
            progress(dict(stage="backward-microbatch" if backward else "trust-microbatch",
                          first_episode=start, episodes=len(rows),
                          joint_kl_mean=stats["joint_kl_mean"]))
    return aggregate_replays(parts)


def trust_decision(stats, stationary_std):
    shift = [value / sigma for value, sigma in
             zip(stats["native_latent_mean_displacement_rms"], stationary_std, strict=True)]
    checks = [stats["joint_kl_mean"], stats["joint_kl_p99"], *shift]
    if not all(math.isfinite(value) for value in checks):
        raise FloatingPointError("nonfinite policy trust diagnostics")
    reject = stats["joint_kl_mean"] > 0.01 or stats["joint_kl_p99"] > 0.10 or max(shift) > 0.5
    return dict(accepted=not reject, stop_round=reject or stats["joint_kl_mean"] > 0.005,
                native_mean_shift_in_stationary_std=shift)


def actor_proposal(controller, optimizer, mask, replay_fn, stationary_std):
    """One proposal; replay_fn must rebuild all current-weight observation prefixes."""
    parameter = controller.edge_magnitude
    before = parameter.detach().clone()
    adam_before = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    try:
        before_stats = replay_fn(True)
        if parameter.grad is None:
            raise ValueError("actor replay produced no gradient")
        parameter.grad.mul_(mask)
        norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
        optimizer.step()
        # Only selected existing magnitudes may change; signs/topology stay fixed.
        with torch.no_grad():
            parameter[mask] = parameter[mask].clamp(0, 8)
            parameter[~mask] = before[~mask]
        if not bool(parameter.isfinite().all()):
            raise FloatingPointError("nonfinite actor parameter")
        after_stats = replay_fn(False)
        decision = trust_decision(after_stats, stationary_std)
        result = dict(before=before_stats, after=after_stats, gradient_norm=float(norm),
                      proposed_edge_delta_l2=float((parameter.detach() - before).norm()),
                      **decision)
        if not decision["accepted"]:
            with torch.no_grad():
                parameter.copy_(before)
            optimizer.load_state_dict(adam_before)
        return result
    except Exception:
        with torch.no_grad():
            parameter.copy_(before)
        optimizer.load_state_dict(adam_before)
        raise
    finally:
        optimizer.zero_grad(set_to_none=True)


def native_score(metrics):
    return (metrics["clean_course_success_rate"],
            min(metrics["clean_course_negative_success_rate"],
                metrics["clean_course_positive_success_rate"]),
            metrics["gates_before_failure_mean"])


def eligible_native(metrics):
    return metrics["ground_contact_rate"] == 0 and metrics["invalid_rate"] == 0


def select_new_native(metrics, best, revision, last_evaluated_revision):
    """A repeated evaluation of unchanged weights cannot manufacture a winner."""
    return (revision > last_evaluated_revision and eligible_native(metrics)
            and native_score(metrics) > native_score(best))
