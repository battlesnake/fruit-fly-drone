"""Full-history recurrent policy gradient, with training-only activation replay.

There are no detached neural boundaries or replayed old neural states. Recompute
pure chunks to reduce activation memory; differentiate warmup as well as flight.
Observations/actions are fixed training data, not differentiable flight physics.
"""

from __future__ import annotations

import torch
from pragmatic_correlated_exploration import conditional_kl, joint_log_prob
from pragmatic_recurrent_policy_gradient import prepare_policy_replay
from torch.utils.checkpoint import checkpoint


def replay_full_policy_gradient(
    controller, observe, latents, old_means, old_log_prob, advantages, valid,
    *, stationary_std, rho, chunk_steps=20, warmup_steps=10, clip=.1, backward=True,
    gradient_scale=1., include_kl_samples=False, checkpointed=True,
):
    """Accumulate one full-episode gradient; never changes weights or clears grads.

    The previous mean crosses boundaries WITH gradient, just like neural state.
    Chunks capture immutable time bounds and reconstruct RGB from physical data;
    they neither capture all images nor append into external diagnostic lists.
    checkpointed=False exists for small reference tests, not large GPU episodes.
    """
    count, previous_latents, old_conditional, std = prepare_policy_replay(
        latents, old_means, old_log_prob, advantages, valid, stationary_std=stationary_std,
        rho=rho, chunk_steps=chunk_steps, warmup_steps=warmup_steps, clip=clip,
        gradient_scale=gradient_scale,
    )
    times, episodes, _ = latents.shape
    neural = controller.initial_state(episodes, device=latents.device, dtype=latents.dtype)

    def warmup(neural):
        observations = observe(0)
        for value in observations:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                if not bool(value.isfinite().all()):
                    raise FloatingPointError("nonfinite warmup observation")
        for _ in range(warmup_steps):
            _, neural = controller(*observations, neural)
        if not bool(neural.isfinite().all()):
            raise FloatingPointError("nonfinite warmup neural state")
        return neural

    def run(function, *inputs):
        if checkpointed and backward:
            return checkpoint(function, *inputs, use_reentrant=False, preserve_rng_state=False)
        return function(*inputs)

    def segment(first, end):
        def forward(neural, previous_mean):
            losses, means, kls, ratios = [], [], [], []
            clamped = latents.new_zeros(4)
            finite = torch.ones((), device=latents.device, dtype=torch.bool)
            for time in range(first, end):
                observations = observe(time)
                for value in observations:
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        finite &= value.isfinite().all()
                motor, neural = controller(*observations, neural)
                finite &= motor.isfinite().all() & neural.isfinite().all()
                bounded = motor.clamp(-.999999, .999999)
                mean = torch.atanh(bounded)
                conditional = (mean if time == 0 else
                               mean + rho * (previous_latents[time] - previous_mean))
                log_prob = joint_log_prob(latents[time], conditional, std[time], squashed=False)
                ratio = (log_prob - old_log_prob[time]).exp()
                finite &= log_prob.isfinite().all() & ratio.isfinite().all()
                loss = -torch.minimum(ratio * advantages[time],
                                      ratio.clamp(1-clip, 1+clip) * advantages[time])
                losses.append(loss[valid[time]].sum() / count)
                with torch.no_grad():
                    means.append(mean.detach())
                    kls.append(conditional_kl(old_conditional[time], conditional, std[time]))
                    ratios.append(ratio.detach())
                    clamped += ((bounded != motor) & valid[time, :, None]).sum(0)
                previous_mean = mean
            loss = torch.stack(losses).sum()
            if not bool(finite & loss.isfinite()):
                raise FloatingPointError("nonfinite full-history policy replay")
            return (neural, previous_mean, loss, torch.stack(means), torch.stack(kls),
                    torch.stack(ratios), clamped)
        return forward

    means, kls, ratios, losses = [], [], [], []
    clamped = latents.new_zeros(4)
    with torch.set_grad_enabled(backward):
        neural = run(warmup, neural) if warmup_steps else neural
        previous_mean = latents.new_zeros(episodes, 4)
        for first in range(0, times, chunk_steps):
            neural, previous_mean, loss, mu, kl, ratio, clamps = run(
                segment(first, min(first + chunk_steps, times)), neural, previous_mean,
            )
            losses.append(loss)
            means.append(mu.detach())
            kls.append(kl.detach())
            ratios.append(ratio.detach())
            clamped += clamps.detach()
        total_loss = torch.stack(losses).sum()
        if backward:
            (gradient_scale * total_loss).backward()
    if backward and any(p.grad is not None and not bool(p.grad.isfinite().all())
                        for p in controller.parameters()):
        raise FloatingPointError("nonfinite full-history gradient; discard accumulated gradients")
    kl = torch.cat(kls)[valid]
    ratio = torch.cat(ratios)[valid]
    displacement = (torch.cat(means) - old_means)[valid]
    if not bool(kl.isfinite().all() & displacement.isfinite().all()):
        raise FloatingPointError("nonfinite full-history diagnostics")
    result = dict(
        loss=float(total_loss.detach()), valid_commands=count,
        joint_kl_mean=float(kl.mean()), joint_kl_p99=float(torch.quantile(kl, .99)),
        joint_kl_max=float(kl.max()), joint_ratio_mean=float(ratio.mean()),
        clipped_command_fraction=float(((ratio-1).abs() > clip).float().mean()),
        native_latent_mean_displacement_rms=displacement.square().mean(0).sqrt().tolist(),
        native_clamped_fraction=(clamped/count).tolist(), activation_chunk_steps=chunk_steps,
        recurrent_gradient_is_truncated=False, warmup_is_differentiated=True,
        checkpointed=checkpointed,
    )
    if include_kl_samples:
        result["joint_kl_samples"] = kl.cpu().tolist()
    return result
