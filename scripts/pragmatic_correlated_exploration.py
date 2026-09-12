#!/usr/bin/env python3
"""Training-only AR(1) action-density helpers and a CPU foreleg bandwidth probe.

No sampler, previous-action buffer or critic is added to the deployed fly. Latent
actions are saved before tanh; the plant receives tanh(latent). Deterministic
deployment still uses the native motor output at every ordinary brain tick.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.hover import ForelegStickPlant, HoverConfig, motor_target_for_rc  # noqa: E402


def ar1_conditional(mean, previous_mean, previous_latent, stationary_std, rho, first):
    """All four latent axes, conditioned on the same recorded action history.

    previous_mean must be recomputed under the CURRENT weights, with gradient.
    first marks actual episode starts, never gates or truncated replay boundaries.
    Prefix detachment still makes the neural gradient a TBPTT approximation.
    """
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be finite and in [0, 1)")
    std = torch.as_tensor(stationary_std, device=mean.device, dtype=mean.dtype)
    if std.shape != (4,) or not bool(torch.isfinite(std).all() & (std > 0).all()):
        raise ValueError("need four finite positive stationary standard deviations")
    if (
        mean.shape[-1] != 4
        or previous_mean.shape != mean.shape
        or previous_latent.shape != mean.shape
    ):
        raise ValueError("current/previous means and latents must have matching four-axis shapes")
    if first.shape != mean.shape[:-1] or first.dtype != torch.bool:
        raise ValueError("episode-start flags must match the non-action dimensions")
    conditional_mean = torch.where(
        first[..., None], mean, mean + rho * (previous_latent - previous_mean)
    )
    conditional_std = torch.where(first[..., None], std, std * math.sqrt(1 - rho * rho))
    return conditional_mean, conditional_std


def joint_log_prob(latent, mean, std, *, squashed=True):
    """Joint four-action density; retain raw latents, never invert clipped commands."""
    density = -0.5 * ((latent - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)
    if squashed:
        # Stable log(1 - tanh(u)^2), including large finite u.
        density = density - 2 * (math.log(2) - latent - functional.softplus(-2 * latent))
    return density.sum(dim=-1)


def conditional_kl(old_mean, new_mean, std):
    """Exact conditional KL for fixed exploration parameters, summed across axes."""
    return 0.5 * ((new_mean - old_mean) / std).square().sum(dim=-1)


@torch.no_grad()
def foreleg_probe(config, stationary_std, tau, *, episodes=256, seconds=30, seed=2026091390):
    """Isolated physical forelegs, not a brain/aircraft/course success test."""
    if tau < 0 or not math.isfinite(tau) or episodes < 1 or seconds <= 2:
        raise ValueError("need nonnegative finite tau, episodes and more than two seconds")
    if not math.isclose(config.dt, 0.01):
        raise ValueError("this probe uses 50 Hz commands and 100 Hz foreleg physics")
    rho = math.exp(-0.02 / tau) if tau else 0.0
    legs = ForelegStickPlant(config)
    sticks = legs.initial_state(episodes, device=torch.device("cpu"), dtype=torch.float32)
    rc = torch.zeros(episodes, 4)
    rc[:, 3] = 1 / config.thrust_to_weight
    mean_motor = motor_target_for_rc(rc, config)
    mean = torch.atanh(mean_motor.clamp(-0.999999, 0.999999))
    sticks.position.copy_(rc)
    sticks.position[:, 3] = 2 * rc[:, 3] - 1
    sticks.joint_position.copy_(torch.asin(sticks.position * math.sin(config.foreleg_joint_limit)))
    generator = torch.Generator().manual_seed(seed)
    previous_latent = mean.clone()
    latent_squares = torch.zeros(4)
    motor_squares = torch.zeros(4)
    stick_squares = torch.zeros(4)
    lag_products = torch.zeros(4)
    previous_squares = torch.zeros(4)
    previous_noise = torch.zeros_like(mean)
    counted = 0
    saturated = 0
    first = torch.ones(episodes, dtype=torch.bool)
    for step in range(round(seconds * 50)):
        conditional_mean, std = ar1_conditional(
            mean, mean, previous_latent, stationary_std, rho, first
        )
        latent = conditional_mean + std * torch.randn(mean.shape, generator=generator)
        motor = torch.tanh(latent)
        for _ in range(2):
            measured_rc, sticks = legs(motor, sticks)
        noise = latent - mean
        if step >= 100:  # Exclude initial physical settling, not a neural warmup.
            latent_squares += noise.square().sum(0)
            motor_squares += (motor - mean_motor).square().sum(0)
            stick_squares += (measured_rc - rc).square().sum(0)
            lag_products += (noise * previous_noise).sum(0)
            previous_squares += previous_noise.square().sum(0)
            saturated += int((sticks.position.abs() >= 0.999).any(1).sum())
            counted += episodes
        previous_latent, previous_noise = latent, noise
        first.fill_(False)
    return dict(
        correlation_seconds=tau,
        rho=rho,
        episodes=episodes,
        seconds=seconds,
        axis_order=["roll", "pitch", "yaw", "throttle"],
        stationary_latent_std=list(stationary_std),
        latent_rms=(latent_squares / counted).sqrt().tolist(),
        motor_perturbation_rms=(motor_squares / counted).sqrt().tolist(),
        measured_rc_perturbation_rms=(stick_squares / counted).sqrt().tolist(),
        latent_lag_one_regression=(lag_products / previous_squares).tolist(),
        stick_saturated_frame_fraction=saturated / counted,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an exploration probe")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = HoverConfig(**payload["hover_config"])
    standard_deviations = (0.006, 0.002, 0.001, 0.0025)
    records = [foreleg_probe(config, standard_deviations, tau) for tau in (0, 0.1, 0.3, 0.6)]
    result = dict(
        scope="CPU isolated foreleg response; no brain, quad flight or learning",
        source_checkpoint=str(args.checkpoint),
        hover_config=vars(config),
        command_hz=50,
        physics_hz=100,
        noise_deployed=False,
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
