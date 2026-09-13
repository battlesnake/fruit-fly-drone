#!/usr/bin/env python3
"""Measure training-only exploration on the real native five-gate controller.

No learning, teacher or checkpoint output. Every condition uses the same initial
courses and full-duration evaluator; only noisy conditions alter native commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_pragmatic_two_gate_zero_shot import evaluate, sample_two_gate_cases  # noqa: E402
from pragmatic_correlated_exploration import ar1_conditional  # noqa: E402
from train_pragmatic_course_replay import GEOMETRY  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402

from flydrone.gate import GateConfig  # noqa: E402
from flydrone.hover import HoverConfig  # noqa: E402
from flydrone.visual_hover import CameraSpec  # noqa: E402

BASE_STD = (0.006, 0.002, 0.001, 0.0025)
CONDITIONS = (
    ("native", False, 0.0, 0.0),
    ("zero-noise-roundtrip", True, 0.0, 0.0),
    ("white-scale-1", True, 0.0, 1.0),
    ("ar-scale-0.25", True, 0.6, 0.25),
    ("ar-scale-0.5", True, 0.6, 0.5),
    ("ar-scale-1", True, 0.6, 1.0),
)


class ExplorationAuditController:
    """Training-only sampler; the underlying controller sees only RGB and attitude."""

    def __init__(self, controller, *, transformed, tau, scale, noise_seed, warmup_steps=10):
        if not math.isfinite(tau) or tau < 0 or not math.isfinite(scale) or scale < 0:
            raise ValueError("correlation time and scale must be finite and nonnegative")
        if warmup_steps < 0:
            raise ValueError("warmup steps must be nonnegative")
        if scale and not transformed:
            raise ValueError("noisy commands require the latent transform")
        self.controller = controller
        self.transformed = transformed
        self.scale = scale
        self.rho = math.exp(-0.02 / tau) if tau else 0.0
        self.noise_seed = noise_seed
        self.warmup_steps = warmup_steps

    def initial_state(self, count, *, device, dtype):
        self.calls = self.samples = self.physical_commands = 0
        self.generator = torch.Generator(device=device).manual_seed(self.noise_seed)
        self.previous_mean = self.previous_latent = None
        self.first = torch.ones(count, device=device, dtype=torch.bool)
        self.squares = torch.zeros(2, 4, device=device, dtype=dtype)
        self.clipped = torch.zeros(4, device=device, dtype=torch.long)
        self.roundtrip_max = torch.zeros(4, device=device, dtype=dtype)
        self.native_abs_max = torch.zeros(4, device=device, dtype=dtype)
        return self.controller.initial_state(count, device=device, dtype=dtype)

    @torch.no_grad()
    def __call__(self, image, attitude, neural):
        native, neural = self.controller(image, attitude, neural)
        self.calls += 1
        if self.calls <= self.warmup_steps:
            return native, neural
        bounded = native.clamp(-0.999999, 0.999999)
        mean = torch.atanh(bounded)
        roundtrip = torch.tanh(mean)
        if self.previous_mean is None:
            self.previous_mean, self.previous_latent = mean, mean
        if self.scale:
            conditional, std = ar1_conditional(
                mean, self.previous_mean, self.previous_latent,
                [value * self.scale for value in BASE_STD], self.rho, self.first,
            )
            innovation = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype,
                                     generator=self.generator)
            latent = conditional + std * innovation
        else:
            latent = mean
        motor = torch.tanh(latent) if self.transformed else native
        # Residuals are against THIS flight's native mean, not baseline-flight actions.
        self.squares[0] += (latent - mean).square().sum(0)
        self.squares[1] += (motor - native).square().sum(0)
        self.clipped += (native != bounded).sum(0)
        self.roundtrip_max = torch.maximum(self.roundtrip_max, (roundtrip - native).abs().amax(0))
        self.native_abs_max = torch.maximum(self.native_abs_max, native.abs().amax(0))
        self.samples += len(native)
        self.physical_commands += 1
        self.previous_mean, self.previous_latent = mean, latent
        self.first.fill_(False)
        return motor, neural

    def summary(self):
        if not self.samples:
            raise ValueError("no physical commands observed")
        return dict(
            command_count_per_episode=self.physical_commands,
            stationary_latent_std=[value * self.scale for value in BASE_STD],
            rho=self.rho,
            latent_residual_rms=(self.squares[0] / self.samples).sqrt().tolist(),
            motor_perturbation_rms=(self.squares[1] / self.samples).sqrt().tolist(),
            native_clamped_fraction=(self.clipped / self.samples).tolist(),
            zero_noise_roundtrip_max_error=self.roundtrip_max.tolist(),
            native_motor_absolute_max=self.native_abs_max.tolist(),
            residual_scope="all physical commands, including post-failure tail",
        )


class FlightTrace:
    def __init__(self):
        self.positions, self.rc, self.failed = [], [], []

    def __call__(self, state, rc, failed):
        self.positions.append(state.position.detach().clone())
        self.rc.append(rc.detach().clone())
        self.failed.append(failed.detach().clone())

    def finish(self):
        return tuple(torch.stack(values).cpu() for values in (self.positions, self.rc, self.failed))


def trace_difference(trace, reference):
    position, rc, failed = trace
    base_position, base_rc, base_failed = reference
    usable = ~(failed | base_failed)
    def rms(delta, mask):
        return delta[mask].square().mean(0).sqrt().tolist() if bool(mask.any()) else None
    return dict(
        scope="matched initial courses; before either flight's first failure",
        common_unfailed_command_samples=int(usable.sum()),
        position_rms_metres=rms(position - base_position, usable),
        rc_difference_rms=rms(rc - base_rc, usable),
    )


def survival_screen(metrics, baseline, difference):
    """Coarse exploration nomination, not learning success or deployment approval."""
    rc, position = difference["rc_difference_rms"], difference["position_rms_metres"]
    criteria = dict(
        zero_ground=metrics["ground_contact_rate"] == 0,
        zero_invalid=metrics["invalid_rate"] == 0,
        retains_half_clean=metrics["clean_course_success_rate"] >= max(
            1 / metrics["episodes"], 0.5 * baseline["clean_course_success_rate"]
        ),
        retains_three_quarters_clean_first=metrics["clean_first_gate_pass_rate"]
        >= 0.75 * baseline["clean_first_gate_pass_rate"],
        limited_added_stick_saturation=metrics["stick_saturation_fraction_mean"]
        <= baseline["stick_saturation_fraction_mean"] + 0.05,
        measurable_roll_stick_change=rc is not None and rc[0] >= 1e-4,
        measurable_trajectory_change=position is not None and max(position) >= 1e-3,
    )
    return dict(eligible=all(criteria.values()), criteria=criteria)


def roundtrip_screen(metrics, baseline, commands):
    """Broad behavior check plus local transform error, not bit-exact replay."""
    tolerance = max(2 / metrics["episodes"], 0.1)
    criteria = dict(
        no_native_clamps=max(commands["native_clamped_fraction"]) == 0,
        negligible_command_error=max(commands["zero_noise_roundtrip_max_error"]) <= 1e-6,
        comparable_clean=abs(metrics["clean_course_success_rate"]
                             - baseline["clean_course_success_rate"]) <= tolerance,
        comparable_first=abs(metrics["clean_first_gate_pass_rate"]
                             - baseline["clean_first_gate_pass_rate"]) <= tolerance,
        no_added_ground=metrics["ground_contact_rate"] <= baseline["ground_contact_rate"],
        no_added_invalid=metrics["invalid_rate"] <= baseline["invalid_rate"],
    )
    return dict(eligible=all(criteria.values()), criteria=criteria)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026091391)
    parser.add_argument("--noise-seed", type=int, default=2026091392)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an exploration calibration")
    if args.pairs < 1:
        raise SystemExit("need positive course pairs")
    device = torch.device(args.device)
    controller, payload = load_controller(args, device)
    controller.eval().requires_grad_(False)
    config = HoverConfig(**payload["hover_config"])
    if not math.isclose(config.dt, 0.01):
        raise SystemExit("calibration requires the existing 100 Hz physics")
    gate_config = replace(GateConfig(**payload["gate_config"]), back_pattern="checkerboard")
    camera = CameraSpec(*payload["image_resolution"], payload["camera_hfov_degrees"])
    bank = sample_two_gate_cases(args.pairs, seed=args.seed, device=device,
                                 hover_config=config, **GEOMETRY)
    result = dict(
        experiment="native-five-gate-exploration-calibration-v1", status="running",
        checkpoint=str(args.checkpoint), seed=args.seed, noise_seed=args.noise_seed,
        scope="training-data calibration; no learning or native success claim",
        geometry=GEOMETRY, episodes=2 * args.pairs, seconds=30,
        image_resolution=[camera.width, camera.height],
        camera_hfov_degrees=camera.horizontal_fov_degrees,
        actor_inputs=["live RGB", "roll", "pitch"],
        command_hz=50, physics_hz=100, warmup_steps=10,
        independent_noise_across_episodes=True, matched_innovations_across_conditions=True,
        deployment_has_noise_or_history=False, conditions={},
        nomination_rule="largest AR scale passing survival screen, not highest success",
        selected_condition=None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    reference_trace = baseline = None
    for name, transformed, tau, scale in CONDITIONS:
        actor = ExplorationAuditController(controller, transformed=transformed, tau=tau,
                                          scale=scale, noise_seed=args.noise_seed)
        observer = FlightTrace()
        metrics = evaluate(actor, *bank, seconds=30, warmup_steps=10,
                           camera=camera, hover_config=config, gate_config=gate_config,
                           trajectory_observer=observer)
        trace = observer.finish()
        if name == "native":
            reference_trace, baseline = trace, metrics
        difference = trace_difference(trace, reference_trace)
        record = dict(correlation_seconds=tau, scale=scale, metrics=metrics,
                      commands=actor.summary(), trajectory_difference=difference)
        if name == "zero-noise-roundtrip":
            result["roundtrip_control"] = roundtrip_screen(metrics, baseline, record["commands"])
        if name.startswith("ar-"):
            record["survival_screen"] = survival_screen(metrics, baseline, difference)
            if (record["survival_screen"]["eligible"]
                    and result["roundtrip_control"]["eligible"]):
                result["selected_condition"] = name
        result["conditions"][name] = record
        result["elapsed_seconds"] = perf_counter() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(stage=name, clean=metrics["clean_course_success_rate"],
                              ground=metrics["ground_contact_rate"],
                              first=metrics["clean_first_gate_pass_rate"],
                              commands=record["commands"], difference=difference,
                              survival=record.get("survival_screen"),
                              elapsed_seconds=result["elapsed_seconds"])), flush=True)
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(stage="complete", selected=result["selected_condition"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
