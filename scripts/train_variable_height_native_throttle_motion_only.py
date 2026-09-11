#!/usr/bin/env python3
"""Test native visual damping learnability with a complete motion-only bank."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_factorial_damping as factorial  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-native-throttle-motion-only-v1"
PROTOCOL_COMMIT = "77340c1"
EXPECTED_GRAPH_SHA256 = assisted.EXPECTED_GRAPH_SHA256
EXPECTED_CHECKPOINT_SHA256 = assisted.EXPECTED_CHECKPOINT_SHA256
EXPECTED_PRODUCER_REPORT_SHA256 = (
    "549259e08db3dd1dc43afda0ea85b76f86fac115a5a28bb0aca6700dbccb9ed9"
)
PRODUCER_REPORT = Path(
    "runs/variable-height-hover/native-throttle-assisted-beta1-zero-ladder-001/report.json"
)

TRAINING_SEED = 450_991
DEVELOPMENT_SEED = 460_991
FORMAL_SEEDS = frozenset((TRAINING_SEED, DEVELOPMENT_SEED))
CASES = 24
POLICY_HZ = assisted.POLICY_HZ
PREFIX_STEPS = assisted.MOTION_PREFIX_STEPS
HEIGHT_AMPLITUDES = assisted.TRAIN_MOTION_HEIGHT_AMPLITUDES
SPEED_AMPLITUDES = assisted.TRAIN_MOTION_SPEED_AMPLITUDES
HISTORY_LENGTHS = assisted.MOTION_HISTORY_LENGTHS
APPROACH_FRACTION = assisted.MOTION_APPROACH_FRACTION
MOTION_SCALE_FLOOR = assisted.DENOMINATOR_RMS_FLOOR
PARAMETER_FAMILIES = assisted.PARAMETER_FAMILIES
OPTIMIZER_BETAS = (0.0, 0.999)
BACKTRACK_SCALES = assisted.BACKTRACK_SCALES
DERIVATIVE_PROBE_SCALES = (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512)
DERIVATIVE_BASELINE_REPEATS = 3
DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES = 2
DERIVATIVE_REPLAY_NOISE_MULTIPLIER = 10.0
DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE = 1.0e-8
DERIVATIVE_RELATIVE_ERROR_LIMIT = 0.20
MINIMUM_OBJECTIVE_IMPROVEMENT = 1.0e-4
MAX_ACCEPTED_UPDATES = 100
MAX_ATTEMPTED_UPDATES = 100
MILESTONE_UPDATE = 25
MILESTONE_NRMSE_IMPROVEMENT = 0.25
MILESTONE_SIGN_FRACTION = 0.50
FINAL_NRMSE_IMPROVEMENT = 0.50
FINAL_SIGN_FRACTION = 0.90
FINAL_GAIN_RANGE = (0.5, 1.5)


def require_nonformal_seeds(*seeds: int) -> None:
    collisions = sorted(FORMAL_SEEDS.intersection(seeds))
    if collisions:
        raise ValueError(f"disposable execution attempted to consume formal seeds: {collisions}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-throttle-motion-only-001",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_authorizing_report(path: Path = PRODUCER_REPORT) -> None:
    resolved = path if path.is_absolute() else REPO_ROOT / path
    if not resolved.is_file():
        raise SystemExit(f"missing authorizing assisted-throttle report: {resolved}")
    if assisted.responsibility.file_sha256(resolved) != EXPECTED_PRODUCER_REPORT_SHA256:
        raise SystemExit("authorizing assisted-throttle report hash mismatch")


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if assisted.responsibility.file_sha256(args.graph) != EXPECTED_GRAPH_SHA256:
        raise SystemExit("graph does not match the preregistered source")
    if assisted.responsibility.file_sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit("checkpoint does not match the preregistered source")
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the motion-only experiment already has a terminal report")
    if CASES != len(HEIGHT_AMPLITUDES) * 2 * len(SPEED_AMPLITUDES) * len(HISTORY_LENGTHS):
        raise SystemExit("the registered motion bank is not one exact full factorial")


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source": "original native visual-hover checkpoint",
        "authorizing_report_sha256": EXPECTED_PRODUCER_REPORT_SHA256,
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "privileged_inputs": [],
            "state": "native MaleCNS recurrence only",
            "outputs": ["roll", "pitch", "yaw", "throttle"],
            "optimized_output": "paired native throttle contrast only",
        },
        "bank": {
            "training_seed": TRAINING_SEED,
            "development_seed": DEVELOPMENT_SEED,
            "cases": CASES,
            "exact_full_factorial": {
                "height_error_magnitudes_metres": list(HEIGHT_AMPLITUDES),
                "height_error_signs": [-1, 1],
                "vertical_speed_magnitudes_mps": list(SPEED_AMPLITUDES),
                "history_lengths_policy_frames": list(HISTORY_LENGTHS),
            },
            "neutral_prefix_frames": PREFIX_STEPS,
            "training_styles": "ordinary matched wall/floor combinations",
            "development_styles": "held-out wall/floor combinations",
            "development_generated_only_after_training_final_pass": True,
        },
        "objective": {
            "terms": ["normalized paired equal-endpoint throttle contrast MSE"],
            "dense_or_source_anchor": False,
            "all_training_pairs_per_update": True,
            "fixed_order": True,
            "normalization": "training teacher-contrast RMS floored at 0.01 motor units",
            "normalization_reused_on_development": True,
            "detached_neutral_prefix": True,
            "differentiated_response_lengths": list(HISTORY_LENGTHS),
            "minimum_fixed_and_full_prefix_improvement": MINIMUM_OBJECTIVE_IMPROVEMENT,
        },
        "optimizer": {
            "name": "Adam",
            "betas": list(OPTIMIZER_BETAS),
            "parameter_families": list(PARAMETER_FAMILIES),
            "edge_and_bias_learning_rate": assisted.EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": assisted.TIME_CONSTANT_LEARNING_RATE,
            "gradient_norm_cap": assisted.GRADIENT_NORM_CAP,
            "ordinary_scales_descending": list(BACKTRACK_SCALES),
            "first_finite_no_scale_result_is_terminal": True,
            "maximum_accepted_updates": MAX_ACCEPTED_UPDATES,
            "maximum_attempted_updates": MAX_ATTEMPTED_UPDATES,
        },
        "derivative_ladder": {
            "baseline_repeats": DERIVATIVE_BASELINE_REPEATS,
            "probe_scales": list(DERIVATIVE_PROBE_SCALES),
            "required_adjacent_passes": DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES,
            "replay_noise_multiplier": DERIVATIVE_REPLAY_NOISE_MULTIPLIER,
            "minimum_objective_change": DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE,
            "relative_error_limit": DERIVATIVE_RELATIVE_ERROR_LIMIT,
            "probes_select_ordinary_step": False,
        },
        "milestone_25": {
            "minimum_source_relative_nrmse_improvement": MILESTONE_NRMSE_IMPROVEMENT,
            "minimum_correct_sign_fraction": MILESTONE_SIGN_FRACTION,
        },
        "training_final_100_and_development": {
            "minimum_source_relative_nrmse_improvement": FINAL_NRMSE_IMPROVEMENT,
            "minimum_correct_sign_fraction_overall_and_each_horizon": FINAL_SIGN_FRACTION,
            "teacher_aligned_gain_range_overall_and_each_horizon": list(FINAL_GAIN_RANGE),
            "exact_endpoint_images": True,
            "finite_recurrence_outputs_and_metrics": True,
            "maximum_motor_absolute": 1.0,
        },
        "passing_authorizes": "a separately preregistered staged assisted-throttle curriculum",
        "assisted_hover": False,
        "native_attitude_reintegration": False,
        "full_native_hover": False,
        "gate_flight": False,
        "promotion": False,
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    return assisted.audit.semantic_sha256(
        {name: value for name, value in payload.items() if name != "sha256"}
    )


@torch.no_grad()
def build_exact_factorial_motion_bank(
    *, seed: int, held_out_styles: bool, device: torch.device, config: HoverConfig
) -> dict[str, Any]:
    bank = assisted.build_motion_bank(
        seed=seed,
        cases=CASES,
        height_amplitudes=HEIGHT_AMPLITUDES,
        speed_amplitudes=SPEED_AMPLITUDES,
        held_out_styles=held_out_styles,
        device=device,
        config=config,
    )
    combinations = list(
        itertools.product(HEIGHT_AMPLITUDES, (-1.0, 1.0), SPEED_AMPLITUDES, HISTORY_LENGTHS)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(combinations), generator=generator)
    ordered = [combinations[index] for index in permutation.tolist()]
    bank["height_error"] = torch.tensor([item[0] for item in ordered], dtype=torch.float32)
    bank["height_sign"] = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
    bank["speed"] = torch.tensor([item[2] for item in ordered], dtype=torch.float32)
    bank["horizon"] = torch.tensor([item[3] for item in ordered], dtype=torch.long)
    bank["approach_steps"] = torch.maximum(
        torch.full_like(bank["horizon"], 5),
        torch.round(bank["horizon"].float() * APPROACH_FRACTION).long(),
    )
    bank["factorial_permutation"] = permutation
    bank["sha256"] = _payload_hash(bank)
    return bank


def motion_bank_manifest(bank: dict[str, Any]) -> dict[str, Any]:
    observed = [
        (
            float(bank["height_error"][index]),
            float(bank["height_sign"][index]),
            float(bank["speed"][index]),
            int(bank["horizon"][index]),
        )
        for index in range(bank["cases"])
    ]
    rounded = {(round(a, 6), int(b), round(c, 6), d) for a, b, c, d in observed}
    expected = {
        (round(a, 6), int(b), round(c, 6), d)
        for a, b, c, d in itertools.product(
            HEIGHT_AMPLITUDES, (-1.0, 1.0), SPEED_AMPLITUDES, HISTORY_LENGTHS
        )
    }
    return {
        "seed": int(bank["seed"]),
        "sha256": bank["sha256"],
        "cases": int(bank["cases"]),
        "held_out_styles": bool(bank["held_out_styles"]),
        "exact_full_factorial": rounded == expected and len(rounded) == CASES,
        "unique_combinations": len(rounded),
        "history_length_counts": {
            str(length): int((bank["horizon"] == length).sum()) for length in HISTORY_LENGTHS
        },
        "height_sign_counts": {
            str(sign): int((bank["height_sign"] == sign).sum()) for sign in (-1.0, 1.0)
        },
        "permutation": bank["factorial_permutation"].tolist(),
    }


def sampled_endpoint_velocity_report(bank: dict[str, Any]) -> dict[str, Any]:
    ratios = []
    for index in range(bank["cases"]):
        horizon = int(bank["horizon"][index])
        approach = bank["approach_steps"][index].reshape(1)
        speed = bank["speed"][index].reshape(1)
        previous = factorial.smooth_return_offset(
            speed,
            step=horizon - 2,
            total_steps=horizon,
            approach_steps=approach,
            policy_hz=POLICY_HZ,
        )
        endpoint = factorial.smooth_return_offset(
            speed,
            step=horizon - 1,
            total_steps=horizon,
            approach_steps=approach,
            policy_hz=POLICY_HZ,
        )
        sampled_velocity = (endpoint - previous) * POLICY_HZ
        ratios.append(float(sampled_velocity / speed))
    finite = all(math.isfinite(value) for value in ratios)
    sign_matches = all(value > 0.0 for value in ratios)
    return {
        "pass": finite and sign_matches,
        "all_finite": finite,
        "all_nonzero_signs_match_label": sign_matches,
        "ratio_sampled_last_frame_to_continuous_endpoint_label": ratios,
        "minimum_ratio": min(ratios),
        "maximum_ratio": max(ratios),
        "fixed_excursion_metres": 0.055,
        "magnitude_match_required": False,
    }


def frozen_motion_scale(bank: dict[str, Any], *, config: HoverConfig) -> dict[str, float]:
    targets = assisted.teacher_motion_contrasts(bank, config=config)
    rms = float(targets.square().mean().sqrt())
    return {"motion": max(rms, MOTION_SCALE_FLOOR), "unfloored_motion": rms}


def _motion_summary(
    losses: list[float], reports: list[dict[str, Any]], bank: dict[str, Any]
) -> dict[str, Any]:
    prediction = torch.stack([report["prediction"] for report in reports]).detach().cpu()
    target = torch.stack([report["target"] for report in reports]).detach().cpu()
    pair_means = torch.stack([report["throttle_pair_mean"] for report in reports]).detach().cpu()
    motor_outputs = torch.cat([report["motor_outputs"] for report in reports]).detach().cpu()
    height_sign = bank["height_sign"].detach().cpu()
    height_error = bank["height_error"].detach().cpu()
    target_power = target.square().sum().clamp_min(1.0e-12)
    by_horizon: dict[str, Any] = {}
    for horizon in HISTORY_LENGTHS:
        indices = [index for index, report in enumerate(reports) if report["horizon"] == horizon]
        horizon_prediction = prediction[indices]
        horizon_target = target[indices]
        horizon_power = horizon_target.square().sum().clamp_min(1.0e-12)
        by_horizon[str(horizon)] = {
            "pairs": len(indices),
            "correct_sign_fraction": float(
                ((horizon_prediction * horizon_target) > 0).float().mean()
            ),
            "teacher_aligned_gain": float(
                (horizon_prediction * horizon_target).sum() / horizon_power
            ),
            "nrmse": float(math.sqrt(sum(losses[index] for index in indices) / len(indices))),
        }
    motor_axis_summary = {}
    for index, name in enumerate(("roll", "pitch", "yaw", "throttle")):
        values = motor_outputs[:, index]
        motor_axis_summary[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "rms": float(values.square().mean().sqrt()),
        }
    height_by_magnitude = {}
    for amplitude in HEIGHT_AMPLITUDES:
        selected = torch.isclose(height_error, torch.tensor(amplitude))
        positive = pair_means[selected & (height_sign > 0)]
        negative = pair_means[selected & (height_sign < 0)]
        height_by_magnitude[str(amplitude)] = float(positive.mean() - negative.mean())
    result = {
        "objective": float(sum(losses) / len(losses)),
        "nrmse": float(math.sqrt(sum(losses) / len(losses))),
        "pairs": len(reports),
        "correct_sign_fraction": float(((prediction * target) > 0).float().mean()),
        "teacher_aligned_gain": float((prediction * target).sum() / target_power),
        "prediction_rms_motor_units": float(prediction.square().mean().sqrt()),
        "target_rms_motor_units": float(target.square().mean().sqrt()),
        "endpoint_image_difference_max": max(
            report["endpoint_image_difference_max"] for report in reports
        ),
        "maximum_motor_absolute": max(report["maximum_motor_absolute"] for report in reports),
        "motor_axis_summary": motor_axis_summary,
        "throttle_pair_means": pair_means.tolist(),
        "descriptive_unpaired_height_sign_contrast_motor_units": float(
            pair_means[height_sign > 0].mean() - pair_means[height_sign < 0].mean()
        ),
        "descriptive_unpaired_height_sign_contrast_by_error_magnitude": height_by_magnitude,
        "height_contrast_is_scene_confounded_and_non_gating": True,
        "by_horizon": by_horizon,
        "all_recurrent_states_and_outputs_finite": bool(
            all(
                report["recurrent_state_finite"] and report["actor_output_finite"]
                for report in reports
            )
        ),
    }
    result["all_metrics_finite"] = assisted._all_finite_nested(result)
    return result


@torch.no_grad()
def motion_fixed_report(
    controller: ConnectomeController,
    bank: dict[str, Any],
    burns: list[Tensor],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    losses, reports = [], []
    for index, burn in enumerate(burns):
        loss, report = assisted.motion_example(
            controller, bank, index, burn, motion_scale=motion_scale, device=device
        )
        losses.append(float(loss))
        reports.append(report)
    return _motion_summary(losses, reports, bank)


@torch.no_grad()
def motion_full_prefix_report(
    controller: ConnectomeController,
    bank: dict[str, Any],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    losses, reports = [], []
    for index in range(bank["cases"]):
        loss, report = assisted.motion_full_prefix_example(
            controller, bank, index, motion_scale=motion_scale, device=device
        )
        losses.append(float(loss))
        reports.append(report)
    return _motion_summary(losses, reports, bank)


def motion_gradient(
    controller: ConnectomeController,
    bank: dict[str, Any],
    burns: list[Tensor],
    *,
    motion_scale: float,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    for parameter in controller.parameters():
        parameter.grad = None
    losses, reports = [], []
    for index, burn in enumerate(burns):
        loss, report = assisted.motion_example(
            controller, bank, index, burn, motion_scale=motion_scale, device=device
        )
        (loss / bank["cases"]).backward()
        losses.append(float(loss.detach()))
        reports.append(report)
    summary = _motion_summary(losses, reports, bank)
    gradients_finite = all(
        getattr(controller, name).grad is not None
        and bool(torch.isfinite(getattr(controller, name).grad).all())
        for name in PARAMETER_FAMILIES
    )
    raw_gradients, gradient_norm = assisted.clone_raw_gradients_and_clip(controller)
    summary.update(
        {
            "gradients_finite": gradients_finite,
            "gradient_norm_before_clipping": float(gradient_norm),
        }
    )
    return summary, raw_gradients


def _reports_finite(*reports: dict[str, Any]) -> bool:
    return all(
        report["all_recurrent_states_and_outputs_finite"]
        and report["all_metrics_finite"]
        and assisted._all_finite_nested(report)
        for report in reports
    )


def _local_derivative_agreement(record: dict[str, Any]) -> bool:
    return bool(
        record["controls"]["pass"]
        and record["all_metrics_finite"]
        and record["predicted_directional_derivative"] < -1.0e-8
        and record["measured_directional_derivative"] < -1.0e-8
        and record["relative_error"] <= DERIVATIVE_RELATIVE_ERROR_LIMIT
        and abs(record["objective_change"]) > record["objective_change_noise_threshold"]
    )


def _probe_decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    def windows(key: str) -> list[list[float]]:
        return [
            [float(item["scale"]) for item in records[begin : begin + 2]]
            for begin in range(len(records) - 1)
            if all(item[key] for item in records[begin : begin + 2])
        ]

    numerical = any(record["numerical_failure"] for record in records)
    passing = windows("local_derivative_agreement")
    above_noise = windows("objective_change_exceeds_noise_threshold")
    if numerical:
        classification = "derivative_probe_numerical_failure"
    elif passing:
        classification = "adjacent_local_derivative_agreement"
    elif above_noise:
        classification = "derivative_probe_above_noise_nonconvergence"
    else:
        classification = "derivative_probe_noise_limited_inconclusive"
    return {
        "pass": bool(passing and not numerical),
        "classification": classification,
        "qualified_consecutive_scale_windows": passing,
        "above_noise_consecutive_scale_windows": above_noise,
    }


def run_derivative_ladder(
    controller: ConnectomeController,
    current: dict[str, Tensor],
    displacement: dict[str, Tensor],
    raw_gradients: dict[str, Tensor],
    authoritative_directional: float,
    baseline_objectives: list[float],
    bank: dict[str, Any],
    burns: list[Tensor],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    replay_noise = assisted.derivative_replay_noise(
        baseline_objectives, expected_repeats=DERIVATIVE_BASELINE_REPEATS
    )
    noise_threshold = max(
        DERIVATIVE_MINIMUM_OBJECTIVE_CHANGE,
        DERIVATIVE_REPLAY_NOISE_MULTIPLIER * replay_noise,
    )
    records = []
    for scale in DERIVATIVE_PROBE_SCALES:
        try:
            controls, effective = assisted.materialize_scaled_proposal(
                controller, current, displacement, scale
            )
            if not controls["pass"]:
                record = {
                    "scale": scale,
                    "pass": False,
                    "local_derivative_agreement": False,
                    "numerical_failure": True,
                    "numerical_failure_reason": "probe failed bounds or canonical control",
                    "objective_change_exceeds_noise_threshold": False,
                    "objective_change_noise_threshold": noise_threshold,
                    "all_metrics_finite": False,
                    "controls": controls,
                    "candidate": None,
                }
            else:
                candidate = motion_fixed_report(
                    controller, bank, burns, motion_scale=motion_scale, device=device
                )
                objective_change = candidate["objective"] - baseline_objectives[0]
                predicted_change = float(
                    sum(
                        (raw_gradients[name].double() * effective[name].double()).sum()
                        for name in PARAMETER_FAMILIES
                    )
                )
                measured = objective_change / scale
                predicted = predicted_change / scale
                relative_error = assisted._relative_error(measured, predicted)
                finite = bool(
                    _reports_finite(candidate)
                    and math.isfinite(objective_change)
                    and math.isfinite(predicted_change)
                    and math.isfinite(measured)
                    and math.isfinite(predicted)
                    and math.isfinite(relative_error)
                )
                record = {
                    "scale": scale,
                    "pass": False,
                    "predicted_directional_derivative": predicted,
                    "measured_directional_derivative": measured,
                    "objective_change": objective_change,
                    "predicted_change": predicted_change,
                    "relative_error": relative_error,
                    "relative_error_limit": DERIVATIVE_RELATIVE_ERROR_LIMIT,
                    "objective_change_exceeds_noise_threshold": (
                        abs(objective_change) > noise_threshold
                    ),
                    "objective_change_noise_threshold": noise_threshold,
                    "uses_exact_frozen_full_bank_and_burns": True,
                    "all_metrics_finite": finite,
                    "numerical_failure": not finite,
                    "numerical_failure_reason": None if finite else "probe was nonfinite",
                    "controls": controls,
                    "candidate": candidate,
                }
                record["local_derivative_agreement"] = _local_derivative_agreement(record)
                record["pass"] = record["local_derivative_agreement"]
        finally:
            assisted._load_parameters(controller, current)
        records.append(record)
        if record["numerical_failure"]:
            break
        if len(records) >= DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES and all(
            item["local_derivative_agreement"]
            for item in records[-DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES:]
        ):
            break
    decision = _probe_decision(records)
    representative = records[-1]
    return {
        **decision,
        "authoritative_full_proposal_directional_derivative": authoritative_directional,
        "scale": representative["scale"],
        "measured_directional_derivative": representative.get(
            "measured_directional_derivative"
        ),
        "predicted_directional_derivative": representative.get(
            "predicted_directional_derivative"
        ),
        "relative_error": representative.get("relative_error"),
        "relative_error_limit": DERIVATIVE_RELATIVE_ERROR_LIMIT,
        "controls": representative["controls"],
        "candidate": representative.get("candidate"),
        "baseline": {
            "authoritative_j0_is_first_replay": True,
            "objectives": baseline_objectives,
            "replay_noise_max_pairwise_absolute": replay_noise,
            "noise_multiplier": DERIVATIVE_REPLAY_NOISE_MULTIPLIER,
            "objective_change_noise_threshold": noise_threshold,
        },
        "probe_scales_registered": list(DERIVATIVE_PROBE_SCALES),
        "required_consecutive_passes": DERIVATIVE_REQUIRED_CONSECUTIVE_PASSES,
        "probes": records,
        "probe_candidates_retained": False,
        "ordinary_step_selected_by_probes": False,
    }


def run_update_attempt(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    bank: dict[str, Any],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    current = assisted._copy_parameters(controller)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    controller_before_sha256 = assisted.audit.semantic_sha256(current)
    optimizer_before_sha256 = assisted.audit.semantic_sha256(optimizer_before)
    burns = [
        assisted.motion_burn_in(controller, bank, index, device=device)
        for index in range(bank["cases"])
    ]
    current_replays = [
        motion_fixed_report(controller, bank, burns, motion_scale=motion_scale, device=device)
        for _ in range(DERIVATIVE_BASELINE_REPEATS)
    ]
    baseline_objectives = [report["objective"] for report in current_replays]
    current_full = motion_full_prefix_report(
        controller, bank, motion_scale=motion_scale, device=device
    )
    gradient, raw_gradients = motion_gradient(
        controller, bank, burns, motion_scale=motion_scale, device=device
    )
    numerical_reasons = []
    if not all(_reports_finite(report) for report in (*current_replays, current_full)):
        numerical_reasons.append("current fixed or full-prefix objective was nonfinite")
    if not gradient["gradients_finite"] or not _reports_finite(gradient):
        numerical_reasons.append("gradient, recurrence, output or report was nonfinite")

    raw, optimizer_after, displacement = assisted.proposal_from_one_adam_transaction(
        controller, optimizer, current
    )
    counter_before = assisted.optimizer_step_counters(optimizer_before)
    counter_after = assisted.optimizer_step_counters(optimizer_after)
    transaction_pass = bool(
        (not counter_before and counter_after == [1.0, 1.0, 1.0])
        or (
            len(counter_before) == len(counter_after) == 3
            and all(
                after == before + 1.0
                for before, after in zip(counter_before, counter_after, strict=True)
            )
        )
    )
    if not transaction_pass:
        numerical_reasons.append("Adam counters did not advance exactly once")
    proposal_controls, effective = assisted.materialize_scaled_proposal(
        controller, current, displacement, 1.0
    )
    if not proposal_controls["pass"]:
        numerical_reasons.append("full proposal failed bounds or canonical control")
    authoritative_directional = float(
        sum(
            (raw_gradients[name].double() * effective[name].double()).sum()
            for name in PARAMETER_FAMILIES
        )
    )
    if not math.isfinite(authoritative_directional) or authoritative_directional >= 0.0:
        numerical_reasons.append("full proposal was not a finite descent direction")

    derivative = {
        "pass": False,
        "classification": "not_run_due_to_prior_numerical_failure",
        "not_run_due_to_prior_numerical_failure": bool(numerical_reasons),
    }
    if not numerical_reasons:
        derivative = run_derivative_ladder(
            controller,
            current,
            displacement,
            raw_gradients,
            authoritative_directional,
            baseline_objectives,
            bank,
            burns,
            motion_scale=motion_scale,
            device=device,
        )
        if not derivative["pass"]:
            numerical_reasons.append(derivative["classification"])
    assisted._load_parameters(controller, current)

    trials = []
    selected = None
    if not numerical_reasons:
        for scale in BACKTRACK_SCALES:
            controls, _ = assisted.materialize_scaled_proposal(
                controller, current, displacement, scale
            )
            if not controls["pass"]:
                numerical_reasons.append(
                    f"ordinary scale {scale:g} failed bounds or canonical control"
                )
                trials.append(
                    {
                        "scale": scale,
                        "pass": False,
                        "fixed_burn_in": None,
                        "full_prefix": None,
                        "fixed_burn_in_improvement": None,
                        "full_prefix_improvement": None,
                        "controls": controls,
                        "fatal_numerical_failure": True,
                    }
                )
                assisted._load_parameters(controller, current)
                break
            fixed = motion_fixed_report(
                controller, bank, burns, motion_scale=motion_scale, device=device
            )
            full = motion_full_prefix_report(
                controller, bank, motion_scale=motion_scale, device=device
            )
            finite = _reports_finite(fixed, full)
            fixed_improvement = current_replays[0]["objective"] - fixed["objective"]
            full_improvement = current_full["objective"] - full["objective"]
            passed = bool(
                finite
                and fixed_improvement >= MINIMUM_OBJECTIVE_IMPROVEMENT
                and full_improvement >= MINIMUM_OBJECTIVE_IMPROVEMENT
            )
            trial = {
                "scale": scale,
                "pass": passed,
                "fixed_burn_in": fixed,
                "full_prefix": full,
                "fixed_burn_in_improvement": fixed_improvement,
                "full_prefix_improvement": full_improvement,
                "controls": controls,
                "fatal_numerical_failure": not finite,
            }
            trials.append(trial)
            if not finite:
                numerical_reasons.append(
                    f"ordinary scale {scale:g} produced a nonfinite result"
                )
                assisted._load_parameters(controller, current)
                break
            if passed:
                selected = trial
                break
            assisted._load_parameters(controller, current)

    if selected is None:
        assisted.restore_unaccepted_update(controller, optimizer, current, optimizer_before)
    else:
        optimizer.load_state_dict(optimizer_after)
    controller_after_sha256 = assisted.audit.semantic_sha256(
        assisted._copy_parameters(controller)
    )
    optimizer_after_loaded_sha256 = assisted.audit.semantic_sha256(optimizer.state_dict())
    restoration_pass = bool(
        selected is not None
        or (
            controller_after_sha256 == controller_before_sha256
            and optimizer_after_loaded_sha256 == optimizer_before_sha256
        )
    )
    return {
        "accepted": selected is not None,
        "accepted_scale": None if selected is None else selected["scale"],
        "fatal_numerical_failure": bool(numerical_reasons),
        "numerical_failure_reasons": numerical_reasons,
        "current_fixed_burn_in": current_replays[0],
        "current_fixed_burn_in_replay_objectives": baseline_objectives,
        "current_full_prefix": current_full,
        "gradient": gradient,
        "optimizer_transaction": {
            "pass": transaction_pass,
            "counters_before": counter_before,
            "counters_after": counter_after,
            "state_before_sha256": optimizer_before_sha256,
            "state_after_sha256": assisted.audit.semantic_sha256(optimizer_after),
        },
        "raw_parameter_family_rms": {
            name: float((raw[name] - current[name]).square().mean().sqrt())
            for name in PARAMETER_FAMILIES
        },
        "directional_derivative": authoritative_directional,
        "finite_difference": derivative,
        "trials": trials,
        "optimizer_pending_state_exact": bool(
            selected is not None
            and optimizer_after_loaded_sha256 == assisted.audit.semantic_sha256(optimizer_after)
        ),
        "unaccepted_restoration_pass": restoration_pass,
    }


def evaluation_with_source_drift(
    controller: ConnectomeController,
    source_report: dict[str, Any],
    bank: dict[str, Any],
    *,
    motion_scale: float,
    device: torch.device,
) -> dict[str, Any]:
    report = motion_full_prefix_report(
        controller, bank, motion_scale=motion_scale, device=device
    )
    candidate_mean = torch.tensor(report["throttle_pair_means"])
    source_mean = torch.tensor(source_report["throttle_pair_means"])
    drift = candidate_mean - source_mean
    report["source_relative_pair_mean_rms_motor_units"] = float(
        drift.square().mean().sqrt()
    )
    report["source_relative_pair_mean_maximum_motor_units"] = float(drift.abs().max())
    report["source_relative_nrmse_improvement"] = float(
        (source_report["nrmse"] - report["nrmse"]) / source_report["nrmse"]
    )
    report["all_metrics_finite"] = assisted._all_finite_nested(report)
    return report


def milestone_decision(candidate: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if not candidate["all_metrics_finite"] or not candidate[
        "all_recurrent_states_and_outputs_finite"
    ]:
        reasons.append("training motion evaluation was nonfinite")
    if candidate["source_relative_nrmse_improvement"] < MILESTONE_NRMSE_IMPROVEMENT:
        reasons.append("training motion NRMSE improved less than 25% from source")
    if candidate["correct_sign_fraction"] < MILESTONE_SIGN_FRACTION:
        reasons.append("training motion correct-sign fraction was below 50%")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "minimum_source_relative_nrmse_improvement": MILESTONE_NRMSE_IMPROVEMENT,
        "minimum_correct_sign_fraction": MILESTONE_SIGN_FRACTION,
    }


def final_motion_decision(
    candidate: dict[str, Any], *, stage: str
) -> dict[str, Any]:
    reasons = []
    if not candidate["all_metrics_finite"] or not candidate[
        "all_recurrent_states_and_outputs_finite"
    ]:
        reasons.append(f"{stage} recurrence, outputs or metrics were nonfinite")
    if candidate["source_relative_nrmse_improvement"] < FINAL_NRMSE_IMPROVEMENT:
        reasons.append(f"{stage} motion NRMSE improved less than 50% from source")
    if candidate["correct_sign_fraction"] < FINAL_SIGN_FRACTION:
        reasons.append(f"{stage} correct-sign fraction was below 90%")
    low_gain, high_gain = FINAL_GAIN_RANGE
    if not low_gain <= candidate["teacher_aligned_gain"] <= high_gain:
        reasons.append(f"{stage} teacher-aligned gain was outside 0.5-1.5")
    for horizon, report in candidate["by_horizon"].items():
        if report["correct_sign_fraction"] < FINAL_SIGN_FRACTION:
            reasons.append(f"{stage} horizon {horizon} correct-sign fraction was below 90%")
        if not low_gain <= report["teacher_aligned_gain"] <= high_gain:
            reasons.append(f"{stage} horizon {horizon} gain was outside 0.5-1.5")
    if candidate["endpoint_image_difference_max"] != 0.0:
        reasons.append(f"{stage} endpoint images were not exactly equal")
    if candidate["maximum_motor_absolute"] > 1.0:
        reasons.append(f"{stage} native motor output exceeded [-1, 1]")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "minimum_source_relative_nrmse_improvement": FINAL_NRMSE_IMPROVEMENT,
        "minimum_correct_sign_fraction_overall_and_each_horizon": FINAL_SIGN_FRACTION,
        "teacher_aligned_gain_range_overall_and_each_horizon": list(FINAL_GAIN_RANGE),
    }


def _make_optimizer(controller: ConnectomeController) -> torch.optim.Adam:
    original_betas = assisted.OPTIMIZER_BETAS
    try:
        assisted.OPTIMIZER_BETAS = OPTIMIZER_BETAS
        return assisted._make_optimizer(controller)
    finally:
        assisted.OPTIMIZER_BETAS = original_betas


def _resume_payload(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> dict[str, Any]:
    controller_state = {
        name: value.detach().cpu() for name, value in controller.state_dict().items()
    }
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return {
        **state,
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "producer_report_sha256": EXPECTED_PRODUCER_REPORT_SHA256,
        "controller": controller_state,
        "controller_parameter_sha256": assisted.audit.semantic_sha256(
            {name: controller_state[name] for name in PARAMETER_FAMILIES}
        ),
        "optimizer": optimizer_state,
        "optimizer_sha256": assisted.audit.semantic_sha256(optimizer_state),
        "optimizer_step_counters": assisted.optimizer_step_counters(optimizer_state),
    }


def _save_resume(
    path: Path,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> str:
    assisted._atomic_torch_save(_resume_payload(controller, optimizer, state), path)
    return assisted.responsibility.file_sha256(path)


def _validate_resume(payload: dict[str, Any]) -> None:
    expected = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "producer_report_sha256": EXPECTED_PRODUCER_REPORT_SHA256,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise SystemExit(f"resume {name} does not match the registered run")
    parameters = {name: payload["controller"][name] for name in PARAMETER_FAMILIES}
    if assisted.audit.semantic_sha256(parameters) != payload["controller_parameter_sha256"]:
        raise SystemExit("resume controller semantic hash mismatch")
    if assisted.audit.semantic_sha256(payload["optimizer"]) != payload["optimizer_sha256"]:
        raise SystemExit("resume optimizer semantic hash mismatch")
    accepted = int(payload.get("accepted_updates", -1))
    attempted = int(payload.get("attempted_updates", -1))
    if not 0 <= accepted <= attempted <= MAX_ATTEMPTED_UPDATES:
        raise SystemExit("resume counters are invalid")
    pending_terminal = payload.get("pending_terminal")
    if pending_terminal is not None and (
        not isinstance(pending_terminal, dict)
        or set(pending_terminal) != {"classification", "stop_reason", "passed"}
        or not isinstance(pending_terminal["passed"], bool)
    ):
        raise SystemExit("resume pending terminal transition is invalid")
    if payload.get("run_state") == "stopped" and pending_terminal is None:
        raise SystemExit("stopped resume is missing its terminal transition")
    if (
        payload.get("run_state") == "active"
        and accepted != attempted
        and pending_terminal is None
    ):
        raise SystemExit("active resume cannot continue after a rejected fixed-bank attempt")
    counters = assisted.optimizer_step_counters(payload["optimizer"])
    expected_counters = [] if accepted == 0 else [float(accepted)] * 3
    if counters != expected_counters or counters != payload["optimizer_step_counters"]:
        raise SystemExit("resume Adam counters do not match accepted updates")
    if any(
        tuple(group.get("betas", ())) != OPTIMIZER_BETAS
        for group in payload["optimizer"]["param_groups"]
    ):
        raise SystemExit("resume Adam betas do not match the registered run")
    bank = payload.get("training_bank")
    if bank is None or bank.get("seed") != TRAINING_SEED or _payload_hash(bank) != bank.get(
        "sha256"
    ):
        raise SystemExit("resume training bank identity mismatch")
    if motion_bank_manifest(bank) != payload.get("training_bank_manifest"):
        raise SystemExit("resume training bank manifest mismatch")
    scale = payload.get("objective_scale")
    if not isinstance(scale, dict) or scale.get("motion", 0.0) <= 0.0:
        raise SystemExit("resume objective scale is invalid")
    if assisted.audit.semantic_sha256(scale) != payload.get("objective_scale_sha256"):
        raise SystemExit("resume objective scale identity mismatch")
    if assisted.audit.semantic_sha256(payload.get("source_training_report")) != payload.get(
        "source_training_report_sha256"
    ):
        raise SystemExit("resume source training report identity mismatch")
    if assisted.audit.semantic_sha256(payload.get("preflight")) != payload.get(
        "preflight_sha256"
    ):
        raise SystemExit("resume preflight identity mismatch")
    if accepted > MILESTONE_UPDATE and not payload.get("milestone_25", {}).get(
        "decision", {}
    ).get("pass", False):
        raise SystemExit("resume advanced beyond an unpassed update-25 milestone")
    if payload.get("development_started") and accepted != MAX_ACCEPTED_UPDATES:
        raise SystemExit("resume opened development before update 100")
    training_final = payload.get("training_final") or {}
    if payload.get("development_started") and not training_final.get("decision", {}).get(
        "pass", False
    ):
        raise SystemExit("resume opened development before a passing training-final gate")
    if payload.get("development_completed"):
        development = payload.get("development")
        if (
            not isinstance(development, dict)
            or development.get("bank_manifest", {}).get("seed") != DEVELOPMENT_SEED
            or not development.get("bank_manifest", {}).get("exact_full_factorial", False)
        ):
            raise SystemExit("resume development identity is invalid")


def _load_resume(
    path: Path,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_resume(payload)
    controller.load_state_dict(payload["controller"])
    optimizer.load_state_dict(payload["optimizer"])
    excluded = {
        "experiment",
        "protocol_commit",
        "protocol",
        "graph_sha256",
        "checkpoint_sha256",
        "producer_report_sha256",
        "controller",
        "controller_parameter_sha256",
        "optimizer",
        "optimizer_sha256",
    }
    return {name: value for name, value in payload.items() if name not in excluded}


def _terminal_report(
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    *,
    resume_path: Path,
    started: float,
) -> dict[str, Any]:
    source_parameters = assisted._copy_parameters(source)
    current_parameters = assisted._copy_parameters(controller)
    terminal = state["pending_terminal"]
    return {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "status": "nonpromotional_native_visual_motion_learnability",
        "classification": terminal["classification"],
        "pass": terminal["passed"],
        "stop_reason": terminal["stop_reason"],
        "accepted_updates": state["accepted_updates"],
        "attempted_updates": state["attempted_updates"],
        "source": {
            "graph": "data/derived/full-visual-connectome-v1.npz",
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "checkpoint": "runs/visual-hover/paired-dynamic-001/controller.pt",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "preflight": state.get("preflight"),
        "training_bank_manifest": state.get("training_bank_manifest"),
        "objective_scale": state.get("objective_scale"),
        "history": state.get("history", []),
        "milestone_25": state.get("milestone_25"),
        "training_final": state.get("training_final"),
        "development": state.get("development"),
        "development_started": state.get("development_started", False),
        "development_completed": state.get("development_completed", False),
        "optimizer_step_counters": state.get("optimizer_step_counters", []),
        "final_parameter_family_rms_from_source": {
            name: float((current_parameters[name] - source_parameters[name]).square().mean().sqrt())
            for name in PARAMETER_FAMILIES
        },
        "terminal_transition": terminal,
        "run_state": "stopped",
        "resume_state_sha256": assisted.responsibility.file_sha256(resume_path),
        "actor_contract_unchanged": True,
        "external_or_engineered_state_added": False,
        "staged_assisted_curriculum_authorized": terminal["passed"],
        "assisted_hover_authorized": False,
        "native_attitude_reintegration_authorized": False,
        "full_native_hover_authorized": False,
        "gate_flight_authorized": False,
        "hover_controller_promoted": False,
        "interpretation_limit": (
            "A pass establishes isolated native visual-motion learnability only; it is not "
            "a hover, attitude-control, takeoff, gate-flight or promotion result."
        ),
        "elapsed_seconds_this_process": perf_counter() - started,
    }


def _write_checkpoint(
    path: Path, controller: ConnectomeController, state: dict[str, Any]
) -> None:
    assisted._atomic_torch_save(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "graph_sha256": EXPECTED_GRAPH_SHA256,
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "controller": {
                name: value.detach().cpu() for name, value in controller.state_dict().items()
            },
            "accepted_updates": state["accepted_updates"],
            "claim": "isolated native visual-motion learnability only",
        },
        path,
    )


def _stop_and_write(
    args: argparse.Namespace,
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    classification: str,
    stop_reason: str,
    passed: bool,
    started: float,
) -> int:
    terminal = {
        "classification": classification,
        "stop_reason": stop_reason,
        "passed": passed,
    }
    existing = state.get("pending_terminal")
    if existing is not None and existing != terminal:
        raise RuntimeError("pending terminal transition does not match requested stop")
    state["pending_terminal"] = terminal
    state["run_state"] = "stopped"
    state["attempt_stage"] = "idle"
    state["optimizer_step_counters"] = assisted.optimizer_step_counters(
        optimizer.state_dict()
    )
    resume_path = args.output_dir / "resume.pt"
    _save_resume(resume_path, controller, optimizer, state)
    if passed:
        _write_checkpoint(args.output_dir / "motion-controller.pt", controller, state)
    report = _terminal_report(
        state, source, controller, resume_path=resume_path, started=started
    )
    assisted._atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": assisted.responsibility.stable_path(args.output_dir / "report.json"),
                "pass": passed,
                "classification": classification,
                "accepted_updates": state["accepted_updates"],
                "attempted_updates": state["attempted_updates"],
                "staged_assisted_curriculum_authorized": passed,
                "full_native_hover_authorized": False,
                "gate_flight_authorized": False,
                "promoted": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _resolve_pending_terminal(
    args: argparse.Namespace,
    state: dict[str, Any],
    source: ConnectomeController,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    started: float,
) -> int:
    terminal = state.get("pending_terminal")
    if not isinstance(terminal, dict):
        raise RuntimeError("no pending terminal transition")
    return _stop_and_write(
        args,
        state,
        source,
        controller,
        optimizer,
        classification=str(terminal["classification"]),
        stop_reason=str(terminal["stop_reason"]),
        passed=bool(terminal["passed"]),
        started=started,
    )


def _attempt_terminal(result: dict[str, Any]) -> tuple[str, str]:
    if not result["fatal_numerical_failure"]:
        return (
            "first_finite_no_scale_rejection",
            "the fixed full-bank proposal had no admissible ordinary scale",
        )
    derivative_class = result.get("finite_difference", {}).get("classification")
    derivative_reasons = {
        "derivative_probe_numerical_failure": (
            "a derivative probe produced a nonfinite or invalid numerical control"
        ),
        "derivative_probe_noise_limited_inconclusive": (
            "no adjacent derivative probes agreed above the replay-noise floor"
        ),
        "derivative_probe_above_noise_nonconvergence": (
            "adjacent above-noise derivative probes did not achieve local agreement"
        ),
    }
    if derivative_class in derivative_reasons:
        return derivative_class, derivative_reasons[derivative_class]
    return (
        "fatal_numerical_control_failure",
        "a mandatory gradient, transaction, direction, bound or canonical control failed",
    )


def _initial_state(
    source: ConnectomeController,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    bank = build_exact_factorial_motion_bank(
        seed=TRAINING_SEED, held_out_styles=False, device=device, config=config
    )
    bank_manifest = motion_bank_manifest(bank)
    scale = frozen_motion_scale(bank, config=config)
    source_report = motion_full_prefix_report(
        source, bank, motion_scale=scale["motion"], device=device
    )
    sampled_velocity = sampled_endpoint_velocity_report(bank)
    teacher_stick = assisted.teacher_motion_stick_positive_control(
        bank, device=device, config=config
    )
    targets = assisted.teacher_motion_contrasts(bank, config=config)
    target_control = {
        "all_finite": bool(torch.isfinite(targets).all()),
        "all_nonzero": bool((targets.abs() > 1.0e-8).all()),
        "minimum_absolute_motor_contrast": float(targets.abs().min()),
        "maximum_absolute_motor_contrast": float(targets.abs().max()),
    }
    preflight = {
        "actor_has_accelerometer": source.uses_accelerometer,
        "actor_has_proprioception": source.uses_proprioception,
        "training_bank": bank_manifest,
        "sampled_endpoint_velocity": sampled_velocity,
        "teacher_motion_stick_positive_control": teacher_stick,
        "teacher_target_control": target_control,
        "source_training_motion": source_report,
        "objective_scale": scale,
        "endpoint_images_exact": source_report["endpoint_image_difference_max"] == 0.0,
        "source_outputs_within_bounds": source_report["maximum_motor_absolute"] <= 1.0,
    }
    preflight["pass"] = bool(
        not source.uses_accelerometer
        and not source.uses_proprioception
        and bank_manifest["exact_full_factorial"]
        and all(count == 8 for count in bank_manifest["history_length_counts"].values())
        and sampled_velocity["pass"]
        and teacher_stick["pass"]
        and target_control["all_finite"]
        and target_control["all_nonzero"]
        and source_report["all_metrics_finite"]
        and source_report["all_recurrent_states_and_outputs_finite"]
        and preflight["endpoint_images_exact"]
        and preflight["source_outputs_within_bounds"]
        and assisted._all_finite_nested(scale)
    )
    return {
        "run_state": "active",
        "accepted_updates": 0,
        "attempted_updates": 0,
        "history": [],
        "training_bank": bank,
        "training_bank_manifest": bank_manifest,
        "objective_scale": scale,
        "objective_scale_sha256": assisted.audit.semantic_sha256(scale),
        "preflight": preflight,
        "preflight_sha256": assisted.audit.semantic_sha256(preflight),
        "source_training_report": source_report,
        "source_training_report_sha256": assisted.audit.semantic_sha256(source_report),
        "milestone_25": None,
        "milestone_25_started": False,
        "training_final": None,
        "training_final_started": False,
        "development": None,
        "development_started": False,
        "development_completed": False,
        "attempt_stage": "idle",
        "pending_terminal": None,
        "optimizer_step_counters": assisted.optimizer_step_counters(optimizer.state_dict()),
    }


def main() -> int:
    args = parse_args()
    validate_authorizing_report()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded.get("graph_sha256") != EXPECTED_GRAPH_SHA256:
        raise SystemExit("checkpoint graph hash does not match the registered graph")
    source = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    source.load_state_dict(loaded["controller"])
    controller.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    controller.eval()
    optimizer = _make_optimizer(controller)
    config = HoverConfig()
    resume_path = args.output_dir / "resume.pt"

    if resume_path.is_file():
        state = _load_resume(resume_path, controller, optimizer)
        print(
            json.dumps(
                {
                    "stage": "resume_loaded",
                    "accepted_updates": state["accepted_updates"],
                    "attempted_updates": state["attempted_updates"],
                    "attempt_stage": state["attempt_stage"],
                }
            ),
            flush=True,
        )
        if state.get("pending_terminal") is not None:
            return _resolve_pending_terminal(
                args, state, source, controller, optimizer, started=started
            )
        if state.get("run_state") != "active":
            raise SystemExit("motion-only resume is not active")
        interruption = None
        if state.get("attempt_stage") != "idle":
            interruption = (
                "interrupted_attempt_closed",
                "a deterministic full-bank update was interrupted and cannot be retried",
            )
        elif state.get("milestone_25_started") and state.get("milestone_25") is None:
            interruption = (
                "interrupted_milestone_closed",
                "the fixed update-25 evaluation was interrupted",
            )
        elif state.get("training_final_started") and state.get("training_final") is None:
            interruption = (
                "interrupted_training_final_closed",
                "the fixed update-100 training evaluation was interrupted",
            )
        elif state.get("development_started") and not state.get("development_completed"):
            interruption = (
                "interrupted_development_closed",
                "the once-only development evaluation was interrupted",
            )
        if interruption is not None:
            return _stop_and_write(
                args,
                state,
                source,
                controller,
                optimizer,
                classification=interruption[0],
                stop_reason=interruption[1],
                passed=False,
                started=started,
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"stage": "building_exact_training_bank"}), flush=True)
        state = _initial_state(
            source, controller, optimizer, device=device, config=config
        )
        if not state["preflight"]["pass"]:
            return _stop_and_write(
                args,
                state,
                source,
                controller,
                optimizer,
                classification="motion_preflight_failed",
                stop_reason="the exact-bank, trajectory, target, foreleg or source control failed",
                passed=False,
                started=started,
            )
        _save_resume(resume_path, controller, optimizer, state)

    while state["accepted_updates"] < MAX_ACCEPTED_UPDATES:
        if state["attempted_updates"] >= MAX_ATTEMPTED_UPDATES:
            return _stop_and_write(
                args,
                state,
                source,
                controller,
                optimizer,
                classification="attempt_budget_exhausted",
                stop_reason="the fixed 100-attempt budget was exhausted",
                passed=False,
                started=started,
            )
        attempted = state["attempted_updates"] + 1
        target = state["accepted_updates"] + 1
        state["attempt_stage"] = "started"
        _save_resume(resume_path, controller, optimizer, state)
        print(
            json.dumps(
                {
                    "stage": "training_attempt",
                    "attempt": attempted,
                    "target_accepted_update": target,
                }
            ),
            flush=True,
        )
        result = run_update_attempt(
            controller,
            optimizer,
            state["training_bank"],
            motion_scale=state["objective_scale"]["motion"],
            device=device,
        )
        result["attempt"] = attempted
        result["target_accepted_update"] = target
        result["accepted_update"] = target if result["accepted"] else None
        state["attempted_updates"] = attempted
        if result["accepted"]:
            state["accepted_updates"] = target
        state["history"].append(result)
        state["attempt_stage"] = "idle"
        state["optimizer_step_counters"] = assisted.optimizer_step_counters(
            optimizer.state_dict()
        )
        if not result["accepted"]:
            classification, reason = _attempt_terminal(result)
            state["pending_terminal"] = {
                "classification": classification,
                "stop_reason": reason,
                "passed": False,
            }
        if state["accepted_updates"] == MILESTONE_UPDATE:
            state["milestone_25_started"] = True
        if state["accepted_updates"] == MAX_ACCEPTED_UPDATES:
            state["training_final_started"] = True
        _save_resume(resume_path, controller, optimizer, state)
        print(
            json.dumps(
                {
                    "progress": "accepted_update" if result["accepted"] else "rejected_attempt",
                    "attempt": attempted,
                    "accepted_updates": state["accepted_updates"],
                    "scale": result["accepted_scale"],
                    "fatal_numerical_failure": result["fatal_numerical_failure"],
                    "fixed_objective": (
                        result["trials"][-1]["fixed_burn_in"]["objective"]
                        if result["accepted"]
                        else result["current_fixed_burn_in"]["objective"]
                    ),
                }
            ),
            flush=True,
        )
        if state.get("pending_terminal") is not None:
            return _resolve_pending_terminal(
                args, state, source, controller, optimizer, started=started
            )

        if state["accepted_updates"] == MILESTONE_UPDATE:
            print(json.dumps({"stage": "training_milestone_25"}), flush=True)
            candidate = evaluation_with_source_drift(
                controller,
                state["source_training_report"],
                state["training_bank"],
                motion_scale=state["objective_scale"]["motion"],
                device=device,
            )
            state["milestone_25"] = {
                "candidate": candidate,
                "decision": milestone_decision(candidate),
            }
            _save_resume(resume_path, controller, optimizer, state)
            if not state["milestone_25"]["decision"]["pass"]:
                return _stop_and_write(
                    args,
                    state,
                    source,
                    controller,
                    optimizer,
                    classification="training_milestone_25_failed",
                    stop_reason="the fixed motion learnability milestone failed",
                    passed=False,
                    started=started,
                )

    if state["training_final"] is None:
        print(json.dumps({"stage": "training_final_100"}), flush=True)
        training_candidate = evaluation_with_source_drift(
            controller,
            state["source_training_report"],
            state["training_bank"],
            motion_scale=state["objective_scale"]["motion"],
            device=device,
        )
        training_decision = final_motion_decision(training_candidate, stage="training")
        state["training_final"] = {
            "candidate": training_candidate,
            "decision": training_decision,
        }
        _save_resume(resume_path, controller, optimizer, state)
    else:
        training_decision = state["training_final"]["decision"]
    if not training_decision["pass"]:
        return _stop_and_write(
            args,
            state,
            source,
            controller,
            optimizer,
            classification="training_final_100_failed",
            stop_reason="the fixed update-100 training motion gate failed",
            passed=False,
            started=started,
        )

    if state["development_completed"]:
        development_decision = state["development"]["decision"]
        return _stop_and_write(
            args,
            state,
            source,
            controller,
            optimizer,
            classification=(
                "motion_learnability_passed"
                if development_decision["pass"]
                else "development_motion_gate_failed"
            ),
            stop_reason=(
                "training and disjoint native visual-motion learnability gates passed"
                if development_decision["pass"]
                else "the once-only disjoint motion gate failed"
            ),
            passed=bool(development_decision["pass"]),
            started=started,
        )

    state["development_started"] = True
    _save_resume(resume_path, controller, optimizer, state)
    print(json.dumps({"stage": "development_evaluation"}), flush=True)
    development_bank = build_exact_factorial_motion_bank(
        seed=DEVELOPMENT_SEED, held_out_styles=True, device=device, config=config
    )
    development_source = motion_full_prefix_report(
        source,
        development_bank,
        motion_scale=state["objective_scale"]["motion"],
        device=device,
    )
    development_candidate = evaluation_with_source_drift(
        controller,
        development_source,
        development_bank,
        motion_scale=state["objective_scale"]["motion"],
        device=device,
    )
    development_decision = final_motion_decision(
        development_candidate, stage="development"
    )
    state["development"] = {
        "bank_manifest": motion_bank_manifest(development_bank),
        "source": development_source,
        "candidate": development_candidate,
        "decision": development_decision,
    }
    state["development_completed"] = True
    _save_resume(resume_path, controller, optimizer, state)
    return _stop_and_write(
        args,
        state,
        source,
        controller,
        optimizer,
        classification=(
            "motion_learnability_passed"
            if development_decision["pass"]
            else "development_motion_gate_failed"
        ),
        stop_reason=(
            "training and disjoint native visual-motion learnability gates passed"
            if development_decision["pass"]
            else "the once-only disjoint motion gate failed"
        ),
        passed=bool(development_decision["pass"]),
        started=started,
    )


if __name__ == "__main__":
    raise SystemExit(main())
