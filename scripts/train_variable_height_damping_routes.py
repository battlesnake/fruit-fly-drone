#!/usr/bin/env python3
"""Train the preregistered shallow native route toward visual vertical damping."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_damping_routes as v1  # noqa: E402
import audit_variable_height_damping_routes_v2 as v2  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

ATTEMPT_LIMIT = 50
MILESTONE_ATTEMPT = 25
MILESTONE_IMPROVEMENT_FRACTION = 0.25
CONSECUTIVE_REJECTION_LIMIT = 5
FINAL_SIGN_ACCURACY = 0.90
FINAL_GAIN_RANGE = (0.50, 1.50)
MILESTONE_SEED_OFFSET = 60_000
TERMINAL_SEED_OFFSET = 70_000


def parse_args() -> argparse.Namespace:
    parser = v1.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=REPO_ROOT / "runs/variable-height-hover/damping-route-train-001",
        seed=270_929,
    )
    parser.add_argument("--attempts", type=int, default=ATTEMPT_LIMIT)
    parser.add_argument("--milestone-attempt", type=int, default=MILESTONE_ATTEMPT)
    parser.add_argument("--gradient-banks", type=int, default=2)
    parser.add_argument("--guard-banks", type=int, default=2)
    parser.add_argument("--evaluation-banks", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    v1.validate_args(args)
    if (
        min(
            args.attempts,
            args.milestone_attempt,
            args.gradient_banks,
            args.guard_banks,
            args.evaluation_banks,
        )
        < 1
    ):
        raise SystemExit("training counts must be positive")
    if args.milestone_attempt > args.attempts:
        raise SystemExit("milestone attempt cannot exceed attempt limit")
    if not args.smoke_test and (
        args.attempts != ATTEMPT_LIMIT or args.milestone_attempt != MILESTONE_ATTEMPT
    ):
        raise SystemExit("v1 is preregistered for 50 attempts and its gate at attempt 25")


def numeric_tree_is_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(numeric_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_tree_is_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def training_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attempt_limit": args.attempts,
        "milestone_attempt": args.milestone_attempt,
        "gradient_banks": args.gradient_banks,
        "guard_banks": args.guard_banks,
        "evaluation_banks": args.evaluation_banks,
        "batch_size": args.batch_size,
        "gradient_pairs_per_attempt": args.gradient_banks * args.batch_size,
        "guard_pairs_per_trial": args.guard_banks * args.batch_size,
        "held_out_pairs_per_evaluation": args.evaluation_banks * args.batch_size,
        "constraint_batch_size": args.constraint_batch_size,
        "unroll": args.unroll,
        "constraint_prefix_steps": args.constraint_prefix_steps,
        "history_steps": args.history_steps,
        "policy_hz": args.policy_hz,
        "device": args.device,
        "smoke_test": args.smoke_test,
        "seed": args.seed,
        "gradient_seed_formula": "seed + attempt * 100 + bank",
        "direction_constraint_seed_formula": "seed + attempt * 100 + internal fixed offsets",
        "guard_seed_offset": 40_000,
        "milestone_seed_offset": MILESTONE_SEED_OFFSET,
        "terminal_seed_offset": TERMINAL_SEED_OFFSET,
        "consecutive_rejection_limit": CONSECUTIVE_REJECTION_LIMIT,
        "selected_family_step_rms_cap": v1.MASK_STEP_FAMILY_RMS_CAP,
        "selected_source_metric_radius": v1.MASK_SOURCE_METRIC_RADIUS,
        "candidate_blend_fractions": list(v2.BLEND_FRACTIONS),
        "backtrack_scales": list(v1.BACKTRACK_SCALES),
        "edge_magnitude_bounds": [0.0, 8.0],
        "per_update_common_throttle_rms": v2.PER_UPDATE_COMMON_RMS,
        "source_common_throttle_rms": v2.SOURCE_COMMON_RMS,
        "source_common_throttle_max_absolute": v2.COMMON_MAX_ABSOLUTE,
        "height_contrast_source_ratio_range": list(v1.HEIGHT_CONTRAST_RATIO_RANGE),
        "minimum_per_guard_bank_motion_nrmse_improvement": (v1.MINIMUM_MOTION_NRMSE_IMPROVEMENT),
        "milestone_motion_nrmse_improvement_fraction": MILESTONE_IMPROVEMENT_FRACTION,
        "final_sign_accuracy": FINAL_SIGN_ACCURACY,
        "final_teacher_aligned_gain_range": list(FINAL_GAIN_RANGE),
    }


def summarize_motion_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    target = torch.tensor([value for metric in metrics for value in metric["target_contrast"]])
    prediction = torch.tensor(
        [value for metric in metrics for value in metric["predicted_contrast"]]
    )
    error = prediction - target
    summary = {
        "pairs": len(target),
        "fixed_scale_nrmse": float((error / v1.MOTION_MOTOR_SCALE).square().mean().sqrt()),
        "relative_target_nrmse": float(
            error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1.0e-12)
        ),
        "teacher_aligned_gain": float(
            (prediction * target).sum() / target.square().sum().clamp_min(1.0e-12)
        ),
        "correct_sign_fraction": float(((prediction * target) > 1.0e-6).float().mean()),
        "target_rms": float(target.square().mean().sqrt()),
        "prediction_rms": float(prediction.square().mean().sqrt()),
    }
    summary["all_finite"] = numeric_tree_is_finite(summary)
    return summary


@torch.no_grad()
def evaluate_motion_banks(
    controller: ConnectomeController,
    *,
    banks: int,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> dict[str, Any]:
    results = []
    for bank in range(banks):
        _, metrics = v1.motion_terms(
            controller,
            controller,
            batch=batch,
            seed=seed + bank,
            history_steps=history_steps,
            policy_hz=policy_hz,
            config=config,
        )
        results.append(metrics)
    return {**summarize_motion_metrics(results), "banks": results}


def guard_bank_improvements(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[float]:
    if len(baseline["banks"]) != len(candidate["banks"]):
        raise ValueError("guard evaluations must contain the same number of banks")
    return [
        before["fixed_scale_nrmse"] - after["fixed_scale_nrmse"]
        for before, after in zip(baseline["banks"], candidate["banks"], strict=True)
    ]


def accumulated_motion_gradient(
    student: ConnectomeController,
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    banks: int,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    accumulated = {
        "edge_magnitude": torch.zeros(len(edge_indices), device=edge_indices.device),
        "bias": torch.zeros(len(bias_indices), device=bias_indices.device),
    }
    metrics = []
    for bank in range(banks):
        loss, report = v1.motion_terms(
            student,
            student,
            batch=batch,
            seed=seed + bank,
            history_steps=history_steps,
            policy_hz=policy_hz,
            config=config,
        )
        gradient = v1.masked_gradient(loss, student, edge_indices, bias_indices)
        for name in v1.MASK_FAMILIES:
            accumulated[name].add_(gradient[name], alpha=1.0 / banks)
        metrics.append(report)
    return accumulated, summarize_motion_metrics(metrics)


def direction_candidates(
    student: ConnectomeController,
    source: ConnectomeController,
    gradient: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    direction_args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> tuple[
    str | None,
    dict[str, dict[str, Tensor]],
    dict[str, Any],
]:
    riesz = {name: -gradient[name] * gradient[name].numel() for name in v1.MASK_FAMILIES}
    raw, raw_scale = v1.cap_masked_displacement(riesz, v1.MASK_STEP_FAMILY_RMS_CAP)
    rows, residual, common_report = v1.common_constraint_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        args=direction_args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    equality_small, projection = v1.project_masked_displacement(raw, rows, residual)
    equality, equality_scale = v2.scale_to_final_cap(equality_small, v1.MASK_STEP_FAMILY_RMS_CAP)
    current_edges = student.edge_magnitude[edge_indices].detach()
    candidates = {}
    bounds = {}
    for fraction in v2.BLEND_FRACTIONS:
        label = f"equality_to_raw_{fraction:.2f}"
        mixed = {
            name: (1.0 - fraction) * equality[name] + fraction * raw[name]
            for name in v1.MASK_FAMILIES
        }
        mixed, _ = v2.scale_to_final_cap(mixed, v1.MASK_STEP_FAMILY_RMS_CAP)
        candidates[label], bounds[label] = v2.apply_edge_bounds(mixed, current_edges)
    selected, screening = v2.screen_candidates(candidates, gradient, rows, residual)
    for label in screening:
        screening[label]["parameter_bounds"] = bounds[label]
    return (
        selected,
        candidates,
        {
            "gradient_bank": None,
            "raw_initial_cap_scale": raw_scale,
            "equality_rescale_to_final_cap": equality_scale,
            "common_constraints": common_report,
            "equality_projection": projection,
            "candidate_screening": screening,
        },
    )


def source_and_step_deltas(
    student: ConnectomeController,
    source_parameters: dict[str, Tensor],
    previous_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    source = v1.masked_source_delta(student, source_parameters, edge_indices, bias_indices)
    step = {
        "edge_magnitude": student.edge_magnitude[edge_indices]
        - previous_parameters["edge_magnitude"][edge_indices],
        "bias": student.bias[bias_indices] - previous_parameters["bias"][bias_indices],
    }
    return source, step


@torch.no_grad()
def evaluate_training_trial(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    source_parameters: dict[str, Tensor],
    previous_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    baseline_guard: dict[str, Any],
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    motion = evaluate_motion_banks(
        student,
        banks=args.guard_banks,
        batch=args.batch_size,
        seed=args.seed + 40_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    improvement = baseline_guard["fixed_scale_nrmse"] - motion["fixed_scale_nrmse"]
    bank_improvements = guard_bank_improvements(baseline_guard, motion)
    height = v1.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = trust.functional_trust_checks(
        student,
        source,
        previous,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        source_parameters=source_parameters,
        previous_parameters=previous_parameters,
    )
    source_delta, step_delta = source_and_step_deltas(
        student,
        source_parameters,
        previous_parameters,
        edge_indices,
        bias_indices,
    )
    source_common_rms = max(
        value["common_error_rms"] for value in functional["source_pair"].values()
    )
    source_common_max = max(
        value["common_error_max_absolute"] for value in functional["source_pair"].values()
    )
    step_common_rms = max(
        value["common_error_rms"] for value in functional["per_update_pair"].values()
    )
    height_ok = all(
        v1.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= v1.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    all_finite = all(
        numeric_tree_is_finite(value)
        for value in (
            baseline_guard,
            motion,
            height,
            functional,
            source_delta,
            step_delta,
        )
    )
    reasons = []
    if not all_finite:
        reasons.append("nonfinite acceptance metric or selected parameter displacement")
    if any(value < v1.MINIMUM_MOTION_NRMSE_IMPROVEMENT for value in bank_improvements):
        reasons.append("one or more fixed guard banks improved by less than 1e-4 NRMSE")
    if not functional["pass"]:
        reasons.append("existing complete-replay trust checks failed")
    if step_common_rms > v2.PER_UPDATE_COMMON_RMS:
        reasons.append("per-update common throttle RMS")
    if source_common_rms > v2.SOURCE_COMMON_RMS:
        reasons.append("source-global common throttle RMS")
    if source_common_max > v2.COMMON_MAX_ABSOLUTE:
        reasons.append("source-global common throttle maximum")
    if not height_ok:
        reasons.append("height contrast left source-relative 10% band")
    if max(v1.masked_family_rms(step_delta).values()) > v1.MASK_STEP_FAMILY_RMS_CAP * (
        1.0 + 1.0e-5
    ):
        reasons.append("selected-family step RMS cap")
    if v1.masked_metric_norm(source_delta) > v1.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("selected-family source metric radius")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "motion": motion,
        "motion_nrmse_improvement": improvement,
        "motion_nrmse_improvement_by_guard_bank": bank_improvements,
        "all_finite": all_finite,
        "height_response": height,
        "functional_trust": functional,
        "per_update_common_throttle_rms": step_common_rms,
        "source_common_throttle_rms": source_common_rms,
        "source_common_throttle_max_absolute": source_common_max,
        "selected_step_family_rms": v1.masked_family_rms(step_delta),
        "selected_source_family_rms": v1.masked_family_rms(source_delta),
        "selected_source_metric_norm": v1.masked_metric_norm(source_delta),
    }


@torch.no_grad()
def preservation_snapshot(
    student: ConnectomeController,
    source: ConnectomeController,
    source_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    current_parameters = trust.clone_parameters(student)
    functional = trust.functional_trust_checks(
        student,
        source,
        student,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
        source_parameters=source_parameters,
        previous_parameters=current_parameters,
    )
    height = v1.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    source_delta = v1.masked_source_delta(student, source_parameters, edge_indices, bias_indices)
    source_common_rms = max(
        value["common_error_rms"] for value in functional["source_pair"].values()
    )
    source_common_max = max(
        value["common_error_max_absolute"] for value in functional["source_pair"].values()
    )
    height_ok = all(
        v1.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= v1.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    all_finite = all(numeric_tree_is_finite(value) for value in (functional, height, source_delta))
    pass_gate = (
        all_finite
        and functional["pass"]
        and height_ok
        and source_common_rms <= v2.SOURCE_COMMON_RMS
        and source_common_max <= v2.COMMON_MAX_ABSOLUTE
        and v1.masked_metric_norm(source_delta) <= v1.MASK_SOURCE_METRIC_RADIUS * 1.00001
    )
    return {
        "pass": pass_gate,
        "all_finite": all_finite,
        "functional_trust": functional,
        "height_response": height,
        "source_common_throttle_rms": source_common_rms,
        "source_common_throttle_max_absolute": source_common_max,
        "selected_source_family_rms": v1.masked_family_rms(source_delta),
        "selected_source_metric_norm": v1.masked_metric_norm(source_delta),
    }


def evaluation_decision(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    preservation: dict[str, Any],
    *,
    milestone: bool,
) -> dict[str, Any]:
    all_finite = numeric_tree_is_finite(candidate) and numeric_tree_is_finite(preservation)
    baseline_scale = baseline["fixed_scale_nrmse"]
    all_finite = (
        all_finite
        and numeric_tree_is_finite(baseline)
        and math.isfinite(baseline_scale)
        and baseline_scale > 0.0
    )
    improvement = (
        1.0 - candidate["fixed_scale_nrmse"] / baseline_scale if all_finite else float("nan")
    )
    replay_pass = (
        all_finite
        and candidate["correct_sign_fraction"] >= FINAL_SIGN_ACCURACY
        and FINAL_GAIN_RANGE[0] <= candidate["teacher_aligned_gain"] <= FINAL_GAIN_RANGE[1]
        and preservation["pass"]
    )
    meaningful = all_finite and ((not milestone) or improvement >= MILESTONE_IMPROVEMENT_FRACTION)
    return {
        "pass": replay_pass and meaningful,
        "all_finite": all_finite,
        "motion_nrmse_improvement_fraction": improvement,
        "meaningful_progress_gate_passed": meaningful,
        "fresh_motion_replay_gate_passed": replay_pass,
        "requirements": {
            "attempt_25_motion_nrmse_improvement_fraction": (
                MILESTONE_IMPROVEMENT_FRACTION if milestone else None
            ),
            "correct_sign_fraction": FINAL_SIGN_ACCURACY,
            "teacher_aligned_gain_range": list(FINAL_GAIN_RANGE),
            "preservation": True,
        },
    }


def checkpoint_payload(
    controller: ConnectomeController,
    *,
    args: argparse.Namespace,
    source_sha256: str,
    graph_sha256: str,
    mask_manifest: dict[str, Any],
    attempted: int,
    accepted: int,
    consecutive_rejections: int,
    history: list[dict[str, Any]],
    stop_reason: str | None,
    milestone_report: dict[str, Any] | None,
    terminal: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 2,
        "experiment": "variable-height-native-damping-route-training-v1",
        "non_promotional": True,
        "controller": controller.state_dict(),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "graph_sha256": graph_sha256,
        "edge_indices_sha256": mask_manifest["edge_indices_sha256"],
        "bias_node_indices_sha256": mask_manifest["bias_node_indices_sha256"],
        "attempted": attempted,
        "accepted": accepted,
        "consecutive_rejections": consecutive_rejections,
        "history": history,
        "protocol": training_protocol(args),
        "stop_reason": stop_reason,
        "milestone": milestone_report,
        "terminal": terminal,
    }


def save_progress(
    controller: ConnectomeController,
    *,
    args: argparse.Namespace,
    source_sha256: str,
    graph_sha256: str,
    mask_manifest: dict[str, Any],
    attempted: int,
    accepted: int,
    consecutive_rejections: int,
    history: list[dict[str, Any]],
    stop_reason: str | None,
    milestone_report: dict[str, Any] | None,
    terminal: bool,
) -> None:
    payload = checkpoint_payload(
        controller,
        args=args,
        source_sha256=source_sha256,
        graph_sha256=graph_sha256,
        mask_manifest=mask_manifest,
        attempted=attempted,
        accepted=accepted,
        consecutive_rejections=consecutive_rejections,
        history=history,
        stop_reason=stop_reason,
        milestone_report=milestone_report,
        terminal=terminal,
    )
    state_path = args.output_dir / "training-state.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(state_path)


def main() -> int:
    args = parse_args()
    validate_training_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = HoverConfig()
    physics_hz = round(1.0 / config.dt)
    if physics_hz % args.policy_hz:
        raise SystemExit("policy frequency must divide physics frequency")
    physics_steps = physics_hz // args.policy_hz
    graph_sha256 = responsibility.file_sha256(args.graph)
    source_sha256 = responsibility.file_sha256(args.checkpoint)
    loaded = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if loaded["graph_sha256"] != graph_sha256:
        raise SystemExit("checkpoint and graph hashes do not match")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    previous.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edge_indices_np, bias_indices_np, mask_manifest = v1.build_route_mask(args.graph, args.raw_dir)
    edge_indices = torch.from_numpy(edge_indices_np).to(device)
    bias_indices = torch.from_numpy(bias_indices_np).to(device)
    source_parameters = trust.clone_parameters(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "training-state.pt"
    protocol = training_protocol(args)
    attempted = accepted = consecutive_rejections = 0
    history: list[dict[str, Any]] = []
    stop_reason = None
    milestone_report = None
    resume_terminal = False
    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"resume state does not exist: {state_path}")
        resume = torch.load(state_path, map_location=device, weights_only=False)
        expected = (
            resume.get("checkpoint_schema_version") == 2
            and resume.get("source_checkpoint_sha256") == source_sha256
            and resume.get("graph_sha256") == graph_sha256
            and resume.get("edge_indices_sha256") == mask_manifest["edge_indices_sha256"]
            and resume.get("bias_node_indices_sha256") == mask_manifest["bias_node_indices_sha256"]
            and resume.get("protocol") == protocol
        )
        if not expected:
            raise SystemExit("resume state does not match source graph/checkpoint/mask/protocol")
        student.load_state_dict(resume["controller"])
        attempted = int(resume["attempted"])
        accepted = int(resume["accepted"])
        consecutive_rejections = int(resume["consecutive_rejections"])
        history = list(resume["history"])
        stop_reason = resume.get("stop_reason")
        milestone_report = resume.get("milestone")
        resume_terminal = bool(resume.get("terminal"))

    started = perf_counter()
    milestone_seed = args.seed + MILESTONE_SEED_OFFSET
    terminal_seed = args.seed + TERMINAL_SEED_OFFSET
    baseline_milestone_evaluation = evaluate_motion_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=milestone_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    baseline_terminal_evaluation = evaluate_motion_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=terminal_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    final_report = None
    attempts = () if resume_terminal else range(attempted + 1, args.attempts + 1)
    for attempt in attempts:
        attempted = attempt
        base_parameters = trust.clone_parameters(student)
        previous.load_state_dict(student.state_dict())
        baseline_guard = evaluate_motion_banks(
            student,
            banks=args.guard_banks,
            batch=args.batch_size,
            seed=args.seed + 40_000,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        gradient, gradient_bank = accumulated_motion_gradient(
            student,
            edge_indices,
            bias_indices,
            banks=args.gradient_banks,
            batch=args.batch_size,
            seed=args.seed + attempt * 100,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        direction_args = SimpleNamespace(**vars(args))
        direction_args.seed = args.seed + attempt * 100
        selected, candidates, direction = direction_candidates(
            student,
            source,
            gradient,
            edge_indices,
            bias_indices,
            direction_args=direction_args,
            physics_steps=physics_steps,
            device=device,
            config=config,
        )
        direction["gradient_bank"] = gradient_bank
        trials = []
        accepted_scale = None
        if selected is not None:
            derivative = direction["candidate_screening"][selected]["first_order_loss_derivative"]
            for scale in v1.BACKTRACK_SCALES:
                v1.set_masked_trial(
                    student,
                    base_parameters,
                    edge_indices,
                    bias_indices,
                    candidates[selected],
                    scale,
                )
                trial = evaluate_training_trial(
                    student,
                    source,
                    previous,
                    source_parameters,
                    base_parameters,
                    edge_indices,
                    bias_indices,
                    baseline_guard=baseline_guard,
                    args=args,
                    physics_steps=physics_steps,
                    device=device,
                    config=config,
                )
                trial.update(
                    {
                        "scale": scale,
                        "predicted_first_order_training_loss_change": scale * derivative,
                    }
                )
                trials.append(trial)
                if trial["pass"]:
                    accepted_scale = scale
                    break
        if accepted_scale is None:
            trust.load_parameters(student, base_parameters)
            consecutive_rejections += 1
        else:
            accepted += 1
            consecutive_rejections = 0
        entry = {
            "attempt": attempt,
            "accepted": accepted_scale is not None,
            "accepted_scale": accepted_scale,
            "consecutive_rejections": consecutive_rejections,
            "baseline_guard_motion": baseline_guard,
            "direction": direction,
            "selected_candidate": selected,
            "trials": trials,
        }
        history.append(entry)
        if consecutive_rejections >= CONSECUTIVE_REJECTION_LIMIT:
            stop_reason = "five consecutive unsafe or non-improving route steps"
        elif args.smoke_test:
            stop_reason = "smoke test ended after one complete attempt"
        elif attempt == args.milestone_attempt:
            candidate = evaluate_motion_banks(
                student,
                banks=args.evaluation_banks,
                batch=args.batch_size,
                seed=milestone_seed,
                history_steps=args.history_steps,
                policy_hz=args.policy_hz,
                config=config,
            )
            preservation = preservation_snapshot(
                student,
                source,
                source_parameters,
                edge_indices,
                bias_indices,
                args=args,
                physics_steps=physics_steps,
                device=device,
                config=config,
            )
            decision = evaluation_decision(
                candidate, baseline_milestone_evaluation, preservation, milestone=True
            )
            milestone_report = {
                "attempt": attempt,
                "candidate": candidate,
                "preservation": preservation,
                "decision": decision,
            }
            if not decision["meaningful_progress_gate_passed"]:
                stop_reason = "attempt-25 motion NRMSE improvement below 25%"
        terminal = stop_reason is not None
        save_progress(
            student,
            args=args,
            source_sha256=source_sha256,
            graph_sha256=graph_sha256,
            mask_manifest=mask_manifest,
            attempted=attempted,
            accepted=accepted,
            consecutive_rejections=consecutive_rejections,
            history=history,
            stop_reason=stop_reason,
            milestone_report=milestone_report,
            terminal=terminal,
        )
        print(
            json.dumps(
                {
                    "progress": "attempt",
                    "attempt": attempt,
                    "accepted": entry["accepted"],
                    "accepted_scale": accepted_scale,
                    "selected_candidate": selected,
                    "consecutive_rejections": consecutive_rejections,
                    "motion_before": baseline_guard["fixed_scale_nrmse"],
                    "motion_after": (trials[-1]["motion"]["fixed_scale_nrmse"] if trials else None),
                    "terminal": terminal,
                    "stop_reason": stop_reason,
                }
            ),
            flush=True,
        )
        if terminal:
            break

    candidate = evaluate_motion_banks(
        student,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=terminal_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    preservation = preservation_snapshot(
        student,
        source,
        source_parameters,
        edge_indices,
        bias_indices,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    decision = evaluation_decision(
        candidate,
        baseline_terminal_evaluation,
        preservation,
        milestone=False,
    )
    milestone_gate_passed = (
        milestone_report is None or milestone_report["decision"]["meaningful_progress_gate_passed"]
    )
    decision["milestone_gate_passed"] = milestone_gate_passed
    decision["pass"] = decision["pass"] and milestone_gate_passed
    final_report = {
        "attempt": attempted,
        "candidate": candidate,
        "preservation": preservation,
        "decision": decision,
    }
    if stop_reason is None:
        stop_reason = (
            "fresh motion replay gate passed"
            if decision["pass"]
            else "attempt limit reached without fresh motion replay pass"
        )
    endpoint = args.output_dir / "nonpromotional-endpoint.pt"
    torch.save(
        checkpoint_payload(
            student,
            args=args,
            source_sha256=source_sha256,
            graph_sha256=graph_sha256,
            mask_manifest=mask_manifest,
            attempted=attempted,
            accepted=accepted,
            consecutive_rejections=consecutive_rejections,
            history=[],
            stop_reason=stop_reason,
            milestone_report=milestone_report,
            terminal=True,
        ),
        endpoint,
    )
    report = {
        "experiment": "variable-height-native-damping-route-training-v1",
        "passed_replay_gate": decision["pass"],
        "promoted": False,
        "claim": "restricted native damping replay only; no closed-loop hover promotion",
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": source_sha256,
        },
        "mask": mask_manifest,
        "protocol": protocol,
        "attempts_completed": attempted,
        "accepted_updates": accepted,
        "consecutive_rejections": consecutive_rejections,
        "stop_reason": stop_reason,
        "baseline_evaluations": {
            "milestone": baseline_milestone_evaluation,
            "terminal": baseline_terminal_evaluation,
        },
        "history": history,
        "milestone": milestone_report,
        "final": final_report,
        "nonpromotional_endpoint": responsibility.stable_path(endpoint),
        "elapsed_seconds": perf_counter() - started,
        "resume_supported": True,
    }
    output = args.output_dir / "report.json"
    output.write_text(f"{json.dumps(report, indent=2, sort_keys=True)}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": responsibility.stable_path(output),
                "passed_replay_gate": report["passed_replay_gate"],
                "attempts_completed": attempted,
                "accepted_updates": accepted,
                "stop_reason": stop_reason,
                "baseline_motion_nrmse": baseline_terminal_evaluation["fixed_scale_nrmse"],
                "final_motion_nrmse": candidate["fixed_scale_nrmse"],
                "final_gain": candidate["teacher_aligned_gain"],
                "final_sign_fraction": candidate["correct_sign_fraction"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
