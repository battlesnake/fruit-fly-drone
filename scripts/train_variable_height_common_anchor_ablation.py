#!/usr/bin/env python3
"""Ablate absolute collective anchoring in expanded native damping training."""

from __future__ import annotations

import argparse
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

import audit_variable_height_damping_routes as shallow  # noqa: E402
import audit_variable_height_damping_routes_v2 as bounded  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import audit_variable_height_upstream_damping_route as upstream  # noqa: E402
import train_variable_height_bridge as bridge  # noqa: E402
import train_variable_height_damping_routes as native_train  # noqa: E402
import train_variable_height_hover as hover  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402
import train_variable_height_upstream_damping_route as upstream_train  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

ATTEMPT_LIMIT = 25
BENCHMARK_SEED_OFFSET = native_train.TERMINAL_SEED_OFFSET
FRESH_QUALIFICATION_SEED_OFFSET = 80_000
USEFUL_PROGRESS_FRACTION = 0.25
MOTOR_BOUND_TOLERANCE = 1.0e-6


def parse_args() -> argparse.Namespace:
    parser = shallow.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=(REPO_ROOT / "runs/variable-height-hover/common-anchor-ablation-001"),
        seed=290_941,
    )
    parser.add_argument("--attempts", type=int, default=ATTEMPT_LIMIT)
    parser.add_argument("--gradient-banks", type=int, default=2)
    parser.add_argument("--guard-banks", type=int, default=2)
    parser.add_argument("--evaluation-banks", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    shallow.validate_args(args)
    if min(args.attempts, args.gradient_banks, args.guard_banks, args.evaluation_banks) < 1:
        raise SystemExit("training counts must be positive")
    if not args.smoke_test and args.attempts != ATTEMPT_LIMIT:
        raise SystemExit("the common-anchor ablation is preregistered for 25 attempts")


def legacy_rpy_nrmse_with_original_denominator(legacy: dict[str, float]) -> float:
    return math.sqrt(
        sum(legacy[f"{axis}_source_nrmse"] ** 2 for axis in ("roll", "pitch", "yaw")) / 4.0
    )


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attempt_limit": args.attempts,
        "gradient_banks": args.gradient_banks,
        "guard_banks": args.guard_banks,
        "evaluation_banks": args.evaluation_banks,
        "batch_size": args.batch_size,
        "gradient_pairs_per_attempt": args.gradient_banks * args.batch_size,
        "guard_pairs_per_trial": args.guard_banks * args.batch_size,
        "benchmark_pairs": args.evaluation_banks * args.batch_size,
        "fresh_qualification_pairs": args.evaluation_banks * args.batch_size,
        "constraint_batch_size": args.constraint_batch_size,
        "unroll": args.unroll,
        "constraint_prefix_steps": args.constraint_prefix_steps,
        "history_steps": args.history_steps,
        "policy_hz": args.policy_hz,
        "device": args.device,
        "smoke_test": args.smoke_test,
        "seed": args.seed,
        "gradient_seed_formula": "seed + attempt * 100 + bank",
        "guard_seed_offset": 40_000,
        "reused_benchmark_seed_offset": BENCHMARK_SEED_OFFSET,
        "fresh_qualification_seed_offset": FRESH_QUALIFICATION_SEED_OFFSET,
        "consecutive_rejection_limit": native_train.CONSECUTIVE_REJECTION_LIMIT,
        "reference_metric_denominators": dict(upstream.REFERENCE_DENOMINATORS),
        "reference_metric_governs": [
            "Riesz descent direction",
            "step family caps",
            "source metric radius",
        ],
        "common_output_projection_or_screening": False,
        "absolute_throttle_source_gates": False,
        "absolute_teacher_throttle_objective_or_gate": False,
        "paired_rpy": "diagnostic only, matching the completed anchored run",
        "legacy_rpy_gate": ("sqrt((roll_nrmse^2 + pitch_nrmse^2 + yaw_nrmse^2) / 4) <= 0.05"),
        "dynamic_rpy_each_nrmse_limit": trust.SOURCE_FUNCTION_NRMSE_LIMIT,
        "selected_family_step_rms_cap": shallow.MASK_STEP_FAMILY_RMS_CAP,
        "selected_source_metric_radius": shallow.MASK_SOURCE_METRIC_RADIUS,
        "backtrack_scales": list(shallow.BACKTRACK_SCALES),
        "edge_magnitude_bounds": [0.0, 8.0],
        "height_contrast_source_ratio_range": list(shallow.HEIGHT_CONTRAST_RATIO_RANGE),
        "minimum_per_guard_bank_motion_nrmse_improvement": (
            shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT
        ),
        "useful_progress_motion_nrmse_improvement_fraction": USEFUL_PROGRESS_FRACTION,
        "fresh_correct_sign_accuracy": native_train.FINAL_SIGN_ACCURACY,
        "fresh_teacher_aligned_gain_range": list(native_train.FINAL_GAIN_RANGE),
        "motor_output_absolute_bound": 1.0,
    }


def raw_direction(
    student: ConnectomeController,
    gradient: dict[str, Tensor],
    edge_indices: Tensor,
) -> tuple[dict[str, Tensor] | None, dict[str, Any]]:
    direction, cap_scale = upstream.scale_to_reference_cap(
        upstream_train.fixed_metric_riesz(gradient), shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    direction, bounds = bounded.apply_edge_bounds(
        direction, student.edge_magnitude[edge_indices].detach()
    )
    derivative = float(
        sum((gradient[name] * direction[name]).sum() for name in shallow.MASK_FAMILIES)
    )
    finite = native_train.numeric_tree_is_finite([direction, derivative])
    admissible = finite and derivative < 0.0
    return (
        direction if admissible else None,
        {
            "candidate": "raw_no_common_anchor",
            "admissible": admissible,
            "all_finite": finite,
            "first_order_loss_derivative": derivative,
            "initial_cap_scale": cap_scale,
            "reference_family_rms": upstream.reference_family_rms(direction),
            "parameter_bounds": bounds,
        },
    )


@torch.no_grad()
def functional_ablation_checks(
    student: ConnectomeController,
    source: ConnectomeController,
    previous: ConnectomeController,
    *,
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    source_pair = {}
    per_update_pair = {}
    for amplitude_index, (amplitude, half_step) in enumerate(bridge.PAIR_HALF_STEPS.items()):
        seed = args.seed + 30_000 + amplitude_index
        hover.seed_everything(seed)
        _, source_pair[amplitude] = bridge._teacher_pair_loss(
            student,
            source,
            half_step=half_step,
            batch=args.constraint_batch_size,
            unroll=args.unroll,
            prefix_steps=args.constraint_prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )
        hover.seed_everything(seed)
        _, per_update_pair[amplitude] = bridge._teacher_pair_loss(
            student,
            previous,
            half_step=half_step,
            batch=args.constraint_batch_size,
            unroll=args.unroll,
            prefix_steps=args.constraint_prefix_steps,
            physics_steps=physics_steps,
            device=device,
            config=config,
            all_style_combinations=True,
        )
    hover.seed_everything(args.seed + 30_010)
    _, legacy = bridge._source_cross_band_loss(
        student,
        source,
        batch=args.constraint_batch_size,
        unroll=args.unroll,
        prefix_steps=args.constraint_prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    hover.seed_everything(args.seed + 30_020)
    _, dynamic = bridge._dynamic_source_replay_loss(
        student,
        source,
        batch=args.constraint_batch_size,
        unroll=args.unroll,
        prefix_steps=args.constraint_prefix_steps,
        physics_steps=physics_steps,
        device=device,
        config=config,
        all_style_combinations=True,
        axis_weights=(1.0, 1.0, 1.0, 0.0),
    )
    legacy_rpy = legacy_rpy_nrmse_with_original_denominator(legacy)
    valid_values = [
        *(value["valid_fraction"] for value in source_pair.values()),
        *(value["valid_fraction"] for value in per_update_pair.values()),
        legacy["valid_fraction"],
        dynamic["valid_fraction"],
    ]
    motor_max = max(
        *(value["student_motor_max_absolute"] for value in source_pair.values()),
        *(value["student_motor_max_absolute"] for value in per_update_pair.values()),
        legacy["student_motor_max_absolute"],
        dynamic["student_motor_max_absolute"],
    )
    common_diagnostics = {
        "source_pair_common_throttle_rms": max(
            value["common_error_rms"] for value in source_pair.values()
        ),
        "source_pair_common_throttle_max_absolute": max(
            value["common_error_max_absolute"] for value in source_pair.values()
        ),
        "per_update_pair_common_throttle_rms": max(
            value["common_error_rms"] for value in per_update_pair.values()
        ),
        "source_pair_absolute_teacher_throttle_nrmse": max(
            value["absolute_teacher_throttle_nrmse"] for value in source_pair.values()
        ),
        "legacy_throttle_source_nrmse": legacy["throttle_source_nrmse"],
        "legacy_absolute_teacher_throttle_nrmse": legacy["absolute_teacher_throttle_nrmse"],
        "dynamic_throttle_source_nrmse": dynamic["throttle_source_nrmse"],
        "dynamic_absolute_teacher_throttle_nrmse": dynamic["absolute_teacher_throttle_nrmse"],
    }
    all_finite = native_train.numeric_tree_is_finite(
        [source_pair, per_update_pair, legacy, dynamic, legacy_rpy, motor_max]
    )
    reasons = []
    if not all_finite:
        reasons.append("nonfinite functional-ablation measurement")
    if legacy_rpy > trust.SOURCE_FUNCTION_NRMSE_LIMIT:
        reasons.append("legacy R/P/Y source error with original four-axis denominator")
    if any(
        dynamic[f"{axis}_source_nrmse"] > trust.SOURCE_FUNCTION_NRMSE_LIMIT
        for axis in ("roll", "pitch", "yaw")
    ):
        reasons.append("dynamic R/P/Y source replay error")
    if min(valid_values) < 1.0:
        reasons.append("invalid complete replay")
    if motor_max > 1.0 + MOTOR_BOUND_TOLERANCE:
        reasons.append("motor output exceeded actuator bound")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "all_finite": all_finite,
        "source_pair": source_pair,
        "per_update_pair": per_update_pair,
        "legacy": legacy,
        "legacy_rpy_nrmse_original_denominator": legacy_rpy,
        "dynamic": dynamic,
        "valid_fraction_minimum": min(valid_values),
        "student_motor_max_absolute": motor_max,
        "absolute_throttle_diagnostics_not_gated": common_diagnostics,
    }


@torch.no_grad()
def evaluate_trial(
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
    motion = native_train.evaluate_motion_banks(
        student,
        banks=args.guard_banks,
        batch=args.batch_size,
        seed=args.seed + 40_000,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    improvement = baseline_guard["fixed_scale_nrmse"] - motion["fixed_scale_nrmse"]
    bank_improvements = native_train.guard_bank_improvements(baseline_guard, motion)
    height = shallow.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = functional_ablation_checks(
        student,
        source,
        previous,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    source_delta, step_delta = upstream_train.selected_deltas(
        student,
        source_parameters,
        previous_parameters,
        edge_indices,
        bias_indices,
    )
    height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    step_rms = upstream.reference_family_rms(step_delta)
    source_rms = upstream.reference_family_rms(source_delta)
    source_metric = upstream.reference_metric_norm(source_delta)
    all_finite = all(
        native_train.numeric_tree_is_finite(value)
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
    if any(value < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT for value in bank_improvements):
        reasons.append("one or more fixed guard banks improved by less than 1e-4 NRMSE")
    if not functional["pass"]:
        reasons.append("non-throttle functional preservation checks failed")
    if not height_ok:
        reasons.append("height contrast left source-relative 10% band")
    if max(step_rms.values()) > shallow.MASK_STEP_FAMILY_RMS_CAP * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected-family step RMS cap")
    if source_metric > shallow.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected-family source metric radius")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "motion": motion,
        "motion_nrmse_improvement": improvement,
        "motion_nrmse_improvement_by_guard_bank": bank_improvements,
        "all_finite": all_finite,
        "height_response": height,
        "functional_ablation": functional,
        "reference_step_family_rms": step_rms,
        "reference_source_family_rms": source_rms,
        "reference_source_metric_norm": source_metric,
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
    functional = functional_ablation_checks(
        student,
        source,
        student,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    height = shallow.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    source_delta = shallow.masked_source_delta(
        student, source_parameters, edge_indices, bias_indices
    )
    height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    source_metric = upstream.reference_metric_norm(source_delta)
    all_finite = all(
        native_train.numeric_tree_is_finite(value) for value in (functional, height, source_delta)
    )
    passed = (
        all_finite
        and functional["pass"]
        and height_ok
        and source_metric <= shallow.MASK_SOURCE_METRIC_RADIUS * 1.00001
    )
    return {
        "pass": passed,
        "all_finite": all_finite,
        "functional_ablation": functional,
        "height_response": height,
        "reference_source_family_rms": upstream.reference_family_rms(source_delta),
        "reference_source_metric_norm": source_metric,
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
    terminal: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-common-anchor-ablation-v1",
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
        "protocol": protocol_manifest(args),
        "stop_reason": stop_reason,
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
        terminal=terminal,
    )
    state_path = args.output_dir / "training-state.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(state_path)


def main() -> int:
    args = parse_args()
    validate_args(args)
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
    if int(loaded.get("policy_hz", args.policy_hz)) != args.policy_hz:
        raise SystemExit("checkpoint and requested policy frequencies do not match")
    student = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    source = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    previous = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    for controller in (student, source, previous):
        controller.load_state_dict(loaded["controller"])
    source.eval().requires_grad_(False)
    previous.eval().requires_grad_(False)
    student.eval()
    student.raw_time_constant.requires_grad_(False)
    edges_np, biases_np, mask_manifest = upstream.build_expanded_route_mask(
        args.graph, args.raw_dir
    )
    upstream.validate_frozen_mask(mask_manifest)
    edge_indices = torch.from_numpy(edges_np).to(device)
    bias_indices = torch.from_numpy(biases_np).to(device)
    source_parameters = trust.clone_parameters(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "training-state.pt"
    protocol = protocol_manifest(args)
    attempted = accepted = consecutive_rejections = 0
    history: list[dict[str, Any]] = []
    stop_reason = None
    resume_terminal = False
    if args.resume:
        if not state_path.is_file():
            raise SystemExit(f"resume state does not exist: {state_path}")
        resume = torch.load(state_path, map_location=device, weights_only=False)
        expected = (
            resume.get("checkpoint_schema_version") == 1
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
        resume_terminal = bool(resume.get("terminal"))

    started = perf_counter()
    benchmark_seed = args.seed + BENCHMARK_SEED_OFFSET
    baseline_benchmark = native_train.evaluate_motion_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=benchmark_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    attempts = () if resume_terminal else range(attempted + 1, args.attempts + 1)
    for attempt in attempts:
        attempted = attempt
        base_parameters = trust.clone_parameters(student)
        previous.load_state_dict(student.state_dict())
        baseline_guard = native_train.evaluate_motion_banks(
            student,
            banks=args.guard_banks,
            batch=args.batch_size,
            seed=args.seed + 40_000,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        gradient, gradient_bank = native_train.accumulated_motion_gradient(
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
        direction, direction_report = raw_direction(student, gradient, edge_indices)
        direction_report["gradient_bank"] = gradient_bank
        trials = []
        accepted_scale = None
        if direction is not None:
            derivative = direction_report["first_order_loss_derivative"]
            for scale in shallow.BACKTRACK_SCALES:
                shallow.set_masked_trial(
                    student,
                    base_parameters,
                    edge_indices,
                    bias_indices,
                    direction,
                    scale,
                )
                trial = evaluate_trial(
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
            "direction": direction_report,
            "trials": trials,
        }
        history.append(entry)
        if consecutive_rejections >= native_train.CONSECUTIVE_REJECTION_LIMIT:
            stop_reason = "five consecutive unsafe or non-improving route steps"
        elif args.smoke_test:
            stop_reason = "smoke test ended after one complete attempt"
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
            terminal=terminal,
        )
        selected_trial = next((trial for trial in trials if trial["pass"]), None)
        print(
            json.dumps(
                {
                    "progress": "attempt",
                    "attempt": attempt,
                    "accepted": entry["accepted"],
                    "accepted_scale": accepted_scale,
                    "consecutive_rejections": consecutive_rejections,
                    "motion_before": baseline_guard["fixed_scale_nrmse"],
                    "motion_after": (trials[-1]["motion"]["fixed_scale_nrmse"] if trials else None),
                    "source_common_throttle_rms": (
                        selected_trial["functional_ablation"][
                            "absolute_throttle_diagnostics_not_gated"
                        ]["source_pair_common_throttle_rms"]
                        if selected_trial is not None
                        else None
                    ),
                    "terminal": terminal,
                    "stop_reason": stop_reason,
                }
            ),
            flush=True,
        )
        if terminal:
            break

    benchmark_candidate = native_train.evaluate_motion_banks(
        student,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=benchmark_seed,
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
    benchmark_improvement = (
        1.0 - benchmark_candidate["fixed_scale_nrmse"] / baseline_benchmark["fixed_scale_nrmse"]
    )
    useful_progress = (
        native_train.numeric_tree_is_finite(
            [baseline_benchmark, benchmark_candidate, preservation, benchmark_improvement]
        )
        and benchmark_improvement >= USEFUL_PROGRESS_FRACTION
        and preservation["pass"]
    )
    fresh_qualification = None
    if useful_progress and not args.smoke_test:
        fresh_seed = args.seed + FRESH_QUALIFICATION_SEED_OFFSET
        fresh_baseline = native_train.evaluate_motion_banks(
            source,
            banks=args.evaluation_banks,
            batch=args.batch_size,
            seed=fresh_seed,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        fresh_candidate = native_train.evaluate_motion_banks(
            student,
            banks=args.evaluation_banks,
            batch=args.batch_size,
            seed=fresh_seed,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        fresh_decision = native_train.evaluation_decision(
            fresh_candidate, fresh_baseline, preservation, milestone=False
        )
        fresh_qualification = {
            "seed": fresh_seed,
            "baseline": fresh_baseline,
            "candidate": fresh_candidate,
            "decision": fresh_decision,
        }
    if stop_reason is None:
        stop_reason = "attempt limit reached"
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
            terminal=True,
        ),
        endpoint,
    )
    report = {
        "experiment": "variable-height-common-anchor-ablation-v1",
        "diagnostic_passed_useful_progress_gate": useful_progress,
        "passed_fresh_qualification": bool(
            fresh_qualification is not None and fresh_qualification["decision"]["pass"]
        ),
        "promoted": False,
        "closed_loop_hover_run": False,
        "claim": "constraint ablation only; no controller promotion or automatic handoff",
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
        "history": history,
        "reused_benchmark": {
            "label": "matched benchmark reused from anchored upstream run",
            "seed": benchmark_seed,
            "baseline": baseline_benchmark,
            "candidate": benchmark_candidate,
            "motion_nrmse_improvement_fraction": benchmark_improvement,
            "required_improvement_fraction": USEFUL_PROGRESS_FRACTION,
            "preservation": preservation,
        },
        "fresh_qualification": fresh_qualification,
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
                "diagnostic_passed_useful_progress_gate": useful_progress,
                "passed_fresh_qualification": report["passed_fresh_qualification"],
                "attempts_completed": attempted,
                "accepted_updates": accepted,
                "stop_reason": stop_reason,
                "benchmark_baseline_nrmse": baseline_benchmark["fixed_scale_nrmse"],
                "benchmark_candidate_nrmse": benchmark_candidate["fixed_scale_nrmse"],
                "benchmark_improvement_fraction": benchmark_improvement,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
