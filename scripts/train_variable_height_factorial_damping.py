#!/usr/bin/env python3
"""Train the preregistered height-null factorial visual-damping route."""

from __future__ import annotations

import argparse
import json
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
import audit_variable_height_factorial_damping as factorial  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import audit_variable_height_upstream_damping_route as upstream  # noqa: E402
import train_variable_height_common_anchor_ablation as ablation  # noqa: E402
import train_variable_height_damping_routes as native_train  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402
import train_variable_height_upstream_damping_route as upstream_train  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

ATTEMPT_LIMIT = 25
GRADIENT_BANKS = 2
GUARD_BANKS = 2
EVALUATION_BANKS = 16
DEVELOPMENT_SEED_OFFSET = 60_000
FRESH_SEED_OFFSET = 70_000
GUARD_SEED_OFFSET = 40_000
USEFUL_PROGRESS_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = shallow.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=REPO_ROOT / "runs/variable-height-hover/factorial-damping-train-001",
        seed=310_949,
        batch_size=4,
    )
    parser.add_argument("--attempts", type=int, default=ATTEMPT_LIMIT)
    parser.add_argument("--gradient-banks", type=int, default=GRADIENT_BANKS)
    parser.add_argument("--guard-banks", type=int, default=GUARD_BANKS)
    parser.add_argument("--evaluation-banks", type=int, default=EVALUATION_BANKS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    shallow.validate_args(args)
    if args.batch_size % 4:
        raise SystemExit("--batch-size must be a multiple of four")
    if args.history_steps < max(factorial.APPROACH_STEPS):
        raise SystemExit("history must cover every preregistered approach duration")
    counts = (args.attempts, args.gradient_banks, args.guard_banks, args.evaluation_banks)
    if min(counts) < 1:
        raise SystemExit("training counts must be positive")
    expected = (ATTEMPT_LIMIT, GRADIENT_BANKS, GUARD_BANKS, EVALUATION_BANKS)
    if not args.smoke_test and counts != expected:
        raise SystemExit(f"the factorial damping run is preregistered for {expected}")


def protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "attempt_limit": args.attempts,
        "gradient_banks": args.gradient_banks,
        "guard_banks": args.guard_banks,
        "evaluation_banks": args.evaluation_banks,
        "scenes_per_bank": args.batch_size,
        "histories_per_scene": len(factorial.BRANCH_SIGNS),
        "gradient_scenes_per_attempt": args.gradient_banks * args.batch_size,
        "gradient_histories_per_attempt": (
            args.gradient_banks * args.batch_size * len(factorial.BRANCH_SIGNS)
        ),
        "guard_scenes_per_trial": args.guard_banks * args.batch_size,
        "guard_histories_per_trial": (
            args.guard_banks * args.batch_size * len(factorial.BRANCH_SIGNS)
        ),
        "evaluation_scenes": args.evaluation_banks * args.batch_size,
        "evaluation_histories": (
            args.evaluation_banks * args.batch_size * len(factorial.BRANCH_SIGNS)
        ),
        "constraint_batch_size": args.constraint_batch_size,
        "unroll": args.unroll,
        "constraint_prefix_steps": args.constraint_prefix_steps,
        "history_steps": args.history_steps,
        "policy_hz": args.policy_hz,
        "device": args.device,
        "smoke_test": args.smoke_test,
        "seed": args.seed,
        "gradient_seed_formula": "seed + attempt * 100 + bank",
        "guard_seed_offset": GUARD_SEED_OFFSET,
        "development_seed_offset": DEVELOPMENT_SEED_OFFSET,
        "fresh_seed_offset": FRESH_SEED_OFFSET,
        "height_error_magnitudes_metres": list(factorial.HEIGHT_ERROR_MAGNITUDES),
        "vertical_speed_magnitudes_metres_per_second": list(
            factorial.VERTICAL_SPEED_MAGNITUDES
        ),
        "approach_steps": list(factorial.APPROACH_STEPS),
        "prefix_steps": list(factorial.PREFIX_STEPS),
        "branch_sign_order_height_velocity": [
            list(item) for item in factorial.BRANCH_SIGNS
        ],
        "objective": "factorial damping component only",
        "matched_height_component_projection": "same cases as every gradient bank",
        "common_component": "unanchored diagnostic only",
        "interaction_component": "diagnostic only",
        "consecutive_rejection_limit": native_train.CONSECUTIVE_REJECTION_LIMIT,
        "reference_metric_denominators": dict(upstream.REFERENCE_DENOMINATORS),
        "reference_metric_governs": [
            "Riesz descent direction",
            "step family caps",
            "source metric radius",
        ],
        "selected_family_step_rms_cap": shallow.MASK_STEP_FAMILY_RMS_CAP,
        "selected_source_metric_radius": shallow.MASK_SOURCE_METRIC_RADIUS,
        "backtrack_scales": list(shallow.BACKTRACK_SCALES),
        "edge_magnitude_bounds": [0.0, 8.0],
        "post_bound_linearized_height_change_relative_maximum": (
            factorial.LINEARIZED_HEIGHT_NULL_RELATIVE_LIMIT
        ),
        "matched_height_nondegenerate_absolute_minimum": (
            factorial.MATCHED_HEIGHT_NONDEGENERATE_ABSOLUTE
        ),
        "matched_height_source_ratio_range": list(shallow.HEIGHT_CONTRAST_RATIO_RANGE),
        "existing_height_source_ratio_range": list(shallow.HEIGHT_CONTRAST_RATIO_RANGE),
        "minimum_per_guard_bank_damping_nrmse_improvement": (
            shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT
        ),
        "useful_progress_damping_nrmse_improvement_fraction": USEFUL_PROGRESS_FRACTION,
        "fresh_correct_sign_accuracy": native_train.FINAL_SIGN_ACCURACY,
        "fresh_teacher_aligned_gain_range": list(native_train.FINAL_GAIN_RANGE),
        "absolute_throttle_source_gate": False,
        "absolute_teacher_throttle_objective_or_gate": False,
        "dynamic_rpy_each_nrmse_limit": trust.SOURCE_FUNCTION_NRMSE_LIMIT,
        "motor_output_absolute_bound": 1.0,
    }


def summarize_factorial_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for component in ("damping", "height", "common", "interaction"):
        target = torch.tensor(
            [value for metric in metrics for value in metric[component]["target"]]
        )
        prediction = torch.tensor(
            [value for metric in metrics for value in metric[component]["prediction"]]
        )
        summary[component] = factorial._component_report(prediction, target)
    summary["scenes"] = sum(len(metric["height"]["target"]) for metric in metrics)
    summary["histories"] = summary["scenes"] * len(factorial.BRANCH_SIGNS)
    summary["endpoint_opposite_velocity_image_max_absolute_difference"] = max(
        metric["endpoint_opposite_velocity_image_max_absolute_difference"]
        for metric in metrics
    )
    summary["all_finite"] = native_train.numeric_tree_is_finite(summary)
    summary["banks"] = metrics
    return summary


@torch.no_grad()
def evaluate_factorial_banks(
    controller: ConnectomeController,
    *,
    banks: int,
    batch: int,
    seed: int,
    history_steps: int,
    policy_hz: int,
    config: HoverConfig,
) -> dict[str, Any]:
    metrics = []
    for bank in range(banks):
        _, report = factorial.factorial_terms(
            controller,
            controller,
            batch=batch,
            seed=seed + bank,
            history_steps=history_steps,
            policy_hz=policy_hz,
            config=config,
        )
        metrics.append(report)
    return summarize_factorial_metrics(metrics)


def accumulated_gradient_and_height_rows(
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
) -> tuple[dict[str, Tensor], list[dict[str, Tensor]], Tensor, dict[str, Any]]:
    accumulated = {
        "edge_magnitude": torch.zeros(len(edge_indices), device=edge_indices.device),
        "bias": torch.zeros(len(bias_indices), device=bias_indices.device),
    }
    rows: list[dict[str, Tensor]] = []
    source_height = []
    metrics = []
    for bank in range(banks):
        forward = factorial._factorial_forward(
            student,
            student,
            batch=batch,
            seed=seed + bank,
            history_steps=history_steps,
            policy_hz=policy_hz,
            config=config,
        )
        damping_error = forward.prediction["damping"] - forward.target["damping"]
        loss = (damping_error / factorial.DAMPING_MOTOR_SCALE).square().mean()
        gradient = shallow.masked_gradient(
            loss,
            student,
            edge_indices,
            bias_indices,
            retain_graph=True,
        )
        for name in shallow.MASK_FAMILIES:
            accumulated[name].add_(gradient[name], alpha=1.0 / banks)
        for item in range(batch):
            rows.append(
                shallow.masked_gradient(
                    forward.prediction["height"][item],
                    student,
                    edge_indices,
                    bias_indices,
                    retain_graph=item + 1 < batch,
                )
            )
        source_height.append(forward.prediction["height"].detach())
        metrics.append(factorial.factorial_forward_report(forward))
    return accumulated, rows, torch.cat(source_height), summarize_factorial_metrics(metrics)


def height_null_direction(
    student: ConnectomeController,
    gradient: dict[str, Tensor],
    rows: list[dict[str, Tensor]],
    source_height_component: Tensor,
    edge_indices: Tensor,
) -> tuple[dict[str, Tensor] | None, dict[str, Any]]:
    raw, raw_scale = upstream.scale_to_reference_cap(
        upstream_train.fixed_metric_riesz(gradient), shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    projected, projection = factorial.bound_aware_height_null_projection(
        raw,
        rows,
        student.edge_magnitude[edge_indices].detach(),
    )
    reports = {
        "raw_unprotected_diagnostic": factorial.direction_report(
            raw, gradient, rows, source_height_component
        ),
        "matched_height_null": factorial.direction_report(
            projected, gradient, rows, source_height_component
        ),
    }
    admissible = (
        reports["matched_height_null"]["admissible"]
        and projection["converged_without_bound_violation"]
    )
    return (
        projected if admissible else None,
        {
            "candidate": "matched_height_null",
            "admissible": admissible,
            "raw_initial_cap_scale": raw_scale,
            "bound_aware_height_projection": projection,
            "candidate_reports": reports,
        },
    )


def matched_height_bank_report(
    candidate: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    if len(candidate["banks"]) != len(source["banks"]):
        raise ValueError("candidate and source must have identical bank counts")
    banks = [
        factorial.matched_height_response(after, before)
        for after, before in zip(candidate["banks"], source["banks"], strict=True)
    ]
    scenes = [item for bank in banks for item in bank["by_scene"]]
    ratios = [
        item["candidate_to_source_ratio"]
        for item in scenes
        if item["nondegenerate"]
    ]
    all_nondegenerate = all(item["nondegenerate"] for item in scenes)
    passed = (
        all_nondegenerate
        and bool(ratios)
        and min(ratios) >= shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        and max(ratios) <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
    )
    return {
        "pass": passed,
        "all_scenes_nondegenerate": all_nondegenerate,
        "ratio_minimum": min(ratios) if ratios else None,
        "ratio_maximum": max(ratios) if ratios else None,
        "banks": banks,
    }


def guard_bank_improvements(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[float]:
    if len(baseline["banks"]) != len(candidate["banks"]):
        raise ValueError("guard evaluations must contain the same number of banks")
    return [
        before["damping"]["fixed_scale_nrmse"]
        - after["damping"]["fixed_scale_nrmse"]
        for before, after in zip(baseline["banks"], candidate["banks"], strict=True)
    ]


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
    source_guard: dict[str, Any],
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    candidate = evaluate_factorial_banks(
        student,
        banks=args.guard_banks,
        batch=args.batch_size,
        seed=args.seed + GUARD_SEED_OFFSET,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    improvements = guard_bank_improvements(baseline_guard, candidate)
    matched_height = matched_height_bank_report(candidate, source_guard)
    height = shallow.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = ablation.functional_ablation_checks(
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
    step_rms = upstream.reference_family_rms(step_delta)
    source_rms = upstream.reference_family_rms(source_delta)
    source_metric = upstream.reference_metric_norm(source_delta)
    existing_height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    finite = all(
        native_train.numeric_tree_is_finite(value)
        for value in (
            baseline_guard,
            source_guard,
            candidate,
            matched_height,
            height,
            functional,
            source_delta,
            step_delta,
        )
    )
    reasons = []
    if not finite:
        reasons.append("nonfinite acceptance metric or selected parameter displacement")
    if any(value < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT for value in improvements):
        reasons.append("one or more fixed guard banks improved D NRMSE by less than 1e-4")
    if not matched_height["pass"]:
        reasons.append("one or more matched factorial P scenes left source-relative 10% band")
    if not existing_height_ok:
        reasons.append("existing visual height response left source-relative 10% band")
    if not functional["pass"]:
        reasons.append("non-throttle functional preservation checks failed")
    if max(step_rms.values()) > shallow.MASK_STEP_FAMILY_RMS_CAP * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected-family step RMS cap")
    if source_metric > shallow.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5):
        reasons.append("fixed-denominator selected-family source metric radius")
    if candidate["endpoint_opposite_velocity_image_max_absolute_difference"] != 0.0:
        reasons.append("opposite-velocity histories did not end on identical pixels")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "all_finite": finite,
        "factorial": candidate,
        "damping_nrmse_improvement": (
            baseline_guard["damping"]["fixed_scale_nrmse"]
            - candidate["damping"]["fixed_scale_nrmse"]
        ),
        "damping_nrmse_improvement_by_guard_bank": improvements,
        "matched_height_response": matched_height,
        "existing_height_response": height,
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
    candidate_factorial: dict[str, Any],
    source_factorial: dict[str, Any],
    args: argparse.Namespace,
    physics_steps: int,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    matched_height = matched_height_bank_report(candidate_factorial, source_factorial)
    height = shallow.height_response_summary(
        student,
        source,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    functional = ablation.functional_ablation_checks(
        student,
        source,
        student,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    source_delta = shallow.masked_source_delta(
        student, source_parameters, edge_indices, bias_indices
    )
    source_metric = upstream.reference_metric_norm(source_delta)
    existing_height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    finite = all(
        native_train.numeric_tree_is_finite(value)
        for value in (candidate_factorial, source_factorial, matched_height, height, functional)
    )
    passed = (
        finite
        and matched_height["pass"]
        and existing_height_ok
        and functional["pass"]
        and source_metric <= shallow.MASK_SOURCE_METRIC_RADIUS * (1.0 + 1.0e-5)
        and candidate_factorial[
            "endpoint_opposite_velocity_image_max_absolute_difference"
        ]
        == 0.0
    )
    return {
        "pass": passed,
        "all_finite": finite,
        "matched_height_response": matched_height,
        "existing_height_response": height,
        "functional_ablation": functional,
        "reference_source_family_rms": upstream.reference_family_rms(source_delta),
        "reference_source_metric_norm": source_metric,
    }


def checkpoint_payload(
    controller: ConnectomeController,
    *,
    args: argparse.Namespace,
    source_sha256: str,
    graph_sha256: str,
    mask: dict[str, Any],
    attempted: int,
    accepted: int,
    consecutive_rejections: int,
    history: list[dict[str, Any]],
    stop_reason: str | None,
    terminal: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-factorial-damping-route-training-v1",
        "non_promotional": True,
        "controller": controller.state_dict(),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "graph_sha256": graph_sha256,
        "edge_indices_sha256": mask["edge_indices_sha256"],
        "bias_node_indices_sha256": mask["bias_node_indices_sha256"],
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
    mask: dict[str, Any],
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
        mask=mask,
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


def fresh_qualification_decision(
    candidate: dict[str, Any],
    matched_height: dict[str, Any],
    preservation: dict[str, Any],
) -> dict[str, Any]:
    damping = candidate["damping"]
    gain = damping["teacher_aligned_gain"]
    reasons = []
    if damping["correct_sign_fraction"] < native_train.FINAL_SIGN_ACCURACY:
        reasons.append("damping correct-sign fraction below 90%")
    if not native_train.FINAL_GAIN_RANGE[0] <= gain <= native_train.FINAL_GAIN_RANGE[1]:
        reasons.append("teacher-aligned damping gain outside [0.5, 1.5]")
    if not matched_height["pass"]:
        reasons.append("fresh matched P response left source-relative 10% band")
    if not preservation["pass"]:
        reasons.append("terminal preservation checks failed")
    if not candidate["all_finite"]:
        reasons.append("nonfinite fresh factorial evaluation")
    return {"pass": not reasons, "reasons": reasons}


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
    edges_np, biases_np, mask = upstream.build_expanded_route_mask(args.graph, args.raw_dir)
    upstream.validate_frozen_mask(mask)
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
            and resume.get("edge_indices_sha256") == mask["edge_indices_sha256"]
            and resume.get("bias_node_indices_sha256") == mask["bias_node_indices_sha256"]
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
    guard_seed = args.seed + GUARD_SEED_OFFSET
    source_guard = evaluate_factorial_banks(
        source,
        banks=args.guard_banks,
        batch=args.batch_size,
        seed=guard_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    development_seed = args.seed + DEVELOPMENT_SEED_OFFSET
    source_development = evaluate_factorial_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=development_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    attempts = () if resume_terminal else range(attempted + 1, args.attempts + 1)
    for attempt in attempts:
        attempted = attempt
        base_parameters = trust.clone_parameters(student)
        previous.load_state_dict(student.state_dict())
        baseline_guard = evaluate_factorial_banks(
            student,
            banks=args.guard_banks,
            batch=args.batch_size,
            seed=guard_seed,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        gradient, rows, height_component, gradient_bank = (
            accumulated_gradient_and_height_rows(
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
        )
        direction, direction_report = height_null_direction(
            student, gradient, rows, height_component, edge_indices
        )
        direction_report["gradient_bank"] = gradient_bank
        trials = []
        accepted_scale = None
        if direction is not None:
            derivative = direction_report["candidate_reports"]["matched_height_null"][
                "first_order_damping_loss_derivative"
            ]
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
                    source_guard=source_guard,
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
            "baseline_guard": baseline_guard,
            "direction": direction_report,
            "trials": trials,
        }
        history.append(entry)
        if consecutive_rejections >= native_train.CONSECUTIVE_REJECTION_LIMIT:
            stop_reason = "five consecutive unsafe or non-improving height-null steps"
        elif args.smoke_test:
            stop_reason = "smoke test ended after one complete attempt"
        terminal = stop_reason is not None
        save_progress(
            student,
            args=args,
            source_sha256=source_sha256,
            graph_sha256=graph_sha256,
            mask=mask,
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
                    "damping_before": baseline_guard["damping"]["fixed_scale_nrmse"],
                    "damping_after": (
                        selected_trial["factorial"]["damping"]["fixed_scale_nrmse"]
                        if selected_trial is not None
                        else None
                    ),
                    "matched_height_ratio_range": (
                        [
                            selected_trial["matched_height_response"]["ratio_minimum"],
                            selected_trial["matched_height_response"]["ratio_maximum"],
                        ]
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

    candidate_development = evaluate_factorial_banks(
        student,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=development_seed,
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
        candidate_factorial=candidate_development,
        source_factorial=source_development,
        args=args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    development_improvement = (
        1.0
        - candidate_development["damping"]["fixed_scale_nrmse"]
        / source_development["damping"]["fixed_scale_nrmse"]
    )
    useful_progress = (
        attempted == ATTEMPT_LIMIT
        and native_train.numeric_tree_is_finite(
            [source_development, candidate_development, preservation, development_improvement]
        )
        and development_improvement >= USEFUL_PROGRESS_FRACTION
        and preservation["pass"]
    )
    fresh_qualification = None
    if useful_progress and not args.smoke_test:
        fresh_seed = args.seed + FRESH_SEED_OFFSET
        source_fresh = evaluate_factorial_banks(
            source,
            banks=args.evaluation_banks,
            batch=args.batch_size,
            seed=fresh_seed,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        candidate_fresh = evaluate_factorial_banks(
            student,
            banks=args.evaluation_banks,
            batch=args.batch_size,
            seed=fresh_seed,
            history_steps=args.history_steps,
            policy_hz=args.policy_hz,
            config=config,
        )
        fresh_height = matched_height_bank_report(candidate_fresh, source_fresh)
        fresh_qualification = {
            "seed": fresh_seed,
            "source": source_fresh,
            "candidate": candidate_fresh,
            "matched_height_response": fresh_height,
            "decision": fresh_qualification_decision(
                candidate_fresh, fresh_height, preservation
            ),
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
            mask=mask,
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
        "experiment": "variable-height-factorial-damping-route-training-v1",
        "passed_useful_progress_gate": useful_progress,
        "passed_fresh_qualification": bool(
            fresh_qualification is not None and fresh_qualification["decision"]["pass"]
        ),
        "promoted": False,
        "closed_loop_hover_run": False,
        "claim": "bounded replay training only; no controller promotion",
        "source": {
            "graph": responsibility.stable_path(args.graph),
            "graph_sha256": graph_sha256,
            "checkpoint": responsibility.stable_path(args.checkpoint),
            "checkpoint_sha256": source_sha256,
        },
        "mask": mask,
        "protocol": protocol,
        "attempts_completed": attempted,
        "accepted_updates": accepted,
        "consecutive_rejections": consecutive_rejections,
        "stop_reason": stop_reason,
        "history": history,
        "development": {
            "seed": development_seed,
            "source": source_development,
            "candidate": candidate_development,
            "damping_nrmse_improvement_fraction": development_improvement,
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
                "passed_useful_progress_gate": useful_progress,
                "passed_fresh_qualification": report["passed_fresh_qualification"],
                "attempts_completed": attempted,
                "accepted_updates": accepted,
                "stop_reason": stop_reason,
                "development_source_damping_nrmse": source_development["damping"][
                    "fixed_scale_nrmse"
                ],
                "development_candidate_damping_nrmse": candidate_development[
                    "damping"
                ]["fixed_scale_nrmse"],
                "development_improvement_fraction": development_improvement,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
