#!/usr/bin/env python3
"""Train the preregistered upstream-expanded native visual-damping route."""

from __future__ import annotations

import argparse
import json
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

import audit_variable_height_damping_routes as shallow  # noqa: E402
import audit_variable_height_damping_routes_v2 as bounded  # noqa: E402
import audit_variable_height_responsibilities as responsibility  # noqa: E402
import audit_variable_height_upstream_damping_route as upstream  # noqa: E402
import train_variable_height_damping_routes as native_train  # noqa: E402
import train_variable_height_trust_region as trust  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = shallow.argument_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir=(REPO_ROOT / "runs/variable-height-hover/upstream-damping-route-train-001"),
        seed=290_941,
    )
    parser.add_argument("--attempts", type=int, default=native_train.ATTEMPT_LIMIT)
    parser.add_argument("--milestone-attempt", type=int, default=native_train.MILESTONE_ATTEMPT)
    parser.add_argument("--gradient-banks", type=int, default=2)
    parser.add_argument("--guard-banks", type=int, default=2)
    parser.add_argument("--evaluation-banks", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    shallow.validate_args(args)
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
        args.attempts != native_train.ATTEMPT_LIMIT
        or args.milestone_attempt != native_train.MILESTONE_ATTEMPT
    ):
        raise SystemExit("v1 is preregistered for 50 attempts and its gate at attempt 25")


def fixed_metric_riesz(gradient: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: -gradient[name] * upstream.REFERENCE_DENOMINATORS[name]
        for name in shallow.MASK_FAMILIES
    }


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
        "milestone_seed_offset": native_train.MILESTONE_SEED_OFFSET,
        "terminal_seed_offset": native_train.TERMINAL_SEED_OFFSET,
        "consecutive_rejection_limit": native_train.CONSECUTIVE_REJECTION_LIMIT,
        "reference_metric_denominators": dict(upstream.REFERENCE_DENOMINATORS),
        "reference_metric_governs": [
            "Riesz descent direction",
            "Jacobian equality projection",
            "step family caps",
            "source metric radius",
        ],
        "selected_family_step_rms_cap": shallow.MASK_STEP_FAMILY_RMS_CAP,
        "selected_source_metric_radius": shallow.MASK_SOURCE_METRIC_RADIUS,
        "candidate_blend_fractions": list(bounded.BLEND_FRACTIONS),
        "backtrack_scales": list(shallow.BACKTRACK_SCALES),
        "edge_magnitude_bounds": [0.0, 8.0],
        "per_update_common_throttle_rms": bounded.PER_UPDATE_COMMON_RMS,
        "source_common_throttle_rms": bounded.SOURCE_COMMON_RMS,
        "source_common_throttle_max_absolute": bounded.COMMON_MAX_ABSOLUTE,
        "height_contrast_source_ratio_range": list(shallow.HEIGHT_CONTRAST_RATIO_RANGE),
        "minimum_per_guard_bank_motion_nrmse_improvement": (
            shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT
        ),
        "milestone_motion_nrmse_improvement_fraction": (
            native_train.MILESTONE_IMPROVEMENT_FRACTION
        ),
        "final_sign_accuracy": native_train.FINAL_SIGN_ACCURACY,
        "final_teacher_aligned_gain_range": list(native_train.FINAL_GAIN_RANGE),
    }


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
) -> tuple[str | None, dict[str, dict[str, Tensor]], dict[str, Any]]:
    raw, raw_scale = upstream.scale_to_reference_cap(
        fixed_metric_riesz(gradient), shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    rows, residual, common_report = shallow.common_constraint_rows(
        student,
        source,
        edge_indices,
        bias_indices,
        args=direction_args,
        physics_steps=physics_steps,
        device=device,
        config=config,
    )
    equality_small, projection = upstream.project_in_reference_metric(raw, rows, residual)
    equality, equality_scale = upstream.scale_to_reference_cap(
        equality_small, shallow.MASK_STEP_FAMILY_RMS_CAP
    )
    current_edges = student.edge_magnitude[edge_indices].detach()
    candidates = {}
    bounds = {}
    for fraction in bounded.BLEND_FRACTIONS:
        label = f"equality_to_raw_{fraction:.2f}"
        mixed = {
            name: (1.0 - fraction) * equality[name] + fraction * raw[name]
            for name in shallow.MASK_FAMILIES
        }
        mixed, _ = upstream.scale_to_reference_cap(mixed, shallow.MASK_STEP_FAMILY_RMS_CAP)
        candidates[label], bounds[label] = bounded.apply_edge_bounds(mixed, current_edges)
    selected, screening = upstream.screen_candidates(candidates, gradient, rows, residual)
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


def selected_deltas(
    student: ConnectomeController,
    source_parameters: dict[str, Tensor],
    previous_parameters: dict[str, Tensor],
    edge_indices: Tensor,
    bias_indices: Tensor,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    source = shallow.masked_source_delta(student, source_parameters, edge_indices, bias_indices)
    step = {
        "edge_magnitude": student.edge_magnitude[edge_indices]
        - previous_parameters["edge_magnitude"][edge_indices],
        "bias": student.bias[bias_indices] - previous_parameters["bias"][bias_indices],
    }
    return source, step


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
    source_delta, step_delta = selected_deltas(
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
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
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
    step_rms = upstream.reference_family_rms(step_delta)
    source_rms = upstream.reference_family_rms(source_delta)
    source_metric = upstream.reference_metric_norm(source_delta)
    reasons = []
    if not all_finite:
        reasons.append("nonfinite acceptance metric or selected parameter displacement")
    if any(value < shallow.MINIMUM_MOTION_NRMSE_IMPROVEMENT for value in bank_improvements):
        reasons.append("one or more fixed guard banks improved by less than 1e-4 NRMSE")
    if not functional["pass"]:
        reasons.append("existing complete-replay trust checks failed")
    if step_common_rms > bounded.PER_UPDATE_COMMON_RMS:
        reasons.append("per-update common throttle RMS")
    if source_common_rms > bounded.SOURCE_COMMON_RMS:
        reasons.append("source-global common throttle RMS")
    if source_common_max > bounded.COMMON_MAX_ABSOLUTE:
        reasons.append("source-global common throttle maximum")
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
        "motion_improvement_per_step_common_rms": (improvement / max(step_common_rms, 1.0e-12)),
        "all_finite": all_finite,
        "height_response": height,
        "functional_trust": functional,
        "per_update_common_throttle_rms": step_common_rms,
        "source_common_throttle_rms": source_common_rms,
        "source_common_throttle_max_absolute": source_common_max,
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
    source_common_rms = max(
        value["common_error_rms"] for value in functional["source_pair"].values()
    )
    source_common_max = max(
        value["common_error_max_absolute"] for value in functional["source_pair"].values()
    )
    height_ok = all(
        shallow.HEIGHT_CONTRAST_RATIO_RANGE[0]
        <= value["student_to_source_ratio"]
        <= shallow.HEIGHT_CONTRAST_RATIO_RANGE[1]
        for value in height.values()
    )
    all_finite = all(
        native_train.numeric_tree_is_finite(value) for value in (functional, height, source_delta)
    )
    source_metric = upstream.reference_metric_norm(source_delta)
    pass_gate = (
        all_finite
        and functional["pass"]
        and height_ok
        and source_common_rms <= bounded.SOURCE_COMMON_RMS
        and source_common_max <= bounded.COMMON_MAX_ABSOLUTE
        and source_metric <= shallow.MASK_SOURCE_METRIC_RADIUS * 1.00001
    )
    return {
        "pass": pass_gate,
        "all_finite": all_finite,
        "functional_trust": functional,
        "height_response": height,
        "source_common_throttle_rms": source_common_rms,
        "source_common_throttle_max_absolute": source_common_max,
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
    milestone_report: dict[str, Any] | None,
    terminal: bool,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": 1,
        "experiment": "variable-height-native-upstream-damping-route-training-v1",
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
        milestone_report = resume.get("milestone")
        resume_terminal = bool(resume.get("terminal"))

    started = perf_counter()
    milestone_seed = args.seed + native_train.MILESTONE_SEED_OFFSET
    terminal_seed = args.seed + native_train.TERMINAL_SEED_OFFSET
    baseline_milestone = native_train.evaluate_motion_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=milestone_seed,
        history_steps=args.history_steps,
        policy_hz=args.policy_hz,
        config=config,
    )
    baseline_terminal = native_train.evaluate_motion_banks(
        source,
        banks=args.evaluation_banks,
        batch=args.batch_size,
        seed=terminal_seed,
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
            for scale in shallow.BACKTRACK_SCALES:
                shallow.set_masked_trial(
                    student,
                    base_parameters,
                    edge_indices,
                    bias_indices,
                    candidates[selected],
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
            "direction": direction,
            "selected_candidate": selected,
            "trials": trials,
        }
        history.append(entry)
        if consecutive_rejections >= native_train.CONSECUTIVE_REJECTION_LIMIT:
            stop_reason = "five consecutive unsafe or non-improving route steps"
        elif args.smoke_test:
            stop_reason = "smoke test ended after one complete attempt"
        elif attempt == args.milestone_attempt:
            candidate = native_train.evaluate_motion_banks(
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
            decision = native_train.evaluation_decision(
                candidate, baseline_milestone, preservation, milestone=True
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

    candidate = native_train.evaluate_motion_banks(
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
    decision = native_train.evaluation_decision(
        candidate, baseline_terminal, preservation, milestone=False
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
    accepted_efficiency = [
        trial["motion_improvement_per_step_common_rms"]
        for entry in history
        if entry["accepted"]
        for trial in entry["trials"]
        if trial["pass"]
    ]
    report = {
        "experiment": "variable-height-native-upstream-damping-route-training-v1",
        "passed_replay_gate": decision["pass"],
        "promoted": False,
        "claim": "expanded native damping replay only; no closed-loop hover promotion",
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
            "milestone": baseline_milestone,
            "terminal": baseline_terminal,
        },
        "history": history,
        "accepted_step_efficiency": {
            "motion_improvement_per_step_common_rms": accepted_efficiency,
            "mean": (
                sum(accepted_efficiency) / len(accepted_efficiency) if accepted_efficiency else None
            ),
        },
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
                "baseline_motion_nrmse": baseline_terminal["fixed_scale_nrmse"],
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
