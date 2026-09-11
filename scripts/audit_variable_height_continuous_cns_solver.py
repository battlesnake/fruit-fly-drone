#!/usr/bin/env python3
"""Extend the CNS reference rate and compare a fourth-order recurrent solver."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-continuous-cns-solver-v1"
PROTOCOL_COMMIT = "2f3e46b"
EXPECTED_RATE_REPORT_SHA256 = "ded09b91a114d08039d5334a937e36812b69ba55183634ae4bd21ed9864baff7"
EXPECTED_CACHE_FILE_SHA256 = "68ea98cc60a8969eb38c1bc62db698bb232ddb6ddaebac426a850ee31596ecb8"
EXPECTED_CACHE_SEMANTIC_SHA256 = "43a576a2abc01283a7b5a72560e3c9ff921bfaaf3e5a481ac6715019843f7d0e"
REFERENCE_EXTENSION_SUBSTEPS = (32, 64)
RK4_STEPS = (1, 2, 4)


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
        "--motion-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-throttle-motion-only-001/report.json",
    )
    parser.add_argument(
        "--motion-resume",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-throttle-motion-only-001/resume.pt",
    )
    parser.add_argument(
        "--rate-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-neural-integration-rate-audit-001/report.json",
    )
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-neural-integration-rate-audit-001/input-cache.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-continuous-cns-solver-001",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "camera_and_action_hz": rate.POLICY_HZ,
        "locked_rate_report_sha256": EXPECTED_RATE_REPORT_SHA256,
        "locked_cache_file_sha256": EXPECTED_CACHE_FILE_SHA256,
        "locked_cache_semantic_sha256": EXPECTED_CACHE_SEMANTIC_SHA256,
        "reference_extension": {
            "exponential_euler_substeps": list(REFERENCE_EXTENSION_SUBSTEPS),
            "evaluate_k64_only_if_k16_to_k32_fails": True,
            "adjacent_contrast_rms_over_teacher_scale_maximum": (rate.CONTRAST_REFINEMENT_LIMIT),
            "adjacent_terminal_motor_rms_maximum": rate.MOTOR_REFINEMENT_LIMIT,
        },
        "rk4_candidates": {
            "steps_per_camera_frame": list(RK4_STEPS),
            "recurrent_graph_evaluations_per_frame": [4 * value for value in RK4_STEPS],
            "continuous_equation": ("ds/dt=(5*tanh((R(tanh(s))+bias+sensory_drive)/5)-s)/tau"),
            "sensory_drive_held_within_camera_frame": True,
        },
        "selection": {
            "fewest_graph_evaluations_among_reference-passing_candidates": True,
            "tie_break": "existing exponential-Euler update",
            "behavior_sign_score_or_runtime_used": False,
        },
        "learning": False,
        "new_actor_inputs_or_external_state": False,
        "training_execution_authorized": False,
        "hover_gate_or_promotion_authorized": False,
    }


def validate_locked_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    base_hashes = rate.validate_inputs(args)
    if not args.rate_report.is_file() or not args.input_cache.is_file():
        raise SystemExit("missing terminal rate report or its immutable input cache")
    if assisted.responsibility.file_sha256(args.rate_report) != EXPECTED_RATE_REPORT_SHA256:
        raise SystemExit("terminal rate-report hash mismatch")
    if assisted.responsibility.file_sha256(args.input_cache) != EXPECTED_CACHE_FILE_SHA256:
        raise SystemExit("immutable input-cache file hash mismatch")
    with args.rate_report.open() as stream:
        report = json.load(stream)
    if (
        report.get("classification") != "no_numerically_adequate_rate"
        or report.get("selected_substeps") is not None
        or report.get("input_cache", {}).get("semantic_sha256") != EXPECTED_CACHE_SEMANTIC_SHA256
    ):
        raise SystemExit("terminal rate report is not the registered failed refinement")
    k16 = report.get("conditions", {}).get("16")
    if not isinstance(k16, dict) or k16.get("substeps") != 16:
        raise SystemExit("terminal rate report is missing its K=16 condition")
    cache = torch.load(args.input_cache, map_location="cpu", weights_only=True)
    rate.validate_cache(cache)
    if cache.get("semantic_sha256") != EXPECTED_CACHE_SEMANTIC_SHA256:
        raise SystemExit("immutable input-cache semantic hash mismatch")
    return {**base_hashes, "rate_report": EXPECTED_RATE_REPORT_SHA256}, {
        "report": report,
        "cache": cache,
    }


def classical_rk4_step(
    derivative: Callable[[Tensor], Tensor], state: Tensor, step_seconds: float
) -> Tensor:
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step_seconds * k1)
    k3 = derivative(state + 0.5 * step_seconds * k2)
    k4 = derivative(state + step_seconds * k3)
    return state + (step_seconds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def continuous_state_derivative(
    controller: ConnectomeController, state: Tensor, sensory_drive: Tensor
) -> Tensor:
    activity = torch.tanh(state)
    edge_weight = controller.edge_sign * controller.edge_magnitude
    messages = activity[:, controller.edge_pre] * edge_weight
    recurrent = torch.zeros_like(state).index_add(1, controller.edge_post, messages)
    target = 5.0 * torch.tanh((recurrent + controller.bias + sensory_drive) / 5.0)
    return (target - state) / controller.time_constant


@torch.inference_mode()
def advance_rk4_frame(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    recurrent: Tensor,
    *,
    solver_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    sensory_drive = controller.sensory_drive(image, roll_pitch)
    finite = torch.isfinite(sensory_drive).all()
    step_seconds = 1.0 / (rate.POLICY_HZ * solver_steps)

    def derivative(value: Tensor) -> Tensor:
        nonlocal finite
        finite = finite & torch.isfinite(value).all()
        result = continuous_state_derivative(controller, value, sensory_drive)
        finite = finite & torch.isfinite(result).all()
        return result

    for _ in range(solver_steps):
        recurrent = classical_rk4_step(derivative, recurrent, step_seconds)
        finite = finite & torch.isfinite(recurrent).all()
    output = controller.motor_drive(recurrent)
    finite = finite & torch.isfinite(output).all()
    return output, recurrent, finite


@torch.inference_mode()
def advance_exponential_frame(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    recurrent: Tensor,
    *,
    solver_steps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    return rate._advance_frame(controller, image, roll_pitch, recurrent, substeps=solver_steps)


@torch.inference_mode()
def evaluate_condition(
    *,
    graph: Path,
    checkpoint_state: dict[str, Tensor],
    cache: dict[str, Any],
    method: str,
    solver_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if method == "exponential_euler":
        if solver_steps not in REFERENCE_EXTENSION_SUBSTEPS:
            raise ValueError("exponential-Euler steps are outside the extension")
        neural_dt = 1.0 / (rate.POLICY_HZ * solver_steps)
        graph_evaluations_per_frame = solver_steps
        advance = advance_exponential_frame
    elif method == "rk4":
        if solver_steps not in RK4_STEPS:
            raise ValueError("RK4 steps are outside the preregistered candidates")
        neural_dt = 1.0 / rate.POLICY_HZ
        graph_evaluations_per_frame = 4 * solver_steps
        advance = advance_rk4_frame
    else:
        raise ValueError(f"unknown solver method: {method}")

    controller = ConnectomeController(graph, neural_dt=neural_dt).to(device)
    controller.load_state_dict(checkpoint_state, strict=True)
    controller.eval().requires_grad_(False)
    checkpoint_sha256 = assisted.audit.semantic_sha256(checkpoint_state)
    source_before = rate._source_state_sha256(controller)
    source_loaded_exactly = source_before == checkpoint_sha256
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()

    prefix_images = cache["prefix_images"].to(device)
    prefix_roll_pitch = cache["prefix_roll_pitch"].to(device)
    recurrent = controller.initial_state(
        int(cache["cases"]), device=device, dtype=prefix_images.dtype
    )
    finite = torch.ones((), dtype=torch.bool, device=device)
    for _ in range(int(cache["prefix_repetitions"])):
        _, recurrent, frame_finite = advance(
            controller,
            prefix_images,
            prefix_roll_pitch,
            recurrent,
            solver_steps=solver_steps,
        )
        finite = finite & frame_finite

    terminal_outputs = torch.empty(int(cache["cases"]), 2, 4, device=device)
    for horizon in motion.HISTORY_LENGTHS:
        group = cache["groups"][str(horizon)]
        case_indices = group["case_indices"].to(device)
        branch_recurrent = recurrent[case_indices].repeat_interleave(2, dim=0)
        roll_pitch = group["roll_pitch"].reshape(-1, 2).to(device)
        images = group["response_images"]
        group_output = torch.zeros(len(case_indices) * 2, 4, device=device)
        for step in range(horizon):
            image = images[step].reshape(-1, *images.shape[3:]).to(device)
            group_output, branch_recurrent, frame_finite = advance(
                controller,
                image,
                roll_pitch,
                branch_recurrent,
                solver_steps=solver_steps,
            )
            finite = finite & frame_finite
        terminal_outputs[case_indices] = group_output.reshape(-1, 2, 4)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = perf_counter() - started
    terminal_outputs_cpu = terminal_outputs.detach().cpu()
    recurrence_finite = bool(finite.detach().cpu())
    source_after = rate._source_state_sha256(controller)
    physical_batch_frames = int(cache["prefix_repetitions"]) + sum(motion.HISTORY_LENGTHS)
    branch_frames = int(cache["prefix_repetitions"]) * int(cache["cases"])
    branch_frames += sum(
        horizon * len(cache["groups"][str(horizon)]["case_indices"]) * 2
        for horizon in motion.HISTORY_LENGTHS
    )
    summary = rate.summarize_outputs(
        terminal_outputs_cpu,
        cache["teacher_targets"],
        cache["horizon"],
        scale=float(cache["teacher_scale"]),
        recurrence_finite=recurrence_finite,
        endpoint_difference_max=float(cache["endpoint_image_difference"].max()),
        wall_time_seconds=wall_time,
        recurrent_calls=physical_batch_frames * graph_evaluations_per_frame,
        branch_state_updates=branch_frames * graph_evaluations_per_frame,
    )
    summary.update(
        {
            "method": method,
            "solver_steps_per_camera_frame": solver_steps,
            "solver_step_seconds": 1.0 / (rate.POLICY_HZ * solver_steps),
            "graph_evaluations_per_camera_frame": graph_evaluations_per_frame,
            "source_state_sha256_before": source_before,
            "source_state_sha256_after": source_after,
            "source_loaded_exactly": source_loaded_exactly,
            "source_restored": source_before == source_after,
        }
    )
    if method == "exponential_euler":
        summary["substeps"] = solver_steps
        summary["neural_dt_seconds"] = neural_dt
    del controller
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def reference_comparison(
    candidate_name: str,
    candidate: dict[str, Any],
    reference_name: str,
    reference: dict[str, Any],
    *,
    scale: float,
) -> dict[str, Any]:
    candidate_contrast = torch.tensor(candidate["prediction_contrasts"], dtype=torch.float64)
    reference_contrast = torch.tensor(reference["prediction_contrasts"], dtype=torch.float64)
    candidate_outputs = torch.tensor(candidate["terminal_motor_outputs"], dtype=torch.float64)
    reference_outputs = torch.tensor(reference["terminal_motor_outputs"], dtype=torch.float64)
    contrast_rms = float((candidate_contrast - reference_contrast).square().mean().sqrt())
    motor_rms = float((candidate_outputs - reference_outputs).square().mean().sqrt())
    identities_pass = bool(
        candidate["source_loaded_exactly"]
        and candidate["source_restored"]
        and candidate["all_recurrent_states_and_outputs_finite"]
        and candidate["all_metrics_finite"]
        and candidate["endpoint_image_difference_max"] == 0.0
        and reference["source_loaded_exactly"]
        and reference["source_restored"]
        and reference["all_recurrent_states_and_outputs_finite"]
        and reference["all_metrics_finite"]
        and reference["endpoint_image_difference_max"] == 0.0
    )
    normalized_contrast = contrast_rms / scale
    return {
        "candidate": candidate_name,
        "reference": reference_name,
        "contrast_rms_difference_motor_units": contrast_rms,
        "contrast_rms_difference_over_teacher_scale": normalized_contrast,
        "terminal_motor_rms_difference": motor_rms,
        "identity_finiteness_and_endpoint_pass": identities_pass,
        "pass": bool(
            identities_pass
            and normalized_contrast <= rate.CONTRAST_REFINEMENT_LIMIT
            and motor_rms <= rate.MOTOR_REFINEMENT_LIMIT
        ),
    }


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [candidate for candidate in candidates if candidate["comparison"]["pass"]]
    if not passing:
        return None
    return min(
        passing,
        key=lambda candidate: (
            candidate["graph_evaluations_per_camera_frame"],
            0 if candidate["method"] == "exponential_euler" else 1,
        ),
    )


def _write_start_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise SystemExit(
            "solver-audit start marker already exists; this declared run cannot be replayed"
        ) from exc


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit("the continuous-CNS solver extension already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes, locked = validate_locked_inputs(args)
    prior_report, cache = locked["report"], locked["cache"]
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_state = loaded["controller"]
    checkpoint_semantic_sha256 = assisted.audit.semantic_sha256(checkpoint_state)
    _write_start_marker(
        args.output_dir / "start.json",
        {
            "experiment": EXPERIMENT,
            "protocol": protocol_manifest(),
            "input_file_sha256": input_hashes,
            "cache_file_sha256": EXPECTED_CACHE_FILE_SHA256,
            "cache_semantic_sha256": EXPECTED_CACHE_SEMANTIC_SHA256,
            "source_checkpoint_semantic_sha256": checkpoint_semantic_sha256,
            "device": str(device),
        },
    )
    started = perf_counter()
    conditions: dict[str, Any] = {}

    print(json.dumps({"stage": "exponential_reference", "substeps": 32}), flush=True)
    conditions["exponential_euler_k32"] = evaluate_condition(
        graph=args.graph,
        checkpoint_state=checkpoint_state,
        cache=cache,
        method="exponential_euler",
        solver_steps=32,
        device=device,
    )
    first_refinement = rate.adjacent_refinement(
        prior_report["conditions"]["16"],
        conditions["exponential_euler_k32"],
        scale=rate.EXPECTED_MOTION_SCALE,
    )
    reference_refinements = [first_refinement]
    if first_refinement["pass"]:
        reference_name = "exponential_euler_k32"
        reference = conditions[reference_name]
        adequate_exponential_name = "exponential_euler_k16"
        adequate_exponential = prior_report["conditions"]["16"]
    else:
        print(json.dumps({"stage": "exponential_reference", "substeps": 64}), flush=True)
        conditions["exponential_euler_k64"] = evaluate_condition(
            graph=args.graph,
            checkpoint_state=checkpoint_state,
            cache=cache,
            method="exponential_euler",
            solver_steps=64,
            device=device,
        )
        second_refinement = rate.adjacent_refinement(
            conditions["exponential_euler_k32"],
            conditions["exponential_euler_k64"],
            scale=rate.EXPECTED_MOTION_SCALE,
        )
        reference_refinements.append(second_refinement)
        if second_refinement["pass"]:
            reference_name = "exponential_euler_k64"
            reference = conditions[reference_name]
            adequate_exponential_name = "exponential_euler_k32"
            adequate_exponential = conditions[adequate_exponential_name]
        else:
            report = {
                "experiment": EXPERIMENT,
                "protocol_commit": PROTOCOL_COMMIT,
                "protocol": protocol_manifest(),
                "classification": "fine_reference_not_established",
                "passed": False,
                "input_file_sha256": input_hashes,
                "source_checkpoint_semantic_sha256": checkpoint_semantic_sha256,
                "cache_file_sha256": EXPECTED_CACHE_FILE_SHA256,
                "cache_semantic_sha256": EXPECTED_CACHE_SEMANTIC_SHA256,
                "conditions": conditions,
                "reference_refinements": reference_refinements,
                "reference_condition": None,
                "rk4_comparisons": [],
                "selected_solver": None,
                "behavior_used_for_selection": False,
                "future_training_may_use_selected_solver": False,
                "training_execution_authorized": False,
                "hover_or_gate_flight_authorized": False,
                "promotion_authorized": False,
                "wall_time_seconds": perf_counter() - started,
            }
            assisted._atomic_json_save(report, args.output_dir / "report.json")
            print(json.dumps({"classification": report["classification"], "pass": False}))
            return 0

    comparisons = []
    candidates = []
    exponential_comparison = reference_comparison(
        adequate_exponential_name,
        adequate_exponential,
        reference_name,
        reference,
        scale=rate.EXPECTED_MOTION_SCALE,
    )
    candidates.append(
        {
            "name": adequate_exponential_name,
            "method": "exponential_euler",
            "graph_evaluations_per_camera_frame": int(
                adequate_exponential["substeps"]
                if "substeps" in adequate_exponential
                else adequate_exponential["solver_steps_per_camera_frame"]
            ),
            "comparison": exponential_comparison,
        }
    )
    comparisons.append(exponential_comparison)
    for solver_steps in RK4_STEPS:
        name = f"rk4_m{solver_steps}"
        print(
            json.dumps({"stage": "rk4_candidate", "solver_steps": solver_steps}),
            flush=True,
        )
        conditions[name] = evaluate_condition(
            graph=args.graph,
            checkpoint_state=checkpoint_state,
            cache=cache,
            method="rk4",
            solver_steps=solver_steps,
            device=device,
        )
        comparison = reference_comparison(
            name,
            conditions[name],
            reference_name,
            reference,
            scale=rate.EXPECTED_MOTION_SCALE,
        )
        comparisons.append(comparison)
        candidates.append(
            {
                "name": name,
                "method": "rk4",
                "graph_evaluations_per_camera_frame": 4 * solver_steps,
                "comparison": comparison,
            }
        )
    selected = select_candidate(candidates)
    classification = (
        "numerically_adequate_solver_selected"
        if selected is not None
        else "no_numerically_adequate_solver"
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": selected is not None,
        "input_file_sha256": input_hashes,
        "source_checkpoint_semantic_sha256": checkpoint_semantic_sha256,
        "cache_file_sha256": EXPECTED_CACHE_FILE_SHA256,
        "cache_semantic_sha256": EXPECTED_CACHE_SEMANTIC_SHA256,
        "conditions": conditions,
        "reference_refinements": reference_refinements,
        "reference_condition": reference_name,
        "adequate_exponential_candidate": adequate_exponential_name,
        "candidate_comparisons": comparisons,
        "selected_solver": selected,
        "behavior_used_for_selection": False,
        "future_training_may_use_selected_solver": selected is not None,
        "training_execution_authorized": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    assisted._atomic_json_save(report, args.output_dir / "report.json")
    print(
        json.dumps(
            {
                "output": assisted.responsibility.stable_path(args.output_dir / "report.json"),
                "classification": classification,
                "pass": report["passed"],
                "reference_condition": reference_name,
                "selected_solver": selected,
                "behavior_used_for_selection": False,
                "training_execution_authorized": False,
                "hover_or_gate_flight_authorized": False,
                "promotion_authorized": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
