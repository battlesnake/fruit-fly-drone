#!/usr/bin/env python3
"""Audit native CNS integration substeps on the frozen opposite-motion bank."""

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

import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402

EXPERIMENT = "variable-height-neural-integration-rate-audit-v1"
PROTOCOL_COMMIT = "e0d07a9"
EXPECTED_REPORT_SHA256 = "ad0f4f0524be5f9a769b74d890d8f8b6925ba25921bb283ac52b8089ddc22c5c"
EXPECTED_RESUME_SHA256 = "04c4b4e0b598f37c3808a7740bad0160c2826d20584149c841694696d1d0e28d"
EXPECTED_BANK_SHA256 = "0887ead1e9eef7755c47b51967f55bbde0e7203b17e76ea9f50456e97ca530fc"
EXPECTED_MOTION_SCALE = 0.043661270290613174
EXPECTED_SOURCE = {
    "nrmse": 1.031749290796319,
    "teacher_aligned_gain": -0.030875112861394882,
    "prediction_rms_motor_units": 0.002292240969836712,
}

POLICY_HZ = 50
SUBSTEPS = (1, 2, 4, 8, 16)
K1_ABSOLUTE_TOLERANCE = 2.0e-5
CONTRAST_REFINEMENT_LIMIT = 0.01
MOTOR_REFINEMENT_LIMIT = 0.005
AXIS_NAMES = ("roll", "pitch", "yaw", "throttle")


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
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-neural-integration-rate-audit-001",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source": "original paired-dynamic-001 controller",
        "camera_and_action_hz": POLICY_HZ,
        "internal_substeps": list(SUBSTEPS),
        "neural_dt_seconds": {str(k): 1.0 / (POLICY_HZ * k) for k in SUBSTEPS},
        "inputs": ["cached 320x200 linear RGB", "cached roll", "cached pitch"],
        "privileged_inputs": [],
        "external_history_or_state": False,
        "learning": False,
        "aircraft_advanced_during_substeps": False,
        "condition_selection": {
            "criterion": "smallest K passing adjacent K-to-2K numerical refinement",
            "contrast_rms_over_frozen_teacher_scale_maximum": CONTRAST_REFINEMENT_LIMIT,
            "terminal_motor_rms_maximum": MOTOR_REFINEMENT_LIMIT,
            "behavior_metrics_used_for_selection": False,
        },
        "selected_rate_may_be_used_by_future_training": (
            "only in a separately preregistered experiment when K > 1"
        ),
        "authorizes_training_execution": False,
        "authorizes_hover_or_gate_flight": False,
        "promotion": False,
    }


def _file_sha256(path: Path) -> str:
    return assisted.responsibility.file_sha256(path)


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    expected = {
        args.graph: assisted.EXPECTED_GRAPH_SHA256,
        args.checkpoint: assisted.EXPECTED_CHECKPOINT_SHA256,
        args.motion_report: EXPECTED_REPORT_SHA256,
        args.motion_resume: EXPECTED_RESUME_SHA256,
    }
    observed: dict[str, str] = {}
    for path, expected_sha256 in expected.items():
        if not path.is_file():
            raise SystemExit(f"missing preregistered input: {path}")
        actual = _file_sha256(path)
        if actual != expected_sha256:
            raise SystemExit(f"preregistered input hash mismatch: {path}")
        observed[assisted.responsibility.stable_path(path)] = actual
    return observed


def load_frozen_bank(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    motion._validate_resume(payload)
    bank = payload["training_bank"]
    if bank.get("sha256") != EXPECTED_BANK_SHA256:
        raise SystemExit("motion-only training-bank hash does not match the preregistration")
    if motion._payload_hash(bank) != EXPECTED_BANK_SHA256:
        raise SystemExit("motion-only training-bank semantic hash is invalid")
    scale = payload["objective_scale"]
    if scale.get("motion") != EXPECTED_MOTION_SCALE:
        raise SystemExit("motion-only teacher-contrast scale does not match the preregistration")
    if payload.get("accepted_updates") != 17 or payload.get("run_state") != "stopped":
        raise SystemExit("motion-only source run is not the registered stopped update-17 run")
    return bank, scale


def _cache_without_hash(cache: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in cache.items() if name != "semantic_sha256"}


@torch.inference_mode()
def build_input_cache(
    bank: dict[str, Any], *, device: torch.device, config: HoverConfig
) -> dict[str, Any]:
    cases = int(bank["cases"])
    all_indices_cpu = torch.arange(cases, dtype=torch.long)
    base_state = assisted._state_from_dict(bank["state"], device=device, indices=all_indices_cpu)
    base_scene = assisted._scene_from_dict(bank["scene"], device=device, indices=all_indices_cpu)
    neutral_marker = base_state.position[:, 2]
    prefix_images = (
        render_visual_hover_scene(base_state, neutral_marker, scene=base_scene)
        .detach()
        .cpu()
        .contiguous()
    )
    prefix_roll_pitch = base_state.euler[:, :2].detach().cpu().contiguous()
    targets = assisted.teacher_motion_contrasts(bank, config=config).detach().cpu().contiguous()

    groups: dict[str, Any] = {}
    endpoint_differences = torch.empty(cases, dtype=torch.float32)
    for horizon in motion.HISTORY_LENGTHS:
        case_indices_cpu = torch.nonzero(bank["horizon"] == horizon).flatten()
        state = assisted._state_from_dict(bank["state"], device=device, indices=case_indices_cpu)
        scene = assisted._scene_from_dict(bank["scene"], device=device, indices=case_indices_cpu)
        height = state.position[:, 2]
        marker = height + (
            bank["height_sign"][case_indices_cpu].to(device)
            * bank["height_error"][case_indices_cpu].to(device)
        )
        speed = bank["speed"][case_indices_cpu].to(device)
        approach = bank["approach_steps"][case_indices_cpu].to(device)
        frames: list[Tensor] = []
        for step in range(horizon):
            pair: list[Tensor] = []
            for velocity_sign in assisted.MOTION_VELOCITY_SIGNS:
                offset = assisted.factorial.smooth_return_offset(
                    velocity_sign * speed,
                    step=step,
                    total_steps=horizon,
                    approach_steps=approach,
                    policy_hz=POLICY_HZ,
                )
                branch = assisted.responsibility.state_at_height(state, height + offset)
                pair.append(render_visual_hover_scene(branch, marker, scene=scene))
            frames.append(torch.stack(pair, dim=1).detach().cpu().contiguous())
        images = torch.stack(frames)
        endpoint = images[-1]
        endpoint_difference = (endpoint[:, 0] - endpoint[:, 1]).abs().flatten(1).amax(1)
        endpoint_differences[case_indices_cpu] = endpoint_difference
        roll_pitch = (
            state.euler[:, None, :2]
            .expand(-1, len(assisted.MOTION_VELOCITY_SIGNS), -1)
            .detach()
            .cpu()
            .contiguous()
        )
        groups[str(horizon)] = {
            "case_indices": case_indices_cpu,
            "response_images": images,
            "roll_pitch": roll_pitch,
        }

    cache: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "source_bank_sha256": EXPECTED_BANK_SHA256,
        "cases": cases,
        "case_order": all_indices_cpu,
        "prefix_images": prefix_images,
        "prefix_roll_pitch": prefix_roll_pitch,
        "prefix_repetitions": motion.PREFIX_STEPS,
        "groups": groups,
        "teacher_targets": targets,
        "teacher_scale": EXPECTED_MOTION_SCALE,
        "horizon": bank["horizon"].detach().cpu().contiguous(),
        "height_error": bank["height_error"].detach().cpu().contiguous(),
        "height_sign": bank["height_sign"].detach().cpu().contiguous(),
        "speed": bank["speed"].detach().cpu().contiguous(),
        "endpoint_image_difference": endpoint_differences,
    }
    cache["semantic_sha256"] = assisted.audit.semantic_sha256(_cache_without_hash(cache))
    return cache


def validate_cache(cache: dict[str, Any]) -> None:
    if cache.get("experiment") != EXPERIMENT:
        raise SystemExit("input cache experiment mismatch")
    if cache.get("source_bank_sha256") != EXPECTED_BANK_SHA256:
        raise SystemExit("input cache bank mismatch")
    if assisted.audit.semantic_sha256(_cache_without_hash(cache)) != cache.get("semantic_sha256"):
        raise SystemExit("input cache semantic hash mismatch")
    if cache.get("cases") != motion.CASES:
        raise SystemExit("input cache case count mismatch")
    if cache.get("prefix_repetitions") != motion.PREFIX_STEPS:
        raise SystemExit("input cache prefix length mismatch")
    if cache.get("teacher_scale") != EXPECTED_MOTION_SCALE:
        raise SystemExit("input cache teacher scale mismatch")
    if not torch.equal(cache["case_order"], torch.arange(motion.CASES)):
        raise SystemExit("input cache case order mismatch")
    if cache["prefix_images"].shape != (motion.CASES, 3, 200, 320):
        raise SystemExit("input cache prefix image shape mismatch")
    if cache["prefix_roll_pitch"].shape != (motion.CASES, 2):
        raise SystemExit("input cache prefix attitude shape mismatch")
    if cache["teacher_targets"].shape != (motion.CASES,):
        raise SystemExit("input cache teacher-target shape mismatch")
    target_rms = float(cache["teacher_targets"].square().mean().sqrt())
    if abs(target_rms - EXPECTED_MOTION_SCALE) > 1.0e-8:
        raise SystemExit("input cache teacher-target scale mismatch")
    if set(cache["groups"]) != {str(value) for value in motion.HISTORY_LENGTHS}:
        raise SystemExit("input cache horizon groups mismatch")
    observed = (
        torch.cat([cache["groups"][str(value)]["case_indices"] for value in motion.HISTORY_LENGTHS])
        .sort()
        .values
    )
    if not torch.equal(observed, torch.arange(motion.CASES)):
        raise SystemExit("input cache horizon groups do not partition the bank")
    if not bool(torch.isfinite(cache["prefix_images"]).all()):
        raise SystemExit("input cache prefix contains nonfinite pixels")
    if not bool(torch.isfinite(cache["prefix_roll_pitch"]).all()):
        raise SystemExit("input cache prefix attitude is nonfinite")
    if not bool(torch.isfinite(cache["teacher_targets"]).all()):
        raise SystemExit("input cache targets are nonfinite")
    for horizon in motion.HISTORY_LENGTHS:
        group = cache["groups"][str(horizon)]
        pairs = int((cache["horizon"] == horizon).sum())
        if group["response_images"].shape != (horizon, pairs, 2, 3, 200, 320):
            raise SystemExit(f"input cache horizon-{horizon} image shape mismatch")
        if group["roll_pitch"].shape != (pairs, 2, 2):
            raise SystemExit(f"input cache horizon-{horizon} attitude shape mismatch")
        if not bool(torch.isfinite(group["response_images"]).all()):
            raise SystemExit(f"input cache horizon-{horizon} images are nonfinite")
        if not bool(torch.isfinite(group["roll_pitch"]).all()):
            raise SystemExit(f"input cache horizon-{horizon} attitude is nonfinite")
    if float(cache["endpoint_image_difference"].max()) != 0.0:
        raise SystemExit("input cache opposite-motion endpoints are not bit-identical")


def _source_state_sha256(controller: ConnectomeController) -> str:
    return assisted.audit.semantic_sha256(controller.state_dict())


@torch.inference_mode()
def _advance_frame(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    recurrent: Tensor,
    *,
    substeps: int,
) -> tuple[Tensor, Tensor, Tensor]:
    finite = torch.ones((), dtype=torch.bool, device=recurrent.device)
    output = torch.zeros(recurrent.shape[0], 4, dtype=recurrent.dtype, device=recurrent.device)
    for _ in range(substeps):
        output, recurrent = controller(image, roll_pitch, recurrent)
        finite = finite & torch.isfinite(output).all() & torch.isfinite(recurrent).all()
    return output, recurrent, finite


def summarize_outputs(
    outputs: Tensor,
    targets: Tensor,
    horizons: Tensor,
    *,
    scale: float,
    recurrence_finite: bool,
    endpoint_difference_max: float,
    wall_time_seconds: float,
    recurrent_calls: int,
    branch_state_updates: int,
) -> dict[str, Any]:
    predictions = outputs[:, 1, 3] - outputs[:, 0, 3]
    losses = ((predictions - targets) / scale).square()
    target_power = targets.square().sum().clamp_min(1.0e-12)
    by_horizon: dict[str, Any] = {}
    for horizon in motion.HISTORY_LENGTHS:
        selected = horizons == horizon
        prediction = predictions[selected]
        target = targets[selected]
        power = target.square().sum().clamp_min(1.0e-12)
        by_horizon[str(horizon)] = {
            "pairs": int(selected.sum()),
            "prediction_contrasts": prediction.tolist(),
            "teacher_contrasts": target.tolist(),
            "correct_sign_fraction": float(((prediction * target) > 0).float().mean()),
            "teacher_aligned_gain": float((prediction * target).sum() / power),
            "nrmse": float(losses[selected].mean().sqrt()),
            "prediction_rms_motor_units": float(prediction.square().mean().sqrt()),
        }
    flat_outputs = outputs.reshape(-1, 4)
    motor_axis_summary: dict[str, Any] = {}
    for index, name in enumerate(AXIS_NAMES):
        values = flat_outputs[:, index]
        motor_axis_summary[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "rms": float(values.square().mean().sqrt()),
        }
    pair_common = outputs[:, :, 3].mean(dim=1)
    result: dict[str, Any] = {
        "pairs": len(predictions),
        "prediction_contrasts": predictions.tolist(),
        "teacher_contrasts": targets.tolist(),
        "terminal_motor_outputs": outputs.tolist(),
        "throttle_pair_means": pair_common.tolist(),
        "correct_sign_fraction": float(((predictions * targets) > 0).float().mean()),
        "teacher_aligned_gain": float((predictions * targets).sum() / target_power),
        "nrmse": float(losses.mean().sqrt()),
        "prediction_rms_motor_units": float(predictions.square().mean().sqrt()),
        "target_rms_motor_units": float(targets.square().mean().sqrt()),
        "pair_common_throttle_rms": float(pair_common.square().mean().sqrt()),
        "pair_common_throttle_minimum": float(pair_common.min()),
        "pair_common_throttle_maximum": float(pair_common.max()),
        "motor_axis_summary": motor_axis_summary,
        "by_horizon": by_horizon,
        "all_recurrent_states_and_outputs_finite": recurrence_finite,
        "endpoint_image_difference_max": endpoint_difference_max,
        "wall_time_seconds": wall_time_seconds,
        "native_recurrent_calls": recurrent_calls,
        "native_branch_state_updates": branch_state_updates,
    }
    result["all_metrics_finite"] = assisted._all_finite_nested(result)
    return result


@torch.inference_mode()
def evaluate_condition(
    *,
    graph: Path,
    checkpoint_state: dict[str, Tensor],
    cache: dict[str, Any],
    substeps: int,
    device: torch.device,
) -> dict[str, Any]:
    if substeps not in SUBSTEPS:
        raise ValueError("substeps are outside the preregistered conditions")
    controller = ConnectomeController(graph, neural_dt=1.0 / (POLICY_HZ * substeps)).to(device)
    controller.load_state_dict(checkpoint_state, strict=True)
    controller.eval().requires_grad_(False)
    source_before = _source_state_sha256(controller)
    source_loaded_exactly = source_before == assisted.audit.semantic_sha256(checkpoint_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()

    prefix_images = cache["prefix_images"].to(device)
    prefix_roll_pitch = cache["prefix_roll_pitch"].to(device)
    recurrent = controller.initial_state(
        int(cache["cases"]), device=device, dtype=prefix_images.dtype
    )
    finite = torch.ones((), dtype=torch.bool, device=device)
    output = torch.zeros(int(cache["cases"]), 4, device=device)
    for _ in range(int(cache["prefix_repetitions"])):
        output, recurrent, frame_finite = _advance_frame(
            controller,
            prefix_images,
            prefix_roll_pitch,
            recurrent,
            substeps=substeps,
        )
        finite = finite & frame_finite

    terminal_outputs = torch.empty(int(cache["cases"]), 2, 4, device=device)
    response_branch_updates = 0
    for horizon in motion.HISTORY_LENGTHS:
        group = cache["groups"][str(horizon)]
        case_indices_cpu = group["case_indices"]
        case_indices = case_indices_cpu.to(device)
        branch_recurrent = recurrent[case_indices].repeat_interleave(2, dim=0)
        roll_pitch = group["roll_pitch"].reshape(-1, 2).to(device)
        images = group["response_images"]
        group_output = torch.zeros(len(case_indices) * 2, 4, device=device)
        for step in range(horizon):
            image = images[step].reshape(-1, *images.shape[3:]).to(device)
            group_output, branch_recurrent, frame_finite = _advance_frame(
                controller,
                image,
                roll_pitch,
                branch_recurrent,
                substeps=substeps,
            )
            finite = finite & frame_finite
        terminal_outputs[case_indices] = group_output.reshape(-1, 2, 4)
        response_branch_updates += horizon * len(case_indices) * 2 * substeps

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_time = perf_counter() - started
    terminal_outputs_cpu = terminal_outputs.detach().cpu()
    recurrence_finite = bool(finite.detach().cpu())
    source_after = _source_state_sha256(controller)
    source_restored = source_after == source_before
    prefix_branch_updates = int(cache["prefix_repetitions"]) * int(cache["cases"]) * substeps
    recurrent_calls = substeps * (int(cache["prefix_repetitions"]) + sum(motion.HISTORY_LENGTHS))
    summary = summarize_outputs(
        terminal_outputs_cpu,
        cache["teacher_targets"],
        cache["horizon"],
        scale=float(cache["teacher_scale"]),
        recurrence_finite=recurrence_finite,
        endpoint_difference_max=float(cache["endpoint_image_difference"].max()),
        wall_time_seconds=wall_time,
        recurrent_calls=recurrent_calls,
        branch_state_updates=prefix_branch_updates + response_branch_updates,
    )
    summary.update(
        {
            "substeps": substeps,
            "neural_dt_seconds": controller.neural_dt,
            "source_state_sha256_before": source_before,
            "source_state_sha256_after": source_after,
            "source_loaded_exactly": source_loaded_exactly,
            "source_restored": source_restored,
        }
    )
    del controller
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def k1_reproduction_decision(result: dict[str, Any]) -> dict[str, Any]:
    differences = {
        name: abs(float(result[name]) - expected) for name, expected in EXPECTED_SOURCE.items()
    }
    reasons = [
        f"{name} differs by {difference:.9g}"
        for name, difference in differences.items()
        if difference > K1_ABSOLUTE_TOLERANCE
    ]
    if result["endpoint_image_difference_max"] != 0.0:
        reasons.append("paired endpoint images were not bit-identical")
    if not result["all_recurrent_states_and_outputs_finite"]:
        reasons.append("K=1 recurrence or outputs were nonfinite")
    if not result["all_metrics_finite"]:
        reasons.append("K=1 metrics were nonfinite")
    if not result["source_restored"]:
        reasons.append("K=1 changed the source controller")
    if not result["source_loaded_exactly"]:
        reasons.append("K=1 did not load the exact source controller")
    return {
        "pass": not reasons,
        "absolute_tolerance": K1_ABSOLUTE_TOLERANCE,
        "absolute_differences": differences,
        "reasons": reasons,
    }


def adjacent_refinement(
    coarse: dict[str, Any], fine: dict[str, Any], *, scale: float
) -> dict[str, Any]:
    coarse_contrast = torch.tensor(coarse["prediction_contrasts"], dtype=torch.float64)
    fine_contrast = torch.tensor(fine["prediction_contrasts"], dtype=torch.float64)
    coarse_output = torch.tensor(coarse["terminal_motor_outputs"], dtype=torch.float64)
    fine_output = torch.tensor(fine["terminal_motor_outputs"], dtype=torch.float64)
    contrast_difference = float((coarse_contrast - fine_contrast).square().mean().sqrt())
    motor_difference = float((coarse_output - fine_output).square().mean().sqrt())
    finite_and_exact = bool(
        coarse["all_recurrent_states_and_outputs_finite"]
        and fine["all_recurrent_states_and_outputs_finite"]
        and coarse["all_metrics_finite"]
        and fine["all_metrics_finite"]
        and coarse["endpoint_image_difference_max"] == 0.0
        and fine["endpoint_image_difference_max"] == 0.0
        and coarse["source_restored"]
        and fine["source_restored"]
        and coarse["source_loaded_exactly"]
        and fine["source_loaded_exactly"]
    )
    normalized_contrast = contrast_difference / scale
    return {
        "coarse_substeps": int(coarse["substeps"]),
        "fine_substeps": int(fine["substeps"]),
        "contrast_rms_difference_motor_units": contrast_difference,
        "contrast_rms_difference_over_teacher_scale": normalized_contrast,
        "terminal_motor_rms_difference": motor_difference,
        "finite_exact_and_source_restored": finite_and_exact,
        "pass": bool(
            finite_and_exact
            and normalized_contrast <= CONTRAST_REFINEMENT_LIMIT
            and motor_difference <= MOTOR_REFINEMENT_LIMIT
        ),
    }


def select_numerically_adequate_rate(
    refinements: list[dict[str, Any]],
) -> int | None:
    for refinement in refinements:
        if refinement["pass"]:
            return int(refinement["coarse_substeps"])
    return None


def _write_start_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise SystemExit(
            "audit start marker already exists; this declared run cannot be replayed"
        ) from exc


def _write_terminal_report(path: Path, payload: dict[str, Any]) -> None:
    assisted._atomic_json_save(payload, path)


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit("the neural-integration-rate audit already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes = validate_inputs(args)
    bank, scale = load_frozen_bank(args.motion_resume)
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if loaded.get("graph_sha256") != assisted.EXPECTED_GRAPH_SHA256:
        raise SystemExit("source checkpoint graph hash mismatch")
    checkpoint_state = loaded["controller"]
    checkpoint_semantic_sha256 = assisted.audit.semantic_sha256(checkpoint_state)

    _write_start_marker(
        args.output_dir / "start.json",
        {
            "experiment": EXPERIMENT,
            "protocol": protocol_manifest(),
            "input_file_sha256": input_hashes,
            "source_checkpoint_semantic_sha256": checkpoint_semantic_sha256,
            "device": str(device),
        },
    )
    print(json.dumps({"stage": "rendering_immutable_input_cache"}), flush=True)
    cache = build_input_cache(bank, device=device, config=HoverConfig())
    validate_cache(cache)
    cache_path = args.output_dir / "input-cache.pt"
    assisted._atomic_torch_save(cache, cache_path)
    cache_file_sha256 = _file_sha256(cache_path)
    reloaded_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    validate_cache(reloaded_cache)
    if reloaded_cache["semantic_sha256"] != cache["semantic_sha256"]:
        raise SystemExit("reloaded input cache identity mismatch")
    del cache, bank

    conditions: dict[str, Any] = {}
    started = perf_counter()
    print(json.dumps({"stage": "evaluating_condition", "substeps": 1}), flush=True)
    conditions["1"] = evaluate_condition(
        graph=args.graph,
        checkpoint_state=checkpoint_state,
        cache=reloaded_cache,
        substeps=1,
        device=device,
    )
    k1_decision = k1_reproduction_decision(conditions["1"])
    common: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_checkpoint_semantic_sha256": checkpoint_semantic_sha256,
        "motion_bank_sha256": EXPECTED_BANK_SHA256,
        "teacher_contrast_scale": scale,
        "input_cache": {
            "path": assisted.responsibility.stable_path(cache_path),
            "file_sha256": cache_file_sha256,
            "semantic_sha256": reloaded_cache["semantic_sha256"],
            "endpoint_image_difference_max": float(
                reloaded_cache["endpoint_image_difference"].max()
            ),
        },
        "k1_reproduction": k1_decision,
    }
    if not k1_decision["pass"]:
        report = {
            **common,
            "classification": "k1_source_reproduction_failed",
            "passed": False,
            "conditions": conditions,
            "adjacent_refinements": [],
            "selected_substeps": None,
            "integration_coarseness_supported": False,
            "future_training_may_use_selected_rate": False,
            "training_execution_authorized": False,
            "hover_or_gate_flight_authorized": False,
            "promotion_authorized": False,
            "wall_time_seconds": perf_counter() - started,
        }
        _write_terminal_report(args.output_dir / "report.json", report)
        print(json.dumps({"classification": report["classification"], "pass": False}), flush=True)
        return 0

    for substeps in SUBSTEPS[1:]:
        print(
            json.dumps({"stage": "evaluating_condition", "substeps": substeps}),
            flush=True,
        )
        conditions[str(substeps)] = evaluate_condition(
            graph=args.graph,
            checkpoint_state=checkpoint_state,
            cache=reloaded_cache,
            substeps=substeps,
            device=device,
        )
    refinements = [
        adjacent_refinement(
            conditions[str(coarse)],
            conditions[str(2 * coarse)],
            scale=EXPECTED_MOTION_SCALE,
        )
        for coarse in SUBSTEPS[:-1]
    ]
    selected = select_numerically_adequate_rate(refinements)
    classification = (
        "numerically_adequate_rate_selected"
        if selected is not None
        else "no_numerically_adequate_rate"
    )
    graded_rate_for_future_training = selected is not None and selected > 1
    report = {
        **common,
        "classification": classification,
        "passed": selected is not None,
        "conditions": conditions,
        "adjacent_refinements": refinements,
        "selected_substeps": selected,
        "selected_neural_hz": None if selected is None else POLICY_HZ * selected,
        "integration_coarseness_supported": selected is not None and selected > 1,
        "integration_coarseness_rejected": selected == 1,
        "behavior_used_for_selection": False,
        "future_training_may_use_selected_rate": graded_rate_for_future_training,
        "training_execution_authorized": False,
        "training_requires_separate_preregistration": True,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    _write_terminal_report(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {
                "output": assisted.responsibility.stable_path(args.output_dir / "report.json"),
                "classification": classification,
                "pass": report["passed"],
                "selected_substeps": selected,
                "integration_coarseness_supported": report["integration_coarseness_supported"],
                "future_training_may_use_selected_rate": report[
                    "future_training_may_use_selected_rate"
                ],
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
