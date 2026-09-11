#!/usr/bin/env python3
"""Test one source-restored damping step at the native throttle-motor readout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_continuous_cns_solver as solver  # noqa: E402
import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import ConnectomeController  # noqa: E402

EXPERIMENT = "variable-height-rk4-throttle-readout-step-v1"
PROTOCOL_COMMIT = "757048e"
EXPECTED_SOLVER_REPORT_SHA256 = "daeb4c000b3421a2d1d4d22890ccb900f654f1587748d035d88c4f884cff85a7"
EXPECTED_EDGE_INDICES_SHA256 = "4978758242ecfc9e53e859e077c098821e5747027a96f2945d3020e65ef81f3d"
EXPECTED_NODE_INDICES_SHA256 = "69c46d204b312950eb04e4538d5802031803e301d5b28dc45e8adb2d35487831"
EXPECTED_NODE_BODY_IDS_SHA256 = "bcdbf3f61b91b16dd7b6792b5622bb990d234fa97d326ba1180df0dd39a24752"
MASKED_EDGE_COUNT = 493
MASKED_NODE_COUNT = 7
EDGE_BIAS_LEARNING_RATE = 1.0e-4
TIME_CONSTANT_LEARNING_RATE = 1.0e-6
OPTIMIZER_BETAS = (0.0, 0.999)
OPTIMIZER_EPSILON = 1.0e-8
GRADIENT_NORM_CAP = 1.0
PROBE_SCALE = 1.0 / 16.0
ORDINARY_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
MINIMUM_NRMSE_IMPROVEMENT = 0.001
PAIR_COMMON_RMS_LIMIT = 0.005
PAIR_COMMON_MAX_LIMIT = 0.01
RPY_RMS_LIMIT = 0.005
RPY_MAX_LIMIT = 0.01
DERIVATIVE_RELATIVE_ERROR_LIMIT = 0.20
REPLAY_NOISE_MULTIPLIER = 10.0
MINIMUM_OBJECTIVE_CHANGE = 1.0e-8
PARAMETER_FAMILIES = ("edge_magnitude", "bias", "raw_time_constant")


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
        "--solver-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-continuous-cns-solver-001/report.json",
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
        default=REPO_ROOT / "runs/variable-height-hover/native-rk4-throttle-readout-step-001",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "solver": "classical RK4, one 20 ms step, four recurrent graph evaluations",
        "actor_inputs": ["cached 320x200 linear RGB", "cached roll", "cached pitch"],
        "external_or_engineered_state": False,
        "parameter_mask": {
            "incoming_throttle_motor_edge_magnitudes": MASKED_EDGE_COUNT,
            "throttle_motor_biases_and_time_constants": MASKED_NODE_COUNT,
            "edge_indices_sha256": EXPECTED_EDGE_INDICES_SHA256,
            "node_indices_sha256": EXPECTED_NODE_INDICES_SHA256,
            "node_body_ids_sha256": EXPECTED_NODE_BODY_IDS_SHA256,
        },
        "objective": "frozen-scale paired endpoint throttle-contrast MSE only",
        "prefix": "source-computed detached 25-frame neutral recurrence",
        "optimizer": {
            "name": "Adam",
            "betas": list(OPTIMIZER_BETAS),
            "epsilon": OPTIMIZER_EPSILON,
            "edge_and_bias_learning_rate": EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": TIME_CONSTANT_LEARNING_RATE,
            "gradient_norm_cap": GRADIENT_NORM_CAP,
            "one_transaction_only": True,
        },
        "probe_scale": PROBE_SCALE,
        "ordinary_scales": list(ORDINARY_SCALES),
        "minimum_fixed_and_full_nrmse_improvement": MINIMUM_NRMSE_IMPROVEMENT,
        "preservation": {
            "pair_common_throttle_rms": PAIR_COMMON_RMS_LIMIT,
            "pair_common_throttle_maximum": PAIR_COMMON_MAX_LIMIT,
            "rpy_rms_per_axis": RPY_RMS_LIMIT,
            "rpy_maximum_per_axis": RPY_MAX_LIMIT,
        },
        "family_only_replays_are_diagnostic": True,
        "candidate_retained": False,
        "training_hover_gate_or_promotion_authorized": False,
    }


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def build_readout_mask(graph_path: Path) -> dict[str, Any]:
    graph = np.load(graph_path)
    begin = int(graph["output_pool_offsets"][6])
    end = int(graph["output_pool_offsets"][8])
    nodes = np.asarray(graph["output_pool_indices"][begin:end], dtype=np.int64)
    edges = np.flatnonzero(np.isin(graph["edge_post"], nodes)).astype(np.int64)
    body_ids = np.asarray(graph["node_ids"][nodes], dtype=np.int64)
    manifest = {
        "edge_count": len(edges),
        "node_count": len(nodes),
        "edge_indices_sha256": array_sha256(edges),
        "node_indices_sha256": array_sha256(nodes),
        "node_body_ids_sha256": array_sha256(body_ids),
        "node_body_ids": body_ids.tolist(),
        "positive_edges": int((graph["edge_sign"][edges] > 0).sum()),
        "negative_edges": int((graph["edge_sign"][edges] < 0).sum()),
    }
    expected = {
        "edge_count": MASKED_EDGE_COUNT,
        "node_count": MASKED_NODE_COUNT,
        "edge_indices_sha256": EXPECTED_EDGE_INDICES_SHA256,
        "node_indices_sha256": EXPECTED_NODE_INDICES_SHA256,
        "node_body_ids_sha256": EXPECTED_NODE_BODY_IDS_SHA256,
    }
    for name, value in expected.items():
        if manifest[name] != value:
            raise SystemExit(f"readout-mask {name} mismatch")
    return {"edge_indices": edges, "node_indices": nodes, "manifest": manifest}


def validate_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    base_hashes = rate.validate_inputs(args)
    expected = {
        args.solver_report: EXPECTED_SOLVER_REPORT_SHA256,
        args.input_cache: solver.EXPECTED_CACHE_FILE_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or assisted.responsibility.file_sha256(path) != digest:
            raise SystemExit(f"locked input is missing or changed: {path}")
    with args.solver_report.open() as stream:
        solver_report = json.load(stream)
    selected = solver_report.get("selected_solver") or {}
    if (
        solver_report.get("classification") != "numerically_adequate_solver_selected"
        or selected.get("name") != "rk4_m1"
        or selected.get("graph_evaluations_per_camera_frame") != 4
    ):
        raise SystemExit("solver report did not select the registered RK4-M=1 method")
    cache = torch.load(args.input_cache, map_location="cpu", weights_only=True)
    rate.validate_cache(cache)
    if cache.get("semantic_sha256") != solver.EXPECTED_CACHE_SEMANTIC_SHA256:
        raise SystemExit("locked input cache semantic hash mismatch")
    return {
        **base_hashes,
        assisted.responsibility.stable_path(args.solver_report): EXPECTED_SOLVER_REPORT_SHA256,
        assisted.responsibility.stable_path(args.input_cache): solver.EXPECTED_CACHE_FILE_SHA256,
    }, {"solver_report": solver_report, "cache": cache}


def _clone_state_dict(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


@torch.no_grad()
def source_prefix_states(
    controller: ConnectomeController, cache: dict[str, Any], *, device: torch.device
) -> Tensor:
    images = cache["prefix_images"].to(device)
    attitude = cache["prefix_roll_pitch"].to(device)
    recurrent = controller.initial_state(int(cache["cases"]), device=device, dtype=images.dtype)
    for _ in range(int(cache["prefix_repetitions"])):
        _, recurrent, finite = solver.advance_rk4_frame(
            controller, images, attitude, recurrent, solver_steps=1
        )
        if not bool(finite):
            raise RuntimeError("source prefix became nonfinite")
    return recurrent.detach()


def advance_rk4_train(
    controller: ConnectomeController,
    image: Tensor,
    roll_pitch: Tensor,
    recurrent: Tensor,
) -> tuple[Tensor, Tensor]:
    sensory = controller.sensory_drive(image, roll_pitch)

    def derivative(value: Tensor) -> Tensor:
        return solver.continuous_state_derivative(controller, value, sensory)

    recurrent = solver.classical_rk4_step(derivative, recurrent, 1.0 / rate.POLICY_HZ)
    return controller.motor_drive(recurrent), recurrent


def fixed_prefix_objective(
    controller: ConnectomeController,
    cache: dict[str, Any],
    prefix_states: Tensor,
    *,
    device: torch.device,
    backward: bool,
) -> tuple[Tensor, dict[str, Any]]:
    cases = int(cache["cases"])
    outputs = torch.empty(cases, 2, 4, device=device)
    finite = True
    total = torch.zeros((), device=device)
    for horizon in motion.HISTORY_LENGTHS:
        group = cache["groups"][str(horizon)]
        for local_index, case_index_tensor in enumerate(group["case_indices"]):
            case_index = int(case_index_tensor)
            recurrent = prefix_states[case_index].expand(2, -1).clone()
            attitude = group["roll_pitch"][local_index].to(device)
            output = torch.zeros(2, 4, device=device)
            for step in range(horizon):
                image = group["response_images"][step, local_index].to(device)
                output, recurrent = advance_rk4_train(controller, image, attitude, recurrent)
            prediction = output[1, 3] - output[0, 3]
            target = cache["teacher_targets"][case_index].to(device)
            loss = ((prediction - target) / float(cache["teacher_scale"])).square()
            if backward:
                (loss / cases).backward()
            total = total + loss.detach()
            outputs[case_index] = output.detach()
            finite = (
                finite
                and bool(torch.isfinite(recurrent).all())
                and bool(torch.isfinite(output).all())
            )
    summary = rate.summarize_outputs(
        outputs.detach().cpu(),
        cache["teacher_targets"],
        cache["horizon"],
        scale=float(cache["teacher_scale"]),
        recurrence_finite=finite,
        endpoint_difference_max=float(cache["endpoint_image_difference"].max()),
        wall_time_seconds=0.0,
        recurrent_calls=4 * sum(motion.HISTORY_LENGTHS) * 8,
        branch_state_updates=4 * sum(motion.HISTORY_LENGTHS) * 8 * 2,
    )
    return total / cases, summary


def _masked_gradients(
    controller: ConnectomeController, mask: dict[str, Any]
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    edge_selected = torch.zeros_like(controller.edge_magnitude, dtype=torch.bool)
    node_selected = torch.zeros_like(controller.bias, dtype=torch.bool)
    edge_selected[torch.from_numpy(mask["edge_indices"]).to(edge_selected.device)] = True
    node_selected[torch.from_numpy(mask["node_indices"]).to(node_selected.device)] = True
    selections = {
        "edge_magnitude": edge_selected,
        "bias": node_selected,
        "raw_time_constant": node_selected,
    }
    raw = {}
    for name, selected in selections.items():
        parameter = getattr(controller, name)
        if parameter.grad is None:
            raise RuntimeError(f"missing gradient for {name}")
        if not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError(f"nonfinite gradient for {name}")
        parameter.grad[~selected] = 0.0
        raw[name] = parameter.grad.detach().clone()
    norm_before = float(torch.sqrt(sum(value.double().square().sum() for value in raw.values())))
    torch.nn.utils.clip_grad_norm_(
        [getattr(controller, name) for name in PARAMETER_FAMILIES],
        GRADIENT_NORM_CAP,
    )
    clipped = {name: getattr(controller, name).grad.detach().clone() for name in PARAMETER_FAMILIES}
    return clipped, {
        "gradient_norm_before_clipping": norm_before,
        "family_gradient_rms_before_clipping": {
            name: float(raw[name][selections[name]].square().mean().sqrt())
            for name in PARAMETER_FAMILIES
        },
        "all_masked_gradients_nonzero": all(
            bool((raw[name][selections[name]] != 0).any()) for name in PARAMETER_FAMILIES
        ),
        "outside_mask_gradients_exactly_zero": all(
            bool((raw[name][~selections[name]] == 0).all()) for name in PARAMETER_FAMILIES
        ),
    }


def make_optimizer(controller: ConnectomeController) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {
                "params": [controller.edge_magnitude, controller.bias],
                "lr": EDGE_BIAS_LEARNING_RATE,
            },
            {
                "params": [controller.raw_time_constant],
                "lr": TIME_CONSTANT_LEARNING_RATE,
            },
        ],
        betas=OPTIMIZER_BETAS,
        eps=OPTIMIZER_EPSILON,
    )


def parameter_displacement(
    source: dict[str, Tensor], candidate: dict[str, Tensor]
) -> dict[str, Tensor]:
    return {name: candidate[name] - source[name] for name in PARAMETER_FAMILIES}


def materialize_candidate(
    source: dict[str, Tensor],
    pending: dict[str, Tensor],
    *,
    scale: float,
    families: tuple[str, ...] = PARAMETER_FAMILIES,
) -> dict[str, Tensor]:
    result = _clone_state_dict(source)
    for name in families:
        result[name] = source[name] + scale * (pending[name] - source[name])
    result["edge_magnitude"].clamp_(0.0, 8.0)
    return result


def outside_mask_equal(
    source: dict[str, Tensor], candidate: dict[str, Tensor], mask: dict[str, Any]
) -> bool:
    edge_selected = torch.zeros(len(source["edge_magnitude"]), dtype=torch.bool)
    node_selected = torch.zeros(len(source["bias"]), dtype=torch.bool)
    edge_selected[torch.from_numpy(mask["edge_indices"])] = True
    node_selected[torch.from_numpy(mask["node_indices"])] = True
    return bool(
        torch.equal(
            source["edge_magnitude"][~edge_selected],
            candidate["edge_magnitude"][~edge_selected],
        )
        and torch.equal(source["bias"][~node_selected], candidate["bias"][~node_selected])
        and torch.equal(
            source["raw_time_constant"][~node_selected],
            candidate["raw_time_constant"][~node_selected],
        )
    )


def output_preservation(source_outputs: Tensor, candidate_outputs: Tensor) -> dict[str, Any]:
    source_common = source_outputs[:, :, 3].mean(dim=1)
    candidate_common = candidate_outputs[:, :, 3].mean(dim=1)
    common_drift = candidate_common - source_common
    rpy = {}
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        drift = candidate_outputs[:, :, axis] - source_outputs[:, :, axis]
        rpy[name] = {
            "rms": float(drift.square().mean().sqrt()),
            "maximum_absolute": float(drift.abs().max()),
        }
    return {
        "pair_common_throttle_drift_rms": float(common_drift.square().mean().sqrt()),
        "pair_common_throttle_drift_maximum_absolute": float(common_drift.abs().max()),
        "rpy_drift": rpy,
    }


def candidate_decision(record: dict[str, Any]) -> dict[str, Any]:
    preservation = record["preservation"]
    reasons = []
    if record["fixed_nrmse_improvement"] < MINIMUM_NRMSE_IMPROVEMENT:
        reasons.append("fixed-prefix NRMSE improvement was below 0.001")
    if record["full_nrmse_improvement"] < MINIMUM_NRMSE_IMPROVEMENT:
        reasons.append("zero-state full-prefix NRMSE improvement was below 0.001")
    if preservation["pair_common_throttle_drift_rms"] > PAIR_COMMON_RMS_LIMIT:
        reasons.append("pair-common throttle RMS drift exceeded 0.005")
    if preservation["pair_common_throttle_drift_maximum_absolute"] > PAIR_COMMON_MAX_LIMIT:
        reasons.append("pair-common throttle maximum drift exceeded 0.01")
    for name, values in preservation["rpy_drift"].items():
        if values["rms"] > RPY_RMS_LIMIT:
            reasons.append(f"{name} RMS drift exceeded 0.005")
        if values["maximum_absolute"] > RPY_MAX_LIMIT:
            reasons.append(f"{name} maximum drift exceeded 0.01")
    if not record["fixed"]["all_recurrent_states_and_outputs_finite"]:
        reasons.append("fixed-prefix recurrence or outputs were nonfinite")
    if not record["full"]["all_recurrent_states_and_outputs_finite"]:
        reasons.append("full-prefix recurrence or outputs were nonfinite")
    if not record["fixed"]["all_metrics_finite"] or not record["full"]["all_metrics_finite"]:
        reasons.append("candidate metrics were nonfinite")
    if record["full"]["endpoint_image_difference_max"] != 0.0:
        reasons.append("paired endpoints were not identical")
    if record["maximum_motor_absolute"] > 1.0:
        reasons.append("candidate motor output left [-1,1]")
    if not record["native_bounds_pass"]:
        reasons.append("candidate native edge bounds failed")
    if not record["outside_mask_parameters_exact"]:
        reasons.append("a parameter outside the readout mask changed")
    if not record["candidate_state_loaded_and_restored"]:
        reasons.append("candidate replay did not preserve its loaded parameter state")
    return {"pass": not reasons, "reasons": reasons}


@torch.no_grad()
def evaluate_fixed_candidate(
    graph: Path,
    state_dict: dict[str, Tensor],
    cache: dict[str, Any],
    prefix_states: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    controller = ConnectomeController(graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(state_dict, strict=True)
    before = rate._source_state_sha256(controller)
    _, summary = fixed_prefix_objective(
        controller, cache, prefix_states, device=device, backward=False
    )
    after = rate._source_state_sha256(controller)
    summary["candidate_state_loaded_and_restored"] = bool(
        before == assisted.audit.semantic_sha256(state_dict) and after == before
    )
    del controller
    return summary


def evaluate_candidate(
    *,
    graph: Path,
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    cache: dict[str, Any],
    prefix_states: Tensor,
    source_fixed: dict[str, Any],
    source_full: dict[str, Any],
    mask: dict[str, Any],
    device: torch.device,
    scale: float,
    families: tuple[str, ...],
) -> dict[str, Any]:
    fixed = evaluate_fixed_candidate(graph, candidate_state, cache, prefix_states, device=device)
    full = solver.evaluate_condition(
        graph=graph,
        checkpoint_state=candidate_state,
        cache=cache,
        method="rk4",
        solver_steps=1,
        device=device,
    )
    source_outputs = torch.tensor(source_full["terminal_motor_outputs"])
    candidate_outputs = torch.tensor(full["terminal_motor_outputs"])
    displacement = parameter_displacement(source_state, candidate_state)
    record = {
        "scale": scale,
        "families": list(families),
        "fixed": fixed,
        "full": full,
        "fixed_nrmse_improvement": source_fixed["nrmse"] - fixed["nrmse"],
        "full_nrmse_improvement": source_full["nrmse"] - full["nrmse"],
        "preservation": output_preservation(source_outputs, candidate_outputs),
        "maximum_motor_absolute": float(candidate_outputs.abs().max()),
        "native_bounds_pass": bool(
            (candidate_state["edge_magnitude"] >= 0).all()
            and (candidate_state["edge_magnitude"] <= 8).all()
        ),
        "outside_mask_parameters_exact": outside_mask_equal(source_state, candidate_state, mask),
        "candidate_state_loaded_and_restored": bool(
            fixed["candidate_state_loaded_and_restored"]
            and full["source_loaded_exactly"]
            and full["source_restored"]
        ),
        "family_displacement_rms": {
            name: float(displacement[name].square().mean().sqrt()) for name in PARAMETER_FAMILIES
        },
    }
    record["decision"] = candidate_decision(record)
    return record


def directional_agreement(
    *,
    baseline_values: list[float],
    probe_objective: float,
    gradient: dict[str, Tensor],
    source: dict[str, Tensor],
    probe: dict[str, Tensor],
) -> dict[str, Any]:
    replay_noise = max(
        abs(left - right)
        for index, left in enumerate(baseline_values)
        for right in baseline_values[index + 1 :]
    )
    measured = probe_objective - baseline_values[0]
    predicted = sum(
        float(
            (gradient[name].detach().cpu().double() * (probe[name] - source[name]).double()).sum()
        )
        for name in PARAMETER_FAMILIES
    )
    relative_error = 2.0 * abs(measured - predicted) / max(abs(measured) + abs(predicted), 1.0e-30)
    noise_threshold = max(MINIMUM_OBJECTIVE_CHANGE, REPLAY_NOISE_MULTIPLIER * replay_noise)
    passed = bool(
        math.isfinite(measured)
        and math.isfinite(predicted)
        and measured < -1.0e-8
        and predicted < -1.0e-8
        and relative_error <= DERIVATIVE_RELATIVE_ERROR_LIMIT
        and abs(measured) > noise_threshold
    )
    return {
        "pass": passed,
        "baseline_objectives": baseline_values,
        "replay_noise": replay_noise,
        "objective_change_noise_threshold": noise_threshold,
        "measured_objective_change": measured,
        "predicted_objective_change": predicted,
        "symmetric_relative_error": relative_error,
    }


def _write_start_marker(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise SystemExit("readout-step start marker already exists; replay is forbidden") from exc


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").exists():
        raise SystemExit("the RK4 throttle-readout step audit already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes, locked = validate_inputs(args)
    cache = locked["cache"]
    source_full = locked["solver_report"]["conditions"]["rk4_m1"]
    mask = build_readout_mask(args.graph)
    loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source_state = _clone_state_dict(loaded["controller"])
    source_sha256 = assisted.audit.semantic_sha256(source_state)
    _write_start_marker(
        args.output_dir / "start.json",
        {
            "experiment": EXPERIMENT,
            "protocol": protocol_manifest(),
            "input_file_sha256": input_hashes,
            "source_state_sha256": source_sha256,
            "readout_mask": mask["manifest"],
            "device": str(device),
        },
    )
    started = perf_counter()
    controller = ConnectomeController(args.graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(source_state, strict=True)
    prefix_states = source_prefix_states(controller, cache, device=device)
    baseline_values = []
    source_fixed = None
    print(json.dumps({"stage": "fixed_source_replays"}), flush=True)
    with torch.no_grad():
        for _ in range(3):
            objective, summary = fixed_prefix_objective(
                controller, cache, prefix_states, device=device, backward=False
            )
            baseline_values.append(float(objective))
            source_fixed = summary
    assert source_fixed is not None
    if abs(source_fixed["nrmse"] - source_full["nrmse"]) > 2.0e-5:
        raise SystemExit("fixed-prefix RK4 source did not reproduce full-prefix source")

    print(json.dumps({"stage": "masked_full_bank_gradient"}), flush=True)
    optimizer = make_optimizer(controller)
    for parameter in controller.parameters():
        parameter.grad = None
    gradient_objective, gradient_summary = fixed_prefix_objective(
        controller, cache, prefix_states, device=device, backward=True
    )
    clipped_gradient, gradient_controls = _masked_gradients(controller, mask)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    controller.project_parameters()
    pending_state = _clone_state_dict(controller.state_dict())
    optimizer_after = copy.deepcopy(optimizer.state_dict())
    full_displacement = parameter_displacement(source_state, pending_state)
    full_direction = sum(
        float(
            (
                clipped_gradient[name].detach().cpu().double() * full_displacement[name].double()
            ).sum()
        )
        for name in PARAMETER_FAMILIES
    )
    controller.load_state_dict(source_state, strict=True)
    source_restored_after_proposal = rate._source_state_sha256(controller) == source_sha256

    probe_state = materialize_candidate(source_state, pending_state, scale=PROBE_SCALE)
    probe_fixed = evaluate_fixed_candidate(
        args.graph, probe_state, cache, prefix_states, device=device
    )
    derivative = directional_agreement(
        baseline_values=baseline_values,
        probe_objective=probe_fixed["nrmse"] ** 2,
        gradient=clipped_gradient,
        source=source_state,
        probe=probe_state,
    )
    numerical_controls_pass = bool(
        gradient_controls["outside_mask_gradients_exactly_zero"]
        and gradient_controls["all_masked_gradients_nonzero"]
        and full_direction < -1.0e-8
        and outside_mask_equal(source_state, pending_state, mask)
        and source_restored_after_proposal
        and derivative["pass"]
    )

    ordinary_records = []
    selected = None
    if numerical_controls_pass:
        for scale in ORDINARY_SCALES:
            print(
                json.dumps({"stage": "ordinary_candidate", "scale": scale}),
                flush=True,
            )
            candidate_state = materialize_candidate(source_state, pending_state, scale=scale)
            record = evaluate_candidate(
                graph=args.graph,
                source_state=source_state,
                candidate_state=candidate_state,
                cache=cache,
                prefix_states=prefix_states,
                source_fixed=source_fixed,
                source_full=source_full,
                mask=mask,
                device=device,
                scale=scale,
                families=PARAMETER_FAMILIES,
            )
            ordinary_records.append(record)
            if record["decision"]["pass"]:
                selected = record
                break

    family_diagnostics = {}
    for family in PARAMETER_FAMILIES:
        print(json.dumps({"stage": "family_diagnostic", "family": family}), flush=True)
        family_state = materialize_candidate(
            source_state, pending_state, scale=1.0, families=(family,)
        )
        family_diagnostics[family] = evaluate_candidate(
            graph=args.graph,
            source_state=source_state,
            candidate_state=family_state,
            cache=cache,
            prefix_states=prefix_states,
            source_fixed=source_fixed,
            source_full=source_full,
            mask=mask,
            device=device,
            scale=1.0,
            families=(family,),
        )

    controller.load_state_dict(source_state, strict=True)
    terminal_source_restored = rate._source_state_sha256(controller) == source_sha256
    passed = bool(numerical_controls_pass and selected is not None and terminal_source_restored)
    classification = (
        "safe_local_readout_step_exists"
        if passed
        else (
            "readout_numerical_control_failed"
            if not numerical_controls_pass
            else "no_safe_local_readout_step"
        )
    )
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha256,
        "readout_mask": mask["manifest"],
        "source_fixed": source_fixed,
        "source_full": source_full,
        "baseline_fixed_objectives": baseline_values,
        "gradient_objective": float(gradient_objective),
        "gradient_summary": gradient_summary,
        "gradient_controls": gradient_controls,
        "optimizer": {
            "betas": list(OPTIMIZER_BETAS),
            "epsilon": OPTIMIZER_EPSILON,
            "state_before_sha256": assisted.audit.semantic_sha256(optimizer_before),
            "state_after_sha256": assisted.audit.semantic_sha256(optimizer_after),
            "step_counters_after": assisted.optimizer_step_counters(optimizer_after),
        },
        "pending_parameter_sha256": assisted.audit.semantic_sha256(pending_state),
        "full_proposal_directional_derivative": full_direction,
        "source_restored_after_proposal": source_restored_after_proposal,
        "derivative_probe": derivative,
        "numerical_controls_pass": numerical_controls_pass,
        "ordinary_candidates": ordinary_records,
        "selected_scale": None if selected is None else selected["scale"],
        "selected_candidate_retained": False,
        "family_only_diagnostics": family_diagnostics,
        "terminal_source_restored": terminal_source_restored,
        "bounded_last_hop_fitting_preregistration_authorized": passed,
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
                "pass": passed,
                "numerical_controls_pass": numerical_controls_pass,
                "selected_scale": report["selected_scale"],
                "bounded_last_hop_fitting_preregistration_authorized": passed,
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
