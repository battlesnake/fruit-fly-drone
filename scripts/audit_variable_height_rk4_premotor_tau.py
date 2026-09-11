#!/usr/bin/env python3
"""Audit upstream premotor time-constant leverage under the selected RK4 solver."""

from __future__ import annotations

import argparse
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
import audit_variable_height_rk4_throttle_readout_step as readout  # noqa: E402
import audit_variable_height_upstream_damping_route as upstream  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402
import train_variable_height_rk4_readout_capacity as capacity  # noqa: E402

from flydrone.connectome_data import (  # noqa: E402
    ANNOTATIONS_FILE,
    _read_annotations,
)
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-rk4-premotor-tau-preflight-v2"
PROTOCOL_COMMIT = "c318a1f"
EXPECTED_FAILED_REPORT_SHA256 = (
    "a2b29f4879313f1763f8f10dbbb524fad3c34d05ea5042debbc799b85c5a637f"
)
EXPECTED_DIRECTION_FILE_SHA256 = (
    "ee2fa3b6b6944addef112544a7ffa7453b4bb8e5ab7ff98b2f3ab21997b5f085"
)
EXPECTED_DIRECTION_SEMANTIC_SHA256 = (
    "ad34ae1a8e7bd1a295b23d3bab494b9c16714821e5558907492b3da1a19b063e"
)
EXPECTED_CAPACITY_REPORT_SHA256 = (
    "7f0dbd25aff387a67ec992e2ea7468c7c9c7b74e7d569914565f99665501b09c"
)
EXPECTED_UPSTREAM_PREFLIGHT_SHA256 = (
    "4a93a09acee7efc96e55043cddfd089f198ed253c44cab1e419972e8c5c8f6ea"
)
EXPECTED_UPSTREAM_TRAINING_SHA256 = (
    "137da6fee2f0fac564931f5db8cdaa7cb3c340747af9286dfa9ef0d3f96fbcec"
)
EXPECTED_CACHE_MANIFESTS_SHA256 = (
    "24ce0c614b0595fda0d5f478463d31e49b67d8ae271488eaf50fc2f95b8a94bb"
)
EXPECTED_SOURCE_REFERENCES_SHA256 = (
    "2d81cf161f652562b5b5849411ca5f3cfac1cb973336e199d224aede628b1690"
)
EXPECTED_NODE_INDICES_SHA256 = upstream.EXPECTED_BIAS_NODE_INDICES_SHA256
SELECTED_NODES = 883
NODE_SUPERCLASS_COUNTS = {
    "ascending_neuron": 18,
    "descending_neuron": 612,
    "vnc_intrinsic": 253,
}

TAU_UNIT_RMS_SECONDS = 0.001
TAU_UNIT_MAX_SECONDS = 0.002
TAU_SCALES = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
FINITE_DIFFERENCE_SCALE = 0.125
TAU_RATIO_RANGE = (0.8, 1.25)
TAU_NATIVE_RANGE_SECONDS = (0.01, 0.25)
TAU_NATIVE_EPSILON_SECONDS = 1.0e-7
MINIMUM_NRMSE_IMPROVEMENT = 1.0e-4
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = 0.20
REPLAY_NOISE_MULTIPLIER = 10.0
MINIMUM_OBJECTIVE_CHANGE = 1.0e-8
DEVELOPMENT_HORIZON_TOLERANCE = 1.0e-5


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
        "--base-cache",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-neural-integration-rate-audit-001/input-cache.pt",
    )
    parser.add_argument(
        "--capacity-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-readout-capacity-001/report.json",
    )
    parser.add_argument(
        "--failed-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-001/report.json",
    )
    parser.add_argument(
        "--direction-archive",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-001/direction.pt",
    )
    parser.add_argument(
        "--cache-manifests",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-readout-capacity-001"
        / "training-cache-manifests.json",
    )
    parser.add_argument(
        "--source-references",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-readout-capacity-001"
        / "training-source-references.json",
    )
    parser.add_argument(
        "--upstream-preflight-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/upstream-damping-route-preflight-001/report.json",
    )
    parser.add_argument(
        "--upstream-training-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/upstream-damping-route-train-001/report.json",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-premotor-tau-preflight-002",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "corrected_replay": {
            "failed_report_sha256": EXPECTED_FAILED_REPORT_SHA256,
            "direction_file_sha256": EXPECTED_DIRECTION_FILE_SHA256,
            "direction_semantic_sha256": EXPECTED_DIRECTION_SEMANTIC_SHA256,
            "only_scientific_path_fix": (
                "derive maximum motor magnitude from combined terminal outputs"
            ),
        },
        "locked_reports": {
            "readout_capacity": EXPECTED_CAPACITY_REPORT_SHA256,
            "older_upstream_preflight": EXPECTED_UPSTREAM_PREFLIGHT_SHA256,
            "older_upstream_training": EXPECTED_UPSTREAM_TRAINING_SHA256,
        },
        "solver": "RK4-M1, requalified against exponential-Euler K32 after tau change",
        "training_cache_seeds": list(capacity.TRAINING_SEEDS),
        "development_cache_seeds": list(capacity.DEVELOPMENT_SEEDS),
        "development_generated_only_after_training_and_solver_pass": True,
        "actor_inputs": ["cached 320x200 linear RGB", "roll", "pitch"],
        "external_or_engineered_state": False,
        "mask": {
            "raw_time_constants_only": True,
            "selected_nonmotor_nodes": SELECTED_NODES,
            "node_indices_sha256": EXPECTED_NODE_INDICES_SHA256,
            "superclass_counts": NODE_SUPERCLASS_COUNTS,
            "edges_biases_motor_taus_and_unselected_taus_frozen": True,
        },
        "physical_tau_direction": {
            "unit_rms_seconds": TAU_UNIT_RMS_SECONDS,
            "unit_maximum_per_cell_seconds": TAU_UNIT_MAX_SECONDS,
            "scales": list(TAU_SCALES),
            "source_ratio_range": list(TAU_RATIO_RANGE),
            "native_seconds_range": list(TAU_NATIVE_RANGE_SECONDS),
            "one_float32_inverse_logit_materialization": True,
        },
        "objective": "96-pair normalized opposite-motion throttle-contrast MSE",
        "source_prefix_replays": 3,
        "finite_difference": {
            "scale": FINITE_DIFFERENCE_SCALE,
            "symmetric_relative_error_limit": FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT,
            "noise_multiplier": REPLAY_NOISE_MULTIPLIER,
            "minimum_objective_change": MINIMUM_OBJECTIVE_CHANGE,
            "selects_candidate": False,
        },
        "minimum_fixed_and_full_nrmse_improvement": MINIMUM_NRMSE_IMPROVEMENT,
        "per_block_preservation": {
            "common_throttle_rms": readout.PAIR_COMMON_RMS_LIMIT,
            "common_throttle_maximum": readout.PAIR_COMMON_MAX_LIMIT,
            "rpy_rms_per_axis": readout.RPY_RMS_LIMIT,
            "rpy_maximum_per_axis": readout.RPY_MAX_LIMIT,
        },
        "solver_agreement": {
            "source_and_candidate_all_training_blocks": True,
            "normalized_contrast_rms_maximum": rate.CONTRAST_REFINEMENT_LIMIT,
            "terminal_motor_rms_maximum": rate.MOTOR_REFINEMENT_LIMIT,
        },
        "development": {
            "one_training_selected_candidate": True,
            "minimum_nrmse_improvement": MINIMUM_NRMSE_IMPROVEMENT,
            "per_horizon_regression_tolerance": DEVELOPMENT_HORIZON_TOLERANCE,
            "alternate_scale_after_development": False,
        },
        "candidate_retained": False,
        "training_hover_gate_or_promotion_authorized": False,
    }


def file_sha256(path: Path) -> str:
    return assisted.responsibility.file_sha256(path)


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    expected = {
        args.graph: assisted.EXPECTED_GRAPH_SHA256,
        args.checkpoint: assisted.EXPECTED_CHECKPOINT_SHA256,
        args.capacity_report: EXPECTED_CAPACITY_REPORT_SHA256,
        args.failed_report: EXPECTED_FAILED_REPORT_SHA256,
        args.direction_archive: EXPECTED_DIRECTION_FILE_SHA256,
        args.cache_manifests: EXPECTED_CACHE_MANIFESTS_SHA256,
        args.source_references: EXPECTED_SOURCE_REFERENCES_SHA256,
        args.upstream_preflight_report: EXPECTED_UPSTREAM_PREFLIGHT_SHA256,
        args.upstream_training_report: EXPECTED_UPSTREAM_TRAINING_SHA256,
    }
    hashes = {}
    for path, digest in expected.items():
        if not path.is_file() or file_sha256(path) != digest:
            raise SystemExit(f"locked premotor-tau input is missing or changed: {path}")
        hashes[assisted.responsibility.stable_path(path)] = digest
    with args.capacity_report.open() as stream:
        capacity_report = json.load(stream)
    if (
        capacity_report.get("classification") != "readout_capacity_constraint_boundary_stop"
        or capacity_report.get("passed")
        or capacity_report.get("accepted_updates") != 2
        or capacity_report.get("development") is not None
    ):
        raise SystemExit("capacity report is not the registered last-hop boundary stop")
    with args.failed_report.open() as stream:
        failed_report = json.load(stream)
    if (
        failed_report.get("classification") != "premotor_tau_exception_failed_closed"
        or failed_report.get("passed")
        or failed_report.get("exception", {}).get("type") != "KeyError"
        or failed_report.get("exception", {}).get("message")
        != "'maximum_motor_absolute'"
        or failed_report.get("selected_scale") is not None
        or failed_report.get("development") is not None
        or not failed_report.get("source_state_restored")
    ):
        raise SystemExit("the locked premotor-tau failure is not the registered harness stop")
    if (args.failed_report.parent / "development-started.json").exists():
        raise SystemExit("the failed premotor-tau run unexpectedly opened development")
    with args.cache_manifests.open() as stream:
        manifests = json.load(stream)
    if [item["seed"] for item in manifests] != list(capacity.TRAINING_SEEDS):
        raise SystemExit("training cache seed set mismatch")
    for manifest in manifests:
        path = REPO_ROOT / manifest["path"]
        if not path.is_file() or file_sha256(path) != manifest["file_sha256"]:
            raise SystemExit(f"training cache changed: {path}")
        hashes[manifest["path"]] = manifest["file_sha256"]
    with args.source_references.open() as stream:
        references = json.load(stream)
    if references.get("semantic_sha256") != capacity._payload_semantic_hash(references):
        raise SystemExit("training source-reference semantic hash mismatch")
    if references.get("cache_file_sha256") != [
        item["file_sha256"] for item in manifests
    ]:
        raise SystemExit("training source references use different caches")
    if not source_references_valid(references):
        raise SystemExit("training source references failed identity or finiteness checks")
    return hashes, manifests, references


def build_tau_mask(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    _, nodes, manifest = upstream.build_expanded_route_mask(args.graph, args.raw_dir)
    if len(nodes) != SELECTED_NODES or manifest["bias_node_indices_sha256"] != (
        EXPECTED_NODE_INDICES_SHA256
    ):
        raise SystemExit("premotor tau node mask changed")
    observed_counts = {
        name: int(count)
        for name, count in manifest["counts"].items()
        if name in NODE_SUPERCLASS_COUNTS
    }
    # The older mask manifest gives the aggregate counts separately; freeze the explicit
    # superclass composition checked against the raw annotations below.
    graph = np.load(args.graph)
    annotations = _read_annotations(args.raw_dir / ANNOTATIONS_FILE)
    rows = {int(body): index for index, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray(
        [str(annotations["superclass"][rows[int(body)]]) for body in graph["node_ids"]]
    )
    names, counts = np.unique(superclass[nodes], return_counts=True)
    composition = {
        str(name): int(count) for name, count in zip(names, counts, strict=True)
    }
    if composition != NODE_SUPERCLASS_COUNTS:
        raise SystemExit("premotor tau superclass composition changed")
    offsets, pools = graph["output_pool_offsets"], graph["output_pool_indices"]
    motor_nodes = pools[offsets[6] : offsets[8]]
    if np.intersect1d(nodes, motor_nodes).size:
        raise SystemExit("premotor tau mask unexpectedly contains throttle motor neurons")
    return nodes, {
        "selected_nodes": len(nodes),
        "node_indices_sha256": manifest["bias_node_indices_sha256"],
        "node_body_ids_sha256": manifest["bias_body_ids_sha256"],
        "superclass_counts": composition,
        "throttle_motor_intersection": 0,
        "legacy_manifest_counts": observed_counts,
    }


def _write_or_validate_start(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as stream:
            if json.load(stream) != payload:
                raise SystemExit("premotor-tau start marker mismatch")
        return
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def source_tau(raw: Tensor) -> Tensor:
    return 0.01 + 0.24 * torch.sigmoid(raw)


def raw_from_tau(tau: Tensor) -> Tensor:
    probability = ((tau - 0.01) / 0.24).clamp(1.0e-12, 1.0 - 1.0e-12)
    return torch.logit(probability)


def all_numeric_values_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_numeric_values_finite(item) for item in value)
    return True


def replay_summary_valid(summary: dict[str, Any]) -> bool:
    return bool(
        summary.get("all_recurrent_states_and_outputs_finite")
        and summary.get("all_metrics_finite")
        and summary.get("endpoint_image_difference_max") == 0.0
        and all_numeric_values_finite(summary)
    )


def source_references_valid(references: dict[str, Any]) -> bool:
    return bool(
        replay_summary_valid(references.get("combined", {}))
        and references.get("blocks")
        and all(
            item.get("full", {}).get("source_loaded_exactly")
            and item.get("full", {}).get("source_restored")
            and replay_summary_valid(item.get("full", {}))
            for item in references["blocks"]
        )
    )


def capped_unit_direction(negative_tau_gradient: Tensor) -> tuple[Tensor, dict[str, Any]]:
    values = negative_tau_gradient.detach().cpu().double()
    if not bool(torch.isfinite(values).all()) or float(values.square().sum()) == 0.0:
        raise ValueError("tau direction requires a finite nonzero gradient")
    low, high = 0.0, 1.0

    def candidate(multiplier: float) -> Tensor:
        return (multiplier * values).clamp(-TAU_UNIT_MAX_SECONDS, TAU_UNIT_MAX_SECONDS)

    while float(candidate(high).square().mean().sqrt()) < TAU_UNIT_RMS_SECONDS:
        high *= 2.0
        if not math.isfinite(high):
            raise ValueError("could not bracket the capped tau direction")
    for _ in range(100):
        middle = (low + high) / 2.0
        if float(candidate(middle).square().mean().sqrt()) < TAU_UNIT_RMS_SECONDS:
            low = middle
        else:
            high = middle
    direction = candidate(high)
    rms = float(direction.square().mean().sqrt())
    return direction, {
        "bisection_iterations": 100,
        "multiplier": high,
        "actual_rms_seconds": rms,
        "maximum_absolute_seconds": float(direction.abs().max()),
        "components_at_cap": int((direction.abs() == TAU_UNIT_MAX_SECONDS).sum()),
    }


def clone_state(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def source_checkpoint(path: Path) -> dict[str, Tensor]:
    return capacity._load_source_checkpoint(path)


def state_sha256(state: dict[str, Tensor]) -> str:
    return assisted.audit.semantic_sha256(state)


def direction_path(args: argparse.Namespace) -> Path:
    return args.direction_archive


def _direction_payload_hash(payload: dict[str, Any]) -> str:
    return capacity._payload_semantic_hash(payload)


def _fixed_replays(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        capacity.fixed_cohort_report(
            controller, args, manifests, prefixes, device=device
        )[0]
        for _ in range(3)
    ]


def build_direction_payload(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    node_indices: np.ndarray,
    *,
    source_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    controller.eval()
    controller.edge_magnitude.requires_grad_(False)
    controller.bias.requires_grad_(False)
    controller.raw_time_constant.requires_grad_(True)
    prefixes = capacity.current_prefixes(controller, args, manifests, device=device)
    replays = _fixed_replays(controller, args, manifests, prefixes, device=device)

    for parameter in controller.parameters():
        parameter.grad = None
    block_objectives = []
    block_summaries = []
    caches = []
    for manifest in manifests:
        cache = capacity._block_cache(args, manifest)
        caches.append(
            {"teacher_targets": cache["teacher_targets"], "horizon": cache["horizon"]}
        )
        objective, summary = readout.fixed_prefix_objective(
            controller,
            cache,
            prefixes[manifest["seed"]].to(device),
            device=device,
            backward=True,
        )
        block_objectives.append(float(objective.detach()))
        block_summaries.append(summary)
    gradient = controller.raw_time_constant.grad
    if gradient is None:
        raise RuntimeError("the accumulated raw-time-constant gradient is missing")
    gradient.div_(len(manifests))
    selected = torch.from_numpy(node_indices).to(gradient.device)
    outside = torch.ones_like(gradient, dtype=torch.bool)
    outside[selected] = False
    full_gradient_finite = bool(torch.isfinite(gradient).all())
    if not full_gradient_finite:
        raise RuntimeError("the full raw-time-constant gradient is nonfinite")
    gradient[outside] = 0.0
    raw_selected = gradient[selected].detach().cpu().double()
    if not bool(torch.isfinite(raw_selected).all()) or not bool((raw_selected != 0).any()):
        raise RuntimeError("the selected raw-time-constant gradient is invalid")
    raw_source = controller.raw_time_constant.detach()[selected].cpu().float()
    sigmoid = torch.sigmoid(raw_source)
    derivative = 0.24 * sigmoid * (1.0 - sigmoid)
    physical_gradient = raw_selected / derivative.double()
    unit_direction, direction_controls = capped_unit_direction(-physical_gradient)
    combined = capacity.combine_summaries(
        block_summaries, caches, scale=rate.EXPECTED_MOTION_SCALE
    )
    baseline_replays_valid = all(replay_summary_valid(item) for item in replays)
    if not baseline_replays_valid:
        raise RuntimeError("a source fixed-prefix replay was invalid")
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "source_state_sha256": source_sha256,
        "node_indices_sha256": EXPECTED_NODE_INDICES_SHA256,
        "node_indices": torch.from_numpy(node_indices.copy()),
        "prefix_states": {str(seed): value for seed, value in prefixes.items()},
        "source_fixed_replays": [capacity.compact_summary(item) for item in replays],
        "block_objectives": block_objectives,
        "combined_gradient_objective": combined["objective"],
        "combined_gradient_summary": capacity.compact_summary(combined),
        "raw_gradient_selected": raw_selected,
        "physical_tau_gradient_selected": physical_gradient,
        "unit_tau_direction_seconds": unit_direction,
        "direction_controls": direction_controls,
        "gradient_controls": {
            "selected_raw_gradient_rms": float(raw_selected.square().mean().sqrt()),
            "selected_physical_gradient_rms_per_second": float(
                physical_gradient.square().mean().sqrt()
            ),
            "selected_physical_gradient_maximum_absolute_per_second": float(
                physical_gradient.abs().max()
            ),
            "raw_to_physical_derivative_minimum_seconds": float(derivative.min()),
            "raw_to_physical_derivative_maximum_seconds": float(derivative.max()),
            "edge_and_bias_gradients_absent": bool(
                controller.edge_magnitude.grad is None and controller.bias.grad is None
            ),
            "full_raw_gradient_finite_before_masking": full_gradient_finite,
            "unselected_raw_gradients_exactly_zero": bool(
                (gradient[outside] == 0).all()
            ),
            "source_fixed_replays_valid": baseline_replays_valid,
        },
    }
    payload["semantic_sha256"] = _direction_payload_hash(payload)
    return payload


def load_or_create_direction(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    node_indices: np.ndarray,
    *,
    source_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    path = direction_path(args)
    if not path.is_file() or file_sha256(path) != EXPECTED_DIRECTION_FILE_SHA256:
        raise SystemExit("the locked premotor-tau direction archive is missing or changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("semantic_sha256") != _direction_payload_hash(payload):
        raise SystemExit("premotor-tau direction archive semantic hash mismatch")
    if (
        payload.get("experiment") != "variable-height-rk4-premotor-tau-preflight-v1"
        or payload.get("protocol_commit") != "393bbfb"
        or payload.get("source_state_sha256") != source_sha256
        or payload.get("node_indices_sha256") != EXPECTED_NODE_INDICES_SHA256
        or not torch.equal(
            payload["node_indices"], torch.from_numpy(node_indices.copy())
        )
    ):
        raise SystemExit("premotor-tau direction archive protocol mismatch")
    if payload.get("semantic_sha256") != EXPECTED_DIRECTION_SEMANTIC_SHA256:
        raise SystemExit("premotor-tau direction archive differs from corrected preregistration")
    controls = payload.get("gradient_controls", {})
    required_controls = (
        controls.get("edge_and_bias_gradients_absent"),
        controls.get("full_raw_gradient_finite_before_masking"),
        controls.get("unselected_raw_gradients_exactly_zero"),
        controls.get("source_fixed_replays_valid"),
        len(payload.get("source_fixed_replays", [])) == 3,
        all(
            replay_summary_valid(item)
            for item in payload.get("source_fixed_replays", [])
        ),
        all_numeric_values_finite(payload.get("raw_gradient_selected")),
        all_numeric_values_finite(payload.get("physical_tau_gradient_selected")),
        all_numeric_values_finite(payload.get("unit_tau_direction_seconds")),
    )
    if not all(required_controls):
        raise SystemExit("premotor-tau direction archive failed numerical controls")
    return payload


def tau_candidate(
    source_state: dict[str, Tensor],
    node_indices: np.ndarray,
    unit_direction: Tensor,
    *,
    scale: float,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    candidate = clone_state(source_state)
    selected = torch.from_numpy(node_indices)
    raw_source = source_state["raw_time_constant"].float()
    tau_source = source_tau(raw_source)
    source_selected = tau_source[selected]
    requested_delta = scale * unit_direction.float()
    requested = source_selected + requested_delta
    lower = torch.maximum(
        TAU_RATIO_RANGE[0] * source_selected + 1.0e-8,
        torch.full_like(source_selected, TAU_NATIVE_RANGE_SECONDS[0] + TAU_NATIVE_EPSILON_SECONDS),
    )
    upper = torch.minimum(
        TAU_RATIO_RANGE[1] * source_selected - 1.0e-8,
        torch.full_like(source_selected, TAU_NATIVE_RANGE_SECONDS[1] - TAU_NATIVE_EPSILON_SECONDS),
    )
    if not bool((lower < upper).all()):
        raise RuntimeError("source tau leaves no valid per-cell trust interval")
    target = requested.clamp(min=lower, max=upper)
    candidate["raw_time_constant"][selected] = raw_from_tau(target)
    actual_tau = source_tau(candidate["raw_time_constant"].float())[selected]
    delta = actual_tau - source_selected
    raw_delta = (
        candidate["raw_time_constant"][selected]
        - source_state["raw_time_constant"][selected]
    )
    ratio = actual_tau / source_selected
    controls = {
        "scale": scale,
        "materialization_dtype": "torch.float32",
        "requested_delta_rms_seconds": float(
            requested_delta.square().mean().sqrt()
        ),
        "requested_delta_maximum_absolute_seconds": float(
            requested_delta.abs().max()
        ),
        "actual_delta_rms_seconds": float(delta.square().mean().sqrt()),
        "actual_delta_maximum_absolute_seconds": float(delta.abs().max()),
        "raw_parameter_displacement_rms": float(raw_delta.square().mean().sqrt()),
        "raw_parameter_displacement_maximum_absolute": float(raw_delta.abs().max()),
        "source_tau_minimum_seconds": float(source_selected.min()),
        "source_tau_maximum_seconds": float(source_selected.max()),
        "candidate_tau_minimum_seconds": float(actual_tau.min()),
        "candidate_tau_maximum_seconds": float(actual_tau.max()),
        "candidate_to_source_ratio_minimum": float(ratio.min()),
        "candidate_to_source_ratio_maximum": float(ratio.max()),
        "components_clamped_lower": int((requested < lower).sum()),
        "components_clamped_upper": int((requested > upper).sum()),
        "ratio_bounds_pass": bool(
            (ratio >= TAU_RATIO_RANGE[0]).all()
            and (ratio <= TAU_RATIO_RANGE[1]).all()
        ),
        "native_bounds_pass": bool(
            (actual_tau > TAU_NATIVE_RANGE_SECONDS[0]).all()
            and (actual_tau < TAU_NATIVE_RANGE_SECONDS[1]).all()
        ),
        "all_tau_finite": bool(torch.isfinite(actual_tau).all()),
    }
    return candidate, controls


def frozen_parameter_controls(
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    node_indices: np.ndarray,
) -> dict[str, Any]:
    selected = torch.zeros_like(source_state["raw_time_constant"], dtype=torch.bool)
    selected[torch.from_numpy(node_indices)] = True
    result = {
        "edge_magnitudes_exact": torch.equal(
            source_state["edge_magnitude"], candidate_state["edge_magnitude"]
        ),
        "biases_exact": torch.equal(source_state["bias"], candidate_state["bias"]),
        "unselected_time_constants_exact": torch.equal(
            source_state["raw_time_constant"][~selected],
            candidate_state["raw_time_constant"][~selected],
        ),
    }
    result["pass"] = all(result.values())
    return result


def prefixes_from_direction(payload: dict[str, Any]) -> dict[int, Tensor]:
    return {int(seed): value for seed, value in payload["prefix_states"].items()}


def maximum_motor_absolute(summary: dict[str, Any]) -> float:
    outputs = torch.as_tensor(summary["terminal_motor_outputs"])
    if outputs.numel() == 0 or not bool(torch.isfinite(outputs).all()):
        return math.inf
    return float(outputs.abs().max())


def candidate_decision(
    *,
    fixed: dict[str, Any],
    full: dict[str, Any],
    source_fixed: dict[str, Any],
    source_full: dict[str, Any],
    preservation: list[dict[str, Any]],
    tau_controls: dict[str, Any],
    frozen_controls: dict[str, Any],
) -> dict[str, Any]:
    fixed_improvement = source_fixed["nrmse"] - fixed["nrmse"]
    full_improvement = source_full["nrmse"] - full["nrmse"]
    reasons = []
    if not replay_summary_valid(source_fixed):
        reasons.append("source fixed-prefix replay was invalid")
    if not replay_summary_valid(source_full):
        reasons.append("source zero-state replay was invalid")
    if not all_numeric_values_finite(
        {
            "fixed_improvement": fixed_improvement,
            "full_improvement": full_improvement,
            "preservation": preservation,
        }
    ):
        reasons.append("a calculated training comparison was nonfinite")
    if fixed_improvement < MINIMUM_NRMSE_IMPROVEMENT:
        reasons.append("fixed-prefix NRMSE improvement was below 0.0001")
    if full_improvement < MINIMUM_NRMSE_IMPROVEMENT:
        reasons.append("zero-state NRMSE improvement was below 0.0001")
    if not all(item["pass"] for item in preservation):
        reasons.append("at least one training block failed source-relative preservation")
    if not tau_controls["ratio_bounds_pass"]:
        reasons.append("a selected tau left its source-relative trust interval")
    if not tau_controls["native_bounds_pass"]:
        reasons.append("a selected tau left its native interval")
    if not tau_controls["all_tau_finite"]:
        reasons.append("a selected tau was nonfinite")
    if not frozen_controls["pass"]:
        reasons.append("a frozen parameter changed")
    for label, summary in (("fixed", fixed), ("full", full)):
        if not summary["all_recurrent_states_and_outputs_finite"]:
            reasons.append(f"{label} recurrence or output was nonfinite")
        if not summary["all_metrics_finite"]:
            reasons.append(f"{label} metrics were nonfinite")
        if summary["endpoint_image_difference_max"] != 0.0:
            reasons.append(f"{label} paired endpoint images differed")
        if maximum_motor_absolute(summary) > 1.0:
            reasons.append(f"{label} motor output left [-1,1]")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "fixed_nrmse_improvement": fixed_improvement,
        "full_nrmse_improvement": full_improvement,
        "all_block_preservation_pass": all(item["pass"] for item in preservation),
        "fixed_maximum_motor_absolute": maximum_motor_absolute(fixed),
        "full_maximum_motor_absolute": maximum_motor_absolute(full),
    }


def evaluate_training_candidate(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    references: dict[str, Any],
    prefixes: dict[int, Tensor],
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    node_indices: np.ndarray,
    tau_controls: dict[str, Any],
    source_fixed: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    controller.load_state_dict(candidate_state, strict=True)
    before = rate._source_state_sha256(controller)
    fixed, fixed_blocks = capacity.fixed_cohort_report(
        controller, args, manifests, prefixes, device=device
    )
    after = rate._source_state_sha256(controller)
    full, full_blocks = capacity.full_cohort_report(
        args, manifests, candidate_state, device=device
    )
    preservation = capacity.cohort_preservation(references, full_blocks, manifests)
    frozen = frozen_parameter_controls(source_state, candidate_state, node_indices)
    loaded_and_restored = bool(
        before == state_sha256(candidate_state)
        and after == before
        and all(
            item["source_loaded_exactly"] and item["source_restored"]
            for item in full_blocks
        )
    )
    decision = candidate_decision(
        fixed=fixed,
        full=full,
        source_fixed=source_fixed,
        source_full=references["combined"],
        preservation=preservation,
        tau_controls=tau_controls,
        frozen_controls=frozen,
    )
    if not loaded_and_restored:
        decision["pass"] = False
        decision["reasons"].append("candidate state was not loaded and replayed exactly")
    return {
        "scale": tau_controls["scale"],
        "decision": decision,
        "tau_controls": tau_controls,
        "frozen_parameter_controls": frozen,
        "candidate_state_loaded_and_restored": loaded_and_restored,
        "candidate_state_sha256": state_sha256(candidate_state),
        "fixed": fixed,
        "fixed_blocks": fixed_blocks,
        "full": full,
        "full_blocks": full_blocks,
        "per_block_preservation": preservation,
    }


def directional_agreement(
    *,
    baseline_objectives: list[float],
    probe_objective: float,
    raw_gradient_selected: Tensor,
    source_state: dict[str, Tensor],
    probe_state: dict[str, Tensor],
    node_indices: np.ndarray,
) -> dict[str, Any]:
    replay_noise = max(
        abs(left - right)
        for index, left in enumerate(baseline_objectives)
        for right in baseline_objectives[index + 1 :]
    )
    measured = probe_objective - baseline_objectives[0]
    selected = torch.from_numpy(node_indices)
    raw_delta = (
        probe_state["raw_time_constant"][selected]
        - source_state["raw_time_constant"][selected]
    ).double()
    predicted = float((raw_gradient_selected.double() * raw_delta).sum())
    relative_error = 2.0 * abs(measured - predicted) / max(
        abs(measured) + abs(predicted), 1.0e-30
    )
    noise_threshold = max(
        MINIMUM_OBJECTIVE_CHANGE, REPLAY_NOISE_MULTIPLIER * replay_noise
    )
    finite = all(math.isfinite(value) for value in (measured, predicted, relative_error))
    passed = bool(
        finite
        and measured < 0.0
        and predicted < 0.0
        and relative_error <= FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
        and abs(measured) > noise_threshold
        and abs(predicted) > noise_threshold
    )
    return {
        "pass": passed,
        "baseline_objectives": baseline_objectives,
        "replay_noise": replay_noise,
        "objective_change_noise_threshold": noise_threshold,
        "measured_objective_change": measured,
        "predicted_objective_change": predicted,
        "symmetric_relative_error": relative_error,
        "all_values_finite": finite,
    }


def run_derivative_probe(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    source_state: dict[str, Tensor],
    direction: dict[str, Any],
    node_indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, Any]:
    probe_state, tau_controls = tau_candidate(
        source_state,
        node_indices,
        direction["unit_tau_direction_seconds"],
        scale=FINITE_DIFFERENCE_SCALE,
    )
    controller.load_state_dict(probe_state, strict=True)
    fixed, _ = capacity.fixed_cohort_report(
        controller, args, manifests, prefixes, device=device
    )
    agreement = directional_agreement(
        baseline_objectives=[
            item["objective"] for item in direction["source_fixed_replays"]
        ],
        probe_objective=fixed["objective"],
        raw_gradient_selected=direction["raw_gradient_selected"],
        source_state=source_state,
        probe_state=probe_state,
        node_indices=node_indices,
    )
    frozen = frozen_parameter_controls(source_state, probe_state, node_indices)
    agreement["pass"] = bool(
        agreement["pass"]
        and frozen["pass"]
        and tau_controls["ratio_bounds_pass"]
        and tau_controls["native_bounds_pass"]
        and fixed["all_recurrent_states_and_outputs_finite"]
        and fixed["all_metrics_finite"]
    )
    return {
        **agreement,
        "scale": FINITE_DIFFERENCE_SCALE,
        "tau_controls": tau_controls,
        "frozen_parameter_controls": frozen,
        "fixed": capacity.compact_summary(fixed),
    }


def solver_requalification(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    references: dict[str, Any],
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    candidate_rk4_blocks: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    source_rk4 = capacity.source_reference_map(references)
    records = []
    for manifest, candidate_rk4 in zip(
        manifests, candidate_rk4_blocks, strict=True
    ):
        seed = manifest["seed"]
        print(json.dumps({"stage": "solver_requalification", "seed": seed}), flush=True)
        cache = capacity._block_cache(args, manifest)
        source_k32 = capacity.full_block_report(
            args,
            source_state,
            cache,
            method="exponential_euler",
            solver_steps=32,
            device=device,
        )
        candidate_k32 = capacity.full_block_report(
            args,
            candidate_state,
            cache,
            method="exponential_euler",
            solver_steps=32,
            device=device,
        )
        source_comparison = solver.reference_comparison(
            "source RK4-M1",
            source_rk4[seed],
            "source exponential-Euler K32",
            source_k32,
            scale=rate.EXPECTED_MOTION_SCALE,
        )
        candidate_comparison = solver.reference_comparison(
            "candidate RK4-M1",
            candidate_rk4,
            "candidate exponential-Euler K32",
            candidate_k32,
            scale=rate.EXPECTED_MOTION_SCALE,
        )
        records.append(
            {
                "seed": seed,
                "source": source_comparison,
                "candidate": candidate_comparison,
                "source_k32": source_k32,
                "candidate_k32": candidate_k32,
            }
        )
    source_pass = all(item["source"]["pass"] for item in records)
    candidate_pass = all(item["candidate"]["pass"] for item in records)
    return {
        "pass": bool(source_pass and candidate_pass),
        "source_all_blocks_pass": source_pass,
        "candidate_all_blocks_pass": candidate_pass,
        "blocks": records,
    }


def development_decision(
    evaluation: dict[str, Any], references: dict[str, Any] | None = None
) -> dict[str, Any]:
    summary = evaluation["combined"]
    source = evaluation["source_combined"]
    reasons = []
    if references is not None and not source_references_valid(references):
        reasons.append("development source references failed identity or finiteness checks")
    candidate_blocks = evaluation.get("blocks", [])
    if not candidate_blocks or not all(
        item.get("source_loaded_exactly")
        and item.get("source_restored")
        and replay_summary_valid(item)
        for item in evaluation.get("blocks", [])
    ):
        reasons.append("development candidate blocks failed identity or finiteness checks")
    if not all_numeric_values_finite(evaluation):
        reasons.append("a calculated development metric or drift was nonfinite")
    if summary["source_absolute_nrmse_improvement"] < MINIMUM_NRMSE_IMPROVEMENT:
        reasons.append("development NRMSE improvement was below 0.0001")
    for horizon in motion.HISTORY_LENGTHS:
        key = str(horizon)
        if (
            summary["by_horizon"][key]["nrmse"]
            > source["by_horizon"][key]["nrmse"]
            + DEVELOPMENT_HORIZON_TOLERANCE
        ):
            reasons.append(
                f"development horizon {horizon} NRMSE regressed by more than 0.00001"
            )
    if not evaluation["all_preservation_pass"]:
        reasons.append("development source-relative output preservation failed")
    if not summary["all_recurrent_states_and_outputs_finite"]:
        reasons.append("development recurrence or output was nonfinite")
    if not summary["all_metrics_finite"]:
        reasons.append("development metrics were nonfinite")
    if summary["endpoint_image_difference_max"] != 0.0:
        reasons.append("development paired endpoint images differed")
    return {"pass": not reasons, "reasons": reasons}


def run_development(
    args: argparse.Namespace,
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    *,
    candidate_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    marker_path = args.output_dir / "development-started.json"
    result_path = args.output_dir / "development-result.json"
    if result_path.is_file():
        with result_path.open() as stream:
            payload = json.load(stream)
        if (
            payload.get("semantic_sha256") != capacity._payload_semantic_hash(payload)
            or payload.get("experiment") != EXPERIMENT
            or payload.get("protocol_commit") != PROTOCOL_COMMIT
            or payload.get("candidate_state_sha256") != candidate_sha256
        ):
            raise RuntimeError("persisted development result failed validation")
        return payload
    if marker_path.exists():
        raise RuntimeError(
            "development previously started without a completed result; refusing seed reuse"
        )
    capacity.write_phase_marker(
        marker_path,
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "candidate_state_sha256": candidate_sha256,
            "seeds": list(capacity.DEVELOPMENT_SEEDS),
        },
    )
    config = HoverConfig()
    manifests = capacity.ensure_cache_manifests(
        args, "development", device=device, config=config
    )
    references = capacity.ensure_source_references(
        args,
        "development",
        manifests,
        source_state,
        device=device,
    )
    evaluation = capacity.add_source_to_evaluation(
        capacity.evaluate_cohort(
            args,
            manifests,
            references,
            candidate_state,
            device=device,
        ),
        references,
    )
    decision = development_decision(evaluation, references)
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "candidate_state_sha256": candidate_sha256,
        "cache_manifests": manifests,
        "source_references_semantic_sha256": references["semantic_sha256"],
        "source_references_file_sha256": file_sha256(
            capacity.references_path(args, "development")
        ),
        "evaluation": evaluation,
        "decision": decision,
    }
    payload["semantic_sha256"] = capacity._payload_semantic_hash(payload)
    assisted._atomic_json_save(payload, result_path)
    return payload


def compact_training_trial(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        **trial,
        "fixed": capacity.compact_summary(trial["fixed"]),
        "fixed_blocks": [
            capacity.compact_summary(item) for item in trial["fixed_blocks"]
        ],
    }


def write_report(args: argparse.Namespace, report: dict[str, Any]) -> None:
    assisted._atomic_json_save(report, args.output_dir / "report.json")


def execute_audit(
    controller: ConnectomeController,
    outcome: dict[str, Any],
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    references: dict[str, Any],
    source_state: dict[str, Tensor],
    node_indices: np.ndarray,
    *,
    source_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    direction = load_or_create_direction(
        controller,
        args,
        manifests,
        node_indices,
        source_sha256=source_sha,
        device=device,
    )
    outcome["direction"] = direction
    prefixes = prefixes_from_direction(direction)
    source_fixed = direction["source_fixed_replays"][0]
    controller.load_state_dict(source_state, strict=True)
    print(json.dumps({"stage": "finite_difference"}), flush=True)
    derivative = run_derivative_probe(
        controller,
        args,
        manifests,
        prefixes,
        source_state,
        direction,
        node_indices,
        device=device,
    )
    outcome["derivative"] = derivative

    trials: list[dict[str, Any]] = outcome["trials"]
    selected_scale: float | None = None
    selected_state: dict[str, Tensor] | None = None
    selected_trial: dict[str, Any] | None = None
    solver_check: dict[str, Any] | None = None
    development: dict[str, Any] | None = None
    passed = False
    if not derivative["pass"]:
        classification = "premotor_tau_derivative_gate_failed"
    else:
        for scale in TAU_SCALES:
            print(json.dumps({"stage": "training_candidate", "scale": scale}), flush=True)
            candidate_state, tau_controls = tau_candidate(
                source_state,
                node_indices,
                direction["unit_tau_direction_seconds"],
                scale=scale,
            )
            trial = evaluate_training_candidate(
                controller,
                args,
                manifests,
                references,
                prefixes,
                source_state,
                candidate_state,
                node_indices,
                tau_controls,
                source_fixed,
                device=device,
            )
            trials.append(trial)
            controller.load_state_dict(source_state, strict=True)
            if trial["decision"]["pass"]:
                selected_scale = scale
                selected_state = candidate_state
                selected_trial = trial
                outcome["selected_scale"] = selected_scale
                outcome["selected_state"] = selected_state
                break
        if selected_state is None or selected_trial is None:
            classification = "premotor_tau_no_bounded_training_candidate"
        else:
            solver_check = solver_requalification(
                args,
                manifests,
                references,
                source_state,
                selected_state,
                selected_trial["full_blocks"],
                device=device,
            )
            outcome["solver_check"] = solver_check
            if not solver_check["source_all_blocks_pass"]:
                classification = "premotor_tau_source_solver_requalification_failed"
            elif not solver_check["candidate_all_blocks_pass"]:
                classification = "premotor_tau_candidate_solver_requalification_failed"
            else:
                print(json.dumps({"stage": "development"}), flush=True)
                development = run_development(
                    args,
                    source_state,
                    selected_state,
                    candidate_sha256=state_sha256(selected_state),
                    device=device,
                )
                outcome["development"] = development
                if development["decision"]["pass"]:
                    classification = "bounded_premotor_tau_route_validated"
                    passed = True
                else:
                    classification = "premotor_tau_development_gate_failed"
    outcome.update(
        {
            "direction": direction,
            "derivative": derivative,
            "trials": trials,
            "selected_scale": selected_scale,
            "selected_state": selected_state,
            "solver_check": solver_check,
            "development": development,
            "classification": classification,
            "passed": passed,
        }
    )
    return outcome


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    if report_path.is_file():
        raise SystemExit("the premotor-tau audit already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    started = perf_counter()
    input_hashes, manifests, references = validate_inputs(args)
    source_state = source_checkpoint(args.checkpoint)
    source_sha = state_sha256(source_state)
    if references.get("source_state_sha256") != source_sha:
        raise SystemExit("locked source references use a different controller")
    node_indices, mask_manifest = build_tau_mask(args)
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha,
        "mask": mask_manifest,
        "device": str(device),
    }
    _write_or_validate_start(args.output_dir / "start.json", start)

    controller = ConnectomeController(
        args.graph, neural_dt=1.0 / rate.POLICY_HZ
    ).to(device)
    controller.load_state_dict(source_state, strict=True)
    outcome: dict[str, Any] = {
        "direction": None,
        "derivative": None,
        "trials": [],
        "selected_scale": None,
        "selected_state": None,
        "solver_check": None,
        "development": None,
        "classification": "premotor_tau_exception_failed_closed",
        "passed": False,
    }
    exception: dict[str, str] | None = None
    try:
        outcome = execute_audit(
            controller,
            outcome,
            args,
            manifests,
            references,
            source_state,
            node_indices,
            source_sha=source_sha,
            device=device,
        )
    except Exception as error:  # fail closed and leave a terminal audit record
        exception = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            controller.load_state_dict(source_state, strict=True)
            restored_sha: str | None = rate._source_state_sha256(controller)
        except Exception as restoration_error:
            restored_sha = None
            exception = {
                "type": type(restoration_error).__name__,
                "message": f"source restoration failed: {restoration_error}",
            }

    direction = outcome["direction"]
    derivative = outcome["derivative"]
    trials = outcome["trials"]
    selected_scale = outcome["selected_scale"]
    selected_state = outcome["selected_state"]
    solver_check = outcome["solver_check"]
    development = outcome["development"]
    classification = outcome["classification"]
    passed = outcome["passed"]
    source_restored = restored_sha == source_sha
    if not source_restored:
        passed = False
        classification = "premotor_tau_source_restoration_failed"
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha,
        "source_state_restored_sha256": restored_sha,
        "source_state_restored": source_restored,
        "mask": mask_manifest,
        "training_cache_manifests": manifests,
        "training_source_references_semantic_sha256": references["semantic_sha256"],
        "direction_archive": (
            {
                "path": assisted.responsibility.stable_path(direction_path(args)),
                "file_sha256": file_sha256(direction_path(args)),
                "semantic_sha256": direction["semantic_sha256"],
                "direction_controls": direction["direction_controls"],
                "gradient_controls": direction["gradient_controls"],
                "source_fixed_replays": direction["source_fixed_replays"],
                "combined_gradient_summary": direction["combined_gradient_summary"],
            }
            if direction is not None
            else None
        ),
        "source_full": capacity.compact_summary(references["combined"]),
        "finite_difference": derivative,
        "training_trials": [compact_training_trial(item) for item in trials],
        "selected_scale": selected_scale,
        "selected_candidate_state_sha256": (
            state_sha256(selected_state) if selected_state is not None else None
        ),
        "solver_requalification": solver_check,
        "development": development,
        "development_result_file_sha256": (
            file_sha256(args.output_dir / "development-result.json")
            if development is not None
            else None
        ),
        "exception": exception,
        "bounded_premotor_tau_fit_preregistration_authorized": passed,
        "candidate_retained": False,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": perf_counter() - started,
    }
    write_report(args, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "classification": classification,
                "passed": passed,
                "selected_scale": selected_scale,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
