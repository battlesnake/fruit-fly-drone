#!/usr/bin/env python3
"""Fit the native RK4 throttle-motor readout under immutable output constraints."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_continuous_cns_solver as solver  # noqa: E402
import audit_variable_height_neural_integration_rate as rate  # noqa: E402
import audit_variable_height_rk4_throttle_readout_step as readout  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import train_variable_height_native_throttle_motion_only as motion  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402

EXPERIMENT = "variable-height-rk4-readout-capacity-v1"
PROTOCOL_COMMIT = "3cda916"
EXPECTED_CANONICAL_REPORT_SHA256 = (
    "cd65d0d77fe88af1335d7b6baf1b57cf1b95c9e22bfa7962324dbc80cfc99d73"
)
EXPECTED_BASE_CACHE_FILE_SHA256 = solver.EXPECTED_CACHE_FILE_SHA256
EXPECTED_BASE_CACHE_SEMANTIC_SHA256 = solver.EXPECTED_CACHE_SEMANTIC_SHA256

TRAINING_SEEDS = (450_991, 450_992, 450_993, 450_994)
DEVELOPMENT_SEEDS = (460_991, 460_992)
QUALIFICATION_SEEDS = tuple(range(470_991, 470_999))
FORMAL_SEEDS = frozenset(TRAINING_SEEDS + DEVELOPMENT_SEEDS + QUALIFICATION_SEEDS)
TRAINING_CASES = 96
DEVELOPMENT_CASES = 48
QUALIFICATION_CASES = 192

ORDINARY_SCALES = (16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
DERIVATIVE_SCALES = (1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512)
DERIVATIVE_REPEATS = 3
DERIVATIVE_RELATIVE_ERROR_LIMIT = 0.20
DERIVATIVE_NOISE_MULTIPLIER = 10.0
DERIVATIVE_MINIMUM_CHANGE = 1.0e-8
MINIMUM_CURRENT_NRMSE_IMPROVEMENT = 0.001
MAXIMUM_ACCEPTED_UPDATES = 50
MILESTONE_UPDATE = 10
MILESTONE_TRAINING_IMPROVEMENT = 0.02
MILESTONE_DEVELOPMENT_IMPROVEMENT = 0.01
MILESTONE_HORIZON_REGRESSION_LIMIT = 0.01
FINAL_NRMSE_MAXIMUM = 0.5
FINAL_SIGN_FRACTION = 0.90
FINAL_GAIN_RANGE = (0.5, 1.5)


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
        "--canonical-report",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-rk4-readout-trust-region-canonical-001/report.json",
    )
    parser.add_argument(
        "--base-cache",
        type=Path,
        default=REPO_ROOT
        / "runs/variable-height-hover/native-neural-integration-rate-audit-001/input-cache.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/native-rk4-readout-capacity-001",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "authorizing_report_sha256": EXPECTED_CANONICAL_REPORT_SHA256,
        "solver": "RK4-M1: one 20 ms step and four recurrent graph evaluations",
        "actor": {
            "inputs": ["cached 320x200 linear RGB", "roll", "pitch"],
            "external_or_engineered_state": False,
            "outputs": ["roll", "pitch", "yaw", "throttle"],
        },
        "blocks": {
            "training_seeds": list(TRAINING_SEEDS),
            "training_cases": TRAINING_CASES,
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "development_cases": DEVELOPMENT_CASES,
            "qualification_seeds": list(QUALIFICATION_SEEDS),
            "qualification_cases": QUALIFICATION_CASES,
            "cases_per_block": motion.CASES,
            "exact_factorial_per_block": True,
            "development_generated_after_update_10_training_pass": True,
            "qualification_generated_after_terminal_training_and_development_pass": True,
            "streamed_one_block_at_a_time": True,
        },
        "readout_mask": {
            "edge_count": readout.MASKED_EDGE_COUNT,
            "node_count": readout.MASKED_NODE_COUNT,
            "edge_indices_sha256": readout.EXPECTED_EDGE_INDICES_SHA256,
            "node_indices_sha256": readout.EXPECTED_NODE_INDICES_SHA256,
            "node_body_ids_sha256": readout.EXPECTED_NODE_BODY_IDS_SHA256,
        },
        "optimizer": {
            "name": "Adam",
            "betas": list(readout.OPTIMIZER_BETAS),
            "epsilon": readout.OPTIMIZER_EPSILON,
            "edge_and_bias_learning_rate": readout.EDGE_BIAS_LEARNING_RATE,
            "raw_time_constant_learning_rate": readout.TIME_CONSTANT_LEARNING_RATE,
            "gradient_norm_cap": readout.GRADIENT_NORM_CAP,
            "pending_transaction_persisted_before_replay": True,
            "interrupted_transaction_regenerated": False,
        },
        "objective": {
            "motion_contrast_only": True,
            "combined_training_pairs": TRAINING_CASES,
            "current_native_prefix_recomputed_each_update": True,
            "detached_prefix_is_training_only": True,
            "complete_zero_state_replay_required": True,
            "minimum_current_nrmse_improvement_fixed_and_full": (
                MINIMUM_CURRENT_NRMSE_IMPROVEMENT
            ),
        },
        "derivative_probe": {
            "baseline_repeats": DERIVATIVE_REPEATS,
            "scales": list(DERIVATIVE_SCALES),
            "required_adjacent_passes": 2,
            "relative_error_limit": DERIVATIVE_RELATIVE_ERROR_LIMIT,
            "noise_multiplier": DERIVATIVE_NOISE_MULTIPLIER,
            "minimum_change": DERIVATIVE_MINIMUM_CHANGE,
            "selects_update": False,
        },
        "ordinary_scales_descending": list(ORDINARY_SCALES),
        "preservation_per_block": {
            "source_relative_common_throttle_rms": readout.PAIR_COMMON_RMS_LIMIT,
            "source_relative_common_throttle_maximum": readout.PAIR_COMMON_MAX_LIMIT,
            "source_relative_rpy_rms_per_axis": readout.RPY_RMS_LIMIT,
            "source_relative_rpy_maximum_per_axis": readout.RPY_MAX_LIMIT,
            "averaging_across_blocks_allowed": False,
        },
        "first_finite_no_scale_is_terminal": True,
        "candidate_projection_or_repair": False,
        "maximum_accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
        "milestone_10": {
            "training_absolute_nrmse_improvement": MILESTONE_TRAINING_IMPROVEMENT,
            "development_absolute_nrmse_improvement": MILESTONE_DEVELOPMENT_IMPROVEMENT,
            "maximum_per_horizon_development_nrmse_regression": (
                MILESTONE_HORIZON_REGRESSION_LIMIT
            ),
        },
        "terminal": {
            "nrmse_maximum_overall_and_per_horizon": FINAL_NRMSE_MAXIMUM,
            "correct_sign_fraction_minimum_overall_and_per_horizon": FINAL_SIGN_FRACTION,
            "gain_range_overall_and_per_horizon": list(FINAL_GAIN_RANGE),
            "qualification_once": True,
            "rk4_vs_k32_normalized_contrast_rms_maximum": rate.CONTRAST_REFINEMENT_LIMIT,
            "rk4_vs_k32_terminal_motor_rms_maximum": rate.MOTOR_REFINEMENT_LIMIT,
        },
        "authorizes_hover_gate_or_promotion": False,
    }


def require_nonformal_seeds(*seeds: int) -> None:
    collisions = sorted(FORMAL_SEEDS.intersection(seeds))
    if collisions:
        raise ValueError(f"disposable execution attempted to consume formal seeds: {collisions}")


def _clone_state(values: dict[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in values.items()}


def _cache_payload_hash(cache: dict[str, Any]) -> str:
    return assisted.audit.semantic_sha256(
        {name: value for name, value in cache.items() if name != "semantic_sha256"}
    )


def block_specs(cohort: str) -> list[dict[str, Any]]:
    if cohort == "training":
        seeds, held_out = TRAINING_SEEDS, False
    elif cohort == "development":
        seeds, held_out = DEVELOPMENT_SEEDS, True
    elif cohort == "qualification":
        seeds, held_out = QUALIFICATION_SEEDS, True
    else:
        raise ValueError(f"unknown cohort: {cohort}")
    return [
        {"cohort": cohort, "seed": seed, "held_out_styles": held_out}
        for seed in seeds
    ]


def cache_path(args: argparse.Namespace, spec: dict[str, Any]) -> Path:
    if spec["cohort"] == "training" and spec["seed"] == TRAINING_SEEDS[0]:
        return args.base_cache
    return args.output_dir / "caches" / f"{spec['cohort']}-{spec['seed']}.pt"


def validate_cache(
    cache: dict[str, Any], spec: dict[str, Any], *, verify_semantic_hash: bool = True
) -> None:
    if int(cache.get("cases", -1)) != motion.CASES:
        raise SystemExit("capacity cache case count mismatch")
    if int(cache.get("block_seed", -1)) != spec["seed"]:
        raise SystemExit("capacity cache seed mismatch")
    if bool(cache.get("held_out_styles")) != spec["held_out_styles"]:
        raise SystemExit("capacity cache style split mismatch")
    if verify_semantic_hash and cache.get("semantic_sha256") != _cache_payload_hash(cache):
        raise SystemExit("capacity cache semantic hash mismatch")
    if float(cache.get("teacher_scale", -1.0)) != rate.EXPECTED_MOTION_SCALE:
        raise SystemExit("capacity cache teacher scale mismatch")
    if not torch.equal(cache["case_order"], torch.arange(motion.CASES)):
        raise SystemExit("capacity cache case order mismatch")
    if cache["prefix_images"].shape != (motion.CASES, 3, 200, 320):
        raise SystemExit("capacity cache prefix image shape mismatch")
    if cache["prefix_roll_pitch"].shape != (motion.CASES, 2):
        raise SystemExit("capacity cache prefix attitude shape mismatch")
    if set(cache["groups"]) != {str(value) for value in motion.HISTORY_LENGTHS}:
        raise SystemExit("capacity cache horizon groups mismatch")
    if float(cache["endpoint_image_difference"].max()) != 0.0:
        raise SystemExit("capacity cache opposite-motion endpoints differ")
    tensors: list[Tensor] = [
        cache["prefix_images"],
        cache["prefix_roll_pitch"],
        cache["teacher_targets"],
        cache["endpoint_image_difference"],
    ]
    for horizon in motion.HISTORY_LENGTHS:
        group = cache["groups"][str(horizon)]
        count = int((cache["horizon"] == horizon).sum())
        if group["response_images"].shape != (horizon, count, 2, 3, 200, 320):
            raise SystemExit(f"capacity cache horizon-{horizon} image shape mismatch")
        if group["roll_pitch"].shape != (count, 2, 2):
            raise SystemExit(f"capacity cache horizon-{horizon} attitude shape mismatch")
        tensors.extend((group["response_images"], group["roll_pitch"]))
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise SystemExit("capacity cache contains a nonfinite tensor")


@torch.inference_mode()
def build_cache(
    spec: dict[str, Any], *, device: torch.device, config: HoverConfig
) -> dict[str, Any]:
    bank = motion.build_exact_factorial_motion_bank(
        seed=spec["seed"],
        held_out_styles=spec["held_out_styles"],
        device=device,
        config=config,
    )
    cache = rate.build_input_cache(bank, device=device, config=config)
    cache.update(
        {
            "experiment": EXPERIMENT,
            "cohort": spec["cohort"],
            "block_seed": spec["seed"],
            "held_out_styles": spec["held_out_styles"],
            "source_bank_sha256": bank["sha256"],
            "teacher_scale": rate.EXPECTED_MOTION_SCALE,
        }
    )
    cache["semantic_sha256"] = _cache_payload_hash(cache)
    validate_cache(cache, spec)
    return cache


def load_cache(
    path: Path, spec: dict[str, Any], *, verify_semantic_hash: bool = True
) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if spec["cohort"] == "training" and spec["seed"] == TRAINING_SEEDS[0]:
        if verify_semantic_hash:
            rate.validate_cache(cache)
        if cache.get("semantic_sha256") != EXPECTED_BASE_CACHE_SEMANTIC_SHA256:
            raise SystemExit("base training cache semantic hash mismatch")
        cache = dict(cache)
        cache.update(
            {
                "cohort": "training",
                "block_seed": TRAINING_SEEDS[0],
                "held_out_styles": False,
            }
        )
        if verify_semantic_hash:
            cache["semantic_sha256"] = _cache_payload_hash(cache)
    validate_cache(cache, spec, verify_semantic_hash=verify_semantic_hash)
    return cache


def ensure_cache(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    device: torch.device,
    config: HoverConfig,
) -> dict[str, Any]:
    path = cache_path(args, spec)
    if path.is_file():
        cache = load_cache(path, spec)
    else:
        if path == args.base_cache:
            raise SystemExit("immutable base cache is missing")
        print(json.dumps({"stage": "render_cache", **spec}), flush=True)
        cache = build_cache(spec, device=device, config=config)
        assisted._atomic_torch_save(cache, path)
        if assisted.responsibility.file_sha256(path) == "":
            raise SystemExit("failed to hash generated cache")
    return {
        "cohort": spec["cohort"],
        "seed": spec["seed"],
        "held_out_styles": spec["held_out_styles"],
        "path": assisted.responsibility.stable_path(path),
        "file_sha256": assisted.responsibility.file_sha256(path),
        "semantic_sha256": cache["semantic_sha256"],
        "bank_sha256": cache["source_bank_sha256"],
        "cases": int(cache["cases"]),
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    expected = {
        args.graph: assisted.EXPECTED_GRAPH_SHA256,
        args.checkpoint: assisted.EXPECTED_CHECKPOINT_SHA256,
        args.canonical_report: EXPECTED_CANONICAL_REPORT_SHA256,
        args.base_cache: EXPECTED_BASE_CACHE_FILE_SHA256,
    }
    observed = {}
    for path, digest in expected.items():
        if not path.is_file():
            raise SystemExit(f"missing preregistered input: {path}")
        actual = assisted.responsibility.file_sha256(path)
        if actual != digest:
            raise SystemExit(f"preregistered input hash mismatch: {path}")
        observed[assisted.responsibility.stable_path(path)] = actual
    with args.canonical_report.open() as stream:
        report = json.load(stream)
    if (
        report.get("classification") != "canonical_direct_readout_trust_region_exists"
        or not report.get("passed")
        or report.get("selected_scale") != 8.0
        or not report.get("bounded_last_hop_fitting_preregistration_authorized")
    ):
        raise SystemExit("canonical trust-region report does not authorize this fit")
    return observed


def _write_or_validate_start(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open() as stream:
            current = json.load(stream)
        if current != payload:
            raise SystemExit("capacity-fit start marker differs from this invocation")
        return
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_source_checkpoint(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return _clone_state(payload["controller"])


def _block_cache(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    spec = {
        "cohort": manifest["cohort"],
        "seed": manifest["seed"],
        "held_out_styles": manifest["held_out_styles"],
    }
    return load_cache(
        REPO_ROOT / manifest["path"], spec, verify_semantic_hash=False
    )


def combine_summaries(
    summaries: list[dict[str, Any]], caches: list[dict[str, Any]], *, scale: float
) -> dict[str, Any]:
    if len(summaries) != len(caches) or not summaries:
        raise ValueError("combined summary requires matching nonempty blocks")
    outputs = torch.cat(
        [torch.tensor(item["terminal_motor_outputs"]) for item in summaries], dim=0
    )
    targets = torch.cat([cache["teacher_targets"] for cache in caches])
    horizons = torch.cat([cache["horizon"] for cache in caches])
    combined = rate.summarize_outputs(
        outputs,
        targets,
        horizons,
        scale=scale,
        recurrence_finite=all(
            item["all_recurrent_states_and_outputs_finite"] for item in summaries
        ),
        endpoint_difference_max=max(
            item["endpoint_image_difference_max"] for item in summaries
        ),
        wall_time_seconds=sum(item.get("wall_time_seconds", 0.0) for item in summaries),
        recurrent_calls=sum(item.get("native_recurrent_calls", 0) for item in summaries),
        branch_state_updates=sum(
            item.get("native_branch_state_updates", 0) for item in summaries
        ),
    )
    combined["block_count"] = len(summaries)
    combined["objective"] = combined["nrmse"] ** 2
    return combined


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "nrmse",
        "objective",
        "pairs",
        "correct_sign_fraction",
        "teacher_aligned_gain",
        "prediction_rms_motor_units",
        "target_rms_motor_units",
        "endpoint_image_difference_max",
        "all_recurrent_states_and_outputs_finite",
        "all_metrics_finite",
        "maximum_motor_absolute",
        "by_horizon",
    )
    return {name: summary[name] for name in keys if name in summary}


def _load_caches(
    args: argparse.Namespace, manifests: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [_block_cache(args, manifest) for manifest in manifests]


def full_block_report(
    args: argparse.Namespace,
    state: dict[str, Tensor],
    cache: dict[str, Any],
    *,
    method: str,
    solver_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    return solver.evaluate_condition(
        graph=args.graph,
        checkpoint_state=state,
        cache=cache,
        method=method,
        solver_steps=solver_steps,
        device=device,
    )


def source_references(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    source_state: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    caches, blocks = [], []
    for manifest in manifests:
        cache = _block_cache(args, manifest)
        caches.append(
            {"teacher_targets": cache["teacher_targets"], "horizon": cache["horizon"]}
        )
        blocks.append(
            {
                "seed": manifest["seed"],
                "full": full_block_report(
                    args, source_state, cache, method="rk4", solver_steps=1, device=device
                ),
            }
        )
    combined = combine_summaries(
        [item["full"] for item in blocks],
        caches,
        scale=rate.EXPECTED_MOTION_SCALE,
    )
    return {"blocks": blocks, "combined": combined}


def source_reference_map(references: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["seed"]): item["full"] for item in references["blocks"]}


@torch.no_grad()
def current_prefixes(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[int, Tensor]:
    result = {}
    for manifest in manifests:
        cache = _block_cache(args, manifest)
        result[manifest["seed"]] = readout.source_prefix_states(
            controller, cache, device=device
        ).detach().cpu()
    return result


@torch.no_grad()
def fixed_cohort_report(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    caches, blocks = [], []
    for manifest in manifests:
        cache = _block_cache(args, manifest)
        caches.append(
            {"teacher_targets": cache["teacher_targets"], "horizon": cache["horizon"]}
        )
        _, summary = readout.fixed_prefix_objective(
            controller,
            cache,
            prefixes[manifest["seed"]].to(device),
            device=device,
            backward=False,
        )
        blocks.append(summary)
    return combine_summaries(blocks, caches, scale=rate.EXPECTED_MOTION_SCALE), blocks


def accumulated_masked_gradient(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    mask: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    for parameter in controller.parameters():
        parameter.grad = None
    objectives = []
    summaries = []
    caches = []
    for manifest in manifests:
        cache = _block_cache(args, manifest)
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
        objectives.append(float(objective.detach()))
        summaries.append(summary)
    block_count = len(manifests)
    for name in readout.PARAMETER_FAMILIES:
        parameter = getattr(controller, name)
        if parameter.grad is None:
            raise RuntimeError(f"missing accumulated gradient for {name}")
        parameter.grad.div_(block_count)
    clipped, controls = readout._masked_gradients(controller, mask)
    combined = combine_summaries(summaries, caches, scale=rate.EXPECTED_MOTION_SCALE)
    controls.update(
        {
            "combined_objective": combined["objective"],
            "block_objectives": objectives,
            "combined": compact_summary(combined),
        }
    )
    return clipped, controls


def full_cohort_report(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    state: dict[str, Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    caches, blocks = [], []
    for manifest in manifests:
        cache = _block_cache(args, manifest)
        caches.append(
            {"teacher_targets": cache["teacher_targets"], "horizon": cache["horizon"]}
        )
        blocks.append(
            full_block_report(
                args, state, cache, method="rk4", solver_steps=1, device=device
            )
        )
    return combine_summaries(blocks, caches, scale=rate.EXPECTED_MOTION_SCALE), blocks


def block_preservation_decision(
    source: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    preservation = readout.output_preservation(
        torch.tensor(source["terminal_motor_outputs"]),
        torch.tensor(candidate["terminal_motor_outputs"]),
    )
    reasons = []
    if preservation["pair_common_throttle_drift_rms"] > readout.PAIR_COMMON_RMS_LIMIT:
        reasons.append("pair-common throttle RMS drift exceeded 0.005")
    if (
        preservation["pair_common_throttle_drift_maximum_absolute"]
        > readout.PAIR_COMMON_MAX_LIMIT
    ):
        reasons.append("pair-common throttle maximum drift exceeded 0.01")
    for axis, values in preservation["rpy_drift"].items():
        if values["rms"] > readout.RPY_RMS_LIMIT:
            reasons.append(f"{axis} RMS drift exceeded 0.005")
        if values["maximum_absolute"] > readout.RPY_MAX_LIMIT:
            reasons.append(f"{axis} maximum drift exceeded 0.01")
    if not candidate["all_recurrent_states_and_outputs_finite"]:
        reasons.append("recurrence or outputs were nonfinite")
    if not candidate["all_metrics_finite"]:
        reasons.append("metrics were nonfinite")
    if candidate["endpoint_image_difference_max"] != 0.0:
        reasons.append("paired endpoint images differed")
    motor = torch.tensor(candidate["terminal_motor_outputs"])
    maximum_motor_absolute = float(motor.abs().max())
    if maximum_motor_absolute > 1.0:
        reasons.append("motor output left [-1,1]")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "preservation": preservation,
        "maximum_motor_absolute": maximum_motor_absolute,
    }


def candidate_decision(
    *,
    fixed: dict[str, Any],
    full: dict[str, Any],
    current_fixed: dict[str, Any],
    current_full: dict[str, Any],
    source_blocks: dict[int, dict[str, Any]],
    candidate_blocks: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    mask: dict[str, Any],
) -> dict[str, Any]:
    fixed_improvement = current_fixed["nrmse"] - fixed["nrmse"]
    full_improvement = current_full["nrmse"] - full["nrmse"]
    reasons = []
    if fixed_improvement < MINIMUM_CURRENT_NRMSE_IMPROVEMENT:
        reasons.append("detached-prefix current NRMSE improvement was below 0.001")
    if full_improvement < MINIMUM_CURRENT_NRMSE_IMPROVEMENT:
        reasons.append("zero-state current NRMSE improvement was below 0.001")
    block_decisions = []
    for manifest, candidate in zip(manifests, candidate_blocks, strict=True):
        decision = block_preservation_decision(source_blocks[manifest["seed"]], candidate)
        block_decisions.append({"seed": manifest["seed"], **decision})
        if not decision["pass"]:
            reasons.append(f"block {manifest['seed']} failed source-relative preservation")
    native_bounds_pass = bool(
        (candidate_state["edge_magnitude"] >= 0).all()
        and (candidate_state["edge_magnitude"] <= 8).all()
    )
    outside_mask_exact = readout.outside_mask_equal(source_state, candidate_state, mask)
    if not native_bounds_pass:
        reasons.append("native edge bounds failed")
    if not outside_mask_exact:
        reasons.append("a parameter outside the readout mask changed")
    if not fixed["all_recurrent_states_and_outputs_finite"] or not fixed["all_metrics_finite"]:
        reasons.append("detached-prefix combined replay was nonfinite")
    if not full["all_recurrent_states_and_outputs_finite"] or not full["all_metrics_finite"]:
        reasons.append("zero-state combined replay was nonfinite")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "fixed_nrmse_improvement_from_current": fixed_improvement,
        "full_nrmse_improvement_from_current": full_improvement,
        "block_decisions": block_decisions,
        "native_bounds_pass": native_bounds_pass,
        "outside_mask_parameters_exact": outside_mask_exact,
    }


def evaluate_candidate(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    source_references_value: dict[str, Any],
    source_state: dict[str, Tensor],
    candidate_state: dict[str, Tensor],
    current_fixed: dict[str, Any],
    current_full: dict[str, Any],
    mask: dict[str, Any],
    *,
    scale: float,
    device: torch.device,
) -> dict[str, Any]:
    controller.load_state_dict(candidate_state, strict=True)
    fixed, fixed_blocks = fixed_cohort_report(
        controller, args, manifests, prefixes, device=device
    )
    full, full_blocks = full_cohort_report(args, manifests, candidate_state, device=device)
    decision = candidate_decision(
        fixed=fixed,
        full=full,
        current_fixed=current_fixed,
        current_full=current_full,
        source_blocks=source_reference_map(source_references_value),
        candidate_blocks=full_blocks,
        manifests=manifests,
        source_state=source_state,
        candidate_state=candidate_state,
        mask=mask,
    )
    return {
        "scale": scale,
        "decision": decision,
        "fixed": fixed,
        "full": full,
        "fixed_blocks": fixed_blocks,
        "full_blocks": full_blocks,
    }


def derivative_decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    passing_windows = [
        [records[index]["scale"], records[index + 1]["scale"]]
        for index in range(len(records) - 1)
        if records[index]["pass"] and records[index + 1]["pass"]
    ]
    numerical_failure = any(record["numerical_failure"] for record in records)
    if numerical_failure:
        classification = "derivative_probe_numerical_failure"
    elif passing_windows:
        classification = "adjacent_local_derivative_agreement"
    else:
        classification = "derivative_probe_noise_limited_or_nonconvergent"
    return {
        "pass": bool(passing_windows and not numerical_failure),
        "classification": classification,
        "passing_adjacent_scale_windows": passing_windows,
    }


def run_derivative_probe(
    controller: ConnectomeController,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    prefixes: dict[int, Tensor],
    current_state: dict[str, Tensor],
    pending_state: dict[str, Tensor],
    gradients: dict[str, Tensor],
    baseline_objectives: list[float],
    *,
    device: torch.device,
) -> dict[str, Any]:
    replay_noise = max(
        abs(left - right)
        for index, left in enumerate(baseline_objectives)
        for right in baseline_objectives[index + 1 :]
    )
    noise_threshold = max(
        DERIVATIVE_MINIMUM_CHANGE, DERIVATIVE_NOISE_MULTIPLIER * replay_noise
    )
    records = []
    for scale in DERIVATIVE_SCALES:
        candidate = readout.materialize_candidate(current_state, pending_state, scale=scale)
        controller.load_state_dict(candidate, strict=True)
        fixed, _ = fixed_cohort_report(controller, args, manifests, prefixes, device=device)
        measured_change = fixed["objective"] - baseline_objectives[0]
        predicted_change = sum(
            float(
                (
                    gradients[name].detach().cpu().double()
                    * (candidate[name] - current_state[name]).double()
                ).sum()
            )
            for name in readout.PARAMETER_FAMILIES
        )
        measured = measured_change / scale
        predicted = predicted_change / scale
        relative_error = 2.0 * abs(measured - predicted) / max(
            abs(measured) + abs(predicted), 1.0e-30
        )
        finite = all(
            math.isfinite(value)
            for value in (measured_change, predicted_change, measured, predicted, relative_error)
        ) and fixed["all_recurrent_states_and_outputs_finite"] and fixed["all_metrics_finite"]
        passed = bool(
            finite
            and measured < -1.0e-8
            and predicted < -1.0e-8
            and relative_error <= DERIVATIVE_RELATIVE_ERROR_LIMIT
            and abs(measured_change) > noise_threshold
        )
        records.append(
            {
                "scale": scale,
                "pass": passed,
                "numerical_failure": not finite,
                "measured_objective_change": measured_change,
                "predicted_objective_change": predicted_change,
                "measured_directional_derivative": measured,
                "predicted_directional_derivative": predicted,
                "symmetric_relative_error": relative_error,
                "objective_change_exceeds_noise": abs(measured_change) > noise_threshold,
                "fixed": compact_summary(fixed),
            }
        )
        controller.load_state_dict(current_state, strict=True)
        if not finite or (
            len(records) >= 2 and records[-1]["pass"] and records[-2]["pass"]
        ):
            break
    return {
        **derivative_decision(records),
        "baseline_objectives": baseline_objectives,
        "replay_noise": replay_noise,
        "objective_change_noise_threshold": noise_threshold,
        "records": records,
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: _cpu_tree(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _payload_semantic_hash(payload: dict[str, Any]) -> str:
    return assisted.audit.semantic_sha256(
        {name: value for name, value in payload.items() if name != "semantic_sha256"}
    )


def pending_path(args: argparse.Namespace, attempt: int) -> Path:
    return args.output_dir / "transactions" / f"attempt-{attempt:03d}.pt"


def build_pending_transaction(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    mask: dict[str, Any],
    current_state: dict[str, Tensor],
    *,
    attempt: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer_before = _cpu_tree(optimizer.state_dict())
    prefixes = current_prefixes(controller, args, manifests, device=device)
    current_replays = []
    for _ in range(DERIVATIVE_REPEATS):
        combined, _ = fixed_cohort_report(
            controller, args, manifests, prefixes, device=device
        )
        current_replays.append(combined)
    current_full, _ = full_cohort_report(args, manifests, current_state, device=device)
    gradients, gradient_controls = accumulated_masked_gradient(
        controller, args, manifests, prefixes, mask, device=device
    )
    optimizer.step()
    controller.project_parameters()
    pending_state = _clone_state(controller.state_dict())
    optimizer_after = _cpu_tree(optimizer.state_dict())
    displacement = readout.parameter_displacement(current_state, pending_state)
    directional = sum(
        float(
            (
                gradients[name].detach().cpu().double()
                * displacement[name].double()
            ).sum()
        )
        for name in readout.PARAMETER_FAMILIES
    )
    counters_before = assisted.optimizer_step_counters(optimizer_before)
    counters_after = assisted.optimizer_step_counters(optimizer_after)
    counters_pass = bool(
        (not counters_before and counters_after == [1.0, 1.0, 1.0])
        or (
            len(counters_before) == len(counters_after) == 3
            and all(
                after == before + 1.0
                for before, after in zip(counters_before, counters_after, strict=True)
            )
        )
    )
    controller.load_state_dict(current_state, strict=True)
    optimizer.load_state_dict(optimizer_before)
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "attempt": attempt,
        "current_state": current_state,
        "pending_state": pending_state,
        "optimizer_before": optimizer_before,
        "optimizer_after": optimizer_after,
        "prefixes": prefixes,
        "clipped_gradients": _cpu_tree(gradients),
        "current_fixed_replays": current_replays,
        "current_full": current_full,
        "gradient_controls": gradient_controls,
        "directional_derivative": directional,
        "optimizer_counters_before": counters_before,
        "optimizer_counters_after": counters_after,
        "optimizer_counters_pass": counters_pass,
        "outside_mask_pending_exact": readout.outside_mask_equal(
            current_state, pending_state, mask
        ),
        "pending_native_bounds_pass": bool(
            (pending_state["edge_magnitude"] >= 0).all()
            and (pending_state["edge_magnitude"] <= 8).all()
        ),
    }
    payload["semantic_sha256"] = _payload_semantic_hash(payload)
    return payload


def validate_pending(
    payload: dict[str, Any],
    *,
    attempt: int,
    current_state: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
) -> None:
    if payload.get("experiment") != EXPERIMENT or payload.get("attempt") != attempt:
        raise SystemExit("persisted transaction identity mismatch")
    if payload.get("semantic_sha256") != _payload_semantic_hash(payload):
        raise SystemExit("persisted transaction semantic hash mismatch")
    if assisted.audit.semantic_sha256(payload["current_state"]) != assisted.audit.semantic_sha256(
        current_state
    ):
        raise SystemExit("persisted transaction controller source mismatch")
    observed_optimizer = assisted.audit.semantic_sha256(optimizer.state_dict())
    if assisted.audit.semantic_sha256(payload["optimizer_before"]) != observed_optimizer:
        raise SystemExit("persisted transaction optimizer source mismatch")


def load_or_create_pending(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    mask: dict[str, Any],
    current_state: dict[str, Tensor],
    *,
    attempt: int,
    device: torch.device,
) -> tuple[dict[str, Any], bool]:
    path = pending_path(args, attempt)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        validate_pending(
            payload,
            attempt=attempt,
            current_state=current_state,
            optimizer=optimizer,
        )
        return payload, True
    print(json.dumps({"stage": "build_pending_transaction", "attempt": attempt}), flush=True)
    payload = build_pending_transaction(
        controller,
        optimizer,
        args,
        manifests,
        mask,
        current_state,
        attempt=attempt,
        device=device,
    )
    assisted._atomic_torch_save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    validate_pending(
        reloaded,
        attempt=attempt,
        current_state=current_state,
        optimizer=optimizer,
    )
    return reloaded, False


def compact_trial(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": trial["scale"],
        "decision": trial["decision"],
        "fixed": compact_summary(trial["fixed"]),
        "full": compact_summary(trial["full"]),
    }


def run_attempt(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    references: dict[str, Any],
    source_state: dict[str, Tensor],
    mask: dict[str, Any],
    *,
    attempt: int,
    device: torch.device,
) -> dict[str, Any]:
    current_state = _clone_state(controller.state_dict())
    current_state_sha256 = assisted.audit.semantic_sha256(current_state)
    optimizer_before_sha256 = assisted.audit.semantic_sha256(optimizer.state_dict())
    pending, resumed_pending = load_or_create_pending(
        controller,
        optimizer,
        args,
        manifests,
        mask,
        current_state,
        attempt=attempt,
        device=device,
    )
    numerical_reasons = []
    gradient_controls = pending["gradient_controls"]
    if not gradient_controls["outside_mask_gradients_exactly_zero"]:
        numerical_reasons.append("gradient outside the readout mask was nonzero")
    if not gradient_controls["all_masked_gradients_nonzero"]:
        numerical_reasons.append("a masked parameter family had no gradient")
    if not pending["optimizer_counters_pass"]:
        numerical_reasons.append("Adam counters did not advance exactly once")
    if not pending["outside_mask_pending_exact"]:
        numerical_reasons.append("pending transaction changed an unmasked parameter")
    if not pending["pending_native_bounds_pass"]:
        numerical_reasons.append("pending transaction violated native edge bounds")
    if not math.isfinite(pending["directional_derivative"]) or pending[
        "directional_derivative"
    ] >= 0.0:
        numerical_reasons.append("pending transaction was not a finite descent direction")
    current_fixed_replays = pending["current_fixed_replays"]
    current_fixed = current_fixed_replays[0]
    current_full = pending["current_full"]
    if not all(
        item["all_recurrent_states_and_outputs_finite"] and item["all_metrics_finite"]
        for item in (*current_fixed_replays, current_full)
    ):
        numerical_reasons.append("current controller replay was nonfinite")

    derivative = {
        "pass": False,
        "classification": "not_run_due_to_prior_numerical_failure",
    }
    if not numerical_reasons:
        print(json.dumps({"stage": "derivative_probe", "attempt": attempt}), flush=True)
        derivative = run_derivative_probe(
            controller,
            args,
            manifests,
            pending["prefixes"],
            pending["current_state"],
            pending["pending_state"],
            pending["clipped_gradients"],
            [item["objective"] for item in current_fixed_replays],
            device=device,
        )
        if not derivative["pass"]:
            numerical_reasons.append(derivative["classification"])
    controller.load_state_dict(current_state, strict=True)

    trials = []
    selected_state = None
    selected = None
    if not numerical_reasons:
        for scale in ORDINARY_SCALES:
            print(
                json.dumps(
                    {"stage": "ordinary_candidate", "attempt": attempt, "scale": scale}
                ),
                flush=True,
            )
            candidate_state = readout.materialize_candidate(
                pending["current_state"], pending["pending_state"], scale=scale
            )
            trial = evaluate_candidate(
                controller,
                args,
                manifests,
                pending["prefixes"],
                references,
                source_state,
                candidate_state,
                current_fixed,
                current_full,
                mask,
                scale=scale,
                device=device,
            )
            trials.append(compact_trial(trial))
            controller.load_state_dict(current_state, strict=True)
            if trial["decision"]["pass"]:
                selected = trial
                selected_state = candidate_state
                break

    if selected is None:
        controller.load_state_dict(current_state, strict=True)
        optimizer.load_state_dict(pending["optimizer_before"])
    else:
        controller.load_state_dict(selected_state, strict=True)
        optimizer.load_state_dict(pending["optimizer_after"])
    terminal_controller_sha256 = assisted.audit.semantic_sha256(controller.state_dict())
    terminal_optimizer_sha256 = assisted.audit.semantic_sha256(optimizer.state_dict())
    restored = bool(
        selected is not None
        or (
            terminal_controller_sha256 == current_state_sha256
            and terminal_optimizer_sha256 == optimizer_before_sha256
        )
    )
    return {
        "attempt": attempt,
        "accepted": selected is not None,
        "accepted_scale": None if selected is None else selected["scale"],
        "resumed_persisted_transaction": resumed_pending,
        "pending_transaction_path": assisted.responsibility.stable_path(
            pending_path(args, attempt)
        ),
        "pending_transaction_semantic_sha256": pending["semantic_sha256"],
        "gradient_controls": gradient_controls,
        "directional_derivative": pending["directional_derivative"],
        "derivative_probe": derivative,
        "current_fixed": compact_summary(current_fixed),
        "current_full": compact_summary(current_full),
        "trials": trials,
        "selected": None if selected is None else compact_trial(selected),
        "numerical_failure": bool(numerical_reasons),
        "numerical_failure_reasons": numerical_reasons,
        "unaccepted_restoration_pass": restored,
    }


def cohort_preservation(
    references: dict[str, Any],
    candidate_blocks: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_blocks = source_reference_map(references)
    return [
        {
            "seed": manifest["seed"],
            **block_preservation_decision(source_blocks[manifest["seed"]], candidate),
        }
        for manifest, candidate in zip(manifests, candidate_blocks, strict=True)
    ]


def evaluate_cohort(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    references: dict[str, Any],
    state: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    combined, blocks = full_cohort_report(args, manifests, state, device=device)
    source = references["combined"]
    preservation = cohort_preservation(references, blocks, manifests)
    combined["source_absolute_nrmse_improvement"] = source["nrmse"] - combined["nrmse"]
    return {
        "combined": combined,
        "blocks": blocks,
        "per_block_preservation": preservation,
        "all_preservation_pass": all(item["pass"] for item in preservation),
    }


def milestone_decision(training: dict[str, Any], development: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    train = training["combined"]
    dev = development["combined"]
    if train["source_absolute_nrmse_improvement"] < MILESTONE_TRAINING_IMPROVEMENT:
        reasons.append("training NRMSE improvement was below 0.02")
    if dev["source_absolute_nrmse_improvement"] < MILESTONE_DEVELOPMENT_IMPROVEMENT:
        reasons.append("development NRMSE improvement was below 0.01")
    source_dev = development["source_combined"]
    for horizon in motion.HISTORY_LENGTHS:
        key = str(horizon)
        if (
            dev["by_horizon"][key]["nrmse"]
            > source_dev["by_horizon"][key]["nrmse"] + MILESTONE_HORIZON_REGRESSION_LIMIT
        ):
            reasons.append(f"development horizon {horizon} NRMSE regressed by more than 0.01")
    if not training["all_preservation_pass"]:
        reasons.append("training source-relative preservation failed")
    if not development["all_preservation_pass"]:
        reasons.append("development source-relative preservation failed")
    return {"pass": not reasons, "reasons": reasons}


def terminal_behavior_decision(evaluation: dict[str, Any], *, cohort: str) -> dict[str, Any]:
    summary = evaluation["combined"]
    reasons = []

    def check(values: dict[str, Any], label: str) -> None:
        if values["nrmse"] > FINAL_NRMSE_MAXIMUM:
            reasons.append(f"{cohort} {label} NRMSE exceeded 0.5")
        if values["correct_sign_fraction"] < FINAL_SIGN_FRACTION:
            reasons.append(f"{cohort} {label} correct-sign fraction was below 90%")
        if not FINAL_GAIN_RANGE[0] <= values["teacher_aligned_gain"] <= FINAL_GAIN_RANGE[1]:
            reasons.append(f"{cohort} {label} gain was outside [0.5,1.5]")

    check(summary, "overall")
    for horizon in motion.HISTORY_LENGTHS:
        check(summary["by_horizon"][str(horizon)], f"horizon {horizon}")
    if not evaluation["all_preservation_pass"]:
        reasons.append(f"{cohort} source-relative preservation failed")
    if not summary["all_recurrent_states_and_outputs_finite"] or not summary["all_metrics_finite"]:
        reasons.append(f"{cohort} recurrence, outputs or metrics were nonfinite")
    if summary["endpoint_image_difference_max"] != 0.0:
        reasons.append(f"{cohort} paired endpoint images differed")
    return {"pass": not reasons, "reasons": reasons}


def cache_manifests_path(args: argparse.Namespace, cohort: str) -> Path:
    return args.output_dir / f"{cohort}-cache-manifests.json"


def ensure_cache_manifests(
    args: argparse.Namespace,
    cohort: str,
    *,
    device: torch.device,
    config: HoverConfig,
) -> list[dict[str, Any]]:
    path = cache_manifests_path(args, cohort)
    specs = block_specs(cohort)
    if path.is_file():
        with path.open() as stream:
            manifests = json.load(stream)
        if [(item["seed"], item["held_out_styles"]) for item in manifests] != [
            (item["seed"], item["held_out_styles"]) for item in specs
        ]:
            raise SystemExit(f"{cohort} cache-manifest set mismatch")
        for spec, manifest in zip(specs, manifests, strict=True):
            cache = load_cache(cache_path(args, spec), spec)
            if assisted.responsibility.file_sha256(cache_path(args, spec)) != manifest[
                "file_sha256"
            ]:
                raise SystemExit(f"{cohort} cache physical hash mismatch")
            if cache["semantic_sha256"] != manifest["semantic_sha256"]:
                raise SystemExit(f"{cohort} cache recorded semantic hash mismatch")
        return manifests
    manifests = [
        ensure_cache(args, spec, device=device, config=config) for spec in specs
    ]
    assisted._atomic_json_save(manifests, path)
    return manifests


def references_path(args: argparse.Namespace, cohort: str) -> Path:
    return args.output_dir / f"{cohort}-source-references.json"


def ensure_source_references(
    args: argparse.Namespace,
    cohort: str,
    manifests: list[dict[str, Any]],
    source_state: dict[str, Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    path = references_path(args, cohort)
    source_sha256 = assisted.audit.semantic_sha256(source_state)
    manifest_hashes = [item["file_sha256"] for item in manifests]
    if path.is_file():
        with path.open() as stream:
            payload = json.load(stream)
        if payload.get("semantic_sha256") != _payload_semantic_hash(payload):
            raise SystemExit(f"{cohort} source-reference semantic hash mismatch")
        if payload.get("source_state_sha256") != source_sha256:
            raise SystemExit(f"{cohort} source-reference controller mismatch")
        if payload.get("cache_file_sha256") != manifest_hashes:
            raise SystemExit(f"{cohort} source-reference cache mismatch")
        return payload
    print(json.dumps({"stage": "source_references", "cohort": cohort}), flush=True)
    references = source_references(args, manifests, source_state, device=device)
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "cohort": cohort,
        "source_state_sha256": source_sha256,
        "cache_file_sha256": manifest_hashes,
        **references,
    }
    payload["semantic_sha256"] = _payload_semantic_hash(payload)
    assisted._atomic_json_save(payload, path)
    return payload


def resume_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "resume.pt"


def save_resume(
    args: argparse.Namespace,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "controller": _clone_state(controller.state_dict()),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        **state,
    }
    payload["semantic_sha256"] = _payload_semantic_hash(payload)
    assisted._atomic_torch_save(payload, resume_path(args))


def load_resume(
    args: argparse.Namespace,
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    *,
    source_sha256: str,
) -> dict[str, Any] | None:
    path = resume_path(args)
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("semantic_sha256") != _payload_semantic_hash(payload):
        raise SystemExit("capacity-fit resume semantic hash mismatch")
    if payload.get("experiment") != EXPERIMENT or payload.get("protocol_commit") != PROTOCOL_COMMIT:
        raise SystemExit("capacity-fit resume protocol mismatch")
    if payload.get("source_state_sha256") != source_sha256:
        raise SystemExit("capacity-fit resume source mismatch")
    controller.load_state_dict(payload["controller"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return {
        "accepted_updates": int(payload["accepted_updates"]),
        "history": payload["history"],
        "development_passed": bool(payload["development_passed"]),
        "development_evaluation": payload.get("development_evaluation"),
        "source_state_sha256": payload["source_state_sha256"],
    }


def development_marker_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "development-started.json"


def qualification_marker_path(args: argparse.Namespace) -> Path:
    return args.output_dir / "qualification-started.json"


def write_phase_marker(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        with path.open() as stream:
            if json.load(stream) != payload:
                raise SystemExit(f"phase marker mismatch: {path}")
        return
    assisted._atomic_json_save(payload, path)


def add_source_to_evaluation(
    evaluation: dict[str, Any], references: dict[str, Any]
) -> dict[str, Any]:
    evaluation["source_combined"] = references["combined"]
    evaluation["combined"]["source_absolute_nrmse_improvement"] = (
        references["combined"]["nrmse"] - evaluation["combined"]["nrmse"]
    )
    return evaluation


def solver_qualification(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    state: dict[str, Tensor],
    rk4_blocks: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    rk4_contrasts, reference_contrasts = [], []
    rk4_outputs, reference_outputs = [], []
    reference_blocks = []
    identities_pass = True
    for manifest, rk4 in zip(manifests, rk4_blocks, strict=True):
        cache = _block_cache(args, manifest)
        reference = full_block_report(
            args,
            state,
            cache,
            method="exponential_euler",
            solver_steps=32,
            device=device,
        )
        reference_blocks.append(reference)
        rk4_contrasts.append(torch.tensor(rk4["prediction_contrasts"], dtype=torch.float64))
        reference_contrasts.append(
            torch.tensor(reference["prediction_contrasts"], dtype=torch.float64)
        )
        rk4_outputs.append(torch.tensor(rk4["terminal_motor_outputs"], dtype=torch.float64))
        reference_outputs.append(
            torch.tensor(reference["terminal_motor_outputs"], dtype=torch.float64)
        )
        identities_pass = identities_pass and all(
            item["source_loaded_exactly"]
            and item["source_restored"]
            and item["all_recurrent_states_and_outputs_finite"]
            and item["all_metrics_finite"]
            and item["endpoint_image_difference_max"] == 0.0
            for item in (rk4, reference)
        )
    contrast_rms = float(
        (torch.cat(rk4_contrasts) - torch.cat(reference_contrasts)).square().mean().sqrt()
    )
    motor_rms = float(
        (torch.cat(rk4_outputs) - torch.cat(reference_outputs)).square().mean().sqrt()
    )
    normalized = contrast_rms / rate.EXPECTED_MOTION_SCALE
    return {
        "pass": bool(
            identities_pass
            and normalized <= rate.CONTRAST_REFINEMENT_LIMIT
            and motor_rms <= rate.MOTOR_REFINEMENT_LIMIT
        ),
        "identity_finiteness_and_endpoint_pass": identities_pass,
        "contrast_rms_difference_motor_units": contrast_rms,
        "contrast_rms_difference_over_teacher_scale": normalized,
        "terminal_motor_rms_difference": motor_rms,
        "contrast_limit": rate.CONTRAST_REFINEMENT_LIMIT,
        "motor_limit": rate.MOTOR_REFINEMENT_LIMIT,
        "reference_blocks": reference_blocks,
    }


def terminal_report(
    args: argparse.Namespace,
    *,
    classification: str,
    passed: bool,
    input_hashes: dict[str, str],
    source_sha256: str,
    mask: dict[str, Any],
    training_manifests: list[dict[str, Any]],
    training_references: dict[str, Any],
    state: dict[str, Any],
    terminal_training: dict[str, Any] | None = None,
    terminal_training_decision: dict[str, Any] | None = None,
    development: dict[str, Any] | None = None,
    development_decision: dict[str, Any] | None = None,
    qualification: dict[str, Any] | None = None,
    qualification_decision: dict[str, Any] | None = None,
    solver_check: dict[str, Any] | None = None,
    stop_attempt: dict[str, Any] | None = None,
    qualified_controller: dict[str, Any] | None = None,
    wall_time_seconds: float,
) -> dict[str, Any]:
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha256,
        "readout_mask": mask["manifest"],
        "training_cache_manifests": training_manifests,
        "training_source_references_sha256": training_references["semantic_sha256"],
        "accepted_updates": state["accepted_updates"],
        "attempted_updates": len(state["history"]) + (1 if stop_attempt is not None else 0),
        "history": state["history"],
        "stop_attempt": stop_attempt,
        "terminal_training": terminal_training,
        "terminal_training_decision": terminal_training_decision,
        "development": development,
        "development_decision": development_decision,
        "qualification": qualification,
        "qualification_decision": qualification_decision,
        "solver_qualification": solver_check,
        "qualified_controller": qualified_controller,
        "bounded_collective_restoration_preregistration_authorized": passed,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
        "wall_time_seconds": wall_time_seconds,
    }
    assisted._atomic_json_save(report, args.output_dir / "report.json")
    return report


def main() -> int:
    args = parse_args()
    if (args.output_dir / "report.json").is_file():
        raise SystemExit("the bounded RK4 readout-capacity fit already has a terminal report")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    input_hashes = validate_inputs(args)
    source_state = _load_source_checkpoint(args.checkpoint)
    source_sha256 = assisted.audit.semantic_sha256(source_state)
    mask = readout.build_readout_mask(args.graph)
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_sha256,
        "readout_mask": mask["manifest"],
        "device": str(device),
    }
    _write_or_validate_start(args.output_dir / "start.json", start)
    started = perf_counter()
    config = HoverConfig()

    training_manifests = ensure_cache_manifests(
        args, "training", device=device, config=config
    )
    training_references = ensure_source_references(
        args, "training", training_manifests, source_state, device=device
    )
    controller = ConnectomeController(args.graph, neural_dt=1.0 / rate.POLICY_HZ).to(device)
    controller.load_state_dict(source_state, strict=True)
    optimizer = readout.make_optimizer(controller)
    state = load_resume(
        args, controller, optimizer, source_sha256=source_sha256
    ) or {
        "accepted_updates": 0,
        "history": [],
        "development_passed": False,
        "development_evaluation": None,
        "source_state_sha256": source_sha256,
    }

    while True:
        accepted_updates = state["accepted_updates"]
        current_state = _clone_state(controller.state_dict())

        if accepted_updates == MILESTONE_UPDATE and not state["development_passed"]:
            print(json.dumps({"stage": "milestone_10"}), flush=True)
            training = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    training_manifests,
                    training_references,
                    current_state,
                    device=device,
                ),
                training_references,
            )
            if (
                training["combined"]["source_absolute_nrmse_improvement"]
                < MILESTONE_TRAINING_IMPROVEMENT
            ):
                decision = {
                    "pass": False,
                    "reasons": ["training NRMSE improvement was below 0.02"],
                }
                terminal_report(
                    args,
                    classification="readout_capacity_update_10_training_gate_failed",
                    passed=False,
                    input_hashes=input_hashes,
                    source_sha256=source_sha256,
                    mask=mask,
                    training_manifests=training_manifests,
                    training_references=training_references,
                    state=state,
                    terminal_training=training,
                    terminal_training_decision=decision,
                    wall_time_seconds=perf_counter() - started,
                )
                break
            write_phase_marker(
                development_marker_path(args),
                {
                    "experiment": EXPERIMENT,
                    "accepted_updates": MILESTONE_UPDATE,
                    "controller_sha256": assisted.audit.semantic_sha256(current_state),
                    "seeds": list(DEVELOPMENT_SEEDS),
                },
            )
            development_manifests = ensure_cache_manifests(
                args, "development", device=device, config=config
            )
            development_references = ensure_source_references(
                args, "development", development_manifests, source_state, device=device
            )
            development = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    development_manifests,
                    development_references,
                    current_state,
                    device=device,
                ),
                development_references,
            )
            milestone = milestone_decision(training, development)
            state["development_evaluation"] = {
                "accepted_updates": MILESTONE_UPDATE,
                "training": training,
                "development": development,
                "decision": milestone,
            }
            if not milestone["pass"]:
                save_resume(args, controller, optimizer, state)
                terminal_report(
                    args,
                    classification="readout_capacity_update_10_development_gate_failed",
                    passed=False,
                    input_hashes=input_hashes,
                    source_sha256=source_sha256,
                    mask=mask,
                    training_manifests=training_manifests,
                    training_references=training_references,
                    state=state,
                    terminal_training=training,
                    terminal_training_decision=milestone,
                    development=development,
                    development_decision=milestone,
                    wall_time_seconds=perf_counter() - started,
                )
                break
            state["development_passed"] = True
            save_resume(args, controller, optimizer, state)
            continue

        if accepted_updates >= MAXIMUM_ACCEPTED_UPDATES:
            print(json.dumps({"stage": "terminal_training"}), flush=True)
            terminal_training = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    training_manifests,
                    training_references,
                    current_state,
                    device=device,
                ),
                training_references,
            )
            terminal_training_decision = terminal_behavior_decision(
                terminal_training, cohort="training"
            )
            if not terminal_training_decision["pass"]:
                terminal_report(
                    args,
                    classification="readout_capacity_terminal_training_gate_failed",
                    passed=False,
                    input_hashes=input_hashes,
                    source_sha256=source_sha256,
                    mask=mask,
                    training_manifests=training_manifests,
                    training_references=training_references,
                    state=state,
                    terminal_training=terminal_training,
                    terminal_training_decision=terminal_training_decision,
                    wall_time_seconds=perf_counter() - started,
                )
                break
            development_manifests = ensure_cache_manifests(
                args, "development", device=device, config=config
            )
            development_references = ensure_source_references(
                args, "development", development_manifests, source_state, device=device
            )
            development = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    development_manifests,
                    development_references,
                    current_state,
                    device=device,
                ),
                development_references,
            )
            development_decision = terminal_behavior_decision(
                development, cohort="development"
            )
            if not development_decision["pass"]:
                terminal_report(
                    args,
                    classification="readout_capacity_terminal_development_gate_failed",
                    passed=False,
                    input_hashes=input_hashes,
                    source_sha256=source_sha256,
                    mask=mask,
                    training_manifests=training_manifests,
                    training_references=training_references,
                    state=state,
                    terminal_training=terminal_training,
                    terminal_training_decision=terminal_training_decision,
                    development=development,
                    development_decision=development_decision,
                    wall_time_seconds=perf_counter() - started,
                )
                break
            write_phase_marker(
                qualification_marker_path(args),
                {
                    "experiment": EXPERIMENT,
                    "accepted_updates": MAXIMUM_ACCEPTED_UPDATES,
                    "controller_sha256": assisted.audit.semantic_sha256(current_state),
                    "seeds": list(QUALIFICATION_SEEDS),
                },
            )
            qualification_manifests = ensure_cache_manifests(
                args, "qualification", device=device, config=config
            )
            qualification_references = ensure_source_references(
                args, "qualification", qualification_manifests, source_state, device=device
            )
            qualification = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    qualification_manifests,
                    qualification_references,
                    current_state,
                    device=device,
                ),
                qualification_references,
            )
            qualification_decision = terminal_behavior_decision(
                qualification, cohort="qualification"
            )
            solver_check = None
            if qualification_decision["pass"]:
                print(json.dumps({"stage": "qualification_solver_check"}), flush=True)
                solver_check = solver_qualification(
                    args,
                    qualification_manifests,
                    current_state,
                    qualification["blocks"],
                    device=device,
                )
            passed = bool(qualification_decision["pass"] and solver_check and solver_check["pass"])
            if passed:
                checkpoint_path = args.output_dir / "qualified-controller.pt"
                assisted._atomic_torch_save(
                    {
                        "experiment": EXPERIMENT,
                        "controller": current_state,
                        "source_state_sha256": source_sha256,
                        "accepted_updates": accepted_updates,
                    },
                    checkpoint_path,
                )
                checkpoint_record = {
                    "path": assisted.responsibility.stable_path(checkpoint_path),
                    "file_sha256": assisted.responsibility.file_sha256(checkpoint_path),
                    "controller_semantic_sha256": assisted.audit.semantic_sha256(current_state),
                }
                classification = "bounded_rk4_readout_capacity_qualified"
            else:
                checkpoint_record = None
                classification = (
                    "readout_capacity_qualification_gate_failed"
                    if not qualification_decision["pass"]
                    else "readout_capacity_fitted_solver_qualification_failed"
                )
            terminal_report(
                args,
                classification=classification,
                passed=passed,
                input_hashes=input_hashes,
                source_sha256=source_sha256,
                mask=mask,
                training_manifests=training_manifests,
                training_references=training_references,
                state=state,
                terminal_training=terminal_training,
                terminal_training_decision=terminal_training_decision,
                development=development,
                development_decision=development_decision,
                qualification=qualification,
                qualification_decision=qualification_decision,
                solver_check=solver_check,
                qualified_controller=checkpoint_record,
                wall_time_seconds=perf_counter() - started,
            )
            break

        attempt = accepted_updates + 1
        attempt_record = run_attempt(
            controller,
            optimizer,
            args,
            training_manifests,
            training_references,
            source_state,
            mask,
            attempt=attempt,
            device=device,
        )
        if not attempt_record["accepted"]:
            current_state = _clone_state(controller.state_dict())
            terminal_training = add_source_to_evaluation(
                evaluate_cohort(
                    args,
                    training_manifests,
                    training_references,
                    current_state,
                    device=device,
                ),
                training_references,
            )
            if attempt_record["numerical_failure"]:
                if attempt_record["derivative_probe"].get("classification") == (
                    "derivative_probe_noise_limited_or_nonconvergent"
                ):
                    classification = "readout_capacity_derivative_probe_inconclusive"
                else:
                    classification = "readout_capacity_numerical_control_failed"
            else:
                classification = "readout_capacity_constraint_boundary_stop"
            terminal_report(
                args,
                classification=classification,
                passed=False,
                input_hashes=input_hashes,
                source_sha256=source_sha256,
                mask=mask,
                training_manifests=training_manifests,
                training_references=training_references,
                state=state,
                terminal_training=terminal_training,
                stop_attempt=attempt_record,
                wall_time_seconds=perf_counter() - started,
            )
            break

        state["accepted_updates"] += 1
        state["history"].append(attempt_record)
        save_resume(args, controller, optimizer, state)
        print(
            json.dumps(
                {
                    "stage": "accepted_update",
                    "accepted_updates": state["accepted_updates"],
                    "scale": attempt_record["accepted_scale"],
                    "full_nrmse": attempt_record["selected"]["full"]["nrmse"],
                }
            ),
            flush=True,
        )

    with (args.output_dir / "report.json").open() as stream:
        report = json.load(stream)
    print(
        json.dumps(
            {
                "output": assisted.responsibility.stable_path(args.output_dir / "report.json"),
                "classification": report["classification"],
                "pass": report["passed"],
                "accepted_updates": report["accepted_updates"],
                "hover_or_gate_flight_authorized": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
