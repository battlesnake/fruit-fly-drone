#!/usr/bin/env python3
"""Run bounded FP32 vertical-motion commissioning after the passing tau audit."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_frozen_optic_motion as frozen_audit  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402
import preflight_vertical_motion_commissioning as preflight  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402

EXPERIMENT = "vertical-t4t5-local-commissioning-training-v1"
PROTOCOL_COMMIT = "05f4a2e"
FP64_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-tau-fp64-001/report.json"
EXPECTED_FP64_REPORT_SHA256 = "e460f079f5a3225d2eb582d57f9c742a588c009ce35c7ff23885a0d61cabe9fc"
TRAINING_SPEEDS = (1, 2, 4, 6)
TRAINING_PHASES = tuple(range(6))
DEVELOPMENT_SPEEDS = (1, 4, 6)
DEVELOPMENT_PHASES = tuple(range(4))
ACCEPTANCE_SPEEDS = (1, 2, 3, 4)
ACCEPTANCE_PHASES = tuple(range(8))
MAX_PROPOSALS = 100
MANDATORY_TRAINING_GATE_PROPOSAL = 25
SCHEDULED_EVALUATIONS = (25, 50, 100)
TRAINING_IMPROVEMENT_FRACTION = 0.25
NUMERICAL_CASES = (0, 1, 24, 25, 48, 49, 60, 61)
NUMERICAL_FINE_SUBSTEPS = 64
NUMERICAL_RMS_LIMIT = 0.005
SIGN_FRACTION_MINIMUM = 0.90
MEDIAN_DSI_MINIMUM = 0.30
ACTIVE_FRACTION_MINIMUM = 0.50
STATIONARY_RATIO_MAXIMUM = 0.10
ACTIVE_ABSOLUTE_FLOOR = 1.0e-6
UNTRAINED_ACCEPTANCE_SPEED = 3


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
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--registration",
        type=Path,
        default=REPO_ROOT / "artifacts/vertical-motion-commissioning-manifest-v1/manifest.json",
    )
    parser.add_argument(
        "--frozen-audit-report",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/frozen-t4t5-audit-002/report.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-commissioning-training-001",
    )
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def balanced_batches(
    specs: list[dict[str, Any]], *, split: str
) -> tuple[tuple[int, int, int, int], ...]:
    layout = {
        "training": (TRAINING_SPEEDS, TRAINING_PHASES),
        "development": (DEVELOPMENT_SPEEDS, DEVELOPMENT_PHASES),
        "acceptance": (ACCEPTANCE_SPEEDS, ACCEPTANCE_PHASES),
    }
    if split not in layout:
        raise ValueError(f"unsupported vertical-motion split: {split}")
    speeds, phases = layout[split]

    def one(**requirements: Any) -> int:
        selected = [
            index
            for index, spec in enumerate(specs)
            if all(spec.get(name) == value for name, value in requirements.items())
        ]
        if len(selected) != 1:
            raise ValueError(f"split {split} does not have one case for {requirements}")
        return selected[0]

    batches = []
    for speed in speeds:
        for phase in phases:
            on = one(
                family="polarity_preserving_edge",
                polarity="ON",
                speed_pixels_per_frame=speed,
                phase_index=phase,
            )
            off = one(
                family="polarity_preserving_edge",
                polarity="OFF",
                speed_pixels_per_frame=speed,
                phase_index=phase,
            )
            textures = [
                index
                for index, spec in enumerate(specs)
                if spec["family"] == "band_limited_texture"
                and spec["speed_pixels_per_frame"] == speed
                and spec["family_id"].endswith((f"-{2 * phase}", f"-{2 * phase + 1}"))
            ]
            if len(textures) != 2:
                raise ValueError(f"split {split} texture pair is not unique")
            batches.append((on, off, *sorted(textures)))
    flattened = [case for batch in batches for case in batch]
    if sorted(flattened) != list(range(len(specs))) or len(flattened) != len(set(flattened)):
        raise ValueError(f"split {split} balanced batches do not partition every case")
    return tuple(batches)


def protocol_manifest() -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "locked_passing_fp64_report_sha256": EXPECTED_FP64_REPORT_SHA256,
        "source_protocol": preflight.protocol_manifest(),
        "training": {
            "cases": 96,
            "balanced_batches": 24,
            "batch_order": "speed [1,2,4,6] major then edge phase [0..5] minor",
            "batch_contents": "matched ON, matched OFF, textures 2*phase and 2*phase+1",
            "max_accepted_or_rejected_proposals": MAX_PROPOSALS,
            "mandatory_gate_proposal": MANDATORY_TRAINING_GATE_PROPOSAL,
            "full_loss_improvement_fraction_minimum": TRAINING_IMPROVEMENT_FRACTION,
            "edge_direction_strata": ["ON_down", "ON_up", "OFF_down", "OFF_up"],
            "each_stratum_zero_or_strictly_improved_at_mandatory_gate": True,
            "source_responses_cached_once": True,
            "normal_and_literal_reverse_pixels_and_source_responses_hashed": True,
        },
        "optimizer": {
            **preflight.protocol_manifest()["optimizer"],
            "state_persists_across_accepted_updates": True,
            "rejected_proposal_restores_parameters_and_optimizer": True,
            "proposal_counter_advances_after_rejection": True,
            "proposal_in_flight_marker_written_before_computation": True,
            "interrupted_proposal_is_terminal_without_retry": True,
            "atomic_resume_state_after_every_proposal": True,
        },
        "scheduled_evaluations": {
            "proposals": list(SCHEDULED_EVALUATIONS),
            "numerical_cases": list(NUMERICAL_CASES),
            "K32_substeps": registration.CNS_SUBSTEPS_PER_FRAME,
            "K64_substeps": NUMERICAL_FINE_SUBSTEPS,
            "activity_opponent_and_motor_rms_maximum": NUMERICAL_RMS_LIMIT,
            "failed_numerical_gate_is_terminal": True,
            "candidate_persisted_and_development_started_marked_before_pixels": True,
            "interrupted_development_is_terminal": True,
        },
        "development": {
            "cases": 48,
            "rendered_only_after_scheduled_numerical_pass": True,
            "source_cache_reused_after_first_open": True,
            "vertical_sign_fraction_minimum": SIGN_FRACTION_MINIMUM,
            "vertical_sign_windows": list(commissioning.WINDOWS),
            "median_dsi_minimum": MEDIAN_DSI_MINIMUM,
            "dsi_window": "integrated",
            "active_fraction_minimum": ACTIVE_FRACTION_MINIMUM,
            "stationary_to_moving_ratio_maximum": STATIONARY_RATIO_MAXIMUM,
            "stationary_ratio_window": "integrated",
            "reverse_sign_inversion_fraction_minimum": SIGN_FRACTION_MINIMUM,
            "reverse_sign_window": "integrated",
            "absolute_activity_floor": ACTIVE_ABSOLUTE_FLOOR,
            "selection": "smallest development loss among pass, exact tie earlier proposal",
            "failure_reporting": (
                "smallest maximum normalized gate shortfall, then loss, then proposal"
            ),
        },
        "acceptance": {
            "cases": 128,
            "opened_once_only_after_selected_development_pass": True,
            "reapply_development_gates": True,
            "all_64_texture_T4_T5_up_down_integrated_terminal_fraction_minimum": (
                SIGN_FRACTION_MINIMUM
            ),
            "novel_texture_speed": UNTRAINED_ACCEPTANCE_SPEED,
            "novel_texture_T4_T5_up_down_integrated_terminal_fraction_minimum": (
                SIGN_FRACTION_MINIMUM
            ),
            "pair_mean_intervention_diagnostic_only": True,
            "throttle_change_or_tonic_preservation_is_not_a_gate": True,
        },
        "retention": {
            "only_acceptance_pass_retains_24_parameter_module": True,
            "full_source_is_never_mutated": True,
            "authorizes_only_motion_to_DN_VNC_routing_preregistration": True,
            "hover_gate_or_promotion_authorized": False,
        },
        "peak_cuda_reserved_bytes_maximum": preflight.PEAK_RESERVED_LIMIT_BYTES,
        "process_lifetime_exclusive_output_lock": True,
    }


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    observed = preflight.validate_inputs(args)
    if (
        not FP64_REPORT.is_file()
        or registration.file_sha256(FP64_REPORT) != EXPECTED_FP64_REPORT_SHA256
    ):
        raise SystemExit("locked passing FP64 tau audit changed")
    with FP64_REPORT.open() as stream:
        report = json.load(stream)
    if not (
        report.get("passed")
        and report.get("classification") == "vertical_motion_tau_fp64_audit_passed"
        and report.get("training_execution_authorized")
        and report.get("candidate_retained") is False
        and report.get("hover_or_gate_flight_authorized") is False
    ):
        raise SystemExit("FP64 tau audit does not authorize bounded training")
    observed[registration.stable_path(FP64_REPORT)] = EXPECTED_FP64_REPORT_SHA256
    return observed


def render_bank(specs: list[dict[str, Any]]) -> dict[tuple[int, bool], Tensor]:
    return {
        (case, reverse): commissioning.render_sequence(spec, reverse=reverse)
        for case, spec in enumerate(specs)
        for reverse in (False, True)
    }


def evaluate_cases(
    controller: torch.nn.Module,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...] | list[int],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    prefix = commissioning.neutral_prefix(
        controller, device=device, checkpoint_frames=checkpoint_frames
    )
    normal = []
    reverse = []
    for case in cases:
        normal.append(
            commissioning.evaluate_pair(
                controller,
                sequences[(case, False)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
        reverse.append(
            commissioning.evaluate_pair(
                controller,
                sequences[(case, True)],
                anatomy,
                device=device,
                checkpoint_frames=checkpoint_frames,
                prefix_state=prefix,
            )
        )
    return normal, reverse


def response_bank_cpu(
    normal: list[dict[str, Tensor]],
    reverse: list[dict[str, Tensor]],
    cases: tuple[int, ...] | list[int],
) -> dict[str, dict[int, dict[str, Tensor]]]:
    return {
        "normal": {
            case: commissioning.response_tree_cpu(response)
            for case, response in zip(cases, normal, strict=True)
        },
        "reverse": {
            case: commissioning.response_tree_cpu(response)
            for case, response in zip(cases, reverse, strict=True)
        },
    }


def batch_references(
    specs: list[dict[str, Any]],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    cases: tuple[int, int, int, int],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    return commissioning.source_references(
        [specs[case] for case in cases],
        [source_bank["normal"][case] for case in cases],
        [source_bank["reverse"][case] for case in cases],
        anatomy,
    )


def batch_loss(
    controller: commissioning.CommissionedController,
    specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    cases: tuple[int, int, int, int],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    normal, reverse = evaluate_cases(
        controller,
        sequences,
        anatomy,
        cases,
        device=device,
        checkpoint_frames=checkpoint_frames,
    )
    references = batch_references(specs, source_bank, cases, anatomy)
    return commissioning.commissioning_loss(
        [specs[case] for case in cases],
        normal,
        reverse,
        references,
        anatomy,
        controller,
    )


def identity_parameter_namespace(values: dict[str, Tensor] | None = None) -> SimpleNamespace:
    if values is None:
        values = {
            "gain": torch.ones(16),
            "bias_offset": torch.zeros(4),
            "tau_ratio": torch.ones(4),
        }
    return SimpleNamespace(**{name: value.detach().cpu() for name, value in values.items()})


def bank_loss(
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    candidate_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    parameter_values: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    controller = identity_parameter_namespace(parameter_values)
    rows = []
    for cases in batches:
        references = batch_references(specs, source_bank, cases, anatomy)
        loss, components = commissioning.commissioning_loss(
            [specs[case] for case in cases],
            [candidate_bank["normal"][case] for case in cases],
            [candidate_bank["reverse"][case] for case in cases],
            references,
            anatomy,
            controller,
        )
        rows.append(
            {
                "cases": list(cases),
                "loss": float(loss),
                "components": {name: float(value) for name, value in components.items()},
            }
        )
    component_names = rows[0]["components"]
    return {
        "loss": float(np.mean([row["loss"] for row in rows])),
        "components": {
            name: float(np.mean([row["components"][name] for row in rows]))
            for name in component_names
        },
        "batches": rows,
    }


def complete_bank_loss(
    specs: list[dict[str, Any]],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    candidate_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    parameter_values: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    cases = tuple(range(len(specs)))
    references = commissioning.source_references(
        specs,
        [source_bank["normal"][case] for case in cases],
        [source_bank["reverse"][case] for case in cases],
        anatomy,
    )
    loss, components = commissioning.commissioning_loss(
        specs,
        [candidate_bank["normal"][case] for case in cases],
        [candidate_bank["reverse"][case] for case in cases],
        references,
        anatomy,
        identity_parameter_namespace(parameter_values),
    )
    return {
        "loss": float(loss),
        "components": {name: float(value) for name, value in components.items()},
        "source_references_semantic_sha256": references["semantic_sha256"],
    }


def edge_direction_strata(
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    candidate_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, float]:
    relative = commissioning.relative_subtype_indices(anatomy)
    values: dict[str, list[float]] = {
        "ON_down": [],
        "ON_up": [],
        "OFF_down": [],
        "OFF_up": [],
    }
    for cases in batches:
        references = batch_references(specs, source_bank, cases, anatomy)
        for pair_index, case in enumerate(cases[:2]):
            spec = specs[case]
            pathway = "T4" if spec["polarity"] == "ON" else "T5"
            for window in commissioning.WINDOWS:
                response = candidate_bank["normal"][case][f"{window}_response"]
                opponent = commissioning._opponent(response, relative, pathway)
                scale = references["population"][window][pair_index][pathway]["normal_scale"]
                down = torch.relu(commissioning.DIRECTION_MARGIN + opponent[0] / scale).square()
                up = torch.relu(commissioning.DIRECTION_MARGIN - opponent[1] / scale).square()
                values[f"{spec['polarity']}_down"].append(float(down))
                values[f"{spec['polarity']}_up"].append(float(up))
    return {name: float(np.mean(items)) for name, items in values.items()}


def mandatory_training_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_strata: dict[str, float],
    candidate_strata: dict[str, float],
) -> dict[str, Any]:
    loss_limit = (1.0 - TRAINING_IMPROVEMENT_FRACTION) * baseline["loss"]
    strata = {
        name: {
            "baseline": baseline_strata[name],
            "candidate": candidate_strata[name],
            "pass": bool(
                candidate_strata[name] == 0.0 or candidate_strata[name] < baseline_strata[name]
            ),
        }
        for name in baseline_strata
    }
    return {
        "pass": bool(
            candidate["loss"] <= loss_limit and all(item["pass"] for item in strata.values())
        ),
        "baseline_loss": baseline["loss"],
        "candidate_loss": candidate["loss"],
        "loss_limit": loss_limit,
        "improvement_fraction": (baseline["loss"] - candidate["loss"])
        / max(abs(baseline["loss"]), 1.0e-30),
        "strata": strata,
    }


def _opponent(response: dict[str, Tensor], window: str, pathway: str, relative: dict) -> Tensor:
    return commissioning._opponent(response[f"{window}_response"], relative, pathway)


def qualification_metrics(
    specs: list[dict[str, Any]],
    bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative = commissioning.relative_subtype_indices(anatomy)

    def edge_cases(pathway: str) -> list[int]:
        polarity = "ON" if pathway == "T4" else "OFF"
        return [
            case
            for case, spec in enumerate(specs)
            if spec["family"] == "polarity_preserving_edge" and spec["polarity"] == polarity
        ]

    sign: dict[str, dict[str, dict[str, Any]]] = {}
    for window in commissioning.WINDOWS:
        sign[window] = {}
        for pathway in commissioning.PATHWAYS:
            relevant = edge_cases(pathway)
            opponents = torch.stack(
                [_opponent(bank["normal"][case], window, pathway, relative) for case in relevant]
            )
            down_fraction = float((opponents[:, 0] < 0.0).float().mean())
            up_fraction = float((opponents[:, 1] > 0.0).float().mean())
            sign[window][pathway] = {
                "cases": len(relevant),
                "down_correct_fraction": down_fraction,
                "up_correct_fraction": up_fraction,
                "down_pass": down_fraction >= SIGN_FRACTION_MINIMUM,
                "up_pass": up_fraction >= SIGN_FRACTION_MINIMUM,
            }

    subtype_dsi = {}
    for pathway in commissioning.PATHWAYS:
        relevant = edge_cases(pathway)
        for suffix, preferred_branch, null_branch in (("c", 1, 0), ("d", 0, 1)):
            positions = relative[pathway + suffix]
            responses = torch.stack(
                [bank["normal"][case]["integrated_response"][:, positions] for case in relevant]
            )
            preferred = responses[:, preferred_branch].mean(dim=0)
            null = responses[:, null_branch].mean(dim=0)
            active = torch.maximum(preferred.abs(), null.abs()) > ACTIVE_ABSOLUTE_FLOOR
            dsi = (preferred - null) / (preferred.abs() + null.abs() + ACTIVE_ABSOLUTE_FLOOR)
            active_fraction = float(active.float().mean())
            median = float(dsi[active].median()) if bool(active.any()) else None
            subtype_dsi[pathway + suffix] = {
                "cells": len(positions),
                "active_cells": int(active.sum()),
                "active_fraction": active_fraction,
                "median_dsi_active": median,
                "pass": bool(
                    median is not None
                    and math.isfinite(median)
                    and active_fraction >= ACTIVE_FRACTION_MINIMUM
                    and median >= MEDIAN_DSI_MINIMUM
                ),
            }

    moving = []
    stationary = []
    reverse_passes = []
    for case, spec in enumerate(specs):
        if spec["family"] != "polarity_preserving_edge":
            continue
        for pathway in commissioning.pathway_for_spec(spec):
            moving_opponent = _opponent(bank["normal"][case], "integrated", pathway, relative)
            static_opponent = commissioning._opponent(
                bank["normal"][case]["integrated_static"], relative, pathway
            )
            moving.append(moving_opponent[1] - moving_opponent[0])
            stationary.append(static_opponent[1] - static_opponent[0])
            reversed_pathway = commissioning.reverse_pathway(spec, pathway)
            reversed_opponent = _opponent(
                bank["reverse"][case], "integrated", reversed_pathway, relative
            )
            for branch in (0, 1):
                normal_value = float(moving_opponent[branch])
                reverse_value = float(reversed_opponent[branch])
                reverse_passes.append(
                    bool(
                        abs(normal_value) > ACTIVE_ABSOLUTE_FLOOR
                        and abs(reverse_value) > ACTIVE_ABSOLUTE_FLOOR
                        and normal_value * reverse_value < 0.0
                    )
                )
    moving_rms = float(torch.stack(moving).square().mean().sqrt())
    stationary_rms = float(torch.stack(stationary).square().mean().sqrt())
    stationary_ratio = stationary_rms / max(moving_rms, 1.0e-30)
    reverse_fraction = sum(reverse_passes) / len(reverse_passes)
    sign_pass = all(
        values[direction]
        for window in sign.values()
        for values in window.values()
        for direction in ("down_pass", "up_pass")
    )
    passed = bool(
        sign_pass
        and all(item["pass"] for item in subtype_dsi.values())
        and stationary_ratio <= STATIONARY_RATIO_MAXIMUM
        and reverse_fraction >= SIGN_FRACTION_MINIMUM
    )
    shortfalls = []
    for window in sign.values():
        for values in window.values():
            shortfalls.extend(
                max(0.0, (SIGN_FRACTION_MINIMUM - values[name]) / SIGN_FRACTION_MINIMUM)
                for name in ("down_correct_fraction", "up_correct_fraction")
            )
    for values in subtype_dsi.values():
        shortfalls.append(
            max(
                0.0, (ACTIVE_FRACTION_MINIMUM - values["active_fraction"]) / ACTIVE_FRACTION_MINIMUM
            )
        )
        if values["median_dsi_active"] is not None and math.isfinite(values["median_dsi_active"]):
            shortfalls.append(
                max(0.0, (MEDIAN_DSI_MINIMUM - values["median_dsi_active"]) / MEDIAN_DSI_MINIMUM)
            )
        else:
            shortfalls.append(1.0e30)
    shortfalls.append(
        max(0.0, (stationary_ratio - STATIONARY_RATIO_MAXIMUM) / STATIONARY_RATIO_MAXIMUM)
    )
    shortfalls.append(max(0.0, (SIGN_FRACTION_MINIMUM - reverse_fraction) / SIGN_FRACTION_MINIMUM))
    return {
        "pass": passed,
        "vertical_sign": sign,
        "subtype_dsi": subtype_dsi,
        "moving_opponent_rms": moving_rms,
        "stationary_opponent_rms": stationary_rms,
        "stationary_to_moving_ratio": stationary_ratio,
        "reverse_sign_inversion_fraction": reverse_fraction,
        "maximum_normalized_gate_shortfall": max(shortfalls),
    }


def texture_direction_metrics(
    specs: list[dict[str, Any]],
    bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    *,
    speed: int | None = None,
) -> dict[str, Any]:
    relative = commissioning.relative_subtype_indices(anatomy)
    cases = [
        case
        for case, spec in enumerate(specs)
        if spec["family"] == "band_limited_texture"
        and (speed is None or spec["speed_pixels_per_frame"] == speed)
    ]
    if not cases:
        raise ValueError("texture direction gate has no matching cases")
    rows = {}
    for window in commissioning.WINDOWS:
        rows[window] = {}
        for pathway in commissioning.PATHWAYS:
            opponents = torch.stack(
                [_opponent(bank["normal"][case], window, pathway, relative) for case in cases]
            )
            down = float((opponents[:, 0] < 0.0).float().mean())
            up = float((opponents[:, 1] > 0.0).float().mean())
            rows[window][pathway] = {
                "cases": len(cases),
                "down_correct_fraction": down,
                "up_correct_fraction": up,
                "pass": bool(down >= SIGN_FRACTION_MINIMUM and up >= SIGN_FRACTION_MINIMUM),
            }
    return {
        "pass": all(item["pass"] for window in rows.values() for item in window.values()),
        "cases": len(cases),
        "speed_pixels_per_frame": speed,
        "by_window_and_pathway": rows,
    }


def novel_texture_speed_metrics(
    specs: list[dict[str, Any]],
    bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    return texture_direction_metrics(specs, bank, anatomy, speed=UNTRAINED_ACCEPTANCE_SPEED)


def _advance_with_substeps(
    controller: torch.nn.Module,
    image: Tensor,
    state: Tensor,
    substeps: int,
) -> tuple[Tensor, Tensor]:
    attitude = torch.zeros(image.shape[0], 2, device=image.device, dtype=image.dtype)
    output = torch.zeros(image.shape[0], 4, device=image.device, dtype=image.dtype)
    for _ in range(substeps):
        output, state = controller(image, attitude, state)
    return output, state


@torch.inference_mode()
def evaluate_cases_at_substeps(
    controller: torch.nn.Module,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...],
    substeps: int,
    *,
    device: torch.device,
) -> dict[str, dict[int, dict[str, Tensor]]]:
    neutral = torch.full((1, 3, registration.HEIGHT, registration.WIDTH), 0.5, device=device)
    prefix = controller.initial_state(1, device=device, dtype=neutral.dtype)
    for _ in range(registration.PREFIX_FRAMES):
        _, prefix = _advance_with_substeps(controller, neutral, prefix, substeps)
    selected = torch.from_numpy(anatomy["target_indices"]).to(device)
    result: dict[str, dict[int, dict[str, Tensor]]] = {"normal": {}, "reverse": {}}
    for mode, reverse in (("normal", False), ("reverse", True)):
        for case in cases:
            state = prefix.expand(4, -1).clone()
            integrated_response = torch.zeros(2, len(selected), device=device)
            integrated_static = torch.zeros_like(integrated_response)
            terminal_response = torch.empty_like(integrated_response)
            terminal_static = torch.empty_like(integrated_response)
            for frame in range(registration.MOTION_FRAMES + registration.TERMINAL_FRAMES):
                gray = sequences[(case, reverse)][frame].to(device)
                image = gray[:, None].expand(-1, 3, -1, -1)
                output, state = _advance_with_substeps(controller, image, state, substeps)
                activity = torch.tanh(state[:, selected])
                response = activity[:2] - activity[2:]
                if frame < registration.MOTION_FRAMES:
                    integrated_response = integrated_response + response
                    integrated_static = integrated_static + activity[2:]
                else:
                    terminal_response = response
                    terminal_static = activity[2:]
            result[mode][case] = commissioning.response_tree_cpu(
                {
                    "integrated_response": (integrated_response / registration.MOTION_FRAMES),
                    "integrated_static": integrated_static / registration.MOTION_FRAMES,
                    "terminal_response": terminal_response,
                    "terminal_static": terminal_static,
                    "terminal_motor": output.reshape(4, 4),
                }
            )
    return result


def numerical_comparison(
    specs: list[dict[str, Any]],
    k32: dict[str, dict[int, dict[str, Tensor]]],
    k64: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
) -> dict[str, Any]:
    relative = commissioning.relative_subtype_indices(anatomy)
    activity_differences = []
    opponent_differences = []
    motor_differences = []
    for mode in ("normal", "reverse"):
        for case in NUMERICAL_CASES:
            first = k32[mode][case]
            second = k64[mode][case]
            first_activity = first["terminal_response"] + first["terminal_static"]
            second_activity = second["terminal_response"] + second["terminal_static"]
            activity_differences.append(first_activity - second_activity)
            motor_differences.append(first["terminal_motor"] - second["terminal_motor"])
            spec = specs[case]
            for pathway in commissioning.pathway_for_spec(spec):
                if mode == "reverse":
                    pathway = commissioning.reverse_pathway(spec, pathway)
                first_opponent = commissioning._opponent(
                    first["terminal_response"], relative, pathway
                )
                second_opponent = commissioning._opponent(
                    second["terminal_response"], relative, pathway
                )
                opponent_differences.append(first_opponent - second_opponent)
    activity_rms = float(
        torch.cat([item.reshape(-1) for item in activity_differences]).square().mean().sqrt()
    )
    opponent_rms = float(
        torch.cat([item.reshape(-1) for item in opponent_differences]).square().mean().sqrt()
    )
    motor_rms = float(
        torch.cat([item.reshape(-1) for item in motor_differences]).square().mean().sqrt()
    )
    return {
        "pass": bool(
            activity_rms <= NUMERICAL_RMS_LIMIT
            and opponent_rms <= NUMERICAL_RMS_LIMIT
            and motor_rms <= NUMERICAL_RMS_LIMIT
        ),
        "selected_activity_rms_difference": activity_rms,
        "opponent_rms_difference": opponent_rms,
        "terminal_motor_rms_difference": motor_rms,
        "limits": {
            "selected_activity_rms": NUMERICAL_RMS_LIMIT,
            "opponent_rms": NUMERICAL_RMS_LIMIT,
            "terminal_motor_rms": NUMERICAL_RMS_LIMIT,
        },
    }


def scheduled_numerical_gate(
    args: argparse.Namespace,
    controller: commissioning.CommissionedController,
    parameter_values: dict[str, Tensor],
    source_state: dict[str, Tensor],
    specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, Any]:
    k32 = evaluate_cases_at_substeps(
        controller,
        sequences,
        anatomy,
        NUMERICAL_CASES,
        registration.CNS_SUBSTEPS_PER_FRAME,
        device=device,
    )
    source64 = commissioning.make_source_controller(args.graph, source_state, device=device)
    source64.neural_dt = 1.0 / (registration.CAMERA_HZ * NUMERICAL_FINE_SUBSTEPS)
    candidate64 = commissioning.CommissionedController(source64, anatomy).to(device)
    candidate64.load_parameter_values(parameter_values)
    k64 = evaluate_cases_at_substeps(
        candidate64,
        sequences,
        anatomy,
        NUMERICAL_CASES,
        NUMERICAL_FINE_SUBSTEPS,
        device=device,
    )
    comparison = numerical_comparison(specs, k32, k64, anatomy)
    comparison["K32_semantic_sha256"] = commissioning.semantic_sha256(k32)
    comparison["K64_semantic_sha256"] = commissioning.semantic_sha256(k64)
    candidate64.load_parameter_values(
        {
            "gain": torch.ones(16),
            "bias_offset": torch.zeros(4),
            "tau_ratio": torch.ones(4),
        }
    )
    return comparison


@torch.inference_mode()
def _intervention_variant(
    controller: commissioning.CommissionedController,
    sequences: dict[tuple[int, bool], Tensor],
    cases: tuple[int, ...],
    selected: np.ndarray,
    target_indices: dict[str, np.ndarray],
    *,
    intervene: bool,
    device: torch.device,
    block_cases: int = 8,
) -> dict[str, Any]:
    prefix = commissioning.neutral_prefix(controller, device=device, checkpoint_frames=False)
    selected_tensor = torch.from_numpy(selected).to(device)
    targets = {
        name: torch.from_numpy(indices).to(device) for name, indices in target_indices.items()
    }
    motor = []
    downstream: dict[str, list[Tensor]] = {name: [] for name in targets}
    finite = True
    pair_mean_maximum = 0.0
    for begin in range(0, len(cases), block_cases):
        block = cases[begin : begin + block_cases]
        sequence = torch.stack([sequences[(case, False)][:, :2] for case in block])
        sequence = sequence.permute(1, 0, 2, 3, 4).reshape(
            registration.MOTION_FRAMES + registration.TERMINAL_FRAMES,
            2 * len(block),
            registration.HEIGHT,
            registration.WIDTH,
        )
        state = prefix.expand(2 * len(block), -1).clone()
        output = torch.zeros(2 * len(block), 4, device=device)
        attitude = torch.zeros(2 * len(block), 2, device=device)
        for frame in range(registration.MOTION_FRAMES + registration.TERMINAL_FRAMES):
            gray = sequence[frame].to(device)
            image = gray[:, None].expand(-1, 3, -1, -1)
            for _ in range(registration.CNS_SUBSTEPS_PER_FRAME):
                output, state = controller(image, attitude, state)
                if intervene:
                    shaped = state.reshape(len(block), 2, -1)
                    pair_activity = torch.tanh(shaped[:, :, selected_tensor])
                    pair_mean = pair_activity.mean(dim=1)
                    pair_mean_maximum = max(pair_mean_maximum, float(pair_mean.abs().max()))
                    pair_state = torch.atanh(pair_mean.clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7))
                    shaped[:, 0, selected_tensor] = pair_state
                    shaped[:, 1, selected_tensor] = pair_state
                    output = controller.motor_drive(state)
                finite = finite and bool(
                    torch.isfinite(state).all() and torch.isfinite(output).all()
                )
        activity = torch.tanh(state).reshape(len(block), 2, -1)
        motor.append(output.reshape(len(block), 2, 4).cpu())
        for name, indices in targets.items():
            downstream[name].append(activity[:, :, indices].mean(dim=2).cpu())
    return {
        "terminal_motor": torch.cat(motor),
        "terminal_targets": {name: torch.cat(values) for name, values in downstream.items()},
        "all_states_and_outputs_finite": finite,
        "pair_mean_activity_maximum_absolute": pair_mean_maximum,
    }


def pair_mean_intervention_report(
    args: argparse.Namespace,
    controller: commissioning.CommissionedController,
    sequences: dict[tuple[int, bool], Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    anatomy = frozen_audit.anatomy_manifest(SimpleNamespace(graph=args.graph, raw_dir=args.raw_dir))
    cases = tuple(range(len({case for case, reverse in sequences if not reverse})))
    baseline = _intervention_variant(
        controller,
        sequences,
        cases,
        anatomy["selected_indices"],
        anatomy["visual_target_type_indices"],
        intervene=False,
        device=device,
    )
    intervened = _intervention_variant(
        controller,
        sequences,
        cases,
        anatomy["selected_indices"],
        anatomy["visual_target_type_indices"],
        intervene=True,
        device=device,
    )
    baseline_contrast = baseline["terminal_motor"][:, 1] - baseline["terminal_motor"][:, 0]
    intervened_contrast = intervened["terminal_motor"][:, 1] - intervened["terminal_motor"][:, 0]
    axes = {}
    for axis, name in enumerate(("roll", "pitch", "yaw", "throttle")):
        source_rms = float(baseline_contrast[:, axis].square().mean().sqrt())
        changed_rms = float(intervened_contrast[:, axis].square().mean().sqrt())
        axes[name] = {
            "baseline_contrast_rms": source_rms,
            "intervened_contrast_rms": changed_rms,
            "rms_ratio": changed_rms / max(source_rms, 1.0e-30),
            "common_output_drift_rms": float(
                (
                    intervened["terminal_motor"][:, :, axis].mean(dim=1)
                    - baseline["terminal_motor"][:, :, axis].mean(dim=1)
                )
                .square()
                .mean()
                .sqrt()
            ),
        }
    targets = {}
    for name in baseline["terminal_targets"]:
        source = baseline["terminal_targets"][name]
        changed = intervened["terminal_targets"][name]
        source_contrast = source[:, 1] - source[:, 0]
        changed_contrast = changed[:, 1] - changed[:, 0]
        source_rms = float(source_contrast.square().mean().sqrt())
        changed_rms = float(changed_contrast.square().mean().sqrt())
        targets[name] = {
            "baseline_contrast_rms": source_rms,
            "intervened_contrast_rms": changed_rms,
            "rms_ratio": changed_rms / max(source_rms, 1.0e-30),
            "common_activity_drift_rms": float(
                (changed.mean(dim=1) - source.mean(dim=1)).square().mean().sqrt()
            ),
        }
    return {
        "diagnostic_only": True,
        "cases": len(cases),
        "motor_axes": axes,
        "visual_targets": targets,
        "all_states_and_outputs_finite": bool(
            baseline["all_states_and_outputs_finite"]
            and intervened["all_states_and_outputs_finite"]
        ),
        "pair_mean_activity_maximum_absolute": intervened["pair_mean_activity_maximum_absolute"],
    }


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def make_optimizer(controller: commissioning.CommissionedController) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [controller.gain], "lr": 0.02, "name": "gain"},
            {"params": [controller.bias_offset], "lr": 0.01, "name": "bias_offset"},
            {"params": [controller.tau_ratio], "lr": 0.01, "name": "tau_ratio"},
        ],
        betas=(0.0, 0.99),
        eps=1.0e-8,
    )


def optimizer_step_counters(optimizer: torch.optim.Adam) -> dict[str, int]:
    counters = {}
    for group in optimizer.param_groups:
        name = str(group["name"])
        if len(group["params"]) != 1:
            raise RuntimeError(f"optimizer group {name} must contain exactly one parameter")
        step = optimizer.state.get(group["params"][0], {}).get("step", 0)
        if isinstance(step, Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise RuntimeError(f"optimizer step for {name} is not a finite scalar")
            value = float(step.detach().cpu())
        else:
            value = float(step)
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise RuntimeError(f"optimizer step for {name} is not a nonnegative integer")
        counters[name] = int(value)
    return counters


def finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, (int, np.integer, bool, str)):
        return True
    if isinstance(value, dict):
        return all(finite_numeric_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_numeric_tree(item) for item in value)
    return value is None


def parameters_are_finite(controller: commissioning.CommissionedController) -> bool:
    return all(
        bool(torch.isfinite(value).all()) for value in controller.parameter_values().values()
    )


def optimizer_state_is_finite(optimizer: torch.optim.Adam) -> bool:
    try:
        optimizer_step_counters(optimizer)
    except RuntimeError:
        return False
    return finite_numeric_tree(optimizer.state_dict())


def optimizer_history_is_consistent(history: list[dict[str, Any]], expected_accepted: int) -> bool:
    accepted = 0
    names = {"gain", "bias_offset", "tau_ratio"}
    for proposal, item in enumerate(history, start=1):
        before = item.get("optimizer_steps_before", {})
        after = item.get("optimizer_steps_after", {})
        if (
            item.get("proposal") != proposal
            or not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != names
            or set(after) != names
            or set(before.values()) != {accepted}
        ):
            return False
        accepted += int(bool(item.get("accepted")))
        if set(after.values()) != {accepted}:
            return False
    return accepted == expected_accepted


def optimizer_proposal(
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    specs: list[dict[str, Any]],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    cases: tuple[int, int, int, int],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> dict[str, Any]:
    base_rates = (0.02, 0.01, 0.01)
    parameter_snapshot = controller.parameter_values()
    optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
    steps_before = optimizer_step_counters(optimizer)
    committed = False
    try:
        for group, rate in zip(optimizer.param_groups, base_rates, strict=True):
            group["lr"] = rate
        controller.zero_grad(set_to_none=True)
        baseline, components = batch_loss(
            controller,
            specs,
            sequences,
            source_bank,
            cases,
            anatomy,
            device=device,
            checkpoint_frames=True,
        )
        baseline_value = float(baseline.detach().cpu())
        baseline_components = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }
        baseline.backward()
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            (controller.gain, controller.bias_offset, controller.tau_ratio),
            preflight.GRADIENT_NORM_CLIP,
        )
        unclipped_norm_value = float(unclipped_norm.detach().cpu())
        gradients = {
            name: getattr(controller, name).grad.detach().cpu().clone()
            for name in ("gain", "bias_offset", "tau_ratio")
        }
        finite_gradients = bool(
            math.isfinite(baseline_value)
            and finite_numeric_tree(baseline_components)
            and math.isfinite(unclipped_norm_value)
            and all(bool(torch.isfinite(value).all()) for value in gradients.values())
        )
        trials = []
        accepted = None
        if finite_gradients:
            for multiplier in preflight.BACKTRACK_MULTIPLIERS:
                controller.load_parameter_values(parameter_snapshot)
                optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
                for group, rate in zip(optimizer.param_groups, base_rates, strict=True):
                    group["lr"] = rate * multiplier
                for name, gradient in gradients.items():
                    getattr(controller, name).grad = gradient.to(device).clone()
                optimizer.step()
                controller.project_parameters()
                with torch.no_grad():
                    candidate, candidate_components = batch_loss(
                        controller,
                        specs,
                        sequences,
                        source_bank,
                        cases,
                        anatomy,
                        device=device,
                        checkpoint_frames=False,
                    )
                candidate_value = float(candidate.detach().cpu())
                candidate_component_values = {
                    name: float(value.detach().cpu())
                    for name, value in candidate_components.items()
                }
                steps_after_trial = optimizer_step_counters(optimizer)
                step_advanced_once = all(
                    steps_after_trial[name] == steps_before[name] + 1 for name in steps_before
                )
                finite = bool(
                    math.isfinite(candidate_value)
                    and finite_numeric_tree(candidate_component_values)
                    and parameters_are_finite(controller)
                    and optimizer_state_is_finite(optimizer)
                    and step_advanced_once
                )
                nonincreasing = bool(finite and candidate_value <= baseline_value)
                trial = {
                    "multiplier": multiplier,
                    "loss": candidate_value,
                    "components": candidate_component_values,
                    "finite": finite,
                    "optimizer_step_advanced_once": step_advanced_once,
                    "nonincreasing": nonincreasing,
                    "accepted": nonincreasing,
                }
                trials.append(trial)
                if nonincreasing:
                    accepted = trial
                    break
        if accepted is None:
            controller.load_parameter_values(parameter_snapshot)
            optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
        steps_after = optimizer_step_counters(optimizer)
        expected_increment = 1 if accepted is not None else 0
        if any(
            steps_after[name] != steps_before[name] + expected_increment for name in steps_before
        ):
            raise RuntimeError("optimizer counters did not advance exactly once on acceptance")
        if not parameters_are_finite(controller) or not optimizer_state_is_finite(optimizer):
            raise RuntimeError("proposal left nonfinite parameters or optimizer state")
        optimizer.zero_grad(set_to_none=True)
        result = {
            "accepted": accepted is not None,
            "cases": list(cases),
            "baseline_loss": baseline_value,
            "baseline_components": baseline_components,
            "finite_gradients": finite_gradients,
            "unclipped_gradient_norm": unclipped_norm_value,
            "accepted_multiplier": accepted["multiplier"] if accepted is not None else None,
            "trials": trials,
            "optimizer_steps_before": steps_before,
            "optimizer_steps_after": steps_after,
            "parameter_values_after": {
                name: value.tolist() for name, value in controller.parameter_values().items()
            },
        }
        committed = accepted is not None
        return result
    finally:
        if not committed:
            controller.load_parameter_values(parameter_snapshot)
            optimizer.load_state_dict(copy.deepcopy(optimizer_snapshot))
        optimizer.zero_grad(set_to_none=True)


def _load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _save_state(
    path: Path,
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
    state: dict[str, Any],
) -> None:
    payload = copy.deepcopy(state)
    payload["parameter_values"] = controller.parameter_values()
    payload["optimizer_state"] = optimizer.state_dict()
    _atomic_torch_save(payload, path)


def _load_or_create_source_bank(
    cache_path: Path,
    metadata_path: Path,
    controller: torch.nn.Module,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[dict[str, dict[int, dict[str, Tensor]]], dict[str, Any]]:
    if cache_path.is_file() or metadata_path.is_file():
        if not cache_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("source cache is incomplete")
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        if registration.file_sha256(cache_path) != metadata["file_sha256"]:
            raise RuntimeError("source cache file hash changed")
        bank = _load_torch(cache_path)
        if commissioning.semantic_sha256(bank) != metadata["semantic_sha256"]:
            raise RuntimeError("source cache tensor hash changed")
        return bank, metadata
    with torch.inference_mode():
        normal, reverse = evaluate_cases(
            controller,
            sequences,
            anatomy,
            cases,
            device=device,
            checkpoint_frames=False,
        )
    bank = response_bank_cpu(normal, reverse, cases)
    semantic = commissioning.semantic_sha256(bank)
    _atomic_torch_save(bank, cache_path)
    metadata = {
        "cases": len(cases),
        "semantic_sha256": semantic,
        "file_sha256": registration.file_sha256(cache_path),
    }
    registration.write_exclusive(metadata_path, metadata)
    return bank, metadata


def _candidate_bank(
    controller: commissioning.CommissionedController,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...],
    *,
    device: torch.device,
) -> dict[str, dict[int, dict[str, Tensor]]]:
    with torch.inference_mode():
        normal, reverse = evaluate_cases(
            controller,
            sequences,
            anatomy,
            cases,
            device=device,
            checkpoint_frames=False,
        )
    return response_bank_cpu(normal, reverse, cases)


def _development_cache(
    args: argparse.Namespace,
    source: torch.nn.Module,
    specs: list[dict[str, Any]],
    anatomy: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[
    dict[tuple[int, bool], Tensor],
    dict[str, dict[int, dict[str, Tensor]]],
    dict[str, Any],
]:
    pixels_path = args.output_dir / "development-pixels.pt"
    source_path = args.output_dir / "development-source-responses.pt"
    metadata_path = args.output_dir / "development-cache.json"
    if pixels_path.is_file() or source_path.is_file() or metadata_path.is_file():
        if not (pixels_path.is_file() and source_path.is_file() and metadata_path.is_file()):
            raise RuntimeError("development cache is incomplete")
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        if (
            registration.file_sha256(pixels_path) != metadata["pixels_file_sha256"]
            or registration.file_sha256(source_path) != metadata["source_file_sha256"]
        ):
            raise RuntimeError("development cache file hash changed")
        sequences = _load_torch(pixels_path)
        source_bank = _load_torch(source_path)
        if (
            commissioning.semantic_sha256(sequences) != metadata["pixels_semantic_sha256"]
            or commissioning.semantic_sha256(source_bank) != metadata["source_semantic_sha256"]
        ):
            raise RuntimeError("development cache tensor hash changed")
        return sequences, source_bank, metadata
    sequences = render_bank(specs)
    _atomic_torch_save(sequences, pixels_path)
    cases = tuple(range(len(specs)))
    source_bank = _candidate_bank(source, sequences, anatomy, cases, device=device)
    _atomic_torch_save(source_bank, source_path)
    metadata = {
        "cases": len(cases),
        "pixels_semantic_sha256": commissioning.semantic_sha256(sequences),
        "pixels_file_sha256": registration.file_sha256(pixels_path),
        "source_semantic_sha256": commissioning.semantic_sha256(source_bank),
        "source_file_sha256": registration.file_sha256(source_path),
    }
    registration.write_exclusive(metadata_path, metadata)
    return sequences, source_bank, metadata


def _snapshot_selection(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not history:
        return None
    passing = [item for item in history if item["qualification"]["pass"]]
    if passing:
        selected = min(passing, key=lambda item: (item["loss"]["loss"], item["proposal"]))
        reason = "smallest development loss among gate-passing snapshots"
        authorized = True
    else:
        selected = min(
            history,
            key=lambda item: (
                item["qualification"]["maximum_normalized_gate_shortfall"],
                item["loss"]["loss"],
                item["proposal"],
            ),
        )
        reason = "reporting only: smallest maximum normalized gate shortfall"
        authorized = False
    return {
        "proposal": selected["proposal"],
        "snapshot_path": selected["snapshot_path"],
        "snapshot_file_sha256": selected["snapshot_file_sha256"],
        "development_loss": selected["loss"]["loss"],
        "maximum_normalized_gate_shortfall": selected["qualification"][
            "maximum_normalized_gate_shortfall"
        ],
        "acceptance_authorized": authorized,
        "reason": reason,
    }


def acquire_run_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = (output_dir / "run.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise SystemExit(
            "another vertical-motion training process owns this output directory"
        ) from None
    return stream


def validate_training_state(
    state: dict[str, Any],
    controller: commissioning.CommissionedController,
    optimizer: torch.optim.Adam,
) -> None:
    completed = state.get("proposal_completed")
    accepted = state.get("accepted_proposals")
    rejected = state.get("rejected_proposals")
    history = state.get("proposal_history", [])
    if not (
        isinstance(completed, int)
        and isinstance(accepted, int)
        and isinstance(rejected, int)
        and 0 <= completed <= MAX_PROPOSALS
        and accepted >= 0
        and rejected >= 0
        and accepted + rejected == completed
        and len(history) == completed
        and [item.get("proposal") for item in history] == list(range(1, completed + 1))
        and optimizer_history_is_consistent(history, accepted)
    ):
        raise RuntimeError("proposal counters or history are inconsistent")
    in_flight = state.get("proposal_in_flight")
    if in_flight is not None and in_flight != completed + 1:
        raise RuntimeError("proposal in-flight marker is inconsistent")
    counters = optimizer_step_counters(optimizer)
    if any(value != accepted for value in counters.values()):
        raise RuntimeError("Adam counters do not equal the accepted-proposal count")
    if not parameters_are_finite(controller) or not optimizer_state_is_finite(optimizer):
        raise RuntimeError("training state contains nonfinite parameters or optimizer values")


def acceptance_prerequisite_errors(state: dict[str, Any]) -> list[str]:
    errors = []
    expected = set(SCHEDULED_EVALUATIONS)
    history = state.get("proposal_history", [])
    training = state.get("training_evaluations", [])
    development = state.get("development_history", [])

    if state.get("proposal_completed") != MAX_PROPOSALS or len(history) != MAX_PROPOSALS:
        errors.append("the complete proposal history is missing")
    if state.get("accepted_proposals", 0) + state.get("rejected_proposals", 0) != MAX_PROPOSALS:
        errors.append("accepted/rejected proposal counters are incomplete")
    if state.get("proposal_in_flight") is not None:
        errors.append("a proposal remains in flight")

    def exact_events(items: list[dict[str, Any]]) -> bool:
        proposals = [item.get("proposal") for item in items]
        return len(proposals) == len(expected) and set(proposals) == expected

    if (
        len(state.get("scheduled_evaluation_started", [])) != len(expected)
        or set(state.get("scheduled_evaluation_started", [])) != expected
    ):
        errors.append("scheduled-evaluation claims are not exactly 25, 50 and 100")
    if not exact_events(training):
        errors.append("training evaluations are not exactly 25, 50 and 100")
    if not exact_events(development):
        errors.append("development evaluations are not exactly 25, 50 and 100")
    if (
        len(state.get("development_started", [])) != len(expected)
        or set(state.get("development_started", [])) != expected
    ):
        errors.append("development-start claims are not exactly 25, 50 and 100")

    by_proposal = {item.get("proposal"): item for item in training}
    if (
        not by_proposal.get(MANDATORY_TRAINING_GATE_PROPOSAL, {})
        .get("mandatory_decision", {})
        .get("pass")
    ):
        errors.append("the mandatory proposal-25 training gate did not pass")
    if not all(
        by_proposal.get(proposal, {}).get("numerical", {}).get("pass") for proposal in expected
    ):
        errors.append("one or more scheduled numerical gates did not pass")
    if any(not item.get("finite_gradients") for item in history):
        errors.append("one or more proposals had nonfinite baseline or gradients")
    if not finite_numeric_tree(training) or not finite_numeric_tree(development):
        errors.append("scheduled evaluation metrics contain nonfinite values")
    return errors


def resume_interruption(state: dict[str, Any]) -> tuple[str, str] | None:
    if state.get("proposal_in_flight") is not None:
        return (
            f"interrupted proposal {state['proposal_in_flight']} cannot be retried",
            "vertical_motion_training_proposal_interrupted",
        )
    completed = {item["proposal"] for item in state["development_history"]}
    interrupted_development = set(state["development_started"]) - completed
    if interrupted_development:
        proposals = ",".join(str(value) for value in sorted(interrupted_development))
        return (
            f"interrupted development evaluation at proposal {proposals}",
            "vertical_motion_training_development_interrupted",
        )
    interrupted_scheduled = set(state["scheduled_evaluation_started"]) - completed
    if interrupted_scheduled:
        proposals = ",".join(str(value) for value in sorted(interrupted_scheduled))
        return (
            f"interrupted scheduled evaluation at proposal {proposals}",
            "vertical_motion_training_scheduled_evaluation_interrupted",
        )
    return None


def main() -> int:
    args = parse_args()
    report_path = args.output_dir / "report.json"
    start_path = args.output_dir / "start.json"
    state_path = args.output_dir / "state.pt"
    _run_lock = acquire_run_lock(args.output_dir)
    if report_path.exists():
        raise SystemExit("vertical-motion commissioning training already terminated")
    deterministic.configure_determinism()
    device = torch.device(args.device)
    runtime = deterministic.runtime_manifest(device)
    input_hashes = validate_inputs(args)
    registered = commissioning.load_registered_manifest(args.registration)
    splits = registered["stimuli"]["splits"]
    training_specs = splits["training"]["specs"]
    training_batches = balanced_batches(training_specs, split="training")
    anatomy = commissioning.anatomy_arrays(args.graph, args.raw_dir / registration.ANNOTATIONS_FILE)
    source_state = commissioning.load_source_checkpoint(args.checkpoint)
    source_state_sha256 = commissioning.semantic_sha256(source_state)
    implementation_sha256 = registration.file_sha256(Path(__file__))
    start = {
        "experiment": EXPERIMENT,
        "protocol": protocol_manifest(),
        "input_file_sha256": input_hashes,
        "source_state_sha256": source_state_sha256,
        "implementation_commit": preflight._git_head(),
        "implementation_file_sha256": {
            registration.stable_path(Path(__file__)): implementation_sha256
        },
        "runtime": runtime,
    }
    if start_path.is_file():
        with start_path.open() as stream:
            existing_start = json.load(stream)
        for key in (
            "experiment",
            "protocol",
            "input_file_sha256",
            "source_state_sha256",
            "implementation_file_sha256",
            "runtime",
        ):
            if existing_start.get(key) != start[key]:
                raise SystemExit(f"training start marker changed at {key}")
        start = existing_start
    else:
        registration.write_exclusive(start_path, start)

    torch.cuda.reset_peak_memory_stats(device)
    source = commissioning.make_source_controller(args.graph, source_state, device=device)
    source_before = commissioning.semantic_sha256(source.state_dict())
    controller = commissioning.CommissionedController(source, anatomy).to(device)
    optimizer = make_optimizer(controller)
    training_sequences = render_bank(training_specs)
    training_pixels_sha256 = commissioning.semantic_sha256(training_sequences)
    training_source, training_source_metadata = _load_or_create_source_bank(
        args.output_dir / "training-source-responses.pt",
        args.output_dir / "training-source-cache.json",
        source,
        training_sequences,
        anatomy,
        tuple(range(len(training_specs))),
        device=device,
    )
    training_baseline = bank_loss(
        training_specs,
        training_batches,
        training_source,
        training_source,
        anatomy,
    )
    baseline_strata = edge_direction_strata(
        training_specs,
        training_batches,
        training_source,
        training_source,
        anatomy,
    )
    if state_path.is_file():
        state = _load_torch(state_path)
        if (
            state.get("protocol_commit") != PROTOCOL_COMMIT
            or state.get("implementation_sha256") != implementation_sha256
            or state.get("training_pixels_sha256") != training_pixels_sha256
            or state.get("training_source_semantic_sha256")
            != training_source_metadata["semantic_sha256"]
            or state.get("training_baseline") != training_baseline
            or state.get("baseline_strata") != baseline_strata
        ):
            raise SystemExit("training resume state changed")
        controller.load_parameter_values(state["parameter_values"])
        optimizer.load_state_dict(state["optimizer_state"])
    else:
        state = {
            "protocol_commit": PROTOCOL_COMMIT,
            "implementation_sha256": implementation_sha256,
            "training_pixels_sha256": training_pixels_sha256,
            "training_source_semantic_sha256": training_source_metadata["semantic_sha256"],
            "training_baseline": training_baseline,
            "baseline_strata": baseline_strata,
            "proposal_completed": 0,
            "proposal_in_flight": None,
            "accepted_proposals": 0,
            "rejected_proposals": 0,
            "proposal_history": [],
            "training_evaluations": [],
            "scheduled_evaluation_started": [],
            "development_started": [],
            "development_history": [],
            "stop_reason": None,
        }
        _save_state(state_path, controller, optimizer, state)

    validate_training_state(state, controller, optimizer)

    classification = "vertical_motion_training_exception_failed_closed"
    passed = False
    exception = None
    selection = None
    acceptance = None
    intervention = None
    retained_checkpoint = None
    interruption = resume_interruption(state)
    if interruption is not None:
        state["stop_reason"], classification = interruption
    acceptance_started_path = args.output_dir / "acceptance-started.json"
    if acceptance_started_path.is_file():
        state["stop_reason"] = "interrupted acceptance evaluation"
        classification = "vertical_motion_training_acceptance_interrupted"
    try:
        while state["stop_reason"] is None and state["proposal_completed"] < MAX_PROPOSALS:
            proposal = state["proposal_completed"] + 1
            cases = training_batches[(proposal - 1) % len(training_batches)]
            state["proposal_in_flight"] = proposal
            _save_state(state_path, controller, optimizer, state)
            print(json.dumps({"stage": "training_proposal", "proposal": proposal}), flush=True)
            proposal_report = optimizer_proposal(
                controller,
                optimizer,
                training_specs,
                training_sequences,
                training_source,
                cases,
                anatomy,
                device=device,
            )
            proposal_report["proposal"] = proposal
            state["proposal_history"].append(proposal_report)
            state["proposal_completed"] = proposal
            state["proposal_in_flight"] = None
            counter = "accepted_proposals" if proposal_report["accepted"] else "rejected_proposals"
            state[counter] += 1
            if proposal in SCHEDULED_EVALUATIONS:
                state["scheduled_evaluation_started"].append(proposal)
            validate_training_state(state, controller, optimizer)
            _save_state(state_path, controller, optimizer, state)

            if proposal not in SCHEDULED_EVALUATIONS:
                continue
            print(
                json.dumps({"stage": "full_training_evaluation", "proposal": proposal}), flush=True
            )
            candidate_training = _candidate_bank(
                controller,
                training_sequences,
                anatomy,
                tuple(range(len(training_specs))),
                device=device,
            )
            training_loss = bank_loss(
                training_specs,
                training_batches,
                training_source,
                candidate_training,
                anatomy,
                controller.parameter_values(),
            )
            candidate_strata = edge_direction_strata(
                training_specs,
                training_batches,
                training_source,
                candidate_training,
                anatomy,
            )
            training_entry = {
                "proposal": proposal,
                "loss": training_loss,
                "edge_direction_strata": candidate_strata,
            }
            if proposal == MANDATORY_TRAINING_GATE_PROPOSAL:
                training_entry["mandatory_decision"] = mandatory_training_decision(
                    training_baseline,
                    training_loss,
                    baseline_strata,
                    candidate_strata,
                )
            state["training_evaluations"].append(training_entry)
            del candidate_training
            if (
                proposal == MANDATORY_TRAINING_GATE_PROPOSAL
                and not training_entry["mandatory_decision"]["pass"]
            ):
                state["stop_reason"] = "mandatory proposal-25 training gate failed"
                classification = "vertical_motion_training_mandatory_gate_failed"
                _save_state(state_path, controller, optimizer, state)
                break

            print(json.dumps({"stage": "K32_vs_K64", "proposal": proposal}), flush=True)
            parameters = controller.parameter_values()
            numerical = scheduled_numerical_gate(
                args,
                controller,
                parameters,
                source_state,
                training_specs,
                training_sequences,
                anatomy,
                device=device,
            )
            training_entry["numerical"] = numerical
            if not numerical["pass"]:
                state["stop_reason"] = f"scheduled numerical gate failed at proposal {proposal}"
                classification = "vertical_motion_training_numerical_gate_failed"
                _save_state(state_path, controller, optimizer, state)
                break

            snapshot_path = args.output_dir / f"candidate-{proposal:03d}.pt"
            snapshot = {
                "experiment": EXPERIMENT,
                "proposal": proposal,
                "source_state_sha256": source_state_sha256,
                "parameter_values": parameters,
            }
            _atomic_torch_save(snapshot, snapshot_path)
            snapshot_sha256 = registration.file_sha256(snapshot_path)
            state["development_started"].append(proposal)
            _save_state(state_path, controller, optimizer, state)
            registration.write_exclusive(
                args.output_dir / f"development-{proposal:03d}-started.json",
                {
                    "experiment": EXPERIMENT,
                    "proposal": proposal,
                    "snapshot_file_sha256": snapshot_sha256,
                },
            )

            print(json.dumps({"stage": "development", "proposal": proposal}), flush=True)
            development_specs = splits["development"]["specs"]
            development_sequences, development_source, development_cache = _development_cache(
                args,
                source,
                development_specs,
                anatomy,
                device=device,
            )
            candidate_development = _candidate_bank(
                controller,
                development_sequences,
                anatomy,
                tuple(range(len(development_specs))),
                device=device,
            )
            development_loss = complete_bank_loss(
                development_specs,
                development_source,
                candidate_development,
                anatomy,
                parameters,
            )
            development_qualification = qualification_metrics(
                development_specs, candidate_development, anatomy
            )
            development_entry = {
                "proposal": proposal,
                "snapshot_path": registration.stable_path(snapshot_path),
                "snapshot_file_sha256": snapshot_sha256,
                "cache": development_cache,
                "loss": development_loss,
                "qualification": development_qualification,
                "response_semantic_sha256": commissioning.semantic_sha256(candidate_development),
            }
            state["development_history"].append(development_entry)
            _save_state(state_path, controller, optimizer, state)
            del candidate_development, development_sequences, development_source

        if state["stop_reason"] is None and state["proposal_completed"] == MAX_PROPOSALS:
            validate_training_state(state, controller, optimizer)
            prerequisite_errors = acceptance_prerequisite_errors(state)
            if prerequisite_errors:
                state["stop_reason"] = "; ".join(prerequisite_errors)
                classification = "vertical_motion_training_acceptance_prerequisites_failed"
            else:
                selection = _snapshot_selection(state["development_history"])
            if state["stop_reason"] is None and (
                selection is None or not selection["acceptance_authorized"]
            ):
                state["stop_reason"] = "no development snapshot passed every gate"
                classification = "vertical_motion_training_development_failed"
            elif state["stop_reason"] is None:
                selected_path = REPO_ROOT / selection["snapshot_path"]
                if registration.file_sha256(selected_path) != selection["snapshot_file_sha256"]:
                    raise RuntimeError("selected development snapshot hash changed")
                selected = _load_torch(selected_path)
                controller.load_parameter_values(selected["parameter_values"])
                registration.write_exclusive(
                    acceptance_started_path,
                    {
                        "experiment": EXPERIMENT,
                        "selected_proposal": selection["proposal"],
                        "snapshot_file_sha256": selection["snapshot_file_sha256"],
                    },
                )
                print(json.dumps({"stage": "acceptance"}), flush=True)
                del training_sequences
                acceptance_specs = splits["acceptance"]["specs"]
                acceptance_sequences = render_bank(acceptance_specs)
                acceptance_pixels_sha256 = commissioning.semantic_sha256(acceptance_sequences)
                acceptance_source = _candidate_bank(
                    source,
                    acceptance_sequences,
                    anatomy,
                    tuple(range(len(acceptance_specs))),
                    device=device,
                )
                candidate_acceptance = _candidate_bank(
                    controller,
                    acceptance_sequences,
                    anatomy,
                    tuple(range(len(acceptance_specs))),
                    device=device,
                )
                acceptance_loss = complete_bank_loss(
                    acceptance_specs,
                    acceptance_source,
                    candidate_acceptance,
                    anatomy,
                    controller.parameter_values(),
                )
                acceptance_qualification = qualification_metrics(
                    acceptance_specs, candidate_acceptance, anatomy
                )
                all_textures = texture_direction_metrics(
                    acceptance_specs, candidate_acceptance, anatomy
                )
                novel = novel_texture_speed_metrics(acceptance_specs, candidate_acceptance, anatomy)
                intervention = pair_mean_intervention_report(
                    args, controller, acceptance_sequences, device=device
                )
                acceptance_finite = bool(
                    parameters_are_finite(controller)
                    and optimizer_state_is_finite(optimizer)
                    and finite_numeric_tree(acceptance_source)
                    and finite_numeric_tree(candidate_acceptance)
                    and finite_numeric_tree(acceptance_loss)
                    and finite_numeric_tree(acceptance_qualification)
                    and finite_numeric_tree(all_textures)
                    and finite_numeric_tree(novel)
                )
                acceptance_pass = bool(
                    acceptance_finite
                    and acceptance_qualification["pass"]
                    and all_textures["pass"]
                    and novel["pass"]
                )
                acceptance = {
                    "pass": acceptance_pass,
                    "all_parameters_optimizer_responses_and_metrics_finite": acceptance_finite,
                    "pixels_semantic_sha256": acceptance_pixels_sha256,
                    "source_response_semantic_sha256": commissioning.semantic_sha256(
                        acceptance_source
                    ),
                    "candidate_response_semantic_sha256": commissioning.semantic_sha256(
                        candidate_acceptance
                    ),
                    "loss": acceptance_loss,
                    "qualification": acceptance_qualification,
                    "all_texture_directions": all_textures,
                    "novel_texture_speed": novel,
                }
                if acceptance_pass:
                    classification = "vertical_motion_training_passed"
                    passed = True
                else:
                    state["stop_reason"] = "acceptance qualification failed"
                    classification = "vertical_motion_training_acceptance_failed"
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}
        state["stop_reason"] = f"exception: {type(error).__name__}: {error}"
    finally:
        source_after = commissioning.semantic_sha256(source.state_dict())
        source_restored = source_after == source_before
        peak_reserved = torch.cuda.max_memory_reserved(device)
        memory_pass = peak_reserved <= preflight.PEAK_RESERVED_LIMIT_BYTES
        if not source_restored:
            passed = False
            classification = "vertical_motion_training_source_restoration_failed"
        elif not memory_pass:
            passed = False
            classification = "vertical_motion_training_memory_failed"

    if passed:
        retained_path = args.output_dir / "vertical-motion-module.pt"
        retained_payload = {
            "experiment": EXPERIMENT,
            "source_state_sha256": source_state_sha256,
            "selected_proposal": selection["proposal"],
            "parameter_values": controller.parameter_values(),
            "scope": "vertical-motion-module research checkpoint only",
        }
        _atomic_torch_save(retained_payload, retained_path)
        retained_checkpoint = {
            "path": registration.stable_path(retained_path),
            "file_sha256": registration.file_sha256(retained_path),
            "parameter_semantic_sha256": commissioning.semantic_sha256(
                retained_payload["parameter_values"]
            ),
        }
    state["stop_reason"] = state["stop_reason"] or (
        "completed with passing acceptance" if passed else "terminal failure"
    )
    _save_state(state_path, controller, optimizer, state)
    report = {
        "experiment": EXPERIMENT,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": protocol_manifest(),
        "classification": classification,
        "passed": passed,
        "input_file_sha256": input_hashes,
        "implementation_commit": start["implementation_commit"],
        "implementation_file_sha256": start["implementation_file_sha256"],
        "runtime": runtime,
        "source_state_sha256": source_state_sha256,
        "source_controller_state_sha256_before": source_before,
        "source_controller_state_sha256_after": source_after,
        "source_restored": source_restored,
        "training_pixels_semantic_sha256": training_pixels_sha256,
        "training_source_cache": training_source_metadata,
        "training_baseline": training_baseline,
        "baseline_edge_direction_strata": baseline_strata,
        "proposals_completed": state["proposal_completed"],
        "accepted_proposals": state["accepted_proposals"],
        "rejected_proposals": state["rejected_proposals"],
        "proposal_history": state["proposal_history"],
        "training_evaluations": state["training_evaluations"],
        "scheduled_evaluation_started": state["scheduled_evaluation_started"],
        "development_started": state["development_started"],
        "development_history": state["development_history"],
        "selection": selection,
        "acceptance": acceptance,
        "pair_mean_intervention": intervention,
        "retained_checkpoint": retained_checkpoint,
        "stop_reason": state["stop_reason"],
        "exception": exception,
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_peak_reserved_gibibytes": peak_reserved / 1024**3,
        "cuda_peak_reserved_memory_pass": memory_pass,
        "vertical_motion_module_retained": retained_checkpoint is not None,
        "motion_to_DN_VNC_routing_preregistration_authorized": passed,
        "hover_or_gate_flight_authorized": False,
        "promotion_authorized": False,
    }
    registration.write_exclusive(report_path, _json_safe(report))
    print(
        json.dumps(
            {
                "report": registration.stable_path(report_path),
                "classification": classification,
                "passed": passed,
                "proposals_completed": state["proposal_completed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
