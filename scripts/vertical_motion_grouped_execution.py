"""Grouped CUDA execution for the unchanged vertical-motion objective."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_vertical_motion_gradient_attribution as attribution  # noqa: E402
import preregister_vertical_motion_commissioning as registration  # noqa: E402
import train_vertical_motion_commissioning as training  # noqa: E402
import vertical_motion_commissioning as commissioning  # noqa: E402


def evaluate_cases_batched(
    controller: torch.nn.Module,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...] | list[int],
    *,
    device: torch.device,
    checkpoint_frames: bool,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Evaluate normal and reversed pairs together without mixing their state."""
    case_list = list(cases)
    if not case_list:
        return [], []
    pair_keys = [
        *((case, False) for case in case_list),
        *((case, True) for case in case_list),
    ]
    pair_count = len(pair_keys)
    sequence = torch.stack([sequences[key] for key in pair_keys])
    sequence = sequence.permute(1, 0, 2, 3, 4).reshape(
        registration.MOTION_FRAMES + registration.TERMINAL_FRAMES,
        4 * pair_count,
        registration.HEIGHT,
        registration.WIDTH,
    )
    prefix = commissioning.neutral_prefix(
        controller, device=device, checkpoint_frames=checkpoint_frames
    )
    state = prefix.expand(4 * pair_count, -1).clone()
    selected = torch.from_numpy(anatomy["target_indices"]).to(device)
    integrated_response = torch.zeros(pair_count, 2, len(selected), device=device)
    integrated_static = torch.zeros_like(integrated_response)
    terminal_response = torch.empty_like(integrated_response)
    terminal_static = torch.empty_like(integrated_response)
    output = torch.empty(4 * pair_count, 4, device=device)
    for frame in range(registration.MOTION_FRAMES + registration.TERMINAL_FRAMES):
        gray = sequence[frame].to(device)
        image = gray[:, None].expand(-1, 3, -1, -1)
        output, state = commissioning._advance_frame(
            controller, image, state, checkpoint_frames=checkpoint_frames
        )
        selected_activity = torch.tanh(state[:, selected]).reshape(pair_count, 4, len(selected))
        moving = selected_activity[:, :2]
        stationary = selected_activity[:, 2:]
        response = moving - stationary
        if frame < registration.MOTION_FRAMES:
            integrated_response = integrated_response + response
            integrated_static = integrated_static + stationary
        else:
            terminal_response = response
            terminal_static = stationary
    terminal_motor = output.reshape(pair_count, 4, 4)
    values = []
    for index in range(pair_count):
        values.append(
            {
                "integrated_response": (integrated_response[index] / registration.MOTION_FRAMES),
                "integrated_static": (integrated_static[index] / registration.MOTION_FRAMES),
                "terminal_response": terminal_response[index],
                "terminal_static": terminal_static[index],
                "terminal_motor": terminal_motor[index],
            }
        )
    count = len(case_list)
    return values[:count], values[count:]


def response_bank_batched(
    controller: torch.nn.Module,
    sequences: dict[tuple[int, bool], Tensor],
    anatomy: dict[str, np.ndarray],
    cases: tuple[int, ...],
    *,
    case_block_size: int,
    device: torch.device,
) -> dict[str, dict[int, dict[str, Tensor]]]:
    if case_block_size <= 0:
        raise ValueError("case block size must be positive")
    result: dict[str, dict[int, dict[str, Tensor]]] = {"normal": {}, "reverse": {}}
    with torch.inference_mode():
        for begin in range(0, len(cases), case_block_size):
            block = cases[begin : begin + case_block_size]
            normal, reverse = evaluate_cases_batched(
                controller,
                sequences,
                anatomy,
                block,
                device=device,
                checkpoint_frames=False,
            )
            converted = training.response_bank_cpu(normal, reverse, block)
            result["normal"].update(converted["normal"])
            result["reverse"].update(converted["reverse"])
    return result


def grouped_objective_gradients(
    controller: commissioning.CommissionedController,
    specs: list[dict[str, Any]],
    batches: tuple[tuple[int, int, int, int], ...],
    sequences: dict[tuple[int, bool], Tensor],
    source_bank: dict[str, dict[int, dict[str, Tensor]]],
    anatomy: dict[str, np.ndarray],
    *,
    group_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], Tensor, dict[str, Tensor]]:
    if group_size <= 0:
        raise ValueError("group size must be positive")
    if not batches:
        raise ValueError("at least one original batch is required")
    parameters = tuple(getattr(controller, name) for name in attribution.PARAMETER_NAMES)
    parameter_count = sum(parameter.numel() for parameter in parameters)
    weighted_full = torch.zeros(parameter_count, dtype=torch.float64)
    weighted_strata = {
        name: torch.zeros(parameter_count, dtype=torch.float64)
        for name in attribution.STRATUM_NAMES
    }
    rows: list[dict[str, Any]] = []
    for begin in range(0, len(batches), group_size):
        grouped_batches = batches[begin : begin + group_size]
        grouped_cases = tuple(case for batch in grouped_batches for case in batch)
        normal, reverse = evaluate_cases_batched(
            controller,
            sequences,
            anatomy,
            grouped_cases,
            device=device,
            checkpoint_frames=True,
        )
        losses: list[Tensor] = []
        stratum_values: dict[str, list[Tensor]] = {name: [] for name in attribution.STRATUM_NAMES}
        for offset, cases in enumerate(grouped_batches):
            start = 4 * offset
            stop = start + 4
            batch_normal = normal[start:stop]
            batch_reverse = reverse[start:stop]
            references = training.batch_references(specs, source_bank, cases, anatomy)
            loss, components = commissioning.commissioning_loss(
                [specs[case] for case in cases],
                batch_normal,
                batch_reverse,
                references,
                anatomy,
                controller,
            )
            strata = attribution.differentiable_edge_strata(batch_normal, references, anatomy)
            losses.append(loss)
            for name in attribution.STRATUM_NAMES:
                stratum_values[name].append(strata[name])
            rows.append(
                {
                    "batch": begin + offset,
                    "cases": list(cases),
                    "loss": float(loss.detach().cpu()),
                    "components": {
                        name: float(value.detach().cpu()) for name, value in components.items()
                    },
                    "strata": {name: float(value.detach().cpu()) for name, value in strata.items()},
                }
            )
        objectives = [
            torch.stack(losses).mean(),
            *(torch.stack(stratum_values[name]).mean() for name in attribution.STRATUM_NAMES),
        ]
        gradient_matrix = attribution.batched_objective_gradients(objectives, parameters)
        weight = len(grouped_batches)
        weighted_full += gradient_matrix[0] * weight
        for index, name in enumerate(attribution.STRATUM_NAMES, start=1):
            weighted_strata[name] += gradient_matrix[index] * weight
        print(
            json.dumps(
                {
                    "stage": "grouped_gradient",
                    "group_size": group_size,
                    "batches_complete": begin + len(grouped_batches),
                    "batches": len(batches),
                }
            ),
            flush=True,
        )
        del normal, reverse, losses, stratum_values, objectives, gradient_matrix

    divisor = float(len(batches))
    full_gradient = weighted_full / divisor
    stratum_gradients = {name: value / divisor for name, value in weighted_strata.items()}
    component_names = rows[0]["components"]
    summary = {
        "loss": float(np.mean([row["loss"] for row in rows])),
        "components": {
            name: float(np.mean([row["components"][name] for row in rows]))
            for name in component_names
        },
        "strata": {
            name: float(np.mean([row["strata"][name] for row in rows]))
            for name in attribution.STRATUM_NAMES
        },
        "parameter_order": list(attribution.PARAMETER_NAMES),
        "parameter_count": int(full_gradient.numel()),
        "full_gradient": full_gradient.tolist(),
        "stratum_gradients": {name: value.tolist() for name, value in stratum_gradients.items()},
        "batch_rows": rows,
    }
    return summary, full_gradient, stratum_gradients


def maximum_response_difference(first: Any, second: Any) -> float:
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            return float("inf")
        return max(
            (maximum_response_difference(first[key], second[key]) for key in first),
            default=0.0,
        )
    if isinstance(first, Tensor) and isinstance(second, Tensor):
        if first.shape != second.shape:
            return float("inf")
        if first.numel() == 0:
            return 0.0
        if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
            return float("inf")
        return float(
            torch.max(
                torch.abs(
                    first.detach().cpu().to(torch.float64) - second.detach().cpu().to(torch.float64)
                )
            )
        )
    return 0.0 if first == second else float("inf")
