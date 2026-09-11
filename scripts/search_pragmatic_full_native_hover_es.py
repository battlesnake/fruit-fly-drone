#!/usr/bin/env python3
"""Tune the native motor interface against complete all-native hover flights."""

from __future__ import annotations

import argparse
import json
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

import audit_variable_height_responsibilities as responsibility  # noqa: E402
import search_gate_motor_interface_es as motor_es  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--candidate-resume",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/variable-height-hover/"
            / "native-throttle-assisted-beta1-zero-ladder-001/resume.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/variable-height-hover/pragmatic-motor-es-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--antithetic-directions", type=int, default=4)
    parser.add_argument("--training-cases", type=int, default=16)
    parser.add_argument("--validation-cases", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--sigma-decay", type=float, default=0.97)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--bias-perturbation-scale", type=float, default=0.01)
    parser.add_argument("--log-gain-perturbation-scale", type=float, default=0.05)
    parser.add_argument("--maximum-bias-delta", type=float, default=0.15)
    parser.add_argument("--maximum-gain-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=530_983)
    parser.add_argument("--validation-seed", type=int, default=540_983)
    parser.add_argument("--archive-size", type=int, default=8)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.candidate_resume):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.training_cases < 16 or args.training_cases % 16:
        raise SystemExit("--training-cases must be a positive multiple of 16")
    if args.validation_cases < 32 or args.validation_cases % 32:
        raise SystemExit("--validation-cases must be a positive multiple of 32")
    if min(
        args.generations,
        args.antithetic_directions,
        args.batch_size,
        args.sigma,
        args.sigma_decay,
        args.learning_rate,
        args.archive_size,
    ) <= 0:
        raise SystemExit("search sizes and scales must be positive")


def load_controller(
    args: argparse.Namespace, device: torch.device
) -> tuple[ConnectomeController, dict[str, Any]]:
    payload = torch.load(args.candidate_resume, map_location="cpu", weights_only=True)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / assisted.POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.eval().requires_grad_(False)
    return controller, payload


def apply_vector(
    controller: ConnectomeController,
    base_bias: Tensor,
    base_edge: Tensor,
    vector: Tensor,
    spec: motor_es.MotorInterfaceSpec,
) -> None:
    with torch.no_grad():
        controller.bias.copy_(base_bias)
        controller.edge_magnitude.copy_(base_edge)
        controller.bias[spec.bias_nodes] += vector[spec.bias_parameter]
        controller.edge_magnitude[spec.gain_edges] = (
            base_edge[spec.gain_edges] * torch.exp(vector[spec.gain_parameter])
        ).clamp(max=8.0)


def fitness(metrics: dict[str, Any]) -> float:
    cases = max(int(metrics["cases"]), 1)
    failure_fraction = (metrics["ground_contact_cases"] + metrics["invalid_cases"]) / cases
    return float(
        0.25 * metrics["success_fraction"]
        - metrics["height_rmse_mean_metres"]
        - 0.25 * metrics["height_rmse_p95_metres"]
        - 0.50 * metrics["vertical_speed_rms_mean_mps"]
        - 0.02 * metrics["tilt_rms_mean_degrees"]
        - failure_fraction
    )


def evaluate_vector(
    controller: ConnectomeController,
    base_bias: Tensor,
    base_edge: Tensor,
    vector: Tensor,
    spec: motor_es.MotorInterfaceSpec,
    cases: dict[str, Any],
    *,
    device: torch.device,
    config: HoverConfig,
    batch_size: int,
    frozen_vision: bool = False,
) -> dict[str, Any]:
    apply_vector(controller, base_bias, base_edge, vector, spec)
    metrics, _ = assisted.evaluate_hover_cases(
        controller,
        cases,
        teacher_all_axes=False,
        native_all_axes=True,
        frozen_vision=frozen_vision,
        device=device,
        config=config,
        batch_size=batch_size,
    )
    metrics["fitness"] = fitness(metrics)
    return metrics


def case_bank(
    *, seed: int, cases: int, held_out: bool, device: torch.device, config: HoverConfig
) -> dict[str, Any]:
    if held_out:
        counts = {
            "train": 0,
            "held_out_marker": cases // 2,
            "held_out_combination": cases // 2,
        }
    else:
        counts = {"train": cases, "held_out_marker": 0, "held_out_combination": 0}
    return assisted.build_hover_case_bank(
        seed=seed,
        counts=counts,
        device=device,
        config=config,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    controller, source_payload = load_controller(args, device)
    if controller.uses_accelerometer or controller.uses_proprioception:
        raise SystemExit("candidate unexpectedly requires added sensor channels")
    config = HoverConfig()
    spec = motor_es.motor_interface_spec(
        controller,
        bias_scale=args.bias_perturbation_scale,
        log_gain_scale=args.log_gain_perturbation_scale,
        maximum_bias_delta=args.maximum_bias_delta,
        maximum_gain_ratio=args.maximum_gain_ratio,
    )
    base_bias = controller.bias.detach().clone()
    base_edge = controller.edge_magnitude.detach().clone()
    zero = torch.zeros(len(spec.labels), device=device)
    center = zero.clone()
    rng = np.random.default_rng(args.seed)
    archive: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    for generation in range(1, args.generations + 1):
        cases = case_bank(
            seed=args.seed + generation,
            cases=args.training_cases,
            held_out=False,
            device=device,
            config=config,
        )
        sigma = args.sigma * args.sigma_decay ** (generation - 1)
        epsilon = torch.from_numpy(
            rng.standard_normal((args.antithetic_directions, len(spec.labels))).astype(np.float32)
        ).to(device)
        delta = sigma * spec.scales * epsilon
        plus = (center + delta).clamp(spec.lower, spec.upper)
        minus = (center - delta).clamp(spec.lower, spec.upper)
        candidates = torch.stack((plus, minus), dim=1).reshape(-1, len(spec.labels))
        candidate_metrics = []
        for index, vector in enumerate(candidates):
            metrics = evaluate_vector(
                controller,
                base_bias,
                base_edge,
                vector,
                spec,
                cases,
                device=device,
                config=config,
                batch_size=args.batch_size,
            )
            candidate_metrics.append(metrics)
            key = motor_es.vector_sha256(vector)
            archive[key] = {
                "vector": vector.detach().cpu(),
                "generation": generation,
                "candidate": index,
                "training": metrics,
            }
        scores = np.asarray([item["fitness"] for item in candidate_metrics], dtype=np.float32)
        ranks = np.empty(len(scores), dtype=np.float32)
        ranks[np.argsort(scores)] = np.linspace(-0.5, 0.5, len(scores))
        utilities = torch.from_numpy(ranks).to(device)
        paired_utility = utilities[0::2] - utilities[1::2]
        gradient = (paired_utility[:, None] * epsilon).mean(dim=0) / sigma
        center = (center + args.learning_rate * spec.scales * gradient).clamp(
            spec.lower, spec.upper
        )
        center_metrics = evaluate_vector(
            controller,
            base_bias,
            base_edge,
            center,
            spec,
            cases,
            device=device,
            config=config,
            batch_size=args.batch_size,
        )
        center_key = motor_es.vector_sha256(center)
        archive[center_key] = {
            "vector": center.detach().cpu(),
            "generation": generation,
            "candidate": "center",
            "training": center_metrics,
        }
        best_index = int(np.argmax(scores))
        entry = {
            "generation": generation,
            "sigma": sigma,
            "case_manifest": assisted.hover_case_support_report(cases),
            "center": center_metrics,
            "best_candidate": candidate_metrics[best_index],
            "best_candidate_index": best_index,
            "elapsed_seconds": perf_counter() - started,
        }
        history.append(entry)
        print(
            json.dumps(
                {
                    "stage": "search",
                    "generation": generation,
                    "center_fitness": center_metrics["fitness"],
                    "center_success": center_metrics["success_fraction"],
                    "center_height_rmse": center_metrics["height_rmse_mean_metres"],
                    "best_fitness": candidate_metrics[best_index]["fitness"],
                    "best_success": candidate_metrics[best_index]["success_fraction"],
                    "best_height_rmse": candidate_metrics[best_index][
                        "height_rmse_mean_metres"
                    ],
                }
            ),
            flush=True,
        )
        torch.save(
            {
                "generation": generation,
                "center": center.detach().cpu(),
                "parameter_labels": spec.labels,
                "history": history,
            },
            args.output_dir / "search-state.pt",
        )

    validation_cases = case_bank(
        seed=args.validation_seed,
        cases=args.validation_cases,
        held_out=True,
        device=device,
        config=config,
    )
    ranked_archive = sorted(
        archive.values(), key=lambda item: item["training"]["fitness"], reverse=True
    )[: args.archive_size]
    validation = []
    source_item = {
        "vector": zero.detach().cpu(),
        "generation": 0,
        "candidate": "source",
    }
    for item in [source_item, *ranked_archive]:
        vector = item["vector"].to(device)
        metrics = evaluate_vector(
            controller,
            base_bias,
            base_edge,
            vector,
            spec,
            validation_cases,
            device=device,
            config=config,
            batch_size=args.batch_size,
        )
        validation.append(
            {
                "vector": vector.detach().cpu(),
                "vector_sha256": motor_es.vector_sha256(vector),
                "generation": item["generation"],
                "candidate": item["candidate"],
                "metrics": metrics,
            }
        )
    winner = max(validation, key=lambda item: item["metrics"]["fitness"])
    winner_vector = winner["vector"].to(device)
    frozen = evaluate_vector(
        controller,
        base_bias,
        base_edge,
        winner_vector,
        spec,
        validation_cases,
        device=device,
        config=config,
        batch_size=args.batch_size,
        frozen_vision=True,
    )
    apply_vector(controller, base_bias, base_edge, winner_vector, spec)
    checkpoint = {
        "controller": {
            name: value.detach().cpu() for name, value in controller.state_dict().items()
        },
        "graph_sha256": responsibility.file_sha256(args.graph),
        "source_resume_sha256": responsibility.file_sha256(args.candidate_resume),
        "image_resolution": [320, 200],
        "camera_hfov_degrees": 125.0,
        "hover_config": vars(config),
        "motor_interface_es": {
            "parameter_labels": list(spec.labels),
            "parameter_vector": winner_vector.detach().cpu().tolist(),
            "parameter_vector_sha256": winner["vector_sha256"],
        },
    }
    torch.save(checkpoint, args.output_dir / "controller.pt")
    serializable_validation = [
        {name: value for name, value in item.items() if name != "vector"}
        for item in validation
    ]
    report = {
        "experiment": "pragmatic-full-native-hover-motor-interface-es-v1",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "state": "native MaleCNS recurrence only",
            "outputs": ["roll", "pitch", "yaw", "throttle"],
            "teacher_action_used": False,
            "accelerometer_used": False,
            "external_history_used": False,
        },
        "configuration": {
            **vars(args),
            "graph": str(args.graph),
            "candidate_resume": str(args.candidate_resume),
            "output_dir": str(args.output_dir),
            "parameter_count": len(spec.labels),
            "parameter_labels": list(spec.labels),
        },
        "history": history,
        "validation_case_manifest": assisted.hover_case_support_report(validation_cases),
        "validation": serializable_validation,
        "selected": {
            name: value for name, value in winner.items() if name != "vector"
        },
        "selected_frozen_vision": frozen,
        "checkpoint": str(args.output_dir / "controller.pt"),
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": perf_counter() - started,
        },
        "source_metadata": {
            "experiment": source_payload.get("experiment"),
            "accepted_updates": source_payload.get("accepted_updates"),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "selected": report["selected"],
                "frozen_vision": frozen,
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
