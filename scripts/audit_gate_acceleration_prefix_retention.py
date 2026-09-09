#!/usr/bin/env python3
"""Test whether an acceleration-trained connectome retains early mass evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_gate import RETINAL_FLIP_X, file_sha256, seed_everything  # noqa: E402
from train_gate_acceleration_oracle_distillation import (  # noqa: E402
    delta_metrics,
)
from train_gate_mass_current_distillation import (  # noqa: E402
    paired_launch,
    stable_path,
    target_bias,
)

from flydrone.gate import GateConfig, render_annular_gate  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "connectome.npz",
    )
    parser.add_argument(
        "--baseline-graph",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-v1" / "connectome.npz",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-accel-v2" / "controller.pt",
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        default=(REPO_ROOT / "runs" / "gate" / "acceleration-oracle-distillation-v1" / "best.pt"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "artifacts" / "gate-mass-oracle-v1" / "candidate.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "gate" / "acceleration-prefix-retention",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=430_031)
    parser.add_argument("--cutoffs", type=float, nargs="+", default=(0.25, 0.50, 0.75))
    parser.add_argument("--end-seconds", type=float, default=1.50)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.25)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    for path in (
        args.graph,
        args.baseline_graph,
        args.source_checkpoint,
        args.candidate_checkpoint,
        args.calibration,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if args.episodes <= 0 or args.episodes % 2:
        raise SystemExit("--episodes must be a positive even number")
    if not args.cutoffs or min(args.cutoffs) < dt:
        raise SystemExit(f"cutoffs must be at least one simulation interval ({dt:g}s)")
    if max(args.cutoffs) >= args.end_seconds:
        raise SystemExit("all cutoffs must occur before --end-seconds")
    if args.sample_interval_seconds < dt:
        raise SystemExit("sample interval must be at least one simulation interval")


def build_controller(
    graph: Path,
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
    neural_dt: float,
) -> ConnectomeController:
    controller = ConnectomeController(
        graph,
        neural_dt=neural_dt,
        retinal_receptive_field=int(checkpoint["retinal_receptive_field"]),
    ).to(device)
    controller.load_state_dict(checkpoint["controller"])
    controller.eval()
    return controller


def paired_prefix(
    candidate: ConnectomeController,
    reference: ConnectomeController,
    *,
    episodes: int,
    steps: int,
    seed: int,
    resolution: int,
    device: torch.device,
    hover_config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a shared baseline-policy flight prefix and return both neural states."""

    seed_everything(seed)
    quad = DifferentiableQuad(hover_config).to(device)
    sticks = ForelegStickPlant(hover_config).to(device)
    state, gate, mass_scale = paired_launch(
        episodes,
        device=device,
        hover_config=hover_config,
        gate_config=gate_config,
    )
    candidate_state = candidate.initial_state(episodes, device=device, dtype=torch.float32)
    reference_state = reference.initial_state(episodes, device=device, dtype=torch.float32)
    stick_state = sticks.initial_state(episodes, device=device, dtype=torch.float32)
    for _ in range(steps):
        image = render_annular_gate(
            state,
            gate,
            resolution=resolution,
            hover_config=hover_config,
            gate_config=gate_config,
        )
        _, candidate_state = candidate(
            image,
            state.euler[:, :2],
            candidate_state,
            state.specific_force,
        )
        reference_motor, reference_state = reference(
            image,
            state.euler[:, :2],
            reference_state,
            state.specific_force,
        )
        rc, stick_state = sticks(reference_motor, stick_state)
        state = quad(rc, state, mass_scale)
    return candidate_state, reference_state, mass_scale


def neutral_continuation(
    candidate: ConnectomeController,
    reference: ConnectomeController,
    candidate_state: Tensor,
    reference_state: Tensor,
    mass_scale: Tensor,
    calibration: dict[str, Any],
    *,
    cutoff_steps: int,
    end_steps: int,
    sample_interval_steps: int,
    resolution: int,
    added_nodes: Tensor,
    reset_added_state: bool,
) -> dict[str, Any]:
    """Continue every episode with identical sensors while retaining recurrent state."""

    candidate_state = candidate_state.clone()
    reference_state = reference_state.clone()
    if reset_added_state:
        candidate_state[:, added_nodes] = 0.0
    oracle_state = reference_state.clone()
    batch = candidate_state.shape[0]
    device = candidate_state.device
    neutral_image = torch.zeros(batch, resolution, resolution, device=device)
    neutral_attitude = torch.zeros(batch, 2, device=device)
    neutral_specific_force = torch.zeros(batch, 3, device=device)
    neutral_specific_force[:, 2] = 9.81
    mass_code = ((mass_scale - 1.0) / 0.08).clamp(-1.0, 1.0)
    oracle_bias = target_bias(mass_code, calibration)
    results: dict[str, Any] = {}
    for zero_step in range(cutoff_steps, end_steps):
        candidate_motor, candidate_state = candidate(
            neutral_image,
            neutral_attitude,
            candidate_state,
            neutral_specific_force,
        )
        reference_motor, reference_state = reference(
            neutral_image,
            neutral_attitude,
            reference_state,
            neutral_specific_force,
        )
        oracle_motor, oracle_state = reference(
            neutral_image,
            neutral_attitude,
            oracle_state,
            neutral_specific_force,
            privileged_throttle_pool_bias=oracle_bias,
        )
        completed_steps = zero_step + 1
        if completed_steps % sample_interval_steps and completed_steps != end_steps:
            continue
        prediction = candidate_motor[:, 3] - reference_motor[:, 3]
        target = oracle_motor[:, 3] - reference_motor[:, 3]
        results[f"{completed_steps * candidate.neural_dt:g}"] = {
            **delta_metrics(prediction, target),
            "prediction_mass_correlation": delta_metrics(prediction, mass_code)[
                "pearson_correlation"
            ],
            "prediction_light_mean": float(prediction[mass_code < 0.0].mean()),
            "prediction_heavy_mean": float(prediction[mass_code >= 0.0].mean()),
        }
    return results


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    source = torch.load(args.source_checkpoint, map_location=device, weights_only=True)
    candidate_checkpoint = torch.load(
        args.candidate_checkpoint, map_location=device, weights_only=True
    )
    graph_hash = file_sha256(args.graph)
    for name, checkpoint in (("source", source), ("candidate", candidate_checkpoint)):
        if checkpoint["graph_sha256"] != graph_hash:
            raise SystemExit(f"{name} checkpoint and graph hashes do not match")
        if bool(checkpoint.get("retinal_flip_x", False)) != RETINAL_FLIP_X:
            raise SystemExit(f"{name} checkpoint retinal orientation does not match evaluator")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("kind") != "privileged_non_biological_mass_oracle":
        raise SystemExit("calibration is not the expected training-only oracle")
    hover_config = HoverConfig(**source["hover_config"])
    gate_config = GateConfig(**source["gate_config"])
    validate_args(args, hover_config.dt)
    reference = build_controller(
        args.graph,
        source,
        device=device,
        neural_dt=hover_config.dt,
    )
    candidate = build_controller(
        args.graph,
        candidate_checkpoint,
        device=device,
        neural_dt=hover_config.dt,
    )
    with np.load(args.graph) as graph, np.load(args.baseline_graph) as baseline:
        added_mask = ~np.isin(graph["node_ids"], baseline["node_ids"])
    added_nodes = torch.from_numpy(np.flatnonzero(added_mask)).to(device=device)
    resolution = int(source["image_resolution"])
    end_steps = round(args.end_seconds / hover_config.dt)
    interval_steps = round(args.sample_interval_seconds / hover_config.dt)
    started = perf_counter()
    results: dict[str, Any] = {}
    with torch.no_grad():
        for cutoff in args.cutoffs:
            cutoff_steps = round(cutoff / hover_config.dt)
            candidate_state, reference_state, mass_scale = paired_prefix(
                candidate,
                reference,
                episodes=args.episodes,
                steps=cutoff_steps,
                seed=args.seed,
                resolution=resolution,
                device=device,
                hover_config=hover_config,
                gate_config=gate_config,
            )
            cutoff_result: dict[str, Any] = {}
            for reset in (False, True):
                label = "preserved_state" if not reset else "added_nodes_reset"
                cutoff_result[label] = neutral_continuation(
                    candidate,
                    reference,
                    candidate_state,
                    reference_state,
                    mass_scale,
                    calibration,
                    cutoff_steps=cutoff_steps,
                    end_steps=end_steps,
                    sample_interval_steps=interval_steps,
                    resolution=resolution,
                    added_nodes=added_nodes,
                    reset_added_state=reset,
                )
            results[f"{cutoff:g}"] = cutoff_result
            print(json.dumps({"cutoff_seconds": cutoff, **cutoff_result}), flush=True)
    report = {
        "experiment": "acceleration-prefix recurrent-retention audit",
        "claim_scope": (
            "After a genuine flight prefix, every episode receives identical black-image, "
            "zero-attitude, 1g continuation inputs. Any later mass separation must therefore "
            "come from connectome state retained at the cutoff."
        ),
        "external_runtime_history_added": False,
        "graph": stable_path(args.graph),
        "graph_sha256": graph_hash,
        "baseline_graph": stable_path(args.baseline_graph),
        "source_checkpoint": stable_path(args.source_checkpoint),
        "source_checkpoint_sha256": file_sha256(args.source_checkpoint),
        "candidate_checkpoint": stable_path(args.candidate_checkpoint),
        "candidate_checkpoint_sha256": file_sha256(args.candidate_checkpoint),
        "calibration": stable_path(args.calibration),
        "calibration_sha256": file_sha256(args.calibration),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "hover_config": asdict(hover_config),
        "gate_config": asdict(gate_config),
        "episodes": args.episodes,
        "seed": args.seed,
        "cutoffs_seconds": args.cutoffs,
        "end_seconds": args.end_seconds,
        "sample_interval_seconds": args.sample_interval_seconds,
        "added_nodes_reset_control_count": int(added_nodes.numel()),
        "results": results,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": stable_path(report_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
