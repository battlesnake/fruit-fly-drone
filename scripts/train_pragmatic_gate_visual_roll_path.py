#!/usr/bin/env python3
"""Tune only short anatomical photoreceptor-to-roll-motor paths for gate steering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
from search_pragmatic_full_native_gate_es import (  # noqa: E402
    evaluate_cases,
    sample_mirrored_cases,
)
from train_pragmatic_full_native_gate_imitation import (  # noqa: E402
    GateRollout,
    new_rollout,
    save_checkpoint,
    teacher_motor,
)

from flydrone.gate import GateConfig, render_annular_gate_rgb  # noqa: E402
from flydrone.hover import (  # noqa: E402
    ConnectomeController,
    DifferentiableQuad,
    ForelegStickPlant,
    HoverConfig,
)
from flydrone.visual_hover import CameraSpec  # noqa: E402

POLICY_HZ = 50
PHYSICS_HZ = 100


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
        default=REPO_ROOT / "runs/gate/pragmatic-full-native-staged-001/best-controller.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs/gate/pragmatic-visual-roll-path-001",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--path-hop-budget", type=int, default=5)
    parser.add_argument("--teacher-updates", type=int, default=200)
    parser.add_argument("--native-updates", type=int, default=200)
    parser.add_argument("--training-pairs", type=int, default=2)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--contrast-weight", type=float, default=4.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--observation-warmup-steps", type=int, default=10)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--evaluation-pairs", type=int, default=16)
    parser.add_argument("--evaluation-seconds", type=float, default=9.0)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate the supplied checkpoint with live and frozen vision, then exit",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        help="optional JSON destination for --evaluate-only",
    )
    parser.add_argument("--seed", type=int, default=600_983)
    parser.add_argument("--evaluation-seed", type=int, default=610_983)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for path in (args.graph, args.checkpoint):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    positive = (
        args.path_hop_budget,
        args.teacher_updates,
        args.native_updates,
        args.training_pairs,
        args.unroll,
        args.learning_rate,
        args.contrast_weight,
        args.anchor_weight,
        args.gradient_clip_norm,
        args.evaluation_interval,
        args.evaluation_pairs,
        args.evaluation_seconds,
    )
    if min(positive) <= 0:
        raise SystemExit("training sizes, rates, and durations must be positive")
    if args.observation_warmup_steps < 0:
        raise SystemExit("observation warmup cannot be negative")


def _minimum_distances(
    node_count: int,
    edge_pre: np.ndarray,
    edge_post: np.ndarray,
    sources: np.ndarray,
    *,
    reverse: bool,
) -> np.ndarray:
    rows = edge_post if reverse else edge_pre
    columns = edge_pre if reverse else edge_post
    virtual_rows = np.full(len(sources), node_count, dtype=np.int64)
    graph = csr_matrix(
        (
            np.ones(len(rows) + len(sources), dtype=np.int8),
            (
                np.concatenate((rows, virtual_rows)),
                np.concatenate((columns, sources.astype(np.int64))),
            ),
        ),
        shape=(node_count + 1, node_count + 1),
    )
    order, predecessor = breadth_first_order(
        graph,
        node_count,
        directed=True,
        return_predecessors=True,
    )
    distance = np.full(node_count + 1, -1, dtype=np.int32)
    distance[node_count] = 0
    for node in order[1:]:
        distance[node] = distance[predecessor[node]] + 1
    return distance[:node_count] - 1


def visual_roll_path_mask(
    graph_path: Path,
    *,
    hop_budget: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, int]]:
    graph = np.load(graph_path)
    node_count = len(graph["node_ids"])
    edge_pre = graph["edge_pre"]
    edge_post = graph["edge_post"]
    visual = graph["visual_node_indices"]
    offsets = graph["output_pool_offsets"]
    pools = graph["output_pool_indices"]
    roll_motors = pools[offsets[0] : offsets[2]]
    from_visual = _minimum_distances(
        node_count,
        edge_pre,
        edge_post,
        visual,
        reverse=False,
    )
    to_roll = _minimum_distances(
        node_count,
        edge_pre,
        edge_post,
        roll_motors,
        reverse=True,
    )
    path_nodes = (from_visual >= 0) & (to_roll >= 0) & ((from_visual + to_roll) <= hop_budget)
    selected = (
        path_nodes[edge_pre]
        & path_nodes[edge_post]
        & ((from_visual[edge_pre] + 1 + to_roll[edge_post]) <= hop_budget)
    )
    mask = torch.from_numpy(selected).to(device=device)
    manifest = {
        "hop_budget": hop_budget,
        "minimum_visual_to_roll_hops": int(from_visual[roll_motors].min()),
        "selected_nodes": int(path_nodes.sum()),
        "selected_edges": int(selected.sum()),
        "selected_photoreceptors": int(path_nodes[visual].sum()),
        "roll_motor_nodes": len(roll_motors),
    }
    if not manifest["selected_edges"]:
        raise ValueError("hop budget selects no visual-to-roll edges")
    return mask, manifest


def load_controller(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConnectomeController, dict[str, Any]]:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload.get("graph_sha256") != responsibility.file_sha256(args.graph):
        raise SystemExit("checkpoint and graph hashes do not match")
    controller = ConnectomeController(args.graph, neural_dt=1.0 / POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.bias.requires_grad_(False)
    controller.raw_time_constant.requires_grad_(False)
    controller.edge_magnitude.requires_grad_(True)
    return controller, payload


@torch.no_grad()
def warm_rollout(
    controller: ConnectomeController,
    rollout: GateRollout,
    *,
    steps: int,
    camera: CameraSpec,
    gate_config: GateConfig,
) -> None:
    if not steps:
        return
    image = render_annular_gate_rgb(
        rollout.state,
        rollout.gate,
        camera=camera,
        gate_config=gate_config,
    )
    for _ in range(steps):
        _, rollout.neural = controller(
            image,
            rollout.state.euler[:, :2],
            rollout.neural,
        )


def roll_training_step(
    controller: ConnectomeController,
    optimizer: torch.optim.Optimizer,
    rollout: GateRollout,
    source_edges: Tensor,
    selected_mask: Tensor,
    *,
    unroll: int,
    native_physics: bool,
    contrast_weight: float,
    anchor_weight: float,
    gradient_clip_norm: float,
    camera: CameraSpec,
    config: HoverConfig,
    gate_config: GateConfig,
) -> tuple[dict[str, float], GateRollout]:
    quad = DifferentiableQuad(config).to(rollout.neural.device)
    stick_plant = ForelegStickPlant(config).to(rollout.neural.device)
    direct_losses = []
    contrast_losses = []
    target_rms = []
    prediction_rms = []
    for _ in range(unroll):
        image = render_annular_gate_rgb(
            rollout.state,
            rollout.gate,
            camera=camera,
            gate_config=gate_config,
        )
        prediction, rollout.neural = controller(
            image,
            rollout.state.euler[:, :2],
            rollout.neural,
        )
        target = teacher_motor(rollout.state, rollout.gate, config, mode="staged")
        direct_losses.append(((prediction[:, 0] - target[:, 0]) / 0.10).square().mean())
        prediction_pair = prediction[:, 0].reshape(-1, 2)
        target_pair = target[:, 0].reshape(-1, 2)
        prediction_contrast = prediction_pair[:, 1] - prediction_pair[:, 0]
        target_contrast = target_pair[:, 1] - target_pair[:, 0]
        contrast_losses.append(((prediction_contrast - target_contrast) / 0.15).square().mean())
        target_rms.append(target[:, 0].square().mean().sqrt().detach())
        prediction_rms.append(prediction[:, 0].square().mean().sqrt().detach())
        applied_motor = prediction.detach() if native_physics else target
        for _ in range(PHYSICS_HZ // POLICY_HZ):
            rc, rollout.sticks = stick_plant(applied_motor, rollout.sticks)
            rollout.state = quad(rc, rollout.state, rollout.mass_scale)
        rollout.state = rollout.state.detach()
        rollout.sticks = rollout.sticks.detach()
        rollout.age += 1
    direct = torch.stack(direct_losses).mean()
    contrast = torch.stack(contrast_losses).mean()
    anchor = (
        ((controller.edge_magnitude[selected_mask] - source_edges[selected_mask]) / 0.25)
        .square()
        .mean()
    )
    loss = direct + contrast_weight * contrast + anchor_weight * anchor
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_((controller.edge_magnitude,), gradient_clip_norm)
    optimizer.step()
    controller.project_parameters()
    rollout.neural = rollout.neural.detach()
    valid = bool(torch.isfinite(rollout.state.position).all()) and bool(
        (rollout.state.position[:, 2] > 0.03).all()
    )
    return (
        {
            "loss": float(loss.detach()),
            "roll_direct": float(direct.detach()),
            "roll_contrast": float(contrast.detach()),
            "selected_edge_anchor": float(anchor.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "target_roll_motor_rms": float(torch.stack(target_rms).mean()),
            "predicted_roll_motor_rms": float(torch.stack(prediction_rms).mean()),
            "rollout_age_seconds": rollout.age / POLICY_HZ,
            "rollout_valid": valid,
        },
        rollout,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    responsibility.seed_everything(args.seed)
    controller, source_payload = load_controller(args, device)
    selected_mask, path_manifest = visual_roll_path_mask(
        args.graph,
        hop_budget=args.path_hop_budget,
        device=device,
    )
    source_edges = controller.edge_magnitude.detach().clone()
    gradient_mask = selected_mask.to(dtype=controller.edge_magnitude.dtype)
    controller.edge_magnitude.register_hook(lambda gradient: gradient * gradient_mask)
    optimizer = torch.optim.Adam(
        (controller.edge_magnitude,),
        lr=args.learning_rate,
    )
    config = HoverConfig()
    gate_config = GateConfig()
    camera = CameraSpec()
    evaluation_cases = sample_mirrored_cases(
        args.evaluation_pairs,
        seed=args.evaluation_seed,
        device=device,
        hover_config=config,
    )
    baseline = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.evaluation_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
        observation_warmup_steps=args.observation_warmup_steps,
    )
    print(
        json.dumps({"stage": "baseline", "path": path_manifest, "flight": baseline}),
        flush=True,
    )
    if args.evaluate_only:
        frozen = evaluate_cases(
            controller,
            evaluation_cases,
            seconds=args.evaluation_seconds,
            camera=camera,
            hover_config=config,
            gate_config=gate_config,
            frozen_vision=True,
            observation_warmup_steps=args.observation_warmup_steps,
        )
        result = {
            "experiment": "pragmatic-full-native-gate-fresh-evaluation-v1",
            "checkpoint": str(args.checkpoint),
            "evaluation_seed": args.evaluation_seed,
            "path": path_manifest,
            "live": baseline,
            "frozen_after_half_second": frozen,
        }
        if args.evaluation_output is not None:
            args.evaluation_output.parent.mkdir(parents=True, exist_ok=True)
            args.evaluation_output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        print(json.dumps({"stage": "evaluation_complete", **result}, indent=2), flush=True)
        return 0
    started = perf_counter()
    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    best_path = args.output_dir / "best-controller.pt"
    rollout_seed = args.seed
    global_update = 0
    for stage, updates, native_physics in (
        ("teacher_paths", args.teacher_updates, False),
        ("native_paths", args.native_updates, True),
    ):
        rollout_seed += 1
        rollout = new_rollout(
            controller,
            pairs=args.training_pairs,
            seed=rollout_seed,
            device=device,
            config=config,
        )
        warm_rollout(
            controller,
            rollout,
            steps=args.observation_warmup_steps,
            camera=camera,
            gate_config=gate_config,
        )
        for stage_update in range(1, updates + 1):
            global_update += 1
            max_age = round(args.evaluation_seconds * POLICY_HZ)
            valid = bool(torch.isfinite(rollout.state.position).all()) and bool(
                (rollout.state.position[:, 2] > 0.03).all()
            )
            if rollout.age + args.unroll > max_age or not valid:
                rollout_seed += 1
                rollout = new_rollout(
                    controller,
                    pairs=args.training_pairs,
                    seed=rollout_seed,
                    device=device,
                    config=config,
                )
                warm_rollout(
                    controller,
                    rollout,
                    steps=args.observation_warmup_steps,
                    camera=camera,
                    gate_config=gate_config,
                )
            metrics, rollout = roll_training_step(
                controller,
                optimizer,
                rollout,
                source_edges,
                selected_mask,
                unroll=args.unroll,
                native_physics=native_physics,
                contrast_weight=args.contrast_weight,
                anchor_weight=args.anchor_weight,
                gradient_clip_norm=args.gradient_clip_norm,
                camera=camera,
                config=config,
                gate_config=gate_config,
            )
            evaluate_now = stage_update % args.evaluation_interval == 0 or stage_update == updates
            entry = {
                "stage": stage,
                "stage_update": stage_update,
                "global_update": global_update,
                **metrics,
                "elapsed_seconds": perf_counter() - started,
            }
            if evaluate_now:
                entry["native_flight"] = evaluate_cases(
                    controller,
                    evaluation_cases,
                    seconds=args.evaluation_seconds,
                    camera=camera,
                    hover_config=config,
                    gate_config=gate_config,
                    observation_warmup_steps=args.observation_warmup_steps,
                )
                flight = entry["native_flight"]
                score = (
                    flight["paired_pass_rate"],
                    min(
                        flight["pass_rate_negative_offset"],
                        flight["pass_rate_positive_offset"],
                    ),
                    flight["pass_rate"],
                    flight["fitness"],
                )
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        best_path,
                        controller,
                        graph=args.graph,
                        source_checkpoint=args.checkpoint,
                        config=config,
                        gate_config=gate_config,
                        camera=camera,
                        stage=stage,
                        update=global_update,
                    )
            if stage_update == 1 or stage_update % 10 == 0 or evaluate_now:
                history.append(entry)
                print(json.dumps(entry), flush=True)

    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    controller.load_state_dict(best_payload["controller"])
    best_live = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.evaluation_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
        observation_warmup_steps=args.observation_warmup_steps,
    )
    best_frozen = evaluate_cases(
        controller,
        evaluation_cases,
        seconds=args.evaluation_seconds,
        camera=camera,
        hover_config=config,
        gate_config=gate_config,
        frozen_vision=True,
        observation_warmup_steps=args.observation_warmup_steps,
    )
    changed = (controller.edge_magnitude.detach() - source_edges).abs()
    report = {
        "experiment": "pragmatic-full-native-gate-visual-roll-path-v1",
        "purpose": "rapid behavioral proof of concept; not a formal promotion run",
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "state": "native MaleCNS recurrence only",
            "outputs": ["roll", "pitch", "yaw", "throttle"],
            "teacher_or_gate_geometry_used_during_deployment": False,
            "external_history_or_state_machine_used": False,
        },
        "path": path_manifest,
        "training": {
            "teacher_updates": args.teacher_updates,
            "native_updates": args.native_updates,
            "training_pairs": args.training_pairs,
            "unroll": args.unroll,
            "learning_rate": args.learning_rate,
            "contrast_weight": args.contrast_weight,
            "anchor_weight": args.anchor_weight,
            "observation_warmup_steps": args.observation_warmup_steps,
        },
        "baseline": baseline,
        "best_score": list(best_score),
        "best_live": best_live,
        "best_frozen_after_half_second": best_frozen,
        "history": history,
        "parameter_audit": {
            "selected_edge_max_absolute_change": float(changed[selected_mask].max()),
            "unselected_edge_max_absolute_change": float(changed[~selected_mask].max()),
            "bias_and_time_constants_frozen": True,
        },
        "checkpoint": str(best_path),
        "source": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": responsibility.file_sha256(args.checkpoint),
            "experiment": source_payload.get("experiment"),
            "stage": source_payload.get("stage"),
            "update": source_payload.get("update"),
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": perf_counter() - started,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "stage": "complete",
                "best_live": best_live,
                "best_frozen": best_frozen,
                "parameter_audit": report["parameter_audit"],
                "checkpoint": str(best_path),
                "report": str(report_path),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
