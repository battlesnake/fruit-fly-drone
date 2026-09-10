#!/usr/bin/env python3
"""Report where paired visual learning changed and activated the MaleCNS graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_visual_height_response import common_prefix, paired_states  # noqa: E402

from flydrone.connectome_data import ANNOTATIONS_FILE, _read_annotations  # noqa: E402
from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402
from flydrone.visual_hover import render_visual_hover_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/pilot-001/controller.pt",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy-hz", type=int, default=50)
    parser.add_argument("--prefix-steps", type=int, default=50)
    parser.add_argument("--response-steps", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def grouped_rms(values: np.ndarray, codes: np.ndarray, group_count: int) -> np.ndarray:
    sums = np.bincount(codes, weights=np.square(values), minlength=group_count)
    counts = np.bincount(codes, minlength=group_count)
    return np.sqrt(sums / np.maximum(counts, 1))


def grouped_mean_absolute(values: np.ndarray, codes: np.ndarray, group_count: int) -> np.ndarray:
    sums = np.bincount(codes, weights=np.abs(values), minlength=group_count)
    counts = np.bincount(codes, minlength=group_count)
    return sums / np.maximum(counts, 1)


@torch.no_grad()
def visual_state_contrast(
    controller: ConnectomeController,
    *,
    device: torch.device,
    prefix_steps: int,
    response_steps: int,
    contrast_clamp: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[float]]:
    torch.manual_seed(901)
    config = HoverConfig()
    state, prefix_target, target_a, target_b = paired_states(1, device, config, 0.20, held_out=True)
    initial_neural, state = common_prefix(controller, state, prefix_target, prefix_steps, config)
    image = torch.cat(
        (
            render_visual_hover_scene(state, target_a),
            render_visual_hover_scene(state, target_b),
        )
    )
    attitude = torch.cat((state.euler[:, :2], state.euler[:, :2]))
    neural = torch.cat((initial_neural.clone(), initial_neural.clone()))
    motor = torch.zeros(2, 4, device=device)
    for _ in range(response_steps):
        _, neural = controller(image, attitude, neural)
        if contrast_clamp is not None:
            # Remove only the marker-dependent difference in the selected population.
            # The paired mean state is preserved, avoiding the much larger distribution
            # shift caused by silencing an entire anatomical superclass.
            common = 0.5 * (neural[0, contrast_clamp] + neural[1, contrast_clamp])
            neural[0, contrast_clamp] = common
            neural[1, contrast_clamp] = common
        motor = controller.motor_drive(neural)
    return (neural[0] - neural[1]).cpu().numpy(), (motor[0] - motor[1]).cpu().tolist()


def main() -> int:
    args = parse_args()
    graph = np.load(args.graph)
    node_ids = graph["node_ids"]
    edge_pre = graph["edge_pre"]
    edge_post = graph["edge_post"]
    annotations = _read_annotations(args.raw_dir / ANNOTATIONS_FILE)
    annotation_rows = {int(body): row for row, body in enumerate(annotations["bodyId"])}
    superclass = np.asarray(
        [str(annotations["superclass"][annotation_rows[int(body)]]) for body in node_ids]
    )
    names, node_codes = np.unique(superclass, return_inverse=True)
    group_count = len(names)

    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)["controller"]
    learned_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    learned = learned_checkpoint["controller"]
    edge_delta = (learned["edge_magnitude"] - source["edge_magnitude"]).numpy()
    bias_delta = (learned["bias"] - source["bias"]).numpy()
    time_delta = (learned["raw_time_constant"] - source["raw_time_constant"]).numpy()

    device = torch.device(args.device)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / args.policy_hz).to(device)
    controller.load_state_dict(learned)
    controller.eval()
    activity_delta, motor_contrast = visual_state_contrast(
        controller,
        device=device,
        prefix_steps=args.prefix_steps,
        response_steps=args.response_steps,
    )

    vnc_names = [name for name in names if str(name).startswith("vnc_")]
    vnc_mask = np.isin(superclass, vnc_names)
    nonmotor_vnc_mask = vnc_mask & (superclass != "vnc_motor")
    intervention_masks = {
        "all_nonmotor_vnc": torch.from_numpy(np.flatnonzero(nonmotor_vnc_mask)).to(device),
        "vnc_intrinsic": torch.from_numpy(np.flatnonzero(superclass == "vnc_intrinsic")).to(device),
        "vnc_sensory": torch.from_numpy(np.flatnonzero(superclass == "vnc_sensory")).to(device),
    }
    contrast_clamp_results = {}
    baseline_throttle_contrast = float(motor_contrast[3])
    for label, mask in intervention_masks.items():
        _, clamped_motor_contrast = visual_state_contrast(
            controller,
            device=device,
            prefix_steps=args.prefix_steps,
            response_steps=args.response_steps,
            contrast_clamp=mask,
        )
        clamped_throttle = float(clamped_motor_contrast[3])
        contrast_clamp_results[label] = {
            "nodes": int(mask.numel()),
            "paired_motor_drive_contrast": clamped_motor_contrast,
            "throttle_contrast_residual_fraction": (
                clamped_throttle / baseline_throttle_contrast
                if abs(baseline_throttle_contrast) > 1.0e-8
                else None
            ),
        }

    node_counts = np.bincount(node_codes, minlength=group_count)
    incoming_codes = node_codes[edge_post]
    outgoing_codes = node_codes[edge_pre]
    incoming_counts = np.bincount(incoming_codes, minlength=group_count)
    bias_rms = grouped_rms(bias_delta, node_codes, group_count)
    time_rms = grouped_rms(time_delta, node_codes, group_count)
    activity_mean = grouped_mean_absolute(activity_delta, node_codes, group_count)
    incoming_edge_rms = grouped_rms(edge_delta, incoming_codes, group_count)
    outgoing_edge_rms = grouped_rms(edge_delta, outgoing_codes, group_count)
    groups = {
        str(name): {
            "nodes": int(node_counts[index]),
            "incoming_edges": int(incoming_counts[index]),
            "marker_state_contrast_mean_absolute": float(activity_mean[index]),
            "bias_update_rms": float(bias_rms[index]),
            "raw_time_constant_update_rms": float(time_rms[index]),
            "incoming_edge_update_rms": float(incoming_edge_rms[index]),
            "outgoing_edge_update_rms": float(outgoing_edge_rms[index]),
        }
        for index, name in enumerate(names)
    }
    report = {
        "source_checkpoint": str(args.source_checkpoint),
        "checkpoint": str(args.checkpoint),
        "prefix_steps": args.prefix_steps,
        "response_steps": args.response_steps,
        "paired_motor_drive_contrast": motor_contrast,
        "interpretation_limit": (
            "The paired contrast clamp is causal for this fixed response assay, but does not "
            "localize a biological algorithm or prove that VNC feedback improves closed-loop "
            "flight. FeCO/stick feedback is not yet connected."
        ),
        "vnc_summary": {
            "nodes": int(vnc_mask.sum()),
            "nonmotor_nodes": int(nonmotor_vnc_mask.sum()),
            "marker_state_contrast_mean_absolute": float(np.abs(activity_delta[vnc_mask]).mean()),
            "nonmotor_marker_state_contrast_mean_absolute": float(
                np.abs(activity_delta[nonmotor_vnc_mask]).mean()
            ),
            "nonmotor_marker_state_contrast_p95_absolute": float(
                np.quantile(np.abs(activity_delta[nonmotor_vnc_mask]), 0.95)
            ),
        },
        "paired_contrast_clamp": {
            "method": (
                "After every recurrent update, replace the two marker-condition states in "
                "the selected population by their pairwise mean; preserve every other state."
            ),
            "baseline_throttle_motor_contrast": baseline_throttle_contrast,
            "interventions": contrast_clamp_results,
        },
        "by_superclass": groups,
    }
    rendered = f"{json.dumps(report, indent=2, sort_keys=True)}\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
