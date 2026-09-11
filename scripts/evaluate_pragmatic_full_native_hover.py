#!/usr/bin/env python3
"""Compare full-MaleCNS hover checkpoints with all four axes under native control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_variable_height_responsibilities as responsibility  # noqa: E402
import preregister_vertical_motion_commissioning as motion_registration  # noqa: E402
import train_variable_height_native_throttle_assisted as assisted  # noqa: E402
import vertical_motion_commissioning as motion  # noqa: E402

from flydrone.hover import ConnectomeController, HoverConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=REPO_ROOT / "runs/visual-hover/paired-dynamic-001/controller.pt",
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
        default=REPO_ROOT / "runs/variable-height-hover/pragmatic-full-native-001",
    )
    parser.add_argument(
        "--motion-archive",
        type=Path,
        default=(
            REPO_ROOT
            / "runs/optic-motion/vertical-motion-commissioning-training-001/state.pt"
        ),
    )
    parser.add_argument(
        "--motion-witness",
        type=Path,
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-frozen-witness-001/report.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=520_983)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-motion-variants", action="store_true")
    return parser.parse_args()


def _load_source(args: argparse.Namespace, device: torch.device) -> ConnectomeController:
    payload = torch.load(args.source_checkpoint, map_location=device, weights_only=True)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / assisted.POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.eval().requires_grad_(False)
    return controller


def _load_candidate(
    args: argparse.Namespace, device: torch.device
) -> tuple[ConnectomeController, dict[str, Any]]:
    payload = torch.load(args.candidate_resume, map_location="cpu", weights_only=True)
    controller = ConnectomeController(args.graph, neural_dt=1.0 / assisted.POLICY_HZ).to(device)
    controller.load_state_dict(payload["controller"])
    controller.eval().requires_grad_(False)
    metadata = {
        "experiment": payload.get("experiment"),
        "protocol_commit": payload.get("protocol_commit"),
        "accepted_updates": payload.get("accepted_updates"),
        "attempted_updates": payload.get("attempted_updates"),
        "run_state": payload.get("run_state"),
    }
    return controller, metadata


def _motion_variant(
    args: argparse.Namespace,
    base: ConnectomeController,
    *,
    apply_witness: bool,
    device: torch.device,
) -> motion.CommissionedController:
    anatomy = motion.anatomy_arrays(
        args.graph,
        REPO_ROOT / "data/raw/malecns-v1.0" / motion_registration.ANNOTATIONS_FILE,
    )
    controller = motion.CommissionedController(base, anatomy).to(device)
    archive = torch.load(args.motion_archive, map_location="cpu", weights_only=True)
    parameters = {
        name: value.detach().clone() for name, value in archive["parameter_values"].items()
    }
    if apply_witness:
        witness = json.loads(args.motion_witness.read_text())
        direction = torch.tensor(
            witness["result"]["witness_control"]["direction"], dtype=torch.float32
        )
        displacement = float(witness["result"]["trials"][0]["reference_displacement_norm"])
        offset = 0
        for name in ("gain", "bias_offset", "tau_ratio"):
            count = parameters[name].numel()
            parameters[name] = parameters[name] + displacement * direction[offset : offset + count]
            offset += count
    controller.load_parameter_values(parameters)
    controller.project_parameters()
    controller.eval().requires_grad_(False)
    return controller


def _evaluate(
    controller: ConnectomeController,
    cases: dict[str, Any],
    *,
    frozen_vision: bool,
    device: torch.device,
    config: HoverConfig,
    batch_size: int,
) -> dict[str, Any]:
    report, _ = assisted.evaluate_hover_cases(
        controller,
        cases,
        teacher_all_axes=False,
        native_all_axes=True,
        frozen_vision=frozen_vision,
        device=device,
        config=config,
        batch_size=batch_size,
    )
    return report


def main() -> int:
    args = parse_args()
    if args.cases < 16 or args.cases % 16:
        raise SystemExit("--cases must be a positive multiple of 16")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    for path in (
        args.graph,
        args.source_checkpoint,
        args.candidate_resume,
        args.motion_archive,
        args.motion_witness,
    ):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    config = HoverConfig()
    source = _load_source(args, device)
    candidate, candidate_metadata = _load_candidate(args, device)
    candidate_with_motion = None
    candidate_with_witness = None
    if not args.skip_motion_variants:
        proposal_base, _ = _load_candidate(args, device)
        witness_base, _ = _load_candidate(args, device)
        candidate_with_motion = _motion_variant(
            args, proposal_base, apply_witness=False, device=device
        )
        candidate_with_witness = _motion_variant(
            args, witness_base, apply_witness=True, device=device
        )
    if source.uses_accelerometer or source.uses_proprioception:
        raise SystemExit("source actor unexpectedly requires added sensor channels")
    if candidate.uses_accelerometer or candidate.uses_proprioception:
        raise SystemExit("candidate actor unexpectedly requires added sensor channels")

    counts = {
        "train": 0,
        "held_out_marker": args.cases // 2,
        "held_out_combination": args.cases // 2,
    }
    cases = assisted.build_hover_case_bank(
        seed=args.seed,
        counts=counts,
        device=device,
        config=config,
    )
    started = perf_counter()
    print(json.dumps({"stage": "source_live", "cases": args.cases}), flush=True)
    source_live = _evaluate(
        source,
        cases,
        frozen_vision=False,
        device=device,
        config=config,
        batch_size=args.batch_size,
    )
    print(json.dumps({"stage": "candidate_live", "cases": args.cases}), flush=True)
    candidate_live = _evaluate(
        candidate,
        cases,
        frozen_vision=False,
        device=device,
        config=config,
        batch_size=args.batch_size,
    )
    candidate_motion = None
    candidate_witness = None
    if candidate_with_motion is not None and candidate_with_witness is not None:
        print(json.dumps({"stage": "candidate_with_motion", "cases": args.cases}), flush=True)
        candidate_motion = _evaluate(
            candidate_with_motion,
            cases,
            frozen_vision=False,
            device=device,
            config=config,
            batch_size=args.batch_size,
        )
        print(
            json.dumps({"stage": "candidate_with_witness", "cases": args.cases}),
            flush=True,
        )
        candidate_witness = _evaluate(
            candidate_with_witness,
            cases,
            frozen_vision=False,
            device=device,
            config=config,
            batch_size=args.batch_size,
        )
    print(json.dumps({"stage": "candidate_frozen", "cases": args.cases}), flush=True)
    candidate_frozen = _evaluate(
        candidate,
        cases,
        frozen_vision=True,
        device=device,
        config=config,
        batch_size=args.batch_size,
    )

    result = {
        "experiment": "pragmatic-full-native-hover-comparison-v1",
        "purpose": "rapid proof-of-concept evaluation; not a promotion or acceptance run",
        "actor": {
            "inputs": ["320x200 linear RGB at 125 degree HFOV", "roll", "pitch"],
            "state": "native MaleCNS recurrence only",
            "outputs": ["roll", "pitch", "yaw", "throttle"],
            "teacher_action_used": False,
            "accelerometer_used": False,
            "external_history_used": False,
        },
        "case_manifest": assisted.hover_case_support_report(cases),
        "source": source_live,
        "candidate": candidate_live,
        "candidate_with_proposal25_motion": candidate_motion,
        "candidate_with_common_descent_witness": candidate_witness,
        "candidate_frozen_vision": candidate_frozen,
        "candidate_metadata": candidate_metadata,
        "inputs": {
            "graph": str(args.graph),
            "graph_sha256": responsibility.file_sha256(args.graph),
            "source_checkpoint": str(args.source_checkpoint),
            "source_checkpoint_sha256": responsibility.file_sha256(args.source_checkpoint),
            "candidate_resume": str(args.candidate_resume),
            "candidate_resume_sha256": responsibility.file_sha256(args.candidate_resume),
            "motion_archive": str(args.motion_archive),
            "motion_archive_sha256": responsibility.file_sha256(args.motion_archive),
            "motion_witness": str(args.motion_witness),
            "motion_witness_sha256": responsibility.file_sha256(args.motion_witness),
        },
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": perf_counter() - started,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
