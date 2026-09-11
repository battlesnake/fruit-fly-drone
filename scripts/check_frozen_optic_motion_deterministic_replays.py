#!/usr/bin/env python3
"""Require three bit-exact fresh-process deterministic optic-motion replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import audit_frozen_optic_motion as base  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports", type=Path, nargs=deterministic.REPLAY_PROCESSES
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"deterministic replay gate already exists: {args.output}")
    resolved_paths = [path.resolve() for path in args.reports]
    if len(set(resolved_paths)) != deterministic.REPLAY_PROCESSES:
        raise SystemExit("deterministic replay reports must have distinct paths")
    reports = []
    for replay_index, path in enumerate(args.reports, start=1):
        terminal = deterministic.validate_completed_phase(
            path, phase=f"replay-{replay_index}"
        )
        with path.open() as stream:
            report = json.load(stream)
        deterministic.validate_replay_report(report)
        if terminal["process_identity"] != report["process_identity"]:
            raise SystemExit("replay terminal identity does not match its report")
        reports.append(report)
    process_identities = [item["process_identity"] for item in reports]
    if len(
        {item["nonce"] for item in process_identities}
    ) != deterministic.REPLAY_PROCESSES:
        raise SystemExit("deterministic replay reports reused a process identity")
    tensor_hashes = {item["response_tensor_sha256"] for item in reports}
    source_hashes = {item["source_state_sha256"] for item in reports}
    runtime_hashes = {
        base.assisted.audit.semantic_sha256(item["runtime"]) for item in reports
    }
    metadata_hashes = {
        base.assisted.audit.semantic_sha256(item["evaluation_metadata"])
        for item in reports
    }
    passed = bool(
        len(tensor_hashes) == 1
        and len(source_hashes) == 1
        and len(runtime_hashes) == 1
        and len(metadata_hashes) == 1
    )
    if not passed:
        raise SystemExit("fresh-process deterministic optic-motion replays differed")
    gate = {
        "experiment": deterministic.EXPERIMENT,
        "protocol_commit": deterministic.PROTOCOL_COMMIT,
        "protocol": deterministic.protocol_manifest(),
        "passed": True,
        "processes": len(reports),
        "response_tensor_sha256": next(iter(tensor_hashes)),
        "source_state_sha256": next(iter(source_hashes)),
        "runtime_semantic_sha256": next(iter(runtime_hashes)),
        "evaluation_metadata_semantic_sha256": next(iter(metadata_hashes)),
        "replay_reports": [
            {
                "path": base.assisted.responsibility.stable_path(path),
                "sha256": base.file_sha256(path),
                "process_identity": report["process_identity"],
            }
            for path, report in zip(args.reports, reports, strict=True)
        ],
        "candidate_retained": False,
        "training_execution_authorized": False,
        "hover_gate_or_promotion_authorized": False,
    }
    base.assisted._atomic_json_save(gate, args.output)
    print(
        json.dumps(
            {
                "gate": base.assisted.responsibility.stable_path(args.output),
                "passed": True,
                "response_tensor_sha256": gate["response_tensor_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
