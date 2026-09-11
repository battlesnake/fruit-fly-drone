#!/usr/bin/env python3
"""Validate a terminal cancellation-resistant commissioning preflight v2 report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_vertical_motion_commissioning_preflight as v1_check  # noqa: E402
import preflight_vertical_motion_commissioning_v2 as v2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-002/report.json",
    )
    return parser.parse_args()


def validate_report(report: dict) -> None:
    original = v1_check.preflight
    try:
        v1_check.preflight = v2
        v1_check.validate_report(report)
    finally:
        v1_check.preflight = original
    if report.get("frozen_source_normalizations_semantic_sha256") != (
        v2.EXPECTED_V1_NORMALIZATIONS_SHA256
    ):
        raise SystemExit("v2 source normalizations differ from v1")
    if report.get("rendered_training_identity_subset_float32_sha256") != (
        v2.EXPECTED_RENDERED_SUBSET_SHA256
    ):
        raise SystemExit("v2 rendered training subset differs from v1")
    if (
        report.get("input_file_sha256", {}).get(v2.v1.registration.stable_path(v2.V1_REPORT))
        != v2.EXPECTED_V1_REPORT_SHA256
    ):
        raise SystemExit("v2 did not lock the terminal v1 report")


def main() -> int:
    args = parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    validate_report(report)
    print(
        json.dumps(
            {
                "report": str(args.report),
                "classification": report["classification"],
                "passed": report["passed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
