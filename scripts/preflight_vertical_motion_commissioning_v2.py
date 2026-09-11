#!/usr/bin/env python3
"""Run the preregistered cancellation-resistant commissioning preflight v2."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_vertical_motion_commissioning as v1  # noqa: E402
import vertical_motion_commissioning as shared  # noqa: E402
import vertical_motion_commissioning_v2 as corrected  # noqa: E402

EXPERIMENT = "vertical-t4t5-local-commissioning-preflight-v2"
PROTOCOL_COMMIT = "552f23d"
EXPECTED_V1_REPORT_SHA256 = "661868629adae98e1ee664a4ec4c4e50607137097a4e4dcdb07164510917f8e4"
EXPECTED_V1_NORMALIZATIONS_SHA256 = (
    "796cf8f0488edff30dd34bf556268b4faac6ad5564244a1ce9b13cfcb131e475"
)
EXPECTED_RENDERED_SUBSET_SHA256 = "c04bb10318cd3b0620c9a71abf7a7420afe166a0eff2cf1d1d154d9a41440576"
V1_REPORT = REPO_ROOT / "runs/optic-motion/vertical-motion-preflight-001/report.json"
FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT = v1.FINITE_DIFFERENCE_RELATIVE_ERROR_LIMIT
MINIMUM_STEP_IMPROVEMENT_FRACTION = v1.MINIMUM_STEP_IMPROVEMENT_FRACTION
_V1_PROTOCOL_MANIFEST = v1.protocol_manifest


def protocol_manifest() -> dict[str, Any]:
    protocol = _V1_PROTOCOL_MANIFEST()
    protocol.update(
        {
            "experiment": EXPERIMENT,
            "protocol_commit": PROTOCOL_COMMIT,
            "locked_v1_report_sha256": EXPECTED_V1_REPORT_SHA256,
            "locked_v1_classification": "vertical_motion_preflight_derivative_failed",
            "numerical_correction": {
                "scope": "selected T4c/T4d/T5c/T5d physical-tau state update only",
                "v1": ("source_next_state + (candidate_alpha-source_alpha) * (target-state)"),
                "v2": "state + candidate_alpha * (target-state)",
                "installation": "unique sorted selected-target index_copy",
                "mathematically_equivalent_before_float32_rounding": True,
            },
            "scientific_or_threshold_changes_from_v1": [],
            "v1_frozen_source_normalizations_sha256_required": (EXPECTED_V1_NORMALIZATIONS_SHA256),
            "v1_rendered_training_subset_sha256_required": (EXPECTED_RENDERED_SUBSET_SHA256),
        }
    )
    return protocol


def _validate_inputs(args):
    observed = _ORIGINAL_VALIDATE_INPUTS(args)
    if not V1_REPORT.is_file() or v1.registration.file_sha256(V1_REPORT) != (
        EXPECTED_V1_REPORT_SHA256
    ):
        raise SystemExit("locked v1 vertical-motion preflight report changed")
    observed[v1.registration.stable_path(V1_REPORT)] = EXPECTED_V1_REPORT_SHA256
    return observed


_ORIGINAL_VALIDATE_INPUTS = v1.validate_inputs


def main() -> int:
    # Reuse the locked v1 orchestration while replacing only the explicitly registered
    # experiment identity, input lock, and local controller class. __file__ values are
    # redirected so the terminal report hashes the actual v2 implementation files.
    v1.EXPERIMENT = EXPERIMENT
    v1.PROTOCOL_COMMIT = PROTOCOL_COMMIT
    v1.protocol_manifest = protocol_manifest
    v1.validate_inputs = _validate_inputs
    v1.__file__ = __file__
    shared.CommissionedController = corrected.CommissionedController
    shared.__file__ = corrected.__file__
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
