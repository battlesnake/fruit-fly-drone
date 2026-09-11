from __future__ import annotations

import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_frozen_optic_motion_deterministic as audit  # noqa: E402
import frozen_optic_motion_deterministic as deterministic  # noqa: E402


def _fake_evaluation() -> dict:
    evaluation = {
        name: torch.zeros(1, 2, 3)
        for name in deterministic.RESPONSE_TENSOR_NAMES
    }
    evaluation["terminal_visual_target_type_means"] = {
        "H1": torch.zeros(1, 2)
    }
    evaluation.update(
        {
            "method": "exponential_euler",
            "steps_per_camera_frame": 32,
            "wall_time_seconds": 1.0,
        }
    )
    return evaluation


def test_protocol_retains_v1_science_and_adds_four_process_gate() -> None:
    protocol = deterministic.protocol_manifest()

    assert protocol["protocol_commit"] == "800c762"
    assert protocol["scientific_changes_from_v1"] == []
    assert protocol["fresh_process_replays"] == 3
    assert protocol["main_process_must_match_replays"] is True
    assert protocol["runtime"]["block_cases"] == 8
    assert protocol["training_execution_authorized"] is False


def test_response_hash_covers_nested_cpu_tensor_tree() -> None:
    first = _fake_evaluation()
    second = _fake_evaluation()

    assert deterministic.response_tensor_sha256(first) == (
        deterministic.response_tensor_sha256(second)
    )
    second["terminal_visual_target_type_means"]["H1"][0, 0] = 1.0
    assert deterministic.response_tensor_sha256(first) != (
        deterministic.response_tensor_sha256(second)
    )


def test_stable_metadata_excludes_only_runtime_duration() -> None:
    evaluation = _fake_evaluation()

    metadata = deterministic.stable_evaluation_metadata(evaluation)
    assert metadata == {
        "method": "exponential_euler",
        "steps_per_camera_frame": 32,
    }


def test_main_process_duplicate_gate_requires_exact_hash() -> None:
    gate = {
        "response_tensor_sha256": "abc",
        "evaluation_metadata_semantic_sha256": "meta",
        "runtime_semantic_sha256": "runtime",
        "replay_reports": [
            {"process_identity": {"pid": index, "nonce": str(index) * 32}}
            for index in range(1, 4)
        ],
    }
    main_identity = {"pid": 4, "nonce": "4" * 32}

    assert audit.deterministic_duplicate_control(
        "abc", "meta", "runtime", main_identity, gate
    )["pass"] is True
    failed = audit.deterministic_duplicate_control(
        "def", "meta", "runtime", main_identity, gate
    )
    assert failed["pass"] is False
    assert failed["numerical_noise_maximum_absolute"] is None


def test_phase_paths_are_distinct_and_sidecar_only() -> None:
    output = Path("runs/optic-motion/example/replay-1.json")

    claim, terminal = deterministic.phase_paths(output)
    assert claim.name == "replay-1.claim.json"
    assert terminal.name == "replay-1.terminal.json"
