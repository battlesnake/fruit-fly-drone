from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_update21_snapshot as snapshot  # noqa: E402


def _tensor_snapshot() -> dict[str, object]:
    return {
        "candidate_parameters": {"edge_magnitude": torch.tensor([1.0, 2.0])},
        "pending_optimizer_state": {"state": {0: {"step": torch.tensor(21.0)}}},
        "repair_tensor_capture": {
            "solver_constraint_rows": [{"edge_magnitude": torch.tensor([0.1, 0.2])}],
            "authoritative_endpoint_damping_row": {"edge_magnitude": torch.tensor([0.3, 0.4])},
            "full_correction_direction": {"edge_magnitude": torch.tensor([0.5, 0.6])},
            "starting_candidate_parameters": {"edge_magnitude": torch.tensor([1.1, 2.1])},
            "full_correction_parameters": {"edge_magnitude": torch.tensor([1.2, 2.2])},
        },
    }


def test_snapshot_semantic_hashes_detect_each_tensor_tree() -> None:
    value = _tensor_snapshot()
    hashes = snapshot.snapshot_semantic_hashes(value)

    assert set(hashes) == {
        "candidate_parameters",
        "pending_optimizer_state",
        "solver_constraint_rows",
        "authoritative_endpoint_damping_row",
        "full_correction_direction",
        "starting_candidate_parameters",
        "full_correction_parameters",
        "repair_tensor_capture",
    }
    changed = _tensor_snapshot()
    changed["repair_tensor_capture"]["full_correction_direction"]["edge_magnitude"][0] = 0.7
    changed_hashes = snapshot.snapshot_semantic_hashes(changed)
    assert hashes["candidate_parameters"] == changed_hashes["candidate_parameters"]
    assert hashes["full_correction_direction"] != changed_hashes["full_correction_direction"]
    assert hashes["repair_tensor_capture"] != changed_hashes["repair_tensor_capture"]


def test_to_cpu_tree_clones_tensors() -> None:
    source = {"items": [torch.tensor([1.0], requires_grad=True)]}
    copied = snapshot.to_cpu_tree(source)
    source["items"][0].data.add_(1.0)

    assert copied["items"][0].device.type == "cpu"
    assert copied["items"][0].requires_grad is False
    assert copied["items"][0].item() == 1.0


def test_development_gate_requires_repaired_selection() -> None:
    controls = {"pass": True, "repair_path_present": True}
    capture = {"full_correction_direction": torch.tensor([1.0])}
    ordinary = [{"acceptance_kind": "ordinary", "decision": {"pass": True}}]
    repaired = [{"acceptance_kind": "repaired", "decision": {"pass": True}}]

    assert not snapshot.repaired_selection_controls_pass(
        snapshot.corrected.REPAIR_PROPOSAL_SCALE,
        ordinary,
        controls,
        capture,
        snapshot.EXPECTED_PENDING_OPTIMIZER_SHA256,
    )
    assert snapshot.repaired_selection_controls_pass(
        snapshot.corrected.REPAIR_PROPOSAL_SCALE,
        repaired,
        controls,
        capture,
        snapshot.EXPECTED_PENDING_OPTIMIZER_SHA256,
    )


def test_verifier_replay_gate_requires_baseline_reproduction() -> None:
    assert snapshot.verifier_replay_preconditions_pass(
        producer_identity_pass=True,
        snapshot_identity_pass=True,
        semantic_pass=True,
        baseline_reproductions={"update20": {"pass": True}},
    )
    assert not snapshot.verifier_replay_preconditions_pass(
        producer_identity_pass=True,
        snapshot_identity_pass=True,
        semantic_pass=True,
        baseline_reproductions={"update20": {"pass": False}},
    )


def test_producer_refuses_existing_outputs(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / snapshot.PRODUCER_REPORT_NAME).write_text("{}", encoding="utf-8")
    args = Namespace(
        phase="produce",
        smoke_test=False,
        output_dir=output,
    )
    paths = {"input": tmp_path / "input"}
    paths["input"].write_text("input", encoding="utf-8")

    original = snapshot.locked_input_paths
    snapshot.locked_input_paths = lambda _args: paths
    try:
        try:
            snapshot.validate_args(args)
        except SystemExit as error:
            assert "cannot be retried" in str(error)
        else:
            raise AssertionError("producer unexpectedly allowed an existing output")
    finally:
        snapshot.locked_input_paths = original


def test_verifier_requires_producer_report(tmp_path: Path) -> None:
    args = Namespace(
        phase="verify",
        smoke_test=False,
        output_dir=tmp_path / "output",
    )
    paths = {"input": tmp_path / "input"}
    paths["input"].write_text("input", encoding="utf-8")

    original = snapshot.locked_input_paths
    snapshot.locked_input_paths = lambda _args: paths
    try:
        try:
            snapshot.validate_args(args)
        except SystemExit as error:
            assert "requires a completed producer report" in str(error)
        else:
            raise AssertionError("verifier unexpectedly allowed a missing producer report")
    finally:
        snapshot.locked_input_paths = original


def test_phase_marker_is_exclusive_and_interruption_finalizes(tmp_path: Path) -> None:
    output = tmp_path / "output"
    args = Namespace(phase="produce", output_dir=output)
    marker = snapshot.write_phase_marker(args)

    try:
        snapshot.write_phase_marker(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("phase marker was not exclusive")

    finalize_args = Namespace(output_dir=output)
    assert snapshot.finalize_interrupted(finalize_args) == 0
    report = json.loads((output / snapshot.FINAL_REPORT_NAME).read_text(encoding="utf-8"))
    assert marker.is_file()
    assert report["pass"] is False
    assert report["classification"] == "snapshot_producer_interrupted_after_phase_start"
