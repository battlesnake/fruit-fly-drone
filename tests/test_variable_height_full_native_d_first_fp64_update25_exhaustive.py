from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_update25_exhaustive as audit25  # noqa: E402


def test_exhaustive_primary_selects_kkt_solution_deterministically() -> None:
    gram = np.eye(2, dtype=np.float64)
    violation = np.array([2.0, -1.0], dtype=np.float64)

    first, first_report = audit25.exhaustive_primary_dual(gram, violation)
    second, second_report = audit25.exhaustive_primary_dual(gram, violation)

    assert np.array_equal(first, np.array([2.0, 0.0]))
    assert np.array_equal(first, second)
    assert first_report["pass"] is True
    assert first_report["used_for_candidate"] is True
    assert first_report["selected_support_mask"] == second_report["selected_support_mask"]


def test_exhaustive_bound_aware_projection_solves_simple_feasible_problem() -> None:
    displacement = {
        name: torch.zeros(1, dtype=torch.float32) for name in audit25.joint.PARAMETER_FAMILIES
    }
    row = {name: torch.zeros(1, dtype=torch.float32) for name in audit25.joint.PARAMETER_FAMILIES}
    row["edge_magnitude"].fill_(1.0)
    specs = [{"name": "edge.maximum", "current_mse": 1.0, "limit_mse": 0.0}]

    projected, report = audit25.exhaustive_bound_aware_projection(
        displacement,
        [row],
        specs,
        torch.tensor([2.0]),
    )

    assert report["pass"] is True
    assert report["converged_without_bound_violation"] is True
    assert len(report["rounds"]) == 1
    assert report["rounds"][0]["free_projection"]["pass"] is True
    assert report["rounds"][0]["free_projection"]["slsqp_independent"]["pass"] is True
    assert projected["edge_magnitude"].item() <= -1.0 + 1.0e-6


def test_archive_hashes_cover_every_generated_tensor_tree() -> None:
    archive = {
        "current_parameters": {"edge_magnitude": torch.tensor([1.0])},
        "optimizer_before": {"state": {0: {"step": torch.tensor(24.0)}}},
        "optimizer_after": {"state": {0: {"step": torch.tensor(25.0)}}},
        "raw_displacement": {"edge_magnitude": torch.tensor([0.1])},
        "constraint_rows": [{"edge_magnitude": torch.tensor([0.2])}],
        "constraint_specs": [{"name": "test", "limit_mse": 1.0}],
        "raw_damping_gradient": {"edge_magnitude": torch.tensor([0.3])},
    }

    hashes = audit25.archive_semantic_hashes(archive)

    assert set(hashes) == {
        "current_parameters",
        "optimizer_before",
        "optimizer_after",
        "raw_displacement",
        "constraint_rows",
        "constraint_specs",
        "raw_damping_gradient",
    }
    changed = audit25.to_cpu_tree(archive)
    changed["raw_displacement"]["edge_magnitude"].add_(1.0)
    changed_hashes = audit25.archive_semantic_hashes(changed)
    assert hashes["current_parameters"] == changed_hashes["current_parameters"]
    assert hashes["raw_displacement"] != changed_hashes["raw_displacement"]


def test_round_path_signature_excludes_nonunique_dual_values() -> None:
    projection = {
        "rounds": [
            {
                "round": 1,
                "fixed_edges_before": 0,
                "fixed_set_sha256_before": "fixed",
                "newly_fixed_edges": 4,
                "newly_fixed_set_sha256": "new",
                "free_projection": {
                    "selected_support_mask": 3,
                    "dual_coefficients": [1.0, 2.0],
                },
            }
        ]
    }

    assert audit25.round_path_signature(projection) == [
        {
            "round": 1,
            "fixed_edges_before": 0,
            "fixed_set_sha256_before": "fixed",
            "newly_fixed_edges": 4,
            "newly_fixed_set_sha256": "new",
            "selected_support_mask": 3,
        }
    ]


def test_protocol_keeps_ordinary_result_out_of_numerical_pass() -> None:
    protocol = audit25.protocol_manifest()

    assert protocol["protocol_commit"] == "18a237b"
    assert protocol["primary_solver"]["used_for_candidate"] is True
    assert protocol["diagnostic_solvers"]["lbfgsb_status_is_diagnostic_not_required"] is True
    assert protocol["bound_aware_projection"]["maximum_rounds"] == 8
    assert protocol["ordinary_training_diagnostic"]["repair_invoked"] is False
    assert protocol["ordinary_training_diagnostic"]["part_of_numerical_pass"] is False
    assert protocol["candidate_or_optimizer_retained"] is False


def test_late_restoration_failure_cannot_receive_qualified_classification() -> None:
    classification = audit25.audit_classification(
        producer_identity_pass=True,
        archive_identity_pass=True,
        baseline_pass=True,
        all_runs_pass=True,
        repeat_agreement_pass=True,
        round_paths_exact=True,
        post_projection_pass=True,
        restoration_pass=False,
        archive_unchanged=True,
        sources_unchanged=True,
        ordinary_scale=0.5,
    )

    assert classification == "verifier_restoration_or_artifact_control_failure"


def test_post_projection_gate_requires_every_repeat() -> None:
    assert audit25.all_post_projection_runs_pass([{"pass": True}, {"pass": True}, {"pass": True}])
    assert not audit25.all_post_projection_runs_pass(
        [{"pass": True}, {"pass": False}, {"pass": True}]
    )
    assert not audit25.all_post_projection_runs_pass([{"pass": True}])


def test_producer_refuses_existing_outputs(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / audit25.PRODUCER_REPORT_NAME).write_text("{}", encoding="utf-8")
    source = tmp_path / "source"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(audit25, "locked_input_paths", lambda args: {"source": source})
    args = Namespace(phase="produce", smoke_test=False, output_dir=output)

    try:
        audit25.validate_args(args)
    except SystemExit as error:
        assert "cannot be retried" in str(error)
    else:
        raise AssertionError("producer unexpectedly accepted an existing output")


def test_interrupted_phase_finalizes_without_retry(tmp_path: Path) -> None:
    output = tmp_path / "output"
    args = Namespace(phase="produce", output_dir=output)
    audit25.write_phase_marker(args)

    assert audit25.finalize_interrupted(Namespace(output_dir=output)) == 0
    report = json.loads((output / audit25.FINAL_REPORT_NAME).read_text(encoding="utf-8"))
    assert report["pass"] is False
    assert report["classification"] == "producer_interrupted_after_phase_start"
    assert report["continuation_authorized"] is False
