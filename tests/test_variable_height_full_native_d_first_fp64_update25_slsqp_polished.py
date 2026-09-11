from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_variable_height_full_native_d_first_fp64_update25_slsqp_polished as polished  # noqa: E402


def test_slsqp_support_uses_only_registered_relative_threshold() -> None:
    dual = np.array([1.0e-11, 2.0, 3.0e-10], dtype=np.float64)

    support, threshold = polished.slsqp_support(dual)

    assert threshold == 2.0e-10
    assert np.array_equal(support, np.array([1, 2]))


def test_support_polish_solves_slsqp_selected_full_rank_system() -> None:
    gram = np.eye(2, dtype=np.float64)
    violation = np.array([2.0, -1.0], dtype=np.float64)
    slsqp = np.array([1.9, 0.0], dtype=np.float64)

    reference, report = polished.polish_slsqp_support(gram, violation, slsqp)

    assert np.array_equal(reference, np.array([2.0, 0.0]))
    assert report["pass"] is True
    assert report["support_indices"] == [0]
    assert report["full_support_rank"] is True
    assert report["primary_support_consulted"] is False
    assert report["solve_count"] == 1
    assert report["iterative_refinement"] is False
    assert report["fallback"] is False


def test_support_polish_fails_rank_deficient_selected_support() -> None:
    gram = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    violation = np.array([1.0, 1.0], dtype=np.float64)
    slsqp = np.array([0.5, 0.5], dtype=np.float64)

    _, report = polished.polish_slsqp_support(gram, violation, slsqp)

    assert report["pass"] is False
    assert report["support_indices"] == [0, 1]
    assert report["full_support_rank"] is False


def test_protocol_has_no_solver_or_candidate_fallback() -> None:
    protocol = polished.protocol_manifest()

    assert protocol["protocol_commit"] == "3340c27"
    assert protocol["support_polish"]["support_source"] == "SLSQP coefficients only"
    assert protocol["support_polish"]["primary_support_consulted"] is False
    assert protocol["support_polish"]["fallback"] is False
    assert protocol["active_set"]["maximum_rounds"] == 8
    assert protocol["actor_instantiated_or_candidate_materialized"] is False
    assert protocol["failure_action"] == "pause this exhaustive-primary integration route"


def test_verifier_refuses_retry_after_started_marker(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.write_text("source", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / polished.STARTED_NAME).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(polished, "locked_input_paths", lambda args: {"source": source})
    args = Namespace(
        smoke_test=False,
        finalize_interrupted=False,
        output_dir=output,
    )

    try:
        polished.validate_args(args)
    except SystemExit as error:
        assert "cannot be retried" in str(error)
    else:
        raise AssertionError("started verifier was allowed to retry")
