from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from search_pragmatic_gate_course_es import centered_ranks, selection_score  # noqa: E402


def test_identical_fitness_does_not_create_an_es_direction():
    assert np.array_equal(centered_ranks(np.ones(8)), np.zeros(8))
    assert np.array_equal(centered_ranks(np.ones(1)), np.zeros(1))


def test_rank_ties_share_their_average_rank():
    assert np.allclose(centered_ranks(np.array([1.0, 4.0, 4.0, 8.0])), [-0.5, 0, 0, 0.5])


def test_course_selection_requires_clean_completion_and_retains_first_gate():
    baseline = dict(
        first_gate_pass_rate=0.8,
        clean_course_success_rate=0.1,
        clean_course_negative_success_rate=0.1,
        clean_course_positive_success_rate=0.1,
        gates_before_failure_mean=2.0,
        course_race_fitness=1.0,
    )
    improved = {**baseline, "clean_course_success_rate": 0.2}
    regressed = {**improved, "first_gate_pass_rate": 0.5}
    assert selection_score(improved, 0.75) > selection_score(baseline, 0.75)
    # Full-course improvement is the goal; a first-gate floor cannot veto it.
    assert selection_score(regressed, 0.75) > selection_score(baseline, 0.75)
    assert selection_score(regressed, 0.75) < selection_score(improved, 0.75)
