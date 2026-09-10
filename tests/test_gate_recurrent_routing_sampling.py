from __future__ import annotations

import numpy as np

from scripts.audit_gate_recurrent_routing_sampling import bootstrap_gap, recover_pair_exposures
from scripts.train_gate_recurrent_routing import balanced_pair_split


def test_recovered_pair_exposures_are_balanced_and_deterministic() -> None:
    counts, batches = recover_pair_exposures(updates=100, minibatch_pairs=8, seed=1_047_131)
    fit, validation = balanced_pair_split()
    assert len(batches) == 100
    assert counts.sum() == 800
    assert counts[fit[:24]].sum() == counts[fit[24:]].sum() == 400
    assert counts[validation].sum() == 0
    assert batches[0] == [39, 3, 35, 17, 23, 48, 5, 38]


def test_pair_bootstrap_detects_a_clear_positive_gap() -> None:
    training = np.linspace(0.10, 0.20, 48)
    validation = np.linspace(0.30, 0.40, 16)
    result = bootstrap_gap(training, validation, replicates=2000, seed=11)
    assert result["observed_validation_minus_training_normalized_mse"] > 0.19
    assert result["confidence_95"][0] > 0.0
