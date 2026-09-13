from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_pragmatic_motor_search_scale as screen  # noqa: E402
from audit_pragmatic_motor_search_scale import (  # noqa: E402
    meaningful_gain,
    nominate_once,
    scale_screen_vectors,
)


def result(clean, *, negative=None, ground=0, invalid=0):
    negative = clean if negative is None else negative
    return dict(episodes=32, clean_course_success_rate=clean / 32,
                clean_course_negative_success_rate=negative / 16,
                clean_course_positive_success_rate=(clean - negative) / 16,
                ground_contact_rate=ground / 32, invalid_rate=invalid / 32,
                ground_or_invalid_rate=max(ground, invalid) / 32,
                first_gate_pass_rate=1, gates_before_failure_mean=3, course_race_fitness=3)


def test_scales_share_exact_directions_and_remain_source_centred():
    spec = SimpleNamespace(scales=torch.tensor([0.0025, 0.05]),
                           lower=torch.tensor([-0.02, -0.69]), upper=torch.tensor([0.02, 0.69]))
    vectors, labels = scale_screen_vectors(spec, 17)
    assert vectors.shape == (16, 2)
    assert torch.equal(vectors[::2], -vectors[1::2])
    assert torch.allclose(vectors[8:], 2.5 * vectors[:8])
    assert [label["direction"] for label in labels[:8]] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [label["scale"] for label in labels] == [0.1] * 8 + [0.25] * 8
    assert torch.equal(vectors, scale_screen_vectors(spec, 17)[0])


@pytest.mark.parametrize("clean,ground,invalid,eligible", [
    (6, 0, 0, False), (7, 0, 0, True), (8, 1, 0, False), (8, 0, 1, False),
])
def test_four_extra_clean_screen_rejects_unsafe_candidates(clean, ground, invalid, eligible):
    assert meaningful_gain(result(clean, ground=ground, invalid=invalid), result(3)) == eligible


def test_nomination_is_completion_first_with_balance_only_as_tiebreaker():
    baseline = result(3, negative=2)
    assert nominate_once([result(6), result(5)], baseline) is None
    # A better total is allowed even if the formerly successful side loses one case.
    assert nominate_once([result(7, negative=4), result(8, negative=1)], baseline) == 1
    assert nominate_once([result(8, negative=8), result(8, negative=4)], baseline) == 1


@pytest.mark.parametrize("outcome,standalone_count", [
    ("no-nominee", 1), ("confirmation-fails", 2), ("development-fails", 4), ("passes", 4),
])
def test_main_never_tries_a_runner_up_or_saves_an_unconfirmed_candidate(
    monkeypatch, tmp_path, outcome, standalone_count,
):
    class Controller(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.edge_magnitude = torch.nn.Parameter(torch.ones(2))
    controller = Controller()
    payload = dict(hover_config={}, gate_config={}, image_resolution=[32, 20],
                   camera_hfov_degrees=125)
    spec = SimpleNamespace(scales=torch.tensor([0.0025, 0.05]),
                           lower=torch.tensor([-0.02, -0.69]),
                           upper=torch.tensor([0.02, 0.69]), labels=["bias", "gain"])
    monkeypatch.setattr(screen, "load_controller", lambda *a: (controller, payload))
    monkeypatch.setattr(screen.motor_es, "motor_interface_spec", lambda *a, **kw: spec)
    monkeypatch.setattr(screen, "apply_vector", lambda *a: None)
    monkeypatch.setattr(screen, "ParameterBatchController", lambda *a, **kw: "batch")
    monkeypatch.setattr(screen, "repeat_course_bank", lambda bank, count: bank)
    seeds, standalone, batches = [], [], []
    def bank(*args, seed, **kwargs):
        seeds.append(seed)
        return None, None
    def evaluate(actor, *args, **kwargs):
        if actor == "batch":
            index = len(batches)
            batches.append(index)
            if index == 0 and outcome != "no-nominee":
                return [result(8), result(7)]  # Both eligible, only the first may be confirmed.
            return [result(3), result(3)]
        index = len(standalone)
        standalone.append(index)
        counts = [3, 6 if outcome == "confirmation-fails" else 8,
                  11, 16 if outcome == "passes" else 14]
        return result(counts[index])
    monkeypatch.setattr(screen, "sample_two_gate_cases", bank)
    monkeypatch.setattr(screen, "evaluate", evaluate)
    destination = tmp_path / "screen"
    monkeypatch.setattr(sys, "argv", ["screen", "--checkpoint", "unused.pt", "--device", "cpu",
                                     "--output-dir", str(destination)])
    assert screen.main() == 0
    report = json.loads((destination / "report.json").read_text())
    assert report["status"] == "complete"
    assert len(batches) == 8 and len(standalone) == standalone_count
    assert len(seeds) == (2 if standalone_count == 4 else 1)
    assert (destination / "candidate-controller.pt").exists() == (outcome == "passes")
    assert report["followup_authorized"] == (outcome == "passes")
    with pytest.raises(SystemExit, match="overwrite"):
        screen.main()
