from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_pragmatic_course_coverage as coverage  # noqa: E402


def test_twelve_update_cycle_covers_every_stage_and_late_phase_for_both_sources():
    plans = [coverage.coverage_plan(update) for update in range(1, 13)]
    for kind in ("native", "roll-assisted"):
        early = [
            stage
            for plan in plans
            for source, phase, stage in plan
            if source == kind and phase == 0
        ]
        late = [
            phase for plan in plans for source, phase, _ in plan if source == kind and phase > 0
        ]
        assert early.count("launch") == early.count("early-approach") == 2
        assert early.count("middle") == early.count("late-approach") == 4
        assert all(late.count(phase) == 3 for phase in range(1, 5))
    assert all(len(plan) == 4 for plan in plans)


def test_every_fresh_round_retains_all_late_phases_and_early_stages():
    old = coverage.older_collection_updates(10)
    assert old == {4, 9}
    for round_index in range(3):
        fresh = [
            coverage.coverage_plan(round_index * 10 + local)
            for local in range(1, 11)
            if not round_index or local not in old
        ]
        assert {phase for plan in fresh for _, phase, _ in plan if phase} == {1, 2, 3, 4}
        assert {stage for plan in fresh for _, phase, stage in plan if not phase} == {
            "launch",
            "early-approach",
            "middle",
            "late-approach",
        }
    for count in range(1, 16):
        old = coverage.older_collection_updates(count)
        fresh_phases = {(step - 1) % 4 for step in range(1, count + 1) if step not in old}
        assert fresh_phases == {(step - 1) % 4 for step in range(1, count + 1)}


def test_better_total_and_balance_are_not_vetoed_by_one_side_declining():
    def metrics(negative, positive):
        return dict(
            clean_course_success_rate=(negative + positive) / 32,
            clean_course_negative_success_rate=negative / 16,
            clean_course_positive_success_rate=positive / 16,
            gates_before_failure_mean=1.0,
            course_race_fitness=1.0,
        )

    assert coverage.native_selection_score(metrics(7, 9)) > coverage.native_selection_score(
        metrics(9, 3)
    )
    assert coverage.native_selection_score(metrics(8, 8)) > coverage.native_selection_score(
        metrics(7, 9)
    )


def test_coverage_selection_reports_fallback_truthfully(monkeypatch):
    native, assisted = object(), object()
    calls = []

    def select(primary, fallback, phase, unroll, rng, kind, device, **kwargs):
        calls.append((primary, fallback, phase, kwargs["early_stage"]))
        return None, dict(source_by_side=["native", "roll-assisted"], substituted_sides=[1])

    monkeypatch.setattr(coverage.replay, "select_balanced_window", select)
    lessons = coverage.choose_lessons(
        {"native": native, "roll-assisted": assisted},
        coverage.coverage_plan(1),
        20,
        np.random.default_rng(0),
        torch.device("cpu"),
    )
    assert calls == [
        (native, assisted, 0, "launch"),
        (assisted, assisted, 0, "launch"),
        (native, assisted, 1, None),
        (assisted, assisted, 1, None),
    ]
    assert lessons[0][1]["requested_source"] == "native"
    assert lessons[0][1]["source_by_side"] == ["native", "roll-assisted"]


def test_batched_fit_diagnostics_keep_each_pair_and_actual_sources(monkeypatch):
    lessons = [
        (index, dict(phase=index + 1, source_by_side=["native", "roll-assisted"]))
        for index in range(5)
    ]
    monkeypatch.setattr(coverage.replay, "combine_replay_columns", lambda windows: windows)

    def loss(actor, windows, *args, **kwargs):
        rows = torch.tensor([float(2 * index + side + 1) for index in windows for side in (0, 1)])
        residual = rows[None, :, None].expand(3, -1, 4)
        return None, None, residual

    monkeypatch.setattr(coverage.replay, "replay_window_loss", loss)
    result = coverage.fit_summary(None, lessons, 3, None, None, 1.0)
    for index, record in enumerate(result["windows"]):
        assert record["phase"] == index + 1
        assert record["source_by_side"] == ["native", "roll-assisted"]
        assert record["motor_rmse_by_side"] == [
            [float(2 * index + 1)] * 4,
            [float(2 * index + 2)] * 4,
        ]


def test_bounded_runner_refreshes_latest_weights_keeps_source_and_scores_native(
    monkeypatch, tmp_path
):
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_magnitude = torch.nn.Parameter(torch.tensor([0.1]))

        def project_parameters(self):
            with torch.no_grad():
                self.edge_magnitude.clamp_(0, 8)

    actor = Actor()
    source = dict(
        hover_config=vars(coverage.replay.HoverConfig()),
        gate_config=vars(coverage.replay.GateConfig()),
        image_resolution=[320, 200],
        camera_hfov_degrees=125,
    )
    args = argparse.Namespace(
        checkpoint=tmp_path / "source.pt",
        output_dir=tmp_path / "run",
        graph=tmp_path / "graph",
        device="cpu",
        rounds=2,
        updates_per_round=8,
        unroll=2,
        native_pairs=2,
        assisted_pairs=1,
        learning_rate=0.01,
        contrast_weight=1,
        hop_budget=7,
        anchor_reference_count=19286,
        seed=2026091310,
        development_seed=1110983,
        development_pairs=16,
        seconds=30,
    )
    monkeypatch.setattr(coverage, "parse_args", lambda: args)
    monkeypatch.setattr(coverage.replay, "load_controller", lambda *a: (actor, source))
    monkeypatch.setattr(
        coverage.replay, "roll_preservation_mask", lambda *a, **k: (torch.tensor([True]), {})
    )
    monkeypatch.setattr(coverage.replay, "sample_two_gate_cases", lambda *a, **k: (None, None))
    evaluations, collections = [], []

    def evaluate(controller, *a, **kwargs):
        evaluations.append(kwargs)
        value = float(controller.edge_magnitude.detach()[0])
        return dict(
            clean_course_success_rate=value,
            clean_course_negative_success_rate=value,
            clean_course_positive_success_rate=value,
            gates_before_failure_mean=1,
            course_race_fitness=1,
            clean_first_gate_pass_rate=1,
            ground_contact_rate=0,
        )

    def collect(controller, pairs, seed, kind, *args, **kwargs):
        reference = args[-1]
        collections.append(
            (
                kind,
                seed,
                float(controller.edge_magnitude.detach()[0]),
                float(reference.edge_magnitude[0]),
                kwargs,
            )
        )

        class Bank:
            def manifest(self):
                return dict(kind=kind, seed=seed)

        return Bank()

    def loss(controller, *args, **kwargs):
        value = (controller.edge_magnitude - 1).square().mean()
        return value, value.detach().expand(4)

    monkeypatch.setattr(coverage.replay, "evaluate", evaluate)
    monkeypatch.setattr(coverage.replay, "collect_bank", collect)
    monkeypatch.setattr(coverage.replay, "replay_window_loss", loss)
    monkeypatch.setattr(coverage.replay, "combine_replay_columns", lambda windows: None)
    monkeypatch.setattr(coverage, "choose_lessons", lambda *a: [(None, {})] * 4)
    monkeypatch.setattr(coverage, "fit_summary", lambda *a: {"windows": []})
    assert coverage.main() == 0
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "complete" and len(report["history"]) == 16
    assert not report["autonomous_goal_established"]
    assert [row[0] for row in collections] == ["native", "roll-assisted"] * 2
    assert [row[1] for row in collections] == [2026091310, 2026091311, 2026091330, 2026091331]
    assert collections[0][2] == collections[1][2] < collections[2][2] == collections[3][2]
    assert all(abs(row[3] - 0.1) < 1e-7 for row in collections)
    assert all(
        row[4] == dict(roll_teacher="current-gate", roll_from_start=True) for row in collections
    )
    assert len(evaluations) == 3 and all(row["seconds"] == 30 for row in evaluations)
    assert all(row["warmup_steps"] == 10 for row in evaluations)
    assert report["history"][11]["collection_round"] == 1  # retained older exposures
    assert report["history"][-1]["collection_round"] == 2
    payload = torch.load(args.output_dir / "best-controller.pt", weights_only=True)
    assert payload["training_update"] == 16 and payload["diagnostic_only"]
    assert not payload["teacher_inputs_are_actor_inputs"]
