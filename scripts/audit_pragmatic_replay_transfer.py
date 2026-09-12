#!/usr/bin/env python3
"""New-course, fixed-input transfer check for a successfully fitted native actor.

This is not autonomous-flight validation. A source-native and source-roll-assisted
bank are each collected once, without coverage-driven resampling. Source and candidate
replay the same physical histories continuously; only scoring is masked by eligibility.
No recorded history, teacher or diagnostic state is deployed in the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_pragmatic_course_replay as replay  # noqa: E402


def phase_errors(residual, bank):
    """Eligibility uses source collection only, never the candidate's errors."""
    if not bool(torch.isfinite(residual[bank.active]).all()):
        raise ValueError("nonfinite residual on a scored source-history frame")
    rows = torch.arange(bank.current.shape[1])
    records = []
    for phase in range(5):
        for side in (0, 1):
            mask = bank.active & (bank.current == phase) & ((rows % 2) == side)[None]
            frames = int(mask.sum())
            episodes = int(mask.any(0).sum())
            records.append(
                dict(
                    phase=phase + 1,
                    side=side,
                    frames=frames,
                    episodes=episodes,
                    eligible=frames >= 20 and episodes >= 2,
                    motor_rmse=residual[mask].square().mean(0).sqrt().tolist() if frames else None,
                )
            )
    return records


@torch.no_grad()
def compare_bank(source, candidate, bank, camera, gate_config):
    """Full continuous neural replay, including unscored prefixes and inactive tails."""
    device = source.bias.device
    count = bank.current.shape[1]
    window = replay.prepare_window(
        bank, tuple(range(count)), (0,) * count, len(bank.current), device
    )
    times = torch.zeros(count, dtype=torch.long, device=device)
    image, attitude = replay.window_observations(window, times, camera, gate_config)
    controllers = (source, candidate)
    states = [
        actor.initial_state(count, device=device, dtype=torch.float32) for actor in controllers
    ]
    for _ in range(10):
        for index, actor in enumerate(controllers):
            _, states[index] = actor(image, attitude, states[index])
    outputs = [[], []]
    for frame in range(len(bank.current)):
        image, attitude = replay.window_observations(
            window, times.fill_(frame), camera, gate_config
        )
        for index, actor in enumerate(controllers):
            prediction, states[index] = actor(image, attitude, states[index])
            outputs[index].append(prediction)
    statistics = [phase_errors(torch.stack(output).cpu() - bank.target, bank) for output in outputs]
    return dict(
        kind=bank.kind,
        seed=bank.seed,
        collection=bank.manifest(),
        source=statistics[0],
        candidate=statistics[1],
    )


def summarize_transfer(records):
    """Average eligible bank/phase MSE equally, with identical source/candidate weights."""
    if sorted(record["kind"] for record in records) != ["native", "roll-assisted"]:
        raise ValueError("exactly one native and one assisted bank are required")
    groups = []
    for bank in records:
        for source, candidate in zip(bank["source"], bank["candidate"], strict=True):
            keys = ("phase", "side", "frames", "episodes", "eligible")
            if any(source[key] != candidate[key] for key in keys):
                raise ValueError("source and candidate must use identical scoring masks")
            if source["phase"] > 1 and source["eligible"]:
                values = source["motor_rmse"] + candidate["motor_rmse"]
                if not all(math.isfinite(value) and value >= 0 for value in values):
                    raise ValueError("nonfinite or negative transfer errors")
                groups.append(
                    dict(
                        kind=bank["kind"],
                        phase=source["phase"],
                        side=source["side"],
                        frames=source["frames"],
                        episodes=source["episodes"],
                        source=source["motor_rmse"][0],
                        candidate=candidate["motor_rmse"][0],
                    )
                )

    def aggregate(kind=None):
        by_side = []
        for side in (0, 1):
            selected = [
                g for g in groups if g["side"] == side and (kind is None or g["kind"] == kind)
            ]
            if not selected:
                by_side.append(
                    dict(groups=0, source_rmse=None, candidate_rmse=None, reduction=None)
                )
                continue
            source = math.sqrt(sum(g["source"] ** 2 for g in selected) / len(selected))
            candidate = math.sqrt(sum(g["candidate"] ** 2 for g in selected) / len(selected))
            by_side.append(
                dict(
                    groups=len(selected),
                    source_rmse=source,
                    candidate_rmse=candidate,
                    reduction=1 - candidate / source if source > 0 else None,
                )
            )
        return by_side

    phase_coverage = [
        [any(g["phase"] == phase and g["side"] == side for g in groups) for side in (0, 1)]
        for phase in range(2, 6)
    ]
    kind_coverage = [
        [any(g["kind"] == kind and g["side"] == side for g in groups) for side in (0, 1)]
        for kind in ("native", "roll-assisted")
    ]
    total, native, assisted = aggregate(), aggregate("native"), aggregate("roll-assisted")
    coverage = all(all(row) for row in phase_coverage + kind_coverage)
    aggregate_pass = coverage and all(
        g["reduction"] is not None and g["reduction"] >= 0.2 for g in total
    )
    native_nonworsening = all(g["reduction"] is not None and g["reduction"] >= 0 for g in native)
    return dict(
        eligible_groups=groups,
        phase_side_coverage=phase_coverage,
        kind_side_coverage=kind_coverage,
        coverage_complete=coverage,
        combined=total,
        native_only=native,
        assisted_only=assisted,
        aggregate_twenty_percent_screen=aggregate_pass,
        native_bank_nonworsening=native_nonworsening,
        assisted_state_only_improvement=aggregate_pass and not native_nonworsening,
        permits_expanded_native_validation=aggregate_pass and native_nonworsening,
        autonomous_success_established=False,
    )


def nominated_checkpoint(fit_dir, hops, update):
    report = json.loads((fit_dir / "report.json").read_text())
    if report.get("roll_labels", "late-preserve") != "late-preserve":
        raise ValueError("this transfer audit uses late-preserve labels, not unified labels")
    entries = [
        entry
        for arm in report["arms"]
        if arm["hops"] == hops
        for entry in arm["history"]
        if entry["update"] == update and "fit" in entry
    ]
    if len(entries) != 1:
        raise ValueError("nominate one evaluated fitting checkpoint")
    entry = entries[0]
    reductions = entry["late_roll_reduction_by_side"]
    if len(reductions) != 2 or not all(
        math.isfinite(value) and value >= 0.5 for value in reductions
    ):
        raise ValueError("candidate has not met the two-sided fifty-percent fitting screen")
    return report, fit_dir / entry["checkpoint"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--hop-budget", type=int, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("refusing to overwrite or recollect an existing transfer audit")
    fitting, candidate_path = nominated_checkpoint(args.fit_dir, args.hop_budget, args.update)
    source_args = argparse.Namespace(
        checkpoint=Path(fitting["arguments"]["checkpoint"]),
        graph=Path(fitting["arguments"]["graph"]),
    )
    if hashlib.sha256(source_args.checkpoint.read_bytes()).hexdigest() != fitting["source_sha256"]:
        raise SystemExit("the retained source changed since fitting")
    device = torch.device(args.device)
    source, payload = replay.load_controller(source_args, device)
    candidate, candidate_payload = replay.load_controller(
        argparse.Namespace(checkpoint=candidate_path, graph=source_args.graph), device
    )
    for key in ("hover_config", "gate_config", "image_resolution", "camera_hfov_degrees"):
        if payload[key] != candidate_payload[key]:
            raise SystemExit(f"candidate interface or plant differs: {key}")
    source.eval().requires_grad_(False)
    candidate.eval().requires_grad_(False)
    config = replay.HoverConfig(**payload["hover_config"])
    gate_config = replace(replay.GateConfig(**payload["gate_config"]), back_pattern="checkerboard")
    camera = replay.CameraSpec(*payload["image_resolution"], payload["camera_hfov_degrees"])
    args.output_dir.mkdir(parents=True)
    result = dict(
        experiment="fixed-input-new-course-native-transfer-v1",
        candidate=str(candidate_path),
        source=str(source_args.checkpoint),
        geometry=replay.GEOMETRY,
        scope="new source-generated physical histories, not autonomous candidate flights",
        collection_pairs_per_bank=8,
        resampling_for_coverage=False,
        eligibility=dict(minimum_active_frames=20, minimum_episodes=2),
        banks=[],
    )

    def save():
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for kind, seed in (("native", 2026091303), ("roll-assisted", 2026091304)):
        bank = replay.collect_bank(
            source,
            8,
            seed,
            kind,
            30.0,
            camera,
            config,
            gate_config,
            "rate-damped",
            source,
            roll_teacher="current-gate",
        )
        torch.save(
            dict(
                states=bank.states,
                gates=[(g.center, g.yaw) for g in bank.gates],
                current=bank.current,
                active=bank.active,
                target=bank.target,
                reference_outputs=bank.reference_outputs,
                failure_steps=bank.failure_steps,
                kind=bank.kind,
                seed=bank.seed,
                roll_teacher=bank.roll_teacher,
            ),
            args.output_dir / f"{kind}-history.pt",
        )
        result["banks"].append(compare_bank(source, candidate, bank, camera, gate_config))
        save()
        print(json.dumps(dict(stage="bank-complete", kind=kind, seed=seed)), flush=True)
    result["summary"] = summarize_transfer(result["banks"])
    result["status"] = "complete"
    save()
    print(json.dumps(result["summary"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
