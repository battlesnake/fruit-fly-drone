#!/usr/bin/env python3
"""Compile a saved ES trial into an ordinary native controller for standalone checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import search_gate_motor_interface_es as motor_es  # noqa: E402
from search_pragmatic_full_native_gate_es import apply_vector  # noqa: E402
from train_pragmatic_gate_visual_roll_path import load_controller  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing candidate checkpoint")
    report = json.loads(args.report.read_text())
    records = [entry for entry in report["history"] if entry["generation"] == args.generation]
    if len(records) != 1 or not 0 <= args.candidate < len(records[0]["candidate_vectors"]):
        raise SystemExit("generation or candidate is absent from the saved report")
    record = records[0]
    settings = report["arguments"]
    controller, source = load_controller(
        argparse.Namespace(
            graph=Path(settings["graph"]), checkpoint=Path(report["source_checkpoint"])
        ),
        torch.device("cpu"),
    )
    spec = motor_es.motor_interface_spec(
        controller,
        bias_scale=settings["bias_scale"],
        log_gain_scale=settings["gain_scale"],
        maximum_bias_delta=0.02,
        maximum_gain_ratio=2.0,
    )
    vector = torch.tensor(record["candidate_vectors"][args.candidate])
    if vector.shape != spec.scales.shape or not bool(torch.isfinite(vector).all()):
        raise SystemExit("invalid saved native parameter vector")
    apply_vector(
        controller, controller.bias.clone(), controller.edge_magnitude.clone(), vector, spec
    )
    payload = dict(source)
    payload.update(
        controller={name: value.detach().cpu() for name, value in controller.state_dict().items()},
        experiment="native-varied-five-gate-es-trial-not-promoted",
        native_search_vector=vector,
        native_search_labels=list(spec.labels),
        candidate_provenance=dict(
            report=str(args.report), generation=args.generation, candidate=args.candidate
        ),
        candidate_training_metrics=record["candidates"][args.candidate],
        selection_metrics=None,
        standalone_development_validated=False,
        course_geometry=report["geometry"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(dict(output=str(args.output), promoted=False)), flush=True)


if __name__ == "__main__":
    main()
