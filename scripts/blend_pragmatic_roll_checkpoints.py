#!/usr/bin/env python3
"""Export static native roll-synapse interpolations for autonomous flight checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def blend_roll_states(source, target, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    if source.keys() != target.keys():
        raise ValueError("controller state layouts differ")
    for key in source.keys() - {"edge_magnitude"}:
        if not torch.equal(source[key], target[key]):
            raise ValueError(f"non-roll-synapse state differs: {key}")
    motors = source["pool_indices"][: int(source["pool_offsets"][2])]
    allowed = torch.isin(source["edge_post"], motors)
    if not torch.equal(source["edge_magnitude"][~allowed], target["edge_magnitude"][~allowed]):
        raise ValueError("non-roll incoming synapses differ")
    result = dict(source)
    result["edge_magnitude"] = torch.lerp(
        source["edge_magnitude"], target["edge_magnitude"], fraction
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    args = parser.parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    target = torch.load(args.target, map_location="cpu", weights_only=True)
    if source["graph_sha256"] != target["graph_sha256"]:
        raise ValueError("checkpoint graphs differ")
    outputs = [args.output_dir / f"blend-{fraction:g}.pt" for fraction in args.fractions]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite an existing blend")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fraction, path in zip(args.fractions, outputs, strict=True):
        payload = dict(source)
        payload.update(
            controller=blend_roll_states(source["controller"], target["controller"], fraction),
            experiment="static-native-roll-synapse-blend-v1",
            selection_metrics=None,
            blend_source=str(args.source),
            blend_target=str(args.target),
            blend_fraction=fraction,
            source_checkpoint=str(args.source),
            teacher_inputs_are_actor_inputs=False,
        )
        torch.save(payload, path)
        print(path, flush=True)


if __name__ == "__main__":
    main()
