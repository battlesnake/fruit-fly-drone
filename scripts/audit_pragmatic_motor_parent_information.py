#!/usr/bin/env python3
"""Read-only information probe on cached native motor-parent neural activity.

Privileged velocity/bearing are diagnostic targets only. No probe, inferred feature,
or teacher is deployed in the fly actor. Recorded trajectories are not a fresh flight
success evaluation, and predictive correlations do not establish causal motion coding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_pragmatic_roll_motor_slice import pair_split_weights  # noqa: E402

from flydrone.hover import rotation_matrix  # noqa: E402


@torch.no_grad()
def audit(cache, heldout_pairs=2):
    parents, attitudes, targets, training, validation = [], [], [], [], []
    for bank in cache["banks"]:
        if not 0 < heldout_pairs < bank["current"].shape[1] // 2:
            raise ValueError("need whole mirrored pairs in both training and validation")
        active = bank["active"]
        position, velocity, euler = [value[active] for value in bank["states"][:3]]
        rotation = rotation_matrix(euler)
        body_velocity = torch.einsum("bij,bi->bj", rotation, velocity)
        centers = torch.stack([gate[0] for gate in bank["gates"]])
        rows = torch.arange(active.shape[1])[None].expand_as(active)[active]
        delta = centers[bank["current"][active], rows] - position
        body_delta = torch.einsum("bij,bi->bj", rotation, delta)
        bearing = torch.atan2(body_delta[:, 1], body_delta[:, 0])
        targets.append(torch.stack((bank["target"][active], body_velocity[:, 1], bearing), dim=1))
        parents.append(bank["features"][active])
        attitudes.append(torch.cat((euler[:, :2].sin(), euler[:, :2].cos()), dim=1))
        training.append(pair_split_weights(bank, heldout_pairs, validation=False)[active])
        validation.append(pair_split_weights(bank, heldout_pairs, validation=True)[active])
    y, w, v = map(torch.cat, (targets, training, validation))
    w, v = w / w.sum(), v / v.sum()
    if not all(bool(torch.isfinite(x).all()) for x in (y, w, v, *parents, *attitudes)):
        raise ValueError("nonfinite active cached probe values")
    centered_y = y - (y * w[:, None]).sum(0)
    baseline = (centered_y.square() * v[:, None]).sum(0)
    records = []
    for name, x in (
        ("current_roll_pitch_only", torch.cat(attitudes)),
        ("native_motor_parents", torch.cat(parents)),
    ):
        mean = (x * w[:, None]).sum(0)
        scale = ((x - mean).square() * w[:, None]).sum(0).sqrt().clamp_min(1e-4)
        x = (x - mean) / scale
        covariance, cross = x.T @ (x * w[:, None]), x.T @ (centered_y * w[:, None])
        for ridge in (0.001, 0.01, 0.1, 1.0):
            beta = torch.linalg.solve(covariance + ridge * torch.eye(x.shape[1]), cross)
            error = ((x @ beta - centered_y).square() * v[:, None]).sum(0)
            records.append(
                dict(
                    features=name,
                    ridge=ridge,
                    heldout_rmse=error.sqrt().tolist(),
                    heldout_r2_against_training_mean=(1 - error / baseline).tolist(),
                )
            )
    return dict(
        experiment="native-motor-parent-information-probe-v1",
        source_sha256=cache["source_sha256"],
        seeds=[bank["seed"] for bank in cache["banks"]],
        heldout_pairs_per_bank=heldout_pairs,
        active_records=len(y),
        targets=["hybrid_roll_motor", "body_lateral_velocity_m_s", "current_gate_bearing_rad"],
        training_mean_baseline_heldout_rmse=baseline.sqrt().tolist(),
        records=records,
        probe_or_privileged_features_deployed=False,
        causal_visual_motion_or_flight_success_established=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing diagnostic report")
    torch.set_num_threads(4)
    result = audit(torch.load(args.cache, map_location="cpu", weights_only=True))
    result["cache"] = str(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
