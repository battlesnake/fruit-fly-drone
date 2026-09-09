#!/usr/bin/env python3
"""Extract the compact MaleCNS graph used by the first hover experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.connectome_data import BuildConfig, build_hover_scaffold  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "malecns-v1.0"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "derived" / "hover-connectome-v1.npz",
    )
    parser.add_argument("--path-weight-threshold", type=int, default=20)
    parser.add_argument("--induced-weight-threshold", type=int, default=10)
    parser.add_argument("--max-path-hops", type=int, default=8)
    parser.add_argument("--visual-per-eye", type=int, default=24)
    parser.add_argument("--attitude-per-channel", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BuildConfig(
        path_weight_threshold=args.path_weight_threshold,
        induced_weight_threshold=args.induced_weight_threshold,
        max_path_hops=args.max_path_hops,
        visual_per_eye=args.visual_per_eye,
        attitude_per_channel=args.attitude_per_channel,
    )
    manifest = build_hover_scaffold(args.raw_dir, args.output, config)
    print(json.dumps(manifest["selection"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
