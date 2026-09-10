#!/usr/bin/env python3
"""Build the full MaleCNS graph and fixed 320x200 RGB retinal interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flydrone.full_connectome_data import (  # noqa: E402
    FullConnectomeConfig,
    build_full_visual_connectome,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/malecns-v1.0")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/derived/full-visual-connectome-v1.npz",
    )
    parser.add_argument("--edge-weight-threshold", type=int, default=10)
    parser.add_argument("--attitude-per-channel", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_full_visual_connectome(
        args.raw_dir,
        args.output,
        FullConnectomeConfig(
            edge_weight_threshold=args.edge_weight_threshold,
            attitude_per_channel=args.attitude_per_channel,
        ),
    )
    print(json.dumps(manifest["selection"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
