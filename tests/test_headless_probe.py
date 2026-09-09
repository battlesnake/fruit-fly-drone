from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE = REPO_ROOT / "scripts" / "probe_headless_stack.py"


def test_cpu_probe_is_deterministic_and_machine_readable() -> None:
    completed = subprocess.run(
        [sys.executable, str(PROBE), "--only", "cpu", "--steps", "10"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["passed"] is True
    assert report["probes"]["cpu_mujoco"]["deterministic"] is True
    assert report["probes"]["cpu_mujoco"]["finite"] is True
    assert report["versions"]["mujoco"] == "3.9.0"
