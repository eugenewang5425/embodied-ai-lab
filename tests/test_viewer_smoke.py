"""Lesson-2 viewer smoke test: the PD window must open, play, and exit cleanly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_viewer_pd_window_opens_and_exits():
    env = {**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "TkAgg"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.viewer",
            "--policy",
            "pd",
            "--seconds",
            "1",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=str(ROOT),
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
