"""Resume registered round 11 without modifying its frozen module or records."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.experiments.navigation_bypass_study import ROOT, episode
from embodied_learning.navigation_robustness import CONDITIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory", type=Path, default=ROOT / "results/navigation_reinforcement_v11"
    )
    args = parser.parse_args()
    out = args.directory
    summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
    p = summary["protocol"]
    assert p["round"] == 11 and not p["smoke"]
    assert digest(out / "protocol.json") == summary["protocol_sha256"]
    assert digest(ROOT / "docs/navigation-bypass-round11-plan.md") == p["registration_sha256"]
    for name, sha in p["source_sha256"].items():
        assert digest(ROOT / name) == digest(out / "source" / name) == sha
    finished = {r["file"] for r in summary["rows"]}
    assert len(finished) == len(summary["rows"])
    for row in summary["rows"]:
        assert digest(out / row["file"]) == row["sha256"]
    base_wall = summary["wall_s"]
    summary.setdefault("execution_segments", []).append(
        {
            "start_after_rows": len(finished),
            "reason": "original Python process absent; checked all saved records and frozen inputs",
        }
    )
    summary["wall_s_semantics"] = (
        "persisted segment time sum; excludes interruption gaps and unfinished work"
    )
    started = time.perf_counter()
    for layout, ob, seed, cohort, delay, method, profile in p["cases"]:
        file = f"{layout}_{ob}_s{seed}_{delay}_{method}_{profile}.npz"
        if file in finished:
            continue
        target = out / file
        if target.exists():
            target.rename(target.with_suffix(".interrupted.npz"))
        tick = time.perf_counter()
        row, data = episode(layout, ob, seed, CONDITIONS[delay], method, profile)
        np.savez_compressed(target, **data)
        row.update(
            file=file, sha256=digest(target), cohort=cohort, wall_s=time.perf_counter() - tick
        )
        summary["rows"].append(row)
        summary["wall_s"] = base_wall + time.perf_counter() - started
        save_json(out / "summary.json", summary)
        print(
            f"{len(summary['rows'])}/{len(p['cases'])} {profile} s{seed} {method}: {row['status']} {row['wall_s']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
