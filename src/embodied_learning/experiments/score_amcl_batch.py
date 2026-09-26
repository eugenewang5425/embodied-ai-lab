"""Unified scoring over the gated AMCL batch (aligned + continuous).

Reads every run under results/p3_amcl_batch_v2, scores each against its
frozen input with the shared nav_scoring functions, and writes
unified_scoring.json with per-run rows plus per-config aggregates.

Run from the repo root (no ROS needed - pure post-processing):
    uv run python src/embodied_learning/experiments/score_amcl_batch.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.nav_scoring import continuous_pose, score_aligned

BASE = Path(__file__).resolve().parents[3]
BATCH = BASE / "results" / "p3_amcl_batch_v2"


def score_run(run_dir):
    frozen = Path(run_dir).parent.name  # "<config>_s<seed>"
    seed = int(frozen.rsplit("_s", 1)[1])
    data = dict(np.load(BASE / "results" / f"amcl_bridge_v2_s{seed}" / "frozen_input.npz"))
    truth, odom, timestamps = data["truth"], data["odom"], data["timestamps"]
    npz = Path(run_dir) / "amcl_poses.npz"
    if not npz.exists():
        return None
    with np.load(npz) as z:
        if len(z["x"]) == 0:
            return {"n_poses": 0, "aligned": None, "continuous": None}
        est = np.column_stack([z["x"], z["y"], z["yaw"]])
        aligned, _ = score_aligned(est, truth, z["stamp"], timestamps)
        cont = continuous_pose(est, odom, z["stamp"], timestamps)
        n = min(len(cont), len(truth))
        cont_err = np.linalg.norm(cont[:n, :2] - truth[:n, :2], axis=1)
        continuous = {
            "n_matched": n,
            "mean_m": float(cont_err.mean()),
            "p95_m": float(np.percentile(cont_err, 95)),
            "end_m": float(cont_err[-1]),
            "coverage": n / len(truth),
        }
        return {"n_poses": len(z["x"]), "aligned": aligned, "continuous": continuous}


def main():
    import re

    rows = []
    for cfg_dir in sorted(BATCH.iterdir()):
        if not cfg_dir.is_dir() or not re.fullmatch(r"A-[a-z]+_s\d+", cfg_dir.name):
            continue
        gname, seed_s = cfg_dir.name.rsplit("_s", 1)
        for rep_dir in sorted(cfg_dir.iterdir()):
            if not rep_dir.is_dir() or not re.fullmatch(r"r\d+", rep_dir.name):
                continue
            sc = score_run(rep_dir)
            params_ok = (rep_dir / "params_effective.json").exists()
            rows.append(
                {
                    "group": gname,
                    "seed": int(seed_s),
                    "rep": rep_dir.name,
                    "params_gate": params_ok,
                    "score": sc,
                }
            )
            a = sc["aligned"] if sc and sc["aligned"] else None
            print(
                f"{gname} s{seed_s} {rep_dir.name}: "
                f"aligned {a['mean_m']:.4f} m (n={a['n_matched']})" if a else f"{gname} {rep_dir.name}: no poses"
            )

    summary = {}
    for gname in sorted({r["group"] for r in rows}):
        valid = [
            r for r in rows if r["group"] == gname and r["score"] and r["score"]["aligned"]
        ]
        if not valid:
            summary[gname] = {"n_valid": 0}
            continue
        am = [r["score"]["aligned"]["mean_m"] for r in valid]
        cm = [r["score"]["continuous"]["mean_m"] for r in valid]
        ups = [r["score"]["n_poses"] for r in valid]
        cov = [r["score"]["aligned"]["coverage"] for r in valid]
        summary[gname] = {
            "n_valid": len(valid),
            "aligned_mean_m": {"mean": float(np.mean(am)), "std": float(np.std(am))},
            "continuous_mean_m": {"mean": float(np.mean(cm)), "std": float(np.std(cm))},
            "n_poses": {"mean": float(np.mean(ups)), "std": float(np.std(ups))},
            "coverage": {"mean": float(np.mean(cov))},
        }

    out = {"rows": rows, "summary": summary}
    (BATCH / "unified_scoring.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print("\n=== summary (aligned | continuous) ===")
    for gname, s in summary.items():
        if s.get("n_valid"):
            print(
                f"  {gname}: {s['aligned_mean_m']['mean']:.4f}±{s['aligned_mean_m']['std']:.4f}"
                f" | {s['continuous_mean_m']['mean']:.4f}±{s['continuous_mean_m']['std']:.4f}"
                f" m, poses {s['n_poses']['mean']:.0f}, n={s['n_valid']}"
            )
        else:
            print(f"  {gname}: no valid runs")


if __name__ == "__main__":
    main()
