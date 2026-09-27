"""Lesson-64 supplement figures: trajectory chains from the v4 record.

Representative-case selection rule (fixed, not cherry-picked): the reachable
task with the LONGEST group-A travelled distance at sensor seed 0, groups B
(odometry) and C (PF) — the pair that shows what 2% wheel-bias drift does
against a map-corrected filter on the identical, most demanding task.

    uv run python docs/img/make_lesson64_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.plotting import configure_plot_font

configure_plot_font()

RESULTS = Path("results")
OUT = Path("docs/img")
DPI = 140

names = {"B": "里程计", "C": "参考图 PF"}


def main():
    report = json.loads(
        (RESULTS / "photo_room_navigation_v4" / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(RESULTS / "photo_room_navigation_v4" / "trajectories.npz") as z:
        archive = {k: z[k].copy() for k in z.files}

    rows0 = [
        r
        for r in report["rows"]
        if r["group"] == "A" and r["task"] != "into_bed_footprint" and r["seed"] == 0
    ]
    task = max(rows0, key=lambda r: r["travelled_m"])["task"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=DPI, layout="constrained")
    for ax, group in zip(axes, ("B", "C"), strict=True):
        key = f"{task}_{group}_s0"
        truth = archive[f"{key}_truth"]
        est = archive[f"{key}_estimate"]
        row = next(
            r
            for r in report["rows"]
            if r["task"] == task and r["group"] == group and r["seed"] == 0
        )
        ax.plot(est[:, 0], est[:, 1], color="#ea580c", lw=1.4, ls="--", label="估计轨迹")
        ax.plot(truth[:, 0], truth[:, 1], color="#1d4ed8", lw=2.0, label="真实轨迹")
        ax.plot(truth[0, 0], truth[0, 1], "o", color="#111827", ms=7, label="起点", zorder=5)
        ax.set_title(
            f"{group} 组（{names[group]}）{task}: {row['result']},"
            f" loc_mean {row['loc_mean_m']:.4f} m",
            fontsize=12,
            loc="left",
        )
        ax.set_xlabel("x (m)", fontsize=11)
        ax.set_ylabel("y (m)", fontsize=11)
        ax.set_aspect("equal")
        ax.grid(color="#e5e7eb", lw=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.04))
    out = OUT / "lesson64-trajectory.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
