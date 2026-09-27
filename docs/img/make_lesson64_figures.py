"""Topic-64 supplement figures: loc-mean bars and trajectory chains (v4).

loc-error:  per-group loc_mean bars over the 60-episode v4 record
            (reachable tasks only; group A is the time-alignment check).
trajectory: representative-case selection rule (fixed, not cherry-picked):
            the reachable task with the LONGEST group-A travelled distance
            at sensor seed 0, groups B (odometry) and C (PF) — the pair
            that shows what 2% wheel-bias drift does against a
            map-corrected filter on the identical, most demanding task.

Trajectory panels keep equal aspect (honest path shape) but enforce a
minimum y-span so near-horizontal tasks stay readable.

    uv run python docs/img/make_lesson64_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from embodied_learning.plotting import configure_plot_font

configure_plot_font()

RESULTS = Path("results")
OUT = Path("docs/img")
DPI = 140

names = {"A": "真值", "B": "里程计", "C": "参考图 PF", "D": "自建图 PF"}


def figure_loc_error(report):
    rows = report["rows"]
    means = []
    for g in ("A", "B", "C", "D"):
        vals = [
            r["loc_mean_m"]
            for r in rows
            if r["group"] == g and r["task"] != "into_bed_footprint"
        ]
        means.append(float(np.mean(vals)))
    fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=DPI, layout="constrained")
    bars = ax.bar(
        [names[g] for g in ("A", "B", "C", "D")],
        means,
        color=["#2563eb", "#b91c1c", "#0f766e", "#9ca3af"],
        width=0.55,
    )
    for b, v in zip(bars, means, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}",
                ha="center", fontsize=12)
    ax.set_ylabel("定位误差 loc_mean (m)", fontsize=13)
    ax.set_ylim(0, max(means) * 1.18)
    ax.set_title("闭环 v4（60 回合）：四组定位来源的平均误差", fontsize=15, loc="left", pad=13)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color="#e5e7eb", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.99,
        0.96,
        "B 组 3/12 漂移超时；全组零接触；不可达 12/12 拒绝",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#374151",
    )
    out = OUT / "lesson64-loc-error.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"saved {out}")


def figure_trajectories(report, archive):
    rows0 = [
        r
        for r in report["rows"]
        if r["group"] == "A" and r["task"] != "into_bed_footprint" and r["seed"] == 0
    ]
    task = max(rows0, key=lambda r: r["travelled_m"])["task"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=DPI, layout="constrained")
    for ax, group in zip(axes, ("B", "C"), strict=True):
        key = f"{task}_{group}_s0"
        truth = archive[f"{key}_truth"]
        est = archive[f"{key}_estimate"]
        row = next(
            r
            for r in report["rows"]
            if r["task"] == task and r["group"] == group and r["seed"] == 0
        )
        ax.plot(est[:, 0], est[:, 1], color="#ea580c", lw=1.6, ls="--", label="估计轨迹")
        ax.plot(truth[:, 0], truth[:, 1], color="#1d4ed8", lw=2.2, label="真实轨迹")
        ax.plot(truth[0, 0], truth[0, 1], "o", color="#111827", ms=8, label="起点", zorder=5)
        ax.set_title(
            f"{group} 组（{names[group]}）{task}: {row['result']}\n"
            f"loc_mean {row['loc_mean_m']:.4f} m",
            fontsize=13,
            loc="left",
            pad=10,
        )
        ax.set_xlabel("x (m)", fontsize=12)
        ax.set_ylabel("y (m)", fontsize=12)
        xs = np.concatenate([est[:, 0], truth[:, 0]])
        ys = np.concatenate([est[:, 1], truth[:, 1]])
        y_center = 0.5 * (ys.min() + ys.max())
        y_span = max(float(ys.max() - ys.min()), 0.55)
        ax.set_ylim(y_center - y_span / 2, y_center + y_span / 2)
        ax.set_xlim(xs.min() - 0.12, xs.max() + 0.12)
        ax.set_aspect("equal")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(color="#e5e7eb", lw=0.7)
        ax.tick_params(labelsize=11)
    axes[0].legend(loc="upper left", fontsize=10.5, framealpha=0.92)
    out = OUT / "lesson64-trajectory.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"saved {out}")


def main():
    report = json.loads(
        (RESULTS / "photo_room_navigation_v4" / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(RESULTS / "photo_room_navigation_v4" / "trajectories.npz") as z:
        archive = {k: z[k].copy() for k in z.files}
    figure_loc_error(report)
    figure_trajectories(report, archive)


if __name__ == "__main__":
    main()
