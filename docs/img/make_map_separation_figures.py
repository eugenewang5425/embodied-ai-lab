"""Figures for the map-separation trial (roadmap Phase 1) and lesson-64 supplements.

Panel 1: per-episode mean error for the three map arms at 100 vs 400
         particles on the FIXED scan set (paired dots per episode).
Panel 2: per-episode localization-error curves for the six ideal-map
         divergences at 400 particles, with onset markers.
Panel 3: lesson-64 closed-loop v4 - per-group localization error chains
         (first seed, first task) plus per-group loc_mean bars.

    uv run python docs/img/make_map_separation_figures.py
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

C_EST = "#b91c1c"
C_IDEAL = "#0f766e"
C_TRUTH = "#2563eb"
C_SELF = "#9ca3af"
C_DIV = "#b91c1c"


def style_axes(ax, title):
    ax.set_title(title, fontsize=15, loc="left", pad=13)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color="#e5e7eb", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def load_rows(result_dir):
    summary = json.loads((RESULTS / result_dir / "summary.json").read_text(encoding="utf-8"))
    return [r for r in summary["rows"] if r["status"] == "ok"]


def panel_map_arms(ax):
    lo = load_rows("navigation_benchmark_2026-09-24_pilot_v4")
    hi = load_rows("nav_sep_particles400_v1")
    key = lambda r: (r["arm"], r["world_seed"], r["sensor_seed"])
    hi_by = {key(r): r for r in hi}
    order = []
    lo_vals = {"ideal": [], "truth": [], "self": []}
    hi_vals = {"ideal": [], "truth": [], "self": []}
    for r in sorted(lo, key=key):
        k = key(r)
        if k not in hi_by:
            continue
        order.append(f"{r['arm']}\nw{r['world_seed']} s{r['sensor_seed']}")
        lo_vals["ideal"].append(r["pf_true_mean_m"])
        lo_vals["truth"].append(r["pf_sensor_truth_mean_m"])
        lo_vals["self"].append(r["pf_self_mean_m"])
        r2 = hi_by[k]
        hi_vals["ideal"].append(r2["pf_true_mean_m"])
        hi_vals["truth"].append(r2["pf_sensor_truth_mean_m"])
        hi_vals["self"].append(r2["pf_self_mean_m"])
    x = np.arange(len(order))
    w = 0.13
    series = (("理想图", "ideal", C_IDEAL), ("真值投影图", "truth", C_TRUTH), ("自建图", "self", C_SELF))
    for i, (lab, field, col) in enumerate(series):
        left = x + (2 * i - 3) * w  # 6 bars: 100p light, 400p saturated
        ax.bar(left, lo_vals[field], w * 0.9, color=col, alpha=0.45,
               label=f"{lab} · 100 粒子")
        ax.bar(left + w, hi_vals[field], w * 0.9, color=col,
               label=f"{lab} · 400 粒子")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("定位平均误差 (m)", fontsize=12)
    ax.legend(fontsize=8.5, ncol=3)
    style_axes(ax, "同一批扫描：三种地图 × 两种粒子预算（逐回合配对）")
    ax.axhline(1.0, color="#6b7280", lw=0.8, ls="--")


def panel_divergences(ax):
    n_curves = 0
    for key in ("iso_w1_s1", "aniso_w1_s0", "aniso_w0_s0", "aniso_w1_s1", "aniso_w2_s0", "aniso_w2_s1"):
        p = RESULTS / "nav_sep_particles400_v1" / key / "trajectories.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            err = z["pf_true_err"].astype(float)
        n = len(err)
        idx = np.arange(n) * (128.8 / n)
        big = err > 1.0
        onset = None
        for k in range(n):
            if big[k] and np.median(err[k:]) > 1.0:
                onset = k
                break
        ax.plot(idx, err, lw=1.0, alpha=0.85, label=key)
        if onset is not None:
            ax.axvline(idx[onset], color=C_DIV, lw=0.8, ls=":", alpha=0.6)
        n_curves += 1
    ax.set_xlabel("仿真时间 (s)", fontsize=12)
    ax.set_ylabel("理想图 PF 位置误差 (m)", fontsize=12)
    ax.set_ylim(0, 11)
    ax.legend(fontsize=8.5)
    style_axes(ax, f"400 粒子下理想图臂的 {n_curves} 次发散（竖线=发散起点，之后未恢复）")


def panel_lesson64(ax):
    report = json.loads(
        (RESULTS / "photo_room_navigation_v4" / "summary.json").read_text(encoding="utf-8")
    )
    rows = report["rows"]
    groups = ["A", "B", "C", "D"]
    names = {"A": "真值", "B": "里程计", "C": "参考图 PF", "D": "自建图 PF"}
    means = []
    for g in groups:
        vals = [r["loc_mean_m"] for r in rows if r["group"] == g and r["task"] != "into_bed_footprint"]
        means.append(float(np.mean(vals)))
    bars = ax.bar(names.values(), means, color=["#2563eb", "#b91c1c", "#0f766e", "#9ca3af"])
    for b, v in zip(bars, means, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=11)
    ax.set_ylabel("定位误差 loc_mean (m)", fontsize=12)
    style_axes(ax, "第 64 课闭环 v4（60 回合）：四组定位来源的平均误差")
    ax.text(
        0.99,
        0.95,
        "B 组 3/12 漂移超时；全组零接触；不可达 12/12 拒绝",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        color="#374151",
    )


def main():
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.5), dpi=DPI, layout="constrained")
    panel_map_arms(axes[0])
    panel_divergences(axes[1])
    panel_lesson64(axes[2])
    out = OUT / "map-separation-charts.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
