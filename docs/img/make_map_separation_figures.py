"""Figures for the map-separation trial (integration topic 65).

Panel 1: per-episode mean error for the three map arms at 100 vs 400
         particles on the FIXED scan set (six paired bars per episode).
Panel 2: per-episode localization-error curves for the six ideal-map
         divergences at 400 particles, with onset markers.

The topic-64 closed-loop loc-mean bars live in make_lesson64_figures.py
(lesson64-loc-error.png); a shared PNG made both docs cite a figure that
was half another experiment.

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
    ax.set_xticklabels(order, fontsize=9.5)
    ax.set_ylabel("定位平均误差 (m)", fontsize=13)
    ax.legend(fontsize=9.5, ncol=3)
    style_axes(ax, "同一批扫描：三种地图 × 两种粒子预算（逐回合配对）")
    ax.axhline(1.0, color="#6b7280", lw=0.8, ls="--")
    ax.text(0.995, 1.03, "虚线 = 发散判据 1 m", transform=ax.transAxes,
            ha="right", fontsize=10, color="#6b7280")


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
        ax.plot(idx, err, lw=1.1, alpha=0.85, label=key)
        if onset is not None:
            ax.axvline(idx[onset], color=C_DIV, lw=0.9, ls=":", alpha=0.6)
        n_curves += 1
    ax.set_xlabel("仿真时间 (s)", fontsize=13)
    ax.set_ylabel("理想图 PF 位置误差 (m)", fontsize=13)
    ax.set_ylim(0, 11)
    ax.legend(fontsize=9.5)
    style_axes(ax, f"400 粒子下理想图臂的 {n_curves} 次发散（竖线=发散起点，之后未恢复）")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.4), dpi=DPI, layout="constrained")
    panel_map_arms(axes[0])
    panel_divergences(axes[1])
    out = OUT / "map-separation-charts.png"
    fig.savefig(out, dpi=DPI, facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
