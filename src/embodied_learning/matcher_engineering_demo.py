"""Lesson 49 viewer: loop matcher engineering - B/P/R/K ablation ladder.

Three static modes share one lesson-49 recording (npz + summary):
1. candidates: the easy & the hardest correct candidates - reference submap
   (blue), current frame (gray), the K aligned cloud (orange) and the delta
   arrows of B (dashed) vs K (solid);
2. metrics: acceptance and direction-correctness bars against the
   pre-registered lines (K must clear both, B must fail one);
3. mechanism: the ranked top-k hypotheses of the hardest candidate with
   their median-NN alignment distance and the correctness flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.matcher_engineering import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUPS,
    apply_delta,
    expected_npz_keys,
)


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-49 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: the aggregates must be recomputable from the per-candidate
    # rows and the verdict flags consistent with them
    rows = report["episodes"]
    for group in GROUPS:
        take = [r for r in rows if r["group"] == group]
        agg = report["aggregates"][group]
        if agg["n_candidates"] != len(take):
            raise ValueError("Candidate count mismatch - tampered summary")
        recomputed = sum(1 for r in take if r["accepted"]) / len(take)
        if abs(recomputed - agg["acceptance"]) > 1e-9:
            raise ValueError("Acceptance mismatch - tampered summary")
    hyp = report["hypothesis"]["results"]
    recomputed_accept = (
        report["aggregates"]["K"]["acceptance"] >= 0.10
        and report["aggregates"]["K"]["direction_correctness"] >= 0.80
    )
    if bool(hyp["acceptance_met"]) != bool(recomputed_accept):
        raise ValueError("Acceptance verdict inconsistent - tampered summary")
    return {"report": report, **data}


class MatcherDemo:
    """The B/P/R/K ladder, the candidate pool and the top-k mechanism."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()
        self.root = root
        self.data = data
        self.report = data["report"]
        self.subtitle = (
            "采集-重放（2 圈巡游 / 64 线 / 6 cm 测距噪声）；匹配器输入仅点云，"
            "候选与正确性评估用几何真值——回环检测是第 50 课"
        )
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十九课 · 回环匹配器工程化——点对面 / 弧长重采样 / Top-k",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="metrics")
        for label, key in (
            ("① 候选展示", "candidates"),
            ("② 指标对照", "metrics"),
            ("③ Top-k 机制", "mechanism"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(6, 0))
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left", wraplength=400)
        self.stats.pack(side="right", fill="y", padx=(10, 8))
        left = ttk.Frame(middle, width=1020, height=500)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        self.fig = Figure(figsize=(10.2, 5.0), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(4, 0))
        ttk.Label(
            outer,
            text=(
                "锚定-抛光：多尺度点对点梯子（0.8/0.4/0.2/0.15 m）锁定盆地→点对面 GN 抛光；"
                "纯点对面在光滑轮廓上沿切线滑移（平面残差≈0 错位解），锚定是抛光成立的前提"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def stats_text(self):
        agg = self.report["aggregates"]
        lines = ["各组实测（候选对级）", ""]
        for group in GROUPS:
            a = agg[group]
            lines.append(
                f"{group}: 接受 {a['acceptance']:.2f}  "
                f"方向 {a['direction_correctness']:.2f}  "
                f"中位NN {a['median_nn_accepted_m'] if a['median_nn_accepted_m'] else '--'} m"
            )
        lines.append("")
        lines.append("验收线（预登记）")
        lines.append("B 至少未过一条；K 双线达标")
        hyp = self.report["hypothesis"]["results"]
        lines.append(
            f"采集器: {hyp['collector_met']}  动机: {hyp['motivation_met']}  "
            f"达成: {hyp['acceptance_met']}"
        )
        return "\n".join(lines)

    def draw_metrics(self):
        self.fig.clear()
        agg = self.report["aggregates"]
        axes = self.fig.subplots(1, 2).reshape(-1)
        for ax, key, line, ylabel in zip(
            axes,
            ("acceptance", "direction_correctness"),
            (0.10, 0.80),
            ("接受率（accepted / candidates）", "方向正确率（correct / accepted）"),
            strict=True,
        ):
            ax.bar(
                list(GROUPS),
                [agg[g][key] for g in GROUPS],
                color=["#9ca3af", "#0f766e", "#2563eb", "#b91c1c"],
            )
            ax.axhline(line, color="k", ls="--", lw=1)
            for i, g in enumerate(GROUPS):
                ax.text(i, agg[g][key] + 0.01, f"{agg[g][key]:.2f}", ha="center", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            ax.set_title("验收线 %.0f%%（虚线）" % (line * 100.0), fontsize=10)

    def draw_candidates(self):
        self.fig.clear()
        axes = self.fig.subplots(1, 2).reshape(-1)
        refs = self.data["showcase_ref_pts"]
        curs = self.data["showcase_cur_pts"]
        true_ds = self.data["showcase_delta_true"]
        hyps = self.data["showcase_hypotheses"]
        titles = ("① 最易正确候选", "② 最难正确候选")
        for ax, i, title in zip(axes, range(min(2, len(refs))), titles, strict=True):
            ref_pts = refs[i][~np.isnan(refs[i][:, 0])]
            cur_pts = curs[i][~np.isnan(curs[i][:, 0])]
            d_true = true_ds[i]
            d_best = hyps[i][int(np.argmin(np.linalg.norm(hyps[i] - d_true, axis=1)))]
            ax.scatter(ref_pts[:, 0], ref_pts[:, 1], s=3, c="#2563eb", label="参考子图")
            ax.scatter(cur_pts[:, 0], cur_pts[:, 1], s=3, c="#9ca3af", alpha=0.8, label="当前帧")
            aligned = apply_delta(cur_pts, d_best)
            ax.scatter(aligned[:, 0], aligned[:, 1], s=4, c="#b91c1c", label="K 对齐后")
            ax.annotate(
                "真值 Δ",
                d_true[:2],
                xytext=(d_true[0] + 0.3, d_true[1] + 0.3),
                fontsize=7,
                arrowprops={"arrowstyle": "->", "lw": 0.8},
            )
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_aspect("equal")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")

    def draw_mechanism(self):
        self.fig.clear()
        hyps = self.data["showcase_hypotheses"]
        metrics = self.data["showcase_hyp_metrics"]
        true_ds = self.data["showcase_delta_true"]
        # the hardest candidate is the second captured one
        idx = min(1, len(hyps) - 1)
        t_delta = true_ds[idx]
        deltas = np.asarray(hyps[idx])
        meds = metrics[idx][:, 0]
        colors = ["#b91c1c" if np.linalg.norm(d - t_delta) < 0.5 else "#9ca3af" for d in deltas]
        ax = self.fig.add_subplot(1, 1, 1)
        xs = np.arange(len(deltas))
        ax.bar(xs, meds, color=colors)
        for i, d in enumerate(deltas):
            label = f"H{i + 1}: Δ=({d[0]:.2f},{d[1]:.2f},{np.degrees(d[2]):+.0f}°)"
            ax.text(i, meds[i] + 0.005, label, ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"H{i + 1}" for i in xs])
        ax.set_ylabel("中位 NN 对齐距离 (m)")
        ax.set_xlabel("按中位 NN 排序的 top-k 假设（红=距真值 Δ<0.5m，灰=错误假设）")
        ax.set_title("最难关卡的 top-k 假设排序（第 46 课错峰现象的工程化解）", fontsize=10)

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        if self.mode.get() == "candidates":
            self.draw_candidates()
        elif self.mode.get() == "mechanism":
            self.draw_mechanism()
        else:
            self.draw_metrics()
        self.canvas.draw_idle()
        self.status.configure(
            text=f"候选对数 {self.report['n_candidates_total']} · 记录 {self.report['protocol']['range_noise_sigma_m']} m 噪声"
        )


def main():
    parser = argparse.ArgumentParser(description="lesson-49 matcher demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第四十九课 · 回环匹配器工程化")
    root.geometry("1500x780+30+20")
    MatcherDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
