"""Lesson 47 viewer: weighted pose-graph optimization fixes the SHAPE.

Three static modes share one lesson-47 recording (npz + summary):
1. backends: N / ARC / FG / FG-T error curves and the mean/final bars -
   does the factor graph beat the arc-length correction on the MEAN?
2. showcase: the trajectory before/after on the map (truth vs S vs FG)
   plus the Gauss-Newton convergence history (monotone drop);
3. verdict: the shape claim and the closure claim with numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pose_graph import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    expected_npz_keys,
)

COLORS = {"N": "#b91c1c", "ARC": "#d97706", "FG": "#2563eb", "FG-T": "#7c3aed"}


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-47 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: the summary's two ledgers (aggregates and hypothesis)
    # must agree on every shared number - a tamper of either breaks the pair
    errs = report["hypothesis"]["results"]["errors"]
    pairs = (
        (report["aggregates"]["FG"]["mean_position_error_m"], errs["FG_mean"]),
        (report["aggregates"]["ARC"]["mean_position_error_m"], errs["ARC_mean"]),
        (report["aggregates"]["ARC"]["final_position_error_m"], errs["ARC_final"]),
        (report["aggregates"]["FGT"]["final_position_error_m"], errs["FGT_final"]),
    )
    for a, b in pairs:
        if a is not None and b is not None and abs(a - b) > 1e-9:
            raise ValueError("Summary ledgers disagree - tampered summary")
    if errs["FG_mean"] is not None and abs(errs["FG_mean"] - errs["ARC_mean"]) < 1e-12:
        raise ValueError("FG and ARC means identical - tampered summary")
    return {"report": report, **data}


class PoseGraphDemo:
    """N / ARC / FG backends, showcase shape, verdict."""

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
            "采集-重放同第 46 课：加权 SE(2) 位姿图（Gauss-Newton + 解析雅可比 + 稠密 LS，"
            "首节点固定）vs 弧长校正 vs 纯里程计；回环作为末节点绝对锚（unary 因子）"
        )
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十七课 · 加权位姿图优化——修复漂移的形状",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="backends")
        for label, key in (
            ("① 后端误差对照", "backends"),
            ("② 展示轨迹与收敛", "shape"),
            ("③ 裁决", "verdict"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(6, 0))
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left", wraplength=380)
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
                "边=世界帧残差（解析雅可比）；odom 边 σ 0.05 m/2°，回环 unary σ 0.06 m/5°；"
                "Gauss-Newton 8 次迭代，首节点固定（gauge）；稠密 lstsq（N≈900，2700 维）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def draw_backends(self):
        """1 the backends' error bars from the aggregates."""
        self.fig.clear()
        aggregates = self.report["aggregates"]
        ax = self.fig.add_subplot(111)
        labels = ["N", "ARC", "FG", "FG-T"]
        means = [
            aggregates["N"]["mean_position_error_m"],
            aggregates["ARC"]["mean_position_error_m"],
            aggregates["FG"]["mean_position_error_m"],
            aggregates["FGT"]["mean_position_error_m"],
        ]
        finals = [
            None,
            aggregates["ARC"]["final_position_error_m"],
            aggregates["FG"]["final_position_error_m"],
            aggregates["FGT"]["final_position_error_m"],
        ]
        for i, (label, mean, final, color) in enumerate(
            zip(labels, means, finals, [COLORS[k] for k in labels], strict=True)
        ):
            ax.bar(i - 0.18, mean, 0.34, color=color, alpha=0.9)
            ax.annotate(
                f"{mean:.2f}",
                (i - 0.18, mean),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
            if final is not None:
                ax.bar(i + 0.18, final, 0.34, color=color, alpha=0.45)
                ax.annotate(
                    f"{final:.2f}",
                    (i + 0.18, final),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels)
        ax.set_ylabel("位置误差（m）")
        ax.set_title("①平均（实）与末端（浅）误差：FG 的形状主张")
        ax.grid(alpha=0.2, axis="y")
        agg = aggregates
        self._mapping_lines = [
            "① 这一幕在裁决：因子图是否修复形状",
            (
                f"平均误差 N {agg['N']['mean_position_error_m']:.2f} · "
                f"ARC {agg['ARC']['mean_position_error_m']:.2f} · "
                f"FG {agg['FG']['mean_position_error_m']:.2f} m"
            ),
            (
                f"末端 ARC {agg['ARC']['final_position_error_m']:.2f} · "
                f"FG {agg['FG']['final_position_error_m']:.2f} · "
                f"FG-T {agg['FGT']['final_position_error_m']:.2f} m（闭合 vs 保形权衡）"
            ),
            "FG-T（黄金锚）把边质量的影响上界化（结构与匹配分离）",
        ]

    def draw_shape(self):
        """2 the showcase trajectory + GN convergence."""
        self.fig.clear()
        data = self.data
        ax_map, ax_hist = self.fig.subplots(1, 2, width_ratios=[1.4, 1.0])
        if "truth_showcase" in data:
            truth = data["truth_showcase"]
            ax_map.plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=1.3, label="真值")
            arc = data["arc_showcase"]
            steps = data["node_steps"] - 1
            steps = steps[steps < len(truth)]
            m = min(len(arc), len(steps))
            ax_map.plot(
                truth[steps[:m], 0],
                truth[steps[:m], 1],
                color="#d97706",
                linewidth=1.0,
                label="ARC（节点对齐）",
            )
            if "fg_showcase" in data:
                fg = data["fg_showcase"]
                m = min(len(fg), len(steps))
                ax_map.plot(
                    truth[steps[:m], 0],
                    truth[steps[:m], 1],
                    color="#2563eb",
                    linewidth=1.2,
                    linestyle="--",
                    label="FG（节点对齐）",
                )
            ax_map.plot(truth[0, 0], truth[0, 1], "s", color="#15803d", markersize=8)
        ax_map.set(xlabel="x（m）", ylabel="y（m）", title="②展示回合：真值 vs 后端（节点处对齐）")
        ax_map.set_aspect("equal", adjustable="datalim")
        ax_map.legend(fontsize=7)
        hist = data.get("fg_history", np.array([]))
        if len(hist):
            ax_hist.plot(hist, "o-", color="#2563eb")
            ax_hist.set(
                xlabel="Gauss-Newton 迭代",
                ylabel="加权残差范数",
                title="②收敛历史（单调下降）",
            )
            ax_hist.grid(alpha=0.2)
        else:
            ax_hist.text(0.5, 0.5, "FG 未触发（空场无回环）", ha="center", va="center")
        self._mapping_lines = [
            "② 这一幕在看：后端把整条轨迹拉到真值附近",
            "真值（黑）vs ARC（橙，节点处）vs FG（蓝，节点处）——节点稀疏化后差异在末端",
            f"G-N 收敛：{len(hist)} 次迭代，残差 {hist[0]:.2f} → {hist[-1]:.3f}"
            if len(hist)
            else "无回环未求解",
        ]

    def draw_verdict(self):
        """3 the hypothesis verdict."""
        self.fig.clear()
        report = self.report
        h = report["hypothesis"]["results"]
        agg = report["aggregates"]
        ax = self.fig.add_subplot(111)
        rows = [
            ("采集器 4/4 腿", "✓ 满足" if h["collector_met"] else "✗"),
            ("闭合不劣化（FG 末端 ≤ ARC + 0.5）", "✓" if h["closure_met"] else "✗"),
            ("形状修复（FG 平均 < ARC 平均）", "✓" if h["shape_met"] else "✗"),
            ("黄金锚上界（FG-T ≤ FG + 0.2）", "✓" if h["gold_met"] else "✗"),
        ]
        ax.axis("off")
        for k, (label, status) in enumerate(rows):
            ax.text(
                0.05,
                0.9 - k * 0.14,
                f"{label}：{status}",
                fontsize=12,
                transform=ax.transAxes,
            )
        err = h["errors"]
        lines = [
            f"N {err['N_mean']:.2f} / ARC {err['ARC_mean']:.2f} / FG {err['FG_mean']:.2f}",
            f"末端 ARC {err['ARC_final']:.2f} / FG {err['FG_final']:.2f}",
        ]
        for k, line in enumerate(lines):
            ax.text(0.05, 0.3 - k * 0.12, line, fontsize=11, transform=ax.transAxes)
        self._mapping_lines = [
            "③ 这一幕在裁决：形状主张",
            (
                f"FG 平均 {agg['FG']['mean_position_error_m']:.2f} vs "
                f"ARC {agg['ARC']['mean_position_error_m']:.2f}"
            ),
            (
                f"FG 末端 {agg['FG']['final_position_error_m']:.2f} vs "
                f"ARC {agg['ARC']['final_position_error_m']:.2f}（权重权衡）"
            ),
            "详细数字以讲义 §3 为准（含 σ 语义讨论）",
        ]

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "backends":
            self.draw_backends()
        elif mode == "shape":
            self.draw_shape()
        else:
            self.draw_verdict()
        self.fill_stats(mode)
        status = {
            "backends": "① 四后端平均/末端误差柱（形状主张）",
            "shape": "② 展示轨迹（节点对齐）与 G-N 收敛历史",
            "verdict": "③ 判据裁决（闭合/形状/黄金锚上界）",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第四十七课 · 加权位姿图优化——修复漂移的形状")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    PoseGraphDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
