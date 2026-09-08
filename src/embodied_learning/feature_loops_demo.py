"""Lesson 50 viewer: discriminator exploration - three candidate mechanisms,
none separates (the honest-negative record).

Two static modes share one lesson-50 recording:
1. separation: score/distance distributions of the true-loop frames vs far
   frames for the three discriminators (sorted histogram, cyclic profile,
   map agreement) - the overlapping histograms are the finding;
2. verdict: the detection-acceptance-correctness ladder against the
   lesson-49 oracle-candidate baseline (K: 4.9%).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.feature_loops import DEFAULT_RESULTS, EXPERIMENT


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-50 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != set(report["archive_keys"]):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: the verdict is recomputable from the summary rows
    rows = report["episodes"]
    accepted = [r for r in rows if r["accepted"]]
    correct = [r for r in accepted if r["correct"]]
    recomputed = len(correct) / max(1, len(accepted))
    if abs(recomputed - report["hypothesis"]["results"]["direction_correctness"]) > 1e-9:
        raise ValueError("Direction-correctness mismatch - tampered summary")
    return {"report": report, **data}


class FeatureLoopDemo:
    """The three discriminators and their overlapping distributions."""

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
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第五十课 · 特征化回环检测——三个判别机制的实测无判别力",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "探索三种“对齐质量之外的判别信息”：排序直方图 / 环移角剖面 / 建图一致性相关峰。"
                "全部高分重叠（无分离）——单帧级判别信息在直墙对称环境不存在"
            ),
        ).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="separation")
        for label, key in (("① 判别分布", "separation"), ("② 结论对照", "verdict")):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(6, 0))
        self.stats = ttk.Label(middle, width=52, anchor="nw", justify="left", wraplength=410)
        self.stats.pack(side="right", fill="y", padx=(10, 8))
        left = ttk.Frame(middle, width=1020, height=500)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        self.fig = Figure(figsize=(10.2, 5.0), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(4, 0))
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def stats_text(self):
        r = self.report["hypothesis"]["results"]
        return (
            "实测指标（无 oracle 检测）\n\n"
            f"检出帧数 {r['detections']} · accepted {r['accepted']} · correct {r['correct']}\n"
            f"方向正确率 {r['direction_correctness']:.3f}\n\n"
            f"判别分布重叠度（0=完全分离，1=完全重叠）\n"
            f"排序直方图 {r['overlap_sorted_histogram']:.2f}\n"
            f"环移剖面     {r['overlap_cyclic_profile']:.2f}\n"
            f"建图一致性   {r['overlap_map_agreement']:.2f}\n\n"
            "对照：第 49 课 K 组（oracle 候选）方向正确率 0.049\n"
            "结论：三项机制均无判别力——下一步 SC/MM（全局因子结构）"
        )

    def draw_separation(self):
        self.fig.clear()
        axes = self.fig.subplots(1, 3).reshape(-1)
        r = self.report["hypothesis"]["results"]
        labels = ("排序直方图", "环移角剖面", "建图一致相关")
        overlaps = (
            r["overlap_sorted_histogram"],
            r["overlap_cyclic_profile"],
            r["overlap_map_agreement"],
        )
        for ax, name, ov in zip(axes, labels, overlaps, strict=True):
            ax.bar(["重叠度"], [ov], color="#b91c1c")
            ax.axhline(0.3, color="k", ls="--", lw=1)
            ax.text(0, ov + 0.02, f"{ov:.2f}", ha="center", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{name}\n（虚线=0.3 可分离线）", fontsize=9)
        self.fig.suptitle("三个判别机制的真/远分布重叠度（全部 >0.5：无判别力）", fontsize=11)

    def draw_verdict(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        groups = ["oracle+K (49)", "无oracle+地图一致 (50)"]
        values = [0.049, self.report["hypothesis"]["results"]["direction_correctness"]]
        colors = ["#9ca3af", "#2563eb"]
        ax.bar(groups, values, color=colors)
        ax.axhline(0.80, color="k", ls="--", lw=1)
        ax.text(1, values[1] + 0.01, f"{values[1]:.3f}", ha="center", fontsize=10)
        ax.set_ylabel("方向正确率（accepted 中 correct）")
        ax.set_ylim(0, 1.05)
        ax.set_title("第 50 课的两段式仍然失败：判别信息未找到——退到全局结构（SC/MM）", fontsize=10)

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        if self.mode.get() == "verdict":
            self.draw_verdict()
        else:
            self.draw_separation()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="lesson-50 feature loop demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第五十课 · 特征化回环检测")
    root.geometry("1400x740+30+20")
    FeatureLoopDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
