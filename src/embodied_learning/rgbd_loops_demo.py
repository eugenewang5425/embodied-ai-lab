"""Lesson 55 viewer: RGB-D visual loop closure - the positive result.

Two static modes share one lesson-55 recording:
1. groups: the three-way comparison (no-retrieval baseline / oracle reference
   / RGB-D top-k treatment) on candidates, acceptance and correctness;
2. retrieval: the probe-frame similarity trace with the retrieved top-k and
   the true-loop region marked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.rgbd_loops import DEFAULT_RESULTS, EXPERIMENT


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-55 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != set(report["archive_keys"]):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: precision recomputable from the archived probe flags
    frames = data["probe_frames_all"]
    near = data["probe_near_start"]
    near_map = dict(zip(frames.tolist(), near.tolist()))
    topk = data["topk_frames"].tolist()
    recomputed = sum(1 for k in topk if near_map.get(k, 0)) / max(1, len(topk))
    if abs(recomputed - report["hypothesis"]["results"]["precision_topk"]) > 1e-9:
        raise ValueError("Precision mismatch - tampered summary")
    return {"report": report, **data}


class RgbdDemo:
    """The three-group verdict and the retrieval trace."""

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
            text="第五十五课 · RGB-D 视觉回环——外观检索补判别（四判据全达成）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "检索层（标记词袋 top-12）供判别，验证层（K 匹配器）供精度："
                "方向正确率 0.302 → 1.000，审查量 289 → 12（4%）"
            ),
        ).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="groups")
        for label, key in (("① 三组对照", "groups"), ("② 检索迹线", "retrieval")):
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
        agg = self.report["aggregates"]
        lines = ["三组对照（修正判据）", ""]
        for g in ("GEO-ALL", "GEO-ORACLE", "RGBD-TOPK"):
            a = agg[g]
            lines.append(
                f"{g}: 候选 {a['n_candidates']} 接受 {a['accepted']} "
                f"正确率 {a['direction_correctness']:.3f}"
            )
        lines.append("")
        lines.append(f"精确率@top-12: {r['precision_topk']:.3f}")
        lines.append(f"管线成功（回环真闭合）: {r['pipeline_success']}")
        lines.append(
            f"效率: {agg['RGBD-TOPK']['n_candidates']}/{agg['GEO-ALL']['n_candidates']} = 4%"
        )
        return "\n".join(lines)

    def draw_groups(self):
        self.fig.clear()
        agg = self.report["aggregates"]
        groups = ("GEO-ALL", "GEO-ORACLE", "RGBD-TOPK")
        names = ("无检索基线", "真值候选参考", "RGB-D top-12")
        correct = [agg[g]["direction_correctness"] for g in groups]
        cands = [agg[g]["n_candidates"] for g in groups]
        axes = self.fig.subplots(1, 2).reshape(-1)
        axes[0].bar(names, correct, color=["#9ca3af", "#2563eb", "#b91c1c"])
        axes[0].axhline(0.80, color="k", ls="--", lw=1)
        for i, v in enumerate(correct):
            axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
        axes[0].set_ylabel("方向正确率（修正判据）")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("方向正确率：0.302 → 1.000", fontsize=10)
        axes[1].bar(names, cands, color=["#9ca3af", "#2563eb", "#b91c1c"])
        for i, v in enumerate(cands):
            axes[1].text(i, v + max(cands) * 0.02, str(v), ha="center", fontsize=9)
        axes[1].set_ylabel("候选审查量")
        axes[1].set_title("审查量：289 → 12（4%）", fontsize=10)

    def draw_retrieval(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        frames = self.data["probe_frames_all"]
        sims = self.data["probe_sims"]
        near = self.data["probe_near_start"]
        topk = set(self.data["topk_frames"].tolist())
        colors = ["#b91c1c" if near[i] else "#9ca3af" for i in range(len(frames))]
        sizes = [12 if f in topk else 6 for f in frames]
        ax.scatter(frames, sims, c=colors, s=sizes)
        ax.set_xlabel("探帧")
        ax.set_ylabel("外观相似度（对起点参考）")
        ax.set_title("检索迹线：红=真回环帧（近起点），大点=top-12 检索", fontsize=10)

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        if self.mode.get() == "retrieval":
            self.draw_retrieval()
        else:
            self.draw_groups()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="lesson-55 RGB-D demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第五十五课 · RGB-D 视觉回环")
    root.geometry("1400x740+30+20")
    RgbdDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
