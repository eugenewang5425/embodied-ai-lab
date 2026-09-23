"""Lesson 54 viewer: anisotropic-environment control + the metric erratum.

Two static modes share one lesson-54 recording:
1. metric erratum: legacy vs corrected direction correctness per arm - the
   bars that show the matcher was right all along;
2. the environment verdict: ISO vs ANISO under the corrected metric with
   the 80% pre-registered line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.aniso_env import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUPS,
)


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-54 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != set(report["archive_keys"]):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: corrected correctness recomputable from the archive
    for arm in ("ISO", "ANISO"):
        acc = data[f"{arm}_K_accepted"]
        cor = data[f"{arm}_K_correct"]
        recomputed = float(cor.sum() / max(1, acc.sum()))
        recorded = report["aggregates"][arm]["K"]["direction_correctness"]
        if abs(recomputed - recorded) > 1e-9:
            raise ValueError("K correctness mismatch - tampered summary")
    return {"report": report, **data}


class AnisoDemo:
    """The erratum bars and the environment verdict."""

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
            text="第五十四课 · 各向异性对照 + 度量勘误——匹配器一直是对的",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "左：旧判据 vs 修正判据（隐含位姿 vs 真值位姿）——旧 0.0/0.6 是评估伪影；"
                "右：修正判据下 ISO 0.722 / ANISO 0.583——环境假设证伪"
            ),
        ).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="erratum")
        for label, key in (("① 度量勘误", "erratum"), ("② 环境判定", "verdict")):
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
        lines = [
            "勘误实证（K 组方向正确率）",
            f"ISO：旧判据 {r['iso_legacy']:.3f} → 修正 {r['iso_corrected']:.3f}",
            f"ANISO：旧判据 {r['aniso_legacy']:.3f} → 修正 {r['aniso_corrected']:.3f}",
            "",
            "内容对齐原理上不可见漂移：旧目标（est 位姿差值）",
            "烤入了 2-9 m 链漂移——要求匹配器复现漂移，",
            "数学上不可达（第 54 课勘误，docs/59）",
            "",
            f"候选池：ISO {r['n_candidates']['ISO']} / ANISO {r['n_candidates']['ANISO']}",
            f"接受率（修正门）：ISO {r['k_accept']['ISO']:.2f} / ANISO {r['k_accept']['ANISO']:.2f}",
        ]
        return "\n".join(lines)

    def draw_erratum(self):
        self.fig.clear()
        agg = self.report["aggregates"]
        ax = self.fig.add_subplot(1, 1, 1)
        arms = ("ISO", "ANISO")
        legacy = [agg[a]["K"]["legacy_direction_correctness"] for a in arms]
        corrected = [agg[a]["K"]["direction_correctness"] for a in arms]
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, legacy, w, color="#9ca3af", label="旧判据（est 位姿差值目标，伪影）")
        ax.bar(x + w / 2, corrected, w, color="#0f766e", label="修正判据（隐含位姿 vs 真值）")
        for i, (lv, co) in enumerate(zip(legacy, corrected, strict=True)):
            ax.text(i - w / 2, lv + 0.01, f"{lv:.3f}", ha="center", fontsize=9)
            ax.text(i + w / 2, co + 0.01, f"{co:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["ISO（方墙）", "ANISO（多边形+杂物）"])
        ax.set_ylabel("K 组方向正确率")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.80, color="k", ls="--", lw=1)
        ax.legend(fontsize=8)
        ax.set_title("度量勘误：同一份匹配器，换判据后从 0.0 升到 0.58-0.72", fontsize=10)

    def draw_verdict(self):
        self.fig.clear()
        agg = self.report["aggregates"]
        ax = self.fig.add_subplot(1, 1, 1)
        groups = GROUPS
        x = np.arange(len(groups))
        w = 0.35
        iso = [agg["ISO"][g]["direction_correctness"] for g in groups]
        aniso = [agg["ANISO"][g]["direction_correctness"] for g in groups]
        ax.bar(x - w / 2, iso, w, color="#2563eb", label="ISO")
        ax.bar(x + w / 2, aniso, w, color="#b91c1c", label="ANISO")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.axhline(0.80, color="k", ls="--", lw=1)
        ax.set_ylabel("方向正确率（修正判据）")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_title("环境判定：ANISO 0.583 不升反略降——环境假设证伪", fontsize=10)

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        if self.mode.get() == "verdict":
            self.draw_verdict()
        else:
            self.draw_erratum()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="lesson-54 aniso demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第五十四课 · 各向异性对照 + 度量勘误")
    root.geometry("1400x740+30+20")
    AnisoDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
