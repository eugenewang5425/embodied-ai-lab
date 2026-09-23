"""Lesson 56 viewer: mapping-localization separation - the world model lesson.

Three static modes share one lesson-56 recording:
1. error curves: EST and distinct particle-filter map inputs over the run;
2. trajectories: the truth chain, the drifting EST chain and the PF chain;
3. kidnap: the five cross-stream kidnap trials (end error bars vs the
   1.0 m recovery line).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.map_localization import (
    DEFAULT_RESULTS,
    EXPERIMENT,
)


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") not in (
        1,
        2,
        3,
        4,
        5,
    ):
        raise ValueError("Incompatible lesson-56 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != set(report["archive_keys"]):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: the recorded means must match the archived curves
    errs = report["hypothesis"]["results"]["errors"]
    if abs(float(np.mean(data["pf_true_err"])) - errs["pf_mean_m"]) > 1e-9:
        raise ValueError("PF mean mismatch - tampered summary")
    if abs(float(np.mean(data["est_err_curve"])) - errs["est_mean_m"]) > 1e-9:
        raise ValueError("EST mean mismatch - tampered summary")
    return {"report": report, **data}


class MapLocDemo:
    """The error curves, the chains and the kidnap trials."""

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
        result = self.report["hypothesis"]["results"]
        errs = result["errors"]
        n_success = sum(bool(t.get("success")) for t in result["kidnap_trials"])
        old_record = self.report["schema_version"] != 5
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第五十六课 · 建图与定位分离：比较地图输入",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                f"平均位置误差：无图 {errs['est_mean_m']:.2f} m，"
                f"理想图 PF {errs['pf_mean_m']:.2f} m，"
                f"自建图 PF {errs.get('pf_self_mean_m', result.get('smear_penalty', float('nan'))):.2f} m；"
                f"理想图绑架恢复 {n_success}/{len(result['kidnap_trials'])}。"
                + ("历史记录：评估口径已勘误。" if old_record else "本轮自建图未达有界定位门槛。")
            ),
        ).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="curves")
        for label, key in (
            ("① 误差曲线", "curves"),
            ("② 轨迹", "chains"),
            ("③ 绑架试验", "kidnap"),
        ):
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
        e = r["errors"]
        lines = [
            "全程对照（无绑架）",
            f"EST（无图漂移链）均值 {e['est_mean_m']:.2f} / 末端 {e['est_final_m']:.2f} m",
            f"PF 理想占据图均值 {e['pf_mean_m']:.2f} / 末端 {e['pf_final_m']:.2f} m",
            f"PF 自建图均值 {e.get('pf_self_mean_m', r.get('smear_penalty', float('nan'))):.2f} m",
            "",
            f"自建表面端点图与理想占据图精确栅格召回 {r['map_hit_rate']:.2f}（仅参考）",
            "",
            "理想图绑架试验（各自恢复窗末判读）",
        ]
        for t in r["kidnap_trials"]:
            lines.append(
                f"  {t['fraction']:.2f}: end {t.get('end_err_m', float('nan')):.2f} m "
                f"success={t.get('success')}"
            )
        if self.report["schema_version"] != 5:
            lines.append("历史记录：采集路线与评分口径已勘误")
        return "\n".join(lines)

    def draw_curves(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(self.data["est_err_curve"], color="#b91c1c", lw=1.2, label="EST 无图漂移链")
        ax.plot(self.data["pf_self_err"], color="#7c3aed", lw=1.0, label="PF 自建图")
        if "pf_sensor_truth_err" in self.data:
            ax.plot(
                self.data["pf_sensor_truth_err"],
                color="#2563eb",
                lw=1.0,
                label="PF 同帧真值位姿投影图",
            )
        ax.plot(self.data["pf_true_err"], color="#0f766e", lw=1.4, label="PF 理想占据图")
        ax.axhline(0.6, color="#0f766e", ls="--", lw=1)
        ax.set_xlabel("帧")
        ax.set_ylabel("位置误差 (m)")
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("位置误差随帧变化：虚线为 0.6 m 目标", fontsize=10)

    def draw_chains(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        truth = self.data["truth_chain"]
        est = self.data["est_chain"]
        pf = self.data["pf_true_chain"]
        ax.plot(truth[:, 0], truth[:, 1], "-", color="k", lw=1.5, label="真值")
        ax.plot(est[:, 0], est[:, 1], "-", color="#b91c1c", lw=1.0, label="EST 漂移链")
        ax.plot(pf[:, 0], pf[:, 1], "-", color="#0f766e", lw=1.0, label="PF 理想占据图")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("同一巡游：真实运动与两条估计轨迹", fontsize=10)

    def draw_kidnap(self):
        self.fig.clear()
        r = self.report["hypothesis"]["results"]
        trials = r["kidnap_trials"]
        labels = [f"{t['fraction']:.2f}" for t in trials]
        ends = [t.get("end_err_m", float("nan")) for t in trials]
        colors = ["#0f766e" if t.get("success") else "#b91c1c" for t in trials]
        ax = self.fig.add_subplot(1, 1, 1)
        ax.bar(labels, ends, color=colors)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        for i, v in enumerate(ends):
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xlabel("绑架时点（巡游进度比例）")
        ax.set_ylabel("恢复窗末误差 (m)")
        ax.set_ylim(0, max(ends) * 1.2)
        n_success = sum(bool(t.get("success")) for t in trials)
        ax.set_title(
            f"理想图绑架重定位 {n_success}/{len(trials)}：各自恢复窗末位置误差", fontsize=10
        )

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        mode = self.mode.get()
        if mode == "chains":
            self.draw_chains()
        elif mode == "kidnap":
            self.draw_kidnap()
        else:
            self.draw_curves()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="lesson-56 map localization demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第五十六课 · 建图-定位分离")
    root.geometry("1400x740+30+20")
    MapLocDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
