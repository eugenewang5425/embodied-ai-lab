"""Lesson 51 viewer: switchable constraints - the lambda-sensitivity curve
and the max-mixture stall (the single-kernel dilemma's second freedom).

Two static modes share one lesson-51 recording:
1. curve: the lambda sweep - SC-G final / SC-B mean vs lambda with the two
   acceptance lines (the sweet spot);
2. verdict: the dilemma chain (48's HUB-G vs SC-G sweet vs MM stall) bars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.switchable_graph import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    SC_LAMBDAS,
)


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-51 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != set(report["archive_keys"]):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    return {"report": report, **data}


class SwitchableDemo:
    """The lambda curve and the dilemma verdict."""

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
            text="第五十一课 · 可切换约束（SC）——单一 λ 无解，中间区间是工程答案",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "第 48 课单核两难（Huber 杀好边/扛毒化不可兼得）需要第二个自由度："
                "SC 每个环边一个开关 s。SC 的 λ 敏感性扫描：λ 小→好边被开关杀死；"
                "λ 大→毒化被保留并拖拽——中间区间（~50）同时满足两线"
            ),
        ).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="curve")
        for label, key in (("① λ 敏感性曲线", "curve"), ("② 两难对决", "verdict")):
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

    def _curve_entry(self, lam):
        r = self.report["hypothesis"]["results"]
        curve = r["lambda_curve"]
        return curve.get(f"{lam:g}", curve.get(str(lam)))

    def stats_text(self):
        r = self.report["hypothesis"]["results"]
        r["lambda_curve"]
        lines = ["治愈组（SC-G）末端误差 / 预防组（SC-B）均值", ""]
        for lam in SC_LAMBDAS:
            c = self._curve_entry(lam)
            lines.append(f"λ={lam:>4}: {c['sc_g_final']:.2f} m / {c['sc_b_mean']:.2f} m")
        lines.append("")
        lines.append(f"甜区 λ* = {r['sweet_spot_lambdas']}")
        lines.append(f"MM 死锁记录: {r['mm_stall_recorded']}")
        lines.append(f"MM valid 占比 G/B: {r['valid_fractions']}")
        return "\n".join(lines)

    def draw_curve(self):
        self.fig.clear()
        r = self.report["hypothesis"]["results"]
        curve = r["lambda_curve"]
        xs = SC_LAMBDAS
        g_final = [
            curve[f"{lam:g}" if f"{lam:g}" in curve else str(lam)]["sc_g_final"] for lam in xs
        ]
        b_mean = [curve[f"{lam:g}" if f"{lam:g}" in curve else str(lam)]["sc_b_mean"] for lam in xs]
        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(xs, g_final, "o-", color="#0f766e", label="SC-G 末端误差（治疗线 1.5 m）")
        ax.plot(xs, b_mean, "s-", color="#b91c1c", label="SC-B 均值（预防线 = N+0.5）")
        ax.axhline(1.5, color="#0f766e", ls="--", lw=1)
        n_mean = self.report["aggregates"]["N"]["mean_position_error_m"]
        ax.axhline(n_mean + 0.5, color="#b91c1c", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("λ（开关先验权重，log）")
        ax.set_ylabel("误差 (m)")
        ax.legend(fontsize=8)
        ax.set_title("SC λ 敏感性：两种失效（λ小杀好边 / λ大留毒化）与中间甜区", fontsize=10)

    def draw_verdict(self):
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        self.report["hypothesis"]["results"]
        finals = {"HUB-G (48 单核)": self.report["aggregates"]["HUB-G"]["final_position_error_m"]}
        for lam in (50.0,):
            finals[f"SC-G λ={lam:g}"] = self._curve_entry(lam)["sc_g_final"]
        finals["MM-G (死锁)"] = self.report["aggregates"]["MM-G"]["final_position_error_m"]
        finals["LS-G (参考)"] = self.report["aggregates"]["LS-G"]["final_position_error_m"]
        ax.bar(
            list(finals),
            [v for v in finals.values()],
            color=["#9ca3af", "#0f766e", "#b91c1c", "#2563eb"],
        )
        ax.axhline(1.5, color="k", ls="--", lw=1)
        ax.set_ylabel("末端误差 (m)")
        ax.set_ylim(0, max(finals.values()) * 1.15)
        ax.set_title("第 48 课单核两难的各解法末端闭合：SC 甜区 ≈ LS，MM 硬选择死锁", fontsize=10)

    def redraw(self):
        self.stats.configure(text=self.stats_text())
        if self.mode.get() == "verdict":
            self.draw_verdict()
        else:
            self.draw_curve()
        self.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description="lesson-51 switchable demo")
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    import tkinter as tk

    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第五十一课 · 可切换约束")
    root.geometry("1400x740+30+20")
    SwitchableDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
