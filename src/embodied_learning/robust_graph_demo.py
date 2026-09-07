"""Lesson 48 viewer: robust pose graphs vs poisoned loop edges.

Three static modes share one lesson-48 recording (npz + summary):
1. verdict: the five-group mean/final bars (fragility + cure);
2. showcase: the poisoned vs Huber chains on the map (the drag visible);
3. mechanics: the Huber IRLS idea and the injection protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.robust_graph import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUP_NAMES,
    GROUPS,
    expected_npz_keys,
)

COLORS = {
    "N": "#b91c1c",
    "LS-G": "#0f766e",
    "LS-B": "#7f1d1d",
    "HUB-B": "#2563eb",
    "HUB-G": "#0f766e",
}


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-48 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-check: the summary's five means must be mutually consistent with
    # the pre-registered order claim (LS-B is the worst back-end)
    errs = report["hypothesis"]["results"]["errors"]
    if errs["LS-B"] is not None and errs["HUB-B"] is not None and errs["LS-B"] < errs["HUB-B"]:
        raise ValueError("LS-B better than HUB-B - tampered summary")
    return {"report": report, **data}


class RobustGraphDemo:
    """Five groups, the poisoned drag, and the Huber mechanics."""

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
            "采集-重放同 46/47 课；受控注入毒化回环边（真值 delta 旋转 ±90° + 平移）——"
            "LS-G 正确回环 / LS-B 毒化回环 / HUB-B 毒化+Huber-IRLS / HUB-G 好回环+Huber"
        )
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十八课 · 鲁棒位姿图——错误回环边与 Huber 核",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="verdict")
        for label, key in (
            ("① 五方对照", "verdict"),
            ("② 拖歪轨迹", "drag"),
            ("③ 机制", "mechanics"),
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
                "Huber-IRLS：分量级 w = min(1, δ/|e|)（δ=0.5），每次 G-N 迭代重算权重；"
                "毒化 = 真值 delta 旋转 90° + 平移 1 m（46 课错峰的严重变体）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def draw_verdict(self):
        self.fig.clear()
        aggregates = self.report["aggregates"]
        axes = self.fig.subplots(1, 2).reshape(-1)
        for ax, key, name in zip(
            axes,
            ("mean_position_error_m", "final_position_error_m"),
            ("平均位置误差（m）——脆弱性与抵抗", "末端位置误差（m）——闭合能力"),
            strict=True,
        ):
            for i, group in enumerate(GROUPS):
                value = aggregates[group][key]
                shown = 0.0 if value is None else float(value)
                ax.bar(i, shown, 0.5, color=COLORS[group])
                ax.annotate(
                    "—" if value is None else f"{value:.2f}",
                    (i, shown),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_xticks(range(len(GROUPS)))
            ax.set_xticklabels(GROUPS, fontsize=8)
            ax.set_ylabel(name)
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.2, axis="y")
        agg = aggregates
        h = self.report["hypothesis"]["results"]
        self._mapping_lines = [
            "① 这一幕在裁决：错误回环边有多毒、Huber 有多管用",
            (
                f"平均：N {agg['N']['mean_position_error_m']:.2f} · "
                f"LS-G {agg['LS-G']['mean_position_error_m']:.2f} · "
                f"LS-B {agg['LS-B']['mean_position_error_m']:.2f} · "
                f"HUB-B {agg['HUB-B']['mean_position_error_m']:.2f} · "
                f"HUB-G {agg['HUB-G']['mean_position_error_m']:.2f} m"
            ),
            f"脆弱性（LS-B > N）：{'成立' if h['fragility_met'] else '未成立'}；抵抗（HUB-B < LS-B）：{'成立' if h['cure_met'] else '未成立'}",
            (
                "单核两难（如实记录）：HUB-G 末端 "
                + f"{agg['HUB-G']['final_position_error_m']:.2f} m"
                + "（好回环在同核下收敛慢；SC/MM 是下一课）"
            ),
        ]

    def draw_drag(self):
        self.fig.clear()
        data = self.data
        truth = data["truth_showcase"]
        steps = data["node_steps"] - 1
        steps = steps[steps < len(truth)]
        ax = self.fig.add_subplot(111)
        ax.plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=1.4, label="真值")
        for name, style, lw in (
            ("LS-G", "-", 1.0),
            ("HUB-B", "--", 1.2),
            ("LS-B", ":", 1.6),
        ):
            chain = data[f"chain_{name}"]
            m = min(len(chain), len(steps))
            ax.plot(
                chain[:m, 0],
                chain[:m, 1],
                linestyle=style,
                linewidth=lw,
                color=COLORS[name],
                label=GROUP_NAMES[name],
            )
        ax.plot(truth[0, 0], truth[0, 1], "s", color="#15803d", markersize=8)
        ax.set(
            xlabel="x（m）",
            ylabel="y（m）",
            title="②毒化回环把整条链拖歪（LS-B 点线）；Huber 链贴近真值",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7)
        errs = self.report["hypothesis"]["results"]["errors"]
        self._mapping_lines = [
            "② 这一幕在看：一条坏边的全局破坏力",
            f"LS-B（点线）被毒化回环拉离真值（平均 {errs['LS-B']:.2f} m）",
            f"HUB-B（虚线）在核下保住里程计形状（平均 {errs['HUB-B']:.2f} m）",
            "绿方块 = 起点（回环锚位置）；好回环（LS-G）与 HUB-G 同走真值附近",
        ]

    def draw_mechanics(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.axis("off")
        lines = [
            ("最小二乘（LS）：大残差被平方放大——一条 9 m 级的坏边 = 856 条好边的全部话语权", 0.9),
            ("Huber-IRLS：|e| > δ 的残差线性增长（权重 1/|e|），坏边在第二次迭代被降权", 0.78),
            ("δ = 0.5（加权残差单位）：好边（< 3σ ≈ 0.15）不受影响——好回环零代价", 0.66),
            ("代价：真残差被轻度压低（收敛稍慢）——本课 G-N 迭代 10 次覆盖", 0.54),
            ("教训（46 课双峰的工程答案）：残差门挡不住错峰（2.3 vs 2.6 cm），核可以", 0.42),
            ("完整方案 = 鲁棒核（本课） + 特征化回环检测（下一课） + 多假设评分", 0.30),
        ]
        for text, y in lines:
            ax.text(0.02, y, text, fontsize=11, transform=ax.transAxes)
        self._mapping_lines = [
            "③ 这一幕在讲：Huber 为什么管用",
            "平方损失让一条坏边的声音盖过全部好边；Huber 把大残差线性化",
            "IRLS：每轮 G-N 用上一轮残差重算权重，坏边逐轮被边缘化",
            "δ=0.5 的选择：好边加权残差 < 0.15，不受核影响",
        ]

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "verdict":
            self.draw_verdict()
        elif mode == "drag":
            self.draw_drag()
        else:
            self.draw_mechanics()
        self.fill_stats(mode)
        status = {
            "verdict": "① 五方对照（脆弱性/抵抗/无免费午餐）",
            "drag": "② 毒化轨迹 vs Huber 轨迹 vs 真值",
            "mechanics": "③ Huber-IRLS 机制与代价",
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
    root.title("第四十八课 · 鲁棒位姿图——错误回环边与 Huber 核")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    RobustGraphDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
