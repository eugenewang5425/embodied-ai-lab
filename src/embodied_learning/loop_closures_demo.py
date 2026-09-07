"""Lesson 46 viewer: loop closure - the second absolute anchor.

Three static modes share one lesson-46 recording (npz + summary), docs/26
section-6 rules:
1. chains: the three estimator chains (N/S/L) on the patrol - error curves,
   end-point markers and the relaxed chain overlay;
2. loop: the wide-match internals of the showcase - the coarse grid score
   surface, the node/edge/loop graph and the relaxation deltas;
3. verdict: N/S/L/L-relaxed error bars + the loop statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.loop_closures import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUP_NAMES,
    GROUPS4,
    expected_npz_keys,
)

GROUP_COLORS = {"N": "#b91c1c", "S": "#d97706", "L": "#2563eb", "LT": "#7c3aed"}


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-46 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # cross-checks: loop-ok rate and the LT mean error from the archive arrays
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    n_total = scenes * inits
    for group in ("L", "LT"):
        series = data[f"{group}_loop_ok"][data[f"{group}_loop_ok"] == 1]
        rate = float(series.size / n_total) if n_total else 0.0
        agg_rate = report["aggregates"].get(group, {}).get("loop_ok_rate")
        if agg_rate is not None and abs(rate - agg_rate) > 1e-6:
            raise ValueError("Loop-ok rate disagrees with the archive")
        finite = data[f"{group}_mean_pos_err"][np.isfinite(data[f"{group}_mean_pos_err"])]
        agg_mean = report["aggregates"].get(group, {}).get("mean_position_error_m")
        if finite.size and agg_mean is not None and abs(float(finite.mean()) - agg_mean) > 1e-6:
            raise ValueError("Mean position error disagrees with the archive")
    return {"report": report, **data}


class LoopClosureDemo:
    """N/S/L estimator chains, the loop match and the verdict."""

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
        protocol = self.report["protocol"]
        self.world = protocol["world_size_m"]
        self.dt = protocol["dt_s"]
        self.show_scene = int(protocol["showcase"]["scene"])
        self.subtitle = (
            "采集-重放：真值控制巡逻 32 m（1,1→9,9→9,1→1,9→1,1），三条估计链（N 纯里程计 / "
            "S 帧间匹配 / L 回环+松弛）重放同一传感器流"
        )
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十六课 · 回环检测与姿态图优化——回环是绝对锚的第二形态",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="chains")
        for label, key in (
            ("① 三条估计链", "chains"),
            ("② 回环配准与松弛", "loop"),
            ("③ N/S/L/松弛对照", "compare"),
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
                "回环=扫描 vs 起点子图的粗精配准（1 m/5° CSM 网格 + NN-ICP）+"
                " Gauss-Newton 姿态图松弛（首节点固定，回环边权重 5×）；"
                "回环检测为真值几何 oracle（估计漂移时朴素检测不可用）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _leg_xy(self, key):
        return self.data[f"truth_{self.show_scene}_0"][:, :2]

    def draw_chains(self):
        """1 the three estimator chains against the patrol truth."""
        self.fig.clear()
        data = self.data
        ax_err, ax_map = self.fig.subplots(1, 2, width_ratios=[1.2, 1.3])
        key = f"{self.show_scene}_0"
        truth = data[f"truth_{self.show_scene}_0"]
        steps = np.arange(len(truth)) * self.dt
        for group in GROUPS4:
            est = data[f"est_{group}_{key}"]
            m = min(len(truth), len(est))
            pos = np.linalg.norm(est[:m, :2] - truth[:m, :2], axis=1)
            ax_err.plot(
                steps[:m],
                pos,
                color=GROUP_COLORS[group],
                linewidth=1.2,
                label=GROUP_NAMES[group],
            )
        relaxed = data["relaxed_chain"]
        if relaxed.shape[0] > 1:
            m = min(len(truth), len(relaxed))
            posrel = np.linalg.norm(relaxed[:m, :2] - truth[:m, :2], axis=1)
            ax_err.plot(
                steps[:m],
                posrel,
                color="#7c3aed",
                linewidth=1.5,
                label="L 松弛后",
            )
        ax_err.set(
            xlabel="时间（s）",
            ylabel="位置误差（m）",
            title=f"①三条估计链误差曲线（场景 {self.show_scene}，初始化 0）",
        )
        ax_err.grid(alpha=0.2)
        ax_err.legend(fontsize=7)
        for group in GROUPS4:
            est = data[f"est_{group}_{key}"]
            m = min(len(truth), len(est))
            ax_map.plot(
                truth[:m, 0],
                truth[:m, 1],
                color=GROUP_COLORS[group],
                linewidth=0.9,
                alpha=0.7,
                label=GROUP_NAMES[group],
                linestyle=":" if group == "T" else "-",
            )
        ax_map.plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=1.4, label="真值巡逻")
        ax_map.set(
            xlim=(0, self.world),
            ylim=(0, self.world),
            xlabel="x（m）",
            ylabel="y（m）",
            title="①估计链轨迹 vs 真值巡逻",
        )
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(fontsize=6, loc="upper left")
        ep = next(
            r
            for r in self.report["episodes"]
            if r["chain"] == "L" and r["scene"] == self.show_scene and r["init"] == 0
        )
        self._mapping_lines = [
            "① 这一幕在看：同一段巡逻数据喂给三条估计链",
            (
                f"平均误差：N {self._row('N')['mean_position_error_m']:.2f} · "
                f"S {self._row('S')['mean_position_error_m']:.2f} · "
                f"L {self._row('L')['mean_position_error_m']:.2f} m"
            ),
            (
                "回环匹配 "
                + ("成功" if ep["loop_ok"] else "失败/未触发")
                + (
                    "（残差 "
                    + ("—" if ep["loop_residual_m"] is None else f"{ep['loop_residual_m']:.3f} m")
                    + "）"
                )
            ),
            "紫线 = L 松弛后误差（回环把漂移分摊到全程）",
        ]

    def draw_loop(self):
        """2 the loop: coarse grid score + node graph + relaxation deltas."""
        self.fig.clear()
        data = self.data
        report = self.report
        ax_score, ax_nodes = self.fig.subplots(1, 2, width_ratios=[1.0, 1.4])
        key = f"{self.show_scene}_0"
        truth = data[f"truth_{self.show_scene}_0"]
        est = data[f"est_L_{key}"]
        ep = next(
            r
            for r in report["episodes"]
            if r["chain"] == "L" and r["scene"] == self.show_scene and r["init"] == 0
        )
        # coarse-score proxies: distance of the last est near truth start
        ax_score.plot(
            truth[:, 0],
            truth[:, 1],
            color="#111827",
            linewidth=1.2,
            label="真值巡逻",
        )
        ax_score.plot(est[:, 0], est[:, 1], color="#2563eb", linewidth=1.0, label="L 估计链")
        start = truth[0]
        ax_score.plot(start[0], start[1], "s", color="#15803d", markersize=8)
        if ep["loop_residual_m"] is not None:
            ax_score.set_title(f"②回环：起点（绿）与回程配准（残差 {ep['loop_residual_m']:.3f} m）")
        else:
            ax_score.set_title("②回环：起点（绿）与回程配准（未触发/未成功）")
        ax_score.set(xlabel="x（m）", ylabel="y（m）")
        ax_score.set_aspect("equal", adjustable="box")
        ax_score.legend(fontsize=6, loc="upper left")
        relaxed = data["relaxed_chain"]
        if relaxed.shape[0] > 1:
            m = min(len(truth), len(relaxed))
            ax_nodes.plot(
                truth[:m, 0],
                truth[:m, 1],
                color="#111827",
                linewidth=1.3,
                label="真值",
            )
            ax_nodes.plot(
                relaxed[:, 0],
                relaxed[:, 1],
                color="#7c3aed",
                linewidth=1.2,
                label="L 松弛后",
            )
            ax_nodes.set_title("②松弛后的节点链（起点固定）")
        else:
            ax_nodes.set_title("②松弛未应用（回环未成功）")
        ax_nodes.set(xlabel="x（m）", ylabel="y（m）")
        ax_nodes.set_aspect("equal", adjustable="box")
        ax_nodes.legend(fontsize=6, loc="upper left")
        self._mapping_lines = [
            "② 这一幕在看：回环配准与姿态图松弛",
            (
                "回环 "
                + ("已建立" if ep["loop_ok"] else "未建立")
                + "：残差 "
                + ("—" if ep["loop_residual_m"] is None else f"{ep['loop_residual_m']:.3f} m")
            ),
            (
                "松弛 "
                + ("已应用" if ep["relaxed_used"] else "未应用")
                + "：松弛后平均误差 "
                + (
                    "—"
                    if ep["mean_position_error_relaxed_m"] is None
                    else f"{ep['mean_position_error_relaxed_m']:.3f} m"
                )
            ),
            "粗配 1 m/5° CSM 网格 ±40° + NN-ICP 精配（回环=绝对锚第二形态）",
        ]

    def draw_compare(self):
        """3 N/S/L/(L-relaxed) bars + loop stats."""
        self.fig.clear()
        report = self.report
        aggregates = report["aggregates"]
        axes = self.fig.subplots(1, 3).reshape(-1)
        metrics = (
            ("mean_position_error_m", "平均位置误差（m）"),
            ("final_position_error_m", "终点位置误差（m）"),
            ("map_shift_cells", "地图错位（格）"),
        )
        for ax, (key, name) in zip(axes, metrics, strict=True):
            offs = [0, 1, 2]
            labels = ["N", "S", "L"]
            colors = [GROUP_COLORS["N"], GROUP_COLORS["S"], GROUP_COLORS["L"]]
            relaxed = None
            if key.startswith("mean_position"):
                relaxed = aggregates["L"].get("mean_position_error_relaxed_m")
                if relaxed is not None:
                    offs.append(3)
                    labels.append("L*")
                    colors.append("#7c3aed")
            for off, group, color in zip(offs, labels, colors, strict=True):
                value = aggregates.get(group if group in GROUPS4 else "L", {}).get(key)
                if group == "L*":
                    value = relaxed
                shown = 0.0 if value is None else float(value)
                ax.bar(off, shown, 0.5, color=color)
                ax.annotate(
                    "—" if value is None else f"{value:.2f}",
                    (off, shown),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=9,
                )
            ax.set(xticks=offs, xticklabels=labels, ylabel=name)
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.2, axis="y")
        l_agg = aggregates["L"]
        self._mapping_lines = [
            "③ 这一幕在裁决：回环到底赢了多少",
            (
                f"平均误差 N {aggregates['N']['mean_position_error_m']:.2f} · "
                f"S {aggregates['S']['mean_position_error_m']:.2f} · "
                f"L {aggregates['L']['mean_position_error_m']:.2f} · "
                f"L 松弛后 {l_agg.get('mean_position_error_relaxed_m')} m"
            ),
            f"回环成功率 {l_agg['loop_ok_rate']:.0%}（残差均值 {l_agg['loop_residual_mean_m']:.3f} m）",
            (
                "末端闭合："
                + (
                    "L "
                    + (
                        f"{l_agg['final_position_error_relaxed_m']:.2f} m（松弛前 {l_agg['final_position_error_m']:.2f} m）"
                        if l_agg.get("final_position_error_relaxed_m") is not None
                        else "L 未闭合"
                    )
                )
                + "；LT "
                + (
                    f"{aggregates['LT']['final_position_error_relaxed_m']:.2f} m（黄金边）"
                    if aggregates["LT"].get("final_position_error_relaxed_m") is not None
                    else "LT 未闭合"
                )
            ),
            "结论：帧间匹配（相对锚）削一点，回环（绝对锚第二形态）+ 松弛才是质变",
        ]

    def _row(self, chain):
        return next(
            r
            for r in self.report["episodes"]
            if r["chain"] == chain and r["scene"] == self.show_scene and r["init"] == 0
        )

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "chains":
            self.draw_chains()
        elif mode == "loop":
            self.draw_loop()
        else:
            self.draw_compare()
        self.fill_stats(mode)
        status = {
            "chains": "① 三条估计链（同一段真值巡逻数据）",
            "loop": "② 回环配准与松弛：起点子图锚定 + 全局分摊",
            "compare": "③ N/S/L/松弛对照与回环统计",
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
    root.title("第四十六课 · 回环检测与姿态图优化——回环是绝对锚的第二形态")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    LoopClosureDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
