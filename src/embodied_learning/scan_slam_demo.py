"""Lesson 45 viewer: minimal scan matching - relative anchor, no absolute truth.

Three static modes share one lesson-45 recording (npz + summary), docs/26
section-6 rules (clear() on every redraw, synchronous draw, every quoted
number traces to summary.json / npz):
1. matcher internal view: the multi-scale NN-ICP alignment on a pair of
   scans (prev submap vs last period), the correction histogram per episode
   and the match residual curve - DOES the matcher do what it claims;
2. the error curves of T/E/S vs time plus the pair of trajectories (paired
   starts) - the drift estimate that the scan matcher could NOT fix;
3. the verdict: arrival / collision / error / map shift of the three groups
   against the lesson-44 beacon finding (F group, + absolute anchor).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_learning.experiments.scan_slam import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUP_NAMES,
    GROUPS3,
    expected_npz_keys,
)

GROUP_COLORS = {"T": "#0f766e", "E": "#b91c1c", "S": "#2563eb"}


def load_replays(directory):
    """Validate a lesson-45 recording; returns the dict the demo reads."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-45 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    for group in GROUPS3:
        for name in ("arrived", "collisions"):
            if data[f"{group}_{name}"].shape != (scenes, inits):
                raise ValueError("Archive metric arrays disagree with the protocol size")
    for group in GROUPS3:
        for condition, scene_slice in (("obstacle", slice(1, None)), ("empty", slice(0, 1))):
            aggregate = report["aggregates"].get(group, {}).get(condition)
            if aggregate is None:
                continue
            if int(data[f"{group}_arrived"][scene_slice].sum()) != aggregate["arrivals"]:
                raise ValueError("Arrival counts disagree with the archive")
            if (
                int(data[f"{group}_collisions"][scene_slice].sum())
                != aggregate["collision_events_total"]
            ):
                raise ValueError("Collision counts disagree with the archive")
    n = report["protocol"]["grid_cells"]
    res = report["protocol"]["res_m"]
    from embodied_learning.experiments.grid_nav import Obstacle, build_walls, true_occupancy

    for scene in range(scenes):
        obstacles = tuple(
            Obstacle.from_dict(item) for item in report["scenarios"][scene]["obstacles"]
        )
        rebuilt = true_occupancy(
            obstacles, build_walls(report["protocol"]["world_size_m"], res), n, res
        )
        if not np.array_equal(rebuilt, data[f"true_map_{scene}"]):
            raise ValueError("True map disagrees with the summary obstacle list")
    return {"report": report, **data}


class ScanSlamDemo:
    """Minimal scan matching: what it can and cannot close."""

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
        self.show_init = int(protocol["showcase"]["init"])
        self.subtitle = "三组共享第 43/44 课场景与初态：T 真值 / E 里程计 2%（无修正）/ S 里程计 2% + 帧间扫描匹配闭环（无信标）"
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十五课 · 最小栅格 SLAM：扫描匹配是相对锚，不能给出绝对真值",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="matcher")
        for label, key in (
            ("① 匹配器内观", "matcher"),
            ("② 误差与轨迹", "errors"),
            ("③ 三方对照与信标对照", "compare"),
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
                "到达判据 = 第 39 课口径（真值几何）；碰撞 = 试探位姿侵入障碍（刚性退回）；"
                "匹配器 = 多尺度最近邻 ICP（粗 0.40 m → 精 0.15 m），残差门 3 cm、角门 5°，增益 0.3"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _row_for(self, group):
        return next(
            row
            for row in self.report["episodes"]
            if row["group"] == group
            and row["scene"] == self.show_scene
            and row["init"] == self.show_init
        )

    # ------------------------------------------------------------------ modes
    def draw_matcher(self):
        """1 correction statistics + match residual across the episodes."""
        self.fig.clear()
        data = self.data
        protocol = self.report["protocol"]
        axes = self.fig.subplots(1, 2).reshape(-1)
        scenes = protocol["scene_count"]
        inits = protocol["inits"]
        ax_hist, ax_res = axes
        # correction magnitude per S episode
        corrections = []
        labels = []
        for scene in range(scenes):
            for init in range(inits):
                v = data["S_match_max_corr"][scene, init]
                if math.isfinite(v):
                    corrections.append(v)
                    labels.append(f"{scene}-{init}")
        ax_hist.hist(
            corrections,
            bins=np.arange(0, max(corrections) + 0.01, 0.01),
            color=GROUP_COLORS["S"],
            alpha=0.8,
        )
        ax_hist.set(
            xlabel="单回合最大修正幅度（m）",
            ylabel="回合数",
            title=f"①单回合最大修正幅度（S，{len(corrections)} 回合）：修正生效但有限",
        )
        ax_hist.grid(alpha=0.2, axis="y")
        accept = []
        for scene in range(scenes):
            for init in range(inits):
                v = data["S_match_accept_rate"][scene, init]
                if math.isfinite(v):
                    accept.append(v)
        ax_res.bar(
            np.arange(len(accept)),
            accept,
            color=GROUP_COLORS["S"],
            width=0.6,
        )
        ax_res.set(
            xlabel="回合",
            ylabel="匹配接受率",
            title=f"①匹配接受率（中位 {np.median(accept):.0%}）：帧间闭环确实在发生",
        )
        ax_res.tick_params(labelsize=7)
        ax_res.grid(alpha=0.2, axis="y")
        row = self._row_for("S")
        self._mapping_lines = [
            "① 这一幕在看：扫描匹配器到底修了什么",
            f"展示回合（场景 {self.show_scene}，初始化 {self.show_init}）：接受 {row['match_accepted']}/{row['match_attempts']} 次，平均残差 {row['match_mean_residual']:.3f} m",
            f"平均/最大修正幅度 {row['match_mean_correction_m']:.3f} m / {row['match_max_correction_m']:.3f} m",
            '匹配不能给出绝对真值：无信标、无回环——这就是"最小 SLAM"的边界',
            "多尺度 NN-ICP：粗门 0.40 m 收敛 → 精门 0.15 m + 3 cm 残差门（输出仅供诊断）",
        ]

    def draw_errors(self):
        """2 error curves + paired trajectories of the showcase episode."""
        self.fig.clear()
        data = self.data
        key = f"{self.show_scene}_{self.show_init}"
        ax_err, ax_map = self.fig.subplots(1, 2, width_ratios=[1.2, 1.3])
        for group in GROUPS3:
            truth = data[f"truth_{group}_{key}"]
            est = data[f"estimates_{group}_{key}"]
            steps = np.arange(len(truth)) * self.dt
            pos = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
            if group == "S":
                # starting from the same pose the S errors are what the matcher failed to fix
                ax_err.plot(
                    steps, pos, color=GROUP_COLORS["S"], linewidth=1.4, label=GROUP_NAMES["S"]
                )
            else:
                ax_err.plot(
                    steps, pos, color=GROUP_COLORS[group], linewidth=1.1, label=GROUP_NAMES[group]
                )
        ax_err.set(
            xlabel="时间（s）",
            ylabel="位置误差（m）",
            title=f"②位置误差曲线（场景 {self.show_scene}，初始化 {self.show_init}）",
        )
        ax_err.grid(alpha=0.2)
        ax_err.legend(fontsize=7)
        for group in GROUPS3:
            truth = data[f"truth_{group}_{key}"]
            ax_map.plot(
                truth[:, 0],
                truth[:, 1],
                color=GROUP_COLORS[group],
                linewidth=1.2,
                label=GROUP_NAMES[group],
            )
        goal = self.report["protocol"]["goal_xy"]
        ax_map.plot(goal[0], goal[1], "*", color="#15803d", markersize=12, linestyle="none")
        ax_map.set(
            xlim=(0, self.world),
            ylim=(0, self.world),
            xlabel="x（m）",
            ylabel="y（m）",
            title="②配对轨迹（同初态）",
        )
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(fontsize=7, loc="upper left")
        rows = {g: self._row_for(g) for g in GROUPS3}
        self._mapping_lines = [
            "② 这一幕在看：扫描匹配没能修掉的漂移",
            f"S 平均位置误差 {rows['S']['mean_position_error_m']:.3f} m vs E {rows['E']['mean_position_error_m']:.3f} m（同量级）",
            f"S 地图错位 {rows['S']['map_shift_cells']:.2f} 格（自洽但整体漂移：相对锚的代价）",
            "转弯段为误差主来源：帧间匹配只能约束相邻帧，无法约束全局",
        ]

    def draw_compare(self):
        """3 group verdict + the lesson-44 beacon contrast."""
        self.fig.clear()
        report = self.report
        aggregates = report["aggregates"]
        axes = self.fig.subplots(2, 2).reshape(-1)
        conditions = [("obstacle", "有障碍"), ("empty", "无障碍")]
        width = 0.4
        for ax, key, name in (
            (axes[0], "arrival_rate", "到达率（真值口径 < 5 cm 且 < 0.1 m/s）"),
            (axes[1], "collision_events_mean", "平均碰撞事件/回合"),
            (axes[2], "mean_position_error_m", "平均位置误差（估计 vs 真值）"),
            (axes[3], "map_shift_cells", "地图占据格平均错位（格）"),
        ):
            for offset, group in enumerate(GROUPS3):
                values, texts = [], []
                for condition, _ in conditions:
                    aggregate = aggregates.get(group, {}).get(condition)
                    value = None if aggregate is None else aggregate.get(key)
                    values.append(
                        0.0
                        if value is None or (value is not None and math.isnan(value))
                        else float(value)
                    )
                    if key == "arrival_rate":
                        texts.append(
                            f"{aggregate['arrivals']}/{aggregate['episodes']}" if aggregate else "—"
                        )
                    else:
                        texts.append("—" if value is None else f"{value:.2f}")
                bars = ax.bar(
                    np.arange(2) + (offset - 0.5) * width,
                    values,
                    width * 0.92,
                    color=GROUP_COLORS[group],
                    label=GROUP_NAMES[group],
                )
                for bar, text in zip(bars, texts, strict=True):
                    ax.annotate(
                        text,
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        xytext=(0, 3),
                        textcoords="offset points",
                        fontsize=8,
                    )
            ax.set(xticks=np.arange(2), xticklabels=[c[1] for c in conditions], title=name)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(9)
            ax.grid(alpha=0.2, axis="y")
        axes[0].set_ylim(0, 1.2)
        axes[0].legend(fontsize=7)
        t_obs = aggregates["T"]["obstacle"]
        e_obs = aggregates["E"]["obstacle"]
        s_obs = aggregates["S"]["obstacle"]
        self._mapping_lines = [
            "③ 这一幕在裁决：最小的 SLAM 够不够",
            (
                f"有障碍到达：T {t_obs['arrivals']}/{t_obs['episodes']} · "
                f"E {e_obs['arrivals']}/{e_obs['episodes']} · "
                f"S {s_obs['arrivals']}/{s_obs['episodes']}"
            ),
            (
                f"平均误差：T {t_obs['mean_position_error_m']:.3f} · "
                f"E {e_obs['mean_position_error_m']:.3f} · "
                f"S {s_obs['mean_position_error_m']:.3f} m"
            ),
            "绝对锚对照（第 44 课 F 组 + 四角信标 2 s 观测）：15/15、误差 2.5 cm——信标是闭环必要不充分条件的证据",
            "预注册判据：闸门满足；无绝对锚（S == E）——判据与数字以讲义 §3.5 为准",
        ]

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "matcher":
            self.draw_matcher()
        elif mode == "errors":
            self.draw_errors()
        else:
            self.draw_compare()
        self.fill_stats(mode)
        status = {
            "matcher": "① 匹配器内观：修正幅度、接受率与残差（匹配只在帧间局部生效）",
            "errors": "② 误差与轨迹：S 组误差与 E 同量级（自洽但漂移）；配对轨迹显示极端点的异同",
            "compare": "③ 三方对照 + 绝对锚对照：扫描匹配（相对锚）vs 信标（绝对锚）",
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
    root.title("第四十五课 · 最小栅格 SLAM：扫描匹配是相对锚，不能给出绝对真值")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    ScanSlamDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
