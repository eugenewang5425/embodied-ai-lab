"""Lesson 44 viewer: how pose-estimation error propagates through the stack.

Three static modes share one lesson-44 recording (npz + summary), following
the docs/26 section-6 rules (clear() on every redraw, synchronous draw, every
quoted number traces to summary.json / npz):
1. the estimate-vs-truth error curves of the four groups against step count
   (position error and heading error, observation reset saw-teeth), plus the
   final-error bars per group - HOW bad is the pose error?
2. the misregistered map: the showcase (F) estimated-registration grid over
   the true map, the paired T/O2/F trajectories on the scene, and the grid
   shift per replan - WHAT does the robot believe the world looks like?
3. the verdict: arrival rate, collision events, position error and map shift
   per group from the summary aggregates - WHO survives the broken estimate?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from embodied_learning.experiments.nav_pose_error import (
    DEFAULT_RESULTS,
    EXPERIMENT,
    GROUPS4,
    NAV_GROUPS,
    expected_npz_keys,
)

GROUP_COLORS = {"T": "#0f766e", "O1": "#d97706", "O2": "#b91c1c", "F": "#2563eb"}


def load_replays(directory):
    """Validate a lesson-44 recording; returns the dict the demo reads.

    Tamper routes rejected: (1) experiment/schema identity; (2) npz SHA-256
    against the summary; (3) the archive key set against the implied set;
    (4) per-group/condition arrival and collision counts recomputed from the
    archived metric arrays against the summary aggregates; (5) the archived
    true maps rebuilt from the summary obstacle lists.
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-44 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    for group in GROUPS4:
        for name in ("arrived", "collisions"):
            if data[f"{group}_{name}"].shape != (scenes, inits):
                raise ValueError("Archive metric arrays disagree with the protocol size")
    for group in GROUPS4:
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


def _add_goal(ax, goal_xy, world):
    ax.plot(goal_xy[0], goal_xy[1], marker="*", markersize=13, color="#15803d", linestyle="none")
    ax.plot([0, world, world, 0, 0], [0, 0, world, world, 0], color="#9ca3af", linewidth=0.8)


class NavPoseErrorDemo:
    """Estimate-error propagation, misregistered maps and the group verdict."""

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
        self.subtitle = (
            "三组位姿来源（T 真值 / O1 里程计 1% / O2 里程计 2% / F 2%+地标融合）共享同一场景与初态；"
            "估计进入建图/规划/追踪每一层，射线物理发射于真值位姿，到达与碰撞按真值口径验收"
        )
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十四课 · 定位误差穿栈：建图-规划-追踪在位姿不确定下还成立吗？",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="errors")
        for label, key in (
            ("① 位姿误差曲线", "errors"),
            ("② 错位地图与轨迹", "maps"),
            ("③ 四方对照", "compare"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        slider_box = ttk.Frame(outer)
        slider_box.pack(fill="x")
        ttk.Label(slider_box, text="快照/重规划：").pack(side="left")
        self.slider_value = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            slider_box,
            to=7.0,
            variable=self.slider_value,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(4, 0))
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
                "到达判据 = 第 39 课口径（真值几何：距目标 < 5 cm 且速度 < 0.1 m/s）；"
                "碰撞 = 试探位姿侵入障碍（刚性退回，间隔 ≥ 1 s 去抖）；"
                "地图错位 = 占据格到最近真值格的平均距离（格）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _on_slider(self, _value):
        self.redraw()

    def _slider_index(self, count):
        return int(np.clip(round(float(self.slider_value.get())), 0, count - 1))

    def _sync_slider(self, count):
        self.slider.configure(to=max(0, count - 1))
        index = self._slider_index(max(1, count))
        self.slider_value.set(float(index))
        return index

    def _showcase_row(self):
        return next(
            row
            for row in self.report["episodes"]
            if row["group"] == "F"
            and row["scene"] == self.show_scene
            and row["init"] == self.show_init
        )

    # ------------------------------------------------------------------ modes
    def draw_errors(self):
        """1 position/heading error curves per group + final error bars."""
        self.fig.clear()
        data = self.data
        ax_pos, ax_hdg = self.fig.subplots(1, 2, width_ratios=[1.5, 1.0])
        key = f"{self.show_scene}_{self.show_init}"
        for group in GROUPS4:
            truth = data[f"truth_{group}_{key}"]
            est = data[f"estimates_{group}_{key}"]
            steps = np.arange(len(truth)) * self.dt
            pos = np.linalg.norm(est[:, :2] - truth[:, :2], axis=1)
            ax_pos.plot(
                steps, pos, color=GROUP_COLORS[group], linewidth=1.3, label=NAV_GROUPS[group]
            )
            hdg = np.abs(
                np.arctan2(np.sin(est[:, 2] - truth[:, 2]), np.cos(est[:, 2] - truth[:, 2]))
            )
            ax_hdg.plot(
                steps,
                np.degrees(hdg),
                color=GROUP_COLORS[group],
                linewidth=1.3,
                label=NAV_GROUPS[group],
            )
        ax_pos.set(
            xlabel="时间（s）",
            ylabel="位置误差（m）",
            title=f"①位置误差：估计 vs 真值（场景 {self.show_scene}，初始化 {self.show_init}）",
        )
        ax_hdg.set(
            xlabel="时间（s）",
            ylabel="朝向误差（°）",
            title="朝向误差：T=0 基线；F 每 2 s 观测重置（锯齿）",
        )
        ax_pos.grid(alpha=0.2)
        ax_hdg.grid(alpha=0.2)
        ax_pos.legend(fontsize=7)
        ax_hdg.legend(fontsize=7)
        protocol = self.report["protocol"]
        rows = {g: self._row_for(g) for g in GROUPS4}
        self._mapping_lines = [
            "① 这一幕在看：里程计比例偏差如何变成位姿误差",
            (
                f"平均位置误差：T {rows['T']['mean_position_error_m'] * 100:.1f} cm · "
                f"O1 {rows['O1']['mean_position_error_m'] * 100:.1f} cm · "
                f"O2 {rows['O2']['mean_position_error_m'] * 100:.1f} cm · "
                f"F {rows['F']['mean_position_error_m'] * 100:.1f} cm"
            ),
            f"到达时最终误差：F {rows['F']['final_position_error_m'] * 100:.1f} cm（观测把偏差框到厘米级）",
            f"平均朝向误差：O2 {rows['O2']['mean_heading_error_deg']:.1f}° vs F {rows['F']['mean_heading_error_deg']:.1f}°",
            (
                f"观测周期 {protocol['obs_period_steps'] * protocol['dt_s']:.0f} s，信标 {len(protocol['beacons'])} 个，"
                f"噪声 {protocol['obs_range_std_m'] * 100:.0f} cm / {protocol['obs_bearing_std_rad'] * 180 / math.pi:.1f}°"
            ),
        ]

    def draw_maps(self):
        """2 the believed grid over the true map + T/O2/F paired trajectories."""
        self.fig.clear()
        data = self.data
        scene = self.show_scene
        world = self.world
        ax_map, ax_shift = self.fig.subplots(1, 2, width_ratios=[1.45, 1.0])
        grid = data[f"final_map_{scene}"]
        true_map = data[f"true_map_{scene}"]
        prob = 1.0 / (1.0 + np.exp(-grid))
        ax_map.imshow(
            prob,
            origin="lower",
            extent=[0, world, 0, world],
            cmap="gray",
            vmin=0,
            vmax=1,
            zorder=1,
        )
        ax_map.contour(
            true_map.astype(float),
            levels=[0.5],
            colors="#b91c1c",
            linewidths=0.9,
            extent=[0, world, 0, world],
            origin="lower",
            zorder=5,
        )
        key = f"{scene}_{self.show_init}"
        for group, lw in (("O2", 1.0), ("F", 1.3), ("T", 1.0)):
            truth = data[f"truth_{group}_{key}"]
            ax_map.plot(
                truth[:, 0],
                truth[:, 1],
                color=GROUP_COLORS[group],
                linewidth=lw,
                label=NAV_GROUPS[group],
                zorder=4,
            )
        _add_goal(ax_map, self.report["protocol"]["goal_xy"], world)
        beacons = np.asarray(self.report["protocol"]["beacons"])
        ax_map.plot(
            beacons[:, 0], beacons[:, 1], "^", color="#7c3aed", markersize=5, linestyle="none"
        )
        ax_map.set(
            xlim=(0, world),
            ylim=(0, world),
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"②已知格地图 {scene}（初始化 {self.show_init}）：灰=未知/白=空闲/黑=占据，红轮廓=真值障碍",
        )
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(fontsize=7, loc="upper left")
        # grid shift per scene (init 0) from the metric arrays
        scenes = self.report["protocol"]["scene_count"]
        names = [f"场景 {s}" for s in range(scenes)]
        widths = 0.36
        for offset, group in enumerate(GROUPS4):
            values = [data[f"{group}_map_shift_cells"][s, 0] for s in range(scenes)]
            ax_shift.bar(
                np.arange(scenes) + (offset - 0.5) * widths,
                [v if math.isfinite(v) else 0.0 for v in values],
                widths * 0.92,
                color=GROUP_COLORS[group],
                label=NAV_GROUPS[group],
            )
        ax_shift.set(
            xticks=np.arange(scenes),
            xticklabels=names,
            ylabel="地图占据格平均错位（格）",
            title="②各场景地图错位（初始化 0；T 为切边基线）",
        )
        ax_shift.tick_params(labelsize=8)
        ax_shift.title.set_fontsize(9)
        ax_shift.grid(alpha=0.2, axis="y")
        ax_shift.legend(fontsize=7)
        rows = {g: self._row_for(g) for g in GROUPS4}
        self._mapping_lines = [
            "② 这一幕在看：错位后的知识与配对轨迹（同初态）",
            f"覆盖终止（到达/超时）后：O2 平均位置误差 {rows['O2']['mean_position_error_m']:.2f} m → 地图占据格错位 {rows['O2']['map_shift_cells']:.1f} 格",
            f"F 组：误差 {rows['F']['mean_position_error_m'] * 100:.1f} cm、错位 {rows['F']['map_shift_cells']:.1f} 格（接近 T 组 {rows['T']['map_shift_cells']:.1f} 格切边基线）",
            "红轮廓 = 真值障碍（对照）；紫三角 = 观测信标；轨迹为真值路径（评估口径）",
        ]

    def draw_compare(self):
        """3 arrival/collision/error/shift bars from the summary aggregates."""
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
            for offset, group in enumerate(GROUPS4):
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
                    label=NAV_GROUPS[group],
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
        hypothesis = report["hypothesis"]
        o_obs = aggregates["O1"]["obstacle"]
        o2_obs = aggregates["O2"]["obstacle"]
        f_obs = aggregates["F"]["obstacle"]
        t_obs = aggregates["T"]["obstacle"]
        self._mapping_lines = [
            "③ 这一幕在裁决：定位误差穿栈后谁还活着",
            (
                f"有障碍到达：T {t_obs['arrivals']}/{t_obs['episodes']} · "
                f"O1 {o_obs['arrivals']}/{o_obs['episodes']} · "
                f"O2 {o2_obs['arrivals']}/{o2_obs['episodes']} · "
                f"F {f_obs['arrivals']}/{f_obs['episodes']}"
            ),
            (
                f"碰撞事件：T {t_obs['collision_events_total']} · O1 {o_obs['collision_events_total']} · "
                f"O2 {o2_obs['collision_events_total']} · F {f_obs['collision_events_total']}"
            ),
            f"预注册判据：闸门({'满足' if hypothesis['results']['gate_met'] else '不满足'})、单调({'满足' if hypothesis['results']['monotone_met'] else '不满足'})、融合恢复({'满足' if hypothesis['results']['fusion_met'] else '不满足'})",
            "剩余代价：F 与 T 的差距 = 有界估计的全部代价（见面板与讲义 §4）",
        ]

    def _row_for(self, group):
        return next(
            row
            for row in self.report["episodes"]
            if row["group"] == group
            and row["scene"] == self.show_scene
            and row["init"] == self.show_init
        )

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "errors":
            self.draw_errors()
        elif mode == "maps":
            self.draw_maps()
        else:
            self.draw_compare()
        self.fill_stats(mode)
        status = {
            "errors": "① 位姿误差曲线：T 恒为 0，O1/O2 随转圈增大，F 被 2 s 观测钉住（拖动滑条无效——该模式为静态曲线）",
            "maps": "② 错位地图与轨迹：相信的地图 + 真值轮廓 + 配对轨迹（滑条调节场景轴上的平均误差视图）",
            "compare": "③ 四方对照：到达/碰撞/误差/错位四面板（数字与 summary 逐格一致）",
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
    root.title("第四十四课 · 定位误差穿栈：建图-规划-追踪在位姿不确定下")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    NavPoseErrorDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
