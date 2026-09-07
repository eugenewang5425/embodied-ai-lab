"""Lesson 43 viewer: occupancy-grid mapping + A* planning + pure pursuit.

Three static modes share one lesson-43 recording (npz + summary), following
the docs/26 section-6 rules (every mode starts from fig.clear(), redraw draws
synchronously, every quoted number traces to summary.json / npz):
1. the mapping process: log-odds snapshots of the showcase episode with the
   live sensor rays, the true obstacle outline for reference and the
   coverage curve - how does an unknown 0.5 grid become a map?
2. planning + tracking: the A* path at each replanning instant against the
   true map and the tracked trajectory up to that step, plus the path-cost
   curve over replans;
3. blind versus planned: paired trajectories, arrival rates, collision
   events and path-length ratios from the summary aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.grid_nav import (
    EXPERIMENT,
    GROUPS,
    SNAPSHOTS,
    Obstacle,
    build_walls,
    expected_npz_keys,
    true_occupancy,
)

DEFAULT_RESULTS = "results/grid_nav_2026-09-06_v2"  # the v2 record (see docs/48 §7 erratum)
GROUP_LABELS = {"A": "A 盲飞（第 21 课控制器）", "B": "B 建图 + A* 规划 + 纯追踪"}


def load_replays(directory):
    """Validate a lesson-43 recording; returns the dict the demo reads.

    Tamper routes rejected: (1) experiment/schema identity; (2) npz SHA-256
    against the summary; (3) the archive key set against the implied set;
    (4) per-group/condition arrival and collision counts recomputed from the
    archived metric arrays against the summary aggregates; (5) the archived
    true maps rebuilt from the summary obstacle lists.
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-43 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    for group in GROUPS:
        for name in ("arrived", "collisions"):
            if data[f"{group}_{name}"].shape != (scenes, inits):
                raise ValueError("Archive metric arrays disagree with the protocol size")
    for group in GROUPS:
        for condition, scene_slice in (
            ("obstacle", slice(1, None)),
            ("empty", slice(0, 1)),
        ):
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
    world = report["protocol"]["world_size_m"]
    for scene in range(scenes):
        obstacles = tuple(
            Obstacle.from_dict(item) for item in report["scenarios"][scene]["obstacles"]
        )
        rebuilt = true_occupancy(obstacles, build_walls(world, res), n, res)
        if not np.array_equal(rebuilt, data[f"true_map_{scene}"]):
            raise ValueError("True map disagrees with the summary obstacle list")
    showcase = report["protocol"]["showcase"]
    if data["snap_grids"].shape != (SNAPSHOTS, n, n) or data["snap_poses"].shape != (SNAPSHOTS, 3):
        raise ValueError("Snapshot arrays disagree with the protocol size")
    if int(showcase["scene"]) >= scenes or int(showcase["init"]) >= inits:
        raise ValueError("Showcase episode outside the recorded protocol")
    return {"report": report, **data}


def _probabilities(log_odds):
    return 1.0 / (1.0 + np.exp(-np.asarray(log_odds, dtype=float)))


def _add_goal(ax, goal_xy, world):
    ax.plot(goal_xy[0], goal_xy[1], marker="*", markersize=13, color="#15803d", linestyle="none")
    ax.plot([0, world, world, 0, 0], [0, 0, world, world, 0], color="#9ca3af", linewidth=0.8)


def _add_true_map(ax, true_map, world, alpha=0.45):
    ax.imshow(
        true_map.astype(float),
        origin="lower",
        extent=[0, world, 0, world],
        cmap="gray_r",
        alpha=alpha,
        vmin=0,
        vmax=1,
        zorder=1,
    )


def _draw_car(ax, pose, size=0.16, color="#111827"):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    body = np.array([(size, 0.0), (-0.7 * size, 0.55 * size), (-0.7 * size, -0.55 * size)])
    points = body @ np.array([[c, s], [-s, c]]).T + np.asarray(pose[:2])
    ax.fill(points[:, 0], points[:, 1], color=color, zorder=6)


class GridNavDemo:
    """Occupancy mapping, A* tracking and the blind-vs-planned comparison."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and labels
        self.root = root
        self.data = data
        self.report = data["report"]
        protocol = self.report["protocol"]
        self.world = protocol["world_size_m"]
        self.subtitle = (
            f"B 组：16 射线测距 → Bresenham 对数概率建图 → 已知栅格 A* 重规划"
            f"（膨胀 {protocol['inflate_cells']} 格 = {protocol['inflate_m']:.2f} m）→ 纯追踪；"
            "A 组：第 21 课盲飞。两组共享起点扰动（配对）。三模式共用同一正式记录，全部为静态图"
        )
        self.show_scene = int(protocol["showcase"]["scene"])
        self.show_init = int(protocol["showcase"]["init"])
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十三课 · 占据栅格建图 + A* 路径规划 + 纯追踪（感知-建图-规划主线）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(outer, text=self.subtitle).pack(anchor="w", pady=(2, 4))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="mapping")
        for label, key in (
            ("① 占据栅格建图过程", "mapping"),
            ("② A* 路径 + 小车跟踪", "tracking"),
            ("③ 盲飞 vs 规划对照", "compare"),
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
            from_=0,
            to=SNAPSHOTS - 1,
            variable=self.slider_value,
            command=self._on_slider,
        )
        self.slider.pack(side="left", fill="x", expand=True, padx=(4, 0))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(6, 0))
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left", wraplength=360)
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
                "到达判据 = 第 39 课口径（距目标 < 5 cm 且速度 < 0.1 m/s）；碰撞 = 试探位姿侵入障碍"
                "（刚性退回，间隔 ≥ 1 s 去抖）；未知格按可通行规划，逐步重规划 + 前向刹车兜底"
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
        index = self._slider_index(count)
        self.slider_value.set(float(index))
        return index

    # ------------------------------------------------------------------ modes
    def draw_mapping(self):
        """1 occupancy snapshots: probability grid, rays, true outline, coverage."""
        self.fig.clear()
        data = self.data
        grids = data["snap_grids"]
        index = self._sync_slider(len(grids))
        grid = grids[index]
        pose = data["snap_poses"][index]
        ranges = data["snap_ranges"][index]
        hits = data["snap_hits"][index]
        cov = float(data["snap_cov"][index])
        step = int(data["snap_steps"][index])
        world = self.world
        ax_map, ax_cov = self.fig.subplots(1, 2, width_ratios=[1.45, 1.0])
        ax_map.imshow(
            _probabilities(grid),
            origin="lower",
            extent=[0, world, 0, world],
            cmap="gray",
            vmin=0,
            vmax=1,
            zorder=1,
        )
        ax_map.contour(
            data[f"true_map_{self.show_scene}"].astype(float),
            levels=[0.5],
            colors="#b91c1c",
            linewidths=0.9,
            extent=[0, world, 0, world],
            origin="lower",
        )
        angles = pose[2] + np.arange(len(ranges)) * (2.0 * np.pi / len(ranges))
        ends = np.asarray(pose[:2]) + ranges[:, None] * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1
        )
        for start, end in zip(np.repeat(pose[None, :2], len(ranges), axis=0), ends, strict=True):
            ax_map.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="#f59e0b",
                linewidth=0.6,
                alpha=0.75,
                zorder=4,
            )
        _draw_car(ax_map, pose)
        goal = self.report["protocol"]["goal_xy"]
        ax_map.plot(goal[0], goal[1], "*", color="#15803d", markersize=12, zorder=6)
        ax_map.plot(
            self.report["protocol"]["start_xy"][0],
            self.report["protocol"]["start_xy"][1],
            "s",
            color="#111827",
            markersize=5,
            zorder=6,
        )
        ax_map.set(
            xlim=(0, world),
            ylim=(0, world),
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"①建图快照 {index + 1}/{len(grids)}（第 {step} 步）：灰=未知 0.5，白=空闲，黑=占据",
        )
        ax_map.set_aspect("equal", adjustable="box")
        steps = data["snap_steps"]
        ax_cov.plot(steps, data["snap_cov"], "o-", color="#0f766e", markersize=4)
        ax_cov.plot(step, cov, "o", color="#b45309", markersize=8)
        ax_cov.set(
            xlabel="步（0.04 s）",
            ylabel="建图覆盖率（已知格占比）",
            title="覆盖率随建图进程增长（红轮廓 = 真值障碍，仅对照）",
            ylim=(0, 1.0),
        )
        ax_cov.grid(alpha=0.2)
        self._mapping_lines = [
            f"① 这一幕在看：未知 0.5 如何变成地图（场景 {self.show_scene}，初始化 {self.show_init}）",
            f"快照 {index + 1}/{len(grids)}：第 {step} 步（{step * self.report['protocol']['dt_s']:.1f} s）",
            f"16 射线命中 {int(hits.sum())}/16（琥珀色），最大量程 2 m",
            f"当前覆盖率 {cov:.1%}",
            f"本回合终值覆盖率 {self._showcase_coverage():.1%}（正式记录）",
            "橙色射线 = 本步传感器；红星 = 目标；黑方块 = 起点",
        ]

    def draw_tracking(self):
        """2 the A* path at each replanning instant + tracked trajectory."""
        self.fig.clear()
        data = self.data
        steps = data["replan_steps"]
        count = max(1, len(steps))
        index = self._sync_slider(count)
        index = min(index, len(steps) - 1) if len(steps) else 0
        world = self.world
        truth = data[f"truth_B_{self.show_scene}_{self.show_init}"]
        ax_map, ax_cost = self.fig.subplots(1, 2, width_ratios=[1.45, 1.0])
        _add_true_map(ax_map, data[f"true_map_{self.show_scene}"], world)
        if len(steps):
            step = int(steps[index])
            path = data["replan_paths"][index]
            path = path[~np.isnan(path[:, 0])]
            ax_map.plot(
                path[:, 0],
                path[:, 1],
                ":",
                color="#d97706",
                linewidth=2.0,
                label="A* 规划路径（该次重规划）",
                zorder=4,
            )
            ax_map.plot(
                truth[: step + 1, 0],
                truth[: step + 1, 1],
                color="#0f766e",
                linewidth=1.2,
                label="已走过轨迹",
                zorder=3,
            )
            _draw_car(ax_map, truth[step])
        else:
            step = 0
        _add_goal(ax_map, self.report["protocol"]["goal_xy"], world)
        ax_map.plot(
            self.report["protocol"]["start_xy"][0],
            self.report["protocol"]["start_xy"][1],
            "s",
            color="#111827",
            markersize=5,
        )
        ax_map.set(
            xlim=(0, world),
            ylim=(0, world),
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"②重规划 {index + 1}/{max(1, len(steps))}（第 {step} 步）：橙点线 = A* 已知栅格路径",
        )
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(fontsize=7, loc="upper left")
        if len(steps):
            costs = data["replan_costs"]
            ax_cost.plot(steps, costs, "o-", color="#0f766e", markersize=3.5)
            ax_cost.plot(step, costs[index], "o", color="#b45309", markersize=8)
        ax_cost.set(
            xlabel="步（0.04 s）",
            ylabel="剩余 A* 路径代价（m）",
            title="剩余路径代价随重规划递减（行进 + 绕行更新）",
        )
        ax_cost.grid(alpha=0.2)
        cost_text = "—" if not len(steps) else f"{float(data['replan_costs'][index]):.2f} m"
        cells_text = "—" if not len(steps) else str(int(data["replan_cells"][index]))
        self._mapping_lines = [
            "② 这一幕在跟：已知栅格上的 A* 重规划与纯追踪",
            f"重规划 {index + 1}/{max(1, len(steps))}：第 {step} 步，路径代价 {cost_text}，{cells_text} 格",
            f"本回合共重规划 {self._showcase_row()['replans']} 次",
            f"结局 {self._showcase_row()['outcome']}，实际路径 {self._showcase_row()['path_length_m']:.2f} m",
            "灰底 = 真值障碍（对照；规划器只看已建出的已知格）",
        ]

    def draw_compare(self):
        """3 paired trajectories + arrival/collision/ratio bars from the summary."""
        self.fig.clear()
        data = self.data
        report = self.report
        world = self.world
        ax_map, ax_arr, ax_col, ax_ratio = self.fig.subplots(2, 2).reshape(-1)
        scene = self.show_scene
        _add_true_map(ax_map, data[f"true_map_{scene}"], world, alpha=0.35)
        truth_a = data[f"truth_A_{scene}_{self.show_init}"]
        truth_b = data[f"truth_B_{scene}_{self.show_init}"]
        ax_map.plot(truth_a[:, 0], truth_a[:, 1], color="#b91c1c", linewidth=1.2, label="A 盲飞")
        ax_map.plot(
            truth_b[:, 0], truth_b[:, 1], color="#0f766e", linewidth=1.2, label="B 建图+规划"
        )
        _add_goal(ax_map, report["protocol"]["goal_xy"], world)
        ax_map.set(
            xlim=(0, world),
            ylim=(0, world),
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"③场景 {scene} 配对轨迹（初始化 {self.show_init}）：A 红 / B 青",
        )
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.legend(fontsize=7, loc="upper left")
        conditions = [("obstacle", "有障碍"), ("empty", "无障碍")]
        width = 0.36
        colors = {"A": "#b91c1c", "B": "#0f766e"}
        for ax, key, name in (
            (ax_arr, "arrival_rate", "到达率（< 5 cm 且 < 0.1 m/s）"),
            (ax_col, "collision_events_mean", "平均碰撞事件/回合"),
            (ax_ratio, "median_path_ratio", "路径比中位（实际/真值最短）"),
        ):
            for offset, group in enumerate(GROUPS):
                values, texts = [], []
                for condition, label in conditions:
                    aggregate = report["aggregates"].get(group, {}).get(condition)
                    value = None if aggregate is None else aggregate.get(key)
                    values.append(0.0 if value is None else float(value))
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
                    color=colors[group],
                    label=GROUP_LABELS[group],
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
        ax_arr.set_ylim(0, 1.15)
        ax_arr.legend(fontsize=7)
        hypothesis = report["hypothesis"]
        aggregates = report["aggregates"]
        self._mapping_lines = [
            "③ 这一幕在裁决：建图 + 规划是否优于盲飞",
            (
                f"有障碍：A 到达 {aggregates['A']['obstacle']['arrivals']}/"
                f"{aggregates['A']['obstacle']['episodes']}，"
                f"B 到达 {aggregates['B']['obstacle']['arrivals']}/"
                f"{aggregates['B']['obstacle']['episodes']}"
            ),
            (
                f"碰撞事件：A 合计 {aggregates['A']['obstacle']['collision_events_total']}，"
                f"B 合计 {aggregates['B']['obstacle']['collision_events_total']}"
            ),
            (
                f"无障碍对照：A {aggregates['A']['empty']['arrivals']}/"
                f"{aggregates['A']['empty']['episodes']}，"
                f"B {aggregates['B']['empty']['arrivals']}/"
                f"{aggregates['B']['empty']['episodes']}"
            ),
            f"预注册判据：{'满足' if hypothesis['met'] else '不满足'}（{hypothesis['criterion']}）",
        ]

    # ------------------------------------------------------------------ panel
    def _showcase_row(self):
        return next(
            row
            for row in self.report["episodes"]
            if row["group"] == "B"
            and row["scene"] == self.show_scene
            and row["init"] == self.show_init
        )

    def _showcase_coverage(self):
        return float(self._showcase_row()["coverage_final"])

    def fill_stats(self, _mode):
        lines = getattr(self, "_mapping_lines", [])
        if lines:
            self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "mapping":
            self.draw_mapping()
        elif mode == "tracking":
            self.draw_tracking()
        else:
            self.draw_compare()
        self.fill_stats(mode)
        status = {
            "mapping": "① 建图过程：Bresenham 射线投射逐步把未知格标成空闲/占据（拖动滑条）",
            "tracking": "② A* 重规划 + 纯追踪：每次重规划在当前已知栅格上搜索最短无碰撞路径（拖动滑条）",
            "compare": "③ 盲飞 vs 规划：配对轨迹、到达率、碰撞事件与路径长度比（数字与 summary 逐格一致）",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第四十三课 · 占据栅格建图 + A* 路径规划 + 纯追踪")
    root.geometry("1600x860")
    root.minsize(1380, 720)
    GridNavDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
