"""Lesson 18 read-only replay: cumulative odometry versus absolute landmark samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import compose, to_parent
from embodied_learning.experiments.landmark_observations import (
    EXPERIMENT,
    LANDMARKS,
    OBS_BEARING_STD_RAD,
    OBS_PERIOD_STEPS,
    OBS_RANGE_STD_M,
    SCENARIOS,
)
from embodied_learning.experiments.mobile_frames import DT, GEOMETRY, SENSOR_IN_BODY
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.odometry import heading_error
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第十八课 · 控制点观测与里程计：谁在累积？"
METRICS = ("位置误差 / cm", "朝向误差 / °")
DEFAULT_RESULTS = "results/mobile_landmarks_2026-09-03"
SCENARIO_KEYS = tuple(key for key, _, _ in SCENARIOS)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-18 recording; returns (routes, ensembles)."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("model") != "ideal_no_slip_velocity_kinematics"
        or report.get("dt_s") != DT
        or report.get("wheel_radius_m") != GEOMETRY.radius_m
        or report.get("track_width_m") != GEOMETRY.track_m
        or not np.array_equal(report.get("sensor_in_body"), SENSOR_IN_BODY)
        or not np.array_equal(report.get("landmarks_world_m"), LANDMARKS)
        or report.get("observation_period_s") != OBS_PERIOD_STEPS * DT
        or report.get("range_noise_std_m") != OBS_RANGE_STD_M
        or report.get("bearing_noise_std_rad") != OBS_BEARING_STD_RAD
    ):
        raise ValueError("Incompatible lesson-18 recording")
    runs = report.get("runs_per_group")
    base = report.get("base_seed")
    if type(runs) is not int or runs < 2 or type(base) is not int:
        raise ValueError("Invalid repetition settings")
    if report.get("seeds") != list(range(base, base + runs)):
        raise ValueError("Invalid seeds")
    if [c["key"] for c in report.get("cases", [])] != list(SCENARIO_KEYS):
        raise ValueError("Missing, duplicated or unexpected scenarios")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    routes, ensembles = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        for scenario_key in SCENARIO_KEYS:
            case = next(c for c in report["cases"] if c["key"] == scenario_key)
            steps = case["steps"]
            if type(steps) is not int or steps != case["observation_frames"][-1]:
                raise ValueError("Invalid timing")
            route = {}
            for name, width in (
                ("true_poses", 3),
                ("wheels", 2),
                ("wheel_angles", 2),
            ):
                value = archive[f"{scenario_key}_{name}"].copy()
                rows = steps if name == "wheels" else steps + 1
                if value.shape != (rows, width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {name}")
                value.flags.writeable = False
                route[name] = value
            for name, width in (("odom_poses", 3), ("landmark_poses", 3)):
                value = archive[f"{scenario_key}_{name}"].copy()
                if value.shape != (runs, steps + 1, width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {name}")
                value.flags.writeable = False
                route[name] = value
            for err_name in ("odom_position_error", "landmark_position_error"):
                value = archive[f"{scenario_key}_{err_name}"].copy()
                if value.shape != (runs, steps + 1) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {err_name}")
                value.flags.writeable = False
                route[err_name] = value
            for run in range(runs):
                expected_pos = np.linalg.norm(
                    route["odom_poses"][run, :, :2] - route["true_poses"][:, :2], axis=1
                )
                if not np.allclose(
                    route["odom_position_error"][run], expected_pos, atol=1e-9, rtol=0
                ):
                    raise ValueError("Inconsistent odometry error")
                expected_lm = np.linalg.norm(
                    route["landmark_poses"][run, :, :2] - route["true_poses"][:, :2], axis=1
                )
                if not np.allclose(
                    route["landmark_position_error"][run], expected_lm, atol=1e-9, rtol=0
                ):
                    raise ValueError("Inconsistent landmark error")
            mean_odom = route["odom_position_error"].mean(axis=0) * 100
            std_odom = route["odom_position_error"].std(axis=0, ddof=1) * 100
            assert len(mean_odom) == steps + 1
            ensembles[scenario_key] = {
                "mean_odom_cm": mean_odom,
                "std_odom_cm": std_odom,
                "observation_frames": case["observation_frames"],
                "observed_table": case["observed_table"],
                "stats": case["stats"],
            }
            routes[scenario_key] = {**route, "steps": steps}
    return routes, ensembles


class LandmarkDemo(MobileDemo):
    """One seeded sample against the statistics; observation dots on the map."""

    def __init__(self, root, routes, ensembles, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.routes, self.ensembles = routes, ensembles
        self.route_key = SCENARIO_KEYS[0]
        self.run = 0
        self.clock = PlaybackClock(self.current()["steps"], DT, speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1120x730")
        root.minsize(980, 660)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="里程计自己累积误差；已知地标（控制点）每次读数都与时间无关",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "紫：编码器里程计（累积）｜绿点：地标观测时刻（独立）｜灰虚线：两次观测之间估计保持旧值\n"
                "观测每 2 s 一次，测距噪声 ±1 cm、测角 ±0.57°；本课不做融合"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.choice = tk.StringVar(value=SCENARIOS[0][1])
        self.case_box = ttk.Combobox(
            controls,
            textvariable=self.choice,
            values=[label for _, label, _ in SCENARIOS],
            state="readonly",
            width=20,
        )
        self.case_box.pack(side="left")
        self.case_box.bind("<<ComboboxSelected>>", self.select_case)
        self.sample = tk.IntVar(value=0)
        sample_box = ttk.Spinbox(
            controls,
            from_=0,
            to=total_runs(routes) - 1,
            textvariable=self.sample,
            width=5,
            command=self.select_sample,
        )
        sample_box.pack(side="left", padx=5)
        sample_box.bind("<Return>", lambda _: self.select_sample())
        ttk.Label(controls, text="样本 #").pack(side="left")
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=7)
        self.play_button.pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="单步", command=self.step, width=6).pack(side="left", padx=3)
        ttk.Button(controls, text="起点", command=lambda: self.seek(0), width=6).pack(side="left")
        ttk.Button(controls, text="下一观测", command=self.next_observation, width=9).pack(
            side="left", padx=3
        )
        self.speed = tk.StringVar(value=f"{speed:g}")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        speed_box.pack(side="left", padx=5)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.clock.steps,
            orient="horizontal",
            showvalue=False,
            command=self.slider_changed,
        )
        self.timeline.pack(fill="x")
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            middle,
            background="#f8fafc",
            highlightthickness=0,
            width=560,
            height=330,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=46, anchor="nw", justify="left")
        self.stats.pack(side="right", fill="y", padx=(10, 0))
        chart_controls = ttk.Frame(outer)
        chart_controls.pack(fill="x", pady=(5, 0))
        self.metric = tk.StringVar(value=METRICS[0])
        metric_box = ttk.Combobox(
            chart_controls,
            textvariable=self.metric,
            values=list(METRICS),
            state="readonly",
            width=16,
        )
        metric_box.pack(side="left")
        metric_box.bind("<<ComboboxSelected>>", lambda _: self.redraw())
        ttk.Label(
            chart_controls,
            text="紫 里程计 ｜ 灰虚线 观测后保持 ｜ 绿点 观测时刻（独立于时间）",
        ).pack(side="left", padx=10)
        self.chart = tk.Canvas(outer, background="white", highlightthickness=0, height=130)
        self.chart.pack(fill="x")
        self.status = ttk.Label(outer)
        self.status.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            outer,
            text=(
                "已知：地标世界坐标、地标识别、传感器安装位姿；观测周期与仿真时钟同步｜无遮挡、无探测失败\n"
                "观测误差与距离有关；两次观测之间没有新信息，只能保持旧值——这正是融合的动机"
            ),
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.chart.bind("<Configure>", lambda _: self.redraw())
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def current(self):
        return self.routes[self.route_key]

    def runs_max(self):
        return len(self.current()["odom_poses"]) - 1

    def select_case(self, _=None):
        index = int(self.case_box.current())
        if index < 0:
            return
        self.route_key = SCENARIO_KEYS[index]
        self.run = min(self.run, self.runs_max())
        self.clock = PlaybackClock(self.current()["steps"], DT, self.clock.speed)
        self.timeline.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.redraw()

    def select_sample(self, _=None):
        self.run = max(0, min(self.sample.get(), self.runs_max()))
        self.sample.set(self.run)
        self.last_tick = time.perf_counter()
        self.redraw()

    def next_observation(self):
        frames = self.ensembles[self.route_key]["observation_frames"]
        target = next((f for f in frames if f > self.clock.index), frames[-1])
        self.seek(target)

    def redraw(self):
        if not hasattr(self, "status"):
            return
        i = self.clock.index
        route, ens = self.current(), self.ensembles[self.route_key]
        truth, odom, lm = (
            route["true_poses"],
            route["odom_poses"][self.run],
            route["landmark_poses"][self.run],
        )
        true_pose, odom_pose, lm_pose = truth[i], odom[i], lm[i]
        odom_err = route["odom_position_error"][self.run][i] * 100
        lm_err = route["landmark_position_error"][self.run][i] * 100
        frames = ens["observation_frames"]
        last_sample = next((f for f in reversed(frames) if f <= i), None)
        if i < self.clock.steps:
            left, right = route["wheels"][i]
            command = f"下一步指令：左 {left:+.3f} / 右 {right:+.3f} rad/s"
        else:
            command = "回放结束：没有下一步轮速"
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        obs_text = "尚无观测（前 2 s 只知初始位姿）"
        if last_sample is not None:
            obs_text = (
                f"上次观测：{last_sample * DT:.1f} s（误差真值 {route['landmark_position_error'][self.run][last_sample] * 100:.2f} cm）\n"
                f"距上次观测已过 {(i - last_sample) * DT:.1f} s，估计保持旧值"
            )
        self.stats.configure(
            text=(
                f"时间 {i * DT:.2f} / {self.clock.steps * DT:.2f} s｜样本 #{self.run}\n\n"
                f"真实：x={true_pose[0]:+.3f}  y={true_pose[1]:+.3f}  θ={np.rad2deg(true_pose[2]):+.1f}°\n"
                f"里程计：x={odom_pose[0]:+.3f}  y={odom_pose[1]:+.3f}  θ={np.rad2deg(odom_pose[2]):+.1f}°\n"
                f"　　误差 {odom_err:.2f} cm（累积，随路线增大）\n"
                f"地标观测：x={lm_pose[0]:+.3f}  y={lm_pose[1]:+.3f}  θ={np.rad2deg(lm_pose[2]):+.1f}°\n"
                f"　　误差 {lm_err:.2f} cm\n\n"
                f"{obs_text}\n\n{command}"
            )
        )
        self.draw_map(i, route, truth, odom, lm)
        self.draw_chart(i, route)
        state = "结束" if i == self.clock.steps else ("暂停" if self.clock.paused else "播放中")
        self.status.configure(
            text=(
                f"{state} · {self.clock.speed:g}× · 单步 {DT:g} s · "
                "下一观测跳到下一个 2 s 采样点；切样本保留当前时刻"
            )
        )

    def draw_map(self, i, route, truth, odom, lm):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        bounds = {
            "straight": (-0.6, 3.2, -1.0, 2.4),
            "square": (-0.6, 3.2, -1.0, 2.4),
            "long": (-0.6, 7.2, -2.6, 2.6),
        }[self.route_key]
        xmin, xmax, ymin, ymax = bounds
        scale = min((w - 70) / (xmax - xmin), (h - 70) / (ymax - ymin))
        center = np.array([w / 2, h / 2 + 6])

        def xy(point):
            return center + (np.asarray(point) - [(xmin + xmax) / 2, (ymin + ymax) / 2]) * [
                scale,
                -scale,
            ]

        font = ("Microsoft YaHei", 9)
        for x in np.arange(np.floor(xmin), xmax + 0.01, 1.0):
            c.create_line(*xy([x, ymin]), *xy([x, ymax]), fill="#e2e8f0")
            c.create_text(*xy([x, ymin]), text=f"{x:g}", font=font, anchor="n", fill="#64748b")
        for y in np.arange(np.floor(ymin), ymax + 0.01, 1.0):
            c.create_line(*xy([xmin, y]), *xy([xmax, y]), fill="#e2e8f0")
            c.create_text(*xy([xmin, y]), text=f"{y:g}", font=font, anchor="e", fill="#64748b")
        c.create_text(
            8,
            8,
            text="世界 XY / m｜蓝=真值 紫=里程计 绿=地标观测 黑三角=已知地标",
            anchor="nw",
            font=font,
        )
        # Sensor rays to landmarks from the current true pose (thin, light).
        sensor = compose(truth[i], SENSOR_IN_BODY)
        for landmark in LANDMARKS:
            c.create_line(*xy(sensor[:2]), *xy(landmark), fill="#cbd5e1", dash=(2, 4))
        # Observation-time dots on the landmark estimate path.
        frames = self.ensembles[self.route_key]["observation_frames"]
        for frame in frames:
            x, y = xy(lm[frame, :2])
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#0f766e", outline="white", width=1)
        if i:
            c.create_line(
                *[v for p in truth[: i + 1, :2] for v in xy(p)], fill="#2563eb", width=2.4
            )
            c.create_line(
                *[v for p in odom[: i + 1, :2] for v in xy(p)],
                fill="#9333ea",
                width=1.6,
                dash=(6, 3),
            )
            c.create_line(
                *[v for p in lm[: i + 1, :2] for v in xy(p)],
                fill="#0f766e",
                width=1.4,
                dash=(2, 3),
            )
        for pose, paint, fill in (
            (truth[i], "#2563eb", "#dbeafe"),
            (odom[i], "#9333ea", ""),
            (lm[i], "#0f766e", ""),
        ):
            corners = [
                to_parent(pose, p)
                for p in ([0.18, 0.12], [0.18, -0.12], [-0.18, -0.12], [-0.18, 0.12])
            ]
            c.create_polygon(
                *[v for p in corners for v in xy(p)],
                fill=fill,
                outline=paint,
                dash=None,
                width=2,
            )
        for landmark in LANDMARKS:
            x, y = xy(landmark)
            c.create_polygon(x, y - 7, x - 6, y + 5, x + 6, y + 5, fill="black", outline="")

    def draw_chart(self, i, route):
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        position_mode = self.metric.get() == METRICS[0]
        if position_mode:
            odom_curve = route["odom_position_error"][self.run] * 100
            lm_curve = route["landmark_position_error"][self.run] * 100
        else:
            odom_curve = np.rad2deg(
                heading_error(route["odom_poses"][self.run, :, 2], route["true_poses"][:, 2])
            )
            lm_curve = np.rad2deg(
                heading_error(route["landmark_poses"][self.run, :, 2], route["true_poses"][:, 2])
            )
        low, high = 0.0, max(1.0, float(max(odom_curve.max(), lm_curve.max()))) * 1.1
        left, right, top, bottom = 60, w - 20, 12, h - 25

        def xy(index, value):
            return left + index / self.clock.steps * (right - left), bottom - (value - low) / (
                high - low
            ) * (bottom - top)

        font = ("Microsoft YaHei", 9)
        for value in np.linspace(low, high, 3):
            _, y = xy(0, value)
            c.create_line(left, y, right, y, fill="#e2e8f0")
            c.create_text(left - 5, y, text=f"{value:.1f}", anchor="e", font=font)
        c.create_line(
            *[v for n, value in enumerate(odom_curve) for v in xy(n, value)],
            fill="#9333ea",
            width=2,
        )
        c.create_line(
            *[v for n, value in enumerate(lm_curve) for v in xy(n, value)],
            fill="#94a3b8",
            width=1.3,
            dash=(5, 3),
        )
        for frame in self.ensembles[self.route_key]["observation_frames"]:
            if self.metric.get() == METRICS[0]:
                value = route["landmark_position_error"][self.run][frame] * 100
            else:
                value = np.rad2deg(
                    heading_error(
                        route["landmark_poses"][self.run, frame, 2], route["true_poses"][frame, 2]
                    )
                )
            x, y = xy(frame, value)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#0f766e", outline="white")
        x, _ = xy(i, low)
        c.create_line(x, top, x, bottom, fill="#0f172a", dash=(4, 3))
        for step in np.linspace(0, self.clock.steps, 5):
            x, _ = xy(step, low)
            c.create_text(x, bottom + 13, text=f"{step * DT:g} s", font=font)


def total_runs(routes):
    key = next(iter(routes))
    return len(routes[key]["odom_poses"])


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    routes, ensembles = load_replays(args.results)
    root = tk.Tk()
    LandmarkDemo(root, routes, ensembles, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
