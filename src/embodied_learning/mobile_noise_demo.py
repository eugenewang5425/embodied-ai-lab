"""Lesson 17 read-only replay: one seeded noise sample against 20-run statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import to_parent
from embodied_learning.experiments.mobile_frames import DT, GEOMETRY
from embodied_learning.experiments.mobile_noise import (
    EXPERIMENT,
    GROUPS,
    INTERVAL_NOISE_STD_RAD,
    SCENARIOS,
    calibrated_factor,
)
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.odometry import estimate_poses, heading_error
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第十七课 · 标定之后：每次为什么仍不同？"
METRICS = ("位置误差 / cm", "朝向误差 / °")
DEFAULT_RESULTS = "results/mobile_noise_2026-09-03"
GROUP_KEYS = tuple(key for key, _, _ in GROUPS)
GROUP_COLORS = {key: color for key, _, color in GROUPS}
GROUP_LABELS = {key: label for key, label, _ in GROUPS}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate and load a lesson-17 recording; returns (routes, ensembles)."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("model") != "ideal_no_slip_velocity_kinematics"
        or report.get("dt_s") != DT
        or report.get("wheel_radius_m") != GEOMETRY.radius_m
        or report.get("track_width_m") != GEOMETRY.track_m
        or report.get("interval_noise_std_rad") != INTERVAL_NOISE_STD_RAD
        or not np.isclose(report.get("correction_factor"), calibrated_factor()[0], atol=1e-12)
    ):
        raise ValueError("Incompatible lesson-17 recording")
    runs = report.get("runs_per_group")
    base = report.get("base_seed")
    if type(runs) is not int or runs < 2 or type(base) is not int:
        raise ValueError("Invalid repetition settings")
    if report.get("seeds") != list(range(base, base + runs)):
        raise ValueError("Invalid seeds")
    if [g.get("key") for g in report.get("groups", [])] != list(GROUP_KEYS):
        raise ValueError("Invalid groups")
    if [c["key"] for c in report.get("cases", [])] != [key for key, _, _ in SCENARIOS]:
        raise ValueError("Missing, duplicated or unexpected scenarios")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    routes, ensembles = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        for scenario_key, _, _ in SCENARIOS:
            steps = next(c["steps"] for c in report["cases"] if c["key"] == scenario_key)
            if type(steps) is not int or steps != SCENARIOS_STEPS[scenario_key]:
                raise ValueError("Invalid step count")
            common = {
                name: archive[f"{scenario_key}_{name}"].copy()
                for name in ("true_poses", "wheels", "wheel_angles")
            }
            for name, value in common.items():
                rows = steps if name == "wheels" else steps + 1
                width = 3 if name == "true_poses" else 2
                if value.shape != (rows, width) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {name}")
                value.flags.writeable = False
            groups = {}
            for group in GROUP_KEYS:
                count = runs if group != "noiseless" else 1
                arrays, seeds = (
                    {},
                    [base] if group == "noiseless" else list(range(base, base + runs)),
                )
                for name, width in (
                    ("poses", 3),
                    ("readings", 2),
                    ("epsilon", None),
                    ("position_error", None),
                    ("heading_error", None),
                ):
                    value = archive[f"{scenario_key}_{group}_{name}"].copy()
                    frames = steps if name == "epsilon" else steps + 1
                    expected = (count, frames) if width is None else (count, frames, width)
                    if value.shape != expected or not np.isfinite(value).all():
                        raise ValueError(f"Invalid array: {group}/{name}")
                    value.flags.writeable = False
                    arrays[name] = value
                truth = common["true_poses"]
                for run in range(count):
                    expected_pos = np.linalg.norm(
                        arrays["poses"][run, :, :2] - truth[:, :2], axis=1
                    )
                    expected_yaw = heading_error(arrays["poses"][run, :, 2], truth[:, 2])
                    if not np.allclose(
                        arrays["position_error"][run], expected_pos, atol=1e-12, rtol=0
                    ):
                        raise ValueError(f"Inconsistent position error: {scenario_key}/{group}")
                    if not np.allclose(
                        arrays["heading_error"][run], expected_yaw, atol=1e-12, rtol=0
                    ):
                        raise ValueError(f"Inconsistent heading error: {scenario_key}/{group}")
                    if not np.array_equal(
                        estimate_poses(arrays["readings"][run]), arrays["poses"][run]
                    ):
                        raise ValueError(
                            f"Estimate does not match readings: {scenario_key}/{group}"
                        )
                groups[group] = {"arrays": arrays, "seeds": seeds, "runs": count}
            routes[scenario_key] = {"steps": steps, "truth": common, "groups": groups}
            ensembles[scenario_key] = {}
            for group in GROUP_KEYS:
                arrays = groups[group]["arrays"]
                pos = arrays["position_error"] * 100
                yaw = arrays["heading_error"] * (180 / np.pi)
                ensembles[scenario_key][group] = {
                    "position_error": (
                        pos.mean(axis=0),
                        pos.std(axis=0, ddof=1) if len(pos) > 1 else np.zeros(steps + 1),
                    ),
                    "heading_error": (
                        yaw.mean(axis=0),
                        yaw.std(axis=0, ddof=1) if len(yaw) > 1 else np.zeros(steps + 1),
                    ),
                }
    return routes, ensembles


SCENARIOS_STEPS = {key: steps for key, _, steps in SCENARIOS}


class NoiseDemo(MobileDemo):
    """Replay one seeded sample; chart shows the 20-run mean ± 1 sigma band."""

    def __init__(self, root, routes, ensembles, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.routes, self.ensembles = routes, ensembles
        self.route_key = SCENARIOS[0][0]
        self.run = 0
        self.group = "fixed"
        self.clock = PlaybackClock(self.current()["steps"], DT, speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1080x710")
        root.minsize(950, 650)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="标定系数已经对了，为什么每次还是不一样？",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "绿：无噪声理想基准 ｜ 蓝：准确标定 + 逐区间噪声 ｜ 紫：未标定 + 噪声\n"
                "同一路线重复 20 次；切换方法不换噪声，只换系数 c"
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
            to=len(routes[SCENARIOS[0][0]]["groups"]["fixed"]["seeds"]) - 1,
            textvariable=self.sample,
            width=5,
            command=self.select_sample,
        )
        sample_box.pack(side="left", padx=5)
        sample_box.bind("<Return>", lambda _: self.select_sample())
        self.group_label = tk.StringVar(value=GROUP_LABELS["fixed"])
        self.group_box = ttk.Combobox(
            controls,
            textvariable=self.group_label,
            values=[GROUP_LABELS[key] for key in GROUP_KEYS],
            state="readonly",
            width=26,
        )
        self.group_box.pack(side="left")
        self.group_box.bind("<<ComboboxSelected>>", self.select_group)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=7)
        self.play_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="单步", command=self.step, width=6).pack(side="left", padx=3)
        ttk.Button(controls, text="起点", command=lambda: self.seek(0), width=6).pack(side="left")
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
            width=550,
            height=320,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=44, anchor="nw", justify="left")
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
            text="阴影 = 20 次均值 ± 1σ ｜ 虚线 = 当前样本 ｜ 绿 无噪声 · 蓝 标定+噪声 · 紫 未标定+噪声",
        ).pack(side="left", padx=10)
        self.chart = tk.Canvas(outer, background="white", highlightthickness=0, height=125)
        self.chart.pack(fill="x")
        self.status = ttk.Label(outer)
        self.status.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            outer,
            text=(
                "真实运动与轮速指令与噪声无关；已知初始位姿、无打滑｜固定 c 消除固定比例偏差，"
                "平均误差回到近零，但每次样本的分散仍然存在｜本课没有滤波器、外部定位或 SLAM"
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

    def select_case(self, _=None):
        index = int(self.case_box.current())
        if index < 0:
            return
        self.route_key = SCENARIOS[index][0]
        self.run = min(self.run, self.runs() - 1)
        self.clock = PlaybackClock(self.current()["steps"], DT, self.clock.speed)
        self.timeline.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.redraw()

    def select_sample(self, _=None):
        value = max(0, min(self.sample.get(), self.runs() - 1))
        self.sample.set(value)
        self.run = value
        self.last_tick = time.perf_counter()
        self.redraw()

    def select_group(self, _=None):
        index = int(self.group_box.current())
        if index < 0:
            return
        self.group = GROUP_KEYS[index]
        self.clock.seek(self.clock.index)  # Same time, same noise; only c changes.
        self.redraw()

    def runs(self):
        return self.current()["groups"]["fixed"]["runs"]

    def _arrays(self):
        route = self.current()
        group = route["groups"][self.group]
        run = self.run if self.group != "noiseless" else 0
        return group["arrays"], group["seeds"][run]

    def redraw(self):
        if not hasattr(self, "status"):
            return
        i = self.clock.index
        data, seed = self._arrays()
        truth = self.current()["truth"]
        pose = data["poses"][self.run if self.group != "noiseless" else 0][i]
        true_pose = truth["true_poses"][i]
        pos_err = data["position_error"][self.run if self.group != "noiseless" else 0][i] * 100
        yaw_err = np.rad2deg(data["heading_error"][self.run if self.group != "noiseless" else 0][i])
        if i:
            real = np.diff(truth["wheel_angles"][i - 1 : i + 1], axis=0)[0, 1]
            raw = np.diff(
                data["readings"][self.run if self.group != "noiseless" else 0][i - 1 : i + 1],
                axis=0,
            )[0, 1]
            eps = data["epsilon"][self.run if self.group != "noiseless" else 0][i - 1]
            factor = calibrated_factor()[0] if self.group != "uncorrected" else 1.0
            reading = (
                f"刚结束区间（右轮）：\n真实 {real:+.4f} → 原始 {raw / factor:+.4f} rad\n"
                f"噪声 ε {eps:+.4f} rad\n送入估计器 {raw:+.4f} rad（×{factor:.6f}）"
            )
        else:
            reading = "尚无已完成区间\n初始位姿已知，真值与估计重合"
        if i < self.clock.steps:
            left, right = truth["wheels"][i]
            command = f"下一步指令：左 {left:+.3f} / 右 {right:+.3f} rad/s"
        else:
            command = "回放结束：没有下一步轮速"
        group = self.ensembles[self.route_key][self.group]
        selected = "位置误差 / cm" if self.metric.get() == METRICS[0] else "朝向误差 / °"
        mkey = "position_error" if self.metric.get() == METRICS[0] else "heading_error"
        mean_val, std_val = group[mkey]
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        unit = "cm" if self.metric.get() == METRICS[0] else "°"
        self.stats.configure(
            text=(
                f"时间 {i * DT:.2f} / {self.clock.steps * DT:.2f} s\n"
                f"样本 #{self.run}（种子 {seed}）｜{GROUP_LABELS[self.group]}\n\n"
                f"真实：x={true_pose[0]:+.3f}  y={true_pose[1]:+.3f} m\n"
                f"θ={np.rad2deg(true_pose[2]):+.2f}°\n"
                f"估计：x={pose[0]:+.3f}  y={pose[1]:+.3f} m\n"
                f"θ̂={np.rad2deg(pose[2]):+.2f}°\n\n"
                f"本次位置误差 {pos_err:.2f} cm\n本次朝向误差 {yaw_err:+.2f}°\n\n"
                f"20 次同刻 {selected}：\n均值 {mean_val[i]:.2f} {unit} ± {std_val[i]:.2f} {unit}\n\n"
                f"{reading}\n{command}"
            )
        )
        self.draw_map(i, data)
        self.draw_chart(i, data)
        state = "结束" if i == self.clock.steps else ("暂停" if self.clock.paused else "播放中")
        self.status.configure(
            text=(
                f"{state} · {self.clock.speed:g}× · 单步 {DT:g} s · "
                "切路线回到起点；切方法/样本保留当前时刻｜无噪声时样本选择无效"
            )
        )

    def draw_map(self, i, data):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        bounds = (-0.4, 2.8, -0.6, 1.1) if self.route_key == "straight" else (-0.5, 1.5, -0.4, 1.4)
        xmin, xmax, ymin, ymax = bounds
        scale = min((w - 60) / (xmax - xmin), (h - 60) / (ymax - ymin))
        center = np.array([w / 2, h / 2])

        def xy(point):
            return center + (np.asarray(point) - [(xmin + xmax) / 2, (ymin + ymax) / 2]) * [
                scale,
                -scale,
            ]

        font = ("Microsoft YaHei", 9)
        for x in np.arange(0, xmax + 0.01, 0.5):
            c.create_line(*xy([x, ymin]), *xy([x, ymax]), fill="#e2e8f0")
            c.create_text(*xy([x, ymin]), text=f"{x:g}", font=font, anchor="n", fill="#64748b")
        for y in np.arange(0, ymax + 0.01, 0.5):
            c.create_line(*xy([xmin, y]), *xy([xmax, y]), fill="#e2e8f0")
            c.create_text(*xy([xmin, y]), text=f"{y:g}", font=font, anchor="e", fill="#64748b")
        c.create_text(
            8,
            8,
            text="世界 XY / m｜蓝色实车 + 彩色估计轮廓（不是第二台车）",
            anchor="nw",
            font=font,
        )
        run = self.run if self.group != "noiseless" else 0
        truths = self.current()["truth"]["true_poses"]
        estimate = data["poses"][run]
        color = GROUP_COLORS[self.group]
        if i:
            c.create_line(
                *[v for p in truths[: i + 1, :2] for v in xy(p)],
                fill="#2563eb",
                width=2,
            )
            c.create_line(
                *[v for p in estimate[: i + 1, :2] for v in xy(p)],
                fill=color,
                width=2,
                dash=(5, 3),
            )
        for pose, paint, dash in (
            (truths[i], "#2563eb", None),
            (estimate[i], color, (5, 3)),
        ):
            corners = [
                to_parent(pose, p)
                for p in ([0.18, 0.12], [0.18, -0.12], [-0.18, -0.12], [-0.18, 0.12])
            ]
            c.create_polygon(
                *[v for p in corners for v in xy(p)],
                fill="#dbeafe" if dash is None else "",
                outline=paint,
                dash=dash,
                width=2,
            )
            c.create_line(
                *xy(pose[:2]),
                *xy(to_parent(pose, [0.28, 0])),
                fill=paint,
                arrow="last",
                dash=dash,
                width=2,
            )

    def draw_chart(self, i, data):
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        mkey = "position_error" if self.metric.get() == METRICS[0] else "heading_error"
        curves = []
        for group in GROUP_KEYS:
            mean, std = self.ensembles[self.route_key][group][mkey]
            curves.append((mean, std, GROUP_COLORS[group]))
        low = 0.0
        high = max(1.0, max(float(curve.max()) for curve, _, _ in curves)) * 1.1
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
        for mean, std, color in curves:
            xs = [xy(n, mean[n])[0] for n in range(len(mean))]
            ys_hi = [xy(n, mean[n] + std[n])[1] for n in range(len(mean))]
            ys_lo = [xy(n, max(mean[n] - std[n], 0))[1] for n in range(len(mean))]
            c.create_polygon(
                *[v for pair in zip(xs, ys_hi) for v in pair],
                *[v for pair in zip(reversed(xs), reversed(ys_lo)) for v in pair],
                fill=color,
                stipple="gray25",
                outline="",
            )
            c.create_line(
                *[v for n, value in enumerate(mean) for v in xy(n, value)],
                fill=color,
                width=2,
            )
        run = self.run if self.group != "noiseless" else 0
        sample = data[mkey][run]
        c.create_line(
            *[v for n, value in enumerate(sample) for v in xy(n, value)],
            fill=GROUP_COLORS[self.group],
            width=1.5,
            dash=(6, 3),
        )
        x, _ = xy(i, low)
        c.create_line(x, top, x, bottom, fill="#0f172a", dash=(4, 3))
        for step in np.linspace(0, self.clock.steps, 5):
            x, _ = xy(step, low)
            c.create_text(x, bottom + 13, text=f"{step * DT:g} s", font=font)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(DEFAULT_RESULTS),
        help="lesson-17 recording directory (reproduce first if missing)",
    )
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    routes, ensembles = load_replays(args.results)
    root = tk.Tk()
    NoiseDemo(root, routes, ensembles, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
