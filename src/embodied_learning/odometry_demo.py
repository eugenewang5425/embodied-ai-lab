"""Lesson 15 read-only replay: one real robot and its encoder-based pose estimate."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import to_parent
from embodied_learning.experiments.mobile_frames import DT, GEOMETRY, LANDMARK_WORLD, SENSOR_IN_BODY
from embodied_learning.experiments.mobile_odometry import (
    ESTIMATE_WIDTHS,
    SCENARIOS,
    SHARED_WIDTHS,
    VARIANTS,
    schedule,
)
from embodied_learning.mobile_demo import MobileDemo, MobileReplay
from embodied_learning.odometry import heading_error
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第十五课 · 编码器里程计：真实位置与估计位置"
METRICS = {
    "位置误差 / cm": ("position_error_m", 100),
    "朝向误差 / °": ("heading_error_rad", 180 / np.pi),
    "地标落图误差 / cm": ("landmark_error_m", 100),
}


def load_replays(directory, *, variants=VARIANTS, experiment="differential_drive_odometry"):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != experiment
        or report.get("schema_version") != 1
        or report.get("model") != "ideal_no_slip_velocity_kinematics"
        or report.get("dt_s") != DT
        or report.get("wheel_radius_m") != GEOMETRY.radius_m
        or report.get("track_width_m") != GEOMETRY.track_m
        or not np.array_equal(report.get("sensor_in_body"), SENSOR_IN_BODY)
        or not np.array_equal(report.get("landmark_world_m"), LANDMARK_WORLD)
    ):
        raise ValueError("Incompatible odometry recording")
    if [c["key"] for c in report["cases"]] != [key for key, _, _ in SCENARIOS]:
        raise ValueError("Missing, duplicate or unexpected scenarios")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    widths = dict(SHARED_WIDTHS)
    widths.update(
        {
            f"{variant}_{name}": width
            for variant, _, _, _ in variants
            for name, width in ESTIMATE_WIDTHS.items()
        }
    )
    replays = []
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {f"{key}_{name}" for key, _, _ in SCENARIOS for name in widths}:
            raise ValueError("Unexpected archive arrays")
        for case, (key, _, steps) in zip(report["cases"], SCENARIOS):
            if type(case["steps"]) is not int or case["steps"] != steps or case.get("dt_s") != DT:
                raise ValueError("Invalid timing")
            if case.get("checkpoints") != schedule(key)[1]:
                raise ValueError("Invalid checkpoints")
            if [(v["key"], v["right_scale"]) for v in case["estimates"]] != [
                (k, s) for k, _, s, _ in variants
            ]:
                raise ValueError("Invalid encoder variants")
            arrays = {}
            for name, width in widths.items():
                value = archive[f"{key}_{name}"].copy()
                rows = steps if name == "wheels_rad_s" else steps + 1
                shape = (rows,) if width is None else (rows, width)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {name}")
                value.flags.writeable = False
                arrays[name] = value
            truth = arrays["true_poses"]
            for variant, _, _, _ in variants:
                poses = arrays[f"{variant}_poses"]
                expected_errors = {
                    "position_error_m": np.linalg.norm(poses[:, :2] - truth[:, :2], axis=1),
                    "heading_error_rad": heading_error(poses[:, 2], truth[:, 2]),
                    "landmark_error_m": np.linalg.norm(
                        arrays[f"{variant}_mapped_landmark"] - LANDMARK_WORLD, axis=1
                    ),
                }
                for name, expected in expected_errors.items():
                    if not np.allclose(arrays[f"{variant}_{name}"], expected, atol=1e-12, rtol=0):
                        raise ValueError(f"Inconsistent diagnostic: {variant}/{name}")
            replays.append(MobileReplay(case, arrays, DT))
    return replays


class OdometryDemo(MobileDemo):
    """Reuse lesson-14 playback transport only; geometry and panels are distinct."""

    def __init__(
        self, root, replays, speed=0.25, *, variants=VARIANTS, calibration=False, parent=None
    ):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.replays, self.replay = root, replays, replays[0]
        self.variants, self.calibration = variants, calibration
        self.show_landmark = tk.BooleanVar(master=root, value=not calibration)
        self.clock = PlaybackClock(self.replay.steps, DT, speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        if parent is None:
            root.title("第十六课 · 编码器比例标定与独立验证" if calibration else WINDOW_TITLE)
            root.geometry("1080x710")
            root.minsize(950, 650)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="③ 换路线验证：只改估计，不改真实运动"
            if calibration
            else "小车没拐弯，里程计为什么觉得它在转？",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "与第十五课对应：不修正 = 旧 +2%；准确尺子标定 ≈ 旧 0%；尺子偏大1%标定 ≈ 旧 +1%"
                if calibration
                else "只有一台真实小车｜蓝色实线：真值；彩色虚线：估计｜仅改变右轮读数，不改变电机指令"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.choice = tk.StringVar(value=self.replay.metadata["label"])
        self.case_combo = ttk.Combobox(
            controls,
            textvariable=self.choice,
            values=[r.metadata["label"] for r in replays],
            state="readonly",
            width=21,
        )
        self.case_combo.pack(side="left")
        self.case_combo.bind("<<ComboboxSelected>>", self.select_case)
        self.variant = tk.StringVar(value=variants[1 if calibration else 2][1])
        variant_box = ttk.Combobox(
            controls,
            textvariable=self.variant,
            values=[label for _, label, _, _ in variants],
            state="readonly",
            width=23 if calibration else 16,
        )
        variant_box.pack(side="left", padx=5)
        variant_box.bind("<<ComboboxSelected>>", self.select_variant)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=7)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="单步", command=self.step, width=6).pack(side="left", padx=4)
        ttk.Button(controls, text="起点", command=lambda: self.seek(0), width=6).pack(side="left")
        ttk.Button(controls, text="下一段", command=self.next_checkpoint, width=8).pack(
            side="left", padx=4
        )
        self.speed = tk.StringVar(value=f"{speed:g}")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        speed_box.pack(side="left")
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.replay.steps,
            orient="horizontal",
            showvalue=False,
            command=self.slider_changed,
        )
        self.timeline.pack(fill="x")
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            middle, background="#f8fafc", highlightthickness=0, width=550, height=325
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=43, anchor="nw", justify="left")
        self.stats.pack(side="right", fill="y", padx=(10, 0))
        chart_controls = ttk.Frame(outer)
        chart_controls.pack(fill="x", pady=(5, 0))
        self.metric = tk.StringVar(value="位置误差 / cm")
        self.metric_box = ttk.Combobox(
            chart_controls,
            textvariable=self.metric,
            values=list(METRICS)[:2] if calibration else list(METRICS),
            state="readonly",
            width=20,
        )
        self.metric_box.pack(side="left")
        self.metric_box.bind("<<ComboboxSelected>>", lambda _: self.redraw())
        ttk.Label(
            chart_controls,
            text=(
                "紫 不修正 · 绿 准确尺子 · 橙 尺子偏大1%"
                if calibration
                else "三组曲线共用时间/刻度：绿 0% · 橙 +1% · 紫 +2%"
            ),
        ).pack(side="left", padx=10)
        if calibration:
            ttk.Checkbutton(
                chart_controls,
                text="辅助：把灯杆放到地图",
                variable=self.show_landmark,
                command=self.toggle_landmark,
            ).pack(side="left")
        self.chart = tk.Canvas(outer, background="white", highlightthickness=0, height=125)
        self.chart.pack(fill="x")
        self.status = ttk.Label(outer)
        self.status.pack(anchor="w", pady=(5, 0))
        ttk.Label(
            outer,
            text=(
                "理想无打滑、精确几何；标定距离由仿真生成｜绿色重合不是硬件精度；无在线纠偏或 SLAM"
                if calibration
                else "理想无打滑、已知初始位姿；无噪声/力矩/闭环控制｜地标只用于检查落图，不参与定位修正"
            ),
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.chart.bind("<Configure>", lambda _: self.redraw())
        if parent is None:
            root.bind("<space>", lambda _: self.toggle())
            root.bind("<Escape>", lambda _: self.close())
            root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def select_case(self, _=None):
        self.replay = next(r for r in self.replays if r.metadata["label"] == self.choice.get())
        self.clock = PlaybackClock(self.replay.steps, DT, self.clock.speed)
        self.timeline.configure(to=self.replay.steps)
        self.last_tick = time.perf_counter()
        self.redraw()

    def select_variant(self, _=None):
        self.clock.seek(self.clock.index)  # Compare at the SAME timestamp, paused.
        self.redraw()

    def toggle_landmark(self):
        values = list(METRICS) if self.show_landmark.get() else list(METRICS)[:2]
        self.metric_box.configure(values=values)
        if self.metric.get() not in values:
            self.metric.set(values[0])
        self.redraw()

    def next_checkpoint(self):
        self.seek(
            next(
                (i for i in self.replay.metadata["checkpoints"] if i > self.clock.index),
                self.replay.steps,
            )
        )

    def redraw(self):
        if not hasattr(self, "status"):
            return
        i, arrays = self.clock.index, self.replay.arrays
        key, _, scale, color = next(v for v in self.variants if v[1] == self.variant.get())
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        truth, estimate = arrays["true_poses"][i], arrays[f"{key}_poses"][i]
        pos_err = arrays[f"{key}_position_error_m"][i] * 100
        yaw_err = np.rad2deg(arrays[f"{key}_heading_error_rad"][i])
        mapped = arrays[f"{key}_mapped_landmark"][i]
        if i:
            real_delta = np.diff(arrays["wheel_angles_rad"][i - 1 : i + 1], axis=0)[0, 1]
            read_delta = np.diff(arrays[f"{key}_encoder_angles_rad"][i - 1 : i + 1], axis=0)[0, 1]
            reading = f"刚结束区间右轮真实转角：{real_delta:+.4f} rad\n编码器报告：{read_delta:+.4f} rad（×{scale:g}）"
            if self.calibration:
                raw_delta = np.diff(arrays["raw_encoder_angles_rad"][i - 1 : i + 1], axis=0)[0, 1]
                factor = next(
                    v["correction_factor"]
                    for v in self.replay.metadata["estimates"]
                    if v["key"] == key
                )
                reading = (
                    f"右轮原始 → 送入估计器：\n{raw_delta:+.4f} → {read_delta:+.4f} rad"
                    f"（c={factor:.6f}）"
                )
        else:
            reading = "尚无已完成区间：编码器增量未产生\n初始位姿已知，真值与估计重合"
        if i < self.replay.steps:
            left, right = arrays["wheels_rad_s"][i]
            command = f"下一步指令：左 {left:+.3f} / 右 {right:+.3f} rad/s"
        else:
            command = "回放结束：没有下一步轮速"
        landmark_text = ""
        if self.show_landmark.get():
            landmark_text = (
                f"固定地标世界坐标：(1.100, 0.800) m\n按估计位姿落图：({mapped[0]:+.3f}, {mapped[1]:+.3f})\n"
                f"落图误差 {arrays[f'{key}_landmark_error_m'][i] * 100:.2f} cm\n"
                + (
                    "灯杆没动；标记分开才表示有误。\n仅显示误差，不参与定位或控制。"
                    if self.calibration
                    else "转换公式正确，输入位姿仍可能有偏差"
                )
            )
        self.stats.configure(
            text=(
                f"时间 {i * DT:.2f} / {self.replay.steps * DT:.2f} s\n\n"
                f"真实：x={truth[0]:+.3f}  y={truth[1]:+.3f} m\nθ={np.rad2deg(truth[2]):+.2f}°\n"
                f"估计：x={estimate[0]:+.3f}  y={estimate[1]:+.3f} m\nθ̂={np.rad2deg(estimate[2]):+.2f}°\n\n"
                f"位置误差 {pos_err:.2f} cm\n朝向误差 {yaw_err:+.2f}°\n\n"
                f"{reading}\n{command}\n\n"
                f"{landmark_text}"
            )
        )
        self.draw_map(arrays, i, key, color)
        self.draw_chart(arrays, i)
        state = "结束" if i == self.replay.steps else ("暂停" if self.clock.paused else "播放中")
        self.status.configure(
            text=f"{state} · {self.clock.speed:g}× · 单步 {DT:g} s · 切方法保留当前时刻；切路线回到起点 · 下一段可快速跳到检查点"
        )

    def draw_map(self, arrays, i, key, color):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        # Fixed for all three estimates within a scenario; no drifting auto-zoom.
        bounds = (
            (-0.4, 2.8, -0.6, 1.1)
            if self.replay.metadata["key"] == "straight"
            else (-0.5, 1.5, -0.4, 1.4)
        )
        xmin, xmax, ymin, ymax = bounds
        scale = min((w - 60) / (xmax - xmin), (h - 60) / (ymax - ymin))
        center = np.array([w / 2, h / 2])

        def xy(point):
            return center + (np.asarray(point) - [(xmin + xmax) / 2, (ymin + ymax) / 2]) * [
                scale,
                -scale,
            ]

        font = ("Microsoft YaHei", 9)
        for x in np.arange(0, xmax, 0.5):
            c.create_line(*xy([x, ymin]), *xy([x, ymax]), fill="#e2e8f0")
            c.create_text(*xy([x, ymin]), text=f"{x:g}", font=font, anchor="n", fill="#64748b")
        for y in np.arange(0, ymax, 0.5):
            c.create_line(*xy([xmin, y]), *xy([xmax, y]), fill="#e2e8f0")
            c.create_text(*xy([xmin, y]), text=f"{y:g}", font=font, anchor="e", fill="#64748b")
        c.create_text(
            8,
            8,
            text="世界 XY / m｜蓝色实车 + 彩色估计轮廓（不是第二台车）",
            anchor="nw",
            font=font,
        )
        for name, paint, dash in (("true_poses", "#2563eb", None), (f"{key}_poses", color, (5, 3))):
            poses = arrays[name]
            if i:
                c.create_line(
                    *[v for p in poses[: i + 1, :2] for v in xy(p)], fill=paint, width=2, dash=dash
                )
            pose = poses[i]
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
        if not self.show_landmark.get():
            return
        first_landmark_item = max(c.find_all(), default=0)
        landmark = LANDMARK_WORLD
        mapped = arrays[f"{key}_mapped_landmark"][i]
        c.create_line(*xy(landmark), *xy(mapped), fill="#64748b", dash=(3, 3))
        x, y = xy(landmark)
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#0f766e", outline="")
        c.create_text(
            x,
            y - 12,
            text="真实灯杆" if self.calibration else "真实地标",
            font=font,
            anchor="s",
            fill="#0f766e",
        )
        x, y = xy(mapped)
        c.create_line(x - 5, y - 5, x + 5, y + 5, fill=color, width=2)
        c.create_line(x - 5, y + 5, x + 5, y - 5, fill=color, width=2)
        c.create_text(
            x + 8,
            y + 12,
            text="地图上的灯杆" if self.calibration else "估计落图",
            font=font,
            anchor="nw",
            fill=color,
        )
        for item in c.find_all():
            if item > first_landmark_item:
                c.addtag_withtag("landmark", item)

    def draw_chart(self, arrays, i):
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        name, factor = METRICS[self.metric.get()]
        curves = [(arrays[f"{key}_{name}"] * factor, color) for key, _, _, color in self.variants]
        low = min(0, min(float(curve.min()) for curve, _ in curves))
        high = max(1, max(float(curve.max()) for curve, _ in curves)) * 1.1
        left, right, top, bottom = 60, w - 20, 12, h - 25

        def xy(index, value):
            return left + index / self.replay.steps * (right - left), bottom - (value - low) / (
                high - low
            ) * (bottom - top)

        font = ("Microsoft YaHei", 9)
        for value in np.linspace(low, high, 3):
            _, y = xy(0, value)
            c.create_line(left, y, right, y, fill="#e2e8f0")
            c.create_text(left - 5, y, text=f"{value:.1f}", anchor="e", font=font)
        for curve, color in curves:
            c.create_line(
                *[v for n, value in enumerate(curve) for v in xy(n, value)], fill=color, width=2
            )
        x, _ = xy(i, low)
        c.create_line(x, top, x, bottom, fill="#0f172a", dash=(4, 3))
        for step in np.linspace(0, self.replay.steps, 5):
            x, _ = xy(step, low)
            c.create_text(x, bottom + 13, text=f"{step * DT:g} s", font=font)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/mobile_odometry_2026-09-03"))
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    replays = load_replays(args.results)
    root = tk.Tk()
    OdometryDemo(root, replays, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
