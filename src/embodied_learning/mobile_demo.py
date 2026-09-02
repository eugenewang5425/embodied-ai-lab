"""Read-only, slow lesson-14 replay. Ideal kinematics, not a physics-engine viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import to_parent
from embodied_learning.experiments.mobile_frames import (
    ARRAY_WIDTHS,
    CASES,
    DT,
    GEOMETRY,
    INTERVAL_ARRAYS,
    LANDMARK_WORLD,
    SECONDS,
    SENSOR_IN_BODY,
)
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第十四课 · 差速小车与三个坐标系"


@dataclass
class MobileReplay:
    metadata: dict
    arrays: dict[str, np.ndarray]
    dt: float

    @property
    def steps(self):
        return len(self.arrays["wheels_rad_s"])


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != "differential_drive_frames"
        or report.get("schema_version") != 1
        or report.get("model") != "ideal_no_slip_velocity_kinematics"
        or report.get("dt_s") != DT
        or report.get("duration_s") != SECONDS
        or report.get("wheel_radius_m") != GEOMETRY.radius_m
        or report.get("track_width_m") != GEOMETRY.track_m
        or not np.array_equal(report.get("sensor_in_body"), SENSOR_IN_BODY)
        or not np.array_equal(report.get("landmark_world_m"), LANDMARK_WORLD)
    ):
        raise ValueError("Not a compatible lesson-14 recording")
    if [case["key"] for case in report["cases"]] != [key for key, _, _ in CASES]:
        raise ValueError("Missing, duplicated or unexpected cases")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    replays = []
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {f"{key}_{name}" for key, _, _ in CASES for name in ARRAY_WIDTHS}
        if set(archive.files) != expected_keys:
            raise ValueError("Unexpected archive arrays")
        for case in report["cases"]:
            n = case["steps"]
            if type(n) is not int or n != round(SECONDS / DT):
                raise ValueError("Invalid step count")
            arrays = {}
            for name, width in ARRAY_WIDTHS.items():
                array = archive[f"{case['key']}_{name}"].copy()
                rows = n if name in INTERVAL_ARRAYS else n + 1
                if array.shape != (rows, width) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid array: {name}")
                array.flags.writeable = False
                arrays[name] = array
            replays.append(MobileReplay(case, arrays, DT))
    return replays


class MobileDemo:
    def __init__(self, root, replays, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.replays = root, replays
        self.replay = next(r for r in replays if r.metadata["key"] == "turn_then_drive")
        self.clock = PlaybackClock(self.replay.steps, self.replay.dt, speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1080x700")
        root.minsize(950, 650)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="地标没动，为什么车上的坐标一直变？", font=("Microsoft YaHei", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer, text="理想无打滑 · 轮速直接执行 · 无电机力矩/定位估计 · 默认暂停 0.25×"
        ).pack(anchor="w", pady=(2, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.choice = tk.StringVar(value=self.replay.metadata["label"])
        self.case_combo = ttk.Combobox(
            controls,
            textvariable=self.choice,
            values=[r.metadata["label"] for r in replays],
            state="readonly",
            width=26,
        )
        self.case_combo.pack(side="left", padx=(0, 8))
        self.case_combo.bind("<<ComboboxSelected>>", self.select_case)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=8)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="单步", command=self.step, width=7).pack(side="left", padx=4)
        ttk.Button(controls, text="回到起点", command=lambda: self.seek(0), width=10).pack(
            side="left"
        )
        self.speed = tk.StringVar(value=f"{speed:g}")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        speed_box.pack(side="left", padx=6)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.show_wrong = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls, text="显示错误坐标变换", variable=self.show_wrong, command=self.redraw
        ).pack(side="left")
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
            middle, background="#f8fafc", highlightthickness=0, width=550, height=330
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(middle, text="", width=42, anchor="nw", justify="left")
        self.stats.pack(side="right", fill="y", padx=(12, 0))
        self.chart = tk.Canvas(outer, background="#ffffff", highlightthickness=0, height=140)
        self.chart.pack(fill="x", pady=(6, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text="W 世界（灰）→ B 车体（蓝）→ S 传感器（紫）｜x 向前，y 向左，逆时针为正｜图与数字均不放大运动",
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.chart.bind("<Configure>", lambda _: self.redraw())
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def select_case(self, _=None):
        self.replay = next(r for r in self.replays if r.metadata["label"] == self.choice.get())
        self.clock = PlaybackClock(self.replay.steps, self.replay.dt, self.clock.speed)
        self.last_tick = time.perf_counter()
        self.redraw()

    def change_speed(self, _=None):
        self.clock.set_speed(float(self.speed.get()))
        self.last_tick = time.perf_counter()
        self.redraw()

    def toggle(self):
        self.clock.toggle()
        self.last_tick = time.perf_counter()
        self.redraw()

    def step(self):
        self.clock.step()
        self.redraw()

    def seek(self, value):
        if not self.updating:
            self.clock.seek(round(float(value)))
            self.redraw()

    def slider_changed(self, value):
        # Tk can dispatch Scale.set callbacks after redraw clears `updating`.
        if round(float(value)) != self.clock.index:
            self.seek(value)

    def tick(self):
        now = time.perf_counter()
        if self.clock.advance(now - self.last_tick):
            self.redraw()
        self.last_tick = now
        self.after_id = self.root.after(20, self.tick)

    def close(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.root.destroy()

    def redraw(self):
        if not hasattr(self, "status"):
            return
        i, arrays = self.clock.index, self.replay.arrays
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        pose, sensor_pose = arrays["poses"][i], arrays["sensor_world_poses"][i]
        body, sensor = arrays["landmark_body"][i], arrays["landmark_sensor"][i]
        world = arrays["reconstructed_world"][i]
        error = np.linalg.norm(world - LANDMARK_WORLD)
        wrong_error = np.linalg.norm(arrays["wrong_world"][i] - LANDMARK_WORLD)
        if i < self.replay.steps:
            left, right = arrays["wheels_rad_s"][i]
            v, omega = arrays["body_velocity"][i]
            action = f"下一步轮速：左 {left:+.3f} / 右 {right:+.3f} rad/s\n车体前进 v={v:+.3f} m/s\n车体转速 ω={np.rad2deg(omega):+.1f} °/s"
        else:
            action = (
                "回放结束：没有下一步轮速\n终点不是反馈控制到达结果\n（本课按预设时间执行动作）"
            )
        self.stats.configure(
            text=(
                f"时间 {i * self.replay.dt:.2f} / {self.replay.steps * self.replay.dt:.2f} s\n\n"
                f"车体 B 在世界 W 中\nx={pose[0]:+.3f} m   y={pose[1]:+.3f} m\nθ={np.rad2deg(pose[2]):+.1f}°\n\n"
                f"{action}\n\n"
                f"同一个地标的坐标（m）\n世界 W：({LANDMARK_WORLD[0]:+.3f}, {LANDMARK_WORLD[1]:+.3f})\n"
                f"车体 B：({body[0]:+.3f}, {body[1]:+.3f})\n传感器 S：({sensor[0]:+.3f}, {sensor[1]:+.3f})\n"
                f"变回世界：({world[0]:+.3f}, {world[1]:+.3f})\n\n"
                f"正确转换误差：{error:.1e} m\n错误转换误差：{wrong_error:.3f} m"
            )
        )
        self.draw_map(arrays, i, pose, sensor_pose)
        self.draw_chart(arrays, i)
        phase = "转弯" if i * self.replay.dt < 2 else "直行"
        if self.replay.metadata["key"] != "turn_then_drive":
            phase = self.replay.metadata["label"]
        state = "结束" if i == self.replay.steps else ("暂停" if self.clock.paused else "播放中")
        self.status.configure(
            text=f"{state} · {self.clock.speed:g}× · 当前动作：{phase} · 单步 {self.replay.dt:g} s · 传感器安装：前 0.12 m / 左 0.04 m / 偏转 +30°"
        )

    def draw_map(self, arrays, i, pose, sensor_pose):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        # Fixed square bounds for every case and both correct/wrong landmark traces.
        low, high = -1.5, 1.6
        scale = min(w - 40, h - 40) / (high - low)
        center = np.array([w / 2, h / 2])

        def xy(point):
            return center + (np.asarray(point) - (low + high) / 2) * [scale, -scale]

        font = ("Microsoft YaHei", 9)
        for value in np.arange(-1.5, 1.6, 0.5):
            c.create_line(*xy([value, low]), *xy([value, high]), fill="#e2e8f0")
            c.create_line(*xy([low, value]), *xy([high, value]), fill="#e2e8f0")
        c.create_text(
            8, 8, text="俯视图｜方格 0.5 m｜实线为已走轨迹", anchor="nw", font=font, fill="#475569"
        )

        def axes(frame, name, color, length):
            # Different direction-arrow lengths keep initially coincident frames legible.
            # These arrows are coordinate guides, not physical parts or sensor range.
            for axis, end in (("x", [length, 0]), ("y", [0, length])):
                tip = to_parent(frame, end)
                c.create_line(*xy(frame[:2]), *xy(tip), fill=color, width=2, arrow="last")
                c.create_text(
                    *xy(tip),
                    text=f"{name}{axis}",
                    fill=color,
                    anchor="nw" if name == "B" else "sw",
                    font=font,
                )

        axes(np.zeros(3), "W", "#64748b", 1.0)
        if i > 0:
            c.create_line(
                *[v for point in arrays["poses"][: i + 1, :2] for v in xy(point)],
                fill=self.replay.metadata["color"],
                width=3,
            )
        corners = [
            to_parent(pose, p) for p in ([0.18, 0.12], [0.18, -0.12], [-0.18, -0.12], [-0.18, 0.12])
        ]
        c.create_polygon(
            *[v for p in corners for v in xy(p)], fill="#dbeafe", outline="#2563eb", width=2
        )
        for side, label in ((1, "L"), (-1, "R")):
            a, b = [to_parent(pose, [x, side * GEOMETRY.track_m / 2]) for x in (-0.08, 0.08)]
            c.create_line(*xy(a), *xy(b), fill="#0f172a", width=5)
            c.create_text(*xy(to_parent(pose, [-0.14, side * 0.21])), text=label, font=font)
        c.create_line(*xy(sensor_pose[:2]), *xy(LANDMARK_WORLD), fill="#a78bfa", dash=(3, 4))
        axes(pose, "B", "#2563eb", 0.55)
        axes(sensor_pose, "S", "#9333ea", 0.8)
        x, y = xy(LANDMARK_WORLD)
        c.create_oval(x - 5, y - 5, x + 5, y + 5, fill="#0f766e", outline="")
        c.create_text(
            x + 12, y - 12, text="固定地标 / 正确变换", anchor="w", font=font, fill="#0f766e"
        )
        if self.show_wrong.get():
            trace = arrays["wrong_world"][: i + 1]
            if len(trace) > 1:
                c.create_line(*[v for p in trace for v in xy(p)], fill="#dc2626", dash=(4, 3))
            x, y = xy(trace[-1])
            c.create_line(x - 6, y - 6, x + 6, y + 6, fill="#dc2626", width=2)
            c.create_line(x - 6, y + 6, x + 6, y - 6, fill="#dc2626", width=2)
            c.create_text(
                x + 8, y + 14, text="错误：只加平移", anchor="w", fill="#dc2626", font=font
            )

    def draw_chart(self, arrays, i):
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        readings = arrays["landmark_sensor"]
        low, high = float(readings.min()) - 0.15, float(readings.max()) + 0.15
        left, top, right, bottom = 65, 25, w - 20, h - 25

        def xy(index, value):
            return left + index / self.replay.steps * (right - left), bottom - (value - low) / (
                high - low
            ) * (bottom - top)

        font = ("Microsoft YaHei", 9)
        c.create_text(
            8,
            4,
            text="固定地标的传感器坐标：蓝 x / 橙 y（m）；完整记录 + 当前时刻游标",
            anchor="nw",
            font=font,
        )
        for value in np.linspace(low, high, 3):
            _, y = xy(0, value)
            c.create_line(left, y, right, y, fill="#e2e8f0")
            c.create_text(left - 6, y, text=f"{value:+.2f}", anchor="e", font=font)
        for axis, color in enumerate(("#2563eb", "#ea580c")):
            c.create_line(
                *[v for n, value in enumerate(readings[:, axis]) for v in xy(n, value)],
                fill=color,
                width=2,
            )
        x, _ = xy(i, low)
        c.create_line(x, top, x, bottom, fill="#0f172a", dash=(4, 3))
        for second in range(5):
            x, _ = xy(second / self.replay.dt, low)
            c.create_text(x, bottom + 14, text=f"{second} s", font=font)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/mobile_frames_2026-09-03"))
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    replays = load_replays(args.results)
    root = tk.Tk()
    MobileDemo(root, replays, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
