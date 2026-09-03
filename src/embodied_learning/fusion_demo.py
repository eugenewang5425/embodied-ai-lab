"""Lesson 19 read-only teaching replay: measurements, pose solving, then fusion."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import compose, rotation, to_parent
from embodied_learning.experiments.landmark_fusion import (
    DEFAULT_RESULTS,
    DT,
    LANDMARKS,
    METHODS,
    SCENARIOS,
    SENSOR_IN_BODY,
    load_recording,
)
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.odometry import heading_error
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第十九课 · 看观测 → 解位置 → 融合慢放"
METRICS = ("位置误差 / cm", "朝向误差 / °")


def pose_text(pose):
    return f"x={pose[0]:+.3f} m，y={pose[1]:+.3f} m，θ={np.rad2deg(pose[2]):+.2f}°"


class FusionDemo(MobileDemo):
    """Shared clock for a measurement sheet and a three-estimator replay."""

    def __init__(self, root, routes, report, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.routes, self.report = root, routes, report
        self.route_key, self.run = "straight", 0
        self.clock = PlaybackClock(len(self.current()["truth"]) - 1, DT, speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1180x830")
        root.minsize(1080, 780)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="轮子告诉你走了多少，地标告诉你现在在哪",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "仿真测距/测角，非摄像头识别｜编码器每 0.04 s，地标每 2 s｜"
                "固定标定已做；重置不是去噪，也不是控制小车"
            ),
        ).pack(anchor="w", pady=(3, 8))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.case_box = ttk.Combobox(
            controls, values=[s[1] for s in SCENARIOS], state="readonly", width=23
        )
        self.case_box.current(0)
        self.case_box.pack(side="left")
        self.case_box.bind("<<ComboboxSelected>>", self.select_case)
        ttk.Label(controls, text="样本 #").pack(side="left", padx=(8, 0))
        self.sample = tk.IntVar(value=0)
        self.sample_box = ttk.Spinbox(
            controls,
            from_=0,
            to=report["runs"] - 1,
            textvariable=self.sample,
            width=4,
            command=self.select_sample,
        )
        self.sample_box.pack(side="left")
        self.sample_box.bind("<Return>", self.select_sample)
        self.sample_box.bind("<FocusOut>", self.select_sample)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=6)
        self.play_button.pack(side="left", padx=5)
        ttk.Button(controls, text="单步", command=self.step, width=6).pack(side="left")
        ttk.Button(controls, text="起点", command=lambda: self.seek(0), width=6).pack(
            side="left", padx=4
        )
        self.next_button = ttk.Button(controls, text="下一次观测", command=self.next_observation)
        self.next_button.pack(side="left")
        self.worse_button = ttk.Button(
            controls, text="看一次校正变差", command=self.next_worse_update
        )
        self.worse_button.pack(side="left", padx=4)
        self.speed = tk.StringVar(value=f"{speed:g}")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        speed_box.pack(side="left", padx=4)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        ttk.Label(controls, text="倍速").pack(side="left")
        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.clock.steps,
            orient="horizontal",
            showvalue=False,
            command=self.slider_changed,
        )
        self.timeline.pack(fill="x")
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        measure = ttk.Frame(self.notebook, padding=10)
        replay = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(measure, text="① 观测怎样解出位置")
        self.notebook.add(replay, text="② 三种方法同图慢放")
        self.measure_status = ttk.Label(measure, text="")
        self.measure_status.pack(anchor="w", pady=(0, 6))
        columns = ("landmark", "world", "range", "bearing", "local", "residual")
        self.readings_table = ttk.Treeview(measure, columns=columns, show="headings", height=3)
        for key, title, width in zip(
            columns,
            (
                "地标编号（已知）",
                "地图坐标 X, Y / m",
                "测距 / m",
                "传感器系测角 / °",
                "换算局部 x, y / m",
                "配准残差 / cm",
            ),
            (130, 190, 100, 150, 210, 130),
        ):
            self.readings_table.heading(key, text=title)
            self.readings_table.column(key, width=width, anchor="center")
        self.readings_table.pack(fill="x")
        self.solve_text = ttk.Label(measure, justify="left", anchor="nw")
        self.solve_text.pack(fill="both", expand=True, pady=10)
        ttk.Label(
            measure,
            text=(
                "σ距离=0.01 m、σ角度=0.01 rad：标准差，不是 ± 上限。三个点一起拟合可分摊噪声，不能消除噪声。\n"
                "地图坐标、地标身份、安装位置均假设已知；没有识别、遮挡、打滑或延迟模型。"
            ),
        ).pack(anchor="w")
        self.canvas = tk.Canvas(
            replay, background="#f8fafc", highlightthickness=0, width=550, height=300
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(replay, width=45, justify="left", anchor="nw")
        self.stats.pack(side="right", fill="y", padx=10)
        self.update_text = ttk.Label(outer, justify="left")
        self.update_text.pack(anchor="w", pady=(7, 5))
        chart_controls = ttk.Frame(outer)
        chart_controls.pack(fill="x")
        self.metric = tk.StringVar(value=METRICS[0])
        metric_box = ttk.Combobox(
            chart_controls, textvariable=self.metric, values=METRICS, state="readonly", width=16
        )
        metric_box.pack(side="left")
        metric_box.bind("<<ComboboxSelected>>", lambda _: self.redraw())
        self.show_hold = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            chart_controls,
            text="显示纯观测保持（取消可放大另两条曲线）",
            variable=self.show_hold,
            command=self.redraw,
        ).pack(side="left", padx=8)
        ttk.Label(chart_controls, text="紫=里程计　灰虚线=保持　橙=融合　绿点=观测到达").pack(
            side="left"
        )
        self.chart = tk.Canvas(outer, background="white", highlightthickness=0, height=150)
        self.chart.pack(fill="x")
        self.status = ttk.Label(outer)
        self.status.pack(anchor="w", pady=(5, 0))
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
        index = self.case_box.current()
        if index < 0:
            return
        self.route_key = SCENARIOS[index][0]
        self.clock = PlaybackClock(len(self.current()["truth"]) - 1, DT, self.clock.speed)
        self.timeline.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.redraw()

    def select_sample(self, _=None):
        try:
            value = int(self.sample_box.get())
        except ValueError:
            value = self.run
        self.run = max(0, min(value, self.report["runs"] - 1))
        self.sample.set(self.run)
        self.redraw()

    def next_observation(self):
        frames = self.current()["observation_frames"]
        target = next((f for f in frames if f > self.clock.index), frames[-1])
        self.seek(target)

    def worse_frames(self):
        route = self.current()
        frames = route["observation_frames"]
        before = np.linalg.norm(
            route["prior"][self.run, frames, :2] - route["truth"][frames, :2], axis=1
        )
        after = np.linalg.norm(
            route["fused"][self.run, frames, :2] - route["truth"][frames, :2], axis=1
        )
        return frames[after > before + 1e-12]

    def next_worse_update(self):
        frames = self.worse_frames()
        if len(frames):
            self.seek(next((f for f in frames if f > self.clock.index), frames[0]))

    def redraw(self):
        if not hasattr(self, "status"):
            return
        route, i = self.current(), self.clock.index
        frames = route["observation_frames"]
        observed = np.flatnonzero(frames <= i)
        sample = int(observed[-1]) if len(observed) else None
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        self.worse_button.configure(state="normal" if len(self.worse_frames()) else "disabled")
        self.draw_measurements(route, sample)
        self.draw_map(route, i)
        self.draw_chart(route, i)
        lines = [
            f"时间 {i * DT:.2f} s｜样本 #{self.run}",
            "",
            "真实位姿（只供评价）：",
            pose_text(route["truth"][i]),
        ]
        for method, label, _ in METHODS:
            error = np.linalg.norm(route[method][self.run, i, :2] - route["truth"][i, :2]) * 100
            lines.extend(
                [
                    "",
                    f"{label}：",
                    pose_text(route[method][self.run, i]),
                    f"位置误差距离 {error:.2f} cm",
                ]
            )
        self.stats.configure(text="\n".join(lines))
        if sample is None:
            update = "还没有地标观测：融合与里程计一致，都是从已知初始位姿推算。点击“下一次观测”。"
        else:
            frame = frames[sample]
            prior, post = route["prior"][self.run, frame], route["fused"][self.run, frame]
            before = np.linalg.norm(prior[:2] - route["truth"][frame, :2]) * 100
            after = np.linalg.norm(post[:2] - route["truth"][frame, :2]) * 100
            delta = post - prior
            update = (
                f"最近一次校正 {frame * DT:.2f} s：位置误差 {before:.2f} → {after:.2f} cm"
                f"（{'变差：这次观测更不准' if after > before else '改善'}；真值只用于这条评价）\n"
                f"估计修正量 Δx={delta[0] * 100:+.2f} cm，Δy={delta[1] * 100:+.2f} cm，"
                f"Δθ={np.rad2deg(delta[2]):+.2f}°；之后 {(i - frame) * DT:.2f} s 用编码器继续推算。"
            )
        self.update_text.configure(text=update)
        state = "结束" if i == self.clock.steps else ("暂停" if self.clock.paused else "播放中")
        self.status.configure(
            text=f"{state} · {self.clock.speed:g}× · {i * DT:.2f}/{self.clock.steps * DT:.2f} s · "
            "已录制实验回放；曲线显示整段，游标标当前；算法无未来观测输入"
        )

    def draw_measurements(self, route, sample):
        self.readings_table.delete(*self.readings_table.get_children())
        if sample is None:
            self.measure_status.configure(
                text="尚无观测；首次测距测角在 2.00 s 到来，不提前展示未来读数。"
            )
            self.solve_text.configure(
                text=(
                    "① 仿真器：真实小车位姿 + 已知地标 → 几何距离/方位角 → 加噪声 → 传感器读数。\n\n"
                    "② 定位器：只看三组读数和地图坐标，把局部点集旋转、平移，与地图控制点对齐。\n\n"
                    "③ 扣除传感器的安装偏移与角度，得到车体位姿。不是把真值直接交给定位器。\n\n"
                    "先点“下一次观测”看真实记录的计算，再切到第二页看三组估计。"
                )
            )
            return
        frame = route["observation_frames"][sample]
        reading = route["observations"][self.run, sample]
        body = route["body_samples"][self.run, sample]
        sensor = compose(body, SENSOR_IN_BODY)
        local = np.column_stack(
            [reading[:, 0] * np.cos(reading[:, 1]), reading[:, 0] * np.sin(reading[:, 1])]
        )
        fitted = local @ rotation(sensor[2]).T + sensor[:2]
        residuals = np.linalg.norm(fitted - LANDMARKS, axis=1)
        for index, (landmark, z, q, residual) in enumerate(
            zip(LANDMARKS, reading, local, residuals), 1
        ):
            self.readings_table.insert(
                "",
                "end",
                values=(
                    f"P{index}",
                    f"{landmark[0]:+.2f}, {landmark[1]:+.2f}",
                    f"{z[0]:.4f}",
                    f"{np.rad2deg(z[1]):+.2f}",
                    f"{q[0]:+.4f}, {q[1]:+.4f}",
                    f"{residual * 100:.3f}",
                ),
            )
        self.measure_status.configure(
            text=f"最近观测 {frame * DT:.2f} s｜样本 #{self.run}｜两次观测间这张表保持旧读数，不冒充实时测量"
        )
        self.solve_text.configure(
            text=(
                "1. 距离 r + 方位角 β → 局部坐标 q = (r cos β, r sin β)。角度相对传感器前方，不是世界北向。\n\n"
                "2. 同时对齐三个点：寻找旋转 R 和平移 t，使 Σ ‖R qᵢ + t − Pᵢ‖² 最小；不拟合缩放。\n"
                f"   解出的传感器位姿：{pose_text(sensor)}\n\n"
                "3. 传感器装在车体前方 0.12 m、左侧 0.04 m，朝向偏转 +30°。\n"
                f"   反算车体位姿：{pose_text(body)}\n\n"
                f"点集配准残差 RMS = {np.sqrt(np.mean(residuals**2)) * 100:.3f} cm。"
                "这是点集吻合程度，不是车体定位误差，也不能证明地图基准正确。"
            )
        )

    def draw_map(self, route, i):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        points = np.vstack(
            [LANDMARKS, route["truth"][:, :2]] + [route[k][self.run, :, :2] for k, _, _ in METHODS]
        )
        lo, hi = points.min(axis=0) - 0.4, points.max(axis=0) + 0.4
        scale = min((w - 60) / (hi[0] - lo[0]), (h - 65) / (hi[1] - lo[1]))

        def xy(point):
            return np.array([w / 2, h / 2 + 10]) + (np.asarray(point) - (lo + hi) / 2) * [
                scale,
                -scale,
            ]

        font = ("Microsoft YaHei", 9)
        c.create_text(
            8, 8, text="世界 XY / m｜蓝=真值 黑三角=已知地标；并非四辆车", anchor="nw", font=font
        )
        for axis in (0, 1):
            for value in np.arange(np.ceil(lo[axis]), hi[axis], 1):
                a, b = lo.copy(), hi.copy()
                a[axis] = b[axis] = value
                c.create_line(*xy(a), *xy(b), fill="#e2e8f0")
                c.create_text(
                    *xy(a), text=f"{value:g}", anchor="n" if axis == 0 else "e", font=font
                )
        for landmark in LANDMARKS:
            x, y = xy(landmark)
            c.create_polygon(x, y - 7, x - 6, y + 5, x + 6, y + 5, fill="#0f172a")
        for key, color in [("truth", "#2563eb")] + [(k, color) for k, _, color in METHODS]:
            poses = route[key] if key == "truth" else route[key][self.run]
            if i:
                c.create_line(
                    *[v for p in poses[: i + 1, :2] for v in xy(p)],
                    fill=color,
                    width=2,
                    dash=(4, 3) if key == "held" else (),
                )
            x, y = xy(poses[i, :2])
            c.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline="white")
            c.create_line(x, y, *xy(to_parent(poses[i], [0.22, 0])), fill=color, width=2)
        for frame in route["observation_frames"]:
            if frame <= i:
                x, y = xy(route["held"][self.run, frame, :2])
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#0f766e", outline="")

    def draw_chart(self, route, i):
        c = self.chart
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 50:
            return
        curves = {}
        for key, _, color in METHODS:
            if key == "held" and not self.show_hold.get():
                continue
            poses = route[key][self.run]
            values = (
                np.linalg.norm(poses[:, :2] - route["truth"][:, :2], axis=1) * 100
                if self.metric.get() == METRICS[0]
                else np.rad2deg(heading_error(poses[:, 2], route["truth"][:, 2]))
            )
            curves[key] = (values, color)
        values = np.concatenate([curve for curve, _ in curves.values()])
        low, high = min(0.0, values.min()), max(0.1, values.max())
        margin = (high - low) * 0.08
        low = 0.0 if self.metric.get() == METRICS[0] else low - margin
        high += margin

        def xy(frame, value):
            return 60 + frame / self.clock.steps * (w - 80), h - 25 - (value - low) / (
                high - low
            ) * (h - 40)

        font = ("Microsoft YaHei", 9)
        for v in np.linspace(low, high, 4):
            _, y = xy(0, v)
            c.create_line(60, y, w - 20, y, fill="#e2e8f0")
            c.create_text(54, y, text=f"{v:.1f}", anchor="e", font=font)
        for key, (curve, color) in curves.items():
            c.create_line(
                *[v for frame, value in enumerate(curve) for v in xy(frame, value)],
                fill=color,
                width=2,
                dash=(4, 3) if key == "held" else (),
            )
        for frame in route["observation_frames"]:
            x, y = xy(frame, curves["fused"][0][frame])
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#0f766e", outline="white")
        for frame in np.linspace(0, self.clock.steps, 5):
            x, _ = xy(frame, 0)
            c.create_text(x, h - 10, text=f"{frame * DT:g} s", font=font)
        x, _ = xy(i, 0)
        c.create_line(x, 15, x, h - 25, fill="#0f172a", dash=(3, 3))


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    routes, report = load_recording(args.results)
    root = tk.Tk()
    FusionDemo(root, routes, report, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
