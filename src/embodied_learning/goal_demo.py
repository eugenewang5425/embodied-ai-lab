"""Motion-first replay of new point-goal feedback trials, not live ROS."""

import argparse
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import to_parent
from embodied_learning.experiments.goal_reaching import (
    CASES,
    DEFAULT_RESULTS,
    DT,
    METHODS,
    load_recording,
)
from embodied_learning.landmark_localization import LANDMARKS
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第二十一课 · 小车真的停到目标了吗？"


class GoalDemo(MobileDemo):
    def __init__(self, root, report, records):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk

        self.root, self.report, self.records = root, report, records
        self.case, self.run = "near", 0
        self.clock = PlaybackClock(self.steps(), DT)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1180x800")
        root.minsize(1080, 740)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        self.map_font = tkfont.Font(root=root, family="Microsoft YaHei", size=10)
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="不再按时间走一圈：根据“我在哪”，自己驶向目标",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="任务：车轴中心停在绿色目标区内（半径 3 cm）｜控制器只能看估计位置，真值只供传感器生成与验收\n本地 Python 反馈运动学仿真的记录回放；不是实车或实时 ROS；没有惯性、打滑与避障。",
        ).pack(anchor="w", pady=6)
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.case_box = ttk.Combobox(
            controls,
            state="readonly",
            values=[f"{label} {goal}" for _, label, goal in CASES],
            width=22,
        )
        self.case_box.current(0)
        self.case_box.pack(side="left")
        self.case_box.bind("<<ComboboxSelected>>", self.select_case)
        ttk.Label(controls, text="样本 #").pack(side="left", padx=(6, 0))
        self.sample = tk.IntVar(value=0)
        self.sample_box = ttk.Spinbox(
            controls,
            textvariable=self.sample,
            from_=0,
            to=report["runs"] - 1,
            width=3,
            command=self.select_sample,
        )
        self.sample_box.pack(side="left")
        self.sample_box.bind("<Return>", self.select_sample)
        self.sample_box.bind("<FocusOut>", self.select_sample)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=6)
        self.play_button.pack(side="left", padx=4)
        ttk.Button(controls, text="单步", command=self.step, width=6).pack(side="left")
        ttk.Button(controls, text="起点", command=self.restart, width=6).pack(side="left", padx=4)
        self.finish_button = ttk.Button(controls, text="看停车结果", command=self.finish)
        self.finish_button.pack(side="left")
        self.failure_button = ttk.Button(
            controls, text="看停偏样本", command=self.show_false_arrival
        )
        self.failure_button.pack(side="left", padx=4)
        self.speed = tk.StringVar(value="0.25")
        box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", self.change_speed)
        options = ttk.Frame(outer)
        options.pack(fill="x", pady=4)
        self.zoom = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options, text="放大目标附近（厘米尺度）", variable=self.zoom, command=self.redraw
        ).pack(side="left")
        ttk.Label(
            options,
            text="左右是两次独立实验：实色车＝实际位置；灰色虚线／空圈＝该车的估计；不是同一条真实轨迹。",
        ).pack(side="left", padx=6)
        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.clock.steps,
            orient="horizontal",
            showvalue=False,
            command=self.slider_changed,
        )
        self.timeline.pack(fill="x")
        panels = ttk.Frame(outer)
        panels.pack(fill="both", expand=True)
        panels.rowconfigure(0, weight=1)
        self.canvases, self.notes, self.labels = {}, {}, {}
        for column, (method, label, color) in enumerate(METHODS):
            panels.columnconfigure(column, weight=1, uniform="panels")
            panel = ttk.Frame(panels, padding=6)
            panel.grid(row=0, column=column, sticky="nsew")
            ttk.Label(
                panel, text=label, foreground=color, font=("Microsoft YaHei", 13, "bold")
            ).pack(anchor="w")
            self.labels[method] = ttk.Label(panel, text="", wraplength=510, justify="left")
            self.labels[method].pack(anchor="w", pady=4)
            canvas = tk.Canvas(
                panel, background="#f8fafc", highlightthickness=0, width=500, height=330
            )
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _: self.redraw())
            self.canvases[method] = canvas
            self.notes[method] = ttk.Label(panel, text="", wraplength=510, justify="left")
            self.notes[method].pack(anchor="w", pady=7)
        self.summary_label = ttk.Label(outer, text="", wraplength=1050, justify="left")
        self.summary_label.pack(anchor="w", pady=7)
        ttk.Label(
            outer,
            text="闭环：估计位置 → 算目标距离和方向 → 左右轮速 → 真实运动 → 新读数 → 再估计。停车后本次任务结束。",
        ).pack(anchor="w")
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=6)
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def current(self, method):
        return self.records[(self.case, self.run, method)]

    def steps(self):
        return max(self.current(m)[1]["steps"] for m, _, _ in METHODS)

    def reset_clock(self):
        self.clock = PlaybackClock(self.steps(), DT, self.clock.speed)
        self.timeline.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.zoom.set(False)
        self.redraw()

    def select_case(self, _=None):
        self.case = CASES[self.case_box.current()][0]
        self.reset_clock()

    def select_sample(self, _=None):
        try:
            value = int(self.sample_box.get())
        except ValueError:
            value = self.run
        value = max(0, min(value, self.report["runs"] - 1))
        self.sample.set(value)
        if value != self.run:
            self.run = value
            self.reset_clock()

    def restart(self):
        self.zoom.set(False)
        self.seek(0)

    def finish(self):
        self.zoom.set(True)
        self.seek(self.clock.steps)

    def failed_runs(self):
        return [
            r
            for r in range(self.report["runs"])
            if self.records[(self.case, r, "odom")][1]["false_arrival"]
        ]

    def show_false_arrival(self):
        candidates = self.failed_runs()
        if candidates:
            self.run = next((r for r in candidates if r > self.run), candidates[0])
            self.sample.set(self.run)
            self.reset_clock()
            self.finish()

    def redraw(self):
        if not hasattr(self, "status"):
            return
        self.updating = True
        self.timeline.set(self.clock.index)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        self.failure_button.configure(state="normal" if self.failed_runs() else "disabled")
        for method, _, color in METHODS:
            arrays, trial = self.current(method)
            frame = min(self.clock.index, trial["steps"])
            goal = np.array(trial["goal"])
            actual = np.linalg.norm(arrays["truth"][frame, :2] - goal) * 100
            believed = np.linalg.norm(arrays["estimated"][frame, :2] - goal) * 100
            done = frame == trial["steps"]
            mode = int(arrays["modes"][frame])
            phase = {
                0: "根据估计，转向并接近目标",
                1: "目标在侧后方，先原地转向",
                2: "自认为进入 2 cm，尝试停稳",
                3: "控制器：我已到达，结束任务",
                4: "达到时限，任务停止",
            }[mode]
            self.labels[method].configure(
                text=f"{phase}\n它认为距目标 {believed:.2f} cm；实际相距 {actual:.2f} cm"
            )
            left, right = arrays["commands"][frame]
            if done:
                outcome = (
                    "实际通过：车轴中心在 3 cm 目标区内"
                    if trial["true_success"]
                    else (
                        "误判到达：估计在区内，实际车停偏了"
                        if trial["false_arrival"]
                        else "超时：没有完成目标"
                    )
                )
                note = f"{outcome}\n{trial['duration_s']:.2f} s 停止；之后仅保持画面，不继续观测或纠偏。"
            else:
                note = f"下一步轮速：左 {left:+.2f} / 右 {right:+.2f} rad/s\n"
                note += (
                    "地标刚到：修正估计，接下来的轮速会受影响。"
                    if frame in arrays["observation_frames"] and method == "fused"
                    else "动作由本帧估计决定，不是提前写好的时间表。"
                )
            self.notes[method].configure(text=note)
            self.draw_map(method, arrays, frame, goal, color)
        comparison = next(c for c in self.report["comparisons"] if c["case"] == self.case)
        stats = comparison["methods"]
        self.summary_label.configure(
            text=f"当前样本 #{self.run}；本目标共 {self.report['runs']} 次：实际通过数 轮子 {stats['odom']['true_success_count']} / {self.report['runs']}，轮子＋地标 {stats['fused']['true_success_count']} / {self.report['runs']}。\n只改定位来源；控制器、限速、判据一致。噪声按种子／时刻配对，但两辆仿真车走出的路线和测量值可以不同。"
        )
        self.status.configure(
            text=f"{'暂停' if self.clock.paused else '播放中'} · {self.clock.speed:g}× · {self.clock.index * DT:.2f} / {self.clock.steps * DT:.2f} s · 只读回放；绿色区域按真实比例绘制，放大不改误差"
        )

    def draw_map(self, method, arrays, frame, goal, color):
        c = self.canvases[method]
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 150:
            return
        local = self.zoom.get()
        if local:
            low, high = goal - 0.22, goal + 0.22
        else:
            points = np.vstack(
                [
                    LANDMARKS,
                    goal,
                    *[
                        self.current(m)[0][k][:, :2]
                        for m, _, _ in METHODS
                        for k in ("truth", "estimated")
                    ],
                ]
            )
            low, high = points.min(0) - 0.35, points.max(0) + 0.35
        font = self.map_font
        line_height = font.metrics("linespace")
        # Reserve separate title, tick and caption rows at the actual Windows DPI.
        scale = min(
            (w - 6 * line_height) / (high[0] - low[0]),
            (h - 5 * line_height) / (high[1] - low[1]),
        )
        if scale <= 0:
            return

        def xy(point):
            return np.array([w / 2, (h - line_height) / 2]) + (
                np.asarray(point) - (low + high) / 2
            ) * [
                scale,
                -scale,
            ]

        def inside(point):
            return bool(np.all(point >= low) and np.all(point <= high))

        c.create_text(
            8,
            8,
            anchor="nw",
            text="目标局部放大｜标记是车轴中心，不是车体大小"
            if local
            else "运动俯视图｜X 向右，Y 向上｜方格 0.5 m",
            font=font,
        )
        spacing = 0.05 if local else 0.5
        label_every = max(1, int(np.ceil((font.measure("-20") + 8) / (spacing * scale))))
        for axis in (0, 1):
            origin = goal[axis] if local else 0
            start = np.ceil((low[axis] - origin) / spacing) * spacing + origin
            for value in np.arange(start, high[axis], spacing):
                a, b = low.copy(), high.copy()
                a[axis] = b[axis] = value
                c.create_line(*xy(a), *xy(b), fill="#e2e8f0")
                if local and round((value - origin) / spacing) % label_every == 0:
                    c.create_text(
                        *xy(a),
                        text=f"{(value - origin) * 100:+.0f}",
                        anchor="n" if axis == 0 else "e",
                        font=font,
                        tags="axis_label",
                    )
        x, y = xy(goal)
        radius = self.report["controller"]["true_acceptance_radius_m"] * scale
        c.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#dcfce7",
            outline="#15803d",
            width=2,
            tags="target_zone",
        )
        c.create_line(x - 7, y, x + 7, y, fill="#15803d", width=2)
        c.create_line(x, y - 7, x, y + 7, fill="#15803d", width=2)
        c.create_text(x + 10, y - 15, text="目标", anchor="w", fill="#15803d", font=font)
        if not local:
            for p in LANDMARKS:
                lx, ly = xy(p)
                c.create_polygon(lx, ly - 6, lx - 5, ly + 5, lx + 5, ly + 5, fill="#15803d")
        for key, stroke in (("truth", color), ("estimated", "#64748b")):
            path = arrays[key][: frame + 1, :2]
            if local:
                for a, b in pairwise(path):
                    if inside(a) and inside(b):
                        c.create_line(
                            *xy(a),
                            *xy(b),
                            fill=stroke,
                            dash=() if key == "truth" else (3, 3),
                            tags=f"{key}_trail",
                        )
            elif frame:
                c.create_line(
                    *[v for p in path for v in xy(p)],
                    fill=stroke,
                    width=2,
                    dash=() if key == "truth" else (4, 4),
                    tags=f"{key}_trail",
                )
            pose = arrays[key][frame]
            if not inside(pose[:2]):
                continue
            cx, cy = xy(pose[:2])
            if key == "truth" and not local:
                body = [[-0.16, -0.11], [0.16, -0.11], [0.16, 0.11], [-0.16, 0.11]]
                c.create_polygon(
                    *[v for p in body for v in xy(to_parent(pose, p))],
                    outline=color,
                    fill="white",
                    width=3,
                    tags="true_body",
                )
                for side in (-0.15, 0.15):
                    c.create_line(
                        *xy(to_parent(pose, [-0.075, side])),
                        *xy(to_parent(pose, [0.075, side])),
                        fill="#0f172a",
                        width=4,
                    )
            c.create_oval(
                cx - 6,
                cy - 6,
                cx + 6,
                cy + 6,
                fill=stroke if key == "truth" else "",
                outline=stroke,
                width=2,
                dash=() if key == "truth" else (2, 2),
                tags=f"{key}_centre",
            )
            c.create_line(
                cx,
                cy,
                cx + 24 * np.cos(pose[2]),
                cy - 24 * np.sin(pose[2]),
                fill=stroke,
                width=2,
                arrow="last",
                tags=f"{key}_heading",
            )
        c.create_text(
            8,
            h - 10,
            anchor="sw",
            text="局部 ΔX / ΔY：cm；估计进圈，不代表实际进圈"
            if local and inside(arrays["truth"][frame, :2])
            else (
                "车辆尚未进入局部视野；可点“看停车结果”"
                if local
                else "实线是实际走出的路；灰色虚线是估计轨迹"
            ),
            font=font,
            fill="#475569",
            tags="map_caption",
        )


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    report, records = load_recording(args.results)
    root = tk.Tk()
    GoalDemo(root, report, records)
    root.mainloop()


if __name__ == "__main__":
    main()
