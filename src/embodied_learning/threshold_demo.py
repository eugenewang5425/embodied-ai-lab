"""Motion-first replay for the lesson-21 stopping-tolerance supplement."""

import argparse
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import to_parent
from embodied_learning.experiments.goal_reaching import CASES, DT, METHODS
from embodied_learning.experiments.goal_thresholds import DEFAULT_RESULTS, VARIANTS, load_thresholds
from embodied_learning.landmark_localization import LANDMARKS
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第二十一课补充 · 停车门限越小越好吗？"


def restart_frames(arrays):
    modes = arrays["modes"][:-1]
    return np.flatnonzero((modes[:-1] == 2) & np.isin(modes[1:], [0, 1])) + 1


def dwell_seconds(arrays, frame):
    """Elapsed completed zero-command intervals, not one extra current frame."""
    if int(arrays["modes"][frame]) not in (2, 3):
        return 0.0
    start = frame
    while start > 0 and arrays["modes"][start - 1] == 2:
        start -= 1
    return (frame - start) * DT


class ThresholdDemo(MobileDemo):
    def __init__(self, root, report, records):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk

        self.root, self.report, self.records = root, report, records
        self.case, self.method, self.run = "near", "fused", 0
        self.clock = PlaybackClock(self.steps(), DT, 0.25)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        root.title(WINDOW_TITLE)
        root.geometry("1180x820")
        root.minsize(1080, 780)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        self.map_font = tkfont.Font(root=root, family="Microsoft YaHei", size=10)
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="只缩小停车门限：车更准了，还是更难停下？",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="同目标、同定位方法、配对噪声；只改 2 / 1 / 0.5 cm。真实验收仍为 3 cm，停车需保持 0.4 s，限时 40 s。\n三次独立仿真叠图，不是三辆车互相竞争。Python 运动学记录回放，无惯性、打滑、避障或实时 ROS。",
        ).pack(anchor="w", pady=6)
        choices = ttk.Frame(outer)
        choices.pack(fill="x")
        self.case_box = ttk.Combobox(
            choices,
            state="readonly",
            values=[f"{label} {goal}" for _, label, goal in CASES],
            width=22,
        )
        self.case_box.current(0)
        self.case_box.pack(side="left")
        self.method_box = ttk.Combobox(
            choices, state="readonly", values=[m[1] for m in METHODS], width=19
        )
        self.method_box.current(1)
        self.method_box.pack(side="left", padx=5)
        for box in (self.case_box, self.method_box):
            box.bind("<<ComboboxSelected>>", self.select)
        ttk.Label(choices, text="样本 #").pack(side="left")
        self.sample = tk.IntVar(value=0)
        self.sample_box = ttk.Spinbox(
            choices,
            from_=0,
            to=report["runs"] - 1,
            textvariable=self.sample,
            width=4,
            command=self.select,
        )
        self.sample_box.pack(side="left")
        self.sample_box.bind("<Return>", self.select)
        self.sample_box.bind("<FocusOut>", self.select)
        self.example_button = ttk.Button(choices, text="远目标超时例", command=self.timeout_example)
        self.example_button.pack(side="left", padx=8)
        self.example_button.configure(state="normal" if self.timeout_runs() else "disabled")
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=5)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle, width=6)
        self.play_button.pack(side="left")
        for label, action in (
            ("单步", self.step),
            ("起点", self.restart),
            ("接近目标时", self.approach),
            ("下一次重新调整", self.next_restart),
            ("看最终结果", self.finish),
        ):
            button = ttk.Button(controls, text=label, command=action)
            button.pack(side="left", padx=3)
        self.speed = tk.StringVar(value="0.25")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            state="readonly",
            values=["0.1", "0.25", "0.5", "1"],
            width=5,
        )
        speed_box.pack(side="left", padx=3)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.zoom, self.estimates = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="局部放大", variable=self.zoom, command=self.redraw).pack(
            side="left"
        )
        ttk.Checkbutton(
            controls, text="估计空圈", variable=self.estimates, command=self.redraw
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
            middle, background="#f8fafc", highlightthickness=0, width=640, height=380
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        right = ttk.Frame(middle, padding=(10, 0, 0, 0))
        right.pack(side="right", fill="y")
        self.cards = {}
        for variant, _, label, color in VARIANTS:
            ttk.Label(
                right,
                text=f"● {label} 估计停车门限",
                foreground=color,
                font=("Microsoft YaHei", 12, "bold"),
            ).pack(anchor="w", pady=(6, 1))
            card = ttk.Label(right, text="", width=35, justify="left")
            card.pack(anchor="w", pady=(0, 6))
            self.cards[variant] = card
        self.summary = ttk.Label(outer, text="", justify="left", wraplength=1100)
        self.summary.pack(anchor="w", pady=6)
        self.event = ttk.Label(outer, text="", justify="left", wraplength=1100)
        self.event.pack(anchor="w")
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def current(self, variant):
        return self.records[(variant, self.case, self.run, self.method)]

    def steps(self):
        return max(self.current(v)[1]["steps"] for v, *_ in VARIANTS)

    def select(self, _=None):
        case = CASES[self.case_box.current()][0]
        method = METHODS[self.method_box.current()][0]
        try:
            run = max(0, min(int(self.sample_box.get()), self.report["runs"] - 1))
        except ValueError:
            run = self.run
        self.sample.set(run)
        if (case, method, run) != (self.case, self.method, self.run):
            self.case, self.method, self.run = case, method, run
            self.reset_clock()

    def reset_clock(self):
        self.clock = PlaybackClock(self.steps(), DT, self.clock.speed)
        self.timeline.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.zoom.set(False)
        self.redraw()

    def restart(self):
        self.zoom.set(False)
        self.seek(0)

    def approach(self):
        first = np.flatnonzero(self.current("cm2")[0]["modes"] == 2)
        self.zoom.set(True)
        self.seek(max(0, int(first[0]) - round(1 / DT)) if len(first) else 0)

    def finish(self):
        self.zoom.set(True)
        self.seek(self.clock.steps)

    def next_restart(self):
        frames = sorted(
            {
                int(f)
                for v, *_ in VARIANTS
                for f in restart_frames(self.current(v)[0])
                if f > self.clock.index
            }
        )
        if frames:
            self.zoom.set(True)
            self.seek(frames[0])
        else:
            self.event.configure(
                text="当前样本后面没有新的“停下后又开始运动”事件；可回到起点或切换样本。"
            )

    def timeout_runs(self):
        return [
            run
            for run in range(self.report["runs"])
            if not self.records[("cm05", "far", run, "fused")][1]["controller_arrived"]
        ]

    def timeout_example(self):
        candidates = self.timeout_runs()
        if not candidates:
            return
        self.run = next((r for r in candidates if r > self.run), candidates[0])
        self.case, self.method = "far", "fused"
        self.case_box.current(1)
        self.method_box.current(1)
        self.sample.set(self.run)
        self.reset_clock()
        self.approach()

    def redraw(self):
        if not hasattr(self, "status"):
            return
        self.updating = True
        self.timeline.set(self.clock.index)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        events = []
        for variant, _, label, _ in VARIANTS:
            arrays, trial = self.current(variant)
            frame = min(self.clock.index, trial["steps"])
            goal = np.array(trial["goal"])
            true = np.linalg.norm(arrays["truth"][frame, :2] - goal) * 100
            estimated = np.linalg.norm(arrays["estimated"][frame, :2] - goal) * 100
            mode = int(arrays["modes"][frame])
            phase = {
                0: "继续靠近",
                1: "原地转向",
                2: "零轮速，累计停车时间",
                3: "已宣布到达",
                4: "超时，未完成到达判断",
            }[mode]
            if mode == 3:
                phase += " · 实际通过" if trial["true_success"] else " · 实际停偏"
            restarts = restart_frames(arrays)
            count = int(np.count_nonzero(restarts <= frame))
            left, right = arrays["commands"][frame]
            done = frame == trial["steps"]
            last_line = (
                f"结束于 {trial['duration_s']:.2f} s；之后不再观测"
                if done
                else f"轮速(rad/s) L {left:+.2f} / R {right:+.2f}"
            )
            self.cards[variant].configure(
                text=f"{phase}\n认为还差 {estimated:.2f} cm\n实际还差 {true:.2f} cm\n停车保持 {dwell_seconds(arrays, frame):.2f} / 0.40 s\n已重新调整 {count} 次\n{last_line}"
            )
            if not done and self.clock.index in restarts:
                events.append(f"{label}：估计离开门限，停车计时清零并恢复运动")
            if (
                not done
                and self.method == "fused"
                and self.clock.index in arrays["observation_frames"]
            ):
                events.append(f"{label}：地标更新了估计，真实车没有瞬移")
        rows = [
            r for r in self.report["rows"] if r["case"] == self.case and r["method"] == self.method
        ]
        self.summary.configure(
            text=f"本目标 / 本定位方法共 {self.report['runs']} 个样本（不是只统计当前样本）：\n"
            + "    ｜    ".join(
                f"{r['label']}：通过 {r['true_success_count']}，停偏 {r['false_arrival_count']}，超时 {r['timeout_count']}"
                for r in rows
            )
        )
        self.event.configure(
            text="；".join(events)
            if events
            else "观察重点：估计更接近零，不等于实际更准；超时也可能在绿圈内，只是没有完成自己的停车判断。"
        )
        self.status.configure(
            text=f"{'暂停' if self.clock.paused else '播放中'} · {self.clock.speed:g}× · {self.clock.index * DT:.2f} / {self.clock.steps * DT:.2f} s · 样本 #{self.run} · 各车到达或超时后只保持最后画面"
        )
        self.draw_map()

    def draw_map(self):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        font, local = self.map_font, self.zoom.get()
        line_height = font.metrics("linespace")
        goal = np.array(self.current("cm2")[1]["goal"])
        if local:
            low, high = goal - 0.20, goal + 0.20
        else:
            points = np.vstack(
                [
                    LANDMARKS,
                    goal,
                    *[
                        self.current(v)[0][k][:, :2]
                        for v, *_ in VARIANTS
                        for k in ("truth", "estimated")
                    ],
                ]
            )
            low, high = points.min(0) - 0.3, points.max(0) + 0.3
        scale = min(
            (w - 6 * line_height) / (high[0] - low[0]), (h - 7 * line_height) / (high[1] - low[1])
        )
        if scale <= 0:
            return

        def xy(point):
            return np.array([w / 2, h / 2]) + (np.asarray(point) - (low + high) / 2) * [
                scale,
                -scale,
            ]

        def inside(point):
            return bool(np.all(point >= low) and np.all(point <= high))

        c.create_text(
            8,
            8,
            anchor="nw",
            font=font,
            text="厘米放大：彩色虚线圆＝估计停车门限"
            if local
            else "俯视运动：X 向右，Y 向上；方格 0.5 m",
            tags="map_header",
        )
        c.create_text(
            8,
            8 + line_height,
            anchor="nw",
            font=font,
            text="实心点＝实际车轴；同色空圈＝估计；重合时会遮住",
            tags="map_header",
        )
        spacing = 0.05 if local else 0.5
        label_every = max(1, int(np.ceil((font.measure("-20") + 8) / (spacing * scale))))
        for axis in (0, 1):
            origin = goal[axis] if local else 0
            start = np.ceil((low[axis] - origin) / spacing) * spacing + origin
            for value in np.arange(start, high[axis] + 1e-10, spacing):
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
        gx, gy = xy(goal)
        radius = self.report["fixed_controller"]["true_acceptance_radius_m"] * scale
        c.create_oval(
            gx - radius,
            gy - radius,
            gx + radius,
            gy + radius,
            fill="#dcfce7",
            outline="#15803d",
            width=2,
            tags="target_zone",
        )
        c.create_line(gx - 5, gy, gx + 5, gy, fill="#15803d")
        c.create_line(gx, gy - 5, gx, gy + 5, fill="#15803d")
        c.create_text(
            gx + radius + 8, gy, text="实际验收 3 cm", anchor="w", font=font, fill="#15803d"
        )
        for variant, tolerance, _, color in VARIANTS:
            arrays, trial = self.current(variant)
            frame = min(self.clock.index, trial["steps"])
            if local:
                r = tolerance * scale
                c.create_oval(
                    gx - r,
                    gy - r,
                    gx + r,
                    gy + r,
                    outline=color,
                    dash=(3, 3),
                    tags=f"{variant}_tolerance",
                )
            keys = ("truth", "estimated") if self.estimates.get() else ("truth",)
            for key in keys:
                path = arrays[key][: frame + 1, :2]
                for a, b in pairwise(path):
                    if inside(a) and inside(b):
                        c.create_line(
                            *xy(a),
                            *xy(b),
                            fill=color,
                            width=2 if key == "truth" else 1,
                            dash=() if key == "truth" else (3, 3),
                            tags=f"{variant}_{key}_trail",
                        )
            for key in reversed(keys):
                pose = arrays[key][frame]
                if not inside(pose[:2]):
                    continue
                x, y = xy(pose[:2])
                if key == "truth" and not local:
                    body = [[-0.16, -0.11], [0.16, -0.11], [0.16, 0.11], [-0.16, 0.11]]
                    c.create_polygon(
                        *[v for p in body for v in xy(to_parent(pose, p))],
                        fill="",
                        outline=color,
                        width=2,
                        tags=f"{variant}_body",
                    )
                    for side in (-0.15, 0.15):
                        c.create_line(
                            *xy(to_parent(pose, [-0.075, side])),
                            *xy(to_parent(pose, [0.075, side])),
                            fill=color,
                            width=4,
                        )
                r = 5 if key == "truth" else 8
                c.create_oval(
                    x - r,
                    y - r,
                    x + r,
                    y + r,
                    fill=color if key == "truth" else "",
                    outline=color,
                    width=2,
                    dash=() if key == "truth" else (2, 2),
                    tags=f"{variant}_{key}_centre",
                )
                if key == "truth":
                    c.create_line(
                        x,
                        y,
                        x + 22 * np.cos(pose[2]),
                        y - 22 * np.sin(pose[2]),
                        arrow="last",
                        fill=color,
                        width=2,
                    )
        c.create_text(
            8,
            h - 8,
            anchor="sw",
            text="ΔX / ΔY：cm；放大不改变距离。绿圈大小始终不变。"
            if local
            else "蓝 2 cm · 橙 1 cm · 紫 0.5 cm；实线为各自已走轨迹。",
            font=font,
            tags="map_caption",
        )


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    report, records = load_thresholds(args.results)
    root = tk.Tk()
    ThresholdDemo(root, report, records)
    root.mainloop()


if __name__ == "__main__":
    main()
