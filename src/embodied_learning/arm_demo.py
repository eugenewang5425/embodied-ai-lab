"""Lesson 8: a static geometry probe and slow replay of real torque-driven arm motion."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.planar_arm import LENGTHS, ArmSimulation, forward_kinematics
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第八课 · 双关节机械臂：角度与末端坐标"


@dataclass
class ArmReplay:
    metadata: dict
    states: np.ndarray
    points: np.ndarray
    torques: np.ndarray
    dt: float


def load_replays(directory):
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != "planar_2r_reaching":
        raise ValueError("Expected arm_reaching results")
    if not np.allclose(report["lengths_m"], LENGTHS, atol=1e-12, rtol=0):
        raise ValueError("Saved arm lengths differ from the geometry probe")
    replays = []
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for case in report["cases"]:
            key, dt = case["key"], float(case["dt_s"])
            states, points, torques = [
                archive[f"{key}_{name}"].copy() for name in ("states", "points", "torques_nm")
            ]
            if torques.ndim != 2 or torques.shape[1] != 2 or len(torques) < 1:
                raise ValueError("Invalid torque array")
            n = len(torques)
            if (
                states.shape != (n + 1, 4)
                or points.shape != (n + 1, 3, 2)
                or not math.isfinite(dt)
                or dt <= 0
                or not all(np.isfinite(v).all() for v in (states, points, torques))
            ):
                raise ValueError("Invalid arm trajectory")
            replays.append(ArmReplay(case, states, points, torques, dt))
    if not replays:
        raise ValueError("No saved arm cases")
    return replays


def draw_arm(canvas, points, q, target=None, path=None, reference=None, waypoint=None):
    w, h = canvas.winfo_width(), canvas.winfo_height()
    if min(w, h) < 100:
        return
    canvas.delete("all")
    scale = (min(w, h) - 70) / 1.6
    center = np.array([w / 2, h / 2])

    def xy(p):
        return center + np.asarray(p) * [scale, -scale]

    font = ("Microsoft YaHei", 10)
    for radius, fill in ((sum(LENGTHS), "#ecfdf5"), (abs(LENGTHS[0] - LENGTHS[1]), "#f1f5f9")):
        a, b = xy([-radius, radius]), xy([radius, -radius])
        canvas.create_oval(*a, *b, fill=fill, outline="#94a3b8", dash=(4, 4))
    canvas.create_line(*xy([-0.77, 0]), *xy([0.77, 0]), fill="#94a3b8", arrow="last")
    canvas.create_line(*xy([0, -0.77]), *xy([0, 0.77]), fill="#94a3b8", arrow="last")
    canvas.create_text(*xy([0.77, -0.06]), text="X", font=font)
    canvas.create_text(*xy([0.05, 0.76]), text="Y", font=font)
    for value in (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6):
        px, py = xy([value, 0])
        canvas.create_text(px, py + 15, text=f"{value * 100:g}", font=font, fill="#64748b")
        px, py = xy([0, value])
        canvas.create_text(px - 20, py, text=f"{value * 100:g}", font=font, fill="#64748b")
    canvas.create_text(
        12,
        12,
        text="俯视 XY 平面｜刻度 cm｜绿色环带为几何可达区",
        anchor="nw",
        font=font,
        fill="#475569",
    )
    if reference is not None and len(reference) > 1:
        canvas.create_line(
            *[v for p in reference for v in xy(p)], fill="#475569", dash=(5, 4), width=2
        )
    if waypoint is not None:
        px, py = xy(waypoint)
        canvas.create_oval(px - 8, py - 8, px + 8, py + 8, outline="#ea580c", width=3)
    if path is not None and len(path) > 1:
        canvas.create_line(*[v for p in path for v in xy(p)], fill="#a78bfa", width=2)
    if target is not None:
        tx, ty = xy(target)
        canvas.create_line(tx - 8, ty - 8, tx + 8, ty + 8, fill="#ea580c", width=3)
        canvas.create_line(tx - 8, ty + 8, tx + 8, ty - 8, fill="#ea580c", width=3)
        canvas.create_text(tx + 12, ty - 12, text="目标", anchor="w", fill="#ea580c", font=font)
    for i, color in enumerate(("#2563eb", "#0f766e")):
        canvas.create_line(
            *xy(points[i]), *xy(points[i + 1]), fill=color, width=10, capstyle="round"
        )
        canvas.create_text(
            12 + i * 115,
            36,
            text=f"杆 {i + 1}：{LENGTHS[i] * 100:.0f} cm",
            anchor="nw",
            fill=color,
            font=font,
        )
    for p in points:
        px, py = xy(p)
        canvas.create_oval(px - 5, py - 5, px + 5, py + 5, fill="#0f172a", outline="")
    canvas.create_text(
        12,
        60,
        text=f"q₁={np.rad2deg(q[0]):+.1f}°；q₂={np.rad2deg(q[1]):+.1f}°（相对角）",
        anchor="nw",
        font=font,
    )


class ArmDemo:
    def __init__(self, root, replays, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.replays, self.replay = root, replays, replays[0]
        self.clock = PlaybackClock(len(self.replay.torques), self.replay.dt, speed)
        self.last_tick, self.updating = time.perf_counter(), False
        self.probe = ArmSimulation()
        root.title(WINDOW_TITLE)
        root.geometry("1120x750")
        root.minsize(960, 680)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        ttk.Style(root).theme_use("clam")
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        self.heading_label = ttk.Label(
            outer,
            text="从关节角到末端坐标：先理解几何，再观察真实运动",
            font=("Microsoft YaHei", 17, "bold"),
        )
        self.heading_label.pack(anchor="w", pady=(0, 8))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        geometry, replay_tab = ttk.Frame(self.tabs, padding=10), ttk.Frame(self.tabs, padding=10)
        self.tabs.add(geometry, text="① 几何探针：角度 → 坐标")
        self.tabs.add(replay_tab, text="② 动力学慢放：坐标 → 关节目标 → 电机")
        self.tabs.bind("<<NotebookTabChanged>>", self.pause_for_tab)
        geometry.columnconfigure(0, weight=1)
        geometry.rowconfigure(1, weight=1)
        self.geometry_hint = ttk.Label(
            geometry, text="此页滑块直接设置姿态，只核验几何；不是电机驱动，也不是运动轨迹。"
        )
        self.geometry_hint.grid(row=0, columnspan=2, sticky="w", pady=(0, 8))
        self.geometry_canvas = tk.Canvas(geometry, background="white", highlightthickness=0)
        self.geometry_canvas.grid(row=1, column=0, sticky="nsew")
        side = ttk.Frame(geometry, padding=(15, 10))
        side.grid(row=1, column=1, sticky="ns")
        self.q_vars = [tk.DoubleVar(value=0), tk.DoubleVar(value=90)]
        for i, label in enumerate(("q₁：相对世界 X 轴", "q₂：相对第一根杆")):
            ttk.Label(side, text=label).pack(anchor="w", pady=(10, 3))
            ttk.Scale(
                side, from_=-180, to=180, variable=self.q_vars[i], command=self.update_geometry
            ).pack(fill="x")
        presets = ttk.Frame(side)
        presets.pack(fill="x", pady=15)
        for pair in ((0, 0), (0, 90), (90, 0), (45, -90)):
            ttk.Button(
                presets, text=f"{pair[0]}°, {pair[1]}°", command=lambda p=pair: self.set_angles(p)
            ).pack(fill="x", pady=3)
        self.geometry_stats = tk.StringVar()
        ttk.Label(side, textvariable=self.geometry_stats, width=29, justify="left").pack(
            anchor="w", pady=10
        )
        self.geometry_footer = ttk.Label(
            geometry,
            text="试一试：固定 q₂，只改变 q₁。再固定 q₁，只改变 q₂。观察肘点和末端分别怎样动。",
            foreground="#475569",
        )
        self.geometry_footer.grid(row=2, columnspan=2, sticky="w", pady=8)
        replay_tab.columnconfigure(0, weight=1)
        replay_tab.rowconfigure(1, weight=1)
        bar = ttk.Frame(replay_tab)
        bar.grid(row=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.play_button = ttk.Button(bar, text="播放", command=self.toggle)
        self.play_button.pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="单步 +0.02 s", command=self.step).pack(side="left", padx=6)
        ttk.Button(bar, text="回起点", command=lambda: self.seek(0)).pack(side="left", padx=6)
        self.speed = tk.StringVar(value=f"{speed:g}×")
        speed_box = ttk.Combobox(
            bar,
            textvariable=self.speed,
            values=["0.1×", "0.25×", "0.5×", "1×"],
            state="readonly",
            width=7,
        )
        speed_box.pack(side="left", padx=12)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.case = tk.StringVar(value=self.replay.metadata["label"])
        case_box = ttk.Combobox(
            bar,
            textvariable=self.case,
            values=[r.metadata["label"] for r in replays],
            state="readonly",
            width=24,
        )
        case_box.pack(side="left", padx=6)
        case_box.bind("<<ComboboxSelected>>", self.change_case)
        self.scene = tk.Canvas(replay_tab, background="white", highlightthickness=0, height=300)
        self.scene.grid(row=1, column=0, sticky="nsew")
        self.stats = tk.StringVar()
        ttk.Label(
            replay_tab, textvariable=self.stats, width=30, justify="left", anchor="nw", padding=12
        ).grid(row=1, column=1, sticky="ns")
        self.status = tk.StringVar()
        ttk.Label(replay_tab, textvariable=self.status).grid(
            row=2, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.slider = ttk.Scale(replay_tab, from_=0, to=self.clock.steps, command=self.scrub)
        self.slider.grid(row=3, columnspan=2, sticky="ew", pady=5)
        self.chart = tk.Canvas(replay_tab, background="white", height=120, highlightthickness=0)
        self.chart.grid(row=4, columnspan=2, sticky="ew")
        self.replay_footer = ttk.Label(
            replay_tab,
            text="MuJoCo 真实力矩驱动记录；紫线是已走过的末端路径。无碰撞、噪声和轨迹规划；慢放不改变物理。",
            foreground="#475569",
        )
        self.replay_footer.grid(row=5, columnspan=2, sticky="w", pady=(8, 0))
        self.geometry_canvas.bind("<Configure>", self.update_geometry)
        self.scene.bind("<Configure>", lambda _: self.refresh())
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        root.bind("<space>", lambda _: self.toggle() if self.tabs.index("current") == 1 else None)
        root.bind("<Right>", lambda _: self.step() if self.tabs.index("current") == 1 else None)
        root.bind("<Home>", lambda _: self.seek(0) if self.tabs.index("current") == 1 else None)
        root.bind("<Escape>", lambda _: root.destroy())
        self.update_geometry()
        self.refresh()
        root.after(20, self.tick)

    def set_angles(self, pair):
        for var, value in zip(self.q_vars, pair, strict=True):
            var.set(value)
        self.update_geometry()

    def update_geometry(self, _=None):
        q = np.deg2rad([var.get() for var in self.q_vars])
        self.probe.reset(q)
        points, fk = self.probe.points(), forward_kinematics(q)
        error = np.linalg.norm(points[-1] - fk)
        self.geometry_stats.set(
            f"q₁ = {np.rad2deg(q[0]):+.1f}°\nq₂ = {np.rad2deg(q[1]):+.1f}°\n第二杆世界方向 = {np.rad2deg(sum(q)):+.1f}°\n\n公式末端（cm）\nX={fk[0] * 100:+.2f}，Y={fk[1] * 100:+.2f}\n\nMuJoCo 末端（cm）\nX={points[-1, 0] * 100:+.2f}，Y={points[-1, 1] * 100:+.2f}\n\n两者差异：{error:.2e} m\n（数值核验，非硬件精度）"
        )
        draw_arm(self.geometry_canvas, points, q)

    def pause_for_tab(self, _=None):
        self.clock.seek(self.clock.index)
        self.refresh()

    def toggle(self):
        self.clock.toggle()
        self.last_tick = time.perf_counter()
        self.refresh()

    def step(self):
        self.clock.step()
        self.refresh()

    def seek(self, index):
        self.clock.seek(index)
        self.refresh()

    def scrub(self, value):
        if not self.updating:
            self.seek(round(float(value)))

    def change_speed(self, _=None):
        self.clock.set_speed(float(self.speed.get().removesuffix("×")))
        self.last_tick = time.perf_counter()
        self.refresh()

    def change_case(self, _=None):
        self.replay = next(r for r in self.replays if r.metadata["label"] == self.case.get())
        self.clock = PlaybackClock(len(self.replay.torques), self.replay.dt, self.clock.speed)
        self.slider.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.refresh()

    def tick(self):
        now = time.perf_counter()
        if self.clock.advance(now - self.last_tick):
            self.refresh()
        self.last_tick = now
        self.root.after(20, self.tick)

    def refresh(self):
        r, i = self.replay, self.clock.index
        q, target = r.states[i, :2], np.asarray(r.metadata["target_m"])
        tip = r.points[i, -1]
        finished = i == self.clock.steps
        mode = (
            ("物理失败" if r.metadata["failure_reason"] else "片段结束")
            if finished
            else ("已暂停" if self.clock.paused else "播放中")
        )
        self.status.set(
            f"{mode}　时间 {i * r.dt:.2f} / {self.clock.steps * r.dt:g} s　｜ {self.clock.speed:g}×　｜ 状态 → PD 力矩 → 下一步状态"
        )
        command = (
            "已结束，无下一步力矩"
            if finished
            else f"下一步力矩（N·m）\nτ₁={r.torques[i, 0]:+.3f}\nτ₂={r.torques[i, 1]:+.3f}"
        )
        goal = np.rad2deg(r.metadata["goal_q_rad"])
        self.stats.set(
            f"当前 q₁={np.rad2deg(q[0]):+.1f}°\n当前 q₂={np.rad2deg(q[1]):+.1f}°\n\nIK 目标关节角\nq₁*={goal[0]:+.1f}°\nq₂*={goal[1]:+.1f}°\n\n目标 X,Y（cm）\n{target[0] * 100:+.1f}，{target[1] * 100:+.1f}\n当前末端 X,Y（cm）\n{tip[0] * 100:+.2f}，{tip[1] * 100:+.2f}\n误差 {np.linalg.norm(tip - target) * 1000:.3f} mm\n\n{command}"
        )
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        self.updating = True
        self.slider.set(i)
        self.updating = False
        draw_arm(self.scene, r.points[i], q, target, r.points[: i + 1, -1])
        self.draw_chart()

    def draw_chart(self):
        c, r = self.chart, self.replay
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 80:
            return
        c.delete("all")
        error = np.linalg.norm(r.points[:, -1] - r.metadata["target_m"], axis=1) * 1000
        left, right, top, bottom = 65, w - 18, 25, h - 27
        peak = max(2, float(error.max())) * 1.1
        times = np.arange(len(error)) * r.dt
        coords = [
            (left + (right - left) * t / times[-1], bottom - (bottom - top) * e / peak)
            for t, e in zip(times, error, strict=True)
        ]
        c.create_text(
            left,
            12,
            text="末端到目标的距离（mm）：虚线全程 / 实线已播放",
            anchor="w",
            font=("Microsoft YaHei", 9),
        )
        c.create_line(*[v for p in coords for v in p], fill="#93c5fd", dash=(3, 3))
        if self.clock.index:
            c.create_line(
                *[v for p in coords[: self.clock.index + 1] for v in p], fill="#2563eb", width=2
            )
        x, y = coords[self.clock.index]
        c.create_line(x, top, x, bottom, fill="#64748b", dash=(3, 3))
        c.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#2563eb", outline="")
        c.create_line(left, top, left, bottom, right, bottom, fill="#64748b")
        for value in (0, peak / 2, peak):
            c.create_text(
                left - 8, bottom - (bottom - top) * value / peak, text=f"{value:.0f}", anchor="e"
            )
        for t in (0, times[-1] / 2, times[-1]):
            c.create_text(left + (right - left) * t / times[-1], bottom + 15, text=f"{t:g}s")


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be in (0, 1]")
    try:
        replays = load_replays(args.results)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    root = tk.Tk()
    demo = ArmDemo(root, replays, args.speed)
    root.mainloop()
    del demo


if __name__ == "__main__":
    main()
