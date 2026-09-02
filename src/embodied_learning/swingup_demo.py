"""Slow replay of saved full-rotation trajectories, with force and controller-mode evidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import wrap_angle
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "倒立摆 · 下垂摆起与强扰动恢复"
MODE_NAMES = {
    "kick": "启动：打破静止对称",
    "swingup": "摆起：补充 / 调整能量",
    "balance": "LQR：扶稳并回中心",
}


@dataclass
class SwingupReplay:
    metadata: dict
    states: np.ndarray
    controls: np.ndarray
    forces: np.ndarray
    modes: np.ndarray
    reference: np.ndarray
    dt: float
    gear: float


def load_replays(directory: Path) -> list[SwingupReplay]:
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != "full_rotation_swingup":
        raise ValueError("Expected swingup_comparison results")
    reference = np.asarray(report["reference"], dtype=float)
    dt, gear = float(report["dt_s"]), float(report["actuator_gear"])
    if (
        reference.shape != (4,)
        or not np.isfinite(reference).all()
        or not all(math.isfinite(v) and v > 0 for v in (dt, gear))
    ):
        raise ValueError("Invalid swing-up metadata")
    replays = []
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for case in report["cases"]:
            key = case["key"]
            states, controls, forces, modes = [
                archive[f"{key}_{suffix}"].copy()
                for suffix in ("states", "controls", "applied_force_n", "modes")
            ]
            n = len(controls)
            if (
                controls.ndim != 1
                or n < 1
                or states.shape != (n + 1, 4)
                or forces.shape != (n,)
                or modes.shape != (n,)
                or not all(np.isfinite(a).all() for a in (states, controls, forces))
                or not all(mode in MODE_NAMES for mode in modes)
            ):
                raise ValueError("Invalid swing-up trajectory")
            replays.append(
                SwingupReplay(case, states, controls, forces, modes, reference, dt, gear)
            )
    if not replays:
        raise ValueError("No saved scenarios")
    return replays


class SwingupDemo:
    def __init__(self, root, replays, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        configure_plot_font()
        self.root, self.replays, self.replay = root, replays, replays[0]
        self.clock = PlaybackClock(len(self.replay.controls), self.replay.dt, speed)
        self.last_tick, self.updating = time.perf_counter(), False
        root.title(WINDOW_TITLE)
        root.geometry("1120x820")
        root.minsize(1020, 760)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        ttk.Style(root).theme_use("clam")
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="先摆起来，再扶稳；受强扰动后，重新摆起",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="真实 MuJoCo 记录｜角度不放大｜杆可转整圈｜轨道 ±2.5 m；采样检测 |x|≥2.4 m 时结束",
        ).pack(anchor="w", pady=(4, 8))
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        self.play = ttk.Button(bar, text="播放", command=self.toggle)
        self.play.pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="单步 +0.04 s", command=self.step).pack(side="left", padx=6)
        ttk.Button(bar, text="回起点", command=lambda: self.seek(0)).pack(side="left", padx=6)
        self.speed = tk.StringVar(value=f"{speed:g}×")
        speed_box = ttk.Combobox(
            bar,
            textvariable=self.speed,
            values=["0.1×", "0.25×", "0.5×", "1×"],
            width=7,
            state="readonly",
        )
        speed_box.pack(side="left", padx=10)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.scenario = tk.StringVar(value=self.replay.metadata["label"])
        box = ttk.Combobox(
            bar,
            textvariable=self.scenario,
            values=[r.metadata["label"] for r in replays],
            width=23,
            state="readonly",
        )
        box.pack(side="left", padx=10)
        box.bind("<<ComboboxSelected>>", self.change_scenario)
        panel = ttk.Frame(outer)
        panel.pack(fill="x", pady=8)
        self.scene = tk.Canvas(
            panel,
            height=220,
            background="white",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
        )
        self.scene.pack(side="left", fill="both", expand=True)
        self.scene.bind("<Configure>", lambda _: self.draw_scene())
        self.stats = tk.StringVar()
        ttk.Label(panel, textvariable=self.stats, width=32, justify="left", padding=(12, 4)).pack(
            side="right", fill="y"
        )
        events = ttk.Frame(outer)
        events.pack(fill="x")
        ttk.Label(events, text="跳到关键时刻：").pack(side="left")
        self.event_buttons = []
        for text, key in [
            ("扰动开始", "push_start_s"),
            ("杆到下方", "first_below_horizontal_s"),
            ("恢复稳定", "settled_at_s"),
        ]:
            button = ttk.Button(events, text=text, command=lambda k=key: self.jump(k))
            button.pack(side="left", padx=5)
            self.event_buttons.append((button, key))
        self.status = tk.StringVar()
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(8, 0))
        self.slider = ttk.Scale(outer, from_=0, to=self.clock.steps, command=self.scrub)
        self.slider.pack(fill="x", pady=5)
        fig = Figure(figsize=(10, 2.7), dpi=100, layout="constrained")
        self.axes = fig.subplots(3, 1, sharex=True)
        self.plot = FigureCanvasTkAgg(fig, master=outer)
        self.plot.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="高度 +1=直立，0=水平，−1=正下方。黄色带=外力持续期间；蓝色=电机力，橙色=外力。",
            foreground="#475569",
        ).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            outer,
            text="空格：播放/暂停　→：单步　Home：起点　拖动时间轴会暂停。慢放不改变物理步长；外力与电机力同时作用。",
            foreground="#475569",
        ).pack(anchor="w")
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Right>", lambda _: self.step())
        root.bind("<Home>", lambda _: self.seek(0))
        root.bind("<Escape>", lambda _: root.destroy())
        self.prepare_chart()
        self.refresh()
        root.after(30, self.tick)

    def prepare_chart(self):
        r = self.replay
        ts = np.arange(len(r.states)) * r.dt
        edges = np.arange(len(r.controls) + 1) * r.dt
        for ax in self.axes:
            ax.clear()
            ax.grid(alpha=0.2)
            ax.set_xlim(0, ts[-1])
            if r.metadata["force_n"]:
                ax.axvspan(
                    r.metadata["push_start_s"],
                    r.metadata["push_start_s"] + r.metadata["push_duration_s"],
                    color="#fbbf24",
                    alpha=0.3,
                )
        self.axes[0].plot(ts, np.cos(r.states[:, 1] - r.reference[1]), color="#0f766e")
        self.axes[0].axhline(0, ls="--", color="gray", linewidth=0.7)
        self.axes[0].set(ylabel="相对高度", ylim=(-1.1, 1.1), yticks=[-1, 0, 1])
        self.axes[1].plot(ts, r.states[:, 0], color="#2563eb")
        self.axes[1].set(ylabel="位置（m）")
        self.axes[2].stairs(r.controls * r.gear, edges, color="#2563eb", label="电机力")
        self.axes[2].stairs(r.forces, edges, color="#ea580c", label="外力")
        self.axes[2].set(ylabel="力（N）", xlabel="仿真时间（s）")
        self.axes[2].legend(loc="upper right", ncols=2, fontsize=8)
        self.cursors = [ax.axvline(0, color="#0f172a", ls="--", linewidth=1) for ax in self.axes]
        for button, key in self.event_buttons:
            enabled = r.metadata.get(key) is not None and (
                key != "push_start_s" or r.metadata["force_n"] != 0
            )
            button.configure(state="normal" if enabled else "disabled")

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

    def jump(self, key):
        value = self.replay.metadata.get(key)
        if value is not None:
            self.seek(round(value / self.replay.dt))

    def change_speed(self, _=None):
        self.clock.set_speed(float(self.speed.get().removesuffix("×")))
        self.last_tick = time.perf_counter()
        self.refresh()

    def change_scenario(self, _=None):
        self.replay = next(r for r in self.replays if r.metadata["label"] == self.scenario.get())
        self.clock = PlaybackClock(len(self.replay.controls), self.replay.dt, self.clock.speed)
        self.slider.configure(to=self.clock.steps)
        self.prepare_chart()
        self.last_tick = time.perf_counter()
        self.refresh()

    def tick(self):
        now = time.perf_counter()
        if self.clock.advance(now - self.last_tick):
            self.refresh()
        self.last_tick = now
        self.root.after(30, self.tick)

    def refresh(self):
        r, i = self.replay, self.clock.index
        x, theta, v, omega = r.states[i]
        angle = float(wrap_angle(theta - r.reference[1]))
        finished = i == self.clock.steps
        state = (
            ("行程 / 安全边界失败" if r.metadata["terminated"] else "记录结束")
            if finished
            else ("已暂停" if self.clock.paused else "播放中")
        )
        self.status.set(
            f"{state}　仿真 {i * r.dt:.2f} / {self.clock.steps * r.dt:g} s　｜ {self.clock.speed:g}×　｜ 第 {i} 步"
        )
        mode = "无下一步控制" if finished else MODE_NAMES[str(r.modes[i])]
        forces = (
            "记录结束，不再施力"
            if finished
            else f"下一步电机力：{r.controls[i] * r.gear:+.1f} N\n下一步外部力：{r.forces[i]:+.1f} N"
        )
        self.stats.set(
            f"{mode}\n\n倾角：{np.rad2deg(angle):+.1f}°\n杆在水平线{'下方' if np.cos(angle) < 0 else '上方'}\n小车：{x:+.3f} m\n车速：{v:+.2f} m/s\n角速度：{omega:+.2f} rad/s\n\n{forces}"
        )
        self.play.configure(text="播放" if self.clock.paused else "暂停")
        self.updating = True
        self.slider.set(i)
        self.updating = False
        for line in self.cursors:
            line.set_xdata([i * r.dt, i * r.dt])
        self.plot.draw_idle()
        self.draw_scene()

    def draw_scene(self):
        c, r, i = self.scene, self.replay, self.clock.index
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50 or h < 50:
            return
        c.delete("all")
        x, theta = r.states[i, :2]
        angle = theta - r.reference[1]
        scale = min((w - 65) / 6.5, (h - 70) / 1.5)
        cx, cy, length = w / 2 + x * scale, h / 2, 0.6 * scale
        ex, ey = cx + length * np.sin(angle), cy - length * np.cos(angle)
        font = ("Microsoft YaHei", 9)
        c.create_rectangle(0, cy, w, h, fill="#fffbeb", outline="")
        c.create_line(20, cy, w - 20, cy, fill="#cbd5e1", dash=(4, 4))
        c.create_text(
            12,
            15,
            text="真实角度 1×｜下方浅黄区域 = 低于铰点",
            anchor="w",
            font=font,
            fill="#475569",
        )
        for value in (-2.5, -2, -1, 0, 1, 2, 2.5):
            px = w / 2 + value * scale
            c.create_line(px, cy + 8, px, cy + 14, fill="#64748b")
            c.create_text(px, h - 14, text=f"{value:g} m", font=font, fill="#64748b")
        for value in (-2.4, 2.4):
            px = w / 2 + value * scale
            c.create_line(px, 30, px, h - 30, fill="#ef4444", dash=(4, 4))
        c.create_rectangle(cx - 14, cy - 7, cx + 14, cy + 7, fill="#2563eb", outline="")
        c.create_line(cx, cy, ex, ey, fill="#0f766e", width=8, capstyle="round")
        c.create_oval(cx - 4, cy - 4, cx + 4, cy + 4, fill="#0f172a", outline="")
        if i < len(r.controls):
            for force, color, y, name in [
                (r.forces[i], "#ea580c", cy - 40, "外力"),
                (r.controls[i] * r.gear, "#2563eb", cy + 40, "电机"),
            ]:
                if abs(force) > 0.1:
                    end = cx + np.clip(force / 600, -1, 1) * 65
                    c.create_line(cx, y, end, y, fill=color, width=3, arrow="last")
                    c.create_text(cx, y - 12, text=f"{name} {force:+.0f} N", font=font, fill=color)


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
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    root = tk.Tk()
    demo = SwingupDemo(root, replays, args.speed)
    root.mainloop()
    del demo


if __name__ == "__main__":
    main()
