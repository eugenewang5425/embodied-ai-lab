"""Slow, pausable teaching replay of real MuJoCo trajectories (no new GUI dependency)."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.experiments.pd_comparison import run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_TITLE = "倒立摆 · 慢速教学演示（多 R 对照）"
CURVE_COLORS = ("#2563eb", "#ea580c", "#15803d", "#9333ea")
CHART_METRICS = ("小车位置（cm）", "真实倾角（°）", "控制输入 u")


@dataclass
class Replay:
    states: np.ndarray
    controls: np.ndarray
    reference: np.ndarray
    dt: float
    r: float
    gear: float
    source: str
    original_seconds: float
    external_forces_n: np.ndarray | None = None
    terminated: bool = False

    @property
    def label(self) -> str:
        return f"R = {self.r:g}"


def chart_samples(replay: Replay, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Preserve state timestamps; show controls as held over [k*dt, (k+1)*dt)."""
    if metric == CHART_METRICS[0]:
        values = replay.states[:, 0] * 100
    elif metric == CHART_METRICS[1]:
        values = np.rad2deg(replay.states[:, 1] - replay.reference[1])
    elif metric == CHART_METRICS[2]:
        edges = np.arange(len(replay.controls) + 1) * replay.dt
        return np.repeat(edges, 2)[1:-1], np.repeat(replay.controls, 2)
    else:
        raise ValueError(f"Unknown chart metric: {metric}")
    return np.arange(len(values)) * replay.dt, values


@dataclass
class PlaybackClock:
    steps: int
    dt: float
    speed: float = 0.25
    index: int = 0
    paused: bool = True
    remainder: float = 0.0

    def __post_init__(self) -> None:
        if self.steps < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Need positive step count and dt")
        self.set_speed(self.speed)

    def set_speed(self, speed: float) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be finite and positive")
        self.speed = speed
        self.remainder = 0.0

    def seek(self, index: int) -> None:
        self.index = max(0, min(self.steps, int(index)))
        self.paused = True
        self.remainder = 0.0

    def step(self) -> None:
        self.seek(self.index + 1)

    def toggle(self) -> None:
        if self.index == self.steps:
            self.seek(0)
        self.paused = not self.paused
        self.remainder = 0.0

    def advance(self, elapsed: float) -> bool:
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("elapsed must be finite and nonnegative")
        if self.paused:
            return False
        self.remainder += elapsed * self.speed
        count = int((self.remainder + 1e-12) / self.dt)
        self.remainder -= count * self.dt
        previous = self.index
        self.index = min(self.steps, self.index + count)
        if self.index == self.steps:
            self.paused = True
            self.remainder = 0.0
        return self.index != previous


def load_recorded_replay(directory: Path, seconds: float) -> Replay:
    """Read the displaced LQR episode; never writes to the user's experiment."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    dt = float(report["dt_s"])
    r = float(report["design"]["R"][0][0])
    gear = float(report["design"]["actuator_gear"])
    reference = np.asarray(report["design"]["reference"], dtype=float)
    if not all(math.isfinite(v) and v > 0 for v in (dt, r, gear)):
        raise ValueError("Invalid recorded dt, R or actuator gear")
    if reference.shape != (4,) or not np.isfinite(reference).all():
        raise ValueError("Invalid recorded reference")
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        states = archive["displaced_lqr_seed0_states"].copy()
        controls = archive["displaced_lqr_seed0_controls"].copy()
    if (
        controls.ndim != 1
        or len(controls) < 1
        or states.shape != (len(controls) + 1, 4)
        or not np.isfinite(states).all()
        or not np.isfinite(controls).all()
    ):
        raise ValueError("Invalid recorded trajectory")
    count = min(len(controls), max(1, math.ceil(seconds / dt)))
    return Replay(
        states[: count + 1],
        controls[:count],
        reference,
        dt,
        r,
        gear,
        f"已有实验：{directory.name}",
        len(controls) * dt,
    )


def build_replays(seconds: float = 8.0, results: Path | None = None) -> list[Replay]:
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("seconds must be in (0, 120]")
    default_record = PROJECT_ROOT / "results" / "lqr_r1_my_run"
    selected = (
        results if results is not None else (default_record if default_record.exists() else None)
    )
    recorded = load_recorded_replay(selected, seconds) if selected is not None else None
    replays = []
    for r in (0.1, 1.0, 10.0):
        if recorded is not None and math.isclose(recorded.r, r):
            replays.append(recorded)
            continue
        design = design_lqr(control_weight=r)
        # Use the recorded exact initial state for the other policies as well.
        initial = (
            recorded.states[0]
            if recorded is not None
            else design.controller.reference + np.array([0.2, 0.05, 0, 0])
        )
        if recorded is not None:
            np.testing.assert_allclose(recorded.reference, design.controller.reference, atol=1e-9)
            if not math.isclose(recorded.dt, design.dt) or not math.isclose(
                recorded.gear, design.actuator_gear
            ):
                raise ValueError("Recorded model timing/transmission differs from current model")
        count = max(1, math.ceil(seconds / design.dt))
        trace = run_episode("lqr", 0, count, design.controller, initial_state=initial)
        replays.append(
            Replay(
                np.vstack([trace.initial_observation, *trace.observations]),
                np.asarray(trace.actions),
                design.controller.reference,
                trace.dt,
                r,
                design.actuator_gear,
                "当前 MuJoCo 模型计算的轨迹",
                trace.length * trace.dt,
            )
        )
    if recorded is not None and not any(math.isclose(recorded.r, p.r) for p in replays):
        replays.append(recorded)
    return replays


def load_push_replays(
    directory: Path, seconds: float = 8.0, seed: int | None = None
) -> list[Replay]:
    """Read every R at one paired seed; never rerun or mutate stored experiments."""
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("seconds must be in (0, 120]")
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != "paired_random_cart_push":
        raise ValueError("Expected an lqr_disturbance result directory")
    seed = report["seeds"][0] if seed is None else seed
    if seed not in report["seeds"]:
        raise ValueError(f"Seed {seed} is not in this experiment")
    dt, gear = float(report["dt_s"]), float(report["actuator_gear"])
    reference = np.asarray(report["reference"], dtype=float)
    if (
        not all(math.isfinite(v) and v > 0 for v in (dt, gear))
        or reference.shape != (4,)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Invalid push metadata")
    replays = []
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for condition in report["conditions"]:
            r = float(condition["R"])
            key = f"r{r:g}_seed{seed}"
            states = archive[f"{key}_states"].copy()
            controls = archive[f"{key}_controls"].copy()
            forces = archive[f"{key}_applied_force_n"].copy()
            if (
                not math.isfinite(r)
                or r <= 0
                or controls.ndim != 1
                or len(controls) < 1
                or states.shape != (len(controls) + 1, 4)
                or forces.shape != controls.shape
                or not all(np.isfinite(a).all() for a in (states, controls, forces))
            ):
                raise ValueError("Invalid push trajectory")
            count = min(len(controls), max(1, math.ceil(seconds / dt)))
            affected = np.flatnonzero(archive[f"seed{seed}_scheduled_force_n"])
            start, end = int(affected[0]), int(affected[-1] + 1)
            force = float(archive[f"seed{seed}_scheduled_force_n"][start])
            replays.append(
                Replay(
                    states[: count + 1],
                    controls[:count],
                    reference,
                    dt,
                    r,
                    gear,
                    f"随机推力 seed={seed}：{force:+.1f} N，{start * dt:.2f}–{end * dt:.2f} s",
                    len(controls) * dt,
                    forces[:count],
                    bool(archive[f"{key}_end_flags"][0]) and count == len(controls),
                )
            )
    if not replays:
        raise ValueError("No R conditions in this experiment")
    return replays


class TeachingDemo:
    def __init__(self, root, replays: list[Replay], speed: float = 0.25, initial=None):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.replays = root, replays
        self.replay = (
            initial if initial is not None else next((p for p in replays if p.r == 1), replays[0])
        )
        self.clock = PlaybackClock(len(self.replay.controls), self.replay.dt, speed)
        self.last_tick, self.updating = time.perf_counter(), False
        root.title(WINDOW_TITLE)
        root.geometry("1120x810")
        root.minsize(930, 720)
        root.option_add("*Font", "{Microsoft YaHei} 11")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TButton", padding=(10, 6), font=("Microsoft YaHei", 11))
        style.configure("TLabel", font=("Microsoft YaHei", 11))
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=3)
        outer.rowconfigure(6, weight=1)
        ttk.Label(
            outer, text="倒立摆：先看清每一步，再理解控制", font=("Microsoft YaHei", 17, "bold")
        ).grid(row=0, sticky="w")
        self.source = tk.StringVar()
        ttk.Label(outer, textvariable=self.source, foreground="#475569").grid(
            row=1, sticky="w", pady=(6, 10)
        )
        toolbar = ttk.Frame(outer)
        toolbar.grid(row=2, sticky="ew", pady=(0, 10))
        self.play_button = ttk.Button(toolbar, text="播放", command=self.toggle)
        self.play_button.pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="单步 +0.04 s", command=self.step).pack(side="left", padx=6)
        ttk.Button(toolbar, text="重播", command=self.restart).pack(side="left", padx=6)
        ttk.Label(toolbar, text="速度").pack(side="left", padx=(20, 4))
        self.speed = tk.StringVar(value=f"{speed:g}×")
        speeds = sorted({0.1, 0.25, 0.5, 1.0, speed})
        speed_box = ttk.Combobox(
            toolbar,
            textvariable=self.speed,
            values=[f"{s:g}×" for s in speeds],
            state="readonly",
            width=7,
        )
        speed_box.pack(side="left")
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        ttk.Label(toolbar, text="对照方案").pack(side="left", padx=(20, 4))
        self.policy = tk.StringVar(value=self.replay.label)
        policy_box = ttk.Combobox(
            toolbar,
            textvariable=self.policy,
            values=[p.label for p in replays],
            state="readonly",
            width=9,
        )
        policy_box.pack(side="left")
        policy_box.bind("<<ComboboxSelected>>", self.change_policy)
        self.magnify = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar, text="摆角视觉放大 8×", variable=self.magnify, command=self.refresh
        ).pack(side="left", padx=15)
        body = ttk.Frame(outer)
        body.grid(row=3, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.scene = tk.Canvas(
            body,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            height=330,
        )
        self.scene.grid(row=0, column=0, sticky="nsew")
        self.scene.bind("<Configure>", lambda _: self.draw_scene())
        self.stats = tk.StringVar()
        ttk.Label(
            body, textvariable=self.stats, justify="left", width=27, anchor="nw", padding=(20, 16)
        ).grid(row=0, column=1, sticky="ns")
        timeline = ttk.Frame(outer)
        timeline.grid(row=4, sticky="ew", pady=10)
        timeline.columnconfigure(0, weight=1)
        self.status = tk.StringVar()
        ttk.Label(timeline, textvariable=self.status).grid(row=0, sticky="w")
        self.slider = ttk.Scale(timeline, from_=0, to=self.clock.steps, command=self.scrub)
        self.slider.grid(row=1, sticky="ew", pady=5)
        chart_bar = ttk.Frame(outer)
        chart_bar.grid(row=5, sticky="ew", pady=(0, 6))
        self.overlay = tk.BooleanVar(value=True)
        self.overlay_button = ttk.Checkbutton(
            chart_bar, text="叠加所有 R 曲线", variable=self.overlay, command=self.draw_chart
        )
        self.overlay_button.pack(side="left")
        self.metric = tk.StringVar(value=CHART_METRICS[0])
        metric_box = ttk.Combobox(
            chart_bar, textvariable=self.metric, values=CHART_METRICS, state="readonly", width=18
        )
        metric_box.pack(side="left", padx=16)
        metric_box.bind("<<ComboboxSelected>>", lambda _: self.draw_chart())
        self.chart_hint = ttk.Label(
            chart_bar, text="同色虚线：完整轨迹；实线：已播放部分", foreground="#475569"
        )
        self.chart_hint.pack(side="left")
        self.chart = tk.Canvas(
            outer,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            height=170,
        )
        self.chart.grid(row=6, sticky="nsew")
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        self.help_label = ttk.Label(
            outer,
            text="空格：播放/暂停　→：单步　Home：回起点　拖动时间轴会暂停；上方小车和右侧数字只对应所选 R。",
            foreground="#475569",
        )
        self.help_label.grid(row=7, sticky="w", pady=(8, 0))
        self.note_label = ttk.Label(
            outer,
            text="真实 MuJoCo 数据的正视示意回放；慢放不改变物理步长。角度放大只作用于杆的画面，右侧数值不放大。",
            foreground="#475569",
        )
        self.note_label.grid(row=8, sticky="w", pady=(4, 0))
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Right>", lambda _: self.step())
        root.bind("<Home>", lambda _: self.home())
        root.bind("<Escape>", lambda _: root.destroy())
        self.refresh()
        root.after(20, self.tick)

    def toggle(self):
        self.clock.toggle()
        self.last_tick = time.perf_counter()
        self.refresh()

    def step(self):
        self.clock.step()
        self.refresh()

    def home(self):
        self.clock.seek(0)
        self.refresh()

    def restart(self):
        self.clock.seek(0)
        self.clock.toggle()
        self.last_tick = time.perf_counter()
        self.refresh()

    def change_speed(self, _=None):
        self.clock.set_speed(float(self.speed.get().removesuffix("×")))
        self.last_tick = time.perf_counter()
        self.refresh()

    def change_policy(self, _=None):
        self.replay = next(p for p in self.replays if p.label == self.policy.get())
        self.clock = PlaybackClock(len(self.replay.controls), self.replay.dt, self.clock.speed)
        self.slider.configure(to=self.clock.steps)
        self.last_tick = time.perf_counter()
        self.refresh()

    def scrub(self, value):
        if not self.updating:
            self.clock.seek(round(float(value)))
            self.refresh()

    def tick(self):
        now = time.perf_counter()
        changed = self.clock.advance(now - self.last_tick)
        self.last_tick = now
        if changed:
            self.refresh()
        self.root.after(20, self.tick)

    def refresh(self):
        index, replay = self.clock.index, self.replay
        x, theta, velocity, omega = replay.states[index]
        angle = math.degrees(theta - replay.reference[1])
        state = (
            ("物理失败，回放停止" if replay.terminated else "片段结束")
            if index == self.clock.steps
            else ("已暂停" if self.clock.paused else "播放中")
        )
        self.source.set(
            f"{replay.source}　｜ R={replay.r:g}　｜ 原记录 {replay.original_seconds:g} s，教学播放 {self.clock.steps * replay.dt:g} s"
        )
        self.status.set(
            f"{state}　仿真时间 {index * replay.dt:.2f} / {self.clock.steps * replay.dt:g} s　｜ 第 {index} 步　｜ {self.clock.speed:g}×：1 秒仿真用 {1 / self.clock.speed:g} 秒观看"
        )
        command = (
            f"下一步控制输入 u\n{replay.controls[index]:+.4f}\n\n对应执行器水平力\n{replay.controls[index] * replay.gear:+.2f} N"
            if index < self.clock.steps
            else "片段已结束\n没有下一步控制输入"
        )
        self.stats.set(
            f"真实数值（不放大）\n\n小车位置：{x * 100:+.2f} cm\n相对竖直：{angle:+.3f}°\n车速：{velocity * 100:+.2f} cm/s\n角速度：{math.degrees(omega):+.2f}°/s\n\n{command}"
            + (
                f"\n\n下一步外部推力：{replay.external_forces_n[index]:+.1f} N"
                if replay.external_forces_n is not None and index < self.clock.steps
                else ""
            )
        )
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        self.updating = True
        self.slider.set(index)
        self.updating = False
        self.draw_scene()
        self.draw_chart()

    def draw_scene(self):
        c = self.scene
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or height < 100:
            return
        c.delete("all")
        scale = min((width - 80) / 1.2, (height - 95) / 0.7)
        center, rail = width / 2, height - 62
        x, theta, *_ = self.replay.states[self.clock.index]
        multiplier = 8 if self.magnify.get() else 1
        angle = (theta - self.replay.reference[1]) * multiplier
        cart_x, hinge_y = center + x * scale, rail - 19
        length = 0.6 * scale
        end_x, end_y = cart_x + length * math.sin(angle), hinge_y - length * math.cos(angle)
        font = ("Microsoft YaHei", 10)
        c.create_text(
            16,
            16,
            text=f"正视示意　摆角画面 ×{multiplier}（数字是真实值）",
            anchor="nw",
            font=font,
            fill="#475569",
        )
        c.create_line(25, rail, width - 25, rail, width=4, fill="#64748b")
        c.create_line(center, 42, center, rail + 8, dash=(5, 5), fill="#94a3b8")
        for value in (-0.4, -0.2, 0, 0.2, 0.4):
            px = center + value * scale
            c.create_line(px, rail - 4, px, rail + 7, fill="#475569")
            c.create_text(px, rail + 23, text=f"{value * 100:+.0f}" if value else "0", font=font)
        c.create_text(
            width - 16,
            rail + 43,
            text="位置（cm）｜虚线为中心",
            anchor="e",
            font=font,
            fill="#475569",
        )
        c.create_rectangle(
            cart_x - 29, rail - 24, cart_x + 29, rail - 3, fill="#2563eb", outline=""
        )
        for wheel_x in (cart_x - 17, cart_x + 17):
            c.create_oval(wheel_x - 5, rail - 8, wheel_x + 5, rail + 2, fill="#334155", outline="")
        c.create_line(cart_x, hinge_y, end_x, end_y, width=11, fill="#0f766e", capstyle="round")
        c.create_oval(cart_x - 5, hinge_y - 5, cart_x + 5, hinge_y + 5, fill="#0f172a", outline="")

    def draw_chart(self):
        c = self.chart
        width, height = c.winfo_width(), c.winfo_height()
        if width < 100 or height < 80:
            return
        c.delete("all")
        left, right, top, bottom = 66, width - 20, 38, height - 32
        shown = self.replays if self.overlay.get() else [self.replay]
        series = [(p, *chart_samples(p, self.metric.get())) for p in shown]
        all_values = np.concatenate([values for _, _, values in series])
        lo, hi = min(0, float(all_values.min())), max(0, float(all_values.max()))
        padding = max((hi - lo) * 0.12, 0.05)
        lo, hi = lo - padding, hi + padding
        duration = max(len(p.controls) * p.dt for p in shown)
        now = self.clock.index * self.replay.dt
        font = ("Microsoft YaHei", 10)
        c.create_text(left, 14, text=self.metric.get(), anchor="w", font=font)

        def point(t, value):
            return (
                left + (right - left) * t / duration,
                bottom - (bottom - top) * (value - lo) / (hi - lo),
            )

        if self.replay.external_forces_n is not None:
            affected = np.flatnonzero(self.replay.external_forces_n)
            if len(affected):
                start_x, _ = point(affected[0] * self.replay.dt, 0)
                end_x, _ = point((affected[-1] + 1) * self.replay.dt, 0)
                c.create_rectangle(start_x, top, end_x, bottom, fill="#fef3c7", outline="")
                c.create_text(
                    (start_x + end_x) / 2, top + 8, text="推力", font=font, fill="#92400e"
                )

        zero_y = bottom - (bottom - top) * (0 - lo) / (hi - lo)
        c.create_line(left, zero_y, right, zero_y, fill="#cbd5e1")
        for tick in np.linspace(lo + padding, hi - padding, 3):
            py = bottom - (bottom - top) * (tick - lo) / (hi - lo)
            c.create_text(left - 12, py, text=f"{tick:.2g}", anchor="e", font=font)
        c.create_line(left, top, left, bottom, fill="#64748b")
        c.create_line(left, bottom, right, bottom, fill="#64748b")
        for t in np.linspace(0, duration, 5):
            px = left + (right - left) * t / duration
            c.create_text(px, bottom + 16, text=f"{t:g}s", font=font)
        for legend_index, (replay, times, values) in enumerate(series):
            color_index = next(i for i, p in enumerate(self.replays) if p is replay)
            color = CURVE_COLORS[color_index % len(CURVE_COLORS)]
            selected = replay is self.replay
            legend_x = left + 190 + legend_index * 150
            c.create_line(legend_x, 14, legend_x + 22, 14, fill=color, width=3)
            c.create_text(
                legend_x + 28,
                14,
                text=replay.label + ("（当前）" if selected else ""),
                anchor="w",
                font=font,
                fill=color,
            )
            coords = [
                coordinate
                for t, value in zip(times, values, strict=True)
                for coordinate in point(t, value)
            ]
            c.create_line(*coords, fill=color, width=1, dash=(3, 4))
            count = int(np.searchsorted(times, now + 1e-12, side="right"))
            if count >= 2:
                c.create_line(*coords[: 2 * count], fill=color, width=3 if selected else 2)
            # No extrapolation or frozen marker after another trajectory has ended.
            if count and now <= times[-1] + 1e-12:
                px, py = point(times[count - 1], values[count - 1])
                c.create_oval(px - 4, py - 4, px + 4, py + 4, fill=color, outline="")
        px, _ = point(now, 0)
        c.create_line(px, top, px, bottom, fill="#334155", dash=(3, 3))


def main() -> None:
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--results", type=Path, help="Existing lqr_comparison result directory (read-only)"
    )
    source.add_argument(
        "--push-results", type=Path, help="Existing lqr_disturbance results (read-only)"
    )
    source.add_argument(
        "--noise-results", type=Path, help="Existing measurement-noise results (read-only)"
    )
    parser.add_argument(
        "--seed", type=int, help="Recorded push/noise seed (default: first saved seed)"
    )
    parser.add_argument(
        "--seconds", type=float, default=8.0, help="Simulation seconds to replay, not wall time"
    )
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be in (0, 1]")
    if args.seed is not None and args.push_results is None and args.noise_results is None:
        parser.error("--seed requires --push-results or --noise-results")
    demo_class = TeachingDemo
    try:
        if args.noise_results is not None:
            from embodied_learning.noise_demo import NoiseDemo, load_noise_replays

            replays = load_noise_replays(args.noise_results, args.seconds, args.seed)
            demo_class = NoiseDemo
        elif args.push_results is not None:
            replays = load_push_replays(args.push_results, args.seconds, args.seed)
        else:
            replays = build_replays(args.seconds, args.results)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    root = tk.Tk()
    demo = demo_class(root, replays, args.speed)
    root.mainloop()
    del demo


if __name__ == "__main__":
    main()
