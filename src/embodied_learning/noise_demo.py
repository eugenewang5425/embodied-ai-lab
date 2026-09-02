"""Read-only noise replay: true s[k], measured z[k], action u[k], then s[k+1]."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.teaching_demo import CHART_METRICS, Replay, TeachingDemo, chart_samples

WINDOW_TITLE = "倒立摆 · 第六课：测量噪声慢放"
TRUE_COLOR = "#0f766e"
MEASURED_COLOR = "#ea580c"
NOISE_COLORS = ("#64748b", "#2563eb", "#dc2626")


@dataclass(kw_only=True)
class NoiseReplay(Replay):
    measurements: np.ndarray
    noise_scale: float

    @property
    def label(self):
        return f"噪声 {self.noise_scale:g}×"


def load_noise_replays(directory: Path, seconds=10.0, seed: int | None = None):
    if not math.isfinite(seconds) or not 0 < seconds <= 120:
        raise ValueError("seconds must be in (0, 120]")
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != "paired_measurement_noise":
        raise ValueError("Expected an lqr_measurement_noise result directory")
    if not report["seeds"]:
        raise ValueError("No recorded seeds")
    seed = report["seeds"][0] if seed is None else seed
    if seed not in report["seeds"]:
        raise ValueError(f"Seed {seed} is not in this experiment")
    dt, r, gear = float(report["dt_s"]), float(report["R"]), float(report["actuator_gear"])
    reference = np.asarray(report["reference"], dtype=float)
    if (
        not all(math.isfinite(v) and v > 0 for v in (dt, r, gear))
        or reference.shape != (4,)
        or not np.isfinite(reference).all()
    ):
        raise ValueError("Invalid noise metadata")
    replays = []
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for index, condition in enumerate(report["conditions"]):
            key = f"case{index}_seed{seed}"
            states, controls, measured, noise, flags = [
                archive[f"{key}_{name}"].copy()
                for name in ("states", "controls", "measurements", "noise", "end_flags")
            ]
            scale = float(condition["noise_scale"])
            n = len(controls)
            if (
                not math.isfinite(scale)
                or scale < 0
                or n < 1
                or controls.ndim != 1
                or states.shape != (n + 1, 4)
                or measured.shape != (n, 4)
                or noise.shape != (n, 4)
                or flags.shape != (2,)
                or not all(np.isfinite(a).all() for a in (states, controls, measured, noise))
            ):
                raise ValueError("Invalid noise trajectory")
            if not np.allclose(measured, states[:-1] + noise, atol=1e-12, rtol=0):
                raise ValueError("Misaligned measurements: expected z[k] = s[k] + noise[k]")
            count = min(n, max(1, math.ceil(seconds / dt)))
            replays.append(
                NoiseReplay(
                    states[: count + 1],
                    controls[:count],
                    reference,
                    dt,
                    r,
                    gear,
                    f"第六课：{scale:g}× 测量噪声，seed={seed}；无外部推力",
                    n * dt,
                    terminated=bool(flags[0]) and count == n,
                    measurements=measured[:count],
                    noise_scale=scale,
                )
            )
    if not replays or len({p.label for p in replays}) != len(replays):
        raise ValueError("Need distinct noise conditions")
    return replays


@dataclass
class NoiseSeries:
    label: str
    times: np.ndarray
    values: np.ndarray
    color: str
    measured: bool = False


def noise_chart_series(replays, selected, metric, overlay):
    shown = replays if overlay else [selected]
    series = []
    for replay in shown:
        times, values = chart_samples(replay, metric)
        index = next(i for i, p in enumerate(replays) if p is replay)
        color = NOISE_COLORS[index % len(NOISE_COLORS)] if overlay else TRUE_COLOR
        name = "动作" if metric == CHART_METRICS[2] else "真值"
        series.append(NoiseSeries(f"{replay.label} {name}", times, values, color))
    if metric != CHART_METRICS[2]:
        measured = selected.measurements
        values = (
            measured[:, 0] * 100
            if metric == CHART_METRICS[0]
            else np.rad2deg(measured[:, 1] - selected.reference[1])
        )
        series.append(
            NoiseSeries(
                f"{selected.label} 读数",
                np.arange(len(measured)) * selected.dt,
                values,
                MEASURED_COLOR,
                True,
            )
        )
    return series


def noise_status_text(replay, index):
    state = replay.states[index] - replay.reference
    units = np.array([100, 180 / np.pi, 100, 180 / np.pi])
    truth = state * units
    names = ("位置 cm", "倾角 °", "车速 cm/s", "角速 °/s")
    if index == len(replay.controls):
        rows = [f"{name}：{value:+.3f}" for name, value in zip(names, truth, strict=True)]
        return (
            "最后真实状态（未放大）\n\n"
            + "\n".join(rows)
            + "\n\n无下一步读数或动作\n不补造最后一次测量"
        )
    measured = (replay.measurements[index] - replay.reference) * units
    rows = [
        f"{name}：{a:+.3f} → {b:+.3f}" for name, a, b in zip(names, truth, measured, strict=True)
    ]
    return (
        "同一时刻：真实 → 读数\n（数值不放大）\n\n"
        + "\n".join(rows)
        + f"\n\n由读数生成下一步动作\nu = {replay.controls[index]:+.4f}"
        + f"\n电机力 = {replay.controls[index] * replay.gear:+.2f} N\n外部推力 = 0 N"
    )


class NoiseDemo(TeachingDemo):
    def __init__(self, root, replays, speed=0.25):
        initial = next((p for p in replays if p.noise_scale == 1), replays[0])
        super().__init__(root, replays, speed, initial=initial)
        root.title(WINDOW_TITLE)
        root.geometry("1120x750")
        self.scene.configure(height=280)
        self.chart.configure(height=150)
        self.overlay.set(False)
        self.overlay_button.configure(text="叠加三组真实曲线")
        self.metric.set(CHART_METRICS[1])
        self.chart_hint.configure(text="实线：真值；橙色点线：当前读数；粗线：已播放")
        self.help_label.configure(
            text="空格：播放/暂停　→：单步　Home：起点　拖动时间轴会暂停；每步 0.04 s，R 固定为 1。"
        )
        self.note_label.configure(
            text="实线杆是真实姿态，橙色虚线是读数示意（不是另一根杆）；8× 仅放大画面角度，不改变数据。"
        )
        self.refresh()

    def refresh(self):
        super().refresh()
        self.stats.set(noise_status_text(self.replay, self.clock.index))

    def draw_scene(self):
        super().draw_scene()
        c, r, i = self.scene, self.replay, self.clock.index
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 100:
            return
        c.create_text(
            16,
            44,
            text="绿色实杆 = 真实；橙色虚杆 = 传感读数",
            anchor="nw",
            font=("Microsoft YaHei", 10),
            fill="#475569",
        )
        if i == len(r.controls):
            return
        scale = min((w - 80) / 1.2, (h - 95) / 0.7)
        x, theta = r.measurements[i, :2]
        angle = (theta - r.reference[1]) * (8 if self.magnify.get() else 1)
        cx, cy, length = w / 2 + x * scale, h - 81, 0.6 * scale
        c.create_line(
            cx,
            cy,
            cx + length * math.sin(angle),
            cy - length * math.cos(angle),
            fill=MEASURED_COLOR,
            width=3,
            dash=(6, 5),
        )
        c.create_rectangle(
            cx - 30, cy - 6, cx + 30, cy + 17, outline=MEASURED_COLOR, width=2, dash=(5, 4)
        )

    def draw_chart(self):
        c = self.chart
        w, h = c.winfo_width(), c.winfo_height()
        if w < 100 or h < 80:
            return
        c.delete("all")
        series = noise_chart_series(
            self.replays, self.replay, self.metric.get(), self.overlay.get()
        )
        values = np.concatenate([s.values for s in series])
        lo, hi = min(0, float(values.min())), max(0, float(values.max()))
        pad = max((hi - lo) * 0.12, 0.005)
        lo, hi = lo - pad, hi + pad
        left, right, top, bottom = 66, w - 20, 44, h - 32
        duration = max(s.times[-1] for s in series)
        now = self.clock.index * self.replay.dt
        font = ("Microsoft YaHei", 9)

        def point(t, value):
            return left + (right - left) * t / duration, bottom - (bottom - top) * (value - lo) / (
                hi - lo
            )

        for j, s in enumerate(series):
            lx = left + j * 180
            c.create_line(
                lx, 16, lx + 20, 16, fill=s.color, width=2, dash=(3, 3) if s.measured else ()
            )
            c.create_text(lx + 26, 16, text=s.label, anchor="w", font=font, fill=s.color)
        c.create_line(left, top, left, bottom, fill="#64748b")
        c.create_line(left, bottom, right, bottom, fill="#64748b")
        _, zy = point(0, 0)
        c.create_line(left, zy, right, zy, fill="#cbd5e1")
        for y in np.linspace(lo + pad, hi - pad, 3):
            _, py = point(0, y)
            c.create_text(left - 9, py, text=f"{y:.2g}", anchor="e", font=font)
        for t in np.linspace(0, duration, 5):
            px, _ = point(t, 0)
            c.create_text(px, bottom + 16, text=f"{t:g}s", font=font)
        for s in series:
            coordinates = [v for t, y in zip(s.times, s.values, strict=True) for v in point(t, y)]
            dash = (2, 4) if s.measured else ()
            if len(s.times) > 1:
                c.create_line(*coordinates, fill=s.color, width=1, dash=dash)
            count = int(np.searchsorted(s.times, now + 1e-12, side="right"))
            if count >= 2:
                c.create_line(
                    *coordinates[: 2 * count], fill=s.color, width=2 if s.measured else 3, dash=dash
                )
            if count and now <= s.times[-1] + 1e-12:
                px, py = point(s.times[count - 1], s.values[count - 1])
                c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=s.color, outline="")
        px, _ = point(now, 0)
        c.create_line(px, top, px, bottom, fill="#0f172a", dash=(3, 3))
