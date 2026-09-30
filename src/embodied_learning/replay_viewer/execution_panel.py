"""Causal display of transport delay, local intervention and expired control."""

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ExecutionPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.note = ttk.Label(self, wraplength=740, style="Hint.TLabel")
        self.note.pack(side="bottom", fill="x")
        self.figure = Figure(figsize=(7, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.key, self.frame = None, 0

    def draw(self, record, key, frame):
        self.key, self.frame = key, frame
        self.figure.clear()
        d = record.telemetry.get(key, {})
        if "local_override" not in d:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(0.04, 0.6, "本记录没有执行端保护。\n请选择69课第五轮。", transform=ax.transAxes)
            self.note.configure(text="旧记录不补造保护事件。")
            self.canvas.draw_idle()
            return
        f = min(frame, record.lengths[key] - 1)
        ix = slice(max(0, f - 70), f + 1)
        t = record.timestamps[ix]
        color = record.track_by_key[key].color
        a, b = self.figure.subplots(1, 2)
        request = np.where(d["braking_active"][ix], 0, d["requested_command"][ix, 0])
        request = np.where(d["control_fresh"][ix], request, np.nan)
        a.step(t, request, where="post", ls="--", c="#64748b", label="新请求（失联处断开）")
        a.step(
            t,
            d["commands"][ix, 0],
            where="post",
            c=color,
            label="电机速度请求" if "world_velocity" in d else "实际执行速度",
        )
        a.plot(t[-1], d["commands"][f, 0], "o", c=color, ms=4)
        a.set(title="执行端送出什么动作", xlabel="时间 (s)", ylabel="速度 (m/s)")
        b.step(t, d["control_age_ticks"][ix] * 0.1, where="post", c="#64748b", label="消息年龄 (s)")
        b.plot(t[-1], d["control_age_ticks"][f] * 0.1, "o", c="#64748b", ms=4)
        b.axhline(0.2, c="#c53e51", ls="--", lw=1, label="本轮过期门槛0.2s")
        b.scatter(
            t[d["local_override"][ix]],
            np.zeros(int(d["local_override"][ix].sum())),
            c=color,
            s=13,
            label="本地接管",
        )
        b.set(
            title="执行端何时接管？",
            xlabel="时间 (s)",
            ylabel="距上次新结果 (s)",
            ylim=(-0.08, max(0.45, float(d["control_age_ticks"][ix].max()) * 0.1 + 0.15)),
        )
        for ax in (a, b):
            ax.legend(loc="upper left", fontsize=7)
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
        r = record.method_info(key)["result"]
        self.note.configure(
            text=(
                f"{record.method_info(key)['label']} · 本组时间 {record.timestamps[f]:.1f}s；新结果：{'有' if d['control_fresh'][f] else '无'}；本地接管：{'是' if d['local_override'][f] else '否'}。\n"
                f"实际排队{r['perturbation']['delay_steps'] * 0.1:g}s；消息年龄{d['control_age_ticks'][f] * 0.1:g}s；待执行运动{int(d['pending_motion'][f])}条；停车仍受减速度限制。\n"
                + (
                    "本轮三组都保留执行端保护；速度请求不是实际速度，实测运动另见接触页。"
                    if "world_velocity" in d
                    else "只有保护组能在执行端减速并清旧动作；过期后锁住本回合。原参照没有接管；停住≠到达。"
                )
            )
        )
        self.canvas.draw_idle()
