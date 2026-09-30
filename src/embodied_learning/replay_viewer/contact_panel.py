"""Measured body motion and physical contacts, distinct from motor requests."""

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ContactPanel(ttk.Frame):
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
        if "world_velocity" not in d:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.04, 0.6, "旧记录没有接触物理数据。\n请选择69课第六轮。", transform=ax.transAxes
            )
            self.note.configure(text="旧运动学记录不能补造成接触力测量。")
            self.canvas.draw_idle()
            return
        f = min(frame, record.lengths[key] - 1)
        ix = slice(max(0, f - 70), f + 1)
        t = record.timestamps[ix]
        color = record.track_by_key[key].color
        a, b = self.figure.subplots(1, 2)
        a.step(
            t, d["commands"][ix, 0], where="post", ls="--", c="#64748b", label="发给电机的前向请求"
        )
        a.plot(
            t,
            np.linalg.norm(d["world_velocity"][ix, :2], axis=1),
            c=color,
            label="整车实际平移速度",
        )
        a.set(
            title="请求不等于实际移动",
            xlabel="时间 (s)",
            ylabel="速度 (m/s)",
            ylim=(
                -0.03,
                max(0.7, float(np.linalg.norm(d["world_velocity"][ix, :2], axis=1).max()) + 0.05),
            ),
        )
        b.step(t, d["contact_interval"][ix, 1], where="pre", c=color, label="前0.1秒接触力峰值")
        b.axhline(40, c="#c53e51", ls="--", lw=1, label="轻接触上限40N")
        b.set(title="是否只是轻微擦碰？", xlabel="时间 (s)", ylabel="法向接触力 (N)")
        for ax in (a, b):
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
        r = record.method_info(key)["result"]
        cm = d["contact_interval"][f]
        self.note.configure(
            text=(
                f"{record.method_info(key)['label']} · t={t[-1]:.1f}s；额外预留{r['controller_margin_m'] * 100:g}cm；允许轻擦：{'是' if r['contact_allowed'] else '否'}。\n"
                f"前0.1秒：接触力峰值{cm[1]:.2f}N、压入{cm[3] * 1000:.3f}mm；实测角速度{d['world_velocity'][f, 2]:.3f}rad/s。\n"
                "通过须真实到点并停稳；40N／5mm／连续0.5s为仿真轻接触上限。6cm外扩仅诊断，不决定本轮通过。"
            )
        )
        self.canvas.draw_idle()
