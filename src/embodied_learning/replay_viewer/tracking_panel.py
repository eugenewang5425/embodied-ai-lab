"""Same-frame tracking and expanded-footprint diagnostics from archived data."""

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class TrackingPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.note = ttk.Label(self, wraplength=720, style="Hint.TLabel")
        self.note.pack(side="bottom", fill="x")
        self.figure = Figure(figsize=(7, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.key, self.frame = None, 0

    def draw(self, record, key, frame):
        self.key, self.frame = key, frame
        self.figure.clear()
        data = record.telemetry.get(key, {})
        if "tracking_cross_m" not in data:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.05,
                0.6,
                "本记录没有跟踪诊断。\n第69课第二轮可查看走偏程度与车身余量。",
                transform=ax.transAxes,
            )
            self.note.configure(text="旧实验保持原记录；不补造缺失的诊断数据。")
            self.canvas.draw_idle()
            return
        f = min(frame, record.lengths[key] - 1)
        t = record.timestamps[: f + 1]
        a, b = self.figure.subplots(1, 2)
        color = record.track_by_key[key].color
        a.plot(t, data["initial_cross_m"][: f + 1] * 100, color=color)
        a.axhline(0, color="#64748b", lw=0.7)
        a.set(title="相对首次规划走偏了多少", xlabel="时间 (s)", ylabel="横向偏差 (cm)")
        b.plot(t, data["expanded_gap_m"][: f + 1] * 100, color=color)
        b.axhline(0, color="#dc7840", ls="--", lw=1, label="外扩足印开始重叠")
        b.set(
            title="外扩6cm的余量（局部放大）",
            xlabel="时间 (s)",
            ylabel="投影分离量 (cm)",
            ylim=(-1, 8),
        )
        b.legend(loc="best", fontsize=7)
        for ax in (a, b):
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
        row = record.method_info(key)["result"]
        self.note.configure(
            text=(
                f"{record.method_info(key)['label']} · t={record.timestamps[f]:.1f}s；横向偏差 {data['initial_cross_m'][f] * 100:+.2f}cm，朝向偏差 {np.rad2deg(data['initial_heading_rad'][f]):+.2f}°（均对首次规划）。\n"
                f"本帧外扩足印分离量 {data['expanded_gap_m'][f] * 100:+.2f}cm；全程实际身体最近间隙 {row['min_clearance_m'] * 100:.2f}cm。\n"
                "负值表示预留空间重叠，不等于碰撞；右图只显示-1～8cm。曲线画到本方法结束帧。"
            )
        )
        self.canvas.draw_idle()
