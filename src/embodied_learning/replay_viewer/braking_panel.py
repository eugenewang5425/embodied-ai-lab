"""Causal archived command/velocity plots, including actual terminal braking."""

from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PHASES = {
    "running": "正常导航",
    "goal_braking": "接近目标后刹停",
    "abort_braking": "任务失败后继续实际刹停",
    "contact_abort": "接触后中止任务",
}


class BrakingPanel(ttk.Frame):
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
        if "incoming_velocity" not in data:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.04,
                0.6,
                "本批次没有独立制动过程诊断。\n请选择第69课第三轮记录。",
                transform=ax.transAxes,
            )
            self.note.configure(text="旧记录保持原有信息，不推算缺失的制动尾段。")
            self.canvas.draw_idle()
            return
        f = min(frame, record.lengths[key] - 1)
        start = max(0, f - 70)
        ix = slice(start, f + 1)
        t, command = record.timestamps[ix], data["commands"][ix]
        color = record.track_by_key[key].color
        a, b = self.figure.subplots(1, 2)
        a.step(
            t,
            command[:, 0],
            where="post",
            color=color,
            label="前向速度请求" if "world_velocity" in data else "执行速度",
        )
        a.plot(t, data["speed_cap"][ix], color="#64748b", ls="--", label="名义速度上界")
        b.step(
            t,
            command[:, 1],
            where="post",
            color=color,
            label="角速度请求" if "world_velocity" in data else "执行角速度",
        )
        brake = data["braking_active"][ix].astype(bool)
        for ax, values in ((a, command[:, 0]), (b, command[:, 1])):
            ax.scatter(t[brake], values[brake], marker="x", s=17, color="#dc7840", label="请求制动")
            ax.grid(alpha=0.15)
            ax.legend(fontsize=7, loc="best")
            ax.tick_params(labelsize=8)
        a.set(title="向前速度如何变化", ylabel="速度 (m/s)", xlabel="时间 (s)")
        b.set(title="转向是否同时减慢", ylabel="角速度 (rad/s)", xlabel="时间 (s)")
        for ax in (a, b):
            ax.title.set_fontsize(10)
        if "world_velocity" in data:
            a.plot(t, data["incoming_velocity"][ix, 0], color="#152b43", lw=1, label="实际前向速度")
            b.plot(t, data["world_velocity"][ix, 2], color="#152b43", lw=1, label="实际角速度")
            for ax in (a, b):
                ax.legend(fontsize=7, loc="best")
        current = data["incoming_velocity"][f]
        next_command = data["commands"][f]
        gap = data["expanded_gap_m"][f] * 100
        self.note.configure(
            text=f"{record.method_info(key)['label']} · {PHASES[str(data['phase'][f])]} · t={record.timestamps[f]:.1f}s\n"
            f"进入本帧 v={current[0]:.3f}m/s、ω={current[1]:.3f}rad/s → 请求 {next_command[0]:.3f}、{next_command[1]:.3f}；外扩分离量 {gap:+.2f}cm。\n"
            "仅画到当前/本组末帧，最多显示最近7秒；最后零命令是末帧标记，是否停稳看进入本帧速度。"
        )
        self.canvas.draw_idle()
