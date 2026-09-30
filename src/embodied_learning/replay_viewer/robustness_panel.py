"""Recorded issue/actuation timing and explicit injected pose errors."""

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class RobustnessPanel(ttk.Frame):
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
        if "requested_command" not in data:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.03,
                0.6,
                "本记录没有独立的请求/执行队列。\n请选择69课第四轮误差或延迟组。",
                transform=ax.transAxes,
            )
            self.note.configure(text="旧数据不补造指令延迟。")
            self.canvas.draw_idle()
            return
        f = min(frame, record.lengths[key] - 1)
        ix = slice(max(0, f - 70), f + 1)
        t = record.timestamps[ix]
        color = record.track_by_key[key].color
        a, b = self.figure.subplots(1, 2)
        desired = np.where(data["braking_active"][ix], 0, data["requested_command"][ix, 0])
        a.step(t, desired, where="post", color="#64748b", ls="--", label="本帧请求（刹车记0）")
        a.step(t, data["commands"][ix, 0], where="post", color=color, label="本帧实际执行")
        b.step(
            t,
            data["braking_active"][ix].astype(int),
            where="post",
            color="#64748b",
            ls="--",
            label="请求制动",
        )
        b.step(
            t,
            data["executed_braking"][ix].astype(int),
            where="post",
            color=color,
            label="执行器制动",
        )
        a.set(title="指令发出后，车何时响应？", xlabel="时间 (s)", ylabel="速度 (m/s)")
        b.set(
            title="制动也经过同一延迟队列",
            xlabel="时间 (s)",
            yticks=[0, 1],
            yticklabels=["行驶", "制动"],
            ylim=(-0.15, 1.5),
        )
        for ax in (a, b):
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(alpha=0.15)
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(10)
        result = record.method_info(key)["result"]
        c = result["perturbation"]
        origin = int(data["executed_request_frame"][f])
        origin_text = f"{origin * 0.1:.1f}秒发出" if origin >= 0 else "初始保持/终止标记"
        if origin == -2:
            origin_text = "无新结果，沿用实际速度"
        self.note.configure(
            text=(
                f"{record.method_info(key)['label']} · 当前/本组末帧 {record.timestamps[f]:.1f}s；执行来源：{origin_text}。\n"
                f"注入世界y偏差 {c['dy_m'] * 100:+g}cm、朝向偏差 {c['yaw_deg']:+g}°、执行延迟 {c['delay_steps'] * 0.1:.1f}s；待执行运动请求 {int(data['pending_motion'][f])} 条。\n"
                + (
                    "本轮有本地保护消融；执行器可能接管，详见执行端保护页。"
                    if record.metadata.get("tracking_round") == 5
                    else "同一个算法的不同条件；"
                )
                + "虚线请求≠实线执行。加减速也耗时；末帧零命令不证明已停稳。"
            )
        )
        self.canvas.draw_idle()
