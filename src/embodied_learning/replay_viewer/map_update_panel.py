"""Timeline-aligned map edits and evaluation-only map disagreement."""

from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MapUpdatePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.note = ttk.Label(self, wraplength=700, style="Hint.TLabel")
        self.note.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.key = None

    def draw(self, record, key, frame):
        self.key = key
        self.figure.clear()
        d = record.telemetry.get(key, {})
        if not record.metadata.get("live_map"):
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.05,
                0.6,
                "请选择第69课第八轮及后续记录。\n第七轮仍使用冻结的原观测更新。",
                transform=ax.transAxes,
            )
            self.note.configure(text="地图修改不代表定位得到改善。")
        else:
            f = min(frame, record.lengths[key] - 1)
            t = record.timestamps[: f + 1]
            a, b = self.figure.subplots(1, 2)
            counts = d["map_counts"][: f + 1]
            a.step(t, counts[:, 0], label="新加入的三维格子", color="#d68026")
            a.step(t, counts[:, 1], label="有证据清除的格子", color="#119c74")
            a.set(title="本帧修改了多少？", xlabel="时间 (s)", ylabel="三维格子数量")
            error = d.get("body_height_map_error", d["map_error"])[: f + 1]
            b.plot(t, error[:, 0], label="地图多标的障碍", color="#c53e51")
            b.plot(t, error[:, 1], label="地图漏标的障碍", color="#247cc1")
            b.set(title="真实场景核对（仅评分）", xlabel="时间 (s)", ylabel="俯视格子数量")
            if "body_height_map_error" in d:
                b.set_title("车身高度内的障碍核对（仅评分）")
            for ax in (a, b):
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), fontsize=7, frameon=False)
                ax.grid(alpha=0.15)
                ax.tick_params(labelsize=8)
                ax.title.set_fontsize(10)
            self.note.configure(
                text=f"{record.method_info(key)['label']} · t={t[-1]:.1f}s；当前视图只用截至本帧的地图。\n三帧深度穿过才清除；遮挡/无效不清除。8m未命中可信只适用于当前解析仿真。\n右图真值核对只用于评分，不参与控制；改地图没有改变机器人估计位置。"
            )
            if "waypoint_progress" in d:
                wp, passed, _ = d["waypoint_progress"][f]
                h, w = record.depth[key]["values"].shape[1:]
                self.note.configure(
                    text=f"{record.method_info(key)['label']} · t={t[-1]:.1f}s · 深度{w}×{h} · 当前路点索引{int(wp)}\n本帧跳过已走首点：{'是' if passed else '否'}；索引是当前路径内的编号，重规划会重置。\n三帧、3×3邻域、12cm预算均不变；右图只评分，未修改自身定位。"
                )
                if record.metadata.get("tracking_round") in (11, 12, 13, 14):
                    used = key in record.metadata["depth_used_by"]
                    self.note.configure(
                        text=f"{record.method_info(key)['label']} · t={t[-1]:.1f}s · 当前路点{int(wp)}\n"
                        f"深度{'参与地图与避障' if used else '只供复盘，没有参与控制'}；障碍起初不在地图中，两侧可绕行。\n"
                        "右图只评分，排除高于整车的高杆；未修改自身定位，RGB为重渲染。"
                    )
        self.canvas.draw_idle()
