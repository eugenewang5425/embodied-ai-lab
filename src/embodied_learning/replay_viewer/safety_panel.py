"""Body-frame diagnostic view; all observations come from current/past frames."""

from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from embodied_learning.navigation_geometry import (
    NavigationRig,
    body_points,
    depth_points,
    lidar_points,
    stop_path,
    world_points,
)

REASONS = {
    "following": "沿当前路径走",
    "swept_body_brake": "刹停包络触及已知障碍，制动",
    "frontal_brake": "前方雷达触发停车",
    "no_observed_route": "膨胀后找不到路，停止",
}


class SafetyPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.note = ttk.Label(self, wraplength=720, style="Hint.TLabel")
        self.note.pack(side="bottom", fill="x")
        self.figure = Figure(figsize=(7, 3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.record = None
        self.key = None
        self.frame = 0

    def draw(self, record, key, frame):
        self.record, self.key, self.frame = record, key, frame
        self.figure.clear()
        if "rig" not in record.metadata:
            ax = self.figure.subplots()
            ax.axis("off")
            ax.text(
                0.04,
                0.7,
                "第68课沿用第66课机器人与控制器。\n没有存档深度，不生成虚构的深度观测。\n请选择第67课记录查看车身标定。",
                transform=ax.transAxes,
                fontsize=12,
            )
            self.note.configure(text="错图修复与身体避障是不同实验；两套控制器尚未合并。")
            self.canvas.draw_idle()
            return
        rig = NavigationRig(**record.metadata["rig"])
        f = min(frame, record.lengths[key] - 1)
        a, b = self.figure.subplots(1, 2)
        pose = record.track_by_key[key].poses[f]
        scan, dep = record.lidar[key], record.depth[key]
        points = lidar_points(scan["ranges"][f], scan["hits"][f], rig)
        current = depth_points(dep["values"][f], dep["valid"][f], rig)
        history = []
        # A labelled subset of past observations for visual explanation, not the
        # controller's full persistent voxel map, and never any future frame.
        for j in range(max(0, f - 30), f, 3):
            p = depth_points(dep["values"][j], dep["valid"][j], rig)[::6]
            p = world_points(p, record.track_by_key[key].poses[j])
            history.append(body_points(p, pose))
        if history:
            past = np.vstack(history)
            a.scatter(past[:, 1], past[:, 0], s=3, color="#b1b7c4", label="过去3秒深度示意")
        a.scatter(points[:, 1], points[:, 0], s=6, color="#00aaa6", label="当前雷达")
        a.scatter(current[::4, 1], current[::4, 0], s=4, color="#8c67be", label="当前深度")
        command = record.telemetry[key]["commands"][f]
        path = stop_path(*command, rig)
        footprint = np.array(
            [
                [-rig.rear_m, -rig.half_width_m, 0],
                [rig.front_m, -rig.half_width_m, 0],
                [rig.front_m, rig.half_width_m, 0],
                [-rig.rear_m, rig.half_width_m, 0],
            ]
        )
        for j in np.unique(np.linspace(0, len(path) - 1, min(8, len(path))).astype(int)):
            polygon = world_points(footprint, path[j])
            a.add_patch(
                Polygon(polygon[:, [1, 0]], facecolor="#eebb44", edgecolor="#bd7d00", alpha=0.15)
            )
        a.add_patch(
            Polygon(
                footprint[:, [1, 0]],
                fill=False,
                edgecolor="#152b43",
                linewidth=2,
                label="车身55×44 cm",
            )
        )
        a.set(
            xlim=(1.15, -1.15),
            ylim=(-0.65, 2),
            aspect="equal",
            xlabel="左 ← 横向 (m) → 右",
            ylabel="车头前方 (m)",
        )
        a.set_title("黄框：按本帧速度刹停扫过的车身", fontsize=10)
        handles, labels = a.get_legend_handles_labels()
        self.figure.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=7)
        image = b.imshow(
            np.ma.array(dep["values"][f], mask=~dep["valid"][f]),
            cmap="magma_r",
            vmin=0,
            vmax=rig.max_range_m,
        )
        b.set_facecolor("#c6ccd4")
        b.set_title(
            "存档深度 (m)" + (" · 本组未使用" if key != "depth" else " · 控制输入"), fontsize=10
        )
        b.set_xlabel("像素列；灰色=无有效深度", fontsize=8)
        b.set_ylabel("像素行", fontsize=8)
        self.figure.colorbar(image, ax=b, shrink=0.65, label="光轴深度 (m)")
        reason = str(record.telemetry[key]["reason"][f])
        distance = np.linalg.norm(path[-1, :2])
        self.note.configure(
            text=f"{record.method_info(key)['label']} · t={record.timestamps[f]:.1f}s · {REASONS.get(reason, reason)}\n"
            f"速度 {command[0]:.2f} m/s，角速度 {command[1]:.2f} rad/s；中心刹停位移 {distance:.2f} m。黄框不含6cm余量；灰点仅是历史抽样。无回波≠安全。"
        )
        self.canvas.draw_idle()
