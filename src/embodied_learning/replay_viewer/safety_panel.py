"""Body-frame diagnostic view; all observations come from current/past frames."""

from dataclasses import replace
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

from embodied_learning.braking_control import braking_path
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
    "queue_body_brake": "把排队动作算进去后不安全，请求制动",
    "planner_no_output": "没有新控制结果，车仍在执行",
    "abort_braking": "任务失败后实际减速",
    "contact_abort": "实体接触，仿真终止",
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
            if record.metadata.get("variant") in ("70", "71"):
                self._draw_decisions(record, key, frame)
                return
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
        if record.metadata.get("tracking_round") == 6:
            rig = replace(rig, margin_m=record.method_info(key)["result"]["controller_margin_m"])
        f = min(frame, record.lengths[key] - 1)
        a, b = self.figure.subplots(1, 2)
        pose = record.track_by_key[key].poses[f]
        scan, dep = record.lidar[key], record.depth[key]
        height, width = dep["values"].shape[1:]
        rig = replace(rig, height=height, width=width)
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
        model = record.metadata.get("braking_models", {}).get(key, "legacy")
        if model != "legacy":
            path = braking_path(command, rig, model == "coupled", rig.latency_s)[0]
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
        a.set_title("黄框：按本组制动模型的0.2秒延迟扫掠", fontsize=10)
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
            "存档深度 (m)"
            + (
                " · 控制输入"
                if key in record.metadata.get("depth_used_by", ["depth"])
                else " · 本组未使用"
            ),
            fontsize=10,
        )
        b.set_xlabel("像素列；灰色=无有效深度", fontsize=8)
        b.set_ylabel("像素行", fontsize=8)
        self.figure.colorbar(image, ax=b, shrink=0.65, label="光轴深度 (m)")
        reason = str(record.telemetry[key]["reason"][f])
        distance = np.linalg.norm(path[-1, :2])
        self.note.configure(
            text=f"{record.method_info(key)['label']} · t={record.timestamps[f]:.1f}s · {REASONS.get(reason, reason)}\n"
            f"速度 {command[0]:.2f} m/s，角速度 {command[1]:.2f} rad/s；中心刹停位移 {distance:.2f} m。黄框不含6cm余量；灰点仅是历史抽样。无回波≠安全。"
            + (" 第三轮还检查立即刹停，此图仅画延迟分支。" if model != "legacy" else "")
            + (
                " 第四轮仍画控制器的0.2秒预留，不是实际队列延迟；实际响应见误差与延迟页。"
                if record.metadata.get("tracking_round") == 4
                else ""
            )
            + (
                " 第五轮此页仅示意旧0.2秒分支，未画完整队列；实际接管见执行端保护页。"
                if record.metadata.get("tracking_round") in (5, 6)
                else ""
            )
        )
        if record.metadata.get("tracking_round") == 6:
            self.note.configure(
                text=self.note.cget("text")
                + " 第六轮黄框是运动学预测示意，未包含伺服滞后或接触侧滑；实际速度/接触另见接触页。"
            )
        if record.metadata.get("tracking_round") == 11:
            used = key in record.metadata["depth_used_by"]
            self.note.configure(
                text=f"{record.method_info(key)['label']} · t={record.timestamps[f]:.1f}s · {REASONS.get(reason, reason)}\n"
                f"黄框示意0.2秒延迟刹停，中心位移{distance:.2f}m；完整排队保护见执行端页。\n"
                f"紫点/深度{'参与控制' if used else '仅供复盘，本组未用'}；灰点是历史抽样。无回波不代表整车高度内安全。"
            )
        self.canvas.draw_idle()

    def _draw_decisions(self, record, key, frame):
        """Display recorded decision evidence; no rendered RGB used as a sensor."""
        from embodied_learning.experiments.campus_patrol import advance

        f = min(frame, record.lengths[key] - 1)
        data = record.telemetry[key]
        left, right = self.figure.subplots(1, 2)
        scan = record.lidar[key]
        angles = np.arange(scan["ranges"].shape[1]) * 2 * np.pi / scan["ranges"].shape[1]
        hit = scan["hits"][f]
        ranges = scan["ranges"][f, hit]
        left.scatter(
            ranges * np.sin(angles[hit]),
            ranges * np.cos(angles[hit]),
            s=9,
            color="#00aaa6",
            label="当前雷达回波",
        )
        left.add_patch(
            Circle((0, 0), 0.25, fill=False, color="#152b43", lw=2, label="半径25cm车身")
        )
        command = data["commands"][f]
        if data["recovery_active"][f]:
            poses = np.array(
                [
                    advance(np.zeros(3), command[0] * t, command[1] * t)
                    for t in np.linspace(0, 3, 30)
                ]
            )
            for p in poses[::5]:
                left.add_patch(Circle((p[1], p[0]), 0.27, color="#eebb44", alpha=0.18))
            left.plot(poses[:, 1], poses[:, 0], color="#bd7d00", label="当前恢复动作的3秒预测")
        left.set(
            xlim=(1.2, -1.2),
            ylim=(-1.1, 1.5),
            aspect="equal",
            xlabel="左 ← 横向 (m) → 右",
            ylabel="车头前方 (m)",
        )
        left.set_title("雷达只观察一个高度", fontsize=10)
        left.legend(loc="upper left", fontsize=7)
        start = max(0, f - 100)
        times = record.timestamps[start : f + 1]
        distance = data["arrival_distance"][start : f + 1].copy()
        budget = data["arrival_budget"][start : f + 1]
        # Goals switch after a declaration; break the line between decisions.
        jumps = np.r_[False, np.abs(np.diff(distance)) > 1.0]
        distance[jumps] = np.nan
        if record.metadata["variant"] == "71":
            right.plot(
                times,
                data["commands"][start : f + 1, 0],
                color="#119c74",
                drawstyle="steps-post",
                label="执行速度 m/s",
            )
            right.plot(
                times, data["arrival_r95"][start : f + 1], color="#8b5cf6", label="粒子半径 m"
            )
            right.axhline(0.25, color="#d97706", ls="--", label="半径拒绝线 .25m")
            active = data["recovery_active"][start : f + 1]
            right.fill_between(
                times,
                0,
                1,
                where=active,
                color="#eebb44",
                alpha=0.18,
                transform=right.get_xaxis_transform(),
            )
            right.set(xlabel="时间 (s)", ylabel="数值（单位见图例）", ylim=(-0.2, 1.05))
            right.set_title("恢复状态 · 黄底=正在恢复", fontsize=10)
        else:
            right.plot(times, distance, color="#64748b", label="估计距目标")
            right.plot(times, distance + budget, color="#119c74", label="距离＋内部预算")
            right.axhline(0.45, color="#d97706", ls="--", label="45cm参考线")
            right.set(xlabel="时间 (s)", ylabel="距离 (m)", ylim=(0, 1.2))
            right.set_title("判定前目标 · 最近12秒", fontsize=10)
        right.legend(loc="upper right", fontsize=7)
        right.grid(alpha=0.15)
        reason = str(data["reason"][f])
        reasons = {
            "following": "沿规划路径/检查到点",
            "observed_low_speed_escape": "观测支持，正在低速脱困",
            "global_route_blocked": "远处路线被阻断，拒绝局部恢复",
            "no_observed_safe_escape": "扫描不支持完整车身通过",
            "localization_ambiguous": "位置分散过大，停止恢复",
            "outside_map": "估计越界，停止",
            "recovery_budget_exhausted": "恢复时限用尽",
            "recovery_attempts_exhausted": "恢复次数用尽",
        }
        self.note.configure(
            text=f"{record.method_info(key)['label']} · t={record.timestamps[f]:.2f}s · {reasons.get(reason, reason)}\n"
            f"距目标 {data['arrival_distance'][f]:.3f}m；粒子半径 {data['arrival_r95'][f]:.3f}m＋固定0.10m预算；"
            f"执行速度 {command[0]:.2f}m/s；上一帧{'已停稳' if data['observed_stopped'][f] else '未停稳'}。\n"
            "预算不是误差保证；45cm真值线不变。近场雷达图只显示约2.4m范围；距离图限1.2m。金盘为预测。"
        )
        self.canvas.draw_idle()
