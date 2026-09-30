"""Plain-language guide, separated trajectories and synchronized error plots."""

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .braking_panel import BrakingPanel
from .execution_panel import ExecutionPanel
from .panels import ErrorPanel
from .robustness_panel import RobustnessPanel
from .safety_panel import SafetyPanel
from .tracking_panel import TrackingPanel

RESULTS = {
    "completed": "全部巡检完成",
    "false_completion": "自认为完成，但有点未真正到达",
    "timeout": "超时/停滞",
    "no_path": "估计位置无法继续规划",
    "collision_abort": "发生接触，仿真终止",
    "goal_overshoot": "刹停后仍在目标圈外",
    "stalled": "停车后无法继续前进",
    "control_expired": "控制信息过期，本地停住",
}


class AnalysisPanel(ttk.Frame):
    def __init__(self, parent, seek_time):
        super().__init__(parent)
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True)
        self.error = ErrorPanel(self.tabs, seek_time)
        self.tabs.add(self.error, text="误差随时间")
        self.separate = ttk.Frame(self.tabs)
        self.tabs.add(self.separate, text="各方法分开看")
        self.figure = Figure(figsize=(7, 3.5), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, self.separate)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        help_frame = ttk.Frame(self.tabs)
        self.tabs.add(help_frame, text="方法含义与效果")
        self.help = tk.Text(
            help_frame, wrap="word", font=("Microsoft YaHei", 10), padx=10, pady=8, height=12
        )
        scroll = ttk.Scrollbar(help_frame, command=self.help.yview)
        self.help.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.help.pack(fill="both", expand=True)
        self.safety = SafetyPanel(self.tabs)
        self.tabs.add(self.safety, text="车身与决策诊断")
        self.tracking = TrackingPanel(self.tabs)
        self.tabs.add(self.tracking, text="跟踪与余量")
        self.braking = BrakingPanel(self.tabs)
        self.tabs.add(self.braking, text="制动过程")
        self.robustness = RobustnessPanel(self.tabs)
        self.tabs.add(self.robustness, text="误差与延迟")
        self.execution = ExecutionPanel(self.tabs)
        self.tabs.add(self.execution, text="执行端保护")
        self.focus_key = None
        self.frame = 0
        self.visible = set()
        self.trails = True
        self.show_lidar = True
        self.record = None
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)

    def set_record(self, record):
        self.record = record
        self.focus_key = record.method_keys[0]
        self.error.set_record(record)
        self.figure.clear()
        keys = record.method_keys
        axes = self.figure.subplots(math.ceil(len(keys) / 2), 2, squeeze=False).ravel()
        self.plots = {}
        extent = record.extent
        all_xy = np.concatenate([t.poses[:, :2] for t in record.tracks])
        x0, y0 = np.minimum([extent[0], extent[2]], all_xy.min(axis=0))
        x1, y1 = np.maximum([extent[1], extent[3]], all_xy.max(axis=0))
        for ax, key in zip(axes, keys):
            ax.imshow(
                record.maps[record.method_info(key).get("display_map", next(iter(record.maps)))],
                origin="lower",
                extent=extent,
                cmap="Greys",
                vmin=0,
                vmax=1,
                alpha=0.4,
            )
            ax.set(xlim=(x0 - 0.2, x1 + 0.2), ylim=(y0 - 0.2, y1 + 0.2), aspect="equal")
            ax.set_title(
                record.method_info(key)["label"], fontsize=9, color=record.track_by_key[key].color
            )
            ax.tick_params(labelsize=7)
            (actual,) = ax.plot([], [], color="#334155", lw=1.2)
            (estimated,) = ax.plot([], [], color=record.track_by_key[key].color, ls="--", lw=1.1)
            (point,) = ax.plot([], [], "o", color=record.track_by_key[key].color, ms=4)
            (goal,) = ax.plot([], [], "*", color="#d79a00", ms=9)
            scan = ax.scatter([], [], c="#00bdb0", s=4)
            self.plots[key] = (actual, estimated, point, goal, scan)
        for ax in axes[len(keys) :]:
            ax.set_visible(False)
        self._write_guide()

    def _write_guide(self):
        r = self.record
        text = [
            "先看任务是否完成，再看定位是否准确。",
            "实线：机器人实际走过的位置。虚线：算法以为自己走到的位置。两线贴合表示定位准；两线偏开表示估计有误。",
            "青色小点：当前雷达打到障碍物上的回波，按所选算法估计的位置放到地图上。定位有误时，点会偏离墙面。没有回波的方向不画成障碍点。",
            "金色星/旗：当前要去的巡检点；金线：当前规划路径。相机中的金色符号是导航提示，目标可能被建筑挡住。",
            "PF 就是粒子滤波：同时保留许多可能的位置，用雷达和地图的匹配程度，给靠谱的猜测更高权重。它依赖地图质量，也可能选错位置。\n",
        ]
        if r.metadata.get("tracking_round") == 4:
            text[-1] = (
                "本轮固定同一个同比例制动控制器，人工注入一种恒定位置/朝向偏差或执行延迟；没有运行PF或视觉定位。角度偏差不能用位置误差均值表示。\n"
            )
        if r.metadata.get("tracking_round") == 5:
            text[-1] = (
                "本轮保持精确定位，比较排队动作预测和本地保护；没有运行PF或视觉定位。规划无输出期间车继续走，本地保护停车也不算到达。\n"
            )
        for key in r.method_keys:
            info = r.method_info(key)
            text += [info["label"], info["description"]]
            if key in r.statistics:
                s = r.statistics[key]
                text += [
                    f"本轮平均位置误差 {s['mean_m']:.3f} m；95% 时刻不超过 {s['p95_m']:.3f} m。"
                ]
                if (
                    "odom" in r.statistics
                    and key != "odom"
                    and r.metadata.get("scene_kind") != "campus"
                ):
                    base = r.statistics["odom"]["mean_m"]
                    delta = (base - s["mean_m"]) / max(base, 1e-12) * 100
                    text += [
                        f"相对只靠轮子推算，本轮平均误差{'降低' if delta >= 0 else '增加'} {abs(delta):.1f}%。这是本轮结果，不代表所有环境。"
                    ]
            result = info.get("result")
            if result:
                text += [
                    f"任务结果：{RESULTS[result['status']]}；真正到达 {result['reached']}/{result['target_count']} 点，自报到达 {result['announced']} 点，接触障碍 {result['contacts']} 帧。"
                ]
                if "perturbation" in result:
                    c = result["perturbation"]
                    text += [
                        f"注入世界y误差 {c['dy_m'] * 100:+g}cm，朝向误差 {c['yaw_deg']:+g}度，实际执行延迟 {c['delay_steps'] * 0.1:.1f}s。"
                    ]
                if result.get("min_clearance_m") is not None:
                    text += [f"实际运动中车身离障碍最近 {result['min_clearance_m']:.3f} m。"]
                if "false_arrivals" in result:
                    text += [
                        f"误报到点 {result['false_arrivals']} 次；恢复动作 {result['recovery_active_frames']} 帧。"
                    ]
                if "expanded_swept_min_m" in result:
                    text += [
                        f"6厘米外扩足印的最小投影分离量 {result['expanded_swept_min_m'] * 100:.2f}cm；负值表示预留空间发生重叠。近场保护制动 {result['braking_events']} 次、{result['braking_frames']} 帧。"
                    ]
                if "physically_stopped" in result:
                    text += [
                        f"实际停稳：{'是' if result['physically_stopped'] else '否'}；完成且全程外扩余量合格：{'是' if result['safe_completed'] else '否'}。终止触发后的实际制动尾段 {result['tail_time_s']:.1f}s、{result['tail_distance_m'] * 100:.1f}cm。"
                    ]
                if "planning_over_200ms" in result and result["planning_calls"]:
                    text += [
                        f"规划调用 {result['planning_calls']} 次，其中 {result['planning_over_200ms']} 次超过200毫秒。仿真会等待计算，不能把播放速度当实时性能。"
                    ]
            text += [""]
        text += [
            "园区各方法各自开车，运行时长和路径不同。均值仅统计各自有效时段，不能当成同一路径上的配对提升；很早失败的方法也可能均值很小。",
            "统计怎么读",
            "当前：这个时刻与实际位置相差几米。朝向差：车头方向错了几度。均值：整段有效记录的平均误差。P95：95% 的时刻误差不超过这个值。末端：这一方法结束时的误差。数值越小越好，但不能替代任务完成率。",
            r.metadata.get("protocol", ""),
            r.metadata.get("lidar_note", ""),
            r.metadata.get("sensor_note", ""),
            "结束较早的方法在回放中保持末帧；这些停留帧不参加统计。第67/69课使用无偏无滑移里程计来隔离规划避障问题，零定位误差是实验设定。",
        ]
        self.help.configure(state="normal")
        self.help.delete("1.0", "end")
        self.help.insert("1.0", "\n\n".join(text))
        self.help.configure(state="disabled")

    def set_view(self, key, lidar=True):
        self.show_lidar = lidar
        self.focus_key = key

    def set_frame(self, frame, visible, trails=True):
        self.frame, self.visible, self.trails = frame, visible, trails
        self.error.set_frame(frame, visible, trails)
        if self.tabs.index(self.tabs.select()) == 1:
            self._draw_separate()
        elif self.tabs.index(self.tabs.select()) == 3:
            self.safety.draw(self.record, self.focus_key, frame)
        elif self.tabs.index(self.tabs.select()) == 4:
            self.tracking.draw(self.record, self.focus_key, frame)
        elif self.tabs.index(self.tabs.select()) == 5:
            self.braking.draw(self.record, self.focus_key, frame)
        elif self.tabs.index(self.tabs.select()) == 6:
            self.robustness.draw(self.record, self.focus_key, frame)
        elif self.tabs.index(self.tabs.select()) == 7:
            self.execution.draw(self.record, self.focus_key, frame)

    def _draw_separate(self):
        r, f = self.record, self.frame
        for key, (actual, estimated, point, goal, scan) in self.plots.items():
            n = min(f + 1, r.lengths.get(key, len(r.timestamps)))
            idx = np.unique(np.r_[np.arange(0, n, 4), n - 1])
            truth = r.track_by_key[r.true_key(key)].poses[idx]
            est = r.track_by_key[key].poses[idx]
            actual.set_data(truth[:, 0], truth[:, 1])
            estimated.set_data(est[:, 0], est[:, 1])
            actual.set_visible(self.trails)
            estimated.set_visible(self.trails)
            point.set_data([est[-1, 0]], [est[-1, 1]])
            info = r.goal_info(f, key)
            goal.set_data([info[1][0]], [info[1][1]]) if info else goal.set_data([], [])
            scan.set_offsets(r.scan_world(f, key) if self.show_lidar else np.empty((0, 2)))
        self.canvas.draw_idle()

    def _tab_changed(self, _event):
        if self.record is not None and self.tabs.index(self.tabs.select()) == 1:
            self._draw_separate()
        elif self.record is not None and self.tabs.index(self.tabs.select()) == 3:
            self.safety.draw(self.record, self.focus_key, self.frame)
        elif self.record is not None and self.tabs.index(self.tabs.select()) == 4:
            self.tracking.draw(self.record, self.focus_key, self.frame)
        elif self.record is not None and self.tabs.index(self.tabs.select()) == 5:
            self.braking.draw(self.record, self.focus_key, self.frame)
        elif self.record is not None and self.tabs.index(self.tabs.select()) == 6:
            self.robustness.draw(self.record, self.focus_key, self.frame)
        elif self.record is not None and self.tabs.index(self.tabs.select()) == 7:
            self.execution.draw(self.record, self.focus_key, self.frame)
