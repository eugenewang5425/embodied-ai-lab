"""Windows teaching replay of verified, real ROS message/TF receipts (not live ROS)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose
from embodied_learning.experiments.ros2_system import DEFAULT_RESULTS
from embodied_learning.mobile_demo import MobileDemo
from embodied_learning.odometry import heading_error
from embodied_learning.ros2_motion_view import MotionView
from embodied_learning.ros2_stream import frame_stamp
from embodied_learning.teaching_demo import PlaybackClock

WINDOW_TITLE = "第二十课 · 看小车运动与定位"
TOPICS = (
    ("encoders", "/lesson20/encoders", "JointState", "传感器 → 定位 + 核验", "base_link"),
    ("landmarks", "/lesson20/landmark_points", "PointCloud2", "传感器 → 定位 + 核验", "sensor"),
    ("odom", "/lesson20/odom_pose", "PoseStamped", "定位 → 核验", "odom"),
    ("fused", "/lesson20/fused_pose", "PoseStamped", "定位 → 核验", "map"),
)


def load_trace(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    path = directory / "ros_trace.jsonl"
    if (
        report.get("experiment") != "ros2_message_and_tf_bridge"
        or report.get("schema_version") != 1
    ):
        raise ValueError("Not a lesson-20 recording")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trace_sha256"):
        raise ValueError("ROS trace checksum mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    steps = report.get("steps")
    if type(steps) is not int or steps < 1 or len(rows) != steps + 1 or report.get("dt_s") != 0.04:
        raise ValueError("Invalid timing or truncated trace")
    observed = set(report["observation_frames"])
    for frame, row in enumerate(rows):
        if row["frame"] != frame or row["time_s"] != frame * 0.04:
            raise ValueError("Disordered trace")
        if (row["stamp_sec"], row["stamp_nanosec"]) != frame_stamp(frame):
            raise ValueError("Invalid message timestamp")
        for name in ("truth", "odom", "fused", "map_to_odom", "map_to_sensor"):
            value = np.asarray(row[name], dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError("Invalid pose in trace")
        for expected, actual in (
            (compose(row["fused"], SENSOR_IN_BODY), row["map_to_sensor"]),
            (compose(row["map_to_odom"], row["odom"]), row["fused"]),
        ):
            if (
                not np.allclose(expected[:2], actual[:2], atol=1e-9, rtol=0)
                or abs(heading_error(expected[2], actual[2])) > 1e-9
            ):
                raise ValueError("Inconsistent TF chain")
        counts = {key: frame + 1 for key in ("encoders", "odom", "fused")}
        counts["landmarks"] = sum(f <= frame for f in observed)
        if row["received_counts"] != counts or row["observation"] != (frame in observed):
            raise ValueError("Incorrect message counts")
    if report["received_counts"] != rows[-1]["received_counts"]:
        raise ValueError("Summary count mismatch")
    return report, rows


class RosSystemDemo(MobileDemo):
    def __init__(self, root, report, rows, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        self.root, self.report, self.rows = root, report, rows
        self.clock = PlaybackClock(report["steps"], report["dt_s"], speed)
        self.last_tick, self.updating, self.after_id = time.perf_counter(), False, None
        self.before_correction = False
        turning = [
            i
            for i in range(1, len(rows))
            if abs(heading_error(rows[i]["truth"][2], rows[i - 1]["truth"][2])) > 1e-8
        ]
        self.turn_frames = [i for i in turning if i - 1 not in turning]
        root.title(WINDOW_TITLE)
        root.geometry("1180x820")
        root.minsize(1100, 780)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer, text="车在往前走，定位程序知道它在哪吗？", font=("Microsoft YaHei", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "本课目标：定位拆成 ROS 程序后，仍算出与原程序一致的位置。不是自动导航，也不是提高精度。\n"
                "运动学仿真 + 实际 ROS 收发记录的回放；非实车、非实时连接；定位不反过来控制轮子。"
            ),
        ).pack(anchor="w", pady=6)
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="单步", command=self.step).pack(side="left", padx=4)
        self.before_button = ttk.Button(
            controls, text="观测前一帧", command=self.before_observation
        )
        self.before_button.pack(side="left")
        self.next_button = ttk.Button(controls, text="下一次观测", command=self.next_observation)
        self.next_button.pack(side="left", padx=4)
        self.compare_button = ttk.Button(
            controls, text="同帧：看校正前", command=self.compare_correction
        )
        self.compare_button.pack(side="left", padx=4)
        self.turn_button = ttk.Button(controls, text="看转弯", command=self.next_turn)
        self.turn_button.pack(side="left", padx=4)
        if not self.turn_frames:
            self.turn_button.state(["disabled"])
        ttk.Button(controls, text="起点", command=lambda: self.seek(0)).pack(side="left")
        self.speed = tk.StringVar(value=f"{speed:g}")
        speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=["0.1", "0.25", "0.5", "1"],
            state="readonly",
            width=5,
        )
        speed_box.pack(side="left", padx=5)
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.timeline = tk.Scale(
            outer,
            from_=0,
            to=self.clock.steps,
            orient="horizontal",
            showvalue=False,
            command=self.slider_changed,
        )
        self.timeline.pack(fill="x")
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        scene = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(scene, text="① 先看小车与定位")
        self.motion_label = ttk.Label(scene, text="", font=("Microsoft YaHei", 12, "bold"))
        self.motion_label.pack(anchor="w", pady=5)
        ttk.Label(
            scene,
            text="蓝色车体＝仿真真值　紫色空圈＝只靠轮子推算　橙色空圈＝加入地标的定位　不是三辆车",
        ).pack(anchor="w")
        maps = ttk.Frame(scene)
        maps.pack(fill="both", expand=True, pady=6)
        maps.columnconfigure(0, weight=3, uniform="maps")
        maps.columnconfigure(1, weight=2, uniform="maps")
        maps.rowconfigure(0, weight=1)
        self.world_canvas = tk.Canvas(
            maps, background="#f8fafc", highlightthickness=0, width=600, height=350
        )
        self.zoom_canvas = tk.Canvas(
            maps, background="white", highlightthickness=0, width=400, height=350
        )
        self.world_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.zoom_canvas.grid(row=0, column=1, sticky="nsew")
        self.motion_view = MotionView(self.world_canvas, self.zoom_canvas, rows)
        self.event_label = ttk.Label(scene, text="", justify="left", wraplength=1040)
        self.event_label.pack(anchor="w", pady=8)
        ttk.Label(
            scene,
            text="整条主线：测量 → 定位 → 规划 → 控制运动。本课只接通测量到定位，后两步还没做。",
            wraplength=1040,
        ).pack(anchor="w")
        messages, frames = (
            ttk.Frame(self.notebook, padding=10),
            ttk.Frame(self.notebook, padding=10),
        )
        self.notebook.add(messages, text="② 再看消息怎么传")
        self.notebook.add(frames, text="③ 需要时看 TF")
        self.canvas = tk.Canvas(messages, height=145, background="#f8fafc", highlightthickness=0)
        self.canvas.pack(fill="x", pady=(0, 6))
        self.state_text = ttk.Label(messages, text="", justify="left")
        self.state_text.pack(anchor="w", pady=4)
        self.message_table = ttk.Treeview(
            messages,
            columns=("topic", "type", "nodes", "frame", "stamp", "count"),
            show="headings",
            height=4,
        )
        for key, label, width in zip(
            self.message_table["columns"],
            ("话题 topic", "消息类型", "谁发给谁", "frame_id", "最近时间戳 / s", "已收到条数"),
            (220, 120, 210, 100, 130, 100),
        ):
            self.message_table.heading(key, text=label)
            self.message_table.column(key, width=width, anchor="center")
        self.message_table.pack(fill="x")
        ttk.Label(
            messages,
            text=(
                "topic 是数据通道；消息类型约定字段；时间戳告诉你测的是哪一刻；frame_id 告诉你数字属于哪个坐标系。\n\n"
                "地标消息只有三个带编号的合成点，不是真实相机点云。编码器已经做过固定比例标定。\n\n"
                "到 2 s 观测时刻，要等该时刻的编码器和地标都到齐才校正；跨话题到达顺序不作为时间先后。\n\n"
                "本课采用本机可靠传输 + 单步确认，不模拟丢包/网络时延，不是实时性能测试。"
            ),
            justify="left",
        ).pack(anchor="w", pady=12)
        self.tf_table = ttk.Treeview(
            frames, columns=("edge", "xy", "yaw", "meaning"), show="headings", height=4
        )
        for key, label, width in zip(
            self.tf_table["columns"],
            ("坐标变换 / 查询", "x, y / m", "朝向 / °", "作用"),
            (290, 200, 110, 350),
        ):
            self.tf_table.heading(key, text=label)
            self.tf_table.column(key, width=width, anchor="center")
        self.tf_table.pack(fill="x")
        ttk.Label(
            frames,
            text=(
                "map → odom：地标校正造成的全局偏移，可跳变。odom → base_link：编码器累计的局部运动，不因地标到来而重置。\n\n"
                "base_link → sensor：固定安装（前 0.12 m、左 0.04 m、+30°），通过 /tf_static 广播。\n\n"
                "map → sensor 是 tf2 沿三条边算出的查询结果，不另外发布一条边；本次每一帧都核验了它。\n\n"
                "新的订阅者也实际收到了此前广播的静态安装变换；TF 不会自动识别或标定错误安装参数。"
            ),
            justify="left",
        ).pack(anchor="w", pady=12)
        self.status = ttk.Label(outer)
        self.status.pack(anchor="w", pady=8)
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        self.world_canvas.bind("<Configure>", lambda _: self.redraw())
        self.zoom_canvas.bind("<Configure>", lambda _: self.redraw())
        root.bind("<space>", lambda _: self.toggle())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()
        self.after_id = root.after(20, self.tick)

    def next_observation(self):
        target = next(
            (f for f in self.report["observation_frames"] if f > self.clock.index), self.clock.steps
        )
        self.seek(target)

    def before_observation(self):
        target = next(
            (f for f in self.report["observation_frames"] if f > self.clock.index), self.clock.steps
        )
        self.seek(target - 1)

    def seek(self, value):
        self.before_correction = False
        super().seek(value)

    def compare_correction(self):
        if self.rows[self.clock.index]["observation"]:
            self.clock.paused = True
            self.before_correction = not self.before_correction
            self.notebook.select(0)
            self.redraw()

    def next_turn(self):
        if self.turn_frames:
            self.seek(
                next((i for i in self.turn_frames if i > self.clock.index), self.turn_frames[0])
            )
            self.notebook.select(0)

    def redraw(self):
        if not hasattr(self, "status"):
            return
        i, row = self.clock.index, self.rows[self.clock.index]
        self.updating = True
        self.timeline.set(i)
        self.updating = False
        self.play_button.configure(text="播放" if self.clock.paused else "暂停")
        if not self.clock.paused or not row["observation"]:
            self.before_correction = False
        self.compare_button.configure(
            state="normal" if row["observation"] else "disabled",
            text="同帧：看校正后" if self.before_correction else "同帧：看校正前",
        )
        snapshot = self.motion_view.draw(i, self.before_correction)
        self.motion_label.configure(
            text=f"{row['time_s']:.2f} / {self.clock.steps * self.clock.dt:.2f} s｜{snapshot['action']}"
        )
        if row["observation"]:
            before, after = snapshot["error_before_cm"], snapshot["error_after_cm"]
            event = f"地标观测到齐 → 更新橙色定位；当前展示同一时刻的【{'校正前预测' if self.before_correction else '校正后结果'}】。切换前／后时蓝色车不移动。\n"
            event += f"这一次位置误差：{before:.2f} → {after:.2f} cm（{'变差：观测也带噪声，校正不保证每次更准' if after > before else '改善'}）。可点“同帧”按钮反复比较。"
        elif i == 0:
            event = "先点“播放”看蓝色小车直行，再点“看转弯”观察它原地转向。\n现在三个位置重合；随着读数噪声积累，估计会与真实位置分开。真值只用于画面和评价。"
        else:
            event = "轮子转动 → 编码器消息 → 位置继续推算。此刻没有新地标，不拿旧观测冒充当前位置。\n点“下一次观测”看估计怎样被修正；右侧以厘米显示实际误差，没有改动数据。"
        self.event_label.configure(text=event)
        state = "观测到齐：先预测，再校正" if row["observation"] else "只有本帧编码器更新：继续推算"
        if i == 0:
            state = "收到初始编码器计数，使用已知初始位姿；尚无地标观测"
        self.state_text.configure(
            text=f"路线 {self.report['route']} · 样本 #{self.report['run_index']} · 仿真时间 {row['time_s']:.2f} s\n"
            f"{state}。消息时间戳从 1 s 起算，以避免 tf2 将零时刻解释成“最新”。"
        )
        self.message_table.delete(*self.message_table.get_children())
        last_obs = next((f for f in reversed(self.report["observation_frames"]) if f <= i), None)
        for key, topic, message_type, nodes, coordinate_frame in TOPICS:
            frame = last_obs if key == "landmarks" else i
            ts = "尚无观测" if frame is None else f"{1 + frame * 0.04:.2f}"
            self.message_table.insert(
                "",
                "end",
                values=(
                    topic,
                    message_type,
                    nodes,
                    coordinate_frame,
                    ts,
                    row["received_counts"][key],
                ),
            )
        self.tf_table.delete(*self.tf_table.get_children())
        for edge, pose, meaning in (
            ("map → odom", row["map_to_odom"], "地图校正"),
            ("odom → base_link", row["odom"], "连续的里程计运动"),
            ("base_link → sensor", SENSOR_IN_BODY, "固定安装；/tf_static"),
            ("查询 map → sensor", row["map_to_sensor"], "三段相乘，非新增广播边"),
        ):
            self.tf_table.insert(
                "",
                "end",
                values=(
                    edge,
                    f"{pose[0]:+.5f}, {pose[1]:+.5f}",
                    f"{np.rad2deg(pose[2]):+.3f}",
                    meaning,
                ),
            )
        self.draw_nodes(row)
        self.status.configure(
            text=f"{'暂停' if self.clock.paused else '播放中'} · {self.clock.speed:g}× · "
            f"第 {i}/{self.clock.steps} 帧 · 先看运动；消息与坐标数据在第二、三页\n"
            "本窗口不启动 ROS 节点；真实运行完成后已清理其子进程。重新运行实验才会产生新消息。"
        )

    def draw_nodes(self, row):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width()
        if w < 50:
            return
        width = (w - 190) / 3
        boxes = [(20 + j * (width + 70), 30) for j in range(3)]
        entries = (
            ("传感器回放节点", "sensor", "发送编码器 / 地标读数", "#2563eb"),
            ("定位节点", "localizer", "只收测量，执行原融合算法", "#ea580c"),
            ("核验节点", "inspector", "接收消息，核验位姿与 TF", "#0f766e"),
        )
        for (x, y), (title, key, label, color) in zip(boxes, entries):
            c.create_rectangle(x, y, x + width, y + 90, outline=color, width=2, fill="white")
            c.create_text(x + width / 2, y + 20, text=title, font=("Microsoft YaHei", 12, "bold"))
            c.create_text(
                x + width / 2,
                y + 48,
                text=f"实际进程 PID {self.report['process_ids'][key]}",
                font=("Microsoft YaHei", 10),
            )
            c.create_text(x + width / 2, y + 71, text=label, font=("Microsoft YaHei", 10))
        for j in (0, 1):
            x, y = boxes[j]
            c.create_line(x + width + 4, y + 40, boxes[j + 1][0] - 5, y + 40, arrow="last", width=2)
        c.create_text(
            w / 2,
            12,
            text=f"真实发布/订阅连接 · 核验节点累计收到 {row['received_counts']['fused']} 条融合位姿",
            font=("Microsoft YaHei", 10),
        )


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    report, rows = load_trace(args.results)
    root = tk.Tk()
    RosSystemDemo(root, report, rows)
    root.mainloop()


if __name__ == "__main__":
    main()
