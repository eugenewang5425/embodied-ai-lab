"""Resizable camera/scene/diagnostics window with a shared record selection."""

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np

from embodied_learning.experiments.so101_grasping import PHASES
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.clock import ReplayClock
from embodied_learning.so101 import TABLE_Z

from .panels import DiagnosticsPanel, ImagePanel
from .renderer import ManipulationRenderer


class ManipulationWindow:
    def __init__(self, root, record, position=None, method=None):
        configure_plot_font()
        self.root, self.record = root, record
        self.closed, self.updating, self.timer = False, False, None
        self.position = position or record.positions[len(record.positions) // 2]
        preferred = record.summary.get("preferred_method") or (
            "center_cartesian" if "center_cartesian" in record.methods else "clamp"
        )
        self.method = method or preferred
        self.engine = ManipulationRenderer()
        self.last_frame = -1
        root.title("第72—73课 · SO-101三维标定与物理抓取")
        root.geometry(
            f"{min(1520, root.winfo_screenwidth() - 70)}x{min(950, root.winfo_screenheight() - 90)}+30+30"
        )
        root.minsize(1080, 760)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei", 10))
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="SO-101：夹爪碰到物体，怎样才算真正抓住？",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        selection = ttk.Frame(outer)
        selection.pack(fill="x", pady=5)
        ttk.Label(selection, text="物体初始位置").pack(side="left")
        self.position_var = tk.StringVar(value=self.position)
        self.position_labels = {}
        for p in record.positions:
            row, _ = record.select(p, record.available_methods(p)[0])
            x, y = row["config"]["object_xyz"][:2]
            yaw = row.get("object_yaw_deg", 0)
            self.position_labels[p] = f"{p} · X={x * 100:.1f}，Y={y * 100:+.1f}cm，朝向{yaw:+.0f}°"
        self.position_box = ttk.Combobox(
            selection, state="readonly", width=45, values=list(self.position_labels.values())
        )
        self.position_box.pack(side="left", padx=8)
        self.position_box.bind("<<ComboboxSelected>>", self._position_selected)
        method_row = ttk.Frame(outer)
        method_row.pack(fill="x", pady=(0, 4))
        self.method_var = tk.StringVar(value=self.method)
        self.method_buttons = {}
        for key, label in record.methods.items():
            button = ttk.Radiobutton(
                method_row,
                text=label.split("：")[0],
                value=key,
                variable=self.method_var,
                command=lambda k=key: self.select_method(k),
            )
            button.pack(side="left", padx=8)
            self.method_buttons[key] = button
        ttk.Button(selection, text="导出统计", command=self.export_current).pack(side="right")
        self.method_note = ttk.Label(outer, text="", wraplength=1400, foreground="#334155")
        self.method_note.pack(fill="x")
        self.method_note.bind(
            "<Configure>", lambda e: self.method_note.configure(wraplength=max(200, e.width))
        )
        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=5)
        self.play_button = ttk.Button(controls, text="播放", command=self.toggle)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="回到开始", command=lambda: self.seek(0)).pack(
            side="left", padx=5
        )
        ttk.Button(controls, text="单步", command=lambda: self.seek(self.clock.frame + 1)).pack(
            side="left"
        )
        self.speed = tk.StringVar(value="1.0")
        box = ttk.Combobox(
            controls,
            textvariable=self.speed,
            values=("0.25", "0.5", "1.0", "2.0"),
            width=5,
            state="readonly",
        )
        box.pack(side="left", padx=8)
        box.bind("<<ComboboxSelected>>", lambda _e: self.clock.set_speed(float(self.speed.get())))
        ttk.Label(controls, text="倍速").pack(side="left")
        self.contacts, self.trails = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="接触点", variable=self.contacts, command=self.refresh).pack(
            side="left", padx=8
        )
        ttk.Checkbutton(controls, text="运动轨迹", variable=self.trails, command=self.refresh).pack(
            side="left"
        )
        self.time_label = ttk.Label(controls, text="")
        self.time_label.pack(side="right")
        self.scale = ttk.Scale(outer, from_=0, to=1, command=self._slider)
        self.scale.pack(fill="x", pady=(0, 5))
        # Status gets space before the expandable camera panels.
        self.stats = ttk.Label(outer, text="", font=("Microsoft YaHei", 10), wraplength=1400)
        self.stats.pack(side="bottom", fill="x", pady=5)
        self.stats.bind("<Configure>", lambda e: self.stats.configure(wraplength=max(200, e.width)))
        grid = ttk.Frame(outer)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure((0, 1), weight=1, uniform="columns")
        grid.rowconfigure((0, 1), weight=1, uniform="rows")
        self.scene = ImagePanel(
            grid,
            "三维场景 · 左拖旋转，滚轮缩放",
            "青线=夹爪轨迹；橙线=物体轨迹；金点=本段目标；粉点=接触；绿区=放置。\n红/绿/蓝短线=底座和夹爪的X/Y/Z轴。",
        )
        self.overhead = ImagePanel(
            grid, "外部相机 · 俯视桌面", "按存档实际状态渲染；金圈是目标提示，未使用图像检测。"
        )
        self.wrist = ImagePanel(
            grid,
            "腕部相机 · 随手臂运动",
            "相机随夹爪移动；金圈是当前目标提示，目标在视野外时提示贴边。",
        )
        self.diagnostics = DiagnosticsPanel(grid)
        for panel, row, col in (
            (self.scene, 0, 0),
            (self.overhead, 0, 1),
            (self.wrist, 1, 0),
            (self.diagnostics, 1, 1),
        ):
            panel.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
        self.scene.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.scene.canvas.bind("<B1-Motion>", self._drag)
        self.scene.canvas.bind("<MouseWheel>", self._zoom)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", lambda _e: self.toggle())
        root.bind("<Escape>", lambda _e: self.close())
        self.load_selection()
        self.timer = root.after(40, self._tick)

    def _position_selected(self, _event):
        self.position = next(
            k for k, v in self.position_labels.items() if v == self.position_box.get()
        )
        self.load_selection()

    def select_method(self, method):
        self.method = method
        self.method_var.set(method)
        self.load_selection()

    def load_selection(self):
        old_time = self.clock.timestamps[self.clock.frame] if hasattr(self, "clock") else 0.0
        playing = self.clock.playing if hasattr(self, "clock") else False
        available = self.record.available_methods(self.position)
        if self.method not in available:
            self.method = available[0]
            self.method_var.set(self.method)
        for key, button in self.method_buttons.items():
            button.configure(state="normal" if key in available else "disabled")
        self.row, self.values = self.record.select(self.position, self.method)
        self.clock = ReplayClock(self.values["timestamps"])
        self.clock.set_speed(float(self.speed.get()))
        self.clock.seek(np.searchsorted(self.clock.timestamps, old_time, side="right") - 1)
        if playing:
            self.clock.play()
        self.position_box.set(self.position_labels[self.position])
        self.engine.load(self.row, self.values)
        self.diagnostics.load(self.record, self.position, self.method)
        self.updating = True
        self.scale.configure(to=len(self.clock.timestamps) - 1)
        self.updating = False
        self.method_note.configure(
            text=self.record.methods[self.method]
            + (
                "。比较空间路径与当前开口校准；空夹/偏差只在登记条件有回合。金点是名义参考中心，校准后的TCP会偏移。"
                if self.record.summary.get("study_round") == 2
                else "。A保留指尖参考，B改用实际指面，C加双侧接触检查，D同时使用；拒绝抬升也计失败。金点是名义参考中心。"
                if self.record.summary.get("study_round") == 3
                else "。三组只改变夹爪或抓取位置；"
            )
            + "使用已知物体位置（初始化给定），相机尚未参与控制。"
            + (
                f"本方法全批完成{sum(r['success'] for r in self.record.rows.values() if r['method'] == self.method)}/{sum(r['method'] == self.method for r in self.record.rows.values())}；开发例不代表泛化，候选尚未全范围通过。"
                if self.record.summary.get("study_round") in (2, 3)
                else ""
            )
        )
        if self.record.summary.get("study_round") == 3:
            count = sum(
                r["success"] for r in self.record.rows.values() if r["method"] == self.method
            )
            total = sum(r["method"] == self.method for r in self.record.rows.values())
            self.method_note.configure(
                text=(
                    f"{self.record.methods[self.method]}。 本批完成{count}/{total}；候选尚未全范围通过。\n"
                    "A保留指尖参考，B用实际指面，C加双侧接触检查，D同时使用；拒绝抬升也计失败。\n"
                    "已知物体位置由初始化给定，相机未参与控制。 金点为名义参考，校准后的TCP会偏移。"
                )
            )
        self.refresh()

    def seek(self, frame):
        self.clock.seek(int(float(frame)))
        self.refresh()

    def _slider(self, value):
        if not self.updating and hasattr(self, "clock"):
            self.seek(value)

    def toggle(self):
        self.clock.pause() if self.clock.playing else self.clock.play()
        self.play_button.configure(text="暂停" if self.clock.playing else "播放")

    def _tick(self):
        if self.closed:
            return
        frame = self.clock.advance()
        if frame != self.last_frame:
            self.refresh()
        self.play_button.configure(text="暂停" if self.clock.playing else "播放")
        self.timer = self.root.after(40, self._tick)

    def refresh(self):
        frame = self.clock.frame
        self.engine.set_frame(frame)
        for panel, view in (
            (self.scene, "scene"),
            (self.overhead, "overhead"),
            (self.wrist, "wrist_cam"),
        ):
            panel.present(
                self.engine.render(view, self.contacts.get(), self.trails.get()), frame, self.method
            )
        self.diagnostics.set_frame(frame, self.clock.timestamps)
        self.last_frame = frame
        self.updating = True
        self.scale.set(frame)
        self.updating = False
        phase = PHASES[int(self.values["phase"][frame])]
        self.time_label.configure(text=f"t={self.clock.timestamps[frame]:.2f}s · {phase}")
        v, row = self.values, self.row
        forces = v["jaw_normal_n"][frame]
        obj = v["object_xyz"][frame]
        target = v["waypoint_xyz"][frame]
        contact_note = ""
        if "gate_elapsed_s" in v:
            f = v["solved_jaw_min_n"][frame]
            elapsed = v["gate_elapsed_s"][frame]
            contact_note = (
                f"\n过去20ms两指最小力={f[0]:.2f}/{f[1]:.2f}N｜夹紧段连续双侧={elapsed:.2f}s（要求0.20s）｜"
                + (
                    "完整回合结果：门控拒绝抬升"
                    if row.get("contact_refused")
                    else "完成以原抓取评分为准"
                )
            )
            if self.method in ("surface", "surface_gate", "open", "offset"):
                cap = self.record.summary["calibration"]["preparation_opening_rad"]
                used_opening = v["qpos"][max(0, frame - 1), 5]
                contact_note += (
                    f"｜指面参考：超{np.rad2deg(cap):.1f}°，使用准备近似"
                    if used_opening > cap
                    else "｜指面参考：标定范围内"
                )
        self.stats.configure(
            text=(
                f"当前：{phase}｜物体中心离桌高度={(obj[2] - TABLE_Z) * 100:.1f}cm｜两指接触力={forces[0]:.2f}/{forces[1]:.2f}N｜"
                f"夹爪关节={np.rad2deg(v['qpos'][frame, 5]):.1f}°\n"
                f"本段名义参考 XYZ=({target[0] * 100:.1f}, {target[1] * 100:.1f}, {target[2] * 100:.1f})cm｜"
                f"完整回合：稳定抬起{'通过' if row['stable_lift'] else '未通过'}、放置{'通过' if row['stable_placement'] else '未通过'}｜"
                f"终点偏差={row['final_xy_error_m'] * 100:.2f}cm｜仿真回放，非实时实机"
                + contact_note
            )
        )

    def _drag_start(self, event):
        self.drag_xy = event.x, event.y

    def _drag(self, event):
        if not hasattr(self, "drag_xy"):
            return
        x, y = self.drag_xy
        self.engine.camera.azimuth -= (event.x - x) * 0.4
        self.engine.camera.elevation = np.clip(
            self.engine.camera.elevation + (event.y - y) * 0.3, -85, -5
        )
        self.drag_xy = event.x, event.y
        self.refresh()

    def _zoom(self, event):
        self.engine.camera.distance = np.clip(
            self.engine.camera.distance * np.exp(-event.delta * 0.001), 0.35, 2.0
        )
        self.refresh()

    def export_current(self, path=None):
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                defaultextension=".json",
                initialfile=f"{self.position}_{self.method}.json",
            )
        if path:
            Path(path).write_text(
                json.dumps(
                    {
                        "episode": self.row,
                        "frame": self.clock.frame,
                        "time_s": float(self.clock.timestamps[self.clock.frame]),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.timer is not None:
            self.root.after_cancel(self.timer)
        self.engine.close()
        self.diagnostics.close()
        self.root.destroy()
