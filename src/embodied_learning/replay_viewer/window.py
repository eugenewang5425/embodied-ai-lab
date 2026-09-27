"""Native resizable replay window with shared selection and timeline."""

import json
import tkinter as tk
from dataclasses import asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from embodied_learning.plotting import configure_plot_font

from .analysis_panel import AnalysisPanel
from .camera_panel import CameraPanel
from .clock import ReplayClock
from .panels import MapPanel, ScenePanel, StatsPanel


class ReplayWindow:
    def __init__(self, root, source, case=None, variant=None):
        configure_plot_font()
        self.root, self.source = root, source
        self.closed = False
        self.record = None
        self.clock = None
        self.timer = None
        self.last_frame = -1
        self.current_selection = None
        self._updating = False
        root.title(source.title)
        root.geometry(
            f"{min(1540, root.winfo_screenwidth() - 90)}x{min(950, root.winfo_screenheight() - 100)}+35+35"
        )
        root.minsize(1100, 860)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei", 10))
        style.configure("PanelTitle.TLabel", font=("Microsoft YaHei", 12, "bold"))
        style.configure("Hint.TLabel", font=("Microsoft YaHei", 9), foreground="#526174")
        style.configure("Treeview", rowheight=27, font=("Microsoft YaHei", 10))
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=source.title, font=("Microsoft YaHei", 17, "bold")).pack(anchor="w")
        select = ttk.Frame(outer)
        select.pack(fill="x", pady=4)
        self.case_labels = {entry.case: entry.label for entry in source.entries}
        self.case_ids = list(self.case_labels)
        self.case_var, self.variant_var = tk.StringVar(), tk.StringVar()
        ttk.Label(select, text="实验回合").pack(side="left")
        self.case_box = ttk.Combobox(
            select,
            state="readonly",
            width=29,
            textvariable=self.case_var,
            values=list(self.case_labels.values()),
        )
        self.case_box.pack(side="left", padx=(6, 16))
        self.case_box.bind("<<ComboboxSelected>>", self._case_selected)
        ttk.Label(select, text=getattr(source, "variant_label", "记录版本")).pack(side="left")
        self.variant_box = ttk.Combobox(
            select, state="readonly", width=9, textvariable=self.variant_var
        )
        self.variant_box.pack(side="left", padx=6)
        self.variant_box.bind("<<ComboboxSelected>>", lambda _e: self.load_selection())
        self.export_button = ttk.Button(select, text="导出当前统计", command=self.export_current)
        self.export_button.pack(side="right")
        ttk.Label(select, text="拖动面板分隔线可调整大小", style="Hint.TLabel").pack(
            side="right", padx=12
        )
        self.layers = ttk.Frame(outer)
        self.track_vars = {}
        self.trails = tk.BooleanVar(value=True)
        viewbar = ttk.Frame(outer)
        viewbar.pack(fill="x", pady=3)
        self.view_mode = tk.StringVar(value="汇总")
        self.view_choice = tk.StringVar(value="summary")
        self.focus_key = None
        self.view_buttons = {}
        ttk.Label(viewbar, text="一键查看：").pack(side="left")
        for key, label in (("summary", "汇总叠图"), ("split", "并排分图")):
            button = ttk.Radiobutton(
                viewbar,
                text=label,
                value=key,
                variable=self.view_choice,
                style="Toolbutton",
                command=lambda k=key: self.choose_view(k),
            )
            button.pack(side="left", padx=2)
            self.view_buttons[key] = button
        self.method_bar = ttk.Frame(outer)
        self.method_bar.pack(fill="x", pady=(0, 3))
        ttk.Button(viewbar, text="自选图层", command=self._toggle_layers).pack(side="left", padx=8)
        self.lidar_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            viewbar, text="雷达探测点", variable=self.lidar_var, command=self.refresh
        ).pack(side="left", padx=8)
        ttk.Button(
            viewbar, text="方法含义与效果", command=lambda: self.analysis.tabs.select(2)
        ).pack(side="right")
        self.method_note = ttk.Label(outer, text="", wraplength=1400, style="Hint.TLabel")
        self.method_note.pack(fill="x", pady=(0, 3))
        self.method_note.bind(
            "<Configure>", lambda e: self.method_note.configure(wraplength=max(200, e.width))
        )

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 5))
        ttk.Button(controls, text="|◀", width=3, command=lambda: self.seek(0)).pack(side="left")
        ttk.Button(controls, text="◀", width=3, command=lambda: self.step(-1)).pack(
            side="left", padx=2
        )
        self.play_button = ttk.Button(controls, text="播放", width=6, command=self.toggle)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="▶", width=3, command=lambda: self.step(1)).pack(
            side="left", padx=2
        )
        ttk.Label(controls, text="速度").pack(side="left", padx=(10, 3))
        self.speed_var = tk.StringVar(value="4")
        self.speed_box = ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            values=("0.25", "0.5", "1", "2", "4", "8", "16"),
            state="readonly",
            width=5,
        )
        self.speed_box.pack(side="left")
        self.speed_box.bind(
            "<<ComboboxSelected>>", lambda _e: self.clock.set_speed(float(self.speed_var.get()))
        )
        ttk.Label(controls, text="倍").pack(side="left", padx=(2, 8))
        self.scale_var = tk.DoubleVar(value=0)
        self.scale = ttk.Scale(
            controls, from_=0, to=1, variable=self.scale_var, command=self._scale_seek
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.time_label = ttk.Label(controls, text="", width=32, anchor="e")
        self.time_label.pack(side="right")

        self.vertical = ttk.Panedwindow(outer, orient="vertical")
        self.vertical.pack(fill="both", expand=True)
        self.top = ttk.Panedwindow(self.vertical, orient="horizontal")
        self.bottom = ttk.Panedwindow(self.vertical, orient="horizontal")
        self.vertical.add(self.top, weight=3)
        self.vertical.add(self.bottom, weight=2)
        self.panels = []
        self.camera = self.add_panel(CameraPanel, "top", 2)
        self.scene = self.add_panel(ScenePanel, "top", 3)
        self.map = self.add_panel(MapPanel, "top", 2)
        self.analysis = self.add_panel(
            lambda parent: AnalysisPanel(parent, self.seek_time), "bottom", 3
        )
        self.error = self.analysis.error
        self.stats = self.add_panel(StatsPanel, "bottom", 4)
        self.status = ttk.Label(outer, text="", style="Hint.TLabel")
        self.status.pack(fill="x", pady=(4, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<space>", self._key_toggle)
        root.bind("<Left>", lambda _e: self.step(-1))
        root.bind("<Right>", lambda _e: self.step(1))
        root.bind("<Escape>", lambda _e: self.close())
        initial = case if case in self.case_labels else self.case_ids[0]
        self.case_var.set(self.case_labels[initial])
        variants = self._variants(initial)
        self.variant_box.configure(values=variants)
        self.variant_var.set(variant if variant in variants else variants[-1])
        self.load_selection()
        # Wait for Tk to map both nested Panedwindows before setting sash positions.
        # Their widths can still be 1 during the first idle callback.
        self.layout_timer = self.root.after(250, self._initial_layout)
        self.timer = root.after(100, self._tick)

    def add_panel(self, factory, section="bottom", weight=1):
        """Register a panel factory and join it to the shared replay lifecycle."""
        if section not in ("top", "bottom"):
            raise ValueError("panel section must be top or bottom")
        container = self.top if section == "top" else self.bottom
        panel = factory(container)
        container.add(panel, weight=weight)
        self.panels.append(panel)
        if self.record is not None:
            panel.set_record(self.record)
            panel.set_frame(
                self.clock.frame,
                {k for k, v in self.track_vars.items() if v.get()},
                self.trails.get(),
            )
        return panel

    def _initial_layout(self):
        self.layout_timer = None
        if self.closed:
            return
        # Initial camera rendering may delay mapping until after the timer expires.
        # Resolve pending geometry before applying sashes; otherwise Tk clamps to 0.
        self.root.update_idletasks()
        height = self.vertical.winfo_height()
        if min(height, self.vertical.winfo_width()) < 100:
            self.layout_timer = self.root.after(100, self._initial_layout)
            return
        self.vertical.sashpos(0, int(min(height * 0.55, max(180, height - 365))))
        self.root.update_idletasks()
        self.top.sashpos(0, int(self.top.winfo_width() * 0.26))
        self.top.sashpos(1, int(self.top.winfo_width() * 0.63))
        self.bottom.sashpos(0, int(self.bottom.winfo_width() * 0.43))
        self.refresh()

    def _variants(self, case):
        return list(dict.fromkeys(e.variant for e in self.source.entries if e.case == case))

    def _selected_case(self):
        return self.case_ids[self.case_box.current()]

    def _case_selected(self, _event=None):
        variants = self._variants(self._selected_case())
        self.variant_box.configure(values=variants)
        if self.variant_var.get() not in variants:
            self.variant_var.set(variants[-1])
        self.load_selection()

    def load_selection(self):
        case, variant = self._selected_case(), self.variant_var.get()
        old = self.record
        if self.clock:
            self.clock.pause()
        old_time = self.record.timestamps[self.clock.frame] if self.clock else 0.0
        try:
            record = self.source.load(case, variant)
            for panel in self.panels:
                panel.set_record(record)
        except Exception as exc:
            if old is None:
                for panel in self.panels:
                    close = getattr(panel, "close", None)
                    if close is not None:
                        close()
                raise
            for panel in self.panels:
                panel.set_record(old)
            previous_case, previous_variant = self.current_selection
            self.case_var.set(self.case_labels[previous_case])
            self.variant_box.configure(values=self._variants(previous_case))
            self.variant_var.set(previous_variant)
            self.status.configure(text=f"加载失败：{exc}")
            self.refresh()
            messagebox.showerror("记录校验失败", str(exc), parent=self.root)
            return
        self.record = record
        self.focus_keys = list(record.method_keys)
        if self.focus_key not in self.focus_keys:
            self.focus_key = "prior" if "prior" in self.focus_keys else self.focus_keys[-1]
        for widget in self.method_bar.winfo_children():
            widget.destroy()
        self.method_buttons = {}
        ttk.Label(self.method_bar, text="单独跟随：").pack(side="left")
        for key in self.focus_keys:
            style_name = f"{key}.Toolbutton"
            ttk.Style(self.root).configure(style_name, foreground=record.track_by_key[key].color)
            button = ttk.Radiobutton(
                self.method_bar,
                text=record.method_info(key)["label"],
                value=key,
                variable=self.view_choice,
                style=style_name,
                command=lambda k=key: self.choose_view(k),
            )
            button.pack(side="left", padx=2)
            self.method_buttons[key] = button
        self.clock = ReplayClock(record.timestamps)
        self.clock.speed = float(self.speed_var.get())
        if self.current_selection is not None and self.current_selection[0] == case:
            self.clock.seek(np.searchsorted(record.timestamps, old_time, side="right") - 1)
        self.current_selection = (case, variant)
        for widget in self.layers.winfo_children():
            widget.destroy()
        old_vars = self.track_vars
        self.track_vars = {}
        ttk.Label(self.layers, text="轨迹图层：").grid(row=0, column=0, sticky="w")
        for track in record.tracks:
            var = tk.BooleanVar(value=old_vars[track.key].get() if track.key in old_vars else True)
            self.track_vars[track.key] = var
            if track.key not in record.method_keys:
                continue
            style_name = f"{track.key}.TCheckbutton"
            ttk.Style(self.root).configure(style_name, foreground=track.color)
            ttk.Checkbutton(
                self.layers,
                text=record.method_info(track.key)["label"],
                variable=var,
                style=style_name,
                command=self._layers_changed,
            ).grid(row=0, column=record.method_keys.index(track.key) + 1, sticky="w", padx=(0, 8))
        ttk.Checkbutton(
            self.layers, text="轨迹尾迹", variable=self.trails, command=self.refresh
        ).grid(row=0, column=len(record.method_keys) + 1, sticky="e")
        self.scale.configure(to=len(record.timestamps) - 1)
        self.status.configure(
            text=record.metadata.get("protocol", "存档数据回放")
            + "  |  Space 播放/暂停，方向键单步，Esc 关闭"
        )
        self.last_frame = -1
        choice = self.view_choice.get()
        self.choose_view(choice if choice in ("summary", "split", *self.focus_keys) else "summary")

    def select_record(self, case, variant):
        """Programmatic selection uses the same path as the UI controls."""
        if case not in self.case_labels or variant not in self._variants(case):
            raise ValueError("record selection is unavailable")
        self.case_var.set(self.case_labels[case])
        self.variant_box.configure(values=self._variants(case))
        self.variant_var.set(variant)
        self.load_selection()

    def refresh(self):
        if self.closed or self.record is None:
            return
        frame = self.clock.frame
        visible = {key for key, var in self.track_vars.items() if var.get()}
        focus = self.focus_key
        if self.view_mode.get() == "单独":
            visible = self.record.method_tracks(focus)
        info = self.record.method_info(focus)
        self.method_note.configure(
            text=f"当前跟随：{info['label']}｜相机、雷达、目标和任务状态同步。{info['description']}"
        )
        for panel in self.panels:
            set_view = getattr(panel, "set_view", None)
            if set_view:
                set_view(focus, self.lidar_var.get())
            panel.set_frame(frame, visible, self.trails.get())
        self._updating = True
        self.scale_var.set(frame)
        self._updating = False
        self.time_label.configure(
            text=f"{self.record.timestamps[frame]:.2f} / {self.record.timestamps[-1]:.2f} s  ·  {frame + 1}/{len(self.record.timestamps)} 帧"
        )
        self.play_button.configure(text="暂停" if self.clock.playing else "播放")
        self.root.title(f"{self.source.title} — {self.record.title}")
        self.last_frame = frame

    def _layers_changed(self):
        for key in self.record.method_keys:
            true_key = self.record.true_key(key)
            if true_key != "truth":
                self.track_vars[true_key].set(self.track_vars[key].get())
        self.view_mode.set("汇总")
        self.view_choice.set("summary")
        self.analysis.tabs.select(0)
        self.refresh()

    def _toggle_layers(self):
        if self.layers.winfo_manager():
            self.layers.pack_forget()
        else:
            self.layers.pack(fill="x", pady=(0, 3), before=self.method_bar)

    def choose_view(self, choice):
        """One UI action sets layers, focus and analysis without seeking or pausing."""
        if choice not in ("summary", "split", *self.focus_keys):
            raise ValueError("view selection is unavailable")
        self.view_choice.set(choice)
        solo = choice in self.focus_keys
        self.view_mode.set("单独" if solo else "汇总")
        if solo:
            self.focus_key = choice
        visible = self.record.method_tracks(choice) if solo else set(self.track_vars)
        for key, var in self.track_vars.items():
            var.set(key in visible)
        layer = (
            self.record.method_info(choice).get("display_map", next(iter(self.record.maps)))
            if solo
            else ("叠加对照" if len(self.record.maps) > 1 else next(iter(self.record.maps)))
        )
        self.map.layer.set(layer)
        self.map._draw_map()
        self.analysis.tabs.select(1 if choice == "split" else 0)
        self.refresh()

    def set_focus(self, key, mode=None):
        if key not in self.focus_keys:
            raise ValueError("method selection is unavailable")
        self.focus_key = key
        self.choose_view(key if (mode or self.view_mode.get()) == "单独" else "summary")

    def _tick(self):
        if self.closed:
            return
        frame = self.clock.advance()
        if frame != self.last_frame:
            self.refresh()
        self.timer = self.root.after(70, self._tick)

    def _scale_seek(self, value):
        if not self._updating and self.clock:
            self.seek(round(float(value)))

    def seek(self, frame):
        self.clock.pause()
        self.clock.seek(frame)
        self.refresh()

    def seek_time(self, value):
        self.seek(np.searchsorted(self.record.timestamps, value, side="right") - 1)

    def step(self, delta):
        self.seek(self.clock.frame + delta)
        return "break"

    def toggle(self):
        if self.clock.playing:
            self.clock.pause()
        else:
            self.clock.play()
        self.refresh()

    def _key_toggle(self, event):
        if event.widget.winfo_class() not in ("TCombobox", "TEntry", "Entry"):
            self.toggle()
            return "break"
        return None

    def export_current(self, path=None):
        if path is None:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                defaultextension=".json",
                filetypes=[("JSON 统计", "*.json")],
                initialfile=f"{self.record.metadata.get('case', 'replay')}_frame{self.clock.frame}.json",
            )
        if not path:
            return
        frame = self.clock.frame
        report = {
            "title": self.record.title,
            "frame": frame,
            "timestamp_s": float(self.record.timestamps[frame]),
            "metadata": self.record.metadata,
            "presentation_camera": {
                "mode": "rerendered_from_archived_truth",
                "pose_source": self.record.true_key(self.focus_key),
                "image_size": [640, 480],
                **asdict(self.record.camera),
            },
            "tracks": {},
            "focus_method": self.focus_key,
            "lidar_provenance": self.record.metadata.get("lidar_mode", "unavailable"),
        }
        goal = self.record.goal_info(frame, report["focus_method"])
        report["current_goal"] = {"name": goal[0], "xy": goal[1].tolist()} if goal else None
        for track in self.record.tracks:
            item = {"pose": track.poses[frame].tolist()}
            if track.key in self.record.errors:
                item.update(
                    error_m=float(self.record.errors[track.key][frame]),
                    yaw_error_deg=float(self.record.yaw_errors[track.key][frame]),
                    whole_run=self.record.statistics[track.key],
                )
            report["tracks"][track.key] = item
        Path(path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.status.configure(text=f"已导出当前帧统计：{path}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.timer:
            self.root.after_cancel(self.timer)
            self.timer = None
        if self.layout_timer:
            self.root.after_cancel(self.layout_timer)
            self.layout_timer = None
        for panel in self.panels:
            close = getattr(panel, "close", None)
            if close is not None:
                close()
        self.root.destroy()
