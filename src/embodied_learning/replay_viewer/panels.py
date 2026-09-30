"""Independent Tk panels sharing set_record / set_frame calls."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import ImageTk

from .scene import SceneRenderer


class ScenePanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=6)
        bar = ttk.Frame(self)
        bar.pack(fill="x")
        ttk.Label(bar, text="3D 场景", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(bar, text="重置视角", command=self.reset).pack(side="right")
        self.note = ttk.Label(self, text="", style="Hint.TLabel", wraplength=480)
        self.note.pack(fill="x")
        self.canvas = tk.Canvas(self, bg="#263343", highlightthickness=0, width=650, height=370)
        self.hint = ttk.Label(
            self,
            text="拖动旋转 · 滚轮缩放 · 双击复位 · 1 m 地面网格",
            style="Hint.TLabel",
            wraplength=480,
        )
        self.hint.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.engine = SceneRenderer()
        self.image_id = self.canvas.create_image(0, 0, anchor="nw")
        self.photo = None
        self.frame = 0
        self.record = None
        self.focus_key = "truth"
        self.visible = set()
        self.trails = True
        self.pending_resize = None
        self.drag = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Double-Button-1>", lambda _e: self.reset())
        self.canvas.bind("<Configure>", self._resize)

    def set_record(self, record):
        self.record = record
        self.engine.load(record)
        self.note.configure(text=record.metadata.get("scene_note", "地图几何的立体展示"))
        if record.metadata.get("scene_kind") == "campus":
            self.hint.configure(text="拖动旋转 · 滚轮缩放 · 金旗=目标 · 青点=雷达回波")
        else:
            self.hint.configure(text="拖动旋转 · 滚轮缩放 · 双击复位 · 1 m 地面网格")

    def set_frame(self, frame, visible, trails=True):
        self.frame, self.visible, self.trails = frame, visible, trails
        if self.record is None or not self.canvas.winfo_ismapped():
            return
        size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        image = self.engine.render(frame, size, visible, trails)
        # Large monitors can exceed the renderer budget; display at panel size.
        if image.size != size:
            image = image.resize(size)
        self.photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        info = self.record.goal_info(frame, self.focus_key)
        if info:
            self.note.configure(
                text=f"{self.record.method_info(self.focus_key)['label']} → {info[0]}（金色旗标/路线）"
            )

    def set_view(self, key, lidar=True):
        self.focus_key = self.engine.focus_key = key
        self.engine.show_lidar = lidar

    def _press(self, event):
        self.drag = (event.x, event.y)

    def _drag(self, event):
        if self.drag:
            self.engine.orbit(event.x - self.drag[0], event.y - self.drag[1])
            self.drag = (event.x, event.y)
            self.set_frame(self.frame, self.visible, self.trails)

    def _wheel(self, event):
        self.engine.zoom(event.delta / 120)
        self.set_frame(self.frame, self.visible, self.trails)

    def reset(self):
        if self.record is not None:
            self.engine.reset_camera()
            self.set_frame(self.frame, self.visible, self.trails)

    def _resize(self, event):
        self.note.configure(wraplength=max(120, event.width))
        self.hint.configure(wraplength=max(120, event.width))
        if self.pending_resize:
            self.after_cancel(self.pending_resize)
        self.pending_resize = self.after(180, self._finish_resize)

    def _finish_resize(self):
        self.pending_resize = None
        self.set_frame(self.frame, self.visible, self.trails)

    def close(self):
        if self.pending_resize:
            self.after_cancel(self.pending_resize)
            self.pending_resize = None
        self.engine.close()


class PlotPanel(ttk.Frame):
    def __init__(self, parent, title, figsize):
        super().__init__(parent, padding=6)
        self.bar = ttk.Frame(self)
        self.bar.pack(fill="x")
        ttk.Label(self.bar, text=title, style="PanelTitle.TLabel").pack(side="left")
        self.figure = Figure(figsize=figsize, dpi=100, layout="constrained")
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.frame = 0
        self.visible = set()
        self.record = None


class MapPanel(PlotPanel):
    def __init__(self, parent):
        super().__init__(parent, "地图与轨迹", (5, 3.5))
        self.layer = tk.StringVar()
        self.combo = ttk.Combobox(self.bar, textvariable=self.layer, state="readonly", width=15)
        self.combo.pack(side="right")
        self.combo.bind("<<ComboboxSelected>>", lambda _e: self._draw_map())
        self.all_poses = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.bar, text="含越界轨迹", variable=self.all_poses, command=self.reset_limits
        ).pack(side="right")
        self.map_hint = ttk.Label(
            self, text="滚轮缩放 · 拖动平移 · 双击复位；地图为固定存档结果", style="Hint.TLabel"
        )
        self.map_hint.pack(side="bottom", anchor="w", before=self.canvas.get_tk_widget())
        self.canvas.mpl_connect("scroll_event", self._zoom)
        self.canvas.mpl_connect("button_press_event", self._press)
        self.canvas.mpl_connect("motion_notify_event", self._drag)
        self.canvas.mpl_connect("button_release_event", lambda _e: setattr(self, "drag", None))
        self.drag = None
        self.focus_key = "truth"
        self.show_lidar = True

    def set_record(self, record):
        self.record = record
        self.frame = 0
        self.map_hint.configure(
            text="滚轮缩放 · 拖动平移 · 双击复位；当前地图随时间和方法同步"
            if record.metadata.get("live_map")
            else "滚轮缩放 · 拖动平移 · 双击复位；地图为固定存档结果"
        )
        self.ax.clear()
        options = list(record.maps)
        if len(options) > 1:
            options.append("叠加对照")
        self.combo.configure(values=options)
        if self.layer.get() not in options:
            self.layer.set(options[-1])
        self.raster = self.ax.imshow(
            next(iter(record.maps.values())),
            origin="lower",
            extent=record.extent,
            interpolation="nearest",
            cmap="Greys",
            vmin=0,
            vmax=1,
        )
        self.lines, self.markers = {}, {}
        for track in record.tracks:
            (self.lines[track.key],) = self.ax.plot(
                [],
                [],
                color=track.color,
                lw=1.4 if track.key != "truth" else 2.3,
                ls="-" if track.role == "truth" or track.key == "truth" else "--",
            )
            (self.markers[track.key],) = self.ax.plot(
                [], [], "o", color=track.color, markersize=5, mec="white", mew=0.7
            )
        self.scan_points = self.ax.scatter([], [], c="#00bdb0", s=13, marker=".", zorder=5)
        (self.goal_marker,) = self.ax.plot(
            [], [], "*", color="#d79a00", ms=14, mec="#523d00", zorder=7
        )
        (self.plan_line,) = self.ax.plot([], [], color="#d79a00", lw=1.5, alpha=0.85)
        self.goal_text = self.ax.text(
            0.02,
            0.02,
            "",
            transform=self.ax.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        self.ax.set_xlabel("x (m)", fontsize=9)
        self.ax.set_ylabel("y (m)", fontsize=9)
        self.ax.tick_params(labelsize=8)
        self.ax.set_aspect("equal")
        self._draw_map()
        self.reset_limits()

    def _draw_map(self):
        if self.record is None:
            return
        key = self.layer.get()
        live = self.record.telemetry.get(self.focus_key, {}).get("map_frames")
        current = live[self.frame] if live is not None else None
        if key == "叠加对照":
            maps = list(self.record.maps.values())
            ref, comparison = maps[0], maps[-1]
            if self.record.metadata.get("live_map") and current is not None:
                comparison = current
            rgb = np.full((*ref.shape, 3), 0.95)
            rgb[ref & ~comparison] = [0.19, 0.23, 0.29]
            rgb[comparison & ~ref] = [0.92, 0.58, 0.24]
            rgb[ref & comparison] = [0.16, 0.55, 0.58]
            self.raster.set_data(rgb)
            names = list(self.record.maps)
            title = f"灰：{names[0]} / 橙：{names[-1]}多标 / 青：重合"
        else:
            self.raster.set_data(
                current
                if key == "当前观测通行图" and current is not None
                else self.record.maps[key]
            )
            title = key
        self.ax.set_title(title, fontsize=9)
        self.canvas.draw_idle()

    def reset_limits(self):
        if self.record is None:
            return
        x0, x1, y0, y1 = self.record.extent
        if self.all_poses.get():
            xy = np.concatenate([t.poses[:, :2] for t in self.record.tracks])
            x0, y0 = np.minimum([x0, y0], xy.min(axis=0))
            x1, y1 = np.maximum([x1, y1], xy.max(axis=0))
        self.ax.set_xlim(x0 - 0.2, x1 + 0.2)
        self.ax.set_ylim(y0 - 0.2, y1 + 0.2)
        self.canvas.draw_idle()

    def set_frame(self, frame, visible, trails=True):
        self.frame, self.visible = frame, visible
        if self.record.metadata.get("live_map"):
            self._draw_map()
        for track in self.record.tracks:
            # Only rendering is thinned; current point and metrics use the exact frame.
            ids = np.unique(np.r_[np.arange(0, frame + 1, 4), frame])
            chain = track.poses[ids]
            self.lines[track.key].set_data(chain[:, 0], chain[:, 1])
            self.lines[track.key].set_visible(track.key in visible and trails)
            self.markers[track.key].set_data([track.poses[frame, 0]], [track.poses[frame, 1]])
            self.markers[track.key].set_visible(track.key in visible)
        self.scan_points.set_offsets(
            self.record.scan_world(frame, self.focus_key) if self.show_lidar else np.empty((0, 2))
        )
        info = self.record.goal_info(frame, self.focus_key)
        self.goal_marker.set_data(
            [info[1][0]], [info[1][1]]
        ) if info else self.goal_marker.set_data([], [])
        self.goal_text.set_text(info[0] if info else "旧记录未保存独立规划目标")
        if self.focus_key in self.record.plans:
            p = self.record.plans[self.focus_key][frame]
            self.plan_line.set_data(p[:, 0], p[:, 1])
        else:
            self.plan_line.set_data([], [])
        self.canvas.draw_idle()

    def set_view(self, key, lidar=True):
        self.focus_key, self.show_lidar = key, lidar

    def _zoom(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        for getter, setter, center in (
            (self.ax.get_xlim, self.ax.set_xlim, event.xdata),
            (self.ax.get_ylim, self.ax.set_ylim, event.ydata),
        ):
            lo, hi = getter()
            setter(center + (lo - center) * scale, center + (hi - center) * scale)
        self.canvas.draw_idle()

    def _press(self, event):
        if event.inaxes == self.ax and event.dblclick:
            self.reset_limits()
        elif event.inaxes == self.ax and event.button == 1:
            self.drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

    def _drag(self, event):
        if self.drag and event.inaxes == self.ax:
            x, y, xlim, ylim = self.drag
            dx = (event.x - x) / self.ax.bbox.width * (xlim[1] - xlim[0])
            dy = (event.y - y) / self.ax.bbox.height * (ylim[1] - ylim[0])
            self.ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
            self.ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
            self.canvas.draw_idle()


class ErrorPanel(PlotPanel):
    def __init__(self, parent, seek_time):
        super().__init__(parent, "位置误差曲线", (6, 2.4))
        ttk.Label(self.bar, text="点击曲线定位时间", style="Hint.TLabel").pack(side="right")
        self.seek_time = seek_time
        self.canvas.mpl_connect("button_press_event", self._click)

    def set_record(self, record):
        self.record = record
        self.ax.clear()
        self.lines, self.markers = {}, {}
        for track in record.estimates:
            n = record.lengths.get(track.key, len(record.timestamps))
            (self.lines[track.key],) = self.ax.plot(
                record.timestamps[:n],
                record.errors[track.key][:n],
                color=track.color,
                lw=1,
                alpha=0.8,
            )
            (self.markers[track.key],) = self.ax.plot([], [], "o", color=track.color, markersize=5)
        self.cursor = self.ax.axvline(record.timestamps[0], color="#253346", lw=1.2)
        self.ax.set_xlim(record.timestamps[0], record.timestamps[-1])
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlabel("模拟时间 (s)", fontsize=9)
        self.ax.set_ylabel("位置误差 (m)", fontsize=9)
        self.ax.tick_params(labelsize=8)
        self.ax.grid(alpha=0.2)

    def set_frame(self, frame, visible, trails=True):
        self.frame, self.visible = frame, visible
        t = self.record.timestamps[frame]
        self.cursor.set_xdata([t, t])
        for key, line in self.lines.items():
            line.set_visible(key in visible)
            measured_frame = min(
                frame, self.record.lengths.get(key, len(self.record.timestamps)) - 1
            )
            self.markers[key].set_data(
                [self.record.timestamps[measured_frame]], [self.record.errors[key][measured_frame]]
            )
            self.markers[key].set_visible(key in visible)
        self.canvas.draw_idle()

    def _click(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            self.seek_time(event.xdata)


class StatsPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=6)
        ttk.Label(self, text="当前帧与全程统计", style="PanelTitle.TLabel").pack(anchor="w")
        self.pose = ttk.Label(self, text="")
        self.pose.pack(anchor="w", pady=(3, 5))
        self.task_state = ttk.Label(self, text="", style="Hint.TLabel", wraplength=650)
        self.task_state.pack(fill="x", pady=(0, 4))
        cols = ("method", "current", "yaw", "mean", "p95", "end")
        self.table = ttk.Treeview(self, columns=cols, show="headings", height=4, selectmode="none")
        for col, text in zip(
            cols,
            ("方法 [全程到点]", "当前 / m", "朝向差 / °", "均值 / m", "P95 / m", "末端 / m"),
            strict=True,
        ):
            self.table.heading(col, text=text)
            self.table.column(
                col,
                width=205 if col == "method" else 76,
                minwidth=175 if col == "method" else 60,
                anchor="w" if col == "method" else "e",
            )
        self.table.pack(fill="x")
        self.table.tag_configure("focused", background="#e2edf7")
        self.notes = []
        for _ in range(4):
            label = ttk.Label(self, text="", wraplength=650, style="Hint.TLabel")
            label.pack(anchor="w", pady=(3, 0), fill="x")
            self.notes.append(label)
        self.bind(
            "<Configure>",
            lambda e: [
                label.configure(wraplength=max(200, e.width - 16))
                for label in [*self.notes, self.task_state]
            ],
        )
        self.frame = 0
        self.focus_key = "truth"

    def set_record(self, record):
        self.record = record
        self.table.delete(*self.table.get_children())
        self.table.configure(height=max(4, min(6, len(record.estimates))))
        for track in record.estimates:
            self.table.tag_configure(track.key, foreground=track.color)
            self.table.insert("", "end", iid=track.key, tags=(track.key,))
        for label, key in zip(
            self.notes, ("settings", "map_metrics", "cohort", "sensor_note"), strict=True
        ):
            label.configure(text=record.metadata.get(key, ""))

    def set_frame(self, frame, visible, trails=True):
        self.frame = frame
        truth = self.record.track_by_key[self.record.true_key(self.focus_key)].poses[frame]
        self.pose.configure(
            text=f"所选机器人的实际位置：x={truth[0]:.2f} m  y={truth[1]:.2f} m  朝向={np.degrees(truth[2]) % 360:.1f}°"
        )
        info = self.record.method_info(self.focus_key)
        result = info.get("result")
        if result:
            reached = sum(e["success"] for e in result["events"] if e["frame"] <= frame)
            from .analysis_panel import RESULTS

            phase = RESULTS[result["status"]] if frame >= result["frames"] - 1 else "执行中"
            goal = self.record.goal_info(frame, self.focus_key)
            self.task_state.configure(
                text=f"{info['label']}：{phase} · 当前已到 {reached}/{result['target_count']} 点\n目标：{goal[0]} · 本方法记录止于 {result['elapsed_sim_s']:.1f} s"
            )
        else:
            self.task_state.configure(
                text="这些线估计的是同一次运动；本记录未保存各方法独立规划的目标。"
            )
        for track in self.record.estimates:
            s = self.record.statistics[track.key]
            ended = frame >= self.record.lengths.get(track.key, len(self.record.timestamps))
            self.table.item(
                track.key,
                tags=(track.key, "focused") if track.key == self.focus_key else (track.key,),
                values=(
                    track.label + self._result_suffix(track.key),
                    "—" if ended else f"{self.record.errors[track.key][frame]:.3f}",
                    "—" if ended else f"{self.record.yaw_errors[track.key][frame]:.2f}",
                    f"{s['mean_m']:.3f}",
                    f"{s['p95_m']:.3f}",
                    f"{s['end_m']:.3f}",
                ),
            )

    def _result_suffix(self, key):
        result = self.record.method_info(key).get("result")
        if result and "passed" in result:
            return f" [{result['reached']}/1·{'通过' if result['passed'] else '未过'}]"
        if result and "safe_completed" in result and result["reached"]:
            return f" [1/1·余量{'合格' if result['safe_completed'] else '未过'}]"
        return f" [{result['reached']}/{result['target_count']}]" if result else ""

    def set_view(self, key, lidar=True):
        self.focus_key = key
