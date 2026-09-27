"""First-person presentation camera on the same archived truth frame."""

import tkinter as tk
from tkinter import ttk

from PIL import ImageOps, ImageTk

from .body_overlay import draw_body_guides
from .goal_overlay import draw_goal
from .scene import SceneRenderer


class CameraPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=6)
        ttk.Label(self, text="机器人相机 · 第一视角", style="PanelTitle.TLabel").pack(anchor="w")
        self.note = ttk.Label(
            self, text="按实际位置重渲染；非原始采集图像", style="Hint.TLabel", wraplength=290
        )
        self.note.pack(fill="x")
        self.canvas = tk.Canvas(self, bg="#18212d", highlightthickness=0, width=320, height=240)
        self.info = ttk.Label(self, text="", style="Hint.TLabel", wraplength=290)
        # Reserve the footer before the expanding image takes the available space.
        self.info.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        self.engine = SceneRenderer()
        self.image_id = self.canvas.create_image(0, 0, anchor="center")
        self.photo = None
        self.image = None
        self.frame = 0
        self.record = None
        self.focus_key = "truth"
        self.canvas.bind("<Configure>", self._resize)

    def set_record(self, record):
        self.record = record
        self.image = None
        self.engine.load(record)

    def set_frame(self, frame, visible, trails=True):
        self.frame = frame
        if self.record is None:
            return
        # Fixed 4:3 intrinsics; resizing the panel only changes letterboxing.
        self.engine.focus_key = self.focus_key
        self.image = draw_goal(
            self.engine.render(frame, (640, 480), set(), False, view="robot"),
            self.record,
            frame,
            self.focus_key,
        )
        self.image = draw_body_guides(self.image, self.record, frame, self.focus_key)
        camera = self.record.camera
        last = self.record.lengths.get(self.focus_key, len(self.record.timestamps)) - 1
        timing = f"t = {self.record.timestamps[min(frame, last)]:.2f} s"
        if frame > last:
            timing += "（已结束，保留末帧）"
        goal = self.record.goal_info(frame, self.focus_key)
        detail = (
            f"目标：{goal[0]}（金色导航提示）"
            if goal
            else f"示意机位：高 {camera.height_m:.2f} m · 垂直视场 {camera.vertical_fov_deg:g}°"
        )
        self.info.configure(
            text=f"{timing} · {self.record.method_info(self.focus_key)['label']}\n" + detail
        )
        self._present()

    def set_view(self, key, lidar=True):
        self.focus_key = key

    def _present(self):
        if self.image is None or not self.canvas.winfo_ismapped():
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        image = ImageOps.contain(self.image, (width, height))
        self.photo = ImageTk.PhotoImage(image, master=self.canvas)
        self.canvas.coords(self.image_id, width / 2, height / 2)
        self.canvas.itemconfigure(self.image_id, image=self.photo)

    def _resize(self, event):
        self.note.configure(wraplength=max(120, event.width))
        self.info.configure(wraplength=max(120, event.width))
        self._present()

    def close(self):
        self.engine.close()
