"""Independent image and diagnostics panels, controlled by one window timeline."""

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageOps, ImageTk


class ImagePanel(ttk.Frame):
    def __init__(self, parent, title, note):
        super().__init__(parent, padding=5)
        ttk.Label(self, text=title, font=("Microsoft YaHei", 12, "bold")).pack(anchor="w")
        self.note = ttk.Label(self, text=note, foreground="#526174", wraplength=500)
        self.note.pack(fill="x", pady=(2, 5))
        self.canvas = tk.Canvas(
            self, background="#17212b", highlightthickness=0, width=400, height=240
        )
        self.canvas.pack(fill="both", expand=True)
        self.image_id = self.canvas.create_image(0, 0, anchor="center")
        self.image = self.photo = None
        self.canvas.bind("<Configure>", self.resize)

    def resize(self, event=None):
        if event is not None:
            self.note.configure(wraplength=max(180, event.width - 10))
        if self.image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if min(width, height) < 2:
            return
        resized = ImageOps.contain(self.image, (width, height))
        self.photo = ImageTk.PhotoImage(resized, master=self.canvas)
        self.canvas.itemconfigure(self.image_id, image=self.photo)
        self.canvas.coords(self.image_id, width / 2, height / 2)

    def present(self, image, frame, method):
        self.image, self.frame, self.method = image, frame, method
        self.resize()


class DiagnosticsPanel(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=5)
        ttk.Label(
            self, text="实验诊断 · 同时看成功与失败", font=("Microsoft YaHei", 12, "bold")
        ).pack(anchor="w")
        self.note = ttk.Label(
            self,
            text="各线来自独立配对回合；竖线为当前时刻。接触力是固定/活动指法向力之和。",
            wraplength=600,
            foreground="#526174",
        )
        self.note.pack(fill="x")
        self.figure, self.axes = plt.subplots(2, 1, figsize=(6, 4.5), layout="constrained")
        self.widget = FigureCanvasTkAgg(self.figure, master=self)
        self.widget.get_tk_widget().pack(fill="both", expand=True)

    def load(self, record, position, method):
        self.position, self.method = position, method
        for axis in self.axes:
            axis.clear()
        for key in record.available_methods(position):
            _, values = record.select(position, key)
            baseline = values["object_xyz"][np.flatnonzero(values["phase"] == 0)[-1], 2]
            height = values["object_xyz"][:, 2] - baseline
            self.axes[0].plot(
                values["timestamps"],
                height * 100,
                color=record.colors[key],
                label=record.methods[key].split("：")[0],
                lw=2.5 if key == method else 1.2,
            )
        self.axes[0].axhline(10, color="#475569", ls="--")
        self.axes[0].set(ylabel="抬升 / cm", ylim=(-1, 16))
        self.axes[0].legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=2,
            fontsize=8,
            frameon=False,
            borderaxespad=0,
        )
        _, values = record.select(position, method)
        solved = record.summary.get("study_round") == 3
        self.note.configure(
            text=(
                "各线为独立配对回合；竖线为当前时刻。下图分别为两指过去20ms的最小求解力。"
                if solved
                else "各线来自独立配对回合；竖线为当前时刻。下图分别为固定/活动指的诊断力。"
            )
        )
        force_values = (
            np.vstack((np.zeros(2), values["physics_jaw_n"][:, :2].reshape(-1, 10, 2).min(axis=1)))
            if solved
            else values["jaw_normal_n"][:, :2]
        )
        for j, label, color in ((0, "固定指", "#2563eb"), (1, "活动指", "#c2410c")):
            self.axes[1].plot(values["timestamps"], force_values[:, j], label=label, color=color)
        self.axes[1].set(
            xlabel="仿真时间 / s", ylabel="过去20ms最小力 / N" if solved else "接触力 / N"
        )
        self.axes[1].legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=2,
            fontsize=8,
            frameon=False,
            borderaxespad=0,
        )
        self.cursors = [axis.axvline(0, color="#111827", lw=1.2) for axis in self.axes]
        for axis in self.axes:
            axis.spines[["top", "right"]].set_visible(False)
        self.widget.draw()

    def set_frame(self, frame, timestamps):
        self.frame = frame
        for cursor in self.cursors:
            cursor.set_xdata([timestamps[frame]] * 2)
        self.widget.draw_idle()

    def close(self):
        plt.close(self.figure)
