"""Shared 2D replay machinery for the lesson-29..33 demos.

Each demo needs a ④ 'best-process replay' mode that shows pole+cart animation.
This module provides :class:`Replay2D` — a small object that owns a Matplotlib
Figure, a per-axis state stream, Tk after-loop play/pause/step bindings, and a
fig.clear() reset. The host demo only supplies which two traces to display and
a function to draw per-frame artists.
"""
from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D


class Replay2D:
    """Stateful 2D cart+pole replay controller.

    Parameters
    ----------
    root
        Tk root (already bound to <Escape> by the host demo).
    fig
        Matplotlib Figure (host already claims it via FigureCanvasTkAgg).
    on_close
        Callable to chain into host's close() to cancel the Tk after-loop.
    """

    def __init__(self, root, fig, on_close=None):
        self.root = root
        self.fig = fig
        self._state = {"running": False, "t0": 0.0, "i": 0, "after_id": None}
        # per-axis payload: states array + (color, label) tuple
        self._axes_payload = []
        # Bind keys (idempotent: replace any previous binding on these).
        root.bind("<space>", lambda _e: self.toggle())
        root.bind("<Right>", lambda _e: self.step())
        self._on_close = on_close

    def setup_axes(self, payload):
        """Define the per-axis state stream and metadata.

        ``payload`` is a list of (ax, states, color, label) tuples. ``states`` is
        an array of (x, alpha, _, _) per step. ``label`` is the subplot title.
        """
        self.fig.clear()
        n_axes = len(payload)
        cols = 1 if n_axes == 1 else 2
        rows = (n_axes + cols - 1) // cols
        if n_axes == 1:
            axes = [self.fig.subplots()]
        else:
            axes = self.fig.subplots(rows, cols).reshape(-1)
        for ax, (_, states, color, label) in zip(axes, payload):
            pass
        # Re-layout: we always redo subplots so layout is consistent
        self.fig.clear()
        if n_axes == 1:
            axes = [self.fig.subplots()]
        else:
            axes = self.fig.subplots(rows, cols).reshape(-1)
        self._axes_payload = []
        for ax, (_, states, color, label) in zip(axes, payload):
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-0.55, 0.55)
            ax.axhline(0, color="#475569", lw=1)
            ax.axvline(-2.4, color="#dc2626", ls=":", lw=0.8)
            ax.axvline(2.4, color="#dc2626", ls=":", lw=0.8)
            ax.set_xlabel("车位 x (m)")
            ax.set_ylabel("摆角 α (rad, 上=0)")
            ax.set_aspect("equal")
            ax.set_title(label, fontsize=9)
            tip_x = np.asarray([s[0] for s in states]) + 0.5 * np.sin(np.asarray([s[1] for s in states]))
            tip_y = 0.5 * np.cos(np.asarray([s[1] for s in states]))
            ax.plot(tip_x, tip_y, color=color, lw=0.6, alpha=0.25)
            self._axes_payload.append((ax, states, color, label))
        self._state["i"] = 0

    def draw_frame(self, i):
        """Redraw each axis at step i (cart + rotated pole + time cursor)."""
        for ax, states, color, _ in self._axes_payload:
            i_eff = min(i, len(states) - 1)
            x, alpha, _, _ = states[i_eff]
            for coll in list(ax.collections):
                coll.remove()
            for patch in list(ax.patches):
                patch.remove()
            ax.add_patch(plt.Rectangle((x - 0.1, -0.05), 0.2, 0.1, color=color, alpha=0.7))
            t = Affine2D().rotate_deg_around(x, 0, math.degrees(alpha)) + ax.transData
            ax.add_patch(plt.Rectangle((x - 0.02, 0), 0.04, 0.5, color=color, transform=t, alpha=0.9))
            ax.axvline(x, color="#0f172a", lw=0.8, alpha=0.5)

    def set_step(self, i):
        self._state["i"] = i
        self.draw_frame(i)
        self.fig.suptitle(
            f"④ 最佳过程回放 — 时间步 {i}    （Space 播放/暂停，→ 单步，Esc 退出）",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def max_i(self):
        return max((len(s) - 1) for _, s, _, _ in self._axes_payload)

    def toggle(self):
        self._state["running"] = not self._state["running"]
        self._state["t0"] = time.perf_counter()
        if self._state["running"]:
            self._tick()

    def step(self):
        self._state["running"] = False
        self.set_step(min(self._state["i"] + 1, self.max_i()))

    def _tick(self):
        if not self._state["running"]:
            return
        now = time.perf_counter()
        i_next = min(self._state["i"] + max(1, int((now - self._state["t0"]) / 0.04)), self.max_i())
        self._state["t0"] = now
        self.set_step(i_next)
        if self._state["i"] >= self.max_i():
            self._state["running"] = False
        if self._state["running"]:
            self._state["after_id"] = self.root.after(20, self._tick)

    def cancel(self):
        """Call from the host's close() to stop the after-loop."""
        if self._state.get("after_id"):
            try:
                self.root.after_cancel(self._state["after_id"])
            except Exception:  # noqa: BLE001, S110
                pass
            self._state["after_id"] = None
