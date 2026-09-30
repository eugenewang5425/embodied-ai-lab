"""Verify synchronous four-arm control and capture this app's own Tk window."""

import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    directory = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(None, None, {"69": directory / "results/execution_safety_v5"})
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_crate_11__delay_4__blackout", "69")
    root.geometry("1680x1060+20+20")
    try:
        app.seek(51)
        app.analysis.tabs.select(7)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 51 and app.analysis.execution.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        app.method_buttons["queue_guard"].invoke()
        root.update()
        app.analysis.execution.canvas.draw()
        root.update()
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        file = directory / "docs/img/lesson69-round5-window.png"
        ImageGrab.grab(window=hwnd).save(file)
        print(file)
    finally:
        app.close()


if __name__ == "__main__":
    main()
