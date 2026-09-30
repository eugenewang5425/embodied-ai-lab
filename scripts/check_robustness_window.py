"""Capture the actual reusable fourth-round window at a preregistered case/time."""

import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    root_dir = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(
        None, None, {"69": root_dir / "results/navigation_robustness_v4"}
    )
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_crate_8__delay", "69")
    root.geometry("1680x1060+20+20")
    try:
        app.seek(60)
        app.analysis.tabs.select(6)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 60 and app.analysis.robustness.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        app.method_buttons["delay_4"].invoke()
        root.update()
        app.analysis.robustness.canvas.draw()
        root.update()
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        target = root_dir / "docs/img/lesson69-round4-window.png"
        ImageGrab.grab(window=hwnd).save(target)
        print(target)
    finally:
        app.close()


if __name__ == "__main__":
    main()
