"""Verify and capture the reusable round-three window at a fixed case/time."""

import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    root_dir = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(None, None, {"69": root_dir / "results/braking_navigation_v3"})
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_crate_0", "69")
    root.geometry("1680x1060+20+20")
    try:
        app.seek(70)
        app.analysis.tabs.select(5)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 70 and app.analysis.braking.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        app.method_buttons["coupled"].invoke()
        root.update()
        app.analysis.braking.canvas.draw()
        root.update()
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        target = root_dir / "docs/img/lesson69-round3-window.png"
        ImageGrab.grab(window=hwnd).save(target)
        print(target)
    finally:
        app.close()


if __name__ == "__main__":
    main()
