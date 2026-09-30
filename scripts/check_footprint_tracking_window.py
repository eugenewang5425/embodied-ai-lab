"""Create, verify and capture only our own replay window."""

import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    directory = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(None, None, {"69": directory / "results/footprint_tracking_v2"})
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_crate_0", "69")
    root.geometry("1600x1000+30+30")
    try:
        app.seek(100)
        app.analysis.tabs.select(4)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 100 and app.analysis.tracking.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        app.analysis.tracking.canvas.draw()
        root.update()
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        target = directory / "docs/img/lesson69-round2-window.png"
        ImageGrab.grab(window=hwnd).save(target)
        print(target)
    finally:
        app.close()


if __name__ == "__main__":
    main()
