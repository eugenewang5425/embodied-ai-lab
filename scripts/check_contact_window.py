"""Exercise all physical-policy selectors and capture the app's own windows."""

import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    root_dir = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(None, None, {"69": root_dir / "results/contact_navigation_v6"})
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_low_14__delay_2__normal", "69")
    root.geometry("1680x1060+20+20")
    try:
        app.seek(130)
        app.analysis.tabs.select(8)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 130 and app.analysis.contact.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        for ob, frame in (("low", 130), ("crate", 85)):
            app.select_record(f"69_narrow_{ob}_14__delay_2__normal", "69")
            app.seek(frame)
            app.method_buttons["zero_light"].invoke()
            app.analysis.tabs.select(8)
            root.update()
            app.analysis.contact.canvas.draw()
            root.update()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
            target = root_dir / f"docs/img/lesson69-round6-window-{ob}.png"
            ImageGrab.grab(window=hwnd).save(target)
            print(target)
    finally:
        app.close()


if __name__ == "__main__":
    main()
