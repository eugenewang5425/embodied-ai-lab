"""Capture our own Tk windows and verify synchronous live map selection."""

import ctypes
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow

ROOT = Path(__file__).resolve().parents[1]


def main():
    ctypes.windll.user32.SetProcessDPIAware()
    for round_number, case, frame, tab, key, name in (
        (7, "69_narrow_crate_17__delay_4__normal", 137, 8, "normal_speed", "speed-contact"),
        (8, "69_street_crate_20__delay_2__shifted", 100, 9, "map_evidence", "map-shifted"),
        (8, "69_street_crate_20__delay_2__removed", 5, 9, "map_evidence", "map-failure"),
    ):
        source = NavigationStudySource(
            None, None, {"69": ROOT / f"results/navigation_reinforcement_v{round_number}"}
        )
        root = tk.Tk()
        root.tk.call("tk", "scaling", 1.1)
        app = ReplayWindow(root, source, case, "69")
        root.geometry("1680x1000+10+10")
        try:
            app.seek(frame)
            app.analysis.tabs.select(tab)
            for method in app.record.method_keys:
                app.method_buttons[method].invoke()
                root.update()
                assert app.clock.frame == frame
                assert (
                    app.camera.focus_key
                    == app.scene.engine.focus_key
                    == app.stats.focus_key
                    == method
                )
                assert app.analysis.tabs.index(app.analysis.tabs.select()) == tab
                if round_number == 8:
                    np.testing.assert_array_equal(
                        app.map.raster.get_array(),
                        app.record.telemetry[method]["map_frames"][frame],
                    )
            app.method_buttons[key].invoke()
            root.update()
            panel = app.analysis.contact if round_number == 7 else app.analysis.map_update
            panel.canvas.draw()
            root.update()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
            target = ROOT / f"docs/img/lesson69-round{round_number}-window-{name}.png"
            ImageGrab.grab(window=hwnd).save(target)
            print(target)
        finally:
            app.close()


if __name__ == "__main__":
    main()
