"""Capture our own Tk windows and verify synchronous live map selection."""

import argparse
import ctypes
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import ImageGrab

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.capture is None:
        for index in range(3):
            subprocess.run(
                [sys.executable, __file__, "--capture", str(index)],
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                check=True,
            )
        return
    ctypes.windll.user32.SetProcessDPIAware()
    cases = (
        (9, "69_street_crate_23__delay_2__unchanged", 200, 9, "progress_fixed", "progress"),
        (10, "69_street_low_26__delay_2__removed_low", 5, 9, "depth_dense", "cleared"),
        (10, "69_street_low_26__delay_2__present_low", 5, 9, "depth_dense", "real-low-wall"),
    )
    for round_number, case, frame, tab, key, name in (cases[args.capture],):
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
                if round_number in (9, 10):
                    np.testing.assert_array_equal(
                        app.map.raster.get_array(),
                        app.record.telemetry[method]["map_frames"][frame],
                    )
            app.method_buttons[key].invoke()
            root.update()
            panel = app.analysis.map_update
            app.analysis.safety.draw(app.record, key, frame)
            app.analysis.safety.canvas.draw()
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
