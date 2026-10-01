"""Verify synchronized selection and capture only our own Tk window."""

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
CASES = (("low", "bypass_low", 180), ("beam", "bypass_beam", 180), ("beam", "clear_overhead", 180))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.capture is None:
        for i in range(3):
            subprocess.run(
                [sys.executable, __file__, "--capture", str(i)],
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                check=True,
            )
        return
    ctypes.windll.user32.SetProcessDPIAware()
    obstacle, profile, frame = CASES[args.capture]
    source = NavigationStudySource(
        None, None, {"69": ROOT / "results/navigation_reinforcement_v11"}
    )
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.1)
    app = ReplayWindow(root, source, f"69_street_{obstacle}_29__delay_2__{profile}", "69")
    root.geometry("1680x1000+10+10")
    try:
        app.seek(frame)
        app.analysis.tabs.select(9)
        assert app.record.metadata["depth_used_by"] == ["lidar_depth"]
        for method in app.record.method_keys:
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == frame
            assert (
                app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == method
            )
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 9
            np.testing.assert_array_equal(
                app.map.raster.get_array(), app.record.telemetry[method]["map_frames"][frame]
            )
        app.method_buttons["lidar_depth"].invoke()
        app.analysis.tabs.select(3)
        app.analysis.safety.canvas.draw()
        root.update()
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        target = ROOT / f"docs/img/lesson69-round11-window-{profile}.png"
        ImageGrab.grab(window=hwnd).save(target)
        print(target)
    finally:
        app.close()


if __name__ == "__main__":
    main()
