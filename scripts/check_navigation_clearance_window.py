"""Capture only the owned window and verify one-click method/time synchronization."""

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
CASES = (("crate", "bypass_crate", 225), ("beam", "bypass_beam", 220), ("low", "bypass_low", 215))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=int, choices=range(3))
    parser.add_argument("--input", type=Path, default=ROOT / "results/navigation_reinforcement_v14")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/img")
    args = parser.parse_args()
    if args.capture is None:
        for i in range(3):
            subprocess.run([sys.executable, __file__, "--capture", str(i), "--input", str(args.input),
                            "--seed", str(args.seed), "--output", str(args.output)],
                           env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}, check=True)
        return
    ctypes.windll.user32.SetProcessDPIAware()
    obstacle, profile, frame = CASES[args.capture]
    source = NavigationStudySource(None, None, {"69": args.input})
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.1)
    app = ReplayWindow(root, source, f"69_street_{obstacle}_{args.seed}__delay_2__{profile}", "69")
    root.geometry("1680x1000+10+10")
    try:
        app.seek(frame)
        app.analysis.tabs.select(9)
        for method in app.record.method_keys:
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == frame
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == method
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 9
            np.testing.assert_array_equal(app.map.raster.get_array(),
                                          app.record.telemetry[method]["map_frames"][frame])
        app.method_buttons["height_depth"].invoke()
        app.analysis.tabs.select(3)
        app.analysis.safety.canvas.draw()
        root.update()
        assert "实际三方向速度" in app.analysis.safety.note.cget("text")
        hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
        args.output.mkdir(parents=True, exist_ok=True)
        target = args.output / f"lesson69-round14-window-{profile}.png"
        ImageGrab.grab(window=hwnd).save(target)
        print(target, flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
