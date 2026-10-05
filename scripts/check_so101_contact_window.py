"""Capture this experiment's own window and check six methods on one clock."""

import argparse
import ctypes
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import ImageGrab

from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.manipulation_viewer.window import ManipulationWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/so101_contact_v4"))
    parser.add_argument("--output", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    ctypes.windll.user32.SetProcessDPIAware()
    record = ManipulationRecord(args.results)
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.1)
    app = ManipulationWindow(root, record, "p04a1")
    root.geometry("1600x1000+10+10")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        root.update()
        app.seek(350)
        for method in record.available_methods("p04a1"):
            app.method_buttons[method].invoke()
            root.update()
            for panel in (app.scene, app.overhead, app.wrist, app.diagnostics):
                assert panel.method == method and panel.frame == 350
            assert np.std(np.asarray(app.wrist.image)) > 15
        captures = (
            ("p04a1", "surface", 525, "lift"),
            ("p00a2", "surface", 525, "drop"),
            ("p02a0", "surface_gate", 400, "refusal"),
        )
        for position, method, frame, name in captures:
            app.position = position
            app.select_method(method)
            app.seek(frame)
            root.update()
            root.after(200, root.quit)
            root.mainloop()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
            ImageGrab.grab(window=hwnd).save(args.output / f"lesson73-round3-window-{name}.png")
        # Switching a 21.2s replay to an 8s refusal must clamp every panel.
        app.position = "p02a0"
        app.select_method("surface")
        app.seek(525)
        app.method_buttons["surface_gate"].invoke()
        assert app.clock.frame == 400
        assert all(p.frame == 400 for p in (app.scene, app.overhead, app.wrist, app.diagnostics))
        app.position = "p00a2"
        app.select_method("surface")
        assert str(app.method_buttons["open"].cget("state")) == "disabled"
        root.geometry("1100x800+10+10")
        root.update()
        app.refresh()
        root.update()
        ImageGrab.grab(window=ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)).save(
            args.output / "lesson73-round3-window-compact.png"
        )
    finally:
        app.close()
    print("Six methods, short refusal and all four panels synchronized; own-window captures saved")


if __name__ == "__main__":
    main()
