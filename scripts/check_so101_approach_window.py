"""Capture only this experiment's own HWND and verify shared selection/time."""

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
    parser.add_argument("--results", type=Path, default=Path("results/so101_approach_v3b"))
    parser.add_argument("--output", type=Path, default=Path("docs/img"))
    args = parser.parse_args()
    ctypes.windll.user32.SetProcessDPIAware()
    record = ManipulationRecord(args.results)
    root = tk.Tk()
    root.tk.call("tk", "scaling", 1.1)
    app = ManipulationWindow(root, record, "p04a1")
    root.geometry("1600x1000+10+10")
    args.output.mkdir(parents=True, exist_ok=True)
    captures = (
        ("p04a1", "center_cartesian", 525, "lift"),
        ("p00a2", "center_cartesian", 525, "failure"),
        ("p04a1", "center_cartesian", 1040, "placed"),
    )
    try:
        root.update()
        app.seek(525)
        for method in record.available_methods("p04a1"):
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == 525
            assert (
                app.scene.method
                == app.overhead.method
                == app.wrist.method
                == app.diagnostics.method
                == method
            )
            assert (
                app.scene.frame
                == app.overhead.frame
                == app.wrist.frame
                == app.diagnostics.frame
                == 525
            )
            assert np.std(np.asarray(app.wrist.image)) > 15
        for position, method, frame, name in captures:
            app.position, app.method = position, method
            app.load_selection()
            app.method_var.set(method)
            app.seek(frame)
            root.update()
            root.after(200, root.quit)
            root.mainloop()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
            ImageGrab.grab(window=hwnd).save(args.output / f"lesson73-round2-window-{name}.png")
        assert (
            str(app.method_buttons["open"].cget("state")) == "disabled"
            if app.position == "p00a2"
            else True
        )
        root.geometry("1100x800+10+10")
        root.update()
        app.refresh()
        root.update()
        ImageGrab.grab(window=ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)).save(
            args.output / "lesson73-round2-window-compact.png"
        )
    finally:
        app.close()
    print("Window camera/scene/diagnostics synchronized; captures saved")


if __name__ == "__main__":
    main()
