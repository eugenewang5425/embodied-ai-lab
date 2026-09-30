"""Render a real Tk window at successful and failed phases for visual inspection."""

import argparse
import ctypes
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.manipulation_viewer.window import ManipulationWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/so101_grasping_v2"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/so101_grasping_v2/_window_check")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    app = ManipulationWindow(root, ManipulationRecord(args.results), "p04")
    root.geometry("1440x940+25+25")
    captures = [
        ("p04", "clamp", 525, "lift"),
        ("p04", "open", 525, "open"),
        ("p04", "offset", 525, "offset"),
        ("p02", "clamp", 380, "failure"),
        ("p04", "clamp", 1150, "placed"),
    ]
    try:
        for position, method, frame, name in captures:
            app.position, app.method = position, method
            app.load_selection()
            app.method_var.set(method)
            app.seek(frame)
            root.lift()
            root.update()
            root.after(200, root.quit)
            root.mainloop()
            # Capture our own HWND, even when a foreground app covers the desktop.
            # Never publish a desktop crop that might show unrelated user content.
            hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
            ImageGrab.grab(window=hwnd).save(args.output / f"{name}.png")
        root.geometry("1100x800+25+25")
        root.update()
        app.refresh()
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        ImageGrab.grab(window=hwnd).save(args.output / "compact.png")
    finally:
        app.close()
    print(f"Saved real window screenshots: {args.output}")


if __name__ == "__main__":
    main()
