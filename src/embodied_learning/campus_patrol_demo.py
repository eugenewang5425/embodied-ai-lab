"""Open recorded campus patrol with first-person targets and lidar evidence."""

import argparse
import tkinter as tk
from pathlib import Path

from embodied_learning.replay_viewer.campus import CampusSource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/campus_patrol_v1",
    )
    parser.add_argument("--seed", default="0")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    root = tk.Tk()
    app = ReplayWindow(root, CampusSource(args.record), args.seed, "v1")
    root.lift()
    root.attributes("-topmost", True)
    root.after(1200, lambda: root.attributes("-topmost", False))
    if args.play:
        root.after(500, app.toggle)
    root.mainloop()


if __name__ == "__main__":
    main()
