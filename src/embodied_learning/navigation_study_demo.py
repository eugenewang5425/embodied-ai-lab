"""Open the reusable lesson 67/68 sensor/body/map comparison window."""

import argparse
import tkinter as tk
from pathlib import Path

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", choices=("67", "68"), default="67")
    parser.add_argument("--case")
    parser.add_argument("--play", action="store_true")
    parser.add_argument(
        "--avoidance",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/observed_navigation_v1",
    )
    parser.add_argument(
        "--repair", type=Path, default=Path(__file__).resolve().parents[2] / "results/map_repair_v1"
    )
    args = parser.parse_args()
    root = tk.Tk()
    app = ReplayWindow(
        root,
        NavigationStudySource(args.avoidance, args.repair),
        args.case or ("67_plaza_low_0" if args.lesson == "67" else "68_1"),
        args.lesson,
    )
    app.choose_view("depth" if args.lesson == "67" else "repaired")
    if args.lesson == "67":
        app.analysis.tabs.select(3)
    if args.play:
        root.after(500, app.toggle)
    root.lift()
    root.mainloop()


if __name__ == "__main__":
    main()
