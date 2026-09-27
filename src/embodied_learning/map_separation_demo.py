"""Open the reusable 3D/map/error replay window on lesson-65 records."""

import argparse
import json
from pathlib import Path

from embodied_learning.replay_viewer.map_separation import MapSeparationSource

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hundred",
        type=Path,
        default=PROJECT_ROOT / "results/navigation_benchmark_2026-09-24_pilot_v4",
    )
    parser.add_argument(
        "--four-hundred", type=Path, default=PROJECT_ROOT / "results/nav_sep_particles400_v1"
    )
    parser.add_argument("--case", default="iso_w1_s1")
    parser.add_argument("--particles", choices=("100", "400"), default="400")
    parser.add_argument("--play", action="store_true")
    parser.add_argument(
        "--verify", action="store_true", help="validate every available recording and exit"
    )
    args = parser.parse_args()
    source = MapSeparationSource(args.hundred, args.four_hundred)
    if args.verify:
        checked = []
        for entry in source.entries:
            record = source.load(entry.case, entry.variant)
            checked.append(
                {
                    "case": entry.case,
                    "particles": entry.variant,
                    "frames": len(record.timestamps),
                    "self_mean_m": record.statistics["self"]["mean_m"],
                }
            )
        print(json.dumps({"verified": len(checked), "records": checked}, indent=2))
        return
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, source, args.case, args.particles)
    root.lift()
    root.attributes("-topmost", True)
    root.after(1200, lambda: root.attributes("-topmost", False))
    if args.play:
        root.after(500, app.toggle)
    root.mainloop()


if __name__ == "__main__":
    main()
