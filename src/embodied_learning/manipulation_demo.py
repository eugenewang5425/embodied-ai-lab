"""Open the reusable SO101 camera/3D/diagnostics window."""

import argparse
import tkinter as tk
from pathlib import Path

from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.manipulation_viewer.window import ManipulationWindow


def main():
    results_root = Path(__file__).resolve().parents[2] / "results"
    latest = results_root / "so101_approach_v3b"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=latest
        if (latest / "summary.json").exists()
        else results_root / "so101_grasping_v2",
    )
    parser.add_argument("--position")
    parser.add_argument("--method", help="使用该记录登记的方法名；默认选择最终候选")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    if not (args.results / "summary.json").exists():
        parser.error("先运行 experiments.so101_grasping 生成记录，或用--results指定已有目录")
    record = ManipulationRecord(args.results)
    root = tk.Tk()
    app = ManipulationWindow(root, record, args.position, args.method)
    if args.play:
        root.after(500, app.toggle)
    root.lift()
    root.mainloop()


if __name__ == "__main__":
    main()
