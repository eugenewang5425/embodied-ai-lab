"""Open the reusable lesson 67–71 navigation comparison window."""

import argparse
import tkinter as tk
from pathlib import Path

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", choices=("67", "68", "69", "70", "71"), default="67")
    result_root = Path(__file__).resolve().parents[2] / "results"
    footprint = next(
        (
            name
            for name in (
                "contact_navigation_v6",
                "execution_safety_v5",
                "navigation_robustness_v4",
                "braking_navigation_v3",
                "footprint_tracking_v2",
                "footprint_navigation_v1",
            )
            if (result_root / name / "summary.json").exists()
        ),
        "footprint_navigation_v1",
    )
    for flag, directory in (
        (
            "footprint",
            footprint,
        ),
        ("arrival", "arrival_decisions_v2"),
        ("recovery", "bounded_recovery_v2"),
    ):
        parser.add_argument(
            "--" + flag,
            type=Path,
            default=Path(__file__).resolve().parents[2] / "results" / directory,
        )
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
    newer = {"69": args.footprint, "70": args.arrival, "71": args.recovery}
    followups = {k: v for k, v in newer.items() if (v / "summary.json").exists()}
    available = {
        **followups,
        **{
            k: v
            for k, v in {"67": args.avoidance, "68": args.repair}.items()
            if (v / "summary.json").exists()
        },
    }
    if args.lesson not in available:
        parser.error(
            f"先生成第{args.lesson}课记录；运行 experiments.navigation_followups --lesson {args.lesson}（67/68见各课复现步骤）。"
        )
    source = NavigationStudySource(available.get("67"), available.get("68"), followups)
    default_case = "69_narrow_crate_0"
    if source.summaries.get("69", {}).get("protocol", {}).get("round") == 4:
        candidates = [e.case for e in source.entries if e.variant == "69"]
        default_case = next(
            (c for c in candidates if c == "69_narrow_crate_8__delay"), candidates[0]
        )
    if source.summaries.get("69", {}).get("protocol", {}).get("round") == 5:
        candidates = [e.case for e in source.entries if e.variant == "69"]
        default_case = next(
            (c for c in candidates if c == "69_narrow_crate_11__delay_4__normal"), candidates[0]
        )
    if source.summaries.get("69", {}).get("protocol", {}).get("round") == 6:
        candidates = [e.case for e in source.entries if e.variant == "69"]
        default_case = next(
            (c for c in candidates if c == "69_narrow_low_14__delay_2__normal"), candidates[0]
        )
    root = tk.Tk()
    app = ReplayWindow(
        root,
        source,
        args.case
        or {
            "67": "67_plaza_low_0",
            "68": "68_1",
            "69": default_case,
            "70": "70_repaired_1",
            "71": "71_reference_0",
        }[args.lesson],
        args.lesson,
    )
    app.choose_view(
        {
            "67": "depth",
            "68": "repaired",
            "69": "zero_light"
            if "zero_light" in app.record.method_keys
            else "queue_guard"
            if "queue_guard" in app.record.method_keys
            else "zero"
            if "zero" in app.record.method_keys
            else "coupled"
            if "coupled" in app.record.method_keys
            else "combined"
            if "combined" in app.record.method_keys
            else "rectangle",
            "70": "joint",
            "71": "recovery",
        }[args.lesson]
    )
    if args.lesson != "68":
        app.analysis.tabs.select(
            {2: 4, 3: 5, 4: 6, 5: 7, 6: 8}.get(app.record.metadata.get("tracking_round"), 3)
        )
    if args.play:
        root.after(500, app.toggle)
    root.lift()
    root.mainloop()


if __name__ == "__main__":
    main()
