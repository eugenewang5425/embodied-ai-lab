"""Capture only the Tk window created by this development QA script."""

import ctypes
import json
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

from embodied_learning.experiments.navigation_followups import DIRECTORIES
from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
from embodied_learning.replay_viewer.window import ReplayWindow


def main():
    root_path = Path(__file__).resolve().parents[1]
    source = NavigationStudySource(
        root_path / "results/observed_navigation_v1",
        root_path / "results/map_repair_v1",
        {k: root_path / "results" / v for k, v in DIRECTORIES.items()},
    )
    root = tk.Tk()
    app = ReplayWindow(root, source, "69_narrow_crate_0", "69")
    root.geometry("1600x1000+30+30")
    evidence = []
    try:
        for case, lesson, key, frame in (
            ("69_narrow_crate_0", "69", "rectangle", 60),
            ("70_repaired_1", "70", "joint", 447),
            ("71_reference_0", "71", "recovery", 285),
        ):
            app.select_record(case, lesson)
            app.analysis.tabs.select(3)
            app.seek(frame)
            app.method_buttons[key].invoke()
            root.update()
            app.analysis.safety.canvas.draw()
            root.update()
            hwnd = ctypes.windll.user32.GetAncestor(root.winfo_id(), 2)
            path = root_path / f"docs/img/lesson{lesson}-window.png"
            ImageGrab.grab(window=hwnd).save(path)
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert app.analysis.safety.key == key and app.clock.frame == frame
            evidence.append(
                {
                    "lesson": lesson,
                    "case": case,
                    "method": key,
                    "frame": frame,
                    "image": str(path.relative_to(root_path)),
                    "size": ImageGrab.grab(window=hwnd).size,
                }
            )
    finally:
        app.close()
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
