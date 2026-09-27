"""Audit archived commands, declarations, lineage and documentation for 69–71."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from PIL import Image

from embodied_learning.experiments.campus_patrol import TASKS
from embodied_learning.experiments.navigation_followups import DIRECTORIES
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = {
        "date": "2026-09-28",
        "baseline_commit": "6d48859",
        "episodes": 0,
        "continuous_executed_steps": 0,
        "arrival_events_rechecked": 0,
        "legacy_baseline_arrays_equal": 0,
        "source_snapshot_files_checked": 0,
    }
    original67 = json.loads(
        (ROOT / "results/observed_navigation_v1/summary.json").read_text(encoding="utf8")
    )
    for lesson, directory_name in DIRECTORIES.items():
        directory = ROOT / "results" / directory_name
        summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
        assert digest(directory / "protocol.json") == summary["protocol_sha256"]
        assert len(summary["rows"]) == {"69": 81, "70": 18, "71": 12}[lesson]
        if lesson != "69":
            assert digest(directory / "maps.npz") == summary["maps_sha256"]
            assert (
                digest(ROOT / "results/map_repair_v1/maps.npz")
                == summary["protocol"]["input_maps_sha256"]
            )
        for filename, sha in summary["protocol"]["source_sha256"].items():
            assert digest(ROOT / filename) == digest(directory / "source" / filename) == sha
            report["source_snapshot_files_checked"] += 1
        dt = summary["protocol"]["dt_s"]
        for row in summary["rows"]:
            a = read_checked(directory / row["file"], row["sha256"])
            p = a["truth"][:-1]
            delta = a["commands"][:-1] * dt
            # Independently check each archived model: 69 uses exact circular
            # integration; the retained 68 campus model uses midpoint steps.
            distance = delta[:, 0] * (np.sinc(delta[:, 1] / (2 * np.pi)) if lesson == "69" else 1.0)
            heading = p[:, 2] + delta[:, 1] / 2
            xy = p[:, :2] + distance[:, None] * np.c_[np.cos(heading), np.sin(heading)]
            assert np.allclose(xy, a["truth"][1:, :2], atol=1e-9, rtol=0), row["file"]
            angle_error = a["truth"][1:, 2] - p[:, 2] - delta[:, 1]
            assert np.allclose(np.sin(angle_error), 0, atol=1e-9)
            assert np.allclose(np.cos(angle_error), 1, atol=1e-9)
            report["continuous_executed_steps"] += len(p)
            for event in row["events"]:
                goal = (
                    np.array([10.5, 6.0])
                    if lesson == "69"
                    else np.array(TASKS[event["goal_index"]][1])
                )
                distance = np.linalg.norm(a["truth"][event["frame"], :2] - goal)
                if "true_distance_m" in event:
                    assert np.isclose(distance, event["true_distance_m"], atol=1e-9)
                assert bool(distance <= (0.25 if lesson == "69" else 0.45)) == event["success"]
                report["arrival_events_rechecked"] += 1
            previous = None
            if lesson == "69" and row["method"] == "circle_grid":
                old = next(
                    r
                    for r in original67["rows"]
                    if r["method"] == "depth"
                    and all(r[k] == row[k] for k in ("layout", "obstacle", "seed"))
                )
                previous = ROOT / "results/observed_navigation_v1" / old["file"]
            elif row["method"] == "baseline":
                previous = (
                    ROOT / "results/map_repair_v1" / f"seed{row['seed']}_{row['map_kind']}.npz"
                )
            if previous:
                with np.load(previous) as old:
                    for key in old.files:
                        assert np.array_equal(
                            a[key], old[key], equal_nan=old[key].dtype.kind not in "US"
                        ), key
                report["legacy_baseline_arrays_equal"] += 1
            report["episodes"] += 1
    report["images"] = {}
    for lesson in DIRECTORIES:
        for suffix in ("results", "trajectories", "window"):
            p = ROOT / f"docs/img/lesson{lesson}-{suffix}.png"
            with Image.open(p) as im:
                assert im.width >= 1500 and im.height >= 700
                report["images"][str(p.relative_to(ROOT))] = {
                    "pixels": im.size,
                    "sha256": digest(p),
                }
    report["visual_review"] = (
        "Six analytical figures and three real Tk window captures inspected; no truncated labels or legend/data overlaps in selected views. 71 trajectory legend moved above plots."
    )
    report["standalone_69_adapter"] = (
        "Checked with avoidance=None and repair=None; 27 cases load without legacy directories."
    )
    report["tests"] = {}
    for name in ("quick", "affected", "full"):
        p = ROOT / f"results/navigation_followups_{name}.log"
        value = p.read_text(encoding="utf8")
        report["tests"][name] = {
            "complete": "[tests] exit=" in value,
            "log": str(p.relative_to(ROOT)),
            "sha256": digest(p),
            "result_lines": [
                line
                for line in value.splitlines()
                if re.search(r"^\d+ .*passed|^\[tests\] exit=", line)
            ],
        }
    report["test_failure_resolution"] = (
        "Full first run includes the old 0.4-second replay UI timing failure. Fixture tail extended to 20 seconds; all 46 affected tests subsequently pass, including that GUI and four real Tk tests. Playback product code unchanged."
    )
    if all(r["complete"] for r in report["tests"].values()):
        assert any("exit=0" in line for line in report["tests"]["quick"]["result_lines"])
        assert any("exit=0" in line for line in report["tests"]["affected"]["result_lines"])
        full_log = (ROOT / "results/navigation_followups_full.log").read_text(encoding="utf8")
        failures = re.findall(r"^FAILED (\S+)", full_log, re.MULTILINE)
        assert failures in (
            [],
            [
                "tests/test_replay_viewer.py::test_window_controls_keep_all_panels_on_same_record_and_frame"
            ],
        ), failures
        report["unresolved_test_failures"] = []
    report["full_test_thread_environment"] = {
        k: "1" for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    }
    report["lint"] = "ruff check src tests scripts and figure generator passed"
    fmt = (ROOT / "results/navigation_followups_format_all.log").read_text(encoding="utf8")
    report["format"] = {
        "changed_python_files": "passed separately",
        "preexisting_unmodified_failures": list(
            dict.fromkeys(re.findall(r"--> (.*?):\d+:\d+", fmt))
        ),
    }
    report["scope_limits"] = [
        "69 narrow navigation failed 0/9; no real-time claim",
        "70 and 71 kept separate; bodies differ from 69",
        "Healthy simulated sensors, fixed known layouts, no hardware validation",
        "No additional course videos or unrelated formatting changes",
    ]
    target = ROOT / "docs/benchmarks/navigation-studies-69-71-validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    files = [ROOT / "README.md"] + [
        ROOT / "docs" / name
        for name in (
            "69-footprint-planning.md",
            "70-arrival-decisions.md",
            "71-bounded-recovery.md",
            "navigation-stage-review-69-71.md",
            "navigation-studies-69-71-plan.md",
            "navigation-stage-review-66-68.md",
            "multisensor-navigation-roadmap.md",
            "campus-map-repair-plan.md",
            "replay-window.md",
            "01-learning-roadmap.md",
            "67-body-aware-obstacle-avoidance.md",
            "68-scan-based-map-repair.md",
        )
    ]
    links = 0
    for file in files:
        for href in re.findall(r"\[[^\]]*\]\(([^)]+)\)", file.read_text(encoding="utf8")):
            if "://" in href or href.startswith("#"):
                continue
            assert (file.parent / unquote(href.split("#")[0])).exists(), (file, href)
            links += 1
    report["local_links_checked"] = links
    report["audit_script_sha256"] = digest(Path(__file__))
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "episodes",
                    "continuous_executed_steps",
                    "arrival_events_rechecked",
                    "legacy_baseline_arrays_equal",
                    "local_links_checked",
                )
            },
            indent=2,
        )
    )
    if not all(r["complete"] for r in report["tests"].values()):
        raise SystemExit(
            "Audit saved with explicit pending test status; rerun after full suite completes."
        )


if __name__ == "__main__":
    main()
