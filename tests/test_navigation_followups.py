"""Known-answer and paired-record checks for isolated navigation mechanisms."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_learning.experiments.campus_patrol import Config, run_episode, world
from embodied_learning.experiments.grid_nav import inflate
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_decisions import (
    NavigationDecision,
    particle_radius,
    scan_supports_motion,
)
from embodied_learning.navigation_geometry import NavigationRig


def study_source():
    from embodied_learning.experiments.navigation_followups import DIRECTORIES
    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    paths = {k: Path("results") / v for k, v in DIRECTORIES.items()}
    if not all((p / "summary.json").exists() for p in paths.values()):
        pytest.skip("formal followup archives generated separately")
    return NavigationStudySource(followups=paths)


def test_rectangle_orientation_and_corner_sweep_are_distinct_from_center_point():
    rig = NavigationRig()
    prior = np.zeros((120, 120), bool)
    field = FootprintField(prior, [[6.29, 6.30]], rig)
    assert field.free([6, 6, 0]) and field.free([6, 6, np.pi / 2])
    assert not field.turn([6, 6], 0, np.pi / 2)
    assert not field.free([0.2, 6, 0])  # rear part exits map
    assert not field.segment([5, 6.3], [7, 6.3])  # endpoints alone are insufficient


def test_sixty_cm_gap_accepts_aligned_rectangle_but_not_circle():
    prior = np.zeros((120, 120), bool)
    prior[:50] = True
    prior[56:] = True
    rectangle = FootprintField(prior, [], NavigationRig())
    circle = FootprintField(prior, [], NavigationRig(), "circle")
    assert rectangle.free([6, 5.3, 0])
    assert not rectangle.free([6, 5.3, np.pi / 4])
    assert not circle.free([6, 5.3, 0])


def test_internal_particle_radius_is_not_true_error_oracle():
    cloud = np.tile([1.0, 1.0, 0.0], (10, 1))
    assert particle_radius(cloud, np.ones(10) / 10, [1, 1, 0]) == 0
    policy = NavigationDecision("joint", np.zeros((120, 120), bool))
    pf = SimpleNamespace(p=cloud, w=np.ones(10) / 10)
    moving, _ = policy.arrival(
        np.array([1.0, 1.0, 0.0]), pf, np.array([1.1, 1.0]), [0.1, 0], Config()
    )
    stopped, limit = policy.arrival(
        np.array([1.0, 1.0, 0.0]), pf, np.array([1.1, 1.0]), [0, 0], Config()
    )
    assert not moving and stopped and limit <= 0.30
    # A concentrated but biased estimate can still pass. No truth is supplied.
    assert policy.info["arrival_budget"] == pytest.approx(0.1)


def test_recovery_requires_whole_disc_support_and_forbids_crossing_wall():
    assert scan_supports_motion(np.full(64, 2.0), np.array([[0, 0, 0], [0.3, 0, 0]]))
    assert not scan_supports_motion(np.full(64, 0.4), np.array([[0, 0, 0], [0.3, 0, 0]]))
    occupied = np.zeros((120, 120), bool)
    occupied[:, 50] = True
    safe = np.zeros_like(occupied)
    safe[:, 52:] = True
    policy = NavigationDecision("recovery", occupied)
    assert (
        policy.recover(
            np.array([4.9, 6, 0]),
            np.array([8.0, 6.0]),
            np.full(64, 8.0),
            np.ones(64, bool),
            safe,
            0,
        )
        is None
    )
    assert policy.info["reason"] == "no_observed_safe_escape"
    # Failure elsewhere in the route must not activate local start repair.
    safe[60, 49] = True
    assert (
        policy.recover(
            np.array([4.9, 6, 0]),
            np.array([8.0, 6.0]),
            np.full(64, 8.0),
            np.ones(64, bool),
            safe,
            1,
        )
        is None
    )
    assert policy.info["reason"] == "global_route_blocked"


def test_recovery_budget_and_uncertain_localization_refuse_motion():
    prior = np.zeros((120, 120), bool)
    policy = NavigationDecision("recovery", prior)
    safe = np.zeros_like(prior)
    policy.info["arrival_r95"] = 0.5
    args = (
        np.array([5.0, 5.0, 0.0]),
        np.array([8.0, 5.0]),
        np.full(64, 8.0),
        np.ones(64, bool),
        safe,
    )
    assert policy.recover(*args, 0) is None
    assert policy.info["reason"] == "localization_ambiguous"
    policy.info["arrival_r95"] = 0.1
    policy.recover_start = 0
    policy.recovery_events = [{}]
    assert policy.recover(*args, 40) is None
    assert policy.info["reason"] == "recovery_budget_exhausted"


def test_baseline_hook_keeps_saved_lesson68_trajectory_bitwise():
    root = Path("results/map_repair_v1")
    if not root.exists():
        pytest.skip("formal archives generated separately")
    grid = np.load(root / "maps.npz")["repaired"]
    geometry, _, _ = world()
    row, a = run_episode(
        "self", 1, geometry, ~inflate(grid, 4), grid, Config(), NavigationDecision("baseline", grid)
    )
    with np.load(root / "seed1_repaired.npz") as old:
        for key in old.files:
            assert np.array_equal(a[key], old[key], equal_nan=True), key
    assert row["status"] == "false_completion" and row["reached"] == 7


def test_arrival_and_recovery_execute_motion_without_relaxing_truth_threshold():
    root = Path("results/map_repair_v1")
    if not root.exists():
        pytest.skip("formal archives generated separately")
    geometry, _, _ = world()
    with np.load(root / "maps.npz") as maps:
        for key, seed, mode in (("repaired", 1, "joint"), ("reference", 0, "recovery")):
            grid = maps[key]
            row, a = run_episode(
                "self",
                seed,
                geometry,
                ~inflate(grid, 4),
                grid,
                Config(),
                NavigationDecision(mode, grid),
            )
            assert row["reached"] == 8 and row["contacts"] == 0
            assert max(e["true_distance_m"] for e in row["events"]) <= 0.45
            from embodied_learning.experiments.campus_patrol import DT, advance

            for i in range(len(a["truth"]) - 1):
                assert np.allclose(
                    advance(a["truth"][i], *(a["commands"][i] * DT)), a["truth"][i + 1]
                )
            if mode == "joint":
                assert all(
                    np.max(np.abs(a["commands"][e["frame"] - 1])) <= 0.05 for e in row["events"]
                )
            else:
                assert row["recovery_active_frames"] > 0
                assert row["recoveries"][0]["outcome"] == "replanned_after_real_motion"


def test_formal_records_preserve_baselines_and_source_hashes():
    import hashlib
    import json

    source = study_source()
    for lesson in ("69", "70", "71"):
        summary = source.summaries[lesson]
        for name, value in summary["protocol"]["source_sha256"].items():
            archived = source.directories[lesson] / "source" / name
            assert hashlib.sha256(archived.read_bytes()).hexdigest() == value
        for entry in (e for e in source.entries if e.variant == lesson):
            record = source.load(entry.case, lesson)
            assert record.telemetry
            assert all(record.lengths[k] >= 1 for k in record.method_keys)
    old = json.loads(Path("results/observed_navigation_v1/summary.json").read_text(encoding="utf8"))
    for row in source.summaries["69"]["rows"]:
        if row["method"] != "circle_grid":
            continue
        previous = next(
            r
            for r in old["rows"]
            if r["method"] == "depth"
            and all(r[k] == row[k] for k in ("layout", "obstacle", "seed"))
        )
        with (
            np.load(source.directories["69"] / row["file"]) as new,
            np.load(Path("results/observed_navigation_v1") / previous["file"]) as original,
        ):
            for key in original.files:
                if original[key].dtype.kind in "US":
                    assert np.array_equal(new[key], original[key]), key
                else:
                    assert np.array_equal(new[key], original[key], equal_nan=True), key


@pytest.mark.isolated_tk
def test_followup_buttons_switch_camera_scene_diagnostics_and_goal_together():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    source = study_source()
    root = tk.Tk()
    app = ReplayWindow(root, source, "70_repaired_1", "70")
    try:
        root.update()
        for case, lesson, method, frame in (
            ("70_repaired_1", "70", "joint", 447),
            ("71_reference_0", "71", "recovery", 285),
            ("69_narrow_crate_0", "69", "rectangle", 50),
        ):
            app.select_record(case, lesson)
            app.seek(frame)
            app.analysis.tabs.select(3)
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == frame
            assert (
                app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == method
            )
            assert app.analysis.safety.key == method
            assert len(app.analysis.safety.figure.axes) >= 2
            assert app.record.goal_info(frame, method) is not None
            assert app.camera.image.size == (640, 480)
            assert (
                "控制输入" in app.analysis.safety.figure.axes[1].get_title()
                if lesson == "69"
                else "粒子半径" in app.analysis.safety.note.cget("text")
            )
    finally:
        app.close()
