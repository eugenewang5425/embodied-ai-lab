"""Known geometry, safety veto and causal replay checks for lesson 69 round two."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_learning.footprint_planning import FootprintController
from embodied_learning.navigation_geometry import NavigationRig, sense
from embodied_learning.path_tracking import RetainedTangentController, TangentController, path_frame


def source_path():
    p = Path("results/footprint_tracking_v2")
    if not (p / "summary.json").exists():
        pytest.skip("round two archives are generated separately")
    return p


def test_local_frame_has_known_signed_error_tangent_and_circle_curvature():
    theta = np.linspace(0, 0.4, 101)
    path = 2 * np.c_[np.sin(theta), 1 - np.cos(theta)]
    pose = np.r_[
        2 * np.array([np.sin(0.2), 1 - np.cos(0.2)]) + 0.02 * np.array([-np.sin(0.2), np.cos(0.2)]),
        0.2,
    ]
    r = path_frame(path, pose)
    assert r["cross_m"] == pytest.approx(0.02, abs=1e-4)
    assert r["heading_rad"] == pytest.approx(0.2, abs=1e-3)
    assert r["curvature_m_inv"] == pytest.approx(0.5, abs=1e-3)


def test_tangent_feedback_steers_toward_line_without_changing_speed_rule():
    grid, rig = np.zeros((120, 120), bool), NavigationRig()
    path = np.c_[np.linspace(2, 5, 76), np.full(76, 6.0)]
    for offset, heading in ((0.03, 0), (-0.03, 0), (0, 0.4)):
        p = np.array([3.0, 6 + offset, heading])
        old, new = FootprintController(grid, rig, "depth"), TangentController(grid, rig, "depth")
        old.path, new.path = path.copy(), path.copy()
        a, b = old.tracking_command(p, np.zeros(2)), new.tracking_command(p, np.zeros(2))
        assert a[0] == b[0]
        assert np.sign(b[1]) == -np.sign(offset or heading)
        assert abs(b[1]) <= 1


def test_retention_does_not_accept_a_gap_narrower_than_padded_body():
    prior = np.zeros((120, 120), bool)
    prior[:50], prior[55:] = True, True
    c = RetainedTangentController(prior, NavigationRig(), "depth")
    c.path = np.array([[3.0, 5.25], [5.0, 5.25], [8.0, 5.25]])
    assert c.make_plan(np.array([4.0, 5.25, 0]), np.array([8.0, 5.25])) is None
    assert c.planning_log[-1]["reason"] == "footprint_start_blocked"


def test_changed_steering_still_obeys_shared_swept_body_brake():
    rig = NavigationRig()
    pose = np.array([1.5, 6.0, 0.0])
    ranges, hits, depth, valid = sense(np.array([[1.95, 2.05, 5.0, 7.0, 0.0, 1.0]]), pose, rig)
    c = RetainedTangentController(np.zeros((120, 120), bool), rig, "depth")
    # A stale route must not bypass the observation-based near-field veto.
    c.make_plan = lambda *_: np.array([[1.5, 6.0], [4.0, 6.0]])
    cmd, reason, clearance = c.step(
        pose, np.array([4.0, 6.0]), ranges, hits, depth, valid, 0, np.array([0.5, 0])
    )
    assert reason == "swept_body_brake" and clearance <= 0
    assert np.array_equal(cmd, [0, 0])


def test_final_archives_have_frozen_sources_identical_initial_plans_and_honest_end_frames():
    from embodied_learning.replay_viewer.campus import read_checked
    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    p = source_path()
    summary = json.loads((p / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 132
    for name, sha in summary["protocol"]["source_sha256"].items():
        assert hashlib.sha256((p / "source" / name).read_bytes()).hexdigest() == sha
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha
    for cohort in ("fixed", "heldout"):
        assert (
            sum(r["cohort"] == cohort for r in summary["rows"])
            == ({"fixed": 108, "heldout": 24}[cohort])
        )
    src = NavigationStudySource(None, None, {"69": p})
    record = src.load("69_narrow_crate_0", "69")
    assert record.lengths["combined"] > record.lengths["original"]
    plans = []
    for key in record.method_keys:
        row = record.method_info(key)["result"]
        a = read_checked(p / row["file"], row["sha256"])
        plans.append(a["plan"][0])
        assert record.lengths[key] == row["frames"]
        assert np.array_equal(
            record.telemetry[key]["initial_cross_m"][: row["frames"]], a["initial_cross_m"]
        )
    assert all(np.array_equal(plans[0], a, equal_nan=True) for a in plans[1:])
    assert sum(bool(r.get("baseline_equal")) for r in summary["rows"]) == 27


@pytest.mark.isolated_tk
def test_four_way_buttons_keep_tracking_camera_scene_goal_and_time_synchronized():
    import tkinter as tk

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
    from embodied_learning.replay_viewer.window import ReplayWindow

    src = NavigationStudySource(None, None, {"69": source_path()})
    root = tk.Tk()
    app = ReplayWindow(root, src, "69_narrow_crate_0", "69")
    try:
        app.seek(100)
        app.analysis.tabs.select(4)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 100
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert app.analysis.tracking.key == key and app.analysis.tracking.frame == 100
            assert app.record.goal_info(100, key) is not None
            assert len(app.analysis.tracking.figure.axes) == 2
            assert "首次规划" in app.analysis.tracking.note.cget("text")
            assert app.camera.image.size == (640, 480)
    finally:
        app.close()
