"""Known-answer timing and biased-estimate contracts for round four."""

import numpy as np
import pytest

from embodied_learning.navigation_geometry import NavigationRig
from embodied_learning.navigation_robustness import (
    CONDITIONS,
    IntentQueue,
    Perturbation,
    biased_pose,
)


def test_pose_error_uses_world_y_and_wraps_heading_without_transforming_truth():
    truth = np.array([2.0, 3.0, np.deg2rad(179)])
    estimate = biased_pose(truth, Perturbation("test", 0.01, 4))
    np.testing.assert_allclose(estimate, [2, 3.01, np.deg2rad(-177)])
    np.testing.assert_allclose(truth, [2, 3, np.deg2rad(179)])
    np.testing.assert_array_equal(biased_pose(truth, CONDITIONS["zero"]), truth)


def test_delayed_brake_cannot_bypass_pending_motion_or_physical_limits():
    queue = IntentQueue(2, NavigationRig())
    incoming = np.zeros(2)
    executed, sources = [], []
    for frame in range(18):
        # Issue travel at k=0..3, then stop; stop at 4 reaches actuator at 6.
        u, _target, brake, source = queue.step([0.6, 0.084], frame >= 4, frame, incoming)
        assert np.all(np.abs(u - incoming) <= [0.08 + 1e-12, 0.18 + 1e-12])
        executed.append(u.copy())
        sources.append(source)
        if frame == 4:
            assert queue.pending_motion == 1 and not brake
        if frame == 6:
            assert brake and u[1] / u[0] == pytest.approx(incoming[1] / incoming[0])
        incoming = u
    np.testing.assert_array_equal(executed[:2], [[0, 0], [0, 0]])
    np.testing.assert_allclose(np.asarray(executed)[2:6, 0], [0.08, 0.16, 0.24, 0.32])
    assert sources == [-1, -1] + list(range(16))
    np.testing.assert_array_equal(incoming, [0, 0])
    assert queue.pending_motion == 0
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError):
            IntentQueue(invalid, NavigationRig())


def test_goal_decision_uses_estimate_not_truth(monkeypatch):
    from embodied_learning.experiments import navigation_robustness as experiment

    class BlindStraight:
        path_capacity = 256

        def __init__(self, prior, rig, mode):
            self.grid = prior
            self.path = np.array([[1.5, 6], [10.5, 6]])
            self.planning_log, self.tracking_log = [], []

        def step(self, estimate, goal, ranges, hits, depth, valid, frame, velocity):
            self.latest = {
                "speed_cap": 0.6,
                "preview_curvature": 0,
                "nominal": np.array([0.6, 0]),
                "stop_clearance": 1,
            }
            return np.array([0.6, 0]), "following", 1.0

    # A clear box-free world isolates the mission layer from real obstacles.
    monkeypatch.setattr(experiment, "BrakingController", BlindStraight)
    monkeypatch.setattr(experiment, "scene", lambda *_: (np.empty((0, 6)), np.empty((0, 6))))
    # y error > goal radius prevents declaration even if true path crosses goal.
    row, data = experiment.episode(
        "plaza", "crate", 0, Perturbation("test", dy_m=0.3), max_steps=180
    )
    assert not row["announced"] and row["status"] == "timeout"
    assert np.min(np.linalg.norm(data["truth"][:, :2] - [10.5, 6], axis=1)) < 0.25
    assert np.min(np.linalg.norm(data["estimate"][:, :2] - [10.5, 6], axis=1)) >= 0.3
    assert row["physically_stopped"]

    class EstimatedGoal(BlindStraight):
        def step(self, estimate, goal, *args):
            super().step(estimate, goal, *args)
            angle = np.arctan2(*(goal - estimate[:2])[::-1]) - estimate[2]
            return np.array([0.6, np.clip(2 * angle, -1, 1)]), "following", 1.0

    monkeypatch.setattr(experiment, "BrakingController", EstimatedGoal)
    row, data = experiment.episode("plaza", "crate", 0, Perturbation("test", dy_m=0.3))
    assert row["announced"] and row["false_arrival"] and not row["reached"]
    assert row["status"] == "false_completion" and row["physically_stopped"]
    assert row["final_estimated_goal_distance_m"] < 0.25 < row["final_goal_distance_m"]

    # Contact ends this kinematic model without claiming physical rest.
    monkeypatch.setattr(experiment, "swept_clearance", lambda *_: -0.001)
    row, data = experiment.episode("plaza", "crate", 0, CONDITIONS["zero"])
    assert row["contacts"] and not row["physically_stopped"] and not row["announced"]
    assert data["incoming_velocity"][-1, 0] > 0
    np.testing.assert_array_equal(data["commands"][-1], [0, 0])


def test_registration_case_counts_and_no_joint_perturbations():
    from collections import Counter

    from embodied_learning.experiments.navigation_robustness import registered_cases

    cases = registered_cases()
    assert Counter(c[3] for c in cases) == {
        "bridge": 27,
        "scan": 189,
        "environment": 12,
        "negative": 5,
    }
    assert len({(a, b, c, e) for a, b, c, _, e in cases}) == 233
    for c in CONDITIONS.values():
        assert sum(bool(v) for v in (c.dy_m, c.yaw_deg, c.delay_steps)) <= 1


def source_or_skip():
    import json
    from pathlib import Path

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    directory = Path("results/navigation_robustness_v4")
    path = directory / "summary.json"
    if not path.exists() or len(json.loads(path.read_text(encoding="utf8"))["rows"]) != 233:
        pytest.skip("generate the frozen 233-case batch separately")
    return NavigationStudySource(None, None, {"69": directory})


def test_formal_condition_groups_reuse_reference_without_double_counting():
    from embodied_learning.navigation_robustness import GROUPS

    source = source_or_skip()
    assert len(source.summaries["69"]["rows"]) == 233
    for group, keys in GROUPS.items():
        record = source.load(f"69_narrow_crate_8__{group}", "69")
        assert record.method_keys == tuple(keys)
        assert record.metadata["tracking_round"] == 4
        assert "同一同比例" in record.metadata["protocol"]
        assert "零定位误差是设定" not in record.metadata["settings"]
        for key in keys:
            assert record.method_info(key)["result"]["perturbation"]["key"] == key
            assert len(record.telemetry[key]["executed_request_frame"]) == len(record.timestamps)
    bridge = source.load("69_plaza_crate_0__zero", "69")
    assert bridge.method_keys == ("zero",)
    negative = source.load("69_blocked_crate_8__delay", "69")
    assert all(negative.lengths[k] == 1 for k in negative.method_keys)


@pytest.mark.isolated_tk
def test_grouped_conditions_switch_all_views_and_keep_delay_page():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, source_or_skip(), "69_narrow_crate_8__delay", "69")
    try:
        app.seek(60)
        app.analysis.tabs.select(6)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert (
                app.clock.frame == 60 and app.analysis.tabs.index(app.analysis.tabs.select()) == 6
            )
            assert app.analysis.robustness.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert app.record.goal_info(60, key) is not None
            assert "执行来源" in app.analysis.robustness.note.cget("text")
        assert len(app.analysis.robustness.figure.axes) == 2
        app.select_record("69_narrow_crate_8__yaw_plus", "69")
        app.analysis.tabs.select(6)
        app.method_buttons["yaw_plus_4"].invoke()
        root.update()
        assert "朝向偏差 +4°" in app.analysis.robustness.note.cget("text")
        assert app.record.statistics["yaw_plus_4"]["mean_m"] == 0
        app.select_record("69_blocked_crate_8__delay", "69")
        app.analysis.tabs.select(6)
        root.update()
        assert "初始保持" in app.analysis.robustness.note.cget("text")
    finally:
        app.close()
