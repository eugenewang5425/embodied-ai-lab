"""Known-answer motion, stopping contract and speed-factor checks for lesson 69."""

import numpy as np
import pytest

from embodied_learning.braking_control import (
    BrakingController,
    brake_step,
    braking_path,
    preview_speed,
)
from embodied_learning.navigation_geometry import NavigationRig


def test_proportional_stop_keeps_curvature_and_both_acceleration_limits():
    rig = NavigationRig()
    before = np.array([0.6, 0.084])
    independent = brake_step(before, rig)
    coupled = brake_step(before, rig, True)
    np.testing.assert_allclose(independent, [0.52, 0])
    np.testing.assert_allclose(coupled, [0.52, 0.0728])
    assert coupled[1] / coupled[0] == pytest.approx(before[1] / before[0])
    for initial in ([0.6, 1], [-0.2, -0.9], [0, 0.7], [0.6, 0], [0, 0]):
        velocity = np.array(initial, float)
        for _ in range(15):
            nxt = brake_step(velocity, rig, True)
            assert np.all(np.abs(nxt - velocity) <= [0.08 + 1e-12, 0.18 + 1e-12])
            assert np.all(np.abs(nxt) <= np.abs(velocity) + 1e-12)
            velocity = nxt
        np.testing.assert_array_equal(velocity, [0, 0])


def test_discrete_stop_prediction_matches_independent_arc_integration():
    rig = NavigationRig()
    for coupled in (False, True):
        for latency in (0, 0.2):
            path, commands = braking_path([0.6, 0.084], rig, coupled, latency)
            pose = np.zeros(3)
            for v, w in commands:
                turn = w * 0.1
                distance = v * 0.1 * np.sinc(turn / (2 * np.pi))
                pose[:2] += distance * np.array(
                    [np.cos(pose[2] + turn / 2), np.sin(pose[2] + turn / 2)]
                )
                pose[2] += turn
            np.testing.assert_allclose(path[-1], pose, atol=1e-12)
            if latency:
                np.testing.assert_allclose(commands[:2], [[0.6, 0.084]] * 2)
            if coupled:
                np.testing.assert_allclose(commands[:, 1] / commands[:, 0], 0.14)
    with pytest.raises(ValueError, match="whole number"):
        braking_path([0.6, 0.1], rig, latency=0.15)


def test_speed_budget_uses_route_curvature_and_preserves_feedback():
    rig = NavigationRig()
    theta = np.linspace(0, 0.5, 100)
    radius = 2.0
    path = np.c_[2 + radius * np.sin(theta), 6 + radius * (1 - np.cos(theta))]
    pose = np.array([2, 6, 0])
    cap, curvature = preview_speed(path, pose, np.array([0.6, 0]), rig)
    # Chord-length sampling approximates arc length, with ~5e-7 curvature error.
    assert curvature == pytest.approx(0.5, abs=1e-6)
    assert cap == pytest.approx(np.sqrt(0.04 / 0.5))
    a = BrakingController(np.zeros((120, 120), bool), rig, "discrete")
    b = BrakingController(np.zeros((120, 120), bool), rig, "speed")
    a.path, b.path = path.copy(), path.copy()
    old, new = a.tracking_command(pose, np.zeros(2)), b.tracking_command(pose, np.zeros(2))
    assert new[0] < old[0]
    assert new[1] - old[1] == pytest.approx(
        (new[0] - old[0]) * a.tracking_log[-1]["curvature_m_inv"]
    )


@pytest.mark.parametrize("method", ["discrete", "coupled"])
def test_no_path_executes_remaining_braking_before_terminal_zero(monkeypatch, method):
    from embodied_learning.experiments import braking_navigation as experiment

    class KnownFailure:
        path_capacity = 256

        def __init__(self, prior, rig, mode):
            self.grid = prior
            self.coupled = mode == "coupled"
            self.path = np.array([[1.5, 6], [10.5, 6]])
            self.tracking_log, self.planning_log = [], []

        def step(self, estimate, goal, ranges, hits, depth, valid, frame, velocity):
            self.latest = {
                "speed_cap": 0.6,
                "preview_curvature": 0,
                "nominal": np.array([0.6, 0.08]),
                "stop_clearance": 1,
            }
            if frame >= 8:
                self.path = None
                return np.zeros(2), "no_observed_route", 1
            return np.array([0.6, 0.08]), "following", 1

    monkeypatch.setattr(experiment, "BrakingController", KnownFailure)
    row, data = experiment.episode("plaza", "crate", 0, method)
    assert row["status"] == "no_path" and row["physically_stopped"]
    assert row["trigger_frame"] == 8 and row["tail_time_s"] > 0.6
    assert row["tail_distance_m"] > 0.18
    np.testing.assert_array_equal(data["incoming_velocity"][-1], [0, 0])
    assert len(data["truth"]) > 9
    differences = np.diff(np.vstack(([0, 0], data["commands"])), axis=0)
    assert np.all(np.abs(differences) <= [0.08 + 1e-12, 0.18 + 1e-12])


def test_new_observation_adapter_matches_legacy_memory_for_same_sensor_inputs():
    from embodied_learning.navigation_geometry import sense
    from embodied_learning.path_tracking import RetainedTangentController

    rig = NavigationRig()
    prior = np.zeros((120, 120), bool)
    old = RetainedTangentController(prior, rig, "depth")
    new = BrakingController(prior, rig, "discrete")
    old.make_plan = lambda *_: np.array([[1.5, 6.0], [4.0, 6.0]])
    rng = np.random.default_rng(21)
    for f in range(4):
        pose = np.array([1.5 + f * 0.02, 6, f * 0.01])
        ranges, hits, depth, valid = sense(
            np.array([[2.5, 2.7, 5.8, 6.2, 0.1, 0.4]]), pose, rig, rng
        )
        old.step(pose, np.array([4, 6]), ranges, hits, depth, valid, f, np.zeros(2))
        new.observe(pose, ranges, hits, depth, valid)
        assert old.voxels.keys() == new.voxels.keys()
        np.testing.assert_array_equal(list(old.voxels.values()), list(new.voxels.values()))
        np.testing.assert_array_equal(old.grid, new.grid)


def formal_source():
    import json
    from pathlib import Path

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    p = Path("results/braking_navigation_v3")
    if (
        not (p / "summary.json").exists()
        or len(json.loads((p / "summary.json").read_text())["rows"]) != 230
    ):
        pytest.skip("generate the frozen 230-case batch separately")
    return NavigationStudySource(None, None, {"69": p})


def test_five_arm_archives_keep_real_stop_state_negative_case_and_frozen_sources():
    import hashlib
    from pathlib import Path

    source = formal_source()
    summary = source.summaries["69"]
    directory = source.directories["69"]
    for name, digest in summary["protocol"]["source_sha256"].items():
        assert hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert sum(r.get("legacy_prefix_equal", False) for r in summary["rows"]) == 27
    record = source.load("69_blocked_crate_5", "69")
    assert len(record.method_keys) == 5
    for key in record.method_keys:
        assert record.lengths[key] == 1
        assert record.method_info(key)["result"]["status"] == "no_path"
        np.testing.assert_array_equal(record.telemetry[key]["incoming_velocity"][0], [0, 0])
    record = source.load("69_narrow_crate_0", "69")
    result = record.method_info("joint_brake")["result"]
    assert result["status"] == "no_path" and result["tail_distance_m"] > 0
    assert result["physically_stopped"]
    final = record.lengths["joint_brake"] - 1
    np.testing.assert_array_equal(
        record.telemetry["joint_brake"]["incoming_velocity"][final], [0, 0]
    )


@pytest.mark.isolated_tk
def test_five_method_switch_keeps_braking_tab_and_all_views_synchronized():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, formal_source(), "69_narrow_crate_0", "69")
    try:
        app.seek(70)
        app.analysis.tabs.select(5)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert (
                app.clock.frame == 70 and app.analysis.tabs.index(app.analysis.tabs.select()) == 5
            )
            assert app.analysis.braking.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert "进入本帧" in app.analysis.braking.note.cget("text")
            assert len(app.analysis.braking.figure.axes) == 2
            assert app.record.goal_info(70, key) is not None
        assert int(app.stats.table.cget("height")) == 5
        app.analysis.tabs.select(3)
        app.method_buttons["coupled"].invoke()
        root.update()
        assert "立即刹停" in app.analysis.safety.note.cget("text")
        app.select_record("69_blocked_crate_5", "69")
        app.analysis.tabs.select(5)
        root.update()
        assert "停稳" in app.analysis.braking.note.cget("text")
    finally:
        app.close()
