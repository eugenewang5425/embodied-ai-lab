"""Sensor geometry, paired control, scoring and viewer telemetry contracts."""

import math
import time
from dataclasses import replace

import numpy as np
import pytest

from embodied_learning.experiments.campus_patrol import (
    DT,
    RADIUS,
    RANGE,
    RES,
    TASKS,
    CampusPF,
    Config,
    advance,
    plan,
    run,
    run_episode,
    scan,
    world,
)
from embodied_learning.experiments.grid_nav import Obstacle, inflate
from embodied_learning.experiments.map_localization import ParticleFilter
from embodied_learning.replay_viewer.goal_overlay import project_goal
from embodied_learning.replay_viewer.model import CameraSpec, ReplayRecord, Track


def test_campus_tasks_and_map_match_physical_obstacles():
    geometry, styles, occupied = world()
    assert len(styles) == len(geometry)
    safe = ~inflate(occupied, math.ceil((RADIUS + 0.1) / RES))
    position = (2.0, 2.0)
    for _, goal in TASKS:
        points = plan(safe, position, goal)
        assert points is not None
        assert np.allclose(points[-1], goal)
        for p in points:
            assert min(o.distance(*p) for o in geometry) >= RADIUS
        position = goal
    for row, col in np.random.default_rng(12).integers(0, len(occupied), (300, 2)):
        distance = min(o.distance((col + 0.5) * RES, (row + 0.5) * RES) for o in geometry)
        if abs(distance) > 1e-9:  # exact surface cells may round to either side
            assert occupied[row, col] == (distance < 0)
    assert plan(safe, (2, 2), (6.6, 6.7)) is None


def test_lidar_hit_and_no_return_known_answer():
    obstacle = Obstacle("circle", 5, 2, r=1)
    ranges, hits = scan([1, 2, 0], (obstacle,))
    assert hits[0] and ranges[0] == pytest.approx(3)
    assert not hits[32] and ranges[32] == RANGE
    noisy, flags = scan([1, 2, 0], (obstacle,), np.full(64, 0.1))
    assert noisy[0] == pytest.approx(3.1)
    assert noisy[32] == RANGE and not flags[32]


def test_vectorized_pf_matches_existing_likelihood_rule():
    grid = np.zeros((100, 100))
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    pf = CampusPF(
        grid, 0.1, 100, np.random.default_rng(1), init_pose=[2, 2, 0], spread=0.01, rays_n=32
    )
    legacy = ParticleFilter(
        grid, 0.1, 100, np.random.default_rng(1), init_pose=[2, 2, 0], spread=0.01, rays_n=32
    )
    obs = np.full(64, 2.1)
    obs[3::5] = 4.0
    legacy.measure(obs, None)
    pf.update(obs, obs < 3.9)
    assert pf.ess() > 50
    assert np.allclose(pf.w, legacy.w, atol=1e-12)


def test_command_scan_goal_and_score_share_the_same_frame():
    geometry, _, grid = world()
    safe = ~inflate(grid, math.ceil((RADIUS + 0.1) / RES))
    config = Config(max_steps=400)
    row, data = run_episode("odom", 0, geometry, safe, grid, config)
    n = row["frames"]
    assert all(len(a) == n for a in data.values())
    for k in (0, 45, 150, n - 2):
        v, w = data["commands"][k]
        assert np.allclose(data["truth"][k + 1], advance(data["truth"][k], v * DT, w * DT))
        _, hits = scan(data["truth"][k], geometry)
        assert np.array_equal(hits, data["hits"][k])
        assert np.allclose(data["goal"][k], TASKS[data["mission"][k]][1])
    assert np.allclose(data["commands"][-1], 0)
    assert row["announced"] > row["reached"]  # falsely arriving is never scored as success
    for event in row["events"]:
        distance = np.linalg.norm(data["truth"][event["frame"], :2] - TASKS[event["goal_index"]][1])
        assert event["true_distance_m"] == pytest.approx(distance)
        assert event["success"] == (distance <= config.true_arrival_m)


def test_reference_controller_can_reach_every_inspection_point():
    geometry, _, grid = world()
    safe = ~inflate(grid, math.ceil((RADIUS + 0.1) / RES))
    row, _ = run_episode("truth", 0, geometry, safe, grid, Config())
    assert row["status"] == "completed"
    assert row["reached"] == len(TASKS) and row["contacts"] == 0


def test_each_estimate_uses_its_own_actual_motion_and_padding_is_not_scored():
    truth = np.zeros((4, 3))
    actual = truth + [10, 0, 0]
    estimate = actual + [[1, 0, 0], [3, 0, 0], [3, 0, 0], [3, 0, 0]]
    r = ReplayRecord(
        "paired",
        np.arange(4.0),
        (
            Track("truth", "参考", "b", truth),
            Track("other_true", "独立实际", "r", actual, "other_true", "truth"),
            Track("other", "独立估计", "r", estimate, "other_true"),
        ),
        {"map": np.zeros((2, 2))},
        1.0,
        lengths={"other": 2},
        lidar={
            "other": {
                "ranges": np.ones((4, 4)),
                "hits": np.tile([True, False, False, False], (4, 1)),
            }
        },
    )
    assert r.statistics["other"]["mean_m"] == 2
    assert len(r.errors) == 1
    assert np.allclose(r.scan_world(0, "other"), [[12, 0]])
    assert np.allclose(r.scan_world(0, "other", estimated=False), [[11, 0]])


def test_camera_goal_projection_handles_front_side_and_behind():
    camera = CameraSpec()
    location, _ = project_goal(np.array([0, 0, 0]), [4, 0], camera, (640, 480))
    assert location[0] == 320
    assert project_goal(np.array([0, 0, 0]), [-4, 0], camera, (640, 480))[0] is None
    assert "后方" in project_goal(np.array([0, 0, 0]), [-4, 0], camera, (640, 480))[1]
    assert "右侧" in project_goal(np.array([0, 0, 0]), [1, -4], camera, (640, 480))[1]
    assert "前下方" in project_goal(np.array([0, 0, 0]), [0.4, 0], camera, (640, 480))[1]
    rotated, _ = project_goal(np.array([0, 0, np.pi / 2]), [0, 4], camera, (640, 480))
    assert rotated[0] == pytest.approx(320)


@pytest.mark.isolated_tk
def test_campus_gui_switches_independent_camera_goals_lidar_and_split_views(tmp_path):
    import tkinter as tk

    from embodied_learning.replay_viewer.campus import CampusSource
    from embodied_learning.replay_viewer.window import ReplayWindow

    directory = tmp_path / "campus"
    run(directory, (0,), replace(Config(), max_steps=20))
    source = CampusSource(directory)
    root = tk.Tk()
    app = ReplayWindow(root, source, "0", "v1")
    try:
        stop = time.perf_counter() + 0.5
        while time.perf_counter() < stop:
            root.update()
            time.sleep(0.01)
        app.seek(10)
        app.analysis.tabs.select(2)
        app.method_buttons["odom"].invoke()
        root.update()
        assert app.clock.frame == 10 and not app.clock.playing
        assert app.analysis.tabs.index(app.analysis.tabs.select()) == 0
        assert app.view_choice.get() == "odom"
        assert app.scene.visible == {"odom", "odom_truth"}
        assert app.camera.focus_key == app.stats.focus_key == "odom"
        expected = app.record.scan_world(10, "odom")
        assert np.allclose(app.map.scan_points.get_offsets(), expected)
        assert "P1" in app.scene.note.cget("text")
        assert np.asarray(app.camera.image).mean() > 1
        app.view_buttons["split"].invoke()
        root.update()
        assert app.analysis.tabs.index(app.analysis.tabs.select()) == 1
        assert len(app.scene.visible) == 7
        assert len(app.analysis.plots) == 4
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            assert app.clock.frame == 10
            assert app.scene.visible == app.record.method_tracks(key)
            assert app.camera.focus_key == app.stats.focus_key == app.map.focus_key == key
            assert app.scene.engine.focus_key == app.camera.engine.focus_key == key
            assert app.map.layer.get() == app.record.method_info(key)["display_map"]
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 0
            assert {k for k, v in app.track_vars.items() if v.get()} == app.scene.visible
            assert all(line.get_visible() == (k == key) for k, line in app.error.lines.items())
            assert np.allclose(app.map.scan_points.get_offsets(), app.record.scan_world(10, key))
            if key in app.record.errors:
                assert "focused" in app.stats.table.item(key, "tags")
        app.clock.play()
        app.method_buttons["prior"].invoke()
        assert app.clock.playing and app.clock.frame == 10
        app.clock.pause()
        app.view_buttons["summary"].invoke()
        assert app.map.layer.get() == "叠加对照"
        assert app.analysis.tabs.index(app.analysis.tabs.select()) == 0
        assert len(app.scene.visible) == 7
        app.lidar_var.set(False)
        app.refresh()
        assert len(app.map.scan_points.get_offsets()) == 0
        app.analysis.tabs.select(2)
        root.update()
        assert "只靠轮子推算" in app.analysis.help.get("1.0", "end")
        app.export_current(tmp_path / "frame.json")
    finally:
        app.close()
