"""Route-order and sensor-resolution counterexamples with known geometry."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from embodied_learning.experiments.navigation_progress_study import SplitSensorNoise, map_scene
from embodied_learning.navigation_geometry import NavigationRig, sense
from embodied_learning.navigation_map_update import EvidenceMap, MapController
from embodied_learning.navigation_progress import ProgressMapController, advance_prefix


def test_passed_retained_start_advances_without_changing_curve():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    np.testing.assert_equal(advance_prefix(path, np.array([0.8, 0.01, 0]), 0)["waypoint"], 1)
    np.testing.assert_equal(path, [[0, 0], [2, 0], [2, 2]])


@pytest.mark.parametrize("pose", [[-0.4, 0, 0], [0.8, 0.06, 0], [2.4, 0, 0], [2, 1, 0]])
def test_unpassed_off_route_and_future_turn_do_not_skip(pose):
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    assert not advance_prefix(path, np.array(pose), 0)["prefix_passed"]


def test_crossing_cannot_jump_to_nearby_later_segment():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.8, 2.0], [0.8, -2.0]])
    assert advance_prefix(path, np.array([0.8, 1.0, 0]), 0)["waypoint"] == 0
    assert advance_prefix(path, np.array([0.8, 0.0, 0]), 0)["waypoint"] == 1
    assert advance_prefix(path, np.array([0.8, 0.0, 0]), 2)["waypoint"] == 2


def test_old_stalled_input_now_requests_motion_with_same_turning_rule():
    path = np.array(
        [[4.57238906, 6.45505844], [5.70187016, 6.81691318], [10.53205714, 6.24177099], [10.5, 6]]
    )
    pose = np.array([5.03069563, 6.59642307, 0.32738472])
    rig = replace(NavigationRig(), margin_m=0)
    controllers = [
        cls(np.empty((0, 6)), rig, True) for cls in (MapController, ProgressMapController)
    ]
    commands = []
    for c in controllers:
        c.path, c.wp = path.copy(), 0
        commands.append(c.tracking_command(pose, np.zeros(2)))
    assert commands[0][0] == 0
    assert commands[1][0] == 0.2
    assert controllers[1].latest["prefix_passed"]
    # Same angular feedback, with only the existing speed-dependent feed-forward changed.
    k = controllers[1].tracking_log[-1]["curvature_m_inv"]
    assert commands[1][1] == pytest.approx(commands[0][1] + 0.2 * k)


def test_corner_lookahead_cannot_disable_tangent_aligned_motion():
    rig = replace(NavigationRig(), margin_m=0)
    c = ProgressMapController(np.empty((0, 6)), rig, True)
    c.path = np.array(
        [[2.84415478, 6.79186893], [3.33346125, 7.13316458], [6.89014755, 7.13466403], [10.5, 6.0]]
    )
    c.wp = 2
    command = c.tracking_command(np.array([3.12944166, 6.97670231, 0.63558796]), np.zeros(2))
    assert command[0] == 0.2
    t = c.tracking_log[-1]
    heading = t["heading_rad"] - np.arctan2(2 * t["cross_m"], 0.6)
    assert command[1] == pytest.approx(
        np.clip(0.2 * t["curvature_m_inv"] + 2 * (heading - 0.63558796), -1, 1)
    )


def test_unaligned_body_rotates_before_moving():
    c = ProgressMapController(np.empty((0, 6)), replace(NavigationRig(), margin_m=0), True)
    c.path = np.array([[0.0, 0.0], [2.0, 0.0]])
    c.wp = 0
    command = c.tracking_command(np.array([0.4, 0, np.pi / 2]), np.zeros(2))
    assert command[0] == 0
    assert command[1] == -1


def stationary_map(profile, dense):
    rig = replace(
        NavigationRig(), margin_m=0, width=320 if dense else 80, height=240 if dense else 60
    )
    prior, actual = map_scene(profile)
    memory = EvidenceMap(prior, rig, True, True)
    rng = SplitSensorNoise(0)
    for _ in range(5):
        memory.update(np.array([1.5, 6.0, 0]), *sense(actual, np.array([1.5, 6.0, 0]), rig, rng))
    return memory


@pytest.mark.parametrize("profile", ["removed", "removed_low"])
def test_dense_depth_can_clear_visible_bottom_where_coarse_cannot(profile):
    coarse, dense = stationary_map(profile, False), stationary_map(profile, True)
    assert coarse.grid[55:65, 56:64].all()
    assert not dense.grid[55:65, 56:64].any()


@pytest.mark.parametrize("profile", ["present_low", "present_beam"])
def test_real_height_blind_obstacle_is_not_removed(profile):
    assert stationary_map(profile, True).grid[55:65, 56:64].all()


def test_occluded_box_stays_occupied_with_dense_camera():
    m = stationary_map("occluded", True)
    assert m.occupied[57:63, 57:63].any(axis=2).all()


def test_depth_draw_count_cannot_change_lidar_noise():
    a, b = SplitSensorNoise(26), SplitSensorNoise(26)
    for _ in range(4):
        np.testing.assert_array_equal(a.normal(0, 0.005, 180), b.normal(0, 0.005, 180))
        a.normal(0, 0.003, 4800)
        b.normal(0, 0.003, 76800)


def archive(number):
    directory = Path(__file__).resolve().parents[1] / f"results/navigation_reinforcement_v{number}"
    if not (directory / "summary.json").exists():
        pytest.skip("generate registered archives first")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    if len(summary["rows"]) != 36:
        pytest.skip("registered batch is still running")
    return directory, summary


@pytest.mark.slow
def test_old_progress_bridge_reproduces_frozen_eighth_round():
    from embodied_learning.experiments.navigation_reinforcement import episode
    from embodied_learning.navigation_robustness import CONDITIONS
    from embodied_learning.replay_viewer.campus import read_checked

    directory, summary = archive(9)
    row = next(
        r
        for r in summary["rows"]
        if r["seed"] == 23 and r["profile"] == "added" and r["method"] == "progress_old"
    )
    _, original = episode("street", "crate", 23, CONDITIONS["delay_2"], "map_evidence", "added")
    stored = read_checked(directory / row["file"], row["sha256"])
    for name in original:
        if name not in ("control_wall_s",):
            np.testing.assert_array_equal(original[name], stored[name], err_msg=name)


@pytest.mark.parametrize(
    "number,case",
    [(9, "69_street_crate_23__delay_2__unchanged"), (10, "69_street_low_26__delay_2__removed_low")],
)
def test_current_maps_and_method_specific_depth_shapes_are_read_only(number, case):
    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    directory, _ = archive(number)
    record = NavigationStudySource(None, None, {"69": directory}).load(case, "69")
    assert record.metadata["live_map"]
    for key in record.method_keys:
        assert not record.telemetry[key]["map_frames"].flags.writeable
        w, h = record.metadata["depth_dimensions_by_method"][key]
        assert record.depth[key]["values"].shape[1:] == (h, w)


@pytest.mark.isolated_tk
def test_dense_depth_selection_updates_diagnostics_map_and_all_views():
    import tkinter as tk

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
    from embodied_learning.replay_viewer.window import ReplayWindow

    directory, _ = archive(10)
    root = tk.Tk()
    app = ReplayWindow(
        root,
        NavigationStudySource(None, None, {"69": directory}),
        "69_street_low_26__delay_2__removed_low",
        "69",
    )
    try:
        app.seek(5)
        app.analysis.tabs.select(9)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 5
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 9
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            np.testing.assert_array_equal(
                app.map.raster.get_array(), app.record.telemetry[key]["map_frames"][5]
            )
            app.analysis.safety.draw(app.record, key, 5)
            app.analysis.safety.canvas.draw()
        app.seek(0)
        app.choose_view("split")
        root.update()
        for key in app.record.method_keys:
            np.testing.assert_array_equal(
                app.analysis.map_rasters[key].get_array(),
                app.record.telemetry[key]["map_frames"][0],
            )
    finally:
        app.close()
