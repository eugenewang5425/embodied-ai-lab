"""Reachable geometry and genuinely sensor-isolated navigation comparisons."""

from dataclasses import replace

import numpy as np
import pytest

from embodied_learning.experiments.navigation_bypass_study import (
    PROFILES,
    feasibility_witness,
    map_scene,
)
from embodied_learning.experiments.navigation_progress_study import SplitSensorNoise
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_geometry import NavigationRig, depth_points, sense, world_points
from embodied_learning.navigation_sensor_ablation import SensorMapController


@pytest.mark.parametrize("profile", PROFILES)
def test_unknown_obstacle_has_a_body_feasible_bypass(profile):
    prior, actual = map_scene(profile)
    assert len(actual) == len(prior) + 1
    np.testing.assert_array_equal(actual[:-1], prior)
    certificate = feasibility_witness(profile)
    assert certificate["sampled_body_clearance_m"] > 0.16
    assert (certificate["straight_body_clearance_m"] > 0) == (profile == "clear_overhead")


@pytest.mark.parametrize("profile", ["bypass_low", "bypass_beam", "clear_overhead"])
def test_horizontal_lidar_cannot_see_height_blind_obstacles(profile):
    rig = replace(NavigationRig(), margin_m=0, width=320, height=240)
    pose = np.array([1.5, 6, 0])
    prior, actual = map_scene(profile)
    base = sense(prior, pose, rig, SplitSensorNoise(0))
    obstacle = sense(actual, pose, rig, SplitSensorNoise(0))
    np.testing.assert_array_equal(base[0], obstacle[0])
    np.testing.assert_array_equal(base[1], obstacle[1])
    if profile != "clear_overhead":
        points = world_points(depth_points(obstacle[2], obstacle[3], rig), pose)
        b = actual[-1]
        assert (
            (points[:, 0] > b[0] - 0.02)
            & (points[:, 0] < b[1] + 0.02)
            & (points[:, 1] > b[2])
            & (points[:, 1] < b[3])
        ).any()


def test_camera_depth_cannot_leak_into_lidar_only_map_or_safety_points():
    rig = replace(NavigationRig(), margin_m=0, width=320, height=240)
    pose = np.array([1.5, 6, 0])
    prior, actual = map_scene("bypass_low")
    ranges, hits, depth, valid = sense(actual, pose, rig, SplitSensorNoise(0))
    a, b = [SensorMapController(prior, rig, "lidar_only") for _ in range(2)]
    p = a.observe(pose, ranges, hits, depth, valid)
    q = b.observe(pose, ranges, hits, np.full_like(depth, 0.3), np.ones_like(valid))
    np.testing.assert_array_equal(a.grid, b.grid)
    np.testing.assert_array_equal(p, q)
    assert not a.grid[54:66, 56:64].any()
    fusion = SensorMapController(prior, rig, "lidar_depth")
    fusion.observe(pose, ranges, hits, depth, valid)
    assert fusion.grid[54:66, 56:64].any()


def test_new_endpoint_is_not_enlarged_into_its_full_storage_cell():
    rig = replace(NavigationRig(), margin_m=0)
    c = SensorMapController(np.empty((0, 6)), rig, "lidar_only")
    pose = np.array([1.51, 1.08, 0])
    ranges = np.full(180, 8.0)
    hits = np.zeros(180, bool)
    ranges[0], hits[0] = 0.23, True
    c.observe(pose, ranges, hits, np.full((60, 80), np.nan), np.zeros((60, 80), bool))
    points = np.array(list(c.voxels.values()))[:, :2]
    assert c.grid.any() and not c.prior.any()
    metric = FootprintField(c.prior, points, rig)
    assert metric.free(pose)
    assert not FootprintField(c.grid, points, rig).free(pose)
    # The return is still an obstacle when the actual body reaches it.
    assert not metric.free(np.array([1.54, 1.08, 0]))


def complete_archive():
    import json
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / "results/navigation_reinforcement_v11"
    if not (directory / "summary.json").exists():
        pytest.skip("generate the registered round-eleven records first")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    if len(summary["rows"]) != 24:
        pytest.skip("formal batch is still collecting")
    return directory


def test_sensor_use_metadata_and_recorded_maps_are_explicit_and_read_only():
    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    directory = complete_archive()
    record = NavigationStudySource(None, None, {"69": directory}).load(
        "69_street_low_29__delay_2__bypass_low", "69"
    )
    assert record.metadata["depth_used_by"] == ["lidar_depth"]
    assert record.metadata["live_map"]
    for key in record.method_keys:
        assert not record.telemetry[key]["planning_prior_frames"].flags.writeable
        assert record.depth[key]["values"].shape[1:] == (240, 320)
    overhead = NavigationStudySource(None, None, {"69": directory}).load(
        "69_street_beam_29__delay_2__clear_overhead", "69"
    )
    for key in overhead.method_keys:
        d = overhead.telemetry[key]
        # The high fixture covers 8x12 cells, but none block this robot's height.
        np.testing.assert_array_equal(d["map_error"][:, 1] - d["body_height_map_error"][:, 1], 96)


@pytest.mark.isolated_tk
def test_reachable_obstacle_method_choice_synchronizes_all_views_and_depth_meaning():
    import tkinter as tk

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource
    from embodied_learning.replay_viewer.window import ReplayWindow

    directory = complete_archive()
    root = tk.Tk()
    app = ReplayWindow(
        root,
        NavigationStudySource(None, None, {"69": directory}),
        "69_street_low_29__delay_2__bypass_low",
        "69",
    )
    try:
        app.seek(180)
        app.analysis.tabs.select(9)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 180
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 9
            np.testing.assert_array_equal(
                app.map.raster.get_array(), app.record.telemetry[key]["map_frames"][180]
            )
            app.analysis.safety.draw(app.record, key, 180)
            titles = [a.get_title() for a in app.analysis.safety.figure.axes]
            assert any(
                ("本组未使用" if key == "lidar_only" else "控制输入") in title for title in titles
            )
    finally:
        app.close()
