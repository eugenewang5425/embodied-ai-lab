"""Known-answer geometry/calibration and sensor-driven control regressions."""

import numpy as np
import pytest

from embodied_learning.navigation_geometry import (
    NavigationRig,
    advance,
    collision_clearance,
    depth_points,
    lidar_obstacle_points,
    lidar_points,
    point_clearance,
    project,
    sense,
    stop_path,
    swept_clearance,
    world_points,
)


def test_wall_distances_include_sensor_offsets_and_round_trip_pixels():
    rig = NavigationRig()
    ranges, hits, depth, valid = sense(np.array([[1.0, 1.1, -2, 2, 0, 2]]), np.zeros(3), rig)
    assert ranges[0] == pytest.approx(1 - rig.lidar_x_m)
    assert depth[30, 40] == pytest.approx(1 - rig.camera_x_m)
    p = lidar_points(ranges, hits, rig)
    assert p[0, 0] == pytest.approx(1.0)
    points = depth_points(depth, valid, rig)
    pixels, front = project(points, rig)
    assert front.all()
    assert np.max(np.abs(pixels - np.round(pixels))) < 1e-10
    # Raw range is NOT the remaining distance of the bumper to the wall.
    assert ranges[0] - (rig.front_m - rig.lidar_x_m) == pytest.approx(0.7)


def test_speed_and_body_size_change_stop_collision_before_center_hits():
    rig = NavigationRig()
    points = np.array([[0.7, 0, 0.15]])
    assert point_clearance(points, stop_path(0.8, 0, rig), rig) < 0
    assert point_clearance(points, stop_path(0.2, 0, rig), rig) > 0
    distance = stop_path(0.8, 0, rig)[-1, 0]
    formula = 0.8 * rig.latency_s + 0.8**2 / (2 * rig.brake_m_s2)
    assert formula <= distance <= formula + 0.04 * 0.8
    assert distance < 0.7  # point robot would incorrectly call this safe


def test_low_obstacle_requires_earlier_depth_when_current_frame_is_blind():
    rig = NavigationRig()
    boxes = np.array([[0.62, 0.8, -0.65, 0.65, 0, 0.14]])
    _ranges, hits, depth, valid = sense(boxes, np.zeros(3), rig)
    assert not hits.any()
    assert len(depth_points(depth, valid, rig)) == 0
    earlier = np.array([-0.6, 0, 0])
    _, _, depth, valid = sense(boxes, earlier, rig)
    memory = world_points(depth_points(depth, valid, rig), earlier)
    assert len(memory) > 0
    assert point_clearance(memory, stop_path(0.6, 0, rig), rig, rig.margin_m) <= 0
    assert min(collision_clearance(p, boxes, rig) for p in stop_path(0.6, 0, rig)) <= 0


def test_height_and_rear_motion_use_whole_body_not_front_cone():
    rig = NavigationRig()
    assert collision_clearance(np.zeros(3), [[0.16, 0.18, -0.02, 0.02, 0.50, 0.56]], rig) <= 0
    assert collision_clearance(np.zeros(3), [[0.16, 0.18, -0.02, 0.02, 0.70, 0.8]], rig) > 0
    ranges, hits, _, _ = sense(np.array([[-0.65, -0.50, -0.5, 0.5, 0, 0.8]]), np.zeros(3), rig)
    assert (
        point_clearance(lidar_obstacle_points(ranges, hits, rig), stop_path(-0.6, 0, rig), rig) <= 0
    )
    # A side obstacle outside the initial box can be hit by the rotating front corner.
    box = [[0.25, 0.31, 0.26, 0.31, 0, 1.0]]
    assert collision_clearance(np.zeros(3), box, rig) > 0
    assert swept_clearance(np.zeros(3), 0, 1.0, 0.6, box, rig) <= 0


def test_calibration_fits_mounts_and_checks_disjoint_targets():
    from embodied_learning.experiments.body_calibration import extrinsic_calibration

    result = extrinsic_calibration(NavigationRig())
    assert result["heldout_targets"] == 40
    assert result["camera_translation_error_mm"] < 1
    assert result["camera_holdout_p95_mm"] < 4
    assert result["lidar_translation_error_mm"] < 1


def test_analytic_depth_matches_independent_mujoco_renderer(tmp_path):
    from scipy.ndimage import binary_erosion

    from embodied_learning.experiments.body_calibration import render_fixture

    rig = NavigationRig()
    boxes = np.array([[0.7, 0.8, -0.7, 0.7, 0.1, 1.4]])
    _, _, depth, valid = sense(boxes, np.zeros(3), rig)
    rendered = render_fixture(boxes, rig, tmp_path / "rgb.png")
    keep = binary_erosion(valid & (depth < 7), iterations=2)
    assert np.percentile(np.abs(depth[keep] - rendered[keep]), 95) < 0.002


def test_observations_drive_detour_and_commands_match_next_frame():
    from embodied_learning.experiments.observed_navigation import DT, episode

    base, _, _ = episode("plaza", "low", "lidar", 0)
    row, data, _ = episode("plaza", "low", "depth", 0)
    assert base["status"] == "collision_abort"
    assert row["status"] == "completed" and row["contacts"] == 0
    assert row["min_clearance_m"] > 0
    for i in range(len(data["truth"]) - 1):
        expected = advance(data["truth"][i], *data["commands"][i], DT)
        assert np.allclose(data["truth"][i + 1], expected)
    assert np.array_equal(data["commands"][-1], [0.0, 0.0])


def test_local_planner_does_not_receive_hidden_world_obstacles():
    from embodied_learning.experiments.observed_navigation import ObservedController

    rig = NavigationRig()
    a = ObservedController(np.zeros((120, 120), bool), rig, "depth")
    empty = (
        np.full(180, 8.0),
        np.zeros(180, bool),
        np.full((60, 80), 8.0),
        np.zeros((60, 80), bool),
    )
    cmd, reason, _ = a.step(np.array([1.5, 6, 0]), np.array([10.5, 6]), *empty, 0, np.zeros(2))
    assert reason == "following" and cmd[0] > 0
    assert not a.grid.any()
