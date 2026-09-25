"""Independent 3D ray/contact checks for the photo-room transfer experiment."""

import copy
import math

import numpy as np
import pytest

from embodied_learning.experiments.photo_room_validation import drive_replay, path_contacts, plan
from embodied_learning.odometry import estimate_poses
from embodied_learning.photo_room import RoomWorld, build_geometry, default_layout


@pytest.fixture(scope="module")
def room_world():
    return RoomWorld(default_layout())


def test_user_desk_measurements_reach_collision_geometry():
    desk = next(g for g in build_geometry(default_layout()) if g["name"] == "desk_top")
    assert desk["pos"][2] + desk["size"][2] / 2 == pytest.approx(0.69)
    assert desk["size"][0] == pytest.approx(0.65)


def test_real_3d_rays_see_bed_only_at_frame_height(room_world):
    pose = [1.5, 2.0, math.pi / 2]
    low, _ = room_world.scan(pose, z=0.18)
    high, _ = room_world.scan(pose, z=0.30)
    assert low[0] > 1.0
    assert 0.2 < high[0] < 0.4


def test_lidar_free_does_not_mean_robot_fits(room_world):
    low = room_world.occupancy(0.18, 0.18)
    volume = room_world.occupancy(0.02, 0.34)
    assert not low[75, 37]
    assert volume[75, 37]
    assert "bed_frame" in room_world.contact_names([1.5, 3.0, 0.0])


def test_height_clearance_changes_contact_outcome():
    layout = copy.deepcopy(default_layout())
    layout["bed"]["under_clearance"] = 0.40
    raised = RoomWorld(layout)
    assert raised.contact_names([1.5, 3.0, 0.0]) == []
    assert not raised.occupancy(0.02, 0.34)[75, 37]


def test_planner_rejection_is_confirmed_by_mujoco(room_world):
    start, goal = [2.7, 0.95], [1.5, 3.0]
    low_path = plan(room_world.occupancy(0.18, 0.18), start, goal)
    assert low_path is not None
    assert path_contacts(room_world, low_path)["contact_frames"] > 0
    assert plan(room_world.occupancy(0.02, 0.34), start, goal) is None


def test_visible_safe_route_has_no_contact(room_world):
    route = plan(room_world.occupancy(0.02, 0.34), [2.7, 0.95], [1.6, 1.92])
    assert route is not None
    assert path_contacts(room_world, route)["contact_frames"] == 0


def test_wheel_replay_closes_and_encoder_bias_breaks_closure():
    points = np.array([[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]])
    truth, enc = drive_replay(points)
    assert truth[-1, :2] == pytest.approx(points[-1], abs=1e-12)
    biased = estimate_poses(enc * np.array([1, 1.02]), truth[0])
    assert np.linalg.norm(biased[-1, :2] - points[-1]) > 0.05
