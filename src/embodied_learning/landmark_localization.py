"""The unchanged lesson-18 geometry, independent of GUI, MuJoCo and ROS imports."""

import math

import numpy as np

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose, rotation, to_child

LANDMARKS = np.array([[0.0, 1.6], [2.6, 1.0], [1.6, -0.9]])
OBS_PERIOD_STEPS = 50
OBS_RANGE_STD_M = 0.01
OBS_BEARING_STD_RAD = 0.01


def inverse_pose(pose):
    """Inverse SE(2), preserving the original unwrapped-angle convention."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [x, y, yaw] pose")
    shift = -(rotation(pose[2]).T @ pose[:2])
    return np.array([shift[0], shift[1], -pose[2]])


def bearing_reading(landmark, sensor_pose):
    """Range and bearing IN THE SENSOR FRAME, including installation rotation."""
    point = to_child(sensor_pose, np.asarray(landmark, dtype=float))
    return float(np.linalg.norm(point)), float(math.atan2(point[1], point[0]))


def observe(pose, landmarks, rng, range_std=OBS_RANGE_STD_M, bearing_std=OBS_BEARING_STD_RAD):
    """Synthetic sensor only: truth generates readings, not estimator inputs."""
    sensor = compose(pose, SENSOR_IN_BODY)
    readings = np.empty((len(landmarks), 2))
    for index, landmark in enumerate(landmarks):
        distance, bearing = bearing_reading(landmark, sensor)
        readings[index] = [
            distance + rng.normal(0.0, range_std),
            bearing + rng.normal(0.0, bearing_std),
        ]
    return readings


def solve_pose(readings, landmarks):
    """Unchanged 2D rigid least-squares fit, then invert sensor installation."""
    readings = np.asarray(readings, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)
    if readings.ndim != 2 or readings.shape != (len(landmarks), 2):
        raise ValueError("Readings must match landmarks")
    if len(landmarks) < 2 or not np.isfinite(readings).all() or not np.isfinite(landmarks).all():
        raise ValueError("Need at least two finite landmarks")
    if np.any(readings[:, 0] <= 0):
        raise ValueError("Range readings must be positive")
    polar = np.column_stack(
        [readings[:, 0] * np.cos(readings[:, 1]), readings[:, 0] * np.sin(readings[:, 1])]
    )
    center_z, center_l = polar.mean(axis=0), landmarks.mean(axis=0)
    zz, ll = polar - center_z, landmarks - center_l
    dot = float(zz[:, 0] @ ll[:, 0] + zz[:, 1] @ ll[:, 1])
    cross = float(zz[:, 0] @ ll[:, 1] - zz[:, 1] @ ll[:, 0])
    yaw = math.atan2(cross, dot)
    translation = center_l - rotation(yaw) @ center_z
    sensor_pose = [translation[0], translation[1], yaw]
    return compose(sensor_pose, inverse_pose(SENSOR_IN_BODY))
