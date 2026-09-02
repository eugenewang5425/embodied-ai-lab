"""Planar, ideal no-slip kinematics; wheel-speed inputs, NOT motor dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def finite_vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite components")
    return result


def rotation(yaw):
    if not math.isfinite(yaw):
        raise ValueError("Yaw must be finite")
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_parent(parent_from_child, point_child):
    """p_parent = translation + R(yaw) @ p_child. Pose is [x, y, yaw]."""
    pose = finite_vector(parent_from_child, 3)
    return pose[:2] + rotation(pose[2]) @ finite_vector(point_child, 2)


def to_child(parent_from_child, point_parent):
    pose = finite_vector(parent_from_child, 3)
    return rotation(pose[2]).T @ (finite_vector(point_parent, 2) - pose[:2])


def compose(parent_from_middle, middle_from_child):
    first, second = finite_vector(parent_from_middle, 3), finite_vector(middle_from_child, 3)
    return np.r_[to_parent(first, second[:2]), first[2] + second[2]]


@dataclass(frozen=True)
class DriveGeometry:
    radius_m: float = 0.05
    track_m: float = 0.30

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.radius_m, self.track_m)):
            raise ValueError("Wheel radius and track width must be finite and positive")

    def body_velocity(self, wheels_rad_s):
        """Input order LEFT, RIGHT; both positive means rolling forward.

        Body x is forward, y is left. Positive yaw is counterclockwise.
        Returns forward velocity [m/s], yaw rate [rad/s]. Lateral speed is zero.
        """
        left, right = self.radius_m * finite_vector(wheels_rad_s, 2)
        return np.array([(left + right) / 2, (right - left) / self.track_m])


def integrate_pose(pose, body_velocity, dt):
    """Exact SE(2) step for a constant forward speed and yaw rate over dt.

    Heading stays unwrapped; this avoids discontinuities in saved plots.
    np.sinc(x) = sin(pi*x)/(pi*x), including its continuous value at zero.
    """
    pose = finite_vector(pose, 3)
    v, omega = finite_vector(body_velocity, 2)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    delta = omega * dt
    distance = v * dt * np.sinc(delta / (2 * np.pi))
    direction = pose[2] + delta / 2
    return pose + np.array([distance * np.cos(direction), distance * np.sin(direction), delta])
