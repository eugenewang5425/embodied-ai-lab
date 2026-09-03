"""Bounded point-goal feedback using estimated pose only (not a path planner)."""

import math
from dataclasses import asdict, dataclass

import numpy as np

from embodied_learning.differential_drive import DriveGeometry, finite_vector
from embodied_learning.odometry import heading_error


@dataclass(frozen=True)
class GoalConfig:
    distance_gain: float = 0.7
    heading_gain: float = 2.0
    max_speed_m_s: float = 0.25
    max_yaw_rad_s: float = 1.2
    max_wheel_rad_s: float = 6.0
    estimated_stop_radius_m: float = 0.02
    true_acceptance_radius_m: float = 0.03  # Evaluator only; never used by policy.
    settle_seconds: float = 0.4

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Goal parameters must be finite and positive")


DEFAULT_CONFIG = GoalConfig()
GEOMETRY = DriveGeometry()


def goal_command(estimated_pose, goal_xy, config=DEFAULT_CONFIG):
    """Return the command for the NEXT interval, never read actual position.

    Rotate first when facing >60 degrees away; otherwise steer and approach.
    Jointly scaling wheels preserves requested curvature when speed is limited.
    Goal orientation is unspecified; only the centre position is controlled.
    """
    pose, goal = finite_vector(estimated_pose, 3), finite_vector(goal_xy, 2)
    delta = goal - pose[:2]
    distance = float(np.linalg.norm(delta))
    if distance <= config.estimated_stop_radius_m:
        return {"wheels": np.zeros(2), "distance": distance, "angle": 0.0, "mode": "settling"}
    angle = float(heading_error(math.atan2(delta[1], delta[0]), pose[2]))
    turning = abs(angle) > math.pi / 3
    speed = (
        0.0
        if turning
        else min(config.max_speed_m_s, config.distance_gain * distance) * math.cos(angle)
    )
    yaw = float(np.clip(config.heading_gain * angle, -config.max_yaw_rad_s, config.max_yaw_rad_s))
    wheels = (
        np.array([speed - GEOMETRY.track_m * yaw / 2, speed + GEOMETRY.track_m * yaw / 2])
        / GEOMETRY.radius_m
    )
    wheels /= max(1.0, float(np.max(np.abs(wheels))) / config.max_wheel_rad_s)
    return {
        "wheels": wheels,
        "distance": distance,
        "angle": angle,
        "mode": "turning" if turning else "driving",
    }
