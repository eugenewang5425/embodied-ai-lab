"""Encoder-only planar dead reckoning; no truth, commands or landmark input."""

import math

import numpy as np

from embodied_learning.differential_drive import DriveGeometry, finite_vector

DEFAULT_GEOMETRY = DriveGeometry()


def encoder_array(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < 1 or not np.isfinite(value).all():
        raise ValueError("Expected at least one finite [left, right] encoder sample")
    return value


def scaled_encoder_readings(wheel_angles_rad, right_scale):
    """Synthetic calibration error; does not modify physical rotation or commands.

    Cumulative readings are continuous, signed radians (no modulo/count quantization).
    Both positive and negative right-wheel increments are multiplied by the same factor.
    """
    if not math.isfinite(right_scale) or right_scale <= 0:
        raise ValueError("Encoder scale must be finite and positive")
    return encoder_array(wheel_angles_rad) * np.array([1.0, right_scale])


def estimate_poses(encoder_angles_rad, initial_pose=(0, 0, 0), geometry=DEFAULT_GEOMETRY):
    """Integrate successive measured wheel-angle differences, assuming no slip.

    Requires a known starting pose, wheel radius and track width. Cannot discover
    absolute position from encoders, correct calibration, or detect loop closure.
    """
    readings = encoder_array(encoder_angles_rad)
    poses = np.empty((len(readings), 3))
    poses[0] = finite_vector(initial_pose, 3)
    for i, delta in enumerate(np.diff(readings, axis=0)):
        left, right = delta * geometry.radius_m
        ds = (left + right) / 2
        dtheta = (right - left) / geometry.track_m
        direction = poses[i, 2] + dtheta / 2
        distance = ds * np.sinc(dtheta / (2 * np.pi))
        poses[i + 1] = poses[i] + [
            distance * np.cos(direction),
            distance * np.sin(direction),
            dtheta,
        ]
    return poses


def heading_error(estimated, true):
    """Signed shortest angular difference, radians; poses themselves stay unwrapped."""
    delta = np.asarray(estimated) - np.asarray(true)
    return np.arctan2(np.sin(delta), np.cos(delta))
