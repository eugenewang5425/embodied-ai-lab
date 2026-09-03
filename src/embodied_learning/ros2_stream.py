"""Timestamp contracts and online lesson-19 fusion; no ROS/GUI/ground-truth input."""

import numpy as np

from embodied_learning.differential_drive import compose, finite_vector
from embodied_learning.landmark_localization import (
    LANDMARKS,
    OBS_PERIOD_STEPS,
    inverse_pose,
    solve_pose,
)
from embodied_learning.odometry import estimate_poses
from embodied_learning.pose_fusion import estimate_with_resets

DT_NS = 40_000_000
EPOCH_NS = 1_000_000_000  # Avoid ROS Time(0), which means "latest" to tf2.


def frame_stamp(frame):
    if type(frame) is not int or frame < 0:
        raise ValueError("Frame must be a nonnegative integer")
    return divmod(EPOCH_NS + frame * DT_NS, 1_000_000_000)


def stamp_frame(sec, nanosec):
    if type(sec) is not int or type(nanosec) is not int or not 0 <= nanosec < 1_000_000_000:
        raise ValueError("Invalid ROS timestamp")
    elapsed = sec * 1_000_000_000 + nanosec - EPOCH_NS
    if elapsed < 0 or elapsed % DT_NS:
        raise ValueError("Timestamp is not on the lesson's 0.04 s grid")
    return elapsed // DT_NS


class StreamingFusion:
    """Wait for matching timestamps, not callback order. Missing data stalls visibly.

    A bounded buffer supports arrivals out of order. At known observation frames
    both messages are required, so a late landmark never corrects the wrong pose.
    No delayed-measurement smoothing, interpolation or clock sync is implemented.
    """

    def __init__(self):
        self.next_frame = 0
        self.encoders, self.observations = {}, {}
        self.previous_encoders = None
        self.odom = np.zeros(3)
        self.fused = np.zeros(3)

    def _check_frame(self, frame, buffer):
        if type(frame) is not int or not self.next_frame <= frame <= self.next_frame + 128:
            raise ValueError("Stale, invalid or excessively future frame")
        if frame in buffer:
            raise ValueError("Duplicate frame")

    def add_encoders(self, frame, angles):
        self._check_frame(frame, self.encoders)
        self.encoders[frame] = finite_vector(angles, 2).copy()
        return self._drain()

    def add_observation(self, frame, readings):
        self._check_frame(frame, self.observations)
        if frame == 0 or frame % OBS_PERIOD_STEPS:
            raise ValueError("Observation outside the sampling schedule")
        # Validate and solve only from this measurement plus known control points.
        self.observations[frame] = solve_pose(readings, LANDMARKS)
        return self._drain()

    def _drain(self):
        ready = []
        while self.next_frame in self.encoders:
            frame = self.next_frame
            correction = frame > 0 and frame % OBS_PERIOD_STEPS == 0
            if correction and frame not in self.observations:
                break
            angles = self.encoders.pop(frame)
            if frame:
                pair = np.array([self.previous_encoders, angles])
                self.odom = estimate_poses(pair, self.odom)[-1]
                observed = self.observations.pop(frame) if correction else None
                posterior, _ = estimate_with_resets(
                    pair,
                    [1] if correction else [],
                    np.array([observed]) if correction else np.empty((0, 3)),
                    self.fused,
                )
                self.fused = posterior[-1]
            self.previous_encoders = angles
            ready.append(
                {
                    "frame": frame,
                    "odom": self.odom.copy(),
                    "fused": self.fused.copy(),
                    "map_to_odom": compose(self.fused, inverse_pose(self.odom)),
                    "correction": correction,
                }
            )
            self.next_frame += 1
        return ready
