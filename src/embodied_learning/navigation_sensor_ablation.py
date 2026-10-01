"""Sensor-only adapters: identical progress, planning, braking and real body.

Both groups archive the same kinds of raw measurements. The LiDAR-only group
must not use camera depth in either map updates or downstream safety checks.
"""

import numpy as np

from embodied_learning.navigation_progress import ProgressMapController


def controller_inputs(method, depth, valid):
    if method == "lidar_only":
        return np.full_like(depth, np.nan), np.zeros_like(valid, dtype=bool)
    if method == "lidar_depth":
        return depth, valid
    raise ValueError(f"unknown sensor ablation: {method}")


class SensorMapController(ProgressMapController):
    def __init__(self, prior_boxes, rig, method):
        self.sensor_method = method
        controller_inputs(method, np.empty((0, 0)), np.empty((0, 0), bool))
        super().__init__(prior_boxes, rig, True)
        self.initial_prior_cells = self.prior.copy()

    def observe(self, estimate, ranges, hits, depth, valid):
        depth, valid = controller_inputs(self.sensor_method, depth, valid)
        points = super().observe(estimate, ranges, hits, depth, valid)
        # A measured endpoint keeps its metric coordinate for body checks.
        # Promoting its storage cell to a full 10cm obstacle double-enlarges it.
        # Only still-occupied original prior cells retain square extent. New
        # returns remain in self.voxels and constrain planning and all guards.
        self.prior = self.initial_prior_cells & self.grid
        return points
