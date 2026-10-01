"""Keep measured obstacle heights when checking the body's separate parts."""

import time
from dataclasses import replace
from itertools import pairwise

import numpy as np
from scipy.spatial import cKDTree

from embodied_learning.clearance_navigation import (
    ClearanceNavigationController,
    GradualClearanceField,
)
from embodied_learning.footprint_planning import FootprintField, footprint_route
from embodied_learning.path_tracking import path_frame


class HeightAwareField(GradualClearanceField):
    """Prior cells remain solid; measured endpoints check each real 3D part.

    The inherited A* uses XY only as a heuristic. Feasibility checks never
    collapse high returns into the wider base, or discard upper-body collisions.
    """

    def __init__(self, prior, points, rig, start):
        super().__init__(prior, np.empty((0, 2)), rig, start)
        self.points_xyz = np.asarray(points, float).reshape(-1, 3)
        self.points = self.points_xyz[:, :2]
        self.height_tree = cKDTree(self.points)
        self.relax_start = not self._part_free(np.asarray(start), rig)

    def _part_free(self, pose, rig):
        # Base encloses all three parts in XY in the registered platform.
        # An unknown-height prior wall remains conservatively blocking.
        self.rig = rig
        if not FootprintField._free(self, pose):
            return False
        ids = self.height_tree.query_ball_point(
            pose[:2], rig.radius + np.sqrt(2) * rig.margin_m
        )
        points = self.points_xyz[ids].copy()
        c, s = np.cos(pose[2]), np.sin(pose[2])
        points[:, :2] = (points[:, :2] - pose[:2]) @ np.array([[c, -s], [s, c]])
        pad = np.array([rig.margin_m, rig.margin_m, 0.0])
        for part in rig.parts:
            lo, hi = part[[0, 2, 4]] - pad, part[[1, 3, 5]] + pad
            if np.any(((points >= lo) & (points <= hi)).all(axis=1)):
                return False
        return True

    def _free(self, pose):
        margin = self.preferred_rig.margin_m
        if self.relax_start:
            margin *= min(1.0, np.linalg.norm(pose[:2] - self.start) / 0.2)
        try:
            return self._part_free(pose, replace(self.preferred_rig, margin_m=margin))
        finally:
            self.rig = self.preferred_rig


class HeightAwareNavigationController(ClearanceNavigationController):
    def make_plan(self, estimate, goal):
        started = time.perf_counter()
        points = np.array(list(self.voxels.values())).reshape(-1, 3)
        field = HeightAwareField(self.prior, points, self.planning_rig, estimate)
        if self.path is not None and len(self.path) > 1:
            info = path_frame(self.path, estimate)
            i = info["index"]
            a, b = self.path[i : i + 2]
            fraction = np.clip(
                (estimate[:2] - a) @ (b - a) / max(np.sum((b - a) ** 2), 1e-16), 0, 1
            )
            near = np.linalg.norm(estimate[:2] - (a + fraction * (b - a))) <= 0.05
            remaining = self.path[i:].copy()
            if near and field.free(estimate) and all(
                field.segment(a, b) for a, b in pairwise(remaining)
            ):
                self.planning_log.append({
                    "reason": "retained_original_curve", "expanded": 0,
                    "relaxed_preferred_start": field.relax_start,
                    "wall_s": time.perf_counter() - started,
                })
                return remaining
        path, info = footprint_route(field, estimate, goal)
        self.planning_log.append({
            **info, "relaxed_preferred_start": field.relax_start,
            "wall_s": time.perf_counter() - started,
        })
        return path
