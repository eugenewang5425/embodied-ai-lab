"""Lesson 69 round two: path geometry feedback using estimated pose only."""

import time
from itertools import pairwise

import numpy as np

from embodied_learning.footprint_planning import (
    FootprintController,
    FootprintField,
    footprint_route,
)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def path_frame(path, pose):
    """Closest polyline projection and local tangent/curvature, in metres/radians."""
    path = np.asarray(path)
    edge = np.diff(path, axis=0)
    length = np.linalg.norm(edge, axis=1)
    valid = length > 1e-8
    if not valid.any():
        return {"cross_m": 0.0, "heading_rad": pose[2], "curvature_m_inv": 0.0, "index": 0}
    t = np.clip(np.sum((pose[:2] - path[:-1]) * edge, axis=1) / np.maximum(length**2, 1e-16), 0, 1)
    projections = path[:-1] + t[:, None] * edge
    distances = np.linalg.norm(projections - pose[:2], axis=1)
    distances[~valid] = np.inf
    i = int(np.argmin(distances))
    headings = np.arctan2(edge[:, 1], edge[:, 0])
    station = np.r_[0.0, np.cumsum(length)]
    midpoint = (station[:-1] + station[1:]) / 2
    angle = np.unwrap(headings[valid])
    coordinate = station[i] + t[i] * length[i]
    heading = np.interp(coordinate, midpoint[valid], angle)
    gradient = np.gradient(angle, midpoint[valid]) if valid.sum() > 1 else np.zeros(1)
    curvature = np.interp(coordinate, midpoint[valid], gradient)
    normal = np.array([-np.sin(heading), np.cos(heading)])
    return {
        "cross_m": float((pose[:2] - projections[i]) @ normal),
        "heading_rad": float(wrap(heading)),
        "curvature_m_inv": float(curvature),
        "index": i,
    }


class TangentController(FootprintController):
    """Same speed rule and safety pipeline; only nominal angular steering changes."""

    def __init__(self, prior, rig, method):
        super().__init__(prior, rig, method, "rectangle")
        self.tracking_log = []

    def tracking_command(self, estimate, velocity):
        command = super().tracking_command(estimate, velocity)
        info = path_frame(self.path, estimate)
        heading_error = float(wrap(estimate[2] - info["heading_rad"]))
        desired_heading = info["heading_rad"] - np.arctan2(2.0 * info["cross_m"], 0.6)
        angular = command[0] * info["curvature_m_inv"] + 2.0 * wrap(desired_heading - estimate[2])
        command[1] = np.clip(angular, -1.0, 1.0)
        self.tracking_log.append(
            {
                **info,
                "heading_error_rad": heading_error,
                "angular_raw": float(angular),
            }
        )
        return command


class CurveRetentionMixin:
    """Trim only the passed prefix, retaining checked curve geometry."""

    def make_plan(self, estimate, goal):
        started = time.perf_counter()
        points = np.array(list(self.voxels.values())).reshape(-1, 3)[:, :2]
        field = FootprintField(self.prior, points, self.rig, self.shape)
        if self.path is not None and len(self.path) > 1:
            info = path_frame(self.path, estimate)
            i = info["index"]
            a, b = self.path[i : i + 2]
            t = np.clip((estimate[:2] - a) @ (b - a) / max(np.sum((b - a) ** 2), 1e-16), 0, 1)
            near = np.linalg.norm(estimate[:2] - (a + t * (b - a))) <= 0.05
            remaining = self.path[i:].copy()
            if (
                near
                and field.free(estimate)
                and all(field.segment(a, b) for a, b in pairwise(remaining))
            ):
                self.planning_log.append(
                    {
                        "reason": "retained_original_curve",
                        "expanded": 0,
                        "wall_s": time.perf_counter() - started,
                    }
                )
                return remaining
        path, info = footprint_route(field, estimate, goal)
        self.planning_log.append({**info, "wall_s": time.perf_counter() - started})
        return path


class RetainedCurveController(CurveRetentionMixin, FootprintController):
    pass


class RetainedTangentController(CurveRetentionMixin, TangentController):
    pass
