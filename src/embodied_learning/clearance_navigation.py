"""Choose routes with observation/tracking room; keep actual contact scoring."""

import time
from dataclasses import replace
from itertools import pairwise

import numpy as np

from embodied_learning.footprint_planning import FootprintField, footprint_route
from embodied_learning.path_tracking import path_frame
from embodied_learning.physical_braking import PhysicalBrakingController


class GradualClearanceField(FootprintField):
    """The current body must be clear; preferred padding ramps over 20cm.

    A robot already closer than the preferred padding still needs a way out.
    Every pose checks the actual body; only the extra route padding varies.
    """

    def __init__(self, prior, points, rig, start):
        super().__init__(prior, points, rig)
        self.start = np.asarray(start[:2]).copy()
        self.preferred_rig = rig
        self.relax_start = not super()._free(np.asarray(start))

    def _free(self, pose):
        margin = self.preferred_rig.margin_m
        if self.relax_start:
            margin *= min(1.0, np.linalg.norm(pose[:2] - self.start) / 0.2)
        self.rig = replace(self.preferred_rig, margin_m=margin)
        try:
            return super()._free(pose)
        finally:
            self.rig = self.preferred_rig


class ClearanceNavigationController(PhysicalBrakingController):
    def __init__(self, prior_boxes, rig, planning_margin_m=0.02):
        super().__init__(prior_boxes, rig)
        self.planning_rig = replace(rig, margin_m=planning_margin_m)

    def make_plan(self, estimate, goal):
        started = time.perf_counter()
        points = np.array(list(self.voxels.values())).reshape(-1, 3)[:, :2]
        field = GradualClearanceField(self.prior, points, self.planning_rig, estimate)
        if self.path is not None and len(self.path) > 1:
            info = path_frame(self.path, estimate)
            i = info["index"]
            a, b = self.path[i : i + 2]
            fraction = np.clip(
                (estimate[:2] - a) @ (b - a) / max(np.sum((b - a) ** 2), 1e-16), 0, 1
            )
            near = np.linalg.norm(estimate[:2] - (a + fraction * (b - a))) <= 0.05
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
                        "relaxed_preferred_start": field.relax_start,
                        "wall_s": time.perf_counter() - started,
                    }
                )
                return remaining
        path, info = footprint_route(field, estimate, goal)
        self.planning_log.append(
            {
                **info,
                "relaxed_preferred_start": field.relax_start,
                "wall_s": time.perf_counter() - started,
            }
        )
        return path
