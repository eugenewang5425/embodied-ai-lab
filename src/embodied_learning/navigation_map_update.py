"""Evidence-only local occupancy updates and speed adapters for rounds 7/8.

Controllers receive an estimated pose, a possibly wrong prior and sensor arrays.
No actual scene geometry or evaluation mask is supplied here.
"""

import numpy as np
from scipy.ndimage import minimum_filter

from embodied_learning.braking_control import BrakingController
from embodied_learning.navigation_geometry import (
    body_points,
    depth_points,
    lidar_obstacle_points,
    project,
    world_points,
)

RESOLUTION_M = 0.1
FREE_CONFIRMATIONS = 3
FREE_DEPTH_BUDGET_M = 0.12


class SpeedController(BrakingController):
    def __init__(self, prior, rig, cap):
        super().__init__(prior, rig, "coupled")
        if not 0 < cap <= 0.6:
            raise ValueError("speed cap must be in (0, 0.6]")
        self.fixed_cap = cap

    def tracking_command(self, estimate, velocity):
        desired = super().tracking_command(estimate, velocity)
        old = desired[0]
        desired[0] = min(old, self.fixed_cap)
        raw = self.tracking_log[-1]["angular_raw"]
        raw += (desired[0] - old) * self.tracking_log[-1]["curvature_m_inv"]
        desired[1] = np.clip(raw, -1, 1)
        self.latest.update(speed_cap=self.fixed_cap, nominal=desired.copy())
        return desired


class EvidenceMap:
    """Small 3D voxel map: positive endpoints, conservative visible-free votes.

    This is a sampled visibility model, not a guarantee that every subvoxel is
    empty. Invalidation resets consecutive votes; there is no age/TTL removal.
    """

    def __init__(self, prior_boxes, rig, clear, trusted_no_return=False):
        self.rig, self.clear = rig, clear
        self.trusted_no_return = trusted_no_return
        self.shape = (120, 120, 6)
        yy, xx, zz = np.indices(self.shape)
        self.centers = np.stack((xx, yy, zz), axis=-1).astype(float)
        self.centers = (self.centers + 0.5) * RESOLUTION_M
        self.occupied = np.zeros(self.shape, bool)
        for box in np.asarray(prior_boxes).reshape(-1, 6):
            lo, hi = box[[0, 2, 4]], box[[1, 3, 5]]
            self.occupied |= ((self.centers >= lo) & (self.centers <= hi)).all(axis=-1)
        self.votes = np.zeros(self.shape, np.uint8)
        self.points = self.centers.copy()
        self.last = {"added": 0, "removed": 0, "free_evidence": 0}

    @property
    def grid(self):
        return self.occupied.any(axis=2)

    def update(self, estimate, ranges, hits, depth, valid):
        rig = self.rig
        endpoints = np.vstack(
            (lidar_obstacle_points(ranges, hits, rig), depth_points(depth, valid, rig))
        )
        world = world_points(endpoints, estimate)
        cell = np.floor(world / RESOLUTION_M).astype(int)
        keep = ((cell >= 0) & (cell < np.array(self.shape)[[1, 0, 2]])).all(axis=1)
        cell, world = cell[keep], world[keep]
        ix = (cell[:, 1], cell[:, 0], cell[:, 2])
        seen = np.zeros(self.shape, bool)
        seen[ix] = True
        before = self.occupied.copy()
        added = int((seen & ~before).sum())
        self.occupied |= seen
        self.points[ix] = world
        self.votes[seen] = 0

        free = np.zeros(self.shape, bool)
        if self.clear:
            indices = np.nonzero(self.occupied)
            centers = self.centers[indices]
            local = body_points(centers, estimate)
            uv, front = project(local, rig)
            pixel = np.rint(np.nan_to_num(uv, nan=-1e6, posinf=1e6, neginf=-1e6)).astype(int)
            u, v = pixel.T
            interior = front & (u >= 1) & (u < rig.width - 1) & (v >= 1) & (v < rig.height - 1)
            # The analytic fixture emits exactly max_range with valid=False
            # for a measured no-hit ray. Missing/invalid camera data have no
            # such guarantee; external callers default to rejecting them all.
            no_return = (~valid) & np.isclose(depth, rig.max_range_m, atol=1e-10, rtol=0)
            usable = valid | (no_return & self.trusted_no_return)
            measured = np.where(usable & np.isfinite(depth), depth, -np.inf)
            conservative = minimum_filter(measured, size=3, mode="constant", cval=-np.inf)
            ids = np.flatnonzero(interior)
            optical_depth = local[:, 0] - rig.camera_x_m
            passed = conservative[v[ids], u[ids]] > optical_depth[ids] + FREE_DEPTH_BUDGET_M
            selected = ids[passed]
            free[tuple(a[selected] for a in indices)] = True
            free &= ~seen
        self.votes[~free] = 0
        self.votes[free] = np.minimum(self.votes[free] + 1, FREE_CONFIRMATIONS)
        remove = free & (self.votes >= FREE_CONFIRMATIONS)
        self.occupied[remove] = False
        self.votes[remove] = 0
        self.last = {
            "added": added,
            "removed": int(remove.sum()),
            "free_evidence": int(free.sum()),
        }
        return self.last


class MapController(SpeedController):
    def __init__(self, prior_boxes, rig, clear):
        self.evidence = EvidenceMap(prior_boxes, rig, clear, trusted_no_return=True)
        super().__init__(self.evidence.grid, rig, 0.2)

    def observe(self, estimate, ranges, hits, depth, valid):
        self.evidence.update(estimate, ranges, hits, depth, valid)
        self.grid = self.evidence.grid
        # The planner must use the current map, not the original immutable mask.
        self.prior = self.grid.copy()
        ids = np.nonzero(self.evidence.occupied)
        points = self.evidence.points[ids]
        self.voxels = {tuple(i): p for i, p in zip(np.column_stack(ids), points, strict=True)}
        local = body_points(points, estimate)
        return local[np.linalg.norm(local[:, :2], axis=1) < 2.5]
