"""Calibrate actual jaw surface sections; gate lifting on causal solved contacts."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from embodied_learning.so101 import HOME, grasp_rotation


class SurfaceBandCalibration:
    """Inner collision surfaces at a declared TCP section, not visual mesh tips.

    Mesh collisions use their convex hull. Primitive collisions use exact ray
    intersections. The calibration data is independent of live physical state.
    Opening above the preparation limit uses that limit's contact reference;
    it does not pretend both open fingers intersect the selected section.
    """

    def __init__(self, sim, band_x=0.0, preparation_opening=0.45):
        if not np.isfinite([band_x, preparation_opening]).all() or preparation_opening <= 0:
            raise ValueError("Finite section and positive preparation opening required")
        self.model, self.site = sim.model, sim.site
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:6] = HOME
        self.band_x, self.preparation_opening = band_x, preparation_opening
        self.geoms = {0: [], 1: []}
        self.hulls = {}
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name
            if not name.startswith(("fixed_jaw_", "moving_jaw_")):
                continue
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):
                continue
            self.geoms[int(name.startswith("moving"))].append(g)
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = self.model.geom_dataid[g]
                a, n = self.model.mesh_vertadr[mesh], self.model.mesh_vertnum[mesh]
                self.hulls[g] = ConvexHull(self.model.mesh_vert[a : a + n]).equations
        self.openings = np.linspace(0, preparation_opening, 131)
        self.offsets = np.array([self.exact_midpoint(op) for op in self.openings])

    def _interval(self, g, origin, direction):
        if g in self.hulls:
            mat = self.data.geom_xmat[g].reshape(3, 3)
            p = mat.T @ (origin - self.data.geom_xpos[g])
            v = mat.T @ direction
            eq = self.hulls[g]
            den, num = eq[:, :3] @ v, -(eq[:, :3] @ p + eq[:, 3])
            if np.any((np.abs(den) < 1e-10) & (num < -1e-10)):
                return None
            lower, upper = -np.inf, np.inf
            if np.any(den < -1e-10):
                lower = float(np.max(num[den < -1e-10] / den[den < -1e-10]))
            if np.any(den > 1e-10):
                upper = float(np.min(num[den > 1e-10] / den[den > 1e-10]))
            return (lower, upper) if lower <= upper else None
        # Starting beyond the jaw bounds makes the two exact ray hits the
        # entry and exit of a convex primitive along the same line.
        span = 0.3
        params = (self.data.geom_xpos[g], self.data.geom_xmat[g], self.model.geom_size[g])
        left = mujoco.mju_rayGeom(
            *params, origin - span * direction, direction, self.model.geom_type[g]
        )
        right = mujoco.mju_rayGeom(
            *params, origin + span * direction, -direction, self.model.geom_type[g]
        )
        return (left - span, span - right) if left >= 0 and right >= 0 else None

    def exact_midpoint(self, opening):
        if not np.isfinite(opening) or not 0 <= opening <= self.preparation_opening:
            raise ValueError("Opening outside calibrated contact range")
        self.data.qpos[:6] = np.r_[HOME[:5], opening]
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site].reshape(3, 3)
        origin = self.data.site_xpos[self.site] + rotation @ [self.band_x, 0, 0]
        direction = rotation[:, 2]
        edges = []
        for channel, geoms in self.geoms.items():
            intervals = [self._interval(g, origin, direction) for g in geoms]
            intervals = [value for value in intervals if value is not None]
            if not intervals:
                raise ValueError("No jaw intersects the declared contact section")
            edges.append(
                max(hi for _, hi in intervals) if channel == 0 else min(lo for lo, _ in intervals)
            )
        return np.array([self.band_x, 0.0, np.mean(edges)])

    def local_midpoint(self, opening):
        if not np.isfinite(opening):
            raise ValueError("Opening must be finite")
        clipped = np.clip(opening, 0, self.preparation_opening)
        return np.array([self.band_x, 0, np.interp(clipped, self.openings, self.offsets[:, 2])])

    def tcp_target(self, center, opening, tilt):
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Expected finite contact reference XYZ")
        local = self.local_midpoint(opening)
        target = center.copy()
        for _ in range(8):
            target = center - grasp_rotation(target, tilt) @ local
        return target, grasp_rotation(target, tilt)


class BilateralContactGate:
    """Read solved forces for elapsed past intervals; no object pose input."""

    def __init__(self, threshold_n=0.05, duration_s=0.2):
        if min(threshold_n, duration_s) <= 0 or not np.isfinite([threshold_n, duration_s]).all():
            raise ValueError("Positive finite contact gate thresholds required")
        self.threshold_n, self.duration_s, self.elapsed = threshold_n, duration_s, 0.0

    def update(self, forces, dt):
        forces = np.asarray(forces, float)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite elapsed interval required")
        bilateral = (
            forces.ndim == 2
            and forces.shape[0] > 0
            and forces.shape[1] == 2
            and np.isfinite(forces).all()
            and np.all(forces > self.threshold_n)
        )
        self.elapsed = self.elapsed + dt if bilateral else 0.0
        return self.ready

    @property
    def ready(self):
        return self.elapsed >= self.duration_s - 1e-10
