"""Calibrate the finger-tip midpoint using geometry and the measured opening."""

from __future__ import annotations

import mujoco
import numpy as np

from embodied_learning.so101 import HOME, grasp_rotation


class FingerMidpointCalibration:
    def __init__(self, sim):
        self.model = sim.model
        self.data = mujoco.MjData(self.model)
        self.site = sim.site
        self.fixed = self.model.geom("fixed_jaw_sph_tip1").id
        self.moving = self.model.geom("moving_jaw_sph_tip1").id

    def local_midpoint(self, opening):
        if not np.isfinite(opening):
            raise ValueError("Opening must be finite")
        self.data.qpos[:6] = np.r_[HOME[:5], opening]
        mujoco.mj_forward(self.model, self.data)
        rotation = self.data.site_xmat[self.site].reshape(3, 3)
        midpoint = (self.data.geom_xpos[self.fixed] + self.data.geom_xpos[self.moving]) / 2
        return rotation.T @ (midpoint - self.data.site_xpos[self.site])

    def tcp_target(self, center, opening, tilt):
        """Compensate XY only; moving-tip height must not lower the arm into the table."""
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Expected finite center XYZ")
        local = self.local_midpoint(opening)
        target = center.copy()
        for _ in range(8):
            offset = grasp_rotation(target, tilt) @ local
            target[:2] = center[:2] - offset[:2]
        return target, grasp_rotation(target, tilt)
