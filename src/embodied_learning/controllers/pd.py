"""A minimal proportional-derivative controller for the inverted pendulum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PDController:
    """Turn joint angle and angular velocity into an actuator command.

    `maximum_force` is a retained legacy parameter name: it bounds the input u,
    not newtons. The stock model maps cart actuator force = 100 * u N.
    This legacy baseline deliberately keeps its original zero-angle target.
    """

    proportional_gain: float
    derivative_gain: float
    maximum_force: float = 3.0

    def action(self, observation: NDArray[np.floating]) -> NDArray[np.float32]:
        pole_angle = float(observation[1])
        pole_angular_velocity = float(observation[3])
        raw_force = (
            self.proportional_gain * pole_angle + self.derivative_gain * pole_angular_velocity
        )
        clipped_force = np.clip(raw_force, -self.maximum_force, self.maximum_force)
        return np.array([clipped_force], dtype=np.float32)
