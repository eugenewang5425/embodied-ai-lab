from __future__ import annotations

import numpy as np

from embodied_learning.controllers import PDController


def test_pd_controller_uses_angle_and_angular_velocity_feedback() -> None:
    controller = PDController(proportional_gain=10.0, derivative_gain=2.0)

    action = controller.action(np.array([0.0, 0.1, 0.0, 0.2]))

    assert action.dtype == np.float32
    assert action.shape == (1,)
    assert action[0] == np.float32(1.4)


def test_pd_controller_clips_force_to_environment_limit() -> None:
    controller = PDController(proportional_gain=100.0, derivative_gain=10.0)

    positive_action = controller.action(np.array([0.0, 1.0, 0.0, 1.0]))
    negative_action = controller.action(np.array([0.0, -1.0, 0.0, -1.0]))

    assert positive_action[0] == np.float32(3.0)
    assert negative_action[0] == np.float32(-3.0)
