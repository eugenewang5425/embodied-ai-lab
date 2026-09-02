from __future__ import annotations

import numpy as np

from embodied_learning.environments import make_inverted_pendulum_environment


def test_mujoco_environment_can_step_without_rendering() -> None:
    env = make_inverted_pendulum_environment()
    observation, _ = env.reset(seed=7)

    try:
        assert np.asarray(observation).shape == env.observation_space.shape
        for _ in range(5):
            action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            observation, reward, terminated, truncated, _ = env.step(action)
            assert np.all(np.isfinite(observation))
            assert np.isfinite(reward)
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
