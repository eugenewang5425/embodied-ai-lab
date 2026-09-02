"""Environment constructors with project-specific compatibility handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco.inverted_pendulum_v5 import InvertedPendulumEnv

ENVIRONMENT_ID = "InvertedPendulum-v5"


class UnicodeSafeInvertedPendulumEnv(InvertedPendulumEnv):
    """Load the small MJCF model through Python when its Windows path is non-ASCII.

    MuJoCo's Windows path loader cannot open this model when the virtual environment
    lives below the current Chinese project path. Python can read the file correctly,
    so passing its contents to MuJoCo avoids moving or renaming the project.
    """

    def _initialize_simulation(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        xml = Path(self.fullpath).read_text(encoding="utf-8")
        model = mujoco.MjModel.from_xml_string(xml)
        return model, mujoco.MjData(model)


def make_inverted_pendulum_environment(
    *, render_mode: str | None = None, max_episode_steps: int = 1000, **kwargs: Any
) -> gym.Env:
    """Create the first MuJoCo task and retain Gymnasium's time-limit behavior."""
    env = UnicodeSafeInvertedPendulumEnv(render_mode=render_mode, **kwargs)
    return gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
