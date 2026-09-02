"""Full-rotation cart-pole task and energy-shaping / LQR hybrid feedback.

This task is independent of the original +/-0.2 rad balance benchmark.
Angles are measured from physical upright and wrapped only for feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from embodied_learning.controllers.lqr import LQRDesign, design_lqr
from embodied_learning.environments import UnicodeSafeInvertedPendulumEnv

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "cartpole_swingup.xml"
SAFE_CART_POSITION = 2.4


def wrap_angle(angle):
    """Physical orientation in [-pi, pi), without losing raw simulator rotations."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class SwingupEnv(UnicodeSafeInvertedPendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(xml_file=str(MODEL_PATH), **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        state = self._get_obs()
        reason = ""
        if not np.isfinite(state).all():
            reason = "nonfinite_state"
        elif abs(state[0]) >= SAFE_CART_POSITION:
            reason = "cart_safety_boundary"
        elif abs(state[2]) > 20 or abs(state[3]) > 40:
            reason = "velocity_safety_boundary"
        elif any(
            self.data.warning[i].number
            for i in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
        ):
            reason = "numerical_warning"
        if self.render_mode == "human":
            self.render()
        return state, float(not reason), bool(reason), False, {"failure_reason": reason}


def make_swingup_environment(max_episode_steps: int = 1000, **kwargs):
    return gym.wrappers.TimeLimit(SwingupEnv(**kwargs), max_episode_steps=max_episode_steps)


@dataclass(frozen=True)
class SwingupParameters:
    energy_gain: float = 1.2
    position_gain: float = 6.0
    velocity_gain: float = 3.0
    acceleration_limit: float = 20.0
    capture_angle: float = 0.3
    capture_omega: float = 2.0
    release_angle: float = 0.5
    kick_acceleration: float = 4.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("swing-up parameters must be finite and positive")
        if not self.capture_angle < self.release_angle < np.pi / 2:
            raise ValueError("need capture_angle < release_angle < pi/2")


class HybridSwingupController:
    """Feedback sees only state. It never receives future/current disturbance forces."""

    def __init__(
        self, model: mujoco.MjModel, design: LQRDesign, parameters: SwingupParameters | None = None
    ):
        self.model = model
        self.design = design
        self.parameters = parameters or SwingupParameters()
        self.data = mujoco.MjData(model)  # private nominal model, not rollout data
        self.mass_matrix = np.zeros((model.nv, model.nv))
        self.mode = "swingup"
        self.last_energy = 0.0
        self.last_acceleration = 0.0
        pole = model.body("pole").id
        mass = float(model.body_mass[pole])
        length = float(np.linalg.norm(model.body_ipos[pole]))
        self.mass_length = mass * length
        self.gravity_energy = self.mass_length * abs(float(model.opt.gravity[2]))
        self.data.qpos[:] = design.controller.reference[:2]
        mujoco.mj_forward(model, self.data)
        mujoco.mj_fullM(model, self.data, self.mass_matrix)
        self.hinge_inertia = float(self.mass_matrix[1, 1])

    def action(self, observation):
        state = np.asarray(observation, dtype=float)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("state must have four finite components")
        x, raw_theta, v, omega = state
        alpha = float(wrap_angle(raw_theta - self.design.controller.reference[1]))
        p = self.parameters
        self.last_energy = 0.5 * self.hinge_inertia * omega**2 + self.gravity_energy * np.cos(alpha)
        if self.mode == "balance" and abs(alpha) > p.release_angle:
            self.mode = "swingup"
        if (
            self.mode != "balance"
            and abs(alpha) < p.capture_angle
            and abs(omega) < p.capture_omega
            and abs(x) < 0.8
            and abs(v) < 1.5
        ):
            self.mode = "balance"
        if self.mode == "balance":
            normalized = state.copy()
            normalized[1] = self.design.controller.reference[1] + alpha
            return self.design.controller.action(normalized)

        energy_error = self.last_energy - self.gravity_energy
        acceleration = (
            p.energy_gain * energy_error * omega * np.cos(alpha)
            - p.position_gain * x
            - p.velocity_gain * v
        )
        # The exactly motionless bottom has omega=0: a bounded feedback kick breaks symmetry.
        if np.cos(alpha) < -0.9 and abs(omega) < 0.15 and abs(x) < 0.65:
            acceleration += p.kick_acceleration
            self.mode = "kick"
        else:
            self.mode = "swingup"
        acceleration = float(np.clip(acceleration, -p.acceleration_limit, p.acceleration_limit))
        self.last_acceleration = acceleration
        self.data.qpos[:] = state[:2]
        self.data.qvel[:] = state[2:]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_fullM(self.model, self.data, self.mass_matrix)
        rhs = self.data.qfrc_passive - self.data.qfrc_bias
        omega_acceleration = (rhs[1] - self.mass_matrix[1, 0] * acceleration) / self.mass_matrix[
            1, 1
        ]
        motor_force = (
            self.mass_matrix[0, 0] * acceleration
            + self.mass_matrix[0, 1] * omega_acceleration
            - rhs[0]
        )
        return np.array(
            [
                np.clip(
                    motor_force / self.design.actuator_gear,
                    -self.design.controller.control_limit,
                    self.design.controller.control_limit,
                )
            ],
            dtype=np.float32,
        )


def design_swingup_lqr(control_weight: float = 1.0):
    return design_lqr(control_weight=control_weight, environment_factory=make_swingup_environment)
