"""Discrete LQR around the actual upright equilibrium of our cart-pole model.

The Jacobians cover one complete Gymnasium action (two RK4 physics steps),
not just one MuJoCo step. Commands are actuator inputs, NOT forces in newtons.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_discrete_are

from embodied_learning.environments import make_inverted_pendulum_environment

Array = NDArray[np.float64]
DEFAULT_STATE_WEIGHTS = (10.0, 100.0, 1.0, 1.0)
DEFAULT_CONTROL_WEIGHT = 0.1


def transition(model: mujoco.MjModel, state: Array, control: float, frame_skip: int) -> Array:
    """Evaluate one action from fresh simulator data, with no hidden-state carryover."""
    data = mujoco.MjData(model)
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:]
    data.ctrl[0] = control
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=frame_skip)
    return np.concatenate((data.qpos, data.qvel))


def upright_reference(model: mujoco.MjModel) -> Array:
    """Find the upright zero-input equilibrium for this named planar cart-pole.

    The built-in pole has a small x offset in its local center of mass. Rotating
    about +y by -atan2(com_x, com_z) places its COM directly above the hinge.
    This is intentionally model-specific, not a general robot trim solver.
    """
    if (model.nq, model.nv, model.nu, model.na) != (2, 2, 1, 0):
        raise ValueError("LQR design supports only the two-joint, one-motor cart-pole")
    hinge = model.joint("hinge").id
    if not np.allclose(model.jnt_axis[hinge], [0, 1, 0]):
        raise ValueError("expected the pole hinge to rotate about +y")
    center = model.body_ipos[model.body("pole").id]
    reference = np.array([0.0, -np.arctan2(center[0], center[2]), 0.0, 0.0])
    data = mujoco.MjData(model)
    data.qpos[:] = reference[:2]
    mujoco.mj_forward(model, data)
    if np.max(np.abs(data.qacc)) > 1e-8:
        raise ValueError("computed reference is not a zero-input equilibrium")
    return reference


def linearize(
    model: mujoco.MjModel, reference: Array, frame_skip: int, epsilon: float = 1e-6
) -> tuple[Array, Array]:
    """Central finite differences of the real discrete simulator transition."""
    if not np.isfinite(epsilon) or epsilon <= 0 or frame_skip < 1:
        raise ValueError("epsilon and frame_skip must be positive")
    a = np.empty((4, 4))
    for column, perturbation in enumerate(np.eye(4) * epsilon):
        a[:, column] = (
            transition(model, reference + perturbation, 0.0, frame_skip)
            - transition(model, reference - perturbation, 0.0, frame_skip)
        ) / (2 * epsilon)
    b = (
        (
            transition(model, reference, epsilon, frame_skip)
            - transition(model, reference, -epsilon, frame_skip)
        )
        / (2 * epsilon)
    ).reshape(4, 1)
    return a, b


@dataclass(frozen=True)
class LQRController:
    gain: Array
    reference: Array
    control_limit: float

    def action(self, observation: NDArray[np.floating]) -> NDArray[np.float32]:
        state = np.asarray(observation, dtype=np.float64)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("observation must contain four finite state values")
        command = float((-self.gain @ (state - self.reference)).item())
        return np.array([np.clip(command, -self.control_limit, self.control_limit)], np.float32)


@dataclass(frozen=True)
class LQRDesign:
    controller: LQRController
    a: Array
    b: Array
    q: Array
    r: Array
    riccati: Array
    dt: float
    frame_skip: int
    actuator_gear: float


def design_lqr(
    state_weights: tuple[float, float, float, float] = DEFAULT_STATE_WEIGHTS,
    control_weight: float = DEFAULT_CONTROL_WEIGHT,
    *,
    environment_factory=make_inverted_pendulum_environment,
) -> LQRDesign:
    """Design offline once; rollouts only evaluate u = -K (state - reference)."""
    weights = np.asarray(state_weights, dtype=np.float64)
    if weights.shape != (4,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("state_weights must be four finite positive numbers")
    if not np.isfinite(control_weight) or control_weight <= 0:
        raise ValueError("control_weight must be finite and positive")
    env = environment_factory()
    try:
        base = env.unwrapped
        model = base.model
        reference = upright_reference(model)
        a, b = linearize(model, reference, base.frame_skip)
        q, r = np.diag(weights), np.array([[control_weight]])
        riccati = solve_discrete_are(a, b, q, r)
        gain = np.linalg.solve(r + b.T @ riccati @ b, b.T @ riccati @ a)
        if np.max(np.abs(np.linalg.eigvals(a - b @ gain))) >= 1:
            raise RuntimeError("the linearized closed-loop design is not stable")
        controller = LQRController(gain, reference, float(env.action_space.high[0]))
        return LQRDesign(
            controller,
            a,
            b,
            q,
            r,
            riccati,
            base.dt,
            base.frame_skip,
            float(model.actuator_gear[0, 0]),
        )
    finally:
        env.close()
