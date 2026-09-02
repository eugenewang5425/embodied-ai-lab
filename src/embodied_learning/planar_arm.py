"""Lesson 8: analytic 2R geometry plus a small, fully actuated MuJoCo arm."""

from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "assets" / "planar_arm.xml"
LENGTHS = np.array([0.4, 0.3])
FRAME_SKIP = 10
KP = np.array([12.0, 8.0])
KD = np.array([2.0, 0.6])
TORQUE_LIMIT = 0.25


def vector2(value):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError("expected two finite values")
    return vector


def angle_error(target, current):
    return (np.asarray(target) - np.asarray(current) + np.pi) % (2 * np.pi) - np.pi


def joint_positions(q):
    """World XY coordinates: fixed base, elbow, tip; joint angles are relative."""
    q1, q2 = vector2(q)
    elbow = LENGTHS[0] * np.array([np.cos(q1), np.sin(q1)])
    tip = elbow + LENGTHS[1] * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    return np.vstack([np.zeros(2), elbow, tip])


def forward_kinematics(q):
    return joint_positions(q)[-1]


def inverse_kinematics(target):
    """Both elbow-sign branches; reject unreachable targets rather than projecting them."""
    x, y = vector2(target)
    l1, l2 = LENGTHS
    radius = np.hypot(x, y)
    if radius < abs(l1 - l2) - 1e-12 or radius > l1 + l2 + 1e-12:
        raise ValueError(
            f"Target outside reachable annulus {abs(l1 - l2):.1f} <= radius <= {l1 + l2:.1f} m"
        )
    cosine = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    elbow = np.arccos(np.clip(cosine, -1.0, 1.0))
    solutions = []
    for q2 in (elbow, -elbow):
        q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
        solutions.append(angle_error([q1, q2], [0, 0]))
    return np.asarray(solutions)


def joint_pd(q, qvel, goal):
    # Same PD principle as before, but this time each motor directly actuates its joint.
    error = angle_error(vector2(goal), vector2(q))
    return np.clip(KP * error - KD * vector2(qvel), -TORQUE_LIMIT, TORQUE_LIMIT)


class ArmSimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL_PATH.read_text(encoding="utf-8"))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * FRAME_SKIP
        self.site_ids = [self.model.site(name).id for name in ("base", "elbow_point", "tip")]
        # Fail loudly if XML edits invalidate the analytic geometry or torque units.
        np.testing.assert_allclose(self.model.body("forearm").pos, [LENGTHS[0], 0, 0])
        np.testing.assert_allclose(self.model.site("tip").pos, [LENGTHS[1], 0, 0])
        np.testing.assert_allclose(self.model.jnt_axis, [[0, 0, 1], [0, 0, 1]])
        np.testing.assert_allclose(self.model.actuator_gear[:, 0], [1, 1])
        np.testing.assert_allclose(
            self.model.actuator_ctrlrange, [[-TORQUE_LIMIT, TORQUE_LIMIT]] * 2
        )
        self.reset()

    def reset(self, q=(0, 0), qvel=(0, 0)):
        q, qvel = vector2(q), vector2(qvel)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).copy()

    def points(self):
        return self.data.site_xpos[self.site_ids, :2].copy()

    def step(self, torque):
        command = np.clip(vector2(torque), -TORQUE_LIMIT, TORQUE_LIMIT)
        self.data.ctrl[:] = command
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        # mj_step's derived site values need a forward update at the FINAL qpos.
        mujoco.mj_forward(self.model, self.data)
        state = self.state()
        warning = any(
            self.data.warning[i].number
            for i in (
                mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL,
                mujoco.mjtWarning.mjWARN_BADQACC,
            )
        )
        failure = (
            "nonfinite_state"
            if not np.isfinite(state).all()
            else (
                "numerical_warning"
                if warning
                else ("velocity_boundary" if np.any(np.abs(state[2:]) > 20) else "")
            )
        )
        return state, command, failure
