"""Nominal inverse-dynamics feedforward; never mutates the real simulation data."""

import mujoco
import numpy as np

from embodied_learning.planar_arm import ArmSimulation, vector2

FEEDFORWARD_METHODS = (("pd", "原 PD"), ("feedforward_pd", "模型前馈 + 原 PD"))
FEEDFORWARD_COLORS = {"pd": "#ea580c", "feedforward_pd": "#0f766e"}


def reference_acceleration(dq_reference, dt):
    """Forward differences of the existing offline velocity plan; terminal velocity=0."""
    dq = np.asarray(dq_reference, dtype=float)
    if dq.ndim != 2 or dq.shape[1] != 2 or len(dq) < 1 or not np.isfinite(dq).all():
        raise ValueError("Expected a finite nonempty N by 2 reference velocity")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    return np.diff(np.vstack([dq, np.zeros((1, 2))]), axis=0) / dt


def inverse_torque(model, scratch, q, dq, ddq):
    """Continuous-time inverse dynamics of the contact-free direct-drive 2R model."""
    if model.nq != 2 or model.nv != 2 or model.nu != 2:
        raise ValueError("Expected the two-joint, two-motor arm")
    scratch.qpos[:] = vector2(q)
    scratch.qvel[:] = vector2(dq)
    scratch.qacc[:] = vector2(ddq)
    mujoco.mj_inverse(model, scratch)
    if scratch.nefc or scratch.ncon:
        raise ValueError("This lesson's feedforward model must be contact/constraint free")
    torque = scratch.qfrc_inverse.copy()
    if not np.isfinite(torque).all():
        raise ValueError("Nonfinite inverse-dynamics torque")
    return torque


def feedforward_reference(model, reference, dt):
    q, dq = reference["q_reference"], reference["dq_reference"]
    ddq = reference_acceleration(dq, dt)
    if np.asarray(q).shape != (len(dq) + 1, 2):
        raise ValueError("Joint position and velocity reference lengths disagree")
    scratch = mujoco.MjData(model)
    torque = np.asarray(
        [inverse_torque(model, scratch, q[k], dq[k], ddq[k]) for k in range(len(dq))]
    )
    return torque, ddq


def audit_inverse_dynamics():
    """Algebra + forward round trip; an audit, not an unconstrained motor rollout."""
    model = ArmSimulation().model
    inverse, forward = mujoco.MjData(model), mujoco.MjData(model)
    rng = np.random.default_rng(1300)
    algebra_errors, acceleration_errors = [], []
    for _ in range(49):
        q, dq, ddq = rng.uniform(-3, 3, 2), rng.uniform(-1, 1, 2), rng.uniform(-2, 2, 2)
        torque = inverse_torque(model, inverse, q, dq, ddq)
        mass = np.zeros((2, 2))
        mujoco.mj_fullM(model, inverse, mass)
        expected = mass @ ddq + inverse.qfrc_bias - inverse.qfrc_passive
        algebra_errors.append(float(np.max(np.abs(torque - expected))))
        # Apply an unrestricted generalized force ONLY in this non-integrating audit.
        mujoco.mj_resetData(model, forward)
        forward.qpos[:] = q
        forward.qvel[:] = dq
        forward.qfrc_applied[:] = torque
        mujoco.mj_forward(model, forward)
        acceleration_errors.append(float(np.max(np.abs(forward.qacc - ddq))))
    report = {
        "states": 49,
        "seed": 1300,
        "max_force_identity_error_nm": max(algebra_errors),
        "max_forward_acceleration_error_rad_s2": max(acceleration_errors),
        "scope": "Continuous-time consistency at 49 states, no time integration or claim of motor feasibility. Real rollouts still use bounded motor ctrl, never qfrc_applied.",
    }
    if max(algebra_errors) > 1e-10 or max(acceleration_errors) > 1e-10:
        raise ValueError("Inverse dynamics audit failed")
    return report
