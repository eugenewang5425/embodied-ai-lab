"""Lesson 9: differential kinematics generates references; joint PD drives physics."""

import numpy as np

from embodied_learning.planar_arm import (
    KD,
    KP,
    LENGTHS,
    TORQUE_LIMIT,
    angle_error,
    forward_kinematics,
    inverse_kinematics,
    vector2,
)

DAMPING = 0.02
REFERENCE_GAIN = 4.0
REFERENCE_SPEED_LIMIT = 1.0
MOVE_SECONDS = 8.0
HOLD_SECONDS = 3.0
TARGET = np.array([0.35, 0.30])
METHODS = (
    ("endpoint_pd", "只给终点（上一课）"),
    ("joint_interpolation", "关节角平滑插值"),
    ("jacobian_path", "Jacobian 直线路径"),
)
COLORS = ("#2563eb", "#ea580c", "#0f766e")
WAYPOINT_IK = ("waypoint_ik", "逐点解析 IK")
REFERENCE_METHODS = (*METHODS, WAYPOINT_IK)
IK_COMPARISON_METHODS = (*METHODS[1:], WAYPOINT_IK)
METHOD_COLORS = {
    **dict(zip((key for key, _ in METHODS), COLORS, strict=True)),
    "waypoint_ik": "#7c3aed",
}
TIMINGS = (("move_8s", 8.0, "#2563eb"), ("move_4s", 4.0, "#ea580c"), ("move_2s", 2.0, "#7c3aed"))


def jacobian(q):
    """World XY velocity = J(q) @ relative joint velocities; m/rad."""
    q1, q2 = vector2(q)
    l1, l2 = LENGTHS
    return np.array(
        [
            [-l1 * np.sin(q1) - l2 * np.sin(q1 + q2), -l2 * np.sin(q1 + q2)],
            [l1 * np.cos(q1) + l2 * np.cos(q1 + q2), l2 * np.cos(q1 + q2)],
        ]
    )


def damped_velocity(q, velocity, damping=DAMPING, speed_limit=REFERENCE_SPEED_LIMIT):
    """Damped least squares, then uniform scaling (not per-component clipping)."""
    if not np.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be positive and finite")
    if not np.isfinite(speed_limit) or speed_limit <= 0:
        raise ValueError("speed_limit must be positive and finite")
    j = jacobian(q)
    dq = j.T @ np.linalg.solve(j @ j.T + damping**2 * np.eye(2), vector2(velocity))
    scale = min(1.0, speed_limit / max(float(np.max(np.abs(dq))), 1e-15))
    return dq * scale, scale


def segment_distance(points, start, end):
    """Distance to a finite segment; overshooting an endpoint counts as error."""
    points = np.asarray(points, dtype=float)
    start, end = vector2(start), vector2(end)
    delta = end - start
    squared = float(delta @ delta)
    if squared <= 1e-24:
        return np.linalg.norm(points - start, axis=-1)
    fraction = np.clip((points - start) @ delta / squared, 0, 1)
    return np.linalg.norm(points - (start + fraction[..., None] * delta), axis=-1)


def validate_line(start, end):
    start, end = vector2(start), vector2(end)
    # Reachable endpoints alone are insufficient: the line might cross the inner hole.
    inner = float(segment_distance(np.zeros(2), start, end))
    outer = max(np.linalg.norm(start), np.linalg.norm(end))
    if inner < abs(LENGTHS[0] - LENGTHS[1]) - 1e-12 or outer > sum(LENGTHS) + 1e-12:
        raise ValueError("Whole line must lie within the reachable annulus")


def progress(time, duration=MOVE_SECONDS):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(time):
        raise ValueError("finite time and positive duration required")
    s = np.clip(time / duration, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5, 30 * s**2 * (1 - s) ** 2 / duration


class ReferenceSpeedError(ValueError):
    """A geometrically valid plan rejected before motor-driven simulation."""

    def __init__(self, peak):
        self.peak_rad_s = float(peak)
        super().__init__(
            f"waypoint IK reference exceeds the 1 rad/s planning limit: {peak:.6f} rad/s"
        )


def validate_timing(move_seconds, hold_seconds, dt):
    if not np.isfinite(move_seconds) or move_seconds <= 0:
        raise ValueError("movement duration must be positive and finite")
    if not np.isfinite(hold_seconds) or hold_seconds < 0.5:
        raise ValueError("hold duration must be finite and at least 0.5 s")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    for duration in (move_seconds, hold_seconds):
        if dt > duration or not np.isclose(duration / dt, round(duration / dt), rtol=0, atol=1e-9):
            raise ValueError("dt must divide movement and hold durations")


def generate_reference(
    method,
    initial_q,
    target=TARGET,
    dt=0.02,
    *,
    move_seconds=MOVE_SECONDS,
    hold_seconds=HOLD_SECONDS,
):
    """No simulation mutation: integrate a geometric reference, not the real arm."""
    if method not in dict(REFERENCE_METHODS):
        raise ValueError("Unknown path method")
    validate_timing(move_seconds, hold_seconds, dt)
    initial_q, target = vector2(initial_q), vector2(target)
    start = forward_kinematics(initial_q)
    validate_line(start, target)
    goal = inverse_kinematics(target)[0]
    delta_q = angle_error(goal, initial_q)
    count = round((move_seconds + hold_seconds) / dt)
    times = np.arange(count + 1) * dt
    fractions, rates = np.asarray([progress(t, move_seconds) for t in times]).T
    desired = start + fractions[:, None] * (target - start)
    velocities = rates[:, None] * (target - start)
    qref = np.empty((count + 1, 2))
    dqref = np.zeros((count, 2))
    scales = np.ones(count)
    if method == "endpoint_pd":
        qref[:] = goal
    elif method == "joint_interpolation":
        qref[:] = initial_q + fractions[:, None] * delta_q
        dqref[:] = rates[:-1, None] * delta_q
    elif method == "waypoint_ik":
        # The positive elbow branch is the same one used by all lesson10 trials.
        first = inverse_kinematics(desired[0])[0]
        if np.max(np.abs(angle_error(first, initial_q))) > 1e-6:
            raise ValueError("waypoint IK requires an initial pose on the positive elbow branch")
        qref[0] = initial_q
        for k in range(1, count + 1):
            solution = inverse_kinematics(desired[k])[0]
            qref[k] = qref[k - 1] + angle_error(solution, qref[k - 1])
        # The whole reference path is planned ahead of execution. This is the
        # finite-interval average reference velocity, not an inverse of singular J.
        dqref[:] = np.diff(qref, axis=0) / dt
        if np.max(np.abs(dqref)) > REFERENCE_SPEED_LIMIT + 1e-10:
            raise ReferenceSpeedError(np.max(np.abs(dqref)))
    else:
        qref[0] = initial_q
        for k in range(count):
            velocity = velocities[k] + REFERENCE_GAIN * (desired[k] - forward_kinematics(qref[k]))
            dqref[k], scales[k] = damped_velocity(qref[k], velocity)
            qref[k + 1] = qref[k] + dt * dqref[k]
    return {
        "desired_points": desired,
        "desired_velocities": velocities,
        "q_reference": qref,
        "dq_reference": dqref,
        "reference_speed_scale": scales,
    }


def tracking_pd(q, qvel, qref, dqref):
    return np.clip(
        KP * angle_error(vector2(qref), vector2(q)) + KD * (vector2(dqref) - vector2(qvel)),
        -TORQUE_LIMIT,
        TORQUE_LIMIT,
    )


def terminal_window_diagnostics(states, points, target, goal, dt, *, move_seconds=MOVE_SECONDS):
    """Explain the existing 0.5 s terminal criterion, not just the final frame."""
    times = np.arange(len(states)) * dt
    window = times >= times[-1] - 0.5 - 1e-10
    tip_error = float(np.max(np.linalg.norm(points[window, -1] - target, axis=1)) * 1000)
    joint_error = np.max(np.abs(angle_error(goal, states[window, :2])), axis=0)
    speed = np.max(np.abs(states[window, 2:]), axis=0)
    violations = []
    if times[-1] < move_seconds + 0.5 - 1e-10:
        violations.append("insufficient_post_movement_window")
    if tip_error > 2:
        violations.append("tip_position")
    if np.any(joint_error > 0.01):
        violations.append("joint_position")
    if np.any(speed > 0.02):
        violations.append("joint_speed")
    return {
        "start_s": float(times[window][0]),
        "end_s": float(times[-1]),
        "max_tip_error_mm": tip_error,
        "max_joint_error_rad": joint_error.tolist(),
        "max_speed_rad_s": speed.tolist(),
        "violations": violations,
    }


def singularity_probe():
    """Static velocity predictions only, not dynamically achievable commands."""
    requested = np.array([0.02, 0])
    records = []
    for elbow in (30, 10, 1, 0.1, 0):
        q = np.deg2rad([0, elbow])
        j = jacobian(q)
        sv = np.linalg.svd(j, compute_uv=False)
        raw = np.linalg.solve(j, requested) if sv[-1] > 1e-12 else None
        dq, scale = damped_velocity(q, requested)
        records.append(
            {
                "q_deg": [0, elbow],
                "sigma_min_m_per_rad": float(sv[-1]),
                "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else None,
                "requested_velocity_m_s": requested.tolist(),
                "inverse_dq_rad_s": raw.tolist() if raw is not None else None,
                "dls_dq_rad_s": dq.tolist(),
                "speed_scale": scale,
                "predicted_velocity_m_s": (j @ dq).tolist(),
                "velocity_residual_m_s": float(np.linalg.norm(j @ dq - requested)),
            }
        )
    return records
