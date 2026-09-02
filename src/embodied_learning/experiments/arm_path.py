"""Compare endpoint PD, joint interpolation and Jacobian-generated straight paths."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.arm_path import (
    DAMPING,
    HOLD_SECONDS,
    IK_COMPARISON_METHODS,
    METHOD_COLORS,
    METHODS,
    MOVE_SECONDS,
    REFERENCE_GAIN,
    REFERENCE_METHODS,
    REFERENCE_SPEED_LIMIT,
    TARGET,
    generate_reference,
    jacobian,
    segment_distance,
    singularity_probe,
    terminal_window_diagnostics,
    tracking_pd,
)
from embodied_learning.experiments.arm_reaching import INITIAL_Q
from embodied_learning.planar_arm import (
    KD,
    KP,
    LENGTHS,
    MODEL_PATH,
    TORQUE_LIMIT,
    ArmSimulation,
    angle_error,
    forward_kinematics,
    inverse_kinematics,
    joint_pd,
    vector2,
)
from embodied_learning.plotting import configure_plot_font


def audit_jacobian():
    sim = ArmSimulation()
    grid = np.deg2rad([-180, -90, -45, 0, 45, 90, 180])
    fd_errors, mj_errors = [], []
    for q1 in grid:
        for q2 in grid:
            q = np.array([q1, q2])
            sim.reset(q)
            jp, jr = np.zeros((3, 2)), np.zeros((3, 2))
            mujoco.mj_jacSite(sim.model, sim.data, jp, jr, sim.site_ids[-1])
            eps = 1e-6
            fd = np.column_stack(
                [
                    (forward_kinematics(q + step) - forward_kinematics(q - step)) / (2 * eps)
                    for step in eps * np.eye(2)
                ]
            )
            fd_errors.append(np.max(np.abs(jacobian(q) - fd)))
            mj_errors.append(np.max(np.abs(jacobian(q) - jp[:2])))
    return {
        "pose_count": len(fd_errors),
        "max_finite_difference_error_m_per_rad": float(max(fd_errors)),
        "max_mujoco_error_m_per_rad": float(max(mj_errors)),
        "finite_difference_step_rad": eps,
    }


def run_path(
    method,
    *,
    initial_q=INITIAL_Q,
    target=TARGET,
    move_seconds=MOVE_SECONDS,
    hold_seconds=HOLD_SECONDS,
):
    initial_q, target = vector2(initial_q), vector2(target)
    sim = ArmSimulation()
    reference = generate_reference(
        method, initial_q, target, dt=sim.dt, move_seconds=move_seconds, hold_seconds=hold_seconds
    )
    state = sim.reset(initial_q)
    states, points, torques = [state], [sim.points()], []
    requested_torques = []
    failure = ""
    for qref, dqref in zip(reference["q_reference"][:-1], reference["dq_reference"], strict=True):
        requested_torques.append(KP * angle_error(qref, state[:2]) + KD * (dqref - state[2:]))
        command = (
            joint_pd(state[:2], state[2:], qref)
            if method == "endpoint_pd"
            else tracking_pd(state[:2], state[2:], qref, dqref)
        )
        state, torque, failure = sim.step(command)
        states.append(state)
        points.append(sim.points())
        torques.append(torque)
        if failure:
            break
    n = len(torques)
    arrays = {
        "states": np.asarray(states),
        "points": np.asarray(points),
        "torques_nm": np.asarray(torques),
        "requested_torques_nm": np.asarray(requested_torques),
        **{
            key: value[: n if key in ("dq_reference", "reference_speed_scale") else n + 1]
            for key, value in reference.items()
        },
    }
    tip = arrays["points"][:, -1]
    times = np.arange(n + 1) * sim.dt
    tracking = np.linalg.norm(tip - arrays["desired_points"], axis=1)
    cross = segment_distance(tip, reference["desired_points"][0], target)
    movement = times <= move_seconds + 1e-10
    goal = inverse_kinematics(target)[0]
    within = (
        (np.linalg.norm(tip - target, axis=1) <= 0.002)
        & (np.max(np.abs(arrays["states"][:, 2:]), axis=1) <= 0.02)
        & (np.max(np.abs(angle_error(goal, arrays["states"][:, :2])), axis=1) <= 0.01)
    )
    tail = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(tail & (times[-1] - times >= 0.5 - 1e-10) & (times >= move_seconds))
    complete = n == len(reference["dq_reference"]) and not failure
    peak_cross = float(cross[movement].max() * 1000)
    settled = float(times[candidates[0]]) if len(candidates) and complete else None
    reference_tip = np.array([forward_kinematics(q) for q in arrays["q_reference"]])
    reference_error = np.linalg.norm(reference_tip - arrays["desired_points"], axis=1)
    execution_error = np.linalg.norm(tip - reference_tip, axis=1)
    report = {
        "key": method,
        "label": dict(REFERENCE_METHODS)[method],
        "target_m": target.tolist(),
        "start_m": reference["desired_points"][0].tolist(),
        "goal_q_rad": goal.tolist(),
        "initial_q_rad": initial_q.tolist(),
        "dt_s": sim.dt,
        "steps": n,
        "duration_s": float(times[-1]),
        "movement_s": float(move_seconds),
        "hold_s": float(hold_seconds),
        "completed": bool(complete),
        "endpoint_success": settled is not None,
        "path_success": bool(complete and settled is not None and peak_cross <= 2),
        "settled_after_movement_at_s": settled,
        "final_tip_error_mm": float(np.linalg.norm(tip[-1] - target) * 1000),
        "final_joint_error_rad": np.abs(angle_error(goal, arrays["states"][-1, :2])).tolist(),
        "final_speed_rad_s": np.abs(arrays["states"][-1, 2:]).tolist(),
        "terminal_window": terminal_window_diagnostics(
            arrays["states"], arrays["points"], target, goal, sim.dt, move_seconds=move_seconds
        ),
        "max_cross_track_mm": peak_cross,
        "rms_cross_track_mm": float(np.sqrt(np.mean(cross[movement] ** 2)) * 1000),
        "rms_timed_tracking_mm": float(np.sqrt(np.mean(tracking[movement] ** 2)) * 1000),
        "peak_torque_nm": np.max(np.abs(torques), axis=0).tolist(),
        "torque_saturated_steps": int(
            np.count_nonzero(np.any(np.abs(torques) >= TORQUE_LIMIT - 1e-12, axis=1))
        ),
        "peak_actual_speed_rad_s": np.max(np.abs(arrays["states"][:, 2:]), axis=0).tolist(),
        "reference_speed_limited_steps": int(np.count_nonzero(arrays["reference_speed_scale"] < 1)),
        "rms_reference_tracking_mm": float(np.sqrt(np.mean(reference_error[movement] ** 2)) * 1000),
        "rms_actual_to_reference_mm": float(
            np.sqrt(np.mean(execution_error[movement] ** 2)) * 1000
        ),
        "max_reference_tracking_mm": float(reference_error[movement].max() * 1000),
        "max_actual_to_reference_mm": float(execution_error[movement].max() * 1000),
        "min_reference_sigma_m_per_rad": float(
            min(np.linalg.svd(jacobian(q), compute_uv=False)[-1] for q in arrays["q_reference"])
        ),
        "min_actual_sigma_m_per_rad": float(
            min(np.linalg.svd(jacobian(q), compute_uv=False)[-1] for q in arrays["states"][:, :2])
        ),
        "max_reference_to_line_mm": float(
            np.max(
                segment_distance(
                    reference_tip,
                    reference["desired_points"][0],
                    target,
                )
            )
            * 1000
        ),
        "failure_reason": failure,
    }
    return arrays, report


def save_plot(path, archive, report):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    xy, cross_ax, timed_ax, torque_ax = axes.ravel()
    featured = (
        "waypoint_ik"
        if any(c["key"] == "waypoint_ik" for c in report["cases"])
        else "jacobian_path"
    )
    for case in report["cases"]:
        color = METHOD_COLORS[case["key"]]
        key = case["key"]
        tip = archive[f"{key}_points"][:, -1]
        desired = archive[f"{key}_desired_points"]
        times = np.arange(len(tip)) * case["dt_s"]
        xy.plot(tip[:, 0] * 100, tip[:, 1] * 100, color=color, label=case["label"])
        cross_ax.plot(
            times,
            segment_distance(tip, desired[0], case["target_m"]) * 1000,
            color=color,
            label=case["label"],
        )
        timed_ax.plot(times, np.linalg.norm(tip - desired, axis=1) * 1000, color=color)
        if key == featured:
            for joint in range(2):
                torque_ax.stairs(
                    archive[f"{key}_torques_nm"][:, joint], times, label=f"关节 {joint + 1}"
                )
    xy.plot(
        desired[[0, -1], 0] * 100, desired[[0, -1], 1] * 100, "k--", linewidth=1, label="规定直线"
    )
    xy.set(
        xlabel="X（cm）", ylabel="Y（cm）", title="真实末端路径：同一初态、同一终点", aspect="equal"
    )
    cross_ax.set(
        xlabel="时间（s）", ylabel="到有限线段距离（mm）", title="路径误差：是否走在直线上？"
    )
    cross_ax.axhline(2, color="gray", linestyle=":", label="2 mm 教学验收线")
    timed_ax.set(
        xlabel="时间（s）",
        ylabel="距同时刻规定点（mm）",
        title="时间误差：只给终点的方案没有时间要求"
        if any(c["key"] == "endpoint_pd" for c in report["cases"])
        else "时间误差：三种方案共享同一时间要求",
    )
    torque_ax.set(
        xlabel="时间（s）",
        ylabel="力矩（N·m）",
        title=f"{dict(REFERENCE_METHODS)[featured]}：仍靠关节 PD 驱动",
    )
    for ax in (cross_ax, timed_ax, torque_ax):
        ax.axvline(MOVE_SECONDS, color="gray", linestyle=":")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    xy.legend(fontsize=8)
    cross_ax.legend(fontsize=8)
    torque_ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(
    output: Path, *, initial_q=INITIAL_Q, target=TARGET, make_plot=True, trial=None, methods=METHODS
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if methods not in (METHODS, IK_COMPARISON_METHODS):
        raise ValueError("Expected the original three methods or the lesson11 comparison")
    audit = audit_jacobian()
    if (
        audit["max_finite_difference_error_m_per_rad"] > 1e-8
        or audit["max_mujoco_error_m_per_rad"] > 1e-12
    ):
        raise ValueError("Jacobian validation failed")
    archive, cases = {}, []
    for method, _ in methods:
        arrays, case = run_path(method, initial_q=initial_q, target=target)
        if trial is not None:
            case["trial"] = trial
        if methods == IK_COMPARISON_METHODS:
            case["lesson"] = 11
        archive.update({f"{method}_{key}": value for key, value in arrays.items()})
        cases.append(case)
    report = {
        "experiment": "planar_2r_path",
        "schema_version": 1,
        "lengths_m": LENGTHS.tolist(),
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "movement_s": MOVE_SECONDS,
        "hold_s": HOLD_SECONDS,
        "reference_gain_per_s": REFERENCE_GAIN,
        "dls_damping_m_per_rad": DAMPING,
        "reference_speed_limit_rad_s": REFERENCE_SPEED_LIMIT,
        "jacobian_audit": audit,
        "singularity_probe": singularity_probe(),
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "metric_window": "path and timed RMS/max: actual state samples from 0 through 8 s inclusive; equal temporal weighting",
        "diagnostic_metrics": "Reference tracking = f(q_reference) minus desired; execution = actual tip minus f(q_reference), both 0..8s. Minimum singular values over full recorded horizon. Saturation/caps are observations, not causal proof of failure.",
        "path_success_protocol": "completed 11s, max finite-segment distance <=2mm during 0..8s, plus endpoint_success",
        "endpoint_success_protocol": "final continuous tail >=0.5s after t>=8s, tip error<=2mm, each joint error<=0.01rad, each speed<=0.02rad/s; completed with no physical failure",
        "archive_alignment": "states, points, desired, q_reference at t[0..N]; torques/dq_reference/speed_scale[k] for interval [k*dt,(k+1)*dt); no fabricated final torque",
        "controller_structure": "Jacobian evaluated on geometric q_reference, not actual state; Euler-integrated reference generator; actual state feedback only in joint PD. No ideal velocity actuator.",
        "fairness": "Joint interpolation and Jacobian share 8s quintic timing, gains, feedforward reference velocity and torque limits. Endpoint PD is unchanged lesson8 arrival-only baseline; timed tracking is descriptive, not its objective.",
        "limitations": [
            "Same ideal horizontal 2R model; no contact, noise, payload or joint angle limits; one fixed line, not a general path planner",
            "DLS trades velocity error for bounded reference speeds; cannot restore missing directions at a singularity",
            "Singularity probe is static kinematic prediction, not a motor-driven velocity experiment",
            "Per-waypoint analytic IK is also sufficient for this 2R task; Jacobian is taught for local velocity relations and singularities, not claimed uniquely necessary",
        ],
        "cases": cases,
    }
    if methods == IK_COMPARISON_METHODS:
        report.update(
            experiment="planar_2r_ik_path",
            controller_structure="Joint PD unchanged. Waypoint IK solves each scheduled XY point on the positive elbow branch, unwraps continuously, and uses forward finite-interval joint-reference velocities. It never sets the simulated qpos after reset.",
            fairness="All three share initial state, scheduled XY line, 8+3s duration, gains and motor limits. Only reference generation (position and velocity) differs. DLS uniformly scales speeds; waypoint IK rejects plans above the same 1rad/s bound instead of silently clipping the path.",
            limitations=[
                "Analytic IK is specific to this ideal horizontal 2R model; not a universal replacement for Jacobian methods",
                "Waypoint velocities are planned forward differences over 0.02s, not exact instantaneous derivatives; q_reference and actual states remain distinct",
                "The positive elbow branch is fixed; no obstacles, payload, noise or automatic branch switching",
                "Numerically leaving a singular initial pose does not remove its instantaneous lost velocity direction",
            ],
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if make_plot:
        save_plot(output / "comparison.png", archive, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_experiment(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for case in report["cases"]:
        print(
            f"{case['key']}: endpoint={case['endpoint_success']}, path={case['path_success']}, max_cross={case['max_cross_track_mm']:.3f}mm, timed_rms={case['rms_timed_tracking_mm']:.3f}mm"
        )


if __name__ == "__main__":
    main()
