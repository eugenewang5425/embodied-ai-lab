"""Audit 2R geometry against MuJoCo, then reach three deterministic goals with joint PD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

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
    joint_positions,
    vector2,
)
from embodied_learning.plotting import configure_plot_font

INITIAL_Q = np.deg2rad([-40, 80])
CASES = (
    ("a_positive", "目标 A：肘角为正", [0.35, 0.30], 0),
    ("a_negative", "目标 A：肘角为负", [0.35, 0.30], 1),
    ("b_positive", "目标 B：换个位置", [0.35, -0.30], 0),
)


def audit_geometry():
    sim = ArmSimulation()
    angles = np.deg2rad([-180, -90, -45, 0, 45, 90, 180])
    errors = []
    for q1 in angles:
        for q2 in angles:
            sim.reset([q1, q2])
            errors.append(np.max(np.linalg.norm(sim.points() - joint_positions([q1, q2]), axis=1)))
    return {
        "pose_count": len(errors),
        "max_point_error_m": float(max(errors)),
        "angle_grid_deg": np.rad2deg(angles).tolist(),
        "compared_points": ["base", "elbow", "tip"],
    }


def run_reach(target, branch=0, steps=300):
    if branch not in (0, 1) or steps < 1:
        raise ValueError("need branch 0/1 and positive steps")
    target = vector2(target)
    goal = inverse_kinematics(target)[branch]
    sim = ArmSimulation()
    state = sim.reset(INITIAL_Q)
    states, points, controls = [state], [sim.points()], []
    failure = ""
    for _ in range(steps):
        state, torque, failure = sim.step(joint_pd(state[:2], state[2:], goal))
        states.append(state)
        points.append(sim.points())
        controls.append(torque)
        if failure:
            break
    arrays = {
        "states": np.asarray(states),
        "points": np.asarray(points),
        "torques_nm": np.asarray(controls),
    }
    times = np.arange(len(states)) * sim.dt
    error = np.linalg.norm(arrays["points"][:, -1] - target, axis=1)
    joint_errors = angle_error(goal, arrays["states"][:, :2])
    within = (
        (error <= 0.002)
        & (np.max(np.abs(arrays["states"][:, 2:]), axis=1) <= 0.02)
        & (np.max(np.abs(joint_errors), axis=1) <= 0.01)
    )
    tail = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(tail & (times[-1] - times >= 0.5 - 1e-10))
    settled = float(times[candidates[0]]) if len(candidates) and not failure else None
    fk_error = max(
        np.linalg.norm(forward_kinematics(s[:2]) - p[-1])
        for s, p in zip(arrays["states"], arrays["points"], strict=True)
    )
    report = {
        "target_m": target.tolist(),
        "branch": branch,
        "goal_q_rad": goal.tolist(),
        "initial_q_rad": INITIAL_Q.tolist(),
        "dt_s": sim.dt,
        "steps": len(controls),
        "duration_s": float(times[-1]),
        "success": settled is not None,
        "settled_at_s": settled,
        "final_tip_error_mm": float(error[-1] * 1000),
        "peak_torque_nm": np.max(np.abs(controls), axis=0).tolist(),
        "max_dynamic_fk_error_m": float(fk_error),
        "failure_reason": failure,
    }
    return arrays, report


def save_plot(path, archive, report):
    configure_plot_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout="constrained")
    for case, color in zip(report["cases"], ("#2563eb", "#ea580c", "#0f766e"), strict=True):
        key, label = case["key"], case["label"]
        p, u = archive[f"{key}_points"], archive[f"{key}_torques_nm"]
        times = np.arange(len(p)) * case["dt_s"]
        target = np.asarray(case["target_m"])
        axes[0].plot(p[:, -1, 0], p[:, -1, 1], color=color, label=label)
        axes[0].plot(*target, marker="x", color=color)
        axes[0].plot(p[-1, :, 0], p[-1, :, 1], "o--", color=color, alpha=0.6)
        error = np.linalg.norm(p[:, -1] - target, axis=1) * 1000
        axes[1].plot(times, error, color=color, label=label)
        if key == "a_positive":
            for j in range(2):
                axes[2].stairs(u[:, j], times, label=f"关节 {j + 1}")
    for radius in (abs(LENGTHS[0] - LENGTHS[1]), sum(LENGTHS)):
        axes[0].add_patch(plt.Circle((0, 0), radius, fill=False, linestyle=":", color="gray"))
    axes[0].set(
        xlabel="世界 X（m）",
        ylabel="世界 Y（m）",
        title="同一点，两种姿态；路径并非直线",
        aspect="equal",
    )
    axes[1].set(xlabel="时间（s）", ylabel="末端误差（mm）", title="IK 给关节目标，PD 推动真实运动")
    axes[1].axhline(2, color="gray", linestyle=":")
    axes[2].set(xlabel="时间（s）", ylabel="关节力矩（N·m）", title="目标 A / 肘角为正：两个电机")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(output: Path):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    audit = audit_geometry()
    if audit["max_point_error_m"] > 1e-10:
        raise ValueError("MuJoCo geometry and analytic forward kinematics disagree")
    archive, cases = {}, []
    for key, label, target, branch in CASES:
        arrays, metadata = run_reach(target, branch)
        archive.update({f"{key}_{name}": value for name, value in arrays.items()})
        cases.append({"key": key, "label": label, **metadata})
    report = {
        "experiment": "planar_2r_reaching",
        "schema_version": 1,
        "lengths_m": LENGTHS.tolist(),
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "geometry_audit": audit,
        "state_order": ["q1_rad", "q2_rad", "dq1_rad_s", "dq2_rad_s"],
        "angle_convention": "CCW about +Z; q1 relative to world +X, q2 relative to link1; link2 world angle=q1+q2",
        "physics_timestep_s": 0.002,
        "control_timestep_s": 0.02,
        "success_protocol": "final continuous tail >=0.5s with tip error<=2mm, each joint error<=0.01rad, each speed<=0.02rad/s; no physical failure",
        "archive_alignment": "states/points s[0..N]; torques[k] held over [k*dt,(k+1)*dt); points recomputed at final qpos of each step",
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "limitations": [
            "Horizontal plane, fixed base, 2 actuated joints; no contact, joint-angle limits or payload",
            "Deterministic targets; no noise, external disturbance or learned controller",
            "Geometric reachability is not collision-free path planning; straight-line tip tracking not implemented",
        ],
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
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
    print("Geometry audit:", report["geometry_audit"])
    for case in report["cases"]:
        print(
            f"{case['key']}: success={case['success']}, settled={case['settled_at_s']}s, final_error={case['final_tip_error_mm']:.6g}mm"
        )


if __name__ == "__main__":
    main()
