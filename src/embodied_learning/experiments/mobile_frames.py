"""Lesson 14: five ideal wheel-speed maneuvers and world/body/sensor frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.differential_drive import (
    DriveGeometry,
    compose,
    integrate_pose,
    to_child,
    to_parent,
)
from embodied_learning.plotting import configure_plot_font

DT, SECONDS = 0.04, 4.0
GEOMETRY = DriveGeometry()
SENSOR_IN_BODY = np.array([0.12, 0.04, np.pi / 6])
LANDMARK_WORLD = np.array([1.1, 0.8])
CASES = (
    ("straight", "两轮同速：直行", "#2563eb"),
    ("spin", "两轮反向：原地左转", "#9333ea"),
    ("left_arc", "右轮更快：向左弯", "#0f766e"),
    ("right_arc", "左轮更快：向右弯", "#ea580c"),
    ("turn_then_drive", "先左转 90°，再向前走", "#0891b2"),
)
ARRAY_WIDTHS = {
    "poses": 3,
    "wheels_rad_s": 2,
    "body_velocity": 2,
    "sensor_world_poses": 3,
    "landmark_body": 2,
    "landmark_sensor": 2,
    "reconstructed_world": 2,
    "wrong_world": 2,
}
INTERVAL_ARRAYS = {"wheels_rad_s", "body_velocity"}


def wheel_schedule(key):
    n = round(SECONDS / DT)
    if key == "turn_then_drive":
        # Exactly pi/2 radians in two seconds, then 0.4 metres forward in body x.
        wheel = GEOMETRY.track_m * (np.pi / 4) / (2 * GEOMETRY.radius_m)
        return np.vstack([np.tile([-wheel, wheel], (n // 2, 1)), np.tile([4, 4], (n // 2, 1))])
    constant = {"straight": [4, 4], "spin": [-2, 2], "left_arc": [2, 4], "right_arc": [4, 2]}
    if key not in constant:
        raise ValueError("Unknown maneuver")
    return np.tile(constant[key], (n, 1)).astype(float)


def expected_endpoint(key):
    """Independent whole-maneuver analytic checks, not a second step-by-step rollout."""
    angle = 4 / 3
    if key == "straight":
        return np.array([0.8, 0, 0])
    if key == "spin":
        return np.array([0, 0, 8 / 3])
    if key in ("left_arc", "right_arc"):
        sign = 1 if key == "left_arc" else -1
        return np.array([0.45 * np.sin(angle), sign * 0.45 * (1 - np.cos(angle)), sign * angle])
    if key == "turn_then_drive":
        return np.array([0, 0.4, np.pi / 2])
    raise ValueError("Unknown maneuver")


def run_case(key):
    wheels = wheel_schedule(key)
    velocities = np.array([GEOMETRY.body_velocity(pair) for pair in wheels])
    poses = np.zeros((len(wheels) + 1, 3))
    for i, velocity in enumerate(velocities):
        poses[i + 1] = integrate_pose(poses[i], velocity, DT)
    sensor_poses = np.array([compose(pose, SENSOR_IN_BODY) for pose in poses])
    in_body = np.array([to_child(pose, LANDMARK_WORLD) for pose in poses])
    in_sensor = np.array([to_child(SENSOR_IN_BODY, point) for point in in_body])
    reconstructed = np.array(
        [to_parent(pose, to_parent(SENSOR_IN_BODY, point)) for pose, point in zip(poses, in_sensor)]
    )
    # Deliberately wrong: treating sensor coordinates as if they shared WORLD axes
    # and originated at the body centre (omits both rotations and sensor offset).
    wrong = poses[:, :2] + in_sensor
    arrays = {
        "poses": poses,
        "wheels_rad_s": wheels,
        "body_velocity": velocities,
        "sensor_world_poses": sensor_poses,
        "landmark_body": in_body,
        "landmark_sensor": in_sensor,
        "reconstructed_world": reconstructed,
        "wrong_world": wrong,
    }
    expected = expected_endpoint(key)
    label, color = next((label, color) for k, label, color in CASES if key == k)
    metrics = {
        "key": key,
        "label": label,
        "color": color,
        "steps": len(wheels),
        "endpoint": poses[-1].tolist(),
        "expected_endpoint": expected.tolist(),
        "endpoint_position_error_m": float(np.linalg.norm(poses[-1, :2] - expected[:2])),
        "endpoint_yaw_error_rad": float(abs(poses[-1, 2] - expected[2])),
        "max_reconstruction_error_m": float(
            np.max(np.linalg.norm(reconstructed - LANDMARK_WORLD, axis=1))
        ),
        "max_wrong_mapping_error_m": float(np.max(np.linalg.norm(wrong - LANDMARK_WORLD, axis=1))),
    }
    return arrays, metrics


def make_plot(all_arrays, output):
    configure_plot_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for key, label, color in CASES:
        poses = all_arrays[key]["poses"]
        axes[0].plot(*poses[:, :2].T, color=color, label=label)
        axes[0].plot(*poses[-1, :2], "o", color=color)
    axes[0].set(xlabel="世界 X / m", ylabel="世界 Y / m", title="轮速决定运动；不是力矩仿真")
    axes[0].axis("equal")
    axes[0].legend(fontsize=8)
    arrays = all_arrays["turn_then_drive"]
    times = np.arange(len(arrays["poses"])) * DT
    for i, color in enumerate(("#2563eb", "#ea580c")):
        axes[1].plot(times, arrays["landmark_sensor"][:, i], color=color, label=f"传感器 {'xy'[i]}")
    axes[1].axvline(2, color="#64748b", linestyle="--", label="转弯→直行")
    axes[1].set(xlabel="仿真时间 / s", ylabel="坐标 / m", title="同一个固定地标，车上读数在变")
    axes[1].legend()
    for name, label, color in (
        ("reconstructed_world", "完整坐标变换", "#0f766e"),
        ("wrong_world", "错误：仅加车体平移", "#dc2626"),
    ):
        axes[2].plot(
            times, np.linalg.norm(arrays[name] - LANDMARK_WORLD, axis=1), color=color, label=label
        )
    axes[2].set(
        xlabel="仿真时间 / s",
        ylabel="地标世界坐标误差 / m",
        title="先转再走：几何一致性，不是定位精度",
    )
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def run_experiment(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arrays_by_key, archive, cases = {}, {}, []
    for key, _, _ in CASES:
        arrays, metrics = run_case(key)
        arrays_by_key[key] = arrays
        archive.update({f"{key}_{name}": value for name, value in arrays.items()})
        cases.append(metrics)
    np.savez_compressed(output / "trajectories.npz", **archive)
    source_dir = Path(__file__).resolve().parents[1]
    sources = [source_dir / "differential_drive.py", Path(__file__).resolve()]
    report = {
        "experiment": "differential_drive_frames",
        "schema_version": 1,
        "dt_s": DT,
        "duration_s": SECONDS,
        "wheel_radius_m": GEOMETRY.radius_m,
        "track_width_m": GEOMETRY.track_m,
        "sensor_in_body": SENSOR_IN_BODY.tolist(),
        "landmark_world_m": LANDMARK_WORLD.tolist(),
        "state_order": ["world_x_m", "world_y_m", "unwrapped_yaw_rad"],
        "input_order": ["left_wheel_rad_s", "right_wheel_rad_s"],
        "alignment": "poses/observations: N+1; wheel speeds/body velocities: N, acting on [t_i,t_i+dt)",
        "model": "ideal_no_slip_velocity_kinematics",
        "limitations": [
            "No motor torque/dynamics",
            "No slip/noise/occlusion/FOV",
            "No feedback or localization estimator",
            "Round-trip geometry uses known pose, not independently measured localization",
        ],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "source_sha256": {
            str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "trajectories_sha256": hashlib.sha256(
            (output / "trajectories.npz").read_bytes()
        ).hexdigest(),
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(arrays_by_key, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.output)
    for case in report["cases"]:
        print(
            f"{case['key']}: pose={case['endpoint']}; frame error={case['max_reconstruction_error_m']:.3g} m; wrong={case['max_wrong_mapping_error_m']:.3f} m"
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
