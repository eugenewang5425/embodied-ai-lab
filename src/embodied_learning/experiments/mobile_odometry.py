"""Lesson 15: identical real motion, three right-encoder calibration scales."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.differential_drive import compose, integrate_pose, to_child, to_parent
from embodied_learning.experiments.mobile_frames import (
    DT,
    GEOMETRY,
    LANDMARK_WORLD,
    SENSOR_IN_BODY,
    wheel_schedule,
)
from embodied_learning.odometry import estimate_poses, heading_error, scaled_encoder_readings
from embodied_learning.plotting import configure_plot_font

SCENARIOS = (("straight", "直行 12 秒", 300), ("square", "走方形一圈 24 秒", 600))
VARIANTS = (
    ("ideal", "无偏差 0%", 1.0, "#0f766e"),
    ("right_1pct", "右轮读数 +1%", 1.01, "#ea580c"),
    ("right_2pct", "右轮读数 +2%", 1.02, "#9333ea"),
)
SHARED_WIDTHS = {"true_poses": 3, "wheels_rad_s": 2, "wheel_angles_rad": 2, "landmark_sensor": 2}
ESTIMATE_WIDTHS = {
    "encoder_angles_rad": 2,
    "poses": 3,
    "mapped_landmark": 2,
    "position_error_m": None,
    "heading_error_rad": None,
    "landmark_error_m": None,
}


def schedule(key):
    straight = wheel_schedule("straight")  # The identical 4s primitive from lesson 14.
    turn = wheel_schedule("turn_then_drive")[:50]
    if key == "straight":
        return np.tile(straight, (3, 1)), [0, 100, 200, 300]
    if key == "square":
        return np.tile(np.vstack([straight, turn]), (4, 1)), [
            0,
            100,
            150,
            250,
            300,
            400,
            450,
            550,
            600,
        ]
    raise ValueError("Unknown odometry scenario")


def simulate_truth(wheels):
    """Plant path depends on commanded wheel speed, NOT on any encoder scale."""
    poses = np.zeros((len(wheels) + 1, 3))
    for i, pair in enumerate(wheels):
        poses[i + 1] = integrate_pose(poses[i], GEOMETRY.body_velocity(pair), DT)
    angles = np.vstack([np.zeros(2), np.cumsum(wheels * DT, axis=0)])
    observations = np.array(
        [to_child(compose(pose, SENSOR_IN_BODY), LANDMARK_WORLD) for pose in poses]
    )
    return {
        "true_poses": poses,
        "wheels_rad_s": wheels.copy(),
        "wheel_angles_rad": angles,
        "landmark_sensor": observations,
    }


def evaluate_estimate(shared, scale):
    # The estimator has only measured angles + known geometry/initial pose.
    readings = scaled_encoder_readings(shared["wheel_angles_rad"], scale)
    return evaluate_readings(shared, readings)


def evaluate_readings(shared, readings):
    """Estimate from supplied measurements; use truth only for diagnostics."""
    estimated = estimate_poses(readings)
    # Downstream mapping demonstration; NEVER fed back to the estimator or plant.
    mapped = np.array(
        [
            to_parent(compose(pose, SENSOR_IN_BODY), observation)
            for pose, observation in zip(estimated, shared["landmark_sensor"])
        ]
    )
    true = shared["true_poses"]
    errors = np.linalg.norm(estimated[:, :2] - true[:, :2], axis=1)
    yaw_errors = heading_error(estimated[:, 2], true[:, 2])
    landmark_errors = np.linalg.norm(mapped - LANDMARK_WORLD, axis=1)
    arrays = {
        "encoder_angles_rad": readings,
        "poses": estimated,
        "mapped_landmark": mapped,
        "position_error_m": errors,
        "heading_error_rad": yaw_errors,
        "landmark_error_m": landmark_errors,
    }
    metrics = {
        "endpoint": estimated[-1].tolist(),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(errors.max()),
        "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
        "final_heading_error_deg": float(np.rad2deg(yaw_errors[-1])),
        "max_abs_heading_error_deg": float(np.rad2deg(np.abs(yaw_errors).max())),
        "final_landmark_error_m": float(landmark_errors[-1]),
        "max_landmark_error_m": float(landmark_errors.max()),
    }
    return arrays, metrics


def run_case(key):
    wheels, checkpoints = schedule(key)
    shared = simulate_truth(wheels)
    archive = dict(shared)
    estimates = []
    for variant, label, scale, color in VARIANTS:
        arrays, metrics = evaluate_estimate(shared, scale)
        archive.update({f"{variant}_{name}": value for name, value in arrays.items()})
        estimates.append(
            {"key": variant, "label": label, "right_scale": scale, "color": color, **metrics}
        )
    expected = np.array([2.4, 0, 0]) if key == "straight" else np.array([0, 0, 2 * np.pi])
    endpoint = shared["true_poses"][-1]
    label = next(label for k, label, _ in SCENARIOS if k == key)
    metadata = {
        "key": key,
        "label": label,
        "steps": len(wheels),
        "dt_s": DT,
        "checkpoints": checkpoints,
        "true_endpoint": endpoint.tolist(),
        "expected_true_endpoint": expected.tolist(),
        "true_endpoint_position_error_m": float(np.linalg.norm(endpoint[:2] - expected[:2])),
        "true_endpoint_yaw_error_rad": float(abs(endpoint[2] - expected[2])),
        "estimates": estimates,
    }
    return archive, metadata


def make_plot(all_arrays, output, variants=VARIANTS):
    configure_plot_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for row, (key, label, _) in enumerate(SCENARIOS):
        a = all_arrays[key]
        times = np.arange(len(a["true_poses"])) * DT
        for variant, name, _, color in variants:
            axes[row, 0].plot(
                *a[f"{variant}_poses"][:, :2].T, color=color, linestyle="--", label=name
            )
            axes[row, 1].plot(
                times, a[f"{variant}_position_error_m"] * 100, color=color, label=name
            )
            axes[row, 2].plot(
                times, np.rad2deg(a[f"{variant}_heading_error_rad"]), color=color, label=name
            )
        axes[row, 0].plot(
            *a["true_poses"][:, :2].T, color="#2563eb", linewidth=2, label="真实轨迹（仅一台车）"
        )
        axes[row, 0].set(xlabel="世界 X / m", ylabel="世界 Y / m", title=label)
        axes[row, 0].axis("equal")
        axes[row, 1].set(
            xlabel="仿真时间 / s", ylabel="位置误差 / cm", title="累计误差不必每时刻单调增大"
        )
        axes[row, 2].set(
            xlabel="仿真时间 / s", ylabel="朝向估计误差 / °", title="右轮多报转动 → 误以为更向左转"
        )
        for ax in axes[row]:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def run_experiment(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    by_key, archive, cases = {}, {}, []
    for key, _, _ in SCENARIOS:
        arrays, case = run_case(key)
        by_key[key] = arrays
        archive.update({f"{key}_{name}": value for name, value in arrays.items()})
        cases.append(case)
    np.savez_compressed(output / "trajectories.npz", **archive)
    base = Path(__file__).resolve().parents[1]
    sources = [
        "differential_drive.py",
        "odometry.py",
        "experiments/mobile_frames.py",
        "experiments/mobile_odometry.py",
    ]
    report = {
        "experiment": "differential_drive_odometry",
        "schema_version": 1,
        "model": "ideal_no_slip_velocity_kinematics",
        "dt_s": DT,
        "wheel_radius_m": GEOMETRY.radius_m,
        "track_width_m": GEOMETRY.track_m,
        "sensor_in_body": SENSOR_IN_BODY.tolist(),
        "landmark_world_m": LANDMARK_WORLD.tolist(),
        "state_order": ["x_m", "y_m", "unwrapped_yaw_rad"],
        "wheel_order": ["left", "right"],
        "bias_model": "right_measured_angle = right_true_angle * right_scale",
        "alignment": "wheel command N intervals; angles/poses/observations/errors N+1 samples; estimator consumes previous interval only",
        "initial_pose_known": True,
        "encoder_quantization": False,
        "limitations": [
            "No dynamics, slip, random noise, gyro, landmarks used for correction or feedback control",
            "Known initial pose and exact geometry",
            "Synthetic encoder angles come from ideal wheel rotation, not hardware",
            "Scale bias changes readings only; a single truth trajectory per scenario",
        ],
        "source_sha256": {
            name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in sources
        },
        "trajectories_sha256": hashlib.sha256(
            (output / "trajectories.npz").read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(by_key, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.output)
    for case in report["cases"]:
        for estimate in case["estimates"]:
            print(
                f"{case['key']}/{estimate['key']}: final position error {estimate['final_position_error_m']:.6f} m; heading {estimate['final_heading_error_deg']:+.6f} deg; max position error {estimate['max_position_error_m']:.6f} m"
            )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
