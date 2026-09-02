"""Lesson 18: absolute landmark observations versus cumulative odometry.

One ideal differential car, the same frozen routes. The right-encoder scale was
already calibrated in lesson 16 and per-interval noise added in lesson 17; here
a forward-looking sensor additionally measures distance+bearing to three known
landmarks every OBS_PERIOD_STEPS intervals. A closed-form 2D orthogonal
Procrustes fit turns those readings into a pose estimate, so each landmark
sample is independent: its error does not accumulate, while odometry drift keeps
growing. No filter is fused here; observation and odometry stay separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import (
    DriveGeometry,
    compose,
    rotation,
    to_child,
)
from embodied_learning.experiments.mobile_frames import DT, SENSOR_IN_BODY, wheel_schedule
from embodied_learning.experiments.mobile_noise import (
    INTERVAL_NOISE_STD_RAD,
    ROUTE_SEED_OFFSET,
    calibrated_factor,
    noise_sequence,
    noisy_readings,
)
from embodied_learning.experiments.mobile_odometry import SCENARIOS, schedule, simulate_truth

LONG_SCENARIO = ("long", "长直行 32 秒（6.4 m）", 800)
SCENARIOS = SCENARIOS + (LONG_SCENARIO,)
from embodied_learning.odometry import estimate_poses, heading_error

EXPERIMENT = "differential_drive_landmark_observations"
# Known control points in world coordinates; identification is assumed.
# The three landmarks form a wide, non-degenerate triangle around the routes.
LANDMARKS = np.array([[0.0, 1.6], [2.6, 1.0], [1.6, -0.9]])
# Sensor samples every OBS_PERIOD_STEPS intervals (2 s), all three landmarks.
OBS_PERIOD_STEPS = 50
OBS_RANGE_STD_M = 0.01
OBS_BEARING_STD_RAD = 0.01
DEFAULT_RUNS = 20
DEFAULT_SEED = 0
GEOMETRY = DriveGeometry()
HOLD_INITIAL = ("初次观测前保持初始位姿已知值", (0.0, 0.0, 0.0))
LONG_SEED_OFFSET = 2_000_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inverse_pose(pose):
    """Inverse SE(2): pose^-1 such that compose(pose, inverse) is identity."""
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError("Expected finite [x, y, yaw] pose")
    shift = -(rotation(pose[2]).T @ pose[:2])
    return np.array([shift[0], shift[1], -pose[2]])


def bearing_reading(landmark, sensor_pose):
    """Distance and bearing of a known landmark IN THE SENSOR FRAME.

    The sensor axis rotates with the body: bearing must be measured against the
    sensor frame, exactly as the lesson-14 coordinate chain insists. Using world
    axes here would silently ignore the sensor's +30 deg yaw and any turning.
    """
    point = to_child(sensor_pose, np.asarray(landmark, dtype=float))
    return float(np.linalg.norm(point)), float(math.atan2(point[1], point[0]))


def observe(pose, landmarks, rng, range_std=OBS_RANGE_STD_M, bearing_std=OBS_BEARING_STD_RAD):
    """One sample: noisy distance+bearing of every landmark from the sensor.

    The sensor frame is SENSOR_IN_BODY attached to the true pose. Noise is
    independent per landmark and per sample; nothing here knows the estimator.
    """
    sensor = compose(pose, SENSOR_IN_BODY)
    readings = np.empty((len(landmarks), 2))
    for index, landmark in enumerate(landmarks):
        distance, bearing = bearing_reading(landmark, sensor)
        readings[index] = [
            distance + rng.normal(0.0, range_std),
            bearing + rng.normal(0.0, bearing_std),
        ]
    return readings


def solve_pose(readings, landmarks):
    """Sensor pose from noisy distance+bearing to >=2 known landmarks.

    Converts polar readings to sensor-frame coordinates, then fits the rigid
    rotation+translation that maps sensor coordinates onto world control points
    (2D orthogonal Procrustes, closed form). The sensor installation pose is
    then inverted to recover the body pose. No scaling is fitted: metres are the
    unit on both sides and range noise was drawn in metres.
    """
    readings = np.asarray(readings, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)
    if readings.ndim != 2 or readings.shape != (len(landmarks), 2):
        raise ValueError("Readings must match landmarks")
    if len(landmarks) < 2 or not np.isfinite(readings).all() or not np.isfinite(landmarks).all():
        raise ValueError("Need at least two finite landmarks")
    if np.any(readings[:, 0] <= 0):
        raise ValueError("Range readings must be positive")
    # Sensor-frame coordinates implied by each (range, bearing) reading.
    polar = np.column_stack(
        [readings[:, 0] * np.cos(readings[:, 1]), readings[:, 0] * np.sin(readings[:, 1])]
    )
    center_z, center_l = polar.mean(axis=0), landmarks.mean(axis=0)
    zz, ll = polar - center_z, landmarks - center_l
    dot = float(zz[:, 0] @ ll[:, 0] + zz[:, 1] @ ll[:, 1])
    cross = float(zz[:, 0] @ ll[:, 1] - zz[:, 1] @ ll[:, 0])
    yaw = math.atan2(cross, dot)
    translation = center_l - rotation(yaw) @ center_z
    sensor_pose = [translation[0], translation[1], yaw]
    return compose(sensor_pose, inverse_pose(SENSOR_IN_BODY))


def hold_estimate(poses_at, observation_frames, initial=(0.0, 0.0, 0.0), steps=None):
    """Piecewise-constant pose estimate: newest observation wins, no dead-reckon."""
    if steps is None:
        steps = observation_frames[-1]
    frames = np.arange(steps + 1)
    result = np.empty((steps + 1, 3))
    result[0] = initial
    previous = 0
    for frame in frames[1:]:
        if previous < len(observation_frames) and frame >= observation_frames[previous]:
            result[frame] = poses_at[previous]
            previous += 1
        else:
            result[frame] = result[frame - 1]
    return result


def run_runs(scenario_key, runs, seed):
    if scenario_key == "long":
        wheels = np.tile(wheel_schedule("straight"), (8, 1))
    else:
        wheels, _ = schedule(scenario_key)
    steps = len(wheels)
    shared = simulate_truth(wheels)
    true = shared["true_poses"]
    observation_frames = list(range(OBS_PERIOD_STEPS, steps + 1, OBS_PERIOD_STEPS))
    odom_poses, lm_poses = [], []
    observed_errors = []
    for r in range(runs):
        route_offset = ROUTE_SEED_OFFSET.get(scenario_key, LONG_SEED_OFFSET)
        rng = np.random.default_rng(seed + route_offset + r)
        readings = noisy_readings(
            shared["wheel_angles_rad"], noise_sequence(rng, steps), calibrated_factor()[0]
        )
        odom = estimate_poses(readings)
        odom_poses.append(odom)
        samples = []
        for frame in observation_frames:
            samples.append(solve_pose(observe(true[frame], LANDMARKS, rng), LANDMARKS))
        held = hold_estimate(samples, observation_frames, steps=steps)
        lm_poses.append(held)
        for frame, sample in zip(observation_frames, samples):
            observed_errors.append(
                {
                    "run": r,
                    "frame": frame,
                    "time_s": frame * DT,
                    "seed": seed + route_offset + r,
                    "position_error_m": float(np.linalg.norm(sample[:2] - true[frame, :2])),
                    "heading_error_rad": float(heading_error(sample[2], true[frame, 2])),
                }
            )
    odom_poses = np.asarray(odom_poses)
    lm_poses = np.asarray(lm_poses)
    odom_err = np.linalg.norm(odom_poses[:, :, :2] - true[:, :2], axis=2)
    lm_err = np.linalg.norm(lm_poses[:, :, :2] - true[:, :2], axis=2)
    obs_pos = np.array([e["position_error_m"] for e in observed_errors])
    return {
        "true_poses": true,
        "wheels": wheels,
        "wheel_angles": shared["wheel_angles_rad"],
        "odom_poses": odom_poses,
        "landmark_poses": lm_poses,
        "odom_position_error": odom_err,
        "landmark_position_error": lm_err,
        "observation_frames": observation_frames,
        "observations": np.array([e["position_error_m"] for e in observed_errors]),
        "stats": {
            "odom_final_mean": float(odom_err[:, -1].mean()),
            "odom_final_std": float(odom_err[:, -1].std(ddof=1)) if runs > 1 else 0.0,
            "landmark_final_mean": float(lm_err[:, -1].mean()),
            "landmark_final_std": float(lm_err[:, -1].std(ddof=1)) if runs > 1 else 0.0,
            "observed_mean": float(obs_pos.mean()),
            "observed_std": float(obs_pos.std(ddof=1)) if len(obs_pos) > 1 else 0.0,
            "observed_at_times": [
                {
                    "time_s": frame * DT,
                    "mean_m": float(obs_pos[frame_index :: len(observation_frames)].mean()),
                    "std_m": float(obs_pos[frame_index :: len(observation_frames)].std(ddof=1))
                    if runs > 1
                    else 0.0,
                }
                for frame_index, frame in enumerate(observation_frames)
            ],
        },
        "observed_table": observed_errors,
    }


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two repetitions for dispersion statistics")
    archive, cases = {}, []
    for scenario_key, _, _ in SCENARIOS:
        arrays = run_runs(scenario_key, runs, seed)
        archive.update(
            {
                f"{scenario_key}_{name}": value
                for name, value in arrays.items()
                if name not in ("observation_frames", "observations", "stats", "observed_table")
            }
        )
        cases.append(
            {
                "key": scenario_key,
                "steps": len(arrays["wheels"]),
                "dt_s": DT,
                "observation_frames": arrays["observation_frames"],
                "stats": arrays["stats"],
                "observed_table": arrays["observed_table"],
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "model": "ideal_no_slip_velocity_kinematics",
        "dt_s": DT,
        "wheel_radius_m": GEOMETRY.radius_m,
        "track_width_m": GEOMETRY.track_m,
        "sensor_in_body": list(SENSOR_IN_BODY),
        "landmarks_world_m": LANDMARKS.tolist(),
        "observation_period_steps": OBS_PERIOD_STEPS,
        "observation_period_s": OBS_PERIOD_STEPS * DT,
        "range_noise_std_m": OBS_RANGE_STD_M,
        "bearing_noise_std_rad": OBS_BEARING_STD_RAD,
        "odometry_reading_scale": 1.02,
        "odometry_correction_factor": 1 / 1.02,
        "odometry_interval_noise_std_rad": INTERVAL_NOISE_STD_RAD,
        "runs_per_group": runs,
        "base_seed": seed,
        "seeds": list(range(seed, seed + runs)),
        "landmark_identification_assumed": True,
        "filter_fused": False,
        "cases": cases,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            name: digest(Path(__file__).resolve().parents[1] / name)
            for name in [
                "differential_drive.py",
                "odometry.py",
                "experiments/mobile_frames.py",
                "experiments/mobile_odometry.py",
                "experiments/mobile_noise.py",
                "experiments/landmark_observations.py",
            ]
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "Landmark identity and world coordinates are assumed known",
            "No occlusions, FOV limits, range limits or detection failures",
            "Sensor samples are synchronised to the simulation clock",
            "Procrustes uses all landmarks every sample; no fusion with odometry",
            "Both estimators share the same truth and initial pose; no SLAM",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(archive, cases, output)
    return summary


def make_plot(archive, cases, output):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for row, (scenario_key, label, _) in enumerate(SCENARIOS):
        case = next(c for c in cases if c["key"] == scenario_key)
        times = np.arange(len(archive[f"{scenario_key}_true_poses"])) * DT
        true = archive[f"{scenario_key}_true_poses"]
        axes[row, 0].plot(*true[:, :2].T, color="#2563eb", lw=2.5, label="真实轨迹")
        odom = archive[f"{scenario_key}_odom_poses"][0]
        lm = archive[f"{scenario_key}_landmark_poses"][0]
        axes[row, 0].plot(
            *odom[:, :2].T, color="#9333ea", ls="--", lw=1.5, label="里程计（第 0 个种子）"
        )
        axes[row, 0].plot(
            *lm[:, :2].T, color="#0f766e", ls=":", lw=1.5, label="地标观测（保持上次）"
        )
        axes[row, 0].plot(*LANDMARKS.T, "k^", ms=7, label="已知地标（控制点）")
        for frame in case["observation_frames"]:
            axes[row, 0].plot(
                *lm[frame, :2], marker="o", ms=5, color="#0f766e", zorder=5, mec="white", mew=0.6
            )
        axes[row, 0].set(xlabel="世界 X / m", ylabel="世界 Y / m", title=label)
        axes[row, 0].axis("equal")
        axes[row, 0].legend(fontsize=8)
        observed = np.array([e["position_error_m"] for e in case["observed_table"]]) * 100
        frames = np.array([e["time_s"] for e in case["observed_table"]])
        axes[row, 1].plot(
            times,
            archive[f"{scenario_key}_odom_position_error"][0] * 100,
            color="#9333ea",
            lw=1.6,
            label="里程计（累积漂移）",
        )
        axes[row, 1].plot(
            times,
            archive[f"{scenario_key}_landmark_position_error"][0] * 100,
            color="#94a3b8",
            lw=1.1,
            ls="--",
            label="观测后保持旧值（锯齿=周内误差线性增大）",
        )
        axes[row, 1].scatter(
            frames, observed, s=20, color="#0f766e", zorder=5, label="观测时刻误差（小且不累积）"
        )
        axes[row, 1].set(
            xlabel="仿真时间 / s",
            ylabel="位置误差 / cm",
            title="观测时刻误差与时间无关；两次观测之间靠旧值顶替",
        )
        axes[row, 1].legend(fontsize=8)
        axes[row, 2].scatter(frames, observed, s=14, color="#0f766e", alpha=0.55)
        samples = case["stats"]["observed_at_times"]
        axes[row, 2].plot(
            [s["time_s"] for s in samples],
            [s["mean_m"] * 100 for s in samples],
            color="#0f172a",
            lw=1.6,
            marker="o",
            ms=4,
            label="各观测时刻 20 次均值",
        )
        axes[row, 2].set(
            xlabel="观测时刻 / s",
            ylabel="观测定位误差 / cm",
            title="每个观测时刻都独立：均值不随路线增长",
        )
        axes[row, 2].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for dispersion statistics")
    report = run_experiment(args.output, runs=args.runs, seed=args.seed)
    for case in report["cases"]:
        stats = case["stats"]
        print(
            f"{case['key']}: odom final mean {stats['odom_final_mean'] * 100:.2f} cm "
            f"(std {stats['odom_final_std'] * 100:.2f}); landmark final mean "
            f"{stats['landmark_final_mean'] * 100:.2f} cm (std {stats['landmark_final_std'] * 100:.2f}); "
            f"observed mean {stats['observed_mean'] * 100:.2f} cm"
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
