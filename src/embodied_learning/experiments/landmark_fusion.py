"""Lesson 19: recorded range/bearing -> body pose -> reset-and-propagate fusion.

The lesson-18 routes and seeded measurements are retained. Only the estimator is
new; raw observations and pre-reset predictions are saved for causal teaching.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.experiments.landmark_observations import (
    DEFAULT_RUNS,
    DEFAULT_SEED,
    GEOMETRY,
    LANDMARKS,
    LONG_SEED_OFFSET,
    OBS_BEARING_STD_RAD,
    OBS_PERIOD_STEPS,
    OBS_RANGE_STD_M,
    SCENARIOS,
    digest,
    hold_estimate,
    observe,
    solve_pose,
)
from embodied_learning.experiments.mobile_frames import DT, SENSOR_IN_BODY, wheel_schedule
from embodied_learning.experiments.mobile_noise import (
    INPUT_RIGHT_SCALE,
    INTERVAL_NOISE_STD_RAD,
    ROUTE_SEED_OFFSET,
    calibrated_factor,
    noise_sequence,
    noisy_readings,
)
from embodied_learning.experiments.mobile_odometry import schedule, simulate_truth
from embodied_learning.odometry import estimate_poses
from embodied_learning.pose_fusion import estimate_with_resets

EXPERIMENT = "landmark_reset_fusion"
DEFAULT_RESULTS = "results/mobile_fusion_2026-09-03"
METHODS = (
    ("odom", "纯里程计", "#9333ea"),
    ("held", "纯观测保持", "#64748b"),
    ("fused", "观测重置＋里程计", "#ea580c"),
)
CONFIG = {
    "experiment": EXPERIMENT,
    "schema_version": 1,
    "model": "ideal_no_slip_velocity_kinematics",
    "dt_s": DT,
    "wheel_radius_m": GEOMETRY.radius_m,
    "track_width_m": GEOMETRY.track_m,
    "sensor_in_body": list(SENSOR_IN_BODY),
    "landmarks_world_m": LANDMARKS.tolist(),
    "observation_period_steps": OBS_PERIOD_STEPS,
    "range_noise_std_m": OBS_RANGE_STD_M,
    "bearing_noise_std_rad": OBS_BEARING_STD_RAD,
    "encoder_scale": INPUT_RIGHT_SCALE,
    "encoder_interval_noise_std_rad": INTERVAL_NOISE_STD_RAD,
    "fusion_rule": "predict_then_reset_at_current_frame; nearest_yaw_branch",
}


def method_stats(poses, truth):
    """Distance norms, signed endpoint bias and spread have distinct meanings."""
    xy_error = poses[:, :, :2] - truth[None, :, :2]
    distance = np.linalg.norm(xy_error, axis=2)
    per_run_mean = distance[:, 1:].mean(axis=1)  # exclude the exact known initial state
    return {
        "time_mean_position_m": float(per_run_mean.mean()),
        "time_mean_position_std_m": float(per_run_mean.std(ddof=1)),
        "final_position_mean_m": float(distance[:, -1].mean()),
        "final_position_std_m": float(distance[:, -1].std(ddof=1)),
        "endpoint_signed_xy_mean_m": xy_error[:, -1].mean(axis=0).tolist(),
        "endpoint_xy_std_m": xy_error[:, -1].std(axis=0, ddof=1).tolist(),
    }


def run_runs(key, runs, seed):
    if key not in {s[0] for s in SCENARIOS}:
        raise ValueError("Unknown route")
    wheels = np.tile(wheel_schedule("straight"), (8, 1)) if key == "long" else schedule(key)[0]
    shared = simulate_truth(wheels)
    truth = shared["true_poses"]
    steps = len(wheels)
    frames = np.arange(OBS_PERIOD_STEPS, steps + 1, OBS_PERIOD_STEPS)
    factor, _ = calibrated_factor()
    values = {name: [] for name in ("encoders", "observations", "body_samples", "prior")}
    values.update({name: [] for name, _, _ in METHODS})
    for run in range(runs):
        # Deliberately identical draw order to lesson 18 for an exact paired comparison.
        rng = np.random.default_rng(seed + ROUTE_SEED_OFFSET.get(key, LONG_SEED_OFFSET) + run)
        encoders = noisy_readings(shared["wheel_angles_rad"], noise_sequence(rng, steps), factor)
        observations = np.array([observe(truth[f], LANDMARKS, rng) for f in frames])
        samples = np.array([solve_pose(reading, LANDMARKS) for reading in observations])
        fused, prior = estimate_with_resets(encoders, frames, samples)
        run_values = {
            "encoders": encoders,
            "observations": observations,
            "body_samples": samples,
            "odom": estimate_poses(encoders),
            "held": hold_estimate(samples, frames, steps=steps),
            "fused": fused,
            "prior": prior,
        }
        for name, value in run_values.items():
            values[name].append(value)
    arrays = {name: np.asarray(value) for name, value in values.items()}
    arrays.update(truth=truth, wheels=wheels, observation_frames=frames)
    return arrays


def case_stats(arrays):
    truth, frames = arrays["truth"], arrays["observation_frames"]
    before = np.linalg.norm(arrays["prior"][:, frames, :2] - truth[frames, :2], axis=2)
    after = np.linalg.norm(arrays["fused"][:, frames, :2] - truth[frames, :2], axis=2)
    return {
        "methods": {key: method_stats(arrays[key], truth) for key, _, _ in METHODS},
        "updates": {
            "count": int(before.size),
            "worse_count": int(np.sum(after > before + 1e-12)),
            "mean_before_position_m": float(before.mean()),
            "mean_after_position_m": float(after.mean()),
        },
    }


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(runs) is not int or runs < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Need integer runs >= 2 and seed >= 0")
    archive, cases = {}, []
    for key, _, steps in SCENARIOS:
        arrays = run_runs(key, runs, seed)
        archive.update({f"{key}_{name}": value for name, value in arrays.items()})
        cases.append({"key": key, "steps": steps, **case_stats(arrays)})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    make_plot(archive, output, runs)
    factor, rows = calibrated_factor()
    source_names = [
        "landmark_localization.py",
        "differential_drive.py",
        "odometry.py",
        "pose_fusion.py",
        "encoder_calibration.py",
        "experiments/mobile_frames.py",
        "experiments/mobile_odometry.py",
        "experiments/encoder_calibration.py",
        "experiments/mobile_noise.py",
        "experiments/landmark_observations.py",
        "experiments/landmark_fusion.py",
    ]
    summary = {
        **CONFIG,
        "runs": runs,
        "seed": seed,
        "route_seed_offsets": {**ROUTE_SEED_OFFSET, "long": LONG_SEED_OFFSET},
        "calibration_factor": factor,
        "calibration_segments": rows,
        "calibration_protocol": "lesson-16 independent synthetic straight runs, recomputed",
        "metric_time_frames": "1..steps; mean over time per run, then mean over runs",
        "cases": cases,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "comparison_sha256": digest(output / "comparison.png"),
        "source_sha256": {
            name: digest(Path(__file__).resolve().parents[1] / name) for name in source_names
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "Synthetic range/bearing only; no camera/LiDAR rendering or landmark detection",
            "Known initial pose, geometry, landmark coordinates/identity and sensor mounting",
            "No slip, dynamics, occlusion, FOV/range limit, latency or outlier rejection",
            "Resetting pose is not online encoder calibration or observation denoising",
            "No covariance, adaptive weights, Kalman filter, SLAM or feedback control",
            "Truth only generates measurements and evaluates error; estimators do not receive it",
            "A noisy absolute reset can increase instantaneous error; no optimality claim",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def make_plot(archive, output, runs):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), layout="constrained")
    for row, (key, label, _) in enumerate(SCENARIOS):
        truth = archive[f"{key}_truth"]
        t = np.arange(len(truth)) * DT
        for col, include_hold in enumerate((True, False)):
            ax = axes[row, col]
            for method, name, color in METHODS:
                if method == "held" and not include_hold:
                    continue
                errors = (
                    np.linalg.norm(archive[f"{key}_{method}"][:, :, :2] - truth[:, :2], axis=2)
                    * 100
                )
                ax.plot(t, errors.mean(axis=0), color=color, label=name)
            ax.set(
                title=f"{label} · {'完整三组' if include_hold else '隐藏保持组，重算纵轴'}",
                xlabel="仿真时间 / s",
                ylabel=f"{runs} 次平均位置误差距离 / cm",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
    fig.suptitle("重置修正漂移，轮子填观测间隙；重置不是消除观测噪声")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def load_recording(directory):
    """Validate provenance, shapes and estimator consistency before read-only replay."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if any(report.get(key) != value for key, value in CONFIG.items()):
        raise ValueError("Incompatible lesson-19 recording")
    runs, seed = report.get("runs"), report.get("seed")
    if type(runs) is not int or runs < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid repetition settings")
    if [c["key"] for c in report["cases"]] != [s[0] for s in SCENARIOS]:
        raise ValueError("Invalid route list")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    routes = {}
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = set()
        for key, _, steps in SCENARIOS:
            frames = np.arange(OBS_PERIOD_STEPS, steps + 1, OBS_PERIOD_STEPS)
            shapes = {
                "truth": (steps + 1, 3),
                "wheels": (steps, 2),
                "encoders": (runs, steps + 1, 2),
                "observations": (runs, len(frames), len(LANDMARKS), 2),
                "body_samples": (runs, len(frames), 3),
                "observation_frames": (len(frames),),
                **{name: (runs, steps + 1, 3) for name in ("odom", "held", "fused", "prior")},
            }
            route = {}
            for name, shape in shapes.items():
                expected_keys.add(f"{key}_{name}")
                value = archive[f"{key}_{name}"].copy()
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid array: {key}_{name}")
                route[name] = value
            if not np.array_equal(route["observation_frames"], frames):
                raise ValueError("Invalid observation timing")
            for run in range(runs):
                samples = np.array([solve_pose(z, LANDMARKS) for z in route["observations"][run]])
                fused, prior = estimate_with_resets(route["encoders"][run], frames, samples)
                recomputed = {
                    "body_samples": samples,
                    "fused": fused,
                    "prior": prior,
                    "odom": estimate_poses(route["encoders"][run]),
                    "held": hold_estimate(samples, frames, steps=steps),
                }
                for name, value in recomputed.items():
                    if not np.allclose(value, route[name][run], atol=1e-10, rtol=0):
                        raise ValueError(f"Inconsistent estimator array: {name}")
            for value in route.values():
                value.flags.writeable = False
            routes[key] = route
        if set(archive.files) != expected_keys:
            raise ValueError("Unexpected archive arrays")
    return routes, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2 or args.seed < 0:
        parser.error("--runs >= 2 and --seed >= 0 required")
    summary = run_experiment(args.output, runs=args.runs, seed=args.seed)
    for case in summary["cases"]:
        print(
            case["key"],
            {
                key: round(stats["time_mean_position_m"] * 100, 3)
                for key, stats in case["methods"].items()
            },
            "time-mean error / cm; worsening updates",
            case["updates"]["worse_count"],
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
