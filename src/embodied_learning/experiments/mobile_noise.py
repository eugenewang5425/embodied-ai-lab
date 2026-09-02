"""Lesson 17: after fixed-ratio calibration, per-interval reading noise remains.

The lesson-16 factor from independent straight runs is applied unchanged (frozen).
The ONLY new change: each 0.04 s interval the right encoder adds an independent
zero-mean noise draw to its reported increment. The same route is repeated with
many seeds so systematic bias (ensemble mean) and dispersion (ensemble spread)
can be separated. Truth, commands and the calibration segments never use the
evaluation routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.encoder_calibration import fit_right_correction
from embodied_learning.experiments.encoder_calibration import calibration_measurements
from embodied_learning.experiments.mobile_frames import DT, GEOMETRY
from embodied_learning.experiments.mobile_odometry import SCENARIOS, schedule, simulate_truth
from embodied_learning.odometry import estimate_poses, heading_error

EXPERIMENT = "encoder_interval_noise"
# Lesson-16 synthetic instrument: the right encoder over-reports every rotation by 2%.
INPUT_RIGHT_SCALE = 1.02
# Per-interval noise added to the raw right measured increment (after the +2% scale).
# 0.008 rad is about 5% of one nominal 0.16 rad step; it does NOT accumulate as a ratio.
INTERVAL_NOISE_STD_RAD = 0.008
DEFAULT_RUNS = 20
DEFAULT_SEED = 0
ROUTE_SEED_OFFSET = {"straight": 0, "square": 1_000_000}
GROUPS = (
    ("noiseless", "无噪声（第十六课理想基准）", "#0f766e"),
    ("fixed", "固定标定后 + 逐区间噪声", "#2563eb"),
    ("uncorrected", "未标定 + 逐区间噪声", "#9333ea"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibrated_factor():
    """Fit c from the same four independent segments as lesson 16 (recomputed here)."""
    rows = calibration_measurements(INPUT_RIGHT_SCALE)
    angles = [row["measured_right_angle_rad"] for row in rows]
    distances = np.array([row["external_distance_m"] for row in rows])
    factor = fit_right_correction(angles, distances, GEOMETRY.radius_m)
    if not np.isclose(factor, 1 / INPUT_RIGHT_SCALE, atol=1e-12):
        raise RuntimeError("Calibration factor no longer matches lesson-16 result")
    return factor, rows


def noisy_readings(true_angles, epsilon, correction):
    """Raw = 1.02*true + noise; reported = correction*raw (correction=1 keeps raw).

    Returns cumulative [left, right] arrays shaped like true_angles. Left wheel
    and the absolute starting counters stay untouched. Noise is a property of the
    measurement, never of the plant or of the scheduled speeds.
    """
    true = np.asarray(true_angles, dtype=float)
    if true.ndim != 2 or true.shape[1] != 2 or len(true) < 1 or not np.isfinite(true).all():
        raise ValueError("Expected finite [left, right] cumulative angles")
    epsilon = np.asarray(epsilon, dtype=float)
    if epsilon.shape != (len(true) - 1,):
        raise ValueError("One noise draw per measured interval")
    if not np.isfinite(epsilon).all():
        raise ValueError("Finite noise required")
    if not np.isfinite(correction) or correction <= 0:
        raise ValueError("Correction factor must be finite and positive")
    left, right = np.diff(true, axis=0).T
    raw_right = INPUT_RIGHT_SCALE * right + epsilon
    readings = np.empty_like(true)
    readings[0] = true[0]
    readings[1:] = readings[0] + np.column_stack([left, correction * raw_right]).cumsum(axis=0)
    return readings


def noise_sequence(rng, steps):
    return rng.normal(0.0, INTERVAL_NOISE_STD_RAD, size=steps)


def evaluate(shared, readings):
    """Estimate poses from measurements only; truth is used solely for diagnostics."""
    estimated = estimate_poses(readings)
    true = shared["true_poses"]
    errors = np.linalg.norm(estimated[:, :2] - true[:, :2], axis=1)
    yaw_errors = heading_error(estimated[:, 2], true[:, 2])
    arrays = {
        "readings": readings,
        "poses": estimated,
        "position_error_m": errors,
        "heading_error_rad": yaw_errors,
    }
    metrics = {
        "endpoint": estimated[-1].tolist(),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(errors.max()),
        "rms_position_error_m": float(np.sqrt(np.mean(errors**2))),
        "final_heading_error_deg": float(np.rad2deg(yaw_errors[-1])),
        "max_abs_heading_error_deg": float(np.rad2deg(np.abs(yaw_errors).max())),
        "endpoint_x_error_m": float(estimated[-1, 0] - true[-1, 0]),
        "endpoint_y_error_m": float(estimated[-1, 1] - true[-1, 1]),
    }
    return arrays, metrics


def ensemble_stats(values):
    """Final-frame and over-time statistics of [runs, frames] diagnostics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Expected [runs, frames] diagnostics")
    final = values[:, -1]
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros_like(mean)
    p10, p50, p90 = np.quantile(final, [0.1, 0.5, 0.9])
    return {
        "final_mean": float(final.mean()),
        "final_std": float(final.std(ddof=1)) if len(final) > 1 else 0.0,
        "final_rms": float(np.sqrt(np.mean(final**2))),
        "final_min": float(final.min()),
        "final_max": float(final.max()),
        "final_p10": float(p10),
        "final_p50": float(p50),
        "final_p90": float(p90),
        "mean_curve": mean,
        "std_curve": std,
    }


def scalar_stats(stats):
    """Drop per-frame numpy curves; they stay in trajectories.npz, not in JSON."""
    return {key: value for key, value in stats.items() if key not in ("mean_curve", "std_curve")}


def signed_endpoint_stats(errors):
    errors = np.asarray(errors, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 2:
        raise ValueError("Expected [runs, 2] endpoint errors")
    mean = errors.mean(axis=0)
    std = errors.std(axis=0, ddof=1) if len(errors) > 1 else np.zeros(2)
    return {
        "bias_x_m": float(mean[0]),
        "bias_y_m": float(mean[1]),
        "std_x_m": float(std[0]),
        "std_y_m": float(std[1]),
    }


def run_runs(scenario_key, factor, runs, seed):
    """Repeat one route: one truth for every run; noise differs per (route, run)."""
    wheels, checkpoints = schedule(scenario_key)
    steps = len(wheels)
    shared = simulate_truth(wheels)
    archive = {
        "true_poses": shared["true_poses"],
        "wheels": wheels,
        "wheel_angles": shared["wheel_angles_rad"],
        "checkpoints": checkpoints,
    }
    per_run = {}
    for group, _, _ in GROUPS:
        noisy = group != "noiseless"
        count = runs if noisy else 1
        correction = factor if group != "uncorrected" else 1.0
        poses_list, readings_list, eps_list, pos_list, yaw_list = [], [], [], [], []
        endpoint_errors, final_metrics = [], []
        for r in range(count):
            run_seed = seed + (r if noisy else 0)
            rng = np.random.default_rng(
                seed + ROUTE_SEED_OFFSET[scenario_key] + (r if noisy else 0)
            )
            epsilon = noise_sequence(rng, steps) if noisy else np.zeros(steps)
            readings = noisy_readings(shared["wheel_angles_rad"], epsilon, correction)
            arrays, metrics = evaluate(shared, readings)
            poses_list.append(arrays["poses"])
            readings_list.append(arrays["readings"])
            eps_list.append(epsilon)
            pos_list.append(arrays["position_error_m"])
            yaw_list.append(arrays["heading_error_rad"])
            endpoint_errors.append([metrics["endpoint_x_error_m"], metrics["endpoint_y_error_m"]])
            final_metrics.append(
                {
                    "seed": run_seed,
                    "final_position_error_m": metrics["final_position_error_m"],
                    "max_position_error_m": metrics["max_position_error_m"],
                    "rms_position_error_m": metrics["rms_position_error_m"],
                    "final_heading_error_deg": metrics["final_heading_error_deg"],
                    "endpoint": metrics["endpoint"],
                }
            )
        archive[f"{group}_poses"] = np.asarray(poses_list)
        archive[f"{group}_readings"] = np.asarray(readings_list)
        archive[f"{group}_epsilon"] = np.asarray(eps_list)
        archive[f"{group}_position_error"] = np.asarray(pos_list)
        archive[f"{group}_heading_error"] = np.asarray(yaw_list)
        per_run[group] = final_metrics
        per_run[f"{group}_ensemble_position"] = scalar_stats(ensemble_stats(np.asarray(pos_list)))
        per_run[f"{group}_ensemble_heading"] = scalar_stats(ensemble_stats(np.asarray(yaw_list)))
        per_run[f"{group}_endpoint_stats"] = signed_endpoint_stats(np.asarray(endpoint_errors))
    return archive, per_run, steps


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two noisy repetitions for dispersion statistics")
    factor, rows = calibrated_factor()
    archive, cases = {}, []
    for scenario_key, _, _ in SCENARIOS:
        arrays, per_run, steps = run_runs(scenario_key, factor, runs, seed)
        archive.update({f"{scenario_key}_{name}": value for name, value in arrays.items()})
        cases.append(
            {
                "key": scenario_key,
                "steps": steps,
                "dt_s": DT,
                "groups": per_run,
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
        "state_order": ["x_m", "y_m", "unwrapped_yaw_rad"],
        "wheel_order": ["left", "right"],
        "raw_measurement_model": "raw right increment = 1.02 * true + N(0, sigma)",
        "reported_measurement_model": "reported = correction * raw; correction = 1 for uncorrected",
        "interval_noise_std_rad": INTERVAL_NOISE_STD_RAD,
        "input_right_scale": INPUT_RIGHT_SCALE,
        "correction_factor": factor,
        "calibration_instrument": "lesson-16 independent signed straight runs (recomputed)",
        "calibration_runs": rows,
        "runs_per_group": runs,
        "base_seed": seed,
        "seeds": list(range(seed, seed + runs)),
        "common_noise_across_groups_per_run": True,
        "groups": [{"key": key, "label": label, "color": color} for key, label, color in GROUPS],
        "cases": cases,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            name: digest(Path(__file__).resolve().parents[1] / name)
            for name in [
                "differential_drive.py",
                "odometry.py",
                "encoder_calibration.py",
                "experiments/mobile_frames.py",
                "experiments/mobile_odometry.py",
                "experiments/encoder_calibration.py",
                "experiments/mobile_noise.py",
            ]
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "Synthetic per-interval noise on the right encoder only; left wheel exact",
            "Known initial pose, exact geometry, no slip, no quantization or delay",
            "Fixed c cannot remove per-interval randomness; no filter, landmark or SLAM",
            "Repeated runs share one truth trajectory; statistics describe the estimator",
            "No hardware accuracy claim: sensors and metrology are simulated",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(archive, output)
    return summary


def make_plot(archive, output):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    symbols = {"noiseless": ("#0f766e", 8), "fixed": ("#2563eb", 4), "uncorrected": ("#9333ea", 4)}
    for row, (key, label, _) in enumerate(SCENARIOS):
        times = np.arange(len(archive[f"{key}_true_poses"])) * DT
        true = archive[f"{key}_true_poses"]
        axes[row, 0].plot(0, 0, marker="*", color="#0f172a", ms=14, label="真实终点")
        for group, (color, size) in symbols.items():
            poses = archive[f"{key}_{group}_poses"]
            for run in range(len(poses)):
                dx, dy = poses[run, -1, :2] - true[-1, :2]
                axes[row, 0].plot(dx, dy, marker="o", ms=size, color=color)
            axes[row, 0].plot([], [], color=color, label=f"{group} 终点")
        axes[row, 0].axhline(0, color="#cbd5e1", lw=0.8)
        axes[row, 0].axvline(0, color="#cbd5e1", lw=0.8)
        axes[row, 0].set(
            xlabel="终点 X 误差 / m",
            ylabel="终点 Y 误差 / m",
            title=f"{label}：同一条真值，20 个噪声样本的终点",
        )
        axes[row, 0].axis("equal")
        axes[row, 0].legend(fontsize=8, loc="upper left")
        for col, (name, scale, unit, title) in enumerate(
            [
                ("position_error", 100, "cm", "20 次均值 ± 1σ：系统偏差与分散程度"),
                (
                    "heading_error",
                    180 / np.pi,
                    "°",
                    "标定后均值近零且仅有分散；未标定仍有固定偏转",
                ),
            ],
            start=1,
        ):
            for group, (color, _) in symbols.items():
                values = archive[f"{key}_{group}_{name}"] * scale
                mean = values.mean(axis=0)
                std = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
                axes[row, col].plot(times, mean, color=color, label=group)
                axes[row, col].fill_between(
                    times,
                    np.clip(mean - std, 0, None) if unit == "cm" else mean - std,
                    mean + std,
                    color=color,
                    alpha=0.18,
                    lw=0,
                )
            axes[row, col].set(xlabel="仿真时间 / s", ylabel=f"误差 / {unit}", title=title)
            axes[row, col].legend(fontsize=8)
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
        stats = case["groups"]
        fixed = stats["fixed_ensemble_position"]
        raw = stats["uncorrected_ensemble_position"]
        print(
            f"{case['key']}: fixed mean={fixed['final_mean'] * 100:.3f} cm "
            f"std={fixed['final_std'] * 100:.3f} cm | uncorrected mean={raw['final_mean'] * 100:.3f} cm "
            f"std={raw['final_std'] * 100:.3f} cm"
        )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
