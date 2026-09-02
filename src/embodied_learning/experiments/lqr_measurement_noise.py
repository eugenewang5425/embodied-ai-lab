"""One-factor sensor-noise experiment; reuse the physical runner and fixed R=1 LQR."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy

from embodied_learning.controllers.lqr import LQRController, design_lqr
from embodied_learning.environments import make_inverted_pendulum_environment
from embodied_learning.experiments.lqr_comparison import SETTLED_TOLERANCES
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode
from embodied_learning.plotting import configure_plot_font

# Synthetic per-sample measurement standard deviations, NOT hardware specifications.
# Order: x [m], theta [rad], velocity [m/s], angular velocity [rad/s].
BASE_SENSOR_STD = np.array([0.002, 0.002, 0.01, 0.01])
NOISE_SCALES = (0.0, 1.0, 3.0)


@dataclass
class MeasuredFeedback:
    """Corrupt only the controller input, never the physical state or termination check."""

    controller: LQRController
    noise: np.ndarray
    measurements: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.noise = np.array(self.noise, dtype=float, copy=True)
        if (
            self.noise.ndim != 2
            or self.noise.shape[1] != 4
            or len(self.noise) < 1
            or not np.isfinite(self.noise).all()
        ):
            raise ValueError("noise must have finite shape (steps, 4)")

    def action(self, observation):
        state = np.asarray(observation, dtype=float)
        if state.shape != (4,) or not np.isfinite(state).all():
            raise ValueError("observation must contain four finite state values")
        index = len(self.measurements)
        if index >= len(self.noise):
            raise ValueError("measurement noise schedule exhausted")
        measured = state + self.noise[index]
        action = self.controller.action(measured)
        self.measurements.append(measured.copy())
        return action


def noise_metrics(
    trace: EpisodeTrace, measurements: np.ndarray, reference: np.ndarray, horizon: int
) -> dict:
    states = np.vstack([trace.initial_observation, *trace.observations])
    if measurements.shape != (trace.length, 4):
        raise ValueError("one pre-action measurement is required for every action")
    errors = states[1:] - reference
    commands = np.asarray(trace.actions)
    jitter = np.diff(commands)
    return {
        "seed": trace.seed,
        "steps": trace.length,
        "duration_s": trace.length * trace.dt,
        "terminated": trace.terminated,
        "truncated": trace.truncated,
        "survived_horizon": trace.length == horizon and not trace.terminated,
        "true_position_rms_cm": float(100 * np.sqrt(np.mean(errors[:, 0] ** 2))),
        "true_angle_rms_deg": float(np.rad2deg(np.sqrt(np.mean(errors[:, 1] ** 2)))),
        "control_rms": float(np.sqrt(np.mean(commands**2))),
        "control_step_change_rms": float(np.sqrt(np.mean(jitter**2))) if len(jitter) else 0.0,
        "peak_absolute_control": float(np.max(np.abs(commands))),
        "saturated_steps": int(np.sum(np.abs(commands) >= 3 - 1e-6)),
        "true_in_tolerance_fraction": float(
            np.mean(np.all(np.abs(errors) <= SETTLED_TOLERANCES, axis=1))
        ),
        # z[k] belongs to s[k], NOT the post-action s[k+1].
        "measurement_error_rms_by_state": np.sqrt(
            np.mean((measurements - states[:-1]) ** 2, axis=0)
        ).tolist(),
    }


def save_plot(output: Path, archive, report: dict):
    configure_plot_font()
    seed, dt = report["seeds"][0], report["dt_s"]
    reference = np.asarray(report["reference"])
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
    key = f"case1_seed{seed}"
    states, measured = archive[f"{key}_states"], archive[f"{key}_measurements"]
    # Focus on the first three seconds, without interpolating or shifting sensor samples.
    times = np.arange(len(measured)) * dt
    mask = times <= 3
    axes[0].plot(
        times[mask],
        np.rad2deg(measured[mask, 1] - reference[1]),
        ".--",
        color="#ea580c",
        alpha=0.7,
        markersize=3,
        linewidth=0.8,
        label="控制器读到的角度 z（1× 噪声）",
    )
    axes[0].plot(
        times[mask],
        np.rad2deg(states[:-1][mask, 1] - reference[1]),
        color="#2563eb",
        linewidth=1.5,
        label="同一时刻真实角度 s",
    )
    axes[0].set(title="传感读数不等于真实运动（放大前 3 秒观察）", ylabel="相对竖直的角度（°）")
    for i, (scale, color) in enumerate(
        zip(NOISE_SCALES, ("#64748b", "#2563eb", "#dc2626"), strict=True)
    ):
        key = f"case{i}_seed{seed}"
        states, commands = archive[f"{key}_states"], archive[f"{key}_controls"]
        ts = np.arange(len(states)) * dt
        mask_s = ts <= 3
        axes[1].plot(
            ts[mask_s],
            np.rad2deg(states[mask_s, 1] - reference[1]),
            color=color,
            label=f"噪声 {scale:g}×",
        )
        tu = np.arange(len(commands)) * dt
        mask_u = tu <= 3
        axes[2].step(
            tu[mask_u], commands[mask_u], where="post", color=color, label=f"噪声 {scale:g}×"
        )
        if archive[f"{key}_end_flags"][0] and ts[-1] <= 3:
            axes[1].plot(ts[-1], np.rad2deg(states[-1, 1] - reference[1]), "x", color=color)
    axes[1].set(title="只改变读数误差，真实运动也会改变", ylabel="真实倾角（°）")
    axes[2].set(title="同一个 R=1 控制器对不同读数作出响应", ylabel="输入 u（执行器力=100u N）")
    for axis in axes:
        axis.axhline(0, color="gray", linewidth=0.5)
        axis.set_xlabel("仿真时间（s）")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.suptitle(f"测量噪声而非外力｜R=1 固定｜配对 seed={seed}｜无滤波")
    figure.savefig(output, dpi=150)
    plt.close(figure)


def run_noise_experiment(
    output: Path, episodes: int = 20, first_seed: int = 200, horizon: int = 250
) -> dict:
    if episodes < 1 or first_seed < 0 or horizon < 2:
        raise ValueError("Need episodes >= 1, first_seed >= 0 and horizon >= 2")
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    design = design_lqr(control_weight=1)
    seeds = list(range(first_seed, first_seed + episodes))
    standard = {seed: np.random.default_rng(seed).standard_normal((horizon, 4)) for seed in seeds}
    archive, conditions = {}, []
    summary_metrics = (
        "true_position_rms_cm",
        "true_angle_rms_deg",
        "control_rms",
        "control_step_change_rms",
        "true_in_tolerance_fraction",
    )
    for i, scale in enumerate(NOISE_SCALES):
        rows = []
        for seed in seeds:
            noise = standard[seed] * BASE_SENSOR_STD * scale
            feedback = MeasuredFeedback(design.controller, noise)
            trace = run_episode(
                "lqr", seed, horizon, feedback, initial_state=design.controller.reference
            )
            measured = np.asarray(feedback.measurements)
            rows.append(noise_metrics(trace, measured, design.controller.reference, horizon))
            key = f"case{i}_seed{seed}"
            archive[f"{key}_states"] = np.vstack([trace.initial_observation, *trace.observations])
            archive[f"{key}_measurements"] = measured
            archive[f"{key}_noise"] = noise[: trace.length]
            archive[f"{key}_controls"] = np.asarray(trace.actions)
            archive[f"{key}_end_flags"] = np.array([trace.terminated, trace.truncated])
        completed = [row for row in rows if row["survived_horizon"]]
        conditions.append(
            {
                "noise_scale": scale,
                "sensor_std": (BASE_SENSOR_STD * scale).tolist(),
                "successful_episodes": len(completed),
                "completed_episode_means": {
                    key: float(np.mean([r[key] for r in completed])) if completed else None
                    for key in summary_metrics
                },
                "episodes": rows,
            }
        )
    for seed, noise in standard.items():
        archive[f"seed{seed}_standard_normal"] = noise
    env = make_inverted_pendulum_environment()
    try:
        model_hash = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
    finally:
        env.close()
    report = {
        "schema_version": 1,
        "experiment": "paired_measurement_noise",
        "dt_s": design.dt,
        "horizon_steps": horizon,
        "seeds": seeds,
        "state_order": ["x_m", "joint_theta_rad", "velocity_m_s", "angular_velocity_rad_s"],
        "reference": design.controller.reference.tolist(),
        "initial_state": design.controller.reference.tolist(),
        "R": 1.0,
        "Q": design.q.tolist(),
        "K": design.controller.gain.tolist(),
        "A": design.a.tolist(),
        "B": design.b.tolist(),
        "actuator_gear": design.actuator_gear,
        "control_limit": design.controller.control_limit,
        "base_sensor_std": BASE_SENSOR_STD.tolist(),
        "noise_protocol": "z[k]=s[k]+scale*std*epsilon[k]; independent standard Gaussian across samples and state channels; identical epsilon for all scales at each seed; injected before action, never into physical state",
        "archive_alignment": "states: s[0..N]; measurements/noise/controls: z/epsilon/u[0..N-1]; u[k] advances s[k] to s[k+1]",
        "metric_protocol": "true-state metrics use post-action s[1..N]; measurement error compares z[k] against s[k]; summaries are mean per-episode metrics among full-horizon survivors only; always report survivor count",
        "tolerance_by_state": SETTLED_TOLERANCES.tolist(),
        "model_xml_sha256": model_hash,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "conditions": conditions,
        "limitations": [
            "Synthetic noise levels, not calibrated hardware specifications",
            "All four states measured; independent velocity noise is an abstraction, not differentiated encoder data",
            "No external force, delay, drift, bias, dropout, filter or model mismatch",
            "Persistent noise: report RMS and tolerance occupancy, not deterministic settling time",
            "Control step change RMS is an input-variation proxy, not motor wear, work or energy",
        ],
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=200)
    args = parser.parse_args()
    try:
        report = run_noise_experiment(args.output, args.episodes, args.first_seed)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for row in report["conditions"]:
        means = row["completed_episode_means"]
        print(
            f"noise={row['noise_scale']:g}x: survived={row['successful_episodes']}/{args.episodes}; completed means={means}"
        )


if __name__ == "__main__":
    main()
