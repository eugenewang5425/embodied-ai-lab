"""Lesson 21: estimate -> wheel command -> actual motion -> fresh measurement.

New closed-loop kinematic trials, not replayed lesson-20 ROS traffic. No ROS,
physics engine, obstacle avoidance, slip model, covariance or new dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import SENSOR_IN_BODY, finite_vector, integrate_pose
from embodied_learning.experiments.mobile_noise import (
    INPUT_RIGHT_SCALE,
    INTERVAL_NOISE_STD_RAD,
    calibrated_factor,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, GEOMETRY, GoalConfig, goal_command
from embodied_learning.landmark_localization import (
    LANDMARKS,
    OBS_BEARING_STD_RAD,
    OBS_PERIOD_STEPS,
    OBS_RANGE_STD_M,
    observe,
    solve_pose,
)
from embodied_learning.odometry import estimate_poses, heading_error

DT = 0.04
MAX_STEPS = 1000
DEFAULT_RESULTS = "results/goal_reaching_2026-09-03"
CASES = (("near", "近目标", (1.6, 0.8)), ("far", "远目标", (4.8, 1.2)))
METHODS = (("odom", "只靠轮子定位", "#9333ea"), ("fused", "轮子＋地标定位", "#ea580c"))
MODE_CODES = {"driving": 0, "turning": 1, "settling": 2, "arrived": 3, "timeout": 4}
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def simulate(
    goal,
    method,
    seed,
    *,
    config=DEFAULT_CONFIG,
    noise_scale=1.0,
    max_steps=MAX_STEPS,
    initial_pose=(0, 0, 0),
    correction=None,
):
    """Paired standardized noise at common timestamps, NOT identical trajectories.

    Frames are post-measurement. commands[k] acts over k -> k+1; the last
    command is a terminal zero, not a fictitious extra movement interval.
    """
    goal, initial = finite_vector(goal, 2), finite_vector(initial_pose, 3)
    if method not in {m[0] for m in METHODS} or type(seed) is not int or seed < 0:
        raise ValueError("Unknown method or invalid seed")
    if (
        type(max_steps) is not int
        or max_steps < 1
        or not np.isfinite(noise_scale)
        or noise_scale < 0
    ):
        raise ValueError("Invalid duration/noise")
    if correction is None:
        correction = calibrated_factor()[0]
    if not np.isfinite(correction) or correction <= 0:
        raise ValueError("Correction must be positive")
    enc_rng, obs_rng = (
        np.random.default_rng(np.random.SeedSequence([seed, stream])) for stream in (0, 1)
    )
    epsilon = enc_rng.normal(0, INTERVAL_NOISE_STD_RAD * noise_scale, max_steps)
    true, estimates, priors = [initial.copy()], [initial.copy()], [initial.copy()]
    encoders, measured = [np.zeros(2)], []
    commands, modes, frames, readings = [], [], [], []
    held = 0
    required_hold = int(np.ceil(config.settle_seconds / DT))
    reason = "timeout"
    for frame in range(max_steps + 1):
        decision = goal_command(estimates[-1], goal, config)
        if decision["mode"] == "settling" and frame and modes[-1] == MODE_CODES["settling"]:
            held += 1
        else:
            held = 0
        if held >= required_hold:
            reason = "arrived"
        if reason == "arrived" or frame == max_steps:
            commands.append(np.zeros(2))
            modes.append(MODE_CODES[reason])
            break
        command = decision["wheels"]
        commands.append(command.copy())
        modes.append(MODE_CODES[decision["mode"]])
        # Plant boundary: ONLY commanded wheels determine actual movement.
        true.append(integrate_pose(true[-1], GEOMETRY.body_velocity(command), DT))
        increments = command * DT
        increments[1] = correction * (INPUT_RIGHT_SCALE * increments[1] + epsilon[frame])
        measured.append(increments.copy())
        encoders.append(encoders[-1] + increments)
        predicted = estimate_poses(np.array(encoders[-2:]), estimates[-1])[-1]
        priors.append(predicted.copy())
        if (frame + 1) % OBS_PERIOD_STEPS == 0:
            # Truth may generate a sensor reading, but is never sent to the policy.
            observation = observe(
                true[-1],
                LANDMARKS,
                obs_rng,
                OBS_RANGE_STD_M * noise_scale,
                OBS_BEARING_STD_RAD * noise_scale,
            )
            frames.append(frame + 1)
            readings.append(observation)
            if method == "fused":
                absolute = solve_pose(observation, LANDMARKS)
                predicted[:2] = absolute[:2]
                predicted[2] += heading_error(absolute[2], predicted[2])
        estimates.append(predicted)
    arrays = {
        "truth": np.array(true),
        "estimated": np.array(estimates),
        "prior": np.array(priors),
        "encoders": np.array(encoders),
        "measured_increments": np.array(measured),
        "commands": np.array(commands),
        "modes": np.array(modes, dtype=np.int64),
        "observation_frames": np.array(frames, dtype=np.int64),
        "observations": np.array(readings).reshape(-1, len(LANDMARKS), 2),
        "encoder_noise": epsilon[: len(true) - 1].copy(),
    }
    metrics = evaluate(arrays, goal, config)
    metrics.update(method=method, seed=seed, goal=goal.tolist())
    return arrays, metrics


def evaluate(arrays, goal, config=DEFAULT_CONFIG):
    goal = finite_vector(goal, 2)
    true_distance = np.linalg.norm(arrays["truth"][:, :2] - goal, axis=1)
    estimated_distance = np.linalg.norm(arrays["estimated"][:, :2] - goal, axis=1)
    arrived = int(arrays["modes"][-1]) == MODE_CODES["arrived"]
    stopped = bool(np.all(arrays["commands"][-1] == 0))
    last = int(np.ceil(config.settle_seconds / DT))
    success = bool(
        arrived
        and stopped
        and np.all(true_distance[-last - 1 :] <= config.true_acceptance_radius_m)
    )
    return {
        "steps": len(true_distance) - 1,
        "duration_s": (len(true_distance) - 1) * DT,
        "controller_arrived": arrived,
        "true_success": success,
        "false_arrival": bool(arrived and not success),
        "true_final_distance_m": float(true_distance[-1]),
        "estimated_final_distance_m": float(estimated_distance[-1]),
        "path_length_m": float(
            np.linalg.norm(np.diff(arrays["truth"][:, :2], axis=0), axis=1).sum()
        ),
        "max_wheel_rad_s": float(np.max(np.abs(arrays["commands"]))),
        "terminal_reason": "arrived" if arrived else "timeout",
    }


def run_experiment(output, *, runs=20, seed=0, config=DEFAULT_CONFIG):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(runs) is not int or runs < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Need integer runs >= 2 and seed >= 0")
    correction, segments = calibrated_factor()
    archive, trials, comparisons = {}, [], []
    for case_index, (key, label, goal) in enumerate(CASES):
        group = []
        for run in range(runs):
            trial_seed = seed + case_index * 1_000_000 + run
            for method, _, _ in METHODS:
                arrays, metrics = simulate(
                    goal, method, trial_seed, correction=correction, config=config
                )
                prefix = f"{key}_{run:02d}_{method}"
                archive.update({f"{prefix}_{name}": value for name, value in arrays.items()})
                trial = {**metrics, "prefix": prefix, "case": key, "run": run}
                trials.append(trial)
                group.append(trial)
        stats = {}
        for method, _, _ in METHODS:
            values = [t for t in group if t["method"] == method]
            stats[method] = {
                "true_success_count": sum(t["true_success"] for t in values),
                "false_arrival_count": sum(t["false_arrival"] for t in values),
                "timeout_count": sum(not t["controller_arrived"] for t in values),
                "mean_true_final_distance_m": float(
                    np.mean([t["true_final_distance_m"] for t in values])
                ),
                "max_true_final_distance_m": max(t["true_final_distance_m"] for t in values),
            }
        comparisons.append({"case": key, "label": label, "goal": list(goal), "methods": stats})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    sources = [
        "goal_control.py",
        "experiments/goal_reaching.py",
        "differential_drive.py",
        "odometry.py",
        "landmark_localization.py",
        "experiments/mobile_noise.py",
        "experiments/encoder_calibration.py",
        "encoder_calibration.py",
        "experiments/mobile_frames.py",
        "experiments/mobile_odometry.py",
    ]
    report = {
        "experiment": "estimated_pose_goal_feedback",
        "schema_version": 1,
        "model": "ideal_no_slip_velocity_kinematics",
        "runtime": "local_python_closed_loop_not_ROS",
        "runs": runs,
        "seed": seed,
        "dt_s": DT,
        "max_steps": MAX_STEPS,
        "controller": asdict(config),
        "geometry": asdict(GEOMETRY),
        "sensor_in_body": SENSOR_IN_BODY.tolist(),
        "landmarks": LANDMARKS.tolist(),
        "observation_period_steps": OBS_PERIOD_STEPS,
        "encoder_scale": INPUT_RIGHT_SCALE,
        "encoder_noise_std_rad": INTERVAL_NOISE_STD_RAD,
        "range_noise_std_m": OBS_RANGE_STD_M,
        "bearing_noise_std_rad": OBS_BEARING_STD_RAD,
        "calibration_factor": correction,
        "calibration_segments": segments,
        "comparisons": comparisons,
        "trials": trials,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {name: digest(SOURCE_ROOT / name) for name in sources},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "limits": [
            "No dynamics, inertia, slip, obstacles or collision checking; wheel speed acts immediately",
            "Known initial pose, map, identities and mount; synthetic unlimited-visibility landmarks",
            "Identical standardized noise by paired seed/time, NOT identical sensor readings or true trajectories",
            "Terminal decision uses estimate only; truth is for sensor generation and independent evaluation",
            "A terminal stop ends the trial: no later observation or continued recovery is simulated",
            "No goal orientation requirement; radius applies to axle centre, not whole body footprint",
            "Noise model retained at rest as well as in motion; not a physical encoder specification",
            "This lesson is Python feedback simulation, not new live ROS communication evidence",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def load_recording(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != "estimated_pose_goal_feedback"
        or report.get("schema_version") != 1
        or report.get("dt_s") != DT
    ):
        raise ValueError("Not a lesson-21 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report["trajectories_sha256"]:
        raise ValueError("Trajectory checksum mismatch")
    config = GoalConfig(**report["controller"])
    if type(report.get("runs")) is not int or report["runs"] < 2:
        raise ValueError("Invalid run count")
    expected_cases = {key: list(goal) for key, _, goal in CASES}
    expected_trials = {
        (key, run, method)
        for key in expected_cases
        for run in range(report["runs"])
        for method, _, _ in METHODS
    }
    results = {}
    with np.load(path, allow_pickle=False) as archive:
        for trial in report["trials"]:
            prefix, n = trial["prefix"], trial["steps"]
            identity = (trial["case"], trial["run"], trial["method"])
            if (
                identity not in expected_trials
                or identity in results
                or type(n) is not int
                or not 1 <= n <= MAX_STEPS
            ):
                raise ValueError("Invalid or duplicate trial")
            if (
                trial["goal"] != expected_cases[trial["case"]]
                or prefix != f"{trial['case']}_{trial['run']:02d}_{trial['method']}"
            ):
                raise ValueError("Invalid trial goal or prefix")
            widths = {
                "truth": (n + 1, 3),
                "estimated": (n + 1, 3),
                "prior": (n + 1, 3),
                "encoders": (n + 1, 2),
                "commands": (n + 1, 2),
                "measured_increments": (n, 2),
                "modes": (n + 1,),
                "encoder_noise": (n,),
            }
            arrays = {
                name: archive[f"{prefix}_{name}"].copy()
                for name in (*widths, "observation_frames", "observations")
            }
            if any(
                arrays[k].shape != shape or not np.isfinite(arrays[k]).all()
                for k, shape in widths.items()
            ):
                raise ValueError("Invalid trajectory shape or values")
            expected_frames = np.arange(OBS_PERIOD_STEPS, n + 1, OBS_PERIOD_STEPS)
            if (
                not np.array_equal(arrays["observation_frames"], expected_frames)
                or arrays["observations"].shape != (len(expected_frames), 3, 2)
                or not np.isfinite(arrays["observations"]).all()
            ):
                raise ValueError("Invalid observation timing")
            if np.any(np.abs(arrays["commands"]) > config.max_wheel_rad_s + 1e-12):
                raise ValueError("Wheel limit violated")
            if not np.isin(arrays["modes"][:-1], [0, 1, 2]).all() or arrays["modes"][-1] not in [
                3,
                4,
            ]:
                raise ValueError("Invalid control mode sequence")
            for key, value in evaluate(arrays, trial["goal"], config).items():
                if isinstance(value, float):
                    if not np.isclose(value, trial[key], atol=1e-12, rtol=1e-12):
                        raise ValueError("Metrics do not match trajectory")
                elif value != trial[key]:
                    raise ValueError("Outcome does not match trajectory")
            results[identity] = (arrays, trial)
    if set(results) != expected_trials:
        raise ValueError("Missing trial")
    if [c["case"] for c in report["comparisons"]] != [c[0] for c in CASES]:
        raise ValueError("Invalid comparison cases")
    for case in report["comparisons"]:
        if case["goal"] != expected_cases[case["case"]]:
            raise ValueError("Comparison goal mismatch")
        for method, _, _ in METHODS:
            values = [results[(case["case"], run, method)][1] for run in range(report["runs"])]
            stats = case["methods"][method]
            checks = {
                "true_success_count": sum(t["true_success"] for t in values),
                "false_arrival_count": sum(t["false_arrival"] for t in values),
                "timeout_count": sum(not t["controller_arrived"] for t in values),
                "mean_true_final_distance_m": float(
                    np.mean([t["true_final_distance_m"] for t in values])
                ),
                "max_true_final_distance_m": max(t["true_final_distance_m"] for t in values),
            }
            if any(not np.isclose(stats[k], v, rtol=1e-12, atol=1e-12) for k, v in checks.items()):
                raise ValueError("Comparison metrics mismatch")
    return report, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, runs=args.runs, seed=args.seed)
    print(json.dumps(report["comparisons"], ensure_ascii=False, indent=2))
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
