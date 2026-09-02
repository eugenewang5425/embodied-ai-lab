"""Lesson 16: separate calibration runs, frozen lesson-15 holdouts, no truth fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.encoder_calibration import correct_right_encoder, fit_right_correction
from embodied_learning.experiments.mobile_frames import GEOMETRY
from embodied_learning.experiments.mobile_odometry import (
    SHARED_WIDTHS,
    evaluate_readings,
    make_plot,
    simulate_truth,
)
from embodied_learning.odometry import scaled_encoder_readings
from embodied_learning.odometry_demo import load_replays

EXPERIMENT = "differential_drive_encoder_calibration"
METHODS = (
    ("raw", "未标定", "#9333ea"),
    ("calibrated", "独立标定", "#0f766e"),
    ("reference_bias", "测距偏大1%", "#ea580c"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_measurements(input_scale):
    """Synthetic external metrology of NEW straight runs, including reverse.

    Simulator truth generates the independent distance instrument. That source
    is intentionally exact in the main case, not a real tape/camera experiment.
    Evaluation routes are not used here; their final poses cannot tune the fit.
    """
    rows = []
    for key, steps, speed in (
        ("forward_040", 50, 4.0),
        ("forward_080", 100, 4.0),
        ("forward_120", 150, 4.0),
        ("reverse_060", 75, -4.0),
    ):
        shared = simulate_truth(np.full((steps, 2), speed))
        reading = scaled_encoder_readings(shared["wheel_angles_rad"], input_scale)
        rows.append(
            {
                "key": key,
                "steps": steps,
                "measured_right_angle_rad": float(reading[-1, 1] - reading[0, 1]),
                "external_distance_m": float(shared["true_poses"][-1, 0]),
            }
        )
    return rows


def fit_methods(rows):
    angles = [row["measured_right_angle_rad"] for row in rows]
    distances = np.array([row["external_distance_m"] for row in rows])
    return (
        1.0,
        fit_right_correction(angles, distances, GEOMETRY.radius_m),
        fit_right_correction(angles, distances * 1.01, GEOMETRY.radius_m),
    )


def run_experiment(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    replays = load_replays(source)  # Validate frozen lesson-15 schema and archive hash.
    source_report = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    input_scale = source_report["cases"][0]["estimates"][2]["right_scale"]
    rows = calibration_measurements(input_scale)
    factors = fit_methods(rows)  # Receives measurements ONLY, never input_scale.
    variants = tuple(
        (key, label, input_scale * factor, color)
        for (key, label, color), factor in zip(METHODS, factors)
    )
    by_key, archive, cases = {}, {}, []
    for replay in replays:
        shared = {name: replay.arrays[name] for name in SHARED_WIDTHS}
        raw = replay.arrays["right_2pct_encoder_angles_rad"]
        arrays, estimates = dict(shared), []
        for (key, label, scale, color), factor in zip(variants, factors):
            corrected = correct_right_encoder(raw, factor)
            values, metrics = evaluate_readings(shared, corrected)
            if key == "raw":
                for name, value in values.items():
                    if not np.array_equal(value, replay.arrays[f"right_2pct_{name}"]):
                        raise ValueError("Lesson-15 raw baseline changed")
            arrays.update({f"{key}_{name}": value for name, value in values.items()})
            estimates.append(
                {
                    "key": key,
                    "label": label,
                    "right_scale": scale,  # Effective scale, diagnostic only.
                    "color": color,
                    "correction_factor": factor,
                    **metrics,
                }
            )
        case = {**replay.metadata, "estimates": estimates}
        by_key[case["key"]] = arrays
        archive.update({f"{case['key']}_{name}": value for name, value in arrays.items()})
        cases.append(case)
    output.mkdir(parents=True, exist_ok=False)
    calibration = {
        "schema_version": 1,
        "instrument": "synthetic independent exact signed distance on verified straight runs",
        "runs": rows,
        "reference_bias_variant_multiplier": 1.01,
        "fitting": "origin least squares: distance = correction * radius * measured angle",
        "fitted_corrections": dict(zip((key for key, _, _ in METHODS), factors)),
        "separation": "new 0.4/0.8/1.2/-0.6 m runs; never fit to straight12s or square24s",
    }
    (output / "calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output / "trajectories.npz", **archive)
    base = Path(__file__).resolve().parents[1]
    sources = [
        "differential_drive.py",
        "odometry.py",
        "encoder_calibration.py",
        "experiments/mobile_frames.py",
        "experiments/mobile_odometry.py",
        "experiments/encoder_calibration.py",
        "odometry_demo.py",
        "calibration_demo.py",
    ]
    report = {
        **source_report,
        "experiment": EXPERIMENT,
        "input_right_scale": input_scale,
        "cases": cases,
        "calibration_sha256": digest(output / "calibration.json"),
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {name: digest(base / name) for name in sources},
        "holdout_source": {
            "directory": str(source.resolve()),
            "summary_sha256": digest(source / "summary.json"),
            "trajectories_sha256": digest(source / "trajectories.npz"),
            "variant": "right_2pct",
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "Deterministic synthetic calibration, not hardware metrology or statistical accuracy",
            "Fixed right encoder scale; exact left wheel, geometry and initial pose; no slip",
            "Reference +1% variant changes calibration distance measurements only",
            "No online correction, landmark localization, feedback control, noise or SLAM",
            "Radius error and encoder scale cannot be separately identified by this fit",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(by_key, output, variants)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/mobile_odometry_2026-09-03"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_experiment(args.source, args.output)
    for case in report["cases"]:
        for estimate in case["estimates"]:
            print(
                f"{case['key']}/{estimate['key']}: c={estimate['correction_factor']:.9f}, "
                f"final={estimate['final_position_error_m']:.9g} m, "
                f"max={estimate['max_position_error_m']:.9g} m"
            )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
