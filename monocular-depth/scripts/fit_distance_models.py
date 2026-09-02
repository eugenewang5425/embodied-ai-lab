"""Fit and compare distance calibration models over all collected distances.

Reads per-distance calibration reports (ground truth = board-pose geometry)
and aggregates the frames at each user-stated approximate distance. Compared
models: identity (raw), scale-only, affine, inverse-affine, and the curved
z-model  z_true = a*z_pred + b + c / z_pred.

Validation: leave-one-distance-out plus explicit fit/heldout splits so the
curved model is tested by cross-validation rather than by its own fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from monocular_depth.records import read_json, write_json

MODELS = ("identity", "scale", "affine", "inv_affine", "z_model")


def collect_reports(root: Path) -> list[Path]:
    return sorted(root.glob("*/distance-calibration.json"))


def point(report: dict) -> dict:
    interior = report["interior_median"]
    ruler = (report.get("ground_truth") or {}).get("ruler") or {}
    distance = ruler.get("measured_distance_m")
    return {
        "predicted_m": float(interior["predicted_m"]),
        "true_m": float(interior["true_m"]),
        "distance_m": float(distance) if distance is not None else None,
        "pixels": int(report["interior_pixels"]),
    }


def group_by_distance(points: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = {}
    for p in points:
        if p["distance_m"] is None:
            raise ValueError("A report lacks the approximate measured distance")
        groups.setdefault(round(p["distance_m"], 2), []).append(p)
    return groups


def fit_model(name: str, pred: np.ndarray, true: np.ndarray):
    if name == "identity":
        return None
    if name == "scale":
        return float(np.median(true / pred))
    if name == "affine":
        return tuple(float(v) for v in np.polyfit(pred, true, 1))
    if name == "inv_affine":
        return tuple(float(v) for v in np.polyfit(1.0 / pred, 1.0 / true, 1))
    if name == "z_model":
        design = np.column_stack((pred, np.ones_like(pred), 1.0 / pred))
        coef, *_ = np.linalg.lstsq(design, true, rcond=None)
        return tuple(float(v) for v in coef)
    raise ValueError(name)


def predict(name: str, params, pred: np.ndarray) -> np.ndarray:
    if name == "identity":
        return pred
    if name == "scale":
        return params * pred
    if name == "affine":
        a, b = params
        return a * pred + b
    if name == "inv_affine":
        p, q = params
        return 1.0 / (p / pred + q)
    if name == "z_model":
        a, b, c = params
        return a * pred + b + c / pred
    raise ValueError(name)


def evaluate(name: str, params, points: list[dict]) -> dict:
    pred = np.array([p["predicted_m"] for p in points])
    true = np.array([p["true_m"] for p in points])
    errors = (predict(name, params, pred) - true) / true
    return {
        "relative_errors": [float(e) for e in errors],
        "rms": float(np.sqrt(np.mean(errors**2))),
        "median_abs": float(np.median(np.abs(errors))),
        "max_abs": float(np.max(np.abs(errors))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare distance calibration models")
    parser.add_argument("--root", type=Path, default=Path("calibration/distance"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = collect_reports(args.root)
    if len(reports) < 5:
        parser.error("Not enough reports; collect the 0.7/1.2 m frames first")
    points = [point(read_json(p)) for p in reports]
    groups = group_by_distance(points)
    distances = sorted(groups)
    median_points = {
        d: {
            "predicted_m": float(np.median([p["predicted_m"] for p in ps])),
            "true_m": float(np.median([p["true_m"] for p in ps])),
            "frames": len(ps),
        }
        for d, ps in groups.items()
    }

    summary = {
        "schema_version": 1,
        "reports": [str(p.resolve()) for p in reports],
        "distances": distances,
        "median_points": median_points,
        "models": {},
        "full_fit": {},
        "leave_one_distance_out": {},
        "holdout_070_120_from_040_060_090": {},
        "notes": (
            "Per-distance medians of board-interior (predicted, true); truth is board-pose "
            "geometry. distance_m is the user-stated approximate median, used for grouping only."
        ),
    }

    for name in MODELS:
        params = fit_model(name, np.array([median_points[d]["predicted_m"] for d in distances]),
                           np.array([median_points[d]["true_m"] for d in distances]))
        summary["full_fit"][name] = {
            "params": params,
            "in_sample": evaluate(name, params, points),
        }

        loo = {}
        for left in distances:
            fit_d = [d for d in distances if d != left]
            pred = np.array([np.median([q["predicted_m"] for q in groups[dd]]) for dd in fit_d])
            true = np.array([np.median([q["true_m"] for q in groups[dd]]) for dd in fit_d])
            params_fold = fit_model(name, pred, true)
            loo[left] = evaluate(name, params_fold, groups[left])
        summary["leave_one_distance_out"][name] = loo

        required = (0.4, 0.6, 0.9)
        if all(d in groups for d in required) and all(d in groups for d in (0.7, 1.2)):
            pred = np.array([np.median([q["predicted_m"] for q in groups[dd]]) for dd in required])
            true = np.array([np.median([q["true_m"] for q in groups[dd]]) for dd in required])
            params_fold = fit_model(name, pred, true)
            held = [p for d in (0.7, 1.2) for p in groups[d]]
            summary["holdout_070_120_from_040_060_090"][name] = evaluate(name, params_fold, held)
        else:
            summary["holdout_070_120_from_040_060_090"][name] = None

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, summary)
        print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
