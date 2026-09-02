"""Assess single-reference scale calibration on independent distances.

Reads per-distance calibration reports written by "depth-distance-calibrate"
(each contains the board-pose ground truth, predicted depths and the fitted
scale). Truth comes from the board pose, not from the approximate ruler values:
the ruler only appears in the provenance cross-check.

Fits:
  1. scale-only  z_true = s * z_pred, using only the training reference(s);
  2. scale+offset z_true = a * z_pred + b over all points (leave-one-out);
  3. baseline: raw model (s=1) errors, to show what the fit bought.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from monocular_depth.records import read_json, write_json


def point_from_report(report: dict) -> dict:
    """Reduce a distance calibration report to its fitted (pred, true) point.

    Uses the board-interior medians (robust, ~180k pixels) rather than the
    single center pixel; the center values are kept for reference.
    """
    interior = report["interior_median"]
    center = report["center_depth"]
    ruler = (report.get("ground_truth") or {}).get("ruler") or {}
    return {
        "predicted_m": float(interior["predicted_m"]),
        "true_m": float(interior["true_m"]),
        "scale": float(report["scale"]),
        "scale_mad": float(report["scale_mad"]),
        "center_predicted_m": float(center["predicted_m"]),
        "center_true_m": float(center["true_m"]),
        "ruler_measured_m": ruler.get("measured_distance_m"),
        "ruler_note": "approximate; ground truth is board-pose geometry, not the ruler",
        "interior_pixels": int(report["interior_pixels"]),
    }


def fit_scale_only(points: list[dict]) -> float:
    return float(np.median([p["scale"] for p in points]))


def fit_affine(points: list[dict]) -> tuple[float, float]:
    pred = np.array([p["predicted_m"] for p in points])
    true = np.array([p["true_m"] for p in points])
    a, b = np.polyfit(pred, true, 1)
    return float(a), float(b)


def relative_error(predicted: float, true: float) -> float:
    return (predicted - true) / true


def assess(fit_paths: list[Path], heldout_paths: list[Path]):
    fit_points = [point_from_report(read_json(p)) for p in fit_paths]
    held_points = [point_from_report(read_json(p)) for p in heldout_paths]
    all_points = fit_points + held_points
    s = fit_scale_only(fit_points)
    a, b = fit_affine(all_points)

    # Explicit evaluations; avoid clever one-liners for auditability.
    scale_only = []
    affine = []
    raw = []
    for p in held_points:
        scale_only.append(relative_error(s * p["predicted_m"], p["true_m"]))
        affine.append(relative_error(a * p["predicted_m"] + b, p["true_m"]))
        raw.append(relative_error(p["predicted_m"], p["true_m"]))
    loo = []
    for index, left in enumerate(all_points):
        train = [p for i, p in enumerate(all_points) if i != index]
        a_, b_ = fit_affine(train)
        loo.append(
            {
                "left_out": {
                    "predicted_m": left["predicted_m"],
                    "true_m": left["true_m"],
                },
                "fit_a": a_,
                "fit_b": b_,
                "relative_error": relative_error(a_ * left["predicted_m"] + b_, left["true_m"]),
            }
        )
    summary = {
        "schema_version": 1,
        "fit_sources": [str(p.resolve()) for p in fit_paths],
        "heldout_sources": [str(p.resolve()) for p in heldout_paths],
        "points": all_points,
        "fit": {
            "scale_only_s": s,
            "affine_a": a,
            "affine_b": b,
            "note": (
                "scale-only fitted on the training reference(s) only; affine fitted on "
                "all points with leave-one-out below. Truth is board-pose geometry; "
                "ruler values are approximate cross-checks, not used."
            ),
        },
        "heldout_scale_only_relative_error": scale_only,
        "heldout_affine_relative_error": affine,
        "heldout_raw_model_relative_error": raw,
        "heldout_scale_only_rms": float(np.sqrt(np.mean(np.square(scale_only)))),
        "heldout_affine_rms": float(np.sqrt(np.mean(np.square(affine)))),
        "heldout_raw_model_rms": float(np.sqrt(np.mean(np.square(raw)))),
        "leave_one_out_affine": loo,
        "validation_note": (
            "Held-out frames are independent captures, but the charuco ground truth and "
            "the depth model are the same system; this checks distance-domain consistency, "
            "not end-to-end world accuracy. Not for robot safety."
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single-reference scale or affine fit on independent distances"
    )
    parser.add_argument("--fit", type=Path, nargs="+", required=True)
    parser.add_argument("--heldout", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = assess(args.fit, args.heldout)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, summary)
        print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
