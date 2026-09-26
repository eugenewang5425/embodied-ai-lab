"""Unified scoring for the dual-localizer comparison.

Reads frozen input + self-PF output + official AMCL output,
computes aligned errors on a common timeline, and generates the report.

Usage:
    uv run python src/embodied_learning/experiments/nav_scoring.py \
        --input results/amcl_bridge_v2 \
        --self-pf results/amcl_bridge_comparison \
        --official-amcl results/amcl_official_v1 \
        --output results/nav_scoring_v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_aligned(estimates, truth, timestamps_est, timestamps_truth):
    """Score by matching estimate timestamps to truth timestamps.

    Returns metrics dict and the matched index pairs.
    """
    matched_est = []
    matched_truth = []
    pairs = []
    for i, te in enumerate(timestamps_est):
        idx = int(np.argmin(np.abs(timestamps_truth - te)))
        if idx < len(truth):
            matched_est.append(estimates[i])
            matched_truth.append(truth[idx])
            pairs.append((i, idx))
    if not matched_est:
        return {"n_matched": 0}, []
    me = np.asarray(matched_est)
    mt = np.asarray(matched_truth)
    errors = np.linalg.norm(me[:, :2] - mt[:, :2], axis=1)
    yaw_err = [
        abs(math.atan2(math.sin(me[i, 2] - mt[i, 2]), math.cos(me[i, 2] - mt[i, 2])))
        for i in range(min(len(me), len(mt)))
    ]
    return {
        "n_matched": len(errors),
        "mean_m": float(errors.mean()),
        "p95_m": float(np.percentile(errors, 95)),
        "end_m": float(errors[-1]),
        "max_m": float(errors.max()),
        "mean_yaw_rad": float(np.mean(yaw_err)),
        "coverage": len(errors) / max(1, len(timestamps_truth)),
    }, pairs


def continuous_pose(estimates, odom, timestamps_est, timestamps_odom):
    """Causal continuous pose: last known correction + subsequent odom motion.

    For each odom timestamp, find the most recent estimate that was available
    BEFORE that timestamp, then propagate using the odom displacement.
    """
    if len(estimates) == 0:
        return np.zeros((len(odom), 3))
    result = np.zeros((len(odom), 3))
    est_idx = 0  # next estimate to use
    last_correction = estimates[0].copy() if len(estimates) > 0 else np.zeros(3)
    last_correction_time = timestamps_est[0] if len(timestamps_est) > 0 else 0.0

    for k in range(len(odom)):
        t = timestamps_odom[k]
        # check if a new estimate is available at or before this time
        while est_idx < len(timestamps_est) and timestamps_est[est_idx] <= t:
            last_correction = estimates[est_idx].copy()
            last_correction_time = timestamps_est[est_idx]
            est_idx += 1
        # odom displacement from last correction time to now
        # find the odom index closest to last_correction_time
        odom_idx = int(np.argmin(np.abs(timestamps_odom - last_correction_time)))
        if k > 0 and odom_idx < k:
            # propagate from last correction using odom displacement
            dx = odom[k][0] - odom[odom_idx][0]
            dy = odom[k][1] - odom[odom_idx][1]
            dth = odom[k][2] - odom[odom_idx][2]
            result[k] = last_correction + np.array([dx, dy, dth])
        else:
            result[k] = last_correction
    return result


def main():
    parser = argparse.ArgumentParser(description="unified navigation scoring")
    parser.add_argument("--input", required=True, help="frozen input directory")
    parser.add_argument("--self-pf", required=True, help="self-PF results (npz)")
    parser.add_argument("--official-amcl", default=None, help="official AMCL results (npz)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load frozen input
    data = dict(np.load(input_dir / "frozen_input.npz", allow_pickle=False))
    truth = data["truth"]
    odom = data["odom"]
    timestamps = data["timestamps"]

    results = {}

    # odometry baseline
    odom_err = np.linalg.norm(odom[:, :2] - truth[:, :2], axis=1)
    results["odometry"] = {
        "mean_m": float(odom_err.mean()),
        "p95_m": float(np.percentile(odom_err, 95)),
        "end_m": float(odom_err[-1]),
        "coverage": 1.0,
        "n_frames": len(odom),
    }

    # self-PF
    if args.self_pf:
        pf_path = Path(args.self_pf) / "trajectories.npz"
        if pf_path.exists():
            with np.load(pf_path, allow_pickle=False) as z:
                pf_est = z.get("self_pf", z.get("estimate"))
            if pf_est is not None:
                n = min(len(pf_est), len(truth))
                pf_err = np.linalg.norm(pf_est[:n, :2] - truth[:n, :2], axis=1)
                results["self_pf"] = {
                    "mean_m": float(pf_err.mean()),
                    "p95_m": float(np.percentile(pf_err, 95)),
                    "end_m": float(pf_err[-1]),
                    "coverage": n / len(truth),
                    "n_frames": n,
                }

    # official AMCL
    if args.official_amcl:
        amcl_path = Path(args.official_amcl) / "amcl_poses.npz"
        if amcl_path.exists():
            with np.load(amcl_path, allow_pickle=False) as z:
                amcl_x = z["x"]
                amcl_y = z["y"]
                amcl_yaw = z["yaw"]
                amcl_stamp = z["stamp"]
            if len(amcl_x) > 0:
                # aligned scoring: match AMCL timestamps to truth timestamps
                amcl_est = np.column_stack([amcl_x, amcl_y, amcl_yaw])
                amcl_aligned, _pairs = score_aligned(amcl_est, truth, amcl_stamp, timestamps)
                results["official_amcl_aligned"] = amcl_aligned

                # continuous scoring: last correction + odom propagation
                amcl_cont = continuous_pose(amcl_est, odom, amcl_stamp, timestamps)
                n = min(len(amcl_cont), len(truth))
                cont_err = np.linalg.norm(amcl_cont[:n, :2] - truth[:n, :2], axis=1)
                results["official_amcl_continuous"] = {
                    "mean_m": float(cont_err.mean()),
                    "p95_m": float(np.percentile(cont_err, 95)),
                    "end_m": float(cont_err[-1]),
                    "coverage": n / len(truth),
                    "n_frames": n,
                }

    # save
    (output_dir / "scoring_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
