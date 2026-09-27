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


def score_aligned(estimates, truth, timestamps_est, timestamps_truth, max_gap_s=None):
    """Score by matching estimate timestamps to truth timestamps.

    max_gap_s caps the time distance allowed for a match: estimates outside
    the tolerance stay unmatched instead of silently binding to the nearest
    truth frame (a wall-clock leak used to match index 0 and poison every
    error).  Coverage counts UNIQUE truth frames so a burst of estimates
    inside one truth interval cannot inflate coverage past 1.
    """
    n_truth = len(timestamps_truth)
    empty = {
        "n_matched": 0,
        "n_unique_truth": 0,
        "n_unmatched_time": 0,
        "mean_m": None,
        "p95_m": None,
        "end_m": None,
        "max_m": None,
        "mean_yaw_rad": None,
        "coverage": 0.0,
    }
    matched_est = []
    matched_truth = []
    pairs = []
    n_late = 0
    for i, te in enumerate(timestamps_est):
        idx = int(np.argmin(np.abs(timestamps_truth - te)))
        if max_gap_s is not None and abs(timestamps_truth[idx] - te) > max_gap_s:
            n_late += 1
            continue
        if idx < n_truth:
            matched_est.append(estimates[i])
            matched_truth.append(truth[idx])
            pairs.append((i, idx))
    if not matched_est:
        out = dict(empty)
        out["n_unmatched_time"] = int(n_late)
        return out, []
    me = np.asarray(matched_est)
    mt = np.asarray(matched_truth)
    errors = np.linalg.norm(me[:, :2] - mt[:, :2], axis=1)
    yaw_err = [
        abs(math.atan2(math.sin(me[i, 2] - mt[i, 2]), math.cos(me[i, 2] - mt[i, 2])))
        for i in range(min(len(me), len(mt)))
    ]
    return {
        "n_matched": len(errors),
        "n_unique_truth": len({idx for _i, idx in pairs}),
        "n_unmatched_time": int(n_late),
        "mean_m": float(errors.mean()),
        "p95_m": float(np.percentile(errors, 95)),
        "end_m": float(errors[-1]),
        "max_m": float(errors.max()),
        "mean_yaw_rad": float(np.mean(yaw_err)),
        "coverage": len({idx for _i, idx in pairs}) / max(1, n_truth),
    }, pairs


def continuous_pose(estimates, odom, timestamps_est, timestamps_odom):
    """Causal continuous pose: last known correction + subsequent odom motion.

    For each odom timestamp, find the most recent estimate that was available
    BEFORE that timestamp, then propagate using the odom displacement.

    Causality: frames timestamped before the FIRST correction have no usable
    correction - they are NaN (invalid), never the first estimate (which
    would borrow a future result).  A locator that never output anything
    yields an all-NaN trajectory, which the scorer reports as zero valid
    coverage instead of a fake-valid zero pose.  Yaw stays wrapped to +-pi.
    """
    n = len(odom)
    result = np.full((n, 3), np.nan)
    if len(estimates) == 0 or len(timestamps_est) == 0:
        return result
    t_first = timestamps_est[0]
    est_idx = 0
    have_correction = False
    last_correction = np.zeros(3)
    last_correction_time = 0.0
    for k in range(n):
        t = timestamps_odom[k]
        if t < t_first:
            continue  # before any correction existed: stays NaN
        while est_idx < len(timestamps_est) and timestamps_est[est_idx] <= t:
            last_correction = np.asarray(estimates[est_idx], dtype=float).copy()
            last_correction_time = timestamps_est[est_idx]
            est_idx += 1
            have_correction = True
        if not have_correction:
            continue
        odom_idx = int(np.argmin(np.abs(timestamps_odom - last_correction_time)))
        if k > 0 and odom_idx < k:
            # odom displacement between the correction time and now, in the
            # shared world frame: position deltas compose directly, the yaw
            # delta is frame-independent up to wrap
            dx = odom[k][0] - odom[odom_idx][0]
            dy = odom[k][1] - odom[odom_idx][1]
            dth = wrap_angle(odom[k][2] - odom[odom_idx][2])
            result[k] = last_correction + np.array([dx, dy, dth])
        else:
            result[k] = last_correction
        result[k, 2] = wrap_angle(result[k, 2])
    return result


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def nan_aware_metrics(poses, truth, timestamps_truth):
    """Position-error metrics over the VALID (non-NaN) frames only.

    Invalid frames are reported as coverage, never dropped silently: the
    mean cannot shrink by hiding frames where the locator had no output.
    """
    n = min(len(poses), len(truth))
    err = np.full(n, np.nan)
    for k in range(n):
        if np.all(np.isfinite(poses[k][:3])):
            err[k] = float(np.linalg.norm(poses[k][:2] - truth[k][:2]))
    valid = err[np.isfinite(err)]
    out = {
        "n_frames": int(n),
        "n_valid": len(valid),
        "valid_coverage": float(len(valid) / max(1, n)),
        "invalid_span_s": None,
    }
    if len(valid):
        invalid_mask = ~np.isfinite(err)
        out["invalid_span_s"] = float(np.sum(invalid_mask) * np.mean(np.diff(timestamps_truth[:n]))) if n > 1 else 0.0
        out.update(
            {
                "mean_m": float(valid.mean()),
                "p95_m": float(np.percentile(valid, 95)),
                "end_m": float(valid[-1]),
            }
        )
    else:
        out.update({"mean_m": None, "p95_m": None, "end_m": None})
    return out


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
            with np.load(amcl_path) as z:
                amcl_x = z["x"]
                amcl_y = z["y"]
                amcl_yaw = z["yaw"]
                amcl_stamp = z["stamp"]
            if len(amcl_x) > 0:
                # aligned scoring: match AMCL timestamps to truth timestamps;
                # matches further than half a control second are rejected
                amcl_est = np.column_stack([amcl_x, amcl_y, amcl_yaw])
                amcl_aligned, _pairs = score_aligned(
                    amcl_est, truth, amcl_stamp, timestamps, max_gap_s=0.5
                )
                results["official_amcl_aligned"] = amcl_aligned

                # continuous scoring: last correction + odom propagation;
                # frames before the first correction are invalid (NaN), the
                # metrics cover valid frames only and report their share
                amcl_cont = continuous_pose(amcl_est, odom, amcl_stamp, timestamps)
                results["official_amcl_continuous"] = nan_aware_metrics(
                    amcl_cont, truth, timestamps
                )

    # save
    (output_dir / "scoring_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
