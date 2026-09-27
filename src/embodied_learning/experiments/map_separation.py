"""Map-separation analysis over the paired-pilot fixed scans (roadmap P1).

Decomposes the self-built-map failure on a FIXED scan set into two gaps:

  representation gap : ideal occupancy map vs same-scan truth-pose
                       projection map (same perfect landing - isolates the
                       map SOURCE: clean geometry vs real scan endpoints);
  landing gap        : same-scan truth-pose projection vs same-scan
                       odometry-pose projection (same scans - isolates the
                       POSE the scans were painted at).

Also localizes per-episode filter divergence (onset frame + arc length at
onset) and compares two particle budgets on the identical scan set.

Inputs are existing benchmark result directories (no re-simulation):
  uv run python -m embodied_learning.experiments.map_separation \
      --hundred results/navigation_benchmark_2026-09-24_pilot_v4 \
      --four-hundred results/nav_sep_particles400_v1 \
      --output results/map_separation_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIVERGENCE_MEAN_M = 1.0  # episode mean above this counts as diverged


def _episode_stats(row):
    return {
        "est_mean_m": row["est_mean_m"],
        "ideal_mean_m": row["pf_true_mean_m"],
        "truth_proj_mean_m": row["pf_sensor_truth_mean_m"],
        "self_mean_m": row["pf_self_mean_m"],
        "self_p95_m": row["pf_self_p95_m"],
        "align_precision": row["map_alignment_precision"],
        "align_recall": row["map_alignment_recall"],
        # divergence is judged PER ARM: a clean ideal-map divergence pins
        # filter instability (the map cannot be the cause); self-built
        # divergence is confounded with its bad map and only counted
        "ideal_diverged": bool(row["pf_true_mean_m"] > DIVERGENCE_MEAN_M),
        "truthproj_diverged": bool(row["pf_sensor_truth_mean_m"] > DIVERGENCE_MEAN_M),
        "self_diverged": bool(row["pf_self_mean_m"] > DIVERGENCE_MEAN_M),
    }


def load_benchmark(result_dir):
    summary = json.loads(
        (Path(result_dir) / "summary.json").read_text(encoding="utf-8")
    )
    episodes = {}
    for row in summary["rows"]:
        if row["status"] != "ok":
            continue
        key = f"{row['arm'].lower()}_w{row['world_seed']}_s{row['sensor_seed']}"
        episodes[key] = {"row": row, "stats": _episode_stats(row), "dir": Path(result_dir) / key}
    return {"particles": summary["plan"]["particles"], "episodes": episodes}


def divergence_onset(episode_dir, chain_key="pf_true_chain", err_key="pf_true_err"):
    """First frame after which the error stays predominantly above threshold.

    Returns (onset_frame, onset_arc_m) or (None, None) for clean episodes.
    Predominantly = the median of the remaining trajectory stays above 1 m,
    so a brief spike during recovery does not count as divergence onset.
    """
    with np.load(Path(episode_dir) / "trajectories.npz") as z:
        err = np.asarray(z[err_key], dtype=float)
        truth = np.asarray(z["truth_chain"], dtype=float)
    n = len(err)
    above = err > DIVERGENCE_MEAN_M
    for k in range(n):
        if above[k] and np.median(err[k:]) > DIVERGENCE_MEAN_M:
            arc = float(np.sum(np.linalg.norm(np.diff(truth[: k + 1, :2], axis=0), axis=1)))
            return k, arc
    return None, None


def analyze(hundred_dir, four_hundred_dir):
    lo = load_benchmark(hundred_dir)
    hi = load_benchmark(four_hundred_dir)
    shared = sorted(set(lo["episodes"]) & set(hi["episodes"]))
    if not shared:
        raise ValueError("the two benchmark dirs share no episode keys")

    def dist(episodes, keys, field):
        vals = np.array([episodes[k]["stats"][field] for k in keys], dtype=float)
        return {
            "median": float(np.median(vals)),
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "n": len(vals),
        }

    report = {
        "experiment": "map_separation",
        "protocol": {
            "fixed": "same patrol scans, same collection route, same seeds; only the PF particle budget differs between the two benchmark dirs",
            "divergence_rule": f"episode mean > {DIVERGENCE_MEAN_M} m in the ideal-map or self-built arm",
            "gaps": {
                "representation": "truth-projection map minus ideal map (same perfect landing; isolates map source)",
                "landing": "self-built minus truth-projection (same scans; isolates odom-pose painting error)",
            },
        },
        "episodes": {},
        "particles": {},
    }
    for label, bundle in (("p100", lo), ("p400", hi)):
        eps = bundle["episodes"]
        keys = sorted(eps)
        report["particles"][label] = {
            "particle_count": bundle["particles"],
            "ideal": dist(eps, keys, "ideal_mean_m"),
            "truth_proj": dist(eps, keys, "truth_proj_mean_m"),
            "self": dist(eps, keys, "self_mean_m"),
            "ideal_diverged_episodes": [k for k in keys if eps[k]["stats"]["ideal_diverged"]],
            "truthproj_diverged_episodes": [
                k for k in keys if eps[k]["stats"]["truthproj_diverged"]
            ],
            "self_diverged_episodes": [k for k in keys if eps[k]["stats"]["self_diverged"]],
            "align_precision_median": float(
                np.median([eps[k]["stats"]["align_precision"] for k in keys])
            ),
            "align_recall_median": float(
                np.median([eps[k]["stats"]["align_recall"] for k in keys])
            ),
        }

    # decomposition on the CLEAN subset (ideal map did not diverge): a
    # diverged episode mixes tracking regimes and would poison the gap
    # estimate; conditional medians isolate map-quality effects
    report["decomposition_m"] = {}
    for label, bundle in (("p100", lo), ("p400", hi)):
        eps = bundle["episodes"]
        clean = [k for k in sorted(eps) if not eps[k]["stats"]["ideal_diverged"]]
        if not clean:
            continue
        rep = float(
            np.median([eps[k]["stats"]["truth_proj_mean_m"] for k in clean])
            - np.median([eps[k]["stats"]["ideal_mean_m"] for k in clean])
        )
        lan = float(
            np.median([eps[k]["stats"]["self_mean_m"] for k in clean])
            - np.median([eps[k]["stats"]["truth_proj_mean_m"] for k in clean])
        )
        report["decomposition_m"][label] = {
            "n_clean_episodes": len(clean),
            "representation_gap_median_m": rep,
            "landing_gap_median_m": lan,
        }
    report["decomposition_m"]["note"] = (
        "conditional on the ideal-map arm not diverging; landing gap dominates"
        " if representation gap ~ 0 while landing gap >> 0"
    )

    # divergence onset localization: IDEAL-map divergences only - the map is
    # clean geometry there, so an onset pins filter behaviour, not map error
    onsets = {}
    for key in shared:
        ep = hi["episodes"][key]
        if not ep["stats"]["ideal_diverged"]:
            continue
        frame, arc = divergence_onset(ep["dir"], err_key="pf_true_err")
        if frame is not None:
            onsets[key] = {"onset_frame": int(frame), "onset_arc_m": round(arc, 2)}
    report["divergence_onsets_ideal_map_p400"] = onsets
    return report


def main():
    parser = argparse.ArgumentParser(description="map-separation decomposition analysis")
    parser.add_argument("--hundred", required=True)
    parser.add_argument("--four-hundred", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = analyze(args.hundred, args.four_hundred)
    report["inputs"] = {
        "hundred": str(Path(args.hundred).resolve()),
        "four_hundred": str(Path(args.four_hundred).resolve()),
        "hundred_summary_sha256": hashlib.sha256(
            (Path(args.hundred) / "summary.json").read_bytes()
        ).hexdigest(),
        "four_hundred_summary_sha256": hashlib.sha256(
            (Path(args.four_hundred) / "summary.json").read_bytes()
        ).hexdigest(),
    }
    (out / "map_separation_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    p = report["particles"]
    print(json.dumps(report["decomposition_m"], indent=1))
    for label in ("p100", "p400"):
        d = p[label]
        print(
            f"{label}: ideal med {d['ideal']['median']:.3f} | truth-proj {d['truth_proj']['median']:.3f}"
            f" | self {d['self']['median']:.3f} | ideal-div {len(d['ideal_diverged_episodes'])}"
            f" truthproj-div {len(d['truthproj_diverged_episodes'])} self-div {len(d['self_diverged_episodes'])}"
        )
    print(f"ideal-map divergence onsets: {len(report['divergence_onsets_ideal_map'])}")


if __name__ == "__main__":
    main()
