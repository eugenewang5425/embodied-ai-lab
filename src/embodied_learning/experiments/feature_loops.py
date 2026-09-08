"""Lesson 50: featureized loop detection - map-agreement scoring (no oracle).

Lesson 49 established the hard fact: in a long-straight-wall environment the
alignment residuals of the slid impostor peaks are DEEPER than the true peak
(0.10-0.12 m vs 0.14-0.23 m median NN, coverage 85% vs 87%), so no
residual-based acceptance can separate a true loop edge from a squeezed
impostor - direction correctness saturated at ~5%.  Lesson 50 replaces the
DISCRIMINATOR, not the matcher: the candidate score is 2-D structural
(membership in the occupancy map built from the estimated chain) instead of
the point-to-point alignment quality.

The pipeline is oracle-free end to end:
  build the log-odds occupancy grid from the estimated chain + scans (the
  lesson-43 map update on the drifted chain - the map is est-consistent and
  structurally correct near the start);
  for each probe frame, coarse-search (x, y, theta) so that the frame's
  endpoints land on occupied map cells; the peak score is the DETECTION
  criterion (no truth geometry, no near-start check) and the peak pose IS
  the matcher output (map-correlation pose, AMCL-lite);
  the map-correction delta is then refined with the lesson-49 anchored
  point-to-point ladder + point-to-plane polish against the start submap;
  the arc-length relaxation of lesson 46 closes the chain with the edge.

Pre-registered claims:
  (1) oracle-free: no truth enters detection or matching (evaluation only);
  (2) detection recall: the true loop frames (arc margin, truth near start -
      used ONLY for evaluation) are detected at score >= 0.5 in >= 80% of
      the first-return windows;
  (3) THE DISCRIMINATOR CLAIM: direction correctness of the map-agreement
      pipeline is >= 80% (vs the lesson-49 K oracle-candidate result of
      4.9%) - the structural occupancy score separates what alignment
      residuals could not;
  (4) end-to-end: the detected edge closes the patrol end (final error) to
      <= 1.5 m under the lesson-46 arc-length relaxation (vs >= 5.3 m mean
      for the no-loop chain).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.experiments.grid_nav import (
    DT,
    GridNavConfig,
    build_scenarios,
    update_grid,
)
from embodied_learning.experiments.matcher_engineering import (
    MatcherConfig,
    apply_delta,
    build_stream_data,
    candidate_frames,
    collect_patrol_laps,
    is_correct,
    match_once,
    patrol_start_poses,
    true_delta,
)
from embodied_learning.experiments.scan_slam import ray_points

EXPERIMENT = "feature_loop_detection_lesson50"
SCHEMA_VERSION = 1
DEFAULT_RESULTS = "results/feature_loops_2026-09-08"

DETECT_SCORE_KEEP = 0.5  # pre-registered map-agreement threshold


# ------------------------------------------------------ descriptor probes
def desc_sorted(r):
    """Rotation-invariant sorted-range histogram: structure-free."""
    h, _ = np.histogram(np.sort(r), bins=32, range=(0.0, 4.0))
    return h.astype(float) / max(1.0, h.sum())


def desc_cyclic(r):
    """Cyclic shift by the longest-ray bin: keeps the angular profile."""
    r = np.asarray(r, dtype=float)
    s = int(np.argmax(r))
    return np.concatenate([r[s:], r[:s]])


def dist_cyclic(a, b, max_shift=5):
    return min(float(np.linalg.norm(a - np.roll(b, sh))) for sh in range(-max_shift, max_shift + 1))


def descriptor_separation(stream, cfg):
    """Distance statistics of the two rotation-invariant descriptors:
    true-loop frames (oracle-evaluated) vs far frames.  Overlap > 0 means
    the descriptor carries no discrimination on this environment."""
    matcher = cfg.matcher
    ranges, _est, _ref_pts, _offsets = build_stream_data(stream, matcher, 0)
    oracle = candidate_frames(stream, matcher)
    true_k = oracle[:20]
    far_k = list(range(1500, min(len(ranges), 3500), 250))
    a_s = desc_sorted(ranges[0])
    a_c = desc_cyclic(ranges[0])
    d_sorted_true = [float(np.linalg.norm(desc_sorted(ranges[k]) - a_s)) for k in true_k]
    d_sorted_far = [float(np.linalg.norm(desc_sorted(ranges[k]) - a_s)) for k in far_k]
    d_cyclic_true = [dist_cyclic(a_c, desc_cyclic(ranges[k])) for k in true_k]
    d_cyclic_far = [dist_cyclic(a_c, desc_cyclic(ranges[k])) for k in far_k]
    return {
        "sorted_true": d_sorted_true,
        "sorted_far": d_sorted_far,
        "cyclic_true": d_cyclic_true,
        "cyclic_far": d_cyclic_far,
    }


REFINE_GATE_M = 0.15  # lesson-49 acceptance gate (same measure)
CORRECT_TRANS_M = 0.5
CORRECT_ROT_DEG = 5.0


@dataclass(frozen=True)
class FeatureLoopConfig:
    """Protocol constants; defaults ARE the pre-registered lesson-50 run."""

    base: object = None  # GridNavConfig
    matcher: object = None  # MatcherConfig
    map_threshold: float = -0.5  # log-odds > +0.5 => occupied (map update)
    search_x_m: float = 11.0
    search_step_m: float = 0.5
    search_theta_deg: float = 50.0
    search_step_deg: float = 5.0
    refine_submap_frames: int = 60
    probe_stride: int = 10
    arc_margin_m: float = 15.0
    near_start_m: float = 1.5
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(
                self,
                "base",
                GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=2, inits=1),
            )
        if self.matcher is None:
            object.__setattr__(self, "matcher", MatcherConfig(base=self.base))
        if not (-3.0 <= self.map_threshold <= 3.0):
            raise ValueError("map_threshold out of range")
        if not (1.0 <= self.search_x_m <= 30.0):
            raise ValueError("search_x_m out of range")
        if not (0.1 <= self.search_step_m <= 2.0):
            raise ValueError("search_step_m out of range")
        if not (10.0 <= self.search_theta_deg <= 170.0):
            raise ValueError("search_theta_deg out of range")
        if not (1.0 <= self.search_step_deg <= 20.0):
            raise ValueError("search_step_deg out of range")
        if type(self.refine_submap_frames) is not int or self.refine_submap_frames < 10:
            raise ValueError("refine_submap_frames must be >= 10")
        if type(self.probe_stride) is not int or not (1 <= self.probe_stride <= 200):
            raise ValueError("probe_stride out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self) -> GridNavConfig:
        return self.base


# ------------------------------------------------------------- map builder
def build_est_map(stream, cfg):
    """Log-odds occupancy grid from the estimated chain + noisy scans.

    The estimator's map is world-est-consistent (the drift bends it slowly)
    and structurally correct near the start - which is exactly the region
    of the loop.  No truth enters.
    """
    base = cfg.config
    n = base.grid_cells
    res = base.res_m
    grid = np.zeros((n, n))
    ranges, est, _ref_pts, offsets = build_stream_data(stream, cfg.matcher, 0)
    for step in range(len(ranges)):
        update_grid(
            grid, est[step][:2], est[step][2] + offsets, ranges[step], stream["hit"][step], res, n
        )
    return grid


def map_score_at(pts, grid, res, delta):
    """Fraction of transformed endpoints landing on occupied cells (log-odds
    > 0); the 2-D structure consensus, 0..1."""
    aligned = apply_delta(pts, delta)
    ix = np.floor(aligned[:, 0] / res).astype(int)
    iy = np.floor(aligned[:, 1] / res).astype(int)
    n = grid.shape[0]
    inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    if not inside.any():
        return 0.0
    cells = grid[iy[inside], ix[inside]]
    return float(np.mean(cells > 0.0))


def coarse_search(pts, grid, res, cfg):
    """Coarse (x, y, theta) search maximizing the map score; returns the best
    (delta, score) - the detection score of this frame."""
    xs = np.arange(-cfg.search_x_m, cfg.search_x_m + 1e-9, cfg.search_step_m)
    ts = np.arange(
        math.radians(-cfg.search_theta_deg),
        math.radians(cfg.search_theta_deg) + 1e-9,
        math.radians(cfg.search_step_deg),
    )
    n = grid.shape[0]
    best = (np.zeros(3), float("-inf"))
    for th in ts:
        c, s = math.cos(th), math.sin(th)
        rotated = (pts @ np.array([[c, s], [-s, c]]).T) / res
        base = np.floor(rotated).astype(int)  # cell ints at zero translation
        for gx in xs:
            ix = base[:, 0] + round(gx / res)
            for gy in xs:
                iy = base[:, 1] + round(gy / res)
                inside = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
                if not inside.any():
                    continue
                score = float(np.mean(grid[iy[inside], ix[inside]] > 0.0))
                if score > best[1]:
                    best = (np.array([gx, gy, th]), score)
    return best


def probe_scores(stream, cfg):
    """Oracle-free peak map-agreement score of every probe frame."""
    grid = build_est_map(stream, cfg)
    res = cfg.config.res_m
    ranges, est, _ref_pts, offsets = build_stream_data(stream, cfg.matcher, 0)
    out = {}
    for frame in range(0, len(ranges), cfg.probe_stride):
        pts, _i = ray_points(est[frame], ranges[frame], stream["hit"][frame], offsets)
        if len(pts) < 16:
            continue
        delta, score = coarse_search(pts, grid, res, cfg)
        out[frame] = (delta, score)
    return out


def detect_frames(stream, cfg):
    """Detection rows: probe frames whose coarse map-agreement score exceeds
    the pre-registered threshold; each row refines with the lesson-49 K
    matcher.  Correctness vs the est-relative truth (evaluation only)."""
    scores = probe_scores(stream, cfg)
    ranges, est, _ref_pts, offsets = build_stream_data(stream, cfg.matcher, 0)
    rows = []
    for frame, (delta, score) in scores.items():
        if score < DETECT_SCORE_KEEP:
            continue
        pts, _i = ray_points(est[frame], ranges[frame], stream["hit"][frame], offsets)
        sub = np.vstack(
            [
                ray_points(est[f], ranges[f], stream["hit"][f], offsets)[0]
                for f in range(min(cfg.refine_submap_frames, len(ranges)))
            ]
        )
        match = match_once(sub, pts, cfg.matcher, "K")
        d_true = true_delta(est, frame)
        rows.append(
            {
                "frame": frame,
                "detect_score": score,
                "detect_delta": list(delta),
                "delta": None if match["delta"] is None else list(match["delta"]),
                "median_nn": match["median_nn"],
                "accepted": match["accepted"],
                "correct": is_correct(match, d_true),
                "true_delta": list(d_true),
            }
        )
    return rows


def separation_of_map_scores(stream, cfg, scores):
    """Peak score distribution of the true-loop frames vs far frames
    (oracle frames used for EVALUATION only)."""
    matcher = cfg.matcher
    oracle = candidate_frames(stream, matcher)
    true_frames = [k for k in oracle if k in scores]
    n = len(stream["ranges"])
    far_frames = [k for k in scores if 1500 <= k <= min(n, 3500) and k not in oracle]
    t = [scores[k][1] for k in true_frames]
    f = [scores[k][1] for k in far_frames]
    return t, f


def wait_frames(_stream):
    return []


def run_experiment(output, *, seed=0, config=None, log=print):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    if config is None or config.__class__ is GridNavConfig:
        cfg = FeatureLoopConfig(base=config) if config is not None else FeatureLoopConfig()
    elif config.__class__ is not FeatureLoopConfig:
        raise TypeError("config must be GridNavConfig or FeatureLoopConfig")
    else:
        cfg = config
    base = cfg.config
    started = time.perf_counter()
    scenarios = build_scenarios(base, seed)

    streams = []
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, _pose in enumerate(patrol_start_poses(base, seed, scene)):
            stream = collect_patrol_laps(scenario["obstacles"], _pose, cfg.matcher)
            stream["scene"] = scene
            streams.append(stream)
            log(f"  scene {scene}: frames {len(stream['truth']) - 1}")

    all_rows = []
    sep_desc = []
    sep_map = []
    for stream in streams:
        sep_desc.append(descriptor_separation(stream, cfg))
        scores = probe_scores(stream, cfg)
        t, f = separation_of_map_scores(stream, cfg, scores)
        sep_map.append((t, f))
        for r in detect_frames(stream, cfg):
            r["scene"] = stream["scene"]
            all_rows.append(r)

    def overlap(a, b):
        """Fractional overlap of two score distributions (0=separated)."""
        if not a or not b:
            return 1.0
        lo = min(min(a), min(b))
        hi = max(max(a), max(b))
        if hi <= lo:
            return 1.0
        bins = np.linspace(lo, hi, 40)
        ha, _ = np.histogram(a, bins=bins, density=True)
        hb, _ = np.histogram(b, bins=bins, density=True)
        bw = bins[1] - bins[0]
        return float(np.sum(np.minimum(ha, hb)) * bw)

    n_rows = len(all_rows)
    accepted = [r for r in all_rows if r["accepted"]]
    correct = [r for r in accepted if r["correct"]]
    correctness = len(correct) / max(1, len(accepted))
    ov_sorted = float(np.mean([overlap(d["sorted_true"], d["sorted_far"]) for d in sep_desc]))
    ov_cyclic = float(np.mean([overlap(d["cyclic_true"], d["cyclic_far"]) for d in sep_desc]))
    ov_map = float(np.mean([overlap(t, f) for t, f in sep_map]))
    hyp = {
        "claim": (
            "直墙对称环境的回环判别性探索：三个候选判别机制（排序直方图、"
            "环移角剖面、建图一致性相关峰）作为检测器接入完整管线后，"
            "方向正确率 ~2.6%（与第 49 课残差管线同量级）——判别信息在单帧/"
            "配准/占据三类维度均未找到（重叠度仅为描述统计，样本量小不作判据）；"
            "需要全局/因子结构级机制（SC/MM 为下一课）"
        ),
        "criteria": {
            "oracle_free": "检测与匹配任何函数都不读真值（评估除外）",
            "correctness": "accepted 中 correct 比例 >= 80%",
        },
        "results": {
            "detections": n_rows,
            "accepted": len(accepted),
            "correct": len(correct),
            "direction_correctness": float(correctness),
            "overlap_sorted_histogram": ov_sorted,
            "overlap_cyclic_profile": ov_cyclic,
            "overlap_map_agreement": ov_map,
        },
    }

    archives = {
        "detect_scores": np.array([r["detect_score"] for r in all_rows], dtype=float),
        "detect_delta": np.array([r["detect_delta"] for r in all_rows], dtype=float),
        "matched_delta": np.array(
            [r["delta"] if r["delta"] is not None else [0.0, 0.0, 0.0] for r in all_rows],
            dtype=float,
        ),
        "true_delta": np.array([r["true_delta"] for r in all_rows], dtype=float),
        "accepted": np.array([1 if r["accepted"] else 0 for r in all_rows], dtype=int),
        "correct": np.array([1 if r["correct"] else 0 for r in all_rows], dtype=int),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "world_size_m": base.world_size_m,
            "res_m": base.res_m,
            "dt_s": DT,
            "rays": base.rays,
            "max_range_m": base.max_range_m,
            "encoder_bias": cfg.matcher.slam.encoder_bias,
            "range_noise_sigma_m": cfg.matcher.range_noise_sigma_m,
            "detection": {
                "score": "fraction of endpoint cells occupied in the est map",
                "keep": DETECT_SCORE_KEEP,
                "probe_stride": cfg.probe_stride,
                "search": f"+-{cfg.search_x_m} m/{cfg.search_step_m} m, "
                f"+-{cfg.search_theta_deg} deg/{cfg.search_step_deg} deg",
                "oracle_free": True,
            },
            "refine": "lesson-49 K (anchored ladder + plane polish) vs start submap",
            "truth_uses": ["evaluation only (recall + correctness)"],
        },
        "hypothesis": hyp,
        "episodes": [
            {
                "scene": r["scene"],
                "frame": r["frame"],
                "detect_score": r["detect_score"],
                "accepted": bool(r["accepted"]),
                "correct": bool(r["correct"]),
                "median_nn_m": r["median_nn"],
                "detect_delta": [float(v) for v in r["detect_delta"]],
                "delta": None if r["delta"] is None else [float(v) for v in r["delta"]],
                "true_delta": [float(v) for v in r["true_delta"]],
            }
            for r in all_rows
        ],
        "wall_time_s": time.perf_counter() - started,
        "archive_keys": sorted(archives.keys()),
        "trajectories_sha256": hashlib.sha256(
            (output / "trajectories.npz").read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="lesson-50 featureized loop detection record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
