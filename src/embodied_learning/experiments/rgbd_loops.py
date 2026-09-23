"""Lesson 55: marker-appearance loop retrieval as the
place discriminator (the corrected-metric treatment).

Lesson 54 corrected the evaluation metric (implied pose vs truth pose) and
measured the plain geometric matcher at 72% direction correctness on the
ISO arena.  This lesson adds a simulated omnidirectional appearance sensor
(180 marker rays, 5 m range; no rendered RGB or usable depth) whose returns carry an APPEARANCE
marker id - the documented abstraction of texture: every wall patch and
obstacle carries a marker drawn per seed from a 12-marker palette; returns
report the marker with 8% confusion and 10% dropout (a real camera's
appearance noise, not geometry).

Pipeline (oracle-free candidate selection): odometry-arc exclusion ->
per-frame marker histograms (bag of markers) ->
cosine similarity against the reference (start) descriptor -> top-k
retrieval -> lesson-49 K geometric verification -> implied pose ->
corrected correctness.  The geometry sensor (64-ray lidar) is unchanged;
appearance is used ONLY for retrieval.

Groups:
  GEO-ALL   geometric matcher on every probe frame (no retrieval) - the
            corrected-metric baseline;
  GEO-ORACLE   geometric matcher on the truth-oracle candidate frames
            (the lesson-54 reference arm; evaluation-only oracle);
  RGBD-TOPK appearance retrieval top-k -> geometric verify (the treatment).

Pre-registered claims (corrected metric throughout):
  (1) precision: >= 80% of marker top-k candidates are true-loop frames;
  (2) correctness: RGBD-TOPK accepted candidates >= 80% direction
      correctness - appearance retrieval keeps WHAT geometry alone
      (GEO-ALL) dilutes;
  (3) efficiency: RGBD-TOPK examines <= 1/3 of the candidates GEO-ALL
      examines while satisfying (2);
  (4) honest: GEO-ALL correctness recorded without claims.
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

from embodied_learning.experiments.grid_nav import GridNavConfig, build_scenarios
from embodied_learning.experiments.matcher_engineering import (
    MatcherConfig,
    implied_pose,
    is_correct_implied,
    match_once,
)
from embodied_learning.experiments.scan_slam import ray_points

EXPERIMENT = "rgbd_loops_lesson55"
SCHEMA_VERSION = 2
GROUPS = ("GEO-ALL", "GEO-ORACLE", "RGBD-TOPK")
GROUP_NAMES = {
    "GEO-ALL": "纯几何匹配 × 全部探帧（修正判据基线）",
    "GEO-ORACLE": "几何匹配 × 真值候选（评估用 oracle，第 54 课参考）",
    "RGBD-TOPK": "外观检索 top-k + 几何验证（处理组）",
}
MARKER_PALETTE = 12
MARKER_CONFUSION = 0.08
MARKER_DROPOUT = 0.10
# omnidirectional appearance (the place-recognition standard: Ladybug/Theta
# class rigs); a 56-deg forward cone misses the return view - recall 0 recorded
CAM_FOV_DEG = 360.0
CAM_RAYS = 180
CAM_MAX_RANGE_M = 5.0
REF_FRAMES = 60
TOP_K = 12
ACCEPT_MEDIAN_NN_M = 0.15
DRIFT_T_M = 6.0
DRIFT_R_DEG = 80.0
PROBE_STRIDE = 5
DEFAULT_RESULTS = "results/rgbd_loops_2026-09-23_v2"


@dataclass(frozen=True)
class RgbdConfig:
    """ISO arena (lesson-54 arm) + the appearance channel; defaults ARE the
    pre-registered run."""

    base: object = None  # GridNavConfig
    matcher: object = None
    laps: int = 2
    range_noise_sigma_m: float = 0.06
    seed: int = 0
    top_k: int = TOP_K
    probe_stride: int = PROBE_STRIDE

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(
                self,
                "base",
                GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
            )
        if self.matcher is None:
            object.__setattr__(
                self,
                "matcher",
                MatcherConfig(
                    base=self.base,
                    laps=self.laps,
                    range_noise_sigma_m=self.range_noise_sigma_m,
                    seed=self.seed,
                ),
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not (2 <= self.top_k <= 48):
            raise ValueError("top_k out of range")
        if not (1 <= self.probe_stride <= 100):
            raise ValueError("probe_stride out of range")

    @property
    def config(self) -> GridNavConfig:
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------- appearance world
def build_appearance_world(base, seed, *, obstacles=None, world_walls=None):
    """Markerize the geometry actually used by the physical world.

    Wall rectangles are subdivided into patches (each patch its own
    marker); clutter primitives keep one marker each.  Returns
    (obstacles, walls, patch_ids) with patch_ids aligned to
    (*obstacles, *walls) after subdivision.
    """
    from embodied_learning.experiments.grid_nav import Obstacle

    rng = np.random.default_rng([seed, 55000])
    if obstacles is None:
        scenarios = build_scenarios(base, seed)
        obstacles = scenarios[1]["obstacles"]

    def subdiv(rect, n):
        cx, cy, hx, hy = rect.cx, rect.cy, rect.hx, rect.hy
        out = []
        if hx >= hy:  # horizontal wall: split along x
            edges = np.linspace(cx - hx, cx + hx, n + 1)
            for i in range(n):
                out.append(
                    Obstacle(
                        "rect",
                        (edges[i] + edges[i + 1]) / 2,
                        cy,
                        (edges[i + 1] - edges[i]) / 2,
                        hy,
                    )
                )
        else:
            edges = np.linspace(cy - hy, cy + hy, n + 1)
            for i in range(n):
                out.append(
                    Obstacle(
                        "rect",
                        cx,
                        (edges[i] + edges[i + 1]) / 2,
                        hx,
                        (edges[i + 1] - edges[i]) / 2,
                    )
                )
        return out

    walls = []
    from embodied_learning.experiments.grid_nav import build_walls

    physical_walls = (
        build_walls(base.world_size_m, base.res_m) if world_walls is None else world_walls
    )
    for wall in physical_walls:
        if wall.kind == "rect":
            walls.extend(subdiv(wall, 6))
        else:
            walls.append(wall)
    ids = [
        *rng.integers(0, MARKER_PALETTE, len(obstacles)),
        *rng.integers(0, MARKER_PALETTE, len(walls)),
    ]
    return obstacles, tuple(walls), np.asarray(ids, dtype=int)


# --------------------------------------------- simulated marker sensor
def cast_rays_ids(px, py, angles, primitives, ids, max_range):
    """Nearest-hit ranges + the marker id of the hit primitive per ray.

    Same analytic primitives as grid_nav.cast_rays, tracking argmin.
    """
    angles = np.asarray(angles, dtype=float)
    dx = np.cos(angles)
    dy = np.sin(angles)
    best = np.full(angles.shape, float(max_range))
    best_idx = np.full(angles.shape, -1, dtype=int)
    for pi, o in enumerate(primitives):
        if o.kind == "circle":
            relx, rely = px - o.cx, py - o.cy
            b = relx * dx + rely * dy
            c0 = relx * relx + rely * rely - o.r * o.r
            disc = b * b - c0
            ok = disc >= 0.0
            t = np.where(
                ok & ((-b + np.sqrt(np.where(ok, disc, 0.0))) >= 0.0),
                np.maximum(-b - np.sqrt(np.where(ok, disc, 0.0)), 0.0),
                np.inf,
            )
        elif o.kind == "rect":
            with np.errstate(divide="ignore", invalid="ignore"):
                tx0 = (o.cx - o.hx - px) / dx
                tx1 = (o.cx + o.hx - px) / dx
                ty0 = (o.cy - o.hy - py) / dy
                ty1 = (o.cy + o.hy - py) / dy
            tx_lo, tx_hi = np.minimum(tx0, tx1), np.maximum(tx0, tx1)
            ty_lo, ty_hi = np.minimum(ty0, ty1), np.maximum(ty0, ty1)
            vertical = np.abs(dx) < 1e-12
            horizontal = np.abs(dy) < 1e-12
            tx_lo = np.where(
                vertical,
                np.where((px >= o.cx - o.hx) & (px <= o.cx + o.hx), -np.inf, np.inf),
                tx_lo,
            )
            ty_lo = np.where(
                horizontal,
                np.where((py >= o.cy - o.hy) & (py <= o.cy + o.hy), -np.inf, np.inf),
                ty_lo,
            )
            t = np.where(
                np.maximum(tx_lo, ty_lo) <= np.minimum(tx_hi, ty_hi),
                np.maximum(np.maximum(tx_lo, ty_lo), 0.0),
                np.inf,
            )
            t = np.where(np.isfinite(t), t, np.inf)
        else:  # segment
            ex, ey = 2.0 * o.hx, 2.0 * o.hy
            wx, wy = (o.cx - o.hx) - px, (o.cy - o.hy) - py
            cross_de = dx * ey - dy * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wx * ey - wy * ex) / cross_de
                s = (wx * dy - wy * dx) / cross_de
            t = np.where(
                (np.abs(cross_de) > 1e-15) & (t >= 0.0) & (s >= 0.0) & (s <= 1.0), t, np.inf
            )
        nearer = t < best
        best = np.where(nearer, t, best)
        best_idx = np.where(nearer, pi, best_idx)
    hit = best < max_range
    return best, hit, np.where(hit, best_idx, -1)


def rgbd_frame(pose, primitives, ids, rng):
    """One simulated marker histogram; no image or depth values are used."""
    offsets = np.linspace(-math.radians(CAM_FOV_DEG / 2), math.radians(CAM_FOV_DEG / 2), CAM_RAYS)
    angles = pose[2] + offsets
    _ranges, hit, idx = cast_rays_ids(pose[0], pose[1], angles, primitives, ids, CAM_MAX_RANGE_M)
    hist = np.zeros(MARKER_PALETTE)
    for ray in np.flatnonzero(hit):
        if idx[ray] < 0:
            continue
        u = rng.uniform()
        if u < MARKER_DROPOUT:
            continue  # unlabelled return
        marker = int(ids[idx[ray]])
        if rng.uniform() < MARKER_CONFUSION:
            marker = int(rng.integers(0, MARKER_PALETTE))
        hist[marker] += 1.0
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 0 else hist


def appearance_descriptors(stream, primitives, ids, cfg):
    """Marker histogram per frame (the retrieval descriptors)."""
    rng = np.random.default_rng([cfg.seed, 55001])
    return np.vstack([rgbd_frame(p, primitives, ids, rng) for p in stream["truth"][:-1]])


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def probes_from_odometry(est, n_frames, stride, min_arc_m=15.0):
    """Choose candidates using only the estimated chain available to the robot."""
    if est.ndim != 2 or est.shape[1] < 2 or len(est) < n_frames + 1:
        raise ValueError("estimated chain must contain every observed frame")
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(est[: n_frames + 1, :2], axis=0), axis=1)))
    )
    probes = [k for k in range(0, n_frames, stride) if arc[k] >= min_arc_m]
    return probes, arc


# ------------------------------------------------------------- collectors
def collect_iso_stream(base_cfg, seed):
    """The lesson-54 ISO arm patrol (square walls + filtered scene clutter),
    on the closed inset patrol."""
    from embodied_learning.experiments.aniso_env import (
        PATROL_INSET,
        _build_world,
        patrol_start,
    )
    from embodied_learning.experiments.matcher_engineering import collect_patrol_laps

    obstacles, walls = _build_world("ISO", seed, base_cfg.base)
    return (
        collect_patrol_laps(
            obstacles,
            patrol_start(base_cfg.base, seed)[0],
            base_cfg,
            walls=None,
            patrol=PATROL_INSET,
        ),
        obstacles,
        walls,
    )


def expected_npz_keys(report):
    return set(report["archive_keys"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, log=print):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    if config is None:
        config = RgbdConfig()
    elif config.__class__ is not RgbdConfig:
        raise TypeError("config must be RgbdConfig")
    from embodied_learning.experiments.matcher_engineering import (
        build_stream_data,
        candidate_frames,
    )

    started = time.perf_counter()
    base_cfg = config.matcher
    stream, obstacles, walls = collect_iso_stream(base_cfg, seed)
    stream["scene"] = 1
    if stream["laps_done"] < config.laps:
        raise RuntimeError("patrol incomplete: invalid world")
    log(f"stream: frames {len(stream['truth']) - 1} laps {stream['laps_done']}")

    # appearance channel
    obstacles, walls, ids = build_appearance_world(base_cfg.base, seed)
    primitives = (*obstacles, *walls)
    descs = appearance_descriptors(stream, primitives, ids, config)
    ref_desc = descs[:REF_FRAMES].mean(axis=0)

    ranges, est, ref_pts, offsets = build_stream_data(stream, base_cfg, 0)
    oracle = candidate_frames(stream, base_cfg)
    oracle_set = set(oracle)

    rows = {group: [] for group in GROUPS}
    # Spatial exclusion uses the estimated odometry chain.  The historical
    # lesson-55 record used truth arc here and is retained as an old protocol.
    probes, est_arc = probes_from_odometry(est, len(ranges), config.probe_stride)
    sims = {k: cosine(descs[k], ref_desc) for k in probes}
    topk = sorted(probes, key=lambda k: -sims[k])[: config.top_k]

    for k in probes:
        cur_pts, _i = ray_points(est[k], ranges[k], stream["hit"][k], offsets)
        if len(cur_pts) < base_cfg.min_points or len(ref_pts) < base_cfg.min_points:
            continue
        m = match_once(ref_pts, cur_pts, base_cfg, "K")
        implied = (
            implied_pose(np.asarray(m["delta"], dtype=float), est[k])
            if m["delta"] is not None
            else None
        )
        drift_t = float(np.linalg.norm(implied[:2] - est[k][:2]))
        drift_r = abs(math.degrees(implied[2] - est[k][2]))
        accepted = bool(
            math.isfinite(m["median_nn"])
            and m["median_nn"] <= ACCEPT_MEDIAN_NN_M
            and drift_t <= DRIFT_T_M
            and drift_r <= DRIFT_R_DEG
        )
        truth_k = stream["truth"][k]
        correct = is_correct_implied(implied, truth_k) if implied is not None else False
        in_oracle = k in oracle_set
        row = {
            "frame": k,
            "accepted": accepted,
            "correct": correct,
            "median_nn": m["median_nn"],
            "sim": sims[k],
            "in_oracle": in_oracle,
        }
        rows["GEO-ALL"].append(row)
        if in_oracle:
            rows["GEO-ORACLE"].append(row)
        if k in topk:
            rows["RGBD-TOPK"].append(row)

    def agg(rows):
        n = len(rows)
        acc = [r for r in rows if r["accepted"]]
        cor = [r for r in acc if r["correct"]]
        return {
            "n_candidates": n,
            "accepted": len(acc),
            "acceptance": len(acc) / max(1, n),
            "direction_correctness": len(cor) / max(1, len(acc)),
            "correct_accept": len(cor),
        }

    aggregates = {group: agg(rows[group]) for group in GROUPS}

    # precision@top-k: a retrieved frame is a TRUE loop frame iff its truth
    # pose is near the start (the same near-start predicate as the oracle;
    # evaluation only).  Exact-frame recall against the stride-3 oracle
    # subsample is unfair - the retriever legitimately finds OTHER frames of
    # the same true-loop region (the recorded first-run lesson).
    def is_true_loop(k):
        return float(np.linalg.norm(stream["truth"][k][:2] - stream["truth"][0][:2])) < 1.5

    near_start = {k: is_true_loop(k) for k in probes}
    precision = len([k for k in topk if near_start[k]]) / max(1, len(topk))
    ratio = aggregates["RGBD-TOPK"]["n_candidates"] / max(1, aggregates["GEO-ALL"]["n_candidates"])
    rg = aggregates["RGBD-TOPK"]
    pipeline_success = any(r["accepted"] and r["correct"] for r in rows["RGBD-TOPK"])
    hypothesis = {
        "claim": (
            "模拟标记外观检索在候选筛选和排序中不使用轨迹真值；top-k 精确率 >= 80%，"
            "几何验证后的方向正确率 >= 80%"
            "（修正判据），且只审查 <= 1/3 的候选——外观是几何之外的第二信息源"
        ),
        "criteria": {
            "precision": "top-k 检索中真回环帧（近起点谓词）占比 >= 80%",
            "correctness": "RGBD-TOPK accepted 中 correct >= 80%（修正判据）",
            "efficiency": "RGBD-TOPK 候选数 <= GEO-ALL 的 1/3",
        },
        "results": {
            "precision_topk": float(precision),
            "precision_met": bool(precision >= 0.80),
            "correctness_met": bool(rg["direction_correctness"] >= 0.80),
            "pipeline_success": bool(pipeline_success),
            "efficiency_met": bool(ratio <= 1 / 3),
            "n_candidates": {g: aggregates[g]["n_candidates"] for g in GROUPS},
        },
    }
    archives = {
        "topk_frames": np.asarray(sorted(topk), dtype=int),
        "oracle_frames": np.asarray(sorted(oracle_set), dtype=int),
        "probe_frames": np.asarray(sorted(probes), dtype=int),
        "probe_accepted": np.array([r["accepted"] for r in rows["GEO-ALL"]], dtype=int),
        "probe_correct": np.array([r["correct"] for r in rows["GEO-ALL"]], dtype=int),
        "probe_sims": np.array([r["sim"] for r in rows["GEO-ALL"]], dtype=float),
        "probe_near_start": np.array([near_start[k] for k in probes], dtype=int),
        "probe_frames_all": np.asarray(probes, dtype=int),
        "estimated_arc_m": est_arc,
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
            "arena": "ISO（第 54 课臂）：方墙 + 稀疏障碍，闭圈内方巡游 16 m 圈",
            "appearance": {
                "palette": MARKER_PALETTE,
                "confusion": MARKER_CONFUSION,
                "dropout": MARKER_DROPOUT,
                "note": "模拟标记直方图，不是 RGB-D 图像；深度值未用于描述子或检索",
                "camera": f"omnidirectional marker rays: {CAM_RAYS} / 360 deg / {CAM_MAX_RANGE_M} m",
            },
            "retrieval": {
                "descriptor": "marker histogram (bag of markers), cosine",
                "reference": f"frames 0..{REF_FRAMES} pooled",
                "top_k": config.top_k,
                "probe_stride": config.probe_stride,
                "probe_exclusion": "estimated odometry arc >= 15 m",
                "oracle_free": True,
            },
            "verification": "lesson-49 K matcher + drift-envelope acceptance (lesson-54 corrected)",
            "correctness": "implied pose vs truth pose, 0.5 m / 5 deg (lesson-54 corrected)",
            "truth_uses": [
                "simulator observation generation at physical pose",
                "evaluation labels and oracle reference group only",
            ],
        },
        "aggregates": aggregates,
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "archive_keys": sorted(archives.keys()),
        "trajectories_sha256": digest(output / "trajectories.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="lesson-55 marker loop record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
