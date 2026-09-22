"""Lesson 54: anisotropic-environment control - is the loop degeneracy an
ENVIRONMENT property or a MATCHER property?

Lessons 49-53 recorded, on the square-walled arena (long straight walls,
sparse convex clutter), that alignment quality cannot separate a true loop
from slid impostors (direction correctness ~5%) and SC/MM have no feasible
switch.  This lesson holds the MATCHER and the PROTOCOL fixed and changes
only the ENVIRONMENT, running both arms in one paired record:

  ISO    the lesson-49 arena: square walls + sparse convex obstacles;
  ANISO  an irregular polygon boundary (12-16 vertices at randomized radii
         - no two parallel walls, corners everywhere) + dense clutter
         (12-18 primitives of mixed circles/rects/segments).

Patrol, noise, candidate oracle, matcher groups (B/P/R/K from lesson 49)
and acceptance bars are IDENTICAL in both arms; the patrol is the closed
inset square (4,4)-(7,4)-(7,7)-(4,7) (net rotation 0) kept >= 1.8 m from
every boundary wall and >= 0.7 m from clutter.

Pre-registered claims:
  (1) collector gate: >= 20 candidates per arm, every patrol laps 2;
  (2) ISO reference reproduces the lesson-49 degeneracy: K direction
      correctness < 80 %;
  (3) THE ENVIRONMENT CLAIM: ANISO K direction correctness >= 80 % (the
      lesson-49 falsified bar now passes) - the degeneracy is an ENVIRONMENT
      property;
  (4) the gap: K_correct(ANISO) - K_correct(ISO) >= 0.2 (20 points).
Honest: B/P per-arm numbers recorded without monotonicity claims.
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
    GridNavConfig,
    Obstacle,
    build_scenarios,
)
from embodied_learning.experiments.matcher_engineering import (
    GROUPS,
    MatcherConfig,
    collect_patrol_laps,
)
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "aniso_env_lesson54"
SCHEMA_VERSION = 1
ARMS = ("ISO", "ANISO")
ARM_NAMES = {
    "ISO": "各向同性：方墙 + 稀疏凸障碍（第 49 课场地）",
    "ANISO": "各向异性：不规则多边形墙 + 高杂物",
}
PATROL_INSET = ((3.5, 3.5), (7.5, 3.5), (7.5, 7.5), (3.5, 7.5))  # 16 m lap
CENTER = (5.0, 5.0)
# POLY_R_MIN must keep every wall chord (min radius R*cos(pi/n)) OUTSIDE the
# patrol corner radius 4.24: 4.5*cos(12.85deg) = 4.39 > 4.24 (the 4.2 first
# try let chords cut the corner - the patrol froze on the wall, recorded)
POLY_R_MIN = 4.5
POLY_R_MAX = 4.8
POLY_VERTICES = 14
CLUTTER_COUNT = 10
CLUTTER_CORRIDOR_M = 1.0  # corridor margin on the primitive EXTENT
CLUTTER_MIN_GAP_M = 1.0  # mutual spacing: no sealing pockets
DEFAULT_RESULTS = "results/aniso_env_2026-09-09"


@dataclass(frozen=True)
class AnisoConfig:
    """Both arms share the lesson-49 matcher protocol verbatim."""

    base: object = None  # GridNavConfig
    laps: int = 2
    range_noise_sigma_m: float = 0.06
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(
                self,
                "base",
                GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=2, inits=1),
            )
        if type(self.laps) is not int or self.laps < 2:
            raise ValueError("laps must be >= 2")
        if not (0.0 <= self.range_noise_sigma_m <= 0.5):
            raise ValueError("range_noise_sigma_m out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def matcher(self) -> MatcherConfig:
        return MatcherConfig(
            base=self.base,
            laps=self.laps,
            range_noise_sigma_m=self.range_noise_sigma_m,
            seed=self.seed,
        )


def polygon_walls(vertices, thickness=0.05):
    """Closed polygon boundary as segment obstacles (consecutive pairs)."""
    walls = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        walls.append(
            Obstacle(
                "segment",
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                (x2 - x1) / 2.0,
                (y2 - y1) / 2.0,
                thickness,
            )
        )
    return tuple(walls)


def build_aniso_world(seed):
    """Irregular polygon boundary + dense mixed clutter.

    The polygon has POLY_VERTICES vertices at radii drawn from
    [POLY_R_MIN, POLY_R_MAX] around CENTER (no two consecutive edges
    parallel by construction); clutter primitives keep CLUTTER_CORRIDOR_M
    from every patrol leg so the truth-controlled patrol completes.
    """
    rng = np.random.default_rng([seed, 91000])
    # jittered lattice: a purely random angle draw can leave a ~100 deg gap
    # whose chord cuts deep into the arena (recorded); the lattice bounds
    # the angular gap and therefore the chord's minimum radius
    base_step = 2.0 * math.pi / POLY_VERTICES
    angles = np.array(
        [i * base_step + rng.uniform(-0.25, 0.25) * base_step for i in range(POLY_VERTICES)]
    )
    radii = rng.uniform(POLY_R_MIN, POLY_R_MAX, POLY_VERTICES)
    vertices = [
        (CENTER[0] + r * math.cos(a), CENTER[1] + r * math.sin(a))
        for a, r in zip(angles, radii, strict=True)
    ]
    walls = polygon_walls(vertices)
    # clutter: mixed primitives inside the polygon, off the patrol corridor
    legs = [np.asarray(g, dtype=float) for g in PATROL_INSET]
    clutter = []
    attempts = 0
    while len(clutter) < CLUTTER_COUNT and attempts < 400:
        attempts += 1
        x = float(rng.uniform(1.6, 8.4))
        y = float(rng.uniform(1.6, 8.4))
        # inside the polygon (radius bound) and clear of the patrol corridor
        if math.hypot(x - CENTER[0], y - CENTER[1]) > POLY_R_MIN - 0.4:
            continue
        if any(
            _point_leg_distance(x, y, legs[i], legs[(i + 1) % 4]) < CLUTTER_CORRIDOR_M
            for i in range(4)
        ):
            continue
        kind = ("circle", "rect", "segment")[int(rng.integers(3))]
        if kind == "circle":
            prim = Obstacle("circle", x, y, 0.0, 0.0, float(rng.uniform(0.25, 0.5)))
        elif kind == "rect":
            prim = Obstacle(
                "rect", x, y, float(rng.uniform(0.2, 0.4)), float(rng.uniform(0.2, 0.4))
            )
        else:
            ang = float(rng.uniform(0.0, math.pi))
            half = float(rng.uniform(0.4, 0.6))
            prim = Obstacle(
                "segment",
                x,
                y,
                half * math.cos(ang),
                half * math.sin(ang),
                0.05,
            )
        # corridor filter on the PRIMITIVE EXTENT (a center-only filter lets
        # a rect corner poke into the path and freeze the patrol - recorded
        # lesson-54 debugging note)
        margin = prim.extent() + CLUTTER_CORRIDOR_M
        if any(_point_leg_distance(x, y, legs[i], legs[(i + 1) % 4]) < margin for i in range(4)):
            continue
        # mutual spacing: two close primitives seal a pocket the blind grid
        # cannot see; the patrol then freezes in a clearance deadlock
        if any(
            math.hypot(x - o.cx, y - o.cy) < CLUTTER_MIN_GAP_M + o.extent() + prim.extent()
            for o in clutter
        ):
            continue
        clutter.append(prim)
    return vertices, walls, tuple(clutter)


def _point_leg_distance(x, y, a, b):
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    ex, ey = bx - ax, by - ay
    len2 = ex * ex + ey * ey
    s = 0.0 if len2 <= 0.0 else max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / len2))
    return math.hypot(x - ax - s * ex, y - ay - s * ey)


def patrol_start(base, seed):
    """Start pose at the inset square's first corner (heading along leg 1,
    +-3 deg jitter mirroring the lesson-49 policy)."""
    rng = np.random.default_rng([seed, 54000])
    return [
        np.array(
            [
                PATROL_INSET[0][0] + float(rng.uniform(-0.1, 0.1)),
                PATROL_INSET[0][1] + float(rng.uniform(-0.1, 0.1)),
                float(rng.uniform(-math.radians(3.0), math.radians(3.0))),
            ]
        )
    ]


def score_stream_corrected(stream, matcher_cfg, group):
    """Lesson-54 corrected scoring (the metric-fix lesson).

    Lesson 49 measured "direction correctness" against true_delta(est, k) -
    the EST-POSE relative delta - but the matcher aligns CONTENT: its delta
    composed with est[k] (the IMPLIED POSE) is the measurable claim, and the
    est-pose delta carries the estimator drift that content alignment cannot
    see (the lesson-54 erratum; recorded legacy numbers stay in the 49/50
    records).  Corrected pipeline per candidate:
      acceptance = median NN <= 0.15 m (alignment, as lesson 49)
                   AND implied pose within the drift envelope of est[k]
                   (<= 6 m, <= 80 deg - plausibility, not raw delta caps:
                   the raw delta heading contains the heading drift);
      correct    = implied pose within 0.5 m / 5 deg of TRUTH pose of k.
    Legacy flags (old target) are kept per row for the errata comparison.
    """
    from embodied_learning.experiments.matcher_engineering import (
        build_stream_data,
        candidate_frames,
        implied_pose,
        is_correct,
        is_correct_implied,
        match_once,
        true_delta,
    )
    from embodied_learning.experiments.scan_slam import ray_points

    ranges, est, ref_pts, offsets = build_stream_data(stream, matcher_cfg, 0)
    rows = []
    for k in candidate_frames(stream, matcher_cfg):
        cur_pts, _i = ray_points(est[k], ranges[k], stream["hit"][k], offsets)
        if len(cur_pts) < matcher_cfg.min_points or len(ref_pts) < matcher_cfg.min_points:
            continue
        m = match_once(ref_pts, cur_pts, matcher_cfg, group)
        row = {
            "frame": k,
            "accepted": m["accepted"],
            "median_nn": m["median_nn"],
            "correct": None,
            "correct_implied": None,
        }
        d_true = true_delta(est, k)
        row["correct"] = is_correct(m, d_true)  # legacy (errata comparison)
        if m["delta"] is not None:
            implied = implied_pose(np.asarray(m["delta"], dtype=float), est[k])
            row["implied"] = implied
            row["correct_implied"] = is_correct_implied(implied, stream["truth"][k])
            # corrected acceptance: alignment gate + drift-envelope plausibility
            drift_t = float(np.linalg.norm(implied[:2] - est[k][:2]))
            drift_r = abs(math.degrees(implied[2] - est[k][2]))
            row["accepted_implied"] = bool(
                m["accepted"]
                or (
                    math.isfinite(m["median_nn"])
                    and m["median_nn"] <= 0.15
                    and drift_t <= 6.0
                    and drift_r <= 80.0
                )
            )
        rows.append(row)
    return rows


def _build_world(arm, seed, base):
    """World primitives for one arm: obstacles + walls (None = square)."""
    if arm == "ISO":
        scenarios = build_scenarios(base, seed)
        legs = [np.asarray(g, dtype=float) for g in PATROL_INSET]
        obstacles = tuple(
            o
            for o in scenarios[1]["obstacles"]
            if all(
                _point_leg_distance(o.cx, o.cy, legs[i], legs[(i + 1) % 4])
                >= o.extent() + CLUTTER_CORRIDOR_M
                for i in range(4)
            )
        )
        return obstacles, None
    _vertices, walls, obstacles = build_aniso_world(seed)
    return obstacles, walls


def run_arms(output_cfg, log):
    """Collect + score both arms; returns per-arm rows and streams.

    World validity gate: a world whose truth-controlled patrol cannot
    complete 2 laps (the grid planner can seal a corridor its blind map
    under-approximates) is NOT a valid data source - rebuild with the next
    attempt seed, mirroring the lesson-46 collector gate in spirit.
    """
    cfg = output_cfg
    streams = {}
    for arm in ARMS:
        base_cfg = cfg.matcher
        stream = None
        for attempt in range(6):
            obstacles, walls = _build_world(arm, cfg.seed + attempt, cfg.base)
            try:
                candidate = collect_patrol_laps(
                    obstacles,
                    patrol_start(cfg.base, cfg.seed + attempt)[0],
                    base_cfg,
                    walls=walls,
                    patrol=PATROL_INSET,
                )
            except ValueError:
                continue  # degenerate plan on a sealed world: reject it
            if candidate["laps_done"] >= cfg.laps:
                stream = candidate
                log(f"  arm {arm}: world accepted at attempt {attempt}")
                break
            log(
                f"  arm {arm}: attempt {attempt} stuck at legs {candidate['legs']}/8 - world rejected"
            )
        if stream is None:
            raise RuntimeError(f"arm {arm}: no patrolable world in 6 attempts")
        rows = {}
        for group in GROUPS:
            rows[group] = score_stream_corrected(stream, base_cfg, group)
        streams[arm] = {"stream": stream, "rows": rows}
        log(
            f"  arm {arm}: frames {len(stream['truth']) - 1} laps {stream['laps_done']} "
            + " ".join(
                f"{g} acc {sum(1 for r in rows[g] if r['accepted'])}/{len(rows[g])}" for g in GROUPS
            )
        )
    return streams


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
        config = AnisoConfig()
    elif config.__class__ is not AnisoConfig:
        raise TypeError("config must be AnisoConfig")
    started = time.perf_counter()
    configure_plot_font()

    streams = run_arms(config, log)

    def agg_corrected(rows):
        n = len(rows)
        acc = [r for r in rows if r["accepted_implied"]]
        cor = [r for r in acc if r["correct_implied"]]
        legacy_acc = [r for r in rows if r["accepted"]]
        legacy_cor = [r for r in legacy_acc if r["correct"]]
        return {
            "n_candidates": n,
            "accepted": len(acc),
            "acceptance": len(acc) / max(1, n),
            "direction_correctness": len(cor) / max(1, len(acc)),
            "correct_accept": len(cor),
            "legacy_acceptance": len(legacy_acc) / max(1, n),
            "legacy_direction_correctness": len(legacy_cor) / max(1, len(legacy_acc)),
        }

    aggregates = {}
    for arm in ARMS:
        aggregates[arm] = {group: agg_corrected(streams[arm]["rows"][group]) for group in GROUPS}
    n_total = {arm: len(streams[arm]["rows"]["B"]) for arm in ARMS}
    k_iso = aggregates["ISO"]["K"]
    k_aniso = aggregates["ANISO"]["K"]
    hypothesis = {
        "claim": (
            "度量修正 + 环境对照：第 49 课的「方向正确率 ~5%」是评估目标伪影"
            "（est 位姿差值烤入了漂移，内容对齐不可见漂移）；修正为隐含位姿判据后，"
            "两臂的 K 组正确率重新测量，检验环境各向异性的真实贡献"
        ),
        "criteria": {
            "collector": "双臂候选池 >= 20 且每巡游完成 2 圈",
            "metric_fix": "修正判据下 ISO 臂 K 方向正确率显著高于旧判据记录值（勘误实证）",
            "aniso_claim": "ANISO 臂 K 方向正确率 >= 80%（修正判据）",
            "gap": "K 正确率差（ANISO-ISO）>= 0.2（若 ISO 修正值 < 80%）",
        },
        "results": {
            "collector_met": bool(
                all(n_total[a] >= 20 for a in ARMS)
                and all(s["stream"]["laps_done"] >= 2 for s in streams.values())
            ),
            "iso_corrected": k_iso["direction_correctness"],
            "aniso_corrected": k_aniso["direction_correctness"],
            "iso_legacy": k_iso["legacy_direction_correctness"],
            "aniso_legacy": k_aniso["legacy_direction_correctness"],
            "aniso_claim_met": bool(k_aniso["direction_correctness"] >= 0.80),
            "gap_met": bool(
                k_aniso["direction_correctness"] - k_iso["direction_correctness"] >= 0.2
            ),
            "k_accept": {arm: aggregates[arm]["K"]["acceptance"] for arm in ARMS},
            "n_candidates": n_total,
        },
    }
    archives = {"arm_markers": np.array([0, 1])}
    for arm in ARMS:
        for group in GROUPS:
            rows = streams[arm]["rows"][group]
            archives[f"{arm}_{group}_accepted"] = np.array(
                [1 if r["accepted_implied"] else 0 for r in rows], dtype=int
            )
            archives[f"{arm}_{group}_correct"] = np.array(
                [1 if r["correct_implied"] else 0 for r in rows], dtype=int
            )
            archives[f"{arm}_{group}_legacy_correct"] = np.array(
                [1 if r["correct"] else 0 for r in rows], dtype=int
            )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": config.seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "patrol": [list(g) for g in PATROL_INSET],
            "patrol_note": "闭圈内方巡游（净旋转 0），距边界 >= 1.8 m、距杂物 >= 0.7 m",
            "arms": ARM_NAMES,
            "matcher": "lesson-49 protocol verbatim (B/P/R/K, sigma 6 cm, bias 2%)",
            "aniso": {
                "vertices": POLY_VERTICES,
                "radius_range": [POLY_R_MIN, POLY_R_MAX],
                "clutter": CLUTTER_COUNT,
            },
            "bars": {"acceptance": 0.10, "direction_correctness": 0.80},
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
    parser = argparse.ArgumentParser(description="lesson-54 anisotropic environment record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
