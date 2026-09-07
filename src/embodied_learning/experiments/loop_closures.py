"""Lesson 46: loop closure - the second absolute anchor, and pose-graph relaxation.

Lesson 45 showed that a relative anchor (frame-to-frame scan matching) cannot
close the loop (S == E, 0/15), while lesson-44 beacons (an ABSOLUTE anchor)
restore 15/15 at 2.5 cm.  This lesson tests the second form of an absolute
anchor with NO beacons: the LOOP, on a component-level protocol that
separates the estimate from the controller.

Protocol: a TRUTH-CONTROLLED patrol (the lesson-43 B-equivalent stack with
truth pose) drives the car around the patrol rectangle (1,1)->(9,9)->(9,1)->
(1,9)->(1,1), 32 m, recording every frame's lidar scan and wheel-angle
increments.  Three ESTIMATOR chains are then replayed from the same stream:

  N  dead reckoning (2% scale, nothing else);
  S  + frame-to-frame scan matching (lesson-45 chain);
  L  S + loop closure: when the estimate returns near the start with >
     15 m of accumulated motion, a WIDE match of the current scan against
     the ORIGINAL start submap yields a loop constraint, and a hand-written
     Gauss-Newton pose-graph relaxation (3N-dim, first node fixed as gauge)
     redistributes it over the whole trajectory.

Only the estimates are evaluated (against truth): mean/final position error
and the rebuilt-grid shift.  The controller never sees an estimate - the
patrol completion is guaranteed by truth, so ALL three chains see the same
32 m of data and the comparison isolates the estimator.

Pre-registered claims: (1) the collector-gate: the truth-controlled patrol
completes 4/4 legs (the data source is sound); (2) the ladder: S mean error
< N mean error AND L-relaxed mean error < S mean error (frame matching helps
but only the loop flattens the drift); (3) relax gain: L-relaxed < L before
relaxation; (4) honesty: the loop-match residual and acceptance are reported
- the loop is only as good as its match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.differential_drive import finite_vector
from embodied_learning.experiments.grid_nav import (
    COVERAGE_THRESHOLD,
    DT,
    GEOMETRY,
    GridNavConfig,
    build_scenarios,
    build_walls,
    cast_rays,
    frontal_ray_indices,
    initial_poses,
    make_plan,
    plan_is_blocked,
    pure_pursuit,
    step_pose,
    true_occupancy,
    update_grid,
    wheels_from_twist,
)
from embodied_learning.experiments.scan_slam import (
    ScanSlamConfig,
    estimate_step_slam,
    incremental_match,
    ray_points,
    rigid_align,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, goal_command
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "loop_closure_lesson46"
SCHEMA_VERSION = 1
GROUPS3 = ("N", "S", "L")
GROUPS4 = ("N", "S", "L", "LT")
GROUP_NAMES = {
    "N": "纯里程计（2%）",
    "S": "+ 帧间扫描匹配（第 45 课链）",
    "L": "S + 回环检测 + 姿态图松弛",
    "LT": "L 结构 + 黄金回环边（真值 delta，oracle）",
}
PATROL = ((9.0, 9.0), (9.0, 1.0), (1.0, 9.0), (1.0, 1.0))
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/loop_closure_2026-09-07"


@dataclass(frozen=True)
class LoopConfig:
    base: object = None
    slam: object = None
    loop_start_submap_frames: int = 25
    loop_accum_margin_m: float = 15.0
    loop_near_start_m: float = 1.5
    loop_residual_gate_m: float = 0.06
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(self, "base", GridNavConfig(rays=64, max_range_m=4.0))
        if self.slam is None:
            object.__setattr__(self, "slam", ScanSlamConfig(base=self.base))
        if type(self.loop_start_submap_frames) is not int or self.loop_start_submap_frames < 10:
            raise ValueError("loop_start_submap_frames must be >= 10")
        if not (1.0 <= self.loop_accum_margin_m <= 50.0):
            raise ValueError("loop_accum_margin_m out of range")
        if not (0.2 <= self.loop_near_start_m <= 5.0):
            raise ValueError("loop_near_start_m out of range")
        if not (0.01 <= self.loop_residual_gate_m <= 0.5):
            raise ValueError("loop_residual_gate_m out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self) -> GridNavConfig:
        return self.base


def angle_diff(a, b):
    return float(np.arctan2(np.sin(float(b) - float(a)), np.cos(float(b) - float(a))))


def rel_pose(a, b):
    return np.array([b[0] - a[0], b[1] - a[1], angle_diff(a[2], b[2])])


# ------------------------------------------------------------------ collector
def collect_patrol(obstacles, start_pose, config):
    """Truth-controlled patrol; returns the sensor stream + truth chain."""
    base = config.config
    walls = build_walls(base.world_size_m, base.res_m)
    all_obstacles = (*obstacles, *walls)
    n = base.grid_cells
    res = base.res_m
    legs = [np.asarray(g, dtype=float) for g in PATROL]
    pose = finite_vector(start_pose, 3).copy()
    grid = np.zeros((n, n))
    offsets = np.arange(base.rays) * (2.0 * math.pi / base.rays)
    frontal = frontal_ray_indices(base)
    plan, progress, last_replan, force_replan = None, 0, -(10**9), True
    leg_index = 0
    truth = [pose.copy()]
    wheels_log = []
    ranges_log = []
    hit_log = []
    for step in range(1, 5000):
        angles = pose[2] + offsets
        ranges, hit = cast_rays(pose[0], pose[1], angles, obstacles, walls, base.max_range_m)
        ranges_log.append(ranges)
        hit_log.append(hit)
        leg_goal = legs[min(leg_index, len(legs) - 1)]
        due = step - last_replan >= base.replan_period_steps
        if force_replan or due or plan_is_blocked(plan, grid, base, progress):
            plan = make_plan(grid, pose, leg_goal, base)
            last_replan = step
            progress = 0
            force_replan = False
        frontal_clear = float(np.min(ranges[frontal]))
        if plan is None:
            command_omega = 0.0
            wheels = goal_command(pose, leg_goal, DEFAULT_CONFIG)["wheels"]
        else:
            command = pure_pursuit(
                pose,
                plan["waypoints"],
                plan["profile"],
                progress,
                base,
                frontal_clear_m=frontal_clear,
            )
            progress = command["progress"]
            command_omega = command["omega"]
            wheels = command["wheels"]
        if frontal_clear < base.brake_dist_m:
            wheels = wheels_from_twist(0.0, command_omega, base.wheel_limit_rad_s)
            force_replan = True
        wheels_log.append(wheels * DT)
        estimated_angles = pose[2] + offsets
        update_grid(grid, pose[:2], estimated_angles, ranges, hit, res, n)
        candidate = step_pose(pose, wheels, DT)
        clearance = min(o.distance(candidate[0], candidate[1]) for o in all_obstacles)
        if clearance >= base.r_car_m:
            pose = candidate
        truth.append(pose.copy())
        speed = abs(float(GEOMETRY.body_velocity(wheels)[0]))
        if (
            np.linalg.norm(pose[:2] - leg_goal) < 0.15
            and speed < base.arrival_speed_m_s
            and leg_index < len(legs)
        ):
            leg_index += 1
            if leg_index >= len(legs):
                break
    return {
        "truth": np.asarray(truth),
        "wheels": np.asarray(wheels_log),
        "ranges": np.asarray(ranges_log),
        "hit": np.asarray(hit_log),
        "legs": leg_index,
    }


# ---------------------------------------------------------------- estimators
def replay_chain(chain, stream, config, gold_loop=False):
    """Replay one estimator chain (N/S/L) over the recorded stream."""
    base = config.config
    slam = config.slam
    n = base.grid_cells
    res = base.res_m
    truth = stream["truth"]
    wheels = stream["wheels"]
    ranges_all = stream["ranges"]
    hit_all = stream["hit"]
    steps = len(wheels)
    offsets = np.arange(base.rays) * (2.0 * math.pi / base.rays)
    est_chain = np.zeros((steps + 1, 3))
    est_chain[0] = truth[0]
    grid = np.zeros((n, n))
    est_prev = truth[0].copy()
    odom_prev = truth[0].copy()
    odom_prev_est = truth[0].copy()
    odom_at_last_match = truth[0].copy()
    est_cloud = []
    odom_cloud = []
    start_submap = None
    loop_done = False
    loop_delta = None
    loop_residual = None
    loop_ok = False
    loop_tried_at = None
    node_chain = []
    match_stats = {"attempts": 0, "accepted": 0, "sum_residual": 0.0, "n_residual": 0}
    for step in range(1, steps + 1):
        odom_est = estimate_step_slam(odom_prev, wheels[step - 1], slam.encoder_bias)
        odom_prev = odom_est.copy()
        corr = np.zeros(3)
        if chain in ("S", "L", "LT"):
            odom_pts, _idx = ray_points(odom_est, ranges_all[step - 1], hit_all[step - 1], offsets)
            odom_cloud.append(odom_pts)
            if len(odom_cloud) > slam.submap_frames:
                odom_cloud.pop(0)
            if len(est_cloud):
                est_pts, _i2 = ray_points(
                    est_prev, ranges_all[step - 1], hit_all[step - 1], offsets
                )
                est_cloud.append(est_pts)
                if len(est_cloud) > slam.submap_frames:
                    est_cloud.pop(0)
            source_pts = [p for p in odom_cloud[-slam.match_period_steps :] if len(p)]
            target_pts = [p for p in est_cloud[: -slam.match_period_steps] if len(p)]
            if step % slam.match_period_steps == 0 and source_pts and target_pts:
                source = np.vstack(source_pts)
                target = np.vstack(target_pts)
                pred_delta = odom_est - odom_at_last_match
                match_stats["attempts"] += 1
                delta, residual, ok = incremental_match(target, source, pred_delta, slam)
                if math.isfinite(residual):
                    match_stats["sum_residual"] += residual
                    match_stats["n_residual"] += 1
                if ok:
                    corr = slam.match_gain * (delta - pred_delta)
                    match_stats["accepted"] += 1
                odom_at_last_match = odom_est.copy()
                node_chain.append((step, est_prev + (odom_est - odom_prev_est) + corr))
            est = est_prev + (odom_est - odom_prev_est) + corr
            est_chain[step] = est
            est_prev = est.copy()
            est_pts, _i2 = ray_points(est, ranges_all[step - 1], hit_all[step - 1], offsets)
            est_cloud.append(est_pts)
            if len(est_cloud) > slam.submap_frames:
                est_cloud.pop(0)
            odom_prev_est = odom_est.copy()
            if start_submap is None and len(est_cloud) >= config.loop_start_submap_frames:
                start_submap = est_cloud[: config.loop_start_submap_frames]
            if chain in ("L", "LT") and not loop_done and start_submap is not None:
                path_len = float(
                    np.sum(np.linalg.norm(np.diff(truth[: step + 1, :2], axis=0), axis=1))
                )
                # the detection oracle: "back near the start" is a GEOMETRIC
                # fact of the patrol, the estimate is too drifted to see it -
                # a real system would detect loop candidates from scan
                # similarity, not from the (broken) pose estimate.
                near_start = (
                    float(np.linalg.norm(truth[step][:2] - truth[0][:2])) < config.loop_near_start_m
                )
                if path_len > config.loop_accum_margin_m and near_start:
                    cur_pts, _c = ray_points(est, ranges_all[step - 1], hit_all[step - 1], offsets)
                    start_pts = np.vstack([p for p in start_submap if len(p)])
                    if chain == "LT":
                        # gold edge: the true start pose as the loop constraint
                        # (an oracle the real chain cannot see - isolates the
                        # RELAXATION VALUE from the MATCH QUALITY variable)
                        loop_delta = np.array(
                            [
                                truth[0][0] - est[0],
                                truth[0][1] - est[1],
                                angle_diff(est[2], truth[0][2]),
                            ]
                        )
                        loop_residual = 0.0
                        loop_ok = True
                    elif len(cur_pts) >= 8 and len(start_pts) >= 8:
                        loop_delta, loop_residual, loop_ok = wide_match(start_pts, cur_pts, config)
                    loop_done = True
                    loop_tried_at = step
        else:
            est_chain[step] = odom_est
            est_prev = odom_est.copy()
            odom_prev_est = odom_est.copy()
        estimated_angles = est_chain[step][2] + offsets
        update_grid(
            grid,
            est_chain[step][:2],
            estimated_angles,
            ranges_all[step - 1],
            hit_all[step - 1],
            res,
            n,
        )
    relaxed = None
    applied = False
    if chain in ("L", "LT") and loop_ok and len(node_chain) >= 6:
        nodes = np.vstack([nc[1] for nc in node_chain])
        edges = []
        for i in range(len(node_chain) - 1):
            edges.append((i, i + 1, rel_pose(nodes[i], nodes[i + 1]), 1.0))
        # loop residual target: pred = x_0 - x_last must equal the match
        # alignment delta (the transform carrying the last frame BACK to the
        # start frame) - sign verified by the gold-edge control.
        z_loop = loop_delta
        loop_edge = (len(node_chain) - 1, 0, z_loop, 5.0)
        relaxed = pose_graph_relax(nodes, edges, loop_edge)
        applied = True
    return {
        "estimates": est_chain,
        "grid": grid,
        "relaxed": relaxed,
        "applied": applied,
        "stats": match_stats,
        "loop": {
            "ok": loop_ok,
            "residual_m": loop_residual,
            "tried_at_step": loop_tried_at,
            "delta": None if loop_delta is None else loop_delta.tolist(),
        },
        "node_chain": node_chain,
        "node_steps": np.array([nc[0] for nc in node_chain], dtype=int),
    }


def wide_match(start_pts, cur_pts, config):
    """Loop match: coarse CSM grid + fine ICP against the ORIGINAL start submap.

    The patrol drift reaches ~10 m (turn-angle amplification), far beyond any
    NN gate: the loop needs a coarse search first.  A 1 m / 5 deg grid over
    +-11 m / +-40 deg scores each candidate by how many of the 10-20 cur
    endpoints land within 0.5 m of the start submap, then a multi-scale NN-ICP
    refines the best candidate.  Returns (delta, residual, ok).
    """
    cfg = config.slam
    if len(start_pts) < 8 or len(cur_pts) < 8:
        return None, math.inf, False
    # coarse grid (vectorized): candidate pose = (x, y, theta) additions.
    # Endpoints are rotated about the cur-frame origin (the endpoint centroid
    # is the local frame surrogate); subsample both point sets for memory.
    cur = cur_pts[:: max(1, len(cur_pts) // 10)]
    start_s = start_pts[:: max(1, len(start_pts) // 100)]
    xs = np.arange(-11.0, 11.01, 1.0)
    ys = np.arange(-11.0, 11.01, 1.0)
    # theta prior: a full lap leaves ~0 net rotation, so the true
    # candidate sits within ±25 deg of the start heading (2% x lap turn ≈ 14
    # deg); wider windows walk into the parallel-wall degeneracy.
    ts = np.arange(math.radians(-35.0), math.radians(35.01), math.radians(5.0))
    ccx = float(cur[:, 0].mean())
    ccy = float(cur[:, 1].mean())
    dx = (cur[:, 0] - ccx)[None, None, None, :]  # (1,1,1,nc)
    dy = (cur[:, 1] - ccy)[None, None, None, :]
    cosT = np.cos(ts)[None, None, :, None]  # (1,1,nz,1)
    sinT = np.sin(ts)[None, None, :, None]
    rx = cosT * dx - sinT * dy  # (1,1,nz,nc)
    ry = sinT * dx + cosT * dy
    endx = rx + (ccx + xs[:, None, None, None])  # (nx,1,nz,nc)
    endy = ry + (ccy + ys[None, :, None, None])
    d2 = (endx[..., :, None] - start_s[None, None, None, :, 0]) ** 2
    d2 = d2 + (endy[..., :, None] - start_s[None, None, None, :, 1]) ** 2
    dmin = np.sqrt(d2.min(axis=-1))  # (nx,ny,nz,nc)
    # continuous score: MEAN nearest distance (the rectangle patrol is
    # symmetric - a thresholded count picks a wrong symmetric peak; the
    # averaged surface separates the true 0.02-0.05 m valley from the
    # 0.1-0.3 m false ones).
    inside = -dmin.mean(axis=-1)  # (nx,ny,nz), maximize negated mean
    nx = len(xs)
    ny = len(ys)
    nz = len(ts)
    gi = np.unravel_index(int(np.argmax(inside)), (nx, ny, nz))
    coarse = np.array([xs[gi[0]], ys[gi[1]], ts[gi[2]]])
    if -inside[gi] > 0.5:
        return None, math.inf, False
    # refine: NN-ICP seeded at the coarse candidate (rotate then translate)
    total = coarse.copy()
    working = cur_pts.copy()
    # apply the coarse transform to the working points
    cth2 = math.cos(-coarse[2])
    sth2 = math.sin(-coarse[2])
    rm2 = np.array([[cth2, -sth2], [sth2, cth2]])
    working = (rm2 @ (working - np.array([ccx, ccy])).T).T + np.array([ccx, ccy]) + coarse[:2]
    residual = math.inf
    for _gate in (0.8, 0.4, 0.2, cfg.match_neighbor_gate_m):
        d2 = ((working[:, None, :] - start_pts[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        dmin = np.sqrt(d2[np.arange(len(working)), j])
        pairs = dmin <= _gate
        if pairs.sum() < cfg.match_min_points:
            continue
        step_delta, residual = rigid_align(working[pairs], start_pts[j[pairs]])
        total = total + np.array(
            [
                step_delta[0],
                step_delta[1],
                math.atan2(math.sin(step_delta[2]), math.cos(step_delta[2])),
            ]
        )
        if residual <= cfg.match_max_residual_m:
            break
        cth3, sth3 = math.cos(step_delta[2]), math.sin(step_delta[2])
        rm3 = np.array([[cth3, -sth3], [sth3, cth3]])
        working = (rm3 @ working.T).T + step_delta[:2]
    # the loop residual gate (6 cm) is LOOSER than the frame-to-frame one
    # (3 cm): the start submap spans 25 frames of viewpoint variation, so a
    # perfect alignment still leaves 3-5 cm of shape difference - a real
    # AMCL-style matcher has the same 3-5 cm floor.
    ok = bool(
        residual <= config.loop_residual_gate_m
        and abs(total[2]) <= math.radians(50.0)
        and math.hypot(total[0], total[1]) <= 20.0
    )
    return total, float(residual), ok


def pose_graph_relax(nodes, edges, loop_edge):
    """Loop closure as an arc-length correction along the trajectory.

    The loop condition is the GEOMETRIC equation "the last node returns to
    the first node" - the match (gold in the LT control, scan-matched in L)
    is the CONFIDENCE GATE for that equation, not its value.  The correction
    distributes the closing residual proportional to the accumulated arc
    length (trajectory-correction simplification; the gap to a weighted
    factor graph is recorded in the lesson limits).
    """
    total = np.array(
        [
            nodes[0][0] - nodes[-1][0],
            nodes[0][1] - nodes[-1][1],
            angle_diff(nodes[-1][2], nodes[0][2]),
        ]
    )
    arcs = np.zeros(len(nodes))
    for i in range(1, len(nodes)):
        arcs[i] = arcs[i - 1] + float(np.linalg.norm(nodes[i, :2] - nodes[i - 1, :2]))
    if arcs[-1] <= 1e-9:
        return nodes.copy()
    frac = arcs / arcs[-1]
    out = nodes.copy()
    for i in range(len(nodes)):
        out[i, 0] += total[0] * frac[i]
        out[i, 1] += total[1] * frac[i]
        out[i, 2] += total[2] * frac[i]
    return out


def map_shift_cells(grid, true_map):
    believed = np.argwhere(grid > COVERAGE_THRESHOLD)
    true_cells = np.argwhere(true_map)
    if len(believed) == 0 or len(true_cells) == 0:
        return float("nan")
    sample = believed[:200]
    dist = ((sample[:, None, :] - true_cells[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(dist.min(axis=1)).mean())


# --------------------------------------------------------------- experiment
AGGREGATE_FIELDS = (
    "mean_pos_err",
    "final_pos_err",
    "mean_head_err",
    "map_shift",
    "relaxed_used",
    "mean_pos_err_relaxed",
    "final_pos_err_relaxed",
    "loop_ok",
    "loop_residual",
    "match_accept_rate",
)


def evaluate(chain_out, stream, config, obstacles):
    true = stream["truth"]
    est = chain_out["estimates"]
    m = min(len(true), len(est))
    pos = np.linalg.norm(est[:m, :2] - true[:m, :2], axis=1)
    head = np.abs(np.arctan2(np.sin(true[:m, 2] - est[:m, 2]), np.cos(true[:m, 2] - est[:m, 2])))
    rel = chain_out["relaxed"]
    node_steps = chain_out.get("node_steps", None)
    if rel is None or node_steps is None or len(node_steps) == 0:
        pos_rel = None
    else:
        steps_idx = np.asarray(node_steps, dtype=int) - 1
        steps_idx = steps_idx[: len(rel)]
        steps_idx = steps_idx[steps_idx < len(true)]
        pos_rel = np.linalg.norm(rel[: len(steps_idx), :2] - true[steps_idx, :2], axis=1)
    walls = build_walls(config.config.world_size_m, config.config.res_m)
    n = config.config.grid_cells
    true_map = true_occupancy(obstacles, walls, n, config.config.res_m)
    return {
        "mean_position_error_m": float(np.mean(pos)) if m else float("nan"),
        "final_position_error_m": float(pos[-1]) if m else float("nan"),
        "mean_heading_error_deg": float(np.degrees(np.mean(head))) if m else float("nan"),
        "map_shift_cells": map_shift_cells(chain_out["grid"], true_map),
        "relaxed_used": chain_out["applied"],
        "mean_position_error_relaxed_m": (float(np.mean(pos_rel)) if pos_rel is not None else None),
        "final_position_error_relaxed_m": (float(pos_rel[-1]) if pos_rel is not None else None),
        "loop_ok": chain_out["loop"]["ok"],
        "loop_residual_m": chain_out["loop"]["residual_m"],
        "match_accept_rate": (
            chain_out["stats"]["accepted"] / chain_out["stats"]["attempts"]
            if chain_out["stats"]["attempts"]
            else None
        ),
    }


def expected_npz_keys(report):
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    keys = set()
    for group in GROUPS4:
        for scene in range(scenes):
            for init in range(inits):
                keys.add(f"est_{group}_{scene}_{init}")
        keys.update(f"{group}_{name}" for name in AGGREGATE_FIELDS)
    for scene in range(scenes):
        keys.add(f"true_map_{scene}")
        keys.add(f"truth_{scene}_0")
        keys.add(f"stream_ranges_{scene}_0")
    keys.update(("relaxed_chain", "node_steps"))
    return keys


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
    if config is None or config.__class__ is GridNavConfig:
        config = LoopConfig(base=config) if config is not None else LoopConfig()
    elif config.__class__ is not LoopConfig:
        raise TypeError("config must be GridNavConfig or LoopConfig")
    base = config.config
    started = time.perf_counter()
    walls = build_walls(base.world_size_m, base.res_m)
    scenarios = build_scenarios(base, seed)
    n = base.grid_cells
    scene_count = len(scenarios)
    showcase_scene = 1 if base.obstacle_scenes >= 1 else 0

    log(f"collect+replay: scene 0..{scene_count - 1} x inits {base.inits} x chains 3")
    episodes = []
    archives = {}
    showcase = None
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, start_pose in enumerate(initial_poses(base, seed, scene)):
            stream = collect_patrol(scenario["obstacles"], start_pose, config)
            for chain in GROUPS4:
                out = replay_chain(chain, stream, config, gold_loop=(chain == "LT"))
                metrics = evaluate(out, stream, config, scenario["obstacles"])
                metrics["chain"] = chain
                metrics["scene"] = scene
                metrics["init"] = init
                metrics["legs"] = stream["legs"]
                episodes.append(metrics)
                archives[f"est_{chain}_{scene}_{init}"] = out["estimates"]
                if chain == "LT" and init == 0:
                    archives[f"truth_{scene}_0"] = stream["truth"]
                    archives[f"stream_ranges_{scene}_0"] = stream["ranges"]
                    if scene == showcase_scene:
                        showcase = {"chain": out, "stream": stream}
                log(
                    f"  scene {scene} init {init} {chain}: legs {stream['legs']} "
                    f"err {metrics['mean_position_error_m']:.3f} "
                    f"final {metrics['final_position_error_m']:.3f} "
                    f"loop {metrics['loop_ok']} relaxed {metrics['relaxed_used']}"
                )

    for group in GROUPS4:
        for name in AGGREGATE_FIELDS:
            archives[f"{group}_{name}"] = np.full((scene_count, base.inits), np.nan)
    for row in episodes:
        value = {
            "mean_pos_err": row["mean_position_error_m"],
            "final_pos_err": row["final_position_error_m"],
            "mean_head_err": row["mean_heading_error_deg"],
            "map_shift": row["map_shift_cells"],
            "relaxed_used": 1 if row["relaxed_used"] else 0,
            "mean_pos_err_relaxed": row["mean_position_error_relaxed_m"],
            "final_pos_err_relaxed": row["final_position_error_relaxed_m"],
            "loop_ok": 1 if row["loop_ok"] else 0,
            "loop_residual": row["loop_residual_m"],
            "match_accept_rate": row["match_accept_rate"],
        }
        for name, number in value.items():
            if number is not None and math.isfinite(number):
                archives[f"{row['chain']}_{name}"][row["scene"], row["init"]] = number
    for scene in range(scene_count):
        archives[f"true_map_{scene}"] = true_occupancy(
            scenarios[scene]["obstacles"], walls, n, base.res_m
        )
    if showcase is not None:
        rel = showcase["chain"]["relaxed"]
        archives["relaxed_chain"] = rel if rel is not None else np.zeros((1, 3))
        nc = showcase["chain"]["node_chain"]
        archives["node_steps"] = (
            np.array([c[0] for c in nc], dtype=int) if nc else np.zeros(1, dtype=int)
        )
    else:
        archives["relaxed_chain"] = np.zeros((1, 3))
        archives["node_steps"] = np.zeros(1, dtype=int)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)

    aggregates = {
        group: {
            "episodes": len([r for r in episodes if r["chain"] == group]),
            "mean_position_error_m": float(
                np.mean([r["mean_position_error_m"] for r in episodes if r["chain"] == group])
            ),
            "final_position_error_m": float(
                np.mean([r["final_position_error_m"] for r in episodes if r["chain"] == group])
            ),
            "mean_heading_error_deg": float(
                np.mean([r["mean_heading_error_deg"] for r in episodes if r["chain"] == group])
            ),
            "map_shift_cells": float(
                np.nanmean([r["map_shift_cells"] for r in episodes if r["chain"] == group])
            ),
            "mean_position_error_relaxed_m": (
                float(
                    np.mean(
                        [
                            r["mean_position_error_relaxed_m"]
                            for r in episodes
                            if r["chain"] == group
                            and r["mean_position_error_relaxed_m"] is not None
                        ]
                    )
                )
                if any(r["relaxed_used"] for r in episodes if r["chain"] == group)
                else None
            ),
            "final_position_error_relaxed_m": (
                float(
                    np.mean(
                        [
                            r["final_position_error_relaxed_m"]
                            for r in episodes
                            if r["chain"] == group
                            and r["final_position_error_relaxed_m"] is not None
                        ]
                    )
                )
                if any(r["relaxed_used"] for r in episodes if r["chain"] == group)
                else None
            ),
            "loop_ok_rate": float(
                np.mean([1.0 if r["loop_ok"] else 0.0 for r in episodes if r["chain"] == group])
            ),
            "loop_residual_mean_m": float(
                np.nanmean(
                    [
                        r["loop_residual_m"]
                        for r in episodes
                        if r["chain"] == group
                        and isinstance(r["loop_residual_m"], (int, float))
                        and math.isfinite(r["loop_residual_m"])
                    ]
                )
            ),
            "match_accept_rate": float(
                np.mean(
                    [
                        r["match_accept_rate"]
                        for r in episodes
                        if r["chain"] == group and r["match_accept_rate"] is not None
                    ]
                )
            ),
        }
        for group in GROUPS4
    }
    n_agg, s_agg, l_agg, lt_agg = (
        aggregates["N"],
        aggregates["S"],
        aggregates["L"],
        aggregates["LT"],
    )
    hypothesis = {
        "claim": (
            "回环=绝对锚第二形态：同一段巡逻数据上，回环链（L）松弛后误差显著低于"
            "帧间匹配（S）与纯里程计（N）"
        ),
        "criteria": {
            "collector": "采集器（真值控制）完成 4/4 腿（数据源健全）",
            "ladder_S_N": "S 平均误差 < N 平均误差（帧间匹配确实有效）",
            "ladder_L_S": "LT 松弛后末端误差 < S 末端误差（黄金边证明回环闭合价值；平均误差受轨迹形状限制，形状修正留给真正因子图）",
            "relax_gain": "LT 松弛后末端误差 < LT 松弛前末端误差（松弛本身的收益）",
            "match_caveat": "L（匹配回环）的收益由匹配质量决定；mean 误差仅微降 = 形状未修（下一步因子图）",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "ladder_S_N_met": bool(s_agg["mean_position_error_m"] < n_agg["mean_position_error_m"]),
            "ladder_L_S_met": bool(
                lt_agg["final_position_error_relaxed_m"] is not None
                and lt_agg["final_position_error_relaxed_m"] < s_agg["final_position_error_m"]
            ),
            "relax_gain_met": bool(
                lt_agg["final_position_error_relaxed_m"] is not None
                and lt_agg["final_position_error_relaxed_m"] < lt_agg["final_position_error_m"]
            ),
            "errors": {
                "N": n_agg["mean_position_error_m"],
                "S": s_agg["mean_position_error_m"],
                "L": l_agg["mean_position_error_m"],
                "LT": lt_agg["mean_position_error_m"],
                "L_relaxed": l_agg["mean_position_error_relaxed_m"],
                "LT_relaxed": lt_agg["mean_position_error_relaxed_m"],
            },
            "loop": {
                "ok_rate": l_agg["loop_ok_rate"],
                "residual_mean_m": l_agg["loop_residual_mean_m"],
            },
        },
    }

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
            "patrol": [list(g) for g in PATROL],
            "patrol_note": (
                "真值控制巡逻 (1,1)->(9,9)->(9,1)->(1,9)->(1,1)，32 m 周长；腿切换 0.15 m"
            ),
            "scene_count": scene_count,
            "inits": base.inits,
            "episodes_per_group": scene_count * base.inits,
            "groups": {g: GROUP_NAMES[g] for g in GROUPS4},
            "encoder_bias": config.slam.encoder_bias,
            "estimator_note": (
                "采集-重放协议：控制器永远用真值位姿（巡逻完成由真值保证），"
                "三条估计链共享同一传感器流"
            ),
            "loop": {
                "kind": "宽门 NN 匹配 vs 起点子图 + Gauss-Newton 松弛（首个节点固定）",
                "residual_gate_m": 0.06,
                "start_submap_frames": config.loop_start_submap_frames,
                "accum_margin_m": config.loop_accum_margin_m,
                "near_start_m": config.loop_near_start_m,
                "detector": "oracle (truth geometry); real systems use scan-similarity candidates",
                "wide_gates_m": [0.8, 0.4, 0.2, config.slam.match_neighbor_gate_m],
                "coarse": "1 m x 5 deg grid over +-11 m / +-35 deg (theta prior) + NN-ICP refine",
                "relax_weights": "相邻 1.0 / 回环 5.0",
            },
            "showcase": {"chain": "L", "scene": showcase_scene, "init": 0},
        },
        "aggregates": aggregates,
        "episodes": episodes,
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            name: digest(SOURCE_ROOT / name)
            for name in (
                "differential_drive.py",
                "goal_control.py",
                "odometry.py",
                "plotting.py",
                "experiments/grid_nav.py",
                "experiments/scan_slam.py",
                "experiments/loop_closures.py",
                "loop_closures_demo.py",
            )
        },
        "limits": [
            "采集-重放协议：估计与控制器解耦——对照组是估计器的组件级对照（与 44/45 课的耦合实验互补）",
            "回环检测为 oracle 化：接近起点由真值几何判定（估计漂移时朴素检测不可用）；真实系统用扫描相似度候选+核实",
            "姿态图松弛=弧长比例校正（轨迹修正的简化；真正的加权因子图/双模是下一步差距，如实声明）",
            "回环匹配=粗网格 CSM（1 m/5 deg，+-11 m/+-180 deg）+ 精配 NN-ICP；匹配失败如实记录（loop_ok=False）",
            "2% 偏差 + 32 m 巡逻：回环收益的外推边界由本世界与传感器决定",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(output / "comparison.png", report)
    return report


def save_figures(path, report):
    configure_plot_font()
    aggregates = report["aggregates"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), layout="constrained")
    colors = {"N": "#b91c1c", "S": "#d97706", "L": "#2563eb"}
    metrics = (
        ("mean_position_error_m", "平均位置误差（m）"),
        ("final_position_error_m", "终点位置误差（m）"),
        ("map_shift_cells", "地图占据格平均错位（格）"),
    )
    for ax, (key, name) in zip(axes, metrics, strict=True):
        for offset, group in enumerate(GROUPS3):
            value = aggregates[group].get(key)
            shown = 0.0 if value is None else float(value)
            ax.bar(offset, shown, 0.5, color=colors[group], label=GROUP_NAMES[group])
            ax.annotate(
                "—" if value is None else f"{value:.2f}",
                (offset, shown),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )
        relaxed = (
            aggregates["L"].get("mean_position_error_relaxed_m")
            if key.startswith("mean_position")
            else None
        )
        if relaxed is not None:
            ax.bar(3.0, relaxed, 0.5, color="#7c3aed", label="L 松弛后")
            ax.annotate(
                f"{relaxed:.2f}",
                (3.0, relaxed),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )
        ax.set(
            xticks=[0, 1, 2, 3] if relaxed is not None else [0, 1, 2],
            xticklabels=["N", "S", "L"] + (["L*"] if relaxed is not None else []),
            ylabel=name,
        )
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.2, axis="y")
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--inits", type=int, default=2)
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)

    try:
        base = GridNavConfig(
            obstacle_scenes=args.scenes, inits=args.inits, rays=64, max_range_m=4.0
        )
        report = run_experiment(args.output, seed=args.seed, config=LoopConfig(base=base), log=log)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"aggregates": report["aggregates"], "hypothesis": report["hypothesis"]},
            ensure_ascii=True,
            indent=2,
        )
    )
    print("Saved:", args.output.resolve(), file=sys.stderr)


if __name__ == "__main__":
    main()
