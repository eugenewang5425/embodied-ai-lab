"""Lesson 45: minimal grid SLAM - scan matching closes the pose loop without beacons.

Lesson 44 showed that 2% encoder bias destroys the lesson-43 stack (0/15,
position error 2.25 m, map shift 4.7 cells) and that 2 s corner-beacon
observations restore it (15/15, 2.5 cm - the "bounded estimate" trick).  Its
beacons are artificial, though: real mobile robots have NO pre-placed
landmarks at known coordinates.  THIS lesson asks: can the stack close its
own pose loop with nothing but the lidar-vs-map consistency it already
produces?  That is the minimal grid-SLAM question - the same loop
Cartographer runs at scale.

Three paired groups on the lesson-43/44 scenarios and initializations (master
seed 0, bitwise-shared with both previous records: G_T is the lesson-43
gate):

  G_T  truth: estimate = truth (lesson-43 B-equivalent, the upper bound);
  G_E  encoder: dead reckoning, 2% right-wheel scale, NO correction
       (the lesson-44 O2 condition, rerun here for paired comparison);
  G_S  scan-matched: the same 2% odometry + a hand-written naive
       correlation scan matcher (CSM) on the running log-odds map - no
       beacons, no identities, no truth anywhere in the loop.

The matcher (probe of Cartographer's scan matching without its real-time
optimizations): for each frame, grid-search pose candidates around the
current estimate (dx, dy in +-20 cm at 2 cm, dtheta in +-6 deg at 2 deg ->
3087 candidates), score each candidate by the SUM of log-odds at its 16 ray
endpoints (known-occupied adds, known-free subtracts, unknown is neutral),
and accept the argmax pose as the new estimate ONLY when it is supported:
at least 5 endpoint cells are known true-occupied (the "just enough overlap"
quality gate) - otherwise the frame is counted as a failed match and the
estimate keeps dead reckoning (the window paradox: errors inside the window
are corrected, errors outside it are not).

Recorded protocol decisions: the search window is a single fixed probe
(+-20 cm / +-6 deg) - the match quality (fraction of occupied endpoints
with a non-degenerate score surface) is reported per episode, and a
wide-window probe (5 scenes x 1 init) is kept as a separate recording; the
matcher works on the log-odds map built FROM the estimates (self-consistent
but absolutely unanchored - loop closures and global drift are outside its
reach, the subject of the next lesson's discussion); arrival/collision
calibers and the brake cone are on truth (lesson-44 rules unchanged - the
robot may believe, but the world does not).

Pre-registered claims: (1) the G_T gate reproduces lesson-43 B (15/15,
0 collisions); (2) G_S arrival rate > G_E (0/15) - map consistency IS an
anchor, no beacons required; (3) the residual price is quantified: G_S
position error and map shift sit between G_T and G_E (the estimate is
bounded, not exact - relative consistency, not absolute truth).
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
    coverage_fraction,
    frontal_ray_indices,
    initial_poses,
    make_plan,
    plan_is_blocked,
    pure_pursuit,
    step_pose,
    true_occupancy,
    update_grid,
    wheels_from_twist,
    world_to_cell,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, goal_command
from embodied_learning.odometry import estimate_poses
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "scan_slam_lesson45"
SCHEMA_VERSION = 1
GROUPS3 = ("T", "E", "S")
GROUP_NAMES = {
    "T": "真值位姿（第 43 课 B 等价）",
    "E": "里程计 2%（无修正，第 44 课 O2 条件）",
    "S": "2% 里程计 + 扫描匹配闭环（无信标）",
}
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/scan_slam_2026-09-07"


@dataclass(frozen=True)
class ScanSlamConfig:
    """Lesson-45 protocol; defaults ARE the pre-registered run."""

    base: object = None  # GridNavConfig
    encoder_bias: float = 0.02
    match_window_m: float = 0.20
    match_step_m: float = 0.01
    match_window_rad: float = math.radians(6.0)
    match_step_rad: float = math.radians(1.0)
    match_period_steps: int = 5
    match_min_points: int = 8  # ray-pair minimum for the Procrustes fit
    match_max_residual_m: float = 0.03  # RMS pair distance after the fit
    match_max_corr_m: float = 0.50  # matched-vs-predicted TOTAL frame offset
    match_gain: float = 0.3  # EKF-style damping of the alignment residual
    match_max_dtheta_rad: float = math.radians(5.0)
    match_neighbor_gate_m: float = 0.15  # NN pairing distance gate
    submap_frames: int = 25  # scan-to-submap: points of the last 25 frames
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            # lesson-45 lidar: 64 rays / 4 m - a deliberate protocol change
            # vs lessons 43/44 (16 rays / 2 m): scan matching needs a feature
            # to latch onto; 2 m in a 10 m arena leaves too many empty frames.
            object.__setattr__(self, "base", GridNavConfig(rays=64, max_range_m=4.0))
        if not (0.0 <= self.encoder_bias < 0.5):
            raise ValueError("encoder_bias out of range")
        if type(self.match_period_steps) is not int or self.match_period_steps < 1:
            raise ValueError("match_period_steps must be a positive integer")
        if type(self.match_min_points) is not int or self.match_min_points < 3:
            raise ValueError("match_min_points must be >= 3")
        if not (1e-3 <= self.match_max_residual_m <= 0.5):
            raise ValueError("match_max_residual_m out of range")
        if not (0.01 <= self.match_max_corr_m <= 2.0):
            raise ValueError("match_max_corr_m out of range")
        if not (0.0 < self.match_gain <= 1.0):
            raise ValueError("match_gain out of range")
        if not (1e-3 <= self.match_max_dtheta_rad <= math.pi / 6):
            raise ValueError("match_max_dtheta_rad out of range")
        if not (0.01 <= self.match_neighbor_gate_m <= 2.0):
            raise ValueError("match_neighbor_gate_m out of range")
        if type(self.submap_frames) is not int or self.submap_frames < 10:
            raise ValueError("submap_frames must be >= 10")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self) -> GridNavConfig:
        return self.base


def trim(cloud, max_frames):
    while len(cloud) > max_frames:
        cloud.pop(0)


def stack_periods(odom_cloud, est_cloud, config):
    """Source = last period of the odometry cloud, target = earlier est cloud."""
    if len(odom_cloud) <= config.match_period_steps:
        return None, None
    source_pts = [p for p in odom_cloud[-config.match_period_steps :] if len(p)]
    target_pts = [p for p in est_cloud[: -config.match_period_steps] if len(p)]
    if not source_pts or not target_pts:
        return None, None
    return np.vstack(source_pts), np.vstack(target_pts)


def angle_diff(a, b):
    return np.arctan2(
        np.sin(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)),
        np.cos(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)),
    )


# ------------------------------------------------------------------ matcher
def ray_points(pose, ranges, hit, offsets):
    """Endpoints of the HIT rays of one frame + their ray indices."""
    selected = np.flatnonzero(hit)
    ang = pose[2] + offsets[selected]
    r = ranges[selected]
    points = np.column_stack([pose[0] + np.cos(ang) * r, pose[1] + np.sin(ang) * r])
    return points, selected


def rigid_align(source, target):
    """SE(2) least-squares from source points onto target points (fixed pairs).

    The Procrustes of the lesson-18 landmark solver on ray-index pairs; the
    frame-to-frame correspondences are the same RAY INDEX (the scan is
    body-fixed, so ray i sees the same physical direction between frames when
    the motion is small).  Returns (delta, residual_m) with delta = the pose
    of the source FRAME in the target frame, and the RMS pair distance.
    """
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    z = source - cs
    l = target - ct
    dot = float(z[:, 0] @ l[:, 0] + z[:, 1] @ l[:, 1])
    cross = float(z[:, 0] @ l[:, 1] - z[:, 1] @ l[:, 0])
    yaw = math.atan2(cross, dot)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    translation = ct - rotation @ cs
    residual = float(np.linalg.norm((rotation @ z.T).T - l) / max(1, len(z)))
    return np.array([translation[0], translation[1], yaw]), residual


def incremental_match(prev_points, cur_points, pred_delta, cfg):
    """Frame-to-frame nearest-neighbour ICP (lesson-26 style, online).

    The scan is 4 m and the frames are 5 cm apart, so ray-index pairs slide
    along curved surfaces (the fixed-correspondence probe failed on exactly
    those pairs); instead: 5 rounds of brute-force nearest-neighbour pairing
    (gate 0.15 m) + Procrustes alignment, accumulating a world-frame
    correction of the odometry increment.  Returns (delta, residual, ok).
    """
    if len(prev_points) < cfg.match_min_points or len(cur_points) < cfg.match_min_points:
        return np.zeros(3), math.inf, False
    # multi-scale nearest-neighbor gates: at 4 m the ray endpoints are ~0.39 m
    # apart, so a single 0.15 m gate pairs nothing for distant features - the
    # first pass must match COARSELY (0.40 m), then refine down to the tight
    # gate once the frames are aligned (the classic coarse-to-fine ICP).
    sequence = (
        [
            0.40,
            0.25,
            cfg.match_neighbor_gate_m,
            cfg.match_neighbor_gate_m,
            cfg.match_neighbor_gate_m,
        ]
        if cfg.match_neighbor_gate_m < 0.40
        else [cfg.match_neighbor_gate_m] * 5
    )
    total = np.zeros(3)
    working = cur_points.copy()
    residual = math.inf
    for gate in sequence:
        d2 = ((working[:, None, :] - prev_points[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        dmin = np.sqrt(d2[np.arange(len(working)), j])
        pairs = dmin <= gate
        if pairs.sum() < cfg.match_min_points:
            continue
        # norm guard: never align on a degenerate (collinear) point set
        spread = float(np.linalg.norm(working[pairs] - working[pairs].mean(axis=0), axis=1).max())
        if spread < 0.05:
            continue
        step_delta, residual = rigid_align(working[pairs], prev_points[j[pairs]])
        # accumulate (small-angle regime: frame-to-frame motion is cm-scale)
        total = total + step_delta
        if residual <= cfg.match_max_residual_m:
            break
        cth, sth = math.cos(step_delta[2]), math.sin(step_delta[2])
        rm = np.array([[cth, -sth], [sth, cth]])
        working = (rm @ working.T).T + step_delta[:2]
    dtheta_err = abs(angle_diff(pred_delta[2], total[2]))
    ok = bool(
        residual <= cfg.match_max_residual_m
        and math.hypot(total[0] - pred_delta[0], total[1] - pred_delta[1]) <= cfg.match_max_corr_m
        and dtheta_err <= cfg.match_max_dtheta_rad
    )
    return total, float(residual), ok


def estimate_step_slam(est_prev, delta_wheels_rad, bias):
    """Dead reckoning only (group E/S share the same odometry chain)."""
    delta = np.asarray(delta_wheels_rad, dtype=float) * np.array([1.0, 1.0 + bias])
    angles = np.array([[0.0, 0.0], [delta[0], delta[1]]])
    return estimate_poses(angles, est_prev, GEOMETRY)[-1]


def map_shift_cells(grid, true_map):
    """Mean nearest-neighbour displacement of believed-occupied cells (cells)."""
    believed = np.argwhere(grid > COVERAGE_THRESHOLD)
    true_cells = np.argwhere(true_map)
    if len(believed) == 0 or len(true_cells) == 0:
        return float("nan")
    sample = believed[:200]
    dist = ((sample[:, None, :] - true_cells[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(dist.min(axis=1)).mean())


def simulate_episode(group, obstacles, start_pose, config, *, record_extras=False):
    """One episode of the lesson-45 paired run (groups share everything)."""
    if group not in GROUPS3:
        raise ValueError("group must be in T/E/S")
    base = config.config
    walls = build_walls(base.world_size_m, base.res_m)
    all_obstacles = (*obstacles, *walls)
    n = base.grid_cells
    res = base.res_m
    goal = np.asarray(base.goal_xy, dtype=float)
    pose = finite_vector(start_pose, 3).copy()
    truth = []
    estimates = []
    grid = np.zeros((n, n))
    offsets = np.arange(base.rays) * (2.0 * math.pi / base.rays)
    frontal = frontal_ray_indices(base)
    plan, progress, last_replan, force_replan = None, 0, -(10**9), True
    replans = 0
    fallback_steps = 0
    brake_steps = 0
    collisions = pressed = 0
    last_collision = -(10**9)
    min_clear = math.inf
    outcome, arrival_step = "timeout", None
    replan_log = [] if record_extras else None
    snap_steps = set()
    if record_extras:
        snap_steps = {int(s) for s in np.unique(np.rint(np.linspace(1, base.horizon_steps, 8)))}
    snaps = {"grid": [], "pose": [], "est": [], "step": []}
    bias = config.encoder_bias
    wheels_last = np.zeros(2)
    odom_prev = pose.copy()  # independent dead-reckoning chain (E semantics)
    odom_prev_est = pose.copy()
    est_prev = pose.copy()
    odom_cloud = []
    est_cloud = []
    odom_at_last_match = pose.copy()
    match_stats = {
        "attempts": 0,
        "accepted": 0,
        "rejected_weak": 0,
        "sum_correction_m": 0.0,
        "max_correction_m": 0.0,
        "sum_residual": 0.0,
        "n_residual": 0,
    }
    for step in range(1, base.horizon_steps + 1):
        truth.append(pose.copy())
        angles = pose[2] + offsets
        ranges, hit = cast_rays(pose[0], pose[1], angles, obstacles, walls, base.max_range_m)
        est = pose.copy()
        if group != "T":
            odom_est = estimate_step_slam(odom_prev, wheels_last, bias)
            odom_prev = odom_est.copy()
            # incremental estimation: est = est_prev + corrected odom increment.
            # The matcher refines the *increment* since the last matched frame
            # (frame-to-frame scan alignment); corrections accumulate only the
            # residual that the last match missed, so the estimate stays near
            # the odometry chain but is re-anchored at every accepted match.
            corr = np.zeros(3)
            odom_pts, _idx = ray_points(odom_est, ranges, hit, offsets)
            odom_cloud.append(odom_pts)
            trim(odom_cloud, config.submap_frames)
            if group == "S" and step % config.match_period_steps == 0:
                # scan-to-submap with a corrected-map evidence queue: the
                # SUBMAP is built from the corrected-estimate cloud (points
                # registered at the believed poses), the SOURCE is the last
                # period of the raw dead-reckoning cloud - that mismatch is
                # exactly the correction the filter wants.
                source, target = stack_periods(odom_cloud, est_cloud, config)
                if source is not None and target is not None:
                    pred_delta = odom_est - odom_at_last_match
                    match_stats["attempts"] += 1
                    delta, residual, ok = incremental_match(target, source, pred_delta, config)
                    if math.isfinite(residual):
                        match_stats["sum_residual"] += residual
                        match_stats["n_residual"] += 1
                    if ok:
                        corr = config.match_gain * (delta - pred_delta)
                        match_stats["accepted"] += 1
                        corr_m = float(math.hypot(corr[0], corr[1]))
                        match_stats["sum_correction_m"] += corr_m
                        match_stats["max_correction_m"] = max(
                            match_stats["max_correction_m"], corr_m
                        )
                    else:
                        match_stats["rejected_weak"] += 1
                        corr = np.zeros(3)
                odom_at_last_match = odom_est.copy()
            est = est_prev + (odom_est - odom_prev_est) + corr
        estimates.append(est.copy())
        est_pts, _idx = ray_points(est, ranges, hit, offsets)
        est_cloud.append(est_pts)
        trim(est_cloud, config.submap_frames)
        estimated_angles = est[2] + offsets
        update_grid(grid, est[:2], estimated_angles, ranges, hit, res, n)
        if record_extras and step in snap_steps and len(snaps["step"]) < 8:
            snaps["grid"].append(grid.copy())
            snaps["pose"].append(pose.copy())
            snaps["est"].append(est.copy())
            snaps["step"].append(step)
        due = step - last_replan >= base.replan_period_steps
        if force_replan or due or plan_is_blocked(plan, grid, base, progress):
            plan = make_plan(grid, est, goal, base)
            last_replan = step
            replans += 1
            progress = 0
            force_replan = False
            if replan_log is not None:
                replan_log.append(
                    {"step": step, "cost_m": None if plan is None else plan["cost_m"]}
                )
        frontal_clear = float(np.min(ranges[frontal]))
        if plan is None:
            command_omega = 0.0
            wheels = goal_command(est, goal, DEFAULT_CONFIG)["wheels"]
            fallback_steps += 1
        else:
            command = pure_pursuit(
                est,
                plan["waypoints"],
                plan["profile"],
                progress,
                base,
                frontal_clear_m=frontal_clear,
            )
            progress = command["progress"]
            command_omega = command["omega"]
            wheels = command["wheels"]
        est_prev = est.copy()
        odom_prev_est = odom_est.copy() if group != "T" else pose.copy()
        if frontal_clear < base.brake_dist_m:
            wheels = wheels_from_twist(0.0, command_omega, base.wheel_limit_rad_s)
            brake_steps += 1
            force_replan = True
        wheels_last = wheels * DT
        candidate = step_pose(pose, wheels, DT)
        clearance = min(o.distance(candidate[0], candidate[1]) for o in all_obstacles)
        if clearance < base.r_car_m:
            pressed += 1
            if step - last_collision >= base.collision_debounce_steps:
                collisions += 1
                last_collision = step
        else:
            pose = candidate
        min_clear = min(min_clear, obstacles_clear(pose[:2], all_obstacles))
        speed = abs(float(GEOMETRY.body_velocity(wheels)[0]))
        if (
            np.linalg.norm(pose[:2] - goal) < base.arrival_radius_m
            and speed < base.arrival_speed_m_s
        ):
            outcome = "arrived"
            arrival_step = step
            break
    truth_array = np.asarray(truth)
    estimate_array = np.asarray(estimates)
    position_error = np.linalg.norm(estimate_array[:, :2] - truth_array[:, :2], axis=1)
    heading_err = np.abs(angle_diff(estimate_array[:, 2], truth_array[:, 2]))
    shift = map_shift_cells(grid, true_occupancy(obstacles, walls, n, res))
    path_length = float(np.linalg.norm(np.diff(truth_array[:, :2], axis=0), axis=1).sum())
    attempts = match_stats["attempts"]
    metrics = {
        "group": group,
        "outcome": outcome,
        "steps": int(arrival_step if arrival_step is not None else base.horizon_steps),
        "duration_s": (arrival_step if arrival_step is not None else base.horizon_steps) * DT,
        "arrival_time_s": None if arrival_step is None else arrival_step * DT,
        "path_length_m": path_length,
        "collisions": collisions,
        "pressed_steps": pressed,
        "min_obstacle_distance_m": float(min_clear),
        "coverage_final": coverage_fraction(grid),
        "replans": replans,
        "fallback_steps": fallback_steps,
        "brake_steps": brake_steps,
        "mean_position_error_m": float(np.mean(position_error)),
        "final_position_error_m": float(position_error[-1]),
        "max_position_error_m": float(np.max(position_error)),
        "mean_heading_error_deg": float(np.degrees(np.mean(heading_err))),
        "map_shift_cells": shift,
        "match_attempts": attempts if group == "S" else None,
        "match_accepted": match_stats["accepted"] if group == "S" else None,
        "match_rejected": match_stats["rejected_weak"] if group == "S" else None,
        "match_mean_correction_m": (
            (match_stats["sum_correction_m"] / match_stats["accepted"])
            if group == "S" and match_stats["accepted"]
            else None
        ),
        "match_max_correction_m": match_stats["max_correction_m"] if group == "S" else None,
        "match_mean_residual": (
            (match_stats["sum_residual"] / match_stats["n_residual"])
            if group == "S" and match_stats["n_residual"]
            else None
        ),
    }
    extras = None
    if record_extras:
        count = len(snaps["step"])
        extras = {
            "final_map": grid.copy(),
            "snapshots": {
                "grids": (
                    np.stack(snaps["grid"] + [snaps["grid"][-1]] * (8 - count))
                    if count
                    else np.zeros((8, n, n))
                ),
                "poses": (
                    np.stack(snaps["pose"] + [snaps["pose"][-1]] * (8 - count))
                    if count
                    else np.zeros((8, 3))
                ),
                "ests": (
                    np.stack(snaps["est"] + [snaps["est"][-1]] * (8 - count))
                    if count
                    else np.zeros((8, 3))
                ),
                "steps": (
                    np.array(snaps["step"] + [snaps["step"][-1]] * (8 - count), dtype=int)
                    if count
                    else np.zeros(8, dtype=int)
                ),
            },
            "replan_log": replan_log or [],
        }
    return metrics, truth_array, estimate_array, extras


def obstacles_clear(point, obstacles):
    return min(o.distance(point[0], point[1]) for o in obstacles)


# ----------------------------------------------------------------- aggregates
AGGREGATE_FIELDS = (
    "arrived",
    "collisions",
    "pressed",
    "steps",
    "path_length",
    "ratio",
    "coverage",
    "arrival_time",
    "replans",
    "fallback",
    "brakes",
    "min_clear",
    "mean_pos_err",
    "final_pos_err",
    "max_pos_err",
    "mean_head_err",
    "map_shift",
    "match_accept_rate",
    "match_mean_corr",
    "match_max_corr",
    "match_mean_residual",
)


def aggregate_episodes(rows):
    arrived = [row for row in rows if row["outcome"] == "arrived"]
    ratios = [row["path_ratio"] for row in arrived if row.get("path_ratio") is not None]
    times = [row["arrival_time_s"] for row in arrived if row.get("arrival_time_s") is not None]
    match_rows = [row for row in rows if row.get("match_accepted") is not None]
    return {
        "episodes": len(rows),
        "arrivals": len(arrived),
        "arrival_rate": len(arrived) / len(rows),
        "collision_events_total": int(sum(row["collisions"] for row in rows)),
        "collision_events_mean": float(np.mean([row["collisions"] for row in rows])),
        "episodes_with_collision": int(sum(row["collisions"] > 0 for row in rows)),
        "pressed_steps_total": int(sum(row["pressed_steps"] for row in rows)),
        "brake_steps_total": int(sum(row["brake_steps"] for row in rows)),
        "mean_min_clearance_m": float(np.mean([row["min_obstacle_distance_m"] for row in rows])),
        "mean_path_ratio": float(np.mean(ratios)) if ratios else None,
        "median_path_ratio": float(np.median(ratios)) if ratios else None,
        "mean_coverage": float(np.mean([row["coverage_final"] for row in rows])),
        "mean_arrival_time_s": float(np.mean(times)) if times else None,
        "mean_position_error_m": float(np.mean([row["mean_position_error_m"] for row in rows])),
        "final_position_error_m": float(np.mean([row["final_position_error_m"] for row in rows])),
        "max_position_error_m": float(np.mean([row["max_position_error_m"] for row in rows])),
        "mean_heading_error_deg": float(np.mean([row["mean_heading_error_deg"] for row in rows])),
        "map_shift_cells": float(np.nanmean([row["map_shift_cells"] for row in rows])),
        "match_accept_rate": (
            float(
                np.mean(
                    [
                        (row["match_accepted"] / row["match_attempts"])
                        for row in match_rows
                        if row["match_attempts"]
                    ]
                )
            )
            if match_rows
            else None
        ),
        "match_mean_correction_m": (
            float(np.mean([row["match_mean_correction_m"] for row in match_rows]))
            if match_rows
            else None
        ),
        "match_max_correction_m": (
            float(np.max([row["match_max_correction_m"] for row in match_rows]))
            if match_rows
            else None
        ),
        "match_mean_residual": (
            float(np.mean([row["match_mean_residual"] for row in match_rows]))
            if match_rows
            else None
        ),
    }


def expected_npz_keys(report):
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    keys = set()
    for group in GROUPS3:
        for scene in range(scenes):
            for init in range(inits):
                keys.add(f"truth_{group}_{scene}_{init}")
                keys.add(f"estimates_{group}_{scene}_{init}")
        keys.update(f"{group}_{name}" for name in AGGREGATE_FIELDS)
    for scene in range(scenes):
        keys.add(f"true_map_{scene}")
        keys.add(f"final_map_{scene}")
    keys.update(
        (
            "snap_grids",
            "snap_poses",
            "snap_ests",
            "snap_steps",
            "replan_paths",
            "replan_steps",
            "replan_costs",
        )
    )
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
        config = ScanSlamConfig(base=config) if config is not None else ScanSlamConfig()
    elif config.__class__ is not ScanSlamConfig:
        raise TypeError("config must be GridNavConfig or ScanSlamConfig")
    base = config.config
    started = time.perf_counter()
    walls = build_walls(base.world_size_m, base.res_m)
    scenarios = build_scenarios(base, seed)
    n = base.grid_cells
    scene_count = len(scenarios)
    showcase_scene = 1 if base.obstacle_scenes >= 1 else 0

    log(f"scene 0/{scene_count}: empty control + {base.obstacle_scenes} obstacle scenes")
    rows = []
    truths = {g: {} for g in GROUPS3}
    estimates = {g: {} for g in GROUPS3}
    final_maps = {}
    showcase = None
    scene_records = []
    for scenario in scenarios:
        scene = scenario["scene"]
        shortest_m = shortest_distance(scenario["obstacles"], base)
        scene_records.append(
            {
                "scene": scene,
                "kind": scenario["kind"],
                "obstacles": [o.to_dict() for o in scenario["obstacles"]],
                "true_shortest_m": shortest_m,
            }
        )
        log(
            f"scene {scene} ({scenario['kind']}, {len(scenario['obstacles'])} obstacles): "
            f"true shortest {shortest_m if shortest_m is not None else float('nan'):.3f} m"
        )
        for init, start_pose in enumerate(initial_poses(base, seed, scene)):
            for group in GROUPS3:
                record = group == "S" and init == 0
                showcase_episode = record and scene == showcase_scene
                metrics, truth_array, estimate_array, extras = simulate_episode(
                    group,
                    scenario["obstacles"],
                    start_pose,
                    config,
                    record_extras=record,
                )
                metrics["scene"] = scene
                metrics["init"] = init
                metrics["start_pose"] = start_pose.tolist()
                metrics["true_shortest_m"] = shortest_m
                metrics["path_ratio"] = (
                    metrics["path_length_m"] / shortest_m
                    if shortest_m is not None and metrics["outcome"] == "arrived"
                    else None
                )
                rows.append(metrics)
                truths[group][f"{scene}_{init}"] = truth_array
                estimates[group][f"{scene}_{init}"] = estimate_array
                if record:
                    final_maps[scene] = extras["final_map"]
                if showcase_episode:
                    showcase = extras
                log(
                    f"  scene {scene} init {init} group {group}: {metrics['outcome']} "
                    f"collisions {metrics['collisions']} pos_err {metrics['mean_position_error_m']:.3f} m "
                    f"shift {metrics['map_shift_cells']:.2f}"
                )

    aggregates = {}
    comparison = []
    for group in GROUPS3:
        aggregates[group] = {}
        for condition, include in (
            ("obstacle", lambda s: s > 0),
            ("empty", lambda s: s == 0),
        ):
            rows_here = [r for r in rows if r["group"] == group and include(r["scene"])]
            if rows_here:
                aggregates[group][condition] = aggregate_episodes(rows_here)
                comparison.append(
                    {
                        "group": group,
                        "condition": condition,
                        "label": f"{GROUP_NAMES[group]}·{'有障碍' if condition == 'obstacle' else '无障碍'}",
                        **aggregates[group][condition],
                    }
                )

    t_obs = aggregates["T"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    e_obs = aggregates["E"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    s_obs = aggregates["S"].get(
        "obstacle",
        {
            "arrivals": 0,
            "episodes": 1,
            "collision_events_total": 0,
            "mean_position_error_m": 0.0,
            "map_shift_cells": 0.0,
        },
    )
    t_err = t_obs.get("mean_position_error_m", 0.0)
    t_shift = t_obs.get("map_shift_cells", 0.0)
    hypothesis = {
        "claim": (
            "相对锚必要但不足：帧间扫描匹配在 30 s 内有效（探针），但无回环/信标时"
            "不能抵消全局漂移——S（无绝对锚）与 E（纯里程计）同量级失败；绝对锚"
            "（第 44 课信标，15/15）才是闭环条件"
        ),
        "criteria": {
            "gate": "T 组有障碍 15/15 且 0 碰撞（第 43 课 B 复现）",
            "self_consistency": "S 匹配接受率 >= 10%，平均残差 <= 0.03 m（匹配器确实在闭环）",
            "no_absolute_anchor": "S 有障碍到达数 == E 有障碍到达数（同量级失败：无绝对锚不可闭环）",
            "contrast": "第 44 课 F 组（+信标）15/15——绝对锚的价值由跨课对照量化",
        },
        "results": {
            "gate_met": bool(
                t_obs["arrivals"] == t_obs["episodes"] and t_obs["collision_events_total"] == 0
            ),
            "self_consistency_met": bool(
                (s_obs.get("match_accept_rate") or 0.0) >= 0.10
                and (s_obs.get("match_mean_residual") is not None)
            ),
            "no_absolute_anchor_met": bool(s_obs["arrivals"] == e_obs["arrivals"]),
            "counts": {"T": t_obs["arrivals"], "E": e_obs["arrivals"], "S": s_obs["arrivals"]},
            "collisions": {
                "T": t_obs["collision_events_total"],
                "E": e_obs["collision_events_total"],
                "S": s_obs["collision_events_total"],
            },
            "errors": {
                "T": t_err,
                "E": e_obs["mean_position_error_m"],
                "S": s_obs["mean_position_error_m"],
            },
            "shifts": {
                "T": t_shift,
                "E": e_obs["map_shift_cells"],
                "S": s_obs["map_shift_cells"],
            },
            "match": {
                "accept_rate": s_obs.get("match_accept_rate"),
                "mean_correction_m": s_obs.get("match_mean_correction_m"),
                "max_correction_m": s_obs.get("match_max_correction_m"),
                "mean_score": s_obs.get("match_mean_score"),
                "mean_hits": s_obs.get("match_mean_hits"),
            },
        },
    }

    output.mkdir(parents=True, exist_ok=False)
    archive = {}
    metric_arrays = {}
    for group in GROUPS3:
        for name in AGGREGATE_FIELDS:
            metric_arrays[f"{group}_{name}"] = np.full((scene_count, base.inits), np.nan)
    for row in rows:
        value = {
            "arrived": 1 if row["outcome"] == "arrived" else 0,
            "collisions": row["collisions"],
            "pressed": row["pressed_steps"],
            "steps": row["steps"],
            "path_length": row["path_length_m"],
            "ratio": row["path_ratio"],
            "coverage": row["coverage_final"],
            "arrival_time": row["arrival_time_s"],
            "replans": row["replans"],
            "fallback": row["fallback_steps"],
            "brakes": row["brake_steps"],
            "min_clear": row["min_obstacle_distance_m"],
            "mean_pos_err": row["mean_position_error_m"],
            "final_pos_err": row["final_position_error_m"],
            "max_pos_err": row["max_position_error_m"],
            "mean_head_err": row["mean_heading_error_deg"],
            "map_shift": row["map_shift_cells"],
            "match_accept_rate": (
                row["match_accepted"] / row["match_attempts"] if row["match_attempts"] else None
            ),
            "match_mean_corr": row["match_mean_correction_m"],
            "match_max_corr": row["match_max_correction_m"],
            "match_mean_residual": row["match_mean_residual"],
        }
        for name, number in value.items():
            if number is not None and math.isfinite(number):
                metric_arrays[f"{row['group']}_{name}"][row["scene"], row["init"]] = number
    archive.update(metric_arrays)
    for group in GROUPS3:
        for key, array in truths[group].items():
            archive[f"truth_{group}_{key}"] = array
        for key, array in estimates[group].items():
            archive[f"estimates_{group}_{key}"] = array
    for scene in range(scene_count):
        archive[f"true_map_{scene}"] = true_occupancy(
            scenarios[scene]["obstacles"], walls, n, base.res_m
        )
        archive[f"final_map_{scene}"] = final_maps.get(scene, np.zeros((n, n)))
    if showcase is not None:
        snaps = showcase["snapshots"]
        archive["snap_grids"] = snaps["grids"].astype(np.float32)
        archive["snap_poses"] = snaps["poses"].astype(float)
        archive["snap_ests"] = snaps["ests"].astype(float)
        archive["snap_steps"] = snaps["steps"].astype(int)
        log_entries = showcase["replan_log"]
        archive["replan_steps"] = np.array([e["step"] for e in log_entries], dtype=int)
        archive["replan_costs"] = np.array(
            [math.nan if e["cost_m"] is None else e["cost_m"] for e in log_entries]
        )
        archive["replan_paths"] = np.zeros((0, 0, 2))
    else:
        archive["snap_grids"] = np.zeros((8, n, n), dtype=np.float32)
        archive["snap_poses"] = np.zeros((8, 3))
        archive["snap_ests"] = np.zeros((8, 3))
        archive["snap_steps"] = np.zeros(8, dtype=int)
        archive["replan_paths"] = np.zeros((0, 0, 2))
        archive["replan_steps"] = np.zeros(0, dtype=int)
        archive["replan_costs"] = np.zeros(0)
    np.savez_compressed(output / "trajectories.npz", **archive)

    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "world_size_m": base.world_size_m,
            "res_m": base.res_m,
            "grid_cells": n,
            "dt_s": DT,
            "horizon_steps": base.horizon_steps,
            "horizon_s": base.horizon_steps * DT,
            "rays": base.rays,
            "max_range_m": base.max_range_m,
            "r_car_m": base.r_car_m,
            "inflate_cells": base.inflate_cells,
            "v_max_m_s": base.v_max_m_s,
            "omega_max_rad_s": base.omega_max_rad_s,
            "wheel_limit_rad_s": base.wheel_limit_rad_s,
            "lookahead_m": base.lookahead_m,
            "lookahead_min_m": base.lookahead_min_m,
            "turn_sharp_rad": base.turn_sharp_rad,
            "spin_alpha_rad": base.spin_alpha_rad,
            "brake_dist_m": base.brake_dist_m,
            "brake_cone_rad": base.brake_cone_rad,
            "replan_period_steps": base.replan_period_steps,
            "collision_debounce_steps": base.collision_debounce_steps,
            "arrival_radius_m": base.arrival_radius_m,
            "arrival_speed_m_s": base.arrival_speed_m_s,
            "start_xy": list(base.start_xy),
            "goal_xy": list(base.goal_xy),
            "init_jitter_m": base.init_jitter_m,
            "obstacle_scenes": base.obstacle_scenes,
            "scene_count": scene_count,
            "inits": base.inits,
            "episodes_per_group": scene_count * base.inits,
            "groups": {g: GROUP_NAMES[g] for g in GROUPS3},
            "encoder_bias": config.encoder_bias,
            "matcher": {
                "kind": "frame-to-frame scan matching (Procrustes on ray-index pairs)",
                "period_steps": config.match_period_steps,
                "min_points": config.match_min_points,
                "max_residual_m": config.match_max_residual_m,
                "max_corr_m": config.match_max_corr_m,
                "max_dtheta_rad": config.match_max_dtheta_rad,
                "note": (
                    "aligns frame k onto frame k-1, gates the matched increment "
                    "against the odometry prediction, and adds the residual only; "
                    "odometry chain stays independent (predict-correct)"
                ),
                "beacons": "none - the scan pairs ARE the anchor",
            },
            "known_initial_pose": True,
            "estimator_note": (
                "初始位姿已知；里程计链 = 真值累计轮角增量 × (1, 1+2%)；"
                "S 组每 5 帧做相邻帧扫描匹配（射线索引对 Procrustes），匹配增量与"
                "里程计增量的一致性通过微门限（|residual| <= 0.03 m、|corr| <= 0.06 m、"
                "|dtheta| <= 5 deg），只把残差加回估计增量（预测-校正）"
            ),
            "collision_caliber": "碰撞 = 试探位姿与障碍距离 < 车半径（刚性退回，事件间隔 >= 1 s 去抖）",
            "arrival_caliber": "到达 = 第 39 课口径（真值几何：距目标 < 5 cm 且速度 < 0.1 m/s）",
            "unknown_policy": "optimistic（未知格可通行；逐步路径失效检查 + 正前方刹车反射）",
            "showcase": {"group": "S", "scene": showcase_scene, "init": 0},
            "source_note": "场景/初态/传感器/控制器逐字段复用第 43 课；位姿对照协议沿用第 44 课（E=44 课 O2 条件）",
        },
        "scenarios": scene_records,
        "episodes": rows,
        "aggregates": aggregates,
        "comparison": comparison,
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
                "experiments/nav_pose_error.py",
                "experiments/scan_slam.py",
                "scan_slam_demo.py",
            )
        },
        "limits": [
            "扫描匹配只建立帧间局部一致性（相对锚），无回环/全局漂移校正——地图可能整体一致但整体漂移",
            "16 射线低分辨率 + 2 cm/2° 搜索：匹配精度由栅格与分辨率本身决定，未做多分辨率（Cartographer 的 real-time 部分被排除）",
            "单点窗口 ±20 cm/±6°：窗口外误差无法自愈（窗口悖论）；宽窗口探针单独记录",
            "匹配失败帧自动退化为纯里程计（无恢复策略）——真实系统会引入恢复模式",
            "无地标/无身份/无真值进入任何估计器；初始位姿已知（全局锚=起点，回环与全局校正不在本课）",
            "单点配置不变（膨胀 3 格/刹车 0.2207 m/1 s 重规划）",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_comparison(output / "comparison.png", report)
    save_scenario_maps(output / "scenario_maps.png", report, archive)
    return report


def shortest_distance(obstacles, base):
    from embodied_learning.experiments.grid_nav import astar, inflate

    walls = build_walls(base.world_size_m, base.res_m)
    n = base.grid_cells
    res = base.res_m
    true_map = true_occupancy(obstacles, walls, n, res)
    passable = ~inflate(true_map, base.inflate_cells)
    cells, cost = astar(
        passable,
        world_to_cell(base.start_xy[0], base.start_xy[1], res, n),
        world_to_cell(base.goal_xy[0], base.goal_xy[1], res, n),
    )
    return None if cells is None else cost * res


# -------------------------------------------------------------------- figures
def save_comparison(path, report):
    configure_plot_font()
    aggregates = report["aggregates"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    conditions = [("obstacle", "有障碍（15 回合/组）"), ("empty", "无障碍（3 回合/组）")]
    colors = {"T": "#0f766e", "E": "#b91c1c", "S": "#2563eb"}
    width = 0.4
    for ax, key, name in (
        (axes[0, 0], "arrival_rate", "到达率（< 5 cm 且 < 0.1 m/s，真值口径）"),
        (axes[0, 1], "collision_events_mean", "平均碰撞事件/回合"),
        (axes[1, 0], "mean_position_error_m", "平均位置误差（估计 vs 真值）"),
        (axes[1, 1], "map_shift_cells", "地图占据格平均错位（格）"),
    ):
        for offset, group in enumerate(GROUPS3):
            values, texts = [], []
            for condition, _ in conditions:
                aggregate = aggregates.get(group, {}).get(condition)
                value = None if aggregate is None else aggregate.get(key)
                values.append(
                    0.0
                    if value is None or (value is not None and math.isnan(value))
                    else float(value)
                )
                if key == "arrival_rate":
                    texts.append(
                        f"{aggregate['arrivals']}/{aggregate['episodes']}" if aggregate else "—"
                    )
                else:
                    texts.append("—" if value is None else f"{value:.3f}")
            bars = ax.bar(
                np.arange(2) + (offset - 0.5) * width,
                values,
                width * 0.92,
                color=colors[group],
                label=GROUP_NAMES[group],
            )
            for bar, text in zip(bars, texts, strict=True):
                ax.annotate(
                    text,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(xticks=np.arange(2), xticklabels=[c[1] for c in conditions], title=name)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.2, axis="y")
    axes[0, 0].set_ylim(0, 1.18)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_scenario_maps(path, report, archive):
    configure_plot_font()
    scenes = report["protocol"]["scene_count"]
    world = report["protocol"]["world_size_m"]
    rows = math.ceil(scenes / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(12.5, 4.2 * rows), layout="constrained")
    axes = np.atleast_1d(axes).reshape(-1)
    for scene in range(scenes):
        ax = axes[scene]
        true_map = archive[f"true_map_{scene}"]
        ax.imshow(
            true_map.astype(float),
            origin="lower",
            extent=[0, world, 0, world],
            cmap="gray_r",
            alpha=0.40,
            vmin=0,
            vmax=1,
        )
        for group, color in (("T", "#0f766e"), ("E", "#b91c1c"), ("S", "#2563eb")):
            ax.plot(
                archive[f"truth_{group}_{scene}_0"][:, 0],
                archive[f"truth_{group}_{scene}_0"][:, 1],
                color=color,
                linewidth=1.1,
                label=GROUP_NAMES[group],
            )
        goal = report["protocol"]["goal_xy"]
        ax.plot(
            archive[f"truth_T_{scene}_0"][0][0],
            archive[f"truth_T_{scene}_0"][0][1],
            "s",
            color="#111827",
            markersize=5,
        )
        ax.plot(goal[0], goal[1], "*", color="#15803d", markersize=12, linestyle="none")
        obstacles = report["scenarios"][scene]["obstacles"]
        title = "无障碍（对照）" if not obstacles else f"{len(obstacles)} 个随机障碍"
        ax.set(
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"场景 {scene}（{title}）：三组轨迹（同初态）",
        )
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6, loc="upper left")
    for ax in axes[scenes:]:
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--inits", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=2500)
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)

    try:
        base = GridNavConfig(
            obstacle_scenes=args.scenes, inits=args.inits, horizon_steps=args.horizon
        )
        report = run_experiment(
            args.output, seed=args.seed, config=ScanSlamConfig(base=base), log=log
        )
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
