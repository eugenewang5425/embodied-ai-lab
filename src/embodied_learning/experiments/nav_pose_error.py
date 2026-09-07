"""Lesson 44: how pose-estimation error propagates through the mapping stack.

Lesson 43 left the pose KNOWN to the planned group ("the map is built from an
ideal sensor, localization is excluded by design"); lesson 21 answered "what
happens when the goal feedback uses a drifted estimate" on an obstacle-free
arena.  This lesson closes that gap with the third question: with the lesson-43
stack (16-ray sensor -> Bresenham log-odds mapping -> known-grid A* replanning
-> differential pure pursuit + frontal brake), WHICH part breaks first when the
pose fed into every layer is no longer truth?

Four paired groups on the identical lesson-43 scenarios and initializations
(the same master seed 0, so obstacles and start poses match the lesson-43
record bit-for-bit - G_T is the lesson-43 sanity gate):

  G_T  truth: estimate = truth (lesson-43 B-equivalent, the upper bound);
  G_O1 encoder 1 %: dead reckoning on scaled wheel angles (lesson-15 chain,
                    right-wheel scale 1.01);
  G_O2 encoder 2 %: the lesson-15 2 % condition itself (known to drift to
                    ~19 cm / 9 deg on a straight 2.4 m run);
  G_F  fusion: the 2 % odometry + lesson-18/19 landmark observations (4 corner
               beacons, every 2 s, sigma 1 cm / 0.57 deg, ideal sensor), fused
               causally by pose reset + encoder propagation (lesson-19 rule,
               verified bitwise against pose_fusion.estimate_with_resets).

The estimate enters EVERY layer exactly as a real driver would use it: rays
are physically emitted from the TRUE pose but mapped into the grid at the
ESTIMATED origin/heading (a drifted pose produces a misregistered map), A* is
run on the estimated registration (what the robot believes the world looks
like), pure pursuit chooses targets/alpha from the estimated pose, while the
world steps, collision geometry, arrival caliber and frontal brake readings
stay on truth.  The brake cone is physical (body-fixed); the brake threshold
comparison uses the true frontal reading, because a brake reflex is a body
event, not a belief.

Protocol decisions recorded: initial pose is KNOWN (odometry needs a starting
pose; observability, not estimation, is the variable); beacons are pure
observation targets (physically invisible: no collision, no occlusion - the
lesson-18 ideal sensor model, unchanged); unknown cells stay optimistic with
per-step invalidation + replanning + brake reflex (lesson-43 rules unchanged);
arrival = truth-consistent (dist < 5 cm and speed < 0.1 m/s, the lesson-39
caliber, evaluated on truth so errors cannot be self-reported away).

The pre-registered claims: (1) G_T reproduces lesson-43 B (15/15, 0
collisions) - the gate; (2) arrival rate decreases monotonically with encoder
bias, G_O1 >= G_O2; (3) the fusion group outperforms the same-bias odometry
group on arrivals AND collisions; (4) the gap between G_F and G_T (the
residual cost of a bounded estimate) is quantified - it is the price of
running a stack on an estimate instead of truth.
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
    astar,
    build_scenarios,
    build_walls,
    cast_rays,
    coverage_fraction,
    frontal_ray_indices,
    inflate,
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
from embodied_learning.landmark_localization import (
    OBS_BEARING_STD_RAD,
    OBS_RANGE_STD_M,
    observe,
    solve_pose,
)
from embodied_learning.odometry import estimate_poses
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "nav_pose_error_lesson44"
SCHEMA_VERSION = 1
GROUPS4 = ("T", "O1", "O2", "F")
NAV_GROUPS = {
    "T": "真值位姿（第 43 课 B 等价）",
    "O1": "里程计 1%",
    "O2": "里程计 2%",
    "F": "2% 里程计 + 地标融合",
}
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/nav_pose_error_2026-09-07"
# Four corner beacons, just inside the walls; observation targets only (no
# collision, no occlusion - the lesson-18 ideal model, unchanged).
BEACONS = np.array([[0.55, 0.55], [9.45, 0.55], [0.55, 9.45], [9.45, 9.45]])


@dataclass(frozen=True)
class NavPoseErrorConfig:
    """Lesson-44 protocol; defaults ARE the pre-registered run."""

    base: object = None  # GridNavConfig
    encoder_biases: dict = None
    obs_period_steps: int = 50  # 2 s at dt = 0.04 s; the lesson-18 period
    obs_range_std_m: float = OBS_RANGE_STD_M
    obs_bearing_std_rad: float = OBS_BEARING_STD_RAD
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(self, "base", GridNavConfig())
        if self.encoder_biases is None:
            # F shares the O2 odometry chain (same bias), differing only in
            # observations - that single difference isolates fusion value.
            object.__setattr__(
                self, "encoder_biases", {"T": 0.0, "O1": 0.01, "O2": 0.02, "F": 0.02}
            )
        if not (0.0 <= self.obs_range_std_m <= 1.0):
            raise ValueError("obs_range_std_m out of range")
        if not (0.0 <= self.obs_bearing_std_rad <= 0.5):
            raise ValueError("obs_bearing_std_rad out of range")
        if type(self.obs_period_steps) is not int or self.obs_period_steps < 1:
            raise ValueError("obs_period_steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if set(self.encoder_biases) != set(GROUPS4):
            raise ValueError("encoder_biases must cover exactly T/O1/O2/F")

    @property
    def config(self) -> GridNavConfig:
        return self.base


def angle_diff(a, b):
    """Smallest signed difference b - a in radians; broadcasts like np arctan2."""
    return np.arctan2(
        np.sin(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)),
        np.cos(np.asarray(b, dtype=float) - np.asarray(a, dtype=float)),
    )


def estimate_step(est_prev, delta_wheels_rad, obs_result, bias):
    """One step of the estimator: integrate the last wheel-angle delta.

    delta_wheels_rad = [left, right] wheel-angle increments of the move just
    finished (rad); the right wheel is scaled by (1 + bias).  F additionally
    applies the lesson-19 reset rule (position = observation, heading +=
    heading_error).  T never calls this (estimate = truth by definition).
    """
    delta = np.asarray(delta_wheels_rad, dtype=float) * np.array([1.0, 1.0 + bias])
    angles = np.array([[0.0, 0.0], [delta[0], delta[1]]])
    est = estimate_poses(angles, est_prev, GEOMETRY)[-1]
    if obs_result is not None:
        # lesson-19 reset rule: position = observation, heading += the
        # smallest angle FROM the estimate TO the observation (heading_error
        # semantics: heading_error(estimated, true) = true - estimated).
        est = est.copy()
        est[:2] = obs_result[:2]
        est[2] += float(angle_diff(est[2], obs_result[2]))
    return est


def map_shift_cells(grid, true_map):
    """Mean displacement of believed-occupied cells to the nearest true cell.

    The map registers obstacle cells at the cells it believes them to be; a
    drifted pose shifts every believed cell AWAY from the true obstacle
    surface.  This is the mean nearest-neighbour distance (in cells) from the
    believed-occupied set P to the true set T, sampled to 200 cells for speed
    (the truth set is unreachable for large worlds; the exact mean is not the
    point of an order-of-magnitude misregistration score).  A clean pose has
    the discretization baseline ~0.3 cells (ray-endpoint cells drawn on a
    curved surface); a drifting pose grows it to 1-3 cells.
    """
    believed = np.argwhere(grid > COVERAGE_THRESHOLD)
    true_cells = np.argwhere(true_map)
    if len(believed) == 0 or len(true_cells) == 0:
        return float("nan")
    sample = believed[:200]
    dist = ((sample[:, None, :] - true_cells[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(dist.min(axis=1)).mean())


def simulate_episode(group, obstacles, start_pose, config, *, record_extras=False):
    """One episode; groups are paired (identical obstacles/start per scene)."""
    if group not in GROUPS4:
        raise ValueError("group must be in T/O1/O2/F")
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
    snaps = {"grid": [], "pose": [], "est": [], "cov": [], "step": [], "err": []}
    rng = np.random.default_rng([config.seed, 4400])
    bias = config.encoder_biases[group]
    wheels_last = np.zeros(2)  # wheel-angle increments (rad) of the last move
    est_prev = pose.copy()  # s0 estimate = known initial pose (lesson-15 rule)
    observations_log = [] if record_extras else None
    for step in range(1, base.horizon_steps + 1):
        # --- truth frame: pose at the START of this step is s_{step-1}
        truth.append(pose.copy())
        # --- sensing is physical: rays emitted from the TRUE pose
        angles = pose[2] + np.arange(base.rays) * (2.0 * math.pi / base.rays)
        ranges, hit = cast_rays(pose[0], pose[1], angles, obstacles, walls, base.max_range_m)
        # --- estimator: same time index as truth (s_{step-1} estimate), built
        # from the wheel-angle increments of the move just finished
        est = pose.copy()
        if group != "T":
            obs_result = None
            if group == "F" and step % config.obs_period_steps == 0:
                readings = observe(pose, BEACONS, rng)
                obs_result = solve_pose(readings, BEACONS)
                if observations_log is not None:
                    observations_log.append({"step": step, "pose": obs_result.tolist()})
            est = estimate_step(est_prev, wheels_last, obs_result, bias)
        est_prev = est.copy()
        estimates.append(est.copy())
        # --- mapping at the ESTIMATED registration (drift misregisters the map)
        estimated_angles = est[2] + np.arange(base.rays) * (2.0 * math.pi / base.rays)
        update_grid(grid, est[:2], estimated_angles, ranges, hit, res, n)
        if record_extras and step in snap_steps and len(snaps["step"]) < 8:
            snaps["grid"].append(grid.copy())
            snaps["pose"].append(pose.copy())
            snaps["est"].append(est.copy())
            snaps["cov"].append(coverage_fraction(grid))
            snaps["step"].append(step)
            snaps["err"].append(float(np.linalg.norm(est[:2] - pose[:2])))
        # --- controller: A* on the believed registration, pursuit on belief
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
        if frontal_clear < base.brake_dist_m:
            wheels = wheels_from_twist(0.0, command_omega, base.wheel_limit_rad_s)
            brake_steps += 1
            force_replan = True
        wheels_last = wheels * DT  # cumulative angle increments for the estimator
        # --- physics: truth integration, truth collision, truth arrival
        candidate = step_pose(pose, wheels, DT)
        clearance = obstacle_clearance(candidate[:2], all_obstacles)
        if clearance < base.r_car_m:
            pressed += 1
            if step - last_collision >= base.collision_debounce_steps:
                collisions += 1
                last_collision = step
        else:
            pose = candidate
        min_clear = min(min_clear, obstacle_clearance(pose[:2], all_obstacles))
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
    map_fid = map_shift_cells(grid, true_occupancy(obstacles, walls, n, res))
    path_length = float(np.linalg.norm(np.diff(truth_array[:, :2], axis=0), axis=1).sum())
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
        "map_shift_cells": map_fid,
        "observations": len(observations_log or []),
    }
    extras = None
    if record_extras:
        extras = {
            "final_map": grid.copy(),
            "snapshots": {
                "grids": (
                    np.stack(snaps["grid"] + [snaps["grid"][-1]] * (8 - len(snaps["grid"])))
                    if snaps["grid"]
                    else np.zeros((8, n, n))
                ),
                "poses": (
                    np.stack(snaps["pose"] + [snaps["pose"][-1]] * (8 - len(snaps["pose"])))
                    if snaps["pose"]
                    else np.zeros((8, 3))
                ),
                "ests": (
                    np.stack(snaps["est"] + [snaps["est"][-1]] * (8 - len(snaps["est"])))
                    if snaps["est"]
                    else np.zeros((8, 3))
                ),
                "steps": (
                    np.array(
                        snaps["step"] + [snaps["step"][-1]] * (8 - len(snaps["step"])), dtype=int
                    )
                    if snaps["step"]
                    else np.zeros(8, dtype=int)
                ),
                "errs": (
                    np.array(snaps["err"] + [snaps["err"][-1]] * (8 - len(snaps["err"])))
                    if snaps["err"]
                    else np.zeros(8)
                ),
            },
            "replan_log": replan_log or [],
        }
    return metrics, truth_array, estimate_array, extras


def obstacle_clearance(point, obstacles):
    return min(obstacle.distance(point[0], point[1]) for obstacle in obstacles)


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
    "map_shift_cells",
)


def aggregate_episodes(rows):
    arrived = [row for row in rows if row["outcome"] == "arrived"]
    ratios = [row["path_ratio"] for row in arrived if row.get("path_ratio") is not None]
    times = [row["arrival_time_s"] for row in arrived if row.get("arrival_time_s") is not None]
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
    }


def expected_npz_keys(report):
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    keys = set()
    for group in GROUPS4:
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
            "snap_errs",
            "snap_steps",
            "replan_paths",
            "replan_steps",
            "replan_costs",
        )
    )
    return keys


# --------------------------------------------------------------- full run
def shortest_distance(obstacles, base):
    """A* shortest path length on the inflated true map (reference denominator)."""
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, log=print):
    """Full paired experiment; new directories only, bitwise repeatable."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    if config is None or config.__class__ is GridNavConfig:
        config = NavPoseErrorConfig(base=config) if config is not None else NavPoseErrorConfig()
    elif config.__class__ is not NavPoseErrorConfig:
        raise TypeError("config must be GridNavConfig or NavPoseErrorConfig")
    base = config.config
    started = time.perf_counter()
    walls = build_walls(base.world_size_m, base.res_m)
    scenarios = build_scenarios(base, seed)
    n = base.grid_cells
    scene_count = len(scenarios)
    showcase_scene = 1 if base.obstacle_scenes >= 1 else 0

    log(f"scene 0/{scene_count}: empty control + {base.obstacle_scenes} obstacle scenes")
    rows = []
    truths = {g: {} for g in GROUPS4}
    estimates = {g: {} for g in GROUPS4}
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
            for group in GROUPS4:
                record = group == "F" and init == 0  # every scene keeps a map
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
    for group in GROUPS4:
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
                        "label": f"{NAV_GROUPS[group]}·{'有障碍' if condition == 'obstacle' else '无障碍'}",
                        **aggregates[group][condition],
                    }
                )

    t_obs = aggregates["T"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    o1_obs = aggregates["O1"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    o2_obs = aggregates["O2"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    f_obs = aggregates["F"].get(
        "obstacle", {"arrivals": 0, "episodes": 1, "collision_events_total": 0}
    )
    hypothesis = {
        "claim": "定位误差穿栈：到达率随里程计偏差单调下降；地标融合同偏差位姿下恢复；有界估计的剩余代价被量化",
        "criteria": {
            "gate": "T 组有障碍 15/15 到达且 0 碰撞（第 43 课 B 组复现 = 健全性闸门）",
            "monotone": "O1 有障碍到达 >= O2 有障碍到达（偏差越大越差）",
            "fusion": "F 有障碍到达 > O2 有障碍到达 且 F 碰撞事件 <= O2 碰撞事件（安全性不劣化）",
            "residual": "F 与 T 的到达数/路径比差距量化（有界估计的剩余代价）",
        },
        "results": {
            "gate_met": bool(
                t_obs["arrivals"] == t_obs["episodes"] and t_obs["collision_events_total"] == 0
            ),
            "monotone_met": bool(o1_obs["arrivals"] >= o2_obs["arrivals"]),
            "fusion_met": bool(
                f_obs["arrivals"] > o2_obs["arrivals"]
                and f_obs["collision_events_total"] <= o2_obs["collision_events_total"]
            ),
            "counts": {
                "T": t_obs["arrivals"],
                "O1": o1_obs["arrivals"],
                "O2": o2_obs["arrivals"],
                "F": f_obs["arrivals"],
            },
            "collisions": {
                "T": t_obs["collision_events_total"],
                "O1": o1_obs["collision_events_total"],
                "O2": o2_obs["collision_events_total"],
                "F": f_obs["collision_events_total"],
            },
        },
    }

    output.mkdir(parents=True, exist_ok=False)
    archive = {}
    metric_arrays = {}
    for group in GROUPS4:
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
            "map_shift_cells": row["map_shift_cells"],
        }
        for name, number in value.items():
            if number is not None and math.isfinite(number):
                metric_arrays[f"{row['group']}_{name}"][row["scene"], row["init"]] = number
    archive.update(metric_arrays)
    for group in GROUPS4:
        for key, array in truths[group].items():
            archive[f"truth_{group}_{key}"] = array
        for key, array in estimates[group].items():
            archive[f"estimates_{group}_{key}"] = array
    for true_map in (true_occupancy(sc["obstacles"], walls, n, base.res_m) for sc in scenarios):
        pass  # recomputed below for the archive keys
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
        archive["snap_errs"] = snaps["errs"].astype(float)
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
        archive["snap_errs"] = np.zeros(8)
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
            "groups": {g: NAV_GROUPS[g] for g in GROUPS4},
            "encoder_biases": config.encoder_biases,
            "beacons": BEACONS.tolist(),
            "beacon_note": "四角固定信标 = 纯观测目标（无碰撞、无遮蔽，第 18 课理想模型）",
            "obs_period_steps": config.obs_period_steps,
            "obs_range_std_m": config.obs_range_std_m,
            "obs_bearing_std_rad": config.obs_bearing_std_rad,
            "known_initial_pose": True,
            "estimator_note": (
                "初始位姿已知（第 15 课惯例）；里程计链 = 真值累计轮角 × 右轮比例偏差；"
                "融合 = 观测重置 + 里程计传播（第 19 课规则，与 pose_fusion.estimate_with_resets 逐位一致）；"
                "估计进入每一层（建图原点/规划/追踪），射线物理发射于真值位姿"
            ),
            "collision_caliber": "碰撞 = 试探位姿与障碍距离 < 车半径（刚性退回，事件间隔 >= 1 s 去抖）",
            "arrival_caliber": "到达 = 第 39 课口径（真值几何：距目标 < 5 cm 且速度 < 0.1 m/s）",
            "unknown_policy": "optimistic（未知格可通行；逐步路径失效检查 + 正前方刹车反射）",
            "showcase": {"group": "F", "scene": showcase_scene, "init": 0},
            "source_note": "场景/初态/传感器/控制器逐字段复用第 43 课（grid_nav.py import，零改写）",
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
                "landmark_localization.py",
                "pose_fusion.py",
                "plotting.py",
                "experiments/grid_nav.py",
                "experiments/nav_pose_error.py",
                "nav_pose_error_demo.py",
            )
        },
        "limits": [
            "位姿真值只用于评估（到达/碰撞/误差曲线），不进入任何估计器；初始位姿已知",
            "地标为理想无遮蔽测距测角模型（第 18 课），无身份匹配问题；真实地标感知不在本课范围",
            "仅右轮比例偏差（固定 0/1/2%），无打滑、无逐区间噪声（第 17 课噪声作为正交变量未联合）",
            "建图错位以'已知格中真值占占据的比例'（map_fidelity）度量，不估算位姿-地图联合后验（SLAM 是下一课）",
            "单点配置不变（膨胀 3 格/刹车 0.2207 m/1 s 重规划）；参数敏感性见 docs/48 §6 与开放 issue",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_comparison(output / "comparison.png", report)
    save_scenario_maps(output / "scenario_maps.png", report, archive)
    return report


# -------------------------------------------------------------------- figures
def save_comparison(path, report):
    configure_plot_font()
    aggregates = report["aggregates"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), layout="constrained")
    conditions = [("obstacle", "有障碍（15 回合/组）"), ("empty", "无障碍（3 回合/组）")]
    colors = {"T": "#0f766e", "O1": "#d97706", "O2": "#b91c1c", "F": "#2563eb"}
    width = 0.4
    for ax, key, name in (
        (axes[0, 0], "arrival_rate", "到达率（< 5 cm 且 < 0.1 m/s，真值口径）"),
        (axes[0, 1], "collision_events_mean", "平均碰撞事件/回合"),
        (axes[1, 0], "mean_position_error_m", "平均位置误差（估计 vs 真值）"),
        (axes[1, 1], "map_shift_cells", "地图占据格平均错位（格）"),
    ):
        for offset, group in enumerate(GROUPS4):
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
                label=NAV_GROUPS[group],
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
        for group, color in (("T", "#0f766e"), ("O2", "#b91c1c"), ("F", "#2563eb")):
            ax.plot(
                archive[f"truth_{group}_{scene}_0"][:, 0],
                archive[f"truth_{group}_{scene}_0"][:, 1],
                color=color,
                linewidth=1.1,
                label=NAV_GROUPS[group],
            )
        start = archive[f"truth_T_{scene}_0"][0]
        goal = report["protocol"]["goal_xy"]
        beacons = np.asarray(report["protocol"]["beacons"])
        ax.plot(start[0], start[1], "s", color="#111827", markersize=5, linestyle="none")
        ax.plot(goal[0], goal[1], "*", color="#15803d", markersize=12, linestyle="none")
        ax.plot(beacons[:, 0], beacons[:, 1], "^", color="#7c3aed", markersize=5, linestyle="none")
        obstacles = report["scenarios"][scene]["obstacles"]
        title = "无障碍（对照）" if not obstacles else f"{len(obstacles)} 个随机障碍"
        ax.set(
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"场景 {scene}（{title}）：三组轨迹（同初态；紫三角=信标）",
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
            args.output, seed=args.seed, config=NavPoseErrorConfig(base=base), log=log
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
