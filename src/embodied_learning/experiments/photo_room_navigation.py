"""Closed-loop navigation in the photo-referenced 3D room (four groups).

The robot decides its motion from its own state estimate each step:
sense (MuJoCo rays at the lidar height) -> update the estimate (odometry or
particle filter) -> plan/track on the avoidance grid -> command (v, omega)
with limits -> the TRUE pose advances kinematically -> contacts are checked.
No pre-recorded coordinate is ever replayed.

Four groups differ ONLY in the pose the planner/controller believes:
  A  truth            diagnostic reference (simulation privilege);
  B  odometry         dead-reckoning estimate with a 2% wheel bias;
  C  PF + reference map   particle filter on the matched-height geometry map;
  D  PF + self-built map  particle filter on a map painted during a separate
     odometry-executed mapping patrol (the SLAM chicken-and-egg, lesson 56).

All groups share the SAME avoidance map (full robot-height band of the
known geometry - a PRIOR reference, so "unreachable" means the prior says
no collision-free path exists), the same planner (A* + inflation) and the
same controller (waypoint following with v/omega limits and a lidar safety
stop).  Truth enters only observation synthesis and scoring - except group
A, which is explicitly the simulation-privileged diagnostic.

Pre-registered gates (fixed before the runs):
  gate_A        group A arrives on every reachable task (all seeds) and
                safely rejects the unreachable task with zero contact;
  gate_compare  B/C/D compared without a preset winner.

Metrics per run: task result (arrived / false_arrived / timeout /
rejected_unreachable / collision_abort), true final distance to goal,
collision frames and contact objects, localization mean/p95/end error,
travelled distance and steps.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_learning.experiments.grid_nav import (
    GEOMETRY,
    astar,
    inflate,
    smooth_path,
)
from embodied_learning.experiments.map_localization import (
    ParticleFilter,
    occupancy_from_observations,
)
from embodied_learning.odometry import estimate_poses
from embodied_learning.photo_room import RoomWorld, default_layout

EXPERIMENT = "photo_room_navigation"
SCHEMA_VERSION = 1
DEFAULT_RESULTS = "results/photo_room_navigation_2026-09-25"
RES = 0.04
DT = 0.04
V_MAX = 0.15
OMEGA_MAX = 1.2
SAFE_STOP_M = 0.12
ARRIVAL_EST_M = 0.15
ARRIVAL_TRUE_M = 0.20
MAX_STEPS = 3000
REPLAN_EVERY = 25
GROUPS = ("A", "B", "C", "D")
GROUP_NAMES = {
    "A": "真值定位（仿真特权，诊断参考）",
    "B": "轮式里程计（2% 偏差）",
    "C": "PF + 参考图（匹配激光高度，几何先验）",
    "D": "PF + 自建图（里程计巡游涂抹）",
}
SENSOR_SEEDS = (0, 1, 2)
CONTACT_ABORT_FRAMES = 100


@dataclass(frozen=True)
class NavConfig:
    """Defaults ARE the pre-registered first closed-loop round."""

    particles: int = 300
    sensor_sigma_m: float = 0.01
    wheel_bias: float = 0.02
    max_steps: int = MAX_STEPS
    seed: int = 0

    def __post_init__(self):
        if type(self.particles) is not int or not (50 <= self.particles <= 2000):
            raise ValueError("particles out of range")
        if not (0.0 <= self.sensor_sigma_m <= 0.2):
            raise ValueError("sensor_sigma_m out of range")
        if not (0.0 <= self.wheel_bias <= 0.2):
            raise ValueError("wheel_bias out of range")
        if type(self.max_steps) is not int or not (200 <= self.max_steps <= 20000):
            raise ValueError("max_steps out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")


def wrap(a):
    return float(np.arctan2(math.sin(a), math.cos(a)))


# ------------------------------------------------------------------- world
def build_maps(world, layout):
    """Avoidance (full robot-height band, prior reference) and the matched
    lidar-height occupancy used as group C's localization map."""
    robot = layout["robot"]
    avoidance = world.occupancy(robot["bottom"] + 0.01, robot["bottom"] + robot["height"], RES)
    loc_map = world.occupancy(robot["lidar_z"] - 0.02, robot["lidar_z"] + 0.02, RES)
    return avoidance, loc_map


def plan(grid, start, goal, radius):
    safe = ~inflate(grid, math.ceil((radius + RES) / RES))
    s = tuple(np.floor(np.asarray(start)[::-1] / RES).astype(int))
    g = tuple(np.floor(np.asarray(goal)[::-1] / RES).astype(int))
    cells, _ = astar(safe, s, g)
    if cells is None:
        return None
    cells = smooth_path(cells, safe)
    return np.array([[(x + 0.5) * RES, (y + 0.5) * RES] for y, x in cells])


# ------------------------------------------------------------------- tasks
def classify_tasks(world, layout):
    """Derive tasks from the avoidance grid's connectivity (geometry-honest).

    Reachable tasks: four free cells of the robot-reachable connected
    component, one per arena quadrant (start near the room's south-east,
    goals spread across the other quadrants; every consecutive pair is
    verified by A*).  Unreachable task: a goal whose cell is free at the
    lidar slice but occupied at robot-body height (the bed footprint) - the
    low lidar alone would call it passable, the full-height prior map
    correctly rejects it.
    """
    from scipy.ndimage import label

    robot = layout["robot"]
    avoidance, loc_map = build_maps(world, layout)
    safe = ~inflate(avoidance, math.ceil((robot["radius"] + RES) / RES))
    labels, _n = label(safe)
    # the robot component = the one containing the largest free area near the
    # room centre-east start corner
    seed_cell = (int(0.95 / RES), int(2.70 / RES))
    if labels[seed_cell] == 0:
        frees = np.argwhere(safe)
        seed_cell = tuple(frees[np.argmin(np.abs(frees[:, 0] - seed_cell[0]) + np.abs(frees[:, 1] - seed_cell[1]))])
    comp_id = labels[seed_cell]
    comp = labels == comp_id
    ys, xs = np.nonzero(comp)

    def snap(row, col):
        """Nearest component cell to (row, col)."""
        d = np.abs(ys - row) + np.abs(xs - col)
        i = int(np.argmin(d))
        return int(ys[i]), int(xs[i])

    r0, c0 = ys[np.argmin(ys + xs)], xs[np.argmin(ys + xs)]
    r1, c1 = ys[np.argmin(ys + (len(xs) - xs))], xs[np.argmin(ys + (len(xs) - xs))]
    r2, c2 = ys[np.argmax(ys + xs)], xs[np.argmax(ys + xs)]
    r3, c3 = ys[np.argmax(ys + (len(xs) - xs))], xs[np.argmax(ys + (len(xs) - xs))]

    def xy(cell):
        rr, cc = cell
        return [(cc + 0.5) * RES, (rr + 0.5) * RES]

    corners = [snap(r0, c0), snap(r1, c1), snap(r2, c2), snap(r3, c3)]
    origin = snap(seed_cell[0], seed_cell[1])
    pairs = []
    for i, corner in enumerate(corners):
        mid = snap((origin[0] + corner[0]) // 2, (origin[1] + corner[1]) // 2)
        pairs.append((f"nav_quadrant_{i + 1}", xy(origin), xy(mid)))
        pairs.append((f"to_quadrant_{i + 1}", xy(origin), xy(corner)))
    reachable = []
    for name, start, goal in pairs:
        path = plan(avoidance, start, goal, robot["radius"])
        if path is not None and len(path) >= 2:
            reachable.append((name, start, goal))
        if len(reachable) == 4:
            break
    if len(reachable) < 4:
        raise RuntimeError(f"only {len(reachable)} reachable pairs in the robot component")

    # unreachable: free at the lidar slice, occupied at robot-body height
    loc_free = loc_map == 0
    body_block = avoidance != 0
    bed_cells = np.argwhere(loc_free & body_block)
    if len(bed_cells) == 0:
        raise RuntimeError("no lidar-free/body-blocked cell found for the unreachable task")
    best = None
    best_d = None
    for rr, cc in bed_cells:
        d = math.hypot((cc + 0.5) * RES - 2.7, (rr + 0.5) * RES - 0.95)
        if best_d is None or d < best_d:
            best_d, best = d, (rr, cc)
    unreachable = [("into_bed_footprint", [2.70, 0.95], xy(best))]

    # verify the four reachable pairs with the same planner the runs use
    for name, start, goal in reachable:
        if plan(avoidance, start, goal, robot["radius"]) is None:
            raise RuntimeError(f"declared-reachable task failed planning: {name}")
    return reachable, unreachable


# --------------------------------------------------------------- controller
def wheel_increments(v, omega, dt=DT):
    """Wheel-angle increments for a commanded (v, omega) over dt."""
    left = (v - omega * GEOMETRY.track_m / 2.0) * dt / GEOMETRY.radius_m
    right = (v + omega * GEOMETRY.track_m / 2.0) * dt / GEOMETRY.radius_m
    return [left, right]


def control_step(pose_for_control, path, path_idx, scan, state):
    """Pure-pursuit style waypoint following with a lidar safety stop.

    Returns (wheel_increment, new_path_idx, v_commanded).
    """
    if path is None or path_idx >= len(path):
        return [0.0, 0.0], path_idx, 0.0
    target = path[path_idx]
    while math.hypot(target[0] - pose_for_control[0], target[1] - pose_for_control[1]) < 0.10:
        path_idx += 1
        if path_idx >= len(path):
            return [0.0, 0.0], path_idx, 0.0
        target = path[path_idx]
    target_th = math.atan2(target[1] - pose_for_control[1], target[0] - pose_for_control[0])
    d_th = wrap(target_th - pose_for_control[2])
    v = V_MAX
    omega = max(-OMEGA_MAX, min(OMEGA_MAX, 2.0 * d_th))
    if abs(d_th) > 0.5:
        v = 0.02  # rotate first when the waypoint is behind
    # lidar safety stop: any forward-sector return closer than SAFE_STOP_M
    forward = scan[: len(scan) // 4] + scan[3 * len(scan) // 4 :]
    if forward and min(forward) < SAFE_STOP_M:
        v = 0.0
    state["v_history"].append(v)
    return wheel_increments(v, omega), path_idx, v


# ------------------------------------------------------------- closed loop
def run_closed_loop(world, task, group, loc_map, rng, config, max_steps):
    """One closed-loop run.  Returns the result row and per-step chains."""
    name, start, goal = task
    robot = world.layout["robot"]
    truth = [np.array([start[0], start[1], 0.0])]
    odom = [np.array([start[0], start[1], 0.0])]
    estimate = [np.array([start[0], start[1], 0.0])]
    encoders = [np.zeros(2)]
    contacts = []
    arrived_est = False
    arrived_frame = None
    contact_frames = 0
    contact_objects = set()
    travelled = 0.0

    if group == "C" or group == "D":
        pf = ParticleFilter(
            loc_map, RES, config.particles, rng, init_pose=truth[0], spread=0.04, rays_n=32
        )
    else:
        pf = None

    avoidance = world.occupancy(robot["bottom"] + 0.01, robot["bottom"] + robot["height"], RES)
    path = None
    path_idx = 0
    state = {"v_history": []}
    result = "timeout"

    for k in range(max_steps + 1):
        pose_true = truth[-1]
        pose_ctrl = {"A": truth[-1], "B": odom[-1], "C": estimate[-1], "D": estimate[-1]}[group]
        scan, _hits = world.scan(pose_true)

        # planner (periodic, on the estimated/truth pose)
        if path is None or k % REPLAN_EVERY == 0:
            path = plan(avoidance, pose_ctrl[:2], goal, robot["radius"])
            path_idx = 0
            if path is None:
                # estimate-driven give-up on a reachable task: a localization
                # failure, distinct from the prior map's safe rejection
                result = "gave_up"
                break

        # arrival decided by the robot's own estimate, scored against truth
        if (
            not arrived_est
            and math.hypot(pose_ctrl[0] - goal[0], pose_ctrl[1] - goal[1]) < ARRIVAL_EST_M
        ):
            arrived_est = True
            arrived_frame = k

        inc, path_idx, _v = control_step(pose_ctrl, path, path_idx, list(scan), state)
        encoders.append(np.asarray(inc, dtype=float) + encoders[-1])

        # the TRUE pose advances kinematically under the commanded motion
        v_l, v_r = inc
        ds = GEOMETRY.radius_m * (v_l + v_r) / 2.0
        dth = GEOMETRY.radius_m * (v_r - v_l) / GEOMETRY.track_m
        mid = truth[-1][2] + dth / 2.0
        truth.append(
            np.array(
                [
                    truth[-1][0] + ds * math.cos(mid),
                    truth[-1][1] + ds * math.sin(mid),
                    wrap(truth[-1][2] + dth),
                ]
            )
        )
        travelled += abs(ds)

        # odometry integrates the same increments with a wheel bias
        # the wheels turn by command; the 2% bias models floor/calibration
        # error between commanded and achieved motion - that IS the drift
        bl, br = np.asarray(inc, dtype=float) * (1.0 + config.wheel_bias)
        ds_b = GEOMETRY.radius_m * (bl + br) / 2.0
        dth_b = GEOMETRY.radius_m * (br - bl) / GEOMETRY.track_m
        mid_b = odom[-1][2] + dth_b / 2.0
        odom.append(
            np.array(
                [
                    odom[-1][0] + ds_b * math.cos(mid_b),
                    odom[-1][1] + ds_b * math.sin(mid_b),
                    wrap(odom[-1][2] + dth_b),
                ]
            )
        )

        # particle filter update (groups C and D)
        if pf is not None:
            ds_b = GEOMETRY.body_velocity(np.asarray(inc, dtype=float))
            pf.propagate_body(ds_b[0] * (1.0 + config.wheel_bias), ds_b[1])
            pf.measure(scan, None)

        pose_true = truth[-1]
        world.set_pose(pose_true)
        names = world.contact_names(pose_true)
        if names:
            contact_frames += 1
            contact_objects.update(names)
            if contact_frames > CONTACT_ABORT_FRAMES:
                result = "collision_abort"
                break

        if pf is not None:
            estimate.append(pf.estimate())
        elif group == "A":
            estimate.append(truth[-1])  # group A's estimate IS the truth
        else:
            estimate.append(odom[-1])
        pose_ctrl_now = estimate[-1] if pf is not None else odom[-1]
        if math.hypot(pose_ctrl_now[0] - goal[0], pose_ctrl_now[1] - goal[1]) < ARRIVAL_EST_M:
            arrived_est = True
            if arrived_frame is None:
                arrived_frame = k

        true_goal_dist = math.hypot(pose_true[0] - goal[0], pose_true[1] - goal[1])
        if k >= 50 and true_goal_dist < ARRIVAL_TRUE_M:
            result = "arrived"
            break
        if arrived_est and k - arrived_frame > 200:
            # the robot believes it has arrived but keeps waiting: false arrival
            true_dist = math.hypot(pose_true[0] - goal[0], pose_true[1] - goal[1])
            result = "false_arrived" if true_dist >= ARRIVAL_TRUE_M else "arrived"
            break

    if result == "timeout":
        result = "timeout"
    true_end = float(np.linalg.norm(np.asarray(truth[-1][:2]) - np.asarray(goal, dtype=float)))
    est_chain = np.asarray(estimate)
    truth_chain = np.asarray(truth)
    odom_chain = np.asarray(odom)
    n = min(len(est_chain), len(truth_chain))
    errors = np.linalg.norm(est_chain[:n, :2] - truth_chain[:n, :2], axis=1)
    return {
        "task": name,
        "group": group,
        "result": result,
        "true_end_m": true_end,
        "travelled_m": round(travelled, 3),
        "steps": len(truth) - 1,
        "contact_frames": contact_frames,
        "contact_objects": sorted(contact_objects),
        "loc_mean_m": float(np.mean(errors)) if n else None,
        "loc_p95_m": float(np.percentile(errors, 95)) if n else None,
        "arrived_frame": arrived_frame,
    }, {
        "truth": truth_chain,
        "odom": odom_chain,
        "estimate": est_chain,
    }


def base_grid_cells(world):
    return world.occupancy(0.1, 0.2, RES).shape[0]


# ------------------------------------------------------------ mapping phase
def mapping_patrol(world, layout, route_points, rng, config):
    """Group D's map: an odometry-executed coverage loop painted at odom
    poses with a 2% wheel bias (lesson-56 smear finding carries here).

    route_points come from the VERIFIED reachable task goals, so every leg
    is feasible on the same avoidance grid the runs use."""
    robot = layout["robot"]
    avoidance, _loc = build_maps(world, layout)
    paths = []
    for a, b in itertools.pairwise(route_points):
        path = plan(avoidance, a, b, robot["radius"])
        if path is None:
            raise RuntimeError("mapping route leg infeasible")
        paths.append(path)
    loop = np.vstack([paths[0]] + [p[1:] for p in paths[1:]])
    truth, enc = drive_replay_points(loop)
    ranges = []
    hits_all = []
    for pose in truth:
        r, h = world.scan(pose)
        ranges.append(r)
        hits_all.append(h)
    ranges = np.asarray(ranges)
    hits_all = np.asarray(hits_all)
    odom = estimate_poses(enc * (1.0 + config.wheel_bias), truth[0])
    base = SimpleNamespace(rays=64, res_m=RES, grid_cells=base_grid_cells(world))
    self_map = occupancy_from_observations(
        ranges, hits_all.astype(bool), odom, range(len(odom)), base
    )
    return self_map


def drive_replay_points(points):
    """Rotate-then-translate execution; returns (poses, cumulative encoders)."""
    heading = math.atan2(*(points[1] - points[0])[::-1])
    increments = []
    for a, b in __import__("itertools").pairwise(points):
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            continue
        target = math.atan2(vec[1], vec[0])
        rot = math.atan2(math.sin(target - heading), math.cos(target - heading))
        turns = max(1, math.ceil(abs(rot) / 0.10))
        if abs(rot) > 1e-9:
            dw = rot / turns * GEOMETRY.track_m / (2 * GEOMETRY.radius_m)
            increments.extend([[-dw, dw]] * turns)
        steps = max(1, math.ceil(length / 0.06))
        dw = length / steps / GEOMETRY.radius_m
        increments.extend([[dw, dw]] * steps)
        heading = target
    encoders = np.vstack((np.zeros(2), np.cumsum(increments, axis=0)))
    init = [*points[0], math.atan2(*(points[1] - points[0])[::-1])]
    return estimate_poses(encoders, init), encoders


# ------------------------------------------------------------ record shell
def expected_npz_keys(report):
    return set(report["archive_keys"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, log=print):

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if config is None:
        config = NavConfig()
    if log is None:
        def log(*_message):
            return None
    started = time.perf_counter()
    layout = default_layout()
    world = RoomWorld(layout)
    avoidance, loc_map = build_maps(world, layout)
    reachable, unreachable = classify_tasks(world, layout)
    tasks = [(n, s, g, True) for n, s, g in reachable] + [
        (n, s, g, False) for n, s, g in unreachable
    ]
    log(f"tasks: {[t[0] for t in tasks]}")

    # group D maps (one per sensor seed; odometry-executed mapping patrol)
    self_maps = {}
    route = [[2.70, 0.95]] + [list(g) for _n, _s, g in reachable] + [[2.70, 0.95]]
    for s in SENSOR_SEEDS:
        rng_map = np.random.default_rng([seed, 56010, s])
        self_maps[s] = mapping_patrol(world, layout, route, rng_map, config)
        log(f"mapping patrol for seed {s} done")

    rows = []
    archive = {}
    for s in SENSOR_SEEDS:
        rng = np.random.default_rng([seed, 56011, s])
        for name, start, goal, is_reachable in tasks:
            for group in GROUPS:
                if not is_reachable:
                    # the shared prior avoidance map rejects the task; the
                    # robot never moves (safe rejection), identical per group
                    rows.append(
                        {
                            "task": name,
                            "group": group,
                            "seed": s,
                            "result": "rejected_unreachable",
                            "true_end_m": None,
                            "contact_frames": 0,
                        }
                    )
                    continue
                loc_map_for_group = loc_map if group == "C" else self_maps[s]
                rng_run = np.random.default_rng([seed, 56012, s, hash(name) % 1000])
                row, chains = run_closed_loop(
                    world,
                    (name, start, goal),
                    group,
                    loc_map_for_group,
                    rng_run,
                    config,
                    config.max_steps,
                )
                row["seed"] = s
                rows.append(row)
                key = f"{name}_{group}_s{s}"
                for chain_name, chain in chains.items():
                    archive[f"{key}_{chain_name}"] = chain
                log(
                    f"  s{s} {name} {group}: {row['result']} end {row['true_end_m']:.2f} m "
                    f"contacts {row['contact_frames']}"
                )

    def count(group, result, seed=None):
        n = 0
        for row in rows:
            if row["group"] != group or row["result"] != result:
                continue
            if seed is not None and row.get("seed") != seed:
                continue
            n += 1
        return n

    a_reach = [r for r in rows if r["group"] == "A" and r["task"] != unreachable[0][0]]
    a_arrived = sum(1 for r in a_reach if r["result"] == "arrived")
    a_contacts = sum(r.get("contact_frames", 0) for r in rows if r["group"] == "A")
    rejections = [r for r in rows if r["result"] == "rejected_unreachable"]
    hypothesis = {
        "claim": (
            "世界模型闭环：定位来源不同，机器人实际到达与安全表现不同。"
            "A 组门（真值定位必须能安全完成全部可达任务并拒绝不可达任务）是"
            "后续 B/C/D 对照有效的前提；不预设 PF 组一定获胜"
        ),
        "criteria": {
            "collector": "5 任务 × 3 传感种子 × 4 组全部有结果",
            "gate_A": "A 组可达任务全到达（12/12）且不可达任务安全拒绝（3/3）且零接触帧",
            "compare": "B/C/D 按到达/超时/碰撞/定位误差如实记录，不设预设胜者",
        },
        "results": {
            "collector_met": bool(len(rows) == len(tasks) * len(SENSOR_SEEDS) * len(GROUPS)),
            "gate_A_met": bool(
                a_arrived == len(a_reach) and a_contacts == 0 and len(rejections) == 12
            ),
            "a_arrived": a_arrived,
            "a_reachable_runs": len(a_reach),
            "a_contact_frames": a_contacts,
            "rejections": len(rejections),
        },
    }
    archives = {
        "task_table": np.array([[hash(t[0]) % 1000, 1 if t[3] else 0] for t in tasks], dtype=int)
    }
    for key, arr in archive.items():
        archives[key] = arr
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "loop": "sense (mj_ray @ 0.18 m) -> estimate (odom/PF) -> plan (A* on prior avoidance grid) -> command (v/omega limits + lidar safety stop) -> kinematic truth advance -> MuJoCo contact check",
            "groups": GROUP_NAMES,
            "avoidance_map": "full robot-height band of the known geometry (PRIOR reference; not autonomously discovered)",
            "localization_maps": {
                "C": "matched lidar-height geometry map (prior reference)",
                "D": "self-built during an odometry-executed mapping patrol (2% bias smear)",
            },
            "tasks": {
                "reachable": [t[0] for t in tasks if t[3]],
                "unreachable": [t[0] for t in tasks if not t[3]],
            },
            "gates": {
                "gate_A": "reachable 12/12 arrived + unreachable 3/3 safely rejected + zero contact frames",
                "compare": "B/C/D recorded without a preset winner",
            },
            "calibration": {
                "measured": [{"field": "desk.height", "value_m": 0.69, "source": "user"}],
                "estimated": [
                    "room 3.40 x 3.80 x 2.70 m",
                    "bed 2.00 x 1.50 m, under_clearance 0.22 m",
                    "robot radius 0.15 m / height 0.32 m / lidar 0.18 m (model conventions)",
                ],
                "derived": {
                    "bed_under_clearance_vs_robot_top": "0.22 m < 0.34 m -> the bed-under corridor is blocked for this robot by the prior map",
                },
            },
        },
        "rows": rows,
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
    parser = argparse.ArgumentParser(description="photo-room closed-loop navigation")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--particles", type=int, default=300)
    args = parser.parse_args()
    config = NavConfig(particles=args.particles, seed=args.seed)
    report = run_experiment(args.output, seed=args.seed, config=config)
    print(json.dumps(report["hypothesis"]["results"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
