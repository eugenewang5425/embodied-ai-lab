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
import zlib
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
    alpha_trans: float = 0.10
    alpha_trans_base_m: float = 0.002
    alpha_rot: float = 0.10
    alpha_rot_base_rad: float = 0.002
    announce_frames: int = 10

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
        for _n in ("alpha_trans", "alpha_rot"):
            if not (0.0 <= getattr(self, _n) <= 1.0):
                raise ValueError(f"{_n} out of range")
        for _n in ("alpha_trans_base_m", "alpha_rot_base_rad"):
            if not (0.0 <= getattr(self, _n) <= 0.5):
                raise ValueError(f"{_n} out of range")
        if type(self.announce_frames) is not int or not (1 <= self.announce_frames <= 200):
            raise ValueError("announce_frames out of range")
        for name in ("alpha_trans", "alpha_rot"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} out of range")
        for name in ("alpha_trans_base_m", "alpha_rot_base_rad"):
            if not (0.0 <= getattr(self, name) <= 0.5):
                raise ValueError(f"{name} out of range")
        if type(self.announce_frames) is not int or not (1 <= self.announce_frames <= 200):
            raise ValueError("announce_frames out of range")


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
        seed_cell = tuple(
            frees[
                np.argmin(np.abs(frees[:, 0] - seed_cell[0]) + np.abs(frees[:, 1] - seed_cell[1]))
            ]
        )
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


def control_step(pose_for_control, path, path_idx, scan, stop_distance):
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
    # lidar safety stop: any forward-sector return closer than stop_distance
    forward = np.concatenate([scan[: len(scan) // 4], scan[3 * len(scan) // 4 :]])
    if forward.size and float(np.min(forward)) < stop_distance:
        v = 0.0
    return wheel_increments(v, omega), path_idx, v


# ------------------------------------------------------------- closed loop
def propagate_motion(pf, ds, dth, rng, sig_trans, sig_rot):
    """Motion-model propagation with noise SCALED BY THE MOTION.

    sig_trans/sig_rot are computed by the caller from this step's measured
    translation and rotation: a 6 mm step must not inject a 5 cm kick (the
    recorded v1 finding - fixed per-step noise masqueraded as PF jitter).
    """
    n = len(pf.p)
    trans = rng.normal(0.0, sig_trans, n)
    rot = rng.normal(0.0, sig_rot, n)
    mid = pf.p[:, 2] + dth / 2.0
    pf.p[:, 0] += (ds + trans) * np.cos(mid)
    pf.p[:, 1] += (ds + trans) * np.sin(mid)
    pf.p[:, 2] = np.arctan2(np.sin(pf.p[:, 2] + dth + rot), np.cos(pf.p[:, 2] + dth + rot))


def measure_recursive(pf, scan, prev_w):
    """Recursive measurement update: posterior is proportional to
    prev_w * likelihood (the shared ParticleFilter.measure normalizes the
    likelihood alone, which would forget history - the recorded v1 finding)."""
    pf.measure(scan, None)
    likelihood = pf.w.copy()
    posterior = prev_w * likelihood
    total = posterior.sum()
    if total <= 1e-300:
        posterior = np.full(len(posterior), 1.0 / len(posterior))
    pf.w = posterior / posterior.sum()
    return pf.w


def noisy_scan(world, pose, rng, sigma):
    raw, hits = world.scan(pose)
    if sigma > 0.0:
        raw = np.clip(raw + rng.normal(0.0, sigma, raw.shape), 0.02, 3.95)
    return raw, hits


def safe_stop_distance(layout, v_max=V_MAX, dt=DT, margin=0.04):
    """Body radius + one control period of motion + margin."""
    return layout["robot"]["radius"] + v_max * dt + margin


def run_closed_loop(world, task, group, loc_map, rng, config, max_steps, layout=None):
    """One corrected-protocol closed-loop run.

    Timestamps: iteration k starts with the robot at truth[k].  Order per
    step: (1) propagate the filter with the last measured wheel increment;
    (2) take the scan at time k and run the RECURSIVE measurement update;
    (3) read the estimate, plan periodically, and announce arrival only from
    the estimate; (4) execute the command - truth/odom advance to k+1 and
    contacts are checked.  Scans, estimates, commands, ESS and resample
    events are recorded per frame (the v1 misalignment finding).
    """
    if layout is None:
        layout = world.layout
    name, start, goal = task
    robot = layout["robot"]
    truth = [np.array([start[0], start[1], 0.0])]
    odom = [np.array([start[0], start[1], 0.0])]
    estimate = []  # appended in the loop at each timestamp k
    encoders = [np.zeros(2)]
    pf = (
        ParticleFilter(
            loc_map, RES, config.particles, rng, init_pose=truth[0], spread=0.04, rays_n=32
        )
        if group in ("C", "D")
        else None
    )
    weights = np.full(config.particles, 1.0 / config.particles) if pf is not None else None
    stop_distance = safe_stop_distance(layout)
    scan_trace = []
    commands = []
    ess_trace = []
    resample_frames = []
    contact_flags = []
    contact_objects = set()
    first_contact = None
    contact_frames = 0
    announce_frame = None
    announce_true = None
    no_path_frame = None
    min_true_goal = math.inf
    travelled = 0.0
    streak = 0
    result = "timeout"
    end_k = max_steps

    avoidance, _loc = build_maps(world, layout)
    path = None
    wp = 0
    state = {}
    stop_dist = safe_stop_distance(layout)
    ds_prev = 0.0
    dth_prev = 0.0

    for k in range(max_steps + 1):
        pose_true = truth[k]

        # (1) filter propagation with the last EXECUTED (measured, biased)
        #     wheel increment: rad/s = increment / DT, then dt = DT
        if k > 0 and pf is not None:
            prev_cmd = commands[k - 1]
            measured = np.asarray(prev_cmd, dtype=float) * (1.0 + config.wheel_bias)
            v_m, om_m = GEOMETRY.body_velocity(measured / DT)
            sig_trans = config.alpha_trans * abs(ds_prev) + config.alpha_trans_base_m
            sig_rot = config.alpha_rot * abs(dth_prev) + config.alpha_rot_base_rad
            propagate_motion(pf, ds_prev, dth_prev, rng, sig_trans, sig_rot)

        # (2) scan at time k, recursive measurement update at time k
        raw_scan, _hits = world.scan(pose_true)
        scan = np.clip(
            raw_scan + rng.normal(0.0, config.sensor_sigma_m, raw_scan.shape), 0.02, 3.95
        )
        scan_trace.append(scan)
        if pf is not None:
            weights = measure_recursive(pf, scan, weights)
            ess_trace.append(pf.ess())
            if pf.ess() < config.particles / 2:
                pf.resample()
                resample_frames.append(k)
            weights = pf.w.copy()
        else:
            ess_trace.append(float(config.particles))

        # (3) the pose the controller believes
        pose_est = pf.estimate() if pf is not None else odom[k]
        if group == "A":
            pose_est = truth[k].copy()
        estimate.append(pose_est.copy())

        # (4) arrival announced ONLY by the estimate, held for announce_frames
        dist_est = math.hypot(pose_est[0] - goal[0], pose_est[1] - goal[1])
        streak = streak + 1 if dist_est < ARRIVAL_EST_M else 0
        if streak >= config.announce_frames and announce_frame is None:
            announce_frame = k
            announce_true = float(math.hypot(pose_true[0] - goal[0], pose_true[1] - goal[1]))
            result = "announced"
            end_k = k
            break

        # (5) periodic planning from the ESTIMATE on the shared prior map
        if path is None or k % REPLAN_EVERY == 0:
            path = plan(avoidance, pose_est[:2], goal, robot["radius"])
            wp = 0
            if path is None:
                no_path_frame = k
                result = "no_path"
                end_k = k
                break

        # (6) control and execution
        inc, wp, v_cmd = control_step(pose_est, path, wp, scan, stop_distance)
        commands.append(inc)
        encoders.append(np.asarray(inc, dtype=float) + encoders[-1])
        ds_true_exec = GEOMETRY.radius_m * (inc[0] + inc[1]) / 2.0
        dth_true_exec = GEOMETRY.radius_m * (inc[1] - inc[0]) / GEOMETRY.track_m
        ds_meas_exec = ds_true_exec * (1.0 + config.wheel_bias)
        dth_meas_exec = dth_true_exec * (1.0 + config.wheel_bias)
        travelled += abs(ds_true_exec)
        mid_t = truth[k][2] + dth_true_exec / 2.0
        pose_next = np.array(
            [
                truth[k][0] + ds_true_exec * math.cos(mid_t),
                truth[k][1] + ds_true_exec * math.sin(mid_t),
                wrap(truth[k][2] + dth_true_exec),
            ]
        )
        mid_o = odom[k][2] + dth_meas_exec / 2.0
        odom_next = np.array(
            [
                odom[k][0] + ds_meas_exec * math.cos(mid_o),
                odom[k][1] + ds_meas_exec * math.sin(mid_o),
                wrap(odom[k][2] + dth_meas_exec),
            ]
        )
        truth.append(pose_next)
        odom.append(odom_next)

        world.set_pose(pose_next)
        names = world.contact_names(pose_next)
        if names:
            contact_frames += 1
            contact_objects.update(names)
            if first_contact is None:
                first_contact = k
            if contact_frames > CONTACT_ABORT_FRAMES:
                result = "collision_abort"
                end_k = k
                break
        contact_flags.append(1 if names else 0)
        ds_prev = ds_meas_exec
        dth_prev = dth_meas_exec

    # ---------------------------------------------------------- scoring
    if result == "announced":
        result = "arrived" if announce_true <= ARRIVAL_TRUE_M else "false_arrived"
    elif result == "no_path":
        pass  # the scorer classifies correct rejection vs wrong give-up
    elif result == "timeout" and min_true_goal < ARRIVAL_TRUE_M:
        result = "timeout_truth_entered"
    true_end = float(np.linalg.norm(np.asarray(truth[end_k][:2]) - np.asarray(goal, dtype=float)))
    est_chain = np.asarray(estimate)
    truth_chain = np.asarray(truth)
    odom_chain = np.asarray(odom)
    n = min(len(est_chain), len(truth_chain))
    errors = np.linalg.norm(est_chain[:n, :2] - truth_chain[:n, :2], axis=1)
    return {
        "task": name,
        "group": group,
        "result": result,
        "announce_frame": announce_frame,
        "announce_true_m": announce_true,
        "true_end_m": true_end,
        "travelled_m": round(travelled, 3),
        "steps": end_k,
        "first_contact_frame": first_contact,
        "contact_frames": contact_frames,
        "contact_objects": sorted(contact_objects),
        "no_path_frame": no_path_frame,
        "min_true_goal_m": float(min_true_goal) if math.isfinite(min_true_goal) else None,
        "loc_mean_m": float(np.mean(errors)) if n else None,
        "loc_p95_m": float(np.percentile(errors, 95)) if n else None,
        "resamples": len(resample_frames),
    }, {
        "truth": truth_chain,
        "odom": odom_chain,
        "estimate": est_chain,
        "scan_trace": np.asarray(scan_trace),
        "commands": np.asarray(commands),
        "ess_trace": np.asarray(ess_trace),
        "contact_flag": np.asarray(contact_flags, dtype=int),
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
    robot = layout["robot"]
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
        rng_sensor = np.random.default_rng([seed, 56011, s])
        rng_pf = np.random.default_rng([seed, 56013, s])
        for name, start, goal, is_reachable in tasks:
            # EVERY task enters the same navigation entry: the planner itself
            # must return no_path for the unreachable goal - the rejection is
            # not pre-written by the task label (the recorded v1 finding)
            for group in GROUPS:
                # ALL tasks (including unreachable) enter run_closed_loop;
                # the planner itself returns no_path for the unreachable goal
                loc_map_for_group = loc_map if group == "C" else self_maps[s]
                rng_run = np.random.default_rng([seed, 56012, s, zlib.crc32(name.encode("utf-8"))])
                row, chains = run_closed_loop(
                    world,
                    (name, start, goal),
                    group,
                    loc_map_for_group,
                    rng_run,
                    config,
                    config.max_steps,
                    layout=layout,
                )
                row["seed"] = s
                # scorer: classify no_path by the task label (correct rejection
                # of the unreachable task vs wrong give-up on a reachable one)
                if row["result"] == "no_path":
                    row["scored_result"] = (
                        "correct_rejection" if not is_reachable else "wrong_give_up"
                    )
                else:
                    row["scored_result"] = row["result"]
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
    rejections = [r for r in rows if r.get("scored_result") == "correct_rejection"]
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
                a_arrived == len(a_reach)
                and a_contacts == 0
                and sum(1 for r in rejections if r["group"] == "A") == 3
            ),
            "a_arrived": a_arrived,
            "a_reachable_runs": len(a_reach),
            "a_contact_frames": a_contacts,
            "rejections": len(rejections),
        },
    }
    task_names = [t[0] for t in tasks]
    task_ids = [zlib.crc32(t[0].encode("utf-8")) % 10000 for t in tasks]
    assert len(set(task_ids)) == len(task_ids), (
        f"task ID collision: {list(zip(task_names, task_ids))}"
    )
    archives = {
        "task_names": np.array(task_names, dtype="U64"),
        "task_ids": np.array(task_ids, dtype=int),
        "task_reachable": np.array([1 if t[3] else 0 for t in tasks], dtype=int),
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
