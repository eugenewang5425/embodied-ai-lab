"""Lesson 43: occupancy-grid mapping + A* planning + pure-pursuit tracking.

Back on the main line (perception -> mapping -> planning) after phase 5
(lessons 28-42) established the information-precision wall for pure learning.
A differential car (lesson-14/21 kinematics, reused verbatim) drives in a
10 m x 10 m world at 0.1 m resolution (100 x 100 cells) with random
rectangle/circle obstacles plus boundary walls.  It carries an ideal 16-ray
range sensor (2 m).  Two groups on identical paired initializations:

  A "blind":   the lesson-21 goal controller on the true pose - drive
               straight at the goal, no sensors, no map, no plan;
  B "planned": Bresenham log-odds occupancy mapping from the rays, A*
               replanning on the known grid (8-connected, diagonal sqrt(2),
               obstacles inflated to the planning clearance), pure-pursuit tracking
               with corner- and obstacle-clamped lookahead plus a frontal
               brake reflex that suppresses translation but PRESERVES the
               yaw rate (a diff drive escapes by rotating in place).

Recorded protocol decisions: the pose is KNOWN to both groups (no SLAM - the
map is built from an ideal sensor, localization is excluded by design);
unknown cells are planned through optimistically, with per-step path
invalidation checks and replanning guarding the gap; a collision is an
ATTEMPTED pose that would penetrate an obstacle (the rigid world reverts the
step - no sliding, no bouncing); the arrival caliber is the lesson-39 rule
(dist < 5 cm and speed < 0.1 m/s); success/path formulas follow lessons
21/39 verbatim.  If tracking under the nonholonomic constraint fails, that
failure is the reported result, not a tuned-away detail.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.differential_drive import DriveGeometry, finite_vector, integrate_pose
from embodied_learning.goal_control import DEFAULT_CONFIG, goal_command
from embodied_learning.odometry import heading_error
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "grid_nav_lesson43"
SCHEMA_VERSION = 1
GROUPS = ("A", "B")
DT = 0.04  # s, the lesson-14/21 interval
L_OCC = math.log(0.7 / 0.3)  # log-odds update for a ray-endpoint cell
L_FREE = math.log(0.4 / 0.6)  # log-odds update for cells a ray passes through
L_MAX = 5.0  # log-odds clipping bound
COVERAGE_THRESHOLD = 0.1  # |log-odds| above this counts as "known"
SNAPSHOTS = 8  # mapping-process snapshots kept for the showcase episode
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/grid_nav_2026-09-06"
GEOMETRY = DriveGeometry()  # r = 0.05 m, track = 0.30 m (lessons 14/21)


@dataclass(frozen=True)
class GridNavConfig:
    """Protocol constants; defaults ARE the pre-registered lesson-43 run."""

    world_size_m: float = 10.0
    res_m: float = 0.1
    horizon_steps: int = 2500  # 100 s at dt = 0.04 s
    obstacle_scenes: int = 5  # plus scene 0 = empty control (walls only)
    inits: int = 3
    rays: int = 16
    max_range_m: float = 2.0
    r_car_m: float = 0.15
    # 3 cells = 0.30 m = car radius + half a cell diagonal (ceil to cells):
    # planned paths keep a TRUE clearance >= r_car + half-diagonal, which is
    # exactly the brake gate below - planner and reflex never contradict.
    inflate_cells: int = 3
    v_max_m_s: float = 0.25  # the lesson-21 speed limit
    omega_max_rad_s: float = 1.2  # the lesson-21 yaw-rate limit
    wheel_limit_rad_s: float = 6.0  # the lesson-21 wheel limit
    lookahead_m: float = 0.5
    lookahead_min_m: float = 0.15
    turn_sharp_rad: float = math.pi / 4
    spin_alpha_rad: float = math.radians(100.0)
    # 0.2207 = r_car + res * sqrt(2) / 2: an ATTEMPTED pose closer than one
    # half-cell-diagonal margin to a surface is braking distance.
    brake_dist_m: float = 0.2207
    brake_cone_rad: float = math.radians(50.0)
    replan_period_steps: int = 25  # 1 s
    collision_debounce_steps: int = 25  # 1 s between counted collision events
    arrival_radius_m: float = 0.05  # the lesson-39 arrival caliber
    arrival_speed_m_s: float = 0.1
    start_xy: tuple[float, float] = (1.0, 1.0)
    goal_xy: tuple[float, float] = (9.0, 9.0)
    init_jitter_m: float = 0.1

    def __post_init__(self):
        positive = (
            self.world_size_m,
            self.res_m,
            self.horizon_steps,
            self.rays,
            self.max_range_m,
            self.r_car_m,
            self.v_max_m_s,
            self.omega_max_rad_s,
            self.wheel_limit_rad_s,
            self.lookahead_m,
            self.brake_dist_m,
            self.replan_period_steps,
            self.collision_debounce_steps,
            self.arrival_radius_m,
            self.arrival_speed_m_s,
        )
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Grid navigation parameters must be finite and positive")
        if type(self.horizon_steps) is not int or type(self.rays) is not int:
            raise ValueError("horizon_steps and rays must be integers")
        if type(self.inflate_cells) is not int or self.inflate_cells < 0:
            raise ValueError("inflate_cells must be a non-negative integer")
        if type(self.obstacle_scenes) is not int or self.obstacle_scenes < 0:
            raise ValueError("obstacle_scenes must be a non-negative integer")
        if type(self.inits) is not int or self.inits < 1:
            raise ValueError("inits must be a positive integer")
        if abs(round(self.world_size_m / self.res_m) * self.res_m - self.world_size_m) > 1e-9:
            raise ValueError("world_size_m must be an exact multiple of res_m")
        if not 0 < self.lookahead_min_m <= self.lookahead_m:
            raise ValueError("need 0 < lookahead_min_m <= lookahead_m")
        if not self.r_car_m < self.brake_dist_m <= self.max_range_m:
            raise ValueError("need r_car_m < brake_dist_m <= max_range_m")
        if not 0 < self.spin_alpha_rad <= math.pi:
            raise ValueError("spin_alpha_rad must lie in (0, pi]")
        if not 0 < self.turn_sharp_rad < math.pi:
            raise ValueError("turn_sharp_rad must lie in (0, pi)")
        for name in ("start_xy", "goal_xy"):
            x, y = getattr(self, name)
            margin = self.r_car_m + self.inflate_cells * self.res_m
            if (
                not all(math.isfinite(v) for v in (x, y))
                or not margin < x < self.world_size_m - margin
                or not margin < y < self.world_size_m - margin
            ):
                raise ValueError(f"{name} must lie strictly inside the inflated arena")
        if not math.isfinite(self.init_jitter_m) or self.init_jitter_m < 0:
            raise ValueError("init_jitter_m must be finite and non-negative")

    @property
    def grid_cells(self) -> int:
        return round(self.world_size_m / self.res_m)

    @property
    def inflate_m(self) -> float:
        return self.inflate_cells * self.res_m


# ------------------------------------------------------------------- world
@dataclass(frozen=True)
class Obstacle:
    """Axis-aligned rectangle (cx, cy, half extents) or circle (cx, cy, r)."""

    kind: str
    cx: float
    cy: float
    hx: float = 0.0
    hy: float = 0.0
    r: float = 0.0

    def __post_init__(self):
        if self.kind not in ("rect", "circle"):
            raise ValueError("kind must be 'rect' or 'circle'")
        values = (self.cx, self.cy, self.hx, self.hy, self.r)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Obstacle parameters must be finite")
        if self.kind == "rect" and (self.hx <= 0 or self.hy <= 0):
            raise ValueError("Rectangle half extents must be positive")
        if self.kind == "circle" and self.r <= 0:
            raise ValueError("Circle radius must be positive")

    def distance(self, x, y):
        """Signed distance from a point to the obstacle surface.

        Circle and rectangle branches: strictly inside returns NEGATIVE (the
        distance to the nearest edge with a minus sign), strictly outside
        returns positive, on the surface returns 0.  The rect branch inside
        test is the signed extension of the `max(..., 0.0, ...)` clamp: a
        point inside a rectangle would otherwise report 0 for every interior
        point (fix from the lesson-44 work: true_occupancy and collision
        clearance both rely on negative-in-side semantics).
        """
        if self.kind == "circle":
            return math.hypot(x - self.cx, y - self.cy) - self.r
        dx = max(self.cx - self.hx - x, 0.0, x - (self.cx + self.hx))
        dy = max(self.cy - self.hy - y, 0.0, y - (self.cy + self.hy))
        if dx == 0.0 and dy == 0.0:
            return -min(
                x - (self.cx - self.hx),
                (self.cx + self.hx) - x,
                y - (self.cy - self.hy),
                (self.cy + self.hy) - y,
            )
        return math.hypot(dx, dy)

    def extent(self):
        return self.r if self.kind == "circle" else math.hypot(self.hx, self.hy)

    def to_dict(self):
        return {
            "kind": self.kind,
            "cx": self.cx,
            "cy": self.cy,
            "hx": self.hx,
            "hy": self.hy,
            "r": self.r,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["kind"], data["cx"], data["cy"], data["hx"], data["hy"], data["r"])


def build_walls(world_size_m, res_m):
    """Four boundary wall rectangles, one cell (res) thick, arena [0, size]^2."""
    half = world_size_m / 2
    t = res_m / 2
    return (
        Obstacle("rect", half, t, half, t),
        Obstacle("rect", half, world_size_m - t, half, t),
        Obstacle("rect", t, half, t, half),
        Obstacle("rect", world_size_m - t, half, t, half),
    )


def obstacle_distance(point, obstacles):
    return min(obstacle.distance(point[0], point[1]) for obstacle in obstacles)


def build_scenarios(config, master_seed):
    """Scene 0 is the empty control; scenes 1..obstacle_scenes have 5-8 obstacles.

    Obstacle sizes scale with the world (the defaults match the 10 m x 10 m
    protocol world); start/goal clearances stay absolute because the car does
    not scale.
    """
    scale = config.world_size_m / 10.0
    low = min(1.2, config.world_size_m / 2)
    high = config.world_size_m - low
    scenarios = []
    for scene in range(config.obstacle_scenes + 1):
        if scene == 0:
            scenarios.append({"scene": 0, "kind": "empty", "obstacles": ()})
            continue
        rng = np.random.default_rng([master_seed, 9100, scene])
        count = int(rng.integers(5, 9))
        obstacles = []
        for _ in range(count):
            shrink = 1.0
            for _try in range(300):
                kind = "circle" if rng.random() < 0.5 else "rect"
                if kind == "circle":
                    obstacle = Obstacle(
                        "circle",
                        float(rng.uniform(low, high)),
                        float(rng.uniform(low, high)),
                        r=float(rng.uniform(0.25, 0.6)) * scale * shrink,
                    )
                else:
                    hx = float(rng.uniform(0.2, 0.7)) * scale * shrink
                    hy = float(rng.uniform(0.2, 0.7)) * scale * shrink
                    obstacle = Obstacle(
                        "rect",
                        float(rng.uniform(low, high)),
                        float(rng.uniform(low, high)),
                        hx=hx,
                        hy=hy,
                    )
                extent = obstacle.extent()
                inside = (
                    obstacle.cx - extent >= 0.35
                    and obstacle.cx + extent <= config.world_size_m - 0.35
                    and obstacle.cy - extent >= 0.35
                    and obstacle.cy + extent <= config.world_size_m - 0.35
                )
                clear_start = obstacle.distance(*config.start_xy) >= 0.5
                clear_goal = obstacle.distance(*config.goal_xy) >= 0.5
                if inside and clear_start and clear_goal:
                    obstacles.append(obstacle)
                    break
                shrink *= 0.85
            else:
                raise ValueError("Could not place an obstacle with the clearances")
        scenarios.append({"scene": scene, "kind": "obstacles", "obstacles": tuple(obstacles)})
    return scenarios


def initial_poses(config, master_seed, scene):
    """Paired start poses (same for A and B): start point jittered, heading uniform."""
    poses = []
    for init in range(config.inits):
        rng = np.random.default_rng([master_seed, 7000, scene, init])
        poses.append(
            np.array(
                [
                    config.start_xy[0]
                    + float(rng.uniform(-config.init_jitter_m, config.init_jitter_m)),
                    config.start_xy[1]
                    + float(rng.uniform(-config.init_jitter_m, config.init_jitter_m)),
                    float(rng.uniform(-math.pi, math.pi)),
                ]
            )
        )
    return poses


def true_occupancy(obstacles, walls, n, res):
    """True map on the grid: border ring plus any cell whose centre is inside."""
    grid = np.zeros((n, n), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    for iy in range(1, n - 1):
        y = (iy + 0.5) * res
        for ix in range(1, n - 1):
            x = (ix + 0.5) * res
            for obstacle in (*obstacles, *walls):
                if obstacle.distance(x, y) < 0.0:
                    grid[iy, ix] = True
                    break
    return grid


def inflate(occupied, radius_cells):
    """Euclidean-disk dilation by radius_cells (cell units)."""
    if radius_cells <= 0:
        return occupied.copy()
    n = occupied.shape[0]
    out = occupied.copy()
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy > radius_cells * radius_cells:
                continue
            src_y = slice(max(0, -dy), n - max(0, dy))
            dst_y = slice(max(0, dy), n - max(0, -dy))
            src_x = slice(max(0, -dx), n - max(0, dx))
            dst_x = slice(max(0, dx), n - max(0, -dx))
            out[dst_y, dst_x] |= occupied[src_y, src_x]
    return out


# ------------------------------------------------------------------ sensing
def cast_rays(px, py, angles, obstacles, walls, max_range):
    """Ideal range sensor: nearest surface distance per ray, capped at max_range.

    Vectorized over rays; analytic ray/circle and ray/rectangle (slab) tests.
    A ray that leaves without a hit returns max_range with hit=False.
    """
    angles = np.asarray(angles, dtype=float)
    dx = np.cos(angles)
    dy = np.sin(angles)
    best = np.full(angles.shape, float(max_range))
    primitives = (*obstacles, *walls)
    circles = [o for o in primitives if o.kind == "circle"]
    rects = [o for o in primitives if o.kind == "rect"]
    if circles:
        cx = np.array([o.cx for o in circles])
        cy = np.array([o.cy for o in circles])
        cr = np.array([o.r for o in circles])
        relx = px - cx[None, :]
        rely = py - cy[None, :]
        b = relx * dx[:, None] + rely * dy[:, None]
        c0 = relx * relx + rely * rely - cr[None, :] ** 2
        disc = b * b - c0
        ok = disc >= 0.0
        root = np.sqrt(np.where(ok, disc, 0.0))
        t_near = -b - root
        t_far = -b + root
        # t_near < 0 with t_far >= 0 means the origin sits inside the primitive
        # (distance 0); a primitive fully BEHIND the ray (t_far < 0) is no hit.
        t = np.where(ok & (t_far >= 0.0), np.maximum(t_near, 0.0), np.inf)
        best = np.minimum(best, np.min(t, axis=1))
    if rects:
        x0 = np.array([o.cx - o.hx for o in rects])
        x1 = np.array([o.cx + o.hx for o in rects])
        y0 = np.array([o.cy - o.hy for o in rects])
        y1 = np.array([o.cy + o.hy for o in rects])
        with np.errstate(divide="ignore", invalid="ignore"):
            tx0 = (x0[None, :] - px) / dx[:, None]
            tx1 = (x1[None, :] - px) / dx[:, None]
            ty0 = (y0[None, :] - py) / dy[:, None]
            ty1 = (y1[None, :] - py) / dy[:, None]
        tx_lo, tx_hi = np.minimum(tx0, tx1), np.maximum(tx0, tx1)
        ty_lo, ty_hi = np.minimum(ty0, ty1), np.maximum(ty0, ty1)
        vertical = np.abs(dx)[:, None] < 1e-12
        horizontal = np.abs(dy)[:, None] < 1e-12
        inside_x = (px >= x0[None, :]) & (px <= x1[None, :])
        inside_y = (py >= y0[None, :]) & (py <= y1[None, :])
        tx_lo = np.where(vertical, np.where(inside_x, -np.inf, np.inf), tx_lo)
        tx_hi = np.where(vertical, np.inf, tx_hi)
        ty_lo = np.where(horizontal, np.where(inside_y, -np.inf, np.inf), ty_lo)
        ty_hi = np.where(horizontal, np.inf, ty_hi)
        t_enter = np.maximum(tx_lo, ty_lo)
        t_exit = np.minimum(tx_hi, ty_hi)
        ok = (t_enter <= t_exit) & (t_exit >= 0.0) & np.isfinite(t_enter)
        t = np.where(ok, np.maximum(t_enter, 0.0), np.inf)
        best = np.minimum(best, np.min(t, axis=1))
    return best, best < max_range


def frontal_ray_indices(config):
    offsets = np.arange(config.rays) * (2.0 * math.pi / config.rays)
    wrapped = np.arctan2(np.sin(offsets), np.cos(offsets))
    return np.flatnonzero(np.abs(wrapped) <= config.brake_cone_rad)


# ------------------------------------------------------------------- mapping
def bresenham(x0, y0, x1, y1):
    """Classic all-octant Bresenham; inclusive list of (x, y) grid cells."""
    cells = [(int(x0), int(y0))]
    x, y = int(x0), int(y0)
    dx, dy = abs(int(x1) - x), -abs(int(y1) - y)
    sx = 1 if x < int(x1) else -1
    sy = 1 if y < int(y1) else -1
    err = dx + dy
    while (x, y) != (int(x1), int(y1)):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        cells.append((x, y))
    return cells


def world_to_cell(x, y, res, n):
    return (
        int(np.clip(math.floor(y / res), 0, n - 1)),
        int(np.clip(math.floor(x / res), 0, n - 1)),
    )


def cell_center(cell, res):
    return np.array([(cell[1] + 0.5) * res, (cell[0] + 0.5) * res])


def update_grid(log_odds, pose_xy, angles, ranges, hit, res, n):
    """One Bresenham scan: free cells along each ray, occupied endpoint on a hit."""
    iy0, ix0 = world_to_cell(pose_xy[0], pose_xy[1], res, n)
    flat = log_odds.ravel()
    for k in range(len(ranges)):
        ex = pose_xy[0] + ranges[k] * math.cos(angles[k])
        ey = pose_xy[1] + ranges[k] * math.sin(angles[k])
        iy1, ix1 = world_to_cell(ex, ey, res, n)
        line = bresenham(ix0, iy0, ix1, iy1)
        if hit[k]:
            free, occupied = line[:-1], line[-1:]
        else:
            free, occupied = line, []
        if free:
            indices = [iy * n + ix for ix, iy in free]
            np.add.at(flat, indices, L_FREE)
        for ix, iy in occupied:
            flat[iy * n + ix] += L_OCC
    np.clip(log_odds, -L_MAX, L_MAX, out=log_odds)


def coverage_fraction(log_odds):
    """Fraction of cells that have moved away from the 0.5 prior."""
    return float(np.mean(np.abs(log_odds) > COVERAGE_THRESHOLD))


# ------------------------------------------------------------------- planning
def astar(passable, start, goal):
    """8-connected A*, octile heuristic, sqrt(2) diagonals.

    A diagonal move is rejected when either adjacent orthogonal cell is
    blocked (no squeezing between two blocked cells).  Returns
    (cells start->goal as (iy, ix), cost in CELL units) or (None, inf).
    """
    n = passable.shape[0]
    sy, sx = int(start[0]), int(start[1])
    gy, gx = int(goal[0]), int(goal[1])
    if not all(0 <= v < n for v in (sy, sx, gy, gx)):
        raise ValueError("start/goal outside the grid")
    if not passable[sy, sx] or not passable[gy, gx]:
        return None, math.inf
    sqrt2 = math.sqrt(2.0)

    def heuristic(y, x):
        ddx, ddy = abs(x - gx), abs(y - gy)
        return (ddx + ddy) + (sqrt2 - 2.0) * min(ddx, ddy)

    g_cost = np.full(n * n, np.inf)
    parent = np.full(n * n, -1, dtype=np.int64)
    start_idx, goal_idx = sy * n + sx, gy * n + gx
    g_cost[start_idx] = 0.0
    heap = [(heuristic(sy, sx), 0.0, start_idx)]
    moves = (
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (1, 1, sqrt2),
        (1, -1, sqrt2),
        (-1, 1, sqrt2),
        (-1, -1, sqrt2),
    )
    while heap:
        _f, g, idx = heapq.heappop(heap)
        if idx == goal_idx:
            path = []
            while idx != -1:
                path.append((int(idx // n), int(idx % n)))
                idx = int(parent[idx])
            return path[::-1], g
        if g > g_cost[idx]:
            continue
        y, x = divmod(idx, n)
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < n and 0 <= nx < n) or not passable[ny, nx]:
                continue
            if dy and dx and (not passable[y + dy, x] or not passable[y, x + dx]):
                continue
            nidx = ny * n + nx
            ng = g + step
            if ng < g_cost[nidx] - 1e-12:
                g_cost[nidx] = ng
                parent[nidx] = idx
                heapq.heappush(heap, (ng + heuristic(ny, nx), ng, nidx))
    return None, math.inf


def line_of_sight(passable, cell_a, cell_b):
    return all(passable[iy, ix] for ix, iy in bresenham(cell_a[1], cell_a[0], cell_b[1], cell_b[0]))


def smooth_path(cells, passable):
    """Greedily skip waypoints that still have grid line-of-sight (subset of cells)."""
    if len(cells) <= 2:
        return list(cells)
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_of_sight(passable, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def nearest_passable(passable, cell, max_ring):
    cy, cx = int(cell[0]), int(cell[1])
    n = passable.shape[0]
    if passable[cy, cx]:
        return (cy, cx)
    for ring in range(1, max_ring + 1):
        for dy in range(-ring, ring + 1):
            for dx in range(-ring, ring + 1):
                if max(abs(dy), abs(dx)) != ring:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < n and 0 <= nx < n and passable[ny, nx]:
                    return (ny, nx)
    return (cy, cx)


def make_plan(log_odds, pose, goal_xy, config):
    """A* + smoothing on the current known map (unknown = traversable)."""
    n = config.grid_cells
    res = config.res_m
    blocked = inflate(log_odds > COVERAGE_THRESHOLD, config.inflate_cells)
    passable = ~blocked
    start = nearest_passable(passable, world_to_cell(pose[0], pose[1], res, n), 5)
    goal_cell = world_to_cell(goal_xy[0], goal_xy[1], res, n)
    cells, cost = astar(passable, start, goal_cell)
    if cells is None:
        return None
    smooth = smooth_path(cells, passable)
    waypoints = np.array([cell_center(cell, res) for cell in smooth[:-1]] + [goal_xy])
    return {
        "cells": cells,
        "smooth": smooth,
        "waypoints": waypoints,
        "profile": path_profile(waypoints, config.turn_sharp_rad),
        "cost_m": cost * res,
    }


def plan_is_blocked(plan, log_odds, config, from_smooth_index=0):
    """True when any remaining (sampled) cell of the plan sits on inflated occupancy."""
    if plan is None:
        return True
    blocked = inflate(log_odds > COVERAGE_THRESHOLD, config.inflate_cells)
    for cell in plan["smooth"][from_smooth_index::2]:
        if blocked[cell[0], cell[1]]:
            return True
    return False


# ------------------------------------------------------------------ tracking
def path_profile(waypoints, turn_sharp_rad):
    """Segment lengths, cumulative arc length and sharp-corner node flags."""
    wp = np.asarray(waypoints, dtype=float)
    if len(wp) < 2:
        raise ValueError("A path needs at least two waypoints")
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    turn = np.zeros(len(wp), dtype=bool)
    for i in range(1, len(wp) - 1):
        v0, v1 = wp[i] - wp[i - 1], wp[i + 1] - wp[i]
        norm = float(np.linalg.norm(v0) * np.linalg.norm(v1))
        if norm < 1e-12:
            turn[i] = True
            continue
        angle = math.acos(float(np.clip(np.dot(v0, v1) / norm, -1.0, 1.0)))
        turn[i] = angle > turn_sharp_rad
    return seg, cum, turn


def wheels_from_twist(v, omega, wheel_limit_rad_s, geometry=GEOMETRY):
    """Forward speed + yaw rate -> [left, right] rad/s with joint limit scaling.

    Same conversion as the lesson-21 controller; equal scaling of both wheels
    preserves the requested curvature.
    """
    wheels = np.array(
        [
            (v - omega * geometry.track_m / 2) / geometry.radius_m,
            (v + omega * geometry.track_m / 2) / geometry.radius_m,
        ]
    )
    return wheels / max(1.0, float(np.max(np.abs(wheels))) / wheel_limit_rad_s)


def step_pose(pose, wheels, dt, geometry=GEOMETRY):
    return integrate_pose(pose, geometry.body_velocity(wheels), dt)


def pure_pursuit(pose, waypoints, profile, progress, config, frontal_clear_m=math.inf):
    """Pure pursuit adapted to a differential drive.

    Curvature kappa = 2 sin(alpha) / L (the pure-pursuit law); the Ackermann
    arctan steering-angle step is skipped because a diff drive commands the
    yaw rate directly: omega = kappa * v.  When the target falls behind
    (> spin_alpha), the car rotates in place instead - the diff-drive
    adaptation that plain pure pursuit lacks.  The effective lookahead is
    clamped before the next sharp corner AND to the nearest frontal ray, so
    the chord never targets through an obstacle that is already close.
    """
    wp = np.asarray(waypoints, dtype=float)
    seg, cum, turn = profile
    pose = finite_vector(pose, 3)
    p = pose[:2]
    lo = max(0, int(progress) - 3)
    dists = np.linalg.norm(wp[lo:] - p, axis=1)
    progress = lo + int(np.argmin(dists))
    dist_to_progress = float(dists[progress - lo])
    goal = wp[-1]
    remaining = float(np.linalg.norm(goal - p))
    if remaining <= config.arrival_radius_m:
        return {
            "wheels": np.zeros(2),
            "mode": "stopping",
            "progress": progress,
            "target": goal.copy(),
            "alpha": 0.0,
            "v": 0.0,
            "omega": 0.0,
            "lookahead_eff": 0.0,
        }
    sharp = np.flatnonzero(turn)[np.flatnonzero(turn) > progress]
    if len(sharp):
        s_turn = dist_to_progress + float(cum[sharp[0]] - cum[progress])
        l_eff = min(config.lookahead_m, max(config.lookahead_min_m, s_turn))
    else:
        l_eff = config.lookahead_m
    if frontal_clear_m < l_eff:
        l_eff = max(config.lookahead_min_m, frontal_clear_m)
    acc = dist_to_progress
    target = goal
    for k in range(progress, len(wp) - 1):
        if acc + seg[k] >= l_eff:
            ratio = float(np.clip((l_eff - acc) / max(seg[k], 1e-12), 0.0, 1.0))
            target = wp[k] + ratio * (wp[k + 1] - wp[k])
            break
        acc += seg[k]
    alpha = float(heading_error(math.atan2(target[1] - p[1], target[0] - p[0]), pose[2]))
    if abs(alpha) > config.spin_alpha_rad:
        v = 0.0
        omega = float(np.clip(2.0 * alpha, -config.omega_max_rad_s, config.omega_max_rad_s))
        mode = "rotating"
    else:
        # omega is computed from the REFERENCE speed, not the cos(alpha)-slowed
        # command: otherwise the turn rate collapses to zero together with v for
        # large |alpha| and a differential car (which CAN spin in place) freezes.
        v_ref = min(config.v_max_m_s, 0.7 * remaining)
        v = max(0.0, v_ref * math.cos(alpha))
        kappa = 2.0 * math.sin(alpha) / l_eff
        omega = float(np.clip(kappa * v_ref, -config.omega_max_rad_s, config.omega_max_rad_s))
        mode = "tracking"
    return {
        "wheels": wheels_from_twist(v, omega, config.wheel_limit_rad_s),
        "mode": mode,
        "progress": progress,
        "target": np.asarray(target, dtype=float).copy(),
        "alpha": alpha,
        "v": v,
        "omega": omega,
        "lookahead_eff": l_eff,
    }


# ------------------------------------------------------------------ episodes
def simulate_episode(group, obstacles, start_pose, config, *, record_extras=False):
    """One paired episode; group A = lesson-21 blind, group B = map+plan+track."""
    if group not in GROUPS:
        raise ValueError("group must be 'A' or 'B'")
    if type(config.horizon_steps) is not int or config.horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer")
    walls = build_walls(config.world_size_m, config.res_m)
    all_obstacles = (*obstacles, *walls)
    n = config.grid_cells
    res = config.world_size_m / n
    goal = np.asarray(config.goal_xy, dtype=float)
    pose = finite_vector(start_pose, 3).copy()
    truth = [pose.copy()]
    grid = np.zeros((n, n)) if group == "B" else None
    frontal = frontal_ray_indices(config)
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
        snap_steps = {
            int(s) for s in np.unique(np.rint(np.linspace(1, config.horizon_steps, SNAPSHOTS)))
        }
    snaps = {"grid": [], "pose": [], "ranges": [], "hits": [], "cov": [], "step": []}
    for step in range(1, config.horizon_steps + 1):
        if group == "B":
            angles = pose[2] + np.arange(config.rays) * (2.0 * math.pi / config.rays)
            ranges, hit = cast_rays(pose[0], pose[1], angles, obstacles, walls, config.max_range_m)
            update_grid(grid, pose[:2], angles, ranges, hit, res, n)
            if record_extras and step in snap_steps and len(snaps["step"]) < SNAPSHOTS:
                snaps["grid"].append(grid.copy())
                snaps["pose"].append(pose.copy())
                snaps["ranges"].append(ranges.copy())
                snaps["hits"].append(hit.copy())
                snaps["cov"].append(coverage_fraction(grid))
                snaps["step"].append(step)
        if group == "A":
            wheels = goal_command(pose, goal, DEFAULT_CONFIG)["wheels"]
        else:
            due = step - last_replan >= config.replan_period_steps
            if force_replan or due or plan_is_blocked(plan, grid, config, progress):
                plan = make_plan(grid, pose, goal, config)
                last_replan = step
                replans += 1
                progress = 0
                force_replan = False
                if replan_log is not None:
                    replan_log.append(
                        {
                            "step": step,
                            "cost_m": None if plan is None else plan["cost_m"],
                            "cells": 0 if plan is None else len(plan["cells"]),
                            "waypoints": (None if plan is None else plan["waypoints"].copy()),
                        }
                    )
            frontal_clear = float(np.min(ranges[frontal]))
            if plan is None:
                command_omega = 0.0
                wheels = goal_command(pose, goal, DEFAULT_CONFIG)["wheels"]
                fallback_steps += 1
            else:
                command = pure_pursuit(
                    pose,
                    plan["waypoints"],
                    plan["profile"],
                    progress,
                    config,
                    frontal_clear_m=frontal_clear,
                )
                progress = command["progress"]
                command_omega = command["omega"]
                wheels = command["wheels"]
            if frontal_clear < config.brake_dist_m:
                # The reflex suppresses TRANSLATION only and preserves the yaw
                # rate: zeroing both wheels would also kill the in-place
                # rotation that is a differential car's only way out of a
                # tight spot, freezing it against the obstacle (the brake
                # livelock probe in the lecture documents that failure).
                wheels = wheels_from_twist(0.0, command_omega, config.wheel_limit_rad_s)
                brake_steps += 1
                force_replan = True
        candidate = step_pose(pose, wheels, DT)
        clearance = obstacle_distance(candidate[:2], all_obstacles)
        if clearance < config.r_car_m:
            pressed += 1
            if step - last_collision >= config.collision_debounce_steps:
                collisions += 1
                last_collision = step
        else:
            pose = candidate
        min_clear = min(min_clear, obstacle_distance(pose[:2], all_obstacles))
        truth.append(pose.copy())
        speed = abs(float(GEOMETRY.body_velocity(wheels)[0]))
        if (
            np.linalg.norm(pose[:2] - goal) < config.arrival_radius_m
            and speed < config.arrival_speed_m_s
        ):
            outcome = "arrived"
            arrival_step = step
            break
    truth_array = np.asarray(truth)
    path_length = float(np.linalg.norm(np.diff(truth_array[:, :2], axis=0), axis=1).sum())
    metrics = {
        "group": group,
        "outcome": outcome,
        "steps": int(arrival_step if arrival_step is not None else config.horizon_steps),
        "duration_s": (arrival_step if arrival_step is not None else config.horizon_steps) * DT,
        "arrival_time_s": None if arrival_step is None else arrival_step * DT,
        "path_length_m": path_length,
        "collisions": collisions,
        "pressed_steps": pressed,
        "min_obstacle_distance_m": float(min_clear),
        "coverage_final": coverage_fraction(grid) if group == "B" else None,
        "replans": replans if group == "B" else None,
        "fallback_steps": fallback_steps if group == "B" else None,
        "brake_steps": brake_steps if group == "B" else None,
    }
    extras = None
    if group == "B":
        extras = {"final_map": grid.copy(), "metrics": metrics}
        if record_extras:
            extras["replan_log"] = replan_log
            count = len(snaps["step"])
            pad = count - 1 if count else 0
            extras["snapshots"] = {
                "grids": (
                    np.stack(snaps["grid"] + [snaps["grid"][pad]] * (SNAPSHOTS - count))
                    if count
                    else np.zeros((SNAPSHOTS, n, n))
                ),
                "poses": (
                    np.stack(snaps["pose"] + [snaps["pose"][pad]] * (SNAPSHOTS - count))
                    if count
                    else np.zeros((SNAPSHOTS, 3))
                ),
                "ranges": (
                    np.stack(snaps["ranges"] + [snaps["ranges"][pad]] * (SNAPSHOTS - count))
                    if count
                    else np.zeros((SNAPSHOTS, config.rays))
                ),
                "hits": (
                    np.stack(snaps["hits"] + [snaps["hits"][pad]] * (SNAPSHOTS - count))
                    if count
                    else np.zeros((SNAPSHOTS, config.rays), dtype=bool)
                ),
                "cov": (
                    np.array(snaps["cov"] + [snaps["cov"][pad]] * (SNAPSHOTS - count))
                    if count
                    else np.zeros(SNAPSHOTS)
                ),
                "steps": (
                    np.array(snaps["step"] + [snaps["step"][pad]] * (SNAPSHOTS - count), dtype=int)
                    if count
                    else np.zeros(SNAPSHOTS, dtype=int)
                ),
            }
    return metrics, truth_array, extras


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
)


def aggregate_episodes(rows):
    """Group-level metrics; the calibers follow lessons 21/39 verbatim."""
    if not rows:
        raise ValueError("aggregate_episodes needs at least one episode")
    arrived = [row for row in rows if row["outcome"] == "arrived"]
    ratios = [row["path_ratio"] for row in arrived if row.get("path_ratio") is not None]
    coverages = [row["coverage_final"] for row in rows if row.get("coverage_final") is not None]
    times = [row["arrival_time_s"] for row in arrived if row.get("arrival_time_s") is not None]
    return {
        "episodes": len(rows),
        "arrivals": len(arrived),
        "arrival_rate": len(arrived) / len(rows),
        "collision_events_total": int(sum(row["collisions"] for row in rows)),
        "collision_events_mean": float(np.mean([row["collisions"] for row in rows])),
        "episodes_with_collision": int(sum(row["collisions"] > 0 for row in rows)),
        "pressed_steps_total": int(sum(row["pressed_steps"] for row in rows)),
        "brake_steps_total": int(sum(row.get("brake_steps") or 0 for row in rows)),
        "mean_min_clearance_m": float(np.mean([row["min_obstacle_distance_m"] for row in rows])),
        "mean_path_ratio": float(np.mean(ratios)) if ratios else None,
        "median_path_ratio": float(np.median(ratios)) if ratios else None,
        "mean_coverage": float(np.mean(coverages)) if coverages else None,
        "mean_arrival_time_s": float(np.mean(times)) if times else None,
    }


def expected_npz_keys(report):
    scenes = report["protocol"]["scene_count"]
    inits = report["protocol"]["inits"]
    keys = set()
    for group in GROUPS:
        for scene in range(scenes):
            for init in range(inits):
                keys.add(f"truth_{group}_{scene}_{init}")
        keys.update(f"{group}_{name}" for name in AGGREGATE_FIELDS)
    for scene in range(scenes):
        keys.add(f"true_map_{scene}")
        keys.add(f"final_map_{scene}")
    keys.update(
        (
            "snap_grids",
            "snap_poses",
            "snap_ranges",
            "snap_hits",
            "snap_cov",
            "snap_steps",
            "replan_paths",
            "replan_steps",
            "replan_costs",
            "replan_cells",
        )
    )
    return keys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, log=print):
    """Full paired experiment; new output directories only, bitwise repeatable."""
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    config = config or GridNavConfig()
    started = time.perf_counter()
    walls = build_walls(config.world_size_m, config.res_m)
    scenarios = build_scenarios(config, seed)
    n = config.grid_cells
    res = config.res_m
    scene_count = len(scenarios)
    showcase_scene = 1 if config.obstacle_scenes >= 1 else 0

    log(f"scene 0/{scene_count}: empty control + {config.obstacle_scenes} obstacle scenes")
    scene_records = []
    truths = {"A": {}, "B": {}}
    final_maps = {}
    true_maps = {}
    rows = []
    showcase = None
    for scenario in scenarios:
        scene = scenario["scene"]
        true_map = true_occupancy(scenario["obstacles"], walls, n, res)
        true_maps[scene] = true_map
        passable = ~inflate(true_map, config.inflate_cells)
        shortest_cells, shortest_cost = astar(
            passable,
            world_to_cell(config.start_xy[0], config.start_xy[1], res, n),
            world_to_cell(config.goal_xy[0], config.goal_xy[1], res, n),
        )
        shortest_m = None if shortest_cells is None else shortest_cost * res
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
        for init, start_pose in enumerate(initial_poses(config, seed, scene)):
            for group in GROUPS:
                record = group == "B" and scene == showcase_scene and init == 0
                metrics, truth_array, extras = simulate_episode(
                    group,
                    scenario["obstacles"],
                    start_pose,
                    config,
                    record_extras=record,
                )
                truths[group][f"{scene}_{init}"] = truth_array
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
                if group == "B":
                    if init == 0:
                        final_maps[scene] = extras["final_map"]
                    if record:
                        showcase = extras
                log(
                    f"  scene {scene} init {init} group {group}: {metrics['outcome']} "
                    f"collisions {metrics['collisions']} length "
                    f"{metrics['path_length_m']:.2f} m replans {metrics['replans']} "
                    f"coverage {metrics['coverage_final']}"
                )

    aggregates = {}
    comparison = []
    labels = {"A": "盲飞（第 21 课控制器，真值位姿）", "B": "建图 + A* 规划 + 纯追踪"}
    for group in GROUPS:
        aggregates[group] = {}
        for condition, rows_here in (
            (
                "obstacle",
                [r for r in rows if r["group"] == group and r["scene"] > 0],
            ),
            (
                "empty",
                [r for r in rows if r["group"] == group and r["scene"] == 0],
            ),
        ):
            if rows_here:
                aggregates[group][condition] = aggregate_episodes(rows_here)
                comparison.append(
                    {
                        "group": group,
                        "condition": condition,
                        "label": f"{labels[group]}·{'有障碍' if condition == 'obstacle' else '无障碍'}",
                        **aggregates[group][condition],
                    }
                )

    b_obstacle = aggregates["B"].get("obstacle", {"arrivals": 0, "episodes": 1})
    a_obstacle = aggregates["A"].get("obstacle", {"arrivals": 0, "episodes": 1})
    hypothesis = {
        "claim": ("在障碍物场景中，建图 + A* 规划 + 纯追踪的到达率显著高于盲飞，且碰撞事件更少"),
        "criterion": (
            f"B 有障碍到达 >= {math.ceil(0.8 * b_obstacle['episodes'])}/"
            f"{b_obstacle['episodes']} 且 B 有障碍到达数 > A 有障碍到达数（预注册）"
        ),
        "b_obstacle_arrivals": b_obstacle["arrivals"],
        "a_obstacle_arrivals": a_obstacle["arrivals"],
        "met": bool(
            b_obstacle["arrivals"] >= math.ceil(0.8 * b_obstacle["episodes"])
            and b_obstacle["arrivals"] > a_obstacle["arrivals"]
        ),
        "empty_control": {
            "a_arrivals": aggregates["A"].get("empty", {}).get("arrivals"),
            "b_arrivals": aggregates["B"].get("empty", {}).get("arrivals"),
            "note": "无障碍对照：两组都应到达（控制器健全性闸门）",
        },
    }

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = {}
    metric_arrays = {}
    for group in GROUPS:
        for name in AGGREGATE_FIELDS:
            metric_arrays[f"{group}_{name}"] = np.full((scene_count, config.inits), np.nan)
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
        }
        for name, number in value.items():
            if number is not None:
                metric_arrays[f"{row['group']}_{name}"][row["scene"], row["init"]] = number
    archive.update(metric_arrays)
    for group in GROUPS:
        for key, array in truths[group].items():
            archive[f"truth_{group}_{key}"] = array
    for scene, true_map in true_maps.items():
        archive[f"true_map_{scene}"] = true_map
    for scene, grid in final_maps.items():
        archive[f"final_map_{scene}"] = grid
    if showcase is not None:
        snaps = showcase["snapshots"]
        archive["snap_grids"] = snaps["grids"].astype(np.float32)
        archive["snap_poses"] = snaps["poses"].astype(float)
        archive["snap_ranges"] = snaps["ranges"].astype(float)
        archive["snap_hits"] = snaps["hits"]
        archive["snap_cov"] = snaps["cov"].astype(float)
        archive["snap_steps"] = snaps["steps"]
        log_entries = showcase.get("replan_log") or []
        if log_entries:
            max_len = max(
                len(entry["waypoints"]) if entry["waypoints"] is not None else 1
                for entry in log_entries
            )
            paths = np.full((len(log_entries), max_len, 2), np.nan)
            for index, entry in enumerate(log_entries):
                if entry["waypoints"] is not None:
                    paths[index, : len(entry["waypoints"])] = entry["waypoints"]
            archive["replan_paths"] = paths.astype(np.float32)
            archive["replan_steps"] = np.array([e["step"] for e in log_entries], dtype=int)
            archive["replan_costs"] = np.array(
                [math.nan if e["cost_m"] is None else e["cost_m"] for e in log_entries]
            )
            archive["replan_cells"] = np.array([e["cells"] for e in log_entries], dtype=int)
        else:
            archive["replan_paths"] = np.zeros((0, 0, 2))
            archive["replan_steps"] = np.zeros(0, dtype=int)
            archive["replan_costs"] = np.zeros(0)
            archive["replan_cells"] = np.zeros(0, dtype=int)
    np.savez_compressed(output / "trajectories.npz", **archive)

    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "world_size_m": config.world_size_m,
            "res_m": config.res_m,
            "grid_cells": n,
            "dt_s": DT,
            "horizon_steps": config.horizon_steps,
            "horizon_s": config.horizon_steps * DT,
            "rays": config.rays,
            "max_range_m": config.max_range_m,
            "sensor_note": "理想 16 射线 360° 距离传感器，装于轮轴中心（无安装偏移），无噪声",
            "r_car_m": config.r_car_m,
            "inflate_cells": config.inflate_cells,
            "inflate_m": config.inflate_m,
            "inflate_note": (
                "膨胀 = 3 格 = 0.30 m = 车半径 0.15 + 半格对角 0.071（向上取整），"
                "使规划路径的真实间隙 >= 刹车反射门限 0.2207 m"
            ),
            "v_max_m_s": config.v_max_m_s,
            "omega_max_rad_s": config.omega_max_rad_s,
            "wheel_limit_rad_s": config.wheel_limit_rad_s,
            "lookahead_m": config.lookahead_m,
            "lookahead_min_m": config.lookahead_min_m,
            "turn_sharp_rad": config.turn_sharp_rad,
            "spin_alpha_rad": config.spin_alpha_rad,
            "brake_dist_m": config.brake_dist_m,
            "brake_cone_rad": config.brake_cone_rad,
            "brake_rule": (
                "前向刹车只抑制平移、保留偏航角速度（原地转脱困）；门限 = 车半径 + 半格对角；"
                "纯追踪前瞻距离同时钳制到正前方最近射线的距离"
            ),
            "replan_period_steps": config.replan_period_steps,
            "collision_debounce_steps": config.collision_debounce_steps,
            "arrival_radius_m": config.arrival_radius_m,
            "arrival_speed_m_s": config.arrival_speed_m_s,
            "start_xy": list(config.start_xy),
            "goal_xy": list(config.goal_xy),
            "init_jitter_m": config.init_jitter_m,
            "obstacle_scenes": config.obstacle_scenes,
            "scene_count": scene_count,
            "inits": config.inits,
            "episodes_per_group": scene_count * config.inits,
            "l_occ": L_OCC,
            "l_free": L_FREE,
            "l_max": L_MAX,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "unknown_policy": "optimistic（未知格可通行；逐步路径失效检查 + 正前方刹车反射）",
            "corner_rule": "对角移动要求两个正交邻格均可通行（不允许挤缝）",
            "collision_caliber": (
                "碰撞 = 试探位姿与障碍距离 < 车半径（刚性世界退回该步，不滑动不反弹）；"
                "事件间隔 >= 1 s 去抖"
            ),
            "groups": {
                "A": "盲飞：goal_command（第 21 课）作用于真值位姿，不建图不规划",
                "B": "建图（Bresenham 对数概率）+ A* 重规划（已知栅格，8 邻域）+ 纯追踪",
            },
            "showcase": {"scene": showcase_scene, "init": 0, "snapshots": SNAPSHOTS},
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
                "grid_nav_demo.py",
            )
        },
        "limits": [
            "无 SLAM：位姿已知（真值），地图仅由理想距离传感器构建；定位误差不在本课范围",
            "障碍静态、传感器无噪声、运动学无动力学；结论不外推到真实激光雷达与实车",
            "未知格按可通行处理（optimistic）：这是记录在案的协议决定，不是对真实建图器的声明",
            "纯追踪在急弯会切内角；前瞻钳制只是最小对策，切弯余量未做系统消融",
            "真值最短路径按膨胀后栅格 A* 计算，是参考分母而非被跟踪的轨迹",
            f"每回合计数有限（每组 {scene_count * config.inits} 回合）；到达率不是总体成功率",
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
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    conditions = [("obstacle", "有障碍（每格 15 回合）"), ("empty", "无障碍（每格 3 回合）")]
    colors = {"A": "#b91c1c", "B": "#0f766e"}
    width = 0.36
    for ax, (key, label) in zip(
        (axes[0, 0], axes[0, 1]),
        (("arrival_rate", "到达率"), ("collision_events_mean", "平均碰撞事件/回合")),
        strict=True,
    ):
        for offset, group in enumerate(GROUPS):
            values = [
                aggregates[group].get(condition, {}).get(key, 0.0) for condition, _ in conditions
            ]
            counts = [
                f"{aggregates[group].get(c, {}).get('arrivals', 0)}/"
                f"{aggregates[group].get(c, {}).get('episodes', 0)}"
                if key == "arrival_rate"
                else f"{value:.2f}"
                for c, value in zip([c for c, _ in conditions], values, strict=True)
            ]
            bars = ax.bar(
                np.arange(2) + (offset - 0.5) * width,
                values,
                width * 0.92,
                color=colors[group],
                label="盲飞（第 21 课）" if group == "A" else "建图+规划+追踪",
            )
            for bar, text in zip(bars, counts, strict=True):
                ax.annotate(
                    text,
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax.set(xticks=np.arange(2), xticklabels=[c[1] for c in conditions], ylabel=label)
        ax.set_title(
            f"{'到达率（到达 5 cm 内且速度 < 0.1 m/s）' if key == 'arrival_rate' else '碰撞事件（试探位姿侵入障碍，去抖 1 s）'}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")
    ax = axes[1, 0]
    for offset, group in enumerate(GROUPS):
        ratios = [
            aggregates[group].get(condition, {}).get("median_path_ratio")
            for condition, _ in conditions
        ]
        values = [0.0 if value is None else value for value in ratios]
        bars = ax.bar(
            np.arange(2) + (offset - 0.5) * width,
            values,
            width * 0.92,
            color=colors[group],
            label="盲飞（第 21 课）" if group == "A" else "建图+规划+追踪",
        )
        for bar, value in zip(bars, ratios, strict=True):
            ax.annotate(
                f"{value:.2f}" if value is not None else "—",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1.0)
    max_ratio = max([1.6] + [value for value in values if value > 0])
    ax.set(
        xticks=np.arange(2),
        xticklabels=[c[1] for c in conditions],
        ylabel="路径长度比（实际/真值最短，成功回合中位）",
        title="路径长度比（1.0 = 与全知最短一致；点线 = 理想）",
        ylim=(0, max_ratio),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    ax = axes[1, 1]
    scenes = report["protocol"]["scene_count"]
    coverages = [
        next(
            (
                row["coverage_final"]
                for row in report["episodes"]
                if row["group"] == "B" and row["scene"] == scene and row["init"] == 0
            ),
            0.0,
        )
        for scene in range(scenes)
    ]
    names = [
        f"场景 {s}\n{'无障碍' if s == 0 else str(len(report['scenarios'][s]['obstacles'])) + ' 障碍'}"
        for s in range(scenes)
    ]
    ax.bar(np.arange(scenes), coverages, color="#0f766e", width=0.55)
    for index, value in enumerate(coverages):
        ax.annotate(
            f"{value:.0%}",
            (index, value),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(
        xticks=np.arange(scenes),
        xticklabels=names,
        ylabel="建图覆盖率（已知格占比，B 组初始化 0）",
        title="建图覆盖率：16 射线 × 2 m 逐步扫描",
        ylim=(0, 1.0),
    )
    ax.grid(alpha=0.2, axis="y")
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
            alpha=0.45,
            vmin=0,
            vmax=1,
        )
        truth_a = archive[f"truth_A_{scene}_0"]
        truth_b = archive[f"truth_B_{scene}_0"]
        ax.plot(truth_a[:, 0], truth_a[:, 1], color="#b91c1c", linewidth=1.1, label="A 盲飞")
        ax.plot(truth_b[:, 0], truth_b[:, 1], color="#0f766e", linewidth=1.1, label="B 建图+规划")
        start_row = next(
            row
            for row in report["episodes"]
            if row["group"] == "A" and row["scene"] == scene and row["init"] == 0
        )
        start = start_row["start_pose"]
        goal = report["protocol"]["goal_xy"]
        ax.plot(start[0], start[1], "s", color="#111827", markersize=5)
        ax.plot(goal[0], goal[1], "*", color="#15803d", markersize=12)
        obstacles = report["scenarios"][scene]["obstacles"]
        title = "无障碍（对照）" if not obstacles else f"{len(obstacles)} 个随机障碍"
        shortest = report["scenarios"][scene]["true_shortest_m"]
        suffix = "∞" if shortest is None else f"{shortest:.2f} m"
        ax.set(
            xlabel="x（m）",
            ylabel="y（m）",
            title=f"场景 {scene}（{title}）：真值最短 {suffix}",
        )
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7, loc="upper left")
    for ax in axes[scenes:]:
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=5, help="障碍场景数（另有 1 个无障碍对照）")
    parser.add_argument("--inits", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=2500)
    parser.add_argument("--world-size", type=float, default=10.0)
    parser.add_argument("--start", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--goal", type=float, nargs=2, default=[9.0, 9.0])
    args = parser.parse_args()
    config = GridNavConfig(
        world_size_m=args.world_size,
        horizon_steps=args.horizon,
        obstacle_scenes=args.scenes,
        inits=args.inits,
        start_xy=tuple(args.start),
        goal_xy=tuple(args.goal),
    )

    def log(message):
        print(message, file=sys.stderr)  # keep stdout a pure JSON document

    try:
        report = run_experiment(args.output, seed=args.seed, config=config, log=log)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "aggregates": report["aggregates"],
                "hypothesis": report["hypothesis"],
                "coverage": {
                    f"scene_{row['scene']}_init_{row['init']}": row["coverage_final"]
                    for row in report["episodes"]
                    if row["group"] == "B"
                },
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    print("Saved:", args.output.resolve(), file=sys.stderr)


if __name__ == "__main__":
    main()
