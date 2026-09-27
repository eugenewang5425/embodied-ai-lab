"""Reusable patrol route generator (roadmap Phase 1 / P5).

Reads an EXPLICIT occupancy map (bool grid, True = obstacle), a start, named
inspection points and the robot envelope, and produces continuous routes
through the existing grid_nav planner (A* + inflation + line-of-sight
smoothing).  Outputs per-point feasibility with machine-readable rejection
reasons, route length, and the minimum obstacle clearance along the polyline
(a robot-envelope validity check, not just "A* found something").

Route protocol tags: the SAME waypoint task can be executed as
  "diagnostic"  - truth-driven replay for data collection / analysis;
  "estimation"  - the robot tracks the waypoints from its own estimate and
                  may never read its true pose.
The generator only builds routes; the execution protocol is declared by the
consumer and recorded alongside results.

CLI (five-task smoke on the photo room):
  uv run python -m embodied_learning.experiments.patrol_routes
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from embodied_learning.experiments.grid_nav import astar, inflate, smooth_path

REASON_OK = "ok"
REASON_OUTSIDE = "point_outside_grid"
REASON_OCCUPIED = "point_in_obstacle"
REASON_INFLATED = "point_blocked_by_robot_envelope"
REASON_NO_PATH = "no_path"
REASON_START_BLOCKED = "start_blocked"


@dataclass(frozen=True)
class PatrolMap:
    """Explicit occupancy grid: True = obstacle; origin = world xy of cell
    (row 0, col 0); resolution in metres per cell."""

    occupied: np.ndarray
    resolution: float
    origin: tuple[float, float] = (0.0, 0.0)

    def to_cell(self, xy):
        col = math.floor((xy[0] - self.origin[0]) / self.resolution)
        row = math.floor((xy[1] - self.origin[1]) / self.resolution)
        return row, col

    def to_world(self, row, col):
        return (
            self.origin[0] + (col + 0.5) * self.resolution,
            self.origin[1] + (row + 0.5) * self.resolution,
        )

    def inside(self, row, col) -> bool:
        n = self.occupied.shape[0]
        return 0 <= row < n and 0 <= col < n


@dataclass(frozen=True)
class PatrolSpec:
    start: tuple[float, float]
    points: tuple[tuple[str, tuple[float, float]], ...]
    close_loop: bool = True
    protocol: str = "diagnostic"  # or "estimation"

    def __post_init__(self):
        if self.protocol not in ("diagnostic", "estimation"):
            raise ValueError("protocol must be 'diagnostic' or 'estimation'")
        if len(self.points) == 0:
            raise ValueError("at least one inspection point is required")


@dataclass
class PatrolRoute:
    name: str
    protocol: str
    feasible: bool
    waypoints_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    length_m: float = 0.0
    min_clearance_m: float | None = None
    envelope_ok: bool = False
    legs: list = field(default_factory=list)
    rejected_points: list = field(default_factory=list)

    def summary(self):
        return {
            "name": self.name,
            "protocol": self.protocol,
            "feasible": self.feasible,
            "n_waypoints": len(self.waypoints_world),
            "length_m": round(self.length_m, 3),
            "min_clearance_m": (
                round(self.min_clearance_m, 4) if self.min_clearance_m is not None else None
            ),
            "envelope_ok": self.envelope_ok,
            "rejected_points": [
                {"name": n, "reason": r} for n, r in self.rejected_points
            ],
        }


def _clearance_field(patrol_map: PatrolMap):
    return distance_transform_edt(~patrol_map.occupied) * patrol_map.resolution


def build_route(patrol_map: PatrolMap, spec: PatrolSpec, robot_radius_m: float) -> PatrolRoute:
    """Plan start -> each point (-> start) with per-leg rejection reasons."""
    if robot_radius_m < 0:
        raise ValueError("robot_radius_m must be nonnegative")
    n = patrol_map.occupied.shape[0]
    radius_cells = math.ceil(robot_radius_m / patrol_map.resolution)
    safe = ~inflate(patrol_map.occupied, radius_cells)
    clearance = _clearance_field(patrol_map)

    route = PatrolRoute(name="", protocol=spec.protocol, feasible=True)
    start_cell = patrol_map.to_cell(spec.start)
    if not patrol_map.inside(*start_cell) or not safe[start_cell]:
        route.rejected_points.append(
            ("__start__", REASON_START_BLOCKED if patrol_map.inside(*start_cell) else REASON_OUTSIDE)
        )
        return route

    polyline_world = [np.asarray(spec.start, dtype=float)]
    start_cell_ = start_cell
    route.name = "->".join([p[0] for p in spec.points])
    prev_cell = start_cell_
    all_ok = True
    targets = list(spec.points)
    if spec.close_loop:
        targets.append(("__return_to_start__", spec.start))
    for pname, pxy in targets:
        cell = patrol_map.to_cell(pxy)
        if not patrol_map.inside(*cell):
            route.rejected_points.append((pname, REASON_OUTSIDE))
            all_ok = False
            continue
        if patrol_map.occupied[cell]:
            route.rejected_points.append((pname, REASON_OCCUPIED))
            all_ok = False
            continue
        if not safe[cell]:
            route.rejected_points.append((pname, REASON_INFLATED))
            all_ok = False
            continue
        leg_cells, _cost = astar(safe, prev_cell, cell)
        if leg_cells is None:
            route.rejected_points.append((pname, REASON_NO_PATH))
            all_ok = False
            continue
        leg_cells = smooth_path(leg_cells, safe)
        leg_world_full = np.array(
            [patrol_map.to_world(r, c) for r, c in leg_cells], dtype=float
        )
        leg_len = float(
            np.sum(np.linalg.norm(np.diff(leg_world_full, axis=0), axis=1))
        )
        # drop the duplicated joint waypoint
        leg_world = leg_world_full[1:]
        polyline_world.extend(leg_world)
        route.legs.append(
            {
                "to": pname,
                "n_cells": len(leg_cells),
                "length_m": round(leg_len, 3),
                "reason": REASON_OK,
            }
        )
        prev_cell = cell

    route.feasible = all_ok
    if len(polyline_world) < 2:
        return route
    wp = np.asarray(polyline_world, dtype=float)
    route.waypoints_world = wp
    seg = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    route.length_m = float(seg.sum())
    # clearance sampled densely along the polyline (envelope validity)
    samples = [wp[0]]
    for a, b, step in zip(wp[:-1], wp[1:], seg, strict=True):
        k = max(1, math.ceil(step / patrol_map.resolution))
        for t in np.linspace(0.0, 1.0, k, endpoint=False):
            samples.append(a + t * (b - a))
    samples.append(wp[-1])
    samples = np.asarray(samples)
    cols = np.floor((samples[:, 0] - patrol_map.origin[0]) / patrol_map.resolution).astype(int)
    rows = np.floor((samples[:, 1] - patrol_map.origin[1]) / patrol_map.resolution).astype(int)
    ok = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
    route.min_clearance_m = float(clearance[rows[ok], cols[ok]].min())
    route.envelope_ok = route.min_clearance_m >= robot_radius_m
    return route


def task_report(patrol_map: PatrolMap, tasks: dict, robot_radius_m: float):
    """Run a named task set; returns serializable summaries."""
    out = {}
    for name, spec in tasks.items():
        route = build_route(patrol_map, spec, robot_radius_m)
        s = route.summary()
        s["name"] = name
        out[name] = s
    return out


# ----------------------------------------------------------- photo-room glue


def photo_room_tasks():
    """The five standard route tasks on the photo-room avoidance grid.

    Point choices mirror the lesson-64 task set (four reachable room corners
    / furniture sides plus the bed-footprint goal that the prior map must
    reject) so the generator can be cross-checked against that classification.
    """
    from embodied_learning.experiments.photo_room_navigation import RES as PR_RES
    from embodied_learning.experiments.photo_room_navigation import (
        build_maps,
        classify_tasks,
    )
    from embodied_learning.photo_room import RoomWorld, default_layout

    layout = default_layout()
    world = RoomWorld(layout)
    avoidance, _loc = build_maps(world, layout)
    reachable, unreachable = classify_tasks(world, layout)
    pmap = PatrolMap(occupied=np.asarray(avoidance, dtype=bool), resolution=PR_RES)
    start = (2.70, 0.95)
    tasks = {}
    for i, (name, _s, goal) in enumerate(reachable):
        tasks[name] = PatrolSpec(start, ((name, tuple(goal)),), close_loop=False)
    bed_name, _s, bed_goal = unreachable[0]
    tasks["bed_footprint_unreachable"] = PatrolSpec(
        start, ((bed_name, tuple(bed_goal)),), close_loop=False
    )
    # a multi-point inspection lap closing at the start
    lap_points = tuple((name, tuple(g)) for name, _s, g in reachable)
    tasks["inspection_lap_closed"] = PatrolSpec(start, lap_points, close_loop=True)
    return pmap, tasks


def main():
    parser = argparse.ArgumentParser(description="patrol route generator smoke")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()
    pmap, tasks = photo_room_tasks()
    report = task_report(pmap, tasks, robot_radius_m=0.15)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
