"""Lesson 67: stop-only, LiDAR replanning and LiDAR+depth body-aware replanning."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from embodied_learning.experiments.grid_nav import astar, inflate, smooth_path
from embodied_learning.navigation_geometry import (
    NavigationRig,
    advance,
    body_points,
    depth_points,
    lidar_obstacle_points,
    point_clearance,
    sense,
    stop_path,
    swept_clearance,
    world_points,
)

DT, RES, SIZE = 0.1, 0.1, 12.0
METHODS = {
    "stop": ("只停车", "前方雷达太近就刹车，沿用先验路线，不根据新障碍绕行。"),
    "lidar": (
        "雷达＋车身包络",
        "雷达观测更新通行图，并检查整台车刹停时扫过的空间；平面雷达有高度盲区。",
    ),
    "depth": (
        "雷达＋深度＋车身包络",
        "再加入相机米制深度及近期观测，发现低矮和悬空障碍；不使用 RGB 外观定位。",
    ),
}
COLORS = {"stop": "#c77600", "lidar": "#b72e82", "depth": "#138257"}


def scene(layout, obstacle):
    static = [
        [0, 12, 0, 0.1, 0, 1.5],
        [0, 12, 11.9, 12, 0, 1.5],
        [0, 0.1, 0, 12, 0, 1.5],
        [11.9, 12, 0, 12, 0, 1.5],
    ]
    if layout == "street":
        static += [[3, 5, 0.1, 4.5, 0, 2.5], [7, 9, 7.5, 11.9, 0, 3.0]]
    elif layout == "narrow":
        static += [[0.1, 11.9, 0, 5.0, 0, 1.5], [0.1, 11.9, 7.0, 12, 0, 1.5]]
    elif layout != "plaza":
        raise ValueError(layout)
    z = {"crate": (0, 1.0), "low": (0, 0.14), "beam": (0.42, 0.57)}[obstacle]
    added = [5.6, 6.4, 5.6, 6.4, *z]
    return np.asarray(static, float), np.asarray([*static, added], float)


def occupancy(boxes):
    yy, xx = np.mgrid[:120, :120]
    x, y = (xx + 0.5) * RES, (yy + 0.5) * RES
    grid = np.zeros((120, 120), bool)
    for b in boxes:
        grid |= (x >= b[0]) & (x <= b[1]) & (y >= b[2]) & (y <= b[3])
    return grid


def route(occupied, start, goal, rig):
    safe = ~inflate(occupied, math.ceil((rig.radius + rig.margin_m) / RES))
    cell = lambda p: tuple(np.floor(np.asarray(p)[::-1] / RES).astype(int))
    s, g = cell(start), cell(goal)
    if any(i < 0 or i >= len(safe) for i in (*s, *g)):
        return None
    cells, _ = astar(safe, s, g)
    if cells is None:
        return None
    points = np.array([[(x + 0.5) * RES, (y + 0.5) * RES] for y, x in smooth_path(cells, safe)])
    points[-1] = goal
    return points


class ObservedController:
    """Only prior, sensors and supplied estimate are available here; no world oracle."""

    waypoint_radius_m = 0.35
    path_capacity = 48

    def __init__(self, prior, rig, method):
        self.prior, self.rig, self.method = prior.copy(), rig, method
        self.grid = prior.copy()
        self.voxels = {}
        self.path = None
        self.wp = 0
        self.last_plan = -100

    def make_plan(self, estimate, goal):
        return route(self.grid, estimate[:2], goal, self.rig)

    def tracking_command(self, estimate, velocity):
        """Nominal steering hook; observation, planning and braking stay shared."""
        while (
            self.wp < len(self.path) - 1
            and np.linalg.norm(estimate[:2] - self.path[self.wp]) < self.waypoint_radius_m
        ):
            self.wp += 1
        delta = self.path[self.wp] - estimate[:2]
        bearing = np.arctan2(delta[1], delta[0]) - estimate[2]
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        return np.array([0.6 if abs(bearing) < 0.3 else 0.0, np.clip(2 * bearing, -1.0, 1.0)])

    def step(self, estimate, goal, ranges, hits, depth, valid, frame, velocity):
        rig = self.rig
        points = lidar_obstacle_points(ranges, hits, rig)
        if self.method == "depth":
            points = np.vstack((points, depth_points(depth, valid, rig)))
        if self.method != "stop":
            world = world_points(points, estimate)
            # Static-obstacle protocol: retain observed voxels, no geometry leakage.
            quant = np.floor(world / 0.08).astype(int)
            _, ids = np.unique(quant, axis=0, return_index=True)
            for i in ids:
                self.voxels[tuple(quant[i])] = world[i]
            if self.voxels:
                memory = np.array(list(self.voxels.values()))
                cell = np.floor(memory[:, :2] / RES).astype(int)
                keep = ((cell >= 0) & (cell < len(self.grid))).all(axis=1)
                self.grid[cell[keep, 1], cell[keep, 0]] = True
                points = body_points(memory, estimate)
                points = points[np.linalg.norm(points[:, :2], axis=1) < 2.5]
        if self.path is None or frame - self.last_plan >= 10:
            self.path = self.make_plan(estimate, goal)
            self.wp, self.last_plan = 0, frame
        if self.path is None:
            return (0.0, 0.0), "no_observed_route", float("inf")
        desired = self.tracking_command(estimate, velocity)
        reason, clearance = "following", float("inf")
        if self.method == "stop":
            angles = np.arange(len(ranges)) * (2 * np.pi / len(ranges))
            front = (angles <= 0.4) | (angles >= 2 * np.pi - 0.4)
            if ranges[front].min() < 0.466:
                desired[0], reason = 0.0, "frontal_brake"
        else:
            reachable = velocity + np.clip(
                desired - velocity,
                [-rig.brake_m_s2 * DT, -rig.angular_brake_rad_s2 * DT],
                [rig.brake_m_s2 * DT, rig.angular_brake_rad_s2 * DT],
            )
            clearance = point_clearance(points, stop_path(*reachable, rig), rig, rig.margin_m)
            if clearance <= 0:
                desired[:], reason = 0.0, "swept_body_brake"
        return desired, reason, clearance


def episode(layout, obstacle, method, seed, max_steps=600, controller_factory=ObservedController):
    rig = NavigationRig()
    prior_boxes, actual_boxes = scene(layout, obstacle)
    controller = controller_factory(occupancy(prior_boxes), rig, method)
    rng = np.random.default_rng([6700, seed])
    actual, estimate = np.array([1.5, 6.0, 0.0]), np.array([1.5, 6.0, 0.0])
    goal, velocity = np.array([10.5, 6.0]), np.zeros(2)
    records = {
        k: []
        for k in (
            "truth",
            "estimate",
            "ranges",
            "hits",
            "depth",
            "depth_valid",
            "goal",
            "mission",
            "plan",
            "commands",
            "reason",
            "safety_clearance",
        )
    }
    clearances, times, status, progress, best = [], [], "timeout", 0, 20.0
    for k in range(max_steps + 1):
        ranges, hits, depth, valid = sense(actual_boxes, actual, rig, rng)
        t0 = time.perf_counter()
        command, reason, clearance = controller.step(
            estimate, goal, ranges, hits, depth, valid, k, velocity
        )
        times.append(time.perf_counter() - t0)
        if (
            status == "timeout"
            and np.linalg.norm(actual[:2] - goal) < 0.25
            and abs(velocity[0]) < 0.05
        ):
            status = "completed"
        if np.linalg.norm(estimate[:2] - goal) < 0.25:
            command = np.zeros(2)
        if status == "timeout" and reason == "no_observed_route":
            status = "no_path"
        distance = float(np.linalg.norm(actual[:2] - goal))
        if distance < best - 0.05:
            best, progress = distance, k
        if status == "timeout" and k - progress > 120:
            status = "stalled"
        if k == max_steps or status != "timeout":
            command = np.zeros(2)
        commanded = velocity + np.clip(
            np.asarray(command) - velocity,
            [-rig.brake_m_s2 * DT, -rig.angular_brake_rad_s2 * DT],
            [rig.brake_m_s2 * DT, rig.angular_brake_rad_s2 * DT],
        )
        # Terminal sample has no outgoing executed command.
        if status != "timeout" or k == max_steps:
            commanded = np.zeros(2)
        path = np.full((controller.path_capacity, 2), np.nan)
        if controller.path is not None:
            if len(controller.path) > len(path):
                raise ValueError("path exceeds archive capacity")
            path[: len(controller.path)] = controller.path
        for key, value in zip(
            records,
            (
                actual.copy(),
                estimate.copy(),
                ranges,
                hits,
                depth,
                valid,
                goal,
                0,
                path,
                commanded,
                reason,
                clearance,
            ),
            strict=True,
        ):
            records[key].append(value)
        if status != "timeout" or k == max_steps:
            break
        gap = swept_clearance(actual, *commanded, DT, actual_boxes, rig)
        clearances.append(gap)
        actual = advance(actual, *commanded, DT)
        # Exact wheel odometry is a controlled diagnostic condition in lesson 67.
        estimate = advance(estimate, *commanded, DT)
        velocity = commanded
        if gap <= 0:
            status = "collision_abort"
    data = {key: np.asarray(value) for key, value in records.items()}
    row = {
        "method": method,
        "seed": seed,
        "layout": layout,
        "obstacle": obstacle,
        "status": status,
        "reached": int(status == "completed"),
        "target_count": 1,
        "announced": int(status == "completed"),
        "contacts": int(status == "collision_abort"),
        "mean_m": 0.0,
        "p95_m": 0.0,
        "end_m": 0.0,
        "frames": len(data["truth"]),
        "elapsed_sim_s": (len(data["truth"]) - 1) * DT,
        "events": [{"frame": len(data["truth"]) - 1, "success": True}]
        if status == "completed"
        else [],
        "min_clearance_m": min(clearances, default=None),
        "controller_p95_ms": float(np.percentile(times, 95) * 1000),
        "travelled_m": float(np.linalg.norm(np.diff(data["truth"][:, :2], axis=0), axis=1).sum()),
        "stop_frames": int(np.sum(np.abs(data["commands"][:, 0]) < 0.01)),
    }
    return row, data, controller.grid


def run(
    output,
    seeds=(0, 1, 2),
    layouts=("plaza", "street", "narrow"),
    obstacles=("crate", "low", "beam"),
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol = {
        "experiment": "observed_navigation_lesson67",
        "rig": NavigationRig().to_dict(),
        "seeds": list(seeds),
        "layouts": layouts,
        "obstacles": obstacles,
        "methods": METHODS,
        "dt_s": DT,
        "resolution_m": RES,
        "size_m": SIZE,
        "localization": "exact wheel odometry diagnostic; zero bias/slip",
        "depth": "80x60 pinhole ray-depth, metric optical-axis; no RGB appearance localization",
        "environment": "static introduced obstacles absent from prior; no dynamic clearing",
        "collision": "box SAT, motion samples <=1cm corner travel; oracle only scores",
        "source_sha256": digest(__file__),
        "dependency_sha256": {
            "navigation_geometry.py": digest(Path(__file__).parents[1] / "navigation_geometry.py"),
            "grid_nav.py": digest(Path(__file__).with_name("grid_nav.py")),
        },
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8"
    )
    rows = []
    started = time.perf_counter()
    for layout in layouts:
        for obstacle in obstacles:
            prior, actual = scene(layout, obstacle)
            for seed in seeds:
                for method in METHODS:
                    row, data, observed = episode(layout, obstacle, method, seed)
                    file = f"{layout}_{obstacle}_s{seed}_{method}.npz"
                    np.savez_compressed(
                        out / file,
                        **data,
                        observed_map=observed,
                        prior_map=occupancy(prior),
                        actual_boxes=actual,
                    )
                    row.update(file=file, sha256=digest(out / file))
                    rows.append(row)
                    print(
                        layout,
                        obstacle,
                        seed,
                        method,
                        row["status"],
                        row["elapsed_sim_s"],
                        flush=True,
                    )
                    summary = {
                        "protocol": protocol,
                        "protocol_sha256": digest(out / "protocol.json"),
                        "rows": rows,
                        "wall_s": time.perf_counter() - started,
                    }
                    (out / "summary.json").write_text(
                        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
                    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/observed_navigation_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--layouts", nargs="+", default=["plaza", "street", "narrow"])
    p.add_argument("--obstacles", nargs="+", default=["crate", "low", "beam"])
    a = p.parse_args()
    run(a.output, a.seeds, a.layouts, a.obstacles)


if __name__ == "__main__":
    main()
