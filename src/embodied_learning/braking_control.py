"""Lesson 69 round three: matched discrete stopping and curvature speed budgets.

The previous controllers remain frozen. This experimental adapter reproduces
their observation update, and exposes a legacy bridge for bitwise prefix checks.
No world geometry, truth pose, seed or scene name is available to this controller.
"""

import math

import numpy as np

from embodied_learning.navigation_geometry import (
    advance,
    body_points,
    depth_points,
    lidar_obstacle_points,
    point_clearance,
    world_points,
)
from embodied_learning.path_tracking import RetainedTangentController, path_frame

DT = 0.1
MODES = {
    "legacy_brake": (False, False),
    "discrete": (False, False),
    "coupled": (True, False),
    "speed": (False, True),
    "joint_brake": (True, True),
}


def reachable(velocity, desired, rig, dt=DT):
    limits = np.array([rig.brake_m_s2, rig.angular_brake_rad_s2]) * dt
    return np.asarray(velocity) + np.clip(np.asarray(desired) - velocity, -limits, limits)


def brake_step(velocity, rig, coupled=False, dt=DT):
    velocity = np.asarray(velocity, float)
    if not coupled:
        return reachable(velocity, np.zeros(2), rig, dt)
    duration = max(abs(velocity[0]) / rig.brake_m_s2, abs(velocity[1]) / rig.angular_brake_rad_s2)
    if duration <= dt + 1e-12:
        return np.zeros(2)
    return velocity * (1 - dt / duration)


def braking_path(velocity, rig, coupled=False, latency=0.0, dt=DT):
    """Exact per-command arcs; <=1cm corner travel samples, including rest.

    Hold velocity during a whole number of latency ticks, then apply the same
    brake_step used by execution before each following integration step.
    """
    hold = round(latency / dt)
    if not np.isclose(hold * dt, latency):
        raise ValueError("latency must be a whole number of command ticks")
    pose, commands, poses = np.zeros(3), [], [np.zeros(3)]
    current = np.asarray(velocity, float).copy()
    maximum = (
        hold
        + math.ceil(
            max(abs(current[0]) / rig.brake_m_s2, abs(current[1]) / rig.angular_brake_rad_s2) / dt
        )
        + 2
    )
    for k in range(maximum):
        if k >= hold:
            current = brake_step(current, rig, coupled, dt)
        if np.max(np.abs(current)) < 1e-12:
            break
        steps = max(1, math.ceil((abs(current[0]) + rig.radius * abs(current[1])) * dt / 0.01))
        poses.extend(advance(pose, *current, t) for t in np.linspace(0, dt, steps + 1)[1:])
        pose = advance(pose, *current, dt)
        commands.append(current.copy())
    else:
        raise RuntimeError("braking did not reach rest within acceleration-derived bound")
    return np.asarray(poses), np.asarray(commands).reshape(-1, 2)


def preview_speed(path, estimate, velocity, rig):
    """Budget lateral acceleration over a distance preview, using planned path only."""
    path = np.asarray(path)
    edges = np.diff(path, axis=0)
    length = np.linalg.norm(edges, axis=1)
    valid = length > 1e-8
    if valid.sum() < 2:
        return 0.6, 0.0
    midpoint = (np.r_[0, np.cumsum(length)][:-1] + np.cumsum(length)) / 2
    headings = np.unwrap(np.arctan2(edges[valid, 1], edges[valid, 0]))
    curvature = np.gradient(headings, midpoint[valid])
    i = path_frame(path, estimate)["index"]
    a, edge = path[i], edges[i]
    fraction = np.clip((estimate[:2] - a) @ edge / max(length[i] ** 2, 1e-16), 0, 1)
    station = length[:i].sum() + fraction * length[i]
    horizon = 0.35 + rig.latency_s * abs(velocity[0]) + velocity[0] ** 2 / (2 * rig.brake_m_s2)
    keep = (midpoint[valid] >= station - length[i]) & (midpoint[valid] <= station + horizon)
    peak = max(
        abs(path_frame(path, estimate)["curvature_m_inv"]),
        np.max(np.abs(curvature[keep]), initial=0),
    )
    return min(0.6, math.sqrt(0.04 / max(peak, 1e-9))), float(peak)


class BrakingController(RetainedTangentController):
    def __init__(self, prior, rig, mode):
        super().__init__(prior, rig, "depth")
        self.mode = mode
        self.coupled, self.speed_budget = MODES[mode]
        self.latest = {}

    def tracking_command(self, estimate, velocity):
        desired = super().tracking_command(estimate, velocity)
        cap, curvature = (
            preview_speed(self.path, estimate, velocity, self.rig)
            if self.speed_budget
            else (0.6, 0.0)
        )
        if self.speed_budget:
            old_v = desired[0]
            desired[0] = min(desired[0], cap)
            # Keep the same feedback gains, changing only v*k feed-forward.
            raw = (
                self.tracking_log[-1]["angular_raw"]
                + (desired[0] - old_v) * self.tracking_log[-1]["curvature_m_inv"]
            )
            desired[1] = np.clip(raw, -1, 1)
        self.latest.update(speed_cap=cap, preview_curvature=curvature, nominal=desired.copy())
        return desired

    def observe(self, estimate, ranges, hits, depth, valid):
        points = np.vstack(
            (lidar_obstacle_points(ranges, hits, self.rig), depth_points(depth, valid, self.rig))
        )
        world = world_points(points, estimate)
        quant = np.floor(world / 0.08).astype(int)
        _, ids = np.unique(quant, axis=0, return_index=True)
        for i in ids:
            self.voxels[tuple(quant[i])] = world[i]
        if self.voxels:
            memory = np.array(list(self.voxels.values()))
            cell = np.floor(memory[:, :2] / 0.1).astype(int)
            keep = ((cell >= 0) & (cell < len(self.grid))).all(axis=1)
            self.grid[cell[keep, 1], cell[keep, 0]] = True
            points = body_points(memory, estimate)
            points = points[np.linalg.norm(points[:, :2], axis=1) < 2.5]
        return points

    def step(self, estimate, goal, ranges, hits, depth, valid, frame, velocity):
        self.latest = {
            "speed_cap": np.nan,
            "preview_curvature": np.nan,
            "nominal": np.zeros(2),
            "stop_clearance": np.nan,
        }
        if self.mode == "legacy_brake":
            return super().step(estimate, goal, ranges, hits, depth, valid, frame, velocity)
        points = self.observe(estimate, ranges, hits, depth, valid)
        if self.path is None or frame - self.last_plan >= 10:
            self.path = self.make_plan(estimate, goal)
            self.wp, self.last_plan = 0, frame
        if self.path is None:
            return np.zeros(2), "no_observed_route", float("inf")
        desired = self.tracking_command(estimate, velocity)
        candidate = reachable(velocity, desired, self.rig)
        paths = [
            braking_path(candidate, self.rig, self.coupled, delay)[0]
            for delay in (0.0, self.rig.latency_s)
        ]
        clearance = point_clearance(points, np.vstack(paths), self.rig, self.rig.margin_m)
        stop = braking_path(velocity, self.rig, self.coupled)[0]
        self.latest["stop_clearance"] = point_clearance(points, stop, self.rig, self.rig.margin_m)
        if clearance <= 0:
            return np.zeros(2), "swept_body_brake", clearance
        return desired, "following", clearance
