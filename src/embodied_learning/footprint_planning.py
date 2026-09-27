"""Observation-only footprint planning; no access to hidden scene geometry.

Prior occupied cells retain their square extent. New measurements retain metric
coordinates instead of inflating their rounded 10 cm storage cells. The circle
and rectangle arms use exactly the same input and graph, isolating footprint shape.
"""

import heapq
import math
import time
from itertools import pairwise

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from embodied_learning.experiments.grid_nav import inflate
from embodied_learning.experiments.observed_navigation import ObservedController
from embodied_learning.navigation_geometry import advance


class FootprintField:
    def __init__(self, prior, points, rig, shape="rectangle", resolution=0.1):
        if shape not in ("circle", "rectangle"):
            raise ValueError(shape)
        self.prior, self.rig, self.shape, self.res = prior, rig, shape, resolution
        boundary = prior & ~binary_erosion(prior)
        y, x = np.nonzero(boundary)
        self.cells = np.column_stack(((x + 0.5) * resolution, (y + 0.5) * resolution))
        self.cell_tree = cKDTree(self.cells)
        self.points = np.asarray(points, float).reshape(-1, 2)
        self.point_tree = cKDTree(self.points)
        self.cache = {}

    def free(self, pose):
        key = tuple(np.round(pose, 6))
        if key in self.cache:
            return self.cache[key]
        result = self._free(np.asarray(pose))
        self.cache[key] = result
        return result

    def _free(self, p):
        r = self.rig
        c, s = np.cos(p[2]), np.sin(p[2])
        hx, hy = (r.front_m + r.rear_m) / 2 + r.margin_m, r.half_width_m + r.margin_m
        center = p[:2] + (r.front_m - r.rear_m) / 2 * np.array([c, s])
        radius = r.radius + r.margin_m
        extents = (
            np.array([radius, radius])
            if self.shape == "circle"
            else np.array([hx * abs(c) + hy * abs(s), hx * abs(s) + hy * abs(c)])
        )
        origin = p[:2] if self.shape == "circle" else center
        if np.any(origin - extents <= 0) or np.any(
            origin + extents >= np.array(self.prior.shape[::-1]) * self.res
        ):
            return False
        cell = np.floor(p[:2] / self.res).astype(int)
        if self.prior[cell[1], cell[0]]:
            return False
        ids = self.cell_tree.query_ball_point(origin, max(radius, np.hypot(hx, hy)) + self.res)
        delta = self.cells[ids] - origin
        half = self.res / 2
        if self.shape == "circle":
            if np.any(np.linalg.norm(np.maximum(np.abs(delta) - half, 0), axis=1) <= radius):
                return False
            return self.point_tree.query(p[:2])[0] > radius
        local = delta @ np.array([[c, -s], [s, c]])
        if len(delta) and np.any(
            (np.abs(delta[:, 0]) <= hx * abs(c) + hy * abs(s) + half)
            & (np.abs(delta[:, 1]) <= hx * abs(s) + hy * abs(c) + half)
            & (np.abs(local[:, 0]) <= hx + half * (abs(c) + abs(s)))
            & (np.abs(local[:, 1]) <= hy + half * (abs(c) + abs(s)))
        ):
            return False
        ids = self.point_tree.query_ball_point(center, np.hypot(hx, hy))
        local = (self.points[ids] - center) @ np.array([[c, -s], [s, c]])
        return not np.any((np.abs(local[:, 0]) <= hx) & (np.abs(local[:, 1]) <= hy))

    def turn(self, xy, a, b):
        delta = math.atan2(math.sin(b - a), math.cos(b - a))
        n = max(1, math.ceil(abs(delta) * self.rig.radius / 0.015))
        return all(self.free([*xy, a + delta * t]) for t in np.linspace(0, 1, n + 1))

    def segment(self, a, b):
        distance = np.linalg.norm(np.asarray(b) - a)
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        n = max(1, math.ceil(distance / 0.02))
        return all(
            self.free([*(np.asarray(a) + (np.asarray(b) - a) * t), yaw])
            for t in np.linspace(0, 1, n + 1)
        )


def footprint_route(field, start, goal, step=0.20, max_expansions=10000):
    """Hybrid A*: sampled curved motions plus turn-in-place, then safe shortcuts.

    The graph is a planner, not the simulated trajectory. Existing steering and
    braking still determine the executed motion and independent collision score.
    """
    start, goal = np.asarray(start), np.asarray(goal)
    if not field.free(start):
        return None, {"reason": "footprint_start_blocked", "expanded": 0}
    # Smooth lane-change seeds keep a gentle curve all the way through the
    # unchanged lookahead tracker. They use only the same field, not scene labels.
    distance = np.linalg.norm(goal - start[:2])
    angle = math.atan2(*(goal - start[:2])[::-1])
    if distance > 0.5 and field.turn(start[:2], start[2], angle):
        forward, side = (
            np.array([np.cos(angle), np.sin(angle)]),
            np.array([-np.sin(angle), np.cos(angle)]),
        )
        ramp = min(4.0, distance / 2.2)

        def lane(x, offset):
            a = np.clip(x / ramp, 0, 1)
            b = np.clip((distance - x) / ramp, 0, 1)
            factor = (3 * a * a - 2 * a * a * a) * (3 * b * b - 2 * b * b * b)
            derivative = np.where(x < ramp, (6 * a - 6 * a * a) / ramp, 0.0) - np.where(
                x > distance - ramp, (6 * b - 6 * b * b) / ramp, 0.0
            )
            xy = start[:2] + x[:, None] * forward + (offset * factor)[:, None] * side
            return np.column_stack((xy, angle + np.arctan(offset * derivative)))

        samples = np.linspace(0, distance, max(2, math.ceil(distance / 0.02) + 1))
        for offset in [0.0] + [sign * a for a in np.arange(0.05, 2.01, 0.05) for sign in (-1, 1)]:
            curve = lane(samples, offset)
            if all(field.free(p) for p in curve):
                sampled = lane(
                    np.linspace(0, distance, min(255, math.ceil(distance / 0.04) + 1)), offset
                )[:, :2]
                if all(field.segment(a, b) for a, b in pairwise(sampled)):
                    return sampled, {"reason": "smooth_observed_route", "expanded": 0}
    # A relaxed 2D obstacle-distance heuristic avoids spending the whole budget
    # pushing into the front face of a box. Final feasibility still uses poses.
    guide_grid = field.prior.copy()
    ids = np.floor(field.points / field.res).astype(int)
    valid = ((ids >= 0) & (ids < np.array(guide_grid.shape[::-1]))).all(axis=1)
    guide_grid[ids[valid, 1], ids[valid, 0]] = True
    blocked = inflate(guide_grid, 2)
    potential = np.full(blocked.shape, np.inf)
    gy, gx = np.floor(goal[::-1] / field.res).astype(int)
    potential[gy, gx] = 0
    frontier = [(0.0, gy, gx)]
    while frontier:
        d, y, x = heapq.heappop(frontier)
        if d > potential[y, x]:
            continue
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            yy, xx = y + dy, x + dx
            dd = d + field.res * math.hypot(dy, dx)
            if (
                0 <= yy < len(blocked)
                and 0 <= xx < blocked.shape[1]
                and not blocked[yy, xx]
                and dd < potential[yy, xx]
            ):
                potential[yy, xx] = dd
                heapq.heappush(frontier, (dd, yy, xx))

    def heuristic(p):
        y, x = np.floor(p[:2][::-1] / field.res).astype(int)
        if 0 <= y < len(potential) and 0 <= x < potential.shape[1] and np.isfinite(potential[y, x]):
            return potential[y, x]
        return np.linalg.norm(goal - p[:2]) + 1.0

    # Quantization merges search states; their stored metric poses are never
    # snapped. Curved primitives can straighten while entering a narrow lane.
    angle_step = np.pi / 16
    identify = lambda p: (round(p[0] / 0.025), round(p[1] / 0.025), round(p[2] / angle_step) % 32)
    initial = identify(start)
    queue = [(float(np.linalg.norm(goal - start[:2])), 0.0, initial)]
    costs, parent = {initial: 0.0}, {}
    poses = {initial: start.copy()}
    expanded = 0
    last = None
    while queue and expanded < max_expansions:
        _, cost, node = heapq.heappop(queue)
        if cost > costs[node] + 1e-9:
            continue
        expanded += 1
        pose = poses[node]
        xy, yaw = pose[:2], pose[2]
        target_yaw = math.atan2(goal[1] - xy[1], goal[0] - xy[0])
        if (
            np.linalg.norm(goal - xy) < 0.25
            and field.turn(xy, yaw, target_yaw)
            and field.segment(xy, goal)
        ):
            last = node
            break
        center_heading = round(yaw / angle_step)
        turns = [
            math.atan2(
                math.sin((center_heading + d) * angle_step - yaw),
                math.cos((center_heading + d) * angle_step - yaw),
            )
            for d in (0, 1, -1)
        ]
        for distance, turn in [(step, a) for a in turns] + [(0, a) for a in turns if abs(a) > 1e-8]:
            destination = advance(pose, distance, turn, 1.0)
            nxt = identify(destination)
            newcost = cost + distance + 0.18 * abs(turn)
            if newcost >= costs.get(nxt, float("inf")):
                continue
            samples = max(2, math.ceil((distance + field.rig.radius * abs(turn)) / 0.02))
            if not field.free(destination) or not all(
                field.free(advance(pose, distance, turn, t)) for t in np.linspace(0, 1, samples + 1)
            ):
                continue
            costs[nxt], parent[nxt], poses[nxt] = newcost, node, destination
            heapq.heappush(queue, (newcost + 1.3 * heuristic(destination), newcost, nxt))
    if last is None:
        return None, {
            "reason": "search_budget" if queue else "no_footprint_route",
            "expanded": expanded,
        }
    nodes = [last]
    while nodes[-1] != initial:
        nodes.append(parent[nodes[-1]])
    path = np.vstack(([poses[n][:2] for n in reversed(nodes)], goal))
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8]]
    selected, current, yaw = [path[0]], 0, start[2]
    while current < len(path) - 1:
        found = False
        for j in range(len(path) - 1, current, -1):
            angle = math.atan2(*(path[j] - path[current])[::-1])
            # Validate the turn onto the shortcut and the next outgoing edge.
            next_angle = math.atan2(*(path[j + 1] - path[j])[::-1]) if j + 1 < len(path) else angle
            if (
                field.turn(path[current], yaw, angle)
                and field.segment(path[current], path[j])
                and field.turn(path[j], angle, next_angle)
            ):
                found = True
                break
        if not found:
            return None, {"reason": "tracking_conversion_blocked", "expanded": expanded}
        selected.append(path[j])
        current, yaw = j, angle
    return np.array(selected), {"reason": "planned", "expanded": expanded}


class FootprintController(ObservedController):
    path_capacity = 256

    def __init__(self, prior, rig, method, shape="rectangle"):
        super().__init__(prior, rig, method)
        self.shape = shape
        self.planning_log = []

    def make_plan(self, estimate, goal):
        started = time.perf_counter()
        points = np.array(list(self.voxels.values())).reshape(-1, 3)[:, :2]
        field = FootprintField(self.prior, points, self.rig, self.shape)
        if self.path is not None:
            remaining = np.vstack((estimate[:2], self.path[self.wp :]))
            if field.free(estimate) and all(field.segment(a, b) for a, b in pairwise(remaining)):
                self.planning_log.append(
                    {
                        "reason": "retained_valid_route",
                        "expanded": 0,
                        "wall_s": time.perf_counter() - started,
                    }
                )
                return remaining
        path, info = footprint_route(field, estimate, goal)
        self.planning_log.append({**info, "wall_s": time.perf_counter() - started})
        return path
