"""Recorded closed-loop campus patrol; geometry, scans, plans and scores agree.

Four paired controllers use the same prior avoidance map and task list. Only
localization differs. Ground truth is confined to simulation/scoring and the
explicit diagnostic group. RGB images are presentation only, not localization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_learning.experiments.grid_nav import Obstacle, astar, cast_rays, inflate, smooth_path
from embodied_learning.experiments.map_localization import (
    ParticleFilter,
    occupancy_from_observations,
)
from embodied_learning.experiments.photo_room_navigation import (
    drive_replay_points,
    propagate_motion,
)
from embodied_learning.odometry import estimate_poses

METHODS = {
    "truth": ("知道真实位置", "仿真参考：直接知道位置，用来检查任务和控制器是否可行。"),
    "odom": ("只靠轮子推算", "基线：累计轮子转动推算位置；轮子读数偏大后，误差会逐渐积累。"),
    "prior": (
        "雷达＋参考地图",
        "把当前雷达点与事先准确的地图对齐，修正轮子推算的位置；需要已知地图。",
    ),
    "self": (
        "雷达＋自建地图",
        "同样用雷达定位，但地图来自另一次有里程计偏差的建图巡游；考察地图画歪后的影响。",
    ),
}
COLORS = {"truth": "#147baa", "odom": "#c77600", "prior": "#138257", "self": "#b72e82"}
SIZE, RES, DT, RADIUS = 24.0, 0.10, 0.12, 0.25
RAYS, RANGE = 64, 8.0
OFFSETS = np.arange(RAYS) * (2 * np.pi / RAYS)
TASKS = (
    ("P1 南门岗亭", (12.0, 2.0)),
    ("P2 停车区", (22.0, 2.0)),
    ("P3 东侧设备间", (22.0, 12.0)),
    ("P4 中央路口", (12.0, 12.0)),
    ("P5 北侧消防点", (12.0, 22.0)),
    ("P6 西北仓库", (2.0, 22.0)),
    ("P7 西侧配电点", (2.0, 12.0)),
    ("P8 返回出发点", (2.0, 2.0)),
)


@dataclass(frozen=True)
class Config:
    particles: int = 300
    wheel_bias: float = 0.02
    sensor_sigma_m: float = 0.03
    max_steps: int = 2600
    speed_m_s: float = 0.8
    estimate_arrival_m: float = 0.30
    true_arrival_m: float = 0.45
    dwell_frames: int = 5
    replan_frames: int = 20


def world():
    """Metric campus abstraction, not a reconstruction of a real site."""
    geometry, styles = [], []

    def rect(name, x, y, hx, hy, height, color, kind="building"):
        geometry.append(Obstacle("rect", x, y, hx, hy))
        styles.append({"name": name, "height": height, "color": color, "kind": kind})

    for args in (
        (12, 0.10, 12, 0.10),
        (12, 23.90, 12, 0.10),
        (0.10, 12, 0.10, 12),
        (23.90, 12, 0.10, 12),
    ):
        rect("园区围墙", *args, 1.1, ".60 .66 .68 1", "wall")
    rect("行政楼", 6.6, 6.7, 3.2, 3.0, 4.3, ".70 .78 .85 1")
    rect("研发楼", 17.4, 6.5, 3.0, 2.9, 5.0, ".75 .69 .57 1")
    rect("仓库", 6.6, 17.5, 3.0, 3.1, 3.2, ".60 .70 .76 1")
    rect("设备楼", 17.5, 17.4, 3.1, 3.0, 3.8, ".71 .75 .63 1")
    # Parked vehicles, a utility cabinet and a fenced dead-end create local structure.
    rect("停放车辆", 20.9, 5.2, 0.55, 1.25, 1.15, ".18 .36 .57 1", "car")
    rect("停放车辆", 21.0, 8.4, 0.55, 1.20, 1.20, ".78 .35 .24 1", "car")
    rect("配电柜", 3.1, 13.8, 0.40, 0.60, 1.3, ".22 .43 .44 1", "cabinet")
    rect("支路围栏", 8.6, 21.0, 1.20, 0.12, 1.4, ".45 .52 .56 1", "wall")
    for x, y in ((10.6, 5.0), (10.7, 8.0), (13.4, 17.5), (4.4, 21.0), (19.8, 21.2)):
        geometry.append(Obstacle("circle", x, y, r=0.22))
        styles.append({"name": "树干", "height": 2.2, "color": ".35 .26 .15 1", "kind": "tree"})
    cells = int(SIZE / RES)
    yy, xx = np.mgrid[:cells, :cells]
    xx, yy = (xx + 0.5) * RES, (yy + 0.5) * RES
    occupied = np.zeros((cells, cells), bool)
    for o in geometry:
        if o.kind == "rect":
            occupied |= (np.abs(xx - o.cx) <= o.hx) & (np.abs(yy - o.cy) <= o.hy)
        else:
            occupied |= (xx - o.cx) ** 2 + (yy - o.cy) ** 2 <= o.r**2
    return tuple(geometry), styles, occupied


def plan(safe, start, goal):
    start_cell = tuple(np.floor(np.asarray(start)[::-1] / RES).astype(int))
    goal_cell = tuple(np.floor(np.asarray(goal)[::-1] / RES).astype(int))
    if any(v < 0 or v >= len(safe) for v in (*start_cell, *goal_cell)):
        return None
    cells, _ = astar(safe, start_cell, goal_cell)
    if cells is None:
        return None
    points = np.asarray([[(c + 0.5) * RES, (r + 0.5) * RES] for r, c in smooth_path(cells, safe)])
    points[-1] = goal
    return points


def scan(pose, geometry, noise=None):
    ranges, hits = cast_rays(*pose[:2], pose[2] + OFFSETS, geometry, (), RANGE)
    if noise is not None:
        ranges = np.where(hits, np.clip(ranges + noise, 0.02, RANGE), RANGE)
    return ranges, hits


class CampusPF(ParticleFilter):
    """Existing likelihood-field PF vectorized and extended to an explicit range.

    Likelihood width and recursive weighting follow photo_room_navigation.
    A ray at the range cap is not treated as an obstacle hit.
    """

    def update(self, ranges, hits):
        idx = self.ray_idx
        ang = self.p[:, 2, None] + self.ray_off
        xy = self.p[:, :2, None] + np.stack((np.cos(ang), np.sin(ang)), axis=1) * ranges[idx]
        col, row = np.floor(xy / self.res).astype(int).transpose(1, 0, 2)
        valid = (
            (col >= 0)
            & (row >= 0)
            & (col < self.lf.shape[1])
            & (row < self.lf.shape[0])
            & hits[idx]
        )
        d = self.lf[np.clip(row, 0, self.lf.shape[0] - 1), np.clip(col, 0, self.lf.shape[1] - 1)]
        count = valid.sum(axis=1)
        likelihood = np.where(
            count >= 4, np.exp(-(d * d * valid).sum(axis=1) / (0.08 * np.maximum(count, 1))), 0
        )
        posterior = self.w * likelihood
        self.w = (
            posterior / posterior.sum()
            if posterior.sum() > 1e-300
            else np.full(len(self.w), 1 / len(self.w))
        )
        if self.ess() < len(self.w) / 2:
            self.resample()


def advance(pose, ds, dth):
    mid = pose[2] + dth / 2
    return np.array(
        [
            pose[0] + ds * np.cos(mid),
            pose[1] + ds * np.sin(mid),
            math.atan2(math.sin(pose[2] + dth), math.cos(pose[2] + dth)),
        ]
    )


def mapping_record(geometry, safe, config):
    current = (2.0, 2.0)
    paths = []
    for _, goal in TASKS:
        path = plan(safe, current, goal)
        if path is None:
            raise ValueError(f"infeasible preregistered patrol leg: {goal}")
        paths.extend(path[:-1])
        current = goal
    paths.append(current)
    truth, enc = drive_replay_points(np.asarray(paths))
    odom = estimate_poses(enc * (1 + config.wheel_bias), truth[0])
    rng = np.random.default_rng(9041)
    observations = [scan(p, geometry, rng.normal(0, config.sensor_sigma_m, RAYS)) for p in truth]
    ranges, hits = map(np.asarray, zip(*observations, strict=True))
    base = SimpleNamespace(rays=RAYS, res_m=RES, grid_cells=int(SIZE / RES))
    built = occupancy_from_observations(ranges, hits, odom, range(len(odom)), base).astype(bool)
    projected = occupancy_from_observations(ranges, hits, truth, range(len(truth)), base).astype(
        bool
    )
    return built, projected, {"truth": truth, "odom": odom, "ranges": ranges, "hits": hits}


def run_episode(method, seed, geometry, safe, loc_map, config):
    noise = np.random.default_rng([7301, seed]).normal(
        0, config.sensor_sigma_m, (config.max_steps + 1, RAYS)
    )
    rng = np.random.default_rng([8331, seed])
    truth, odom = np.array([2.0, 2.0, 0.0]), np.array([2.0, 2.0, 0.0])
    pf = (
        CampusPF(loc_map, RES, config.particles, rng, init_pose=truth, spread=0.04, rays_n=32)
        if method in ("prior", "self")
        else None
    )
    mission, dwell, path, wp = 0, 0, None, 0
    ds_prev = dth_prev = 0.0
    records = {
        k: []
        for k in (
            "truth",
            "estimate",
            "odom",
            "ranges",
            "hits",
            "goal",
            "mission",
            "waypoint",
            "commands",
            "plan",
        )
    }
    events, contacts, status = [], 0, "timeout"
    for k in range(config.max_steps + 1):
        ranges, hits = scan(truth, geometry, noise[k])
        if pf is not None:
            if k:
                propagate_motion(
                    pf,
                    ds_prev,
                    dth_prev,
                    rng,
                    0.10 * abs(ds_prev) + 0.002,
                    0.10 * abs(dth_prev) + 0.002,
                )
            pf.update(ranges, hits)
        estimated = (
            truth.copy() if method == "truth" else (odom.copy() if pf is None else pf.estimate())
        )
        goal = np.asarray(TASKS[mission][1])
        dist_est = float(np.linalg.norm(estimated[:2] - goal))
        dwell = dwell + 1 if dist_est < config.estimate_arrival_m else 0
        if dwell >= config.dwell_frames:
            distance = float(np.linalg.norm(truth[:2] - goal))
            events.append(
                {
                    "frame": k,
                    "goal_index": mission,
                    "true_distance_m": distance,
                    "success": distance <= config.true_arrival_m,
                }
            )
            if mission == len(TASKS) - 1:
                status = "completed" if all(e["success"] for e in events) else "false_completion"
            else:
                mission += 1
                goal = np.asarray(TASKS[mission][1])
                path, dwell = None, 0
        if path is None or k % config.replan_frames == 0:
            path = plan(safe, estimated[:2], goal)
            wp = 0
        if path is None and status == "timeout":
            status = "no_path"
        target = goal.copy()
        v = omega = 0.0
        if path is not None:
            while wp < len(path) - 1 and np.linalg.norm(estimated[:2] - path[wp]) < 0.35:
                wp += 1
            target = path[wp]
            bearing = math.atan2(*(target - estimated[:2])[::-1]) - estimated[2]
            bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            omega = float(np.clip(2.0 * bearing, -1.1, 1.1))
            v = config.speed_m_s if abs(bearing) < 0.35 else 0.0
            if dist_est < config.estimate_arrival_m:
                v = omega = 0.0
            forward = (OFFSETS <= 0.40) | (OFFSETS >= 2 * np.pi - 0.40)
            if ranges[forward].min() < RADIUS + config.speed_m_s * DT + 0.12:
                v = 0.0
        # Contact scoring observes current state; it never corrects the estimate.
        if min(o.distance(*truth[:2]) for o in geometry) < RADIUS:
            contacts += 1
            status = "collision_abort"
        if k == config.max_steps or status != "timeout":
            v = omega = 0.0
        packed_path = np.full((32, 2), np.nan)
        if path is not None:
            if len(path) > 32:
                raise ValueError("archive path capacity exceeded")
            packed_path[: len(path)] = path
        values = (
            truth.copy(),
            estimated.copy(),
            odom.copy(),
            ranges,
            hits,
            goal.copy(),
            mission,
            target.copy(),
            (v, omega),
            packed_path,
        )
        for key, value in zip(records, values, strict=True):
            records[key].append(value)
        if k == config.max_steps or status != "timeout":
            break
        ds, dth = v * DT, omega * DT
        truth = advance(truth, ds, dth)
        ds_prev, dth_prev = ds * (1 + config.wheel_bias), dth * (1 + config.wheel_bias)
        odom = advance(odom, ds_prev, dth_prev)
    data = {key: np.asarray(value) for key, value in records.items()}
    error = np.linalg.norm(data["truth"][:, :2] - data["estimate"][:, :2], axis=1)
    report = {
        "method": method,
        "seed": seed,
        "status": status,
        "frames": len(error),
        "elapsed_sim_s": (len(error) - 1) * DT,
        "reached": sum(e["success"] for e in events),
        "announced": len(events),
        "target_count": len(TASKS),
        "contacts": contacts,
        "mean_m": float(error.mean()),
        "p95_m": float(np.percentile(error, 95)),
        "end_m": float(error[-1]),
        "events": events,
        "travelled_m": float(np.linalg.norm(np.diff(data["truth"][:, :2], axis=0), axis=1).sum()),
    }
    return report, data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output, seeds=(0, 1, 2), config=None):
    config = Config() if config is None else config
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    (output / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    geometry, styles, occupied = world()
    safe = ~inflate(occupied, math.ceil((RADIUS + 0.10) / RES))
    protocol = {
        "schema": 1,
        "experiment": "campus_patrol",
        "config": asdict(config),
        "seeds": list(seeds),
        "size_m": SIZE,
        "resolution_m": RES,
        "robot_radius_m": RADIUS,
        "dt_s": DT,
        "lidar_rays": RAYS,
        "lidar_range_m": RANGE,
        "task_points": TASKS,
        "methods": METHODS,
        "wheel_model": "common translation and rotation scale",
        "avoidance_map": "shared prior geometry",
        "mapping_protocol": "separate truth-driven diagnostic coverage, map painted using biased odometry",
        "camera": "rerendered for display only; no RGB localization",
        "styles": styles,
        "geometry": [asdict(o) for o in geometry],
        "source_sha256": digest(__file__),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    built, projected, mapping = mapping_record(geometry, safe, config)
    np.savez_compressed(output / "maps.npz", reference=occupied, built=built, projected=projected)
    np.savez_compressed(output / "mapping.npz", **mapping)
    rows = []
    for seed in seeds:
        for method in METHODS:
            t0 = time.perf_counter()
            row, data = run_episode(
                method, seed, geometry, safe, built if method == "self" else occupied, config
            )
            path = output / f"seed{seed}_{method}.npz"
            np.savez_compressed(path, **data)
            row.update(
                file=path.name, sha256=digest(path), wall_s=round(time.perf_counter() - t0, 3)
            )
            rows.append(row)
            print(
                f"seed{seed} {method}: {row['status']}, {row['reached']}/{len(TASKS)} points, mean={row['mean_m']:.3f}m, {row['wall_s']}s",
                flush=True,
            )
            report = {
                "protocol": protocol,
                "protocol_sha256": digest(output / "protocol.json"),
                "maps_sha256": digest(output / "maps.npz"),
                "mapping_sha256": digest(output / "mapping.npz"),
                "rows": rows,
                "wall_s": time.perf_counter() - started,
            }
            (output / "summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/campus_patrol_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run(args.output, args.seeds)


if __name__ == "__main__":
    main()
