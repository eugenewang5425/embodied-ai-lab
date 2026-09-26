"""Export a frozen common input archive for dual-localizer comparison.

Reads the photo-room A-group trajectory (truth-assisted control, but the
localizers receive only biased odometry + noisy scans) and produces:
  - occupancy grid map in ROS map_server format (.pgm + .yaml)
  - per-frame LaserScan-equivalent arrays (angles, ranges, timestamps)
  - per-frame odometry poses (biased integration)
  - initial pose and uncertainty
  - ground-truth poses for scoring (not given to localizers)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from embodied_learning.experiments.grid_nav import (
    DT,
    GridNavConfig,
)
from embodied_learning.photo_room import RoomWorld, default_layout

RES = 0.04
MAP_OUT = Path("results/amcl_bridge")


def export_map(world, layout, out_dir):
    """Export the matched-height occupancy grid as ROS map_server files."""
    robot = layout["robot"]
    grid = world.occupancy(robot["lidar_z"] - 0.02, robot["lidar_z"] + 0.02, RES)
    n = grid.shape[0]
    w_m = layout["room"]["width"]
    d_m = layout["room"]["depth"]
    # ROS map: PGM where 0=occupied, 254=free, 205=unknown
    pgm = np.where(grid > 0, 0, 254).astype(np.uint8)
    # ROS map Y axis is flipped (row 0 = top = max Y)
    pgm_img = np.flipud(pgm)
    Image.fromarray(pgm_img).save(out_dir / "map.pgm")
    yaml_data = {
        "image": "map.pgm",
        "resolution": RES,
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "width_cells": n,
        "height_cells": n,
        "width_m": w_m,
        "depth_m": d_m,
        "coordinate_frame": "map; x right, y up (ROS convention); origin at (0,0)",
        "occupied_value": 0,
        "free_value": 254,
        "note": "matched lidar-height slice (0.16-0.20 m) of the photo-room geometry",
    }
    (out_dir / "map.yaml").write_text(
        __import__("yaml").dump(yaml_data, default_flow_style=False), encoding="utf-8"
    )
    return grid, yaml_data


def export_scan_and_odom(world, layout, truth_chain, rng_sensor):
    """Generate noisy scans and biased odometry for a frozen trajectory."""
    robot = layout["robot"]
    offsets = np.arange(64) * (2.0 * math.pi / 64)
    ranges = []
    hits_all = []
    timestamps = []
    for k, pose in enumerate(truth_chain):
        angles = pose[2] + offsets
        r, h = world.scan(pose)
        noisy = np.clip(r + rng_sensor.normal(0.0, 0.01, r.shape), 0.02, 3.95)
        ranges.append(noisy)
        hits_all.append(h)
        timestamps.append(k * DT)
    return np.asarray(ranges), np.asarray(hits_all), np.asarray(timestamps)


def run(output, *, seed=0, log=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    layout = default_layout()
    world = RoomWorld(layout)
    robot = layout["robot"]

    # ---- A-group trajectory (truth-assisted control, frozen)
    grid_cfg = GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1)
    from embodied_learning.experiments.aniso_env import _build_world, patrol_start

    obstacles, walls = _build_world("ISO", seed, grid_cfg)
    # a fixed patrol: start -> four quadrants -> back (16 m loop)
    from embodied_learning.experiments.matcher_engineering import collect_patrol_laps

    start_pose = patrol_start(grid_cfg, seed)[0]
    stream = collect_patrol_laps(obstacles, start_pose, MatcherConfigLike(grid_cfg), walls=None)
    if stream["laps_done"] < 1:
        raise RuntimeError("patrol incomplete for export")
    truth = stream["truth"].copy()
    log(f"truth chain: {len(truth)} frames, laps {stream['laps_done']}")

    # ---- map export
    grid, map_yaml = export_map(world, layout, output)
    log(f"map: {grid.shape} cells, occupied {int(grid.sum())}")

    # ---- biased odometry (2% wheel bias)

    odom = [truth[0].copy()]
    for k in range(len(truth) - 1):
        delta = truth[k + 1] - truth[k]
        ds = math.hypot(delta[0], delta[1])
        dth = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
        ds_b = ds * 1.02
        dth_b = dth * 1.02
        mid = odom[-1][2] + dth_b / 2.0
        odom.append(
            np.array(
                [
                    odom[-1][0] + ds_b * math.cos(mid),
                    odom[-1][1] + ds_b * math.sin(mid),
                    odom[-1][2] + dth_b,
                ]
            )
        )
    odom = np.asarray(odom)

    # ---- noisy scans at truth poses (frozen, same for both localizers)
    rng_sensor = np.random.default_rng([seed, 71000])
    ranges, hits_all, timestamps = export_scan_and_odom(world, layout, truth, rng_sensor)

    # ---- initial pose and uncertainty
    init_pose = [float(truth[0][0]), float(truth[0][1]), float(truth[0][2])]
    init_spread = 0.04

    # ---- ground truth (scoring only, NOT given to localizers)
    truth_list = truth[:, :3].tolist()

    # ---- save everything
    np.savez_compressed(
        output / "frozen_input.npz",
        truth=np.asarray(truth_list),
        odom=odom,
        ranges=ranges,
        hits=hits_all,
        timestamps=timestamps,
        map_grid=grid,
        init_pose=np.array(init_pose),
        init_spread=np.array(init_spread),
    )
    manifest = {
        "experiment": "amcl_bridge_frozen_input",
        "source_experiment": "photo_room_navigation_v3",
        "source_commit": "dfc5721",
        "seed": seed,
        "trajectory_source": "A-group patrol (truth-assisted control, frozen for replay)",
        "n_frames": len(truth),
        "dt_s": DT,
        "map": map_yaml,
        "init_pose": init_pose,
        "init_spread_m": init_spread,
        "sensor_sigma_m": 0.01,
        "wheel_bias": 0.02,
        "scan_angles": {
            "n_rays": 64,
            "start_rad": 0.0,
            "step_rad": 2 * math.pi / 64,
            "max_range_m": 4.0,
            "no_return": "range = max_range, hit = false",
        },
        "odom_model": "biased integration (2% wheel bias), own heading",
        "coordinate_frame": "map; x right, y up; origin at (0,0)",
        "localizers": ["self_pf", "nav2_amcl"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"frozen input saved to {output}")
    return manifest


def MatcherConfigLike(base):
    """Minimal shim matching the collect_patrol_laps config interface."""
    from embodied_learning.experiments.matcher_engineering import MatcherConfig

    return MatcherConfig(base=base)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="export frozen AMCL bridge input")
    parser.add_argument("--output", default=str(MAP_OUT))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
