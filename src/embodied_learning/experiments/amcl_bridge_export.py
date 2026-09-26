"""Export 10m grid_nav world data as a frozen common input for dual-localizer comparison.

Scene-consistent: truth, map, scans and odometry all come from the same
photo-room layout (results/map_localization_2026-09-24_v5).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

DT = 0.04
SENSOR_SIGMA = 0.01
WHEEL_BIAS = 0.02
SOURCE_RECORD = "results/map_localization_2026-09-24_v5"
OUTPUT = Path("results/amcl_bridge_v2")


def export(output, *, seed=0, log=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    src = Path(SOURCE_RECORD)
    with np.load(src / "trajectories.npz", allow_pickle=False) as z:
        truth = z["truth_chain"].copy()
        true_grid = z["true_grid"].copy()
    log(f"truth chain: {len(truth)} frames from {src}")
    map_res = 0.1  # true_grid cell size (grid_nav)

    world_size = 10.0
    room_w = world_size
    room_d = world_size

    inside = (
        (truth[:, 0] >= 0.0)
        & (truth[:, 0] <= room_w)
        & (truth[:, 1] >= 0.0)
        & (truth[:, 1] <= room_d)
    )
    if not inside.all():
        raise RuntimeError(f"{int((~inside).sum())} truth poses outside room bounds")
    log(f"scene check: all {len(truth)} poses inside {room_w}x{room_d} m room")

    rng = np.random.default_rng([seed, 71000])
    n_rays = 64
    max_range = 4.0
    grid_res = 0.1  # true_grid cell size
    n_grid = true_grid.shape[0]
    ranges = []
    hits_all = []
    for pose in truth:
        angles = pose[2] + np.arange(n_rays) * (2.0 * math.pi / n_rays)
        dist = np.full(n_rays, max_range)
        hit = np.zeros(n_rays, dtype=bool)
        for s in np.arange(0.1, max_range, 0.05):
            ix = np.floor((pose[0] + s * np.cos(angles)) / grid_res).astype(int)
            iy = np.floor((pose[1] + s * np.sin(angles)) / grid_res).astype(int)
            ok = (ix >= 0) & (ix < n_grid) & (iy >= 0) & (iy < n_grid)
            occ = np.zeros(n_rays, dtype=bool)
            occ[ok] = true_grid[iy[ok], ix[ok]] > 0
            newly = occ & (~hit)
            dist = np.where(newly, s, dist)
            hit = hit | newly
        noisy = np.where(
            hit,
            np.clip(dist + rng.normal(0.0, SENSOR_SIGMA, n_rays), 0.02, max_range - 0.05),
            max_range,
        )
        ranges.append(noisy)
        hits_all.append(hit)
    ranges = np.asarray(ranges)
    hits_all = np.asarray(hits_all)
    ranges[~hits_all] = max_range
    log(f"scans: {ranges.shape}, no-return frac {1 - hits_all.mean():.2%}")

    odom = [truth[0].copy()]
    for k in range(len(truth) - 1):
        delta = truth[k + 1] - truth[k]
        ds = math.hypot(delta[0], delta[1])
        dth = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
        ds_b = ds * (1.0 + WHEEL_BIAS)
        dth_b = dth * (1.0 + WHEEL_BIAS)
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
    odom_err = np.linalg.norm(odom[:, :2] - truth[:, :2], axis=1)
    log(f"odom: mean_err {odom_err.mean():.4f} end {odom_err[-1]:.4f}")

    n = true_grid.shape[0]
    pgm = np.where(true_grid > 0, 0, 254).astype(np.uint8)
    pgm_img = np.flipud(pgm)
    Image.fromarray(pgm_img).save(output / "map.pgm")
    map_yaml = {
        "image": "map.pgm",
        "resolution": map_res,
        "origin": [0.0, 0.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
        "note": "true_grid from map_localization_2026-09-24_v5 (10m world, 0.1m res)",
    }
    (output / "map.yaml").write_text(yaml.dump(map_yaml), encoding="utf-8")
    log(f"map: {n}x{n} cells, occupied {int(true_grid.sum())}")

    init_pose = [float(truth[0][0]), float(truth[0][1]), float(truth[0][2])]
    timestamps = np.arange(len(truth)) * DT

    np.savez_compressed(
        output / "frozen_input.npz",
        truth=truth,
        odom=odom,
        ranges=ranges,
        hits=hits_all,
        timestamps=timestamps,
        map_grid=true_grid,
        init_pose=np.array(init_pose),
        init_spread=np.array(0.04),
    )
    manifest = {
        "experiment": "amcl_bridge_frozen_input_v2",
        "scene": "grid_nav 10m world (ISO arm)",
        "source_record": SOURCE_RECORD,
        "source_commit": "dfc5721",
        "export_seed": seed,
        "n_frames": len(truth),
        "dt_s": DT,
        "map_resolution_m": map_res,
        "map": map_yaml,
        "init_pose": init_pose,
        "init_spread_m": 0.04,
        "sensor_sigma_m": SENSOR_SIGMA,
        "wheel_bias": WHEEL_BIAS,
        "scan": {
            "n_rays": n_rays,
            "start_rad": 0.0,
            "step_rad": 2 * math.pi / n_rays,
            "max_range_m": max_range,
            "no_return_encoding": "range = max_range (4.0)",
        },
        "odom_model": "biased integration (2% wheel bias), own heading integration",
        "coordinate_frame": "map; x right, y up; origin at (0,0)",
        "scene_check": "all truth poses inside room bounds",
        "localizers": ["self_pf", "nav2_amcl"],
        "truth_usage": "scoring only; NOT given to localizers",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(f"frozen input saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    export(args.output, seed=args.seed)


if __name__ == "__main__":
    main()
