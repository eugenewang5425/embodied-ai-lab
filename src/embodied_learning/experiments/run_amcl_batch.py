"""Run all AMCL configurations on frozen inputs inside WSL.

Must be run inside WSL with ROS 2 Jazzy sourced.
"""

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

BASE = Path("/mnt/d/项目/具身人工智能")
RESULTS = BASE / "results"

CONFIGS = {
    "A-base": {
        "update_min_d": 0.10,
        "update_min_a": 0.10,
        "max_particles": 300,
        "min_particles": 100,
        "laser_max_beams": 32,
    },
    "A-dense": {
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 300,
        "min_particles": 100,
        "laser_max_beams": 32,
    },
    "A-budget": {
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 1000,
        "min_particles": 500,
        "laser_max_beams": 64,
    },
}
INPUT_SEEDS = (0, 1, 2)
REPEATS = 3


def make_launch(config_name, params, map_file, out_dir):
    code = f'\nfrom launch import LaunchDescription\nfrom launch_ros.actions import Node\n\ndef generate_launch_description():\n    return LaunchDescription([\n        Node(package="nav2_map_server", executable="map_server", name="map_server",\n             output="screen", parameters=[{{"yaml_filename": "{map_file}"}}, {{"use_sim_time": True}}]),\n        Node(package="nav2_amcl", executable="amcl", name="amcl",\n             output="screen", parameters=[{{"use_sim_time": True}},\n             {{"alpha1": 0.10}}, {{"alpha2": 0.10}}, {{"alpha3": 0.10}}, {{"alpha4": 0.10}}, {{"alpha5": 0.10}},\n             {{"set_initial_pose": False}}, {{"scan_topic": "scan"}}, {{"map_topic": "map"}},\n             {{"odom_frame_id": "odom"}}, {{"base_frame_id": "base_link"}}, {{"scan_frame_id": "laser"}},\n             {{"update_min_d": {params["update_min_d"]}}}, {{"update_min_a": {params["update_min_a"]}}},\n             {{"max_particles": {params["max_particles"]}}}, {{"min_particles": {params["min_particles"]}}},\n             {{"laser_max_beams": {params["laser_max_beams"]}}}]),\n        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",\n             name="lifecycle_manager_localization", output="screen",\n             parameters=[{{"use_sim_time": True}}, {{"autostart": True}},\n             {{"node_names": ["map_server", "amcl"]}}]),\n    ])\n'
    lp = out_dir / f"launch_{config_name}.py"
    lp.write_text(code, encoding="utf-8")
    return str(lp)


def run_bridge(frozen_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    bridge = BASE / "src" / "embodied_learning" / "experiments" / "amcl_ros2_bridge.py"
    result = subprocess.run(
        [
            "python3",
            str(bridge),
            "--input",
            str(frozen_dir),
            "--output",
            str(output_dir),
            "--replay-rate",
            "100",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(BASE),
    )
    npz = output_dir / "amcl_poses.npz"
    if result.returncode != 0 or not npz.exists():
        return None
    with np.load(npz) as z:
        return {k: z[k].copy() for k in z.files}


def score(amcl, truth, timestamps):
    if amcl is None or len(amcl["x"]) == 0:
        return {"mean_m": None, "n_updates": 0}
    errs = []
    for i in range(len(amcl["x"])):
        idx = int(np.argmin(np.abs(timestamps - amcl["stamp"][i])))
        if idx < len(truth):
            errs.append(math.hypot(amcl["x"][i] - truth[idx, 0], amcl["y"][i] - truth[idx, 1]))
    if not errs:
        return {"mean_m": None, "n_updates": 0}
    ea = np.asarray(errs)
    return {
        "mean_m": float(ea.mean()),
        "p95_m": float(np.percentile(ea, 95)),
        "end_m": float(ea[-1]),
        "n_updates": len(ea),
        "coverage": len(ea) / len(truth),
    }


def main():
    out_root = RESULTS / "p3_amcl_batch"
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for gname, params in CONFIGS.items():
        print(f"\n=== {gname} ===")
        for si in INPUT_SEEDS:
            frozen = RESULTS / f"amcl_bridge_v2_s{si}"
            if not frozen.exists():
                continue
            data = dict(np.load(frozen / "frozen_input.npz", allow_pickle=False))
            truth, timestamps = data["truth"], data["timestamps"]
            map_file = str(frozen / "map.yaml")
            cfg_dir = out_root / f"{gname}_s{si}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            launch_py = make_launch(gname, params, map_file, cfg_dir)

            for rep in range(REPEATS):
                run_dir = cfg_dir / f"r{rep}"
                print(f"  {gname} s{si} r{rep}...")
                t0 = time.perf_counter()
                amcl = run_bridge(frozen, run_dir)
                wall = time.perf_counter() - t0
                sc = score(amcl, truth, timestamps)
                mean_str = f"{sc['mean_m']:.4f}" if sc["mean_m"] is not None else "None"
                print(f"    mean {mean_str} updates {sc['n_updates']} wall {wall:.1f}s")
                all_rows.append(
                    {
                        "group": gname,
                        "input_seed": si,
                        "repeat": rep,
                        "score": sc,
                        "wall_s": round(wall, 1),
                    }
                )

    (out_root / "amcl_results.json").write_text(
        json.dumps(all_rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("\n=== summary ===")
    for gname in CONFIGS:
        valid = [
            r for r in all_rows if r["group"] == gname and r["score"].get("mean_m") is not None
        ]
        if valid:
            means = [r["score"]["mean_m"] for r in valid]
            ups = [r["score"]["n_updates"] for r in valid]
            print(
                f"  {gname}: mean {np.mean(means):.4f} m, updates {np.mean(ups):.0f}, n={len(valid)}"
            )
        else:
            print(f"  {gname}: no successful runs")


if __name__ == "__main__":
    main()
