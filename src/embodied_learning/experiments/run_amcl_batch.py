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
        "max_beams": 32,
    },
    "A-dense": {
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 300,
        "min_particles": 100,
        "max_beams": 32,
    },
    "A-budget": {
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 1000,
        "min_particles": 500,
        "max_beams": 64,
    },
}
INPUT_SEEDS = (0, 1, 2)
REPEATS = 3


AMCL_PARAMS = (
    "update_min_d",
    "update_min_a",
    "max_particles",
    "min_particles",
    "max_beams",  # nav2_amcl name; ROS1 AMCL's laser_max_beams silently no-ops
)


def make_launch(params, out_dir):
    """Write a per-run launch file baking in the AMCL parameters.

    The bridge launches THIS file (passed via --launch), so the parameters
    on disk are the single source of truth for what AMCL will run with.
    """
    param_lines = ",\n".join(f'            "{k}": {params[k]!r}' for k in AMCL_PARAMS)
    code = f'''"""Generated launch file: map_server + amcl + lifecycle manager."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

AMCL_PARAMS = {{
{param_lines},
}}


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("map_file"),
        Node(package="nav2_map_server", executable="map_server", name="map_server",
             output="screen",
             parameters=[{{"yaml_filename": LaunchConfiguration("map_file")}},
                         {{"use_sim_time": True}}]),
        Node(package="nav2_amcl", executable="amcl", name="amcl",
             output="screen",
             parameters=[{{"use_sim_time": True}},
                         {{"alpha1": 0.10}}, {{"alpha2": 0.10}}, {{"alpha3": 0.10}},
                         {{"alpha4": 0.10}}, {{"alpha5": 0.10}},
                         {{"set_initial_pose": False}}, {{"scan_topic": "scan"}},
                         {{"map_topic": "map"}}, {{"odom_frame_id": "odom"}},
                         {{"base_frame_id": "base_link"}}, {{"scan_frame_id": "laser"}},
                         AMCL_PARAMS]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
             name="lifecycle_manager_localization", output="screen",
             parameters=[{{"use_sim_time": True}}, {{"autostart": True}},
                         {{"node_names": ["map_server", "amcl"]}}]),
    ])
'''
    lp = out_dir / "launch_generated.py"
    lp.write_text(code, encoding="utf-8")
    return str(lp)


def verify_params(output_dir, params):
    """Hard gate: the bridge must have read back AMCL params matching config."""
    eff_path = output_dir / "params_effective.json"
    if not eff_path.exists():
        return False, "params_effective.json missing"
    try:
        eff = json.loads(eff_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return False, f"unreadable: {e}"
    for k in AMCL_PARAMS:
        if k not in eff:
            return False, f"{k} absent from effective params"
        if abs(float(eff[k]) - float(params[k])) > 1e-9:
            return False, f"{k} effective {eff[k]} != configured {params[k]}"
    return True, "ok"


def run_bridge(frozen_dir, output_dir, launch_file, params):
    """Replay through the bridge using the GIVEN launch file, then verify
    the AMCL parameters actually took effect (raise on mismatch)."""
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
            "--launch",
            str(launch_file),
            "--expect-params",
            json.dumps({k: params[k] for k in AMCL_PARAMS}),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(BASE),
        check=False,
    )
    ok, why = verify_params(output_dir, params)
    if not ok:
        raise RuntimeError(f"parameter gate failed for {output_dir}: {why}")
    npz = output_dir / "amcl_poses.npz"
    if result.returncode != 0 or not npz.exists():
        return None
    with np.load(npz) as z:
        poses = {k: z[k].copy() for k in z.files}
    # a completed replay with ZERO recorded poses is a failed run (the
    # lifecycle bring-up race), not a valid data point
    if len(poses.get("x", [])) == 0:
        return None
    return poses


def score(amcl, truth, timestamps):
    """Always returns the full key set; missing data yields None fields."""
    out = {
        "mean_m": None,
        "p95_m": None,
        "end_m": None,
        "n_updates": 0,
        "coverage": None,
    }
    if amcl is None or len(amcl.get("x", [])) == 0:
        return out
    errs = []
    for i in range(len(amcl["x"])):
        idx = int(np.argmin(np.abs(timestamps - amcl["stamp"][i])))
        if idx < len(truth):
            errs.append(math.hypot(amcl["x"][i] - truth[idx, 0], amcl["y"][i] - truth[idx, 1]))
    if not errs:
        return out
    ea = np.asarray(errs)
    out.update(
        {
            "mean_m": float(ea.mean()),
            "p95_m": float(np.percentile(ea, 95)),
            "end_m": float(ea[-1]),
            "n_updates": len(ea),
            "coverage": len(ea) / len(truth),
        }
    )
    return out


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AMCL batch runner with parameter gate")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="gate mode: one run per config on input seed 0 only (verify params differ)",
    )
    cli = parser.parse_args()

    out_root = RESULTS / "p3_amcl_batch_v2"
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    input_seeds = (0,) if cli.gate else INPUT_SEEDS
    repeats = (0,) if cli.gate else range(REPEATS)

    for gname, params in CONFIGS.items():
        print(f"\n=== {gname} ===")
        for si in input_seeds:
            frozen = RESULTS / f"amcl_bridge_v2_s{si}"
            if not frozen.exists():
                continue
            data = dict(np.load(frozen / "frozen_input.npz", allow_pickle=False))
            truth, timestamps = data["truth"], data["timestamps"]
            cfg_dir = out_root / f"{gname}_s{si}"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            launch_py = make_launch(params, cfg_dir)

            for rep in repeats:
                run_dir = cfg_dir / f"r{rep}"
                print(f"  {gname} s{si} r{rep}...")
                amcl = None
                wall = 0.0
                for attempt in (1, 2):  # one retry: lifecycle bring-up race
                    t0 = time.perf_counter()
                    try:
                        amcl = run_bridge(frozen, run_dir, launch_py, params)
                        wall = time.perf_counter() - t0
                        break
                    except RuntimeError as e:
                        wall = time.perf_counter() - t0
                        print(f"    attempt {attempt} FAILED: {e}")
                        time.sleep(3.0)  # let the previous DDS graph settle
                sc = score(amcl, truth, timestamps)
                if amcl is None:
                    print(f"    result FAILED after retries (wall {wall:.1f}s)")
                    all_rows.append(
                        {
                            "group": gname,
                            "input_seed": si,
                            "repeat": rep,
                            "result": "failed",
                            "score": sc,
                            "params": params,
                            "wall_s": round(wall, 1),
                        }
                    )
                    if cli.gate:
                        raise SystemExit(
                            f"gate FAILED: {gname} s{si} r{rep} produced no valid run"
                        )
                    continue
                mean_str = f"{sc['mean_m']:.4f}" if sc["mean_m"] is not None else "None"
                cov_str = f"{sc['coverage']:.1%}" if sc["coverage"] is not None else "n/a"
                print(
                    f"    mean {mean_str} updates {sc['n_updates']} "
                    f"coverage {cov_str} wall {wall:.1f}s"
                )
                all_rows.append(
                    {
                        "group": gname,
                        "input_seed": si,
                        "repeat": rep,
                        "result": "ok",
                        "score": sc,
                        "params": params,
                        "wall_s": round(wall, 1),
                    }
                )
                time.sleep(2.0)  # DDS settle between runs

    (out_root / "amcl_results.json").write_text(
        json.dumps(all_rows, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print("\n=== summary ===")
    for gname in CONFIGS:
        valid = [
            r
            for r in all_rows
            if r["group"] == gname and r.get("result") == "ok" and r["score"].get("mean_m")
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
