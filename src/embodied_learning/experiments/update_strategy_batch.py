"""P3/P4 batch: update strategy and particle budget comparison (45 replays).

Five groups on three frozen inputs, three independent runs each:
  P-full    self-PF, every frame update
  P-gated   self-PF, measurement gated by motion (0.05 m / 0.05 rad)
  A-base    official AMCL with default thresholds
  A-dense   official AMCL with lowered thresholds
  A-budget  A-dense + more particles/beams

Resource measurement (P4) is embedded in each run.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

RES = 0.1  # grid_nav true_grid resolution
N_PARTICLES = 300
DT = 0.04
GATE_D = 0.05  # P-gated translation gate (m)
GATE_A = 0.05  # P-gated rotation gate (rad)

GROUP_CONFIGS = {
    "P-full": {"type": "self_pf", "update_gate_d": None, "update_gate_a": None},
    "P-gated": {"type": "self_pf", "update_gate_d": GATE_D, "update_gate_a": GATE_A},
    "A-base": {
        "type": "amcl",
        "update_min_d": 0.10,
        "update_min_a": 0.10,
        "max_particles": 300,
        "min_particles": 100,
        "max_beams": 32,
    },
    "A-dense": {
        "type": "amcl",
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 300,
        "min_particles": 100,
        "max_beams": 32,
    },
    "A-budget": {
        "type": "amcl",
        "update_min_d": 0.01,
        "update_min_a": 0.01,
        "max_particles": 1000,
        "min_particles": 500,
        "max_beams": 64,
    },
}
INPUT_SEEDS = (0, 1, 2)
REPEATS = 3


def score(estimates, truth):
    n = min(len(estimates), len(truth))
    errors = np.linalg.norm(estimates[:n, :2] - truth[:n, :2], axis=1)
    yaw_err = [
        abs(
            math.atan2(
                math.sin(estimates[i, 2] - truth[i, 2]), math.cos(estimates[i, 2] - truth[i, 2])
            )
        )
        for i in range(n)
    ]
    return {
        "mean_m": float(errors.mean()),
        "p95_m": float(np.percentile(errors, 95)),
        "end_m": float(errors[-1]),
        "max_m": float(errors.max()),
        "mean_yaw_rad": float(np.mean(yaw_err)),
        "n_frames": n,
    }


def run_group(group, data, res, seed):
    """Run one group on one frozen input. Returns (estimates, update_frames, metrics)."""
    cfg = GROUP_CONFIGS[group]
    t0 = time.perf_counter()

    if cfg["type"] == "self_pf":
        from embodied_learning.experiments.amcl_compare import run_self_pf

        est, update_frames = run_self_pf(
            data,
            res,
            seed=seed,
            update_gate_d=cfg.get("update_gate_d"),
            update_gate_a=cfg.get("update_gate_a"),
        )
        cpu_time = time.perf_counter() - t0
        return est, update_frames, {"cpu_s": round(cpu_time, 2), "type": "self_pf"}

    # AMCL groups need ROS 2 (handled separately by the caller)
    raise NotImplementedError(f"AMCL group {group} requires ROS 2 bridge")


def run_batch(output, *, log=print):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for input_seed in INPUT_SEEDS:
        input_dir = Path(f"results/amcl_bridge_v2_s{input_seed}")
        data = dict(np.load(input_dir / "frozen_input.npz", allow_pickle=False))
        truth = data["truth"]
        res = 0.1  # true_grid resolution

        for group in GROUP_CONFIGS:
            cfg = GROUP_CONFIGS[group]
            if cfg["type"] != "self_pf":
                continue  # AMCL groups handled separately

            for repeat in range(REPEATS):
                run_seed = input_seed * 100 + repeat
                t0 = time.perf_counter()
                est, update_frames, res_info = run_group(group, data, res, run_seed)
                wall = time.perf_counter() - t0
                sc = score(est, truth)
                row = {
                    "input_seed": input_seed,
                    "group": group,
                    "repeat": repeat,
                    "result": "ok",
                    "score": sc,
                    "update_count": len(update_frames),
                    "update_frames_sample": update_frames[:20],
                    "resource": {**res_info, "wall_s": round(wall, 2)},
                    "config": {k: v for k, v in cfg.items() if k != "type"},
                }
                all_rows.append(row)
                log(
                    f"  s{input_seed} {group} r{repeat}: "
                    f"mean {sc['mean_m']:.4f} updates {len(update_frames)} cpu {res_info['cpu_s']}s"
                )

        # AMCL groups: write configs for WSL execution
        for group in GROUP_CONFIGS:
            cfg = GROUP_CONFIGS[group]
            if cfg["type"] != "amcl":
                continue
            amcl_cfg = {
                "group": group,
                "input_seed": input_seed,
                "input_dir": str(input_dir),
                "params": cfg,
            }
            cfg_path = output / f"amcl_config_{group}_s{input_seed}.json"
            cfg_path.write_text(json.dumps(amcl_cfg, indent=2), encoding="utf-8")

    # save all self-PF results
    (output / "self_pf_results.json").write_text(
        json.dumps(all_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # summary table
    print("\n=== Self-PF groups summary ===")
    for group in ("P-full", "P-gated"):
        g_rows = [r for r in all_rows if r["group"] == group]
        means = [r["score"]["mean_m"] for r in g_rows]
        updates = [r["update_count"] for r in g_rows]
        cpus = [r["resource"]["cpu_s"] for r in g_rows]
        print(
            f"  {group}: mean {np.mean(means):.4f} m, "
            f"updates {np.mean(updates):.0f}, cpu {np.mean(cpus):.1f}s"
        )

    return all_rows


def main():
    parser = argparse.ArgumentParser(description="P3/P4 batch comparison")
    parser.add_argument("--output", default="results/p3_batch")
    args = parser.parse_args()
    run_batch(args.output)


if __name__ == "__main__":
    main()
