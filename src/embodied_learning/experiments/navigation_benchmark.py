"""Paired map/localization replay benchmark across fixed world and sensor seeds.

The patrol is truth-controlled so that all map arms see the same physical
route.  This evaluates mapping/localization separately from closed-loop
navigation.  The result gate deliberately blocks a Nav2 comparison until
self-built-map localization generalizes beyond one world.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.experiments.map_localization import MapLocConfig, run_experiment


@dataclass(frozen=True)
class BenchmarkPlan:
    arms: tuple[str, ...] = ("ISO", "ANISO")
    world_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sensor_seeds: tuple[int, ...] = (0, 1, 2)
    particles: int = 100

    def __post_init__(self):
        if not self.arms or any(arm not in ("ISO", "ANISO") for arm in self.arms):
            raise ValueError("arms must contain ISO or ANISO")
        if not self.world_seeds or not self.sensor_seeds:
            raise ValueError("both seed axes must be nonempty")
        if any(
            type(seed) is not int or seed < 0 for seed in (*self.world_seeds, *self.sensor_seeds)
        ):
            raise ValueError("seeds must be nonnegative integers")
        if len(set(self.world_seeds)) != len(self.world_seeds):
            raise ValueError("duplicate world seed")
        if len(set(self.sensor_seeds)) != len(self.sensor_seeds):
            raise ValueError("duplicate sensor seed")
        if not (50 <= self.particles <= 2000):
            raise ValueError("particles out of range")


def summarize(rows, plan):
    """Count independent worlds before evaluating the deployment gate."""
    by_arm = {}
    for arm in plan.arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        valid = [r for r in arm_rows if r["status"] == "ok"]
        by_world = defaultdict(list)
        for row in valid:
            by_world[row["world_seed"]].append(row)
        world_scores = []
        for world_seed, world_rows in sorted(by_world.items()):
            complete = len(world_rows) == len(plan.sensor_seeds)
            mean_better = all(r["pf_self_mean_m"] < r["est_mean_m"] for r in world_rows)
            tail_better = all(r["pf_self_p95_m"] < r["est_p95_m"] for r in world_rows)
            map_aligned = all(
                r["map_alignment_precision"] >= 0.70 and r["map_alignment_recall"] >= 0.70
                for r in world_rows
            )
            world_scores.append(
                {
                    "world_seed": world_seed,
                    "complete": complete,
                    "mean_better_all_sensor_seeds": mean_better,
                    "p95_better_all_sensor_seeds": tail_better,
                    "map_aligned_all_sensor_seeds": map_aligned,
                }
            )
        by_arm[arm] = {
            "valid_episodes": len(valid),
            "invalid_episodes": len(arm_rows) - len(valid),
            "valid_worlds": sum(score["complete"] for score in world_scores),
            "mean_errors_m": (
                {
                    key: float(np.mean([row[key] for row in valid]))
                    for key in (
                        "est_mean_m",
                        "pf_true_mean_m",
                        "pf_sensor_truth_mean_m",
                        "pf_self_mean_m",
                        "est_p95_m",
                        "pf_self_p95_m",
                    )
                }
                if valid
                else None
            ),
            "world_scores": world_scores,
        }
    # Precommitted gate: two geometry families, five valid worlds in each,
    # three sensor seeds/world, and at least four worlds/arm improving in all
    # three dimensions.  A pilot can diagnose but cannot pass by small n.
    gate = len(plan.arms) >= 2 and all(
        item["valid_worlds"] >= 5
        and sum(
            score["complete"]
            and score["mean_better_all_sensor_seeds"]
            and score["p95_better_all_sensor_seeds"]
            and score["map_aligned_all_sensor_seeds"]
            for score in item["world_scores"]
        )
        >= 4
        for item in by_arm.values()
    )
    return {"by_arm": by_arm, "selfbuilt_localization_gate_met": bool(gate)}


def run_benchmark(output, plan, log=print):
    output = Path(output)
    if (output / "summary.json").exists():
        raise FileExistsError(f"completed benchmark already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    plan_json = json.dumps(
        {
            "arms": plan.arms,
            "world_seeds": plan.world_seeds,
            "sensor_seeds": plan.sensor_seeds,
            "particles": plan.particles,
        },
        sort_keys=True,
    )
    plan_file = output / "plan.json"
    if plan_file.exists() and plan_file.read_text(encoding="utf-8") != plan_json:
        raise ValueError("existing benchmark directory uses a different plan")
    plan_file.write_text(plan_json, encoding="utf-8")
    rows = []
    for arm in plan.arms:
        for world_seed in plan.world_seeds:
            collector_failure = None
            for sensor_seed in plan.sensor_seeds:
                target = output / f"{arm.lower()}_w{world_seed}_s{sensor_seed}"
                row = {"arm": arm, "world_seed": world_seed, "sensor_seed": sensor_seed}
                if collector_failure:
                    row.update(status="collector_invalid", reason=collector_failure)
                else:
                    cfg = MapLocConfig(
                        seed=world_seed,
                        arm=arm,
                        pf_particles=plan.particles,
                        run_kidnap=False,
                    )
                    try:
                        if target.exists():
                            report = json.loads(
                                (target / "summary.json").read_text(encoding="utf-8")
                            )
                            actual_hash = hashlib.sha256(
                                (target / "trajectories.npz").read_bytes()
                            ).hexdigest()
                            if report["trajectories_sha256"] != actual_hash:
                                raise ValueError("existing trajectory checksum mismatch")
                        else:
                            report = run_experiment(
                                target,
                                seed=world_seed,
                                sensor_seed=sensor_seed,
                                config=cfg,
                                log=None,
                            )
                    except (RuntimeError, ValueError) as exc:
                        if not any(
                            marker in str(exc)
                            for marker in ("patrol incomplete", "path needs at least two waypoints")
                        ):
                            raise
                        collector_failure = str(exc)
                        row.update(status="collector_invalid", reason=collector_failure)
                    else:
                        result = report["hypothesis"]["results"]
                        errors = result["errors"]
                        alignment = result["paired_scan_alignment"]
                        row.update(
                            status="ok",
                            est_mean_m=errors["est_mean_m"],
                            pf_true_mean_m=errors["pf_mean_m"],
                            pf_sensor_truth_mean_m=errors["pf_sensor_truth_mean_m"],
                            pf_self_mean_m=errors["pf_self_mean_m"],
                            est_p95_m=errors["est_p95_m"],
                            pf_self_p95_m=errors["pf_self_p95_m"],
                            map_alignment_precision=alignment["precision"],
                            map_alignment_recall=alignment["recall"],
                            map_median_offset_m=alignment["median_offset_m"],
                            trajectory_sha256=report["trajectories_sha256"],
                            result_dir=str(target),
                        )
                rows.append(row)
                log(f"{arm} world={world_seed} sensor={sensor_seed}: {row['status']}")
    source = Path(__file__)
    report = {
        "experiment": "navigation_map_localization_benchmark",
        "schema_version": 2,
        "python": platform.python_version(),
        "plan": {
            "arms": plan.arms,
            "world_seeds": plan.world_seeds,
            "sensor_seeds": plan.sensor_seeds,
            "particles": plan.particles,
            "mapping_control": "truth-controlled replay; not autonomous mapping or navigation",
            "pairing": "same geometry and physical patrol per world; sensor seeds change noisy localization observations and PF randomness",
            "map_arms": "ideal occupancy, same-scan truth-pose projection, same-scan odometry-pose projection",
            "gate": "both arms >=5 valid worlds; >=4 worlds/arm with all sensor seeds improving selfbuilt mean and p95 vs odometry and same-scan map precision/recall >=0.70",
        },
        "rows": rows,
        "summary": summarize(rows, plan),
        "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="paired navigation localization replay benchmark")
    parser.add_argument("--output", default="results/navigation_benchmark_2026-09-24_full_v1")
    parser.add_argument("--arms", default="ISO,ANISO")
    parser.add_argument("--world-seeds", default="0,1,2,3,4")
    parser.add_argument("--sensor-seeds", default="0,1,2")
    parser.add_argument("--particles", type=int, default=100)
    args = parser.parse_args()
    plan = BenchmarkPlan(
        arms=tuple(args.arms.split(",")),
        world_seeds=tuple(int(x) for x in args.world_seeds.split(",")),
        sensor_seeds=tuple(int(x) for x in args.sensor_seeds.split(",")),
        particles=args.particles,
    )
    report = run_benchmark(args.output, plan)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
