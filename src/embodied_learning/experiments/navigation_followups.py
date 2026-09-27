"""Registered, separate navigation experiments 69 (footprint), 70/71 (decisions)."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodied_learning.experiments import campus_patrol as campus
from embodied_learning.experiments import observed_navigation as observed
from embodied_learning.experiments.grid_nav import inflate
from embodied_learning.footprint_planning import FootprintController
from embodied_learning.navigation_decisions import NavigationDecision
from embodied_learning.navigation_geometry import NavigationRig
from embodied_learning.replay_viewer.campus import read_checked

METHODS = {
    "69": ("circle_grid", "circle_metric", "rectangle"),
    "70": ("baseline", "uncertainty", "joint"),
    "71": ("baseline", "recovery"),
}
DIRECTORIES = {
    "69": "footprint_navigation_v1",
    "70": "arrival_decisions_v2",
    "71": "bounded_recovery_v2",
}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")


def sampled_disc_clearance(record, geometry):
    best = float("inf")
    for pose, command in zip(record["truth"], record["commands"], strict=True):
        n = max(1, math.ceil(abs(command[0]) * campus.DT / 0.01))
        for t in np.linspace(0, campus.DT, n + 1):
            point = campus.advance(pose, command[0] * t, command[1] * t)
            best = min(best, min(o.distance(*point[:2]) for o in geometry) - campus.RADIUS)
    return best


def run(
    lesson,
    output=None,
    source="results/map_repair_v1",
    seeds=(0, 1, 2),
    layouts=("plaza", "street", "narrow"),
    obstacles=("crate", "low", "beam"),
):
    out = Path(output) if output else Path("results") / DIRECTORIES[lesson]
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[3]
    files = [
        Path(__file__),
        Path(__file__).parents[1] / "navigation_geometry.py",
        Path(__file__).parents[1] / "experiments/grid_nav.py",
    ]
    files += (
        [Path(observed.__file__), Path(__file__).parents[1] / "footprint_planning.py"]
        if lesson == "69"
        else [
            Path(campus.__file__),
            Path(__file__).parents[1] / "navigation_decisions.py",
            Path(__file__).parents[1] / "experiments/map_localization.py",
        ]
    )
    for file in files:
        target = out / "source" / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    protocol = {
        "lesson": lesson,
        "methods": METHODS[lesson],
        "seeds": list(seeds),
        "source_sha256": {str(f.relative_to(root)): digest(f) for f in files},
        "causal_access": "prior/frozen map, current/past scans and estimate; truth only scores",
        "comparison": "paired noise arrays and PF seed; trajectories can diverge after decisions",
    }
    if lesson == "69":
        protocol.update(
            rig=NavigationRig().to_dict(),
            dt_s=observed.DT,
            layouts=list(layouts),
            obstacles=list(obstacles),
            localization="exact wheel odometry; controlled condition",
            planner="smooth lane seeds (4 m ramp, 5 cm lateral increments); hybrid A* fallback, 20 cm curved primitives, 32 absolute headings, 2.5 cm state merge, 10000 expansion cap; output at <=4 cm (256 slots)",
            tracking="unchanged lesson67 tracker and swept stop; circle_metric and rectangle share inputs/graph",
            input_representation="prior cells keep extent; observed metric points retained; 6 cm margin unchanged",
        )
        cases = [
            (layout, obstacle, seed)
            for layout in layouts
            for obstacle in obstacles
            for seed in seeds
        ]
    else:
        source = Path(source)
        summary = json.loads((source / "summary.json").read_text(encoding="utf8"))
        maps = read_checked(source / "maps.npz", summary["maps_sha256"])
        np.savez_compressed(out / "maps.npz", **maps)
        geometry, styles, _ = campus.world()
        protocol.update(
            input_maps_sha256=digest(source / "maps.npz"),
            input_summary_sha256=digest(source / "summary.json"),
            config=asdict(campus.Config()),
            dt_s=campus.DT,
            maps=["repaired", "reference"],
            system_margin_m=0.10,
            true_arrival_m=0.45,
            maximum_recoveries=3,
            recovery_frames=40,
            recovery_speed_m_s=0.15,
            recovery_horizon_s=3.0,
            recovery_max_r95_m=0.25,
            frozen_map=True,
        )
        cases = [(grid, seed) for grid in ("repaired", "reference") for seed in seeds]
    save_json(out / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    if lesson != "69":
        summary.update(
            maps_sha256=digest(out / "maps.npz"),
            geometry=[asdict(o) for o in geometry],
            styles=styles,
        )
    started = time.perf_counter()
    for case in cases:
        for method in METHODS[lesson]:
            tick = time.perf_counter()
            if lesson == "69":
                layout, obstacle, seed = case
                controllers = []

                def factory(prior, rig, sensor_method, method=method, controllers=controllers):
                    c = (
                        observed.ObservedController(prior, rig, sensor_method)
                        if method == "circle_grid"
                        else FootprintController(
                            prior,
                            rig,
                            sensor_method,
                            "circle" if method == "circle_metric" else "rectangle",
                        )
                    )
                    controllers.append(c)
                    return c

                row, data, grid = observed.episode(
                    layout, obstacle, "depth", seed, controller_factory=factory
                )
                before, actual = observed.scene(layout, obstacle)
                data.update(
                    observed_map=grid, prior_map=observed.occupancy(before), actual_boxes=actual
                )
                logs = getattr(controllers[0], "planning_log", [])
                row.update(
                    planning_calls=len(logs),
                    planning_reasons=dict(Counter(l["reason"] for l in logs)),
                    planning_wall_s=sum(l["wall_s"] for l in logs),
                    planning_over_200ms=sum(l["wall_s"] > 0.2 for l in logs),
                    planning_log=logs,
                )
                name = f"{layout}_{obstacle}_s{seed}_{method}.npz"
            else:
                kind, seed = case
                grid = maps[kind]
                policy = NavigationDecision(method, grid)
                row, data = campus.run_episode(
                    "self", seed, geometry, ~inflate(grid, 4), grid, campus.Config(), policy
                )
                row.update(map_kind=kind, min_clearance_m=sampled_disc_clearance(data, geometry))
                name = f"{kind}_s{seed}_{method}.npz"
            row["method"] = method
            np.savez_compressed(out / name, **data)
            row.update(file=name, sha256=digest(out / name), wall_s=time.perf_counter() - tick)
            summary["rows"].append(row)
            summary["wall_s"] = time.perf_counter() - started
            save_json(out / "summary.json", summary)
            print(
                lesson,
                *case,
                method,
                row["status"],
                f"{row['reached']}/{row['target_count']}",
                f"{row['wall_s']:.2f}s",
                flush=True,
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson", choices=tuple(METHODS), required=True)
    parser.add_argument("--output")
    parser.add_argument("--source", default="results/map_repair_v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--layouts", nargs="+", default=["plaza", "street", "narrow"])
    parser.add_argument("--obstacles", nargs="+", default=["crate", "low", "beam"])
    a = parser.parse_args()
    run(a.lesson, a.output, a.source, a.seeds, a.layouts, a.obstacles)


if __name__ == "__main__":
    main()
