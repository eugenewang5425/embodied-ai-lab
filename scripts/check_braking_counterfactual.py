"""Freeze identical archived inputs, then compare two complete braking motions."""

import json
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import BrakingController, braking_path
from embodied_learning.experiments.footprint_tracking import digest, expanded_separation, save_json
from embodied_learning.navigation_geometry import (
    NavigationRig,
    collision_clearance,
    point_clearance,
    rotation,
)
from embodied_learning.replay_viewer.campus import read_checked


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "results/footprint_tracking_v2"
    output = root / "results/braking_counterfactual_v3"
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / "summary.json").read_text(encoding="utf8"))
    rig, results = NavigationRig(), []
    saved = {}
    for seed, frame in ((0, 58), (2, 57)):
        row = next(
            r
            for r in summary["rows"]
            if r["method"] == "combined"
            and r["layout"] == "narrow"
            and r["obstacle"] == "crate"
            and r["seed"] == seed
        )
        d = read_checked(source / row["file"], row["sha256"])
        controller = BrakingController(d["prior_map"], rig, "discrete")
        for i in range(frame + 1):
            points = controller.observe(
                d["estimate"][i], d["ranges"][i], d["hits"][i], d["depth"][i], d["depth_valid"][i]
            )
        pose, velocity = d["truth"][frame], d["commands"][frame - 1]
        assert d["reason"][frame] == "swept_body_brake"
        for delay in (0, 0.2):
            for coupled in (False, True):
                path, commands = braking_path(velocity, rig, coupled, delay)
                world = path.copy()
                world[:, :2] = path[:, :2] @ rotation(pose[2]).T + pose[:2]
                world[:, 2] += pose[2]
                label = f"s{seed}_delay{delay}_{'coupled' if coupled else 'independent'}"
                saved[label + "_path"], saved[label + "_commands"] = world, commands
                results.append(
                    {
                        "seed": seed,
                        "frame": frame,
                        "delay_s": delay,
                        "coupled": coupled,
                        "input_sha256": row["sha256"],
                        "input_pose": pose.tolist(),
                        "input_velocity": velocity.tolist(),
                        "first_executed": commands[0].tolist(),
                        "duration_until_rest_s": (len(commands) + 1) * 0.1,
                        "endpoint": world[-1].tolist(),
                        "observed_min_gap_m": point_clearance(points, path, rig, rig.margin_m),
                        "true_body_min_gap_m": min(
                            collision_clearance(p, d["actual_boxes"], rig) for p in world
                        ),
                        "true_expanded_min_gap_m": min(
                            expanded_separation(p, d["actual_boxes"], rig) for p in world
                        ),
                    }
                )
    np.savez_compressed(output / "traces.npz", **saved)
    files = [Path(__file__), root / "src/embodied_learning/braking_control.py"]
    for file in files:
        target = output / "source" / file.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    result = {
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in files},
        "traces_sha256": digest(output / "traces.npz"),
        "rows": results,
    }
    save_json(output / "summary.json", result)
    for r in results:
        print(
            r["seed"],
            r["delay_s"],
            r["coupled"],
            "expanded",
            round(r["true_expanded_min_gap_m"] * 1000, 3),
            "mm",
        )


if __name__ == "__main__":
    main()
