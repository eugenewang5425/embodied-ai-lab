"""Use identical archived observations and pending intentions to isolate prediction."""

import json
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import BrakingController, braking_path, reachable
from embodied_learning.execution_safety import queued_stop_path
from embodied_learning.experiments.footprint_tracking import save_json
from embodied_learning.navigation_geometry import NavigationRig, point_clearance
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/navigation_robustness_v4"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    row = next(
        r
        for r in summary["rows"]
        if r["layout"] == "narrow"
        and r["obstacle"] == "crate"
        and r["seed"] == 8
        and r["method"] == "delay_4"
    )
    a = read_checked(directory / row["file"], row["sha256"])
    rig = NavigationRig()
    controller = BrakingController(a["prior_map"], rig, "coupled")
    rows = []
    for k in range(len(a["truth"]) - 1):
        points = controller.observe(
            a["estimate"][k], a["ranges"][k], a["hits"][k], a["depth"][k], a["depth_valid"][k]
        )
        if a["reason"][k] != "following":
            continue
        pending = [
            (a["requested_command"][j], bool(a["braking_active"][j]), j)
            if j >= 0
            else (np.zeros(2), True, -1)
            for j in range(k - 4, k)
        ]
        candidate = reachable(a["incoming_velocity"][k], a["requested_command"][k], rig)
        old = np.vstack([braking_path(candidate, rig, True, latency)[0] for latency in (0, 0.2)])
        new, commands = queued_stop_path(
            a["incoming_velocity"][k], pending, a["requested_command"][k], False, rig
        )
        stopping, _ = queued_stop_path(a["incoming_velocity"][k], pending, [0, 0], True, rig)
        rows.append(
            {
                "frame": k,
                "time_s": k * 0.1,
                "old_gap_m": point_clearance(points, old, rig, rig.margin_m),
                "queue_gap_m": point_clearance(points, new, rig, rig.margin_m),
                "queue_stop_gap_m": point_clearance(points, stopping, rig, rig.margin_m),
                "predicted_distance_m": float(
                    np.linalg.norm(np.diff(new[:, :2], axis=0), axis=1).sum()
                ),
                "pending_targets": [np.asarray(t).tolist() for t, _, _ in pending],
                "first_execution": commands[0].tolist(),
            }
        )
    disagreement = next(
        (
            r
            for r in rows
            if r["old_gap_m"] > 0 and min(r["queue_gap_m"], r["queue_stop_gap_m"]) <= 0
        ),
        None,
    )
    result = {
        "source_npz_sha256": row["sha256"],
        "selection": "fixed prior narrow/crate/s8 delay0.4s; first old-accept/new-reject frame",
        "first_disagreement": disagreement,
        "rows": rows,
        "meaning": "same inputs only; forecast rejection is not proof a new closed-loop policy succeeds",
    }
    save_json(ROOT / "docs/benchmarks/execution-counterfactual-v5.json", result)
    print(json.dumps(disagreement, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
