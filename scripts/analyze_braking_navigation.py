"""Hash-checked aggregation and observation-vs-prior failure diagnosis."""

import json
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import MODES, BrakingController
from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_geometry import NavigationRig, body_points
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/braking_navigation_v3"


def percentiles(values):
    return (
        dict(
            zip(
                ("p95_s", "p99_s", "max_s"),
                map(float, (*np.percentile(values, [95, 99]), max(values))),
                strict=True,
            )
        )
        if len(values)
        else None
    )


def main():
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 230
    assert digest(DATA / "protocol.json") == summary["protocol_sha256"]
    rows = summary["rows"]
    archives = {r["file"]: read_checked(DATA / r["file"], r["sha256"]) for r in rows}
    aggregates = {}
    for cohort in ("fixed", "heldout_noise", "heldout_layout", "negative"):
        aggregates[cohort] = {}
        for mode in MODES:
            rr = [r for r in rows if r["cohort"] == cohort and r["method"] == mode]
            controls = np.concatenate([archives[r["file"]]["control_wall_s"] for r in rr])
            plans = [p["wall_s"] for r in rr for p in r["planning_log"]]
            first = [r["planning_log"][0]["wall_s"] for r in rr if r["planning_log"]]
            later = [p["wall_s"] for r in rr for p in r["planning_log"][1:]]
            aggregates[cohort][mode] = {
                "total": len(rr),
                "reached": sum(r["reached"] for r in rr),
                "safe_completed": sum(r["safe_completed"] for r in rr),
                "contacts": sum(r["contacts"] for r in rr),
                "expanded_violation_episodes": sum(r["expanded_swept_min_m"] <= 0 for r in rr),
                "expanded_violation_frames": sum(r["expanded_overlap_frames"] for r in rr),
                "body_min_m": min(r["min_clearance_m"] for r in rr),
                "expanded_min_m": min(r["expanded_swept_min_m"] for r in rr),
                "arrival_times_s": [r["elapsed_sim_s"] for r in rr if r["reached"]],
                "controller": percentiles(controls),
                "planning": percentiles(plans),
                "first_planning": percentiles(first),
                "later_planning": percentiles(later),
                "control_calls": len(controls),
                "control_over_100ms": int((controls > 0.1).sum()),
            }
    failures = []
    rig = NavigationRig()
    # All positive-case failures: use only observations up to the trigger frame.
    for r in rows:
        if r["status"] == "completed" or r["cohort"] == "negative":
            continue
        d = archives[r["file"]]
        f = r["trigger_frame"]
        c = BrakingController(d["prior_map"], rig, "discrete")
        for j in range(f + 1):
            c.observe(
                d["estimate"][j], d["ranges"][j], d["hits"][j], d["depth"][j], d["depth_valid"][j]
            )
        pose = d["estimate"][f]
        points = np.array(list(c.voxels.values())).reshape(-1, 3)
        prior_free = FootprintField(d["prior_map"], np.empty((0, 2)), rig).free(pose)
        combined_free = FootprintField(d["prior_map"], points[:, :2], rig).free(pose)
        local = body_points(points, pose)
        inside = (
            (local[:, 0] >= -rig.rear_m - rig.margin_m)
            & (local[:, 0] <= rig.front_m + rig.margin_m)
            & (np.abs(local[:, 1]) <= rig.half_width_m + rig.margin_m)
        )
        failures.append(
            {
                "file": r["file"],
                "trigger_frame": f,
                "reason": r["planning_log"][-1]["reason"],
                "prior_alone_free": bool(prior_free),
                "prior_and_observed_free": bool(combined_free),
                "observed_points_inside_padded_footprint": int(inside.sum()),
                "intruding_observed_points_world_sample": points[inside][:10].tolist(),
                "actual_padded_gap_at_trigger_m": float(d["expanded_gap_m"][f]),
                "incoming_velocity": d["incoming_velocity"][f].tolist(),
                "tail_distance_m": r["tail_distance_m"],
                "tail_time_s": r["tail_time_s"],
            }
        )
    counter = ROOT / "results/braking_counterfactual_v3/summary.json"
    output = {
        "summary_sha256": digest(DATA / "summary.json"),
        "protocol": summary["protocol"],
        "protocol_sha256": summary["protocol_sha256"],
        "wall_s": summary["wall_s"],
        "aggregates": aggregates,
        "failure_diagnosis": failures,
        "counterfactual": json.loads(counter.read_text(encoding="utf8")),
        "counterfactual_summary_sha256": digest(counter),
        "rows": [{k: v for k, v in r.items() if k != "planning_log"} for r in rows],
    }
    save_json(ROOT / "docs/benchmarks/braking-navigation-v3.json", output)
    for cohort, values in aggregates.items():
        print(
            cohort,
            {
                k: (v["reached"], v["safe_completed"], v["total"], v["expanded_violation_episodes"])
                for k, v in values.items()
            },
        )
    print("failure_count", len(failures), "wall", summary["wall_s"])


if __name__ == "__main__":
    main()
