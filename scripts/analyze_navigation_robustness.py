"""Hash-checked single-factor scan, failure reasons, and causal delay evidence."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import BrakingController
from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_geometry import NavigationRig
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/navigation_robustness_v4"


def aggregate(rows, controls):
    timing = np.concatenate([controls[r["file"]] for r in rows])
    first_plans = [r["planning_log"][0]["wall_s"] for r in rows if r["planning_log"]]
    later_plans = [p["wall_s"] for r in rows for p in r["planning_log"][1:]]
    return {
        "total": len(rows),
        "reached": sum(r["reached"] for r in rows),
        "announced": sum(r["announced"] for r in rows),
        "false_arrivals": sum(r["false_arrival"] for r in rows),
        "safe_completed": sum(r["safe_completed"] for r in rows),
        "contacts": sum(r["contacts"] for r in rows),
        "physically_stopped": sum(r["physically_stopped"] for r in rows),
        "expanded_violation_episodes": sum(r["expanded_swept_min_m"] <= 0 for r in rows),
        "body_min_m": min(r["min_clearance_m"] for r in rows),
        "expanded_min_m": min(r["expanded_swept_min_m"] for r in rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
        "controller_p95_ms": float(np.percentile(timing, 95) * 1000),
        "controller_max_ms": float(timing.max() * 1000),
        "controller_calls": len(timing),
        "controller_over_100ms": int((timing > 0.1).sum()),
        "first_planning_max_s": max(first_plans, default=None),
        "later_planning_max_s": max(later_plans, default=None),
        "later_planning_calls": len(later_plans),
        "planning_reasons": dict(Counter(p["reason"] for r in rows for p in r["planning_log"])),
    }


def main():
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 233
    assert digest(DATA / "protocol.json") == summary["protocol_sha256"]
    controls, failures, rows = {}, [], []
    rig = NavigationRig()
    for row in summary["rows"]:
        d = read_checked(DATA / row["file"], row["sha256"])
        controls[row["file"]] = d["control_wall_s"]
        r = dict(row)
        requests = np.flatnonzero(d["reason"] == "swept_body_brake")
        first = int(requests[0]) if len(requests) else None
        execution = (
            np.flatnonzero(d["executed_request_frame"] == first) if first is not None else []
        )
        applied = int(execution[0]) if len(execution) else None
        r.update(
            first_protective_brake_frame=first,
            first_protective_brake_applied_frame=applied,
            delay_distance_after_first_brake_m=float(
                np.linalg.norm(np.diff(d["truth"][first : applied + 1, :2], axis=0), axis=1).sum()
            )
            if first is not None and applied is not None
            else None,
            final_estimated_pose=d["estimate"][-1].tolist(),
            final_true_pose=d["truth"][-1].tolist(),
        )
        rows.append(r)
        if row["status"] != "no_path" or row["cohort"] == "negative":
            continue
        f = row["trigger_frame"]
        c = BrakingController(d["prior_map"], rig, "coupled")
        for j in range(f + 1):
            c.observe(
                d["estimate"][j], d["ranges"][j], d["hits"][j], d["depth"][j], d["depth_valid"][j]
            )
        points = np.array(list(c.voxels.values())).reshape(-1, 3)
        failures.append(
            {
                "file": row["file"],
                "method": row["method"],
                "cohort": row["cohort"],
                "trigger_frame": f,
                "planner_reason": row["planning_log"][-1]["reason"],
                "last_search_expansions": row["planning_log"][-1].get("expanded", 0),
                "prior_estimated_start_free": bool(
                    FootprintField(d["prior_map"], np.empty((0, 2)), rig).free(d["estimate"][f])
                ),
                "observed_estimated_start_free": bool(
                    FootprintField(d["prior_map"], points[:, :2], rig).free(d["estimate"][f])
                ),
                "true_padded_gap_at_trigger_m": float(d["expanded_gap_m"][f]),
                "last_plan_wall_s": row["planning_log"][-1]["wall_s"],
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["cohort"], row["layout"], row["method"]].append(row)
    aggregates = {}
    for (cohort, layout, key), subset in grouped.items():
        aggregates.setdefault(cohort, {}).setdefault(layout, {})[key] = aggregate(subset, controls)
    result = {
        "summary_sha256": digest(DATA / "summary.json"),
        "protocol": summary["protocol"],
        "protocol_sha256": summary["protocol_sha256"],
        "wall_s": summary["wall_s"],
        "aggregates": aggregates,
        "no_path_diagnosis": failures,
        "rows": [{k: v for k, v in r.items() if k != "planning_log"} for r in rows],
        "analysis_source_sha256": digest(Path(__file__)),
    }
    save_json(ROOT / "docs/benchmarks/navigation-robustness-v4.json", result)
    for key, v in aggregates["scan"]["narrow"].items():
        print(
            key,
            v["reached"],
            v["safe_completed"],
            v["contacts"],
            round(v["expanded_min_m"] * 1000, 2),
            v["statuses"],
        )
    print("failure_reasons", dict(Counter(f["planner_reason"] for f in failures)))
    print("wall_s", summary["wall_s"])


if __name__ == "__main__":
    main()
