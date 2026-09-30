"""Four-arm outcomes separated by delay, environment and fault cohort."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from validate_footprint_tracking import read

from embodied_learning.braking_control import BrakingController
from embodied_learning.execution_safety import command_stop_path
from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.navigation_geometry import NavigationRig, point_clearance
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/execution_safety_v5"


def aggregate(rows):
    return {
        "total": len(rows),
        "safe_completed": sum(r["safe_completed"] for r in rows),
        "reached": sum(r["reached"] for r in rows),
        "contacts": sum(r["contacts"] for r in rows),
        "false_arrivals": sum(r["false_arrival"] for r in rows),
        "stopped": sum(r["physically_stopped"] for r in rows),
        "margin_violations": sum(r["expanded_swept_min_m"] <= 0 for r in rows),
        "expanded_min_m": min(r["expanded_swept_min_m"] for r in rows),
        "body_min_m": min(r["min_clearance_m"] for r in rows),
        "local_interventions": sum(r["local_override_frames"] > 0 for r in rows),
        "fault_encountered": sum(r["blackout_encountered"] for r in rows),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }


def main():
    summary = read(DATA / "summary.json")
    assert len(summary["rows"]) == 135 and not summary["protocol"]["smoke"]
    results = []
    for r in summary["rows"]:
        d = read_checked(DATA / r["file"], r["sha256"])
        override = np.flatnonzero(d["local_override"])
        expired = np.flatnonzero(d["expiry_active"])
        requested = np.flatnonzero(d["reason"] == "queue_body_brake")
        result = dict(r)
        result.update(
            first_local_frame=int(override[0]) if len(override) else None,
            first_expiry_frame=int(expired[0]) if len(expired) else None,
            first_queue_brake_frame=int(requested[0]) if len(requested) else None,
            control_max_ms=float(d["control_wall_s"].max() * 1000),
        )
        results.append(result)
    groups = []
    for cohort in ("scan", "environment", "blackout", "negative"):
        for delay in ("zero", "delay_2", "delay_4"):
            for method in summary["protocol"]["methods"]:
                rows = [
                    r
                    for r in results
                    if r["cohort"] == cohort
                    and r["condition_key"] == delay
                    and r["method"] == method
                ]
                if rows:
                    groups.append(
                        {"cohort": cohort, "delay": delay, "method": method, **aggregate(rows)}
                    )
    benchmark = {
        "summary_sha256": digest(DATA / "summary.json"),
        "protocol": summary["protocol"],
        "protocol_sha256": summary["protocol_sha256"],
        "wall_s": summary["wall_s"],
        "groups": groups,
        "rows": results,
        "limitations": [
            "three seeds repeated across heights, not nine independent seeds",
            "expiry stops conservatively without recovery",
            "fixed blackout simulation, not measured asynchronous wall-clock deployment",
            "point-based local safety can miss or falsely place obstacles",
            "finite swept sampling, not continuous safety proof",
        ],
    }
    # Diagnostic only: never feed this truth-based selection back into control.
    worst = min(
        (r for r in results if r["cohort"] == "scan" and r["method"] == "local_guard"),
        key=lambda r: r["expanded_swept_min_m"],
    )
    d = read_checked(DATA / worst["file"], worst["sha256"])
    first = int(np.flatnonzero(d["expanded_gap_m"] <= 0)[0])
    rig = NavigationRig()
    c = BrakingController(d["prior_map"], rig, "coupled")
    for k in range(first + 1):
        points = c.observe(
            d["estimate"][k], d["ranges"][k], d["hits"][k], d["depth"][k], d["depth_valid"][k]
        )
    full = SimpleNamespace(
        parts=np.array(
            [[-rig.rear_m, rig.front_m, -rig.half_width_m, rig.half_width_m, 0, rig.top_m]]
        )
    )
    path = command_stop_path(d["commands"][first], rig)
    benchmark["geometry_diagnostic"] = {
        "selection": "scan/local_guard worst full-height expanded separation; first invalid recorded pose",
        "file": worst["file"],
        "sha256": worst["sha256"],
        "frame": first,
        "obstacle": worst["obstacle"],
        "seed": worst["seed"],
        "stored_component_candidate_gap_m": float(d["local_candidate_gap"][first]),
        "same_points_full_height_candidate_gap_m": point_clearance(
            points, path, full, rig.margin_m
        ),
        "true_current_full_height_gap_m": float(d["expanded_gap_m"][first]),
        "interpretation": "component-based local guard and conservative full-height footprint score use different volumes; no sensor or controller was changed after formal batch",
    }
    save_json(ROOT / "docs/benchmarks/execution-safety-v5.json", benchmark)
    for g in groups:
        print(
            f"{g['cohort']} {g['delay']} {g['method']}: safe {g['safe_completed']}/{g['total']}, contacts {g['contacts']}, margin {g['expanded_min_m'] * 1000:.2f}mm, {g['statuses']}"
        )
    print(f"135 episodes; wall {summary['wall_s']:.2f}s")


if __name__ == "__main__":
    main()
