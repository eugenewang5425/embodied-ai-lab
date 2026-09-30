"""Lesson 69 round four: frozen controller, pose-error and execution-delay boundaries."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import DT, BrakingController
from embodied_learning.experiments import observed_navigation as observed
from embodied_learning.experiments.braking_navigation import scene
from embodied_learning.experiments.footprint_tracking import diagnostics, digest, save_json
from embodied_learning.navigation_geometry import (
    NavigationRig,
    advance,
    collision_clearance,
    sense,
    swept_clearance,
)
from embodied_learning.navigation_robustness import CONDITIONS, GROUPS, IntentQueue, biased_pose
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = "results/navigation_robustness_v4"


def episode(layout, obstacle, seed, condition, max_steps=600):
    rig = NavigationRig()
    prior, boxes = scene(layout, obstacle)
    c = BrakingController(observed.occupancy(prior), rig, "coupled")
    transport = IntentQueue(condition.delay_steps, rig)
    rng = np.random.default_rng([6700, seed])
    pose, velocity, goal = np.array([1.5, 6.0, 0.0]), np.zeros(2), np.array([10.5, 6.0])
    names = (
        "truth",
        "estimate",
        "ranges",
        "hits",
        "depth",
        "depth_valid",
        "goal",
        "mission",
        "plan",
        "commands",
        "reason",
        "safety_clearance",
        "phase",
        "incoming_velocity",
        "braking_active",
        "speed_cap",
        "preview_curvature",
        "nominal_command",
        "stop_clearance",
        "control_wall_s",
        "tracking_angular_raw",
        "requested_command",
        "executed_target",
        "executed_braking",
        "executed_request_frame",
        "pending_motion",
    )
    records = {name: [] for name in names}
    phase, failure, trigger, eligible = "running", None, None, None
    best, progress = 20.0, 0
    gaps, timings = [], []
    # Allow at most two seconds of actual braking beyond the nominal timeout.
    for k in range(max_steps + condition.delay_steps + 21):
        estimate = biased_pose(pose, condition)
        ranges, hits, depth, valid = sense(boxes, pose, rig, rng)
        started = time.perf_counter()
        if phase in ("running", "goal_braking"):
            before = len(c.tracking_log)
            desired, reason, clearance = c.step(
                estimate.copy(), goal, ranges, hits, depth, valid, k, velocity
            )
            angular = c.tracking_log[-1]["angular_raw"] if len(c.tracking_log) > before else np.nan
        else:
            desired, reason, clearance, angular = np.zeros(2), "abort_braking", np.nan, np.nan
            c.latest = {
                "speed_cap": np.nan,
                "preview_curvature": np.nan,
                "nominal": np.zeros(2),
                "stop_clearance": np.nan,
            }
        distance = float(np.linalg.norm(estimate[:2] - goal))
        if distance < best - 0.05:
            best, progress = distance, k
        if eligible is None and distance < 0.25 and abs(velocity[0]) < 0.05:
            eligible = k
        if phase == "running":
            if distance < 0.25:
                phase, trigger = "goal_braking", k
            elif reason == "no_observed_route":
                phase, failure, trigger = "abort_braking", "no_path", k
            elif k >= max_steps:
                phase, failure, trigger = "abort_braking", "timeout", k
            elif k - progress > 120:
                phase, failure, trigger = "abort_braking", "stalled", k
        stopping = phase != "running" or reason == "swept_body_brake"
        command, executed_target, executed_braking, source_frame = transport.step(
            desired, stopping, k, velocity
        )
        at_rest = np.max(np.abs(velocity)) < 1e-12
        terminal = phase != "running" and at_rest and not transport.pending_motion
        terminal = terminal and np.max(np.abs(command)) < 1e-12
        if terminal:
            command = np.zeros(2)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        plan = np.full((c.path_capacity, 2), np.nan)
        if c.path is not None:
            plan[: len(c.path)] = c.path
        values = (
            pose.copy(),
            estimate.copy(),
            ranges,
            hits,
            depth,
            valid,
            goal,
            0,
            plan,
            command.copy(),
            reason,
            clearance,
            phase,
            velocity.copy(),
            stopping,
            c.latest["speed_cap"],
            c.latest["preview_curvature"],
            c.latest["nominal"],
            c.latest["stop_clearance"],
            elapsed,
            angular,
            desired.copy(),
            executed_target.copy(),
            executed_braking,
            source_frame,
            transport.pending_motion,
        )
        for name, value in zip(names, values, strict=True):
            records[name].append(value)
        if terminal:
            break
        gap = swept_clearance(pose, *command, DT, boxes, rig)
        gaps.append(gap)
        pose = advance(pose, *command, DT)
        velocity = command
        if gap <= 0:
            # Preserve the contact endpoint. A zero marker is NOT proof of a stop.
            phase, failure = "contact_abort", "collision_abort"
            if trigger is None:
                trigger = k + 1
            ranges, hits, depth, valid = sense(boxes, pose, rig, rng)
            values = (
                pose.copy(),
                biased_pose(pose, condition),
                ranges,
                hits,
                depth,
                valid,
                goal,
                0,
                plan.copy(),
                np.zeros(2),
                "contact_abort",
                np.nan,
                phase,
                velocity.copy(),
                True,
                np.nan,
                np.nan,
                np.zeros(2),
                np.nan,
                0.0,
                np.nan,
                np.zeros(2),
                np.zeros(2),
                False,
                -1,
                transport.pending_motion,
            )
            for name, value in zip(names, values, strict=True):
                records[name].append(value)
            break
    else:
        raise RuntimeError("terminal deceleration exceeded physical bound")
    data = {name: np.asarray(values) for name, values in records.items()}
    final_distance = float(np.linalg.norm(data["truth"][-1, :2] - goal))
    stopped = bool(np.max(np.abs(data["incoming_velocity"][-1])) < 1e-12)
    estimated_distance = float(np.linalg.norm(data["estimate"][-1, :2] - goal))
    announced = phase == "goal_braking" and stopped and estimated_distance < 0.25
    completed = announced and final_distance < 0.25
    status = (
        "completed"
        if completed
        else "false_completion"
        if announced
        else failure or "goal_overshoot"
    )
    diagnostic, minimum = diagnostics(data, prior, boxes, rig)
    if minimum is None:
        minimum = float(diagnostic["expanded_gap_m"].min())
    data.update(
        diagnostic, actual_boxes=boxes, prior_map=observed.occupancy(prior), observed_map=c.grid
    )
    brake = data["reason"] == "swept_body_brake"
    trigger = len(data["truth"]) - 1 if trigger is None else trigger
    row = {
        "method": condition.key,
        "perturbation": condition.to_dict(),
        "layout": layout,
        "obstacle": obstacle,
        "seed": seed,
        "status": status,
        "reached": int(completed),
        "announced": int(announced),
        "false_arrival": bool(announced and not completed),
        "final_estimated_goal_distance_m": estimated_distance,
        "target_count": 1,
        "contacts": int(failure == "collision_abort"),
        "mean_m": float(abs(condition.dy_m)),
        "p95_m": float(abs(condition.dy_m)),
        "end_m": float(abs(condition.dy_m)),
        "abs_heading_error_deg": float(abs(condition.yaw_deg)),
        "frames": len(data["truth"]),
        "elapsed_sim_s": (len(data["truth"]) - 1) * DT,
        "events": [{"frame": len(data["truth"]) - 1, "success": completed}] if announced else [],
        "min_clearance_m": min(gaps, default=collision_clearance(pose, boxes, rig)),
        "expanded_swept_min_m": minimum,
        "expanded_overlap_frames": int(np.sum(data["expanded_gap_m"] <= 0)),
        "safe_completed": bool(completed and minimum is not None and minimum > 0),
        "braking_frames": int(brake.sum()),
        "braking_events": int(np.sum(brake & ~np.r_[False, brake[:-1]])),
        "trigger_frame": trigger,
        "arrival_eligible_frame": eligible,
        "physically_stopped": stopped,
        "terminal_incoming_velocity": data["incoming_velocity"][-1].tolist(),
        "tail_time_s": (len(data["truth"]) - 1 - trigger) * DT,
        "tail_distance_m": float(
            np.linalg.norm(np.diff(data["truth"][trigger:, :2], axis=0), axis=1).sum()
        ),
        "final_goal_distance_m": final_distance,
        "travelled_m": float(np.linalg.norm(np.diff(data["truth"][:, :2], axis=0), axis=1).sum()),
        "controller_p95_ms": float(np.percentile(timings, 95) * 1000),
        "planning_log": c.planning_log,
        "planning_reasons": dict(Counter(x["reason"] for x in c.planning_log)),
    }
    return row, data


def registered_cases(smoke=False):
    if smoke:
        return [
            ("narrow", "crate", 0, "development", key)
            for key in (
                "zero",
                "y_plus_1",
                "y_minus_1",
                "yaw_plus_1",
                "yaw_minus_1",
                "delay_2",
                "delay_4",
            )
        ]
    cases = [
        (layout, ob, seed, "bridge", "zero")
        for layout in ("plaza", "street", "narrow")
        for ob in ("crate", "low", "beam")
        for seed in (0, 1, 2)
    ]
    cases += [
        ("narrow", ob, seed, "scan", key)
        for ob in ("crate", "low", "beam")
        for seed in (8, 9, 10)
        for key in CONDITIONS
    ]
    cases += [
        (layout, "crate", 8, "environment", key)
        for layout in ("plaza", "street")
        for key in ("zero", "y_plus_4", "y_minus_4", "yaw_plus_4", "yaw_minus_4", "delay_4")
    ]
    cases += [
        ("blocked", "crate", 8, "negative", key)
        for key in ("zero", "y_plus_4", "y_minus_4", "delay_2", "delay_4")
    ]
    assert len(cases) == 233
    return cases


def check_bridge(row, data, source, previous):
    old = next(
        r
        for r in previous["rows"]
        if r["method"] == "coupled" and all(r[k] == row[k] for k in ("layout", "obstacle", "seed"))
    )
    arrays = read_checked(source / old["file"], old["sha256"])
    checked = []
    for key, values in arrays.items():
        if key == "control_wall_s":
            continue
        assert np.array_equal(values, data[key], equal_nan=values.dtype.kind not in "US"), key
        checked.append(key)
    for key in ("status", "reached", "contacts", "safe_completed", "frames", "trigger_frame"):
        assert old[key] == row[key], key
    row.update(bridge_equal=True, bridge_sha256=old["sha256"], bridge_arrays=checked)


def run(output=DEFAULT_OUTPUT, smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    previous_dir = ROOT / "results/braking_navigation_v3"
    previous = json.loads((previous_dir / "summary.json").read_text(encoding="utf8"))
    sources = [ROOT / name for name in previous["protocol"]["source_sha256"]]
    sources += [Path(__file__), ROOT / "src/embodied_learning/navigation_robustness.py"]
    for name, expected in previous["protocol"]["source_sha256"].items():
        assert digest(ROOT / name) == expected, f"frozen round-three source changed: {name}"
    for file in sources:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    cases = registered_cases(smoke)
    protocol = {
        "lesson": "69",
        "round": 4,
        "dt_s": DT,
        "smoke": smoke,
        "methods": list(CONDITIONS),
        "conditions": {k: c.to_dict() for k, c in CONDITIONS.items()},
        "view_groups": GROUPS,
        "cases": cases,
        "rig": NavigationRig().to_dict(),
        "controller": "frozen round-three coupled; no curvature speed cap",
        "pose_error": "estimate y=true y+dy; yaw=wrap(true yaw+bias); no map or goal transform",
        "delay": "FIFO intentions including all brakes; issue k -> execution k+n, then physical acceleration limit; fresh sensors and current executed velocity",
        "arrival": "estimated goal circle triggers stop; declare after rest and no queued motion; true final circle independently scores success",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "baseline_summary_sha256": digest(previous_dir / "summary.json"),
        "registration_sha256": digest(ROOT / "docs/69-robustness-round4-plan.md"),
    }
    save_json(out / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    started = time.perf_counter()
    for layout, obstacle, seed, cohort, key in cases:
        tick = time.perf_counter()
        row, data = episode(layout, obstacle, seed, CONDITIONS[key])
        if cohort == "bridge" or (smoke and key == "zero"):
            check_bridge(row, data, previous_dir, previous)
        file = f"{layout}_{obstacle}_s{seed}_{key}.npz"
        np.savez_compressed(out / file, **data)
        row.update(
            file=file, sha256=digest(out / file), cohort=cohort, wall_s=time.perf_counter() - tick
        )
        summary["rows"].append(row)
        summary["wall_s"] = time.perf_counter() - started
        save_json(out / "summary.json", summary)
        print(
            f"{len(summary['rows'])}/{len(cases)} {cohort} {layout}/{obstacle}/s{seed} {key}: {row['status']} safe={row['safe_completed']} gap={row['expanded_swept_min_m']:.5f} {row['wall_s']:.1f}s",
            flush=True,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke)


if __name__ == "__main__":
    main()
