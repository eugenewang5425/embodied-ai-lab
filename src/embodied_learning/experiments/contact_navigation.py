"""Lesson 69 round six: arrival with physically bounded light contact."""

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import DT, BrakingController
from embodied_learning.contact_motion import LIMITS, PHYSICS_DT, ContactPlant, contact_allowed
from embodied_learning.execution_safety import GuardedTransport, queued_stop_path
from embodied_learning.experiments import observed_navigation as observed
from embodied_learning.experiments.braking_navigation import scene
from embodied_learning.experiments.footprint_tracking import (
    diagnostics,
    digest,
    expanded_separation,
    save_json,
)
from embodied_learning.navigation_geometry import (
    NavigationRig,
    body_points,
    collision_clearance,
    point_clearance,
    sense,
)
from embodied_learning.navigation_robustness import CONDITIONS, biased_pose

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = "results/contact_navigation_v6"
METHODS = {"margin_strict": (0.06, False), "zero_strict": (0.0, False), "zero_light": (0.0, True)}


def episode(layout, obstacle, seed, condition, method, profile="normal", max_steps=600):
    margin, allow_contact = METHODS[method]
    rig = replace(NavigationRig(), margin_m=margin)
    prior, boxes = scene(layout, obstacle)
    if layout == "blocked":
        prior[-2, 3], prior[-1, 2] = 5.85, 6.15
        prior[-2:, 0], prior[-2:, 1] = 3.0, 9.0
        boxes[:-1] = prior
    c = BrakingController(observed.occupancy(prior), rig, "coupled")
    forecast, guarded = True, True
    transport = GuardedTransport(condition.delay_steps, rig, guarded)
    rng = np.random.default_rng([6700, seed])
    pose, velocity, goal = np.array([1.5, 6.0, 0.0]), np.zeros(2), np.array([10.5, 6.0])
    plant = ContactPlant(boxes, rig, pose)
    substeps, applied, contact_steps = [], [], []
    last_contact = np.zeros(5)
    names = (
        "world_velocity",
        "contact_interval",
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
        "control_fresh",
        "local_override",
        "expiry_active",
        "control_age_ticks",
        "local_candidate_gap",
        "queue_candidate_gap",
        "queue_stop_gap",
    )
    records = {name: [] for name in names}
    phase, failure, trigger, eligible = "running", None, None, None
    best, progress = 20.0, 0
    timings = []
    # Allow at most two seconds of actual braking beyond the nominal timeout.
    for k in range(max_steps + condition.delay_steps + 21):
        estimate = biased_pose(pose, condition)
        ranges, hits, depth, valid = sense(boxes, pose, rig, rng)
        started = time.perf_counter()
        fresh = not (profile == "blackout" and 50 <= k < 60)
        if fresh and phase in ("running", "goal_braking"):
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
        if not fresh:
            c.observe(estimate, ranges, hits, depth, valid)
            reason = "planner_no_output"
        # Shared physical-plant adaptation: approach gently using estimated
        # distance. No true geometry/pose is consulted by the controller.
        approach_cap = min(0.6, 0.8 * float(np.linalg.norm(estimate[:2] - goal)))
        desired[0] = np.clip(desired[0], -approach_cap, approach_cap)
        points = body_points(np.array(list(c.voxels.values())).reshape(-1, 3), estimate)
        points = points[np.linalg.norm(points[:, :2], axis=1) < 2.5]
        queue_gap = queue_stop = np.nan
        if forecast and fresh and reason == "following":
            candidate, _ = queued_stop_path(velocity, tuple(transport.queue), desired, False, rig)
            stopping_path, _ = queued_stop_path(
                velocity, tuple(transport.queue), np.zeros(2), True, rig
            )
            queue_gap = point_clearance(points, candidate, rig, rig.margin_m)
            queue_stop = point_clearance(points, stopping_path, rig, rig.margin_m)
            if min(queue_gap, queue_stop) <= 0:
                desired, reason, clearance = (
                    np.zeros(2),
                    "queue_body_brake",
                    min(queue_gap, queue_stop),
                )
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
        stopping = phase != "running" or reason in ("swept_body_brake", "queue_body_brake")
        (
            command,
            executed_target,
            executed_braking,
            source_frame,
            override,
            expiry,
            local_gap,
            age,
        ) = transport.execute(desired, stopping, k, velocity, points, fresh)
        if transport.latched and phase == "running":
            phase, failure, trigger = "abort_braking", "control_expired", k
        at_rest = plant.stopped
        terminal = phase != "running" and at_rest and not transport.pending_motion
        terminal = terminal and np.max(np.abs(command)) < 0.01
        if terminal:
            command = np.zeros(2)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        plan = np.full((c.path_capacity, 2), np.nan)
        if c.path is not None:
            plan[: len(c.path)] = c.path
        values = (
            plant.data.qvel.copy(),
            last_contact.copy(),
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
            fresh,
            override,
            expiry,
            age,
            local_gap,
            queue_gap,
            queue_stop,
        )
        for name, value in zip(names, values, strict=True):
            records[name].append(value)
        if terminal:
            break
        state, force, contacts = plant.step(command)
        substeps.append(state)
        applied.append(force)
        contact_steps.append(contacts)
        last_contact = np.array(
            [
                contacts[:, 0].max(),
                contacts[:, 1].max(),
                contacts[:, 2].sum() * PHYSICS_DT,
                contacts[:, 3].max(),
                contacts[:, 4].max(),
            ]
        )
        pose, velocity = plant.pose, plant.velocity
        touched = bool((contacts[:, 0] > 0).any())
        severe = not contact_allowed(
            contacts[:, 1].max(), contacts[:, 3].max(), plant.max_continuous_s
        )
        if (touched and not allow_contact) or severe:
            phase = "contact_abort"
            failure = "heavy_contact_abort" if severe else "collision_abort"
            if trigger is None:
                trigger = k + 1
    else:
        raise RuntimeError("terminal deceleration exceeded physical bound")
    data = {name: np.asarray(values) for name, values in records.items()}
    final_distance = float(np.linalg.norm(data["truth"][-1, :2] - goal))
    stopped = bool(plant.stopped)
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
    diagnostic, _ = diagnostics(data, prior, boxes, rig)
    states = np.asarray(substeps).reshape(-1, 50, 6)
    forces = np.asarray(applied).reshape(-1, 50, 3)
    contacts = np.asarray(contact_steps).reshape(-1, 50, 5)
    full = contacts.reshape(-1, 5)
    samples = np.vstack((data["truth"][0], states[:, :, :3].reshape(-1, 3)))
    minimum = min(expanded_separation(p, boxes, rig) for p in samples)
    six_cm_min = min(expanded_separation(p, boxes, NavigationRig()) for p in samples)
    physical_min = min(collision_clearance(p, boxes, rig) for p in samples)
    active = full[:, 0] > 0
    contact_events = int(np.sum(active & ~np.r_[False, active[:-1]]))
    mild = contact_allowed(
        full[:, 1].max(initial=0), full[:, 3].max(initial=0), full[:, 4].max(initial=0)
    )
    passed = bool(completed and mild and (allow_contact or not active.any()))
    data.update(
        diagnostic,
        actual_boxes=boxes,
        prior_map=observed.occupancy(prior),
        observed_map=c.grid,
        substep_state=states,
        substep_force=forces,
        substep_contact=contacts,
        model_xml=np.array(plant.xml),
    )
    brake = np.isin(data["reason"], ["swept_body_brake", "queue_body_brake"])
    trigger = len(data["truth"]) - 1 if trigger is None else trigger
    row = {
        "method": method,
        "controller_margin_m": margin,
        "contact_allowed": allow_contact,
        "passed": passed,
        "contact_events": contact_events,
        "contact_peak_force_n": float(full[:, 1].max(initial=0)),
        "contact_impulse_ns": float(full[:, 2].sum() * PHYSICS_DT),
        "contact_penetration_m": float(full[:, 3].max(initial=0)),
        "contact_total_s": float(active.sum() * PHYSICS_DT),
        "contact_longest_s": float(full[:, 4].max(initial=0)),
        "severe_contact": not mild,
        "six_cm_diagnostic_min_m": six_cm_min,
        "terminal_world_velocity": plant.data.qvel.tolist(),
        "condition_key": condition.key,
        "profile": profile,
        "local_override_frames": int(data["local_override"].sum()),
        "expiry_frames": int(data["expiry_active"].sum()),
        "blackout_encountered": bool((~data["control_fresh"][:-1]).any()),
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
        "contacts": contact_events,
        "mean_m": float(abs(condition.dy_m)),
        "p95_m": float(abs(condition.dy_m)),
        "end_m": float(abs(condition.dy_m)),
        "abs_heading_error_deg": float(abs(condition.yaw_deg)),
        "frames": len(data["truth"]),
        "elapsed_sim_s": (len(data["truth"]) - 1) * DT,
        "events": [{"frame": len(data["truth"]) - 1, "success": completed}] if announced else [],
        "min_clearance_m": physical_min,
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
        return [("narrow", "crate", 0, "development", "delay_2", m, "normal") for m in METHODS]
    cases = [
        ("narrow", ob, seed, "scan", "delay_2", m, "normal")
        for ob in ("crate", "low", "beam")
        for seed in (14, 15, 16)
        for m in METHODS
    ]
    cases += [
        ("narrow", "crate", 14, "delay", d, m, "normal")
        for d in ("zero", "delay_4")
        for m in METHODS
    ]
    cases += [("street", "crate", 14, "environment", "delay_2", m, "normal") for m in METHODS]
    cases += [("blocked", "crate", 14, "negative", "delay_2", m, "normal") for m in METHODS]
    assert len(cases) == 39
    return cases


def run(output=DEFAULT_OUTPUT, smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    previous_dir = ROOT / "results/execution_safety_v5"
    previous = json.loads((previous_dir / "summary.json").read_text(encoding="utf8"))
    sources = [ROOT / name for name in previous["protocol"]["source_sha256"]]
    sources += [Path(__file__), ROOT / "src/embodied_learning/contact_motion.py"]
    for name, expected in previous["protocol"]["source_sha256"].items():
        assert digest(ROOT / name) == expected, f"frozen source changed: {name}"
    for file in sources:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    cases = registered_cases(smoke)
    import mujoco

    protocol = {
        "lesson": "69",
        "round": 6,
        "dt_s": DT,
        "physics_dt_s": PHYSICS_DT,
        "smoke": smoke,
        "methods": list(METHODS),
        "factors": METHODS,
        "cases": cases,
        "rig": replace(NavigationRig(), margin_m=0).to_dict(),
        "contact_limits": LIMITS,
        "mujoco_version": mujoco.__version__,
        "plant": "8kg planar x/y/yaw body; bounded world velocity servo; contact response at 500Hz",
        "controller": "frozen coupled + queue forecast + local guard; measured forward and yaw velocity",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "baseline_summary_sha256": digest(previous_dir / "summary.json"),
        "registration_sha256": digest(ROOT / "docs/69-contact-round6-plan.md"),
    }
    save_json(out / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    started = time.perf_counter()
    for layout, ob, seed, cohort, delay, method, profile in cases:
        tick = time.perf_counter()
        row, data = episode(layout, ob, seed, CONDITIONS[delay], method, profile)
        file = f"{layout}_{ob}_s{seed}_{delay}_{method}_{profile}.npz"
        np.savez_compressed(out / file, **data)
        row.update(
            file=file, sha256=digest(out / file), cohort=cohort, wall_s=time.perf_counter() - tick
        )
        summary["rows"].append(row)
        summary["wall_s"] = time.perf_counter() - started
        save_json(out / "summary.json", summary)
        print(
            f"{len(summary['rows'])}/{len(cases)} {layout}/{ob}/s{seed} {delay} {method}: {row['status']} passed={row['passed']} contact={row['contacts']} F={row['contact_peak_force_n']:.2f} {row['wall_s']:.1f}s",
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
