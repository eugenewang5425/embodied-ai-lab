"""Lesson 69 round three: five-arm braking study with real terminal deceleration."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_learning.braking_control import DT, MODES, BrakingController, brake_step, reachable
from embodied_learning.experiments import observed_navigation as observed
from embodied_learning.experiments.footprint_tracking import diagnostics, digest, save_json
from embodied_learning.navigation_geometry import (
    NavigationRig,
    advance,
    collision_clearance,
    sense,
    swept_clearance,
)
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = "results/braking_navigation_v3"


def scene(layout, obstacle):
    if layout not in ("offset_narrow", "blocked"):
        return observed.scene(layout, obstacle)
    prior, boxes = observed.scene("narrow", obstacle)
    if layout == "offset_narrow":
        boxes[-1, :4] = [4.8, 5.6, 5.7, 6.5]
    else:
        prior[-2, 3], prior[-1, 2] = 5.75, 6.25
        boxes[:-1] = prior
    return prior, boxes


def episode(layout, obstacle, seed, method, max_steps=600):
    rig = NavigationRig()
    prior, boxes = scene(layout, obstacle)
    c = BrakingController(observed.occupancy(prior), rig, method)
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
    )
    records = {name: [] for name in names}
    phase, failure, trigger, eligible = "running", None, None, None
    best, progress = 20.0, 0
    gaps, timings = [], []
    # Allow at most two seconds of actual braking beyond the nominal timeout.
    for k in range(max_steps + 21):
        ranges, hits, depth, valid = sense(boxes, pose, rig, rng)
        started = time.perf_counter()
        if phase in ("running", "goal_braking"):
            before = len(c.tracking_log)
            desired, reason, clearance = c.step(
                pose.copy(), goal, ranges, hits, depth, valid, k, velocity
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
        distance = float(np.linalg.norm(pose[:2] - goal))
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
        command = (
            brake_step(velocity, rig, c.coupled) if stopping else reachable(velocity, desired, rig)
        )
        at_rest = np.max(np.abs(velocity)) < 1e-12
        terminal = phase != "running" and at_rest
        if terminal:
            command = np.zeros(2)
        elapsed = time.perf_counter() - started
        timings.append(elapsed)
        plan = np.full((c.path_capacity, 2), np.nan)
        if c.path is not None:
            plan[: len(c.path)] = c.path
        values = (
            pose.copy(),
            pose.copy(),
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
                pose.copy(),
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
            )
            for name, value in zip(names, values, strict=True):
                records[name].append(value)
            break
    else:
        raise RuntimeError("terminal deceleration exceeded physical bound")
    data = {name: np.asarray(values) for name, values in records.items()}
    final_distance = float(np.linalg.norm(data["truth"][-1, :2] - goal))
    stopped = bool(np.max(np.abs(data["incoming_velocity"][-1])) < 1e-12)
    completed = phase == "goal_braking" and stopped and final_distance < 0.25
    status = "completed" if completed else failure or "goal_overshoot"
    diagnostic, minimum = diagnostics(data, prior, boxes, rig)
    if minimum is None:
        minimum = float(diagnostic["expanded_gap_m"].min())
    data.update(
        diagnostic, actual_boxes=boxes, prior_map=observed.occupancy(prior), observed_map=c.grid
    )
    brake = data["reason"] == "swept_body_brake"
    trigger = len(data["truth"]) - 1 if trigger is None else trigger
    row = {
        "method": method,
        "layout": layout,
        "obstacle": obstacle,
        "seed": seed,
        "status": status,
        "reached": int(completed),
        "announced": int(completed),
        "target_count": 1,
        "contacts": int(failure == "collision_abort"),
        "mean_m": 0.0,
        "p95_m": 0.0,
        "end_m": 0.0,
        "frames": len(data["truth"]),
        "elapsed_sim_s": (len(data["truth"]) - 1) * DT,
        "events": [{"frame": len(data["truth"]) - 1, "success": True}] if completed else [],
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


def verify_prefix(row, data, source, old):
    r = next(
        r
        for r in old["rows"]
        if r["method"] == "combined" and all(r[k] == row[k] for k in ("layout", "obstacle", "seed"))
    )
    before = read_checked(source / r["file"], r["sha256"])
    n = len(before["truth"])
    for key in (
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
    ):
        # All old executed frames; terminal sample had no outgoing motion.
        assert np.array_equal(
            before[key][:-1], data[key][: n - 1], equal_nan=before[key].dtype.kind not in "US"
        ), (row.get("file", row), key)
    assert np.array_equal(before["truth"][-1], data["truth"][n - 1])
    row.update(legacy_prefix_equal=True, legacy_executed_frames=n - 1, baseline_sha256=r["sha256"])


def run(output=DEFAULT_OUTPUT, smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__),
        *[
            ROOT / "src/embodied_learning" / p
            for p in (
                "braking_control.py",
                "path_tracking.py",
                "footprint_planning.py",
                "navigation_geometry.py",
                "experiments/footprint_tracking.py",
                "experiments/observed_navigation.py",
                "experiments/grid_nav.py",
            )
        ],
    ]
    for file in files:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    source = ROOT / "results/footprint_tracking_v2"
    old = json.loads((source / "summary.json").read_text(encoding="utf8"))
    cases = [
        (layout, ob, seed, "fixed")
        for layout in ("plaza", "street", "narrow")
        for ob in ("crate", "low", "beam")
        for seed in (0, 1, 2)
    ]
    cases += [
        (layout, ob, seed, cohort)
        for layout, cohort in (("narrow", "heldout_noise"), ("offset_narrow", "heldout_layout"))
        for ob in ("crate", "low", "beam")
        for seed in (5, 6, 7)
    ]
    cases += [("blocked", "crate", 5, "negative")]
    if smoke:
        cases = [("narrow", "crate", seed, "development") for seed in (0, 2)]
    protocol = {
        "lesson": "69",
        "round": 3,
        "methods": list(MODES),
        "rig": NavigationRig().to_dict(),
        "dt_s": DT,
        "design": "legacy bridge plus 2x2 independent/coupled braking x fixed/curvature speed cap",
        "cases": cases,
        "smoke": smoke,
        "lateral_acceleration_budget_m_s2": 0.04,
        "preview_distance": "0.35+latency*abs(v)+v*v/(2*brake)",
        "forecast": "same 0.1s commands as execution; union of immediate and 0.2s delayed stops; <=1cm corner-travel samples",
        "terminal": "latch goal/no_path/stall/timeout and execute deceleration to exact rest; collision abort retains incoming speed and no fictitious rest",
        "arrival": "true distance <0.25m and v<0.05m/s eligibility retained; complete only after full stop still within same circle",
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
        "baseline_summary_sha256": digest(source / "summary.json"),
        "registration_sha256": digest(ROOT / "docs/69-braking-round3-plan.md"),
        "score_access": "truth and actual geometry only sensing and offline scoring; controller receives estimate, prior, observations",
    }
    save_json(out / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    started = time.perf_counter()
    for layout, obstacle, seed, cohort in cases:
        for method in MODES:
            tick = time.perf_counter()
            row, data = episode(layout, obstacle, seed, method)
            if (
                method == "legacy_brake"
                and layout in ("plaza", "street", "narrow")
                and seed in (0, 1, 2)
            ):
                verify_prefix(row, data, source, old)
            file = f"{layout}_{obstacle}_s{seed}_{method}.npz"
            np.savez_compressed(out / file, **data)
            row.update(
                file=file,
                sha256=digest(out / file),
                cohort=cohort,
                wall_s=time.perf_counter() - tick,
            )
            summary["rows"].append(row)
            summary["wall_s"] = time.perf_counter() - started
            save_json(out / "summary.json", summary)
            print(
                cohort,
                layout,
                obstacle,
                seed,
                method,
                row["status"],
                f"safe={row['safe_completed']}",
                f"{row['wall_s']:.1f}s",
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
