"""Independent round-three action, complete-stop, corner geometry and provenance audit."""

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from validate_footprint_tracking import arrays, corner_separation, read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/braking_navigation_v3"


def integrate(poses, commands, times):
    turn = commands[:, 1] * times
    yaw = poses[:, 2] + turn / 2
    travel = commands[:, 0] * times * np.sinc(turn / (2 * np.pi))
    xy = poses[:, :2] + travel[:, None] * np.c_[np.cos(yaw), np.sin(yaw)]
    angle = poses[:, 2] + turn
    return np.c_[xy, np.arctan2(np.sin(angle), np.cos(angle))]


def sampled_motion(pose, commands, rig):
    if len(pose) == 1:
        return pose.copy()
    radius = np.hypot(max(rig["front_m"], rig["rear_m"]), rig["half_width_m"])
    count = np.maximum(
        1,
        np.ceil((np.abs(commands[:-1, 0]) + radius * np.abs(commands[:-1, 1])) * 0.1 / 0.01).astype(
            int
        ),
    )
    indices = np.repeat(np.arange(len(count)), count + 1)
    times = np.concatenate([np.linspace(0, 0.1, n + 1) for n in count])
    return integrate(pose[indices], commands[indices], times)


def physical_body_gap(poses, boxes, rig):
    # Independently enumerate the three physical components, including heights.
    parts = [
        (-rig["rear_m"], rig["front_m"], rig["half_width_m"], 0, 0.30),
        (0.14, 0.19, 0.025, 0.30, 0.56),
        (0.15, 0.20, 0.03, 0.52, rig["top_m"]),
    ]
    minimum = np.full(len(poses), np.inf)
    for back, front, half_width, z0, z1 in parts:
        obstacles = boxes[(boxes[:, 5] > z0) & (boxes[:, 4] < z1)]
        part = {
            **rig,
            "rear_m": -back,
            "front_m": front,
            "half_width_m": half_width,
            "top_m": z1,
            "margin_m": 0,
        }
        minimum = np.minimum(minimum, corner_separation(poses, obstacles, part))
    return minimum


def main():
    tick = time.perf_counter()
    summary, protocol = read(DATA / "summary.json"), read(DATA / "protocol.json")
    assert protocol == summary["protocol"] and len(summary["rows"]) == 230 and not protocol["smoke"]
    assert sha(DATA / "protocol.json") == summary["protocol_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert sha(DATA / "source" / name) == sha(ROOT / name) == expected
    assert sha(ROOT / "docs/69-braking-round3-plan.md") == protocol["registration_sha256"]
    benchmark_path = ROOT / "docs/benchmarks/braking-navigation-v3.json"
    benchmark = read(benchmark_path)
    assert sha(DATA / "summary.json") == benchmark["summary_sha256"]
    old_dir = ROOT / "results/footprint_tracking_v2"
    assert sha(old_dir / "summary.json") == protocol["baseline_summary_sha256"]
    old = {
        (r["layout"], r["obstacle"], r["seed"]): r
        for r in read(old_dir / "summary.json")["rows"]
        if r["method"] == "combined"
    }
    counts = Counter(r["cohort"] for r in summary["rows"])
    assert counts == {"fixed": 135, "heldout_noise": 45, "heldout_layout": 45, "negative": 5}
    plans = defaultdict(dict)
    steps, frames, prefixes, swept_samples = 0, 0, 0, 0
    worst_motion, worst_expanded, worst_body = 0.0, 0.0, 0.0
    for row in summary["rows"]:
        d = arrays(DATA / row["file"], row["sha256"])
        p, u, incoming = d["truth"], d["commands"], d["incoming_velocity"]
        n = len(p)
        assert n == row["frames"] == len(u) == len(incoming) == len(d["phase"])
        np.testing.assert_array_equal(p, d["estimate"])
        np.testing.assert_array_equal(incoming[0], [0, 0])
        np.testing.assert_array_equal(incoming[1:], u[:-1])
        prediction = integrate(p[:-1], u[:-1], np.full(n - 1, 0.1))
        error = np.max(np.abs(prediction - p[1:]), initial=0)
        worst_motion = max(worst_motion, float(error))
        assert error < 1e-10
        assert np.all(np.abs(u[:-1] - incoming[:-1]) <= [0.08 + 1e-12, 0.18 + 1e-12])
        assert np.all(np.abs(u) <= [0.6 + 1e-12, 1 + 1e-12])
        braking = d["braking_active"][:-1].astype(bool)
        for v, executed in zip(incoming[:-1][braking], u[:-1][braking], strict=True):
            if row["method"] in ("coupled", "joint_brake"):
                duration = max(abs(v[0]) / 0.8, abs(v[1]) / 1.8)
                expected = v * max(0, 1 - 0.1 / duration) if duration > 1e-12 else np.zeros(2)
            else:
                expected = np.sign(v) * np.maximum(0, np.abs(v) - [0.08, 0.18])
            np.testing.assert_allclose(executed, expected, atol=1e-12, rtol=0)
        stopped = np.max(np.abs(incoming[-1])) < 1e-12
        assert stopped == row["physically_stopped"]
        assert row["contacts"] or stopped
        assert np.all(u[-1] == 0)
        arrived = (
            stopped
            and np.linalg.norm(p[-1, :2] - d["goal"][-1]) < 0.25
            and d["phase"][-1] == "goal_braking"
        )
        assert bool(arrived) == bool(row["reached"]) == (row["status"] == "completed")
        samples = sampled_motion(p, u, protocol["rig"])
        expanded = corner_separation(samples, d["actual_boxes"], protocol["rig"])
        body = physical_body_gap(samples, d["actual_boxes"], protocol["rig"])
        worst_expanded = max(
            worst_expanded, abs(float(expanded.min()) - row["expanded_swept_min_m"])
        )
        worst_body = max(worst_body, abs(float(body.min()) - row["min_clearance_m"]))
        assert worst_expanded < 1e-10 and worst_body < 1e-10
        assert bool(body.min() <= 0) == bool(row["contacts"])
        frame_gap = corner_separation(p, d["actual_boxes"], protocol["rig"])
        np.testing.assert_allclose(frame_gap, d["expanded_gap_m"], atol=1e-10, rtol=0)
        assert int((frame_gap <= 0).sum()) == row["expanded_overlap_frames"]
        assert bool(arrived and expanded.min() > 0) == row["safe_completed"]
        f = row["trigger_frame"]
        assert np.isclose((n - 1 - f) * 0.1, row["tail_time_s"])
        distance = np.linalg.norm(np.diff(p[f:, :2], axis=0), axis=1).sum()
        assert np.isclose(distance, row["tail_distance_m"])
        case = row["layout"], row["obstacle"], row["seed"]
        assert row["method"] not in plans[case]
        plans[case][row["method"]] = d["plan"][0]
        if row["method"] == "legacy_brake" and row["cohort"] == "fixed":
            previous = old[case]
            a = arrays(old_dir / previous["file"], previous["sha256"])
            m = len(a["truth"]) - 1
            for key in (
                "truth",
                "estimate",
                "commands",
                "ranges",
                "hits",
                "depth",
                "depth_valid",
                "goal",
                "mission",
                "plan",
                "reason",
                "safety_clearance",
            ):
                assert np.array_equal(
                    a[key][:-1], d[key][:m], equal_nan=a[key].dtype.kind not in "US"
                ), key
            np.testing.assert_array_equal(a["truth"][-1], p[m])
            prefixes += 1
        if row["cohort"] == "negative":
            assert (
                n == 1
                and row["status"] == "no_path"
                and not row["reached"]
                and row["travelled_m"] == 0
            )
        frames += n
        steps += n - 1
        swept_samples += len(samples)
    assert prefixes == 27 and len(plans) == 46
    for case, methods in plans.items():
        assert set(methods) == set(protocol["methods"])
        assert all(
            np.array_equal(p, methods["legacy_brake"], equal_nan=True) for p in methods.values()
        ), case
    images = {}
    for path in sorted((ROOT / "docs/img").glob("lesson69-round3-*.png")):
        with Image.open(path) as im:
            assert im.width >= 1200 and im.height >= 600
            images[path.name] = {"sha256": sha(path), "size": im.size}
    assert len(images) == 5
    logs = {}
    for name, expected in (("quick", "839 passed, 186 deselected"), ("affected", "60 passed")):
        path = ROOT / f"results/braking_navigation_{name}.log"
        text = path.read_text(encoding="utf8", errors="replace")
        assert expected in text and "exit=0" in text
        logs[name] = {"result": expected, "sha256": sha(path), "wrapper": text.splitlines()[-1]}
    output = ROOT / "docs/benchmarks/braking-navigation-v3-validation.json"
    documents = [ROOT / "README.md"] + [
        ROOT / "docs" / name
        for name in (
            "69-braking-round3.md",
            "69-braking-round3-plan.md",
            "69-footprint-planning.md",
            "69-footprint-tracking-round2.md",
            "navigation-stage-review-69-71.md",
            "01-learning-roadmap.md",
            "multisensor-navigation-roadmap.md",
            "replay-window.md",
        )
    ]
    links = 0
    for path in documents:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            assert resolved.exists() or resolved == output, (path, target)
            links += 1
    report = {
        "date": "2026-09-30",
        "records": 230,
        "frames": frames,
        "executed_steps": steps,
        "independent_swept_geometry_samples": swept_samples,
        "source_live_and_snapshot_hashes": protocol["source_sha256"],
        "legacy_bitwise_executed_prefixes": prefixes,
        "same_initial_plan_five_arm_cases": len(plans),
        "max_motion_difference": worst_motion,
        "max_expanded_gap_difference_m": worst_expanded,
        "max_physical_body_gap_difference_m": worst_body,
        "physically_stopped_records": sum(r["physically_stopped"] for r in summary["rows"]),
        "collision_records": sum(r["contacts"] for r in summary["rows"]),
        "actual_arrivals": sum(r["reached"] for r in summary["rows"]),
        "arrived_and_margin_clear": sum(r["safe_completed"] for r in summary["rows"]),
        "no_motion_negative_records": 5,
        "terminal_rule": "Rest confirmed by incoming executed velocity, not final zero marker; failure tail included in independent swept scoring.",
        "geometry_boundary": "Finite <=1cm corner-travel samples and static boxes; no continuous-time proof or hardware certification.",
        "summary_sha256": sha(DATA / "summary.json"),
        "benchmark_sha256": sha(benchmark_path),
        "images": images,
        "visual_review": "Four charts and own Tk window inspected; five methods, safe-completion labels, camera, goal, LiDAR and sixth diagnostic tab synchronized.",
        "logs": logs,
        "regression_scope": "quick plus all affected navigation/window tests; frozen legacy algorithm modules unchanged; no new whole-repository slow run",
        "local_links_checked": links,
        "documents_sha256": {str(p.relative_to(ROOT)): sha(p) for p in documents},
        "audit_sources_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (Path(__file__), ROOT / "scripts/validate_footprint_tracking.py")
        },
        "wall_s": time.perf_counter() - tick,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "records",
                    "frames",
                    "executed_steps",
                    "independent_swept_geometry_samples",
                    "actual_arrivals",
                    "arrived_and_margin_clear",
                    "wall_s",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
