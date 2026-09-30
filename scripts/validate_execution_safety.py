"""Independent reconstruction of queue, local override, motion and true clearance."""

import math
import re
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image
from validate_braking_navigation import integrate, physical_body_gap, sampled_motion
from validate_footprint_tracking import arrays, corner_separation, read, sha

from embodied_learning.navigation_geometry import (
    NavigationRig,
    body_points,
    depth_points,
    lidar_obstacle_points,
    world_points,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/execution_safety_v5"


def brake(u):
    t = max(abs(u[0]) / 0.8, abs(u[1]) / 1.8)
    return np.zeros(2) if t <= 0.1 + 1e-12 else u * (1 - 0.1 / t)


def limited(u, target):
    return u + np.minimum(np.maximum(target - u, [-0.08, -0.18]), [0.08, 0.18])


def predicted(u, sequence, rig):
    pose, poses = np.zeros(3), [np.zeros(3)]
    current = np.asarray(u, float).copy()
    seq = list(sequence)
    while seq or np.max(np.abs(current)) >= 1e-12:
        if seq:
            target, stop = seq.pop(0)
            current = brake(current) if stop else limited(current, target)
        else:
            current = brake(current)
        n = max(1, math.ceil((abs(current[0]) + rig.radius * abs(current[1])) * 0.1 / 0.01))
        times = np.linspace(0, 0.1, n + 1)[1:]
        samples = integrate(
            np.repeat(pose[None], n, axis=0), np.repeat(current[None], n, axis=0), times
        )
        poses.extend(samples)
        pose = samples[-1]
    return np.asarray(poses)


def executing_path(u, rig):
    n = max(1, math.ceil((abs(u[0]) + rig.radius * abs(u[1])) * 0.1 / 0.01))
    head = integrate(
        np.zeros((n + 1, 3)),
        np.repeat(np.asarray(u)[None], n + 1, axis=0),
        np.linspace(0, 0.1, n + 1),
    )
    tail = predicted(u, [], rig)
    theta = head[-1, 2]
    tail[:, :2] = (
        tail[:, :2] @ [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        + head[-1, :2]
    )
    tail[:, 2] += theta
    return np.vstack((head, tail[1:]))


def point_gap(points, poses, rig):
    if len(points) == 0:
        return float("inf")
    # Vectorized independent box signed-distance calculation in chunks.
    minimum = np.inf
    for start in range(0, len(poses), 16):
        p = poses[start : start + 16]
        delta = points[None, :, :2] - p[:, None, :2]
        c, s = np.cos(p[:, 2, None]), np.sin(p[:, 2, None])
        x, y = delta[:, :, 0] * c + delta[:, :, 1] * s, -delta[:, :, 0] * s + delta[:, :, 1] * c
        for part in rig.parts:
            q = np.stack(
                (
                    np.abs(x - (part[0] + part[1]) / 2) - (part[1] - part[0]) / 2 - rig.margin_m,
                    np.abs(y - (part[2] + part[3]) / 2) - (part[3] - part[2]) / 2 - rig.margin_m,
                    np.broadcast_to(
                        np.abs(points[:, 2] - (part[4] + part[5]) / 2) - (part[5] - part[4]) / 2,
                        x.shape,
                    ),
                ),
                axis=2,
            )
            distance = np.sqrt((np.maximum(q, 0) ** 2).sum(axis=2)) + np.minimum(q.max(axis=2), 0)
            minimum = min(minimum, float(distance.min()))
    return minimum


def main():
    started = time.perf_counter()
    summary, protocol = read(DATA / "summary.json"), read(DATA / "protocol.json")
    assert protocol == summary["protocol"] and not protocol["smoke"]
    assert sha(DATA / "protocol.json") == summary["protocol_sha256"]
    assert len(summary["rows"]) == 135
    assert Counter(r["cohort"] for r in summary["rows"]) == {
        "bridge": 3,
        "scan": 108,
        "environment": 8,
        "negative": 4,
        "blackout": 12,
    }
    assert sha(ROOT / "docs/69-execution-round5-plan.md") == protocol["registration_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert sha(DATA / "source" / name) == sha(ROOT / name) == expected
    benchmark_path = ROOT / "docs/benchmarks/execution-safety-v5.json"
    assert read(benchmark_path)["summary_sha256"] == sha(DATA / "summary.json")
    old_dir = ROOT / "results/navigation_robustness_v4"
    old = read(old_dir / "summary.json")
    assert sha(old_dir / "summary.json") == protocol["baseline_summary_sha256"]
    steps = frames = samples_count = bridges = overrides = expiries = forecasts = 0
    worst_motion = worst_body = worst_margin = worst_prediction = 0.0
    rig = NavigationRig(**protocol["rig"])
    for row in summary["rows"]:
        d = arrays(DATA / row["file"], row["sha256"])
        p, u, inc = d["truth"], d["commands"], d["incoming_velocity"]
        n, lag = len(p), row["perturbation"]["delay_steps"]
        forecast, guard = protocol["factors"][row["method"]]
        np.testing.assert_array_equal(p, d["estimate"])
        np.testing.assert_array_equal(inc[0], [0, 0])
        np.testing.assert_array_equal(inc[1:], u[:-1])
        predicted_motion = integrate(p[:-1], u[:-1], np.full(n - 1, 0.1))
        worst_motion = max(worst_motion, float(np.max(np.abs(predicted_motion - p[1:]), initial=0)))
        assert worst_motion < 1e-10
        assert np.all(np.abs(u[:-1] - inc[:-1]) <= [0.08 + 1e-12, 0.18 + 1e-12])
        queue = deque((np.zeros(2), True, -1) for _ in range(lag))
        memory, last_fresh, latch = {}, 0, False
        for k in range(n - int(bool(row["contacts"]))):
            fresh = not (row["profile"] == "blackout" and 50 <= k < 60)
            assert bool(d["control_fresh"][k]) == fresh
            if d["reason"][k] != "abort_braking" or not fresh:
                points = np.vstack(
                    (
                        lidar_obstacle_points(d["ranges"][k], d["hits"][k], rig),
                        depth_points(d["depth"][k], d["depth_valid"][k], rig),
                    )
                )
                world = world_points(points, p[k])
                quant = np.floor(world / 0.08).astype(int)
                _, ids = np.unique(quant, axis=0, return_index=True)
                for i in ids:
                    memory[tuple(quant[i])] = world[i]
            points = body_points(np.array(list(memory.values())).reshape(-1, 3), p[k])
            points = points[np.linalg.norm(points[:, :2], axis=1) < 2.5]
            if np.isfinite(d["queue_candidate_gap"][k]):
                assert forecast and fresh
                sequence = [(t, b) for t, b, _ in queue]
                candidate_gap = point_gap(
                    points,
                    predicted(inc[k], sequence + [(d["nominal_command"][k], False)], rig),
                    rig,
                )
                stop_gap = point_gap(
                    points, predicted(inc[k], sequence + [(np.zeros(2), True)], rig), rig
                )
                worst_prediction = max(
                    worst_prediction,
                    abs(candidate_gap - d["queue_candidate_gap"][k]),
                    abs(stop_gap - d["queue_stop_gap"][k]),
                )
                assert (d["reason"][k] == "queue_body_brake") == (min(candidate_gap, stop_gap) <= 0)
                forecasts += 1
            if fresh:
                last_fresh = k
                queue.append((d["requested_command"][k], bool(d["braking_active"][k]), k))
                target, stop, source = queue.popleft()
            elif queue:
                queue.append((inc[k], False, -2))
                target, stop, source = queue.popleft()
            else:
                target, stop, source = inc[k], False, -2
            proposed = brake(inc[k]) if stop else limited(inc[k], target)
            gap = point_gap(points, executing_path(proposed, rig), rig)
            if np.isinf(gap):
                assert gap == d["local_candidate_gap"][k]
            else:
                worst_prediction = max(worst_prediction, abs(gap - d["local_candidate_gap"][k]))
            assert worst_prediction < 1e-10
            expired = guard and k - last_fresh >= 2
            latch |= expired
            override = guard and (gap <= 0 or latch)
            expected = brake(inc[k]) if override else proposed
            if override:
                queue = deque((np.zeros(2), True, -1) for _ in queue)
            np.testing.assert_allclose(u[k], expected, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(d["executed_target"][k], target)
            assert d["executed_request_frame"][k] == source
            assert bool(d["executed_braking"][k]) == (stop or override)
            assert bool(d["local_override"][k]) == override
            assert bool(d["expiry_active"][k]) == expired
            assert d["control_age_ticks"][k] == k - last_fresh
            assert d["pending_motion"][k] == sum(
                not b and np.max(np.abs(t)) > 1e-12 for t, b, _ in queue
            )
            overrides += int(override)
            expiries += int(expired)
        stopped = np.max(np.abs(inc[-1])) < 1e-12
        assert stopped == row["physically_stopped"]
        assert row["contacts"] or (stopped and d["pending_motion"][-1] == 0)
        announced = (
            stopped
            and d["phase"][-1] == "goal_braking"
            and np.linalg.norm(p[-1, :2] - d["goal"][-1]) < 0.25
        )
        assert announced == bool(row["announced"]) == bool(row["reached"])
        assert not row["false_arrival"]
        samples = sampled_motion(p, u, protocol["rig"])
        margin = corner_separation(samples, d["actual_boxes"], protocol["rig"])
        body = physical_body_gap(samples, d["actual_boxes"], protocol["rig"])
        worst_margin = max(worst_margin, abs(float(margin.min()) - row["expanded_swept_min_m"]))
        worst_body = max(worst_body, abs(float(body.min()) - row["min_clearance_m"]))
        assert worst_margin < 1e-10 and worst_body < 1e-10
        assert bool(body.min() <= 0) == bool(row["contacts"])
        assert bool(announced and margin.min() > 0) == row["safe_completed"]
        if row["cohort"] == "bridge":
            before = next(
                r
                for r in old["rows"]
                if all(r[a] == row[a] for a in ("layout", "obstacle", "seed"))
                and r["method"] == row["condition_key"]
            )
            saved = arrays(old_dir / before["file"], before["sha256"])
            for key, values in saved.items():
                if key != "control_wall_s":
                    assert np.array_equal(
                        values, d[key], equal_nan=values.dtype.kind not in "US"
                    ), key
            bridges += 1
        if row["cohort"] == "negative":
            assert n == 1 and row["travelled_m"] == 0 and row["status"] == "no_path"
        frames += n
        steps += n - 1
        samples_count += len(samples)
    images = {}
    for file in sorted((ROOT / "docs/img").glob("lesson69-round5-*.png")):
        with Image.open(file) as im:
            assert im.width >= 1200 and im.height >= 600
            images[file.name] = {"sha256": sha(file), "size": list(im.size)}
    assert len(images) == 4
    logs = {}
    for name in ("quick", "affected"):
        path = ROOT / f"results/execution_safety_{name}.log"
        text = path.read_text(encoding="utf8")
        assert " passed" in text and not re.search(r"\d+ failed|ERRORS", text)
        logs[name] = {
            "sha256": sha(path),
            "summary": next(s for s in reversed(text.splitlines()) if " passed" in s),
        }
    doc_paths = [ROOT / "README.md"] + [
        ROOT / "docs" / name
        for name in (
            "69-execution-round5.md",
            "69-execution-round5-plan.md",
            "69-robustness-round4.md",
            "navigation-stage-review-69-71.md",
            "01-learning-roadmap.md",
            "multisensor-navigation-roadmap.md",
            "34-experiment-decision-log.md",
        )
    ]
    checked_links = 0
    for path in doc_paths:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf8")):
            target = target.split("#", 1)[0].strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            assert (path.parent / target).exists(), (path, target)
            checked_links += 1
    report = {
        "episodes": 135,
        "frames": frames,
        "steps": steps,
        "sweep_samples": samples_count,
        "bridges": bridges,
        "forecast_checks": forecasts,
        "local_overrides": overrides,
        "expiry_checks": expiries,
        "max_motion_error": worst_motion,
        "max_physical_gap_error": worst_body,
        "max_margin_error": worst_margin,
        "max_prediction_gap_error": worst_prediction,
        "images": images,
        "tests": logs,
        "summary_sha256": sha(DATA / "summary.json"),
        "benchmark_sha256": sha(benchmark_path),
        "audit_sha256": sha(Path(__file__)),
        "doc_sha256": {str(p.relative_to(ROOT)): sha(p) for p in doc_paths},
        "local_links_checked": checked_links,
        "audit_wall_s": time.perf_counter() - started,
        "boundary": "independent queue/override and motion/geometry checks; shared sensor transforms; fixed blackout schedule; finite physical-corner sampling <=1cm; no hard real-time deployment proof",
    }
    save = ROOT / "docs/benchmarks/execution-safety-v5-validation.json"
    import json

    save.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    print(report)


if __name__ == "__main__":
    main()
