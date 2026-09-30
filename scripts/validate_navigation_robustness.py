"""Independent queue, motion, injected-error, truth-score and geometry audit."""

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from validate_braking_navigation import integrate, physical_body_gap, sampled_motion
from validate_footprint_tracking import arrays, corner_separation, read, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/navigation_robustness_v4"


def main():
    tick = time.perf_counter()
    summary, protocol = read(DATA / "summary.json"), read(DATA / "protocol.json")
    assert protocol == summary["protocol"] and not protocol["smoke"]
    assert sha(DATA / "protocol.json") == summary["protocol_sha256"]
    assert len(summary["rows"]) == 233
    assert Counter(r["cohort"] for r in summary["rows"]) == {
        "bridge": 27,
        "scan": 189,
        "environment": 12,
        "negative": 5,
    }
    assert sha(ROOT / "docs/69-robustness-round4-plan.md") == protocol["registration_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert sha(DATA / "source" / name) == sha(ROOT / name) == expected
    previous_dir = ROOT / "results/braking_navigation_v3"
    assert sha(previous_dir / "summary.json") == protocol["baseline_summary_sha256"]
    previous = read(previous_dir / "summary.json")
    previous_rows = {
        (r["layout"], r["obstacle"], r["seed"]): r
        for r in previous["rows"]
        if r["method"] == "coupled"
    }
    benchmark_path = ROOT / "docs/benchmarks/navigation-robustness-v4.json"
    benchmark = read(benchmark_path)
    assert benchmark["summary_sha256"] == sha(DATA / "summary.json")
    assert benchmark["protocol"] == protocol
    assert len(benchmark["rows"]) == 233
    frames = steps = samples_count = bridges = delay_steps_checked = 0
    worst_motion = worst_body = worst_margin = 0.0
    for row in summary["rows"]:
        d = arrays(DATA / row["file"], row["sha256"])
        p, u, inc, e = (d[k] for k in ("truth", "commands", "incoming_velocity", "estimate"))
        c = row["perturbation"]
        assert c == protocol["conditions"][row["method"]]
        n, lag = len(p), c["delay_steps"]
        assert n == row["frames"]
        np.testing.assert_array_equal(inc[0], [0, 0])
        np.testing.assert_array_equal(inc[1:], u[:-1])
        np.testing.assert_allclose(
            e[:, :2] - p[:, :2], np.tile([0, c["dy_m"]], (n, 1)), atol=1e-12, rtol=0
        )
        turn = e[:, 2] - p[:, 2]
        np.testing.assert_allclose(
            np.arctan2(np.sin(turn), np.cos(turn)), np.deg2rad(c["yaw_deg"]), atol=1e-12, rtol=0
        )
        predicted = integrate(p[:-1], u[:-1], np.full(n - 1, 0.1))
        worst_motion = max(worst_motion, float(np.max(np.abs(predicted - p[1:]), initial=0)))
        assert worst_motion < 1e-10
        assert np.all(np.abs(u[:-1] - inc[:-1]) <= [0.08 + 1e-12, 0.18 + 1e-12])
        assert np.all(np.abs(u) <= [0.6 + 1e-12, 1 + 1e-12])
        for k in range(n - 1):
            source = k - lag
            expected_index = source if source >= 0 else -1
            assert d["executed_request_frame"][k] == expected_index
            target = d["requested_command"][source] if source >= 0 else np.zeros(2)
            brake = bool(d["braking_active"][source]) if source >= 0 else True
            assert bool(d["executed_braking"][k]) == brake
            np.testing.assert_array_equal(d["executed_target"][k], target)
            if brake:
                duration = max(abs(inc[k, 0]) / 0.8, abs(inc[k, 1]) / 1.8)
                expected = inc[k] * max(0, 1 - 0.1 / duration) if duration > 1e-12 else np.zeros(2)
            else:
                expected = inc[k] + np.clip(target - inc[k], [-0.08, -0.18], [0.08, 0.18])
            np.testing.assert_allclose(u[k], expected, atol=1e-12, rtol=0)
            pending = sum(
                not d["braking_active"][j] and np.max(np.abs(d["requested_command"][j])) > 1e-12
                for j in range(max(0, k - lag + 1), k + 1)
            )
            assert d["pending_motion"][k] == pending
            if lag:
                delay_steps_checked += 1
        stopped = np.max(np.abs(inc[-1])) < 1e-12
        assert row["physically_stopped"] == stopped
        assert row["contacts"] or (stopped and d["pending_motion"][-1] == 0)
        np.testing.assert_array_equal(u[-1], [0, 0])
        goal = d["goal"][-1]
        estimated_distance, true_distance = (
            np.linalg.norm(e[-1, :2] - goal),
            np.linalg.norm(p[-1, :2] - goal),
        )
        announced = stopped and d["phase"][-1] == "goal_braking" and estimated_distance < 0.25
        arrived = announced and true_distance < 0.25
        assert bool(announced) == bool(row["announced"])
        assert bool(arrived) == bool(row["reached"])
        assert bool(announced and not arrived) == row["false_arrival"]
        assert np.isclose(estimated_distance, row["final_estimated_goal_distance_m"])
        assert np.isclose(true_distance, row["final_goal_distance_m"])
        trigger = row["trigger_frame"]
        if d["phase"][trigger] == "goal_braking":
            assert np.linalg.norm(e[trigger, :2] - goal) < 0.25
        assert np.isclose(row["tail_time_s"], (n - 1 - trigger) * 0.1)
        assert np.isclose(
            row["tail_distance_m"], np.linalg.norm(np.diff(p[trigger:, :2], axis=0), axis=1).sum()
        )
        samples = sampled_motion(p, u, protocol["rig"])
        margin = corner_separation(samples, d["actual_boxes"], protocol["rig"])
        body = physical_body_gap(samples, d["actual_boxes"], protocol["rig"])
        worst_body = max(worst_body, abs(float(body.min()) - row["min_clearance_m"]))
        worst_margin = max(worst_margin, abs(float(margin.min()) - row["expanded_swept_min_m"]))
        assert worst_body < 1e-10 and worst_margin < 1e-10
        assert bool(body.min() <= 0) == bool(row["contacts"])
        assert bool(arrived and margin.min() > 0) == row["safe_completed"]
        if row["cohort"] == "bridge":
            old = previous_rows[row["layout"], row["obstacle"], row["seed"]]
            before = arrays(previous_dir / old["file"], old["sha256"])
            for key, value in before.items():
                if key != "control_wall_s":
                    assert np.array_equal(value, d[key], equal_nan=value.dtype.kind not in "US"), (
                        key
                    )
            assert row["bridge_equal"] and row["bridge_sha256"] == old["sha256"]
            bridges += 1
        if row["cohort"] == "negative":
            assert n == 1 and row["status"] == "no_path" and row["travelled_m"] == 0
        frames += n
        steps += n - 1
        samples_count += len(samples)
    assert bridges == 27
    images = {}
    for file in sorted((ROOT / "docs/img").glob("lesson69-round4-*.png")):
        with Image.open(file) as im:
            assert im.width >= 1200 and im.height >= 600
            images[file.name] = {"sha256": sha(file), "size": im.size}
    assert len(images) == 5
    logs = {}
    for name in ("quick", "affected"):
        file = ROOT / f"results/navigation_robustness_{name}.log"
        text = file.read_text(encoding="utf8", errors="replace")
        assert "exit=0" in text and " passed" in text
        logs[name] = {
            "sha256": sha(file),
            "wrapper": text.splitlines()[-1],
            "summary": re.findall(r"\d+ passed[^\n]*", text)[-1],
        }
    documents = [ROOT / "README.md"] + [
        ROOT / "docs" / name
        for name in (
            "69-robustness-round4-plan.md",
            "69-robustness-round4.md",
            "69-braking-round3.md",
            "69-footprint-planning.md",
            "69-footprint-tracking-round2.md",
            "navigation-stage-review-69-71.md",
            "01-learning-roadmap.md",
            "multisensor-navigation-roadmap.md",
            "replay-window.md",
            "34-experiment-decision-log.md",
        )
    ]
    links = 0
    output = ROOT / "docs/benchmarks/navigation-robustness-v4-validation.json"
    for file in documents:
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text(encoding="utf8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (file.parent / target.split("#")[0]).resolve()
            assert resolved.exists() or resolved == output, (file, target)
            links += 1
    report = {
        "date": "2026-09-30",
        "records": 233,
        "frames": frames,
        "executed_steps": steps,
        "delayed_executed_steps_checked": delay_steps_checked,
        "swept_samples": samples_count,
        "bridge_full_arrays_equal": bridges,
        "max_motion_difference": worst_motion,
        "max_body_gap_difference_m": worst_body,
        "max_expanded_gap_difference_m": worst_margin,
        "actual_arrivals": sum(r["reached"] for r in summary["rows"]),
        "safe_completed": sum(r["safe_completed"] for r in summary["rows"]),
        "false_arrivals": sum(r["false_arrival"] for r in summary["rows"]),
        "contacts": sum(r["contacts"] for r in summary["rows"]),
        "stopped": sum(r["physically_stopped"] for r in summary["rows"]),
        "no_motion_negative_records": 5,
        "source_live_and_snapshot_hashes": protocol["source_sha256"],
        "summary_sha256": sha(DATA / "summary.json"),
        "benchmark_sha256": sha(benchmark_path),
        "boundary": "Constant single-factor pose errors or FIFO command-intention delay, fresh sensors; <=1cm physical-corner travel sampling, padded geometry evaluated at the same poses, no continuous or hardware safety guarantee.",
        "visual_review": "Four plots and own Tk window inspected; grouped conditions, requested/applied braking, truth/estimate, camera, map and goals synchronized.",
        "images": images,
        "test_logs": logs,
        "local_links_checked": links,
        "documents_sha256": {str(p.relative_to(ROOT)): sha(p) for p in documents},
        "audit_sources_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                Path(__file__),
                ROOT / "scripts/validate_braking_navigation.py",
                ROOT / "scripts/validate_footprint_tracking.py",
            )
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
                    "swept_samples",
                    "actual_arrivals",
                    "safe_completed",
                    "contacts",
                    "stopped",
                    "wall_s",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
