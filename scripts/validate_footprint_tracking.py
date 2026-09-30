"""Independent audit of frozen lesson 69 round-two motion, geometry and evidence."""

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/footprint_tracking_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def arrays(path, expected):
    assert sha(path) == expected, path
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def corner_separation(poses, boxes, rig):
    """Project actual corner sets onto four SAT axes; no runtime scorer calls."""
    margin = rig["margin_m"]
    corners = np.array(
        [
            [x, y]
            for x in (-rig["rear_m"] - margin, rig["front_m"] + margin)
            for y in (-rig["half_width_m"] - margin, rig["half_width_m"] + margin)
        ]
    )
    c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    rotation = np.stack((c, -s, s, c), axis=-1).reshape(-1, 2, 2)
    body = np.einsum("nij,kj->nki", rotation, corners) + poses[:, None, :2]
    axes = np.concatenate(
        (np.tile(np.eye(2), (len(poses), 1, 1)), rotation.transpose(0, 2, 1)), axis=1
    )
    bp = np.einsum("nki,nai->nka", body, axes)
    gap = np.full(len(poses), np.inf)
    for x0, x1, y0, y1, z0, z1 in boxes:
        if z1 <= 0 or z0 >= rig["top_m"]:
            continue
        obstacle = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)])
        op = np.einsum("ki,nai->nka", obstacle, axes)
        separated = np.maximum(op.min(axis=1) - bp.max(axis=1), bp.min(axis=1) - op.max(axis=1))
        gap = np.minimum(gap, separated.max(axis=1))
    return gap


def main():
    started = time.perf_counter()
    summary, protocol = read(DATA / "summary.json"), read(DATA / "protocol.json")
    benchmark = ROOT / "docs/benchmarks/footprint-tracking-v2.json"
    assert sha(DATA / "summary.json") == read(benchmark)["summary_sha256"]
    assert sha(DATA / "protocol.json") == summary["protocol_sha256"]
    assert protocol == summary["protocol"] and not protocol["smoke"]
    for name, digest in protocol["source_sha256"].items():
        assert sha(DATA / "source" / name) == sha(ROOT / name) == digest, name
    old_dir = ROOT / "results/footprint_navigation_v1"
    assert sha(old_dir / "summary.json") == protocol["baseline_summary_sha256"]
    old = {
        (r["layout"], r["obstacle"], r["seed"]): r
        for r in read(old_dir / "summary.json")["rows"]
        if r["method"] == "rectangle"
    }
    rows = summary["rows"]
    assert len(rows) == 132
    assert Counter(r["cohort"] for r in rows) == {"fixed": 108, "heldout": 24}
    plans = defaultdict(dict)
    baselines, frames, steps, successes = 0, 0, 0, 0
    max_motion_error, max_gap_error = 0.0, 0.0
    for row in rows:
        data = arrays(DATA / row["file"], row["sha256"])
        pose, cmd = data["truth"], data["commands"]
        n, dt = len(pose), protocol["dt_s"]
        assert n == row["frames"] == len(cmd) == len(data["control_wall_s"])
        assert np.array_equal(pose, data["estimate"])
        assert np.array_equal(cmd[-1], [0, 0])  # terminal marker, not an executed stop
        outgoing = cmd[:-1]
        delta = np.diff(np.vstack(([0, 0], outgoing)), axis=0)
        assert np.all(np.abs(delta) <= np.array([0.8, 1.8]) * dt + 1e-12)
        assert np.all(np.abs(outgoing) <= [0.6 + 1e-12, 1.0 + 1e-12])
        # Analytic midpoint/sinc form, independently of the runtime advance().
        turn = outgoing[:, 1] * dt
        heading = pose[:-1, 2] + turn / 2
        travel = outgoing[:, 0] * dt * np.sinc(turn / (2 * np.pi))
        predicted_xy = pose[:-1, :2] + travel[:, None] * np.c_[np.cos(heading), np.sin(heading)]
        difference = pose[1:, 2] - pose[:-1, 2] - turn
        errors = [np.max(np.abs(predicted_xy - pose[1:, :2])), np.max(np.abs(np.sin(difference)))]
        max_motion_error = max(max_motion_error, *errors)
        assert max(errors) < 1e-10, row["file"]
        assert np.all(np.cos(difference) > 0)
        arrival = np.linalg.norm(pose[-1, :2] - data["goal"][-1]) < 0.25 and abs(cmd[-2, 0]) < 0.05
        assert bool(row["reached"]) == bool(arrival) == (row["status"] == "completed")
        assert row["announced"] == row["reached"] and row["contacts"] == 0
        assert row["min_clearance_m"] > 0
        gap = corner_separation(pose, data["actual_boxes"], protocol["rig"])
        error = float(np.max(np.abs(gap - data["expanded_gap_m"])))
        max_gap_error = max(max_gap_error, error)
        assert error < 1e-10
        assert row["expanded_overlap_frames"] == int((gap <= 0).sum())
        assert row["expanded_swept_min_m"] <= gap.min() + 1e-10
        case = (row["layout"], row["obstacle"], row["seed"])
        assert row["method"] not in plans[case]
        plans[case][row["method"]] = data["plan"][0]
        if row["method"] == "original" and row["cohort"] == "fixed":
            r = old[case]
            baseline = arrays(old_dir / r["file"], r["sha256"])
            for name, value in baseline.items():
                assert np.array_equal(value, data[name], equal_nan=value.dtype.kind not in "US"), (
                    name
                )
            baselines += 1
        frames += n
        steps += n - 1
        successes += row["reached"]
    assert len(plans) == 33 and baselines == 27
    for case, methods in plans.items():
        assert set(methods) == set(protocol["methods"])
        assert all(
            np.array_equal(p, methods["original"], equal_nan=True) for p in methods.values()
        ), case
    images = {}
    for path in sorted((ROOT / "docs/img").glob("lesson69-round2-*.png")):
        with Image.open(path) as im:
            assert im.width >= 1200 and im.height >= 600
            images[path.name] = {"sha256": sha(path), "size": im.size}
    assert len(images) == 4
    logs = {}
    for name, expected in (
        ("quick", "832 passed, 185 deselected"),
        ("affected_initial", "1 failed, 51 passed"),
        ("affected", "52 passed"),
    ):
        p = ROOT / f"results/footprint_tracking_{name}.log"
        text = p.read_text(encoding="utf8", errors="replace")
        assert expected in text
        code = 1 if name == "affected_initial" else 0
        assert f"exit={code}" in text
        logs[name] = {
            "sha256": sha(p),
            "result": expected,
            "exit": code,
            "wrapper": text.splitlines()[-1],
        }
    output = ROOT / "docs/benchmarks/footprint-tracking-v2-validation.json"
    documents = [ROOT / "README.md"] + [
        ROOT / "docs" / name
        for name in (
            "69-footprint-planning.md",
            "69-footprint-tracking-round2.md",
            "69-tracking-round2-plan.md",
            "navigation-stage-review-69-71.md",
            "01-learning-roadmap.md",
            "multisensor-navigation-roadmap.md",
            "replay-window.md",
        )
    ]
    links = 0
    for p in documents:
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf8")):
            if "://" in target or target.startswith("#"):
                continue
            resolved = (p.parent / target.split("#")[0]).resolve()
            assert resolved.exists() or resolved == output, (p, target)
            links += 1
    result = {
        "date": "2026-09-30",
        "records": len(rows),
        "frames": frames,
        "executed_steps": steps,
        "true_arrivals": successes,
        "bitwise_legacy_baselines": baselines,
        "identical_four_arm_initial_plan_cases": len(plans),
        "max_independent_motion_error": max_motion_error,
        "max_independent_corner_projection_gap_error_m": max_gap_error,
        "source_live_and_snapshot_hashes_verified": protocol["source_sha256"],
        "summary_sha256": sha(DATA / "summary.json"),
        "benchmark_sha256": sha(benchmark),
        "images": images,
        "visual_review": "Three charts and actual own-window capture reviewed; labels, margins, frame, target and source synchronization checked.",
        "logs": logs,
        "retry_reason": "New fifth diagnostic tab was reset by source selection; product choose_view now preserves tabs 3 and 4. Initial failure retained.",
        "regression_scope": "832 quick and 52 affected complete selection. No new whole-repository slow run; unchanged lesson53 dense solver excluded. Historical whole run evidence remains separate.",
        "terminal_boundary": "Last zero command is a non-executed archive marker. No-path termination does not establish a safe physical stop after abort.",
        "geometry_boundary": "Corner SAT verified at saved frames; saved swept minimum uses <=1cm corner-travel sampling, not continuous safety proof. No runtime controllers used by this audit.",
        "local_links_checked": links,
        "documents_sha256": {str(p.relative_to(ROOT)): sha(p) for p in documents},
        "audit_sha256": sha(Path(__file__)),
        "audit_wall_s": time.perf_counter() - started,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "records",
                    "frames",
                    "executed_steps",
                    "true_arrivals",
                    "local_links_checked",
                    "audit_wall_s",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
