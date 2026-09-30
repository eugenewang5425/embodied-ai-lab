"""Lesson 69 round two: registered 2x2 tracking/curve-retention experiment."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from embodied_learning.experiments import observed_navigation as observed
from embodied_learning.footprint_planning import FootprintController
from embodied_learning.navigation_geometry import NavigationRig, advance, rotation
from embodied_learning.path_tracking import (
    RetainedCurveController,
    RetainedTangentController,
    TangentController,
    path_frame,
    wrap,
)
from embodied_learning.replay_viewer.campus import read_checked

METHODS = {
    "original": FootprintController,
    "tangent": TangentController,
    "retained": RetainedCurveController,
    "combined": RetainedTangentController,
}
DEFAULT_OUTPUT = "results/footprint_tracking_v2"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def expanded_separation(pose, boxes, rig):
    """Independent 2D SAT separation for the expanded footprint; scoring only.

    A negative value indicates overlap, not an exact Euclidean penetration depth.
    Project obstacles that intersect robot height; this is not actual 3D contact.
    """
    boxes = boxes[(boxes[:, 5] > 0) & (boxes[:, 4] < rig.top_m)]
    if not len(boxes):
        return float("inf")
    r = rotation(pose[2])
    axes = np.vstack((np.eye(2), r.T))
    center = pose[:2] + r @ np.array([(rig.front_m - rig.rear_m) / 2, 0])
    half = np.array([(rig.front_m + rig.rear_m) / 2, rig.half_width_m]) + rig.margin_m
    centers = (boxes[:, [0, 2]] + boxes[:, [1, 3]]) / 2
    halves = (boxes[:, [1, 3]] - boxes[:, [0, 2]]) / 2
    gaps = np.abs((centers - center) @ axes.T) - (np.abs(axes @ r) @ half) - halves @ np.abs(axes).T
    return float(gaps.max(axis=1).min())


def diagnostics(data, prior_boxes, boxes, rig):
    first = data["plan"][0]
    first = first[np.isfinite(first).all(axis=1)]
    values = {
        k: []
        for k in (
            "tracking_cross_m",
            "tracking_heading_rad",
            "path_curvature_m_inv",
            "initial_cross_m",
            "initial_heading_rad",
            "expanded_gap_m",
            "prior_expanded_gap_m",
        )
    }
    swept = []
    for j, (pose, estimate, plan) in enumerate(
        zip(data["truth"], data["estimate"], data["plan"], strict=True)
    ):
        path = plan[np.isfinite(plan).all(axis=1)]
        current = path_frame(path, estimate) if len(path) > 1 else None
        original = path_frame(first, estimate)
        row = (
            current["cross_m"] if current else np.nan,
            wrap(estimate[2] - current["heading_rad"]) if current else np.nan,
            current["curvature_m_inv"] if current else np.nan,
            original["cross_m"],
            wrap(estimate[2] - original["heading_rad"]),
            expanded_separation(pose, boxes, rig),
            expanded_separation(pose, prior_boxes, rig),
        )
        for key, value in zip(values, row, strict=True):
            values[key].append(value)
        if j < len(data["truth"]) - 1:
            v, w = data["commands"][j]
            steps = max(1, math.ceil((abs(v) + rig.radius * abs(w)) * observed.DT / 0.01))
            swept.extend(
                expanded_separation(advance(pose, v, w, t), boxes, rig)
                for t in np.linspace(0, observed.DT, steps + 1)
            )
    return {k: np.asarray(v) for k, v in values.items()}, min(swept, default=None)


def run_episode(layout, obstacle, seed, method):
    controllers = []

    class Timed(METHODS[method]):
        def step(self, *args):
            start = time.perf_counter()
            result = super().step(*args)
            self.step_times.append(time.perf_counter() - start)
            return result

    def factory(*args):
        c = Timed(*args)
        c.step_times = []
        controllers.append(c)
        return c

    row, data, grid = observed.episode(layout, obstacle, "depth", seed, controller_factory=factory)
    before, boxes = observed.scene(layout, obstacle)
    rig = NavigationRig()
    c = controllers[0]
    diagnostic, minimum = diagnostics(data, before, boxes, rig)
    data.update(
        diagnostic,
        observed_map=grid,
        prior_map=observed.occupancy(before),
        actual_boxes=boxes,
        control_wall_s=np.asarray(c.step_times),
    )
    # One entry for each frame that produced nominal steering, including an
    # arrival frame where the experiment subsequently commands zero velocity.
    angular = np.full(len(data["truth"]), np.nan)
    if hasattr(c, "tracking_log"):
        active = data["reason"] != "no_observed_route"
        assert int(active.sum()) == len(c.tracking_log)
        angular[active] = [r["angular_raw"] for r in c.tracking_log]
    data["tracking_angular_raw"] = angular
    brake = data["reason"] == "swept_body_brake"
    row.update(
        method=method,
        expanded_swept_min_m=minimum,
        expanded_overlap_frames=int(np.sum(data["expanded_gap_m"] <= 0)),
        tracking_cross_p95_m=float(np.nanpercentile(np.abs(data["tracking_cross_m"]), 95)),
        initial_cross_max_m=float(np.max(np.abs(data["initial_cross_m"]))),
        initial_heading_max_rad=float(np.max(np.abs(data["initial_heading_rad"]))),
        braking_frames=int(brake.sum()),
        braking_events=int(np.sum(brake & ~np.r_[False, brake[:-1]])),
        planning_log=c.planning_log,
        planning_reasons=dict(Counter(r["reason"] for r in c.planning_log)),
        control_timing={
            name: float(value)
            for name, value in zip(
                ("p95_s", "p99_s", "max_s"),
                (*np.percentile(c.step_times, [95, 99]), max(c.step_times)),
                strict=True,
            )
        },
    )
    return row, data


def run(output=DEFAULT_OUTPUT, source="results/footprint_navigation_v1", smoke=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[3]
    files = [
        Path(__file__),
        *[
            root / "src/embodied_learning" / name
            for name in (
                "path_tracking.py",
                "footprint_planning.py",
                "navigation_geometry.py",
                "experiments/observed_navigation.py",
                "experiments/grid_nav.py",
            )
        ],
    ]
    for f in files:
        target = out / "source" / f.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f.read_bytes())
    source = Path(source)
    old = json.loads((source / "summary.json").read_text(encoding="utf8"))
    protocol = {
        "lesson": "69",
        "round": 2,
        "methods": list(METHODS),
        "design": "2x2: original/tangent tracking x shortcut/original-curve retention",
        "rig": NavigationRig().to_dict(),
        "dt_s": observed.DT,
        "tracking": {
            "heading_gain_per_s": 2.0,
            "cross_gain_per_s": 2.0,
            "normalizing_speed_m_s": 0.6,
            "speed_lookahead_m": 0.35,
            "max_v_m_s": 0.6,
            "max_w_rad_s": 1.0,
        },
        "retention_max_projection_distance_m": 0.05,
        "development_case": "narrow/crate/seed0 only; four archived probes before freeze",
        "fixed_seeds": [0, 1, 2],
        "heldout_narrow_seeds": [3, 4],
        "baseline_summary_sha256": digest(source / "summary.json"),
        "source_sha256": {str(f.relative_to(root)): digest(f) for f in files},
        "score_access": "truth and actual boxes used only after commands or for independent scoring; controller gets prior, observation and estimate",
        "expanded_score": "2D SAT separation of 6cm padded rectangle; <=1cm corner-travel sweep; not 3D collision or exact penetration depth",
        "timing": "wall per whole controller.step including observation, planning and safety; diagnostic SAT scoring outside controller; simulation waits for computation",
        "smoke": smoke,
    }
    cases = [
        (layout, obstacle, seed, "fixed")
        for layout in ("plaza", "street", "narrow")
        for obstacle in ("crate", "low", "beam")
        for seed in (0, 1, 2)
    ]
    cases += [
        ("narrow", obstacle, seed, "heldout")
        for obstacle in ("crate", "low", "beam")
        for seed in (3, 4)
    ]
    if smoke:
        cases = [("narrow", "crate", 0, "development")]
    save_json(out / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    started = time.perf_counter()
    for layout, obstacle, seed, cohort in cases:
        for method in METHODS:
            tick = time.perf_counter()
            row, data = run_episode(layout, obstacle, seed, method)
            if method == "original" and seed in (0, 1, 2):
                old_row = next(
                    r
                    for r in old["rows"]
                    if r["method"] == "rectangle"
                    and (r["layout"], r["obstacle"], r["seed"]) == (layout, obstacle, seed)
                )
                original = read_checked(source / old_row["file"], old_row["sha256"])
                for key, a in original.items():
                    assert np.array_equal(a, data[key], equal_nan=a.dtype.kind not in "US"), key
                row["baseline_equal"] = True
                row["baseline_sha256"] = old_row["sha256"]
            name = f"{layout}_{obstacle}_s{seed}_{method}.npz"
            np.savez_compressed(out / name, **data)
            row.update(
                file=name,
                sha256=digest(out / name),
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
                f"{row['wall_s']:.2f}s",
                flush=True,
            )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--source", default="results/footprint_navigation_v1")
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    run(a.output, a.source, a.smoke)


if __name__ == "__main__":
    main()
