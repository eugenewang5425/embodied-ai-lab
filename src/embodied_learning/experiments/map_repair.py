"""Lesson 68: estimate wheel scales from scans and rebuild a frozen distorted map.

Estimator input is exclusively odometry and raw ranges/hits. Ground truth is
accepted only by the separate evaluation routine. Scope: constant wheel-scale
errors in a static scene; this is not an unrestricted lifelong-SLAM solution.
"""

import argparse
import hashlib
import json
import math
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from embodied_learning.experiments.campus_patrol import (
    RES,
    SIZE,
    Config,
    run_episode,
    world,
)
from embodied_learning.experiments.grid_nav import Obstacle, cast_rays, inflate
from embodied_learning.experiments.map_localization import occupancy_from_observations


def wrap(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))


def transform(points, pose):
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return points @ np.array([[c, s], [-s, c]]) + pose[:2]


def relative(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.r_[np.array([[c, s], [-s, c]]) @ (b[:2] - a[:2]), wrap(b[2] - a[2])]


def compose(a, b):
    return np.r_[transform(b[None, :2], a)[0], wrap(a[2] + b[2])]


def cloud(ranges, hits):
    angles = np.arange(len(ranges)) * (2 * np.pi / len(ranges))
    points = ranges[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    return points, np.asarray(hits, bool)


def match_scans(a_range, a_hit, b_range, b_hit, initial):
    """Robust point-to-segment ICP; reference segments require adjacent returns."""
    reference, ah = cloud(a_range, a_hit)
    source, bh = cloud(b_range, b_hit)
    source = source[bh]
    end = np.roll(reference, -1, axis=0)
    mask = ah & np.roll(ah, -1) & (np.linalg.norm(end - reference, axis=1) < 1.3)
    starts, ends = reference[mask], end[mask]
    if len(starts) < 8 or len(source) < 12:
        return initial.copy(), {"accepted": False, "reason": "few_returns"}
    vectors = ends - starts
    norm = np.maximum(np.sum(vectors**2, axis=1), 1e-12)

    def differences(pose):
        moved = transform(source, pose)
        delta = moved[:, None, :] - starts
        fraction = np.clip(np.sum(delta * vectors, axis=2) / norm, 0, 1)
        diff = delta - fraction[:, :, None] * vectors
        closest = np.argmin(np.sum(diff**2, axis=2), axis=1)
        residual = diff[np.arange(len(diff)), closest]
        # Point-to-line: endpoint clipping otherwise pulls motion estimates toward
        # the visible end of a wall, giving a systematic translation shrinkage.
        normal = np.column_stack((-vectors[:, 1], vectors[:, 0])) / np.sqrt(norm)[:, None]
        chosen = normal[closest]
        residual = (residual * chosen).sum(axis=1)[:, None] * chosen
        # Far points are occlusion/outliers; cap their influence continuously.
        length = np.linalg.norm(residual, axis=1)
        return residual * np.minimum(1, 0.4 / np.maximum(length, 1e-9))[:, None]

    result = least_squares(
        lambda p: differences(p).ravel(),
        initial,
        bounds=(initial - [0.35, 0.35, 0.15], initial + [0.35, 0.35, 0.15]),
        loss="soft_l1",
        f_scale=0.025,
        max_nfev=24,
        diff_step=1e-4,
    )
    distances = np.linalg.norm(differences(result.x), axis=1)
    s = np.linalg.svd(result.jac, compute_uv=False)
    inliers = distances < 0.10
    rms = float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else 1.0
    accepted = bool(inliers.mean() >= 0.65 and rms < 0.065 and s[-1] > 0.5)
    return result.x, {
        "accepted": accepted,
        "reason": "ok" if accepted else "ambiguous_or_mismatch",
        "inlier_fraction": float(inliers.mean()),
        "rms_m": rms,
        "minimum_singular_value": float(s[-1]),
    }


def integrate_scales(odom, scale_t, scale_r):
    poses = [odom[0].copy()]
    for a, b in pairwise(odom):
        increment = relative(a, b)
        dtheta = increment[2]
        distance = np.linalg.norm(increment[:2])
        # Differential-drive records integrate translation at the midpoint heading.
        signed = np.sign(increment[0]) * distance
        newtheta = dtheta * scale_r
        local = np.array(
            [
                signed * scale_t * np.cos(newtheta / 2),
                signed * scale_t * np.sin(newtheta / 2),
                newtheta,
            ]
        )
        poses.append(compose(poses[-1], local))
    return np.array(poses)


def repair(odom, ranges, hits):
    """No truth/map access. Estimate two persistent scales from observable scan pairs."""
    ids = [0]
    for i in range(1, len(odom)):
        d = relative(odom[ids[-1]], odom[i])
        if np.linalg.norm(d[:2]) > 0.25 or abs(d[2]) > 0.14:
            ids.append(i)
    fitted, predicted, matches = [], [], []
    for a, b in pairwise(ids):
        initial = relative(odom[a], odom[b])
        estimate, info = match_scans(ranges[a], hits[a], ranges[b], hits[b], initial)
        info.update(a=a, b=b, predicted=initial.tolist(), measured=estimate.tolist())
        matches.append(info)
        if info["accepted"]:
            predicted.append(initial)
            fitted.append(estimate)
    predicted, fitted = np.asarray(predicted), np.asarray(fitted)
    if len(predicted) < 10:
        raise ValueError("too few observable scan constraints for wheel-scale calibration")
    translation = np.linalg.norm(predicted[:, :2], axis=1) > 0.20
    turn = np.abs(predicted[:, 2]) > 0.10
    if translation.sum() < 8 or turn.sum() < 5:
        raise ValueError("translation/rotation scales are not both observable")

    def residual(scales):
        trans = (
            np.linalg.norm(fitted[translation, :2], axis=1)
            - scales[0] * np.linalg.norm(predicted[translation, :2], axis=1)
        ) / 0.02
        yaw = wrap(fitted[turn, 2] - scales[1] * predicted[turn, 2]) / 0.015
        return np.r_[trans, yaw]

    fit = least_squares(residual, [1.0, 1.0], bounds=([0.8, 0.8], [1.2, 1.2]), loss="soft_l1")
    corrected = integrate_scales(odom, *fit.x)
    # Best rigid transform available from the SAME sensor-derived corrected chain.
    # This control cannot change internal shape, even with this favorable target.
    a, b = odom[:, :2], corrected[:, :2]
    u, _, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
    rm = vt.T @ np.diag([1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    yaw = np.arctan2(rm[1, 0], rm[0, 0])
    rigid = np.column_stack(((a - a.mean(0)) @ rm.T + b.mean(0), wrap(odom[:, 2] + yaw)))
    return (
        corrected,
        rigid,
        {
            "translation_scale": float(fit.x[0]),
            "rotation_scale": float(fit.x[1]),
            "accepted": len(predicted),
            "tested": len(matches),
            "translation_constraints": int(translation.sum()),
            "rotation_constraints": int(turn.sum()),
            "matches": matches,
        },
    )


def map_metrics(candidate, reference, tolerance=0.20):
    a, b = np.argwhere(candidate) * RES, np.argwhere(reference) * RES
    if not len(a) or not len(b):
        return {"precision": 0.0, "recall": 0.0, "symmetric_mean_m": None}
    da, db = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    return {
        "precision": float(np.mean(da <= tolerance)),
        "recall": float(np.mean(db <= tolerance)),
        "symmetric_mean_m": float((da.mean() + db.mean()) / 2),
    }


def observability_probe():
    scenes = {
        "long_corridor": [Obstacle("rect", 0, 2, 100, 0.1), Obstacle("rect", 0, -2, 100, 0.1)],
        "closed_room": [
            Obstacle("rect", 3, 0, 0.1, 3),
            Obstacle("rect", -3, 0, 0.1, 3),
            Obstacle("rect", 0, 3, 3, 0.1),
            Obstacle("rect", 0, -3, 3, 0.1),
        ],
        "street_corner": [
            Obstacle("rect", 3, 0, 0.1, 3),
            Obstacle("rect", 0, 3, 3, 0.1),
            Obstacle("circle", 1, 1, r=0.4),
        ],
    }
    target = np.array([0.25, 0.04, 0.06])
    angles = np.arange(64) * 2 * np.pi / 64
    rows = []
    for name, geometry in scenes.items():
        for noise in (0.0, 0.03):
            rng = np.random.default_rng(6800)
            ar, ah = cast_rays(0, 0, angles, geometry, (), 8.0)
            br, bh = cast_rays(*target[:2], angles + target[2], geometry, (), 8.0)
            ar += rng.normal(0, noise, len(ar)) * ah
            br += rng.normal(0, noise, len(br)) * bh
            estimate, info = match_scans(ar, ah, br, bh, target + [0.03, -0.02, 0.01])
            rows.append(
                {
                    "environment": name,
                    "noise_sigma_m": noise,
                    "translation_error_m": float(np.linalg.norm(estimate[:2] - target[:2])),
                    "yaw_error_deg": float(np.degrees(abs(wrap(estimate[2] - target[2])))),
                    **info,
                }
            )
    return rows


def run(output, source="results/campus_patrol_v1", seeds=(0, 1, 2)):
    out, source = Path(output), Path(source)
    out.mkdir(parents=True, exist_ok=False)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol = {
        "experiment": "map_repair_lesson68",
        "input_sha256": digest(source / "mapping.npz"),
        "input": str(source),
        "source_sha256": digest(__file__),
        "dependency_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in ("campus_patrol.py", "map_localization.py", "grid_nav.py")
        },
        "map_res_m": RES,
        "seeds": list(seeds),
        "estimator_access": ["odom", "ranges", "hits"],
        "repair": "offline scan-derived constant translation/rotation scale correction",
        "query": "independent closed-loop traversal; frozen map; each arm plans on its own map",
        "controller": "lesson66 fixed planner/tracker/front stop; not lesson67 depth avoidance",
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8"
    )
    data = dict(np.load(source / "mapping.npz"))
    started = time.perf_counter()
    corrected, rigid, estimate = repair(data["odom"], data["ranges"], data["hits"])
    print("estimated scales", estimate["translation_scale"], estimate["rotation_scale"], flush=True)
    base = SimpleNamespace(rays=data["ranges"].shape[1], res_m=RES, grid_cells=int(SIZE / RES))
    maps, quality = {}, {}
    geometry, styles, ideal = world()
    projected = occupancy_from_observations(
        data["ranges"], data["hits"], data["truth"], range(len(corrected)), base
    ).astype(bool)
    for key, poses in (("raw", data["odom"]), ("rigid", rigid), ("repaired", corrected)):
        maps[key] = occupancy_from_observations(
            data["ranges"], data["hits"], poses, range(len(poses)), base
        ).astype(bool)
        error = np.linalg.norm(poses[:, :2] - data["truth"][:, :2], axis=1)
        quality[key] = {
            "pose_mean_m": float(error.mean()),
            "pose_p95_m": float(np.percentile(error, 95)),
            **map_metrics(maps[key], projected),
        }
    maps["reference"] = ideal
    np.savez_compressed(out / "maps.npz", **maps, projected=projected)
    np.savez_compressed(out / "repair.npz", corrected=corrected, rigid=rigid, **data)
    (out / "constraints.json").write_text(
        json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf8"
    )
    rows = []
    observability = observability_probe()
    for seed in seeds:
        for key, grid in maps.items():
            safe = ~inflate(grid, math.ceil((0.25 + 0.10) / RES))
            row, record = run_episode("self", seed, geometry, safe, grid, Config())
            row["method"] = key
            file = f"seed{seed}_{key}.npz"
            np.savez_compressed(out / file, **record)
            row.update(file=file, sha256=digest(out / file))
            rows.append(row)
            print(seed, key, row["status"], row["reached"], round(row["mean_m"], 3), flush=True)
            summary = {
                "protocol": protocol,
                "protocol_sha256": digest(out / "protocol.json"),
                "quality": quality,
                "observability_probe": observability,
                "calibration": {k: v for k, v in estimate.items() if k != "matches"},
                "rows": rows,
                "wall_s": time.perf_counter() - started,
                "maps_sha256": digest(out / "maps.npz"),
                "repair_sha256": digest(out / "repair.npz"),
                "geometry": [asdict_box(o) for o in geometry],
                "styles": styles,
            }
            (out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
            )
    return summary


def asdict_box(obstacle):
    from dataclasses import asdict

    return asdict(obstacle)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/map_repair_v1")
    p.add_argument("--source", default="results/campus_patrol_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = p.parse_args()
    run(a.output, a.source, a.seeds)


if __name__ == "__main__":
    main()
