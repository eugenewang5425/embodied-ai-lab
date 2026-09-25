"""Controlled algorithm transfer into a photo-referenced MuJoCo 3D room.

The room is an uncalibrated proxy. Motion is a prescribed differential-drive
kinematic replay with MuJoCo contact checks, not a closed-loop dynamics test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import time
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from embodied_learning.experiments.grid_nav import (
    GEOMETRY,
    astar,
    inflate,
    smooth_path,
)
from embodied_learning.experiments.map_localization import (
    ParticleFilter,
    occupancy_from_observations,
    scan_map_alignment,
)
from embodied_learning.odometry import estimate_poses
from embodied_learning.photo_room import RoomWorld, default_layout

RES = 0.04


def plan(grid, start, goal, radius=0.15):
    safe = ~inflate(grid, math.ceil((radius + RES) / RES))
    s = tuple(np.floor(np.asarray(start)[::-1] / RES).astype(int))
    g = tuple(np.floor(np.asarray(goal)[::-1] / RES).astype(int))
    cells, _ = astar(safe, s, g)
    if cells is None:
        return None
    cells = smooth_path(cells, safe)
    return np.array([[(x + 0.5) * RES, (y + 0.5) * RES] for y, x in cells])


def drive_replay(points):
    """Rotate then translate with exact wheel increments; no slip model."""
    heading = math.atan2(*(points[1] - points[0])[::-1])
    increments = []
    for a, b in pairwise(points):
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            continue
        target = math.atan2(vec[1], vec[0])
        rot = math.atan2(math.sin(target - heading), math.cos(target - heading))
        turns = max(1, math.ceil(abs(rot) / 0.10))
        if abs(rot) > 1e-9:
            dw = rot / turns * GEOMETRY.track_m / (2 * GEOMETRY.radius_m)
            increments.extend([[-dw, dw]] * turns)
        steps = max(1, math.ceil(length / 0.06))
        dw = length / steps / GEOMETRY.radius_m
        increments.extend([[dw, dw]] * steps)
        heading = target
    encoders = np.vstack((np.zeros(2), np.cumsum(increments, axis=0)))
    init = [*points[0], math.atan2(*(points[1] - points[0])[::-1])]
    return estimate_poses(encoders, init), encoders


def path_contacts(world, points):
    if points is None:
        return {
            "path_exists": False,
            "contact_frames": 0,
            "sampled_frames": 0,
            "contact_objects": [],
        }
    poses, _ = drive_replay(points)
    names = set()
    count = 0
    for pose in poses:
        hit = world.contact_names(pose)
        count += bool(hit)
        names.update(hit)
    return {
        "path_exists": True,
        "contact_frames": count,
        "sampled_frames": len(poses),
        "contact_objects": sorted(names),
    }


def metrics(estimate, truth):
    errors = np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1)
    return {
        "mean_m": float(errors.mean()),
        "p95_m": float(np.percentile(errors, 95)),
        "end_m": float(errors[-1]),
    }


def run(output, layout, seeds=(0, 1, 2), particles=200, log=print):
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not 50 <= particles <= 2000:
        raise ValueError("particles must be in [50, 2000]")
    started = time.perf_counter()
    out = Path(output)
    if (out / "summary.json").exists():
        raise FileExistsError(f"completed run exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    world = RoomWorld(layout)
    robot = layout["robot"]
    z = robot["lidar_z"]
    scan_grid = world.occupancy(z, z, RES)
    body_grid = world.occupancy(robot["bottom"], robot["bottom"] + robot["height"], RES)
    layout_json = json.dumps(layout, ensure_ascii=False, sort_keys=True)
    protocol = {
        "experiment": "photo_room_3d_transfer_v1",
        "seeds": list(seeds),
        "particles": particles,
        "resolution_m": RES,
        "range_noise_sigma_m": 0.01,
        "encoder_right_bias": 0.02,
        "control": "truth-controlled rotate/translate kinematic replay; MuJoCo contacts checked; no dynamic wheel simulation",
        "sensor": "64 horizontal MuJoCo mj_ray rays at 0.18m, max range 4m; opaque geometric surfaces",
        "mapping": "first reference lap; identical scans projected with true or biased odometry poses",
        "localization": "two fresh replay laps, known initial pose shared by every method; no truth-fed resets",
        "evidence_scope": "one photo-referenced estimated geometry, three sensor seeds are NOT three independent rooms",
        "gates": {"reference_route_contact_free": True, "PF_improves_mean_and_p95_all_seeds": True},
        "layout_sha256": hashlib.sha256(layout_json.encode()).hexdigest(),
        "scene_code_sha256": hashlib.sha256(
            Path(inspect.getfile(RoomWorld)).read_bytes()
        ).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
    }
    (out / "plan.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # All tasks are declared before evaluating their contact outcome.
    tasks = {
        "under_bed": ([2.7, 0.95], [1.50, 3.00]),
        "central_aisle": ([2.7, 0.95], [1.60, 1.92]),
        "under_desk": ([2.7, 0.95], [0.34, 1.55]),
    }
    planning = {}
    for name, (start, goal) in tasks.items():
        planning[name] = {}
        for label, grid in (("low_scan_slice", scan_grid), ("robot_height_volume", body_grid)):
            points = plan(grid, start, goal, robot["radius"])
            planning[name][label] = path_contacts(world, points)
            if points is not None:
                planning[name][label]["path_xy"] = points.tolist()
    waypoints = [[2.7, 0.95], [2.45, 1.93], [1.60, 1.93], [1.60, 1.0], [2.7, 0.95]]
    paths = [plan(body_grid, a, b, robot["radius"]) for a, b in pairwise(waypoints)]
    if any(p is None for p in paths):
        raise RuntimeError("predeclared reference route is not feasible in this layout")
    loop = np.vstack([paths[0]] + [p[1:] for p in paths[1:]])
    contact_check = path_contacts(world, loop)
    if contact_check["contact_frames"]:
        raise RuntimeError(f"reference route collided: {contact_check}")
    map_truth, map_enc = drive_replay(loop)
    map_odom = estimate_poses(map_enc * np.array([1.0, 1.02]), map_truth[0])
    loc_points = np.vstack([loop, loop[1:]])
    truth, enc = drive_replay(loc_points)
    odom = estimate_poses(enc * np.array([1.0, 1.02]), truth[0])
    raw = []
    hits = []
    for pose in map_truth:
        r, h = world.scan(pose)
        raw.append(r)
        hits.append(h)
    raw = np.array(raw)
    hits = np.array(hits)
    base = SimpleNamespace(rays=64, res_m=RES, grid_cells=scan_grid.shape[0])
    observed_truth = occupancy_from_observations(raw, hits, map_truth, range(len(raw)), base)
    observed_odom = occupancy_from_observations(raw, hits, map_odom, range(len(raw)), base)
    maps = {
        "scan_height_oracle": scan_grid,
        "body_projection_oracle": body_grid,
        "same_scan_truth_map": observed_truth,
        "selfbuilt_odom_map": observed_odom,
    }
    clean = np.array([world.scan(p)[0] for p in truth])
    rows = []
    archive = {
        "truth": truth,
        "odom": odom,
        "encoder_angles": enc,
        "map_truth": map_truth,
        "map_odom": map_odom,
        "map_ranges": raw,
        "map_hits": hits,
        "clean_ranges": clean,
        **{f"grid_{k}": v for k, v in maps.items()},
    }
    deltas = np.diff(enc * np.array([1.0, 1.02]), axis=0)
    body_deltas = np.array([GEOMETRY.body_velocity(d) for d in deltas])
    log(
        f"Reference route: {len(truth)} localization frames, contact-free; starting {len(seeds)} seeds."
    )
    for seed in seeds:
        rng = np.random.default_rng(seed)
        observed = np.where(
            clean < 4, np.clip(clean + rng.normal(0, 0.01, clean.shape), 0.02, 3.99), 4.0
        )
        archive[f"observed_seed{seed}"] = observed
        row = {"seed": seed, "odometry": metrics(odom, truth)}
        for name, grid in maps.items():
            pf = ParticleFilter(
                grid,
                RES,
                particles,
                np.random.default_rng(seed + 1000),
                init_pose=truth[0],
                spread=0.04,
                rays_n=32,
            )
            estimates = []
            for k, obs in enumerate(observed):
                if k:
                    ds, dt = body_deltas[k - 1]
                    pf.propagate_body(ds, dt, dt=1.0)
                pf.measure(obs, None)
                estimates.append(pf.estimate())
                pf.resample()
            estimates = np.array(estimates)
            row[name] = metrics(estimates, truth)
            archive[f"{name}_seed{seed}"] = estimates
        rows.append(row)
        log(
            f"seed {seed}: odom={row['odometry']['mean_m']:.3f}m; matched PF={row['scan_height_oracle']['mean_m']:.3f}m; selfmap PF={row['selfbuilt_odom_map']['mean_m']:.3f}m"
        )
    verdicts = {
        name: all(
            r[name]["mean_m"] < r["odometry"]["mean_m"]
            and r[name]["p95_m"] < r["odometry"]["p95_m"]
            for r in rows
        )
        for name in maps
    }
    np.savez_compressed(out / "trajectories.npz", **archive)
    sensitivity = []
    for clearance in (0.18, 0.22, 0.40):
        variant = copy.deepcopy(layout)
        variant["bed"]["under_clearance"] = clearance
        variant_world = RoomWorld(variant)
        for lidar_z in (0.12, 0.18, 0.30):
            grid = variant_world.occupancy(lidar_z, lidar_z, RES)
            path = plan(grid, *tasks["under_bed"], robot["radius"])
            sensitivity.append(
                {
                    "bed_clearance_m": clearance,
                    "lidar_height_m": lidar_z,
                    **path_contacts(variant_world, path),
                }
            )
    report = {
        "schema_version": 1,
        "protocol": protocol,
        "layout": layout,
        "planning": planning,
        "height_sensitivity": sensitivity,
        "reference_route_contact_check": contact_check,
        "rows": rows,
        "map_alignment": scan_map_alignment(observed_odom, observed_truth, RES, tolerance_cells=2),
        "PF_improves_mean_and_p95_all_seeds": verdicts,
        "wall_time_s": time.perf_counter() - started,
        "trajectories_sha256": hashlib.sha256((out / "trajectories.npz").read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "preview_path.json").write_text(json.dumps(loop.tolist()), encoding="utf-8")
    save_plots(out, report, archive)
    save_sensor_images(out, world, truth[0])
    return report


def save_plots(out, report, archive):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "odometry": "#c44740",
        "scan_height_oracle": "#007c83",
        "body_projection_oracle": "#a77720",
        "same_scan_truth_map": "#6862ab",
        "selfbuilt_odom_map": "#337bbb",
    }
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    grid = archive["grid_body_projection_oracle"]
    ax.imshow(
        grid,
        origin="lower",
        extent=[0, grid.shape[1] * RES, 0, grid.shape[0] * RES],
        cmap="Greys",
        alpha=0.22,
    )
    ax.plot(
        *archive["truth"][:, :2].T, color="black", lw=2, label="Reference route (truth controlled)"
    )
    preview_seed = report["rows"][0]["seed"]
    for name in ("odometry", "scan_height_oracle", "selfbuilt_odom_map"):
        arr = archive["odom"] if name == "odometry" else archive[f"{name}_seed{preview_seed}"]
        ax.plot(*arr[:, :2].T, color=colors[name], lw=1.2, label=name.replace("_", " "))
    ax.set(
        xlabel="x (m)",
        ylabel="y (m)",
        title=f"3D room replay | sensor seed {preview_seed} | estimated room dimensions",
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    names = list(colors)
    x = np.arange(len(report["rows"]))
    for ax, metric in zip(axes, ["mean_m", "p95_m"]):
        for j, name in enumerate(names):
            ax.bar(
                x + (j - 2) * 0.15,
                [r[name][metric] for r in report["rows"]],
                width=0.14,
                color=colors[name],
                label=name.replace("_", " "),
            )
        ax.set_xticks(x, [f"sensor seed {r['seed']}" for r in report["rows"]])
        ax.set_ylabel("Mean error (m)" if metric == "mean_m" else "P95 error (m)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1), fontsize=9)
    fig.savefig(out / "localization_metrics.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["grid_scan_height_oracle", "grid_body_projection_oracle"],
        ["LiDAR plane z=0.18 m", "Robot volume z=0.02..0.34 m"],
    ):
        grid = archive[key]
        ax.imshow(
            grid,
            origin="lower",
            extent=[0, grid.shape[1] * RES, 0, grid.shape[0] * RES],
            cmap="Greys",
        )
        label = "low_scan_slice" if key == "grid_scan_height_oracle" else "robot_height_volume"
        row = report["planning"]["under_bed"][label]
        if row["path_exists"]:
            p = np.array(row["path_xy"])
            ax.plot(*p.T, color="#d24e3d", lw=2)
        ax.scatter(1.5, 3.0, marker="x", s=60, color="#d24e3d")
        ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
        ax.text(
            0.02,
            0.02,
            f"Path: {row['path_exists']} | contact frames: {row['contact_frames']}",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
            fontsize=9,
        )
    fig.savefig(out / "height_planning.png", dpi=160)
    plt.close(fig)


def save_sensor_images(out, world, pose):
    from PIL import Image

    world.set_pose(pose)
    with mujoco.Renderer(world.model, height=480, width=640) as renderer:
        renderer.update_scene(world.data, camera="robot_rgbd")
        Image.fromarray(renderer.render()).save(out / "robot_rgb.png")
        renderer.enable_depth_rendering()
        renderer.update_scene(world.data, camera="robot_rgbd")
        depth = renderer.render().copy()
        np.save(out / "robot_depth_m.npy", depth)
        # Deterministic scientific depth visualization, not a generated photo.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        im = ax.imshow(depth, vmin=0, vmax=4, cmap="viridis")
        ax.set_title("Rendered camera depth (m) | estimated room geometry")
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Depth (m)")
        fig.savefig(out / "robot_depth.png", dpi=130)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--output", default="results/photo_room_v1/validation_v3")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--particles", type=int, default=200)
    args = parser.parse_args()
    layout = (
        json.loads(args.layout.read_text(encoding="utf-8")) if args.layout else default_layout()
    )
    run(args.output, layout, tuple(int(s) for s in args.seeds.split(",")), args.particles)


if __name__ == "__main__":
    main()
