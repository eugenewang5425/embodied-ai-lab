"""Lesson 67 calibration fixtures: metric perception versus swept body collision."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from embodied_learning.navigation_geometry import (
    NavigationRig,
    collision_clearance,
    depth_points,
    lidar_obstacle_points,
    point_clearance,
    project,
    sense,
    stop_path,
    world_points,
)


def fixtures():
    # name, obstacle xyz limits, intended (v,w); no controller sees these boxes.
    return [
        ("front_wall", [0.70, 0.80, -1, 1, 0, 1.4], (0.8, 0)),
        ("low_curb", [0.62, 0.80, -0.65, 0.65, 0, 0.14], (0.6, 0)),
        ("mast_bar", [0.52, 0.68, -0.6, 0.6, 0.42, 0.57], (0.6, 0)),
        ("turn_corner", [0.25, 0.35, 0.26, 0.36, 0, 1.0], (0, 1.2)),
        ("side_post", [0.05, 0.15, 0.25, 0.35, 0, 0.8], (0.6, 0.8)),
        ("rear_obstacle", [-0.65, -0.5, -0.5, 0.5, 0, 0.8], (-0.6, 0)),
        ("safe_overhead", [0.6, 0.8, -0.6, 0.6, 0.72, 0.95], (0.6, 0)),
        ("safe_side", [0.4, 0.8, 0.7, 0.9, 0, 0.8], (0.6, 0)),
    ]


def fit_rigid(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    sign = np.ones(source.shape[1])
    sign[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(sign) @ u.T
    return rotation, b - rotation @ a


def extrinsic_calibration(rig):
    """Known metric target locations; fit sensor-to-body transforms on training targets."""
    rng = np.random.default_rng(67091)
    target = rng.uniform([0.8, -0.8, 0.1], [2.5, 0.8, 0.9], size=(100, 3))
    # Camera optical xyz = image-right, image-down, forward.
    expected = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    offset = np.array([rig.camera_x_m, 0, rig.camera_z_m])
    optical = (target - offset) @ expected
    optical += rng.normal(0, 0.001, optical.shape)
    rm, t = fit_rigid(optical[:60], target[:60])
    camera_error = np.linalg.norm(optical[60:] @ rm.T + t - target[60:], axis=1)
    lidar_target = target[:, :2]
    lidar_local = lidar_target - [rig.lidar_x_m, 0]
    lidar_local += rng.normal(0, 0.001, lidar_local.shape)
    lr, lt = fit_rigid(lidar_local[:60], lidar_target[:60])
    return {
        "training_targets": 60,
        "heldout_targets": 40,
        "point_sigma_m": 0.001,
        "camera_translation_m": t.tolist(),
        "camera_rotation": rm.tolist(),
        "camera_translation_error_mm": float(np.linalg.norm(t - offset) * 1000),
        "camera_holdout_p95_mm": float(np.percentile(camera_error, 95) * 1000),
        "lidar_translation_xy_m": lt.tolist(),
        "lidar_translation_error_mm": float(np.linalg.norm(lt - [rig.lidar_x_m, 0]) * 1000),
        "lidar_yaw_deg": float(np.degrees(np.arctan2(lr[1, 0], lr[0, 0]))),
        "unestimated": "camera intrinsics and body dimensions are known fixture inputs; planar LiDAR height must be measured separately",
    }


def render_fixture(boxes, rig, path):
    """Independent MuJoCo RGB/depth verifies the analytic sensor convention."""
    f = rig
    geoms = []
    for b in boxes:
        center = (np.asarray(b)[[0, 2, 4]] + np.asarray(b)[[1, 3, 5]]) / 2
        half = (np.asarray(b)[[1, 3, 5]] - np.asarray(b)[[0, 2, 4]]) / 2
        geoms.append(
            f'<geom type="box" pos="{" ".join(map(str, center))}" size="{" ".join(map(str, half))}" rgba=".25 .45 .75 1"/>'
        )
    xml = f'''<mujoco><visual><global offwidth="640" offheight="480"/></visual>
    <worldbody><light pos="0 0 3"/><geom type="plane" size="20 20 .05" rgba=".5 .5 .5 1"/>
    {"".join(geoms)}<camera name="rgbd" pos="{f.camera_x_m} 0 {f.camera_z_m}"
    xyaxes="0 -1 0 0 0 1" fovy="{f.vfov_deg}"/></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.map.znear = 0.01 / model.stat.extent
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=f.height, width=f.width)
    try:
        renderer.update_scene(data, camera="rgbd")
        rgb = renderer.render().copy()
        Image.fromarray(rgb).resize((640, 480)).save(path)
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
    finally:
        renderer._gl_context.make_current()
        renderer.close()
    return depth


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rig = NavigationRig()
    protocol = {
        "rig": rig.to_dict(),
        "fixtures": fixtures(),
        "scope": "simulation geometry calibration; no real camera/robot calibration",
        "oracle": "independent SAT box collision, stop poses sampled at 0.04 s",
        "sensor": "analytic metric RGB-D depth and planar LiDAR; MuJoCo cross-check",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rig_source_sha256": hashlib.sha256(
            (Path(__file__).parents[1] / "navigation_geometry.py").read_bytes()
        ).hexdigest(),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8"
    )
    rows = []
    for name, box, command in fixtures():
        boxes = np.array([box], float)
        ranges, hit, depth, valid = sense(boxes, np.zeros(3), rig)
        lidar = lidar_obstacle_points(ranges, hit, rig)
        camera = depth_points(depth, valid, rig)
        path = stop_path(*command, rig)
        oracle = min(collision_clearance(p, boxes, rig) for p in path)
        observed = np.vstack((lidar, camera))
        point = min(
            (
                np.linalg.norm(observed[:, None, :2] - path[None, :, :2], axis=2).min()
                if len(observed)
                else float("inf")
            ),
            100.0,
        )
        l = point_clearance(lidar, path, rig, rig.margin_m)
        fused = point_clearance(np.vstack((lidar, camera)), path, rig, rig.margin_m)
        # A scan seen 60 cm earlier is transformed into the current body frame.
        earlier_pose = np.array([-0.6, 0, 0])
        _, _, old_depth, old_valid = sense(boxes, earlier_pose, rig)
        memory = world_points(depth_points(old_depth, old_valid, rig), earlier_pose)
        remembered = point_clearance(np.vstack((lidar, camera, memory)), path, rig, rig.margin_m)
        rendered = render_fixture(boxes, rig, output / f"{name}-rgb.png")
        # Interior pixels only: rasterized triangle edges differ from point rays.
        from scipy.ndimage import binary_erosion

        mask = binary_erosion(valid & (depth < rig.max_range_m - 0.1), iterations=2)
        discrepancy = np.abs(depth[mask] - rendered[mask])
        pts = depth_points(depth, valid, rig)
        uv, in_front = project(pts, rig)
        rows.append(
            {
                "case": name,
                "command": command,
                "oracle_collision": oracle <= 0,
                "oracle_clearance_m": oracle if np.isfinite(oracle) else None,
                "point_robot_hazard": bool(point <= 0.01),
                "lidar_body_hazard": l <= 0,
                "fused_body_hazard": fused <= 0,
                "with_recent_depth_hazard": remembered <= 0,
                "lidar_hits": int(hit.sum()),
                "depth_obstacle_points": len(pts),
                "depth_p95_difference_m": float(np.percentile(discrepancy, 95)),
                "stop_center_distance_m": float(np.linalg.norm(path[-1, :2])),
            }
        )
        np.savez_compressed(
            output / f"{name}.npz",
            ranges=ranges,
            hits=hit,
            depth=depth,
            valid=valid,
            camera_points=pts,
            uv=uv[in_front],
            stop_path=path,
        )
    sweep = []
    for x in np.arange(0.4, 1.61, 0.1):
        _, _, depth, valid = sense(
            np.array([[x, x + 0.18, -0.65, 0.65, 0, 0.14]]), np.zeros(3), rig
        )
        points = depth_points(depth, valid, rig)
        usable = max(0.0, x - rig.front_m - rig.margin_m)
        vmax = -rig.brake_m_s2 * rig.latency_s + np.sqrt(
            (rig.brake_m_s2 * rig.latency_s) ** 2 + 2 * rig.brake_m_s2 * usable
        )
        sweep.append(
            {
                "front_x_m": float(x),
                "visible_points": len(points),
                "maximum_braking_speed_m_s": float(vmax),
            }
        )
    report = {
        "protocol": protocol,
        "extrinsics": extrinsic_calibration(rig),
        "rows": rows,
        "low_obstacle_visibility": sweep,
        "interpretation": "No return is unknown, not free space. Memory requires earlier visibility and reliable short-term odometry.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/body_calibration_v1")
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
