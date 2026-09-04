"""Lesson 22: pinhole camera — world to pixels, and back via depth.

A 3D scene (ground grid + a vertical pole) in an ENU world frame is observed by
one pinhole camera (intrinsics K, extrinsics [R | t] from look-at). Four checks:
(1) world -> pixel -> world round trip with exact depth is exact;
(2) without depth a pixel only defines a ray, so scale is lost (the monocular
depth limitation);
(3) depth noise makes the 3D point cloud error grow with distance;
(4) projecting the same scene from a moved camera and unprojecting with the new
depth returns the SAME world points (pose consistency).
No camera model library, rendering or learned depth is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

EXPERIMENT = "pinhole_camera_projection"
# Canvas: 640 x 480 px; principal point at the centre.
WIDTH_PX, HEIGHT_PX = 640, 480
FOCAL_PX = 600.0
CX_PX, CY_PX = 320.0, 240.0
K_INTRINSIC = np.array([[FOCAL_PX, 0.0, CX_PX], [0.0, FOCAL_PX, CY_PX], [0.0, 0.0, 1.0]])
EYE = np.array([2.0, 1.5, 1.4])  # camera position (E, N, U)
TARGET = np.array([3.0, 3.0, 0.0])  # looking at the middle of the ground grid
DEPTH_NOISE_STD_M = 0.05
# Camera working distance: points closer than this are outside the near plane.
NEAR_PLANE_M = 0.5
DEFAULT_RUNS = 20
DEFAULT_SEED = 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera pose [R|t]: world -> camera, camera axes x=right y=down z=forward."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Up axis parallel to viewing direction")
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)  # r x d = f requires d = f x r
    down = down / np.linalg.norm(down)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ eye
    return rotation, translation


def to_camera(points_world, rotation, translation):
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected [N, 3] world points")
    return (rotation @ points.T).T + translation


def to_world(points_camera, rotation, translation):
    points = np.asarray(points_camera, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected [N, 3] camera points")
    return (rotation.T @ (points - translation).T).T


def project(points_world, rotation, translation, intrinsic=K_INTRINSIC, near_plane=None):
    """Project world points to pixels; near-plane points are dropped.

    near_plane=None uses NEAR_PLANE_M; pass 0.0 when re-projecting points that
    already live in front of the camera (back-projected clouds).
    """
    if near_plane is None:
        near_plane = NEAR_PLANE_M
    camera = to_camera(points_world, rotation, translation)
    z_mask = camera[:, 2] > near_plane
    camera = camera[z_mask]
    pixels = np.empty((len(camera), 2))
    for index, (x, y, z) in enumerate(camera):
        v = intrinsic @ np.array([x, y, z])
        if v[2] <= 0:
            raise ValueError("Invalid projection")
        pixels[index] = [v[0] / v[2], v[1] / v[2]]
    return pixels, z_mask, camera


def project_with_depth(points_world, rotation, translation, intrinsic=K_INTRINSIC):
    """Project and return aligned (pixels, depths, world) for visible points.

    depth is the camera-frame z; the near plane (0.5 m) acts as the minimum
    working distance, mirroring real stereo/depth sensors.
    """
    camera = to_camera(points_world, rotation, translation)
    keep = camera[:, 2] > NEAR_PLANE_M
    camera_visible = camera[keep]
    world_visible = np.asarray(points_world, dtype=float)[keep]
    pixels = np.empty((len(camera_visible), 2))
    for index, (x, y, z) in enumerate(camera_visible):
        v = intrinsic @ np.array([x, y, z])
        if v[2] <= 0:
            raise ValueError("Invalid projection")
        pixels[index] = [v[0] / v[2], v[1] / v[2]]
    return pixels, camera_visible[:, 2], world_visible


def unproject(pixels, depth, rotation, translation, intrinsic=K_INTRINSIC):
    """Generalized inverse: pixel + scalar depth (no z-dependent backproject)."""
    pixels = np.asarray(pixels, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if pixels.shape != (len(depth), 2) or depth.ndim != 1 or np.any(depth <= 0):
        raise ValueError("Need per-pixel positive depths")
    inverse = np.linalg.inv(intrinsic)
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    ray_camera = (inverse @ homogeneous.T).T
    camera_points = ray_camera * depth[:, None]
    return to_world(camera_points, rotation, translation)


def scene_points():
    """Ground 11x11 grid plus a vertical pole; every point is a known 3D point."""
    xs, ns = np.meshgrid(np.linspace(0.75, 5.25, 10), np.linspace(0.75, 5.25, 10))
    ground = np.column_stack([xs.ravel(), ns.ravel(), np.zeros(xs.size)])
    heights = np.linspace(0.0, 2.0, 6)
    pole = np.column_stack([np.full(heights.size, 4.2), np.full(heights.size, 3.6), heights])
    return np.vstack([ground, pole]), ground, pole


def repeated_noisy_cloud(seed_count, seed, depth_std=DEPTH_NOISE_STD_M):
    """Per-seed noisy depth unprojection for the same scene: error vs distance."""
    points, _, _ = scene_points()
    rotation, translation = look_at(EYE, TARGET)
    exact_pixels, depths_exact, world_exact = project_with_depth(points, rotation, translation)
    # Data: pixels, exact depths, world points that survived the projection.
    pixel_of, depth_of, world_of = [], [], []
    errors, reprojection = [], []
    for s in range(seed_count):
        rng = np.random.default_rng(seed + 1000 * s)
        noisy = depths_exact + rng.normal(0.0, depth_std, size=len(depths_exact))
        clouds = unproject(exact_pixels, noisy, rotation, translation)
        error = np.linalg.norm(clouds - world_exact, axis=1)
        errors.append(error)
        back, _, _ = project(clouds, rotation, translation, near_plane=0.0)
        reprojection.append(np.linalg.norm(back - exact_pixels, axis=1))
        if s == 0:
            pixel_of, depth_of, world_of = exact_pixels, depths_exact, world_exact
    errors = np.asarray(errors)
    reprojection = np.asarray(reprojection)
    return {
        "pixels": pixel_of,
        "depths": depth_of,
        "world": world_of,
        "errors_m": errors,
        "reprojection_px": reprojection,
        "mean_error_by_depth": np.array(
            [
                errors[:, depth_of >= d].mean() if np.any(depth_of >= d) else 0.0
                for d in (2.0, 3.0, 4.0, 5.0, 6.0)
            ]
        ),
        "depth_bins": np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
    }


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two noise repetitions")
    points, _, _ = scene_points()
    rotation, translation = look_at(EYE, TARGET)
    pixels, depths, visible_world = project_with_depth(points, rotation, translation)
    # Round trip with exact depth.
    clouds = unproject(pixels, depths, rotation, translation)
    roundtrip_error = np.linalg.norm(clouds - visible_world, axis=1)
    # No-depth ambiguity: three depth guesses along the same ray.
    pixel_choice = pixels[len(pixels) // 2]
    ray_points = unproject(
        np.tile(pixel_choice, (3, 1)), np.array([2.0, 4.0, 6.0]), rotation, translation
    )
    # Pose consistency: move the camera, re-project, unproject, compare worlds.
    target_eye = EYE + np.array([0.5, 0.0, 0.3])
    rot2, trans2 = look_at(target_eye, TARGET)
    pixels2, depths2, visible_world2 = project_with_depth(points, rot2, trans2)
    clouds2 = unproject(pixels2, depths2, rot2, trans2)
    pose_error = np.linalg.norm(clouds2 - visible_world2, axis=1)
    matched = pose_error < 1e-6
    noisy = repeated_noisy_cloud(runs, seed)
    inverse = np.linalg.inv(K_INTRINSIC)
    homogeneous = np.column_stack([noisy["pixels"], np.ones(len(noisy["pixels"]))])
    ray_norm = np.linalg.norm((inverse @ homogeneous.T).T, axis=1)
    ratio = noisy["errors_m"] / ray_norm[None, :]
    archive = {
        "world_points": points,
        "projected_pixels": pixels,
        "camera_points": to_camera(visible_world, rotation, translation),
        "roundtrip_error_m": roundtrip_error,
        "ray_points": ray_points,
        "ray_depths": np.array([2.0, 4.0, 6.0]),
        "noisy_cloud_errors_m": noisy["errors_m"],
        "noisy_cloud_reprojection_px": noisy["reprojection_px"],
        "cloud_depths": noisy["depths"],
        "cloud_world": noisy["world"],
        "mean_error_by_depth": noisy["mean_error_by_depth"],
        "depth_bins": noisy["depth_bins"],
        "pose_error_m": pose_error,
        "ray_norm": ray_norm,
        "depth_noise_estimates_m": ratio,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "camera_model": "pinhole {u = K [R|t] X / z}",
        "world_frame": "ENU (x east, y north, z up, right-handed)",
        "camera_frame": "x right, y down, z forward (conventional image axes)",
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "intrinsic": K_INTRINSIC.tolist(),
        "focal_px": FOCAL_PX,
        "principal_point_px": [CX_PX, CY_PX],
        "eye_m": EYE.tolist(),
        "target_m": TARGET.tolist(),
        "scene_points": len(points),
        "visible_points": len(visible_world),
        "depth_noise_std_m": DEPTH_NOISE_STD_M,
        "runs_per_group": runs,
        "base_seed": seed,
        "roundtrip_max_error_m": float(np.max(roundtrip_error)),
        "pose_consistency_max_error_m": float(np.max(pose_error)),
        "pose_all_matched": bool(np.all(matched)),
        "noisy_mean_error_m": float(noisy["errors_m"].mean()),
        "noisy_max_error_m": float(noisy["errors_m"].max()),
        "noisy_mean_reprojection_px": float(noisy["reprojection_px"].mean()),
        "noise_estimate_mean_m": float(ratio.mean()),
        "noise_estimate_at_near_m": float(ratio[:, noisy["depths"] < 1.5].mean()),
        "noise_estimate_at_far_m": float(ratio[:, noisy["depths"] > 3.5].mean()),
        "mean_error_by_depth": noisy["mean_error_by_depth"].tolist(),
        "depth_bins": noisy["depth_bins"].tolist(),
        "ray_payload": {
            "pixel": pixel_choice.tolist(),
            "depths_m": [2.0, 4.0, 6.0],
            "points_m": ray_points.tolist(),
        },
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            "experiments/pinhole_projection.py": digest(Path(__file__).resolve()),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            "Ideal geometric camera: no lens distortion, no rolling shutter",
            "Depth is supplied or synthetic; no learned monocular metric depth",
            "Single camera; no calibration estimation, no ground truth from real imagery",
            "No rendering, occlusion, texture, or image-level noise",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(archive, summary, output)
    return summary


def make_plot(archive, summary, output):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_pixels = fig.add_subplot(2, 3, 2)
    ax_roundtrip = fig.add_subplot(2, 3, 3)
    ax_error = fig.add_subplot(2, 3, 4)
    ax_reproj = fig.add_subplot(2, 3, 5)
    ax_ray = fig.add_subplot(2, 3, 6, projection="3d")
    points = archive["world_points"]
    pixels = archive["projected_pixels"]
    ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c="#2563eb", s=6)
    ax3d.plot(*summary["eye_m"], marker="o", color="#dc2626", ms=8)
    ax3d.set(
        xlabel="东 / m",
        ylabel="北 / m",
        zlabel="高 / m",
        title="世界场景：地面网格与竖直杆（ENU）",
    )
    ax3d.view_init(elev=45, azim=-55)
    ax_pixels.scatter(pixels[:, 0], HEIGHT_PX - pixels[:, 1], s=6, color="#2563eb")
    ax_pixels.set_xlim(0, WIDTH_PX)
    ax_pixels.set_ylim(0, HEIGHT_PX)
    ax_pixels.set(xlabel="u / px", ylabel="v / px（已翻转，上为 0）", title="投影到 640×480 图像")
    ax_pixels.set_aspect("equal")
    depths = archive["camera_points"][:, 2]
    ax_roundtrip.plot(depths, archive["roundtrip_error_m"], ".", ms=4, color="#0f766e")
    ax_roundtrip.set_yscale("log")
    ax_roundtrip.set(
        xlabel="相机深度 / m",
        ylabel="往返误差 / m",
        title="有精确深度：往返一致性（数值量级）",
    )
    errs = archive["noisy_cloud_errors_m"]
    d = archive["cloud_depths"]
    bins = np.linspace(float(d.min()), float(d.max()) + 1e-9, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    means = np.array(
        [errs[:, (d >= bins[i]) & (d < bins[i + 1])].mean() for i in range(len(bins) - 1)]
    )
    ax_error.plot(centers, means * 100, "-o", ms=3, color="#9333ea")
    ax_error.set(
        xlabel="深度 / m",
        ylabel="点云位置误差 / cm",
        title="点云误差：近处（图像边缘）被射线倍率放大",
    )
    ratio = archive["depth_noise_estimates_m"]
    means_ratio = np.array(
        [ratio[:, (d >= bins[i]) & (d < bins[i + 1])].mean() for i in range(len(bins) - 1)]
    )
    ax_reproj.plot(centers, means_ratio * 100, "-o", ms=3, color="#ea580c")
    ax_reproj.axhline(DEPTH_NOISE_STD_M * 100, color="#0f172a", ls="--", lw=1.2)
    ax_reproj.set(
        xlabel="深度 / m",
        ylabel="估计深度噪声 / cm",
        title="机制验证：误差 ÷ 射线倍率 ≈ 真实深度噪声（5 cm 虚线）",
    )
    ray = archive["ray_points"]
    ax_ray.scatter(ray[:, 0], ray[:, 1], ray[:, 2], c="#0891b2", s=40)
    ax_ray.plot(ray[:, 0], ray[:, 1], ray[:, 2], "-", color="#0891b2")
    ax_ray.set(
        xlabel="东 / m",
        ylabel="北 / m",
        zlabel="高 / m",
        title="无深度：同一像素只给一条射线（三个深度猜测）",
    )
    ax_ray.view_init(elev=35, azim=-60)
    for ax in (ax3d, ax_pixels, ax_roundtrip, ax_error, ax_reproj, ax_ray):
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for noise statistics")
    report = run_experiment(args.output, runs=args.runs, seed=args.seed)
    print(
        f"roundtrip max error {report['roundtrip_max_error_m']:.3e} m; "
        f"pose consistency max {report['pose_consistency_max_error_m']:.3e} m; "
        f"noisy mean {report['noisy_mean_error_m'] * 100:.2f} cm "
        f"(max {report['noisy_max_error_m'] * 100:.2f} cm)"
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
