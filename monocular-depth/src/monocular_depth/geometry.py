"""Experimental camera-frame geometry; never promotes predictions to measured truth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .calibration import validate_calibration
from .focus import validate_frame_focus
from .io import read_image
from .records import read_json, timestamp, write_json


def validate_intrinsics(matrix: np.ndarray) -> None:
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or abs(matrix[0, 1]) > 1e-9
        or abs(matrix[1, 0]) > 1e-9
    ):
        raise ValueError("Invalid pinhole camera matrix")


def backproject(depth, bgr_image, matrix, *, stride=4, max_depth=5.0):
    matrix = np.asarray(matrix, dtype=float)
    validate_intrinsics(matrix)
    if stride < 1 or not np.isfinite(max_depth) or max_depth <= 0:
        raise ValueError("stride and max_depth must be positive")
    if depth.ndim != 2 or bgr_image.shape != (*depth.shape, 3):
        raise ValueError("RGB and depth dimensions must match")
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0) & (z <= max_depth)
    z, u, v = z[valid], cols[valid], rows[valid]
    points = np.column_stack(
        (
            (u - matrix[0, 2]) * z / matrix[0, 0],
            (v - matrix[1, 2]) * z / matrix[1, 1],
            z,
        )
    )
    colors = bgr_image[::stride, ::stride][valid, ::-1].astype(float) / 255.0
    return points, colors


def load_metric_capture(metadata_path: Path):
    """Validate a calibrated metric capture and return (record, rgb, depth, matrix)."""
    record = read_json(metadata_path)
    if record.get("variant") != "metric" or record.get("unit") != "m":
        raise ValueError("Metric evaluation requires metric depth; relative values are NOT meters")
    if record.get("calibrated") is not True or record.get("image_space") != "undistorted":
        raise ValueError(
            "Metric evaluation requires real calibration and the undistorted image"
        )
    calibration = record.get("calibration")
    if not isinstance(calibration, dict):
        raise TypeError("Missing calibration provenance")
    validate_calibration(calibration)
    if calibration.get("focus_lock_verified_for_accepted_frames"):
        if not record.get("camera_focus_verified"):
            raise ValueError(
                "Metric evaluation requires verified capture focus for this calibration"
            )
        target = validate_frame_focus(record["focus_before_frame"], record["focus_after_frame"])
        if target != calibration.get("fixed_focus_driver_units"):
            raise ValueError("Capture focus does not match calibration")
    if record.get("image_size") != calibration["image_size"]:
        raise ValueError("Metadata and calibration resolutions differ")
    matrix = np.asarray(record["camera_matrix"], dtype=float)
    validate_intrinsics(matrix)
    size = tuple(record["image_size"])
    expected, _ = cv2.getOptimalNewCameraMatrix(
        np.asarray(calibration["camera_matrix"], dtype=float),
        np.asarray(calibration["distortion_coefficients"], dtype=float),
        size,
        0,
    )
    if not np.allclose(matrix, expected, rtol=1e-7, atol=1e-7):
        raise ValueError("Intrinsics do not match the undistortion transform")

    def local_file(key):
        name = record[key]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Metadata must reference adjacent artifact filenames")
        return metadata_path.parent / name

    depth = np.load(local_file("depth_file"), allow_pickle=False)
    rgb = read_image(local_file("rgb_file"))
    if rgb is None or [rgb.shape[1], rgb.shape[0]] != list(size):
        raise ValueError("RGB does not match the recorded resolution")
    return record, rgb, depth, matrix


def geometry_from_record(metadata_path: Path, stride=4, max_depth=5.0):
    record, rgb, depth, matrix = load_metric_capture(metadata_path)
    points, colors = backproject(depth, rgb, matrix, stride=stride, max_depth=max_depth)
    if len(points) < 3:
        raise ValueError("Fewer than three valid points after filtering")
    return points, colors, record


def process_cloud(points, colors, voxel_size=0.01, fit_plane=False, plane_threshold=0.02):
    import open3d as o3d

    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size must be finite and nonnegative")
    if not np.isfinite(plane_threshold) or plane_threshold <= 0:
        raise ValueError("plane_threshold must be finite and positive")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.points) < 3:
        raise ValueError("Fewer than three points after voxel sampling")
    plane = None
    if fit_plane:
        o3d.utility.random.seed(42)
        equation, indices = cloud.segment_plane(plane_threshold, 3, 1000)
        normal = np.asarray(equation[:3])
        if len(indices) < 3 or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-10:
            raise ValueError("No non-degenerate plane found; retry without --fit-plane")
        distances = (np.asarray(cloud.points)[indices] @ normal + equation[3]) / np.linalg.norm(
            normal
        )
        plane = {
            "equation_camera_frame": list(equation),
            "inlier_count": len(indices),
            "inlier_ratio": len(indices) / len(cloud.points),
            "inlier_rms_m_predicted": float(np.sqrt(np.mean(distances**2))),
            "threshold_m_predicted": plane_threshold,
            "semantic_label": None,
            "note": "Dominant fitted plane, not necessarily a table; not a scale validation.",
        }
    return np.asarray(cloud.points), np.asarray(cloud.colors), plane


def write_ply(path: Path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            "comment Predicted geometry; camera frame x-right y-down z-forward\n"
            f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        rgb = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
        np.savetxt(stream, np.column_stack((points, rgb)), fmt="%.7f %.7f %.7f %d %d %d")


def export_cloud(
    metadata_path,
    output,
    stride=4,
    max_depth=5.0,
    voxel_size=0.01,
    fit_plane=False,
    plane_threshold=0.02,
):
    report_path = output.with_suffix(".json")
    if output.suffix.lower() != ".ply":
        raise ValueError("Point cloud output must use .ply extension")
    if output.exists() or report_path.exists():
        raise FileExistsError("Point cloud output already exists; choose a new --output")
    points, colors, source = geometry_from_record(metadata_path, stride, max_depth)
    original_count = len(points)
    points, colors, plane = process_cloud(points, colors, voxel_size, fit_plane, plane_threshold)
    report = {
        "schema_version": 1,
        "timestamp_utc": timestamp(),
        "source_metadata": str(metadata_path.resolve()),
        "coordinate_frame": "camera_opencv_x_right_y_down_z_forward",
        "unit": "m_predicted",
        "metric_accuracy_validated": False,
        "world_pose": None,
        "stride": stride,
        "max_depth": max_depth,
        "voxel_size": voxel_size,
        "points_before_voxel": original_count,
        "points_exported": len(points),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
        "dominant_plane": plane,
        "checkpoint_sha256": source.get("checkpoint_sha256"),
        "warning": "Predicted single-view geometry only. No occluded geometry or safety guarantee.",
    }
    write_ply(output, points, colors)
    write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Export calibrated metric predictions to camera-frame PLY"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--fit-plane", action="store_true")
    parser.add_argument("--plane-threshold", type=float, default=0.02)
    args = parser.parse_args()
    output = args.output or args.metadata.with_name(args.metadata.stem + "_cloud.ply")
    try:
        report = export_cloud(
            args.metadata,
            output,
            args.stride,
            args.max_depth,
            args.voxel_size,
            args.fit_plane,
            args.plane_threshold,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"Point cloud refused: {error}\n")
    print(f"Exported {report['points_exported']} predicted points: {output.resolve()}")
    print("Camera frame only; metric accuracy NOT validated; NOT for robot safety")
