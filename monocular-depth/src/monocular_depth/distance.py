"""Single-reference distance calibration for the metric depth pipeline.

The ChArUco board was printed at a known 25 mm square pitch, so the calibrated
intrinsics let the board pose recovered from a saved frame give the board plane
in the camera frame. That plane is the metric reference: it does not depend on
the lens-front ruler datum and it is independent of the depth model.

One reference distance can only fit a multiplicative scale (the model already
outputs nominally metric depth). A second, independent distance is required to
fit scale+offset, and any fitted scale must be validated on data that was not
used for the fit.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .calibration import CALIBRATION_DIR, detect_board, make_board
from .geometry import load_metric_capture
from .io import write_image
from .records import timestamp, write_json

DISTANCE_CALIBRATION_DIR = CALIBRATION_DIR / "distance"
MIN_INTERIOR_PIXELS = 2000
MIN_CHARUCO_CORNERS = 10
MAX_POSE_RMS_PX = 2.0


def board_pose(detection, matrix) -> dict:
    """Recover the board plane in the camera frame from a ChArUco detection."""
    object_points, image_points, _, _ = detection
    matrix = np.asarray(matrix, dtype=np.float64)
    zero_distortion = np.zeros(5, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, matrix, zero_distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, matrix, zero_distortion, flags=cv2.SOLVEPNP_ITERATIVE
        )
    if not ok or rvec is None or tvec is None:
        raise ValueError("Board pose estimation failed")
    rotation, _ = cv2.Rodrigues(rvec)
    normal = np.asarray(rotation[:, 2], dtype=np.float64).reshape(3)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    if normal[2] < 0:
        # The board pattern must face the camera, so its +z axis points to the viewer.
        raise ValueError("Board normal points away from the camera")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, zero_distortion)
    residual = np.linalg.norm(
        projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        axis=1,
    )
    rms_px = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(rms_px) or rms_px > MAX_POSE_RMS_PX:
        raise ValueError(f"Board pose reprojection too poor: {rms_px:.2f} px")
    plane_distance = float(normal @ translation)
    if not np.isfinite(plane_distance) or plane_distance <= 0:
        raise ValueError("Board plane is not in front of the camera")
    angle = math.degrees(
        math.acos(max(-1.0, min(1.0, normal[2] / np.linalg.norm(normal))))
    )
    rvec_flat = np.asarray(rvec, dtype=np.float64).reshape(3).tolist()
    return {
        "normal": normal.tolist(),
        "plane_distance_m": plane_distance,
        "translation_m": translation.tolist(),
        "rotation_vector": rvec_flat,
        "reprojection_rms_px": rms_px,
        "normal_angle_to_optical_axis_deg": angle,
        "pose_method": "ippe_single_solution",
    }


def plane_depth_map(pose: dict, matrix, image_size) -> np.ndarray:
    """True depth of the board plane along the optical axis, per pixel (m)."""
    height, width = image_size
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    ray = np.stack(
        (
            (cols - matrix[0, 2]) / matrix[0, 0],
            (rows - matrix[1, 2]) / matrix[1, 1],
            np.ones_like(rows),
        ),
        axis=-1,
    )
    normal = np.asarray(pose["normal"], dtype=np.float64)
    denom = ray @ normal
    z = pose["plane_distance_m"] / denom
    z[denom <= 0] = np.nan
    z[~np.isfinite(z)] = np.nan
    return z


def region_mask(image_points, image_size, margin_fraction=0.12) -> np.ndarray:
    """Interior of the board quadrilateral, eroded by margin_fraction of its span."""
    if not np.isfinite(margin_fraction) or margin_fraction < 0 or margin_fraction >= 0.5:
        raise ValueError("margin_fraction must be in [0, 0.5)")
    hull = cv2.convexHull(np.asarray(image_points, dtype=np.float32).reshape(-1, 2))
    mask = np.zeros(image_size, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    points = hull.reshape(-1, 2)
    span = min(
        float(points[:, 0].max() - points[:, 0].min()),
        float(points[:, 1].max() - points[:, 1].min()),
    )
    kernel = max(1, round(margin_fraction * span))
    if kernel > 1:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel)))
    return mask


def calibrate_from_metadata(
    metadata_path: Path,
    *,
    measured_distance_m: float | None = None,
    margin_fraction: float = 0.12,
    min_interior_pixels: int = MIN_INTERIOR_PIXELS,
) -> tuple[dict, dict | None]:
    """Fit the scale-only distance calibration from one saved metric capture.

    Returns (report, debug) where debug holds optional visualization arrays for
    the caller to persist; the report contains the fit and its limitations.
    """
    if measured_distance_m is not None and (
        not np.isfinite(measured_distance_m) or measured_distance_m <= 0
    ):
        raise ValueError("measured_distance_m must be finite and positive")
    record, rgb, depth, matrix = load_metric_capture(metadata_path)
    board_spec = record["calibration"]["board"]
    detection = detect_board(rgb, make_board(board_spec))
    if detection is None:
        raise ValueError("ChArUco board not detected reliably in the saved frame")
    _, image_points, _, ids = detection
    if len(ids) < MIN_CHARUCO_CORNERS:
        raise ValueError(f"Board detection too weak: {len(ids)} corners")
    pose = board_pose(detection, matrix)
    true_z = plane_depth_map(pose, matrix, (depth.shape[0], depth.shape[1]))
    mask = region_mask(image_points, (depth.shape[0], depth.shape[1]), margin_fraction)
    finite = np.isfinite(true_z) & np.isfinite(depth) & (depth > 0) & (mask > 0)
    count = int(finite.sum())
    if count < min_interior_pixels:
        raise ValueError(f"Interior pixels too few for a stable fit: {count}")
    z_true = true_z[finite].astype(np.float64)
    z_pred = depth[finite].astype(np.float64)
    ratio = z_true / z_pred
    scale = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - scale)))
    p05, p95 = float(np.percentile(ratio, 5)), float(np.percentile(ratio, 95))
    rel_before = (z_pred - z_true) / z_true
    rel_after = (scale * z_pred - z_true) / z_true
    residual_before = z_pred - z_true
    residual_after = scale * z_pred - z_true

    # Board center reference along the optical axis (projection-center datum).
    rvec_c = np.asarray(pose["rotation_vector"], dtype=np.float64).reshape(3, 1)
    tvec_c = np.asarray(pose["translation_m"], dtype=np.float64).reshape(3, 1)
    center_projected, _ = cv2.projectPoints(
        np.zeros((1, 3), dtype=np.float64), rvec_c, tvec_c, matrix, np.zeros(5)
    )
    center_ray = np.array(
        [
            (center_projected[0, 0, 0] - matrix[0, 2]) / matrix[0, 0],
            (center_projected[0, 0, 1] - matrix[1, 2]) / matrix[1, 1],
            1.0,
        ]
    )
    center_true = pose["plane_distance_m"] / float(center_ray @ np.asarray(pose["normal"]))
    center_u = round(center_projected[0, 0, 0])
    center_v = round(center_projected[0, 0, 1])
    center_pred = float(depth[center_v, center_u])
    ruler = None
    if measured_distance_m is not None:
        ruler = {
            "measured_distance_m": measured_distance_m,
            "projection_center_offset_m": float(center_true - measured_distance_m),
            "ruler_only_scale": float(measured_distance_m / max(center_pred, 1e-9)),
            "note": (
                "Ruler datum is the front lens/housing, not the projection center; "
                "the offset is inferred, not measured."
            ),
        }

    report = {
        "schema_version": 1,
        "timestamp_utc": timestamp(),
        "source_metadata": str(metadata_path.resolve()),
        "capture_timestamp_utc": record.get("timestamp_utc"),
        "model": record.get("model"),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
        "input_size": record.get("input_size"),
        "calibration_file": record.get("calibration_file"),
        "image_size": [depth.shape[1], depth.shape[0]],
        "board": board_spec,
        "board_pose": pose,
        "margin_fraction": margin_fraction,
        "interior_pixels": count,
        "corner_count": len(ids),
        "ground_truth": {
            "source": "charuco_board_pose",
            "board_pitch_m": board_spec["square_length_m"],
            "ruler": ruler,
        },
        "center_depth": {
            "projected_center_px": [
                float(center_projected[0, 0, 0]),
                float(center_projected[0, 0, 1]),
            ],
            "true_m": float(center_true),
            "predicted_m": float(center_pred),
        },
        "interior_median": {
            "true_m": float(np.median(z_true)),
            "predicted_m": float(np.median(z_pred)),
        },
        "scale": scale,
        "scale_mad": mad,
        "scale_p05_p95": [p05, p95],
        "residual_before_rms_m": float(np.sqrt(np.mean(residual_before**2))),
        "residual_after_rms_m": float(np.sqrt(np.mean(residual_after**2))),
        "residual_before_rms_relative": float(np.sqrt(np.mean(rel_before**2))),
        "residual_after_mad_relative": float(np.median(np.abs(rel_after))),
        "fit": "scale_only",
        "fit_note": (
            "One reference distance fits only a multiplicative scale. A second, "
            "independent distance is required to fit scale+offset or to validate this scale."
        ),
        "validation": {
            "independent": False,
            "note": "The same capture is used to fit and then describe itself; it is not validation.",
        },
        "metric_accuracy_validated": False,
    }

    error_map = np.full(depth.shape, np.nan, dtype=np.float32)
    error_map[finite] = rel_after.astype(np.float32)
    debug = {"rgb": rgb.copy(), "error_after_relative": error_map, "mask": mask}
    return report, debug


def render_debug(debug: dict) -> np.ndarray:
    """Compose the RGB board area with the residual error map after scale fitting."""
    mask = debug["mask"].astype(bool)
    error = np.zeros(debug["rgb"].shape[:2], dtype=np.float32)
    error[mask] = np.clip(debug["error_after_relative"][mask] * 100.0, -20.0, 20.0)
    heat = cv2.applyColorMap(
        np.clip((error + 20.0) / 40.0, 0.0, 1.0).astype(np.uint8) * 255,
        cv2.COLORMAP_JET,
    )
    shown = debug["rgb"].copy()
    overlay = shown.copy()
    overlay[mask] = heat[mask]
    cv2.addWeighted(overlay, 0.65, shown, 0.35, 0, shown)
    cv2.putText(
        shown,
        "relative residual after scale fit, clipped to +/-20%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return shown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the metric depth scale with one known working distance"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--measured-distance-m", type=float)
    parser.add_argument("--margin-fraction", type=float, default=0.12)
    args = parser.parse_args()
    output_dir = args.output_dir or DISTANCE_CALIBRATION_DIR / datetime.now(UTC).strftime(
        "session_%Y%m%d_%H%M%S"
    )
    if output_dir.exists():
        parser.error(f"Refusing to overwrite existing output: {output_dir}")
    try:
        report, debug = calibrate_from_metadata(
            args.metadata,
            measured_distance_m=args.measured_distance_m,
            margin_fraction=args.margin_fraction,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.exit(2, f"Distance calibration refused: {error}\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "distance-calibration.json", report)
    write_image(output_dir / "distance-calibration-debug.png", render_debug(debug))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
