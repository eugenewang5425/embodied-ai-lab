"""Synthetic single-reference distance calibration tests (no real capture required)."""

import cv2
import numpy as np
import pytest

from monocular_depth.calibration import DEFAULT_BOARD, detect_board, generate_board, make_board
from monocular_depth.distance import (
    board_pose,
    calibrate_from_metadata,
    plane_depth_map,
    region_mask,
    render_debug,
)
from monocular_depth.io import read_image, save_result

MATRIX = np.array([[800.0, 0, 640], [0, 810, 360], [0, 0, 1]])
SIZE = (1280, 720)  # width, height
SHAPE = SIZE[::-1]  # (height, width) for array operations
POSE = {
    "rotation_vector": np.array([0.12, -0.08, 0.05], dtype=np.float64),
    "translation": np.array([0.0, 0.0, 0.55], dtype=np.float64),
}


def synthetic_calibration():
    return {
        "camera_matrix": MATRIX.tolist(),
        "distortion_coefficients": [0.0] * 5,
        "image_size": list(SIZE),
        "source": "charuco_images",
        "quality_pass": True,
        "rms_px": 0.1,
        "valid_views": 12,
        "per_view_rms_px": [0.1] * 12,
        "board": DEFAULT_BOARD,
    }


def analytic_plane(rotation_vector, translation):
    rotation, _ = cv2.Rodrigues(rotation_vector)
    normal = rotation[:, 2]
    return {"normal": normal.tolist(), "plane_distance_m": float(normal @ translation)}


def render_board(rng, scale=0.8, tmp_path=None):
    """Render a frontal ChArUco board and a model-like depth map with a known scale."""
    board_dir = tmp_path / "board"
    generate_board(board_dir)
    texture = read_image(board_dir / "board.png")
    corrected, _ = cv2.getOptimalNewCameraMatrix(MATRIX, np.zeros(5), SIZE, 0)
    world = np.array(
        [[0, 0, 0], [0.175, 0, 0], [0.175, 0.125, 0], [0, 0.125, 0]], dtype=np.float64
    )
    texture_corners = np.array([[0, 0], [1399, 0], [1399, 999], [0, 999]], dtype=np.float32)
    projected, _ = cv2.projectPoints(
        world, POSE["rotation_vector"], POSE["translation"], corrected, np.zeros(5)
    )
    transform = cv2.getPerspectiveTransform(texture_corners, projected.reshape(4, 2).astype(np.float32))
    rgb = cv2.warpPerspective(texture, transform, SIZE, borderValue=(255, 255, 255))
    plane = analytic_plane(POSE["rotation_vector"], POSE["translation"])
    true_z = plane_depth_map(plane, corrected, SHAPE)
    depth = true_z / scale
    depth = depth * (1.0 + rng.normal(0, 0.01, depth.shape))
    metadata = save_result(
        tmp_path / "capture",
        "SYNTHETIC_METRIC_TEST",
        rgb,
        depth.astype(np.float32),
        variant="metric",
        metadata={
            "calibrated": True,
            "image_space": "undistorted",
            "camera_matrix": corrected.tolist(),
            "calibration": synthetic_calibration(),
            "model": "synthetic test model",
            "checkpoint_sha256": None,
            "input_size": 518,
            "calibration_file": str(tmp_path / "synthetic.json"),
        },
    )[-1]
    return metadata, rgb, depth, corrected


def test_single_reference_recovers_known_scale(tmp_path):
    rng = np.random.default_rng(293)
    metadata, rgb, _, _ = render_board(rng, scale=0.8, tmp_path=tmp_path)
    report, debug = calibrate_from_metadata(metadata)
    assert report["interior_pixels"] > 5000
    assert report["corner_count"] >= 10
    assert report["board_pose"]["reprojection_rms_px"] < 2.0
    assert report["scale"] == pytest.approx(0.8, rel=0.02)
    assert report["scale_p05_p95"][0] < report["scale"] < report["scale_p05_p95"][1]
    assert report["residual_after_mad_relative"] < 0.02
    assert report["fit"] == "scale_only"
    assert report["validation"]["independent"] is False
    assert report["metric_accuracy_validated"] is False
    assert debug is not None
    assert debug["mask"].shape == rgb.shape[:2]
    view = render_debug(debug)
    assert view.shape == rgb.shape


def test_ruler_crosscheck_and_measured_validation(tmp_path):
    rng = np.random.default_rng(7)
    metadata, _, _, _ = render_board(rng, scale=0.8, tmp_path=tmp_path)
    report, _ = calibrate_from_metadata(metadata, measured_distance_m=0.60)
    assert report["ground_truth"]["ruler"] is not None
    assert report["center_depth"]["true_m"] == pytest.approx(0.55, abs=0.02)
    assert report["center_depth"]["predicted_m"] > 0
    offset = report["ground_truth"]["ruler"]["projection_center_offset_m"]
    assert -0.2 < offset < 0.2
    assert offset == pytest.approx(report["center_depth"]["true_m"] - 0.60)
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_from_metadata(metadata, measured_distance_m=-1.0)


def test_relative_or_uncalibrated_capture_refused(tmp_path):
    paths = save_result(
        tmp_path / "rel",
        "SYNTHETIC_RELATIVE_TEST",
        np.zeros((*SIZE[::-1], 3), np.uint8),
        np.full(SIZE[::-1], 0.5, np.float32),
        variant="relative",
    )
    with pytest.raises(ValueError, match="metric"):
        calibrate_from_metadata(paths[-1])


def test_board_missing_refused(tmp_path):
    corrected, _ = cv2.getOptimalNewCameraMatrix(MATRIX, np.zeros(5), SIZE, 0)
    metadata = save_result(
        tmp_path / "nocapture",
        "SYNTHETIC_NO_BOARD",
        np.full((*SIZE[::-1], 3), 128, np.uint8),
        np.full(SIZE[::-1], 0.6, np.float32),
        variant="metric",
        metadata={
            "calibrated": True,
            "image_space": "undistorted",
            "camera_matrix": corrected.tolist(),
            "calibration": synthetic_calibration(),
        },
    )[-1]
    with pytest.raises(ValueError, match="not detected"):
        calibrate_from_metadata(metadata)


def test_region_mask_erodes_and_rejects_invalid_margin():
    points = np.array([[100, 100], [900, 120], [880, 600], [120, 580]], dtype=np.float32)
    full = region_mask(points, (720, 1280), 0.0)
    eroded = region_mask(points, (720, 1280), 0.12)
    assert eroded.sum() < full.sum()
    with pytest.raises(ValueError, match="margin"):
        region_mask(points, (720, 1280), 0.6)


def test_plane_depth_map_center_equals_plane_distance():
    frontal = {"normal": [0.0, 0.0, 1.0], "plane_distance_m": 0.6}
    z = plane_depth_map(frontal, MATRIX, (100, 200))
    assert z[50, 100] == pytest.approx(0.6)


def test_board_pose_recovers_known_translation(tmp_path):
    rng = np.random.default_rng(11)
    _, rgb, _, corrected = render_board(rng, scale=0.8, tmp_path=tmp_path)
    detection = detect_board(rgb, make_board(DEFAULT_BOARD))
    assert detection is not None
    pose = board_pose(detection, corrected)
    assert pose["plane_distance_m"] == pytest.approx(0.55, rel=0.02)
    assert pose["normal_angle_to_optical_axis_deg"] < 15
