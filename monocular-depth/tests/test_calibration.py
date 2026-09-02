import cv2
import numpy as np
import pytest

from monocular_depth.calibration import (
    DEFAULT_BOARD,
    detect_board,
    generate_board,
    load_calibration,
    make_board,
    prepare_image,
    solve_directory,
    solve_observations,
)
from monocular_depth.io import read_image, write_image
from monocular_depth.records import read_json, write_json


@pytest.fixture
def synthetic_calibration():
    """Synthetic test fixture only. Never written as the user's camera calibration."""
    return {
        "camera_matrix": [[800, 0, 640], [0, 810, 360], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
        "image_size": [1280, 720],
        "source": "charuco_images",
        "quality_pass": True,
        "rms_px": 0.1,
        "per_view_rms_px": [0.1] * 20,
        "valid_views": 20,
    }


def test_printable_board_detection_and_overwrite(tmp_path):
    page = generate_board(tmp_path)
    assert "width:175mm" in page.read_text(encoding="utf-8")
    assert read_json(tmp_path / "board.json") == DEFAULT_BOARD
    detection = detect_board(read_image(tmp_path / "board.png"), make_board(DEFAULT_BOARD))
    assert len(detection[3]) == 24
    with pytest.raises(FileExistsError):
        generate_board(tmp_path)


def test_synthetic_multi_view_calibration_recovers_intrinsics(synthetic_calibration):
    rng = np.random.default_rng(123)
    matrix = np.asarray(synthetic_calibration["camera_matrix"], dtype=float)
    obj = make_board(DEFAULT_BOARD).getChessboardCorners()
    objects, pixels = [], []
    for _ in range(30):
        rotation = rng.uniform(-0.5, 0.5, 3)
        translation = np.array(
            [rng.uniform(-0.2, 0.1), rng.uniform(-0.15, 0.1), rng.uniform(0.45, 0.85)]
        )
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, np.zeros(5))
        projected += rng.normal(0, 0.08, projected.shape).astype(np.float32)
        objects.append(obj)
        pixels.append(projected)
    result = solve_observations(objects, pixels, (1280, 720))
    assert result["quality_pass"]
    estimated = np.array(result["camera_matrix"])
    np.testing.assert_allclose(estimated[[0, 1], [0, 1]], matrix[[0, 1], [0, 1]], rtol=0.02)
    np.testing.assert_allclose(estimated[:2, 2], matrix[:2, 2], atol=8)


def test_insufficient_and_duplicate_views_rejected(tmp_path):
    generate_board(tmp_path / "board")
    image = read_image(tmp_path / "board" / "board.png")
    for index in range(15):
        write_image(tmp_path / "views" / f"{index}.png", image)
    output = tmp_path / "camera.json"
    with pytest.raises(ValueError, match="at least 12"):
        solve_directory(tmp_path / "views", tmp_path / "board" / "board.json", output)
    assert not output.exists()


def test_calibration_gates_and_resolution(tmp_path, synthetic_calibration):
    path = tmp_path / "synthetic_test_only.json"
    write_json(path, synthetic_calibration)
    calibration = load_calibration(path)
    with pytest.raises(ValueError, match="resolution"):
        prepare_image(np.zeros((360, 640, 3), np.uint8), calibration)
    calibration["quality_pass"] = False
    write_json(tmp_path / "failed.json", calibration)
    with pytest.raises(ValueError, match="quality gate"):
        load_calibration(tmp_path / "failed.json")
    calibration["quality_pass"] = True
    calibration["rms_px"] = 99
    write_json(tmp_path / "false_flag.json", calibration)
    with pytest.raises(ValueError, match="statistics"):
        load_calibration(tmp_path / "false_flag.json")


def test_prepare_preserves_matching_intrinsics(synthetic_calibration):
    frame = np.zeros((720, 1280, 3), np.uint8)
    corrected, valid, info = prepare_image(frame, synthetic_calibration)
    assert corrected.shape == frame.shape
    assert valid.shape == frame.shape[:2]
    assert info["image_space"] == "undistorted"
    assert info["calibrated"]
    assert np.isfinite(info["camera_matrix"]).all()


def test_rendered_board_images_end_to_end(tmp_path, synthetic_calibration):
    """Exercise image detection + solve; all images are synthetic and live in pytest temp."""
    generate_board(tmp_path / "board")
    texture = read_image(tmp_path / "board" / "board.png")
    matrix = np.asarray(synthetic_calibration["camera_matrix"], dtype=float)
    world_corners = np.array(
        [[0, 0, 0], [0.175, 0, 0], [0.175, 0.125, 0], [0, 0.125, 0]], dtype=np.float32
    )
    texture_corners = np.array([[0, 0], [1399, 0], [1399, 999], [0, 999]], dtype=np.float32)
    rng = np.random.default_rng(9)
    for index in range(24):
        rotation = rng.uniform(-0.65, 0.65, 3)
        translation = np.array(
            [rng.uniform(-0.18, 0.05), rng.uniform(-0.13, 0.01), rng.uniform(0.4, 0.65)]
        )
        projected, _ = cv2.projectPoints(world_corners, rotation, translation, matrix, np.zeros(5))
        transform = cv2.getPerspectiveTransform(texture_corners, projected.reshape(4, 2))
        frame = cv2.warpPerspective(texture, transform, (1280, 720), borderValue=(255, 255, 255))
        write_image(tmp_path / "synthetic_views" / f"view_{index:02d}.png", frame)
    output = tmp_path / "synthetic_calibration.json"
    result = solve_directory(
        tmp_path / "synthetic_views", tmp_path / "board" / "board.json", output
    )
    assert result["quality_pass"]
    assert result["valid_views"] >= 20
    np.testing.assert_allclose(
        np.array(load_calibration(output)["camera_matrix"])[[0, 1], [0, 1]],
        matrix[[0, 1], [0, 1]],
        rtol=0.08,
    )
