"""Synthetic per-frame scale estimation tests."""

import cv2
import numpy as np
import pytest

from monocular_depth.calibration import generate_board
from monocular_depth.distance import plane_depth_map, region_mask
from monocular_depth.io import read_image
from monocular_depth.scaling import apply_scale, estimate_scale

MATRIX = np.array([[800.0, 0, 640], [0, 810, 360], [0, 0, 1]])
WIDTH, HEIGHT = 1280, 720
SHAPE = (HEIGHT, WIDTH)
POSE = {"rotation_vector": np.array([0.12, -0.08, 0.05]), "translation": np.array([0, 0, 0.55])}


def _plane():
    rotation, _ = cv2.Rodrigues(POSE["rotation_vector"])
    normal = rotation[:, 2]
    return {"normal": normal.tolist(), "plane_distance_m": float(normal @ POSE["translation"])}


def test_estimate_and_apply_scale(tmp_path):
    board_dir = tmp_path / "board"
    generate_board(board_dir)
    texture = read_image(board_dir / "board.png")
    corrected, _ = cv2.getOptimalNewCameraMatrix(MATRIX, np.zeros(5), (WIDTH, HEIGHT), 0)
    world = np.array([[0, 0, 0], [0.175, 0, 0], [0.175, 0.125, 0], [0, 0.125, 0]])
    tex_corners = np.array([[0, 0], [1399, 0], [1399, 999], [0, 999]], dtype=np.float32)
    projected, _ = cv2.projectPoints(
        world, POSE["rotation_vector"], POSE["translation"], corrected, np.zeros(5)
    )
    transform = cv2.getPerspectiveTransform(tex_corners, projected.reshape(4, 2).astype(np.float32))
    rgb = cv2.warpPerspective(texture, transform, (WIDTH, HEIGHT), borderValue=(255, 255, 255))
    true_z = plane_depth_map(_plane(), corrected, SHAPE)
    depth = (true_z / 0.8).astype(np.float32)  # model underestimates scale by 0.8
    info = estimate_scale(depth, rgb, corrected, {
        "squares_x": 7,
        "squares_y": 5,
        "square_length_m": 0.025,
        "marker_length_m": 0.018,
        "dictionary": "DICT_4X4_50",
        "legacy_pattern": False,
    })
    assert info is not None
    assert info["scale"] == pytest.approx(0.8, rel=0.02)
    assert info["mad"] < 0.02
    scaled = apply_scale(depth, info["scale"])
    mask = region_mask(projected.reshape(-1, 2), SHAPE, 0.12) > 0
    ratio = (true_z[mask] / scaled[mask])
    assert np.median(ratio) == pytest.approx(1.0, abs=0.02)


def test_no_board_returns_none(tmp_path):
    rgb = np.full((*SHAPE, 3), 128, np.uint8)
    depth = np.full(SHAPE, 0.7, np.float32)
    board = {"squares_x": 7, "squares_y": 5, "square_length_m": 0.025,
             "marker_length_m": 0.018, "dictionary": "DICT_4X4_50", "legacy_pattern": False}
    assert estimate_scale(depth, rgb, MATRIX, board) is None
