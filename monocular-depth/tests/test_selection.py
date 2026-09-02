import cv2
import numpy as np
import pytest

from monocular_depth.calibration import DEFAULT_BOARD, make_board
from monocular_depth.selection import coverage, heldout_errors, select_views


def synthetic_observations(count=18):
    rng = np.random.default_rng(293)
    matrix = np.array([[850.0, 0, 640], [0, 850.0, 360], [0, 0, 1]])
    obj = make_board(DEFAULT_BOARD).getChessboardCorners()
    objects, pixels = [], []
    for i in range(count):
        rotation = rng.uniform(-0.5, 0.5, 3)
        translation = [rng.uniform(-0.2, 0.1), rng.uniform(-0.16, 0.06), rng.uniform(0.5, 0.8)]
        img, _ = cv2.projectPoints(obj, rotation, np.array(translation), matrix, np.zeros(5))
        img += rng.normal(0, 1.6 if i == 5 else 0.05, img.shape).astype(np.float32)
        objects.append(obj.copy())
        pixels.append(img)
    return objects, pixels, (1280, 720)


def test_selection_preserves_geometry_inputs_and_minimum_count():
    objects, pixels, size = synthetic_observations()
    originals = [p.copy() for p in pixels]
    selected, frontier, policy = select_views(objects, pixels, size)
    assert len(selected["indices"]) >= 15
    assert [len(p["indices"]) for p in frontier] == list(range(18, 14, -1))
    assert selected["fit"]["quality_pass"]
    assert all(
        not p["fit"]["quality_pass"]
        for p in frontier
        if len(p["indices"]) > len(selected["indices"])
    )
    assert (
        coverage(pixels, selected["indices"], size) >= 0.9 * policy["initial_hull_image_fraction"]
    )
    for original, current in zip(originals, pixels, strict=True):
        np.testing.assert_array_equal(original, current)
    errors, squared = heldout_errors(objects, pixels, [0], selected["fit"])
    assert errors[0]["index"] == 0
    assert len(squared) == len(objects[0])
    assert np.isfinite(squared).all()


@pytest.mark.parametrize("minimum", [11, 19])
def test_selection_refuses_invalid_minimum(minimum):
    objects, pixels, size = synthetic_observations()
    with pytest.raises(ValueError):
        select_views(objects, pixels, size, min_views=minimum)
