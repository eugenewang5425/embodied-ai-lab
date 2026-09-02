import numpy as np

from monocular_depth.io import read_image, write_image
from monocular_depth.model import comparison_view, normalize_depth


def test_normalize_depth_spans_uint8_range() -> None:
    depth = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    normalized = normalize_depth(depth)

    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_normalize_constant_depth_is_stable() -> None:
    depth = np.full((3, 4), 7.0, dtype=np.float32)

    assert np.count_nonzero(normalize_depth(depth)) == 0


def test_comparison_view_preserves_height_and_adds_both_panels() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    depth = np.arange(200, dtype=np.float32).reshape(10, 20)

    comparison = comparison_view(image, depth)

    assert comparison.shape == (10, 52, 3)


def test_image_round_trip_supports_unicode_path(tmp_path) -> None:
    image = np.full((5, 7, 3), 123, dtype=np.uint8)
    path = tmp_path / "深度图.png"

    write_image(path, image)
    restored = read_image(path)

    assert restored is not None
    np.testing.assert_array_equal(restored, image)
