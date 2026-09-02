import cv2
import numpy as np
import pytest

from monocular_depth.geometry import backproject, export_cloud, geometry_from_record, write_ply
from monocular_depth.io import save_result
from monocular_depth.records import read_json, write_json


def test_backprojection_axes_color_and_invalid_filter():
    depth = np.array([[1, 2, np.nan], [0, -1, 8]], dtype=np.float32)
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[:] = [10, 20, 30]
    matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
    points, colors = backproject(depth, rgb, matrix, stride=1, max_depth=5)
    np.testing.assert_allclose(points, [[0, 0, 1], [1, 0, 2]])
    np.testing.assert_allclose(colors, [[30 / 255, 20 / 255, 10 / 255]] * 2)
    with pytest.raises(ValueError):
        backproject(depth, rgb, np.zeros((3, 3)))


@pytest.mark.parametrize("variant", ["relative", "metric"])
def test_uncalibrated_pointcloud_refused(tmp_path, variant):
    paths = save_result(
        tmp_path,
        "frame",
        np.zeros((4, 4, 3), np.uint8),
        np.ones((4, 4), np.float32),
        variant=variant,
    )
    with pytest.raises(ValueError, match="relative|calibration"):
        geometry_from_record(paths[-1])
    with pytest.raises(FileExistsError):
        save_result(tmp_path, "frame", np.zeros((4, 4, 3), np.uint8), np.ones((4, 4)))
    record = read_json(paths[-1])
    assert record["unit"] == ("m" if variant == "metric" else "relative")
    assert not record["metric_accuracy_validated"]


def synthetic_record(tmp_path):
    # Explicit synthetic fixture, not a real calibration artifact.
    matrix = np.array([[80.0, 0, 32], [0, 80, 24], [0, 0, 1]])
    corrected, _ = cv2.getOptimalNewCameraMatrix(matrix, np.zeros(5), (64, 48), 0)
    calibration = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": [0.0] * 5,
        "image_size": [64, 48],
        "source": "charuco_images",
        "quality_pass": True,
        "rms_px": 0.1,
        "valid_views": 12,
        "per_view_rms_px": [0.1] * 12,
    }
    return save_result(
        tmp_path,
        "SYNTHETIC_TEST_ONLY",
        np.full((48, 64, 3), 127, np.uint8),
        np.full((48, 64), 2, np.float32),
        variant="metric",
        metadata={
            "calibrated": True,
            "image_space": "undistorted",
            "camera_matrix": corrected.tolist(),
            "calibration": calibration,
        },
    )[-1]


def test_synthetic_plane_export_and_unicode_roundtrip(tmp_path):
    import open3d as o3d

    metadata = synthetic_record(tmp_path)
    output = tmp_path / "合成平面.ply"
    report = export_cloud(metadata, output, stride=1, fit_plane=True)
    assert report["points_exported"] == 64 * 48
    assert report["dominant_plane"]["inlier_ratio"] == 1
    assert report["dominant_plane"]["inlier_rms_m_predicted"] < 1e-6
    assert report["world_pose"] is None
    # Open3D's reader can be platform-specific for Unicode; our ASCII PLY writer is not.
    ascii_path = tmp_path / "roundtrip.ply"
    points, colors, _ = geometry_from_record(metadata, stride=1)
    write_ply(ascii_path, points, colors)
    restored = o3d.io.read_point_cloud(str(ascii_path))
    assert len(restored.points) == len(points)
    assert "end_header" in output.read_text(encoding="ascii")
    with pytest.raises(FileExistsError):
        export_cloud(metadata, output)


def test_mismatched_undistortion_and_traversal_refused(tmp_path):
    path = synthetic_record(tmp_path)
    record = read_json(path)
    record["camera_matrix"][0][0] *= 2
    write_json(tmp_path / "bad_matrix.json", record)
    with pytest.raises(ValueError, match="undistortion"):
        geometry_from_record(tmp_path / "bad_matrix.json")
    record = read_json(path)
    record["depth_file"] = "../other.npy"
    write_json(tmp_path / "bad_path.json", record)
    with pytest.raises(ValueError, match="adjacent"):
        geometry_from_record(tmp_path / "bad_path.json")


def test_recorded_calibrated_focus_is_required_and_must_match(tmp_path):
    path = synthetic_record(tmp_path)
    record = read_json(path)
    record["calibration"]["focus_lock_verified_for_accepted_frames"] = True
    record["calibration"]["fixed_focus_driver_units"] = 293
    missing = tmp_path / "missing_focus.json"
    write_json(missing, record)
    with pytest.raises(ValueError, match="verified capture focus"):
        geometry_from_record(missing)
    state = {"state": "LOCKED", "target": 293, "focus": 293, "autofocus": 0}
    record.update(camera_focus_verified=True, focus_before_frame=state, focus_after_frame=state)
    valid = tmp_path / "valid_focus.json"
    write_json(valid, record)
    assert len(geometry_from_record(valid)[0]) > 3
    record["calibration"]["fixed_focus_driver_units"] = 286
    mismatch = tmp_path / "mismatched_focus.json"
    write_json(mismatch, record)
    with pytest.raises(ValueError, match="does not match calibration"):
        geometry_from_record(mismatch)
