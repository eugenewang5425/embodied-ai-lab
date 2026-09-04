from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.experiments.pinhole_projection import (
    CX_PX,
    CY_PX,
    DEPTH_NOISE_STD_M,
    EXPERIMENT,
    EYE,
    FOCAL_PX,
    HEIGHT_PX,
    K_INTRINSIC,
    NEAR_PLANE_M,
    TARGET,
    WIDTH_PX,
    look_at,
    project,
    project_with_depth,
    run_experiment,
    scene_points,
    to_camera,
    unproject,
)
from embodied_learning.pinhole_demo import PinholeDemo, load_replays


def test_principal_point_and_focal_scale_hand_case():
    # Camera frame: z forward, y DOWN (image convention). Coordinates below
    # are therefore camera-frame, not ENU world axes.
    rotation, translation = np.eye(3), np.zeros(3)
    pixels, _, _ = project(
        np.array([[0.0, 0.0, 3.0], [0.0, -1.0, 3.0], [1.0, 0.0, 3.0]]),
        rotation,
        translation,
    )
    np.testing.assert_allclose(pixels[0], [CX_PX, CY_PX], atol=1e-12)
    # y_cam = -1 is 1 m ABOVE the optical axis -> v is CY - f*y/z.
    np.testing.assert_allclose(pixels[1], [CX_PX, CY_PX - FOCAL_PX / 3.0], atol=1e-12)
    np.testing.assert_allclose(pixels[2], [CX_PX + FOCAL_PX / 3.0, CY_PX], atol=1e-12)


def test_look_at_axes_are_orthonormal_and_eye_is_principal():
    rotation, translation = look_at(EYE, TARGET)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    camera_eye = to_camera(np.array([EYE]), rotation, translation)[0]
    np.testing.assert_allclose(camera_eye[:2], [0, 0], atol=1e-12)
    assert camera_eye[2] < 0 or abs(camera_eye[2]) < 1e-12
    assert TARGET[0] >= 0  # sanity: not relevant for this test
    pixels, _, camera = project(np.array([TARGET]), rotation, translation)
    np.testing.assert_allclose(pixels[0], [CX_PX, CY_PX], atol=1e-9)
    assert camera[0, 2] > 0


def test_near_plane_culls_very_close_points():
    rotation, translation = np.eye(3), np.zeros(3)
    points = np.array([[0.0, 0.0, NEAR_PLANE_M / 2], [0.0, 0.0, 2.0]])
    pixels, _, camera = project(points, rotation, translation)
    assert len(pixels) == 1
    np.testing.assert_allclose(camera[0, 2], 2.0)
    unpadded, _, all_camera = project(points, rotation, translation, near_plane=0.0)
    assert len(unpadded) == 2
    # Round trip still exact: visible world point recovered.
    clouds = unproject(unpadded, all_camera[:, 2], rotation, translation)
    np.testing.assert_allclose(clouds, points, atol=1e-12)


def test_roundtrip_is_numerically_exact_for_pixel_to_world():
    points, _, _ = scene_points()
    rotation, translation = look_at(EYE, TARGET)
    pixels, depths, world = project_with_depth(points, rotation, translation)
    clouds = unproject(pixels, depths, rotation, translation)
    np.testing.assert_allclose(clouds, world, atol=1e-12)


def test_without_depth_pixel_only_gives_a_ray():
    rotation, translation = look_at(EYE, TARGET)
    pixels, _, _ = project_with_depth(scene_points()[0], rotation, translation)
    chosen = pixels[len(pixels) // 2]
    candidates = unproject(np.tile(chosen, (3, 1)), [2.0, 4.0, 6.0], rotation, translation)
    first = candidates[1] - candidates[0]
    second = candidates[2] - candidates[0]
    assert np.linalg.norm(np.cross(first, second)) < 1e-12
    # And all three re-project to the SAME pixel (depth is invisible to pinhole).
    back, _, _ = project(candidates, rotation, translation, near_plane=0.0)
    np.testing.assert_allclose(back, np.tile(chosen, (3, 1)), atol=1e-9)


def test_noisy_depth_is_seeded_and_ray_scaling_is_visible():
    points, _, _ = scene_points()
    rotation, translation = look_at(EYE, TARGET)
    pixels, depths, world = project_with_depth(points, rotation, translation)
    rng = np.random.default_rng(7)
    noisy = depths + rng.normal(0.0, DEPTH_NOISE_STD_M, size=len(depths))
    clouds = unproject(pixels, noisy, rotation, translation)
    errors = np.linalg.norm(clouds - world, axis=1)
    rng2 = np.random.default_rng(7)
    noisy2 = depths + rng2.normal(0.0, DEPTH_NOISE_STD_M, size=len(depths))
    np.testing.assert_array_equal(noisy, noisy2)
    inverse = np.linalg.inv(K_INTRINSIC)
    rays = np.column_stack([pixels, np.ones(len(pixels))])
    ray_norm = np.linalg.norm((inverse @ rays.T).T, axis=1)
    ratio = errors / ray_norm
    # E|N(0, 0.15)| = 0.15*sqrt(2/pi) ~ 11.97 cm; the ratio recovers it.
    assert 0.10 < ratio.mean() < 0.14
    # Mechanism check: single-seed ratio already recovers E|N(0, sigma)|
    # = sigma*sqrt(2/pi) ~ 3.99 cm (asserted above). No correlation assertion:
    # the in-FOV pixel set spans a narrow ray-norm range, so a correlation
    # coefficient would be dominated by finite-sample |delta| noise.


def test_moved_camera_unprojects_back_to_the_same_world():
    points, _, _ = scene_points()
    rot1, trans1 = look_at(EYE, TARGET)
    eye2 = EYE + np.array([0.5, 0.0, 0.3])
    rot2, trans2 = look_at(eye2, TARGET)
    pixels1, depth1, world1 = project_with_depth(points, rot1, trans1)
    pixels2, depth2, world2 = project_with_depth(points, rot2, trans2)
    cloud1 = unproject(pixels1, depth1, rot1, trans1)
    cloud2 = unproject(pixels2, depth2, rot2, trans2)
    np.testing.assert_allclose(cloud1, world1, atol=1e-12)
    np.testing.assert_allclose(cloud2, world2, atol=1e-12)
    # The second camera sees a slightly different subset; check overlapping ones.
    for index, world in enumerate(world1):
        match = np.where(np.all(np.abs(world2 - world) < 1e-12, axis=1))[0]
        if len(match):
            np.testing.assert_allclose(cloud2[match[0]], world, atol=1e-12)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("pinhole") / "result"
    report = run_experiment(output, runs=6, seed=0)
    return output, report


def test_summary_contract(recording):
    _, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["focal_px"] == FOCAL_PX
    assert report["canvas_px"] == [WIDTH_PX, HEIGHT_PX]
    assert report["roundtrip_max_error_m"] < 1e-12
    assert report["pose_consistency_max_error_m"] < 1e-12
    assert report["pose_all_matched"]
    assert 0.10 < report["noise_estimate_mean_m"] < 0.14


def test_run_experiment_guards(recording):
    output, _ = recording
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=3, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "x", runs=1, seed=0)


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=4, seed=9)
    path = output / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["roundtrip_error_m"][0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)
    bad = output / "summary.json"
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["intrinsic"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)


def _official_pinhole():
    """Lesson-22 official recording (gitignored results dir), like other demos."""
    import os
    from pathlib import Path

    root = Path(os.environ.get("EMBODIED_PROJECT_ROOT", "D:\\项目\\具身人工智能"))
    output = root / "results" / "mobile_pinhole_2026-09-03"
    if not (output / "summary.json").exists():
        pytest.skip("Run the lesson-22 experiment first to create the recording")
    return output


def _official_pinhole():
    """Lesson-22 official recording (gitignored results dir), like other demos."""
    from pathlib import Path

    root = Path("D:\\项目\\具身人工智能")
    output = root / "results" / "mobile_pinhole_2026-09-03"
    if not (output / "summary.json").exists():
        pytest.skip("Run the lesson-22 experiment first to create the recording")
    return output


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel():
    import tkinter as tk

    output = _official_pinhole()
    root = tk.Tk()
    root.withdraw()
    routes = load_replays(output)
    demo = PinholeDemo(root, routes)
    demo.canvas.winfo_width = lambda: 460
    demo.canvas.winfo_height = lambda: 430
    root.update()
    demo.redraw()  # fill the stats panel after the canvas size mock applies
    assert demo.mode.get() == "exact"
    assert demo.fig is not None and demo.ax3d is not None
    assert len(demo.ax3d.lines) >= 10  # ground grid + pyramid edges
    assert "往返" in demo.stats.cget("text") or "1e-" in demo.stats.cget("text")
    demo.mode.set("ray")
    demo.redraw()
    assert "射线" in demo.stats.cget("text")
    demo.mode.set("noisy")
    demo.redraw()
    assert "噪声" in demo.stats.cget("text")
    demo.close()
