"""Lesson 25: camera intrinsics from a synthetic chessboard (minimal Zhang)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.experiments.camera_intrinsics import (
    BOARD_COLS,
    BOARD_ROWS,
    CX_PX,
    CY_PX,
    EXPERIMENT,
    FOCAL_PX,
    HEIGHT_PX,
    M_VALUES,
    MIN_NORMAL_ANGLE_DEG,
    NAIVE_K,
    PROBE_POINT_M,
    REFERENCE_M,
    REFERENCE_SIGMA_PX,
    SIGMA_VALUES_PX,
    SQUARE_M,
    WIDTH_PX,
    board_corners,
    build_degenerate_views,
    build_orbit_views,
    build_pose_pool,
    decompose_extrinsics,
    homography_dlt,
    intrinsic_from_homographies,
    intrinsic_matrix,
    plane_normal_camera,
    pool_poses,
    project_points,
    reprojection_rms,
    reprojection_rms_known_pose,
    roundtrip_error,
    run_experiment,
    sample_pose_indices,
)
from embodied_learning.experiments.pinhole_projection import K_INTRINSIC, look_at
from embodied_learning.intrinsics_demo import IntrinsicsDemo, load_replays


def test_board_corners_layout():
    corners = board_corners()
    assert corners.shape == (BOARD_COLS * BOARD_ROWS, 3)
    assert (corners[:, 2] == 0.0).all()  # the board IS its own plane frame
    xs = np.sort(np.unique(corners[:, 0]))
    ys = np.sort(np.unique(corners[:, 1]))
    assert len(xs) == BOARD_COLS and len(ys) == BOARD_ROWS
    np.testing.assert_allclose(np.diff(xs), SQUARE_M, atol=1e-12)
    np.testing.assert_allclose(np.diff(ys), SQUARE_M, atol=1e-12)
    assert abs(xs[0] + xs[-1]) < 1e-12 and abs(ys[0] + ys[-1]) < 1e-12  # centred


def test_homography_dlt_hand_case_and_exact_view():
    # Hand case: identity mapping through four unit-square corners.
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    identity = homography_dlt(square, square)
    np.testing.assert_allclose(identity, np.eye(3), atol=1e-9)
    # Scaled and shifted mapping: pixel = (2X + 10, 3Y + 20).
    truth = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]])
    pixels = (truth @ np.column_stack([square, np.ones(4)]).T).T
    fitted = homography_dlt(square, pixels[:, :2])
    np.testing.assert_allclose(fitted, truth / truth[2, 2], atol=1e-9)
    with pytest.raises(ValueError, match="four"):
        homography_dlt(square[:3], square[:3])
    # Exact synthetic view: the fitted H reproduces all 54 corner pixels.
    corners = board_corners()
    pool_eye, pool_target, _ = build_pose_pool(corners)
    rotation, translation = look_at(pool_eye[0], pool_target[0])
    pixels = project_points(corners, rotation, translation, K_INTRINSIC)
    homography = homography_dlt(corners[:, :2], pixels)
    mapped = (homography @ np.column_stack([corners[:, :2], np.ones(54)]).T).T
    np.testing.assert_allclose(mapped[:, :2] / mapped[:, 2:3], pixels, atol=1e-8)


def test_noiseless_calibration_recovers_k_and_poses():
    corners = board_corners()
    pool_eye, pool_target, _ = build_pose_pool(corners)
    views = (0, 8, 16, 1, 9)  # spans all three tilt bands
    poses = pool_poses(pool_eye, pool_target, views)
    pixels = [
        project_points(corners, rotation, translation, K_INTRINSIC)
        for rotation, translation in poses
    ]
    homographies = [homography_dlt(corners[:, :2], view) for view in pixels]
    estimate = intrinsic_from_homographies(homographies)
    assert abs(estimate["f"] - FOCAL_PX) < 1e-6
    assert abs(estimate["cx"] - CX_PX) < 1e-6
    assert abs(estimate["cy"] - CY_PX) < 1e-6
    assert estimate["rank"] == 3
    # The per-view poses decompose back to the true look-at extrinsics.
    for homography, (rotation, translation) in zip(homographies, poses):
        decomposed = decompose_extrinsics(homography, K_INTRINSIC)
        np.testing.assert_allclose(decomposed["rotation"], rotation, atol=1e-9)
        np.testing.assert_allclose(decomposed["translation"], translation, atol=1e-9)
        assert decomposed["orthogonality_error"] < 1e-9
    # Known-pose reprojection and the probe round trip are numerical zeros.
    assert reprojection_rms_known_pose(corners, poses, K_INTRINSIC, pixels) < 1e-8
    assert (
        roundtrip_error(
            PROBE_POINT_M,
            poses,
            [decompose_extrinsics(h, K_INTRINSIC) for h in homographies],
            K_INTRINSIC,
        )
        < 1e-12
    )


def test_wrong_k_is_visible_only_outside_the_homography_self_consistency():
    corners = board_corners()
    pool_eye, pool_target, _ = build_pose_pool(corners)
    poses = pool_poses(pool_eye, pool_target, (0, 8, 16, 1, 9))
    rng = np.random.default_rng(3)
    pixels = [
        project_points(corners, rotation, translation, K_INTRINSIC)
        + rng.normal(0.0, REFERENCE_SIGMA_PX, (54, 2))
        for rotation, translation in poses
    ]
    homographies = [homography_dlt(corners[:, :2], view) for view in pixels]
    estimate = intrinsic_from_homographies(homographies)
    k_est = intrinsic_matrix(estimate["f"], estimate["cx"], estimate["cy"])
    # With the TRUE poses held fixed, a wrong K shows up in full ...
    naive_rms = reprojection_rms_known_pose(corners, poses, NAIVE_K, pixels)
    est_rms = reprojection_rms_known_pose(corners, poses, k_est, pixels)
    true_rms = reprojection_rms_known_pose(corners, poses, K_INTRINSIC, pixels)
    assert naive_rms > 3.0 * est_rms  # the guessed K is clearly worse
    assert est_rms < 0.5 * naive_rms
    assert true_rms < 1.5 * np.sqrt(2.0) * REFERENCE_SIGMA_PX
    # ... but re-projecting through the *decomposed* poses is self-consistent
    # for ANY K (the pose absorbs the K error) - the honest pitfall.
    decomposed_naive = reprojection_rms(
        corners, [decompose_extrinsics(h, NAIVE_K) for h in homographies], NAIVE_K, pixels
    )
    decomposed_est = reprojection_rms(
        corners, [decompose_extrinsics(h, k_est) for h in homographies], k_est, pixels
    )
    assert abs(decomposed_naive - decomposed_est) < 0.2 * decomposed_est


def test_degenerate_views_trigger_rank_guard():
    corners = board_corners()
    for builder in (build_degenerate_views, build_orbit_views):
        _, pixels = builder(corners)
        homographies = [homography_dlt(corners[:, :2], view) for view in pixels]
        with pytest.raises(ValueError, match="Degenerate") as info:
            intrinsic_from_homographies(homographies)
        assert info.value.rank < 3
        assert info.value.cond > 1e12
        # The forced minimum-norm solution does not even yield a valid K.
        with pytest.raises(ValueError, match="Ill-conditioned"):
            intrinsic_from_homographies(homographies, check_rank=False)


def test_pose_sampler_requires_normal_diversity():
    corners = board_corners()
    pool_eye, pool_target, _ = build_pose_pool(corners)
    poses = pool_poses(pool_eye, pool_target, range(len(pool_eye)))
    normals = np.asarray([plane_normal_camera(rotation) for rotation, _ in poses])
    indices, tries = sample_pose_indices(normals, REFERENCE_M, np.random.default_rng(0))
    unit = normals[indices] / np.linalg.norm(normals[indices], axis=1, keepdims=True)
    spread = np.rad2deg(np.arccos(np.clip(unit @ unit.T, -1.0, 1.0)).max())
    assert spread >= MIN_NORMAL_ANGLE_DEG
    assert len(set(indices)) == REFERENCE_M
    repeat = sample_pose_indices(normals, REFERENCE_M, np.random.default_rng(0))
    np.testing.assert_array_equal(indices, repeat[0])
    assert tries == repeat[1]
    with pytest.raises(ValueError):
        sample_pose_indices(normals, 1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_pose_indices(normals, len(normals) + 1, np.random.default_rng(0))
    # A translation-only pool (identical normals) can never pass the check.
    flat = np.tile(plane_normal_camera(poses[0][0]), (len(normals), 1))
    with pytest.raises(RuntimeError, match="parallel"):
        sample_pose_indices(flat, 2, np.random.default_rng(0))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("intrinsics") / "result"
    report = run_experiment(output, runs=4, seed=0)
    return output, report


def test_run_experiment_guards_and_contract(recording):
    output, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["schema_version"] == 1
    assert report["canvas_px"] == [WIDTH_PX, HEIGHT_PX]
    assert report["k_true"] == K_INTRINSIC.tolist()
    assert report["m_values"] == list(M_VALUES)
    assert report["sigma_values_px"] == list(SIGMA_VALUES_PX)
    assert report["board"]["corners_total"] == BOARD_COLS * BOARD_ROWS
    assert report["sigma_zero"]["f_error_px"] < 1e-6
    assert report["sigma_zero"]["cx_error_px"] < 1e-6
    assert report["sigma_zero"]["cy_error_px"] < 1e-6
    assert report["pose_pool"]["kept"] >= max(M_VALUES)
    assert report["nonlinear_refinement_done"] is False
    assert report["degenerate_case"]["parallel_translation"]["guard_triggered"]
    assert report["degenerate_case"]["parallel_orbit_fixed_tilt"]["guard_triggered"]
    assert (output / "comparison.png").exists()
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=2, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "other", runs=1, seed=0)


def test_sweep_ordering_and_noise_floor(recording):
    _, report = recording
    f_err_median = np.array(report["k_est_f_err_median_px"])
    f_err_mean = np.array(report["k_est_f_err_mean_px"])
    assert (f_err_mean[:, 0] < 1e-9).all()  # sigma = 0 is numerically exact
    # More images help in the mean (medians are robust to bad pose draws).
    assert f_err_median[-1, 2] < f_err_median[0, 2]  # M=20 beats M=2 at sigma=1
    assert f_err_median[-1, 3] < f_err_median[0, 3]  # ... and at sigma=2
    reproj_naive = np.array(report["reproj_rms_naive_px"])
    reproj_est = np.array(report["reproj_rms_est_px"])
    reproj_true = np.array(report["reproj_rms_true_px"])
    assert (reproj_naive > reproj_est).all()  # the guessed K is always worse...
    ratio = reproj_naive / np.maximum(reproj_est, 1e-9)
    assert float(np.median(ratio[:, 1:])) > 3.0  # ... by 3x or more in noisy cells
    # The true-K reprojection sits exactly on the 2D noise floor sqrt(2)*sigma.
    for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
        if sigma == 0.0:
            assert reproj_true[:, sigma_index].max() < 1e-8
        else:
            np.testing.assert_allclose(reproj_true[:, sigma_index], np.sqrt(2.0) * sigma, rtol=0.05)
    assert np.array(report["fit_failures"]).sum() == 0


def test_f_error_grows_with_sigma(recording):
    _, report = recording
    f_err_median = np.array(report["k_est_f_err_median_px"])
    row = f_err_median[3]  # M = 10
    assert row[3] > row[2] > row[1] > 0.0  # sigma = 2 > 1 > 0.5 px
    assert 1.2 < row[3] / row[2] < 3.4  # roughly linear in sigma


def test_mechanism_c_is_approximately_constant(recording):
    _, report = recording
    c_table = np.array(report["mechanism_c_px"])
    assert np.isnan(c_table[:, 0]).all()  # sigma = 0 has no spread to compare
    fixed_sigma = c_table[:, 1]  # sigma = 0.5 px: least heavy-tailed column
    assert np.nanmax(fixed_sigma) / np.nanmin(fixed_sigma) < 4.0
    pooled = float(np.nanmedian(c_table[:, 1:]))
    assert 20.0 < pooled < 300.0  # C ~ 60 px for this board/camera geometry


def test_seed_determinism(tmp_path):
    first = run_experiment(tmp_path / "first", runs=2, seed=7)
    second = run_experiment(tmp_path / "second", runs=2, seed=7)
    # mechanism_c_px holds NaNs (sigma = 0), so compare via JSON text form.
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    digest_first = hashlib.sha256(
        (tmp_path / "first" / "trajectories.npz").read_bytes()
    ).hexdigest()
    digest_second = hashlib.sha256(
        (tmp_path / "second" / "trajectories.npz").read_bytes()
    ).hexdigest()
    assert digest_first == digest_second


def test_npz_arrays_and_reference_bundle(recording):
    output, report = recording
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        corners = data["board_corners_m"]
        ref_rotation = data["ref_rotation"]
        ref_translation = data["ref_translation"]
        clean = data["ref_pixels_clean"]
        noisy = data["ref_pixels_noisy"]
        k_est = data["ref_k_est"]
        m_values = data["m_values"]
        f_err = data["k_est_f_err_mean_px"]
    assert corners.shape == (54, 3)
    assert (
        f_err.shape
        == (len(M_VALUES), len(SIGMA_VALUES_PX))
        == (np.array(report["k_est_f_mean_px"]).shape)
    )
    assert np.array_equal(m_values, np.array(M_VALUES))
    assert ref_rotation.shape == (REFERENCE_M, 3, 3)
    for view in range(REFERENCE_M):
        np.testing.assert_allclose(ref_rotation[view] @ ref_rotation[view].T, np.eye(3), atol=1e-12)
    assert ref_translation.shape == (REFERENCE_M, 3)
    sigma_hat = float(np.std(noisy - clean))
    assert 0.6 * REFERENCE_SIGMA_PX < sigma_hat < 1.6 * REFERENCE_SIGMA_PX
    assert abs(k_est[0] - FOCAL_PX) < 60.0  # the stored estimate is not the truth


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=2, seed=9)
    path = output / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # Tamper the summary first (npz still intact).
    summary_path = output / "summary.json"
    original = summary_path.read_text(encoding="utf-8")
    data = json.loads(original)
    data["f_true_px"] = 42.0
    summary_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)
    # Restore the summary, then tamper the archive.
    summary_path.write_text(original, encoding="utf-8")
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["k_est_f_err_mean_px"][0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(tmp_path):
    import tkinter as tk

    output = tmp_path / "recording"
    run_experiment(output, runs=2, seed=1)
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = IntrinsicsDemo(root, data)
    root.update()
    assert demo.mode.get() == "poses"
    assert demo.fig is not None
    assert "姿态" in demo.stats.cget("text")
    demo.mode.set("reproj")
    demo.redraw()
    assert "重投影" in demo.stats.cget("text")
    demo.mode.set("converge")
    demo.redraw()
    assert "收敛" in demo.stats.cget("text")
    demo.close()
