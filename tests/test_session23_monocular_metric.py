"""Lesson 23: monocular relative-depth to metric-scale calibration."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.experiments.monocular_metric import (
    A_TRUE,
    B_TRUE,
    EXPERIMENT,
    GRID_X_MAX_M,
    GRID_X_MIN_M,
    GRID_Y_MAX_M,
    GRID_Y_MIN_M,
    HEIGHT_PX,
    MIN_INV_DEPTH_SPAN,
    N_VALUES,
    NAIVE_A,
    NAIVE_B,
    POLE_RADIUS_M,
    POLE_TOP_M,
    POLE_X_M,
    POLE_Y_M,
    REFERENCE_N,
    SIGMA_VALUES,
    WIDTH_PX,
    fit_inverse_affine,
    metric_from_relative,
    relative_from_depth,
    render_depth_map,
    run_experiment,
    sample_control_pixels,
)
from embodied_learning.experiments.pinhole_projection import (
    EYE,
    K_INTRINSIC,
    NEAR_PLANE_M,
    TARGET,
    look_at,
    unproject,
)
from embodied_learning.monocular_metric_demo import MonocularMetricDemo, load_replays


def test_depth_map_pixels_unproject_onto_scene_surfaces():
    rendered = render_depth_map()
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    rng = np.random.default_rng(11)
    flat = np.flatnonzero(valid.ravel())
    chosen = rng.choice(flat, size=2000, replace=False)
    rows, cols = np.unravel_index(chosen, valid.shape)
    pixels = np.column_stack([cols, rows]).astype(float)
    depths = depth.ravel()[chosen]
    rotation, translation = look_at(EYE, TARGET)
    world = unproject(pixels, depths, rotation, translation)
    is_pole = pole.ravel()[chosen]
    ground, on_pole = world[~is_pole], world[is_pole]
    assert len(ground) > 1000 and len(on_pole) > 10
    # Ground pixels land exactly on z=0 inside the lesson-22 grid extent.
    np.testing.assert_allclose(ground[:, 2], 0.0, atol=1e-9)
    assert ground[:, 0].min() >= GRID_X_MIN_M - 1e-9
    assert ground[:, 0].max() <= GRID_X_MAX_M + 1e-9
    assert ground[:, 1].min() >= GRID_Y_MIN_M - 1e-9
    assert ground[:, 1].max() <= GRID_Y_MAX_M + 1e-9
    # Pole pixels sit exactly on the cylinder surface within its height range.
    radius = np.hypot(on_pole[:, 0] - POLE_X_M, on_pole[:, 1] - POLE_Y_M)
    np.testing.assert_allclose(radius, POLE_RADIUS_M, atol=1e-9)
    assert on_pole[:, 2].min() >= -1e-9 and on_pole[:, 2].max() <= POLE_TOP_M + 1e-9


def test_depth_map_near_plane_and_field_of_view():
    rendered = render_depth_map()
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    assert valid.sum() < valid.size  # sky / outside-grid pixels stay invalid
    assert (depth[valid] > NEAR_PLANE_M).all()
    assert depth[valid].max() < 8.0
    assert pole.sum() > 0  # the solid pole is visible
    # Occlusion is per-ray: where a pole pixel also has a ground hit behind it,
    # the recorded (pole) depth must be the nearer intersection.
    rows, cols = np.nonzero(pole)
    homogeneous = np.stack([cols.astype(float), rows.astype(float), np.ones(len(rows))])
    rotation, _ = look_at(EYE, TARGET)
    rays = rotation.T @ (np.linalg.inv(K_INTRINSIC) @ homogeneous)
    with np.errstate(divide="ignore", invalid="ignore"):
        ground_s = np.where(rays[2] < -1e-12, -EYE[2] / rays[2], np.nan)
        ground_x = EYE[0] + ground_s * rays[0]
        ground_y = EYE[1] + ground_s * rays[1]
    behind = (
        np.isfinite(ground_s)
        & (ground_x >= GRID_X_MIN_M)
        & (ground_x <= GRID_X_MAX_M)
        & (ground_y >= GRID_Y_MIN_M)
        & (ground_y <= GRID_Y_MAX_M)
    )
    assert behind.sum() > 0  # the pole really hides ground behind it
    assert (depth[rows, cols][behind] < ground_s[behind]).all()


def test_relative_proxy_affine_ambiguity():
    rendered = render_depth_map()
    depth, valid = rendered["depth_m"], rendered["valid"]
    relative = relative_from_depth(depth, valid)
    inverse_depth = 1.0 / depth[valid]
    np.testing.assert_allclose(relative[valid], A_TRUE * inverse_depth + B_TRUE, atol=1e-12)
    # A different (a, b) pair describes the SAME geometry ...
    other_a, other_b = 2.7, -0.4
    other = np.full(depth.shape, np.nan)
    other[valid] = other_a * inverse_depth + other_b
    recovered = metric_from_relative(other, other_a, other_b, valid)
    np.testing.assert_allclose(recovered[valid], depth[valid], atol=1e-9)
    # ... but inverting it with the WRONG pair gives wrong metres.
    wrong = metric_from_relative(other, NAIVE_A, NAIVE_B, valid)
    assert np.abs(wrong[valid] - depth[valid]).max() > 0.3
    with pytest.raises(ValueError, match="offset"):
        metric_from_relative(np.array([[0.1]]), A_TRUE, B_TRUE, np.array([[True]]))


def test_fit_inverse_affine_hand_case_and_guards():
    # Hand case: r = 5*(1/Z) + 0.2 through u = 0.5 and 0.25.
    a, b = fit_inverse_affine(np.array([0.5, 0.25]), np.array([2.7, 1.45]))
    np.testing.assert_allclose([a, b], [A_TRUE, B_TRUE], atol=1e-12)
    with pytest.raises(ValueError, match="two"):
        fit_inverse_affine(np.array([0.5]), np.array([2.7]))
    with pytest.raises(ValueError, match="span"):
        fit_inverse_affine(np.array([0.5, 0.5]), np.array([2.7, 2.7]))


def test_noiseless_calibration_recovers_truth():
    rendered = render_depth_map()
    depth, valid = rendered["depth_m"], rendered["valid"]
    rng = np.random.default_rng(5)
    _, depths = sample_control_pixels(depth, valid, 5, rng)
    a, b = fit_inverse_affine(1.0 / depths, A_TRUE / depths + B_TRUE)
    np.testing.assert_allclose([a, b], [A_TRUE, B_TRUE], atol=1e-9)
    z_hat = metric_from_relative(relative_from_depth(depth, valid), a, b, valid)
    np.testing.assert_allclose(z_hat[valid], depth[valid], atol=1e-9)


def test_control_sampling_span_and_reproducibility():
    rendered = render_depth_map()
    depth, valid = rendered["depth_m"], rendered["valid"]
    for count in (2, 10):
        pixels, depths = sample_control_pixels(depth, valid, count, np.random.default_rng(0))
        assert pixels.shape == (count, 2)
        assert np.ptp(1.0 / depths) >= MIN_INV_DEPTH_SPAN
        again = sample_control_pixels(depth, valid, count, np.random.default_rng(0))
        np.testing.assert_array_equal(pixels, again[0])
        np.testing.assert_array_equal(depths, again[1])
    with pytest.raises(ValueError):
        sample_control_pixels(depth, valid, 1, np.random.default_rng(0))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("monocular") / "result"
    report = run_experiment(output, runs=4, seed=0)
    return output, report


def test_run_experiment_guards_and_contract(recording):
    output, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["schema_version"] == 1
    assert report["canvas_px"] == [WIDTH_PX, HEIGHT_PX]
    assert report["a_true"] == A_TRUE
    assert report["b_true"] == B_TRUE
    assert report["n_values"] == list(N_VALUES)
    assert report["sigma_values_relative"] == list(SIGMA_VALUES)
    assert report["perfect_prior_max_abs_error_m"] < 1e-12
    assert report["naive_mean_abs_error_m"] > 1.0  # assuming a=1, b=0 is wrong by metres
    assert np.array(report["fit_failures"]).sum() == 0
    assert (output / "comparison.png").exists()
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=3, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "other", runs=1, seed=0)


def test_baseline_and_group_ordering(recording):
    _, report = recording
    mean = np.array(report["group_mean_abs_error_m"])
    assert (mean[:, 0] < 1e-9).all()  # sigma=0: the affine proxy is recovered exactly
    assert (mean[:, 2] > mean[:, 1]).all()  # 3% noise hurts more than 1%
    assert mean[3, 1] < mean[0, 1] and mean[3, 2] < mean[0, 2]  # N=10 beats N=2
    assert mean.max() < report["naive_mean_abs_error_m"]  # calibrated beats uncalibrated


def test_error_stratification_and_sigma_linearity(recording):
    _, report = recording
    mean = np.array(report["group_mean_abs_error_m"])
    near = np.array(report["group_near_mean_abs_error_m"])
    far = np.array(report["group_far_mean_abs_error_m"])
    signed = np.abs(np.array(report["group_mean_signed_error_m"]))
    # delta Z ~ Z^2 * delta(1/Z): the far layer must be clearly worse.
    assert (far[:, 2] > 2.0 * near[:, 2]).all()
    # More reading noise hurts every N.
    assert (mean[:, 2] > mean[:, 1]).all()
    # The mean-to-mean sigma ratio is deliberately NOT asserted here: with a
    # handful of seeds it is dominated by unlucky control-point draws (lesson
    # notes report this). Linear scaling is instead verified exactly by the
    # sandwich covariance matching at BOTH sigma values in
    # test_mechanism_covariance_matches_prediction.
    # OLS on r ~ a*(1/Z)+b puts noise on the dependent side: no big bias.
    assert (signed[:, 2] < mean[:, 2]).all()


def test_mechanism_covariance_matches_prediction(recording):
    _, report = recording
    ratio = np.array(report["mechanism_ratio_median"])
    var_a = np.array(report["mechanism_var_a_ratio"])
    var_b = np.array(report["mechanism_var_b_ratio"])
    sigma_cells = slice(1, len(SIGMA_VALUES))  # sigma=0 has no spread to compare
    assert np.isfinite(ratio[:, sigma_cells]).all()
    assert (np.abs(ratio[:, sigma_cells] - 1.0) < 0.15).all()
    assert (np.abs(var_a[:, sigma_cells] - 1.0) < 0.25).all()
    assert (np.abs(var_b[:, sigma_cells] - 1.0) < 0.25).all()


def test_npz_arrays_and_reference_group(recording):
    output, _ = recording
    with np.load(output / "trajectories.npz", allow_pickle=False) as data:
        depth = data["depth_map_m"]
        valid = data["valid_mask"]
        relative = data["relative_map"]
        ref_hat = data["reference_z_hat_m"]
        ref_error = data["reference_error_m"]
        ctrl = data["reference_ctrl_pixels"]
        predicted = data["mechanism_predicted_std_m"]
        empirical = data["mechanism_empirical_std_m"]
        failures = data["fit_failures"]
    assert depth.shape == (HEIGHT_PX, WIDTH_PX) == relative.shape
    assert np.isfinite(depth[valid]).all() and np.isnan(depth[~valid]).all()
    np.testing.assert_allclose(relative[valid], A_TRUE / depth[valid] + B_TRUE, atol=1e-12)
    np.testing.assert_allclose(ref_error[valid], ref_hat[valid] - depth[valid], atol=1e-12)
    assert np.abs(ref_error[valid]).mean() > 1e-6  # the calibrated map is not the truth
    assert ctrl.shape == (REFERENCE_N, 2)
    assert (predicted[valid] > 0).all() and (empirical[valid] > 0).all()
    assert failures.shape == (len(N_VALUES), len(SIGMA_VALUES))


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=3, seed=9)
    path = output / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["depth_map_m"][0, 0] = 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)
    bad = output / "summary.json"
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["a_true"] = 42.0
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(tmp_path):
    import tkinter as tk

    output = tmp_path / "recording"
    run_experiment(output, runs=2, seed=1)
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = MonocularMetricDemo(root, data)
    root.update()
    assert demo.mode.get() == "maps"
    assert demo.fig is not None
    assert "无标定" in demo.stats.cget("text")
    demo.mode.set("fit")
    demo.redraw()
    assert "拟合" in demo.stats.cget("text")
    demo.mode.set("error")
    demo.redraw()
    assert "误差" in demo.stats.cget("text")
    demo.close()
