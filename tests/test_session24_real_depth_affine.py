"""Lesson 24: affine check of the real relative-depth output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_learning.experiments.monocular_metric import (
    A_TRUE,
    B_TRUE,
    MIN_INV_DEPTH_SPAN,
    render_depth_map,
)
from embodied_learning.experiments.pinhole_projection import HEIGHT_PX, WIDTH_PX
from embodied_learning.experiments.real_depth_affine import (
    EXPERIMENT,
    N_VALUES,
    REFERENCE_N,
    digest,
    load_bench_npz,
    run_experiment,
)

# Known NON-affine distortion for the synthetic check field: a horizontal
# image-position term (an affine law in 1/Z cannot absorb it) plus quadratic
# curvature in 1/Z. Magnitudes are chosen so the structure is well above the
# numerical noise floor and clearly detectable by every reported statistic.
DISTORT_SPATIAL = 0.5  # amplitude of 0.5 * (x/W - 0.5)
DISTORT_CURVATURE = 1.5  # amplitude of 1.5 * (1/Z - 1/3)^2
RNG_SEED = 0


def distorted_relative(depth, valid):
    _, cols = np.nonzero(valid)
    u = 1.0 / depth[valid]
    return (
        A_TRUE * u
        + B_TRUE
        + DISTORT_SPATIAL * (cols / WIDTH_PX - 0.5)
        + DISTORT_CURVATURE * (u - 1.0 / 3.0) ** 2
    )


def write_bench_npz(path, depth, valid, pole, relative):
    rng = np.random.default_rng(7)
    np.savez_compressed(
        path,
        depth_m=depth,
        valid=valid,
        pole=pole,
        rgb=rng.integers(0, 255, size=(HEIGHT_PX, WIDTH_PX, 3), dtype=np.uint8),
        r_pred=relative,
        meta_json=np.array(json.dumps({"model": "synthetic_check", "input_size": 518})),
    )


@pytest.fixture(scope="module")
def bench_inputs(tmp_path_factory):
    rendered = render_depth_map()
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    u = 1.0 / depth[valid]
    ideal = np.full(depth.shape, np.nan)
    ideal[valid] = A_TRUE * u + B_TRUE
    distorted = np.full(depth.shape, np.nan)
    distorted[valid] = distorted_relative(depth, valid)
    root = tmp_path_factory.mktemp("bench")
    write_bench_npz(root / "ideal.npz", depth, valid, pole, ideal)
    write_bench_npz(root / "distorted.npz", depth, valid, pole, distorted)
    return {"ideal": root / "ideal.npz", "distorted": root / "distorted.npz"}


@pytest.fixture(scope="module")
def ideal_run(bench_inputs, tmp_path_factory):
    output = tmp_path_factory.mktemp("ideal") / "result"
    return output, run_experiment(bench_inputs["ideal"], output, runs=3, seed=RNG_SEED)


@pytest.fixture(scope="module")
def distorted_run(bench_inputs, tmp_path_factory):
    output = tmp_path_factory.mktemp("distorted") / "result"
    return output, run_experiment(bench_inputs["distorted"], output, runs=3, seed=RNG_SEED)


def test_load_bench_input_guards(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bench_npz(tmp_path / "missing.npz")
    depth = np.zeros((4, 4))
    meta = np.array(json.dumps({"model": "x"}))
    np.savez_compressed(
        tmp_path / "no_key.npz",
        depth_m=depth,
        valid=depth > 0,
        pole=depth > 0,
        rgb=np.zeros((4, 4, 3), np.uint8),
        meta_json=meta,
    )
    with pytest.raises(ValueError, match="missing keys"):
        load_bench_npz(tmp_path / "no_key.npz")
    np.savez_compressed(
        tmp_path / "bad_shape.npz",
        depth_m=depth,
        valid=depth > 0,
        pole=depth > 0,
        rgb=np.zeros((4, 4, 3), np.uint8),
        r_pred=depth,
        meta_json=meta,
    )
    with pytest.raises(ValueError, match="canvas"):
        load_bench_npz(tmp_path / "bad_shape.npz")
    good = {key: np.zeros((HEIGHT_PX, WIDTH_PX)) for key in ("depth_m", "r_pred")}
    good["valid"] = good["pole"] = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=bool)
    good["rgb"] = np.zeros((HEIGHT_PX, WIDTH_PX, 3), np.uint8)
    good["r_pred"][0, 0] = np.nan
    good["valid"][0, 0] = True
    np.savez_compressed(tmp_path / "bad_finite.npz", meta_json=meta, **good)
    with pytest.raises(ValueError, match="non-finite"):
        load_bench_npz(tmp_path / "bad_finite.npz")
    no_model = dict(good)
    no_model["r_pred"] = np.ones((HEIGHT_PX, WIDTH_PX))
    np.savez_compressed(
        tmp_path / "no_model.npz", meta_json=np.array(json.dumps({"input_size": 518})), **no_model
    )
    with pytest.raises(ValueError, match="model"):
        load_bench_npz(tmp_path / "no_model.npz")


def test_run_guards_output_and_runs(ideal_run, bench_inputs, tmp_path):
    output, _ = ideal_run
    with pytest.raises(FileExistsError):
        run_experiment(bench_inputs["ideal"], output, runs=2, seed=0)
    with pytest.raises(ValueError, match="two repetitions"):
        run_experiment(bench_inputs["ideal"], tmp_path / "fresh", runs=1, seed=0)


def test_ideal_field_is_recovered_exactly(ideal_run):
    output, report = ideal_run
    assert report["experiment"] == EXPERIMENT
    assert report["canvas_px"] == [WIDTH_PX, HEIGHT_PX]
    assert report["n_values"] == list(N_VALUES)
    dense = report["dense_fit"]
    # The dense all-pixel fit recovers the generating affine law ...
    np.testing.assert_allclose([dense["a"], dense["b"]], [A_TRUE, B_TRUE], atol=1e-6)
    assert dense["r_squared"] > 1.0 - 1e-12
    assert dense["residual_std"] < 1e-9
    assert report["dense_fit_proxy_sanity"] == pytest.approx({"a": A_TRUE, "b": B_TRUE}, abs=1e-6)
    # ... and the control-point protocol reproduces the depth map exactly.
    assert (np.array(report["real_group_mean_abs_error_m"]) < 1e-9).all()
    assert (np.array(report["real_group_max_abs_error_m"]) < 1e-9).all()
    assert (np.array(report["real_group_r2_ctrl"]) > 1.0 - 1e-9).all()
    assert sum(report["real_fit_failures"]) == 0
    assert report["dense_metric_error"]["mean_abs_error_m"] < 1e-9
    assert (output / "comparison.png").exists()


def test_ideal_scan_matches_paired_proxy_exactly(ideal_run):
    _, report = ideal_run
    # Same field, same paired control points: the two scans must agree bitwise.
    np.testing.assert_allclose(
        report["real_group_mean_abs_error_m"],
        report["proxy_group_mean_abs_error_m"],
        atol=1e-15,
    )
    assert (np.array(report["proxy_group_mean_abs_error_m"]) < 1e-12).all()


def test_distorted_dense_residual_structure_detected(distorted_run):
    _, report = distorted_run
    dense = report["dense_fit"]
    assert dense["r_squared"] < 0.99  # the affine law no longer explains everything
    assert dense["residual_std"] > 0.05
    assert dense["residual_max_abs"] > 0.1
    assert dense["bin_residual_range"] > 0.05  # residual vs 1/Z is not flat
    assert dense["residual_correlation_u_px"] > 0.5  # horizontal spatial structure
    assert abs(dense["residual_mean_left_half"] + dense["residual_mean_right_half"]) < 0.05
    assert dense["residual_mean_left_half"] * dense["residual_mean_right_half"] < 0.0


def test_distorted_calibration_error_against_ideal_bound(distorted_run):
    _, report = distorted_run
    real = np.array(report["real_group_mean_abs_error_m"])
    proxy = np.array(report["proxy_group_mean_abs_error_m"])
    assert sum(report["real_fit_failures"]) == 0
    # The non-affine residual does NOT vanish with more control points: even
    # N=20 stays orders of magnitude above the ideal paired upper bound.
    assert real[-1] > 0.05
    assert real[-1] > 1e6 * proxy[-1]
    # Metric errors stay stratified: delta Z ~ Z^2 * delta(1/Z) keeps far worse.
    assert (
        report["real_group_far_mean_abs_error_m"][-1]
        > (report["real_group_near_mean_abs_error_m"][-1])
    )
    assert report["dense_metric_error"]["mean_abs_error_m"] > 0.05


def input_valid_mask(report):
    with np.load(report["input_path"], allow_pickle=False) as data:
        return np.asarray(data["valid"], dtype=bool)


def test_analysis_npz_contract_arrays(distorted_run):
    output, report = distorted_run
    with np.load(output / "analysis.npz", allow_pickle=False) as data:
        residual = data["residual_map"]
        dense_error = data["dense_error_m"]
        ctrl = data["real_reference_ctrl_pixels"]
        ctrl_depth = data["real_reference_ctrl_depth_m"]
        fit_ab = data["real_reference_fit_ab"]
        bin_sizes = data["bin_sizes"]
        n_values = data["n_values"]
    valid = input_valid_mask(report)
    assert residual.shape == (HEIGHT_PX, WIDTH_PX)
    assert np.isfinite(residual[valid]).all() and np.isnan(residual[~valid]).all()
    assert np.isfinite(dense_error[valid]).all()
    assert ctrl.shape == (REFERENCE_N, 2)
    assert np.ptp(1.0 / ctrl_depth) >= MIN_INV_DEPTH_SPAN
    assert fit_ab.shape == (2,) and np.isfinite(fit_ab).all() and fit_ab[0] > 0
    assert bin_sizes.sum() == valid.sum()
    np.testing.assert_array_equal(n_values, N_VALUES)


def test_summary_hashes_detect_tampering(ideal_run, bench_inputs):
    output, report = ideal_run
    assert report["analysis_npz_sha256"] == digest(output / "analysis.npz")
    assert report["input_sha256"] == digest(bench_inputs["ideal"])
    assert report["trajectories_sha256"] == report["analysis_npz_sha256"]
    with np.load(output / "analysis.npz", allow_pickle=False) as data:
        arrays = dict(data)
    arrays["residual_map"][0, 0] = 5.0
    np.savez_compressed(output / "analysis.npz", **arrays)
    assert digest(output / "analysis.npz") != report["analysis_npz_sha256"]


def test_seed_reproducibility_identical_summaries(bench_inputs, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_a = run_experiment(bench_inputs["ideal"], first, runs=2, seed=3)
    run_experiment(bench_inputs["ideal"], second, runs=2, seed=3)
    repeated = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    assert report_a == repeated


def test_pole_split_and_structure_fields_present(distorted_run):
    _, report = distorted_run
    dense = report["dense_fit"]
    assert dense["pole_mean_abs_residual"] > 0.0
    assert dense["ground_mean_abs_residual"] > 0.0
    for key in (
        "residual_mean_top_half",
        "residual_mean_bottom_half",
        "residual_correlation_v_px",
        "residual_std_over_r_std",
    ):
        assert dense[key] is not None
    assert report["reference_group"]["n"] == REFERENCE_N
    assert len(report["reference_group"]["ctrl_r"]) == REFERENCE_N


def test_cli_end_to_end_subprocess(bench_inputs, tmp_path):
    source_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_root / "src")
    output = tmp_path / "cli_result"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.real_depth_affine",
            "--input",
            str(bench_inputs["ideal"]),
            "--output",
            str(output),
            "--runs",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(source_root),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for name in ("summary.json", "analysis.npz", "comparison.png"):
        assert (output / name).exists()
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["experiment"] == EXPERIMENT
    assert report["runs_per_group"] == 2
    assert (np.array(report["real_group_mean_abs_error_m"]) < 1e-9).all()
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "embodied_learning.experiments.real_depth_affine",
                "--input",
                str(bench_inputs["ideal"]),
                "--output",
                str(output),
                "--runs",
                "2",
            ],
            capture_output=True,
            env=env,
            cwd=str(source_root),
            check=True,
        )
