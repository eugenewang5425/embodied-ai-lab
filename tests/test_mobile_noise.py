from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.experiments.mobile_noise import (
    EXPERIMENT,
    GROUPS,
    INPUT_RIGHT_SCALE,
    INTERVAL_NOISE_STD_RAD,
    calibrated_factor,
    noise_sequence,
    noisy_readings,
    run_experiment,
)
from embodied_learning.experiments.mobile_odometry import run_case
from embodied_learning.mobile_noise_demo import GROUP_KEYS, NoiseDemo, load_replays
from embodied_learning.odometry import estimate_poses


def test_calibrated_factor_matches_lesson16_independent_runs():
    factor, rows = calibrated_factor()
    assert factor == pytest.approx(1 / 1.02, abs=1e-12)
    np.testing.assert_allclose(
        [row["external_distance_m"] for row in rows], [0.4, 0.8, 1.2, -0.6], atol=1e-12
    )


def test_noisy_readings_chain_and_left_wheel_identity():
    true = np.array([[0.0, 0.0], [0.16, 0.16], [0.32, 0.32], [0.32, 0.48]])
    epsilon = np.array([0.001, -0.002, 0.0005])
    readings = noisy_readings(true, epsilon, 1.0)
    np.testing.assert_array_equal(readings[:, 0], true[:, 0])
    raw_right = np.diff(true[:, 1]) * INPUT_RIGHT_SCALE + epsilon
    np.testing.assert_allclose(np.diff(readings[:, 1]), raw_right)
    # epsilon=0 with c=1/1.02 recovers the true angles exactly.
    exact = noisy_readings(true, np.zeros(3), 1 / INPUT_RIGHT_SCALE)
    np.testing.assert_allclose(exact, true, atol=1e-15)
    # Noise is not a scale: a different sequence changes increments, not the scale.
    other = noisy_readings(true, np.zeros(3), 1.0)
    np.testing.assert_allclose(other[:, 0], true[:, 0])
    np.testing.assert_allclose(other[:, 1], true[:, 1] * INPUT_RIGHT_SCALE)


@pytest.mark.parametrize(
    "angles, epsilon, factor",
    [
        (np.zeros((3, 2)), np.zeros(3), 0),
        (np.zeros((3, 2)), np.ones(2), 1.0),
        (np.zeros((3, 2)), np.zeros(3), np.nan),
        (np.array([[0.0, 0.0], [np.nan, 0.2]]), np.zeros(1), 1.0),
        (np.zeros((3, 2)), np.array([0.0, np.inf, 0.0]), 1.0),
    ],
)
def test_invalid_noisy_readings_rejected(angles, epsilon, factor):
    with pytest.raises(ValueError):
        noisy_readings(angles, factor, epsilon)


def test_noise_is_seeded_independent_and_zero_mean():
    first = noise_sequence(np.random.default_rng(11), 400)
    second = noise_sequence(np.random.default_rng(11), 400)
    np.testing.assert_array_equal(first, second)
    third = noise_sequence(np.random.default_rng(12), 400)
    assert not np.array_equal(first, third)
    assert np.abs(first.mean()) < 4 * INTERVAL_NOISE_STD_RAD / 20
    assert first.std() == pytest.approx(INTERVAL_NOISE_STD_RAD, rel=0.05)
    # Individual draws do not accumulate into a permanent ratio: the spread of
    # N sums grows like sigma*sqrt(N), not like N.
    sums = [noise_sequence(np.random.default_rng(1000 + s), 25).sum() for s in range(1500)]
    np.testing.assert_allclose(np.std(sums), INTERVAL_NOISE_STD_RAD * np.sqrt(25), rtol=0.08)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("noise") / "result"
    report = run_experiment(output, runs=6, seed=0)
    return output, report


def test_repeated_runs_share_truth_and_common_noise_across_groups(recording):
    output, _ = recording
    routes, _ = load_replays(output)
    old_straight, _ = run_case("straight")
    np.testing.assert_array_equal(
        routes["straight"]["truth"]["true_poses"], old_straight["true_poses"]
    )
    np.testing.assert_array_equal(
        routes["straight"]["truth"]["wheel_angles"], old_straight["wheel_angles_rad"]
    )
    fixed = routes["straight"]["groups"]["fixed"]
    raw = routes["straight"]["groups"]["uncorrected"]
    np.testing.assert_array_equal(fixed["arrays"]["epsilon"], raw["arrays"]["epsilon"])
    left = routes["straight"]["truth"]["wheel_angles"][:, 0]
    np.testing.assert_array_equal(
        fixed["arrays"]["readings"][:, :, 0], np.broadcast_to(left, (6, len(left)))
    )
    np.testing.assert_array_equal(
        raw["arrays"]["readings"][:, :, 0], np.broadcast_to(left, (6, len(left)))
    )
    for run in range(6):
        np.testing.assert_array_equal(
            estimate_poses(fixed["arrays"]["readings"][run]), fixed["arrays"]["poses"][run]
        )
        np.testing.assert_array_equal(
            estimate_poses(raw["arrays"]["readings"][run]), raw["arrays"]["poses"][run]
        )


def test_noiseless_reference_reproduces_lesson16_ideal(recording):
    output, _ = recording
    routes, _ = load_replays(output)
    noiseless = routes["straight"]["groups"]["noiseless"]
    np.testing.assert_allclose(
        noiseless["arrays"]["poses"][0], routes["straight"]["truth"]["true_poses"], atol=1e-9
    )
    assert float(noiseless["arrays"]["position_error"][0].max()) < 1e-9


def test_bias_and_dispersion_are_separated(recording):
    output, _ = recording
    routes, _ = load_replays(output)
    straight = routes["straight"]["groups"]
    fixed = straight["fixed"]["arrays"]["position_error"]
    unc = straight["uncorrected"]["arrays"]["position_error"]
    np.testing.assert_array_equal(
        straight["fixed"]["arrays"]["epsilon"], straight["uncorrected"]["arrays"]["epsilon"]
    )
    # Fixed: mean low, dispersion clearly non-zero (random walk in heading).
    assert fixed.sum(axis=0)[-1] / 6 < 0.04
    assert fixed.std(axis=0, ddof=1)[-1] > 0.005
    # Uncorrected: the ~19.4 cm systematic bias dominates each repetition.
    assert 0.18 < unc.sum(axis=0)[-1] / 6 < 0.215
    # Signed endpoint bias: random noise averages out, the fixed scale does not.
    fixed_endpoint = (
        straight["fixed"]["arrays"]["poses"][:, -1, :2]
        - routes["straight"]["truth"]["true_poses"][-1, :2]
    )
    unc_endpoint = (
        straight["uncorrected"]["arrays"]["poses"][:, -1, :2]
        - routes["straight"]["truth"]["true_poses"][-1, :2]
    )
    assert abs(fixed_endpoint[:, 1].mean()) < 0.02
    assert unc_endpoint[:, 1].mean() > 0.15
    # Square route: noiseless still exact; uncorrected stays biased.
    square = routes["square"]["groups"]
    assert float(square["noiseless"]["arrays"]["position_error"][0].max()) < 1e-9
    assert 0.14 < square["uncorrected"]["arrays"]["position_error"].sum(axis=0)[-1] / 6 < 0.18


def test_ensemble_curves_grow_and_restricted_keys(recording):
    output, _ = recording
    _, ensembles = load_replays(output)
    mean, std = ensembles["straight"]["fixed"]["position_error"]
    assert std[-1] > std[0] + 0.5  # cm; dispersion grows, it does not stay zero.
    assert mean[-1] > mean[100]
    with np.load(output / "trajectories.npz", allow_pickle=False) as npz:
        assert "straight_checkpoints" in npz.files
        assert npz["straight_fixed_epsilon"].shape == (6, 300)
        assert npz["straight_fixed_readings"].shape == (6, 301, 2)
        assert npz["straight_fixed_poses"].shape == (6, 301, 3)
        assert npz["straight_noiseless_poses"].shape == (1, 301, 3)


def test_run_experiment_guards(recording):
    output, _ = recording
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=3, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "x", runs=1, seed=0)


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=4, seed=3)
    assert report["base_seed"] == 3 and report["seeds"] == [3, 4, 5, 6]
    path = output / "trajectories.npz"
    original = path.read_bytes()
    checksum = report["trajectories_sha256"]
    assert hashlib.sha256(original).hexdigest() == checksum
    bad = output / "summary.json"
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["experiment"] = "other"
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)
    data["experiment"] = EXPERIMENT
    bad.write_text(json.dumps(data), encoding="utf-8")
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["straight_fixed_poses"][0, 0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)


def test_summary_group_contract(recording):
    _, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["interval_noise_std_rad"] == INTERVAL_NOISE_STD_RAD
    assert report["runs_per_group"] == 6
    assert [c["key"] for c in report["cases"]] == ["straight", "square"]
    for case in report["cases"]:
        groups = case["groups"]
        assert set(groups) == {
            f"{key}_{suffix}"
            for key, _, _ in GROUPS
            for suffix in ("ensemble_position", "ensemble_heading", "endpoint_stats")
        } | {key for key, _, _ in GROUPS}
        assert (
            groups["fixed_ensemble_position"]["final_mean"]
            < groups["uncorrected_ensemble_position"]["final_mean"]
        )


@pytest.mark.isolated_tk
def test_tk_noise_demo_playback_and_group_switch(recording):
    import tkinter as tk

    output, _ = recording
    root = tk.Tk()
    root.withdraw()
    routes, ensembles = load_replays(output)
    demo = NoiseDemo(root, routes, ensembles)
    demo.canvas.winfo_width = lambda: 650
    demo.canvas.winfo_height = lambda: 320
    demo.chart.winfo_width = lambda: 1000
    demo.chart.winfo_height = lambda: 125
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert demo.group == "fixed" and demo.route_key == "straight"
        demo.toggle()
        demo.clock.advance(0.16)
        demo.redraw()
        assert demo.clock.index >= 1 and not demo.clock.paused
        demo.toggle()
        demo.seek(100)
        assert "均值" in demo.stats.cget("text")
        before = demo.clock.index
        demo.group_box.current(next(i for i, key in enumerate(GROUP_KEYS) if key == "uncorrected"))
        demo.select_group()
        assert demo.clock.index == before and demo.clock.paused
        assert "未标定" in demo.stats.cget("text")
        demo.sample.set(2)
        demo.select_sample()
        assert "样本 #2" in demo.stats.cget("text")
        demo.case_box.current(1)
        demo.select_case()
        assert demo.route_key == "square" and demo.clock.index == 0
        demo.seek(600)
        assert "24.00 / 24.00" in demo.stats.cget("text")
    finally:
        demo.close()
