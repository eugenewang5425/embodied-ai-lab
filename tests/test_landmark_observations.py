from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.differential_drive import compose
from embodied_learning.experiments.landmark_observations import (
    EXPERIMENT,
    LANDMARKS,
    OBS_PERIOD_STEPS,
    OBS_RANGE_STD_M,
    bearing_reading,
    hold_estimate,
    inverse_pose,
    observe,
    run_experiment,
    solve_pose,
)
from embodied_learning.experiments.mobile_frames import SENSOR_IN_BODY
from embodied_learning.landmark_demo import SCENARIO_KEYS, LandmarkDemo, load_replays


def test_inverse_pose_roundtrip():
    for pose in ([0.3, -0.7, 1.2], [-2, 4, -2.5], [0, 0, 0]):
        recovered = compose(pose, inverse_pose(pose))
        np.testing.assert_allclose(recovered, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(compose(inverse_pose(pose), pose), [0, 0, 0], atol=1e-12)


def test_bearing_is_measured_in_the_sensor_frame():
    # Body at origin; sensor sits at (0.12, 0.04) with +30 deg yaw.
    # A landmark straight north (0, 1) would be 90 deg in WORLD axes, but the
    # sensor frame is rotated, so the true bearing differs from 90 deg.
    sensor = compose([0, 0, 0], SENSOR_IN_BODY)
    distance, bearing = bearing_reading((0, 1), sensor)
    # to_child: R(-30) @ ((0,1)-(0.12,0.04))
    point = np.array([-0.12, 0.96])
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    expected_point = np.array([c * point[0] + s * point[1], -s * point[0] + c * point[1]])
    np.testing.assert_allclose(
        [distance, bearing],
        [np.linalg.norm(expected_point), np.arctan2(expected_point[1], expected_point[0])],
        atol=1e-12,
    )
    # The crucial point: 90 deg (world-axes bearing) would be wrong by ~30 deg.
    assert abs(np.rad2deg(bearing) - 90) > 15


@pytest.mark.parametrize(
    "pose",
    [
        [0.0, 0.0, 0.0],
        [1.2, 0.3, 0.7],
        [2.4, 0.0, 0.0],
        [0.7, 0.9, -1.2],
        [6.4, 1.1, 2.2],
    ],
)
def test_solve_pose_noiseless_is_exact(pose):
    rng = np.random.default_rng(3)
    readings = observe(pose, LANDMARKS, rng, 0.0, 0.0)
    estimate = solve_pose(readings, LANDMARKS)
    np.testing.assert_allclose(estimate, pose, atol=1e-9)


def test_solve_pose_noisy_has_no_systematic_bias():
    pose = np.array([0.7, 0.9, -1.2])
    errors = []
    for seed in range(120):
        readings = observe(pose, LANDMARKS, np.random.default_rng(seed))
        estimate = solve_pose(readings, LANDMARKS)
        errors.append(estimate[:2] - pose[:2])
    errors = np.asarray(errors)
    mean = errors.mean(axis=0)
    std = errors.std(axis=0, ddof=1)
    assert np.linalg.norm(mean) < 2 * np.linalg.norm(std) / np.sqrt(120) * 4
    assert np.sqrt((errors**2).sum(axis=1)).mean() < 0.03
    # Bearing noise dominates at long range: error grows with landmark distance.
    far = np.array([6.4, 0.0, 0.0])
    far_errors = []
    for seed in range(60):
        readings = observe(far, LANDMARKS, np.random.default_rng(seed))
        estimate = solve_pose(readings, LANDMARKS)
        far_errors.append(estimate[:2] - far[:2])
    far_rms = float(np.sqrt(np.square(far_errors).mean()))
    assert far_rms > float(np.sqrt(np.square(errors).mean()))


def test_observations_are_seeded_and_deterministic():
    first = observe([0, 0, 0], LANDMARKS, np.random.default_rng(9))
    second = observe([0, 0, 0], LANDMARKS, np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    third = observe([0, 0, 0], LANDMARKS, np.random.default_rng(10))
    assert not np.array_equal(first, third)
    assert np.all(first[:, 0] > 0)


def test_hold_estimate_is_piecewise_constant_and_starts_known():
    poses = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0]])
    frames = [10, 20, 30]
    held = hold_estimate(poses, frames, steps=30)
    np.testing.assert_array_equal(held[0], [0, 0, 0])
    np.testing.assert_array_equal(held[:10], np.tile([0, 0, 0], (10, 1)))
    np.testing.assert_array_equal(held[10:20], np.tile(poses[0], (10, 1)))
    np.testing.assert_array_equal(held[20:30], np.tile(poses[1], (10, 1)))
    np.testing.assert_array_equal(held[30], poses[2])


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("landmarks") / "result"
    report = run_experiment(output, runs=6, seed=0)
    return output, report


def test_odometry_grows_but_observation_does_not(recording):
    output, _ = recording
    routes, ensembles = load_replays(output)
    obs_final = ensembles["straight"]["stats"]
    assert obs_final["odom_final_mean"] > obs_final["landmark_final_mean"]
    assert obs_final["observed_mean"] < obs_final["odom_final_mean"]
    sq = ensembles["square"]["stats"]
    assert sq["odom_final_mean"] > sq["landmark_final_mean"]
    assert sq["observed_mean"] < sq["odom_final_mean"]
    long_stats = ensembles["long"]["stats"]
    assert long_stats["odom_final_mean"] > long_stats["observed_mean"]
    # Observation-time errors do not drift upward in time on straight/square.
    for ens in (ensembles["straight"], ensembles["square"]):
        samples = ens["stats"]["observed_at_times"]
        first, last = samples[0]["mean_m"], samples[-1]["mean_m"]
        assert abs(last - first) < 0.02
        assert max(s["std_m"] for s in samples) < 0.05
    # The observation estimate is fresh at sample times: its final-frame error
    # is the last observation error, not a growing drift.
    assert routes["straight"]["landmark_position_error"][:, -1].mean() < 0.05


def test_observation_period_and_contract(recording):
    _, report = recording
    assert report["experiment"] == EXPERIMENT
    assert report["observation_period_s"] == OBS_PERIOD_STEPS * 0.04
    assert report["range_noise_std_m"] == OBS_RANGE_STD_M
    assert [c["key"] for c in report["cases"]] == list(SCENARIO_KEYS)
    for case in report["cases"]:
        frames = case["observation_frames"]
        assert frames == list(range(OBS_PERIOD_STEPS, case["steps"] + 1, OBS_PERIOD_STEPS))
        assert frames[-1] == case["steps"]


def test_run_experiment_guards(recording):
    output, _ = recording
    with pytest.raises(FileExistsError):
        run_experiment(output, runs=3, seed=0)
    with pytest.raises(ValueError):
        run_experiment(output.parent / "x", runs=1, seed=0)


def test_recording_rejects_tampering(tmp_path):
    output = tmp_path / "result"
    report = run_experiment(output, runs=4, seed=7)
    path = output / "trajectories.npz"
    assert report["trajectories_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["straight_odom_poses"][0, 0, 0] += 1.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(output)
    bad = output / "summary.json"
    data = json.loads(bad.read_text(encoding="utf-8"))
    data["landmarks_world_m"] = [[9, 9], [9, 9], [9, 9]]
    bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_replays(output)


@pytest.mark.isolated_tk
def test_tk_demo_playback_next_observation_and_sample(recording):
    import tkinter as tk

    output, _ = recording
    root = tk.Tk()
    root.withdraw()
    routes, ensembles = load_replays(output)
    demo = LandmarkDemo(root, routes, ensembles)
    demo.canvas.winfo_width = lambda: 650
    demo.canvas.winfo_height = lambda: 330
    demo.chart.winfo_width = lambda: 1000
    demo.chart.winfo_height = lambda: 130
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert demo.route_key == "straight" and demo.run == 0
        demo.toggle()
        demo.clock.advance(0.16)
        demo.redraw()
        assert demo.clock.index >= 1
        demo.toggle()
        demo.next_observation()
        assert demo.clock.index == OBS_PERIOD_STEPS
        assert "上次观测" in demo.stats.cget("text")
        demo.sample.set(2)
        demo.select_sample()
        assert "样本 #2" in demo.stats.cget("text")
        demo.case_box.current(1)
        demo.select_case()
        assert demo.route_key == "square" and demo.clock.index == 0
        demo.seek(demo.current()["steps"])
        assert "回放结束" in demo.stats.cget("text")
        demo.metric.set("朝向误差 / °")
        demo.redraw()
    finally:
        demo.close()
