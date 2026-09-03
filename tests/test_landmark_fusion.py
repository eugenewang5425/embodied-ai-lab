from __future__ import annotations

import json

import numpy as np
import pytest

from embodied_learning.differential_drive import compose
from embodied_learning.experiments.landmark_fusion import (
    SCENARIOS,
    case_stats,
    digest,
    load_recording,
    run_experiment,
    run_runs,
)
from embodied_learning.experiments.landmark_observations import run_runs as lesson18_runs
from embodied_learning.fusion_demo import FusionDemo
from embodied_learning.odometry import estimate_poses, heading_error
from embodied_learning.pose_fusion import estimate_with_resets


def test_without_observations_matches_odometry():
    encoders = np.random.default_rng(8).normal(size=(51, 2)).cumsum(axis=0)
    initial = [1.5, -0.7, 2.8]
    poses, prior = estimate_with_resets(encoders, [], np.empty((0, 3)), initial)
    np.testing.assert_array_equal(poses, estimate_poses(encoders, initial))
    np.testing.assert_array_equal(prior, poses)


def test_reset_corrects_position_and_heading_then_uses_new_heading():
    encoders = np.tile(np.arange(5.0)[:, None], (1, 2))
    poses, prior = estimate_with_resets(encoders, [2], [[3, 4, np.pi / 2]])
    np.testing.assert_allclose(poses[2], [3, 4, np.pi / 2])
    assert prior[2, 0] > prior[1, 0]
    assert poses[3, 1] > poses[2, 1]
    assert poses[3, 0] == pytest.approx(3)
    np.testing.assert_allclose(poses[3:], estimate_poses(encoders[2:], poses[2])[1:])


def test_yaw_reset_uses_shortest_branch():
    initial = [0, 0, np.deg2rad(179)]
    poses, _ = estimate_with_resets(np.zeros((3, 2)), [1], [[0, 0, np.deg2rad(-179)]], initial)
    assert np.rad2deg(poses[1, 2]) == pytest.approx(181)
    np.testing.assert_allclose(heading_error(poses[1, 2], np.deg2rad(-179)), 0, atol=1e-14)


def test_no_future_leakage_and_prefix_invariance():
    encoders = np.tile(np.arange(12.0)[:, None], (1, 2))
    a, _ = estimate_with_resets(encoders, [3, 8], [[1, 2, 0.3], [5, 6, 0.4]])
    b, _ = estimate_with_resets(encoders, [3, 8], [[1, 2, 0.3], [50, -6, -0.4]])
    prefix, _ = estimate_with_resets(encoders[:8], [3], [[1, 2, 0.3]])
    np.testing.assert_array_equal(a[:8], b[:8])
    np.testing.assert_array_equal(a[:8], prefix)
    assert not np.array_equal(a[8:], b[8:])


@pytest.mark.parametrize("frames", [[0], [-1], [5], [1.5], [True], [2, 1], [1, 1], [[1]]])
def test_rejects_invalid_observation_frames(frames):
    with pytest.raises(ValueError, match="frames"):
        estimate_with_resets(np.zeros((5, 2)), frames, np.zeros((len(frames), 3)))


@pytest.mark.parametrize("samples", [[[0, 0]], [[0, np.nan, 0]], [[0, 0, 0], [1, 1, 1]]])
def test_rejects_invalid_observation_poses(samples):
    with pytest.raises(ValueError, match="pose"):
        estimate_with_resets(np.zeros((5, 2)), [2], samples)


def test_outlier_is_transferred_not_magically_filtered():
    # Exact stationary odometry, but one bad external pose: resetting makes it worse.
    poses, prior = estimate_with_resets(np.zeros((4, 2)), [2], [[1.0, -2.0, 0.2]])
    np.testing.assert_array_equal(prior[2], [0, 0, 0])
    np.testing.assert_allclose(poses[2:], [[1, -2, 0.2], [1, -2, 0.2]])


@pytest.mark.parametrize("key", [s[0] for s in SCENARIOS])
def test_lesson18_measurements_and_baselines_unchanged(key):
    old = lesson18_runs(key, 2, 0)
    new = run_runs(key, 2, 0)
    for before, after in (
        ("true_poses", "truth"),
        ("odom_poses", "odom"),
        ("landmark_poses", "held"),
    ):
        np.testing.assert_array_equal(old[before], new[after])
    frames = new["observation_frames"]
    np.testing.assert_array_equal(new["fused"][:, frames, :2], new["body_samples"][:, :, :2])
    np.testing.assert_allclose(
        heading_error(new["fused"][:, frames, 2], new["body_samples"][:, :, 2]), 0, atol=1e-14
    )
    np.testing.assert_array_equal(new["fused"][:, : frames[0]], new["odom"][:, : frames[0]])


def test_world_frame_transform_equivariance():
    encoders = np.array([[0, 0], [0.2, 0.3], [0.4, 0.5], [0.6, 0.9]])
    initial, observation, world_change = [0, 0, 0], [1, -2, 0.4], [5, 3, 0.8]
    a, _ = estimate_with_resets(encoders, [2], [observation], initial)
    b, _ = estimate_with_resets(
        encoders, [2], [compose(world_change, observation)], compose(world_change, initial)
    )
    np.testing.assert_allclose(b, [compose(world_change, p) for p in a], atol=1e-12)


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    output = tmp_path_factory.mktemp("fusion") / "recording"
    report = run_experiment(output, runs=6, seed=0)
    return output, report


def test_recording_has_raw_measurements_and_truth_separated(recording):
    output, report = recording
    routes, loaded = load_recording(output)
    assert loaded == report
    for key, _, _ in SCENARIOS:
        route = routes[key]
        assert route["observations"].shape[2:] == (3, 2)
        assert all(not arr.flags.writeable for arr in route.values())
        assert not np.shares_memory(route["fused"], route["truth"])
        expected = case_stats(route)
        actual = next(c for c in report["cases"] if c["key"] == key)
        assert expected["methods"] == actual["methods"]
        assert expected["updates"] == actual["updates"]


def test_reset_does_not_claim_better_final_frame_than_observation(recording):
    output, _ = recording
    routes, _ = load_recording(output)
    for route in routes.values():
        np.testing.assert_array_equal(route["fused"][:, -1, :2], route["held"][:, -1, :2])


def test_seeded_fusion_improves_time_mean_but_not_every_update(recording):
    _, report = recording
    worse = 0
    for case in report["cases"]:
        stats = case["methods"]
        assert stats["fused"]["time_mean_position_m"] < stats["held"]["time_mean_position_m"]
        # The long route is where accumulated odometry drift motivates resetting.
        if case["key"] == "long":
            assert stats["fused"]["time_mean_position_m"] < stats["odom"]["time_mean_position_m"]
        worse += case["updates"]["worse_count"]
    assert worse > 0


@pytest.mark.parametrize("runs,seed", [(1, 0), (True, 0), (2, -1), (2, 0.5)])
def test_experiment_validates_settings(tmp_path, runs, seed):
    with pytest.raises(ValueError):
        run_experiment(tmp_path / "invalid", runs=runs, seed=seed)
    assert not (tmp_path / "invalid").exists()


def test_output_is_not_overwritten(recording):
    output, _ = recording
    with pytest.raises(FileExistsError):
        run_experiment(output)


def test_checksum_and_recomputed_estimator_reject_corruption(tmp_path, recording):
    output, report = recording
    report = dict(report)
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        arrays = {key: value.copy() for key, value in archive.items()}
    arrays["straight_fused"][0, 80, 0] += 0.5
    np.savez_compressed(tmp_path / "trajectories.npz", **arrays)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_recording(tmp_path)
    report["trajectories_sha256"] = digest(tmp_path / "trajectories.npz")
    summary_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent estimator"):
        load_recording(tmp_path)


@pytest.mark.isolated_tk
def test_teaching_ui_measurements_playback_and_reset_explanation(recording):
    import tkinter as tk

    output, _ = recording
    routes, report = load_recording(output)
    root = tk.Tk()
    root.withdraw()
    demo = FusionDemo(root, routes, report)
    demo.canvas.winfo_width = lambda: 650
    demo.canvas.winfo_height = lambda: 320
    demo.chart.winfo_width = lambda: 1100
    demo.chart.winfo_height = lambda: 150
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert not demo.readings_table.get_children()
        assert "尚无观测" in demo.measure_status.cget("text")
        demo.next_button.invoke()
        assert demo.clock.index == 50 and demo.clock.paused
        assert len(demo.readings_table.get_children()) == 3
        assert "传感器位姿" in demo.solve_text.cget("text")
        assert "最近一次校正" in demo.update_text.cget("text")
        snapshot = [
            demo.readings_table.item(row)["values"] for row in demo.readings_table.get_children()
        ]
        demo.step()
        assert demo.clock.index == 51
        assert snapshot == [
            demo.readings_table.item(row)["values"] for row in demo.readings_table.get_children()
        ]
        demo.notebook.select(1)
        demo.show_hold.set(False)
        demo.metric.set("朝向误差 / °")
        demo.redraw()
        assert demo.metric.get() == "朝向误差 / °"
        demo.sample.set(2)
        demo.select_sample()
        assert demo.run == 2 and demo.clock.index == 51
        demo.sample.set("bad")
        demo.select_sample()
        assert demo.run == 2
        demo.case_box.current(2)
        demo.select_case()
        assert demo.route_key == "long" and demo.clock.index == 0
        demo.next_worse_update()
        assert "变差" in demo.update_text.cget("text")
        demo.seek(demo.clock.steps)
        assert "结束" in demo.status.cget("text")
        demo.seek(0)
        demo.toggle()
        demo.clock.advance(0.16)
        assert demo.clock.index == 1
        demo.toggle()
        assert demo.clock.paused
    finally:
        demo.close()
