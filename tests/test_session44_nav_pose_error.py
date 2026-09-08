"""Lesson 44: pose-error propagation through the mapping stack.

Unit checks pin the contract: the estimator's zero-delta identity, the
bias-integration known answer (the lesson-15 2% drift chain: ~19.4 cm and
~9.2 deg on a 2.4 m run), the observation reset rule (moved TO the
observation, 2-pi branch), the ideal corner-beacon solve accuracy (geometry
DOP ~2.8 cm at 1 cm range noise), the truth/estimate same-index invariant
(T error exactly zero), the bitwise agreement with
pose_fusion.estimate_with_resets, the grid-shift metric known answer, the
angle-diff branch behavior, config validation, the paired-start contract
with lesson 43, the error hierarchy (O2 > O1 > ~0; F bounded by
observations), micro determinism, the shrunk record contract, the CLI, the
demo loader's tamper rejections and the real Tk demo.  No full-scale run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

# Same Agg-first policy as lesson-43 tests (Tcl interpreter isolation).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.differential_drive import DriveGeometry
from embodied_learning.experiments.grid_nav import (
    GridNavConfig,
    build_scenarios,
    initial_poses,
)
from embodied_learning.experiments.nav_pose_error import (
    BEACONS,
    EXPERIMENT,
    GROUPS4,
    NavPoseErrorConfig,
    angle_diff,
    estimate_step,
    expected_npz_keys,
    map_shift_cells,
    run_experiment,
    simulate_episode,
)
from embodied_learning.landmark_localization import observe, solve_pose
from embodied_learning.nav_pose_error_demo import load_replays
from embodied_learning.odometry import scaled_encoder_readings
from embodied_learning.pose_fusion import estimate_with_resets

SMALL_CONFIG = NavPoseErrorConfig(
    base=GridNavConfig(
        world_size_m=4.0,
        horizon_steps=700,
        obstacle_scenes=2,
        inits=2,
        start_xy=(0.6, 0.6),
        goal_xy=(3.4, 3.4),
    )
)
TINY_CONFIG = NavPoseErrorConfig(
    base=GridNavConfig(
        world_size_m=4.0,
        horizon_steps=250,
        obstacle_scenes=1,
        inits=1,
        start_xy=(0.6, 0.6),
        goal_xy=(3.4, 3.4),
    )
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("nav_pose_error_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)
    return output, report


# ------------------------------------------------------------------ estimator
def test_estimate_zero_delta_identity():
    """Zero wheel-angle increments: the estimate stays put."""
    est = np.array([1.0, 2.0, 0.7])
    out = estimate_step(est, np.zeros(2), None, 0.02)
    np.testing.assert_array_equal(out, est)


def test_estimate_bias_known_answer():
    """2% right-wheel scale reproduces the lesson-15 drift chain exactly."""
    # a straight run: v = 0.25 m/s -> wheels 5 rad/s each, dt = 0.04 -> 240
    # steps -> the car physically drives the 2.4 m straight; the estimator,
    # fed a 2% oversized right-wheel reading, integrates a CURVED path.  The
    # lesson-15 calibration contract quotes ~19.4 cm position error and a
    # ~9.2 deg heading error on exactly this 2.4 m run at 2%.
    steps = 240
    delta = np.array([5.0 * 0.04, 5.0 * 0.04])
    est = np.array([0.0, 0.0, 0.0])
    for _ in range(steps):
        est = estimate_step(est, delta, None, 0.02)
    # the lesson-15 position-error DISTANCE (estimate vs physical endpoint) on
    # the 2.4 m run at 2% bias: ~19.4 cm (curved estimate vs straight truth),
    # plus the ~9.2 deg heading error quoted there.
    assert 0.14 < np.linalg.norm(est[:2] - np.array([2.4, 0.0])) < 0.24
    assert abs(est[2]) == pytest.approx(0.16, abs=0.01)  # ~9.2 deg, lesson 15


def test_estimate_reset_rule_and_2pi_branch():
    """Reset moves TO the observation; continuous yaw branches land nearby."""
    est = np.array([1.0, 2.0, 5.9])  # wrapped observation of a -0.5 rad heading
    obs = np.array([0.98, 2.01, -0.5])
    out = estimate_step(est, np.zeros(2), obs, 0.02)
    np.testing.assert_allclose(out[:2], obs[:2], atol=1e-12)
    assert abs(angle_diff(out[2], obs[2])) < 1e-9
    est2 = np.array([0.0, 0.0, 3.0])
    obs2 = np.array([0.0, 0.0, 3.1])
    out2 = estimate_step(est2, np.zeros(2), obs2, 0.02)
    assert abs(out2[2] - 3.1) < 1e-9


def test_angle_diff_broadcast_and_branch():
    assert angle_diff(0.0, 0.1) == pytest.approx(0.1)
    assert angle_diff(0.1, -0.1) == pytest.approx(-0.2)
    assert abs(angle_diff(3.0, -3.0) - (2 * np.pi - 6.0)) < 1e-9
    arr = angle_diff(np.array([0.0, 0.0]), np.array([0.1, -0.1]))
    np.testing.assert_allclose(arr, [0.1, -0.1])


def test_solve_pose_beacon_accuracy():
    """Corner beacons: the ideal observation solver is exact within its DOP."""
    rng = np.random.default_rng(7)
    for pose in [
        np.array([1.0, 1.0, 0.5]),
        np.array([3.2, 1.5, -0.48]),
        np.array([8.9, 7.1, 2.9]),
    ]:
        solved = solve_pose(observe(pose, BEACONS, rng), BEACONS)
        assert np.linalg.norm(solved[:2] - pose[:2]) < 0.05  # DOP ~2.8 cm
        assert abs(angle_diff(solved[2], pose[2])) < 0.012


def test_streaming_matches_pose_fusion():
    """Incremental lesson-19 estimator equals estimate_with_resets bitwise."""
    rng = np.random.default_rng(1)
    steps = 400
    wheel = rng.uniform(1.0, 4.0, size=(steps, 2))
    encoder = np.vstack([[0.0, 0.0], np.cumsum(wheel * 0.04, axis=0)])
    # a fixed corridor of true poses anchored at the start; observations are
    # generated from the truth by the SAME solution chain the experiment uses
    frames = [50, 100, 150, 199, 300, 399]
    poses = []
    truth = np.zeros((steps + 1, 3))
    truth[0] = [1.0, 1.0, 0.3]
    for i in range(steps):
        truth[i + 1] = truth[i] + np.array([0.03, 0.02, 0.011])
    obs_rng = np.random.default_rng(3)
    for frame in frames:
        poses.append(solve_pose(observe(truth[frame], BEACONS, obs_rng), BEACONS))
    pos_dict = {f: p for f, p in zip(frames, poses, strict=True)}
    posterior, _prior = estimate_with_resets(
        scaled_encoder_readings(encoder, 1.02),
        np.array(frames),
        np.array(poses),
        initial_pose=truth[0],
        geometry=DriveGeometry(),
    )
    est = truth[0].copy()
    stream = [est.copy()]
    for step in range(1, steps + 1):
        delta = wheel[step - 1] * 0.04  # the move that produced frame `step`
        est = estimate_step(est, delta, pos_dict.get(step), 0.02)
        stream.append(est.copy())
    stream = np.asarray(stream)
    np.testing.assert_allclose(stream[1:], posterior[1:], atol=1e-12)


def test_map_shift_known_answer():
    """Single believed cell at an offset: nearest-true distance by hand."""
    grid = np.zeros((10, 10))
    grid[8, 8] = 5.0
    true_map = np.zeros((10, 10), dtype=bool)
    true_map[5, 5] = True
    assert map_shift_cells(grid, true_map) == pytest.approx(math.sqrt(18.0), abs=1e-9)
    empty = np.zeros((10, 10))
    assert np.isnan(map_shift_cells(empty, true_map))


# ------------------------------------------------------------------ protocol
def test_config_validation():
    with pytest.raises(ValueError):
        NavPoseErrorConfig(encoder_biases={"T": 0.0, "O1": 0.01, "O2": 0.02})
    with pytest.raises(ValueError):
        NavPoseErrorConfig(obs_period_steps=0)
    with pytest.raises(ValueError):
        NavPoseErrorConfig(obs_range_std_m=-1.0)
    ok = NavPoseErrorConfig()
    assert set(ok.encoder_biases) == set(GROUPS4)
    assert ok.config is not None


def test_paired_starts_with_lesson43():
    """The same seed/scene reproduce the lesson-43 start poses bitwise."""
    base = GridNavConfig()
    mine = initial_poses(base, 0, 0)[0]
    ref = np.array([0.9325284480510454, 0.9396373018921536, -1.0625653956067165])
    np.testing.assert_allclose(mine, ref, atol=1e-12)


@pytest.mark.slow
def test_truth_same_index_and_t_zero_error():
    """T estimate = truth at the SAME index (no phase offset), error exactly 0."""
    config = SMALL_CONFIG
    scene = build_scenarios(config.config, 0)[1]
    start = initial_poses(config.config, 0, 1)[0]
    metrics, truth, est, _ex = simulate_episode("T", scene["obstacles"], start, config)
    assert len(truth) == len(est)
    np.testing.assert_allclose(est, truth, atol=1e-12)  # same index, exact
    assert metrics["mean_position_error_m"] < 1e-9


@pytest.mark.slow
def test_error_hierarchy_small_run(small_run):
    """Over identical pairs: O2 error > O1 error > ~0, F bounded by observations."""
    output, _report = small_run
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        o2 = archive["O2_mean_pos_err"][1, 0]
        o1 = archive["O1_mean_pos_err"][1, 0]
        f = archive["F_mean_pos_err"][1, 0]
        t = archive["T_mean_pos_err"][1, 0]
    assert o1 > 0.01
    assert o2 > o1
    assert 0.0 <= f < 0.3 * o2
    assert abs(t) < 1e-9


@pytest.mark.slow
def test_micro_run_deterministic(tmp_path):
    first = run_experiment(tmp_path / "one", seed=0, config=TINY_CONFIG, log=None)
    second = run_experiment(tmp_path / "two", seed=0, config=TINY_CONFIG, log=None)
    assert first["trajectories_sha256"] == second["trajectories_sha256"]
    assert json.dumps(first["aggregates"], sort_keys=True) == json.dumps(
        second["aggregates"], sort_keys=True
    )


@pytest.mark.slow
def test_small_run_record_contract(small_run, tmp_path):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT and report["schema_version"] == 1
    protocol = report["protocol"]
    assert protocol["scene_count"] == 3 and protocol["inits"] == 2
    assert len(report["episodes"]) == protocol["scene_count"] * protocol["inits"] * 4
    rows = report["episodes"]
    for scene in range(protocol["scene_count"]):
        for init in range(protocol["inits"]):
            by_group = {r["group"]: r for r in rows if r["scene"] == scene and r["init"] == init}
            assert set(by_group) == set(GROUPS4)
            for group in GROUPS4:
                assert by_group[group]["start_pose"] == by_group["T"]["start_pose"]  # paired
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["snap_grids"].shape == (8, 40, 40)
        assert len(archive["replan_steps"]) >= 1
    for name in ("summary.json", "trajectories.npz", "comparison.png", "scenario_maps.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)


@pytest.mark.slow
def test_hypothesis_keys(small_run):
    _, report = small_run
    h = report["hypothesis"]
    for key in ("gate_met", "monotone_met", "fusion_met"):
        assert isinstance(h["results"][key], bool)
    assert set(h["results"]["counts"]) == set(GROUPS4)


@pytest.mark.slow
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.nav_pose_error",
            "--output",
            str(out),
            "--seed",
            "0",
            "--scenes",
            "1",
            "--inits",
            "1",
            "--horizon",
            "400",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert set(payload["aggregates"]) == set(GROUPS4)
    for name in ("summary.json", "trajectories.npz", "comparison.png", "scenario_maps.png"):
        assert (out / name).is_file()


@pytest.mark.slow
def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered_aggregates"
    shutil.copytree(output, work)
    parsed = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    parsed["aggregates"]["F"]["obstacle"]["arrivals"] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Arrival counts disagree"):
        load_replays(work)

    work2 = tmp_path / "tampered_bytes"
    shutil.copytree(output, work2)
    path = work2 / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["snap_errs"] = payload["snap_errs"] + 0.5
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work2)

    del payload["snap_errs"]
    np.savez_compressed(path, **payload)
    parsed2 = json.loads((work2 / "summary.json").read_text(encoding="utf-8"))
    parsed2["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work2 / "summary.json").write_text(
        json.dumps(parsed2, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work2)

    work4 = tmp_path / "tampered_map"
    shutil.copytree(output, work4)
    path4 = work4 / "trajectories.npz"
    with np.load(path4, allow_pickle=False) as archive:
        payload4 = {key: archive[key].copy() for key in archive.files}
    payload4["true_map_0"] = ~payload4["true_map_0"]
    np.savez_compressed(path4, **payload4)
    parsed4 = json.loads((work4 / "summary.json").read_text(encoding="utf-8"))
    parsed4["trajectories_sha256"] = hashlib.sha256(path4.read_bytes()).hexdigest()
    (work4 / "summary.json").write_text(
        json.dumps(parsed4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="True map disagrees"):
        load_replays(work4)


@pytest.mark.isolated_tk
@pytest.mark.slow
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.nav_pose_error_demo import NavPoseErrorDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = NavPoseErrorDemo(root, data)
    root.update()
    assert demo.mode.get() == "errors"
    assert len(demo.fig.axes) == 2
    assert "位置误差" in demo.stats.cget("text")
    demo.mode.set("maps")
    demo.redraw()
    assert len(demo.fig.axes) == 2
    assert "错位" in demo.stats.cget("text")
    demo.mode.set("compare")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    obstacle = report["aggregates"]["F"]["obstacle"]
    assert f"{obstacle['arrivals']}/{obstacle['episodes']}" in panel
    demo.redraw()
    assert len(demo.fig.axes) == 4
    demo.close()
