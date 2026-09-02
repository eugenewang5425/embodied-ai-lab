import hashlib
import json

import numpy as np
import pytest

from embodied_learning.arm_path import (
    IK_COMPARISON_METHODS,
    METHODS,
    generate_reference,
    validate_line,
)
from embodied_learning.arm_path_demo import PathDemo, load_replays
from embodied_learning.experiments.arm_ik_comparison import (
    load_source,
    run_comparison,
    verify_retained_trajectories,
)
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import run_batch
from embodied_learning.planar_arm import forward_kinematics, inverse_kinematics, joint_positions


def test_waypoint_reference_solves_each_point_and_unwraps_across_pi():
    initial = np.deg2rad([179, 45])
    target = forward_kinematics(np.deg2rad([190, 80]))
    ref = generate_reference("waypoint_ik", initial, target)
    np.testing.assert_array_equal(ref["q_reference"][0], initial)
    tips = np.array([forward_kinematics(q) for q in ref["q_reference"]])
    np.testing.assert_allclose(tips, ref["desired_points"], atol=1e-12)
    assert ref["q_reference"][-1, 0] > np.pi
    assert np.max(np.abs(np.diff(ref["q_reference"], axis=0))) < 0.02
    np.testing.assert_allclose(ref["dq_reference"], np.diff(ref["q_reference"], axis=0) / 0.02)
    np.testing.assert_allclose(ref["dq_reference"][400:], 0, atol=1e-12)


def test_waypoint_ik_drives_real_motor_motion_from_exact_singularity():
    arrays, case = run_path("waypoint_ik", initial_q=[0, 0], target=[0.5, 0])
    assert case["path_success"] and case["endpoint_success"]
    assert case["max_cross_track_mm"] < 2
    assert case["rms_timed_tracking_mm"] < 1
    assert case["torque_saturated_steps"] == 0
    assert case["min_reference_sigma_m_per_rad"] == 0
    assert case["max_reference_tracking_mm"] < 1e-8
    assert case["max_actual_to_reference_mm"] > 1
    assert np.any(arrays["torques_nm"] != 0)
    np.testing.assert_array_equal(arrays["states"][0], 0)
    assert np.max(np.abs(arrays["states"][:, :2] - arrays["q_reference"])) > 1e-3
    for state, points in zip(arrays["states"], arrays["points"], strict=True):
        np.testing.assert_allclose(points, joint_positions(state[:2]), atol=1e-12)


def test_wrong_branch_is_rejected_instead_of_jump_at_start():
    with pytest.raises(ValueError, match="positive elbow"):
        generate_reference("waypoint_ik", np.deg2rad([-40, -80]), [0.35, 0.3])


def test_too_fast_geometrically_valid_reference_is_rejected_not_clipped():
    start, end = [0.100001, -0.3], [0.100001, 0.3]
    validate_line(start, end)
    with pytest.raises(ValueError, match="planning limit"):
        generate_reference("waypoint_ik", inverse_kinematics(start)[0], end)


@pytest.mark.parametrize("dt", [0, -0.02, np.nan, np.inf, 1e20])
def test_reference_requires_valid_nonempty_clock(dt):
    with pytest.raises(ValueError):
        generate_reference("waypoint_ik", [0, 0], [0.5, 0], dt=dt)


@pytest.fixture(scope="module")
def source_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ik_source") / "source"
    run_batch(directory, per_group=1)
    return directory


def file_hashes(directory):
    return {
        p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_frozen_recomparison_keeps_baselines_source_files_and_old_loader(source_dir, tmp_path):
    before = file_hashes(source_dir)
    output = tmp_path / "ik"
    report = run_comparison(source_dir, output)
    assert file_hashes(source_dir) == before
    assert (output / "manifest.json").read_bytes() == (source_dir / "manifest.json").read_bytes()
    assert report["source_manifest_sha256"] == report["manifest_sha256"]
    assert report["controller_episode_count"] == 9
    assert report["groups"]["diagnostic"]["waypoint_ik"]["path_successes"] == 1
    assert all(t["retained_trajectories_identical"] for t in report["trials"])
    replays = load_replays(output / "trials/singular_inward")
    assert [r.metadata["key"] for r in replays] == [key for key, _ in IK_COMPARISON_METHODS]
    assert all(r.metadata["lesson"] == 11 for r in replays)
    assert "通过" in PathDemo.replay_result_text(replays[-1])
    assert "未通过" in PathDemo.replay_result_text(replays[-2])
    old = load_replays(source_dir / "trials/singular_inward")
    assert [r.metadata["key"] for r in old] == [key for key, _ in METHODS]
    saved = file_hashes(output)
    with pytest.raises(FileExistsError):
        run_comparison(source_dir, output)
    assert saved == file_hashes(output)


@pytest.mark.parametrize("damage", ["hash", "path", "duplicate", "geometry", "model", "count"])
def test_invalid_source_stops_before_writing_results(source_dir, tmp_path, damage):
    manifest = json.loads((source_dir / "manifest.json").read_bytes())
    summary = json.loads((source_dir / "summary.json").read_bytes())
    if damage == "path":
        manifest["trials"][0]["id"] = "../escape"
    elif damage == "duplicate":
        manifest["trials"][1]["id"] = manifest["trials"][0]["id"]
    elif damage == "geometry":
        manifest["trials"][0]["start_m"] = [99, 99]
    elif damage == "model":
        summary["model_xml_sha256"] = "wrong-model"
    elif damage == "count":
        manifest["per_group"] = 0
    raw = json.dumps(manifest).encode("utf-8")
    summary["manifest_sha256"] = (
        "wrong-hash" if damage == "hash" else hashlib.sha256(raw).hexdigest()
    )
    source = tmp_path / "invalid_source"
    source.mkdir()
    (source / "manifest.json").write_bytes(raw)
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "new_output"
    with pytest.raises(ValueError):
        run_comparison(source, output)
    assert not output.exists()


def test_retained_controller_drift_is_not_hidden(source_dir, tmp_path):
    source = source_dir / "trials/singular_inward"
    with np.load(source / "trajectories.npz") as archive:
        changed = {key: archive[key].copy() for key in archive.files}
    changed["jacobian_path_states"][1, 0] += 0.001
    np.savez_compressed(tmp_path / "trajectories.npz", **changed)
    with pytest.raises(ValueError, match="trajectory changed"):
        verify_retained_trajectories(source, tmp_path)
    load_source(source_dir)
