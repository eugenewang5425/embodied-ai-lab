import copy
import hashlib
import json

import numpy as np
import pytest

from embodied_learning.arm_path import METHODS, terminal_window_diagnostics, validate_line
from embodied_learning.arm_path_demo import PathDemo, load_replays
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import aggregate, generate_manifest, run_batch
from embodied_learning.planar_arm import forward_kinematics, joint_positions


def test_manifest_is_seeded_group_independent_and_not_outcome_filtered():
    first = generate_manifest(400, 2)
    assert first == generate_manifest(400, 2)
    assert first != generate_manifest(401, 2)
    larger = generate_manifest(400, 3)
    for group in ("interior", "near_extension"):
        a = [t for t in first["trials"] if t["group"] == group]
        b = [t for t in larger["trials"] if t["group"] == group]
        assert a == b[:2]
    for trial in generate_manifest()["trials"]:
        start = forward_kinematics(trial["initial_q_rad"])
        np.testing.assert_allclose(start, trial["start_m"])
        validate_line(start, trial["target_m"])
        if trial["group"] == "near_extension":
            assert 0.6995 <= np.linalg.norm(trial["target_m"]) <= 0.69999
        assert "success" not in trial


@pytest.mark.parametrize(
    "seed, count", [(-1, 2), (2**32, 2), (True, 2), (1.5, 2), (400, 0), (400, 101), (400, False)]
)
def test_bad_sampler_arguments_rejected(seed, count):
    with pytest.raises(ValueError):
        generate_manifest(seed, count)


def test_impossible_geometry_never_becomes_a_controller_failure():
    manifest = generate_manifest()
    assert len(manifest["trials"]) == 25
    assert sum(t["kind"] == "seeded" for t in manifest["trials"]) == 24
    assert len(manifest["geometry_checks"]) == 3
    assert all(c["rejected"] for c in manifest["geometry_checks"])
    crossing = next(c for c in manifest["geometry_checks"] if c["id"] == "line_crosses_hole")
    assert np.linalg.norm(crossing["start_m"]) == pytest.approx(0.35)
    assert np.linalg.norm(crossing["target_m"]) == pytest.approx(0.35)
    with pytest.raises(ValueError):
        run_path("jacobian_path", target=[0.75, 0])


def test_exact_singularity_stalls_reference_not_motor_and_cross_error_is_insufficient():
    arrays, case = run_path("jacobian_path", initial_q=[0, 0], target=[0.5, 0])
    np.testing.assert_array_equal(arrays["q_reference"], 0)
    np.testing.assert_array_equal(arrays["states"], 0)
    np.testing.assert_array_equal(arrays["torques_nm"], 0)
    assert case["completed"] and not case["failure_reason"]
    assert not case["endpoint_success"] and not case["path_success"]
    assert case["max_cross_track_mm"] == pytest.approx(0)
    assert case["final_tip_error_mm"] == pytest.approx(200)
    assert case["rms_actual_to_reference_mm"] == 0
    assert case["rms_reference_tracking_mm"] > 100
    assert case["torque_saturated_steps"] == 0
    assert case["terminal_window"]["violations"] == ["tip_position", "joint_position"]


def test_terminal_audit_catches_duration_not_just_last_frame():
    specification = next(
        t for t in generate_manifest(400, 2)["trials"] if t["id"] == "near_extension_01"
    )
    _, case = run_path(
        "jacobian_path", initial_q=specification["initial_q_rad"], target=specification["target_m"]
    )
    assert max(case["final_joint_error_rad"]) < 0.01
    assert case["final_tip_error_mm"] < 0.1
    assert case["max_cross_track_mm"] < 2
    assert not case["endpoint_success"]
    assert case["terminal_window"]["max_joint_error_rad"][1] > 0.01
    assert case["terminal_window"]["violations"] == ["joint_position"]


def test_short_terminal_window_is_not_success():
    states = np.zeros((20, 4))
    points = np.repeat(joint_positions([0, 0])[None], len(states), axis=0)
    audit = terminal_window_diagnostics(states, points, [0.7, 0], [0, 0], 0.02)
    assert audit["violations"] == ["insufficient_post_movement_window"]


def test_batch_archives_are_paired_replayable_and_preserve_failures(tmp_path):
    output = tmp_path / "batch"
    report = run_batch(output, per_group=2)
    assert report["controller_episode_count"] == 15
    assert report["random_path_count"] == 4
    assert report["groups"]["near_extension"]["jacobian_path"]["path_successes"] == 1
    assert (
        report["manifest_sha256"]
        == hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    )
    for trial in report["trials"]:
        replays = load_replays(output / trial["relative_results"])
        assert len(replays) == 3
        for replay in replays:
            np.testing.assert_allclose(replay.states[0, :2], trial["initial_q_rad"])
            np.testing.assert_allclose(replay.metadata["target_m"], trial["target_m"])
            np.testing.assert_array_equal(replay.desired, replays[0].desired)
            assert not replay.states.flags.writeable
            assert replay.terminal_check == replay.metadata["terminal_window"]
        with np.load(output / trial["relative_results"] / "trajectories.npz") as archive:
            for method, _ in METHODS:
                np.testing.assert_allclose(
                    archive[f"{method}_torques_nm"],
                    np.clip(archive[f"{method}_requested_torques_nm"], -0.25, 0.25),
                    atol=1e-14,
                )
    diagnostic = load_replays(output / "trials/singular_inward")[-1]
    assert "未通过" in PathDemo.replay_result_text(diagnostic)
    assert "末端位置" in PathDemo.replay_result_text(diagnostic)
    files = {
        p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        run_batch(output)
    assert files == {
        p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    # Incomplete physical failures stay in denominators, but not full-horizon medians.
    fake_trials = copy.deepcopy(report["trials"])
    case = fake_trials[0]["cases"][-1]
    case.update(completed=False, path_success=False, endpoint_success=False, failure_reason="test")
    group = aggregate(fake_trials)["interior"]["jacobian_path"]
    assert group["episodes"] == 2 and group["completed"] == 1
    assert group["physical_failures"] == 1 and group["path_successes"] == 1
    json.dumps(report, allow_nan=False)
