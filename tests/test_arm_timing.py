import hashlib
import json

import numpy as np
import pytest

from embodied_learning.arm_path import ReferenceSpeedError, generate_reference, segment_distance
from embodied_learning.arm_path_demo import PathDemo, load_replays
from embodied_learning.experiments.arm_ik_comparison import run_comparison
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import generate_manifest, run_batch
from embodied_learning.experiments.arm_timing import (
    run_experiment,
    timing_diagnostics,
    verify_eight_second_baseline,
)
from embodied_learning.planar_arm import ArmSimulation


@pytest.mark.parametrize("movement", [8.0, 4.0, 2.0])
def test_time_scaling_preserves_geometric_path_and_hold(movement):
    initial = [0.2, 1.2]
    target = [0.4, 0.3]
    slow = generate_reference("waypoint_ik", initial, target)
    ref = generate_reference("waypoint_ik", initial, target, move_seconds=movement)
    count = round(movement / 0.02)
    stride = round(8 / movement)
    assert len(ref["dq_reference"]) == count + 150
    np.testing.assert_allclose(
        ref["desired_points"][: count + 1], slow["desired_points"][:401:stride]
    )
    np.testing.assert_allclose(
        ref["q_reference"][: count + 1], slow["q_reference"][:401:stride], atol=1e-12
    )
    np.testing.assert_allclose(
        ref["desired_velocities"][: count + 1],
        slow["desired_velocities"][:401:stride] * stride,
        atol=1e-12,
    )
    np.testing.assert_allclose(ref["dq_reference"][count:], 0, atol=1e-12)


@pytest.mark.parametrize(
    "movement,hold",
    [(0, 3), (-2, 3), (np.inf, 3), (np.nan, 3), (2.01, 3), (2, 0.1), (2, np.inf), (2, 0.51)],
)
def test_invalid_timing_is_rejected(movement, hold):
    with pytest.raises(ValueError):
        generate_reference(
            "waypoint_ik", [0, 0], [0.5, 0], move_seconds=movement, hold_seconds=hold
        )


def test_speed_rejected_before_any_physics_step(monkeypatch):
    def unexpected_step(*args, **kwargs):
        pytest.fail("A rejected reference must never be executed")

    monkeypatch.setattr(ArmSimulation, "step", unexpected_step)
    with pytest.raises(ReferenceSpeedError) as error:
        run_path("waypoint_ik", initial_q=[0, 0], target=[0.5, 0], move_seconds=2)
    assert 1.26 < error.value.peak_rad_s < 1.28


def test_movement_window_and_terminal_check_follow_actual_timing():
    arrays, case = run_path("waypoint_ik", initial_q=[0, 0], target=[0.5, 0], move_seconds=4)
    assert case["steps"] == 350 and case["duration_s"] == 7
    assert case["endpoint_success"] and not case["path_success"]
    assert case["terminal_window"]["violations"] == []
    assert case["terminal_window"]["start_s"] == 6.5
    assert case["settled_after_movement_at_s"] >= 4
    tip = arrays["points"][:201, -1]
    np.testing.assert_allclose(
        case["max_cross_track_mm"], segment_distance(tip, [0.7, 0], [0.5, 0]).max() * 1000
    )
    np.testing.assert_allclose(
        case["rms_timed_tracking_mm"],
        np.sqrt(np.mean(np.sum((tip - arrays["desired_points"][:201]) ** 2, axis=1))) * 1000,
    )


def test_real_torque_clipping_is_separate_from_planning_limit():
    spec = next(
        s for s in generate_manifest(seed=400, per_group=12)["trials"] if s["id"] == "interior_08"
    )
    arrays, case = run_path(
        "waypoint_ik", initial_q=spec["initial_q_rad"], target=spec["target_m"], move_seconds=2
    )
    metrics = timing_diagnostics(arrays, case)
    assert case["completed"] and case["endpoint_success"] and not case["path_success"]
    assert metrics["peak_reference_speed_rad_s"] < 1
    assert metrics["peak_requested_pd_torque_nm"][0] > 0.9
    assert metrics["clipped_steps"] > 0
    np.testing.assert_array_equal(
        arrays["torques_nm"], np.clip(arrays["requested_torques_nm"], -0.25, 0.25)
    )
    assert metrics["clipped_duration_s"] == metrics["clipped_steps"] * case["dt_s"]


@pytest.fixture(scope="module")
def timing_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("timing")
    baseline, source, output = root / "lesson10", root / "lesson11", root / "lesson12"
    run_batch(baseline, per_group=1)
    run_comparison(baseline, source)
    before = hashes(source)
    report = run_experiment(source, output)
    assert before == hashes(source)
    return source, output, report


def hashes(directory):
    return {
        p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


def test_batch_denominators_and_nonexecuted_records(timing_results):
    source, output, report = timing_results
    assert (output / "source_manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    assert report["groups"]["all"]["move_8s"]["path_successes"] == 3
    assert report["groups"]["all"]["move_2s"]["planning_rejected"] == 2
    assert report["groups"]["all"]["move_2s"]["executed"] == 1
    for trial in report["trials"]:
        with np.load(output / trial["relative_results"] / "trajectories.npz") as arrays:
            for case in trial["cases"]:
                key = case["key"]
                if case["status"] == "planning_rejected":
                    assert not any(name.startswith(key + "_") for name in arrays.files)
                    assert "path_success" not in case and "peak_torque_nm" not in case
                elif key == "move_8s":
                    assert case["baseline_identical"]


def test_variable_length_readonly_replays_and_rejected_plans(timing_results):
    _, output, _ = timing_results
    all_three = load_replays(output / "trials/interior_00")
    assert [len(r.torques) for r in all_three] == [550, 350, 250]
    assert all(not r.requested_torques.flags.writeable for r in all_three)
    assert "偏离路径" in PathDemo.replay_result_text(all_three[-1])
    partial = load_replays(output / "trials/singular_inward")
    assert len(partial) == 2
    assert partial[1].terminal_check["violations"] == []
    assert "偏离路径" in PathDemo.replay_result_text(partial[1])


def test_no_overwrite_and_baseline_drift(timing_results, tmp_path):
    source, output, _ = timing_results
    before = hashes(output)
    with pytest.raises(FileExistsError):
        run_experiment(source, output)
    assert hashes(output) == before
    with np.load(source / "trials/interior_00/trajectories.npz") as archive:
        states = archive["waypoint_ik_states"].copy()
    states[1, 0] += 0.001
    with pytest.raises(ValueError, match="baseline changed"):
        verify_eight_second_baseline(source / "trials/interior_00", {"states": states})


@pytest.mark.parametrize(
    "damage", ["rejected_arrays", "unknown_status", "wrong_duration", "wrong_torque"]
)
def test_loader_rejects_inconsistent_timing_results(timing_results, tmp_path, damage):
    _, output, _ = timing_results
    trial = output / "trials/singular_inward"
    report = json.loads((trial / "summary.json").read_text(encoding="utf-8"))
    with np.load(trial / "trajectories.npz") as data:
        archive = {k: data[k].copy() for k in data.files}
    if damage == "rejected_arrays":
        archive["move_2s_states"] = archive["move_8s_states"]
    elif damage == "unknown_status":
        report["cases"][-1]["status"] = "ignored"
    elif damage == "wrong_duration":
        report["cases"][0]["movement_s"] = 99
    else:
        archive["move_4s_torques_nm"][0, 0] = 9
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    np.savez_compressed(tmp_path / "trajectories.npz", **archive)
    with pytest.raises(ValueError):
        load_replays(tmp_path)


@pytest.mark.isolated_tk
def test_tk_timing_view_constructs_and_changes_clock(timing_results):
    import tkinter as tk

    _, output, _ = timing_results
    root = tk.Tk()
    root.withdraw()
    try:
        replays = load_replays(output / "trials/interior_00")
        demo = PathDemo(root, replays)
        assert demo.clock.paused and demo.clock.speed == 0.25
        demo.case.set(replays[-1].metadata["label"])
        demo.change_case()
        assert demo.clock.steps == 250 and demo.clock.paused
        demo.seek(250)
        assert "已结束，无下一步力矩" in demo.stats.get()
        assert "动作 2s + 停留 3s" in demo.status.get()
        assert "第十二课" in root.title()
    finally:
        for callback in root.tk.call("after", "info"):
            root.after_cancel(callback)
        root.destroy()
