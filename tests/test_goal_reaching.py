"""Independent feedback, plant, observation and UI checks for lesson 21."""

import json
from dataclasses import replace

import numpy as np
import pytest

from embodied_learning.differential_drive import integrate_pose
from embodied_learning.experiments.goal_reaching import (
    DT,
    evaluate,
    load_recording,
    run_experiment,
    simulate,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, GEOMETRY, GoalConfig, goal_command
from embodied_learning.goal_demo import GoalDemo
from embodied_learning.landmark_localization import LANDMARKS, solve_pose
from embodied_learning.odometry import estimate_poses, heading_error


@pytest.mark.parametrize("goal", [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1]])
def test_policy_direction_and_common_wheel_limit(goal):
    decision = goal_command([0, 0, 0], goal)
    v, omega = GEOMETRY.body_velocity(decision["wheels"])
    assert abs(decision["wheels"]).max() <= DEFAULT_CONFIG.max_wheel_rad_s
    assert v >= -1e-12
    assert np.sign(omega) == np.sign(decision["angle"])
    if abs(decision["angle"]) > np.pi / 3:
        assert abs(v) < 1e-12 and decision["mode"] == "turning"


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_reject_bad_configuration(bad):
    with pytest.raises(ValueError):
        GoalConfig(max_wheel_rad_s=bad)


def test_saturation_preserves_curvature_and_evaluator_radius_cannot_change_policy():
    config = replace(DEFAULT_CONFIG, max_wheel_rad_s=0.5)
    limited = goal_command([0, 0, 0.2], [2, 1], config)["wheels"]
    normal = goal_command([0, 0, 0.2], [2, 1])["wheels"]
    np.testing.assert_allclose(limited / normal, [0.5 / abs(normal).max()] * 2)
    other = replace(DEFAULT_CONFIG, true_acceptance_radius_m=10)
    np.testing.assert_array_equal(
        goal_command([0, 0, 0], [1, 1], other)["wheels"], goal_command([0, 0, 0], [1, 1])["wheels"]
    )


@pytest.mark.parametrize("method", ["odom", "fused"])
@pytest.mark.parametrize("goal", [[1.6, 0.8], [-1, 0.5]])
def test_noiseless_closed_loop_reaches_front_and_behind(method, goal):
    arrays, result = simulate(goal, method, 42, noise_scale=0)
    assert result["true_success"] and result["controller_arrived"]
    np.testing.assert_allclose(arrays["estimated"], arrays["truth"], atol=1e-12)
    assert result["estimated_final_distance_m"] <= 0.02
    assert np.all(arrays["commands"][-11:] == 0)


def test_already_at_goal_requires_real_settling_intervals_and_timeout_is_not_arrival():
    arrays, result = simulate([0, 0], "odom", 0, noise_scale=0)
    assert result["steps"] == 10 and result["duration_s"] == 0.4
    assert result["true_success"]
    np.testing.assert_array_equal(arrays["truth"], np.zeros((11, 3)))
    arrays, result = simulate([4.8, 1.2], "fused", 0, max_steps=1)
    assert not result["controller_arrived"] and not result["true_success"]
    assert result["terminal_reason"] == "timeout"
    assert np.all(arrays["commands"][-1] == 0)


def test_noise_is_paired_but_real_paths_and_commands_change():
    a, _ = simulate([4.8, 1.2], "odom", 1_000_000)
    b, _ = simulate([4.8, 1.2], "fused", 1_000_000)
    n = min(len(a["encoder_noise"]), len(b["encoder_noise"]))
    np.testing.assert_array_equal(a["encoder_noise"][:n], b["encoder_noise"][:n])
    np.testing.assert_array_equal(a["truth"][:51], b["truth"][:51])
    np.testing.assert_array_equal(a["observations"][0], b["observations"][0])
    assert not np.allclose(a["commands"][50:n], b["commands"][50:n])
    assert not np.allclose(a["truth"][51:n], b["truth"][51:n])
    again, _ = simulate([4.8, 1.2], "fused", 1_000_000)
    for key in b:
        np.testing.assert_array_equal(again[key], b[key])


@pytest.mark.parametrize("method", ["odom", "fused"])
def test_every_step_is_causal_feedback_and_matches_plant_and_encoder_model(method):
    arrays, _ = simulate([1.6, 0.8], method, 4)
    observed = {int(f): z for f, z in zip(arrays["observation_frames"], arrays["observations"])}
    for frame in range(len(arrays["truth"]) - 1):
        np.testing.assert_array_equal(
            arrays["commands"][frame],
            goal_command(arrays["estimated"][frame], [1.6, 0.8])["wheels"],
        )
        plant = integrate_pose(
            arrays["truth"][frame], GEOMETRY.body_velocity(arrays["commands"][frame]), DT
        )
        np.testing.assert_array_equal(plant, arrays["truth"][frame + 1])
        prediction = estimate_poses(
            arrays["encoders"][frame : frame + 2], arrays["estimated"][frame]
        )[-1]
        np.testing.assert_array_equal(prediction, arrays["prior"][frame + 1])
        if frame + 1 in observed and method == "fused":
            observation = solve_pose(observed[frame + 1], LANDMARKS)
            np.testing.assert_allclose(
                observation[:2], arrays["estimated"][frame + 1, :2], atol=1e-12
            )
            assert abs(heading_error(observation[2], arrays["estimated"][frame + 1, 2])) < 1e-12
        else:
            np.testing.assert_array_equal(prediction, arrays["estimated"][frame + 1])


def test_false_arrival_is_not_reported_as_success():
    arrays, result = simulate([4.8, 1.2], "odom", 1_000_000)
    assert result["controller_arrived"] and result["false_arrival"]
    assert not result["true_success"]
    assert result["estimated_final_distance_m"] <= 0.02
    assert result["true_final_distance_m"] > 0.03
    shifted = {k: v.copy() for k, v in arrays.items()}
    shifted["truth"][:, :2] = [4.8, 1.2]
    assert evaluate(shifted, [4.8, 1.2])["true_success"]
    np.testing.assert_array_equal(shifted["commands"], arrays["commands"])


@pytest.fixture
def recording(tmp_path):
    directory = tmp_path / "goal"
    run_experiment(directory, runs=2)
    return directory


def test_recording_roundtrip_no_overwrite_and_metrics_validation(recording):
    report, results = load_recording(recording)
    assert len(results) == 8
    with pytest.raises(FileExistsError):
        run_experiment(recording, runs=2)
    report["comparisons"][0]["methods"]["odom"]["true_success_count"] += 1
    (recording / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="metrics"):
        load_recording(recording)


def test_recording_detects_missing_trial(recording):
    report = json.loads((recording / "summary.json").read_text(encoding="utf-8"))
    report["trials"].pop()
    (recording / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing"):
        load_recording(recording)


@pytest.mark.isolated_tk
def test_teaching_view_movement_result_zoom_failure_and_replay(recording):
    import tkinter as tk

    report, results = load_recording(recording)
    root = tk.Tk()
    root.withdraw()
    demo = GoalDemo(root, report, results)
    for canvas in demo.canvases.values():
        canvas.winfo_width = lambda: 530
        canvas.winfo_height = lambda: 360
    try:
        root.update()
        demo.redraw()
        assert demo.clock.paused and demo.clock.speed == 0.25 and not demo.zoom.get()
        initial = demo.canvases["odom"].coords("true_body")
        demo.seek(50)
        assert demo.canvases["odom"].coords("true_body") != initial
        demo.finish_button.invoke()
        assert demo.zoom.get() and demo.clock.index == demo.clock.steps
        assert "实际通过" in demo.notes["odom"].cget("text")
        assert demo.canvases["odom"].find_withtag("target_zone")
        for canvas in demo.canvases.values():
            assert canvas.bbox("axis_label")[3] < canvas.bbox("map_caption")[1]
        demo.case_box.current(1)
        demo.select_case()
        assert demo.clock.index == 0 and not demo.zoom.get()
        demo.failure_button.invoke()
        assert demo.run == 1 and demo.clock.paused and demo.zoom.get()
        assert "误判到达" in demo.notes["odom"].cget("text")
        demo.restart()
        demo.toggle()
        demo.clock.advance(0.16)
        assert demo.clock.index == 1
        demo.toggle()
        demo.step()
        assert demo.clock.index == 2
    finally:
        demo.close()
