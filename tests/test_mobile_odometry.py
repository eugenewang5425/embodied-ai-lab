from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.differential_drive import compose, to_child, to_parent
from embodied_learning.experiments.mobile_frames import (
    LANDMARK_WORLD,
    SENSOR_IN_BODY,
)
from embodied_learning.experiments.mobile_frames import (
    run_case as run_lesson14_case,
)
from embodied_learning.experiments.mobile_odometry import (
    DT,
    SCENARIOS,
    VARIANTS,
    evaluate_estimate,
    run_case,
    run_experiment,
    schedule,
    simulate_truth,
)
from embodied_learning.odometry import estimate_poses, heading_error, scaled_encoder_readings
from embodied_learning.odometry_demo import METRICS, OdometryDemo, load_replays


def test_encoder_odometry_hand_cases_and_absolute_encoder_offset():
    readings = np.array([[0, 0], [4, 4], [8, 8]])
    np.testing.assert_allclose(
        estimate_poses(readings), [[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0]], atol=1e-15
    )
    np.testing.assert_allclose(
        estimate_poses(readings + [12, -30]), estimate_poses(readings), atol=1e-15
    )
    wheel = 0.3 * np.pi / (4 * 0.05)
    np.testing.assert_allclose(
        estimate_poses([[0, 0], [-wheel, wheel]])[-1], [0, 0, np.pi / 2], atol=1e-15
    )
    np.testing.assert_allclose(estimate_poses([[0, 0], [-4, -4]])[-1], [-0.2, 0, 0], atol=1e-15)
    np.testing.assert_allclose(
        estimate_poses(readings, [1, 2, np.pi / 2])[-1], [1, 2.4, np.pi / 2], atol=1e-15
    )
    np.testing.assert_array_equal(estimate_poses([[18, 34]], [1, 2, 3]), [[1, 2, 3]])
    np.testing.assert_array_equal(estimate_poses(np.zeros((5, 2))), np.zeros((5, 3)))


@pytest.mark.parametrize("invalid", [[], [0, 0], [[0, 0, 0]], [[np.nan, 0]], [[0, np.inf]]])
def test_invalid_encoder_arrays(invalid):
    with pytest.raises(ValueError):
        estimate_poses(invalid)
    with pytest.raises(ValueError):
        scaled_encoder_readings(invalid, 1)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        scaled_encoder_readings([[0, 0]], scale)


def test_signed_scale_does_not_mutate_inputs_and_can_underread():
    true = np.array([[0.0, 0.0], [-3, 3], [-4, -4]])
    before = true.copy()
    scaled = scaled_encoder_readings(true, 1.02)
    np.testing.assert_array_equal(scaled[:, 0], true[:, 0])
    np.testing.assert_allclose(scaled[:, 1], true[:, 1] * 1.02)
    estimate_poses(scaled)
    np.testing.assert_array_equal(true, before)
    assert estimate_poses(scaled)[-1, 2] < 0  # Negative cumulative right rotation remains negative.
    assert estimate_poses(scaled_encoder_readings([[0, 0], [4, 4]], 0.98))[-1, 2] < 0


def test_estimation_is_causal_and_heading_error_wraps():
    rng = np.random.default_rng(1500)
    readings = np.cumsum(rng.normal(size=(41, 2)), axis=0)
    all_poses = estimate_poses(readings)
    for count in (1, 2, 12, 30):
        np.testing.assert_array_equal(estimate_poses(readings[:count]), all_poses[:count])
    assert np.rad2deg(heading_error(np.deg2rad(1), np.deg2rad(359))) == pytest.approx(2)


@pytest.mark.parametrize("key", [k for k, _, _ in SCENARIOS])
def test_true_path_analytic_endpoint_and_shared_estimates(key):
    arrays, case = run_case(key)
    np.testing.assert_allclose(arrays["true_poses"][-1], case["expected_true_endpoint"], atol=1e-12)
    np.testing.assert_allclose(arrays["ideal_poses"], arrays["true_poses"], atol=1e-12)
    shared = simulate_truth(schedule(key)[0])
    copies = {name: value.copy() for name, value in shared.items()}
    for variant, _, scale, _ in VARIANTS:
        result, _ = evaluate_estimate(shared, scale)
        for name, value in result.items():
            np.testing.assert_array_equal(value, arrays[f"{variant}_{name}"])
        for name, value in shared.items():
            np.testing.assert_array_equal(value, copies[name])
    assert (
        arrays["right_2pct_position_error_m"][-1] > arrays["right_1pct_position_error_m"][-1] > 0.05
    )
    assert case["steps"] * DT == (12 if key == "straight" else 24)
    if key == "square":
        # Drift is not necessarily monotonic; a brief decrease is not a correction.
        assert np.any(np.diff(arrays["right_2pct_position_error_m"]) < -1e-8)
        assert arrays["right_2pct_position_error_m"][-1] > 0.15


def test_lesson14_motion_primitive_is_unchanged_and_reading_alignment():
    old, _ = run_lesson14_case("straight")
    current, _ = run_case("straight")
    np.testing.assert_array_equal(current["true_poses"][:101], old["poses"])
    np.testing.assert_array_equal(current["wheels_rad_s"][:100], old["wheels_rad_s"])
    assert current["wheel_angles_rad"][0, 1] == 0
    assert current["wheel_angles_rad"][1, 1] == pytest.approx(0.16)
    assert current["right_2pct_encoder_angles_rad"][1, 1] == pytest.approx(0.1632)
    wheels, events = schedule("square")
    assert wheels[99, 0] == wheels[99, 1] == 4
    assert wheels[100, 0] < 0 < wheels[100, 1]
    assert wheels[150, 0] == wheels[150, 1] == 4
    assert events[-1] == len(wheels)
    with pytest.raises(ValueError):
        schedule("missing")


def test_straight_biased_estimate_matches_independent_whole_arc_formula():
    arrays, _ = run_case("straight")
    # +2% right readout: v_hat=.202 m/s, omega_hat=.004/.3 rad/s.
    omega = 0.004 / 0.3
    radius, yaw = 0.202 / omega, omega * 12
    expected = [radius * np.sin(yaw), radius * (1 - np.cos(yaw)), yaw]
    np.testing.assert_allclose(arrays["right_2pct_poses"][-1], expected, atol=1e-12)
    assert arrays["right_2pct_position_error_m"][-1] == pytest.approx(0.1939889632101769)


def test_correct_transform_with_wrong_pose_still_misplaces_landmark():
    arrays, _ = run_case("straight")
    for i in (0, 1, 100, 300):
        sensed = arrays["landmark_sensor"][i]
        np.testing.assert_allclose(
            to_child(compose(arrays["true_poses"][i], SENSOR_IN_BODY), LANDMARK_WORLD),
            sensed,
            atol=1e-12,
        )
        actual_world = to_parent(compose(arrays["true_poses"][i], SENSOR_IN_BODY), sensed)
        np.testing.assert_allclose(actual_world, LANDMARK_WORLD, atol=1e-12)
        estimated_world = to_parent(compose(arrays["right_2pct_poses"][i], SENSOR_IN_BODY), sensed)
        np.testing.assert_allclose(
            estimated_world, arrays["right_2pct_mapped_landmark"][i], atol=1e-12
        )
    assert arrays["right_2pct_landmark_error_m"][-1] > 0.09
    # Mapping observations cannot alter encoder-only pose estimates.
    shared = simulate_truth(schedule("straight")[0])
    original, _ = evaluate_estimate(shared, 1.02)
    shared["landmark_sensor"] = np.zeros_like(shared["landmark_sensor"])
    changed, _ = evaluate_estimate(shared, 1.02)
    np.testing.assert_array_equal(changed["poses"], original["poses"])


@pytest.fixture
def recording(tmp_path):
    output = tmp_path / "odometry"
    report = run_experiment(output)
    return output, report


def fingerprints(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_recording_is_readonly_repeatable_and_will_not_overwrite(recording):
    output, report = recording
    before = fingerprints(output)
    replays = load_replays(output)
    for replay in replays:
        originals, _ = run_case(replay.metadata["key"])
        for name, value in replay.arrays.items():
            assert not value.flags.writeable
            np.testing.assert_array_equal(value, originals[name])
    assert len(report["source_sha256"]) == 4
    with pytest.raises(FileExistsError):
        run_experiment(output)
    assert fingerprints(output) == before


@pytest.mark.parametrize(
    "field, value",
    [("dt_s", 0), ("wheel_radius_m", 0.1), ("schema_version", 2), ("sensor_in_body", [0, 0, 0])],
)
def test_bad_recording_metadata_rejected(recording, field, value):
    output, report = recording
    report[field] = value
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(output)


@pytest.mark.parametrize(
    "change", ["checksum", "shape", "nan", "errors", "missing", "scale", "steps", "checkpoint"]
)
def test_corrupt_recordings_rejected(recording, change):
    output, report = recording
    path = output / "trajectories.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if change == "shape":
        arrays["straight_true_poses"] = arrays["straight_true_poses"][:-1]
    elif change == "nan":
        arrays["straight_true_poses"][2, 0] = np.nan
    elif change == "errors":
        arrays["straight_right_2pct_position_error_m"][:] = 0
    elif change == "missing":
        arrays.pop("straight_true_poses")
    elif change == "scale":
        report["cases"][0]["estimates"][2]["right_scale"] = 1
    elif change == "steps":
        report["cases"][0]["steps"] = 600
    elif change == "checkpoint":
        report["cases"][0]["checkpoints"] = [0, 600]
    else:
        arrays["straight_true_poses"][0, 0] = 3
    np.savez_compressed(path, **arrays)
    if change != "checksum":
        report["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(output)


@pytest.mark.isolated_tk
def test_tk_transport_same_time_comparison_variable_duration_and_drawing(recording):
    import tkinter as tk

    output, _ = recording
    root = tk.Tk()
    root.withdraw()
    demo = OdometryDemo(root, load_replays(output))
    # Exercise canvas drawing even when Tk is withdrawn; real UI QA is separate.
    demo.canvas.winfo_width = lambda: 650
    demo.canvas.winfo_height = lambda: 330
    demo.chart.winfo_width = lambda: 1000
    demo.chart.winfo_height = lambda: 125
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert demo.variant.get() == VARIANTS[2][1]
        demo.toggle()
        demo.clock.advance(0.16)
        demo.redraw()
        root.update()
        assert demo.clock.index >= 1 and not demo.clock.paused
        demo.seek(0)
        assert "增量未产生" in demo.stats.cget("text")
        demo.step()
        assert demo.clock.index == 1 and demo.clock.paused
        assert "+0.1632 rad" in demo.stats.cget("text")
        demo.next_checkpoint()
        assert demo.clock.index == 100
        demo.seek(300)
        assert "没有下一步轮速" in demo.stats.cget("text")
        for label in METRICS:
            demo.metric.set(label)
            demo.redraw()
        demo.variant.set(VARIANTS[0][1])
        demo.select_variant()
        assert demo.clock.index == 300 and demo.clock.paused
        assert "位置误差 0.00 cm" in demo.stats.cget("text")
        demo.choice.set(SCENARIOS[1][1])
        demo.select_case()
        assert demo.clock.index == 0 and demo.clock.steps == 600
        assert demo.timeline.cget("to") == 600
        demo.seek(600)
        assert "24.00 / 24.00" in demo.stats.cget("text")
        demo.choice.set(SCENARIOS[0][1])
        demo.select_case()
        assert demo.clock.steps == 300 and demo.timeline.cget("to") == 300
        demo.speed.set("0.1")
        demo.change_speed()
        assert demo.clock.speed == 0.1
    finally:
        demo.close()
