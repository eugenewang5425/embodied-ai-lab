from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from embodied_learning.differential_drive import (
    DriveGeometry,
    compose,
    integrate_pose,
    rotation,
    to_child,
    to_parent,
)
from embodied_learning.experiments.mobile_frames import (
    ARRAY_WIDTHS,
    CASES,
    DT,
    GEOMETRY,
    LANDMARK_WORLD,
    SENSOR_IN_BODY,
    expected_endpoint,
    run_case,
    run_experiment,
    wheel_schedule,
)
from embodied_learning.mobile_demo import MobileDemo, load_replays


@pytest.mark.parametrize(
    "wheels, expected",
    [
        ([4, 4], [0.2, 0]),
        ([-4, -4], [-0.2, 0]),
        ([-2, 2], [0, 2 / 3]),
        ([2, 4], [0.15, 1 / 3]),
        ([4, 2], [0.15, -1 / 3]),
        ([0, 0], [0, 0]),
    ],
)
def test_wheel_order_signs_and_units(wheels, expected):
    np.testing.assert_allclose(GEOMETRY.body_velocity(wheels), expected, atol=1e-15)


@pytest.mark.parametrize("bad", [0, -0.1, np.nan, np.inf])
def test_invalid_geometry_and_dt(bad):
    with pytest.raises(ValueError):
        DriveGeometry(radius_m=bad)
    with pytest.raises(ValueError):
        DriveGeometry(track_m=bad)
    with pytest.raises(ValueError):
        integrate_pose([0, 0, 0], [1, 0], bad)


@pytest.mark.parametrize("bad", [[1], [1, 2, 3], [np.nan, 0], [0, np.inf]])
def test_invalid_wheel_input(bad):
    with pytest.raises(ValueError):
        GEOMETRY.body_velocity(bad)


def test_hand_calculated_90_degree_frames_and_composition_order():
    world_from_body = [1, 2, np.pi / 2]
    body_from_sensor = [0.2, 0.1, -np.pi / 2]
    np.testing.assert_allclose(to_parent(world_from_body, [1, 0]), [1, 3], atol=1e-15)
    np.testing.assert_allclose(to_child(world_from_body, [0, 2]), [0, 1], atol=1e-15)
    combined = compose(world_from_body, body_from_sensor)
    np.testing.assert_allclose(combined, [0.9, 2.2, 0], atol=1e-15)
    np.testing.assert_allclose(to_parent(combined, [0.3, 0]), [1.2, 2.2], atol=1e-15)
    assert not np.allclose(compose(body_from_sensor, world_from_body), combined)


def test_seeded_transform_roundtrip_and_chain_matches_direct_transform():
    rng = np.random.default_rng(1400)
    for _ in range(100):
        first, second, point = rng.normal(size=3), rng.normal(size=3), rng.normal(size=2)
        np.testing.assert_allclose(to_child(first, to_parent(first, point)), point, atol=3e-15)
        np.testing.assert_allclose(
            to_parent(compose(first, second), point),
            to_parent(first, to_parent(second, point)),
            atol=3e-15,
        )
    with pytest.raises(ValueError):
        rotation(np.nan)
    with pytest.raises(ValueError):
        to_parent([0, 0, np.inf], [0, 0])


def test_integrator_analytic_circle_zero_limit_reverse_and_subdivision():
    pose = np.array([1.0, 2.0, np.pi / 2])
    original = pose.copy()
    np.testing.assert_allclose(integrate_pose(pose, [0.2, 0], 2), [1, 2.4, np.pi / 2], atol=1e-15)
    np.testing.assert_allclose(
        integrate_pose([0, 0, 0], [1, 1], 2 * np.pi), [0, 0, 2 * np.pi], atol=1e-15
    )
    np.testing.assert_allclose(
        integrate_pose(pose, [-0.2, 1e-14], 2), [1, 1.6, np.pi / 2], atol=3e-14
    )
    total = integrate_pose(pose, [0.3, -0.7], 4)
    divided = pose.copy()
    for _ in range(100):
        divided = integrate_pose(divided, [0.3, -0.7], 0.04)
    np.testing.assert_allclose(divided, total, atol=1e-14)
    np.testing.assert_array_equal(pose, original)


@pytest.mark.parametrize("key", [key for key, _, _ in CASES])
def test_every_case_matches_whole_motion_analytic_solution_and_frame_invariants(key):
    arrays, metrics = run_case(key)
    assert arrays["poses"].shape == (101, 3)
    assert arrays["wheels_rad_s"].shape == (100, 2)
    assert set(arrays) == set(ARRAY_WIDTHS)
    np.testing.assert_allclose(arrays["poses"][-1], expected_endpoint(key), atol=1e-14)
    np.testing.assert_allclose(
        arrays["reconstructed_world"], np.tile(LANDMARK_WORLD, (101, 1)), atol=1e-14
    )
    for i, pose in enumerate(arrays["sensor_world_poses"]):
        np.testing.assert_allclose(
            to_child(pose, LANDMARK_WORLD), arrays["landmark_sensor"][i], atol=1e-14
        )
    assert metrics["max_reconstruction_error_m"] < 1e-14
    assert metrics["max_wrong_mapping_error_m"] > 0.3
    again, _ = run_case(key)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], again[name])


def test_turn_then_drive_uses_next_interval_and_does_not_strafe():
    arrays, _ = run_case("turn_then_drive")
    np.testing.assert_allclose(arrays["poses"][50], [0, 0, np.pi / 2], atol=1e-14)
    assert arrays["body_velocity"][49, 0] == 0
    assert arrays["body_velocity"][50, 0] == pytest.approx(0.2)
    assert arrays["body_velocity"][50, 1] == 0
    np.testing.assert_allclose(arrays["poses"][51, :2], [0, 0.008], atol=1e-14)
    np.testing.assert_allclose(arrays["landmark_body"][-1], [0.4, -1.1], atol=1e-14)
    # At theta=0 even initially, ignoring sensor installation is already wrong.
    assert np.linalg.norm(arrays["wrong_world"][0] - LANDMARK_WORLD) > 0.3


def test_arcs_are_mirrored_and_spin_has_no_body_translation():
    left, _ = run_case("left_arc")
    right, _ = run_case("right_arc")
    spin, _ = run_case("spin")
    np.testing.assert_allclose(left["poses"] * [1, -1, -1], right["poses"], atol=1e-15)
    np.testing.assert_array_equal(spin["poses"][:, :2], np.zeros((101, 2)))
    # The offset sensor DOES translate during rotation about the body centre.
    assert np.linalg.norm(spin["sensor_world_poses"][-1, :2] - SENSOR_IN_BODY[:2]) > 0.1
    with pytest.raises(ValueError):
        wheel_schedule("unknown")
    with pytest.raises(ValueError):
        expected_endpoint("unknown")


@pytest.fixture
def recording(tmp_path):
    directory = tmp_path / "mobile"
    report = run_experiment(directory)
    return directory, report


def fingerprints(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_roundtrip_loader_is_read_only_and_output_cannot_overwrite(recording):
    directory, report = recording
    before = fingerprints(directory)
    replays = load_replays(directory)
    assert len(replays) == 5
    for replay in replays:
        originals, _ = run_case(replay.metadata["key"])
        assert replay.dt == DT and replay.steps == 100
        for name, value in replay.arrays.items():
            assert not value.flags.writeable
            np.testing.assert_array_equal(value, originals[name])
    with pytest.raises(FileExistsError):
        run_experiment(directory)
    assert before == fingerprints(directory)
    assert report["model"] == "ideal_no_slip_velocity_kinematics"
    assert len(report["source_sha256"]) == 2


@pytest.mark.parametrize(
    "field, value",
    [("dt_s", 0), ("schema_version", 2), ("wheel_radius_m", 0.1), ("sensor_in_body", [0, 0, 0])],
)
def test_loader_rejects_incompatible_metadata(recording, field, value):
    directory, report = recording
    report[field] = value
    (directory / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(directory)


@pytest.mark.parametrize("change", ["checksum", "missing", "shape", "nan", "steps", "duplicate"])
def test_loader_rejects_corruption(recording, change):
    directory, report = recording
    path = directory / "trajectories.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if change == "missing":
        arrays.pop("straight_poses")
    elif change == "shape":
        arrays["straight_poses"] = arrays["straight_poses"][:-1]
    elif change == "nan":
        arrays["straight_poses"][5, 0] = np.nan
    elif change == "steps":
        report["cases"][0]["steps"] = True
    elif change == "duplicate":
        report["cases"][1]["key"] = "straight"
    else:
        arrays["straight_poses"][0, 0] = 5
    np.savez_compressed(path, **arrays)
    if change != "checksum":
        report["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        load_replays(directory)


@pytest.mark.isolated_tk
def test_tk_playback_controls_boundaries_and_labels(recording):
    import tkinter as tk

    directory, _ = recording
    root = tk.Tk()
    root.withdraw()
    demo = MobileDemo(root, load_replays(directory))
    try:
        root.update()
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert demo.replay.metadata["key"] == "turn_then_drive"
        demo.toggle()
        demo.clock.advance(0.16)
        demo.redraw()
        root.update()
        assert demo.clock.index == 1 and not demo.clock.paused
        demo.step()
        assert demo.clock.index == 2 and demo.clock.paused
        demo.seek(50)
        assert "θ=+90.0°" in demo.stats.cget("text")
        assert "左 +4.000 / 右 +4.000" in demo.stats.cget("text")
        demo.show_wrong.set(True)
        demo.redraw()
        demo.seek(100)
        assert "没有下一步轮速" in demo.stats.cget("text")
        demo.toggle()
        assert demo.clock.index == 0 and not demo.clock.paused
        demo.choice.set(CASES[2][1])
        demo.select_case()
        assert demo.clock.index == 0 and demo.clock.paused
        assert demo.replay.metadata["key"] == "left_arc"
        demo.speed.set("0.1")
        demo.change_speed()
        assert demo.clock.speed == 0.1
    finally:
        demo.close()
