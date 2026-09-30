"""Known answers for bounded speed and causal map clearing."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from embodied_learning.navigation_geometry import NavigationRig, sense
from embodied_learning.navigation_map_update import EvidenceMap, SpeedController


def inputs(rig, depth_value=8.0, valid=True):
    return (
        np.full(rig.lidar_rays, 8.0),
        np.zeros(rig.lidar_rays, bool),
        np.full((rig.height, rig.width), depth_value),
        np.full((rig.height, rig.width), valid),
    )


def fixture(clear=True, trusted=False):
    rig = replace(NavigationRig(), margin_m=0)
    box = np.array([[3, 3.1, 6, 6.1, 0.3, 0.4]])
    return EvidenceMap(box, rig, clear, trusted), np.array([1.5, 6, 0.0]), rig


def test_speed_is_budgeted_before_stop_forecast_and_keeps_feedback():
    rig = NavigationRig()
    prior = np.zeros((120, 120), bool)
    normal, slow = (SpeedController(prior, rig, c) for c in (0.6, 0.2))
    for c in (normal, slow):
        c.path = np.array([[1.5, 6], [4, 6.3], [8, 6], [10.5, 6]])
    pose, velocity = np.array([1.5, 6, 0.0]), np.zeros(2)
    a, b = (c.tracking_command(pose, velocity) for c in (normal, slow))
    assert a[0] > b[0] and b[0] == 0.2
    info = slow.tracking_log[-1]
    expected = info["angular_raw"] + (b[0] - a[0]) * info["curvature_m_inv"]
    assert b[1] == pytest.approx(np.clip(expected, -1, 1))
    assert slow.latest["speed_cap"] == 0.2


def test_three_consecutive_visible_free_frames_clear_only_evidence_arm():
    for clear in (False, True):
        m, pose, rig = fixture(clear)
        for _ in range(2):
            m.update(pose, *inputs(rig))
            assert m.occupied[60, 30, 3]
        m.update(pose, *inputs(rig))
        assert bool(m.occupied[60, 30, 3]) == (not clear)


def test_invalid_frame_resets_votes_and_does_not_clear():
    m, pose, rig = fixture()
    for _ in range(2):
        m.update(pose, *inputs(rig))
    m.update(pose, *inputs(rig, np.nan, False))
    m.update(pose, *inputs(rig))
    assert m.occupied[60, 30, 3] and m.votes[60, 30, 3] == 1
    for _ in range(4):
        m.update(pose, *inputs(rig, 1.0, False))
    assert m.occupied[60, 30, 3]


def test_occluded_cell_is_kept_and_has_no_timeout_deletion():
    m, pose, rig = fixture()
    for _ in range(20):
        m.update(pose, *inputs(rig, 1.0, True))
    assert m.occupied[60, 30, 3] and m.votes[60, 30, 3] == 0


def test_current_real_hit_overrules_clear_votes():
    m, pose, rig = fixture()
    actual = np.array([[3, 3.1, 6, 6.1, 0, 1.0]])
    for _ in range(8):
        m.update(pose, *sense(actual, pose, rig))
    assert m.occupied[60, 30, 3]


def test_lidar_miss_cannot_clear_height_blind_cell():
    m, pose, rig = fixture()
    for _ in range(6):
        m.update(pose, *inputs(rig, 0, False))
    assert m.occupied[60, 30, 3]


def test_no_return_evidence_requires_explicit_fixture_contract():
    for trusted in (False, True):
        m, pose, rig = fixture(trusted=trusted)
        for _ in range(3):
            m.update(pose, *inputs(rig, 8.0, False))
        assert bool(m.occupied[60, 30, 3]) == (not trusted)


def test_camera_free_projection_uses_supplied_estimated_yaw():
    rig = NavigationRig()
    m = EvidenceMap([[6, 6.1, 4, 4.1, 0.3, 0.4]], rig, True)
    for _ in range(3):
        m.update(np.array([6, 2, np.pi / 2]), *inputs(rig))
    assert not m.occupied[40, 60, 3]


@pytest.mark.record
@pytest.mark.slow
def test_normal_speed_bridge_matches_frozen_sixth_round():
    from embodied_learning.experiments.contact_navigation import episode as old
    from embodied_learning.experiments.navigation_reinforcement import episode as new
    from embodied_learning.navigation_robustness import CONDITIONS

    _, a = old("narrow", "crate", 0, CONDITIONS["delay_2"], "zero_light")
    _, b = new("narrow", "crate", 0, CONDITIONS["delay_2"], "normal_speed")
    for key in a:
        if key != "control_wall_s":
            assert np.array_equal(a[key], b[key], equal_nan=a[key].dtype.kind not in "US"), key


def archive_source():
    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    directory = Path("results/navigation_reinforcement_v8")
    if not (directory / "summary.json").exists():
        pytest.skip("generate round-eight archives first")
    return NavigationStudySource(None, None, {"69": directory})


def test_live_archive_is_immutable_and_has_no_future_map():
    record = archive_source().load("69_street_crate_20__delay_2__shifted", "69")
    assert record.metadata["live_map"]
    for key in record.method_keys:
        d = record.telemetry[key]
        assert not d["map_frames"].flags.writeable
        assert np.any(d["map_frames"][0] != d["map_frames"][-1])
    assert record.method_info("map_evidence")["display_map"] == "当前观测通行图"


@pytest.mark.isolated_tk
def test_shared_choice_preserves_live_map_page_camera_goal_and_time():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, archive_source(), "69_street_crate_20__delay_2__shifted", "69")
    try:
        app.seek(100)
        app.analysis.tabs.select(9)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.clock.frame == 100
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 9
            assert app.analysis.map_update.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            np.testing.assert_array_equal(
                app.map.raster.get_array(), app.record.telemetry[key]["map_frames"][100]
            )
        app.seek(0)
        root.update()
        np.testing.assert_array_equal(
            app.map.raster.get_array(), app.record.telemetry[key]["map_frames"][0]
        )
        app.choose_view("split")
        root.update()
        for key in app.record.method_keys:
            np.testing.assert_array_equal(
                app.analysis.map_rasters[key].get_array(),
                app.record.telemetry[key]["map_frames"][0],
            )
    finally:
        app.close()
