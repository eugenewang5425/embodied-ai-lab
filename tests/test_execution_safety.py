"""Known physical and causal failures that a queue/local protector must handle."""

from collections import deque

import numpy as np
import pytest

from embodied_learning.execution_safety import GuardedTransport, command_stop_path, queued_stop_path
from embodied_learning.navigation_geometry import NavigationRig, advance


def test_queue_forecast_includes_pending_acceleration_and_keeps_input_untouched():
    rig = NavigationRig()
    pending = [(np.array([0.6, 0.2]), False, j) for j in range(4)]
    path, u = queued_stop_path([0.28, 0.2], pending, [0, 0], True, rig)
    np.testing.assert_allclose(u[:4, 0], [0.36, 0.44, 0.52, 0.6])
    assert path[-1, 0] > 0.3
    assert u[4, 1] / u[4, 0] == pytest.approx(u[3, 1] / u[3, 0])
    np.testing.assert_array_equal(pending[0][0], [0.6, 0.2])
    p = np.zeros(3)
    for v, w in u:
        p = advance(p, v, w, 0.1)
    np.testing.assert_allclose(path[-1], p, atol=1e-12)


def test_local_candidate_path_does_not_restart_acceleration_from_zero():
    rig = NavigationRig()
    path = command_stop_path([0.6, 0], rig)
    # 0.06m now + .052+.044+.036+.028+.020+.012+.004 in subsequent ticks.
    assert path[-1, 0] == pytest.approx(0.256)


def test_local_obstacle_brake_purges_queued_motion_and_preserves_limits():
    rig = NavigationRig()
    t = GuardedTransport(4, rig, True)
    t.queue = deque((np.array([0.6, 0]), False, j) for j in range(4))
    # Actual body needs 0.30m ahead + 0.06m margin; point is inside stop sweep.
    values = t.execute([0.6, 0], False, 4, [0.6, 0], [[0.55, 0, 0.15]])
    assert values[4] and not values[5] and values[6] <= 0
    assert values[0][0] == pytest.approx(0.52)
    assert t.pending_motion == 0 and not t.latched
    # Safe immediate stop may be impossible if obstacle is detected too late.
    assert values[0][0] != 0


def test_expiry_checks_age_latches_and_late_plan_cannot_restart_car():
    rig = NavigationRig()
    t = GuardedTransport(0, rig, True)
    points = np.empty((0, 3))
    u = t.execute([0.6, 0], False, 0, [0.6, 0], points)[0]
    first = t.execute([0, 0], True, 1, u, points, False)
    assert not first[4] and first[7] == 1 and first[0][0] == pytest.approx(0.6)
    second = t.execute([0, 0], True, 2, first[0], points, False)
    assert second[4] and second[5] and second[0][0] == pytest.approx(0.52)
    u = second[0]
    for frame in range(3, 14):
        u = t.execute([0.6, 0], False, frame, u, points, True)[0]
    np.testing.assert_array_equal(u, [0, 0])
    assert t.latched


def test_blackout_reference_advances_and_does_not_wait_for_planner():
    t = GuardedTransport(0, NavigationRig(), False)
    u = t.execute([0, 0], True, 30, [0.6, 0.1], np.empty((0, 3)), False)
    np.testing.assert_array_equal(u[0], [0.6, 0.1])
    assert not u[4] and u[3] == -2


def test_registered_four_arm_design_preserves_negative_and_unseen_seeds():
    from collections import Counter

    from embodied_learning.experiments.execution_navigation import registered_cases

    cases = registered_cases()
    assert Counter(c[3] for c in cases) == {
        "bridge": 3,
        "scan": 108,
        "environment": 8,
        "negative": 4,
        "blackout": 12,
    }
    assert {c[2] for c in cases if c[3] == "scan"} == {11, 12, 13}
    assert len({(a, b, c, e, f, g) for a, b, c, _, e, f, g in cases}) == 135


def archived_source():
    import json
    from pathlib import Path

    from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource

    directory = Path("results/execution_safety_v5")
    path = directory / "summary.json"
    if not path.exists() or len(json.loads(path.read_text(encoding="utf8"))["rows"]) != 135:
        pytest.skip("generate the registered 135-case batch separately")
    return NavigationStudySource(None, None, {"69": directory})


def test_window_adapter_keeps_conditions_separate_and_protection_telemetry():
    source = archived_source()
    for profile in ("normal", "blackout"):
        r = source.load(f"69_narrow_crate_11__delay_4__{profile}", "69")
        assert len(r.method_keys) == 4 and r.metadata["tracking_round"] == 5
        assert "没有视觉定位" in r.metadata["sensor_note"]
        for key in r.method_keys:
            assert r.method_info(key)["result"]["profile"] == profile
            assert "local_override" in r.telemetry[key]
            assert r.goal_info(51, key) is not None
    bridge = source.load("69_narrow_crate_8__zero__normal", "69")
    assert bridge.method_keys == ("baseline_exec",)


@pytest.mark.isolated_tk
def test_one_action_switches_camera_scene_stats_and_local_protection_page():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, archived_source(), "69_narrow_crate_11__delay_4__blackout", "69")
    try:
        app.seek(51)
        app.analysis.tabs.select(7)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert (
                app.clock.frame == 51 and app.analysis.tabs.index(app.analysis.tabs.select()) == 7
            )
            assert app.analysis.execution.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
            assert "本地接管" in app.analysis.execution.note.cget("text")
        app.select_record("69_narrow_crate_11__zero__normal", "69")
        app.analysis.tabs.select(7)
        root.update()
        assert app.analysis.execution.key in app.record.method_keys
    finally:
        app.close()
