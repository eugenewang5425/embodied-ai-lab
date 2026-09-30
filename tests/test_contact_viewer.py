"""New contact records and shared selection across camera, scene and diagnostics."""

from pathlib import Path

import pytest

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource


def source():
    directory = Path("results/contact_navigation_v6")
    if not (directory / "summary.json").exists():
        pytest.skip("generate formal round-six archives first")
    return NavigationStudySource(None, None, {"69": directory})


def test_contact_archive_does_not_mislabel_zero_error_or_margin_gate():
    s = source()
    r = s.load("69_narrow_low_14__delay_2__normal", "69")
    assert r.metadata["tracking_round"] == 6
    assert r.method_info("zero_light")["result"]["passed"]
    assert "没有视觉定位" in r.metadata["sensor_note"] or "无视觉定位" in r.metadata["sensor_note"]
    assert not r.telemetry["zero_light"]["world_velocity"].flags.writeable
    assert len(r.telemetry["zero_light"]["contact_interval"]) == len(r.timestamps)
    n = s.load("69_blocked_crate_14__delay_2__normal", "69")
    assert n.lengths["zero_light"] == 1
    assert not n.method_info("zero_light")["result"]["passed"]


@pytest.mark.isolated_tk
def test_contact_tab_one_click_is_shared_and_keeps_time():
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, source(), "69_narrow_crate_14__delay_2__normal", "69")
    try:
        app.seek(85)
        app.analysis.tabs.select(8)
        for key in app.record.method_keys:
            app.method_buttons[key].invoke()
            root.update()
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == 8
            assert app.clock.frame == 85
            assert app.analysis.contact.key == key
            assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == key
        for page in range(3, 9):
            app.analysis.tabs.select(page)
            root.update()
            assert app.analysis.tabs.index(app.analysis.tabs.select()) == page
        assert "实测运动另见接触页" in app.analysis.execution.note.cget("text")
        assert "40N" in app.analysis.contact.note.cget("text")
        assert "6cm外扩仅诊断" in app.analysis.contact.note.cget("text")
        app.select_record("69_narrow_low_14__delay_2__normal", "69")
        app.method_buttons["zero_light"].invoke()
        assert "通过" in app.stats._result_suffix("zero_light")
    finally:
        app.close()
