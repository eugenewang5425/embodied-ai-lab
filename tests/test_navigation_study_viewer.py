import json
from dataclasses import replace

import numpy as np
import pytest

from embodied_learning.replay_viewer.navigation_studies import NavigationStudySource


@pytest.fixture
def source():
    from pathlib import Path

    if not Path("results/observed_navigation_v1/summary.json").is_file():
        pytest.skip("formal replay archives are generated separately")
    return NavigationStudySource()


def test_checked_records_preserve_short_lengths_and_sensor_origins(source):
    r = source.load("67_narrow_crate_0", "67")
    assert r.lengths["depth"] == 1
    assert len(r.timestamps) > 1
    assert not r.depth["depth"]["values"].flags.writeable
    scan = r.lidar["depth"]
    pose = r.track_by_key["depth"].poses[0]
    angles = np.arange(180) * 2 * np.pi / 180 + pose[2]
    expected = (
        pose[:2] + [0.08, 0] + np.c_[np.cos(angles), np.sin(angles)] * scan["ranges"][0, :, None]
    )
    assert np.allclose(r.scan_world(0, "depth"), expected[scan["hits"][0]])
    assert len(r.maps) == 1  # no final observed map presented as current knowledge
    for e in (e for e in source.entries if e.variant == "68"):
        rr = source.load(e.case, e.variant)
        assert rr.metadata["methods"]["repaired"]["result"]["reached"] in (7, 8)


def test_floating_bar_and_whole_body_geometry_are_rendered(source):
    from embodied_learning.replay_viewer.scene import build_model

    r = source.load("67_plaza_beam_0", "67")
    model = build_model(r)
    # The 15 cm-high bar must start at z=.42, not become a floor-to-ceiling box.
    ids = np.flatnonzero(np.all(np.isclose(model.geom_pos[:, :2], [6, 6]), axis=1))
    assert any(
        np.isclose(model.geom_pos[i, 2], 0.495) and np.isclose(model.geom_size[i, 2], 0.075)
        for i in ids
    )
    body = model.body("actor_0")
    adr = int(body.geomadr[0])
    assert np.allclose(model.geom_size[adr], [0.275, 0.22, 0.15])
    bad = {"depth": {"values": np.zeros((2, 3, 4)), "valid": np.zeros((2, 3, 4), bool)}}
    with pytest.raises(ValueError, match="timestamps"):
        replace(r, depth=bad)


@pytest.mark.isolated_tk
def test_one_click_switch_preserves_calibration_tab_and_time(source, tmp_path):
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, source, "67_plaza_low_0", "67")
    try:
        root.update()
        app.seek(50)
        app.analysis.tabs.select(3)
        root.update()
        app.method_buttons["depth"].invoke()
        root.update()
        assert app.clock.frame == 50
        assert app.analysis.tabs.index(app.analysis.tabs.select()) == 3
        assert app.analysis.safety.key == "depth"
        assert app.camera.focus_key == app.scene.engine.focus_key == app.stats.focus_key == "depth"
        assert app.camera.image.size == (640, 480)
        app.method_buttons["lidar"].invoke()
        root.update()
        assert app.analysis.safety.key == "lidar"
        assert app.clock.frame == 50
        app.select_record("68_1", "68")
        app.method_buttons["repaired"].invoke()
        app.seek(len(app.record.timestamps) - 1)
        assert "7/8" in app.stats.task_state.cget("text")
        assert "未真正到达" in app.stats.task_state.cget("text")
        path = tmp_path / "frame.json"
        app.export_current(path)
        assert json.loads(path.read_text(encoding="utf8"))["focus_method"] == "repaired"
    finally:
        app.close()
