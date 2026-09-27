"""Known-answer synchronization, data-boundary and real-window checks."""

import json
import math
import time
from dataclasses import replace

import numpy as np
import pytest

from embodied_learning.replay_viewer.clock import ReplayClock
from embodied_learning.replay_viewer.model import CameraSpec, ReplayEntry, ReplayRecord, Track


def small_record(offset=0.5):
    times = np.array([0.0, 0.04, 0.10, 0.22, 0.40])
    truth = np.array([[1 + t, 1.5, math.radians(179)] for t in times])
    estimate = truth.copy()
    estimate[:, 0] += offset
    estimate[:, 2] = math.radians(-179)
    grid = np.zeros((20, 30), dtype=bool)
    grid[[0, -1], :] = True
    grid[:, [0, -1]] = True
    return ReplayRecord(
        "Known answer",
        times,
        (Track("truth", "真值", "#1684b8", truth), Track("est", "测试估计", "#cd3c88", estimate)),
        {"参考图": grid, "测试图": grid},
        0.2,
    )


def test_metrics_known_answer_and_angle_wrap():
    record = small_record()
    assert np.allclose(record.errors["est"], 0.5)
    assert np.allclose(record.yaw_errors["est"], 2.0)
    assert record.statistics["est"]["mean_m"] == pytest.approx(0.5)
    assert record.extent == (0.0, 6.0, 0.0, 4.0)


def test_record_owns_readonly_arrays():
    record = small_record()
    with pytest.raises(ValueError):
        record.tracks[0].poses[0, 0] = 20
    with pytest.raises(ValueError):
        record.maps["参考图"][0, 0] = False


@pytest.mark.parametrize("times", ([0.0, 0.0], [1.0, 0.0], [0.0, float("nan")]))
def test_record_rejects_ambiguous_time(times):
    with pytest.raises(ValueError, match="timestamps"):
        ReplayRecord("bad", np.array(times), (), {}, 0.1)


def test_record_rejects_misaligned_track_lengths():
    record = small_record()
    with pytest.raises(ValueError, match="poses"):
        ReplayRecord(
            "bad",
            record.timestamps,
            (Track("truth", "truth", "k", record.tracks[0].poses[:-1]),),
            record.maps,
            0.2,
        )


def test_clock_follows_time_not_number_of_render_callbacks():
    clock = ReplayClock([0.0, 0.04, 0.10, 0.22, 0.40])
    clock.play(now=10.0)
    assert clock.advance(now=10.099) == 1
    assert clock.advance(now=10.3) == 3
    clock.pause(now=10.3)
    assert clock.advance(now=90) == 3
    clock.seek(1, now=100)
    clock.set_speed(2, now=100)
    clock.play(now=100)
    assert clock.advance(now=100.1) == 3
    assert clock.advance(now=101) == 4
    assert not clock.playing


def test_clock_boundaries_and_speed_guard():
    clock = ReplayClock([0.0, 0.1, 0.2])
    clock.seek(999)
    assert clock.frame == 2
    clock.play(now=0)
    assert clock.frame == 0
    clock.seek(-10)
    assert clock.frame == 0
    with pytest.raises(ValueError):
        clock.set_speed(0)


class OtherExperimentSource:
    """Independent adapter proves the window has no benchmark-file dependency."""

    title = "已知答案实验窗口"
    entries = (
        ReplayEntry("case", "A", "短轨迹"),
        ReplayEntry("case", "B", "短轨迹"),
        ReplayEntry("other", "A", "另一回合"),
    )

    def load(self, case, variant):
        # Leave time to click Pause after a real OpenGL/Tk render. With a 0.4 s
        # recording, slow renders can finish playback before the second click;
        # that click then correctly restarts it. End-of-stream clock behavior
        # is checked separately by test_clock_boundaries_and_speed_guard.
        return replace(
            small_record(0.5 if variant == "A" else 1.0),
            timestamps=np.array([0.0, 0.04, 0.10, 0.22, 20.0]),
        )


def test_robot_camera_extrinsics_and_clean_view_follow_truth():
    from embodied_learning.replay_viewer.scene import SceneRenderer

    record = small_record()
    poses = record.tracks[0].poses.copy()
    poses[0] = [1.0, 1.5, 0]
    poses[1] = [1.0, 1.5, np.pi / 2]
    record = replace(
        record,
        tracks=(replace(record.tracks[0], poses=poses), record.tracks[1]),
        camera=CameraSpec(height_m=0.25, forward_m=0.2, vertical_fov_deg=70),
    )
    engine = SceneRenderer()
    engine.load(record)
    try:
        first = np.array(engine.render(0, (320, 240), {"truth", "est"}, True, "robot"))
        cid = engine.model.camera("robot_front").id
        assert np.allclose(engine.data.cam_xpos[cid], [1.2, 1.5, 0.25])
        orientation = engine.data.cam_xmat[cid].reshape(3, 3)
        assert np.allclose(-orientation[:, 2], [1, 0, 0], atol=1e-12)
        assert np.allclose(orientation[:, 1], [0, 0, 1], atol=1e-12)
        assert engine.model.cam_fovy[cid] == 70
        # First-person pixels do not contain optional estimate markers or tails.
        clean = np.array(engine.render(0, (320, 240), set(), False, "robot"))
        assert np.array_equal(first, clean)
        turned = np.array(engine.render(1, (320, 240), set(), False, "robot"))
        assert np.allclose(engine.data.cam_xpos[cid], [1.0, 1.7, 0.25])
        orientation = engine.data.cam_xmat[cid].reshape(3, 3)
        assert np.allclose(-orientation[:, 2], [0, 1, 0], atol=1e-12)
        assert np.mean(np.abs(first.astype(float) - turned.astype(float))) > 1
    finally:
        engine.close()


def test_close_obstacle_is_rendered_at_its_metric_depth_in_large_world():
    from embodied_learning.experiments.grid_nav import Obstacle
    from embodied_learning.replay_viewer.scene import SceneRenderer

    base = small_record()
    record = replace(
        base,
        tracks=tuple(replace(t, poses=np.zeros_like(t.poses)) for t in base.tracks),
        maps={"obstacles": np.zeros((240, 240), bool)},
        resolution_m=0.1,
        primitives=(Obstacle("rect", 0.36, 0, 0.01, 0.4),),
        camera=CameraSpec(0.55, 0.18, 65),
        metadata={"robot_radius_m": 0.25, "scene_style": [{"height": 1.5}]},
    )
    engine = SceneRenderer()
    engine.load(record)
    try:
        engine.render(0, (320, 240), set(), False, "robot")
        # Near face x=.35, pinhole x=.18: a solid wall 17 cm in front.
        # It is outside the entire 25 cm robot footprint and must occlude.
        assert engine.model.vis.map.znear * engine.model.stat.extent == pytest.approx(0.01)
        engine.renderer.enable_depth_rendering()
        depth = engine.renderer.render()
        assert depth[120, 160] == pytest.approx(0.17, abs=1e-4)
    finally:
        engine.close()


def test_resizing_overview_does_not_destroy_camera_framebuffer():
    from embodied_learning.replay_viewer.scene import SceneRenderer

    record = small_record()
    camera, overview = SceneRenderer(), SceneRenderer()
    camera.load(record)
    overview.load(record)
    try:
        overview.render(0, (480, 320), {"truth"})
        before = np.array(camera.render(0, (320, 240), set(), False, "robot"))
        assert before.mean() > 1
        # Resizing closes an existing renderer while the other context is current.
        overview.render(0, (500, 320), {"truth"})
        after = np.array(camera.render(0, (320, 240), set(), False, "robot"))
        assert np.array_equal(before, after)
        # Exercise the reverse direction as well.
        overview_before = np.array(overview.render(0, (500, 320), {"truth"}))
        camera.render(0, (340, 240), set(), False, "robot")
        overview_after = np.array(overview.render(0, (500, 320), {"truth"}))
        assert np.array_equal(overview_before, overview_after)
    finally:
        camera.close()
        overview.close()


@pytest.mark.isolated_tk
def test_window_controls_keep_all_panels_on_same_record_and_frame(tmp_path):
    import tkinter as tk

    from embodied_learning.replay_viewer.window import ReplayWindow

    root = tk.Tk()
    app = ReplayWindow(root, OtherExperimentSource(), "case", "A")

    def pump(seconds=0.1):
        until = time.perf_counter() + seconds
        while time.perf_counter() < until:
            root.update()
            time.sleep(0.01)

    try:
        pump(0.4)
        assert app.stats.winfo_width() > 350
        assert app.camera.photo is not None
        assert app.camera.image.size == (640, 480)
        app.scale.set(3)  # exercise the actual slider command
        root.update()
        assert app.clock.frame == 3
        assert all(p.frame == 3 for p in app.panels)
        assert app.camera.record is app.record
        assert "0.22 s" in app.camera.info.cget("text")
        app.variant_var.set("B")
        app.variant_box.event_generate("<<ComboboxSelected>>")
        root.update()
        assert app.clock.frame == 3
        assert float(app.stats.table.item("est", "values")[1]) == pytest.approx(1.0)
        app.track_vars["est"].set(False)
        app.refresh()
        assert not app.map.lines["est"].get_visible()
        assert not app.error.lines["est"].get_visible()
        before = app.scene.engine.camera.azimuth
        app.scene.canvas.event_generate("<ButtonPress-1>", x=50, y=50)
        app.scene.canvas.event_generate("<B1-Motion>", x=90, y=70)
        root.update()
        assert app.scene.engine.camera.azimuth != before
        app.seek_time(0.105)
        assert app.clock.frame == 2
        app.step(-1)
        assert app.clock.frame == 1
        app.seek(0)
        app.speed_var.set("1")
        app.speed_box.event_generate("<<ComboboxSelected>>")
        app.play_button.invoke()
        pump(0.13)
        assert app.clock.frame > 0
        app.play_button.invoke()
        assert not app.clock.playing
        paused = app.clock.frame
        pump(0.1)
        assert app.clock.frame == paused
        app.select_record("other", "A")
        assert app.clock.frame == 0
        path = tmp_path / "frame.json"
        app.export_current(path)
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["frame"] == 0
        assert saved["tracks"]["est"]["error_m"] == pytest.approx(0.5)
        assert saved["tracks"]["truth"]["pose"] == [1.0, 1.5, math.radians(179)]
        assert saved["presentation_camera"]["mode"] == "rerendered_from_archived_truth"

        class ProbePanel(tk.Frame):
            def set_record(self, record):
                self.record = record

            def set_frame(self, frame, visible, trails):
                self.frame = frame

            def close(self):
                self.closed = True

        probe = app.add_panel(ProbePanel)
        app.seek(2)
        assert probe.record is app.record and probe.frame == 2
    finally:
        app.close()
    assert app.closed and app.timer is None and app.scene.engine.renderer is None
    assert app.camera.engine.renderer is None
    assert probe.closed
