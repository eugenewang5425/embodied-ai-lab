"""Meaningful surface geometry, causal contact gating and physical regressions."""

import numpy as np
import pytest

from embodied_learning.gripper_contact import BilateralContactGate, SurfaceBandCalibration
from embodied_learning.so101 import SceneConfig, SO101Simulation


def test_surface_prediction_matches_real_jaws_and_does_not_write_physics():
    sim = SO101Simulation()
    q, v = sim.data.qpos.copy(), sim.data.qvel.copy()
    calibration = SurfaceBandCalibration(sim)
    for opening in (0.017, 0.126, 0.283, 0.434):
        exact = calibration.exact_midpoint(opening)
        predicted = calibration.local_midpoint(opening)
        np.testing.assert_allclose(exact, predicted, rtol=0, atol=3e-5)
        center = np.array([0.225, 0.035, 0.713])
        target, rotation = calibration.tcp_target(center, opening, -0.3)
        np.testing.assert_allclose(target + rotation @ predicted, center, atol=1e-9)
    np.testing.assert_array_equal(q, sim.data.qpos)
    np.testing.assert_array_equal(v, sim.data.qvel)
    np.testing.assert_array_equal(calibration.local_midpoint(1.1), calibration.local_midpoint(0.45))


def test_collision_ray_interval_has_known_box_answer():
    sim = SO101Simulation()
    calibration = SurfaceBandCalibration(sim)
    g = sim.model.geom("table").id
    center = calibration.data.geom_xpos[g]
    interval = calibration._interval(g, center, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(interval, (-0.025, 0.025), atol=1e-12)
    assert calibration._interval(g, center + [1, 0, 0], np.array([0.0, 0.0, 1.0])) is None
    with pytest.raises(ValueError):
        SurfaceBandCalibration(sim, band_x=np.nan)
    with pytest.raises(ValueError):
        calibration.local_midpoint(np.nan)


def test_gate_requires_continuous_solved_bilateral_intervals():
    gate = BilateralContactGate()
    both = np.ones((10, 2)) * 0.1
    for _ in range(9):
        assert not gate.update(both, 0.02)
    assert gate.update(both, 0.02)
    one = both.copy()
    one[5, 0] = 0.01  # One lost physical substep invalidates the whole interval.
    assert not gate.update(one, 0.02) and gate.elapsed == 0
    for value in (np.full((10, 2), np.nan), np.empty((0, 2)), np.zeros((10, 2))):
        assert not gate.update(value, 0.02)
    with pytest.raises(ValueError):
        gate.update(both, 0)
    with pytest.raises(ValueError):
        BilateralContactGate(duration_s=np.nan)


@pytest.mark.slow
def test_fixed_old_regression_rescued_and_gate_refusal_is_failure():
    from embodied_learning.experiments.so101_contact_study import run_episode

    config = SceneConfig(object_xyz=(0.205, -0.025, 0.715))
    base, a = run_episode(config, "baseline", 15)
    better, b = run_episode(config, "surface_gate", 15)
    stopped, c = run_episode(config, "gate", 15)
    assert not base["success"] and better["success"]
    assert not stopped["success"] and stopped["contact_refused"]
    assert not np.any(c["phase"] >= 4)
    assert b["gate_elapsed_s"][np.flatnonzero(b["phase"] == 3)[-1]] >= 0.2
    np.testing.assert_array_equal(a["commands"][: len(c["commands"])], c["commands"])
    assert len(b["physics_time"]) == (len(b["timestamps"]) - 1) * 10


@pytest.mark.isolated_tk
def test_round3_one_click_switch_and_short_refusal_clock():
    import tkinter as tk
    from pathlib import Path

    from embodied_learning.manipulation_record import ManipulationRecord
    from embodied_learning.manipulation_viewer.window import ManipulationWindow

    path = Path("results/so101_contact_v4")
    if not (path / "summary.json").exists():
        pytest.skip("collect the frozen round-three batch first")
    record = ManipulationRecord(path)
    root = tk.Tk()
    app = ManipulationWindow(root, record, "p04a1")
    try:
        root.update()
        assert app.method == "surface"
        app.seek(350)
        for method in record.available_methods(app.position):
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == 350
            for panel in (app.scene, app.wrist, app.overhead, app.diagnostics):
                assert panel.method == method and panel.frame == 350
        assert "拒绝抬升也计失败" in app.method_note.cget("text")
        assert np.std(np.asarray(app.wrist.image)) > 15
    finally:
        app.close()
