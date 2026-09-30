"""Geometric guards, physical negative controls and synchronized record presentation."""

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_learning.experiments.so101_grasping import (
    calibration_audit,
    collect,
    consecutive_duration,
    run_episode,
    score_episode,
)
from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.so101 import HOME, TABLE_Z, SceneConfig, SO101Simulation, verify_assets


def test_pinned_assets_license_and_joint_layout():
    manifest = verify_assets()
    assert manifest["license"] == "Apache-2.0"
    assert len(manifest["files"]) == 22
    sim = SO101Simulation()
    assert (sim.model.nq, sim.model.nv, sim.model.nu) == (13, 12, 6)
    assert sim.model.neq == 0  # No weld to carry the object.
    np.testing.assert_allclose(sim.model.actuator_forcerange[5], [-0.3, 0.3])


def test_jacobian_and_camera_audit_exposes_unreachable_pose():
    audit = calibration_audit()
    assert audit["max_jacobian_error_m_per_rad"] < 1e-7
    assert audit["max_camera_roundtrip_error_m"] < 1e-12
    rows = {v["key"]: v for v in audit["ik_cases"]}
    assert rows["table_pick"]["reachable"] and rows["tilted_lift"]["reachable"]
    assert not rows["vertical_lift"]["reachable"]
    assert not rows["outside_workspace"]["reachable"]


def test_bad_scene_and_control_are_rejected():
    with pytest.raises(ValueError):
        SceneConfig(object_mass=-1)
    sim = SO101Simulation()
    with pytest.raises(ValueError):
        sim.step(np.r_[HOME[:5], np.nan])
    with pytest.raises(ValueError):
        sim.step(np.r_[HOME[:5], 3.0])
    with pytest.raises(ValueError):
        sim.ik([0.2, 0, 0.8], np.zeros((3, 3)))
    with pytest.raises(ValueError):
        sim.reset(np.ones(6) * 9)


def test_duration_requires_elapsed_time_not_sample_count():
    assert consecutive_duration([True]) == 0
    assert consecutive_duration([True] * 51) == 1.0
    assert consecutive_duration([True] * 25 + [False] + [True] * 26) == 0.5


@pytest.fixture(scope="module")
def physical_cases():
    return {method: run_episode(method=method) for method in ("clamp", "open", "offset")}


def test_real_pick_place_and_paired_negative_controls(physical_cases):
    normal, values = physical_cases["clamp"]
    assert normal["success"] and normal["stable_lift"] and normal["stable_placement"]
    assert normal["lift_hold_s"] >= 1.0
    assert 0.1 < normal["max_lift_m"] < 0.15
    assert normal["final_xy_error_m"] < 0.025
    for method in ("open", "offset"):
        row, _ = physical_cases[method]
        assert not row["stable_lift"] and not row["success"]
    assert len(values["physics_time"]) == (len(values["timestamps"]) - 1) * 10
    assert np.all(values["physics_qpos"][:, 5] >= -0.174533 - 1e-3)


def test_lift_without_bilateral_contact_cannot_pass(physical_cases):
    _, values = physical_cases["clamp"]
    modified = {k: v.copy() for k, v in values.items()}
    modified["jaw_normal_n"][:, 1] = 0
    score = score_episode(modified, SceneConfig())
    assert not score["stable_lift"] and not score["success"]


def test_initial_settling_height_is_not_counted_as_lift(physical_cases):
    _, values = physical_cases["open"]
    before = score_episode(values, SceneConfig())
    modified = {k: v.copy() for k, v in values.items()}
    modified["object_xyz"][0, 2] += 0.5
    after = score_episode(modified, SceneConfig())
    assert before["max_lift_m"] == after["max_lift_m"]
    assert not after["success"]


@pytest.mark.slow
def test_camera_projection_agrees_with_rendered_object_pixels():
    import mujoco

    from embodied_learning.so101 import project_point

    sim = SO101Simulation()
    q = HOME.copy()
    q[0] = 0.8  # Move the arm away from the cube, so the camera can see it.
    sim.reset(q)
    renderer = mujoco.Renderer(sim.model, 480, 640)
    try:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(sim.data, camera="overhead")
        segmentation = renderer.render().copy()
        rows, columns = np.nonzero(segmentation[..., 0] == sim.object_geom)
        assert len(rows) > 100
        uv, _ = project_point(sim, "overhead", sim.data.xpos[sim.object_body])
        # Rendered box silhouette vs center projection; allow perspective/subpixel effects.
        assert np.linalg.norm(uv - [columns.mean(), rows.mean()]) < 3.0
    finally:
        renderer.close()


def test_known_failure_does_not_become_success_from_servo_motion():
    row, values = run_episode(SceneConfig(object_xyz=(0.22, 0.055, TABLE_Z + 0.025)))
    assert not row["success"] and row["max_lift_m"] < 0.01
    assert np.ptp(values["tcp_xyz"][:, 2]) > 0.1  # Hand moved, object did not follow.


@pytest.fixture(scope="module")
def saved_record(tmp_path_factory):
    output = tmp_path_factory.mktemp("so101") / "record"
    collect(output, smoke=True)
    return output


@pytest.mark.slow
def test_record_integrity_immutability_and_overwrite_guard(saved_record, tmp_path):
    record = ManipulationRecord(saved_record)
    row, values = record.select("p00", "clamp")
    assert row["success"] and not values["qpos"].flags.writeable
    with pytest.raises(FileExistsError):
        collect(saved_record, smoke=True)
    import shutil

    damaged = tmp_path / "damaged"
    shutil.copytree(saved_record, damaged)
    summary = json.loads((damaged / "summary.json").read_text(encoding="utf-8"))
    summary["model_manifest_sha256"] = "0" * 64
    (damaged / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="different SO101"):
        ManipulationRecord(damaged)
    (damaged / "summary.json").write_bytes((saved_record / "summary.json").read_bytes())
    (damaged / "trajectories.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        ManipulationRecord(damaged)


@pytest.mark.isolated_tk
def test_window_switches_all_views_preserves_time_and_exports():
    import tkinter as tk

    from embodied_learning.manipulation_viewer.window import ManipulationWindow

    directory = Path("results/so101_grasping_v2")
    if not (directory / "summary.json").exists():
        pytest.skip("generate SO101 formal records first")
    record = ManipulationRecord(directory)
    root = tk.Tk()
    app = ManipulationWindow(root, record, "p04")
    try:
        root.update()
        app.seek(525)
        for method in ("open", "offset", "clamp"):
            app.method_buttons[method].invoke()
            root.update()
            assert app.clock.frame == 525
            assert app.scene.method == app.overhead.method == app.wrist.method == method
            assert app.diagnostics.method == method
            assert app.scene.frame == app.overhead.frame == app.wrist.frame == app.diagnostics.frame
            assert np.std(np.asarray(app.wrist.image)) > 15
        app.position = "p02"
        app.load_selection()
        root.update()
        assert app.clock.frame == 525 and not app.row["success"]
        app.seek(0)
        assert app.clock.frame == 0
        assert "已知物体位置" in app.method_note.cget("text")
        output = directory / "_window_check_export.json"
        app.export_current(output)
        assert json.loads(output.read_text(encoding="utf-8"))["episode"]["key"] == "p02_clamp"
    finally:
        app.close()
