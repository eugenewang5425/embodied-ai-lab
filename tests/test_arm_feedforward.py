import hashlib
import json

import mujoco
import numpy as np
import pytest

from embodied_learning.arm_dynamics import (
    audit_inverse_dynamics,
    feedforward_reference,
    inverse_torque,
    reference_acceleration,
)
from embodied_learning.arm_path import generate_reference
from embodied_learning.arm_path_demo import PathDemo, load_replays
from embodied_learning.experiments.arm_feedforward import run_experiment, verify_pd_baseline
from embodied_learning.experiments.arm_ik_comparison import run_comparison
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import run_batch
from embodied_learning.experiments.arm_timing import run_experiment as run_timing
from embodied_learning.planar_arm import KD, KP, ArmSimulation, angle_error, joint_positions


def test_inverse_dynamics_matches_equation_and_forward_acceleration():
    report = audit_inverse_dynamics()
    assert report["states"] == 49
    assert report["max_force_identity_error_nm"] < 1e-12
    assert report["max_forward_acceleration_error_rad_s2"] < 1e-12


def test_horizontal_static_gravity_and_passive_compensation():
    sim = ArmSimulation()
    scratch = mujoco.MjData(sim.model)
    np.testing.assert_allclose(
        inverse_torque(sim.model, scratch, [0.3, 0.9], [0, 0], [0, 0]), 0, atol=1e-12
    )
    torque = inverse_torque(sim.model, scratch, [0.3, 0.9], [0.2, -0.3], [0, 0])
    np.testing.assert_allclose(
        scratch.qfrc_passive, -sim.model.dof_damping * [0.2, -0.3], atol=1e-12
    )
    np.testing.assert_allclose(torque, scratch.qfrc_bias - scratch.qfrc_passive, atol=1e-12)


def test_forward_velocity_difference_has_explicit_zero_terminal_velocity():
    dq = np.array([[0.1, 0.2], [0.3, 0.4], [0, 0]])
    np.testing.assert_allclose(reference_acceleration(dq, 0.02), [[10, 10], [-15, -20], [0, 0]])
    np.testing.assert_array_equal(dq, [[0.1, 0.2], [0.3, 0.4], [0, 0]])


@pytest.mark.parametrize(
    "dq,dt", [([], 0.02), ([[0]], 0.02), ([[np.nan, 0]], 0.02), ([[0, 0]], 0), ([[0, 0]], np.inf)]
)
def test_invalid_acceleration_plan(dq, dt):
    with pytest.raises(ValueError):
        reference_acceleration(dq, dt)


def test_planning_is_offline_and_never_sets_real_simulated_pose_or_force():
    sim = ArmSimulation()
    sim.reset([0.1, 0.2], [0.01, 0.02])
    before = {
        name: getattr(sim.data, name).copy()
        for name in ("qpos", "qvel", "qacc", "ctrl", "qfrc_applied", "xfrc_applied")
    }
    ref = generate_reference("waypoint_ik", [0, 0], [0.5, 0], move_seconds=4)
    saved = {k: v.copy() for k, v in ref.items()}
    ff, ddq = feedforward_reference(sim.model, ref, sim.dt)
    assert ff.shape == ddq.shape == (350, 2)
    assert sim.data.time == 0
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(sim.data, name), value)
    for name, value in saved.items():
        np.testing.assert_array_equal(ref[name], value)
    np.testing.assert_allclose(ff[200:], 0, atol=1e-12)


def test_feedforward_rollout_keeps_feedback_and_motor_bounds(monkeypatch):
    original = ArmSimulation.step

    def observed_step(self, command):
        assert np.max(np.abs(command)) <= 0.25
        assert np.all(self.data.qfrc_applied == 0) and np.all(self.data.xfrc_applied == 0)
        return original(self, command)

    monkeypatch.setattr(ArmSimulation, "step", observed_step)
    arrays, case = run_path(
        "waypoint_ik",
        initial_q=[0, 0],
        target=[0.5, 0],
        move_seconds=4,
        controller="feedforward_pd",
    )
    assert case["completed"] and case["path_success"] and case["endpoint_success"]
    assert 0.7 < case["max_cross_track_mm"] < 1
    assert case["torque_saturated_steps"] == 1
    expected_feedback = KP * angle_error(
        arrays["q_reference"][:-1], arrays["states"][:-1, :2]
    ) + KD * (arrays["dq_reference"] - arrays["states"][:-1, 2:])
    np.testing.assert_allclose(arrays["feedback_torques_nm"], expected_feedback, atol=1e-12)
    np.testing.assert_array_equal(
        arrays["requested_torques_nm"],
        arrays["feedforward_torques_nm"] + arrays["feedback_torques_nm"],
    )
    np.testing.assert_array_equal(
        arrays["torques_nm"], np.clip(arrays["requested_torques_nm"], -0.25, 0.25)
    )
    assert np.max(np.abs(arrays["feedback_torques_nm"])) > 0.01
    assert np.max(np.abs(arrays["q_reference"] - arrays["states"][:, :2])) > 0.001
    for state, points in zip(arrays["states"], arrays["points"], strict=True):
        np.testing.assert_allclose(points, joint_positions(state[:2]), atol=1e-12)


def test_default_pd_does_not_call_inverse_dynamics(monkeypatch):
    def unwanted(*args):
        pytest.fail("PD-only baseline must not add model feedforward")

    monkeypatch.setattr("embodied_learning.experiments.arm_path.feedforward_reference", unwanted)
    arrays, _ = run_path("waypoint_ik", initial_q=[0, 0], target=[0.5, 0], move_seconds=4)
    assert "feedforward_torques_nm" not in arrays


@pytest.mark.parametrize(
    "method,controller", [("waypoint_ik", "unknown"), ("endpoint_pd", "feedforward_pd")]
)
def test_controller_scope_is_explicit(method, controller):
    with pytest.raises(ValueError):
        run_path(method, controller=controller)


def hashes(directory):
    return {
        p.relative_to(directory): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file()
    }


@pytest.fixture(scope="module")
def ff_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("feedforward")
    ten, eleven, twelve, output = [root / name for name in ("ten", "eleven", "twelve", "thirteen")]
    run_batch(ten, per_group=1)
    run_comparison(ten, eleven)
    run_timing(eleven, twelve)
    before = hashes(twelve)
    report = run_experiment(twelve, output)
    assert hashes(twelve) == before
    return twelve, output, report


def test_paired_replay_components_and_old_results_remain_readable(ff_results):
    source, output, report = ff_results
    assert (source / "source_manifest.json").read_bytes() == (
        output / "source_manifest.json"
    ).read_bytes()
    assert report["groups"]["all"]["feedforward_pd"]["path_successes"] == 3
    replays = load_replays(output / "trials/singular_inward")
    assert len(replays) == 2
    pd, ff = replays
    for field in ("desired", "q_reference", "dq_reference"):
        np.testing.assert_array_equal(getattr(pd, field), getattr(ff, field))
    assert all(
        not v.flags.writeable
        for r in replays
        for v in (r.feedforward_torques, r.feedback_torques, r.requested_torques)
    )
    assert "偏离路径" in PathDemo.replay_result_text(pd)
    assert "路径与停稳通过" in PathDemo.replay_result_text(ff)
    assert len(load_replays(source / "trials/singular_inward")) == 2


def test_output_no_overwrite_and_pd_drift(ff_results):
    source, output, _ = ff_results
    before = hashes(output)
    with pytest.raises(FileExistsError):
        run_experiment(source, output)
    assert before == hashes(output)
    with np.load(source / "trials/singular_inward/trajectories.npz") as data:
        array = data["move_4s_states"].copy()
    array[2, 0] += 0.001
    with pytest.raises(ValueError, match="baseline changed"):
        verify_pd_baseline(source / "trials/singular_inward", {"states": array})


@pytest.mark.parametrize("damage", ["kp", "hash", "limit"])
def test_changed_source_stops_before_writing(ff_results, tmp_path, damage):
    source, _, _ = ff_results
    raw = (source / "source_manifest.json").read_bytes()
    report = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if damage == "kp":
        report["kp"][0] += 1
    elif damage == "hash":
        report["source_manifest_sha256"] = "incorrect"
    else:
        report["torque_limit_nm"] = 5
    (tmp_path / "source_manifest.json").write_bytes(raw)
    (tmp_path / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        run_experiment(tmp_path, output)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["sum", "baseline_ff", "nonfinite", "shape"])
def test_inconsistent_torque_decomposition_is_rejected(ff_results, tmp_path, damage):
    _, output, _ = ff_results
    source = output / "trials/singular_inward"
    (tmp_path / "summary.json").write_bytes((source / "summary.json").read_bytes())
    with np.load(source / "trajectories.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}
    if damage == "sum":
        arrays["feedforward_pd_feedforward_torques_nm"][0, 0] += 1
    elif damage == "baseline_ff":
        arrays["pd_feedforward_torques_nm"][0, 0] += 0.01
        arrays["pd_feedback_torques_nm"][0, 0] -= 0.01
    elif damage == "nonfinite":
        arrays["pd_feedback_torques_nm"][0, 0] = np.nan
    else:
        arrays["pd_feedforward_torques_nm"] = np.zeros((1, 2))
    np.savez_compressed(tmp_path / "trajectories.npz", **arrays)
    with pytest.raises(ValueError):
        load_replays(tmp_path)


@pytest.mark.isolated_tk
def test_tk_force_components_and_terminal_frame(ff_results):
    import tkinter as tk

    _, output, _ = ff_results
    root = tk.Tk()
    root.withdraw()
    try:
        demo = PathDemo(root, load_replays(output / "trials/singular_inward"))
        assert demo.replay.metadata["key"] == "feedforward_pd"
        assert demo.clock.paused and demo.clock.speed == 0.25
        assert "第十三课" in root.title()
        assert "模型前馈" in demo.stats.get() and "PD修正" in demo.stats.get()
        assert "正在截断" in demo.stats.get()
        demo.seek(350)
        assert "已结束，无下一步力矩" in demo.stats.get()
        demo.case.set("原 PD")
        demo.change_case()
        assert demo.clock.index == 0 and demo.clock.paused
        assert "模型前馈 +0.000，+0.000" in demo.stats.get()
    finally:
        for callback in root.tk.call("after", "info"):
            root.after_cancel(callback)
        root.destroy()
