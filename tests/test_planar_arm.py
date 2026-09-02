import hashlib

import numpy as np
import pytest

from embodied_learning.arm_demo import load_replays
from embodied_learning.experiments.arm_reaching import (
    CASES,
    audit_geometry,
    run_experiment,
    run_reach,
)
from embodied_learning.planar_arm import (
    TORQUE_LIMIT,
    ArmSimulation,
    angle_error,
    forward_kinematics,
    inverse_kinematics,
    joint_pd,
    joint_positions,
)


@pytest.mark.parametrize(
    "degrees, expected",
    [([0, 0], [0.7, 0]), ([90, 0], [0, 0.7]), ([0, 90], [0.4, 0.3]), ([0, 180], [0.1, 0])],
)
def test_known_forward_geometry(degrees, expected):
    q = np.deg2rad(degrees)
    np.testing.assert_allclose(forward_kinematics(q), expected, atol=1e-14)
    sim = ArmSimulation()
    sim.reset(q)
    np.testing.assert_allclose(sim.points()[-1], expected, atol=1e-14)


def test_elbow_angle_is_relative_and_does_not_move_elbow():
    a, b = joint_positions(np.deg2rad([45, 0])), joint_positions(np.deg2rad([45, 90]))
    np.testing.assert_allclose(a[1], b[1])
    np.testing.assert_allclose(b[2] - b[1], 0.3 * np.array([-1, 1]) / np.sqrt(2))
    assert audit_geometry()["max_point_error_m"] < 1e-12


@pytest.mark.parametrize(
    "target", [[0.35, 0.3], [0.35, -0.3], [-0.35, 0.3], [-0.35, -0.3], [0.7, 0], [0.1, 0]]
)
def test_inverse_branches_return_the_same_target(target):
    solutions = inverse_kinematics(target)
    assert solutions.shape == (2, 2)
    for q in solutions:
        np.testing.assert_allclose(forward_kinematics(q), target, atol=1e-12)
    if 0.1 < np.linalg.norm(target) < 0.7:
        assert solutions[0, 1] > 0 > solutions[1, 1]


@pytest.mark.parametrize("target", [[0.8, 0], [0, 0], [0.05, 0], [np.nan, 0.3], [1, 2, 3]])
def test_unreachable_or_invalid_target_is_rejected(target):
    with pytest.raises(ValueError):
        inverse_kinematics(target)


def test_pd_units_saturation_and_angular_equivalence():
    action = joint_pd([0, 0], [0, 0], [0.1, -0.2])
    np.testing.assert_allclose(action, [TORQUE_LIMIT, -TORQUE_LIMIT])
    np.testing.assert_allclose(action, joint_pd([2 * np.pi, -2 * np.pi], [0, 0], [0.1, -0.2]))
    sim = ArmSimulation()
    state, command, reason = sim.step([999, -999])
    assert not reason and np.isfinite(state).all()
    np.testing.assert_allclose(command, [TORQUE_LIMIT, -TORQUE_LIMIT])
    np.testing.assert_allclose(sim.data.qfrc_actuator, command)
    assert sim.dt == pytest.approx(0.02)
    assert sim.data.time == pytest.approx(0.02)
    # Site coordinates must correspond to final qpos, not stale pre-integration data.
    np.testing.assert_allclose(sim.points(), joint_positions(state[:2]), atol=1e-12)


@pytest.mark.parametrize("key,label,target,branch", CASES, ids=[c[0] for c in CASES])
def test_dynamics_reaching_success_requires_position_and_velocity(key, label, target, branch):
    arrays, report = run_reach(target, branch)
    assert report["success"] and not report["failure_reason"]
    assert arrays["states"].shape == (301, 4)
    assert arrays["points"].shape == (301, 3, 2)
    assert arrays["torques_nm"].shape == (300, 2)
    assert np.max(np.abs(arrays["torques_nm"])) <= TORQUE_LIMIT
    assert report["max_dynamic_fk_error_m"] < 1e-12
    assert report["settled_at_s"] <= 5.5
    assert np.max(np.linalg.norm(arrays["points"][-26:, -1] - target, axis=1)) <= 0.002
    assert np.max(np.abs(arrays["states"][-26:, 2:])) <= 0.02
    assert np.max(np.abs(angle_error(report["goal_q_rad"], arrays["states"][-26:, :2]))) <= 0.01


def test_short_episode_is_not_claimed_as_settled():
    arrays, report = run_reach([0.35, 0.3], steps=10)
    assert not report["success"] and report["settled_at_s"] is None
    assert len(arrays["states"]) == 11


def test_archive_roundtrip_no_overwrite_and_no_fake_endpoint(tmp_path):
    output = tmp_path / "arm"
    report = run_experiment(output)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    replays = load_replays(output)
    assert len(replays) == 3
    for replay, case in zip(replays, report["cases"], strict=True):
        assert replay.metadata == case
        assert len(replay.states) == len(replay.torques) + 1
        for state, points in zip(replay.states, replay.points, strict=True):
            np.testing.assert_allclose(points, joint_positions(state[:2]), atol=1e-12)
    with pytest.raises(FileExistsError):
        run_experiment(output)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert before == after


def test_arm_model_does_not_change_old_cartpole_task():
    from embodied_learning.environments import make_inverted_pendulum_environment

    env = make_inverted_pendulum_environment()
    try:
        assert env.unwrapped.model.nu == 1
        np.testing.assert_allclose(env.unwrapped.model.actuator_gear[0, 0], 100)
        np.testing.assert_allclose(env.unwrapped.model.jnt_range[0], [-1, 1])
        arm = ArmSimulation()
        assert arm.model.nu == arm.model.nv == arm.model.nq == 2
    finally:
        env.close()
