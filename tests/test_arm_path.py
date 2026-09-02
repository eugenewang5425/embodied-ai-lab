import hashlib
import json

import numpy as np
import pytest

from embodied_learning.arm_path import (
    METHODS,
    MOVE_SECONDS,
    damped_velocity,
    generate_reference,
    jacobian,
    progress,
    segment_distance,
    singularity_probe,
    tracking_pd,
    validate_line,
)
from embodied_learning.arm_path_demo import load_replays
from embodied_learning.experiments.arm_path import audit_jacobian, run_experiment, run_path
from embodied_learning.experiments.arm_reaching import INITIAL_Q, run_reach
from embodied_learning.planar_arm import TORQUE_LIMIT, forward_kinematics, joint_positions


def test_jacobian_against_two_independent_references():
    result = audit_jacobian()
    assert result["pose_count"] == 49
    assert result["max_finite_difference_error_m_per_rad"] < 1e-8
    assert result["max_mujoco_error_m_per_rad"] < 1e-12
    np.testing.assert_allclose(jacobian([0, 0]), [[0, 0], [0.7, 0.3]], atol=1e-14)
    assert np.linalg.det(jacobian([0.3, 0.7])) == pytest.approx(0.4 * 0.3 * np.sin(0.7))


def test_damping_bounds_velocity_but_cannot_restore_missing_direction():
    records = singularity_probe()
    assert records[-2]["inverse_dq_rad_s"][1] < -60
    assert max(abs(v) for v in records[-2]["dls_dq_rad_s"]) < 0.02
    assert records[-1]["inverse_dq_rad_s"] is None
    assert records[-1]["velocity_residual_m_s"] == pytest.approx(0.02)
    np.testing.assert_allclose(damped_velocity([0, 0], [0.02, 0])[0], [0, 0])
    # The lost direction rotates with q1; being straight does not prevent all motion.
    dq, _ = damped_velocity([np.pi / 2, 0], [0.02, 0])
    predicted = jacobian([np.pi / 2, 0]) @ dq
    assert predicted[0] > 0.0199
    assert abs(predicted[1]) < 1e-14


@pytest.mark.parametrize(
    "kwargs", [{"damping": 0}, {"damping": np.nan}, {"speed_limit": -1}, {"speed_limit": np.inf}]
)
def test_invalid_damping_or_speed_rejected(kwargs):
    with pytest.raises(ValueError):
        damped_velocity([0, 1], [0.02, 0], **kwargs)


def test_reference_speed_limit_is_uniform_scaling():
    q, v = [0.2, 1.1], [1.0, -0.5]
    unlimited, _ = damped_velocity(q, v, speed_limit=100)
    limited, scale = damped_velocity(q, v, speed_limit=0.1)
    assert scale < 1
    np.testing.assert_allclose(limited, unlimited * scale)
    assert np.max(np.abs(limited)) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "time, expected", [(-1, (0, 0)), (0, (0, 0)), (4, (0.5, 1.875 / 8)), (8, (1, 0)), (11, (1, 0))]
)
def test_quintic_schedule(time, expected):
    np.testing.assert_allclose(progress(time), expected)


def test_line_reachability_includes_inner_hole_and_segment_endpoints():
    validate_line([0.35, 0.3], [0.35, -0.3])
    with pytest.raises(ValueError):
        validate_line([0.35, 0], [-0.35, 0])
    with pytest.raises(ValueError):
        validate_line([0.35, 0], [0.8, 0])
    np.testing.assert_allclose(
        segment_distance([[0.5, 0.2], [2, 0], [-1, 0]], [0, 0], [1, 0]), [0.2, 1, 1]
    )
    assert segment_distance([1, 0], [0, 0], [0, 0]) == pytest.approx(1)


@pytest.mark.parametrize("method,label", METHODS)
def test_reference_and_controller_alignment(method, label):
    initial = INITIAL_Q.copy()
    reference = generate_reference(method, initial)
    np.testing.assert_array_equal(initial, INITIAL_Q)
    assert reference["q_reference"].shape == (551, 2)
    assert reference["dq_reference"].shape == (550, 2)
    assert reference["reference_speed_scale"].shape == (550,)
    assert reference["desired_points"].shape == (551, 2)
    np.testing.assert_allclose(reference["desired_points"][0], forward_kinematics(initial))
    np.testing.assert_allclose(reference["desired_velocities"][[0, 400, 550]], 0, atol=1e-14)
    assert np.max(np.abs(tracking_pd([0, 0], [0, 0], [1, 2], [3, 4]))) <= TORQUE_LIMIT
    if method == "jacobian_path":
        np.testing.assert_allclose(
            np.diff(reference["q_reference"], axis=0), reference["dq_reference"] * 0.02, atol=1e-14
        )


def test_real_dynamics_improves_path_without_changing_old_pd():
    results = {method: run_path(method) for method, _ in METHODS}
    for arrays, case in results.values():
        assert case["completed"] and case["endpoint_success"] and not case["failure_reason"]
        assert arrays["states"].shape == (551, 4)
        assert arrays["torques_nm"].shape == (550, 2)
        assert np.max(np.abs(arrays["torques_nm"])) <= TORQUE_LIMIT
        for state, points in zip(arrays["states"], arrays["points"], strict=True):
            np.testing.assert_allclose(points, joint_positions(state[:2]), atol=1e-12)
    old_arrays, _ = run_reach([0.35, 0.3])
    np.testing.assert_allclose(
        results["endpoint_pd"][0]["states"][:301], old_arrays["states"], atol=1e-14
    )
    linear = results["joint_interpolation"][1]
    jac = results["jacobian_path"][1]
    assert linear["max_cross_track_mm"] > 30
    assert not linear["path_success"]
    assert jac["path_success"] and jac["max_cross_track_mm"] < 1
    assert jac["rms_timed_tracking_mm"] < 2
    assert jac["torque_saturated_steps"] == 0
    # The real arm is not teleported onto the geometric reference.
    actual = results["jacobian_path"][0]
    assert np.max(np.abs(actual["states"][:, :2] - actual["q_reference"])) > 1e-4
    assert jac["settled_after_movement_at_s"] >= MOVE_SECONDS


def test_path_replay_roundtrip_and_read_only_results(tmp_path):
    output = tmp_path / "path"
    report = run_experiment(output)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    replays = load_replays(output)
    for replay, case in zip(replays, report["cases"], strict=True):
        assert replay.metadata == case
        assert len(replay.states) == len(replay.torques) + 1
        assert not replay.states.flags.writeable
    with pytest.raises(FileExistsError):
        run_experiment(output)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    # Reject corrupted metadata; do not replay it against mismatched timestamps.
    report["cases"][0]["duration_s"] += 1
    (output / "summary.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="duration"):
        load_replays(output)
