import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.environments import make_inverted_pendulum_environment
from embodied_learning.experiments.swingup_comparison import (
    SCENARIOS,
    Scenario,
    recovery_metrics,
    run_experiment,
    run_scenario,
)
from embodied_learning.swingup import (
    HybridSwingupController,
    SwingupParameters,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)
from embodied_learning.swingup_demo import load_replays


@pytest.fixture(scope="module")
def design():
    return design_swingup_lqr()


def test_old_model_and_design_remain_unchanged(design):
    old, new = make_inverted_pendulum_environment(), make_swingup_environment()
    try:
        a, b = old.unwrapped.model, new.unwrapped.model
        assert a.jnt_limited[1] and not b.jnt_limited[1]
        np.testing.assert_allclose(a.jnt_range[0], [-1, 1])
        np.testing.assert_allclose(b.jnt_range[0], [-2.5, 2.5])
        for name in [
            "body_mass",
            "body_inertia",
            "body_ipos",
            "dof_damping",
            "actuator_gear",
            "actuator_ctrlrange",
        ]:
            np.testing.assert_allclose(getattr(a, name), getattr(b, name))
        np.testing.assert_allclose(
            design.controller.gain, design_lqr(control_weight=1).controller.gain
        )
        old.reset(seed=0)
        old.unwrapped.set_state(np.array([0.0, 0.25]), np.zeros(2))
        assert old.step(np.zeros(1))[2]
        new.reset(seed=0)
        new.unwrapped.set_state(design.controller.reference[:2] + [0, -np.pi], np.zeros(2))
        assert not new.step(np.zeros(1))[2]
        assert new.unwrapped.dt == pytest.approx(0.04)
    finally:
        old.close()
        new.close()


@pytest.mark.parametrize("case", SCENARIOS[:5], ids=[c.key for c in SCENARIOS[:5]])
def test_requested_scenarios_recover_with_actual_dynamics(case, design):
    arrays, metadata = run_scenario(case, design)
    metrics = recovery_metrics(arrays, metadata, design.controller.reference, design.dt)
    assert metrics["recovered"] and not metrics["terminated"]
    assert metrics["truncated"]  # observation horizon, not physical failure
    assert len(arrays["states"]) == 751
    assert metrics["max_abs_cart_position_m"] < 2.4
    assert np.max(np.abs(arrays["controls"])) <= 3
    assert np.any(arrays["modes"] == "swingup")
    assert arrays["modes"][-1] == "balance"
    errors = arrays["states"][-51:] - design.controller.reference
    errors[:, 1] = wrap_angle(errors[:, 1])
    assert np.all(np.abs(errors) <= np.array([0.02, 0.01, 0.02, 0.02]))
    if case.force_n:
        assert metrics["below_horizontal_during_push"]
        assert metrics["first_below_horizontal_s"] == pytest.approx(5.32)
        assert metrics["first_old_angle_limit_crossing_s"] < 5.4
        assert metrics["opposing_motor_steps_during_push"] == 9
        assert arrays["applied_force_n"][125] == case.force_n
        assert arrays["applied_force_n"][134] == case.force_n
        assert arrays["applied_force_n"][135] == 0
        assert arrays["controls"][126] * case.force_n < 0
        assert abs(arrays["controls"][126]) == pytest.approx(3)


def test_overload_is_real_failure_not_reset_or_fake_recovery(design):
    arrays, meta = run_scenario(SCENARIOS[-1], design)
    metrics = recovery_metrics(arrays, meta, design.controller.reference, design.dt)
    assert metrics["terminated"] and not metrics["recovered"]
    assert metrics["failure_reason"] == "cart_safety_boundary"
    assert len(arrays["controls"]) < 750
    assert len(arrays["scheduled_force_n"]) == 750
    assert len(arrays["applied_force_n"]) == len(arrays["controls"])


def test_kick_wrap_and_hysteresis(design):
    env = make_swingup_environment()
    try:
        model = env.unwrapped.model
        a, b = HybridSwingupController(model, design), HybridSwingupController(model, design)
        down = design.controller.reference + [0, -np.pi, 0, 0]
        assert abs(a.action(down)[0]) > 0
        assert a.mode == "kick"
        state = design.controller.reference + [0.02, 0.1, 0.1, -0.2]
        first = a.action(state)
        shifted = state.copy()
        shifted[1] += 2 * np.pi
        np.testing.assert_allclose(first, b.action(shifted), atol=1e-7)
        assert a.mode == b.mode == "balance"
        a.action(design.controller.reference + [0, 0.4, 0, 0])
        assert a.mode == "balance"
        a.action(design.controller.reference + [0, 0.6, 0, 0])
        assert a.mode == "swingup"
        with pytest.raises(ValueError):
            a.action(np.array([0, np.nan, 0, 0]))
    finally:
        env.close()


def test_controller_does_not_read_external_force(design):
    env = make_swingup_environment()
    try:
        env.reset(seed=0)
        a = HybridSwingupController(env.unwrapped.model, design)
        b = HybridSwingupController(env.unwrapped.model, design)
        state = design.controller.reference + [0.1, -2, 0, 0.3]
        expected = a.action(state)
        env.unwrapped.data.qfrc_applied[0] = 400
        np.testing.assert_array_equal(expected, b.action(state))
        # Nominal-model data must not alias the live simulator's disturbance channel.
        assert not np.shares_memory(b.data.qfrc_applied, env.unwrapped.data.qfrc_applied)
    finally:
        env.close()


def test_force_conversion_realizes_requested_nominal_acceleration(design):
    env = make_swingup_environment()
    try:
        env.reset(seed=0)
        state = design.controller.reference + [0.02, -np.pi, 0, 0]
        policy = HybridSwingupController(env.unwrapped.model, design)
        action = policy.action(state)
        env.unwrapped.set_state(state[:2], state[2:])
        env.unwrapped.data.ctrl[:] = action
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        assert env.unwrapped.data.qacc[0] == pytest.approx(policy.last_acceleration, abs=1e-5)
    finally:
        env.close()


def test_safety_boundary_retained(design):
    env = make_swingup_environment()
    try:
        env.reset(seed=0)
        env.unwrapped.set_state(design.controller.reference[:2] + [2.41, 0], np.zeros(2))
        _, _, terminated, _, info = env.step(np.zeros(1))
        assert terminated and info["failure_reason"] == "cart_safety_boundary"
    finally:
        env.close()


def test_settling_requires_two_seconds_and_does_not_ignore_later_excursion(design):
    a, meta = run_scenario(Scenario("short", "short", 0), design, horizon=49)
    assert not recovery_metrics(a, meta, design.controller.reference, design.dt)["recovered"]
    a, meta = run_scenario(Scenario("short", "short", 0), design, horizon=50)
    assert recovery_metrics(a, meta, design.controller.reference, design.dt)["settled_at_s"] == 0
    a["states"][-1, 2] = 0.1
    assert not recovery_metrics(a, meta, design.controller.reference, design.dt)["recovered"]


def test_archive_roundtrip_readonly_and_no_overwrite(tmp_path):
    output = tmp_path / "swingup"
    report = run_experiment(output)
    contents = (output / "trajectories.npz").read_bytes()
    replays = load_replays(output)
    assert len(replays) == 6
    assert replays[0].metadata["settled_at_s"] == pytest.approx(4.76)
    assert len(replays[-1].controls) < len(replays[0].controls)
    assert replays[-1].metadata["terminated"]
    assert (output / "trajectories.npz").read_bytes() == contents
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        run_experiment(output)
    assert (output / "trajectories.npz").read_bytes() == contents


@pytest.mark.parametrize(
    "kw", [{"energy_gain": -1}, {"velocity_gain": np.nan}, {"capture_angle": 0.6}]
)
def test_parameter_validation(kw):
    with pytest.raises(ValueError):
        SwingupParameters(**kw)


def test_invalid_schedule_rejected(design):
    with pytest.raises(ValueError):
        run_scenario(replace(SCENARIOS[0], push_duration_s=0.13), design)
    with pytest.raises(ValueError):
        run_scenario(replace(SCENARIOS[0], angle_deg=np.nan), design)
