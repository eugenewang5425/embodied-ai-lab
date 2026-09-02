from __future__ import annotations

import mujoco
import numpy as np
import pytest

from embodied_learning.controllers import PDController
from embodied_learning.controllers.lqr import design_lqr, linearize, transition, upright_reference
from embodied_learning.environments import make_inverted_pendulum_environment
from embodied_learning.experiments.lqr_comparison import episode_metrics, run_comparison
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode, summarize


@pytest.fixture(scope="module")
def design():
    return design_lqr()


def test_reference_is_equilibrium_and_input_is_geared():
    env = make_inverted_pendulum_environment()
    try:
        model = env.unwrapped.model
        ref = upright_reference(model)
        assert ref[1] == pytest.approx(-np.arctan2(0.001, 0.6))
        np.testing.assert_allclose(transition(model, ref, 0.0, 2), ref, atol=1e-12)
        data = mujoco.MjData(model)
        data.qpos[:] = ref[:2]
        data.ctrl[0] = 1.0
        mujoco.mj_forward(model, data)
        assert data.qfrc_actuator[0] == pytest.approx(100.0)
    finally:
        env.close()


def test_linearization_matches_full_gym_step_and_is_repeatable(design):
    env = make_inverted_pendulum_environment()
    try:
        base = env.unwrapped
        state = design.controller.reference + np.array([1e-5, -2e-5, 1e-5, 2e-5])
        command = 1e-5
        env.reset(seed=0)
        base.set_state(state[:2], state[2:])
        observation, *_ = env.step(np.array([command]))
        exact = transition(base.model, state, command, base.frame_skip)
        np.testing.assert_allclose(exact, observation, atol=1e-12)
        predicted = design.controller.reference + design.a @ (state - design.controller.reference)
        predicted += design.b[:, 0] * command
        np.testing.assert_allclose(predicted, exact, atol=1e-9)
        other_a, other_b = linearize(base.model, design.controller.reference, base.frame_skip, 1e-5)
        np.testing.assert_allclose(design.a, other_a, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(design.b, other_b, rtol=1e-6, atol=1e-8)
    finally:
        env.close()


def test_riccati_residual_controllability_and_closed_loop_stability(design):
    a, b, p, q, r = design.a, design.b, design.riccati, design.q, design.r
    residual = a.T @ p @ a - p - a.T @ p @ b @ np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a) + q
    assert np.linalg.norm(residual) < 1e-7
    controllability = np.hstack([np.linalg.matrix_power(a, i) @ b for i in range(4)])
    assert np.linalg.matrix_rank(controllability) == 4
    assert max(abs(np.linalg.eigvals(a - b @ design.controller.gain))) < 1
    assert design.dt == pytest.approx(0.04)


def test_lqr_action_sign_shape_clipping_and_reference(design):
    controller = design.controller
    np.testing.assert_array_equal(controller.action(controller.reference), np.zeros(1))
    state = controller.reference + np.array([100, 0, 0, 0])
    assert controller.action(state).dtype == np.float32
    assert controller.action(state).shape == (1,)
    assert controller.action(state)[0] == 3
    assert controller.action(2 * controller.reference - state)[0] == -3


@pytest.mark.parametrize("seed", [0, 7, 19, 20, 39])
def test_lqr_survives_long_horizon_and_settles(design, seed):
    trace = run_episode("lqr", seed, 1000, design.controller)
    metrics = episode_metrics(trace, design.controller.reference, 1000)
    assert metrics["survived_horizon"] and metrics["settled_at_end"]
    assert abs(metrics["final_state"][0]) < 1e-4
    assert not trace.terminated and trace.truncated


def test_displaced_cart_recenters_while_legacy_pd_drifts(design):
    initial = design.controller.reference + np.array([0.2, 0.05, 0, 0])
    trace = run_episode("lqr", 0, 1000, design.controller, initial_state=initial)
    assert episode_metrics(trace, design.controller.reference, 1000)["settled_at_end"]
    pd_trace = run_episode("pd", 0, 1000, PDController(40, 1), initial_state=initial)
    assert pd_trace.terminated and pd_trace.length < 1000
    replay = run_episode("lqr", 0, 1000, design.controller, initial_state=initial)
    np.testing.assert_array_equal(trace.observations, replay.observations)


def test_failure_on_last_step_is_not_success():
    failed = EpisodeTrace(0, [np.zeros(4)], [0.0], [0.0], True, [], truncated=True)
    passed = EpisodeTrace(0, [np.zeros(4)], [0.0], [1.0], False, [], truncated=True)
    assert summarize([failed], 1)["successful_episodes"] == 0
    assert summarize([passed], 1)["successful_episodes"] == 1
    assert not episode_metrics(failed, np.zeros(4), 1)["settled_at_end"]


def test_force_metrics_have_physical_units():
    trace = EpisodeTrace(0, [np.zeros(4)], [1.0], [1.0], False, [], actuator_gear=100)
    metrics = summarize([trace], 1)
    assert metrics["root_mean_square_control"] == 1
    assert metrics["root_mean_square_force_n"] == 100


@pytest.mark.parametrize("weight", [0, -1, np.nan, np.inf])
def test_invalid_control_cost_is_rejected(weight):
    with pytest.raises(ValueError):
        design_lqr(control_weight=weight)


def test_invalid_inputs_are_rejected_without_overwriting_results(design, tmp_path):
    with pytest.raises(ValueError):
        design.controller.action(np.array([0, 0, np.nan, 0]))
    with pytest.raises(ValueError):
        design_lqr(state_weights=(1, 1, -1, 1))
    with pytest.raises(ValueError):
        run_episode("lqr", 0, 0, design.controller)
    with pytest.raises(ValueError):
        run_episode("lqr", 0, 5)
    with pytest.raises(ValueError):
        run_episode("random", 0, 5, initial_state=np.zeros(3))
    with pytest.raises(FileExistsError):
        run_comparison(tmp_path)
