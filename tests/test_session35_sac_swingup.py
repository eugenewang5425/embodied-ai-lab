"""Lesson 35: hand-written numpy SAC on the down-start swing-up.

Unit checks pin the tanh-squashed Gaussian log-prob and mean clamping to hand
answers, the TD target to a hand-computed Bellman value (with the double-Q min
contract and the terminated/no-bootstrap mask), the soft-update step to its
closed form, the policy reparameterization gradient to finite differences
(1e-4 relative), the automatic temperature to the target-entropy equilibrium,
and the replay buffer to redraw determinism and ring eviction; the coverage
entropy reduces to known values on degenerate histograms. Further checks cover
the micro-training determinism (two runs bitwise identical, critic loss
decreasing, alpha flat in the fixed tier), the reused lesson-7 acceptance, the
checkpoint first-success/first-arrival reductions, the shrunk end-to-end
record contract, the demo loader's tamper rejections, the CLI and the real
Tk demo. No full training is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

# The isolated_tk child re-imports this module before any pyplot use: forcing
# the Agg backend keeps run_experiment's figure saving from creating (and
# destroying) a first Tcl interpreter, whose leftovers make the demo test's
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows
# (D-2026-09-05-12).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.experiments.ppo_swingup import (
    EVAL_FIELDS,
    RewardFunction,
    run_policy_episode,
)
from embodied_learning.experiments.sac_swingup import (
    ALPHA_INIT,
    BATCH_SIZE,
    EXPERIMENT,
    GAMMA,
    LOG_STD_MAX,
    LOG_STD_MIN,
    ReplayBuffer,
    SACConfig,
    SquashedGaussianPolicy,
    alpha_step,
    clone_mlp,
    coverage_entropy,
    critic_loss_and_gradients,
    critic_values_and_input_grads,
    deterministic_sac_episode,
    episode_metrics,
    expected_npz_keys,
    first_arrival_eval_steps,
    first_success_eval_steps,
    run_experiment,
    sac_policy_loss_and_gradients,
    soft_update_mlp,
    squashed_gaussian_log_prob,
    tanh_squash_params,
    td_targets,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.swingup import design_swingup_lqr

SMALL_CONFIG = {
    "config": SACConfig(
        n_envs=2,
        train_steps=800,
        buffer_size=2000,
        batch_size=64,
        update_every_env_steps=4,
        eval_every_steps=400,
        train_episode_steps=50,
        warmup_steps=64,
    ),
    "train_seeds": 1,
    "eval_seed_count": 2,
}


class _FlatQ:
    """Duck-typed policy stand-in returning hand-set actions / log-probs."""

    def __init__(self, actions, log_prob):
        self.actions = np.asarray(actions, dtype=float)
        self.log_prob = np.asarray(log_prob, dtype=float)

    def sample(self, _obs, _rng):
        return self.actions, self.log_prob


def _linear_q(weights, bias, seed=0):
    """A 6->1 linear MLP (input_dim 6, no hidden) with hand-set values."""
    from embodied_learning.experiments.sac_swingup import MLP

    net = MLP(6, (), 1, seed)
    net.weights[0] = np.asarray(weights, dtype=float).reshape(6, 1)
    net.biases[0] = np.asarray(bias, dtype=float).reshape(1)
    return net


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("sac_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    reward = RewardFunction(design.controller.reference)
    return design, reward


# ------------------------------------------------------- TD target / Bellman
def test_td_target_bellman_known_answer():
    """y = r + gamma*(1-term)*(min(Qbar1, Qbar2) - alpha*log pi) by hand."""
    # two next states, two samples: one alive, one terminated (no bootstrap)
    next_obs = np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 1.0, 0.0, 0.0]])
    rewards = np.asarray([0.7, -3.0])
    terminated = np.asarray([False, True])
    # linear targets: Qbar_i(s, a) = w_i . (s, a) + b_i
    w1 = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 0.5])
    w2 = np.asarray([2.0, 1.0, 2.0, 1.0, 2.0, -0.25])
    target_q1 = _linear_q(w1, 0.0)
    target_q2 = _linear_q(w2, 0.5)
    next_actions = np.asarray([0.24, -0.5])
    log_prob = np.asarray([-1.7, -2.9])
    policy = _FlatQ(next_actions, log_prob)
    alpha, gamma = 0.35, 0.9
    targets = td_targets(
        rewards, terminated, next_obs, target_q1, target_q2, policy, None, alpha, gamma
    )
    step0 = next_obs[0].tolist() + [next_actions[0]]
    step1 = next_obs[1].tolist() + [next_actions[1]]
    q1_0 = float(np.dot(w1, step0))
    q2_0 = float(np.dot(w2, step0) + 0.5)
    q1_1 = float(np.dot(w1, step1) + 0.0)
    q2_1 = float(np.dot(w2, step1) + 0.5)
    expected_0 = rewards[0] + gamma * (min(q1_0, q2_0) - alpha * log_prob[0])
    expected_1 = rewards[1]  # terminated: no bootstrap at all
    assert targets[0] == pytest.approx(expected_0)
    assert targets[1] == pytest.approx(expected_1)
    assert min(q1_0, q2_0) == q2_0  # the double-Q min actually picked q2 here
    assert min(q1_1, q2_1) == q1_1
    # the double-Q MIN contract: flip the sign of w2 so q2 becomes the argmin
    w2_flip = np.asarray([-2.0, -1.0, -2.0, -1.0, -2.0, -0.25])
    target_q2_b = _linear_q(w2_flip, 0.5)
    targets_b = td_targets(
        rewards, terminated, next_obs, target_q1, target_q2_b, policy, None, alpha, gamma
    )
    q2b_0 = float(np.dot(w2_flip, step0) + 0.5)
    expected_b0 = rewards[0] + gamma * (min(q1_0, q2b_0) - alpha * log_prob[0])
    assert targets_b[0] == pytest.approx(expected_b0)
    assert min(q1_0, q2b_0) == q2b_0
    # critic gradients of a linear Q against a known target are 2 (Q - y) x / B
    observations = np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]])
    actions = np.asarray([0.5])
    q_net = _linear_q(np.zeros(6), 0.0)
    loss, grads = critic_loss_and_gradients(q_net, observations, actions, np.asarray([1.0]))
    q_value = (q_net.forward(np.concatenate([observations, actions.reshape(-1, 1)], axis=1))[0])[
        0, 0
    ]
    assert loss == pytest.approx((q_value - 1.0) ** 2)
    assert grads[0][0, 0] == pytest.approx(2.0 * (q_value - 1.0) * 1.0)
    assert grads[1][0] == pytest.approx(2.0 * (q_value - 1.0))


def test_target_soft_update_step():
    """theta_target <- tau*theta + (1-tau)*theta_target, exactly one step."""
    q = _linear_q(np.arange(1.0, 7.0), 0.5)
    target = _linear_q(np.zeros(6), 0.0)
    tau = 0.005
    soft_update_mlp(target, q, tau)
    assert target.weights[0][:, 0] == pytest.approx(tau * np.arange(1.0, 7.0))
    assert target.biases[0] == pytest.approx(tau * 0.5)
    # second step compounds the interpolation
    soft_update_mlp(target, q, tau)
    assert target.weights[0][0, 0] == pytest.approx(tau + (1 - tau) * tau * 1.0)


# ------------------------------------------------------- policy / critic math
def test_policy_reparam_gradient_finite_difference():
    """The analytic SAC policy gradient matches a finite-difference check."""
    from embodied_learning.experiments.sac_swingup import MLP

    policy = SquashedGaussianPolicy(5, (64, 64), 7)
    q1 = MLP(6, (64, 64), 1, [7, 1])
    q2 = clone_mlp(q1)
    obs = np.random.default_rng(0).normal(size=(16, 5))
    noise_seed, alpha = 42, 0.35
    eps = np.random.default_rng(noise_seed).standard_normal((16, 1))
    mean_raw, log_std_raw, _ = policy.heads(obs)
    mean, log_std = tanh_squash_params(
        mean_raw, log_std_raw, policy.action_bound, policy.log_std_min, policy.log_std_max
    )
    u = mean + np.exp(log_std) * eps
    a0 = policy.action_bound * np.tanh(u)[:, 0]
    q_value, q_grad = critic_values_and_input_grads(q1, q2, obs, a0)
    metrics, grads = sac_policy_loss_and_gradients(
        policy, obs, q_value, q_grad, np.random.default_rng(noise_seed), alpha
    )
    assert np.isfinite(metrics["loss"])

    def objective(theta_weights, theta_biases):
        clone = SquashedGaussianPolicy(5, (64, 64), 7)
        clone.trunk.weights = theta_weights
        clone.trunk.biases = theta_biases
        mean_raw, log_std_raw, _ = clone.heads(obs)
        mean, log_std = tanh_squash_params(
            mean_raw, log_std_raw, clone.action_bound, clone.log_std_min, clone.log_std_max
        )
        u = mean + np.exp(log_std) * eps
        a = clone.action_bound * np.tanh(u)[:, 0]
        log_prob = squashed_gaussian_log_prob(u, mean, log_std, clone.action_bound)[:, 0]
        qv = q1.forward(np.concatenate([obs, a.reshape(-1, 1)], axis=1))[0][:, 0]
        return float(np.mean(alpha * log_prob - qv))

    theta_weights = [w.copy() for w in policy.trunk.weights]
    theta_biases = [b.copy() for b in policy.trunk.biases]
    h = 1e-6
    errors = []
    for index, grad in enumerate(grads):
        if index < 3:
            target_weights = theta_weights[index]
            target_weights[0, 0] += h
            plus = objective(theta_weights, theta_biases)
            target_weights[0, 0] -= 2 * h
            minus = objective(theta_weights, theta_biases)
            target_weights[0, 0] += h
            fd = (plus - minus) / (2 * h)
            analytic = float(grad[0, 0])
        else:
            target_bias = theta_biases[index - 3]
            target_bias[0] += h
            plus = objective(theta_weights, theta_biases)
            target_bias[0] -= 2 * h
            minus = objective(theta_weights, theta_biases)
            target_bias[0] += h
            fd = (plus - minus) / (2 * h)
            analytic = float(grad[0])
        errors.append(abs(fd - analytic) / max(1e-9, abs(analytic)))
    assert max(errors) < 1e-4


def test_tanh_squashed_log_prob_and_clamps_known_answers():
    """mu = 3*tanh(mu_raw), log_std tanh-clamped; logp includes the Jacobian."""
    policy = SquashedGaussianPolicy(5, (64, 64), 1)
    mean_raw = np.asarray([[0.0], [2.0], [-1.0]])
    log_std_raw = np.asarray([[0.0], [10.0], [-10.0]])
    mean, log_std = tanh_squash_params(
        mean_raw, log_std_raw, policy.action_bound, policy.log_std_min, policy.log_std_max
    )
    assert mean[0, 0] == pytest.approx(0.0)
    assert mean[1, 0] == pytest.approx(3.0 * np.tanh(2.0))
    assert log_std[0, 0] == pytest.approx((LOG_STD_MIN + LOG_STD_MAX) / 2.0)
    assert log_std[1, 0] == pytest.approx(LOG_STD_MAX)  # tanh saturates
    assert log_std[2, 0] == pytest.approx(LOG_STD_MIN)
    # hand computation of the squashed log-prob at a known point
    u = np.asarray([[0.3]])
    mean_p = np.asarray([[0.1]])
    log_std_p = np.asarray([[-0.5]])
    value = squashed_gaussian_log_prob(u, mean_p, log_std_p, 3.0)
    manual = (
        -0.5 * ((0.3 - 0.1) / np.exp(-0.5)) ** 2
        - (-0.5)
        - 0.5 * np.log(2.0 * np.pi)
        - np.log(1.0 - np.tanh(0.3) ** 2 + 1e-6)
        - np.log(3.0)
    )
    assert value[0, 0] == pytest.approx(manual)
    # the deterministic (squashed mean) action and the interface guard
    assert policy.mean(np.zeros((2, 5))).shape == (2,)
    assert float(policy.mean(np.zeros((1, 5)))[0]) == pytest.approx(0.0)
    action, log_prob = policy.sample(np.zeros((1, 5)), np.random.default_rng(3))
    assert action.shape == (1,) and log_prob.shape == (1,)
    assert abs(action[0]) <= 3.0 + 1e-12


def test_alpha_auto_reaches_target_entropy():
    """The automatic temperature moves toward E[log pi] = target entropy."""
    # direction: entropy below the target (log pi above -H_bar) -> alpha up;
    # entropy above the target -> alpha down (alpha is floored at alpha_min)
    assert alpha_step(0.2, np.asarray([10.0]), -1.0, 3e-4) > 0.2
    assert alpha_step(0.2, np.asarray([-10.0]), -1.0, 3e-4) < 0.2
    repeatedly = alpha_step(0.001, np.asarray([-10.0]), -1.0, 3e-4)
    assert repeatedly == pytest.approx(max(1e-8, 0.001 + 3e-4 * (-11.0)))
    # iterations with log probs far above the target grow alpha without bound
    alpha = 0.2
    for _ in range(5000):
        alpha = alpha_step(alpha, np.asarray([9.0, 9.5, 10.0]), -1.0, 3e-4)
    assert alpha > 1.0
    # iterations with log probs far below the target decay alpha to the floor
    alpha = 0.2
    for _ in range(5000):
        alpha = alpha_step(alpha, np.asarray([-9.0, -9.5, -10.0]), -1.0, 3e-4)
    assert alpha == pytest.approx(1e-8)


# ------------------------------------------------------------------ replay
def test_replay_buffer_redraw_determinism_and_eviction():
    """Same-seed sampling is bitwise identical; capacity evicts the oldest.

    The exactness segment pins the column contract that the trainer's stores
    (pre-step observation, pre-reset next observation), which is what makes the
    TD targets a correct Bellman backup.
    """
    buffer = ReplayBuffer(10, 5, 1)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(20, 5))
    for index in range(20):
        buffer.push(obs[index : index + 1], 0.5, float(index), obs[index : index + 1], False)
    assert buffer.size == 10  # the first 10 entries were evicted
    sampled = buffer.sample(4, np.random.default_rng(7))
    sampled_again = buffer.sample(4, np.random.default_rng(7))
    np.testing.assert_array_equal(sampled["observations"], sampled_again["observations"])
    np.testing.assert_array_equal(sampled["rewards"], sampled_again["rewards"])
    # exactness: every column holds exactly what was pushed (no shifting)
    exact = ReplayBuffer(8, 5, 1)
    pushes = [
        (np.full((1, 5), 1.0), np.asarray([0.1]), 0.7, np.full((1, 5), 2.0), False),
        (np.full((1, 5), 3.0), np.asarray([-0.2]), -3.0, np.full((1, 5), 4.0), True),
        (np.full((1, 5), 5.0), np.asarray([0.3]), 0.0, np.full((1, 5), 6.0), False),
    ]
    for ob, ac, reward, nxt, term in pushes:
        exact.push(ob, ac, reward, nxt, term)
    for index, (_ob, _ac, _reward, _nxt, _term) in enumerate(pushes):
        np.testing.assert_array_equal(exact.observations[index], _ob[0])
        np.testing.assert_array_equal(exact.actions[index], _ac)
        assert exact.rewards[index] == _reward
        np.testing.assert_array_equal(exact.next_obs[index], _nxt[0])
        assert bool(exact.terminated[index]) is _term
    assert exact.size == 3 and exact.cursor == 3
    assert set(np.unique(sampled["rewards"])) <= {
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
    }
    with pytest.raises(ValueError):
        buffer.sample(11, np.random.default_rng(7))
    stored = buffer.stored_observations()
    assert stored.shape == (10, 5)
    # large batches without replacement: 5 draws of 2 x 10-cell buffer
    indices = np.asarray([buffer.sample(2, np.random.default_rng(i))["rewards"] for i in range(5)])
    assert indices.shape == (5, 2)


def test_coverage_entropy_known_answers():
    """Degenerate histograms reduce to hand values; the full grid caps at log(192)."""
    cell = np.asarray([[0.0, 0.0, 1.0, 0.0, 0.0]])  # x=0, angle=+pi/2, omega=0
    entropy, fraction = coverage_entropy(cell)
    assert fraction == pytest.approx(1.0 / 192.0)
    assert entropy == pytest.approx(0.0)
    identical = np.repeat(cell, 10, axis=0)
    assert coverage_entropy(identical) == (0.0, 1.0 / 192.0)
    # one sample per cell of the angle axis (cell centers - the edges suffer
    # floating-point boundary jitter) at a fixed position/velocity
    angles = -np.pi + (np.arange(12) + 0.5) * (2.0 * np.pi / 12.0)
    grid = np.zeros((12, 5))
    grid[:, 1] = np.cos(angles)
    grid[:, 2] = np.sin(angles)
    entropy, fraction = coverage_entropy(grid)
    assert fraction == pytest.approx(12.0 / 192.0)
    assert entropy == pytest.approx(np.log(12.0))
    empty = np.empty((0, 5))
    assert coverage_entropy(empty) == (0.0, 0.0)


# ---------------------------------------------------------- training / eval
def test_micro_training_is_deterministic_and_learns():
    """Fixed seeds reproduce bitwise; critic loss falls; fixed tier alpha flat."""
    config = SACConfig(
        n_envs=2,
        train_steps=600,
        buffer_size=2000,
        batch_size=32,
        update_every_env_steps=4,
        eval_every_steps=600,
        train_episode_steps=50,
        warmup_steps=32,
    )

    def one_run(auto=True):
        from embodied_learning.experiments.ppo_swingup import VecSwingup
        from embodied_learning.experiments.sac_swingup import RewardFunction, train_sac

        design = design_swingup_lqr()
        reward = RewardFunction(design.controller.reference)
        vec = VecSwingup(
            reward,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=7,
            task_envs=config.n_envs,
        )
        try:
            return train_sac(
                vec,
                config=SACConfig(**{**config.__dict__, "auto_alpha": auto}),
                policy_seed=[0, 0, 7000],
                action_seed=[0, 0, 5000],
                buffer_seed=[0, 0, 9000],
                eval_hook=lambda policy, steps: {
                    "success": False,
                    "settled_at_s": None,
                    "return": 0.0,
                    "first_arrival_s": None,
                },
            )
        finally:
            vec.close()

    first, second = one_run(), one_run()
    np.testing.assert_array_equal(first["curves"]["reward_curve"], second["curves"]["reward_curve"])
    np.testing.assert_array_equal(
        first["curves"]["critic_loss_curve"], second["curves"]["critic_loss_curve"]
    )
    np.testing.assert_array_equal(first["curves"]["alpha_curve"], second["curves"]["alpha_curve"])
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
    assert first["curves"]["critic_loss_curve"][0] > first["curves"]["critic_loss_curve"][-1]
    assert np.isfinite(first["curves"]["reward_curve"]).all()
    # the fixed tier keeps alpha constant at its init value
    fixed = one_run(auto=False)
    np.testing.assert_allclose(fixed["curves"]["alpha_curve"], ALPHA_INIT, atol=1e-12)
    auto = one_run(auto=True)
    assert auto["curves"]["alpha_curve"][0] == pytest.approx(ALPHA_INIT, abs=1e-3)


def test_acceptance_reuses_lesson7_recovery_metrics(design_and_reward):
    """episode_metrics is the lesson-7 function on the same arrays."""
    design, task_reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    policy = SquashedGaussianPolicy(5, (64, 64), seed=2)
    arrays, reason = run_policy_episode(
        policy, task_reward, reference, horizon=50, env_seed=0, deterministic=True
    )
    through_shim = episode_metrics(arrays, reason, reference, dt)
    direct_view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    direct = recovery_metrics(direct_view, {"failure_reason": reason}, reference, dt)
    for field in EVAL_FIELDS:
        assert through_shim[field] == direct[field]
    record, det_arrays = deterministic_sac_episode(policy, task_reward, reference, dt)
    assert {
        "recovered",
        "terminated",
        "settled_at_s",
        "failure_reason",
        "return",
        "first_arrival_s",
    } == set(record)
    steps_return = [
        task_reward(st, u, bool(record["terminated"]) and i == len(det_arrays["controls"]) - 1)
        for i, (st, u) in enumerate(zip(det_arrays["states"][1:], det_arrays["controls"]))
    ]
    assert record["return"] == pytest.approx(sum(steps_return))


def test_first_success_and_arrival_metrics():
    """The checkpoint metrics reduce to known answers."""
    result = {
        "eval_steps": [50_000, 100_000, 150_000],
        "eval_records": [
            {"success": False, "first_arrival_s": None},
            {"success": False, "first_arrival_s": 1.24},
            {"success": True, "first_arrival_s": 1.02},
        ],
    }
    assert first_success_eval_steps(result) == 150_000
    assert first_arrival_eval_steps(result) == 100_000
    never = {"eval_steps": [50_000], "eval_records": [{"success": False, "first_arrival_s": None}]}
    assert first_success_eval_steps(never) is None
    assert first_arrival_eval_steps(never) is None
    assert BATCH_SIZE == 500 and GAMMA == 0.99


# --------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["protocol"]["method_decisions"]["gamma"] == 0.99
    assert report["protocol"]["method_decisions"]["no_advantage_normalization"] is True
    assert report["protocol"]["reward"]["failure_penalty"] == 10.0
    assert report["protocol"]["reward"]["alive_bonus"] == 0.25
    assert set(report["training"]["tiers"]) == {"auto", "fixed"}
    for tier in ("auto", "fixed"):
        tier_block = report["sac_evaluation"]["tiers"][tier]
        record = tier_block["per_seed"][0]
        for key in (
            "first_successful_eval_steps",
            "first_arrival_eval_steps",
            "eval_curve",
            "final_alpha",
            "final_cover_entropy",
        ):
            assert key in record
        assert record["final_alpha"] == pytest.approx(ALPHA_INIT) if tier == "fixed" else True
    assert len(report["five_way_comparison"]) == 5
    assert (
        report["five_way_comparison"][1]["successes"],
        report["five_way_comparison"][1]["episodes"],
    ) == (0, 60)
    assert (
        report["five_way_comparison"][2]["successes"],
        report["five_way_comparison"][2]["episodes"],
    ) == (0, 60)
    assert "SAC" in report["five_way_comparison"][-1]["label"]
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["alpha_curve_fixed_0"][0] == pytest.approx(ALPHA_INIT)
        assert archive["eval_settled_s_auto_0"].shape == (2,)
        assert archive["det_controls_auto_0"].shape[0] + 1 == archive["det_states_auto_0"].shape[0]
    for name in (
        "summary.json",
        "trajectories.npz",
        "training_curves.png",
        "comparison.png",
        "featured_cases.png",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, **SMALL_CONFIG)


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    from embodied_learning.sac_demo import load_replays

    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["_report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")
    path = work / "trajectories.npz"

    # (1) summary successes disagree with the archive
    parsed = json.loads(summary_text)
    parsed["sac_evaluation"]["tiers"]["auto"]["aggregate"]["successes_per_seed"][0] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")

    # (2) archive bytes tampered -> checksum mismatch
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["reward_curve_auto_0"] = payload["reward_curve_auto_0"] * 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    # (3) an array removed with the hash refreshed -> unexpected key set
    del payload["case0_states"]
    np.savez_compressed(path, **payload)
    parsed = json.loads(summary_text)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) the det episode tampered (recomputation guard) with a fresh hash
    payload = {key: data[key] for key in data if key != "_report"}
    payload["det_controls_auto_0"] = payload["det_controls_auto_0"] * 3.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="disagrees"):
        load_replays(work)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.sac_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--train-steps",
            "800",
            "--n-envs",
            "2",
            "--batch-size",
            "64",
            "--buffer-size",
            "2000",
            "--update-every",
            "4",
            "--eval-every-steps",
            "400",
            "--train-episode-steps",
            "50",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["baseline"] == 2
    assert set(payload["tiers"]) == {"auto", "fixed"}
    assert payload["tiers"]["auto"]["episodes"] == 2
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.sac_swingup",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.sac_demo import SacDemo, load_replays

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = SacDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "环境步" in panel and "种子 0" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("trajectories")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "SAC" in panel
    assert f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    assert "0/60" in panel  # the cited pure-PPO comparison row wording
    demo.mode.set("outcome")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "首达" in panel and "训练后" in panel
    demo.close()
