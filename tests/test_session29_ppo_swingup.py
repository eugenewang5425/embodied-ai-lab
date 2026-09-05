"""Lesson 29: hand-written numpy PPO on the lesson-7 swing-up task.

Unit checks hand-compute GAE on known answers and finite-difference the PPO
loss; the end-to-end checks shrink the real experiment (1 training seed, 2
eval seeds, a few updates) into tmp dirs so the module stays fast while still
exercising training, evaluation, the lesson-7 acceptance, archive, demo
loader and the Tk demo.
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
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_FIELDS,
    EXPERIMENT,
    FAILURE_PENALTY,
    AdamOptimizer,
    GaussianPolicy,
    MLPTower,
    PPOConfig,
    RewardFunction,
    VecSwingup,
    clip_gradients_,
    compute_gae,
    down_start_state,
    episode_metrics,
    episode_rewards,
    expected_npz_keys,
    gaussian_entropy,
    make_push_plans,
    normalize_observation,
    ppo_losses_and_gradients,
    push_schedule,
    run_experiment,
    run_policy_episode,
    standardize,
    train_ppo,
)
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.ppo_demo import load_replays
from embodied_learning.swingup import design_swingup_lqr

SMALL_CONFIG = {
    "config": PPOConfig(
        n_envs=2,
        rollout_steps=16,
        updates=3,
        epochs=2,
        minibatch=16,
        train_episode_steps=32,
        eval_every=1,
        task_envs=1,
    ),
    "train_seeds": 1,
    "eval_seed_count": 2,
}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("ppo_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


# ------------------------------------------------------------------- PPO core
def test_gae_matches_hand_computed_known_answer():
    """GAE with truncation bootstrap, boundary cut and segment-end bootstrap."""
    rewards = np.array([[1.0], [1.0], [2.0], [1.0]])
    values = np.array([[2.0], [3.0], [1.0], [2.0]])
    terminated = np.zeros((4, 1), dtype=bool)
    truncated = np.array([[False], [False], [True], [False]])
    terminal_values = np.array([[9.0], [9.0], [5.0], [7.0]])
    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, terminal_values, gamma=0.5, gae_lambda=0.5
    )
    # step 3: segment end -> bootstrap terminal_values[3]; step 2: truncated ->
    # bootstrap terminal_values[2] then CUT the recursion; step 1/0 continue
    # within the episode.
    np.testing.assert_allclose(advantages[:, 0], [0.125, -1.5, 4.125, 2.5], atol=1e-12)
    np.testing.assert_allclose(returns[:, 0], advantages[:, 0] + values[:, 0], atol=1e-12)


def test_gae_cut_stops_reward_leak_across_reset():
    """A truncated step must not import the next episode's advantages.

    Regression guard for the first implementation, which kept the recursion
    running through TimeLimit resets: with the same numbers, the leaked
    advantage of step 1 would be -1.5 + 0.25*4.125 = -0.46875 instead of
    the correct -1.5.
    """
    rewards = np.array([[1.0], [1.0], [2.0], [1.0]])
    values = np.array([[2.0], [3.0], [1.0], [2.0]])
    terminated = np.zeros((4, 1), dtype=bool)
    truncated = np.array([[False], [False], [True], [False]])
    terminal_values = np.zeros((4, 1))
    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, terminal_values, gamma=0.5, gae_lambda=0.5
    )
    assert advantages[1, 0] == pytest.approx(-1.5, abs=1e-12)


def test_ppo_loss_gradients_match_finite_differences():
    """Central-difference check over policy, log-std and value parameters."""
    config = PPOConfig(updates=1, eval_every=1, learn_std=True, log_std_init=0.7)
    policy = GaussianPolicy(5, (6, 5), seed=0, log_std_init=0.7)
    value = MLPTower(5, (6, 5), 1, [0, 1])
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, (16, 5))
    actions = rng.normal(0, 1, 16)
    old_logp = rng.normal(0, 0.1, 16)
    advantages = rng.normal(0, 1, 16)
    returns = rng.normal(0, 2, 16)
    # jitter the biases off their zero init: a ReLU kink at exactly 0 is a
    # genuine subgradient point where any finite difference must disagree
    jitter = np.random.default_rng(11)
    for bias in (*policy.trunk.biases, *value.biases):
        bias += jitter.normal(0, 0.05, bias.shape)

    _, gradients = ppo_losses_and_gradients(
        policy, value, obs, actions, old_logp, advantages, returns, config
    )
    parameters = [
        *policy.trunk.weights,
        *policy.trunk.biases,
        policy.log_std,
        *value.weights,
        *value.biases,
    ]
    eps = 1e-6
    worst = 0.0
    for parameter, gradient in zip(parameters, gradients, strict=True):
        flat, grad = parameter.ravel(), gradient.ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            loss_plus, _ = ppo_losses_and_gradients(
                policy, value, obs, actions, old_logp, advantages, returns, config
            )
            flat[index] = original - eps
            loss_minus, _ = ppo_losses_and_gradients(
                policy, value, obs, actions, old_logp, advantages, returns, config
            )
            flat[index] = original
            numeric = (loss_plus["total"] - loss_minus["total"]) / (2 * eps)
            worst = max(
                worst, abs(numeric - grad[index]) / max(abs(numeric), abs(grad[index]), 1e-8)
            )
    assert worst < 1e-5


def test_fixed_std_gets_exactly_zero_gradient():
    """learn_std=False freezes the exploration std (the pilot failure mode)."""
    config = PPOConfig(updates=1, eval_every=1, learn_std=False, log_std_init=0.7)
    policy = GaussianPolicy(5, (6, 5), seed=1, log_std_init=0.7)
    value = MLPTower(5, (6, 5), 1, [1, 2])
    rng = np.random.default_rng(5)
    batch = (
        rng.normal(0, 1, (16, 5)),
        rng.normal(0, 1, 16),
        rng.normal(0, 0.1, 16),
        rng.normal(0, 1, 16),
        rng.normal(0, 2, 16),
    )
    _, gradients = ppo_losses_and_gradients(policy, value, *batch, config)
    log_std_gradient = gradients[len(policy.trunk.weights) + len(policy.trunk.biases)]
    np.testing.assert_array_equal(log_std_gradient, np.zeros_like(policy.log_std))


def test_standardize_and_gradient_clipping_contracts():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    standardized = standardize(values)
    assert standardized.mean() == pytest.approx(0.0, abs=1e-12)
    assert standardized.std() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(standardize(np.ones(5)), np.zeros(5))  # degenerate batch
    with pytest.raises(ValueError):
        standardize(np.array([1.0, np.nan]))
    gradients = [np.array([[3.0, 0.0]]), np.array([4.0])]
    norm = clip_gradients_(gradients, 2.5)  # global norm is 5, clipped to 2.5
    assert norm == pytest.approx(5.0)
    clipped_norm = np.sqrt(sum(float(np.sum(g * g)) for g in gradients))
    assert clipped_norm == pytest.approx(2.5)
    assert gaussian_entropy(np.array([np.log(2.0)])) == pytest.approx(
        np.log(2.0) + 0.5 * np.log(2 * np.pi * np.e)
    )


def test_tiny_training_is_deterministic_and_loss_decreases():
    """Fixed seeds reproduce bitwise; repeated steps on one batch cut the loss."""
    config = PPOConfig(
        n_envs=2,
        rollout_steps=32,
        updates=8,
        epochs=2,
        minibatch=32,
        train_episode_steps=64,
        eval_every=8,
        task_envs=1,
    )

    def one_run():
        reward = RewardFunction(np.array([0.0, -0.001666665, 0.0, 0.0]))
        vec = VecSwingup(
            reward, n_envs=config.n_envs, episode_steps=config.train_episode_steps, base_seed=7
        )
        result = train_ppo(
            vec,
            config=config,
            init_seed=[0, 7000],
            act_seed=[0, 5000],
            shuffle_seed=[0, 9000],
        )
        vec.close()
        return result

    first, second = one_run(), one_run()
    np.testing.assert_array_equal(first["reward_curve"], second["reward_curve"])
    np.testing.assert_array_equal(first["value_loss_curve"], second["value_loss_curve"])
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)

    # a fixed batch under repeated gradient steps: the loss must come down
    policy = GaussianPolicy(5, (8, 8), seed=2, log_std_init=0.7)
    value = MLPTower(5, (8, 8), 1, [2, 1])
    rng = np.random.default_rng(9)
    obs = rng.normal(0, 1, (32, 5))
    actions = rng.normal(0, 0.5, 32)
    old_logp = rng.normal(0, 0.05, 32)
    advantages = standardize(rng.normal(0, 1, 32))
    returns = rng.normal(0, 1, 32)
    parameters = [*policy.parameters(), *value.weights, *value.biases]
    optimizer = AdamOptimizer(parameters, lr=3e-3)
    losses = []
    for _ in range(40):
        loss, gradients = ppo_losses_and_gradients(
            policy, value, obs, actions, old_logp, advantages, returns, config
        )
        clip_gradients_(gradients, config.grad_clip)
        optimizer.step(parameters, gradients)
        losses.append(loss["total"])
    assert losses[-1] < 0.5 * losses[0]
    assert np.all(np.isfinite(losses))


# ------------------------------------------------------- reward and acceptance
def test_reward_terms_and_failure_penalty():
    design = design_swingup_lqr()
    reference = design.controller.reference
    reward = RewardFunction(reference)
    upright = reference.copy()  # alpha = 0 -> upright term 1, no control
    terms = reward.terms(upright, 0.0, False)
    assert terms["upright"] == pytest.approx(1.0)
    assert terms["alive"] == pytest.approx(0.25)
    assert terms["total"] == pytest.approx(1.25)
    down = down_start_state(reference)
    hanging = reward.terms(down, 0.0, False)
    assert hanging["upright"] == pytest.approx(0.0)
    assert hanging["total"] == pytest.approx(0.25)  # the hang-still local optimum
    costed = reward.terms(upright, 3.0, False)
    assert costed["control_cost"] == pytest.approx(0.01 * CONTROL_LIMIT**2)
    failed = reward.terms(upright, 0.0, True)
    assert failed["total"] == pytest.approx(-FAILURE_PENALTY)  # replaces everything
    # a real saturated command from the down start trips the cart boundary and
    # the failing step's stored reward is the failure penalty
    policy = GaussianPolicy(5, (4, 4), seed=0, log_std_init=1e-3)
    policy.trunk.weights[-1][:] = 0.0
    policy.trunk.biases[-1][:] = -10.0  # mean action << -CONTROL_LIMIT, always clipped
    arrays, reason = run_policy_episode(
        policy, reward, reference, horizon=200, env_seed=0, deterministic=True
    )
    assert reason == "cart_safety_boundary"
    step_rewards = episode_rewards(arrays, reward)
    assert step_rewards[-1] == pytest.approx(-FAILURE_PENALTY)
    assert step_rewards[:-1].min() >= -1e-9


def test_policy_evaluation_reuses_lesson7_acceptance():
    """episode_metrics is lesson-7 recovery_metrics on the same arrays."""
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    arrays, metadata = run_scenario(Scenario("down", "down"), design, 750)
    direct = recovery_metrics(arrays, metadata, reference, dt)
    through_shim = episode_metrics(arrays, metadata["failure_reason"], reference, dt)
    assert through_shim["recovered"] and through_shim["settled_at_s"] == pytest.approx(4.76)
    for field in EVAL_FIELDS:
        assert through_shim[field] == direct[field]
    assert set(EVAL_FIELDS) <= set(through_shim)


def test_down_start_observation_and_curriculum_contract():
    design = design_swingup_lqr()
    reference = design.controller.reference
    down = down_start_state(reference)
    assert down[1] == pytest.approx(reference[1] - np.pi, abs=1e-12)
    obs = normalize_observation(down, reference)
    assert obs.shape == (5,)
    assert obs[1] == pytest.approx(-1.0) and obs[2] == pytest.approx(0.0, abs=1e-12)
    tilted = reference.copy()
    tilted[1] += 1.234
    any_obs = normalize_observation(tilted, reference)
    assert any_obs[1] ** 2 + any_obs[2] ** 2 == pytest.approx(1.0)  # cos/sin unit circle
    reward = RewardFunction(reference)
    vec = VecSwingup(reward, n_envs=4, episode_steps=8, base_seed=3, task_envs=2)
    try:
        observations = vec.reset()
        task_start = normalize_observation(down_start_state(reference), reference)
        np.testing.assert_array_equal(observations[0], task_start)
        np.testing.assert_array_equal(observations[1], task_start)
        assert np.abs(observations[2] - task_start).sum() > 0  # random banks moved
    finally:
        vec.close()
    with pytest.raises(ValueError):
        VecSwingup(reward, n_envs=2, episode_steps=8, base_seed=3, task_envs=3)
    with pytest.raises(ValueError):
        PPOConfig(updates=1, eval_every=1, minibatch=10**6)
    with pytest.raises(ValueError):
        PPOConfig(updates=1, eval_every=1, task_envs=99)


def test_push_plans_are_paired_and_aligned():
    dt = 0.04
    plans = make_push_plans(dt, 20, master_seed=0)
    assert len(plans) == 20
    assert {plan["force_n"] for plan in plans} == {-200.0, 200.0}
    assert all(6.0 <= plan["start_s"] <= 8.0 for plan in plans)
    schedule = push_schedule(plans[0], dt, 750)
    start_step = round(plans[0]["start_s"] / dt)
    assert schedule[:start_step].sum() == 0
    assert np.all(schedule[start_step : start_step + 5] == plans[0]["force_n"])
    assert schedule[start_step + 5 :].sum() == 0
    same = make_push_plans(dt, 20, master_seed=0)
    assert same == plans  # both controllers see the identical plans


# ------------------------------------------------------ shrunk end-to-end record
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert report["training"]["env_steps_per_seed"] == 3 * 16 * 2
    assert report["baseline"]["episodes"] == 2 and report["baseline"]["successes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["push_test"]["plans"][0]["force_n"] in (-200.0, 200.0)
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        files = set(archive.files)
        reward_curve = archive["reward_curve_0"]
        assert reward_curve.shape == (3,)
        assert archive["eval_det_states"].shape[0] == 1
    assert files == expected_npz_keys(report)
    assert (output / "summary.json").is_file()
    assert (output / "training_curves.png").is_file()
    assert (output / "evaluation.png").is_file()
    assert (output / "push_and_failures.png").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, **SMALL_CONFIG)


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")

    parsed = json.loads(summary_text)
    parsed["ppo_evaluation"]["aggregate"]["successes_per_seed"][0] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="successes disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")

    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["reward_curve_0"] = payload["reward_curve_0"] * 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    del payload["case0_states"]
    np.savez_compressed(path, **payload)
    parsed = json.loads(summary_text)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)
    # the shared small_run directory stays pristine; only the copy was damaged


# ------------------------------------------------------------------- CLI and UI
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.ppo_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--updates",
            "3",
            "--n-envs",
            "2",
            "--rollout-steps",
            "16",
            "--epochs",
            "2",
            "--minibatch",
            "16",
            "--train-episode-steps",
            "32",
            "--eval-every",
            "1",
            "--task-envs",
            "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "training_curves.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.ppo_swingup",
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

    from embodied_learning.ppo_demo import PpoDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = PpoDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 2
    panel = demo.stats.cget("text")
    assert "种子" in panel and "环境步" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2  # mode switches must not accumulate axes
    demo.mode.set("trajectories")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "PPO" in panel
    assert (
        f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    )  # panel numbers come from the summary
    demo.mode.set("push")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "推力" in panel
    eval_counts = report["failure_analysis"]["ppo_eval_counts"]["cart_safety_boundary"]
    assert f"出界 {eval_counts}" in panel
    demo.close()
