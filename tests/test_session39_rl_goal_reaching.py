"""Lesson 39: pure-learning goal reaching on the differential car.

Unit checks pin the protocol contract: the reward function's hand-computed
answers, the lesson-14/21 kinematics inside the env (a step IS one exact SE(2)
integrate_pose), the arrival speed gate, the out-of-bounds penalty, the
lesson-21 evaluation caliber (path length / final distance formulas), the
re-run manual baseline on the showcase goals, the 2-D squashed-Gaussian joint
log-probability known answer, the SAC policy gradients against central finite
differences, micro-training determinism, the shrunk end-to-end record
contract, the demo loader's tamper rejections, the CLI and the real Tk demo.
No real training is run.
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

from embodied_learning.differential_drive import DriveGeometry, integrate_pose
from embodied_learning.experiments.goal_reaching import DT as LESSON21_DT
from embodied_learning.experiments.goal_reaching import evaluate as lesson21_evaluate
from embodied_learning.experiments.rl_goal_reaching import (
    ARENA_HALF_M,
    ARRIVAL_BONUS,
    ARRIVAL_RADIUS_M,
    ARRIVAL_SPEED_M_S,
    DT,
    EXPERIMENT,
    GOAL_MAX_COORD_M,
    GOAL_MIN_DISTANCE_M,
    OUT_OF_BOUNDS_PENALTY,
    STATE_SCALE_M,
    WHEEL_LIMIT_RAD_S,
    GaussianPolicy2D,
    GoalReachingEnv,
    GoalSACConfig,
    expected_npz_keys,
    make_eval_goals,
    manual_actor,
    observation_for,
    policy_loss_and_gradients,
    run_episode,
    run_experiment,
    step_reward,
    train_sac_goal,
    trajectory_metrics,
)
from embodied_learning.experiments.sac_swingup import (
    LOG_PROB_EPS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    squashed_gaussian_log_prob,
)
from embodied_learning.goal_control import DEFAULT_CONFIG
from embodied_learning.rl_goal_demo import load_replays

SMALL_CONFIG = GoalSACConfig(
    train_steps=400,
    episode_steps=50,
    eval_goal_count=5,
    batch_size=32,
    buffer_size=2000,
    warmup_steps=100,
    update_every_env_steps=2,
    eval_every_steps=200,
    hidden=(32, 32),
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("rl_goal_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, train_seeds=2, log=None)
    return output, report


# ------------------------------------------------------- reward + environment
def test_step_reward_known_answers():
    """Hand-computed: -dist, +10 on arrival, -10 out of bounds, additive."""
    assert step_reward(1.0, arrived=False, out_of_bounds=False) == pytest.approx(-1.0)
    assert step_reward(0.03, arrived=True, out_of_bounds=False) == pytest.approx(
        -0.03 + ARRIVAL_BONUS
    )
    assert step_reward(0.5, arrived=False, out_of_bounds=True) == pytest.approx(
        -0.5 - OUT_OF_BOUNDS_PENALTY
    )
    assert step_reward(0.0, arrived=True, out_of_bounds=True) == pytest.approx(
        ARRIVAL_BONUS - OUT_OF_BOUNDS_PENALTY
    )
    with pytest.raises(ValueError):
        step_reward(float("nan"), arrived=False, out_of_bounds=False)


def test_env_kinematics_and_observation_contract():
    """One step IS one exact integrate_pose; obs matches the protocol."""
    env = GoalReachingEnv(goal_seed=[1, 2])
    observation = env.reset(goal=[1.0, 0.5])
    np.testing.assert_allclose(
        observation, [0.0, 0.0, 1.0, 0.0, 1.0 / STATE_SCALE_M, 0.5 / STATE_SCALE_M]
    )
    wheels = np.array([2.0, 4.0])
    next_obs, reward, terminated, truncated, info = env.step(wheels)
    expected_pose = integrate_pose(np.zeros(3), DriveGeometry().body_velocity(wheels), DT)
    np.testing.assert_allclose(env.pose, expected_pose, atol=1e-15)
    np.testing.assert_allclose(next_obs, observation_for(env.pose, env.goal))
    assert not terminated and not truncated
    assert info["outcome"] == "timeout"
    assert reward == pytest.approx(-float(np.linalg.norm(env.pose[:2] - env.goal)))
    assert 1 <= env.step_index <= env.episode_steps
    with pytest.raises(ValueError):
        env.step(np.array([WHEEL_LIMIT_RAD_S + 0.1, 0.0]))


def test_goal_sampling_deterministic_and_bounded():
    """Same seed -> same goal stream; goals respect the arena and min distance."""
    first, second = GoalReachingEnv(goal_seed=7), GoalReachingEnv(goal_seed=7)
    for _ in range(6):
        first.reset()
        second.reset()
        np.testing.assert_array_equal(first.goal, second.goal)
        assert np.linalg.norm(first.goal) >= GOAL_MIN_DISTANCE_M
        assert np.all(np.abs(first.goal) <= GOAL_MAX_COORD_M)


def test_arrival_requires_speed_gate_and_pays_bonus():
    """Crossing 5 cm fast is not an arrival; slowing into the circle is."""
    env = GoalReachingEnv(goal_seed=0)
    env.reset(goal=[0.03, 0.0])
    _obs, reward, terminated, _truncated, info = env.step(np.array([6.0, 6.0]))
    assert not terminated  # speed 0.3 m/s >= the 0.1 m/s gate
    assert info["outcome"] == "timeout"
    assert reward == pytest.approx(-info["distance_m"])
    env.reset(goal=[0.03, 0.0])
    _obs, reward, terminated, _truncated, info = env.step(np.zeros(2))
    assert terminated and info["outcome"] == "arrived"
    assert reward == pytest.approx(-0.03 + ARRIVAL_BONUS)
    assert info["distance_m"] < ARRIVAL_RADIUS_M and info["speed_m_s"] < ARRIVAL_SPEED_M_S


def test_out_of_bounds_terminates_with_penalty():
    """Driving straight through the wall ends the episode with -10 added."""
    env = GoalReachingEnv(goal_seed=0)
    env.reset(goal=[2.0, 0.0])
    reward = None
    for _ in range(env.episode_steps):
        _obs, reward, terminated, _truncated, info = env.step(np.array([6.0, 6.0]))
        if terminated:
            break
    assert terminated and info["outcome"] == "out_of_bounds"
    assert abs(env.pose[0]) > ARENA_HALF_M
    assert reward == pytest.approx(-info["distance_m"] - OUT_OF_BOUNDS_PENALTY)


# ------------------------------------------------------- caliber + baseline
def test_evaluation_caliber_matches_lesson21():
    """Path length and final distance reuse the lesson-21 formulas verbatim."""
    truth = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.05], [0.1, 0.2, 0.1], [0.4, 0.2, 0.0]], dtype=float
    )
    goal = np.array([0.5, 0.3])
    lesson_arrays = {
        "truth": truth,
        "estimated": truth,
        "modes": np.zeros(len(truth), dtype=np.int64),
        "commands": np.zeros((len(truth), 2)),
    }
    lesson = lesson21_evaluate(lesson_arrays, goal, DEFAULT_CONFIG)
    mine = trajectory_metrics(truth, goal, "timeout", None)
    assert LESSON21_DT == DT
    assert mine["path_length_m"] == pytest.approx(lesson["path_length_m"])
    assert mine["final_distance_m"] == pytest.approx(lesson["true_final_distance_m"])
    assert mine["steps"] == lesson["steps"]


def test_manual_baseline_reaches_showcase_goals():
    """The lesson-21 controller on the true pose arrives on both showcases."""
    env = GoalReachingEnv(goal_seed=0)
    for goal in ((1.6, 0.8), (-2.2, 1.2)):
        first, truth = run_episode(manual_actor, env, np.asarray(goal, dtype=float), record=True)
        second, _truth = run_episode(manual_actor, env, np.asarray(goal, dtype=float), record=True)
        assert first["outcome"] == "arrived"
        assert first["final_distance_m"] < ARRIVAL_RADIUS_M
        assert first["path_efficiency"] < 1.15  # turn-in-place costs no path length
        assert 0 < first["arrival_time_s"] < env.episode_steps * DT
        for key in ("outcome", "steps", "arrival_time_s", "path_length_m"):
            assert first[key] == second[key]
        assert len(truth) == first["steps"] + 1


# ---------------------------------------------------------------- SAC pieces
def test_joint_log_prob_known_answer():
    """The 2-D joint density sums per-dim squashed Gaussian densities."""
    policy = GaussianPolicy2D(6, (8,), seed=[3, 3])
    policy.trunk.weights[-1][:] = 0.0
    policy.trunk.biases[-1][:] = np.array([0.4, -0.3, 1.0, -0.5])
    rng = np.random.default_rng(5)
    action, log_prob = policy.sample(np.zeros((1, 6)), rng)
    mean_raw = np.array([0.4, -0.3])
    log_std_raw = np.array([1.0, -0.5])
    mean = np.tanh(mean_raw)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (np.tanh(log_std_raw) + 1.0)
    noise = np.random.default_rng(5).standard_normal((1, 2))
    u = mean + np.exp(log_std) * noise
    np.testing.assert_allclose(action, np.tanh(u), atol=1e-12)
    expected = (
        -0.5 * ((u - mean) / np.exp(log_std)) ** 2
        - log_std
        - 0.5 * np.log(2.0 * np.pi)
        - np.log(1.0 - np.tanh(u) ** 2 + LOG_PROB_EPS)
    ).sum(axis=1)
    assert log_prob[0] == pytest.approx(expected[0], rel=1e-12)
    # the deterministic action is the squashed mean
    np.testing.assert_allclose(policy.mean(np.zeros((1, 6)))[0], mean, atol=1e-12)


def test_policy_gradient_matches_finite_differences():
    """Central-difference check of the SAC policy loss through both dims.

    The numeric loss must carry the reparameterized critic Q(s, a(s)): the
    analytic gradient contains -dQ/da * da/du * du/dtheta, so a surrogate
    with a constant Q would disagree by exactly that term (verified against
    complex-step differentiation while debugging).  A linear surrogate
    Q_tilde = q_value + q_grad . (a - a_ref) reproduces the same gradient at
    the expansion point.  The policy uses a mid-range log-std band: the
    production bounds ([-20, 2]) put untrained sigma near e^-9, where u can
    cross the eps-regularized tanh knee at |u| ~ 3.8 (same "genuine knee"
    caveat as the lesson-38 ReLU note) and FD loses resolution.
    """
    rng = np.random.default_rng(11)
    policy = GaussianPolicy2D(6, (6, 5), seed=[1, 2], log_std_min=-1.5, log_std_max=-0.5)
    observations = rng.normal(size=(8, 6))
    q_value = rng.normal(size=8)
    q_grad = rng.normal(size=(8, 2))
    alpha = 0.17
    sigma = np.exp(policy.distribution(observations)[1])
    assert 0.2 < sigma.min() and sigma.max() < 0.65
    # jitter the biases off their zero init: a ReLU kink at exactly 0 is a
    # genuine subgradient point where any finite difference must disagree
    for biases in policy.trunk.biases:
        biases += np.random.default_rng(13).normal(0, 0.05, biases.shape)

    def action_at():
        mean, log_std, _cache = policy.distribution(observations)
        noise = np.random.default_rng(17).standard_normal(mean.shape)
        return policy.action_bound * np.tanh(mean + np.exp(log_std) * noise)

    action_ref = action_at()

    def surrogate():
        mean, log_std, _cache = policy.distribution(observations)
        noise = np.random.default_rng(17).standard_normal(mean.shape)
        u = mean + np.exp(log_std) * noise
        action = policy.action_bound * np.tanh(u)
        joint = squashed_gaussian_log_prob(u, mean, log_std, policy.action_bound).sum(axis=1)
        q_tilde = q_value + ((action - action_ref) * q_grad).sum(axis=1)
        return float(np.mean(alpha * joint - q_tilde))

    _metrics, gradients = policy_loss_and_gradients(
        policy, observations, q_value, q_grad, np.random.default_rng(17), alpha
    )
    parameters = policy.parameters()
    eps = 1e-6
    worst = 0.0
    for parameter, gradient in zip(parameters, gradients, strict=True):
        flat, grad = parameter.ravel(), gradient.ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            plus = surrogate()
            flat[index] = original - eps
            minus = surrogate()
            flat[index] = original
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - grad[index]) / max(1e-6, abs(grad[index])))
    assert worst < 1e-4


def test_micro_training_is_deterministic():
    """Fixed seeds reproduce the SAC run bitwise; the reward moves."""
    config = GoalSACConfig(
        train_steps=60,
        episode_steps=25,
        eval_goal_count=4,
        batch_size=16,
        buffer_size=500,
        warmup_steps=20,
        update_every_env_steps=2,
        eval_every_steps=30,
        hidden=(16, 16),
    )

    def one_run():
        train_env = GoalReachingEnv([0, 4200, 0], episode_steps=config.episode_steps)
        eval_env = GoalReachingEnv([0, 4100])
        return train_sac_goal(
            train_env,
            eval_env,
            make_eval_goals(0, config.eval_goal_count),
            config,
            master_seed=0,
            seed_index=0,
            log=None,
        )

    first, second = one_run(), one_run()
    for name, curve in first["curves"].items():
        np.testing.assert_array_equal(curve, second["curves"][name])
        assert np.isfinite(curve).all()
    # updates fire at env steps warmup, warmup+2, ..., 60 (both endpoints)
    assert len(first["curves"]["reward_curve"]) == (60 - 20) // 2 + 1
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
    assert first["eval_curve"] == second["eval_curve"]
    assert first["env_steps"] == 60


# ---------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 2
    assert report["baseline"]["aggregate"]["episodes"] == 5
    assert len(report["baseline"]["per_goal"]) == 5
    assert len(report["comparison"]) == 3
    assert report["comparison"][0]["label"].startswith("手工")
    for record in report["rl_evaluation"]["per_seed"]:
        assert record["env_steps"] == SMALL_CONFIG.train_steps
        assert len(record["eval_curve"]) >= 2
        assert record["aggregate"]["episodes"] == 5
        assert record["aggregate"]["successes"] == sum(
            episode["outcome"] == "arrived" for episode in record["per_goal"]
        )
        for episode in record["per_goal"]:
            assert episode["outcome"] in ("arrived", "timeout", "out_of_bounds")
    assert report["hypothesis"]["per_seed_successes"] == [
        record["aggregate"]["successes"] for record in report["rl_evaluation"]["per_seed"]
    ]
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        goals = archive["eval_goals"]
        assert goals.shape == (5, 2)
        assert archive["eval_truth_0_0"].shape[1] == 3
        assert archive["eval_outcome_0"].shape == (5,)
    for name in ("summary.json", "trajectories.npz", "training_curves.png", "comparison.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, config=SMALL_CONFIG, train_seeds=2, log=None)


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")
    parsed = json.loads(summary_text)

    # (1) summary successes disagree with the archived outcomes
    parsed["rl_evaluation"]["per_seed"][0]["aggregate"]["successes"] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="successes disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")
    parsed = json.loads(summary_text)

    # (2) archive bytes tampered -> checksum mismatch
    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["reward_curve_0"] = payload["reward_curve_0"] * 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    # (3) an array removed with the hash refreshed -> unexpected key set
    del payload["baseline_outcome"]
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) showcase goals swapped in the archive (fresh pristine copy)
    work4 = tmp_path / "tampered_goals"
    shutil.copytree(output, work4)
    path4 = work4 / "trajectories.npz"
    with np.load(path4, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["eval_goals"] = payload["eval_goals"] + 0.5
    np.savez_compressed(path4, **payload)
    parsed4 = json.loads((work4 / "summary.json").read_text(encoding="utf-8"))
    parsed4["trajectories_sha256"] = hashlib.sha256(path4.read_bytes()).hexdigest()
    (work4 / "summary.json").write_text(
        json.dumps(parsed4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="showcase goals"):
        load_replays(work4)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.rl_goal_reaching",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--train-steps",
            "200",
            "--episode-steps",
            "30",
            "--eval-goals",
            "4",
            "--batch-size",
            "16",
            "--buffer-size",
            "500",
            "--warmup",
            "50",
            "--update-every",
            "2",
            "--eval-every-steps",
            "100",
            "--hidden",
            "16",
            "16",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["baseline"]["episodes"] == 4
    assert payload["seeds"][0]["episodes"] == 4
    assert "per_seed_successes" in payload["hypothesis"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.rl_goal_reaching",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.rl_goal_demo import RlGoalDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = RlGoalDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "预算" in panel and "α 末值" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("trajectories")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    baseline = report["baseline"]["aggregate"]
    assert "手工控制器" in panel
    assert f"{baseline['successes']}/{baseline['episodes']}" in panel  # summary numbers
    demo.mode.set("efficiency")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "效率中位" in panel and "失败案例" in panel
    demo.close()
