"""Lesson 40: pure-learning reaching on the full-actuated planar 2R arm.

Unit checks pin the protocol contract: the reward function's hand-computed
answers, the lesson-8 FK/dynamics inside the env (49-pose audit; one env step
IS one exact ArmSimulation.step), the observation protocol (no dq, previous
torque memory), the goal box sampling, the sustained 0.5 s success window
semantics, the instantaneous arrival gate paying +10 in the training env, the
joint-envelope termination, the analytic baseline reproducing the lesson-8
run_reach BITWISE in this arena (same-arena contract), micro-training
determinism, the shrunk end-to-end record contract, the demo loader's tamper
rejections, the CLI and the real Tk demo.  No real training is run.
"""

from __future__ import annotations

import hashlib
import json
import math
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

from embodied_learning.experiments.arm_reaching import INITIAL_Q, run_reach
from embodied_learning.experiments.rl_arm_reaching import (
    ARRIVAL_BONUS,
    ARRIVAL_RADIUS_M,
    ARRIVAL_SPEED_RAD_S,
    DT,
    EXPERIMENT,
    GOAL_HIGH_M,
    GOAL_LOW_M,
    GOAL_MIN_DISTANCE_M,
    JOINT_LIMIT_PENALTY,
    JOINT_LIMIT_RAD,
    OBS_DIM,
    SHOWCASE_GOALS,
    TORQUE_LIMIT_NM,
    ArmReachingEnv,
    ArmSACConfig,
    aggregate_episodes,
    baseline_actor,
    expected_npz_keys,
    make_eval_goals,
    observation_for,
    policy_actor,
    run_episode,
    run_experiment,
    step_reward,
    sustained_window,
    train_sac_arm,
)
from embodied_learning.planar_arm import (
    ArmSimulation,
    forward_kinematics,
    inverse_kinematics,
    joint_pd,
)
from embodied_learning.rl_arm_demo import load_replays

SMALL_CONFIG = ArmSACConfig(
    train_steps=300,
    episode_steps=40,
    eval_goal_count=5,
    batch_size=32,
    buffer_size=2000,
    warmup_steps=100,
    update_every_env_steps=2,
    eval_every_steps=150,
    hidden=(32, 32),
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("rl_arm_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, train_seeds=2, log=None)
    return output, report


# ------------------------------------------------------- reward + environment
def test_step_reward_known_answers():
    """Hand-computed: -dist, +10 on arrival, -10 on the joint envelope."""
    assert step_reward(0.35, arrived=False, joint_limit=False) == pytest.approx(-0.35)
    assert step_reward(0.001, arrived=True, joint_limit=False) == pytest.approx(
        -0.001 + ARRIVAL_BONUS
    )
    assert step_reward(0.4, arrived=False, joint_limit=True) == pytest.approx(
        -0.4 - JOINT_LIMIT_PENALTY
    )
    assert step_reward(0.0, arrived=True, joint_limit=True) == pytest.approx(
        ARRIVAL_BONUS - JOINT_LIMIT_PENALTY
    )
    assert ARRIVAL_RADIUS_M == 0.002  # the lesson-8 2 mm gate
    with pytest.raises(ValueError):
        step_reward(float("nan"), arrived=False, joint_limit=False)


def test_env_fk_and_dynamics_match_lesson8():
    """49-pose FK audit passes and one env step IS one ArmSimulation.step."""
    from embodied_learning.experiments.arm_reaching import audit_geometry

    audit = audit_geometry()
    assert audit["pose_count"] == 49
    assert audit["max_point_error_m"] < 1e-10  # lesson 8 measured ~2.26e-16
    env = ArmReachingEnv(goal_seed=[1, 2])
    sim = ArmSimulation()
    env.reset(goal=[0.35, 0.30])
    sim.reset(INITIAL_Q)
    np.testing.assert_array_equal(env.state, sim.state())
    for torque in ([0.2, -0.1], [-0.25, 0.25], [0.0, 0.0]):
        sim_state, _command, _failure = sim.step(np.asarray(torque, dtype=float))
        env.step(np.asarray(torque, dtype=float))
        np.testing.assert_array_equal(env.state, sim_state)
        np.testing.assert_allclose(env.tip, forward_kinematics(env.state[:2]), atol=1e-12)
        assert abs(sim.dt - DT) < 1e-12


def test_observation_contract():
    """Reset obs is the protocol vector; the applied torque returns scaled."""
    env = ArmReachingEnv(goal_seed=[3, 4])
    observation = env.reset(goal=[0.35, 0.30])
    q0 = INITIAL_Q
    expected = [
        math.cos(q0[0]),
        math.sin(q0[0]),
        math.cos(q0[1]),
        math.sin(q0[1]),
        0.0,
        0.0,
        0.35 / 0.5,
        0.30 / 0.5,
    ]
    np.testing.assert_allclose(observation, expected, atol=1e-12)
    assert env.state[2:] == pytest.approx([0.0, 0.0])  # starts at rest
    next_obs, _reward, _terminated, _truncated, _info = env.step(np.array([0.25, -0.125]))
    np.testing.assert_allclose(next_obs[4:6], [1.0, -0.5], atol=1e-12)
    np.testing.assert_allclose(
        next_obs[:4],
        [
            math.cos(env.state[0]),
            math.sin(env.state[0]),
            math.cos(env.state[1]),
            math.sin(env.state[1]),
        ],
        atol=1e-15,
    )
    # observation_for agrees with the env, and the policy actor maps to N m
    np.testing.assert_allclose(
        next_obs, observation_for(env.state[:2], env.last_torque, env.goal), atol=1e-15
    )
    assert OBS_DIM == 8 and TORQUE_LIMIT_NM == 0.25
    with pytest.raises(ValueError):
        env.step(np.array([TORQUE_LIMIT_NM + 0.01, 0.0]))
    # the deterministic actor maps the squashed mean to N m; the baseline
    # actor reproduces the lesson-8 PD law on the same state
    from embodied_learning.experiments.rl_goal_reaching import GaussianPolicy2D

    policy = GaussianPolicy2D(OBS_DIM, (8,), seed=[3, 3])
    current = observation_for(env.state[:2], env.last_torque, env.goal)
    mean = policy.mean(current[None, :])[0]
    torque = policy_actor(policy)(env.state, env.goal, env.last_torque)
    np.testing.assert_allclose(torque, TORQUE_LIMIT_NM * mean, atol=1e-15)
    assert np.all(np.abs(torque) <= TORQUE_LIMIT_NM)
    goal_q = inverse_kinematics(np.asarray([0.35, 0.30]))[0]
    np.testing.assert_allclose(
        baseline_actor(env.state, env.goal, env.last_torque),
        joint_pd(env.state[:2], env.state[2:], goal_q),
        atol=1e-15,
    )


def test_goal_sampling_deterministic_and_bounded():
    """Same seed -> same goal stream; goals respect box, 0.15 m and annulus."""
    first, second = ArmReachingEnv(goal_seed=7), ArmReachingEnv(goal_seed=7)
    for _ in range(5):
        first.reset()
        second.reset()
        np.testing.assert_array_equal(first.goal, second.goal)
        norm = float(np.linalg.norm(first.goal))
        assert norm >= GOAL_MIN_DISTANCE_M
        assert abs(0.1 - 0.3) < norm < 0.4 + 0.3  # strictly inside the annulus
        np.testing.assert_array_less(GOAL_LOW_M - 1e-12, first.goal)
        np.testing.assert_array_less(first.goal, GOAL_HIGH_M + 1e-12)
    eval_goals = make_eval_goals(0, 20)
    assert eval_goals.shape == (20, 2)
    np.testing.assert_allclose(eval_goals[0], SHOWCASE_GOALS[0], atol=1e-12)
    np.testing.assert_allclose(eval_goals[1], SHOWCASE_GOALS[1], atol=1e-12)
    assert all(np.linalg.norm(goal) >= GOAL_MIN_DISTANCE_M for goal in eval_goals)


def test_sustained_window_semantics():
    """Success needs a 0.5 s hold: a fast fly-through or a 0.3 s dip fails."""
    samples_1s = 51  # 0..50 -> 1.0 s span
    hold_ok = [0.001] * samples_1s, [0.0] * samples_1s
    success, first, settled = sustained_window(*hold_ok, dt=DT)
    assert success and first == 0 and settled == 0
    # a 0.3 s (15 samples) dip into the gate: never 26 consecutive samples
    distances = [0.5] * samples_1s
    distances[20:35] = [0.001] * 15
    speeds = [0.0] * samples_1s
    success, first, settled = sustained_window(distances, speeds, dt=DT)
    assert not success and first is None
    # a 0.6 s hold starting late: first = window start, tail reaches the end
    distances = [0.5] * samples_1s
    distances[30:] = [0.001] * 21  # 21 samples = 0.4 s span only
    success, first, _settled = sustained_window(distances, speeds, dt=DT)
    assert not success
    distances[25:] = [0.001] * 26  # exactly 0.5 s span
    success, first, settled = sustained_window(distances, speeds, dt=DT)
    assert success and first == 25 and settled == 25
    # a hold that ends before the record: success without a settled tail
    distances = [0.5] * samples_1s
    distances[10:40] = [0.001] * 30
    success, first, settled = sustained_window(distances, speeds, dt=DT)
    assert success and first == 10 and settled is None


def test_terminal_events_arrival_bonus_and_joint_limit():
    """Arrival pays +10 and terminates; the joint envelope pays -10."""
    env = ArmReachingEnv(goal_seed=0)  # training env: arrival terminates
    tip = forward_kinematics(INITIAL_Q)
    env.reset(goal=tip)
    _obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and not truncated
    assert info["outcome"] == "arrived"
    assert info["distance_m"] < ARRIVAL_RADIUS_M
    assert info["speed_rad_s"] < ARRIVAL_SPEED_RAD_S
    assert reward == pytest.approx(-info["distance_m"] + ARRIVAL_BONUS)
    # the evaluation env does NOT terminate on the instantaneous gate
    eval_env = ArmReachingEnv(goal_seed=0, terminate_on_arrival=False)
    eval_env.reset(goal=tip)
    _obs, _reward, terminated, _truncated, info = eval_env.step(np.zeros(2))
    assert not terminated and info["outcome"] == "arrived"
    # sustained -0.25 N m on joint 1 drives |q1| past pi: terminal with -10
    env = ArmReachingEnv(goal_seed=0)
    env.reset(goal=[0.35, 0.30])
    info, reward = None, None
    for _ in range(env.episode_steps):
        _obs, reward, terminated, _truncated, info = env.step(np.array([-0.25, 0.0]))
        if terminated:
            break
    assert terminated and info["outcome"] == "joint_limit"
    assert abs(env.state[0]) > JOINT_LIMIT_RAD
    assert reward == pytest.approx(-info["distance_m"] - JOINT_LIMIT_PENALTY)
    assert info["speed_rad_s"] < 20.0  # far from the simulator failure gate


# ------------------------------------------------------- baseline + caliber
def test_baseline_reproduces_lesson8_reach_bitwise():
    """The IK+PD baseline in this arena IS the lesson-8 rollout, bit for bit."""
    for target in ([0.35, 0.30], [0.35, -0.30]):
        lesson_arrays, _meta = run_reach(target, branch=0)
        env = ArmReachingEnv(goal_seed=[5, 5], terminate_on_arrival=False)
        summary, points = run_episode(
            baseline_actor, env, np.asarray(target, dtype=float), record=True
        )
        rows = len(lesson_arrays["points"])
        assert summary["steps"] == 500 and len(points) == 501
        np.testing.assert_array_equal(points[:rows], lesson_arrays["points"])
        # same IK solution, same PD law, same MuJoCo trajectory -> same verdict
        assert summary["outcome"] == "arrived"
        assert summary["settled_tail_s"] is not None
        assert 0 < summary["arrival_time_s"] < 500 * DT


def test_baseline_arrives_on_showcase_goals_deterministically():
    """Both lesson-8 acceptance targets: sustained arrival, bitwise rerun."""
    env = ArmReachingEnv(goal_seed=0, terminate_on_arrival=False)
    for goal in SHOWCASE_GOALS:
        first, points = run_episode(baseline_actor, env, np.asarray(goal), record=True)
        second, points_again = run_episode(baseline_actor, env, np.asarray(goal), record=True)
        assert first["outcome"] == "arrived"
        assert first["arrival_time_s"] is not None
        assert first["final_distance_m"] < ARRIVAL_RADIUS_M
        assert first["path_efficiency"] > 1.0  # the tip swings on an arc
        for key in ("outcome", "steps", "arrival_time_s", "path_length_m", "min_distance_m"):
            assert first[key] == second[key]
        np.testing.assert_array_equal(points, points_again)


def test_aggregate_episodes_known_answers():
    """Aggregates count successes, terminations and medians as documented."""
    episodes = [
        {
            "outcome": "arrived",
            "arrival_time_s": 2.0,
            "settled_tail_s": 2.0,
            "path_efficiency": 1.5,
            "final_distance_m": 0.0,
            "min_distance_m": 0.0,
        },
        {
            "outcome": "arrived",
            "arrival_time_s": 4.0,
            "settled_tail_s": None,
            "path_efficiency": 1.1,
            "final_distance_m": 0.001,
            "min_distance_m": 0.0005,
        },
        {
            "outcome": "timeout",
            "arrival_time_s": None,
            "settled_tail_s": None,
            "path_efficiency": 1.3,
            "final_distance_m": 0.2,
            "min_distance_m": 0.05,
        },
        {
            "outcome": "joint_limit",
            "arrival_time_s": None,
            "settled_tail_s": None,
            "path_efficiency": 2.0,
            "final_distance_m": 0.3,
            "min_distance_m": 0.1,
        },
    ]
    aggregate = aggregate_episodes(episodes)
    assert aggregate["episodes"] == 4
    assert aggregate["successes"] == 2
    assert aggregate["success_rate"] == 0.5
    assert aggregate["settled_successes"] == 1
    assert aggregate["joint_limits"] == 1
    assert aggregate["failures"] == 0
    assert aggregate["median_arrival_time_s"] == pytest.approx(3.0)
    assert aggregate["median_path_efficiency"] == pytest.approx(1.3)
    assert aggregate["median_min_distance_m"] == pytest.approx(0.02525)


# ---------------------------------------------------------------- SAC pieces
def test_micro_training_is_deterministic():
    """Fixed seeds reproduce the SAC run bitwise; the update count matches."""
    config = ArmSACConfig(
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
        train_env = ArmReachingEnv([0, 12000, 0], episode_steps=config.episode_steps)
        eval_env = ArmReachingEnv([0, 14001], terminate_on_arrival=False)
        return train_sac_arm(
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
    # the policy head really is an 8 -> 2 squashed Gaussian on this task
    action = first["policy"].mean(np.zeros((1, OBS_DIM)))
    assert action.shape == (1, 2) and np.all(np.abs(action) <= 1.0)


# ---------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 2
    assert report["geometry_audit"]["pose_count"] == 49
    assert report["baseline"]["aggregate"]["episodes"] == 5
    assert report["baseline"]["aggregate"]["successes"] == 5
    assert len(report["baseline"]["per_goal"]) == 5
    assert len(report["comparison"]) == 3
    assert report["comparison"][0]["label"].startswith("解析 IK + PD")
    for record in report["rl_evaluation"]["per_seed"]:
        assert record["env_steps"] == SMALL_CONFIG.train_steps
        assert len(record["eval_curve"]) >= 2
        assert record["aggregate"]["episodes"] == 5
        assert record["aggregate"]["successes"] == sum(
            episode["outcome"] == "arrived" for episode in record["per_goal"]
        )
        for episode in record["per_goal"]:
            assert episode["outcome"] in ("arrived", "timeout", "joint_limit", "failure")
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
        assert archive["eval_truth_0_0"].shape[1:] == (3, 2)  # base, elbow, tip
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
            "embodied_learning.experiments.rl_arm_reaching",
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
    assert payload["baseline"]["successes"] == 4
    assert payload["seeds"][0]["episodes"] == 4
    assert "per_seed_successes" in payload["hypothesis"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.rl_arm_reaching",
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

    from embodied_learning.rl_arm_demo import RlArmDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = RlArmDemo(root, data)
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
    assert "IK+PD 基线" in panel
    assert f"{baseline['successes']}/{baseline['episodes']}" in panel  # summary numbers
    demo.mode.set("arrival")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "最近中位" in panel and "失败案例" in panel
    demo.close()
