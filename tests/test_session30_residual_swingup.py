"""Lesson 30: residual RL swing-up on top of the lesson-7 energy controller.

Unit checks pin the residual contract (a = 0 reproduces the lesson-7 run
bitwise, |u_RL| <= a everywhere, finite differences through the clip), the
reused lesson-29 trainer (determinism) and the lesson-7 acceptance; shrunk
end-to-end checks cover the record contract, the demo loader's tamper
rejections, the CLI and the real Tk demo. No real training is run.
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
    FAILURE_PENALTY,
    GaussianPolicy,
    PPOConfig,
    RewardFunction,
    make_push_plans,
    push_schedule,
)
from embodied_learning.experiments.residual_swingup import (
    EVAL_EPISODE_STEPS,
    EXPERIMENT,
    GEAR,
    LESSON29_PPO_REFERENCE,
    ResidualVecSwingup,
    capture_time_s,
    default_training_config,
    deterministic_residual,
    expected_npz_keys,
    residual_command,
    residual_episode_metrics,
    residual_episode_rewards,
    residual_stats,
    run_experiment,
    run_residual_episode,
)
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.residual_demo import load_replays
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
    "limits_n": (25.0, 100.0),
}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("residual_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    return design, RewardFunction(design.controller.reference)


# ------------------------------------------------------- residual contract
def test_residual_command_contract():
    """clip(u_energy + clip(u_RL, +/-a), +/-3): the two-level clip, in units."""
    base = np.array([2.9], dtype=np.float32)
    command, residual = residual_command(base, 5.0, 0.5)
    assert residual == pytest.approx(0.5)
    assert command[0] == pytest.approx(3.0)  # 2.9 + 0.5 clipped to the actuator limit
    command, residual = residual_command(base, -5.0, 0.5)
    assert residual == pytest.approx(-0.5)
    assert command[0] == pytest.approx(2.4)
    with pytest.raises(ValueError):
        residual_command(base, 0.0, -0.1)
    with pytest.raises(ValueError):
        residual_command(base, 0.0, CONTROL_LIMIT + 0.1)


def test_guard_is_bitwise_identical_to_lesson7(design_and_reward):
    """a = 0 through the residual pipeline: the lesson-7 run, bit for bit."""
    design, reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    arrays, reason = run_residual_episode(
        None,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=0.0,
        env_seed=0,
        deterministic=True,
    )
    lesson7_arrays, _metadata = run_scenario(Scenario("down", "down"), design, EVAL_EPISODE_STEPS)
    assert np.array_equal(arrays["states"], lesson7_arrays["states"])
    assert np.array_equal(arrays["controls"], lesson7_arrays["controls"])
    assert np.all(arrays["residuals"] == 0.0)
    metrics = residual_episode_metrics(arrays, reason, reference, dt)
    assert metrics["recovered"] and metrics["settled_at_s"] == pytest.approx(4.76)


def test_residual_clip_contract_always_holds(design_and_reward):
    """|u_RL| <= a in training rollouts and evaluation, saturating on purpose."""
    design, reward = design_and_reward
    limit_norm = 0.25
    policy = GaussianPolicy(5, (4, 4), seed=0, log_std_init=1.0)
    policy.trunk.weights[-1][:] = 0.0
    policy.trunk.biases[-1][:] = 10.0  # mean action far beyond the budget
    arrays, _reason = run_residual_episode(
        policy,
        reward,
        design,
        horizon=200,
        residual_limit_norm=limit_norm,
        env_seed=0,
        deterministic=True,
    )
    assert np.all(np.abs(arrays["residuals"]) <= limit_norm + 1e-12)
    assert np.mean(np.abs(arrays["residuals"]) >= limit_norm - 1e-9) == pytest.approx(1.0)
    assert np.all(np.abs(arrays["controls"]) <= CONTROL_LIMIT)
    # at the training interface: huge samples are indistinguishable from the
    # clipped ones - the environment only ever sees clip(u_RL, +/-a)
    vec_huge = ResidualVecSwingup(
        reward,
        design=design,
        residual_limit_norm=limit_norm,
        n_envs=2,
        episode_steps=8,
        base_seed=5,
        task_envs=2,
    )
    vec_clipped = ResidualVecSwingup(
        reward,
        design=design,
        residual_limit_norm=limit_norm,
        n_envs=2,
        episode_steps=8,
        base_seed=5,
        task_envs=2,
    )
    try:
        obs_huge = vec_huge.reset()
        obs_clipped = vec_clipped.reset()
        np.testing.assert_array_equal(obs_huge, obs_clipped)
        for _ in range(3):
            out_huge = vec_huge.step(np.array([1e9, -1e9]))
            out_clipped = vec_clipped.step(np.array([limit_norm, -limit_norm]))
        for value_huge, value_clipped in zip(out_huge, out_clipped, strict=False):
            np.testing.assert_array_equal(np.asarray(value_huge), np.asarray(value_clipped))
    finally:
        vec_huge.close()
        vec_clipped.close()
    with pytest.raises(ValueError):
        ResidualVecSwingup(
            reward,
            design=design,
            residual_limit_norm=-1.0,
            n_envs=2,
            episode_steps=8,
            base_seed=5,
        )


def test_residual_pipeline_gradients_match_finite_differences():
    """Central-difference check through trunk -> clip -> quadratic cost."""
    policy = GaussianPolicy(5, (6, 5), seed=0, log_std_init=0.7)
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, (16, 5))
    base = rng.normal(0, 1.0, 16)
    weights = rng.normal(0, 1, 16)
    limit_norm = 0.5
    # jitter the biases off their zero init: a ReLU kink at exactly 0 is a
    # genuine subgradient point where any finite difference must disagree
    for bias in policy.trunk.biases:
        bias += np.random.default_rng(11).normal(0, 0.05, bias.shape)

    mean_out, cache = policy.trunk.forward(obs)
    mean = mean_out[:, 0]
    residual = np.clip(mean, -limit_norm, limit_norm)
    total = np.clip(base + residual, -CONTROL_LIMIT, CONTROL_LIMIT)
    interior = (np.abs(mean) < limit_norm).astype(float)
    not_saturated = (np.abs(base + residual) < CONTROL_LIMIT).astype(float)
    grad_mean = (2.0 * weights * total * not_saturated * interior).reshape(-1, 1)
    grad_weights, grad_biases = policy.trunk.backward(cache, grad_mean)

    def loss_of_policy(policy_):
        mean_out, _ = policy_.trunk.forward(obs)
        mean_ = mean_out[:, 0]
        residual_ = np.clip(mean_, -limit_norm, limit_norm)
        total_ = np.clip(base + residual_, -CONTROL_LIMIT, CONTROL_LIMIT)
        return float(np.sum(weights * total_**2))

    eps = 1e-6
    worst = 0.0
    for parameters, gradients in (
        (policy.trunk.weights, grad_weights),
        (policy.trunk.biases, grad_biases),
    ):
        for parameter, gradient in zip(parameters, gradients, strict=True):
            flat, grad = parameter.ravel(), gradient.ravel()
            for index in range(flat.size):
                original = flat[index]
                flat[index] = original + eps
                loss_plus = loss_of_policy(policy)
                flat[index] = original - eps
                loss_minus = loss_of_policy(policy)
                flat[index] = original
                numeric = (loss_plus - loss_minus) / (2 * eps)
                worst = max(worst, abs(numeric - grad[index]) / max(1e-8, abs(grad[index])))
    assert worst < 1e-5


def test_micro_training_is_deterministic(design_and_reward):
    """Fixed seeds reproduce the residual training bitwise."""
    design, reward = design_and_reward
    config = PPOConfig(
        n_envs=2,
        rollout_steps=32,
        updates=6,
        epochs=2,
        minibatch=32,
        train_episode_steps=64,
        eval_every=6,
        task_envs=1,
    )

    def one_run():
        from embodied_learning.experiments.ppo_swingup import train_ppo

        vec = ResidualVecSwingup(
            reward,
            design=design,
            residual_limit_norm=0.5,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=7,
            task_envs=config.task_envs,
        )
        result = train_ppo(
            vec,
            config=config,
            init_seed=[0, 7000, 0],
            act_seed=[0, 5000, 0],
            shuffle_seed=[0, 9000, 0],
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
    assert default_training_config().updates == 125
    assert np.isfinite(first["reward_curve"]).all()


# ------------------------------------------------- reward and acceptance
def test_reward_terms_judge_the_residual(design_and_reward):
    """The reward's control cost acts on the residual, at the failure step it
    is replaced by the penalty; the base command is never the judged action."""
    design, reward = design_and_reward
    upright = design.controller.reference
    terms = reward.terms(upright, 0.0, False)
    assert terms["upright"] == pytest.approx(1.0) and terms["total"] == pytest.approx(1.25)
    costed = reward.terms(upright, 1.0, False)  # a = 100 N -> limit_norm = 1.0
    assert costed["control_cost"] == pytest.approx(0.01 * 1.0**2)
    costed_small = reward.terms(upright, 0.25, False)  # a = 25 N
    assert costed_small["control_cost"] == pytest.approx(0.01 * 0.25**2)
    failed = reward.terms(upright, 0.5, True)
    assert failed["total"] == pytest.approx(-FAILURE_PENALTY)
    # a real residual episode recomputes the same rewards from stored arrays
    policy = GaussianPolicy(5, (4, 4), seed=1, log_std_init=1e-3)
    policy.trunk.weights[-1][:] = 0.0
    policy.trunk.biases[-1][:] = 3.0
    arrays, _reason = run_residual_episode(
        policy,
        reward,
        design,
        horizon=120,
        residual_limit_norm=1.0,
        env_seed=0,
        deterministic=True,
    )
    step_rewards = residual_episode_rewards(arrays, reward)
    assert len(step_rewards) == len(arrays["residuals"])
    assert np.isfinite(step_rewards).all()
    for step in range(len(step_rewards)):
        expected = reward.terms(
            arrays["states"][step + 1],
            float(arrays["residuals"][step]),
            bool(arrays["end_flags"][0]) and step == len(step_rewards) - 1,
        )["total"]
        assert step_rewards[step] == pytest.approx(expected)


def test_acceptance_reuses_lesson7_recovery_metrics(design_and_reward):
    """residual_episode_metrics is the lesson-7 function on the same arrays."""
    design, reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    policy = GaussianPolicy(5, (4, 4), seed=2, log_std_init=0.1)
    arrays, reason = run_residual_episode(
        policy,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=1.0,
        env_seed=0,
        deterministic=True,
    )
    through_shim = residual_episode_metrics(arrays, reason, reference, dt)
    direct = recovery_metrics(arrays, {"failure_reason": reason}, reference, dt)
    for field in EVAL_FIELDS:
        assert through_shim[field] == direct[field]
    assert set(EVAL_FIELDS) <= set(through_shim)
    det_arrays, det_reason, det_metrics = deterministic_residual(
        policy, reward, design, limit_norm=0.5
    )
    assert det_reason == ""  # the base never fails from the down start
    assert isinstance(det_metrics["settled_at_s"], float) or det_metrics["settled_at_s"] is None
    capture = capture_time_s(det_arrays["modes"], dt)
    # the base still hands over to LQR; the residual may delay the capture
    assert capture is not None and 0.0 < capture < EVAL_EPISODE_STEPS * dt


def test_residual_stats_and_capture_contract():
    """Usage statistics against the budget; capture time from mode strings."""
    residuals = np.array([0.0, 0.1, 0.48, 0.5, -0.5, -0.25])
    stats = residual_stats(residuals, 0.5)
    assert stats["budget_n"] == pytest.approx(50.0)
    assert stats["steps"] == 6
    # |u_RL| >= 0.95 * budget: 0.48, +0.5, -0.5 (0.25 does not count as below half either)
    assert stats["fraction_at_limit"] == pytest.approx(3.0 / 6.0)
    assert stats["fraction_below_half"] == pytest.approx(2.0 / 6.0)
    assert stats["max_abs_n"] == pytest.approx(50.0)
    zeros = residual_stats(np.zeros(4), 0.0)
    assert zeros["mean_abs_n"] == 0.0 and zeros["fraction_at_limit"] == 0.0
    with pytest.raises(ValueError):
        residual_stats(np.array([np.nan]), 0.5)
    modes = np.array(["swingup", "swingup", "balance", "balance"])
    assert capture_time_s(modes, 0.04) == pytest.approx(0.08)
    assert capture_time_s(np.array(["swingup"]), 0.04) is None


def test_push_plans_are_paired_with_lesson29():
    """Same seed stream as lesson 29: identical plans for both controllers."""
    dt = 0.04
    plans = make_push_plans(dt, 20, master_seed=0)
    assert len(plans) == 20
    assert {plan["force_n"] for plan in plans} == {-200.0, 200.0}
    assert all(6.0 <= plan["start_s"] <= 8.0 for plan in plans)
    schedule = push_schedule(plans[0], dt, EVAL_EPISODE_STEPS)
    start_step = round(plans[0]["start_s"] / dt)
    assert schedule[:start_step].sum() == 0
    assert np.all(schedule[start_step : start_step + 5] == plans[0]["force_n"])
    assert schedule[start_step + 5 :].sum() == 0
    same = make_push_plans(dt, 20, master_seed=0)
    assert same == plans
    assert GEAR == 100.0  # recovery_metrics reports 100 * normalized command


# ------------------------------------------------------ shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert [entry["limit_n"] for entry in report["sweep"]] == [25.0, 100.0]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["guard"]["bitwise_identical_states"] is True
    assert report["guard"]["bitwise_identical_controls"] is True
    assert report["guard"]["settled_at_s"] == pytest.approx(4.76)
    for entry in report["sweep"]:
        assert entry["stochastic"]["episodes"] == 2
        assert entry["push"]["episodes"] == 2
        assert entry["residual_stats"]["deterministic"]["budget_n"] == entry["limit_n"]
    assert len(report["three_way_comparison"]) == 4
    assert report["three_way_comparison"][1]["successes"] == LESSON29_PPO_REFERENCE["successes"]
    assert len(report["hypothesis"]["per_amplitude"]) == 2
    assert report["push_test"]["plans"][0]["force_n"] in (-200.0, 200.0)
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["guard_states"].shape == archive["baseline_states"].shape
        assert archive["eval_settled_s_0"].shape == (1, 2)
    for name in (
        "summary.json",
        "trajectories.npz",
        "training_curves.png",
        "comparison.png",
        "residual_analysis.png",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, **SMALL_CONFIG)


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")

    # (1) summary successes disagree with the archive
    parsed = json.loads(summary_text)
    parsed["sweep"][0]["stochastic"]["successes_per_seed"][0] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="successes disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")

    # (2) archive bytes tampered -> checksum mismatch
    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["reward_curve_0_0"] = payload["reward_curve_0_0"] * 2.0
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

    # (4) the guard claim tampered in the archive itself (fresh pristine copy)
    work4 = tmp_path / "tampered_guard"
    shutil.copytree(output, work4)
    path4 = work4 / "trajectories.npz"
    with np.load(path4, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["guard_states"] = payload["guard_states"] + 1.0
    np.savez_compressed(path4, **payload)
    parsed = json.loads((work4 / "summary.json").read_text(encoding="utf-8"))
    parsed["trajectories_sha256"] = hashlib.sha256(path4.read_bytes()).hexdigest()
    (work4 / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Guard states"):
        load_replays(work4)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.residual_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--limits",
            "50",
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
    payload = json.loads(result.stdout)
    assert payload["guard"]["bitwise_identical_states"] is True
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.residual_swingup",
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

    from embodied_learning.residual_demo import ResidualDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = ResidualDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 2
    panel = demo.stats.cget("text")
    assert "环境步" in panel and "a=25" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2  # mode switches must not accumulate axes
    demo.mode.set("trajectories")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "切入 LQR" in panel  # handoff wording as shown in the panel
    assert (
        f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    )  # panel numbers come from the summary
    demo.mode.set("usage")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "推力" in panel and "均值" in panel  # usage panel wording
    assert f"a={report['sweep'][0]['limit_n']:.0f} N" in panel
    demo.close()
