"""Lesson 38: combo swing-up - energy base + chunked multi-modal residual.

Unit checks pin the combination contract (a = 0 reproduces the lesson-7 run
bitwise, |u_res| <= a everywhere, the chunk execution alignment, the mixture
log-density and its PPO gradients by finite differences and a hand-computed
K=2 known answer, the reward boundaries, the not-degrade verdict function and
the reused lesson-7 acceptance); shrunk end-to-end checks cover the training
determinism, the record contract, the demo loader's tamper rejections, the
CLI and the real Tk demo. No real training is run.
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

from embodied_learning.combo_demo import load_replays
from embodied_learning.experiments.combo_swingup import (
    CHUNK_H,
    CONTROL_COST_COEF,
    EVAL_EPISODE_STEPS,
    EXPERIMENT,
    FAILURE_PENALTY,
    NOT_DEGRADE_RATE,
    W_BOTTOM,
    W_UPRIGHT,
    ChunkResidualPolicy,
    ChunkResidualVecSwingup,
    ComboConfig,
    ComboReward,
    combo_episode_metrics,
    combo_episode_rewards,
    combo_ppo_losses_and_gradients,
    deterministic_combo,
    expected_npz_keys,
    mixture_log_prob,
    not_degrade_verdict,
    run_combo_episode,
    run_experiment,
    train_combo_ppo,
)
from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    MLPTower,
    normalize_observation,
)
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.swingup import design_swingup_lqr

SMALL_CONFIG = ComboConfig(
    n_envs=2,
    chunks_per_segment=4,
    updates=3,
    epochs=2,
    minibatch=8,
    eval_every=3,
    task_envs=1,
)
SMALL_KWARGS = {"train_seeds": 1, "eval_seed_count": 2, "limits_n": (25.0, 50.0)}


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    return design, ComboReward(design.controller.reference)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("combo_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, log=None, **SMALL_KWARGS)
    return output, report


# ------------------------------------------------------- reward and verdict
def test_combo_reward_terms_boundaries(design_and_reward):
    """Heavy top, light bottom, cost on the residual, failure replaces all."""
    design, reward = design_and_reward
    upright = design.controller.reference
    bottom = upright.copy()
    bottom[1] += np.pi
    assert reward.terms(upright, 0.0, False)["total"] == pytest.approx(W_UPRIGHT)
    assert reward.terms(bottom, 0.0, False)["total"] == pytest.approx(-W_BOTTOM)
    costed = reward.terms(upright, 0.25, False)  # a = 25 N -> limit_norm = 0.25
    assert costed["upright"] == pytest.approx(W_UPRIGHT)
    assert costed["control_cost"] == pytest.approx(CONTROL_COST_COEF * 0.25**2)
    assert costed["total"] == pytest.approx(W_UPRIGHT - CONTROL_COST_COEF * 0.25**2)
    failed = reward.terms(upright, 0.0, True)
    assert failed["total"] == pytest.approx(-FAILURE_PENALTY)
    # a real combo episode recomputes the same rewards from stored arrays
    policy = ChunkResidualPolicy(2, CHUNK_H, (4, 4), seed=(0, 1), log_std=float(np.log(0.125)))
    arrays, _reason = run_combo_episode(
        policy,
        reward,
        design,
        horizon=120,
        residual_limit_norm=0.25,
        env_seed=0,
        deterministic=True,
    )
    step_rewards = combo_episode_rewards(arrays, reward)
    assert len(step_rewards) == len(arrays["residuals"])
    for step in range(len(step_rewards)):
        expected = reward.terms(
            arrays["states"][step + 1],
            float(arrays["residuals"][step]),
            bool(arrays["end_flags"][0]) and step == len(step_rewards) - 1,
        )["total"]
        assert step_rewards[step] == pytest.approx(expected)


def test_not_degrade_verdict_boundaries():
    """The protocol criterion: every seed's rate >= 18/20, aggregate as well."""
    verdict = not_degrade_verdict([20, 19, 18], 20)
    assert verdict["not_degrade"] is True
    assert verdict["aggregate_rate"] == pytest.approx(57 / 60)
    assert not_degrade_verdict([20, 20, 17], 20)["not_degrade"] is False
    assert not_degrade_verdict([19, 19, 19], 20)["not_degrade"] is True
    scaled = not_degrade_verdict([2], 2)  # micro runs use the same 90% rate
    assert scaled["not_degrade"] is True
    assert scaled["threshold_rate"] == pytest.approx(NOT_DEGRADE_RATE)
    with pytest.raises(ValueError):
        not_degrade_verdict([21], 20)
    with pytest.raises(ValueError):
        not_degrade_verdict([], 20)


# ------------------------------------------------------------- guard + clip
def test_guard_is_bitwise_identical_to_lesson7(design_and_reward):
    """a = 0 through the combo pipeline: the lesson-7 run, bit for bit."""
    design, reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    arrays, reason = run_combo_episode(
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
    metrics = combo_episode_metrics(arrays, reason, reference, dt)
    assert metrics["recovered"] and metrics["settled_at_s"] == pytest.approx(4.76)
    # consistency guards: the a=0 run never queries the policy
    with pytest.raises(ValueError):
        run_combo_episode(
            ChunkResidualPolicy(2, 8, (4, 4), seed=(0, 0)),
            reward,
            design,
            horizon=16,
            residual_limit_norm=0.0,
            env_seed=0,
            deterministic=True,
        )
    with pytest.raises(ValueError):
        run_combo_episode(
            None,
            reward,
            design,
            horizon=16,
            residual_limit_norm=0.25,
            env_seed=0,
            deterministic=True,
        )


def test_residual_clip_contract_always_holds(design_and_reward):
    """|u_res| <= a in evaluation chunks and at the training interface."""
    design, reward = design_and_reward
    limit_norm = 0.25
    policy = ChunkResidualPolicy(2, CHUNK_H, (4, 4), seed=(1, 2), log_std=float(np.log(0.125)))
    # push both experts far beyond the budget: saturation is by construction
    policy.expert_weights[-1][:] = 0.0
    policy.expert_biases[-1][:] = 10.0
    arrays, _reason = run_combo_episode(
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
    # at the training interface huge chunk samples are indistinguishable from
    # their clipped versions - the environment only ever sees clip(z, +/-a)
    vec_huge = ChunkResidualVecSwingup(
        reward,
        design=design,
        residual_limit_norm=limit_norm,
        chunk_h=CHUNK_H,
        n_envs=2,
        episode_steps=3 * CHUNK_H,
        base_seed=5,
        task_envs=2,
    )
    vec_clipped = ChunkResidualVecSwingup(
        reward,
        design=design,
        residual_limit_norm=limit_norm,
        chunk_h=CHUNK_H,
        n_envs=2,
        episode_steps=3 * CHUNK_H,
        base_seed=5,
        task_envs=2,
    )
    try:
        np.testing.assert_array_equal(vec_huge.reset(), vec_clipped.reset())
        for _ in range(3):
            out_huge = vec_huge.step(np.full((2, CHUNK_H), 1e9))
            out_clipped = vec_clipped.step(np.full((2, CHUNK_H), limit_norm))
        for value_huge, value_clipped in zip(out_huge, out_clipped, strict=False):
            np.testing.assert_array_equal(np.asarray(value_huge), np.asarray(value_clipped))
    finally:
        vec_huge.close()
        vec_clipped.close()
    with pytest.raises(ValueError):
        ChunkResidualVecSwingup(
            reward,
            design=design,
            residual_limit_norm=-1.0,
            chunk_h=CHUNK_H,
            n_envs=2,
            episode_steps=8,
            base_seed=5,
        )
    with pytest.raises(ValueError):
        vec_huge.step(np.zeros((2, CHUNK_H + 1)))


# ------------------------------------------------ mixture density + training
def test_mixture_log_prob_known_answer_K2():
    """Hand-computed two-expert chunk density (gated sum of Gaussians)."""
    policy = ChunkResidualPolicy(2, 2, (6, 6), seed=(0, 0), log_std=float(np.log(0.5)))
    policy.expert_weights[0] = np.zeros((6, 2))
    policy.expert_biases[0] = np.array([1.0, 1.0])
    policy.expert_weights[1] = np.zeros((6, 2))
    policy.expert_biases[1] = np.array([-2.0, -2.0])
    policy.gate_weight = np.zeros((6, 2))
    policy.gate_bias = np.array([0.0, 0.0])
    obs = np.zeros((1, 5))
    chunk = np.array([[1.0, 1.0]])
    sigma = 0.5
    log_w = -np.log(2.0)
    log_n0 = -0.5 * 0.0 / sigma**2 - 2 * (np.log(sigma) + 0.5 * np.log(2 * np.pi))
    error1 = ((1.0 - (-2.0)) ** 2) * 2
    log_n1 = -0.5 * error1 / sigma**2 - 2 * (np.log(sigma) + 0.5 * np.log(2 * np.pi))
    expected = np.log(np.exp(log_w + log_n0) + np.exp(log_w + log_n1))
    assert mixture_log_prob(policy, obs, chunk)[0] == pytest.approx(expected, rel=1e-12)
    # the deterministic plan of a uniform gate is the mean of the two chunks
    plan = policy.mean_chunks(obs)[0]
    np.testing.assert_allclose(plan, [-0.5, -0.5])
    rng = np.random.default_rng(3)
    sampled, _logp = policy.sample_chunks(obs, rng)
    assert sampled.shape == (1, 2)


def test_combo_ppo_gradients_match_finite_differences():
    """Central-difference check of the full PPO loss through the mixture."""
    rng = np.random.default_rng(7)
    policy = ChunkResidualPolicy(2, 3, (6, 5), seed=(1, 2), log_std=float(np.log(0.25)))
    value = MLPTower(5, (6, 5), 1, [1, 2, 1])
    config = ComboConfig(
        chunks_per_segment=4, n_envs=2, chunk_h=3, epochs=1, minibatch=8, task_envs=2
    )
    obs = rng.normal(size=(8, 5))
    chunks = rng.normal(0, 0.25, size=(8, 3))
    old_logp = mixture_log_prob(policy, obs, chunks)
    advantages = rng.normal(size=8)
    returns = rng.normal(size=8)
    # jitter the biases off their zero init: a ReLU kink at exactly 0 is a
    # genuine subgradient point where any finite difference must disagree
    for biases in (*policy.trunk.biases, *value.biases):
        biases += np.random.default_rng(11).normal(0, 0.05, biases.shape)
    _losses, gradients = combo_ppo_losses_and_gradients(
        policy, value, obs, chunks, old_logp, advantages, returns, config
    )
    parameters = [*policy.parameters(), *value.weights, *value.biases]

    def total_of(policy_core, value_core):
        losses, _grads = combo_ppo_losses_and_gradients(
            policy_core, value_core, obs, chunks, old_logp, advantages, returns, config
        )
        return losses["total"]

    eps = 1e-6
    worst = 0.0
    for parameter, gradient in zip(parameters, gradients, strict=True):
        flat, grad = parameter.ravel(), gradient.ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            plus = total_of(policy, value)
            flat[index] = original - eps
            minus = total_of(policy, value)
            flat[index] = original
            numeric = (plus - minus) / (2 * eps)
            worst = max(worst, abs(numeric - grad[index]) / max(1e-6, abs(grad[index])))
    assert worst < 1e-4


def test_micro_training_is_deterministic(design_and_reward):
    """Fixed seeds reproduce the chunk-level PPO run bitwise; the reward moves."""
    design, reward = design_and_reward
    config = ComboConfig(
        n_envs=2,
        chunks_per_segment=4,
        updates=6,
        epochs=2,
        minibatch=8,
        eval_every=6,
        task_envs=1,
    )

    def one_run():
        vec = ChunkResidualVecSwingup(
            reward,
            design=design,
            residual_limit_norm=0.25,
            chunk_h=config.chunk_h,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=7,
            task_envs=config.task_envs,
        )
        try:
            return train_combo_ppo(
                vec,
                config=config,
                init_seed=[0, 7000, 0],
                act_seed=[0, 5000, 0],
                shuffle_seed=[0, 9000, 0],
            )
        finally:
            vec.close()

    first, second = one_run(), one_run()
    np.testing.assert_array_equal(first["reward_curve"], second["reward_curve"])
    np.testing.assert_array_equal(first["value_loss_curve"], second["value_loss_curve"])
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
    assert np.isfinite(first["reward_curve"]).all()
    assert first["env_steps"] == 6 * 2 * 4 * CHUNK_H


# --------------------------------------------------- chunk execution contract
def test_chunk_execution_open_loop_alignment(design_and_reward):
    """Residuals equal the chunk planned at each block start (open-loop, H).

    A steep state-dependent policy makes this discriminating: per-step
    replanning would plan from the per-step states, open-loop chunk execution
    plans once per H steps from the block-start state and freezes the plan.
    """
    design, reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    policy = ChunkResidualPolicy(2, CHUNK_H, (8, 8), seed=(3, 4), log_std=float(np.log(0.125)))
    # steep state dependence: the mixture mean saturates toward +/- the budget
    for weight in policy.trunk.weights:
        weight *= 50.0
    arrays, _reason = run_combo_episode(
        policy,
        reward,
        design,
        horizon=8 * CHUNK_H,
        residual_limit_norm=0.25,
        env_seed=0,
        deterministic=True,
    )
    # lesson-7 alignment: states[k] precedes controls[k]
    assert len(arrays["states"]) == len(arrays["controls"]) + 1
    residuals = np.asarray(arrays["residuals"], dtype=float)
    blocks = len(residuals) // CHUNK_H
    assert blocks == 8
    for block in range(blocks):
        start_state = arrays["states"][block * CHUNK_H]
        obs = normalize_observation(start_state, reference)[None, :]
        plan = np.clip(policy.mean_chunks(obs)[0], -0.25, 0.25)
        np.testing.assert_allclose(
            residuals[block * CHUNK_H : (block + 1) * CHUNK_H], plan, atol=1e-6
        )
    # modes are recorded for the capture metric; the base still captures
    det_arrays, det_reason, _det_metrics = deterministic_combo(
        policy, reward, design, limit_norm=0.25
    )
    capture_time_s = next(
        (index * dt for index, mode in enumerate(det_arrays["modes"]) if str(mode) == "balance"),
        None,
    )
    assert capture_time_s is not None and 0.0 < capture_time_s < EVAL_EPISODE_STEPS * dt
    through_shim = combo_episode_metrics(det_arrays, det_reason, reference, dt)
    direct = recovery_metrics(det_arrays, {"failure_reason": det_reason}, reference, dt)
    for field in ("recovered", "terminated", "truncated", "settled_at_s", "peak_abs_motor_force_n"):
        assert through_shim[field] == direct[field]


# ---------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert [entry["limit_n"] for entry in report["sweep"]] == [25.0, 50.0]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["guard"]["bitwise_identical_states"] is True
    assert report["guard"]["bitwise_identical_controls"] is True
    assert report["guard"]["settled_at_s"] == pytest.approx(4.76)
    assert report["teacher_verification"]["gate_passed"] is True
    for entry in report["sweep"]:
        assert entry["stochastic"]["episodes"] == 2
        assert entry["push"]["episodes"] == 2
        assert entry["residual_stats"]["deterministic"]["budget_n"] == entry["limit_n"]
        assert entry["sigma"] == pytest.approx(entry["limit_norm"] * 0.5)
        assert entry["not_degrade_detail"]["threshold_rate"] == pytest.approx(NOT_DEGRADE_RATE)
    assert len(report["four_way_comparison"]) == 5
    assert report["four_way_comparison"][1]["successes"] == 0  # lesson-30 cited row
    assert report["four_way_comparison"][2]["successes"] == 0  # lesson-37 cited row
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
        run_experiment(output, seed=0, config=SMALL_CONFIG, log=None, **SMALL_KWARGS)


def test_acceptance_matches_lesson7_and_first_success_recorded(small_run):
    """Evaluation caliber and process metrics in the record."""
    _output, report = small_run
    for entry in report["sweep"]:
        record = entry["training"][0]
        assert record["env_steps"] == (
            SMALL_CONFIG.updates * SMALL_CONFIG.n_envs * SMALL_CONFIG.chunks_per_segment * CHUNK_H
        )
        assert len(record["eval_curve"]) >= 1
        for point in record["eval_curve"]:
            assert set(point) >= {"env_steps", "success", "settled_at_s", "first_arrival_s"}
        det = record["deterministic"]
        assert set(det) >= {"recovered", "terminated", "settled_at_s", "capture_time_s"}
        # the push plans are paired with the baseline's stream
        assert entry["push"]["episodes"] == 2


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
            "embodied_learning.experiments.combo_swingup",
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
            "2",
            "--n-envs",
            "2",
            "--chunks-per-segment",
            "4",
            "--epochs",
            "2",
            "--minibatch",
            "8",
            "--eval-every",
            "2",
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
    assert "a=50N" in payload["combo"] and "a=50N" in payload["not_degrade"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.combo_swingup",
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

    from embodied_learning.combo_demo import ComboDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = ComboDemo(root, data)
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
    assert "基线" in panel and "朴素残差" in panel  # the three-way trajectory panel
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
