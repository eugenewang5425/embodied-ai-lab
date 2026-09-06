"""Lesson 36: DAgger online correction on the lesson-29 PPO stack.

Unit checks pin the teacher annotation against hand-computed answers (kick and
balance branches from the explicit cart-pole dynamics, determinism, the drop
counting), the data-mixing protocol (student self-collected rollouts appended
with teacher labels, obs/state alignment), the w_BC annealing contract and the
objective's w_BC = 0 gradient contract against lesson 29, the per-episode GAE
known answer, the micro training (loss decreases, fixed seeds bitwise), the
lesson-7 acceptance reuse, the teacher quality gate and the shrunk end-to-end
record contract, the demo loader's tamper rejections, the CLI and the real Tk
demo. No real training is run.
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

import mujoco
import numpy as np
import pytest

from embodied_learning.dagger_demo import load_replays
from embodied_learning.experiments.dagger_swingup import (
    EXPERIMENT,
    bc_weight_at,
    collect_student_rollouts,
    compute_gae_episodes,
    dagger_losses_and_gradients,
    evaluate_dagger_round,
    expected_npz_keys,
    filter_annotation_pairs,
    run_experiment,
    train_dagger_round,
)
from embodied_learning.experiments.dapg_swingup import first_arrival_time_s
from embodied_learning.experiments.ppo_swingup import (
    EVAL_FIELDS,
    GaussianPolicy,
    MLPTower,
    PPOConfig,
    RewardFunction,
    baseline_evaluations,
    down_start_state,
    evaluate_policy,
    normalize_observation,
    ppo_losses_and_gradients,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.swingup import (
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
)

SMALL_CONFIG = {
    "train_seeds": 1,
    "eval_seed_count": 2,
    "rounds": 2,
    "rollouts": 2,
    "updates_per_round": 2,
    "w_bc_levels": (10.0, 0.0),
}


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    reward = RewardFunction(design.controller.reference)
    return design, reward


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("dagger_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


# -------------------------------------------------------- teacher annotation
def test_teacher_annotation_known_answer_kick_and_balance(design_and_reward):
    """Known answers: the kick branch and the LQR balance branch labels.

    At the exact resting down start the energy terms vanish (omega = 0) and
    the controller adds its bounded kick acceleration +4.0; the returned motor
    force follows the explicit cart-pole dynamics F = m00*a + m01*wacc - rhs0
    with wacc = (rhs1 - m10*a)/m11. In the capture region the controller hands
    over to the lesson-7 LQR design, so the label is design.controller.action
    on the same state.
    """
    design = design_swingup_lqr()
    reference = design.controller.reference
    env = make_swingup_environment(max_episode_steps=100)
    try:
        env.reset(seed=0)
        controller = HybridSwingupController(env.unwrapped.model, design)
        down = down_start_state(reference)
        label = controller.action(down)
        assert controller.mode == "kick"

        model = env.unwrapped.model
        data = mujoco.MjData(model)
        data.qpos[:] = down[:2]
        data.qvel[:] = down[2:]
        mujoco.mj_forward(model, data)
        mass_matrix = np.zeros((2, 2))
        mujoco.mj_fullM(model, data, mass_matrix)
        # hand derivation: alpha = pi, omega = 0, x = 0, v = 0
        acceleration = 4.0  # the kick term only; energy/position/velocity vanish
        rhs = data.qfrc_passive - data.qfrc_bias
        omega_acc = (rhs[1] - mass_matrix[1, 0] * acceleration) / mass_matrix[1, 1]
        motor = mass_matrix[0, 0] * acceleration + mass_matrix[0, 1] * omega_acc - rhs[0]
        expected = np.clip(
            motor / design.actuator_gear,
            -design.controller.control_limit,
            design.controller.control_limit,
        )
        assert float(label[0]) == pytest.approx(float(expected), rel=1e-6)

        balance_state = reference.copy()
        balance_state[0] = 0.0
        balance_state[1] += 0.1
        balance_state[2] = 0.0
        balance_state[3] = 0.05
        controller2 = HybridSwingupController(env.unwrapped.model, design)
        label2 = controller2.action(balance_state)
        assert controller2.mode == "balance"
        np.testing.assert_allclose(label2, design.controller.action(balance_state), rtol=1e-6)
    finally:
        env.close()


def test_teacher_annotation_deterministic_and_drop_counting(design_and_reward):
    """Re-annotation is bitwise identical; non-finite pairs are dropped + counted."""
    design, _reward = design_and_reward
    reference = design.controller.reference
    env = make_swingup_environment(max_episode_steps=200)
    try:
        env.reset(seed=0)
        controller = HybridSwingupController(env.unwrapped.model, design)
        states = [down_start_state(reference)]
        state = down_start_state(reference)
        for _ in range(60):
            action = controller.action(state)
            state, _, done, truncated, _info = env.step(action)
            states.append(np.asarray(state, dtype=float).copy())
            if done or truncated:
                break
        controller_reset = HybridSwingupController(env.unwrapped.model, design)
        episode = {"states": np.asarray(states, dtype=float)}
        from embodied_learning.experiments.dagger_swingup import annotate_episode

        obs1, labels1, dropped1 = annotate_episode(controller_reset, episode, reference)
        replay = HybridSwingupController(env.unwrapped.model, design)
        obs2, labels2, dropped2 = annotate_episode(replay, episode, reference)
        assert dropped1 == dropped2 == 0
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_array_equal(obs1, obs2)
        assert np.isfinite(labels1).all()

        polluted = episode.copy()
        # a non-finite frame mid-sequence: the labeler must drop and count it
        polluted["states"] = np.vstack(
            [episode["states"][:1], np.full((1, 4), np.nan), episode["states"][1:]]
        )
        _obs3, _labels3, dropped3 = annotate_episode(controller_reset, polluted, reference)
        assert dropped3 == 1  # the non-finite frame is dropped and counted

        _obs4, labels4, dropped4 = filter_annotation_pairs(
            np.vstack([obs1, np.full((2, 5), np.nan)]),
            np.append(labels1, [np.nan, np.nan]),
        )
        assert dropped4 == 2
        assert len(labels4) == len(labels1)
        np.testing.assert_array_equal(labels4, labels1)
    finally:
        env.close()


def test_data_mixing_protocol(design_and_reward):
    """Student self-collected rollouts appended with teacher labels (alignment)."""
    design, reward = design_and_reward
    reference = design.controller.reference
    policy = GaussianPolicy(5, (8,), seed=1, log_std_init=0.5)
    value = MLPTower(5, (8,), 1, seed=2)
    envs = [make_swingup_environment(max_episode_steps=64) for _ in range(2)]
    try:
        for index, env in enumerate(envs):
            env.reset(seed=100 + index)
        episodes = collect_student_rollouts(
            envs,
            policy,
            value,
            reward,
            reference,
            master_seed=0,
            level_index=0,
            seed_index=0,
            round_index=0,
            horizon=64,
            action_rng=np.random.default_rng(7),
        )
        from embodied_learning.experiments.dagger_swingup import annotate_rollouts

        pairs, dropped, consistent = annotate_rollouts(envs, design, episodes, reference)
        assert dropped == 0 and consistent is True
        total = int(sum(len(labels) for _obs, labels in pairs))
        assert total == int(sum(len(ep["rewards"]) for ep in episodes))
        agg_obs = np.concatenate([obs for obs, _l in pairs], axis=0)
        agg_actions = np.concatenate([labels for _obs, labels in pairs], axis=0)
        assert agg_obs.shape == (total, 5) and agg_actions.shape == (total,)
        # alignment: the first pair labels the very first recorded state
        np.testing.assert_array_equal(
            pairs[0][0][0], normalize_observation(episodes[0]["states"][0], reference)
        )
        fresh = HybridSwingupController(envs[0].unwrapped.model, design)
        assert float(pairs[0][1][0]) == pytest.approx(
            float(fresh.action(episodes[0]["states"][0])[0])
        )
        # aggregation is append-only: one more round concatenates, never replaces
        episodes2 = collect_student_rollouts(
            envs,
            policy,
            value,
            reward,
            reference,
            master_seed=0,
            level_index=0,
            seed_index=0,
            round_index=1,
            horizon=64,
            action_rng=np.random.default_rng(7),
        )
        pairs2, dropped2, _ = annotate_rollouts(envs, design, episodes2, reference)
        agg2_obs = np.concatenate([obs for obs, _l in pairs2], axis=0)
        assert int(len(agg_obs) + len(agg2_obs)) > len(agg_obs)
        assert dropped2 == 0
    finally:
        for env in envs:
            env.close()


# ------------------------------------------------------------- schedules etc.
def test_w_bc_annealing_contract():
    """Step schedule: constant inside a round, linear across the rounds."""
    schedule = [bc_weight_at(r, 6, 10.0, 0.0) for r in range(6)]
    assert schedule == [10.0, 8.0, 6.0, 4.0, 2.0, 0.0]
    assert bc_weight_at(0, 6, 1.0, 0.0) == pytest.approx(1.0)
    assert bc_weight_at(5, 6, 1.0, 0.0) == pytest.approx(0.0)
    assert bc_weight_at(0, 1, 10.0, 0.0) == pytest.approx(10.0)  # never anneals
    with pytest.raises(ValueError):
        bc_weight_at(0, 0, 1.0)
    with pytest.raises(ValueError):
        bc_weight_at(0, 6, w_init=1.0, w_min=2.0)
    with pytest.raises(ValueError):
        bc_weight_at(0, 6, w_init=-1.0)
    with pytest.raises(ValueError):
        bc_weight_at(6, 6, 1.0)
    with pytest.raises(ValueError):
        bc_weight_at(-1, 6, 1.0)


def test_dagger_objective_gradient_contract():
    """w_BC = 0 gives the lesson-29 gradients bitwise; w > 0 adds w * BC grads."""
    policy = GaussianPolicy(5, (4,), seed=3, log_std_init=0.4)
    value = MLPTower(5, (4,), 1, seed=4)
    config = PPOConfig(
        n_envs=2,
        rollout_steps=8,
        updates=1,
        epochs=1,
        minibatch=8,
        eval_every=1,
        task_envs=2,
        hidden=(4,),
    )
    rng = np.random.default_rng(11)
    obs = rng.normal(0.0, 0.5, (8, 5))
    actions = rng.uniform(-3.0, 3.0, 8)
    old_logp = rng.normal(0.0, 0.1, 8)
    advantages = rng.normal(0.0, 1.0, 8)
    returns = rng.normal(0.0, 1.0, 8)
    agg_obs = rng.normal(0.0, 0.5, (16, 5))
    agg_actions = rng.uniform(-3.0, 3.0, 16)
    plain_losses, plain_grads = ppo_losses_and_gradients(
        policy, value, obs, actions, old_logp, advantages, returns, config
    )
    losses, grads = dagger_losses_and_gradients(
        policy,
        value,
        obs,
        actions,
        old_logp,
        advantages,
        returns,
        config,
        0.0,
        agg_obs,
        agg_actions,
    )
    for plain, dagg in zip(plain_grads, grads, strict=True):
        np.testing.assert_array_equal(plain, dagg)
    assert losses["policy"] == plain_losses["policy"]
    assert losses["total"] == pytest.approx(plain_losses["total"])

    from embodied_learning.experiments.dapg_swingup import bc_loss_and_gradient

    bc_mse, bc_grads = bc_loss_and_gradient(policy, agg_obs, agg_actions)
    losses_w, grads_w = dagger_losses_and_gradients(
        policy,
        value,
        obs,
        actions,
        old_logp,
        advantages,
        returns,
        config,
        3.0,
        agg_obs,
        agg_actions,
    )
    n_policy = 2 * (len(config.hidden) + 1)  # trunk weights + trunk biases
    for index in range(n_policy):
        np.testing.assert_allclose(
            grads_w[index], plain_grads[index] + 3.0 * bc_grads[index], rtol=1e-12
        )
    for index in range(n_policy, len(plain_grads)):
        np.testing.assert_array_equal(grads_w[index], plain_grads[index])
    assert losses_w["bc"] == pytest.approx(bc_mse)
    assert losses_w["total"] == pytest.approx(plain_losses["total"] + 3.0 * bc_mse)


def test_gae_episodes_known_answer():
    """Per-episode GAE pinned against hand-computed values (term + truncation)."""
    episodes = [
        {
            "values": np.array([2.0, 1.0]),
            "rewards": np.array([1.0, 0.5]),
            "terminated": np.array([False, True]),
            "truncated": np.array([False, False]),
            "terminal_values": np.array([99.0, 99.0]),
        }
    ]
    advantages = compute_gae_episodes(episodes, 0.99, 0.95)
    # step 1: terminated -> no bootstrap, delta = 0.5 - 1.0
    # step 0: next value = values[1] = 1.0, delta = 1.0 + 0.99 - 2.0
    np.testing.assert_allclose(advantages[0], [-0.01, -0.5], rtol=1e-12)
    truncated = [
        {
            "values": np.array([2.0]),
            "rewards": np.array([1.0]),
            "terminated": np.array([False]),
            "truncated": np.array([True]),
            "terminal_values": np.array([3.0]),
        }
    ]
    advantages2 = compute_gae_episodes(truncated, 0.99, 0.95)
    np.testing.assert_allclose(advantages2[0], [1.0 + 0.99 * 3.0 - 2.0], rtol=1e-12)


# ------------------------------------------------------------ micro training
def test_micro_training_deterministic_and_improves(design_and_reward):
    """One round: fixed seeds reproduce bitwise; the BC loss decreases."""
    design, reward = design_and_reward
    reference = design.controller.reference
    config = PPOConfig(
        n_envs=2,
        rollout_steps=64,
        updates=8,
        epochs=1,
        minibatch=64,
        train_episode_steps=64,
        eval_every=8,
        task_envs=2,
        hidden=(8,),
    )
    rng = np.random.default_rng(21)
    agg_obs = rng.normal(0.0, 1.0, (32, 5))
    agg_actions = rng.uniform(-3.0, 3.0, 32)

    def one_run():
        policy = GaussianPolicy(5, (8,), seed=5, log_std_init=0.3)
        value = MLPTower(5, (8,), 1, seed=6)
        parameters = [*policy.parameters(), *value.weights, *value.biases]
        from embodied_learning.experiments.bc_imitation import AdamOptimizer

        optimizer = AdamOptimizer(parameters, lr=config.lr)
        envs = [make_swingup_environment(max_episode_steps=64) for _ in range(2)]
        for index, env in enumerate(envs):
            env.reset(seed=77 + index)
        try:
            episodes = collect_student_rollouts(
                envs,
                policy,
                value,
                reward,
                reference,
                master_seed=0,
                level_index=0,
                seed_index=0,
                round_index=0,
                horizon=64,
                action_rng=np.random.default_rng(11),
            )
            from embodied_learning.experiments.dagger_swingup import rollout_to_batch

            batch, _reward_mean = rollout_to_batch(
                episodes,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                reward_scale=config.reward_scale,
            )
            bc_curve = np.empty(config.updates)
            for update in range(config.updates):
                stats = train_dagger_round(
                    policy,
                    value,
                    optimizer,
                    parameters,
                    batch,
                    agg_obs,
                    agg_actions,
                    config=config,
                    w_bc=10.0,
                    update_index=update,
                    total_updates=config.updates,
                    mb_rng=np.random.default_rng(3),
                    bc_rng=np.random.default_rng(4),
                )
                bc_curve[update] = stats["bc_mean"]
            return bc_curve, policy
        finally:
            for env in envs:
                env.close()

    first, policy_a = one_run()
    second, _policy_b = one_run()
    np.testing.assert_array_equal(first, second)
    for weight_a, weight_b in zip(policy_a.trunk.weights, _policy_b.trunk.weights, strict=True):
        np.testing.assert_array_equal(weight_a, weight_b)
    assert np.isfinite(first).all() and (first > 0.0).all()
    assert first[-1] < first[0]  # the round's training pulls the mean to the teacher


# ----------------------------------------------------------------- evaluation
def test_eval_matches_lesson7_acceptance(design_and_reward):
    """evaluate_dagger_round wraps the lesson-7 recovery_metrics verbatim."""
    _design, reward = design_and_reward
    reference, dt = reward.reference, 0.04
    policy = GaussianPolicy(5, (4, 4), seed=2, log_std_init=0.1)
    episodes = evaluate_dagger_round(policy, reward, reference, dt, master_seed=0, count=2)
    assert len(episodes) == 2
    plain = evaluate_policy(policy, reward, reference, dt, master_seed=0, count=2)
    for episode, baseline in zip(episodes, plain, strict=True):
        for field in EVAL_FIELDS:
            assert field in episode
            assert episode[field] == baseline[field]
        np.testing.assert_array_equal(episode["arrays"]["states"], baseline["arrays"]["states"])
        assert episode["first_arrival_s"] == first_arrival_time_s(
            episode["arrays"]["states"], reference, dt
        )
        view = {**episode["arrays"], "modes": np.full(len(episode["arrays"]["controls"]), "rl")}
        direct = recovery_metrics(
            view, {"failure_reason": episode["failure_reason"]}, reference, dt
        )
        assert episode["recovered"] == direct["recovered"]
        assert episode["settled_at_s"] == direct["settled_at_s"]


def test_teacher_quality_gate_and_baseline():
    """The teacher re-run passes 20/20 and its repeats coincide (deterministic)."""
    design = design_swingup_lqr()
    records, _states, controls, identical = baseline_evaluations(design, 20)
    assert len(records) == 20
    assert all(r["recovered"] for r in records)
    assert identical is True
    assert controls.shape[0] >= 100  # the teacher settles well within the horizon
    assert np.all(np.abs(controls) <= 3.0 + 1e-9)


# ------------------------------------------------------ shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert [entry["w_bc"] for entry in report["tiers"]] == [10.0, 0.0]
    assert [entry["label"] for entry in report["tiers"]] == ["DAgger", "纯PG微调"]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["teacher_verification"]["gate_passed"] is True
    assert report["training"]["rounds"] == 2
    da = report["tiers"][0]["per_seed"][0]["rounds"]
    sizes = [r["dataset_size"] for r in da]
    assert sizes[1] > sizes[0] > 0  # the aggregation grows the cloud
    assert all(r["dropped_labels"] == 0 for r in da)
    assert all(r["self_consistency"] for r in da)
    assert da[-1]["w_bc"] == pytest.approx(0.0)  # annealed to w_min at round T-1
    assert report["tiers"][1]["per_seed"][0]["rounds"][0]["dataset_size"] == 0
    assert report["tiers"][0]["label"] == "DAgger"
    assert "first_success_per_seed" in report["tiers"][0]["aggregate"]
    assert len(report["comparison"]) == 7
    assert report["comparison"][1]["successes"] == 0  # cited lesson-29 row
    assert report["comparison"][2]["successes"] == 0  # cited lesson-32 row
    assert report["comparison"][5]["episodes"] == 2  # DAgger tier final round
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["eval_recovered_0_0"].shape == (2, 2)
        assert archive["eval_settled_s_0_0"].shape == (2, 2)
        assert archive["label_hist_0_0"].sum() == da[0]["dataset_added"]
        assert archive["label_hist_0_1"].sum() == da[1]["dataset_added"]
        expected_w = np.repeat([10.0, 0.0], 2)
        np.testing.assert_array_equal(archive["w_bc_curve_0_0"], expected_w)
        assert archive["bc_curve_1_0"].shape == (4,)
        assert archive["det_states_0_0_r1"].shape[1] == 4
    for name in (
        "summary.json",
        "trajectories.npz",
        "round_evolution.png",
        "correction_analysis.png",
        "comparison.png",
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
    summary_path = work / "summary.json"
    summary_text = summary_path.read_text(encoding="utf-8")

    def rewrite(parsed):
        summary_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # (1) summary successes disagree with the archive
    parsed = json.loads(summary_text)
    parsed["tiers"][0]["per_seed"][0]["per_round_successes"][0] += 1
    rewrite(parsed)
    with pytest.raises(ValueError, match="successes disagree"):
        load_replays(work)
    summary_path.write_text(summary_text, encoding="utf-8")

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
    rewrite(parsed)
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) the BC curve end tampered with a fresh hash -> BC curve end mismatch
    payload = {key: data[key] for key in data if key != "report"}
    payload["bc_curve_0_0"][-1] = payload["bc_curve_0_0"][-1] / 2.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(parsed)
    with pytest.raises(ValueError, match="BC curve end"):
        load_replays(work)

    # (5) the w_BC schedule tampered -> protocol disagreement
    with np.load(output / "trajectories.npz", allow_pickle=False) as pristine:
        payload["bc_curve_0_0"] = pristine["bc_curve_0_0"].copy()
        payload["w_bc_curve_0_0"] = pristine["w_bc_curve_0_0"].copy()
    payload["w_bc_curve_0_0"] = payload["w_bc_curve_0_0"] * 2.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(parsed)
    with pytest.raises(ValueError, match="w_BC schedule"):
        load_replays(work)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.dagger_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--rounds",
            "2",
            "--rollouts",
            "2",
            "--updates-per-round",
            "2",
            "--w-bc",
            "10.0",
            "0.0",
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
    assert set(payload["tiers"]) == {"DAgger", "纯PG微调"}
    assert "first_success_per_seed" in payload["tiers"]["DAgger"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "round_evolution.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.dagger_swingup",
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

    from embodied_learning.dagger_demo import DaggerDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = DaggerDemo(root, data)
    root.update()
    assert demo.mode.get() == "evolution"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "轮次" in panel and "成功率" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("trajectory")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "教师" in panel and "数据集" in panel
    assert f"{report['baseline']['median_settled_at_s']:.2f} s" in panel
    demo.mode.set("comparison")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "纯 PPO" in panel and "0/60" in panel  # cited comparison wording
    assert f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    demo.close()
