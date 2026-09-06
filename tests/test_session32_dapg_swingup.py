"""Lesson 32: DAPG-style demonstration airdrop on the lesson-29 PPO stack.

Unit checks pin the BC objective against hand-computed answers (known-answer
loss on a zeroed policy, finite-difference gradient, the linear annealing
schedule), the w_BC = 0 guard twice (bitwise environment stream and bitwise
micro training against lesson-29 train_ppo), the demonstration generator
(quality gate, determinism, hash), the reused lesson-7 acceptance and the
process metrics (upright first arrival, headline first accepted success);
shrunk end-to-end checks cover the record contract, the demo loader's tamper
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

from embodied_learning.dapg_demo import load_replays
from embodied_learning.experiments.dapg_swingup import (
    CAPTURE_ANGLE_RAD,
    DEMO_COUNT,
    EXPERIMENT,
    annotate_case_w_bc,
    bc_loss_and_gradient,
    bc_weight_at,
    dapg_losses_and_gradients,
    default_training_config,
    demonstration_hash,
    demonstration_pairs,
    deterministic_dapg_episode,
    evaluate_dapg_policy,
    expected_npz_keys,
    first_arrival_eval_steps,
    first_arrival_index,
    first_arrival_time_s,
    first_success_eval_steps,
    generate_demonstrations,
    run_experiment,
    w_bc_zero_pipeline_guard,
    w_bc_zero_training_guard,
)
from embodied_learning.experiments.ppo_swingup import (
    DOWN_ANGLE_DEG,
    EVAL_FIELDS,
    GaussianPolicy,
    MLPTower,
    PPOConfig,
    RewardFunction,
    evaluate_policy,
    make_push_plans,
    normalize_observation,
    ppo_losses_and_gradients,
    push_schedule,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.swingup import design_swingup_lqr, wrap_angle

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
    "w_bc_levels": (10.0, 1.0),
    "demo_count": 2,
}


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    reward = RewardFunction(design.controller.reference)
    return design, reward


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("dapg_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


def zeroed_constant_policy(hidden=(2,), output_bias=0.5):
    """A policy whose mean is the constant `output_bias` for every input."""
    policy = GaussianPolicy(5, hidden, seed=3, log_std_init=0.5)
    for weight in policy.trunk.weights:
        weight[:] = 0.0
    for bias in policy.trunk.biases:
        bias[:] = 0.0
    policy.trunk.biases[-1][:] = output_bias
    return policy


# ----------------------------------------------------------- BC objective
def test_bc_loss_hand_computed():
    """Known answer on a constant policy; the annealed weight scales exactly."""
    policy = zeroed_constant_policy()
    demo_obs = np.zeros((3, 5))
    demo_actions = np.array([1.0, -1.0, 0.5])
    mse, grads = bc_loss_and_gradient(policy, demo_obs, demo_actions)
    # residuals: 0.5-1.0, 0.5-(-1.0), 0.5-0.5 -> MSE = (0.25 + 2.25 + 0) / 3
    assert mse == pytest.approx(2.5 / 3.0)
    assert len(grads) == len(policy.trunk.weights) + len(policy.trunk.biases)
    # the constant-mean shortcut: only the output bias has a nonzero gradient
    assert float(grads[-1][0]) == pytest.approx(2.0 * float(np.array([-0.5, 1.5, 0.0]).mean()))
    assert all(np.allclose(g, 0.0) for g in grads[:-1])

    config = PPOConfig(
        n_envs=2,
        rollout_steps=8,
        updates=1,
        epochs=1,
        minibatch=8,
        eval_every=1,
        task_envs=2,
        hidden=(2,),
    )
    rng = np.random.default_rng(11)
    obs = rng.normal(0.0, 0.5, (8, 5))
    actions = rng.uniform(-3.0, 3.0, 8)
    old_logp = rng.normal(0.0, 0.1, 8)
    advantages = rng.normal(0.0, 1.0, 8)
    returns = rng.normal(0.0, 1.0, 8)
    value_net = MLPTower(5, (2,), 1, seed=5)
    plain_losses, plain_grads = ppo_losses_and_gradients(
        policy, value_net, obs, actions, old_logp, advantages, returns, config
    )
    for w_bc in (10.0, 1.0):
        losses, gradients = dapg_losses_and_gradients(
            policy,
            value_net,
            obs,
            actions,
            old_logp,
            advantages,
            returns,
            config,
            w_bc,
            demo_obs,
            demo_actions,
        )
        assert losses["bc"] == mse  # the curve records the raw MSE, unscaled
        assert losses["total"] == pytest.approx(plain_losses["total"] + w_bc * mse)
        n_policy = 2 * (len(config.hidden) + 1)  # trunk weights + trunk biases
        for plain_g, dapg_g, bc_g in zip(
            plain_grads[:n_policy], gradients[:n_policy], grads, strict=True
        ):
            np.testing.assert_allclose(dapg_g, plain_g + w_bc * bc_g, rtol=1e-12)
        # log_std and the value network receive no BC gradient
        for plain_g, dapg_g in zip(plain_grads[n_policy:], gradients[n_policy:], strict=True):
            np.testing.assert_array_equal(plain_g, dapg_g)


def test_bc_gradient_finite_difference():
    """The analytic BC gradient matches a central finite difference."""
    policy = GaussianPolicy(5, (3,), seed=7, log_std_init=0.3)
    rng = np.random.default_rng(13)
    demo_obs = rng.normal(0.0, 0.5, (5, 5))
    demo_actions = rng.uniform(-3.0, 3.0, 5)
    _mse, grads = bc_loss_and_gradient(policy, demo_obs, demo_actions)
    flat_analytic = np.concatenate([g.ravel() for g in grads])
    flat_params = np.concatenate(
        [w.ravel() for w in policy.trunk.weights] + [b.ravel() for b in policy.trunk.biases]
    )

    def loss_at(vector):
        cursor = 0
        for weight in policy.trunk.weights:
            size = weight.size
            weight[:] = vector[cursor : cursor + size].reshape(weight.shape)
            cursor += size
        for bias in policy.trunk.biases:
            size = bias.size
            bias[:] = vector[cursor : cursor + size].reshape(bias.shape)
            cursor += size
        mean = policy.trunk.forward(demo_obs)[0][:, 0]
        return float(np.mean((mean - demo_actions) ** 2))

    epsilon = 1e-6
    numeric = np.empty_like(flat_params)
    for index in range(len(flat_params)):
        plus, minus = flat_params.copy(), flat_params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric[index] = (loss_at(plus) - loss_at(minus)) / (2.0 * epsilon)
    np.testing.assert_allclose(numeric, flat_analytic, rtol=1e-5, atol=1e-8)


def test_bc_weight_annealing_contract():
    """Linear from w_init (first update) to w_min (last update); inputs validated."""
    schedule = [bc_weight_at(u, 250, 10.0, 0.0) for u in range(250)]
    assert schedule[0] == pytest.approx(10.0)
    assert schedule[-1] == pytest.approx(0.0)
    assert schedule[125] == pytest.approx(10.0 - 10.0 * 125 / 249)  # linear in the update index
    steps = np.diff(schedule)
    assert np.allclose(steps, steps[0])  # constant decrement: a linear schedule
    two = [bc_weight_at(u, 2, 1.0, 0.0) for u in range(2)]
    assert two == [pytest.approx(1.0), pytest.approx(0.0)]
    single = bc_weight_at(0, 1, 1.0, 0.0)
    assert single == pytest.approx(1.0)  # a one-update run never anneals
    with pytest.raises(ValueError):
        bc_weight_at(0, 0, 1.0)
    with pytest.raises(ValueError):
        bc_weight_at(0, 10, w_init=1.0, w_min=2.0)
    with pytest.raises(ValueError):
        bc_weight_at(0, 10, w_init=-1.0)


# ------------------------------------------------------------- w_BC = 0 guard
def test_w_bc_zero_pipeline_guard_bitwise_and_annotated_cases(design_and_reward):
    """w_BC = 0: the experiment env emits the lesson-29 states/obs/rewards bitwise."""
    _design, reward = design_and_reward
    config = PPOConfig(
        n_envs=2, rollout_steps=16, updates=3, minibatch=16, eval_every=3, task_envs=1
    )
    guard = w_bc_zero_pipeline_guard(reward, config)
    assert guard["bitwise_identical_rewards"] is True
    assert guard["bitwise_identical_states"] is True
    assert guard["bitwise_identical_observations"] is True
    assert guard["steps"] == 40

    # annotate_case_w_bc mirrors pick_failure_cases' fixed scan order
    def make_episode(w_bc, failed):
        return {
            "w_bc": w_bc,
            "terminated": failed,
            "recovered": not failed,
            "failure_reason": "cart_safety_boundary" if failed else "",
        }

    cases = [{"kind": "eval_failure"}, {"kind": "push_failure"}]
    eval_episodes = [make_episode(10.0, False), make_episode(10.0, True), make_episode(1.0, True)]
    push_episodes = [make_episode(1.0, False), make_episode(1.0, True)]
    annotate_case_w_bc(cases, eval_episodes, push_episodes)
    assert cases[0]["w_bc"] == 10.0  # the first failing eval episode
    assert cases[1]["w_bc"] == 1.0  # the first failing push episode


def test_w_bc_zero_training_guard_matches_lesson29(design_and_reward):
    """w_BC = 0 micro training reproduces lesson-29 train_ppo bit for bit."""
    _design, reward = design_and_reward
    guard = w_bc_zero_training_guard(reward, updates=3)
    assert guard["bitwise_identical_reward_curve"] is True
    assert guard["bitwise_identical_value_loss_curve"] is True
    assert guard["bitwise_identical_policy_weights"] is True


# ------------------------------------------------------------ demonstrations
def test_demonstration_generation_quality_determinism_hash(design_and_reward):
    """All demos pass the lesson-7 gate, regenerate bitwise, and hash stably."""
    design, _reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    demos = generate_demonstrations(design, reference, dt, count=3, master_seed=0)
    assert len(demos) == 3
    assert all(demo["success"] for demo in demos)
    assert all(demo["settled_at_s"] is not None for demo in demos)
    down = reference[1] + np.deg2rad(DOWN_ANGLE_DEG)
    for demo in demos:
        offset = float(wrap_angle(demo["start_state"][1] - down))
        assert abs(offset) <= 0.15 + 1e-12
        assert abs(demo["start_state"][0]) <= 0.10 + 1e-12
        assert abs(demo["start_state"][3]) <= 0.30 + 1e-12
    again = generate_demonstrations(design, reference, dt, count=3, master_seed=0)
    for first, second in zip(demos, again, strict=True):
        np.testing.assert_array_equal(first["states"], second["states"])
        np.testing.assert_array_equal(first["controls"], second["controls"])
    assert demonstration_hash(demos) == demonstration_hash(again)
    other = generate_demonstrations(design, reference, dt, count=3, master_seed=1)
    assert demonstration_hash(other) != demonstration_hash(demos)

    obs, actions = demonstration_pairs(demos, reference)
    total_steps = sum(len(demo["controls"]) for demo in demos)
    assert obs.shape == (total_steps, 5) and actions.shape == (total_steps,)
    assert np.all(np.abs(actions) <= 3.0)
    np.testing.assert_array_equal(obs[0], normalize_observation(demos[0]["states"][0], reference))


# ------------------------------------------------------------------ training
def test_micro_training_deterministic(design_and_reward):
    """Fixed seeds reproduce DAPG training bitwise; the budget matches lesson 29."""
    from embodied_learning.experiments.dapg_swingup import train_dapg

    _design, reward = design_and_reward
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
    rng = np.random.default_rng(21)
    demo_obs = rng.normal(0.0, 1.0, (24, 5))
    demo_actions = rng.uniform(-3.0, 3.0, 24)

    def one_run():
        from embodied_learning.experiments.dapg_swingup import make_training_env

        vec = make_training_env(reward, config=config, base_seed=7)
        try:
            return train_dapg(
                vec,
                demo_obs,
                demo_actions,
                config=config,
                init_seed=[0, 7000, 0],
                act_seed=[0, 5000, 0],
                shuffle_seed=[0, 9000, 0],
                demo_seed=[0, 4400, 0],
                w_bc_init=1.0,
                w_min=0.0,
            )
        finally:
            vec.close()

    first, second = one_run(), one_run()
    np.testing.assert_array_equal(first["reward_curve"], second["reward_curve"])
    np.testing.assert_array_equal(first["bc_curve"], second["bc_curve"])
    np.testing.assert_array_equal(first["value_loss_curve"], second["value_loss_curve"])
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
    assert np.isfinite(first["reward_curve"]).all() and np.isfinite(first["bc_curve"]).all()
    assert (first["bc_curve"] > 0.0).all()  # a raw MSE on random demo pairs stays positive
    assert default_training_config().updates == 250  # the lesson-29 budget is kept
    assert default_training_config().n_envs * default_training_config().rollout_steps == 2000
    assert DEMO_COUNT == 8  # DAPG's handful of demonstrations


# ------------------------------------------------------------------- metrics
def test_first_arrival_and_first_success_metrics(design_and_reward):
    """Upright first arrival and the headline first-success checkpoint steps."""
    _design, reward = design_and_reward
    reference = reward.reference

    def shifted(alpha):
        state = np.asarray(reference, dtype=float).copy()
        state[1] += alpha
        return state

    states = np.asarray([shifted(np.pi), shifted(1.0), shifted(0.2), shifted(0.1)])
    assert first_arrival_index(states, reference) == 2
    assert first_arrival_time_s(states, reference, 0.04) == pytest.approx(0.08)
    outside = np.asarray([shifted(1.0), shifted(-1.0)])
    assert first_arrival_index(outside, reference) is None
    assert first_arrival_time_s(outside, reference, 0.04) is None
    boundary = np.asarray([shifted(CAPTURE_ANGLE_RAD)])
    assert first_arrival_index(boundary, reference) == 0  # the threshold is inclusive

    result = {
        "eval_steps": np.asarray([50, 100, 150]),
        "eval_records": [
            {"success": False, "first_arrival_s": None},
            {"success": False, "first_arrival_s": 1.2},
            {"success": True, "first_arrival_s": 1.0},
        ],
    }
    assert first_arrival_eval_steps(result) == 100
    assert first_success_eval_steps(result) == 150
    never = {
        "eval_steps": np.asarray([50]),
        "eval_records": [{"success": False, "first_arrival_s": None}],
    }
    assert first_success_eval_steps(never) is None
    assert first_arrival_eval_steps(never) is None


def test_acceptance_reuses_lesson7_recovery_metrics(design_and_reward):
    """deterministic_dapg_episode/evaluate wrap the lesson-7 recovery_metrics."""
    _design, reward = design_and_reward
    reference, dt = reward.reference, 0.04
    policy = GaussianPolicy(5, (4, 4), seed=2, log_std_init=0.1)
    record, arrays = deterministic_dapg_episode(policy, reward, reference, dt)
    direct_view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    reason = record["failure_reason"]
    direct = recovery_metrics(direct_view, {"failure_reason": reason}, reference, dt)
    for field in EVAL_FIELDS:
        if field in record:
            assert record[field] == direct[field]
    assert record["first_arrival_s"] == first_arrival_time_s(arrays["states"], reference, dt)
    episodes = evaluate_dapg_policy(policy, reward, reference, dt, master_seed=0, count=2)
    assert len(episodes) == 2
    for episode in episodes:
        for field in EVAL_FIELDS:
            assert field in episode
        assert episode["first_arrival_s"] == first_arrival_time_s(
            episode["arrays"]["states"], reference, dt
        )
    same = evaluate_policy(policy, reward, reference, dt, master_seed=0, count=2)
    for episode, plain in zip(episodes, same, strict=True):
        assert episode["recovered"] == plain["recovered"]
        assert episode["settled_at_s"] == plain["settled_at_s"]
        np.testing.assert_array_equal(episode["arrays"]["states"], plain["arrays"]["states"])


def test_push_plans_are_paired_with_lesson29():
    """Same seed stream as lesson 29: identical plans for every controller."""
    from embodied_learning.experiments.ppo_swingup import EVAL_EPISODE_STEPS as HORIZON

    dt = 0.04
    plans = make_push_plans(dt, 20, master_seed=0)
    assert len(plans) == 20
    assert {plan["force_n"] for plan in plans} == {-200.0, 200.0}
    assert all(6.0 <= plan["start_s"] <= 8.0 for plan in plans)
    schedule = push_schedule(plans[0], dt, HORIZON)
    start_step = round(plans[0]["start_s"] / dt)
    assert schedule[:start_step].sum() == 0
    assert np.all(schedule[start_step : start_step + 5] == plans[0]["force_n"])
    same = make_push_plans(dt, 20, master_seed=0)
    assert same == plans


# ------------------------------------------------------ shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert [entry["w_bc"] for entry in report["sweep"]] == [10.0, 1.0]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    pipeline = report["guard"]["pipeline"]
    assert pipeline["bitwise_identical_rewards"] is True
    assert pipeline["bitwise_identical_states"] is True
    training_guard = report["guard"]["training"]
    assert training_guard["bitwise_identical_policy_weights"] is True
    demo = report["protocol"]["demonstrations"]
    assert demo["count"] == 2 and demo["successes"] == 2
    assert len(demo["sha256"]) == 64
    for entry in report["sweep"]:
        assert entry["stochastic"]["episodes"] == 2
        assert entry["push"]["episodes"] == 2
        assert len(entry["arrival"]["first_arrival_s_per_episode"]) == 2
        assert "shaping" not in entry  # the reward is the lesson-29 task reward
        assert len(entry["bc_decay"]["per_seed"]) == 1
        assert entry["first_success"]["any"] in (True, False)
        for record in entry["training"]:
            assert "first_arrival_eval_steps" in record
            assert "first_successful_eval_steps" in record
            assert record["final_w_bc"] == pytest.approx(0.0)  # annealed to w_min
            assert all("first_arrival_s" in point for point in record["eval_curve"])
    assert len(report["three_way_comparison"]) == 6
    assert report["three_way_comparison"][1]["successes"] == 0  # cited lesson-29 PPO
    assert report["three_way_comparison"][1]["episodes"] == 60
    assert report["three_way_comparison"][2]["successes"] == 0  # cited lesson-31 PBRS
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["eval_settled_s_0"].shape == (1, 2)
        assert archive["demo_states"].shape[0] == 2
        assert archive["bc_curve_0_0"].shape == (report["hyperparameters"]["updates"],)
    for name in (
        "summary.json",
        "trajectories.npz",
        "training_curves.png",
        "airdrop_analysis.png",
        "three_way.png",
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

    def rewrite(work_dir, parsed):
        (work_dir / "summary.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    # (1) summary successes disagree with the archive
    parsed = json.loads(summary_text)
    parsed["sweep"][0]["stochastic"]["successes_per_seed"][0] += 1
    rewrite(work, parsed)
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
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) the BC curve end tampered with a fresh hash; start from the complete
    # pristine payload so only the BC curve differs
    payload = {key: data[key] for key in data if key != "report"}
    payload["bc_curve_0_0"][-1] = payload["bc_curve_0_0"][-1] / 2.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="BC curve end"):
        load_replays(work)

    # (5) demonstration arrays tampered with a fresh hash; restore the BC curve
    # (aliased with `data` in step 4) from the pristine record so only the
    # demonstration arrays differ from the recorded hash
    with np.load(output / "trajectories.npz", allow_pickle=False) as pristine:
        payload["bc_curve_0_0"] = pristine["bc_curve_0_0"].copy()
    payload["demo_controls"][0, 0] = 2.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rewrite(work, parsed)
    with pytest.raises(ValueError, match="recorded hash"):
        load_replays(work)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.dapg_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--w-bc",
            "1.0",
            "--demo-count",
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
    payload = json.loads(result.stdout)
    assert payload["guard"]["pipeline"]["bitwise_identical_rewards"] is True
    assert payload["guard"]["training"]["bitwise_identical_policy_weights"] is True
    assert payload["baseline"] == 2
    assert payload["dapg"]["w=1"] == 0
    assert payload["first_success"]["w=1"] in (True, False)
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "airdrop_analysis.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.dapg_swingup",
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

    from embodied_learning.dapg_demo import DapgDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = DapgDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 3
    panel = demo.stats.cget("text")
    assert "环境步" in panel and "w=10" in panel and "BC" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 3  # mode switches must not accumulate axes
    demo.mode.set("airdrop")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "教师" in panel and "示教质量" in panel
    assert f"{report['baseline']['median_settled_at_s']:.2f} s" in panel
    demo.mode.set("outcome")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "纯 PPO" in panel  # multi-way wording
    assert (
        f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    )  # panel numbers come from the summary
    assert (
        f"{report['lesson29_reference']['successes']}/{report['lesson29_reference']['episodes']}"
        in panel
    )
    demo.close()
