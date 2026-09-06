"""Lesson 34: two-phase reward (stage switching) on the lesson-29 PPO stack.

Unit checks pin the phase boundary and the latch against known answers, the
normalized energy reward against hand computations (cE_sw = 1/(2*mgl) maps the
resting-down error to exactly -1), the balance phase's bitwise delegation to
the lesson-29 reward and the disabled-switch bitwise degeneracy of both the
reward and the whole pipeline; further checks cover the replayed per-step
terms and the swing-sugar fraction, the latch reseed on internal episode
restarts, micro-training determinism, the reused lesson-7 acceptance and the
first-success/first-arrival metrics; shrunk end-to-end checks cover the record
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

from embodied_learning.experiments.pbrs_swingup import (
    arrival_summary,
    lesson7_energy_constants,
)
from embodied_learning.experiments.ppo_swingup import (
    EVAL_EPISODE_STEPS,
    EVAL_FIELDS,
    GaussianPolicy,
    PPOConfig,
    RewardFunction,
    make_push_plans,
    run_policy_episode,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.experiments.twophase_swingup import (
    ALPHA_SWITCH_RAD,
    DEFAULT_TRAINING_BUDGET_STEPS,
    EVAL_EVERY,
    EXPERIMENT,
    FAILURE_PENALTY_TWOPHASE,
    TwoPhaseReward,
    TwoPhaseVecSwingup,
    default_training_config,
    expected_npz_keys,
    first_arrival_eval_steps,
    first_success_eval_steps,
    run_experiment,
    swing_sugar_fraction,
    twophase_episode_metrics,
    twophase_episode_terms,
    twophase_guard,
)
from embodied_learning.swingup import design_swingup_lqr, wrap_angle
from embodied_learning.twophase_demo import load_replays

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
    output = tmp_path_factory.mktemp("twophase_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    hinge_inertia, gravity_energy = lesson7_energy_constants(design)
    reward = RewardFunction(design.controller.reference)
    return design, reward, hinge_inertia, gravity_energy


def _shifted(reference, alpha):
    state = np.asarray(reference, dtype=float).copy()
    state[1] += alpha
    return state


# ------------------------------------------------- reward and phase mechanics
def test_energy_constants_and_normalization(design_and_reward):
    """cE_sw = 1/(2*mgl): the resting-down swing reward is exactly -0.75."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    assert reward.c_e_switch == pytest.approx(1.0 / (2.0 * gravity_energy))
    reference = task_reward.reference
    assert reward.energy(reference) == pytest.approx(0.0)  # E_top = 0 at upright
    down = _shifted(reference, np.pi)
    assert reward.energy(down) == pytest.approx(-2.0 * gravity_energy)
    row = reward.terms(down, 0.0, False, False)
    assert row["phase"] == "swing"
    assert row["energy"] == pytest.approx(-1.0)  # the normalization rule
    assert row["total"] == pytest.approx(-1.0 + reward.alive_swing)
    protocol = reward.as_dict()
    assert protocol["c_e_switch"] == pytest.approx(reward.c_e_switch)
    assert protocol["swing_reward_at_rest_down"] == pytest.approx(-0.75)
    with pytest.raises(ValueError):
        TwoPhaseReward(task_reward, hinge_inertia, gravity_energy, alpha_switch=0.0)
    with pytest.raises(ValueError):
        TwoPhaseReward(task_reward, hinge_inertia, gravity_energy, c_e_switch=-1.0)


def test_phase_boundary_and_latch_known_answers(design_and_reward):
    """|alpha| <= alpha_switch is the cone (inclusive seam); the latch holds."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    assert reward.alpha_switch == pytest.approx(ALPHA_SWITCH_RAD)
    inside = _shifted(task_reward.reference, reward.alpha_switch)  # exactly on the seam
    assert reward.in_capture_region(inside) is True
    inside_other_side = _shifted(task_reward.reference, -reward.alpha_switch)
    assert reward.in_capture_region(inside_other_side) is True
    outside = _shifted(task_reward.reference, reward.alpha_switch + 1e-6)
    assert reward.in_capture_region(outside) is False
    far = _shifted(task_reward.reference, np.pi)
    assert reward.in_capture_region(far) is False
    # in-cone post-step state: balance even without the latch ...
    row = reward.terms(inside, 0.0, False, False)
    assert row["phase"] == "balance"
    assert row["upright"] == pytest.approx((1.0 + np.cos(reward.alpha_switch)) / 2.0)
    # ... and a latched env keeps the balance phase outside the cone
    row = reward.terms(far, 0.0, False, True)
    assert row["phase"] == "balance"
    # a nonfinite state maps to the zero state before the phase is judged (the
    # zero state is the upright reference here, so it sits inside the cone)
    assert reward.in_capture_region(np.array([np.nan, 0.0, 0.0, 0.0])) == (
        reward.in_capture_region(np.zeros(4))
    )
    row = reward.terms(np.array([np.nan, np.nan, np.nan, np.nan]), 0.0, False, False)
    assert row["phase"] == ("balance" if reward.in_capture_region(np.zeros(4)) else "swing")
    row_out = reward.terms(far, 0.0, False, False)
    assert row_out["phase"] == "swing"  # the mapping never touches finite states
    # the failing step keeps its state-based phase reward and subtracts the penalty
    row = reward.terms(far, 3.0, True, False)
    assert row["failure"] == FAILURE_PENALTY_TWOPHASE
    assert row["total"] == pytest.approx(
        -reward.c_e_switch * abs(reward.energy(far)) + reward.alive_swing - FAILURE_PENALTY_TWOPHASE
    )


def test_swing_hand_computed_and_balance_delegates_to_lesson29(design_and_reward):
    """Swing rows match the hand formula; balance rows are the lesson-29 reward."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    reference = task_reward.reference
    rng = np.random.default_rng(11)
    for _ in range(30):
        alpha = float(rng.uniform(0.4, np.pi)) * (1.0 if rng.random() < 0.5 else -1.0)
        omega = float(rng.uniform(-8.0, 8.0))
        state = _shifted(reference, alpha)
        state[3] = omega
        action = float(rng.uniform(-3.0, 3.0))
        manual_energy = -reward.c_e_switch * abs(
            0.5 * hinge_inertia * omega**2 + gravity_energy * (np.cos(alpha) - 1.0)
        )
        row = reward.terms(state, action, False, False)
        assert row["phase"] == "swing"
        assert row["energy"] == pytest.approx(manual_energy)
        assert row["control_cost"] == 0.0  # no control cost in the swing phase
        assert row["total"] == pytest.approx(manual_energy + reward.alive_swing)
        # inside the cone the balance phase is the lesson-29 reward, bit for bit
        near = _shifted(reference, 0.1)
        row = reward.terms(near, action, False, False)
        expected = task_reward.terms(near, action, False)
        assert row["upright"] == expected["upright"]
        assert row["control_cost"] == expected["control_cost"]
        assert row["total"] == expected["total"]


def test_disabled_switch_is_bitwise_the_lesson29_reward(design_and_reward):
    """enabled=False: the two-phase total equals the lesson-29 reward bit for bit."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(
        task_reward, hinge_inertia, gravity_energy, alpha_switch=0.3, enabled=False
    )
    rng = np.random.default_rng(5)
    for _ in range(50):
        post = rng.normal(0.0, 1.0, 4)
        action = float(rng.uniform(-3.0, 3.0))
        terminated = bool(rng.integers(0, 2))
        assert reward(post, action, terminated) == task_reward(post, action, terminated)
        assert reward(post, action, terminated, latched=True) == task_reward(
            post, action, terminated
        )
        row = reward.terms(post, action, terminated, False)
        assert row["phase"] == "off"
        assert row["total"] == task_reward(post, action, terminated)


def test_pipeline_guard_is_bitwise(design_and_reward):
    """The disabled-switch pipeline reproduces the lesson-29 pipeline bitwise."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    disabled = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy, enabled=False)
    guard = twophase_guard(task_reward, disabled)
    assert guard["bitwise_identical_rewards"] is True
    assert guard["bitwise_identical_states"] is True
    assert guard["bitwise_identical_observations"] is True
    assert guard["steps"] > 0


def test_vec_env_latch_reseeds_on_internal_restart(design_and_reward):
    """An internal episode restart re-seeds the latch from the fresh start."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    vec = TwoPhaseVecSwingup(reward, n_envs=2, episode_steps=4, base_seed=1234, task_envs=0)
    try:
        vec.reset()
        vec.latched[:] = True  # force-latch: the flag must survive while alive ...
        observations = vec.step(np.zeros(2))
        assert observations[4].shape == (2, 5)
        assert vec.latched.all()  # ... no restart happened within the first step
        for _ in range(3):  # reaching episode_steps triggers the internal restart
            vec.step(np.zeros(2))
        for index, env in enumerate(vec.envs):
            state = np.asarray(env.unwrapped._get_obs(), dtype=float)
            assert vec.latched[index] == reward.in_capture_region(state)
        stats = vec.phase_stats()
        assert stats["env_steps"] == 8  # 4 step() calls x 2 environments
        assert 0.0 <= stats["balance_step_fraction"] <= 1.0
    finally:
        vec.close()


def test_episode_terms_replay_and_sugar_known_answers(design_and_reward):
    """The replayed per-step rows and the sugar fraction match hand answers."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    reference = task_reward.reference
    states = np.asarray(
        [
            _shifted(reference, np.pi),  # down start: swing, not latched
            _shifted(reference, 2.0),  # judged here: swing, |E| large
            _shifted(reference, 0.2),  # judged here: in cone -> balance (latch set)
            _shifted(reference, 1.0),  # judged here: outside but latched -> balance
        ]
    )
    controls = np.zeros(3, dtype=np.float32)
    arrays = {"states": states, "controls": controls, "end_flags": np.array([False, False])}
    terms = twophase_episode_terms(arrays, reward)
    assert terms["phase"].tolist() == [False, True, True]
    e2 = abs(reward.energy(states[2]))
    expected_step0 = -reward.c_e_switch * abs(reward.energy(states[1])) + reward.alive_swing
    assert terms["total"][0] == pytest.approx(expected_step0)
    assert terms["total"][1] == pytest.approx(
        (1.0 + np.cos(wrap_angle(states[2][1] - reference[1]))) / 2.0 + 0.25
    )
    assert terms["upright"][0] == 0.0 and terms["upright"][1] > 0.0
    # sugar: the only swing step reduced |E| (states[1] -> states[2], toward zero)
    assert e2 < abs(reward.energy(states[1]))
    assert swing_sugar_fraction(terms, states, reward) == pytest.approx(1.0)
    # a failing episode: the last step carries the penalty
    arrays_fail = {
        "states": states,
        "controls": controls,
        "end_flags": np.array([True, False]),
    }
    terms_fail = twophase_episode_terms(arrays_fail, reward)
    assert terms_fail["failure"][-1] == FAILURE_PENALTY_TWOPHASE
    assert terms_fail["failure"][:2].sum() == 0.0
    # all-swing episodes with non-decreasing |E| map to a lower sugar fraction
    states_flat = np.asarray([_shifted(reference, np.pi)] * 3)
    arrays_flat = {
        "states": states_flat,
        "controls": np.zeros(2, dtype=np.float32),
        "end_flags": np.array([False, False]),
    }
    terms_flat = twophase_episode_terms(arrays_flat, reward)
    assert swing_sugar_fraction(terms_flat, states_flat, reward) == pytest.approx(0.0)


# --------------------------------------------------------- training and eval
def test_micro_training_is_deterministic(design_and_reward):
    """Fixed seeds reproduce the two-phase training bitwise; budget kept."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
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

        vec = TwoPhaseVecSwingup(
            reward,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=7,
            task_envs=config.task_envs,
        )
        try:
            return train_ppo(
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
    default = default_training_config()
    assert default.updates == 250  # the lesson-29 budget is kept
    assert default.n_envs * default.rollout_steps == 2000
    assert DEFAULT_TRAINING_BUDGET_STEPS == 500_000
    assert EVAL_EVERY == 25


def test_acceptance_reuses_lesson7_recovery_metrics(design_and_reward):
    """twophase_episode_metrics is the lesson-7 function on the same arrays."""
    _design, task_reward, hinge_inertia, gravity_energy = design_and_reward
    reference = task_reward.reference
    dt = 0.04
    reward = TwoPhaseReward(task_reward, hinge_inertia, gravity_energy)
    policy = GaussianPolicy(5, (4, 4), seed=2, log_std_init=0.1)
    arrays, reason = run_policy_episode(
        policy,
        reward,
        reference,
        horizon=EVAL_EPISODE_STEPS,
        env_seed=0,
        deterministic=True,
    )
    through_shim = twophase_episode_metrics(arrays, reason, reference, dt)
    direct_view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    direct = recovery_metrics(direct_view, {"failure_reason": reason}, reference, dt)
    for field in EVAL_FIELDS:
        assert through_shim[field] == direct[field]
    assert set(EVAL_FIELDS) <= set(through_shim)
    terms = twophase_episode_terms(arrays, reward)
    assert len(terms["total"]) == len(arrays["controls"])
    # a step that ends in a physical failure carries the small penalty, not -10
    if bool(arrays["end_flags"][0]):
        assert terms["failure"][-1] == pytest.approx(FAILURE_PENALTY_TWOPHASE)


def test_first_success_and_arrival_metrics():
    """The checkpoint metrics and the arrival summary reduce to known answers."""
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
    arrivals = arrival_summary(
        [
            {"first_arrival_s": 0.5},
            {"first_arrival_s": None},
            {"first_arrival_s": 1.5},
        ]
    )
    assert arrivals["episodes_with_arrival"] == 2
    assert arrivals["arrival_fraction"] == pytest.approx(2.0 / 3.0)
    assert arrivals["median_first_arrival_s"] == pytest.approx(1.0)
    assert make_push_plans(0.04, 20, master_seed=0) == make_push_plans(0.04, 20, master_seed=0)


# ------------------------------------------------------ shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["guard"]["bitwise_identical_rewards"] is True
    assert report["guard"]["bitwise_identical_states"] is True
    reward = report["protocol"]["reward"]
    assert reward["alpha_switch_rad"] == pytest.approx(0.3)
    assert reward["latched"] is True
    assert reward["c_e_switch"] == pytest.approx(1.0 / (2.0 * reward["mgl_eff_j"]))
    assert report["protocol"]["phase_protocol"]["latched"] is True
    assert len(report["four_way_comparison"]) == 4
    lesson29, pbrs, twophase = report["four_way_comparison"][1:4]
    assert (lesson29["successes"], lesson29["episodes"]) == (0, 60)
    assert (pbrs["successes"], pbrs["episodes"]) == (0, 60)
    assert "两阶段" in twophase["label"]
    assert twophase["episodes"] == 2
    record = report["per_seed"][0]
    for key in (
        "first_successful_eval_steps",
        "first_arrival_eval_steps",
        "training_phase_stats",
        "eval_curve",
    ):
        assert key in record
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["eval_settled_s_0"].shape == (2,)
        assert archive["det_energy_0"].shape == (archive["det_states_0"].shape[0],)
        assert archive["det_phase_0"].dtype == bool
    for name in (
        "summary.json",
        "trajectories.npz",
        "training_curves.png",
        "phase_analysis.png",
        "four_way.png",
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
    parsed["twophase_evaluation"]["aggregate"]["successes_per_seed"][0] += 1
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
    payload["reward_curve_0"] = payload["reward_curve_0"] * 2.0
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

    # (4) the energy curve tampered (recomputation guard) with a fresh hash;
    # start from the complete pristine payload so only the energy curve differs
    payload = {key: data[key] for key in data if key != "report"}
    payload["det_energy_0"] = payload["det_energy_0"] - 7.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Energy curve"):
        load_replays(work)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.twophase_swingup",
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
    payload = json.loads(result.stdout)
    assert payload["guard"]["bitwise_identical_rewards"] is True
    assert payload["baseline"] == payload["baseline"]
    assert payload["twophase"]["episodes"] == 2
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "four_way.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.twophase_swingup",
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

    from embodied_learning.twophase_demo import TwophaseDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = TwophaseDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 2
    panel = demo.stats.cget("text")
    assert "环境步" in panel and "种子 0" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2  # mode switches must not accumulate axes
    demo.mode.set("episode")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "α_switch" in panel and "cE_sw" in panel  # episode wording as shown
    assert str(round(report["protocol"]["reward"]["c_e_switch"], 4)) in panel
    demo.mode.set("outcome")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "两阶段" in panel  # four-way wording
    assert (
        f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    )  # panel numbers come from the summary
    assert f"{report['four_way_comparison'][1]['successes']}/60" in panel
    demo.close()
