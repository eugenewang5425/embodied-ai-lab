"""Lesson 31: PBRS potential-based reward shaping on the lesson-29 PPO stack.

Unit checks pin the potential and the shaping term against hand-computed
answers (the lesson-7 energy constants, Phi at the down start and at upright),
the gamma*Phi(s') - Phi(s) contract including the bitwise c_e = 0 degeneracy
to the lesson-29 reward and pipeline, the upright first-arrival metric and the
reused lesson-7 acceptance; shrunk end-to-end checks cover the record
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
    CAPTURE_ANGLE_RAD,
    EVAL_EPISODE_STEPS,
    EXPERIMENT,
    GAMMA,
    ShapedReward,
    ShapedVecSwingup,
    annotate_case_c_e,
    arrival_summary,
    default_training_config,
    expected_npz_keys,
    first_arrival_index,
    first_arrival_time_s,
    lesson7_energy_constants,
    pole_energy,
    run_experiment,
    shaped_episode_metrics,
    shaped_step_terms,
    shaping_guard,
    shaping_summary,
)
from embodied_learning.experiments.ppo_swingup import (
    EVAL_FIELDS,
    FAILURE_PENALTY,
    GaussianPolicy,
    PPOConfig,
    RewardFunction,
    make_push_plans,
    push_schedule,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.pbrs_demo import load_replays
from embodied_learning.swingup import (
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
)

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
    "c_e_levels": (0.5, 2.0),
}


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("pbrs_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    hinge_inertia, gravity_energy = lesson7_energy_constants(design)
    reward = RewardFunction(design.controller.reference)
    return design, reward, hinge_inertia, gravity_energy


# ------------------------------------------------- potential and shaping
def test_energy_constants_match_lesson7_controller(design_and_reward):
    """The constants are the lesson-7 controller's own attributes, unchanged."""
    design, reward, hinge_inertia, gravity_energy = design_and_reward
    env = make_swingup_environment(max_episode_steps=2)
    try:
        controller = HybridSwingupController(env.unwrapped.model, design)
    finally:
        env.close()
    assert hinge_inertia == pytest.approx(controller.hinge_inertia)
    assert gravity_energy == pytest.approx(controller.gravity_energy)
    # docs/09: target energy m*g*l ~ 14.770 J, resting down needs ~29.539 J injected
    assert gravity_energy == pytest.approx(14.770, abs=0.01)
    reference = reward.reference
    # alpha = 0 at the reference: E_top is exactly zero in the task frame
    assert pole_energy(reference, reference, hinge_inertia, gravity_energy) == pytest.approx(0.0)
    down = down_state(reference)
    assert pole_energy(down, reference, hinge_inertia, gravity_energy) == pytest.approx(
        -2.0 * gravity_energy
    )


def down_state(reference):
    state = np.asarray(reference, dtype=float).copy()
    state[1] += np.pi
    return state


def test_potential_and_shaping_hand_computed(design_and_reward):
    """Phi = -cE*|E - E_top| on known states; shaping increments by hand."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    reference = reward.reference
    shaped = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=0.5, gamma=GAMMA)
    up, down = reference, down_state(reference)
    assert shaped.potential(up) == pytest.approx(0.0)
    assert shaped.potential(down) == pytest.approx(-0.5 * 2.0 * gravity_energy)
    # one step down -> upright at rest: gamma*0 - (-cE*2mgl) = cE*2mgl > 0
    assert shaped.terms(down, up, 0.0, False)["shaping"] == pytest.approx(
        0.5 * 2.0 * gravity_energy
    )
    # one step upright -> down: gamma*(-cE*2mgl) - 0 < 0
    assert shaped.terms(up, down, 0.0, False)["shaping"] == pytest.approx(
        -GAMMA * 0.5 * 2.0 * gravity_energy
    )
    # hanging still at the bottom: (1-gamma)*cE*2mgl, small but positive
    assert shaped.terms(down, down, 0.0, False)["shaping"] == pytest.approx(
        (1.0 - GAMMA) * 0.5 * 2.0 * gravity_energy
    )
    # the total is the task reward plus the shaping term; the failing step
    # keeps the shaping and replaces only the task part by -failure_penalty
    terms = shaped.terms(down, down, 1.0, True)
    assert terms["task"] == pytest.approx(-FAILURE_PENALTY)
    assert terms["total"] == pytest.approx(
        -FAILURE_PENALTY + (1.0 - GAMMA) * 0.5 * 2.0 * gravity_energy
    )
    with pytest.raises(ValueError):
        ShapedReward(reward, hinge_inertia, gravity_energy, c_e=-0.1)


def test_shaped_reward_zero_ce_is_bitwise_the_lesson29_reward(design_and_reward):
    """c_e = 0: the shaped total equals the lesson-29 reward bit for bit."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    shaped = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=0.0, gamma=GAMMA)
    rng = np.random.default_rng(5)
    for _ in range(50):
        state = rng.normal(0.0, 1.0, 4)
        post = rng.normal(0.0, 1.0, 4)
        action = float(rng.uniform(-3.0, 3.0))
        terminated = bool(rng.integers(0, 2))
        total = shaped(state, post, action, terminated)
        expected = reward(post, action, terminated)
        assert total == expected  # bitwise, not approx


def test_shaping_contract_matches_manual_gamma_phi(design_and_reward):
    """r_total - r_task equals gamma*Phi(s') - Phi(s) on random transitions."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    shaped = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=2.0, gamma=GAMMA)
    rng = np.random.default_rng(9)
    for _ in range(30):
        pre = rng.normal(0.0, 1.0, 4)
        post = rng.normal(0.0, 1.0, 4)
        action = float(rng.uniform(-3.0, 3.0))
        row = shaped.terms(pre, post, action, False)
        manual = GAMMA * shaped.potential(post) - shaped.potential(pre)
        assert row["shaping"] == pytest.approx(manual)
        assert row["total"] == pytest.approx(reward(post, action, False) + manual)
    # nonfinite states map to the zero state before the potential is evaluated
    finite = shaped.potential(np.zeros(4))
    assert shaped.potential(np.array([np.nan, 0.0, 0.0, 0.0])) == pytest.approx(finite)


def test_shaping_guard_is_bitwise_and_annotated_cases(design_and_reward):
    """c_e = 0 through the shaped pipeline equals the lesson-29 pipeline bitwise."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    shaped_zero = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=0.0, gamma=GAMMA)
    guard = shaping_guard(reward, shaped_zero)
    assert guard["bitwise_identical_rewards"] is True
    assert guard["bitwise_identical_states"] is True
    assert guard["bitwise_identical_observations"] is True
    assert guard["steps"] > 0

    # annotate_case_c_e mirrors pick_failure_cases' fixed scan order
    def make_episode(c_e, failed):
        return {
            "c_e": c_e,
            "terminated": failed,
            "recovered": not failed,
            "failure_reason": "cart_safety_boundary" if failed else "",
        }

    cases = [{"kind": "eval_failure"}, {"kind": "push_failure"}]
    eval_episodes = [make_episode(0.5, False), make_episode(0.5, True), make_episode(2.0, True)]
    push_episodes = [make_episode(2.0, False), make_episode(2.0, True)]
    annotate_case_c_e(cases, eval_episodes, push_episodes)
    assert cases[0]["c_e"] == 0.5  # the first failing eval episode
    assert cases[1]["c_e"] == 2.0  # the first failing push episode
    untouched = [{"kind": "eval_failure"}]
    annotate_case_c_e(untouched, eval_episodes, push_episodes)
    assert untouched[0]["c_e"] == 0.5


# ------------------------------------------------- metrics and training
def test_first_arrival_metric_known_answers(design_and_reward):
    """First index/time with |alpha| <= 0.3 rad; None when the region is never reached."""
    _design, reward, _hinge, _mgl = design_and_reward
    reference = reward.reference
    states = np.asarray(
        [
            down_state(reference),  # |alpha| = pi
            _shifted(reference, 1.0),  # outside
            _shifted(reference, 0.2),  # inside -> arrival here
            _shifted(reference, 0.1),
        ]
    )
    assert first_arrival_index(states, reference) == 2
    assert first_arrival_time_s(states, reference, 0.04) == pytest.approx(0.08)
    outside = np.asarray([_shifted(reference, 1.0), _shifted(reference, -1.0)])
    assert first_arrival_index(outside, reference) is None
    assert first_arrival_time_s(outside, reference, 0.04) is None
    boundary = np.asarray([_shifted(reference, CAPTURE_ANGLE_RAD)])
    assert first_arrival_index(boundary, reference) == 0  # the threshold is inclusive


def _shifted(reference, alpha):
    state = np.asarray(reference, dtype=float).copy()
    state[1] += alpha
    return state


def test_micro_training_is_deterministic(design_and_reward):
    """Fixed seeds reproduce the shaped training bitwise; the budget matches lesson 29."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    shaped = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=1.0, gamma=GAMMA)
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

        vec = ShapedVecSwingup(
            reward,
            shaped,
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
    assert default_training_config().updates == 250  # the lesson-29 budget is kept
    assert default_training_config().n_envs * default_training_config().rollout_steps == 2000


def test_acceptance_reuses_lesson7_recovery_metrics(design_and_reward):
    """shaped_episode_metrics is the lesson-7 function on the same arrays."""
    _design, reward, hinge_inertia, gravity_energy = design_and_reward
    reference = reward.reference
    dt = 0.04
    shaped = ShapedReward(reward, hinge_inertia, gravity_energy, c_e=2.0, gamma=GAMMA)
    policy = GaussianPolicy(5, (4, 4), seed=2, log_std_init=0.1)
    from embodied_learning.experiments.pbrs_swingup import run_shaped_episode

    arrays, reason = run_shaped_episode(
        policy,
        shaped,
        reference,
        horizon=EVAL_EPISODE_STEPS,
        env_seed=0,
        deterministic=True,
    )
    through_shim = shaped_episode_metrics(arrays, reason, reference, dt)
    direct_view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    direct = recovery_metrics(direct_view, {"failure_reason": reason}, reference, dt)
    for field in EVAL_FIELDS:
        assert through_shim[field] == direct[field]
    assert set(EVAL_FIELDS) <= set(through_shim)
    terms = shaped_step_terms(arrays, shaped)
    assert len(terms["task"]) == len(arrays["controls"])
    # a step that ends in the actuator boundary has the -10 task reward
    if bool(arrays["end_flags"][0]):
        assert terms["task"][-1] == pytest.approx(-FAILURE_PENALTY)


def test_shaping_and_arrival_summaries_hand_computed():
    """The diagnostic summaries reduce episode lists to known numbers."""
    episodes = [
        {"task_return": 10.0, "shaping_return": 5.0},
        {"task_return": 20.0, "shaping_return": 10.0},
    ]
    shaping = shaping_summary(episodes)
    assert shaping["mean_task_return"] == pytest.approx(15.0)
    assert shaping["mean_shaping_return"] == pytest.approx(7.5)
    assert shaping["shaping_fraction_of_undiscounted_return"] == pytest.approx(7.5 / 22.5)
    arrivals = arrival_summary(
        [
            {"first_arrival_s": 0.5},
            {"first_arrival_s": None},
            {"first_arrival_s": 1.5},
            {"first_arrival_s": None},
        ]
    )
    assert arrivals["episodes_with_arrival"] == 2
    assert arrivals["arrival_fraction"] == pytest.approx(0.5)
    assert arrivals["median_first_arrival_s"] == pytest.approx(1.0)
    never = arrival_summary([{"first_arrival_s": None}])
    assert never["median_first_arrival_s"] is None and never["arrival_fraction"] == 0.0


def test_push_plans_are_paired_with_lesson29():
    """Same seed stream as lesson 29: identical plans for every controller."""
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


# ------------------------------------------------------ shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["training"]["train_seeds"] == 1
    assert [entry["c_e"] for entry in report["sweep"]] == [0.5, 2.0]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["guard"]["bitwise_identical_rewards"] is True
    assert report["guard"]["bitwise_identical_states"] is True
    potential = report["protocol"]["potential"]
    assert potential["hinge_inertia_kg_m2"] > 0.0
    assert potential["e_rest_down_j"] == pytest.approx(-2.0 * potential["mgl_eff_j"])
    for entry in report["sweep"]:
        assert entry["stochastic"]["episodes"] == 2
        assert entry["push"]["episodes"] == 2
        assert len(entry["arrival"]["first_arrival_s_per_episode"]) == 2
        assert "shaping_fraction_of_undiscounted_return" in entry["shaping"]
        for record in entry["training"]:
            assert "first_arrival_eval_steps" in record
            assert all("first_arrival_s" in point for point in record["eval_curve"])
    assert len(report["three_way_comparison"]) == 4
    assert report["three_way_comparison"][1]["successes"] == 0  # cited lesson-29 PPO
    assert report["three_way_comparison"][1]["episodes"] == 60
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["eval_settled_s_0"].shape == (1, 2)
        assert archive["det_phi_0_0"].shape == (archive["det_states_0_0"].shape[0],)
    for name in (
        "summary.json",
        "trajectories.npz",
        "training_curves.png",
        "ladder_analysis.png",
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

    # (4) the potential curve tampered (recomputation guard) with a fresh hash;
    # start from the complete pristine payload so only Phi differs
    payload = {key: data[key] for key in data if key != "report"}
    payload["det_phi_0_0"] = payload["det_phi_0_0"] - 7.0
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Potential curve"):
        load_replays(work)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.pbrs_swingup",
            "--output",
            str(out),
            "--seed",
            "0",
            "--train-seeds",
            "1",
            "--eval-seeds",
            "2",
            "--c-e",
            "2.0",
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
    assert payload["pbrs"]["cE=2"] == payload["pbrs"]["cE=2"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "ladder_analysis.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.pbrs_swingup",
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

    from embodied_learning.pbrs_demo import PbrsDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = PbrsDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 2
    panel = demo.stats.cget("text")
    assert "环境步" in panel and "cE=0.5" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2  # mode switches must not accumulate axes
    demo.mode.set("ladder")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "Φ" in panel and "mgl" in panel  # ladder wording as shown in the panel
    assert str(round(report["protocol"]["potential"]["mgl_eff_j"], 2)) in panel
    demo.mode.set("outcome")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "基线" in panel and "纯 PPO" in panel  # three-way wording
    assert (
        f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    )  # panel numbers come from the summary
    assert (
        f"{report['lesson29_reference']['successes']}/{report['lesson29_reference']['episodes']}"
        in panel
    )
    demo.close()
