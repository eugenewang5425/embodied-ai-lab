"""Lesson 41: target-entropy ablation for SAC goal reaching (single variable).

Unit checks pin the single-variable contract: the lesson-39 pipeline is
imported (not re-implemented) and only the temperature setpoint moves, the
automatic-temperature step against hand-computed answers, the low-target ->
low-alpha direction inside real micro-training, bitwise micro-training
determinism, the paired lesson-39 eval goals, the closest-approach/speed
recomputation from recorded truths, the shrunk end-to-end record contract,
the verdict logic, the lesson-39 reference loader, the three tamper routes
and the CLI.  No real training is run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.experiments import entropy_ablation as ea
from embodied_learning.experiments import rl_goal_reaching as l39
from embodied_learning.experiments.rl_goal_reaching import (
    ARRIVAL_RADIUS_M,
    GoalReachingEnv,
    GoalSACConfig,
    make_eval_goals,
    run_episode,
    train_sac_goal,
)
from embodied_learning.experiments.sac_swingup import alpha_step
from embodied_learning.goal_control import GEOMETRY

SMALL_BASE = GoalSACConfig(
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


def _write_fake_reference(root):
    """A minimal lesson-39-shaped record for hermetic reference integration."""
    record = root / "lesson39"
    record.mkdir(parents=True)
    summary = {
        "experiment": l39.EXPERIMENT,
        "hyperparameters": {**ea.asdict_json(SMALL_BASE), "target_entropy": -2.0},
        "baseline": {"aggregate": {"episodes": 5, "successes": 5}},
        "rl_evaluation": {
            "across_seeds": {
                "episodes": 10,
                "successes": 0,
                "mean_final_distance_m": 1.0,
                "median_arrival_time_s": None,
            },
            "per_seed": [
                {"aggregate": {"successes": 0}, "final_alpha": 0.006, "final_policy_entropy": -2.0},
                {"aggregate": {"successes": 0}, "final_alpha": 0.007, "final_policy_entropy": -2.0},
            ],
        },
    }
    (record / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end ablation (2 arms x 2 seeds) + a fake -2 reference."""
    root = tmp_path_factory.mktemp("entropy_small")
    reference = _write_fake_reference(root)
    output = root / "run"
    report = ea.run_experiment(
        output,
        seed=0,
        target_entropies=(-0.5, -0.1),
        config=SMALL_BASE,
        train_seeds=2,
        reference=reference,
        log=None,
    )
    return output, report


# --------------------------------------------------------- single-variable contract
def test_pipeline_is_lesson39_imported_and_only_entropy_moves():
    """The env/reward/eval/training objects ARE the lesson-39 objects."""
    assert ea.GoalReachingEnv is l39.GoalReachingEnv
    assert ea.train_sac_goal is l39.train_sac_goal
    assert ea.make_eval_goals is l39.make_eval_goals
    assert ea.run_episode is l39.run_episode
    assert ea.aggregate_episodes is l39.aggregate_episodes
    assert ea.policy_actor is l39.policy_actor
    assert ea.default_training_config is l39.default_training_config
    assert ea.DT == l39.DT
    mine = ea.asdict_json(ea.default_training_config())
    theirs = ea.asdict_json(l39.default_training_config())
    assert mine == theirs  # the base config is the lesson-39 config, setpoint included
    arm = ea.arm_config(ea.default_training_config(), -0.5)
    changed = {name for name in mine if ea.asdict_json(arm)[name] != mine[name]}
    assert changed == {"target_entropy"}
    with pytest.raises(ValueError, match="negative"):
        ea.arm_config(ea.default_training_config(), 0.0)
    with pytest.raises(ValueError, match="lesson-39 control"):
        ea.validate_target_entropies((-2.0,))


# ------------------------------------------------------------ temperature mechanics
def test_alpha_step_hand_computed():
    """alpha += lr * (mean log pi + target): both directions and equilibrium."""
    log_prob = [-1.0, -1.0]  # mean -1, so the MC entropy estimate is 1.0
    assert alpha_step(0.5, log_prob, -2.0, 0.1) == pytest.approx(0.5 + 0.1 * (-1.0 - 2.0))
    assert alpha_step(0.5, log_prob, -2.0, 0.1) == pytest.approx(0.2)
    assert alpha_step(0.5, log_prob, -0.5, 0.1) == pytest.approx(0.35)
    assert alpha_step(0.5, log_prob, -0.1, 0.1) == pytest.approx(0.39)
    # equilibrium: mean log pi equals minus the setpoint (the recorded entropy
    # -E[log pi] then sits exactly AT the target, as lesson 39's -2.000 pin)
    assert alpha_step(0.3, [2.0], -2.0, 0.1) == pytest.approx(0.3)
    assert alpha_step(0.3, [0.5], -0.5, 0.1) == pytest.approx(0.3)
    # the lesson-39 mechanism: same entropy, lower target -> faster collapse
    assert alpha_step(0.5, log_prob, -2.0, 0.1) < alpha_step(0.5, log_prob, -0.5, 0.1)


def test_lower_target_entropy_drives_alpha_down():
    """Same seeds, only the setpoint moved: alpha orders -2 < -0.5 < -0.1."""
    eval_goals = make_eval_goals(0, SMALL_BASE.eval_goal_count)

    def one(target_entropy):
        config = dataclasses.replace(SMALL_BASE, target_entropy=target_entropy)
        train_env = GoalReachingEnv([0, 4200, 0], episode_steps=config.episode_steps)
        eval_env = GoalReachingEnv([0, 4100])
        return train_sac_goal(
            train_env, eval_env, eval_goals, config, master_seed=0, seed_index=0, log=None
        )

    runs = {target: one(target) for target in (-2.0, -0.5, -0.1)}
    tails = {}
    for target, result in runs.items():
        alpha_curve = result["curves"]["alpha_curve"]
        assert len(alpha_curve) == (SMALL_BASE.train_steps - SMALL_BASE.warmup_steps) // 2 + 1
        tails[target] = float(np.mean(alpha_curve[len(alpha_curve) // 2 :]))
    assert tails[-2.0] < tails[-0.5] < tails[-0.1]


def test_micro_training_bitwise_deterministic():
    """One arm twice: identical curves, eval points and weights."""
    config = dataclasses.replace(
        SMALL_BASE,
        train_steps=60,
        episode_steps=25,
        eval_goal_count=4,
        batch_size=16,
        buffer_size=500,
        warmup_steps=20,
        hidden=(16, 16),
        eval_every_steps=30,
    )

    def one():
        train_env = GoalReachingEnv([0, 4200, 0], episode_steps=config.episode_steps)
        eval_env = GoalReachingEnv([0, 4100])
        return train_sac_goal(
            train_env,
            eval_env,
            make_eval_goals(0, 4),
            config,
            master_seed=0,
            seed_index=0,
            log=None,
        )

    first, second = one(), one()
    for name, curve in first["curves"].items():
        np.testing.assert_array_equal(curve, second["curves"][name])
        assert np.isfinite(curve).all()
    for weight_a, weight_b in zip(
        first["policy"].trunk.weights, second["policy"].trunk.weights, strict=True
    ):
        assert np.array_equal(weight_a, weight_b)
    assert first["eval_curve"] == second["eval_curve"]


def test_resume_reuses_checkpoints_and_matches_fresh_run(tmp_path, monkeypatch):
    """An interrupted run resumes bitwise-identically from its checkpoints."""
    tiny = dataclasses.replace(
        SMALL_BASE,
        train_steps=120,
        episode_steps=30,
        eval_goal_count=4,
        batch_size=16,
        buffer_size=500,
        warmup_steps=40,
        hidden=(16, 16),
        eval_every_steps=60,
    )

    def attempt(output, **kwargs):
        ea.run_experiment(
            output,
            seed=0,
            target_entropies=(-0.5,),
            config=tiny,
            train_seeds=2,
            reference=None,
            **kwargs,
        )

    real_train = train_sac_goal
    calls = {"count": 0}

    def flaky(*args, **kwargs):
        result = real_train(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("interrupted mid-run")
        return result

    output = tmp_path / "resumed"
    monkeypatch.setattr(ea, "train_sac_goal", flaky)
    with pytest.raises(RuntimeError, match="interrupted"):
        attempt(output)
    assert (output / "_progress_arm0_seed0.npz").is_file()
    assert not (output / "summary.json").is_file()
    monkeypatch.undo()
    attempt(output, resume=True)
    assert not list(output.glob("_progress_*.npz"))  # no checkpoint debris
    with pytest.raises(FileExistsError):
        attempt(output, resume=True)  # a complete record is never reopened
    fresh = tmp_path / "fresh"
    attempt(fresh)
    with (
        np.load(output / "trajectories.npz", allow_pickle=False) as resumed,
        np.load(fresh / "trajectories.npz", allow_pickle=False) as pristine,
    ):
        assert set(resumed.files) == set(pristine.files)
        for key in resumed.files:
            np.testing.assert_array_equal(resumed[key], pristine[key])
    resumed_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    fresh_summary = json.loads((fresh / "summary.json").read_text(encoding="utf-8"))
    for summary in (resumed_summary, fresh_summary):
        for arm in summary["arms"]:
            arm.pop("arm_wall_time_s")
            for record in arm["per_seed"]:
                record.pop("wall_time_s")
        summary["training"].pop("wall_time_s_total")
        summary.pop("trajectories_sha256")
    assert resumed_summary == fresh_summary


# ------------------------------------------------------------- paired eval surface
def test_eval_goals_paired_with_lesson39():
    """The 20 fixed goals are the lesson-39 stream, showcase goals first."""
    mine, lesson = ea.make_eval_goals(0, 20), l39.make_eval_goals(0, 20)
    np.testing.assert_array_equal(mine, lesson)
    np.testing.assert_allclose(mine[0], [1.6, 0.8])
    np.testing.assert_allclose(mine[1], [-2.2, 1.2])
    assert np.linalg.norm(mine, axis=1).min() >= 0.5
    assert np.abs(mine).max() <= 2.4


def test_episode_profile_matches_truth_recomputation():
    """Closest approach and median speed recompute exactly from the truth path."""
    env = GoalReachingEnv(goal_seed=0)
    goal = np.array([1.6, 0.8])
    summary, truth = run_episode(ea.manual_actor, env, goal, record=True)
    profile = ea.episode_profile(truth, goal, ea.manual_actor)
    distances = np.linalg.norm(truth[:, :2] - goal[None, :], axis=1)
    assert profile["closest_distance_m"] == pytest.approx(float(distances.min()))
    wheels = np.asarray([ea.manual_actor(truth[index], goal) for index in range(len(truth) - 1)])
    speeds = np.abs([GEOMETRY.body_velocity(wheel)[0] for wheel in wheels])
    assert profile["median_speed_m_s"] == pytest.approx(float(np.median(speeds)))
    assert summary["outcome"] == "arrived"
    assert profile["closest_distance_m"] <= summary["final_distance_m"] + 1e-12
    assert profile["closest_distance_m"] < ARRIVAL_RADIUS_M


# --------------------------------------------------------------- shrunk end-to-end
@pytest.mark.slow
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == ea.EXPERIMENT
    assert [arm["target_entropy"] for arm in report["arms"]] == [-0.5, -0.1]
    assert report["single_variable"]["config_mismatches_vs_reference"] == []
    assert len(report["comparison"]) == 4
    assert report["comparison"][0]["kind"] == "baseline"
    assert report["comparison"][1]["kind"] == "reference"
    assert report["comparison"][1]["target_entropy"] == -2.0
    for arm in report["arms"]:
        assert len(arm["per_seed"]) == 2
        assert arm["across_seeds"]["episodes"] == 10
        assert "median_closest_distance_m" in arm["across_seeds"]
    assert report["verdict"]["reference_successes"] == 0
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == ea.expected_npz_keys(report)
        assert archive["eval_goals"].shape == (5, 2)
        assert archive["eval_truth_0_0_0"].shape[1] == 3
        assert archive["eval_outcome_0_0"].shape == (5,)
    for name in (
        "summary.json",
        "trajectories.npz",
        "temperature_comparison.png",
        "arm_comparison.png",
    ):
        assert (output / name).is_file() and (output / name).stat().st_size > 0
    assert not list(output.glob("_progress_*.npz"))  # finished records are clean
    with pytest.raises(FileExistsError):
        ea.run_experiment(
            output,
            seed=0,
            target_entropies=(-0.5, -0.1),
            config=SMALL_BASE,
            train_seeds=2,
            log=None,
        )


@pytest.mark.slow
def test_arm_aggregates_and_verdict_consistent(small_run):
    _output, report = small_run
    for arm in report["arms"]:
        needed = math.ceil(0.9 * SMALL_BASE.eval_goal_count)
        assert arm["criterion_successes_per_seed"] == needed
        successes = [record["aggregate"]["successes"] for record in arm["per_seed"]]
        assert arm["meets_criterion"] == all(value >= needed for value in successes)
        for record in arm["per_seed"]:
            assert record["aggregate"]["successes"] == sum(
                episode["outcome"] == "arrived" for episode in record["per_goal"]
            )
            closests = [episode["closest_distance_m"] for episode in record["per_goal"]]
            assert record["aggregate"]["median_closest_distance_m"] == pytest.approx(
                float(np.median(closests))
            )
    verdict = report["verdict"]
    assert verdict["per_arm_successes"] == {
        "-0.5": [record["aggregate"]["successes"] for record in report["arms"][0]["per_seed"]],
        "-0.1": [record["aggregate"]["successes"] for record in report["arms"][1]["per_seed"]],
    }
    assert verdict["conclusion"] in ("supported", "not_supported_within_budget")
    assert (verdict["conclusion"] == "supported") == bool(verdict["arms_meeting_criterion"])


def test_verdict_logic_branches():
    """Criterion met -> supported; else the null is recorded with raw numbers."""

    def arm(target, seed_successes, closest):
        return {
            "target_entropy": target,
            "per_seed": [{"aggregate": {"successes": value}} for value in seed_successes],
            "across_seeds": {
                "successes": sum(seed_successes),
                "episodes": 20 * len(seed_successes),
                "median_closest_distance_m": closest,
            },
        }

    reference = {
        "across_seeds": {"successes": 0},
        "median_closest_distance_m": 0.6,
    }
    supported = ea.build_verdict([arm(-0.5, [20, 18], 0.03), arm(-0.1, [0, 1], 0.5)], reference, 20)
    assert supported["conclusion"] == "supported"
    assert supported["arms_meeting_criterion"] == [-0.5]
    assert (
        supported["criterion"]
        == "every seed >= 90% of the 20 eval goals (>= 18/20 per seed, the lesson-39 bar)"
    )

    refuted = ea.build_verdict([arm(-0.5, [0, 0], 0.7)], reference, 20)
    assert refuted["conclusion"] == "not_supported_within_budget"
    assert refuted["best_arm"]["target_entropy"] == -0.5
    assert refuted["closest_improvement_vs_reference"]["improved"] is False
    assert refuted["reference_successes"] == 0
    assert (
        ea.build_verdict([arm(-0.5, [0, 0], 0.5)], reference, 20)[
            "closest_improvement_vs_reference"
        ]["improved"]
        is True
    )


# ---------------------------------------------------------------- reference loader
def test_lesson39_reference_loader(tmp_path):
    assert ea.load_lesson39_reference(tmp_path / "nothing") is None
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    (wrong / "summary.json").write_text(
        json.dumps({"experiment": "something_else"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected"):
        ea.load_lesson39_reference(wrong)
    real = ea.load_lesson39_reference()  # the shipped lesson-39 record, repo-relative
    if real is not None:
        assert real["target_entropy"] == -2.0
        assert real["per_seed_successes"] == [0, 0, 0]
        assert (
            ea.hyperparameter_mismatches(l39.default_training_config(), real["hyperparameters"])
            == []
        )
        assert real["eval_goals"] is not None and real["eval_goals"].shape == (20, 2)
        assert real["median_closest_distance_m"] > ARRIVAL_RADIUS_M
        assert set(real["curves"]) == {"alpha_curve", "entropy_curve", "reward_curve"}
        assert len(real["curves"]["alpha_curve"]) == 3


# ----------------------------------------------------------------- tamper rejection
@pytest.mark.slow
def test_record_loader_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    pristine = ea.load_entropy_record(output)
    assert pristine["report"]["experiment"] == ea.EXPERIMENT

    # (1) summary successes disagree with the archived outcomes
    work = tmp_path / "tampered_successes"
    shutil.copytree(output, work)
    parsed = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    parsed["arms"][0]["per_seed"][0]["aggregate"]["successes"] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="successes disagree"):
        ea.load_entropy_record(work)

    # (2) archive bytes modified -> checksum mismatch
    work = tmp_path / "tampered_checksum"
    shutil.copytree(output, work)
    path = work / "trajectories.npz"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="checksum"):
        ea.load_entropy_record(work)

    # (3) an array removed with the hash refreshed -> unexpected key set
    work = tmp_path / "tampered_keys"
    shutil.copytree(output, work)
    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    del payload["baseline_outcome"]
    np.savez_compressed(path, **payload)
    parsed = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        ea.load_entropy_record(work)


# ------------------------------------------------------------------------------ CLI
@pytest.mark.slow
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.entropy_ablation",
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
            "--reference",
            str(tmp_path / "absent"),
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
    assert [arm["target_entropy"] for arm in payload["arms"]] == [-0.5, -0.1]
    assert payload["baseline"]["episodes"] == 4
    assert payload["verdict"]["conclusion"] in ("supported", "not_supported_within_budget")
    for name in ("summary.json", "trajectories.npz", "temperature_comparison.png"):
        assert (out / name).is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.entropy_ablation",
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
    bad_arm = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.entropy_ablation",
            "--output",
            str(tmp_path / "bad_arm"),
            "--arms",
            "0.5",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert bad_arm.returncode != 0
    assert "negative" in bad_arm.stderr
