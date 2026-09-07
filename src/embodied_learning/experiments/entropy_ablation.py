"""Lesson 41: target-entropy ablation for SAC goal reaching (single variable).

Lesson 39 attributed the arrival-level failure of pure SAC on the differential
car (0/60 vs the manual 20/20) to the maximum-entropy setpoint: with the SAC
standard target entropy -dim(a) = -2 nats the automatic temperature pinned the
policy entropy at exactly -2.000 nats from the first checkpoint, alpha
collapsed to ~0.006, and the policy carried a permanent ~sigma 0.66 per-dim
action noise (~+-4 rad/s wheel noise).  The arrival event (dist < 5 cm AND
speed < 0.1 m/s) was essentially never sampled, the +10 bonus never left a
positive trace in 897,003 sliding-window records, and the learner optimized a
pure -distance stream into a hover solution.

This lesson runs the pre-registered single-variable follow-up.  EVERYTHING is
the lesson-39 configuration field-for-field - the environment, reward, SAC
machinery, evaluation calibers, network sizes, seed streams, budget and the
20 paired goals are IMPORTED from the lesson-39 module, not re-implemented
(the tests pin that reuse by object identity) - and the ONLY moved knob is
the automatic-temperature setpoint:

    control    target entropy -2.0  (the lesson-39 record, NOT re-run)
    arm low    target entropy -0.5  (this run)
    arm vlow   target entropy -0.1  (this run)

Budget: 300k env steps per seed x 3 seeds per arm (the lesson-39 budget), one
gradient update per env step.  A shared master seed makes the network
initialization, warmup actions, training-goal streams, buffer sampling and
eval goals bitwise identical across arms - only the temperature setpoint
differs, so any behavioral divergence is attributable to it.

Metrics: arrival rate (the lesson-39 rule), arrival time, closest approach
distance, final distance, path efficiency, per-episode median speed (the
hover picture), plus the lesson-39 attribution triad: entropy trajectory,
alpha trajectory and the reward record.

Honesty rule: if no lowered arm unlocks arrival within the budget, that null
is the formal conclusion - the bottleneck then moves to value learning /
reward visibility, not to a post-hoc story.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.experiments.rl_goal_reaching import (
    ARRIVAL_BONUS,
    BATCH_SIZE,
    BUFFER_SIZE,
    CRITERION_RATE,
    DT,
    EVAL_EVERY_STEPS,
    EVAL_GOAL_COUNT,
    HIDDEN,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    OUT_OF_BOUNDS_PENALTY,
    SHOWCASE_GOALS,
    TRAIN_SEEDS,
    TRAIN_STEPS,
    UPDATE_EVERY_ENV_STEPS,
    WARMUP_STEPS,
    WHEEL_LIMIT_RAD_S,
    GaussianPolicy2D,
    GoalReachingEnv,
    GoalSACConfig,
    aggregate_episodes,
    default_training_config,
    first_checkpoint,
    make_eval_goals,
    manual_actor,
    policy_actor,
    run_episode,
    train_sac_goal,
)
from embodied_learning.experiments.rl_goal_reaching import EXPERIMENT as LESSON39_EXPERIMENT
from embodied_learning.goal_control import GEOMETRY
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "entropy_ablation_lesson41"
SCHEMA_VERSION = 1

REFERENCE_ENTROPY = -2.0  # the lesson-39 control; lives in that record, never re-run here
ABLATION_TARGET_ENTROPIES = (-0.5, -0.1)  # this run, lower exploration setpoints
DEFAULT_REFERENCE = Path("results/rl_goal_reaching_2026-09-06")

ARM_COLORS = ("#0f766e", "#b45309", "#7c3aed")
REFERENCE_COLOR = "#64748b"


# ------------------------------------------------------------ arm construction
def validate_target_entropies(values):
    """Distinct, finite, strictly negative setpoints; -2 is reference-only.

    The caller's order is preserved (the CLI/test expectation is -0.5 first).
    """
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("At least one target-entropy arm is required")
    for value in values:
        if not np.isfinite(value) or value >= 0.0:
            raise ValueError("Every target entropy must be finite and negative (SAC convention)")
    if len(set(values)) != len(values):
        raise ValueError("Target-entropy arms must be distinct")
    if any(abs(value - REFERENCE_ENTROPY) < 1e-12 for value in values):
        raise ValueError(
            "Target entropy -2 is the lesson-39 control and lives in that record; "
            "run only the lowered arms here"
        )
    return values


def arm_config(base_config, target_entropy):
    """The lesson-39 config with exactly one field moved (the setpoint)."""
    target_entropy = float(target_entropy)
    if not np.isfinite(target_entropy) or target_entropy >= 0.0:
        raise ValueError("target_entropy must be finite and negative")
    return dataclasses.replace(base_config, target_entropy=target_entropy)


def hyperparameter_mismatches(base_config, reference_hyperparameters):
    """Fields where the ablation base config differs from a reference record."""
    mine = asdict_json(base_config)
    mine.pop("target_entropy")
    theirs = dict(reference_hyperparameters or {})
    theirs.pop("target_entropy", None)
    if "hidden" in theirs:
        theirs["hidden"] = list(theirs["hidden"])
    return sorted(name for name in mine if name in theirs and mine[name] != theirs[name])


def asdict_json(config):
    """asdict with tuples listified (JSON-safe round trip)."""
    payload = dataclasses.asdict(config)
    payload["hidden"] = list(config.hidden)
    return payload


# ----------------------------------------------------------- episode analytics
def closest_approach(truth, goal):
    """Minimum true distance to the goal over the recorded pose sequence."""
    truth = np.asarray(truth, dtype=float)
    goal = np.asarray(goal, dtype=float)
    return float(np.min(np.linalg.norm(truth[:, :2] - goal[None, :], axis=1)))


def speed_profile(truth, goal, actor, geometry=GEOMETRY):
    """Median |forward speed| over the actions applied along the truth path.

    The deterministic actor is a pure function of (pose, goal) and the truth
    poses were recorded every step, so the applied wheel commands - and hence
    the env's speed measure |v_x| - can be recomputed exactly.
    """
    truth = np.asarray(truth, dtype=float)
    goal = np.asarray(goal, dtype=float)
    speeds = np.empty(len(truth) - 1, dtype=float)
    for index in range(len(truth) - 1):
        wheels = np.asarray(actor(truth[index], goal), dtype=float)
        speeds[index] = abs(geometry.body_velocity(wheels)[0])
    return float(np.median(speeds))


def episode_profile(truth, goal, actor):
    """Closest approach + hover picture for one recorded episode."""
    return {
        "closest_distance_m": closest_approach(truth, goal),
        "median_speed_m_s": speed_profile(truth, goal, actor),
    }


def extended_aggregate(aggregate, episodes):
    """Lesson-39 aggregate plus the approach/hover summary fields."""
    closests = [episode["closest_distance_m"] for episode in episodes]
    speeds = [episode["median_speed_m_s"] for episode in episodes]
    return {
        **aggregate,
        "median_closest_distance_m": float(np.median(closests)),
        "min_closest_distance_m": float(np.min(closests)),
        "median_episode_speed_m_s": float(np.median(speeds)),
    }


def episode_arrays(episodes):
    """Columnar per-episode fields for the archive (NaN/-1 = not applicable)."""
    return {
        "outcome": np.asarray([episode["outcome"] for episode in episodes], dtype="<U16"),
        "arrival_step": np.asarray(
            [
                round(episode["arrival_time_s"] / DT)
                if episode["arrival_time_s"] is not None
                else -1
                for episode in episodes
            ],
            dtype=int,
        ),
        "closest": np.asarray([episode["closest_distance_m"] for episode in episodes]),
        "median_speed": np.asarray([episode["median_speed_m_s"] for episode in episodes]),
        "final_distance": np.asarray([episode["final_distance_m"] for episode in episodes]),
        "arrival_time": np.asarray(
            [
                episode["arrival_time_s"] if episode["arrival_time_s"] is not None else np.nan
                for episode in episodes
            ]
        ),
    }


# ----------------------------------------------------------- lesson-39 reference
def load_lesson39_reference(path=DEFAULT_REFERENCE):
    """Read the lesson-39 (-2 nats) record; None when the directory is absent.

    The reference supplies the control arm of the comparison (aggregate rows,
    alpha/entropy curves and the closest-approach statistics recomputed from
    the archived truth trajectories).  It is never re-trained here.
    """
    path = Path(path)
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("experiment") != LESSON39_EXPERIMENT:
        raise ValueError(
            f"{summary_path} is a {summary.get('experiment')!r} record, "
            f"expected {LESSON39_EXPERIMENT!r}"
        )
    hyper = dict(summary.get("hyperparameters", {}))
    per_seed = summary.get("rl_evaluation", {}).get("per_seed", [])
    reference = {
        "record_dir": str(path),
        "target_entropy": hyper.get("target_entropy", REFERENCE_ENTROPY),
        "hyperparameters": hyper,
        "baseline_aggregate": summary.get("baseline", {}).get("aggregate"),
        "across_seeds": summary.get("rl_evaluation", {}).get("across_seeds"),
        "per_seed_successes": [record.get("aggregate", {}).get("successes") for record in per_seed],
        "final_alphas": [record.get("final_alpha") for record in per_seed],
        "final_entropies": [record.get("final_policy_entropy") for record in per_seed],
        "eval_goals": None,
        "curves": {},
        "median_closest_distance_m": None,
        "min_closest_distance_m": None,
    }
    npz_path = path / "trajectories.npz"
    if npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as data:
            if "eval_goals" in data.files:
                reference["eval_goals"] = data["eval_goals"].copy()
            for name in ("alpha_curve", "entropy_curve", "reward_curve"):
                curves = [
                    data[f"{name}_{seed}"].copy()
                    for seed in range(len(reference["per_seed_successes"]))
                    if f"{name}_{seed}" in data.files
                ]
                if curves:
                    reference["curves"][name] = curves
            if reference["eval_goals"] is not None and per_seed:
                closests = []
                for seed in range(len(reference["per_seed_successes"])):
                    for goal_index in range(len(reference["eval_goals"])):
                        key = f"eval_truth_{seed}_{goal_index}"
                        if key in data.files:
                            closests.append(
                                closest_approach(data[key], reference["eval_goals"][goal_index])
                            )
                if closests:
                    reference["median_closest_distance_m"] = float(np.median(closests))
                    reference["min_closest_distance_m"] = float(np.min(closests))
    return reference


def reference_for_report(reference, goals_match):
    """JSON-safe projection of the reference (arrays stay out of summary.json)."""
    if reference is None:
        return None
    return {
        "record_dir": reference["record_dir"],
        "target_entropy": reference["target_entropy"],
        "hyperparameters": reference["hyperparameters"],
        "across_seeds": reference["across_seeds"],
        "per_seed_successes": reference["per_seed_successes"],
        "final_alphas": reference["final_alphas"],
        "final_entropies": reference["final_entropies"],
        "median_closest_distance_m": reference["median_closest_distance_m"],
        "min_closest_distance_m": reference["min_closest_distance_m"],
        "curves_available": sorted(reference["curves"]),
        "goals_match_this_run": bool(goals_match),
    }


# --------------------------------------------------------------- record contract
def expected_npz_keys(report):
    """The full archive key set implied by a summary (tamper-evidence)."""
    arms = len(report["arms"])
    seeds = report["training"]["train_seeds"]
    goals = report["training"]["eval_goal_count"]
    curves = ("reward_curve", "critic_loss_curve", "alpha_curve", "entropy_curve")
    episode_fields = (
        "outcome",
        "arrival_step",
        "closest",
        "median_speed",
        "final_distance",
        "arrival_time",
    )
    keys = {"eval_goals"}
    keys.update(f"baseline_{name}" for name in episode_fields)
    keys.update(
        f"{curve}_{arm}_{seed}" for arm in range(arms) for seed in range(seeds) for curve in curves
    )
    keys.update(
        f"eval_{name}_{arm}_{seed}"
        for arm in range(arms)
        for seed in range(seeds)
        for name in ("steps", "successes")
    )
    keys.update(
        f"eval_{name}_{arm}_{seed}"
        for arm in range(arms)
        for seed in range(seeds)
        for name in episode_fields
    )
    keys.update(
        f"eval_truth_{arm}_{seed}_{goal}"
        for arm in range(arms)
        for seed in range(seeds)
        for goal in range(goals)
    )
    return keys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ checkpoints
CHECKPOINT_PREFIX = "_progress_"


def _checkpoint_path(output, arm_index, seed_index):
    """Per-seed checkpoint; deleted once the record is written (clean record)."""
    return Path(output) / f"{CHECKPOINT_PREFIX}arm{arm_index}_seed{seed_index}.npz"


def _write_seed_checkpoint(path, result, arm_sac_config):
    """Persist one finished seed (curves, eval curve, policy weights, meta).

    Each seed's RNG streams are independent of every other seed's, so a
    resumed run completes bitwise identically to an uninterrupted one.
    """
    payload = {f"curve_{name}": curve for name, curve in result["curves"].items()}
    for layer, (weights, biases) in enumerate(
        zip(result["policy"].trunk.weights, result["policy"].trunk.biases, strict=True)
    ):
        payload[f"policy_weight_{layer}"] = weights
        payload[f"policy_bias_{layer}"] = biases
    payload["meta"] = np.asarray(
        json.dumps(
            {
                "config": asdict_json(arm_sac_config),
                "env_steps": result["env_steps"],
                "wall_time_s": result["wall_time_s"],
                "final_alpha": result["final_alpha"],
                "final_policy_entropy": result["final_policy_entropy"],
                "final_reward_mean": result["final_reward_mean"],
                "eval_curve": result["eval_curve"],
            }
        )
    )
    np.savez_compressed(path, **payload)


def _read_seed_checkpoint(path, arm_sac_config):
    """Reload a checkpointed seed; None when absent, corrupt or mismatched."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("config") != asdict_json(arm_sac_config):
                raise ValueError("checkpoint configuration mismatch")
            curves = {
                name: data[f"curve_{name}"].copy()
                for name in ("reward_curve", "critic_loss_curve", "alpha_curve", "entropy_curve")
                if f"curve_{name}" in data.files
            }
            weights, biases = [], []
            layer = 0
            while f"policy_weight_{layer}" in data.files:
                weights.append(data[f"policy_weight_{layer}"].copy())
                biases.append(data[f"policy_bias_{layer}"].copy())
                layer += 1
    except (OSError, ValueError, KeyError):
        return None
    if len(curves) != 4 or not weights:
        return None
    policy = GaussianPolicy2D(OBS_DIM, tuple(meta["config"]["hidden"]), [0])
    for stored_weights, stored_biases, live_weights, live_biases in zip(
        weights, biases, policy.trunk.weights, policy.trunk.biases, strict=True
    ):
        live_weights[...] = stored_weights
        live_biases[...] = stored_biases
    return {
        "policy": policy,
        "curves": curves,
        "eval_curve": meta["eval_curve"],
        "env_steps": meta["env_steps"],
        "wall_time_s": meta["wall_time_s"],
        "final_alpha": meta["final_alpha"],
        "final_policy_entropy": meta["final_policy_entropy"],
        "final_reward_mean": meta["final_reward_mean"],
    }


def load_entropy_record(output):
    """Read a record back and cross-check it (three tamper routes rejected)."""
    output = Path(output)
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    npz_path = output / "trajectories.npz"
    checksum = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if checksum != report.get("trajectories_sha256"):
        raise ValueError(
            "trajectories.npz checksum mismatch: the archive was modified after the record "
            "was written"
        )
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        expected = expected_npz_keys(report)
        if keys != expected:
            raise ValueError(f"Unexpected archive arrays: {sorted(keys ^ expected)}")
        arrays = {key: data[key].copy() for key in keys}
    for arm_index, arm in enumerate(report["arms"]):
        for seed_record in arm["per_seed"]:
            seed_index = seed_record["seed_index"]
            counted = int(np.sum(arrays[f"eval_outcome_{arm_index}_{seed_index}"] == "arrived"))
            if counted != seed_record["aggregate"]["successes"]:
                raise ValueError(
                    f"successes disagree with the archive for arm "
                    f"{arm['target_entropy']} seed {seed_index}"
                )
    baseline_counted = int(np.sum(arrays["baseline_outcome"] == "arrived"))
    if baseline_counted != report["baseline"]["aggregate"]["successes"]:
        raise ValueError("successes disagree with the archive for the manual baseline")
    return {"report": report, "arrays": arrays}


# ------------------------------------------------------------------- report
def single_variable_block(base_config, target_entropies, mismatches):
    common = asdict_json(base_config)
    common.pop("target_entropy")
    return {
        "varied": "target_entropy",
        "route": (
            "the lesson-39 SAC goal-reaching pipeline imported unchanged (env, reward, SAC "
            "machinery, calibers, seed streams, budget, paired goals); the only moved knob per "
            "arm is the automatic-temperature setpoint"
        ),
        "lesson39_record": LESSON39_EXPERIMENT,
        "arms": [
            {
                "target_entropy": REFERENCE_ENTROPY,
                "role": "control reference (lesson-39 record, not re-run)",
            },
            *[
                {
                    "target_entropy": value,
                    "role": "lowered-exploration ablation (this run)",
                }
                for value in target_entropies
            ],
        ],
        "common_hyperparameters": common,
        "config_mismatches_vs_reference": mismatches,
    }


def protocol_block(config):
    """The lesson-39 protocol, field for field, reproduced for the audit."""
    return {
        "identical_to": LESSON39_EXPERIMENT,
        "note": (
            "environment, reward, observation, goals, calibers and training cadence are the "
            "lesson-39 objects imported unchanged; only the temperature setpoint varies by arm"
        ),
        "dt_s": DT,
        "horizon_steps": config.episode_steps,
        "horizon_s": config.episode_steps * DT,
        "action": {
            "space": "(v_l, v_r) wheel speeds in rad/s",
            "limit_rad_s": WHEEL_LIMIT_RAD_S,
            "policy_output": "a in (-1, 1)^2 (tanh-squashed), wheels = 6 * a",
        },
        "observation": {"features": "[x/3, y/3, cos(theta), sin(theta), gx/3, gy/3]"},
        "reward": {
            "distance_term": "-dist(pose, goal)",
            "arrival_bonus": ARRIVAL_BONUS,
            "arrival_rule": "dist < 0.05 m AND forward speed < 0.1 m/s",
            "out_of_bounds_penalty": OUT_OF_BOUNDS_PENALTY,
            "out_of_bounds_rule": "|x| > 3 m or |y| > 3 m",
            "terminal_note": (
                "arrival and out-of-bounds are true terminals; time-limit truncations bootstrap"
            ),
        },
        "goal_sampling": {
            "train": "uniform in [-2.4, 2.4]^2, distance >= 0.5 m from the start, resampled every episode",
            "eval": (
                f"{config.eval_goal_count} fixed goals (showcase #0 {SHOWCASE_GOALS[0]}, "
                f"#1 {SHOWCASE_GOALS[1]} + sampled), the lesson-39 stream (paired)"
            ),
        },
        "method_decisions": {
            "gamma": config.gamma,
            "lr": config.lr,
            "tau": config.tau,
            "alpha_init": config.alpha_init,
            "update_schedule": "one gradient update per env step (the lesson-39 cadence)",
            "warmup_steps": config.warmup_steps,
            "eval_every_steps": config.eval_every_steps,
            "learning_criterion": (
                f"every seed succeeds on >= {CRITERION_RATE:.0%} of the {config.eval_goal_count} "
                "eval goals (the lesson-39 bar, applied per arm)"
            ),
        },
        "seed_streams": {
            "network_init": "default_rng([master, 7000, seed]); identical across arms",
            "action_sampling": "default_rng([master, 5000, seed]); identical across arms",
            "buffer_sampling": "default_rng([master, 9000, seed]); identical across arms",
            "train_goal_stream": "default_rng([master, 4200, seed]); identical across arms",
            "eval_goal_stream": "default_rng([master, 4000]) (the lesson-39 stream)",
        },
    }


def comparison_rows(baseline_aggregate, reference, arm_records):
    subset_fields = (
        "episodes",
        "successes",
        "success_rate",
        "out_of_bounds",
        "median_arrival_time_s",
        "median_path_efficiency",
        "mean_final_distance_m",
        "median_closest_distance_m",
        "min_closest_distance_m",
        "median_episode_speed_m_s",
    )

    def subset(aggregate):
        return {name: aggregate[name] for name in subset_fields if name in aggregate}

    rows = [
        {
            "label": "手工控制器（第 21 课控制器，真值位姿复跑）",
            "target_entropy": None,
            "kind": "baseline",
            **subset(baseline_aggregate),
        }
    ]
    if reference is not None and reference.get("across_seeds"):
        rows.append(
            {
                "label": f"SAC 目标熵 {REFERENCE_ENTROPY:.1f}（第 39 课记录）",
                "target_entropy": REFERENCE_ENTROPY,
                "kind": "reference",
                **subset(reference["across_seeds"]),
                "median_closest_distance_m": reference.get("median_closest_distance_m"),
                "min_closest_distance_m": reference.get("min_closest_distance_m"),
            }
        )
    for record in arm_records:
        rows.append(
            {
                "label": f"SAC 目标熵 {record['target_entropy']:.1f}（本课）",
                "target_entropy": record["target_entropy"],
                "kind": "ablation",
                **subset(record["across_seeds"]),
            }
        )
    return rows


def build_verdict(arm_records, reference, eval_goal_count, criterion_rate=CRITERION_RATE):
    """Pre-registered reading: did lowering the setpoint unlock arrival?"""
    needed = math.ceil(criterion_rate * eval_goal_count)
    per_arm_successes = {
        f"{record['target_entropy']:.1f}": [
            seed_record["aggregate"]["successes"] for seed_record in record["per_seed"]
        ]
        for record in arm_records
    }
    winners = [
        record["target_entropy"]
        for record in arm_records
        if all(
            seed_record["aggregate"]["successes"] >= needed for seed_record in record["per_seed"]
        )
    ]
    best = max(
        arm_records,
        key=lambda record: (record["across_seeds"]["successes"], record["target_entropy"]),
    )
    best_closest = best["across_seeds"]["median_closest_distance_m"]
    reference_closest = (reference or {}).get("median_closest_distance_m")
    closest_improved = (
        None
        if reference_closest is None or best_closest is None
        else bool(best_closest < reference_closest)
    )
    if winners:
        conclusion = "supported"
    else:
        conclusion = "not_supported_within_budget"
    return {
        "claim": (
            "lesson-39 attribution: the arrival-level failure is caused by the target entropy "
            "-2 setpoint (permanent exploration noise); lowering it should let the policy learn "
            "to decelerate into the 5 cm circle"
        ),
        "criterion": (
            f"every seed >= {criterion_rate:.0%} of the {eval_goal_count} eval goals "
            f"(>= {needed}/20 per seed, the lesson-39 bar)"
        ),
        "per_arm_successes": per_arm_successes,
        "arms_meeting_criterion": winners,
        "reference_successes": ((reference or {}).get("across_seeds") or {}).get("successes"),
        "best_arm": {
            "target_entropy": best["target_entropy"],
            "successes": best["across_seeds"]["successes"],
            "episodes": best["across_seeds"]["episodes"],
            "median_closest_distance_m": best_closest,
        },
        "closest_improvement_vs_reference": {
            "reference_median_closest_distance_m": reference_closest,
            "best_median_closest_distance_m": best_closest,
            "improved": closest_improved,
        },
        "conclusion": conclusion,
    }


def build_report(
    *,
    seed,
    base_config,
    target_entropies,
    train_seeds,
    eval_goals,
    baseline_episodes,
    baseline_aggregate,
    arm_records,
    reference,
    goals_match,
    elapsed,
):
    mismatches = hyperparameter_mismatches(base_config, (reference or {}).get("hyperparameters"))
    limitations = [
        (
            "Single-variable by construction: only the temperature setpoint moves.  A null "
            "result here shifts the blame down the stack (value learning / reward visibility), "
            "it does not exonerate every other hyperparameter."
        ),
        (
            "The -2 control arm is the lesson-39 record, not a fresh run: library or hardware "
            "drift between then and now is not controlled (the deterministic seed streams make "
            "this a second-order concern only)."
        ),
        (
            "Pure kinematics, true pose, 20 fixed eval goals per seed - the lesson-39 "
            "information condition and finite sample apply unchanged."
        ),
    ]
    if reference is None:
        limitations.append(
            "The lesson-39 record was not found on disk, so the -2 control row and the alpha/"
            "entropy comparison against it are absent from this record."
        )
    elif not goals_match:
        limitations.append(
            "The eval goals do not match the lesson-39 stream (different master seed or goal "
            "count), so the -2 comparison rows are not goal-paired with this run."
        )
    return {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "numpy_version": np.__version__,
        "single_variable": single_variable_block(base_config, target_entropies, mismatches),
        "protocol": protocol_block(base_config),
        "training": {
            "train_seeds": train_seeds,
            "env_steps_per_seed": base_config.train_steps,
            "eval_goal_count": base_config.eval_goal_count,
            "total_env_steps": base_config.train_steps * train_seeds * len(target_entropies),
            "wall_time_s_total": elapsed,
            "curves_note": (
                "reward/critic_loss/alpha/entropy curves per arm per seed live in trajectories.npz"
            ),
        },
        "baseline": {
            "label": "第 21 课手工控制器（goal_command，真值位姿复跑；训练前闸门）",
            "aggregate": baseline_aggregate,
            "per_goal": baseline_episodes,
        },
        "lesson39_reference": reference_for_report(reference, goals_match),
        "arms": arm_records,
        "comparison": comparison_rows(baseline_aggregate, reference, arm_records),
        "verdict": build_verdict(arm_records, reference, base_config.eval_goal_count),
        "limitations": limitations,
    }


# ---------------------------------------------------------------- experiment
def run_experiment(
    output,
    *,
    seed=0,
    target_entropies=ABLATION_TARGET_ENTROPIES,
    config=None,
    train_seeds=TRAIN_SEEDS,
    reference=DEFAULT_REFERENCE,
    log=print,
    resume=False,
):
    """Two lowered-setpoint arms under the exact lesson-39 protocol.

    resume=True continues an interrupted run: finished seeds are reloaded from
    their per-seed checkpoints (bitwise identical to uninterrupted training)
    and the checkpoints are removed once the record is written.
    """
    output = Path(output)
    if output.exists() and ((output / "summary.json").is_file() or not resume):
        raise FileExistsError(f"Choose a new --output directory: {output}")
    target_entropies = validate_target_entropies(target_entropies)
    if type(train_seeds) is not int or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    base_config = config or default_training_config()

    started = time.perf_counter()
    eval_goals = make_eval_goals(seed, base_config.eval_goal_count)
    eval_env = GoalReachingEnv([seed, 4100])

    reference_data = None if reference is None else load_lesson39_reference(reference)
    goals_match = False
    if (
        reference_data is not None
        and reference_data["eval_goals"] is not None
        and reference_data["eval_goals"].shape == eval_goals.shape
    ):
        goals_match = bool(np.array_equal(reference_data["eval_goals"], eval_goals))

    baseline_episodes = []
    for goal in eval_goals:
        summary, truth = run_episode(manual_actor, eval_env, goal, record=True)
        summary.update(episode_profile(truth, goal, manual_actor))
        baseline_episodes.append(summary)
    baseline_aggregate = extended_aggregate(
        aggregate_episodes(baseline_episodes), baseline_episodes
    )
    if baseline_aggregate["successes"] != len(eval_goals):
        raise ValueError(
            f"manual baseline gate failed: {baseline_aggregate['successes']}/{len(eval_goals)}"
        )
    if log is not None:
        log(
            f"manual baseline gate: {baseline_aggregate['successes']}/"
            f"{baseline_aggregate['episodes']} successes"
        )

    output.mkdir(parents=True, exist_ok=resume)
    arm_records = []
    curve_stores = {}
    episode_stores = {}
    for arm_index, target_entropy in enumerate(target_entropies):
        arm_started = time.perf_counter()
        arm_sac_config = arm_config(base_config, target_entropy)
        needed = math.ceil(CRITERION_RATE * arm_sac_config.eval_goal_count)
        per_seed_records = []
        for seed_index in range(train_seeds):
            checkpoint = _checkpoint_path(output, arm_index, seed_index)
            result = _read_seed_checkpoint(checkpoint, arm_sac_config)
            if result is not None:
                if log is not None:
                    log(f"arm {target_entropy}: seed {seed_index} loaded from checkpoint")
            else:
                train_env = GoalReachingEnv(
                    [seed, 4200, seed_index], episode_steps=arm_sac_config.episode_steps
                )
                result = train_sac_goal(
                    train_env,
                    eval_env,
                    eval_goals,
                    arm_sac_config,
                    master_seed=seed,
                    seed_index=seed_index,
                    log=log,
                )
                _write_seed_checkpoint(checkpoint, result, arm_sac_config)
            policy = result["policy"]
            for name, curve in result["curves"].items():
                curve_stores[f"{name}_{arm_index}_{seed_index}"] = curve
            curve_stores[f"eval_steps_{arm_index}_{seed_index}"] = np.asarray(
                [point["env_steps"] for point in result["eval_curve"]], dtype=int
            )
            curve_stores[f"eval_successes_{arm_index}_{seed_index}"] = np.asarray(
                [point["successes"] for point in result["eval_curve"]], dtype=int
            )

            actor = policy_actor(policy)
            episodes, truths = [], []
            for goal in eval_goals:
                summary, truth = run_episode(actor, eval_env, goal, record=True)
                summary.update(episode_profile(truth, goal, actor))
                episodes.append(summary)
                truths.append(truth)
            for goal_index, truth in enumerate(truths):
                episode_stores[f"eval_truth_{arm_index}_{seed_index}_{goal_index}"] = truth
            for name, values in episode_arrays(episodes).items():
                episode_stores[f"eval_{name}_{arm_index}_{seed_index}"] = values
            aggregate = extended_aggregate(aggregate_episodes(episodes), episodes)
            per_seed_records.append(
                {
                    "seed_index": seed_index,
                    "env_steps": result["env_steps"],
                    "wall_time_s": result["wall_time_s"],
                    "final_alpha": result["final_alpha"],
                    "final_policy_entropy": result["final_policy_entropy"],
                    "final_reward_mean": result["final_reward_mean"],
                    "eval_curve": result["eval_curve"],
                    "first_success_checkpoint_steps": first_checkpoint(
                        result["eval_curve"], lambda point: point["successes"] > 0
                    ),
                    "first_criterion_checkpoint_steps": first_checkpoint(
                        result["eval_curve"],
                        lambda point: (
                            point["successes"] >= math.ceil(CRITERION_RATE * point["episodes"])
                        ),
                    ),
                    "per_goal": [dict(episode) for episode in episodes],
                    "aggregate": aggregate,
                }
            )
            if log is not None:
                log(
                    f"arm {target_entropy}: seed {seed_index} final eval "
                    f"{aggregate['successes']}/{aggregate['episodes']}, median closest "
                    f"{aggregate['median_closest_distance_m']:.3f} m, final alpha "
                    f"{result['final_alpha']:.4f}, wall {result['wall_time_s']:.0f}s"
                )
        all_episodes = [episode for record in per_seed_records for episode in record["per_goal"]]
        arm_records.append(
            {
                "target_entropy": float(target_entropy),
                "train_seeds": train_seeds,
                "env_steps_per_seed": arm_sac_config.train_steps,
                "total_env_steps": arm_sac_config.train_steps * train_seeds,
                "criterion_successes_per_seed": needed,
                "meets_criterion": all(
                    record["aggregate"]["successes"] >= needed for record in per_seed_records
                ),
                "per_seed": per_seed_records,
                "across_seeds": extended_aggregate(aggregate_episodes(all_episodes), all_episodes),
                "arm_wall_time_s": time.perf_counter() - arm_started,
            }
        )

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        base_config=base_config,
        target_entropies=target_entropies,
        train_seeds=train_seeds,
        eval_goals=eval_goals,
        baseline_episodes=baseline_episodes,
        baseline_aggregate=baseline_aggregate,
        arm_records=arm_records,
        reference=reference_data,
        goals_match=goals_match,
        elapsed=elapsed,
    )
    archive = {"eval_goals": np.asarray(eval_goals, dtype=float)}
    archive.update(curve_stores)
    archive.update(episode_stores)
    for name, values in episode_arrays(baseline_episodes).items():
        archive[f"baseline_{name}"] = values
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = digest(output / "trajectories.npz")
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_temperature_comparison(
        output / "temperature_comparison.png", report, output, reference_data
    )
    save_arm_comparison(output / "arm_comparison.png", report)
    for leftover in sorted(output.glob(f"{CHECKPOINT_PREFIX}*.npz")):
        leftover.unlink()  # the finished record carries no checkpoint debris
    return report


# -------------------------------------------------------------------- figures
def _stride(array):
    return max(1, len(array) // 1500)


def save_temperature_comparison(path, report, output, reference):
    """Alpha and policy-entropy trajectories: -2 reference vs the lowered arms."""
    configure_plot_font()
    arms = report["arms"]
    seeds = report["training"]["train_seeds"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        arm_curves = {
            name: [
                [data[f"{name}_{arm}_{seed}"] for seed in range(seeds)] for arm in range(len(arms))
            ]
            for name in ("alpha_curve", "entropy_curve")
        }
    ref_curves = (reference or {}).get("curves", {})

    def draw_series(axis, series, color, label):
        array = np.asarray(series, dtype=float)
        updates = np.arange(1, array.shape[1] + 1)
        stride = _stride(array[0])
        for curve in array:
            axis.plot(updates[::stride], curve[::stride], color=color, alpha=0.3, linewidth=0.7)
        axis.plot(
            updates[::stride],
            array.mean(axis=0)[::stride],
            color=color,
            linewidth=1.8,
            label=label,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    if ref_curves.get("alpha_curve"):
        draw_series(
            axes[0],
            ref_curves["alpha_curve"],
            REFERENCE_COLOR,
            f"目标熵 {REFERENCE_ENTROPY:.1f}（第 39 课记录）",
        )
    for arm_index, arm in enumerate(arms):
        draw_series(
            axes[0],
            arm_curves["alpha_curve"][arm_index],
            ARM_COLORS[arm_index % len(ARM_COLORS)],
            f"目标熵 {arm['target_entropy']:.1f}",
        )
    axes[0].set(
        xlabel="SAC 更新轮次（每环境步 1 次）",
        ylabel="温度 α",
        title="α 轨迹：自动温度对不同目标熵的定价",
    )
    axes[0].legend(fontsize=8)
    if ref_curves.get("entropy_curve"):
        draw_series(
            axes[1],
            ref_curves["entropy_curve"],
            REFERENCE_COLOR,
            f"目标熵 {REFERENCE_ENTROPY:.1f}（第 39 课记录）",
        )
    for arm_index, arm in enumerate(arms):
        color = ARM_COLORS[arm_index % len(ARM_COLORS)]
        draw_series(
            axes[1],
            arm_curves["entropy_curve"][arm_index],
            color,
            f"目标熵 {arm['target_entropy']:.1f}",
        )
        axes[1].axhline(
            arm["target_entropy"], color=color, linestyle="--", linewidth=1.0, alpha=0.8
        )
    if ref_curves.get("entropy_curve"):
        axes[1].axhline(
            REFERENCE_ENTROPY, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, alpha=0.8
        )
    axes[1].set(
        xlabel="SAC 更新轮次",
        ylabel="策略熵（−E[log π]，nats）",
        title="策略熵轨迹：虚线 = 各档目标熵设定点",
    )
    axes[1].legend(fontsize=8, loc="lower right")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_arm_comparison(path, report):
    """Success / arrival time / closest approach / final distance per group."""
    configure_plot_font()
    rows = report["comparison"]
    labels = []
    for row in rows:
        if row["kind"] == "baseline":
            labels.append("手工\n控制器")
        elif row["kind"] == "reference":
            labels.append("SAC\n−2（39课）")
        else:
            labels.append(f"SAC\n{row['target_entropy']:.1f}")
    colors = ["#64748b", *ARM_COLORS[: len(rows) - 1]]
    panels = (
        ("rate", 1.0, "{:.0f}", "成功率（%）", "成功率：到达 5 cm 内且速度 < 0.1"),
        ("median_arrival_time_s", 1.0, "{:.1f} s", "到达时间中位（s）", "到达时间（成功回合中位）"),
        (
            "median_closest_distance_m",
            100.0,
            "{:.1f}",
            "最近接近距离中位（cm）",
            "最近接近距离（越小越接近刹车）",
        ),
        ("mean_final_distance_m", 100.0, "{:.1f}", "终距均值（cm）", "终点距离均值"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0), layout="constrained")
    for axis, (field, scale, template, ylabel, title) in zip(axes, panels, strict=True):
        heights, annotations = [], []
        for row in rows:
            value = row.get(field)
            if field == "rate":
                successes, total = row.get("successes"), row.get("episodes")
                heights.append(100.0 * successes / total if successes is not None else 0.0)
                annotations.append(
                    f"{successes}/{total}" if successes is not None and total else "—"
                )
            else:
                heights.append(value * scale if value is not None else 0.0)
                annotations.append(template.format(value * scale) if value is not None else "—")
        bars = axis.bar(range(len(rows)), heights, color=colors, width=0.55)
        for bar, text in zip(bars, annotations, strict=True):
            axis.annotate(
                text,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels(labels, fontsize=8)
        axis.set(ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 115)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", type=float, nargs="+", default=list(ABLATION_TARGET_ENTROPIES))
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--eval-goals", type=int, default=EVAL_GOAL_COUNT)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY_ENV_STEPS)
    parser.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(HIDDEN))
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted run, reloading finished seeds from checkpoints",
    )
    args = parser.parse_args()
    base_config = GoalSACConfig(
        train_steps=args.train_steps,
        episode_steps=args.episode_steps,
        eval_goal_count=args.eval_goals,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        warmup_steps=args.warmup,
        update_every_env_steps=args.update_every,
        eval_every_steps=args.eval_every_steps,
        hidden=tuple(args.hidden),
    )

    def log(message):
        print(message, file=sys.stderr)  # keep stdout a pure JSON document

    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            target_entropies=tuple(args.arms),
            config=base_config,
            train_seeds=args.train_seeds,
            reference=args.reference,
            log=log,
            resume=args.resume,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["aggregate"],
                "arms": [
                    {
                        "target_entropy": arm["target_entropy"],
                        "meets_criterion": arm["meets_criterion"],
                        "across_seeds": arm["across_seeds"],
                        "per_seed_successes": [
                            record["aggregate"]["successes"] for record in arm["per_seed"]
                        ],
                    }
                    for arm in report["arms"]
                ],
                "verdict": report["verdict"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
