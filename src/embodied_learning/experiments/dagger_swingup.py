"""Lesson 36: DAgger online correction - the teacher corrects the closed-loop policy.

Lesson 32's airdrop demonstrated DAPG-style imitation: the lesson-7 teacher's
offline rollouts anchored the policy in the upright region (33/60 arrivals) but
the open-loop BC regression (9-15 N of mean error) could not deliver the
+/-0.01 rad/s closed-loop invariant the acceptance needs - BC minimises a
per-step residual, not the closed-loop settled tail. This lesson is DAgger
(Ross, Gordon & Bagnell, RSS 2010): instead of the teacher's own trajectories,
each round labels the STATES THE STUDENT REACHED under its current policy and
the student is trained on the aggregated (student-state, teacher-action)
dataset - "the data follows the policy", which is exactly the compound-error
fix that offline BC lacks.

The loop (per tier, per training seed):

    1. roll out N = 8 episodes from the exact resting down start + the lesson-32
       jitter with the CURRENT stochastic student policy, recording the state
       sequence s_t and the student action a_t (the on-policy PPO batch);
    2. the teacher (lesson-7 HybridSwingupController, re-verified 20/20 under
       the annotation quality gate) labels every recorded state: a_teacher(s_t).
       One controller instance per episode replays the recorded state sequence,
       so its internal hysteresis mode (kick/swingup/balance) evolves exactly as
       it would on-line. Non-finite states/labels are dropped and counted, and a
       re-annotation of the first episode per round must be bitwise identical;
    3. D_r = D_{r-1} U {(s_t, a_teacher(s_t))} - data aggregation;
    4. objective: L = L_PPO + w_BC * MSE(mean_theta(s), a_teacher) with w_BC
       constant inside a round and annealed linearly across the rounds from
       w_init (round 0) down to w_min = 0 (round T-1); the student's own action
       a_t enters only the PPO term;
    5. evaluation at the end of every round: eval_seed_count stochastic episodes
       from the exact down start under the verbatim lesson-7 recovery_metrics,
       plus one mean-action episode for the figures; the fresh policy is also
       evaluated before round 0 so the evolution curve starts at a true zero.

Two tiers run the identical loop: w_BC = 10.0 (DAgger correction) and
w_BC = 0.0 (pure on-line policy-gradient fine-tuning - the mixed-strategy
control that isolates "does the correction data matter, or is any on-line
fine-tuning enough?"). The annotation is skipped for the control tier (its
objective has no BC term) and recorded as such.

The PPO stack (policy/value nets, GAE, clipping, hand Adam, seed streams) is
the lesson-29 stack verbatim; the BC term is the lesson-32 objective reused on
the aggregated online dataset. No file outside this lesson is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.dapg_swingup import (
    DEMO_ANGLE_JITTER,
    DEMO_CART_JITTER,
    DEMO_OMEGA_JITTER,
    bc_loss_and_gradient,
    deterministic_dapg_episode,
    first_arrival_time_s,
)
from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    EVAL_SEEDS,
    SEED_OFFSET_INIT,
    STATE_INPUTS,
    TRAIN_SEEDS,
    GaussianPolicy,
    MLPTower,
    PPOConfig,
    RewardFunction,
    baseline_evaluations,
    clip_gradients_,
    down_start_state,
    evaluate_policy,
    failure_counts,
    normalize_observation,
    policy_array_names,
    ppo_losses_and_gradients,
    standardize,
    summarize_episodes,
)
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
)

EXPERIMENT = "dagger_swingup_lesson36"
SCHEMA_VERSION = 1

# ------------------------------------------------------------ DAgger protocol
W_BC_LEVELS = (10.0, 0.0)  # DAgger tier vs pure on-line PG fine-tuning control
W_BC_MIN = 0.0
ROUNDS = 6  # T
ANNOTATE_ROLLOUTS = 8  # N episodes per round (parallel collectors)
UPDATES_PER_ROUND = 15
EPOCHS_FOR_DAGGER = 1  # one epoch of minibatches per update (the batch is small)
MINIBATCH = 480
BC_MINIBATCH = 256
ROLLOUT_HORIZON = EVAL_EPISODE_STEPS  # 30 s, the teacher rollout horizon
# The lesson-32 demo jitter is reused verbatim so the annotation-start
# distribution matches the demonstration convention (the student policy is
# stochastic, so episodes are distinct even without it; recorded, not hidden).
JITTER_ANGLE = DEMO_ANGLE_JITTER  # 0.15 rad
JITTER_OMEGA = DEMO_OMEGA_JITTER  # 0.3 rad/s
JITTER_CART = DEMO_CART_JITTER  # 0.10 m
LABEL_BINS = np.linspace(-CONTROL_LIMIT, CONTROL_LIMIT, 13)  # teacher action hist

SEED_OFFSET_ACT = 5000  # rollout action sampling (lesson 29 convention)
SEED_OFFSET_SHUFFLE = 9000  # minibatch order
SEED_OFFSET_BC = 4500  # BC dataset minibatch sampling
SEED_OFFSET_ROLL_JITTER = 6100  # annotation start jitter
BASE_ENV = 10_000  # base env stream: 10000 + master*1000 + level*100 + seed_index

SHORT_FAILURE = {
    "cart_safety_boundary": "出界",
    "velocity_safety_boundary": "超速",
    "timeout_without_settling": "超时未稳",
    "nonfinite_state": "数值发散",
    "numerical_warning": "数值警告",
}

# Context numbers imported verbatim from the official records (docs/33, docs/37,
# docs/39, docs/40); used only in the comparison table.
LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "first_arrival": "never (docs/33 section 3.4, mechanism 1)",
}
LESSON32_DAPG_REFERENCE = {
    "source": "results/dapg_swingup_2026-09-06 (official lesson-32 record, docs/37)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "first_arrival": "w=10: 33/60 (median 2.12 s); w=1: 27/60 (median 2.04 s)",
}
LESSON34_TWOPHASE_REFERENCE = {
    "source": "results/twophase_swingup_2026-09-06 (official lesson-34 record, docs/39)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "first_arrival": "2/3 seeds (100k / 200k checkpoints)",
}
LESSON35_SAC_REFERENCE = {
    "source": "results/sac_swingup_2026-09-06 (official lesson-35 record, docs/40)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "first_arrival": "0/60 (alpha auto and fixed); curriculum group 5/60",
}


def failure_label(case):
    """Short Chinese failure tag for figure titles (long English keys overlap panels)."""
    reason = case.get("failure_reason") or ""
    return SHORT_FAILURE.get(reason, reason or "未达标")


def tier_label(w_bc):
    return f"w={w_bc:g}"


def tier_short_label(w_bc):
    return "DAgger" if w_bc > 0.0 else "纯PG微调"


# ------------------------------------------------------------- w_BC schedule
def bc_weight_at(round_index, rounds, w_init, w_min=W_BC_MIN):
    """Step schedule: constant inside a round, linear across the rounds.

    Round 0 carries w_init, round rounds-1 carries w_min (the lesson-32
    annealing convention quantised onto the round grid: the anchor holds for the
    first corrections and hands over to the pure policy gradient at the end).
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if w_init < 0.0 or w_min < 0.0 or w_min > w_init:
        raise ValueError("BC weights must satisfy 0 <= w_min <= w_init")
    if not 0 <= round_index < rounds:
        raise ValueError("round_index out of range")
    if rounds == 1:
        return float(w_init)
    fraction = round_index / (rounds - 1)
    return float(w_init + (w_min - w_init) * fraction)


# --------------------------------------------------------------- the objective
def dagger_losses_and_gradients(
    policy, value, obs, actions, old_logp, advantages, returns, config, w_bc, agg_obs, agg_actions
):
    """Lesson-29 PPO losses/gradients plus the annealed BC regularizer.

    The aggregated dataset is the DAgger cloud: student-own states labeled by
    the teacher. With w_bc = 0 the gradients are the lesson-29 gradients bit
    for bit (the contract lesson 32 pinned for its w_BC = 0 guard).
    """
    losses, gradients = ppo_losses_and_gradients(
        policy, value, obs, actions, old_logp, advantages, returns, config
    )
    bc_mse, bc_grads = bc_loss_and_gradient(policy, agg_obs, agg_actions)
    if not all(np.isfinite(g).all() for g in bc_grads) or not np.isfinite(bc_mse):
        raise ValueError("Nonfinite BC loss or gradients; training diverged")
    if w_bc > 0.0:
        n_policy_arrays = 2 * (len(config.hidden) + 1)
        for index in range(n_policy_arrays):
            gradients[index] = gradients[index] + w_bc * bc_grads[index]
        losses["total"] = losses["total"] + w_bc * bc_mse
    losses["bc"] = bc_mse
    if not np.isfinite(losses["total"]):
        raise ValueError("Nonfinite DAgger loss; training diverged")
    return losses, gradients


# ---------------------------------------------------------- rollout machinery
def compute_gae_episodes(episodes, gamma, gae_lambda):
    """GAE(lambda) over variable-length episodes with per-step bootstrap values.

    Each episode carries, aligned by step: values (V of the pre-action state),
    rewards, terminated, truncated and terminal_values (V of the post-step
    state, the bootstrap at a TimeLimit truncation and at the segment end; a
    true termination contributes no bootstrap through the nonterminal mask).
    """
    advantages = []
    for ep in episodes:
        values = ep["values"]
        rewards = ep["rewards"]
        terminated = ep["terminated"]
        truncated = ep["truncated"]
        terminal_values = ep["terminal_values"]
        length = len(values)
        adv = np.zeros(length)
        running = 0.0
        for step in reversed(range(length)):
            if truncated[step] or step == length - 1:
                next_value = terminal_values[step]
            else:
                next_value = values[step + 1]
            nonterminal = 1.0 - terminated[step]
            delta = rewards[step] + gamma * next_value * nonterminal - values[step]
            running = delta + gamma * gae_lambda * nonterminal * running
            adv[step] = running
            boundary = terminated[step] or truncated[step]
            running = 0.0 if boundary else running
        advantages.append(adv)
    return advantages


def rollout_to_batch(episodes, *, gamma, gae_lambda, reward_scale):
    """Flatten per-episode record holders into one on-policy batch with GAE.

    The GAE recursion runs on the reward-scaled transitions (value targets in
    the scaled units, lesson-29 convention); the batch keeps the raw rewards
    for reporting and the scaled advantages for the objective.
    """
    obs = np.concatenate([ep["obs"] for ep in episodes], axis=0)
    actions = np.concatenate([ep["actions"] for ep in episodes])
    logp = np.concatenate([ep["logp"] for ep in episodes])
    values = np.concatenate([ep["values"] for ep in episodes])
    rewards = np.concatenate([ep["rewards"] for ep in episodes])
    scaled_episodes = [
        {
            "values": ep["values"],
            "rewards": ep["rewards"] * reward_scale,
            "terminated": ep["terminated"],
            "truncated": ep["truncated"],
            "terminal_values": ep["terminal_values"],
        }
        for ep in episodes
    ]
    advantages = np.concatenate(compute_gae_episodes(scaled_episodes, gamma, gae_lambda))
    batch = {
        "obs": obs,
        "actions": actions,
        "logp": logp,
        "values": values,
        "rewards": rewards,
        "advantages": standardize(advantages),
        "returns": advantages + values,
    }
    return batch, float(np.mean(rewards))


def collect_student_rollouts(
    envs,
    policy,
    value,
    reward,
    reference,
    *,
    master_seed,
    level_index,
    seed_index,
    round_index,
    horizon,
    action_rng,
):
    """N parallel student episodes from the exact down start + lesson-32 jitter.

    The raw state sequence (lesson-7 alignment: states[k] precedes action k) and
    the student's own action a_t are kept per episode: the states feed the
    teacher annotation, the actions feed the PPO term.
    """
    n_envs = len(envs)
    base_seed = BASE_ENV + master_seed * 1000 + level_index * 100 + seed_index
    starts = []
    for index, env in enumerate(envs):
        seed_rng = np.random.default_rng([base_seed, SEED_OFFSET_ROLL_JITTER, round_index, index])
        start = down_start_state(reference)
        start[0] += float(seed_rng.uniform(-JITTER_CART, JITTER_CART))
        start[1] += float(seed_rng.uniform(-JITTER_ANGLE, JITTER_ANGLE))
        start[3] += float(seed_rng.uniform(-JITTER_OMEGA, JITTER_OMEGA))
        env.unwrapped.set_state(start[:2], start[2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        starts.append(start)
    episodes = [
        {
            "start_state": start,
            "states": [start.copy()],
            "obs": [],
            "actions": [],
            "logp": [],
            "values": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "terminal_values": [],
        }
        for start in starts
    ]
    states = [start.copy() for start in starts]
    alive = np.ones(n_envs, dtype=bool)
    for _step in range(horizon):
        if not alive.any():
            break
        for index, env in enumerate(envs):
            if not alive[index]:
                continue
            obs = normalize_observation(states[index], reference)[None, :]
            action, logp = policy.sample(obs, action_rng)
            val = float(value.forward(obs)[0][0, 0])
            command = np.array([np.clip(action[0], -CONTROL_LIMIT, CONTROL_LIMIT)], np.float32)
            next_state, _raw_reward, terminated, truncated, _info = env.step(command)
            terminal = bool(terminated)
            truncated_flag = bool(truncated)
            term_val = float(
                value.forward(normalize_observation(next_state, reference)[None, :])[0][0, 0]
            )
            ep = episodes[index]
            ep["states"].append(np.asarray(next_state, dtype=float).copy())
            ep["obs"].append(obs[0])
            ep["actions"].append(float(action[0]))
            ep["logp"].append(float(logp[0]))
            ep["values"].append(val)
            ep["rewards"].append(float(reward(next_state, float(command[0]), terminal)))
            ep["terminated"].append(terminal)
            ep["truncated"].append(truncated_flag)
            ep["terminal_values"].append(term_val)
            if terminal or truncated_flag:
                alive[index] = False
            else:
                states[index] = next_state
    for ep in episodes:
        ep["obs"] = np.asarray(ep["obs"], dtype=float)
        ep["actions"] = np.asarray(ep["actions"], dtype=float)
        ep["logp"] = np.asarray(ep["logp"], dtype=float)
        ep["values"] = np.asarray(ep["values"], dtype=float)
        ep["rewards"] = np.asarray(ep["rewards"], dtype=float)
        ep["terminated"] = np.asarray(ep["terminated"], dtype=bool)
        ep["truncated"] = np.asarray(ep["truncated"], dtype=bool)
        ep["terminal_values"] = np.asarray(ep["terminal_values"], dtype=float)
        ep["states"] = np.asarray(ep["states"], dtype=float)
    return episodes


# ----------------------------------------------------------- teacher labeling
def annotate_episode(controller, episode, reference):
    """Label every recorded pre-action state with the teacher's action.

    One controller instance per episode: the teacher's mode hysteresis
    (kick / swingup / balance) runs along the state sequence it labels, exactly
    as it would have on-line. Non-finite pairs are counted as dropped.
    """
    states = episode["states"]
    obs_list, labels = [], []
    dropped = 0
    for state in states[:-1]:
        obs = normalize_observation(state, reference)
        try:
            label = float(controller.action(state)[0])
        except (ValueError, FloatingPointError):
            dropped += 1
            continue
        if not np.isfinite(obs).all() or not np.isfinite(label):
            dropped += 1
            continue
        obs_list.append(obs)
        labels.append(label)
    return np.asarray(obs_list, dtype=float), np.asarray(labels, dtype=float), dropped


def annotate_rollouts(envs, design, episodes, reference):
    """Teacher labels for one round of student rollouts + quality bookkeeping.

    Returns the per-episode (obs, label) pair lists, the total dropped count and
    a self-consistency flag: annotating the first episode twice must yield
    bitwise identical labels (the teacher is deterministic).
    """
    model = envs[0].unwrapped.model
    pairs = []
    dropped_total = 0
    consistency = True
    for index, episode in enumerate(episodes):
        controller = HybridSwingupController(model, design)
        obs, labels, dropped = annotate_episode(controller, episode, reference)
        dropped_total += dropped
        pairs.append((obs, labels))
        if index == 0:
            replay = HybridSwingupController(model, design)
            obs2, labels2, _ = annotate_episode(replay, episode, reference)
            consistency = (
                labels.shape == labels2.shape
                and np.array_equal(labels, labels2)
                and np.array_equal(obs, obs2)
            )
    return pairs, dropped_total, bool(consistency)


def filter_annotation_pairs(obs, labels):
    """Drop non-finite pairs (and count them) before they enter the dataset."""
    obs = np.asarray(obs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if obs.shape[0] != labels.shape[0]:
        raise ValueError("observation and label arrays must align")
    if obs.ndim != 2 or obs.shape[1] != STATE_INPUTS:
        raise ValueError("observations must be a (N, 5) batch")
    finite = np.isfinite(obs).all(axis=1) & np.isfinite(labels)
    dropped = int((~finite).sum())
    return obs[finite], labels[finite], dropped


# -------------------------------------------------------------- training loop
def train_dagger_round(
    policy,
    value,
    optimizer,
    parameters,
    batch,
    agg_obs,
    agg_actions,
    *,
    config,
    w_bc,
    update_index,
    total_updates,
    mb_rng,
    bc_rng,
):
    """One round of PPO + annealed-BC training on the aggregated dataset."""
    size = len(batch["obs"])
    minibatch = min(MINIBATCH, size)
    order = mb_rng.permutation(size)
    value_loss_sum, clip_sum, bc_sum, total_sum, grad_steps = 0.0, 0.0, 0.0, 0.0, 0
    for _epoch in range(EPOCHS_FOR_DAGGER):
        for start in range(0, size, minibatch):
            idx = order[start : start + minibatch]
            if len(agg_obs):
                bc_idx = bc_rng.integers(0, len(agg_obs), size=min(BC_MINIBATCH, len(agg_obs)))
                agg_obs_batch, agg_actions_batch = agg_obs[bc_idx], agg_actions[bc_idx]
            else:
                agg_obs_batch = np.zeros((1, STATE_INPUTS))
                agg_actions_batch = np.zeros(1)
            losses, gradients = dagger_losses_and_gradients(
                policy,
                value,
                batch["obs"][idx],
                batch["actions"][idx],
                batch["logp"][idx],
                batch["advantages"][idx],
                batch["returns"][idx],
                config,
                w_bc,
                agg_obs_batch,
                agg_actions_batch,
            )
            clip_gradients_(gradients, config.grad_clip)
            optimizer.step(parameters, gradients)
            optimizer.lr = config.lr * (1.0 - (update_index + 1) / total_updates)
            value_loss_sum += losses["value"]
            clip_sum += losses["clip_fraction"]
            bc_sum += losses["bc"]
            total_sum += losses["total"]
            grad_steps += 1
        update_index += 1
    return {
        "loss_mean": total_sum / grad_steps,
        "value_loss_mean": value_loss_sum / grad_steps,
        "clip_fraction_mean": clip_sum / grad_steps,
        "bc_mean": bc_sum / grad_steps,
        "grad_steps": grad_steps,
    }


# ----------------------------------------------------------------- evaluation
def evaluate_dagger_round(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start + upright arrivals."""
    episodes = evaluate_policy(policy, reward, reference, dt, master_seed=master_seed, count=count)
    for episode in episodes:
        episode["first_arrival_s"] = first_arrival_time_s(
            episode["arrays"]["states"], reference, dt
        )
    return episodes


def eval_episode_summary(episodes):
    """Lesson-7 acceptance + upright arrival aggregate over one eval batch."""
    summary = summarize_episodes(episodes)
    arrivals = [ep["first_arrival_s"] for ep in episodes]
    got = [value for value in arrivals if value is not None]
    summary["arrival"] = {
        "episodes_with_arrival": len(got),
        "arrival_fraction": len(got) / len(episodes) if episodes else 0.0,
        "median_first_arrival_s": float(np.median(got)) if got else None,
    }
    summary["first_arrival_s_per_episode"] = arrivals
    summary["settled_at_s_per_episode"] = [
        ep["settled_at_s"] if ep["settled_at_s"] is not None else None for ep in episodes
    ]
    summary["success_per_episode"] = [
        bool(ep["recovered"] and not ep["terminated"]) for ep in episodes
    ]
    return summary


def label_history_summary(pairs):
    """Aggregate teacher-label statistics over one round's annotation."""
    labels = np.concatenate([labels for _obs, labels in pairs]) if pairs else np.array([])
    if labels.size == 0:
        return {
            "pairs": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "saturated_fraction": None,
        }
    return {
        "pairs": int(labels.size),
        "mean": float(np.mean(labels)),
        "std": float(np.std(labels)),
        "min": float(np.min(labels)),
        "max": float(np.max(labels)),
        "saturated_fraction": float(np.mean(np.abs(labels) >= CONTROL_LIMIT - 1e-6)),
    }


def eval_step_count(episodes):
    return int(sum(len(ep["arrays"]["controls"]) for ep in episodes))


def default_config(rounds, updates_per_round, rollouts):
    return PPOConfig(
        n_envs=rollouts,
        rollout_steps=ROLLOUT_HORIZON,
        updates=rounds * updates_per_round,
        epochs=EPOCHS_FOR_DAGGER,
        minibatch=MINIBATCH,
        train_episode_steps=ROLLOUT_HORIZON,
        eval_every=rounds * updates_per_round,
        task_envs=rollouts,
    )


# ------------------------------------------------------------------ experiment
def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    train_seeds=TRAIN_SEEDS,
    eval_seed_count=EVAL_SEEDS,
    rounds=ROUNDS,
    rollouts=ANNOTATE_ROLLOUTS,
    updates_per_round=UPDATES_PER_ROUND,
    w_bc_levels=W_BC_LEVELS,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    if not isinstance(rounds, int) or not 1 <= rounds <= 20:
        raise ValueError("rounds must be an integer in [1, 20]")
    if not isinstance(updates_per_round, int) or not 1 <= updates_per_round <= 200:
        raise ValueError("updates_per_round must be an integer in [1, 200]")
    if not isinstance(rollouts, int) or not 1 <= rollouts <= 64:
        raise ValueError("rollouts must be an integer in [1, 64]")
    levels = tuple(float(value) for value in w_bc_levels)
    if not levels or any(value < 0.0 for value in levels):
        raise ValueError("w_BC levels must be non-negative")
    config = config or default_config(rounds, updates_per_round, rollouts)
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    reward = RewardFunction(reference)

    # Teacher quality gate AND the record's baseline: the lesson-7 controller
    # re-run from the exact down start; annotation requires 20/20.
    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    baseline_summary = summarize_episodes(baseline_records)
    teacher_gate = {
        "guard": (
            "annotation only proceeds when the teacher re-run passes the lesson-7 "
            "acceptance on every repeat"
        ),
        "episodes": len(baseline_records),
        "successes": baseline_summary["successes"],
        "gate_passed": bool(all(r["recovered"] for r in baseline_records)),
        "median_settled_at_s": baseline_summary["median_settled_at_s"],
    }
    if not teacher_gate["gate_passed"]:
        raise ValueError("teacher quality gate failed: annotation data source not verified")

    output.mkdir(parents=True, exist_ok=False)
    total_env_steps = eval_seed_count * EVAL_EPISODE_STEPS * train_seeds * len(levels)
    tier_entries = []
    reward_curves, bc_curves, w_curves = {}, {}, {}
    dataset_sizes, dropped_counts = {}, {}
    eval_record_arrays = {}  # (level, seed) -> dict of (R, 20) arrays
    initial_eval_arrays = {}
    det_states, det_controls = {}, {}
    label_hist_acc = {}  # (level, round) -> bin counts
    policy_payloads = {}
    final_round_episodes = {}  # (level, seed) -> final-round eval episodes (with arrays)
    featured_cases = []

    for level_index, w_bc in enumerate(levels):
        per_seed_records = []
        for seed_index in range(train_seeds):
            seed_started = time.perf_counter()
            seed_env_steps = 0
            init_seed = [seed, SEED_OFFSET_INIT + level_index, seed_index]
            policy = GaussianPolicy(STATE_INPUTS, config.hidden, init_seed, config.log_std_init)
            value = MLPTower(STATE_INPUTS, config.hidden, 1, [*init_seed, 1])
            parameters = [*policy.parameters(), *value.weights, *value.biases]
            optimizer = AdamOptimizer(parameters, lr=config.lr)
            action_rng = np.random.default_rng([seed, SEED_OFFSET_ACT + level_index, seed_index])
            mb_rng = np.random.default_rng([seed, SEED_OFFSET_SHUFFLE + level_index, seed_index])
            bc_rng = np.random.default_rng([seed, SEED_OFFSET_BC + level_index, seed_index])
            total_updates = rounds * updates_per_round

            envs = [
                make_swingup_environment(max_episode_steps=ROLLOUT_HORIZON) for _ in range(rollouts)
            ]
            for index, env in enumerate(envs):
                env.reset(seed=BASE_ENV + seed * 1000 + level_index * 100 + seed_index + index)

            # The evolution curve's zero point: the fresh policy evaluated under
            # the verbatim acceptance before the first correction.
            initial_episodes = evaluate_dagger_round(
                policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
            )
            seed_env_steps += eval_step_count(initial_episodes)
            initial_summary = eval_episode_summary(initial_episodes)
            initial_eval_arrays[(level_index, seed_index)] = (
                np.asarray(initial_summary["success_per_episode"], dtype=bool),
                np.asarray(
                    [
                        value if value is not None else np.nan
                        for value in initial_summary["settled_at_s_per_episode"]
                    ],
                    dtype=float,
                ),
                np.asarray(
                    [
                        value if value is not None else np.nan
                        for value in initial_summary["first_arrival_s_per_episode"]
                    ],
                    dtype=float,
                ),
            )

            agg_obs = np.empty((0, STATE_INPUTS))
            agg_actions = np.empty(0)
            rounds_records = []
            reward_curve = np.empty(rounds * updates_per_round)
            bc_curve = np.empty(rounds * updates_per_round)
            w_curve = np.empty(rounds * updates_per_round)
            pos = 0
            first_successful_round = None
            eval_records_here = {
                "recovered": [],
                "terminated": [],
                "settled": [],
                "arrival": [],
                "returns": [],
            }
            for round_index in range(rounds):
                w_here = bc_weight_at(round_index, rounds, w_bc, W_BC_MIN)
                episodes = collect_student_rollouts(
                    envs,
                    policy,
                    value,
                    reward,
                    reference,
                    master_seed=seed,
                    level_index=level_index,
                    seed_index=seed_index,
                    round_index=round_index,
                    horizon=ROLLOUT_HORIZON,
                    action_rng=action_rng,
                )
                rollout_steps = int(sum(len(ep["rewards"]) for ep in episodes))
                seed_env_steps += rollout_steps
                total_env_steps += rollout_steps
                batch, reward_mean = rollout_to_batch(
                    episodes,
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                    reward_scale=config.reward_scale,
                )
                self_consistent = True
                dropped = 0
                label_stats = {
                    "pairs": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "saturated_fraction": None,
                }
                # The tier decides whether the teacher labels at all; the round
                # weight decides how strongly the aggregated labels pull. A
                # DAgger tier therefore keeps aggregating even in the final
                # round where w_here = 0 (handover to the pure policy gradient).
                dataset_before = len(agg_obs)
                if w_bc > 0.0:
                    pairs, dropped, self_consistent = annotate_rollouts(
                        envs, design, episodes, reference
                    )
                    for obs, labels in pairs:
                        obs, labels, extra = filter_annotation_pairs(obs, labels)
                        dropped += extra
                        agg_obs = np.concatenate([agg_obs, obs], axis=0)
                        agg_actions = np.concatenate([agg_actions, labels], axis=0)
                    label_stats = label_history_summary(pairs)
                    key = (level_index, round_index)
                    if key not in label_hist_acc:
                        label_hist_acc[key] = np.zeros(len(LABEL_BINS) - 1, dtype=int)
                    for obs, labels in pairs:
                        counts, _ = np.histogram(labels, bins=LABEL_BINS)
                        label_hist_acc[key] += counts
                loss_first = loss_last = bc_first = bc_last = np.nan
                for update_in_round in range(updates_per_round):
                    stats = train_dagger_round(
                        policy,
                        value,
                        optimizer,
                        parameters,
                        batch,
                        agg_obs,
                        agg_actions,
                        config=config,
                        w_bc=w_here,
                        update_index=round_index * updates_per_round + update_in_round,
                        total_updates=total_updates,
                        mb_rng=mb_rng,
                        bc_rng=bc_rng,
                    )
                    reward_curve[pos] = stats["loss_mean"]
                    bc_curve[pos] = stats["bc_mean"]
                    w_curve[pos] = w_here
                    pos += 1
                    if update_in_round == 0:
                        loss_first, bc_first = stats["loss_mean"], stats["bc_mean"]
                    loss_last, bc_last = stats["loss_mean"], stats["bc_mean"]

                eval_episodes = evaluate_dagger_round(
                    policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
                )
                seed_env_steps += eval_step_count(eval_episodes)
                total_env_steps += eval_step_count(eval_episodes)
                eval_summary = eval_episode_summary(eval_episodes)
                det_record, det_arrays = deterministic_dapg_episode(policy, reward, reference, dt)
                det_states[(level_index, seed_index, round_index)] = det_arrays["states"]
                det_controls[(level_index, seed_index, round_index)] = det_arrays["controls"]
                eval_records_here["recovered"].append(eval_summary["success_per_episode"])
                eval_records_here["terminated"].append(
                    [bool(ep["terminated"]) for ep in eval_episodes]
                )
                eval_records_here["settled"].append(
                    [
                        value if value is not None else np.nan
                        for value in eval_summary["settled_at_s_per_episode"]
                    ]
                )
                eval_records_here["arrival"].append(
                    [
                        value if value is not None else np.nan
                        for value in eval_summary["first_arrival_s_per_episode"]
                    ]
                )
                eval_records_here["returns"].append([float(ep["return"]) for ep in eval_episodes])
                if first_successful_round is None and eval_summary["successes"] > 0:
                    first_successful_round = round_index
                if round_index == rounds - 1:
                    final_round_episodes[(level_index, seed_index)] = eval_episodes
                rounds_records.append(
                    {
                        "round_index": round_index,
                        "w_bc": w_here,
                        "rollout_steps": rollout_steps,
                        "dataset_size": len(agg_obs),
                        "dataset_added": int(len(agg_obs) - dataset_before),
                        "dropped_labels": dropped,
                        "label_stats": label_stats,
                        "self_consistency": self_consistent,
                        "reward_mean": reward_mean,
                        "loss_first": float(loss_first),
                        "loss_last": float(loss_last),
                        "bc_first": float(bc_first),
                        "bc_last": float(bc_last),
                        "eval": eval_summary,
                        "deterministic": {
                            "recovered": bool(det_record["recovered"]),
                            "settled_at_s": det_record["settled_at_s"],
                            "first_arrival_s": det_record["first_arrival_s"],
                        },
                    }
                )
                if log is not None:
                    log(
                        f"tier w={w_bc:g} seed {seed_index} round {round_index + 1}/{rounds}: "
                        f"dataset {len(agg_obs)}, w {w_here:.3f}, "
                        f"ok {eval_summary['successes']}/{eval_seed_count}, "
                        f"arrival {eval_summary['arrival']['episodes_with_arrival']}, "
                        f"bc {bc_last:.4f}"
                    )
            reward_curves[(level_index, seed_index)] = reward_curve
            bc_curves[(level_index, seed_index)] = bc_curve
            w_curves[(level_index, seed_index)] = w_curve
            dataset_sizes[(level_index, seed_index)] = np.asarray(
                [r["dataset_size"] for r in rounds_records], dtype=int
            )
            dropped_counts[(level_index, seed_index)] = np.asarray(
                [r["dropped_labels"] for r in rounds_records], dtype=int
            )
            eval_record_arrays[(level_index, seed_index)] = {
                "recovered": np.asarray(eval_records_here["recovered"], dtype=bool),
                "terminated": np.asarray(eval_records_here["terminated"], dtype=bool),
                "settled": np.asarray(eval_records_here["settled"], dtype=float),
                "arrival": np.asarray(eval_records_here["arrival"], dtype=float),
                "returns": np.asarray(eval_records_here["returns"], dtype=float),
            }
            policy_payloads[(level_index, seed_index)] = policy.arrays()
            per_seed_records.append(
                {
                    "seed_index": seed_index,
                    "env_steps": seed_env_steps,
                    "wall_time_s": time.perf_counter() - seed_started,
                    "initial_eval": initial_summary,
                    "rounds": rounds_records,
                    "per_round_successes": [r["eval"]["successes"] for r in rounds_records],
                    "first_successful_round": first_successful_round,
                }
            )
            if log is not None:
                log(
                    f"tier w={w_bc:g} seed {seed_index} done: first_success "
                    f"{'round ' + str(first_successful_round + 1) if first_successful_round is not None else 'never'}"
                )
            for env in envs:
                env.close()

        successes_per_seed_per_round = [
            [r["eval"]["successes"] for r in seed_run["rounds"]] for seed_run in per_seed_records
        ]
        arrivals_per_seed_per_round = [
            [r["eval"]["arrival"]["episodes_with_arrival"] for r in seed_run["rounds"]]
            for seed_run in per_seed_records
        ]
        initial_successes = [s["initial_eval"]["successes"] for s in per_seed_records]
        initial_arrivals = [
            s["initial_eval"]["arrival"]["episodes_with_arrival"] for s in per_seed_records
        ]
        final_successes = [row[-1] for row in successes_per_seed_per_round]
        final_arrivals = [row[-1] for row in arrivals_per_seed_per_round]
        first_success_per_seed = [s["first_successful_round"] for s in per_seed_records]
        settled_all = [
            value
            for seed_run in per_seed_records
            for record in seed_run["rounds"]
            if record["round_index"] == rounds - 1
            for value in record["eval"]["settled_at_s_per_episode"]
            if value is not None
        ]
        tier_entry = {
            "w_bc": w_bc,
            "label": tier_short_label(w_bc),
            "evaluations_per_round": eval_seed_count,
            "per_seed": per_seed_records,
            "initial_successes_per_seed": initial_successes,
            "initial_arrivals_per_seed": initial_arrivals,
            "final_successes_per_seed": final_successes,
            "final_arrivals_per_seed": final_arrivals,
            "successes_per_seed_per_round": successes_per_seed_per_round,
            "arrivals_per_seed_per_round": arrivals_per_seed_per_round,
            "first_success_per_seed": first_success_per_seed,
            "eval_total": eval_seed_count * train_seeds,
            "eval_successes": int(sum(final_successes)),
            "median_settled_at_s": float(np.median(settled_all)) if settled_all else None,
            "aggregate": {
                "initial_total_successes": int(sum(initial_successes)),
                "final_round_total_successes": int(sum(final_successes)),
                "final_round_total_arrivals": int(sum(final_arrivals)),
                "first_success_any": any(v is not None for v in first_success_per_seed),
                "rounds_to_first_success": (
                    min(v for v in first_success_per_seed if v is not None)
                    if any(v is not None for v in first_success_per_seed)
                    else None
                ),
                "first_success_per_seed": first_success_per_seed,
            },
        }
        last_episodes = [
            ep
            for seed_index in range(train_seeds)
            for ep in final_round_episodes[(level_index, seed_index)]
        ]
        tier_entry["final_failure_counts"] = failure_counts(last_episodes)
        tier_entries.append(tier_entry)

    # Featured failure case: first failing episode of the first DAgger tier's
    # final round under (seed, eval) scan order; the archive keeps its arrays.
    for level_index, w_bc in enumerate(levels):
        if w_bc <= 0.0:
            continue
        for seed_index in range(train_seeds):
            for episode in final_round_episodes[(level_index, seed_index)]:
                if episode["terminated"] or not episode["recovered"]:
                    featured_cases.append(
                        {
                            "kind": "dagger_final_eval_failure",
                            "w_bc_label": tier_label(w_bc),
                            "seed_index": seed_index,
                            "eval_seed": episode["eval_seed"],
                            "failure_reason": episode["failure_reason"],
                            "terminated": bool(episode["terminated"]),
                            "recovered": bool(episode["recovered"]),
                            "settled_at_s": episode["settled_at_s"],
                            "max_abs_cart_position_m": episode["max_abs_cart_position_m"],
                            "peak_abs_motor_force_n": episode["peak_abs_motor_force_n"],
                            "arrays": episode["arrays"],
                        }
                    )
                    break
            if featured_cases:
                break
        if featured_cases:
            break

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        teacher_gate=teacher_gate,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        rounds=rounds,
        rollouts=rollouts,
        updates_per_round=updates_per_round,
        levels=levels,
        tier_entries=tier_entries,
        featured_cases=featured_cases,
        total_env_steps=total_env_steps,
        elapsed=elapsed,
    )
    archive = build_archive(
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        reward_curves=reward_curves,
        bc_curves=bc_curves,
        w_curves=w_curves,
        dataset_sizes=dataset_sizes,
        dropped_counts=dropped_counts,
        eval_record_arrays=eval_record_arrays,
        initial_eval_arrays=initial_eval_arrays,
        det_states=det_states,
        det_controls=det_controls,
        label_hist_acc=label_hist_acc,
        policy_payloads=policy_payloads,
        featured_cases=featured_cases,
        rounds=rounds,
        levels=len(levels),
        train_seeds=train_seeds,
    )
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_round_evolution(output / "round_evolution.png", report, output)
    save_correction_analysis(output / "correction_analysis.png", report, output)
    save_comparison(output / "comparison.png", report, output)
    return report


def build_report(
    *,
    seed,
    config,
    design,
    baseline_records,
    baseline_identical,
    teacher_gate,
    train_seeds,
    eval_seed_count,
    rounds,
    rollouts,
    updates_per_round,
    levels,
    tier_entries,
    featured_cases,
    total_env_steps,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    comparison = [
        {
            "label": "基线（第 7 课能量整形+LQR，零样本）",
            "episodes": baseline_summary["episodes"],
            "successes": baseline_summary["successes"],
            "median_settled_at_s": baseline_summary["median_settled_at_s"],
            "source": "this record",
        },
        {
            "label": "纯 PPO（第 29 课，只凭奖励）",
            "episodes": LESSON29_PPO_REFERENCE["episodes"],
            "successes": LESSON29_PPO_REFERENCE["successes"],
            "median_settled_at_s": LESSON29_PPO_REFERENCE["median_settled_at_s"],
            "first_arrival": LESSON29_PPO_REFERENCE["first_arrival"],
            "source": LESSON29_PPO_REFERENCE["source"],
        },
        {
            "label": "DAPG 离线示教（第 32 课，w=10）",
            "episodes": LESSON32_DAPG_REFERENCE["episodes"],
            "successes": LESSON32_DAPG_REFERENCE["successes"],
            "median_settled_at_s": LESSON32_DAPG_REFERENCE["median_settled_at_s"],
            "first_arrival": LESSON32_DAPG_REFERENCE["first_arrival"],
            "source": LESSON32_DAPG_REFERENCE["source"],
        },
        {
            "label": "两阶段奖励（第 34 课）",
            "episodes": LESSON34_TWOPHASE_REFERENCE["episodes"],
            "successes": LESSON34_TWOPHASE_REFERENCE["successes"],
            "median_settled_at_s": LESSON34_TWOPHASE_REFERENCE["median_settled_at_s"],
            "first_arrival": LESSON34_TWOPHASE_REFERENCE["first_arrival"],
            "source": LESSON34_TWOPHASE_REFERENCE["source"],
        },
        {
            "label": "SAC（第 35 课）",
            "episodes": LESSON35_SAC_REFERENCE["episodes"],
            "successes": LESSON35_SAC_REFERENCE["successes"],
            "median_settled_at_s": LESSON35_SAC_REFERENCE["median_settled_at_s"],
            "first_arrival": LESSON35_SAC_REFERENCE["first_arrival"],
            "source": LESSON35_SAC_REFERENCE["source"],
        },
    ]
    for entry in tier_entries:
        comparison.append(
            {
                "label": f"DAgger 在线纠错（{entry['label']}，末轮）",
                "episodes": entry["eval_total"],
                "successes": entry["eval_successes"],
                "median_settled_at_s": entry["median_settled_at_s"],
                "source": "this record",
            }
        )
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "protocol": {
            "task": "lesson-7 full-rotation swing-up, exact down start (reference angle -180 deg)",
            "route": (
                "on-line teacher correction (DAgger, Ross et al. RSS 2010): the teacher labels "
                "the states the student reached under its own policy, round by round; the "
                "objective is lesson-29 PPO plus the lesson-32 BC regularizer on the "
                "aggregated online dataset - a direct attack on the closed-loop precision gap "
                "lesson 32 left open (open-loop BC regression vs the +/-0.01 rad/s invariant)"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "teacher": {
                "controller": (
                    "lesson-7 HybridSwingupController (energy shaping + LQR hysteresis), "
                    "zero-shot; quality known 20/20 (docs/09)"
                ),
                "quality_gate": (
                    "the teacher re-run from the exact down start must pass the lesson-7 "
                    "acceptance on every repeat; annotation raises otherwise"
                ),
                "annotation": {
                    "mechanism": (
                        "one controller instance per student episode replays the recorded state "
                        "sequence, so its hysteresis mode (kick/swingup/balance) evolves along "
                        "the states it labels - the on-line teacher"
                    ),
                    "drop_rule": (
                        "a pair (obs(s_t), a_teacher(s_t)) with a non-finite state or label is "
                        "dropped and counted; the teacher is deterministic, so re-annotation of "
                        "the first episode per round must be bitwise identical"
                    ),
                },
            },
            "student": {
                "choice": (
                    "lesson-29 PPO stack (5x64x64 ReLU Gaussian policy + value net, hand-written "
                    "numpy) with the lesson-32 BC-regularized objective"
                ),
                "rationale": (
                    "closest continuity to the DAPG line (lesson 32): same policy class, same "
                    "objective shape, so the only change is WHERE the (s, a) pairs come from - "
                    "the teacher's own rollouts (offline) vs the student's own states (online). "
                    "The on-policy batches also fit DAgger's round structure naturally: each "
                    "round's batch is exactly the current policy's behavior"
                ),
                "objective": "L = L_PPO + w_BC * MSE(mean_theta(s), a_teacher)",
                "initialization": (
                    "fresh per tier and seed (the lesson-32 recorded weights are NOT reused); "
                    "the data-aggregation property is the new variable"
                ),
                "student_action_role": (
                    "the student's own action a_t enters the PPO term as the on-policy action "
                    "(recorded during the rollout); BC matches mean_theta(s) to a_teacher(s) only"
                ),
            },
            "dagger_loop": {
                "rounds": rounds,
                "annotated_rollouts_per_round": rollouts,
                "initial_state": (
                    "exact resting down start + the lesson-32 jitter: angle +/-"
                    f"{JITTER_ANGLE} rad, omega +/-{JITTER_OMEGA} rad/s, cart +/-{JITTER_CART} m "
                    "(the student policy is stochastic, so episodes are distinct without it)"
                ),
                "horizon_steps": ROLLOUT_HORIZON,
                "data_aggregation": (
                    "D_r = D_{r-1} U {(s, a_teacher(s)) : s from the student's own round-r "
                    "rollouts}; all rounds kept (DAgger aggregation - the data follows the policy)"
                ),
                "training_per_round": {
                    "updates": updates_per_round,
                    "epochs": EPOCHS_FOR_DAGGER,
                    "minibatch": MINIBATCH,
                    "bc_minibatch": BC_MINIBATCH,
                    "lr_schedule": (
                        "lesson-29 annealing on the total update count: "
                        "lr(u) = lr0 * (1 - (u+1)/(rounds*updates_per_round))"
                    ),
                },
                "eval_per_round": (
                    f"{eval_seed_count} stochastic episodes from the exact down start under the "
                    "verbatim lesson-7 recovery_metrics, plus one mean-action episode; the fresh "
                    "policy is evaluated before round 0 as the evolution curve's zero point"
                ),
            },
            "w_bc_schedule": {
                "tiers": list(levels),
                "w_min": W_BC_MIN,
                "schedule": (
                    "constant inside a round, linear across the rounds from w_init (round 0) to "
                    "w_min (round rounds-1) - the lesson-32 convention quantised onto the round "
                    "grid"
                ),
                "control_tier": (
                    "w_BC = 0 runs the identical loop without the BC term: pure on-line policy "
                    "gradient fine-tuning; its annotation loop is skipped (no BC term) and "
                    "recorded as such"
                ),
            },
            "reward": RewardFunction(reference).as_dict(),
            "reward_scale_for_learning": config.reward_scale,
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * design.dt,
            "cart_failure_boundary_m": SAFE_CART_POSITION,
            "control_limit": CONTROL_LIMIT,
            "actuator_gear": design.actuator_gear,
            "settled_tolerances": [0.02, 0.01, 0.02, 0.02],
            "minimum_settled_tail_s": 2.0,
            "acceptance": (
                "lesson-7 recovery_metrics reused verbatim: all four wrapped state errors within "
                "tolerances for the final continuous tail >= 2 s; success = recovered with no "
                "physical failure; swing-up time = settled_at_s counted from t=0"
            ),
            "upright_region": {
                "definition": "|alpha| <= 0.3 rad (the lesson-7 capture threshold)",
            },
            "first_success": (
                "the first round whose per-round evaluation passes the lesson-7 acceptance with "
                ">= 1 successful episode - the headline 0 -> 1 of this lesson"
            ),
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lessons 29/32",
            },
            "seed_streams": {
                "network_init": "default_rng([master, 7000 + level, train_seed]); value [.., 1]",
                "rollout_actions": "default_rng([master, 5000 + level, train_seed])",
                "minibatch_order": "default_rng([master, 9000 + level, train_seed])",
                "bc_dataset_minibatch": "default_rng([master, 4500 + level, train_seed])",
                "rollout_start_jitter": (
                    "default_rng([base, 6100, round, env]); base = 10000 + master*1000 + "
                    "level*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
            },
        },
        "hyperparameters": {
            **asdict(config),
            "hidden": list(config.hidden),
            "dagger_rounds": rounds,
            "dagger_rollouts": rollouts,
            "dagger_updates_per_round": updates_per_round,
            "w_bc_levels": list(levels),
            "w_bc_min": W_BC_MIN,
        },
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{eval_seed_count} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
            "per_episode": baseline_records,
        },
        "teacher_verification": teacher_gate,
        "tiers": tier_entries,
        "comparison": comparison,
        "lesson29_reference": LESSON29_PPO_REFERENCE,
        "lesson32_reference": LESSON32_DAPG_REFERENCE,
        "lesson34_reference": LESSON34_TWOPHASE_REFERENCE,
        "lesson35_reference": LESSON35_SAC_REFERENCE,
        "hypothesis": {
            "claim": (
                "DAgger labels the states the student reaches, so each round corrects the "
                "specific closed-loop mistakes of the CURRENT policy - the compound-error fix "
                "lesson 32's offline BC could not provide. Question: does the per-round success "
                "rate on the exact down start move 0 -> >0 (the first historical 0 -> 1)? The "
                "w_BC = 0 tier separates the correction data from plain on-line fine-tuning"
            ),
            "verdict": "see tiers[*].aggregate.first_success_any (final round, per seed)",
        },
        "failure_analysis": {
            "final_round_counts": {
                f"{entry['label']}（w={entry['w_bc']:g}）": entry["final_failure_counts"]
                for entry in tier_entries
            },
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in featured_cases
            ],
        },
        "training": {
            "train_seeds": train_seeds,
            "rounds": rounds,
            "rollouts": rollouts,
            "env_steps_total": total_env_steps,
            "wall_time_s_total": elapsed,
            "curves_note": (
                "reward_curve_<tier>_<seed> (per-update total loss), bc_curve_<tier>_<seed>, "
                "w_bc_curve_<tier>_<seed>, dataset_size_<tier>_<seed>, dropped_<tier>_<seed>, "
                "eval_* arrays, initial_eval_* arrays, det_states/det_controls and the teacher "
                "label histograms live in trajectories.npz"
            ),
        },
        "limitations": [
            (
                "w_BC init at a single point (10.0, the lesson-32 strong-anchor level), one "
                "annealing shape (per-round step, linear across rounds) and one teacher "
                "(lesson-7 hybrid controller); no grid over w_init or schedule shape."
            ),
            (
                "The annotation is an offline replay of recorded student states - the teacher "
                "never interacts with the environment while watching, and no noise/delay/mass "
                "error is present."
            ),
            (
                f"{rounds} rounds x {rollouts} rollouts is a small correction budget relative to "
                "a full 500k-step RL run; per-round evaluations are finite-sample counts."
            ),
            (
                "The PPO batch is the same rollouts that get annotated; a larger or separate "
                "training rollout stream was not swept."
            ),
            (
                "The cited lesson-29/32/34/35 comparison rows come from their official records, "
                "not re-run here; the w_BC = 0 tier is run in the identical loop, and its PPO "
                "gradient path is the lesson-29 code (pinned by the unit w_BC = 0 contract)."
            ),
        ],
    }
    return report


def build_archive(
    *,
    baseline_states,
    baseline_controls,
    reward_curves,
    bc_curves,
    w_curves,
    dataset_sizes,
    dropped_counts,
    eval_record_arrays,
    initial_eval_arrays,
    det_states,
    det_controls,
    label_hist_acc,
    policy_payloads,
    featured_cases,
    rounds,
    levels,
    train_seeds,
):
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=float),
        "baseline_controls": np.asarray(baseline_controls, dtype=float),
        "label_bin_edges": LABEL_BINS,
    }
    for level in range(levels):
        for seed in range(train_seeds):
            archive[f"reward_curve_{level}_{seed}"] = reward_curves[(level, seed)]
            archive[f"bc_curve_{level}_{seed}"] = bc_curves[(level, seed)]
            archive[f"w_bc_curve_{level}_{seed}"] = w_curves[(level, seed)]
            archive[f"dataset_size_{level}_{seed}"] = dataset_sizes[(level, seed)]
            archive[f"dropped_{level}_{seed}"] = dropped_counts[(level, seed)]
            arrays = eval_record_arrays[(level, seed)]
            archive[f"eval_recovered_{level}_{seed}"] = arrays["recovered"]
            archive[f"eval_terminated_{level}_{seed}"] = arrays["terminated"]
            archive[f"eval_settled_s_{level}_{seed}"] = arrays["settled"]
            archive[f"eval_arrival_s_{level}_{seed}"] = arrays["arrival"]
            archive[f"eval_returns_{level}_{seed}"] = arrays["returns"]
            initial = initial_eval_arrays[(level, seed)]
            archive[f"initial_eval_recovered_{level}_{seed}"] = initial[0]
            archive[f"initial_eval_settled_s_{level}_{seed}"] = initial[1]
            archive[f"initial_eval_arrival_s_{level}_{seed}"] = initial[2]
            for r in range(rounds):
                archive[f"det_states_{level}_{seed}_r{r}"] = det_states[(level, seed, r)]
                archive[f"det_controls_{level}_{seed}_r{r}"] = det_controls[(level, seed, r)]
            for name, array in policy_payloads[(level, seed)].items():
                archive[f"policy_{level}_{seed}_{name}"] = array
        for r in range(rounds):
            counts = label_hist_acc.get((level, r))
            if counts is None:
                counts = np.zeros(len(LABEL_BINS) - 1, dtype=int)
            archive[f"label_hist_{level}_{r}"] = counts
    for index, case in enumerate(featured_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    levels = len(report["protocol"]["w_bc_schedule"]["tiers"])
    seeds = report["training"]["train_seeds"]
    rounds = report["training"]["rounds"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    keys = {
        "baseline_states",
        "baseline_controls",
        "label_bin_edges",
    }
    for level in range(levels):
        for seed in range(seeds):
            keys.update(
                f"{name}_{level}_{seed}"
                for name in (
                    "reward_curve",
                    "bc_curve",
                    "w_bc_curve",
                    "dataset_size",
                    "dropped",
                    "eval_recovered",
                    "eval_terminated",
                    "eval_settled_s",
                    "eval_arrival_s",
                    "eval_returns",
                    "initial_eval_recovered",
                    "initial_eval_settled_s",
                    "initial_eval_arrival_s",
                )
            )
            keys.update(f"policy_{level}_{seed}_{name}" for name in policy_array_names(hidden))
            keys.update(
                f"det_{suffix}_{level}_{seed}_r{r}"
                for r in range(rounds)
                for suffix in ("states", "controls")
            )
        keys.update(f"label_hist_{level}_{r}" for r in range(rounds))
    keys.update(
        f"case{index}_{suffix}"
        for index in range(len(report["failure_analysis"]["featured_cases"]))
        for suffix in ("states", "controls")
    )
    return keys


# -------------------------------------------------------------------- figures
def save_round_evolution(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    rounds = report["training"]["rounds"]
    eval_per_round = report["tiers"][0]["evaluations_per_round"]
    colors = ("#b91c1c", "#64748b")
    x = np.arange(rounds + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for tier_index, entry in enumerate(report["tiers"]):
        color = colors[tier_index % len(colors)]
        per_round = np.asarray(entry["successes_per_seed_per_round"], dtype=float).T  # (R, seeds)
        rows = np.concatenate(
            [
                np.asarray(entry["initial_successes_per_seed"], dtype=float)[None, :],
                per_round,
            ],
            axis=0,
        )
        for seed in range(seeds):
            axes[0, 0].plot(x, rows[:, seed], "o--", markersize=3, alpha=0.35, color=color)
        axes[0, 0].plot(
            x,
            rows.mean(axis=1),
            "o-",
            linewidth=1.8,
            color=color,
            label=f"{entry['label']}（w={entry['w_bc']:g}）",
        )
    axes[0, 0].axhline(eval_per_round, color="gray", linestyle=":", linewidth=0.8)
    axes[0, 0].set(
        xlabel="轮次（0 = 训练前初值）",
        ylabel=f"成功回合数（/{eval_per_round}）",
        title=f"轮次演化：下方初态成功率（{eval_per_round} 回合/轮/种子）",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)
    for tier_index, entry in enumerate(report["tiers"]):
        color = colors[tier_index % len(colors)]
        per_round = np.asarray(entry["arrivals_per_seed_per_round"], dtype=float).T
        rows = np.concatenate(
            [
                np.asarray(entry["initial_arrivals_per_seed"], dtype=float)[None, :],
                per_round,
            ],
            axis=0,
        )
        for seed in range(seeds):
            axes[0, 1].plot(x, rows[:, seed], "o--", markersize=3, alpha=0.35, color=color)
        axes[0, 1].plot(
            x, rows.mean(axis=1), "o-", linewidth=1.8, color=color, label=entry["label"]
        )
    axes[0, 1].set(
        xlabel="轮次（0 = 训练前初值）",
        ylabel=f"直立首达回合数（/{eval_per_round}）",
        title="直立首达（|α|≤0.3 rad）随轮次演化",
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[tier_index % len(colors)]
            sizes = np.asarray(
                [data[f"dataset_size_{tier_index}_{seed}"] for seed in range(seeds)], dtype=float
            )
            axes[1, 0].plot(
                np.arange(1, rounds + 1),
                sizes.mean(axis=0),
                "o-",
                linewidth=1.8,
                color=color,
                label=f"{entry['label']} 数据集规模",
            )
            dropped = np.asarray(
                [data[f"dropped_{tier_index}_{seed}"] for seed in range(seeds)], dtype=float
            )
            if dropped.size and dropped.max() > 0:
                axes[1, 0].plot(
                    np.arange(1, rounds + 1),
                    dropped.mean(axis=0),
                    "s--",
                    markersize=4,
                    color=color,
                    label=f"{entry['label']} 丢弃标注",
                )
        bcs = np.stack(
            [
                np.stack([data[f"bc_curve_{tier_index}_{seed}"] for seed in range(seeds)], axis=0)
                for tier_index in range(len(report["tiers"]))
            ],
            axis=0,
        )
        ws = np.stack(
            [
                np.stack([data[f"w_bc_curve_{tier_index}_{seed}"] for seed in range(seeds)], axis=0)
                for tier_index in range(len(report["tiers"]))
            ],
            axis=0,
        )
    for tier_index, entry in enumerate(report["tiers"]):
        color = colors[tier_index % len(colors)]
        if entry["w_bc"] > 0.0:
            axes[1, 1].plot(
                np.arange(1, len(bcs[tier_index][0]) + 1),
                np.nanmean(bcs[tier_index], axis=0),
                linewidth=1.8,
                color=color,
                label=f"{entry['label']} BC MSE",
            )
    for tier_index, entry in enumerate(report["tiers"]):
        color = colors[tier_index % len(colors)]
        if entry["w_bc"] > 0.0:
            axes[1, 1].plot(
                np.arange(1, len(ws[tier_index][0]) + 1),
                np.nanmean(ws[tier_index], axis=0),
                linestyle=":",
                linewidth=1.2,
                color=color,
                label=f"{entry['label']} w_BC 调度",
            )
    axes[1, 1].set(
        xlabel="梯度更新序号（轮次 × 每轮更新数）",
        ylabel="MSE / 权重",
        title="学生与教师标注的逐帧残差（BC）与 w_BC 退火",
    )
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_correction_analysis(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    gear = report["protocol"]["actuator_gear"]
    rounds = report["training"]["rounds"]
    last = rounds - 1
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        teacher = data["baseline_states"]
        teacher_controls = data["baseline_controls"]
        det0 = data["det_states_0_0_r0"]
        det_last = data[f"det_states_0_0_r{last}"]
        det0_controls = data["det_controls_0_0_r0"]
        det_last_controls = data[f"det_controls_0_0_r{last}"]
        edges = data["label_bin_edges"]
        hist0 = data["label_hist_0_0"]
        hist_last = data[f"label_hist_0_{last}"]
    axes[0, 0].plot(
        np.arange(len(teacher)) * dt,
        np.cos(teacher[:, 1] - ref_theta),
        "--",
        color="#64748b",
        label="教师（第 7 课基线）",
    )
    axes[0, 0].plot(
        np.arange(len(det0)) * dt,
        np.cos(det0[:, 1] - ref_theta),
        color="#2563eb",
        linewidth=1.3,
        label="第 1 轮后（均值动作）",
    )
    axes[0, 0].plot(
        np.arange(len(det_last)) * dt,
        np.cos(det_last[:, 1] - ref_theta),
        color="#b91c1c",
        linewidth=1.3,
        label="末轮后（均值动作）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title="同一下方初态：教师 vs 学生（第 1 轮 → 末轮）",
    )
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].plot(
        np.arange(len(teacher)) * dt, teacher[:, 0], "--", color="#64748b", label="教师"
    )
    axes[0, 1].plot(np.arange(len(det_last)) * dt, det_last[:, 0], color="#b91c1c", label="末轮后")
    axes[0, 1].set(
        ylabel="小车位置（m）",
        xlabel="仿真时间（s）",
        title=f"小车位置（红点线 = ±{SAFE_CART_POSITION:g} m 失败边界）",
    )
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].stairs(
        teacher_controls * gear,
        np.arange(len(teacher_controls) + 1) * dt,
        color="#64748b",
        label="教师",
    )
    axes[1, 0].stairs(
        det_last_controls * gear,
        np.arange(len(det_last_controls) + 1) * dt,
        color="#b91c1c",
        label="末轮后",
    )
    axes[1, 0].stairs(
        det0_controls * gear,
        np.arange(len(det0_controls) + 1) * dt,
        color="#2563eb",
        alpha=0.6,
        label="第 1 轮后",
    )
    axes[1, 0].set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入（±300 N 限幅）")
    axes[1, 0].legend(fontsize=7)
    width = np.diff(edges)
    axes[1, 1].bar(
        edges[:-1],
        hist0,
        width=width * 0.95,
        align="edge",
        alpha=0.55,
        color="#2563eb",
        label="轮 0 标注（全部种子聚合）",
    )
    axes[1, 1].bar(
        edges[:-1],
        hist_last,
        width=width * 0.95,
        align="edge",
        alpha=0.55,
        color="#b91c1c",
        label=f"轮 {last} 标注（全部种子聚合）",
    )
    axes[1, 1].set(
        xlabel="教师逐帧标注动作（-3..3，×100 = 电机力）",
        ylabel="帧数",
        title="教师标注分布：学生状态分布上的教师动作",
    )
    axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    rows = report["comparison"]
    labels = ["基线", "PPO\n29", "DAPG\n32", "两阶段\n34", "SAC\n35"] + [
        f"{r['label']}\n末轮" for r in report["tiers"]
    ]
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    colors = ("#64748b", "#b91c1c", "#7c3aed", "#9333ea", "#d97706", "#0f766e", "#2563eb")
    bars = axes[0, 0].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=colors[: len(labels)],
        width=0.62,
    )
    for bar, s, t in zip(bars, successes, totals, strict=True):
        axes[0, 0].annotate(
            f"{s}/{t}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0, 0].set(ylabel="验收通过率（%）", ylim=(0, 112), title="多方式成功率（第 7 课口径）")
    axes[0, 0].tick_params(axis="x", labelsize=7)
    lines = ["0→1 是否出现（头条）："]
    for entry in report["tiers"]:
        per_seed = entry["first_success_per_seed"]
        lines.append(
            f"  {entry['label']}：每种子首成功轮次 "
            + "、".join(str(v + 1) if v is not None else "从未" for v in per_seed)
        )
    for entry in report["tiers"]:
        if entry["aggregate"]["first_success_any"]:
            lines.append(f"  → {entry['label']} 出现历史性 0→1")
    lines.append("")
    lines.append("对照（官方记录）：")
    lines.append("  PPO（29）：0/60，评估从未首达")
    lines.append("  DAPG（32）：0/60，首达 33/60（w=10）")
    lines.append("  两阶段（34）：0/60，首达 2/3 种子")
    lines.append("  SAC（35）：0/60，首达 0/60")
    axes[0, 1].axis("off")
    axes[0, 1].text(
        0.02,
        0.96,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8,
        transform=axes[0, 1].transAxes,
    )
    axes[0, 1].set(title="直立首达与首次成功（头条）")
    cases = report["failure_analysis"]["featured_cases"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        if cases:
            case = cases[0]
            states = data["case0_states"]
            axes[1, 0].plot(
                np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c"
            )
            axes[1, 0].axhspan(-1, 0, alpha=0.08, color="orange")
            w_text = f"（{case.get('w_bc_label', '')}）"
            axes[1, 0].set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{failure_label(case)}{w_text}",
            )
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "本记录没有学生失败回合",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )
            axes[1, 0].set(title="失败案例：无（全班通过）")
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[(tier_index + 4) % len(colors)]
            for seed, row in enumerate(entry["successes_per_seed_per_round"]):
                axes[1, 1].plot(
                    np.arange(len(row)) + 1,
                    row,
                    "o--",
                    markersize=4,
                    alpha=0.6,
                    color=color,
                    label=f"{entry['label']} 种子 {seed}",
                )
        axes[1, 1].set(xlabel="轮次", ylabel="成功回合数（/20）", title="每轮成功率：逐种子")
        axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=int, default=EVAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--rollouts", type=int, default=ANNOTATE_ROLLOUTS)
    parser.add_argument("--updates-per-round", type=int, default=UPDATES_PER_ROUND)
    parser.add_argument("--w-bc", type=float, nargs="+", default=list(W_BC_LEVELS))
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)

    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            train_seeds=args.train_seeds,
            eval_seed_count=args.eval_seeds,
            rounds=args.rounds,
            rollouts=args.rollouts,
            updates_per_round=args.updates_per_round,
            w_bc_levels=tuple(args.w_bc),
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["successes"],
                "tiers": {
                    entry["label"]: {
                        "final_round_successes": entry["aggregate"]["final_round_total_successes"],
                        "first_success_any": entry["aggregate"]["first_success_any"],
                        "first_success_per_seed": entry["aggregate"]["first_success_per_seed"],
                    }
                    for entry in report["tiers"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
