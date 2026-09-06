"""Lesson 32: DAPG-style demonstration airdrop - anchor the policy on the teacher.

Lesson 31's ladder (PBRS energy shaping) produced the first pure-gradient touch of
the cliff top (one training checkpoint at 150k steps) but never held it: the right
energy is not the right pose, and the last mile (capture, cart recentering, the
>= 2 s settled tail) still had no teacher. This lesson is the "airdrop": the
lesson-7 hybrid controller (energy shaping + LQR, zero-shot 20/20) rolls out D = 8
successful demonstrations from (lightly jittered) resting-down starts, and the
lesson-29 PPO objective gains an auxiliary behavior-cloning loss,

    L = L_PPO + w_BC * MSE(mu_theta(s_demo), a_demo),

with w_BC annealed linearly over the update count down to w_min = 0 (the DAPG
convention). This is the deliberately simplified DAPG (Rajeswaran et al.,
RSS 2018): the demonstrations never enter the replay buffer - they only feed the
BC loss on their stored (s, a) pairs - and the Q-filter (filtering the BC loss by
advantage sign) is NOT implemented, which is recorded as an honest deviation.

The environment, reward, observation, two-bank curriculum, budget (8 envs x 250
steps x 250 updates = 500k steps per seed), seeds and the strict lesson-7
acceptance are the lesson-29 stack verbatim; only the objective gains the BC
term. The w_BC = 0 guard is pinned bitwise against the lesson-29 pipeline twice:
(1) under one identical action stream the experiment environment emits the same
states / observations / rewards as a directly constructed lesson-29 VecSwingup,
and (2) a micro training with w_BC = 0 reproduces lesson-29 train_ppo's curves
and policy weights bit for bit.

Checks (same conditions throughout): the lesson-7 baseline re-run from the exact
down start (it is also the teacher of record), the cited lesson-29 pure-PPO and
lesson-31 PBRS rows, and the two new DAPG tiers w_BC in {10.0, 1.0} x 3 seeds.
Process metrics beyond success: the upright first arrival (|alpha| <= 0.3 rad;
lesson 31 touched it once at its 150k checkpoint), the headline first accepted
success (does 0 -> 1 happen at all?), and the BC loss decay along training (is
the demonstration remembered or forgotten?).

Honesty rule: if the airdrop also fails within the fixed budget, the failure is
the formal result together with its analysis (e.g. the BC anchor pinning the
policy on teacher behavior and suppressing exploration, or compounding error
re-appearing). Nothing is smoothed over.
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
from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    EVAL_SEEDS,
    PUSH_DURATION_S,
    PUSH_FIELDS,
    PUSH_FORCE_N,
    PUSH_START_WINDOW_S,
    SEED_OFFSET_ACT,
    SEED_OFFSET_INIT,
    SEED_OFFSET_SHUFFLE,
    STATE_INPUTS,
    TRAIN_SEEDS,
    GaussianPolicy,
    MLPTower,
    PPOConfig,
    RewardFunction,
    VecSwingup,
    baseline_evaluations,
    baseline_push_evaluations,
    clip_gradients_,
    collect_rollout,
    compute_gae,
    down_start_state,
    episode_metrics,
    episode_rewards,
    evaluate_policy,
    failure_counts,
    gaussian_entropy,
    make_push_plans,
    normalize_observation,
    pick_failure_cases,
    policy_array_names,
    ppo_losses_and_gradients,
    push_schedule,
    run_policy_episode,
    select_metrics,
    stack_controls,
    stack_trajectories,
    standardize,
    summarize_episodes,
    train_ppo,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)

EXPERIMENT = "dapg_swingup_lesson32"
SCHEMA_VERSION = 1

W_BC_LEVELS = (10.0, 1.0)  # the single new hand knob: the initial BC anchor weight
W_BC_MIN = 0.0  # annealed linearly to zero over training (DAPG convention)
DEMO_COUNT = 8  # DAPG showed a handful of demonstrations suffices
DEMO_ANGLE_JITTER = 0.15  # rad around the exact resting down start
DEMO_OMEGA_JITTER = 0.3  # rad/s
DEMO_CART_JITTER = 0.10  # m
DEMO_BATCH = 256  # demo pairs per gradient step (sampled from the demo set)
CAPTURE_ANGLE_RAD = 0.3  # lesson-7 capture threshold defines the upright region
GEAR = 100.0  # recovery_metrics reports forces as 100 * normalized command
EVAL_EVERY = 25

SEED_OFFSET_DEMO = 4000  # demonstration start jitter
SEED_OFFSET_DEMO_SAMPLE = 4400  # demo minibatch sampling inside the updates

# Context numbers imported verbatim from the lesson-29/31 official records
# (results/ppo_swingup_2026-09-06 docs/33, results/pbrs_swingup_2026-09-06
# docs/36); used only in the multi-way comparison table.
LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "median_peak_abs_motor_force_n": 300.0,
    "final_reward_mean_per_seed": [0.4103964962827423, 0.41928791619542005, 0.41858490469934645],
    "first_upright_arrival": "never (docs/33 section 3.4, mechanism 1)",
}
PBRS_REFERENCE = {
    "source": "results/pbrs_swingup_2026-09-06 (official lesson-31 record, docs/36)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "median_peak_abs_motor_force_n": 300.0,
    "first_arrival": "training checkpoint touch once (cE=2.0 seed 1 @ 150k steps, 1.24 s)",
}

SHORT_FAILURE = {
    "cart_safety_boundary": "出界",
    "velocity_safety_boundary": "超速",
    "timeout_without_settling": "超时未稳",
    "nonfinite_state": "数值发散",
    "numerical_warning": "数值警告",
}


def failure_label(case):
    """Short Chinese failure tag for figure titles (long English keys overlap panels)."""
    reason = case.get("failure_reason") or ""
    return SHORT_FAILURE.get(reason, reason or "未达标")


# --------------------------------------------------------------- BC objective
def bc_weight_at(update, updates, w_init, w_min=W_BC_MIN):
    """Linear anneal from w_init (first update) to w_min (last update).

    This is the DAPG schedule: the anchor is strongest at the start, when the
    policy is far from the teacher, and hands over to the policy gradient.
    """
    if updates < 1:
        raise ValueError("updates must be >= 1")
    if w_init < 0.0 or w_min < 0.0 or w_min > w_init:
        raise ValueError("BC weights must satisfy 0 <= w_min <= w_init")
    fraction = update / (updates - 1) if updates > 1 else 0.0
    return float(w_init + (w_min - w_init) * fraction)


def bc_loss_and_gradient(policy, demo_obs, demo_actions):
    """BC mean squared error on demo (obs, action) pairs plus its parameter gradients.

    The BC term matches the policy mean head only: the exploration std and the
    value network receive no BC gradient. Returns (mse, gradients) with
    gradients ordered like [*trunk.weights, *trunk.biases].
    """
    demo_obs = np.asarray(demo_obs, dtype=float)
    demo_actions = np.asarray(demo_actions, dtype=float)
    if demo_obs.ndim != 2 or demo_obs.shape[1] != STATE_INPUTS:
        raise ValueError("demo observations must be a (N, 5) batch")
    if demo_actions.shape != (len(demo_obs),):
        raise ValueError("demo actions must align with the observations")
    if len(demo_obs) == 0:
        raise ValueError("the demo batch must be nonempty")
    mean_out, cache = policy.trunk.forward(demo_obs)
    residual = mean_out[:, 0] - demo_actions
    mse = float(np.mean(residual**2))
    grad_mean = (2.0 * residual / len(demo_obs)).reshape(-1, 1)
    grad_w, grad_b = policy.trunk.backward(cache, grad_mean)
    return mse, [*grad_w, *grad_b]


def dapg_losses_and_gradients(
    policy, value, obs, actions, old_logp, advantages, returns, config, w_bc, demo_obs, demo_actions
):
    """Lesson-29 PPO losses/gradients plus the (annealed) BC regularizer.

    The demonstrations enter the objective only - never the rollout buffer. With
    w_bc = 0 the returned gradients are the lesson-29 gradients bit for bit.
    """
    losses, gradients = ppo_losses_and_gradients(
        policy, value, obs, actions, old_logp, advantages, returns, config
    )
    bc_mse, bc_grads = bc_loss_and_gradient(policy, demo_obs, demo_actions)
    if not all(np.isfinite(g).all() for g in bc_grads) or not np.isfinite(bc_mse):
        raise ValueError("Nonfinite BC loss or gradients; training diverged")
    if w_bc > 0.0:
        n_policy_arrays = 2 * (len(config.hidden) + 1)
        for index in range(n_policy_arrays):
            gradients[index] = gradients[index] + w_bc * bc_grads[index]
        losses["total"] = losses["total"] + w_bc * bc_mse
    losses["bc"] = bc_mse
    if not np.isfinite(losses["total"]):
        raise ValueError("Nonfinite DAPG loss; training diverged")
    return losses, gradients


# ------------------------------------------------------------- demonstrations
def generate_demonstrations(
    design,
    reference,
    dt,
    *,
    count=DEMO_COUNT,
    master_seed=0,
    horizon=EVAL_EPISODE_STEPS,
    angle_jitter=DEMO_ANGLE_JITTER,
    omega_jitter=DEMO_OMEGA_JITTER,
    cart_jitter=DEMO_CART_JITTER,
):
    """D successful lesson-7 rollouts from (lightly jittered) resting-down starts.

    The teacher is the unchanged lesson-7 HybridSwingupController. The jitter
    exists to make the D demonstrations distinct (the controller is
    deterministic); it stays far inside the basin where the kick/capture logic
    operates (|x| < 0.65 m, |down-angle offset| < 0.45 rad). Quality gate: every
    demonstration must pass the lesson-7 acceptance (recovered, no physical
    failure) - the teacher quality is known 20/20 - otherwise this raises.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    demos = []
    for index in range(count):
        rng = np.random.default_rng([int(master_seed), SEED_OFFSET_DEMO, index])
        start = down_start_state(reference)
        start[0] += float(rng.uniform(-cart_jitter, cart_jitter))
        start[1] += float(rng.uniform(-angle_jitter, angle_jitter))
        start[3] += float(rng.uniform(-omega_jitter, omega_jitter))
        env = make_swingup_environment(max_episode_steps=horizon)
        try:
            env.reset(seed=index)
            env.unwrapped.set_state(start[:2], start[2:])
            env.unwrapped.data.qfrc_applied[0] = 0.0
            controller = HybridSwingupController(env.unwrapped.model, design)
            states, controls, modes = [start.copy()], [], []
            terminated = truncated = False
            failure_reason = ""
            for _ in range(horizon):
                action = controller.action(states[-1])
                state, _, terminated, truncated, info = env.step(action)
                controls.append(float(action[0]))
                modes.append(controller.mode)
                states.append(np.asarray(state, dtype=float).copy())
                failure_reason = info["failure_reason"]
                if terminated or truncated:
                    break
        finally:
            env.close()
        arrays = {
            "states": np.asarray(states, dtype=float),
            "controls": np.asarray(controls, dtype=float),
            "applied_force_n": np.zeros(len(controls)),
            "scheduled_force_n": np.zeros(horizon),
            "end_flags": np.array([terminated, truncated]),
            "modes": np.asarray(modes),
        }
        metrics = recovery_metrics(arrays, {"failure_reason": failure_reason}, reference, dt)
        demos.append(
            {
                "index": index,
                "start_state": start,
                "states": arrays["states"],
                "controls": arrays["controls"],
                "settled_at_s": metrics["settled_at_s"],
                "success": bool(metrics["recovered"] and not metrics["terminated"]),
                "failure_reason": failure_reason,
            }
        )
    failed = [demo["index"] for demo in demos if not demo["success"]]
    if failed:
        raise ValueError(
            f"demonstration quality gate failed for demos {failed} "
            "(the lesson-7 teacher is expected to succeed on every airdrop rollout)"
        )
    return demos


def demonstration_pairs(demos, reference):
    """(obs, action) BC dataset: states[k] precedes action k (lesson-7 alignment)."""
    obs_list, action_list = [], []
    for demo in demos:
        states, controls = demo["states"], demo["controls"]
        for step in range(len(controls)):
            obs_list.append(normalize_observation(states[step], reference))
            action_list.append(float(controls[step]))
    return np.asarray(obs_list, dtype=float), np.asarray(action_list, dtype=float)


def demonstration_hash(demos):
    """SHA-256 over every demo state/control array (loader cross-check)."""
    digest = hashlib.sha256()
    for demo in demos:
        digest.update(np.ascontiguousarray(demo["start_state"]).tobytes())
        digest.update(np.ascontiguousarray(demo["states"]).tobytes())
        digest.update(np.ascontiguousarray(demo["controls"]).tobytes())
    return digest.hexdigest()


def demonstrations_summary(demos):
    settled = [demo["settled_at_s"] for demo in demos if demo["settled_at_s"] is not None]
    return {
        "count": len(demos),
        "successes": int(sum(demo["success"] for demo in demos)),
        "median_settled_at_s": float(np.median(settled)) if settled else None,
        "settled_times_s": [demo["settled_at_s"] for demo in demos],
    }


# ----------------------------------------------------------------- the guard
def make_training_env(reward, *, config, base_seed):
    """The single environment construction path used by run_experiment.

    Pinned by the guard: it must forward to the lesson-29 VecSwingup with the
    lesson-29 curriculum unchanged - the demonstrations touch the objective
    only, never the environment or the reward.
    """
    return VecSwingup(
        reward,
        n_envs=config.n_envs,
        episode_steps=config.train_episode_steps,
        base_seed=base_seed,
        task_envs=config.task_envs,
    )


def w_bc_zero_pipeline_guard(reward, config, steps=40):
    """w_BC = 0 guard, environment half: the stream IS the lesson-29 stream.

    Under one identical action stream (covering training-style steps and internal
    resets) the experiment's own factory and a directly constructed lesson-29
    VecSwingup must emit bitwise the same states, observations and rewards.
    """
    base_seed = 4242
    dapg_env = make_training_env(reward, config=config, base_seed=base_seed)
    plain = VecSwingup(
        reward,
        n_envs=config.n_envs,
        episode_steps=config.train_episode_steps,
        base_seed=base_seed,
        task_envs=config.task_envs,
    )
    try:
        actions = np.random.default_rng([7, 31]).uniform(-3.0, 3.0, (steps, config.n_envs))
        bitwise_obs = bool(np.array_equal(dapg_env.reset(), plain.reset()))
        bitwise_states, bitwise_rewards = True, True
        for chunk in actions:
            out_plain = plain.step(chunk)
            out_dapg = dapg_env.step(chunk)
            bitwise_states &= all(
                np.array_equal(a, b) for a, b in zip(out_plain, out_dapg, strict=True)
            )
            bitwise_rewards &= bool(np.array_equal(out_plain[1], out_dapg[1]))
        return {
            "w_bc": 0.0,
            "claim": (
                "with w_BC = 0 the DAPG experiment environment must emit bitwise the same "
                "states, observations and rewards as the lesson-29 pipeline under one "
                "identical action stream (the demos touch the objective, never the env)"
            ),
            "steps": int(steps),
            "bitwise_identical_rewards": bool(bitwise_rewards),
            "bitwise_identical_states": bool(bitwise_states),
            "bitwise_identical_observations": bool(bitwise_obs),
        }
    finally:
        dapg_env.close()
        plain.close()


def w_bc_zero_training_guard(reward, *, updates=3):
    """w_BC = 0 guard, trainer half: micro training equals lesson-29 train_ppo bitwise.

    Same vec env (hence the same curriculum jitter stream), same seeds, same
    config; the BC batch is synthetic because at w_BC = 0 it must not matter.
    """
    config = PPOConfig(
        n_envs=2,
        rollout_steps=16,
        updates=int(updates),
        epochs=2,
        minibatch=16,
        train_episode_steps=32,
        eval_every=int(updates),
        task_envs=1,
    )
    rng = np.random.default_rng([7, 32])
    demo_obs = rng.normal(0.0, 1.0, (16, STATE_INPUTS))
    demo_actions = rng.uniform(-CONTROL_LIMIT, CONTROL_LIMIT, 16)

    def one(trainer, **kwargs):
        vec = VecSwingup(
            reward,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=4242,
            task_envs=config.task_envs,
        )
        try:
            return trainer(
                vec,
                config=config,
                init_seed=[0, SEED_OFFSET_INIT, 0],
                act_seed=[0, SEED_OFFSET_ACT, 0],
                shuffle_seed=[0, SEED_OFFSET_SHUFFLE, 0],
                **kwargs,
            )
        finally:
            vec.close()

    plain = one(train_ppo)
    dapg = one(
        lambda vec, **kw: train_dapg(
            vec,
            demo_obs,
            demo_actions,
            demo_seed=[0, SEED_OFFSET_DEMO_SAMPLE, 0],
            w_bc_init=0.0,
            w_min=0.0,
            **kw,
        )
    )
    return {
        "claim": (
            "with w_BC = 0 the DAPG trainer must reproduce the lesson-29 trainer bitwise "
            "(same reward and value-loss curves, same policy weights)"
        ),
        "updates": int(updates),
        "bitwise_identical_reward_curve": bool(
            np.array_equal(plain["reward_curve"], dapg["reward_curve"])
        ),
        "bitwise_identical_value_loss_curve": bool(
            np.array_equal(plain["value_loss_curve"], dapg["value_loss_curve"])
        ),
        "bitwise_identical_policy_weights": all(
            np.array_equal(a, b)
            for a, b in zip(
                plain["policy"].trunk.weights, dapg["policy"].trunk.weights, strict=True
            )
        ),
    }


# ------------------------------------------------------------------- trainer
def train_dapg(
    vec_env,
    demo_obs,
    demo_actions,
    *,
    config,
    init_seed,
    act_seed,
    shuffle_seed,
    demo_seed,
    w_bc_init,
    w_min=W_BC_MIN,
    eval_hook=None,
    log=None,
):
    """One DAPG training run: the lesson-29 loop plus the annealed BC term.

    Every gradient step draws a fresh demo minibatch (dedicated seeded RNG) and
    adds w_BC(update) * grad MSE(mu(s_demo), a_demo) to the policy parameters;
    the demonstrations never enter the rollout buffer. bc_curve records the raw
    BC MSE per update (is the demo remembered or forgotten?).
    """
    if len(demo_obs) != len(demo_actions) or len(demo_obs) == 0:
        raise ValueError("demonstration pairs must be nonempty and aligned")
    if w_bc_init < 0.0 or w_min < 0.0 or w_min > w_bc_init:
        raise ValueError("BC weights must satisfy 0 <= w_min <= w_bc_init")
    policy = GaussianPolicy(STATE_INPUTS, config.hidden, init_seed, config.log_std_init)
    value = MLPTower(STATE_INPUTS, config.hidden, 1, [*init_seed, 1])
    parameters = [*policy.parameters(), *value.weights, *value.biases]
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    action_rng = np.random.default_rng(act_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    demo_rng = np.random.default_rng(demo_seed)
    size = config.n_envs * config.rollout_steps

    reward_curve = np.empty(config.updates)
    bc_curve = np.empty(config.updates)
    terminated_curve = np.empty(config.updates)
    value_loss_curve = np.empty(config.updates)
    entropy_curve = np.empty(config.updates)
    clip_fraction_curve = np.empty(config.updates)
    eval_steps, eval_records = [], []
    observations = vec_env.reset()
    started = time.perf_counter()

    for update in range(config.updates):
        w_bc = bc_weight_at(update, config.updates, w_bc_init, w_min)
        rollout = collect_rollout(vec_env, policy, value, config, action_rng, observations)
        observations = rollout["observations"]
        advantages, returns = compute_gae(
            rollout["rewards"].reshape(config.rollout_steps, config.n_envs) * config.reward_scale,
            rollout["values"].reshape(config.rollout_steps, config.n_envs),
            rollout["terminated"].reshape(config.rollout_steps, config.n_envs),
            rollout["truncated"].reshape(config.rollout_steps, config.n_envs),
            rollout["terminal_values"].reshape(config.rollout_steps, config.n_envs),
            config.gamma,
            config.gae_lambda,
        )
        advantages = standardize(advantages.ravel())
        returns = returns.ravel()
        order = shuffle_rng.permutation(size)

        value_loss_sum, clip_fraction_sum, bc_sum, grad_steps = 0.0, 0.0, 0.0, 0
        for _epoch in range(config.epochs):
            for start in range(0, size, config.minibatch):
                minibatch = order[start : start + config.minibatch]
                demo_idx = demo_rng.integers(0, len(demo_obs), size=min(DEMO_BATCH, len(demo_obs)))
                losses, gradients = dapg_losses_and_gradients(
                    policy,
                    value,
                    rollout["obs"][minibatch],
                    rollout["actions"][minibatch],
                    rollout["logp"][minibatch],
                    advantages[minibatch],
                    returns[minibatch],
                    config,
                    w_bc,
                    demo_obs[demo_idx],
                    demo_actions[demo_idx],
                )
                clip_gradients_(gradients, config.grad_clip)
                optimizer.step(parameters, gradients)
                value_loss_sum += losses["value"]
                clip_fraction_sum += losses["clip_fraction"]
                bc_sum += losses["bc"]
                grad_steps += 1

        reward_curve[update] = rollout["reward_mean"]
        bc_curve[update] = bc_sum / grad_steps
        terminated_curve[update] = rollout["terminated_frac"]
        value_loss_curve[update] = value_loss_sum / grad_steps
        entropy_curve[update] = gaussian_entropy(policy.log_std)
        clip_fraction_curve[update] = clip_fraction_sum / grad_steps
        optimizer.lr = config.lr * (1.0 - (update + 1) / config.updates)

        env_steps = (update + 1) * size
        if eval_hook is not None and (
            (update + 1) % config.eval_every == 0 or update == config.updates - 1
        ):
            eval_steps.append(env_steps)
            eval_records.append(eval_hook(policy, env_steps))
        if log is not None and (update + 1) % 25 == 0:
            message = (
                f"update {update + 1}/{config.updates}, steps {env_steps}, "
                f"reward {rollout['reward_mean']:.2f}, w_bc {w_bc:.3f}, "
                f"bc {bc_curve[update]:.3f}, term {rollout['terminated_frac']:.2f}"
            )
            if eval_records:
                message += f", eval {eval_records[-1]}"
            log(message)
    return {
        "policy": policy,
        "value": value,
        "reward_curve": reward_curve,
        "bc_curve": bc_curve,
        "terminated_curve": terminated_curve,
        "value_loss_curve": value_loss_curve,
        "entropy_curve": entropy_curve,
        "clip_fraction_curve": clip_fraction_curve,
        "eval_steps": np.asarray(eval_steps, dtype=int),
        "eval_records": eval_records,
        "env_steps": int(config.updates * size),
        "wall_time_s": time.perf_counter() - started,
    }


# ------------------------------------------------------- process metrics
def first_arrival_index(states, reference, capture_angle=CAPTURE_ANGLE_RAD):
    """First index k with |alpha| <= capture_angle in the stored state sequence."""
    states = np.asarray(states, dtype=float)
    alphas = wrap_angle(states[:, 1] - reference[1])
    finite = np.isfinite(alphas)
    hits = np.flatnonzero(finite & (np.abs(alphas) <= capture_angle))
    return int(hits[0]) if len(hits) else None


def first_arrival_time_s(states, reference, dt, capture_angle=CAPTURE_ANGLE_RAD):
    index = first_arrival_index(states, reference, capture_angle)
    return float(index * dt) if index is not None else None


def first_checkpoint_steps(result, predicate):
    """First periodic-eval env-steps whose record satisfies the predicate."""
    return next(
        (
            int(step)
            for step, point in zip(result["eval_steps"], result["eval_records"], strict=True)
            if predicate(point)
        ),
        None,
    )


def first_success_eval_steps(result):
    return first_checkpoint_steps(result, lambda point: bool(point["success"]))


def first_arrival_eval_steps(result):
    return first_checkpoint_steps(result, lambda point: point["first_arrival_s"] is not None)


def arrival_summary(episodes):
    """First-arrival statistics of the upright capture region (the process metric)."""
    times = [episode["first_arrival_s"] for episode in episodes]
    arrived = [value for value in times if value is not None]
    return {
        "episodes": len(episodes),
        "episodes_with_arrival": len(arrived),
        "arrival_fraction": len(arrived) / len(episodes) if episodes else 0.0,
        "median_first_arrival_s": float(np.median(arrived)) if arrived else None,
        "first_arrival_s_per_episode": times,
    }


def bc_decay_summary(per_seed_records):
    """BC loss first/last per seed: is the demonstration remembered or forgotten?"""
    rows = []
    for record in per_seed_records:
        first, last = record["bc_loss_first"], record["bc_loss_last"]
        rows.append(
            {
                "seed_index": record["seed_index"],
                "first": first,
                "last": last,
                "last_over_first": last / first if first > 1e-12 else None,
            }
        )
    ratios = [row["last_over_first"] for row in rows if row["last_over_first"] is not None]
    return {
        "per_seed": rows,
        "mean_last_over_first": float(np.mean(ratios)) if ratios else None,
    }


# ---------------------------------------------------------------- evaluation
def evaluate_dapg_policy(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start + upright arrivals."""
    episodes = evaluate_policy(policy, reward, reference, dt, master_seed=master_seed, count=count)
    for episode in episodes:
        episode["first_arrival_s"] = first_arrival_time_s(
            episode["arrays"]["states"], reference, dt
        )
    return episodes


def deterministic_dapg_episode(policy, reward, reference, dt):
    """One mean-action episode from the exact down start with full bookkeeping."""
    arrays, reason = run_policy_episode(
        policy, reward, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
    )
    metrics = episode_metrics(arrays, reason, reference, dt)
    record = {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "return": float(episode_rewards(arrays, reward).sum()),
        "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
        "failure_reason": reason,
    }
    return record, arrays


def default_training_config(updates=250):
    """Lesson-29 hyperparameters and budget, unchanged."""
    updates = int(updates)
    if not 1 <= updates <= 1000:
        raise ValueError("updates must be in [1, 1000]")
    return PPOConfig(updates=updates, eval_every=min(EVAL_EVERY, updates))


def choose_featured_level(entries):
    """Deterministic figure pick: most successes, then most arrivals, then earliest."""
    best, best_key = 0, None
    for index, entry in enumerate(entries):
        settled = entry["stochastic"]["median_settled_at_s"]
        settled_key = -settled if settled is not None else float("-inf")
        key = (
            entry["stochastic"]["successes"],
            entry["arrival"]["episodes_with_arrival"],
            settled_key,
        )
        if best_key is None or key > best_key:
            best, best_key = index, key
    return best


def annotate_case_w_bc(cases, eval_episodes, push_episodes):
    """Copy the sweep level onto each featured case (fixed scan order mirror)."""
    pools = {"eval_failure": eval_episodes, "push_failure": push_episodes}
    for case in cases:
        for episode in pools[case["kind"]]:
            if episode["terminated"] or not episode["recovered"]:
                case["w_bc"] = episode["w_bc"]
                break
    return cases


# ------------------------------------------------------------- run experiment
def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    train_seeds=TRAIN_SEEDS,
    eval_seed_count=EVAL_SEEDS,
    w_bc_levels=W_BC_LEVELS,
    demo_count=DEMO_COUNT,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    if not isinstance(demo_count, int) or not 1 <= demo_count <= 100:
        raise ValueError("demo_count must be an integer in [1, 100]")
    levels = tuple(float(value) for value in w_bc_levels)
    if not levels or any(value <= 0.0 for value in levels):
        raise ValueError("w_BC levels must be positive (w_BC = 0 is the guard, not a tier)")
    config = config or default_training_config()
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    if abs(design.actuator_gear - GEAR) > 1e-12:
        raise ValueError("actuator gear disagrees with the recovery_metrics convention")
    reward = RewardFunction(reference)

    demos = generate_demonstrations(design, reference, dt, count=demo_count, master_seed=seed)
    demo_obs, demo_actions = demonstration_pairs(demos, reference)
    demo_summary = demonstrations_summary(demos)

    guard = {
        "pipeline": w_bc_zero_pipeline_guard(reward, config),
        "training": w_bc_zero_training_guard(reward),
    }

    # The exact-down-start lesson-7 run is both the baseline of record and the
    # teacher trajectory for the same-initial-state comparison in the figures.
    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    plans = make_push_plans(dt, eval_seed_count, seed)
    baseline_push_records, baseline_push_states, baseline_push_controls = baseline_push_evaluations(
        design, plans, EVAL_EPISODE_STEPS
    )

    output.mkdir(parents=True, exist_ok=False)
    entries = []
    policy_payloads = {}  # (level_index, seed_index) -> policy arrays
    reward_curves, bc_curves = {}, {}
    det_store = {}  # (level_index, seed_index) -> deterministic episode arrays
    eval_stores = {index: [] for index in range(len(levels))}
    push_stores = {index: [] for index in range(len(levels))}
    all_eval_episodes, all_push_episodes = [], []

    for level_index, w_bc in enumerate(levels):
        per_seed_records, det_records = [], []
        for seed_index in range(train_seeds):
            vec_env = make_training_env(
                reward,
                config=config,
                base_seed=10_000 + seed * 1000 + level_index * 100 + seed_index,
            )

            def eval_hook(policy, _env_steps, _reward=reward, _reference=reference, _dt=dt):
                record, _arrays = deterministic_dapg_episode(policy, _reward, _reference, _dt)
                return {
                    "success": bool(record["recovered"] and not record["terminated"]),
                    "settled_at_s": record["settled_at_s"],
                    "return": record["return"],
                    "first_arrival_s": record["first_arrival_s"],
                }

            result = train_dapg(
                vec_env,
                demo_obs,
                demo_actions,
                config=config,
                init_seed=[seed, SEED_OFFSET_INIT + level_index, seed_index],
                act_seed=[seed, SEED_OFFSET_ACT + level_index, seed_index],
                shuffle_seed=[seed, SEED_OFFSET_SHUFFLE + level_index, seed_index],
                demo_seed=[seed, SEED_OFFSET_DEMO_SAMPLE + level_index, seed_index],
                w_bc_init=w_bc,
                w_min=W_BC_MIN,
                eval_hook=eval_hook,
                log=log,
            )
            vec_env.close()
            policy = result["policy"]
            policy_payloads[(level_index, seed_index)] = policy.arrays()
            reward_curves[(level_index, seed_index)] = result["reward_curve"]
            bc_curves[(level_index, seed_index)] = result["bc_curve"]

            eval_episodes = evaluate_dapg_policy(
                policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
            )
            det_record, det_arrays = deterministic_dapg_episode(policy, reward, reference, dt)
            det_store[(level_index, seed_index)] = det_arrays
            push_episodes = []
            for plan in plans:
                arrays, reason = run_policy_episode(
                    policy,
                    reward,
                    reference,
                    horizon=EVAL_EPISODE_STEPS,
                    env_seed=plan["index"],
                    deterministic=True,
                    schedule=push_schedule(plan, dt, EVAL_EPISODE_STEPS),
                )
                metrics = episode_metrics(arrays, reason, reference, dt)
                push_episodes.append(
                    {
                        "plan_index": plan["index"],
                        "force_n": plan["force_n"],
                        "start_s": plan["start_s"],
                        **select_metrics(metrics, PUSH_FIELDS),
                        "return": float(episode_rewards(arrays, reward).sum()),
                        "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
                        "arrays": arrays,
                    }
                )

            all_eval_episodes.extend({**episode, "w_bc": w_bc} for episode in eval_episodes)
            all_push_episodes.extend({**episode, "w_bc": w_bc} for episode in push_episodes)
            det_records.append(det_record)
            eval_stores[level_index].append(
                {
                    "states": [episode["arrays"]["states"] for episode in eval_episodes],
                    "controls": [episode["arrays"]["controls"] for episode in eval_episodes],
                    "lengths": [len(episode["arrays"]["states"]) for episode in eval_episodes],
                    "terminated": [bool(episode["terminated"]) for episode in eval_episodes],
                    "settled": [
                        float(episode["settled_at_s"])
                        if episode["settled_at_s"] is not None
                        else np.nan
                        for episode in eval_episodes
                    ],
                    "returns": [float(episode["return"]) for episode in eval_episodes],
                    "arrival": [
                        float(episode["first_arrival_s"])
                        if episode["first_arrival_s"] is not None
                        else np.nan
                        for episode in eval_episodes
                    ],
                    "peaks": [
                        float(episode["peak_abs_motor_force_n"]) for episode in eval_episodes
                    ],
                    "max_x": [
                        float(episode["max_abs_cart_position_m"]) for episode in eval_episodes
                    ],
                }
            )
            push_stores[level_index].append(
                {
                    "states": [episode["arrays"]["states"] for episode in push_episodes],
                    "lengths": [len(episode["arrays"]["states"]) for episode in push_episodes],
                    "recovery": [
                        float(episode["recovery_after_push_end_s"])
                        if episode["recovery_after_push_end_s"] is not None
                        else np.nan
                        for episode in push_episodes
                    ],
                }
            )
            final_window = slice(max(0, config.updates - 10), config.updates)
            record = {
                "seed_index": seed_index,
                "env_steps": result["env_steps"],
                "wall_time_s": result["wall_time_s"],
                "final_reward_mean": float(np.mean(result["reward_curve"][final_window])),
                "final_log_std": float(policy.log_std[0]),
                "final_w_bc": bc_weight_at(config.updates - 1, config.updates, w_bc, W_BC_MIN),
                "bc_loss_first": float(result["bc_curve"][0]),
                "bc_loss_last": float(result["bc_curve"][-1]),
                "first_successful_eval_steps": first_success_eval_steps(result),
                "first_arrival_eval_steps": first_arrival_eval_steps(result),
                "eval_curve": [
                    {"env_steps": int(step), **point}
                    for step, point in zip(
                        result["eval_steps"], result["eval_records"], strict=True
                    )
                ],
                "stochastic": summarize_episodes(eval_episodes),
                "deterministic": det_record,
                "push": summarize_episodes(push_episodes),
            }
            per_seed_records.append(record)
            if log is not None:
                arrived = sum(
                    1 for episode in eval_episodes if episode["first_arrival_s"] is not None
                )
                log(
                    f"w_bc {w_bc:g} seed {seed_index}: steps {record['env_steps']}, "
                    f"wall {record['wall_time_s']:.1f}s, "
                    f"stoch {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                    f"upright arrival {arrived}/{len(eval_episodes)}, "
                    f"bc {record['bc_loss_first']:.3f}->{record['bc_loss_last']:.3f}, "
                    f"det settled {det_record['settled_at_s']}, "
                    f"det arrival {det_record['first_arrival_s']}"
                )

        settled_all = [
            value
            for seed_store in eval_stores[level_index]
            for value in seed_store["settled"]
            if not np.isnan(value)
        ]
        peaks_all = [
            value for seed_store in eval_stores[level_index] for value in seed_store["peaks"]
        ]
        level_eval_episodes = all_eval_episodes[-eval_seed_count * train_seeds :]
        successes = [record["stochastic"]["successes"] for record in per_seed_records]
        stochastic_summary = {
            "episodes": eval_seed_count * train_seeds,
            "successes": int(sum(successes)),
            "success_rate": float(np.mean(successes) / eval_seed_count),
            "successes_per_seed": successes,
            "median_settled_at_s": float(np.median(settled_all)) if settled_all else None,
            "median_peak_abs_motor_force_n": float(np.median(peaks_all)),
        }
        push_summary = {
            "episodes": eval_seed_count * train_seeds,
            "successes": int(sum(record["push"]["successes"] for record in per_seed_records)),
            "successes_per_seed": [record["push"]["successes"] for record in per_seed_records],
            "recovery_times_s": [
                None if np.isnan(value) else float(value)
                for seed_store in push_stores[level_index]
                for value in seed_store["recovery"]
            ],
        }
        entries.append(
            {
                "w_bc": w_bc,
                "training": per_seed_records,
                "deterministic": det_records,
                "stochastic": stochastic_summary,
                "push": push_summary,
                "arrival": arrival_summary(level_eval_episodes),
                "bc_decay": bc_decay_summary(per_seed_records),
                "first_success": {
                    "any": any(
                        record["first_successful_eval_steps"] is not None
                        for record in per_seed_records
                    ),
                    "per_seed_eval_steps": [
                        record["first_successful_eval_steps"] for record in per_seed_records
                    ],
                },
            }
        )

    failure_cases = annotate_case_w_bc(
        pick_failure_cases(all_eval_episodes, all_push_episodes),
        all_eval_episodes,
        all_push_episodes,
    )
    featured = choose_featured_level(entries)
    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        levels=levels,
        plans=plans,
        demos=demos,
        demo_obs_count=len(demo_obs),
        demo_summary=demo_summary,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        baseline_push_records=baseline_push_records,
        guard=guard,
        entries=entries,
        featured=featured,
        failure_cases=failure_cases,
        all_eval_episodes=all_eval_episodes,
        all_push_episodes=all_push_episodes,
        elapsed=elapsed,
    )
    archive = build_archive(
        demos=demos,
        reward_curves=reward_curves,
        bc_curves=bc_curves,
        policy_payloads=policy_payloads,
        det_store=det_store,
        eval_stores=eval_stores,
        push_stores=push_stores,
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        baseline_push_states=baseline_push_states,
        baseline_push_controls=baseline_push_controls,
        failure_cases=failure_cases,
    )
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_training_curves(output / "training_curves.png", report, output)
    save_airdrop(output / "airdrop_analysis.png", report, output)
    save_three_way(output / "three_way.png", report, output)
    return report


def teacher_first_arrival(baseline_states, reference, dt):
    return first_arrival_time_s(baseline_states, reference, dt)


def build_report(
    *,
    seed,
    config,
    design,
    train_seeds,
    eval_seed_count,
    levels,
    plans,
    demos,
    demo_obs_count,
    demo_summary,
    baseline_records,
    baseline_identical,
    baseline_push_records,
    guard,
    entries,
    featured,
    failure_cases,
    all_eval_episodes,
    all_push_episodes,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    baseline_push_summary = summarize_episodes(baseline_push_records)
    three_way = [
        {
            "label": "基线（第 7 课能量整形+LQR，零样本）",
            "episodes": baseline_summary["episodes"],
            "successes": baseline_summary["successes"],
            "median_settled_at_s": baseline_summary["median_settled_at_s"],
            "median_peak_abs_motor_force_n": baseline_summary["median_peak_abs_motor_force_n"],
            "source": "this record",
        },
        {
            "label": "纯 PPO（第 29 课，只凭奖励）",
            "episodes": LESSON29_PPO_REFERENCE["episodes"],
            "successes": LESSON29_PPO_REFERENCE["successes"],
            "median_settled_at_s": LESSON29_PPO_REFERENCE["median_settled_at_s"],
            "median_peak_abs_motor_force_n": LESSON29_PPO_REFERENCE[
                "median_peak_abs_motor_force_n"
            ],
            "source": LESSON29_PPO_REFERENCE["source"],
        },
        {
            "label": "PBRS（cE=0.5，第 31 课）",
            "episodes": PBRS_REFERENCE["episodes"],
            "successes": PBRS_REFERENCE["successes"],
            "median_settled_at_s": PBRS_REFERENCE["median_settled_at_s"],
            "median_peak_abs_motor_force_n": PBRS_REFERENCE["median_peak_abs_motor_force_n"],
            "source": PBRS_REFERENCE["source"],
        },
        {
            "label": "PBRS（cE=2，第 31 课）",
            "episodes": PBRS_REFERENCE["episodes"],
            "successes": PBRS_REFERENCE["successes"],
            "median_settled_at_s": PBRS_REFERENCE["median_settled_at_s"],
            "median_peak_abs_motor_force_n": PBRS_REFERENCE["median_peak_abs_motor_force_n"],
            "source": PBRS_REFERENCE["source"],
        },
    ]
    for entry in entries:
        three_way.append(
            {
                "label": f"示教空投（w={entry['w_bc']:g}）",
                "episodes": entry["stochastic"]["episodes"],
                "successes": entry["stochastic"]["successes"],
                "median_settled_at_s": entry["stochastic"]["median_settled_at_s"],
                "median_peak_abs_motor_force_n": entry["stochastic"][
                    "median_peak_abs_motor_force_n"
                ],
                "source": "this record",
            }
        )
    demo_settled = [demo["settled_at_s"] for demo in demos if demo["settled_at_s"] is not None]
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
                "airdrop the teacher (DAPG-style RL from demonstrations): the lesson-29 pure "
                "PPO stack is reused verbatim and only the objective gains a BC regularizer "
                "on lesson-7 teacher pairs; sister of lesson 30's cable car and lesson 31's "
                "ladder - directly aimed at the last mile (capture + settle)"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "bc_objective": {
                "formula": "L = L_PPO + w_BC * MSE(mean_theta(s_demo), a_demo)",
                "w_bc_levels": list(levels),
                "w_bc_min": W_BC_MIN,
                "annealing": (
                    "w_BC(update) linear from w_init (first update) to w_min (last update) "
                    "over the update count (DAPG convention)"
                ),
                "demo_batch_pairs": DEMO_BATCH,
                "replay_pool": (
                    "demonstrations never enter the rollout/replay buffer; they only feed "
                    "the BC loss on their stored (s, a) pairs (DAPG style)"
                ),
                "gradient": (
                    "the BC term matches the policy mean head only; the exploration std and "
                    "the value network receive no BC gradient"
                ),
                "q_filter": (
                    "NOT implemented (simplified DAPG): the original DAPG filters the BC "
                    "loss by the advantage sign; here the BC term pulls on every demo pair "
                    "- recorded as an honest deviation"
                ),
            },
            "demonstrations": {
                "source": (
                    "lesson-7 HybridSwingupController (energy shaping + LQR hysteresis), "
                    "zero-shot; teacher quality known 20/20 (docs/09)"
                ),
                "count": len(demos),
                "bc_pairs": demo_obs_count,
                "initial_state": (
                    "exact resting down start + small seeded jitter: angle +/-"
                    f"{DEMO_ANGLE_JITTER} rad, omega +/-{DEMO_OMEGA_JITTER} rad/s, cart +/-"
                    f"{DEMO_CART_JITTER} m (the controller is deterministic; the jitter makes "
                    "the D rollouts distinct)"
                ),
                "horizon_steps": EVAL_EPISODE_STEPS,
                "quality_gate": (
                    "every demonstration must pass the lesson-7 acceptance (recovered, no "
                    "physical failure); generation raises otherwise"
                ),
                "seed_stream": "default_rng([master, 4000, demo_index]) for the start jitter",
                "alignment": (
                    "states[k] precedes controls[k] (lesson-7 archive alignment); BC pairs "
                    "are (obs(s_k), a_k) with the lesson-29 observation scaling"
                ),
                "sha256": demonstration_hash(demos),
                "successes": demo_summary["successes"],
                "median_settled_at_s": demo_summary["median_settled_at_s"],
                "settled_times_s": demo_settled,
            },
            "reward": RewardFunction(reference).as_dict(),
            "reward_scale_for_learning": config.reward_scale,
            "train_episode_steps": config.train_episode_steps,
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * design.dt,
            "train_initial_state": {
                "note": (
                    "lesson-29 two-bank curriculum reused: half of the parallel environments "
                    "always restart from the exact resting down start, the rest randomize the "
                    "pole direction over the full circle and the initial angular velocity"
                ),
            },
            "eval_initial_state": "exact resting down start (reference angle -180 deg), no jitter",
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
                "first_arrival": (
                    "first episode step whose stored state satisfies the definition; lesson 29 "
                    "never arrived in evaluation, lesson 31 touched it once at a 150k "
                    "training checkpoint"
                ),
            },
            "first_success": (
                "first training checkpoint whose periodic evaluation (mean action, exact down "
                "start) passes the lesson-7 acceptance - the headline 0 -> 1 of this lesson"
            ),
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lesson 29",
            },
            "seed_streams": {
                "network_init": "default_rng([master, 7000 + level, train_seed]); value net [.., 1]",
                "action_sampling": "default_rng([master, 5000 + level, train_seed])",
                "minibatch_order": "default_rng([master, 9000 + level, train_seed])",
                "demo_minibatch": "default_rng([master, 4400 + level, train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*1000 "
                    "+ level*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lessons 29/30/31)",
            },
        },
        "hyperparameters": {**asdict(config), "hidden": list(config.hidden)},
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{eval_seed_count} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
            "per_episode": baseline_records,
        },
        "guard": {
            "w_bc": 0.0,
            "claim": (
                "with w_BC = 0 the DAPG pipeline must reproduce the lesson-29 pipeline bitwise - "
                "both through the environment stream (states / observations / rewards) and "
                "through micro training (curves and policy weights)"
            ),
            "pipeline": guard["pipeline"],
            "training": guard["training"],
        },
        "sweep": entries,
        "featured_level_index": featured,
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; DAPG policies never saw pushes during training; "
                    "the same plans run from the same exact down start; mean-action episodes"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": (
                    "the same plans are applied to the baseline and to every w_BC level and "
                    "training seed"
                ),
            },
            "plans": plans,
            "baseline": baseline_push_summary,
            "per_level": [
                {
                    "w_bc": entry["w_bc"],
                    "successes": entry["push"]["successes"],
                    "episodes": entry["push"]["episodes"],
                    "successes_per_seed": entry["push"]["successes_per_seed"],
                    "recovery_times_s": entry["push"]["recovery_times_s"],
                }
                for entry in entries
            ],
        },
        "three_way_comparison": three_way,
        "lesson29_reference": LESSON29_PPO_REFERENCE,
        "pbrs_reference": PBRS_REFERENCE,
        "hypothesis": {
            "claim": (
                "DAPG (Rajeswaran et al. RSS 2018) shows that a handful of demonstrations plus "
                "an auxiliary BC loss inside the policy gradient can solve hard manipulation; "
                "lesson 31 proved pure gradients can touch the cliff top but not hold it. "
                "Primary question: does anchoring the policy on the airdropped lesson-7 teacher "
                "produce the first accepted success (0 -> 1) within the fixed budget? A null "
                "result is recorded as the formal conclusion"
            ),
            "per_level": [
                {
                    "w_bc": entry["w_bc"],
                    "successes": entry["stochastic"]["successes"],
                    "episodes": entry["stochastic"]["episodes"],
                    "episodes_with_arrival": entry["arrival"]["episodes_with_arrival"],
                    "first_success_any": entry["first_success"]["any"],
                    "first_arrival_eval_steps_per_seed": [
                        record["first_arrival_eval_steps"] for record in entry["training"]
                    ],
                    "first_successful_eval_steps_per_seed": entry["first_success"][
                        "per_seed_eval_steps"
                    ],
                    "verdict": (
                        "first accepted success achieved (0 -> 1)"
                        if entry["first_success"]["any"]
                        else "airdrop did not produce a single accepted success within the budget"
                    ),
                }
                for entry in entries
            ],
        },
        "failure_analysis": {
            "eval_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in all_eval_episodes]
            ),
            "push_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in all_push_episodes]
            ),
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in failure_cases
            ],
        },
        "training": {
            "train_seeds": train_seeds,
            "levels": list(levels),
            "env_steps_per_seed": entries[0]["training"][0]["env_steps"]
            if entries and entries[0]["training"]
            else 0,
            "total_env_steps": sum(
                record["env_steps"] for entry in entries for record in entry["training"]
            ),
            "wall_time_s_total": elapsed,
            "curves_note": (
                "reward_curve_<level>_<seed> and bc_curve_<level>_<seed> live in trajectories.npz"
            ),
        },
        "limitations": [
            (
                "w_BC init in {10.0, 1.0} is a hand-picked two-point grid; the single new "
                "hidden hand-knob of this lesson (the BC anchor weight and its annealing "
                "schedule) was not tuned further."
            ),
            (
                "Simplified DAPG: the Q-filter (BC loss gated by advantage sign) is not "
                "implemented, and all demonstrations come from the single lesson-7 baseline "
                "controller - the anchor is one behavior, not a diverse teacher set."
            ),
            (
                "One task, one nominal MuJoCo model, no noise/delay/mass error; success "
                "requires the strict lesson-7 settled tail (0.02 m, 0.01 rad, 0.02 m/s, "
                "0.02 rad/s held >= 2 s)."
            ),
            (
                "The cited lesson-29/31 rows in three_way_comparison come from those official "
                "records (60 episodes each), not re-run here; w_BC = 0 is covered by the "
                "bitwise pipeline and training guards instead of a full re-run."
            ),
            (
                "Demonstrations are 30 s model-time rollouts of a feedback controller; BC on "
                "8 rollouts does not bound what a larger or more diverse demonstration set "
                "could achieve."
            ),
        ],
    }
    return report


def build_archive(
    *,
    demos,
    reward_curves,
    bc_curves,
    policy_payloads,
    det_store,
    eval_stores,
    push_stores,
    baseline_states,
    baseline_controls,
    baseline_push_states,
    baseline_push_controls,
    failure_cases,
):
    horizon = EVAL_EPISODE_STEPS
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=float),
        "baseline_controls": np.asarray(baseline_controls, dtype=float),
        "baseline_push_states": stack_trajectories(baseline_push_states, horizon)[0],
        "baseline_push_lengths": stack_trajectories(baseline_push_states, horizon)[1],
        "baseline_push_controls": stack_controls(baseline_push_controls, horizon),
        "demo_states": np.stack([demo["states"] for demo in demos], axis=0),
        "demo_controls": np.stack([demo["controls"] for demo in demos], axis=0),
        "demo_start_states": np.stack([demo["start_state"] for demo in demos], axis=0),
        "demo_settled_s": np.asarray([demo["settled_at_s"] for demo in demos], dtype=float),
    }
    for (level_index, seed_index), curve in reward_curves.items():
        archive[f"reward_curve_{level_index}_{seed_index}"] = curve
    for (level_index, seed_index), curve in bc_curves.items():
        archive[f"bc_curve_{level_index}_{seed_index}"] = curve
    for (level_index, seed_index), payload in policy_payloads.items():
        for name, array in payload.items():
            archive[f"policy_{level_index}_{seed_index}_{name}"] = array
    for (level_index, seed_index), arrays in det_store.items():
        archive[f"det_states_{level_index}_{seed_index}"] = arrays["states"]
        archive[f"det_controls_{level_index}_{seed_index}"] = arrays["controls"]
    for level_index, seed_stores in eval_stores.items():
        flat_states = [s for store in seed_stores for s in store["states"]]
        flat_controls = [c for store in seed_stores for c in store["controls"]]
        archive[f"eval_states_{level_index}"] = stack_trajectories(flat_states, horizon)[0]
        archive[f"eval_lengths_{level_index}"] = np.asarray(
            [n for store in seed_stores for n in store["lengths"]], dtype=int
        )
        archive[f"eval_controls_{level_index}"] = stack_controls(flat_controls, horizon)
        archive[f"eval_terminated_{level_index}"] = np.asarray(
            [t for store in seed_stores for t in store["terminated"]], dtype=bool
        )
        for name, suffix in (
            ("settled", "settled_s"),
            ("returns", "returns"),
            ("arrival", "first_arrival_s"),
            ("peaks", "peak_force_n"),
            ("max_x", "max_x_m"),
        ):
            archive[f"eval_{suffix}_{level_index}"] = np.asarray(
                [v for store in seed_stores for v in store[name]], dtype=float
            ).reshape(len(seed_stores), -1)
        push_seed_stores = push_stores[level_index]
        archive[f"push_states_{level_index}"] = stack_trajectories(
            [s for store in push_seed_stores for s in store["states"]], horizon
        )[0]
        archive[f"push_lengths_{level_index}"] = np.asarray(
            [n for store in push_seed_stores for n in store["lengths"]], dtype=int
        )
        archive[f"push_recovery_s_{level_index}"] = np.asarray(
            [v for store in push_seed_stores for v in store["recovery"]], dtype=float
        ).reshape(len(push_seed_stores), -1)
    for index, case in enumerate(failure_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by the summary (used by the demo loader)."""
    levels = len(report["sweep"])
    seeds = report["training"]["train_seeds"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
        "demo_states",
        "demo_controls",
        "demo_start_states",
        "demo_settled_s",
    }
    for level in range(levels):
        keys.update(f"reward_curve_{level}_{seed}" for seed in range(seeds))
        keys.update(f"bc_curve_{level}_{seed}" for seed in range(seeds))
        keys.update(
            f"policy_{level}_{seed}_{name}"
            for seed in range(seeds)
            for name in policy_array_names(hidden)
        )
        keys.update(
            f"det_{suffix}_{level}_{seed}"
            for seed in range(seeds)
            for suffix in ("states", "controls")
        )
        keys.update(
            f"eval_{suffix}_{level}"
            for suffix in (
                "states",
                "lengths",
                "controls",
                "terminated",
                "settled_s",
                "returns",
                "first_arrival_s",
                "peak_force_n",
                "max_x_m",
            )
        )
        keys.update(f"push_{suffix}_{level}" for suffix in ("states", "lengths", "recovery_s"))
    keys.update(
        f"case{index}_{suffix}"
        for index in range(len(report["failure_analysis"]["featured_cases"]))
        for suffix in ("states", "controls")
    )
    return keys


# -------------------------------------------------------------------- figures
def level_label(entry):
    return f"w={entry['w_bc']:g}"


def save_training_curves(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        rewards = [
            np.stack([data[f"reward_curve_{index}_{seed}"] for seed in range(seeds)], axis=0)
            for index in range(len(report["sweep"]))
        ]
        bcs = [
            np.stack([data[f"bc_curve_{index}_{seed}"] for seed in range(seeds)], axis=0)
            for index in range(len(report["sweep"]))
        ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
    colors = ("#2563eb", "#0f766e")
    lesson29_final = float(np.mean(LESSON29_PPO_REFERENCE["final_reward_mean_per_seed"]))
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        for seed_row in range(seeds):
            axes[0].plot(updates, rewards[index][seed_row], alpha=0.25, linewidth=0.8, color=color)
        axes[0].plot(
            updates,
            rewards[index].mean(axis=0),
            color=color,
            linewidth=1.8,
            label=f"{level_label(entry)}（{seeds} 种子均值）",
        )
    axes[0].axhline(
        lesson29_final,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.2,
        label="纯 PPO（第 29 课）末段均值",
    )
    axes[0].set(
        xlabel="PPO 更新轮次",
        ylabel="批内平均奖励（任务奖励，每步）",
        title="DAPG 训练奖励：细线 = 单个种子",
    )
    axes[0].legend(fontsize=7, loc="lower right")
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        success = np.asarray(
            [
                [int(point["success"]) for point in record["eval_curve"]]
                for record in entry["training"]
            ],
            dtype=float,
        )
        steps = (
            np.asarray(
                [point["env_steps"] for point in entry["training"][0]["eval_curve"]], dtype=float
            )
            / 1000.0
        )
        for seed_row in range(success.shape[0]):
            axes[1].plot(steps, success[seed_row], "o--", markersize=4, alpha=0.5, color=color)
        axes[1].plot(steps, success.mean(axis=0), "o-", color=color, label=level_label(entry))
    axes[1].set(
        xlabel="环境步数（×1000）",
        ylabel="下方初态验收通过（1=成功）",
        ylim=(-0.08, 1.08),
        yticks=[0, 0.5, 1],
        title="训练中周期评估：均值动作、下方初态",
    )
    axes[1].legend(fontsize=7, loc="upper left")
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        curves = np.maximum(bcs[index], 1e-8)
        for seed_row in range(seeds):
            axes[2].plot(updates, curves[seed_row], alpha=0.25, linewidth=0.8, color=color)
        axes[2].plot(
            updates,
            curves.mean(axis=0),
            color=color,
            linewidth=1.8,
            label=f"{level_label(entry)}（{seeds} 种子均值）",
        )
    axes[2].set(
        xlabel="PPO 更新轮次",
        ylabel="BC 损失 MSE（对数轴）",
        yscale="log",
        title="BC 项衰减：示教被记住还是被遗忘",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_airdrop(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    featured = report["featured_level_index"]
    entry = report["sweep"][featured]
    seed_index = 0
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        teacher = data["baseline_states"]
        det_states = data[f"det_states_{featured}_{seed_index}"]
        det_controls = data[f"det_controls_{featured}_{seed_index}"]
        demo_states = data["demo_states"]
        teacher_controls = data["baseline_controls"]
    demo = report["protocol"]["demonstrations"]
    det_record = entry["deterministic"][seed_index]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    axes[0, 0].plot(
        np.arange(len(teacher)) * dt,
        np.cos(teacher[:, 1] - ref_theta),
        "--",
        color="#64748b",
        label="教师（第 7 课基线，同初态）",
    )
    axes[0, 0].plot(
        np.arange(len(det_states)) * dt,
        np.cos(det_states[:, 1] - ref_theta),
        color="#2563eb",
        label=f"学得策略（{level_label(entry)} 种子 {seed_index}，均值动作）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    det_arrival = det_record["first_arrival_s"]
    det_settled = det_record["settled_at_s"]
    learned_note = (
        f"学得：首达 {det_arrival:.2f} s" if det_arrival is not None else "学得：未进入直立区"
    )
    if det_settled is not None:
        learned_note += f"，稳定 {det_settled:.2f} s"
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"同一下方初态对照（教师稳定 {report['baseline']['median_settled_at_s']:.2f} s；{learned_note}）",
    )
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].plot(
        np.arange(len(teacher)) * dt, teacher[:, 0], "--", color="#64748b", label="教师"
    )
    axes[0, 1].plot(
        np.arange(len(det_states)) * dt, det_states[:, 0], color="#2563eb", label="学得策略"
    )
    axes[0, 1].set(
        ylabel="小车位置（m）",
        xlabel="仿真时间（s）",
        title=f"小车位置（红点线 = ±{SAFE_CART_POSITION:g} m 失败边界）",
    )
    axes[0, 1].legend(fontsize=7)
    for demo_index in range(len(demo_states)):
        axes[1, 0].plot(
            np.arange(len(demo_states[demo_index])) * dt,
            np.cos(demo_states[demo_index, :, 1] - ref_theta),
            alpha=0.3,
            linewidth=0.8,
            color="#b45309",
        )
    axes[1, 0].plot(
        np.arange(len(teacher)) * dt,
        np.cos(teacher[:, 1] - ref_theta),
        color="#64748b",
        linewidth=1.6,
        label="教师参考（精确下方初态）",
    )
    axes[1, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[1, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"空投的 {demo['count']} 条示教（细线，全部通过验收）",
    )
    axes[1, 0].legend(fontsize=7, loc="lower right")
    edges_t = np.arange(len(teacher_controls) + 1) * dt
    edges_l = np.arange(len(det_controls) + 1) * dt
    axes[1, 1].stairs(
        teacher_controls * report["protocol"]["actuator_gear"],
        edges_t,
        color="#64748b",
        label="教师",
    )
    axes[1, 1].stairs(
        det_controls * report["protocol"]["actuator_gear"],
        edges_l,
        color="#2563eb",
        label="学得策略",
    )
    axes[1, 1].set(
        ylabel="电机力（N）",
        xlabel="仿真时间（s）",
        title="电机输入（±300 N 限幅）",
    )
    axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_three_way(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    colors = ("#64748b", "#b91c1c", "#7c3aed", "#9333ea", "#2563eb", "#0f766e")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    rows = report["three_way_comparison"]
    labels = ["基线", "纯PPO\n(29课)", "PBRS\ncE=0.5", "PBRS\ncE=2"] + [
        f"示教\n{level_label(entry)}" for entry in report["sweep"]
    ]
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
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
    axes[0, 0].set(
        ylabel="验收通过率（%）",
        ylim=(0, 112),
        title="三方+示教两档成功率（第 7 课口径）",
    )
    axes[0, 0].tick_params(axis="x", labelsize=7)

    lines = ["直立首达（|α|≤0.3 rad）与首次成功：", ""]
    for entry in report["sweep"]:
        arrival = entry["arrival"]
        median = arrival["median_first_arrival_s"]
        median_text = f"{median:.2f} s" if median is not None else "—"
        lines.append(
            f"  {level_label(entry)}：直立首达 "
            f"{arrival['episodes_with_arrival']}/{arrival['episodes']}（中位 {median_text}）"
        )
        for record in entry["training"]:
            first_arr = record["first_arrival_eval_steps"]
            first_ok = record["first_successful_eval_steps"]
            arr_text = f"{first_arr / 1000:.0f}k 步" if first_arr is not None else "从未"
            ok_text = f"{first_ok / 1000:.0f}k 步" if first_ok is not None else "从未"
            lines.append(
                f"    种子 {record['seed_index']}：检查点首达 {arr_text}，首次成功 {ok_text}"
            )
    lines.append("")
    lines.append("对照：第 29 课评估从未首达；第 31 课训练检查点一次（150k 步）")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
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
    axes[0, 1].set(title="直立首达与首次成功（0→1 是否出现）")

    baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
    featured = report["featured_level_index"]
    featured_entry = report["sweep"][featured]
    featured_recovery = report["push_test"]["per_level"][featured]["recovery_times_s"]
    seeds = report["training"]["train_seeds"]
    plans = report["push_test"]["plans"]
    plan_indices = np.arange(len(plans))
    axes[1, 0].plot(
        plan_indices,
        [np.nan if v is None else v for v in baseline_recovery],
        "s",
        color="#64748b",
        label="基线",
    )
    seed_recovery = np.asarray(
        [np.nan if v is None else v for v in featured_recovery], dtype=float
    ).reshape(seeds, len(plans))
    for seed_row in range(seed_recovery.shape[0]):
        axes[1, 0].plot(
            plan_indices,
            seed_recovery[seed_row],
            "o",
            markersize=4,
            alpha=0.7,
            color=colors[(featured + 4) % len(colors)],
            label=f"示教 {level_label(featured_entry)}（种子 {seed_row}）",
        )
    axes[1, 0].set(
        xlabel="推力方案编号",
        ylabel="推力结束后恢复时间（s）",
        title="±200 N 配对推力恢复",
    )
    axes[1, 0].legend(fontsize=7, loc="upper left")

    cases = report["failure_analysis"]["featured_cases"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        if cases:
            case_states = data["case0_states"]
            case = cases[0]
            axes[1, 1].plot(
                np.arange(len(case_states)) * dt,
                np.cos(case_states[:, 1] - ref_theta),
                color="#b91c1c",
            )
            axes[1, 1].axhspan(-1, 0, alpha=0.08, color="orange")
            w_text = f"w={case['w_bc']:g}，" if case.get("w_bc") is not None else ""
            axes[1, 1].set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{w_text}{failure_label(case)}",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "本记录没有示教策略失败回合",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set(title="失败案例：无（全部回合通过验收）")
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
    parser.add_argument("--w-bc", type=float, nargs="+", default=list(W_BC_LEVELS))
    parser.add_argument("--demo-count", type=int, default=DEMO_COUNT)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--minibatch", type=int, default=500)
    parser.add_argument("--train-episode-steps", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--task-envs", type=int, default=4)
    args = parser.parse_args()
    config = PPOConfig(
        updates=args.updates,
        eval_every=min(args.eval_every, args.updates),
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch=args.minibatch,
        train_episode_steps=args.train_episode_steps,
        task_envs=args.task_envs,
    )

    def log(message):
        print(message, file=sys.stderr)  # keep stdout a pure JSON document

    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            config=config,
            train_seeds=args.train_seeds,
            eval_seed_count=args.eval_seeds,
            w_bc_levels=tuple(args.w_bc),
            demo_count=args.demo_count,
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "guard": {
                    part: {
                        key: value
                        for key, value in report["guard"][part].items()
                        if key.startswith("bitwise")
                    }
                    for part in ("pipeline", "training")
                },
                "baseline": report["baseline"]["successes"],
                "dapg": {
                    level_label(entry): entry["stochastic"]["successes"]
                    for entry in report["sweep"]
                },
                "first_success": {
                    level_label(entry): entry["first_success"]["any"] for entry in report["sweep"]
                },
                "arrival": {
                    level_label(entry): entry["arrival"]["episodes_with_arrival"]
                    for entry in report["sweep"]
                },
                "push": {
                    level_label(entry): entry["push"]["successes"] for entry in report["sweep"]
                },
                "bc": {
                    level_label(entry): {
                        "first": float(np.mean([r["bc_loss_first"] for r in entry["training"]])),
                        "last": float(np.mean([r["bc_loss_last"] for r in entry["training"]])),
                    }
                    for entry in report["sweep"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
