"""Lesson 31: PBRS potential-based reward shaping - repair the ladder, not the summit.

Lesson 29's first failure mechanism was "no gradient below": the upright region
was never visited, so the upright term of the reward never had a gradient there
(0/60 evaluation episodes while the zero-shot lesson-7 baseline scores 20/20).
The Ng/Harada/Russell theorem (ICML 1999) says a shaping reward of the exact
form F(s, s') = gamma * Phi(s') - Phi(s) leaves the optimal policy unchanged.
This lesson keeps the lesson-29 pure PPO stack verbatim - no base controller,
no teacher, same environment, same curriculum, same budget and acceptance -
and only changes the reward to

    r_total = r_task + gamma * Phi(s') - Phi(s),
    Phi(s)  = -c_e * |E(s) - E_top|,
    E(s)    = 0.5 * I_eff * omega^2 + m * g * l_eff * (cos(alpha) - 1),

where alpha is the pole angle relative to upright, so E_top = 0 at upright and
the resting down start sits at E = -2*m*g*l. The energy constants are the
lesson-7 controller's own model constants (hinge inertia and m*g*l extracted
from the unchanged XML), making the energy error a "natural" potential: every
step that moves the pole's energy toward the target energy earns a positive
shaping increment - the ladder gets one rung of sugar per step - while the
summit (the optimal policy) is guaranteed unchanged by the theorem. c_e is the
single new hidden hand-knob and is swept over {0.5, 2.0}; c_e = 0 degenerates
to the lesson-29 reward exactly, which is pinned as a bitwise pipeline guard.

This is the "repair the ladder" sister of lesson 30's "cable car": lesson 30
moved the energy job to a hand-built base controller and let PPO learn a
bounded residual; this lesson injects no controller at all and only reshapes
the reward. The comparison is three-way plus guard: the lesson-7 baseline
(zero-shot, re-run in this record), the lesson-29 pure PPO (cited from its
official record), and the PBRS tiers (new).

Honesty rule: if PBRS also fails within the budget, the failure is the formal
result (e.g. the shaped objective can still be hard: Phi counts energy, not
pose - a fast-spinning pole with target energy sits at Phi = 0 too - and the
strict lesson-7 settled tail stays hard). Nothing is smoothed over.

Process metrics recorded beyond success/failure: the fraction of shaping in
the collected return, the Phi profile and per-step shaping along a featured
trajectory (is there really sugar on every rung?), and the first arrival of
the upright capture region (|alpha| <= 0.3 rad, the lesson-7 capture angle) -
the region lesson 29 never reached at all.
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

from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    EVAL_SEEDS,
    GAMMA,
    PUSH_DURATION_S,
    PUSH_FIELDS,
    PUSH_FORCE_N,
    PUSH_START_WINDOW_S,
    SEED_OFFSET_ACT,
    SEED_OFFSET_EVAL,
    SEED_OFFSET_INIT,
    SEED_OFFSET_SHUFFLE,
    STATE_INPUTS,
    TRAIN_SEEDS,
    PPOConfig,
    RewardFunction,
    VecSwingup,
    baseline_evaluations,
    baseline_push_evaluations,
    failure_counts,
    make_push_plans,
    normalize_observation,
    pick_failure_cases,
    policy_array_names,
    push_schedule,
    run_policy_episode,
    select_metrics,
    stack_controls,
    stack_trajectories,
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

EXPERIMENT = "pbrs_swingup_lesson31"
SCHEMA_VERSION = 1

C_E_LEVELS = (0.5, 2.0)  # the single new hand knob, swept at fixed budget
CAPTURE_ANGLE_RAD = 0.3  # lesson-7 capture threshold defines the upright region
GEAR = 100.0  # recovery_metrics reports forces as 100 * normalized command
EVAL_EVERY = 25

# Context numbers imported verbatim from the lesson-29 official record
# (results/ppo_swingup_2026-09-06, docs/33); used only in the three-way table.
LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "median_peak_abs_motor_force_n": 300.0,
    "final_reward_mean_per_seed": [0.4103964962827423, 0.41928791619542005, 0.41858490469934645],
    "first_upright_arrival": "never (docs/33 section 3.4, mechanism 1)",
}


def lesson7_energy_constants(design):
    """The lesson-7 controller's own energy constants on the unchanged model.

    hinge_inertia is I_eff (pole inertia about the hinge) and gravity_energy is
    m * g * l_eff; both are exactly the attributes the lesson-7 energy-shaping
    controller uses, extracted here from the same XML through the same class.
    """
    probe = make_swingup_environment(max_episode_steps=2)
    try:
        controller = HybridSwingupController(probe.unwrapped.model, design)
        return float(controller.hinge_inertia), float(controller.gravity_energy)
    finally:
        probe.close()


def pole_energy(state, reference, hinge_inertia, gravity_energy):
    """Task-frame mechanical energy about the pivot: E_top = 0, resting down = -2*m*g*l.

    This is the lesson-7 energy E7 = 0.5*I*omega^2 + m*g*l*cos(alpha) shifted
    by its target value, so the upright target sits at exactly zero.
    """
    state = np.asarray(state, dtype=float)
    alpha = float(wrap_angle(state[1] - reference[1]))
    return 0.5 * hinge_inertia * float(state[3]) ** 2 + gravity_energy * (np.cos(alpha) - 1.0)


class ShapedReward:
    """r_total = r_task + gamma * Phi(s') - Phi(s), the PBRS form (Ng et al. 1999).

    The shaping is applied on every step including the failing one (a uniform,
    recorded convention). A nonfinite state maps to the zero state before the
    potential is evaluated - the same convention the vec environment uses for
    its terminal observation; physical failures here are finite states.
    With c_e = 0 the total is bitwise the lesson-29 task reward.
    """

    def __init__(self, task_reward, hinge_inertia, gravity_energy, c_e, gamma=GAMMA):
        self.task_reward = task_reward
        self.reference = task_reward.reference
        self.hinge_inertia = float(hinge_inertia)
        self.gravity_energy = float(gravity_energy)
        self.c_e = float(c_e)
        self.gamma = float(gamma)
        if self.c_e < 0.0:
            raise ValueError("c_e must be non-negative")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")

    def energy(self, state):
        return pole_energy(state, self.reference, self.hinge_inertia, self.gravity_energy)

    def potential(self, state):
        """Phi(s) = -c_e * |E(s) - E_top| with E_top = 0 in the task frame."""
        state = np.asarray(state, dtype=float)
        if not np.isfinite(state).all():
            state = np.zeros(4)
        return -self.c_e * abs(self.energy(state))

    def terms(self, pre_state, post_state, action, terminated):
        task = float(self.task_reward(post_state, float(action), terminated))
        shaping = self.gamma * self.potential(post_state) - self.potential(pre_state)
        return {"task": task, "shaping": float(shaping), "total": task + float(shaping)}

    def __call__(self, pre_state, post_state, action, terminated):
        return self.terms(pre_state, post_state, action, terminated)["total"]

    def as_dict(self):
        return {
            "formula": (
                "r_total = r_task + gamma*Phi(s') - Phi(s); "
                "Phi(s) = -c_e*|E(s) - E_top|; "
                "E(s) = 0.5*I_eff*omega^2 + m*g*l_eff*(cos(alpha) - 1)"
            ),
            "c_e": self.c_e,
            "gamma": self.gamma,
            "hinge_inertia_kg_m2": self.hinge_inertia,
            "mgl_eff_j": self.gravity_energy,
            "e_top_j": 0.0,
            "e_rest_down_j": -2.0 * self.gravity_energy,
            "phi_rest_down_j": -self.c_e * 2.0 * self.gravity_energy,
            "failure_step_convention": (
                "shaping is applied on every step including the failing one; "
                "the failing step's task reward is the lesson-29 -failure_penalty"
            ),
            "theorem": (
                "Ng/Harada/Russell ICML 1999: F = gamma*Phi(s') - Phi(s) leaves the "
                "optimal policy of the task reward unchanged (c_e only reshapes learning)"
            ),
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


class ShapedVecSwingup(VecSwingup):
    """Lesson-29 parallel environments with the PBRS shaping added per step.

    step() replicates the lesson-29 loop (identical action->command clip, the
    same task reward on the post-step state, the same curriculum and RNG use);
    the only addition is the shaping term, so with c_e = 0 the rollout stream
    is bitwise the lesson-29 pipeline (pinned by shaping_guard and tests).
    """

    def __init__(self, reward, shaped, **kwargs):
        super().__init__(reward, **kwargs)
        self.shaped = shaped

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n_envs,):
            raise ValueError("one action per environment")
        reference = self.reward.reference
        terminal_obs = np.empty((self.n_envs, STATE_INPUTS))
        live_obs = np.empty((self.n_envs, STATE_INPUTS))
        rewards = np.empty(self.n_envs)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for index, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            pre_state = np.asarray(env.unwrapped._get_obs(), dtype=float)
            command = np.array([np.clip(action, -CONTROL_LIMIT, CONTROL_LIMIT)], np.float32)
            state, _, done, timed_out, _ = env.step(command)
            safe = state if np.isfinite(state).all() else np.zeros(4)
            terminal_obs[index] = normalize_observation(safe, reference)
            rewards[index] = self.shaped(pre_state, state, float(command[0]), done)
            terminated[index], truncated[index] = done, timed_out
            if done or timed_out:
                state = self._start_state(index)
                env.unwrapped.set_state(state[:2], state[2:])
                env.unwrapped.data.qfrc_applied[0] = 0.0
            live_obs[index] = normalize_observation(state, reference)
        return terminal_obs, rewards, terminated, truncated, live_obs


def shaping_guard(reward, shaped_zero, steps=40):
    """c_e = 0 through the shaped pipeline must equal the lesson-29 pipeline bitwise.

    Both environments are constructed with the same base seed (hence the same
    curriculum jitter stream) and stepped with an identical action sequence,
    covering training-style steps and internal episode resets.
    """
    base_seed = 4242
    plain = VecSwingup(reward, n_envs=2, episode_steps=16, base_seed=base_seed, task_envs=1)
    shaped_env = ShapedVecSwingup(
        reward, shaped_zero, n_envs=2, episode_steps=16, base_seed=base_seed, task_envs=1
    )
    try:
        actions = np.random.default_rng([7, 31]).uniform(-3.0, 3.0, (steps, 2))
        obs_plain, obs_shaped = plain.reset(), shaped_env.reset()
        bitwise_obs = bool(np.array_equal(obs_plain, obs_shaped))
        bitwise_states = True
        bitwise_rewards = True
        for chunk in actions:
            out_plain = plain.step(chunk)
            out_shaped = shaped_env.step(chunk)
            bitwise_states &= all(
                np.array_equal(a, b) for a, b in zip(out_plain, out_shaped, strict=True)
            )
            bitwise_rewards &= bool(np.array_equal(out_plain[1], out_shaped[1]))
        return {
            "c_e": 0.0,
            "claim": (
                "with c_e = 0 the shaped pipeline must reproduce the lesson-29 reward "
                "pipeline bitwise (same states, same observations, same rewards) under "
                "an identical action stream"
            ),
            "steps": int(steps),
            "bitwise_identical_rewards": bool(bitwise_rewards),
            "bitwise_identical_states": bool(bitwise_states),
            "bitwise_identical_observations": bool(bitwise_obs),
        }
    finally:
        plain.close()
        shaped_env.close()


def shaped_step_terms(arrays, shaped):
    """Per-step task/shaping rewards recomputed from stored episode arrays.

    Array alignment follows lesson 7: states[k] is the state before action k,
    so step k is judged on (states[k], states[k+1], controls[k]).
    """
    states, controls, end_flags = arrays["states"], arrays["controls"], arrays["end_flags"]
    last = len(controls) - 1
    task, shaping = [], []
    for step in range(len(controls)):
        terminated = bool(end_flags[0]) and step == last
        row = shaped.terms(states[step], states[step + 1], float(controls[step]), terminated)
        task.append(row["task"])
        shaping.append(row["shaping"])
    return {
        "task": np.asarray(task, dtype=float),
        "shaping": np.asarray(shaping, dtype=float),
        "total": np.asarray(task, dtype=float) + np.asarray(shaping, dtype=float),
    }


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


def run_shaped_episode(
    policy, shaped, reference, *, horizon, env_seed, deterministic, rng=None, schedule=None
):
    """One lesson-29 episode from the exact down start (the runner is unchanged)."""
    return run_policy_episode(
        policy,
        shaped.task_reward,
        reference,
        horizon=horizon,
        env_seed=env_seed,
        deterministic=deterministic,
        rng=rng,
        schedule=schedule,
    )


def shaped_episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance reused verbatim (the same recovery_metrics)."""
    view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    return recovery_metrics(view, {"failure_reason": failure_reason}, reference, dt)


def evaluate_shaped_policy(policy, shaped, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start, shaped bookkeeping."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, SEED_OFFSET_EVAL, eval_seed])
        arrays, reason = run_shaped_episode(
            policy,
            shaped,
            reference,
            horizon=EVAL_EPISODE_STEPS,
            env_seed=eval_seed,
            deterministic=False,
            rng=rng,
        )
        metrics = shaped_episode_metrics(arrays, reason, reference, dt)
        terms = shaped_step_terms(arrays, shaped)
        episodes.append(
            {
                "eval_seed": eval_seed,
                **select_metrics(metrics),
                "task_return": float(terms["task"].sum()),
                "shaping_return": float(terms["shaping"].sum()),
                "return": float(terms["total"].sum()),
                "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
                "arrays": arrays,
            }
        )
    return episodes


def deterministic_shaped_episode(policy, shaped, reference, dt):
    """One mean-action episode with the full shaped decomposition archived."""
    arrays, reason = run_shaped_episode(
        policy, shaped, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
    )
    metrics = shaped_episode_metrics(arrays, reason, reference, dt)
    terms = shaped_step_terms(arrays, shaped)
    phi_curve = np.asarray([shaped.potential(state) for state in arrays["states"]], dtype=float)
    record = {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "task_return": float(terms["task"].sum()),
        "shaping_return": float(terms["shaping"].sum()),
        "return": float(terms["total"].sum()),
        "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
        "failure_reason": reason,
    }
    arrays_payload = {
        "states": arrays["states"],
        "controls": arrays["controls"],
        "phi": phi_curve,
        "shaping": terms["shaping"],
        "task": terms["task"],
    }
    return record, arrays_payload


def shaping_summary(episodes):
    """Shaping fraction of the (undiscounted) collected return over episodes.

    The undiscounted sum is a diagnostic only: with gamma < 1 the shaped
    objective the agent optimizes is the discounted one, where the PBRS
    invariance theorem applies exactly.
    """
    task = float(np.mean([episode["task_return"] for episode in episodes]))
    shaping = float(np.mean([episode["shaping_return"] for episode in episodes]))
    total = task + shaping
    return {
        "episodes": len(episodes),
        "mean_task_return": task,
        "mean_shaping_return": shaping,
        "mean_total_return": total,
        "shaping_fraction_of_undiscounted_return": shaping / total if abs(total) > 1e-12 else None,
    }


def arrival_summary(episodes):
    """First-arrival statistics of the upright capture region (the key process metric)."""
    times = [episode["first_arrival_s"] for episode in episodes]
    arrived = [value for value in times if value is not None]
    return {
        "episodes": len(episodes),
        "episodes_with_arrival": len(arrived),
        "arrival_fraction": len(arrived) / len(episodes) if episodes else 0.0,
        "median_first_arrival_s": float(np.median(arrived)) if arrived else None,
        "first_arrival_s_per_episode": times,
    }


def annotate_case_c_e(cases, eval_episodes, push_episodes):
    """Copy the scan level onto each featured case.

    pick_failure_cases drops extra episode fields; the pools are scanned in the
    same fixed order, so the first failing episode of each kind is the case.
    """
    pools = {"eval_failure": eval_episodes, "push_failure": push_episodes}
    for case in cases:
        for episode in pools[case["kind"]]:
            if episode["terminated"] or not episode["recovered"]:
                case["c_e"] = episode["c_e"]
                break
    return cases


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


def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    train_seeds=TRAIN_SEEDS,
    eval_seed_count=EVAL_SEEDS,
    c_e_levels=C_E_LEVELS,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    levels = tuple(float(value) for value in c_e_levels)
    if not levels or any(value < 0.0 for value in levels) or 0.0 in levels:
        raise ValueError("c_e levels must be positive (c_e = 0 is the guard, not a tier)")
    config = config or default_training_config()
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    if abs(design.actuator_gear - GEAR) > 1e-12:
        raise ValueError("actuator gear disagrees with the recovery_metrics convention")
    hinge_inertia, gravity_energy = lesson7_energy_constants(design)
    task_reward = RewardFunction(reference)
    shaped_zero = ShapedReward(task_reward, hinge_inertia, gravity_energy, c_e=0.0, gamma=GAMMA)
    guard = shaping_guard(task_reward, shaped_zero)

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
    reward_curves = {}
    det_store = {}  # (level_index, seed_index) -> shaped decomposition payload
    eval_stores = {index: [] for index in range(len(levels))}
    push_stores = {index: [] for index in range(len(levels))}
    all_eval_episodes, all_push_episodes = [], []

    for level_index, c_e in enumerate(levels):
        shaped = ShapedReward(task_reward, hinge_inertia, gravity_energy, c_e=c_e, gamma=GAMMA)
        per_seed_records, det_records = [], []
        for seed_index in range(train_seeds):
            vec_env = ShapedVecSwingup(
                task_reward,
                shaped,
                n_envs=config.n_envs,
                episode_steps=config.train_episode_steps,
                base_seed=10_000 + seed * 1000 + level_index * 100 + seed_index,
                task_envs=config.task_envs,
            )

            def eval_hook(policy, _env_steps, _shaped=shaped, _reference=reference, _dt=dt):
                record, _payload = deterministic_shaped_episode(policy, _shaped, _reference, _dt)
                return {
                    "success": bool(record["recovered"] and not record["terminated"]),
                    "settled_at_s": record["settled_at_s"],
                    "return": record["return"],
                    "first_arrival_s": record["first_arrival_s"],
                }

            result = train_ppo(
                vec_env,
                config=config,
                init_seed=[seed, SEED_OFFSET_INIT + level_index, seed_index],
                act_seed=[seed, SEED_OFFSET_ACT + level_index, seed_index],
                shuffle_seed=[seed, SEED_OFFSET_SHUFFLE + level_index, seed_index],
                eval_hook=eval_hook,
                log=log,
            )
            vec_env.close()
            policy = result["policy"]
            policy_payloads[(level_index, seed_index)] = policy.arrays()
            reward_curves[(level_index, seed_index)] = result["reward_curve"]

            eval_episodes = evaluate_shaped_policy(
                policy, shaped, reference, dt, master_seed=seed, count=eval_seed_count
            )
            det_record, det_payload = deterministic_shaped_episode(policy, shaped, reference, dt)
            det_store[(level_index, seed_index)] = det_payload
            push_episodes = []
            for plan in plans:
                arrays, reason = run_shaped_episode(
                    policy,
                    shaped,
                    reference,
                    horizon=EVAL_EPISODE_STEPS,
                    env_seed=plan["index"],
                    deterministic=True,
                    schedule=push_schedule(plan, dt, EVAL_EPISODE_STEPS),
                )
                metrics = shaped_episode_metrics(arrays, reason, reference, dt)
                terms = shaped_step_terms(arrays, shaped)
                push_episodes.append(
                    {
                        "plan_index": plan["index"],
                        "force_n": plan["force_n"],
                        "start_s": plan["start_s"],
                        **select_metrics(metrics, PUSH_FIELDS),
                        "return": float(terms["total"].sum()),
                        "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
                        "arrays": arrays,
                    }
                )

            all_eval_episodes.extend({**episode, "c_e": c_e} for episode in eval_episodes)
            all_push_episodes.extend({**episode, "c_e": c_e} for episode in push_episodes)
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
                    "task_returns": [float(episode["task_return"]) for episode in eval_episodes],
                    "shaping_returns": [
                        float(episode["shaping_return"]) for episode in eval_episodes
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
            arrival_steps = next(
                (
                    int(step)
                    for step, point in zip(
                        result["eval_steps"], result["eval_records"], strict=True
                    )
                    if point["first_arrival_s"] is not None
                ),
                None,
            )
            record = {
                "seed_index": seed_index,
                "env_steps": result["env_steps"],
                "wall_time_s": result["wall_time_s"],
                "final_reward_mean": float(np.mean(result["reward_curve"][final_window])),
                "final_log_std": float(policy.log_std[0]),
                "first_successful_eval_steps": next(
                    (
                        int(step)
                        for step, point in zip(
                            result["eval_steps"], result["eval_records"], strict=True
                        )
                        if point["success"]
                    ),
                    None,
                ),
                "first_arrival_eval_steps": arrival_steps,
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
                    f"c_e {c_e:g} seed {seed_index}: steps {record['env_steps']}, "
                    f"wall {record['wall_time_s']:.1f}s, "
                    f"stoch {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                    f"upright arrival {arrived}/{len(eval_episodes)}, "
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
                "c_e": c_e,
                "training": per_seed_records,
                "deterministic": det_records,
                "stochastic": stochastic_summary,
                "push": push_summary,
                "shaping": shaping_summary(level_eval_episodes),
                "arrival": arrival_summary(level_eval_episodes),
            }
        )

    failure_cases = annotate_case_c_e(
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
        hinge_inertia=hinge_inertia,
        gravity_energy=gravity_energy,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        levels=levels,
        plans=plans,
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
        reward_curves=reward_curves,
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
    save_ladder(output / "ladder_analysis.png", report, output)
    save_three_way(output / "three_way.png", report, output)
    return report


def build_report(
    *,
    seed,
    config,
    design,
    hinge_inertia,
    gravity_energy,
    train_seeds,
    eval_seed_count,
    levels,
    plans,
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
    ]
    for entry in entries:
        three_way.append(
            {
                "label": f"PBRS（cE={entry['c_e']:g}）",
                "episodes": entry["stochastic"]["episodes"],
                "successes": entry["stochastic"]["successes"],
                "median_settled_at_s": entry["stochastic"]["median_settled_at_s"],
                "median_peak_abs_motor_force_n": entry["stochastic"][
                    "median_peak_abs_motor_force_n"
                ],
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
                "repair the ladder (PBRS reward shaping): the lesson-29 pure PPO stack is "
                "reused verbatim - no base controller, no teacher - only the reward gains "
                "the potential-based term; sister of lesson 30's cable car (residual RL)"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "potential": ShapedReward(
                RewardFunction(reference), hinge_inertia, gravity_energy, c_e=1.0, gamma=GAMMA
            ).as_dict()
            | {"c_e_levels": list(levels)},
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
                    "never arrived in any evaluation episode (docs/33 mechanism 1)"
                ),
            },
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lesson 29",
            },
            "seed_streams": {
                "network_init": "default_rng([master, 7000 + level, train_seed]); value net [.., 1]",
                "action_sampling": "default_rng([master, 5000 + level, train_seed])",
                "minibatch_order": "default_rng([master, 9000 + level, train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*1000 "
                    "+ level*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lessons 29/30)",
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
        "guard": guard,
        "sweep": entries,
        "featured_level_index": featured,
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; PBRS policies never saw pushes during training; "
                    "the same plans run from the same exact down start; mean-action episodes"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": (
                    "the same plans are applied to the baseline and to every c_e level and "
                    "training seed"
                ),
            },
            "plans": plans,
            "baseline": baseline_push_summary,
            "per_level": [
                {
                    "c_e": entry["c_e"],
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
        "hypothesis": {
            "claim": (
                "PBRS leaves the optimal policy unchanged (theorem) but says nothing about "
                "learnability within a fixed budget: the primary question is whether the "
                "shaped objective gets a pure-PPO agent from the exact down start to the "
                "strict lesson-7 acceptance at all, and whether the upright region is "
                "visited at all (first arrival) - a null result is recorded as the formal "
                "conclusion"
            ),
            "per_level": [
                {
                    "c_e": entry["c_e"],
                    "successes": entry["stochastic"]["successes"],
                    "episodes": entry["stochastic"]["episodes"],
                    "episodes_with_arrival": entry["arrival"]["episodes_with_arrival"],
                    "first_arrival_eval_steps_per_seed": [
                        record["first_arrival_eval_steps"] for record in entry["training"]
                    ],
                    "first_successful_eval_steps_per_seed": [
                        record["first_successful_eval_steps"] for record in entry["training"]
                    ],
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
            "curves_note": "reward_curve_<level>_<seed> lives in trajectories.npz",
        },
        "limitations": [
            (
                "c_e in {0.5, 2.0} is a hand-picked two-point grid; the single new hidden "
                "hand-knob of this lesson (the potential scale) was not tuned further."
            ),
            (
                "Phi counts energy, not pose: any state with the target energy sits at "
                "Phi = 0, including a fast-spinning pole far from upright; the ladder tops "
                "out at the right energy, and the last mile (capture + settle) stays with "
                "the task reward alone."
            ),
            (
                "With gamma < 1 the undiscounted shaping return of a long low-potential "
                "episode is positive; the agent optimizes the discounted objective, where "
                "the policy-invariance theorem applies exactly, but the reported "
                "undiscounted shaping fraction is a diagnostic, not the training signal."
            ),
            (
                "The theorem guarantees the optimal policy is unchanged; it guarantees "
                "nothing about sample efficiency within this fixed budget - a null result "
                "here does not refute PBRS in general."
            ),
            (
                "The lesson-29 reference numbers in three_way_comparison are imported from "
                "that record (60 reward-only PPO episodes), not re-run here; c_e = 0 is "
                "covered by the bitwise pipeline guard instead of a full re-run."
            ),
            (
                "One task, one nominal MuJoCo model, no noise/delay/mass error; success "
                "requires the strict lesson-7 settled tail (0.02 m, 0.01 rad, 0.02 m/s, "
                "0.02 rad/s held >= 2 s)."
            ),
        ],
    }
    return report


def build_archive(
    *,
    reward_curves,
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
        "baseline_states": np.asarray(baseline_states, dtype=np.float64),
        "baseline_controls": np.asarray(baseline_controls, dtype=np.float64),
        "baseline_push_states": stack_trajectories(baseline_push_states, horizon)[0],
        "baseline_push_lengths": stack_trajectories(baseline_push_states, horizon)[1],
        "baseline_push_controls": stack_controls(baseline_push_controls, horizon),
    }
    for (level_index, seed_index), curve in reward_curves.items():
        archive[f"reward_curve_{level_index}_{seed_index}"] = curve
    for (level_index, seed_index), payload in policy_payloads.items():
        for name, array in payload.items():
            archive[f"policy_{level_index}_{seed_index}_{name}"] = array
    for (level_index, seed_index), payload in det_store.items():
        for suffix in ("states", "controls", "phi", "shaping", "task"):
            archive[f"det_{suffix}_{level_index}_{seed_index}"] = payload[suffix]
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
            ("task_returns", "task_returns"),
            ("shaping_returns", "shaping_returns"),
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
    }
    for level in range(levels):
        keys.update(f"reward_curve_{level}_{seed}" for seed in range(seeds))
        keys.update(
            f"policy_{level}_{seed}_{name}"
            for seed in range(seeds)
            for name in policy_array_names(hidden)
        )
        keys.update(
            f"det_{suffix}_{level}_{seed}"
            for seed in range(seeds)
            for suffix in ("states", "controls", "phi", "shaping", "task")
        )
        keys.update(
            f"eval_{suffix}_{level}"
            for suffix in (
                "states",
                "lengths",
                "controls",
                "terminated",
                "settled_s",
                "task_returns",
                "shaping_returns",
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


def save_training_curves(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        curves = [
            np.stack([data[f"reward_curve_{index}_{seed}"] for seed in range(seeds)], axis=0)
            for index in range(len(report["sweep"]))
        ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
    colors = ("#2563eb", "#0f766e", "#b45309")
    lesson29_final = float(np.mean(LESSON29_PPO_REFERENCE["final_reward_mean_per_seed"]))
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        for seed_row in range(seeds):
            axes[0].plot(updates, curves[index][seed_row], alpha=0.25, linewidth=0.8, color=color)
        axes[0].plot(
            updates,
            curves[index].mean(axis=0),
            color=color,
            linewidth=1.8,
            label=f"cE={entry['c_e']:g}（{seeds} 种子均值）",
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
        ylabel="批内平均奖励（含塑形，每步）",
        title="PBRS 训练奖励：细线 = 单个种子",
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
        axes[1].plot(steps, success.mean(axis=0), "o-", color=color, label=f"cE={entry['c_e']:g}")
    axes[1].set(
        xlabel="环境步数（×1000）",
        ylabel="下方初态验收通过（1=成功）",
        ylim=(-0.08, 1.08),
        yticks=[0, 0.5, 1],
        title="训练中周期评估：均值动作、下方初态",
    )
    axes[1].legend(fontsize=7, loc="upper left")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_ladder(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    potential = report["protocol"]["potential"]
    mgl = float(potential["mgl_eff_j"])
    c_e = report["sweep"][report["featured_level_index"]]["c_e"]
    seed_index = 0
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        det_states = data[f"det_states_{report['featured_level_index']}_{seed_index}"]
        det_phi = data[f"det_phi_{report['featured_level_index']}_{seed_index}"]
        det_shaping = data[f"det_shaping_{report['featured_level_index']}_{seed_index}"]
        det_task = data[f"det_task_{report['featured_level_index']}_{seed_index}"]
    entry = report["sweep"][report["featured_level_index"]]
    arrival = entry["deterministic"][seed_index]["first_arrival_s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    alphas = np.linspace(-np.pi, np.pi, 721)
    phi_static = -c_e * np.abs(mgl * (np.cos(alphas) - 1.0))
    axes[0, 0].plot(alphas, phi_static, color="#0f766e")
    axes[0, 0].axvline(0.0, color="gray", linestyle=":", linewidth=0.9)
    axes[0, 0].annotate("直立 Φ=0", (0.0, 0.0), xytext=(0.35, -0.72 * phi_static.min()), fontsize=8)
    axes[0, 0].annotate(
        f"正下方 Φ={phi_static.min():.1f} J",
        (-np.pi + 0.1, phi_static.min() * 1.02),
        fontsize=8,
    )
    axes[0, 0].set(
        xlabel="杆相对直立的角度 α（rad，ω=0）",
        ylabel="Φ（J）",
        title=f"静态梯子：Φ(α) = −cE·|E−E_top|，cE={c_e:g}",
    )
    axes[0, 1].plot(
        np.arange(len(det_states)) * dt,
        np.cos(det_states[:, 1] - ref_theta),
        color="#2563eb",
    )
    axes[0, 1].axhspan(-1, 0, alpha=0.08, color="orange")
    if arrival is not None:
        axes[0, 1].axvline(arrival, color="#b45309", linewidth=1.2)
        axes[0, 1].annotate(
            f"直立首达 {arrival:.2f} s",
            (arrival, 0.0),
            xytext=(6, -30),
            textcoords="offset points",
            fontsize=8,
        )
    else:
        axes[0, 1].annotate(
            "未进入直立区（|α|≤0.3 rad）", (0.02, 0.92), xycoords="axes fraction", fontsize=8
        )
    axes[0, 1].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"典型轨迹（cE={c_e:g} 种子 {seed_index}，均值动作）",
    )
    axes[1, 0].plot(np.arange(len(det_phi)) * dt, det_phi, color="#7c3aed")
    axes[1, 0].set(
        ylabel="Φ（J）",
        xlabel="仿真时间（s）",
        title="势函数沿轨迹（爬梯 = Φ 升向 0）",
    )
    edges = np.arange(len(det_shaping) + 1) * dt
    axes[1, 1].stairs(det_shaping, edges, color="#b45309", label="塑形 γΦ(s′)−Φ(s)")
    axes[1, 1].stairs(det_task, edges, color="#64748b", alpha=0.85, label="任务奖励")
    axes[1, 1].set(
        ylabel="每步奖励",
        xlabel="仿真时间（s）",
        title="每格有糖：塑形项 vs 任务项（失败步任务项 = −10）",
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
    colors = ("#64748b", "#b91c1c", "#2563eb", "#0f766e")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    rows = report["three_way_comparison"]
    labels = ["基线", "纯PPO\n(第29课)"] + [
        f"PBRS\ncE={entry['c_e']:g}" for entry in report["sweep"]
    ]
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    bars = axes[0, 0].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=colors[: len(labels)],
        width=0.6,
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
        title="三方成功率（第 7 课口径）",
    )
    axes[0, 0].tick_params(axis="x", labelsize=7)
    any_arrival = False
    for index, entry in enumerate(report["sweep"]):
        arrival = np.asarray(entry["arrival"]["first_arrival_s_per_episode"], dtype=float)
        arrived = np.sort(arrival[~np.isnan(arrival)])
        any_arrival |= len(arrived) > 0
        axes[0, 1].plot(
            np.arange(1, len(arrived) + 1),
            arrived,
            "o",
            markersize=4,
            color=colors[(index + 2) % len(colors)],
            label=f"cE={entry['c_e']:g}（{len(arrived)}/{entry['arrival']['episodes']}）",
        )
    if any_arrival:
        axes[0, 1].set(
            xlabel="到达回合序号（按首达时间排序）",
            ylabel="直立区首次到达时刻（s）",
            title=f"直立首达（|α|≤{CAPTURE_ANGLE_RAD:g} rad）",
        )
        axes[0, 1].legend(fontsize=7)
    else:
        axes[0, 1].set_xlim(-0.5, 0.5)
        axes[0, 1].set_ylim(-0.5, 0.5)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        checkpoints = [
            (entry["c_e"], record["seed_index"], record["first_arrival_eval_steps"])
            for entry in report["sweep"]
            for record in entry["training"]
            if record["first_arrival_eval_steps"] is not None
        ]
        if checkpoints:
            detail = "；".join(
                f"cE={c_e:g} 种子 {seed} @ {steps / 1000:.0f}k 步"
                for c_e, seed, steps in checkpoints
            )
            note = f"评估回合从未进入直立区\n训练中检查点短暂触达：{detail}"
        else:
            note = "评估回合与训练检查点均从未进入直立区"
        axes[0, 1].text(
            0.5,
            0.55,
            note,
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
            fontsize=8,
        )
        axes[0, 1].set(title=f"直立首达 0/60（第 29 课亦 0/60，|α|≤{CAPTURE_ANGLE_RAD:g} rad）")
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
            color=colors[(featured + 2) % len(colors)],
            label=f"PBRS cE={featured_entry['c_e']:g}（种子 {seed_row}）",
        )
    axes[1, 0].set(
        xlabel="推力方案编号",
        ylabel="推力结束后恢复时间（s）",
        title="±200 N 配对推力恢复",
    )
    axes[1, 0].legend(fontsize=7)
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
            c_e_text = f"cE={case['c_e']:g}，" if case.get("c_e") is not None else ""
            axes[1, 1].set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{c_e_text}{failure_label(case)}",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "本记录没有 PBRS 失败回合",
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
    parser.add_argument("--c-e", type=float, nargs="+", default=list(C_E_LEVELS))
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
            c_e_levels=tuple(args.c_e),
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "guard": {
                    key: report["guard"][key]
                    for key in (
                        "bitwise_identical_rewards",
                        "bitwise_identical_states",
                        "bitwise_identical_observations",
                    )
                },
                "baseline": report["baseline"]["successes"],
                "pbrs": {
                    f"cE={entry['c_e']:g}": entry["stochastic"]["successes"]
                    for entry in report["sweep"]
                },
                "arrival": {
                    f"cE={entry['c_e']:g}": entry["arrival"]["episodes_with_arrival"]
                    for entry in report["sweep"]
                },
                "push": {
                    f"cE={entry['c_e']:g}": entry["push"]["successes"] for entry in report["sweep"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
