"""Lesson 30: residual RL swing-up - the lesson-7 controller keeps the energy job.

Lesson 29 showed that reward-only PPO cannot cross the exploration cliff in
budget (0/60 from the exact down start while the zero-shot lesson-7 baseline
scores 20/20), and its nearest miss failed exactly at capture-then-recenter
coordination: the pole reached 0.990 relative height while the cart hit the
2.4 m boundary. This lesson keeps the hand-designed controller responsible for
energy injection - the lesson-7 energy-shaping + LQR hysteresis switch runs
unchanged inside the loop as the base - and trains the lesson-29 numpy PPO
stack to output only a bounded residual:

    u = clip(u_energy + clip(u_RL, +/-a), +/-300 N)

with the residual budget a in {25, 50, 100} N (a = 0 degenerates to the pure
baseline and is run as a guard: through the residual pipeline it must reproduce
the lesson-7 trajectory bitwise). The learner's capacity is thus spent on the
new task - improve the capture/handoff segment without breaking the base -
instead of re-discovering energy injection (failure mechanisms 1 and 3 of
lesson 29 are removed by the base; mechanisms 2 and 4 are caught by it).

Reused verbatim from lesson 29: the 5->64->64 Gaussian policy + value tower,
hand GAE(lambda), clipped surrogate, hand Adam, the two-bank start curriculum,
the reward family, the cos/sin observation, the seed-stream bookkeeping, and
the lesson-7 acceptance (recovery_metrics). Two recorded hand choices are new:
the reward's control cost acts on the residual the learner chooses (it cannot
influence the base command), and PPO updates on the pre-clip Gaussian sample
while both training and execution apply clip(u_RL, +/-a), so |u_RL| <= a holds
by construction everywhere.

Checks (same conditions throughout, single variable = the base/residual split):
(1) baseline: the lesson-7 controller, zero-shot, 20 down-start repeats with
    the lesson-7 acceptance (deterministic, so repeats coincide - recorded);
(2) guard: a = 0 through the residual pipeline, bitwise-identical states and
    commands against the baseline run;
(3) residual PPO: a in {25, 50, 100} N x 3 training seeds at half the
    lesson-29 budget (250k steps per seed), training curves in the lesson-29
    caliber, then 20 stochastic + 1 mean-action episode per seed on the same
    exact down start with the same acceptance;
(4) main hypothesis: residual success >= baseline AND a recomputable
    improvement in a handoff metric (swing-up time or input peak); a null or
    negative result is recorded as the formal conclusion, not smoothed over;
(5) disturbance: the same 20 paired +/-200 N plans for baseline and residual;
(6) residual usage: the |u_RL| distribution against the budget - all-zero
    means the residual degenerated, at-limit means the base was not enough;
    both are reported as measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    down_start_state,
    failure_counts,
    make_push_plans,
    normalize_observation,
    pick_failure_cases,
    policy_array_names,
    push_schedule,
    select_metrics,
    stack_controls,
    stack_trajectories,
    summarize_episodes,
    train_ppo,
)
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
)

EXPERIMENT = "residual_swingup_lesson30"
SCHEMA_VERSION = 1

RESIDUAL_LIMITS_N = (25.0, 50.0, 100.0)
UPDATES = 125  # half of lesson-29's 250 updates: 250k steps per seed per amplitude
EVAL_EVERY = 25
LIMIT_SATURATION = 0.95  # |u_RL| >= 95% of the budget counts as "at the limit"

# Context numbers imported verbatim from the lesson-29 official record
# (results/ppo_swingup_2026-09-06, docs/33); used only in the three-way table.
LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "median_peak_abs_motor_force_n": 300.0,
}

GEAR = 100.0  # recovery_metrics reports forces as 100 * normalized command


def default_training_config(updates=UPDATES):
    """Lesson-29 hyperparameters at the halved update budget."""
    updates = int(updates)
    if not 1 <= updates <= 1000:
        raise ValueError("updates must be in [1, 1000]")
    return PPOConfig(updates=updates, eval_every=min(EVAL_EVERY, updates))


def residual_command(base, z, limit_norm):
    """u = clip(u_energy + clip(u_RL, +/-a), +/-3) in normalized units.

    Returns the float32 command handed to the actuator and the clipped residual
    (the action the reward judges). With a = 0 the command is the base output
    unchanged, which is what the guard check pins bitwise.
    """
    limit_norm = float(limit_norm)
    if not 0.0 <= limit_norm <= CONTROL_LIMIT:
        raise ValueError("residual limit must be within [0, control_limit]")
    residual = float(np.clip(z, -limit_norm, limit_norm))
    total = np.asarray(base, dtype=np.float32) + np.float32(residual)
    command = np.clip(total, -CONTROL_LIMIT, CONTROL_LIMIT)
    return command.astype(np.float32, copy=False), residual


class ResidualVecSwingup(VecSwingup):
    """Lesson-29 parallel environments with the lesson-7 base in the loop.

    step() receives the raw Gaussian samples (PPO keeps updating on the
    pre-clip sample); the environment only ever sees the clipped residual added
    to the base command. Each environment owns one HybridSwingupController and
    a fresh one is created at every episode start, matching the lesson-7
    run_scenario bookkeeping.
    """

    def __init__(self, reward, *, design, residual_limit_norm, **kwargs):
        super().__init__(reward, **kwargs)
        self.design = design
        self.residual_limit_norm = float(residual_limit_norm)
        if not 0.0 <= self.residual_limit_norm <= design.controller.control_limit:
            raise ValueError("residual limit must be within [0, control_limit]")
        self.bases = [HybridSwingupController(env.unwrapped.model, design) for env in self.envs]

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
            state = np.asarray(env.unwrapped._get_obs(), dtype=float)  # pre-step raw state
            base = self.bases[index].action(state)
            command, residual = residual_command(base, float(action), self.residual_limit_norm)
            env.unwrapped.data.qfrc_applied[0] = 0.0
            state_after, _, done, timed_out, _ = env.step(command)
            safe = state_after if np.isfinite(state_after).all() else np.zeros(4)
            terminal_obs[index] = normalize_observation(safe, reference)
            rewards[index] = self.reward(state_after, residual, done)
            terminated[index], truncated[index] = done, timed_out
            if done or timed_out:
                state = self._start_state(index)
                env.unwrapped.set_state(state[:2], state[2:])
                env.unwrapped.data.qfrc_applied[0] = 0.0
                self.bases[index] = HybridSwingupController(env.unwrapped.model, self.design)
            live_obs[index] = normalize_observation(state, reference)
        return terminal_obs, rewards, terminated, truncated, live_obs


def run_residual_episode(
    policy,
    reward,
    design,
    *,
    horizon,
    residual_limit_norm,
    env_seed,
    deterministic,
    rng=None,
    schedule=None,
):
    """One episode from the exact down start with the base controller active.

    Array alignment follows lesson 7: states[k] is the state before action k,
    controls/residuals/base_controls[k] act on [k*dt, (k+1)*dt). With a = 0 the
    policy is never queried and the loop reduces to the lesson-7 run_scenario.
    """
    if schedule is None:
        schedule = np.zeros(horizon)
    schedule = np.asarray(schedule, dtype=float)
    if schedule.shape != (horizon,):
        raise ValueError("push schedule must cover the horizon")
    limit_norm = float(residual_limit_norm)
    if not 0.0 <= limit_norm <= CONTROL_LIMIT:
        raise ValueError("residual limit must be within [0, control_limit]")
    if limit_norm > 0.0 and policy is None:
        raise ValueError("a policy is required for a nonzero residual budget")
    env = make_swingup_environment(max_episode_steps=horizon)
    try:
        env.reset(seed=int(env_seed))
        env.unwrapped.data.qfrc_applied[0] = 0.0
        state = down_start_state(reward.reference)
        env.unwrapped.set_state(state[:2], state[2:])
        controller = HybridSwingupController(env.unwrapped.model, design)
        states, controls, residuals, base_controls, forces, modes = (
            [state.copy()],
            [],
            [],
            [],
            [],
            [],
        )
        terminated = truncated = False
        failure_reason = ""
        for force in schedule:
            if limit_norm > 0.0:
                obs = normalize_observation(state, reward.reference)[None, :]
                if deterministic:
                    z = float(policy.mean(obs)[0])
                else:
                    z = float(policy.sample(obs, rng)[0][0])
            else:
                z = 0.0
            base = controller.action(state)
            command, residual = residual_command(base, z, limit_norm)
            env.unwrapped.data.qfrc_applied[0] = float(force)
            state, _, terminated, truncated, info = env.step(command)
            states.append(state.copy())
            controls.append(float(command[0]))
            residuals.append(residual)
            base_controls.append(float(base[0]))
            forces.append(float(force))
            modes.append(controller.mode)
            failure_reason = info["failure_reason"]
            if terminated or truncated:
                break
        arrays = {
            # float64 on purpose: the guard compares these bitwise against the
            # lesson-7 run_scenario arrays, which also store raw MuJoCo states
            "states": np.asarray(states, dtype=np.float64),
            "controls": np.asarray(controls, dtype=np.float32),
            "residuals": np.asarray(residuals, dtype=np.float32),
            "base_controls": np.asarray(base_controls, dtype=np.float32),
            "applied_force_n": np.asarray(forces, dtype=np.float32),
            "scheduled_force_n": schedule,
            "modes": np.asarray(modes),
            "end_flags": np.array([terminated, truncated]),
        }
        return arrays, failure_reason
    finally:
        env.close()


def residual_episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance applied to a residual episode (the same function)."""
    metadata = {"failure_reason": failure_reason}
    return recovery_metrics(arrays, metadata, reference, dt)


def residual_episode_rewards(arrays, reward):
    """Recompute per-step rewards with the residual as the judged action."""
    residuals, end_flags, states = arrays["residuals"], arrays["end_flags"], arrays["states"]
    last = len(residuals) - 1
    return np.asarray(
        [
            reward(states[step + 1], float(residuals[step]), bool(end_flags[0]) and step == last)
            for step in range(len(residuals))
        ],
        dtype=float,
    )


def capture_time_s(modes, dt):
    """First time the hybrid base hands control to its LQR balance mode."""
    for index, mode in enumerate(np.asarray(modes)):
        if str(mode) == "balance":
            return float(index * dt)
    return None


def residual_stats(residual_norm, limit_norm, gear=GEAR):
    """|u_RL| usage against the budget (recorded for every amplitude)."""
    residual_norm = np.asarray(residual_norm, dtype=float).ravel()
    if residual_norm.size == 0:
        raise ValueError("residual sample is empty")
    if not np.isfinite(residual_norm).all():
        raise ValueError("residual sample must be finite")
    force = np.abs(residual_norm) * gear
    budget_n = float(limit_norm) * gear
    if budget_n <= 0.0:
        return {
            "budget_n": 0.0,
            "steps": int(residual_norm.size),
            "mean_abs_n": 0.0,
            "median_abs_n": 0.0,
            "p95_abs_n": 0.0,
            "max_abs_n": 0.0,
            "fraction_at_limit": 0.0,
            "fraction_below_half": 0.0,
        }
    abs_norm = np.abs(residual_norm)
    return {
        "budget_n": budget_n,
        "steps": int(residual_norm.size),
        "mean_abs_n": float(force.mean()),
        "median_abs_n": float(np.median(force)),
        "p95_abs_n": float(np.percentile(force, 95)),
        "max_abs_n": float(force.max()),
        "fraction_at_limit": float(np.mean(abs_norm >= LIMIT_SATURATION * limit_norm)),
        "fraction_below_half": float(np.mean(abs_norm < 0.5 * limit_norm)),
    }


def evaluate_residual_policy(policy, reward, design, *, master_seed, limit_norm, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start (sampling noise only)."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, SEED_OFFSET_EVAL, eval_seed])
        arrays, reason = run_residual_episode(
            policy,
            reward,
            design,
            horizon=EVAL_EPISODE_STEPS,
            residual_limit_norm=limit_norm,
            env_seed=eval_seed,
            deterministic=False,
            rng=rng,
        )
        metrics = residual_episode_metrics(arrays, reason, reward.reference, design.dt)
        episodes.append(
            {
                "eval_seed": eval_seed,
                **select_metrics(metrics),
                "return": float(residual_episode_rewards(arrays, reward).sum()),
                "arrays": arrays,
            }
        )
    return episodes


def deterministic_residual(policy, reward, design, *, limit_norm):
    """One mean-action episode (the executed-policy view) with capture timing."""
    arrays, reason = run_residual_episode(
        policy,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=limit_norm,
        env_seed=0,
        deterministic=True,
    )
    metrics = residual_episode_metrics(arrays, reason, reward.reference, design.dt)
    return arrays, reason, metrics


def baseline_guard(design, reward, reference, dt):
    """a = 0 through the residual pipeline must equal the lesson-7 run bitwise."""
    arrays, reason = run_residual_episode(
        None,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=0.0,
        env_seed=0,
        deterministic=True,
    )
    reference_arrays, _metadata = run_scenario(Scenario("down", "down"), design, EVAL_EPISODE_STEPS)
    metrics = residual_episode_metrics(arrays, reason, reference, dt)
    return {
        "limit_n": 0.0,
        "bitwise_identical_states": bool(
            np.array_equal(arrays["states"], reference_arrays["states"])
        ),
        "bitwise_identical_controls": bool(
            np.array_equal(arrays["controls"], reference_arrays["controls"])
        ),
        "settled_at_s": metrics["settled_at_s"],
        "peak_abs_motor_force_n": metrics["peak_abs_motor_force_n"],
        "max_abs_cart_position_m": metrics["max_abs_cart_position_m"],
        "capture_time_s": capture_time_s(arrays["modes"], dt),
        "arrays": arrays,
    }


def annotate_case_limits(cases, eval_episodes, push_episodes):
    """Copy the scan amplitude onto each featured case.

    pick_failure_cases drops extra episode fields; the pools are scanned in the
    same fixed order, so the first failing episode of each kind is the case.
    """
    pools = {"eval_failure": eval_episodes, "push_failure": push_episodes}
    for case in cases:
        for episode in pools[case["kind"]]:
            if episode["terminated"] or not episode["recovered"]:
                case["limit_n"] = episode["limit_n"]
                break
    return cases


def choose_featured_amplitude(sweep_entries):
    """Deterministic figure pick: most successes, then smallest swing-up time."""
    best, best_key = 0, None
    for index, entry in enumerate(sweep_entries):
        aggregate = entry["stochastic"]
        settled = aggregate["median_settled_at_s"]
        settled_key = -settled if settled is not None else float("-inf")
        key = (aggregate["successes"], settled_key)
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
    limits_n=RESIDUAL_LIMITS_N,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    limits = tuple(float(value) for value in limits_n)
    if not limits or any(not 0.0 < value <= GEAR * CONTROL_LIMIT for value in limits):
        raise ValueError("residual limits must be in (0, 300] N")
    config = config or default_training_config()
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    if abs(design.actuator_gear - GEAR) > 1e-12:
        raise ValueError("actuator gear disagrees with the recovery_metrics convention")
    reward = RewardFunction(reference)

    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    guard = baseline_guard(design, reward, reference, dt)
    plans = make_push_plans(dt, eval_seed_count, seed)
    baseline_push_records, baseline_push_states, baseline_push_controls = baseline_push_evaluations(
        design, plans, EVAL_EPISODE_STEPS
    )

    output.mkdir(parents=True, exist_ok=False)
    sweep_entries = []
    policy_payloads = {}  # (amp_index, seed_index) -> policy arrays
    reward_curves = {}  # (amp_index, seed_index) -> per-update reward
    det_arrays_store = {}  # (amp_index, seed_index) -> mean-action episode arrays
    eval_stores = {index: [] for index in range(len(limits))}
    push_stores = {index: [] for index in range(len(limits))}
    all_eval_episodes, all_push_episodes = [], []

    for amp_index, limit_n in enumerate(limits):
        limit_norm = limit_n / GEAR
        per_seed_records, det_records = [], []
        det_residual_pool, stoch_residual_pool = [], []
        for seed_index in range(train_seeds):
            vec_env = ResidualVecSwingup(
                reward,
                design=design,
                residual_limit_norm=limit_norm,
                n_envs=config.n_envs,
                episode_steps=config.train_episode_steps,
                base_seed=10_000 + seed * 1000 + amp_index * 100 + seed_index,
                task_envs=config.task_envs,
            )

            def eval_hook(policy, _env_steps, _limit=limit_norm):
                arrays, _reason, metrics = deterministic_residual(
                    policy, reward, design, limit_norm=_limit
                )
                return {
                    "success": bool(metrics["recovered"] and not metrics["terminated"]),
                    "settled_at_s": metrics["settled_at_s"],
                    "return": float(residual_episode_rewards(arrays, reward).sum()),
                }

            result = train_ppo(
                vec_env,
                config=config,
                init_seed=[seed, SEED_OFFSET_INIT + amp_index, seed_index],
                act_seed=[seed, SEED_OFFSET_ACT + amp_index, seed_index],
                shuffle_seed=[seed, SEED_OFFSET_SHUFFLE + amp_index, seed_index],
                eval_hook=eval_hook,
                log=log,
            )
            vec_env.close()
            policy = result["policy"]
            policy_payloads[(amp_index, seed_index)] = policy.arrays()
            reward_curves[(amp_index, seed_index)] = result["reward_curve"]

            eval_episodes = evaluate_residual_policy(
                policy,
                reward,
                design,
                master_seed=seed,
                limit_norm=limit_norm,
                count=eval_seed_count,
            )
            det_arrays, _det_reason, det_metrics = deterministic_residual(
                policy, reward, design, limit_norm=limit_norm
            )
            push_episodes = []
            for plan in plans:
                arrays, reason = run_residual_episode(
                    policy,
                    reward,
                    design,
                    horizon=EVAL_EPISODE_STEPS,
                    residual_limit_norm=limit_norm,
                    env_seed=plan["index"],
                    deterministic=True,
                    schedule=push_schedule(plan, dt, EVAL_EPISODE_STEPS),
                )
                metrics = residual_episode_metrics(arrays, reason, reference, dt)
                push_episodes.append(
                    {
                        "plan_index": plan["index"],
                        "force_n": plan["force_n"],
                        "start_s": plan["start_s"],
                        **select_metrics(metrics, PUSH_FIELDS),
                        "return": float(residual_episode_rewards(arrays, reward).sum()),
                        "arrays": arrays,
                    }
                )

            det_record = {
                "seed_index": seed_index,
                "recovered": bool(det_metrics["recovered"]),
                "terminated": bool(det_metrics["terminated"]),
                "settled_at_s": det_metrics["settled_at_s"],
                "capture_time_s": capture_time_s(det_arrays["modes"], dt),
                "return": float(residual_episode_rewards(det_arrays, reward).sum()),
            }
            det_records.append(det_record)
            det_arrays_store[(amp_index, seed_index)] = det_arrays
            det_residual_pool.append(np.asarray(det_arrays["residuals"], dtype=float))
            stoch_residual_pool.extend(
                np.asarray(episode["arrays"]["residuals"], dtype=float) for episode in eval_episodes
            )
            all_eval_episodes.extend({**episode, "limit_n": limit_n} for episode in eval_episodes)
            all_push_episodes.extend({**episode, "limit_n": limit_n} for episode in push_episodes)
            eval_stores[amp_index].append(
                {
                    "states": [episode["arrays"]["states"] for episode in eval_episodes],
                    "controls": [episode["arrays"]["controls"] for episode in eval_episodes],
                    "residuals": [episode["arrays"]["residuals"] for episode in eval_episodes],
                    "lengths": [len(episode["arrays"]["states"]) for episode in eval_episodes],
                    "terminated": [bool(episode["terminated"]) for episode in eval_episodes],
                    "settled": [
                        float(episode["settled_at_s"])
                        if episode["settled_at_s"] is not None
                        else np.nan
                        for episode in eval_episodes
                    ],
                    "returns": [float(episode["return"]) for episode in eval_episodes],
                    "peaks": [
                        float(episode["peak_abs_motor_force_n"]) for episode in eval_episodes
                    ],
                    "max_x": [
                        float(episode["max_abs_cart_position_m"]) for episode in eval_episodes
                    ],
                }
            )
            push_stores[amp_index].append(
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
                "first_successful_eval_steps": next(
                    (
                        int(step)
                        for step, entry in zip(
                            result["eval_steps"], result["eval_records"], strict=True
                        )
                        if entry["success"]
                    ),
                    None,
                ),
                "eval_curve": [
                    {"env_steps": int(step), **entry}
                    for step, entry in zip(
                        result["eval_steps"], result["eval_records"], strict=True
                    )
                ],
                "stochastic": summarize_episodes(eval_episodes),
                "deterministic": det_record,
                "push": summarize_episodes(push_episodes),
            }
            per_seed_records.append(record)
            if log is not None:
                log(
                    f"residual {limit_n:.0f} N seed {seed_index}: steps {record['env_steps']}, "
                    f"wall {record['wall_time_s']:.1f}s, "
                    f"stoch {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                    f"det settled {det_record['settled_at_s']}, "
                    f"push {record['push']['successes']}/{record['push']['episodes']}"
                )

        settled_all = [
            value
            for seed_store in eval_stores[amp_index]
            for value in seed_store["settled"]
            if not np.isnan(value)
        ]
        peaks_all = [
            value for seed_store in eval_stores[amp_index] for value in seed_store["peaks"]
        ]
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
                for seed_store in push_stores[amp_index]
                for value in seed_store["recovery"]
            ],
        }
        sweep_entries.append(
            {
                "limit_n": limit_n,
                "limit_norm": limit_norm,
                "training": per_seed_records,
                "deterministic": det_records,
                "stochastic": stochastic_summary,
                "push": push_summary,
                "residual_stats": {
                    "deterministic": residual_stats(np.concatenate(det_residual_pool), limit_norm),
                    "stochastic": residual_stats(np.concatenate(stoch_residual_pool), limit_norm),
                },
            }
        )

    failure_cases = pick_failure_cases(all_eval_episodes, all_push_episodes)
    annotate_case_limits(failure_cases, all_eval_episodes, all_push_episodes)
    featured = choose_featured_amplitude(sweep_entries)
    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        reward=reward,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        limits_n=limits,
        plans=plans,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        baseline_push_records=baseline_push_records,
        guard=guard,
        sweep_entries=sweep_entries,
        featured=featured,
        failure_cases=failure_cases,
        all_eval_episodes=all_eval_episodes,
        all_push_episodes=all_push_episodes,
        elapsed=elapsed,
    )
    archive = build_archive(
        reward_curves=reward_curves,
        policy_payloads=policy_payloads,
        det_arrays_store=det_arrays_store,
        eval_stores=eval_stores,
        push_stores=push_stores,
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        baseline_push_states=baseline_push_states,
        baseline_push_controls=baseline_push_controls,
        guard=guard,
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
    save_comparison(output / "comparison.png", report, output)
    save_residual_analysis(output / "residual_analysis.png", report, output)
    return report


def build_report(
    *,
    seed,
    config,
    design,
    reward,
    train_seeds,
    eval_seed_count,
    limits_n,
    plans,
    baseline_records,
    baseline_identical,
    baseline_push_records,
    guard,
    sweep_entries,
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
    for entry in sweep_entries:
        three_way.append(
            {
                "label": f"残差 PPO（a={entry['limit_n']:.0f} N）",
                "episodes": entry["stochastic"]["episodes"],
                "successes": entry["stochastic"]["successes"],
                "median_settled_at_s": entry["stochastic"]["median_settled_at_s"],
                "median_peak_abs_motor_force_n": entry["stochastic"][
                    "median_peak_abs_motor_force_n"
                ],
                "source": "this record",
            }
        )

    hypothesis = []
    for entry in sweep_entries:
        aggregate = entry["stochastic"]
        settled = aggregate["median_settled_at_s"]
        peak = aggregate["median_peak_abs_motor_force_n"]
        base_settled = baseline_summary["median_settled_at_s"]
        base_peak = baseline_summary["median_peak_abs_motor_force_n"]
        not_below = aggregate["successes"] >= baseline_summary["successes"]
        settled_delta = None if settled is None else settled - base_settled
        peak_delta = None if peak is None else peak - base_peak
        improved = (settled_delta is not None and settled_delta < 0.0) or (
            peak_delta is not None and peak_delta < 0.0
        )
        if not_below and improved:
            verdict = "supports: success not below baseline and a handoff metric improved"
        elif not_below:
            verdict = "not destructive but no handoff improvement over the baseline"
        else:
            verdict = "degrades the baseline success"
        hypothesis.append(
            {
                "limit_n": entry["limit_n"],
                "successes": aggregate["successes"],
                "episodes": aggregate["episodes"],
                "baseline_successes": baseline_summary["successes"],
                "baseline_episodes": baseline_summary["episodes"],
                "settled_delta_s": settled_delta,
                "peak_delta_n": peak_delta,
                "success_not_below_baseline": bool(not_below),
                "handoff_improved": bool(improved),
                "verdict": verdict,
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
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "control_law": (
                "u = clip(u_energy + clip(u_RL, +/-a), +/-3); u_energy is the unchanged "
                "lesson-7 HybridSwingupController (energy shaping + LQR hysteresis) evaluated "
                "on the same pre-step state; u_RL is the Gaussian policy output"
            ),
            "residual_limits_n": list(limits_n),
            "guard_limit_n": 0.0,
            "ppo_on_preclip_sample": (
                "PPO updates on the pre-clip Gaussian sample; training and execution both "
                "apply clip(u_RL, +/-a), so |u_RL| <= a holds by construction"
            ),
            "reward_action_is_residual": (
                "the reward's control cost acts on the residual the learner chooses; the base "
                "command is not the learner's decision and is not penalized"
            ),
            "train_episode_steps": config.train_episode_steps,
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * design.dt,
            "train_initial_state": {
                "note": (
                    "lesson-29 two-bank curriculum reused: half of the parallel environments "
                    "always restart from the exact resting down start, the rest randomize the "
                    "pole direction and the initial angular velocity"
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
                "physical failure; swing-up time = settled_at_s counted from t=0; handoff = "
                "first entry of the base into its LQR balance mode"
            ),
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lesson 29",
            },
            "reward": {
                **reward.as_dict(),
                "u": (
                    "the clipped residual in normalized units; the control cost therefore "
                    "scales with the budget actually granted to the learner"
                ),
            },
            "reward_scale_for_learning": config.reward_scale,
            "seed_streams": {
                "network_init": "default_rng([master, 7000 + amp, train_seed]); value net [.., 1]",
                "action_sampling": "default_rng([master, 5000 + amp, train_seed])",
                "minibatch_order": "default_rng([master, 9000 + amp, train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*1000 "
                    "+ amp*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lesson 29)",
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
            "claim": (
                "with a = 0 the residual pipeline must reproduce the lesson-7 run bitwise: "
                "same states, same commands, same acceptance"
            ),
            **{key: value for key, value in guard.items() if key != "arrays"},
        },
        "sweep": sweep_entries,
        "featured_amplitude_index": featured,
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; neither the base nor the residual policy saw "
                    "pushes during training; both controllers run the same plans from the "
                    "same exact down start; residual episodes use the mean action"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": (
                    "the same plans are applied to the baseline and to every residual "
                    "amplitude and training seed"
                ),
            },
            "plans": plans,
            "baseline": baseline_push_summary,
            "per_amplitude": [
                {
                    "limit_n": entry["limit_n"],
                    "successes": entry["push"]["successes"],
                    "episodes": entry["push"]["episodes"],
                    "successes_per_seed": entry["push"]["successes_per_seed"],
                    "recovery_times_s": entry["push"]["recovery_times_s"],
                }
                for entry in sweep_entries
            ],
        },
        "three_way_comparison": three_way,
        "hypothesis": {
            "claim": (
                "residual success >= baseline (the base is not broken) AND a recomputable "
                "handoff improvement (swing-up time or input peak); a null or negative "
                "outcome is recorded as the formal conclusion"
            ),
            "per_amplitude": hypothesis,
        },
        "residual_usage_note": (
            "fraction_at_limit close to 1 means the residual budget was saturated (the base "
            "alone was not enough, or exploration noise dominates); fraction_below_half means "
            "the learner left most of the budget unused; an all-zero residual would mean the "
            "policy degenerated to the pure base"
        ),
        "failure_analysis": {
            "eval_counts": failure_counts(
                [
                    {k: v for k, v in episode.items() if k != "arrays"}
                    for episode in all_eval_episodes
                ]
            ),
            "push_counts": failure_counts(
                [
                    {k: v for k, v in episode.items() if k != "arrays"}
                    for episode in all_push_episodes
                ]
            ),
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in failure_cases
            ],
        },
        "training": {
            "train_seeds": train_seeds,
            "amplitudes": list(limits_n),
            "env_steps_per_seed": sweep_entries[0]["training"][0]["env_steps"]
            if sweep_entries and sweep_entries[0]["training"]
            else 0,
            "total_env_steps": sum(
                record["env_steps"] for entry in sweep_entries for record in entry["training"]
            ),
            "wall_time_s_total": elapsed,
            "curves_note": "reward_curve_<amp>_<seed> lives in trajectories.npz",
        },
        "lesson29_reference": LESSON29_PPO_REFERENCE,
        "limitations": [
            (
                "The residual budget grid {25, 50, 100} N is hand-picked; budgets outside it "
                "(including budgets comparable to the base's own saturation) were not tried."
            ),
            (
                "One task (lesson-7 swing-up), one nominal MuJoCo model, no noise/delay/mass "
                "error; no claim about general robot residual-RL practice."
            ),
            (
                "The residual policy may in principle learn to fight the base (e.g., cancel "
                "its pumping); the recorded |u_RL| statistics and trajectories show what it "
                "actually did, but no optimality or safety guarantee is claimed."
            ),
            (
                "The lesson-29 reference numbers in three_way_comparison are imported from "
                "that record (60 reward-only PPO episodes), not re-run here."
            ),
            (
                "Success requires the strict lesson-7 settled tail; the stochastic protocol "
                "carries policy sampling noise (20 episodes per seed), so differences of a "
                "few tenths of a second are within sampling variation."
            ),
            (
                "The push plans were drawn once (master seed) and shared by every controller; "
                "plan-level pairing is exact but the plan sample itself is finite."
            ),
        ],
    }
    return report


def build_archive(
    *,
    reward_curves,
    policy_payloads,
    det_arrays_store,
    eval_stores,
    push_stores,
    baseline_states,
    baseline_controls,
    baseline_push_states,
    baseline_push_controls,
    guard,
    failure_cases,
):
    horizon = EVAL_EPISODE_STEPS
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=np.float64),
        "baseline_controls": np.asarray(baseline_controls, dtype=np.float64),
        "baseline_push_states": stack_trajectories(baseline_push_states, horizon)[0],
        "baseline_push_lengths": stack_trajectories(baseline_push_states, horizon)[1],
        "baseline_push_controls": stack_controls(baseline_push_controls, horizon),
        "guard_states": guard["arrays"]["states"],
        "guard_controls": guard["arrays"]["controls"],
    }
    for (amp_index, seed_index), curve in reward_curves.items():
        archive[f"reward_curve_{amp_index}_{seed_index}"] = curve
    for (amp_index, seed_index), payload in policy_payloads.items():
        for name, array in payload.items():
            archive[f"policy_{amp_index}_{seed_index}_{name}"] = array
    for (amp_index, seed_index), arrays in det_arrays_store.items():
        archive[f"det_states_{amp_index}_{seed_index}"] = arrays["states"]
        archive[f"det_controls_{amp_index}_{seed_index}"] = arrays["controls"]
        archive[f"det_residuals_{amp_index}_{seed_index}"] = arrays["residuals"]
        archive[f"det_base_controls_{amp_index}_{seed_index}"] = arrays["base_controls"]
        archive[f"det_modes_{amp_index}_{seed_index}"] = arrays["modes"]
    for amp_index, seed_stores in eval_stores.items():
        flat_states = [s for store in seed_stores for s in store["states"]]
        flat_controls = [c for store in seed_stores for c in store["controls"]]
        flat_residuals = [r for store in seed_stores for r in store["residuals"]]
        archive[f"eval_states_{amp_index}"] = stack_trajectories(flat_states, horizon)[0]
        archive[f"eval_lengths_{amp_index}"] = np.asarray(
            [n for store in seed_stores for n in store["lengths"]], dtype=int
        )
        archive[f"eval_controls_{amp_index}"] = stack_controls(flat_controls, horizon)
        archive[f"eval_residuals_{amp_index}"] = stack_controls(flat_residuals, horizon)
        archive[f"eval_terminated_{amp_index}"] = np.asarray(
            [t for store in seed_stores for t in store["terminated"]], dtype=bool
        )
        archive[f"eval_settled_s_{amp_index}"] = np.asarray(
            [v for store in seed_stores for v in store["settled"]], dtype=float
        ).reshape(len(seed_stores), -1)
        archive[f"eval_returns_{amp_index}"] = np.asarray(
            [v for store in seed_stores for v in store["returns"]], dtype=float
        ).reshape(len(seed_stores), -1)
        archive[f"eval_peak_force_n_{amp_index}"] = np.asarray(
            [v for store in seed_stores for v in store["peaks"]], dtype=float
        ).reshape(len(seed_stores), -1)
        archive[f"eval_max_x_m_{amp_index}"] = np.asarray(
            [v for store in seed_stores for v in store["max_x"]], dtype=float
        ).reshape(len(seed_stores), -1)
        push_seed_stores = push_stores[amp_index]
        archive[f"push_states_{amp_index}"] = stack_trajectories(
            [s for store in push_seed_stores for s in store["states"]], horizon
        )[0]
        archive[f"push_lengths_{amp_index}"] = np.asarray(
            [n for store in push_seed_stores for n in store["lengths"]], dtype=int
        )
        archive[f"push_recovery_s_{amp_index}"] = np.asarray(
            [v for store in push_seed_stores for v in store["recovery"]], dtype=float
        ).reshape(len(push_seed_stores), -1)
    for index, case in enumerate(failure_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by the summary (used by the demo loader)."""
    amps = len(report["sweep"])
    seeds = report["training"]["train_seeds"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
        "guard_states",
        "guard_controls",
    }
    for amp in range(amps):
        keys.update(f"reward_curve_{amp}_{seed}" for seed in range(seeds))
        keys.update(
            f"policy_{amp}_{seed}_{name}"
            for seed in range(seeds)
            for name in policy_array_names(hidden)
        )
        keys.update(
            f"det_{suffix}_{amp}_{seed}"
            for seed in range(seeds)
            for suffix in ("states", "controls", "residuals", "base_controls", "modes")
        )
        keys.update(
            f"eval_{suffix}_{amp}"
            for suffix in (
                "states",
                "lengths",
                "controls",
                "residuals",
                "terminated",
                "settled_s",
                "returns",
                "peak_force_n",
                "max_x_m",
            )
        )
        keys.update(f"push_{suffix}_{amp}" for suffix in ("states", "lengths", "recovery_s"))
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
    colors = ("#2563eb", "#0f766e", "#b45309", "#7c3aed")
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        for seed in range(seeds):
            axes[0].plot(updates, curves[index][seed], alpha=0.25, linewidth=0.8, color=color)
        axes[0].plot(
            updates,
            curves[index].mean(axis=0),
            color=color,
            linewidth=1.8,
            label=f"a={entry['limit_n']:.0f} N（{seeds} 种子均值）",
        )
    axes[0].set(
        xlabel="PPO 更新轮次",
        ylabel="批内平均原始奖励（每步）",
        title="残差训练奖励：细线 = 单个种子",
    )
    axes[0].legend(fontsize=8)
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
        for seed in range(success.shape[0]):
            axes[1].plot(steps, success[seed], "o--", markersize=4, alpha=0.5, color=color)
        axes[1].plot(
            steps, success.mean(axis=0), "o-", color=color, label=f"a={entry['limit_n']:.0f} N"
        )
    axes[1].set(
        xlabel="环境步数（×1000）",
        ylabel="下方初态验收通过（1=成功）",
        ylim=(-0.08, 1.08),
        yticks=[0, 0.5, 1],
        title="训练中周期评估：均值动作、单回合",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    gear = report["protocol"]["actuator_gear"]
    ref_theta = report["protocol"]["reference_state"][1]
    featured = report["featured_amplitude_index"]
    entry = report["sweep"][featured]
    limit_n = entry["limit_n"]
    seed_index = 0
    handoff_start = report["guard"]["capture_time_s"]
    handoff_end = report["guard"]["settled_at_s"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        baseline_states = data["baseline_states"]
        baseline_controls = data["baseline_controls"]
        det_states = data[f"det_states_{featured}_{seed_index}"]
        det_controls = data[f"det_controls_{featured}_{seed_index}"]
        det_residual = data[f"det_residuals_{featured}_{seed_index}"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    if handoff_start is not None and handoff_end is not None:
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.axvspan(handoff_start, handoff_end, alpha=0.15, color="#fbbf24")
    ts = np.arange(len(baseline_states)) * dt
    edges = np.arange(len(baseline_controls) + 1) * dt
    det_ts = np.arange(len(det_states)) * dt
    det_edges = np.arange(len(det_controls) + 1) * dt
    axes[0, 0].plot(
        ts, np.cos(baseline_states[:, 1] - ref_theta), "--", color="gray", label="基线（能量+LQR）"
    )
    axes[0, 0].plot(
        det_ts,
        np.cos(det_states[:, 1] - ref_theta),
        color="#0f766e",
        label=f"残差 a={limit_n:.0f} N（种子 {seed_index}，均值动作）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        title="同一下方初态：摆起轨迹（黄带 = 基线交接段）",
    )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 1].plot(ts, baseline_states[:, 0], "--", color="gray")
    axes[0, 1].plot(det_ts, det_states[:, 0], color="#2563eb")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].set(ylabel="小车位置（m）", title="小车位置（红点线 = ±2.4 m 失败边界）")
    axes[1, 0].stairs(baseline_controls * gear, edges, color="gray", label="基线")
    axes[1, 0].stairs(det_controls * gear, det_edges, color="#2563eb", label="残差合计输入")
    residual_ts = np.arange(len(det_residual)) * dt
    axes[1, 0].plot(
        residual_ts, det_residual * gear, color="#b45309", linewidth=1.0, label="其中残差 u_RL"
    )
    axes[1, 0].set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入与残差分量")
    axes[1, 0].legend(fontsize=8, loc="upper right")
    labels = ["基线", "纯PPO"] + [f"残差\na={entry['limit_n']:.0f} N" for entry in report["sweep"]]
    successes = [row["successes"] for row in report["three_way_comparison"]]
    totals = [row["episodes"] for row in report["three_way_comparison"]]
    bar_colors = ("#64748b", "#b91c1c", "#2563eb", "#0f766e", "#b45309")
    bars = axes[1, 1].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))],
        width=0.6,
    )
    for bar, success, total in zip(bars, successes, totals, strict=True):
        axes[1, 1].annotate(
            f"{success}/{total}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set(
        ylabel="验收通过率（%）",
        ylim=(0, 112),
        title="三方对照成功率（第 7 课同口径验收）",
    )
    axes[1, 1].tick_params(axis="x", labelsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_residual_analysis(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    gear = report["protocol"]["actuator_gear"]
    featured = report["featured_amplitude_index"]
    entry = report["sweep"][featured]
    seeds = report["training"]["train_seeds"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        pools = [
            np.concatenate([data[f"det_residuals_{index}_{seed}"] for seed in range(seeds)]) * gear
            for index in range(len(report["sweep"]))
        ]
        cases = report["failure_analysis"]["featured_cases"]
        case_arrays = (data["case0_states"], data["case0_controls"]) if cases else None
    baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
    featured_recovery = report["push_test"]["per_amplitude"][featured]["recovery_times_s"]
    plans = report["push_test"]["plans"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    colors = ("#2563eb", "#0f766e", "#b45309")
    bins = np.linspace(0.0, 1.05 * max(report["protocol"]["residual_limits_n"]), 42)
    for index, item in enumerate(report["sweep"]):
        axes[0, 0].hist(
            np.abs(pools[index]),
            bins=bins,
            alpha=0.55,
            color=colors[index % len(colors)],
            label=f"a={item['limit_n']:.0f} N",
        )
    for index, item in enumerate(report["sweep"]):
        axes[0, 0].axvline(
            item["limit_n"], color=colors[index % len(colors)], linestyle=":", linewidth=1.2
        )
    axes[0, 0].set(
        xlabel="|残差 u_RL|（N，均值动作回合合并 3 种子）",
        ylabel="步数",
        title="残差幅值分布（点线 = 各档预算上限）",
    )
    axes[0, 0].legend(fontsize=8)
    xs = np.arange(len(report["sweep"]))
    det_means = [item["residual_stats"]["deterministic"]["mean_abs_n"] for item in report["sweep"]]
    at_limits = [
        item["residual_stats"]["deterministic"]["fraction_at_limit"] for item in report["sweep"]
    ]
    bars = axes[0, 1].bar(xs, at_limits, color=[colors[i % len(colors)] for i in xs], width=0.55)
    for bar, mean_value in zip(bars, det_means, strict=True):
        axes[0, 1].annotate(
            f"均值 {mean_value:.1f} N",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0, 1].set(
        xticks=xs,
        xticklabels=[f"a={item['limit_n']:.0f} N" for item in report["sweep"]],
        ylabel="触限步占比",
        ylim=(0, 1.12),
        title="残差预算使用（|u_RL| ≥ 95% 预算的步占比）",
    )
    plan_indices = np.arange(len(plans))
    axes[1, 0].plot(
        plan_indices,
        [np.nan if v is None else v for v in baseline_recovery],
        "s",
        color="#64748b",
        label="基线",
    )
    # the flat per-amplitude list is seed-major: one marker series per seed
    seed_recovery = np.asarray(
        [np.nan if v is None else v for v in featured_recovery], dtype=float
    ).reshape(report["training"]["train_seeds"], len(plans))
    for seed in range(seed_recovery.shape[0]):
        axes[1, 0].plot(
            plan_indices,
            seed_recovery[seed],
            "o",
            markersize=4,
            alpha=0.7,
            color=colors[featured % len(colors)],
            label=f"残差 a={entry['limit_n']:.0f} N（种子 {seed}）",
        )
    axes[1, 0].set(
        xlabel="推力方案编号",
        ylabel="推力结束后恢复时间（s）",
        title="±200 N 配对推力恢复（缺口 = 未恢复）",
    )
    axes[1, 0].legend(fontsize=8)
    if case_arrays is not None:
        states, _controls = case_arrays
        case = cases[0]
        axes[1, 1].plot(
            np.arange(len(states)) * dt,
            np.cos(states[:, 1] - ref_theta),
            color="#b91c1c",
        )
        axes[1, 1].axhspan(-1, 0, alpha=0.08, color="orange")
        limit_text = f"a={case['limit_n']:.0f} N，" if case.get("limit_n") is not None else ""
        axes[1, 1].set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"失败案例：{limit_text}（{case['failure_reason'] or '未达标'}）",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "本记录没有残差失败回合",
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
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=int, default=EVAL_SEEDS)
    parser.add_argument("--limits", type=float, nargs="+", default=list(RESIDUAL_LIMITS_N))
    parser.add_argument("--updates", type=int, default=UPDATES)
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
            limits_n=tuple(args.limits),
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "guard": {
                    key: report["guard"][key]
                    for key in ("bitwise_identical_states", "bitwise_identical_controls")
                },
                "baseline": report["baseline"]["successes"],
                "residual": {
                    f"a={entry['limit_n']:.0f}N": entry["stochastic"]["successes"]
                    for entry in report["sweep"]
                },
                "push": {
                    f"a={entry['limit_n']:.0f}N": entry["push"]["successes"]
                    for entry in report["sweep"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
