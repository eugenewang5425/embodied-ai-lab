"""Lesson 34: two-phase reward on the pure-PPO stack - switch the goal, not the ladder.

Lesson 29's mechanism 1 was "no gradient below": the upright term of the
lesson-29 reward is flat exactly where a down-start episode begins, so pure PPO
was never paid for pumping energy into the pole (0/60 while the zero-shot
lesson-7 baseline scores 20/20). Lesson 31 repaired the gradient with a
potential-based energy ladder glued on top of the task reward (PBRS: task +
gamma*Phi(s') - Phi(s)); the ladder kept the summit fixed by the Ng theorem but
"right energy" is not "right pose", and the last mile stayed without its own
target. This lesson implements the literature's other standard route for
swing-up - a two-phase learning protocol (MDPI 2024, "reward function and
two-phase learning protocol"; Dulac-Arnold et al. 2021 classify
swing-up-then-balance as a stage-switching problem): instead of adding a term,
the TARGET itself switches with the phase,

    swing phase   |alpha| > alpha_switch:
        r = -cE_sw * |E(s) - E_top| + alive_swing        (energy error only)
    balance phase latched once |alpha| <= alpha_switch:
        r = (1 + cos(alpha))/2 + 0.25 - 0.01*u^2         (the lesson-29 reward)
    failing step (either phase):
        r = phase reward - failure_penalty (1.0)

with E_top = 0 in the task frame and cE_sw = 1/(2*m*g*l), which maps the
resting-down energy error to exactly -1 - the same scale as the task reward's
upright term. The phase judged on the post-step state; the first in-cone
post-step state LATCHES the balance phase until the episode ends (the
non-latched caliber - a pure state function that flips back and forth at the
seam - is discussed in the lecture, not run). The failing step keeps its
state-based phase reward minus a small penalty: lesson-29's -10 replacement
suppressed pumping through the GAE recursion (mechanism 3), but penalty zero
would make "crash now" strictly better than enduring the negative swing stream
(-0.75/step at rest-down), so the penalty is pinned just above that bar at
1.0 - ten times below lesson 29. The swing phase charges no control cost
(pumping needs saturation; lesson 30 showed how taxing the learner's command
strangles it); the balance phase is the lesson-29 shape verbatim.

Everything else is the lesson-29 stack reused verbatim through its public
components: the same 5x64x64 Gaussian PPO, observations, curriculum, budget
(500k steps x 3 seeds), seed streams and the lesson-7 acceptance. With the
phase switch disabled the reward and the whole pipeline degenerate bitwise to
lesson 29 - pinned as the pipeline guard, so any difference below comes from
the two-phase reward alone.

Checks (same conditions throughout): the lesson-7 baseline re-run zero-shot on
the exact down start, the cited lesson-29 pure-PPO row (0/60, never arrived)
and lesson-31 PBRS best-tier row (0/60, one 150k-checkpoint touch), and the new
two-phase row. Process metrics: the upright first arrival (contrast lesson 31),
the first successful periodic evaluation (contrast lessons 29/31/32, all
"never"), the energy trajectory along episodes (is there really sugar on every
rung of the swing phase?), and the phase-switch statistics.

Honesty rule: if the two-phase reward also fails within the budget, the failure
is the formal result with its mechanism analysis. Nothing is smoothed over.
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

from embodied_learning.experiments.dapg_swingup import (
    LESSON29_PPO_REFERENCE,
    PBRS_REFERENCE,
    first_arrival_eval_steps,
    first_success_eval_steps,
)
from embodied_learning.experiments.pbrs_swingup import (
    CAPTURE_ANGLE_RAD,
    arrival_summary,
    failure_label,
    first_arrival_time_s,
    lesson7_energy_constants,
    pole_energy,
)
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
from embodied_learning.swingup import MODEL_PATH, SAFE_CART_POSITION, design_swingup_lqr, wrap_angle

EXPERIMENT = "twophase_swingup_lesson34"
SCHEMA_VERSION = 1

EVAL_EVERY = 25
GEAR = 100.0  # recovery_metrics reports forces as 100 * normalized command

ALPHA_SWITCH_RAD = CAPTURE_ANGLE_RAD  # 0.3 rad: the lesson-7 capture threshold
ALIVE_SWING = 0.25  # same survival bonus in both phases (lesson-29 value)
FAILURE_PENALTY_TWOPHASE = 1.0  # see the module docstring: just above the crash-early bar
LATCHED = True  # design decision: the balance phase holds until the episode ends
DEFAULT_TRAINING_BUDGET_STEPS = 500_000  # 250 updates x 8 envs x 250 steps, per seed

# Final training rewards cited verbatim from the lesson-31 official record
# (results/pbrs_swingup_2026-09-06, docs/36 section 2.2); position reference
# only - the shaped objective is a different quantity and not comparable.
PBRS_FINAL_REWARD_PER_TIER = {
    0.5: [0.8827144982356454, 1.0205072493060332, 1.0501227008230658],
    2.0: [2.5893009152257136, 2.53983577065462, 2.9283402507293212],
}
LESSON32_FIRST_SUCCESS = (
    "never (docs/37 section 2.3: both DAPG tiers 0/60, first success never appeared)"
)


def default_training_config(updates=250):
    """Lesson-29 hyperparameters and budget, unchanged."""
    updates = int(updates)
    if not 1 <= updates <= 1000:
        raise ValueError("updates must be in [1, 1000]")
    return PPOConfig(updates=updates, eval_every=min(EVAL_EVERY, updates))


class TwoPhaseReward:
    """Phase-switching reward judged on the post-step state (latch held by the env).

    The balance phase delegates to the lesson-29 RewardFunction (so its shape is
    the lesson-29 reward bit for bit); the swing phase is the normalized energy
    error. With enabled=False every method degenerates to the lesson-29 reward
    exactly - the pipeline guard's reason to exist.
    """

    def __init__(
        self,
        task_reward,
        hinge_inertia,
        gravity_energy,
        *,
        alpha_switch=ALPHA_SWITCH_RAD,
        c_e_switch=None,
        alive_swing=ALIVE_SWING,
        failure_penalty=FAILURE_PENALTY_TWOPHASE,
        enabled=True,
    ):
        self.task_reward = task_reward
        self.reference = task_reward.reference
        self.hinge_inertia = float(hinge_inertia)
        self.gravity_energy = float(gravity_energy)
        self.alpha_switch = float(alpha_switch)
        # Normalization rule: the resting-down energy error maps to exactly -1,
        # the same scale as the task reward's upright term.
        self.c_e_switch = (
            float(1.0 / (2.0 * gravity_energy)) if c_e_switch is None else float(c_e_switch)
        )
        self.alive_swing = float(alive_swing)
        self.failure_penalty = float(failure_penalty)
        self.enabled = bool(enabled)
        if not 0.0 < self.alpha_switch <= np.pi:
            raise ValueError("alpha_switch must be in (0, pi]")
        if not np.isfinite(self.c_e_switch) or self.c_e_switch <= 0.0:
            raise ValueError("c_e_switch must be finite and positive")
        if self.alive_swing < 0.0 or self.failure_penalty < 0.0:
            raise ValueError("alive_swing and failure_penalty must be non-negative")

    # ------------------------------------------------------------ components
    def energy(self, state):
        """Task-frame mechanical energy: E_top = 0, resting down = -2*m*g*l."""
        return pole_energy(state, self.reference, self.hinge_inertia, self.gravity_energy)

    def _safe_state(self, state):
        state = np.asarray(state, dtype=float)
        return state if np.isfinite(state).all() else np.zeros(4)

    def in_capture_region(self, state):
        """|alpha| <= alpha_switch on the wrapped pole angle (inclusive seam)."""
        state = self._safe_state(state)
        alpha = float(wrap_angle(state[1] - self.reference[1]))
        return bool(abs(alpha) <= self.alpha_switch)

    def swing_terms(self, state, _action):
        energy_reward = -self.c_e_switch * abs(float(self.energy(state)))
        return {
            "upright": 0.0,
            "energy": float(energy_reward),
            "alive": self.alive_swing,
            "control_cost": 0.0,
            "failure": 0.0,
            "total": float(energy_reward + self.alive_swing),
        }

    def balance_terms(self, state, action):
        """The lesson-29 reward shape, delegated verbatim (no failure handling)."""
        base = self.task_reward.terms(state, float(action), False)
        return {
            "upright": base["upright"],
            "energy": 0.0,
            "alive": base["alive"],
            "control_cost": base["control_cost"],
            "failure": 0.0,
            "total": base["total"],
        }

    # ------------------------------------------------------------------ step
    def terms(self, state, action, terminated, latched=False):
        if not self.enabled:
            base = self.task_reward.terms(state, float(action), terminated)
            return {
                "phase": "off",
                "upright": base["upright"],
                "energy": 0.0,
                "alive": base["alive"],
                "control_cost": base["control_cost"],
                "failure": base["failure"],
                "total": base["total"],
            }
        state = self._safe_state(state)
        balance = bool(latched) or self.in_capture_region(state)
        row = self.balance_terms(state, action) if balance else self.swing_terms(state, action)
        if terminated:
            row = {
                **row,
                "failure": self.failure_penalty,
                "total": row["total"] - self.failure_penalty,
            }
        return {"phase": "balance" if balance else "swing", **row}

    def reward(self, state, action, terminated, latched=False):
        if not self.enabled:
            return self.task_reward(state, float(action), terminated)
        return self.terms(state, action, terminated, latched)["total"]

    def __call__(self, state, action, terminated, latched=False):
        return self.reward(state, action, terminated, latched)

    def as_dict(self):
        return {
            "formula_swing": "-cE_sw*|E(s) - E_top| + alive_swing (no control cost: pumping "
            "needs saturation, the lesson-30 lesson)",
            "formula_balance": "(1+cos(alpha))/2 + 0.25 - 0.01*u^2 - the lesson-29 reward shape, "
            "delegated verbatim to the lesson-29 RewardFunction",
            "failure_step": "phase reward - failure_penalty (the state-based reward is kept); "
            "lesson 29 replaced the whole step by -10",
            "phase_judged_on": "the post-step state; the first in-cone post-step state latches "
            "the balance phase until the episode ends (episode restart clears it)",
            "alpha_switch_rad": self.alpha_switch,
            "latched": LATCHED,
            "c_e_switch": self.c_e_switch,
            "c_e_switch_rule": "1/(2*m*g*l_eff): the resting-down energy error maps to exactly "
            "-1, the task reward's upright scale",
            "alive_swing": self.alive_swing,
            "failure_penalty": self.failure_penalty,
            "failure_penalty_rationale": (
                "must keep a crash strictly worse than the worst ongoing swing step "
                "(-0.75/step at rest-down) or 'crash now' dominates the negative stream; "
                "10x below the lesson-29 -10 that suppressed pumping (mechanism 3)"
            ),
            "hinge_inertia_kg_m2": self.hinge_inertia,
            "mgl_eff_j": self.gravity_energy,
            "e_top_j": 0.0,
            "e_rest_down_j": -2.0 * self.gravity_energy,
            "swing_reward_at_rest_down": -1.0 + self.alive_swing,
            "enabled": self.enabled,
            "note": "defined experiment-side; the lesson-7 environment file is unchanged",
        }


class TwoPhaseVecSwingup(VecSwingup):
    """Lesson-29 parallel environments with the phase-switching reward.

    The env owns the per-env latch (the reward itself is stateless across envs):
    reset seeds it from the start state, every step judges the reward with the
    pre-step latch and sets it when the post-step state enters the cone, and an
    internal episode restart re-seeds it from the fresh start. With the switch
    disabled the whole stream is bitwise the lesson-29 pipeline (the guard).
    Phase counters over the training run are exposed by phase_stats().
    """

    def __init__(self, reward, **kwargs):
        super().__init__(reward, **kwargs)
        self.latched = np.zeros(self.n_envs, dtype=bool)
        self.counted_steps = 0
        self.balance_steps = 0
        self.latch_events = 0
        self.failures = 0

    def reset(self):
        observations = np.empty((self.n_envs, STATE_INPUTS))
        self.latched = np.zeros(self.n_envs, dtype=bool)
        self.counted_steps = self.balance_steps = self.latch_events = self.failures = 0
        for index, env in enumerate(self.envs):
            state = self._start_state(index)
            env.unwrapped.set_state(state[:2], state[2:])
            env.unwrapped.data.qfrc_applied[0] = 0.0
            observations[index] = normalize_observation(state, self.reward.reference)
            self.latched[index] = self.reward.in_capture_region(state)
        return observations

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
            command = np.array([np.clip(action, -CONTROL_LIMIT, CONTROL_LIMIT)], np.float32)
            state, _, done, timed_out, _ = env.step(command)
            safe = state if np.isfinite(state).all() else np.zeros(4)
            terminal_obs[index] = normalize_observation(safe, reference)
            pre_latched = bool(self.latched[index])
            rewards[index] = self.reward(safe, float(command[0]), done, pre_latched)
            in_cone = self.reward.in_capture_region(safe)
            self.counted_steps += 1
            self.balance_steps += int(pre_latched or in_cone)
            self.latch_events += int(in_cone and not pre_latched)
            self.failures += int(done)
            terminated[index], truncated[index] = done, timed_out
            if done or timed_out:
                state = self._start_state(index)
                env.unwrapped.set_state(state[:2], state[2:])
                env.unwrapped.data.qfrc_applied[0] = 0.0
                self.latched[index] = self.reward.in_capture_region(state)
            elif in_cone:
                self.latched[index] = True
            live_obs[index] = normalize_observation(state, reference)
        return terminal_obs, rewards, terminated, truncated, live_obs

    def phase_stats(self):
        return {
            "env_steps": int(self.counted_steps),
            "balance_step_fraction": self.balance_steps / max(1, self.counted_steps),
            "latch_events": int(self.latch_events),
            "failures": int(self.failures),
        }


def twophase_guard(task_reward, disabled_reward, steps=40):
    """enabled=False through the two-phase pipeline must equal lesson 29 bitwise.

    Both environments share the base seed (hence the same curriculum jitter
    stream) and are stepped with an identical action sequence covering
    training-style steps and internal episode resets.
    """
    base_seed = 4242
    plain = VecSwingup(task_reward, n_envs=2, episode_steps=16, base_seed=base_seed, task_envs=1)
    switched = TwoPhaseVecSwingup(
        disabled_reward, n_envs=2, episode_steps=16, base_seed=base_seed, task_envs=1
    )
    try:
        actions = np.random.default_rng([7, 31]).uniform(-3.0, 3.0, (steps, 2))
        obs_plain, obs_switched = plain.reset(), switched.reset()
        bitwise_obs = bool(np.array_equal(obs_plain, obs_switched))
        bitwise_states = True
        bitwise_rewards = True
        for chunk in actions:
            out_plain = plain.step(chunk)
            out_switched = switched.step(chunk)
            bitwise_states &= all(
                np.array_equal(a, b) for a, b in zip(out_plain, out_switched, strict=True)
            )
            bitwise_rewards &= bool(np.array_equal(out_plain[1], out_switched[1]))
        return {
            "enabled": False,
            "claim": (
                "with the phase switch disabled the two-phase pipeline must reproduce the "
                "lesson-29 reward pipeline bitwise (same states, observations and rewards) "
                "under an identical action stream"
            ),
            "steps": int(steps),
            "bitwise_identical_rewards": bool(bitwise_rewards),
            "bitwise_identical_states": bool(bitwise_states),
            "bitwise_identical_observations": bool(bitwise_obs),
        }
    finally:
        plain.close()
        switched.close()


def twophase_episode_terms(arrays, reward):
    """Per-step two-phase rewards replayed from stored arrays (latch included).

    Array alignment follows lesson 7: states[k] is the state before action k, so
    step k is judged on states[k+1]; the latch replays from states[0] exactly as
    the live environment seeded it at reset.
    """
    states, controls, end_flags = arrays["states"], arrays["controls"], arrays["end_flags"]
    last = len(controls) - 1
    latched = reward.in_capture_region(states[0])
    rows = {key: [] for key in ("phase", "upright", "energy", "alive", "control_cost", "failure")}
    totals = []
    for step in range(len(controls)):
        terminated = bool(end_flags[0]) and step == last
        row = reward.terms(states[step + 1], float(controls[step]), terminated, latched)
        for key, bucket in rows.items():
            bucket.append(row[key] == "balance" if key == "phase" else row[key])
        totals.append(row["total"])
        latched = latched or reward.in_capture_region(states[step + 1])
    return {
        "phase": np.asarray(rows["phase"], dtype=bool),  # True = balance phase
        "upright": np.asarray(rows["upright"], dtype=float),
        "energy": np.asarray(rows["energy"], dtype=float),
        "alive": np.asarray(rows["alive"], dtype=float),
        "control_cost": np.asarray(rows["control_cost"], dtype=float),
        "failure": np.asarray(rows["failure"], dtype=float),
        "total": np.asarray(totals, dtype=float),
    }


def energy_curve(states, reward):
    """E(s) along a stored state sequence (states[0] is the episode start)."""
    return np.asarray([float(reward.energy(state)) for state in states], dtype=float)


def swing_sugar_fraction(terms, states, reward):
    """Share of swing-phase steps that reduced |E| (is the ladder sugared everywhere?)."""
    energies = energy_curve(states, reward)
    swing = ~terms["phase"]
    if not swing.any():
        return None
    before = np.abs(energies[:-1])[swing]
    after = np.abs(energies[1:])[swing]
    return float(np.mean(after < before))


def episode_phase_summary(terms):
    """Phase bookkeeping for one stored episode."""
    balance = terms["phase"]
    return {
        "balance_step_fraction": float(np.mean(balance)),
        "balance_steps": int(balance.sum()),
        "steps": len(balance),
        "failure_step_penalty": float(terms["failure"].sum()),
    }


def twophase_episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance reused verbatim (the same recovery_metrics)."""
    view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    return recovery_metrics(view, {"failure_reason": failure_reason}, reference, dt)


def evaluate_twophase_policy(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start, two-phase bookkeeping."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, SEED_OFFSET_EVAL, eval_seed])
        arrays, reason = run_policy_episode(
            policy,
            reward,
            reference,
            horizon=EVAL_EPISODE_STEPS,
            env_seed=eval_seed,
            deterministic=False,
            rng=rng,
        )
        metrics = twophase_episode_metrics(arrays, reason, reference, dt)
        terms = twophase_episode_terms(arrays, reward)
        episodes.append(
            {
                "eval_seed": eval_seed,
                **select_metrics(metrics),
                "return": float(terms["total"].sum()),
                "first_arrival_s": first_arrival_time_s(
                    arrays["states"], reference, dt, capture_angle=reward.alpha_switch
                ),
                "phase": episode_phase_summary(terms),
                "sugar_fraction": swing_sugar_fraction(terms, arrays["states"], reward),
                "arrays": arrays,
            }
        )
    return episodes


def deterministic_twophase_episode(policy, reward, reference, dt):
    """One mean-action episode with the full two-phase decomposition archived."""
    arrays, reason = run_policy_episode(
        policy, reward, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
    )
    metrics = twophase_episode_metrics(arrays, reason, reference, dt)
    terms = twophase_episode_terms(arrays, reward)
    record = {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "return": float(terms["total"].sum()),
        "first_arrival_s": first_arrival_time_s(
            arrays["states"], reference, dt, capture_angle=reward.alpha_switch
        ),
        "failure_reason": reason,
        "phase": episode_phase_summary(terms),
        "sugar_fraction": swing_sugar_fraction(terms, arrays["states"], reward),
    }
    payload = {
        "states": arrays["states"],
        "controls": arrays["controls"],
        "energy": energy_curve(arrays["states"], reward),
        "phase": terms["phase"],
        "upright": terms["upright"],
        "energy_r": terms["energy"],
        "alive": terms["alive"],
        "control_cost": terms["control_cost"],
        "failure": terms["failure"],
        "total": terms["total"],
    }
    return record, payload


def phase_summary_over_episodes(episodes):
    """Aggregate phase statistics over stored episodes."""
    fractions = [episode["phase"]["balance_step_fraction"] for episode in episodes]
    sugars = [
        episode["sugar_fraction"] for episode in episodes if episode["sugar_fraction"] is not None
    ]
    return {
        "episodes": len(episodes),
        "mean_balance_step_fraction": float(np.mean(fractions)) if fractions else None,
        "mean_swing_sugar_fraction": float(np.mean(sugars)) if sugars else None,
    }


def choose_featured_seed(records):
    """Deterministic figure pick: most successes, then most arrivals, then earliest."""
    best, best_key = 0, None
    for index, record in enumerate(records):
        settled = record["deterministic"]["settled_at_s"]
        settled_key = -settled if settled is not None else float("-inf")
        key = (record["stochastic"]["successes"], record["arrival_count"], settled_key)
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
    alpha_switch=ALPHA_SWITCH_RAD,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
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
    reward = TwoPhaseReward(
        task_reward, hinge_inertia, gravity_energy, alpha_switch=float(alpha_switch)
    )
    disabled_reward = TwoPhaseReward(
        task_reward, hinge_inertia, gravity_energy, alpha_switch=float(alpha_switch), enabled=False
    )
    guard = twophase_guard(task_reward, disabled_reward)

    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    plans = make_push_plans(dt, eval_seed_count, seed)
    baseline_push_records, baseline_push_states, baseline_push_controls = baseline_push_evaluations(
        design, plans, EVAL_EPISODE_STEPS
    )

    output.mkdir(parents=True, exist_ok=False)
    per_seed_records, det_payloads, reward_curves = [], {}, {}
    policy_payloads = {}
    eval_stores, push_stores = [], []
    all_eval_episodes, all_push_episodes = [], []

    for seed_index in range(train_seeds):
        vec_env = TwoPhaseVecSwingup(
            reward,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=10_000 + seed * 100 + seed_index,  # the lesson-29 env stream
            task_envs=config.task_envs,
        )

        def eval_hook(policy, _env_steps, _reward=reward, _reference=reference, _dt=dt):
            record, _payload = deterministic_twophase_episode(policy, _reward, _reference, _dt)
            return {
                "success": bool(record["recovered"] and not record["terminated"]),
                "settled_at_s": record["settled_at_s"],
                "return": record["return"],
                "first_arrival_s": record["first_arrival_s"],
            }

        result = train_ppo(
            vec_env,
            config=config,
            init_seed=[seed, SEED_OFFSET_INIT + seed_index],
            act_seed=[seed, SEED_OFFSET_ACT + seed_index],
            shuffle_seed=[seed, SEED_OFFSET_SHUFFLE + seed_index],
            eval_hook=eval_hook,
            log=log,
        )
        training_phase_stats = vec_env.phase_stats()
        vec_env.close()
        policy = result["policy"]
        reward_curves[seed_index] = result["reward_curve"]
        policy_payloads[seed_index] = policy.arrays()
        det_record, det_payload = deterministic_twophase_episode(policy, reward, reference, dt)
        det_payloads[seed_index] = det_payload

        eval_episodes = evaluate_twophase_policy(
            policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
        )
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
            metrics = twophase_episode_metrics(arrays, reason, reference, dt)
            terms = twophase_episode_terms(arrays, reward)
            push_episodes.append(
                {
                    "plan_index": plan["index"],
                    "force_n": plan["force_n"],
                    "start_s": plan["start_s"],
                    **select_metrics(metrics, PUSH_FIELDS),
                    "return": float(terms["total"].sum()),
                    "first_arrival_s": first_arrival_time_s(
                        arrays["states"], reference, dt, capture_angle=reward.alpha_switch
                    ),
                    "arrays": arrays,
                }
            )

        all_eval_episodes.extend(eval_episodes)
        all_push_episodes.extend(push_episodes)
        final_window = slice(max(0, config.updates - 10), config.updates)
        arrival_count = sum(
            1 for episode in eval_episodes if episode["first_arrival_s"] is not None
        )
        record = {
            "seed_index": seed_index,
            "env_steps": result["env_steps"],
            "wall_time_s": result["wall_time_s"],
            "final_reward_mean": float(np.mean(result["reward_curve"][final_window])),
            "final_log_std": float(policy.log_std[0]),
            "first_successful_eval_steps": first_success_eval_steps(result),
            "first_arrival_eval_steps": first_arrival_eval_steps(result),
            "arrival_count": arrival_count,
            "eval_curve": [
                {"env_steps": int(step), **point}
                for step, point in zip(result["eval_steps"], result["eval_records"], strict=True)
            ],
            "training_phase_stats": training_phase_stats,
            "stochastic": summarize_episodes(eval_episodes),
            "deterministic": det_record,
            "push": summarize_episodes(push_episodes),
        }
        per_seed_records.append(record)
        eval_stores.append(
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
                "balance": [
                    float(episode["phase"]["balance_step_fraction"]) for episode in eval_episodes
                ],
                "sugar": [
                    float(episode["sugar_fraction"])
                    if episode["sugar_fraction"] is not None
                    else np.nan
                    for episode in eval_episodes
                ],
                "peaks": [float(episode["peak_abs_motor_force_n"]) for episode in eval_episodes],
                "max_x": [float(episode["max_abs_cart_position_m"]) for episode in eval_episodes],
            }
        )
        push_stores.append(
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
        if log is not None:
            log(
                f"seed {seed_index}: steps {record['env_steps']}, "
                f"wall {record['wall_time_s']:.1f}s, "
                f"stoch {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                f"upright arrival {arrival_count}/{len(eval_episodes)}, "
                f"first success {record['first_successful_eval_steps']}, "
                f"train balance {training_phase_stats['balance_step_fraction']:.2f}"
            )

    success_list = [record["stochastic"]["successes"] for record in per_seed_records]
    settled_all = [
        value for store in eval_stores for value in store["settled"] if not np.isnan(value)
    ]
    peaks_all = [value for store in eval_stores for value in store["peaks"]]
    aggregate = {
        "episodes": eval_seed_count * train_seeds,
        "successes": int(sum(success_list)),
        "success_rate": float(np.mean(success_list) / eval_seed_count),
        "successes_per_seed": success_list,
        "median_settled_at_s": float(np.median(settled_all)) if settled_all else None,
        "median_peak_abs_motor_force_n": float(np.median(peaks_all)),
    }
    first_success_any = any(
        record["first_successful_eval_steps"] is not None for record in per_seed_records
    )
    featured_seed = choose_featured_seed(per_seed_records)
    failure_cases = pick_failure_cases(all_eval_episodes, all_push_episodes)
    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        reward=reward,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        plans=plans,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        baseline_push_records=baseline_push_records,
        guard=guard,
        per_seed_records=per_seed_records,
        aggregate=aggregate,
        eval_episodes=all_eval_episodes,
        push_episodes=all_push_episodes,
        first_success_any=first_success_any,
        failure_cases=failure_cases,
        elapsed=elapsed,
    )
    report["featured_seed_index"] = featured_seed
    archive = build_archive(
        reward_curves=reward_curves,
        policy_payloads=policy_payloads,
        det_payloads=det_payloads,
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
    save_phase_analysis(output / "phase_analysis.png", report, output)
    save_four_way(output / "four_way.png", report, output)
    return report


def build_report(
    *,
    seed,
    config,
    design,
    reward,
    train_seeds,
    eval_seed_count,
    plans,
    baseline_records,
    baseline_identical,
    baseline_push_records,
    guard,
    per_seed_records,
    aggregate,
    eval_episodes,
    push_episodes,
    first_success_any,
    failure_cases,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    baseline_push_summary = summarize_episodes(baseline_push_records)
    push_per_seed = [
        {"seed_index": record["seed_index"], **record["push"]} for record in per_seed_records
    ]
    push_successes = [record["push"]["successes"] for record in per_seed_records]
    four_way = [
        {
            "label": "基线（第 7 课能量整形+LQR，零样本）",
            "episodes": baseline_summary["episodes"],
            "successes": baseline_summary["successes"],
            "median_settled_at_s": baseline_summary["median_settled_at_s"],
            "upright_first_arrival": "hand-designed controller (not a learning result)",
            "first_success": "zero-shot",
            "source": "this record",
        },
        {
            "label": "纯 PPO（第 29 课，只凭奖励）",
            "episodes": LESSON29_PPO_REFERENCE["episodes"],
            "successes": LESSON29_PPO_REFERENCE["successes"],
            "median_settled_at_s": LESSON29_PPO_REFERENCE["median_settled_at_s"],
            "upright_first_arrival": LESSON29_PPO_REFERENCE["first_upright_arrival"],
            "first_success": "never (docs/33: no successful periodic evaluation)",
            "source": LESSON29_PPO_REFERENCE["source"],
        },
        {
            "label": "PBRS 最好档（第 31 课，cE=2.0）",
            "episodes": PBRS_REFERENCE["episodes"],
            "successes": PBRS_REFERENCE["successes"],
            "median_settled_at_s": PBRS_REFERENCE["median_settled_at_s"],
            "upright_first_arrival": PBRS_REFERENCE["first_arrival"],
            "first_success": "never (docs/36)",
            "source": PBRS_REFERENCE["source"],
        },
        {
            "label": "两阶段奖励（本课，锁存切换）",
            "episodes": aggregate["episodes"],
            "successes": aggregate["successes"],
            "median_settled_at_s": aggregate["median_settled_at_s"],
            "upright_first_arrival": arrival_summary(eval_episodes),
            "first_success": (
                "first periodic-eval success at "
                f"{min(r['first_successful_eval_steps'] for r in per_seed_records if r['first_successful_eval_steps'] is not None)} steps"
                if first_success_any
                else "never within the budget"
            ),
            "source": "this record",
        },
    ]
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
                "two-phase reward (stage switching): the lesson-29 pure PPO stack is reused "
                "verbatim - no base controller, no teacher, no shaping term - only the reward "
                "target switches with the phase; the literature-standard two-phase protocol "
                "(MDPI 2024; Dulac-Arnold et al. 2021 call swing-up+balance stage-switching)"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "reward": reward.as_dict(),
            "phase_protocol": {
                "alpha_switch_rad": reward.alpha_switch,
                "capture_region": (
                    f"|alpha| <= {reward.alpha_switch:g} rad (the lesson-7 capture threshold; "
                    "inclusive seam, the same caliber as the lesson-31 first arrival)"
                ),
                "latched": LATCHED,
                "latch_rationale": (
                    "the balance phase holds until the episode ends: pumping crosses the seam "
                    "repeatedly and a flipping target would make the value function chase two "
                    "objectives at the seam; the latch mirrors the lesson-7 hysteresis "
                    "philosophy (capture and hold). The non-latched caliber (a pure state "
                    "function that flips back) is discussed in the lecture, not run"
                ),
                "swing_reward_scale": (
                    f"cE_sw = 1/(2*mgl) = {reward.c_e_switch:.6f}: the resting-down energy "
                    "error is exactly -1 per step, the upright term's scale"
                ),
            },
            "out_of_bounds_policy": (
                "no large failure penalty: the failing step keeps its state-based phase reward "
                f"minus {reward.failure_penalty:g} and the episode just ends (restart); lesson "
                "29's -10 replacement suppressed pumping through GAE (mechanism 3)"
            ),
            "reward_scale_for_learning": config.reward_scale,
            "reward_scale_note": (
                "learning uses reward * reward_scale (only rescales the value targets); reported "
                "rewards and returns are unscaled"
            ),
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
                "definition": f"|alpha| <= {reward.alpha_switch:g} rad (the lesson-7 capture angle)",
                "first_arrival": (
                    "first episode step whose stored state satisfies the definition; lesson 29 "
                    "never arrived in any evaluation episode, lesson 31 arrived once during "
                    "training (cE=2.0 seed 1 @ the 150k-step checkpoint)"
                ),
            },
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lesson 29 (the latch itself is not observed)",
            },
            "seed_streams": {
                "network_init": "default_rng([master, 7000 + train_seed]); value net [.., 1]",
                "action_sampling": "default_rng([master, 5000 + train_seed])",
                "minibatch_order": "default_rng([master, 9000 + train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*100 + train_seed "
                    "(the lesson-29 stream, single configuration)"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lessons 29-32)",
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
        "training": {
            "train_seeds": train_seeds,
            "env_steps_per_seed": per_seed_records[0]["env_steps"] if per_seed_records else 0,
            "total_env_steps": sum(record["env_steps"] for record in per_seed_records),
            "wall_time_s_total": elapsed,
            "curves_note": "reward_curve_<seed> lives in trajectories.npz",
        },
        "per_seed": per_seed_records,
        "twophase_evaluation": {
            "protocol": (
                f"{eval_seed_count} stochastic episodes per training seed (sampling noise only; "
                "the initial state is the exact down start), plus one mean-action episode per seed"
            ),
            "aggregate": aggregate,
            "arrival": arrival_summary(eval_episodes),
            "phase": phase_summary_over_episodes(eval_episodes),
        },
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; the two-phase policies never saw pushes during "
                    "training; the same plans run from the same exact down start; mean-action "
                    "episodes"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": "the same plans are applied to the baseline and to every training seed",
            },
            "plans": plans,
            "baseline": baseline_push_summary,
            "per_seed": push_per_seed,
            "aggregate": {
                "episodes": eval_seed_count * train_seeds,
                "successes": int(sum(push_successes)),
                "successes_per_seed": push_successes,
            },
        },
        "four_way_comparison": four_way,
        "references": {
            "lesson29": LESSON29_PPO_REFERENCE,
            "lesson31_pbrs": PBRS_REFERENCE,
            "lesson31_pbrs_final_reward_per_tier": {
                str(k): v for k, v in PBRS_FINAL_REWARD_PER_TIER.items()
            },
            "lesson32_first_success": LESSON32_FIRST_SUCCESS,
        },
        "hypothesis": {
            "claim": (
                "can a two-phase reward - dense energy-error gradient below, balance target "
                "above, no large out-of-bounds penalty - carry the pure lesson-29 PPO stack "
                "from the exact down start to the strict lesson-7 acceptance within the fixed "
                "budget? A null result is recorded as the formal conclusion"
            ),
            "first_success_any": bool(first_success_any),
            "first_successful_eval_steps_per_seed": [
                record["first_successful_eval_steps"] for record in per_seed_records
            ],
            "first_arrival_eval_steps_per_seed": [
                record["first_arrival_eval_steps"] for record in per_seed_records
            ],
        },
        "failure_analysis": {
            "eval_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in eval_episodes]
            ),
            "push_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in push_episodes]
            ),
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in failure_cases
            ],
        },
        "limitations": [
            (
                "alpha_switch = 0.3 rad is a single hand-picked point (the lesson-7 capture "
                "angle), not a sweep; cE_sw follows the normalization rule 1/(2*mgl) and was "
                "not tuned."
            ),
            (
                "The latch is a design decision; the non-latched caliber (pure state function, "
                "flips back at the seam) is discussed in the lecture but not implemented or run."
            ),
            (
                "The latch variable is not in the observation: the reward is not strictly "
                "Markov in the observation, though inside the cone the latch correlates with "
                "|omega|, which is observed."
            ),
            (
                "The swing phase charges no control cost (a recorded hand choice enabling "
                "bang-bang pumping) and the failure penalty 1.0 was pinned by the "
                "crash-early argument, not ablated."
            ),
            (
                "The lesson-29/31 rows in four_way_comparison are imported from those records, "
                "not re-run here; the disabled-switch degeneracy is covered by the bitwise "
                "pipeline guard instead of a full re-run."
            ),
            (
                "One task, one nominal MuJoCo model, no noise/delay/mass error; success "
                "requires the strict lesson-7 settled tail (0.02 m, 0.01 rad, 0.02 m/s, "
                "0.02 rad/s held >= 2 s); 20 episodes per seed is a finite sample."
            ),
        ],
    }
    return report


def build_archive(
    *,
    reward_curves,
    policy_payloads,
    det_payloads,
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
    for seed_index, curve in reward_curves.items():
        archive[f"reward_curve_{seed_index}"] = curve
    for seed_index, payload in policy_payloads.items():
        for name, array in payload.items():
            archive[f"policy_{seed_index}_{name}"] = array
    for seed_index, payload in det_payloads.items():
        for suffix in (
            "states",
            "controls",
            "energy",
            "phase",
            "upright",
            "energy_r",
            "alive",
            "control_cost",
            "failure",
            "total",
        ):
            archive[f"det_{suffix}_{seed_index}"] = payload[suffix]
    for seed_index, store in enumerate(eval_stores):
        archive[f"eval_states_{seed_index}"] = stack_trajectories(store["states"], horizon)[0]
        archive[f"eval_lengths_{seed_index}"] = np.asarray(store["lengths"], dtype=int)
        archive[f"eval_controls_{seed_index}"] = stack_controls(store["controls"], horizon)
        archive[f"eval_terminated_{seed_index}"] = np.asarray(store["terminated"], dtype=bool)
        for name in (
            "settled",
            "returns",
            "arrival",
            "balance",
            "sugar",
            "peaks",
            "max_x",
        ):
            suffix = {
                "settled": "settled_s",
                "returns": "returns",
                "arrival": "first_arrival_s",
                "balance": "balance_fraction",
                "sugar": "sugar_fraction",
                "peaks": "peak_force_n",
                "max_x": "max_x_m",
            }[name]
            archive[f"eval_{suffix}_{seed_index}"] = np.asarray(store[name], dtype=float)
    for seed_index, store in enumerate(push_stores):
        archive[f"push_states_{seed_index}"] = stack_trajectories(store["states"], horizon)[0]
        archive[f"push_lengths_{seed_index}"] = np.asarray(store["lengths"], dtype=int)
        archive[f"push_recovery_s_{seed_index}"] = np.asarray(store["recovery"], dtype=float)
    for index, case in enumerate(failure_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by the summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    det_suffixes = (
        "states",
        "controls",
        "energy",
        "phase",
        "upright",
        "energy_r",
        "alive",
        "control_cost",
        "failure",
        "total",
    )
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
    }
    keys.update(f"reward_curve_{seed}" for seed in range(seeds))
    keys.update(f"det_{suffix}_{seed}" for seed in range(seeds) for suffix in det_suffixes)
    keys.update(
        f"policy_{seed}_{name}" for seed in range(seeds) for name in policy_array_names(hidden)
    )
    keys.update(
        f"eval_{suffix}_{seed}"
        for seed in range(seeds)
        for suffix in (
            "states",
            "lengths",
            "controls",
            "terminated",
            "settled_s",
            "returns",
            "first_arrival_s",
            "balance_fraction",
            "sugar_fraction",
            "peak_force_n",
            "max_x_m",
        )
    )
    keys.update(
        f"push_{suffix}_{seed}"
        for seed in range(seeds)
        for suffix in ("states", "lengths", "recovery_s")
    )
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
        curves = np.stack([data[f"reward_curve_{index}"] for index in range(seeds)], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
    for index in range(seeds):
        axes[0].plot(updates, curves[index], alpha=0.35, linewidth=0.9, color="#64748b")
    axes[0].plot(
        updates,
        curves.mean(axis=0),
        color="#0f766e",
        linewidth=1.8,
        label=f"{seeds} 个训练种子均值",
    )
    lesson29_final = float(np.mean(LESSON29_PPO_REFERENCE["final_reward_mean_per_seed"]))
    axes[0].axhline(
        lesson29_final,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.2,
        label="纯 PPO（第 29 课）末段均值",
    )
    for tier, values in report["references"]["lesson31_pbrs_final_reward_per_tier"].items():
        axes[0].axhline(
            float(np.mean(values)),
            color="#b45309",
            linestyle=":",
            linewidth=1.0,
            label=f"PBRS cE={tier} 末段均值（第 31 课）",
        )
    axes[0].set(
        xlabel="PPO 更新轮次",
        ylabel="批内平均奖励（两阶段口径，每步）",
        title="两阶段奖励训练曲线：细线 = 单个训练种子",
    )
    axes[0].legend(fontsize=7, loc="lower right")
    per_seed = report["per_seed"]
    eval_steps = np.asarray([p["env_steps"] for p in per_seed[0]["eval_curve"]]) / 1000.0
    eval_success = np.asarray(
        [[int(p["success"]) for p in record["eval_curve"]] for record in per_seed], dtype=float
    )
    for index in range(seeds):
        axes[1].plot(
            eval_steps, eval_success[index], "o--", markersize=4, alpha=0.6, color="#64748b"
        )
    axes[1].plot(
        eval_steps,
        eval_success.mean(axis=0),
        "o-",
        color="#b91c1c",
        label=f"{seeds} 个训练种子均值",
    )
    axes[1].set(
        xlabel="环境步数（×1000）",
        ylabel="下方初态验收通过（1=成功）",
        ylim=(-0.08, 1.08),
        yticks=[0, 0.5, 1],
        title="训练中周期评估：均值动作、下方初态",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_phase_analysis(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    reward_protocol = report["protocol"]["reward"]
    alpha_switch = float(reward_protocol["alpha_switch_rad"])
    mgl = float(reward_protocol["mgl_eff_j"])
    c_e_sw = float(reward_protocol["c_e_switch"])
    alive_swing = float(reward_protocol["alive_swing"])
    featured = report["featured_seed_index"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        det_states = data[f"det_states_{featured}"]
        det_energy = data[f"det_energy_{featured}"]
        det_phase = data[f"det_phase_{featured}"]
        det_upright = data[f"det_upright_{featured}"]
        det_energy_r = data[f"det_energy_r_{featured}"]
    latch_step = int(np.argmax(det_phase)) if det_phase.any() else None
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    alphas = np.linspace(-np.pi, np.pi, 1441)
    swing_branch = -c_e_sw * np.abs(mgl * (np.cos(alphas) - 1.0)) + alive_swing
    balance_branch = (1.0 + np.cos(alphas)) / 2.0 + 0.25
    landscape = np.where(np.abs(alphas) <= alpha_switch, balance_branch, swing_branch)
    axes[0, 0].plot(alphas, landscape, color="#0f766e", linewidth=1.6)
    axes[0, 0].axvspan(-alpha_switch, alpha_switch, alpha=0.15, color="#10b981")
    for seam in (-alpha_switch, alpha_switch):
        axes[0, 0].axvline(seam, color="#b45309", linestyle=":", linewidth=1.0)
    axes[0, 0].annotate(
        f"平衡阶段（锁存）\n|α|≤{alpha_switch:g}",
        (0.0, float(balance_branch[720]) - 0.42),
        ha="center",
        fontsize=8,
    )
    axes[0, 0].annotate(
        "荡起阶段：−cE·|E−E_top|\n正下方 = −0.75",
        (np.deg2rad(150), -0.72),
        ha="center",
        fontsize=8,
    )
    axes[0, 0].set(
        xlabel="杆相对直立的角度 α（rad，ω=0、u=0）",
        ylabel="每步奖励",
        ylim=(-1.15, 1.55),
        title="两阶段奖励地形：目标随相位切换（ω=0 剖面）",
    )
    ts = np.arange(len(det_states)) * dt
    axes[0, 1].plot(ts, np.cos(det_states[:, 1] - ref_theta), color="#2563eb", linewidth=1.1)
    axes[0, 1].axhspan(-1, 0, alpha=0.08, color="orange")
    if latch_step is not None:
        axes[0, 1].axvspan(
            latch_step * dt, ts[-1], alpha=0.18, color="#10b981", label="平衡阶段（锁存后）"
        )
        axes[0, 1].axvline(latch_step * dt, color="#b45309", linewidth=1.2, label="锁存时刻")
        axes[0, 1].legend(fontsize=7, loc="lower right")
    axes[0, 1].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"典型回合（种子 {featured}，均值动作）：角度轨迹与相位",
    )
    axes[1, 0].plot(np.arange(len(det_energy)) * dt, det_energy, color="#7c3aed", linewidth=1.2)
    axes[1, 0].axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
    axes[1, 0].axhline(-2.0 * mgl, color="gray", linestyle=":", linewidth=0.9)
    axes[1, 0].annotate("E_top = 0（目标）", (0.02, 0.04), xycoords="axes fraction", fontsize=8)
    axes[1, 0].annotate(
        f"正下方 E = {-2.0 * mgl:.2f} J",
        (0.02, 0.90),
        xycoords="axes fraction",
        fontsize=8,
    )
    axes[1, 0].set(
        ylabel="杆机械能 E（J）",
        xlabel="仿真时间（s）",
        title="能量轨迹：荡起阶段的糖 = |E| 向 0 收敛",
    )
    edges = np.arange(len(det_energy_r) + 1) * dt
    axes[1, 1].stairs(det_energy_r, edges, color="#b45309", label="能量项（荡起）")
    axes[1, 1].stairs(det_upright, edges, color="#0f766e", label="直立项（平衡）")
    axes[1, 1].set(
        ylabel="每步奖励分量",
        xlabel="仿真时间（s）",
        title="每步奖励分解：阶段决定哪一项在给糖",
    )
    axes[1, 1].legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_four_way(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    colors = ("#64748b", "#b91c1c", "#b45309", "#0f766e")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    rows = report["four_way_comparison"]
    labels = [
        "基线\n(第7课)",
        "纯PPO\n(第29课)",
        "PBRS cE=2\n(第31课)",
        "两阶段\n(本课)",
    ]
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    bars = axes[0, 0].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=colors,
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
        title="四方成功率（第 7 课口径，下方初态）",
    )
    axes[0, 0].tick_params(axis="x", labelsize=7)
    arrival = report["twophase_evaluation"]["arrival"]
    arrived = np.sort(
        np.asarray(
            [v for v in arrival["first_arrival_s_per_episode"] if v is not None], dtype=float
        )
    )
    checkpoints = [
        (record["seed_index"], record["first_arrival_eval_steps"])
        for record in report["per_seed"]
        if record["first_arrival_eval_steps"] is not None
    ]
    if len(arrived):
        axes[0, 1].plot(
            np.arange(1, len(arrived) + 1),
            arrived,
            "o",
            markersize=4,
            color="#0f766e",
            label=f"评估回合首达 {len(arrived)}/{arrival['episodes']}",
        )
        axes[0, 1].set(
            xlabel="到达回合序号（按首达时间排序）",
            ylabel="直立区首次到达时刻（s）",
            title=f"直立首达（|α|≤{report['protocol']['reward']['alpha_switch_rad']:g} rad）",
        )
        axes[0, 1].legend(fontsize=7)
    else:
        axes[0, 1].set_xlim(-0.5, 0.5)
        axes[0, 1].set_ylim(-0.5, 0.5)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        if checkpoints:
            detail = "；".join(
                f"种子 {seed} @ {steps / 1000:.0f}k 步" for seed, steps in checkpoints
            )
            note = f"评估回合未进入直立区\n训练检查点触达：{detail}\n（第 31 课：一次 @ 150k 步）"
        else:
            note = "评估回合与训练检查点\n均从未进入直立区\n（第 29 课：从未；第 31 课：一次）"
        axes[0, 1].text(
            0.5, 0.55, note, ha="center", va="center", transform=axes[0, 1].transAxes, fontsize=8
        )
        axes[0, 1].set(title="直立首达")
    first_steps = [record["first_successful_eval_steps"] for record in report["per_seed"]]
    if any(step is not None for step in first_steps):
        bar_x = np.arange(len(first_steps))
        bar_h = [np.nan if v is None else v / 1000.0 for v in first_steps]
        axes[1, 0].bar(bar_x, bar_h, color="#0f766e", width=0.5)
        axes[1, 0].set(
            xlabel="训练种子",
            ylabel="首次成功步数（×1000）",
            title="首次成功（训练中周期评估；第 29/31/32 课从未）",
        )
    else:
        axes[1, 0].text(
            0.5,
            0.55,
            "首次成功：从未（预算内）\n第 29 课：从未；第 31 课：从未；第 32 课：从未",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
            fontsize=8,
        )
        axes[1, 0].set(title="首次成功（训练中周期评估）")
    axes[1, 0].set_xticks([0, 1, 2])
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
            axes[1, 1].set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{failure_label(case)}（{case['kind']}）",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "本记录没有两阶段奖励失败回合",
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
    parser.add_argument("--alpha-switch", type=float, default=ALPHA_SWITCH_RAD)
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
            alpha_switch=args.alpha_switch,
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    featured = report["per_seed"][report["featured_seed_index"]]
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
                "twophase": report["twophase_evaluation"]["aggregate"],
                "arrival": report["twophase_evaluation"]["arrival"]["episodes_with_arrival"],
                "first_success_any": report["hypothesis"]["first_success_any"],
                "push": report["push_test"]["aggregate"]["successes"],
                "train_balance_fraction": [
                    record["training_phase_stats"]["balance_step_fraction"]
                    for record in report["per_seed"]
                ],
                "det_settled_s": featured["deterministic"]["settled_at_s"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
