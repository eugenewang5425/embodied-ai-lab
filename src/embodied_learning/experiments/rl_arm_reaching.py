"""Lesson 40: pure-learning reaching on the full-actuated planar 2R arm.

Lesson 39 half-refuted the "full actuation is enough" hypothesis on the
differential car: no exploration cliff, but the arrival level lost to the
target-entropy constant (entropy pinned at -2 nats, alpha collapsed to ~0.006,
0/60 vs the manual controller's 20/20).  This lesson reruns the same test one
task simpler: the lesson-8 planar 2R arm (2 DOF, 2 direct joint drives - no
nonholonomic constraint, no arena walls), with the goal box FIXED and small
([0.1, 0.6] x [-0.3, 0.3] m, >= 0.15 m from the base) instead of the car's
whole-arena resampling, against the lesson-8 analytic baseline (IK + joint PD,
49-pose 2 mm audit, 100% on its acceptance cases) re-run in the SAME MuJoCo
arena on the SAME 20 goals.

Task (protocol, fixed before the run):
    observations  [cos(q1), sin(q1), cos(q2), sin(q2), tau1/0.25, tau2/0.25,
                   gx/0.5, gy/0.5]   (protocol verbatim; NO dq - recorded as
                   the protocol's own partial observability, the previous
                   torque carries one step of actuation memory)
    actions       (tau1, tau2) joint torques, |tau| <= 0.25 N m (the lesson-8
                  motor limit); policy output a in (-1, 1)^2, tau = 0.25 a
    reward        r = -dist(tip, goal) + 10 on arrival (dist < 2 mm AND
                  max|dq| < 0.1 rad/s, terminates) - 10 on leaving the joint
                  envelope (|q| > pi; the MuJoCo model has no mechanical
                  stops, the envelope is a recorded task definition)
    episodes      500 steps x 0.02 s = 10 s; start always q = (-40, 80) deg,
                  dq = 0 (the lesson-8 initial posture); physics is the
                  lesson-8 ArmSimulation reused byte-for-byte (0.002 s RK-ish
                  MuJoCo steps, control every 0.02 s)

Success caliber: the protocol's sustained rule - a window of >= 0.5 s in
which dist < 2 mm AND max|dq| < 0.1 rad/s (arrival_time = window start; the
settled-tail variant held to the end of the record is recorded separately,
the lesson-8 tail style).  Training termination uses the instantaneous
version of the same gate, exactly as lesson 39 terminated on its instant
arrival rule.  Path length / final distance / efficiency reuse the
lesson-21/39 formulas verbatim on the tip trajectory.

Budget: 500k env steps per seed x 3 seeds, one gradient update per env step
(batch 256, buffer 300k), automatic temperature (target entropy -2 = -dim a,
the SAC standard kept unchanged from lesson 39 - that constant is exactly
what the series is probing).  Components reused unchanged from lesson 39:
GaussianPolicy2D, the 2-D ring replay buffer, sample_batch, sac_update,
first_checkpoint; from lesson 35 via lesson 39: MLP, soft targets, automatic
temperature step, Adam (imported from the lesson-28 BC module).  Nothing in
any existing file is modified.

Honesty rule: if pure learning does not reach the lesson-8 baseline within
the budget, the null result is the formal conclusion, decomposed and
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.differential_drive import finite_vector
from embodied_learning.experiments.arm_reaching import INITIAL_Q, audit_geometry
from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.rl_goal_reaching import (
    GaussianPolicy2D,
    GoalReplayBuffer,
    first_checkpoint,
    sac_update,
    sample_batch,
)
from embodied_learning.experiments.rl_goal_reaching import (
    step_reward as terminal_step_reward,
)
from embodied_learning.experiments.sac_swingup import MLP, clone_mlp
from embodied_learning.planar_arm import (
    TORQUE_LIMIT,
    ArmSimulation,
    inverse_kinematics,
    joint_pd,
)
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "rl_arm_reaching_lesson40"
SCHEMA_VERSION = 1

DT = 0.02  # s, the lesson-8 control interval (0.002 s physics x FRAME_SKIP 10)
SIM_DT_TOLERANCE = 1e-12
MAX_EPISODE_STEPS = 500  # 10 s
TORQUE_LIMIT_NM = float(TORQUE_LIMIT)  # 0.25 N m, the lesson-8 motor limit
JOINT_LIMIT_RAD = math.pi  # recorded convention: no mechanical stops in the model
ARRIVAL_RADIUS_M = 0.002  # 2 mm, the lesson-8 tip-error gate
ARRIVAL_SPEED_RAD_S = 0.1  # max|dq| gate of this lesson's protocol
ARRIVAL_BONUS = 10.0
JOINT_LIMIT_PENALTY = 10.0
HOLD_TIME_S = 0.5  # evaluation success must persist this long
GEOMETRY_AUDIT_LIMIT_M = 1e-10  # lesson-8 audit gate (measured ~2.26e-16 there)
GOAL_LOW_M = np.array([0.1, -0.3])  # protocol goal box
GOAL_HIGH_M = np.array([0.6, 0.3])
GOAL_MIN_DISTANCE_M = 0.15  # from the base, inside the 0.1-0.7 m annulus
GOAL_SCALE = 0.5  # observation goal scale
OBS_DIM = 8
ACTION_DIM = 2
SHOWCASE_GOALS = ((0.35, 0.30), (0.35, -0.30))  # the lesson-8 targets A and B
IK_BRANCH = 0  # positive-elbow branch for the analytic baseline (recorded)

TRAIN_STEPS = 500_000
TRAIN_SEEDS = 3
EVAL_GOAL_COUNT = 20
BUFFER_SIZE = 300_000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
TAU = 0.005
ALPHA_INIT = 0.2
ALPHA_LR = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)  # -2, the SAC standard kept from lesson 39
HIDDEN = (64, 64)
WARMUP_STEPS = 1_000  # uniform-random actions before the first update
UPDATE_EVERY_ENV_STEPS = 1
EVAL_EVERY_STEPS = 25_000
REWARD_WINDOW = 1_000  # running per-step reward window recorded at each update
CRITERION_RATE = 0.9  # pre-registered learning bar: every seed >= 18/20

DEFAULT_RESULTS = "results/rl_arm_reaching_2026-09-06"


@dataclass(frozen=True)
class ArmSACConfig:
    train_steps: int = TRAIN_STEPS
    episode_steps: int = MAX_EPISODE_STEPS
    buffer_size: int = BUFFER_SIZE
    batch_size: int = BATCH_SIZE
    gamma: float = GAMMA
    lr: float = LEARNING_RATE
    tau: float = TAU
    alpha_init: float = ALPHA_INIT
    alpha_lr: float = ALPHA_LR
    alpha_min: float = 1e-8
    target_entropy: float = TARGET_ENTROPY
    hidden: tuple[int, ...] = HIDDEN
    warmup_steps: int = WARMUP_STEPS
    update_every_env_steps: int = UPDATE_EVERY_ENV_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS
    eval_goal_count: int = EVAL_GOAL_COUNT
    reward_window: int = REWARD_WINDOW

    def __post_init__(self):
        positive = (
            self.train_steps,
            self.episode_steps,
            self.buffer_size,
            self.batch_size,
            self.gamma,
            self.lr,
            self.tau,
            self.alpha_init,
            self.alpha_lr,
            self.update_every_env_steps,
            self.eval_every_steps,
            self.warmup_steps,
            self.reward_window,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("SAC hyperparameters must be finite and positive")
        if self.target_entropy >= 0.0:
            raise ValueError("target_entropy must be negative (the SAC convention)")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if not 0 < self.warmup_steps <= self.train_steps:
            raise ValueError("warmup_steps must be in (0, train_steps]")
        if self.eval_every_steps > self.train_steps:
            raise ValueError("eval_every_steps cannot exceed train_steps")
        if self.eval_goal_count < len(SHOWCASE_GOALS) + 2:
            raise ValueError("eval_goal_count must leave room for sampled goals")
        if any(units < 1 for units in self.hidden):
            raise ValueError("hidden layer sizes must be positive")


def default_training_config():
    """The pre-registered configuration (one run per seed)."""
    return ArmSACConfig()


# ----------------------------------------------------------------- environment
def step_reward(distance, *, arrived, joint_limit):
    """r = -dist + 10 on arrival - 10 on the joint envelope (protocol).

    Additive structure reused verbatim from the lesson-39 reward; only the
    terminal-penalty trigger is this lesson's joint envelope.
    """
    return terminal_step_reward(distance, arrived=arrived, out_of_bounds=joint_limit)


def observation_for(q, prev_torque, goal):
    """[cos q1, sin q1, cos q2, sin q2, tau_prev/0.25, gx/0.5, gy/0.5]."""
    q = finite_vector(q, 2)
    prev = finite_vector(prev_torque, 2)
    goal = finite_vector(goal, 2)
    return np.array(
        [
            math.cos(q[0]),
            math.sin(q[0]),
            math.cos(q[1]),
            math.sin(q[1]),
            prev[0] / TORQUE_LIMIT_NM,
            prev[1] / TORQUE_LIMIT_NM,
            goal[0] / GOAL_SCALE,
            goal[1] / GOAL_SCALE,
        ]
    )


class ArmReachingEnv:
    """Lesson-8 MuJoCo arm, random goal in the fixed box, torque inputs.

    The SAME step rule serves training, evaluation and the analytic baseline;
    reset(goal=...) pins the goal for paired evaluation and reset() samples
    one from the environment's own goal stream.  One step IS one
    ArmSimulation.step (10 x 0.002 s physics, control held over the step).
    Training episodes terminate on the instantaneous arrival gate (it is the
    +10 reward's terminal event, the lesson-39 convention); evaluation
    episodes set terminate_on_arrival=False so the full horizon stays
    observable for the protocol's SUSTAINED 0.5 s success window.
    """

    def __init__(self, goal_seed, *, episode_steps=MAX_EPISODE_STEPS, terminate_on_arrival=True):
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        self.episode_steps = int(episode_steps)
        self.terminate_on_arrival = bool(terminate_on_arrival)
        self._sim = ArmSimulation()
        if abs(self._sim.dt - DT) > SIM_DT_TOLERANCE:
            raise ValueError("Lesson-8 control dt disagrees with this lesson's dt")
        self._goal_rng = np.random.default_rng(goal_seed)
        self.state = np.zeros(4)
        self.goal = np.zeros(2)
        self.last_torque = np.zeros(2)
        self.step_index = 0

    def reset(self, goal=None):
        self.state = self._sim.reset(INITIAL_Q)
        self.last_torque = np.zeros(2)
        self.goal = self._sample_goal() if goal is None else finite_vector(goal, 2).copy()
        self.step_index = 0
        return observation_for(self.state[:2], self.last_torque, self.goal)

    def _sample_goal(self):
        while True:
            candidate = self._goal_rng.uniform(GOAL_LOW_M, GOAL_HIGH_M)
            if np.linalg.norm(candidate) >= GOAL_MIN_DISTANCE_M:
                return candidate

    @property
    def tip(self):
        return self._sim.points()[-1]

    def step(self, torque):
        torque = finite_vector(torque, 2)
        if np.any(np.abs(torque) > TORQUE_LIMIT_NM + 1e-9):
            raise ValueError("Torque limit violated")
        self.state, applied, failure = self._sim.step(torque)
        self.last_torque = np.asarray(applied, dtype=float).copy()
        self.step_index += 1
        distance = float(np.linalg.norm(self.tip - self.goal))
        speed = float(np.max(np.abs(self.state[2:])))
        joint_limit = bool(np.any(np.abs(self.state[:2]) > JOINT_LIMIT_RAD))
        arrived = bool(distance < ARRIVAL_RADIUS_M and speed < ARRIVAL_SPEED_RAD_S)
        terminated = joint_limit or (arrived and self.terminate_on_arrival) or bool(failure)
        truncated = (not terminated) and self.step_index >= self.episode_steps
        reward = step_reward(distance, arrived=arrived, joint_limit=joint_limit)
        outcome = (
            "failure"
            if failure
            else "joint_limit"
            if joint_limit
            else "arrived"
            if arrived
            else "timeout"
        )
        info = {"outcome": outcome, "distance_m": distance, "speed_rad_s": speed}
        return (
            observation_for(self.state[:2], self.last_torque, self.goal),
            reward,
            (terminated),
            truncated,
            info,
        )


# -------------------------------------------------------------------- actors
def baseline_actor(state, goal, _prev_torque):
    """Lesson-8 analytic baseline: IK (positive elbow) + joint PD, verbatim."""
    goal_q = inverse_kinematics(goal)[IK_BRANCH]
    return joint_pd(state[:2], state[2:], goal_q)


def policy_actor(policy):
    """Deterministic (squashed-mean) SAC actor mapped to N m."""

    def actor(state, goal, prev_torque):
        observation = observation_for(state[:2], prev_torque, goal)[None, :]
        return TORQUE_LIMIT_NM * policy.mean(observation)[0]

    return actor


def sustained_window(distances, speeds, dt):
    """Protocol success: a >= 0.5 s window with dist < 2 mm and speed < gate.

    Returns (success, first_window_start_sample, settled_tail_start_sample).
    The settled tail is the lesson-8 style variant: the condition holds from
    that sample to the END of the record for at least the hold time.
    """
    ok = (np.asarray(distances, dtype=float) < ARRIVAL_RADIUS_M) & (
        np.asarray(speeds, dtype=float) < ARRIVAL_SPEED_RAD_S
    )
    samples = math.floor(HOLD_TIME_S / dt + 1e-9) + 1  # 26 samples span 0.5 s
    if len(ok) < samples:
        return False, None, None
    cumulative = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    windows = cumulative[samples:] - cumulative[:-samples]  # window starting at i
    starts = np.flatnonzero(windows == samples)
    success = bool(len(starts) > 0)
    first = int(starts[0]) if success else None
    settled = None
    if ok[-samples:].all():
        index = len(ok) - 1
        while index >= 0 and ok[index]:
            index -= 1
        settled = index + 1
    return success, first, settled


def run_episode(actor, env, goal, *, record=False):
    """One fixed-goal episode through the env; returns metrics (+points)."""
    env.reset(goal=goal)
    points = [env._sim.points().copy()] if record else None
    distances = [float(np.linalg.norm(env.tip - goal))]
    speeds = [0.0]  # the initial posture starts at rest
    terminal_outcome, arrival_sample = "timeout", None
    steps = 0
    for step in range(1, env.episode_steps + 1):
        _obs, _reward, terminated, truncated, info = env.step(
            actor(env.state, env.goal, env.last_torque)
        )
        steps = step
        distances.append(info["distance_m"])
        speeds.append(info["speed_rad_s"])
        if record:
            points.append(env._sim.points().copy())
        if terminated or truncated:
            terminal_outcome = info["outcome"]
            break
    success, first, settled = sustained_window(distances, speeds, DT)
    if success:
        arrival_sample = first
    outcome = "arrived" if success else terminal_outcome
    points_array = np.asarray(points) if record else None
    tip = points_array[:, -1, :] if record else None
    summary = {
        "outcome": outcome,
        "steps": steps,
        "duration_s": steps * DT,
        "arrival_time_s": None if arrival_sample is None else arrival_sample * DT,
        "settled_tail_s": None if settled is None else settled * DT,
        "min_distance_m": float(np.min(distances)),
    }
    if not record:
        return summary, None
    final_distance = float(np.linalg.norm(tip[-1] - goal))
    straight = float(np.linalg.norm(tip[0] - goal))
    path_length = float(np.linalg.norm(np.diff(tip, axis=0), axis=1).sum())
    summary.update(
        {
            "final_distance_m": final_distance,
            "path_length_m": path_length,
            "straight_distance_m": straight,
            "path_efficiency": path_length / straight if straight > 1e-9 else None,
        }
    )
    return summary, points_array


def aggregate_episodes(episodes):
    """Success count, arrival time, efficiency and closest approach over episodes."""
    successes = [episode for episode in episodes if episode["outcome"] == "arrived"]
    arrivals = [episode["arrival_time_s"] for episode in successes]
    efficiencies = [
        episode["path_efficiency"]
        for episode in successes
        if episode["path_efficiency"] is not None
    ]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "settled_successes": sum(episode["settled_tail_s"] is not None for episode in episodes),
        "joint_limits": sum(episode["outcome"] == "joint_limit" for episode in episodes),
        "failures": sum(episode["outcome"] == "failure" for episode in episodes),
        "median_arrival_time_s": float(np.median(arrivals)) if arrivals else None,
        "median_path_efficiency": float(np.median(efficiencies)) if efficiencies else None,
        "mean_final_distance_m": float(np.mean([e["final_distance_m"] for e in episodes])),
        "median_min_distance_m": float(np.median([e["min_distance_m"] for e in episodes])),
    }


# ------------------------------------------------------------------- training
def quick_eval(policy, env, goals):
    """Deterministic mean-action episodes; outcome and arrival only."""
    actor = policy_actor(policy)
    outcomes, arrivals = [], []
    for goal in goals:
        summary, _points = run_episode(actor, env, goal, record=False)
        outcomes.append(summary["outcome"])
        arrivals.append(summary["arrival_time_s"])
    successes = [value for value in outcomes if value == "arrived"]
    times = [value for value in arrivals if value is not None]
    return {
        "episodes": len(goals),
        "successes": len(successes),
        "median_arrival_time_s": float(np.median(times)) if times else None,
    }


def train_sac_arm(train_env, eval_env, eval_goals, config, *, master_seed, seed_index, log=None):
    """One SAC run (one seed); fixed seed streams, bitwise reproducible."""
    init_seed = [master_seed, 17000, seed_index]
    policy = GaussianPolicy2D(OBS_DIM, config.hidden, init_seed)
    q1 = MLP(OBS_DIM + ACTION_DIM, config.hidden, 1, [*init_seed, 1])
    q2 = MLP(OBS_DIM + ACTION_DIM, config.hidden, 1, [*init_seed, 2])
    target_q1, target_q2 = clone_mlp(q1), clone_mlp(q2)
    policy_optimizer = AdamOptimizer(policy.parameters(), lr=config.lr)
    q1_optimizer = AdamOptimizer(q1.parameters(), lr=config.lr)
    q2_optimizer = AdamOptimizer(q2.parameters(), lr=config.lr)
    action_rng = np.random.default_rng([master_seed, 15000, seed_index])
    buffer_rng = np.random.default_rng([master_seed, 19000, seed_index])
    replay = GoalReplayBuffer(config.buffer_size, OBS_DIM, ACTION_DIM)
    alpha = float(config.alpha_init)

    reward_curve, critic_curve, alpha_curve, entropy_curve = [], [], [], []
    recent = np.zeros(config.reward_window, dtype=float)
    eval_curve = []
    started = time.perf_counter()
    observations = train_env.reset()
    for step in range(config.train_steps):
        if replay.size < config.warmup_steps:
            action = action_rng.uniform(-1.0, 1.0, ACTION_DIM)
        else:
            action, _log_prob = policy.sample(observations[None, :], action_rng)
            action = action[0]
        torque = TORQUE_LIMIT_NM * action
        next_obs, reward, terminated, truncated, _info = train_env.step(torque)
        # store the pre-reset observation; arrival/joint-limit are true
        # terminals (no bootstrap), time-limit truncations keep bootstrapping
        # (the lesson-35/39 rule)
        replay.push(observations, action, reward, next_obs, terminated)
        recent[step % config.reward_window] = reward
        observations = train_env.reset() if (terminated or truncated) else next_obs

        if replay.size >= config.warmup_steps and (step + 1) % config.update_every_env_steps == 0:
            batch = sample_batch(replay, config.batch_size, buffer_rng)
            metrics = sac_update(
                q1,
                q2,
                target_q1,
                target_q2,
                policy,
                q1_optimizer,
                q2_optimizer,
                policy_optimizer,
                batch,
                alpha,
                buffer_rng,
                config,
            )
            alpha = metrics["alpha"]
            reward_curve.append(float(np.mean(recent)))
            critic_curve.append(0.5 * (metrics["q1_loss"] + metrics["q2_loss"]))
            alpha_curve.append(alpha)
            entropy_curve.append(metrics["policy_entropy"])

        env_steps = step + 1
        if env_steps % config.eval_every_steps == 0 or env_steps == config.train_steps:
            counts = quick_eval(policy, eval_env, eval_goals)
            eval_curve.append({"env_steps": env_steps, **counts})
            if log is not None:
                log(
                    f"  seed {seed_index}: steps {env_steps}, updates {len(reward_curve)}, "
                    f"reward {reward_curve[-1]:.3f}, alpha {alpha:.3f}, "
                    f"eval {counts['successes']}/{counts['episodes']}"
                )
    return {
        "policy": policy,
        "curves": {
            "reward_curve": np.asarray(reward_curve, dtype=float),
            "critic_loss_curve": np.asarray(critic_curve, dtype=float),
            "alpha_curve": np.asarray(alpha_curve, dtype=float),
            "entropy_curve": np.asarray(entropy_curve, dtype=float),
        },
        "eval_curve": eval_curve,
        "env_steps": config.train_steps,
        "wall_time_s": time.perf_counter() - started,
        "final_alpha": float(alpha),
        "final_policy_entropy": float(entropy_curve[-1]) if entropy_curve else None,
        "final_reward_mean": float(np.mean(reward_curve[-10:])) if reward_curve else None,
    }


# ------------------------------------------------------------------ experiment
def make_eval_goals(master_seed, count):
    """Eval goal #0/#1 are the lesson-8 showcase targets, the rest are sampled."""
    count = int(count)
    if count < len(SHOWCASE_GOALS) + 2:
        raise ValueError("Need at least the showcase goals plus two sampled ones")
    rng = np.random.default_rng([master_seed, 14000])
    goals = [
        np.asarray(SHOWCASE_GOALS[0], dtype=float),
        np.asarray(SHOWCASE_GOALS[1], dtype=float),
    ]
    while len(goals) < count:
        candidate = rng.uniform(GOAL_LOW_M, GOAL_HIGH_M)
        if np.linalg.norm(candidate) >= GOAL_MIN_DISTANCE_M:
            goals.append(candidate)
    return np.asarray(goals)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, train_seeds=TRAIN_SEEDS, log=print):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(train_seeds) is not int or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    config = config or default_training_config()

    started = time.perf_counter()
    geometry = audit_geometry()
    if geometry["max_point_error_m"] > GEOMETRY_AUDIT_LIMIT_M:
        raise ValueError("Lesson-8 MuJoCo geometry and analytic FK disagree")
    eval_goals = make_eval_goals(seed, config.eval_goal_count)
    eval_env = ArmReachingEnv([seed, 14001], terminate_on_arrival=False)

    baseline_episodes = []
    baseline_truths = []
    for goal in eval_goals:
        summary, points = run_episode(baseline_actor, eval_env, goal, record=True)
        baseline_episodes.append(summary)
        baseline_truths.append(points)
    baseline_aggregate = aggregate_episodes(baseline_episodes)
    if log is not None:
        log(
            f"analytic baseline (lesson-8 IK + PD, positive elbow): "
            f"{baseline_aggregate['successes']}/{baseline_aggregate['episodes']} successes"
        )

    output.mkdir(parents=True, exist_ok=False)
    per_seed_records = []
    curve_stores = {}
    eval_stores = {}
    for seed_index in range(train_seeds):
        train_env = ArmReachingEnv([seed, 12000, seed_index], episode_steps=config.episode_steps)
        result = train_sac_arm(
            train_env,
            eval_env,
            eval_goals,
            config,
            master_seed=seed,
            seed_index=seed_index,
            log=log,
        )
        policy = result["policy"]
        for name, curve in result["curves"].items():
            curve_stores[f"{name}_{seed_index}"] = curve
        eval_curve_steps = np.asarray(
            [point["env_steps"] for point in result["eval_curve"]], dtype=int
        )
        eval_curve_successes = np.asarray(
            [point["successes"] for point in result["eval_curve"]], dtype=int
        )
        curve_stores[f"eval_curve_steps_{seed_index}"] = eval_curve_steps
        curve_stores[f"eval_curve_successes_{seed_index}"] = eval_curve_successes

        episodes, truths = [], []
        actor = policy_actor(policy)
        for goal in eval_goals:
            summary, points = run_episode(actor, eval_env, goal, record=True)
            episodes.append(summary)
            truths.append(points)
        aggregate = aggregate_episodes(episodes)
        outcomes = [episode["outcome"] for episode in episodes]
        eval_stores[seed_index] = {
            "truths": truths,
            "outcomes": outcomes,
            "episodes": episodes,
        }
        record = {
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
                lambda point: point["successes"] >= math.ceil(CRITERION_RATE * point["episodes"]),
            ),
            "per_goal": [{key: value for key, value in episode.items()} for episode in episodes],
            "aggregate": aggregate,
        }
        per_seed_records.append(record)
        if log is not None:
            log(
                f"seed {seed_index}: final eval {aggregate['successes']}/"
                f"{aggregate['episodes']}, median arrival "
                f"{aggregate['median_arrival_time_s']}, median efficiency "
                f"{aggregate['median_path_efficiency']}, wall {result['wall_time_s']:.0f}s"
            )

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        train_seeds=train_seeds,
        eval_goals=eval_goals,
        geometry_audit=geometry,
        baseline_episodes=baseline_episodes,
        baseline_aggregate=baseline_aggregate,
        per_seed_records=per_seed_records,
        elapsed=elapsed,
    )
    archive = build_archive(
        eval_goals=eval_goals,
        baseline_truths=baseline_truths,
        baseline_episodes=baseline_episodes,
        curve_stores=curve_stores,
        eval_stores=eval_stores,
    )
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = digest(output / "trajectories.npz")
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_training_curves(output / "training_curves.png", report, output)
    save_comparison(output / "comparison.png", report, output)
    return report


def build_report(
    *,
    seed,
    config,
    train_seeds,
    eval_goals,
    geometry_audit,
    baseline_episodes,
    baseline_aggregate,
    per_seed_records,
    elapsed,
):
    across = aggregate_episodes(
        [episode for record in per_seed_records for episode in record["per_goal"]]
    )
    seed_successes = [record["aggregate"]["successes"] for record in per_seed_records]
    comparison = [
        {
            "label": "解析 IK + PD（第 8 课基线，正肘解，同一 MuJoCo 竞技场）",
            "source": "planar_arm.inverse_kinematics + joint_pd",
            **baseline_aggregate,
        },
        *[
            {
                "label": f"纯学习 SAC（种子 {record['seed_index']}）",
                "source": "this record",
                **record["aggregate"],
            }
            for record in per_seed_records
        ],
    ]
    return {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "numpy_version": np.__version__,
        "geometry_audit": geometry_audit,
        "protocol": {
            "task": (
                "planar 2R arm reaching (links 0.4/0.3 m), fixed start q = (-40, 80) deg at "
                "rest, goals resampled per training episode inside the FIXED box "
                "[0.1, 0.6] x [-0.3, 0.3] m with distance >= 0.15 m from the base"
            ),
            "route": (
                "pure learning: SAC (replay buffer + twin Q + soft targets + automatic "
                "temperature), hand-written numpy reusing the lesson-39 components; no base "
                "controller, no demonstrations, no reward shaping, no curriculum"
            ),
            "dt_s": DT,
            "horizon_steps": config.episode_steps,
            "horizon_s": config.episode_steps * DT,
            "action": {
                "space": "(tau1, tau2) joint torques in N m",
                "limit_nm": TORQUE_LIMIT_NM,
                "policy_output": "a in (-1, 1)^2 (tanh-squashed), tau = 0.25 * a",
            },
            "observation": {
                "features": (
                    "[cos(q1), sin(q1), cos(q2), sin(q2), tau1/0.25, tau2/0.25, gx/0.5, gy/0.5]"
                ),
                "partial_observability_note": (
                    "RECORDED protocol property: dq is NOT observed; the previous normalized "
                    "torque carries one step of actuation memory. The analytic baseline DOES "
                    "use dq (its PD law needs it) - the information asymmetry runs against "
                    "the learner and is kept because it is the protocol verbatim"
                ),
            },
            "reward": {
                "distance_term": "-dist(tip, goal), evaluated on the state the step lands on",
                "arrival_bonus": ARRIVAL_BONUS,
                "arrival_rule": "dist < 0.002 m AND max|dq| < 0.1 rad/s (terminates)",
                "joint_limit_penalty": JOINT_LIMIT_PENALTY,
                "joint_limit_rule": (
                    "|q1| or |q2| > pi - RECORDED convention: the MuJoCo model has no "
                    "mechanical stops; this envelope is a task definition, not hardware"
                ),
                "terminal_note": (
                    "arrival, joint-limit and simulator-failure exits are true terminals (no "
                    "bootstrap); time-limit truncations bootstrap (the lesson-35/39 convention)"
                ),
            },
            "goal_sampling": {
                "train": (
                    "uniform in [0.1, 0.6] x [-0.3, 0.3] m, distance >= 0.15 m from the base, "
                    "resampled every episode; the whole box lies inside the 0.1-0.7 m annulus"
                ),
                "eval": (
                    "20 fixed goals: showcase #0 (0.35, 0.30) and #1 (0.35, -0.30) - the "
                    "lesson-8 acceptance targets A and B - plus 18 sampled, shared by baseline "
                    "and every seed (paired)"
                ),
            },
            "acceptance": {
                "success": (
                    "a window of >= 0.5 s with dist < 2 mm AND max|dq| < 0.1 rad/s; "
                    "arrival_time = window start; the settled-tail variant (held to the end "
                    "of the record, the lesson-8 style) is reported separately"
                ),
                "training_arrival_rule": (
                    "the instantaneous version of the same gate terminates TRAINING episodes "
                    "and pays the +10 (lesson-39 convention); EVALUATION episodes run the "
                    "full horizon (terminate_on_arrival=False) so the sustained window is "
                    "observable - a fast fly-through never counts as success"
                ),
                "caliber_note": (
                    "path length / final distance / efficiency reuse the lesson-21/39 "
                    "evaluate() formulas verbatim, applied to the tip trajectory; the "
                    "lesson-8 extra gates (joint target error <= 0.01 rad, |dq| <= 0.02 "
                    "rad/s) belong to the IK+PD controller and are not reused"
                ),
                "learning_criterion": f"every seed succeeds on >= {CRITERION_RATE:.0%} of the 20 eval goals",
            },
            "information_parity": (
                "both players act on the TRUE joint state of the same MuJoCo model; the "
                "baseline additionally reads dq (its PD law), the learner does not (protocol "
                "observation verbatim) - recorded asymmetry against the learner"
            ),
            "method_decisions": {
                "gamma": config.gamma,
                "gamma_rationale": "0.99, the SAC standard (as lessons 35/39)",
                "lr": config.lr,
                "tau": config.tau,
                "target_entropy": config.target_entropy,
                "target_entropy_note": (
                    "-dim(a) = -2 kept UNCHANGED from lesson 39: the probe is whether the "
                    "full-actuated SIMPLER task (fixed small goal box, no arena walls) "
                    "neutralizes the same standard configuration"
                ),
                "alpha_init": config.alpha_init,
                "update_schedule": "one gradient update per env step (the textbook SAC cadence)",
                "warmup_steps": config.warmup_steps,
                "warmup_note": "uniform-random actions until warmup transitions are stored",
                "buffer_sampling": (
                    "uniform i.i.d. indices with replacement; batch << buffer makes the "
                    "difference from without-replacement negligible"
                ),
                "reward_smoothing": f"recorded reward is a {config.reward_window}-step running mean",
                "baseline_ik_branch": (
                    "branch 0 (positive elbow) for every eval goal, recorded; lesson 8 "
                    "showed both branches settle (2.70 s vs 4.26 s on target A)"
                ),
            },
            "seed_streams": {
                "network_init": "default_rng([master, 17000, seed]); Q1 [.., 1]; Q2 [.., 2]",
                "action_sampling": "default_rng([master, 15000, seed])",
                "buffer_sampling": "default_rng([master, 19000, seed])",
                "train_goal_stream": "default_rng([master, 12000, seed])",
                "eval_goal_stream": "default_rng([master, 14000])",
                "eval_env_stream": "default_rng([master, 14001]) (goal pinned per episode)",
            },
        },
        "hyperparameters": {**asdict(config), "hidden": list(config.hidden)},
        "baseline": {
            "label": "解析 IK + PD（第 8 课基线，正肘解，同一 MuJoCo 竞技场）",
            "reference": "docs/10-session-08-planar-arm.md (49-pose audit ~2.26e-16 m; 2 mm gate)",
            "aggregate": baseline_aggregate,
            "per_goal": baseline_episodes,
        },
        "rl_evaluation": {
            "protocol": (
                f"{config.eval_goal_count} fixed goals, deterministic (squashed-mean) policy, "
                "same goals as the baseline (paired)"
            ),
            "per_seed": per_seed_records,
            "across_seeds": across,
        },
        "comparison": comparison,
        "hypothesis": {
            "claim": (
                "full actuation (2 DOF / 2 direct drives) plus a FIXED small goal box makes "
                "pure learning sufficient for reaching, one task simpler than the lesson-39 "
                "differential car"
            ),
            "criterion": f"every seed >= {CRITERION_RATE:.0%} of the 20 eval goals",
            "per_seed_successes": seed_successes,
            "all_seeds_meet_criterion": bool(
                seed_successes
                and min(seed_successes) >= math.ceil(CRITERION_RATE * config.eval_goal_count)
            ),
            "baseline_successes": baseline_aggregate["successes"],
            "baseline_level_reached": bool(
                seed_successes and min(seed_successes) >= baseline_aggregate["successes"]
            ),
        },
        "training": {
            "train_seeds": train_seeds,
            "env_steps_per_seed": config.train_steps,
            "total_env_steps": int(config.train_steps * train_seeds),
            "wall_time_s_total": elapsed,
            "curves_note": "reward/critic_loss/alpha/entropy curves per seed live in trajectories.npz",
        },
        "limitations": [
            (
                "Single fixed SAC configuration (gamma 0.99, lr 3e-4, tau 0.005, target entropy "
                "-2, alpha init 0.2, batch 256, buffer 300k, 64x64 nets); no hyperparameter "
                "sweep - a failure cannot separate 'the algorithm' from 'this configuration'."
            ),
            (
                "The observation carries no dq (protocol verbatim) while the PD baseline reads "
                "dq; the task is therefore partially observable for the learner and the "
                "information asymmetry runs against it."
            ),
            (
                "The joint envelope |q| > pi is a recorded task convention (no mechanical "
                "stops in the model); joint-limit counts are about this envelope, not hardware."
            ),
            (
                "Ideal simulation: no friction modelling error, no sensor noise, no payload, "
                "no collision; both players see the true joint state."
            ),
            (
                "20 eval goals per seed is a finite sample; success rates are not population "
                "estimates."
            ),
        ],
    }


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    goals = report["hyperparameters"]["eval_goal_count"]
    curves = ("reward_curve", "critic_loss_curve", "alpha_curve", "entropy_curve")
    per_goal = ("outcome", "arrival_sample", "len", "settled_sample")
    stats = (
        "final_distance",
        "path_length",
        "straight",
        "efficiency",
        "arrival_time",
        "min_distance",
    )
    keys = {"eval_goals"}
    keys.update(f"{curve}_{seed}" for seed in range(seeds) for curve in curves)
    keys.update(
        f"eval_curve_{name}_{seed}" for seed in range(seeds) for name in ("steps", "successes")
    )
    keys.update(f"baseline_truth_{goal}" for goal in range(goals))
    keys.update(f"eval_truth_{seed}_{goal}" for seed in range(seeds) for goal in range(goals))
    keys.update(f"eval_{name}_{seed}" for seed in range(seeds) for name in (*per_goal, *stats))
    keys.update(f"baseline_{name}" for name in (*per_goal, *stats))
    return keys


def build_archive(*, eval_goals, baseline_truths, baseline_episodes, curve_stores, eval_stores):
    archive = {"eval_goals": np.asarray(eval_goals, dtype=float)}
    archive.update(curve_stores)
    for goal_index, truth in enumerate(baseline_truths):
        archive[f"baseline_truth_{goal_index}"] = truth
    for name, values in _episode_arrays(baseline_episodes).items():
        archive[f"baseline_{name}"] = values
    for seed_index, store in eval_stores.items():
        for goal_index, truth in enumerate(store["truths"]):
            archive[f"eval_truth_{seed_index}_{goal_index}"] = truth
        for name, values in _episode_arrays(store["episodes"]).items():
            archive[f"eval_{name}_{seed_index}"] = values
    return archive


def _episode_arrays(episodes):
    """Columnar per-episode fields for the archive (NaN/-1 = not applicable)."""

    def column(name, dtype=float):
        return np.asarray([episode[name] for episode in episodes], dtype=dtype)

    def optional(value, dtype):
        return dtype(-1) if value is None else dtype(round(value))

    return {
        "outcome": column("outcome", dtype="<U16"),
        "arrival_sample": np.asarray(
            [
                optional(e["arrival_time_s"] / DT if e["arrival_time_s"] is not None else None, int)
                for e in episodes
            ],
            dtype=int,
        ),
        "len": np.asarray([episode["steps"] for episode in episodes], dtype=int) + 1,
        "settled_sample": np.asarray(
            [
                optional(e["settled_tail_s"] / DT if e["settled_tail_s"] is not None else None, int)
                for e in episodes
            ],
            dtype=int,
        ),
        "final_distance": column("final_distance_m"),
        "path_length": column("path_length_m"),
        "straight": column("straight_distance_m"),
        "efficiency": np.asarray(
            [
                episode["path_efficiency"] if episode["path_efficiency"] is not None else np.nan
                for episode in episodes
            ],
            dtype=float,
        ),
        "arrival_time": np.asarray(
            [
                episode["arrival_time_s"] if episode["arrival_time_s"] is not None else np.nan
                for episode in episodes
            ],
            dtype=float,
        ),
        "min_distance": column("min_distance_m"),
    }


# -------------------------------------------------------------------- figures
def _plot_stride(array):
    return max(1, len(array) // 2000)


def save_training_curves(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    colors = ("#0f766e", "#2563eb", "#b45309")
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        curves = {
            name: np.asarray([data[f"{name}_{index}"] for index in range(seeds)], dtype=float)
            for name in ("reward_curve", "alpha_curve", "entropy_curve")
        }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    reward = curves["reward_curve"]
    updates = np.arange(1, reward.shape[1] + 1)
    for index in range(seeds):
        stride = _plot_stride(reward[index])
        axes[0, 0].plot(
            updates[::stride],
            reward[index][::stride],
            alpha=0.4,
            linewidth=0.8,
            color=colors[index % 3],
        )
    axes[0, 0].plot(
        updates[::stride],
        reward.mean(axis=0)[::stride],
        color="#111827",
        linewidth=1.6,
        label="3 种子均值",
    )
    axes[0, 0].set(
        xlabel="SAC 更新轮次（每环境步 1 次）",
        ylabel="环境奖励（1000 步滑动均值）",
        title="训练奖励：细线 = 单个种子（r ≈ −末端距离，到达 +10）",
    )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    alpha = curves["alpha_curve"]
    for index in range(seeds):
        stride = _plot_stride(alpha[index])
        axes[0, 1].plot(
            updates[::stride],
            alpha[index][::stride],
            color=colors[index % 3],
            label=f"种子 {index}",
        )
    axes[0, 1].set(
        xlabel="SAC 更新轮次",
        ylabel="温度 α",
        title="α 轨迹（自动温度，目标熵 −2）",
    )
    axes[0, 1].legend(fontsize=8)
    entropy = curves["entropy_curve"]
    for index in range(seeds):
        stride = _plot_stride(entropy[index])
        axes[1, 0].plot(
            updates[::stride],
            entropy[index][::stride],
            color=colors[index % 3],
            label=f"种子 {index}",
        )
    axes[1, 0].set(
        xlabel="SAC 更新轮次",
        ylabel="策略熵（−E[log π]，nats）",
        title="策略熵：探索还活着吗",
    )
    axes[1, 0].legend(fontsize=8)
    for index in range(seeds):
        steps = report["rl_evaluation"]["per_seed"][index]["eval_curve"]
        axes[1, 1].plot(
            [point["env_steps"] / 1000 for point in steps],
            [point["successes"] / point["episodes"] for point in steps],
            "o-",
            markersize=3.5,
            color=colors[index % 3],
            label=f"种子 {index}",
        )
    axes[1, 1].axhline(CRITERION_RATE, color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[1, 1].text(
        0.02,
        CRITERION_RATE + 0.03,
        f"预注册判据 {CRITERION_RATE:.0%}",
        color="#b91c1c",
        fontsize=7.5,
        transform=axes[1, 1].get_yaxis_transform(),
    )
    axes[1, 1].set(
        xlabel="环境步数（×1000）",
        ylabel="周期评估成功率",
        ylim=(-0.05, 1.08),
        title="训练中周期评估（20 个固定目标，确定性策略）",
    )
    axes[1, 1].legend(fontsize=8, loc="lower right")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    comparison = report["comparison"]
    labels = ["IK+PD\n基线", "SAC\n种子 0", "SAC\n种子 1", "SAC\n种子 2"][: len(comparison)]
    colors = ("#64748b", "#0f766e", "#2563eb", "#b45309")[: len(comparison)]
    successes = [row["successes"] for row in comparison]
    totals = [row["episodes"] for row in comparison]
    medians = [row["median_arrival_time_s"] for row in comparison]
    efficiencies = [row["median_path_efficiency"] for row in comparison]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    bars = axes[0].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=colors,
        width=0.55,
    )
    for bar, s, t in zip(bars, successes, totals, strict=True):
        axes[0].annotate(
            f"{s}/{t}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_ylim(0, 115)
    axes[0].set(ylabel="成功率（%）", title="成功率：2 mm 内且 |dq| < 0.1 持续 0.5 s")
    for index, value in enumerate(medians):
        axes[1].bar(index, value if value is not None else 0.0, color=colors[index], width=0.55)
        axes[1].annotate(
            f"{value:.2f} s" if value is not None else "无",
            (index, 0.15 if value is None else value + 0.05),
            ha="center",
            fontsize=9,
        )
    axes[1].set(xlabel="与左图同序", ylabel="到达时间中位（s）", title="到达时间（成功回合中位）")
    for index, value in enumerate(efficiencies):
        axes[2].bar(index, value if value is not None else 0.0, color=colors[index], width=0.55)
        axes[2].annotate(
            f"{value:.2f}" if value is not None else "—",
            (index, 0.03 if value is None else value + 0.02),
            ha="center",
            fontsize=9,
        )
    axes[2].axhline(1.0, color="#111827", linestyle=":", linewidth=1.0)
    axes[2].set(
        xlabel="与左图同序",
        ylabel="路径效率中位（直线/实际）",
        title="路径效率（1.0 = 沿直线；点线 = 理想值）",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
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
    args = parser.parse_args()
    config = ArmSACConfig(
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
            args.output, seed=args.seed, config=config, train_seeds=args.train_seeds, log=log
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["aggregate"],
                "seeds": [
                    {"seed_index": record["seed_index"], **record["aggregate"]}
                    for record in report["rl_evaluation"]["per_seed"]
                ],
                "hypothesis": report["hypothesis"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
