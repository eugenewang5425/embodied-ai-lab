"""Lesson 39: pure-learning goal reaching on the full-actuated differential car.

Lesson 38 concluded that the cart-pole resists every purely-reward-driven
learner because it is UNDERACTUATED (2 DOF, 1 actuator): the energy must pass
through one input while the acceptance is a strict linearized tail.  This
lesson tests the converse hypothesis on a FULL-ACTUATED system - the
lesson-14/21 differential car (planar pose 3 DOF with the no-slip constraint,
2 independent wheel drives): plain SAC, with NO base controller, NO
demonstrations and NO reward shaping, should learn point goal reaching from
scratch, and is compared against the lesson-21 manual controller re-run in the
SAME arena on the SAME goals with the SAME true-pose information.

Task (protocol, fixed before the run):
    observations  [x/3, y/3, cos(theta), sin(theta), gx/3, gy/3]
    actions       (v_l, v_r) wheel speeds in rad/s, |v| <= 6 (lesson-21 limit)
    reward        r = -dist(pose, goal) + 10 on arrival (dist < 0.05 m AND
                  forward speed < 0.1 m/s) - 10 on leaving the arena
                  (|x| > 3 m or |y| > 3 m); all evaluated on the pose the
                  step lands on, speed from the wheels applied over the step
    goals         uniform in [-2.4, 2.4]^2 with distance >= 0.5 m from the
                  start; start always (0, 0, 0); horizon 1000 steps x 0.04 s
                  = 40 s (the lesson-21 horizon)

The goal coordinates ARE part of the observation (a recorded protocol
decision): with random goals the four pose-only features would leave the goal
unobservable and the task unsolvable for a reason unrelated to actuation.
Both the learner and the re-run manual controller act on the TRUE pose -
odometry drift (lessons 15-21) is deliberately excluded so the comparison
isolates "learned vs hand-designed" on identical information.

Budget: 300k env steps per seed x 3 seeds, one gradient update per env step
(batch 256, buffer 300k), automatic temperature (target entropy -2 = -dim a).
Components reused from lesson 35: numpy MLP, tanh-squashed Gaussian policy
math, soft target updates, automatic-temperature step, Adam.  The ring replay
buffer is re-implemented here as the 2-D-action generalization of the
lesson-35 buffer (its push() flattens action batches and only ever stored
1-D actions); storage and sampling semantics are identical.
Evaluation caliber: path length / final distance use the lesson-21
evaluate() formulas verbatim; success is the lesson-39 arrival rule (5 cm and
0.1 m/s).  The lesson-21 controller thresholds (2 cm stop / 3 cm acceptance)
belong to that controller and are not reused here.

Honesty rule: if pure learning does not reach the manual baseline within the
budget, the null result is the formal conclusion, decomposed and recorded.
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

from embodied_learning.differential_drive import DriveGeometry, finite_vector, integrate_pose
from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.goal_reaching import DT as LESSON21_DT
from embodied_learning.experiments.sac_swingup import (
    LOG_PROB_EPS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    MLP,
    alpha_step,
    clone_mlp,
    soft_update_mlp,
    squashed_gaussian_log_prob,
    tanh_squash_params,
)
from embodied_learning.goal_control import DEFAULT_CONFIG, GEOMETRY, goal_command
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "rl_goal_reaching_lesson39"
SCHEMA_VERSION = 1

DT = 0.04  # s, the lesson-14/21 interval
LESSON_DT_TOLERANCE = 1e-15
MAX_EPISODE_STEPS = 1000  # 40 s, the lesson-21 horizon
WHEEL_LIMIT_RAD_S = 6.0  # the lesson-21 wheel limit
ARENA_HALF_M = 3.0  # |x| > 3 or |y| > 3 is out of bounds
ARRIVAL_RADIUS_M = 0.05  # the lesson-39 success caliber (5 cm)
ARRIVAL_SPEED_M_S = 0.1  # the lesson-39 success caliber (0.1 m/s)
ARRIVAL_BONUS = 10.0
OUT_OF_BOUNDS_PENALTY = 10.0
GOAL_MIN_DISTANCE_M = 0.5  # from the start, keeps episodes non-degenerate
GOAL_MAX_COORD_M = 2.4  # inside the arena with margin
STATE_SCALE_M = 3.0  # observation position scale
OBS_DIM = 6
ACTION_DIM = 2
SHOWCASE_GOALS = ((1.6, 0.8), (-2.2, 1.2))  # eval goal #0 (lesson-21 near), #1 (far, behind)

TRAIN_STEPS = 300_000
TRAIN_SEEDS = 3
EVAL_GOAL_COUNT = 20
BUFFER_SIZE = 300_000
BATCH_SIZE = 256
GAMMA = 0.99
LEARNING_RATE = 3e-4
TAU = 0.005
ALPHA_INIT = 0.2
ALPHA_LR = 3e-4
TARGET_ENTROPY = -float(ACTION_DIM)  # -2, the SAC standard for two actions
HIDDEN = (64, 64)
WARMUP_STEPS = 1_000  # uniform-random actions before the first update
UPDATE_EVERY_ENV_STEPS = 1
EVAL_EVERY_STEPS = 25_000
REWARD_WINDOW = 1_000  # running per-step reward window recorded at each update
CRITERION_RATE = 0.9  # pre-registered learning bar: every seed >= 18/20

SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/rl_goal_reaching_2026-09-06"


@dataclass(frozen=True)
class GoalSACConfig:
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
    return GoalSACConfig()


# ----------------------------------------------------------------- environment
def step_reward(distance, *, arrived, out_of_bounds):
    """r = -dist + 10 on arrival - 10 out of bounds (the protocol, verbatim)."""
    if not np.isfinite(distance):
        raise ValueError("Distance must be finite")
    reward = -float(distance)
    if out_of_bounds:
        reward -= OUT_OF_BOUNDS_PENALTY
    if arrived:
        reward += ARRIVAL_BONUS
    return reward


def observation_for(pose, goal):
    """[x/3, y/3, cos(theta), sin(theta), gx/3, gy/3]; theta may be unwrapped."""
    pose = finite_vector(pose, 3)
    goal = finite_vector(goal, 2)
    return np.array(
        [
            pose[0] / STATE_SCALE_M,
            pose[1] / STATE_SCALE_M,
            math.cos(pose[2]),
            math.sin(pose[2]),
            goal[0] / STATE_SCALE_M,
            goal[1] / STATE_SCALE_M,
        ]
    )


class GoalReachingEnv:
    """Lesson-14/21 kinematics, random goal, wheel-speed inputs in rad/s.

    The SAME step rule serves training, evaluation and the re-run manual
    controller; reset(goal=...) pins the goal for paired evaluation and
    reset() samples one from the environment's own goal stream.
    """

    def __init__(self, goal_seed, *, episode_steps=MAX_EPISODE_STEPS, geometry=GEOMETRY):
        if type(episode_steps) is not int or episode_steps < 1:
            raise ValueError("episode_steps must be a positive integer")
        if not isinstance(geometry, DriveGeometry):
            raise TypeError("geometry must be a DriveGeometry")
        self.episode_steps = int(episode_steps)
        self.geometry = geometry
        self._goal_rng = np.random.default_rng(goal_seed)
        self.pose = np.zeros(3)
        self.goal = np.zeros(2)
        self.step_index = 0

    def reset(self, goal=None):
        self.pose = np.zeros(3)
        self.goal = self._sample_goal() if goal is None else finite_vector(goal, 2).copy()
        self.step_index = 0
        return observation_for(self.pose, self.goal)

    def _sample_goal(self):
        while True:
            candidate = self._goal_rng.uniform(-GOAL_MAX_COORD_M, GOAL_MAX_COORD_M, 2)
            if np.linalg.norm(candidate) >= GOAL_MIN_DISTANCE_M:
                return candidate

    def step(self, wheels):
        wheels = finite_vector(wheels, 2)
        if np.any(np.abs(wheels) > WHEEL_LIMIT_RAD_S + 1e-9):
            raise ValueError("Wheel speed limit violated")
        body_velocity = self.geometry.body_velocity(wheels)
        self.pose = integrate_pose(self.pose, body_velocity, DT)
        self.step_index += 1
        distance = float(np.linalg.norm(self.pose[:2] - self.goal))
        out_of_bounds = bool(abs(self.pose[0]) > ARENA_HALF_M or abs(self.pose[1]) > ARENA_HALF_M)
        speed = float(abs(body_velocity[0]))
        arrived = bool(distance < ARRIVAL_RADIUS_M and speed < ARRIVAL_SPEED_M_S)
        terminated = out_of_bounds or arrived
        truncated = (not terminated) and self.step_index >= self.episode_steps
        reward = step_reward(distance, arrived=arrived, out_of_bounds=out_of_bounds)
        outcome = "out_of_bounds" if out_of_bounds else "arrived" if arrived else "timeout"
        info = {"outcome": outcome, "distance_m": distance, "speed_m_s": speed}
        return observation_for(self.pose, self.goal), reward, terminated, truncated, info


# -------------------------------------------------------------------- actors
def manual_actor(pose, goal):
    """The lesson-21 controller on the true pose (information parity)."""
    return goal_command(pose, goal, DEFAULT_CONFIG)["wheels"]


def policy_actor(policy):
    """Deterministic (squashed-mean) SAC actor mapped to rad/s."""

    def actor(pose, goal):
        action = policy.mean(observation_for(pose, goal)[None, :])[0]
        return WHEEL_LIMIT_RAD_S * action

    return actor


def run_episode(actor, env, goal, *, record=False):
    """One fixed-goal episode through the env; returns metrics (+truth)."""
    env.reset(goal=goal)
    truth = [env.pose.copy()] if record else None
    arrival_step = None
    outcome = "timeout"
    steps = 0
    for step in range(1, env.episode_steps + 1):
        _obs, _reward, terminated, truncated, info = env.step(actor(env.pose, env.goal))
        steps = step
        if record:
            truth.append(env.pose.copy())
        if terminated or truncated:
            outcome = info["outcome"]
            if outcome == "arrived":
                arrival_step = step
            break
    summary = {
        "outcome": outcome,
        "steps": steps,
        "arrival_time_s": None if arrival_step is None else arrival_step * DT,
    }
    if not record:
        return summary, None
    truth_array = np.asarray(truth)
    summary.update(trajectory_metrics(truth_array, env.goal, outcome, arrival_step))
    return summary, truth_array


def trajectory_metrics(truth, goal, outcome, arrival_step, dt=DT):
    """Path/final-distance formulas identical to the lesson-21 evaluate()."""
    steps = len(truth) - 1
    final_distance = float(np.linalg.norm(truth[-1, :2] - goal))
    straight = float(np.linalg.norm(truth[0, :2] - goal))
    path_length = float(np.linalg.norm(np.diff(truth[:, :2], axis=0), axis=1).sum())
    return {
        "duration_s": steps * dt,
        "final_distance_m": final_distance,
        "path_length_m": path_length,
        "straight_distance_m": straight,
        "path_efficiency": path_length / straight if straight > 1e-9 else None,
        "arrival_time_s": None if arrival_step is None else arrival_step * dt,
        "steps": steps,
        "outcome": outcome,
    }


def aggregate_episodes(episodes):
    """Success count, arrival time and path efficiency over recorded episodes."""
    successes = [episode for episode in episodes if episode["outcome"] == "arrived"]
    arrivals = [episode["arrival_time_s"] for episode in successes]
    efficiencies = [episode["path_efficiency"] for episode in successes]
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "out_of_bounds": sum(episode["outcome"] == "out_of_bounds" for episode in episodes),
        "median_arrival_time_s": float(np.median(arrivals)) if arrivals else None,
        "median_path_efficiency": float(np.median(efficiencies)) if efficiencies else None,
        "mean_final_distance_m": float(np.mean([e["final_distance_m"] for e in episodes])),
    }


# -------------------------------------------------------------- SAC machinery
class GaussianPolicy2D:
    """OBS-2x64 trunk -> [mu_raw, log_std_raw] per action dim; a = tanh(u).

    The multi-dim generalization of the lesson-35 policy: the joint log
    probability sums the per-dim squashed Gaussian densities.
    """

    def __init__(
        self,
        input_dim,
        hidden,
        seed,
        action_dim=ACTION_DIM,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
    ):
        self.trunk = MLP(input_dim, hidden, int(action_dim) * 2, seed)
        self.action_dim = int(action_dim)
        self.action_bound = 1.0
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def heads(self, obs):
        out, cache = self.trunk.forward(obs)
        return out[:, : self.action_dim], out[:, self.action_dim :], cache

    def distribution(self, obs):
        mean_raw, log_std_raw, cache = self.heads(obs)
        mean, log_std = tanh_squash_params(
            mean_raw, log_std_raw, self.action_bound, self.log_std_min, self.log_std_max
        )
        return mean, log_std, cache

    def mean(self, obs):
        """Deterministic (squashed-mean) actions, shape (B, action_dim)."""
        mean_raw, _log_std_raw, _cache = self.heads(obs)
        return self.action_bound * np.tanh(mean_raw)

    def sample(self, obs, rng):
        mean, log_std, _cache = self.distribution(obs)
        u = mean + np.exp(log_std) * rng.standard_normal(mean.shape)
        action = self.action_bound * np.tanh(u)
        log_prob = squashed_gaussian_log_prob(u, mean, log_std, self.action_bound).sum(axis=1)
        return action, log_prob

    def parameters(self):
        return self.trunk.parameters()


def state_action(observations, actions):
    return np.concatenate([np.asarray(observations, dtype=float), actions], axis=1)


def td_targets(rewards, terminated, next_obs, target_q1, target_q2, policy, rng, alpha, gamma):
    """y = r + gamma (1 - term) (min Qbar - alpha log pi(a'|s'))."""
    rewards = np.asarray(rewards, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    next_actions, next_log_prob = policy.sample(next_obs, rng)
    sa = state_action(next_obs, next_actions)
    q1 = target_q1.forward(sa)[0][:, 0]
    q2 = target_q2.forward(sa)[0][:, 0]
    soft_value = np.minimum(q1, q2) - alpha * next_log_prob
    return rewards + gamma * (1.0 - terminated.astype(float)) * soft_value


def critic_loss_and_gradients(q_network, observations, actions, targets):
    """MSE toward the fixed targets; hand gradients (actions stay 2-D)."""
    output, cache = q_network.forward(state_action(observations, actions))
    deltas = output[:, 0] - np.asarray(targets, dtype=float)
    loss = float(np.mean(deltas**2))
    delta = (2.0 * deltas / float(len(deltas))).reshape(-1, 1)
    grad_weights, grad_biases, _ = q_network.backward(cache, delta)
    return loss, [*grad_weights, *grad_biases]


def critic_values_and_input_grads(q1, q2, observations, actions):
    """min(Q1, Q2) plus dQ/da of the argmin network (last columns)."""
    sa = state_action(observations, actions)
    out1, cache1 = q1.forward(sa)
    out2, cache2 = q2.forward(sa)
    value1, value2 = out1[:, 0], out2[:, 0]
    use_one = value1 <= value2
    ones = np.ones_like(value1).reshape(-1, 1)
    grad_input1 = q1.backward(cache1, ones, need_input_grad=True)[2]
    grad_input2 = q2.backward(cache2, ones, need_input_grad=True)[2]
    grad_input = np.where(use_one.reshape(-1, 1), grad_input1, grad_input2)
    return np.minimum(value1, value2), grad_input[:, -actions.shape[1] :]


def policy_loss_and_gradients(policy, observations, q_value, q_grad, rng, alpha):
    """L = mean(alpha log pi(a|s) - min Q(s, a)); analytic per-dim gradients.

    Generalization of the lesson-35 1-D derivation to two action dims (each
    dim follows the same chain rule through its own tanh; the joint log
    probability sums the dims).  Finite-difference checked in the tests.
    """
    observations = np.asarray(observations, dtype=float)
    q_value = np.asarray(q_value, dtype=float).reshape(-1)
    q_grad = np.asarray(q_grad, dtype=float)
    if q_grad.shape != (len(observations), policy.action_dim):
        raise ValueError("q_grad must carry one gradient per action dim")
    mean_raw, log_std_raw, cache = policy.heads(observations)
    mean, log_std = tanh_squash_params(
        mean_raw, log_std_raw, policy.action_bound, policy.log_std_min, policy.log_std_max
    )
    noise = rng.standard_normal(mean.shape)
    u = mean + np.exp(log_std) * noise
    action = policy.action_bound * np.tanh(u)
    log_prob = squashed_gaussian_log_prob(u, mean, log_std, policy.action_bound)
    joint_log_prob = log_prob.sum(axis=1)
    loss = float(np.mean(alpha * joint_log_prob - q_value))
    tanh_u = np.tanh(u)
    t2 = tanh_u**2
    inv = 1.0 / (1.0 - t2 + LOG_PROB_EPS)
    d_u = 2.0 * tanh_u * (1.0 - t2) * inv
    mu_prime = policy.action_bound * (1.0 - np.tanh(mean_raw) ** 2)
    h_l = 0.5 * (policy.log_std_max - policy.log_std_min) * (1.0 - np.tanh(log_std_raw) ** 2)
    scale = policy.action_bound * (1.0 - t2)
    sigma = np.exp(log_std)
    count = float(len(observations))
    grad_mu = mu_prime * (alpha * d_u - q_grad * scale) / count
    grad_logstd = (
        h_l * (alpha * (d_u * sigma * noise - 1.0) - q_grad * scale * sigma * noise) / count
    )
    grad_weights, grad_biases, _ = policy.trunk.backward(
        cache, np.concatenate([grad_mu, grad_logstd], axis=1)
    )
    metrics = {
        "loss": loss,
        "joint_log_prob": joint_log_prob.copy(),
        "entropy": -float(np.mean(joint_log_prob)),
        "action_abs_mean": float(np.mean(np.abs(action))),
    }
    return metrics, [*grad_weights, *grad_biases]


def sample_batch(replay, batch_size, rng):
    """Uniform i.i.d. indices (with replacement; batch << buffer)."""
    if batch_size > replay.size:
        raise ValueError("not enough transitions stored yet")
    indices = rng.integers(0, replay.size, int(batch_size))
    return {
        "observations": replay.observations[indices],
        "actions": replay.actions[indices],
        "rewards": replay.rewards[indices],
        "next_obs": replay.next_obs[indices],
        "terminated": replay.terminated[indices],
    }


class GoalReplayBuffer:
    """Ring buffer of (obs, action, reward, next_obs, terminated) with 2-D actions.

    The lesson-35 ReplayBuffer is reused everywhere else (MLP, policy math,
    soft updates, temperature step); its push() flattens the action batch
    because it only ever stored 1-D actions, so it cannot carry (B, 2) wheel
    commands.  This is that ring buffer generalized to multi-dim actions:
    storage layout, terminal flag semantics and uniform sampling are identical.
    """

    def __init__(self, capacity, state_dim, action_dim):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("action_dim must be a positive integer")
        self.capacity = capacity
        self.observations = np.empty((capacity, state_dim), dtype=float)
        self.actions = np.empty((capacity, action_dim), dtype=float)
        self.rewards = np.empty(capacity, dtype=float)
        self.next_obs = np.empty((capacity, state_dim), dtype=float)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.size = 0
        self.cursor = 0

    def push(self, observation, action, reward, next_observation, terminated):
        """One transition; arrival and out-of-bounds pass terminated=True."""
        observation = finite_vector(observation, self.observations.shape[1])
        action = finite_vector(action, self.actions.shape[1])
        next_observation = finite_vector(next_observation, self.observations.shape[1])
        if not np.isfinite(reward) or not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("Reward must be finite and terminated a single bool")
        index = self.cursor % self.capacity
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.next_obs[index] = next_observation
        self.terminated[index] = bool(terminated)
        self.cursor += 1
        self.size = min(self.capacity, self.size + 1)


def sac_update(
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
    rng,
    config,
):
    """One SAC update: twin-Q critic, then policy, then alpha, then targets."""
    observations, actions = batch["observations"], batch["actions"]
    sa = state_action(observations, actions)
    q1_out, q1_cache = q1.forward(sa)
    q2_out, q2_cache = q2.forward(sa)
    targets = td_targets(
        batch["rewards"],
        batch["terminated"],
        batch["next_obs"],
        target_q1,
        target_q2,
        policy,
        rng,
        alpha,
        config.gamma,
    )
    count = float(len(actions))
    delta1 = (2.0 * (q1_out[:, 0] - targets) / count).reshape(-1, 1)
    delta2 = (2.0 * (q2_out[:, 0] - targets) / count).reshape(-1, 1)
    grad_weights1, grad_biases1, _ = q1.backward(q1_cache, delta1)
    grad_weights2, grad_biases2, _ = q2.backward(q2_cache, delta2)
    q1_loss = float(np.mean((q1_out[:, 0] - targets) ** 2))
    q2_loss = float(np.mean((q2_out[:, 0] - targets) ** 2))
    q1_optimizer.step(q1.parameters(), [*grad_weights1, *grad_biases1])
    q2_optimizer.step(q2.parameters(), [*grad_weights2, *grad_biases2])

    # the policy gradient uses the UPDATED critics (fresh forward per network)
    q_value, q_grad = critic_values_and_input_grads(q1, q2, observations, actions)
    metrics, policy_grads = policy_loss_and_gradients(
        policy, observations, q_value, q_grad, rng, alpha
    )
    alpha = alpha_step(alpha, metrics["joint_log_prob"], config.target_entropy, config.alpha_lr)
    policy_optimizer.step(policy.parameters(), policy_grads)
    soft_update_mlp(target_q1, q1, config.tau)
    soft_update_mlp(target_q2, q2, config.tau)
    return {
        "q1_loss": q1_loss,
        "q2_loss": q2_loss,
        "policy_loss": metrics["loss"],
        "policy_entropy": metrics["entropy"],
        "action_abs_mean": metrics["action_abs_mean"],
        "alpha": alpha,
    }


# ----------------------------------------------------------------- evaluation
def quick_eval(policy, env, goals):
    """Deterministic mean-action episodes; outcome and arrival only."""
    actor = policy_actor(policy)
    outcomes, arrivals = [], []
    for goal in goals:
        summary, _truth = run_episode(actor, env, goal, record=False)
        outcomes.append(summary["outcome"])
        arrivals.append(summary["arrival_time_s"])
    successes = [value for value in outcomes if value == "arrived"]
    times = [value for value in arrivals if value is not None]
    return {
        "episodes": len(goals),
        "successes": len(successes),
        "median_arrival_time_s": float(np.median(times)) if times else None,
    }


def first_checkpoint(eval_curve, predicate):
    return next(
        (point["env_steps"] for point in eval_curve if predicate(point)),
        None,
    )


# ------------------------------------------------------------------- training
def train_sac_goal(train_env, eval_env, eval_goals, config, *, master_seed, seed_index, log=None):
    """One SAC run (one seed); fixed seed streams, bitwise reproducible."""
    init_seed = [master_seed, 7000, seed_index]
    policy = GaussianPolicy2D(OBS_DIM, config.hidden, init_seed)
    q1 = MLP(OBS_DIM + ACTION_DIM, config.hidden, 1, [*init_seed, 1])
    q2 = MLP(OBS_DIM + ACTION_DIM, config.hidden, 1, [*init_seed, 2])
    target_q1, target_q2 = clone_mlp(q1), clone_mlp(q2)
    policy_optimizer = AdamOptimizer(policy.parameters(), lr=config.lr)
    q1_optimizer = AdamOptimizer(q1.parameters(), lr=config.lr)
    q2_optimizer = AdamOptimizer(q2.parameters(), lr=config.lr)
    action_rng = np.random.default_rng([master_seed, 5000, seed_index])
    buffer_rng = np.random.default_rng([master_seed, 9000, seed_index])
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
        wheels = WHEEL_LIMIT_RAD_S * action
        next_obs, reward, terminated, truncated, _info = train_env.step(wheels)
        # store the pre-reset observation; arrival/OOB are true terminals (no
        # bootstrap), time-limit truncations keep bootstrapping (lesson-35 rule)
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
    """Eval goal #0/#1 are the fixed showcase goals, the rest are sampled."""
    count = int(count)
    if count < len(SHOWCASE_GOALS) + 2:
        raise ValueError("Need at least the showcase goals plus two sampled ones")
    rng = np.random.default_rng([master_seed, 4000])
    goals = [np.asarray(SHOWCASE_GOALS[0], dtype=float), np.asarray(SHOWCASE_GOALS[1], dtype=float)]
    while len(goals) < count:
        candidate = rng.uniform(-GOAL_MAX_COORD_M, GOAL_MAX_COORD_M, 2)
        if np.linalg.norm(candidate) >= GOAL_MIN_DISTANCE_M:
            goals.append(candidate)
    return np.asarray(goals)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, train_seeds=TRAIN_SEEDS, log=print):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if abs(LESSON21_DT - DT) > LESSON_DT_TOLERANCE:
        raise ValueError("Lesson-21 dt disagrees with this lesson's dt")
    if type(train_seeds) is not int or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    config = config or default_training_config()

    started = time.perf_counter()
    eval_goals = make_eval_goals(seed, config.eval_goal_count)
    eval_env = GoalReachingEnv([seed, 4100])

    baseline_episodes = []
    baseline_truths = []
    for goal in eval_goals:
        summary, truth = run_episode(manual_actor, eval_env, goal, record=True)
        baseline_episodes.append(summary)
        baseline_truths.append(truth)
    baseline_aggregate = aggregate_episodes(baseline_episodes)
    if log is not None:
        log(
            f"manual baseline (lesson-21 controller, true pose): "
            f"{baseline_aggregate['successes']}/{baseline_aggregate['episodes']} successes"
        )

    output.mkdir(parents=True, exist_ok=False)
    per_seed_records = []
    curve_stores = {}
    eval_stores = {}
    for seed_index in range(train_seeds):
        train_env = GoalReachingEnv([seed, 4200, seed_index], episode_steps=config.episode_steps)
        result = train_sac_goal(
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
            summary, truth = run_episode(actor, eval_env, goal, record=True)
            episodes.append(summary)
            truths.append(truth)
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
            "label": "手工控制器（第 21 课控制器，真值位姿复跑）",
            "source": "goal_control.goal_command on the true pose",
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
        "protocol": {
            "task": (
                "differential-car point goal reaching, start (0, 0, 0), random goals, "
                "arena |x|,|y| <= 3 m"
            ),
            "route": (
                "pure learning: SAC (replay buffer + twin Q + soft targets + automatic "
                "temperature), hand-written numpy reusing the lesson-35 components; no base "
                "controller, no demonstrations, no reward shaping, no curriculum"
            ),
            "dt_s": DT,
            "horizon_steps": config.episode_steps,
            "horizon_s": config.episode_steps * DT,
            "action": {
                "space": "(v_l, v_r) wheel speeds in rad/s",
                "limit_rad_s": WHEEL_LIMIT_RAD_S,
                "policy_output": "a in (-1, 1)^2 (tanh-squashed), wheels = 6 * a",
            },
            "observation": {
                "features": "[x/3, y/3, cos(theta), sin(theta), gx/3, gy/3]",
                "goal_in_observation_note": (
                    "RECORDED DEVIATION from the four pose-only features: with random goals the "
                    "goal must be observable, otherwise the task is unsolvable for a reason "
                    "unrelated to actuation; the goal coordinates are appended unscaled by "
                    "learning choices"
                ),
            },
            "reward": {
                "distance_term": "-dist(pose, goal), evaluated on the pose the step lands on",
                "arrival_bonus": ARRIVAL_BONUS,
                "arrival_rule": "dist < 0.05 m AND forward speed < 0.1 m/s",
                "out_of_bounds_penalty": OUT_OF_BOUNDS_PENALTY,
                "out_of_bounds_rule": "|x| > 3 m or |y| > 3 m",
                "terminal_note": (
                    "arrival and out-of-bounds are true terminals (no bootstrap); time-limit "
                    "truncations bootstrap (the lesson-35 convention)"
                ),
            },
            "goal_sampling": {
                "train": "uniform in [-2.4, 2.4]^2, distance >= 0.5 m from the start, resampled every episode",
                "eval": "20 fixed goals: showcase #0 (1.6, 0.8) and #1 (-2.2, 1.2) + 18 sampled, shared by baseline and every seed (paired)",
            },
            "acceptance": {
                "success": "arrival rule fires within the horizon (dist < 0.05 m AND speed < 0.1 m/s)",
                "caliber_note": (
                    "path length and final distance use the lesson-21 evaluate() formulas "
                    "verbatim; the lesson-21 controller thresholds (2 cm stop, 3 cm acceptance) "
                    "belong to that controller and are not reused"
                ),
                "learning_criterion": f"every seed succeeds on >= {CRITERION_RATE:.0%} of the 20 eval goals",
            },
            "information_parity": (
                "both the learner and the re-run lesson-21 controller act on the TRUE pose; "
                "odometry drift is deliberately excluded from this lesson"
            ),
            "method_decisions": {
                "gamma": config.gamma,
                "gamma_rationale": "0.99, the SAC standard (as lesson 35)",
                "lr": config.lr,
                "tau": config.tau,
                "target_entropy": config.target_entropy,
                "alpha_init": config.alpha_init,
                "update_schedule": "one gradient update per env step (the textbook SAC cadence)",
                "warmup_steps": config.warmup_steps,
                "warmup_note": "uniform-random actions until warmup transitions are stored",
                "buffer_sampling": (
                    "uniform i.i.d. indices with replacement; batch << buffer makes the "
                    "difference from without-replacement negligible and avoids the O(buffer) "
                    "permutation cost at 300k updates"
                ),
                "reward_smoothing": f"recorded reward is a {config.reward_window}-step running mean",
            },
            "seed_streams": {
                "network_init": "default_rng([master, 7000, seed]); Q1 [.., 1]; Q2 [.., 2]",
                "action_sampling": "default_rng([master, 5000, seed])",
                "buffer_sampling": "default_rng([master, 9000, seed])",
                "train_goal_stream": "default_rng([master, 4200, seed])",
                "eval_goal_stream": "default_rng([master, 4000])",
                "eval_env_stream": "default_rng([master, 4100]) (goal pinned per episode)",
            },
        },
        "hyperparameters": {**asdict(config), "hidden": list(config.hidden)},
        "baseline": {
            "label": "第 21 课手工控制器（goal_command，真值位姿复跑）",
            "reference": "docs/23-session-21-goal-feedback.md (original: noisy estimates, fused 20/20 near, 11/20 far)",
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
                "full actuation removes the exploration cliff, so pure learning should master "
                "goal reaching that the underactuated cart-pole resisted in lessons 29-38"
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
                "Pure kinematics: no odometry drift, no sensor noise, no slip, no dynamics - "
                "both players see the true pose; the lesson-21 original ran on noisy estimates."
            ),
            (
                "The lesson-21 far goal (4.8, 1.2) lies OUTSIDE this arena (|x| <= 3); the "
                "comparison uses this arena's goal distribution instead of that coordinate."
            ),
            (
                "20 eval goals per seed is a finite sample; success rates are not population "
                "estimates."
            ),
            (
                "Success is the lesson-39 arrival rule (5 cm, 0.1 m/s); it does not require the "
                "lesson-21 0.4 s settled tail, so the two calibers are related but not identical."
            ),
        ],
    }


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    goals = report["hyperparameters"]["eval_goal_count"]
    curves = ("reward_curve", "critic_loss_curve", "alpha_curve", "entropy_curve")
    per_goal = ("outcome", "arrival_step", "len")
    stats = ("final_distance", "path_length", "straight", "efficiency", "arrival_time")
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
    """Columnar per-episode fields for the archive (NaN = not applicable)."""

    def column(name, dtype=float):
        return np.asarray([episode[name] for episode in episodes], dtype=dtype)

    return {
        "outcome": column("outcome", dtype="<U16"),
        "arrival_step": np.asarray(
            [
                round(episode["arrival_time_s"] / DT)
                if episode["arrival_time_s"] is not None
                else -1
                for episode in episodes
            ],
            dtype=int,
        ),
        "len": np.asarray([episode["steps"] for episode in episodes], dtype=int) + 1,
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
        title="训练奖励：细线 = 单个种子（r ≈ -距离，到达 +10）",
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
    labels = ["手工\n控制器", "SAC\n种子 0", "SAC\n种子 1", "SAC\n种子 2"][: len(comparison)]
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
    axes[0].set(ylabel="成功率（%）", title="成功率：到达 5 cm 内且速度 < 0.1")
    for index, value in enumerate(medians):
        axes[1].bar(index, value if value is not None else 0.0, color=colors[index], width=0.55)
        axes[1].annotate(
            f"{value:.1f} s" if value is not None else "无",
            (index, 0.4 if value is None else value + 0.3),
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
        ylabel="路径效率中位（实际/直线）",
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
    config = GoalSACConfig(
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
            config=config,
            train_seeds=args.train_seeds,
            log=log,
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
