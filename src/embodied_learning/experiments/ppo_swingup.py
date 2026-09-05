"""Lesson 29: PPO swing-up from reward only, against the lesson-7 energy baseline.

The lesson-7 hybrid controller (energy shaping + LQR hysteresis switch) encodes
model knowledge by hand. This lesson asks the opposite question: with no model
knowledge injected - only a reward signal and trial-and-error - can a plain PPO
agent learn the same down-start swing-up task, how sample-hungry is it, and how
does the learned policy compare with the hand-designed zero-shot baseline on
disturbance recovery (which it never saw during training)?

Everything RL-side is hand-written numpy: a 5x64x64 ReLU Gaussian policy mean
head with a state-independent exploration std plus a same-shape value network,
hand backpropagation (finite-difference checked), GAE(lambda), clipped
surrogate, value loss, entropy bonus, the tested lesson-28 hand Adam, and
global-norm gradient clipping. No torch, no stable-baselines3.

Two hand choices recorded as part of the method (protocol fields below):
cos/sin pole coordinates (a wrapped angle is discontinuous exactly at the
bottom, where a swing-up policy must act), and a training-start curriculum -
half of the parallel environments always restart from the exact resting down
start, the rest randomize the pole direction over the full circle and the
initial angular velocity. Exploration probes (recorded in the lecture) show
why: from the resting down start, action noise never lifts the pole past
horizontal within an episode, so a pure down-start curriculum never sees the
upright region at all. Every evaluation runs from the exact resting down
start regardless.

The environment is the unchanged lesson-7 swing-up task; the reward is defined
experiment-side (in RL the reward IS part of the method, so it is recorded here
rather than hidden - the lecture discusses it as the hidden hand-design):

    r = (1 + cos(alpha)) / 2     upright shaping; alpha = wrapped pole angle
        + alive_bonus            survival bonus on every non-failing step
        - control_cost * u^2     u = normalized motor command in [-3, 3]
        - failure_penalty        replaces the step reward on the failing step

Checks (same conditions throughout, lesson-7 acceptance reused verbatim):
(1) baseline: the lesson-7 energy+LQR controller, zero-shot, run 20 times on
    the exact down start with the lesson-7 recovery_metrics (deterministic, so
    all 20 repeats coincide - recorded, not hidden);
(2) PPO: 3 training seeds, training curves (raw reward and periodic settled
    evaluation vs environment steps), then 20 stochastic episodes per seed on
    the same down start with the same acceptance;
(3) disturbance: lesson-5-style +/-200 N random pushes (the same 20 plans and
    the same exact down start for both controllers, mean-action PPO episodes)
    that PPO never saw - the sim-to-real gap is reported;
(4) failures: featured failing episodes plus out-of-bounds/divergence counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)

EXPERIMENT = "ppo_swingup_lesson29"

STATE_INPUTS = 5  # x, cos(alpha), sin(alpha), v, omega: continuous through the bottom seam
HIDDEN = (64, 64)
LOG_STD_INIT = 1.0  # initial standard deviation; the network stores log(1.0) = 0
N_ENVS = 8
ROLLOUT_STEPS = 250
UPDATES = 250
EPOCHS = 10
MINIBATCH = 500
GAMMA = 0.995
GAE_LAMBDA = 0.98
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
ENT_COEF = 0.001
VF_COEF = 0.5
GRAD_CLIP_NORM = 0.5
REWARD_SCALE = 0.1
LEARN_STD = False  # pilots: a learned std collapses exploration within ~50 updates
TASK_ENVS = 4  # envs that always restart from the exact resting down start
TRAIN_EPISODE_STEPS = 250
EVAL_EPISODE_STEPS = 750
EVAL_EVERY = 25

UPRIGHT_WEIGHT = 1.0
ALIVE_BONUS = 0.25
CONTROL_COST_COEF = 0.01
FAILURE_PENALTY = 10.0
CONTROL_LIMIT = 3.0
OBS_SCALE = (2.5, 5.0, 10.0)  # cart position, cart velocity, pole angular velocity
# Training-start randomization (recorded in the summary; the standard RL
# random-start curriculum). Exploration probes: from the resting down start,
# sigma<=1.5 action noise never lifts the pole past horizontal in 20 s, so the
# upright region is never visited and no gradient toward it exists. Random
# starts put the near-upright capture on the training distribution; evaluation
# always runs from the exact resting down start of the lesson-7 task.
INIT_ANGLE_JITTER = float(np.pi)
INIT_OMEGA_JITTER = 5.0

DOWN_ANGLE_DEG = -180.0
EVAL_SEEDS = 20
TRAIN_SEEDS = 3
PUSH_FORCE_N = 200.0
PUSH_DURATION_S = 0.2
PUSH_START_WINDOW_S = (6.0, 8.0)

SEED_OFFSET_PUSH, SEED_OFFSET_EVAL, SEED_OFFSET_INIT = 3000, 2000, 7000
SEED_OFFSET_ACT, SEED_OFFSET_SHUFFLE, SEED_OFFSET_JITTER = 5000, 9000, 6000

EVAL_FIELDS = (
    "recovered",
    "terminated",
    "truncated",
    "settled_at_s",
    "peak_abs_motor_force_n",
    "max_abs_cart_position_m",
    "failure_reason",
)
PUSH_FIELDS = (*EVAL_FIELDS, "recovery_after_push_end_s")


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = N_ENVS
    rollout_steps: int = ROLLOUT_STEPS
    updates: int = UPDATES
    epochs: int = EPOCHS
    minibatch: int = MINIBATCH
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    clip_eps: float = CLIP_EPS
    lr: float = LEARNING_RATE
    ent_coef: float = ENT_COEF
    vf_coef: float = VF_COEF
    grad_clip: float = GRAD_CLIP_NORM
    log_std_init: float = LOG_STD_INIT
    learn_std: bool = LEARN_STD
    reward_scale: float = REWARD_SCALE
    hidden: tuple[int, ...] = HIDDEN
    train_episode_steps: int = TRAIN_EPISODE_STEPS
    eval_every: int = EVAL_EVERY
    task_envs: int = TASK_ENVS

    def __post_init__(self):
        positive = (
            self.n_envs,
            self.rollout_steps,
            self.updates,
            self.epochs,
            self.minibatch,
            self.gamma,
            self.gae_lambda,
            self.clip_eps,
            self.lr,
            self.vf_coef,
            self.grad_clip,
            self.reward_scale,
            self.train_episode_steps,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("PPO hyperparameters must be finite and positive")
        if self.ent_coef < 0 or self.log_std_init <= 0:
            raise ValueError("ent_coef must be >= 0 and log_std_init must be > 0")
        if self.minibatch > self.n_envs * self.rollout_steps:
            raise ValueError("minibatch cannot exceed the rollout size")
        if not 0 < self.eval_every <= self.updates:
            raise ValueError("eval_every must be in (0, updates]")
        if any(units < 1 for units in self.hidden):
            raise ValueError("hidden layer sizes must be positive")
        if not 0 <= self.task_envs <= self.n_envs:
            raise ValueError("task_envs must be between 0 and n_envs")


class RewardFunction:
    """Experiment-side reward evaluated on the unchanged lesson-7 environment."""

    def __init__(
        self,
        reference,
        upright_weight=UPRIGHT_WEIGHT,
        alive_bonus=ALIVE_BONUS,
        control_cost_coef=CONTROL_COST_COEF,
        failure_penalty=FAILURE_PENALTY,
    ):
        self.reference = np.asarray(reference, dtype=float)
        self.upright_weight = float(upright_weight)
        self.alive_bonus = float(alive_bonus)
        self.control_cost_coef = float(control_cost_coef)
        self.failure_penalty = float(failure_penalty)
        for value in (upright_weight, alive_bonus, control_cost_coef, failure_penalty):
            if not np.isfinite(value):
                raise ValueError("reward constants must be finite")

    def terms(self, state, action, terminated):
        """Reward for one transition, judged by the post-step state."""
        action = float(action)
        if terminated:
            return {
                "upright": 0.0,
                "alive": 0.0,
                "control_cost": 0.0,
                "failure": self.failure_penalty,
                "total": -self.failure_penalty,
            }
        alpha = float(wrap_angle(state[1] - self.reference[1]))
        upright = self.upright_weight * (1.0 + np.cos(alpha)) / 2.0
        control_cost = self.control_cost_coef * action * action
        total = upright + self.alive_bonus - control_cost
        return {
            "upright": float(upright),
            "alive": self.alive_bonus,
            "control_cost": float(control_cost),
            "failure": 0.0,
            "total": float(total),
        }

    def __call__(self, state, action, terminated):
        return self.terms(state, action, terminated)["total"]

    def as_dict(self):
        return {
            "formula": (
                "r = upright_weight*(1+cos(alpha))/2 + alive_bonus - control_cost_coef*u^2; "
                "on the failing step the reward is -failure_penalty instead"
            ),
            "upright_weight": self.upright_weight,
            "alive_bonus": self.alive_bonus,
            "control_cost_coef": self.control_cost_coef,
            "failure_penalty": self.failure_penalty,
            "alpha": "wrapped(pole_angle - reference_angle); 0 upright, +/-pi down",
            "u": "normalized motor command in [-3, 3]; actuator force = gear(100) * u",
            "note": "defined experiment-side; the lesson-7 environment file is unchanged",
        }


def normalize_observation(state, reference):
    """Hand-scaled observations; cos/sin encode the pole angle.

    A wrapped angle jumps from +pi to -pi exactly at the bottom - the region a
    swing-up policy must act in - so the pole direction enters as
    (cos(alpha), sin(alpha)), continuous through every full rotation.
    """
    state = np.asarray(state, dtype=float)
    alpha = float(wrap_angle(state[1] - reference[1]))
    return np.array(
        [
            state[0] / OBS_SCALE[0],
            np.cos(alpha),
            np.sin(alpha),
            state[2] / OBS_SCALE[1],
            state[3] / OBS_SCALE[2],
        ],
        dtype=float,
    )


class MLPTower:
    """Numpy MLP with ReLU hidden layers and a linear output; manual backward.

    backward() consumes an upstream gradient of a scalar loss, so the same
    tower serves the policy mean head and the value head.
    """

    def __init__(self, input_dim, hidden, output_dim, seed):
        rng = np.random.default_rng(seed)
        sizes = (input_dim, *hidden, output_dim)
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))
            for fan_in, fan_out in pairwise(sizes)
        ]
        self.biases = [np.zeros(fan_out) for fan_out in sizes[1:]]

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Expected a finite 2-D input batch")
        activations, preacts = [x], []
        layer = x
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            preact = layer @ weight + bias
            preacts.append(preact)
            layer = preact if index == last else np.maximum(preact, 0.0)
            activations.append(layer)
        return activations[-1], (activations, preacts)

    def backward(self, cache, grad_output):
        activations, preacts = cache
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        delta = np.asarray(grad_output, dtype=float)
        for index in reversed(range(len(self.weights))):
            grad_weights[index] = activations[index].T @ delta
            grad_biases[index] = delta.sum(axis=0)
            if index:
                delta = (delta @ self.weights[index].T) * (preacts[index - 1] > 0.0)
        return grad_weights, grad_biases


def gaussian_log_prob(mean, log_std, actions):
    """Diagonal-Gaussian log density for one-dimensional actions."""
    mean = np.asarray(mean, dtype=float)
    actions = np.asarray(actions, dtype=float)
    log_std = np.asarray(log_std, dtype=float)
    if mean.shape != actions.shape:
        raise ValueError("mean and actions must share the batch shape")
    return -0.5 * ((actions - mean) / np.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2.0 * np.pi)


def gaussian_entropy(log_std):
    """Differential entropy of the (single-dim) policy Gaussian."""
    return float(np.mean(np.asarray(log_std) + 0.5 * np.log(2.0 * np.pi * np.e)))


class GaussianPolicy:
    """2x64 mean head plus a state-independent learned log-std; actions in [-3, 3]."""

    def __init__(self, input_dim, hidden, seed, log_std_init=LOG_STD_INIT):
        self.trunk = MLPTower(input_dim, hidden, 1, seed)
        self.log_std = np.full(1, float(np.log(log_std_init)))

    def mean(self, obs):
        return self.trunk.forward(obs)[0][:, 0]

    def sample(self, obs, rng):
        mean = self.mean(obs)
        actions = mean + np.exp(self.log_std) * rng.standard_normal(mean.shape)
        return actions, gaussian_log_prob(mean, self.log_std, actions)

    def parameters(self):
        return [*self.trunk.weights, *self.trunk.biases, self.log_std]

    def arrays(self):
        payload = {"log_std": self.log_std.copy()}
        for index, (weight, bias) in enumerate(zip(self.trunk.weights, self.trunk.biases)):
            payload[f"weight_{index}"] = weight.copy()
            payload[f"bias_{index}"] = bias.copy()
        return payload


def policy_array_names(hidden):
    """Archive key names of one serialized policy (derived from the layer count)."""
    names = ["log_std"]
    for index in range(len(hidden) + 1):
        names.extend((f"weight_{index}", f"bias_{index}"))
    return tuple(names)


def standardize(values):
    """Per-batch advantage normalization; a degenerate batch maps to zeros."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot standardize an empty array")
    if not np.isfinite(values).all():
        raise ValueError("cannot standardize nonfinite values")
    std = float(values.std())
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / (std + 1e-12)


def clip_gradients_(gradients, max_norm):
    """In-place global-norm clipping; returns the pre-clip norm."""
    total = float(np.sqrt(sum(float(np.sum(g * g)) for g in gradients)))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for gradient in gradients:
            gradient *= scale
    return total


def ppo_losses_and_gradients(policy, value, obs, actions, old_logp, advantages, returns, config):
    """Clipped-surrogate + value + entropy loss with hand-written gradients.

    advantages must already be standardized by the caller; the clip branch is
    taken where ratio*advantage <= clipped*advantage (standard subgradient).
    """
    obs = np.asarray(obs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    old_logp = np.asarray(old_logp, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if not 0 < len(obs) == len(actions) == len(old_logp) == len(advantages) == len(returns):
        raise ValueError("Expected nonempty aligned 1-D batches")
    if not all(np.isfinite(a).all() for a in (obs, actions, old_logp, advantages, returns)):
        raise ValueError("Nonfinite PPO batch")

    batch = len(obs)
    mean_out, policy_cache = policy.trunk.forward(obs)
    mean = mean_out[:, 0]
    logp = gaussian_log_prob(mean, policy.log_std, actions)
    ratio = np.exp(logp - old_logp)
    clipped = np.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
    surrogate = np.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -float(np.mean(surrogate))

    value_out, value_cache = value.forward(obs)
    values = value_out[:, 0]
    value_loss = config.vf_coef * float(np.mean((values - returns) ** 2))

    entropy = gaussian_entropy(policy.log_std)
    total = policy_loss + value_loss
    if config.learn_std:
        total -= config.ent_coef * entropy  # with a fixed std this is a parameter-free constant

    unclipped = (ratio * advantages) <= (clipped * advantages)
    grad_logp = np.where(unclipped, -advantages * ratio / batch, 0.0)
    difference = actions - mean
    variance = float(np.exp(policy.log_std[0]) ** 2)
    grad_mean = (grad_logp * difference / variance).reshape(batch, 1)
    # With learn_std=False the exploration std is held at its initial value
    # (zero gradient): a learned std collapsed sigma and froze the policy at
    # the hang-still local optimum in every pilot of this lesson.
    grad_log_std = 0.0
    if config.learn_std:
        grad_log_std = float(np.sum(grad_logp * (difference**2 / variance - 1.0)))
        grad_log_std -= config.ent_coef
    policy_grad_w, policy_grad_b = policy.trunk.backward(policy_cache, grad_mean)

    grad_value = (2.0 * config.vf_coef * (values - returns) / batch).reshape(batch, 1)
    value_grad_w, value_grad_b = value.backward(value_cache, grad_value)

    gradients = [
        *policy_grad_w,
        *policy_grad_b,
        np.array([grad_log_std]),
        *value_grad_w,
        *value_grad_b,
    ]
    if not np.isfinite(total) or not all(np.isfinite(g).all() for g in gradients):
        raise ValueError("Nonfinite PPO loss or gradients; training diverged")
    losses = {
        "total": float(total),
        "policy": policy_loss,
        "value": value_loss,
        "entropy": entropy,
        "clip_fraction": float(np.mean(~unclipped)),
    }
    return losses, gradients


def compute_gae(
    rewards, values, terminated, truncated, terminal_values, gamma=GAMMA, gae_lambda=GAE_LAMBDA
):
    """GAE(lambda) with bootstrapping across TimeLimit truncations.

    terminal_values[t] holds V(the state that ended step t, before any reset):
    it is the bootstrap at a TimeLimit truncation and at the segment end, while
    a true termination contributes no bootstrap. The advantage recursion itself
    is cut at every episode boundary (terminated or truncated): steps after a
    reset belong to a new episode and must not leak into the previous one.
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)
    terminal_values = np.asarray(terminal_values, dtype=float)
    if rewards.ndim != 2 or rewards.shape != values.shape or rewards.shape != terminated.shape:
        raise ValueError("GAE expects aligned (T, N) arrays")
    if terminal_values.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("GAE expects aligned (T, N) arrays")
    horizon = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1])
    for step in reversed(range(horizon)):
        if step == horizon - 1:
            # At the segment end terminal_obs is the bootstrap state for every
            # env: the next observation when the episode continues, the pre-reset
            # state at a truncation; a termination is cut by `nonterminal`.
            next_value = terminal_values[step]
        else:
            next_value = np.where(truncated[step], terminal_values[step], values[step + 1])
        nonterminal = 1.0 - terminated[step].astype(float)
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[step] = running
        boundary = terminated[step] | truncated[step]
        running = np.where(boundary, 0.0, running)
    return advantages, advantages + values


class VecSwingup:
    """Parallel lesson-7 environments with the experiment-side reward.

    step() returns the pre-reset observation (needed to bootstrap a TimeLimit
    truncation) together with the post-reset observation the policy acts on
    next. Episodes start at the exact down position with a small recorded
    jitter so the stochastic policy can break the resting symmetry.
    """

    def __init__(
        self,
        reward,
        *,
        n_envs,
        episode_steps,
        base_seed,
        angle_jitter=INIT_ANGLE_JITTER,
        omega_jitter=INIT_OMEGA_JITTER,
        task_envs=None,
    ):
        if n_envs < 1 or episode_steps < 1:
            raise ValueError("n_envs and episode_steps must be positive")
        self.reward = reward
        self.n_envs = int(n_envs)
        self.angle_jitter = float(angle_jitter)
        self.omega_jitter = float(omega_jitter)
        # Two-bank curriculum: the first `task_envs` environments always start
        # from the exact resting down start (the task stays on-policy), the
        # rest use the random-start distribution. Without the task bank the
        # balancing skill dominates the gradient and the down-start reflex is
        # never corrected (see the lecture's failure analysis).
        self.task_envs = self.n_envs if task_envs is None else int(task_envs)
        if not 0 <= self.task_envs <= self.n_envs:
            raise ValueError("task_envs must be between 0 and n_envs")
        self.envs = [
            make_swingup_environment(max_episode_steps=episode_steps) for _ in range(self.n_envs)
        ]
        for index, env in enumerate(self.envs):
            env.reset(seed=int(base_seed) + index)
        self.rng = np.random.default_rng([int(base_seed), SEED_OFFSET_JITTER])

    def _start_state(self, index=None):
        """Episode start: exact resting down start for the task bank, else random."""
        state = self.reward.reference.copy()
        if index is not None and index < self.task_envs:
            state[1] += np.deg2rad(DOWN_ANGLE_DEG)
            return state
        state[1] += np.deg2rad(DOWN_ANGLE_DEG) + self.rng.uniform(
            -self.angle_jitter, self.angle_jitter
        )
        state[3] += self.rng.uniform(-self.omega_jitter, self.omega_jitter)
        return state

    def reset(self):
        observations = np.empty((self.n_envs, STATE_INPUTS))
        for index, env in enumerate(self.envs):
            state = self._start_state(index)
            env.unwrapped.set_state(state[:2], state[2:])
            env.unwrapped.data.qfrc_applied[0] = 0.0
            observations[index] = normalize_observation(state, self.reward.reference)
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
            rewards[index] = self.reward(state, float(command[0]), done)
            terminated[index], truncated[index] = done, timed_out
            if done or timed_out:
                state = self._start_state(index)
                env.unwrapped.set_state(state[:2], state[2:])
                env.unwrapped.data.qfrc_applied[0] = 0.0
            live_obs[index] = normalize_observation(state, reference)
        return terminal_obs, rewards, terminated, truncated, live_obs

    def close(self):
        for env in self.envs:
            env.close()


def collect_rollout(vec_env, policy, value, config, action_rng, observations):
    """One rollout segment with per-step bootstrap values for GAE."""
    steps, n_envs = config.rollout_steps, config.n_envs
    batch_obs = np.empty((steps, n_envs, STATE_INPUTS))
    batch_actions = np.empty((steps, n_envs))
    batch_logp = np.empty((steps, n_envs))
    batch_values = np.empty((steps, n_envs))
    batch_rewards = np.empty((steps, n_envs))
    batch_terminated = np.zeros((steps, n_envs), dtype=bool)
    batch_truncated = np.zeros((steps, n_envs), dtype=bool)
    batch_terminal_values = np.zeros((steps, n_envs))
    reward_sum, terminated_count = 0.0, 0
    for step in range(steps):
        actions, logp = policy.sample(observations, action_rng)
        batch_obs[step] = observations
        batch_actions[step] = actions
        batch_logp[step] = logp
        batch_values[step] = value.forward(observations)[0][:, 0]
        terminal_obs, rewards_raw, terminated, truncated, observations = vec_env.step(actions)
        batch_rewards[step] = rewards_raw
        batch_terminated[step] = terminated
        batch_truncated[step] = truncated
        batch_terminal_values[step] = value.forward(terminal_obs)[0][:, 0]
        reward_sum += float(rewards_raw.sum())
        terminated_count += int(terminated.sum())
    return {
        "obs": batch_obs.reshape((-1, STATE_INPUTS)),
        "actions": batch_actions.ravel(),
        "logp": batch_logp.ravel(),
        "values": batch_values.ravel(),
        "rewards": batch_rewards.ravel(),
        "terminated": batch_terminated.ravel(),
        "truncated": batch_truncated.ravel(),
        "terminal_values": batch_terminal_values.ravel(),
        "observations": observations,
        "reward_mean": reward_sum / (steps * n_envs),
        "terminated_frac": terminated_count / (steps * n_envs),
    }


def train_ppo(vec_env, *, config, init_seed, act_seed, shuffle_seed, eval_hook=None, log=None):
    """One PPO training run on a vec env; returns curves and final parameters.

    The trainer is reward-agnostic: the environment owns the reward. eval_hook,
    when given, receives (policy, env_steps) every eval_every updates (and at
    the last update) and its return value is appended to the evaluation curve.
    """
    policy = GaussianPolicy(STATE_INPUTS, config.hidden, init_seed, config.log_std_init)
    value = MLPTower(STATE_INPUTS, config.hidden, 1, [*init_seed, 1])
    parameters = [*policy.parameters(), *value.weights, *value.biases]
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    action_rng = np.random.default_rng(act_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    size = config.n_envs * config.rollout_steps

    reward_curve = np.empty(config.updates)
    terminated_curve = np.empty(config.updates)
    value_loss_curve = np.empty(config.updates)
    entropy_curve = np.empty(config.updates)
    clip_fraction_curve = np.empty(config.updates)
    eval_steps, eval_records = [], []
    observations = vec_env.reset()
    started = time.perf_counter()

    for update in range(config.updates):
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

        value_loss_sum, clip_fraction_sum, grad_steps = 0.0, 0.0, 0
        for _epoch in range(config.epochs):
            for start in range(0, size, config.minibatch):
                minibatch = order[start : start + config.minibatch]
                losses, gradients = ppo_losses_and_gradients(
                    policy,
                    value,
                    rollout["obs"][minibatch],
                    rollout["actions"][minibatch],
                    rollout["logp"][minibatch],
                    advantages[minibatch],
                    returns[minibatch],
                    config,
                )
                clip_gradients_(gradients, config.grad_clip)
                optimizer.step(parameters, gradients)
                value_loss_sum += losses["value"]
                clip_fraction_sum += losses["clip_fraction"]
                grad_steps += 1

        reward_curve[update] = rollout["reward_mean"]
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
                f"reward {rollout['reward_mean']:.2f}, term {rollout['terminated_frac']:.2f}"
            )
            if eval_records:
                message += f", eval {eval_records[-1]}"
            log(message)
    return {
        "policy": policy,
        "value": value,
        "reward_curve": reward_curve,
        "terminated_curve": terminated_curve,
        "value_loss_curve": value_loss_curve,
        "entropy_curve": entropy_curve,
        "clip_fraction_curve": clip_fraction_curve,
        "eval_steps": np.asarray(eval_steps, dtype=int),
        "eval_records": eval_records,
        "env_steps": int(config.updates * size),
        "wall_time_s": time.perf_counter() - started,
    }


def down_start_state(reference):
    """The lesson-7 down scenario start: reference angle rotated by -180 degrees."""
    state = np.asarray(reference, dtype=float).copy()
    state[1] += np.deg2rad(DOWN_ANGLE_DEG)
    return state


def run_policy_episode(
    policy, reward, reference, *, horizon, env_seed, deterministic, rng=None, schedule=None
):
    """One episode from the exact down start; arrays follow lesson-7 alignment.

    states[k] is the state before action k; controls and applied forces act on
    [k*dt, (k+1)*dt); the episode stops at the first physical failure.
    """
    if schedule is None:
        schedule = np.zeros(horizon)
    schedule = np.asarray(schedule, dtype=float)
    if schedule.shape != (horizon,):
        raise ValueError("push schedule must cover the horizon")
    env = make_swingup_environment(max_episode_steps=horizon)
    try:
        env.reset(seed=int(env_seed))
        env.unwrapped.data.qfrc_applied[0] = 0.0
        state = down_start_state(reference)
        env.unwrapped.set_state(state[:2], state[2:])
        states, controls, forces = [state.copy()], [], []
        terminated = truncated = False
        failure_reason = ""
        for force in schedule:
            obs = normalize_observation(state, reference)[None, :]
            if deterministic:
                action = float(policy.mean(obs)[0])
            else:
                action = float(policy.sample(obs, rng)[0][0])
            command = np.array([np.clip(action, -CONTROL_LIMIT, CONTROL_LIMIT)], np.float32)
            env.unwrapped.data.qfrc_applied[0] = float(force)
            state, _, terminated, truncated, info = env.step(command)
            states.append(state.copy())
            controls.append(float(command[0]))
            forces.append(float(force))
            failure_reason = info["failure_reason"]
            if terminated or truncated:
                break
        arrays = {
            "states": np.asarray(states, dtype=np.float32),
            "controls": np.asarray(controls, dtype=np.float32),
            "applied_force_n": np.asarray(forces, dtype=np.float32),
            "scheduled_force_n": schedule,
            "end_flags": np.array([terminated, truncated]),
        }
        return arrays, failure_reason
    finally:
        env.close()


def episode_rewards(arrays, reward):
    """Recompute per-step rewards from stored arrays (a pure function of them)."""
    controls, end_flags, states = arrays["controls"], arrays["end_flags"], arrays["states"]
    last = len(controls) - 1
    return np.asarray(
        [
            reward(states[step + 1], controls[step], bool(end_flags[0]) and step == last)
            for step in range(len(controls))
        ],
        dtype=float,
    )


def episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance applied to a policy episode (the same recovery_metrics)."""
    view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    return recovery_metrics(view, {"failure_reason": failure_reason}, reference, dt)


def select_metrics(metrics, fields=EVAL_FIELDS):
    return {field: metrics[field] for field in fields}


def evaluate_policy(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start (sampling noise only)."""
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
        metrics = episode_metrics(arrays, reason, reference, dt)
        episodes.append(
            {
                "eval_seed": eval_seed,
                **select_metrics(metrics),
                "return": float(episode_rewards(arrays, reward).sum()),
                "arrays": arrays,
            }
        )
    return episodes


def make_push_plans(dt, count, master_seed):
    """Lesson-5-style paired push plans: fixed magnitude, random sign and timing."""
    rng = np.random.default_rng([master_seed, SEED_OFFSET_PUSH])
    plans = []
    for index in range(count):
        start_s = round(rng.uniform(*PUSH_START_WINDOW_S) / dt) * dt
        sign = 1.0 if rng.random() < 0.5 else -1.0
        plans.append(
            {
                "index": index,
                "force_n": sign * PUSH_FORCE_N,
                "start_s": float(start_s),
                "duration_s": PUSH_DURATION_S,
            }
        )
    return plans


def push_schedule(plan, dt, horizon):
    start = round(plan["start_s"] / dt)
    duration = round(plan["duration_s"] / dt)
    schedule = np.zeros(horizon)
    schedule[start : start + duration] = plan["force_n"]
    return schedule


def baseline_evaluations(design, repeats):
    """Zero-shot lesson-7 controller on the down start, run `repeats` times."""
    records, trajectories, controls = [], [], None
    for repeat in range(repeats):
        arrays, metadata = run_scenario(
            Scenario("down", f"down repeat {repeat}"), design, EVAL_EPISODE_STEPS
        )
        metrics = recovery_metrics(arrays, metadata, design.controller.reference, design.dt)
        records.append({"repeat": repeat, **select_metrics(metrics)})
        trajectories.append(arrays["states"])
        controls = arrays["controls"]
    identical = all(np.array_equal(trajectories[0], states) for states in trajectories[1:])
    return records, trajectories[0], controls, bool(identical)


def baseline_push_evaluations(design, plans, horizon):
    """Zero-shot lesson-7 controller under the shared push plans.

    The push episodes use the same exact down start as every other check, so
    baseline and PPO differ in the controller alone (single-variable pairing).
    """
    records, states_list, controls_list = [], [], []
    for plan in plans:
        scenario = Scenario(
            "push", "push", DOWN_ANGLE_DEG, plan["force_n"], plan["start_s"], plan["duration_s"]
        )
        arrays, metadata = run_scenario(scenario, design, horizon)
        metrics = recovery_metrics(arrays, metadata, design.controller.reference, design.dt)
        records.append(
            {
                "plan_index": plan["index"],
                "force_n": plan["force_n"],
                "start_s": plan["start_s"],
                **select_metrics(metrics, PUSH_FIELDS),
            }
        )
        states_list.append(arrays["states"])
        controls_list.append(arrays["controls"])
    return records, states_list, controls_list


def failure_counts(episodes):
    counts = {
        "cart_safety_boundary": 0,
        "velocity_safety_boundary": 0,
        "timeout_without_settling": 0,
    }
    for episode in episodes:
        if episode["terminated"]:
            reason = episode["failure_reason"]
            counts[reason] = counts.get(reason, 0) + 1
        elif not episode["recovered"]:
            counts["timeout_without_settling"] += 1
    return counts


def summarize_episodes(episodes):
    successes = [e for e in episodes if e["recovered"] and not e["terminated"]]
    settled = [e["settled_at_s"] for e in successes if e["settled_at_s"] is not None]
    recovery = [
        e["recovery_after_push_end_s"]
        for e in successes
        if e.get("recovery_after_push_end_s") is not None
    ]
    forces = [e["peak_abs_motor_force_n"] for e in episodes]
    summary = {
        "episodes": len(episodes),
        "successes": len(successes),
        "median_settled_at_s": float(np.median(settled)) if settled else None,
        "median_peak_abs_motor_force_n": float(np.median(forces)),
        "failure_counts": failure_counts(episodes),
    }
    if any("recovery_after_push_end_s" in e for e in episodes):
        summary["median_recovery_after_push_end_s"] = (
            float(np.median(recovery)) if recovery else None
        )
        summary["recovery_times_s"] = [
            e["recovery_after_push_end_s"] if e["recovery_after_push_end_s"] is not None else None
            for e in episodes
        ]
    return summary


def stack_trajectories(states_list, horizon):
    """Pad variable-length episode states with NaN into (E, horizon+1, 4)."""
    cube = np.full((len(states_list), horizon + 1, 4), np.nan, dtype=np.float32)
    lengths = np.zeros(len(states_list), dtype=int)
    for index, states in enumerate(states_list):
        cube[index, : len(states)] = states
        lengths[index] = len(states)
    return cube, lengths


def stack_controls(controls_list, horizon):
    cube = np.full((len(controls_list), horizon), np.nan, dtype=np.float32)
    for index, controls in enumerate(controls_list):
        cube[index, : len(controls)] = controls
    return cube


def pick_failure_cases(eval_episodes, push_episodes):
    """First stochastic-eval failure, then first push failure (fixed scan order)."""
    cases = []
    for label, pool in (("eval_failure", eval_episodes), ("push_failure", push_episodes)):
        episode = next((e for e in pool if e["terminated"] or not e["recovered"]), None)
        if episode is not None:
            cases.append(
                {
                    "kind": label,
                    "eval_seed": episode.get("eval_seed"),
                    "plan_index": episode.get("plan_index"),
                    "force_n": episode.get("force_n"),
                    "start_s": episode.get("start_s"),
                    "failure_reason": episode["failure_reason"],
                    "terminated": bool(episode["terminated"]),
                    "recovered": bool(episode["recovered"]),
                    "settled_at_s": episode["settled_at_s"],
                    "max_abs_cart_position_m": episode["max_abs_cart_position_m"],
                    "peak_abs_motor_force_n": episode["peak_abs_motor_force_n"],
                    "arrays": episode["arrays"],
                }
            )
    return cases


def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    train_seeds=TRAIN_SEEDS,
    eval_seed_count=EVAL_SEEDS,
    log=print,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    config = config or PPOConfig()
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    reward = RewardFunction(reference)

    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    plans = make_push_plans(dt, eval_seed_count, seed)
    baseline_push_records, baseline_push_states, baseline_push_controls = baseline_push_evaluations(
        design, plans, EVAL_EPISODE_STEPS
    )

    output.mkdir(parents=True, exist_ok=False)
    training_records, policy_payloads, curves_list = [], [], []
    eval_states_cube, eval_lengths, eval_controls_cube = [], [], []
    eval_terminated, eval_settled, eval_returns = [], [], []
    eval_peak_force, eval_max_x = [], []
    det_records, det_states_list, det_controls_list = [], [], []
    push_states_cube, push_lengths, push_controls_cube = [], [], []
    push_terminated, push_settled, push_recovery = [], [], []
    all_eval_episodes, all_push_episodes = [], []

    for seed_index in range(train_seeds):
        vec_env = VecSwingup(
            reward,
            n_envs=config.n_envs,
            episode_steps=config.train_episode_steps,
            base_seed=10_000 + seed * 100 + seed_index,
            task_envs=config.task_envs,
        )

        def eval_hook(policy, _env_steps, _reward=reward, _reference=reference, _dt=dt):
            arrays, reason = run_policy_episode(
                policy,
                _reward,
                _reference,
                horizon=EVAL_EPISODE_STEPS,
                env_seed=0,
                deterministic=True,
            )
            metrics = episode_metrics(arrays, reason, _reference, _dt)
            return {
                "success": bool(metrics["recovered"] and not metrics["terminated"]),
                "settled_at_s": metrics["settled_at_s"],
                "return": float(episode_rewards(arrays, _reward).sum()),
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
        vec_env.close()
        policy = result["policy"]
        policy_payloads.append(policy.arrays())
        curves_list.append(
            {
                "reward_curve": result["reward_curve"],
                "terminated_curve": result["terminated_curve"],
                "value_loss_curve": result["value_loss_curve"],
                "entropy_curve": result["entropy_curve"],
                "clip_fraction_curve": result["clip_fraction_curve"],
            }
        )

        eval_episodes = evaluate_policy(
            policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
        )
        det_arrays, det_reason = run_policy_episode(
            policy, reward, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
        )
        det_metrics = episode_metrics(det_arrays, det_reason, reference, dt)
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
                    "arrays": arrays,
                }
            )

        all_eval_episodes.extend(eval_episodes)
        all_push_episodes.extend(push_episodes)
        final_window = slice(max(0, config.updates - 10), config.updates)
        training_records.append(
            {
                "seed_index": seed_index,
                "env_steps": result["env_steps"],
                "wall_time_s": result["wall_time_s"],
                "final_reward_mean": float(np.mean(result["reward_curve"][final_window])),
                "final_log_std": float(policy.log_std[0]),
                "first_successful_eval_steps": next(
                    (
                        int(step)
                        for step, record in zip(
                            result["eval_steps"], result["eval_records"], strict=True
                        )
                        if record["success"]
                    ),
                    None,
                ),
                "eval_curve": [
                    {"env_steps": int(step), **record}
                    for step, record in zip(
                        result["eval_steps"], result["eval_records"], strict=True
                    )
                ],
                "stochastic": summarize_episodes(eval_episodes),
                "deterministic": {
                    "recovered": bool(det_metrics["recovered"]),
                    "terminated": bool(det_metrics["terminated"]),
                    "settled_at_s": det_metrics["settled_at_s"],
                    "return": float(episode_rewards(det_arrays, reward).sum()),
                },
                "push": summarize_episodes(push_episodes),
            }
        )
        eval_states_cube.append([e["arrays"]["states"] for e in eval_episodes])
        eval_lengths.append([len(e["arrays"]["states"]) for e in eval_episodes])
        eval_controls_cube.append([e["arrays"]["controls"] for e in eval_episodes])
        eval_terminated.append([bool(e["terminated"]) for e in eval_episodes])
        eval_settled.append(
            [
                float(e["settled_at_s"]) if e["settled_at_s"] is not None else np.nan
                for e in eval_episodes
            ]
        )
        eval_returns.append([float(e["return"]) for e in eval_episodes])
        eval_peak_force.append([float(e["peak_abs_motor_force_n"]) for e in eval_episodes])
        eval_max_x.append([float(e["max_abs_cart_position_m"]) for e in eval_episodes])
        det_records.append(
            {
                key: det_metrics[key]
                for key in ("recovered", "terminated", "truncated", "settled_at_s")
            }
        )
        det_states_list.append(det_arrays["states"])
        det_controls_list.append(det_arrays["controls"])
        push_states_cube.append([e["arrays"]["states"] for e in push_episodes])
        push_lengths.append([len(e["arrays"]["states"]) for e in push_episodes])
        push_controls_cube.append([e["arrays"]["controls"] for e in push_episodes])
        push_terminated.append([bool(e["terminated"]) for e in push_episodes])
        push_settled.append(
            [
                float(e["settled_at_s"]) if e["settled_at_s"] is not None else np.nan
                for e in push_episodes
            ]
        )
        push_recovery.append(
            [
                float(e["recovery_after_push_end_s"])
                if e["recovery_after_push_end_s"] is not None
                else np.nan
                for e in push_episodes
            ]
        )
        if log is not None:
            record = training_records[-1]
            log(
                f"training seed {seed_index}: steps {record['env_steps']}, "
                f"wall {record['wall_time_s']:.1f}s, "
                f"stochastic {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                f"det settled {record['deterministic']['settled_at_s']}, "
                f"push {record['push']['successes']}/{record['push']['episodes']}"
            )

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
        training_records=training_records,
        all_eval_episodes=all_eval_episodes,
        all_push_episodes=all_push_episodes,
        failure_cases=failure_cases,
        elapsed=elapsed,
    )
    archive = build_archive(
        training_records=training_records,
        policy_payloads=policy_payloads,
        curves_list=curves_list,
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        baseline_push_states=baseline_push_states,
        baseline_push_controls=baseline_push_controls,
        eval_states_cube=eval_states_cube,
        eval_lengths=eval_lengths,
        eval_controls_cube=eval_controls_cube,
        eval_terminated=eval_terminated,
        eval_settled=eval_settled,
        eval_returns=eval_returns,
        eval_peak_force=eval_peak_force,
        eval_max_x=eval_max_x,
        det_states_list=det_states_list,
        det_controls_list=det_controls_list,
        push_states_cube=push_states_cube,
        push_lengths=push_lengths,
        push_controls_cube=push_controls_cube,
        push_terminated=push_terminated,
        push_settled=push_settled,
        push_recovery=push_recovery,
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
    save_evaluation(output / "evaluation.png", report, output)
    save_push_and_failures(output / "push_and_failures.png", report, output)
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
    training_records,
    all_eval_episodes,
    all_push_episodes,
    failure_cases,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    baseline_push_summary = summarize_episodes(baseline_push_records)
    ppo_per_seed = [
        {
            "seed_index": record["seed_index"],
            "env_steps": record["env_steps"],
            "wall_time_s": record["wall_time_s"],
            "final_reward_mean": record["final_reward_mean"],
            "final_log_std": record["final_log_std"],
            "first_successful_eval_steps": record["first_successful_eval_steps"],
            "eval_curve": record["eval_curve"],
            "stochastic": record["stochastic"],
            "deterministic": record["deterministic"],
            "push": record["push"],
        }
        for record in training_records
    ]
    stochastic_successes = [r["stochastic"]["successes"] for r in ppo_per_seed]
    push_successes = [r["push"]["successes"] for r in ppo_per_seed]
    settled_all = [
        e["settled_at_s"]
        for e in all_eval_episodes
        if e["recovered"] and not e["terminated"] and e["settled_at_s"] is not None
    ]
    report = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "master_seed": seed,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "protocol": {
            "task": "lesson-7 full-rotation swing-up, exact down start (reference angle -180 deg)",
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "train_episode_steps": config.train_episode_steps,
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * design.dt,
            "train_initial_state": {
                "pole_angle_rad_jitter": INIT_ANGLE_JITTER,
                "pole_omega_rad_s_jitter": INIT_OMEGA_JITTER,
                "cart_position_m": 0.0,
                "cart_velocity_m_s": 0.0,
                "note": (
                    "training starts randomize the pole direction over the full circle and "
                    "the initial angular velocity (random-start curriculum, recorded here); "
                    "every evaluation runs from the exact resting down start"
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
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": (
                    "hand scaling chosen before training; cos/sin keep the observation "
                    "continuous through the bottom seam (a wrapped angle jumps +/-pi there)"
                ),
            },
            "reward": reward.as_dict(),
            "reward_scale_for_learning": config.reward_scale,
            "reward_scale_note": (
                "learning uses reward * reward_scale (only rescales the value targets); reported "
                "rewards and returns are unscaled"
            ),
            "seed_streams": {
                "network_init": (
                    "default_rng([master, 7000 + train_seed]); value net [master, 7000 + seed, 1]"
                ),
                "action_sampling": "default_rng([master, 5000 + train_seed])",
                "minibatch_order": "default_rng([master, 9000 + train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000])",
            },
        },
        "hyperparameters": asdict(config),
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{eval_seed_count} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
            "per_episode": baseline_records,
        },
        "ppo_evaluation": {
            "protocol": (
                f"{eval_seed_count} stochastic episodes per training seed (sampling noise only; "
                "the initial state is the exact down start), plus one mean-action episode per seed"
            ),
            "per_training_seed": ppo_per_seed,
            "aggregate": {
                "episodes": len(all_eval_episodes),
                "successes": int(sum(stochastic_successes)),
                "success_rate": float(np.mean(stochastic_successes) / eval_seed_count),
                "successes_per_seed": stochastic_successes,
                "median_settled_at_s": float(np.median(settled_all)) if settled_all else None,
            },
        },
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; PPO never saw pushes during training; both "
                    "controllers run the same plans from the same exact down start; PPO "
                    "episodes use the mean action (the baseline is deterministic)"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": "the same plans are applied to the baseline and to every PPO seed",
            },
            "plans": plans,
            "baseline": baseline_push_summary,
            "ppo_per_seed": [{"seed_index": r["seed_index"], **r["push"]} for r in ppo_per_seed],
            "ppo_aggregate": {
                "episodes": len(all_push_episodes),
                "successes": int(sum(push_successes)),
                "successes_per_seed": push_successes,
            },
        },
        "failure_analysis": {
            "ppo_eval_counts": failure_counts(all_eval_episodes),
            "ppo_push_counts": failure_counts(all_push_episodes),
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in failure_cases
            ],
        },
        "training": {
            "train_seeds": train_seeds,
            "env_steps_per_seed": training_records[0]["env_steps"] if training_records else 0,
            "total_env_steps": sum(r["env_steps"] for r in training_records),
            "wall_time_s_total": elapsed,
            "curves_note": "reward_curve_* and other per-update curves live in trajectories.npz",
        },
        "limitations": [
            "Pure-numpy 2x64 PPO at one fixed configuration; the pilot history (learned vs fixed std, four gamma/lambda pairs, three penalty levels, three curricula) is narrated in the lecture, not swept here - this is one existence probe, not a tuned benchmark.",
            "The reward, cos/sin observation, curriculum and network sizes are hand choices made before the runs - in RL these are part of the method (protocol fields).",
            "PPO never saw pushes, sensor noise or model mismatch; the push comparison quantifies the gap between training and this disturbance, not a general robustness claim.",
            "The baseline is deterministic, so its repeats coincide; the repeated protocol is a same-caliber formality, not sampling evidence.",
            "Success requires the strict lesson-7 settled tail (0.02 m, 0.01 rad, 0.02 m/s, 0.02 rad/s held >= 2 s); a policy that balances slightly worse counts as failed.",
            "Training is confined to 20 s model time per episode on 8 parallel environments; no claim is made beyond this budget.",
        ],
    }
    return report


def build_archive(
    *,
    training_records,
    policy_payloads,
    curves_list,
    baseline_states,
    baseline_controls,
    baseline_push_states,
    baseline_push_controls,
    eval_states_cube,
    eval_lengths,
    eval_controls_cube,
    eval_terminated,
    eval_settled,
    eval_returns,
    eval_peak_force,
    eval_max_x,
    det_states_list,
    det_controls_list,
    push_states_cube,
    push_lengths,
    push_controls_cube,
    push_terminated,
    push_settled,
    push_recovery,
    failure_cases,
):
    horizon = EVAL_EPISODE_STEPS
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=np.float64),
        "baseline_controls": np.asarray(baseline_controls, dtype=np.float64),
        "baseline_push_states": stack_trajectories(baseline_push_states, horizon)[0],
        "baseline_push_lengths": stack_trajectories(baseline_push_states, horizon)[1],
        "baseline_push_controls": stack_controls(baseline_push_controls, horizon),
        "eval_states": stack_trajectories(
            [s for seed_list in eval_states_cube for s in seed_list], horizon
        )[0],
        "eval_lengths": np.asarray([n for seed_list in eval_lengths for n in seed_list]),
        "eval_controls": stack_controls(
            [c for seed_list in eval_controls_cube for c in seed_list], horizon
        ),
        "eval_terminated": np.asarray(eval_terminated, dtype=bool),
        "eval_settled_s": np.asarray(eval_settled, dtype=float),
        "eval_returns": np.asarray(eval_returns, dtype=float),
        "eval_peak_force_n": np.asarray(eval_peak_force, dtype=float),
        "eval_max_x_m": np.asarray(eval_max_x, dtype=float),
        "eval_det_states": stack_trajectories(det_states_list, horizon)[0],
        "eval_det_controls": stack_controls(det_controls_list, horizon),
        "push_states": stack_trajectories(
            [s for seed_list in push_states_cube for s in seed_list], horizon
        )[0],
        "push_lengths": np.asarray([n for seed_list in push_lengths for n in seed_list]),
        "push_controls": stack_controls(
            [c for seed_list in push_controls_cube for c in seed_list], horizon
        ),
        "push_terminated": np.asarray(push_terminated, dtype=bool),
        "push_settled_s": np.asarray(push_settled, dtype=float),
        "push_recovery_s": np.asarray(push_recovery, dtype=float),
    }
    for seed_index, curves in enumerate(curves_list):
        for name, curve in curves.items():
            archive[f"{name}_{seed_index}"] = curve
    for index, case in enumerate(failure_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    for seed_index, payload in enumerate(policy_payloads):
        for name, array in payload.items():
            archive[f"policy{seed_index}_{name}"] = array
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    curves = (
        "reward_curve",
        "terminated_curve",
        "value_loss_curve",
        "entropy_curve",
        "clip_fraction_curve",
    )
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
        "eval_states",
        "eval_lengths",
        "eval_controls",
        "eval_terminated",
        "eval_settled_s",
        "eval_returns",
        "eval_peak_force_n",
        "eval_max_x_m",
        "eval_det_states",
        "eval_det_controls",
        "push_states",
        "push_lengths",
        "push_controls",
        "push_terminated",
        "push_settled_s",
        "push_recovery_s",
    }
    keys.update(f"{curve}_{seed}" for seed in range(seeds) for curve in curves)
    keys.update(
        f"policy{seed}_{name}"
        for seed in range(seeds)
        for name in policy_array_names(report["hyperparameters"]["hidden"])
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
        reward = np.asarray([data[f"reward_curve_{index}"] for index in range(seeds)], dtype=float)
    per_seed = report["ppo_evaluation"]["per_training_seed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    updates = np.arange(1, reward.shape[1] + 1)
    for index in range(seeds):
        axes[0].plot(updates, reward[index], alpha=0.35, linewidth=0.9, color="#64748b")
    axes[0].plot(
        updates,
        reward.mean(axis=0),
        color="#0f766e",
        linewidth=1.8,
        label=f"{seeds} 个训练种子均值",
    )
    axes[0].set(
        xlabel="PPO 更新轮次",
        ylabel="批内平均原始奖励（每步）",
        title="训练奖励：细线 = 单个训练种子",
    )
    axes[0].legend(fontsize=8)
    eval_steps = np.asarray([p["env_steps"] for p in per_seed[0]["eval_curve"]]) / 1000.0
    eval_success = np.asarray(
        [[p["success"] for p in record["eval_curve"]] for record in per_seed], dtype=float
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
        title="训练中周期评估：均值动作、单回合",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_evaluation(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    gear = report["protocol"]["actuator_gear"]
    ref_theta = report["protocol"]["reference_state"][1]
    baseline_summary = report["baseline"]
    aggregate = report["ppo_evaluation"]["aggregate"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        baseline_states = data["baseline_states"]
        baseline_controls = data["baseline_controls"]
        det_states = data["eval_det_states"]
        det_controls = data["eval_det_controls"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    ts = np.arange(len(baseline_states)) * dt
    edges = np.arange(len(baseline_controls) + 1) * dt
    det_ts = np.arange(det_states.shape[1]) * dt
    axes[0, 0].plot(
        ts, np.cos(baseline_states[:, 1] - ref_theta), "--", color="gray", label="基线（能量+LQR）"
    )
    axes[0, 0].plot(
        det_ts, np.cos(det_states[0, :, 1] - ref_theta), color="#0f766e", label="PPO（均值动作）"
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(ylabel="杆端相对高度", ylim=(-1.1, 1.1), title="同一下方初态：摆起轨迹")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(ts, baseline_states[:, 0], "--", color="gray")
    axes[0, 1].plot(det_ts, det_states[0, :, 0], color="#2563eb")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].set(ylabel="小车位置（m）", title="小车位置（红点线 = ±2.4 m 失败边界）")
    axes[1, 0].stairs(baseline_controls * gear, edges, color="gray", label="基线")
    axes[1, 0].stairs(det_controls[0] * gear, edges, color="#2563eb", label="PPO")
    axes[1, 0].set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入")
    axes[1, 0].legend(fontsize=8)
    labels = [
        f"基线\n{baseline_summary['episodes']} 次重复",
        f"PPO\n{aggregate['episodes']} 个随机回合",
    ]
    successes = [baseline_summary["successes"], aggregate["successes"]]
    totals = [baseline_summary["episodes"], aggregate["episodes"]]
    bars = axes[1, 1].bar(
        labels,
        [success / total * 100 for success, total in zip(successes, totals)],
        color=["#64748b", "#0f766e"],
        width=0.55,
    )
    for bar, success, total in zip(bars, successes, totals):
        axes[1, 1].annotate(
            f"{success}/{total}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1, 1].set(ylabel="验收通过率（%）", ylim=(0, 112), title="同口径成功率（第 7 课验收）")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_push_and_failures(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    plans = report["push_test"]["plans"]
    plan_count = len(plans)
    baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
    per_seed = report["push_test"]["ppo_per_seed"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        baseline_states = data["baseline_push_states"]
        ppo_states = data["push_states"]
        lengths = data["push_lengths"]
        cases = report["failure_analysis"]["featured_cases"]
        case_arrays = [
            (data[f"case{index}_states"], data[f"case{index}_controls"])
            for index in range(len(cases))
        ]
    paired = next(
        (
            index
            for index in range(plan_count)
            if baseline_recovery[index] is not None
            and per_seed[0]["recovery_times_s"][index] is not None
        ),
        0,
    )
    plan = plans[paired]
    gear = report["protocol"]["actuator_gear"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    axes[0, 0].axvspan(
        plan["start_s"], plan["start_s"] + plan["duration_s"], alpha=0.2, color="#fbbf24"
    )
    axes[0, 0].plot(
        np.arange(len(baseline_states[paired])) * dt,
        np.cos(baseline_states[paired, :, 1] - ref_theta),
        "--",
        color="gray",
        label="基线（能量+LQR）",
    )
    axes[0, 0].plot(
        np.arange(lengths[paired]) * dt,
        np.cos(ppo_states[paired, : lengths[paired], 1] - ref_theta),
        color="#0f766e",
        label="PPO（训练种子 0）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"配对推力案例：{plan['force_n']:+.0f} N @ {plan['start_s']:.2f} s（黄色带）",
    )
    axes[0, 0].legend(fontsize=8)
    plan_indices = np.arange(plan_count)
    axes[0, 1].plot(
        plan_indices,
        [value if value is not None else np.nan for value in baseline_recovery],
        "s",
        color="#64748b",
        label="基线",
    )
    for seed_index, record in enumerate(per_seed):
        axes[0, 1].plot(
            plan_indices,
            [value if value is not None else np.nan for value in record["recovery_times_s"]],
            "o",
            markersize=4,
            alpha=0.55,
            label=f"PPO 种子 {seed_index}",
        )
    axes[0, 1].set(
        xlabel="推力方案编号",
        ylabel="推力结束后恢复时间（s）",
        title="推力后恢复时间（缺 = 未恢复）",
    )
    axes[0, 1].legend(fontsize=7)
    if case_arrays:
        states, _controls = case_arrays[0]
        case = cases[0]
        axes[1, 0].plot(
            np.arange(len(states)) * dt,
            np.cos(states[:, 1] - ref_theta),
            color="#b91c1c",
        )
        axes[1, 0].axhspan(-1, 0, alpha=0.08, color="orange")
        axes[1, 0].set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"失败案例：{case['kind']}（{case['failure_reason'] or '未达标'}）",
        )
        if len(case_arrays) > 1:
            edges = np.arange(len(case_arrays[1][1]) + 1) * dt
            axes[1, 1].stairs(case_arrays[1][1] * gear, edges, color="#b91c1c")
        else:
            edges = np.arange(len(case_arrays[0][1]) + 1) * dt
            axes[1, 1].stairs(case_arrays[0][1] * gear, edges, color="#b91c1c")
        axes[1, 1].set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="失败回合电机输入（±300 N 限幅）",
        )
    else:
        for ax, title in (
            (axes[1, 0], "失败案例：无（全部回合通过验收）"),
            (axes[1, 1], "失败回合电机输入：无"),
        ):
            ax.text(
                0.5,
                0.5,
                "本记录没有 PPO 失败回合",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set(title=title)
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
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--minibatch", type=int, default=MINIBATCH)
    parser.add_argument("--train-episode-steps", type=int, default=TRAIN_EPISODE_STEPS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--task-envs", type=int, default=TASK_ENVS)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(HIDDEN))
    args = parser.parse_args()
    config = PPOConfig(
        updates=args.updates,
        n_envs=args.n_envs,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch=args.minibatch,
        train_episode_steps=args.train_episode_steps,
        eval_every=args.eval_every,
        task_envs=args.task_envs,
        hidden=tuple(args.hidden),
    )
    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            config=config,
            train_seeds=args.train_seeds,
            eval_seed_count=args.eval_seeds,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["successes"],
                "ppo": report["ppo_evaluation"]["aggregate"],
                "push": report["push_test"]["ppo_aggregate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
