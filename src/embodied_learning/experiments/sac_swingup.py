"""Lesson 35: hand-written numpy Soft Actor-Critic on the down-start swing-up.

Lessons 29-34 tried six rungs of the "let it learn" ladder (plain PPO, residual
RL, PBRS shaping, DAPG demonstrations, Go-Explore, two-phase rewards) and all
six ended 0/60 on the strict lesson-7 acceptance, with the upright arrival
improving (1/60 PBRS, 2/60 two-phase) while the "stay on top for 2 s" tail
never appeared. This lesson runs the literature's mainline answer - SAC
(Haarnoja et al. 2018, ICML, arXiv:1801.01290; automatic temperature from
Haarnoja et al. 2019, arXiv:1812.05905) - as the single off-policy +
maximum-entropy alternative, hand-written in numpy, on the SAME environment,
reward, observations, evaluation caliber and budget as lesson 29:

    r = (1 + cos(alpha))/2 + 0.25 - 0.01*u^2   (lesson-29 task reward, verbatim)
    failing step: r = -10                      (lesson-29 caliber, verbatim)

Nothing is shaped, no base controller, no teacher, no demonstration. Two
single-variable rows run 3 seeds each (500k env steps per seed):
  * alpha auto-tuned (target entropy = -action_dim = -1, the SAC standard),
  * alpha fixed at 0.2 (the manual-caliber contrast; everything else identical).

Method decisions recorded in the protocol (single-variable spirit):
  * gamma = 0.99, lr = 3e-4, tau = 0.005: the SAC standard values. Lesson 29's
    gamma = 0.995 was a PPO tuning choice; the algorithm runs on ITS OWN
    standard hyperparameters (recorded, not swept).
  * no reward scaling (reward_scale = 1.0; PPO used 0.1 for its value targets);
  * no random-start curriculum: lesson 29 needed it because its noise probe
    never lifted the pole past horizontal from the resting down start, and it
    was recorded there as hidden hand-design. This lesson removes it entirely
    so the algorithm is the only variable - every training episode restarts
    from the EXACT RESTING DOWN START, the same start every evaluation uses;
  * one gradient update per vector step (8 parallel environments = every 8
    environment steps; the nearest whole-step reading of the textbook
    "1 update per env step" on the vectorized runner; buffer 100k, batch 500).

Components, all hand-written numpy: a 5x64x64 tanh-squashed Gaussian policy
(mu = 3*tanh(mu_raw); log_std tanh-clamped in [-20, 2]; action = 3*tanh(u)
with the log|da/du| correction in the log-prob), twin Q networks (5x64x64->1)
with target copies softly updated at tau, a ring replay buffer (capacity 100k),
and the sample-based automatic temperature. No advantage normalization - SAC
does not use advantages; that is part of the point.

Process metrics: upright first arrival (|alpha| <= 0.3 rad, the lesson-7
capture cone), first successful periodic evaluation, the replay-buffer
coverage entropy at every evaluation checkpoint (is exploration still alive
late in training - the "entropy collapse" hypothesis is measured directly),
the alpha trajectory and the policy entropy. Evaluations are the same 20
stochastic episodes per seed plus the mean-action episode; the same paired
+/-200 N push plans as lessons 29-34.

Honesty rule: if SAC also fails within the budget, the failure (alpha collapse
/ buffer entropy collapse / value underfit) is the formal result, decomposed
to components. Nothing is smoothed over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.dapg_swingup import LESSON29_PPO_REFERENCE
from embodied_learning.experiments.pbrs_swingup import (
    CAPTURE_ANGLE_RAD,
    arrival_summary,
    first_arrival_time_s,
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
    RewardFunction,
    VecSwingup,
    baseline_evaluations,
    baseline_push_evaluations,
    episode_rewards,
    failure_counts,
    make_push_plans,
    pick_failure_cases,
    push_schedule,
    run_policy_episode,
    select_metrics,
    stack_controls,
    stack_trajectories,
    summarize_episodes,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import MODEL_PATH, SAFE_CART_POSITION, design_swingup_lqr

EXPERIMENT = "sac_swingup_lesson35"
SCHEMA_VERSION = 1

ACTION_DIM = 1
HIDDEN = (64, 64)
ACTION_BOUND = CONTROL_LIMIT  # normalized command limit [+/-3] == [+/-300 N]
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
LOG_PROB_EPS = 1e-6

N_ENVS = 8
TRAIN_STEPS = 500_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 500
GAMMA = 0.99
LEARNING_RATE = 3e-4
TAU = 0.005
ALPHA_INIT = 0.2
TARGET_ENTROPY = -float(ACTION_DIM)  # = -1 (the SAC standard for one action)
ALPHA_LR = 3e-4
ALPHA_MIN = 1e-8
TRAIN_EPISODE_STEPS = 250
EVAL_EVERY_STEPS = 25_000
UPDATE_EVERY_ENV_STEPS = 32  # one update per four vector steps (32 env steps)
ENV_BASE_SEED = 10_000  # the lesson-29 env stream: base + master*100 + seed

ALPHA_TIERS = ("auto", "fixed")

EVAL_FIELDS = (
    "recovered",
    "terminated",
    "truncated",
    "settled_at_s",
    "peak_abs_motor_force_n",
    "max_abs_cart_position_m",
    "failure_reason",
)
DOWN_ANGLE_DEG = -180.0

# Reference rows imported verbatim from the official lesson 29-34 records
# (docs/33-39); used only by the comparison blocks in this record.
LESSON30_RESIDUAL_REFERENCE = {
    "source": "results/residual_swingup_2026-09-06 (official lesson-30 record, docs/35)",
    "episodes": 180,
    "successes": 0,
    "median_settled_at_s": None,
    "first_upright_arrival": (
        "guaranteed by the lesson-7 base controller (three amplitude tiers a = 25/50/100 N)"
    ),
    "note": "three amplitude tiers, all 0/60 (60 episodes each)",
}
LESSON31_PBRS_REFERENCE = {
    "source": "results/pbrs_swingup_2026-09-06 (official lesson-31 record, docs/36)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "first_upright_arrival": ("1 evaluation touch (cE=2.0 seed 1 @ the 150k checkpoint, 1.24 s)"),
}
LESSON32_DAPG_REFERENCE = {
    "source": "results/dapg_swingup_2026-09-06 (official lesson-32 record, docs/37)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "upright_arrival_episodes": "33/60 (w=10 tier; the demonstration bank guarantees the arrival)",
}
LESSON33_GOEXPLORE_REFERENCE = {
    "source": "results/goexplore_swingup_2026-09-06 (official lesson-33 record, docs/38)",
    "episodes": 20,
    "successes": 0,
    "median_settled_at_s": None,
    "upright_arrival_episodes": "11/20 (Go-Explore+BC, archive resets)",
}
LESSON34_TWOPHASE_REFERENCE = {
    "source": "results/twophase_swingup_2026-09-06 (official lesson-34 record, docs/39)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "median_peak_abs_motor_force_n": 300.0,
    "upright_first_arrival": (
        "2/60 evaluation episodes (seed 1 @ 100k and seed 2 @ 200k checkpoints; median 1.22 s)"
    ),
    "first_success": "never within the budget",
}


@dataclass(frozen=True)
class SACConfig:
    n_envs: int = N_ENVS
    train_steps: int = TRAIN_STEPS
    buffer_size: int = BUFFER_SIZE
    batch_size: int = BATCH_SIZE
    gamma: float = GAMMA
    lr: float = LEARNING_RATE
    tau: float = TAU
    alpha_init: float = ALPHA_INIT
    alpha_lr: float = ALPHA_LR
    alpha_min: float = ALPHA_MIN
    target_entropy: float = TARGET_ENTROPY
    auto_alpha: bool = True
    hidden: tuple[int, ...] = HIDDEN
    train_episode_steps: int = TRAIN_EPISODE_STEPS
    eval_every_steps: int = EVAL_EVERY_STEPS
    update_every_env_steps: int = UPDATE_EVERY_ENV_STEPS
    warmup_steps: int = BATCH_SIZE  # wait for the first batch before any update

    def __post_init__(self):
        positive = (
            self.n_envs,
            self.train_steps,
            self.buffer_size,
            self.batch_size,
            self.gamma,
            self.lr,
            self.tau,
            self.train_episode_steps,
            self.update_every_env_steps,
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("SAC hyperparameters must be finite and positive")
        if not isinstance(self.auto_alpha, bool):
            raise TypeError("auto_alpha must be a bool")
        if self.alpha_init <= 0.0 or not np.isfinite(self.alpha_init):
            raise ValueError("alpha_init must be finite and positive")
        if self.target_entropy >= 0.0:
            raise ValueError("target_entropy must be negative (the SAC convention)")
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size cannot exceed buffer_size")
        if not 0 <= self.warmup_steps <= self.buffer_size:
            raise ValueError("warmup_steps must be in [0, buffer_size]")
        if self.eval_every_steps > self.train_steps:
            raise ValueError("eval_every_steps cannot exceed train_steps")
        if any(units < 1 for units in self.hidden):
            raise ValueError("hidden layer sizes must be positive")


def default_training_config():
    """The standard SAC configuration, budget and all (one run per seed/tier)."""
    return SACConfig()


# ------------------------------------------------------------------ networks
class MLP:
    """Numpy MLP with ReLU hidden layers and a linear output; manual backward.

    backward() consumes the upstream gradient of a scalar loss and optionally
    returns the gradient w.r.t. the input layer as well (the twin Q-s of SAC
    need dQ/da to flow back into the policy action).
    """

    def __init__(self, input_dim, hidden, output_dim, seed):
        rng = np.random.default_rng(seed)
        sizes = (int(input_dim), *tuple(int(units) for units in hidden), int(output_dim))
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

    def backward(self, cache, grad_output, need_input_grad=False):
        activations, preacts = cache
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        delta = np.asarray(grad_output, dtype=float)
        for index in reversed(range(len(self.weights))):
            grad_weights[index] = activations[index].T @ delta
            grad_biases[index] = delta.sum(axis=0)
            if index:
                delta = (delta @ self.weights[index].T) * (preacts[index - 1] > 0.0)
        grad_input = delta @ self.weights[0].T if need_input_grad else None
        return grad_weights, grad_biases, grad_input

    def parameters(self):
        return [*self.weights, *self.biases]

    def arrays(self, prefix=""):
        payload = {}
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            payload[f"{prefix}weight_{index}"] = weight.copy()
            payload[f"{prefix}bias_{index}"] = bias.copy()
        return payload


def clone_mlp(source):
    """Deep copy of an MLP (the target-network initial copies)."""
    clone = MLP(1, (), 1, seed=0)
    clone.weights = [np.array(weight, dtype=float) for weight in source.weights]
    clone.biases = [np.array(bias, dtype=float) for bias in source.biases]
    return clone


def soft_update_mlp(target, source, tau):
    """theta_target <- tau * theta_source + (1 - tau) * theta_target, in place."""
    for target_weight, source_weight in zip(target.weights, source.weights, strict=True):
        target_weight *= 1.0 - tau
        target_weight += tau * source_weight
    for target_bias, source_bias in zip(target.biases, source.biases, strict=True):
        target_bias *= 1.0 - tau
        target_bias += tau * source_bias


def squashed_gaussian_log_prob(u, mean, log_std, action_bound, eps=LOG_PROB_EPS):
    """log p(a|s) for a = action_bound * tanh(u), u ~ N(mean, exp(log_std)).

    Includes the log|da/du| = log(action_bound) + log(1 - tanh^2 u) Jacobian
    correction (the SAC tanh-squashed Gaussian density).
    """
    u = np.asarray(u, dtype=float)
    mean = np.asarray(mean, dtype=float)
    log_std = np.asarray(log_std, dtype=float)
    normal = -0.5 * ((u - mean) / np.exp(log_std)) ** 2 - log_std - 0.5 * np.log(2.0 * np.pi)
    correction = -np.log(1.0 - np.tanh(u) ** 2 + eps) - np.log(float(action_bound))
    return normal + correction


def tanh_squash_params(mean_raw, log_std_raw, action_bound, log_std_min, log_std_max):
    """mu and log_std from the raw trunk outputs (both tanh-mapped)."""
    mean = float(action_bound) * np.tanh(mean_raw)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (np.tanh(log_std_raw) + 1.0)
    return mean, log_std


class SquashedGaussianPolicy:
    """5x64x64 trunk -> [mu_raw, log_std_raw]; a = action_bound * tanh(u).

    Interface mirrors the lesson-29 GaussianPolicy (mean / sample) so the
    shared episode runner, evaluation and record pipeline are reused verbatim;
    the deterministic action is the squashed mean.
    """

    def __init__(
        self,
        input_dim,
        hidden,
        seed,
        action_bound=ACTION_BOUND,
        log_std_min=LOG_STD_MIN,
        log_std_max=LOG_STD_MAX,
    ):
        self.trunk = MLP(input_dim, hidden, ACTION_DIM * 2, seed)
        self.action_bound = float(action_bound)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def heads(self, obs):
        out, cache = self.trunk.forward(obs)
        return out[:, :ACTION_DIM], out[:, ACTION_DIM:], cache

    def mean(self, obs):
        """Deterministic (squashed-mean) actions, shape (B,)."""
        mean_raw, _, _ = self.heads(obs)
        return (self.action_bound * np.tanh(mean_raw))[:, 0]

    def sample(self, obs, rng):
        mean_raw, log_std_raw, _ = self.heads(obs)
        mean, log_std = tanh_squash_params(
            mean_raw, log_std_raw, self.action_bound, self.log_std_min, self.log_std_max
        )
        u = mean + np.exp(log_std) * rng.standard_normal(mean.shape)
        action = self.action_bound * np.tanh(u)
        log_prob = squashed_gaussian_log_prob(u, mean, log_std, self.action_bound)
        # lesson-29 interface: actions / log-probs are 1-D (actions[0] is a scalar)
        return action[:, 0], log_prob[:, 0]

    def parameters(self):
        return self.trunk.parameters()

    def arrays(self):
        return self.trunk.arrays()


def policy_array_names(hidden):
    """Archive key names of one serialized network (derived from the layer count)."""
    names = []
    for index in range(len(tuple(hidden)) + 1):
        names.extend((f"weight_{index}", f"bias_{index}"))
    return tuple(names)


# ------------------------------------------------------------------ replay
class ReplayBuffer:
    """Fixed-capacity ring buffer of (obs, action, reward, next_obs, terminated).

    next_obs is the pre-reset observation of the transition (the state the
    episode actually reached); `terminated` marks a physical failure - the only
    case where the TD target must not bootstrap (time-limit truncations DO
    bootstrap, the same convention as lesson 29's GAE). Sampling is uniform
    over the stored entries without replacement within one batch.
    """

    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.observations = np.empty((self.capacity, int(state_dim)), dtype=float)
        self.actions = np.empty((self.capacity, int(action_dim)), dtype=float)
        self.rewards = np.empty(self.capacity, dtype=float)
        self.next_obs = np.empty((self.capacity, int(state_dim)), dtype=float)
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.size = 0
        self.cursor = 0

    def push(self, observations, actions, rewards, next_obs, terminated):
        observations = np.asarray(observations, dtype=float)
        actions = np.asarray(actions, dtype=float).reshape(-1)
        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=float)
        terminated = np.asarray(terminated, dtype=bool).reshape(-1)
        if len(observations) != len(actions) or len(observations) != len(rewards):
            raise ValueError("one transition per element")
        for offset in range(len(observations)):
            index = self.cursor % self.capacity
            self.observations[index] = observations[offset]
            self.actions[index] = actions[offset]
            self.rewards[index] = rewards[offset]
            self.next_obs[index] = next_obs[offset]
            self.terminated[index] = bool(terminated[offset])
            self.cursor += 1
        self.size = min(self.capacity, self.size + len(observations))

    def sample(self, batch_size, rng):
        if batch_size > self.size:
            raise ValueError("not enough transitions stored yet")
        indices = rng.choice(self.size, size=int(batch_size), replace=False)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "terminated": self.terminated[indices],
        }

    def stored_observations(self):
        """All valid entries (positions do not matter for the histogram)."""
        return self.observations[: self.size] if self.size <= self.capacity else self.observations


def coverage_entropy(observations, n_angle=12, n_x=4, n_omega=4):
    """Shannon entropy (nats) + coverage of the buffer over a coarse state grid.

    Bins: wrapped pole angle (12 x 30 deg), cart position (4 bins over the
    +/-2.5 m rail), pole angular velocity (4 bins over +/-20 rad/s) = 192
    cells. An exploration collapse shows up as entropy -> 0 / coverage small.
    """
    observations = np.asarray(observations, dtype=float)
    n_angle, n_x, n_omega = int(n_angle), int(n_x), int(n_omega)
    angle = np.arctan2(observations[:, 2], observations[:, 1])
    x = observations[:, 0] * 2.5
    omega = observations[:, 4] * 10.0
    b_angle = np.mod(np.floor((angle + np.pi) / (2.0 * np.pi / n_angle)), n_angle).astype(int)
    b_x = np.clip(np.floor((x + 2.5) / 5.0 * n_x), 0, n_x - 1).astype(int)
    b_omega = np.clip(np.floor((omega + 20.0) / 40.0 * n_omega), 0, n_omega - 1).astype(int)
    flat = b_angle * (n_x * n_omega) + b_x * n_omega + b_omega
    counts = np.bincount(flat, minlength=n_angle * n_x * n_omega).astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0, 0.0
    p = counts / total
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    coverage = float((counts > 0).mean())
    return entropy, coverage


# ------------------------------------------------------------- SAC updates
def td_targets(rewards, terminated, next_obs, target_q1, target_q2, policy, rng, alpha, gamma):
    """y = r + gamma * (1 - term) * (min(Qbar1, Qbar2) - alpha * log pi(a'|s')).

    a' is re-sampled from the current policy for every next state; the minimum
    of the two target Q-values is the clipped double-Q estimate.
    """
    rewards = np.asarray(rewards, dtype=float)
    terminated = np.asarray(terminated, dtype=bool)
    next_actions, next_log_prob = policy.sample(next_obs, rng)
    state_action = np.concatenate([next_obs, next_actions.reshape(-1, 1)], axis=1)
    q1 = target_q1.forward(state_action)[0][:, 0]
    q2 = target_q2.forward(state_action)[0][:, 0]
    soft_q = np.minimum(q1, q2) - alpha * next_log_prob
    nonterminal = 1.0 - terminated.astype(float)
    return rewards + gamma * nonterminal * soft_q


def critic_loss_and_gradients(q_network, observations, actions, targets):
    """Mean squared error toward the fixed target; hand gradients returned."""
    state_action = np.concatenate(
        [np.asarray(observations, dtype=float), np.asarray(actions, dtype=float).reshape(-1, 1)],
        axis=1,
    )
    targets = np.asarray(targets, dtype=float)
    output, cache = q_network.forward(state_action)
    deltas = output[:, 0] - targets
    loss = float(np.mean(deltas**2))
    delta = (2.0 * deltas / float(len(deltas))).reshape(-1, 1)
    grad_weights, grad_biases, _ = q_network.backward(cache, delta)
    return loss, [*grad_weights, *grad_biases]


def critic_values_and_input_grads(q1, q2, observations, actions):
    """min(Q1, Q2) at (s, a) plus the input (action) gradients of the argmin net.

    One backward pass per network (the weight gradients are computed on the way
    but discarded - the input gradient is what the policy needs).
    """
    state_action = np.concatenate(
        [np.asarray(observations, dtype=float), np.asarray(actions, dtype=float).reshape(-1, 1)],
        axis=1,
    )
    out1, cache1 = q1.forward(state_action)
    out2, cache2 = q2.forward(state_action)
    value1 = out1[:, 0]
    value2 = out2[:, 0]
    use_one = value1 <= value2
    q_value = np.minimum(value1, value2)
    ones = np.ones_like(q_value).reshape(-1, 1)
    grad_input1 = q1.backward(cache1, ones, need_input_grad=True)[2]
    grad_input2 = q2.backward(cache2, ones, need_input_grad=True)[2]
    grad_input = np.where(use_one.reshape(-1, 1), grad_input1, grad_input2)
    return q_value, grad_input[:, -1]


def sac_policy_loss_and_gradients(policy, observations, q_value, q_grad, rng, alpha):
    """L = mean(alpha * log pi - min(Q)(s, a)), a reparameterized through tanh.

    Analytic gradients w.r.t. the raw trunk outputs [mu_raw, log_std_raw]
    (finite-difference checked in the tests). With u = mu + sigma*eps,
    a = B*tanh(u), log pi = log N(u; mu, sigma) - log B - log(1 - tanh^2 u + e):
        dL/dmu_raw     = mu' * (alpha * D_u - dQ/da * B(1 - tanh^2 u))
        dL/dlog_std_raw = h_l * (alpha * (D_u * sigma * eps - 1)
                                 - dQ/da * B(1 - tanh^2 u) * sigma * eps)
    with mu' = B(1 - tanh^2 mu_raw), h_l = 0.5*(Lmax-Lmin)*(1 - tanh^2 log_std_raw),
    D_u = 2 tanh u (1 - tanh^2 u)/(1 - tanh^2 u + e). Q is frozen (its gradient
    is NOT backpropagated into the policy - the reparameterization trick).
    """
    observations = np.asarray(observations, dtype=float)
    q_value = np.asarray(q_value, dtype=float).reshape(-1)
    q_grad = np.asarray(q_grad, dtype=float).reshape(-1, 1)
    mean_raw, log_std_raw, cache = policy.heads(observations)
    mean, log_std = tanh_squash_params(
        mean_raw, log_std_raw, policy.action_bound, policy.log_std_min, policy.log_std_max
    )
    noise = rng.standard_normal(mean.shape)
    u = mean + np.exp(log_std) * noise
    action = policy.action_bound * np.tanh(u)
    log_prob = squashed_gaussian_log_prob(u, mean, log_std, policy.action_bound)
    loss = float(np.mean(alpha * log_prob - q_value))
    tanh_u = np.tanh(u)
    t2 = tanh_u**2
    inv = 1.0 / (1.0 - t2 + LOG_PROB_EPS)
    d_u = 2.0 * tanh_u * (1.0 - t2) * inv
    mu_prime = policy.action_bound * (1.0 - np.tanh(mean_raw) ** 2)
    h_l = 0.5 * (policy.log_std_max - policy.log_std_min) * (1.0 - np.tanh(log_std_raw) ** 2)
    scale = policy.action_bound * (1.0 - t2)
    sigma = np.exp(log_std)
    grad_mu = mu_prime * (alpha * d_u - q_grad * scale) / float(len(observations))
    grad_logstd = (
        h_l
        * (alpha * (d_u * sigma * noise - 1.0) - q_grad * scale * sigma * noise)
        / float(len(observations))
    )
    grad_weights, grad_biases, _ = policy.trunk.backward(
        cache, np.concatenate([grad_mu, grad_logstd], axis=1)
    )
    return {
        "loss": loss,
        "log_prob": log_prob[:, 0].copy(),
        "log_std_mean": float(np.mean(log_std)),
        "entropy": -float(np.mean(log_prob)),
        "action_mean": float(np.mean(action)),
        "action_abs_mean": float(np.mean(np.abs(action))),
    }, [*grad_weights, *grad_biases]


def alpha_step(alpha, log_prob, target_entropy, lr, alpha_min=ALPHA_MIN):
    """One automatic-temperature step: alpha += lr * mean(log pi + H_target).

    mean(log pi) is the MC entropy estimate (-entropy), so when the entropy sits
    below the target the multiplier grows and vice versa.
    """
    log_prob = np.asarray(log_prob, dtype=float)
    step = float(np.mean(log_prob) + float(target_entropy))
    return float(max(alpha_min, float(alpha) + float(lr) * step))


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
    alpha_auto,
    rng,
    config,
):
    """One SAC update: twin-Q critic, then alpha, then policy, then soft targets.

    The critic forward is done once per Q network and the cache is reused for
    both the MSE weight gradients and the dQ/da input gradients (the one
    forward + two back per network arrangement, measured at ~4 ms/update).
    """
    observations = batch["observations"]
    actions = batch["actions"]
    state_action = np.concatenate([observations, actions.reshape(-1, 1)], axis=1)
    q1_out, q1_cache = q1.forward(state_action)
    q2_out, q2_cache = q2.forward(state_action)
    q_targets = td_targets(
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
    batch_size = float(len(actions))
    delta1 = (2.0 * (q1_out[:, 0] - q_targets) / batch_size).reshape(-1, 1)
    delta2 = (2.0 * (q2_out[:, 0] - q_targets) / batch_size).reshape(-1, 1)
    grad_weights1, grad_biases1, _ = q1.backward(q1_cache, delta1)
    grad_weights2, grad_biases2, _ = q2.backward(q2_cache, delta2)
    q1_loss = float(np.mean((q1_out[:, 0] - q_targets) ** 2))
    q2_loss = float(np.mean((q2_out[:, 0] - q_targets) ** 2))
    q1_optimizer.step(q1.parameters(), [*grad_weights1, *grad_biases1])
    q2_optimizer.step(q2.parameters(), [*grad_weights2, *grad_biases2])

    # the policy gradient uses the UPDATED critics: fresh forward per network
    q_value, q_grad = critic_values_and_input_grads(q1, q2, observations, actions)
    metrics, policy_grads = sac_policy_loss_and_gradients(
        policy, observations, q_value, q_grad, rng, alpha
    )
    if alpha_auto:
        alpha = alpha_step(alpha, metrics["log_prob"], config.target_entropy, config.alpha_lr)
    policy_optimizer.step(policy.parameters(), policy_grads)
    soft_update_mlp(target_q1, q1, config.tau)
    soft_update_mlp(target_q2, q2, config.tau)
    return {
        "q1_loss": q1_loss,
        "q2_loss": q2_loss,
        "policy_loss": metrics["loss"],
        "policy_entropy": metrics["entropy"],
        "log_std_mean": metrics["log_std_mean"],
        "action_mean": metrics["action_mean"],
        "action_abs_mean": metrics["action_abs_mean"],
        "batch_reward": float(np.mean(batch["rewards"])),
        "batch_terminated_frac": float(np.mean(batch["terminated"])),
        "alpha": alpha,
    }


# ------------------------------------------------------------------ runner
def episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance reused verbatim (the same recovery_metrics)."""
    view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl", dtype="<U2")}
    return recovery_metrics(view, {"failure_reason": failure_reason}, reference, dt)


def evaluate_sac_policy(
    policy, reward, reference, dt, *, master_seed, tier_index, count=EVAL_SEEDS
):
    """`count` stochastic episodes from the exact down start + arrival metric."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, tier_index, SEED_OFFSET_EVAL, eval_seed])
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
                "first_arrival_s": first_arrival_time_s(
                    arrays["states"], reference, dt, capture_angle=CAPTURE_ANGLE_RAD
                ),
                "arrays": arrays,
            }
        )
    return episodes


def deterministic_sac_episode(policy, reward, reference, dt):
    """One mean-action episode (the squashed mean is the deterministic policy)."""
    arrays, reason = run_policy_episode(
        policy, reward, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
    )
    metrics = episode_metrics(arrays, reason, reference, dt)
    return {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "failure_reason": reason,
        "return": float(episode_rewards(arrays, reward).sum()),
        "first_arrival_s": first_arrival_time_s(
            arrays["states"], reference, dt, capture_angle=CAPTURE_ANGLE_RAD
        ),
    }, arrays


def train_sac(vec_env, config, policy_seed, action_seed, buffer_seed, eval_hook=None, log=None):
    """One SAC training run (one tier, one seed); returns curves and networks.

    Every environment step enters the replay buffer; one gradient update runs
    per `update_every_env_steps` environment steps. The periodic eval_hook
    receives the policy at every `eval_every_steps` checkpoint; the buffer
    coverage entropy is measured at the same checkpoints.
    """
    policy = SquashedGaussianPolicy(STATE_INPUTS, config.hidden, policy_seed)
    q1 = MLP(STATE_INPUTS + ACTION_DIM, config.hidden, 1, [*policy_seed, 1])
    q2 = MLP(STATE_INPUTS + ACTION_DIM, config.hidden, 1, [*policy_seed, 2])
    target_q1, target_q2 = clone_mlp(q1), clone_mlp(q2)
    policy_optimizer = AdamOptimizer(policy.parameters(), lr=config.lr)
    q1_optimizer = AdamOptimizer(q1.parameters(), lr=config.lr)
    q2_optimizer = AdamOptimizer(q2.parameters(), lr=config.lr)
    action_rng = np.random.default_rng(action_seed)
    buffer_rng = np.random.default_rng(buffer_seed)
    replay = ReplayBuffer(config.buffer_size, STATE_INPUTS, ACTION_DIM)
    alpha = float(config.alpha_init)

    total_vector_steps = config.train_steps // config.n_envs
    vector_steps_per_update = max(1, math.ceil(config.update_every_env_steps / config.n_envs))
    updates = max(1, total_vector_steps // vector_steps_per_update)
    reward_curve = np.empty(updates)
    critic_loss_curve = np.empty(updates)
    alpha_curve = np.empty(updates)
    entropy_curve = np.empty(updates)
    log_std_curve = np.empty(updates)
    interim_reward_sum, interim_reward_count = 0.0, 0
    eval_steps, eval_records = [], []
    cover_entropy_curve, cover_fraction_curve = [], []

    observations = vec_env.reset()
    update_index = 0
    env_steps = 0
    started = time.perf_counter()

    for vector_step in range(total_vector_steps):
        current_obs = observations
        actions, _ = policy.sample(current_obs, action_rng)
        terminal_obs, rewards, terminated, _truncated, observations = vec_env.step(actions)
        # the buffer entry is (pre-step obs, action, reward, pre-reset terminal
        # obs); storing the reassigned `observations` would shift the state
        # column by one step and corrupt the time-limit/reset boundaries
        replay.push(current_obs, actions, rewards, terminal_obs, terminated)
        env_steps += config.n_envs
        interim_reward_sum += float(rewards.sum())
        interim_reward_count += config.n_envs

        if (
            replay.size >= config.warmup_steps
            and (env_steps % config.update_every_env_steps == 0)
            and vector_step % vector_steps_per_update == vector_steps_per_update - 1
        ):
            batch = replay.sample(config.batch_size, buffer_rng)
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
                config.auto_alpha,
                buffer_rng,
                config,
            )
            alpha = metrics["alpha"]
            reward_curve[update_index] = interim_reward_sum / max(1, interim_reward_count)
            interim_reward_sum, interim_reward_count = 0.0, 0
            critic_loss_curve[update_index] = 0.5 * (metrics["q1_loss"] + metrics["q2_loss"])
            alpha_curve[update_index] = alpha
            entropy_curve[update_index] = metrics["policy_entropy"]
            log_std_curve[update_index] = metrics["log_std_mean"]
            update_index += 1

        if env_steps % config.eval_every_steps == 0 or vector_step == total_vector_steps - 1:
            eval_steps.append(env_steps)
            eval_records.append(eval_hook(policy, env_steps))
            cover_entropy, cover_fraction = coverage_entropy(replay.stored_observations())
            cover_entropy_curve.append(cover_entropy)
            cover_fraction_curve.append(cover_fraction)
        if log is not None and update_index > 0 and update_index % 2500 == 0:
            log(
                f"  steps {env_steps}, updates {update_index}, "
                f"reward {reward_curve[update_index - 1]:.3f}, alpha {alpha:.3f}"
            )

    curves = {
        "reward_curve": reward_curve[:update_index].copy(),
        "critic_loss_curve": critic_loss_curve[:update_index].copy(),
        "alpha_curve": alpha_curve[:update_index].copy(),
        "entropy_curve": entropy_curve[:update_index].copy(),
        "log_std_curve": log_std_curve[:update_index].copy(),
    }
    eval_curve_records = [
        {"env_steps": int(step), **record}
        for step, record in zip(eval_steps, eval_records, strict=True)
    ]
    return {
        "policy": policy,
        "q1": q1,
        "q2": q2,
        "curves": curves,
        "eval_curve": eval_curve_records,
        "eval_steps": np.asarray(eval_steps, dtype=int),
        "eval_records": eval_records,
        "cover_entropy_curve": np.asarray(cover_entropy_curve, dtype=float),
        "cover_fraction_curve": np.asarray(cover_fraction_curve, dtype=float),
        "env_steps": int(env_steps),
        "wall_time_s": time.perf_counter() - started,
        "final_alpha": float(alpha),
        "final_policy_entropy": float(curves["entropy_curve"][-1]) if update_index else None,
        "final_cover_entropy": float(cover_entropy_curve[-1]) if cover_entropy_curve else None,
        "final_cover_fraction": float(cover_fraction_curve[-1]) if cover_fraction_curve else None,
        "buffer_size": replay.size,
    }


def first_checkpoint_steps(result, predicate):
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
    tiers=ALPHA_TIERS,
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
    if isinstance(tiers, (str, bytes)):
        tiers = (str(tiers),)
    tiers = tuple(tiers)
    if not tiers or any(tier not in ALPHA_TIERS for tier in tiers):
        raise ValueError("tiers must be a non-empty subset of ('auto', 'fixed')")
    config = config or default_training_config()

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

    tier_records = {}
    for tier_index, tier in enumerate(tiers):
        tier_config = SACConfig(**{**asdict(config), "auto_alpha": tier == "auto"})
        per_seed_records = []
        train_curves = {}
        cover_curves = {}
        policy_payloads = {}
        q_payloads = {}
        det_payloads = {}
        eval_stores, push_stores = [], []
        all_eval_episodes, all_push_episodes = [], []

        for seed_index in range(train_seeds):
            vec_env = VecSwingup(
                reward,
                n_envs=tier_config.n_envs,
                episode_steps=tier_config.train_episode_steps,
                base_seed=ENV_BASE_SEED + seed * 100 + seed_index,
                task_envs=tier_config.n_envs,  # every env restarts at the exact down start
            )

            def eval_hook(policy, _env_steps, _reward=reward, _reference=reference, _dt=dt):
                record, _arrays = deterministic_sac_episode(policy, _reward, _reference, _dt)
                return {
                    "success": bool(record["recovered"] and not record["terminated"]),
                    "settled_at_s": record["settled_at_s"],
                    "return": record["return"],
                    "first_arrival_s": record["first_arrival_s"],
                }

            result = train_sac(
                vec_env,
                config=tier_config,
                policy_seed=[seed, tier_index, SEED_OFFSET_INIT + seed_index],
                action_seed=[seed, tier_index, SEED_OFFSET_ACT + seed_index],
                buffer_seed=[seed, tier_index, SEED_OFFSET_SHUFFLE + seed_index],
                eval_hook=eval_hook,
                log=log,
            )
            vec_env.close()
            policy = result["policy"]
            train_curves[seed_index] = result["curves"]
            cover_curves[seed_index] = {
                "cover_entropy": result["cover_entropy_curve"],
                "cover_fraction": result["cover_fraction_curve"],
            }
            policy_payloads[seed_index] = policy.arrays()
            q_payloads[seed_index] = {
                **{f"q1_{name}": value for name, value in result["q1"].arrays().items()},
                **{f"q2_{name}": value for name, value in result["q2"].arrays().items()},
            }
            det_record, det_arrays = deterministic_sac_episode(policy, reward, reference, dt)
            det_payloads[seed_index] = {
                "states": det_arrays["states"],
                "controls": det_arrays["controls"],
            }

            eval_episodes = evaluate_sac_policy(
                policy,
                reward,
                reference,
                dt,
                master_seed=seed,
                tier_index=tier_index,
                count=eval_seed_count,
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
                metrics = episode_metrics(arrays, reason, reference, dt)
                push_episodes.append(
                    {
                        "plan_index": plan["index"],
                        "force_n": plan["force_n"],
                        "start_s": plan["start_s"],
                        **select_metrics(metrics, PUSH_FIELDS),
                        "return": float(episode_rewards(arrays, reward).sum()),
                        "first_arrival_s": first_arrival_time_s(
                            arrays["states"], reference, dt, capture_angle=CAPTURE_ANGLE_RAD
                        ),
                        "arrays": arrays,
                    }
                )

            all_eval_episodes.extend(eval_episodes)
            all_push_episodes.extend(push_episodes)
            final_window = slice(max(0, len(result["curves"]["reward_curve"]) - 10), None)
            final_reward = float(np.mean(result["curves"]["reward_curve"][final_window]))
            arrival_count = sum(
                1 for episode in eval_episodes if episode["first_arrival_s"] is not None
            )
            per_seed_records.append(
                {
                    "seed_index": seed_index,
                    "env_steps": result["env_steps"],
                    "wall_time_s": result["wall_time_s"],
                    "final_reward_mean": final_reward,
                    "final_alpha": result["final_alpha"],
                    "final_policy_entropy": result["final_policy_entropy"],
                    "final_cover_entropy": result["final_cover_entropy"],
                    "final_cover_fraction": result["final_cover_fraction"],
                    "first_successful_eval_steps": first_success_eval_steps(result),
                    "first_arrival_eval_steps": first_arrival_eval_steps(result),
                    "arrival_count": arrival_count,
                    "eval_curve": result["eval_curve"],
                    "stochastic": summarize_episodes(eval_episodes),
                    "deterministic": {
                        key: det_record[key]
                        for key in ("recovered", "terminated", "settled_at_s", "failure_reason")
                    },
                    "deterministic_return": det_record["return"],
                    "deterministic_first_arrival_s": det_record["first_arrival_s"],
                    "push": summarize_episodes(push_episodes),
                }
            )
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
                    "peaks": [
                        float(episode["peak_abs_motor_force_n"]) for episode in eval_episodes
                    ],
                    "max_x": [
                        float(episode["max_abs_cart_position_m"]) for episode in eval_episodes
                    ],
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
                    f"tier {tier} seed {seed_index}: steps {per_seed_records[-1]['env_steps']}, "
                    f"wall {per_seed_records[-1]['wall_time_s']:.0f}s, "
                    f"stoch {per_seed_records[-1]['stochastic']['successes']}/"
                    f"{per_seed_records[-1]['stochastic']['episodes']}, "
                    f"arrival {arrival_count}/{len(eval_episodes)}, "
                    f"first success {per_seed_records[-1]['first_successful_eval_steps']}, "
                    f"final alpha {per_seed_records[-1]['final_alpha']:.4f}, "
                    f"cover entropy {per_seed_records[-1]['final_cover_entropy']:.2f}"
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
        featured_cases_full = pick_failure_cases(all_eval_episodes, all_push_episodes)
        for case in featured_cases_full:
            case["tier"] = tier
        tier_records[tier] = {
            "automatic_temperature": tier == "auto",
            "alpha_init": tier_config.alpha_init,
            "per_seed": per_seed_records,
            "aggregate": aggregate,
            "arrival": arrival_summary(all_eval_episodes),
            "failure_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in all_eval_episodes]
            ),
            "push_failure_counts": failure_counts(
                [{k: v for k, v in e.items() if k != "arrays"} for e in all_push_episodes]
            ),
            "featured_seed_index": choose_featured_seed(per_seed_records),
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in featured_cases_full
            ],
            "featured_cases_arrays": featured_cases_full,
            "train_curves": train_curves,
            "cover_curves": cover_curves,
            "policy_payloads": policy_payloads,
            "q_payloads": q_payloads,
            "det_payloads": det_payloads,
            "eval_stores": eval_stores,
            "push_stores": push_stores,
            "first_success_any": bool(
                any(
                    record["first_successful_eval_steps"] is not None for record in per_seed_records
                )
            ),
        }

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        reward=reward,
        tiers=tiers,
        train_seeds=train_seeds,
        eval_seed_count=eval_seed_count,
        plans=plans,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        baseline_push_records=baseline_push_records,
        tier_records=tier_records,
        elapsed=elapsed,
    )
    archive = build_archive(
        tier_records=tier_records,
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        baseline_push_states=baseline_push_states,
        baseline_push_controls=baseline_push_controls,
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
    save_featured_cases(output / "featured_cases.png", report, output)
    return report


def arrival_phrase(arrival):
    if arrival["episodes_with_arrival"] == 0:
        return "never in any evaluation episode"
    return (
        f"{arrival['episodes_with_arrival']}/{arrival['episodes']} episodes, "
        f"median {arrival['median_first_arrival_s']:.2f} s"
    )


def first_success_steps(tier_record):
    values = [
        record["first_successful_eval_steps"]
        for record in tier_record["per_seed"]
        if record["first_successful_eval_steps"] is not None
    ]
    return f"{min(values)} steps" if values else "never"


def build_report(
    *,
    seed,
    config,
    design,
    reward,
    tiers,
    train_seeds,
    eval_seed_count,
    plans,
    baseline_records,
    baseline_identical,
    baseline_push_records,
    tier_records,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    baseline_push_summary = summarize_episodes(baseline_push_records)
    five_way = [
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
            "first_success": "never (docs/33)",
            "source": LESSON29_PPO_REFERENCE["source"],
        },
        {
            "label": "两阶段奖励（第 34 课，锁存切换）",
            "episodes": LESSON34_TWOPHASE_REFERENCE["episodes"],
            "successes": LESSON34_TWOPHASE_REFERENCE["successes"],
            "median_settled_at_s": LESSON34_TWOPHASE_REFERENCE["median_settled_at_s"],
            "upright_first_arrival": LESSON34_TWOPHASE_REFERENCE["upright_first_arrival"],
            "first_success": LESSON34_TWOPHASE_REFERENCE["first_success"],
            "source": LESSON34_TWOPHASE_REFERENCE["source"],
        },
    ]
    for tier in tiers:
        five_way.append(
            {
                "label": (
                    "SAC α 自动（第 35 课，本记录）"
                    if tier == "auto"
                    else "SAC α=0.2 固定（第 35 课，本记录）"
                ),
                "episodes": tier_records[tier]["aggregate"]["episodes"],
                "successes": tier_records[tier]["aggregate"]["successes"],
                "median_settled_at_s": tier_records[tier]["aggregate"]["median_settled_at_s"],
                "upright_first_arrival": arrival_phrase(tier_records[tier]["arrival"]),
                "first_success": (
                    f"first periodic-eval success at {first_success_steps(tier_records[tier])}"
                    if tier_records[tier]["first_success_any"]
                    else "never within the budget"
                ),
                "source": "this record",
            }
        )
    per_tier_brief = {}
    for tier in tiers:
        records = tier_records[tier]["per_seed"]
        per_tier_brief[tier] = {
            "final_reward_mean_per_seed": [record["final_reward_mean"] for record in records],
            "final_alpha_per_seed": [record["final_alpha"] for record in records],
            "final_cover_entropy_per_seed": [record["final_cover_entropy"] for record in records],
            "final_cover_fraction_per_seed": [record["final_cover_fraction"] for record in records],
            "final_policy_entropy_per_seed": [record["final_policy_entropy"] for record in records],
        }
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
                "SAC (off-policy maximum entropy): replay buffer + twin Q + soft targets + "
                "automatic temperature; hand-written numpy; no base controller, no teacher, "
                "no shaping term - the lesson-29 task reward verbatim"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "reward": reward.as_dict(),
            "no_shaping_note": (
                "the reward is the lesson-29 RewardFunction verbatim (upright + alive - u^2 "
                "with the -10 failing step); nothing is shaped"
            ),
            "reward_scale_for_learning": 1.0,
            "reward_scale_note": (
                "SAC uses the raw reward (no value-target scaling; PPO's 0.1 was a PPO choice)"
            ),
            "train_episode_steps": config.train_episode_steps,
            "train_initial_state": {
                "note": (
                    "every training episode restarts from the exact resting down start (the "
                    "lesson-29 random-start curriculum is REMOVED - it was a recorded hand "
                    "choice there; this lesson makes the algorithm the only variable)"
                ),
            },
            "eval_initial_state": "exact resting down start (reference angle -180 deg), no jitter",
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
                "definition": f"|alpha| <= {CAPTURE_ANGLE_RAD:g} rad (the lesson-7 capture angle)",
                "first_arrival": (
                    "first episode step whose stored state satisfies the definition; lessons 29/34 "
                    "arrived never / 2 evaluation episodes respectively"
                ),
            },
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lesson 29",
            },
            "method_decisions": {
                "gamma": config.gamma,
                "gamma_rationale": (
                    "0.99 is the SAC standard; lesson 29 used 0.995 as a PPO tuning choice. "
                    "Single-variable spirit: the algorithm runs on ITS OWN standard "
                    "hyperparameters - sweeps are left to failure attribution."
                ),
                "lr": config.lr,
                "tau": config.tau,
                "target_entropy": config.target_entropy,
                "target_entropy_note": "-action_dim = -1 (the SAC standard)",
                "alpha_init": config.alpha_init,
                "alpha_tiers_definition": (
                    "'auto' tunes alpha toward the target entropy; 'fixed' holds alpha at the "
                    "same init value 0.2 - the single-variable manual contrast"
                ),
                "no_advantage_normalization": True,
                "why_no_advantage_normalization": (
                    "SAC's policy gradient is a reparameterized value gradient - it does not use "
                    "advantages; normalizing Q-values would change the objective for no benefit"
                ),
                "update_schedule": {
                    "n_envs": config.n_envs,
                    "update_every_env_steps": config.update_every_env_steps,
                    "note": (
                        "one gradient update per four vector steps = every 32 environment "
                        "steps (recorded in the protocol: the wall-clock budget of the two "
                        "tiers x 3 seeds on this CPU sets the cadence - fewer gradient steps "
                        "per sample than the textbook 1-per-step SAC; the replay buffer keeps "
                        "the data-efficiency advantage regardless)"
                    ),
                },
                "no_curriculum": True,
                "no_curriculum_note": (
                    "lesson 29 documented the random-start curriculum as hidden hand-design; "
                    "removed here, so the outcome attributes to the algorithm alone"
                ),
            },
            "seed_streams": {
                "network_init": (
                    "default_rng([master, tier_index, 7000 + train_seed]); Q1 [.., 1]; Q2 [.., 2]"
                ),
                "action_sampling": "default_rng([master, tier_index, 5000 + train_seed])",
                "buffer_sampling": "default_rng([master, tier_index, 9000 + train_seed])",
                "env_stream": (
                    "default_rng([10_000 + master*100 + train_seed, 6000]) - the lesson-29 env "
                    "seed stream; all environments restart at the exact down start (no jitter)"
                ),
                "eval_actions": "default_rng([master, tier_index, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lessons 29-34)",
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
        "push_test": {
            "protocol": {
                "style": (
                    "lesson-5 random pushes; SAC never saw pushes during training; the same plans "
                    "run from the same exact down start; mean-action episodes"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": "the same plans are applied to the baseline and to every tier/seed",
            },
            "plans": plans,
            "baseline": baseline_push_summary,
        },
        "sac_evaluation": {
            "protocol": (
                f"{eval_seed_count} stochastic episodes per training seed per tier (sampling "
                "noise only; the initial state is the exact down start), plus one mean-action "
                "episode per seed"
            ),
            "tiers": {
                tier: {
                    "automatic_temperature": tier_records[tier]["automatic_temperature"],
                    "alpha_init": tier_records[tier]["alpha_init"],
                    "per_seed": [
                        {
                            key: record[key]
                            for key in (
                                "seed_index",
                                "env_steps",
                                "wall_time_s",
                                "final_reward_mean",
                                "final_alpha",
                                "final_policy_entropy",
                                "final_cover_entropy",
                                "final_cover_fraction",
                                "first_successful_eval_steps",
                                "first_arrival_eval_steps",
                                "arrival_count",
                                "eval_curve",
                                "stochastic",
                                "deterministic",
                                "deterministic_return",
                                "deterministic_first_arrival_s",
                                "push",
                            )
                        }
                        for record in tier_records[tier]["per_seed"]
                    ],
                    "aggregate": tier_records[tier]["aggregate"],
                    "arrival": tier_records[tier]["arrival"],
                    "featured_seed_index": tier_records[tier]["featured_seed_index"],
                    "failure_counts": tier_records[tier]["failure_counts"],
                    "push_aggregate": {
                        "episodes": eval_seed_count * train_seeds,
                        "successes": int(
                            sum(r["push"]["successes"] for r in tier_records[tier]["per_seed"])
                        ),
                        "successes_per_seed": [
                            r["push"]["successes"] for r in tier_records[tier]["per_seed"]
                        ],
                    },
                }
                for tier in tiers
            },
        },
        "five_way_comparison": five_way,
        "references": {
            "lesson29_ppo": LESSON29_PPO_REFERENCE,
            "lesson30_residual": LESSON30_RESIDUAL_REFERENCE,
            "lesson31_pbrs": LESSON31_PBRS_REFERENCE,
            "lesson32_dapg": LESSON32_DAPG_REFERENCE,
            "lesson33_goexplore": LESSON33_GOEXPLORE_REFERENCE,
            "lesson34_twophase": LESSON34_TWOPHASE_REFERENCE,
        },
        "per_tier_summary": per_tier_brief,
        "hypothesis": {
            "claim": (
                "can SAC - replay buffer (data reused), maximum-entropy exploration, twin Q - "
                "carry a purely reward-driven learner from the exact resting down start to the "
                "strict lesson-7 acceptance within the same 500k-step budget, after six rungs "
                "of the ladder all ended 0/60? A null result is recorded as the formal conclusion."
            ),
            "per_tier_first_success_any": {
                tier: bool(tier_records[tier]["first_success_any"]) for tier in tiers
            },
            "per_tier_first_successful_eval_steps_per_seed": {
                tier: [
                    record["first_successful_eval_steps"]
                    for record in tier_records[tier]["per_seed"]
                ]
                for tier in tiers
            },
            "per_tier_first_arrival_eval_steps_per_seed": {
                tier: [
                    record["first_arrival_eval_steps"] for record in tier_records[tier]["per_seed"]
                ]
                for tier in tiers
            },
        },
        "failure_analysis": {
            "featured_cases": [
                case for tier in tiers for case in tier_records[tier]["featured_cases"]
            ],
        },
        "training": {
            "train_seeds": train_seeds,
            "tiers": list(tiers),
            "env_steps_per_seed": config.train_steps,
            "total_env_steps": int(config.train_steps * train_seeds * len(tiers)),
            "wall_time_s_total": elapsed,
            "curves_note": "reward_curve_<tier>_<seed> and friends live in trajectories.npz",
        },
        "limitations": [
            (
                "Pure-numpy 2x64 SAC at one fixed standard configuration (gamma 0.99, lr 3e-4, "
                "tau 0.005, target entropy -1, alpha init 0.2, batch 500, buffer 100k); the "
                "auto-vs-fixed alpha contrast is the single sweep performed - no lr/tau/gamma/batch "
                "sweep, no bigger network."
            ),
            (
                "The deterministic evaluation uses the squashed mean (the standard convention), "
                "which is not the stochastic training policy; success requires the strict lesson-7 "
                "settled tail (0.02 m, 0.01 rad, 0.02 m/s, 0.02 rad/s held >= 2 s)."
            ),
            (
                "One gradient update per 32 environment steps (8 parallel envs, one update per "
                "four vector steps) - fewer gradient steps per sample than the textbook "
                "1-per-step SAC; the wall-clock budget sets the cadence."
            ),
            (
                "The replay buffer covers only the exact-down-start trajectory family (no "
                "curriculum, by design): a negative result cannot separate 'the algorithm' from "
                "'no starting-state diversity' in the same way a curriculum row would - recorded "
                "as the cost of the single-variable claim."
            ),
            (
                "SAC never saw pushes, sensor noise or model mismatch; the push comparison "
                "quantifies the gap between training and this disturbance, not a general "
                "robustness claim."
            ),
            (
                "20 episodes per seed is a finite sample; the baseline repeats are deterministic "
                "and coincide (a same-caliber formality, not sampling evidence)."
            ),
        ],
    }
    return report


def build_archive(
    *,
    tier_records,
    baseline_states,
    baseline_controls,
    baseline_push_states,
    baseline_push_controls,
):
    horizon = EVAL_EPISODE_STEPS
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=np.float64),
        "baseline_controls": np.asarray(baseline_controls, dtype=np.float64),
        "baseline_push_states": stack_trajectories(baseline_push_states, horizon)[0],
        "baseline_push_lengths": stack_trajectories(baseline_push_states, horizon)[1],
        "baseline_push_controls": stack_controls(baseline_push_controls, horizon),
    }
    case_index = 0
    for tier, record in tier_records.items():
        for seed_index, curves in record["train_curves"].items():
            for name, curve in curves.items():
                archive[f"{name}_{tier}_{seed_index}"] = curve
        for seed_index, cover in record["cover_curves"].items():
            for name, curve in cover.items():
                archive[f"buffer_{name}_curve_{tier}_{seed_index}"] = curve
        for seed_index, payload in record["policy_payloads"].items():
            for name, array in payload.items():
                archive[f"policy_{tier}_{seed_index}_{name}"] = array
        for seed_index, payload in record["q_payloads"].items():
            for name, array in payload.items():
                archive[f"q_{tier}_{seed_index}_{name}"] = array
        for seed_index, payload in record["det_payloads"].items():
            for suffix in ("states", "controls"):
                archive[f"det_{suffix}_{tier}_{seed_index}"] = payload[suffix]
        for seed_index, store in enumerate(record["eval_stores"]):
            archive[f"eval_states_{tier}_{seed_index}"] = stack_trajectories(
                store["states"], horizon
            )[0]
            archive[f"eval_lengths_{tier}_{seed_index}"] = np.asarray(store["lengths"], dtype=int)
            archive[f"eval_controls_{tier}_{seed_index}"] = stack_controls(
                store["controls"], horizon
            )
            archive[f"eval_terminated_{tier}_{seed_index}"] = np.asarray(
                store["terminated"], dtype=bool
            )
            for name, suffix in (
                ("settled", "settled_s"),
                ("returns", "returns"),
                ("arrival", "first_arrival_s"),
                ("peaks", "peak_force_n"),
                ("max_x", "max_x_m"),
            ):
                archive[f"eval_{suffix}_{tier}_{seed_index}"] = np.asarray(store[name], dtype=float)
        for seed_index, store in enumerate(record["push_stores"]):
            archive[f"push_states_{tier}_{seed_index}"] = stack_trajectories(
                store["states"], horizon
            )[0]
            archive[f"push_lengths_{tier}_{seed_index}"] = np.asarray(store["lengths"], dtype=int)
            archive[f"push_recovery_s_{tier}_{seed_index}"] = np.asarray(
                store["recovery"], dtype=float
            )
        for case in record["featured_cases_arrays"]:
            archive[f"case{case_index}_states"] = case["arrays"]["states"]
            archive[f"case{case_index}_controls"] = case["arrays"]["controls"]
            case_index += 1
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    tiers = report["training"]["tiers"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    curves = (
        "reward_curve",
        "critic_loss_curve",
        "alpha_curve",
        "entropy_curve",
        "log_std_curve",
    )
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
    }
    keys.update(
        f"{curve}_{tier}_{seed}" for tier in tiers for seed in range(seeds) for curve in curves
    )
    keys.update(
        f"buffer_{name}_curve_{tier}_{seed}"
        for tier in tiers
        for seed in range(seeds)
        for name in ("cover_entropy", "cover_fraction")
    )
    keys.update(
        f"policy_{tier}_{seed}_{name}"
        for tier in tiers
        for seed in range(seeds)
        for name in policy_array_names(hidden)
    )
    keys.update(
        f"q_{tier}_{seed}_{prefix}{name}"
        for tier in tiers
        for seed in range(seeds)
        for prefix in ("q1_", "q2_")
        for name in policy_array_names(hidden)
    )
    keys.update(
        f"det_{suffix}_{tier}_{seed}"
        for tier in tiers
        for seed in range(seeds)
        for suffix in ("states", "controls")
    )
    keys.update(
        f"eval_{suffix}_{tier}_{seed}"
        for tier in tiers
        for seed in range(seeds)
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
    keys.update(
        f"push_{suffix}_{tier}_{seed}"
        for tier in tiers
        for seed in range(seeds)
        for suffix in ("states", "lengths", "recovery_s")
    )
    keys.update(
        f"case{index}_{suffix}"
        for index in range(len(report["failure_analysis"]["featured_cases"]))
        for suffix in ("states", "controls")
    )
    return keys


# ------------------------------------------------------------------ figures
def load_tier_curves(report, output, name, tier):
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        return np.asarray(
            [data[f"{name}_{tier}_{index}"] for index in range(report["training"]["train_seeds"])],
            dtype=float,
        )


def load_tier_array(report, output, name, tier, index):
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        return data[f"{name}_{tier}_{index}"]


def save_training_curves(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    tiers = report["training"]["tiers"]
    colors = {"auto": "#0f766e", "fixed": "#b45309"}
    labels = {"auto": "α 自动", "fixed": "α=0.2 固定"}
    reward = {tier: load_tier_curves(report, output, "reward_curve", tier) for tier in tiers}
    alpha = {tier: load_tier_curves(report, output, "alpha_curve", tier) for tier in tiers}
    entropy = {tier: load_tier_curves(report, output, "entropy_curve", tier) for tier in tiers}
    cover = {
        tier: load_tier_curves(report, output, "buffer_cover_entropy_curve", tier) for tier in tiers
    }
    updates = np.arange(1, reward[tiers[0]].shape[1] + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    for tier in tiers:
        for index in range(seeds):
            axes[0, 0].plot(
                updates, reward[tier][index], alpha=0.30, linewidth=0.8, color=colors[tier]
            )
        axes[0, 0].plot(
            updates,
            reward[tier].mean(axis=0),
            color=colors[tier],
            linewidth=1.8,
            label=f"SAC {labels[tier]}（{seeds} 种子均值）",
        )
    axes[0, 0].axhline(0.25, color="gray", linestyle=":", linewidth=1.0, label="悬挂不动 ≈0.25/步")
    axes[0, 0].set(
        xlabel="SAC 更新轮次（每 32 环境步 1 次）",
        ylabel="环境奖励滑动均值（每步）",
        title="训练奖励：细线 = 单个种子",
    )
    axes[0, 0].legend(fontsize=7, loc="lower right")
    for tier in tiers:
        axes[0, 1].plot(
            updates,
            alpha[tier].mean(axis=0),
            color=colors[tier],
            linewidth=1.8,
            label=f"SAC {labels[tier]}",
        )
    axes[0, 1].axhline(-1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0, 1].set(
        xlabel="SAC 更新轮次", ylabel="温度 α", title="α 轨迹（灰点线 = 目标熵 −1 参考）"
    )
    axes[0, 1].legend(fontsize=7, loc="upper right")
    for tier in tiers:
        axes[1, 0].plot(
            updates,
            entropy[tier].mean(axis=0),
            color=colors[tier],
            linewidth=1.8,
            label=f"SAC {labels[tier]}",
        )
    axes[1, 0].set(
        xlabel="SAC 更新轮次", ylabel="策略熵（−E[log π]，nats）", title="策略熵：探索还活着吗"
    )
    axes[1, 0].legend(fontsize=7, loc="upper right")
    checkpoints = np.arange(1, cover[tiers[0]].shape[1] + 1) * 25
    for tier in tiers:
        axes[1, 1].plot(
            checkpoints,
            cover[tier].mean(axis=0),
            "o-",
            color=colors[tier],
            linewidth=1.4,
            label=f"SAC {labels[tier]}",
        )
    axes[1, 1].axhline(np.log(192.0), color="gray", linestyle=":", linewidth=1.0)
    axes[1, 1].set(
        xlabel="环境步数（×1000）",
        ylabel="回放池覆盖熵（nats）",
        title="回放池覆盖熵（灰点线 = 192 格满覆盖上限）",
    )
    axes[1, 1].legend(fontsize=7, loc="upper left")
    eval_points = report["sac_evaluation"]["tiers"][tiers[0]]["per_seed"][0]["eval_curve"]
    eval_steps_axis = np.asarray([point["env_steps"] for point in eval_points]) / 1000.0
    for tier in tiers:
        per_seed = report["sac_evaluation"]["tiers"][tier]["per_seed"]
        success = np.asarray(
            [[int(point["success"]) for point in record["eval_curve"]] for record in per_seed],
            dtype=float,
        )
        for index in range(seeds):
            axes[1, 1].plot(
                eval_steps_axis[: len(success[index])] if False else eval_steps_axis,
                success[index],
                "o--",
                markersize=3,
                alpha=0.45,
                color=colors[tier],
            )
    # the evaluation curve gets its own twin panel: replace the lower-right
    # coverage plot's twin by drawing into a dedicated inset axis
    ax_eval = axes[1, 1].twinx()
    for tier in tiers:
        per_seed = report["sac_evaluation"]["tiers"][tier]["per_seed"]
        ax_eval.plot(
            eval_steps_axis,
            np.asarray(
                [[int(point["success"]) for point in record["eval_curve"]] for record in per_seed],
                dtype=float,
            ).mean(axis=0),
            "*-",
            color=colors[tier],
            linewidth=1.2,
            markersize=4,
            label=f"评估成功率 {labels[tier]}",
        )
    ax_eval.set_ylabel("周期评估成功率（1=成功）", color="#6b7280")
    ax_eval.set_ylim(-0.05, 1.05)
    ax_eval.tick_params(axis="y", labelcolor="#6b7280")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    five_way = report["five_way_comparison"]
    short = ("基线", "PPO", "两阶段", "SAC α自动", "SAC α=0.2")
    labels = short[: len(five_way)]
    totals = [row["episodes"] for row in five_way]
    successes = [row["successes"] for row in five_way]
    settled = [row["median_settled_at_s"] for row in five_way]
    colors = ["#64748b", "#7c3aed", "#2563eb", "#0f766e", "#b45309"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    bars = axes[0].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=colors[: len(labels)],
        width=0.55,
    )
    for bar, s, t in zip(bars, successes, totals, strict=True):
        axes[0].annotate(
            f"{s}/{t}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set_ylim(0, 118)
    axes[0].set(ylabel="验收通过率（%）", title="同口径成功率（第 7 课验收）")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
    for index, row in enumerate(five_way):
        value = settled[index]
        axes[1].bar(index, value if value is not None else 0.0, color=colors[index], width=0.55)
        axes[1].annotate(
            f"{value:.2f} s" if value is not None else "无",
            (index, 0.06 if value is None else value + 0.10),
            ha="center",
            fontsize=8,
        )
    axes[1].set(ylabel="稳定时刻中位（s）", title="稳定时刻（无 = 无成功回合）")
    for index, row in enumerate(five_way):
        text = row["upright_first_arrival"]
        if isinstance(text, str) and "episodes" in text and "never" not in text:
            value = float(text.split("median ")[1].split(" s")[0])
        else:
            value = None
        axes[2].bar(index, value if value is not None else 0.0, color=colors[index], width=0.55)
        axes[2].annotate(
            f"{value:.2f} s" if value is not None else "—",
            (index, 0.03 if value is None else value + 0.05),
            ha="center",
            fontsize=8,
        )
    axes[2].set(ylabel="直立首达中位（s）", title="直立首达（|α|≤0.3 rad，学习行）")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", labelsize=7)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_featured_cases(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    cases = report["failure_analysis"]["featured_cases"]
    tiers = report["training"]["tiers"]
    tier_order = "auto" if "auto" in tiers else tiers[0]
    featured_seed = report["sac_evaluation"]["tiers"][tier_order]["featured_seed_index"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        det_states = data[f"det_states_{tier_order}_{featured_seed}"]
        det_controls = data[f"det_controls_{tier_order}_{featured_seed}"]
        baseline_states = data["baseline_states"]
        baseline_controls = data["baseline_controls"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    ts = np.arange(len(baseline_states)) * dt
    axes[0, 0].plot(
        ts, np.cos(baseline_states[:, 1] - ref_theta), "--", color="gray", label="基线（能量+LQR）"
    )
    axes[0, 0].plot(
        np.arange(len(det_states)) * dt,
        np.cos(det_states[:, 1] - ref_theta),
        color="#0f766e",
        label="SAC（均值动作）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"同一下方初态：典型回合（{tier_order} 档，种子 {featured_seed}）",
    )
    axes[0, 0].legend(fontsize=7, loc="upper left")
    axes[0, 1].plot(ts, baseline_states[:, 0], "--", color="gray")
    axes[0, 1].plot(np.arange(len(det_states)) * dt, det_states[:, 0], color="#2563eb")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].set(
        ylabel="小车位置（m）", xlabel="仿真时间（s）", title="小车位置（红点线 = ±2.4 m 边界）"
    )
    axes[1, 0].stairs(
        baseline_controls * 100.0,
        np.arange(len(baseline_controls) + 1) * dt,
        color="gray",
        label="基线",
    )
    axes[1, 0].stairs(
        det_controls * 100.0, np.arange(len(det_controls) + 1) * dt, color="#2563eb", label="SAC"
    )
    axes[1, 0].set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    counts = report["sac_evaluation"]["tiers"]
    lines = ["训练后失败计数（随机评估）："]
    for tier in tiers:
        lines.append(
            f"  {tier}: 出界 {counts[tier]['failure_counts']['cart_safety_boundary']}、"
            f"速度 {counts[tier]['failure_counts']['velocity_safety_boundary']}、"
            f"超时未稳 {counts[tier]['failure_counts']['timeout_without_settling']}"
        )
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cbd5e1", "alpha": 0.95},
    )
    if cases:
        case = cases[0]
        case_states = np.load(Path(output) / "trajectories.npz", allow_pickle=False)["case0_states"]
        axes[1, 1].plot(
            np.arange(len(case_states)) * dt,
            np.cos(case_states[:, 1] - ref_theta),
            color="#b91c1c",
            linewidth=1.0,
        )
        axes[1, 1].axhspan(-1, 0, alpha=0.08, color="orange")
        axes[1, 1].set(
            ylabel="杆端相对高度",
            xlabel="仿真时间（s）",
            title=f"失败案例：{case['kind']}（{case.get('failure_reason') or '未达标'}）",
        )
    else:
        axes[1, 1].set(title="失败案例：无（成功率见对照图）")
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
    parser.add_argument("--tiers", type=str, nargs="+", default=list(ALPHA_TIERS))
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--update-every", type=int, default=UPDATE_EVERY_ENV_STEPS)
    parser.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--train-episode-steps", type=int, default=TRAIN_EPISODE_STEPS)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(HIDDEN))
    parser.add_argument("--alpha", type=float, default=ALPHA_INIT)
    args = parser.parse_args()
    config = SACConfig(
        train_steps=args.train_steps,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_every_env_steps=args.update_every,
        eval_every_steps=args.eval_every_steps,
        train_episode_steps=args.train_episode_steps,
        alpha_init=args.alpha,
        hidden=tuple(args.hidden),
        warmup_steps=min(args.batch_size, args.train_steps),
    )

    def log(message):
        print(message, file=sys.stderr)  # keep stdout a pure JSON document

    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            config=config,
            tiers=tuple(args.tiers),
            train_seeds=args.train_seeds,
            eval_seed_count=args.eval_seeds,
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["successes"],
                "tiers": {
                    tier: report["sac_evaluation"]["tiers"][tier]["aggregate"]
                    for tier in report["training"]["tiers"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
