"""Lesson 38: combo swing-up - the lesson-7 energy base under a chunked residual.

The user hypothesis (2026-09-06): the inverted pendulum is hard to HOLD upright
because the task is underactuated, so pure learning cannot cross the cliff; a
combination (classical base + learning) should. The literature supports the
architecture: Residual MPC (arXiv 2510.12717) shows "model controller base + RL
residual" beating both pure RL and pure MPC, and residual RL (Johannink et al.,
ICRA 2019) frames the same split. The project's own lesson 30 ran the naive
version and failed FOR RECORDED REASONS - a single-step Gaussian MSE policy
with a fixed sigma turned its budget into a bang-bang disturbance (at-limit
95.8-99.6%, 177/180 timeout-without-settling, 0/60) - while lesson 37 showed
the representation that DOES express the teacher's piecewise hysteretic
behavior: shared trunk + K action-chunk heads + softmax gating (deterministic
arrival 3/3 where the single-step mean path scores 0/3).

This lesson therefore combines the two directly:

    u = clip(u_base + clip(u_residual, +/-a), +/-300 N)

with the base = the lesson-7 HybridSwingupController (energy shaping + LQR
hysteresis) byte-for-byte unchanged, evaluated on the pre-step state, and the
residual = the lesson-37 chunk/multi-modal policy (K=2 experts, H=8-step
chunks) clipped to a in {25, 50} N. Training reuses the lesson-29/30 PPO loop
(chosen over SAC because the two-bank curriculum, the reward-shaping precedent
and the residual-vec machinery all exist there and were FD-verified) at CHUNK
level: one decision = one sampled chunk executed open-loop for H=8 env steps,
with the clipped-surrogate ratio taken on the full mixture density
pi(chunk|s) = sum_k w_k(s) prod_h N(z_h | mu_k[h], sigma^2).  One recorded
lesson-30 failure cause was its fixed sigma = 1.0 exploration, which alone
pins the executed residual at the clip limit ~80% of the time regardless of
the learned mean; this lesson therefore scales the (still fixed, still
unlearned) exploration to the budget, sigma = a/2 (12.5 N at a = 25 N, 25 N at
a = 50 N), so saturation is a policy property rather than a sampling floor.
The reward is reshaped to keep the base in charge of the swing-up and confine
the residual to handoff fine-tuning:

    r = W_up * (1+cos(alpha))/2 - W_down * (1-cos(alpha))/2 - c * z^2
    failing step: r = -failure_penalty

with W_up = 2.0 (heavy top reward), W_down = 0.25 (light bottom penalty),
c = 0.01 on the CLIPPED residual only (the base command is not the learner's
decision - the lesson-30 convention). Budget: 3 seeds x 2 amplitudes x
499,712 env steps per seed (8 envs x 32 chunk decisions x H=8 x 244 updates).

Checks (same conditions throughout, single variable = base+residual combo):
(1) baseline: the lesson-7 controller zero-shot, 20 down-start repeats, the
    lesson-7 acceptance verbatim (recovery_metrics) - also the teacher gate;
(2) guard: a = 0 through the combo pipeline must reproduce the lesson-7 run
    bitwise (states, commands, acceptance);
(3) combo: success >= 18/20 per seed (the not-degrade criterion: not worse
    than the 20/20 baseline), residual |u| mean and at-limit fraction (lesson
    30 died at 95%+; a usable combo must NOT saturate), first success steps,
    and the paired +/-200 N push plans of lessons 29/30;
(4) four-way comparison: baseline (lesson 7, 20/20) / naive residual (lesson
    30, 0/60, cited) / chunked policy WITHOUT the base (lesson 37, 0/60,
    cited) / the combo (this record);
(5) honesty rule: if the combo still fails 0 -> 1 or degrades the base, the
    negative result plus the residual-behavior analysis is the formal record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.act_swingup import log_softmax, softmax
from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.dapg_swingup import first_arrival_time_s
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
    MLPTower,
    VecSwingup,
    baseline_evaluations,
    baseline_push_evaluations,
    clip_gradients_,
    compute_gae,
    down_start_state,
    failure_counts,
    make_push_plans,
    normalize_observation,
    push_schedule,
    select_metrics,
    stack_controls,
    stack_trajectories,
    standardize,
    summarize_episodes,
)
from embodied_learning.experiments.residual_swingup import (
    GEAR,
    capture_time_s,
    residual_command,
    residual_stats,
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
    wrap_angle,
)

EXPERIMENT = "combo_swingup_lesson38"
SCHEMA_VERSION = 1

# --------------------------------------------------------------- protocol knobs
K_EXPERTS = 2  # the lesson-37 multi-modal gate, smallest multi-expert setting
CHUNK_H = 8  # the lesson-37 H=8 chunk (0.32 s at dt = 0.04 s)
TRUNK_HIDDEN = (64, 64)
SIGMA_FRACTION = 0.5  # fixed exploration sigma = a * 0.5 (budget-scaled, unlearned)
RESIDUAL_LIMITS_N = (25.0, 50.0)  # the lesson-30 grid's two smallest budgets
LIMIT_SATURATION = 0.95  # |u_res| >= 95% of the budget counts as "at the limit"

W_UPRIGHT = 2.0  # heavy top reward: the upright half-cosine term
W_BOTTOM = 0.25  # light bottom penalty: the downward half-cosine term
CONTROL_COST_COEF = 0.01  # acts on the clipped residual only (lesson-30 rule)
FAILURE_PENALTY = 10.0
REWARD_SCALE = 0.1  # learning-only rescale of the value targets (lesson-29)

GAMMA_STEP = 0.995  # lesson-29 per-env-step discount...
GAMMA_CHUNK = GAMMA_STEP**CHUNK_H  # ...applied at chunk granularity (0.9607)
GAE_LAMBDA = 0.98
CLIP_EPS = 0.2
LEARNING_RATE = 3e-4
VF_COEF = 0.5
GRAD_CLIP_NORM = 0.5
PPO_EPOCHS = 10

N_ENVS = 8
CHUNKS_PER_SEGMENT = 32  # chunk decisions per PPO segment per env
TRAIN_EPISODE_STEPS = CHUNKS_PER_SEGMENT * CHUNK_H  # 256 env steps = 10.24 s
UPDATES = 244  # 244 x 8 envs x 32 chunks x 8 steps = 499,712 env steps/seed
ENV_STEPS_PER_SEED = UPDATES * N_ENVS * CHUNKS_PER_SEGMENT * CHUNK_H
EVAL_EVERY = 20  # periodic deterministic evaluations (12 checkpoints + last)

SEED_OFFSET_READOUT = 77  # chunk/gate head init stream (lesson-37 convention)

DOWN_ANGLE_DEG = -180.0
NOT_DEGRADE_RATE = 18.0 / 20.0  # the protocol's not-degrade criterion: >= 18/20
DEFAULT_LOG_STD = float(np.log(0.25 * SIGMA_FRACTION))  # sigma = a/2 at a = 25 N


# Cited four-way rows (official records, not re-run here - the act_swingup
# precedent for citing cross-lesson rows).
LESSON30_NAIVE_RESIDUAL_REFERENCE = {
    "source": "results/residual_swingup_2026-09-06 (official lesson-30 record, docs/35)",
    "episodes": 60,
    "successes": 0,
    "limits_n": [25.0, 50.0, 100.0],
    "residual_mean_abs_n": [24.95, 49.64, 97.83],
    "fraction_at_limit_deterministic": [0.9956, 0.9844, 0.9582],
    "failure_mode": "timeout_without_settling 177/180 (out-of-bounds only 3)",
    "push": "0/60 vs baseline 20/20 (median recovery 2.88 s)",
}
LESSON37_CHUNK_NO_BASE_REFERENCE = {
    "source": "results/act_swingup_2026-09-06 (official lesson-37 record, docs/42)",
    "episodes": 60,
    "successes": 0,
    "tiers": ["mse_single", "block_mse", "moe_k2", "moe_k4", "moe_k4_h8"],
    "stochastic_first_arrival": "moe tiers 60/60, block 49/60, single-step 58/60",
    "deterministic": "MoE arrival 3/3 but settled never (0/3 per tier)",
}


@dataclass(frozen=True)
class ComboConfig:
    n_envs: int = N_ENVS
    chunks_per_segment: int = CHUNKS_PER_SEGMENT
    chunk_h: int = CHUNK_H
    updates: int = UPDATES
    epochs: int = PPO_EPOCHS
    minibatch: int = 256  # chunk decisions per gradient minibatch
    gamma: float = GAMMA_CHUNK
    gae_lambda: float = GAE_LAMBDA
    clip_eps: float = CLIP_EPS
    lr: float = LEARNING_RATE
    vf_coef: float = VF_COEF
    grad_clip: float = GRAD_CLIP_NORM
    log_std: float = DEFAULT_LOG_STD  # sigma = a/2 at a = 25 N (budget-scaled)
    reward_scale: float = REWARD_SCALE
    hidden: tuple[int, ...] = TRUNK_HIDDEN
    n_experts: int = K_EXPERTS
    eval_every: int = EVAL_EVERY
    task_envs: int = 4  # the lesson-29/30 two-bank curriculum

    @property
    def train_episode_steps(self):
        """Env steps per training episode; chunk-aligned by construction."""
        return self.chunks_per_segment * self.chunk_h

    def __post_init__(self):
        positive = (
            self.n_envs,
            self.chunks_per_segment,
            self.chunk_h,
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
        )
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("combo hyperparameters must be finite and positive")
        if not 1 <= self.chunk_h <= 16:
            raise ValueError("chunk_h must be in [1, 16]")
        if not 2 <= self.n_experts <= 8:
            raise ValueError("n_experts must be in [2, 8] (the multi-modal chunk gate)")
        if not 0 <= self.task_envs <= self.n_envs:
            raise ValueError("task_envs must be between 0 and n_envs")
        if not 0 < self.eval_every <= self.updates:
            raise ValueError("eval_every must be in (0, updates]")
        if self.minibatch > self.n_envs * self.chunks_per_segment:
            raise ValueError("minibatch cannot exceed the segment decision count")
        if any(units < 1 for units in self.hidden):
            raise ValueError("hidden layer sizes must be positive")


# ------------------------------------------------------------------- the reward
class ComboReward:
    """Experiment-side reward: heavy top, light bottom, cost on the residual.

    r = W_up*(1+cos(alpha))/2 - W_down*(1-cos(alpha))/2 - c*z^2 for a live
    step; the failing step is replaced by -failure_penalty.  alpha is the
    wrapped pole error (0 upright, +/-pi down), z is the CLIPPED residual the
    learner executed - the base command is never the judged action (the
    lesson-30 rule), and the base therefore keeps the swing-up job while the
    gradient only pushes the residual toward "cheap, and keep the top".
    """

    def __init__(
        self,
        reference,
        w_upright=W_UPRIGHT,
        w_bottom=W_BOTTOM,
        control_cost_coef=CONTROL_COST_COEF,
        failure_penalty=FAILURE_PENALTY,
    ):
        self.reference = np.asarray(reference, dtype=float)
        self.w_upright = float(w_upright)
        self.w_bottom = float(w_bottom)
        self.control_cost_coef = float(control_cost_coef)
        self.failure_penalty = float(failure_penalty)
        for value in (w_upright, w_bottom, control_cost_coef, failure_penalty):
            if not np.isfinite(value):
                raise ValueError("reward constants must be finite")
        if self.w_upright <= 0 or self.w_bottom < 0:
            raise ValueError("need w_upright > 0 and w_bottom >= 0")

    def terms(self, state, residual, terminated):
        """Reward for one env step, judged by the post-step state."""
        residual = float(residual)
        if terminated:
            return {
                "upright": 0.0,
                "bottom": 0.0,
                "control_cost": 0.0,
                "failure": self.failure_penalty,
                "total": -self.failure_penalty,
            }
        alpha = float(wrap_angle(state[1] - self.reference[1]))
        upright = self.w_upright * (1.0 + np.cos(alpha)) / 2.0
        bottom = self.w_bottom * (1.0 - np.cos(alpha)) / 2.0
        cost = self.control_cost_coef * residual * residual
        return {
            "upright": float(upright),
            "bottom": float(bottom),
            "control_cost": float(cost),
            "failure": 0.0,
            "total": float(upright - bottom - cost),
        }

    def __call__(self, state, residual, terminated):
        return self.terms(state, residual, terminated)["total"]

    def as_dict(self):
        return {
            "formula": (
                "r = W_up*(1+cos(alpha))/2 - W_down*(1-cos(alpha))/2 "
                "- control_cost_coef*z^2; on the failing step r = -failure_penalty"
            ),
            "w_upright": self.w_upright,
            "w_bottom": self.w_bottom,
            "control_cost_coef": self.control_cost_coef,
            "failure_penalty": self.failure_penalty,
            "alpha": "wrapped(pole_angle - reference_angle); 0 upright, +/-pi down",
            "z": "the clipped residual in normalized units (a=25 N -> 0.25)",
            "rationale": (
                "heavy top + light bottom keeps the base in charge of the swing-up and "
                "confines the residual gradient to handoff fine-tuning and cheap hold"
            ),
        }


# ------------------------------------------------------- the chunked MoE policy
class ChunkResidualPolicy:
    """Lesson-37 architecture as a residual policy: trunk + K chunk heads + gate.

    A decision is one H-step chunk.  Sampling draws expert k ~ softmax(gate(s))
    and then z = mu_k + sigma*eps with sigma FIXED and UNLEARNED (exp(log_std),
    the lesson-29/37 convention of a fixed std, but budget-scaled here: the
    caller passes log(a/2) so exploration stays inside the residual budget - a
    recorded response to lesson-30's fixed sigma = 1.0, which pinned the
    executed residual at the clip limit regardless of the learned mean).  The
    sampled chunk's density is the full mixture
    pi(z|s) = sum_k w_k(s) prod_h N(z_h | mu_k[h], sigma^2), which is what the
    PPO ratio uses.  The deterministic plan is the gated mixture-mean chunk.
    """

    def __init__(self, n_experts, horizon, hidden, seed, log_std=DEFAULT_LOG_STD):
        self.n_experts = int(n_experts)
        self.horizon = int(horizon)
        self.trunk = MLPTower(STATE_INPUTS, tuple(hidden), tuple(hidden)[-1], seed)
        rng = np.random.default_rng([*seed, SEED_OFFSET_READOUT])
        fan = tuple(hidden)[-1]
        scale = np.sqrt(2.0 / fan)
        self.expert_weights = [
            rng.normal(0.0, scale, (fan, self.horizon)) for _ in range(self.n_experts)
        ]
        self.expert_biases = [np.zeros(self.horizon) for _ in range(self.n_experts)]
        self.gate_weight = rng.normal(0.0, scale, (fan, self.n_experts))
        self.gate_bias = np.zeros(self.n_experts)
        self.log_std = np.full(1, float(log_std))  # fixed: no gradient, not a parameter

    def parameters(self):
        return [
            *self.trunk.weights,
            *self.trunk.biases,
            *self.expert_weights,
            *self.expert_biases,
            self.gate_weight,
            self.gate_bias,
        ]

    def forward(self, obs):
        """(means (B,K,H), gate logits (B,K), trunk output h (B,fan))."""
        h = self.trunk.forward(np.asarray(obs, dtype=float))[0]
        means = np.stack(
            [h @ w + b for w, b in zip(self.expert_weights, self.expert_biases, strict=True)],
            axis=1,
        )
        gate = h @ self.gate_weight + self.gate_bias
        return means, gate, h

    def gate_weights(self, obs):
        h = self.trunk.forward(np.asarray(obs, dtype=float))[0]
        return softmax(h @ self.gate_weight + self.gate_bias)

    def mean_chunks(self, obs):
        """Gated mixture-mean chunk per row (the deterministic plan)."""
        means, gate, _h = self.forward(obs)
        weights = softmax(gate)
        return (weights[:, :, None] * means).sum(axis=1)

    def sample_chunks(self, obs, rng):
        """Categorical-expert then Gaussian chunk sample, with mixture logp."""
        obs = np.asarray(obs, dtype=float)
        means, gate, _h = self.forward(obs)
        weights = softmax(gate)
        sigma = float(np.exp(self.log_std[0]))
        chunks = np.empty((len(obs), self.horizon))
        for row in range(len(obs)):
            expert = int(rng.choice(self.n_experts, p=weights[row]))
            chunks[row] = means[row, expert] + sigma * rng.standard_normal(self.horizon)
        logp = mixture_log_prob(self, obs, chunks)
        return chunks, logp

    def arrays(self):
        payload = {"log_std": self.log_std.copy()}
        for index, (weight, bias) in enumerate(zip(self.trunk.weights, self.trunk.biases)):
            payload[f"trunk_weight_{index}"] = weight.copy()
            payload[f"trunk_bias_{index}"] = bias.copy()
        for expert in range(self.n_experts):
            payload[f"expert_{expert}_weight"] = self.expert_weights[expert].copy()
            payload[f"expert_{expert}_bias"] = self.expert_biases[expert].copy()
        payload["gate_weight"] = self.gate_weight.copy()
        payload["gate_bias"] = self.gate_bias.copy()
        return payload


def policy_array_names(hidden, n_experts):
    """Archive key names of one serialized chunk policy payload."""
    names = ["log_std"]
    for index in range(len(hidden) + 1):
        names.extend((f"trunk_weight_{index}", f"trunk_bias_{index}"))
    for expert in range(n_experts):
        names.extend((f"expert_{expert}_weight", f"expert_{expert}_bias"))
    names.extend(("gate_weight", "gate_bias"))
    return tuple(names)


def mixture_log_prob(policy, obs, chunks):
    """log pi(chunk|s) = logsumexp_k [log w_k + sum_h log N(z_h|mu_kh, sigma)]."""
    means, gate, _h = policy.forward(obs)
    chunks = np.asarray(chunks, dtype=float)
    sigma = float(np.exp(policy.log_std[0]))
    log_w = log_softmax(gate)
    error2 = ((chunks[:, None, :] - means) ** 2).sum(axis=2)
    constant = -policy.horizon * (np.log(sigma) + 0.5 * np.log(2.0 * np.pi))
    log_p = log_w - 0.5 * error2 / sigma**2 + constant
    top = np.max(log_p, axis=1, keepdims=True)
    return top[:, 0] + np.log(np.exp(log_p - top).sum(axis=1))


def mixture_log_prob_parts(policy, obs, chunks):
    """(means, gate, h, log_w, sigma2, logp, posterior) - shared math core."""
    means, gate, h = policy.forward(obs)
    chunks = np.asarray(chunks, dtype=float)
    sigma = float(np.exp(policy.log_std[0]))
    sigma2 = sigma * sigma
    log_w = log_softmax(gate)
    error2 = ((chunks[:, None, :] - means) ** 2).sum(axis=2)
    constant = -policy.horizon * (np.log(sigma) + 0.5 * np.log(2.0 * np.pi))
    log_p = log_w - 0.5 * error2 / sigma2 + constant
    top = np.max(log_p, axis=1, keepdims=True)
    logp = top[:, 0] + np.log(np.exp(log_p - top).sum(axis=1))
    posterior = np.exp(log_p - top)
    posterior /= posterior.sum(axis=1, keepdims=True)
    return means, gate, h, log_w, sigma2, logp, posterior


def combo_ppo_losses_and_gradients(
    policy, value, obs, chunks, old_logp, advantages, returns, config
):
    """Clipped surrogate on the mixture density + value loss; hand gradients.

    The surrogate gradient flows through the SAME analytic posterior as the
    lesson-37 gated NLL (p_k = softmax of gate logit + expert log density),
    weighted per-sample by the surrogate coefficient instead of 1/B.
    """
    obs = np.asarray(obs, dtype=float)
    chunks = np.asarray(chunks, dtype=float)
    old_logp = np.asarray(old_logp, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if not 0 < len(obs) == len(chunks) == len(old_logp) == len(advantages) == len(returns):
        raise ValueError("expected nonempty aligned 1-D batches")
    if not all(np.isfinite(a).all() for a in (obs, chunks, old_logp, advantages, returns)):
        raise ValueError("nonfinite combo PPO batch")
    batch = len(obs)
    means, _gate, h, log_w, sigma2, logp, posterior = mixture_log_prob_parts(policy, obs, chunks)
    ratio = np.exp(logp - old_logp)
    clipped = np.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
    surrogate = np.minimum(ratio * advantages, clipped * advantages)
    policy_loss = -float(np.mean(surrogate))

    value_out, value_cache = value.forward(obs)
    values = value_out[:, 0]
    value_loss = config.vf_coef * float(np.mean((values - returns) ** 2))
    total = policy_loss + value_loss

    unclipped = (ratio * advantages) <= (clipped * advantages)
    # d policy_loss / d logp_i  (the clip branch contributes zero, as in lesson 29)
    coef = np.where(unclipped, -advantages * ratio / batch, 0.0)
    weights = np.exp(log_w)
    # d logp_i / d gate logit = posterior - weight (logsumexp chain through the
    # softmax jacobian; the lesson-37 formula w - p is the NEGATIVE because the
    # lesson-37 loss is -logp)
    d_gate = coef[:, None] * (posterior - weights)
    d_mu = (
        coef[:, None, None] * posterior[:, :, None] * (chunks[:, None, :] - means) / sigma2
    )  # d logp_i / d mu_k[h]
    grad_expert_w = [h.T @ d_mu[:, expert, :] for expert in range(policy.n_experts)]
    grad_expert_b = [d_mu[:, expert, :].sum(axis=0) for expert in range(policy.n_experts)]
    grad_gate_w = h.T @ d_gate
    grad_gate_b = d_gate.sum(axis=0)
    grad_h = d_gate @ policy.gate_weight.T
    for expert in range(policy.n_experts):
        grad_h += d_mu[:, expert, :] @ policy.expert_weights[expert].T
    grad_trunk_w, grad_trunk_b = policy.trunk.backward(policy.trunk.forward(obs)[1], grad_h)

    grad_value = (2.0 * config.vf_coef * (values - returns) / batch).reshape(batch, 1)
    value_grad_w, value_grad_b = value.backward(value_cache, grad_value)

    gradients = [
        *grad_trunk_w,
        *grad_trunk_b,
        *grad_expert_w,
        *grad_expert_b,
        grad_gate_w,
        grad_gate_b,
        *value_grad_w,
        *value_grad_b,
    ]
    if not np.isfinite(total) or not all(np.isfinite(g).all() for g in gradients):
        raise ValueError("nonfinite combo PPO loss or gradients; training diverged")
    losses = {
        "total": float(total),
        "policy": policy_loss,
        "value": value_loss,
        "clip_fraction": float(np.mean(~unclipped)),
        "mean_logp": float(np.mean(logp)),
    }
    return losses, gradients


# ---------------------------------------------------------------- the vec env
class ChunkResidualVecSwingup(VecSwingup):
    """Lesson-29 parallel envs at chunk level with the lesson-7 base in the loop.

    step() receives H-step pre-clip chunk samples per env and executes them
    open-loop: every env step the base controller is evaluated on the current
    state and the command is clip(u_base + clip(z_h, +/-a), +/-3).  Training
    episodes are chunk-aligned (train_episode_steps = chunks_per_segment*H), so
    a TimeLimit truncation can only land on a chunk boundary; a physical
    failure may land mid-chunk, in which case the fresh episode runs the BASE
    ALONE (residual zero) until the next chunk boundary - the stale chunk is
    discarded instead of leaking into the new episode (recorded convention).
    Each env owns one HybridSwingupController, refreshed at every episode
    start, matching the lesson-7/30 bookkeeping.
    """

    def __init__(self, reward, *, design, residual_limit_norm, chunk_h, **kwargs):
        super().__init__(reward, **kwargs)
        self.design = design
        self.chunk_h = int(chunk_h)
        self.residual_limit_norm = float(residual_limit_norm)
        if not 0.0 <= self.residual_limit_norm <= design.controller.control_limit:
            raise ValueError("residual limit must be within [0, control_limit]")
        self.bases = [HybridSwingupController(env.unwrapped.model, design) for env in self.envs]

    def step(self, chunks):
        """One chunk decision per env: chunks has shape (n_envs, chunk_h)."""
        chunks = np.asarray(chunks, dtype=float)
        if chunks.shape != (self.n_envs, self.chunk_h):
            raise ValueError(f"chunks must have shape ({self.n_envs}, {self.chunk_h})")
        reference = self.reward.reference
        terminal_obs = np.empty((self.n_envs, STATE_INPUTS))
        live_obs = np.empty((self.n_envs, STATE_INPUTS))
        rewards = np.empty(self.n_envs)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        for index, env in enumerate(self.envs):
            chunk_reward = 0.0
            failed = timed_out = False
            state_after = np.zeros(4)
            for hop in range(self.chunk_h):
                state = np.asarray(env.unwrapped._get_obs(), dtype=float)
                if failed or timed_out:
                    executed = 0.0  # fresh episode: the base runs alone
                else:
                    limit = self.residual_limit_norm
                    executed = float(np.clip(float(chunks[index, hop]), -limit, limit))
                base = self.bases[index].action(state)
                command, executed = residual_command(base, executed, self.residual_limit_norm)
                env.unwrapped.data.qfrc_applied[0] = 0.0
                state_after, _, done, timed, _info = env.step(command)
                chunk_reward += self.reward(state_after, float(executed), done)
                failed |= done
                timed_out |= timed
                if done or timed:
                    fresh = self._start_state(index)
                    env.unwrapped.set_state(fresh[:2], fresh[2:])
                    env.unwrapped.data.qfrc_applied[0] = 0.0
                    self.bases[index] = HybridSwingupController(env.unwrapped.model, self.design)
            safe = state_after if np.isfinite(state_after).all() else np.zeros(4)
            terminal_obs[index] = normalize_observation(safe, reference)
            live_state = np.asarray(env.unwrapped._get_obs(), dtype=float)
            live_obs[index] = normalize_observation(live_state, reference)
            rewards[index] = chunk_reward
            terminated[index], truncated[index] = failed, timed_out
        return terminal_obs, rewards, terminated, truncated, live_obs


# ------------------------------------------------------- chunk-level PPO loop
def train_combo_ppo(
    vec_env, *, config, init_seed, act_seed, shuffle_seed, eval_hook=None, log=None
):
    """Chunk-level PPO: one decision = one open-loop H-step chunk per env."""
    policy = ChunkResidualPolicy(
        config.n_experts, config.chunk_h, config.hidden, init_seed, config.log_std
    )
    value = MLPTower(STATE_INPUTS, config.hidden, 1, [*init_seed, 1])
    parameters = [*policy.parameters(), *value.weights, *value.biases]
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    action_rng = np.random.default_rng(act_seed)
    shuffle_rng = np.random.default_rng(shuffle_seed)
    decisions = config.n_envs * config.chunks_per_segment
    env_steps_per_update = decisions * config.chunk_h

    reward_curve = np.empty(config.updates)
    value_loss_curve = np.empty(config.updates)
    clip_fraction_curve = np.empty(config.updates)
    terminated_curve = np.empty(config.updates)
    eval_steps, eval_records = [], []
    observations = vec_env.reset()
    started = time.perf_counter()

    for update in range(config.updates):
        batch_obs = np.empty((config.chunks_per_segment, config.n_envs, STATE_INPUTS))
        batch_chunks = np.empty((config.chunks_per_segment, config.n_envs, config.chunk_h))
        batch_logp = np.empty((config.chunks_per_segment, config.n_envs))
        batch_values = np.empty((config.chunks_per_segment, config.n_envs))
        batch_rewards = np.empty((config.chunks_per_segment, config.n_envs))
        batch_terminated = np.zeros((config.chunks_per_segment, config.n_envs), dtype=bool)
        batch_truncated = np.zeros((config.chunks_per_segment, config.n_envs), dtype=bool)
        batch_terminal_values = np.empty((config.chunks_per_segment, config.n_envs))
        reward_sum, terminated_count = 0.0, 0
        for step in range(config.chunks_per_segment):
            chunks, logp = policy.sample_chunks(observations, action_rng)
            batch_obs[step] = observations
            batch_chunks[step] = chunks
            batch_logp[step] = logp
            batch_values[step] = value.forward(observations)[0][:, 0]
            terminal_obs, rewards, terminated, truncated, observations = vec_env.step(chunks)
            batch_rewards[step] = rewards
            batch_terminated[step] = terminated
            batch_truncated[step] = truncated
            batch_terminal_values[step] = value.forward(terminal_obs)[0][:, 0]
            reward_sum += float(rewards.sum())
            terminated_count += int(terminated.sum())
        advantages, returns = compute_gae(
            batch_rewards.reshape(-1, config.n_envs) * config.reward_scale,
            batch_values.reshape(-1, config.n_envs),
            batch_terminated.reshape(-1, config.n_envs),
            batch_truncated.reshape(-1, config.n_envs),
            batch_terminal_values.reshape(-1, config.n_envs),
            config.gamma,
            config.gae_lambda,
        )
        advantages = standardize(advantages.ravel())
        returns = returns.ravel()
        order = shuffle_rng.permutation(decisions)

        value_loss_sum, clip_sum, grad_steps = 0.0, 0.0, 0
        flat_obs = batch_obs.reshape((-1, STATE_INPUTS))
        flat_chunks = batch_chunks.reshape((-1, config.chunk_h))
        flat_logp = batch_logp.ravel()
        for _epoch in range(config.epochs):
            for start in range(0, decisions, config.minibatch):
                minibatch = order[start : start + config.minibatch]
                losses, gradients = combo_ppo_losses_and_gradients(
                    policy,
                    value,
                    flat_obs[minibatch],
                    flat_chunks[minibatch],
                    flat_logp[minibatch],
                    advantages[minibatch],
                    returns[minibatch],
                    config,
                )
                clip_gradients_(gradients, config.grad_clip)
                optimizer.step(parameters, gradients)
                value_loss_sum += losses["value"]
                clip_sum += losses["clip_fraction"]
                grad_steps += 1

        reward_curve[update] = reward_sum / env_steps_per_update
        terminated_curve[update] = terminated_count / env_steps_per_update
        value_loss_curve[update] = value_loss_sum / grad_steps
        clip_fraction_curve[update] = clip_sum / grad_steps
        optimizer.lr = config.lr * (1.0 - (update + 1) / config.updates)

        env_steps = (update + 1) * env_steps_per_update
        if eval_hook is not None and (
            (update + 1) % config.eval_every == 0 or update == config.updates - 1
        ):
            eval_steps.append(env_steps)
            eval_records.append(eval_hook(policy, env_steps))
        if log is not None and (update + 1) % 25 == 0:
            message = (
                f"update {update + 1}/{config.updates}, steps {env_steps}, "
                f"reward {reward_curve[update]:.3f}, term {terminated_curve[update]:.2f}"
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
        "clip_fraction_curve": clip_fraction_curve,
        "eval_steps": np.asarray(eval_steps, dtype=int),
        "eval_records": eval_records,
        "env_steps": int(config.updates * env_steps_per_update),
        "wall_time_s": time.perf_counter() - started,
    }


# ------------------------------------------------------------------ evaluation
def run_combo_episode(
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
    chunk_h=None,
):
    """One episode from the exact down start, base + open-loop chunked residual.

    Array alignment follows lesson 7: states[k] is the state before action k;
    controls/residuals/base_controls[k] act on [k*dt, (k+1)*dt).  With a = 0
    the policy is never queried and the loop reduces to the lesson-7
    run_scenario (the guard pins this bitwise).  A chunk is planned every
    chunk_h steps (the gated mixture mean when deterministic, one sampled
    expert chunk otherwise) and executed open-loop.
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
    if limit_norm == 0.0 and policy is not None:
        raise ValueError("pass policy=None for the a=0 guard run")
    if not deterministic and rng is None:
        raise ValueError("sampled episodes need an rng")
    hop_total = int(policy.horizon if policy is not None else (chunk_h or 1))
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
        planned = None
        hop = 0
        for force in schedule:
            if limit_norm > 0.0 and hop == 0:
                obs = normalize_observation(state, reward.reference)[None, :]
                if deterministic:
                    planned = policy.mean_chunks(obs)[0]
                else:
                    sampled, _logp = policy.sample_chunks(obs, rng)
                    planned = np.asarray(sampled[0], dtype=float)
                planned = np.asarray(planned, dtype=float)
            z = float(planned[hop]) if limit_norm > 0.0 else 0.0
            base = controller.action(state)
            command, executed = residual_command(base, z, limit_norm)
            env.unwrapped.data.qfrc_applied[0] = float(force)
            state, _, terminated, truncated, info = env.step(command)
            states.append(state.copy())
            controls.append(float(command[0]))
            residuals.append(float(executed))
            base_controls.append(float(base[0]))
            forces.append(float(force))
            modes.append(controller.mode)
            failure_reason = info["failure_reason"]
            hop = (hop + 1) % hop_total
            if terminated or truncated:
                break
        arrays = {
            # float64 on purpose: the guard compares these bitwise against the
            # lesson-7 run_scenario arrays (the lesson-30 convention)
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


def combo_episode_metrics(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance applied to a combo episode (the same function)."""
    return recovery_metrics(arrays, {"failure_reason": failure_reason}, reference, dt)


def combo_episode_rewards(arrays, reward):
    """Recompute per-step rewards with the executed residual as judged action."""
    residuals, end_flags, states = arrays["residuals"], arrays["end_flags"], arrays["states"]
    last = len(residuals) - 1
    return np.asarray(
        [
            reward(states[step + 1], float(residuals[step]), bool(end_flags[0]) and step == last)
            for step in range(len(residuals))
        ],
        dtype=float,
    )


def evaluate_combo_stochastic(policy, reward, design, *, master_seed, limit_norm, count=EVAL_SEEDS):
    """`count` chunk-sampled episodes from the exact down start."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, SEED_OFFSET_EVAL, eval_seed])
        arrays, reason = run_combo_episode(
            policy,
            reward,
            design,
            horizon=EVAL_EPISODE_STEPS,
            residual_limit_norm=limit_norm,
            env_seed=eval_seed,
            deterministic=False,
            rng=rng,
        )
        metrics = combo_episode_metrics(arrays, reason, reward.reference, design.dt)
        episodes.append(
            {
                "eval_seed": eval_seed,
                **select_metrics(metrics),
                "first_arrival_s": first_arrival_time_s(
                    arrays["states"], reward.reference, design.dt
                ),
                "return": float(combo_episode_rewards(arrays, reward).sum()),
                "arrays": arrays,
            }
        )
    return episodes


def deterministic_combo(policy, reward, design, *, limit_norm):
    """One deterministic-chunk episode (the executed-policy view)."""
    arrays, reason = run_combo_episode(
        policy,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=limit_norm,
        env_seed=0,
        deterministic=True,
    )
    metrics = combo_episode_metrics(arrays, reason, reward.reference, design.dt)
    return arrays, reason, metrics


def baseline_guard(design, reward, reference, dt):
    """a = 0 through the combo pipeline must equal the lesson-7 run bitwise."""
    arrays, reason = run_combo_episode(
        None,
        reward,
        design,
        horizon=EVAL_EPISODE_STEPS,
        residual_limit_norm=0.0,
        env_seed=0,
        deterministic=True,
    )
    reference_arrays, _metadata = run_scenario(Scenario("down", "down"), design, EVAL_EPISODE_STEPS)
    metrics = combo_episode_metrics(arrays, reason, reference, dt)
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


def not_degrade_verdict(successes_per_seed, episodes_per_seed, threshold_rate=NOT_DEGRADE_RATE):
    """The protocol's not-degrade criterion: every seed's rate >= 18/20.

    Returns a dict with the per-seed rates, the aggregate rate and the boolean
    verdict - a pure function so the criterion itself is unit-tested.
    """
    successes_per_seed = [int(value) for value in successes_per_seed]
    episodes_per_seed = int(episodes_per_seed)
    if episodes_per_seed <= 0 or not successes_per_seed:
        raise ValueError("need a positive episode count and at least one seed")
    if any(not 0 <= value <= episodes_per_seed for value in successes_per_seed):
        raise ValueError("success counts must live in [0, episodes_per_seed]")
    rates = [value / episodes_per_seed for value in successes_per_seed]
    aggregate = sum(successes_per_seed) / (episodes_per_seed * len(successes_per_seed))
    return {
        "episodes_per_seed": episodes_per_seed,
        "successes_per_seed": successes_per_seed,
        "rates_per_seed": rates,
        "aggregate_rate": aggregate,
        "threshold_rate": float(threshold_rate),
        "not_degrade": bool(min(rates) >= threshold_rate and aggregate >= threshold_rate),
    }


def annotate_case_limits(cases, eval_episodes, push_episodes):
    """Copy the scan amplitude onto each featured case (lesson-30 pattern)."""
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


# ------------------------------------------------------------------ experiment
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
    """Run the lesson-38 combo probe into `output`; the directory must not exist."""
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(train_seeds, int) or not 1 <= train_seeds <= 8:
        raise ValueError("train_seeds must be an integer in [1, 8]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    limits = tuple(float(value) for value in limits_n)
    if not limits or any(not 0.0 < value <= GEAR * CONTROL_LIMIT for value in limits):
        raise ValueError("residual limits must be in (0, 300] N")
    config = config or ComboConfig()
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    if abs(design.actuator_gear - GEAR) > 1e-12:
        raise ValueError("actuator gear disagrees with the recovery_metrics convention")
    reward = ComboReward(reference)

    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )
    baseline_summary = summarize_episodes(baseline_records)
    teacher_gate = {
        "guard": "the lesson-7 re-run must pass the lesson-7 acceptance on every repeat",
        "episodes": len(baseline_records),
        "successes": baseline_summary["successes"],
        "gate_passed": bool(all(r["recovered"] for r in baseline_records)),
        "median_settled_at_s": baseline_summary["median_settled_at_s"],
    }
    if not teacher_gate["gate_passed"]:
        raise ValueError("teacher quality gate failed: the base is not verified")
    guard = baseline_guard(design, reward, reference, dt)
    if not guard["bitwise_identical_states"] or not guard["bitwise_identical_controls"]:
        raise ValueError("a=0 guard failed: the combo pipeline is not the lesson-7 pipeline")
    plans = make_push_plans(dt, eval_seed_count, seed)
    baseline_push_records, baseline_push_states, baseline_push_controls = baseline_push_evaluations(
        design, plans, EVAL_EPISODE_STEPS
    )

    output.mkdir(parents=True, exist_ok=False)
    sweep_entries = []
    policy_payloads = {}  # (amp_index, seed_index) -> policy arrays
    reward_curves = {}  # (amp_index, seed_index) -> per-update reward
    det_arrays_store = {}
    eval_stores = {index: [] for index in range(len(limits))}
    push_stores = {index: [] for index in range(len(limits))}
    all_eval_episodes, all_push_episodes = [], []

    for amp_index, limit_n in enumerate(limits):
        limit_norm = limit_n / GEAR
        # The fixed exploration sigma is scaled to the budget (sigma = a/2) so
        # the at-limit fraction measures the policy, not the sampling floor.
        amp_config = replace(config, log_std=float(np.log(limit_norm * SIGMA_FRACTION)))
        per_seed_records, det_records = [], []
        det_residual_pool, stoch_residual_pool = [], []
        for seed_index in range(train_seeds):
            vec_env = ChunkResidualVecSwingup(
                reward,
                design=design,
                residual_limit_norm=limit_norm,
                chunk_h=config.chunk_h,
                n_envs=config.n_envs,
                episode_steps=config.train_episode_steps,
                base_seed=10_000 + seed * 1000 + amp_index * 100 + seed_index,
                task_envs=config.task_envs,
            )

            def eval_hook(policy, _env_steps, _limit=limit_norm):
                arrays, _reason, metrics = deterministic_combo(
                    policy, reward, design, limit_norm=_limit
                )
                return {
                    "success": bool(metrics["recovered"] and not metrics["terminated"]),
                    "settled_at_s": metrics["settled_at_s"],
                    "first_arrival_s": first_arrival_time_s(
                        arrays["states"], reward.reference, design.dt
                    ),
                    "return": float(combo_episode_rewards(arrays, reward).sum()),
                }

            result = train_combo_ppo(
                vec_env,
                config=amp_config,
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

            eval_episodes = evaluate_combo_stochastic(
                policy,
                reward,
                design,
                master_seed=seed,
                limit_norm=limit_norm,
                count=eval_seed_count,
            )
            det_arrays, _det_reason, det_metrics = deterministic_combo(
                policy, reward, design, limit_norm=limit_norm
            )
            push_episodes = []
            for plan in plans:
                arrays, reason = run_combo_episode(
                    policy,
                    reward,
                    design,
                    horizon=EVAL_EPISODE_STEPS,
                    residual_limit_norm=limit_norm,
                    env_seed=plan["index"],
                    deterministic=True,
                    schedule=push_schedule(plan, dt, EVAL_EPISODE_STEPS),
                )
                metrics = combo_episode_metrics(arrays, reason, reference, dt)
                push_episodes.append(
                    {
                        "plan_index": plan["index"],
                        "force_n": plan["force_n"],
                        "start_s": plan["start_s"],
                        **select_metrics(metrics, PUSH_FIELDS),
                        "return": float(combo_episode_rewards(arrays, reward).sum()),
                        "arrays": arrays,
                    }
                )

            det_record = {
                "seed_index": seed_index,
                "recovered": bool(det_metrics["recovered"]),
                "terminated": bool(det_metrics["terminated"]),
                "settled_at_s": det_metrics["settled_at_s"],
                "first_arrival_s": first_arrival_time_s(det_arrays["states"], reference, dt),
                "capture_time_s": capture_time_s(det_arrays["modes"], dt),
                "return": float(combo_episode_rewards(det_arrays, reward).sum()),
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
                    "arrivals": [
                        float(episode["first_arrival_s"])
                        if episode["first_arrival_s"] is not None
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
                "first_arrival_eval_steps": next(
                    (
                        int(step)
                        for step, entry in zip(
                            result["eval_steps"], result["eval_records"], strict=True
                        )
                        if entry.get("first_arrival_s") is not None
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
                    f"combo a={limit_n:.0f} N seed {seed_index}: steps {record['env_steps']}, "
                    f"wall {record['wall_time_s']:.1f}s, "
                    f"stoch {record['stochastic']['successes']}/{record['stochastic']['episodes']}, "
                    f"first success at {record['first_successful_eval_steps']}, "
                    f"push {record['push']['successes']}/{record['push']['episodes']}"
                )

        settled_all = [
            value
            for seed_store in eval_stores[amp_index]
            for value in seed_store["settled"]
            if not np.isnan(value)
        ]
        arrivals_all = [
            value
            for seed_store in eval_stores[amp_index]
            for value in seed_store["arrivals"]
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
            "median_first_arrival_s": float(np.median(arrivals_all)) if arrivals_all else None,
            "arrivals": len(arrivals_all),
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
        verdict = not_degrade_verdict(successes, eval_seed_count)
        sweep_entries.append(
            {
                "limit_n": limit_n,
                "limit_norm": limit_norm,
                "sigma": float(limit_norm * SIGMA_FRACTION),
                "training": per_seed_records,
                "deterministic": det_records,
                "stochastic": stochastic_summary,
                "push": push_summary,
                "residual_stats": {
                    "deterministic": residual_stats(np.concatenate(det_residual_pool), limit_norm),
                    "stochastic": residual_stats(np.concatenate(stoch_residual_pool), limit_norm),
                },
                "not_degrade": verdict["not_degrade"],
                "not_degrade_detail": verdict,
            }
        )

    failure_cases = pick_featured_cases(all_eval_episodes, all_push_episodes)
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
        teacher_gate=teacher_gate,
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
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_training_curves(output / "training_curves.png", report, output)
    save_comparison(output / "comparison.png", report, output)
    save_residual_analysis(output / "residual_analysis.png", report, output)
    return report


def pick_featured_cases(eval_episodes, push_episodes):
    """First stochastic-eval failure, then first push failure (fixed order)."""
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
    teacher_gate,
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
    four_way = [
        {
            "label": "基线（第 7 课能量整形+LQR，零样本）",
            "episodes": baseline_summary["episodes"],
            "successes": baseline_summary["successes"],
            "median_settled_at_s": baseline_summary["median_settled_at_s"],
            "source": "this record",
        },
        {
            "label": "朴素残差（第 30 课，单步高斯，已证毁底座）",
            "episodes": LESSON30_NAIVE_RESIDUAL_REFERENCE["episodes"],
            "successes": LESSON30_NAIVE_RESIDUAL_REFERENCE["successes"],
            "median_settled_at_s": None,
            "source": LESSON30_NAIVE_RESIDUAL_REFERENCE["source"],
        },
        {
            "label": "多峰块无底座（第 37 课思路，0/60 × 5 层）",
            "episodes": LESSON37_CHUNK_NO_BASE_REFERENCE["episodes"],
            "successes": LESSON37_CHUNK_NO_BASE_REFERENCE["successes"],
            "median_settled_at_s": None,
            "source": LESSON37_CHUNK_NO_BASE_REFERENCE["source"],
        },
    ]
    hypothesis = []
    for entry in sweep_entries:
        aggregate = entry["stochastic"]
        settled = aggregate["median_settled_at_s"]
        base_settled = baseline_summary["median_settled_at_s"]
        not_below = entry["not_degrade"]
        settled_delta = None if settled is None else settled - base_settled
        if not_below and settled_delta is not None and settled_delta < 0.0:
            verdict = "not degrade + faster settle than the baseline"
        elif not_below:
            verdict = "not degrade (no settle-time improvement over the baseline)"
        else:
            verdict = "degrades the baseline success"
        four_way.append(
            {
                "label": f"组合（本课：底座 + 多峰块残差 a={entry['limit_n']:.0f} N）",
                "episodes": aggregate["episodes"],
                "successes": aggregate["successes"],
                "median_settled_at_s": settled,
                "source": "this record",
            }
        )
        hypothesis.append(
            {
                "limit_n": entry["limit_n"],
                "successes": aggregate["successes"],
                "episodes": aggregate["episodes"],
                "successes_per_seed": aggregate["successes_per_seed"],
                "baseline_successes": baseline_summary["successes"],
                "baseline_episodes": baseline_summary["episodes"],
                "settled_delta_s": settled_delta,
                "threshold_rate": NOT_DEGRADE_RATE,
                "not_degrade": bool(not_below),
                "not_degrade_detail": entry["not_degrade_detail"],
                "verdict": verdict,
            }
        )
    zero_to_one = {
        "first_success_steps_per_seed": {
            f"a={entry['limit_n']:.0f}N": [
                record["first_successful_eval_steps"] for record in entry["training"]
            ]
            for entry in sweep_entries
        },
        "first_arrival_eval_steps_per_seed": {
            f"a={entry['limit_n']:.0f}N": [
                record["first_arrival_eval_steps"] for record in entry["training"]
            ]
            for entry in sweep_entries
        },
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
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "control_law": (
                "u = clip(u_base + clip(u_residual, +/-a), +/-3); u_base is the unchanged "
                "lesson-7 HybridSwingupController (energy shaping + LQR hysteresis) evaluated "
                "on the same pre-step state; u_residual comes from the lesson-37 chunk/multi-"
                "modal policy (K=2 chunk heads + softmax gate), a in {25, 50} N"
            ),
            "residual_limits_n": list(limits_n),
            "guard_limit_n": 0.0,
            "chunk_convention": {
                "experts": config.n_experts,
                "horizon_steps": config.chunk_h,
                "horizon_s": config.chunk_h * design.dt,
                "sampling": (
                    "expert k ~ softmax(gate(s)), then chunk = mu_k + sigma*eps with sigma "
                    "FIXED and UNLEARNED at a/2 (12.5 N at a=25 N, 25 N at a=50 N - a recorded "
                    "response to lesson-30's fixed sigma=1.0, which alone pinned the executed "
                    "residual at the clip limit); the chunk executes OPEN-LOOP for H env "
                    "steps, then replans"
                ),
                "deterministic_plan": (
                    "the gated mixture-mean chunk, planned every H steps (open-loop execution "
                    "like training; per-step replanning is the H=1 degenerate)"
                ),
                "early_failure": (
                    "on a mid-chunk physical failure the fresh episode runs the BASE ALONE "
                    "until the next chunk boundary (the stale chunk is discarded, not leaked)"
                ),
                "ppo_ratio": (
                    "the clipped-surrogate ratio uses the full mixture density "
                    "pi(chunk|s) = sum_k w_k(s) prod_h N(z_h|mu_k[h], sigma^2); the analytic "
                    "posterior gradient is the lesson-37 gated-NLL math at chunk level"
                ),
            },
            "why_ppo": (
                "the lesson-29/30 PPO loop already carries the two-bank curriculum, the "
                "reward-scale convention, GAE with truncation bootstrapping and the residual-"
                "vec machinery (all FD-verified there); SAC would add a replay buffer + "
                "off-policy corrections on top of chunk-level mixture sampling with no "
                "lesson-30 precedent to reuse"
            ),
            "reward": reward.as_dict(),
            "reward_scale_for_learning": config.reward_scale,
            "gamma_note": (
                f"gamma = {GAMMA_STEP}^H = {GAMMA_CHUNK:.4f} per chunk decision (the lesson-29 "
                "per-step discount applied at chunk granularity); within-chunk rewards are "
                "summed undiscounted"
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
                "lesson-7 recovery_metrics reused verbatim: all four wrapped state errors "
                "within tolerances for the final continuous tail >= 2 s; success = recovered "
                "with no physical failure; not-degrade criterion: every seed's success rate "
                ">= 18/20 (the baseline is 20/20)"
            ),
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lessons 29/30/37",
            },
            "literature": {
                "residual_mpc": (
                    "Residual MPC (arXiv 2510.12717): model controller base + RL residual "
                    "beats pure RL and pure MPC - the architecture under test"
                ),
                "residual_rl": (
                    "Johannink et al., ICRA 2019 (arXiv 1812.03355): the residual-RL framing"
                ),
                "lesson30_failure": (
                    "the naive version failed for recorded reasons (single-step Gaussian MSE, "
                    "bang-bang saturation 95.8-99.6%, 177/180 timeout); this lesson changes "
                    "the residual REPRESENTATION (chunked multi-modal) and adds residual "
                    "clipping with a reshaped reward"
                ),
            },
            "seed_streams": {
                "policy_init": "default_rng([master, 7000 + amp, train_seed]); readouts [.., 77]",
                "action_sampling": "default_rng([master, 5000 + amp, train_seed])",
                "minibatch_order": "default_rng([master, 9000 + amp, train_seed])",
                "env_jitter": (
                    "default_rng([base_env_seed, 6000]); base = 10000 + master*1000 "
                    "+ amp*100 + train_seed"
                ),
                "eval_actions": "default_rng([master, 2000, eval_seed])",
                "push_plans": "default_rng([master, 3000]) (identical stream to lessons 29/30)",
            },
        },
        "hyperparameters": {
            **asdict(config),
            "hidden": list(config.hidden),
            "sigma_rule": (
                "the sweep overrides log_std per amplitude: sigma = a/2 (fixed, unlearned); "
                "the log_std field here is the a=25 N default"
            ),
        },
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{eval_seed_count} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
            "per_episode": baseline_records,
        },
        "teacher_verification": teacher_gate,
        "guard": {
            "claim": (
                "with a = 0 the combo pipeline must reproduce the lesson-7 run bitwise: "
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
                    "same exact down start; combo episodes use deterministic chunks"
                ),
                "force_n": PUSH_FORCE_N,
                "duration_s": PUSH_DURATION_S,
                "start_window_s": list(PUSH_START_WINDOW_S),
                "plans": len(plans),
                "paired": "the same plans are applied to the baseline and to every amplitude/seed",
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
        "four_way_comparison": four_way,
        "hypothesis": {
            "claim": (
                "the base+residual combo reaches down-start closed-loop success (>0 where the "
                "pure-learning lineage scored 0) WITHOUT degrading the base (every seed's "
                "success rate >= 18/20 vs the 20/20 baseline); a null or negative outcome is "
                "recorded as the formal conclusion with the residual-behavior analysis"
            ),
            "per_amplitude": hypothesis,
            "zero_to_one": zero_to_one,
        },
        "residual_usage_note": (
            "fraction_at_limit near 1 repeats the lesson-30 bang-bang failure; fraction_below_"
            "half means the learner left most of the budget unused; an all-zero residual means "
            "the policy degenerated to the pure base (which is the guard, not a learned combo)"
        ),
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
            "amplitudes": list(limits_n),
            "env_steps_per_seed": ENV_STEPS_PER_SEED,
            "env_steps_note": (
                f"{config.updates} updates x {config.n_envs} envs x {config.chunks_per_segment} "
                f"chunk decisions x H={config.chunk_h} = {ENV_STEPS_PER_SEED} env steps per seed"
            ),
            "total_env_steps": sum(
                record["env_steps"] for entry in sweep_entries for record in entry["training"]
            ),
            "wall_time_s_total": elapsed,
            "curves_note": "reward_curve_<amp>_<seed> lives in trajectories.npz",
        },
        "lesson30_reference": LESSON30_NAIVE_RESIDUAL_REFERENCE,
        "lesson37_reference": LESSON37_CHUNK_NO_BASE_REFERENCE,
        "limitations": [
            (
                "The residual budget grid {25, 50} N is hand-picked (the two smallest lesson-30 "
                "budgets); budgets above 50 N, budget annealing and state-gated residuals were "
                "not tried."
            ),
            (
                "K=2 and H=8 are single hand choices from the lesson-37 grid; no K/H sweep, no "
                "learned sigma, no gate-temperature or entropy regularization of the gate."
            ),
            (
                "One task (lesson-7 swing-up), one nominal MuJoCo model, no noise/latency/mass "
                "error; no claim about general robot residual-RL practice."
            ),
            ("The cited lesson-30/37 rows come from their official records, not re-run here."),
            (
                "Success requires the strict lesson-7 settled tail; 20 stochastic episodes per "
                "seed carry sampling noise, so differences of a few tenths of a second are "
                "within sampling variation."
            ),
            (
                "The push plans were drawn once (master seed) and shared by every controller; "
                "plan-level pairing is exact but the plan sample itself is finite."
            ),
            (
                "Mid-chunk failures leave the fresh episode on the base alone until the next "
                "chunk boundary (recorded convention); with chunk-aligned episodes this only "
                "affects early physical failures."
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
        "guard_residuals": guard["arrays"]["residuals"],
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
        archive[f"eval_first_arrival_s_{amp_index}"] = np.asarray(
            [v for store in seed_stores for v in store["arrivals"]], dtype=float
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
    """Full archive key set implied by a summary (used by the demo loader)."""
    amps = len(report["sweep"])
    seeds = report["training"]["train_seeds"]
    hidden = tuple(report["hyperparameters"]["hidden"])
    experts = int(report["hyperparameters"]["n_experts"])
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_push_states",
        "baseline_push_lengths",
        "baseline_push_controls",
        "guard_states",
        "guard_controls",
        "guard_residuals",
    }
    for amp in range(amps):
        keys.update(f"reward_curve_{amp}_{seed}" for seed in range(seeds))
        keys.update(
            f"policy_{amp}_{seed}_{name}"
            for seed in range(seeds)
            for name in policy_array_names(hidden, experts)
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
                "first_arrival_s",
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


# ---------------------------------------------------------------------- figures
def load_archive(directory):
    payload = {}
    with np.load(Path(directory) / "trajectories.npz", allow_pickle=False) as npz:
        for key in npz.files:
            payload[key] = npz[key]
    return payload


def save_training_curves(path, report, output):
    configure_plot_font()
    data = load_archive(output)
    seeds = report["training"]["train_seeds"]
    updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = ("#2563eb", "#0f766e", "#b45309", "#7c3aed")
    for index, entry in enumerate(report["sweep"]):
        color = colors[index % len(colors)]
        curves = np.stack([data[f"reward_curve_{index}_{seed}"] for seed in range(seeds)])
        for seed in range(seeds):
            axes[0].plot(updates, curves[seed], alpha=0.25, linewidth=0.8, color=color)
        axes[0].plot(
            updates,
            curves.mean(axis=0),
            color=color,
            linewidth=1.8,
            label=f"a={entry['limit_n']:.0f} N（{seeds} 种子均值）",
        )
        success = np.asarray(
            [
                [int(point["success"]) for point in record["eval_curve"]]
                for record in entry["training"]
            ],
            dtype=float,
        )
        settled = np.asarray(
            [
                [
                    np.nan if point["settled_at_s"] is None else point["settled_at_s"]
                    for point in record["eval_curve"]
                ]
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
            finite = np.isfinite(settled[seed])
            if finite.any():
                axes[1].plot(
                    steps[finite],
                    settled[seed][finite],
                    ":",
                    linewidth=1.0,
                    alpha=0.7,
                    color=color,
                )
        axes[1].plot(
            steps, success.mean(axis=0), "o-", color=color, label=f"a={entry['limit_n']:.0f} N"
        )
    axes[0].set(
        xlabel="PPO 更新轮次",
        ylabel="批内平均原始奖励（每环境步）",
        title="组合训练奖励：细线 = 单个种子",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(
        xlabel="环境步数（×1000）",
        ylabel="下方初态验收通过（1=成功）/ 稳定时刻（s）",
        title="训练中周期确定性评估（实点 = 成功，虚线 = 稳定时刻）",
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
    seed_index = 0
    handoff_start = report["guard"]["capture_time_s"]
    handoff_end = report["guard"]["settled_at_s"]
    data = load_archive(output)
    baseline_states = data["baseline_states"]
    baseline_controls = data["baseline_controls"]
    det_states = data[f"det_states_{featured}_{seed_index}"]
    det_controls = data[f"det_controls_{featured}_{seed_index}"]
    det_residual = data[f"det_residuals_{featured}_{seed_index}"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    if handoff_start is not None and handoff_end is not None:
        for ax in (axes[0, 0], axes[0, 1]):
            ax.axvspan(handoff_start, handoff_end, alpha=0.15, color="#fbbf24")
    ts = np.arange(len(baseline_states)) * dt
    det_ts = np.arange(len(det_states)) * dt
    axes[0, 0].plot(
        ts, np.cos(baseline_states[:, 1] - ref_theta), "--", color="gray", label="基线（第 7 课）"
    )
    axes[0, 0].plot(
        det_ts,
        np.cos(det_states[:, 1] - ref_theta),
        color="#0f766e",
        label=f"组合 a={entry['limit_n']:.0f} N（种子 {seed_index}，确定性块）",
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        title="同一下方初态：基线 vs 组合轨迹（黄带 = 基线交接段）",
    )
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 1].plot(ts, baseline_states[:, 0], "--", color="gray", label="基线")
    axes[0, 1].plot(det_ts, det_states[:, 0], color="#2563eb", label="组合")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].set(ylabel="小车位置（m）", title="小车位置（红点线 = ±2.4 m 失败边界）")
    axes[0, 1].legend(fontsize=8)
    det_edges = np.arange(len(det_controls) + 1) * dt
    axes[1, 0].stairs(baseline_controls * gear, ts, color="gray", label="基线")
    axes[1, 0].stairs(det_controls * gear, det_edges, color="#2563eb", label="组合合计输入")
    residual_ts = np.arange(len(det_residual)) * dt
    axes[1, 0].plot(
        residual_ts, det_residual * gear, color="#b45309", linewidth=1.0, label="其中残差 u_res"
    )
    axes[1, 0].set(
        ylabel="电机力（N）",
        xlabel="仿真时间（s）",
        title="电机输入与残差分量（±300 N 总限幅）",
    )
    axes[1, 0].legend(fontsize=8, loc="upper right")
    rows = report["four_way_comparison"]
    labels = []
    for row in rows:
        if row["label"].startswith("基线"):
            labels.append("基线\n第7课")
        elif row["label"].startswith("朴素残差"):
            labels.append("朴素残差\n第30课")
        elif row["label"].startswith("多峰块"):
            labels.append("块无底座\n第37课")
        else:
            labels.append(row["label"].split("a=")[1].split(" ")[0].replace("N）", "") + " N\n本课")
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    bar_colors = ("#64748b", "#b91c1c", "#7c3aed", "#2563eb", "#0f766e")
    bars = axes[1, 1].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))],
        width=0.6,
    )
    for bar, s, t in zip(bars, successes, totals, strict=True):
        axes[1, 1].annotate(
            f"{s}/{t}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1, 1].set(
        ylabel="验收通过率（%）",
        ylim=(0, 112),
        title="四方对照成功率（第 7 课同口径验收）",
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
    featured = report["featured_amplitude_index"]
    entry = report["sweep"][featured]
    seeds = report["training"]["train_seeds"]
    data = load_archive(output)
    pools = [
        np.concatenate([data[f"det_residuals_{index}_{seed}"] for seed in range(seeds)]) * 100.0
        for index in range(len(report["sweep"]))
    ]
    cases = report["failure_analysis"]["featured_cases"]
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
        xlabel="|残差 u_res|（N，确定性块回合合并种子）",
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
    axes[0, 1].axhline(0.937, color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[0, 1].text(
        len(report["sweep"]) - 0.5,
        0.945,
        "第 30 课朴素残差 93.7%",
        color="#b91c1c",
        fontsize=7,
        ha="right",
    )
    axes[0, 1].set(
        xticks=xs,
        xticklabels=[f"a={item['limit_n']:.0f} N" for item in report["sweep"]],
        ylabel="触限步占比",
        ylim=(0, 1.12),
        title="残差预算使用（|u_res| ≥ 95% 预算的步占比）",
    )
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
    ).reshape(report["training"]["train_seeds"], len(plans))
    for seed in range(seed_recovery.shape[0]):
        axes[1, 0].plot(
            plan_indices,
            seed_recovery[seed],
            "o",
            markersize=4,
            alpha=0.7,
            color=colors[featured % len(colors)],
            label=f"组合 a={entry['limit_n']:.0f} N（种子 {seed}）",
        )
    axes[1, 0].set(
        xlabel="推力方案编号",
        ylabel="推力结束后恢复时间（s）",
        title="±200 N 配对推力恢复（缺口 = 未恢复）",
    )
    axes[1, 0].legend(fontsize=8)
    if cases:
        case = cases[0]
        states = data["case0_states"]
        axes[1, 1].plot(
            np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c"
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
        det_states = data[f"det_states_{featured}_0"]
        axes[1, 1].plot(
            np.arange(len(det_states)) * dt,
            np.cos(det_states[:, 1] - ref_theta),
            color="#0f766e",
        )
        axes[1, 1].axhspan(-1, 0, alpha=0.08, color="orange")
        axes[1, 1].set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="失败案例：无（展示 featured 档确定性轨迹）",
        )
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new result directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=int, default=EVAL_SEEDS)
    parser.add_argument("--limits", type=float, nargs="+", default=list(RESIDUAL_LIMITS_N))
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--chunks-per-segment", type=int, default=CHUNKS_PER_SEGMENT)
    parser.add_argument("--chunk-h", type=int, default=CHUNK_H)
    parser.add_argument("--epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--task-envs", type=int, default=4)
    args = parser.parse_args()
    config = ComboConfig(
        updates=args.updates,
        n_envs=args.n_envs,
        chunks_per_segment=args.chunks_per_segment,
        chunk_h=args.chunk_h,
        epochs=args.epochs,
        minibatch=args.minibatch,
        eval_every=min(args.eval_every, args.updates),
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
                "combo": {
                    f"a={entry['limit_n']:.0f}N": entry["stochastic"]["successes"]
                    for entry in report["sweep"]
                },
                "not_degrade": {
                    f"a={entry['limit_n']:.0f}N": entry["not_degrade"] for entry in report["sweep"]
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
