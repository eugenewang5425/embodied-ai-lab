"""Lesson 37: ACT / diffusion-style minimal probe - chunked multi-expert policy.

Lessons 32/36 established that both offline (teacher-rollout) and online
(student-state) demonstration data are effective at the "arrival" layer
(33/60 and 5/60 upright arrivals) but the deterministic mean path never enters
the upright region and the settled tail never appears: the verdict of lesson 36
was that the remaining suspect is the REPRESENTATION - a single-step MSE
objective over a continuous MLP mean cannot express the teacher's piecewise
hysteretic controller (kick / energy shaping / LQR with a hidden mode), whose
annotations are bimodal at +/-3 (the +/-300 N saturated corrections, 4% to 11%
of the lesson-36 labels).

This lesson is the minimal ACT (Zhao et al., T-RO 2023) / Diffusion-Policy
(Chi et al., RSS 2023) probe: no torch, no Transformer, no denoising.  A numpy
MLP trunk feeds K action-chunk heads (each head outputs a full H-step action
sequence) plus a gating head that weights the experts by state.  The training
objective is the gated maximum likelihood of the teacher's chunk labels,

    L = -log sum_k  w_k(s_t) * prod_h N(a_{t+h} | mu_k[t+h], sigma^2)

with a fixed sigma = 1.0 (the lesson-29 fixed-std convention), computed in one
hand-written backward pass.  The question: does the block/multi-modal
representation let the DETERMINISTIC evaluation path (gated mixture mean, the
direct analog of the lesson-29/32/36 mean-action path) enter the upright
region from the exact resting down start and settle under the verbatim
lesson-7 acceptance - the historical 0 -> 1.

Data: the lesson-36 teacher annotations (8,877 (s, a_teacher) pairs across 3
seeds x 6 rounds x 8 rollouts of the DAgger loop).  Lesson 36 did NOT persist
the raw pairs (only aggregate sizes and label histograms), so this lesson
REPRODUCES them by re-running the lesson-36 DAgger loop with the lesson-36
seed streams (about 25 s wall clock) and validates the reproduction against the
official record's per-round-per-seed dataset sizes (bitwise) and per-round
label statistics (<= 1e-9).  The record's task/acceptance/teacher are reused
verbatim; no online DAgger loop, no environment interaction during training.

Checks (same conditions throughout): the lesson-7 teacher re-run 20/20 with
bitwise identical repeats; the cited lesson-29/32/36 rows; and the new tiers
x 3 seeds: single-step MSE MLP (the lesson-32/36 objective re-trained on this
same dataset), block-output single head (K=1, H=16, chunk MSE), Mixture-of-
Experts K=2 and K=4 (H=16, gated NLL) and K=4 H=8 (horizon control).  Each
tier is evaluated with 20 stochastic episodes, a deterministic mixture episode
(the headline mean path), a deterministic top-1 episode and - for the main
tier - an open-chunk execution episode.  Process metrics: the first deterministic
success checkpoint (the 0 -> 1), the gate distribution along the teacher and
the learned trajectories (did the gate learn a phase switch between the kick
and the LQR phases?), and the deterministic action MSE on the training chunks
(the comparable fit-quality measure across tiers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.dagger_swingup import (
    ANNOTATE_ROLLOUTS,
    BASE_ENV,
    ROUNDS,
    SEED_OFFSET_ACT,
    SEED_OFFSET_BC,
    SEED_OFFSET_INIT,
    SEED_OFFSET_SHUFFLE,
    UPDATES_PER_ROUND,
    W_BC_LEVELS,
    annotate_rollouts,
    bc_weight_at,
    collect_student_rollouts,
    filter_annotation_pairs,
    rollout_to_batch,
    train_dagger_round,
)
from embodied_learning.experiments.dagger_swingup import (
    default_config as dagger_default_config,
)
from embodied_learning.experiments.dapg_swingup import (
    bc_loss_and_gradient,
    deterministic_dapg_episode,
    first_arrival_time_s,
)
from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    EVAL_SEEDS,
    STATE_INPUTS,
    GaussianPolicy,
    MLPTower,
    RewardFunction,
    baseline_evaluations,
    down_start_state,
    episode_metrics,
    episode_rewards,
    evaluate_policy,
    normalize_observation,
    summarize_episodes,
)
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
)

EXPERIMENT = "act_swingup_lesson37"
SCHEMA_VERSION = 1

# --------------------------------------------------------------- protocol knobs
TRUNK_HIDDEN = (64, 64)
CHUNK_H = 16
CHUNK_H_CONTROL = 8
K_EXPERTS = 4
K_EXPERTS_MIN = 2
FIXED_LOG_STD = 0.0  # sigma = exp(0) = 1.0, the lesson-29 fixed-std convention
GEAR = 100.0  # recovery_metrics reports forces as GEAR * normalized command
EPOCHS = 300
MINIBATCH = 512
LEARNING_RATE = 1e-3
EVAL_EVERY_EPOCHS = 25
SEED_OFFSET_DATASET_VERIFY = 9900  # (unused, reserved for potential extensions)
SEED_OFFSET_POLICY = 5000  # the chunk policy init stream (trunk [+ level, seed])
SEED_OFFSET_READOUT = 77
SEED_OFFSET_SHUFFLE_ACT = 9000  # chunk minibatch order
SEED_OFFSET_EVAL_ACT = 2000  # stochastic eval (lesson-29 convention)

# The lesson-36 official record used for dataset reproduction verification
# (project-relative, the CLI always runs from the repository root).
LESSON36_RECORD_DIR = Path("results/dagger_swingup_2026-09-06")

# Tier order: the control (single-step MSE) first, then the block-output and
# the multi-modal tiers.  "mse_single" re-trains the lesson-32/36 objective on
# the same reproduced teacher dataset to keep the comparison internal; the
# cited lesson-29/32/36 rows come from their official records.
TIER_SPECS = (
    {
        "name": "mse_single",
        "label": "单步 MSE（K=1,H=1）",
        "n_experts": 1,
        "horizon": 1,
        "loss": "mse",
    },
    {
        "name": "block_mse",
        "label": "块输出单头（K=1,H=16）",
        "n_experts": 1,
        "horizon": CHUNK_H,
        "loss": "mse",
    },
    {
        "name": "moe_k2",
        "label": "多峰门控 K=2（H=16）",
        "n_experts": K_EXPERTS_MIN,
        "horizon": CHUNK_H,
        "loss": "nll",
    },
    {
        "name": "moe_k4",
        "label": "多峰门控 K=4（H=16）",
        "n_experts": K_EXPERTS,
        "horizon": CHUNK_H,
        "loss": "nll",
    },
    {
        "name": "moe_k4_h8",
        "label": "多峰门控 K=4（H=8）",
        "n_experts": K_EXPERTS,
        "horizon": CHUNK_H_CONTROL,
        "loss": "nll",
    },
)
MAIN_TIER_INDEX = 3  # moe_k4, the headline tier (gets the open-chunk episode)

LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "first_arrival": "never (docs/33 section 3.4, mechanism 1)",
    "deterministic": "never (mean-action episodes report no arrival)",
}
LESSON32_DAPG_REFERENCE = {
    "source": "results/dapg_swingup_2026-09-06 (official lesson-32 record, docs/37)",
    "episodes": 60,
    "successes": 0,
    "first_arrival": "w=10: 33/60 (median 2.12 s); w=1: 27/60 (median 2.04 s)",
    "deterministic": "mean paths reached the upright (2.28/1.52/1.76 s arrivals) but never settled",
}
LESSON36_DAGGER_REFERENCE = {
    "source": "results/dagger_swingup_2026-09-06 (official lesson-36 record, docs/41)",
    "episodes": 60,
    "successes": 0,
    "first_arrival": "5/60 (seed 0, rounds 4-6)",
    "deterministic": "never (all mean-action episodes report no arrival)",
}


@dataclass(frozen=True)
class ACTConfig:
    hidden: tuple[int, ...] = TRUNK_HIDDEN
    lr: float = LEARNING_RATE
    epochs: int = EPOCHS
    minibatch: int = MINIBATCH
    eval_every_epochs: int = EVAL_EVERY_EPOCHS
    log_std: float = FIXED_LOG_STD
    train_seeds: int = 3
    eval_seed_count: int = EVAL_SEEDS

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.lr, self.epochs, self.minibatch)):
            raise ValueError("ACT hyperparameters must be finite and positive")
        if not 1 <= self.eval_every_epochs <= self.epochs:
            raise ValueError("eval_every_epochs must be in [1, epochs]")
        if any(units < 1 for units in self.hidden):
            raise ValueError("hidden layer sizes must be positive")
        if self.minibatch < 1:
            raise ValueError("minibatch must be positive")
        if not 1 <= self.train_seeds <= 8:
            raise ValueError("train_seeds must be in [1, 8]")
        if not 1 <= self.eval_seed_count <= 100:
            raise ValueError("eval_seed_count must be in [1, 100]")


def tier_spec(name):
    for spec in TIER_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise ValueError(f"unknown tier {name!r}; choose from {[t['name'] for t in TIER_SPECS]}")


def log_softmax(values):
    """Numerically stable row-wise log-softmax over the last axis."""
    values = np.asarray(values, dtype=float)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(values):
    return np.exp(log_softmax(values))


# --------------------------------------------------------------- chunk policy
class ChunkMoEPolicy:
    """Shared ReLU trunk; K chunk heads (H steps each) plus a state gating head.

    The deterministic action of the mixture path is the gated mean of the
    experts' first action; the top-1 path is the argmax expert's first action;
    a sampled action draws an expert from the gate distribution and adds
    sigma * white noise.  sigma is a fixed scalar (exp(log_std)), the
    lesson-29 fixed-std convention.
    """

    def __init__(self, n_experts, horizon, hidden, seed, log_std=FIXED_LOG_STD):
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
        self.log_std = np.full(1, float(log_std))

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
        """Returns (means (B,K,H), gate logits (B,K), trunk output h (B,fan))."""
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

    def deterministic_first(self, obs, mode="mixture"):
        """Deterministic first action (scalar) for one normalized observation."""
        means, gate, _ = self.forward(obs)
        weights = softmax(gate)[0]
        if mode == "top1":
            return float(means[0, int(np.argmax(weights)), 0])
        return float(float((weights[:, None] * means[0]).sum(axis=0)[0]))

    def sample_first(self, obs, rng):
        """Gated Gaussian sample (scalar) for one normalized observation."""
        means, gate, _ = self.forward(obs)
        weights = softmax(gate)[0]
        expert = int(rng.choice(self.n_experts, p=weights))
        mean = float(means[0, expert, 0])
        return float(mean + np.exp(self.log_std[0]) * rng.standard_normal())

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


# ----------------------------------------------------------------- the losses
def chunk_mse_loss_and_gradients(policy, obs, targets):
    """Block MSE on the single head (K=1): mean over batch and horizon steps.

    Both the mixture and the top-1 deterministic action coincide for K=1, so
    this tier cleanly separates "block output" from "multi-modality".
    """
    if policy.n_experts != 1:
        raise ValueError("chunk_mse_loss_and_gradients needs a single-head policy")
    means, _gate, h = policy.forward(obs)
    residual = means[:, 0, :] - targets
    loss = float(np.mean(residual * residual))
    grad_mean = (2.0 * residual / (len(obs) * policy.horizon)).reshape(len(obs), 1, policy.horizon)
    grad_expert_w = [h.T @ grad_mean[:, 0, :]]
    grad_expert_b = [grad_mean[:, 0, :].sum(axis=0)]
    grad_gate_w = np.zeros_like(policy.gate_weight)
    grad_gate_b = np.zeros_like(policy.gate_bias)
    grad_h = grad_mean[:, 0, :] @ policy.expert_weights[0].T
    grad_trunk_w, grad_trunk_b = policy.trunk.backward(
        policy.trunk.forward(np.asarray(obs, dtype=float))[1], grad_h
    )
    return loss, [
        *grad_trunk_w,
        *grad_trunk_b,
        *grad_expert_w,
        *grad_expert_b,
        grad_gate_w,
        grad_gate_b,
    ]


def moe_nll_loss_and_gradients(policy, obs, targets, return_grads=True):
    """Gated maximum likelihood over the teacher's chunk labels.

    L = -log sum_k w_k(s) prod_h N(a_{t+h} | mu_k[t+h], sigma^2) with sigma
    fixed.  The per-sample posterior p_k (the soft EM assignment) weights the
    expert gradients; the gate gradient is (w_k - p_k) for the same log-
    marginal.  Returns (loss, gradients) with gradients ordered like
    policy.parameters(); with return_grads=False only the loss is computed
    (finite-difference checks call this on perturbed copies).
    """
    if policy.n_experts < 1:
        raise ValueError("a chunk policy needs at least one expert")
    obs = np.asarray(obs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    means, gate, h = policy.forward(obs)
    batch = len(obs)
    sigma2 = float(np.exp(policy.log_std[0]) ** 2)
    log_w = log_softmax(gate)
    error2 = ((targets[:, None, :] - means) ** 2).sum(axis=2)
    log_e = -0.5 * error2 / sigma2
    log_p = log_w + log_e
    maximum = np.max(log_p, axis=1, keepdims=True)
    denom = np.exp(log_p - maximum).sum(axis=1, keepdims=True)
    posterior = np.exp(log_p - maximum) / denom
    loss = float(np.mean(-(maximum[:, 0] + np.log(denom[:, 0]))))
    if not return_grads:
        return loss, None
    if not np.isfinite(loss) or not np.isfinite(posterior).all():
        raise ValueError("Nonfinite MoE NLL or posterior; training diverged")
    dz = (np.exp(log_w) - posterior) / batch
    dmu = -(posterior[:, :, None] * (targets[:, None, :] - means)) / sigma2 / batch
    grad_expert_w = [h.T @ dmu[:, expert, :] for expert in range(policy.n_experts)]
    grad_expert_b = [dmu[:, expert, :].sum(axis=0) for expert in range(policy.n_experts)]
    grad_gate_w = h.T @ dz
    grad_gate_b = dz.sum(axis=0)
    grad_h = dz @ policy.gate_weight.T
    for expert in range(policy.n_experts):
        grad_h += dmu[:, expert, :] @ policy.expert_weights[expert].T
    grad_trunk_w, grad_trunk_b = policy.trunk.backward(policy.trunk.forward(obs)[1], grad_h)
    return loss, [
        *grad_trunk_w,
        *grad_trunk_b,
        *grad_expert_w,
        *grad_expert_b,
        grad_gate_w,
        grad_gate_b,
    ]


def deterministic_action_mse(policy, obs, targets):
    """MSE of the deterministic first action applied to the whole chunk target.

    The mixture mean is the deterministic action of the MoE tiers; it is the
    direct analog of the lesson-32/36 mean path, so this measure (computed as
    the mean squared error of the mixture-mean action against the teacher
    actions) is comparable ACROSS tiers - the fit-quality control.
    """
    if policy.n_experts == 1:
        means, _gate, _h = policy.forward(obs)
        action = means[:, 0, 0]
        return float(np.mean((action - targets[:, 0]) ** 2))
    means, gate, _h = policy.forward(obs)
    weights = softmax(gate)
    action = (weights * means[:, :, 0]).sum(axis=1)
    return float(np.mean((action - targets[:, 0]) ** 2))


# --------------------------------------------------------- dataset reproduction
def reproduce_teacher_pairs(
    *,
    master_seed=0,
    rounds=ROUNDS,
    rollouts=ANNOTATE_ROLLOUTS,
    updates_per_round=UPDATES_PER_ROUND,
    train_seeds=3,
    log=print,
):
    """Re-run the lesson-36 DAgger loop and return the aggregate teacher labels.

    The lesson-36 record stores only dataset sizes and label histograms, not
    the raw (s, a_teacher) pairs; this function re-runs the lesson-36 loop
    (tier w_BC = 10.0, the only annotating tier) with the lesson-36 seed
    streams and collects the per-episode pairs as they are annotated.

    Returns (per_seed_rounds, totals) with
    per_seed_rounds[seed][round] = [(obs, labels), ...] - the episode pairs of
    that seed and round, in annotation order (the record's aggregation order).
    """
    design = design_swingup_lqr()
    reference = design.controller.reference
    reward = RewardFunction(reference)
    config = dagger_default_config(rounds, updates_per_round, rollouts)
    per_seed_rounds = []
    for seed_index in range(int(train_seeds)):
        init_seed = [master_seed, SEED_OFFSET_INIT + 0, seed_index]
        policy = GaussianPolicy(STATE_INPUTS, config.hidden, init_seed, config.log_std_init)
        value = MLPTower(STATE_INPUTS, config.hidden, 1, [*init_seed, 1])
        parameters = [*policy.parameters(), *value.weights, *value.biases]
        optimizer = AdamOptimizer(parameters, lr=config.lr)
        action_rng = np.random.default_rng([master_seed, SEED_OFFSET_ACT + 0, seed_index])
        mb_rng = np.random.default_rng([master_seed, SEED_OFFSET_SHUFFLE + 0, seed_index])
        bc_rng = np.random.default_rng([master_seed, SEED_OFFSET_BC + 0, seed_index])
        total_updates = rounds * updates_per_round
        envs = [
            make_swingup_environment(max_episode_steps=config.rollout_steps)
            for _ in range(rollouts)
        ]
        for index, env in enumerate(envs):
            env.reset(seed=BASE_ENV + master_seed * 1000 + 0 * 100 + seed_index + index)
        agg_obs = np.empty((0, STATE_INPUTS))
        agg_actions = np.empty(0)
        seed_rounds = []
        for round_index in range(rounds):
            w_here = bc_weight_at(round_index, rounds, W_BC_LEVELS[0], 0.0)
            episodes = collect_student_rollouts(
                envs,
                policy,
                value,
                reward,
                reference,
                master_seed=master_seed,
                level_index=0,
                seed_index=seed_index,
                round_index=round_index,
                horizon=config.rollout_steps,
                action_rng=action_rng,
            )
            batch, _reward_mean = rollout_to_batch(
                episodes,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                reward_scale=config.reward_scale,
            )
            round_pairs = []
            # The DAgger tier (w_BC = 10) keeps annotating in the final round
            # where the per-round weight anneals to 0 (handover; lesson-36
            # protocol: the annotation loop follows the TIER, not the weight).
            pairs, dropped, self_consistent = annotate_rollouts(envs, design, episodes, reference)
            if not self_consistent:
                raise ValueError("teacher re-annotation disagreed; the dataset is not reproducible")
            for obs, labels in pairs:
                obs, labels, extra = filter_annotation_pairs(obs, labels)
                dropped += extra
                round_pairs.append((obs.copy(), labels.copy()))
                agg_obs = np.concatenate([agg_obs, obs], axis=0)
                agg_actions = np.concatenate([agg_actions, labels], axis=0)
            if dropped != 0:
                raise ValueError(f"non-finite teacher labels dropped: {dropped}")
            seed_rounds.append(round_pairs)
            for update_in_round in range(updates_per_round):
                train_dagger_round(
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
        for env in envs:
            env.close()
        if log is not None:
            log(
                f"  dataset seed {seed_index}: sizes "
                f"{[sum(len(l) for rp in seed_rounds[:r] for _o, l in rp) for r in range(1, rounds + 1)]}"
            )
        per_seed_rounds.append(seed_rounds)
    total = int(
        sum(len(l) for seed_rounds in per_seed_rounds for rp in seed_rounds for _o, l in rp)
    )
    return per_seed_rounds, total


def verify_dataset_against_lesson36(
    per_seed_rounds, record_dir, *, rounds=ROUNDS, train_seeds=None, log=print
):
    """Check the reproduced pairs against the lesson-36 official record.

    The strongest available evidence: per-seed per-round dataset sizes must
    match bitwise and per-seed per-round label statistics (mean/std/min/max/
    saturated fraction) must match to 1e-9.  Returns a dict with the check
    status for the summary; the reconstruction is only accepted when every
    check passes.
    """
    record_dir = Path(record_dir)
    if not (record_dir / "summary.json").exists():
        return {
            "passed": False,
            "reason": "lesson-36 record missing; reproduction not verified",
            "sizes_match": False,
            "stats_match": False,
        }
    report = json.loads((record_dir / "summary.json").read_text(encoding="utf-8"))
    tiers = [entry for entry in report["tiers"] if entry["w_bc"] > 0.0]
    if len(tiers) != 1:
        return {
            "passed": False,
            "reason": "lesson-36 tier structure unexpected",
            "sizes_match": False,
            "stats_match": False,
        }
    per_seed_records = tiers[0]["per_seed"]
    if train_seeds is not None and len(per_seed_rounds) != int(train_seeds):
        return {
            "passed": False,
            "reason": "reproduced seed count disagree with the request",
            "sizes_match": False,
            "stats_match": False,
        }
    if train_seeds is not None:
        per_seed_records = per_seed_records[: int(train_seeds)]
    if len(per_seed_records) != len(per_seed_rounds):
        return {
            "passed": False,
            "reason": "seed count disagree with the lesson-36 record",
            "sizes_match": False,
            "stats_match": False,
        }
    sizes_match = True
    stats_match = True
    detail = []
    for seed_index, seed_run in enumerate(per_seed_records):
        produced = per_seed_rounds[seed_index]
        expected_sizes = [record["dataset_size"] for record in seed_run["rounds"]]
        produced_sizes = []
        accumulated = 0
        for round_pairs in produced:
            accumulated += int(sum(len(labels) for _obs, labels in round_pairs))
            produced_sizes.append(accumulated)
        if produced_sizes != expected_sizes:
            sizes_match = False
            detail.append(f"seed {seed_index} sizes {produced_sizes} != {expected_sizes}")
            continue
        for round_index, round_pairs in enumerate(produced):
            record_stats = seed_run["rounds"][round_index]["label_stats"]
            labels = (
                np.concatenate([labels for _obs, labels in round_pairs])
                if round_pairs
                else np.array([])
            )
            mine = {
                "pairs": int(labels.size),
                "mean": float(np.mean(labels)),
                "std": float(np.std(labels)),
                "min": float(np.min(labels)),
                "max": float(np.max(labels)),
                "saturated_fraction": float(np.mean(np.abs(labels) >= CONTROL_LIMIT - 1e-6)),
            }
            for key in ("mean", "std", "min", "max", "saturated_fraction"):
                expected = record_stats.get(key)
                if expected is None or abs(mine[key] - float(expected)) > 1e-9:
                    stats_match = False
                    detail.append(
                        f"seed {seed_index} round {round_index} {key}: {mine[key]} vs {expected}"
                    )
                    break
            if mine["pairs"] != record_stats["pairs"]:
                stats_match = False
                detail.append(f"seed {seed_index} round {round_index} pairs {mine['pairs']}")
    return {
        "passed": bool(sizes_match and stats_match),
        "record": str(record_dir),
        "sizes_match": bool(sizes_match),
        "stats_match": bool(stats_match),
        "details": detail,
    }


def build_chunks(episode_pairs, horizon):
    """Slice (s_t, a_t..a_{t+H-1}) chunks; chunks never cross episode ends.

    Column-exact alignment contract: obs[k] is the observation of the state
    that precedes teacher action k (lesson-7 alignment), so the chunk target
    of obs[t] is labels[t:t+horizon].
    """
    obs_rows, target_rows = [], []
    for obs, labels in episode_pairs:
        if len(labels) != len(obs):
            raise ValueError("observation/label arrays must align within an episode")
        for t in range(len(labels) - horizon + 1):
            obs_rows.append(obs[t])
            target_rows.append(labels[t : t + horizon])
    if not obs_rows:
        raise ValueError("the chunk dataset is empty")
    return np.asarray(obs_rows, dtype=float), np.asarray(target_rows, dtype=float)


def dataset_sha256(obs, targets):
    digest = hashlib.sha256()
    digest.update(obs.tobytes())
    digest.update(targets.tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------- evaluation
def teacher_mode_replay(design, baseline_states):
    """Replay the baseline state sequence and return the teacher's phase per state.

    The controller is deterministic, so replaying the recorded baseline
    trajectory recovers exactly the mode (kick / swingup / balance) it used
    when it generated the trajectory.  Verifies bitwise against the recorded
    controls as a guard.
    """
    model = make_swingup_environment(max_episode_steps=1).unwrapped.model
    controller = HybridSwingupController(model, design)
    modes = []
    actions = []
    for state in np.asarray(baseline_states, dtype=float)[:-1]:
        action = controller.action(state)
        modes.append(controller.mode)
        actions.append(float(action[0]))
    return modes, np.asarray(actions, dtype=float)


def gate_phase_stats(policy, baseline_states, design):
    """Gate weights on the teacher-trajectory states, split by teacher phase.

    Returns (per_phase_mean (3, K) ordered kick/swingup/balance, tv) where tv
    is the total variation between the kick-phase and balance-phase gate
    distributions (0 = the gate does not separate the phases at all).
    """
    modes, _actions = teacher_mode_replay(design, baseline_states)
    _means, gate, _h = policy.forward(
        np.stack(
            [normalize_observation(s, design.controller.reference) for s in baseline_states[:-1]]
        )
    )
    weights = softmax(gate)
    labels = np.asarray(modes)
    rows = []
    for phase in ("kick", "swingup", "balance"):
        pick = labels == phase
        rows.append(
            weights[pick].mean(axis=0)
            if pick.any()
            else np.full(policy.n_experts, 1.0 / policy.n_experts)
        )
    per_phase = np.asarray(rows, dtype=float)
    tv = float(0.5 * np.abs(per_phase[2] - per_phase[0]).sum())
    return per_phase, tv, weights


def gate_entropy(policy, baseline_states, design):
    """Mean gate entropy (nats) on the teacher-trajectory states."""
    _per_phase, _tv, weights = gate_phase_stats(policy, baseline_states, design)
    return float(-np.mean(np.sum(weights * (np.log(weights + 1e-12)), axis=1)))


def run_act_episode(
    policy,
    reward,
    reference,
    dt,
    *,
    mode,
    eval_seed=0,
    rng=None,
    open_chunk=False,
    horizon=EVAL_EPISODE_STEPS,
):
    """One episode from the exact down start under the lesson-7 alignment.

    modes: "mixture" (gated first-action mean, the headline mean path),
    "top1" (argmax expert), "sample" (gated Gaussian, rng mandatory) and
    "open_chunk" (plan the full H-step chunk, execute it open-loop, replan;
    applies to the MoE tiers only).  Returns the episode arrays, the metrics
    record, the gate weights at every decision state and the failure reason.
    """
    if mode == "sample" and rng is None:
        raise ValueError("sampled episodes need an rng")
    env = make_swingup_environment(max_episode_steps=horizon)
    try:
        env.reset(seed=int(eval_seed))
        env.unwrapped.data.qfrc_applied[0] = 0.0
        state = down_start_state(reference)
        env.unwrapped.set_state(state[:2], state[2:])
        states, controls, forces = [state.copy()], [], []
        gates = []
        terminated = truncated = False
        failure_reason = ""
        planned = []  # remaining open-chunk actions
        planned_gate = None
        step = 0
        while step < horizon:
            if planned:
                action = planned.pop(0)
                gate_row = planned_gate
            else:
                obs = normalize_observation(state, reference)[None, :]
                if mode == "sample":
                    action = policy.sample_first(obs, rng)
                    gate_row = policy.gate_weights(obs)[0]
                elif mode == "open_chunk":
                    means, gate, _h = policy.forward(obs)
                    weights = softmax(gate)[0]
                    chunk = (weights[:, None] * means[0]).sum(axis=0)
                    planned = [float(value) for value in chunk]
                    action = planned.pop(0)
                    gate_row = weights
                else:
                    action = policy.deterministic_first(obs, mode)
                    gate_row = policy.gate_weights(obs)[0]
                planned_gate = gate_row
            gates.append(gate_row)
            command = np.array([np.clip(action, -CONTROL_LIMIT, CONTROL_LIMIT)], np.float32)
            env.unwrapped.data.qfrc_applied[0] = 0.0
            state, _raw, terminated, truncated, info = env.step(command)
            states.append(state.copy())
            controls.append(float(command[0]))
            forces.append(0.0)
            failure_reason = info["failure_reason"]
            step += 1
            if terminated or truncated:
                break
        arrays = {
            "states": np.asarray(states, dtype=np.float32),
            "controls": np.asarray(controls, dtype=np.float32),
            "applied_force_n": np.asarray(forces, dtype=np.float32),
            "scheduled_force_n": np.zeros(len(controls)),
            "end_flags": np.array([terminated, truncated]),
        }
        metrics = episode_metrics(arrays, failure_reason, reference, dt)
        record = {
            "mode": mode,
            "recovered": bool(metrics["recovered"]),
            "terminated": bool(metrics["terminated"]),
            "settled_at_s": metrics["settled_at_s"],
            "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
            "failure_reason": failure_reason,
            "return": float(episode_rewards(arrays, reward).sum()),
            "peak_abs_motor_force_n": float(np.max(np.abs(np.asarray(controls))) * GEAR)
            if controls
            else 0.0,
            "max_abs_cart_position_m": float(np.max(np.abs(np.asarray(states)[:, 0])))
            if states
            else 0.0,
        }
        return record, arrays, np.asarray(gates, dtype=float)
    finally:
        env.close()


def evaluate_act_stochastic(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` gated-stochastic episodes from the exact down start."""
    episodes = []
    for eval_seed in range(count):
        rng = np.random.default_rng([master_seed, SEED_OFFSET_EVAL_ACT, eval_seed])
        record, arrays, _gates = run_act_episode(
            policy, reward, reference, dt, mode="sample", eval_seed=eval_seed, rng=rng
        )
        case = {"eval_seed": eval_seed, "arrays": arrays, **record}
        episodes.append(case)
    return episodes


def episode_summary(episodes):
    """The evaluate_policy-shaped summary plus the upright-arrival block."""
    summary = summarize_episodes(episodes)
    arrivals = [ep["first_arrival_s"] for ep in episodes]
    got = [value for value in arrivals if value is not None]
    summary["arrival"] = {
        "episodes_with_arrival": len(got),
        "median_first_arrival_s": float(np.median(got)) if got else None,
    }
    summary["success_per_episode"] = [
        bool(ep["recovered"] and not ep["terminated"]) for ep in episodes
    ]
    summary["first_arrival_s_per_episode"] = arrivals
    summary["settled_at_s_per_episode"] = [
        ep["settled_at_s"] if ep["settled_at_s"] is not None else None for ep in episodes
    ]
    summary["terminated_per_episode"] = [bool(ep["terminated"]) for ep in episodes]
    summary["returns_per_episode"] = [float(ep["return"]) for ep in episodes]
    return summary


# ------------------------------------------------------------------ training
def train_act_tier(
    spec,
    obs,
    targets,
    config,
    *,
    master_seed,
    seed_index,
    teacher_states,
    design,
):
    """Train one chunked policy on the chunk dataset; periodic deterministic eval.

    Returns (policy, history) with history['loss'] (epochs,), checkpoints
    (recovered / settled / first-arrival / processed-chunks / deterministic
    action MSE / gate entropy / phase TV).
    """
    policy = ChunkMoEPolicy(
        spec["n_experts"],
        spec["horizon"],
        config.hidden,
        [master_seed, SEED_OFFSET_POLICY, seed_index],
    )
    parameters = policy.parameters()
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    rng = np.random.default_rng([master_seed, SEED_OFFSET_SHUFFLE_ACT, seed_index])
    epochs = int(config.epochs)
    eval_every = int(config.eval_every_epochs)
    positions = sorted({*range(eval_every, epochs + 1, eval_every), epochs})
    losses = np.empty(epochs)
    ckpt_positions, ckpt_recovered, ckpt_settled = [], [], []
    ckpt_arrival, ckpt_mse, ckpt_entropy, ckpt_tv = [], [], [], []
    reward = RewardFunction(design.controller.reference)
    for epoch in range(epochs):
        order = rng.permutation(len(obs))
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(obs), config.minibatch):
            idx = order[start : start + config.minibatch]
            if spec["loss"] == "mse":
                loss, gradients = chunk_mse_loss_and_gradients(policy, obs[idx], targets[idx])
            else:
                loss, gradients = moe_nll_loss_and_gradients(policy, obs[idx], targets[idx])
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
                raise ValueError("Nonfinite ACT loss or gradients; training diverged")
            optimizer.step(parameters, gradients)
            epoch_loss += loss
            steps += 1
        losses[epoch] = epoch_loss / steps
        if epoch + 1 in positions:
            record, _arrays, _gates = run_act_episode(
                policy, reward, design.controller.reference, design.dt, mode="mixture"
            )
            ckpt_positions.append(epoch + 1)
            ckpt_recovered.append(bool(record["recovered"]))
            ckpt_settled.append(record["settled_at_s"])
            ckpt_arrival.append(record["first_arrival_s"])
            ckpt_mse.append(deterministic_action_mse(policy, obs, targets))
            ckpt_entropy.append(gate_entropy(policy, teacher_states, design))
            ckpt_tv.append(gate_phase_stats(policy, teacher_states, design)[1])
    history = {
        "loss": losses,
        "positions": np.asarray(ckpt_positions, dtype=int),
        "recovered": np.asarray(ckpt_recovered, dtype=bool),
        "settled_s": np.asarray(
            [value if value is not None else np.nan for value in ckpt_settled], dtype=float
        ),
        "arrival_s": np.asarray(
            [value if value is not None else np.nan for value in ckpt_arrival], dtype=float
        ),
        "det_mse": np.asarray(ckpt_mse, dtype=float),
        "gate_entropy": np.asarray(ckpt_entropy, dtype=float),
        "phase_tv": np.asarray(ckpt_tv, dtype=float),
        "processed": np.asarray(ckpt_positions, dtype=int) * len(obs),
    }
    return policy, history


def train_mse_tier(
    obs,
    actions,
    config,
    *,
    master_seed,
    seed_index,
    teacher_states,
    design,
):
    """Single-step MSE MLP (the lesson-32/36 objective) on the pair dataset.

    Uses the lesson-29 GaussianPolicy (mean head + fixed std) and the
    lesson-32 BC loss verbatim, so this tier re-runs the exact objective of
    lessons 32/36 on the same reproduced teacher dataset - the internal
    control that isolates "block/multi-modal" from "the objective changed".
    """
    policy = GaussianPolicy(
        STATE_INPUTS, config.hidden, [master_seed, SEED_OFFSET_POLICY, seed_index], 1.0
    )
    parameters = policy.parameters()
    optimizer = AdamOptimizer(parameters, lr=config.lr)
    rng = np.random.default_rng([master_seed, SEED_OFFSET_SHUFFLE_ACT, seed_index])
    epochs = int(config.epochs)
    eval_every = int(config.eval_every_epochs)
    positions = sorted({*range(eval_every, epochs + 1, eval_every), epochs})
    losses = np.empty(epochs)
    ckpt_positions, ckpt_recovered, ckpt_settled = [], [], []
    ckpt_arrival, ckpt_mse = [], []
    reward = RewardFunction(design.controller.reference)
    for epoch in range(epochs):
        order = rng.permutation(len(obs))
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(obs), config.minibatch):
            idx = order[start : start + config.minibatch]
            mse, gradients = bc_loss_and_gradient(policy, obs[idx], actions[idx])
            if not np.isfinite(mse) or not all(np.isfinite(g).all() for g in gradients):
                raise ValueError("Nonfinite BC loss or gradients; training diverged")
            # The fixed exploration std receives no BC gradient (lesson-32
            # convention: BC matches the mean head only).
            optimizer.step(parameters, [*gradients, np.zeros(1)])
            epoch_loss += mse
            steps += 1
        losses[epoch] = epoch_loss / steps
        if epoch + 1 in positions:
            record, _arrays = deterministic_dapg_episode(
                policy, reward, design.controller.reference, design.dt
            )
            ckpt_positions.append(epoch + 1)
            ckpt_recovered.append(bool(record["recovered"]))
            ckpt_settled.append(record["settled_at_s"])
            ckpt_arrival.append(record["first_arrival_s"])
            ckpt_mse.append(float(mse))
    history = {
        "loss": losses,
        "positions": np.asarray(ckpt_positions, dtype=int),
        "recovered": np.asarray(ckpt_recovered, dtype=bool),
        "settled_s": np.asarray(
            [value if value is not None else np.nan for value in ckpt_settled], dtype=float
        ),
        "arrival_s": np.asarray(
            [value if value is not None else np.nan for value in ckpt_arrival], dtype=float
        ),
        "det_mse": np.asarray(ckpt_mse, dtype=float),
        "gate_entropy": np.full(len(ckpt_positions), np.nan),
        "phase_tv": np.full(len(ckpt_positions), np.nan),
        "processed": np.asarray(ckpt_positions, dtype=int) * len(obs),
    }
    return policy, history


def first_success_checkpoint(history):
    """First periodic deterministic-eval success (epoch or None) - the 0 -> 1."""
    hits = np.flatnonzero(history["recovered"])
    if hits.size == 0:
        return None
    return int(history["positions"][hits[0]])


def first_arrival_checkpoint(history):
    """First periodic deterministic-eval upright arrival (epoch or None)."""
    arrivals = np.flatnonzero(np.isfinite(history["arrival_s"]))
    if arrivals.size == 0:
        return None
    return int(history["positions"][arrivals[0]])


# ---------------------------------------------------------------- experiment
# ---------------------------------------------------------------- experiment
def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    tiers=tuple(spec["name"] for spec in TIER_SPECS),
    train_seeds=None,
    eval_seed_count=None,
    dataset_budget=None,
    log=print,
):
    """Run the lesson-37 probe into `output`; the directory must not exist."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    spec_list = [tier_spec(name) for name in tiers]
    if len({spec["name"] for spec in spec_list}) != len(spec_list):
        raise ValueError("duplicate tiers requested")
    config = config or ACTConfig()
    if train_seeds is not None:
        config = ACTConfig(**{**asdict(config), "train_seeds": int(train_seeds)})
    if eval_seed_count is not None:
        config = ACTConfig(**{**asdict(config), "eval_seed_count": int(eval_seed_count)})
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    reward = RewardFunction(reference)
    teacher_geometry = design

    # Teacher quality gate AND the record's baseline (the lesson-7 re-run);
    # the full-budget run always uses the verbatim lesson-7 20-repeat format.
    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, config.eval_seed_count
    )
    baseline_summary = summarize_episodes(baseline_records)
    teacher_gate = {
        "guard": "the teacher re-run must pass the lesson-7 acceptance on every repeat",
        "episodes": len(baseline_records),
        "successes": baseline_summary["successes"],
        "gate_passed": bool(all(r["recovered"] for r in baseline_records)),
        "median_settled_at_s": baseline_summary["median_settled_at_s"],
    }
    if not teacher_gate["gate_passed"]:
        raise ValueError("teacher quality gate failed: the data source is not verified")

    # The lesson-36 teacher dataset, reproduced and verified against the record.
    budget = dataset_budget or {}
    ds_rounds = int(budget.get("rounds", ROUNDS))
    ds_rollouts = int(budget.get("rollouts", ANNOTATE_ROLLOUTS))
    ds_updates = int(budget.get("updates_per_round", UPDATES_PER_ROUND))
    ds_seeds = int(budget.get("train_seeds", 3))
    full_budget = (ds_rounds, ds_rollouts, ds_updates, ds_seeds) == (
        ROUNDS,
        ANNOTATE_ROLLOUTS,
        UPDATES_PER_ROUND,
        3,
    )
    if log is not None:
        log(
            f"reproducing the lesson-36 teacher label dataset "
            f"({ds_rounds} rounds x {ds_rollouts} rollouts x {ds_seeds} seeds)..."
        )
    per_seed_rounds, total_pairs = reproduce_teacher_pairs(
        master_seed=seed,
        rounds=ds_rounds,
        rollouts=ds_rollouts,
        updates_per_round=ds_updates,
        train_seeds=ds_seeds,
        log=log,
    )
    verification = verify_dataset_against_lesson36(per_seed_rounds, LESSON36_RECORD_DIR)
    if full_budget and not verification["passed"]:
        raise ValueError("dataset reproduction verification failed against the lesson-36 record")
    if not full_budget:
        verification = {
            "passed": None,
            "reason": "micro dataset budget; verification against the lesson-36 record skipped",
        }
    episode_pairs = [pair for seed_rounds in per_seed_rounds for rp in seed_rounds for pair in rp]

    output.mkdir(parents=True, exist_ok=False)

    # Pair dataset for the single-step control tier; H-chunks for the chunked tiers.
    flat_obs = np.concatenate([obs for obs, _labels in episode_pairs], axis=0)
    flat_actions = np.concatenate([labels for _obs, labels in episode_pairs], axis=0)
    dataset_sha = dataset_sha256(flat_obs, flat_actions)

    tier_entries = []
    tier_modes = {}
    tier_loss_curves, tier_ckpts = {}, {}
    archive_policy, archive_det = {}, {}
    featured_cases = []
    for tier_index, spec in enumerate(spec_list):
        per_seed = []
        for seed_index in range(config.train_seeds):
            seed_started = time.perf_counter()
            if spec["loss"] == "mse" and spec["horizon"] == 1:
                policy, history = train_mse_tier(
                    flat_obs,
                    flat_actions,
                    config,
                    master_seed=seed,
                    seed_index=seed_index,
                    teacher_states=baseline_states,
                    design=teacher_geometry,
                )
            else:
                chunk_obs, chunk_targets = build_chunks(episode_pairs, spec["horizon"])
                policy, history = train_act_tier(
                    spec,
                    chunk_obs,
                    chunk_targets,
                    config,
                    master_seed=seed,
                    seed_index=seed_index,
                    teacher_states=baseline_states,
                    design=teacher_geometry,
                )
            tier_loss_curves[(tier_index, seed_index)] = history["loss"]
            tier_ckpts[(tier_index, seed_index)] = history
            archive_policy[(tier_index, seed_index)] = policy

            # Stochastic evaluation under the verbatim lesson-7 acceptance.
            if hasattr(policy, "sample_first"):
                eval_episodes = evaluate_act_stochastic(
                    policy,
                    reward,
                    reference,
                    dt,
                    master_seed=seed,
                    count=config.eval_seed_count,
                )
            else:
                eval_episodes = evaluate_policy(
                    policy, reward, reference, dt, master_seed=seed, count=config.eval_seed_count
                )
            for episode in eval_episodes:
                episode["first_arrival_s"] = first_arrival_time_s(
                    episode["arrays"]["states"], reference, dt
                )
            eval_summary = episode_summary(eval_episodes)

            # Deterministic evaluation (the headline mean path).
            if hasattr(policy, "sample_first"):
                det_episodes = []
                for mode in ("mixture", "top1"):
                    record, arrays, gates = run_act_episode(
                        policy, reward, reference, dt, mode=mode
                    )
                    det_episodes.append({"record": record, "arrays": arrays, "gates": gates})
                if tier_index == MAIN_TIER_INDEX:
                    record, arrays, gates = run_act_episode(
                        policy, reward, reference, dt, mode="open_chunk"
                    )
                    det_episodes.append({"record": record, "arrays": arrays, "gates": gates})
            else:
                record, arrays = deterministic_dapg_episode(policy, reward, reference, dt)
                det_episodes = [{"record": record, "arrays": arrays, "gates": None}]

            # Gate analysis on the teacher trajectory and on the det trajectory.
            if hasattr(policy, "sample_first"):
                per_phase, phase_tv, _gate_all = gate_phase_stats(
                    policy, baseline_states, teacher_geometry
                )
                gate_det = det_episodes[0]["gates"]
            else:
                per_phase = None
                phase_tv = None
                gate_det = None
            tier_det_key = (tier_index, seed_index)
            archive_det[tier_det_key] = {
                "det_episodes": det_episodes,
                "per_phase": per_phase,
                "phase_tv": phase_tv,
                "gate_det": gate_det,
                "eval_summary": eval_summary,
            }
            first_success_epoch = first_success_checkpoint(history)
            per_seed.append(
                {
                    "seed_index": seed_index,
                    "wall_time_s": time.perf_counter() - seed_started,
                    "loss_first": float(history["loss"][0]),
                    "loss_last": float(history["loss"][-1]),
                    "det_mse_first": (
                        float(history["det_mse"][0]) if np.isfinite(history["det_mse"][0]) else None
                    ),
                    "det_mse_last": (
                        float(history["det_mse"][-1])
                        if np.isfinite(history["det_mse"][-1])
                        else None
                    ),
                    "first_success_epoch": first_success_epoch,
                    "first_arrival_epoch": first_arrival_checkpoint(history),
                    "eval": eval_summary,
                    "deterministic": {
                        "mixture": det_episodes[0]["record"],
                        "top1": det_episodes[1]["record"] if len(det_episodes) >= 2 else None,
                        "open_chunk": det_episodes[2]["record"] if len(det_episodes) >= 3 else None,
                    },
                    "gate": {
                        "phase_tv": phase_tv,
                        "per_phase_mean": per_phase.tolist() if per_phase is not None else None,
                    },
                }
            )
            if hasattr(policy, "sample_first"):
                det_modes = ["mixture", "top1"]
                if tier_index == MAIN_TIER_INDEX and spec["n_experts"] > 1:
                    det_modes.append("open_chunk")
            else:
                det_modes = ["mean"]
            tier_modes[(tier_index,)] = det_modes
            failing = [ep for ep in eval_episodes if ep["terminated"] or not ep["recovered"]]
            if failing and len(featured_cases) < 3:
                featured_cases.append(
                    {
                        "kind": f"act_{spec['name']}_final_eval_failure",
                        "tier": spec["name"],
                        "seed_index": seed_index,
                        "eval_seed": failing[0]["eval_seed"],
                        "failure_reason": failing[0]["failure_reason"],
                        "terminated": bool(failing[0]["terminated"]),
                        "recovered": bool(failing[0]["recovered"]),
                        "settled_at_s": failing[0]["settled_at_s"],
                        "max_abs_cart_position_m": float(
                            np.max(np.abs(failing[0]["arrays"]["states"][:, 0]))
                        ),
                        "peak_abs_motor_force_n": float(
                            np.max(np.abs(failing[0]["arrays"]["controls"])) * design.actuator_gear
                        ),
                        "arrays": failing[0]["arrays"],
                    }
                )
            if log is not None:
                log(
                    f"tier {spec['name']} seed {seed_index}: loss "
                    f"{history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}, stochastic ok "
                    f"{eval_summary['successes']}/{config.eval_seed_count}, arrival "
                    f"{eval_summary['arrival']['episodes_with_arrival']}, first det success "
                    f"epoch {first_success_epoch}"
                )
        tier_entries.append({"spec": spec, "per_seed": per_seed})

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        dataset_sha=dataset_sha,
        tier_modes=tier_modes,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        teacher_gate=teacher_gate,
        tier_entries=tier_entries,
        featured_cases=featured_cases,
        total_pairs=total_pairs,
        verification=verification,
        full_budget=full_budget,
        elapsed=elapsed,
    )
    archive = build_archive(
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        tier_loss_curves=tier_loss_curves,
        tier_ckpts=tier_ckpts,
        archive_det=archive_det,
        archive_policy=archive_policy,
        featured_cases=featured_cases,
        spec_list=spec_list,
        train_seeds=config.train_seeds,
        design=design,
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
    save_gate_analysis(output / "gate_analysis.png", report, output)
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
    tier_entries,
    featured_cases,
    total_pairs,
    verification,
    full_budget,
    elapsed,
    dataset_sha=None,
    tier_modes=None,
):
    tier_modes = tier_modes or {}
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
            "median_settled_at_s": None,
            "first_arrival": LESSON29_PPO_REFERENCE["first_arrival"],
            "deterministic": LESSON29_PPO_REFERENCE["deterministic"],
            "source": LESSON29_PPO_REFERENCE["source"],
        },
        {
            "label": "DAPG 离线示教（第 32 课，w=10）",
            "episodes": LESSON32_DAPG_REFERENCE["episodes"],
            "successes": LESSON32_DAPG_REFERENCE["successes"],
            "median_settled_at_s": None,
            "first_arrival": LESSON32_DAPG_REFERENCE["first_arrival"],
            "deterministic": LESSON32_DAPG_REFERENCE["deterministic"],
            "source": LESSON32_DAPG_REFERENCE["source"],
        },
        {
            "label": "DAgger 在线纠错（第 36 课，末轮）",
            "episodes": LESSON36_DAGGER_REFERENCE["episodes"],
            "successes": LESSON36_DAGGER_REFERENCE["successes"],
            "median_settled_at_s": None,
            "first_arrival": LESSON36_DAGGER_REFERENCE["first_arrival"],
            "deterministic": LESSON36_DAGGER_REFERENCE["deterministic"],
            "source": LESSON36_DAGGER_REFERENCE["source"],
        },
    ]
    for entry in tier_entries:
        spec = entry["spec"]
        seeds = entry["per_seed"]
        stochastic_total = int(sum(s["eval"]["episodes"] for s in seeds))
        stochastic_successes = int(sum(s["eval"]["successes"] for s in seeds))
        arrivals = int(sum(s["eval"]["arrival"]["episodes_with_arrival"] for s in seeds))
        settled_all = [
            value
            for s in seeds
            for value in s["eval"]["settled_at_s_per_episode"]
            if value is not None
        ]
        det_successes = int(sum(bool(s["deterministic"]["mixture"]["recovered"]) for s in seeds))
        first_success_epochs = [s["first_success_epoch"] for s in seeds]
        phase_tv = [s["gate"]["phase_tv"] for s in seeds if s["gate"]["phase_tv"] is not None]
        comparison.append(
            {
                "label": f"{spec['label']}（随机 {config.eval_seed_count}/种子）",
                "episodes": stochastic_total,
                "successes": stochastic_successes,
                "median_settled_at_s": float(np.median(settled_all)) if settled_all else None,
                "first_arrival": f"{arrivals}/{stochastic_total}",
                "deterministic": (
                    f"mixture {det_successes}/{len(seeds)}; "
                    "first det success epoch "
                    + (
                        str(min(v for v in first_success_epochs if v is not None))
                        if any(v is not None for v in first_success_epochs)
                        else "never"
                    )
                    + (f"; phase TV {np.mean(phase_tv):.3f}" if phase_tv else "")
                ),
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
            "task": "lesson-7 full-rotation swing-up, exact down start",
            "route": (
                "minimal ACT / diffusion-policy probe (Zhao et al. T-RO 2023; Chi et al. RSS "
                "2023): a numpy trunk gating K action-chunk heads, trained by gated maximum "
                "likelihood on the lesson-36 teacher annotations - an attack on the lesson-36 "
                "verdict that the remaining bottleneck is the REPRESENTATION (a continuous MLP "
                "mean cannot express the teacher's piecewise hysteretic controller)"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "teacher": {
                "controller": (
                    "lesson-7 HybridSwingupController (energy shaping + LQR hysteresis), "
                    "zero-shot; re-run as the quality gate and the record baseline"
                ),
                "acceptance": (
                    "lesson-7 recovery_metrics reused verbatim: all four wrapped state errors "
                    "within tolerances for the final continuous tail >= 2 s"
                ),
            },
            "dataset": {
                "source": (
                    "the lesson-36 teacher annotations: 8,877 (s, a_teacher) pairs across 3 "
                    "seeds x 6 rounds x 8 rollouts of the lesson-36 DAgger loop, tier w_BC = 10"
                ),
                "reproduction": (
                    "lesson 36 did not persist the raw pairs; this lesson re-runs the lesson-36 "
                    "loop with the lesson-36 seed streams and verifies the reproduction against "
                    "the official record: per-seed per-round dataset sizes bitwise and per-round "
                    "label statistics to 1e-9"
                ),
                "verification": verification,
                "chunk_alignment": (
                    "obs[t] is the observation of the state preceding teacher action t; the "
                    "chunk target of obs[t] is labels[t:t+H] - chunks never cross episode ends "
                    "(column-exactness contract, lesson-30/35 lesson)"
                ),
                "total_pairs": total_pairs,
                "dataset_sha256": dataset_sha if dataset_sha is not None else "",
            },
            "policy": {
                "architecture": (
                    "shared 5-64-64 ReLU trunk; K linear chunk heads (H steps each) plus a "
                    "linear gating head; softmax gate weights the experts by state"
                ),
                "experts": "K in {1, 2, 4}; horizons H in {1, 8, 16}",
                "fixed_sigma": (
                    "sigma = 1.0 for both the training NLL and the sampled evaluation - the "
                    "lesson-29 fixed-std convention; no learned per-expert std (recorded choice)"
                ),
                "objective": (
                    "L = -log sum_k w_k(s_t) prod_h N(a_{t+h} | mu_k[t+h], sigma^2) for the "
                    "MoE tiers (gated NLL, soft EM assignment via the posterior); block MSE for "
                    "the single-head tier; entry MSE for the single-step control tier"
                ),
                "deterministic_paths": (
                    "gated mixture mean (headline, the direct analog of the lesson-29/32/36 "
                    "mean path), top-1 expert, and - main tier only - open-chunk execution "
                    "(plan H steps, execute them open-loop, replan); replanning once a step "
                    "otherwise"
                ),
            },
            "tiers": [
                {
                    "name": spec["name"],
                    "label": spec["label"],
                    "n_experts": spec["n_experts"],
                    "horizon": spec["horizon"],
                    "loss": spec["loss"],
                    "det_modes": tier_modes.get((tier_index,), ["mixture", "top1"]),
                }
                for tier_index, spec in enumerate(entry["spec"] for entry in tier_entries)
            ],
            "no_online_prediction": (
                "no DAgger loop and no environment interaction during training: the lesson-36 "
                "verdict said the DATA was effective and the REPRESENTATION was the suspect, so "
                "the only variable here is the policy representation"
            ),
            "upright_region": {"definition": "|alpha| <= 0.3 rad (lesson-7 capture threshold)"},
            "seed_streams": {
                "dataset_replay": "lesson-36 streams verbatim (default_rng([master, 7000/5000/9000/4500 + ..., seed]))",
                "policy_init": "default_rng([master, 5000 + tier, train_seed]); readouts [.., 77]",
                "chunk_shuffle": "default_rng([master, 9000 + tier, train_seed])",
                "stochastic_eval": "default_rng([master, 2000, eval_seed])",
            },
        },
        "hyperparameters": {
            **asdict(config),
            "hidden": list(config.hidden),
            "fixed_sigma": float(np.exp(config.log_std)),
            "minibatch": config.minibatch,
            "epochs": config.epochs,
            "eval_every_epochs": config.eval_every_epochs,
            "tiers": [spec["name"] for spec in (entry["spec"] for entry in tier_entries)],
        },
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{len(baseline_records)} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
            "per_episode": baseline_records,
        },
        "teacher_verification": teacher_gate,
        "tiers": [
            {
                "name": entry["spec"]["name"],
                "label": entry["spec"]["label"],
                "n_experts": entry["spec"]["n_experts"],
                "horizon": entry["spec"]["horizon"],
                "loss": entry["spec"]["loss"],
                "det_modes": tier_modes.get((tier_index,), ["mixture", "top1"]),
                "per_seed": [dict(record) for record in entry["per_seed"]],
                "aggregate": aggregate_tier(entry, config),
            }
            for tier_index, entry in enumerate(tier_entries)
        ],
        "comparison": comparison,
        "lesson29_reference": LESSON29_PPO_REFERENCE,
        "lesson32_reference": LESSON32_DAPG_REFERENCE,
        "lesson36_reference": LESSON36_DAGGER_REFERENCE,
        "hypothesis": {
            "claim": (
                "the lesson-36 attribution: a per-step MSE objective over a continuous MLP "
                "mean cannot express the piecewise hysteretic teacher (+/-300 N bimodal "
                "annotations). Question: do action-chunked multi-expert heads let the "
                "DETERMINISTIC evaluation path (gated mixture mean) enter the upright region "
                "from the exact down start and settle under the verbatim lesson-7 acceptance "
                "- the first historical 0 -> 1?"
            ),
            "verdict": "see tiers[*].aggregate.first_det_success_epoch / stochastic success",
        },
        "failure_analysis": {
            "featured_cases": [
                {k: v for k, v in case.items() if k != "arrays"} for case in featured_cases
            ],
        },
        "training": {
            "train_seeds": config.train_seeds,
            "epochs": config.epochs,
            "chunks_processed_note": (
                "offline supervised training: the headline 0 -> 1 is measured on the periodic "
                "DETERMINISTIC evaluation during training (first success epoch) and on the "
                "final deterministic / stochastic evaluations"
            ),
            "total_pairs": total_pairs,
            "wall_time_s_total": elapsed,
            "full_dataset_budget": full_budget,
        },
        "limitations": [
            (
                "This is a minimal probe, not ACT or Diffusion Policy: no Transformer, no "
                "attention, no denoising; the chunk heads are linear and the gate is a softmax "
                "on the trunk output. The conclusion concerns the block/multi-modal "
                "representation class, not the full algorithms."
            ),
            (
                "The gate conditions on the 5-dim observation only; the teacher's hysteresis "
                "mode is a hidden state, so state-ambiguous annotations cannot be resolved "
                "from the observation alone (the data itself is bimodal at the same state)."
            ),
            (
                "The dataset is the lesson-36 annotation set reproduced bitwise-verifiably; "
                "single teacher, single task, nominal MuJoCo model, no noise/latency/mass error."
            ),
            (
                "sigma is fixed at 1.0 (lesson-29 convention); no learned per-expert std, no "
                "temperature / entropy regularization of the gate in the objective."
            ),
            (
                "The cited lesson-29/32/36 rows come from their official records, not re-run "
                "here; the mse_single tier re-trains the lesson-32/36 objective on this "
                "lesson's dataset as the internal control."
            ),
        ],
    }
    return report


def aggregate_tier(entry, config):
    seeds = entry["per_seed"]
    return {
        "stochastic_total": int(sum(s["eval"]["episodes"] for s in seeds)),
        "stochastic_successes": int(sum(s["eval"]["successes"] for s in seeds)),
        "stochastic_arrivals": int(
            sum(s["eval"]["arrival"]["episodes_with_arrival"] for s in seeds)
        ),
        "det_successes": int(sum(bool(s["deterministic"]["mixture"]["recovered"]) for s in seeds)),
        "det_arrivals": int(
            sum(s["deterministic"]["mixture"]["first_arrival_s"] is not None for s in seeds)
        ),
        "first_det_success_epoch": (
            min(v for v in (s["first_success_epoch"] for s in seeds) if v is not None)
            if any(s["first_success_epoch"] is not None for s in seeds)
            else None
        ),
        "first_det_arrival_epoch": (
            min(v for v in (s["first_arrival_epoch"] for s in seeds) if v is not None)
            if any(s["first_arrival_epoch"] is not None for s in seeds)
            else None
        ),
        "mean_phase_tv": (
            float(
                np.mean([s["gate"]["phase_tv"] for s in seeds if s["gate"]["phase_tv"] is not None])
            )
            if any(s["gate"]["phase_tv"] is not None for s in seeds)
            else None
        ),
    }


# -------------------------------------------------------------------- figures
def load_archive(directory):
    payload = {}
    with np.load(Path(directory) / "trajectories.npz", allow_pickle=False) as npz:
        for key in npz.files:
            payload[key] = npz[key]
    return payload


def tier_index_of(report, name):
    for index, tier in enumerate(report["protocol"]["tiers"]):
        if tier["name"] == name:
            return index
    raise ValueError(name)


def save_training_curves(path, report, output):
    configure_plot_font()
    data = load_archive(output)
    seeds = report["training"]["train_seeds"]
    tiers = report["protocol"]["tiers"]
    colors = plt.get_cmap("tab10").colors
    epochs = report["hyperparameters"]["epochs"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    ax_loss, ax_mse, ax_success, ax_gate = axes.reshape(-1)

    def column_nanmean(values):
        mask = np.isfinite(values)
        counts = mask.sum(axis=0)
        sums = np.nansum(values, axis=0)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    for tier_index, tier in enumerate(tiers):
        color = colors[tier_index % len(colors)]
        per_seed = np.stack([data[f"loss_curve_{tier_index}_{seed}"] for seed in range(seeds)])
        mean = per_seed.mean(axis=0)
        ax_loss.plot(np.arange(1, epochs + 1), mean, color=color, label=tier["label"])
        ax_loss.fill_between(
            np.arange(1, epochs + 1),
            per_seed.min(axis=0),
            per_seed.max(axis=0),
            alpha=0.15,
            color=color,
        )
        ckpts = data[f"ckpt_positions_{tier_index}_0"]
        mse = np.stack([data[f"ckpt_det_mse_{tier_index}_{seed}"] for seed in range(seeds)])
        ax_mse.plot(
            ckpts, mse.mean(axis=0), marker="o", markersize=3, color=color, label=tier["label"]
        )
        for seed in range(seeds):
            ax_mse.plot(
                ckpts,
                data[f"ckpt_det_mse_{tier_index}_{seed}"],
                linestyle="--",
                linewidth=0.7,
                alpha=0.35,
                color=color,
            )
        success = np.stack([data[f"ckpt_recovered_{tier_index}_{seed}"] for seed in range(seeds)])
        ax_success.plot(
            ckpts,
            success.mean(axis=0),
            marker="o",
            markersize=3,
            color=color,
            label=f"{tier['label']}（确定性成功占比）",
        )
        arrival = np.stack([data[f"ckpt_arrival_s_{tier_index}_{seed}"] for seed in range(seeds)])
        if np.isfinite(arrival).any():
            ax_success.plot(
                ckpts,
                column_nanmean(arrival),
                linestyle=":",
                linewidth=1.0,
                color=color,
            )
        entropy = np.stack(
            [data[f"ckpt_gate_entropy_{tier_index}_{seed}"] for seed in range(seeds)]
        )
        tv = np.stack([data[f"ckpt_phase_tv_{tier_index}_{seed}"] for seed in range(seeds)])
        if np.isfinite(entropy).any():
            ax_gate.plot(
                ckpts,
                column_nanmean(entropy),
                marker="o",
                markersize=3,
                color=color,
                label=f"{tier['label']} 门控熵",
            )
        if np.isfinite(tv).any():
            ax_gate.plot(
                ckpts,
                column_nanmean(tv),
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.7,
            )
    ax_loss.set(
        xlabel="训练 epoch", ylabel="目标损失", title="训练目标损失（各层目标不同仅作量级参考）"
    )
    ax_loss.legend(fontsize=6.5)
    ax_loss.grid(alpha=0.2)
    ax_mse.set(
        xlabel="训练 epoch（周期确定性评估）",
        ylabel="确定性动作 MSE（教师动作 - 均值路径第一动作）",
        title="拟合质量（跨层可直接比较）",
    )
    ax_mse.legend(fontsize=6.5)
    ax_mse.grid(alpha=0.2)
    ax_success.set(
        xlabel="训练 epoch（周期确定性评估）",
        ylabel="确定性成功占比 / 首达时刻（s）",
        title="确定性评估：成功占比（实线）与首达时刻（虚线）",
    )
    ax_success.legend(fontsize=6.5)
    ax_success.grid(alpha=0.2)
    ax_gate.set(
        xlabel="训练 epoch（周期确定性评估）",
        ylabel="门控熵（nats）/ 分相TV",
        title="门控分布演化：熵（实线）与 kick<->balance 相位 TV（虚线）",
    )
    ax_gate.legend(fontsize=6.5)
    ax_gate.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_gate_analysis(path, report, output):
    configure_plot_font()
    data = load_archive(output)
    tiers = report["protocol"]["tiers"]
    dt = report["protocol"]["dt_s"]
    main_index = next((i for i, tier in enumerate(tiers) if f"gate_det_{i}_0" in data), 0)
    main = tiers[main_index]
    seed = 0
    teacher_controls = data["baseline_controls"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    ax_teach, ax_det, ax_phase, ax_experts = axes.reshape(-1)
    gate_teacher = data.get(f"gate_teacher_{main_index}_{seed}", None)
    if gate_teacher is None:
        fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "本记录没有带门控的策略层（tiers 均不含多峰/门控读出）",
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    time_teacher = np.arange(len(gate_teacher)) * dt
    ax_teach.stackplot(
        time_teacher,
        (gate_teacher.T if gate_teacher.ndim > 1 else gate_teacher[None, :]),
        labels=[f"专家 {k}" for k in range(main["n_experts"])],
        alpha=0.7,
    )
    ax_teach.set(
        ylabel="门控权重",
        xlabel="仿真时间（s）",
        title=f"门控沿教师轨迹（{main['label']}，种子 0）",
    )
    ax_teach.legend(fontsize=6.5, loc="upper left")
    gate_det = data[f"gate_det_{main_index}_{seed}"]
    time_det = np.arange(len(gate_det)) * dt
    ax_det.stackplot(
        time_det,
        gate_det.T if gate_det.ndim > 1 else gate_det[None, :],
        labels=[f"专家 {k}" for k in range(main["n_experts"])],
        alpha=0.7,
    )
    ax_det.axhspan(-0.02, 1.0, alpha=0.05, color="orange")
    ax_det.set(
        ylabel="门控权重", xlabel="仿真时间（s）", title="门控沿确定性路径（均值路径，种子 0）"
    )
    ax_det.legend(fontsize=6.5, loc="upper left")
    phase = data[f"gate_phase_{main_index}_{seed}"]
    phases = ("kick", "swingup", "balance")
    width = 0.22
    xpos = np.arange(len(phases))
    for expert in range(main["n_experts"]):
        ax_phase.bar(
            xpos + (expert - (main["n_experts"] - 1) / 2) * width,
            phase[:, expert],
            width=width,
            label=f"专家 {expert}",
        )
    ax_phase.set_xticks(xpos, phases)
    ax_phase.set(ylabel="平均门控权重", title="按教师相位分组的门控分布（是否学得分相切换）")
    ax_phase.legend(fontsize=6.5)
    ax_phase.grid(alpha=0.2, axis="y")
    # expert first actions along the teacher trajectory vs the teacher itself
    means_0 = data.get(f"experts_first_{main_index}_{seed}")
    if means_0 is None:
        means_0 = gate_teacher[:, :1] * 0.0
    ax_experts.plot(
        time_teacher,
        teacher_controls,
        color="black",
        linewidth=1.2,
        label="教师动作",
    )
    for expert in range(main["n_experts"]):
        if means_0.shape[1] > expert:
            ax_experts.plot(
                time_teacher,
                means_0[:, expert],
                linewidth=1.0,
                label=f"专家 {expert} mu0(s)",
            )
    ax_experts.set(
        ylabel="动作（-3..3）",
        xlabel="仿真时间（s）",
        title="教师动作 vs 专家第一动作（沿教师轨迹）",
    )
    ax_experts.legend(fontsize=6.5)
    ax_experts.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    data = load_archive(output)
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    rows = report["comparison"]
    labels = []
    for row in rows:
        if row["label"].startswith("基线"):
            labels.append("教师\n第7课")
        elif row["label"].startswith("纯 PPO"):
            labels.append("PPO\n第29课")
        elif row["label"].startswith("DAPG"):
            labels.append("DAPG\n第32课")
        elif row["label"].startswith("DAgger"):
            labels.append("DAgger\n第36课")
        else:
            labels.append(row["label"].split("（")[0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    ax_rate, ax_det, ax_case, ax_arrival = axes.reshape(-1)
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    colors = plt.get_cmap("tab10").colors
    bars = ax_rate.bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=[colors[i % 10] for i in range(len(labels))],
        width=0.62,
    )
    for bar, s, t in zip(bars, successes, totals, strict=True):
        ax_rate.annotate(
            f"{s}/{t}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax_rate.set(
        ylabel="随机评估成功率（%）", ylim=(0, 112), title="成功率对照（第 7 课口径，随机评估）"
    )
    ax_rate.tick_params(axis="x", labelsize=7)
    lines = ["确定性路径（均值路径）头条："]
    tiers = report["tiers"]
    for tier in tiers:
        aggregate = tier["aggregate"]
        lines.append(
            f"  {tier['label']}：确定性成功 {aggregate['det_successes']}/{len(tier['per_seed'])}，"
            f"首成 epoch {aggregate['first_det_success_epoch'] or '从未'}"
        )
    lines.append(
        f"  教师（第 7 课）：{report['baseline']['successes']}/"
        f"{report['baseline']['episodes']}（中位 {report['baseline']['median_settled_at_s']:.2f} s）"
    )
    lines.append(
        "  对照：PPO 29 = 0/60 从未首达；DAPG 32 = 0/60 首达 33/60；DAgger 36 = 0/60 首达 5/60"
    )
    lines.append("  前 36 课：确定性均值路径从未进入直立区（首达全部 None）")
    ax_det.axis("off")
    ax_det.text(
        0.02, 0.96, "\n".join(lines), ha="left", va="top", fontsize=8, transform=ax_det.transAxes
    )
    ax_det.set(title="确定性评估裁决：0→1 是否出现")
    cases = report["failure_analysis"]["featured_cases"]
    if cases:
        case = cases[0]
        states = data["case0_states"]
        ax_case.plot(np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c")
        ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_case.set(
            ylabel="杆端相对高度",
            xlabel="仿真时间（s）",
            title=f"失败案例：{case['tier']}（种子 {case['seed_index']}）",
        )
        ax_case.grid(alpha=0.2)
    else:
        ax_case.text(
            0.5,
            0.5,
            "没有失败回合",
            ha="center",
            va="center",
            transform=ax_case.transAxes,
            fontsize=9,
        )
        ax_case.set(title="失败案例")
    for tier_index, tier in enumerate(tiers):
        color = colors[tier_index % 10]
        arrivals = np.stack(
            [
                data[f"eval_arrival_s_{tier_index}_{seed}"]
                for seed in range(report["training"]["train_seeds"])
            ]
        )
        recovered = np.stack(
            [
                data[f"eval_recovered_{tier_index}_{seed}"]
                for seed in range(report["training"]["train_seeds"])
            ]
        )
        det_ok = np.asarray(
            [bool(s["deterministic"]["mixture"]["recovered"]) for s in tier["per_seed"]]
        )
        xpos = np.arange(len(recovered)) + 1
        ax_arrival.bar(
            xpos - 0.2,
            np.isfinite(arrivals).sum(axis=1),
            width=0.38,
            color=color,
            alpha=0.8,
            label=f"{tier['label']} 随机首达",
        )
        ax_arrival.bar(
            xpos + 0.2,
            det_ok.astype(float),
            width=0.38,
            color=color,
            hatch="//",
            alpha=0.9,
            label=f"{tier['label']} 确定性成功",
        )
    ax_arrival.set(xlabel="训练种子", ylabel="计数", title="每种子：随机首达 vs 确定性成功")
    ax_arrival.legend(fontsize=6)
    ax_arrival.grid(alpha=0.2, axis="y")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiers", type=str, default=",".join(spec["name"] for spec in TIER_SPECS))
    parser.add_argument("--train-seeds", type=int, default=None)
    parser.add_argument("--eval-seed-count", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY_EPOCHS)
    parser.add_argument("--minibatch", type=int, default=MINIBATCH)
    parser.add_argument(
        "--dataset-rounds", type=int, default=ROUNDS, help="micro dataset budget (tests)"
    )
    parser.add_argument("--dataset-rollouts", type=int, default=ANNOTATE_ROLLOUTS)
    parser.add_argument("--dataset-updates", type=int, default=UPDATES_PER_ROUND)
    parser.add_argument("--dataset-seeds", type=int, default=3)
    args = parser.parse_args()
    config = ACTConfig(
        epochs=int(args.epochs),
        eval_every_epochs=int(args.eval_every),
        minibatch=int(args.minibatch),
    )
    if args.train_seeds is not None:
        config = ACTConfig(**{**asdict(config), "train_seeds": int(args.train_seeds)})
    if args.eval_seed_count is not None:
        config = ACTConfig(**{**asdict(config), "eval_seed_count": int(args.eval_seed_count)})
    run_experiment(
        args.output,
        seed=int(args.seed),
        config=config,
        tiers=tuple(name.strip() for name in args.tiers.split(",") if name.strip()),
        dataset_budget={
            "rounds": int(args.dataset_rounds),
            "rollouts": int(args.dataset_rollouts),
            "updates_per_round": int(args.dataset_updates),
            "train_seeds": int(args.dataset_seeds),
        },
    )


def build_archive(
    *,
    baseline_states,
    baseline_controls,
    tier_loss_curves,
    tier_ckpts,
    archive_det,
    archive_policy,
    featured_cases,
    spec_list,
    train_seeds,
    design,
):
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=float),
        "baseline_controls": np.asarray(baseline_controls, dtype=float),
    }
    for tier_index, spec in enumerate(spec_list):
        for seed_index in range(train_seeds):
            key = (tier_index, seed_index)
            history = tier_ckpts[key]
            archive[f"loss_curve_{tier_index}_{seed_index}"] = history["loss"]
            archive[f"ckpt_positions_{tier_index}_{seed_index}"] = history["positions"]
            archive[f"ckpt_recovered_{tier_index}_{seed_index}"] = history["recovered"]
            archive[f"ckpt_settled_s_{tier_index}_{seed_index}"] = history["settled_s"]
            archive[f"ckpt_arrival_s_{tier_index}_{seed_index}"] = history["arrival_s"]
            archive[f"ckpt_det_mse_{tier_index}_{seed_index}"] = history["det_mse"]
            archive[f"ckpt_gate_entropy_{tier_index}_{seed_index}"] = history["gate_entropy"]
            archive[f"ckpt_phase_tv_{tier_index}_{seed_index}"] = history["phase_tv"]
            detail = archive_det[key]
            if detail["det_episodes"][0]["gates"] is not None:
                archive[f"gate_det_{tier_index}_{seed_index}"] = detail["det_episodes"][0]["gates"]
                policy = archive_policy[key]
                per_phase, _tv, gate_teacher = gate_phase_stats(policy, baseline_states, design)
                archive[f"gate_teacher_{tier_index}_{seed_index}"] = gate_teacher
                archive[f"gate_phase_{tier_index}_{seed_index}"] = per_phase
            episodes = detail["det_episodes"]
            for mode_index, episode in enumerate(episodes):
                mode = "mixture" if mode_index == 0 else ("top1" if mode_index == 1 else "open")
                archive[f"det_{mode}_states_{tier_index}_{seed_index}"] = episode["arrays"][
                    "states"
                ]
                archive[f"det_{mode}_controls_{tier_index}_{seed_index}"] = episode["arrays"][
                    "controls"
                ]
            if "eval_summary" in detail:
                summary = detail["eval_summary"]
                archive[f"eval_recovered_{tier_index}_{seed_index}"] = np.asarray(
                    summary["success_per_episode"], dtype=bool
                )
                archive[f"eval_terminated_{tier_index}_{seed_index}"] = np.asarray(
                    summary.get("terminated_per_episode", [False] * summary["episodes"]), dtype=bool
                )
                archive[f"eval_settled_s_{tier_index}_{seed_index}"] = np.asarray(
                    [v if v is not None else np.nan for v in summary["settled_at_s_per_episode"]],
                    dtype=float,
                )
                archive[f"eval_arrival_s_{tier_index}_{seed_index}"] = np.asarray(
                    [
                        v if v is not None else np.nan
                        for v in summary["first_arrival_s_per_episode"]
                    ],
                    dtype=float,
                )
                archive[f"eval_returns_{tier_index}_{seed_index}"] = np.asarray(
                    summary["returns_per_episode"], dtype=float
                )
            policy = archive_policy[key]
            if hasattr(policy, "gate_weights"):
                for name, array in policy.arrays().items():
                    archive[f"policy_{tier_index}_{seed_index}_{name}"] = array
    for index, case in enumerate(featured_cases):
        archive[f"case{index}_states"] = case["arrays"]["states"]
        archive[f"case{index}_controls"] = case["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    tiers = report["protocol"]["tiers"]
    seeds = report["training"]["train_seeds"]
    keys = {"baseline_states", "baseline_controls"}
    for tier_index, tier in enumerate(tiers):
        for seed_index in range(seeds):
            keys.update(
                f"{name}_{tier_index}_{seed_index}"
                for name in (
                    "loss_curve",
                    "ckpt_positions",
                    "ckpt_recovered",
                    "ckpt_settled_s",
                    "ckpt_arrival_s",
                    "ckpt_det_mse",
                    "ckpt_gate_entropy",
                    "ckpt_phase_tv",
                    "det_mixture_states",
                    "det_mixture_controls",
                    "eval_recovered",
                    "eval_terminated",
                    "eval_settled_s",
                    "eval_arrival_s",
                    "eval_returns",
                )
            )
            modes = tier.get("det_modes", ["mixture", "top1"])
            for mode, keyword in (("top1", "top1"), ("open", "open_chunk")):
                if keyword in modes:
                    keys.update(
                        f"det_{mode}_{part}_{tier_index}_{seed_index}"
                        for part in ("states", "controls")
                    )
            if tier["n_experts"] > 1 or tier["name"] == "block_mse":
                keys.add(f"gate_det_{tier_index}_{seed_index}")
                keys.add(f"gate_teacher_{tier_index}_{seed_index}")
                keys.add(f"gate_phase_{tier_index}_{seed_index}")
                keys.update(
                    f"policy_{tier_index}_{seed_index}_{name}"
                    for name in moe_policy_array_names(tier)
                )
    keys.update(
        f"case{index}_{suffix}"
        for index in range(len(report["failure_analysis"]["featured_cases"]))
        for suffix in ("states", "controls")
    )
    return keys


def moe_policy_array_names(tier):
    """Archive key names of one chunked policy payload (K experts, H chunk)."""
    layers = int(tier.get("trunk_layers", 3))
    names = [f"trunk_weight_{i}" for i in range(layers)]
    names += [f"trunk_bias_{i}" for i in range(layers)]
    for expert in range(tier["n_experts"]):
        names += [f"expert_{expert}_weight", f"expert_{expert}_bias"]
    names += ["gate_weight", "gate_bias", "log_std"]
    return tuple(names)


if __name__ == "__main__":
    main()
