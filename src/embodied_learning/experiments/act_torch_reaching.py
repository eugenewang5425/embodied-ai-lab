"""Lesson 42: torch ACT on the 2R arm - can a chunked Transformer break the mm barrier?

Lessons 39/40 refuted "full actuation is enough for pure learning" on two machines:
numpy SAC reached millimetre CLOSEST approaches on the 2R arm (6/60 episodes even
touched inside the 2 mm ball) but never produced the protocol's sustained 0.5 s
hold - 0/60 against the analytic IK+PD baseline's 20/20.  The recorded suspicion:
a single-step continuous MLP MEAN policy cannot express the "rush then freeze"
temporal behaviour that 2 mm x 0.5 s demands.  This lesson - the series' first
torch dependency, explicitly authorised as one last attempt with a stronger
policy representation - tests exactly that suspicion with ACT (Zhao et al.,
T-RO 2023): a small Transformer that maps the last T=4 observations to a chunk
of H=16 future actions, executed T steps at a time before replanning.

Task/environment: byte-for-byte the lesson-40 contract, imported, nothing in any
existing file modified.  Same MuJoCo 2R arm, same start q = (-40, 80) deg, same
torque actions |tau| <= 0.25 N m, same 8-dim observation (protocol verbatim,
NO dq), same goal box, same reward, same evaluation caliber (dist < 2 mm AND
max|dq| < 0.1 rad/s sustained 0.5 s, terminate_on_arrival=False, 20 fixed goals
shared with the baseline), same lesson-8 IK+PD baseline actor, same
run_episode/aggregate_episodes/sustained_window code paths.

Data: the lesson-8 analytic controller (baseline_actor) rolled out from the
lesson-8 start posture for a fixed goal set in the protocol box - the full
swing + arrival + settled tail, the exact behaviour the SAC learner failed to
produce.  Behaviour cloning on teacher (window -> action chunk) pairs; loss =
MSE + 0.1 * L1 on normalized actions; AdamW lr 1e-4 with cosine schedule;
3 seeds x 500 epochs, wall clock < 20 min.

Recorded simplifications (the protocol allows the no-z fallback):
1. NO CVAE latent: the teacher is a deterministic controller, so the training
   distribution is unimodal - the multi-modality argument for the z-latent does
   not apply here; the tested ingredients are chunking + temporal context.
2. NO temporal ensembling at inference: the first T steps of each fresh chunk
   are executed, then the policy replans (the protocol's own wording).

Comparison rows: the analytic IK+PD baseline (rerun here, same arena) versus
the official lesson-40 SAC record (CITED, its npz showcase trajectories embedded
with sha256 provenance) versus this lesson's ACT seeds.
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
import torch
from torch import nn

from embodied_learning.experiments.arm_reaching import audit_geometry
from embodied_learning.experiments.rl_arm_reaching import (
    ACTION_DIM,
    ARRIVAL_RADIUS_M,
    CRITERION_RATE,
    DT,
    GEOMETRY_AUDIT_LIMIT_M,
    MAX_EPISODE_STEPS,
    OBS_DIM,
    SHOWCASE_GOALS,
    TORQUE_LIMIT_NM,
    ArmReachingEnv,
    aggregate_episodes,
    baseline_actor,
    make_eval_goals,
    observation_for,
    run_episode,
)
from embodied_learning.experiments.rl_arm_reaching import (
    _episode_arrays as episode_arrays,  # same columnar schema as the lesson-40 archive
)
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "act_torch_reaching_lesson42"
SCHEMA_VERSION = 1

TEACHER_EPISODES = 36
EPOCHS = 500
WINDOWS_PER_EPOCH = 2048
BATCH_SIZE = 512
CHUNK_T = 4  # observation history length (protocol)
CHUNK_H = 16  # action chunk length (protocol)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2  # encoder and decoder depth (protocol: 2 layers, 4 heads, d_model 128)
FFN = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
W_L1 = 0.1  # loss = MSE + W_L1 * L1 on normalized actions
EVAL_EVERY_EPOCHS = 100
EVAL_SUBSET = 5  # periodic-eval goals during training: showcase A/B + first sampled
TRAIN_SEEDS = 3
VAL_STRIDE = 8  # every 8th window is validation (deterministic split)
ATTENTION_PLAN_INDEX = 30  # capture attention at the 30th replan (mid approach)

DEFAULT_RESULTS = "results/act_torch_reaching_2026-09-06"
SAC_REFERENCE_DEFAULT = "results/rl_arm_reaching_2026-09-06"
SAC_EXPERIMENT = "rl_arm_reaching_lesson40"
SAC_MAX_SEEDS = 3
SAC_GOALS_EMBEDDED = 2  # showcase goals A/B tip trajectories for the demo


@dataclass(frozen=True)
class ACTConfig:
    teacher_episodes: int = TEACHER_EPISODES
    episode_steps: int = MAX_EPISODE_STEPS
    epochs: int = EPOCHS
    windows_per_epoch: int = WINDOWS_PER_EPOCH
    batch_size: int = BATCH_SIZE
    chunk_t: int = CHUNK_T
    chunk_h: int = CHUNK_H
    d_model: int = D_MODEL
    n_heads: int = N_HEADS
    n_layers: int = N_LAYERS
    ffn: int = FFN
    lr: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    w_l1: float = W_L1
    eval_every_epochs: int = EVAL_EVERY_EPOCHS
    eval_goal_count: int = 20
    eval_subset: int = EVAL_SUBSET

    def __post_init__(self):
        integers = (
            self.teacher_episodes,
            self.episode_steps,
            self.epochs,
            self.windows_per_epoch,
            self.batch_size,
            self.chunk_t,
            self.chunk_h,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.ffn,
            self.eval_every_epochs,
            self.eval_goal_count,
            self.eval_subset,
        )
        if not all(type(v) is int and v > 0 for v in integers):
            raise ValueError("ACT hyperparameters must be positive integers")
        if not all(np.isfinite(v) and v > 0 for v in (self.lr,)):
            raise ValueError("learning rate must be finite and positive")
        if self.weight_decay < 0.0 or self.w_l1 < 0.0:
            raise ValueError("weight_decay and w_l1 must be non-negative")
        if self.chunk_t > self.chunk_h:
            raise ValueError("chunk_t cannot exceed chunk_h")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.eval_every_epochs > self.epochs:
            raise ValueError("eval_every_epochs cannot exceed epochs")
        if self.eval_goal_count < len(SHOWCASE_GOALS) + 2:
            raise ValueError("eval_goal_count must leave room for sampled goals")
        if not 1 <= self.eval_subset <= self.eval_goal_count:
            raise ValueError("eval_subset must be within [1, eval_goal_count]")


def default_training_config():
    """The pre-registered configuration."""
    return ACTConfig()


# --------------------------------------------------------------- torch policy
class ActTransformer(nn.Module):
    """Tiny ACT: obs window (B, T, 8) -> tanh action chunk (B, H, 2).

    Learned positional embedding on the T observation tokens; a pre-LN
    Transformer encoder (n_layers, n_heads, d_model); H learned chunk queries
    decoded by a Transformer decoder with cross-attention over the T memory
    tokens; a GELU MLP head squashed by tanh to the normalized action range.
    Dropout is zero and every module is deterministic on CPU.
    """

    def __init__(
        self,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ffn=FFN,
        chunk_t=CHUNK_T,
        chunk_h=CHUNK_H,
    ):
        super().__init__()
        self.chunk_t = int(chunk_t)
        self.chunk_h = int(chunk_h)
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, chunk_t, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.queries = nn.Parameter(torch.zeros(1, chunk_h, d_model))
        nn.init.normal_(self.queries, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            n_heads,
            ffn,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def embed(self, obs):
        return self.input_proj(obs) + self.pos_emb[:, : obs.shape[1]]

    def forward(self, obs):
        memory = self.encoder(self.embed(obs))
        queries = self.queries.expand(obs.shape[0], -1, -1)
        decoded = self.head_norm(self.decoder(queries, memory))
        return torch.tanh(self.head(decoded))

    @torch.no_grad()
    def attention_maps(self, obs):
        """Head-averaged encoder self-attention (T, T) and decoder cross-attention (H, T).

        torch 2.x layer calls return no weights, so the LAST layer's inner
        attention modules are queried directly on the same normalized inputs
        the forward pass feeds them (pre-LN replicas; recorded for honesty).
        """
        was_training = self.training
        self.eval()
        h = self.embed(obs)
        for layer in self.encoder.layers[:-1]:
            h = layer(h)
        last_encoder_layer = self.encoder.layers[-1]
        normalized = last_encoder_layer.norm1(h)
        _out, enc_weights = last_encoder_layer.self_attn(
            normalized, normalized, normalized, need_weights=True, average_attn_weights=True
        )
        memory = self.encoder.norm(h)
        queries = self.queries.expand(obs.shape[0], -1, -1)
        for layer in self.decoder.layers[:-1]:
            queries = layer(queries, memory)
        last_decoder_layer = self.decoder.layers[-1]
        normalized_queries = last_decoder_layer.norm2(queries)
        _out, cross_weights = last_decoder_layer.multihead_attn(
            normalized_queries, memory, memory, need_weights=True, average_attn_weights=True
        )
        if was_training:
            self.train()
        return enc_weights[0].cpu().numpy(), cross_weights[0].cpu().numpy()


def bc_loss(pred, target, w_l1):
    """L = MSE + w_l1 * L1 on normalized action chunks; returns (total, mse, l1)."""
    diff = pred - target
    mse = (diff**2).mean()
    l1 = diff.abs().mean()
    return mse + w_l1 * l1, mse, l1


# -------------------------------------------------------------------- teacher
def build_teacher_dataset(seed, config):
    """Lesson-8 IK+PD teacher rollouts: full swing + arrival + settled tail.

    The eval-style env (terminate_on_arrival=False) keeps every episode at the
    full horizon so the holding behaviour after arrival is in the data - the
    exact behaviour the SAC learners of lessons 39/40 never produced.
    """
    env = ArmReachingEnv(
        [seed, 21000], episode_steps=config.episode_steps, terminate_on_arrival=False
    )
    observations, actions, goals = [], [], []
    for _ in range(config.teacher_episodes):
        env.reset()
        goals.append(env.goal.copy())
        obs_list = [observation_for(env.state[:2], env.last_torque, env.goal)]
        act_list = []
        for _ in range(env.episode_steps):
            torque = baseline_actor(env.state, env.goal, env.last_torque)
            act_list.append(np.asarray(torque, dtype=float) / TORQUE_LIMIT_NM)
            env.step(torque)
            obs_list.append(observation_for(env.state[:2], env.last_torque, env.goal))
        observations.append(np.asarray(obs_list, dtype=float))
        actions.append(np.asarray(act_list, dtype=float))
    return observations, actions, goals


def dataset_digest(observations, actions):
    """SHA-256 over the raw float64 teacher arrays (episode order kept)."""
    hasher = hashlib.sha256()
    for arr in (*observations, *actions):
        hasher.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def build_windows(observations, actions, chunk_t, chunk_h):
    """Every (T-obs window ending at t, H-action chunk starting at t) pair.

    Left edge: the window is padded with the episode's first observation; right
    edge: the chunk is padded by repeating the teacher's final (holding) action.
    Returns float32 (N, T, 8) and (N, H, 2) arrays; window t aligns with the
    teacher's own information state at step t (only past observations).
    """
    obs_windows, act_windows = [], []
    for obs, act in zip(observations, actions, strict=True):
        length = act.shape[0]
        starts = np.arange(length)
        obs_idx = np.clip(starts[:, None] + np.arange(-(chunk_t - 1), 1), 0, None)
        act_idx = np.clip(starts[:, None] + np.arange(chunk_h), 0, length - 1)
        obs_windows.append(obs[obs_idx])
        act_windows.append(act[act_idx])
    return (
        np.concatenate(obs_windows).astype(np.float32),
        np.concatenate(act_windows).astype(np.float32),
    )


# ---------------------------------------------------------------- evaluation
class ActActor:
    """Chunked closed-loop actor: replan every chunk_t steps, execute the chunk.

    The observation history is rebuilt from the same observation_for contract
    the env reports; the initial window is padded with the first observation.
    The policy predicts a in (-1, 1)^2; the actor returns tau = 0.25 * a.
    """

    def __init__(self, policy, chunk_t, *, capture_plan=None):
        self.policy = policy
        self.chunk_t = int(chunk_t)
        self.capture_plan = capture_plan
        self.captured_window = None
        self.plans = 0
        self._history = None
        self._chunk = None
        self._cursor = 0

    def reset(self):
        self.captured_window = None
        self.plans = 0
        self._history = None
        self._chunk = None
        self._cursor = 0

    def _plan(self, window):
        if self.capture_plan is not None and self.plans == self.capture_plan:
            self.captured_window = window.copy()
            self.capture_plan = None  # capture once, at the attention goal's approach
        with torch.no_grad():
            chunk = self.policy(torch.from_numpy(window[None, :, :].astype(np.float32)))
        self._chunk = chunk[0].cpu().numpy().astype(float)
        self._cursor = 0
        self.plans += 1

    def __call__(self, state, goal, prev_torque):
        obs = observation_for(state[:2], prev_torque, goal)
        if self._history is None:
            self._history = np.tile(obs, (self.chunk_t, 1))
        else:
            self._history = np.concatenate([self._history[1:], obs[None, :]], axis=0)
        if self._chunk is None or self._cursor >= self.chunk_t:
            self._plan(self._history)
        action = self._chunk[self._cursor]
        self._cursor += 1
        return TORQUE_LIMIT_NM * np.clip(action, -1.0, 1.0)


def evaluate_act(policy, env, goals, config, *, record=False, capture_plan=None):
    """Same-caliber evaluation as lesson 40: run_episode + sustained window."""
    policy.eval()
    actor = ActActor(policy, config.chunk_t, capture_plan=capture_plan)
    episodes, truths = [], []
    for goal in goals:
        actor.reset()
        summary, points = run_episode(actor, env, goal, record=record)
        episodes.append(summary)
        if record:
            truths.append(points)
    return episodes, truths, actor


# ------------------------------------------------------------------- training
def train_act(
    obs_windows,
    act_windows,
    train_indices,
    val_indices,
    eval_env,
    eval_goals,
    config,
    *,
    master_seed,
    seed_index,
    log=None,
):
    """One ACT training run (one seed); fixed seed streams, bitwise reproducible."""
    torch.manual_seed(
        int(np.random.default_rng([master_seed, 22000, seed_index]).integers(0, 2**31 - 1))
    )
    policy = ActTransformer(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        ffn=config.ffn,
        chunk_t=config.chunk_t,
        chunk_h=config.chunk_h,
    )
    param_count = sum(p.numel() for p in policy.parameters())
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    shuffle_rng = np.random.default_rng([master_seed, 23000, seed_index])
    obs_tensor = torch.from_numpy(obs_windows)
    act_tensor = torch.from_numpy(act_windows)

    def run_batches(indices):
        order = shuffle_rng.permutation(len(indices))[: config.windows_per_epoch]
        totals, mses, l1s = [], [], []
        for start in range(0, len(order), config.batch_size):
            selected = torch.from_numpy(indices[order[start : start + config.batch_size]])
            pred = policy(obs_tensor[selected])
            total, mse, l1 = bc_loss(pred, act_tensor[selected], config.w_l1)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            totals.append(float(total.detach()))
            mses.append(float(mse.detach()))
            l1s.append(float(l1.detach()))
        return totals, mses, l1s

    @torch.no_grad()
    def run_val():
        totals, mses, l1s = [], [], []
        for start in range(0, len(val_indices), config.batch_size):
            selected = torch.from_numpy(val_indices[start : start + config.batch_size])
            pred = policy(obs_tensor[selected])
            total, mse, l1 = bc_loss(pred, act_tensor[selected], config.w_l1)
            totals.append(float(total))
            mses.append(float(mse))
            l1s.append(float(l1))
        return float(np.mean(totals)), float(np.mean(mses)), float(np.mean(l1s))

    def periodic_eval(epoch):
        subset = eval_goals[: config.eval_subset]
        episodes, _truths, _actor = evaluate_act(policy, eval_env, subset, config)
        return {
            "epoch": epoch,
            "successes": sum(episode["outcome"] == "arrived" for episode in episodes),
            "episodes": len(episodes),
            "median_min_distance_m": float(
                np.median([episode["min_distance_m"] for episode in episodes])
            ),
        }

    train_curve, val_curve = [], []
    eval_curve = []
    started = time.perf_counter()
    policy.train()
    initial_val = run_val()
    for epoch in range(1, config.epochs + 1):
        totals, _mses, _l1s = run_batches(train_indices)
        train_curve.append(float(np.mean(totals)))
        val_total, val_mse, val_l1 = run_val()
        val_curve.append({"total": val_total, "mse": val_mse, "l1": val_l1})
        scheduler.step()
        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs:
            point = periodic_eval(epoch)
            eval_curve.append(point)
            if log is not None:
                log(
                    f"  seed {seed_index}: epoch {epoch}, train loss {train_curve[-1]:.5f}, "
                    f"val total {val_total:.5f}, eval {point['successes']}/{point['episodes']}"
                )
    wall = time.perf_counter() - started
    final_val_total, final_val_mse, final_val_l1 = (
        val_curve[-1]["total"],
        val_curve[-1]["mse"],
        val_curve[-1]["l1"],
    )
    return {
        "policy": policy,
        "param_count": param_count,
        "wall_time_s": wall,
        "initial_val_total": initial_val[0],
        "train_loss_curve": np.asarray(train_curve, dtype=float),
        "val_total_curve": np.asarray([point["total"] for point in val_curve]),
        "val_mse_curve": np.asarray([point["mse"] for point in val_curve]),
        "val_l1_curve": np.asarray([point["l1"] for point in val_curve]),
        "eval_curve": eval_curve,
        "final_train_loss": float(train_curve[-1]),
        "final_val_total": float(final_val_total),
        "final_val_rmse_mnm": float(np.sqrt(final_val_mse) * TORQUE_LIMIT_NM * 1000),
        "final_val_l1_mnm": float(final_val_l1 * TORQUE_LIMIT_NM * 1000),
    }


# ----------------------------------------------------------------- experiment
def load_sac_reference(path, eval_goals, *, max_seeds=SAC_MAX_SEEDS):
    """Cite the official lesson-40 record; embed its showcase tip trajectories.

    Returns (metadata, arrays).  The reference is accepted when this record's
    eval goals are a PREFIX of the lesson-40 goals (both come from the same
    deterministic make_eval_goals stream, so a shorter record samples the same
    first goals) or equal to them; anything else is refused.
    """
    directory = Path(path)
    base = {
        "loaded": False,
        "source": str(directory),
        "experiment": SAC_EXPERIMENT,
    }
    summary_path, npz_path = directory / "summary.json", directory / "trajectories.npz"
    if not summary_path.is_file() or not npz_path.is_file():
        return {**base, "reason": "lesson-40 record not found"}, {}
    sac_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if sac_summary.get("experiment") != SAC_EXPERIMENT:
        return {**base, "reason": "unexpected experiment id"}, {}
    with np.load(npz_path, allow_pickle=False) as npz:
        their_goals = np.asarray(npz["eval_goals"], dtype=float)
        prefix = len(their_goals) >= len(eval_goals) and np.allclose(
            their_goals[: len(eval_goals)], eval_goals, atol=1e-12
        )
        if not prefix:
            return {**base, "reason": "eval goals differ from the lesson-40 record"}, {}
        seeds = min(int(sac_summary["training"]["train_seeds"]), max_seeds)
        arrays = {}
        for seed_index in range(seeds):
            arrays[f"sac_outcome_{seed_index}"] = npz[f"eval_outcome_{seed_index}"].copy()
            arrays[f"sac_arrival_time_{seed_index}"] = npz[f"eval_arrival_time_{seed_index}"].copy()
            arrays[f"sac_min_distance_{seed_index}"] = npz[f"eval_min_distance_{seed_index}"].copy()
            for goal_index in range(SAC_GOALS_EMBEDDED):
                arrays[f"sac_truth_{seed_index}_{goal_index}"] = npz[
                    f"eval_truth_{seed_index}_{goal_index}"
                ].copy()
    across = sac_summary["rl_evaluation"]["across_seeds"]
    metadata = {
        "loaded": True,
        "source": str(directory),
        "experiment": SAC_EXPERIMENT,
        "trajectories_sha256": sac_summary.get("trajectories_sha256"),
        "seeds": seeds,
        "goals": SAC_GOALS_EMBEDDED,
        "training": sac_summary["training"],
        "across_seeds": across,
        "per_seed": [
            {
                "seed_index": record["seed_index"],
                "aggregate": record["aggregate"],
            }
            for record in sac_summary["rl_evaluation"]["per_seed"][:seeds]
        ],
    }
    return metadata, arrays


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(
    output,
    *,
    seed=0,
    config=None,
    train_seeds=TRAIN_SEEDS,
    sac_reference=SAC_REFERENCE_DEFAULT,
    log=print,
):
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
    if baseline_aggregate["successes"] != baseline_aggregate["episodes"]:
        raise ValueError("Analytic baseline gate failed: the arena is not solvable by IK+PD")
    if log is not None:
        log(
            f"analytic baseline (lesson-8 IK + PD, positive elbow): "
            f"{baseline_aggregate['successes']}/{baseline_aggregate['episodes']} successes"
        )

    sac_metadata, sac_arrays = load_sac_reference(sac_reference, eval_goals)
    if log is not None:
        if sac_metadata["loaded"]:
            across = sac_metadata["across_seeds"]
            log(
                f"cited lesson-40 SAC record ({sac_metadata['source']}): "
                f"{across['successes']}/{across['episodes']} successes"
            )
        else:
            log(f"lesson-40 SAC citation unavailable: {sac_metadata.get('reason')}")

    if log is not None:
        log(f"building teacher dataset ({config.teacher_episodes} IK+PD rollouts)")
    teacher_started = time.perf_counter()
    observations, actions, _teacher_goals = build_teacher_dataset(seed, config)
    teacher_wall = time.perf_counter() - teacher_started
    teacher_digest = dataset_digest(observations, actions)
    obs_windows, act_windows = build_windows(observations, actions, config.chunk_t, config.chunk_h)
    val_mask = (np.arange(len(obs_windows)) % VAL_STRIDE) == 0
    train_indices = np.flatnonzero(~val_mask).astype(np.int64)
    val_indices = np.flatnonzero(val_mask).astype(np.int64)
    if log is not None:
        log(
            f"teacher digest {teacher_digest[:16]}..., windows {len(obs_windows)} "
            f"(train {len(train_indices)} / val {len(val_indices)})"
        )

    output.mkdir(parents=True, exist_ok=False)
    per_seed_records = []
    curve_stores = {}
    eval_stores = {}
    attention_stores = {}
    for seed_index in range(train_seeds):
        result = train_act(
            obs_windows,
            act_windows,
            train_indices,
            val_indices,
            eval_env,
            eval_goals,
            config,
            master_seed=seed,
            seed_index=seed_index,
            log=log,
        )
        policy = result["policy"]
        curve_stores[f"train_loss_curve_{seed_index}"] = result["train_loss_curve"]
        curve_stores[f"val_total_curve_{seed_index}"] = result["val_total_curve"]
        curve_stores[f"val_mse_curve_{seed_index}"] = result["val_mse_curve"]
        curve_stores[f"val_l1_curve_{seed_index}"] = result["val_l1_curve"]
        curve_stores[f"eval_curve_epochs_{seed_index}"] = np.asarray(
            [point["epoch"] for point in result["eval_curve"]], dtype=int
        )
        curve_stores[f"eval_curve_successes_{seed_index}"] = np.asarray(
            [point["successes"] for point in result["eval_curve"]], dtype=int
        )
        curve_stores[f"eval_curve_min_distance_{seed_index}"] = np.asarray(
            [point["median_min_distance_m"] for point in result["eval_curve"]], dtype=float
        )

        episodes, truths, actor = evaluate_act(
            policy, eval_env, eval_goals, config, record=True, capture_plan=ATTENTION_PLAN_INDEX
        )
        window = actor.captured_window
        if window is None:  # short horizons in smoke runs may never reach the index
            window = np.tile(
                observation_for(eval_env.state[:2], eval_env.last_torque, eval_goals[0]),
                (config.chunk_t, 1),
            )
        enc_attn, cross_attn = policy.attention_maps(
            torch.from_numpy(window[None, :, :].astype(np.float32))
        )
        attention_stores[f"attention_encoder_{seed_index}"] = enc_attn.astype(float)
        attention_stores[f"attention_decoder_{seed_index}"] = cross_attn.astype(float)

        aggregate = aggregate_episodes(episodes)
        outcomes = [episode["outcome"] for episode in episodes]
        eval_stores[seed_index] = {
            "truths": truths,
            "outcomes": outcomes,
            "episodes": episodes,
        }
        first_success = first_checkpoint(result["eval_curve"], lambda p: p["successes"] > 0)
        first_criterion = first_checkpoint(
            result["eval_curve"],
            lambda p: p["successes"] >= math.ceil(CRITERION_RATE * p["episodes"]),
        )
        record = {
            "seed_index": seed_index,
            "epochs": config.epochs,
            "wall_time_s": result["wall_time_s"],
            "param_count": result["param_count"],
            "initial_val_total": result["initial_val_total"],
            "final_train_loss": result["final_train_loss"],
            "final_val_total": result["final_val_total"],
            "final_val_rmse_mnm": result["final_val_rmse_mnm"],
            "final_val_l1_mnm": result["final_val_l1_mnm"],
            "eval_curve": result["eval_curve"],
            "first_success_checkpoint_epochs": first_success,
            "first_criterion_checkpoint_epochs": first_criterion,
            "attention_plan_index": ATTENTION_PLAN_INDEX,
            "per_goal": [{key: value for key, value in episode.items()} for episode in episodes],
            "aggregate": aggregate,
        }
        per_seed_records.append(record)
        if log is not None:
            log(
                f"seed {seed_index}: final eval {aggregate['successes']}/"
                f"{aggregate['episodes']}, median arrival {aggregate['median_arrival_time_s']}, "
                f"val rmse {result['final_val_rmse_mnm']:.2f} mN m, "
                f"wall {result['wall_time_s']:.0f}s"
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
        teacher_digest=teacher_digest,
        window_count=len(obs_windows),
        teacher_wall=teacher_wall,
        sac_metadata=sac_metadata,
        per_seed_records=per_seed_records,
        elapsed=elapsed,
    )
    archive = build_archive(
        eval_goals=eval_goals,
        baseline_truths=baseline_truths,
        baseline_episodes=baseline_episodes,
        curve_stores=curve_stores,
        eval_stores=eval_stores,
        attention_stores=attention_stores,
        sac_arrays=sac_arrays,
    )
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = digest(output / "trajectories.npz")
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_training_curves(output / "training_curves.png", report, output)
    save_comparison(output / "comparison.png", report, output)
    return report


def first_checkpoint(curve, predicate):
    for point in curve:
        if predicate(point):
            return point["epoch"]
    return None


def build_report(
    *,
    seed,
    config,
    train_seeds,
    eval_goals,
    geometry_audit,
    baseline_episodes,
    baseline_aggregate,
    teacher_digest,
    window_count,
    teacher_wall,
    sac_metadata,
    per_seed_records,
    elapsed,
):
    across = aggregate_episodes(
        [episode for record in per_seed_records for episode in record["per_goal"]]
    )
    seed_successes = [record["aggregate"]["successes"] for record in per_seed_records]
    comparison = [
        {
            "label": "解析 IK + PD（第 8 课基线 = 本课教师，正肘解，同一 MuJoCo 竞技场）",
            "source": "planar_arm.inverse_kinematics + joint_pd",
            **baseline_aggregate,
        },
        *(
            [
                {
                    "label": "纯 numpy SAC（第 40 课正式记录，引用）",
                    "source": sac_metadata["source"],
                    **sac_metadata["across_seeds"],
                }
            ]
            if sac_metadata["loaded"]
            else []
        ),
        *[
            {
                "label": f"ACT torch（种子 {record['seed_index']}）",
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
        "torch_version": torch.__version__,
        "geometry_audit": geometry_audit,
        "protocol": {
            "task": (
                "planar 2R arm reaching (links 0.4/0.3 m) - the lesson-40 task imported "
                "unchanged: same start q = (-40, 80) deg at rest, same torque actions "
                "|tau| <= 0.25 N m, same 8-dim observation (protocol verbatim, no dq), "
                "same goal box [0.1, 0.6] x [-0.3, 0.3] m with distance >= 0.15 m, same "
                "reward, same 2 mm x 0.5 s sustained-window acceptance, same 20 fixed "
                "eval goals (paired with the baseline and the lesson-40 record)"
            ),
            "route": (
                "behaviour cloning with a torch ACT policy: the lesson-8 analytic "
                "controller (the same IK+PD chain as the baseline) labels full rollouts "
                "from the fixed start posture; no reward, no replay, no online interaction "
                "during training"
            ),
            "dt_s": DT,
            "horizon_steps": config.episode_steps,
            "horizon_s": config.episode_steps * DT,
            "action": {
                "space": "(tau1, tau2) joint torques in N m",
                "limit_nm": TORQUE_LIMIT_NM,
                "policy_output": "a in (-1, 1)^2 (tanh), tau = 0.25 * a",
            },
            "observation": {
                "features": (
                    "[cos(q1), sin(q1), cos(q2), sin(q2), tau1/0.25, tau2/0.25, gx/0.5, gy/0.5]"
                ),
                "history_note": (
                    "the policy reads the last T=4 observations; the history lets the "
                    "Transformer infer joint velocities (absent from the protocol "
                    "observation) from consecutive poses and the previous normalized torque"
                ),
            },
            "acceptance": {
                "success": (
                    "a window of >= 0.5 s with dist < 2 mm AND max|dq| < 0.1 rad/s; "
                    "evaluation episodes run the full horizon (terminate_on_arrival=False, "
                    "the lesson-40 caliber, byte-for-byte the same run_episode code path)"
                ),
                "learning_criterion": (
                    f"every seed succeeds on >= {CRITERION_RATE:.0%} of the 20 eval goals"
                ),
            },
            "chunked_inference": {
                "chunk_t": config.chunk_t,
                "chunk_h": config.chunk_h,
                "rule": (
                    "replan every chunk_t steps: the first chunk_t actions of each fresh "
                    "H-step chunk are executed, then the policy replans (protocol wording)"
                ),
                "plans_per_episode": math.ceil(config.episode_steps / config.chunk_t),
                "temporal_ensembling": "none (recorded simplification)",
            },
            "simplifications": [
                (
                    "NO CVAE latent z: the teacher is a deterministic controller so the "
                    "label distribution is unimodal - the multi-modality argument for the "
                    "z-latent does not apply; the tested ingredients are action chunking "
                    "and temporal context (the protocol's allowed no-z fallback)"
                ),
                "NO temporal ensembling of overlapping chunks at execution time",
            ],
            "seed_streams": {
                "teacher_goal_stream": "default_rng([master, 21000]) via the env goal stream",
                "torch_init": "torch.manual_seed(default_rng([master, 22000, seed]).integer)",
                "window_shuffle": "default_rng([master, 23000, seed])",
                "eval_goal_stream": "default_rng([master, 14000]) (lesson-40 verbatim)",
                "eval_env_stream": "default_rng([master, 14001]) (goal pinned per episode)",
            },
        },
        "teacher": {
            "source": (
                "lesson-8 analytic IK (positive elbow) + joint PD via "
                "rl_arm_reaching.baseline_actor, the SAME chain as the comparison baseline"
            ),
            "episodes": config.teacher_episodes,
            "steps_per_episode": config.episode_steps,
            "env_steps": config.teacher_episodes * config.episode_steps,
            "digest_sha256": teacher_digest,
            "window_count": window_count,
            "validation_note": (
                f"every {VAL_STRIDE}-th window is validation (deterministic split); "
                "all reported reconstruction errors are on the validation split"
            ),
            "wall_time_s": teacher_wall,
        },
        "architecture": {
            "name": "ActTransformer (minimal ACT, no CVAE)",
            "input": f"last {config.chunk_t} observations ({OBS_DIM} dims each)",
            "encoder": (
                f"linear projection + learned positions -> {config.n_layers}-layer "
                "pre-LN TransformerEncoder, "
                f"{config.n_heads} heads, d_model {config.d_model}, ffn {config.ffn}, "
                "dropout 0"
            ),
            "decoder": (
                f"{config.chunk_h} learned chunk queries -> {config.n_layers}-layer "
                "TransformerDecoder with cross-attention over the observation tokens"
            ),
            "head": "LayerNorm -> Linear(d, d) -> GELU -> Linear(d, 2) -> tanh",
            "loss": f"MSE + {config.w_l1} * L1 on normalized action chunks",
            "optimizer": f"AdamW(lr {config.lr}, weight_decay {config.weight_decay})",
            "schedule": f"cosine annealing to 0 over {config.epochs} epochs",
            "param_count_per_seed": [record["param_count"] for record in per_seed_records],
        },
        "hyperparameters": {**asdict(config)},
        "baseline": {
            "label": "解析 IK + PD（第 8 课基线 = 本课教师，正肘解，同一 MuJoCo 竞技场）",
            "reference": "docs/10-session-08-planar-arm.md (49-pose audit ~2.26e-16 m)",
            "aggregate": baseline_aggregate,
            "per_goal": baseline_episodes,
        },
        "sac_reference": sac_metadata,
        "bc_evaluation": {
            "protocol": (
                f"{config.eval_goal_count} fixed goals per seed, chunked closed-loop "
                "execution, same goals as the baseline and the lesson-40 SAC record (paired)"
            ),
            "per_seed": per_seed_records,
            "across_seeds": across,
        },
        "comparison": comparison,
        "hypothesis": {
            "claim": (
                "the failed 'arrival' level of lessons 39/40 is a REPRESENTATION limit: a "
                "chunked Transformer policy (ACT) cloned from the analytic teacher breaks "
                "the millimetre sustained-arrival barrier that numpy SAC never touched"
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
            "sac_comparison": (sac_metadata["across_seeds"] if sac_metadata["loaded"] else None),
        },
        "sample_efficiency": {
            "teacher_env_steps": config.teacher_episodes * config.episode_steps,
            "sac_env_steps_cited": (
                sac_metadata["training"]["total_env_steps"] if sac_metadata["loaded"] else None
            ),
            "note": (
                "ACT trains on ~1.8e4 teacher-labelled env steps (a near-optimal "
                "supervisor) versus 1.5e6 exploratory SAC steps with 0/60 - different "
                "information sources, recorded side by side, not equated"
            ),
        },
        "training": {
            "train_seeds": train_seeds,
            "epochs_per_seed": config.epochs,
            "windows_per_epoch": config.windows_per_epoch,
            "total_wall_time_s": elapsed,
            "per_seed_wall_time_s": [record["wall_time_s"] for record in per_seed_records],
            "curves_note": (
                "train loss / val total / val mse / val l1 curves per seed live in trajectories.npz"
            ),
        },
        "limitations": [
            (
                "Behaviour cloning bounds ACT by its teacher: the analytic IK+PD chain is "
                "itself the 20/20 comparison baseline, so ACT success measures "
                "imitation fidelity, not learning from scratch."
            ),
            (
                "The protocol observation has no dq while the teacher's PD law reads dq; "
                "the label mapping obs -> action is therefore not a function during fast "
                "motion (the T=4 history only partially restores velocity information)."
            ),
            (
                "Single fixed architecture/hyperparameter set (d_model 128, 2+2 layers, "
                "T=4, H=16, lr 1e-4, 500 epochs); no sweep - failures cannot separate "
                "'the representation' from 'this configuration'."
            ),
            (
                "Ideal simulation: no friction modelling error, no sensor noise, no "
                "payload, no collision; conclusions do not transfer to a real arm."
            ),
            (
                "20 eval goals per seed is a finite sample; success rates are not "
                "population estimates."
            ),
        ],
    }


def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    seeds = report["training"]["train_seeds"]
    goals = report["hyperparameters"]["eval_goal_count"]
    per_goal = ("outcome", "arrival_sample", "len", "settled_sample")
    stats = (
        "final_distance",
        "path_length",
        "straight",
        "efficiency",
        "arrival_time",
        "min_distance",
    )
    curves = (
        "train_loss_curve",
        "val_total_curve",
        "val_mse_curve",
        "val_l1_curve",
        "eval_curve_epochs",
        "eval_curve_successes",
        "eval_curve_min_distance",
    )
    keys = {"eval_goals"}
    keys.update(
        f"{name}_{seed}"
        for seed in range(seeds)
        for name in (*curves, "attention_encoder", "attention_decoder")
    )
    keys.update(f"eval_truth_{seed}_{goal}" for seed in range(seeds) for goal in range(goals))
    keys.update(f"eval_{name}_{seed}" for seed in range(seeds) for name in (*per_goal, *stats))
    keys.update(f"baseline_{name}" for name in (*per_goal, *stats))
    keys.update(f"baseline_truth_{goal}" for goal in range(goals))
    sac = report.get("sac_reference", {})
    if sac.get("loaded"):
        keys.update(
            f"sac_truth_{seed}_{goal}"
            for seed in range(sac["seeds"])
            for goal in range(sac["goals"])
        )
        keys.update(
            f"sac_{name}_{seed}"
            for seed in range(sac["seeds"])
            for name in ("outcome", "arrival_time", "min_distance")
        )
    return keys


def build_archive(
    *,
    eval_goals,
    baseline_truths,
    baseline_episodes,
    curve_stores,
    eval_stores,
    attention_stores,
    sac_arrays,
):
    archive = {"eval_goals": np.asarray(eval_goals, dtype=float)}
    archive.update(curve_stores)
    archive.update(attention_stores)
    for goal_index, truth in enumerate(baseline_truths):
        archive[f"baseline_truth_{goal_index}"] = truth
    for name, values in episode_arrays(baseline_episodes).items():
        archive[f"baseline_{name}"] = values
    for seed_index, store in eval_stores.items():
        for goal_index, truth in enumerate(store["truths"]):
            archive[f"eval_truth_{seed_index}_{goal_index}"] = truth
        for name, values in episode_arrays(store["episodes"]).items():
            archive[f"eval_{name}_{seed_index}"] = values
    archive.update(sac_arrays)
    return archive


# -------------------------------------------------------------------- figures
def save_training_curves(path, report, output):
    configure_plot_font()
    seeds = report["training"]["train_seeds"]
    colors = ("#0f766e", "#2563eb", "#b45309")
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        curves = {
            name: np.asarray([data[f"{name}_{index}"] for index in range(seeds)], dtype=float)
            for name in (
                "train_loss_curve",
                "val_total_curve",
                "eval_curve_epochs",
                "eval_curve_successes",
                "eval_curve_min_distance",
            )
        }
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout="constrained")
    epochs = np.arange(1, curves["train_loss_curve"].shape[1] + 1)
    for index in range(seeds):
        axes[0, 0].plot(
            epochs,
            curves["train_loss_curve"][index],
            color=colors[index % 3],
            linewidth=1.0,
            label=f"种子 {index}",
        )
    axes[0, 0].set(
        xlabel="epoch", ylabel="训练损失（MSE + 0.1·L1）", title="训练损失（全窗口均值，对数轴）"
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    for index in range(seeds):
        axes[0, 1].plot(
            epochs,
            curves["val_total_curve"][index],
            color=colors[index % 3],
            linewidth=1.0,
            label=f"种子 {index}",
        )
    axes[0, 1].set(
        xlabel="epoch",
        ylabel="验证损失（未见窗口）",
        title="验证损失：教师标注重建（泛化，对数轴）",
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)
    for index in range(seeds):
        steps = report["bc_evaluation"]["per_seed"][index]["eval_curve"]
        axes[1, 0].plot(
            [point["epoch"] for point in steps],
            [point["successes"] / point["episodes"] for point in steps],
            "o-",
            markersize=3.5,
            color=colors[index % 3],
            label=f"种子 {index}",
        )
    axes[1, 0].axhline(CRITERION_RATE, color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[1, 0].text(
        0.02,
        CRITERION_RATE + 0.03,
        f"预注册判据 {CRITERION_RATE:.0%}",
        color="#b91c1c",
        fontsize=7.5,
        transform=axes[1, 0].get_yaxis_transform(),
    )
    axes[1, 0].set(
        xlabel="epoch",
        ylabel="周期评估成功率",
        ylim=(-0.05, 1.08),
        title=f"训练中周期评估（前 {report['hyperparameters']['eval_subset']} 个固定目标）",
    )
    axes[1, 0].legend(fontsize=8, loc="lower right")
    for index in range(seeds):
        axes[1, 1].plot(
            curves["eval_curve_epochs"][index],
            np.maximum(curves["eval_curve_min_distance"][index] * 1000, 1e-2),
            "o-",
            markersize=3.5,
            color=colors[index % 3],
            label=f"种子 {index}",
        )
    axes[1, 1].axhline(ARRIVAL_RADIUS_M * 1000, color="#15803d", linestyle="--", linewidth=1.0)
    axes[1, 1].set(
        xlabel="epoch",
        ylabel="最近接近中位（mm，对数轴）",
        title="周期评估的最近接近（绿虚线 = 2 mm 门限）",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_comparison(path, report, output):
    configure_plot_font()
    comparison = report["comparison"]
    seeds = report["training"]["train_seeds"]
    sac_loaded = report["sac_reference"]["loaded"]
    labels = [
        "IK+PD\n基线",
        *(["SAC\n(40 课引用)"] if sac_loaded else []),
        *[f"ACT\n种子 {index}" for index in range(seeds)],
    ]
    colors = [
        "#64748b",
        *(["#7c3aed"] if sac_loaded else []),
        *["#0f766e", "#2563eb", "#b45309"][:seeds],
    ]
    successes = [row["successes"] for row in comparison]
    totals = [row["episodes"] for row in comparison]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        eval_goals = data["eval_goals"]
        baseline_times = np.asarray(data["baseline_arrival_time"], dtype=float)
        baseline_min = np.asarray(data["baseline_min_distance"], dtype=float)
        act_times = [
            np.asarray(data[f"eval_arrival_time_{index}"], dtype=float) for index in range(seeds)
        ]
        act_min = [
            np.asarray(data[f"eval_min_distance_{index}"], dtype=float) for index in range(seeds)
        ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
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
    axes[0].tick_params(axis="x", labelsize=7.5)
    goal_indices = np.arange(len(eval_goals))
    axes[1].plot(
        goal_indices, baseline_times, "s", color="#64748b", markersize=5, label="IK+PD 基线"
    )
    for index in range(seeds):
        axes[1].plot(
            goal_indices,
            act_times[index],
            "o",
            markersize=4,
            alpha=0.8,
            color=colors[-seeds + index] if seeds else colors[-1],
            label=f"ACT 种子 {index}",
        )
    axes[1].set(
        xlabel="评估目标编号（#0/#1 = 第 8 课验收点）",
        ylabel="到达时间（s）",
        title="逐目标到达时间（缺口 = 未到达）",
    )
    axes[1].legend(fontsize=7)
    axes[2].semilogy(
        goal_indices,
        np.maximum(baseline_min * 1000, 1e-2),
        "D",
        color="#64748b",
        markersize=5,
        label="IK+PD 基线",
    )
    for index in range(seeds):
        axes[2].semilogy(
            goal_indices,
            np.maximum(act_min[index] * 1000, 1e-2),
            "o",
            markersize=4,
            alpha=0.8,
            color=colors[-seeds + index] if seeds else colors[-1],
            label=f"ACT 种子 {index}",
        )
    axes[2].axhline(ARRIVAL_RADIUS_M * 1000, color="#15803d", linestyle="--", linewidth=1.0)
    axes[2].set(
        xlabel="评估目标编号",
        ylabel="最近接近（mm，对数轴）",
        title="逐目标最近接近（绿虚线 = 2 mm 门限）",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seeds", type=int, default=TRAIN_SEEDS)
    parser.add_argument("--teacher-episodes", type=int, default=TEACHER_EPISODES)
    parser.add_argument("--episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--windows-per-epoch", type=int, default=WINDOWS_PER_EPOCH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--chunk-t", type=int, default=CHUNK_T)
    parser.add_argument("--chunk-h", type=int, default=CHUNK_H)
    parser.add_argument("--d-model", type=int, default=D_MODEL)
    parser.add_argument("--n-heads", type=int, default=N_HEADS)
    parser.add_argument("--n-layers", type=int, default=N_LAYERS)
    parser.add_argument("--ffn", type=int, default=FFN)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--w-l1", type=float, default=W_L1)
    parser.add_argument("--eval-goals", type=int, default=20)
    parser.add_argument("--eval-every-epochs", type=int, default=EVAL_EVERY_EPOCHS)
    parser.add_argument("--eval-subset", type=int, default=EVAL_SUBSET)
    args = parser.parse_args()
    config = ACTConfig(
        teacher_episodes=args.teacher_episodes,
        episode_steps=args.episode_steps,
        epochs=args.epochs,
        windows_per_epoch=args.windows_per_epoch,
        batch_size=args.batch_size,
        chunk_t=args.chunk_t,
        chunk_h=args.chunk_h,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn=args.ffn,
        lr=args.lr,
        w_l1=args.w_l1,
        eval_goal_count=args.eval_goals,
        eval_every_epochs=args.eval_every_epochs,
        eval_subset=args.eval_subset,
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
                "sac_reference": {
                    "loaded": report["sac_reference"]["loaded"],
                    "source": report["sac_reference"]["source"],
                },
                "seeds": [
                    {"seed_index": record["seed_index"], **record["aggregate"]}
                    for record in report["bc_evaluation"]["per_seed"]
                ],
                "hypothesis": report["hypothesis"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
