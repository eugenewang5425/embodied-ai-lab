"""Lesson 33: Go-Explore - archive the cliff top, return to it, and try to walk there.

Lesson 32 left the four-ways map at two partials: the airdrop made upright arrival
the evaluation norm (33/60 of the w=10 episodes entered |alpha| <= 0.3 rad) but the
policies still never held it (0/60 accepted successes). This lesson plays the last
of the four documented cliff-crossing routes: Go-Explore (Ecoffet et al., 2021).
Its two claims are (1) hard exploration fails through detachment (promising states
are forgotten because the agent cannot come back) and derailment (returning
attempts wreck the promising state), and (2) in a simulator both are dissolved by
an archive: remember the best state per coarse cell and RESET the simulator
directly into it - a privilege no physical agent has, and exactly what the
lesson-29..32 pipelines never used.

Phase 1 (archive exploration, budget ceiling 8 min wall clock; the fixed step
budget actually consumed is far below it and is what is recorded): repeat
[selected cell -> set_state into that cell's member state -> N seeded exploration
steps -> re-bin every visited state into the archive]. Exploration actions are
uniform random in [-3, 3] OVERLAID with the simplest available stabilizer: the
lesson-7 balance LQR engaged by the lesson-7 capture hysteresis (in at
|alpha| < 0.3 rad and |omega| < 2 rad/s, out at |alpha| > 0.5 rad) - no energy
shaping, no swing-up policy; the overlay is recorded in the protocol. Cells are
(angle bin 12 x cart-position bin 6) over wrapped alpha in [-pi, pi) and
x in [-2.4, 2.4]. Each cell keeps ONE member chosen by the recorded lexicographic
criterion: smaller max-normalized settled error of the member's final state first
(same four tolerances as the lesson-7 acceptance, 0.02 m / 0.01 rad / 0.02 m/s /
0.02 rad/s), then shorter stitched path from the exact down start, then earlier
creation. Phase 1 succeeds when a segment ends with the full lesson-7 settled
tail (all four errors inside tolerance for a continuous >= 2 s tail, verified by
re-running the lesson-7 recovery_metrics on the sliced trajectory): the stable
band on the cliff top has been found AND archived.

Phase 2 (robustification): the first kept captures are stitched across cells into
complete teacher trajectories from the exact down start to the settled tail
(stitching is physically exact - every (s, a) pair was really executed, and the
return-fidelity guard replays a stitched slice from a fresh environment bit for
bit). A behavior-cloning policy (the lesson-32 BC objective: MSE on the Gaussian
policy mean head, hand Adam, hand backprop) is trained on those pairs and then
evaluated CLOSED-LOOP from the exact down start with NO state reset: 20
stochastic episodes plus one mean-action episode under the lesson-7 acceptance -
the historical 0 -> 1 question. If robustification fails, the failure is the
formal result ("the map finds the top, open-loop cloning cannot walk there"),
pointing at DAgger/online-correction follow-ups; nothing is smoothed over.

Checks (same conditions throughout): the lesson-7 baseline re-run from the exact
down start (20 repeats), the cited lesson-29 pure-PPO row (0/60) and lesson-32
DAPG rows (0/60 with 33/60 upright arrivals), and the new Go-Explore+BC row.
Process metrics: archive coverage over exploration steps, the first archived
upright-band cell, the first stable-band capture, BC training loss and the
closed-loop outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.dapg_swingup import (
    bc_loss_and_gradient as lesson32_bc_loss_and_gradient,
)
from embodied_learning.experiments.dapg_swingup import first_arrival_time_s
from embodied_learning.experiments.lqr_comparison import SETTLED_TOLERANCES
from embodied_learning.experiments.ppo_swingup import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    EVAL_SEEDS,
    STATE_INPUTS,
    GaussianPolicy,
    RewardFunction,
    baseline_evaluations,
    clip_gradients_,
    down_start_state,
    episode_metrics,
    episode_rewards,
    evaluate_policy,
    failure_counts,
    normalize_observation,
    run_policy_episode,
    stack_controls,
    stack_trajectories,
    summarize_episodes,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)

EXPERIMENT = "goexplore_swingup_lesson33"
SCHEMA_VERSION = 1

# Archive grid (parameters are part of the protocol, not tuning secrets)
ANGLE_BINS = 12  # wrapped alpha in [-pi, pi)
CART_BINS = 6  # cart position in [-SAFE_CART_POSITION, SAFE_CART_POSITION)

# Phase 1: archive exploration
SEGMENT_STEPS = 150  # 6 s of model time per return-then-explore segment
MAX_SEGMENTS = 6000
ACTION_SCALE = 3.0  # uniform random exploration range (the full ±300 N authority)
BUDGET_WALL_CLOCK_S = 480.0  # the <= 8 min envelope; the step budget stops first
CATCH_ANGLE_RAD = 0.3  # lesson-7 capture threshold
CATCH_OMEGA_RAD_S = 2.0  # lesson-7 capture threshold
RELEASE_ANGLE_RAD = 0.5  # lesson-7 release threshold (hysteresis)
CAPTURE_TAIL_S = 2.0  # lesson-7 settled tail length for a capture
CAPTURES_KEPT = 8  # teacher trajectories stored/stitched for phase 2

# Phase 2: robustification (the lesson-32 BC objective, lesson-28 trainer shape)
BC_HIDDEN = (64, 64)
BC_EPOCHS = 300
BC_BATCH = 256
BC_LR = 1e-3
BC_GRAD_CLIP = 0.5
BC_LOG_STD_INIT = 1.0  # exploration std convention of lessons 29/31/32

SEED_OFFSET_SELECT = 8100  # cell selection stream
SEED_OFFSET_ACT = 8200  # exploration action stream
SEED_OFFSET_BC_INIT = 8300  # BC network init
SEED_OFFSET_BC_SHUFFLE = 8400  # BC minibatch order

# Context numbers cited verbatim from the lesson-29/32 official records
# (results/ppo_swingup_2026-09-06 docs/33, results/dapg_swingup_2026-09-06
# docs/37); used only in the four-way comparison table.
DAPG_REFERENCE = {
    "source": "results/dapg_swingup_2026-09-06 (official lesson-32 record, docs/37)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "episodes_with_upright_arrival": 33,
    "tier": "w_BC = 10 (the better of the two tiers)",
}
LESSON29_PPO_REFERENCE = {
    "source": "results/ppo_swingup_2026-09-06 (official lesson-29 record, docs/33)",
    "episodes": 60,
    "successes": 0,
    "median_settled_at_s": None,
    "episodes_with_upright_arrival": 0,
}


FAILURE_LABELS_CN = {
    "cart_safety_boundary": "出界",
    "velocity_safety_boundary": "超速",
    "timeout_without_settling": "超时未稳",
    "nonfinite_state": "数值发散",
    "numerical_warning": "数值警告",
}


def failure_label_cn(reason):
    """Short Chinese failure tag shared by the figures and the demo."""
    reason = reason or ""
    return FAILURE_LABELS_CN.get(reason, reason or "超时未稳")


# --------------------------------------------------------------- the archive
def state_cell(state, reference, *, angle_bins=ANGLE_BINS, cart_bins=CART_BINS):
    """Coarse cell (angle bin, cart bin) of a full state.

    The pole enters through its wrapped error to the reference (continuous cell
    identity across any number of rotations); the cart through its position on
    the [-2.4, 2.4] track. Values on the outer edge clamp into the last bin.
    """
    state = np.asarray(state, dtype=float)
    alpha = float(wrap_angle(state[1] - reference[1]))
    if not np.isfinite(alpha) or not np.isfinite(state[0]):
        raise ValueError("cell state must be finite")
    fraction = (alpha + np.pi) / (2.0 * np.pi)
    angle_bin = int(np.clip(np.floor(fraction * angle_bins), 0, angle_bins - 1))
    span = 2.0 * SAFE_CART_POSITION
    cart_fraction = (float(state[0]) + SAFE_CART_POSITION) / span
    cart_bin = int(np.clip(np.floor(cart_fraction * cart_bins), 0, cart_bins - 1))
    return angle_bin, cart_bin


def cell_centers(cell, reference, *, angle_bins=ANGLE_BINS, cart_bins=CART_BINS):
    """(alpha, x) center of a cell in physical units (figures and demo)."""
    angle_bin, cart_bin = cell
    alpha = -np.pi + (angle_bin + 0.5) * 2.0 * np.pi / angle_bins
    x = -SAFE_CART_POSITION + (cart_bin + 0.5) * 2.0 * SAFE_CART_POSITION / cart_bins
    return alpha, x


def normalized_settled_error(state, reference):
    """max_i |error_i| / tolerance_i - distance to the lesson-7 settled band.

    Exactly the four errors of the lesson-7 acceptance (x, wrapped alpha, v,
    omega) normalized by the settled tolerances (0.02 m / 0.01 rad / 0.02 m/s /
    0.02 rad/s); a state with value <= 1 sits inside the settled band itself.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must have four finite components")
    error = state - reference
    error[1] = wrap_angle(error[1])
    return float(np.max(np.abs(error) / SETTLED_TOLERANCES))


class Archive:
    """Go-Explore archive: one member per coarse cell + the return bookkeeping.

    A member records HOW its state was reached: the segment arrays from the
    parent member's state to this state, plus the parent cell - so a full
    trajectory from the exact down start can be stitched by walking the parent
    chain (every stitched (s, a) pair was really executed; the fidelity guard
    replays slices from a fresh environment bit for bit).
    """

    def __init__(self, reference, *, angle_bins=ANGLE_BINS, cart_bins=CART_BINS):
        self.reference = np.asarray(reference, dtype=float)
        self.angle_bins = int(angle_bins)
        self.cart_bins = int(cart_bins)
        self.cells: dict[tuple[int, int], dict] = {}
        self.serial = 0

    def add(self, cell, state, *, parent_member, seg_states, seg_controls, steps):
        """Offer a candidate; the recorded criterion decides membership.

        `parent_member` is the member dict the segment physically started from -
        kept as an object reference, not a cell key, so later replacements of
        that cell can never corrupt an already-stitched chain (every chain is
        frozen at creation time and stays physically exact).
        """
        self.serial += 1
        current = self.cells.get(cell)
        candidate_key = (normalized_settled_error(state, self.reference), int(steps), self.serial)
        if current is not None:
            current_key = (current["error_max"], current["steps"], current["serial"])
            if candidate_key >= current_key:
                return False
        self.cells[cell] = {
            "state": np.asarray(state, dtype=float).copy(),
            "parent": parent_member,
            "seg_states": np.asarray(seg_states, dtype=float),
            "seg_controls": np.asarray(seg_controls, dtype=float),
            "steps": int(steps),
            "error_max": candidate_key[0],
            "serial": self.serial,
            "sel": 0 if current is None else current["sel"],
        }
        return True

    def select(self, rng):
        """Pick a cell: prefer never-selected cells, then the least-selected."""
        if not self.cells:
            raise ValueError("cannot select from an empty archive")
        keys = list(self.cells)
        fresh = [key for key in keys if self.cells[key]["sel"] == 0]
        if fresh:
            pool = fresh
        else:
            least = min(self.cells[key]["sel"] for key in keys)
            pool = [key for key in keys if self.cells[key]["sel"] == least]
        cell = pool[int(rng.integers(len(pool)))]
        self.cells[cell]["sel"] += 1
        return cell

    @staticmethod
    def member_path(member):
        """Stitched (states, controls) along a frozen member chain to its state."""
        chain = []
        node = member
        while node is not None:
            chain.append(node)
            node = node["parent"]
        chain.reverse()
        states = [chain[0]["seg_states"][0]]
        controls = []
        for node in chain:
            if len(node["seg_controls"]) == 0:
                continue
            states.extend(node["seg_states"][1:])
            controls.extend(node["seg_controls"])
        return np.asarray(states, dtype=float), np.asarray(controls, dtype=float)

    def full_path(self, cell):
        """Stitched path to the current member of `cell`."""
        return self.member_path(self.cells[cell])

    def coverage(self):
        return len(self.cells), self.angle_bins * self.cart_bins

    def upright_cells(self):
        """Occupied cells whose member sits in the |alpha| <= 0.3 rad band."""
        return [
            cell
            for cell, member in self.cells.items()
            if abs(float(wrap_angle(member["state"][1] - self.reference[1]))) <= CATCH_ANGLE_RAD
        ]

    def grids(self):
        """NaN-padded physical grids of the archive (npz + heatmap)."""
        shape = (self.angle_bins, self.cart_bins)
        state_grid = np.full((*shape, 4), np.nan)
        error_grid = np.full(shape, np.nan)
        steps_grid = np.full(shape, np.nan)
        visits_grid = np.zeros(shape)
        for (angle_bin, cart_bin), member in self.cells.items():
            state_grid[angle_bin, cart_bin] = member["state"]
            error_grid[angle_bin, cart_bin] = member["error_max"]
            steps_grid[angle_bin, cart_bin] = member["steps"]
            visits_grid[angle_bin, cart_bin] = member["sel"]
        return state_grid, error_grid, steps_grid, visits_grid


# --------------------------------------------------------- exploration loop
def balance_engaged(state, engaged, reference):
    """Lesson-7 capture hysteresis around the balance LQR (no energy shaping)."""
    alpha = float(wrap_angle(state[1] - reference[1]))
    if engaged and abs(alpha) > RELEASE_ANGLE_RAD:
        return False
    if not engaged and abs(alpha) < CATCH_ANGLE_RAD and abs(float(state[3])) < CATCH_OMEGA_RAD_S:
        return True
    return engaged


def exploration_action(state, engaged, lqr, rng, reference):
    """LQR while engaged, else uniform random in [-ACTION_SCALE, ACTION_SCALE].

    The LQR is linear around the reference: the raw (possibly multi-turn) pole
    angle is first replaced by reference[1] + wrapped(alpha), exactly the
    lesson-7 controller's balance-mode convention.
    """
    if engaged:
        alpha = float(wrap_angle(state[1] - reference[1]))
        normalized = np.asarray(state, dtype=float).copy()
        normalized[1] = reference[1] + alpha
        return lqr.action(normalized), "lqr"
    return (
        np.array([rng.uniform(-ACTION_SCALE, ACTION_SCALE)], np.float32),
        "random",
    )


def capture_slice_metrics(states, controls, reference, dt):
    """Lesson-7 recovery_metrics on the tail slice (the capture verification)."""
    arrays = {
        "states": np.asarray(states, dtype=float),
        "controls": np.asarray(controls, dtype=float),
        "applied_force_n": np.zeros(len(controls)),
        "scheduled_force_n": np.zeros(len(controls)),
        "end_flags": np.array([False, True]),
        "modes": np.full(len(controls), "cap", dtype="<U3"),
    }
    metrics = recovery_metrics(arrays, {"failure_reason": ""}, reference, dt)
    return metrics


def find_capture_start(states, reference, dt):
    """First index whose consecutive in-tolerance run spans >= CAPTURE_TAIL_S.

    Returns None when no such window exists. The run must reach the end of the
    segment for recovery_metrics to certify it (a tail that later breaks is not
    a capture), which the caller verifies through capture_slice_metrics.
    """
    states = np.asarray(states, dtype=float)
    error = states - reference
    error[:, 1] = wrap_angle(error[:, 1])
    in_tolerance = np.all(np.abs(error) <= SETTLED_TOLERANCES, axis=1)
    need = round(CAPTURE_TAIL_S / dt)
    run_length = 0
    for index in range(len(states)):
        run_length = run_length + 1 if in_tolerance[index] else 0
        if run_length >= need + 1:
            return index - need
    return None


def run_exploration(
    reference,
    dt,
    lqr,
    *,
    segments=MAX_SEGMENTS,
    segment_steps=SEGMENT_STEPS,
    master_seed=0,
    angle_bins=ANGLE_BINS,
    cart_bins=CART_BINS,
    captures_kept=CAPTURES_KEPT,
    log=None,
):
    """Phase 1: return-then-explore until the fixed segment budget is spent.

    Every visited state is re-binned into the archive; a segment captures when
    it ends inside the lesson-7 settled tail entered >= 2 s earlier from an
    archive state. The step budget is fixed in advance (determinism); the wall
    clock is only recorded against the 8-minute envelope.
    """
    if segments < 1:
        raise ValueError("segments must be positive")
    if segment_steps < 2:
        raise ValueError("segment_steps must be >= 2")
    archive = Archive(reference, angle_bins=angle_bins, cart_bins=cart_bins)
    root = down_start_state(reference)
    root_cell = state_cell(root, reference, angle_bins=angle_bins, cart_bins=cart_bins)
    archive.add(root_cell, root, parent_member=None, seg_states=[root], seg_controls=[], steps=0)
    select_rng = np.random.default_rng([int(master_seed), SEED_OFFSET_SELECT])
    action_rng = np.random.default_rng([int(master_seed), SEED_OFFSET_ACT])
    env = make_swingup_environment(max_episode_steps=segment_steps)
    coverage_curve = np.empty(segments, dtype=float)
    steps_curve = np.empty(segments, dtype=int)
    selection_cells = np.empty((segments, 2), dtype=int)
    captures = []
    first_upright_step = None
    first_capture_step = None
    failure_counts_stage = {
        "cart_safety_boundary": 0,
        "velocity_safety_boundary": 0,
        "timeout_without_settling": 0,
        "nonfinite_state": 0,
        "numerical_warning": 0,
    }
    lqr_steps = 0
    env_steps = 0
    started = time.perf_counter()
    try:
        for segment in range(segments):
            cell = archive.select(select_rng)
            selection_cells[segment] = cell
            member = archive.cells[cell]
            start = member["state"].copy()
            base_steps = member["steps"]
            env.reset(seed=segment)
            env.unwrapped.set_state(start[:2], start[2:])
            env.unwrapped.data.qfrc_applied[0] = 0.0
            states, controls, modes = [start.copy()], [], []
            engaged = balance_engaged(start, False, reference)
            terminated = False
            failure_reason = ""
            for _ in range(segment_steps):
                action, mode = exploration_action(states[-1], engaged, lqr, action_rng, reference)
                engaged = mode == "lqr"
                lqr_steps += int(engaged)
                state, _, terminated, _truncated, info = env.step(action)
                env_steps += 1
                controls.append(float(action[0]))
                modes.append(mode)
                states.append(np.asarray(state, dtype=float).copy())
                failure_reason = info["failure_reason"]
                if terminated:
                    break
            states_array = np.asarray(states, dtype=float)
            controls_array = np.asarray(controls, dtype=float)
            for step in range(1, len(states_array)):
                archive.add(
                    state_cell(
                        states_array[step], reference, angle_bins=angle_bins, cart_bins=cart_bins
                    ),
                    states_array[step],
                    parent_member=member,
                    seg_states=states_array[: step + 1],
                    seg_controls=controls_array[:step],
                    steps=base_steps + step,
                )
            coverage_curve[segment] = len(archive.cells)
            steps_curve[segment] = env_steps
            if first_upright_step is None and archive.upright_cells():
                first_upright_step = env_steps
            cap_start = find_capture_start(states_array, reference, dt)
            if cap_start is not None:
                tail_states = states_array[cap_start:]
                tail_controls = controls_array[cap_start:]
                metrics = capture_slice_metrics(tail_states, tail_controls, reference, dt)
                if metrics["recovered"]:
                    capture_cell = state_cell(
                        states_array[cap_start],
                        reference,
                        angle_bins=angle_bins,
                        cart_bins=cart_bins,
                    )
                    entry = {
                        "segment": segment,
                        "env_steps": env_steps,
                        "selected_cell": list(cell),
                        "start_state": start,
                        "cap_start": cap_start,
                        "cell": list(capture_cell),
                        "settled_at_s": float(metrics["settled_at_s"]),
                        "tail_s": float(tail_controls.size * dt),
                        "failure_reason": failure_reason,
                        "seg_states": states_array,
                        "seg_controls": controls_array,
                    }
                    if len(captures) < captures_kept:
                        # the whole physically executed path for the teacher data:
                        # the frozen chain to the archive state, then every step of
                        # this segment (wandering, catch, settled tail) - states[k]
                        # precedes controls[k]
                        parent_states, parent_controls = archive.member_path(member)
                        entry["full_states"] = np.vstack([parent_states, states_array[1:]])
                        entry["full_controls"] = np.concatenate([parent_controls, controls_array])
                    captures.append(entry)
                    if first_capture_step is None:
                        first_capture_step = env_steps
            if terminated and failure_reason:
                failure_counts_stage[failure_reason] = (
                    failure_counts_stage.get(failure_reason, 0) + 1
                )
            if log is not None and (segment + 1) % 500 == 0:
                occupied, total = archive.coverage()
                log(
                    f"segment {segment + 1}/{segments}, steps {env_steps}, "
                    f"coverage {occupied}/{total}, captures {len(captures)}"
                )
    finally:
        env.close()
    wall = time.perf_counter() - started
    kept = captures[:captures_kept]
    for index, capture in enumerate(kept):
        capture["index"] = index
    state_grid, error_grid, steps_grid, visits_grid = archive.grids()
    occupied, total = archive.coverage()
    return {
        "archive": archive,
        "segments": segments,
        "segment_steps": segment_steps,
        "env_steps": env_steps,
        "wall_time_s": wall,
        "coverage_curve": coverage_curve,
        "steps_curve": steps_curve,
        "selection_cells": selection_cells,
        "first_upright_step": first_upright_step,
        "first_capture_step": first_capture_step,
        "captures": captures,
        "kept_captures": kept,
        "capture_count": len(captures),
        "failure_counts": failure_counts_stage,
        "lqr_step_fraction": lqr_steps / max(env_steps, 1),
        "grids": {
            "cell_states": state_grid,
            "cell_error": error_grid,
            "cell_steps": steps_grid,
            "cell_visits": visits_grid,
        },
        "coverage_final": occupied / total,
        "cells_occupied": occupied,
        "cells_total": total,
        "upright_cells": len(archive.upright_cells()),
    }


# -------------------------------------------------------------- phase 2: BC
def build_teacher_pairs(captures, reference):
    """BC pairs (lesson-7 alignment): states[k] precedes action k."""
    obs_list, action_list = [], []
    for capture in captures:
        states, controls = capture["full_states"], capture["full_controls"]
        for step in range(len(controls)):
            obs_list.append(normalize_observation(states[step], reference))
            action_list.append(float(controls[step]))
    if not obs_list:
        raise ValueError("no teacher captures: phase 1 found no stable band")
    return np.asarray(obs_list, dtype=float), np.asarray(action_list, dtype=float)


def bc_loss_and_gradient(policy, obs, actions):
    """Lesson-32 BC objective, re-exported for the robustification stage."""
    return lesson32_bc_loss_and_gradient(policy, obs, actions)


def train_robustified_bc(
    obs,
    actions,
    *,
    epochs=BC_EPOCHS,
    batch_size=BC_BATCH,
    lr=BC_LR,
    init_seed=0,
    shuffle_seed=0,
    hidden=BC_HIDDEN,
    grad_clip=BC_GRAD_CLIP,
):
    """Pure BC on the archive teacher pairs (lesson-28 loop, lesson-32 loss).

    The policy is the lesson-29 Gaussian tower (5-64-64-1 mean head + fixed
    exploration std); only the mean head is trained, by the hand Adam.
    """
    obs = np.asarray(obs, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != STATE_INPUTS or obs.shape[0] != len(actions):
        raise ValueError("teacher pairs must be aligned (N, 5) obs and (N,) actions")
    if min(len(obs), epochs) < 1:
        raise ValueError("teacher pairs and epochs must be positive")
    policy = GaussianPolicy(STATE_INPUTS, hidden, init_seed, BC_LOG_STD_INIT)
    parameters = [*policy.trunk.weights, *policy.trunk.biases]
    optimizer = AdamOptimizer(parameters, lr=lr)
    rng = np.random.default_rng(shuffle_seed)
    loss_curve = np.empty(epochs, dtype=float)
    for epoch in range(epochs):
        order = rng.permutation(len(obs))
        total, seen = 0.0, 0
        for start in range(0, len(obs), batch_size):
            batch = order[start : start + batch_size]
            loss, gradients = bc_loss_and_gradient(policy, obs[batch], actions[batch])
            clip_gradients_(gradients, grad_clip)
            optimizer.step(parameters, gradients)
            total += loss * len(batch)
            seen += len(batch)
        loss_curve[epoch] = total / seen
    return policy, loss_curve


def deterministic_bc_episode(policy, reward, reference, dt):
    """One mean-action closed-loop episode from the exact down start (no reset)."""
    arrays, reason = run_policy_episode(
        policy, reward, reference, horizon=EVAL_EPISODE_STEPS, env_seed=0, deterministic=True
    )
    metrics = episode_metrics(arrays, reason, reference, dt)
    record = {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "return": float(episode_rewards(arrays, reward).sum()),
        "first_arrival_s": first_arrival_time_s(arrays["states"], reference, dt),
        "failure_reason": reason,
    }
    return record, arrays


def evaluate_bc_policy(policy, reward, reference, dt, *, master_seed, count=EVAL_SEEDS):
    """`count` stochastic episodes from the exact down start (lesson-29 stream)."""
    episodes = evaluate_policy(policy, reward, reference, dt, master_seed=master_seed, count=count)
    for episode in episodes:
        episode["first_arrival_s"] = first_arrival_time_s(
            episode["arrays"]["states"], reference, dt
        )
    return episodes


# -------------------------------------------------------------------- guard
def return_fidelity_guard(reference, dt, *, master_seed=0, steps=8):
    """The lesson's pipeline guard: the return privilege must be exact.

    (1) set_state round trip: a written state must be read back bit for bit
    (including a wrapped-multi-turn theta). (2) A stitched archive path must be
    physically exact: replaying a recorded slice's actions from its first state
    in a fresh environment reproduces the recorded states bit for bit.
    """
    cases = [down_start_state(reference)]
    rotated = down_start_state(reference)
    rotated[1] += 3.0 * 2.0 * np.pi + 0.7  # multi-turn raw theta
    rotated[3] = -1.5
    cases.append(rotated)
    round_trip = []
    env = make_swingup_environment(max_episode_steps=steps + 1)
    try:
        for case in cases:
            env.reset(seed=0)
            env.unwrapped.set_state(case[:2], case[2:])
            env.unwrapped.data.qfrc_applied[0] = 0.0
            qpos = np.asarray(env.unwrapped.data.qpos, dtype=float).copy()
            qvel = np.asarray(env.unwrapped.data.qvel, dtype=float).copy()
            round_trip.append(
                bool(np.array_equal(qpos, case[:2]) and np.array_equal(qvel, case[2:]))
            )

        rng = np.random.default_rng([int(master_seed), SEED_OFFSET_ACT])
        states, controls = [down_start_state(reference)], []
        env.reset(seed=0)
        env.unwrapped.set_state(states[0][:2], states[0][2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        for _ in range(steps):
            action = np.array([rng.uniform(-ACTION_SCALE, ACTION_SCALE)], np.float32)
            state, _, terminated, _truncated, _info = env.step(action)
            if terminated:
                break
            controls.append(float(action[0]))
            states.append(np.asarray(state, dtype=float).copy())
        env.reset(seed=99)
        replayed = [states[0].copy()]
        env.unwrapped.set_state(states[0][:2], states[0][2:])
        env.unwrapped.data.qfrc_applied[0] = 0.0
        for action in controls:
            state, _, terminated, _truncated, _info = env.step(np.array([action], np.float32))
            replayed.append(np.asarray(state, dtype=float).copy())
            if terminated:
                break
        replay_identical = len(replayed) == len(states) and all(
            np.array_equal(a, b) for a, b in zip(states, replayed, strict=True)
        )
    finally:
        env.close()
    return {
        "claim": (
            "the Go-Explore return must be exact: set_state writes a member state bit "
            "for bit, and a stitched archive slice replayed from a fresh environment "
            "reproduces its recorded states bit for bit (the teacher data of phase 2 "
            "is real physics, not a paper trajectory)"
        ),
        "steps": int(steps),
        "bitwise_identical_set_state": bool(all(round_trip)),
        "bitwise_identical_stitched_replay": bool(replay_identical),
    }


# ------------------------------------------------------------- run experiment
def run_experiment(
    output,
    *,
    seed=0,
    segments=MAX_SEGMENTS,
    segment_steps=SEGMENT_STEPS,
    eval_seed_count=EVAL_SEEDS,
    bc_epochs=BC_EPOCHS,
    angle_bins=ANGLE_BINS,
    cart_bins=CART_BINS,
    captures_kept=CAPTURES_KEPT,
    log=None,
):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(segments, int) or not 1 <= segments <= 100_000:
        raise ValueError("segments must be an integer in [1, 100000]")
    if not isinstance(segment_steps, int) or not 2 <= segment_steps <= EVAL_EPISODE_STEPS:
        raise ValueError("segment_steps must be an integer in [2, 750]")
    if not isinstance(eval_seed_count, int) or not 1 <= eval_seed_count <= 100:
        raise ValueError("eval_seed_count must be an integer in [1, 100]")
    if not isinstance(bc_epochs, int) or not 1 <= bc_epochs <= 5000:
        raise ValueError("bc_epochs must be an integer in [1, 5000]")
    if not isinstance(angle_bins, int) or not 2 <= angle_bins <= 180:
        raise ValueError("angle_bins must be an integer in [2, 180]")
    if not isinstance(cart_bins, int) or not 2 <= cart_bins <= 120:
        raise ValueError("cart_bins must be an integer in [2, 120]")
    if not isinstance(captures_kept, int) or not 1 <= captures_kept <= 100:
        raise ValueError("captures_kept must be an integer in [1, 100]")
    log = log if log is not None else (lambda message: None)
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12:
        raise ValueError("control limit disagrees with the lesson-7 design")
    reward = RewardFunction(reference)

    guard = return_fidelity_guard(reference, dt, master_seed=seed)
    baseline_records, baseline_states, baseline_controls, baseline_identical = baseline_evaluations(
        design, eval_seed_count
    )

    exploration = run_exploration(
        reference,
        dt,
        design.controller,
        segments=segments,
        segment_steps=segment_steps,
        master_seed=seed,
        angle_bins=angle_bins,
        cart_bins=cart_bins,
        captures_kept=captures_kept,
        log=log,
    )
    phase2 = {
        "run": bool(exploration["kept_captures"]),
        "teacher_trajectories": len(exploration["kept_captures"]),
        "teacher_pairs": 0,
        "bc_loss_first": None,
        "bc_loss_last": None,
        "wall_time_s": None,
    }
    policy = None
    bc_curve = None
    episodes, det_record, det_arrays = [], None, None
    if exploration["kept_captures"]:
        bc_started = time.perf_counter()
        obs, actions = build_teacher_pairs(exploration["kept_captures"], reference)
        policy, bc_curve = train_robustified_bc(
            obs,
            actions,
            epochs=bc_epochs,
            init_seed=[seed, SEED_OFFSET_BC_INIT],
            shuffle_seed=[seed, SEED_OFFSET_BC_SHUFFLE],
        )
        episodes = evaluate_bc_policy(
            policy, reward, reference, dt, master_seed=seed, count=eval_seed_count
        )
        det_record, det_arrays = deterministic_bc_episode(policy, reward, reference, dt)
        phase2.update(
            {
                "teacher_pairs": len(obs),
                "bc_loss_first": float(bc_curve[0]),
                "bc_loss_last": float(bc_curve[-1]),
                "wall_time_s": time.perf_counter() - bc_started,
                "run": True,
            }
        )
    else:
        det_record = {
            "recovered": None,
            "terminated": None,
            "settled_at_s": None,
            "return": None,
            "first_arrival_s": None,
            "failure_reason": "phase 1 found no stable band; no policy was trained",
        }

    failure_cases = [
        episode for episode in episodes if episode["terminated"] or not episode["recovered"]
    ]

    output.mkdir(parents=True, exist_ok=False)
    report = build_report(
        seed=seed,
        design=design,
        eval_seed_count=eval_seed_count,
        bc_epochs=bc_epochs,
        segments=segments,
        segment_steps=segment_steps,
        angle_bins=angle_bins,
        cart_bins=cart_bins,
        captures_kept=captures_kept,
        guard=guard,
        baseline_records=baseline_records,
        baseline_identical=baseline_identical,
        exploration=exploration,
        phase2=phase2,
        episodes=episodes,
        det_record=det_record,
        elapsed=time.perf_counter() - started,
    )
    archive_npz = build_archive(
        exploration=exploration,
        baseline_states=baseline_states,
        baseline_controls=baseline_controls,
        bc_curve=bc_curve,
        policy=policy,
        episodes=episodes,
        det_arrays=det_arrays,
        failure_cases=failure_cases[:1],
    )
    np.savez_compressed(output / "trajectories.npz", **archive_npz)
    report["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_archive_map(output / "archive_map.png", report, output)
    save_capture_analysis(output / "capture_analysis.png", report, output)
    save_robustification(output / "robustification.png", report, output)
    return report


def arrival_summary(episodes):
    """First-arrival statistics (|alpha| <= 0.3 rad) over evaluation episodes."""
    times = [episode.get("first_arrival_s") for episode in episodes]
    arrived = [value for value in times if value is not None]
    return {
        "episodes": len(episodes),
        "episodes_with_arrival": len(arrived),
        "arrival_fraction": len(arrived) / len(episodes) if episodes else 0.0,
        "median_first_arrival_s": float(np.median(arrived)) if arrived else None,
    }


def four_way_comparison(
    baseline_summary, bc_summary, bc_arrival, dapg=DAPG_REFERENCE, ppo=LESSON29_PPO_REFERENCE
):
    """The four cliff-crossing routes on the same lesson-7 acceptance."""
    return [
        {
            "label": "基线（第 7 课能量整形+LQR，零样本）",
            "episodes": baseline_summary["episodes"],
            "successes": baseline_summary["successes"],
            "median_settled_at_s": baseline_summary["median_settled_at_s"],
            "episodes_with_upright_arrival": None,
            "source": "this record",
        },
        {
            "label": "纯 PPO（第 29 课，只凭奖励）",
            "episodes": ppo["episodes"],
            "successes": ppo["successes"],
            "median_settled_at_s": ppo["median_settled_at_s"],
            "episodes_with_upright_arrival": ppo["episodes_with_upright_arrival"],
            "source": ppo["source"],
        },
        {
            "label": "DAPG 示教空投（第 32 课，w=10 档）",
            "episodes": dapg["episodes"],
            "successes": dapg["successes"],
            "median_settled_at_s": dapg["median_settled_at_s"],
            "episodes_with_upright_arrival": dapg["episodes_with_upright_arrival"],
            "source": dapg["source"],
        },
        {
            "label": "Go-Explore+BC（第 33 课，本记录）",
            "episodes": bc_summary["episodes"],
            "successes": bc_summary["successes"],
            "median_settled_at_s": bc_summary["median_settled_at_s"],
            "episodes_with_upright_arrival": bc_arrival["episodes_with_arrival"],
            "source": "this record",
        },
    ]


def build_report(
    *,
    seed,
    design,
    eval_seed_count,
    bc_epochs,
    segments,
    segment_steps,
    angle_bins,
    cart_bins,
    captures_kept,
    guard,
    baseline_records,
    baseline_identical,
    exploration,
    phase2,
    episodes,
    det_record,
    elapsed,
):
    reference = design.controller.reference
    baseline_summary = summarize_episodes(baseline_records)
    bc_summary = (
        summarize_episodes(episodes)
        if episodes
        else {
            "episodes": 0,
            "successes": 0,
            "median_settled_at_s": None,
            "median_peak_abs_motor_force_n": None,
            "failure_counts": failure_counts([]),
        }
    )
    bc_arrival = arrival_summary(episodes)
    first_up = exploration["first_upright_step"]
    first_cap = exploration["first_capture_step"]
    phase1_verdict = (
        "stable band found and archived (>= 2 s lesson-7 settled tail from an archive state)"
        if exploration["capture_count"] > 0
        else "no stable band captured within the budget"
    )
    if not phase2["run"]:
        phase2_verdict = "no teacher data: phase 1 failed, phase 2 not run"
    elif bc_summary["successes"] > 0:
        phase2_verdict = "robustified policy accepted from the down start (0 -> 1)"
    else:
        phase2_verdict = (
            "the archive finds the top but open-loop BC cannot walk there "
            "(closed-loop failures point at DAgger / policy-gradient fine-tuning)"
        )
    return {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "protocol": {
            "task": "lesson-7 full-rotation swing-up, exact down start (reference angle -180 deg)",
            "route": (
                "Go-Explore (Ecoffet et al. 2021), the last of the four documented "
                "cliff-crossing routes: archive promising states per coarse cell and reset "
                "the simulator directly into them (detachment/derailment dissolved by the "
                "simulation-only return privilege); sister of lessons 30-32, aimed at the "
                "never-held settled tail on the cliff top"
            ),
            "dt_s": design.dt,
            "reference_state": reference.tolist(),
            "grid": {
                "angle_bins": angle_bins,
                "cart_bins": cart_bins,
                "cells_total": angle_bins * cart_bins,
                "angle": "wrapped alpha in [-pi, pi) over 12 equal bins",
                "cart": "x in [-2.4, 2.4] m over 6 equal bins (the failure boundary)",
                "note": "the cell key is coarse; the member stores the full 4-D state",
            },
            "member_criterion": {
                "primary": (
                    "smaller max-normalized settled error of the member's final state: "
                    "max(|x|/0.02, |wrap(alpha)|/0.01, |v|/0.02, |omega|/0.02) - the lesson-7 "
                    "settled tolerances; <= 1.0 means the state itself is inside the band"
                ),
                "secondary": "fewer stitched steps from the exact down start (return cost)",
                "tie": "earlier creation serial (full determinism)",
            },
            "selection_rule": (
                "prefer cells never selected, otherwise the least-selected occupied cells; "
                "uniform choice inside the pool through the seeded selection stream"
            ),
            "exploration": {
                "action": (
                    "uniform random in [-3, 3] overlaid with the lesson-7 balance LQR: engaged "
                    "at |alpha| < 0.3 rad and |omega| < 2 rad/s, released at |alpha| > 0.5 rad "
                    "(the lesson-7 capture hysteresis) - no energy shaping, no swing-up policy"
                ),
                "segment_steps": segment_steps,
                "segments": segments,
                "max_env_steps": segments * segment_steps,
                "budget_wall_clock_s": BUDGET_WALL_CLOCK_S,
                "budget_note": (
                    "the segment budget is fixed in advance for determinism and is what the "
                    "run consumes; the wall-clock envelope (8 min) is recorded, never used "
                    "as a stopping rule"
                ),
                "return": (
                    "gymnasium MujocoEnv set_state(qpos, qvel) into the member state, applied "
                    "force cleared to zero - the simulation-only privilege this lesson tests"
                ),
                "seed_streams": {
                    "cell_selection": "default_rng([master, 8100])",
                    "exploration_actions": "default_rng([master, 8200])",
                    "segment_reset": "env.reset(seed=segment_index)",
                },
            },
            "capture_criterion": {
                "definition": (
                    "a segment ends inside the lesson-7 settled band entered >= 2 s earlier "
                    "from an archive state: all four wrapped errors within 0.02 m / 0.01 rad "
                    "/ 0.02 m/s / 0.02 rad/s continuously to the segment end"
                ),
                "verification": (
                    "lesson-7 recovery_metrics re-run on the sliced trajectory (recovered == "
                    "settled_at_s is not None); the first captured state's cell is marked a "
                    "stable-band cell"
                ),
            },
            "robustification": {
                "teacher": (
                    "the first kept captures stitched across cells into full trajectories "
                    "from the exact down start to the settled tail (states[k] precedes "
                    "controls[k], the lesson-7 archive alignment)"
                ),
                "captures_kept": captures_kept,
                "bc": "lesson-32 objective MSE(mean(s), a) on the lesson-29 Gaussian tower",
                "hidden": list(BC_HIDDEN),
                "epochs": bc_epochs,
                "batch": BC_BATCH,
                "lr": BC_LR,
                "grad_clip": BC_GRAD_CLIP,
                "log_std_init": BC_LOG_STD_INIT,
                "seed_streams": {
                    "network_init": "default_rng([master, 8300])",
                    "minibatch_order": "default_rng([master, 8400])",
                },
            },
            "eval_initial_state": "exact resting down start (reference angle -180 deg), no jitter",
            "eval_protocol": (
                f"{eval_seed_count} stochastic episodes (lesson-29 action stream, "
                f"default_rng([master, 2000, eval_seed])) + one mean-action episode; "
                "closed-loop with NO state reset; lesson-7 recovery_metrics reused verbatim "
                "(0.02 m / 0.01 rad / 0.02 m/s / 0.02 rad/s held >= 2 s)"
            ),
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * design.dt,
            "cart_failure_boundary_m": SAFE_CART_POSITION,
            "control_limit": CONTROL_LIMIT,
            "actuator_gear": design.actuator_gear,
            "observation": {
                "features": "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]",
                "note": "identical to lessons 29-32",
            },
            "reward_for_reported_returns": RewardFunction(reference).as_dict(),
        },
        "guard": guard,
        "baseline": {
            "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), zero-shot",
            "protocol": f"{eval_seed_count} repeats of the lesson-7 down scenario",
            "deterministic_identical_repeats": baseline_identical,
            **baseline_summary,
        },
        "phase1": {
            "segments_run": exploration["segments"],
            "segment_steps": exploration["segment_steps"],
            "env_steps": exploration["env_steps"],
            "wall_time_s": exploration["wall_time_s"],
            "cells_occupied": exploration["cells_occupied"],
            "cells_total": exploration["cells_total"],
            "coverage_final": exploration["coverage_final"],
            "upright_cells": exploration["upright_cells"],
            "first_upright_step": first_up,
            "first_capture_step": first_cap,
            "capture_count": exploration["capture_count"],
            "kept_captures": len(exploration["kept_captures"]),
            "lqr_step_fraction": exploration["lqr_step_fraction"],
            "failure_counts": exploration["failure_counts"],
            "per_capture": [
                {
                    "index": capture["index"],
                    "segment": capture["segment"],
                    "env_steps": capture["env_steps"],
                    "cell": capture["cell"],
                    "cap_start": capture["cap_start"],
                    "settled_at_s": capture["settled_at_s"],
                    "tail_s": capture["tail_s"],
                    "teacher_steps": len(capture["full_controls"]),
                }
                for capture in exploration["kept_captures"]
            ],
            "verdict": phase1_verdict,
        },
        "phase2": {
            **phase2,
            "stochastic": bc_summary,
            "arrival": bc_arrival,
            "deterministic": det_record,
            "verdict": phase2_verdict,
        },
        "four_way_comparison": four_way_comparison(baseline_summary, bc_summary, bc_arrival),
        "lesson29_reference": LESSON29_PPO_REFERENCE,
        "dapg_reference": DAPG_REFERENCE,
        "hypothesis": {
            "claim": (
                "Go-Explore (Ecoffet et al. 2021) claims that archiving reached states and "
                "resetting the simulator into them turns 'has been there' into 'can be back "
                "at will', letting exploration run on the cliff top. Primary question 1: "
                "does archive exploration find AND archive the stable tail region (the "
                ">= 2 s settled band) that no lesson-29..32 policy ever held? Primary "
                "question 2: does a BC policy robustified on the stitched archive "
                "trajectories achieve the historical first accepted success from the exact "
                "down start (0 -> 1)? A negative answer to either is recorded as the "
                "formal result"
            ),
            "phase1_captured": exploration["capture_count"] > 0,
            "phase2_zero_to_one": bool(bc_summary["successes"] > 0),
        },
        "failure_analysis": {
            "eval_counts": failure_counts(episodes),
            "featured_case": next(
                (
                    {
                        "eval_seed": episode["eval_seed"],
                        "failure_reason": episode["failure_reason"],
                        "terminated": bool(episode["terminated"]),
                        "recovered": bool(episode["recovered"]),
                        "settled_at_s": episode["settled_at_s"],
                        "max_abs_cart_position_m": episode["max_abs_cart_position_m"],
                        "peak_abs_motor_force_n": episode["peak_abs_motor_force_n"],
                    }
                    for episode in episodes
                    if episode["terminated"] or not episode["recovered"]
                ),
                None,
            ),
        },
        "training": {
            "phase1_env_steps": exploration["env_steps"],
            "phase1_wall_time_s": exploration["wall_time_s"],
            "phase2_wall_time_s": phase2["wall_time_s"],
            "wall_time_s_total": elapsed,
            "curves_note": (
                "coverage_curve, steps_curve and the BC loss curve live in trajectories.npz"
            ),
        },
        "limitations": [
            (
                "The exploration overlay is the lesson-7 balance LQR behind the lesson-7 "
                "capture hysteresis - a stabilizer, not a swing-up policy; pure-random "
                "exploration without it is not covered by this record."
            ),
            (
                "The return privilege exists only in the simulator; on hardware the archive "
                "would need a real reset mechanism or a reachability policy, which is "
                "precisely what Go-Explore's robustification is for."
            ),
            (
                "BC clones one stitched path per capture; the pumping sections are random "
                "actions whose per-state mean carries no information - an open-loop student "
                "of a stochastic teacher was the predictable weak point, and the record "
                "keeps whatever the closed-loop numbers turn out to be."
            ),
            (
                "One task, one nominal MuJoCo model, no noise/delay/mass error; the grid "
                "(12 x 6), the member criterion and the overlay thresholds are hand-picked "
                "and recorded, not tuned."
            ),
            (
                "The cited lesson-29/32 rows in four_way_comparison come from those official "
                "records (60 episodes each), not re-run here; 20 fresh episodes bound the "
                "new row only."
            ),
        ],
    }


def build_archive(
    *,
    exploration,
    baseline_states,
    baseline_controls,
    bc_curve,
    policy,
    episodes,
    det_arrays,
    failure_cases,
):
    horizon = EVAL_EPISODE_STEPS
    archive = {
        "baseline_states": np.asarray(baseline_states, dtype=float),
        "baseline_controls": np.asarray(baseline_controls, dtype=float),
        "coverage_curve": exploration["coverage_curve"],
        "steps_curve": exploration["steps_curve"],
        "selection_cells": exploration["selection_cells"],
        **{f"grid_{name}": grid for name, grid in exploration["grids"].items()},
    }
    for index, capture in enumerate(exploration["kept_captures"]):
        archive[f"capture{index}_start_state"] = capture["start_state"]
        archive[f"capture{index}_seg_states"] = capture["seg_states"]
        archive[f"capture{index}_seg_controls"] = capture["seg_controls"]
        archive[f"capture{index}_full_states"] = capture["full_states"]
        archive[f"capture{index}_full_controls"] = capture["full_controls"]
    if bc_curve is not None:
        archive["bc_loss_curve"] = bc_curve
    if policy is not None:
        for name, array in policy.arrays().items():
            archive[f"policy_{name}"] = array
    if episodes:
        stacked, lengths = stack_trajectories(
            [episode["arrays"]["states"] for episode in episodes], horizon
        )
        archive["eval_states"] = stacked
        archive["eval_lengths"] = lengths
        archive["eval_controls"] = stack_controls(
            [episode["arrays"]["controls"] for episode in episodes], horizon
        )
        archive["eval_terminated"] = np.asarray(
            [bool(episode["terminated"]) for episode in episodes], dtype=bool
        )
        archive["eval_settled_s"] = np.asarray(
            [
                float(episode["settled_at_s"]) if episode["settled_at_s"] is not None else np.nan
                for episode in episodes
            ],
            dtype=float,
        )
        archive["eval_returns"] = np.asarray(
            [float(episode["return"]) for episode in episodes], dtype=float
        )
        archive["eval_first_arrival_s"] = np.asarray(
            [
                float(episode["first_arrival_s"])
                if episode["first_arrival_s"] is not None
                else np.nan
                for episode in episodes
            ],
            dtype=float,
        )
    if det_arrays is not None:
        archive["det_states"] = det_arrays["states"]
        archive["det_controls"] = det_arrays["controls"]
    if failure_cases:
        archive["case0_states"] = failure_cases[0]["arrays"]["states"]
        archive["case0_controls"] = failure_cases[0]["arrays"]["controls"]
    return archive


def expected_npz_keys(report):
    """Full archive key set implied by the summary (used by the demo loader)."""
    keys = {
        "baseline_states",
        "baseline_controls",
        "coverage_curve",
        "steps_curve",
        "selection_cells",
        "grid_cell_states",
        "grid_cell_error",
        "grid_cell_steps",
        "grid_cell_visits",
    }
    kept = report["phase1"]["kept_captures"]
    keys.update(
        f"capture{index}_{suffix}"
        for index in range(kept)
        for suffix in ("start_state", "seg_states", "seg_controls", "full_states", "full_controls")
    )
    if report["phase2"]["run"]:
        hidden = tuple(report["protocol"]["robustification"]["hidden"])
        names = ["log_std"]
        for index in range(len(hidden) + 1):
            names.extend((f"weight_{index}", f"bias_{index}"))
        keys.update(f"policy_{name}" for name in names)
        keys.add("bc_loss_curve")
        keys.update(
            f"eval_{suffix}"
            for suffix in (
                "states",
                "lengths",
                "controls",
                "terminated",
                "settled_s",
                "returns",
                "first_arrival_s",
            )
        )
        keys.update(f"det_{suffix}" for suffix in ("states", "controls"))
    if report["failure_analysis"]["featured_case"] is not None:
        keys.update(f"case0_{suffix}" for suffix in ("states", "controls"))
    return keys


# ------------------------------------------------------------------- figures
def save_archive_map(path, report, output):
    """Mode-1 figure: member-error heatmap, selection trace and the coverage curve."""
    configure_plot_font()
    from matplotlib.colors import LogNorm

    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        errors = data["grid_cell_error"]
        selection = data["selection_cells"]
        coverage = data["coverage_curve"]
        steps_curve = data["steps_curve"]
    angle_bins, cart_bins = errors.shape
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    shown = np.where(~np.isnan(errors), errors, np.nan)
    image = axes[0].imshow(
        shown,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
        norm=LogNorm(),  # the member error spans two orders of magnitude
        extent=[-SAFE_CART_POSITION, SAFE_CART_POSITION, -180.0, 180.0],
    )
    fig.colorbar(image, ax=axes[0], label="成员 max|误差|/容差（对数轴，1 = 稳定带）")
    trace = selection[: min(600, len(selection))]
    alpha_centers = np.degrees(-np.pi + (trace[:, 0] + 0.5) * 2.0 * np.pi / angle_bins)
    cart_centers = -SAFE_CART_POSITION + (trace[:, 1] + 0.5) * 2.0 * SAFE_CART_POSITION / cart_bins
    axes[0].plot(cart_centers, alpha_centers, "-", color="#f97316", linewidth=0.7, alpha=0.7)
    upright_alpha = np.degrees(0.3)
    axes[0].axhspan(-upright_alpha, upright_alpha, color="#ef4444", alpha=0.12)
    axes[0].set(
        xlabel="小车位置（m）",
        ylabel="杆角 α（度，0 = 直立）",
        title=f"档案热图：{report['phase1']['cells_occupied']}/{report['phase1']['cells_total']} 格"
        f"（覆盖 {report['phase1']['coverage_final'] * 100:.0f}%，红线 = 直立带）",
    )
    axes[1].plot(
        np.arange(1, len(coverage) + 1),
        coverage * 100.0 / (angle_bins * cart_bins),
        color="#2563eb",
    )
    first_cap = report["phase1"]["first_capture_step"]
    if first_cap is not None:
        cap_index = int(np.searchsorted(steps_curve, first_cap))
        axes[1].axvline(
            cap_index + 1,
            color="#16a34a",
            linestyle="--",
            linewidth=1.2,
            label=f"首次捕获稳定带（{first_cap / 1000:.0f}k 步）",
        )
        axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set(
        xlabel="探索段编号",
        ylabel="档案覆盖率（%）",
        title="覆盖曲线：格子数 / 总格数",
        ylim=(0, 105),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_capture_analysis(path, report, output):
    """Mode-2 figure: the first capture's state trajectories around the moment."""
    configure_plot_font()
    if not report["phase1"]["per_capture"]:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.text(
            0.5,
            0.5,
            "阶段一未捕获稳定带：无捕获轨迹可展示",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set(title="稳定带捕获：无")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    capture = report["phase1"]["per_capture"][0]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        full_states = data["capture0_full_states"]
        full_controls = data["capture0_full_controls"]
        seg_controls = data["capture0_seg_controls"]
    cap_start = capture["cap_start"]
    # the stitched path ends with this segment: the archive reset sits where the
    # parent chain hands over to the segment
    archive_index = len(full_states) - len(seg_controls) - 1
    times = np.arange(len(full_states)) * dt
    tail_start = archive_index + cap_start
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    for ax in axes.ravel():
        ax.axvspan(times[tail_start], times[-1], color="#16a34a", alpha=0.10)
        ax.axvline(times[archive_index], color="#7c3aed", linestyle=":", linewidth=1.2)
        ax.axvline(times[tail_start], color="#16a34a", linestyle="--", linewidth=1.2)
    axes[0, 0].plot(times, np.cos(full_states[:, 1] - ref_theta), color="#2563eb")
    axes[0, 0].axhspan(-1, 0, alpha=0.06, color="orange")
    axes[0, 0].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title=f"第一次捕获（段 {capture['segment']}）：绿区 = ≥2 s 稳定尾段，紫点线 = 档案重置",
    )
    axes[0, 1].plot(times, full_states[:, 0], color="#0f766e")
    for bound in (-SAFE_CART_POSITION, SAFE_CART_POSITION):
        axes[0, 1].axhline(bound, color="red", linestyle=":", linewidth=0.8)
    axes[0, 1].set(
        ylabel="小车位置（m）",
        xlabel="仿真时间（s）",
        title="小车位置（红点线 = ±2.4 m 边界）",
    )
    axes[1, 0].plot(times, full_states[:, 3], color="#7c3aed")
    axes[1, 0].axhline(2.0, color="#94a3b8", linestyle=":", linewidth=0.8)
    axes[1, 0].axhline(-2.0, color="#94a3b8", linestyle=":", linewidth=0.8)
    axes[1, 0].set(
        ylabel="杆角速度（rad/s）",
        xlabel="仿真时间（s）",
        title="角速度（虚线 = 第 7 课抓取阈值 ±2）",
    )
    edges = np.arange(len(full_controls) + 1) * dt
    axes[1, 1].stairs(full_controls * report["protocol"]["actuator_gear"], edges, color="#b45309")
    axes[1, 1].set(
        ylabel="电机力（N）",
        xlabel="仿真时间（s）",
        title="电机输入：随机泵能段 + LQR 抓取/保持段",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_robustification(path, report, output):
    """Mode-3 figure: four-way bars, BC closed-loop vs baseline, BC loss curve."""
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), layout="constrained")
    rows = report["four_way_comparison"]
    labels = ["基线\n(第7课)", "纯PPO\n(29课)", "DAPG\n(32课)", "GoExplore\n+BC(本课)"]
    colors = ("#64748b", "#b91c1c", "#7c3aed", "#2563eb")
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    bars = axes[0, 0].bar(
        labels,
        [s / t * 100 if t else 0.0 for s, t in zip(successes, totals, strict=True)],
        color=colors,
        width=0.62,
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
        title="四种过崖方式：下方初态验收（第 7 课口径）",
    )
    axes[0, 0].tick_params(axis="x", labelsize=7.5)

    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as data:
        baseline = data["baseline_states"]
        det_states = data["det_states"] if report["phase2"]["run"] else None
        case = report["failure_analysis"]["featured_case"]
        case_states = data["case0_states"] if case is not None else None
        bc_curve = data["bc_loss_curve"] if report["phase2"]["run"] else None
    axes[0, 1].plot(
        np.arange(len(baseline)) * dt,
        np.cos(baseline[:, 1] - ref_theta),
        "--",
        color="#64748b",
        label="第 7 课基线",
    )
    if det_states is not None:
        det = report["phase2"]["deterministic"]
        det_arrival = det["first_arrival_s"]
        det_text = f"首达 {det_arrival:.2f} s" if det_arrival is not None else "未进入直立区"
        axes[0, 1].plot(
            np.arange(len(det_states)) * dt,
            np.cos(det_states[:, 1] - ref_theta),
            color="#2563eb",
            label=f"BC 策略（均值动作，{det_text}）",
        )
        axes[0, 1].legend(fontsize=8, loc="lower right")
    axes[0, 1].axhspan(-1, 0, alpha=0.06, color="orange")
    axes[0, 1].set(
        ylabel="杆端相对高度",
        ylim=(-1.1, 1.1),
        xlabel="仿真时间（s）",
        title="鲁棒化策略闭环：同一下方初态（无状态重置）",
    )
    if bc_curve is not None:
        axes[1, 0].plot(
            np.arange(1, len(bc_curve) + 1), np.maximum(bc_curve, 1e-8), color="#2563eb"
        )
        axes[1, 0].set(
            xlabel="BC 轮次",
            ylabel="教师 MSE（对数轴）",
            yscale="log",
            title=f"BC 损失：{bc_curve[0]:.3f} → {bc_curve[-1]:.3f}"
            f"（{report['phase2']['teacher_pairs']} 对）",
        )
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "阶段一未捕获稳定带，阶段二未运行",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set(title="BC 损失：无")
    if case_states is not None:
        case = report["failure_analysis"]["featured_case"]
        axes[1, 1].plot(
            np.arange(len(case_states)) * dt,
            np.cos(case_states[:, 1] - ref_theta),
            color="#b91c1c",
        )
        axes[1, 1].axhspan(-1, 0, alpha=0.06, color="orange")
        axes[1, 1].set(
            ylabel="杆端相对高度",
            xlabel="仿真时间（s）",
            title=f"失败案例（评估种子 {case['eval_seed']}）：{failure_label_cn(case['failure_reason'])}",
        )
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "本记录没有失败回合",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set(title="失败案例：无")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segments", type=int, default=MAX_SEGMENTS)
    parser.add_argument("--segment-steps", type=int, default=SEGMENT_STEPS)
    parser.add_argument("--eval-seeds", type=int, default=EVAL_SEEDS)
    parser.add_argument("--bc-epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--angle-bins", type=int, default=ANGLE_BINS)
    parser.add_argument("--cart-bins", type=int, default=CART_BINS)
    parser.add_argument("--captures-kept", type=int, default=CAPTURES_KEPT)
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)  # keep stdout a pure JSON document

    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            segments=args.segments,
            segment_steps=args.segment_steps,
            eval_seed_count=args.eval_seeds,
            bc_epochs=args.bc_epochs,
            angle_bins=args.angle_bins,
            cart_bins=args.cart_bins,
            captures_kept=args.captures_kept,
            log=log,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "guard": {
                    key: value
                    for key, value in report["guard"].items()
                    if key.startswith("bitwise")
                },
                "coverage": f"{report['phase1']['cells_occupied']}/{report['phase1']['cells_total']}",
                "first_upright_step": report["phase1"]["first_upright_step"],
                "first_capture_step": report["phase1"]["first_capture_step"],
                "captures": report["phase1"]["capture_count"],
                "teacher_pairs": report["phase2"]["teacher_pairs"],
                "bc": {
                    "first": report["phase2"]["bc_loss_first"],
                    "last": report["phase2"]["bc_loss_last"],
                },
                "baseline": f"{report['baseline']['successes']}/{report['baseline']['episodes']}",
                "goexplore_bc": (
                    f"{report['phase2']['stochastic']['successes']}"
                    f"/{report['phase2']['stochastic']['episodes']}"
                ),
                "zero_to_one": report["hypothesis"]["phase2_zero_to_one"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
