"""Lesson 41: staged minimal verification - base first, then minimal corrections.

Lesson 38 separated the combo architecture's two abilities: the lesson-7 base
(energy shaping + LQR hysteresis) is protective, but every learned residual so
far destroys the strict settled tail.  Before upgrading any algorithm, this
lesson walks the verification route with the simplest possible methods, one
stage at a time, each gate passed before the next one runs:

stage 1 (no training): re-run the frozen lesson-7 controller 20 times on the
    exact down start with the lesson-7 acceptance (20/20 at 4.76 s expected)
    and record the handoff segment - the per-step transition from |alpha|
    < 0.3 rad into the 0.01 rad tolerance band - as the reference for the
    learning stages;
stage 2: a 5-parameter linear handoff correction delta_u = W s + b with
    s = [x, alpha_wrapped, v, omega], supervised-initialized to delta_u = 0 on
    the base handoff pairs (least squares against zero targets), then updated
    once from the shortest settled rollout of a light +/-0.05 N exploration
    inside the handoff window; |delta_u| <= 5 N everywhere (1.7% of the base
    300 N motor limit);
stage 3 (only if stage 2 improves the settled time at 20/20): the same
    protocol with a 2x64 MLP correction, plus a +/-25 N budget variant as the
    amplified-limit control.  If stage 2 does not improve, stage 3 is skipped
    by protocol and reported as such.

Control law: u = clip(u_base + delta_u / gear, +/-3); acceptance is the
lesson-7 recovery_metrics reused verbatim on the exact resting down start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.bc_imitation import AdamOptimizer
from embodied_learning.experiments.ppo_swingup import MLPTower
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    HybridSwingupController,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)

EXPERIMENT = "staged_verification_lesson41"
SCHEMA_VERSION = 1

DOWN_ANGLE_DEG = -180.0
EVAL_EPISODE_STEPS = 750  # the lesson-7 30 s horizon
GEAR = 100.0  # actuator gear: normalized input [-3, 3] maps to +/-300 N
CONTROL_LIMIT = 3.0

CAPTURE_ANGLE_RAD = 0.3  # the base LQR capture condition (lesson-7 parameter)
HANDOFF_TOLERANCE_RAD = 0.01  # the settled angle tolerance: where the window ends

EXPLORE_ROLLOUTS = 10
EXPLORE_NOISE_N = 0.05
RESIDUAL_LIMIT_N = 5.0
RESIDUAL_LIMIT_LARGE_N = 25.0
EVAL_REPEATS = 20
NOT_DEGRADE_RATE = 0.9  # lesson-38 convention: >= 18/20 per group

MLP_HIDDEN = (64, 64)
MLP_EPOCHS = 400
MLP_LR = 1e-3

SEED_OFFSET_LINEAR = 4100
SEED_OFFSET_MLP5 = 4101
SEED_OFFSET_MLP25 = 4102
SEED_OFFSET_MLP_INIT = 9000

GROUPS = {
    "B_linear": {
        "archive": "lin",
        "offset": SEED_OFFSET_LINEAR,
        "label": "base + linear (5 params)",
    },
    "C_mlp": {"archive": "mlp5", "offset": SEED_OFFSET_MLP5, "label": "base + MLP 2x64"},
    "D_mlp_large": {
        "archive": "mlp25",
        "offset": SEED_OFFSET_MLP25,
        "label": "base + MLP 2x64, +/-25 N",
    },
}
GROUP_ORDER = ("A_base", "B_linear", "C_mlp", "D_mlp_large")


@dataclass(frozen=True)
class StagedConfig:
    exploration_rollouts: int = EXPLORE_ROLLOUTS
    eval_repeats: int = EVAL_REPEATS
    residual_limit_n: float = RESIDUAL_LIMIT_N
    residual_limit_large_n: float = RESIDUAL_LIMIT_LARGE_N
    exploration_noise_n: float = EXPLORE_NOISE_N
    mlp_hidden: tuple[int, ...] = MLP_HIDDEN
    mlp_epochs: int = MLP_EPOCHS
    mlp_lr: float = MLP_LR

    def __post_init__(self):
        integers = (self.exploration_rollouts, self.eval_repeats, self.mlp_epochs)
        if not all(isinstance(value, int) and value > 0 for value in integers):
            raise ValueError("rollout/repeat/epoch counts must be positive integers")
        if not np.isfinite(self.mlp_lr) or self.mlp_lr <= 0:
            raise ValueError("the MLP learning rate must be finite and positive")
        if not 1 <= self.eval_repeats <= 100:
            raise ValueError("eval_repeats must be an integer in [1, 100]")
        for limit in (self.residual_limit_n, self.residual_limit_large_n):
            if not np.isfinite(limit) or not 0 < limit <= 300.0:
                raise ValueError("residual limits must be finite in (0, 300] N")
        if self.residual_limit_large_n <= self.residual_limit_n:
            raise ValueError("the amplified limit (group D) must exceed the base limit")
        if not np.isfinite(self.exploration_noise_n) or not 0 < self.exploration_noise_n <= (
            self.residual_limit_n
        ):
            raise ValueError("exploration noise must be finite in (0, residual_limit_n] N")
        if any(units < 1 for units in self.mlp_hidden):
            raise ValueError("MLP hidden layer sizes must be positive")


# ----------------------------------------------------------------- handoff
def correction_features(state, reference_theta):
    """The 4-dim correction input [x, alpha_wrapped, v, omega].

    The wrapped angle follows the base controller's own feedback convention
    (angles measured from upright, wrapped for feedback only), so the
    correction is invariant to how many full rotations preceded the capture.
    """
    state = np.asarray(state, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("state must have four finite components")
    return np.array(
        [state[0], float(wrap_angle(state[1] - reference_theta)), state[2], state[3]], dtype=float
    )


def handoff_window(modes, states, reference_theta):
    """[capture, end) step window of one rollout, or None when never captured.

    capture: the first step the base controller acts in balance (LQR) mode,
    i.e. |alpha| < 0.3 rad; end: the first subsequent step whose pre-action
    wrapped angle error is within the 0.01 rad tolerance.  The window is the
    transition |alpha|: 0.3 rad -> 0.01 rad the lesson learns to correct.
    """
    modes = np.asarray(modes)
    balance = modes == "balance"
    if not balance.any():
        return None
    capture = int(np.argmax(balance))
    alpha = wrap_angle(np.asarray(states)[:, 1] - reference_theta)
    inside = np.flatnonzero(np.abs(alpha[capture:]) <= HANDOFF_TOLERANCE_RAD)
    end = capture + int(inside[0]) if len(inside) else len(modes)
    return capture, end


# --------------------------------------------------------------- correctors
class LinearCorrector:
    """delta_u = clip(W s + b, +/- limit_n); exactly five parameters."""

    parameter_count = 5

    def __init__(self, weight, bias, limit_n):
        self.weight = np.asarray(weight, dtype=float).reshape(1, 4)
        self.bias = float(bias)
        self.limit_n = float(limit_n)
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias):
            raise ValueError("linear correction parameters must be finite")
        if not np.isfinite(self.limit_n) or self.limit_n <= 0:
            raise ValueError("the residual limit must be finite and positive")

    def residual(self, features):
        """Commanded residual in newtons, inside the handoff limit."""
        value = float(features @ self.weight[0]) + self.bias
        return float(np.clip(value, -self.limit_n, self.limit_n))

    def fit_from(self, features, targets):
        """Refit W, b by least squares on (s, delta_u) pairs; returns self."""
        weight, bias = fit_linear_correction(features, targets)
        self.weight, self.bias = weight, bias
        return self

    def arrays(self):
        return {"weight": self.weight.copy(), "bias": np.array([self.bias])}


class MLPCorrector:
    """2x64 ReLU MLP residual with a zero-initialized output layer."""

    def __init__(self, hidden, seed, limit_n):
        self.trunk = MLPTower(4, tuple(hidden), 1, seed)
        self.trunk.weights[-1][:] = 0.0  # supervised zero init: delta_u == 0 exactly
        self.limit_n = float(limit_n)
        if not np.isfinite(self.limit_n) or self.limit_n <= 0:
            raise ValueError("the residual limit must be finite and positive")

    @property
    def parameter_count(self):
        return int(sum(w.size + b.size for w, b in zip(self.trunk.weights, self.trunk.biases)))

    def residual(self, features):
        value = float(self.trunk.forward(np.asarray(features, dtype=float)[None, :])[0][0, 0])
        return float(np.clip(value, -self.limit_n, self.limit_n))

    def fit_from(self, features, targets, *, epochs, lr):
        """Full-batch Adam on the MSE against the best rollout's residuals."""
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float).reshape(-1, 1)
        if len(features) == 0 or len(features) != len(targets):
            raise ValueError("fit needs nonempty aligned (s, delta_u) pairs")
        parameters = [*self.trunk.weights, *self.trunk.biases]
        optimizer = AdamOptimizer(parameters, lr=lr)
        history = np.empty(epochs, dtype=float)
        for epoch in range(epochs):
            output, cache = self.trunk.forward(features)
            error = output - targets
            history[epoch] = float(np.mean(error * error))
            gradient = 2.0 * error / len(features)
            grad_w, grad_b = self.trunk.backward(cache, gradient)
            optimizer.step(parameters, [*grad_w, *grad_b])
        return history

    def arrays(self):
        payload = {}
        for index, (weight, bias) in enumerate(zip(self.trunk.weights, self.trunk.biases)):
            payload[f"weight_{index}"] = weight.copy()
            payload[f"bias_{index}"] = bias.copy()
        return payload


def fit_linear_correction(features, targets):
    """Least-squares W (1x4), b for targets = W s + b.

    With the supervised zero targets this returns exact zeros - the initial
    correction is delta_u = 0 and the corrected rollout is the base rollout
    bit for bit.
    """
    features = np.asarray(features, dtype=float).reshape(-1, 4)
    targets = np.asarray(targets, dtype=float).ravel()
    if len(features) != len(targets):
        raise ValueError("features and targets must align")
    if len(features) == 0:
        return np.zeros((1, 4)), 0.0
    design = np.concatenate([features, np.ones((len(features), 1))], axis=1)
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return solution[:4].reshape(1, 4), float(solution[4])


# ----------------------------------------------------------------- rollouts
def run_corrected_episode(
    corrector, design, *, horizon=EVAL_EPISODE_STEPS, noise_rng=None, noise_n=0.0
):
    """One down-start episode: frozen base + limited correction in the window.

    The window is latched from the corrected run itself: it opens at the first
    step the base enters balance mode and closes at the first subsequent step
    whose pre-action wrapped angle error is within 0.01 rad, for good.
    Outside the window (and whenever corrector is None) the residual is zero,
    which reproduces the lesson-7 run bit for bit.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if corrector is None and (noise_rng is not None or noise_n > 0):
        raise ValueError("exploration noise requires a corrector")
    env = make_swingup_environment(max_episode_steps=horizon)
    try:
        env.reset(seed=0)
        state = design.controller.reference.copy()
        state[1] += np.deg2rad(DOWN_ANGLE_DEG)
        env.unwrapped.set_state(state[:2], state[2:])
        controller = HybridSwingupController(env.unwrapped.model, design)
        reference_theta = design.controller.reference[1]
        states, controls, base_controls = [state.copy()], [], []
        residuals, actives, modes, forces = [], [], [], []
        captured, entered = False, False
        terminated = truncated = False
        failure_reason = ""
        for _ in range(horizon):
            alpha = float(wrap_angle(state[1] - reference_theta))
            base_action = controller.action(state)  # also updates controller.mode
            if controller.mode == "balance":
                captured = True
            if captured and abs(alpha) <= HANDOFF_TOLERANCE_RAD:
                entered = True
            active = captured and not entered and controller.mode == "balance"
            residual_n = 0.0
            if active:
                residual_n = corrector.residual(correction_features(state, reference_theta))
                if noise_rng is not None:
                    residual_n += float(noise_rng.uniform(-noise_n, noise_n))
            total = float(
                np.clip(float(base_action[0]) + residual_n / GEAR, -CONTROL_LIMIT, CONTROL_LIMIT)
            )
            command = np.array([total], dtype=np.float32)
            state, _, terminated, truncated, info = env.step(command)
            states.append(state.copy())
            controls.append(float(command[0]))
            base_controls.append(float(base_action[0]))
            residuals.append(residual_n)
            actives.append(active)
            modes.append(controller.mode)
            forces.append(0.0)
            failure_reason = info["failure_reason"]
            if terminated or truncated:
                break
        arrays = {
            "states": np.asarray(states, dtype=np.float64),
            "controls": np.asarray(controls, dtype=np.float64),
            "base_controls": np.asarray(base_controls, dtype=np.float64),
            "residuals_n": np.asarray(residuals, dtype=np.float64),
            "active": np.asarray(actives, dtype=bool),
            "applied_force_n": np.asarray(forces, dtype=np.float64),
            "scheduled_force_n": np.zeros(horizon),
            "modes": np.asarray(modes),
            "end_flags": np.array([terminated, truncated]),
        }
        return arrays, failure_reason
    finally:
        env.close()


def episode_summary(arrays, failure_reason, reference, dt):
    """Lesson-7 acceptance plus this lesson's window and residual measures."""
    metrics = recovery_metrics(arrays, {"failure_reason": failure_reason}, reference, dt)
    residuals = np.asarray(arrays["residuals_n"], dtype=float)
    active = np.asarray(arrays["active"], dtype=bool)
    window_residuals = np.abs(residuals[active])
    window = handoff_window(arrays["modes"], arrays["states"], reference[1])
    return {
        "recovered": bool(metrics["recovered"]),
        "terminated": bool(metrics["terminated"]),
        "settled_at_s": metrics["settled_at_s"],
        "capture_time_s": (window[0] * dt if window else None),
        "window_end_s": (window[1] * dt if window else None),
        "residual_mean_abs_n": float(window_residuals.mean()) if window_residuals.size else 0.0,
        "residual_max_abs_n": float(window_residuals.max()) if window_residuals.size else 0.0,
        "max_abs_cart_position_m": metrics["max_abs_cart_position_m"],
        "failure_reason": metrics["failure_reason"],
    }


def base_evaluations(design, reference, dt, repeats):
    """Stage 1: the frozen lesson-7 controller, run `repeats` times, no training."""
    records, trajectories, controls, modes = [], [], None, None
    for repeat in range(repeats):
        arrays, metadata = run_scenario(Scenario("down", f"down repeat {repeat}"), design)
        metrics = recovery_metrics(arrays, metadata, reference, dt)
        records.append(
            {
                "repeat": repeat,
                "recovered": bool(metrics["recovered"]),
                "terminated": bool(metrics["terminated"]),
                "settled_at_s": metrics["settled_at_s"],
            }
        )
        trajectories.append(arrays["states"])
        controls = arrays["controls"]
        modes = arrays["modes"]
    identical = all(np.array_equal(trajectories[0], states) for states in trajectories[1:])
    settled = [r["settled_at_s"] for r in records if r["recovered"] and not r["terminated"]]
    summary = {
        "controller": "lesson-7 HybridSwingupController (energy shaping + LQR), frozen, zero-shot",
        "episodes": repeats,
        "successes": len(settled),
        "median_settled_at_s": float(np.median(settled)) if settled else None,
        "deterministic_identical_repeats": bool(identical),
        "per_episode": records,
    }
    return summary, trajectories[0], controls, modes


def stage1_handoff(baseline_states, baseline_modes, baseline_controls, reference, dt):
    """Record the handoff segment of the base run as the learning reference."""
    reference_theta = reference[1]
    window = handoff_window(baseline_modes, baseline_states, reference_theta)
    if window is None:
        raise ValueError("the base run never captured the pole; stage 1 gate failed")
    capture, end = window
    if end <= capture:
        raise ValueError("the base captured inside the tolerance band; the handoff window is empty")
    states = baseline_states[capture : end + 1]
    features = np.asarray([correction_features(s, reference_theta) for s in states[:-1]])
    return {
        "definition": {
            "capture_condition": f"|alpha| < {CAPTURE_ANGLE_RAD} rad (first balance step)",
            "end_condition": f"first step with |alpha| <= {HANDOFF_TOLERANCE_RAD} rad",
            "active_window": "[capture, end): correction off before capture and after tolerance entry",
        },
        "capture_step": capture,
        "end_step": end,
        "capture_time_s": capture * dt,
        "window_start_s": capture * dt,
        "window_end_s": end * dt,
        "steps": int(end - capture),
        "start_state": features[0].tolist(),
        "start_state_note": "[x, alpha_wrapped, v, omega] at the window start",
        "features": features,
        "controls": np.asarray(baseline_controls[capture:end], dtype=float),
        "states": np.asarray(states, dtype=float),
        "alpha": wrap_angle(states[:, 1] - reference_theta),
        "times": np.arange(capture, end + 1) * dt,
    }


# ------------------------------------------------------------ group runners
def explore_rollouts(
    corrector, design, reference, dt, *, count, noise_n, master_seed, offset, log, label
):
    """Light-noise exploration inside the handoff window; no gradient steps."""
    episodes = []
    for index in range(count):
        rng = np.random.default_rng([master_seed, offset, index])
        arrays, reason = run_corrected_episode(corrector, design, noise_rng=rng, noise_n=noise_n)
        episodes.append(
            {"index": index, "arrays": arrays, **episode_summary(arrays, reason, reference, dt)}
        )
        if log is not None:
            log(
                f"{label} explore {index + 1}/{count}: settled "
                f"{episodes[-1]['settled_at_s']}, |du| mean {episodes[-1]['residual_mean_abs_n']:.3f} N"
            )
    return episodes


def pick_best_rollout(episodes):
    """The shortest settled time among successful explorations (or None)."""
    successful = [e for e in episodes if e["recovered"] and not e["terminated"]]
    if not successful:
        return None
    return min(
        successful,
        key=lambda e: float("inf") if e["settled_at_s"] is None else e["settled_at_s"],
    )


def window_pairs(arrays):
    """(features, commanded residuals) of the active window of one rollout."""
    states = arrays["states"][:-1]
    active = np.asarray(arrays["active"], dtype=bool)
    features = np.asarray([row for row, flag in zip(states, active, strict=True) if flag])
    targets = np.asarray(arrays["residuals_n"], dtype=float)[active]
    return features, targets


def evaluate_group(corrector, design, reference, dt, repeats):
    """`repeats` deterministic episodes from the exact down start (no noise)."""
    records, trajectories = [], []
    first_arrays, first_reason = None, ""
    for repeat in range(repeats):
        arrays, reason = run_corrected_episode(corrector, design)
        summary = episode_summary(arrays, reason, reference, dt)
        limit = corrector.limit_n
        summary["fraction_at_limit"] = float(
            np.mean(np.abs(arrays["residuals_n"][arrays["active"]]) >= 0.95 * limit)
            if arrays["active"].any()
            else 0.0
        )
        records.append({"repeat": repeat, **summary})
        trajectories.append(arrays["states"])
        if first_arrays is None:
            first_arrays, first_reason = arrays, reason
    identical = all(np.array_equal(trajectories[0], states) for states in trajectories[1:])
    settled = [r["settled_at_s"] for r in records if r["recovered"] and not r["terminated"]]
    return (
        {
            "episodes": repeats,
            "successes": len(settled),
            "median_settled_at_s": float(np.median(settled)) if settled else None,
            "settled_at_s_per_episode": [r["settled_at_s"] for r in records],
            "deterministic_identical_repeats": bool(identical),
            "residual_mean_abs_n": records[0]["residual_mean_abs_n"],
            "residual_max_abs_n": records[0]["residual_max_abs_n"],
            "fraction_at_limit": records[0]["fraction_at_limit"],
            "capture_time_s": records[0]["capture_time_s"],
            "window_end_s": records[0]["window_end_s"],
            "failure_counts": {
                "terminated": sum(r["terminated"] for r in records),
                "timeout_without_settling": sum(
                    not r["terminated"] and not r["recovered"] for r in records
                ),
            },
        },
        first_arrays,
        first_reason,
    )


def improvement_verdict(evaluation, baseline):
    """Protocol gates: not_degrade (>= 90%) and improved (20/20 AND faster).

    improved = same successes as the baseline AND a strictly smaller median
    settled time.  With a deterministic controller every repeat coincides, so
    the median is that single settled time.
    """
    rate = evaluation["successes"] / evaluation["episodes"]
    not_degrade = rate >= NOT_DEGRADE_RATE
    improved = (
        evaluation["successes"] == evaluation["episodes"] == baseline["successes"]
        and evaluation["median_settled_at_s"] is not None
        and baseline["median_settled_at_s"] is not None
        and evaluation["median_settled_at_s"] < baseline["median_settled_at_s"] - 1e-9
    )
    return bool(not_degrade), bool(improved)


def run_linear_group(design, reference, dt, handoff, *, config, master_seed, log):
    """Stage 2: the 5-parameter linear handoff correction (group B)."""
    group = GROUPS["B_linear"]
    features = handoff["features"]
    weight, bias = fit_linear_correction(features, np.zeros(len(features)))
    if np.any(weight != 0.0) or bias != 0.0:
        raise ValueError("supervised zero-target fit must return exact zeros")
    corrector = LinearCorrector(weight, bias, config.residual_limit_n)
    episodes = explore_rollouts(
        corrector,
        design,
        reference,
        dt,
        count=config.exploration_rollouts,
        noise_n=config.exploration_noise_n,
        master_seed=master_seed,
        offset=group["offset"],
        log=log,
        label="B linear",
    )
    best = pick_best_rollout(episodes)
    update = {
        "rule": "one least-squares update on the best rollout's (s, commanded du) window pairs",
        "best_index": best["index"] if best else None,
        "best_settled_at_s": best["settled_at_s"] if best else None,
        "fit_pairs": 0,
    }
    if best is not None:
        pair_features, pair_targets = window_pairs(best["arrays"])
        corrector.fit_from(pair_features, pair_targets)
        update["fit_pairs"] = len(pair_targets)
    evaluation, eval_arrays, _reason = evaluate_group(
        corrector, design, reference, dt, config.eval_repeats
    )
    not_degrade, improved = improvement_verdict(evaluation, handoff["baseline"])
    if log is not None:
        log(
            f"B linear: {evaluation['successes']}/{evaluation['episodes']}, "
            f"settled {evaluation['median_settled_at_s']}, improved {improved}"
        )
    record = {
        "label": group["label"],
        "limit_n": config.residual_limit_n,
        "parameter_count": LinearCorrector.parameter_count,
        "parameters": {
            "W": corrector.weight.tolist(),
            "b": corrector.bias,
            "input": "[x, alpha_wrapped, v, omega]",
        },
        "training": {
            "supervised_init": {
                "method": "least squares on the base handoff (s, u_base) pairs, target du = 0",
                "pairs": len(features),
                "max_abs_residual_n": 0.0,
            },
            "exploration": {
                "rollouts": config.exploration_rollouts,
                "noise_n": config.exploration_noise_n,
                "region": "handoff window only",
                "seed_stream": f"default_rng([master, {group['offset']}, rollout])",
                "settled_times_s": [e["settled_at_s"] for e in episodes],
                "recovered": [e["recovered"] for e in episodes],
                "successful_rollouts": sum(
                    e["recovered"] and not e["terminated"] for e in episodes
                ),
            },
            "update": update,
        },
        "evaluation": evaluation,
        "not_degrade": not_degrade,
        "improved": improved,
    }
    archive = {
        **{f"lin_explore_{key}": value for key, value in _explore_arrays(episodes).items()},
        "lin_weight": corrector.weight.copy(),
        "lin_bias": np.array([corrector.bias]),
        **_eval_arrays(eval_arrays, "lin"),
    }
    return record, archive


def _explore_arrays(episodes):
    return {
        "settled_s": np.asarray(
            [
                float(e["settled_at_s"]) if e["settled_at_s"] is not None else np.nan
                for e in episodes
            ]
        ),
        "recovered": np.asarray([e["recovered"] for e in episodes], dtype=bool),
    }


def _eval_arrays(arrays, prefix):
    return {
        f"{prefix}_eval_states": arrays["states"],
        f"{prefix}_eval_controls": arrays["controls"],
        f"{prefix}_eval_base_controls": arrays["base_controls"],
        f"{prefix}_eval_residuals_n": arrays["residuals_n"],
        f"{prefix}_eval_active": arrays["active"],
        f"{prefix}_eval_modes": arrays["modes"],
    }


def run_mlp_group(key, design, reference, dt, handoff, *, config, master_seed, log):
    """Stage 3: the 2x64 MLP handoff correction (groups C and D)."""
    group = GROUPS[key]
    prefix = group["archive"]
    limit_n = config.residual_limit_n if key == "C_mlp" else config.residual_limit_large_n
    corrector = MLPCorrector(
        config.mlp_hidden, [master_seed, group["offset"], SEED_OFFSET_MLP_INIT], limit_n
    )
    handoff_features = handoff["features"]
    zero_check = max(
        abs(corrector.residual(row)) for row in handoff_features[: min(8, len(handoff_features))]
    )
    if zero_check != 0.0:
        raise ValueError("MLP zero initialization must produce an exactly zero residual")
    episodes = explore_rollouts(
        corrector,
        design,
        reference,
        dt,
        count=config.exploration_rollouts,
        noise_n=config.exploration_noise_n,
        master_seed=master_seed,
        offset=group["offset"],
        log=log,
        label=f"{key} MLP",
    )
    best = pick_best_rollout(episodes)
    update = {
        "rule": "full-batch Adam on the best rollout's (s, commanded du) window pairs",
        "best_index": best["index"] if best else None,
        "best_settled_at_s": best["settled_at_s"] if best else None,
        "epochs": config.mlp_epochs,
        "lr": config.mlp_lr,
        "initial_loss": None,
        "final_loss": None,
    }
    if best is not None:
        pair_features, pair_targets = window_pairs(best["arrays"])
        history = corrector.fit_from(
            pair_features, pair_targets, epochs=config.mlp_epochs, lr=config.mlp_lr
        )
        update["initial_loss"] = float(history[0])
        update["final_loss"] = float(history[-1])
    evaluation, eval_arrays, _reason = evaluate_group(
        corrector, design, reference, dt, config.eval_repeats
    )
    not_degrade, improved = improvement_verdict(evaluation, handoff["baseline"])
    if log is not None:
        log(
            f"{key}: {evaluation['successes']}/{evaluation['episodes']}, "
            f"settled {evaluation['median_settled_at_s']}, improved {improved}"
        )
    record = {
        "label": group["label"],
        "limit_n": limit_n,
        "parameter_count": corrector.parameter_count,
        "parameters": {"hidden": list(config.mlp_hidden), "input": "[x, alpha_wrapped, v, omega]"},
        "training": {
            "supervised_init": {
                "method": "zero-initialized output layer; residual exactly 0 before exploration",
                "pairs": len(handoff_features),
                "max_abs_residual_n": 0.0,
            },
            "exploration": {
                "rollouts": config.exploration_rollouts,
                "noise_n": config.exploration_noise_n,
                "region": "handoff window only",
                "seed_stream": f"default_rng([master, {group['offset']}, rollout])",
                "settled_times_s": [e["settled_at_s"] for e in episodes],
                "recovered": [e["recovered"] for e in episodes],
                "successful_rollouts": sum(
                    e["recovered"] and not e["terminated"] for e in episodes
                ),
            },
            "update": update,
        },
        "evaluation": evaluation,
        "not_degrade": not_degrade,
        "improved": improved,
    }
    archive = {
        **{f"{prefix}_explore_{key2}": value for key2, value in _explore_arrays(episodes).items()},
        **{f"{prefix}_{name}": value for name, value in corrector.arrays().items()},
        **_eval_arrays(eval_arrays, prefix),
    }
    return record, archive


# ----------------------------------------------------------------- record
def expected_npz_keys(report):
    """Full archive key set implied by a summary (used by the demo loader)."""
    keys = {
        "baseline_states",
        "baseline_controls",
        "baseline_modes",
        "handoff_states",
        "handoff_controls",
        "handoff_alpha",
        "handoff_times",
        "lin_weight",
        "lin_bias",
    }
    prefixes = {"B_linear": "lin", "C_mlp": "mlp5", "D_mlp_large": "mlp25"}
    for key, prefix in prefixes.items():
        if report["groups"][key] is None:
            continue
        keys.update(
            {
                f"{prefix}_eval_states",
                f"{prefix}_eval_controls",
                f"{prefix}_eval_base_controls",
                f"{prefix}_eval_residuals_n",
                f"{prefix}_eval_active",
                f"{prefix}_eval_modes",
                f"{prefix}_explore_settled_s",
                f"{prefix}_explore_recovered",
            }
        )
        if key != "B_linear":
            layers = len(report["protocol"]["mlp_hidden"]) + 1
            keys.update(f"{prefix}_weight_{index}" for index in range(layers))
            keys.update(f"{prefix}_bias_{index}" for index in range(layers))
    return keys


def build_comparison_table(baseline, groups):
    """A/B/C/D rows; skipped groups keep their place with null measures."""
    rows = [
        {
            "group": "A",
            "key": "A_base",
            "label": "base (lesson-7)",
            "residual_limit_n": 0.0,
            **_measures(baseline, "ran"),
        }
    ]
    labels = {
        "B_linear": ("B", "base + linear 5-param"),
        "C_mlp": ("C", "base + MLP 2x64"),
        "D_mlp_large": ("D", "base + MLP 2x64, +/-25 N"),
    }
    for key, (letter, label) in labels.items():
        record = groups[key]
        status = "ran" if record is not None else "skipped_step3_gate"
        row = {
            "group": letter,
            "key": key,
            "label": label,
            "residual_limit_n": record["limit_n"] if record else None,
            **_measures(record["evaluation"] if record else None, status),
        }
        if record is not None:
            row["improved"] = record["improved"]
            row["not_degrade"] = record["not_degrade"]
        rows.append(row)
    return rows


def _measures(evaluation, status):
    if evaluation is None:
        return {
            "status": status,
            "episodes": None,
            "successes": None,
            "median_settled_at_s": None,
            "residual_mean_abs_n": None,
            "residual_max_abs_n": None,
        }
    return {
        "status": status,
        "episodes": evaluation["episodes"],
        "successes": evaluation["successes"],
        "median_settled_at_s": evaluation["median_settled_at_s"],
        "residual_mean_abs_n": evaluation.get("residual_mean_abs_n", 0.0),
        "residual_max_abs_n": evaluation.get("residual_max_abs_n", 0.0),
    }


def build_report(*, seed, config, design, reference, dt, baseline, handoff, groups, elapsed):
    mlp_ran = groups["C_mlp"] is not None
    linear_improved = groups["B_linear"]["improved"]
    mlp_improved = groups["C_mlp"]["improved"] if mlp_ran else None
    if not linear_improved:
        conclusion = (
            "base_optimal_no_headroom: the 5-parameter linear correction (|du| <= "
            f"{config.residual_limit_n:.0f} N) did not improve the settled time at "
            f"{groups['B_linear']['evaluation']['successes']}/{groups['B_linear']['evaluation']['episodes']}; "
            "step 3 was skipped by protocol - in this base-optimal setting even a minimal "
            "learned correction has no value to add"
        )
    elif not mlp_improved:
        conclusion = (
            "linear helped, MLP did not: the 2x64 MLP correction at both budgets failed to "
            "improve on the linear result"
        )
    else:
        conclusion = (
            "staged minimal verification improved the handoff: MLP corrections pass the "
            "improvement gate at 20/20 with a shorter settled time"
        )
    return {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "protocol": {
            "task": "lesson-7 full-rotation swing-up, exact resting down start, frozen base",
            "dt_s": dt,
            "reference_state": reference.tolist(),
            "control_law": (
                "u = clip(u_base + delta_u/gear, +/-3); delta_u = clip(f(s), +/- limit) "
                "(+ exploration noise inside the handoff window during training only)"
            ),
            "correction_input": "[x, alpha_wrapped, v, omega]",
            "handoff_definition": handoff["definition"],
            "residual_limits_n": {
                "B_linear": config.residual_limit_n,
                "C_mlp": config.residual_limit_n,
                "D_mlp_large": config.residual_limit_large_n,
            },
            "exploration": {
                "rollouts": config.exploration_rollouts,
                "noise_n": config.exploration_noise_n,
                "rule": "uniform +/- noise on the commanded residual, window steps only",
            },
            "eval_horizon_steps": EVAL_EPISODE_STEPS,
            "eval_horizon_s": EVAL_EPISODE_STEPS * dt,
            "eval_initial_state": "exact resting down start (reference angle -180 deg), no jitter",
            "settled_tolerances": [0.02, 0.01, 0.02, 0.02],
            "minimum_settled_tail_s": 2.0,
            "acceptance": (
                "lesson-7 recovery_metrics reused verbatim: all four wrapped state errors within "
                "tolerances for the final continuous tail >= 2 s; success = recovered with no "
                "physical failure; settled time counted from t=0"
            ),
            "gates": {
                "not_degrade_rate": NOT_DEGRADE_RATE,
                "improved": "successes == episodes == baseline successes AND median settled < baseline",
                "step3": "groups C and D run only when B improved",
            },
            "mlp_hidden": list(config.mlp_hidden),
            "mlp_epochs": config.mlp_epochs,
            "mlp_lr": config.mlp_lr,
            "seed_streams": {
                "linear_exploration": f"default_rng([master, {SEED_OFFSET_LINEAR}, rollout])",
                "mlp5_exploration": f"default_rng([master, {SEED_OFFSET_MLP5}, rollout])",
                "mlp25_exploration": f"default_rng([master, {SEED_OFFSET_MLP25}, rollout])",
                "mlp_init": f"default_rng([master, group_offset, {SEED_OFFSET_MLP_INIT}])",
            },
        },
        "step1_base": {**baseline, "handoff": handoff_public(handoff)},
        "groups": groups,
        "comparison_table": build_comparison_table(baseline, groups),
        "verdict": {
            "linear_improved": linear_improved,
            "mlp_ran": mlp_ran,
            "mlp_improved": mlp_improved,
            "conclusion": conclusion,
        },
        "training": {"wall_time_s_total": elapsed},
        "limitations": [
            (
                "The correction is selected from the single best of a few noisy rollouts and then "
                "fitted as a state function: this is best-of-noise selection, so any apparent "
                "settled-time gain is expected to be selection bias, not generalizable improvement."
            ),
            (
                "The handoff window is one latched transition of one deterministic start; no other "
                "initial states, disturbances or noise are covered."
            ),
            (
                "The correction input [x, alpha, v, omega] and the 5 N / 25 N budgets are single "
                "hand choices; no architecture, budget or window-definition sweep is performed."
            ),
            (
                "The base controller is frozen and deterministic, so all improvement statements are "
                "about one calibrated scenario repeated 20 times, not a robustness distribution."
            ),
            (
                "No claim is made about hardware, delays, model mismatch or tasks beyond this "
                "cart-pole swing-up."
            ),
        ],
    }


def handoff_public(handoff):
    """Summary-safe handoff record (arrays stay in the archive)."""
    return {
        "definition": handoff["definition"],
        "capture_step": handoff["capture_step"],
        "end_step": handoff["end_step"],
        "capture_time_s": handoff["capture_time_s"],
        "window_start_s": handoff["window_start_s"],
        "window_end_s": handoff["window_end_s"],
        "steps": handoff["steps"],
        "start_state": handoff["start_state"],
        "start_state_note": handoff["start_state_note"],
    }


def save_comparison(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    ref_theta = report["protocol"]["reference_state"][1]
    handoff = report["step1_base"]["handoff"]
    groups = report["groups"]
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as npz:
        baseline_states = npz["baseline_states"]
        baseline_controls = npz["baseline_controls"]
        handoff_alpha = npz["handoff_alpha"]
        handoff_times = npz["handoff_times"]
        lin_states = npz["lin_eval_states"]
        lin_controls = npz["lin_eval_controls"]
        lin_residuals = npz["lin_eval_residuals_n"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), layout="constrained")
    ts = np.arange(len(baseline_states)) * dt
    axes[0, 0].plot(
        ts, np.cos(baseline_states[:, 1] - ref_theta), "--", color="#64748b", label="base 20/20"
    )
    axes[0, 0].plot(
        np.arange(len(lin_states)) * dt,
        np.cos(lin_states[:, 1] - ref_theta),
        color="#2563eb",
        linewidth=1.1,
        label="base + linear",
    )
    axes[0, 0].axvspan(
        handoff["window_start_s"], handoff["window_end_s"], alpha=0.15, color="#f59e0b"
    )
    axes[0, 0].axhspan(-1, 0, alpha=0.08, color="orange")
    axes[0, 0].set(
        ylabel="pole tip height cos(alpha)",
        ylim=(-1.1, 1.1),
        xlabel="simulation time (s)",
        title="stage 1+2: swing-up with the handoff window shaded",
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(
        handoff_times, handoff_alpha, "o-", markersize=3, color="#b45309", label="base handoff"
    )
    for key, prefix, color in (("C_mlp", "mlp5", "#0f766e"), ("D_mlp_large", "mlp25", "#7c3aed")):
        if groups[key] is None:
            continue
        states = npz[f"{prefix}_eval_states"]
        window = handoff_window(npz[f"{prefix}_eval_modes"], states, ref_theta)
        if window is None:
            continue
        capture, end = window
        alpha = wrap_angle(states[capture : end + 1, 1] - ref_theta)
        axes[0, 1].plot(
            np.arange(capture, end + 1) * dt,
            alpha,
            "o-",
            markersize=2,
            alpha=0.75,
            color=color,
            label=groups[key]["label"],
        )
    for bound in (-CAPTURE_ANGLE_RAD, CAPTURE_ANGLE_RAD):
        axes[0, 1].axhline(bound, color="#64748b", linestyle="--", linewidth=0.9)
    for bound in (-HANDOFF_TOLERANCE_RAD, HANDOFF_TOLERANCE_RAD):
        axes[0, 1].axhline(bound, color="#b91c1c", linestyle=":", linewidth=0.9)
    axes[0, 1].set(
        xlabel="simulation time (s)",
        ylabel="wrapped angle error alpha (rad)",
        title="handoff window: 0.3 rad (dashed) -> 0.01 rad (dotted)",
    )
    axes[0, 1].legend(fontsize=7)
    edges = np.arange(len(baseline_controls) + 1) * dt
    axes[1, 0].stairs(baseline_controls * GEAR, edges, color="#64748b", label="base")
    axes[1, 0].stairs(lin_controls * GEAR, edges, color="#2563eb", alpha=0.8, label="base + linear")
    axes[1, 0].plot(
        np.arange(len(lin_residuals)) * dt,
        lin_residuals,
        color="#b45309",
        linewidth=1.0,
        label="residual du",
    )
    axes[1, 0].set(
        ylabel="motor force (N)",
        xlabel="simulation time (s)",
        title="motor input and the +/-5 N residual",
    )
    axes[1, 0].legend(fontsize=8)
    rows = [row for row in report["comparison_table"] if row["status"] == "ran"]
    labels = [f"{row['group']}: {row['label']}" for row in rows]
    successes = [row["successes"] for row in rows]
    totals = [row["episodes"] for row in rows]
    colors = ("#64748b", "#2563eb", "#0f766e", "#7c3aed")
    bars = axes[1, 1].bar(
        labels,
        [s / t * 100 for s, t in zip(successes, totals, strict=True)],
        color=[colors[index % len(colors)] for index in range(len(labels))],
        width=0.55,
    )
    for bar, row in zip(bars, rows, strict=True):
        settled = row["median_settled_at_s"]
        note = f"{row['successes']}/{row['episodes']}" + (
            f"\n{settled:.2f} s" if settled is not None else ""
        )
        axes[1, 1].annotate(
            note,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1, 1].set(
        ylabel="acceptance pass rate (%)",
        ylim=(0, 116),
        title="lesson-7 acceptance, exact down start",
    )
    axes[1, 1].tick_params(axis="x", labelsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_correction(path, report, output):
    configure_plot_font()
    dt = report["protocol"]["dt_s"]
    limit_n = report["protocol"]["residual_limits_n"]["B_linear"]
    ran = [
        (key, GROUPS[key]["archive"], GROUPS[key]["label"])
        for key in GROUPS
        if report["groups"][key]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout="constrained")
    colors = {"lin": "#2563eb", "mlp5": "#0f766e", "mlp25": "#7c3aed"}
    with np.load(Path(output) / "trajectories.npz", allow_pickle=False) as npz:
        for _key, prefix, label in ran:
            residuals = npz[f"{prefix}_eval_residuals_n"]
            axes[0].plot(
                np.arange(len(residuals)) * dt,
                residuals,
                linewidth=0.9,
                color=colors[prefix],
                label=label,
            )
        baseline_settled = report["step1_base"]["median_settled_at_s"]
        for _key, prefix, label in ran:
            settled = npz[f"{prefix}_explore_settled_s"]
            axes[1].plot(
                np.arange(len(settled)),
                settled,
                "o-",
                markersize=4,
                color=colors[prefix],
                label=label,
            )
    for bound in (-limit_n, limit_n):
        axes[0].axhline(bound, color="#b91c1c", linestyle=":", linewidth=0.9)
    axes[0].set(
        xlabel="simulation time (s)",
        ylabel="commanded residual du (N)",
        title="residual usage (dotted = the 5 N budget; groups C/D share it or use 25 N)",
    )
    axes[0].legend(fontsize=8)
    axes[1].axhline(baseline_settled, color="#64748b", linestyle="--", linewidth=1.0)
    axes[1].text(
        0.02,
        baseline_settled,
        f" base median {baseline_settled:.2f} s",
        va="bottom",
        fontsize=8,
        color="#64748b",
    )
    axes[1].set(
        xlabel="exploration rollout",
        ylabel="settled time (s)",
        title="exploration: best-of-noise selection per group (gaps = failed)",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- entry point
def run_experiment(output, *, seed=0, config=None, log=print):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    config = config or StagedConfig()
    if not isinstance(config, StagedConfig):
        raise TypeError("config must be a StagedConfig")
    started = time.perf_counter()
    design = design_swingup_lqr()
    reference, dt = design.controller.reference, design.dt
    if (
        abs(design.controller.control_limit - CONTROL_LIMIT) > 1e-12
        or abs(design.actuator_gear - GEAR) > 1e-12
    ):
        raise ValueError("actuator contract disagrees with the lesson-7 design")

    baseline, baseline_states, baseline_controls, baseline_modes = base_evaluations(
        design, reference, dt, config.eval_repeats
    )
    if baseline["successes"] != baseline["episodes"]:
        raise ValueError("stage 1 gate failed: the base no longer passes every repeat")
    handoff = stage1_handoff(baseline_states, baseline_modes, baseline_controls, reference, dt)
    handoff["baseline"] = baseline
    if log is not None:
        log(
            f"stage 1 base: {baseline['successes']}/{baseline['episodes']}, "
            f"median {baseline['median_settled_at_s']} s, window "
            f"{handoff['window_start_s']:.2f}-{handoff['window_end_s']:.2f} s "
            f"({handoff['steps']} steps)"
        )

    linear_record, linear_archive = run_linear_group(
        design, reference, dt, handoff, config=config, master_seed=seed, log=log
    )
    groups = {"B_linear": linear_record, "C_mlp": None, "D_mlp_large": None}
    archive = {
        "baseline_states": baseline_states,
        "baseline_controls": baseline_controls,
        "baseline_modes": baseline_modes,
        "handoff_states": handoff["states"],
        "handoff_controls": handoff["controls"],
        "handoff_alpha": handoff["alpha"],
        "handoff_times": handoff["times"],
        **linear_archive,
    }
    if linear_record["improved"]:
        for key in ("C_mlp", "D_mlp_large"):
            record, group_archive = run_mlp_group(
                key, design, reference, dt, handoff, config=config, master_seed=seed, log=log
            )
            groups[key] = record
            archive.update(group_archive)

    elapsed = time.perf_counter() - started
    report = build_report(
        seed=seed,
        config=config,
        design=design,
        reference=reference,
        dt=dt,
        baseline=baseline,
        handoff=handoff,
        groups=groups,
        elapsed=elapsed,
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_comparison(output / "comparison.png", report, output)
    save_correction(output / "correction.png", report, output)
    return report


def _log_stderr(message):
    """Progress goes to stderr so stdout stays a single JSON document."""
    print(message, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration-rollouts", type=int, default=EXPLORE_ROLLOUTS)
    parser.add_argument("--eval-repeats", type=int, default=EVAL_REPEATS)
    parser.add_argument("--residual-limit", type=float, default=RESIDUAL_LIMIT_N)
    parser.add_argument("--residual-limit-large", type=float, default=RESIDUAL_LIMIT_LARGE_N)
    parser.add_argument("--noise-n", type=float, default=EXPLORE_NOISE_N)
    parser.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS)
    args = parser.parse_args()
    config = StagedConfig(
        exploration_rollouts=args.exploration_rollouts,
        eval_repeats=args.eval_repeats,
        residual_limit_n=args.residual_limit,
        residual_limit_large_n=args.residual_limit_large,
        exploration_noise_n=args.noise_n,
        mlp_epochs=args.mlp_epochs,
    )
    try:
        report = run_experiment(args.output, seed=args.seed, config=config, log=_log_stderr)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "base": {
                    "successes": report["step1_base"]["successes"],
                    "median_settled_at_s": report["step1_base"]["median_settled_at_s"],
                },
                "linear": {
                    "successes": report["groups"]["B_linear"]["evaluation"]["successes"],
                    "episodes": report["groups"]["B_linear"]["evaluation"]["episodes"],
                    "median_settled_at_s": report["groups"]["B_linear"]["evaluation"][
                        "median_settled_at_s"
                    ],
                    "improved": report["groups"]["B_linear"]["improved"],
                },
                "step3_ran": report["verdict"]["mlp_ran"],
                "mlp": report["verdict"]["mlp_improved"],
                "conclusion": report["verdict"]["conclusion"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
