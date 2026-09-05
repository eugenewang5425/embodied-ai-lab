"""Lesson 28: behavioural cloning of the lesson-13 feedforward+PD expert.

The lesson-13 expert (reference inverse-dynamics feedforward + unchanged joint
PD) drives the 2R arm along 25 frozen seed-400 paths (4 s movement + 3 s hold).
Every control step yields one supervised pair: current state [q1, q2, dq1, dq2]
-> applied motor torque. A pure-numpy MLP (2 hidden layers x 64 ReLU units,
linear output) is trained on those pairs with hand-written backpropagation and
a hand-written Adam update - no torch, no sklearn.

The policy is deliberately "ignorant": it sees only the current joint state,
never the reference, the goal or the clock. The expert it imitates DOES read
the planned reference (both its feedforward and its PD term). This caliber
difference is part of the lesson and is reported, not hidden.

Checks (same conditions throughout, expert re-verified first):
(1) expert re-verification on the frozen lesson-10/13 manifest must reproduce
    the lesson-13 all-pass record before any cloning happens (hard gate);
(2) data scaling: per path 10/50/200/all recorded steps x 3 data seeds ->
    train -> closed-loop rollout with the BC torque DIRECTLY driving the motors
    (no PD safety net - a different control structure than the expert, stated
    as such) on the same 25 paths (in-sample) and on 25 fresh seed-402 paths
    (generalization), scored by the lesson-13 acceptance criteria;
(3) training MSE vs closed-loop success (compounding error / distribution
    shift made visible); featured episodes (best BC episode of the largest
    data budget plus the first in-sample/generalization failure) retain both
    the BC and the expert trajectory of the same path for the demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.arm_path import (
    ReferenceSpeedError,
    generate_reference,
    segment_distance,
    terminal_window_diagnostics,
)
from embodied_learning.experiments.arm_feedforward import load_timing_source
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import generate_manifest
from embodied_learning.planar_arm import (
    KD,
    KP,
    LENGTHS,
    MODEL_PATH,
    TORQUE_LIMIT,
    ArmSimulation,
    angle_error,
    inverse_kinematics,
)
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "bc_imitation_2r"
EXPERT_PATH_SEED = 400  # byte-identical to the frozen lesson-10/13 manifest
GENERALIZATION_PATH_SEED = 402  # same sampling ranges, all 25 plans executable
MOVEMENT_S, HOLD_S = 4.0, 3.0
STATE_INPUTS = 4  # [q1, q2, dq1, dq2]; no reference, no goal, no clock
HIDDEN = (64, 64)
SAMPLE_SIZES = (10, 50, 200, 0)  # steps per path; 0 = every recorded step (last entry)
DATA_SEEDS = 3
EPOCHS = 400
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
ADAM_BETA1, ADAM_BETA2, ADAM_EPS = 0.9, 0.999, 1e-8
PATH_GATE_MM = 2.0
TERMINAL_WINDOW_S = 0.5
INIT_SEED_OFFSET, SHUFFLE_SEED_OFFSET, DATA_SEED_OFFSET = 7000, 9000, 1000

BASE_NPZ_KEYS = (
    "expert_train_x",
    "expert_train_y",
    "expert_gen_x",
    "expert_gen_y",
    "sample_sizes",
    "success_train",
    "success_gen",
    "max_cross_train_mm",
    "max_cross_gen_mm",
    "final_train_mse",
    "full_train_mse",
    "full_gen_mse",
    "loss_curves",
)


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def manifest_bytes(manifest):
    return (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )


def sample_step_indices(rng, size, total):
    """Sorted without-replacement step sample; size 0 means every step."""
    if size == 0 or size == total:
        return np.arange(total)
    if not 1 <= size < total:
        raise ValueError(f"sample size must be 0 or in [1, {total})")
    return np.sort(rng.choice(total, size=size, replace=False))


class MLP:
    """Numpy MLP: ReLU hidden layers, linear output, hand-written backprop."""

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
        return activations, preacts

    def predict(self, x):
        return self.forward(x)[0][-1]

    def loss(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("Expected nonempty aligned 2-D batches")
        difference = self.forward(x)[0][-1] - y
        return float(np.mean(difference * difference))

    def loss_and_gradients(self, x, y):
        """Mean squared error over every output component; gradients dL/dW, dL/db."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("Expected nonempty aligned 2-D batches")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Nonfinite training data")
        activations, preacts = self.forward(x)
        difference = activations[-1] - y
        loss = float(np.mean(difference * difference))
        delta = 2.0 * difference / difference.size
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        for index in reversed(range(len(self.weights))):
            grad_weights[index] = activations[index].T @ delta
            grad_biases[index] = delta.sum(axis=0)
            if index:
                delta = (delta @ self.weights[index].T) * (preacts[index - 1] > 0.0)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grad_weights):
            raise ValueError("Nonfinite loss or gradients; training diverged")
        return loss, grad_weights, grad_biases


class AdamOptimizer:
    """Hand-written Adam; parameter arrays are updated strictly in place."""

    def __init__(
        self, parameters, lr=LEARNING_RATE, beta1=ADAM_BETA1, beta2=ADAM_BETA2, eps=ADAM_EPS
    ):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, parameters, gradients):
        if len(parameters) != len(self.m) or len(gradients) != len(self.m):
            raise ValueError("Parameter/gradient count mismatch")
        self.t += 1
        for parameter, gradient, m, v in zip(parameters, gradients, self.m, self.v, strict=True):
            m *= self.beta1
            m += (1.0 - self.beta1) * gradient
            v *= self.beta2
            v += (1.0 - self.beta2) * gradient * gradient
            m_hat = m / (1.0 - self.beta1**self.t)
            v_hat = v / (1.0 - self.beta2**self.t)
            parameter -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def train_bc(x, y, *, epochs, batch_size, lr, init_seed, shuffle_seed):
    net = MLP(x.shape[1], HIDDEN, y.shape[1], init_seed)
    parameters = [*net.weights, *net.biases]
    optimizer = AdamOptimizer(parameters, lr=lr)
    rng = np.random.default_rng(shuffle_seed)
    history = np.empty(epochs, dtype=float)
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total, seen = 0.0, 0
        for start in range(0, len(x), batch_size):
            batch = order[start : start + batch_size]
            loss, grad_weights, grad_biases = net.loss_and_gradients(x[batch], y[batch])
            optimizer.step(parameters, [*grad_weights, *grad_biases])
            total += loss * len(batch)
            seen += len(batch)
        history[epoch] = total / seen
    return net, history


def plan_path(specification, dt):
    """The shared geometric plan; BC rollouts score against the same line."""
    return generate_reference(
        "waypoint_ik",
        specification["initial_q_rad"],
        specification["target_m"],
        dt=dt,
        move_seconds=MOVEMENT_S,
        hold_seconds=HOLD_S,
    )


def bc_rollout(policy, specification, reference, dt):
    """Pure BC: the network torque drives the motors directly, no PD, no FF."""
    sim = ArmSimulation()
    state = sim.reset(specification["initial_q_rad"])
    states, points, requested, applied = [state], [sim.points()], [], []
    failure = ""
    for _ in range(len(reference["dq_reference"])):
        command = policy.predict(state[None, :STATE_INPUTS])[0]
        requested.append(np.asarray(command, dtype=float))
        state, applied_torque, failure = sim.step(command)
        states.append(state)
        points.append(sim.points())
        applied.append(applied_torque)
        if failure:
            break
    arrays = {
        "states": np.asarray(states),
        "points": np.asarray(points),
        "torques_nm": np.asarray(applied),
        "requested_torques_nm": np.asarray(requested),
        # Truncated to the executed horizon: a physically failed episode ends
        # before the plan does, so every array stays aligned for acceptance.
        "desired_points": np.asarray(reference["desired_points"], dtype=float)[: len(states)],
    }
    return arrays, failure


def evaluate_trajectory(arrays, target, move_seconds, dt, *, completed):
    """Lesson-13 acceptance, recomputed with the same functions and constants.

    Mirrors experiments.arm_path.run_path: 2 mm finite-segment gate during the
    movement window, then a continuous >=0.5 s tail (tip <=2 mm, joints
    <=0.01 rad, speeds <=0.02 rad/s) after the movement ends. A physical
    failure truncates the horizon; the truncated arrays stay aligned.
    """
    states, points = arrays["states"], arrays["points"]
    desired = np.asarray(arrays["desired_points"], dtype=float)
    tip = points[:, -1]
    times = np.arange(len(states)) * dt
    cross = segment_distance(tip, desired[0], target)
    movement = times <= move_seconds + 1e-10
    peak_cross = float(cross[movement].max() * 1000.0)
    goal = inverse_kinematics(target)[0]
    settled, timed_rms, terminal = None, float("nan"), {"violations": ["nonfinite_state"]}
    if np.isfinite(states).all() and np.isfinite(cross).all():
        tracking = np.linalg.norm(tip - desired, axis=1)
        within = (
            (np.linalg.norm(tip - target, axis=1) <= 0.002)
            & (np.max(np.abs(states[:, 2:]), axis=1) <= 0.02)
            & (np.max(np.abs(angle_error(goal, states[:, :2])), axis=1) <= 0.01)
        )
        tail = np.logical_and.accumulate(within[::-1])[::-1]
        candidates = np.flatnonzero(
            tail & (times[-1] - times >= TERMINAL_WINDOW_S - 1e-10) & (times >= move_seconds)
        )
        if len(candidates) and completed:
            settled = float(times[candidates[0]])
        timed_rms = float(np.sqrt(np.mean(tracking[movement] ** 2)) * 1000.0)
        terminal = terminal_window_diagnostics(
            states, points, target, goal, dt, move_seconds=move_seconds
        )
    return {
        "completed": bool(completed),
        "path_success": bool(completed and settled is not None and peak_cross <= PATH_GATE_MM),
        "endpoint_success": settled is not None,
        "max_cross_track_mm": peak_cross,
        "rms_timed_tracking_mm": timed_rms,
        "settled_after_movement_at_s": settled,
        "final_tip_error_mm": float(np.linalg.norm(tip[-1] - target) * 1000.0),
        "terminal_window": terminal,
        "clipped_steps": int(
            np.count_nonzero(np.abs(arrays["requested_torques_nm"]) > TORQUE_LIMIT)
        ),
    }


def collect_expert(manifest, dt):
    """Expert rollouts (lesson-13 controller) -> dataset + per-path records."""
    records, plans, xs, ys, arrays_by_id = [], [], [], [], {}
    for specification in manifest["trials"]:
        meta = {"id": specification["id"], "group": specification["group"]}
        try:
            arrays, case = run_path(
                "waypoint_ik",
                initial_q=specification["initial_q_rad"],
                target=specification["target_m"],
                move_seconds=MOVEMENT_S,
                hold_seconds=HOLD_S,
                controller="feedforward_pd",
            )
        except ReferenceSpeedError as exc:
            records.append({**meta, "status": "planning_rejected", "reason": str(exc)})
            continue
        plans.append(plan_path(specification, dt))
        arrays_by_id[specification["id"]] = arrays
        xs.append(arrays["states"][:-1, :STATE_INPUTS].copy())
        ys.append(arrays["torques_nm"].copy())
        records.append(
            {
                **meta,
                "status": "executed",
                "path_success": case["path_success"],
                "endpoint_success": case["endpoint_success"],
                "max_cross_track_mm": case["max_cross_track_mm"],
                "pairs": len(ys[-1]),
            }
        )
    if not xs:
        raise ValueError("No executable expert path in the manifest")
    return records, plans, np.concatenate(xs), np.concatenate(ys), arrays_by_id


def verify_expert(records):
    executed = [r for r in records if r["status"] == "executed"]
    return {
        "planned": len(records),
        "executed": len(executed),
        "path_successes": sum(r["path_success"] for r in executed),
        "planning_rejected": len(records) - len(executed),
        "protocol": (
            "run_path('waypoint_ik', move_seconds=4, controller='feedforward_pd'): "
            "the lesson-13 function, gains and configuration"
        ),
    }


def evaluate_policy(policy, plans, specifications, dt):
    successes, crosses, trajectories = [], [], []
    for specification, reference in zip(specifications, plans, strict=True):
        arrays, failure = bc_rollout(policy, specification, reference, dt)
        report = evaluate_trajectory(
            arrays,
            specification["target_m"],
            MOVEMENT_S,
            dt,
            completed=(not failure and len(arrays["torques_nm"]) == len(reference["dq_reference"])),
        )
        report["id"], report["group"] = specification["id"], specification["group"]
        report["target_m"] = list(map(float, specification["target_m"]))
        report["failure_reason"] = failure
        successes.append(report["path_success"])
        crosses.append(report["max_cross_track_mm"])
        trajectories.append((arrays, report))
    return np.asarray(successes, dtype=bool), np.asarray(crosses), trajectories


def pick_failures(candidates, size_count):
    """First failing train and gen episode, scanning from the largest data size."""
    chosen = []
    for label in ("train", "gen"):
        for size_index in reversed(range(size_count)):
            match = next(
                (c for c in candidates if c["set"] == label and c["size_index"] == size_index),
                None,
            )
            if match is not None:
                chosen.append(match)
                break
    return chosen


def pick_best(success_last, cross_last, arrays_pool):
    """Best train episode at the largest data budget: success first, min cross.

    success_last/cross_last: (data_seeds, episodes) at the largest size index;
    arrays_pool maps (seed_index, episode) -> the rollout arrays. Returns the
    episode key or None when nothing is finite (cannot happen in practice).
    """
    finite = np.isfinite(cross_last)
    ranked = [
        (bool(success_last[k, e]), cross_last[k, e], k, e)
        for k in range(success_last.shape[0])
        for e in range(success_last.shape[1])
        if finite[k, e] and (k, e) in arrays_pool
    ]
    if not ranked:
        return None
    success, cross, seed_index, episode = max(ranked, key=lambda item: (item[0], -item[1]))
    return seed_index, episode, bool(success), float(cross)


def run_experiment(
    output,
    *,
    seed=0,
    per_group=12,
    sample_sizes=SAMPLE_SIZES,
    data_seeds=DATA_SEEDS,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    source_results=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not isinstance(per_group, int) or not 1 <= per_group <= 100:
        raise ValueError("per_group must be an integer in [1, 100]")
    if not isinstance(data_seeds, int) or data_seeds < 1:
        raise ValueError("data_seeds must be a positive integer")
    if epochs < 1 or batch_size < 1 or not lr > 0:
        raise ValueError("epochs and batch_size must be positive and lr must be > 0")
    sizes = tuple(int(s) for s in sample_sizes)
    dt = ArmSimulation().dt
    steps_per_path = round((MOVEMENT_S + HOLD_S) / dt)
    if not sizes or len(set(sizes)) != len(sizes) or sizes[-1] != 0:
        raise ValueError("sample sizes must end with one 0 (all recorded steps)")
    ordered = tuple(steps_per_path if s == 0 else s for s in sizes)
    if any(not 1 <= s <= steps_per_path for s in ordered) or ordered != tuple(sorted(ordered)):
        raise ValueError(
            f"sample sizes must be ascending steps in [1, {steps_per_path}] with 0 last"
        )

    expert_manifest = generate_manifest(EXPERT_PATH_SEED, per_group)
    expert_raw = manifest_bytes(expert_manifest)
    if source_results is not None:
        _, frozen_raw, _ = load_timing_source(Path(source_results))
        if frozen_raw != expert_raw:
            raise ValueError("Regenerated expert manifest differs from the frozen source record")
    generalization_manifest = generate_manifest(GENERALIZATION_PATH_SEED, per_group)

    train_records, train_plans, train_x, train_y, expert_train_arrays = collect_expert(
        expert_manifest, dt
    )
    train_check = verify_expert(train_records)
    if train_check["planning_rejected"] or train_check["path_successes"] != train_check["executed"]:
        raise ValueError("Expert re-verification failed; expected the lesson-13 all-pass record")
    gen_records, gen_plans, gen_x, gen_y, expert_gen_arrays = collect_expert(
        generalization_manifest, dt
    )
    gen_check = verify_expert(gen_records)
    train_specs = [
        spec
        for spec, record in zip(expert_manifest["trials"], train_records, strict=True)
        if record["status"] == "executed"
    ]
    gen_specs = [
        spec
        for spec, record in zip(generalization_manifest["trials"], gen_records, strict=True)
        if record["status"] == "executed"
    ]

    output.mkdir(parents=True, exist_ok=False)
    (output / "expert_manifest.json").write_bytes(expert_raw)
    (output / "generalization_manifest.json").write_bytes(manifest_bytes(generalization_manifest))

    reference_net = MLP(STATE_INPUTS, HIDDEN, 2, 0)
    protocol = {
        "expert_seed": EXPERT_PATH_SEED,
        "generalization_seed": GENERALIZATION_PATH_SEED,
        "per_group": per_group,
        "expert_manifest_sha256": hashlib.sha256(expert_raw).hexdigest(),
        "generalization_manifest_sha256": hashlib.sha256(
            manifest_bytes(generalization_manifest)
        ).hexdigest(),
        "source_results": str(Path(source_results).resolve()) if source_results else None,
        "movement_s": MOVEMENT_S,
        "hold_s": HOLD_S,
        "dt_s": dt,
        "steps_per_path": steps_per_path,
        "lengths_m": LENGTHS.tolist(),
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "master_seed": seed,
        "network": {
            "input": "state[:4] = [q1, q2, dq1, dq2]; no reference, no goal, no clock (ignorant policy)",
            "hidden": list(HIDDEN),
            "activation": "relu",
            "output": 2,
            "loss": "mse",
            "optimizer": "adam (hand-written)",
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "init": "He normal, zero biases",
            "parameter_count": int(
                sum(w.size for w in reference_net.weights)
                + sum(b.size for b in reference_net.biases)
            ),
            "seed_streams": {
                "data_sampling": "default_rng([master, 1000 + k])",
                "network_init": "default_rng([master, 7000 + k]); shared across sizes of one k",
                "batch_order": "default_rng([master, 9000 + k])",
            },
        },
        "expert_controller": (
            "clip(FF(qref,dqref,ddqref) + PD(qref-q, dqref-dq), +/-0.25 Nm); the expert reads "
            "the planned reference, the BC policy does not"
        ),
        "bc_controller": (
            "tau = MLP(state[:4]); applied = clip(tau, +/-0.25 Nm) inside ArmSimulation.step; "
            "no PD, no feedforward, no reference"
        ),
        "sample_protocol": (
            f"per path, sorted without-replacement step indices from the {steps_per_path} recorded "
            "(state, torque) pairs; size 0 = all steps"
        ),
        "acceptance": (
            "lesson-13 criteria: completed 7 s, max finite-segment distance <= 2 mm during "
            "[0,4 s], continuous >=0.5 s terminal window (tip <=2 mm, joints <=0.01 rad, "
            "speeds <=0.02 rad/s); recomputed with segment_distance/terminal_window_diagnostics "
            "and cross-checked in tests against run_path"
        ),
    }

    path_slices = [
        slice(i * steps_per_path, (i + 1) * steps_per_path) for i in range(len(train_specs))
    ]
    success_train = np.zeros((len(sizes), data_seeds, len(train_specs)), dtype=bool)
    success_gen = np.zeros((len(sizes), data_seeds, len(gen_specs)), dtype=bool)
    cross_train = np.full(success_train.shape, np.nan)
    cross_gen = np.full(success_gen.shape, np.nan)
    final_train_mse = np.zeros((len(sizes), data_seeds))
    full_train_mse = np.zeros((len(sizes), data_seeds))
    full_gen_mse = np.zeros((len(sizes), data_seeds))
    loss_curves = np.zeros((len(sizes), data_seeds, epochs))
    candidates, largest_arrays = [], {}
    print(
        f"expert train {train_check['path_successes']}/{train_check['executed']}, "
        f"gen {gen_check['path_successes']}/{gen_check['executed']}, "
        f"dataset {len(train_x)} pairs",
        flush=True,
    )

    for size_index, size in enumerate(sizes):
        for seed_index in range(data_seeds):
            rng = np.random.default_rng([seed, DATA_SEED_OFFSET + seed_index])
            xs, ys = [], []
            for path_x, path_y in zip(
                (train_x[sl] for sl in path_slices),
                (train_y[sl] for sl in path_slices),
                strict=True,
            ):
                indices = sample_step_indices(rng, size, steps_per_path)
                xs.append(path_x[indices])
                ys.append(path_y[indices])
            x, y = np.concatenate(xs), np.concatenate(ys)
            net, history = train_bc(
                x,
                y,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                init_seed=[seed, INIT_SEED_OFFSET + seed_index],
                shuffle_seed=[seed, SHUFFLE_SEED_OFFSET + seed_index],
            )
            final_train_mse[size_index, seed_index] = history[-1]
            full_train_mse[size_index, seed_index] = float(
                np.mean((net.predict(train_x) - train_y) ** 2)
            )
            full_gen_mse[size_index, seed_index] = float(np.mean((net.predict(gen_x) - gen_y) ** 2))
            loss_curves[size_index, seed_index] = history
            train_result = evaluate_policy(net, train_plans, train_specs, dt)
            gen_result = evaluate_policy(net, gen_plans, gen_specs, dt)
            success_train[size_index, seed_index] = train_result[0]
            cross_train[size_index, seed_index] = train_result[1]
            success_gen[size_index, seed_index] = gen_result[0]
            cross_gen[size_index, seed_index] = gen_result[1]
            if size_index == len(sizes) - 1:
                for episode, (arrays, _) in enumerate(train_result[2]):
                    largest_arrays[(seed_index, episode)] = arrays
            for label, result in (("train", train_result), ("gen", gen_result)):
                _, _, trajectories = result
                for episode, (arrays, case) in enumerate(trajectories):
                    meta = {
                        "set": label,
                        "id": case["id"],
                        "group": case["group"],
                        "size_index": size_index,
                        "sample_size": size or steps_per_path,
                        "data_seed_index": seed_index,
                        "max_cross_track_mm": finite_or_none(case["max_cross_track_mm"]),
                        "final_tip_error_mm": finite_or_none(case["final_tip_error_mm"]),
                        "target_m": case["target_m"],
                    }
                    if not case["path_success"]:
                        reason = case["failure_reason"] or (
                            "incomplete_trajectory"
                            if not case["completed"]
                            else "acceptance_criteria"
                        )
                        candidates.append({**meta, "failure_reason": reason, "arrays": arrays})
            print(
                f"size={size or steps_per_path} seed_index={seed_index}: "
                f"train {int(train_result[0].sum())}/{len(train_result[0])}, "
                f"gen {int(gen_result[0].sum())}/{len(gen_result[0])}, "
                f"mse={final_train_mse[size_index, seed_index]:.5f} "
                f"full={full_train_mse[size_index, seed_index]:.5f}",
                flush=True,
            )

    chosen = pick_failures(candidates, len(sizes))
    best = pick_best(success_train[-1], cross_train[-1], largest_arrays)
    featured = []
    if best is not None:
        seed_index, episode, success, cross = best
        spec_id = train_specs[episode]["id"]
        featured.append(
            {
                "kind": "best",
                "set": "train",
                "id": spec_id,
                "group": train_specs[episode]["group"],
                "size_index": len(sizes) - 1,
                "sample_size": sizes[-1] or steps_per_path,
                "data_seed_index": seed_index,
                "bc_path_success": success,
                "bc_max_cross_track_mm": cross,
                "target_m": list(map(float, train_specs[episode]["target_m"])),
                "bc_arrays": largest_arrays[(seed_index, episode)],
                "expert_arrays": expert_train_arrays[spec_id],
            }
        )
    for failure in chosen:
        expert_arrays = (
            expert_train_arrays if failure["set"] == "train" else expert_gen_arrays
        ).get(failure["id"])
        if expert_arrays is None or (
            featured
            and failure["set"] == featured[0]["set"]
            and failure["id"] == featured[0]["id"]
            and failure["data_seed_index"] == featured[0]["data_seed_index"]
        ):
            continue
        featured.append(
            {
                **{k: v for k, v in failure.items() if k != "arrays"},
                "kind": f"failure_{failure['set']}",
                "bc_path_success": False,
                "bc_max_cross_track_mm": failure["max_cross_track_mm"],
                "bc_arrays": failure["arrays"],
                "expert_arrays": expert_arrays,
            }
        )
    failure_records = [
        {k: v for k, v in case.items() if k not in ("arrays", "bc_arrays", "expert_arrays")}
        for case in featured
        if case["kind"] != "best"
    ]
    scaling = {
        "sample_sizes_executed": [s or steps_per_path for s in sizes],
        "train_success_mean": [float(v) for v in success_train.mean(axis=(1, 2))],
        "gen_success_mean": [float(v) for v in success_gen.mean(axis=(1, 2))],
        "train_success_per_seed": [
            [float(v) for v in success_train[i].mean(axis=1)] for i in range(len(sizes))
        ],
        "gen_success_per_seed": [
            [float(v) for v in success_gen[i].mean(axis=1)] for i in range(len(sizes))
        ],
        "final_train_mse_mean": [float(v) for v in final_train_mse.mean(axis=1)],
        "full_train_mse_mean": [float(v) for v in full_train_mse.mean(axis=1)],
        "full_gen_mse_mean": [float(v) for v in full_gen_mse.mean(axis=1)],
        "monotone_train_mean": bool(np.all(np.diff(success_train.mean(axis=(1, 2))) >= -1e-12)),
    }
    scatter = [
        {
            "sample_size": sizes[i] or steps_per_path,
            "data_seed_index": k,
            "full_train_mse": float(full_train_mse[i, k]),
            "full_gen_mse": float(full_gen_mse[i, k]),
            "train_success_fraction": float(success_train[i, k].mean()),
            "gen_success_fraction": float(success_gen[i, k].mean()),
        }
        for i in range(len(sizes))
        for k in range(data_seeds)
    ]
    report = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        **protocol,
        "expert_verification": {"train": train_check, "generalization": gen_check},
        "train_records": train_records,
        "generalization_records": gen_records,
        "train_pairs_full": len(train_x),
        "generalization_pairs_full": len(gen_x),
        "scaling": scaling,
        "mse_vs_success": scatter,
        "featured_cases": [
            {k: v for k, v in case.items() if not k.endswith("_arrays")} for case in featured
        ],
        "failure_cases": failure_records,
        "limitations": [
            "The policy is the ignorant variant: it sees only [q1,q2,dq1,dq2] while the imitated expert reads the planned reference (both in its feedforward and its PD term). Success-rate gaps therefore mix 'learning from data' with this caliber difference.",
            "Pure BC torque drives the motors directly; the expert keeps a PD feedback loop. A failed BC episode has no feedback safety net; requested-vs-applied clipping is counted per episode.",
            "Closed-loop states leave the expert state distribution, so open-loop MSE (measured on expert states) is not a success predictor; the scatter table records their (non-)correlation.",
            "Small 2x64 network, one nominal model, no noise, payload, contact or delay; small path batches are small samples, not population success rates.",
            "At the largest sample size the three data seeds share the same recorded steps per path and differ only by network init and batch order (recorded, not hidden).",
        ],
    }
    archive = {
        "expert_train_x": train_x,
        "expert_train_y": train_y,
        "expert_gen_x": gen_x,
        "expert_gen_y": gen_y,
        "sample_sizes": np.asarray(sizes),
        "success_train": success_train,
        "success_gen": success_gen,
        "max_cross_train_mm": cross_train,
        "max_cross_gen_mm": cross_gen,
        "final_train_mse": final_train_mse,
        "full_train_mse": full_train_mse,
        "full_gen_mse": full_gen_mse,
        "loss_curves": loss_curves,
    }
    for index, case in enumerate(featured):
        for role in ("bc", "expert"):
            arrays = case[f"{role}_arrays"]
            archive.update(
                {
                    f"case{index}_{role}_states": arrays["states"],
                    f"case{index}_{role}_points": arrays["points"],
                    f"case{index}_{role}_desired_points": arrays["desired_points"],
                    f"case{index}_{role}_torques_nm": arrays["torques_nm"],
                    f"case{index}_{role}_requested_torques_nm": arrays["requested_torques_nm"],
                }
            )
    np.savez_compressed(output / "trajectories.npz", **archive)
    report["trajectories_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_overview(output / "overview.png", report)
    save_featured_cases(output / "featured_cases.png", featured, dt)
    return report


def save_overview(path, report):
    configure_plot_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    rates, scatter = axes
    sizes = report["scaling"]["sample_sizes_executed"]
    positions = np.arange(len(sizes))
    with np.load(Path(path).parent / "trajectories.npz", allow_pickle=False) as data:
        success_train, success_gen = data["success_train"], data["success_gen"]
        mse = data["full_train_mse"]
    seeds = success_train.shape[1]
    for success, label, color in (
        (success_train, "训练内（同批路径）", "#0f766e"),
        (success_gen, "泛化（新种子路径）", "#b91c1c"),
    ):
        for seed_index in range(seeds):
            rates.scatter(
                positions,
                success[:, seed_index].mean(axis=1) * 100,
                s=18,
                color=color,
                alpha=0.45,
            )
        rates.plot(positions, success.mean(axis=(1, 2)) * 100, "o-", color=color, label=label)
    rates.set(
        xticks=positions,
        xticklabels=[str(s) for s in sizes],
        xlabel="每条路径采样的步数（全量 = 最后一点）",
        ylabel="路径验收通过率（%）",
        ylim=(-3, 105),
        title="数据量–成功率：小点是单个数据种子",
    )
    rates.legend(fontsize=8)
    train_fraction = success_train.mean(axis=(1, 2)) * 100
    gen_fraction = success_gen.mean(axis=(1, 2)) * 100
    scatter.scatter(mse.mean(axis=1), train_fraction, color="#0f766e", label="训练内")
    scatter.scatter(mse.mean(axis=1), gen_fraction, color="#b91c1c", label="泛化")
    for x, y, size in zip(mse.mean(axis=1), train_fraction, sizes, strict=True):
        scatter.annotate(str(size), (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
    scatter.set(
        xlabel="专家状态分布上的全量 MSE（按数据量取均值）",
        ylabel="闭环成功率（%）",
        title="开环 MSE 与闭环成功率的关系",
        xscale="log",
        ylim=(-3, 105),  # honest flat-zero rates must not be autoscaled into noise
    )
    scatter.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_featured_cases(path, featured, dt):
    """Best BC episode and failure episodes, each against its expert rollout."""
    configure_plot_font()
    colors = {"best": "#0f766e", "failure_train": "#ea580c", "failure_gen": "#b91c1c"}
    kinds = {"best": "最佳 BC 回合", "failure_train": "训练内失败", "failure_gen": "泛化失败"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    xy, timed, cross, torque = axes.ravel()
    shown = [case for case in featured if case["kind"] != "best"] or featured[:1]
    for case in featured:  # each featured episode: BC solid, its expert dashed
        color = colors[case["kind"]]
        label = f"{kinds[case['kind']]} / {case['id']}"
        bc, expert = case["bc_arrays"], case["expert_arrays"]
        bc_tip, expert_tip = bc["points"][:, -1], expert["points"][:, -1]
        xy.plot(
            expert_tip[:, 0] * 100,
            expert_tip[:, 1] * 100,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="专家（前馈+PD）" if case is featured[0] else None,
        )
        xy.plot(bc_tip[:, 0] * 100, bc_tip[:, 1] * 100, color=color, linewidth=1.4, label=label)
        xy.plot(
            bc["desired_points"][[0, -1], 0] * 100,
            bc["desired_points"][[0, -1], 1] * 100,
            "k:",
            linewidth=1,
            label="规定直线" if case is featured[0] else None,
        )
        target = np.asarray(case["target_m"])
        bc_times = np.arange(len(bc_tip)) * dt
        expert_times = np.arange(len(expert_tip)) * dt
        timed.plot(
            bc_times,
            np.linalg.norm(bc_tip - bc["desired_points"], axis=1) * 1000,
            color=color,
            label=label,
        )
        timed.plot(
            expert_times,
            np.linalg.norm(expert_tip - expert["desired_points"], axis=1) * 1000,
            color="gray",
            linestyle="--",
            linewidth=1.1,
        )
        cross.plot(
            bc_times,
            segment_distance(bc_tip, bc["desired_points"][0], target) * 1000,
            color=color,
            label=label,
        )
        if any(case is item for item in shown):
            # stairs needs one more edge than values: torque[k] acts on [k*dt,(k+1)*dt)
            steps = (np.arange(len(bc["torques_nm"]) + 1)) * dt
            for joint, joint_color in enumerate(("#2563eb", "#f97316")):
                torque.stairs(
                    bc["torques_nm"][:, joint],
                    steps,
                    color=joint_color,
                    alpha=0.4 if len(shown) > 1 else 0.75,
                    label=f"关节 {joint + 1}（{label}）" if joint == 0 else None,
                )
    xy.set(
        xlabel="X（cm）",
        ylabel="Y（cm）",
        title="末端路径：BC 力矩直接驱动 vs 专家",
        aspect="equal",
    )
    xy.legend(fontsize=7, loc="upper left")
    timed.set(
        xlabel="时间（s）",
        ylabel="距同时刻规定点（mm）",
        title="时间跟踪：实线 BC，虚线专家",
    )
    timed.axhline(PATH_GATE_MM, color="gray", linestyle=":")
    timed.legend(fontsize=7)
    cross.set(
        xlabel="时间（s）",
        ylabel="到规定线段距离（mm）",
        title="路径门限：移动期最大值 <=2 mm 才通过",
    )
    cross.axhline(PATH_GATE_MM, color="gray", linestyle=":")
    cross.legend(fontsize=7)
    torque.set(
        xlabel="时间（s）",
        ylabel="力矩（N·m）",
        title="失败回合实际施加力矩（BC 输出经 ±0.25 N·m 限幅）",
    )
    for bound in (-TORQUE_LIMIT, TORQUE_LIMIT):
        torque.axhline(bound, color="gray", linestyle=":")
    torque.legend(fontsize=7)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-group", type=int, default=12)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=list(SAMPLE_SIZES))
    parser.add_argument("--data-seeds", type=int, default=DATA_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--source-results", type=Path, default=None)
    args = parser.parse_args()
    try:
        report = run_experiment(
            args.output,
            seed=args.seed,
            per_group=args.per_group,
            sample_sizes=args.sample_sizes,
            data_seeds=args.data_seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            source_results=args.source_results,
        )
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["scaling"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
