"""Lesson 28: behavioural cloning of the lesson-13 expert (pure numpy, no torch).

Unit checks run on tiny synthetic datasets; the end-to-end checks shrink the
real experiment (1 path per group, 10/all steps, few epochs) into tmp dirs so
the module stays fast while still exercising expert collection, gate, training
rollout, acceptance, archive and the Tk demo loader.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# The isolated_tk child re-imports this module before any pyplot use: forcing
# the Agg backend keeps run_experiment's figure saving from creating (and
# destroying) a first Tcl interpreter, whose leftovers make the demo test's
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.bc_demo import expected_npz_keys, featured_max_cross_mm, load_replays
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_path_batch import generate_manifest
from embodied_learning.experiments.bc_imitation import (
    EXPERIMENT,
    MLP,
    STATE_INPUTS,
    AdamOptimizer,
    bc_rollout,
    evaluate_trajectory,
    plan_path,
    run_experiment,
    sample_step_indices,
    train_bc,
)
from embodied_learning.planar_arm import ArmSimulation

SMALL_CONFIG = {
    "per_group": 1,
    "sample_sizes": (10, 0),
    "data_seeds": 2,
    "epochs": 300,
    "batch_size": 256,
    "lr": 1e-3,
}
FROZEN_SOURCE = Path("results/arm_timing_2026-09-02")


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("bc_small") / "run"
    report = run_experiment(output, seed=0, **SMALL_CONFIG)
    return output, report


# ------------------------------------------------------------- network and data
def test_mlp_backprop_matches_finite_differences():
    """Central-difference check at the 1e-6 level on every weight and bias.

    Biases are jittered off their zero init first: a dead first hidden layer
    can put a second-layer preact at exactly 0, a genuine ReLU kink where the
    analytic subgradient (relu'(0)=0) and any finite difference must disagree.
    """
    net = MLP(4, (6, 5), 2, seed=0)
    rng = np.random.default_rng(1)
    for bias in net.biases:
        bias += rng.normal(0, 0.05, bias.shape)
    x = rng.normal(0, 1, (12, 4))
    y = 0.25 * np.tanh(x @ rng.normal(0, 0.5, (4, 2)))
    _, grad_weights, grad_biases = net.loss_and_gradients(x, y)
    parameters = [*net.weights, *net.biases]
    gradients = [*grad_weights, *grad_biases]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        assert parameter.shape == gradient.shape
    eps = 1e-6
    worst = 0.0
    for parameter, gradient in zip(parameters, gradients, strict=True):
        flat, grad = parameter.ravel(), gradient.ravel()
        for index in range(flat.size):
            original = flat[index]
            flat[index] = original + eps
            loss_plus = net.loss(x, y)
            flat[index] = original - eps
            loss_minus = net.loss(x, y)
            flat[index] = original
            numeric = (loss_plus - loss_minus) / (2 * eps)
            worst = max(
                worst, abs(numeric - grad[index]) / max(abs(numeric), abs(grad[index]), 1e-10)
            )
    assert worst < 1e-6


def test_tiny_dataset_overfits_to_near_zero_loss():
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, (32, 4))
    y = 0.25 * np.tanh(x @ rng.normal(0, 0.5, (4, 2)))
    _, history = train_bc(x, y, epochs=800, batch_size=32, lr=2e-3, init_seed=0, shuffle_seed=1)
    assert history[-1] < 1e-10  # 2x64 ReLU has the capacity to memorize 32 pairs
    assert np.isfinite(history).all() and (np.diff(history) <= 1e-12).mean() > 0.99


def test_same_seed_training_reproduces_bitwise():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (64, 4))
    y = 0.1 * np.sin(x @ rng.normal(0, 0.5, (4, 2)))
    net_a, history_a = train_bc(
        x, y, epochs=40, batch_size=16, lr=1e-3, init_seed=7, shuffle_seed=8
    )
    net_b, history_b = train_bc(
        x, y, epochs=40, batch_size=16, lr=1e-3, init_seed=7, shuffle_seed=8
    )
    for weight_a, weight_b in zip(net_a.weights, net_b.weights, strict=True):
        assert np.array_equal(weight_a, weight_b)
    for bias_a, bias_b in zip(net_a.biases, net_b.biases, strict=True):
        assert np.array_equal(bias_a, bias_b)
    assert np.array_equal(history_a, history_b)
    net_c, _ = train_bc(x, y, epochs=40, batch_size=16, lr=1e-3, init_seed=9, shuffle_seed=8)
    assert not np.array_equal(net_c.weights[0], net_a.weights[0])


def test_adam_updates_parameters_in_place():
    net = MLP(4, (5, 4), 2, seed=4)
    parameters = [*net.weights, *net.biases]
    before = [p.copy() for p in parameters]
    optimizer = AdamOptimizer(parameters, lr=0.1)
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, (3, 4))
    y = rng.normal(0, 0.1, (3, 2))
    _, grad_weights, grad_biases = net.loss_and_gradients(x, y)
    gradients = [*grad_weights, *grad_biases]
    assert all(np.abs(grad).max() > 0.0 for grad in gradients)  # every array gets a push
    optimizer.step(parameters, gradients)
    assert optimizer.t == 1
    live = [*net.weights, *net.biases]
    assert all(updated is array for updated, array in zip(parameters, live, strict=True))
    for reference, updated in zip(before, parameters, strict=True):
        assert not np.array_equal(reference, updated)  # the caller's arrays moved in place


def test_sample_step_indices_contract():
    rng = np.random.default_rng(5)
    drawn = sample_step_indices(rng, 10, 350)
    assert len(drawn) == 10 and np.all(np.diff(drawn) > 0) and drawn[-1] < 350
    assert np.array_equal(sample_step_indices(rng, 0, 350), np.arange(350))
    assert np.array_equal(sample_step_indices(rng, 350, 350), np.arange(350))
    with pytest.raises(ValueError):
        sample_step_indices(rng, 351, 350)


# ------------------------------------------------------------------- acceptance
def test_expert_gate_and_acceptance_match_lesson13_run_path():
    """The recomputed acceptance must agree with lesson-13's own run_path."""
    manifest = generate_manifest(400, 12)
    dt = ArmSimulation().dt
    for spec_id in ("interior_00", "singular_inward"):
        spec = next(t for t in manifest["trials"] if t["id"] == spec_id)
        arrays, case = run_path(
            "waypoint_ik",
            initial_q=spec["initial_q_rad"],
            target=spec["target_m"],
            move_seconds=4.0,
            hold_seconds=3.0,
            controller="feedforward_pd",
        )
        assert case["path_success"] and case["endpoint_success"]  # lesson-13 all-pass record
        subset = {
            key: arrays[key]
            for key in ("states", "points", "desired_points", "requested_torques_nm")
        }
        recomputed = evaluate_trajectory(subset, spec["target_m"], 4.0, dt, completed=True)
        assert recomputed["path_success"]
        assert recomputed["max_cross_track_mm"] == pytest.approx(
            case["max_cross_track_mm"], abs=1e-9
        )
        assert recomputed["settled_after_movement_at_s"] == pytest.approx(
            case["settled_after_movement_at_s"], abs=1e-9
        )
        assert recomputed["endpoint_success"] == case["endpoint_success"]


def test_bc_rollout_applies_clipped_policy_torque():
    manifest = generate_manifest(400, 1)
    spec = manifest["trials"][0]
    reference = plan_path(spec, ArmSimulation().dt)
    policy = MLP(STATE_INPUTS, (8, 8), 2, seed=6)
    arrays, failure = bc_rollout(policy, spec, reference, ArmSimulation().dt)
    assert failure == ""  # an untrained small net does not trip the physics guards
    assert len(arrays["states"]) == len(arrays["torques_nm"]) + 1 == 351
    assert len(arrays["desired_points"]) == len(arrays["states"])
    np.testing.assert_allclose(arrays["states"][0, :2], spec["initial_q_rad"], atol=1e-12)
    limit = 0.25
    np.testing.assert_allclose(
        arrays["torques_nm"], np.clip(arrays["requested_torques_nm"], -limit, limit)
    )


def test_evaluate_trajectory_handles_truncated_horizon():
    """A physically failed episode ends before the plan; arrays stay aligned.

    Regression guard: desired_points must be truncated with the executed
    horizon, otherwise a failed episode crashes the acceptance recomputation
    on a shape mismatch. (In this plant the 20 rad/s guard is unreachable at
    the 0.25 N*m limit because joint damping caps speed near 12.5 rad/s, so
    the truncation is exercised synthetically.)
    """
    manifest = generate_manifest(400, 1)
    spec = manifest["trials"][0]
    dt = ArmSimulation().dt
    arrays, case = run_path(
        "waypoint_ik",
        initial_q=spec["initial_q_rad"],
        target=spec["target_m"],
        move_seconds=4.0,
        hold_seconds=3.0,
        controller="feedforward_pd",
    )
    assert case["path_success"]
    cut = 60
    truncated = {
        "states": arrays["states"][: cut + 1],
        "points": arrays["points"][: cut + 1],
        "desired_points": arrays["desired_points"][: cut + 1],
        "requested_torques_nm": arrays["requested_torques_nm"][:cut],
    }
    report = evaluate_trajectory(truncated, spec["target_m"], 4.0, dt, completed=False)
    assert not report["completed"] and not report["path_success"]
    assert report["endpoint_success"] is False
    assert report["max_cross_track_mm"] >= 0.0
    assert report["terminal_window"]["violations"][0] == "insufficient_post_movement_window"


# ------------------------------------------------------- shrunk end-to-end record
def test_small_run_expert_gate_and_archive_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    for side in ("train", "generalization"):
        check = report["expert_verification"][side]
        assert check["planned"] == check["executed"] == check["path_successes"] == 3
        assert check["planning_rejected"] == 0
    assert report["scaling"]["sample_sizes_executed"] == [10, 350]
    assert report["train_pairs_full"] == 3 * 350
    assert report["trajectories_sha256"] == hashlib_sha256(output / "trajectories.npz")
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert archive["success_train"].shape == (2, 2, 3)
        assert archive["success_gen"].shape == (2, 2, 3)
        assert archive["loss_curves"].shape == (2, 2, 300)
        assert archive["sample_sizes"].tolist() == [10, 0]
        files = set(archive.files)
    assert files == expected_npz_keys(len(report["featured_cases"]))
    assert (output / "overview.png").is_file() and (output / "featured_cases.png").is_file()
    assert (output / "expert_manifest.json").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, **SMALL_CONFIG)
    # the frozen-manifest gate fires when the regenerated manifest cannot match
    with pytest.raises(ValueError, match="frozen source record"):
        run_experiment(
            output.parent / "gated",
            per_group=1,
            source_results=FROZEN_SOURCE,
            **{key: value for key, value in SMALL_CONFIG.items() if key != "per_group"},
        )


def test_data_scaling_improves_fit_in_the_mean(small_run):
    """Data-volume monotonicity in the mean sense: in-tube fit must improve.

    The comparable quantity is the MSE on the FULL expert dataset, not on the
    trained subset (a smaller subset can overfit to a lower final batch loss).
    Out-of-tube (generalization) MSE is NOT asserted to drop: three training
    paths cannot cover the generalization paths, and the real record shows
    that MSE saturates there — the distribution-shift finding itself.
    """
    _, report = small_run
    scaling = report["scaling"]
    full_train = scaling["full_train_mse_mean"]
    assert full_train[-1] < full_train[0]
    assert isinstance(scaling["monotone_train_mean"], bool)
    assert scaling["monotone_train_mean"] == bool(
        np.all(np.diff(scaling["train_success_mean"]) >= -1e-12)
    )


def test_featured_cases_agree_with_arrays_and_records(small_run):
    output, report = small_run
    dt = report["dt_s"]
    kinds = [case["kind"] for case in report["featured_cases"]]
    assert kinds[0] == "best"
    records = {
        "train": [r for r in report["train_records"] if r["status"] == "executed"],
        "gen": [r for r in report["generalization_records"] if r["status"] == "executed"],
    }
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        for index, case in enumerate(report["featured_cases"]):
            bc_points = archive[f"case{index}_bc_points"]
            bc_desired = archive[f"case{index}_bc_desired_points"]
            expert_points = archive[f"case{index}_expert_points"]
            expert_desired = archive[f"case{index}_expert_desired_points"]
            target = np.asarray(case["target_m"])
            assert featured_max_cross_mm(bc_points, bc_desired, target, dt) == pytest.approx(
                case["bc_max_cross_track_mm"], abs=1e-6
            )
            record = next(r for r in records[case["set"]] if r["id"] == case["id"])
            assert featured_max_cross_mm(
                expert_points, expert_desired, target, dt
            ) == pytest.approx(record["max_cross_track_mm"], abs=1e-6)
    failures = [case for case in report["featured_cases"] if case["kind"] != "best"]
    assert len(failures) >= 1  # failed episodes are retained, never filtered out


def test_demo_loader_validates_and_rejects_tampering(small_run, tmp_path):
    output, _ = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")

    report = json.loads(summary_text)
    report["scaling"]["train_success_mean"][0] += 0.01
    (work / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Train success mean"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")

    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["full_train_mse"][0, 0] *= 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    del payload["case0_bc_states"]
    np.savez_compressed(path, **payload)
    report = json.loads(summary_text)
    report["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)
    # the shared small_run directory stays pristine; only the copy was damaged


# ------------------------------------------------------------------- CLI and UI
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.bc_imitation",
            "--output",
            str(out),
            "--seed",
            "0",
            "--per-group",
            "1",
            "--sample-sizes",
            "10",
            "0",
            "--data-seeds",
            "1",
            "--epochs",
            "30",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "overview.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.bc_imitation",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.bc_demo import BcDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = BcDemo(root, data)
    root.update()
    assert demo.mode.get() == "paths"
    assert len(demo.fig.axes) == 2
    assert "专家" in demo.stats.cget("text")
    assert "BC" in demo.stats.cget("text")
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 2  # mode switches must not accumulate axes
    demo.mode.set("scaling")
    demo.redraw()
    assert len(demo.fig.axes) == 1
    panel = demo.stats.cget("text")
    assert "训练内" in panel and "泛化" in panel
    first_mean = report["scaling"]["train_success_mean"][0] * 100
    assert f"{first_mean:.1f}%" in panel  # panel numbers come from the summary
    demo.mode.set("mechanism")
    demo.redraw()
    assert len(demo.fig.axes) == 2
    assert "复合" in demo.stats.cget("text")
    demo.close()


def hashlib_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
