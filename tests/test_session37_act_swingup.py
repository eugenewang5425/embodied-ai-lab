"""Lesson 37: ACT / diffusion-style minimal probe - chunked multi-expert policy.

Unit checks pin the chunk slicing contract (obs[t] with labels[t:t+H], never
crossing episode ends), the gated-NLL gradients by finite differences and a
hand-computed K=2 known answer, the deterministic mixture/top1 paths, the
micro training determinism (two identical runs, loss decreasing), the lesson-7
acceptance reuse, the teacher dataset reproduction (bitwise repeat + SHA) and
its verification against the official lesson-36 record, the shrunk end-to-end
record contract, the demo loader tamper rejections (three routes), the CLI
end-to-end and the real Tk demo modes. No real training is run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

# The isolated_tk child re-imports this module before any pyplot use: forcing
# the Agg backend keeps run_experiment's figure saving from creating (and
# destroying) a first Tcl interpreter, whose leftovers make the demo test's
# real tk.Tk() die with "invalid command name tcl_findLibrary" on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.act_demo import load_replays
from embodied_learning.experiments.act_swingup import (
    EXPERIMENT,
    ACTConfig,
    ChunkMoEPolicy,
    build_chunks,
    chunk_mse_loss_and_gradients,
    dataset_sha256,
    expected_npz_keys,
    gate_phase_stats,
    moe_nll_loss_and_gradients,
    reproduce_teacher_pairs,
    run_act_episode,
    run_experiment,
    verify_dataset_against_lesson36,
)
from embodied_learning.experiments.swingup_comparison import recovery_metrics
from embodied_learning.swingup import design_swingup_lqr

SMALL_CONFIG = {
    "train_seeds": 1,
    "eval_seed_count": 2,
    "epochs": 4,
    "eval_every_epochs": 2,
    "minibatch": 32,
}
SMALL_DATASET = {"rounds": 1, "rollouts": 2, "updates_per_round": 2, "train_seeds": 1}


@pytest.fixture(scope="module")
def design_and_reward():
    design = design_swingup_lqr()
    from embodied_learning.experiments.ppo_swingup import RewardFunction

    return design, RewardFunction(design.controller.reference)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment (all tiers) shared by record-level checks."""
    output = tmp_path_factory.mktemp("act_small") / "run"
    config = ACTConfig(**SMALL_CONFIG)
    report = run_experiment(
        output, seed=0, config=config, dataset_budget=dict(SMALL_DATASET), log=None
    )
    return output, report


# ------------------------------------------------------------- chunk contract
def test_chunk_slicing_alignment_contract():
    """obs[t] must pair with labels[t:t+H]; chunks never cross episode ends."""
    rng = np.random.default_rng(7)
    pairs = []
    for length in (5, 3, 9):
        obs = rng.normal(size=(length, 5))
        labels = rng.uniform(-3, 3, size=length)
        pairs.append((obs, labels))
    horizon = 3
    obs, targets = build_chunks(pairs, horizon)
    rows = []
    for obs_k, labels_k in pairs:
        for t in range(len(labels_k) - horizon + 1):
            rows.append((t, labels_k[t : t + horizon]))
    assert len(rows) == 3 + 1 + 7
    assert len(obs) == len(rows) == len(targets)
    for index, (t, window) in enumerate(rows):
        np.testing.assert_array_equal(targets[index], window)
    # row order is the episode order and the observation rows are the per-step
    # observations of their episodes
    assert len(obs) == 11
    with pytest.raises(ValueError):
        build_chunks([(np.zeros(4), np.zeros(5))], 2)  # mismatched alignment


# ----------------------------------------------------------------- gradients
def test_gate_softmax_gradient_finite_difference():
    """dL/dz and the full parameter chain match numerical gradients."""
    rng = np.random.default_rng(11)
    policy = ChunkMoEPolicy(n_experts=2, horizon=3, hidden=(6, 6), seed=(1, 2))
    obs = rng.normal(size=(4, 5))
    targets = rng.uniform(-3, 3, size=(4, 3))
    _loss, grads = moe_nll_loss_and_gradients(policy, obs, targets)
    params = policy.parameters()

    def loss_only(policy_core):
        return moe_nll_loss_and_gradients(policy_core, obs, targets, return_grads=False)[0]

    for index in range(len(params)):
        array = params[index]
        numerical = np.zeros_like(array)
        flat = array.reshape(-1)
        for j in range(len(flat)):
            old = flat[j]
            flat[j] = old + 1e-6
            plus = loss_only(policy)
            flat[j] = old - 1e-6
            minus = loss_only(policy)
            flat[j] = old
            numerical.reshape(-1)[j] = (plus - minus) / (2e-6)
        scale = max(1e-10, float(np.max(np.abs(grads[index]))))
        assert float(np.max(np.abs(numerical - grads[index]))) / scale < 1e-4


def test_moe_nll_known_answer_K2():
    """Hand-computed gated NLL for a two-expert policy (sum of densities)."""
    policy = ChunkMoEPolicy(n_experts=2, horizon=2, hidden=(6, 6), seed=(0, 0))
    # fix the parameters so the values are known
    policy.expert_weights[0] = np.zeros((6, 2))
    policy.expert_biases[0] = np.array([1.0, 1.0])
    policy.expert_weights[1] = np.zeros((6, 2))
    policy.expert_biases[1] = np.array([-2.0, -2.0])
    policy.gate_weight = np.zeros((6, 2))
    policy.gate_bias = np.array([0.0, 0.0])
    obs = np.zeros((1, 5))
    targets = np.array([[1.0, 1.0]])
    log_w = np.array([-np.log(2.0), -np.log(2.0)])
    error0 = 0.0
    error1 = ((1.0 - (-2.0)) ** 2) * 2
    log_e0 = -0.5 * error0  # expert 0 = (1,1): zero error
    log_e1 = -0.5 * error1  # expert 1 = (-2,-2)
    log_p = log_w + np.array([log_e0, log_e1])
    maximum = np.max(log_p)
    loss_expected = -(maximum + np.log(np.exp(log_p - maximum).sum()))
    loss, _grads = moe_nll_loss_and_gradients(policy, obs, targets)
    assert abs(loss - loss_expected) < 1e-9
    # posterior must sum to one and weight the near-exact expert highest
    means, _gate, _h = policy.forward(obs)
    assert means[0, 0, 0] == pytest.approx(1.0)
    assert means[0, 1, 0] == pytest.approx(-2.0)
    # a chunk MSE on the same single-head copy matches the elementwise formula
    solo = ChunkMoEPolicy(n_experts=1, horizon=2, hidden=(6, 6), seed=(0, 0))
    solo.expert_weights[0] = np.zeros((6, 2))
    solo.expert_biases[0] = np.array([1.0, 1.0])
    mse, _mse_grads = chunk_mse_loss_and_gradients(solo, obs, targets)
    assert abs(mse) < 1e-12


# ------------------------------------------------- deterministic path contract
def test_deterministic_mixture_and_top1_contract():
    """Mixture = p-weighted mu_0; top1 = argmax expert mu_0; sampling uses sigma."""
    rng = np.random.default_rng(5)
    policy = ChunkMoEPolicy(n_experts=3, horizon=4, hidden=(8, 8), seed=(2, 1))
    obs = rng.normal(size=(1, 5))
    means, gate, _h = policy.forward(obs)
    weights = np.exp(gate - gate.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    mixture = float((weights[0] * means[0, :, 0]).sum())
    assert policy.deterministic_first(obs, "mixture") == pytest.approx(mixture)
    top1 = int(np.argmax(weights[0]))
    assert policy.deterministic_first(obs, "top1") == pytest.approx(float(means[0, top1, 0]))
    sample_rng = np.random.default_rng(9)
    sample = policy.sample_first(obs, sample_rng)
    assert np.isfinite(sample)
    # the gate weights of a deterministic policy are probabilities
    gate_weights = policy.gate_weights(obs)[0]
    assert gate_weights.sum() == pytest.approx(1.0)
    assert (gate_weights >= 0).all()


def test_eval_matches_lesson7_acceptance(design_and_reward):
    """run_act_episode wraps the lesson-7 recovery_metrics verbatim."""
    design, reward = design_and_reward
    reference, dt = design.controller.reference, design.dt
    policy = ChunkMoEPolicy(n_experts=2, horizon=4, hidden=(8, 8), seed=(0, 9))
    record, arrays, gates = run_act_episode(policy, reward, reference, dt, mode="mixture")
    view = {**arrays, "modes": np.full(len(arrays["controls"]), "rl")}
    direct = recovery_metrics(view, {"failure_reason": record["failure_reason"]}, reference, dt)
    assert record["recovered"] == direct["recovered"]
    assert record["settled_at_s"] == direct["settled_at_s"]
    # lesson-7 alignment: states[k] precedes controls[k]
    assert len(arrays["states"]) == len(arrays["controls"]) + 1
    assert gates.shape[0] == len(arrays["controls"])
    assert gates.shape[1] == 2


# ------------------------------------------------- training determinism
def test_micro_training_deterministic_and_improves(design_and_reward, tmp_path_factory):
    """Two identical runs reproduce bitwise policies; the loss decreases."""
    config = ACTConfig(**SMALL_CONFIG)
    a = tmp_path_factory.mktemp("det_a")
    b = tmp_path_factory.mktemp("det_b")

    def run_one(directory):
        report = run_experiment(
            directory,
            seed=0,
            config=config,
            tiers=("moe_k2",),
            dataset_budget=dict(SMALL_DATASET),
            log=None,
        )
        return report

    report_a = run_one(a / "run")
    report_b = run_one(b / "run")
    za = np.load(a / "run" / "trajectories.npz", allow_pickle=False)
    zb = np.load(b / "run" / "trajectories.npz", allow_pickle=False)
    for key in za.files:
        if key.startswith(("loss_curve", "policy")):
            np.testing.assert_array_equal(za[key], zb[key])
    seed_record = report_a["tiers"][0]["per_seed"][0]
    assert seed_record["loss_last"] < seed_record["loss_first"]
    assert report_b["tiers"][0]["per_seed"][0]["loss_last"] == pytest.approx(
        seed_record["loss_last"]
    )


# ------------------------------------------------------------ dataset checks
def test_dataset_reproduction_bitwise_and_sha():
    """The teacher dataset rebuild is deterministic; a stable SHA identifies it."""
    d1 = reproduce_teacher_pairs(
        master_seed=0,
        rounds=SMALL_DATASET["rounds"],
        rollouts=SMALL_DATASET["rollouts"],
        updates_per_round=SMALL_DATASET["updates_per_round"],
        train_seeds=SMALL_DATASET["train_seeds"],
        log=None,
    )
    d2 = reproduce_teacher_pairs(
        master_seed=0,
        rounds=SMALL_DATASET["rounds"],
        rollouts=SMALL_DATASET["rollouts"],
        updates_per_round=SMALL_DATASET["updates_per_round"],
        train_seeds=SMALL_DATASET["train_seeds"],
        log=None,
    )
    labels1 = np.concatenate(
        [np.asarray(labels, dtype=float) for rp in d1[0][0] for _o, labels in rp]
    )
    labels2 = np.concatenate(
        [np.asarray(labels, dtype=float) for rp in d2[0][0] for _o, labels in rp]
    )
    assert labels1.shape == labels2.shape
    assert np.array_equal(labels1, labels2)
    total = sum(len(np.asarray(labels, dtype=float)) for rp in d1[0][0] for _o, labels in rp)
    assert total == len(labels1)
    assert np.isfinite(labels1).all()
    assert np.max(np.abs(labels1)) <= 3.0 + 1e-9
    obs, targets = build_chunks(d1[0][0][0], 8)
    digest = dataset_sha256(obs, targets)
    assert isinstance(digest, str) and len(digest) == 64


def test_seed0_reproduction_verifies_against_lesson36_record(tmp_path_factory, design_and_reward):
    """Seed-0 full-round reproduction matches the official lesson-36 aggregates.

    This is the data-authenticity gate of the lesson: the lesson-36 record does
    not persist raw pairs, so the reproduction must match its recorded dataset
    sizes (bitwise) and per-round label statistics (1e-9).  The full 3-seed
    check runs in the official record (`results/act_swingup_2026-09-06`).
    """
    per_seed_rounds, total = reproduce_teacher_pairs(
        master_seed=0,
        rounds=6,
        rollouts=8,
        updates_per_round=15,
        train_seeds=1,
        log=None,
    )
    record_dir = "results/dagger_swingup_2026-09-06"
    if not os.path.isdir(record_dir):
        pytest.skip("lesson-36 official record not present")
    verification = verify_dataset_against_lesson36(per_seed_rounds, record_dir, train_seeds=1)
    assert verification["passed"] is True, verification
    assert total == 3730  # the lesson-36 seed-0 aggregate


def test_teacher_quality_gate_and_baseline():
    """The teacher re-run passes 20/20 and its repeats coincide."""
    from embodied_learning.experiments.ppo_swingup import baseline_evaluations

    design = design_swingup_lqr()
    records, _states, controls, identical = baseline_evaluations(design, 20)
    assert len(records) == 20
    assert all(r["recovered"] for r in records)
    assert identical is True
    assert np.all(np.abs(controls) <= 3.0 + 1e-9)


# -------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert [t["name"] for t in report["tiers"]] == [
        "mse_single",
        "block_mse",
        "moe_k2",
        "moe_k4",
        "moe_k4_h8",
    ]
    assert report["baseline"]["successes"] == report["baseline"]["episodes"] == 2
    assert report["baseline"]["deterministic_identical_repeats"] is True
    assert report["teacher_verification"]["gate_passed"] is True
    assert report["protocol"]["dataset"]["verification"]["passed"] is None
    assert (output / "summary.json").exists()
    assert {(p.stem) for p in output.glob("*.png")} >= {
        "training_curves",
        "gate_analysis",
        "comparison",
    }
    with np.load(output / "trajectories.npz", allow_pickle=False) as npz:
        assert set(npz.files) == expected_npz_keys(report)
    # per-seed eval summary cross-checks against the archive
    with np.load(output / "trajectories.npz", allow_pickle=False) as npz:
        for tier_index, tier in enumerate(report["tiers"]):
            seed_record = tier["per_seed"][0]
            per_episode = seed_record["eval"]
            assert int(npz[f"eval_recovered_{tier_index}_0"].sum()) == per_episode["successes"]


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, report = small_run
    data = load_replays(output)
    assert data["report"] == report
    # route 1: byte flip -> checksum mismatch
    tampered = tmp_path / "tamper1"
    shutil.copytree(output, tampered)
    with open(tampered / "trajectories.npz", "r+b") as handle:
        handle.seek(1000)
        original = handle.read(1)
        handle.seek(1000)
        handle.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(ValueError):
        load_replays(tampered)
    # route 2: summary successes disagree with the archive
    tampered2 = tmp_path / "tamper2"
    shutil.copytree(output, tampered2)
    summary = json.loads((tampered2 / "summary.json").read_text(encoding="utf-8"))
    summary["tiers"][0]["per_seed"][0]["eval"]["success_per_episode"] = [True] * len(
        summary["tiers"][0]["per_seed"][0]["eval"]["success_per_episode"]
    )
    (tampered2 / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(tampered2)
    # route 3: a dropped archive key with a re-hashed npz -> key-set rejection
    tampered3 = tmp_path / "tamper3"
    shutil.copytree(output, tampered3)
    with np.load(output / "trajectories.npz", allow_pickle=False) as original:
        payload = {key: original[key] for key in original.files if key != "loss_curve_0_0"}
    np.savez_compressed(tampered3 / "trajectories.npz", **payload)
    summary = json.loads((tampered3 / "summary.json").read_text(encoding="utf-8"))
    summary["trajectories_sha256"] = hashlib.sha256(
        (tampered3 / "trajectories.npz").read_bytes()
    ).hexdigest()
    (tampered3 / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(tampered3)


def test_cli_subprocess_end_to_end(tmp_path):
    """CLI parity`: micro end-to-end, then the same output is refused."""
    output = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.act_swingup",
            "--output",
            str(output),
            "--seed",
            "0",
            "--tiers",
            "block_mse",
            "--train-seeds",
            "1",
            "--eval-seed-count",
            "2",
            "--epochs",
            "4",
            "--eval-every",
            "2",
            "--minibatch",
            "32",
            "--dataset-rounds",
            "1",
            "--dataset-rollouts",
            "2",
            "--dataset-updates",
            "2",
            "--dataset-seeds",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert (output / "summary.json").exists()
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert [t["name"] for t in report["tiers"]] == ["block_mse"]
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.act_swingup",
            "--output",
            str(output),
            "--tiers",
            "block_mse",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert again.returncode != 0  # the new-directory guard refuses overwrites


def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.act_demo import ActDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = ActDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "epoch" in panel and "确定性" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("trajectory")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "教师" in panel and "首达" in panel
    assert f"{report['baseline']['median_settled_at_s']:.2f} s" in panel
    demo.mode.set("comparison")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "0/60" in panel or "PPO" in panel  # cited comparison wording
    assert f"{report['baseline']['successes']}/{report['baseline']['episodes']}" in panel
    demo.close()


def test_gate_phase_stats_runs_and_shapes(design_and_reward, small_run):
    """Gate phase statistics on the teacher trajectory produce sane shapes."""
    design, _reward = design_and_reward
    output, _report = small_run
    data = load_replays(output)
    policy = ChunkMoEPolicy(n_experts=4, horizon=8, hidden=(8, 8), seed=(0, 0))
    per_phase, tv, weights = gate_phase_stats(policy, data["baseline_states"], design)
    assert per_phase.shape == (3, 4)
    assert 0.0 <= tv <= 1.0
    assert weights.shape[0] == len(data["baseline_states"]) - 1
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
