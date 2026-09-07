"""Lesson 42: torch ACT chunked policy on the 2R reaching task.

Unit checks pin the contract: the Transformer forward/attention shapes, the
window/chunk slicing alignment (hand-computed), the BC loss hand-computed
answer, the teacher dataset (deterministic, protocol-aligned, bounded), the
chunked actor's replan cadence and torque mapping, the evaluation path being
the lesson-40 code verbatim, micro-training determinism and convergence, the
shrunk end-to-end record contract (npz keys, sha256, embedded lesson-40 SAC
citation, attention maps), the teacher digest recomputation, the demo loader's
tamper rejections, the CLI and the real Tk demo.  No real training is run.
"""

from __future__ import annotations

import hashlib
import json
import math
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
import torch

from embodied_learning.act_torch_demo import load_replays
from embodied_learning.experiments import act_torch_reaching as mod
from embodied_learning.experiments import rl_arm_reaching as lesson40
from embodied_learning.experiments.act_torch_reaching import (
    ATTENTION_PLAN_INDEX,
    EXPERIMENT,
    ACTConfig,
    ActTransformer,
    bc_loss,
    build_teacher_dataset,
    build_windows,
    dataset_digest,
    evaluate_act,
    expected_npz_keys,
    observation_for,
    run_experiment,
    train_act,
)
from embodied_learning.experiments.rl_arm_reaching import (
    INITIAL_Q,
    TORQUE_LIMIT_NM,
    ArmReachingEnv,
    run_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_RECORD = PROJECT_ROOT / "results" / "rl_arm_reaching_2026-09-06"

SMALL_CONFIG = ACTConfig(
    teacher_episodes=4,
    episode_steps=200,
    epochs=30,
    windows_per_epoch=256,
    batch_size=64,
    chunk_t=2,
    chunk_h=4,
    d_model=32,
    n_heads=2,
    n_layers=1,
    ffn=64,
    eval_goal_count=5,
    eval_subset=3,
    eval_every_epochs=10,
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("act_torch_small") / "run"
    report = run_experiment(
        output, seed=0, config=SMALL_CONFIG, train_seeds=2, sac_reference=SAC_RECORD, log=None
    )
    return output, report


# --------------------------------------------------------------- torch policy
def test_transformer_forward_and_attention_contract():
    """(B,T,8) obs -> (B,H,2) tanh chunk; attention maps (T,T)/(H,T) rows sum 1."""
    torch.manual_seed(0)
    policy = ActTransformer(d_model=32, n_heads=2, n_layers=1, ffn=64, chunk_t=3, chunk_h=5)
    obs = torch.randn(7, 3, mod.OBS_DIM)
    chunk = policy(obs)
    assert chunk.shape == (7, 5, 2)
    assert chunk.min() >= -1.0 and chunk.max() <= 1.0
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    # deterministic inference on the same input (eval + no_grad twice)
    policy.eval()
    with torch.no_grad():
        first = policy(obs)
        again = policy(obs)
    assert torch.equal(first, again)
    assert torch.allclose(chunk, first, atol=1e-5)  # grad/no-grad kernels differ ~1e-7
    enc_map, cross_map = policy.attention_maps(obs[:1])
    assert enc_map.shape == (3, 3) and cross_map.shape == (5, 3)
    for map_array in (enc_map, cross_map):
        assert np.isfinite(map_array).all()
        assert map_array.min() >= -1e-6
        np.testing.assert_allclose(map_array.sum(axis=-1), 1.0, atol=1e-4)


def test_chunk_slicing_alignment_known_answers():
    """Window t = obs[t-T+1..t] (left clamp), chunk t = actions[t..t+H-1] (right clamp)."""
    obs = np.array([[float(100 + t)] * 8 for t in range(6)])  # rows carry their index
    actions = np.array([[float(t), -float(t)] for t in range(5)])
    windows, chunks = build_windows([obs], [actions], chunk_t=3, chunk_h=4)
    assert windows.shape == (5, 3, 8) and chunks.shape == (5, 4, 2)
    assert windows.dtype == np.float32 and chunks.dtype == np.float32
    # t=0: window clamped to the first observation three times; chunk = 0,1,2,3
    np.testing.assert_array_equal(windows[0], np.tile(obs[0], (3, 1)))
    np.testing.assert_array_equal(chunks[0], [[0, 0], [1, -1], [2, -2], [3, -3]])
    # t=3: window = obs[1..3]; chunk = 3,4,4,4 (right-clamped repeat of the last action)
    np.testing.assert_array_equal(windows[3, 0], obs[1])
    np.testing.assert_array_equal(windows[3, -1], obs[3])
    np.testing.assert_array_equal(chunks[3], [[3, -3], [4, -4], [4, -4], [4, -4]])


def test_bc_loss_hand_computed():
    """L = MSE + w_l1 * L1 on normalized action chunks (hand-computed)."""
    pred = torch.tensor([[[0.5, -0.5]], [[0.0, 1.0]]])  # (2, 1, 2)
    target = torch.tensor([[[0.0, 0.0]], [[0.0, 0.5]]])
    diff = pred - target
    mse = float((diff**2).mean())  # (0.25+0.25+0+0.25)/4 = 0.1875
    l1 = float(diff.abs().mean())  # (0.5+0.5+0+0.5)/4 = 0.375
    total, mse_out, l1_out = bc_loss(pred, target, w_l1=0.1)
    assert mse_out == pytest.approx(mse)
    assert l1_out == pytest.approx(l1)
    assert total == pytest.approx(mse + 0.1 * l1)
    zero_total, _, _ = bc_loss(pred, pred, w_l1=0.1)
    assert zero_total == 0.0
    with pytest.raises(RuntimeError):
        bc_loss(pred, torch.zeros(2, 1, 3), w_l1=0.1)  # shape mismatch must fail loudly


# -------------------------------------------------------------------- teacher
def test_teacher_dataset_deterministic_and_aligned():
    """Same seed -> bitwise same rollouts; obs/actions follow the protocol state."""
    config = ACTConfig(teacher_episodes=2, episode_steps=60)
    first_obs, first_act, first_goals = build_teacher_dataset(0, config)
    second_obs, second_act, second_goals = build_teacher_dataset(0, config)
    for a, b in zip((*first_obs, *first_act), (*second_obs, *second_act), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.asarray(first_goals), np.asarray(second_goals))
    assert dataset_digest(first_obs, first_act) == dataset_digest(second_obs, second_act)
    # the first observation of every episode is the protocol vector at the start pose
    for observation, goal in zip(first_obs, first_goals, strict=True):
        np.testing.assert_allclose(
            observation[0], observation_for(INITIAL_Q, [0.0, 0.0], goal), atol=1e-12
        )
        assert np.all(np.abs(first_act[0]) <= 1.0)  # teacher torques fit the motor limit
    # two different seeds disagree (the digest is seed sensitive)
    other_obs, other_act, _other_goals = build_teacher_dataset(1, config)
    assert dataset_digest(first_obs, first_act) != dataset_digest(other_obs, other_act)


def test_chunked_actor_replans_every_chunk_t():
    """Full episode: ceil(steps / chunk_t) forwards, torques within the limit."""
    policy = ActTransformer(d_model=32, n_heads=2, n_layers=1, ffn=64)
    env = ArmReachingEnv([9, 9], terminate_on_arrival=False)
    goal = np.asarray([0.35, 0.30])
    actor = mod.ActActor(policy, chunk_t=4)
    actor.reset()
    env.reset(goal=goal)
    steps = 0
    for _ in range(env.episode_steps):
        torque = actor(env.state, env.goal, env.last_torque)
        assert np.all(np.abs(torque) <= TORQUE_LIMIT_NM + 1e-12)
        assert np.all(np.isfinite(torque))
        env.step(torque)
        steps += 1
    assert steps == 500
    assert actor.plans == math.ceil(500 / 4) == 125
    assert actor.captured_window is None  # no capture requested
    # a fresh episode after reset replans from zero
    actor.reset()
    assert actor.plans == 0 and actor._chunk is None


def test_evaluation_path_is_lesson40_code():
    """The eval path IS lesson 40's run_episode (identity) with equal outputs."""
    assert mod.run_episode is lesson40.run_episode
    assert mod.aggregate_episodes is lesson40.aggregate_episodes
    assert mod.observation_for is lesson40.observation_for
    assert mod.make_eval_goals is lesson40.make_eval_goals
    assert mod.baseline_actor is lesson40.baseline_actor

    config = ACTConfig(chunk_t=2, chunk_h=4)

    class ZeroPolicy(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(obs.shape[0], config.chunk_h, 2)

    env_a = ArmReachingEnv([4, 4], terminate_on_arrival=False)
    env_b = ArmReachingEnv([4, 4], terminate_on_arrival=False)
    goal = np.asarray([0.35, -0.30])

    def zero_actor(_state, _goal, _prev_torque):
        return np.zeros(2)

    manual_summary, manual_points = run_episode(zero_actor, env_a, goal, record=True)
    episodes, points, actor = evaluate_act(ZeroPolicy(), env_b, [goal], config, record=True)
    assert episodes[0] == manual_summary
    np.testing.assert_array_equal(points[0], manual_points)
    assert actor.plans == math.ceil(manual_summary["steps"] / config.chunk_t)


# ------------------------------------------------------------------- training
def test_micro_training_deterministic_and_converging():
    """Fixed seeds reproduce the run bitwise; the val loss goes down."""
    config = ACTConfig(
        teacher_episodes=2,
        episode_steps=100,
        epochs=60,
        windows_per_epoch=128,
        batch_size=32,
        d_model=32,
        n_heads=2,
        n_layers=1,
        ffn=64,
        eval_goal_count=4,
        eval_subset=2,
        eval_every_epochs=30,
    )
    obs, actions, _goals = build_teacher_dataset(0, config)
    windows, chunks = build_windows(obs, actions, config.chunk_t, config.chunk_h)
    val_mask = (np.arange(len(windows)) % mod.VAL_STRIDE) == 0
    train_idx, val_idx = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    eval_env = ArmReachingEnv([0, 14001], terminate_on_arrival=False)
    eval_goals = mod.make_eval_goals(0, config.eval_goal_count)

    def one_run():
        return train_act(
            windows,
            chunks,
            train_idx,
            val_idx,
            eval_env,
            eval_goals,
            config,
            master_seed=0,
            seed_index=0,
            log=None,
        )

    first, second = one_run(), one_run()
    for name in ("train_loss_curve", "val_total_curve", "val_mse_curve", "val_l1_curve"):
        np.testing.assert_array_equal(first[name], second[name])
        assert np.isfinite(first[name]).all()
    for weights_a, weights_b in zip(
        first["policy"].parameters(), second["policy"].parameters(), strict=True
    ):
        assert np.array_equal(weights_a.detach().numpy(), weights_b.detach().numpy())
    assert first["eval_curve"] == second["eval_curve"]
    # learning happened: final training loss and validation loss beat the
    # untrained validation baseline
    assert first["final_train_loss"] < first["initial_val_total"]
    assert first["val_total_curve"][-1] < first["val_total_curve"][0]
    assert 0.0 < first["final_val_rmse_mnm"] <= TORQUE_LIMIT_NM * 1000
    assert len(first["train_loss_curve"]) == config.epochs


# ---------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == EXPERIMENT
    assert report["schema_version"] == 1
    assert report["training"]["train_seeds"] == 2
    assert report["geometry_audit"]["pose_count"] == 49
    assert report["baseline"]["aggregate"]["episodes"] == 5
    assert report["baseline"]["aggregate"]["successes"] == 5  # the analytic gate
    assert report["teacher"]["episodes"] == SMALL_CONFIG.teacher_episodes
    assert report["teacher"]["env_steps"] == (
        SMALL_CONFIG.teacher_episodes * SMALL_CONFIG.episode_steps
    )
    assert report["sac_reference"]["loaded"] is True
    assert report["sac_reference"]["across_seeds"]["successes"] == 0
    assert len(report["comparison"]) == 4  # baseline + SAC cited + 2 ACT seeds
    assert report["comparison"][1]["label"].startswith("纯 numpy SAC")
    assert report["hyperparameters"]["chunk_t"] == SMALL_CONFIG.chunk_t
    for record in report["bc_evaluation"]["per_seed"]:
        assert record["aggregate"]["episodes"] == 5
        assert record["aggregate"]["successes"] == sum(
            episode["outcome"] == "arrived" for episode in record["per_goal"]
        )
        for episode in record["per_goal"]:
            assert episode["outcome"] in ("arrived", "timeout", "joint_limit", "failure")
        assert len(record["eval_curve"]) == 3  # epochs 10/20/30
    assert report["hypothesis"]["per_seed_successes"] == [
        record["aggregate"]["successes"] for record in report["bc_evaluation"]["per_seed"]
    ]
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        goals = archive["eval_goals"]
        assert goals.shape == (5, 2)
        np.testing.assert_allclose(goals, mod.make_eval_goals(0, 5), atol=1e-12)
        assert archive["eval_truth_0_0"].shape[1:] == (3, 2)
        assert archive["sac_truth_0_0"].shape[1:] == (3, 2)
        assert archive["attention_encoder_0"].shape == (SMALL_CONFIG.chunk_t,) * 2
        assert archive["eval_outcome_0"].shape == (5,)
    for name in ("summary.json", "trajectories.npz", "training_curves.png", "comparison.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(
            output, seed=0, config=SMALL_CONFIG, train_seeds=2, sac_reference=SAC_RECORD, log=None
        )


def test_teacher_digest_recomputable_from_summary(small_run):
    """The recorded teacher hash is the digest of the regenerated dataset."""
    _output, report = small_run
    config = ACTConfig(
        teacher_episodes=SMALL_CONFIG.teacher_episodes,
        episode_steps=SMALL_CONFIG.episode_steps,
    )
    observations, actions, _goals = build_teacher_dataset(report["master_seed"], config)
    assert report["teacher"]["digest_sha256"] == dataset_digest(observations, actions)


def test_attention_maps_in_archive(small_run):
    """Archived attention maps are proper softmax rows at the protocol shapes."""
    _output, report = small_run
    chunk_t = report["hyperparameters"]["chunk_t"]
    chunk_h = report["hyperparameters"]["chunk_h"]
    with np.load(Path(_output) / "trajectories.npz", allow_pickle=False) as archive:
        for seed_index in range(report["training"]["train_seeds"]):
            encoder_map = archive[f"attention_encoder_{seed_index}"]
            cross_map = archive[f"attention_decoder_{seed_index}"]
            assert encoder_map.shape == (chunk_t, chunk_t)
            assert cross_map.shape == (chunk_h, chunk_t)
            for map_array in (encoder_map, cross_map):
                assert np.isfinite(map_array).all()
                assert map_array.min() >= 0.0
                np.testing.assert_allclose(map_array.sum(axis=-1), 1.0, atol=1e-4)
    first = report["bc_evaluation"]["per_seed"][0]
    assert first["attention_plan_index"] == ATTENTION_PLAN_INDEX


def test_demo_loader_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    summary_text = (work / "summary.json").read_text(encoding="utf-8")
    parsed = json.loads(summary_text)

    # (1) summary ACT successes disagree with the archived outcomes
    parsed["bc_evaluation"]["per_seed"][0]["aggregate"]["successes"] += 1
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="ACT successes disagree"):
        load_replays(work)
    (work / "summary.json").write_text(summary_text, encoding="utf-8")
    parsed = json.loads(summary_text)

    # (2) archive bytes tampered -> checksum mismatch
    path = work / "trajectories.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["train_loss_curve_0"] = payload["train_loss_curve_0"] * 2.0
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    # (3) an array removed with the hash refreshed -> unexpected key set
    del payload["baseline_outcome"]
    np.savez_compressed(path, **payload)
    parsed["trajectories_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unexpected archive arrays"):
        load_replays(work)

    # (4) showcase goals swapped in the archive (fresh pristine copy)
    work4 = tmp_path / "tampered_goals"
    shutil.copytree(output, work4)
    path4 = work4 / "trajectories.npz"
    with np.load(path4, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["eval_goals"] = payload["eval_goals"] + 0.5
    np.savez_compressed(path4, **payload)
    parsed4 = json.loads((work4 / "summary.json").read_text(encoding="utf-8"))
    parsed4["trajectories_sha256"] = hashlib.sha256(path4.read_bytes()).hexdigest()
    (work4 / "summary.json").write_text(
        json.dumps(parsed4, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="showcase goals"):
        load_replays(work4)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.act_torch_reaching",
            "--output",
            str(out),
            "--seed",
            "7",  # a different master seed -> goals differ from the lesson-40 stream
            "--train-seeds",
            "1",
            "--teacher-episodes",
            "3",
            "--episode-steps",
            "200",
            "--epochs",
            "20",
            "--windows-per-epoch",
            "128",
            "--batch-size",
            "32",
            "--chunk-t",
            "2",
            "--chunk-h",
            "4",
            "--d-model",
            "32",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--ffn",
            "64",
            "--eval-goals",
            "4",
            "--eval-every-epochs",
            "10",
            "--eval-subset",
            "2",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["baseline"]["episodes"] == 4
    assert payload["baseline"]["successes"] == 4
    assert payload["seeds"][0]["episodes"] == 4
    assert payload["sac_reference"]["loaded"] is False  # 4 goals != the lesson-40 goals
    assert "per_seed_successes" in payload["hypothesis"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    with np.load(out / "trajectories.npz", allow_pickle=False) as archive:
        assert not any(key.startswith("sac_") for key in archive.files)
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.act_torch_reaching",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )
    assert again.returncode != 0  # new directories are never overwritten


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.act_torch_demo import ActTorchDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = ActTorchDemo(root, data)
    root.update()
    assert demo.mode.get() == "training"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "预算" in panel and "验证 RMSE" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("trajectories")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    baseline = report["baseline"]["aggregate"]
    assert "IK+PD 教师" in panel
    assert f"{baseline['successes']}/{baseline['episodes']}" in panel  # summary numbers
    assert "40 课引用" in panel
    demo.mode.set("attention")
    demo.redraw()
    assert len(demo.fig.axes) == 6  # four panels + two attention colorbars
    panel = demo.stats.cget("text")
    assert "注意力捕获" in panel and "无 CVAE" in panel
    demo.close()
