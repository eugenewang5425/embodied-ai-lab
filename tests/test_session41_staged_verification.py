"""Lesson 41: staged minimal verification - base first, then minimal fixes.

Unit checks pin the staged contract: a zero linear correction reproduces the
lesson-7 run bitwise, the residual limit and total control clip always hold,
the handoff window extraction matches the capture->tolerance transition, the
supervised zero-target fit returns exact zeros, the micro training and the
MLP fit are deterministic, the improvement verdict boundaries match the
protocol; shrunk end-to-end checks cover the record contract, the lesson-7
acceptance caliber, the demo loader's tamper rejections, the CLI and the real
Tk demo. No real training is run.
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

from embodied_learning.experiments.staged_verification import (
    CONTROL_LIMIT,
    EVAL_EPISODE_STEPS,
    GEAR,
    RESIDUAL_LIMIT_N,
    LinearCorrector,
    MLPCorrector,
    StagedConfig,
    correction_features,
    expected_npz_keys,
    fit_linear_correction,
    handoff_window,
    improvement_verdict,
    run_corrected_episode,
    run_experiment,
    run_linear_group,
    stage1_handoff,
)
from embodied_learning.experiments.swingup_comparison import (
    Scenario,
    recovery_metrics,
    run_scenario,
)
from embodied_learning.staged_demo import load_replays
from embodied_learning.swingup import design_swingup_lqr

SMALL_CONFIG = StagedConfig(exploration_rollouts=2, eval_repeats=2, mlp_epochs=5)


@pytest.fixture(scope="module")
def design():
    return design_swingup_lqr()


@pytest.fixture(scope="module")
def base_rollout(design):
    """The frozen lesson-7 down run, shared by the contract checks."""
    arrays, metadata = run_scenario(Scenario("down", "down"), design)
    return arrays, metadata


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    """One shrunk end-to-end experiment shared by the record-level checks."""
    output = tmp_path_factory.mktemp("staged_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)
    return output, report


# ------------------------------------------------- zero guard + clip contract
def test_zero_correction_reproduces_base_bitwise(design, base_rollout):
    """W = 0, b = 0 through the corrected pipeline: the lesson-7 run, bit for bit."""
    arrays, _metadata = base_rollout
    zero = LinearCorrector(np.zeros((1, 4)), 0.0, RESIDUAL_LIMIT_N)
    corrected, reason = run_corrected_episode(zero, design, horizon=EVAL_EPISODE_STEPS)
    assert np.array_equal(corrected["states"], arrays["states"])
    assert np.array_equal(corrected["controls"], arrays["controls"])
    assert np.all(corrected["residuals_n"] == 0.0)
    assert corrected["active"].any()  # the window opens on the base capture
    metrics = recovery_metrics(
        corrected, {"failure_reason": reason}, design.controller.reference, design.dt
    )
    assert metrics["recovered"] and metrics["settled_at_s"] == pytest.approx(4.76)


def test_residual_limit_and_control_clip_contract(design):
    """|delta_u| <= 5 N by construction, |u| <= 3 normalized; invalid inputs rejected."""
    huge = LinearCorrector(np.full((1, 4), 1e6), 1e6, RESIDUAL_LIMIT_N)
    arrays, _reason = run_corrected_episode(huge, design, horizon=200)
    residuals = np.asarray(arrays["residuals_n"], dtype=float)
    assert np.all(np.abs(residuals) <= RESIDUAL_LIMIT_N + 1e-12)
    assert np.all(np.abs(residuals[arrays["active"]]) >= RESIDUAL_LIMIT_N - 1e-9)
    assert np.all(np.abs(arrays["controls"]) <= CONTROL_LIMIT)
    assert np.max(np.abs(arrays["controls"] * GEAR)) <= 300.0 + 1e-9
    with pytest.raises(ValueError):
        LinearCorrector(np.zeros((1, 4)), 0.0, 0.0)  # limit must be positive
    with pytest.raises(ValueError):
        LinearCorrector(np.full((1, 4), np.nan), 0.0, RESIDUAL_LIMIT_N)
    with pytest.raises(ValueError):
        run_corrected_episode(None, design, horizon=8, noise_rng=np.random.default_rng(0))


# ------------------------------------------------------------ handoff window
def test_handoff_window_extraction(design, base_rollout):
    """[capture, end) is exactly the 0.3 rad -> 0.01 rad transition of the run."""
    arrays, _metadata = base_rollout
    reference_theta = design.controller.reference[1]
    window = handoff_window(arrays["modes"], arrays["states"], reference_theta)
    assert window is not None
    capture, end = window
    assert str(arrays["modes"][capture]) == "balance"
    assert capture * design.dt == pytest.approx(1.68)
    alpha = np.abs(
        np.arctan2(
            np.sin(arrays["states"][:, 1] - reference_theta),
            np.cos(arrays["states"][:, 1] - reference_theta),
        )
    )
    assert alpha[capture] < 0.3
    assert np.all(alpha[capture:end] > 0.01)  # every window step is between the bounds
    assert alpha[end] <= 0.01  # the step that ends the window enters the tolerance
    handoff = stage1_handoff(
        arrays["states"],
        arrays["modes"],
        arrays["controls"],
        design.controller.reference,
        design.dt,
    )
    assert handoff["capture_step"] == capture and handoff["end_step"] == end
    assert handoff["steps"] == end - capture
    features = np.asarray([correction_features(s, reference_theta) for s in handoff["states"][:-1]])
    assert features.shape == (handoff["steps"], 4) and np.isfinite(features).all()
    np.testing.assert_allclose(handoff["controls"], arrays["controls"][capture:end])


# --------------------------------------------------- supervised init and fits
def test_supervised_zero_target_fit_returns_exact_zeros(design, base_rollout):
    """Least squares against du = 0 targets gives exact zeros; real fits recover lines."""
    arrays, _metadata = base_rollout
    handoff = stage1_handoff(
        arrays["states"],
        arrays["modes"],
        arrays["controls"],
        design.controller.reference,
        design.dt,
    )
    features = handoff["features"]
    weight, bias = fit_linear_correction(features, np.zeros(len(features)))
    assert np.all(weight == 0.0) and bias == 0.0
    rng = np.random.default_rng(0)
    true_weight = np.array([[0.02, -0.03, 0.01, 0.005]])
    true_bias = 0.04
    states = rng.normal(size=(50, 4))
    targets = (states @ true_weight[0]) + true_bias
    fit_weight, fit_bias = fit_linear_correction(states, targets)
    np.testing.assert_allclose(fit_weight, true_weight, atol=1e-10)
    assert fit_bias == pytest.approx(true_bias, abs=1e-10)
    net = MLPCorrector((8, 8), (0, 1, 2), RESIDUAL_LIMIT_N)
    assert max(abs(net.residual(row)) for row in features) == 0.0  # zero-init output layer


def test_mlp_fit_is_deterministic_and_reduces_loss(design, base_rollout):
    """Same seed, same fit bitwise; the MSE drops and clipped outputs stay finite."""
    arrays, _metadata = base_rollout
    handoff = stage1_handoff(
        arrays["states"],
        arrays["modes"],
        arrays["controls"],
        design.controller.reference,
        design.dt,
    )
    features, targets = handoff["features"], handoff["controls"] * 0.05

    def one_fit():
        net = MLPCorrector((8, 8), (0, 7, 9000), RESIDUAL_LIMIT_N)
        history = net.fit_from(features, targets, epochs=60, lr=1e-3)
        return net, history

    net_a, history_a = one_fit()
    net_b, history_b = one_fit()
    np.testing.assert_array_equal(history_a, history_b)
    for weight_a, weight_b in zip(net_a.trunk.weights, net_b.trunk.weights, strict=True):
        assert np.array_equal(weight_a, weight_b)
    assert history_a[-1] < history_a[0]
    outputs = np.asarray([net_a.residual(row) for row in features])
    assert np.isfinite(outputs).all() and np.all(np.abs(outputs) <= RESIDUAL_LIMIT_N)


def test_linear_group_micro_training_is_deterministic(design, base_rollout):
    """Fixed seeds reproduce the whole group (exploration, fit, evaluation) bitwise."""
    arrays, _metadata = base_rollout
    handoff = stage1_handoff(
        arrays["states"],
        arrays["modes"],
        arrays["controls"],
        design.controller.reference,
        design.dt,
    )
    handoff["baseline"] = {"successes": 2, "episodes": 2, "median_settled_at_s": 4.76}
    config = StagedConfig(exploration_rollouts=2, eval_repeats=1, mlp_epochs=5)

    def one_group():
        record, _archive = run_linear_group(
            design,
            design.controller.reference,
            design.dt,
            handoff,
            config=config,
            master_seed=0,
            log=None,
        )
        return record

    first, second = one_group(), one_group()
    assert first["parameters"]["W"] == second["parameters"]["W"]
    assert first["parameters"]["b"] == second["parameters"]["b"]
    assert np.isfinite(first["parameters"]["W"]).all()
    assert first["evaluation"]["median_settled_at_s"] == second["evaluation"]["median_settled_at_s"]
    assert (
        first["training"]["exploration"]["settled_times_s"]
        == (second["training"]["exploration"]["settled_times_s"])
    )
    assert first["evaluation"]["successes"] == first["evaluation"]["episodes"]


def test_improvement_verdict_boundaries():
    """not_degrade >= 18/20; improved = full success AND a strictly faster median."""
    baseline = {"successes": 20, "episodes": 20, "median_settled_at_s": 4.76}

    def evaluation(successes, settled):
        return {
            "episodes": 20,
            "successes": successes,
            "median_settled_at_s": settled,
        }

    not_degrade, improved = improvement_verdict(evaluation(20, 4.72), baseline)
    assert not_degrade and improved
    not_degrade, improved = improvement_verdict(evaluation(20, 4.76), baseline)
    assert not_degrade and not improved  # equal time is not an improvement
    not_degrade, improved = improvement_verdict(evaluation(18, 4.00), baseline)
    assert not_degrade and not improved  # degraded successes block the gate
    not_degrade, improved = improvement_verdict(evaluation(17, 4.00), baseline)
    assert not not_degrade and not improved
    not_degrade, improved = improvement_verdict(evaluation(20, None), baseline)
    assert not_degrade and not improved


# ---------------------------------------------------------- shrunk end-to-end
def test_small_run_record_contract(small_run):
    output, report = small_run
    assert report["experiment"] == "staged_verification_lesson41"
    step1 = report["step1_base"]
    assert step1["successes"] == step1["episodes"] == SMALL_CONFIG.eval_repeats
    assert step1["deterministic_identical_repeats"] is True
    assert step1["handoff"]["steps"] > 0
    groups = report["groups"]
    assert groups["B_linear"] is not None
    assert groups["B_linear"]["not_degrade"] is True
    improved = groups["B_linear"]["improved"]
    assert groups["C_mlp"] is not None if improved else groups["C_mlp"] is None
    assert groups["D_mlp_large"] is not None if improved else groups["D_mlp_large"] is None
    assert report["verdict"]["mlp_ran"] == bool(improved)
    rows = report["comparison_table"]
    assert [row["group"] for row in rows] == ["A", "B", "C", "D"]
    assert rows[0]["status"] == "ran" and rows[1]["status"] == "ran"
    assert all(row["status"] == "skipped_step3_gate" for row in rows[2:]) or improved
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
        assert archive["lin_eval_active"].dtype == bool
        assert archive["lin_weight"].shape == (1, 4)
    for name in ("summary.json", "trajectories.npz", "comparison.png", "correction.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)


def test_small_run_acceptance_matches_lesson7_caliber(small_run, design, base_rollout):
    """The lesson-7 acceptance is reused verbatim and the base rerun is bitwise stable."""
    output, report = small_run
    reference, dt = design.controller.reference, design.dt
    assert report["protocol"]["settled_tolerances"] == [0.02, 0.01, 0.02, 0.02]
    assert report["protocol"]["minimum_settled_tail_s"] == 2.0
    fresh_states, _metadata = run_scenario(Scenario("down", "down"), design)
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["baseline_states"], fresh_states["states"])
        data = {key: archive[key].copy() for key in archive.files}
    evaluation = report["groups"]["B_linear"]["evaluation"]
    arrays = {
        "states": data["lin_eval_states"],
        "controls": data["lin_eval_controls"],
        "applied_force_n": np.zeros(len(data["lin_eval_controls"])),
        "scheduled_force_n": np.zeros(report["protocol"]["eval_horizon_steps"]),
        "modes": data["lin_eval_modes"],
        "end_flags": np.array([False, False]),
    }
    metrics = recovery_metrics(arrays, {"failure_reason": ""}, reference, dt)
    assert metrics["recovered"] == (evaluation["successes"] == evaluation["episodes"])
    assert metrics["settled_at_s"] == pytest.approx(evaluation["median_settled_at_s"])
    residuals = data["lin_eval_residuals_n"]
    limit = report["protocol"]["residual_limits_n"]["B_linear"]
    assert np.all(np.abs(residuals[data["lin_eval_active"]]) <= limit + 1e-9)
    assert np.all(residuals[~data["lin_eval_active"]] == 0.0)


def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)  # the pristine record passes every cross-check
    assert data["report"]["experiment"] == "staged_verification_lesson41"

    work = tmp_path / "tampered_bytes"
    shutil.copytree(output, work)
    path = work / "trajectories.npz"
    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="checksum"):
        load_replays(work)

    work2 = tmp_path / "tampered_summary"
    shutil.copytree(output, work2)
    parsed = json.loads((work2 / "summary.json").read_text(encoding="utf-8"))
    parsed["step1_base"]["successes"] += 1
    (work2 / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Base successes disagree"):
        load_replays(work2)

    work3 = tmp_path / "tampered_residual"
    shutil.copytree(output, work3)
    path3 = work3 / "trajectories.npz"
    with np.load(path3, allow_pickle=False) as archive:
        payload = {key: archive[key].copy() for key in archive.files}
    payload["lin_eval_residuals_n"] = payload["lin_eval_residuals_n"] + 50.0
    np.savez_compressed(path3, **payload)
    parsed = json.loads((work3 / "summary.json").read_text(encoding="utf-8"))
    parsed["trajectories_sha256"] = hashlib.sha256(path3.read_bytes()).hexdigest()
    (work3 / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exceeds its budget"):
        load_replays(work3)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.staged_verification",
            "--output",
            str(out),
            "--seed",
            "0",
            "--exploration-rollouts",
            "2",
            "--eval-repeats",
            "2",
            "--mlp-epochs",
            "5",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["base"]["successes"] == 2
    assert payload["base"]["median_settled_at_s"] == pytest.approx(4.76)
    assert payload["linear"]["successes"] == payload["linear"]["episodes"] == 2
    assert isinstance(payload["step3_ran"], bool)
    assert payload["conclusion"]
    assert (out / "summary.json").is_file()
    assert (out / "trajectories.npz").is_file()
    assert (out / "comparison.png").is_file()
    again = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.staged_verification",
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

    from embodied_learning.staged_demo import StagedDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = StagedDemo(root, data)
    root.update()
    assert demo.mode.get() == "base"
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert f"{report['step1_base']['successes']}/{report['step1_base']['episodes']}" in panel
    assert "交接段" in panel
    demo.redraw()
    demo.redraw()
    assert len(demo.fig.axes) == 4  # mode switches must not accumulate axes
    demo.mode.set("linear")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "评估" in panel and "残差" in panel  # the linear-correction panel
    demo.mode.set("mlp")
    demo.redraw()
    assert len(demo.fig.axes) == 4
    panel = demo.stats.cget("text")
    assert "对照表" in panel  # the comparison-table panel
    linear = report["groups"]["B_linear"]
    assert f"{linear['evaluation']['successes']}/{linear['evaluation']['episodes']}" in panel
    if report["verdict"]["mlp_ran"]:
        assert "C" in panel and "D" in panel
    else:
        assert "未运行" in panel  # the step-3 gate skipped the MLP groups
    demo.close()
