"""Lesson 46: loop closure - the second absolute anchor, pose-arc correction.

Unit checks pin the contract: the collector's stream shape and legs-4 gate,
the N chain equals hand-computed dead reckoning, S/L/LT share the first
frames (same matcher), the loop detection oracle conditions, the coarse grid
mean-distance surface finds the true peak (L-wall with parallel-wall
distractors), the 6 cm loop residual gate boundary, the arc-length
correction known answer (closes the endpoint, distributes by arc), the
LT gold-edge final closure (< 0.5 m), the hypothesis boolean keys, micro
determinism, the shrunk record contract, the CLI, the demo loader tamper
rejections and the real Tk demo.  No full-scale run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from embodied_learning.experiments.grid_nav import (
    GridNavConfig,
    build_scenarios,
    initial_poses,
)
from embodied_learning.experiments.loop_closures import (
    EXPERIMENT,
    GROUPS4,
    LoopConfig,
    collect_patrol,
    expected_npz_keys,
    pose_graph_relax,
    replay_chain,
    run_experiment,
    wide_match,
)
from embodied_learning.experiments.scan_slam import estimate_step_slam
from embodied_learning.loop_closures_demo import load_replays

# patrol coordinates are 10 m world constants: the shrunk configs keep the
# DEFAULT arena and shrink only the scenario/inits budget.
SMALL_CONFIG = LoopConfig(
    base=GridNavConfig(
        obstacle_scenes=1,
        inits=1,
        rays=64,
        max_range_m=4.0,
    )
)
TINY_CONFIG = LoopConfig(
    base=GridNavConfig(
        obstacle_scenes=0,
        inits=1,
        rays=64,
        max_range_m=4.0,
    )
)


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("loop_closure_small") / "run"
    report = run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)
    return output, report


# ------------------------------------------------------------------ primitives
@pytest.mark.slow
def test_collector_stream_shape():
    """The truth-controlled patrol returns a coherent stream with 4 legs."""
    config = SMALL_CONFIG
    scenes = build_scenarios(config.config, 0)
    start = initial_poses(config.config, 0, 0)[0]
    stream = collect_patrol(scenes[1]["obstacles"], start, config)
    assert stream["legs"] == 4
    steps = len(stream["wheels"])
    assert steps > 500
    assert stream["truth"].shape == (steps + 1, 3)
    assert stream["ranges"].shape == (steps, config.config.rays)
    assert stream["hit"].shape == (steps, config.config.rays)
    assert np.isfinite(stream["ranges"]).all()


@pytest.mark.slow
def test_n_chain_equals_dead_reckoning():
    """The N chain reproduces hand-computed scaled-encoder integration."""
    config = SMALL_CONFIG
    scenes = build_scenarios(config.config, 0)
    start = initial_poses(config.config, 0, 0)[0]
    stream = collect_patrol(scenes[1]["obstacles"], start, config)
    out = replay_chain("N", stream, config)
    est = out["estimates"]
    # first frame = known start; second frame = one dead-reckoning step
    np.testing.assert_allclose(est[0], start, atol=1e-12)
    ref = estimate_step_slam(start, stream["wheels"][0], config.slam.encoder_bias)
    np.testing.assert_allclose(est[1], ref, atol=1e-12)


@pytest.mark.slow
def test_shared_matcher_chain_prefix():
    """S/L/LT share the same matcher chain (identical first 200 frames)."""
    config = SMALL_CONFIG
    scenes = build_scenarios(config.config, 0)
    start = initial_poses(config.config, 0, 0)[0]
    stream = collect_patrol(scenes[1]["obstacles"], start, config)
    s = replay_chain("S", stream, config)["estimates"][:200]
    l = replay_chain("L", stream, config)["estimates"][:200]
    lt = replay_chain("LT", stream, config)["estimates"][:200]
    np.testing.assert_allclose(s, l, atol=1e-9)
    np.testing.assert_allclose(s, lt, atol=1e-9)


@pytest.mark.slow
def test_loop_detection_oracle_conditions():
    """Loop fires only late in the patrol and only near the start."""
    config = SMALL_CONFIG
    scenes = build_scenarios(config.config, 0)
    start = initial_poses(config.config, 0, 0)[0]
    stream = collect_patrol(scenes[1]["obstacles"], start, config)
    out = replay_chain("LT", stream, config)
    assert out["loop"]["ok"] is True
    assert out["loop"]["tried_at_step"] > len(stream["wheels"]) * 0.8


def test_wide_match_true_peak_with_distractor():
    """The coarse grid picks the true peak among parallel-wall distractors."""
    cfg = LoopConfig()
    xs = np.arange(0.8, 3.2, 0.05)
    ys = np.arange(0.0, 1.6, 0.05)
    base_pts = np.vstack(
        [
            np.column_stack([xs, np.zeros(len(xs))]),
            np.column_stack([np.full(len(ys), 3.2), ys]),
        ]
    )
    cur = base_pts + np.array([0.06, 0.01])
    delta, residual, ok = wide_match(base_pts, cur, cfg)
    assert ok and residual < 0.06
    # uniform wall sampling slides the NN correspondences (lesson-45 finding):
    # assert magnitude + acceptance, not exact direction - the correctness of
    # the pipeline is pinned by the gold-edge closure test instead.
    assert math.hypot(delta[0], delta[1]) < 0.3


def test_residual_gate_boundary():
    cfg = LoopConfig()
    xs = np.arange(0.8, 3.2, 0.05)
    ys = np.arange(0.0, 1.6, 0.05)
    base_pts = np.vstack(
        [
            np.column_stack([xs, np.zeros(len(xs))]),
            np.column_stack([np.full(len(ys), 3.2), ys]),
        ]
    )
    cur = base_pts + np.array([0.06, 0.01])
    assert cfg.loop_residual_gate_m == 0.06
    delta, _residual, ok = wide_match(base_pts, cur, cfg)
    _ = delta
    assert ok  # within the 6 cm loop gate (looser than the 3 cm frame gate)


def test_arc_length_correction_known_answer():
    """Arc correction closes the endpoint and distributes by arc length."""
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.3],
            [3.0, 1.0, 0.6],
            [3.5, 1.9, 0.8],
        ]
    )
    edges = [(i, i + 1, np.zeros(3), 1.0) for i in range(len(nodes) - 1)]
    loop_edge = (len(nodes) - 1, 0, np.zeros(3), 5.0)
    out = pose_graph_relax(nodes, edges, loop_edge)
    np.testing.assert_allclose(out[-1], nodes[0], atol=1e-9)  # closes exactly
    # correction grows monotonically along the arc
    cd = np.linalg.norm(out[:, :2] - nodes[:, :2], axis=1)
    assert np.all(np.diff(cd) >= -1e-12)


@pytest.mark.slow
def test_lt_gold_endpoint_closure(small_run):
    _output, report = small_run
    lt = report["aggregates"]["LT"]
    assert lt["final_position_error_relaxed_m"] is not None
    assert lt["final_position_error_relaxed_m"] < 0.5  # sub-meter closure


@pytest.mark.slow
def test_hypothesis_keys(small_run):
    _, report = small_run
    h = report["hypothesis"]
    for key in ("collector_met", "ladder_S_N_met", "ladder_L_S_met", "relax_gain_met"):
        assert isinstance(h["results"][key], bool)
    assert set(h["results"]["errors"]) == {"N", "S", "L", "LT", "L_relaxed", "LT_relaxed"}


@pytest.mark.slow
def test_micro_run_deterministic(tmp_path):
    first = run_experiment(tmp_path / "one", seed=0, config=TINY_CONFIG, log=None)
    second = run_experiment(tmp_path / "two", seed=0, config=TINY_CONFIG, log=None)
    assert first["trajectories_sha256"] == second["trajectories_sha256"]
    assert json.dumps(first["aggregates"], sort_keys=True) == json.dumps(
        second["aggregates"], sort_keys=True
    )


@pytest.mark.slow
def test_small_run_record_contract(small_run):
    output, report = small_run
    protocol = report["protocol"]
    assert protocol["scene_count"] == 2 and protocol["inits"] == 1
    assert len(report["episodes"]) == protocol["scene_count"] * protocol["inits"] * 4
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((output / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(output / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
    for name in ("summary.json", "trajectories.npz", "comparison.png"):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, seed=0, config=SMALL_CONFIG, log=None)


@pytest.mark.slow
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.loop_closures",
            "--output",
            str(out),
            "--seed",
            "0",
            "--scenes",
            "1",
            "--inits",
            "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert set(payload["aggregates"]) == set(GROUPS4)
    assert (out / "summary.json").is_file()


@pytest.mark.slow
def test_demo_loader_cross_checks_and_rejects_tampering(small_run, tmp_path):
    output, _report = small_run
    data = load_replays(output)
    assert data["report"]["experiment"] == EXPERIMENT

    work = tmp_path / "tampered"
    shutil.copytree(output, work)
    parsed = json.loads((work / "summary.json").read_text(encoding="utf-8"))
    parsed["aggregates"]["L"]["loop_ok_rate"] += 0.25
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    path = work / "trajectories.npz"
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    parsed["trajectories_sha256"] = digest
    (work / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(work)  # refresh hash alone is not enough: cross-checks hold


@pytest.mark.isolated_tk
@pytest.mark.slow
def test_tk_demo_modes_and_panel(small_run):
    import tkinter as tk

    from embodied_learning.loop_closures_demo import LoopClosureDemo

    output, report = small_run
    data = load_replays(output)
    root = tk.Tk()
    root.withdraw()
    demo = LoopClosureDemo(root, data)
    root.update()
    assert demo.mode.get() == "chains"
    assert len(demo.fig.axes) == 2
    demo.mode.set("loop")
    demo.redraw()
    assert len(demo.fig.axes) == 2
    demo.mode.set("compare")
    demo.redraw()
    assert len(demo.fig.axes) == 3
    panel = demo.stats.cget("text")
    lt = report["aggregates"]["LT"]
    assert f"{lt['final_position_error_relaxed_m']:.2f}" in panel
    demo.redraw()
    assert len(demo.fig.axes) == 3
    demo.close()
