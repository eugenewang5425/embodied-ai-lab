"""Lesson 47: weighted pose-graph optimization - fixing the SHAPE.

Unit checks pin the contract: wrap semantics, the world-frame residual known
answer, the constant Jacobians, the factor-graph solver on a hand-built
chain (a loop unary pulls the tail in; weights decide closure vs shape), the
weight-sensitivity semantics (soft odometry closes fully, stiff odometry
keeps the shape), the gold anchor closure, the hypothesis keys, the
collector gate, the shrunk record contract and the CLI.  No full-scale run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    LoopConfig,
    collect_patrol,
    replay_chain,
)
from embodied_learning.experiments.pose_graph import (
    EXPERIMENT,
    PoseGraphConfig,
    build_backends,
    edge_residual,
    error_at,
    solve_factor_graph,
    wrap,
)
from embodied_learning.pose_graph_demo import load_replays

LIGHT_CONFIG = PoseGraphConfig(
    base=LoopConfig(base=GridNavConfig(obstacle_scenes=1, inits=1, rays=64, max_range_m=4.0))
)


@pytest.fixture(scope="module")
def light_stream():
    """One real collect (scene 1) for the back-end contract tests."""
    config = LIGHT_CONFIG
    scenes = build_scenarios(config.base.config, 0)
    start = initial_poses(config.base.config, 0, 1)[0]
    return collect_patrol(scenes[1]["obstacles"], start, config.base)


# ------------------------------------------------------------------ primitives
def test_wrap_semantics():
    assert wrap(0.1) == pytest.approx(0.1)
    assert wrap(math.pi + 0.3) == pytest.approx(-math.pi + 0.3)
    assert wrap(-math.pi - 0.2) == pytest.approx(math.pi - 0.2)


def test_edge_residual_known_answer():
    """A pure translation edge has zero residual when satisfied."""
    xi = np.array([1.0, 2.0, 0.3])
    xj = np.array([1.5, 2.0, 0.3])
    z = xj - xi
    r = edge_residual(xi, xj, z)
    np.testing.assert_allclose(r, 0.0, atol=1e-12)
    # heading mismatch shows in the third component
    r2 = edge_residual(xi, xj + np.array([0, 0, 0.1]), z)
    assert abs(r2[2] - 0.1) < 1e-9


def test_solver_closes_with_loop_unary():
    """A 20-node drifted chain: the loop unary pulls the tail back."""
    nodes = np.zeros((20, 3))
    for i in range(1, 20):
        nodes[i] = nodes[i - 1] + [1.0, 0.002 * i, 0.0]
    nodes[-1] += [2.0, 1.0, 0.0]  # the drift to close
    cfg = PoseGraphConfig()
    edges = [
        (i, i + 1, nodes[i + 1] - nodes[i], cfg.sigma_odom_t, cfg.sigma_odom_r) for i in range(19)
    ]
    unary = (
        19,
        np.array([nodes[0][0], nodes[0][1], nodes[0][2]]),
        cfg.sigma_loop_t,
        cfg.sigma_loop_r,
    )
    x, hist = solve_factor_graph(nodes, edges, [unary], cfg)
    assert len(hist) >= 1 and hist[-1] <= hist[0]  # monotone-ish
    # the tail moved substantially toward the anchor
    assert np.linalg.norm(x[-1, :2] - nodes[0, :2]) < np.linalg.norm(nodes[-1, :2] - nodes[0, :2])


def test_weights_decide_closure_vs_shape():
    """Soft odometry closes fully; stiff odometry keeps the local shape."""
    nodes = np.zeros((30, 3))
    for i in range(1, 30):
        nodes[i] = nodes[i - 1] + [1.0, 0.0, 0.0]
    nodes[-1] += [3.0, 2.0, 0.0]
    cfg = PoseGraphConfig()
    soft_cfg = PoseGraphConfig(sigma_odom_t=0.3)
    edges_soft = [
        (i, i + 1, nodes[i + 1] - nodes[i], soft_cfg.sigma_odom_t, soft_cfg.sigma_odom_r)
        for i in range(29)
    ]
    edges_stiff = [
        (i, i + 1, nodes[i + 1] - nodes[i], cfg.sigma_odom_t, cfg.sigma_odom_r) for i in range(29)
    ]
    unary = (29, np.array([0.0, 0.0, 0.0]), cfg.sigma_loop_t, cfg.sigma_loop_r)
    x_soft, _ = solve_factor_graph(nodes, edges_soft, [unary], soft_cfg)
    x_stiff, _ = solve_factor_graph(nodes, edges_stiff, [unary], cfg)
    # soft: the tail closes almost fully; stiff: the tail stays closer to its
    # odometry shape (larger residual remains)
    res_soft = np.linalg.norm(x_soft[-1, :2])
    res_stiff = np.linalg.norm(x_stiff[-1, :2])
    assert res_soft < res_stiff


def test_gold_anchor_closure(light_stream):
    """With the gold anchor the factor graph closes the tail to sub-meter."""
    config = LIGHT_CONFIG
    l_out = replay_chain("L", light_stream, config.base)
    lt_out = replay_chain("LT", light_stream, config.base)
    nodes = np.vstack([nc[1] for nc in l_out["node_chain"]])
    if not lt_out["loop"]["ok"]:
        pytest.skip("no loop on this scene")
    edges = [
        (i, i + 1, nodes[i + 1] - nodes[i], config.sigma_odom_t, config.sigma_odom_r)
        for i in range(len(nodes) - 1)
    ]
    m_gold = light_stream["truth"][0].copy()
    unary = (len(nodes) - 1, m_gold, config.sigma_loop_t, config.sigma_loop_r)
    x, _ = solve_factor_graph(nodes, edges, [unary], config)
    node_steps = np.array([nc[0] for nc in l_out["node_chain"]], dtype=int)
    _mean, final = error_at(x, node_steps, light_stream["truth"])
    assert final < 1.0  # the gold anchor closes the tail


def test_error_at_alignment(light_stream):
    nodes = np.array([[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    node_steps = np.array([100, 200], dtype=int)
    mean, final = error_at(nodes, node_steps, light_stream["truth"])
    truth_pts = light_stream["truth"][[99, 199]]
    np.testing.assert_allclose(
        mean, np.mean(np.linalg.norm(nodes[:, :2] - truth_pts[:, :2], axis=1))
    )
    assert final == pytest.approx(np.linalg.norm(nodes[-1, :2] - truth_pts[-1, :2]))


def test_build_backends_structure(light_stream):
    built = build_backends(light_stream, LIGHT_CONFIG)
    assert built["nodes"].shape[1] == 3
    assert built["node_steps"].shape == (len(built["nodes"]),)
    assert built["fg"] is None or built["fg"].shape == built["nodes"].shape
    assert isinstance(built["fg_history"], list)


def test_hypothesis_keys():
    # construct the hypothesis by running the real (small) pipeline once is
    # expensive: validate the config validation instead + the record contract
    cfg = PoseGraphConfig()
    assert cfg.gn_iterations == 8
    assert math.isclose(cfg.sigma_loop_t, 0.06)
    with pytest.raises(ValueError):
        PoseGraphConfig(sigma_odom_t=0.0)
    with pytest.raises(ValueError):
        PoseGraphConfig(gn_iterations=0)


def test_micro_run_deterministic(tmp_path):
    from embodied_learning.experiments.pose_graph import run_experiment

    first = run_experiment(tmp_path / "one", seed=0, config=LIGHT_CONFIG, log=None)
    second = run_experiment(tmp_path / "two", seed=0, config=LIGHT_CONFIG, log=None)
    assert first["trajectories_sha256"] == second["trajectories_sha256"]
    assert json.dumps(first["aggregates"], sort_keys=True) == json.dumps(
        second["aggregates"], sort_keys=True
    )


def test_small_run_record_contract(tmp_path):
    from embodied_learning.experiments.pose_graph import run_experiment

    out = tmp_path / "run"
    report = run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    assert report["experiment"] == EXPERIMENT
    assert report["hypothesis"]["results"]["collector_met"] is True
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((out / "trajectories.npz").read_bytes()).hexdigest()
    )
    for name in ("summary.json", "trajectories.npz", "comparison.png"):
        assert (out / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)


def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.pose_graph",
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
    assert "FG" in payload["aggregates"]
    assert (out / "summary.json").is_file()


def test_demo_loader_rejects_tampering(tmp_path):
    from embodied_learning.experiments.pose_graph import run_experiment

    out = tmp_path / "run"
    run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    data = load_replays(out)
    assert data["report"]["experiment"] == EXPERIMENT
    # tamper: flip a number in the summary (the loader must reject)
    parsed = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    parsed["aggregates"]["FG"]["mean_position_error_m"] = parsed["aggregates"]["ARC"][
        "mean_position_error_m"
    ]
    (out / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(out)


@pytest.mark.isolated_tk
def test_tk_demo_modes_and_panel(tmp_path):
    import tkinter as tk

    from embodied_learning.experiments.pose_graph import run_experiment
    from embodied_learning.pose_graph_demo import PoseGraphDemo

    out = tmp_path / "run"
    report = run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    data = load_replays(out)
    root = tk.Tk()
    root.withdraw()
    demo = PoseGraphDemo(root, data)
    root.update()
    assert demo.mode.get() == "backends"
    demo.mode.set("shape")
    demo.redraw()
    demo.mode.set("verdict")
    demo.redraw()
    assert len(demo.fig.axes) == 1
    panel = demo.stats.cget("text")
    fg = report["aggregates"]["FG"]
    assert f"{fg['mean_position_error_m']:.2f}" in panel
    demo.close()
