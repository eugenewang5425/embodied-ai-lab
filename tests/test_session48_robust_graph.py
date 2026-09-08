"""Lesson 48: robust pose graphs - poisoned loop edges and the single-kernel dilemma.

Unit checks: plain-LS fragility on a hand chain (a poisoned unary anchor
drags the chain far from odometry), Huber resistance on the same chain
(the from-scratch kernel keeps the odometry shape), the good-loop dilemma
(the from-scratch kernel slows a GOOD loop's closure - the recorded negative
result), the poison geometry contract, build_groups structure, the group
hypothesis keys, the collector gate, the shrunk record contract, the CLI
and the demo loader cross-check.  No full-scale run.
"""

from __future__ import annotations

import hashlib
import json
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
from embodied_learning.experiments.loop_closures import LoopConfig, collect_patrol
from embodied_learning.experiments.pose_graph import PoseGraphConfig
from embodied_learning.experiments.robust_graph import (
    EXPERIMENT,
    GROUPS,
    RobustConfig,
    build_groups,
    expected_npz_keys,
    solve_graph,
)
from embodied_learning.robust_graph_demo import load_replays

LIGHT_CONFIG = RobustConfig(
    base=PoseGraphConfig(
        base=LoopConfig(base=GridNavConfig(obstacle_scenes=1, inits=1, rays=64, max_range_m=4.0))
    )
)


@pytest.fixture(scope="module")
def light_stream():
    scenes = build_scenarios(LIGHT_CONFIG.base.base.config, 0)
    start = initial_poses(LIGHT_CONFIG.base.base.config, 0, 1)[0]
    return collect_patrol(scenes[1]["obstacles"], start, LIGHT_CONFIG.base.base)


def _chain_nodes():
    nodes = np.zeros((12, 3))
    for i in range(1, 12):
        nodes[i] = nodes[i - 1] + [1.0, 0.0, 0.0]
    return nodes


# ------------------------------------------------------------------ solver
def test_plain_ls_poisoned_unary_drags_chain():
    """A poisoned absolute anchor drags the LS solution far from odometry."""
    nodes = _chain_nodes()
    cfg = RobustConfig()
    edges = [
        (i, i + 1, nodes[i + 1] - nodes[i], cfg.base.sigma_odom_t, cfg.base.sigma_odom_r)
        for i in range(11)
    ]
    bad = (11, np.array([11.0, 5.0, 0.0]), cfg.base.sigma_loop_t, cfg.base.sigma_loop_r)
    x, _ = solve_graph(nodes, edges, [bad], cfg, robust=False)
    assert abs(x[6, 1]) > 1.0  # mid-chain dragged in y


def test_huber_resists_the_same_poison():
    """From-scratch Huber keeps the chain near the odometry line."""
    nodes = _chain_nodes()
    cfg = RobustConfig()
    edges = [
        (i, i + 1, nodes[i + 1] - nodes[i], cfg.base.sigma_odom_t, cfg.base.sigma_odom_r)
        for i in range(11)
    ]
    bad = (11, np.array([11.0, 5.0, 0.0]), cfg.base.sigma_loop_t, cfg.base.sigma_loop_r)
    x, _ = solve_graph(nodes, edges, [bad], cfg, robust=True)
    assert abs(x[6, 1]) < 0.5 * 5.0  # much closer to the odometry line


def test_good_loop_dilemma():
    """From-scratch Huber + good anchor: the tail does NOT reach the anchor
    within the budget (the recorded dilemma) while LS does."""
    nodes = _chain_nodes()
    nodes[-1] += [0.0, 3.0, 0.0]  # the drift the good anchor must close
    cfg = RobustConfig()
    edges = [
        (i, i + 1, nodes[i + 1] - nodes[i], cfg.base.sigma_odom_t, cfg.base.sigma_odom_r)
        for i in range(11)
    ]
    good = (11, np.array([11.0, 0.0, 0.0]), cfg.base.sigma_loop_t, cfg.base.sigma_loop_r)
    x_ls, _ = solve_graph(nodes, edges, [good], cfg, robust=False)
    x_hub, _ = solve_graph(nodes, edges, [good], cfg, robust=True)
    d_ls = np.linalg.norm(x_ls[-1, :2] - np.array([11.0, 0.0]))
    d_hub = np.linalg.norm(x_hub[-1, :2] - np.array([11.0, 0.0]))
    assert d_ls < 0.5  # LS closes (10 plain iterations leave ~0.35 m)
    assert d_hub > d_ls  # the kernel slows the closure (the dilemma)


def test_config_and_poison_contract():
    cfg = RobustConfig(poison_rot_deg=90.0, poison_shift_m=1.0)
    assert cfg.poison_rot_deg == 90.0 and cfg.poison_shift_m == 1.0
    with pytest.raises(ValueError):
        RobustConfig(poison_rot_deg=10.0)
    with pytest.raises(ValueError):
        RobustConfig(huber_delta=100.0)
    with pytest.raises(ValueError):
        RobustConfig(gn_iterations=0)


@pytest.mark.slow
def test_build_groups_contract(light_stream):
    built = build_groups(light_stream, LIGHT_CONFIG)
    assert built["nodes"].shape[1] == 3
    if built["bad_unary"] is not None:
        good_m = built["good_unary"][1]
        bad_m = built["bad_unary"][1]
        # the poisoned anchor is a rotated/shifted copy: well away from good
        assert np.linalg.norm(bad_m[:2] - good_m[:2]) > 1.0


@pytest.mark.slow
def test_small_run_record_contract(tmp_path):
    from embodied_learning.experiments.robust_graph import run_experiment

    out = tmp_path / "run"
    report = run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    assert report["experiment"] == EXPERIMENT
    assert report["hypothesis"]["results"]["collector_met"] is True
    assert report["hypothesis"]["results"]["fragility_met"] is True
    assert report["hypothesis"]["results"]["cure_met"] is True
    assert (
        report["trajectories_sha256"]
        == hashlib.sha256((out / "trajectories.npz").read_bytes()).hexdigest()
    )
    with np.load(out / "trajectories.npz", allow_pickle=False) as archive:
        assert set(archive.files) == expected_npz_keys(report)
    from embodied_learning.experiments.robust_graph import run_experiment as re

    with pytest.raises(FileExistsError):
        re(out, seed=0, config=LIGHT_CONFIG, log=None)


@pytest.mark.slow
def test_hypothesis_keys(tmp_path):
    from embodied_learning.experiments.robust_graph import run_experiment

    out = tmp_path / "run"
    report = run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    h = report["hypothesis"]["results"]
    for key in ("collector_met", "fragility_met", "cure_met", "dilemma_recorded"):
        assert isinstance(h[key], bool)
    assert set(h["errors"]) == set(GROUPS)


@pytest.mark.slow
def test_cli_subprocess_end_to_end(tmp_path):
    out = tmp_path / "cli_run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodied_learning.experiments.robust_graph",
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
    assert set(payload["aggregates"]) == set(GROUPS)


@pytest.mark.slow
def test_demo_loader_crosscheck(tmp_path):
    from embodied_learning.experiments.robust_graph import run_experiment

    out = tmp_path / "run"
    run_experiment(out, seed=0, config=LIGHT_CONFIG, log=None)
    data = load_replays(out)
    assert data["report"]["experiment"] == EXPERIMENT
    # tamper: swap the fragility order in the summary
    parsed = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    errs = parsed["hypothesis"]["results"]["errors"]
    errs["LS-B"], errs["HUB-B"] = errs["HUB-B"], errs["LS-B"]
    (out / "summary.json").write_text(
        json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_replays(out)
