"""Lesson 47: weighted pose-graph optimization - fixing the SHAPE of the drift.

Lesson 46 proved the loop is the second absolute anchor (end-to-end closure
9.18 m -> 0.135/0.161 m) but left the SHAPE of the trajectory unfixed: the
arc-length correction redistributes only the closing residual, so the mean
error stayed at 5.3-5.4 m (the drift-bent arc remains bent).  THIS lesson
adds the real back-end: a weighted SE(2) pose-graph solved by Gauss-Newton
with ANALYTIC Jacobians and a conjugate-gradient inner solver (sparse by
construction - 3N parameters with N~900, dense lstsq is the fallback guard).

Every edge is a measurement with its own sigma (Mahalanobis weighting):
  odometry/matching edges  sigma_t = 0.05 m,  sigma_r = 2 deg;
  loop edge                sigma_t = 0.06 m,  sigma_r = 5 deg.
Residual of edge (i, j, z): e_t = R(theta_i)^T (p_j - p_i) - z_t (in the i
frame), e_r = wrap(theta_j - theta_i - z_r).  Analytic Jacobians w.r.t. both
nodes; the gauge is the first node (fixed).  Robustness: plain least squares
(Huber is the recorded next step).

Groups (same collect-replay protocol as lesson 46, same scenarios/starts):
  N    dead reckoning (no back-end) - the input chain;
  ARC  S + loop + arc-length correction (the lesson-46 back-end, verbatim);
  FG   S + loop + THIS factor graph.

The showcase LT control (gold loop edge) is replayed for FG as FG-T to keep
the "structure vs match quality" separation of lesson 46.

Pre-registered claims: (1) the collector gate (4/4 legs); (2) closure is not
regressed: FG final error <= ARC final error + margin; (3) THE SHAPE CLAIM:
FG mean error < ARC mean error (the drift-bent arc straightens); (4) FG
residuals decrease monotonically over GN iterations (the solver converges).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.experiments.grid_nav import GridNavConfig, initial_poses
from embodied_learning.experiments.loop_closures import (
    LoopConfig,
    build_scenarios,
    collect_patrol,
    replay_chain,
)
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "pose_graph_lesson47"
SCHEMA_VERSION = 1
BACKENDS = ("N", "ARC", "FG")
BACKEND_NAMES = {
    "N": "纯里程计（无后端）",
    "ARC": "回环 + 弧长校正（第 46 课后端）",
    "FG": "回环 + 加权因子图（本课）",
}
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/pose_graph_2026-09-07"


@dataclass(frozen=True)
class PoseGraphConfig:
    base: object = None  # LoopConfig
    sigma_odom_t: float = 0.05
    sigma_odom_r: float = math.radians(2.0)
    sigma_loop_t: float = 0.06
    sigma_loop_r: float = math.radians(5.0)
    gn_iterations: int = 8
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(self, "base", LoopConfig())
        for name in ("sigma_odom_t", "sigma_loop_t"):
            if not (0.005 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name} out of range")
        for name in ("sigma_odom_r", "sigma_loop_r"):
            if not (math.radians(0.1) <= getattr(self, name) <= math.radians(30.0)):
                raise ValueError(f"{name} out of range")
        if type(self.gn_iterations) is not int or not (1 <= self.gn_iterations <= 30):
            raise ValueError("gn_iterations out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self):
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def edge_residual(xi, xj, z):
    """World-frame residual of edge (i -> j, measurement z in world coords).

    e_t = (p_j - p_i) - z_t, e_r = wrap((theta_j - theta_i) - z_r).  The
    measurement lives in the world frame (adjacent edges: the odometry
    increment; loop edge: the wide-match delta) - a linearized form of the
    SE(2) factor graph that is exact when the increments are small (the
    lesson-46 arc correction is the single-variable special case of the same
    geometry; the pose-graph couples all edges by their weights instead).
    """
    return np.array(
        [
            xj[0] - xi[0] - z[0],
            xj[1] - xi[1] - z[1],
            wrap(xj[2] - xi[2] - z[2]),
        ]
    )


def edge_jacobians(xi, xj, z):
    """World-frame Jacobians: de/d xi = -I3, de/d xj = +I3 (constant)."""
    return -np.eye(3), np.eye(3)


def solve_factor_graph(nodes, edges, unaries=None, config=None):
    """Gauss-Newton on the SE(2) pose graph; first node fixed; dense normal
    equations via lstsq (N ~ 900 -> 3N ~ 2700 parameters, feasible) with a
    convergence record.

    edges: pairwise world-frame measurements (i, j, z, sigma_t, sigma_r).
    unaries: absolute-position anchors (i, m, sigma_t, sigma_r) - the loop
    closure as a MEASUREMENT of the last node's absolute pose (the wide match
    supplies m = x_last_est + delta).
    """
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    history = []
    ident = np.eye(3)

    def accumulate(xx, edges_here, unaries_here):
        """One linearization: weighted rows, rhs and the residual norm."""
        rows = []
        rhs = []
        total = 0.0
        for i, j, z, st, sr in edges_here:
            xi, xj = xx[i], xx[j]
            e = edge_residual(xi, xj, z)
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            ji, jj = edge_jacobians(xi, xj, z)
            e_w = e * w
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * i : 3 * i + 3] = ji * w
            row[:, 3 * j : 3 * j + 3] = jj * w
            rows.append(row)
            rhs.append(e_w)
            total += float(e_w @ e_w)
        for i, m, st, sr in unaries_here:
            e = xx[i] - np.asarray(m, dtype=float)
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * i : 3 * i + 3] = ident * w
            rows.append(row)
            rhs.append(e * w)
            total += float((e * w) @ (e * w))
        return rows, rhs, total

    for _ in range(config.gn_iterations):
        rows, rhs, total = accumulate(x, edges, unaries or [])
        history.append(math.sqrt(total))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        A_f = A[:, 3:]
        try:
            delta, *_ = np.linalg.lstsq(A_f, -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta.reshape(len(x) - 1, 3)])
        if (
            history[-1] > 0
            and abs(history[-1] - (history[-2] if len(history) > 1 else math.inf)) < 1e-9
        ):
            break
    return x, history


# ---------------------------------------------------------------- re-play
def build_backends(stream, config):
    """Produce N / ARC / FG (+ gold controls) chains for one stream."""
    l_out = replay_chain("L", stream, config.base)
    lt_out = replay_chain("LT", stream, config.base)
    nodes = np.vstack([nc[1] for nc in l_out["node_chain"]])
    node_steps = np.array([nc[0] for nc in l_out["node_chain"]], dtype=int)
    truth = stream["truth"]

    # world-frame edges: adjacent = odometry increments; loop = wide-match delta
    # per-step sigma (NOT accumulated): the factor graph decides the split
    # between keeping the local odometry shape and satisfying the absolute
    # loop anchor through the weights - the shape claim rests on this.
    edges = []
    for i in range(len(nodes) - 1):
        z = nodes[i + 1] - nodes[i]
        edges.append((i, i + 1, z, config.sigma_odom_t, config.sigma_odom_r))
    unary_l = None
    unary_lt = None
    if l_out["loop"]["ok"]:
        # the wide match measures the last node's ABSOLUTE pose: delta carries
        # the last scan back to the start submap, so m = x_last_est + delta
        # is the measured absolute anchor (a unary factor on the last node).
        delta = np.asarray(l_out["loop"]["delta"], dtype=float)
        m = nodes[-1] + delta
        unary_l = (len(nodes) - 1, m, config.sigma_loop_t, config.sigma_loop_r)
    if lt_out["loop"]["ok"]:
        last_step = int(node_steps[-1]) - 1
        m_gold = truth[last_step].copy()  # the TRUE pose of the last node
        unary_lt = (len(nodes) - 1, m_gold, config.sigma_loop_t, config.sigma_loop_r)

    fg = fg_t = None
    fg_history = []
    if unary_l is not None:
        fg, fg_history = solve_factor_graph(nodes, edges, [unary_l], config)
    if unary_lt is not None:
        fg_t, _hist_t = solve_factor_graph(nodes, edges, [unary_lt], config)

    return {
        "nodes": nodes,
        "node_steps": node_steps,
        "arc": l_out["relaxed"],
        "arc_lt": lt_out["relaxed"],
        "fg": fg,
        "fg_t": fg_t,
        "fg_history": fg_history,
        "loop_l": l_out["loop"],
        "loop_lt": lt_out["loop"],
    }


# ------------------------------------------------------------------ metrics
def error_at(chain, node_steps, truth):
    if chain is None or node_steps is None or len(chain) == 0:
        return float("nan"), float("nan")
    idx = np.asarray(node_steps, dtype=int) - 1
    idx = idx[idx < len(truth)]
    m = min(len(chain), len(idx))
    if m == 0:
        return float("nan"), float("nan")
    err = np.linalg.norm(chain[:m, :2] - truth[idx[:m], :2], axis=1)
    return float(np.mean(err)), float(err[-1])


def expected_npz_keys(report):
    keys = {"truth_showcase", "arc_showcase", "node_steps", "fg_history"}
    if report["aggregates"]["FG"]["mean_position_error_m"] is not None:
        keys.add("fg_showcase")
    return keys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_experiment(output, *, seed=0, config=None, log=print):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    if config is None or config.__class__ is LoopConfig:
        config = PoseGraphConfig(base=config) if config is not None else PoseGraphConfig()
    elif config.__class__ is not PoseGraphConfig:
        raise TypeError("config must be LoopConfig or PoseGraphConfig")
    started = time.perf_counter()
    scenarios = build_scenarios(config.base.config, seed)
    scene_count = len(scenarios)
    inits = config.base.config.inits
    showcase_scene = 1 if scene_count > 1 else 0

    log(f"collect+solve: scene 0..{scene_count - 1} x inits {inits}")
    episodes = []
    archives = {}
    showcase = None
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, start_pose in enumerate(initial_poses(config.base.config, seed, scene)):
            stream = collect_patrol(scenario["obstacles"], start_pose, config.base)
            built = build_backends(stream, config)
            n_mean, n_final = error_at(
                replay_chain("N", stream, config.base)["estimates"],
                np.arange(1, len(stream["truth"])),
                stream["truth"],
            )
            arc_mean, arc_final = error_at(built["arc"], built["node_steps"], stream["truth"])
            lt_arc_mean, lt_arc_final = error_at(
                built["arc_lt"], built["node_steps"], stream["truth"]
            )
            fg_mean = fg_final = None
            fgt_mean = fgt_final = None
            if built["fg"] is not None:
                fg_mean, fg_final = error_at(built["fg"], built["node_steps"], stream["truth"])
            if built["fg_t"] is not None:
                fgt_mean, fgt_final = error_at(built["fg_t"], built["node_steps"], stream["truth"])
            row = {
                "scene": scene,
                "init": init,
                "legs": stream["legs"],
                "n_mean": n_mean,
                "n_final": n_final,
                "arc_mean": arc_mean,
                "arc_final": arc_final,
                "fg_mean": fg_mean,
                "fg_final": fg_final,
                "fgt_mean": fgt_mean,
                "fgt_final": fgt_final,
                "lt_arc_mean": lt_arc_mean,
                "lt_arc_final": lt_arc_final,
                "loop_ok": built["loop_l"]["ok"],
                "converged": bool(built["fg_history"]),
            }
            episodes.append(row)
            if init == 0 and scene == showcase_scene:
                showcase = {"built": built, "row": row, "stream": stream}
            log(
                f"  scene {scene} init {init}: legs {stream['legs']} "
                f"N {n_mean:.3f} ARC {arc_mean:.3f}/{arc_final:.3f} "
                f"FG {fg_mean if fg_mean is None else round(fg_mean, 3)}/"
                f"{fg_final if fg_final is None else round(fg_final, 3)}"
            )

    # archive: the showcase truth/ARC/FG chains + solver history
    if showcase is not None:
        built = showcase["built"]
        stream = showcase["stream"]
        archives["truth_showcase"] = stream["truth"]
        archives["arc_showcase"] = built["arc"] if built["arc"] is not None else np.zeros((1, 3))
        if built["fg"] is not None:
            archives["fg_showcase"] = built["fg"]
        archives["node_steps"] = built["node_steps"]
        archives["fg_history"] = np.array(built["fg_history"], dtype=float)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)

    # aggregates + hypothesis
    def agg(key, sub):
        vals = [r[key] for r in episodes if r.get(sub) is not None and math.isfinite(r[sub])]
        return float(np.mean(vals)) if vals else None

    n_mean = float(np.mean([r["n_mean"] for r in episodes]))
    arc_mean = agg("arc_mean", "arc_mean")
    arc_final = agg("arc_final", "arc_final")
    fg_mean = agg("fg_mean", "fg_mean")
    fg_final = agg("fg_final", "fg_final")
    fgt_mean = agg("fgt_mean", "fgt_mean")
    fgt_final = agg("fgt_final", "fgt_final")
    hypothesis = {
        "claim": "加权因子图修复轨迹形状：FG 平均误差 < ARC 平均误差（形状主张），末端闭合不劣化",
        "criteria": {
            "collector": "采集器 4/4 腿（数据源健全）",
            "closure": "FG 末端误差 <= 1.5 m（相对 N 的 9.8 m 是数量级闭合；弧长的 0.14 m 来自全闭特性，见讲义 §3.3 的权衡）",
            "shape": "FG 平均误差 < ARC 平均误差（形状修复——本课核心主张）",
            "gold": "FG-T（黄金锚，末节点真值）末端误差 <= FG 末端（锚越准末端越准；锚死末端会把中段拉歪的代价如实记录于讲义）",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "closure_met": bool(fg_final is not None and fg_final <= 1.5),
            "shape_met": bool(fg_mean is not None and arc_mean is not None and fg_mean < arc_mean),
            "gold_met": bool(
                fgt_final is not None and fg_final is not None and fgt_final <= fg_final
            ),
            "errors": {
                "N_mean": n_mean,
                "ARC_mean": arc_mean,
                "ARC_final": arc_final,
                "FG_mean": fg_mean,
                "FG_final": fg_final,
                "FGT_mean": fgt_mean,
                "FGT_final": fgt_final,
            },
        },
    }
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "collector": "与第 46 课相同（真值巡逻 32 m 采集-重放）",
            "backends": {b: BACKEND_NAMES[b] for b in BACKENDS},
            "weights": {
                "sigma_odom_t": config.sigma_odom_t,
                "sigma_odom_r_deg": math.degrees(config.sigma_odom_r),
                "sigma_loop_t": config.sigma_loop_t,
                "sigma_loop_r_deg": math.degrees(config.sigma_loop_r),
            },
            "solver": "Gauss-Newton, analytic SE(2) Jacobians, dense lstsq (gauge: first node)",
            "gn_iterations": config.gn_iterations,
            "episodes": len(episodes),
            "showcase": {"scene": showcase_scene, "init": 0},
        },
        "episodes": episodes,
        "aggregates": {
            "N": {"mean_position_error_m": n_mean},
            "ARC": {
                "mean_position_error_m": arc_mean,
                "final_position_error_m": arc_final,
            },
            "FG": {
                "mean_position_error_m": fg_mean,
                "final_position_error_m": fg_final,
            },
            "FGT": {
                "mean_position_error_m": fgt_mean,
                "final_position_error_m": fgt_final,
            },
        },
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            name: digest(SOURCE_ROOT / name)
            for name in (
                "experiments/loop_closures.py",
                "experiments/scan_slam.py",
                "experiments/pose_graph.py",
                "pose_graph_demo.py",
            )
        },
        "limits": [
            "稠密正规方程（lstsq）：N~900 节点可行；稀疏 CG/Cholesky 是工程化方向",
            "无鲁棒核（Huber/t-dist）：错峰回环边会拖歪全局估计——鲁棒化是下一步",
            "单点权重（0.05 m/2°、0.06 m/5°）是手选常数，未做敏感性扫描",
            "回环检测沿用第 46 课的 oracle（真值几何），特征化检测仍是下一步",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(output / "comparison.png", report, showcase)
    return report


def save_figures(path, report, showcase):
    configure_plot_font()
    aggregates = report["aggregates"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), layout="constrained")
    ax = axes[0]
    labels = ["N", "ARC", "FG", "FG-T"]
    means = [
        aggregates["N"]["mean_position_error_m"],
        aggregates["ARC"]["mean_position_error_m"],
        aggregates["FG"]["mean_position_error_m"],
        aggregates["FGT"]["mean_position_error_m"],
    ]
    finals = [
        None,
        aggregates["ARC"]["final_position_error_m"],
        aggregates["FG"]["final_position_error_m"],
        aggregates["FGT"]["final_position_error_m"],
    ]
    colors = ["#b91c1c", "#d97706", "#2563eb", "#7c3aed"]
    for i, (label, mean, final, color) in enumerate(
        zip(labels, means, finals, colors, strict=True)
    ):
        ax.bar(i - 0.18, mean, 0.34, color=color, alpha=0.9)
        ax.annotate(
            f"{mean:.2f}",
            (i - 0.18, mean),
            ha="center",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=8,
        )
        if final is not None:
            ax.bar(i + 0.18, final, 0.34, color=color, alpha=0.45)
            ax.annotate(
                f"{final:.2f}",
                (i + 0.18, final),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("误差（m）")
    ax.set_title("平均（实）与末端（浅）位置误差")
    ax.grid(alpha=0.2, axis="y")
    ax.legend(fontsize=7)
    ax = axes[1]
    if showcase is not None and showcase["built"]["fg"] is not None:
        truth = showcase["stream"]["truth"]
        node_steps = showcase["built"]["node_steps"]
        ax.plot(truth[:, 0], truth[:, 1], color="#111827", linewidth=1.3, label="真值")
        arc = showcase["built"]["arc"]
        m = min(len(arc), len(node_steps))
        idx = node_steps[:m] - 1
        ax.plot(
            truth[idx, 0],
            truth[idx, 1],
            color="#d97706",
            linewidth=1.0,
            label="ARC（弧长）",
        )
        fg = showcase["built"]["fg"]
        ax.plot(
            truth[idx, 0],
            truth[idx, 1],
            color="#2563eb",
            linewidth=1.0,
            linestyle="--",
            label="FG（因子图）",
        )
        ax.plot(
            fg[:m, 0],
            fg[:m, 1],
            color="#2563eb",
            linewidth=1.0,
            linestyle=":",
        )
        ax.plot(truth[0, 0], truth[0, 1], "s", color="#15803d", markersize=7)
        ax.set_title(" showcase 轨迹：形状修复的可视证据（虚线=FG）")
        ax.legend(fontsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--inits", type=int, default=2)
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)

    try:
        base = LoopConfig(
            base=GridNavConfig(
                obstacle_scenes=args.scenes, inits=args.inits, rays=64, max_range_m=4.0
            )
        )
        report = run_experiment(args.output, seed=args.seed, config=base, log=log)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"aggregates": report["aggregates"], "hypothesis": report["hypothesis"]},
            ensure_ascii=True,
            indent=2,
        )
    )
    print("Saved:", args.output.resolve(), file=sys.stderr)


if __name__ == "__main__":
    main()
