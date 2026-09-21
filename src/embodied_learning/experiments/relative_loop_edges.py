"""Lesson 52: relative loop edges - giving SC/MM their real working domain.

Lesson 51 recorded the boundary result: on the ABSOLUTE-anchor injection the
SC lambda feasible set is EMPTY (the poisoned anchor "closes" the same way
the good anchor does - the two closures are the same action) and the hard-
selection MM deadlocks from the open chain.  The literature works with
RELATIVE loop edges: the matcher measures z = (R, t) from node i to node j
and a WRONG z must fight the odometry chain's SHAPE (a wrong relative
measurement disagrees with the whole chain of odometry increments between
i and j, and the disagreement survives closure).

This lesson swaps the edge type and reruns the SC/MM comparison:
  z_good = the true relative pose (the stand-in for a correct wide match);
  z_bad  = the true relative pose rotated by +-90 deg + shifted (lesson-48
           injection, now as a RELATIVE measurement).

Groups (same collect-replay data, same solver family as lesson 51):
  N       dead reckoning;
  REL-LS-G / REL-LS-B   plain LS factor graph with the relative loop edge;
  SC-G / SC-B           the lesson-51 switchable solver, lambda = 50 (the
                        lesson-51 sweet-spot candidate), on relative edges;
  MM-G / MM-B           the hard-selection mixture on relative edges.

Pre-registered claims:
  (1) collector gate (4/4 legs);
  (2) fragility survives the edge type: REL-LS-B mean > N mean;
  (3) TREATMENT: SC-G final <= 1.5 m (closure with the relative edge);
  (4) PREVENTION: SC-B mean <= N + 0.5 m AND SC-B s* <= 0.3 (the switch
      rejects the wrong relative measurement - observable classification);
  (5) MM-G final <= 1.5 m (the mixture closes when the good component can
      win once the chain is shape-consistent).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from embodied_learning.experiments.loop_closures import (
    collect_patrol,
)
from embodied_learning.experiments.loop_closures import (
    replay_chain as replay_chain_l,
)
from embodied_learning.experiments.robust_graph import (
    build_groups,
    build_scenarios,
    error_at,
    initial_poses,
)

EXPERIMENT = "relative_loop_edges_lesson52"
SCHEMA_VERSION = 1
SC_LAMBDA = 50.0  # the lesson-51 sweet-spot candidate
MIX_DELTA = 0.5
MIX_RATIO = 100.0
GROUPS = ("N", "REL-LS-G", "REL-LS-B", "SC-G", "SC-B", "MM-G", "MM-B")
GROUP_NAMES = {
    "N": "纯里程计（无后端）",
    "REL-LS-G": "相对回环边 LS + 好边（新控制）",
    "REL-LS-B": "相对回环边 LS + 毒化边（脆弱性）",
    "SC-G": "SC λ=50 + 好边（治疗）",
    "SC-B": "SC λ=50 + 毒化边（预防）",
    "MM-G": "MM + 好边（治疗）",
    "MM-B": "MM + 毒化边（预防）",
}
DEFAULT_RESULTS = "results/relative_loops_2026-09-09"


@dataclass(frozen=True)
class RelativeConfig:
    """Lessons 46-51 protocol; defaults ARE the pre-registered run."""

    base: object = None  # PoseGraphConfig
    gn_iterations: int = 12
    seed: int = 0
    sc_lambda: float = SC_LAMBDA
    mix_delta: float = MIX_DELTA
    mix_ratio: float = MIX_RATIO

    def __post_init__(self):
        from embodied_learning.experiments.pose_graph import PoseGraphConfig

        if self.base is None:
            object.__setattr__(self, "base", PoseGraphConfig())
        if type(self.gn_iterations) is not int or not (1 <= self.gn_iterations <= 30):
            raise ValueError("gn_iterations out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not (0.05 <= self.sc_lambda <= 2000.0):
            raise ValueError("sc_lambda out of range")
        if not (0.01 <= self.mix_delta <= 10.0):
            raise ValueError("mix_delta out of range")
        if not (2.0 <= self.mix_ratio <= 1000.0):
            raise ValueError("mix_ratio out of range")

    @property
    def config(self):
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def edge_residual_rel(xi, xj, z):
    """World-frame relative residual (the lesson-47 factor form)."""
    return np.array([xj[0] - xi[0] - z[0], xj[1] - xi[1] - z[1], wrap(xj[2] - xi[2] - z[2])])


def build_rel_edges(nodes, config):
    """Adjacent odometry as relative edges (same as lesson 47)."""
    return [
        (
            i,
            i + 1,
            nodes[i + 1] - nodes[i],
            config.base.sigma_odom_t,
            config.base.sigma_odom_r,
        )
        for i in range(len(nodes) - 1)
    ]


def true_relative(nodes, node_steps, truth):
    """The TRUE relative pose of the loop pair (i=first, j=last node), in the
    rotate-then-translate measurement convention of the matcher."""
    i_step, j_step = int(node_steps[0]) - 1, int(node_steps[-1]) - 1
    ti, tj = truth[i_step], truth[j_step]
    dth = wrap(ti[2] - tj[2])
    c, s = math.cos(dth), math.sin(dth)
    return np.array([ti[0] - c * tj[0] + s * tj[1], ti[1] - s * tj[0] - c * tj[1], dth])


def poisoned_relative(z, deg, shift_m):
    """Lesson-48 injection as a relative measurement: rotate by deg + shift."""
    rot = math.radians(deg)
    c, s = math.cos(rot), math.sin(rot)
    rm = np.array([[c, -s], [s, c]])
    out = z.copy()
    out[:2] = rm @ z[:2] + np.array([shift_m, 0.0])
    out[2] = wrap(z[2] + rot)
    return out


def solve_rel_ls(nodes, edges, loop_edge, config):
    """Plain LS factor graph with an extra (relative) loop edge."""
    from embodied_learning.experiments.pose_graph import solve_factor_graph

    all_edges = edges + [loop_edge]
    x, _hist = solve_factor_graph(nodes, all_edges, None, config)
    return x


def solve_rel_sc(nodes, edges, loop_edge, config):
    """Switchable constraint on a RELATIVE loop edge (joint G-N over poses+s).

    The loop-edge Jacobians are the lesson-47 constant -I/+I; the s column
    is d(s*w*e)/ds = w*e evaluated at the current linearization.
    """
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    s = 1.0
    ident = np.eye(3)
    i, j, z, st, sr = loop_edge
    w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
    s_hist = []
    for _it in range(config.gn_iterations):
        rows = []
        rhs = []
        total = 0.0
        for ai, aj, az, ast, asr in edges:
            e = edge_residual_rel(x[ai], x[aj], az)
            aw = np.array([1.0 / ast, 1.0 / ast, 1.0 / asr])
            row = np.zeros((3, n_nodes * 3 + 1))
            row[:, 3 * ai : 3 * ai + 3] = -ident * aw
            row[:, 3 * aj : 3 * aj + 3] = ident * aw
            rows.append(row)
            rhs.append(e * aw)
            total += float(np.sum((e * aw) ** 2))
        e_loop = edge_residual_rel(x[i], x[j], z)
        row_p = np.zeros((3, n_nodes * 3 + 1))
        row_p[:, 3 * i : 3 * i + 3] = -ident * (s * w)
        row_p[:, 3 * j : 3 * j + 3] = ident * (s * w)
        row_p[:, -1] = e_loop * w
        rows.append(row_p)
        rhs.append(s * e_loop * w)
        total += float(np.sum((s * e_loop * w) ** 2))
        row_s = np.zeros((1, n_nodes * 3 + 1))
        row_s[:, -1] = math.sqrt(config.sc_lambda)
        rows.append(row_s)
        rhs.append([math.sqrt(config.sc_lambda) * (s - 1.0)])
        total += float(config.sc_lambda * (s - 1.0) ** 2)
        s_hist.append(float(s))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta[: (n_nodes - 1) * 3].reshape(n_nodes - 1, 3)])
        s = float(np.clip(s + delta[n_nodes * 3 - 3 :][0], 0.0, 2.0))
    return x, float(np.clip(s, 0.0, 2.0)), s_hist


def solve_rel_mm(nodes, edges, loop_edge, config, anneal=True):
    """Hard-selection max-mixture on a RELATIVE loop edge.

    anneal: the mixture prior starts LARGE (valid always selected, the G-N
    is pulled to close) and decays to mix_delta - the graduated non-convexity
    of lesson 48, here FORCING the first iterations to try the valid
    component so the selection sees a closed-chain residual, not the open
    one (the lesson-51 deadlock came from selection on the open chain)."""
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    ident = np.eye(3)
    i, j, z, st, sr = loop_edge
    w_valid = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
    w_invalid = w_valid / config.mix_ratio
    valid_frac = 0.0
    n_it = config.gn_iterations
    for it in range(n_it):
        rows = []
        rhs = []
        total = 0.0
        for ai, aj, az, ast, asr in edges:
            e = edge_residual_rel(x[ai], x[aj], az)
            aw = np.array([1.0 / ast, 1.0 / ast, 1.0 / asr])
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * ai : 3 * ai + 3] = -ident * aw
            row[:, 3 * aj : 3 * aj + 3] = ident * aw
            rows.append(row)
            rhs.append(e * aw)
            total += float(np.sum((e * aw) ** 2))
        e_loop = edge_residual_rel(x[i], x[j], z)
        delta_now = (config.mix_delta * 100.0) * (0.5**it) if anneal else config.mix_delta
        cost_valid = float(np.sum((e_loop * w_valid) ** 2))
        cost_invalid = float(np.sum((e_loop * w_invalid) ** 2)) + delta_now
        use_valid = cost_valid <= cost_invalid
        valid_frac += 1.0 if use_valid else 0.0
        w_used = w_valid if use_valid else w_invalid
        row = np.zeros((3, n_nodes * 3))
        row[:, 3 * i : 3 * i + 3] = -ident * w_used
        row[:, 3 * j : 3 * j + 3] = ident * w_used
        rows.append(row)
        rhs.append(e_loop * w_used)
        total += float(np.sum((e_loop * w_used) ** 2))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta.reshape(len(x) - 1, 3)])
    return x, valid_frac / n_it


def expected_npz_keys(report):
    return set(report["archive_keys"])


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
    if config is None:
        config = RelativeConfig()
    elif config.__class__ is not RelativeConfig:
        raise TypeError("config must be RelativeConfig")
    started = time.perf_counter()
    scenarios = build_scenarios(config.base.base.config, seed)
    episodes = []
    archives = {}
    chains = {}
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, start_pose in enumerate(initial_poses(config.base.base.config, seed, scene)):
            stream = collect_patrol(scenario["obstacles"], start_pose, config.base.base)
            shim = SimpleNamespace(base=config.base, poison_rot_deg=90.0)
            built = build_groups(stream, shim)
            nodes = built["nodes"]
            node_steps = built["node_steps"]
            edges = build_rel_edges(nodes, config)
            n_est = replay_chain_l("N", stream, config.base.base)["estimates"]
            n_mean, _ = error_at(n_est, np.arange(1, len(stream["truth"])), stream["truth"])
            row = {"scene": scene, "init": init, "legs": stream["legs"], "n_mean": n_mean}
            z_true = true_relative(nodes, node_steps, stream["truth"])
            z_good = (0, len(nodes) - 1, z_true, config.base.sigma_loop_t, config.base.sigma_loop_r)
            z_bad = (
                0,
                len(nodes) - 1,
                poisoned_relative(z_true, 90.0, 1.0),
                config.base.sigma_loop_t,
                config.base.sigma_loop_r,
            )
            for name, loop_edge, kind in (
                ("REL-LS-G", z_good, "ls"),
                ("REL-LS-B", z_bad, "ls"),
                ("SC-G", z_good, "sc"),
                ("SC-B", z_bad, "sc"),
                ("MM-G", z_good, "mm"),
                ("MM-B", z_bad, "mm"),
            ):
                if kind == "sc":
                    x, s, s_hist = solve_rel_sc(nodes, edges, loop_edge, config)
                    row[f"{name}_s"] = s
                elif kind == "mm":
                    x, frac = solve_rel_mm(nodes, edges, loop_edge, config)
                    row[f"{name}_s"] = float(frac)
                else:
                    x = solve_rel_ls(nodes, edges, loop_edge, config)
                    row[f"{name}_s"] = None
                mean, final = error_at(x, node_steps, stream["truth"])
                row[f"{name}_mean"] = mean
                row[f"{name}_final"] = final
                if scene == 1 and init == 0:
                    chains[name] = x
            episodes.append(row)
            log(
                f"  scene {scene} init {init}: legs {stream['legs']} N {n_mean:.3f}  ".join(
                    f"{name} {row.get(f'{name}_mean') if row.get(f'{name}_mean') is None else round(row[f'{name}_mean'], 3)}"
                    for name in GROUPS[1:]
                )
            )
    archives["truth_showcase"] = np.zeros((1, 3))
    archives["node_steps"] = np.zeros(1, dtype=int)
    for name in GROUPS[1:]:
        archives[f"chain_{name}"] = chains.get(name, np.zeros((1, 3)))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)

    def agg(name, field):
        vals = [r[f"{name}_{field}"] for r in episodes if r.get(f"{name}_{field}") is not None]
        return float(np.mean(vals)) if vals else None

    aggregates = {
        name: {
            "mean_position_error_m": agg(name, "mean"),
            "final_position_error_m": agg(name, "final"),
            "switch_final": agg(name, "s"),
        }
        for name in GROUPS
    }
    if aggregates["N"]["mean_position_error_m"] is None:
        aggregates["N"]["mean_position_error_m"] = float(np.mean([r["n_mean"] for r in episodes]))
    n_mean = aggregates["N"]["mean_position_error_m"]
    lsb = aggregates["REL-LS-B"]["mean_position_error_m"]
    scg_final = aggregates["SC-G"]["final_position_error_m"]
    scb_mean = aggregates["SC-B"]["mean_position_error_m"]
    scb_s = aggregates["SC-B"]["switch_final"]
    mmg_final = aggregates["MM-G"]["final_position_error_m"]
    hypothesis = {
        "claim": (
            "相对回环边（匹配器 z 测量 i↔j）给 SC/MM 真实工作域：错误 z 必须对抗"
            "里程计链的形状——SC 开关把毒化相对边压到 s≈0（预防）且好边照常闭合"
            "（治疗）；毒化相对边在 LS 下仍然比无回环更糟（脆弱性跨边型存在）"
        ),
        "criteria": {
            "collector": "采集器 4/4 腿",
            "fragility": "REL-LS-B 均值 > N 均值（脆弱性跨边型存在）",
            "treatment": "SC-G 末端 <= 1.5 m（好相对边照常闭合）",
            "prevention": "SC-B 均值 <= N + 0.5 m 且 SC-B 开关 <= 0.3（毒化被开关拒绝，可观测）",
            "mm": "MM-G 末端 <= 1.5 m（好组件在形状一致后获胜）",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "fragility_met": bool(lsb is not None and lsb > n_mean),
            "treatment_met": bool(scg_final is not None and scg_final <= 1.5),
            "prevention_met": bool(
                scb_mean is not None
                and scb_mean <= n_mean + 0.5
                and scb_s is not None
                and scb_s <= 0.3
            ),
            "mm_met": bool(mmg_final is not None and mmg_final <= 1.5),
            "errors": {name: aggregates[name]["mean_position_error_m"] for name in GROUPS},
            "finals": {name: aggregates[name]["final_position_error_m"] for name in GROUPS},
            "switches": {
                name: aggregates[name]["switch_final"] for name in ("SC-G", "SC-B", "MM-G", "MM-B")
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
            "collector": "与第 46-51 课相同（真值巡逻采集-重放）",
            "groups": {g: GROUP_NAMES[g] for g in GROUPS},
            "loop_edge": {
                "kind": "relative (i=first node, j=last node), matcher-convention z",
                "poison": "true z rotated +90 deg + 1 m shift (the lesson-48 injection, relative form)",
                "sigma": f"{config.base.sigma_loop_t} m / {math.degrees(config.base.sigma_loop_r):.0f} deg",
            },
            "sc": {"lambda": config.sc_lambda, "s0": 1.0, "bounds": [0.0, 2.0]},
            "mixture": {
                "delta": config.mix_delta,
                "ratio": config.mix_ratio,
                "selection": "hard per-iteration cheaper component",
            },
        },
        "aggregates": aggregates,
        "episodes": [
            {
                "scene": r["scene"],
                "init": r["init"],
                "legs": r["legs"],
                **{
                    f"{name}_{field}": r.get(f"{name}_{field}")
                    for name in GROUPS[1:]
                    for field in ("mean", "final", "s")
                },
            }
            for r in episodes
        ],
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "archive_keys": sorted(archives.keys()),
        "trajectories_sha256": digest(output / "trajectories.npz"),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="lesson-52 relative loop edges record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
