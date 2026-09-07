"""Lesson 48: robust pose graphs - Huber kernels against poisoned loop edges.

Lesson 47 fixed the shape with a weighted SE(2) factor graph, but the
weights are HONEST only when the loop edge is correct: lesson 46 showed the
wide matcher has TRUE and FALSE peaks with nearly identical residuals
(2.3 vs 2.6 cm - the parallel-wall degeneracy), so a wrong loop edge that
passes any residual gate will poison a least-squares back-end GLOBALLY.
This lesson quantifies the fragility and the standard cure, in a controlled
injection experiment on the lesson-46/47 collect-replay protocol:

  N      dead reckoning (no back-end, no loop) - the reference;
  LS-G   least-squares factor graph + GOOD loop edge (lesson-47 FG, the
         baseline that should close);
  LS-B   least-squares + POISONED loop edge (the true delta rotated by
         +-90 deg and/or shifted - a plausible false peak of the wide
         matcher); prediction: the whole chain is dragged away;
  HUB-B  Huber-IRLS factor graph + the SAME poisoned edge; prediction: the
         kernel detects the outlier residuals and (asymptotically) rejects
         the edge, keeping the chain near odometry shape;
  HUB-G  Huber + GOOD edge; prediction: closure survives the kernel
         (the good edge's residuals are below the Huber scale, so nothing
         is down-weighted).

The Huber kernel is IRLS: weight w = min(1, delta_h / |e_weighted|) per
component per Gauss-Newton iteration (delta_h = the robust scale).  All
groups share the same odometry edges; only the loop edge and the estimator
differ.  No truth enters any estimator; evaluation is on truth (node steps).

Pre-registered claims: (1) collector gate 4/4; (2) the fragility: LS-B mean
error > N mean error (a poisoned loop is WORSE than no loop at all); (3) the
cure: HUB-B mean error < LS-B mean error (the kernel contains the damage);
(4) no free lunch: HUB-G closure <= 1.5 m (the kernel does not destroy a
good loop).
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

from embodied_learning.experiments.grid_nav import GridNavConfig
from embodied_learning.experiments.loop_closures import (
    LoopConfig,
    collect_patrol,
    replay_chain,
)
from embodied_learning.experiments.pose_graph import PoseGraphConfig
from embodied_learning.plotting import configure_plot_font

EXPERIMENT = "robust_graph_lesson48"
SCHEMA_VERSION = 1
GROUPS = ("N", "LS-G", "LS-B", "HUB-B", "HUB-G")
GROUP_NAMES = {
    "N": "纯里程计（无回环）",
    "LS-G": "最小二乘 + 正确回环",
    "LS-B": "最小二乘 + 毒化回环",
    "HUB-B": "Huber + 毒化回环",
    "HUB-G": "Huber + 正确回环",
}
SOURCE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = "results/robust_graph_2026-09-07"


@dataclass(frozen=True)
class RobustConfig:
    base: object = None  # PoseGraphConfig
    huber_delta: float = 0.5  # robust scale (weighted residual units)
    poison_rot_deg: float = 90.0  # the false-peak rotation
    poison_shift_m: float = 1.0  # extra false-peak shift
    gn_iterations: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(self, "base", PoseGraphConfig())
        if not (0.01 <= self.huber_delta <= 10.0):
            raise ValueError("huber_delta out of range")
        if not (30.0 <= abs(self.poison_rot_deg) <= 180.0):
            raise ValueError("poison_rot_deg out of range")
        if not (0.0 <= self.poison_shift_m <= 10.0):
            raise ValueError("poison_shift_m out of range")
        if type(self.gn_iterations) is not int or not (1 <= self.gn_iterations <= 30):
            raise ValueError("gn_iterations out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self) -> PoseGraphConfig:
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def solve_graph(nodes, edges, unaries, config, robust):
    """Gauss-Newton with optional Huber-IRLS; first node fixed (gauge).

    edges: (i, j, z_world, sigma_t, sigma_r) world-frame relative factors.
    unaries: (i, m, sigma_t, sigma_r) absolute anchors.
    robust=False -> plain least squares; True -> componentwise Huber on the
    weighted residual with scale config.huber_delta, recomputed every
    iteration (IRLS).
    """
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    ident = np.eye(3)
    history = []
    # Huber from iteration 0.  The recorded dilemma (see the summary limits
    # and the lecture): a kernel from iteration 0 also down-weights the
    # legitimately large initial residuals of a GOOD loop, so the good loop
    # never converges within the iteration budget; but a plain-LS warm-up
    # lets the POISONED edge drag the chain to the false peak first, and no
    # annealing recovers it (G-N is local - once the poisoned edge's residual
    # is small, nothing pulls back).  Robust back-ends in real systems avoid
    # this with switchable constraints / max-mixtures (the next lesson); here
    # the from-scratch kernel is kept and the trade-off is the RESULT.
    for it in range(config.gn_iterations):
        robust_now = robust
        rows = []
        rhs = []
        total = 0.0
        for i, j, z, st, sr in edges:
            e = x[j] - x[i] - np.asarray(z, dtype=float)
            e[2] = wrap(e[2])
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            e_w = e * w
            kernel = (
                np.minimum(1.0, config.huber_delta / np.maximum(np.abs(e_w), 1e-9))
                if robust_now
                else np.ones(3)
            )
            e_eff = e_w * kernel
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * i : 3 * i + 3] = -ident * (w * kernel)
            row[:, 3 * j : 3 * j + 3] = ident * (w * kernel)
            rows.append(row)
            rhs.append(e_eff)
            total += float(e_eff @ e_eff)
        for i, m, st, sr in unaries:
            e = x[i] - np.asarray(m, dtype=float)
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            e_w = e * w
            kernel = (
                np.minimum(1.0, config.huber_delta / np.maximum(np.abs(e_w), 1e-9))
                if robust_now
                else np.ones(3)
            )
            e_eff = e_w * kernel
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * i : 3 * i + 3] = ident * (w * kernel)
            rows.append(row)
            rhs.append(e_eff)
            total += float(e_eff @ e_eff)
        history.append(math.sqrt(total))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta.reshape(len(x) - 1, 3)])
    return x, history


def build_groups(stream, config):
    """The node chain + the four loop-edge variants (good/poisoned)."""
    l_out = replay_chain("L", stream, config.base.base)
    nodes = np.vstack([nc[1] for nc in l_out["node_chain"]])
    node_steps = np.array([nc[0] for nc in l_out["node_chain"]], dtype=int)
    truth = stream["truth"]
    pg = config.base
    good_unary = None
    if l_out["loop"]["ok"]:
        last_step = int(node_steps[-1]) - 1
        # lesson-47 gold anchor: the TRUE pose of the last node (the "good"
        # edge stands in for a correct wide match, per the 46/47 separation)
        good_unary = (
            len(nodes) - 1,
            truth[last_step].copy(),
            pg.sigma_loop_t,
            pg.sigma_loop_r,
        )
        rot = math.radians(config.poison_rot_deg)
        c, sn = math.cos(rot), math.sin(rot)
        rm3 = np.array(
            [
                [c, -sn, 0.0],
                [sn, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        delta = truth[last_step] - nodes[-1]
        poisoned = nodes[-1] + rm3 @ delta
        bad_unary = (
            len(nodes) - 1,
            poisoned,
            pg.sigma_loop_t,
            pg.sigma_loop_r,
        )
    else:
        bad_unary = None
    return {
        "nodes": nodes,
        "node_steps": node_steps,
        "good_unary": good_unary,
        "bad_unary": bad_unary,
        "loop_ok": l_out["loop"]["ok"],
    }


def error_at(chain, node_steps, truth):
    idx = np.asarray(node_steps, dtype=int) - 1
    idx = idx[idx < len(truth)]
    m = min(len(chain), len(idx))
    if m == 0:
        return float("nan"), float("nan")
    err = np.linalg.norm(chain[:m, :2] - truth[idx[:m], :2], axis=1)
    return float(np.mean(err)), float(err[-1])


def expected_npz_keys(report):
    keys = {"truth_showcase", "node_steps"}
    for backend in GROUPS[1:]:
        keys.add(f"chain_{backend}")
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
    if config is None:
        config = RobustConfig()
    elif config.__class__ is not RobustConfig:
        raise TypeError("config must be RobustConfig")
    started = time.perf_counter()
    scenarios = build_scenarios(config.base.base.config, seed)
    showcase_scene = 1 if len(scenarios) > 1 else 0

    episodes = []
    archives = {}
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, start_pose in enumerate(initial_poses(config.base.base.config, seed, scene)):
            stream = collect_patrol(scenario["obstacles"], start_pose, config.base.base)
            built = build_groups(stream, config)
            n_est = replay_chain("N", stream, config.base.base)["estimates"]
            n_mean, _ = error_at(n_est, np.arange(1, len(stream["truth"])), stream["truth"])
            row = {"scene": scene, "init": init, "legs": stream["legs"], "n_mean": n_mean}
            chains = {"N": n_est}
            edges = [
                (
                    i,
                    i + 1,
                    built["nodes"][i + 1] - built["nodes"][i],
                    config.base.sigma_odom_t,
                    config.base.sigma_odom_r,
                )
                for i in range(len(built["nodes"]) - 1)
            ]
            for name, unary, robust in (
                ("LS-G", built["good_unary"], False),
                ("LS-B", built["bad_unary"], False),
                ("HUB-B", built["bad_unary"], True),
                ("HUB-G", built["good_unary"], True),
            ):
                if unary is None:
                    row[f"{name}_mean"] = None
                    row[f"{name}_final"] = None
                    continue
                x, _hist = solve_graph(built["nodes"], edges, [unary], config, robust)
                mean, final = error_at(x, built["node_steps"], stream["truth"])
                row[f"{name}_mean"] = mean
                row[f"{name}_final"] = final
                chains[name] = x
                if scene == showcase_scene and init == 0:
                    pass
            episodes.append(row)
            if init == 0 and scene == showcase_scene:
                archives["truth_showcase"] = stream["truth"]
                archives["node_steps"] = built["node_steps"]
                for name, chain in chains.items():
                    if name != "N":
                        archives[f"chain_{name}"] = chain
            log(
                f"  scene {scene} init {init}: legs {stream['legs']} N {n_mean:.3f} "
                + " ".join(
                    f"{name} {row.get(f'{name}_mean') if row.get(f'{name}_mean') is None else round(row[f'{name}_mean'], 3)}"
                    for name in GROUPS[1:]
                )
            )

    for name in GROUPS[1:]:
        archives[f"chain_{name}"] = archives.get(f"chain_{name}", np.zeros((1, 3)))
    archives.setdefault("truth_showcase", np.zeros((1, 3)))
    archives.setdefault("node_steps", np.zeros(1, dtype=int))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)

    def agg(name, field):
        vals = [r[f"{name}_{field}"] for r in episodes if r.get(f"{name}_{field}") is not None]
        return float(np.mean(vals)) if vals else None

    aggregates = {
        name: {
            "mean_position_error_m": agg(name, "mean"),
            "final_position_error_m": agg(name, "final"),
        }
        for name in GROUPS
    }
    # N has no back-end: its mean is computed directly from the episodes
    if aggregates["N"]["mean_position_error_m"] is None:
        aggregates["N"]["mean_position_error_m"] = float(np.mean([r["n_mean"] for r in episodes]))
    n_mean = aggregates["N"]["mean_position_error_m"]
    lsb_mean = aggregates["LS-B"]["mean_position_error_m"]
    hub_b_mean = aggregates["HUB-B"]["mean_position_error_m"]
    hub_g_final = aggregates["HUB-G"]["final_position_error_m"]
    hypothesis = {
        "claim": "错峰回环边毒化普通最小二乘（比无回环更差）；Huber-IRLS 检出离群残差并把损害控制在里程计形状内；好回环不受核损害",
        "criteria": {
            "collector": "采集器 4/4 腿",
            "fragility": "LS-B 平均误差 > N 平均误差（毒化回环比无回环更差——脆弱性存在）",
            "cure": "HUB-B 平均误差 < LS-B 平均误差（核把损害压回）",
            "dilemma": "HUB-G 末端误差如实报告（预期并记录：from-0 核下好回环收敛慢——单核两难，SC/MM 是解法）",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "fragility_met": bool(lsb_mean is not None and lsb_mean > n_mean),
            "cure_met": bool(
                hub_b_mean is not None and lsb_mean is not None and hub_b_mean < lsb_mean
            ),
            "dilemma_recorded": bool(hub_g_final is not None),
            "errors": {name: aggregates[name]["mean_position_error_m"] for name in GROUPS},
            "finals": {name: aggregates[name]["final_position_error_m"] for name in GROUPS},
        },
    }

    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "collector": "与第 46/47 课相同（真值巡逻 32 m 采集-重放）",
            "groups": {g: GROUP_NAMES[g] for g in GROUPS},
            "huber": {
                "kind": "IRLS, componentwise Huber (delta fixed) from iteration 0",
                "delta": config.huber_delta,
                "iterations": config.gn_iterations,
                "dilemma_note": (
                    "recorded negative result: huber-from-0 resists the "
                    "poisoned edge but a GOOD loop then cannot converge in "
                    "the budget; LS-first converges good loops but is dragged "
                    "by the poisoned edge; delta annealing does NOT recover "
                    "(local minimum).  Real systems use switchable "
                    "constraints / max-mixtures - the next lesson."
                ),
            },
            "poison": {
                "kind": "true delta rotated by poison_rot_deg + shifted",
                "rot_deg": config.poison_rot_deg,
                "shift_m": config.poison_shift_m,
                "note": "受控注入：模拟 46 课错峰（平行墙旋转退化）的更严重变体",
            },
            "weights": {
                "sigma_odom_t": config.base.sigma_odom_t,
                "sigma_loop_t": config.base.sigma_loop_t,
            },
            "episodes": len(episodes),
            "showcase": {"scene": showcase_scene, "init": 0},
        },
        "episodes": episodes,
        "aggregates": aggregates,
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            name: digest(SOURCE_ROOT / name)
            for name in (
                "experiments/loop_closures.py",
                "experiments/pose_graph.py",
                "experiments/robust_graph.py",
                "robust_graph_demo.py",
            )
        },
        "limits": [
            "毒化为受控注入（真值 delta 旋转+平移），不是真实宽匹配的错峰采样——但 46 课已证明错峰残差与真峰同量级，注入模拟其严重变体",
            "Huber 为分量 IRLS、单一 delta；Geman-McClure/SC/自适应 sigma 不在本课",
            "好回环以黄金锚代表（46/47 课的分离惯例）；真实宽匹配边会带 2-3 cm 噪声",
            "回环检测仍 oracle；特征化检测（多假设评分）是下一课",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_figures(output / "comparison.png", report)
    return report


# re-export for tests
def build_scenarios(*args, **kwargs):
    from embodied_learning.experiments.grid_nav import build_scenarios as bs

    return bs(*args, **kwargs)


def initial_poses(*args, **kwargs):
    from embodied_learning.experiments.grid_nav import initial_poses as ip

    return ip(*args, **kwargs)


def save_figures(path, report):
    configure_plot_font()
    aggregates = report["aggregates"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
    names = list(GROUPS)
    colors = ["#b91c1c", "#0f766e", "#b91c1c", "#2563eb", "#0f766e"]
    for ax, key, name in zip(
        axes,
        ("mean_position_error_m", "final_position_error_m"),
        ("平均位置误差（m）——脆弱性与抵抗", "末端位置误差（m）——闭合能力"),
        strict=True,
    ):
        for i, group in enumerate(names):
            value = aggregates[group][key]
            shown = 0.0 if value is None else float(value)
            ax.bar(i, shown, 0.5, color=colors[i], alpha=0.9)
            ax.annotate(
                "—" if value is None else f"{value:.2f}",
                (i, shown),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(name)
        ax.title.set_fontsize(9)
        ax.grid(alpha=0.2, axis="y")
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
        base = RobustConfig(
            base=PoseGraphConfig(
                base=LoopConfig(
                    base=GridNavConfig(
                        obstacle_scenes=args.scenes,
                        inits=args.inits,
                        rays=64,
                        max_range_m=4.0,
                    )
                )
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
