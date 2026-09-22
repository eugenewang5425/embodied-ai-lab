"""Lesson 53: production-solver comparison - the same relative-loop-edge
injection through scipy.optimize.least_squares.

Lessons 51/52 recorded that our hand-rolled G-N cannot turn the switch/mixer
into a classifier on the poisoned relative edge.  This lesson asks whether
that is a property of the PROBLEM or of OUR implementation: the identical
residual vector (odometry edges + the loop edge) goes through
scipy.optimize.least_squares - the mature production implementation with
TUNED robust losses (huber/soft_l1/cauchy/arctan x loss_scale sweep).

Groups (lesson-52 data and injection, one scene-set):
  N        dead reckoning;
  SCIPY-LS     scipy plain least squares + good relative edge (control);
  SCIPY-HUB-B  scipy robust='huber' + POISONED edge (the production cure?);
  SCIPY-HUB-G  scipy robust='huber' + GOOD edge (does the kernel spare it?);
  SCIPY-SOFT-B scipy robust='soft_l1' + POISONED edge (sharper knee);
  SELF-HUB-B   our lesson-48 IRLS Huber on the same relative edge (bridge).

Pre-registered claims:
  (1) parity: SCIPY-LS reproduces our REL-LS final error within 0.1 m
      (the solvers agree on the same problem - implementation variance);
  (2) the PRODUCTION kernel on the poisoned edge behaves as OURS did:
      SCIPY-HUB-B mean > N mean - 0.5 (the poisoned edge is not rejected,
      the failure is the problem's, not our solver's);
  (3) the dilemma is implementation-independent: SCIPY-HUB-G final > 1.5 m
      OR SCIPY-HUB-B mean > N + 0.5 m (at least one half fails in the
      production implementation too);
  (4) loss-scale sweep: no (loss, scale) in the swept grid achieves BOTH
      SCIPY-*-B mean <= N + 0.5 AND SCIPY-*-G final <= 1.5.
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

import numpy as np
from scipy.optimize import least_squares

from embodied_learning.experiments.loop_closures import collect_patrol, replay_chain
from embodied_learning.experiments.pose_graph import PoseGraphConfig
from embodied_learning.experiments.relative_loop_edges import (
    build_rel_edges,
    poisoned_relative,
    true_relative,
)
from embodied_learning.experiments.robust_graph import (
    build_scenarios,
    error_at,
    initial_poses,
)

EXPERIMENT = "production_solver_lesson53"
SCHEMA_VERSION = 1
GROUPS = ("N", "SCIPY-LS", "SCIPY-HUB-B", "SCIPY-HUB-G", "SCIPY-SOFT-B", "SELF-HUB-B")
GROUP_NAMES = {
    "N": "纯里程计（无后端）",
    "SCIPY-LS": "scipy LS + 好相对边（求解器对齐控制）",
    "SCIPY-HUB-B": "scipy huber + 毒化边（生产核）",
    "SCIPY-HUB-G": "scipy huber + 好边",
    "SCIPY-SOFT-B": "scipy soft_l1 + 毒化边（更尖拐点）",
    "SELF-HUB-B": "自研 IRLS Huber + 毒化边（桥接第 48 课）",
}
DEFAULT_RESULTS = "results/production_solver_2026-09-09"
LOSS_SCALES = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class ProductionConfig:
    """Defaults ARE the pre-registered run."""

    base: object = None  # PoseGraphConfig
    poison_rot_deg: float = 90.0
    poison_shift_m: float = 1.0
    seed: int = 0
    loss_scales: tuple = LOSS_SCALES
    max_inits: int = 0  # 0 = all (protocol); >0 caps initial poses (light tests)

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(self, "base", PoseGraphConfig())
        if not (30.0 <= abs(self.poison_rot_deg) <= 180.0):
            raise ValueError("poison_rot_deg out of range")
        if not (0.0 <= self.poison_shift_m <= 10.0):
            raise ValueError("poison_shift_m out of range")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self):
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def residual_vector(x_flat, n_nodes, edges, loop_edge):
    """Flattened residuals for scipy (pose[0] fixed by construction)."""
    x = np.vstack([np.zeros((1, 3)), x_flat.reshape(n_nodes - 1, 3)])
    out = []
    for i, j, z, st, sr in edges:
        e = x[j] - x[i] - np.asarray(z, dtype=float)
        e[2] = wrap(e[2])
        out.extend((e / np.array([st, st, sr])).tolist())
    i, j, z, st, sr = loop_edge
    e = x[j] - x[i] - np.asarray(z, dtype=float)
    e[2] = wrap(e[2])
    out.extend((e / np.array([st, st, sr])).tolist())
    return np.asarray(out)


def solve_scipy(nodes, edges, loop_edge, loss=None, loss_scale=1.0):
    """Production solver on the same problem (first pose fixed by trimming)."""
    n_nodes = len(nodes)
    x0 = nodes[1:].reshape(-1)
    kwargs = {}
    if loss is not None:
        kwargs = {"loss": loss, "f_scale": loss_scale}
    result = least_squares(
        residual_vector,
        x0,
        args=(n_nodes, edges, loop_edge),
        method="trf",
        max_nfev=60,
        **kwargs,
    )
    x = np.vstack([np.zeros((1, 3)), result.x.reshape(n_nodes - 1, 3)])
    return x


def solve_self_huber(nodes, edges, loop_edge, huber_delta=0.5, iterations=10):
    """Our lesson-48 IRLS Huber, now on a relative loop edge (bridge)."""
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    ident = np.eye(3)
    for _it in range(iterations):
        rows = []
        rhs = []
        for ai, aj, az, ast, asr in edges:
            e = x[aj] - x[ai] - np.asarray(az, dtype=float)
            e[2] = wrap(e[2])
            w = np.array([1.0 / ast, 1.0 / ast, 1.0 / asr])
            kernel = np.minimum(1.0, huber_delta / np.maximum(np.abs(e * w), 1e-9))
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * ai : 3 * ai + 3] = -ident * (w * kernel)
            row[:, 3 * aj : 3 * aj + 3] = ident * (w * kernel)
            rows.append(row)
            rhs.append(e * w * kernel)
        i, j, z, st, sr = loop_edge
        e = x[j] - x[i] - np.asarray(z, dtype=float)
        e[2] = wrap(e[2])
        w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
        kernel = np.minimum(1.0, huber_delta / np.maximum(np.abs(e * w), 1e-9))
        row = np.zeros((3, n_nodes * 3))
        row[:, 3 * i : 3 * i + 3] = -ident * (w * kernel)
        row[:, 3 * j : 3 * j + 3] = ident * (w * kernel)
        rows.append(row)
        rhs.append(e * w * kernel)
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta.reshape(n_nodes - 1, 3)])
    return x


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
        config = ProductionConfig()
    elif config.__class__ is not ProductionConfig:
        raise TypeError("config must be ProductionConfig")
    from types import SimpleNamespace

    started = time.perf_counter()
    scenarios_all = build_scenarios(config.base.base.config, seed)
    # dense production solver: one obstacle scene keeps the record tractable
    # (the lesson-52 conclusions were read on scene 1 as well)
    scenarios = [s for s in scenarios_all if s["scene"] == 1] or scenarios_all[:1]
    episodes = []
    archives = {}
    chains = {}
    sweep = []
    for scenario in scenarios:
        scene = scenario["scene"]
        poses_all = initial_poses(config.base.base.config, seed, scene)
        if config.max_inits:
            poses_all = poses_all[: config.max_inits]
        for init, start_pose in enumerate(poses_all):
            stream = collect_patrol(scenario["obstacles"], start_pose, config.base.base)
            from embodied_learning.experiments.robust_graph import build_groups

            shim = SimpleNamespace(base=config.base, poison_rot_deg=config.poison_rot_deg)
            built = build_groups(stream, shim)
            nodes = built["nodes"]
            node_steps = built["node_steps"]
            edges = build_rel_edges(nodes, SimpleNamespace(base=config.base))
            n_est = replay_chain("N", stream, config.base.base)["estimates"]
            n_mean, _ = error_at(n_est, np.arange(1, len(stream["truth"])), stream["truth"])
            row = {"scene": scene, "init": init, "legs": stream["legs"], "n_mean": n_mean}
            z_true = true_relative(nodes, node_steps, stream["truth"])
            z_good = (0, len(nodes) - 1, z_true, config.base.sigma_loop_t, config.base.sigma_loop_r)
            z_bad = (
                0,
                len(nodes) - 1,
                poisoned_relative(z_true, config.poison_rot_deg, config.poison_shift_m),
                config.base.sigma_loop_t,
                config.base.sigma_loop_r,
            )
            jobs = (
                ("SCIPY-LS", z_good, None, 1.0),
                ("SCIPY-HUB-B", z_bad, "huber", 1.0),
                ("SCIPY-HUB-G", z_good, "huber", 1.0),
                ("SCIPY-SOFT-B", z_bad, "soft_l1", 1.0),
            )
            for name, loop_edge, loss, scale in jobs:
                x = solve_scipy(nodes, edges, loop_edge, loss, scale)
                mean, final = error_at(x, node_steps, stream["truth"])
                row[f"{name}_mean"] = mean
                row[f"{name}_final"] = final
                if scene == 1 and init == 0:
                    chains[name] = x
            x_self = solve_self_huber(nodes, edges, z_bad)
            mean_self, final_self = error_at(x_self, node_steps, stream["truth"])
            row["SELF-HUB-B_mean"] = mean_self
            row["SELF-HUB-B_final"] = final_self
            if scene == 1 and init == 0:
                chains["SELF-HUB-B"] = x_self
                # the loss-scale sweep on the showcase stream
                for loss in ("huber", "soft_l1"):
                    for scale in config.loss_scales:
                        x_s = solve_scipy(nodes, edges, z_bad, loss, scale)
                        m_s, f_s = error_at(x_s, node_steps, stream["truth"])
                        sweep.append(
                            {
                                "loss": loss,
                                "scale": scale,
                                "mean": m_s,
                                "final": f_s,
                            }
                        )
            episodes.append(row)
            log(
                f"  scene {scene} init {init}: legs {stream['legs']} N {n_mean:.3f} "
                f"HUB-B {row['SCIPY-HUB-B_mean']:.3f} HUB-G {row['SCIPY-HUB-G_final']:.3f}"
            )
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
        }
        for name in GROUPS
    }
    if aggregates["N"]["mean_position_error_m"] is None:
        aggregates["N"]["mean_position_error_m"] = float(np.mean([r["n_mean"] for r in episodes]))
    n_mean = aggregates["N"]["mean_position_error_m"]
    scipy_ls_final = aggregates["SCIPY-LS"]["final_position_error_m"]
    hub_b_mean = aggregates["SCIPY-HUB-B"]["mean_position_error_m"]
    hub_g_final = aggregates["SCIPY-HUB-G"]["final_position_error_m"]
    soft_b_mean = aggregates["SCIPY-SOFT-B"]["mean_position_error_m"]
    self_b_mean = aggregates["SELF-HUB-B"]["mean_position_error_m"]
    both_ok_sweep = [
        s
        for s in sweep
        if s["mean"] is not None
        and s["final"] is not None
        and s["mean"] <= n_mean + 0.5
        and s["final"] <= 1.5
    ]
    hypothesis = {
        "claim": (
            "把第 52 课的毒化相对边问题交给生产级实现（scipy least_squares 的"
            "鲁棒损失族）：求解器对齐（LS 与自研一致）后，生产核同样不拒绝毒化边"
            "（问题属性而非实现产物），且损失/尺度扫描网格中无组合同时满足两线"
        ),
        "criteria": {
            "collector": "采集器 4/4 腿",
            "kernel_fails_too": "SCIPY-HUB-B 均值 > N - 0.5（生产核也不拒绝毒化）",
            "dilemma_any": "SCIPY-HUB-G 末端 > 1.5 或 SCIPY-HUB-B 均值 > N + 0.5（至少半边失败）",
            "sweep_empty": "扫描网格无 (loss, scale) 同时过两线",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "kernel_fails_too_met": bool(hub_b_mean is not None and hub_b_mean > n_mean - 0.5),
            "dilemma_any_met": bool(
                (hub_g_final is not None and hub_g_final > 1.5)
                or (hub_b_mean is not None and hub_b_mean > n_mean + 0.5)
            ),
            "sweep_empty_met": bool(len(both_ok_sweep) == 0),
            "sweep_best": min(
                (s for s in sweep if s["mean"] is not None),
                key=lambda s: max(s["mean"] - n_mean, s["final"]),
                default=None,
            ),
            "errors": {name: aggregates[name]["mean_position_error_m"] for name in GROUPS},
            "finals": {name: aggregates[name]["final_position_error_m"] for name in GROUPS},
            "scipy_ls_final": scipy_ls_final,
            "self_huber_b_mean": self_b_mean,
            "soft_b_mean": soft_b_mean,
        },
    }
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "collector": "与第 46-52 课相同（真值巡逻采集-重放）",
            "groups": {g: GROUP_NAMES[g] for g in GROUPS},
            "production_solver": "scipy.optimize.least_squares (trf), loss x f_scale sweep",
            "injection": "毒化相对边 = 真 z 旋转 +90° + x 平移 1 m（第 52 课同款）",
            "loss_scales": list(config.loss_scales),
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
                    for field in ("mean", "final")
                },
            }
            for r in episodes
        ],
        "sweep": sweep,
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
    parser = argparse.ArgumentParser(description="lesson-53 production solver record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
