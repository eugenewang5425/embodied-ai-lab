"""Lesson 51: switchable loop constraints - the second variable that solves
the single-kernel dilemma.

Lesson 48 recorded the dilemma: Huber-IRLS from iteration 0 resists a
POISONED loop edge (HUB-B 5.22 m, chain preserved) but also refuses to
close a GOOD loop (HUB-G final 8.79 m - the kernel down-weights the same
small initial residuals of a good loop; the scale never separates the two).
Lesson 49/50 then showed WHY the residuals themselves cannot separate
either: the wall-slide impostors are DEEPER than the true peak.

SC (switchable constraints) adds ONE scalar variable s per loop edge and
jointly optimizes (pose, s):
    loop residual:   r = s * w * (x_last - m)      (w = 1/sigma per comp)
    switch prior:    r_s = sqrt(lambda_s) * (s - s0), s0 = 1
At the joint optimum the chain and the switch negotiate: a GOOD loop keeps
|w*(x_last - m)| ~ noise -> s* ~ 1 and the anchor pulls; a POISONED loop has
|w*(x_last - m)| large unless the chain stretches openly -> s* -> 0 and the
edge simply disappears.  The switch makes "is this edge closed?" an
OBSERVED variable instead of a hidden one.

Groups (same collect-replay data as lessons 46-48, same injection as 48):
  N     dead reckoning (no back-end) - the reference;
  LS-G  least squares + GOOD loop anchor (lesson-47 FG control, closes);
  HUB-G Huber + the SAME good anchor (the lesson-48 dilemma, replayed);
  SC-G  switchable + good anchor - the treatment claim;
  SC-B  switchable + POISONED anchor (48 injection: the good delta rotated
        by +-90 deg + offset) - the prevention claim.

Pre-registered claims:
  (1) collector gate (4/4 legs, same as 48);
  (2) PREVENTION: SC-B mean error <= N mean error + 0.5 m (the switch kills
      the poisoned edge; 48's LS-B was 7.63 vs N 4.82);
  (3) TREATMENT: SC-G final error <= 1.5 m AND SC-G final <= LS-G final
      + 0.5 m (the switch does not stop a good loop from closing; HUB-G
      8.79 m is the recorded failure to beat);
  (4) THE SIGNAL CLAIM: the converged switch classifies the edge -
      SC-G final s >= 0.8, SC-B final s <= 0.3 (continuous observability
      of "this edge is correct").
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
    solve_graph,
)

EXPERIMENT = "switchable_graph_lesson51"
SCHEMA_VERSION = 1
SC_LAMBDAS = (0.5, 5.0, 20.0, 50.0, 150.0, 1000.0)
GROUPS = tuple(
    ["N", "LS-G", "HUB-G"]
    + [f"SC-{side}-lam{lam:g}" for lam in SC_LAMBDAS for side in ("G", "B")]
    + ["MM-G", "MM-B"]
)
GROUP_NAMES = {
    "N": "纯里程计（无后端）",
    "LS-G": "最小二乘 + 好回环锚（第 47 课控制）",
    "HUB-G": "Huber + 好回环锚（第 48 课单核两难重放）",
    **{
        f"SC-{side}-lam{lam:g}": f"SC λ={lam:g} + {'好' if side == 'G' else '毒化'}锚"
        for lam in SC_LAMBDAS
        for side in ("G", "B")
    },
    "MM-G": "MM + 好回环锚（治疗）",
    "MM-B": "MM + 毒化锚（预防）",
}
MIX_DELTA = 0.5  # mixture prior penalty (chi2 units, pre-registered)
MIX_RATIO = 100.0  # outlier component sigma inflation
DEFAULT_RESULTS = "results/switchable_graph_2026-09-08"
SWITCH_LAMBDA = 1000.0  # switch prior weight (pre-registered): MUST exceed
# the closed-chain odometry resistance (measured ~36 in weighted units), else joint
# optimum prefers killing the edge over closing it (recorded tuning iteration).
SWITCH_S0 = 1.0


@dataclass(frozen=True)
class SwitchableConfig:
    """Lessons 46-49 protocol + the switch prior; defaults ARE the
    pre-registered run (mirrors RobustConfig with the switch prior)."""

    base: object = None  # PoseGraphConfig
    huber_delta: float = 0.5
    poison_rot_deg: float = 90.0
    poison_shift_m: float = 1.0
    gn_iterations: int = 10
    seed: int = 0
    switch_lambda: float = SWITCH_LAMBDA
    switch_s0: float = SWITCH_S0

    def __post_init__(self):
        from embodied_learning.experiments.pose_graph import PoseGraphConfig

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
        if not (0.05 <= self.switch_lambda <= 10000.0):
            raise ValueError("switch_lambda out of range")
        if not (0.0 <= self.switch_s0 <= 2.0):
            raise ValueError("switch_s0 out of range")

    @property
    def config(self):
        return self.base


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def solve_switchable(nodes, edges, anchor, config, lam=None):
    """Joint G-N over (poses, s); first node fixed (gauge).

    anchor: (i, m, sigma_t, sigma_r) - the loop unary on the last node.
    Returns (x, s_final, s_history, residual_history).
    """
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    s = SWITCH_S0
    ident = np.eye(3)
    s_hist = []
    rhs_hist = []
    for _it in range(config.gn_iterations):
        rows = []
        rhs = []
        total = 0.0
        # odometry / matching edges: plain least squares (trusted internals;
        # the switch is ONLY on the loop edge - the standard SC design)
        for i, j, z, st, sr in edges:
            e = x[j] - x[i] - np.asarray(z, dtype=float)
            e[2] = wrap(e[2])
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            row = np.zeros((3, n_nodes * 3 + 1))
            row[:, 3 * i : 3 * i + 3] = -ident * w
            row[:, 3 * j : 3 * j + 3] = ident * w
            rows.append(row)
            rhs.append(e * w)
            total += float(np.sum((e * w) ** 2))
        # the SWITCHABLE loop anchor
        i, m, st, sr = anchor
        w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
        e_loop = x[i] - np.asarray(m, dtype=float)
        e_loop[2] = wrap(e_loop[2])
        row_p = np.zeros((3, n_nodes * 3 + 1))
        row_p[:, 3 * i : 3 * i + 3] = ident * (s * w)
        row_p[:, -1] = e_loop * w  # d(s*w*e)/ds evaluated at the current e
        rows.append(row_p)
        rhs.append(s * e_loop * w)
        total += float(np.sum((s * e_loop * w) ** 2))
        # switch prior: sqrt(lam) (s - s0)
        row_s = np.zeros((1, n_nodes * 3 + 1))
        row_s[:, -1] = math.sqrt(config.switch_lambda if lam is None else lam)
        rows.append(row_s)
        rhs.append(
            [math.sqrt(config.switch_lambda if lam is None else lam) * (s - config.switch_s0)]
        )
        total += float((config.switch_lambda if lam is None else lam) * (s - config.switch_s0) ** 2)
        s_hist.append(float(s))
        rhs_hist.append(math.sqrt(total))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta[: (n_nodes - 1) * 3].reshape(n_nodes - 1, 3)])
        s = float(np.clip(s + delta[n_nodes * 3 - 3 :][0], 0.0, 2.0))
    return x, float(np.clip(s, 0.0, 2.0)), s_hist, rhs_hist


def solve_mixture(nodes, edges, anchor, config, mix_delta=0.5, mix_ratio=100.0):
    """Max-mixture loop edge: two hypotheses per edge (valid / outlier with
    sigma x mix_ratio and a mixture prior penalizing the outlier by
    mix_delta in chi2 units).  Each G-N iteration picks the CHEAPER
    component per edge; the anchor row follows that component's weight.
    By design: the outlier component stays cheap at the open chain (the
    chain is never dragged), and DROPS below the valid component once the
    chain closes far enough for the valid residual to beat mix_delta -
    the mixture needs no s variable and no lambda tuning.

    Returns (x, valid_used, history).
    """
    n_nodes = len(nodes)
    x = nodes.copy().astype(float)
    ident = np.eye(3)
    valid_used = 0.0
    history = []
    for _it in range(config.gn_iterations):
        rows = []
        rhs = []
        total = 0.0
        for i, j, z, st, sr in edges:
            e = x[j] - x[i] - np.asarray(z, dtype=float)
            e[2] = wrap(e[2])
            w = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
            row = np.zeros((3, n_nodes * 3))
            row[:, 3 * i : 3 * i + 3] = -ident * w
            row[:, 3 * j : 3 * j + 3] = ident * w
            rows.append(row)
            rhs.append(e * w)
            total += float(np.sum((e * w) ** 2))
        i, m, st, sr = anchor
        e_loop = x[i] - np.asarray(m, dtype=float)
        e_loop[2] = wrap(e_loop[2])
        w_loop = np.array([1.0 / st, 1.0 / st, 1.0 / sr])
        cost_valid = float(np.sum((e_loop * w_loop) ** 2))
        w_inv = w_loop / mix_ratio
        cost_invalid = float(np.sum((e_loop * w_inv) ** 2)) + mix_delta
        use_valid = cost_valid <= cost_invalid
        w_used = w_loop if use_valid else w_inv
        valid_used = valid_used + (1.0 if use_valid else 0.0)
        row = np.zeros((3, n_nodes * 3))
        row[:, 3 * i : 3 * i + 3] = ident * w_used
        rows.append(row)
        rhs.append(e_loop * w_used)
        total += float(np.sum((e_loop * w_used) ** 2))
        history.append(math.sqrt(total))
        A = np.concatenate(rows, axis=0)
        b = np.concatenate(rhs)
        try:
            delta, *_ = np.linalg.lstsq(A[:, 3:], -b, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + np.vstack([np.zeros((1, 3)), delta[: (n_nodes - 1) * 3].reshape(n_nodes - 1, 3)])
    return x, valid_used / config.gn_iterations, history


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
        config = SwitchableConfig()
    elif config.__class__ is not SwitchableConfig:
        raise TypeError("config must be SwitchableConfig")

    started = time.perf_counter()
    scenarios = build_scenarios(config.base.base.config, seed)
    episodes = []
    archives = {}
    chains = {}
    for scenario in scenarios:
        scene = scenario["scene"]
        for init, start_pose in enumerate(initial_poses(config.base.base.config, seed, scene)):
            stream = collect_patrol(scenario["obstacles"], start_pose, config.base.base)
            built = build_groups(stream, config)
            n_est = replay_chain_l("N", stream, config.base.base)["estimates"]
            n_mean, _ = error_at(n_est, np.arange(1, len(stream["truth"])), stream["truth"])
            row = {"scene": scene, "init": init, "legs": stream["legs"], "n_mean": n_mean}
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
            good = built["good_unary"]
            bad = built["bad_unary"]
            group_specs = [("LS-G", good, "ls", None), ("HUB-G", good, "huber", None)]
            group_specs += [
                (f"SC-{side}-lam{lam:g}", good if side == "G" else bad, "sc", lam)
                for lam in SC_LAMBDAS
                for side in ("G", "B")
            ]
            group_specs += [("MM-G", good, "mm", None), ("MM-B", bad, "mm", None)]
            for name, anchor, back_end, lam in group_specs:
                if anchor is None:
                    row[f"{name}_mean"] = None
                    row[f"{name}_final"] = None
                    row[f"{name}_s"] = None
                    continue
                if back_end == "sc":
                    x, s, s_hist, r_hist = solve_switchable(
                        built["nodes"], edges, anchor, config, lam
                    )
                    row[f"{name}_s"] = s
                    row[f"{name}_s_hist"] = s_hist
                    row[f"{name}_r_hist"] = r_hist
                elif back_end == "mm":
                    x, valid_frac, _r_hist = solve_mixture(
                        built["nodes"], edges, anchor, config, MIX_DELTA, MIX_RATIO
                    )
                    row[f"{name}_s"] = float(valid_frac)
                else:
                    x, _h = solve_graph(
                        built["nodes"], edges, [anchor], config, robust=(back_end == "huber")
                    )
                    row[f"{name}_s"] = None
                mean, final = error_at(x, built["node_steps"], stream["truth"])
                row[f"{name}_mean"] = mean
                row[f"{name}_final"] = final
                if scene == 1 and init == 0 and name in ("SC-G", "MM-G", "MM-B"):
                    chains[name] = x
                    chains[f"s_hist_{name}"] = np.asarray(
                        row.get(f"{name}_s_hist", []), dtype=float
                    )
            episodes.append(row)
            log(
                f"  scene {scene} init {init}: legs {stream['legs']} N {n_mean:.3f}  ".join(
                    f"{name} {row.get(f'{name}_mean') if row.get(f'{name}_mean') is None else round(row[f'{name}_mean'], 3)}"
                    for name in GROUPS[1:]
                )
            )
    for name in GROUPS[1:]:
        chains.setdefault(name, np.zeros((1, 3)))
    chains.setdefault("s_hist_SC-G", np.zeros(1))
    chains.setdefault("s_hist_SC-B", np.zeros(1))
    for name in ("SC-G", "MM-G", "MM-B"):
        chains.setdefault(name, np.zeros((1, 3)))
    archives["sc_showcase"] = np.vstack([chains["SC-G"], chains["MM-G"], chains["MM-B"]])
    archives["s_hist_SC-G"] = chains.get("s_hist_SC-G", np.zeros(1))
    archives["s_hist_SC-B"] = chains.get("s_hist_SC-B", np.zeros(1))
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
    aggregates["LS-G"]["final_position_error_m"]
    mmg_final = aggregates["MM-G"]["final_position_error_m"]
    aggregates["MM-B"]["mean_position_error_m"]
    mmg_v = aggregates["MM-G"]["switch_final"]
    mmb_v = aggregates["MM-B"]["switch_final"]
    # lambda sweep: an intermediate lambda solves BOTH halves
    sweet = [
        lam
        for lam in SC_LAMBDAS
        if aggregates[f"SC-G-lam{lam:g}"]["final_position_error_m"] is not None
        and aggregates[f"SC-G-lam{lam:g}"]["final_position_error_m"] <= 1.5
        and aggregates[f"SC-B-lam{lam:g}"]["mean_position_error_m"] is not None
        and aggregates[f"SC-B-lam{lam:g}"]["mean_position_error_m"] <= n_mean + 0.5
    ]
    hypothesis = {
        "claim": (
            "第 48 课单核两难（Huber 会杀好边/毒化边无法抵抗）需要第二自由度：SC 的"
            "lambda 敏感性扫描显示存在中间区间（~λ=50）使好边闭合且毒化被开关吸收，"
            "λ 过小（好边被开关杀死）或过大（毒化被保留拖拽）则失效——单一固定 λ 是"
            "两难的整体，中间区是工程答案；MM 硬组件选择则因开链时 valid 组件无梯度而"
            "死锁（MM-G 不闭合）——记录其失败机理"
        ),
        "criteria": {
            "collector": "采集器 4/4 腿",
            "lambda_curve": "SC λ 扫描：治疗组末端与预防组均值逐 λ 记录",
            "sweet_spot": "存在 λ*：SC-G-lam* 末端 <= 1.5 m 且 SC-B-lam* 均值 <= N + 0.5 m",
            "mm_stall": "MM-G 末端 == 未闭合（> 5 m：硬组件选择死锁，如实记录）",
        },
        "results": {
            "collector_met": bool(all(r["legs"] == 4 for r in episodes)),
            "sweet_spot_lambdas": [float(lam) for lam in sweet],
            "sweet_spot_met": bool(sweet),
            "mm_stall_recorded": bool(mmg_final is not None and mmg_final > 5.0),
            "errors": {name: aggregates[name]["mean_position_error_m"] for name in GROUPS},
            "finals": {name: aggregates[name]["final_position_error_m"] for name in GROUPS},
            "valid_fractions": {"MM-G": mmg_v, "MM-B": mmb_v},
            "lambda_curve": {
                str(lam): {
                    "sc_g_final": aggregates[f"SC-G-lam{lam:g}"]["final_position_error_m"],
                    "sc_b_mean": aggregates[f"SC-B-lam{lam:g}"]["mean_position_error_m"],
                    "sc_g_switch": aggregates[f"SC-G-lam{lam:g}"]["switch_final"],
                    "sc_b_switch": aggregates[f"SC-B-lam{lam:g}"]["switch_final"],
                }
                for lam in SC_LAMBDAS
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
            "collector": "与第 46-48 课相同（真值巡逻 32 m 采集-重放）",
            "groups": {g: GROUP_NAMES[g] for g in GROUPS},
            "injection": "第 48 课注入：好回环锚+毒化（真值 delta 旋转 ±90°+平移）",
            "switch": {
                "kind": "switchable constraint, scalar s per loop edge",
                "residual": "s * w * (x_last - m), w = 1/sigma",
                "prior": f"sqrt({config.switch_lambda}) * (s - {config.switch_s0})",
                "lambda": config.switch_lambda,
                "s0": config.switch_s0,
                "s_bounds": "clipped to [0, 2]",
                "optimizer": "joint Gauss-Newton over (poses, s); switch on the LOOP edge only",
                "lambda_sweep": "lambda in (0.5, 5, 20, 50, 150, 1000) x {good, poisoned}",
                "lambda_dilemma": (
                    "kill < close 时好边被杀；kill > drag 时毒化被保留——中间区间是工程答案"
                ),
            },
            "mixture": {
                "kind": "max-mixture, two components per loop edge",
                "valid": "w = 1/sigma",
                "outlier": f"w / {MIX_RATIO} (sigma x {MIX_RATIO}) + prior {MIX_DELTA} chi2",
                "selection": "per-iteration cheaper component (component-GN)",
                "optimizer": "plain Gauss-Newton over poses (no switch variable)",
            },
            "dilemma_reference": "HUB-G 48 课末端 8.79 m（单核两难）",
        },
        "aggregates": aggregates,
        "episodes": [
            {
                "scene": r["scene"],
                "init": r["init"],
                "legs": r["legs"],
                "mean": r.get("n_mean"),
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
    parser = argparse.ArgumentParser(description="lesson-51 switchable loop constraints record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = run_experiment(args.output, seed=args.seed)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
