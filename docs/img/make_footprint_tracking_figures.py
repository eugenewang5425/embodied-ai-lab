"""Rebuild round-two charts and a compact benchmark from hash-checked records."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.navigation_studies import LABELS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/footprint_tracking_v2"
OUT = Path(__file__).parent
KEYS = ("original", "tangent", "retained", "combined")
NAMES = [LABELS[k][0] for k in KEYS]
COLORS = [LABELS[k][2] for k in KEYS]


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    configure_plot_font()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 132
    assert (
        hashlib.sha256((DATA / "protocol.json").read_bytes()).hexdigest()
        == summary["protocol_sha256"]
    )
    rows = summary["rows"]
    records = {r["file"]: read_checked(DATA / r["file"], r["sha256"]) for r in rows}

    def record(key, seed=0):
        return records[f"narrow_crate_s{seed}_{key}.npz"]

    aggregate = {}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
    for i, key in enumerate(KEYS):
        values = [
            sum(
                r["reached"]
                for r in rows
                if r["method"] == key and r["cohort"] == "fixed" and r["layout"] == layout
            )
            for layout in ("plaza", "street", "narrow")
        ]
        bars = a.bar(np.arange(3) + (i - 1.5) * 0.2, values, 0.18, color=COLORS[i], label=NAMES[i])
        a.bar_label(bars, labels=[f"{v}/9" for v in values], padding=3, fontsize=9)
        held = sum(r["reached"] for r in rows if r["method"] == key and r["cohort"] == "heldout")
        b.bar(i, held, 0.6, color=COLORS[i])
        b.scatter(i, held, color=COLORS[i], s=40)
        b.text(i, 0.22, f"{held}/6", ha="center", fontsize=11)
        rr = [r for r in rows if r["method"] == key]
        controls = np.concatenate([records[r["file"]]["control_wall_s"] for r in rr])
        plans = [x["wall_s"] for r in rr for x in r["planning_log"]]
        timing = lambda v: dict(
            zip(
                ("p95_s", "p99_s", "max_s"),
                map(float, (*np.percentile(v, [95, 99]), max(v))),
                strict=True,
            )
        )
        aggregate[key] = {
            "fixed_reached": sum(values),
            "fixed_total": 27,
            "fixed_by_layout": dict(zip(("plaza", "street", "narrow"), values, strict=True)),
            "heldout_reached": held,
            "heldout_total": 6,
            "contacts": sum(r["contacts"] for r in rr),
            "min_body_clearance_m": min(r["min_clearance_m"] for r in rr),
            "min_expanded_separation_m": min(r["expanded_swept_min_m"] for r in rr),
            "controller": timing(controls),
            "planning": timing(plans),
            "control_calls": len(controls),
            "control_calls_over_100ms": int(np.sum(controls > 0.1)),
        }
    a.set(
        xticks=np.arange(3),
        xticklabels=["开阔场地", "街区通道", "窄通道"],
        ylim=(0, 10.5),
        ylabel="真正到达的回合数",
        title="固定批次：每种环境9回合",
    )
    b.set(
        xticks=range(4),
        xticklabels=["旧组", "只换跟踪", "只保留曲线", "两项组合"],
        ylim=(0, 6.7),
        ylabel="真正到达的回合数",
        title="留出种子3、4：窄路仍为0/6",
    )
    for ax in (a, b):
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=10)
    save(fig, "lesson69-round2-results.png")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), layout="constrained")
    for i, (ax, key) in enumerate(zip(axes.flat, KEYS, strict=True)):
        d = record(key)
        for box in d["actual_boxes"]:
            ax.add_patch(
                Rectangle(
                    (box[0], box[2]), box[1] - box[0], box[3] - box[2], color="#94a3b8", alpha=0.5
                )
            )
        path = d["plan"][0]
        path = path[np.isfinite(path).all(1)]
        ax.plot(path[:, 0], path[:, 1], ls="--", lw=1.3, color="#64748b", label="相同的首次规划")
        ax.plot(d["truth"][:, 0], d["truth"][:, 1], color=COLORS[i], lw=2.2, label="实际运动")
        ax.scatter(
            *d["truth"][-1, :2],
            marker="o" if key == "combined" else "x",
            s=65,
            color=COLORS[i],
            zorder=5,
        )
        ax.scatter(10.5, 6, marker="*", s=135, color="#d79a00", label="目标")
        ax.set(
            xlim=(1.1, 10.9),
            ylim=(4.85, 7.15),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{NAMES[i]} · {(len(d['truth']) - 1) * 0.1:.1f}s · {'到达' if key == 'combined' else '停止'}",
        )
        ax.grid(alpha=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles[1] = Line2D([], [], color="#334155", lw=2.2)
    labels[1] = "彩色实线：各组实际运动"
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=10)
    save(fig, "lesson69-round2-trajectories.png")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), layout="constrained")
    for seed, color in ((0, "#119c74"), (2, "#d97706")):
        d = record("combined", seed)
        t = np.arange(len(d["truth"])) * 0.1
        label = f"组合组·种子{seed}（{'到达' if seed == 0 else '失败'}）"
        axes[0, 0].plot(t, d["initial_cross_m"] * 100, color=color, label=label)
        axes[0, 1].plot(t, np.rad2deg(d["initial_heading_rad"]), color=color)
        axes[1, 0].plot(t, d["expanded_gap_m"] * 100, color=color)
        axes[1, 1].plot(
            t,
            d["commands"][:, 0],
            color=color,
            drawstyle="steps-post",
            label=f"种子{seed}执行线速度",
        )
        braking = d["reason"] == "swept_body_brake"
        axes[1, 1].scatter(t[braking], d["commands"][braking, 0], color=color, s=18, marker="x")
    titles = [
        "相对相同首次规划：横向偏差",
        "相对相同首次规划：朝向偏差",
        "外扩足印侵入余量，但实体尚未接触",
        "刹车触发时刻不同（叉号）",
    ]
    ylabels = ["偏差 (cm)", "偏差 (°)", "投影分离量 (cm)", "执行速度 (m/s)"]
    for ax, title, y in zip(axes.flat, titles, ylabels, strict=True):
        ax.set(xlim=(4.8, 8.0), xlabel="仿真时间 (s)", ylabel=y, title=title)
        ax.grid(alpha=0.17)
    axes[1, 0].set_ylim(-1, 5)
    axes[1, 0].axhline(0, color="#64748b", ls="--", lw=1)
    axes[1, 1].set_ylim(-0.03, 0.67)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=10
    )
    save(fig, "lesson69-round2-braking.png")
    failures = []
    reference = record("combined", 0)
    for seed in range(5):
        d = record("combined", seed)
        n = min(len(d["truth"]), len(reference["truth"]))
        diff = np.flatnonzero(
            np.any(np.abs(d["truth"][:n] - reference["truth"][:n]) > 1e-12, axis=1)
        )
        brakes = np.flatnonzero(d["reason"] == "swept_body_brake")
        r = next(
            r
            for r in rows
            if r["method"] == "combined"
            and r["layout"] == "narrow"
            and r["obstacle"] == "crate"
            and r["seed"] == seed
        )
        k = int(brakes[0])
        failures.append(
            {
                "seed": seed,
                "status": r["status"],
                "first_plan_equal_to_seed0": bool(
                    np.array_equal(d["plan"][0], reference["plan"][0], equal_nan=True)
                ),
                "first_motion_difference_frame": int(diff[0]) if len(diff) else None,
                "brake_frames": brakes.tolist(),
                "first_brake_previous_command": d["commands"][k - 1].tolist(),
                "first_brake_executed_command": d["commands"][k].tolist(),
                "terminal_prior_expanded_gap_m": float(d["prior_expanded_gap_m"][-1]),
                "terminal_expanded_gap_m": float(d["expanded_gap_m"][-1]),
                "terminal_planning_reason": r["planning_log"][-1]["reason"],
            }
        )
    payload = {
        "summary_sha256": hashlib.sha256((DATA / "summary.json").read_bytes()).hexdigest(),
        "protocol": summary["protocol"],
        "protocol_sha256": summary["protocol_sha256"],
        "wall_s": summary["wall_s"],
        "aggregates": aggregate,
        "seed_diagnostics": failures,
        "rows": [{k: v for k, v in r.items() if k != "planning_log"} for r in rows],
    }
    (ROOT / "docs/benchmarks/footprint-tracking-v2.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
