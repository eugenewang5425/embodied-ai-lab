"""Result, braking-mechanism and executed-trajectory charts from frozen records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from embodied_learning.braking_control import MODES
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.navigation_studies import LABELS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/braking_navigation_v3"
OUT = Path(__file__).parent
KEYS = list(MODES)
NAMES = ["旧版\n参照", "一致预测\n独立减速", "仅同比例\n制动", "仅曲率\n限速", "制动\n＋限速"]
COLORS = [LABELS[k][2] for k in KEYS]


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    configure_plot_font()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 230
    rows = summary["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), layout="constrained")
    selections = (
        (
            "固定普通场景（开阔＋街区）",
            lambda r: r["cohort"] == "fixed" and r["layout"] != "narrow",
            18,
        ),
        ("固定窄路（种子0/1/2）", lambda r: r["cohort"] == "fixed" and r["layout"] == "narrow", 9),
        ("新噪声留出（种子5/6/7）", lambda r: r["cohort"] == "heldout_noise", 9),
        ("偏置窄弯留出（种子5/6/7）", lambda r: r["cohort"] == "heldout_layout", 9),
    )
    for ax, (title, keep, total) in zip(axes.flat, selections, strict=True):
        for i, key in enumerate(KEYS):
            rr = [r for r in rows if keep(r) and r["method"] == key]
            assert len(rr) == total
            reached, safe = sum(r["reached"] for r in rr), sum(r["safe_completed"] for r in rr)
            ax.bar(i - 0.18, reached, 0.33, color=COLORS[i], alpha=0.35)
            ax.bar(
                i + 0.18, safe, 0.33, color=COLORS[i], hatch="//", edgecolor="white", linewidth=0.3
            )
            for x, value in ((i - 0.18, reached), (i + 0.18, safe)):
                ax.text(x, value + 0.18, str(value), ha="center", fontsize=10)
        ax.set(
            title=f"{title} · 每组{total}回合",
            xticks=range(5),
            xticklabels=NAMES,
            ylim=(0, total * 1.17),
            ylabel="回合数",
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor="#64748b", alpha=0.35, label="到达"),
            Patch(
                facecolor="#64748b", hatch="//", edgecolor="white", label="到达且全程外扩余量合格"
            ),
        ],
        loc="outside upper center",
        ncol=2,
        fontsize=11,
    )
    save(fig, "lesson69-round3-results.png")

    counter_dir = ROOT / "results/braking_counterfactual_v3"
    counter = json.loads((counter_dir / "summary.json").read_text(encoding="utf8"))
    traces = read_checked(counter_dir / "traces.npz", counter["traces_sha256"])
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5), layout="constrained")
    for coupled, color, label in ((False, "#dc7840", "独立减速"), (True, "#119c74", "同比例减速")):
        name = f"s2_delay0.2_{'coupled' if coupled else 'independent'}"
        commands = np.vstack((traces[name + "_commands"], [0, 0]))
        # The acceleration-limited stop clock, including the 0.2s held command.
        times = np.arange(len(commands)) * 0.1
        axes[0].step(times, commands[:, 1], where="post", color=color, label=label)
        rr = sorted(
            (r for r in counter["rows"] if r["delay_s"] == 0.2 and r["coupled"] == coupled),
            key=lambda r: r["seed"],
        )
        xs = np.arange(2) + (0.18 if coupled else -0.18)
        bars = axes[1].bar(xs, [r["true_expanded_min_gap_m"] * 1000 for r in rr], 0.32, color=color)
        axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[0].set(
        title="同一输入：转向是先停下还是一起减慢",
        xlabel="从触发状态起的时间 (s)",
        ylabel="角速度 (rad/s)",
    )
    axes[0].legend()
    axes[1].set(
        title="相同0.2秒延迟下，完整刹停的最小余量",
        xticks=[0, 1],
        xticklabels=["开发种子0触发帧", "开发种子2触发帧"],
        ylabel="外扩分离量 (mm)",
        ylim=(-10, 13),
    )
    axes[1].axhline(0, color="#64748b", ls="--", lw=1)
    for ax in axes:
        ax.grid(alpha=0.15)
        ax.set_axisbelow(True)
    save(fig, "lesson69-round3-stop-mechanism.png")

    rr = [
        r for r in rows if r["layout"] == "narrow" and r["obstacle"] == "crate" and r["seed"] == 0
    ]
    records = {r["method"]: read_checked(DATA / r["file"], r["sha256"]) for r in rr}
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.5), layout="constrained")
    for key in KEYS[1:]:
        d, color = records[key], LABELS[key][2]
        t = np.arange(len(d["truth"])) * 0.1
        axes[0, 0].step(t, d["commands"][:, 0], where="post", color=color, label=LABELS[key][0])
        axes[0, 1].step(t, d["commands"][:, 1], where="post", color=color)
        axes[1, 0].plot(t, d["initial_cross_m"] * 100, color=color)
        axes[1, 1].plot(t, d["expanded_gap_m"] * 100, color=color)
    for ax, title, ylabel in zip(
        axes.flat,
        (
            "实际线速度",
            "实际角速度",
            "相对相同首次规划的横向偏差",
            "外扩余量：正值仍可能被噪声观测拒绝",
        ),
        ("m/s", "rad/s", "cm", "cm"),
        strict=True,
    ):
        ax.set(xlim=(4.5, 11), title=title, xlabel="仿真时间 (s)", ylabel=ylabel)
        ax.grid(alpha=0.15)
    axes[1, 1].set_ylim(-1, 5)
    axes[1, 1].axhline(0, color="#64748b", ls="--", lw=1)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=10
    )
    save(fig, "lesson69-round3-control.png")

    fig, axes = plt.subplots(3, 2, figsize=(12.6, 8.5), layout="constrained")
    for ax, key in zip(axes.flat, KEYS):
        d = records[key]
        for box in d["actual_boxes"]:
            ax.add_patch(
                Rectangle(
                    (box[0], box[2]), box[1] - box[0], box[3] - box[2], color="#94a3b8", alpha=0.5
                )
            )
        plan = d["plan"][0]
        plan = plan[np.isfinite(plan).all(axis=1)]
        ax.plot(plan[:, 0], plan[:, 1], ls="--", color="#64748b", lw=1)
        ax.plot(d["truth"][:, 0], d["truth"][:, 1], color=LABELS[key][2], lw=2)
        ax.scatter(10.5, 6, marker="*", s=90, color="#d79a00")
        r = next(r for r in rr if r["method"] == key)
        ax.scatter(
            *d["truth"][-1, :2], marker="o" if r["reached"] else "x", color=LABELS[key][2], s=40
        )
        ax.set(
            xlim=(1.1, 10.9),
            ylim=(4.85, 7.15),
            aspect="equal",
            title=f"{LABELS[key][0]} · {r['elapsed_sim_s']:.1f}s · {'到达' if r['reached'] else '终止'}",
            xlabel="x (m)",
            ylabel="y (m)",
        )
        ax.grid(alpha=0.15)
    axes[2, 1].axis("off")
    axes[2, 1].text(
        0.03,
        0.67,
        "固定选例：窄路 / 普通箱体 / 种子0\n\n圆点=到达；叉号=失败后实际停稳\n旧版到达仍侵入余量；仅同比例制动合格\n完整230回合结果见汇总图",
        transform=axes[2, 1].transAxes,
        fontsize=12,
        va="center",
    )
    fig.legend(
        handles=[
            Line2D([], [], color="#64748b", ls="--", label="相同首次规划"),
            Line2D([], [], color="#334155", label="彩色线：实际轨迹（含制动尾段）"),
            Line2D([], [], color="#d79a00", marker="*", ls="", label="目标"),
        ],
        loc="outside upper center",
        ncol=3,
    )
    save(fig, "lesson69-round3-trajectories.png")


if __name__ == "__main__":
    main()
