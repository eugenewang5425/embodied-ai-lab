"""Plot frozen round-five outcomes and synchronized actuator-fault evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.navigation_studies import LABELS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/execution_safety_v5"
OUT = Path(__file__).parent
SHORT = ("原参照", "排队预测", "本地保护", "两者结合")


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
    assert len(summary["rows"]) == 135
    keys = summary["protocol"]["methods"]
    rows = summary["rows"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), layout="constrained")
    outcome_colors = ("#119c74", "#d68026", "#94a3b8", "#c53e51")
    for i, (delay, title) in enumerate(
        (("zero", "立即执行"), ("delay_2", "排队0.2秒"), ("delay_4", "排队0.4秒"))
    ):
        groups = [
            [
                r
                for r in rows
                if r["cohort"] == "scan" and r["condition_key"] == delay and r["method"] == key
            ]
            for key in keys
        ]
        counts = np.array(
            [
                [
                    sum(r["safe_completed"] for r in g),
                    sum(r["reached"] for r in g) - sum(r["safe_completed"] for r in g),
                    len(g) - sum(r["reached"] for r in g) - sum(r["contacts"] for r in g),
                    sum(r["contacts"] for r in g),
                ]
                for g in groups
            ]
        )
        top, bottom = axes[:, i]
        base = np.zeros(4)
        for category, color in enumerate(outcome_colors):
            top.bar(range(4), counts[:, category], 0.65, bottom=base, color=color)
            base += counts[:, category]
        for j, g in enumerate(groups):
            top.text(j, 9.3, f"{sum(r['safe_completed'] for r in g)}/9", ha="center", fontsize=9)
        top.set(title=title, ylabel="回合数", ylim=(0, 10.5), xticks=range(4), xticklabels=SHORT)
        mins = [min(r["expanded_swept_min_m"] for r in g) * 1000 for g in groups]
        medians = [np.median([r["expanded_swept_min_m"] for r in g]) * 1000 for g in groups]
        bottom.plot(range(4), mins, "o-", c="#c53e51", label="9回合最小")
        bottom.plot(range(4), medians, "s--", c="#247cc1", label="9回合中位数")
        bottom.axhline(0, c="#475569", lw=1)
        bottom.set(
            title="预留空间是否始终守住？",
            ylabel="外扩分离量 (mm)",
            xticks=range(4),
            xticklabels=SHORT,
        )
        bottom.legend(fontsize=8)
        bottom.grid(alpha=0.15)
    fig.suptitle(
        "第69课第五轮：提前预测与本地保护，分别改善了什么？\n每格=三高度×种子11/12/13；柱顶为到达且余量合格",
        fontsize=14,
    )
    fig.legend(
        handles=[
            Patch(color=c, label=t)
            for c, t in zip(
                outcome_colors,
                ("到达且余量合格", "到达但余量不足", "未到达、无接触", "实体接触"),
                strict=True,
            )
        ],
        loc="outside lower center",
        ncol=4,
    )
    save(fig, "lesson69-round5-results.png")

    selected = [
        next(
            r
            for r in rows
            if r["layout"] == "narrow"
            and r["obstacle"] == "crate"
            and r["seed"] == 11
            and r["condition_key"] == "delay_4"
            and r["profile"] == "normal"
            and r["method"] == k
        )
        for k in keys
    ]
    records = [read_checked(DATA / r["file"], r["sha256"]) for r in selected]
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), layout="constrained")
    for ax, key, row, d in zip(axes.ravel(), keys, selected, records, strict=True):
        for box in d["actual_boxes"]:
            ax.add_patch(
                Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], color="#cbd5e1")
            )
        plan = d["plan"][0]
        plan = plan[np.isfinite(plan).all(axis=1)]
        ax.plot(plan[:, 0], plan[:, 1], c="#b58b20", ls=":", lw=1.4)
        ax.plot(d["truth"][:, 0], d["truth"][:, 1], c=LABELS[key][2], lw=2)
        ax.plot(*d["truth"][-1, :2], "x", c=LABELS[key][2], ms=9)
        ax.plot(*d["goal"][0], "*", c="#b58b20", ms=12)
        ax.set(
            xlim=(1, 11),
            ylim=(4.4, 7.6),
            aspect="equal",
            xlabel="世界x (m)",
            ylabel="世界y (m)",
            title=LABELS[key][0]
            + f" · {'接触终止' if row['contacts'] else '停住、未到达'}\n最小外扩分离量 {row['expanded_swept_min_m'] * 1000:+.2f}mm",
        )
        ax.grid(alpha=0.15)
    fig.suptitle(
        "相同初始条件下的独立实际轨迹：窄路/箱体/种子11/排队0.4秒\n位置估计精确，实际与估计重合；淡灰为实体障碍，外扩余量另行评分",
        fontsize=13,
    )
    fig.legend(
        handles=[
            Line2D([], [], c="#475569", lw=2, label="实际运动（面板对应色）"),
            Line2D([], [], c="#b58b20", ls=":", label="初次计划，未必执行完"),
            Line2D([], [], c="#b58b20", marker="*", ls="", label="目标"),
            Line2D([], [], c="#475569", marker="x", ls="", label="终点"),
        ],
        loc="outside lower center",
        ncol=4,
    )
    save(fig, "lesson69-round5-trajectories.png")

    selected = [
        next(r for r in rows if r["cohort"] == "blackout" and r["seed"] == 11 and r["method"] == k)
        for k in keys
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, layout="constrained")
    for key, row, style in zip(keys, selected, ("-", "--", "-", "--"), strict=True):
        d = read_checked(DATA / row["file"], row["sha256"])
        t = np.arange(len(d["truth"])) * 0.1
        axes[0].step(
            t,
            d["incoming_velocity"][:, 0],
            where="post",
            c=LABELS[key][2],
            ls=style,
            lw=2,
            label=LABELS[key][0],
        )
        axes[1].plot(t, d["expanded_gap_m"] * 1000, c=LABELS[key][2], ls=style, lw=2)
        if row["contacts"]:
            axes[0].plot(t[-1], d["incoming_velocity"][-1, 0], "x", c=LABELS[key][2], ms=9)
    for ax in axes:
        ax.axvspan(5, 6, color="#cbd5e1", alpha=0.3)
        ax.axvline(5.1, c="#475569", ls=":", lw=1)
        ax.grid(alpha=0.15)
        ax.set_xlim(4.5, 6.6)
    axes[0].set(
        title="5–6秒不给新控制结果；5.1秒消息年龄达到0.2秒",
        ylabel="当前真实速度 (m/s)",
        ylim=(-0.03, 0.75),
    )
    fig.legend(
        handles=[
            Line2D([], [], c=LABELS[k][2], ls=style, lw=2, label=LABELS[k][0])
            for k, style in zip(keys, ("-", "--", "-", "--"), strict=True)
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=9,
    )
    axes[1].axhline(0, c="#475569", lw=1)
    axes[1].set(ylabel="当前外扩分离量 (mm)", xlabel="仿真时间 (s)", ylim=(-110, 600))
    fig.suptitle(
        "失联期间车继续走：只有执行端保护组真正停住\n固定种子11；参照/预测轨迹重合，保护/结合重合；×保留接触时非零速度",
        fontsize=13,
    )
    save(fig, "lesson69-round5-blackout.png")


if __name__ == "__main__":
    main()
