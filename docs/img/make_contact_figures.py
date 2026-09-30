"""Readable fixed-case plots for the contact-tolerant navigation study."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/contact_navigation_v6"
OUT = Path(__file__).parent
KEYS = ("margin_strict", "zero_strict", "zero_light")
LABELS = ("6cm·禁接触", "0cm·禁接触", "0cm·允许轻擦")
COLORS = ("#c53e51", "#247cc1", "#119c74")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    configure_plot_font()
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    s = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    rows = s["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    for ax, ob, title in zip(
        axes, ("crate", "low", "beam"), ("普通箱体", "14cm低路障", "悬空横杆"), strict=True
    ):
        groups = [
            [r for r in rows if r["cohort"] == "scan" and r["obstacle"] == ob and r["method"] == k]
            for k in KEYS
        ]
        values = [sum(r["passed"] for r in g) for g in groups]
        bars = ax.bar(range(3), values, color=COLORS)
        for bar, v, g in zip(bars, values, groups, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.08,
                f"{v}/{len(g)}",
                ha="center",
                fontsize=12,
            )
        ax.set(
            ylim=(0, 3.6),
            xticks=range(3),
            xticklabels=("6cm\n禁接触", "0cm\n禁接触", "0cm\n允许轻擦"),
            title=title,
            ylabel="到点并停稳的回合数",
        )
        ax.set_yticks(range(4))
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("窄路主批次：真实到点是主标准，6cm余量仅诊断（排队0.2s）", fontsize=15)
    save(fig, "lesson69-round6-outcomes.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, ob, title in zip(
        axes,
        ("low", "crate"),
        ("成功对照：低路障，固定种子14", "失败对照：普通箱体，固定种子14"),
        strict=True,
    ):
        for key, label, color in zip(KEYS, LABELS, COLORS, strict=True):
            r = next(
                r
                for r in rows
                if r["layout"] == "narrow"
                and r["obstacle"] == ob
                and r["seed"] == 14
                and r["condition_key"] == "delay_2"
                and r["method"] == key
            )
            a = read_checked(DATA / r["file"], r["sha256"])
            xy = a["truth"][:, :2]
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                c=color,
                lw=2,
                label=f"{label}：{'到达' if r['passed'] else '未到达'}",
            )
            ax.plot(*xy[-1], "o", c=color, ms=7)
        for b in a["actual_boxes"]:
            ax.add_patch(
                Rectangle(
                    (b[0], b[2]),
                    b[1] - b[0],
                    b[3] - b[2],
                    facecolor="#cbd5e1",
                    edgecolor="#64748b",
                    alpha=0.7,
                )
            )
        ax.plot(10.5, 6, "*", c="#d79a00", ms=16, label="终点（25cm圈）")
        ax.add_patch(plt.Circle((10.5, 6), 0.25, fill=False, ec="#d79a00"))
        ax.set(
            xlim=(1, 11.2),
            ylim=(4.8, 7.2),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=title,
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=10)
        ax.grid(alpha=0.15)
    save(fig, "lesson69-round6-trajectories.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for delay, color, label in (
        ("delay_2", "#119c74", "实际排队0.2s"),
        ("delay_4", "#c53e51", "实际排队0.4s"),
    ):
        r = next(
            r
            for r in rows
            if r["layout"] == "narrow"
            and r["obstacle"] == "crate"
            and r["seed"] == 14
            and r["condition_key"] == delay
            and r["method"] == "zero_light"
        )
        a = read_checked(DATA / r["file"], r["sha256"])
        c = a["substep_contact"].reshape(-1, 5)
        tt = np.arange(1, len(c) + 1) * 0.002
        axes[0].plot(
            tt, c[:, 1], c=color, lw=1, label=f"{label} · 峰值{r['contact_peak_force_n']:.1f}N"
        )
        axes[1].plot(tt, c[:, 3] * 1000, c=color, lw=1, label=label)
    axes[0].axhline(40, c="#64748b", ls="--", label="轻擦上限40N")
    axes[1].axhline(5, c="#64748b", ls="--", label="压入上限5mm")
    axes[0].set(title="短促接触也可能有较强冲击", ylabel="单点法向力 (N)", xlabel="时间 (s)")
    axes[1].set(title="接触程度不能只看压入量", ylabel="求解器接触压入 (mm)", xlabel="时间 (s)")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(alpha=0.15)
    save(fig, "lesson69-round6-contact.png")


if __name__ == "__main__":
    main()
