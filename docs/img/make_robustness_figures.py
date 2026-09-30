"""Fourth-round plots: signed sweeps, matched environments, paths and actual delay."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from embodied_learning.experiments.footprint_tracking import digest
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.navigation_studies import condition_label

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/navigation_robustness_v4"
OUT = Path(__file__).parent
OUTCOMES = ("到达且余量合格", "到达但余量未过", "未到达且无接触", "实体接触")
OUTCOME_COLORS = ("#119c74", "#d68026", "#94a3b8", "#c53e51")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def categories(rows):
    safe = sum(r["safe_completed"] for r in rows)
    unsafe = sum(r["reached"] for r in rows) - safe
    contacts = sum(r["contacts"] for r in rows)
    return [safe, unsafe, len(rows) - safe - unsafe - contacts, contacts]


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
    benchmark = json.loads(
        (ROOT / "docs/benchmarks/navigation-robustness-v4.json").read_text(encoding="utf8")
    )
    assert len(summary["rows"]) == 233 and benchmark["summary_sha256"] == digest(
        DATA / "summary.json"
    )
    rows = summary["rows"]
    conditions = summary["protocol"]["conditions"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for column, (factor, title, unit) in enumerate(
        (
            ("dy_m", "世界y位置误差", "cm"),
            ("yaw_deg", "朝向误差", "°"),
            ("delay_steps", "实际执行延迟", "s"),
        )
    ):
        keys = ["zero"] + [k for k, c in conditions.items() if c[factor]]
        keys.sort(key=lambda k: conditions[k][factor])
        values = [
            conditions[k][factor]
            * (100 if factor == "dy_m" else 0.1 if factor == "delay_steps" else 1)
            for k in keys
        ]
        subsets = [[r for r in rows if r["cohort"] == "scan" and r["method"] == k] for k in keys]
        counts = np.array([categories(rr) for rr in subsets])
        a, b = axes[:, column]
        bottom = np.zeros(len(keys))
        for i, color in enumerate(OUTCOME_COLORS):
            a.bar(range(len(keys)), counts[:, i], 0.7, bottom=bottom, color=color)
            bottom += counts[:, i]
        for i, count in enumerate(counts[:, 0]):
            a.text(i, 9.2, f"{count}/9", ha="center", fontsize=9)
        labels = [f"{v:g}" for v in values]
        a.set(
            title=title,
            xticks=range(len(keys)),
            xticklabels=labels,
            ylim=(0, 10.6),
            ylabel="回合数",
            xlabel=f"注入值 ({unit})",
        )
        minimum = [min(r["expanded_swept_min_m"] for r in rr) * 1000 for rr in subsets]
        median = [np.median([r["expanded_swept_min_m"] for r in rr]) * 1000 for rr in subsets]
        b.plot(range(len(keys)), minimum, "o-", color="#c53e51", label="9回合中最小")
        b.plot(range(len(keys)), median, "s--", color="#247cc1", label="9回合中位数")
        b.axhline(0, color="#475569", lw=1)
        b.set_yscale("symlog", linthresh=10)
        b.set_yticks([-100, -10, 0, 10, 100, 1000], ["−100", "−10", "0", "10", "100", "1000"])
        b.set(
            xticks=range(len(keys)),
            xticklabels=labels,
            xlabel=f"注入值 ({unit})",
            ylabel="最小外扩分离量 (mm；近0放大)",
        )
        b.legend(fontsize=9)
        for ax in (a, b):
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
    fig.suptitle("窄路边界扫描：每个等级3种高度×3个新种子；柱上数字为余量合格到达", fontsize=15)
    fig.legend(
        handles=[Patch(color=c, label=l) for c, l in zip(OUTCOME_COLORS, OUTCOMES, strict=True)],
        loc="outside lower center",
        ncol=4,
    )
    save(fig, "lesson69-round4-boundaries.png")

    fig, ax = plt.subplots(figsize=(14, 4.6), layout="constrained")
    chosen = ("zero", "y_minus_4", "y_plus_4", "yaw_minus_4", "yaw_plus_4", "delay_4")
    labels = ["零扰动", "y−4cm", "y+4cm", "朝向−4°", "朝向+4°", "延迟0.4s"]
    for j, layout in enumerate(("plaza", "street", "narrow")):
        rr = [
            r for r in rows if r["layout"] == layout and r["obstacle"] == "crate" and r["seed"] == 8
        ]
        for i, key in enumerate(chosen):
            r = next(r for r in rr if r["method"] == key)
            category = categories([r]).index(1)
            ax.add_patch(
                Rectangle(
                    (i - 0.46, j - 0.43),
                    0.92,
                    0.86,
                    facecolor=OUTCOME_COLORS[category],
                    edgecolor="white",
                )
            )
            ax.text(
                i,
                j,
                ("到达且余量合格", "到达但余量未过", "未到达", "实体接触")[category],
                ha="center",
                va="center",
                fontsize=11,
                color="white" if category != 2 else "#243041",
            )
    ax.set(
        xlim=(-0.6, 5.6),
        ylim=(2.6, -0.6),
        xticks=range(6),
        xticklabels=labels,
        yticks=range(3),
        yticklabels=["开阔场地", "街区", "窄路"],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle("同算法、同扰动、同普通箱体和种子8：环境如何改变结果？", fontsize=15)
    fig.text(
        0.5,
        -0.06,
        "每个格仅1回合；主扫描还包含另外两种高度与两个种子，不能把本图当作环境泛化率。",
        ha="center",
        fontsize=11,
    )
    save(fig, "lesson69-round4-environments.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 5.6), layout="constrained")
    selected = ("zero", "y_plus_4", "yaw_plus_4", "delay_4")
    for ax, key in zip(axes.flat, selected, strict=True):
        row = next(
            r
            for r in rows
            if r["layout"] == "narrow"
            and r["obstacle"] == "crate"
            and r["seed"] == 8
            and r["method"] == key
        )
        d = read_checked(DATA / row["file"], row["sha256"])
        label, _, color = condition_label(conditions[key])
        for x0, x1, y0, y1, *_ in d["actual_boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#bfc8cd", ec="white"))
        first = d["plan"][0]
        ax.plot(first[:, 0], first[:, 1], ls=":", color="#d68026", lw=1.3)
        ax.plot(d["truth"][:, 0], d["truth"][:, 1], color=color, lw=2)
        ax.plot(d["estimate"][:, 0], d["estimate"][:, 1], color="#334155", ls="--", lw=1)
        ax.scatter(*d["truth"][-1, :2], marker="x", c=color, s=55, zorder=6)
        ax.scatter(*d["goal"][-1], marker="*", c="#d79a00", s=90, zorder=6)
        status = "实体接触" if row["contacts"] else "到达" if row["reached"] else "未到达"
        ax.set(
            title=f"{label} · {status} · {row['elapsed_sim_s']:.1f}s",
            xlim=(1, 11),
            ylim=(4.8, 7.2),
            xlabel="x (m)",
            ylabel="y (m)",
        )
        ax.set_aspect("equal")
    fig.suptitle("固定选例：窄路 / 普通箱体 / 新种子8；展示最大正偏差与最大延迟", fontsize=14)
    fig.legend(
        handles=[
            Line2D([], [], color="#475569", label="实际运动（彩色实线）"),
            Line2D([], [], color="#334155", ls="--", label="有偏估计"),
            Line2D([], [], color="#d68026", ls=":", label="首次规划"),
            Line2D([], [], marker="x", color="#475569", ls="", label="实际终点"),
        ],
        loc="outside lower center",
        ncol=4,
    )
    save(fig, "lesson69-round4-trajectories.png")

    row = next(
        r
        for r in benchmark["rows"]
        if r["layout"] == "narrow"
        and r["obstacle"] == "crate"
        and r["seed"] == 8
        and r["method"] == "delay_4"
    )
    d = read_checked(DATA / row["file"], row["sha256"])
    t = np.arange(len(d["truth"])) * 0.1
    first, applied = (
        row["first_protective_brake_frame"],
        row["first_protective_brake_applied_frame"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    shown = t >= max(0, first * 0.1 - 0.8) - 1e-9
    for ax, index, label in zip(axes[:2], (0, 1), ("线速度 (m/s)", "角速度 (rad/s)"), strict=True):
        desired = np.where(d["braking_active"], 0, d["requested_command"][:, index])
        ax.step(
            t[shown],
            desired[shown],
            where="post",
            ls="--",
            color="#64748b",
            label="当前请求（刹车记0）",
        )
        executed = np.r_[d["commands"][:-1, index], d["incoming_velocity"][-1, index]]
        ax.step(t[shown], executed[shown], where="post", color="#c53e51", label="实际执行")
        ax.scatter(t[-1], d["incoming_velocity"][-1, index], color="#c53e51", marker="x", s=55)
        ax.set(ylabel=label)
        ax.legend(fontsize=9, loc="best")
    axes[2].plot(t[shown], d["expanded_gap_m"][shown] * 1000, color="#c53e51", label="实际外扩间隙")
    axes[2].axhline(0, color="#475569", lw=1)
    axes[2].set(ylabel="外扩分离量 (mm)")
    for ax in axes:
        for frame, color, style in ((first, "#d68026", ":"), (applied, "#8b5cf6", "--")):
            if frame is not None:
                ax.axvline(frame * 0.1, color=color, ls=style, lw=1.3)
        ax.set(xlim=(max(0, first * 0.1 - 0.8), t[-1] + 0.06), xlabel="时间 (s)")
        ax.grid(alpha=0.15)
    fig.suptitle("种子8 / 延迟0.4秒：保护指令已发出，实际执行仍在继续", fontsize=15)
    fig.text(
        0.5,
        -0.02,
        f"橙点线：{first * 0.1:.1f}s请求制动；紫虚线：{applied * 0.1:.1f}s执行制动。末端叉号是接触时速度；不画终止零标记冒充停稳。",
        ha="center",
        fontsize=11,
    )
    save(fig, "lesson69-round4-delay.png")


if __name__ == "__main__":
    main()
