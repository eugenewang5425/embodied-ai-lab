"""Registered fixed examples for speed and observation map studies."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from embodied_learning.experiments.navigation_reinforcement import map_scene
from embodied_learning.navigation_geometry import NavigationRig
from embodied_learning.navigation_map_update import EvidenceMap
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
COLORS = ("#c53e51", "#119c74")
NAMES = {
    "normal_speed": "正常速度0.6m/s",
    "low_speed": "固定低速0.2m/s",
    "map_additive": "地图只加不删",
    "map_evidence": "观测证据修正地图",
}
PROFILES = {
    "added": "新增箱体",
    "removed": "移走箱体墙",
    "shifted": "箱体侧移90cm",
    "unchanged": "正确箱体",
    "removed_low": "移走低路障墙",
    "occluded": "旧箱体被遮挡",
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load(round_number):
    data = ROOT / f"results/navigation_reinforcement_v{round_number}"
    summary = json.loads((data / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == len(summary["protocol"]["cases"])
    return data, summary


def trajectories(round_number, data, summary, filters, name):
    fig, axes = plt.subplots(
        1, len(filters), figsize=(12.5, 4.2), layout="constrained", squeeze=False
    )
    for ax, (match, title) in zip(axes.ravel(), filters, strict=True):
        rows = [r for r in summary["rows"] if all(r[k] == v for k, v in match.items())]
        for color, key in zip(COLORS, summary["protocol"]["methods"], strict=True):
            row = next(r for r in rows if r["method"] == key)
            a = read_checked(data / row["file"], row["sha256"])
            ax.plot(a["truth"][:, 0], a["truth"][:, 1], c=color, label=NAMES[key])
            ax.plot(*a["truth"][-1, :2], "o", c=color)
        for box in a["actual_boxes"]:
            ax.add_patch(
                Rectangle(
                    (box[0], box[2]), box[1] - box[0], box[3] - box[2], fc="#d8dee6", alpha=0.5
                )
            )
        ax.plot(*a["goal"][-1], "*", c="#d79a00", ms=13)
        ax.set(
            title=title,
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(1, 11),
            ylim=(4.3, 7.7),
            aspect="equal",
        )
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        fontsize=11,
        frameon=False,
    )
    save(fig, name)


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
    data, s = load(7)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), layout="constrained")
    for ax, obstacle, title in zip(
        axes, ("crate", "low", "beam"), ("箱体", "14cm低路障", "悬空横杆"), strict=True
    ):
        groups = [
            [
                r
                for r in s["rows"]
                if r["cohort"] == "scan" and r["obstacle"] == obstacle and r["method"] == m
            ]
            for m in s["protocol"]["methods"]
        ]
        for i, (g, color) in enumerate(zip(groups, COLORS, strict=True)):
            n = sum(r["passed"] for r in g)
            ax.bar(i, n, color=color, width=0.55)
            ax.text(i, n + 0.08, f"{n}/{len(g)}", ha="center", fontsize=12)
        ax.set(
            title=title,
            xticks=(0, 1),
            xticklabels=("正常速度\n最高0.6m/s", "固定低速\n最高0.2m/s"),
            ylim=(0, 3.65),
            ylabel="实际到点并停稳",
        )
    save(fig, "lesson69-round7-outcomes.png")
    trajectories(
        7,
        data,
        s,
        [
            ({"cohort": "scan", "obstacle": "crate", "seed": 17}, "登记选例：箱体种子17，0.2s延迟"),
            (
                {"cohort": "delay", "condition_key": "delay_4", "seed": 17},
                "登记选例：同一种子，0.4s延迟",
            ),
        ],
        "lesson69-round7-trajectories.png",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), layout="constrained")
    for color, m in zip(COLORS, s["protocol"]["methods"], strict=True):
        row = next(
            r
            for r in s["rows"]
            if r["seed"] == 17 and r["condition_key"] == "delay_4" and r["method"] == m
        )
        a = read_checked(data / row["file"], row["sha256"])
        t = np.arange(len(a["truth"])) * 0.1
        axes[0].plot(t, np.linalg.norm(a["world_velocity"][:, :2], axis=1), c=color, label=NAMES[m])
        axes[1].step(t, a["contact_interval"][:, 1], c=color, where="pre", label=NAMES[m])
    axes[0].set(title="低速的代价：行驶更久", xlabel="时间 (s)", ylabel="身体实际速度 (m/s)")
    axes[1].axhline(40, c="#64748b", ls="--", label="轻擦上限40N")
    axes[1].set(title="延迟0.4s的撞角冲击", xlabel="时间 (s)", ylabel="前0.1s单点力峰值 (N)")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), fontsize=9, frameon=False)
        ax.grid(alpha=0.15)
    save(fig, "lesson69-round7-speed-contact.png")
    data, s = load(8)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), layout="constrained")
    for ax, profile in zip(axes.ravel(), PROFILES, strict=True):
        for i, (m, color) in enumerate(zip(s["protocol"]["methods"], COLORS, strict=True)):
            rows = [r for r in s["rows"] if r["profile"] == profile and r["method"] == m]
            n = sum(r["passed"] for r in rows)
            ax.bar(i, n, width=0.55, color=color)
            ax.text(i, n + 0.08, f"{n}/{len(rows)}", ha="center", fontsize=12)
        ax.set(
            title=PROFILES[profile],
            xticks=(0, 1),
            xticklabels=("只加不删", "证据更新"),
            ylim=(0, 3.7),
            ylabel="实际到点并停稳",
        )
    save(fig, "lesson69-round8-outcomes.png")
    trajectories(
        8,
        data,
        s,
        [
            ({"profile": "removed", "seed": 20}, "移走墙：剩余不可确认空闲的底层仍挡路"),
            ({"profile": "shifted", "seed": 20}, "错位箱体：地图删旧加新不一定提高到达"),
        ],
        "lesson69-round8-trajectories.png",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    row = next(
        r
        for r in s["rows"]
        if r["profile"] == "shifted" and r["seed"] == 20 and r["method"] == "map_evidence"
    )
    a = read_checked(data / row["file"], row["sha256"])
    prior, _ = map_scene("shifted")
    memory = EvidenceMap(prior, NavigationRig(**s["protocol"]["rig"]), True, True)
    before_layer = memory.occupied[:, :, 3].copy()
    layers = {}
    for frame in range(min(100, len(a["truth"]) - 1) + 1):
        if a["reason"][frame] != "abort_braking":
            memory.update(
                a["estimate"][frame],
                a["ranges"][frame],
                a["hits"][frame],
                a["depth"][frame],
                a["depth_valid"][frame],
            )
        layers[frame] = memory.occupied[:, :, 3].copy()
    for ax, frame in zip(axes[0], (0, min(100, len(a["truth"]) - 1)), strict=True):
        before = before_layer
        current = layers[frame]
        rgb = np.ones((*before.shape, 3)) * 0.96
        rgb[before & current] = [0.3, 0.35, 0.42]
        rgb[before & ~current] = [0.1, 0.65, 0.5]
        rgb[~before & current] = [0.95, 0.57, 0.2]
        ax.imshow(rgb, origin="lower", extent=(0, 12, 0, 12), interpolation="nearest")
        ax.set(
            title=f"35cm高度层 · t={frame * 0.1:.1f}s",
            xlim=(4.8, 7.2),
            ylim=(4.8, 8),
            xlabel="x (m)",
            ylabel="y (m)",
        )
    for color, m in zip(COLORS, s["protocol"]["methods"], strict=True):
        row = next(
            r
            for r in s["rows"]
            if r["profile"] == "shifted" and r["seed"] == 20 and r["method"] == m
        )
        b = read_checked(data / row["file"], row["sha256"])
        t = np.arange(len(b["truth"])) * 0.1
        axes[1, 0].plot(t, b["map_error"][:, 0], c=color, label=NAMES[m])
        axes[1, 1].plot(t, b["map_counts"][:, 1].cumsum(), c=color, label=NAMES[m])
    axes[1, 0].set(title="多标的俯视格子（仅评分）", xlabel="时间 (s)", ylabel="10cm格子数量")
    axes[1, 1].set(title="累计有证据清除的三维格子", xlabel="时间 (s)", ylabel="10cm体素数量")
    for ax in axes[1]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), fontsize=9, frameon=False)
        ax.grid(alpha=0.15)
    fig.suptitle("灰：仍占据  ·  绿：先验占据已清除  ·  橙：先验没有、观测新加入", fontsize=12)
    save(fig, "lesson69-round8-map-evidence.png")


if __name__ == "__main__":
    main()
