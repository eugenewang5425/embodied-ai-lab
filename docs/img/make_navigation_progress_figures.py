"""Fixed registered examples for rounds nine/ten, rendered from checked archives."""

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

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("#c53e51", "#119c74")
NAMES = {
    "progress_old": "旧进度与门控",
    "progress_fixed": "进度与门控一致",
    "depth_coarse": "深度80×60",
    "depth_dense": "深度320×240",
}
PROFILES = {
    "added": "新增箱体",
    "removed": "移走箱体墙",
    "shifted": "箱体侧移90cm",
    "unchanged": "正确箱体",
    "removed_low": "移走14cm低墙",
    "occluded": "箱体被遮挡",
    "present_low": "真实14cm低墙（负例）",
    "present_beam": "真实悬空横杆墙（负例）",
}


def load(round_number):
    directory = ROOT / f"results/navigation_reinforcement_v{round_number}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 36
    return directory, summary


def save(fig, name):
    fig.savefig(Path(__file__).parent / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected(directory, summary, profile, seed, method):
    row = next(
        r
        for r in summary["rows"]
        if r["profile"] == profile and r["seed"] == seed and r["method"] == method
    )
    return row, read_checked(directory / row["file"], row["sha256"])


def outcomes(number, summary):
    profiles = list(dict.fromkeys(r["profile"] for r in summary["rows"]))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), layout="constrained")
    for ax, profile in zip(axes.flat, profiles, strict=True):
        for i, (method, color) in enumerate(
            zip(summary["protocol"]["methods"], COLORS, strict=True)
        ):
            rows = [r for r in summary["rows"] if r["profile"] == profile and r["method"] == method]
            n = sum(r["passed"] for r in rows)
            ax.bar(i, n, color=color, width=0.55)
            ax.text(i, n + 0.1, f"{n}/{len(rows)}", ha="center", fontsize=12)
        ax.set(
            title=PROFILES[profile],
            xticks=(0, 1),
            xticklabels=[NAMES[m] for m in summary["protocol"]["methods"]],
            ylim=(0, 3.7),
            ylabel="真实到点并停稳",
        )
    save(fig, f"lesson69-round{number}-outcomes.png")


def motion(number, directory, summary, profiles, seed):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), layout="constrained")
    for col, profile in enumerate(profiles):
        for color, method in zip(COLORS, summary["protocol"]["methods"], strict=True):
            _, a = selected(directory, summary, profile, seed, method)
            axes[0, col].plot(a["truth"][:, 0], a["truth"][:, 1], color=color, label=NAMES[method])
            axes[0, col].plot(*a["truth"][-1, :2], "o", color=color)
            t = np.arange(len(a["truth"])) * 0.1
            axes[1, col].step(t, a["requested_command"][:, 0], color=color, label=NAMES[method])
            axes[1, col].plot(
                t,
                np.linalg.norm(a["world_velocity"][:, :2], axis=1),
                color=color,
                ls="--",
                lw=1.1,
            )
        for b in a["actual_boxes"]:
            axes[0, col].add_patch(
                Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc="#d8dee6", alpha=0.65)
            )
        axes[0, col].plot(*a["goal"][-1], "*", color="#d79a00", ms=13)
        axes[0, col].set(
            title=f"种子{seed}：{PROFILES[profile]}",
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(1, 11),
            ylim=(4.3, 7.7),
            aspect="equal",
        )
        axes[1, col].set(
            title="前进请求与身体实际速度",
            xlabel="时间 (s)",
            ylabel="速度 (m/s)",
            ylim=(-0.02, 0.23),
        )
        axes[1, col].grid(alpha=0.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [Line2D([], [], color="#334155", ls="-"), Line2D([], [], color="#334155", ls="--")]
    labels += ["下排实线：前进请求", "下排虚线：身体实际速度"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    save(fig, f"lesson69-round{number}-motion.png")


def warmup_maps(directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), layout="constrained")
    for ax, method in zip(axes, summary["protocol"]["methods"], strict=True):
        _, a = selected(directory, summary, "removed_low", 26, method)
        initial, current = a["prior_map"], a["map_frames"][5]
        rgb = np.full((*initial.shape, 3), 0.96)
        rgb[initial & current] = [0.3, 0.35, 0.42]
        rgb[initial & ~current] = [0.1, 0.65, 0.5]
        rgb[~initial & current] = [0.95, 0.57, 0.2]
        ax.imshow(rgb, origin="lower", extent=(0, 12, 0, 12), interpolation="nearest")
        ax.set(
            title=f"{NAMES[method]} · t=0.5s",
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(4.8, 7.2),
            ylim=(4.5, 7.5),
        )
    fig.legend(
        handles=[
            Patch(color="#4d596b", label="旧占据仍在"),
            Patch(color="#1aa680", label="已有证据清除"),
            Patch(color="#f29133", label="观测新加入"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
    )
    save(fig, "lesson69-round10-warmup-map.png")


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
    for number in (9, 10):
        directory, summary = load(number)
        outcomes(number, summary)
        motion(
            number,
            directory,
            summary,
            ("occluded", "unchanged") if number == 9 else ("removed_low", "present_low"),
            23 if number == 9 else 26,
        )
        if number == 10:
            warmup_maps(directory, summary)


if __name__ == "__main__":
    main()
