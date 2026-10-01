"""Registered seed 29 examples, from checked round-eleven archives."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from embodied_learning.navigation_geometry import NavigationRig, world_points
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "bypass_low": "14cm低路障：直行会碰，两侧可绕",
    "bypass_beam": "42–56cm横杆：上部会碰，两侧可绕",
    "bypass_crate": "1m箱体：雷达也能看见",
    "clear_overhead": "75cm高杆：高于整车，可直接通过",
}
METHODS = {"lidar_only": ("仅平面雷达", "#c53e51"), "lidar_depth": ("雷达＋深度", "#119c74")}


def main():
    configure_plot_font()
    directory = ROOT / "results/navigation_reinforcement_v11"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    assert len(summary["rows"]) == 24
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.2), layout="constrained")
    for ax, profile in zip(axes.flat, PROFILES, strict=True):
        annotations = []
        for method, (name, color) in METHODS.items():
            row = next(
                r
                for r in summary["rows"]
                if r["profile"] == profile and r["seed"] == 29 and r["method"] == method
            )
            a = read_checked(directory / row["file"], row["sha256"])
            ax.plot(a["truth"][:, 0], a["truth"][:, 1], color=color, lw=2)
            ax.plot(*a["truth"][-1, :2], "o" if row["passed"] else "x", color=color, ms=8)
            status = (
                "到达"
                if row["passed"]
                else {
                    "heavy_contact_abort": "接触中止",
                    "no_path": "找不到路",
                    "stalled": "停滞",
                    "timeout": "超时",
                }.get(row["status"], row["status"])
            )
            annotations.append(f"{name}：{status}，{row['elapsed_sim_s']:.1f}s")
        for b in a["actual_boxes"][:-1]:
            ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fc="#d8dee6"))
        b = a["actual_boxes"][-1]
        ax.add_patch(
            Rectangle(
                (b[0], b[2]),
                b[1] - b[0],
                b[3] - b[2],
                fc="#e8b06f" if profile != "clear_overhead" else "none",
                ec="#a76815",
                ls="--" if profile == "clear_overhead" else "-",
                alpha=0.75,
            )
        )
        ax.plot(*a["goal"][-1], "*", color="#c38b00", ms=13)
        ax.set(
            title=PROFILES[profile],
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(1, 11),
            ylim=(4.3, 7.8),
            aspect="equal",
        )
        ax.text(
            0.02,
            0.03,
            "\n".join(annotations),
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    fig.legend(
        [Line2D([], [], color=c, lw=2) for _, c in METHODS.values()]
        + [
            Line2D([], [], color="#c38b00", marker="*", ls="none"),
            Line2D([], [], color="#334155", marker="x", ls="none"),
        ],
        [n for n, _ in METHODS.values()] + ["真实目标", "接触中止位置"],
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(
        Path(__file__).parent / "lesson69-round11-trajectories.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    failures = [
        (directory, r, "正式")
        for r in summary["rows"]
        if r["method"] == "lidar_depth" and not r["passed"]
    ]
    dev_directory = ROOT / "results/navigation_bypass_development_v11_metric"
    dev = json.loads((dev_directory / "summary.json").read_text(encoding="utf8"))
    failures += [
        (dev_directory, r, "开发")
        for r in dev["rows"]
        if r["profile"] == "bypass_crate" and r["method"] == "lidar_depth" and r["seed"] == 0
    ]
    fig, axes = plt.subplots(
        1, len(failures), figsize=(5 * len(failures), 4.6), layout="constrained", squeeze=False
    )
    rig = NavigationRig()
    for ax, (folder, row, cohort) in zip(axes.flat, failures, strict=True):
        a = read_checked(folder / row["file"], row["sha256"])
        ax.plot(a["truth"][:, 0], a["truth"][:, 1], color="#119c74", lw=2)
        valid_plans = [p[np.isfinite(p).all(axis=1)] for p in a["plan"] if np.isfinite(p).any()]
        if valid_plans:
            p = valid_plans[-1]
            ax.plot(p[:, 0], p[:, 1], color="#c38b00", ls="--", lw=1.2)
        box = a["actual_boxes"][-1]
        ax.add_patch(
            Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fc="#e8b06f", alpha=0.65)
        )
        for i, b in enumerate(rig.parts):
            corners = np.array([[b[0], b[2], 0], [b[1], b[2], 0], [b[1], b[3], 0], [b[0], b[3], 0]])
            p = world_points(corners, a["truth"][-1])
            ax.add_patch(
                Polygon(p[:, :2], fill=i > 0, fc="#247cc1", ec="#247cc1", lw=1.5, alpha=0.7)
            )
        ax.set(
            title=f"{cohort}种子{row['seed']} · {PROFILES[row['profile']].split('：')[0]}\n"
            + ("接触中止" if row["status"] == "heavy_contact_abort" else "无路退出"),
            xlabel="x (m)",
            ylabel="y (m)",
            xlim=(4.1, 7.0),
            ylim=(4.7, 7.3),
            aspect="equal",
        )
        ax.text(
            0.02,
            0.02,
            f"峰值接触力 {row['contact_peak_force_n']:.2f}N\n全程最小真实间隙 {row['min_clearance_m'] * 100:.2f}cm",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
    fig.legend(
        [
            Line2D([], [], color="#119c74", lw=2),
            Line2D([], [], color="#c38b00", ls="--"),
            Line2D([], [], color="#247cc1", lw=2),
        ],
        ["实际轨迹", "最后一条有效规划", "蓝框底盘/深蓝支架相机"],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    fig.savefig(
        Path(__file__).parent / "lesson69-round11-failures.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for ax, profile in zip(axes, PROFILES, strict=True):
        for i, (method, (_, color)) in enumerate(METHODS.items()):
            rows = [r for r in summary["rows"] if r["profile"] == profile and r["method"] == method]
            n = sum(r["passed"] for r in rows)
            ax.bar(i, n, color=color, width=0.55)
            ax.text(i, n + 0.1, f"{n}/{len(rows)}", ha="center", fontsize=12)
        ax.set(
            title=PROFILES[profile].split("：")[0],
            xticks=(0, 1),
            xticklabels=[n for n, _ in METHODS.values()],
            ylabel="真实到点并停稳",
            ylim=(0, 3.7),
        )
    fig.savefig(
        Path(__file__).parent / "lesson69-round11-outcomes.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
