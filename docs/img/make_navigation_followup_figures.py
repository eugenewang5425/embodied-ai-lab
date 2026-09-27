"""Generate lessons 69–71 figures from checked, completed experiment archives."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from embodied_learning.experiments.campus_patrol import TASKS
from embodied_learning.experiments.navigation_followups import DIRECTORIES, METHODS
from embodied_learning.footprint_planning import FootprintField
from embodied_learning.navigation_geometry import NavigationRig, world_points
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
COLORS = ["#dc7840", "#8b5cf6", "#119c74"]
NAMES = {
    "69": ["圆形栅格", "圆形米制", "朝向矩形"],
    "70": ["原判据", "位置预算", "预算＋停稳"],
    "71": ["无路即停", "低速恢复"],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(lesson):
    directory = ROOT / "results" / DIRECTORIES[lesson]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf8"))
    assert digest(directory / "protocol.json") == summary["protocol_sha256"]
    assert len(summary["rows"]) == {"69": 81, "70": 18, "71": 12}[lesson]
    return directory, summary


def record(directory, summary, method, **condition):
    row = next(
        r
        for r in summary["rows"]
        if r["method"] == method and all(r[k] == v for k, v in condition.items())
    )
    return row, read_checked(directory / row["file"], row["sha256"])


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def decorate(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.16)


def boxes(ax, data):
    for b in data["actual_boxes"]:
        ax.add_patch(Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], color="#94a3b8", alpha=0.55))


def footprint(ax, pose, margin, color, **kwargs):
    rig = NavigationRig()
    points = np.array(
        [
            [-rig.rear_m - margin, -rig.half_width_m - margin, 0],
            [rig.front_m + margin, -rig.half_width_m - margin, 0],
            [rig.front_m + margin, rig.half_width_m + margin, 0],
            [-rig.rear_m - margin, rig.half_width_m + margin, 0],
        ]
    )
    xy = world_points(points, pose)[:, :2]
    ax.add_patch(Polygon(xy, fill=False, edgecolor=color, **kwargs))


def lesson69(directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), layout="constrained")
    x = np.arange(3)
    for i, method in enumerate(METHODS["69"]):
        rows = [r for r in summary["rows"] if r["method"] == method]
        values = [
            sum(r["reached"] for r in rows if r["layout"] == layout)
            for layout in ("plaza", "street", "narrow")
        ]
        bar = axes[0].bar(x + (i - 1) * 0.23, values, 0.22, color=COLORS[i], label=NAMES["69"][i])
        axes[0].bar_label(bar, labels=[f"{v}/9" for v in values], padding=3, fontsize=10)
        narrow = [r["travelled_m"] for r in rows if r["layout"] == "narrow"]
        axes[1].scatter(np.full(9, i) + np.linspace(-0.09, 0.09, 9), narrow, color=COLORS[i], s=35)
    axes[0].set(
        xticks=x,
        xticklabels=["开阔场地", "街区通道", "窄通道"],
        ylim=(0, 11.5),
        ylabel="实际到达回合数",
        title="每布局9回合 · 三组均零碰撞",
    )
    axes[0].legend(loc="upper center", ncol=3, fontsize=9)
    axes[1].set(
        xticks=x,
        xticklabels=NAMES["69"],
        ylim=(-0.15, 4.2),
        ylabel="实际行驶里程 (m)",
        title="窄路仍全失败；矩形组走了多远？",
    )
    for ax in axes:
        decorate(ax)
    save(fig, "lesson69-results.png")

    row, a = record(directory, summary, "rectangle", layout="narrow", obstacle="crate", seed=0)
    plan = a["plan"][0]
    plan = plan[np.isfinite(plan).all(axis=1)]
    truth = a["truth"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), layout="constrained")
    for ax in axes:
        boxes(ax, a)
        ax.plot(plan[:, 0], plan[:, 1], ls="--", color="#64748b", label="首次规划（含未走部分）")
        ax.plot(truth[:, 0], truth[:, 1], color=COLORS[2], lw=2.3, label="实际轨迹")
        ax.scatter(*truth[-1, :2], marker="x", s=65, color="#b91c1c", label="停止位置")
        ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)")
    axes[0].set(xlim=(1, 11), ylim=(3.9, 8.1), title="固定选例：窄路 / 箱体 / 种子0")
    axes[0].scatter(10.5, 6, marker="*", s=120, color="#d79a00", label="目标")
    axes[0].legend(loc="lower center", fontsize=9, ncol=2)
    axes[1].set(
        xlim=(4.4, 6.3),
        ylim=(4.7, 6.6),
        title=f"停止时车身 · 实际最小间隙 {100 * row['min_clearance_m']:.2f} cm",
    )
    footprint(axes[1], truth[-1], 0, COLORS[2], linewidth=2, label="实际车身")
    footprint(axes[1], truth[-1], 0.06, "#d97706", linewidth=1.4, linestyle="--", label="加6cm余量")
    axes[1].legend(
        handles=axes[1].get_legend_handles_labels()[0][-2:], loc="upper left", fontsize=9
    )
    save(fig, "lesson69-trajectories.png")


def goals_chart(ax, summary, lesson):
    for i, method in enumerate(METHODS[lesson]):
        count = [
            sum(
                r["reached"]
                for r in summary["rows"]
                if r["method"] == method and r["map_kind"] == kind
            )
            for kind in ("repaired", "reference")
        ]
        width = 0.23 if lesson == "70" else 0.3
        color = COLORS[i] if lesson == "70" else COLORS[2 * i]
        bars = ax.bar(
            np.arange(2) + (i - (len(METHODS[lesson]) - 1) / 2) * width,
            count,
            width * 0.93,
            color=color,
            label=NAMES[lesson][i],
        )
        ax.bar_label(bars, labels=[f"{v}/24" for v in count], padding=3)
    ax.set(
        xticks=[0, 1],
        xticklabels=["扫描修后图", "准确参照图"],
        ylim=(0, 29),
        ylabel="真正到达的巡检点",
        title="每种地图3轮 × 每轮8个点",
    )
    ax.legend(loc="upper center", ncol=len(METHODS[lesson]), fontsize=9)
    decorate(ax)


def lesson70(directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), layout="constrained")
    goals_chart(axes[0], summary, "70")
    for i, method in enumerate(METHODS["70"]):
        distances = [
            e["true_distance_m"]
            for r in summary["rows"]
            if r["method"] == method
            for e in r["events"]
        ]
        axes[1].scatter(
            np.full(len(distances), i) + np.linspace(-0.12, 0.12, len(distances)),
            distances,
            color=COLORS[i],
            s=21,
            alpha=0.7,
        )
    axes[1].axhline(0.45, color="#d97706", ls="--", label="真实到点上限 0.45m")
    axes[1].set(
        xticks=[0, 1, 2],
        xticklabels=NAMES["70"],
        ylabel="宣布到点时的真实距离 (m)",
        ylim=(0, 0.56),
        title="每点为一次宣布；未宣布的目标不画点",
    )
    axes[1].legend(loc="upper right", fontsize=9)
    decorate(axes[1])
    save(fig, "lesson70-results.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), layout="constrained")
    goal = np.array(TASKS[3][1])
    axes[0].add_patch(
        Circle(goal, 0.45, fill=False, ls="--", color="#d97706", label="真实45cm验收圈")
    )
    axes[0].scatter(*goal, color="#d79a00", marker="*", s=130)
    actuals, estimates = [], []
    for i, method in enumerate(METHODS["70"]):
        row, a = record(directory, summary, method, map_kind="repaired", seed=1)
        event = row["events"][3]
        f = event["frame"]
        trajectory = a["truth"][max(0, f - 24) : f + 1]
        axes[0].plot(
            trajectory[:, 0], trajectory[:, 1], color=COLORS[i], label=NAMES["70"][i], lw=1.7
        )
        axes[0].scatter(*a["truth"][f, :2], color=COLORS[i], s=50, marker="o")
        actuals.append(event["true_distance_m"])
        estimates.append(a["arrival_distance"][f])
    axes[0].set(
        aspect="equal",
        xlim=(goal[0] - 1, goal[0] + 1),
        ylim=(goal[1] - 0.85, goal[1] + 1.15),
        xlabel="x (m)",
        ylabel="y (m)",
        title="修后图 / 种子1 / P4：最后2.9秒的实际运动",
    )
    axes[0].legend(loc="lower left", fontsize=9)
    for offset, values, color, label in (
        (-0.17, estimates, "#64748b", "估计距离"),
        (0.17, actuals, "#119c74", "真实距离"),
    ):
        bar = axes[1].bar(np.arange(3) + offset, values, 0.30, color=color, label=label)
        axes[1].bar_label(bar, fmt="%.3f", padding=3)
    axes[1].axhline(0.45, color="#d97706", ls="--")
    axes[1].set(
        xticks=[0, 1, 2],
        xticklabels=NAMES["70"],
        ylim=(0, 0.60),
        ylabel="宣布到点时的距离 (m)",
        title="估计靠近，不代表身体已经靠近",
    )
    axes[1].legend(loc="upper right", fontsize=9)
    decorate(axes[1])
    save(fig, "lesson70-trajectories.png")


def lesson71(directory, summary):
    row, a = record(directory, summary, "recovery", map_kind="reference", seed=0)
    _, b = record(directory, summary, "baseline", map_kind="reference", seed=0)
    event = row["recoveries"][0]
    start, end = event["start_frame"], event["end_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), layout="constrained")
    goals_chart(axes[0], summary, "71")
    maps = read_checked(directory / "maps.npz", summary["maps_sha256"])
    axes[1].imshow(
        maps["reference"],
        origin="lower",
        extent=(0, 24, 0, 24),
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.35,
    )
    for key, color, style, label in (
        ("truth", "#152b43", "-", "实际中心"),
        ("estimate", "#119c74", "--", "估计中心"),
    ):
        points = a[key][start : end + 1]
        axes[1].plot(points[:, 0], points[:, 1], style, color=color, lw=2, label=label)
        axes[1].scatter(*points[0, :2], color=color, marker="o", s=35)
        axes[1].scatter(*points[-1, :2], color=color, marker="^", s=55)
    for f, ls in ((start, "--"), (end, "-")):
        axes[1].add_patch(
            Circle(a["truth"][f, :2], 0.25, fill=False, color="#152b43", ls=ls, lw=1.2)
        )
    x, y = a["truth"][start, :2]
    axes[1].set(
        xlim=(x - 0.55, x + 0.55),
        ylim=(y - 0.4, y + 0.7),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="真实低速移动23.4cm后恢复路径",
    )
    axes[1].legend(loc="upper left", fontsize=9)
    save(fig, "lesson71-results.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), layout="constrained")
    axes[0].imshow(
        maps["reference"], origin="lower", extent=(0, 24, 0, 24), cmap="Greys", alpha=0.4
    )
    for data, color, label in (
        (a, COLORS[2], "恢复组：8/8真实到点"),
        (b, COLORS[0], "原组：2/8后终止"),
    ):
        axes[0].plot(data["truth"][:, 0], data["truth"][:, 1], color=color, lw=1.8, label=label)
    for i, (_, goal) in enumerate(TASKS):
        axes[0].scatter(*goal, color="#d79a00", marker="*", s=55)
        axes[0].annotate(f"P{i + 1}", goal, xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[0].set(
        xlim=(0, 24),
        ylim=(0, 24),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="准确参照图 / 种子0 · 实际轨迹",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=10
    )
    idx = np.arange(start - 10, end + 16)
    t = idx * 0.12
    axes[1].plot(
        t, a["commands"][idx, 0], color=COLORS[2], drawstyle="steps-post", label="本帧线速度 (m/s)"
    )
    axes[1].plot(t, a["arrival_r95"][idx], color="#8b5cf6", label="粒子95%半径 (m)")
    axes[1].axvspan(start * 0.12, end * 0.12, alpha=0.14, color="#eebb44", label="13帧恢复：1.56秒")
    axes[1].set(
        xlabel="时间 (s)", ylabel="数值（单位见图例）", title="先慢行；规划恢复后才回到普通速度"
    )
    axes[1].legend(loc="center right", fontsize=9)
    decorate(axes[1])
    save(fig, "lesson71-trajectories.png")


def arrival_stop_audit():
    directory, summary = read("70")
    result = {}
    for method in METHODS["70"]:
        speeds = []
        for row in summary["rows"]:
            if row["method"] == method:
                data = read_checked(directory / row["file"], row["sha256"])
                speeds.extend(data["commands"][max(0, e["frame"] - 1)] for e in row["events"])
        values = np.abs(speeds)
        result[method] = {
            "declarations": len(values),
            "max_previous_v_m_s": float(values[:, 0].max()),
            "max_previous_w_rad_s": float(values[:, 1].max()),
        }
    return result


def main():
    configure_plot_font()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10})
    summaries = {}
    for lesson, function in (("69", lesson69), ("70", lesson70), ("71", lesson71)):
        directory, summary = read(lesson)
        function(directory, summary)
        summaries[lesson] = {
            "directory": str(directory.relative_to(ROOT)),
            "summary_sha256": digest(directory / "summary.json"),
            **summary,
        }
    target = ROOT / "docs/benchmarks/navigation-studies-69-71-v1.json"
    directory, summary = read("69")
    _, record69 = record(directory, summary, "rectangle", layout="narrow", obstacle="crate", seed=0)
    pose = record69["truth"][-1]
    path = record69["plan"][0]
    path = path[np.isfinite(path).all(axis=1)]
    nearest = int(np.argmin(np.linalg.norm(path - pose[:2], axis=1)))
    tangent = np.arctan2(*(path[nearest + 1] - path[nearest])[::-1])
    rig = NavigationRig()
    diagnostic = {
        "case": "narrow/crate/seed0",
        "end_pose": pose.tolist(),
        "nearest_initial_path_sample_m": float(np.linalg.norm(path[nearest] - pose[:2])),
        "yaw_minus_forward_chord_deg": float(np.rad2deg(pose[2] - tangent)),
        "margin_projected_to_wall_m": float(
            rig.margin_m * (abs(np.cos(pose[2])) + abs(np.sin(pose[2])))
        ),
        "prior_only_rejects_end_pose": not FootprintField(record69["prior_map"], [], rig).free(
            pose
        ),
    }
    target.write_text(
        json.dumps(
            {
                "schema": 1,
                "selection": "69 narrow/crate/seed0; 70 repaired/seed1/P4; 71 reference/seed0; failures selected before formal batch",
                "figure_script_sha256": digest(Path(__file__)),
                "narrow_diagnostic": diagnostic,
                "arrival_stop_audit": arrival_stop_audit(),
                "studies": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    print("Generated 6 figures and checked benchmark summary.")


if __name__ == "__main__":
    main()
