"""Rebuild lessons 67/68 figures and audit summaries from immutable records."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from embodied_learning.experiments.body_calibration import fixtures
from embodied_learning.experiments.observed_navigation import scene
from embodied_learning.navigation_geometry import NavigationRig, collision_clearance, world_points
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked
from embodied_learning.replay_viewer.navigation_studies import LABELS, LAYOUTS

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
COLORS = {key: info[2] for key, info in LABELS.items()}


def read(directory):
    return json.loads((directory / "summary.json").read_text(encoding="utf8"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid(ax, values, color, label, alpha=1):
    y, x = np.nonzero(values)
    ax.scatter(
        (x + 0.5) * 0.1,
        (y + 0.5) * 0.1,
        s=2,
        color=color,
        alpha=alpha,
        label=label,
        rasterized=True,
    )


def narrow_probe():
    rig = NavigationRig()
    _, boxes = scene("narrow", "crate")
    poses = []
    for x in np.linspace(1.5, 10.5, 1801):
        t = np.clip((x - 1.5) / 3.4, 0, 1)
        y = 6 - 0.7 * (3 * t * t - 2 * t * t * t)
        slope = -0.7 * (6 * t - 6 * t * t) / 3.4 if t < 1 else 0
        poses.append([x, y, np.arctan(slope)])
    values = [collision_clearance(np.array(p), boxes, rig) for p in poses]
    return np.array(poses), {
        "kind": "posthoc_oracle_geometry_diagnostic_not_controller_success",
        "samples": len(poses),
        "min_separating_clearance_m": min(values),
        "gap_m": 0.6,
        "body_width_m": 0.44,
        "planner_inflation_m": 0.5,
        "formula": "x=1.5..10.5, t=clip((x-1.5)/3.4,0,1), y=6-.7*(3*t^2-2*t^3), yaw=atan(dy/dx)",
    }


def calibration(directory, summary):
    rig = NavigationRig()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for ax, name in zip(axes[:2], ("front_wall", "turn_corner"), strict=True):
        a = np.load(directory / f"{name}.npz")
        row = next(r for r in summary["rows"] if r["case"] == name)
        box = next(b for n, b, _ in fixtures() if n == name)
        ax.add_patch(
            Rectangle(
                (box[0], box[2]),
                box[1] - box[0],
                box[3] - box[2],
                color="#cf6262",
                alpha=0.7,
                label="真实障碍",
            )
        )
        footprint = np.array(
            [
                [-rig.rear_m, -rig.half_width_m, 0],
                [rig.front_m, -rig.half_width_m, 0],
                [rig.front_m, rig.half_width_m, 0],
                [-rig.rear_m, rig.half_width_m, 0],
            ]
        )
        for j in np.unique(np.linspace(0, len(a["stop_path"]) - 1, 9).astype(int)):
            pp = world_points(footprint, a["stop_path"][j])
            ax.add_patch(Polygon(pp[:, :2], facecolor="#e7b442", alpha=0.18, edgecolor="#bb810e"))
        ax.add_patch(
            Polygon(footprint[:, :2], fill=False, edgecolor="#253c50", lw=2, label="当前车身")
        )
        ax.plot(a["stop_path"][:, 0], a["stop_path"][:, 1], "k--", label="刹停中心轨迹")
        ax.scatter([0.08, 0.2], [0, 0], c=["#00aaa6", "#8b5cf6"], s=35)
        ax.set(
            xlim=(-0.45, 1),
            ylim=(-0.7, 0.7),
            aspect="equal",
            xlabel="车头方向 x (m)",
            ylabel="车身左侧 y (m)",
        )
        ax.set_title(
            ("直行 0.8 m/s" if name == "front_wall" else "原地转弯 1.2 rad/s")
            + "\n中心没碰到，车身已扫到障碍",
            fontsize=12,
        )
        if name == "front_wall":
            ax.legend(loc="lower right", fontsize=8)
        assert row["oracle_collision"]
    sweep = summary["low_obstacle_visibility"]
    axes[2].plot(
        [r["front_x_m"] for r in sweep], [r["visible_points"] for r in sweep], "o-", color="#8b5cf6"
    )
    axes[2].axvspan(0.4, 0.62, color="#e76a65", alpha=0.18)
    axes[2].set(
        title="14 cm低路障：越近反而看不见",
        xlabel="障碍前沿相对车身原点 x (m)",
        ylabel="深度图可用障碍像素数",
        ylim=(0, 800),
    )
    axes[2].text(0.43, 670, "近场盲区\n必须保留先前观测", fontsize=10)
    save(fig, "body-calibration-geometry.png")

    names = {
        "front_wall": "正前方墙",
        "low_curb": "14 cm低障碍",
        "mast_bar": "支架高度横杆",
        "turn_corner": "转弯角点",
        "side_post": "侧柱 / 余量预警",
        "rear_obstacle": "倒车后方障碍",
        "safe_overhead": "高于全车的横杆",
        "safe_side": "远侧安全障碍",
    }
    columns = [
        ("oracle_collision", "真实会碰撞"),
        ("point_robot_hazard", "只看中心点"),
        ("lidar_body_hazard", "雷达＋车身"),
        ("fused_body_hazard", "当前雷达＋深度"),
        ("with_recent_depth_hazard", "再加入过去深度"),
    ]
    matrix = np.array([[r[k] for k, _ in columns] for r in summary["rows"]], int)
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    ax.imshow(
        matrix,
        cmap=matplotlib.colors.ListedColormap(["#edf1f5", "#edb658"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(5), [v for _, v in columns])
    ax.set_yticks(range(8), [names[r["case"]] for r in summary["rows"]])
    for y in range(8):
        for x in range(5):
            ax.text(
                x,
                y,
                ("碰撞" if x == 0 else "预警") if matrix[y, x] else "否",
                ha="center",
                va="center",
                fontsize=10,
            )
    ax.set_title("预警要和真实碰撞分开：侧柱距车身8 mm，虽未接触仍应触发6 cm余量", fontsize=12)
    save(fig, "body-calibration-detection.png")


def avoidance(directory, summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    methods = ("stop", "lidar", "depth")
    for ax, layout in zip(axes, LAYOUTS, strict=True):
        rr = [r for r in summary["rows"] if r["layout"] == layout]
        bottom = np.zeros(3)
        for status, label, color in [
            ("completed", "到达", "#169c76"),
            ("collision_abort", "碰撞", "#d76558"),
            ("stalled", "停滞", "#e3ba62"),
            ("no_path", "规划拒绝", "#8d91aa"),
        ]:
            count = np.array(
                [sum(r["status"] == status and r["method"] == k for r in rr) for k in methods]
            )
            ax.bar(range(3), count, bottom=bottom, color=color, label=label)
            for i, c in enumerate(count):
                if c:
                    ax.text(
                        i,
                        bottom[i] + c / 2,
                        str(c),
                        ha="center",
                        va="center",
                        color="white",
                        weight="bold",
                    )
            bottom += count
        ax.set_xticks(range(3), ["只停车", "雷达\n＋车身", "雷达＋深度\n＋车身"])
        ax.set(title=LAYOUTS[layout] + " / 每方法9回合", ylabel="回合数", ylim=(0, 10))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    save(fig, "observed-navigation-results.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, layout in zip(axes, ("plaza", "narrow"), strict=True):
        _, boxes = scene(layout, "low")
        for b in boxes:
            ax.add_patch(
                Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], color="#6e7a89", alpha=0.5)
            )
        for k in methods:
            r = next(
                r
                for r in summary["rows"]
                if r["layout"] == layout
                and r["obstacle"] == "low"
                and r["seed"] == 0
                and r["method"] == k
            )
            a = read_checked(directory / r["file"], r["sha256"])
            ax.plot(
                a["truth"][:, 0],
                a["truth"][:, 1],
                color=COLORS[k],
                label=LABELS[k][0],
                lw=2.4 if k == "stop" else 1.5,
                ls="--" if k == "lidar" else "-",
            )
            ax.scatter(*a["truth"][-1, :2], c=COLORS[k], s=45, marker="x")
        ax.scatter(10.5, 6, marker="*", s=110, c="#c99418", label="任务终点")
        ax.set(
            xlim=(1, 11),
            ylim=(4.6, 7.4),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=LAYOUTS[layout] + " / 低障碍 / 种子0",
        )
    poses, diagnostic = narrow_probe()
    axes[1].plot(poses[:, 0], poses[:, 1], color="#267bb2", ls=":", label="事后可通行几何路径")
    axes[1].text(2, 6.65, "规划器拒绝 ≠ 车身一定过不去\n诊断路径最小间隙6.64 cm", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    save(fig, "observed-navigation-trajectories.png")
    return diagnostic


def repair(directory, summary):
    maps = read_checked(directory / "maps.npz", summary["maps_sha256"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
    for ax, key in zip(axes, ("raw", "rigid", "repaired"), strict=True):
        grid(ax, maps["projected"], "#35475a", "真值落点参照", 0.6)
        grid(ax, maps[key], COLORS[key], LABELS[key][0], 0.7)
        q = summary["quality"][key]
        ax.set(
            xlim=(0, 24),
            ylim=(0, 24),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{LABELS[key][0]}\n墙面双向平均距离 {q['symmetric_mean_m']:.3f} m",
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8)
    save(fig, "map-repair-maps.png")
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), layout="constrained")
    for ax, key in zip(axes, ("raw", "rigid", "repaired", "reference"), strict=True):
        r = next(r for r in summary["rows"] if r["seed"] == 1 and r["method"] == key)
        a = read_checked(directory / r["file"], r["sha256"])
        grid(ax, maps[key], "#8493a3", "本组规划地图", 0.35)
        ax.plot(a["truth"][:, 0], a["truth"][:, 1], color="#233446", lw=1.8, label="实际轨迹")
        ax.plot(
            a["estimate"][:, 0], a["estimate"][:, 1], color=COLORS[key], ls="--", label="估计轨迹"
        )
        for event in r["events"]:
            if not event["success"]:
                p = a["truth"][event["frame"], :2]
                ax.scatter(*p, c="#d43f40", marker="x", s=80, label="误报到达")
        ax.set(
            xlim=(0, 24),
            ylim=(0, 24),
            aspect="equal",
            xlabel="x (m)",
            title=f"{LABELS[key][0]}\n种子1：实际到达 {r['reached']}/8",
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            fontsize=7,
        )
    save(fig, "map-repair-navigation.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key, name in [
        ("calibration", "body_calibration_v1"),
        ("avoidance", "observed_navigation_v1"),
        ("repair", "map_repair_v1"),
    ]:
        p.add_argument("--" + key, type=Path, default=ROOT / "results" / name)
    a = p.parse_args()
    configure_plot_font()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    summaries = {k: read(getattr(a, k)) for k in ("calibration", "avoidance", "repair")}
    calibration(a.calibration, summaries["calibration"])
    probe = avoidance(a.avoidance, summaries["avoidance"])
    repair(a.repair, summaries["repair"])
    bench = ROOT / "docs/benchmarks"
    report = {"records": summaries, "narrow_probe": probe, "input_sha256": {}}
    for key in summaries:
        directory = getattr(a, key)
        report["input_sha256"][key] = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(directory.iterdir())
            if f.suffix in (".json", ".npz", ".png", ".py")
        }
    (bench / "navigation-studies-67-68-v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print("6 figures; 81 avoidance episodes; 12 independent repair queries; input hashes saved")


if __name__ == "__main__":
    main()
