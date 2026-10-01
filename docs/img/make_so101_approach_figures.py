"""Reproducible round-two figures and paired failure diagnosis from signed records."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.plotting import configure_plot_font

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).parent


def main():
    configure_plot_font()
    record = ManipulationRecord(ROOT / "results/so101_approach_v3b")
    methods = ("joint", "center_cartesian")
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for axis, method, heading in zip(
        axes, methods, ("原策略：17/27", "空间路径＋开口中心：21/27"), strict=True
    ):
        grid = np.array(
            [
                [int(record.rows[f"p{i:02}a{k}_{method}"]["success"]) for i in range(9)]
                for k in range(3)
            ]
        )
        axis.imshow(
            grid, cmap=ListedColormap(["#f3c6b6", "#a8d8c7"]), vmin=0, vmax=1, aspect="auto"
        )
        for k in range(3):
            for i in range(9):
                axis.text(
                    i, k, "成功" if grid[k, i] else "失败", ha="center", va="center", fontsize=10
                )
        positions = record.summary["positions_xy_m"]
        axis.set_xticks(
            range(9), [f"{x * 100:.1f}\n{y * 100:+.1f}" for x, y in positions], fontsize=9
        )
        axis.set_yticks(range(3), ["−15°", "0°", "+15°"])
        axis.set(xlabel="初始位置 / cm（上行X，下行Y）", ylabel="初始方块朝向", title=heading)
    figure.suptitle("相同27个新位置／朝向条件；两种负对照各0/9，完整物理评分不变", fontsize=13)
    figure.savefig(DEST / "lesson73-round2-results.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    examples = (
        ("p05a1", "固定中间X、最大Y、0°：原失败被救回"),
        ("p00a2", "按键名第一个候选失败：原成功变失败"),
    )
    for row_index, (position, heading) in enumerate(examples):
        for method in methods:
            _, a = record.select(position, method)
            base = a["object_xyz"][np.flatnonzero(a["phase"] == 0)[-1], 2]
            color = record.colors[method]
            label = "原策略" if method == "joint" else "空间路径＋开口中心"
            axes[row_index, 0].plot(
                a["timestamps"], (a["object_xyz"][:, 2] - base) * 100, color=color, label=label
            )
            axes[row_index, 1].plot(
                a["object_xyz"][:, 0] * 100, a["object_xyz"][:, 1] * 100, color=color, label=label
            )
        _, a = record.select(position, "center_cartesian")
        axes[row_index, 2].plot(
            a["timestamps"], a["jaw_normal_n"][:, 0], color="#2563eb", label="固定指"
        )
        axes[row_index, 2].plot(
            a["timestamps"], a["jaw_normal_n"][:, 1], color="#c2410c", label="活动指"
        )
        axes[row_index, 0].axhline(10, color="#475569", ls="--")
        axes[row_index, 0].axvspan(10, 11.2, color="#e2e8f0", alpha=0.5)
        axes[row_index, 0].set(xlabel="仿真时间 / s", ylabel="物体抬升 / cm", title=heading)
        axes[row_index, 1].scatter(18, -9, marker="x", color="#111827", label="放置目标")
        axes[row_index, 1].set(xlabel="世界X / cm", ylabel="世界Y / cm", title="物体真实轨迹")
        axes[row_index, 1].set_aspect("equal", adjustable="datalim")
        axes[row_index, 2].set(
            xlabel="仿真时间 / s",
            ylabel="控制帧法向力 / N",
            title="候选两指力：有一侧受力不等于夹牢",
        )
        for axis in axes[row_index]:
            axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, fontsize=9, frameon=False
            )
            axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(DEST / "lesson73-round2-traces.png", dpi=160)
    plt.close(figure)
    diagnosis = {
        "input_summary_sha256": hashlib.sha256(
            (record.directory / "summary.json").read_bytes()
        ).hexdigest(),
        "archive_sha256": record.summary["archive_sha256"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "paired": {},
        "failure_rows": [],
    }
    for position in record.positions:
        base = record.rows[position + "_joint"]
        candidate = record.rows[position + "_center_cartesian"]
        key = (
            "both_success"
            if base["success"] and candidate["success"]
            else "rescued"
            if candidate["success"]
            else "regressed"
            if base["success"]
            else "both_failure"
        )
        diagnosis["paired"].setdefault(key, []).append(position)
        if not candidate["success"]:
            a = record.arrays[candidate["key"]]
            stages = []
            for phase in (2, 3, 4, 5):
                end = np.flatnonzero(a["phase"] == phase)[-1]
                stages.append(
                    {
                        "phase": phase,
                        "end_jaw_force_n": a["jaw_normal_n"][end].tolist(),
                        "end_object_xyz": a["object_xyz"][end].tolist(),
                        "end_opening_rad": float(a["qpos"][end, 5]),
                    }
                )
            diagnosis["failure_rows"].append(
                {
                    "key": candidate["key"],
                    "initial_xyz": candidate["config"]["object_xyz"],
                    "yaw_deg": candidate["object_yaw_deg"],
                    "max_lift_m": candidate["max_lift_m"],
                    "stages": stages,
                }
            )
    with (
        np.load(ROOT / "results/so101_grasping_v2/trajectories.npz") as old,
        np.load(ROOT / "results/so101_approach_development_v3a/p01_joint.npz") as dev,
    ):
        diagnosis["legacy_bridge"] = {
            "arrays": len(dev.files),
            "bitwise_identical": all(
                np.array_equal(dev[k], old["p05_clamp__" + k]) for k in dev.files
            ),
            "note": "同开发位置(22,5.5)cm原策略与首轮全部15数组逐值相同",
        }
    (ROOT / "docs/benchmarks/so101-approach-v3b-diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Figures and paired diagnosis saved")


if __name__ == "__main__":
    main()
