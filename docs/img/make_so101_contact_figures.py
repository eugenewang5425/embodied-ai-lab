"""Round-three figures and paired diagnostics from immutable physical records."""

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
NORMAL = ("baseline", "surface", "gate", "surface_gate")
SHORT = dict(zip(NORMAL, ("A 指尖", "B 指面", "C 门控", "D 指面＋门控"), strict=True))


def paired(record, reference, candidate):
    values = {k: [] for k in ("both_success", "rescued", "regressed", "both_failure")}
    for position in record.positions:
        a = record.rows[position + "_" + reference]["success"]
        b = record.rows[position + "_" + candidate]["success"]
        key = (
            "both_success" if a and b else "rescued" if b else "regressed" if a else "both_failure"
        )
        values[key].append(position)
    return values


def main():
    configure_plot_font()
    record = ManipulationRecord(ROOT / "results/so101_contact_v4")
    counts = {}
    for method in record.methods:
        rows = [r for r in record.rows.values() if r["method"] == method]
        counts[method] = {
            "total": len(rows),
            "success": sum(r["success"] for r in rows),
            "refused": sum(r["contact_refused"] for r in rows),
            "physical_failure": sum(not r["success"] and not r["contact_refused"] for r in rows),
        }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.3), layout="constrained")
    x = np.arange(4)
    for field, color, label in (
        ("success", "#95cfb7", "完成抓取和放置"),
        ("refused", "#c8b8e5", "门控拒绝（失败）"),
        ("physical_failure", "#efbc9b", "继续执行但失败"),
    ):
        bottom = np.array(
            [
                counts[m]["success"]
                if field == "refused"
                else counts[m]["success"] + counts[m]["refused"]
                if field == "physical_failure"
                else 0
                for m in NORMAL
            ]
        )
        height = np.array([counts[m][field] for m in NORMAL])
        axes[0].bar(x, height, bottom=bottom, color=color, label=label, width=0.65)
        for i, (h, b) in enumerate(zip(height, bottom, strict=True)):
            if h:
                axes[0].text(i, b + h / 2, str(h), ha="center", va="center", fontsize=12)
    axes[0].set(
        xticks=x,
        xticklabels=list(SHORT.values()),
        ylim=(0, 30),
        ylabel="回合数 / 各27",
        title="相同27个新条件：校准有收益，门控不等于抓稳",
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
    grid = np.array(
        [
            [counts[m]["success"] for m in ("baseline", "surface")],
            [counts[m]["success"] for m in ("gate", "surface_gate")],
        ]
    )
    axes[1].imshow(grid, cmap="Greens", vmin=0, vmax=27)
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j,
                i,
                f"{grid[i, j]}/27",
                ha="center",
                va="center",
                fontsize=25,
                color="white" if grid[i, j] >= 20 else "#1e293b",
            )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["旧指尖参考", "新指面参考"],
        yticks=[0, 1],
        yticklabels=["不门控", "先门控"],
        title="两项改动分开：2×2对照",
    )
    figure.suptitle("原模型、限矩、物体和评分不变；空夹 / 偏5.5cm 两种负对照各0/9", fontsize=13)
    figure.savefig(DEST / "lesson73-round3-results.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for axis, method in zip(axes.flat, NORMAL, strict=True):
        grid = np.array(
            [
                [
                    2
                    if record.rows[f"p{i:02}a{k}_{method}"]["contact_refused"]
                    else int(record.rows[f"p{i:02}a{k}_{method}"]["success"])
                    for i in range(9)
                ]
                for k in range(3)
            ]
        )
        axis.imshow(
            grid,
            cmap=ListedColormap(["#efbc9b", "#95cfb7", "#c8b8e5"]),
            vmin=0,
            vmax=2,
            aspect="auto",
        )
        for k in range(3):
            for i in range(9):
                row = record.rows[f"p{i:02}a{k}_{method}"]
                axis.text(
                    i,
                    k,
                    "完成" if row["success"] else "拒绝" if row["contact_refused"] else "失败",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        axis.set_xticks(
            range(9),
            [f"{100 * x:.1f}\n{100 * y:+.1f}" for x, y in record.summary["positions_xy_m"]],
            fontsize=9,
        )
        axis.set_yticks(range(3), ["−20°", "+5°", "+20°"])
        axis.set(
            xlabel="初始位置 / cm（上X、下Y）",
            ylabel="方块初始朝向",
            title=f"{SHORT[method]}：{counts[method]['success']}/27",
        )
    figure.suptitle("同一个格子是相同初态；三例剩余失败均在Y=−3.5cm、+20°", fontsize=14)
    figure.savefig(DEST / "lesson73-round3-grid.png", dpi=160)
    plt.close(figure)

    # Examples are selected by a declared rule: lexicographically first rescue,
    # first remaining B failure and first B-success/D-refusal mismatch.
    rescued = paired(record, "baseline", "surface")["rescued"][0]
    failed = next(p for p in record.positions if not record.rows[p + "_surface"]["success"])
    mismatch = next(
        p
        for p in record.positions
        if record.rows[p + "_surface"]["success"]
        and record.rows[p + "_surface_gate"]["contact_refused"]
    )
    examples = (
        (rescued, "首个被救回：指面几何改善接触", ("baseline", "surface")),
        (failed, "首个仍失败：两指有力却中途掉落", ("baseline", "surface")),
        (mismatch, "首个门控误拒：原本能完成", ("surface", "surface_gate")),
    )
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
    for i, (position, heading, methods) in enumerate(examples):
        for method in methods:
            _, a = record.select(position, method)
            base = a["object_xyz"][np.flatnonzero(a["phase"] == 0)[-1], 2]
            axes[i, 0].plot(
                a["timestamps"],
                (a["object_xyz"][:, 2] - base) * 100,
                color=record.colors[method],
                label=SHORT[method],
                lw=2,
            )
            axes[i, 1].plot(
                a["object_xyz"][:, 0] * 100,
                a["object_xyz"][:, 1] * 100,
                color=record.colors[method],
                label=SHORT[method],
                lw=2,
            )
        _, a = record.select(position, "surface")
        for j, label, color in (
            (0, "固定指", "#2563eb"),
            (1, "活动指", "#c2410c"),
            (2, "桌面等支持", "#64748b"),
        ):
            axes[i, 2].plot(
                a["physics_time"], a["physics_jaw_n"][:, j], label=label, color=color, lw=1
            )
        axes[i, 0].axhline(10, color="#475569", ls="--")
        axes[i, 0].axvspan(10, 11.2, color="#e2e8f0", alpha=0.5)
        axes[i, 0].set(
            xlabel="仿真时间 / s", ylabel="物体抬升 / cm", title=f"{position} · {heading}"
        )
        axes[i, 1].scatter(18, -9, marker="x", color="#111827", label="放置目标")
        axes[i, 1].set(xlabel="世界X / cm", ylabel="世界Y / cm", title="自由方块的真实XY轨迹")
        axes[i, 1].set_aspect("equal", adjustable="datalim")
        axes[i, 2].axvline(8, color="#111827", ls="--", lw=1)
        axes[i, 2].set(
            xlabel="仿真时间 / s",
            ylabel="500Hz法向力 / N",
            xlim=(5, 10),
            title="B组求解力；8s虚线开始抬升",
        )
        for axis in axes[i]:
            axis.legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9, frameon=False
            )
            axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(DEST / "lesson73-round3-traces.png", dpi=160)
    plt.close(figure)
    diagnosis = {
        "archive_sha256": record.summary["archive_sha256"],
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts,
        "A_B": paired(record, "baseline", "surface"),
        "A_C": paired(record, "baseline", "gate"),
        "B_D": paired(record, "surface", "surface_gate"),
        "examples": {"rescued": rescued, "remaining_failure": failed, "gate_mismatch": mismatch},
        "surface_failures": [],
        "calibration_range_at_clamp_end": [],
        "development": {},
    }
    for p in record.positions:
        row, a = record.select(p, "surface")
        clamp = np.flatnonzero(a["phase"] == 3)
        op = a["qpos"][clamp - 1, 5]  # Opening available before each request, not its result.
        cap = record.summary["calibration"]["preparation_opening_rad"]
        diagnosis["calibration_range_at_clamp_end"].append(
            {
                "position": p,
                "success": row["success"],
                "last_request_opening_rad": float(op[-1]),
                "clamp_time_above_cap_s": float(np.count_nonzero(op > cap) * 0.02),
                "preparation_approximation_at_end": bool(op[-1] > cap),
            }
        )
        if row["success"]:
            continue
        end = np.flatnonzero(a["phase"] == 3)[-1]
        start = (end - 1) * 10
        diagnosis["surface_failures"].append(
            {
                "key": row["key"],
                "xyz": row["config"]["object_xyz"],
                "yaw": row["object_yaw_deg"],
                "max_lift_m": row["max_lift_m"],
                "prelift_gate_elapsed_s": float(a["gate_elapsed_s"][end]),
                "prelift_min_forces_n": a["physics_jaw_n"][start : start + 10].min(axis=0).tolist(),
                "peak_jaw_normal_n": row["peak_jaw_normal_n"],
            }
        )
    for name in ("so101_contact_dev_v1", "so101_contact_dev_v2"):
        dev = ManipulationRecord(ROOT / "results" / name)
        diagnosis["development"][name] = {
            "archive_sha256": dev.summary["archive_sha256"],
            "source_sha256": dev.summary["source_sha256"],
            "counts": {
                m: {
                    "success": sum(r["success"] for r in dev.rows.values() if r["method"] == m),
                    "total": 8,
                }
                for m in NORMAL
            },
        }
    diagnosis["section_probe"] = json.loads(
        (ROOT / "results/so101_contact_section_probe/summary.json").read_text(encoding="utf-8")
    )
    diagnosis["section_probe_provenance"] = {
        "source_snapshot_captured": False,
        "file_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "results/so101_contact_section_probe").glob("*"))
            if p.is_file()
        },
        "note": "十次截面探针为开发诊断，保存参数、指标和数组；未保存独立源码快照，不冒充正式独立验证。两批完整开发及正式采集另有源快照。",
    }
    (ROOT / "docs/benchmarks/so101-contact-v4-diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Three figures and complete paired diagnostics saved")


if __name__ == "__main__":
    main()
