"""Lesson 21 / 21a summary figures, recomputed read-only from official records.

Reads an existing recording directory (summary.json + trajectories.npz),
recomputes every plotted number from the trajectories and trials, and writes a
single static PNG. This module never writes into results/; the only output is
the requested image path (intended for docs/img).

Examples:
    uv run python -m embodied_learning.experiments.goal_figures ^
        --source results/goal_reaching_2026-09-03 ^
        --output docs/img/lesson-21-goal-reaching.png
    uv run python -m embodied_learning.experiments.goal_figures ^
        --source results/goal_thresholds_2026-09-03 ^
        --output docs/img/lesson-21a-thresholds.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from embodied_learning.experiments.goal_reaching import CASES, METHODS, load_recording
from embodied_learning.experiments.goal_thresholds import VARIANTS, load_thresholds
from embodied_learning.plotting import configure_plot_font

FIG_NOTE = "由正式记录重算生成（只读），生成命令见讲义"
SERIES_COLORS = {
    ("near", "odom"): "#7c3aed",
    ("near", "fused"): "#f59e0b",
    ("far", "odom"): "#2563eb",
    ("far", "fused"): "#16a34a",
}
SERIES_LABELS = {
    ("near", "odom"): "近目标·纯里程计",
    ("near", "fused"): "近目标·融合",
    ("far", "odom"): "远目标·纯里程计",
    ("far", "fused"): "远目标·融合",
}


def compare_counts(report, records):
    """Recompute near/far x odom/fused outcomes from trials; cross-check summary."""
    table = {}
    for case, _, _ in CASES:
        for method, _, _ in METHODS:
            trials = [records[(case, run, method)][1] for run in range(report["runs"])]
            table[(case, method)] = {
                "true_success_count": sum(t["true_success"] for t in trials),
                "false_arrival_count": sum(t["false_arrival"] for t in trials),
                "timeout_count": sum(not t["controller_arrived"] for t in trials),
            }
    for comparison in report["comparisons"]:
        for method, _, _ in METHODS:
            stats = comparison["methods"][method]
            recomputed = table[(comparison["case"], method)]
            if any(stats[key] != value for key, value in recomputed.items()):
                raise ValueError("Recomputed counts differ from summary comparisons")
    return table


def select_example_trials(records, runs):
    """Pick one successful near run and the worst false-arrival far run (both odom)."""
    near_runs = [run for run in range(runs) if records[("near", run, "odom")][1]["true_success"]]
    far_runs = [run for run in range(runs) if records[("far", run, "odom")][1]["false_arrival"]]
    if not near_runs or not far_runs:
        raise ValueError("Recording lacks the expected near-success / far-false examples")
    near = ("near", near_runs[0], "odom")
    far = (
        "far",
        max(far_runs, key=lambda r: records[("far", r, "odom")][1]["true_final_distance_m"]),
        "odom",
    )
    return near, far


def threshold_metrics(report, records):
    """Recompute per-variant error/duration/timeout from trials; cross-check rows."""
    metrics = {}
    for variant, _, _, _ in VARIANTS:
        for case, _, _ in CASES:
            for method, _, _ in METHODS:
                trials = [records[(variant, case, run, method)][1] for run in range(report["runs"])]
                metrics[(variant, case, method)] = {
                    "mean_error_cm": 100.0
                    * float(np.mean([t["true_final_distance_m"] for t in trials])),
                    "mean_duration_s": float(np.mean([t["duration_s"] for t in trials])),
                    "timeout_count": sum(not t["controller_arrived"] for t in trials),
                }
    indexed = {tuple(row[k] for k in ("variant", "case", "method")): row for row in report["rows"]}
    if set(indexed) != set(metrics):
        raise ValueError("Summary rows do not cover every variant/case/method")
    for key, values in metrics.items():
        row = indexed[key]
        if (
            not np.isclose(100.0 * row["mean_true_final_distance_m"], values["mean_error_cm"])
            or not np.isclose(row["mean_duration_s"], values["mean_duration_s"])
            or row["timeout_count"] != values["timeout_count"]
        ):
            raise ValueError("Recomputed threshold metrics differ from summary rows")
    return metrics


def _draw_trial_content(axis, arrays, trial, acceptance_radius_m, *, legend):
    """Draw paths, goal zone and stop markers; labels only when legend is set."""
    truth, estimated, goal = arrays["truth"], arrays["estimated"], np.asarray(trial["goal"])
    names = ("目标验收区（真值半径 3 cm）", "真值轨迹", "估计轨迹", "起点", "真实停车点", "目标点")
    labels = names if legend else ("_nolegend_",) * len(names)
    axis.add_patch(
        Circle(
            goal,
            acceptance_radius_m,
            facecolor="#16a34a",
            alpha=0.18,
            edgecolor="#16a34a",
            ls="--",
            lw=1.2,
            label=labels[0],
        )
    )
    axis.plot(truth[:, 0], truth[:, 1], color="#1d4ed8", lw=1.6, label=labels[1])
    axis.plot(estimated[:, 0], estimated[:, 1], color="#94a3b8", lw=0.9, ls="--", label=labels[2])
    axis.plot(*truth[0, :2], "s", color="#0f172a", ms=5, label=labels[3])
    axis.plot(*truth[-1, :2], "x", color="#dc2626", ms=9, mew=2.4, label=labels[4])
    axis.plot(*goal, "*", color="#16a34a", ms=13, label=labels[5])


def draw_trial_axis(axis, arrays, trial, acceptance_radius_m, title):
    """Top view of one trial with a zoomed inset on the goal acceptance zone."""
    _draw_trial_content(axis, arrays, trial, acceptance_radius_m, legend=True)
    axis.set_aspect("equal")
    axis.set_title(
        f"{title}\n真实终点误差 {100.0 * trial['true_final_distance_m']:.1f} cm", fontsize=9
    )
    axis.tick_params(labelsize=7)
    axis.grid(True, lw=0.3, alpha=0.4)
    goal = np.asarray(trial["goal"])
    window = max(0.09, 1.6 * trial["true_final_distance_m"] + 0.06)
    inset = axis.inset_axes([0.03, 0.58, 0.44, 0.40])
    _draw_trial_content(inset, arrays, trial, acceptance_radius_m, legend=False)
    inset.set_xlim(goal[0] - window, goal[0] + window)
    inset.set_ylim(goal[1] - window, goal[1] + window)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("目标区放大", fontsize=6, pad=2)
    axis.indicate_inset_zoom(inset, edgecolor="#475569", alpha=0.7, lw=0.8)


def draw_goal_figure(report, records, output, source):
    """Lesson 21: outcome bars plus one near-success and one false-arrival trajectory."""
    counts = compare_counts(report, records)
    acceptance = report["controller"]["true_acceptance_radius_m"]
    figure = plt.figure(figsize=(8.8, 5.6), dpi=100, layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1, 1])
    bar_axis = figure.add_subplot(grid[:, 0])
    positions = np.arange(4)
    success = [
        counts[key]["true_success_count"]
        for key in (("near", "odom"), ("near", "fused"), ("far", "odom"), ("far", "fused"))
    ]
    false = [
        counts[key]["false_arrival_count"]
        for key in (("near", "odom"), ("near", "fused"), ("far", "odom"), ("far", "fused"))
    ]
    bar_axis.bar(
        positions - 0.19,
        success,
        width=0.38,
        color="#16a34a",
        label="实际通过（真值 3 cm 内）",
    )
    bar_axis.bar(
        positions + 0.19,
        false,
        width=0.38,
        color="#dc2626",
        label="误判到达（估计过门限但真值停偏）",
    )
    for position, (hit, miss) in enumerate(zip(success, false)):
        bar_axis.text(position - 0.19, hit + 0.4, str(hit), ha="center", fontsize=9)
        bar_axis.text(position + 0.19, miss + 0.4, str(miss), ha="center", fontsize=9)
    bar_axis.set_xticks(
        positions,
        ["近目标\n纯里程计", "近目标\n融合", "远目标\n纯里程计", "远目标\n融合"],
        fontsize=8,
    )
    bar_axis.set_ylim(0, 24)
    bar_axis.set_ylabel("回合数（每格 20 回合）")
    bar_axis.set_title("实际到达对照：近目标 16/20→20/20，远目标 5/20→11/20", fontsize=10)
    bar_axis.legend(fontsize=7, loc="upper left")
    bar_axis.grid(True, axis="y", lw=0.3, alpha=0.4)

    near_identity, far_identity = select_example_trials(records, report["runs"])
    trajectory_axes = []
    for row, identity, heading in (
        (0, near_identity, "成功例（近目标）"),
        (1, far_identity, "误判例（远目标）"),
    ):
        axis = figure.add_subplot(grid[row, 1])
        arrays, trial = records[identity]
        draw_trial_axis(axis, arrays, trial, acceptance, heading)
        trajectory_axes.append(axis)
    trajectory_axes[0].legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    figure.suptitle(
        "第二十一课：根据估计位置驶向目标——纯里程计 vs 地标融合（每格 20 回合）",
        fontsize=11,
    )
    figure.supxlabel(f"{FIG_NOTE}；来源 {Path(source).as_posix()}", fontsize=6.5, color="#475569")
    _save(figure, output)


def draw_threshold_figure(report, records, output, source):
    """Lesson 21a: 2/1/0.5 cm stop tolerance vs true error, duration and timeouts."""
    metrics = threshold_metrics(report, records)
    variants = [(key, label, color) for key, _, label, color in VARIANTS]
    series = (("near", "odom"), ("near", "fused"), ("far", "odom"), ("far", "fused"))
    figure, axes = plt.subplots(1, 3, figsize=(8.8, 3.6), dpi=100, layout="constrained")
    width = 0.2
    specs = (
        ("mean_error_cm", "平均真实终点误差（cm，越低越好）", "{:.2f}"),
        ("mean_duration_s", "平均耗时（s，含超时回合）", "{:.1f}"),
        ("timeout_count", "超时回合数（每格 20 回合）", "{:.0f}"),
    )
    positions = np.arange(len(variants))
    for axis, (field, title, template) in zip(axes, specs):
        for index, identity in enumerate(series):
            values = [metrics[(key, *identity)][field] for key, _, _ in variants]
            offset = (index - 1.5) * width
            axis.bar(
                positions + offset,
                values,
                width=width,
                color=SERIES_COLORS[identity],
                label=SERIES_LABELS[identity],
            )
            for position, value in zip(positions, values):
                axis.text(
                    position + offset,
                    value,
                    template.format(value),
                    ha="center",
                    va="bottom",
                    fontsize=5.6,
                    rotation=90,
                )
        axis.set_xticks(positions, [f"{label}\n({key})" for key, label, _ in variants], fontsize=8)
        axis.set_title(title, fontsize=9)
        axis.tick_params(labelsize=7)
        axis.grid(True, axis="y", lw=0.3, alpha=0.4)
    axes[0].set_ylim(0, 9.0)
    axes[1].set_ylim(0, 42.0)
    axes[2].set_ylim(0, 17.0)
    axes[2].set_yticks(range(0, 16, 5))
    axes[2].legend(fontsize=6, loc="upper left", framealpha=0.9)
    figure.suptitle(
        "第二十一课补充：停车门限 2/1/0.5 cm 对照（240 回合，只改估计门限，真值验收 3 cm）",
        fontsize=10,
    )
    figure.supxlabel(f"{FIG_NOTE}；来源 {Path(source).as_posix()}", fontsize=6.5, color="#475569")
    _save(figure, output)


def _save(figure, output):
    output = Path(output)
    if "results" in output.parts:
        raise ValueError("Figures must not be written into results/")
    if "docs" not in output.parts or output.suffix != ".png":
        raise ValueError("Output must be a PNG under docs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100)
    plt.close(figure)
    print(f"Saved: {output} ({output.stat().st_size} bytes)")


def build_figure(source, output):
    """Detect the recording kind from summary.json and draw the matching figure."""
    source = Path(source)
    report = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    kind = report.get("experiment")
    if kind == "estimated_pose_goal_feedback":
        validated, records = load_recording(source)
        draw_goal_figure(validated, records, output, source)
    elif kind == "estimated_stopping_tolerance_comparison":
        validated, records = load_thresholds(source)
        draw_threshold_figure(validated, records, output, source)
    else:
        raise ValueError(f"Unsupported recording kind: {kind}")
    return kind


def main():
    configure_plot_font()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Read-only recording directory")
    parser.add_argument("--output", type=Path, required=True, help="PNG path, e.g. docs/img/x.png")
    args = parser.parse_args()
    kind = build_figure(args.source, args.output)
    print(f"Recording kind: {kind}")


if __name__ == "__main__":
    main()
