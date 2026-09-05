"""Rebuild lesson-20/21 figure plates read-only from frozen formal records.

This module never writes into results/ and never re-runs an experiment: every
panel is computed from an existing frozen record directory (summary.json,
*.npz, ros_trace.jsonl). Numbers printed on a plate must be traceable to those
files, so annotations are formatted from loaded data, never hand-typed. The
only output is the requested image path (docs/img by default).

Examples (run from the project root):
    uv run python -m embodied_learning.experiments.record_figures --kind ros2
    uv run python -m embodied_learning.experiments.record_figures --kind goal
    uv run python -m embodied_learning.experiments.record_figures --kind threshold
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.experiments.goal_reaching import CASES, METHODS, load_recording
from embodied_learning.experiments.goal_thresholds import VARIANTS, load_thresholds
from embodied_learning.landmark_localization import LANDMARKS
from embodied_learning.plotting import configure_plot_font

OUTPUT_WIDTH_PX = 880
RENDER_DPI = 150

DEFAULT_RECORDS = {
    "ros2": Path("results/ros2_system_2026-09-03_v2"),
    "goal": Path("results/goal_reaching_2026-09-03"),
    "threshold": Path("results/goal_thresholds_2026-09-03"),
}
DEFAULT_OUTPUTS = {
    "ros2": Path("docs/img/lesson-20-ros2-timeline.png"),
    "goal": Path("docs/img/lesson-21-goal-outcomes.png"),
    "threshold": Path("docs/img/lesson-21a-thresholds.png"),
}

COLOR_TRUTH = "#2563eb"
COLOR_ODOM = "#9333ea"
COLOR_FUSED = "#ea580c"
COLOR_PASS = "#16a34a"
COLOR_FALSE = "#dc2626"
COLOR_TIMEOUT = "#6b7280"
COLOR_LANDMARK = "#15803d"

# Per-series annotation offsets, tuned so point labels never overlap each other.
DIST_STYLES = {
    ("near", "odom"): (COLOR_ODOM, "o", "近·只靠轮子", (5, -13), (5, -13)),
    ("near", "fused"): (COLOR_FUSED, "s", "近·轮子＋地标", (5, 8), (5, 8)),
    ("far", "odom"): ("#7c3aed", "^", "远·只靠轮子", (5, 8), (5, 8)),
    ("far", "fused"): (COLOR_TRUTH, "D", "远·轮子＋地标", (5, 8), (5, -13)),
}


def format_sci(value):
    """Short scientific notation used for float-consistency annotations."""
    return f"{value:.2e}"


def cm(value, digits=3):
    """Meters from a record, restated in cm with fixed digits."""
    return f"{value * 100:.{digits}f}"


def ros2_total_counts_text(summary):
    counts = summary["received_counts"]
    return (
        f"收到总数：编码器 {counts['encoders']} · 地标 {counts['landmarks']} · "
        f"里程计位姿 {counts['odom']} · 融合位姿 {counts['fused']}"
    )


def ros2_float_note(summary):
    return (
        "定位输出与第十九课最大分量差 "
        f"{format_sci(summary['reference_max_abs_difference'])}，"
        f"TF 链查询差 {format_sci(summary['tf_chain_max_abs_difference'])}"
        "（浮点舍入量级，不是定位精度）"
    )


def observation_times(trace):
    return [entry["time_s"] for entry in trace if entry["observation"]]


def goal_panel_title(case_label, method_label, stats, runs):
    return (
        f"{case_label} · {method_label}\n"
        f"通过 {stats['true_success_count']}/{runs}，"
        f"误判到达 {stats['false_arrival_count']}/{runs}，"
        f"平均最终距离 {cm(stats['mean_true_final_distance_m'])} cm"
    )


def threshold_bar_label(row):
    return f"{row['true_success_count']}/{row['false_arrival_count']}/{row['timeout_count']}"


def threshold_sample_lines(trials):
    """Outcome lines for one (case, run, method) across the three tolerances.

    ``trials`` maps variant key (cm2/cm1/cm05) to that trial's metrics dict.
    """
    lines = []
    for variant, _, label, _ in VARIANTS:
        trial = trials[variant]
        duration = f"{trial['duration_s']:.2f} s "
        if trial["terminal_reason"] == "timeout":
            lines.append(
                f"{label}：{duration}超时，实际差 {cm(trial['true_final_distance_m'])} cm，"
                f"估计差 {cm(trial['estimated_final_distance_m'])} cm"
            )
        elif trial["false_arrival"]:
            lines.append(
                f"{label}：{duration}宣布到达但停偏，实际差 "
                f"{cm(trial['true_final_distance_m'])} cm"
                f"（估计差 {cm(trial['estimated_final_distance_m'])} cm）"
            )
        else:
            lines.append(f"{label}：{duration}通过，实际差 {cm(trial['true_final_distance_m'])} cm")
    return lines


def load_ros2_record(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("experiment") != "ros2_message_and_tf_bridge":
        raise ValueError(f"Not a lesson-20 recording: {directory}")
    with np.load(directory / "reference.npz", allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    with (directory / "ros_trace.jsonl").open(encoding="utf-8") as stream:
        trace = [json.loads(line) for line in stream]
    if not trace or len(reference["truth"]) != len(trace):
        raise ValueError("Trace frames do not match reference trajectory length")
    return reference, trace, summary


def _save_plate(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    width_in = float(fig.get_size_inches()[0])
    try:
        from PIL import Image
    except ImportError:
        fig.savefig(output, dpi=OUTPUT_WIDTH_PX / width_in)
        plt.close(fig)
        return
    scratch = output.with_name(output.stem + "-full.png")
    try:
        fig.savefig(scratch, dpi=RENDER_DPI)
        with Image.open(scratch) as image:
            height = round(image.height * OUTPUT_WIDTH_PX / image.width)
            image.convert("RGBA").resize((OUTPUT_WIDTH_PX, height), Image.LANCZOS).save(output)
    finally:
        scratch.unlink(missing_ok=True)
    plt.close(fig)


def _legend(ax, loc, **kwargs):
    return ax.legend(fontsize=8, framealpha=0.92, loc=loc, **kwargs)


def render_ros2(record_dir, output):
    reference, trace, summary = load_ros2_record(record_dir)
    configure_plot_font()
    truth, odom, fused = reference["truth"], reference["odom"], reference["fused"]
    times = np.array([entry["time_s"] for entry in trace])
    map_to_odom = np.array([entry["map_to_odom"] for entry in trace])
    counts = [entry["received_counts"] for entry in trace]
    per_frame_synced = all(c["encoders"] == c["odom"] == c["fused"] for c in counts)
    encoders = np.array([c["encoders"] for c in counts])
    landmark_counts = np.array([c["landmarks"] for c in counts])
    obs_times = np.array(observation_times(trace))
    err_odom = np.linalg.norm(odom[:, :2] - truth[:, :2], axis=1) * 100
    err_fused = np.linalg.norm(fused[:, :2] - truth[:, :2], axis=1) * 100

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), layout="constrained")
    (ax_map, ax_timeline), (ax_tf, ax_err) = axes

    ax_map.plot(truth[:, 0], truth[:, 1], color=COLOR_TRUTH, lw=2.6, label="运动学真值", zorder=6)
    ax_map.plot(
        odom[::20, 0],
        odom[::20, 1],
        "o",
        ms=4.5,
        mfc="none",
        color=COLOR_ODOM,
        label="纯里程计估计",
        zorder=3,
    )
    ax_map.plot(
        fused[::20, 0],
        fused[::20, 1],
        "o",
        ms=4.5,
        mfc="none",
        color=COLOR_FUSED,
        label="地标融合估计",
        zorder=4,
    )
    ax_map.plot(
        LANDMARKS[:, 0],
        LANDMARKS[:, 1],
        "^",
        color=COLOR_LANDMARK,
        ms=9,
        ls="none",
        label="已知地标",
        zorder=5,
    )
    ax_map.plot(truth[0, 0], truth[0, 1], "s", color="k", ms=6, zorder=7)
    ax_map.annotate(
        "起点", (truth[0, 0], truth[0, 1]), textcoords="offset points", xytext=(6, 6), fontsize=8
    )
    ax_map.set(
        xlabel="世界 X（m）",
        ylabel="世界 Y（m）",
        title="方形路线 600 步 / 24 s：真值（实线）与两种估计（空圈）",
        aspect="equal",
        xlim=(-0.35, 2.95),
        ylim=(-1.25, 1.95),
    )
    _legend(ax_map, "upper right")

    for t in obs_times:
        ax_timeline.axvline(t, color="#cbd5e1", lw=0.6, zorder=1)
    total = summary["received_counts"]
    if per_frame_synced:
        ax_timeline.plot(
            times,
            encoders,
            color="#334155",
            lw=1.6,
            label=f"编码器＝里程计＝融合（逐帧同步，累计 {total['encoders']} 条）",
        )
    else:
        ax_timeline.plot(times, encoders, color="#334155", lw=1.6, label="编码器累计")
        ax_timeline.plot(times, [c["odom"] for c in counts], lw=1.2, label="里程计位姿累计")
        ax_timeline.plot(times, [c["fused"] for c in counts], lw=1.2, label="融合位姿累计")
    ax_timeline.set(
        xlabel="教学时间 time_s（s）",
        ylabel="位姿／编码器累计条数",
        ylim=(-10, 720),
        title="消息时间线：三话题逐帧同步，地标每 2 s 一条",
    )
    landmark_axis = ax_timeline.twinx()
    landmark_axis.step(
        times,
        landmark_counts,
        where="post",
        color=COLOR_FUSED,
        lw=1.8,
        label=f"地标阶梯（右轴，每 2 s 一条，累计 {total['landmarks']} 条）",
    )
    landmark_axis.set(
        ylabel="地标累计条数（右轴）",
        ylim=(0, 13.5),
        yticks=[0, 2, 4, 6, 8, 10, 12],
    )
    ax_timeline.text(
        0.02,
        0.97,
        ros2_total_counts_text(summary),
        transform=ax_timeline.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "#f8fafc", "ec": "#94a3b8"},
    )
    timeline_handles, timeline_labels = ax_timeline.get_legend_handles_labels()
    landmark_handles, landmark_labels = landmark_axis.get_legend_handles_labels()
    landmark_axis.legend(
        timeline_handles + landmark_handles,
        timeline_labels + landmark_labels,
        fontsize=8,
        framealpha=0.92,
        loc="lower right",
    )

    ax_tf.plot(times, map_to_odom[:, 0] * 100, color=COLOR_TRUTH, lw=1.5, label="map→odom x（cm）")
    ax_tf.plot(times, map_to_odom[:, 1] * 100, color=COLOR_ODOM, lw=1.5, label="map→odom y（cm）")
    ax_tf.set(
        xlabel="教学时间 time_s（s）",
        ylabel="map→odom 平移分量（cm）",
        ylim=(-3.5, 5.5),
        title="map→odom 修正边：只在观测时刻（每 2 s）跳变",
    )
    yaw_axis = ax_tf.twinx()
    yaw_axis.plot(
        times,
        np.degrees(map_to_odom[:, 2]),
        color=COLOR_FUSED,
        lw=1.4,
        ls="--",
        label="map→odom 偏航（°，右轴）",
    )
    yaw_axis.set(ylabel="偏航修正（°）", ylim=(-9, 15), yticks=[-8, -4, 0, 4, 8])
    final_tf = map_to_odom[-1]
    ax_tf.annotate(
        f"24 s：平移 {np.linalg.norm(final_tf[:2]) * 100:.2f} cm，"
        f"偏航 {np.degrees(final_tf[2]):.2f}°",
        xy=(times[-1], final_tf[1] * 100),
        xytext=(-160, 26),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    frame0 = trace[0]["map_to_sensor"]
    ax_tf.text(
        0.02,
        0.97,
        f"第 0 帧 map→sensor 查询 = ({frame0[0]:.2f}, {frame0[1]:.2f}, "
        f"+{np.degrees(frame0[2]):.0f}°)（base_link→sensor 固定安装，tf_static）",
        transform=ax_tf.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "#f8fafc", "ec": "#94a3b8"},
    )
    tf_handles, tf_labels = ax_tf.get_legend_handles_labels()
    yaw_handles, yaw_labels = yaw_axis.get_legend_handles_labels()
    yaw_axis.legend(
        tf_handles + yaw_handles,
        tf_labels + yaw_labels,
        fontsize=8,
        framealpha=0.92,
        loc="lower left",
    )

    ax_err.plot(times, err_odom, color=COLOR_ODOM, lw=1.6, label="纯里程计位置误差")
    ax_err.plot(times, err_fused, color=COLOR_FUSED, lw=1.6, label="地标融合位置误差")
    for t in obs_times:
        ax_err.axvline(t, color="#cbd5e1", lw=0.6, zorder=1)
    ax_err.set(
        xlabel="教学时间 time_s（s）",
        ylabel="相对真值的位置误差（cm）",
        ylim=(0, 4.3),
        title=f"逐帧定位误差：最大 {err_odom.max():.2f} cm（里程计）/ "
        f"{err_fused.max():.2f} cm（融合）",
    )
    ax_err.text(
        0.02,
        0.52,
        ros2_float_note(summary),
        transform=ax_err.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round", "fc": "#fffbeb", "ec": "#f59e0b"},
    )
    _legend(ax_err, "upper left")

    for ax in (ax_timeline, ax_tf, ax_err):
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"第二十课图版：三进程 ROS 2 回放（只读重绘自冻结记录 {Path(record_dir).as_posix()}）",
        fontsize=12,
    )
    _save_plate(fig, output)


def render_goal(record_dir, output):
    report, results = load_recording(record_dir)
    configure_plot_font()
    runs = report["runs"]
    stats_by = {
        (case["case"], method): case["methods"][method]
        for case in report["comparisons"]
        for method, _, _ in METHODS
    }
    method_labels = {key: label for key, label, _ in METHODS}

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.8), layout="constrained")
    panels = [(case, method) for case, _, _ in CASES for method, _, _ in METHODS]
    case_labels = {key: label for key, label, _ in CASES}
    case_goals = {key: np.asarray(goal) for key, _, goal in CASES}
    acceptance = report["controller"]["true_acceptance_radius_m"]
    map_axes = axes[:, :2].ravel()
    for ax, (case, method) in zip(map_axes, panels, strict=True):
        goal = case_goals[case]
        stats = stats_by[(case, method)]
        ax.set_title(
            goal_panel_title(case_labels[case], method_labels[method], stats, runs), fontsize=10
        )
        for run in range(runs):
            arrays, trial = results[(case, run, method)]
            color = COLOR_FALSE if trial["false_arrival"] else COLOR_PASS
            ax.plot(
                arrays["truth"][:, 0],
                arrays["truth"][:, 1],
                color=color,
                alpha=0.45,
                lw=0.9,
                zorder=2,
            )
            end = arrays["truth"][-1, :2]
            ax.plot(end[0], end[1], "o", color=color, ms=5.5, zorder=4)
        ax.plot(goal[0], goal[1], "*", color="k", ms=13, zorder=6)
        ax.add_patch(
            plt.Circle(goal, acceptance, fill=False, color=COLOR_PASS, ls="--", lw=1.4, zorder=6)
        )
        ax.plot(0, 0, "s", color="k", ms=6, zorder=6)
        ax.annotate("起点", (0, 0), textcoords="offset points", xytext=(6, -13), fontsize=8)
        ax.set(
            xlabel="世界 X（m）",
            ylabel="世界 Y（m）",
            aspect="equal",
            xlim=(-0.4, goal[0] + 0.55),
            ylim=(-0.5, goal[1] + 0.6),
        )
        ax.grid(alpha=0.2)
    half = 0.17
    marker_by_method = {"odom": "o", "fused": "s"}
    for zoom_ax, case in ((axes[0, 2], "near"), (axes[1, 2], "far")):
        goal = case_goals[case]
        zoom_ax.set_title(f"{case_labels[case]}终点放大（窗宽 {half * 2:.2f} m）", fontsize=10)
        for method, _, _ in METHODS:
            for run in range(runs):
                arrays, trial = results[(case, run, method)]
                color = COLOR_FALSE if trial["false_arrival"] else COLOR_PASS
                end = arrays["truth"][-1, :2]
                zoom_ax.plot(
                    end[0],
                    end[1],
                    marker_by_method[method],
                    color=color,
                    ms=5,
                    alpha=0.9,
                    zorder=3,
                )
                if run == 0:
                    zoom_ax.plot(
                        end[0],
                        end[1],
                        marker_by_method[method],
                        ms=13,
                        mfc="none",
                        mec="k",
                        mew=1.1,
                        zorder=4,
                    )
                    if method == "odom":
                        zoom_ax.annotate(
                            "样本 #0",
                            end,
                            textcoords="offset points",
                            xytext=(7, -11),
                            fontsize=7.5,
                        )
        zoom_ax.add_patch(
            plt.Circle(goal, acceptance, fill=False, color=COLOR_PASS, ls="--", lw=1.4, zorder=5)
        )
        zoom_ax.plot(goal[0], goal[1], "*", color="k", ms=13, zorder=6)
        zoom_ax.set(
            xlabel="世界 X（m）",
            ylabel="世界 Y（m）",
            aspect="equal",
            xlim=(goal[0] - half, goal[0] + half),
            ylim=(goal[1] - half, goal[1] + half),
        )
        zoom_ax.grid(alpha=0.2)
    fig.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=COLOR_PASS,
                marker="o",
                lw=1.4,
                ls="none",
                label="实际通过（终点在 3 cm 验收圈内）",
            ),
            plt.Line2D(
                [],
                [],
                color=COLOR_FALSE,
                marker="o",
                lw=1.4,
                ls="none",
                label="误判到达（实际停在圈外）",
            ),
            plt.Line2D(
                [], [], color="#475569", marker="o", lw=1.2, ls="none", label="圆形点＝只靠轮子"
            ),
            plt.Line2D(
                [], [], color="#475569", marker="s", lw=1.2, ls="none", label="方形点＝轮子＋地标"
            ),
            plt.Line2D(
                [], [], color="k", marker="*", ls="none", ms=11, label="目标（虚线圈 r=3 cm）"
            ),
        ],
        loc="outside lower center",
        ncol=5,
        fontsize=9,
    )
    longest = max(t["duration_s"] for _, t in results.values())
    fig.suptitle(
        "第二十一课图版：同一控制器只换定位，80/80 宣布到达（0 超时，最长 "
        f"{longest:.2f} s）——只读重绘自 {Path(record_dir).as_posix()}",
        fontsize=12,
    )
    _save_plate(fig, output)


def render_threshold(record_dir, output):
    report, records = load_thresholds(record_dir)
    configure_plot_font()
    rows = {(row["case"], row["method"], row["variant"]): row for row in report["rows"]}
    variant_keys = [variant for variant, _, _, _ in VARIANTS]
    variant_labels = [label for _, _, label, _ in VARIANTS]
    groups = [(case, method) for case, _, _ in CASES for method, _, _ in METHODS]
    group_labels = {key: label for key, label, _ in CASES}
    method_labels = {key: label for key, label, _ in METHODS}

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), layout="constrained")
    (ax_bar, ax_dist), (ax_time, ax_case) = axes

    bar_width = 0.26
    for group_index, (case, method) in enumerate(groups):
        for variant_index, (variant, _, _, _) in enumerate(VARIANTS):
            row = rows[(case, method, variant)]
            x = group_index + (variant_index - 1) * bar_width
            bottom = 0.0
            for count, outcome_color in (
                (row["true_success_count"], COLOR_PASS),
                (row["false_arrival_count"], COLOR_FALSE),
                (row["timeout_count"], COLOR_TIMEOUT),
            ):
                if count:
                    ax_bar.bar(
                        x,
                        count,
                        width=bar_width,
                        bottom=bottom,
                        color=outcome_color,
                        edgecolor="white",
                        lw=0.4,
                    )
                    bottom += count
            ax_bar.annotate(
                threshold_bar_label(row),
                (x, bottom),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    ax_bar.set(
        xticks=range(len(groups)),
        xticklabels=[f"{group_labels[c]}\n{method_labels[m]}" for c, m in groups],
        ylabel="回合数（每格 20）",
        ylim=(0, 26),
        title="通过／停偏／超时计数（格顶标注为 通过/停偏/超时）",
    )
    ax_bar.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=COLOR_PASS, label="实际通过"),
            plt.Rectangle((0, 0), 1, 1, color=COLOR_FALSE, label="误判停偏"),
            plt.Rectangle((0, 0), 1, 1, color=COLOR_TIMEOUT, label="超时"),
        ],
        fontsize=8,
        loc="upper center",
        ncol=3,
    )

    x_positions = np.arange(len(VARIANTS))
    for (case, method), (color, marker, label, offset_first, offset_last) in DIST_STYLES.items():
        values = [rows[(case, method, v)]["mean_true_final_distance_m"] * 100 for v in variant_keys]
        ax_dist.plot(x_positions, values, marker=marker, color=color, lw=1.5, ms=5, label=label)
        for axis_index, value, offset in (
            (0, values[0], offset_first),
            (len(VARIANTS) - 1, values[-1], offset_last),
        ):
            ax_dist.annotate(
                f"{value:.3f}",
                (axis_index, value),
                textcoords="offset points",
                xytext=offset,
                fontsize=7,
                color=color,
                ha="center",
            )
    ax_dist.set(
        xticks=x_positions,
        xticklabels=variant_labels,
        xlim=(-0.3, 2.3),
        xlabel="估计停车门限（真实验收圈固定 3 cm）",
        ylabel="平均最终实际距离（cm）",
        ylim=(-0.6, 8.2),
        title="门限越小停得越近（近目标）；门限修不了里程计漂移（远·轮子）",
    )
    ax_dist.grid(alpha=0.2)
    _legend(ax_dist, "center right")

    for (case, method), (color, marker, label, _, _) in DIST_STYLES.items():
        values = [rows[(case, method, v)]["mean_duration_s"] for v in variant_keys]
        ax_time.plot(x_positions, values, marker=marker, color=color, lw=1.5, ms=5, label=label)
        offsets = {
            ("near", "odom"): (5, -13),
            ("near", "fused"): (5, 8),
            ("far", "odom"): (5, -13),
            ("far", "fused"): (5, 8),
        }
        offset = offsets[(case, method)]
        for axis_index, value in ((0, values[0]), (len(VARIANTS) - 1, values[-1])):
            ax_time.annotate(
                f"{value:.3f}",
                (axis_index, value),
                textcoords="offset points",
                xytext=offset,
                fontsize=7,
                color=color,
                ha="center",
            )
    ax_time.set(
        xticks=x_positions,
        xticklabels=variant_labels,
        xlim=(-0.3, 2.3),
        ylim=(8, 42),
        xlabel="估计停车门限（均值含超时回合的 40 s）",
        ylabel="平均回合时长（s）",
        title="精度用时间购买：远·融合 0.5 cm 组含 13/20 超时",
    )
    ax_time.grid(alpha=0.2)
    _legend(ax_time, "upper left")

    case_key, sample_run, method = "far", 1, "fused"
    goal = next(np.asarray(g) for key, _, g in CASES if key == case_key)
    ax_case.set_title(
        f"{group_labels[case_key]} · {method_labels[method]} · 样本 #{sample_run}："
        "三个门限下的同一配对回合",
        fontsize=10,
    )
    trials = {}
    for variant, _, label, color in VARIANTS:
        arrays, trial = records[(variant, case_key, sample_run, method)]
        trials[variant] = trial
        ax_case.plot(
            arrays["truth"][:, 0],
            arrays["truth"][:, 1],
            color=color,
            lw=1.4,
            label=f"{label} 轨迹",
        )
        end = arrays["truth"][-1, :2]
        if trial["terminal_reason"] == "timeout":
            ax_case.plot(end[0], end[1], "x", color=color, ms=9, mew=2.2, zorder=5)
        else:
            ax_case.plot(end[0], end[1], "o", color=color, ms=7, zorder=5)
    ax_case.plot(goal[0], goal[1], "*", color="k", ms=13, zorder=6)
    ax_case.add_patch(
        plt.Circle(goal, 0.03, fill=False, color=COLOR_PASS, ls="--", lw=1.4, zorder=6)
    )
    ax_case.plot(0, 0, "s", color="k", ms=6, zorder=6)
    ax_case.annotate("起点", (0, 0), textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax_case.set(
        xlabel="世界 X（m）",
        ylabel="世界 Y（m）",
        aspect="equal",
        xlim=(-0.5, 5.5),
        ylim=(-0.75, 1.9),
    )
    ax_case.grid(alpha=0.2)
    inset = ax_case.inset_axes([0.52, 0.06, 0.45, 0.40])
    for variant, _, _, color in VARIANTS:
        arrays, _ = records[(variant, case_key, sample_run, method)]
        inset.plot(arrays["truth"][:, 0], arrays["truth"][:, 1], color=color, lw=1.3)
        inset.plot(arrays["truth"][-1, 0], arrays["truth"][-1, 1], "o", color=color, ms=5)
    inset.add_patch(plt.Circle(goal, 0.03, fill=False, color=COLOR_PASS, ls="--", lw=1.2))
    inset.set(xlim=(4.72, 4.88), ylim=(1.12, 1.28), xticks=[], yticks=[])
    inset.set_title("目标附近放大", fontsize=8)
    ax_case.indicate_inset_zoom(inset, edgecolor="#94a3b8", alpha=0.8)
    ax_case.text(
        0.02,
        0.97,
        "\n".join(threshold_sample_lines(trials)),
        transform=ax_case.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round", "fc": "#f8fafc", "ec": "#94a3b8"},
    )
    _legend(ax_case, "lower left")

    fig.suptitle(
        "第二十一课补充图版：只改估计停车门限，240 回合（只读重绘自 "
        f"{Path(record_dir).as_posix()}）",
        fontsize=12,
    )
    _save_plate(fig, output)


RENDERERS = {
    "ros2": render_ros2,
    "goal": render_goal,
    "threshold": render_threshold,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(RENDERERS), required=True)
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Frozen record directory (default: the formal 2026-09-03 record)",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output PNG path (default: docs/img lesson plate)"
    )
    args = parser.parse_args()
    record_dir = args.record or DEFAULT_RECORDS[args.kind]
    output = args.output or DEFAULT_OUTPUTS[args.kind]
    RENDERERS[args.kind](record_dir, output)
    print(f"Saved: {Path(output).as_posix()}")


if __name__ == "__main__":
    main()
