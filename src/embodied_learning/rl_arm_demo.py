"""Lesson 40 viewer: pure-learning reaching on the planar 2R arm.

Three static modes share one lesson-40 recording (npz + summary):
1. the training curves: per-update reward (1000-step running mean), the
   automatic-temperature trajectory, the policy entropy and the periodic
   deterministic evaluation on the 20 fixed goals - did the SAC learner
   converge, and when?
2. the trajectory comparison on two showcase goals (the lesson-8 acceptance
   targets A and B): the analytic IK+PD baseline versus the three learned
   policies in the tip plane, plus the paired success-rate bars and the
   per-goal arrival times;
3. arrival quality and failure cases: per-goal final and closest distances
   against the 2 mm gate, one featured failure (or the slowest arrival when
   there is none) drawn with its final arm posture, and the outcome counts.

Layout and Esc handling follow the lesson-28..39 demos (docs/26 section 6):
every mode calls fig.clear() first, redraw draws synchronously, panel lines
stay short (manual breaks, <= 26 chars) and every quoted number is traceable
to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.rl_arm_reaching import (
    ARRIVAL_RADIUS_M,
    EXPERIMENT,
    GOAL_HIGH_M,
    GOAL_LOW_M,
    SHOWCASE_GOALS,
    expected_npz_keys,
)

DEFAULT_RESULTS = "results/rl_arm_reaching_2026-09-06"
SEED_COLORS = ("#0f766e", "#2563eb", "#b45309")
BASELINE_COLOR = "#64748b"


def seconds_text(value):
    """'2.60 s' or '未到达'; NaN (never arrived) counts as not arrived."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "未到达"
    return f"{value:.2f} s"


def load_replays(directory):
    """Validate a lesson-40 recording; returns the data dict for the demo.

    Tamper routes rejected here: (1) the npz SHA-256 against the summary;
    (2) the archive key set against the implied key set; (3) the per-seed and
    baseline success counts recomputed from the archived outcomes against the
    summary; (4) the showcase goals archived in the record against the
    protocol constants (the lesson-8 targets A and B).
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-40 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    goals = np.asarray(data["eval_goals"], dtype=float)
    if (
        goals.shape != (report["hyperparameters"]["eval_goal_count"], 2)
        or not np.allclose(goals[0], SHOWCASE_GOALS[0], atol=1e-12)
        or not np.allclose(goals[1], SHOWCASE_GOALS[1], atol=1e-12)
    ):
        raise ValueError("Archived eval goals disagree with the protocol showcase goals")
    seeds = report["training"]["train_seeds"]
    for seed_index in range(seeds):
        outcomes = np.asarray(data[f"eval_outcome_{seed_index}"])
        derived = int((outcomes == "arrived").sum())
        recorded = report["rl_evaluation"]["per_seed"][seed_index]["aggregate"]["successes"]
        if derived != recorded:
            raise ValueError("Learned successes disagree with the archive")
    baseline_derived = int((np.asarray(data["baseline_outcome"]) == "arrived").sum())
    if baseline_derived != report["baseline"]["aggregate"]["successes"]:
        raise ValueError("Baseline successes disagree with the archive")
    return {"report": report, **data}


def _add_goal_marker(ax, goal, radius_m):
    """Green star at the goal; the 2 mm gate itself is far too small to draw."""
    from matplotlib.patches import Circle

    ax.plot(goal[0], goal[1], marker="*", markersize=13, color="#15803d", linestyle="none")
    ax.add_patch(Circle((goal[0], goal[1]), radius_m, fill=False, color="#15803d", linewidth=1.1))


def _add_arm_scene(ax):
    """Fixed stage shared by every tip map: annulus, goal box, base."""
    from matplotlib.patches import Circle, Rectangle

    ax.add_patch(Circle((0, 0), 0.7, fill=False, linestyle=":", color="#9ca3af", linewidth=1.0))
    ax.add_patch(Circle((0, 0), 0.1, fill=False, linestyle=":", color="#9ca3af", linewidth=1.0))
    ax.add_patch(
        Rectangle(
            (GOAL_LOW_M[0], GOAL_LOW_M[1]),
            GOAL_HIGH_M[0] - GOAL_LOW_M[0],
            GOAL_HIGH_M[1] - GOAL_LOW_M[1],
            fill=False,
            linestyle="--",
            color="#d97706",
            linewidth=1.0,
        )
    )
    ax.plot(0, 0, "s", color="#111827", markersize=6)


def _draw_arm_posture(ax, points_row, color, linewidth=1.6):
    """One arm posture: base -> elbow -> tip polyline with joint dots."""
    ax.plot(points_row[:, 0], points_row[:, 1], "-", color=color, linewidth=linewidth)
    ax.plot(points_row[:, 0], points_row[:, 1], "o", color=color, markersize=3)


class RlArmDemo:
    """Pure-learning arm reaching: training, trajectories, arrival quality."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and labels
        self.root = root
        self.data = data
        self.report = data["report"]
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第四十课 · 2R 机械臂纯学习到达（全驱动 + 固定小目标盒）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "手写 numpy SAC（复用第 39 课组件，动作 = 关节力矩 ±0.25 N·m）："
                "无基线、无示教、无塑形；对照 = 第 8 课解析 IK + PD（正肘解）。"
                "奖励 r = −末端距离，到达 +10（2 mm 且 |dq|<0.1），出关节包络 −10。"
                "三模式共用同一正式记录，全部为静态图"
            ),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线（奖励 + α + 周期评估）", "training"),
            ("② 基线 IK vs 纯学习轨迹对照", "trajectories"),
            ("③ 到达率与失败案例", "arrival"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        self.stats = ttk.Label(middle, width=46, anchor="nw", justify="left", wraplength=400)
        self.stats.pack(side="right", fill="y", padx=(12, 0))
        left = ttk.Frame(middle, width=1030, height=430)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        self.fig = Figure(figsize=(10.4, 4.3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "成功判据：末端距目标 < 2 mm 且 max|dq| < 0.1 rad/s 持续 0.5 s；"
                "出关节包络 |q| > π 终止；成功率对照同 20 个固定目标（配对），"
                "预注册学习判据 = 每种子 ≥ 18/20"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """1 reward + alpha + entropy + periodic evaluation per seed."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_alpha, ax_entropy, ax_eval = self.fig.subplots(2, 2).reshape(-1)
        reward = np.asarray([data[f"reward_curve_{index}"] for index in range(seeds)], dtype=float)
        updates = np.arange(1, reward.shape[1] + 1)
        stride = max(1, len(updates) // 2000)
        for index in range(seeds):
            ax_reward.plot(
                updates[::stride],
                reward[index][::stride],
                alpha=0.4,
                linewidth=0.8,
                color=SEED_COLORS[index % len(SEED_COLORS)],
            )
        ax_reward.plot(
            updates[::stride],
            reward.mean(axis=0)[::stride],
            color="#111827",
            linewidth=1.6,
            label=f"{seeds} 种子均值",
        )
        ax_reward.set(
            xlabel="SAC 更新轮次（每环境步 1 次）",
            ylabel="环境奖励（1000 步滑动均值）",
            title="①训练奖励：细线 = 单个种子（r ≈ −末端距离）",
        )
        ax_reward.legend(fontsize=8, loc="lower right")
        alpha = np.asarray([data[f"alpha_curve_{index}"] for index in range(seeds)], dtype=float)
        for index in range(seeds):
            ax_alpha.plot(
                updates[::stride],
                alpha[index][::stride],
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_alpha.set(xlabel="SAC 更新轮次", ylabel="温度 α", title="α 轨迹（自动温度，目标熵 −2）")
        ax_alpha.legend(fontsize=8)
        entropy = np.asarray(
            [data[f"entropy_curve_{index}"] for index in range(seeds)], dtype=float
        )
        for index in range(seeds):
            ax_entropy.plot(
                updates[::stride],
                entropy[index][::stride],
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_entropy.set(xlabel="SAC 更新轮次", ylabel="策略熵（nats）", title="策略熵：探索还活着吗")
        ax_entropy.legend(fontsize=8)
        for index in range(seeds):
            curve = report["rl_evaluation"]["per_seed"][index]["eval_curve"]
            ax_eval.plot(
                [point["env_steps"] / 1000 for point in curve],
                [point["successes"] / point["episodes"] for point in curve],
                "o-",
                markersize=3.5,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_eval.set(
            xlabel="环境步数（×1000）",
            ylabel="周期评估成功率",
            ylim=(-0.05, 1.08),
            title="训练中周期评估（20 个固定目标，确定性策略）",
        )
        ax_eval.legend(fontsize=8, loc="lower right")
        for ax in (ax_reward, ax_alpha, ax_entropy, ax_eval):
            ax.grid(alpha=0.2)

    def draw_trajectories(self):
        """2 showcase-goal tip maps + success bars + arrival times."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        goals = np.asarray(data["eval_goals"], dtype=float)
        ax_a, ax_b, ax_bars, ax_times = self.fig.subplots(2, 2).reshape(-1)
        for ax, goal_index, name in (
            (ax_a, 0, "目标 A（第 8 课验收点）"),
            (ax_b, 1, "目标 B（第 8 课验收点）"),
        ):
            goal = goals[goal_index]
            _add_arm_scene(ax)
            baseline = data[f"baseline_truth_{goal_index}"]
            ax.plot(
                baseline[:, -1, 0],
                baseline[:, -1, 1],
                "--",
                color=BASELINE_COLOR,
                linewidth=1.4,
                label="IK+PD 基线（末端轨迹）",
            )
            _draw_arm_posture(ax, baseline[-1], BASELINE_COLOR, linewidth=1.0)
            for seed_index in range(seeds):
                truth = data[f"eval_truth_{seed_index}_{goal_index}"]
                ax.plot(
                    truth[:, -1, 0],
                    truth[:, -1, 1],
                    color=SEED_COLORS[seed_index % len(SEED_COLORS)],
                    linewidth=1.0,
                    alpha=0.85,
                    label=f"SAC 种子 {seed_index}",
                )
            _add_goal_marker(ax, goal, 0.02)  # 2 cm 参考圈（2 mm 门限画不出来）
            ax.set(
                xlabel="x（m）",
                ylabel="y（m）",
                title=f"②{name} ({goal[0]:.2f}, {goal[1]:+.2f})：绿星 + 2 cm 参考圈",
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=6.5, loc="upper left")
        comparison = report["comparison"]
        labels = ["IK+PD\n基线"] + [f"SAC\n种子 {index}" for index in range(seeds)]
        colors = (BASELINE_COLOR, *SEED_COLORS[:seeds])
        totals = [row["episodes"] for row in comparison]
        successes = [row["successes"] for row in comparison]
        bars = ax_bars.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=colors,
            width=0.55,
        )
        for bar, s, t in zip(bars, successes, totals, strict=True):
            ax_bars.annotate(
                f"{s}/{t}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax_bars.set(ylabel="成功率（%）", ylim=(0, 115), title="同一 20 个固定目标的配对成功率")
        ax_bars.tick_params(axis="x", labelsize=7)
        baseline_times = np.asarray(data["baseline_arrival_time"], dtype=float)
        ax_times.plot(
            np.arange(len(baseline_times)),
            baseline_times,
            "s",
            color=BASELINE_COLOR,
            markersize=5,
            label="IK+PD 基线",
        )
        for seed_index in range(seeds):
            times = np.asarray(data[f"eval_arrival_time_{seed_index}"], dtype=float)
            ax_times.plot(
                np.arange(len(times)),
                times,
                "o",
                markersize=4,
                alpha=0.8,
                color=SEED_COLORS[seed_index % len(SEED_COLORS)],
                label=f"SAC 种子 {seed_index}",
            )
        ax_times.set(
            xlabel="评估目标编号（#0 目标 A / #1 目标 B）",
            ylabel="到达时间（s）",
            title="逐目标到达时间（缺口 = 未到达）",
        )
        ax_times.legend(fontsize=7)
        for ax in (ax_bars, ax_times):
            ax.grid(alpha=0.2)

    def draw_arrival(self):
        """3 final/closest distances + featured case + outcome counts."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        goals = np.asarray(data["eval_goals"], dtype=float)
        ax_final, ax_close, ax_case, ax_counts = self.fig.subplots(2, 2).reshape(-1)
        goal_indices = np.arange(len(goals))
        gate_mm = ARRIVAL_RADIUS_M * 1000
        baseline_dist = np.asarray(data["baseline_final_distance"], dtype=float) * 1000
        ax_final.semilogy(
            goal_indices,
            np.maximum(baseline_dist, 1e-2),
            "D",
            color=BASELINE_COLOR,
            markersize=5,
            label="IK+PD 基线",
        )
        for seed_index in range(seeds):
            distances = np.asarray(data[f"eval_final_distance_{seed_index}"], dtype=float) * 1000
            ax_final.semilogy(
                goal_indices,
                np.maximum(distances, 1e-2),
                "o",
                markersize=4,
                alpha=0.8,
                color=SEED_COLORS[seed_index % len(SEED_COLORS)],
                label=f"SAC 种子 {seed_index}",
            )
        ax_final.axhline(gate_mm, color="#15803d", linestyle="--", linewidth=1.0)
        ax_final.set(
            xlabel="评估目标编号",
            ylabel="最终距目标（mm，对数轴）",
            title="③最终末端误差（绿虚线 = 2 mm 到达门限）",
        )
        ax_final.legend(fontsize=7)
        baseline_close = np.asarray(data["baseline_min_distance"], dtype=float) * 1000
        ax_close.semilogy(
            goal_indices,
            np.maximum(baseline_close, 1e-2),
            "D",
            color=BASELINE_COLOR,
            markersize=5,
            label="IK+PD 基线",
        )
        for seed_index in range(seeds):
            closest = np.asarray(data[f"eval_min_distance_{seed_index}"], dtype=float) * 1000
            ax_close.semilogy(
                goal_indices,
                np.maximum(closest, 1e-2),
                "o",
                markersize=4,
                alpha=0.8,
                color=SEED_COLORS[seed_index % len(SEED_COLORS)],
                label=f"SAC 种子 {seed_index}",
            )
        ax_close.axhline(gate_mm, color="#15803d", linestyle="--", linewidth=1.0)
        ax_close.set(
            xlabel="评估目标编号",
            ylabel="回合内最近距目标（mm，对数轴）",
            title="逐目标最近接近（进没进过 2 mm 球）",
        )
        ax_close.legend(fontsize=7)
        self._draw_featured_case(ax_case, goals)
        baseline_outcomes = np.asarray(data["baseline_outcome"])
        outcome_labels = ("arrived", "timeout", "joint_limit", "failure")
        outcome_names = ("到达", "超时", "限位", "失败")
        width = 0.8 / (seeds + 1)
        for offset, outcomes in enumerate(
            [
                baseline_outcomes,
                *[np.asarray(data[f"eval_outcome_{index}"]) for index in range(seeds)],
            ]
        ):
            counts = [int((outcomes == label).sum()) for label in outcome_labels]
            color = (BASELINE_COLOR, *SEED_COLORS[:seeds])[offset]
            label = "IK+PD 基线" if offset == 0 else f"SAC 种子 {offset - 1}"
            bars = ax_counts.bar(
                np.arange(4) + offset * width, counts, width * 0.92, color=color, label=label
            )
            for bar, count in zip(bars, counts, strict=True):
                if count:
                    ax_counts.annotate(
                        str(count),
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center",
                        xytext=(0, 2),
                        textcoords="offset points",
                        fontsize=7,
                    )
        ax_counts.set(
            xticks=np.arange(4) + width * seeds / 2,
            xticklabels=outcome_names,
            ylabel="回合数",
            title="结局计数（每列 20 个目标）",
        )
        ax_counts.legend(fontsize=7)
        for ax in (ax_final, ax_close, ax_case, ax_counts):
            ax.grid(alpha=0.2)

    def _draw_featured_case(self, ax, goals):
        """A failing goal of the weakest seed; or the slowest arrival honestly."""
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        successes = [
            report["rl_evaluation"]["per_seed"][index]["aggregate"]["successes"]
            for index in range(seeds)
        ]
        featured = int(np.argmin(successes))
        outcomes = np.asarray(data[f"eval_outcome_{featured}"])
        fail_indices = np.flatnonzero(outcomes != "arrived")
        if len(fail_indices):
            goal_index = int(fail_indices[0])
            truth = data[f"eval_truth_{featured}_{goal_index}"]
            goal = goals[goal_index]
            _add_arm_scene(ax)
            ax.plot(
                truth[:, -1, 0],
                truth[:, -1, 1],
                color="#b91c1c",
                linewidth=1.1,
                label="末端轨迹",
            )
            _draw_arm_posture(ax, truth[-1], "#b91c1c")
            _add_goal_marker(ax, goal, 0.02)
            outcome = outcomes[goal_index]
            outcome_name = {
                "timeout": "超时",
                "joint_limit": "限位",
                "failure": "失败",
            }.get(str(outcome), str(outcome))
            ax.set(
                xlabel="x（m）",
                ylabel="y（m）",
                title=f"失败案例：种子 {featured} 目标 #{goal_index}（{outcome_name}）",
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=6.5, loc="upper left")
        else:
            times = np.asarray(data[f"eval_arrival_time_{featured}"], dtype=float)
            goal_index = int(np.nanargmax(times))
            truth = data[f"eval_truth_{featured}_{goal_index}"]
            goal = goals[goal_index]
            _add_arm_scene(ax)
            ax.plot(truth[:, -1, 0], truth[:, -1, 1], color="#0f766e", linewidth=1.1)
            _draw_arm_posture(ax, truth[-1], "#0f766e")
            _add_goal_marker(ax, goal, 0.02)
            ax.set(
                xlabel="x（m）",
                ylabel="y（m）",
                title=f"无失败案例：种子 {featured} 最慢到达（目标 #{goal_index}）",
            )
            ax.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        data = self.data
        seeds = report["training"]["train_seeds"]
        baseline = report["baseline"]["aggregate"]
        if mode == "training":
            lines = ["① 这一幕在追踪：训练收敛了吗"]
            lines.append(f"预算 {report['training']['env_steps_per_seed']} 环境步/种子")
            lines.append(f"总墙钟 {report['training']['wall_time_s_total']:.0f} s，每步 1 更新")
            reward = np.asarray(
                [data[f"reward_curve_{index}"] for index in range(seeds)], dtype=float
            )
            lines.append(
                f"奖励 首→末（均值）：{reward[:, 0].mean():.2f} → {reward[:, -1].mean():.2f}"
            )
            for record in report["rl_evaluation"]["per_seed"]:
                lines.append(
                    f"种子 {record['seed_index']}：α 末值 {record['final_alpha']:.3f}，"
                    f"终评 {record['aggregate']['successes']}/{record['aggregate']['episodes']}"
                )
                lines.append(
                    f"  首成检查点 {record['first_success_checkpoint_steps'] or '从未'}，"
                    f"达标检查点 {record['first_criterion_checkpoint_steps'] or '从未'}"
                )
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同 20 目标配对"]
            lines.append(f"IK+PD 基线（正肘解）：{baseline['successes']}/{baseline['episodes']}")
            lines.append(
                f"  到达中位 {seconds_text(baseline['median_arrival_time_s'])}，"
                f"效率中位 {baseline['median_path_efficiency']:.2f}"
            )
            goals = np.asarray(data["eval_goals"], dtype=float)
            for goal_index, name in ((0, "A"), (1, "B")):
                base_time = np.asarray(data["baseline_arrival_time"], dtype=float)[goal_index]
                seed_times = [
                    np.asarray(data[f"eval_arrival_time_{index}"], dtype=float)[goal_index]
                    for index in range(seeds)
                ]
                seed_arrived = sum(not np.isnan(value) for value in seed_times)
                goal_text = f"({goals[goal_index][0]:.2f}, {goals[goal_index][1]:+.2f})"
                lines.append(f"目标 {name} {goal_text}：基线 {seconds_text(base_time)}")
                if seed_arrived:
                    detail = "、".join(
                        f"种子 {index} {seconds_text(value)}"
                        for index, value in enumerate(seed_times)
                        if not np.isnan(value)
                    )
                    lines.append(f"  种子到达 {seed_arrived}/{seeds}：{detail}")
                else:
                    lines.append(f"  种子到达 {seed_arrived}/{seeds}（全部未到达）")
            across = report["rl_evaluation"]["across_seeds"]
            lines.append(f"学习方合计 {across['successes']}/{across['episodes']}")
            lines.append("对照判据：每种子 ≥ 18/20（预注册）")
        else:
            lines = ["③ 这一幕在裁决：到达质量与失败模式"]
            lines.append(f"基线：终距均值 {baseline['mean_final_distance_m'] * 1000:.3f} mm")
            for record in report["rl_evaluation"]["per_seed"]:
                aggregate = record["aggregate"]
                lines.append(
                    f"种子 {record['seed_index']}：最近中位 "
                    f"{aggregate['median_min_distance_m'] * 1000:.1f} mm，"
                    f"终距均值 {aggregate['mean_final_distance_m'] * 1000:.0f} mm，"
                    f"限位 {aggregate['joint_limits']}"
                )
            across = report["rl_evaluation"]["across_seeds"]
            lines.append(f"学习方合计成功 {across['successes']}/{across['episodes']}")
            lines.append("失败案例面板：最差种子的首个未达目标")
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "training":
            self.draw_training()
        elif mode == "trajectories":
            self.draw_trajectories()
        else:
            self.draw_arrival()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：奖励 / α / 策略熵曲线与训练中周期评估（20 个固定目标）",
            "trajectories": "② 这一幕在对照：目标 A/B 末端轨迹地图（IK+PD 虚线 vs 三种子）+ 配对成功率柱",
            "arrival": "③ 这一幕在裁决：最终/最近末端距离、失败案例（含最终臂姿）与结局计数",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: without it the canvas keeps stale pixels


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第四十课 · 2R 机械臂纯学习到达（全驱动 + 固定小目标盒）")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    RlArmDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
