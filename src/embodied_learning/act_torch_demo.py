"""Lesson 42 viewer: torch ACT chunked policy on the 2R reaching task.

Three static modes share one lesson-42 recording (npz + summary):
1. the training curves: the behaviour-cloning loss (MSE + 0.1 L1, train and
   held-out validation) and the periodic deterministic evaluation during
   training (success rate and closest approach on the first five fixed goals)
   - did the chunk policy converge to the teacher, and when?
2. the trajectory comparison on the two showcase goals (the lesson-8
   acceptance targets A and B): the analytic IK+PD teacher (dashed) versus the
   cited lesson-40 SAC seeds (thin purple) versus this lesson's ACT seeds in
   the tip plane, plus the paired success-rate bars and per-goal arrival
   times;
3. attention and failure cases: the encoder self-attention (T=4 tokens), the
   decoder cross-attention (H=16 chunk queries over the T=4 memory tokens)
   captured at the attention goal's approach, the per-goal closest approach
   against the 2 mm gate, and one featured ACT failure (or the slowest
   arrival when there is none) drawn with its final arm posture.

Layout and Esc handling follow the lesson-28..40 demos (docs/26 section 6):
every mode calls fig.clear() first, redraw draws synchronously, panel lines
stay short (manual breaks) and every quoted number is traceable to
summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.act_torch_reaching import (
    ARRIVAL_RADIUS_M,
    EXPERIMENT,
    SHOWCASE_GOALS,
    expected_npz_keys,
)
from embodied_learning.experiments.rl_arm_reaching import GOAL_HIGH_M, GOAL_LOW_M

DEFAULT_RESULTS = "results/act_torch_reaching_2026-09-06"
SEED_COLORS = ("#0f766e", "#2563eb", "#b45309")
BASELINE_COLOR = "#64748b"
SAC_COLOR = "#7c3aed"


def seconds_text(value):
    """'2.60 s' or '未到达'; NaN (never arrived) counts as not arrived."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "未到达"
    return f"{value:.2f} s"


def load_replays(directory):
    """Validate a lesson-42 recording; returns the data dict for the demo.

    Tamper routes rejected here: (1) the experiment/schema identity; (2) the
    npz SHA-256 against the summary; (3) the archive key set against the
    implied key set; (4) the per-seed ACT, baseline and cited-SAC success
    counts recomputed from the archived outcomes against the summary; (5) the
    showcase goals archived in the record against the protocol constants.
    """
    from embodied_learning.experiments.act_torch_reaching import ATTENTION_PLAN_INDEX

    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-42 recording")
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
        recorded = report["bc_evaluation"]["per_seed"][seed_index]["aggregate"]["successes"]
        if derived != recorded:
            raise ValueError("ACT successes disagree with the archive")
    baseline_derived = int((np.asarray(data["baseline_outcome"]) == "arrived").sum())
    if baseline_derived != report["baseline"]["aggregate"]["successes"]:
        raise ValueError("Baseline successes disagree with the archive")
    sac = report.get("sac_reference", {})
    if sac.get("loaded"):
        for seed_index, cited in enumerate(sac["per_seed"]):
            outcomes = np.asarray(data[f"sac_outcome_{seed_index}"])
            derived = int((outcomes == "arrived").sum())
            if derived != cited["aggregate"]["successes"]:
                raise ValueError("Cited SAC successes disagree with the archive")
    if any(f"attention_encoder_{index}" not in data for index in range(seeds)):
        raise ValueError("Attention maps missing")
    if report["bc_evaluation"]["per_seed"][0]["attention_plan_index"] != ATTENTION_PLAN_INDEX:
        raise ValueError("Attention capture index disagrees with the module constant")
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


class ActTorchDemo:
    """torch ACT arm reaching: training, trajectory comparison, attention."""

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
            text="第四十二课 · torch ACT 块策略冲击毫米级到达（2R 臂，第 40 课同环境同口径）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "本课首次引入 torch：小 Transformer（2 层 / 4 头 / d_model 128）读最近 4 步观测，"
                "一次输出 16 步动作块，每 4 步重规划；行为克隆教师 = 第 8 课解析 IK+PD（同基线链路）。"
                "无 CVAE、无时序集成（记录在案的简化）。对照 = 同场 IK+PD 基线与第 40 课纯 numpy SAC（引用）。"
                "三模式共用同一正式记录，全部为静态图"
            ),
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练损失与评估成功率", "training"),
            ("② 教师 vs ACT vs SAC 轨迹对照", "trajectories"),
            ("③ 注意力可视化与失败案例", "attention"),
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
                "成功判据与第 40 课逐字相同：末端距目标 < 2 mm 且 max|dq| < 0.1 rad/s 持续 0.5 s；"
                "成功率对照同 20 个固定目标（配对）；预注册学习判据 = 每种子 ≥ 18/20"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _event: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """1 train/val loss + periodic eval success/closest approach per seed."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_train, ax_val, ax_success, ax_close = self.fig.subplots(2, 2).reshape(-1)
        epochs = np.arange(1, report["training"]["epochs_per_seed"] + 1)
        for index in range(seeds):
            ax_train.plot(
                epochs,
                data[f"train_loss_curve_{index}"],
                linewidth=1.0,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_train.set(
            xlabel="epoch",
            ylabel="训练损失（MSE + 0.1·L1）",
            title="①训练损失（对数轴）：拟合教师动作块",
        )
        ax_train.set_yscale("log")
        ax_train.legend(fontsize=8)
        for index in range(seeds):
            ax_val.plot(
                epochs,
                data[f"val_total_curve_{index}"],
                linewidth=1.0,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_val.set(
            xlabel="epoch",
            ylabel="验证损失（未见窗口）",
            title="验证损失：教师标注重建的泛化（对数轴）",
        )
        ax_val.set_yscale("log")
        ax_val.legend(fontsize=8)
        for index in range(seeds):
            curve = report["bc_evaluation"]["per_seed"][index]["eval_curve"]
            ax_success.plot(
                [point["epoch"] for point in curve],
                [point["successes"] / point["episodes"] for point in curve],
                "o-",
                markersize=3.5,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_success.set(
            xlabel="epoch",
            ylabel="周期评估成功率",
            ylim=(-0.05, 1.08),
            title=(
                "训练中周期评估（前 "
                f"{report['hyperparameters']['eval_subset']} 个固定目标，块执行闭环）"
            ),
        )
        ax_success.legend(fontsize=8, loc="lower right")
        for index in range(seeds):
            ax_close.plot(
                data[f"eval_curve_epochs_{index}"],
                np.maximum(np.asarray(data[f"eval_curve_min_distance_{index}"]) * 1000, 1e-2),
                "o-",
                markersize=3.5,
                color=SEED_COLORS[index % len(SEED_COLORS)],
                label=f"种子 {index}",
            )
        ax_close.axhline(ARRIVAL_RADIUS_M * 1000, color="#15803d", linestyle="--", linewidth=1.0)
        ax_close.set(
            xlabel="epoch",
            ylabel="最近接近中位（mm，对数轴）",
            title="周期评估最近接近（绿虚线 = 2 mm 到达门限）",
        )
        ax_close.set_yscale("log")
        ax_close.legend(fontsize=8)
        for ax in (ax_train, ax_val, ax_success, ax_close):
            ax.grid(alpha=0.2)

    def draw_trajectories(self):
        """2 showcase tip maps: teacher vs cited SAC vs ACT + success bars."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        sac = report["sac_reference"]
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
                linewidth=1.6,
                label="IK+PD 教师（=基线）",
            )
            if sac.get("loaded"):
                for seed_index in range(sac["seeds"]):
                    truth = data[f"sac_truth_{seed_index}_{goal_index}"]
                    ax.plot(
                        truth[:, -1, 0],
                        truth[:, -1, 1],
                        color=SAC_COLOR,
                        linewidth=0.8,
                        alpha=0.65,
                        label=f"SAC 种子 {seed_index}（40 课引用）" if seed_index == 0 else None,
                    )
            for seed_index in range(seeds):
                truth = data[f"eval_truth_{seed_index}_{goal_index}"]
                ax.plot(
                    truth[:, -1, 0],
                    truth[:, -1, 1],
                    color=SEED_COLORS[seed_index % len(SEED_COLORS)],
                    linewidth=1.0,
                    alpha=0.85,
                    label=f"ACT 种子 {seed_index}",
                )
            _add_goal_marker(ax, goal, 0.02)  # 2 cm 参考圈（2 mm 门限画不出来）
            ax.set(
                xlabel="x（m）",
                ylabel="y（m）",
                title=f"②{name} ({goal[0]:.2f}, {goal[1]:+.2f})：绿星 + 2 cm 参考圈",
            )
            ax.set_aspect("equal", adjustable="datalim")
            ax.legend(fontsize=6, loc="upper left")
        baseline_agg = report["baseline"]["aggregate"]
        bar_rows = [
            ("IK+PD\n教师", BASELINE_COLOR, baseline_agg["successes"], baseline_agg["episodes"])
        ]
        sac = report["sac_reference"]
        if sac.get("loaded"):
            across = sac["across_seeds"]
            bar_rows.append(
                ("SAC\n(40 课引用)", SAC_COLOR, across["successes"], across["episodes"])
            )
        for index, record in enumerate(report["bc_evaluation"]["per_seed"]):
            aggregate = record["aggregate"]
            bar_rows.append(
                (
                    f"ACT\n种子 {index}",
                    SEED_COLORS[index % len(SEED_COLORS)],
                    aggregate["successes"],
                    aggregate["episodes"],
                )
            )
        labels = [row[0] for row in bar_rows]
        colors = [row[1] for row in bar_rows]
        successes = [row[2] for row in bar_rows]
        totals = [row[3] for row in bar_rows]
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
        ax_bars.tick_params(axis="x", labelsize=6.5)
        baseline_times = np.asarray(data["baseline_arrival_time"], dtype=float)
        ax_times.plot(
            np.arange(len(baseline_times)),
            baseline_times,
            "s",
            color=BASELINE_COLOR,
            markersize=5,
            label="IK+PD 教师",
        )
        if sac.get("loaded"):
            sac_first = np.asarray(data["sac_arrival_time_0"], dtype=float)
            ax_times.plot(
                np.arange(len(sac_first)),
                sac_first,
                "^",
                color=SAC_COLOR,
                markersize=4,
                alpha=0.7,
                label="SAC 种子 0（全部缺口见 40 课）",
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
                label=f"ACT 种子 {seed_index}",
            )
        ax_times.set(
            xlabel="评估目标编号（#0 目标 A / #1 目标 B）",
            ylabel="到达时间（s）",
            title="逐目标到达时间（缺口 = 未到达）",
        )
        ax_times.legend(fontsize=6)
        for ax in (ax_bars, ax_times):
            ax.grid(alpha=0.2)

    def draw_attention(self):
        """3 attention heatmaps + closest approach + featured failure case."""
        self.fig.clear()
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        goals = np.asarray(data["eval_goals"], dtype=float)
        ax_enc, ax_cross, ax_close, ax_case = self.fig.subplots(2, 2).reshape(-1)
        attention_seed = 0
        chunk_t = report["hyperparameters"]["chunk_t"]
        chunk_h = report["hyperparameters"]["chunk_h"]
        encoder_map = np.asarray(data[f"attention_encoder_{attention_seed}"], dtype=float)
        image = ax_enc.imshow(encoder_map, cmap="magma", vmin=0.0, vmax=1.0)
        ax_enc.set_xticks(range(chunk_t), [f"t−{chunk_t - 1 - i}" for i in range(chunk_t)])
        ax_enc.set_yticks(range(chunk_t), [f"t−{chunk_t - 1 - i}" for i in range(chunk_t)])
        ax_enc.set(
            xlabel="被注意的观测（过去）",
            ylabel="发出注意的观测",
            title=f"③编码器自注意力（种子 0，T={chunk_t}；捕获于第 "
            f"{report['bc_evaluation']['per_seed'][0]['attention_plan_index']} 次重规划）",
        )
        self.fig.colorbar(image, ax=ax_enc, fraction=0.046)
        cross_map = np.asarray(data[f"attention_decoder_{attention_seed}"], dtype=float)
        image = ax_cross.imshow(cross_map, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax_cross.set_xticks(range(chunk_t), [f"t−{chunk_t - 1 - i}" for i in range(chunk_t)])
        ax_cross.set_yticks(
            range(0, chunk_h, max(1, chunk_h // 8)),
            [str(i) for i in range(0, chunk_h, max(1, chunk_h // 8))],
        )
        ax_cross.set(
            xlabel="交叉注意的记忆 token（T 个观测）",
            ylabel="动作块步（H 个查询）",
            title=f"解码器交叉注意力（H={chunk_h} 查询 × T={chunk_t} 记忆，头平均）",
        )
        self.fig.colorbar(image, ax=ax_cross, fraction=0.046)
        goal_indices = np.arange(len(goals))
        gate_mm = ARRIVAL_RADIUS_M * 1000
        baseline_close = np.asarray(data["baseline_min_distance"], dtype=float) * 1000
        ax_close.semilogy(
            goal_indices,
            np.maximum(baseline_close, 1e-2),
            "D",
            color=BASELINE_COLOR,
            markersize=5,
            label="IK+PD 教师",
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
                label=f"ACT 种子 {seed_index}",
            )
        ax_close.axhline(gate_mm, color="#15803d", linestyle="--", linewidth=1.0)
        ax_close.set(
            xlabel="评估目标编号",
            ylabel="回合内最近距目标（mm，对数轴）",
            title="逐目标最近接近（进没进过 2 mm 球）",
        )
        ax_close.legend(fontsize=7)
        self._draw_featured_case(ax_case, goals)
        for ax in (ax_enc, ax_cross, ax_close, ax_case):
            ax.grid(alpha=0.2)

    def _draw_featured_case(self, ax, goals):
        """A failing goal of the weakest ACT seed; or the slowest arrival honestly."""
        data = self.data
        report = self.report
        seeds = report["training"]["train_seeds"]
        successes = [
            report["bc_evaluation"]["per_seed"][index]["aggregate"]["successes"]
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
            closest = float(np.asarray(data[f"eval_min_distance_{featured}"])[goal_index])
            ax.set(
                xlabel="x（m）",
                ylabel="y（m）",
                title=(
                    f"失败案例：ACT 种子 {featured} 目标 #{goal_index}"
                    f"（{outcome_name}，最近 {closest * 1000:.1f} mm）"
                ),
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
                title=f"无失败案例：ACT 种子 {featured} 最慢到达（目标 #{goal_index}）",
            )
            ax.set_aspect("equal", adjustable="datalim")

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        baseline = report["baseline"]["aggregate"]
        if mode == "training":
            lines = ["① 这一幕在追踪：块策略拟合教师了吗"]
            lines.append(
                f"预算 {report['training']['epochs_per_seed']} epoch/种子，"
                f"{report['training']['windows_per_epoch']} 窗口/epoch"
            )
            lines.append(f"总墙钟 {report['training']['total_wall_time_s']:.0f} s")
            lines.append(
                f"教师数据 {report['teacher']['env_steps']} 环境步"
                f"（{report['teacher']['episodes']} 条 IK+PD 轨迹）"
            )
            for record in report["bc_evaluation"]["per_seed"]:
                lines.append(
                    f"种子 {record['seed_index']}：损失 "
                    f"{record['final_train_loss']:.4f}，验证 RMSE "
                    f"{record['final_val_rmse_mnm']:.1f} mN·m"
                )
                lines.append(
                    f"  终评 {record['aggregate']['successes']}/"
                    f"{record['aggregate']['episodes']}，首成检查点 "
                    f"epoch {record['first_success_checkpoint_epochs'] or '从未'}"
                )
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同 20 目标配对"]
            lines.append(
                f"IK+PD 教师：{baseline['successes']}/{baseline['episodes']}"
                f"（到达中位 {seconds_text(baseline['median_arrival_time_s'])}）"
            )
            sac = report["sac_reference"]
            if sac.get("loaded"):
                across = sac["across_seeds"]
                lines.append(
                    f"纯 numpy SAC（40 课引用）：{across['successes']}/{across['episodes']}"
                )
                lines.append(
                    f"  最近中位 {across['median_min_distance_m'] * 1000:.1f} mm，"
                    f"来源 sha {str(sac.get('trajectories_sha256'))[:8]}…"
                )
            for record in report["bc_evaluation"]["per_seed"]:
                aggregate = record["aggregate"]
                lines.append(
                    f"ACT 种子 {record['seed_index']}：{aggregate['successes']}/"
                    f"{aggregate['episodes']}，到达中位 "
                    f"{seconds_text(aggregate['median_arrival_time_s'])}"
                )
            across = report["bc_evaluation"]["across_seeds"]
            lines.append(f"ACT 合计 {across['successes']}/{across['episodes']}")
            lines.append("对照判据：每种子 ≥ 18/20（预注册）")
        else:
            lines = ["③ 这一幕在解剖：注意力与失败模式"]
            for record in report["bc_evaluation"]["per_seed"]:
                lines.append(
                    f"种子 {record['seed_index']}：验证 RMSE "
                    f"{record['final_val_rmse_mnm']:.1f} mN·m，L1 "
                    f"{record['final_val_l1_mnm']:.1f} mN·m"
                )
            across = report["bc_evaluation"]["across_seeds"]
            lines.append(
                f"ACT 合计成功 {across['successes']}/{across['episodes']}，"
                f"限位 {across['joint_limits']}"
            )
            lines.append(f"最近中位 {across['median_min_distance_m'] * 1000:.1f} mm")
            lines.append(
                f"注意力捕获：目标 A 第 {report['bc_evaluation']['per_seed'][0]['attention_plan_index']} 次重规划"
            )
            lines.append("无 CVAE / 无门控（简化记录在案）")
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
            self.draw_attention()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：训练/验证损失曲线与训练中周期评估（块执行闭环）",
            "trajectories": "② 这一幕在对照：目标 A/B 末端轨迹地图（IK+PD 虚线 vs SAC 引用 vs ACT）+ 配对成功率柱",
            "attention": "③ 这一幕在解剖：编码器自注意力 / 解码器交叉注意力、最近接近与失败案例（含最终臂姿）",
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
    root.title("第四十二课 · torch ACT 块策略冲击毫米级到达（2R 臂）")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    ActTorchDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
