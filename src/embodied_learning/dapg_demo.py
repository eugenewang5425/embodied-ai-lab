"""Lesson 32 viewer: DAPG-style demonstration airdrop - the teacher as an anchor.

Three static modes share one lesson-32 recording (npz + summary):
1. training curves: per-update task reward and periodic down-start evaluation for
   each initial BC weight w_BC, plus the BC-loss decay (is the demonstration
   remembered or forgotten?);
2. the airdrop comparison: teacher (lesson-7 baseline) versus the learned policy
   from the same exact down start (upright first arrival / settled time
   annotated), cart positions against the failure boundary, the airdropped
   demonstration bundle and the motor inputs;
3. the outcome: baseline / lesson-29 pure PPO / lesson-31 PBRS (cited) versus
   the DAPG tiers, the upright first arrival and headline first accepted
   success (0 -> 1), the paired push recovery and the featured failure case.
Layout and Esc handling follow the lesson-28..31 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.dapg_swingup import (
    EXPERIMENT,
    expected_npz_keys,
    failure_label,
)

DEFAULT_RESULTS = "results/dapg_swingup_2026-09-06"


def level_label(entry):
    return f"w={entry['w_bc']:g}"


def load_replays(directory):
    """Validate a lesson-32 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-32 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["learn_std"]:
        raise ValueError("Unexpected network contract")
    if (
        report["protocol"]["observation"]["features"]
        != "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]"
    ):
        raise ValueError("Unexpected observation contract")
    guard = report["guard"]
    pipeline = guard["pipeline"]
    if not all(pipeline[key] for key in ("bitwise_identical_rewards", "bitwise_identical_states")):
        raise ValueError("Recording claims a broken w_BC = 0 pipeline guard")
    if not guard["training"]["bitwise_identical_policy_weights"]:
        raise ValueError("Recording claims a broken w_BC = 0 training guard")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    seeds = report["training"]["train_seeds"]
    for index, entry in enumerate(report["sweep"]):
        count = entry["stochastic"]["episodes"] // seeds
        terminated = data[f"eval_terminated_{index}"]
        settled = data[f"eval_settled_s_{index}"]
        if terminated.shape != (seeds * count,) or settled.shape != (seeds, count):
            raise ValueError("Archive evaluation shapes disagree with the summary")
        recovered = (~terminated) & (~np.isnan(settled.ravel()))
        derived = recovered.reshape(seeds, count).sum(axis=1).astype(int).tolist()
        if derived != [int(v) for v in entry["stochastic"]["successes_per_seed"]]:
            raise ValueError("DAPG successes disagree with the archive")
        for record in entry["training"]:
            curve = data[f"bc_curve_{index}_{record['seed_index']}"]
            if curve.shape != (report["hyperparameters"]["updates"],):
                raise ValueError("BC curve length disagrees with the update count")
            if abs(float(curve[0]) - record["bc_loss_first"]) > 1e-9 * max(
                1.0, abs(record["bc_loss_first"])
            ):
                raise ValueError("BC curve start disagrees with the summary")
            if abs(float(curve[-1]) - record["bc_loss_last"]) > 1e-9 * max(
                1.0, abs(record["bc_loss_last"])
            ):
                raise ValueError("BC curve end disagrees with the summary")

    demo = report["protocol"]["demonstrations"]
    if (
        data["demo_states"].shape[0] != demo["count"]
        or data["demo_controls"].shape[0] != demo["count"]
    ):
        raise ValueError("Demonstration arrays disagree with the summary count")
    digest = hashlib.sha256()
    for start, states, controls in zip(
        data["demo_start_states"], data["demo_states"], data["demo_controls"], strict=True
    ):
        digest.update(np.ascontiguousarray(start).tobytes())
        digest.update(np.ascontiguousarray(states).tobytes())
        digest.update(np.ascontiguousarray(controls).tobytes())
    if digest.hexdigest() != demo["sha256"]:
        raise ValueError("Demonstration arrays disagree with the recorded hash")
    if int(np.sum(~np.isnan(data["demo_settled_s"]))) != demo["successes"]:
        raise ValueError("Demonstration successes disagree with the archived settle times")

    baseline = report["baseline"]
    per_episode = [row["settled_at_s"] for row in baseline["per_episode"]]
    median_gap = abs(float(np.median(per_episode)) - baseline["median_settled_at_s"])
    if baseline["deterministic_identical_repeats"] and per_episode and median_gap > 1e-9:
        raise ValueError("Baseline median settle time disagrees with its episodes")

    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        case_states = data[f"case{index}_states"]
        if case_states.ndim != 2 or case_states.shape[1] != 4:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        max_x = float(np.max(np.abs(case_states[:, 0])))
        if abs(max_x - case["max_abs_cart_position_m"]) > 1e-4:
            raise ValueError(f"Featured case {index} cart excursion disagrees with its arrays")
    return {"report": report, **data}


class DapgDemo:
    """Demonstration airdrop versus the pure-PPO / PBRS lineage and the baseline."""

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
            text="第三十二课 · DAPG 式示教空投：把策略锚在教师身上，补上最后一公里",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "第 29 课 PPO 骨架原样保留，目标只加一项 w·MSE(μ(s示教), a示教)，w 随更新线性退火到 0（DAPG 惯例）。\n"
                "三个模式共用同一条正式记录：① 训练曲线（奖励 + BC 项衰减）② 教师 vs 学得策略同初态对照 "
                "③ 三方+示教档成功率与失败案例；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：奖励与 BC 项衰减", "training"),
            ("② 空投对照：教师 vs 学得策略（同初态）", "airdrop"),
            ("③ 三方+示教档成功率与失败案例", "outcome"),
            ("④ 最佳过程回放：教师 vs 学得策略 determin 段", "replay"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        # Claim the panel's width on the right BEFORE the expanding canvas
        # frame; the left frame's requested size is frozen (pack_propagate off)
        # so the TkAgg canvas cannot push the status line out of the window.
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left", wraplength=430)
        self.stats.pack(side="right", fill="y", padx=(12, 0))
        left = ttk.Frame(middle, width=1010, height=430)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        self.fig = Figure(figsize=(10.2, 4.3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "DAPG（Rajeswaran 2018）：示教只作 BC 损失的 (s,a) 数据集，不进重放池；Q-filter 未做（如实声明）。"
                "成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        if hasattr(self, "_replay"):
            self._replay.cancel()
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """① task reward, periodic evaluation and the BC decay, per w_BC level."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_eval, ax_bc = self.fig.subplots(1, 3)
        colors = ("#2563eb", "#0f766e")
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        lesson29 = report["lesson29_reference"]
        lesson29_final = float(np.mean(lesson29["final_reward_mean_per_seed"]))
        for index, entry in enumerate(report["sweep"]):
            color = colors[index % len(colors)]
            rewards = np.stack(
                [self.data[f"reward_curve_{index}_{seed}"] for seed in range(seeds)], axis=0
            )
            for seed in range(seeds):
                ax_reward.plot(updates, rewards[seed], alpha=0.3, linewidth=0.8, color=color)
            ax_reward.plot(
                updates,
                rewards.mean(axis=0),
                color=color,
                linewidth=1.8,
                label=f"{level_label(entry)} 均值",
            )
            bcs = np.stack([self.data[f"bc_curve_{index}_{seed}"] for seed in range(seeds)], axis=0)
            for seed in range(seeds):
                ax_bc.plot(
                    updates, np.maximum(bcs[seed], 1e-8), alpha=0.3, linewidth=0.8, color=color
                )
            ax_bc.plot(
                updates,
                np.maximum(bcs.mean(axis=0), 1e-8),
                color=color,
                linewidth=1.8,
                label=level_label(entry),
            )
        ax_reward.axhline(
            lesson29_final,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.2,
            label="纯 PPO 末段",
        )
        ax_reward.set(
            xlabel="PPO 更新轮次",
            ylabel="平均任务奖励",
            title="训练奖励",
        )
        ax_reward.legend(fontsize=7, loc="lower right")
        for index, entry in enumerate(report["sweep"]):
            color = colors[index % len(colors)]
            success = np.asarray(
                [
                    [int(point["success"]) for point in record["eval_curve"]]
                    for record in entry["training"]
                ],
                dtype=float,
            )
            steps = (
                np.asarray(
                    [point["env_steps"] for point in entry["training"][0]["eval_curve"]],
                    dtype=float,
                )
                / 1000.0
            )
            for seed in range(success.shape[0]):
                ax_eval.plot(steps, success[seed], "o--", markersize=4, alpha=0.5, color=color)
            ax_eval.plot(steps, success.mean(axis=0), "o-", color=color, label=level_label(entry))
        ax_eval.set(
            xlabel="环境步数（×1000）",
            ylabel="验收通过（1=成功）",
            ylim=(-0.08, 1.08),
            yticks=[0, 0.5, 1],
            title="周期评估",
        )
        ax_eval.legend(fontsize=7, loc="upper left")
        ax_bc.set(
            xlabel="PPO 更新轮次",
            ylabel="BC 损失 MSE",
            yscale="log",
            title="BC 衰减",
        )
        ax_bc.legend(fontsize=7)
        for ax in (ax_reward, ax_eval, ax_bc):
            ax.grid(alpha=0.2)

    def draw_airdrop(self):
        """② teacher vs learned policy from the same exact down start."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        featured = report["featured_level_index"]
        entry = report["sweep"][featured]
        seed_index = 0
        det_states = self.data[f"det_states_{featured}_{seed_index}"]
        det_controls = self.data[f"det_controls_{featured}_{seed_index}"]
        teacher = self.data["baseline_states"]
        teacher_controls = self.data["baseline_controls"]
        demo_states = self.data["demo_states"]
        demo = report["protocol"]["demonstrations"]
        det_record = entry["deterministic"][seed_index]
        arrival = det_record["first_arrival_s"]
        settled = det_record["settled_at_s"]
        arrival_text = f"{arrival:.2f} s" if arrival is not None else "未到达"
        settled_text = f"{settled:.2f} s" if settled is not None else "未稳定"
        ax_pole, ax_cart, ax_bundle, ax_force = self.fig.subplots(2, 2).reshape(-1)
        ax_pole.plot(
            np.arange(len(teacher)) * dt,
            np.cos(teacher[:, 1] - ref_theta),
            "--",
            color="#64748b",
            label="教师（第 7 课基线）",
        )
        ax_pole.plot(
            np.arange(len(det_states)) * dt,
            np.cos(det_states[:, 1] - ref_theta),
            color="#2563eb",
            label=f"学得策略（{level_label(entry)} 种子 {seed_index}）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"同初态：学得首达 {arrival_text}、稳定 {settled_text}",
        )
        ax_pole.legend(fontsize=7, loc="lower right")
        for bound in (-2.4, 2.4):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.plot(
            np.arange(len(teacher)) * dt, teacher[:, 0], "--", color="#64748b", label="教师"
        )
        ax_cart.plot(
            np.arange(len(det_states)) * dt, det_states[:, 0], color="#2563eb", label="学得策略"
        )
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title="小车位置（红点线 = ±2.4 m 边界）",
        )
        ax_cart.legend(fontsize=7)
        for demo_index in range(len(demo_states)):
            ax_bundle.plot(
                np.arange(len(demo_states[demo_index])) * dt,
                np.cos(demo_states[demo_index, :, 1] - ref_theta),
                alpha=0.3,
                linewidth=0.8,
                color="#b45309",
            )
        ax_bundle.plot(
            np.arange(len(teacher)) * dt,
            np.cos(teacher[:, 1] - ref_theta),
            color="#64748b",
            linewidth=1.6,
            label="教师参考（精确下方初态）",
        )
        ax_bundle.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_bundle.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"空投示教 {demo['count']} 条（全部通过）",
        )
        ax_bundle.legend(fontsize=7, loc="lower right")
        gear = report["protocol"]["actuator_gear"]
        ax_force.stairs(
            teacher_controls * gear,
            np.arange(len(teacher_controls) + 1) * dt,
            color="#64748b",
            label="教师",
        )
        ax_force.stairs(
            det_controls * gear,
            np.arange(len(det_controls) + 1) * dt,
            color="#2563eb",
            label="学得策略",
        )
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="电机输入（±300 N 限幅）",
        )
        ax_force.legend(fontsize=7)
        for ax in (ax_pole, ax_cart, ax_bundle, ax_force):
            ax.grid(alpha=0.2)

    def draw_outcome(self):
        """③ multi-way success bars, first-success panel, pushes, failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        colors = ("#64748b", "#b91c1c", "#7c3aed", "#9333ea", "#2563eb", "#0f766e")
        rows = report["three_way_comparison"]
        labels = ["基线", "纯PPO\n(29)", "PBRS\n0.5", "PBRS\n2"] + [
            f"空投\n{level_label(entry)}" for entry in report["sweep"]
        ]
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        ax_rate, ax_first, ax_push, ax_case = self.fig.subplots(2, 2).reshape(-1)
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=colors[: len(labels)],
            width=0.62,
        )
        for bar, s, t in zip(bars, successes, totals, strict=True):
            ax_rate.annotate(
                f"{s}/{t}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax_rate.set(
            ylabel="验收通过率（%）",
            ylim=(0, 112),
            title="三方+示教两档成功率",
        )
        ax_rate.tick_params(axis="x", labelsize=7)

        lines = ["直立首达与首次成功（0→1 是否出现）："]
        for entry in report["sweep"]:
            arrival = entry["arrival"]
            median = arrival["median_first_arrival_s"]
            median_text = f"{median:.2f} s" if median is not None else "—"
            lines.append(
                f"  {level_label(entry)}：首达 "
                f"{arrival['episodes_with_arrival']}/{arrival['episodes']}（中位 {median_text}），"
                f"首次成功 {'是' if entry['first_success']['any'] else '否'}"
            )
        lines.append("  对照：第 29 课从未首达；第 31 课一次（150k）")
        ax_first.set_xlim(0, 1)
        ax_first.set_ylim(0, 1)
        ax_first.axis("off")
        ax_first.text(
            0.02,
            0.96,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=7.5,
            transform=ax_first.transAxes,
        )
        ax_first.set(title="直立首达与首次成功（头条）")

        baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
        featured = report["featured_level_index"]
        featured_entry = report["sweep"][featured]
        featured_recovery = report["push_test"]["per_level"][featured]["recovery_times_s"]
        seeds = report["training"]["train_seeds"]
        plans = report["push_test"]["plans"]
        plan_indices = np.arange(len(plans))
        ax_push.plot(
            plan_indices,
            [np.nan if v is None else v for v in baseline_recovery],
            "s",
            color="#64748b",
            label="基线",
        )
        seed_recovery = np.asarray(
            [np.nan if v is None else v for v in featured_recovery], dtype=float
        ).reshape(seeds, len(plans))
        for seed in range(seed_recovery.shape[0]):
            ax_push.plot(
                plan_indices,
                seed_recovery[seed],
                "o",
                markersize=4,
                alpha=0.7,
                color=colors[(featured + 4) % len(colors)],
                label=f"示教 {level_label(featured_entry)}（种子 {seed}）",
            )
        ax_push.set(
            xlabel="推力方案编号",
            ylabel="推力结束后恢复时间（s）",
            title="±200 N 配对推力恢复",
        )
        ax_push.set_ylim(2.7, 3.1)  # headroom so the legend never covers the markers
        ax_push.legend(fontsize=7, loc="lower left")

        cases = report["failure_analysis"]["featured_cases"]
        if cases:
            case_states = self.data["case0_states"]
            case = cases[0]
            ax_case.plot(
                np.arange(len(case_states)) * dt,
                np.cos(case_states[:, 1] - ref_theta),
                color="#b91c1c",
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            w_text = f"w={case['w_bc']:g}，" if case.get("w_bc") is not None else ""
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{w_text}{failure_label(case)}",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有示教策略失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例：无（全部回合通过验收）")
        for ax in (ax_rate, ax_push, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        if mode == "training":
            lines = ["① 这一幕在追踪：锚住教师后的训练"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步，"
                f"共 {report['training']['total_env_steps'] / 1000:.0f}k 步"
            )
            lines.append(f"  墙钟 {report['training']['wall_time_s_total']:.0f} s")
            demo = report["protocol"]["demonstrations"]
            lines.append(f"  示教 {demo['count']} 条（{demo['successes']}/{demo['count']} 通过）")
            lines.append(f"  共 {demo['bc_pairs']} 对 (s,a)")
            for entry in report["sweep"]:
                for record in entry["training"]:
                    lines.append(
                        f"  {level_label(entry)} 种子 {record['seed_index']}："
                        f"BC {record['bc_loss_first']:.2f}→{record['bc_loss_last']:.2f}"
                    )
                    first = record["first_successful_eval_steps"]
                    first_text = f"{first / 1000:.0f}k 步" if first is not None else "从未"
                    lines.append(f"    首次成功 {first_text}")
            lines.append("  BC 只进目标函数，不进重放池")
        elif mode == "airdrop":
            lines = ["② 这一幕在对照：空投有没有用"]
            featured = report["featured_level_index"]
            entry = report["sweep"][featured]
            det = entry["deterministic"][0]
            arrival = det["first_arrival_s"]
            arrival_text = f"{arrival:.2f} s" if arrival is not None else "未到达"
            settled = det["settled_at_s"]
            settled_text = f"{settled:.2f} s" if settled is not None else "未稳定"
            lines.append(f"  教师稳定（基线）：{report['baseline']['median_settled_at_s']:.2f} s")
            lines.append(f"  学得策略（{level_label(entry)}）：首达 {arrival_text}")
            lines.append(f"    稳定 {settled_text}")
            demo = report["protocol"]["demonstrations"]
            lines.append(f"  示教质量：{demo['successes']}/{demo['count']} 通过")
            lines.append(f"    稳定中位 {demo['median_settled_at_s']:.2f} s")
            lines.append("  起点抖动 ±0.15 rad、±0.3 rad/s")
            lines.append("    ±0.10 m")
            for other in report["sweep"]:
                lines.append(
                    f"  {level_label(other)} 评估首达 "
                    f"{other['arrival']['episodes_with_arrival']}"
                    f"/{other['arrival']['episodes']}"
                )
        else:
            lines = ["③ 这一幕在裁决：0→1 出现了吗"]
            compact = (
                "基线（第 7 课能量整形+LQR，零样本）",
                "纯 PPO（第 29 课，只凭奖励）",
            )
            for row in report["three_way_comparison"]:
                label = row["label"]
                if label.startswith(compact[0]):
                    label = "基线（第 7 课）"
                elif label.startswith(compact[1]):
                    label = "纯 PPO（29 课）"
                elif label.startswith("PBRS"):
                    label = label.replace("，第 31 课", " 31 课")
                lines.append(f"  {label}：{row['successes']}/{row['episodes']}")
            for entry in report["sweep"]:
                if entry["first_success"]["any"]:
                    steps_text = fmt_steps(entry)
                    lines.append(f"  示教{level_label(entry)}：首次成功是（{steps_text}）")
                else:
                    lines.append(f"  示教{level_label(entry)}：首次成功否")
            push = report["push_test"]
            lines.append(
                f"  推力：基线 {push['baseline']['successes']}/{push['baseline']['episodes']}"
            )
            for item in push["per_level"]:
                lines.append(f"    空投 w={item['w_bc']:g}：{item['successes']}/{item['episodes']}")
            counts = report["failure_analysis"]["eval_counts"]
            lines.append(f"  失败：出界 {counts['cart_safety_boundary']}")
            lines.append(f"        超时未稳 {counts['timeout_without_settling']}")
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "training":
            self.draw_training()
        elif mode == "airdrop":
            self.draw_airdrop()
        elif mode == "replay":
            self.draw_replay()
        else:
            self.draw_outcome()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：两档初始 BC 权重的训练曲线与 BC 项衰减，对照第 29 课纯 PPO 口径",
            "airdrop": "② 这一幕在对照：教师与学得策略同初态轨迹、空投示教束与电机输入",
            "outcome": "③ 这一幕在裁决：基线 / 纯 PPO / PBRS / 示教两档成功率，直立首达与首次成功",
            "replay": "④ 这一幕在回放：教师 4.76 s 抓取 vs 学得策略 determin 段（return 最大）",
        }[mode]
        if mode == "replay":
            self.status.configure(text=status + "｜播放/暂停/单步/调速。按 Esc 退出。")
        else:
            self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches

    def draw_replay(self):
        """④ 2D replay: teacher demo vs the best deterministic run (max return)."""
        from embodied_learning._replay2d import Replay2D

        demo = self.data["demo_states"][0]  # first teacher trajectory (751,4)
        # best deterministic run: highest mean_return across seeds
        dets = self.report["sweep"][0]["deterministic"]
        best = max(range(len(dets)), key=lambda i: dets[i]["return"])
        best_states = self.data[f"det_states_0_{best}"]
        ret = dets[best]["return"]
        if not hasattr(self, "_replay"):
            self._replay = Replay2D(self.root, self.fig, on_close=None)
        self._replay.setup_axes([
            (None, demo, "#0f766e", "教师：第 7 课基线（4.76 s 抓取）"),
            (None, best_states, "#b91c1c",
             f"学得策略 determin 段（seed {best}，return={ret:.0f}）"),
        ])
        self._replay.set_step(0)


def fmt_steps(entry):
    steps = [s for s in entry["first_success"]["per_seed_eval_steps"] if s is not None]
    return "、".join(f"{s / 1000:.0f}k 步" for s in steps)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第三十二课 · DAPG 式示教空投：锚住教师，补上最后一公里")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    DapgDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
