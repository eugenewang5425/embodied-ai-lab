"""Lesson 29 viewer: what reward-only PPO learned - and what it did not.

Three static modes share one lesson-29 recording (npz + summary):
1. training curves: per-update reward and periodic down-start evaluation,
   per training seed and averaged;
2. the exact down start: lesson-7 energy+LQR baseline versus the trained PPO
   policy (mean action), trajectories and same-caliber success rates;
3. disturbance recovery under +/-200 N pushes PPO never saw, plus the
   featured failure case and the failure statistics.
Layout and Esc handling follow the lesson-28 demo: every mode calls fig.clear()
first and every quoted number is traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.ppo_swingup import EXPERIMENT, expected_npz_keys

DEFAULT_RESULTS = "results/ppo_swingup_2026-09-06"


def push_recovery_times(summary_block, seed_index=None):
    """recovery_times_s of the baseline (seed_index None) or one PPO seed."""
    if seed_index is None:
        return summary_block["recovery_times_s"]
    for record in summary_block["ppo_per_seed"]:
        if record["seed_index"] == seed_index:
            return record["recovery_times_s"]
    raise ValueError(f"Unknown PPO training seed index {seed_index}")


def load_replays(directory):
    """Validate a lesson-29 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-29 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["learn_std"]:
        raise ValueError("Unexpected network contract")
    if (
        report["protocol"]["observation"]["features"]
        != "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]"
    ):
        raise ValueError("Unexpected observation contract")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    seeds = report["training"]["train_seeds"]
    if data["eval_terminated"].shape[0] != seeds:
        raise ValueError("Archive seed count disagrees with the summary")
    success_per_seed = report["ppo_evaluation"]["aggregate"]["successes_per_seed"]
    derived = (
        ((~data["eval_terminated"]) & (~np.isnan(data["eval_settled_s"]))).sum(axis=1).tolist()
    )
    if [int(v) for v in derived] != [int(v) for v in success_per_seed]:
        raise ValueError("PPO successes disagree with the archive")
    for seed_index in range(seeds):
        times = push_recovery_times(report["push_test"], seed_index)
        derived_push = [
            None if not np.isfinite(v) else float(v) for v in data["push_recovery_s"][seed_index]
        ]
        if [None if t is None else float(t) for t in times] != derived_push:
            raise ValueError(f"Push recovery times of seed {seed_index} disagree with the archive")
    baseline = report["baseline"]
    baseline_successes = sum(
        1
        for episode in baseline["per_episode"]
        if episode["recovered"] and not episode["terminated"]
    )
    if baseline_successes != baseline["successes"]:
        raise ValueError("Baseline successes disagree with the per-episode records")
    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        states = data[f"case{index}_states"]
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        max_x = float(np.max(np.abs(states[:, 0])))
        if abs(max_x - case["max_abs_cart_position_m"]) > 1e-4:
            raise ValueError(f"Featured case {index} cart excursion disagrees with its arrays")
    return {"report": report, **data}


class PpoDemo:
    """Reward-only PPO versus the hand-designed lesson-7 baseline."""

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
            text="第二十九课 · 强化学习入口：不注入模型知识，PPO 只凭奖励能学会摆起吗",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "纯 numpy 手写 PPO（5×64×64 高斯策略＋价值网络，GAE+clip+熵）对照第七课能量整形+LQR 基线。\n"
                "三个模式共用同一条正式记录：① 训练曲线（3 种子）② 同一下方初态轨迹对照 ③ 推力恢复与失败案例；"
                "全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：奖励与验收（3 训练种子）", "training"),
            ("② 同一下方初态：基线 vs PPO 轨迹与成功率", "trajectories"),
            ("③ ±200 N 推力恢复对照 + 失败案例", "push"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        # Claim the panel's width on the right BEFORE the expanding canvas
        # frame: packed afterwards with side=left+expand the canvas consumes
        # the whole cavity and the panel is never mapped. The left frame's
        # requested size is frozen (pack_propagate off) because the TkAgg
        # canvas re-syncs its request to the stretched size on redraw, which
        # would push the status and caption labels out of the 760 px window.
        self.stats = ttk.Label(middle, width=48, anchor="nw", justify="left")
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
                "训练起点随机化与奖励都是 RL 方法的一部分，已写入记录并如实讨论；"
                "所有评估固定在精确静止下方初态；成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """① per-update reward curves and periodic down-start evaluations."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_eval = self.fig.subplots(1, 2)
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        rewards = np.stack([self.data[f"reward_curve_{index}"] for index in range(seeds)], axis=0)
        for index in range(seeds):
            ax_reward.plot(
                updates,
                rewards[index],
                alpha=0.35,
                linewidth=0.9,
                color="#64748b",
                label=f"种子 {index}",
            )
        ax_reward.plot(
            updates,
            rewards.mean(axis=0),
            color="#0f766e",
            linewidth=1.8,
            label=f"{seeds} 种子均值",
        )
        ax_reward.set(
            xlabel="PPO 更新轮次",
            ylabel="批内平均原始奖励（每步）",
            title="训练奖励：细线 = 单个种子",
        )
        ax_reward.legend(fontsize=7, loc="lower right")
        per_seed = report["ppo_evaluation"]["per_training_seed"]
        steps = (
            np.asarray([point["env_steps"] for point in per_seed[0]["eval_curve"]], dtype=float)
            / 1000.0
        )
        success = np.asarray(
            [[int(point["success"]) for point in record["eval_curve"]] for record in per_seed],
            dtype=float,
        )
        for index in range(seeds):
            ax_eval.plot(
                steps,
                success[index],
                "o--",
                markersize=4,
                alpha=0.6,
                color="#64748b",
                label=f"种子 {index}",
            )
        ax_eval.plot(steps, success.mean(axis=0), "o-", color="#b91c1c", label=f"{seeds} 种子均值")
        ax_eval.set(
            xlabel="环境步数（×1000）",
            ylabel="下方初态验收通过（1=成功）",
            ylim=(-0.08, 1.08),
            yticks=[0, 0.5, 1],
            title="周期评估：均值动作、下方初态",
        )
        ax_eval.legend(fontsize=7, loc="upper left")
        for ax in (ax_reward, ax_eval):
            ax.grid(alpha=0.2)

    def draw_trajectories(self):
        """② exact down start: baseline vs PPO mean action, plus success bars."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        gear = report["protocol"]["actuator_gear"]
        ref_theta = report["protocol"]["reference_state"][1]
        boundary = report["protocol"]["cart_failure_boundary_m"]
        ax_pole, ax_cart, ax_force, ax_rate = self.fig.subplots(2, 2).reshape(-1)
        baseline_states = self.data["baseline_states"]
        det_states = self.data["eval_det_states"][0]
        baseline_ts = np.arange(len(baseline_states)) * dt
        ppo_ts = np.arange(len(det_states)) * dt
        ax_pole.plot(
            baseline_ts,
            np.cos(baseline_states[:, 1] - ref_theta),
            "--",
            color="gray",
            label="基线（能量+LQR，零样本）",
        )
        ax_pole.plot(
            ppo_ts, np.cos(det_states[:, 1] - ref_theta), color="#0f766e", label="PPO（均值动作）"
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            title="摆起轨迹（橙带 = 杆端低于铰点）",
        )
        ax_pole.legend(fontsize=7, loc="upper left")
        ax_cart.plot(baseline_ts, baseline_states[:, 0], "--", color="gray")
        ax_cart.plot(ppo_ts, det_states[:, 0], color="#2563eb")
        for bound in (-boundary, boundary):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title=f"小车位置与 ±{boundary:.1f} m 边界",
        )
        baseline_controls = self.data["baseline_controls"]
        det_controls = self.data["eval_det_controls"][0]
        baseline_edges = np.arange(len(baseline_controls) + 1) * dt
        ppo_edges = np.arange(len(det_controls) + 1) * dt
        ax_force.stairs(baseline_controls * gear, baseline_edges, color="gray", label="基线")
        ax_force.stairs(det_controls * gear, ppo_edges, color="#2563eb", label="PPO")
        ax_force.set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入")
        ax_force.legend(fontsize=7, loc="upper right")
        baseline_summary = report["baseline"]
        aggregate = report["ppo_evaluation"]["aggregate"]
        labels = [
            f"基线\n{baseline_summary['episodes']} 次重复",
            f"PPO\n{aggregate['episodes']} 个随机回合",
        ]
        totals = [baseline_summary["episodes"], aggregate["episodes"]]
        successes = [baseline_summary["successes"], aggregate["successes"]]
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=["#64748b", "#0f766e"],
            width=0.55,
        )
        for bar, s, t in zip(bars, successes, totals, strict=True):
            ax_rate.annotate(
                f"{s}/{t}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=9,
            )
        ax_rate.set(
            ylabel="验收通过率（%）",
            ylim=(0, 112),
            title="同口径成功率（第 7 课验收）",
        )
        for ax in (ax_pole, ax_cart, ax_force, ax_rate):
            ax.grid(alpha=0.2)

    def draw_push(self):
        """③ paired ±200 N pushes, recovery times, featured failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        gear = report["protocol"]["actuator_gear"]
        ref_theta = report["protocol"]["reference_state"][1]
        plans = report["push_test"]["plans"]
        plan_count = len(plans)
        baseline_recovery = push_recovery_times(report["push_test"]["baseline"])
        per_seed = report["push_test"]["ppo_per_seed"]
        cases = report["failure_analysis"]["featured_cases"]
        ax_pair, ax_times, ax_case, ax_counts = self.fig.subplots(2, 2).reshape(-1)

        paired = next(
            (
                index
                for index in range(plan_count)
                if baseline_recovery[index] is not None
                and per_seed[0]["recovery_times_s"][index] is not None
            ),
            0,
        )
        plan = plans[paired]
        baseline_pair = self.data["baseline_push_states"][paired]
        ppo_pair = self.data["push_states"][paired]
        ppo_pair_len = int(self.data["push_lengths"][paired])
        ax_pair.axvspan(
            plan["start_s"], plan["start_s"] + plan["duration_s"], alpha=0.2, color="#fbbf24"
        )
        ax_pair.plot(
            np.arange(len(baseline_pair)) * dt,
            np.cos(baseline_pair[:, 1] - ref_theta),
            "--",
            color="gray",
            label="基线（能量+LQR）",
        )
        ax_pair.plot(
            np.arange(ppo_pair_len) * dt,
            np.cos(ppo_pair[:ppo_pair_len, 1] - ref_theta),
            color="#0f766e",
            label="PPO（训练种子 0）",
        )
        ax_pair.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pair.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"配对推力：{plan['force_n']:+.0f} N @ {plan['start_s']:.2f} s",
        )
        ax_pair.legend(fontsize=7, loc="upper left")

        plan_indices = np.arange(plan_count)
        ax_times.plot(
            plan_indices,
            [np.nan if v is None else v for v in baseline_recovery],
            "s",
            color="#64748b",
            label="基线",
        )
        for seed_index, record in enumerate(per_seed):
            ax_times.plot(
                plan_indices,
                [np.nan if v is None else v for v in record["recovery_times_s"]],
                "o",
                markersize=4,
                alpha=0.55,
                label=f"PPO 种子 {seed_index}",
            )
        ax_times.set(
            xlabel="推力方案编号",
            ylabel="推力结束后恢复时间（s）",
            title="恢复时间（缺口 = 未恢复）",
        )
        ax_times.legend(fontsize=7, loc="upper left")

        if cases:
            case = cases[0]
            case_states = self.data["case0_states"]
            ax_case.plot(
                np.arange(len(case_states)) * dt,
                np.cos(case_states[:, 1] - ref_theta),
                color="#b91c1c",
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                ylim=(-1.1, 1.1),
                xlabel="仿真时间（s）",
                title=f"失败案例（{case['failure_reason'] or '未达标'}）",
            )
            case_controls = self.data["case0_controls"]
            edges = np.arange(len(case_controls) + 1) * dt
            ax_counts.stairs(case_controls * gear, edges, color="#b91c1c")
            ax_counts.set(
                ylabel="电机力（N）",
                xlabel="仿真时间（s）",
                title="失败回合电机输入",
            )
        else:
            for ax, title in (
                (ax_case, "失败案例：无（全部回合通过验收）"),
                (ax_counts, "失败回合电机输入：无"),
            ):
                ax.text(
                    0.5,
                    0.5,
                    "本记录没有 PPO 失败回合",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set(title=title)
        for ax in (ax_pair, ax_times, ax_case, ax_counts):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        aggregate = report["ppo_evaluation"]["aggregate"]
        baseline = report["baseline"]
        if mode == "training":
            lines = ["① 这一幕在追踪：训练过程本身\n"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步，"
                f"共 {report['training']['total_env_steps'] / 1000:.0f}k 步；"
                f"墙钟 {report['training']['wall_time_s_total']:.0f} s"
            )
            for record in report["ppo_evaluation"]["per_training_seed"]:
                first = record["first_successful_eval_steps"]
                first_text = f"{first / 1000:.0f}k 步" if first is not None else "从未出现"
                lines.append(
                    f"  种子 {record['seed_index']}：末段奖励均值 "
                    f"{record['final_reward_mean']:.3f}，首次评估成功 {first_text}"
                )
            lines.append("\n  评估 = 均值动作、精确静止下方初态、第 7 课验收；")
            lines.append("  参照：悬挂 ≈0.25/步（存活项），直立平衡 ≈1.2/步。")
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同一下方初态，模型知识 vs 只凭奖励\n"]
            if baseline["median_settled_at_s"] is not None:
                lines.append(
                    f"  基线（能量+LQR 零样本）：{baseline['successes']}/"
                    f"{baseline['episodes']}，稳定 {baseline['median_settled_at_s']:.2f} s"
                )
            else:
                lines.append(f"  基线：{baseline['successes']}/{baseline['episodes']}")
            if baseline["median_settled_at_s"] is not None:
                lines.append(f"    输入峰值中位 {baseline['median_peak_abs_motor_force_n']:.0f} N")
            lines.append(
                f"  PPO：{aggregate['successes']}/{aggregate['episodes']} 通过"
                f"（每种子 {aggregate['successes_per_seed']}）"
            )
            det = report["ppo_evaluation"]["per_training_seed"][0]["deterministic"]
            det_text = (
                f"稳定于 {det['settled_at_s']:.2f} s"
                if det["settled_at_s"] is not None
                else ("触发出界失败" if det["terminated"] else "未通过验收")
            )
            lines.append(f"  PPO 均值动作单回合：{det_text}，回报 {det['return']:.2f}")
            lines.append("\n  同口径 = 同一验收、同一初态、同一 30 s 上限。")
        else:
            push = report["push_test"]
            lines = ["③ 这一幕在扰动：PPO 没见过的 ±200 N 推力\n"]
            baseline_push = push["baseline"]
            lines.append(f"  基线恢复 {baseline_push['successes']}/{baseline_push['episodes']}")
            if baseline_push.get("median_recovery_after_push_end_s") is not None:
                lines.append(
                    f"    推力结束后恢复中位 {baseline_push['median_recovery_after_push_end_s']:.2f} s"
                )
            for record in push["ppo_per_seed"]:
                lines.append(
                    f"  PPO 种子 {record['seed_index']}：恢复 {record['successes']}/"
                    f"{record['episodes']}"
                )
            counts = report["failure_analysis"]
            lines.append(
                f"  PPO 评估失败计数：出界 {counts['ppo_eval_counts']['cart_safety_boundary']}，"
                f"速度 {counts['ppo_eval_counts']['velocity_safety_boundary']}，"
                f"超时未稳 {counts['ppo_eval_counts']['timeout_without_settling']}"
            )
            lines.append("\n  同一推力方案、同一下方初态配对施加；PPO 输出均值动作。")
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
            self.draw_push()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：奖励曲线、周期验收与训练预算",
            "trajectories": "② 这一幕在对照：第七课手工控制器与只凭奖励训练出的策略",
            "push": "③ 这一幕在扰动：训练分布之外的推力与失败形态",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第二十九课 · 强化学习入口：手写 PPO 摆起 vs 能量整形基线")
    root.geometry("1600x760")
    root.minsize(1380, 660)
    PpoDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
