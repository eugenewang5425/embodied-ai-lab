"""Lesson 35 viewer: hand-written numpy SAC on the down-start swing-up.

Three static modes share one lesson-35 recording (npz + summary):
1. training curves: per-update reward, the alpha trajectory (auto vs fixed
   tiers) and the policy entropy, plus the replay-buffer coverage entropy and
   the periodic down-start evaluation;
2. the exact down start: lesson-7 energy+LQR baseline versus the trained SAC
   mean-action episode, cart positions and motor commands, same-caliber
   success rates of all five comparison rows;
3. first arrival / first success statistics and the featured failure case per
   tier, with the failure counts and the push recovery scatter.
Layout and Esc handling follow the lesson-28..34 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pbrs_swingup import first_arrival_time_s
from embodied_learning.experiments.sac_swingup import EXPERIMENT, expected_npz_keys

DEFAULT_RESULTS = "results/sac_swingup_2026-09-06"


def load_replays(directory):
    """Validate a lesson-35 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-35 recording")
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
    data["_report"] = report
    tiers = report["training"]["tiers"]
    seeds = report["training"]["train_seeds"]
    for tier in tiers:
        per_seed = report["sac_evaluation"]["tiers"][tier]["per_seed"]
        success_per_seed = report["sac_evaluation"]["tiers"][tier]["aggregate"][
            "successes_per_seed"
        ]
        for seed_index in range(seeds):
            terminated = data[f"eval_terminated_{tier}_{seed_index}"]
            settled = data[f"eval_settled_s_{tier}_{seed_index}"]
            derived = int(((~terminated) & (~np.isnan(settled))).sum())
            if derived != int(success_per_seed[seed_index]):
                raise ValueError(f"SAC successes of {tier} seed {seed_index} disagree")
            record = per_seed[seed_index]
            det_states = data[f"det_states_{tier}_{seed_index}"]
            arrival = first_arrival_time_s(
                det_states,
                np.asarray(report["protocol"]["reference_state"], dtype=float),
                report["protocol"]["dt_s"],
            )
            if (arrival is None) != (record["deterministic_first_arrival_s"] is None):
                raise ValueError(f"First arrival of {tier} seed {seed_index} disagrees")
            if (
                arrival is not None
                and abs(arrival - float(record["deterministic_first_arrival_s"])) > 1e-6
            ):
                raise ValueError(f"First arrival of {tier} seed {seed_index} disagrees")
            # the return of the mean-action episode matches the stored arrays
            # (the lesson-29 reward shape; a terminated episode replaces the
            # final step's reward by -10)
            controls = data[f"det_controls_{tier}_{seed_index}"].astype(float)
            states = data[f"det_states_{tier}_{seed_index}"].astype(float)
            reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
            alphas = ((states[1:, 1] - reference[1] + np.pi) % (2 * np.pi)) - np.pi
            upright = (1.0 + np.cos(alphas)) / 2.0
            step_return = upright + 0.25 - 0.01 * controls**2
            if record["deterministic"]["terminated"]:
                return_recomputed = float(np.sum(step_return[:-1]) - 10.0)
            else:
                return_recomputed = float(np.sum(step_return))
            if abs(return_recomputed - float(record["deterministic_return"])) > 1e-4:
                raise ValueError(f"Det return of {tier} seed {seed_index} disagrees")
    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        states = data[f"case{index}_states"]
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError(f"Featured case {index} shapes disagree with the contract")
    return data


class SacDemo:
    """Reward-only SAC versus the hand-designed lesson-7 baseline."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and labels
        self.root = root
        self.data = data
        self.report = data["_report"]
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第三十五课 · 手写 numpy SAC：回放池+最大熵+孪生 Q 能否直接学会摆起",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "纯 numpy 手写 SAC（5×64×64 tanh 高斯策略＋孪生 Q＋目标网络＋自动温度）对照第七课能量整形+LQR 基线；\n"
                "奖励 = 第 29 课任务奖励（不塑形、无底座、无示教、无随机起点课程）。"
                "α 自动与 α=0.2 固定两档各 3 种子 × 50 万步。三个模式共用同一条正式记录。"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：奖励 / α 轨迹 / 熵与回放覆盖", "training"),
            ("② 同一下方初态：基线 vs SAC 轨迹与成功对照", "trajectories"),
            ("③ 首达/首成与失败案例", "outcome"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        # Claim the panel's width on the right BEFORE the expanding canvas frame
        # (the lesson-29 layout lesson: packed side=left+expand afterwards the
        # canvas consumes the whole cavity and the panel is never mapped; the
        # left frame's requested size is frozen for the same reason).
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
                "奖励/观测/无课程都是方法的一部分，已写入记录并如实讨论；"
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
        """① reward / alpha / entropy / buffer coverage curves per tier."""
        self.fig.clear()
        report = self.report
        tiers = report["training"]["tiers"]
        seeds = report["training"]["train_seeds"]
        colors = {"auto": "#0f766e", "fixed": "#b45309"}
        names = {"auto": "α 自动", "fixed": "α=0.2 固定"}
        # ONE 2 x 2 grid: two separate subplots(1, 2) calls would place the
        # second pair on top of the first (the overlap defect the mode-1
        # real-window check caught in this lesson)
        ax_reward, ax_alpha, ax_entropy, ax_cover = self.fig.subplots(2, 2).reshape(-1)
        updates = np.arange(1, len(self.data[f"reward_curve_{tiers[0]}_0"]) + 1)
        for tier in tiers:
            rewards = np.stack(
                [self.data[f"reward_curve_{tier}_{index}"] for index in range(seeds)], axis=0
            )
            for index in range(seeds):
                ax_reward.plot(
                    updates, rewards[index], alpha=0.32, linewidth=0.8, color=colors[tier]
                )
            ax_reward.plot(
                updates,
                rewards.mean(axis=0),
                color=colors[tier],
                linewidth=1.8,
                label=f"SAC {names[tier]}（{seeds} 种子均值）",
            )
        ax_reward.axhline(
            0.25, color="gray", linestyle=":", linewidth=1.0, label="悬挂不动 ≈0.25/步"
        )
        ax_reward.set(
            xlabel="SAC 更新轮次（每 32 环境步 1 次）",
            ylabel="环境奖励滑动均值（每步）",
            title="训练奖励：细线 = 单个种子",
        )
        ax_reward.legend(fontsize=7, loc="lower right")
        for tier in tiers:
            alphas = np.stack(
                [self.data[f"alpha_curve_{tier}_{index}"] for index in range(seeds)], axis=0
            )
            ax_alpha.plot(
                updates,
                alphas.mean(axis=0),
                color=colors[tier],
                linewidth=1.8,
                label=f"SAC {names[tier]}",
            )
        ax_alpha.axhline(-1.0, color="gray", linestyle=":", linewidth=1.0)
        ax_alpha.set(
            xlabel="SAC 更新轮次", ylabel="温度 α", title="α 轨迹（灰点线 = 目标熵 −1 参考）"
        )
        ax_alpha.legend(fontsize=7, loc="upper right")
        for tier in tiers:
            entropy = np.stack(
                [self.data[f"entropy_curve_{tier}_{index}"] for index in range(seeds)], axis=0
            )
            ax_entropy.plot(
                updates,
                entropy.mean(axis=0),
                color=colors[tier],
                linewidth=1.8,
                label=f"SAC {names[tier]}",
            )
        ax_entropy.set(
            xlabel="SAC 更新轮次", ylabel="策略熵（−E[log π]，nats）", title="策略熵：探索还活着吗"
        )
        ax_entropy.legend(fontsize=7, loc="upper right")
        for tier in tiers:
            cover = np.stack(
                [self.data[f"buffer_cover_entropy_curve_{tier}_{index}"] for index in range(seeds)],
                axis=0,
            )
            checkpoints = np.arange(1, cover.shape[1] + 1) * 25
            ax_cover.plot(
                checkpoints,
                cover.mean(axis=0),
                "o-",
                color=colors[tier],
                linewidth=1.4,
                label=f"SAC {names[tier]}",
            )
        ax_cover.axhline(np.log(192.0), color="gray", linestyle=":", linewidth=1.0)
        ax_cover.set(
            xlabel="环境步数（×1000）",
            ylabel="回放池覆盖熵（nats）",
            title="回放池覆盖熵（灰点线 = 满覆盖 192 格）",
        )
        ax_cover.legend(fontsize=7, loc="upper left")
        for ax in (ax_reward, ax_alpha, ax_entropy, ax_cover):
            ax.grid(alpha=0.2)

    def draw_trajectories(self):
        """② exact down start: baseline vs SAC mean-action, plus success bars."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        gear = report["protocol"]["actuator_gear"]
        ref_theta = report["protocol"]["reference_state"][1]
        boundary = report["protocol"]["cart_failure_boundary_m"]
        five_way = report["five_way_comparison"]
        tier_order = (
            "auto" if "auto" in report["training"]["tiers"] else report["training"]["tiers"][0]
        )
        featured_seed = report["sac_evaluation"]["tiers"][tier_order]["featured_seed_index"]
        ax_pole, ax_cart, ax_force, ax_rate = self.fig.subplots(2, 2).reshape(-1)
        baseline_states = self.data["baseline_states"]
        det_states = self.data[f"det_states_{tier_order}_{featured_seed}"]
        baseline_ts = np.arange(len(baseline_states)) * dt
        sac_ts = np.arange(len(det_states)) * dt
        ax_pole.plot(
            baseline_ts,
            np.cos(baseline_states[:, 1] - ref_theta),
            "--",
            color="gray",
            label="基线（能量+LQR，零样本）",
        )
        ax_pole.plot(
            sac_ts,
            np.cos(det_states[:, 1] - ref_theta),
            color="#0f766e",
            label=f"SAC（{tier_order} 档均值动作）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度", ylim=(-1.1, 1.1), title="摆起轨迹（橙带 = 杆端低于铰点）"
        )
        ax_pole.legend(fontsize=7, loc="upper left")
        ax_cart.plot(baseline_ts, baseline_states[:, 0], "--", color="gray")
        ax_cart.plot(sac_ts, det_states[:, 0], color="#2563eb")
        for bound in (-boundary, boundary):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title=f"小车位置与 ±{boundary:.1f} m 边界",
        )
        baseline_controls = self.data["baseline_controls"]
        det_controls = self.data[f"det_controls_{tier_order}_{featured_seed}"]
        ax_force.stairs(
            baseline_controls * gear,
            np.arange(len(baseline_controls) + 1) * dt,
            color="gray",
            label="基线",
        )
        ax_force.stairs(
            det_controls * gear, np.arange(len(det_controls) + 1) * dt, color="#2563eb", label="SAC"
        )
        ax_force.set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入")
        ax_force.legend(fontsize=7, loc="upper right")
        totals = [row["episodes"] for row in five_way]
        successes = [row["successes"] for row in five_way]
        labels = [row["label"].split("（")[0] for row in five_way]
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=["#64748b", "#7c3aed", "#2563eb", "#0f766e", "#b45309"],
            width=0.55,
        )
        for bar, s, t in zip(bars, successes, totals, strict=True):
            ax_rate.annotate(
                f"{s}/{t}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax_rate.set(
            ylabel="验收通过率（%）",
            ylim=(0, 118),
            title="同口径成功率（第 7 课验收）",
        )
        for ax in (ax_pole, ax_cart, ax_force, ax_rate):
            ax.grid(alpha=0.2)
            ax.tick_params(axis="x", labelsize=7)

    def draw_outcome(self):
        """③ first arrival / first success statistics and failure cases."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        tiers = report["training"]["tiers"]
        names = {"auto": "α 自动", "fixed": "α=0.2 固定"}
        ax_arr, ax_first, ax_case, ax_push = self.fig.subplots(2, 2).reshape(-1)
        five_way = report["five_way_comparison"]
        short = ("基线", "PPO", "两阶段", "SAC α自动", "SAC α=0.2")
        labels = short[: len(five_way)]
        arrivals = []
        for index, row in enumerate(five_way):
            text = row["upright_first_arrival"]
            value = None
            if isinstance(text, str) and "episodes" in text and "never" not in text:
                value = float(text.split("median ")[1].split(" s")[0])
            arrivals.append(value)
        bars = ax_arr.bar(
            labels,
            [v if v is not None else 0.0 for v in arrivals],
            color=["#64748b", "#7c3aed", "#2563eb", "#0f766e", "#b45309"],
            width=0.55,
        )
        for bar, v in zip(bars, arrivals, strict=True):
            ax_arr.annotate(
                f"{v:.2f} s" if v is not None else "—",
                (bar.get_x() + bar.get_width() / 2, 0.03 if v is None else v + 0.08),
                ha="center",
                fontsize=8,
            )
        top = max([v for v in arrivals if v is not None], default=0.0)
        ax_arr.set_ylim(0.0, max(1.0, top * 1.45))
        ax_arr.set(ylabel="直立首达中位（s）", title="直立首达（|α|≤0.3 rad）")
        rows = []
        for tier in tiers:
            per_seed = report["sac_evaluation"]["tiers"][tier]["per_seed"]
            first_success = [record["first_successful_eval_steps"] for record in per_seed]
            arrival = report["sac_evaluation"]["tiers"][tier]["arrival"]
            rows.append(
                (
                    names[tier],
                    f"{arrival['episodes_with_arrival']}/{arrival['episodes']} 到达",
                    ["从未" if step is None else f"{step // 1000}k 步" for step in first_success],
                )
            )
        ax_first.axis("off")
        text_lines = ["首达/首成（过程指标，每次周期评估均值动作单回合）："]
        for tier, arrival_text, first_success in rows:
            text_lines.append(
                f"  {tier}：直立首达 {arrival_text}；首次成功 = "
                + "、".join(first_success)
                + "（每种子）"
            )
        aggregate = {tier: report["sac_evaluation"]["tiers"][tier]["aggregate"] for tier in tiers}
        for tier in tiers:
            text_lines.append(
                f"  {tier}：训练后 {aggregate[tier]['successes']}/"
                f"{aggregate[tier]['episodes']}，稳定中位 "
                f"{aggregate[tier]['median_settled_at_s'] if aggregate[tier]['median_settled_at_s'] else '无'}"
            )
        ax_first.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax_first.transAxes,
            va="top",
            fontsize=8,
        )
        ax_first.set(title="首达/首成与训练后成功率")

        cases = report["failure_analysis"]["featured_cases"]
        if cases:
            case = cases[0]
            case_states = self.data["case0_states"]
            ax_case.plot(
                np.arange(len(case_states)) * dt,
                np.cos(case_states[:, 1] - ref_theta),
                color="#b91c1c",
                linewidth=1.0,
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例（{case.get('failure_reason') or '未达标'}）",
            )
        else:
            ax_case.set(title="失败案例：无（全部回合通过验收）")
        counts = report["sac_evaluation"]["tiers"]
        lines = ["训练后失败计数（随机评估）："]
        for tier in tiers:
            failure = counts[tier]["failure_counts"]
            lines.append(
                f"  {names[tier]}：出界 {failure['cart_safety_boundary']}、"
                f"速度 {failure['velocity_safety_boundary']}、超时未稳 {failure['timeout_without_settling']}"
            )
        push = report["sac_evaluation"]["tiers"]
        for tier in tiers:
            push_agg = push[tier]["push_aggregate"]
            lines.append(
                f"  {names[tier]}：±200 N 推力恢复 {push_agg['successes']}/{push_agg['episodes']}"
            )
        ax_push.text(
            0.02, 0.98, "\n".join(lines), transform=ax_push.transAxes, va="top", fontsize=8.5
        )
        ax_push.set(title="失败形态与推力")
        for ax in (ax_arr, ax_case):
            ax.grid(alpha=0.2)
        ax_push.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        if mode == "training":
            lines = ["① 这一幕在追踪：训练过程本身\n"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步 × "
                f"{report['training']['train_seeds']} 种子 × {len(report['training']['tiers'])} 档；"
                f"总墙钟 {report['training']['wall_time_s_total'] / 60:.1f} min"
            )
            lines.append(
                f"  配置：γ={report['protocol']['method_decisions']['gamma']}、"
                f"lr={report['protocol']['method_decisions']['lr']}、"
                f"τ={report['protocol']['method_decisions']['tau']}、"
                f"batch={report['hyperparameters']['batch_size']}、"
                f"buffer={report['hyperparameters']['buffer_size'] / 1000:.0f}k、"
                f"每 {report['hyperparameters']['update_every_env_steps']} 环境步 1 次更新"
            )
            for tier in report["training"]["tiers"]:
                record = report["sac_evaluation"]["tiers"][tier]["per_seed"][0]
                lines.append(
                    f"  种子 0（{tier}）：末段奖励 {record['final_reward_mean']:.3f}，"
                    f"末 α {record['final_alpha']:.4f}，"
                    f"回放池覆盖熵 {record['final_cover_entropy']:.2f} nats"
                )
            lines.append("\n  评估 = 均值动作、精确静止下方初态、第 7 课验收。")
            lines.append("  悬挂不动参考值 ≈0.25/步（存活项）。")
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同一下方初态，模型知识 vs 均衡后的算法\n"]
            baseline = report["baseline"]
            lines.append(
                f"  基线（能量+LQR 零样本）：{baseline['successes']}/{baseline['episodes']}，"
                f"稳定 {baseline['median_settled_at_s']:.2f} s"
            )
            for row in report["five_way_comparison"][1:]:
                lines.append(
                    f"  {row['label'].split('（')[0]}：{row['successes']}/{row['episodes']}，"
                    f"首达 {row['upright_first_arrival']}"
                )
            lines.append("\n  同口径 = 同一验收、同一初态、同一 30 s 上限。")
        else:
            lines = ["③ 这一幕在总结：首达/首成、失败形态与推力\n"]
            for tier in report["training"]["tiers"]:
                aggregate = report["sac_evaluation"]["tiers"][tier]["aggregate"]
                arrival = report["sac_evaluation"]["tiers"][tier]["arrival"]
                lines.append(
                    f"  {tier}：训练后 {aggregate['successes']}/{aggregate['episodes']}；"
                    f"直立首达 {arrival['episodes_with_arrival']}/{arrival['episodes']}"
                )
            lines.append("\n  过程指标来自每次周期评估；成功率来自训练后 20 回合/种子。")
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
            self.draw_outcome()
        self.fill_stats(mode)
        self.status.configure(
            text={
                "training": "① 这一幕在追踪：奖励、温度 α、策略熵与回放池覆盖熵",
                "trajectories": "② 这一幕在对照：第七课手工控制器与均衡后的 SAC 策略",
                "outcome": "③ 这一幕在总结：直立首达、首次成功与失败形态",
            }[mode]
            + "｜静态图，无动画。按 Esc 退出。"
        )
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第三十五课 · 手写 numpy SAC：最大熵+回放池直接摆起")
    root.geometry("1600x760")
    root.minsize(1380, 660)
    SacDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
