"""Lesson 30 viewer: residual RL - the base keeps energy, RL learns coordination.

Three static modes share one lesson-30 recording (npz + summary):
1. training curves: per-update reward and periodic down-start evaluation for
   each residual budget a in {25, 50, 100} N;
2. the exact down start: lesson-7 baseline versus the trained residual policy
   (mean action), the handoff segment highlighted, and the three-way success
   table (baseline / lesson-29 pure PPO / residual);
3. residual usage: the |u_RL| distribution against each budget, the paired
   +/-200 N push recovery, and the featured failure case.
Layout and Esc handling follow the lesson-28/29 demos: every mode calls
fig.clear() first and every quoted number is traceable to summary.json /
trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.residual_swingup import EXPERIMENT, expected_npz_keys

DEFAULT_RESULTS = "results/residual_swingup_2026-09-06"


def short_three_way(report):
    """Three-way rows with compact labels (long labels overlap on bar axes)."""
    rows = []
    for index, row in enumerate(report["three_way_comparison"]):
        label = ("基线", "纯PPO", "残差")[index] if index < 3 else "残差"
        if index >= 2:
            limit = report["sweep"][index - 2]["limit_n"]
            label = f"残差 a={limit:.0f} N"
        rows.append({**row, "label": label})
    return rows


def per_seed_successes(data, amp_index, seeds, count):
    """Successes per training seed derived from the archive (not the summary)."""
    terminated = data[f"eval_terminated_{amp_index}"]
    settled = data[f"eval_settled_s_{amp_index}"]
    if terminated.shape != (seeds * count,) or settled.shape != (seeds, count):
        raise ValueError("Archive evaluation shapes disagree with the summary")
    recovered = (~terminated) & (~np.isnan(settled.ravel()))
    return recovered.reshape(seeds, count).sum(axis=1).astype(int).tolist()


def load_replays(directory):
    """Validate a lesson-30 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-30 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["learn_std"]:
        raise ValueError("Unexpected network contract")
    if (
        report["protocol"]["observation"]["features"]
        != "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]"
    ):
        raise ValueError("Unexpected observation contract")
    if not report["guard"]["bitwise_identical_states"]:
        raise ValueError("Recording claims a broken baseline guard")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    if not np.array_equal(data["guard_states"], data["baseline_states"]):
        raise ValueError("Guard states are not bitwise identical to the baseline run")
    seeds = report["training"]["train_seeds"]
    for index, entry in enumerate(report["sweep"]):
        count = entry["stochastic"]["episodes"] // seeds
        derived = per_seed_successes(data, index, seeds, count)
        if derived != [int(v) for v in entry["stochastic"]["successes_per_seed"]]:
            raise ValueError("Residual successes disagree with the archive")
        residuals = data[f"det_residuals_{index}_0"]
        limit_norm = entry["limit_norm"]
        if residuals.size and np.max(np.abs(residuals)) > limit_norm + 1e-6:
            raise ValueError(f"Residual budget exceeded in the archive (a={entry['limit_n']} N)")
    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        states = data[f"case{index}_states"]
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        max_x = float(np.max(np.abs(states[:, 0])))
        if abs(max_x - case["max_abs_cart_position_m"]) > 1e-4:
            raise ValueError(f"Featured case {index} cart excursion disagrees with its arrays")
    return {"report": report, **data}


class ResidualDemo:
    """Residual RL versus the hand-designed lesson-7 baseline."""

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
            text="第三十课 · 残差强化学习：底座管能量注入，PPO 只学限幅残差",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "第七课能量整形+LQR 原样做底座，第二十九课手写 PPO 输出限幅残差 u = clip(u_energy + clip(u_RL, ±a), ±300 N)。\n"
                "三个模式共用同一条正式记录：① 训练曲线（三档限幅）② 同一下方初态对照（交接段高亮）③ 残差幅值与推力；"
                "全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：三档残差预算", "training"),
            ("② 同一下方初态：基线 vs 残差（交接段）", "trajectories"),
            ("③ 残差幅值分布与 ±200 N 推力对照", "usage"),
            ("④ 最佳过程回放：基线成功 vs 残差最佳失败回合", "replay"),
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
        # would push the status and caption labels out of the window.
        self.stats = ttk.Label(middle, width=48, anchor="nw", justify="left", wraplength=400)
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
                "残差奖励的控制代价只计 RL 自选的残差；PPO 在限幅前的采样上更新，训练与执行都限幅；"
                "所有评估固定在精确静止下方初态，成功率为有限样本计数"
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
        """① per-update reward curves and periodic evaluations, per budget."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_eval = self.fig.subplots(1, 2)
        colors = ("#2563eb", "#0f766e", "#b45309")
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        for index, entry in enumerate(report["sweep"]):
            color = colors[index % len(colors)]
            curves = np.stack(
                [self.data[f"reward_curve_{index}_{seed}"] for seed in range(seeds)], axis=0
            )
            for seed in range(seeds):
                ax_reward.plot(updates, curves[seed], alpha=0.3, linewidth=0.8, color=color)
            ax_reward.plot(
                updates,
                curves.mean(axis=0),
                color=color,
                linewidth=1.8,
                label=f"a={entry['limit_n']:.0f} N（{seeds} 种子均值）",
            )
        ax_reward.set(
            xlabel="PPO 更新轮次",
            ylabel="批内平均原始奖励（每步）",
            title="残差训练奖励：细线 = 单个种子",
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
            ax_eval.plot(
                steps, success.mean(axis=0), "o-", color=color, label=f"a={entry['limit_n']:.0f} N"
            )
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
        """② down start: baseline vs residual, handoff highlighted, 3-way bars."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        gear = report["protocol"]["actuator_gear"]
        ref_theta = report["protocol"]["reference_state"][1]
        boundary = report["protocol"]["cart_failure_boundary_m"]
        featured = report["featured_amplitude_index"]
        entry = report["sweep"][featured]
        det_states = self.data[f"det_states_{featured}_0"]
        det_controls = self.data[f"det_controls_{featured}_0"]
        det_residual = self.data[f"det_residuals_{featured}_0"]
        baseline_states = self.data["baseline_states"]
        baseline_controls = self.data["baseline_controls"]
        handoff_start = report["guard"]["capture_time_s"]
        handoff_end = report["guard"]["settled_at_s"]
        ax_pole, ax_cart, ax_force, ax_rate = self.fig.subplots(2, 2).reshape(-1)
        if handoff_start is not None and handoff_end is not None:
            for ax in (ax_pole, ax_cart, ax_force):
                ax.axvspan(handoff_start, handoff_end, alpha=0.15, color="#fbbf24")
        baseline_ts = np.arange(len(baseline_states)) * dt
        det_ts = np.arange(len(det_states)) * dt
        ax_pole.plot(
            baseline_ts,
            np.cos(baseline_states[:, 1] - ref_theta),
            "--",
            color="gray",
            label="基线（能量+LQR，零样本）",
        )
        ax_pole.plot(
            det_ts,
            np.cos(det_states[:, 1] - ref_theta),
            color="#0f766e",
            label=f"残差 a={entry['limit_n']:.0f} N（均值动作）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            title="摆起轨迹（黄带 = 基线交接段）",
        )
        ax_pole.legend(fontsize=7, loc="lower right")
        ax_cart.plot(baseline_ts, baseline_states[:, 0], "--", color="gray")
        ax_cart.plot(det_ts, det_states[:, 0], color="#2563eb")
        for bound in (-boundary, boundary):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.set(
            ylabel="小车位置（m）",
            title=f"小车位置与 ±{boundary:.1f} m 边界",
        )
        baseline_edges = np.arange(len(baseline_controls) + 1) * dt
        det_edges = np.arange(len(det_controls) + 1) * dt
        ax_force.stairs(baseline_controls * gear, baseline_edges, color="gray", label="基线")
        ax_force.stairs(det_controls * gear, det_edges, color="#2563eb", label="残差合计")
        residual_ts = np.arange(len(det_residual)) * dt
        ax_force.plot(
            residual_ts, det_residual * gear, color="#b45309", linewidth=1.0, label="残差 u_RL"
        )
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="电机输入与残差分量",
        )
        ax_force.legend(fontsize=7, loc="upper right")
        labels = [row["label"] for row in short_three_way(report)]
        successes = [row["successes"] for row in report["three_way_comparison"]]
        totals = [row["episodes"] for row in report["three_way_comparison"]]
        bar_colors = ("#64748b", "#b91c1c", "#2563eb", "#0f766e", "#b45309")
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=[bar_colors[i % len(bar_colors)] for i in range(len(labels))],
            width=0.6,
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
            ylim=(0, 112),
            title="三方对照成功率（第 7 课验收）",
        )
        ax_rate.tick_params(axis="x", labelsize=7)
        for ax in (ax_pole, ax_cart, ax_force, ax_rate):
            ax.grid(alpha=0.2)

    def draw_usage(self):
        """③ |u_RL| distributions, budget usage bars, paired pushes, failure."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        gear = report["protocol"]["actuator_gear"]
        ref_theta = report["protocol"]["reference_state"][1]
        seeds = report["training"]["train_seeds"]
        colors = ("#2563eb", "#0f766e", "#b45309")
        ax_hist, ax_usage, ax_push, ax_case = self.fig.subplots(2, 2).reshape(-1)
        bins = np.linspace(0.0, 1.05 * max(report["protocol"]["residual_limits_n"]), 42)
        for index, entry in enumerate(report["sweep"]):
            pool = (
                np.concatenate(
                    [self.data[f"det_residuals_{index}_{seed}"] for seed in range(seeds)]
                )
                * gear
            )
            ax_hist.hist(
                np.abs(pool),
                bins=bins,
                alpha=0.55,
                color=colors[index % len(colors)],
                label=f"a={entry['limit_n']:.0f} N",
            )
        for index, entry in enumerate(report["sweep"]):
            ax_hist.axvline(
                entry["limit_n"], color=colors[index % len(colors)], linestyle=":", linewidth=1.2
            )
        ax_hist.set(
            xlabel="|残差 u_RL|（N）",
            ylabel="步数",
            title="残差幅值分布（点线 = 预算上限）",
        )
        ax_hist.legend(fontsize=7, loc="upper right")
        xs = np.arange(len(report["sweep"]))
        at_limits = [
            entry["residual_stats"]["deterministic"]["fraction_at_limit"]
            for entry in report["sweep"]
        ]
        means = [
            entry["residual_stats"]["deterministic"]["mean_abs_n"] for entry in report["sweep"]
        ]
        bars = ax_usage.bar(xs, at_limits, color=[colors[i % len(colors)] for i in xs], width=0.55)
        for bar, mean_value in zip(bars, means, strict=True):
            ax_usage.annotate(
                f"均值 {mean_value:.1f} N",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax_usage.set(
            xticks=xs,
            xticklabels=[f"a={entry['limit_n']:.0f} N" for entry in report["sweep"]],
            ylabel="触限步占比",
            ylim=(0, 1.12),
            title="预算使用（|u_RL| ≥ 95% 预算）",
        )
        plans = report["push_test"]["plans"]
        baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
        featured = report["featured_amplitude_index"]
        featured_entry = report["sweep"][featured]
        featured_recovery = report["push_test"]["per_amplitude"][featured]["recovery_times_s"]
        plan_indices = np.arange(len(plans))
        ax_push.plot(
            plan_indices,
            [np.nan if v is None else v for v in baseline_recovery],
            "s",
            color="#64748b",
            label="基线",
        )
        # the flat per-amplitude list is seed-major: one marker series per seed
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
                color=colors[featured % len(colors)],
                label=f"残差 a={featured_entry['limit_n']:.0f} N（种子 {seed}）",
            )
        ax_push.set(
            xlabel="推力方案编号",
            ylabel="推力结束后恢复时间（s）",
            title="±200 N 配对推力恢复（缺口 = 未恢复）",
        )
        ax_push.legend(fontsize=7, loc="upper left")
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
            # records annotated with the scan amplitude show it; otherwise the
            # title stays silent instead of guessing
            limit_text = f"a={case['limit_n']:.0f} N，" if case.get("limit_n") is not None else ""
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{limit_text}（{case['failure_reason'] or '未达标'}）",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有残差失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例：无（全部回合通过验收）")
        for ax in (ax_hist, ax_usage, ax_push, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        if mode == "training":
            lines = ["① 这一幕在追踪：残差训练本身\n"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步，"
                f"共 {report['training']['total_env_steps'] / 1000:.0f}k 步"
            )
            lines.append(f"  墙钟 {report['training']['wall_time_s_total']:.0f} s\n")
            for entry in report["sweep"]:
                for record in entry["training"]:
                    first = record["first_successful_eval_steps"]
                    first_text = f"{first / 1000:.0f}k 步" if first is not None else "从未出现"
                    lines.append(
                        f"  a={entry['limit_n']:.0f} N 种子 {record['seed_index']}："
                        f"奖励 {record['final_reward_mean']:.3f}"
                    )
                    lines.append(f"    首次评估成功：{first_text}")
            lines.append("\n  底座消掉能量探索；残差只学交接。")
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同一下方初态\n"]
            baseline = report["baseline"]
            lines.append(
                f"  基线（零样本）：{baseline['successes']}/{baseline['episodes']}"
                f"，稳定 {baseline['median_settled_at_s']:.2f} s"
            )
            lines.append(f"    输入峰值中位 {baseline['median_peak_abs_motor_force_n']:.0f} N")
            guard = report["guard"]
            lines.append(f"  a=0 守卫：逐位一致={guard['bitwise_identical_states']}")
            lines.append(f"    切入 LQR {guard['capture_time_s']:.2f} s")
            featured = report["featured_amplitude_index"]
            entry = report["sweep"][featured]
            det = entry["deterministic"][0]
            det_text = (
                f"稳定于 {det['settled_at_s']:.2f} s"
                if det["settled_at_s"] is not None
                else ("触发出界失败" if det["terminated"] else "未通过验收")
            )
            lines.append(f"  残差 a={entry['limit_n']:.0f} N 单回合：{det_text}")
            lines.append(f"    回报 {det['return']:.2f}")
            lines.append("\n  三方对照（同口径验收）：")
            for row in short_three_way(report):
                lines.append(f"    {row['label']}：{row['successes']}/{row['episodes']}")
        else:
            lines = ["③ 这一幕在追问：预算用到哪了\n"]
            for entry in report["sweep"]:
                stats = entry["residual_stats"]["deterministic"]
                lines.append(
                    f"  a={entry['limit_n']:.0f} N：均值 {stats['mean_abs_n']:.1f} N，"
                    f"触限 {stats['fraction_at_limit'] * 100:.1f}%"
                )
            push = report["push_test"]
            lines.append(
                f"  推力：基线 {push['baseline']['successes']}/{push['baseline']['episodes']}"
            )
            for item in push["per_amplitude"]:
                lines.append(
                    f"    a={item['limit_n']:.0f} N：{item['successes']}/{item['episodes']}"
                )
            counts = report["failure_analysis"]["eval_counts"]
            lines.append(
                f"  失败计数：出界 {counts['cart_safety_boundary']}，"
                f"超时未稳 {counts['timeout_without_settling']}"
            )
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "training":
            self.draw_training()
        elif mode == "trajectories":
            self.draw_trajectories()
        elif mode == "replay":
            self.draw_replay()
        else:
            self.draw_usage()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：三档残差预算下的训练曲线与周期验收",
            "trajectories": "② 这一幕在对照：能量底座 vs 底座+限幅残差的交接段",
            "usage": "③ 这一幕在追问：RL 实际用了多少残差预算，推力下是否守住",
            "replay": "④ 这一幕在回放：基线 4.76 s 抓取 vs 残差最佳失败回合（max|cart|≈1.85 m）",
        }[mode]
        if mode == "replay":
            self.status.configure(text=status + "｜播放/暂停/单步/调速。按 Esc 退出。")
        else:
            self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches

    def draw_replay(self):
        """④ 2D replay: baseline success vs the best failed residual run."""
        from embodied_learning._replay2d import Replay2D

        baseline = self.data["baseline_states"]
        # best episode by mean_return (per seed)
        # residual sweeps 3 amps × 3 seeds = 9 eval trajectories; pick highest determin return
        amp = 0  # use the first amplitude (a=25N) for selection
        dets = self.report["sweep"][amp]["deterministic"]
        best_idx = int(np.argmax([d["return"] for d in dets]))
        eval_states = self.data[f"det_states_0_{best_idx}"]
        ret = float(dets[best_idx]["return"])
        if not hasattr(self, "_replay"):
            self._replay = Replay2D(self.root, self.fig, on_close=None)
        self._replay.setup_axes([
            (None, baseline, "#0f766e", "基线（第 7 课，4.76 s 抓取）"),
            (None, eval_states, "#b91c1c",
             f"残差 a=25N best determin 回合（seed {best_idx}，return={ret:.0f}）"),
        ])
        self._replay.set_step(0)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第三十课 · 残差强化学习：能量整形底座 + PPO 残差")
    root.geometry("1600x760")
    root.minsize(1380, 660)
    ResidualDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
