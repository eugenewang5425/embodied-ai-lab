"""Lesson 38 viewer: combo swing-up - energy base + chunked multi-modal residual.

Three static modes share one lesson-38 recording (npz + summary):
1. the training curves: per-update reward and the periodic deterministic
   down-start evaluation for each residual budget a in {25, 50} N - the
   headline question: did the combo keep the base's 20/20 while training?
2. the trajectory comparison on the exact down start: the lesson-7 baseline
   versus the lesson-30 naive residual (cited from that official record, shown
   when it is readable) versus this lesson's combo, with the cart position
   against the failure boundary, the motor input with its residual component,
   and the four-way success table;
3. residual usage: the |u_residual| distribution against each budget, the
   at-limit fraction against the lesson-30 bang-bang precedent, the paired
   +/-200 N push recovery and the featured failure case.

Layout and Esc handling follow the lesson-28..37 demos (docs/26 section 6):
every mode calls fig.clear() first, redraw draws synchronously, every quoted
number is traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.combo_swingup import EXPERIMENT, expected_npz_keys

DEFAULT_RESULTS = "results/combo_swingup_2026-09-06"
DEFAULT_LESSON30 = "results/residual_swingup_2026-09-06"


def failure_label(case):
    reason = case.get("failure_reason") or ""
    short = {
        "cart_safety_boundary": "出界",
        "velocity_safety_boundary": "超速",
        "timeout_without_settling": "超时未稳",
        "nonfinite_state": "数值发散",
        "numerical_warning": "数值警告",
    }
    if not reason:
        return "超时未稳"
    return short.get(reason, reason or "未达标")


def null_to_text(value):
    return f"{value:.2f} s" if value is not None else "未到达"


def per_seed_successes(data, amp_index, seeds, count):
    """Successes per training seed derived from the archive (not the summary)."""
    terminated = data[f"eval_terminated_{amp_index}"]
    settled = data[f"eval_settled_s_{amp_index}"]
    if terminated.shape != (seeds * count,) or settled.shape != (seeds, count):
        raise ValueError("Archive evaluation shapes disagree with the summary")
    recovered = (~terminated) & (~np.isnan(settled.ravel()))
    return recovered.reshape(seeds, count).sum(axis=1).astype(int).tolist()


def load_replays(directory):
    """Validate a lesson-38 recording; returns the data dict for the demo.

    Tamper routes rejected here: (1) the npz SHA-256 against the summary and
    the archive key set against the implied key set; (2) the per-seed
    stochastic successes recomputed from the archive against the summary;
    (3) the a=0 guard states bitwise against the baseline run; (4) the
    residual budget and the featured-case excursion recomputed from arrays.
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-38 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["n_experts"] != 2 or hyper["chunk_h"] != 8:
        raise ValueError("Unexpected network contract")
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
    if np.max(np.abs(data["guard_residuals"])) > 0.0:
        raise ValueError("Guard residuals are not identically zero")
    seeds = report["training"]["train_seeds"]
    for index, entry in enumerate(report["sweep"]):
        count = entry["stochastic"]["episodes"] // seeds
        derived = per_seed_successes(data, index, seeds, count)
        if derived != [int(v) for v in entry["stochastic"]["successes_per_seed"]]:
            raise ValueError("Combo successes disagree with the archive")
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
    baseline = report["baseline"]
    per_episode = [row["settled_at_s"] for row in baseline["per_episode"]]
    if baseline["deterministic_identical_repeats"] and per_episode:
        median = float(np.median(per_episode))
        if abs(median - baseline["median_settled_at_s"]) > 1e-9:
            raise ValueError("Baseline median settle time disagrees with its episodes")
    return {"report": report, **data}


def load_naive_residual_30(directory=DEFAULT_LESSON30):
    """The lesson-30 a=25 N mean-action trajectory, cited for mode 2 (optional).

    Returns a dict {states, controls, residuals, limit_n, successes, episodes,
    source} or None when the official lesson-30 record is not readable.  This
    is CITED material (clearly labeled in the panel), not part of this
    lesson's tamper-checked recording.
    """
    directory = Path(directory)
    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary.get("experiment") != "residual_swingup_lesson30":
            return None
        entry = summary["sweep"][0]  # a = 25 N, the shared budget
        with np.load(directory / "trajectories.npz", allow_pickle=False) as npz:
            states = npz["det_states_0_0"].copy()
            controls = npz["det_controls_0_0"].copy()
            residuals = npz["det_residuals_0_0"].copy()
        return {
            "states": states,
            "controls": controls,
            "residuals": residuals,
            "limit_n": float(entry["limit_n"]),
            "successes": int(entry["stochastic"]["successes"]),
            "episodes": int(entry["stochastic"]["episodes"]),
            "fraction_at_limit": float(
                entry["residual_stats"]["deterministic"]["fraction_at_limit"]
            ),
            "source": "results/residual_swingup_2026-09-06（第 30 课正式记录）",
        }
    except (OSError, KeyError, json.JSONDecodeError):
        return None


class ComboDemo:
    """Combo swing-up: the lesson-7 base under the lesson-37 chunked residual."""

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
        self.lesson30 = None  # loaded lazily in mode 2 (cited, optional)
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第三十八课 · 倒立摆组合学习：能量整形底座 + 多峰块残差",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "底座 = 第 7 课 HybridSwingupController（逐字节不改）；残差 = 第 37 课多峰块策略\n"
                "（K=2、H=8，限幅 a∈{25,50} N，σ=a/2）；u = clip(u_base + clip(u_res, ±a), ±300 N)。"
                "三模式共用同一正式记录，全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线（奖励 + 周期评估）", "training"),
            ("② 基线 vs 朴素残差 vs 组合轨迹", "trajectories"),
            ("③ 残差幅值分布 + 扰动恢复", "usage"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left", wraplength=430)
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
                "不劣化判据：每种子成功率 ≥ 18/20（基线 20/20）；训练 = 块级 PPO"
                "（混合密度重要性比），奖励 = 顶部重（2·(1+cosα)/2）+ 底部轻罚（0.25）+ 残差代价"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """1 reward per update + periodic deterministic evaluation."""
        self.fig.clear()
        report = self.report
        data = self.data
        seeds = report["training"]["train_seeds"]
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        colors = ("#2563eb", "#0f766e", "#b45309", "#7c3aed")
        ax_reward, ax_eval = self.fig.subplots(1, 2)
        for index, entry in enumerate(report["sweep"]):
            color = colors[index % len(colors)]
            curves = np.stack([data[f"reward_curve_{index}_{s}"] for s in range(seeds)])
            for seed in range(seeds):
                ax_reward.plot(updates, curves[seed], alpha=0.25, linewidth=0.8, color=color)
            ax_reward.plot(
                updates,
                curves.mean(axis=0),
                color=color,
                linewidth=1.8,
                label=f"a={entry['limit_n']:.0f} N（{seeds} 种子均值）",
            )
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
                ax_eval.plot(steps, success[seed], "o--", markersize=4, alpha=0.55, color=color)
            ax_eval.plot(
                steps,
                success.mean(axis=0),
                "o-",
                color=color,
                label=f"a={entry['limit_n']:.0f} N",
            )
        ax_reward.set(
            xlabel="PPO 更新轮次（块级决策）",
            ylabel="批内平均原始奖励（每环境步）",
            title="①组合训练奖励：细线 = 单个种子",
        )
        ax_reward.legend(fontsize=8)
        ax_eval.set(
            xlabel="环境步数（×1000）",
            ylabel="下方初态验收通过（1=成功）",
            ylim=(-0.08, 1.35),
            yticks=[0, 0.5, 1],
            title="训练中周期确定性评估（单回合）",
        )
        ax_eval.legend(fontsize=8)
        for ax in (ax_reward, ax_eval):
            ax.grid(alpha=0.2)

    def draw_trajectories(self):
        """2 baseline vs lesson-30 naive residual vs combo, same down start."""
        self.fig.clear()
        if self.lesson30 is None:
            self.lesson30 = load_naive_residual_30()
        report = self.report
        data = self.data
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        featured = report["featured_amplitude_index"]
        entry = report["sweep"][featured]
        ax_pole, ax_cart, ax_force, ax_rate = self.fig.subplots(2, 2).reshape(-1)
        baseline_states = data["baseline_states"]
        baseline_controls = data["baseline_controls"]
        det_states = data[f"det_states_{featured}_0"]
        det_controls = data[f"det_controls_{featured}_0"]
        det_residual = data[f"det_residuals_{featured}_0"]
        ts = np.arange(len(baseline_states)) * dt
        ax_pole.plot(
            ts,
            np.cos(baseline_states[:, 1] - ref_theta),
            "--",
            color="#64748b",
            label="基线（第 7 课，20/20）",
        )
        if self.lesson30 is not None:
            l30 = self.lesson30
            ax_pole.plot(
                np.arange(len(l30["states"])) * dt,
                np.cos(l30["states"][:, 1] - ref_theta),
                color="#b91c1c",
                linewidth=1.2,
                label=f"朴素残差 a={l30['limit_n']:.0f} N（第 30 课，{l30['successes']}/{l30['episodes']}）",
            )
        ax_pole.plot(
            np.arange(len(det_states)) * dt,
            np.cos(det_states[:, 1] - ref_theta),
            color="#0f766e",
            linewidth=1.3,
            label=f"组合 a={entry['limit_n']:.0f} N（本课，种子 0）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            title="②同一下方初态：基线 vs 朴素残差 vs 组合",
        )
        ax_pole.legend(fontsize=6.5, loc="lower right")
        for bound in (-2.4, 2.4):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.plot(ts, baseline_states[:, 0], "--", color="#64748b", label="基线")
        ax_cart.plot(
            np.arange(len(det_states)) * dt,
            det_states[:, 0],
            color="#0f766e",
            label=f"组合 a={entry['limit_n']:.0f} N",
        )
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title="小车位置（红点线 = ±2.4 m 边界）",
        )
        ax_cart.legend(fontsize=6.5)
        ax_force.stairs(
            baseline_controls * 100.0,
            np.arange(len(baseline_controls) + 1) * dt,
            color="#64748b",
            label="基线",
        )
        ax_force.stairs(
            det_controls * 100.0,
            np.arange(len(det_controls) + 1) * dt,
            color="#0f766e",
            label="组合合计输入",
        )
        ax_force.plot(
            np.arange(len(det_residual)) * dt,
            det_residual * 100.0,
            color="#b45309",
            linewidth=1.0,
            label="其中残差 u_res",
        )
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="电机输入与残差分量（±300 N 总限幅）",
        )
        ax_force.legend(fontsize=6.5, loc="upper right")
        rows = report["four_way_comparison"]
        labels = []
        for row in rows:
            if row["label"].startswith("基线"):
                labels.append("基线\n第7课")
            elif row["label"].startswith("朴素残差"):
                labels.append("朴素残差\n第30课")
            elif row["label"].startswith("多峰块"):
                labels.append("块无底座\n第37课")
            else:
                labels.append(row["label"].split("a=")[1].split(" ")[0] + " N\n本课")
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        bar_colors = ("#64748b", "#b91c1c", "#7c3aed", "#2563eb", "#0f766e")
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
            title="四方对照成功率（第 7 课同口径验收）",
        )
        ax_rate.tick_params(axis="x", labelsize=7)
        for ax in (ax_pole, ax_cart, ax_force, ax_rate):
            ax.grid(alpha=0.2)

    def draw_usage(self):
        """3 residual usage, at-limit fraction, paired pushes, failure case."""
        self.fig.clear()
        report = self.report
        data = self.data
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        featured = report["featured_amplitude_index"]
        entry = report["sweep"][featured]
        seeds = report["training"]["train_seeds"]
        colors = ("#2563eb", "#0f766e", "#b45309")
        ax_hist, ax_limit, ax_push, ax_case = self.fig.subplots(2, 2).reshape(-1)
        pools = [
            np.concatenate([data[f"det_residuals_{index}_{s}"] for s in range(seeds)]) * 100.0
            for index in range(len(report["sweep"]))
        ]
        bins = np.linspace(0.0, 1.05 * max(report["protocol"]["residual_limits_n"]), 42)
        for index, item in enumerate(report["sweep"]):
            ax_hist.hist(
                np.abs(pools[index]),
                bins=bins,
                alpha=0.55,
                color=colors[index % len(colors)],
                label=f"a={item['limit_n']:.0f} N",
            )
            ax_hist.axvline(
                item["limit_n"], color=colors[index % len(colors)], linestyle=":", linewidth=1.2
            )
        ax_hist.set(
            xlabel="|残差 u_res|（N，确定性块回合合并种子）",
            ylabel="步数",
            title="③残差幅值分布（点线 = 各档预算上限）",
        )
        ax_hist.legend(fontsize=8)
        xs = np.arange(len(report["sweep"]))
        det_means = [
            item["residual_stats"]["deterministic"]["mean_abs_n"] for item in report["sweep"]
        ]
        at_limits = [
            item["residual_stats"]["deterministic"]["fraction_at_limit"] for item in report["sweep"]
        ]
        bars = ax_limit.bar(xs, at_limits, color=[colors[i % len(colors)] for i in xs], width=0.55)
        for bar, mean_value in zip(bars, det_means, strict=True):
            ax_limit.annotate(
                f"均值 {mean_value:.1f} N",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax_limit.axhline(0.937, color="#b91c1c", linestyle="--", linewidth=1.0)
        ax_limit.text(
            len(report["sweep"]) - 0.5,
            0.965,
            "第 30 课朴素残差 93.7%",
            color="#b91c1c",
            fontsize=7,
            ha="right",
            va="bottom",
        )
        ax_limit.set(
            xticks=xs,
            xticklabels=[f"a={item['limit_n']:.0f} N" for item in report["sweep"]],
            ylabel="触限步占比",
            ylim=(0, 1.12),
            title="残差预算使用（|u_res| ≥ 95% 预算的步占比）",
        )
        plans = report["push_test"]["plans"]
        plan_indices = np.arange(len(plans))
        baseline_recovery = report["push_test"]["baseline"]["recovery_times_s"]
        featured_recovery = entry["push"]["recovery_times_s"]
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
        for seed in range(seeds):
            ax_push.plot(
                plan_indices,
                seed_recovery[seed],
                "o",
                markersize=4,
                alpha=0.7,
                color=colors[featured % len(colors)],
                label=f"组合 a={entry['limit_n']:.0f} N（种子 {seed}）",
            )
        ax_push.set(
            xlabel="推力方案编号",
            ylabel="推力结束后恢复时间（s）",
            title="±200 N 配对推力恢复（缺口 = 未恢复）",
        )
        ax_push.legend(fontsize=8)
        cases = report["failure_analysis"]["featured_cases"]
        if cases:
            case = cases[0]
            states = data["case0_states"]
            ax_case.plot(
                np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c"
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            limit_text = f"a={case['limit_n']:.0f} N，" if case.get("limit_n") is not None else ""
            ax_case.set(
                ylabel="杆端相对高度",
                ylim=(-1.1, 1.1),
                xlabel="仿真时间（s）",
                title=f"失败案例：{limit_text}（{failure_label(case)}）",
            )
        else:
            det_states = data[f"det_states_{featured}_0"]
            ax_case.plot(
                np.arange(len(det_states)) * dt,
                np.cos(det_states[:, 1] - ref_theta),
                color="#0f766e",
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                ylim=(-1.1, 1.1),
                xlabel="仿真时间（s）",
                title="失败案例：无（展示 featured 档确定性轨迹）",
            )
        for ax in (ax_hist, ax_limit, ax_push, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        baseline = report["baseline"]
        if mode == "training":
            lines = ["① 这一幕在追踪：训练是否保住底座并学到东西"]
            lines.append(
                f"  预算 {report['training']['env_steps_per_seed']} 环境步/种子，"
                f"总墙钟 {report['training']['wall_time_s_total']:.0f} s"
            )
            for index, entry in enumerate(report["sweep"]):
                record = entry["training"][0]
                reward_first = float(self.data[f"reward_curve_{index}_0"][0])
                lines.append(
                    f"  a={entry['limit_n']:.0f} N：奖励首→末 "
                    f"{reward_first:.2f}→{record['final_reward_mean']:.2f}，随机 "
                    f"{entry['stochastic']['successes']}/{entry['stochastic']['episodes']}，"
                    f"首成步 {record['first_successful_eval_steps'] or '从未'}"
                )
            lines.append(
                f"  底座（教师闸门）：{report['teacher_verification']['successes']}/"
                f"{report['teacher_verification']['episodes']}，a=0 守卫逐位一致"
            )
        elif mode == "trajectories":
            lines = ["② 这一幕在对照：同一精确下方初态的三条轨迹"]
            lines.append(
                f"  基线：{baseline['successes']}/{baseline['episodes']}"
                f"（中位 {baseline['median_settled_at_s']:.2f} s）"
            )
            if self.lesson30 is not None:
                l30 = self.lesson30
                lines.append(
                    f"  朴素残差（第 30 课，a={l30['limit_n']:.0f} N）：{l30['successes']}/"
                    f"{l30['episodes']}，触限 {l30['fraction_at_limit'] * 100:.1f}%（引自其正式记录）"
                )
            else:
                lines.append("  朴素残差（第 30 课）：0/60（记录不可读，引自 docs/35）")
            featured = report["featured_amplitude_index"]
            entry = report["sweep"][featured]
            det = entry["deterministic"][0]
            lines.append(
                f"  组合 a={entry['limit_n']:.0f} N：确定性首达 "
                f"{null_to_text(det['first_arrival_s'])}、稳定 {null_to_text(det['settled_at_s'])}"
            )
            row37 = report["four_way_comparison"][2]
            lines.append(
                f"  四方对照（随机）：{row37['label'].split('（')[0]} "
                f"{row37['successes']}/{row37['episodes']}；组合两档判据见③模式"
            )
        else:
            lines = ["③ 这一幕在裁决：残差用得克制吗，扰动恢复呢"]
            for entry in report["sweep"]:
                stats = entry["residual_stats"]["deterministic"]
                verdict = "达标" if entry["not_degrade"] else "未达标"
                lines.append(
                    f"  a={entry['limit_n']:.0f} N：|u_res| 均值 {stats['mean_abs_n']:.1f} N，"
                    f"触限 {stats['fraction_at_limit'] * 100:.1f}%，随机 "
                    f"{entry['stochastic']['successes']}/{entry['stochastic']['episodes']}"
                    f"（判据 ≥18/20：{verdict}）"
                )
            push = report["push_test"]
            lines.append(
                f"  推力 ±200 N 配对：基线 {push['baseline']['successes']}/"
                f"{push['baseline']['episodes']}"
            )
            lines.append(
                "  "
                + "、".join(
                    f"a={item['limit_n']:.0f} N {item['successes']}/{item['episodes']}"
                    for item in push["per_amplitude"]
                )
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
        else:
            self.draw_usage()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：各档奖励曲线与训练中周期确定性评估（下方初态、单回合）",
            "trajectories": "② 这一幕在对照：基线 vs 朴素残差（第 30 课，引用）vs 组合的确定性轨迹与四方成功率",
            "usage": "③ 这一幕在裁决：残差幅值分布 / 触限率 vs 第 30 课先例 / ±200 N 配对推力 / 失败案例",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--lesson30", type=Path, default=Path(DEFAULT_LESSON30))
    args = parser.parse_args()
    data = load_replays(args.results)
    lesson30 = load_naive_residual_30(args.lesson30)
    root = tk.Tk()
    root.title("第三十八课 · 倒立摆组合学习：能量整形底座 + 多峰块残差")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    demo = ComboDemo(root, data)
    demo.lesson30 = lesson30
    demo.redraw()
    root.mainloop()


if __name__ == "__main__":
    main()
