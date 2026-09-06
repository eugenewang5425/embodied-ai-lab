"""Lesson 37 viewer: ACT / diffusion-style minimal probe - chunked multi-expert policy.

Three static modes share one lesson-37 recording (npz + summary):
1. the training curves: per-tier objective loss, the comparable deterministic
   action MSE (fit quality), the periodic deterministic-evaluation success /
   first-arrival evolution and the gate distribution evolution (entropy and the
   kick<->balance phase TV) - the headline question: did the deterministic path
   go 0 -> 1 during training?
2. the trajectory comparison: teacher versus block-output head versus the
   single-step MSE control, all from the same exact down start, with the cart
   position against the failure boundary, the motor inputs and the gate weights
   along each deterministic path (the key: does the deterministic path enter
   the upright region?);
3. the comparison: baseline / PPO(29) / DAPG(32) / DAgger(36) cited rows versus
   this lesson's tiers (stochastic + deterministic), the first-success verdict
   and the featured failure cases.

Layout and Esc handling follow the lesson-28..36 demos (docs/26 section 6):
every mode calls fig.clear() first, redraw draws synchronously, every quoted
number is traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.act_swingup import (
    EXPERIMENT,
    expected_npz_keys,
    load_archive,
)

DEFAULT_RESULTS = "results/act_swingup_2026-09-06"


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


def short_tier_name(label):
    for key, short in (
        ("单步 MSE", "单步MSE"),
        ("块输出", "块输出"),
        ("多峰门控 K=4（H=8）", "MoE K4·H8"),
        ("多峰门控 K=4", "MoE K4"),
        ("多峰门控 K=2", "MoE K2"),
    ):
        if label.startswith(key):
            return short
    return label.split("（")[0]


def headline_tier(tiers):
    """The headline tier: K=4, H=16 (moe_k4) when present, else the richest."""
    for name in ("moe_k4",):
        for tier in tiers:
            if tier["name"] == name:
                return tier
    return max(tiers, key=lambda t: (t["n_experts"], t["horizon"]))


def load_replays(directory):
    """Validate a lesson-37 recording; returns the data dict for the demo.

    Tamper routes rejected here: (1) the npz SHA-256 against the summary and
    the archive key set against the implied key set; (2) the stochastic success
    counts against the recorded archives and the per-seed summaries; (3) the
    deterministic records against the archived det arrays (settle time and
    cart excursion recomputed from the stored states).
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-37 recording")
    if tuple(t["name"] for t in report["tiers"]) != tuple(
        t["name"] for t in report["protocol"]["tiers"]
    ):
        raise ValueError("The tier list disagrees with the protocol")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    data = load_archive(directory)
    expected = expected_npz_keys(report)
    if set(data) != expected:
        raise ValueError("Unexpected archive arrays")

    seeds = report["training"]["train_seeds"]
    epochs = report["hyperparameters"]["epochs"]
    eval_every = report["hyperparameters"]["eval_every_epochs"]
    positions = sorted({*range(eval_every, epochs + 1, eval_every), epochs})
    for tier_index, tier in enumerate(report["protocol"]["tiers"]):
        for seed_index in range(seeds):
            per_seed = report["tiers"][tier_index]["per_seed"][seed_index]
            npz_positions = data[f"ckpt_positions_{tier_index}_{seed_index}"]
            if list(npz_positions) != positions:
                raise ValueError("Checkpoint positions disagree with the training protocol")
            recovered = data[f"eval_recovered_{tier_index}_{seed_index}"]
            per_episode = np.asarray(per_seed["eval"]["success_per_episode"], dtype=bool)
            if int(per_episode.sum()) != int(per_seed["eval"]["successes"]):
                raise ValueError("The stochastic success count disagrees with its episodes")
            if not np.array_equal(recovered, per_episode):
                raise ValueError("Stochastic successes disagree with the archive")
            det = per_seed["deterministic"]["mixture"]
            det_states = data[f"det_mixture_states_{tier_index}_{seed_index}"]
            if det is not None and det_states.ndim != 2:
                raise ValueError("Deterministic-state archive shape disagrees")
            if det is not None and not det["terminated"] and det.get("settled_at_s") is not None:
                max_x = float(np.max(np.abs(det_states[:, 0])))
                recorded = det.get("max_abs_cart_position_m")
                if recorded is not None and abs(max_x - float(recorded)) > 1e-4:
                    raise ValueError("Det cart excursion disagrees with its arrays")
            curve = data[f"loss_curve_{tier_index}_{seed_index}"]
            if curve.shape != (epochs,):
                raise ValueError("Loss curve length disagrees with the epoch count")
            if abs(float(curve[0]) - per_seed["loss_first"]) > 1e-6 * max(
                1.0, abs(per_seed["loss_first"])
            ):
                raise ValueError("Loss curve start disagrees with the summary")
            if abs(float(curve[-1]) - per_seed["loss_last"]) > 1e-6 * max(
                1.0, abs(per_seed["loss_last"])
            ):
                raise ValueError("Loss curve end disagrees with the summary")
    baseline = report["baseline"]
    per_episode = [row["settled_at_s"] for row in baseline["per_episode"]]
    if baseline["deterministic_identical_repeats"] and per_episode:
        median = float(np.median(per_episode))
        if abs(median - baseline["median_settled_at_s"]) > 1e-9:
            raise ValueError("Baseline median settle time disagrees with its episodes")
    return {"report": report, **data}


class ActDemo:
    """Chunked multi-expert policy versus the swing-up learning lineage."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()
        self.root = root
        self.data = data
        self.report = data["report"]
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第三十七课 · ACT/扩散策略最小实验：动作块 + 多峰（MoE）表示检验",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "教师（第 7 课控制器）的 ±300 N 双峰标注（8,877 对，逐位复现第 36 课数据集）训练\n"
                "Numpy 手写「共享主干 + K 个动作块头 + 门控」策略，门控混合最大似然目标；"
                "三模式共用同一正式记录，全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线 + 门控分布演化", "training"),
            ("② 块输出 vs 教师 vs 单步 MSE 轨迹", "trajectory"),
            ("③ 与 29/32/36 对照表 + 失败案例", "comparison"),
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
                "最小版声明：无 torch、无 Transformer、无去噪；K 个线性动作块头 + softmax 门控，"
                "固定 σ=1.0；确定性路径 = 门控加权均值（头条）/ top1 / 开环块执行（主档）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """① loss / det-MSE / deterministic evolution / gate evolution."""
        self.fig.clear()
        report = self.report
        data = self.data
        seeds = report["training"]["train_seeds"]
        tiers = report["tiers"]
        epochs = report["hyperparameters"]["epochs"]
        colors = plt_cache_colors()
        ax_loss, ax_mse, ax_det, ax_gate = self.fig.subplots(2, 2).reshape(-1)
        x_epoch = np.arange(1, epochs + 1)
        for tier_index, tier in enumerate(tiers):
            color = colors[tier_index % len(colors)]
            per_seed = np.stack([data[f"loss_curve_{tier_index}_{s}"] for s in range(seeds)])
            mean = per_seed.mean(axis=0)
            ax_loss.plot(x_epoch, mean, color=color, label=tier["label"])
            ax_loss.fill_between(
                x_epoch, per_seed.min(axis=0), per_seed.max(axis=0), alpha=0.15, color=color
            )
            ckpts = data[f"ckpt_positions_{tier_index}_0"]
            mse = np.stack([data[f"ckpt_det_mse_{tier_index}_{s}"] for s in range(seeds)])
            ax_mse.plot(
                ckpts, mse.mean(axis=0), marker="o", markersize=3, color=color, label=tier["label"]
            )
            success = np.stack([data[f"ckpt_recovered_{tier_index}_{s}"] for s in range(seeds)])
            ax_det.plot(
                ckpts,
                success.mean(axis=0),
                marker="o",
                markersize=3,
                color=color,
                label=f"{tier['label']} 成功占比",
            )
            arrival = np.stack([data[f"ckpt_arrival_s_{tier_index}_{s}"] for s in range(seeds)])
            if np.isfinite(arrival).any():
                ax_det.plot(
                    ckpts,
                    nanmean_safe(arrival),
                    linestyle=":",
                    linewidth=1.0,
                    color=color,
                )
            entropy = np.stack([data[f"ckpt_gate_entropy_{tier_index}_{s}"] for s in range(seeds)])
            tv = np.stack([data[f"ckpt_phase_tv_{tier_index}_{s}"] for s in range(seeds)])
            if np.isfinite(entropy).any():
                ax_gate.plot(
                    ckpts,
                    nanmean_safe(entropy),
                    marker="o",
                    markersize=3,
                    color=color,
                    label=f"{tier['label']} 门控熵",
                )
            if np.isfinite(tv).any():
                ax_gate.plot(
                    ckpts,
                    nanmean_safe(tv),
                    linestyle="--",
                    linewidth=1.0,
                    color=color,
                    alpha=0.7,
                )
        for ax, title, ylabel in (
            (ax_loss, "训练目标损失（各层目标不同，仅量级参考）", "目标损失"),
            (ax_mse, "确定性动作 MSE（拟合质量，可直接比较）", "MSE"),
            (ax_det, "确定性评估：成功占比（实线）与首达时刻（虚线）", "成功占比 / 首达 s"),
            (ax_gate, "门控演化：熵（实线）与 kick<->balance 相位 TV（虚线）", "熵 / TV"),
        ):
            ax.set(
                xlabel=("epoch" if ax is not ax_det else "epoch（周期确定性评估）"),
                title=title,
                ylabel=ylabel,
            )
            ax.legend(fontsize=6)
            ax.grid(alpha=0.2)

    def draw_trajectory(self):
        """② teacher vs block-output vs single-step MSE, same exact down start."""
        self.fig.clear()
        report = self.report
        data = self.data
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        by_name = {t["name"]: i for i, t in enumerate(report["tiers"])}
        teacher = data["baseline_states"]
        teacher_controls = data["baseline_controls"]
        mse_index = by_name.get("mse_single")
        block_index = by_name.get("block_mse")
        main_tier = headline_tier(report["tiers"])
        main_index = next(
            i for i, t in enumerate(report["tiers"]) if t["name"] == main_tier["name"]
        )
        ax_pole, ax_cart, ax_force, ax_gate = self.fig.subplots(2, 2).reshape(-1)
        ax_pole.plot(
            np.arange(len(teacher)) * dt,
            np.cos(teacher[:, 1] - ref_theta),
            "--",
            color="#64748b",
            label="教师（第 7 课，20/20）",
        )
        if mse_index is not None:
            mse_states = data[f"det_mixture_states_{mse_index}_0"]
            ax_pole.plot(
                np.arange(len(mse_states)) * dt,
                np.cos(mse_states[:, 1] - ref_theta),
                color="#2563eb",
                linewidth=1.3,
                label="单步 MSE（第 32/36 课同款目标）",
            )
        if block_index is not None:
            block_states = data[f"det_mixture_states_{block_index}_0"]
            ax_pole.plot(
                np.arange(len(block_states)) * dt,
                np.cos(block_states[:, 1] - ref_theta),
                color="#b91c1c",
                linewidth=1.3,
                label="块输出单头（K=1, H=16）",
            )
        main_states = data[f"det_mixture_states_{main_index}_0"]
        ax_pole.plot(
            np.arange(len(main_states)) * dt,
            np.cos(main_states[:, 1] - ref_theta),
            color="#7c3aed",
            linewidth=1.3,
            label=f"{report['tiers'][main_index]['label']}（确定性均值路径）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="②同一下方初态：确定性路径是否进入直立区（|α|≤0.3 rad）",
        )
        ax_pole.legend(fontsize=6.5, loc="lower right")
        for bound in (-2.4, 2.4):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.plot(
            np.arange(len(teacher)) * dt, teacher[:, 0], "--", color="#64748b", label="教师"
        )
        ax_cart.plot(
            np.arange(len(main_states)) * dt,
            main_states[:, 0],
            color="#7c3aed",
            label=report["tiers"][main_index]["label"],
        )
        ax_cart.set(
            ylabel="小车位置（m）", xlabel="仿真时间（s）", title="小车位置（红点线 = ±2.4 m 边界）"
        )
        ax_cart.legend(fontsize=6.5)
        ax_force.stairs(
            teacher_controls * 100.0,
            np.arange(len(teacher_controls) + 1) * dt,
            color="#64748b",
            label="教师",
        )
        main_controls = data[f"det_mixture_controls_{main_index}_0"]
        ax_force.stairs(
            main_controls * 100.0,
            np.arange(len(main_controls) + 1) * dt,
            color="#7c3aed",
            label=report["tiers"][main_index]["label"],
        )
        if mse_index is not None:
            mse_controls = data[f"det_mixture_controls_{mse_index}_0"]
            ax_force.stairs(
                mse_controls * 100.0,
                np.arange(len(mse_controls) + 1) * dt,
                color="#2563eb",
                alpha=0.6,
                label="单步 MSE",
            )
        ax_force.set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入（±300 N 限幅）")
        ax_force.legend(fontsize=6.5)
        gate_det = data.get(f"gate_det_{main_index}_0")
        if gate_det is not None:
            time_axis = np.arange(len(gate_det)) * dt
            ax_gate.stackplot(
                time_axis,
                gate_det.T,
                labels=[f"专家 {k}" for k in range(gate_det.shape[1])],
                alpha=0.7,
            )
        ax_gate.set(
            ylabel="门控权重",
            xlabel="仿真时间（s）",
            title="主档门控沿确定性路径（是否分相切换）",
        )
        ax_gate.legend(fontsize=6.5, loc="upper left")
        for ax in (ax_pole, ax_cart, ax_force, ax_gate):
            ax.grid(alpha=0.2)

    def draw_comparison(self):
        """③ multi-way success table, deterministic verdict, failure case."""
        self.fig.clear()
        report = self.report
        data = self.data
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        rows = report["comparison"]
        labels = []
        for row in rows:
            if row["label"].startswith("基线"):
                labels.append("教师\n第7课")
            elif row["label"].startswith("纯 PPO"):
                labels.append("PPO\n第29课")
            elif row["label"].startswith("DAPG"):
                labels.append("DAPG\n第32课")
            elif row["label"].startswith("DAgger"):
                labels.append("DAgger\n第36课")
            else:
                labels.append(row["label"].split("（")[0])
        ax_rate, ax_verdict, ax_case, ax_detail = self.fig.subplots(2, 2).reshape(-1)
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        colors = plt_cache_colors()
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=[colors[i % 10] for i in range(len(labels))],
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
        ax_rate.set(ylabel="随机评估成功率（%）", ylim=(0, 112), title="③成功率对照（第 7 课口径）")
        ax_rate.tick_params(axis="x", labelsize=7)
        lines = ["确定性路径（均值路径）头条："]
        for tier in report["tiers"]:
            agg = tier["aggregate"]
            lines.append(
                f"  {tier['label']}：确定性成功 {agg['det_successes']}/{len(tier['per_seed'])}，"
                f"首成于 epoch {agg['first_det_success_epoch'] or '从未'}"
            )
        lines.append(
            f"  教师（第 7 课）：{report['baseline']['successes']}/"
            f"{report['baseline']['episodes']}（中位 "
            f"{report['baseline']['median_settled_at_s']:.2f} s）"
        )
        lines.append(
            "  对照随机评估：纯 PPO 29 = 0/60；DAPG 32 = 0/60（首达 33/60）；"
            "DAgger 36 = 0/60（首达 5/60）"
        )
        lines.append("  第 32 课均值路径到达（2.28/1.52/1.76 s）但从未稳定")
        lines.append("  第 36 课均值路径从未到达（首达全部未到达）")
        ax_verdict.axis("off")
        ax_verdict.text(
            0.02,
            0.96,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=8,
            transform=ax_verdict.transAxes,
        )
        ax_verdict.set(title="确定性评估裁决：0→1 是否出现")
        cases = report["failure_analysis"]["featured_cases"]
        if cases:
            case = cases[0]
            states = data["case0_states"]
            ax_case.plot(
                np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c"
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{case['tier']}（种子 {case['seed_index']}，{failure_label(case)}）",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例")
        for tier_index, tier in enumerate(report["tiers"]):
            color = colors[tier_index % 10]
            arrivals = np.stack(
                [
                    data[f"eval_arrival_s_{tier_index}_{s}"]
                    for s in range(report["training"]["train_seeds"])
                ]
            )
            recovered = np.stack(
                [
                    data[f"eval_recovered_{tier_index}_{s}"]
                    for s in range(report["training"]["train_seeds"])
                ]
            )
            xpos = np.arange(len(recovered)) + 1
            ax_detail.bar(
                xpos - 0.2,
                np.isfinite(arrivals).sum(axis=1),
                width=0.38,
                color=color,
                alpha=0.8,
                label=f"{tier['label']} 随机首达",
            )
            det_ok = np.asarray(
                [bool(s["deterministic"]["mixture"]["recovered"]) for s in tier["per_seed"]]
            )
            ax_detail.bar(
                xpos + 0.2,
                det_ok.astype(float),
                width=0.38,
                color=color,
                hatch="//",
                alpha=0.9,
                label=f"{tier['label']} 确定性成功",
            )
        ax_detail.set(xlabel="训练种子", ylabel="计数", title="每种子：随机首达 vs 确定性成功")
        handles, labels = ax_detail.get_legend_handles_labels()
        ax_detail.legend(handles[:2], ["随机首达", "确定性成功"], fontsize=7)
        for ax in (ax_rate, ax_case, ax_detail):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        tiers = report["tiers"]
        if mode == "training":
            lines = ["① 这一幕在追踪：确定性路径是否随训练从 0 变 1"]
            lines.append(
                f"  epoch {report['hyperparameters']['epochs']}、"
                f"数据 {report['training']['total_pairs']} 对、墙钟 "
                f"{report['training']['wall_time_s_total']:.0f} s"
            )
            for tier in tiers:
                agg = tier["aggregate"]
                lines.append(
                    f"  {tier['label']}：随机 {agg['stochastic_successes']}/"
                    f"{agg['stochastic_total']}，确定性 {agg['det_successes']}/"
                    f"{len(tier['per_seed'])}，首成 epoch "
                    f"{agg['first_det_success_epoch'] or '从未'}"
                )
            main = headline_tier(tiers)
            if main["aggregate"]["mean_phase_tv"] is not None:
                lines.append(
                    f"  {main['label']} 门控分相 TV：{main['aggregate']['mean_phase_tv']:.3f}"
                )
            lines.append(
                f"  教师复验 {report['teacher_verification']['successes']}/"
                f"{report['teacher_verification']['episodes']}，闸门通过"
            )
        elif mode == "trajectory":
            lines = ["② 这一幕在对照：确定性路径进不进直立区"]
            main = headline_tier(tiers)
            det = main["per_seed"][0]["deterministic"]["mixture"]
            lines.append(f"  主档：{main['label']}")
            lines.append(
                f"    确定性（混合均值）首达 {null_to_text(det['first_arrival_s'])}、"
                f"稳定 {null_to_text(det['settled_at_s'])}"
            )
            lines.append(f"    教师稳定：{report['baseline']['median_settled_at_s']:.2f} s")
            for tier in tiers:
                if tier["name"] in ("mse_single", "block_mse"):
                    record = tier["per_seed"][0]["deterministic"]["mixture"]
                    lines.append(
                        f"  {tier['label']}：首达 {null_to_text(record['first_arrival_s'])}、"
                        f"稳定 {null_to_text(record['settled_at_s'])}"
                    )
            lines.append("  对照：第 32 课均值路径到达但未稳（2.28/1.52/1.76 s）")
            lines.append("  第 36 课均值路径从未到达（首达全部 None）")
        else:
            lines = ["③ 这一幕在裁决：0→1 出现了吗"]
            for row in report["comparison"]:
                label = row["label"]
                short = label
                for key, replacement in (
                    ("基线", "基线（第 7 课）"),
                    ("纯 PPO", "纯 PPO（29）"),
                    ("DAPG", "DAPG 离线（32）"),
                    ("DAgger", "DAgger 在线（36）"),
                ):
                    if label.startswith(key):
                        short = replacement
                        break
                lines.append(f"  {short}：{row['successes']}/{row['episodes']}")
            for tier in tiers:
                agg = tier["aggregate"]
                lines.append(
                    f"  {tier['label']}：随机 {agg['stochastic_successes']}/"
                    f"{agg['stochastic_total']}；确定性 {agg['det_successes']}/"
                    f"{len(tier['per_seed'])}，首成 "
                    + ("是" if agg["first_det_success_epoch"] else "否")
                )
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "training":
            self.draw_training()
        elif mode == "trajectory":
            self.draw_trajectory()
        else:
            self.draw_comparison()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：各层目标损失 / 确定性动作 MSE / 周期确定性评估成功与首达 / 门控熵与分相 TV",
            "trajectory": "② 这一幕在对照：教师 vs 单步 MSE vs 块输出 vs 主档（同一精确下方初态）的确定性轨迹",
            "comparison": "③ 这一幕在裁决：教师 / PPO(29) / DAPG(32) / DAgger(36) / 本课各层的成功与首次成功",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def nanmean_safe(values, axis=0):
    """Column-wise nanmean that never warns on all-NaN slices (returns NaN)."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    counts = mask.sum(axis=axis, keepdims=True)
    filled = np.where(mask, values, 0.0)
    sums = filled.sum(axis=axis, keepdims=True)
    return np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan).reshape(
        -1 if axis == 1 else (values.shape[1] if axis == 0 else values.shape[axis])
    )


def plt_cache_colors():
    from matplotlib import pyplot as plt

    return plt.get_cmap("tab10").colors


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第三十七课 · ACT/扩散策略最小实验：动作块 + 多峰（MoE）表示检验")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    ActDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
