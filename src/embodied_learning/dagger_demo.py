"""Lesson 36 viewer: DAgger online correction - the teacher labels student states.

Three static modes share one lesson-36 recording (npz + summary):
1. the round evolution: per-round success / upright-arrival curves on the exact
   down start (initial = the fresh policy before round 0), the aggregated
   teacher-label distribution and the dataset growth - the headline 0 -> 1;
2. the trajectory comparison: the teacher (lesson-7 baseline) versus the student
   from the same exact down start (after round 0 and after the last round), cart
   position against the failure boundary, motor inputs and the teacher-label
   intensity per round;
3. the outcome: baseline / lesson-29 PPO / lesson-32 DAPG / lesson-34 two-phase /
   lesson-35 SAC (cited official records) versus the DAgger tiers' final rounds,
   the first-success verdict and the featured failure case.

Layout and Esc handling follow the lesson-28..35 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz (docs/26 section 6 standard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.dagger_swingup import (
    EXPERIMENT,
    bc_weight_at,
    expected_npz_keys,
    failure_label,
)

DEFAULT_RESULTS = "results/dagger_swingup_2026-09-06"


def load_replays(directory):
    """Validate a lesson-36 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-36 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["learn_std"]:
        raise ValueError("Unexpected network contract")
    if (
        report["protocol"]["observation"]["features"]
        != "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]"
    ):
        raise ValueError("Unexpected observation contract")
    scheme = report["protocol"]["w_bc_schedule"]
    if tuple(scheme["tiers"]) != tuple(hyper["w_bc_levels"]):
        raise ValueError("The tier list disagrees with the recorded hyperparameters")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    rounds = report["training"]["rounds"]
    eval_count = report["tiers"][0]["evaluations_per_round"]
    for level_index, entry in enumerate(report["tiers"]):
        w_init = scheme["tiers"][level_index]
        per_seed = entry["per_seed"]
        recovered = data[f"eval_recovered_{level_index}_0"]
        if recovered.shape != (rounds, eval_count):
            raise ValueError("Archive evaluation shapes disagree with the summary")
        for seed_run in per_seed:
            seed = seed_run["seed_index"]
            per_round = data[f"eval_recovered_{level_index}_{seed}"]
            if not np.array_equal(
                per_round.sum(axis=1), np.asarray(seed_run["per_round_successes"])
            ):
                raise ValueError("DAgger successes disagree with the archive")
            curve = data[f"bc_curve_{level_index}_{seed}"]
            if curve.shape != (rounds * report["hyperparameters"]["dagger_updates_per_round"],):
                raise ValueError("BC curve length disagrees with the update count")
            expected_w = np.repeat(
                [bc_weight_at(r, rounds, w_init) for r in range(rounds)],
                report["hyperparameters"]["dagger_updates_per_round"],
            )
            if not np.allclose(data[f"w_bc_curve_{level_index}_{seed}"], expected_w):
                raise ValueError("The recorded w_BC schedule disagrees with the protocol")
            first_record = seed_run["rounds"][0]
            last_record = seed_run["rounds"][-1]
            if (
                abs(float(curve[0]) - first_record["bc_first"])
                > 1e-9 * max(1.0, abs(first_record["bc_first"]))
                and w_init > 0.0
            ):
                raise ValueError("BC curve start disagrees with the summary")
            if (
                abs(float(curve[-1]) - last_record["bc_last"])
                > 1e-9 * max(1.0, abs(last_record["bc_last"]))
                and w_init > 0.0
            ):
                raise ValueError("BC curve end disagrees with the summary")
        if entry["label"] == "DAgger":
            sizes = data[f"dataset_size_{level_index}_0"]
            if sizes.shape != (rounds,):
                raise ValueError("Dataset size curve length disagrees with the rounds")
            for r in range(rounds):
                hist = data[f"label_hist_{level_index}_{r}"]
                added = sum(seed_run["rounds"][r]["dataset_added"] for seed_run in per_seed)
                if int(hist.sum()) != added:
                    raise ValueError(
                        "The teacher label histogram disagrees with the annotated pairs"
                    )

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


class DaggerDemo:
    """DAgger online correction versus the swing-up learning lineage."""

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
            text="第三十六课 · DAgger 在线纠错：教师逐帧标注学生的状态分布",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "每轮：学生（第 29 课 PPO）从精确下方初态 rollout → 教师（第 7 课控制器，已验 20/20）"
                "逐帧标注 → 汇总集 D 进入\nL = L_PPO + w_BC·MSE(μ(s), a_教师)，w 跨轮退火到 0。"
                "三模式共用同一正式记录，全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="evolution")
        for label, key in (
            ("① 成功率轮次演化 + 教师标注分布", "evolution"),
            ("② 教师 vs 学生轨迹（同初态）", "trajectory"),
            ("③ 与 29/32/35 对照表 + 失败案例", "comparison"),
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
                "DAgger（Ross et al. RSS 2010）：数据随策略演化——教师标注的是学生自己走到的状态，"
                "不是教师自己的轨迹。成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_evolution(self):
        """① per-round success and arrival evolution + the teacher label cloud."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        rounds = report["training"]["rounds"]
        eval_count = report["tiers"][0]["evaluations_per_round"]
        colors = ("#b91c1c", "#64748b")
        x = np.arange(rounds + 1)
        ax_succ, ax_arr, ax_hist, ax_data = self.fig.subplots(2, 2).reshape(-1)
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[tier_index % len(colors)]
            per_round = np.asarray(entry["successes_per_seed_per_round"], dtype=float).T
            rows = np.concatenate(
                [np.asarray(entry["initial_successes_per_seed"], dtype=float)[None, :], per_round],
                axis=0,
            )
            for seed in range(seeds):
                ax_succ.plot(x, rows[:, seed], "o--", markersize=3, alpha=0.35, color=color)
            ax_succ.plot(
                x,
                rows.mean(axis=1),
                "o-",
                linewidth=1.8,
                color=color,
                label=f"{entry['label']}（w={entry['w_bc']:g}）",
            )
        ax_succ.axhline(eval_count, color="gray", linestyle=":", linewidth=0.8)
        ax_succ.set(
            xlabel="轮次（0 = 训练前初值）",
            ylabel=f"成功回合数（/{eval_count}）",
            title="①下方初态成功率轮次演化",
        )
        ax_succ.legend(fontsize=7)
        ax_succ.grid(alpha=0.2)
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[tier_index % len(colors)]
            per_round = np.asarray(entry["arrivals_per_seed_per_round"], dtype=float).T
            rows = np.concatenate(
                [
                    np.asarray(entry["initial_arrivals_per_seed"], dtype=float)[None, :],
                    per_round,
                ],
                axis=0,
            )
            for seed in range(seeds):
                ax_arr.plot(x, rows[:, seed], "o--", markersize=3, alpha=0.35, color=color)
            ax_arr.plot(
                x, rows.mean(axis=1), "o-", linewidth=1.8, color=color, label=entry["label"]
            )
        ax_arr.set(
            xlabel="轮次（0 = 训练前初值）",
            ylabel=f"直立首达回合数（/{eval_count}）",
            title="直立首达（|α|≤0.3 rad）",
        )
        ax_arr.legend(fontsize=7)
        ax_arr.grid(alpha=0.2)
        edges = self.data["label_bin_edges"]
        width = np.diff(edges)
        last = rounds - 1
        ax_hist.bar(
            edges[:-1],
            self.data["label_hist_0_0"],
            width=width * 0.95,
            align="edge",
            alpha=0.55,
            color="#2563eb",
            label="轮 0 标注",
        )
        ax_hist.bar(
            edges[:-1],
            self.data[f"label_hist_0_{last}"],
            width=width * 0.95,
            align="edge",
            alpha=0.55,
            color="#b91c1c",
            label=f"轮 {last} 标注",
        )
        ax_hist.set(
            xlabel="教师逐帧标注动作（-3..3）",
            ylabel="帧数",
            title="教师标注分布（DAgger 档）",
        )
        ax_hist.legend(fontsize=7)
        ax_hist.grid(alpha=0.2)
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[tier_index % len(colors)]
            sizes = np.asarray(
                [self.data[f"dataset_size_{tier_index}_{seed}"] for seed in range(seeds)],
                dtype=float,
            )
            ax_data.plot(
                np.arange(1, rounds + 1),
                sizes.mean(axis=0),
                "o-",
                linewidth=1.8,
                color=color,
                label=f"{entry['label']} 数据集",
            )
        if report["tiers"][0]["label"] == "DAgger":
            bcs = np.stack([self.data[f"bc_curve_0_{seed}"] for seed in range(seeds)], axis=0)
            ax_data.plot(
                np.arange(1, len(bcs[0]) + 1),
                np.nanmean(bcs, axis=0),
                ":",
                linewidth=1.2,
                color="#2563eb",
                label="BC MSE（DAgger）",
            )
        ax_data.set(
            xlabel="轮次（数据集）/ 梯度更新序号（BC）",
            ylabel="对数 / MSE",
            title="数据聚合规模与 BC 残差",
        )
        ax_data.legend(fontsize=7)
        ax_data.grid(alpha=0.2)

    def draw_trajectory(self):
        """② teacher vs student from the same exact down start."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        gear = report["protocol"]["actuator_gear"]
        rounds = report["training"]["rounds"]
        last = rounds - 1
        teacher = self.data["baseline_states"]
        teacher_controls = self.data["baseline_controls"]
        det0 = self.data["det_states_0_0_r0"]
        det_last = self.data[f"det_states_0_0_r{last}"]
        det0_controls = self.data["det_controls_0_0_r0"]
        det_last_controls = self.data[f"det_controls_0_0_r{last}"]
        ax_pole, ax_cart, ax_force, ax_label = self.fig.subplots(2, 2).reshape(-1)
        ax_pole.plot(
            np.arange(len(teacher)) * dt,
            np.cos(teacher[:, 1] - ref_theta),
            "--",
            color="#64748b",
            label="教师（第 7 课基线，20/20）",
        )
        ax_pole.plot(
            np.arange(len(det0)) * dt,
            np.cos(det0[:, 1] - ref_theta),
            color="#2563eb",
            linewidth=1.3,
            label="学生：第 1 轮后（均值动作）",
        )
        ax_pole.plot(
            np.arange(len(det_last)) * dt,
            np.cos(det_last[:, 1] - ref_theta),
            color="#b91c1c",
            linewidth=1.3,
            label="学生：末轮后（均值动作）",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="②同一下方初态：教师 vs 学生（均值动作）",
        )
        ax_pole.legend(fontsize=7, loc="lower right")
        for bound in (-2.4, 2.4):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.plot(
            np.arange(len(teacher)) * dt, teacher[:, 0], "--", color="#64748b", label="教师"
        )
        ax_cart.plot(np.arange(len(det_last)) * dt, det_last[:, 0], color="#b91c1c", label="末轮后")
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title="小车位置（红点线 = ±2.4 m 失败边界）",
        )
        ax_cart.legend(fontsize=7)
        ax_force.stairs(
            teacher_controls * gear,
            np.arange(len(teacher_controls) + 1) * dt,
            color="#64748b",
            label="教师",
        )
        ax_force.stairs(
            det_last_controls * gear,
            np.arange(len(det_last_controls) + 1) * dt,
            color="#b91c1c",
            label="末轮后",
        )
        ax_force.stairs(
            det0_controls * gear,
            np.arange(len(det0_controls) + 1) * dt,
            color="#2563eb",
            alpha=0.6,
            label="第 1 轮后",
        )
        ax_force.set(ylabel="电机力（N）", xlabel="仿真时间（s）", title="电机输入（±300 N 限幅）")
        ax_force.legend(fontsize=7)
        entry = report["tiers"][0]
        per_round_mean = np.asarray(
            [
                np.nanmean(
                    [
                        record["label_stats"]["mean"] or 0.0
                        for seed_run in entry["per_seed"]
                        for record in seed_run["rounds"]
                        if record["round_index"] == r
                    ]
                )
                for r in range(rounds)
            ]
        )
        ax_label.plot(
            np.arange(1, rounds + 1),
            per_round_mean,
            "o-",
            linewidth=1.8,
            color="#b91c1c",
            label="教师标注均值",
        )
        saturated = np.asarray(
            [
                np.nanmean(
                    [
                        record["label_stats"]["saturated_fraction"] or 0.0
                        for seed_run in entry["per_seed"]
                        for record in seed_run["rounds"]
                        if record["round_index"] == r
                    ]
                )
                for r in range(rounds)
            ]
        )
        ax_label.plot(
            np.arange(1, rounds + 1),
            saturated,
            "s--",
            markersize=4,
            color="#2563eb",
            label="饱和标注占比（|a|≥2.99）",
        )
        ax_label.set(
            xlabel="轮次",
            ylabel="标注动作 / 占比",
            title="教师标注强度：学生状态分布上的教师指令均值",
        )
        ax_label.legend(fontsize=7)
        for ax in (ax_pole, ax_cart, ax_force, ax_label):
            ax.grid(alpha=0.2)

    def draw_comparison(self):
        """③ multi-way success table, first-success verdict and failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        ax_rate, ax_verdict, ax_case, ax_detail = self.fig.subplots(2, 2).reshape(-1)
        rows = report["comparison"]
        labels = ["基线", "PPO\n29", "DAPG\n32", "两阶段\n34", "SAC\n35"] + [
            f"{row['label'].split('（')[-1].strip('）')}" for row in rows[5:]
        ]
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        colors = ("#64748b", "#b91c1c", "#7c3aed", "#9333ea", "#d97706", "#0f766e", "#2563eb")
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
        ax_rate.set(ylabel="验收通过率（%）", ylim=(0, 112), title="③多方式成功率（第 7 课口径）")
        ax_rate.tick_params(axis="x", labelsize=7)
        lines = ["0→1 是否出现（头条）："]
        for entry in report["tiers"]:
            per_seed = entry["first_success_per_seed"]
            lines.append(
                f"  {entry['label']}：每种子首成功轮次 "
                + "、".join(str(v + 1) if v is not None else "从未" for v in per_seed)
            )
        for entry in report["tiers"]:
            if entry["aggregate"]["first_success_any"]:
                lines.append(f"  → {entry['label']} 出现历史性 0→1")
        lines.append(
            f"  教师（第 7 课）：{report['baseline']['successes']}/"
            f"{report['baseline']['episodes']}（中位 "
            f"{report['baseline']['median_settled_at_s']:.2f} s）"
        )
        lines.append("  对照：PPO 29 = 0/60（从未首达）；DAPG 32 = 0/60（首达 33/60）")
        lines.append("  两阶段 34 = 0/60；SAC 35 = 0/60（首达 0/60）")
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
        ax_verdict.set(title="直立首达与首次成功")
        cases = report["failure_analysis"]["featured_cases"]
        if cases:
            case = cases[0]
            states = self.data["case0_states"]
            ax_case.plot(
                np.arange(len(states)) * dt, np.cos(states[:, 1] - ref_theta), color="#b91c1c"
            )
            ax_case.axhspan(-1, 0, alpha=0.08, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{failure_label(case)}（{case.get('w_bc_label', '')}）",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有学生失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例：无（全班通过）")
        for tier_index, entry in enumerate(report["tiers"]):
            color = colors[(tier_index + 4) % len(colors)]
            for seed, row in enumerate(entry["successes_per_seed_per_round"]):
                ax_detail.plot(
                    np.arange(len(row)) + 1,
                    row,
                    "o--",
                    markersize=4,
                    alpha=0.6,
                    color=color,
                    label=f"{entry['label']} 种子 {seed}",
                )
        ax_detail.set(xlabel="轮次", ylabel="成功回合数（/20）", title="每轮成功率：逐种子")
        ax_detail.legend(fontsize=7)
        for ax in (ax_rate, ax_case, ax_detail):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        rounds = report["training"]["rounds"]
        if mode == "evolution":
            lines = ["① 这一幕在追踪：纠错是否把成功率从 0 拉起来"]
            lines.append(
                f"  轮数 {rounds}、每轮 rollout "
                f"{report['protocol']['dagger_loop']['annotated_rollouts_per_round']} 回合"
            )
            lines.append(
                f"  墙钟 {report['training']['wall_time_s_total']:.0f} s、"
                f"环境步 {report['training']['env_steps_total']}"
            )
            for entry in report["tiers"]:
                for seed_run in entry["per_seed"]:
                    line = (
                        f"  {entry['label']} 种子 {seed_run['seed_index']}："
                        f"初值 {seed_run['initial_eval']['successes']} 成功 "
                        f"→ 末轮 {seed_run['rounds'][-1]['eval']['successes']}"
                    )
                    lines.append(line)
                agg = entry["aggregate"]
                lines.append(
                    f"    末轮合计 {agg['final_round_total_successes']}/"
                    f"{entry['eval_total']}；首成功轮次 "
                    + (
                        str(agg["rounds_to_first_success"] + 1)
                        if agg["rounds_to_first_success"] is not None
                        else "从未"
                    )
                )
            lines.append(
                f"  教师复验 {report['teacher_verification']['successes']}/"
                f"{report['teacher_verification']['episodes']}，闸门通过"
            )
        elif mode == "trajectory":
            lines = ["② 这一幕在对照：学得行为像不像教师"]
            entry = report["tiers"][0]
            det = entry["per_seed"][0]["rounds"][-1]["deterministic"]
            lines.append(f"  教师稳定（基线）：{report['baseline']['median_settled_at_s']:.2f} s")
            lines.append(
                f"  学生（{entry['label']} 种子 0）："
                f"第 1 轮后 首达 "
                f"{null_to_text(entry['per_seed'][0]['rounds'][0]['deterministic']['first_arrival_s'])}、"
                f"末轮后 首达 {null_to_text(det['first_arrival_s'])}"
            )
            lines.append(f"  末轮后稳定 {null_to_text(det['settled_at_s'])}")
            rounds_records = entry["per_seed"][0]["rounds"]
            for r in range(rounds):
                record = rounds_records[r]
                lines.append(
                    f"  轮 {r + 1}：数据集 {record['dataset_size']}（+{record['dataset_added']}），"
                    f"丢弃 {record['dropped_labels']}，标注数 {record['label_stats']['pairs']}"
                )
            lines.append("  教师标注均值与饱和占比见右下图")
        else:
            lines = ["③ 这一幕在裁决：0→1 出现了吗"]
            for row in report["comparison"]:
                label = row["label"]
                if label.startswith("基线"):
                    label = "基线（第 7 课）"
                elif label.startswith("纯 PPO"):
                    label = "纯 PPO（29）"
                elif label.startswith("DAPG"):
                    label = "DAPG 离线（32）"
                elif label.startswith("两阶段"):
                    label = "两阶段（34）"
                elif label.startswith("SAC"):
                    label = "SAC（35）"
                else:
                    label = f"本课 {row['label'].split('（')[1].split('）')[0]}"
                lines.append(f"  {label}：{row['successes']}/{row['episodes']}")
            for entry in report["tiers"]:
                agg = entry["aggregate"]
                lines.append(
                    f"  本课 {entry['label']}：末轮 {agg['final_round_total_successes']}/"
                    f"{entry['eval_total']}，首成功 " + ("是" if agg["first_success_any"] else "否")
                )
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "evolution":
            self.draw_evolution()
        elif mode == "trajectory":
            self.draw_trajectory()
        else:
            self.draw_comparison()
        self.fill_stats(mode)
        status = {
            "evolution": "① 这一幕在追踪：每轮末下方初态成功率/首达演化（0 = 训练前初值），教师标注分布与数据聚合",
            "trajectory": "② 这一幕在对照：教师与学生的同初态轨迹、小车位置、电机输入、教师标注强度",
            "comparison": "③ 这一幕在裁决：基线 / PPO(29) / DAPG(32) / 两阶段(34) / SAC(35) / 本课对比表与失败案例",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw()  # synchronous: draw_idle left stale pixels on mode switches


def null_to_text(value):
    return f"{value:.2f} s" if value is not None else "未到达"


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.title("第三十六课 · DAgger 在线纠错：教师逐帧标注学生的状态分布")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    DaggerDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
