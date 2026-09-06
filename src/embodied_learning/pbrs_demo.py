"""Lesson 31 viewer: PBRS shaping - repair the ladder, the summit stays put.

Three static modes share one lesson-31 recording (npz + summary):
1. training curves: per-update shaped reward and periodic down-start evaluation
   for each potential scale c_e, with the lesson-29 pure-PPO final reward as
   the cited reference line;
2. the Phi ladder: the static potential over pole angles, and along a typical
   (mean-action) trajectory the Phi profile plus the per-step shaping versus
   task reward, with the first arrival of the upright capture region marked;
3. the three-way outcome: lesson-7 baseline versus lesson-29 pure PPO (both
   cited/re-run numbers in the record) versus the PBRS tiers, the upright
   first-arrival times (lesson 29 never arrived), and the featured failure
   case.
Layout and Esc handling follow the lesson-28/29/30 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pbrs_swingup import (
    CAPTURE_ANGLE_RAD,
    EXPERIMENT,
    expected_npz_keys,
    failure_label,
    pole_energy,
)

DEFAULT_RESULTS = "results/pbrs_swingup_2026-09-06"


def level_label(entry):
    return f"cE={entry['c_e']:g}"


def load_replays(directory):
    """Validate a lesson-31 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-31 recording")
    hyper = report["hyperparameters"]
    if hyper["hidden"] != [64, 64] or hyper["learn_std"]:
        raise ValueError("Unexpected network contract")
    if (
        report["protocol"]["observation"]["features"]
        != "[x/2.5, cos(alpha), sin(alpha), v/5, omega/10]"
    ):
        raise ValueError("Unexpected observation contract")
    guard = report["guard"]
    if not all(guard[key] for key in ("bitwise_identical_rewards", "bitwise_identical_states")):
        raise ValueError("Recording claims a broken c_e = 0 pipeline guard")
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
            raise ValueError("PBRS successes disagree with the archive")

    # The potential curve must be recomputable from the protocol constants.
    potential = report["protocol"]["potential"]
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    phi = data["det_phi_0_0"]
    states = data["det_states_0_0"]
    recomputed = np.asarray(
        [
            -report["sweep"][0]["c_e"]
            * abs(
                pole_energy(
                    state, reference, potential["hinge_inertia_kg_m2"], potential["mgl_eff_j"]
                )
            )
            for state in states
        ],
        dtype=float,
    )
    if phi.shape != recomputed.shape or np.max(np.abs(phi - recomputed)) > 1e-6:
        raise ValueError("Potential curve disagrees with the protocol constants")

    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        case_states = data[f"case{index}_states"]
        if case_states.ndim != 2 or case_states.shape[1] != 4:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        max_x = float(np.max(np.abs(case_states[:, 0])))
        if abs(max_x - case["max_abs_cart_position_m"]) > 1e-4:
            raise ValueError(f"Featured case {index} cart excursion disagrees with its arrays")
    return {"report": report, **data}


class PbrsDemo:
    """PBRS reward shaping versus the lesson-29 pure PPO and the lesson-7 baseline."""

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
            text="第三十一课 · 势函数奖励塑形：给悬崖修梯子，山顶不动（PBRS）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "第二十九课纯 PPO 原样保留（无底座、无教师），奖励只加一项 γΦ(s′)−Φ(s)，Φ=−cE·|E−E_top|。\n"
                "三个模式共用同一条正式记录：① 训练曲线（PBRS vs 纯 PPO 口径）② Φ 梯子与奖励分解＋直立首达 "
                "③ 三方成功率与失败案例；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：PBRS 两档 vs 纯 PPO 口径", "training"),
            ("② Φ 梯子：沿典型轨迹的势与奖励分解", "ladder"),
            ("③ 三方成功率与直立首达、失败案例", "outcome"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        # Claim the panel's width on the right BEFORE the expanding canvas
        # frame; the left frame's requested size is frozen (pack_propagate off)
        # so the TkAgg canvas cannot push the status line out of the window.
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
                "Ng/Harada/Russell 1999：塑形项取 γΦ(s′)−Φ(s) 形状时不改变最优策略；cE 是本课唯一的隐形手工旋钮。"
                "评估固定在精确静止下方初态，成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """① shaped reward curves and periodic evaluations, per c_e level."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_eval = self.fig.subplots(1, 2)
        colors = ("#2563eb", "#0f766e", "#b45309")
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        lesson29_final = float(np.mean(report["lesson29_reference"]["final_reward_mean_per_seed"]))
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
                label=f"{level_label(entry)}（{seeds} 种子均值）",
            )
        ax_reward.axhline(
            lesson29_final,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.2,
            label="纯 PPO（第 29 课）末段均值",
        )
        ax_reward.set(
            xlabel="PPO 更新轮次",
            ylabel="批内平均奖励（含塑形，每步）",
            title="PBRS 训练奖励：细线 = 单个种子",
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
            ylabel="下方初态验收通过（1=成功）",
            ylim=(-0.08, 1.08),
            yticks=[0, 0.5, 1],
            title="周期评估：均值动作、下方初态",
        )
        ax_eval.legend(fontsize=7, loc="upper left")
        for ax in (ax_reward, ax_eval):
            ax.grid(alpha=0.2)

    def draw_ladder(self):
        """② static Phi ladder + the featured trajectory decomposition."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        potential = report["protocol"]["potential"]
        featured = report["featured_level_index"]
        entry = report["sweep"][featured]
        c_e = entry["c_e"]
        seed_index = 0
        det_states = self.data[f"det_states_{featured}_{seed_index}"]
        det_phi = self.data[f"det_phi_{featured}_{seed_index}"]
        det_shaping = self.data[f"det_shaping_{featured}_{seed_index}"]
        det_task = self.data[f"det_task_{featured}_{seed_index}"]
        arrival = entry["deterministic"][seed_index]["first_arrival_s"]
        ax_ladder, ax_pole, ax_phi, ax_reward = self.fig.subplots(2, 2).reshape(-1)
        mgl = float(potential["mgl_eff_j"])
        alphas = np.linspace(-np.pi, np.pi, 721)
        phi_static = -c_e * np.abs(mgl * (np.cos(alphas) - 1.0))
        ax_ladder.plot(alphas, phi_static, color="#0f766e")
        ax_ladder.axvline(0.0, color="gray", linestyle=":", linewidth=0.9)
        ax_ladder.set(
            xlabel="杆相对直立的角度 α（rad，ω=0）",
            ylabel="Φ（J）",
            title=f"静态梯子 Φ(α) = −cE·|E−E_top|（cE={c_e:g}）",
        )
        ax_pole.plot(
            np.arange(len(det_states)) * dt,
            np.cos(det_states[:, 1] - ref_theta),
            color="#2563eb",
        )
        ax_pole.axhspan(-1, 0, alpha=0.08, color="orange")
        arrival_text = f"{arrival:.2f} s" if arrival is not None else "未到达"
        ax_pole.set(
            ylabel="杆端相对高度",
            xlabel="仿真时间（s）",
            ylim=(-1.1, 1.1),
            title=f"典型轨迹：直立首达 {arrival_text}",
        )
        ax_phi.plot(np.arange(len(det_phi)) * dt, det_phi, color="#7c3aed")
        ax_phi.set(
            ylabel="Φ（J）",
            xlabel="仿真时间（s）",
            title="势沿轨迹（爬梯 = Φ 升向 0）",
        )
        edges = np.arange(len(det_shaping) + 1) * dt
        ax_reward.stairs(det_shaping, edges, color="#b45309", label="塑形 γΦ(s′)−Φ(s)")
        ax_reward.stairs(det_task, edges, color="#64748b", alpha=0.85, label="任务奖励")
        ax_reward.set(
            ylabel="每步奖励",
            xlabel="仿真时间（s）",
            title="每格有糖：塑形项 vs 任务项",
        )
        ax_reward.legend(fontsize=7, loc="upper right")
        for ax in (ax_ladder, ax_pole, ax_phi, ax_reward):
            ax.grid(alpha=0.2)

    def draw_outcome(self):
        """③ three-way success bars, upright arrivals, failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        colors = ("#64748b", "#b91c1c", "#2563eb", "#0f766e")
        rows = report["three_way_comparison"]
        labels = ["基线", "纯PPO\n(第29课)"] + [
            f"PBRS\n{level_label(entry)}" for entry in report["sweep"]
        ]
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        ax_rate, ax_arrival, ax_push, ax_case = self.fig.subplots(2, 2).reshape(-1)
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=colors[: len(labels)],
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
            title="三方成功率",
        )
        ax_rate.tick_params(axis="x", labelsize=7)
        any_arrival = False
        for index, entry in enumerate(report["sweep"]):
            arrival = np.asarray(entry["arrival"]["first_arrival_s_per_episode"], dtype=float)
            arrived = np.sort(arrival[~np.isnan(arrival)])
            any_arrival |= len(arrived) > 0
            ax_arrival.plot(
                np.arange(1, len(arrived) + 1),
                arrived,
                "o",
                markersize=4,
                color=colors[(index + 2) % len(colors)],
                label=f"{level_label(entry)}（{len(arrived)}/{entry['arrival']['episodes']}）",
            )
        if any_arrival:
            ax_arrival.set(
                xlabel="到达回合序号（按首达时间排序）",
                ylabel="直立区首次到达时刻（s）",
                title=f"直立首达（|α|≤{CAPTURE_ANGLE_RAD:g} rad）",
            )
            ax_arrival.legend(fontsize=7)
        else:
            ax_arrival.set_xlim(-0.5, 0.5)
            ax_arrival.set_ylim(-0.5, 0.5)
            ax_arrival.set_xticks([])
            ax_arrival.set_yticks([])
            checkpoints = [
                (entry["c_e"], record["seed_index"], record["first_arrival_eval_steps"])
                for entry in report["sweep"]
                for record in entry["training"]
                if record["first_arrival_eval_steps"] is not None
            ]
            if checkpoints:
                detail = "\n".join(
                    f"cE={c_e:g} 种子 {seed} @ {steps / 1000:.0f}k 步"
                    for c_e, seed, steps in checkpoints
                )
                note = f"评估回合从未进入直立区（第 29 课亦 0/60）\n训练检查点短暂触达：\n{detail}"
            else:
                note = "评估回合与训练检查点\n均从未进入直立区（第 29 课亦 0/60）"
            ax_arrival.text(
                0.5,
                0.55,
                note,
                ha="center",
                va="center",
                transform=ax_arrival.transAxes,
                fontsize=8,
            )
            ax_arrival.set(title="直立首达（评估 0/60）")
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
                color=colors[(featured + 2) % len(colors)],
                label=f"PBRS {level_label(featured_entry)}（种子 {seed}）",
            )
        ax_push.set(
            xlabel="推力方案编号",
            ylabel="推力结束后恢复时间（s）",
            title="±200 N 配对推力恢复",
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
            c_e_text = f"cE={case['c_e']:g}，" if case.get("c_e") is not None else ""
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{c_e_text}{failure_label(case)}",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有 PBRS 失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例：无（全部回合通过验收）")
        for ax in (ax_rate, ax_arrival, ax_push, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        if mode == "training":
            lines = ["① 这一幕在追踪：修梯子后的训练"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步，"
                f"共 {report['training']['total_env_steps'] / 1000:.0f}k 步，"
                f"墙钟 {report['training']['wall_time_s_total']:.0f} s"
            )
            lesson29 = report["lesson29_reference"]
            lines.append(
                f"  纯 PPO（第 29 课）：{lesson29['successes']}/{lesson29['episodes']}，"
                f"首次成功从未"
            )
            for entry in report["sweep"]:
                for record in entry["training"]:
                    first = record["first_successful_eval_steps"]
                    first_text = f"{first / 1000:.0f}k 步" if first is not None else "从未"
                    lines.append(
                        f"  {level_label(entry)} 种子 {record['seed_index']}："
                        f"奖励 {record['final_reward_mean']:.3f}，首次成功 {first_text}"
                    )
            lines.append("  奖励含塑形项，与第 29 课数值不可直接比")
        elif mode == "ladder":
            lines = ["② 这一幕在对照：梯子长什么样\n"]
            potential = report["protocol"]["potential"]
            lines.append(f"  Φ(s) = −cE·|E(s)−E_top|，γ = {potential['gamma']}")
            lines.append(f"  E_top = 0，正下方 E = {potential['e_rest_down_j']:.2f} J")
            lines.append(
                f"  模型常量：I = {potential['hinge_inertia_kg_m2']:.4f} kg·m²，"
                f"mgl = {potential['mgl_eff_j']:.2f} J（第 7 课同源）"
            )
            featured = report["featured_level_index"]
            entry = report["sweep"][featured]
            det = entry["deterministic"][0]
            arrival = det["first_arrival_s"]
            arrival_text = f"{arrival:.2f} s" if arrival is not None else "未到达"
            lines.append(f"  典型轨迹（{level_label(entry)} 均值动作）：首达 {arrival_text}")
            lines.append(f"    任务回报 {det['task_return']:.2f}")
            lines.append(f"    塑形回报 {det['shaping_return']:.2f}（未折扣）")
            shaping = entry["shaping"]
            lines.append(
                f"  评估回合塑形占比 {shaping['shaping_fraction_of_undiscounted_return'] * 100:.0f}%"
            )
            lines.append(
                f"    （任务 {shaping['mean_task_return']:.2f} / "
                f"塑形 {shaping['mean_shaping_return']:.2f}）"
            )
        else:
            lines = ["③ 这一幕在裁决：梯子通到了吗\n"]
            for row in report["three_way_comparison"]:
                lines.append(f"  {row['label']}：{row['successes']}/{row['episodes']}")
            lines.append("")
            for entry in report["sweep"]:
                arrival = entry["arrival"]
                median = arrival["median_first_arrival_s"]
                median_text = f"{median:.2f} s" if median is not None else "—"
                lines.append(
                    f"  {level_label(entry)}：直立首达 "
                    f"{arrival['episodes_with_arrival']}/{arrival['episodes']}，"
                    f"中位 {median_text}"
                )
            push = report["push_test"]
            lines.append(
                f"  推力：基线 {push['baseline']['successes']}/{push['baseline']['episodes']}"
            )
            for item in push["per_level"]:
                lines.append(f"    {level_label(item)}：{item['successes']}/{item['episodes']}")
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
        elif mode == "ladder":
            self.draw_ladder()
        else:
            self.draw_outcome()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：PBRS 两档的训练曲线与周期验收，对照第 29 课纯 PPO 口径",
            "ladder": "② 这一幕在对照：势函数梯子的形状，沿典型轨迹的 Φ 与每步奖励分解",
            "outcome": "③ 这一幕在裁决：基线 / 纯 PPO / PBRS 三方成功率，直立首达与失败案例",
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
    root.title("第三十一课 · 势函数奖励塑形：修梯子，不动山顶")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    PbrsDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
