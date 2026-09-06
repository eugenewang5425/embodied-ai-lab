"""Lesson 34 viewer: two-phase reward - switch the goal, not the ladder.

Three static modes share one lesson-34 recording (npz + summary):
1. training curves: per-update two-phase reward for each seed with the cited
   lesson-29 pure-PPO and lesson-31 PBRS final rewards as position reference
   lines (different objectives - not directly comparable), plus the periodic
   down-start evaluation;
2. the featured episode: the wrapped-angle trajectory with the capture cone and
   the latch moment, the energy trajectory against E_top = 0, the per-step
   reward decomposition, and the static reward landscape at the phase seam;
3. the four-way outcome: lesson-7 baseline versus the cited lesson-29 pure PPO
   and lesson-31 PBRS best tier versus the new two-phase rows, with the upright
   first arrival and the first-success statistics.
Layout and Esc handling follow the lesson-29..33 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pbrs_swingup import failure_label
from embodied_learning.experiments.twophase_swingup import EXPERIMENT, expected_npz_keys

DEFAULT_RESULTS = "results/twophase_swingup_2026-09-06"


def load_replays(directory):
    """Validate a lesson-34 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-34 recording")
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
        raise ValueError("Recording claims a broken disabled-switch pipeline guard")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    seeds = report["training"]["train_seeds"]
    per_seed = report["twophase_evaluation"]["aggregate"]["successes_per_seed"]
    count = report["twophase_evaluation"]["aggregate"]["episodes"] // seeds
    for index in range(seeds):
        terminated = data[f"eval_terminated_{index}"]
        settled = data[f"eval_settled_s_{index}"]
        if terminated.shape != (count,) or settled.shape != (count,):
            raise ValueError("Archive evaluation shapes disagree with the summary")
        successes = int(((~terminated) & (~np.isnan(settled))).sum())
        if successes != int(per_seed[index]):
            raise ValueError("Two-phase successes disagree with the archive")

    # The energy curve must be recomputable from the protocol constants.
    reward = report["protocol"]["reward"]
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    hinge, mgl = float(reward["hinge_inertia_kg_m2"]), float(reward["mgl_eff_j"])
    states = data["det_states_0"]
    recomputed = 0.5 * hinge * states[:, 3].astype(float) ** 2 + mgl * (
        np.cos(states[:, 1].astype(float) - reference[1]) - 1.0
    )
    if data["det_energy_0"].shape != recomputed.shape or (
        np.max(np.abs(data["det_energy_0"] - recomputed)) > 1e-6
    ):
        raise ValueError("Energy curve disagrees with the protocol constants")

    cases = report["failure_analysis"]["featured_cases"]
    for index, case in enumerate(cases):
        case_states = data[f"case{index}_states"]
        if case_states.ndim != 2 or case_states.shape[1] != 4:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        max_x = float(np.max(np.abs(case_states[:, 0])))
        if abs(max_x - case["max_abs_cart_position_m"]) > 1e-4:
            raise ValueError(f"Featured case {index} cart excursion disagrees with its arrays")
    return {"report": report, **data}


class TwophaseDemo:
    """Two-phase reward versus the lesson-29 pure PPO, the lesson-31 PBRS tier."""

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
            text="第三十四课 · 两阶段奖励：荡起阶段追能量，进锥锁存平衡（纯 PPO）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "第 29 课 PPO 原样（同网络/预算/课程），只改奖励：荡起段 −cE·|E−E_top|（正下方恰为 −0.75），"
                "|α|≤0.3 rad 锁存切平衡段（第 29 课奖励原形），失败步只减 1.0（非 −10）。\n"
                "三个模式共用同一条正式记录：① 训练曲线 ② 典型回合的能量-角度双轨迹与相位切换 "
                "③ 四方对照与首达/首成统计；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="training")
        for label, key in (
            ("① 训练曲线：两阶段 vs 纯 PPO vs PBRS 口径", "training"),
            ("② 典型回合：能量-角度双轨迹与阶段切换", "episode"),
            ("③ 四方对照与首达/首成统计", "outcome"),
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
                "文献口径：两阶段/分阶段目标是摆起 RL 的标准做法（MDPI 2024；Dulac-Arnold 2021 称"
                " stage-switching）。与第 31 课 PBRS 的区别：那是任务奖励+势函数项，这里是按阶段直接换目标。"
                "成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_training(self):
        """① two-phase reward curves and periodic evaluations, per seed."""
        self.fig.clear()
        report = self.report
        seeds = report["training"]["train_seeds"]
        ax_reward, ax_eval = self.fig.subplots(1, 2)
        updates = np.arange(1, report["hyperparameters"]["updates"] + 1)
        curves = np.stack([self.data[f"reward_curve_{seed}"] for seed in range(seeds)], axis=0)
        for seed in range(seeds):
            ax_reward.plot(updates, curves[seed], alpha=0.3, linewidth=0.8, color="#64748b")
        ax_reward.plot(
            updates,
            curves.mean(axis=0),
            color="#0f766e",
            linewidth=1.8,
            label=f"两阶段（{seeds} 种子均值）",
        )
        lesson29 = float(np.mean(report["references"]["lesson29"]["final_reward_mean_per_seed"]))
        ax_reward.axhline(
            lesson29,
            color="#b91c1c",
            linestyle="--",
            linewidth=1.2,
            label="纯 PPO（第 29 课）末段均值",
        )
        for tier, values in report["references"]["lesson31_pbrs_final_reward_per_tier"].items():
            ax_reward.axhline(
                float(np.mean(values)),
                color="#b45309",
                linestyle=":",
                linewidth=1.0,
                label=f"PBRS cE={tier}（第 31 课）",
            )
        ax_reward.set(
            xlabel="PPO 更新轮次",
            ylabel="批内平均奖励（两阶段口径，每步）",
            title="训练奖励：细线 = 单个种子（虚/点线为他课口径，仅位置参照）",
        )
        ax_reward.legend(fontsize=7, loc="lower right")
        per_seed = report["per_seed"]
        steps = (
            np.asarray([point["env_steps"] for point in per_seed[0]["eval_curve"]], dtype=float)
            / 1000.0
        )
        success = np.asarray(
            [[int(point["success"]) for point in record["eval_curve"]] for record in per_seed],
            dtype=float,
        )
        for seed in range(success.shape[0]):
            ax_eval.plot(steps, success[seed], "o--", markersize=4, alpha=0.5, color="#64748b")
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

    def draw_episode(self):
        """② the featured episode: angle/energy trajectories and the phase seam."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        reward = report["protocol"]["reward"]
        alpha_switch = float(reward["alpha_switch_rad"])
        mgl = float(reward["mgl_eff_j"])
        c_e_sw = float(reward["c_e_switch"])
        featured = report["featured_seed_index"]
        det = report["per_seed"][featured]["deterministic"]
        states = self.data[f"det_states_{featured}"]
        energy = self.data[f"det_energy_{featured}"]
        phase = self.data[f"det_phase_{featured}"]
        upright = self.data[f"det_upright_{featured}"]
        energy_r = self.data[f"det_energy_r_{featured}"]
        latch_step = int(np.argmax(phase)) if phase.any() else None
        ax_land, ax_angle, ax_energy, ax_reward = self.fig.subplots(2, 2).reshape(-1)
        alphas = np.linspace(-np.pi, np.pi, 1441)
        landscape = np.where(
            np.abs(alphas) <= alpha_switch,
            (1.0 + np.cos(alphas)) / 2.0 + 0.25,
            -c_e_sw * np.abs(mgl * (np.cos(alphas) - 1.0)) + float(reward["alive_swing"]),
        )
        ax_land.plot(alphas, landscape, color="#0f766e", linewidth=1.5)
        ax_land.axvspan(-alpha_switch, alpha_switch, alpha=0.15, color="#10b981")
        for seam in (-alpha_switch, alpha_switch):
            ax_land.axvline(seam, color="#b45309", linestyle=":", linewidth=1.0)
        ax_land.set(
            xlabel="杆相对直立的角度 α（rad，ω=0）",
            ylabel="每步奖励",
            ylim=(-1.15, 1.55),
            title=f"奖励地形：接缝 |α|={alpha_switch:g} rad",
        )
        ts = np.arange(len(states)) * dt
        ax_angle.plot(ts, np.cos(states[:, 1] - ref_theta), color="#2563eb", linewidth=1.1)
        ax_angle.axhspan(-1, 0, alpha=0.08, color="orange")
        if latch_step is not None:
            ax_angle.axvspan(latch_step * dt, ts[-1], alpha=0.18, color="#10b981")
            ax_angle.axvline(latch_step * dt, color="#b45309", linewidth=1.2, label="锁存时刻")
            ax_angle.legend(fontsize=7, loc="lower right")
        latch_text = f"{latch_step * dt:.2f} s" if latch_step is not None else "未进入"
        ax_angle.set(
            ylabel="杆端相对高度",
            xlabel="仿真时间（s）",
            ylim=(-1.1, 1.1),
            title=f"典型回合（种子 {featured}）：锁存 {latch_text}，"
            f"平衡步占比 {det['phase']['balance_step_fraction'] * 100:.0f}%",
        )
        ax_energy.plot(np.arange(len(energy)) * dt, energy, color="#7c3aed", linewidth=1.2)
        ax_energy.axhline(0.0, color="gray", linestyle=":", linewidth=0.9)
        ax_energy.axhline(-2.0 * mgl, color="gray", linestyle=":", linewidth=0.9)
        ax_energy.set(
            ylabel="杆机械能 E（J）",
            xlabel="仿真时间（s）",
            title="能量轨迹：荡起段的糖 = |E| 向 E_top = 0 收敛",
        )
        edges = np.arange(len(energy_r) + 1) * dt
        ax_reward.stairs(energy_r, edges, color="#b45309", label="能量项（荡起）")
        ax_reward.stairs(upright, edges, color="#0f766e", label="直立项（平衡）")
        ax_reward.set(
            ylabel="每步奖励分量",
            xlabel="仿真时间（s）",
            title="每步奖励分解（失败步另减 1.0）",
        )
        ax_reward.legend(fontsize=7, loc="upper right")
        for ax in (ax_land, ax_angle, ax_energy, ax_reward):
            ax.grid(alpha=0.2)

    def draw_outcome(self):
        """③ four-way success bars, arrivals, first success, failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        colors = ("#64748b", "#b91c1c", "#b45309", "#0f766e")
        rows = report["four_way_comparison"]
        ax_rate, ax_arrival, ax_first, ax_case = self.fig.subplots(2, 2).reshape(-1)
        labels = ["基线\n(第7课)", "纯PPO\n(第29课)", "PBRS cE=2\n(第31课)", "两阶段\n(本课)"]
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        bars = ax_rate.bar(
            labels,
            [s / t * 100 for s, t in zip(successes, totals, strict=True)],
            color=colors,
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
        ax_rate.set(ylabel="验收通过率（%）", ylim=(0, 112), title="四方成功率（第 7 课口径）")
        ax_rate.tick_params(axis="x", labelsize=7)
        arrival = report["twophase_evaluation"]["arrival"]
        arrived = np.sort(
            np.asarray([v for v in arrival["first_arrival_s_per_episode"] if v is not None])
        )
        checkpoints = [
            (record["seed_index"], record["first_arrival_eval_steps"])
            for record in report["per_seed"]
            if record["first_arrival_eval_steps"] is not None
        ]
        if len(arrived):
            ax_arrival.plot(
                np.arange(1, len(arrived) + 1),
                arrived,
                "o",
                markersize=4,
                color="#0f766e",
                label=f"评估首达 {len(arrived)}/{arrival['episodes']}",
            )
            ax_arrival.set(
                xlabel="到达回合序号（按首达时间排序）",
                ylabel="直立区首次到达时刻（s）",
                title=f"直立首达（|α|≤{report['protocol']['reward']['alpha_switch_rad']:g} rad）",
            )
            ax_arrival.legend(fontsize=7)
        else:
            ax_arrival.set_xlim(-0.5, 0.5)
            ax_arrival.set_ylim(-0.5, 0.5)
            ax_arrival.set_xticks([])
            ax_arrival.set_yticks([])
            if checkpoints:
                detail = "；".join(
                    f"种子 {seed} @ {steps / 1000:.0f}k 步" for seed, steps in checkpoints
                )
                note = (
                    f"评估回合未进入直立区\n训练检查点触达：{detail}\n（第 31 课：一次 @ 150k 步）"
                )
            else:
                note = "评估回合与训练检查点\n均从未进入直立区\n（第 29 课：从未；第 31 课：一次）"
            ax_arrival.text(
                0.5,
                0.55,
                note,
                ha="center",
                va="center",
                transform=ax_arrival.transAxes,
                fontsize=8,
            )
            ax_arrival.set(title="直立首达")
        first_steps = [record["first_successful_eval_steps"] for record in report["per_seed"]]
        if any(step is not None for step in first_steps):
            ax_first.bar(
                np.arange(len(first_steps)),
                [np.nan if v is None else v / 1000.0 for v in first_steps],
                color="#0f766e",
                width=0.5,
            )
            ax_first.set(
                xlabel="训练种子",
                ylabel="首次成功步数（×1000）",
                title="首次成功（训练中周期评估；第 29/31/32 课从未）",
            )
        else:
            ax_first.text(
                0.5,
                0.55,
                "首次成功：从未（预算内）\n第 29 课：从未\n第 31 课：从未；第 32 课：从未",
                ha="center",
                va="center",
                transform=ax_first.transAxes,
                fontsize=8,
            )
            ax_first.set(title="首次成功（训练中周期评估）")
        ax_first.set_xticks([0, 1, 2])
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
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例：{failure_label(case)}（{case['kind']}）",
            )
        else:
            ax_case.text(
                0.5,
                0.5,
                "本记录没有两阶段奖励失败回合",
                ha="center",
                va="center",
                transform=ax_case.transAxes,
            )
            ax_case.set(title="失败案例：无（全部回合通过验收）")
        for ax in (ax_rate, ax_arrival, ax_first, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        reward = report["protocol"]["reward"]
        if mode == "training":
            lines = ["① 这一幕在追踪：换目标之后的训练"]
            lines.append(
                f"  每种子 {report['training']['env_steps_per_seed'] / 1000:.0f}k 环境步，"
                f"共 {report['training']['total_env_steps'] / 1000:.0f}k 步，"
                f"墙钟 {report['training']['wall_time_s_total']:.0f} s"
            )
            for record in report["per_seed"]:
                first = record["first_successful_eval_steps"]
                first_text = f"{first / 1000:.0f}k 步" if first is not None else "从未"
                arrival = record["first_arrival_eval_steps"]
                arrival_text = f"{arrival / 1000:.0f}k 步" if arrival is not None else "从未"
                lines.append(
                    f"  种子 {record['seed_index']}：奖励 {record['final_reward_mean']:.3f}，"
                    f"首成 {first_text}，首达 {arrival_text}"
                )
            lines.append(
                f"  训练期平衡步占比 "
                f"{np.mean([r['training_phase_stats']['balance_step_fraction'] for r in report['per_seed']]) * 100:.0f}%"
            )
            lines.append("  虚/点线为他课奖励口径，仅位置参照")
        elif mode == "episode":
            lines = ["② 这一幕在对照：典型回合长什么样\n"]
            lines.append(f"  α_switch = {reward['alpha_switch_rad']:g} rad（锁存到回合结束）")
            lines.append(f"  cE_sw = {float(reward['c_e_switch']):.4f} = 1/(2·mgl)")
            lines.append(f"  E_top = 0，正下方 E = {float(reward['e_rest_down_j']):.2f} J")
            lines.append("  荡起段：−cE·|E−E_top| + 0.25（正下方 = −0.75）")
            lines.append("  平衡段：第 29 课奖励原形（无控制代价差）")
            featured = report["featured_seed_index"]
            det = report["per_seed"][featured]["deterministic"]
            lines.append(f"  典型回合（种子 {featured}，均值动作）：")
            lines.append(f"    平衡步占比 {det['phase']['balance_step_fraction'] * 100:.0f}%")
            sugar = det["sugar_fraction"]
            lines.append(
                f"    荡起步有糖占比（|E| 收敛） {sugar * 100:.0f}%"
                if sugar is not None
                else "    无荡起步（全程平衡段）"
            )
            lines.append(f"    回报 {det['return']:.2f}")
        else:
            lines = ["③ 这一幕在裁决：四方对照\n"]
            for row in report["four_way_comparison"]:
                lines.append(f"  {row['label']}：{row['successes']}/{row['episodes']}")
            lines.append("")
            arrival = report["twophase_evaluation"]["arrival"]
            median = arrival["median_first_arrival_s"]
            median_text = f"{median:.2f} s" if median is not None else "—"
            lines.append(
                f"  两阶段：直立首达 {arrival['episodes_with_arrival']}/{arrival['episodes']}，"
                f"中位 {median_text}"
            )
            lines.append(
                f"  首次成功：{'出现' if report['hypothesis']['first_success_any'] else '从未'}"
            )
            push = report["push_test"]
            lines.append(
                f"  推力：基线 {push['baseline']['successes']}/{push['baseline']['episodes']}，"
                f"两阶段 {push['aggregate']['successes']}/{push['aggregate']['episodes']}"
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
        elif mode == "episode":
            self.draw_episode()
        else:
            self.draw_outcome()
        self.fill_stats(mode)
        status = {
            "training": "① 这一幕在追踪：两阶段奖励的训练曲线与周期验收，对照第 29/31 课口径",
            "episode": "② 这一幕在对照：典型回合的能量-角度双轨迹、锁存时刻与每步奖励分解",
            "outcome": "③ 这一幕在裁决：基线 / 纯 PPO / PBRS / 两阶段四方成功率，首达与首成",
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
    root.title("第三十四课 · 两阶段奖励：换目标，不修梯子")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    TwophaseDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
