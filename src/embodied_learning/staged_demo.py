"""Lesson 41 viewer: staged minimal verification - base, then minimal fixes.

Three static modes share one lesson-41 recording (npz + summary):
1. stage 1, the base re-verification: the frozen lesson-7 controller on the
   exact down start (20/20 at 4.76 s expected) with the handoff window - the
   per-step transition from |alpha| < 0.3 rad into the 0.01 rad tolerance -
   highlighted as the reference trajectory of the learning stages;
2. stage 2, the 5-parameter linear handoff correction versus the base on the
   same start: trajectories, the window zoom, the motor input with its
   residual component and the residual against its 5 N budget;
3. stage 3 (only present when stage 2 improved): the MLP corrections versus
   the base, plus the A/B/C/D comparison table.  When stage 2 did not
   improve, the mode shows the comparison of what ran and the step-3 gate
   that skipped groups C/D by protocol.

Layout and Esc handling follow docs/26 section 6: every mode calls fig.clear()
first, redraw draws synchronously, every quoted number is traceable to
summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.staged_verification import (
    EXPERIMENT,
    GEAR,
    SCHEMA_VERSION,
    correction_features,
    expected_npz_keys,
    handoff_window,
)

DEFAULT_RESULTS = "results/staged_verification_2026-09-06"


def settled_text(value):
    return f"{value:.2f} s" if value is not None else "未到达"


def _zeros(count):
    return np.zeros(count)


def _recomputed_metrics(data, prefix, limit_note=""):
    """Recovery metrics recomputed from one archived episode (loader check)."""
    from embodied_learning.experiments.swingup_comparison import recovery_metrics

    states = data[f"{prefix}_states"]
    controls = data[f"{prefix}_controls"]
    modes = data[f"{prefix}_modes"]
    report = data["report"]
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    dt = report["protocol"]["dt_s"]
    arrays = {
        "states": states,
        "controls": controls,
        "applied_force_n": _zeros(len(controls)),
        "scheduled_force_n": _zeros(report["protocol"]["eval_horizon_steps"]),
        "modes": modes,
        "end_flags": np.array([False, len(controls) >= report["protocol"]["eval_horizon_steps"]]),
    }
    metrics = recovery_metrics(arrays, {"failure_reason": ""}, reference, dt)
    if limit_note:
        residuals = data[f"{prefix}_residuals_n"]
        active = data[f"{prefix}_active"]
        if np.max(np.abs(residuals[active]), initial=0.0) > limit_note + 1e-6:
            raise ValueError(f"{prefix} residual exceeds its budget")
        if np.max(np.abs(residuals[~active]), initial=0.0) > 0.0:
            raise ValueError(f"{prefix} residual is nonzero outside the handoff window")
    return metrics


def load_replays(directory):
    """Validate a lesson-41 recording; returns the data dict for the demo.

    Tamper routes rejected here: (1) the npz SHA-256 against the summary and
    the archive key set against the implied key set; (2) the stage-1 base
    metrics and the handoff window recomputed from the archive against the
    summary; (3) every run group's acceptance metrics recomputed from its
    stored episode against the summary, plus the residual budget and the
    window-only activity contract.
    """
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible lesson-41 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    data["report"] = report

    # stage 1: the base run must reproduce the summary, window included
    step1 = report["step1_base"]
    if not step1["deterministic_identical_repeats"]:
        raise ValueError("Recording claims non-identical base repeats; cannot cross-check")
    metrics = _recomputed_metrics(data, "baseline")
    first = step1["per_episode"][0]
    if (
        metrics["recovered"] != first["recovered"]
        or metrics["settled_at_s"] != first["settled_at_s"]
    ):
        raise ValueError("Base acceptance disagrees with the archive")
    successes = sum(row["recovered"] for row in step1["per_episode"])
    if successes != step1["successes"]:
        raise ValueError("Base successes disagree with the summary")
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    window = handoff_window(data["baseline_modes"], data["baseline_states"], reference[1])
    handoff = step1["handoff"]
    if window is None or (window[0], window[1]) != (handoff["capture_step"], handoff["end_step"]):
        raise ValueError("Handoff window disagrees with the archive")
    start_state = correction_features(data["handoff_states"][0], reference[1])
    if not np.allclose(start_state, handoff["start_state"], atol=1e-12):
        raise ValueError("Handoff start state disagrees with the archive")

    # run groups: acceptance and residual contracts recomputed from arrays
    prefixes = {"B_linear": ("lin", report["protocol"]["residual_limits_n"]["B_linear"])}
    if report["groups"]["C_mlp"] is not None:
        prefixes["C_mlp"] = ("mlp5", report["protocol"]["residual_limits_n"]["C_mlp"])
    if report["groups"]["D_mlp_large"] is not None:
        prefixes["D_mlp_large"] = ("mlp25", report["protocol"]["residual_limits_n"]["D_mlp_large"])
    for key, (prefix, limit) in prefixes.items():
        evaluation = report["groups"][key]["evaluation"]
        if not evaluation["deterministic_identical_repeats"]:
            raise ValueError(f"{key} claims non-identical repeats; cannot cross-check")
        metrics = _recomputed_metrics(data, f"{prefix}_eval", limit_note=limit)
        if metrics["recovered"] != (evaluation["successes"] == evaluation["episodes"]):
            raise ValueError(f"{key} success disagrees with the archive")
        if (
            abs((metrics["settled_at_s"] or 0.0) - (evaluation["median_settled_at_s"] or 0.0))
            > 1e-9
        ):
            raise ValueError(f"{key} settled time disagrees with the archive")
        archived = data[f"{prefix}_eval_residuals_n"][data[f"{prefix}_eval_active"]]
        mean_abs = float(np.mean(np.abs(archived))) if archived.size else 0.0
        if abs(mean_abs - evaluation["residual_mean_abs_n"]) > 1e-9:
            raise ValueError(f"{key} residual usage disagrees with the archive")
    return data


class StagedDemo:
    """Staged verification: the frozen base under minimal learned corrections."""

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
            text="第四十一课 · 分阶段最小验证：底座 → 线性交接修正 →（如过门）MLP 修正",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "底座 = 第 7 课能量整形+LQR（冻结）；学习 = 交接段内的极小修正 δu（限幅 ±5 N，"
                "底座力矩的 1.7%）；u = clip(u_base + δu, ±300 N)。第三步只在第二步改善后运行"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="base")
        for label, key in (
            ("① 底座复验 + 交接段轨迹", "base"),
            ("② 线性修正 vs 底座对照", "linear"),
            ("③ MLP 修正（如有）与对照表", "mlp"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        self.stats = ttk.Label(middle, width=52, anchor="nw", justify="left", wraplength=430)
        self.stats.pack(side="right", fill="y", padx=(12, 0))
        left = ttk.Frame(middle, width=1030, height=430)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)
        self.fig = Figure(figsize=(10.4, 4.3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        verdict = self.report["verdict"]
        ttk.Label(
            outer,
            text=(
                f"裁决：线性修正{'改善' if verdict['linear_improved'] else '未改善'}；"
                f"第三步{'已运行' if verdict['mlp_ran'] else '按协议跳过'}。"
                "验收 = 第 7 课口径（四态误差容差尾段 ≥2 s，精确下方初态）"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ---------------------------------------------------------------- helpers
    def _dt(self):
        return self.report["protocol"]["dt_s"]

    def _ref_theta(self):
        return self.report["protocol"]["reference_state"][1]

    def _handoff(self):
        return self.report["step1_base"]["handoff"]

    def _draw_modes(self, ax):
        modes = self.data["baseline_modes"]
        dt = self._dt()
        edges = np.arange(len(modes) + 1) * dt
        ax.stairs(
            np.where(modes == "balance", 2, np.where(modes == "kick", 0, 1)),
            edges,
            color="#0f766e",
        )
        ax.set(
            yticks=[0, 1, 2],
            yticklabels=["启动", "摆起", "LQR"],
            ylabel="控制模式",
            xlabel="仿真时间（s）",
            title="底座控制模式（切换记录）",
        )

    # ------------------------------------------------------------------ modes
    def draw_base(self):
        """1 base re-verification with the handoff window highlighted."""
        self.fig.clear()
        dt = self._dt()
        ref_theta = self._ref_theta()
        handoff = self._handoff()
        data = self.data
        ax_height, ax_alpha, ax_force, ax_mode = self.fig.subplots(2, 2).reshape(-1)
        states = data["baseline_states"]
        ts = np.arange(len(states)) * dt
        ax_height.plot(ts, np.cos(states[:, 1] - ref_theta), color="#0f766e")
        ax_height.axvspan(
            handoff["window_start_s"], handoff["window_end_s"], alpha=0.2, color="#f59e0b"
        )
        ax_height.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_height.axvline(handoff["capture_time_s"], color="#64748b", linestyle="--", linewidth=0.9)
        settled = self.report["step1_base"]["median_settled_at_s"]
        if settled is not None:
            ax_height.axvline(settled, color="#b91c1c", linestyle=":", linewidth=0.9)
        ax_height.set(
            ylabel="杆端相对高度 cos(α)",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"①底座复验 {self.report['step1_base']['successes']}/"
            f"{self.report['step1_base']['episodes']}：橙色带 = 交接段",
        )
        ax_alpha.plot(
            data["handoff_times"],
            data["handoff_alpha"],
            "o-",
            markersize=3.5,
            color="#b45309",
            label="交接段 |α|: 0.3 → 0.01 rad",
        )
        for bound in (-0.3, 0.3):
            ax_alpha.axhline(bound, color="#64748b", linestyle="--", linewidth=0.9)
        for bound in (-0.01, 0.01):
            ax_alpha.axhline(bound, color="#b91c1c", linestyle=":", linewidth=0.9)
        ax_alpha.set(
            xlabel="仿真时间（s）",
            ylabel="包裹角误差 α（rad）",
            title="交接段放大：捕获 0.3 rad（虚线）→ 容差 0.01 rad（点线）",
        )
        ax_alpha.legend(fontsize=7)
        controls = data["baseline_controls"]
        edges = np.arange(len(controls) + 1) * dt
        ax_force.stairs(controls * GEAR, edges, color="#2563eb")
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="底座电机输入（±300 N 限幅）",
        )
        self._draw_modes(ax_mode)
        for ax in (ax_height, ax_alpha, ax_force, ax_mode):
            ax.grid(alpha=0.2)

    def draw_linear(self):
        """2 the linear correction versus the base, same down start."""
        self.fig.clear()
        dt = self._dt()
        ref_theta = self._ref_theta()
        data = self.data
        report = self.report
        ax_height, ax_alpha, ax_force, ax_res = self.fig.subplots(2, 2).reshape(-1)
        base_states = data["baseline_states"]
        lin_states = data["lin_eval_states"]
        ts_base = np.arange(len(base_states)) * dt
        ts_lin = np.arange(len(lin_states)) * dt
        ax_height.plot(
            ts_base, np.cos(base_states[:, 1] - ref_theta), "--", color="#64748b", label="底座"
        )
        ax_height.plot(
            ts_lin, np.cos(lin_states[:, 1] - ref_theta), color="#2563eb", label="底座+线性修正"
        )
        ax_height.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_height.set(
            ylabel="杆端相对高度 cos(α)",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="②同一下方初态：底座 vs 线性修正",
        )
        ax_height.legend(fontsize=8)
        window = handoff_window(data["lin_eval_modes"], lin_states, ref_theta)
        if window is not None:
            capture, end = window
            lin_alpha = lin_states[capture : end + 1, 1] - ref_theta
            lin_alpha = np.arctan2(np.sin(lin_alpha), np.cos(lin_alpha))
            ax_alpha.plot(
                np.arange(capture, end + 1) * dt,
                lin_alpha,
                "o-",
                markersize=3,
                color="#2563eb",
                label="线性修正评估回合",
            )
        ax_alpha.plot(
            data["handoff_times"],
            data["handoff_alpha"],
            "s--",
            markersize=3,
            color="#b45309",
            label="底座交接段",
        )
        for bound in (-0.01, 0.01):
            ax_alpha.axhline(bound, color="#b91c1c", linestyle=":", linewidth=0.9)
        ax_alpha.set(
            xlabel="仿真时间（s）",
            ylabel="包裹角误差 α（rad）",
            title="交接段对照（点线 = 0.01 rad 容差）",
        )
        ax_alpha.legend(fontsize=7)
        base_controls = data["baseline_controls"]
        lin_controls = data["lin_eval_controls"]
        residuals = data["lin_eval_residuals_n"]
        limit_n = report["protocol"]["residual_limits_n"]["B_linear"]
        edges = np.arange(len(base_controls) + 1) * dt
        ax_force.stairs(base_controls * GEAR, edges, color="#64748b", label="底座")
        ax_force.stairs(lin_controls * GEAR, edges, color="#2563eb", alpha=0.85, label="合计输入")
        ax_force.plot(
            np.arange(len(residuals)) * dt,
            residuals,
            color="#b45309",
            linewidth=1.0,
            label="残差 δu",
        )
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="电机输入与残差分量（±300 N 总限幅）",
        )
        ax_force.legend(fontsize=8)
        active = data["lin_eval_active"]
        ax_res.fill_between(
            np.flatnonzero(active) * dt,
            np.abs(residuals[active]),
            step="mid",
            alpha=0.55,
            color="#2563eb",
            label="|δu|（交接段内）",
        )
        ax_res.axhline(limit_n, color="#b91c1c", linestyle=":", linewidth=1.1)
        ax_res.set(
            xlabel="仿真时间（s）",
            ylabel="|残差 δu|（N）",
            xlim=(0.0, len(residuals) * dt),
            title="残差幅值（交接段外恒为 0）",
        )
        ax_res.text(
            0.4,
            limit_n,
            f"预算 ±{limit_n:.0f} N",
            va="bottom",
            fontsize=8,
            color="#b91c1c",
        )
        ax_res.legend(fontsize=8, loc="center right")
        for ax in (ax_height, ax_alpha, ax_force, ax_res):
            ax.grid(alpha=0.2)

    def draw_mlp(self):
        """3 MLP corrections when they ran, else the gate + comparison table."""
        self.fig.clear()
        dt = self._dt()
        ref_theta = self._ref_theta()
        data = self.data
        report = self.report
        mlp_ran = report["verdict"]["mlp_ran"]
        ax_height, ax_bars, ax_settle, ax_gate = self.fig.subplots(2, 2).reshape(-1)
        series = [("lin", "#2563eb", "线性修正（B）")]
        if mlp_ran:
            series.append(("mlp5", "#0f766e", "MLP ±5 N（C）"))
            series.append(("mlp25", "#7c3aed", "MLP ±25 N（D）"))
        base_states = data["baseline_states"]
        ts = np.arange(len(base_states)) * dt
        ax_height.plot(
            ts, np.cos(base_states[:, 1] - ref_theta), "--", color="#64748b", label="底座（A）"
        )
        for prefix, color, label in series:
            states = data[f"{prefix}_eval_states"]
            ax_height.plot(
                np.arange(len(states)) * dt,
                np.cos(states[:, 1] - ref_theta),
                color=color,
                alpha=0.9 if prefix != "lin" else 1.0,
                linewidth=1.0,
                label=label,
            )
        ax_height.axhspan(-1, 0, alpha=0.08, color="orange")
        ax_height.set(
            ylabel="杆端相对高度 cos(α)",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="③同一下方初态：全部已运行组",
        )
        ax_height.legend(fontsize=7)
        rows = [row for row in report["comparison_table"] if row["status"] == "ran"]
        labels = [f"{row['group']}:{row['label']}" for row in rows]
        bar_colors = ("#64748b", "#2563eb", "#0f766e", "#7c3aed")
        bars = ax_bars.bar(
            labels,
            [row["successes"] / row["episodes"] * 100 for row in rows],
            color=[bar_colors[index % len(bar_colors)] for index in range(len(labels))],
            width=0.55,
        )
        for bar, row in zip(bars, rows, strict=True):
            ax_bars.annotate(
                f"{row['successes']}/{row['episodes']}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax_bars.set(
            ylabel="验收通过率（%）",
            ylim=(0, 116),
            title="A/B（+C/D 如已运行）成功率（第 7 课口径）",
        )
        ax_bars.tick_params(axis="x", labelsize=7)
        settled = [row["median_settled_at_s"] for row in rows]
        ax_settle.bar(
            labels,
            [value if value is not None else 0.0 for value in settled],
            color=[bar_colors[index % len(bar_colors)] for index in range(len(labels))],
            width=0.55,
        )
        baseline_settled = report["step1_base"]["median_settled_at_s"]
        if baseline_settled is not None:
            ax_settle.axhline(baseline_settled, color="#b91c1c", linestyle="--", linewidth=1.0)
            ax_settle.text(
                0.02,
                baseline_settled,
                f" 底座 {baseline_settled:.2f} s",
                va="bottom",
                fontsize=8,
                color="#b91c1c",
            )
        for index, row in enumerate(rows):
            if row["median_settled_at_s"] is not None:
                ax_settle.annotate(
                    f"{row['median_settled_at_s']:.2f} s",
                    (index, row["median_settled_at_s"]),
                    ha="center",
                    xytext=(0, 3),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax_settle.set(
            ylabel="稳定时刻（s，自 t=0）",
            ylim=(0, max(baseline_settled or 5.0, 5.0) * 1.25),
            xlabel="组",
            title="稳定时间对照（更短 = 更好）",
        )
        ax_settle.tick_params(axis="x", labelsize=7)
        if mlp_ran:
            ax_gate.text(
                0.5,
                0.55,
                "第三步已运行：C/D 两档 MLP 修正评估数字见对照表与左列",
                ha="center",
                va="center",
                transform=ax_gate.transAxes,
                fontsize=10,
            )
            ax_gate.set(title="第三步说明", xticks=[], yticks=[])
        else:
            ax_gate.axis("off")
            ax_gate.text(
                0.5,
                0.62,
                "第三步按协议跳过",
                ha="center",
                va="center",
                transform=ax_gate.transAxes,
                fontsize=13,
                fontweight="bold",
                color="#b91c1c",
            )
            ax_gate.text(
                0.5,
                0.38,
                "第二部门槛：成功率保持 20/20 且稳定时间严格改善\n"
                "实际结果：20/20，稳定 4.76 s（与底座相同）\n"
                "→ 极小线性修正（|δu| ≤ 5 N）没有可兑现的增值，\n"
                "  MLP（C/D）按协议不运行，如实入库",
                ha="center",
                va="center",
                transform=ax_gate.transAxes,
                fontsize=9.5,
            )
            ax_gate.set(title="第三步门槛（改善才运行）")
        for ax in (ax_height, ax_bars, ax_settle):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        step1 = report["step1_base"]
        handoff = step1["handoff"]
        linear = report["groups"]["B_linear"]
        evaluation = linear["evaluation"]
        if mode == "base":
            lines = ["① 这一幕在验证：底座（冻结，不训练）"]
            lines.append(
                f"  底座复验：{step1['successes']}/{step1['episodes']}，"
                f"中位稳定 {settled_text(step1['median_settled_at_s'])}，"
                f"20 次逐位一致：{step1['deterministic_identical_repeats']}"
            )
            lines.append(
                f"  交接段：捕获 {handoff['capture_time_s']:.2f} s（第 {handoff['capture_step']} 步）→ "
                f"进入容差 {handoff['window_end_s']:.2f} s（第 {handoff['end_step']} 步），"
                f"共 {handoff['steps']} 步"
            )
            start = handoff["start_state"]
            lines.append(
                f"  段首状态 [x, α, v, ω]：[{start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}, {start[3]:.3f}]"
            )
            lines.append("  该段逐帧（状态+动作+时间）已存档，作为后续学习的参考轨迹")
        elif mode == "linear":
            lines = ["② 这一幕在检验：5 参数线性交接修正（限幅 ±5 N）"]
            lines.append(
                f"  训练：监督初始化 δu≡0 → ±0.05 N 轻扰动探索 {linear['training']['exploration']['rollouts']} 次"
                f"（全部稳定 {settled_text(linear['training']['update']['best_settled_at_s'])}）→ 最短一次最小二乘更新"
            )
            lines.append(
                f"  评估：{evaluation['successes']}/{evaluation['episodes']}，"
                f"稳定 {settled_text(evaluation['median_settled_at_s'])}"
                f"（底座 {settled_text(report['step1_base']['median_settled_at_s'])}），"
                f"捕获 {evaluation['capture_time_s']:.2f} s"
            )
            lines.append(
                f"  残差 |δu| 均值 {evaluation['residual_mean_abs_n']:.3f} N、"
                f"峰值 {evaluation['residual_max_abs_n']:.3f} N（预算 5 N，触限率 "
                f"{evaluation['fraction_at_limit'] * 100:.0f}%）"
            )
            verdict = "有价值" if linear["improved"] else "无改善（不劣化，但也没变快）"
            lines.append(f"  判定：{verdict}")
        else:
            lines = ["③ 这一幕在收口：第三步是否运行 + 总对照表"]
            verdict = report["verdict"]
            lines.append(
                f"  线性修正改善：{verdict['linear_improved']}；第三步运行：{verdict['mlp_ran']}"
            )
            for row in report["comparison_table"]:
                if row["status"] == "ran":
                    lines.append(
                        f"  {row['group']}（{row['label']}）：{row['successes']}/{row['episodes']}，"
                        f"稳定 {settled_text(row['median_settled_at_s'])}"
                    )
                else:
                    lines.append(f"  {row['group']}（{row['label']}）：未运行（第二步未过改善门）")
            lines.append(f"  结论：{verdict['conclusion'].split(':')[0]}")
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "base":
            self.draw_base()
        elif mode == "linear":
            self.draw_linear()
        else:
            self.draw_mlp()
        self.fill_stats(mode)
        status = {
            "base": "① 这一幕在验证：底座 20/20 复验 + 交接段（0.3 rad → 0.01 rad）逐帧参考轨迹",
            "linear": "② 这一幕在检验：5 参数线性修正 vs 底座（同初态、同验收口径、限幅 ±5 N）",
            "mlp": "③ 这一幕在收口：MLP 修正（如第二步过门）与 A/B/C/D 对照表",
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
    root.title("第四十一课 · 分阶段最小验证：底座 → 线性交接修正 →（如过门）MLP 修正")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    StagedDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
