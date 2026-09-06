"""Lesson 33 viewer: Go-Explore - the archive, the stable band and the walk.

Three static modes share one lesson-33 recording (npz + summary):
1. the archive: occupancy of the (pole angle 12 x cart position 6) grid, the
   cell-selection trace and the coverage curve with the first stable-band
   capture marked - "the map the agent drew of its own cliff";
2. the capture: the first kept capture's segment trajectories (pole height,
   cart position, angular velocity, motor input) with the >= 2 s settled tail
   shaded - the moment the cliff top became a returnable archive state;
3. the outcome: the four cliff-crossing routes on the lesson-7 acceptance
   (baseline / pure PPO / DAPG airdrop / Go-Explore+BC), the robustified BC
   policy closed-loop from the exact down start against the baseline, the BC
   loss curve and the featured failure case.
Layout and Esc handling follow the lesson-28..32 demos: every mode calls
fig.clear() first, redraw draws synchronously, and every quoted number is
traceable to summary.json / trajectories.npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.goexplore_swingup import (
    CAPTURE_TAIL_S,
    EXPERIMENT,
    SCHEMA_VERSION,
    capture_slice_metrics,
    expected_npz_keys,
    failure_label_cn,
)

DEFAULT_RESULTS = "results/goexplore_swingup_2026-09-06"


def load_replays(directory):
    """Validate a lesson-33 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Incompatible lesson-33 recording")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(report):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}

    phase1 = report["phase1"]
    # the archive grids must agree with the recorded occupancy
    occupied = int(np.sum(~np.isnan(data["grid_cell_error"])))
    if (
        occupied != phase1["cells_occupied"]
        or phase1["cells_total"] != data["grid_cell_visits"].size
    ):
        raise ValueError("Archive occupancy disagrees with the summary")
    if abs(occupied / data["grid_cell_visits"].size - phase1["coverage_final"]) > 1e-9:
        raise ValueError("Archive coverage disagrees with the summary")
    if (
        len(data["coverage_curve"]) != phase1["segments_run"]
        or len(data["steps_curve"]) != phase1["segments_run"]
        or len(data["selection_cells"]) != phase1["segments_run"]
    ):
        raise ValueError("Coverage curve length disagrees with the segment budget")

    # every kept capture must re-verify through the lesson-7 recovery_metrics
    reference = np.asarray(report["protocol"]["reference_state"], dtype=float)
    dt = float(report["protocol"]["dt_s"])
    for capture in phase1["per_capture"]:
        index = capture["index"]
        seg_states = data[f"capture{index}_seg_states"]
        seg_controls = data[f"capture{index}_seg_controls"]
        cap_start = int(capture["cap_start"])
        metrics = capture_slice_metrics(
            seg_states[cap_start:], seg_controls[cap_start:], reference, dt
        )
        if not metrics["recovered"]:
            raise ValueError(f"Capture {index} no longer passes the lesson-7 settled criterion")
        if abs(metrics["settled_at_s"] - capture["settled_at_s"]) > 1e-9:
            raise ValueError(f"Capture {index} settle time disagrees with the summary")
        if len(data[f"capture{index}_full_controls"]) != capture["teacher_steps"]:
            raise ValueError(f"Capture {index} teacher length disagrees with the summary")

    phase2 = report["phase2"]
    if phase2["run"]:
        curve = data["bc_loss_curve"]
        if curve.shape != (report["protocol"]["robustification"]["epochs"],):
            raise ValueError("BC loss curve length disagrees with the epoch count")
        if abs(float(curve[0]) - phase2["bc_loss_first"]) > 1e-9 * max(
            1.0, abs(phase2["bc_loss_first"])
        ):
            raise ValueError("BC curve start disagrees with the summary")
        if abs(float(curve[-1]) - phase2["bc_loss_last"]) > 1e-9 * max(
            1.0, abs(phase2["bc_loss_last"])
        ):
            raise ValueError("BC curve end disagrees with the summary")
        terminated = data["eval_terminated"]
        settled = data["eval_settled_s"]
        successes = int(np.sum((~terminated) & (~np.isnan(settled))))
        if successes != phase2["stochastic"]["successes"]:
            raise ValueError("BC successes disagree with the archive")
        arrivals = int(np.sum(~np.isnan(data["eval_first_arrival_s"])))
        if arrivals != phase2["arrival"]["episodes_with_arrival"]:
            raise ValueError("BC upright arrivals disagree with the archive")
    return {"report": report, **data}


class GoExploreDemo:
    """The archive map, the stable-band capture and the four-ways outcome."""

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
            text="第三十三课 · Go-Explore 画地图过崖：把“到过”变成“可重返”",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "阶段一：选格 → set_state 直接返回档案态 → 种子化随机探索（叠加第 7 课平衡 LQR）→ 重新入档。\n"
                "阶段二：把捕获轨迹跨格串成教师数据做 BC，再在【不做状态重置】的下方初态闭环评估。三个模式共用同一条正式记录"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="archive")
        for label, key in (
            ("① 档案热图：覆盖与选格轨迹", "archive"),
            ("② 稳定带捕获：捕获前后的状态轨迹", "capture"),
            ("③ 鲁棒化闭环评估 vs 四种过崖方式", "outcome"),
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
                "Go-Explore（Ecoffet 2021）：状态重置返回是仿真特权；格子/成员准则/捕获条件全部写入 protocol。"
                "成功率为有限样本计数"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_archive(self):
        """① occupancy heatmap, selection trace and the coverage curve."""
        self.fig.clear()
        report = self.report
        phase1 = report["phase1"]
        errors = self.data["grid_cell_error"]
        selection = self.data["selection_cells"]
        coverage = self.data["coverage_curve"]
        from matplotlib.colors import LogNorm

        angle_bins, cart_bins = errors.shape
        ax_map, ax_curve = self.fig.subplots(1, 2)
        shown = np.where(~np.isnan(errors), errors, np.nan)
        image = ax_map.imshow(
            shown,
            origin="lower",
            aspect="auto",
            cmap="viridis_r",
            norm=LogNorm(),  # the member error spans two orders of magnitude
            extent=[-2.4, 2.4, -180.0, 180.0],
        )
        self.fig.colorbar(image, ax=ax_map, label="成员 max|误差|/容差（对数轴，1 = 稳定带）")
        trace = selection[: min(600, len(selection))]
        alpha_centers = np.degrees(-np.pi + (trace[:, 0] + 0.5) * 2.0 * np.pi / angle_bins)
        cart_centers = -2.4 + (trace[:, 1] + 0.5) * 4.8 / cart_bins
        ax_map.plot(cart_centers, alpha_centers, "-", color="#f97316", linewidth=0.7, alpha=0.7)
        upright = np.degrees(0.3)
        ax_map.axhspan(-upright, upright, color="#ef4444", alpha=0.12)
        for capture in report["phase1"]["per_capture"][:8]:
            angle_bin, cart_bin = capture["cell"]
            cell_alpha = -180.0 + (angle_bin + 0.5) * 360.0 / angle_bins
            cell_x = -2.4 + (cart_bin + 0.5) * 4.8 / cart_bins
            ax_map.plot([cell_x], [cell_alpha], "r*", markersize=9)
        ax_map.set(
            xlabel="小车位置（m）",
            ylabel="杆角 α（度，0 = 直立）",
            title=f"档案热图：{phase1['cells_occupied']}/{phase1['cells_total']} 格（红星 = 稳定带格）",
        )
        ax_curve.plot(
            np.arange(1, len(coverage) + 1),
            coverage * 100.0 / (angle_bins * cart_bins),
            color="#2563eb",
        )
        first_cap = phase1["first_capture_step"]
        if first_cap is not None:
            cap_index = int(np.searchsorted(self.data["steps_curve"], first_cap))
            ax_curve.axvline(
                cap_index + 1,
                color="#16a34a",
                linestyle="--",
                linewidth=1.2,
                label=f"首次捕获（{first_cap / 1000:.0f}k 步）",
            )
            ax_curve.legend(fontsize=7, loc="lower right")
        ax_curve.set(
            xlabel="探索段编号",
            ylabel="档案覆盖率（%）",
            title="覆盖曲线（格子数 / 总格数）",
            ylim=(0, 105),
        )
        for ax in (ax_map, ax_curve):
            ax.grid(alpha=0.2)

    def draw_capture(self):
        """② the first kept capture: states around the settled-tail moment."""
        self.fig.clear()
        report = self.report
        if not report["phase1"]["per_capture"]:
            ax = self.fig.subplots(1, 1)
            ax.text(
                0.5,
                0.5,
                "阶段一未捕获稳定带：无捕获轨迹可展示",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set(title="稳定带捕获：无")
            return
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        capture = report["phase1"]["per_capture"][0]
        full_states = self.data["capture0_full_states"]
        full_controls = self.data["capture0_full_controls"]
        seg_controls = self.data["capture0_seg_controls"]
        cap_start = int(capture["cap_start"])
        # the stitched path ends with this segment: the archive reset sits where
        # the parent chain hands over to the segment (full_len - seg_len + 1)
        archive_index = len(full_states) - len(seg_controls) - 1
        times = np.arange(len(full_states)) * dt
        tail_start = archive_index + cap_start
        ax_pole, ax_cart, ax_omega, ax_force = self.fig.subplots(2, 2).reshape(-1)
        for ax in (ax_pole, ax_cart, ax_omega, ax_force):
            ax.axvspan(times[tail_start], times[-1], color="#16a34a", alpha=0.10)
            ax.axvline(times[archive_index], color="#7c3aed", linestyle=":", linewidth=1.2)
            ax.axvline(times[tail_start], color="#16a34a", linestyle="--", linewidth=1.2)
        ax_pole.plot(times, np.cos(full_states[:, 1] - ref_theta), color="#2563eb")
        ax_pole.axhspan(-1, 0, alpha=0.06, color="orange")
        ax_pole.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title=f"第一次捕获（段 {capture['segment']}）：绿区 = ≥2 s 稳定尾段，紫点线 = 档案重置",
        )
        for bound in (-2.4, 2.4):
            ax_cart.axhline(bound, color="red", linestyle=":", linewidth=0.8)
        ax_cart.plot(times, full_states[:, 0], color="#0f766e")
        ax_cart.set(
            ylabel="小车位置（m）",
            xlabel="仿真时间（s）",
            title="小车位置（红点线 = ±2.4 m 边界）",
        )
        ax_omega.plot(times, full_states[:, 3], color="#7c3aed")
        for bound in (-2.0, 2.0):
            ax_omega.axhline(bound, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax_omega.set(
            ylabel="杆角速度（rad/s）",
            xlabel="仿真时间（s）",
            title="角速度（虚线 = 第 7 课抓取阈值 ±2）",
        )
        edges = np.arange(len(full_controls) + 1) * dt
        ax_force.stairs(full_controls * report["protocol"]["actuator_gear"], edges, color="#b45309")
        ax_force.set(
            ylabel="电机力（N）",
            xlabel="仿真时间（s）",
            title="电机输入：随机泵能段 + LQR 抓取/保持段",
        )
        for ax in (ax_pole, ax_cart, ax_omega, ax_force):
            ax.grid(alpha=0.2)

    def draw_outcome(self):
        """③ four-way bars, BC closed-loop vs baseline, BC loss, failure case."""
        self.fig.clear()
        report = self.report
        dt = report["protocol"]["dt_s"]
        ref_theta = report["protocol"]["reference_state"][1]
        rows = report["four_way_comparison"]
        labels = ["基线\n(第7课)", "纯PPO\n(29课)", "DAPG\n(32课)", "GoExplore\n+BC(本课)"]
        colors = ("#64748b", "#b91c1c", "#7c3aed", "#2563eb")
        ax_rate, ax_loop, ax_bc, ax_case = self.fig.subplots(2, 2).reshape(-1)
        successes = [row["successes"] for row in rows]
        totals = [row["episodes"] for row in rows]
        bars = ax_rate.bar(
            labels,
            [s / t * 100 if t else 0.0 for s, t in zip(successes, totals, strict=True)],
            color=colors,
            width=0.62,
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
            title="四种过崖方式：下方初态验收（第 7 课口径）",
        )
        ax_rate.tick_params(axis="x", labelsize=7.5)

        baseline = self.data["baseline_states"]
        ax_loop.plot(
            np.arange(len(baseline)) * dt,
            np.cos(baseline[:, 1] - ref_theta),
            "--",
            color="#64748b",
            label="第 7 课基线",
        )
        if report["phase2"]["run"]:
            det_states = self.data["det_states"]
            det = report["phase2"]["deterministic"]
            det_text = (
                f"首达 {det['first_arrival_s']:.2f} s"
                if det["first_arrival_s"] is not None
                else "未进入直立区"
            )
            ax_loop.plot(
                np.arange(len(det_states)) * dt,
                np.cos(det_states[:, 1] - ref_theta),
                color="#2563eb",
                label=f"BC 策略（均值动作，{det_text}）",
            )
            ax_loop.legend(fontsize=7, loc="lower right")
        ax_loop.axhspan(-1, 0, alpha=0.06, color="orange")
        ax_loop.set(
            ylabel="杆端相对高度",
            ylim=(-1.1, 1.1),
            xlabel="仿真时间（s）",
            title="鲁棒化策略闭环：同一下方初态（无状态重置）",
        )

        if report["phase2"]["run"]:
            bc_curve = self.data["bc_loss_curve"]
            ax_bc.plot(np.arange(1, len(bc_curve) + 1), np.maximum(bc_curve, 1e-8), color="#2563eb")
            ax_bc.set(
                xlabel="BC 轮次",
                ylabel="教师 MSE（对数轴）",
                yscale="log",
                title=f"BC 损失：{bc_curve[0]:.3f} → {bc_curve[-1]:.3f}",
            )
        else:
            ax_bc.text(
                0.5, 0.5, "阶段二未运行", ha="center", va="center", transform=ax_bc.transAxes
            )
            ax_bc.set(title="BC 损失：无")

        case = report["failure_analysis"]["featured_case"]
        if case is not None:
            case_states = self.data["case0_states"]
            ax_case.plot(
                np.arange(len(case_states)) * dt,
                np.cos(case_states[:, 1] - ref_theta),
                color="#b91c1c",
            )
            ax_case.axhspan(-1, 0, alpha=0.06, color="orange")
            ax_case.set(
                ylabel="杆端相对高度",
                xlabel="仿真时间（s）",
                title=f"失败案例（评估种子 {case['eval_seed']}）：{failure_label_cn(case['failure_reason'])}",
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
            ax_case.set(title="失败案例：无")
        for ax in (ax_rate, ax_loop, ax_bc, ax_case):
            ax.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        phase1 = report["phase1"]
        phase2 = report["phase2"]
        if mode == "archive":
            lines = ["① 这一幕在看：智能体给自己画的地图"]
            lines.append(
                f"  覆盖 {phase1['cells_occupied']}/{phase1['cells_total']} 格"
                f"（{phase1['coverage_final'] * 100:.0f}%）"
            )
            first_up = phase1["first_upright_step"]
            first_cap = phase1["first_capture_step"]
            lines.append(
                f"  首次入档直立带：{'从未' if first_up is None else f'{first_up / 1000:.1f}k 步'}"
            )
            lines.append(
                f"  首次捕获稳定带：{'从未' if first_cap is None else f'{first_cap / 1000:.1f}k 步'}"
            )
            lines.append(f"  捕获总数 {phase1['capture_count']}，保留 {phase1['kept_captures']} 条")
            lines.append(f"  探索 {phase1['env_steps']} 步，墙钟 {phase1['wall_time_s']:.1f} s")
            lines.append(f"  LQR 保持层步数占比 {phase1['lqr_step_fraction'] * 100:.0f}%")
            lines.append("  红星格：从该格状态起连续 ≥2 s 稳定尾段")
        elif mode == "capture":
            lines = ["② 这一幕在看：山顶第一次被人住满 2 秒"]
            if not phase1["per_capture"]:
                lines.append("  阶段一未捕获稳定带")
            else:
                capture = phase1["per_capture"][0]
                lines.append(f"  捕获段 #{capture['segment']}（{capture['env_steps']} 步时）")
                lines.append(f"  起始格（杆角 bin, 车位 bin）= {tuple(capture['cell'])}")
                lines.append(
                    f"  稳定尾段 {capture['tail_s']:.2f} s（判据 ≥{CAPTURE_TAIL_S:.0f} s）"
                )
                lines.append(f"  教师全长 {capture['teacher_steps']} 步（下方初态→稳定尾段）")
                lines.append("  第 7 课 recovery_metrics 复核：通过")
                lines.append(f"  全记录捕获 {phase1['capture_count']} 次")
        else:
            lines = ["③ 这一幕在裁决：地图能否变成路"]
            for row in report["four_way_comparison"]:
                lines.append(f"  {row['label']}：{row['successes']}/{row['episodes']}")
            lines.append(
                f"  教师：{phase2['teacher_trajectories']} 条轨迹 "
                f"{phase2['teacher_pairs']} 对 (s,a)"
            )
            if phase2["run"]:
                lines.append(
                    f"  BC 损失 {phase2['bc_loss_first']:.2f}→{phase2['bc_loss_last']:.2f}"
                )
                lines.append(
                    f"  闭环评估 {phase2['stochastic']['successes']}"
                    f"/{phase2['stochastic']['episodes']}"
                )
                lines.append(
                    f"  直立首达 {phase2['arrival']['episodes_with_arrival']}"
                    f"/{phase2['arrival']['episodes']}"
                )
                counts = phase2["stochastic"]["failure_counts"]
                lines.append(
                    f"  失败：出界 {counts['cart_safety_boundary']}，"
                    f"超时未稳 {counts['timeout_without_settling']}"
                )
            else:
                lines.append("  阶段二未运行（阶段一无教师数据）")
            zero_to_one = report["hypothesis"]["phase2_zero_to_one"]
            lines.append(f"  0→1（历史性首次成功）：{'是' if zero_to_one else '否'}")
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "archive":
            self.draw_archive()
        elif mode == "capture":
            self.draw_capture()
        else:
            self.draw_outcome()
        self.fill_stats(mode)
        status = {
            "archive": "① 这一幕在看：杆角×车位的档案覆盖、选格轨迹与覆盖曲线（红星 = 稳定带格）",
            "capture": "② 这一幕在看：第一次捕获的完整段轨迹，绿区为连续 ≥2 s 的稳定尾段",
            "outcome": "③ 这一幕在裁决：四种过崖方式同口径对照，BC 闭环评估与失败案例",
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
    root.title("第三十三课 · Go-Explore 画地图过崖：把“到过”变成“可重返”")
    root.geometry("1600x820")
    root.minsize(1380, 700)
    GoExploreDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
