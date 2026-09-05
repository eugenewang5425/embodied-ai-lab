"""Lesson 23 viewer: relative depth vs metric depth, calibration, error anatomy.

Three static modes share one lesson-23 recording:
1. relative map vs metric map vs the naive "1/r" misreading (no scale);
2. control points on the map + the inverse-depth affine fit;
3. the calibrated error map + the N/sigma comparison curves.
Layout, Esc handling and the meaning panel follow the lesson-22 demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.monocular_metric import (
    A_TRUE,
    B_TRUE,
    EXPERIMENT,
    K_INTRINSIC,
    MIN_INV_DEPTH_SPAN,
    N_VALUES,
    NAIVE_A,
    NAIVE_B,
    REFERENCE_N,
    REFERENCE_SIGMA,
    SIGMA_VALUES,
    relative_from_depth,
    render_depth_map,
)

DEFAULT_RESULTS = "results/mobile_monocular_2026-09-05"
EXPECTED_NPZ_KEYS = {
    "depth_map_m",
    "valid_mask",
    "pole_mask",
    "relative_map",
    "near_mask",
    "far_mask",
    "n_values",
    "sigma_values",
    "runs_per_group",
    "group_mean_abs_error_m",
    "group_median_abs_error_m",
    "group_std_abs_error_m",
    "group_max_abs_error_m",
    "group_near_mean_abs_error_m",
    "group_far_mean_abs_error_m",
    "group_mean_signed_error_m",
    "fit_failures",
    "mechanism_ratio_median",
    "mechanism_var_a_ratio",
    "mechanism_var_b_ratio",
    "mechanism_empirical_std_m",
    "mechanism_predicted_std_m",
    "reference_z_hat_m",
    "reference_error_m",
    "reference_fit_ab",
    "reference_ctrl_pixels",
    "reference_ctrl_depth_m",
    "reference_ctrl_relative",
    "reference_ctrl_relative_noisy",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-23 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("canvas_px") != [640, 480]
        or report.get("a_true") != A_TRUE
        or report.get("b_true") != B_TRUE
        or report.get("naive_assumed_a") != NAIVE_A
        or report.get("naive_assumed_b") != NAIVE_B
        or report.get("n_values") != list(N_VALUES)
        or report.get("sigma_values_relative") != list(SIGMA_VALUES)
        or not np.array_equal(np.array(report.get("intrinsic")), K_INTRINSIC)
    ):
        raise ValueError("Incompatible lesson-23 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    depth = data["depth_map_m"]
    valid = data["valid_mask"]
    # Re-derive the whole scene from the lesson-22 camera and surfaces.
    fresh = render_depth_map()
    if not np.array_equal(fresh["valid"], valid):
        raise ValueError("Stored validity mask does not match the scene/camera")
    if not np.allclose(fresh["depth_m"], depth, atol=1e-12, equal_nan=True):
        raise ValueError("Stored depth map does not match the scene/camera")
    if not np.allclose(
        relative_from_depth(depth, valid), data["relative_map"], atol=1e-12, equal_nan=True
    ):
        raise ValueError("Stored relative map is not the r = a/Z + b proxy")
    if np.allclose(data["reference_z_hat_m"], depth, atol=1e-9, equal_nan=True):
        raise ValueError("Stored calibrated map equals truth (bug guard)")
    return {"report": report, **data}


class MonocularMetricDemo:
    """Relative-vs-metric depth narrative: maps, fit, error anatomy."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and colorbars
        self.root = root
        self.data = data
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第二十三课 · 单目相对深度 → 米制尺度标定",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "场景与相机沿用第二十二课（同一地面、同一竖直杆、同一 K 与 look-at）；"
                "相对深度 r = a·(1/Z)+b 的 a、b 未知且任意\n"
                "三个模式共用同一条正式实验记录：① 无尺度对照 ② 控制点标定 ③ 误差与 N/σ 对照；"
                "全部为静态图，无动画"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="maps")
        for label, key in (
            ("① 相对深度 vs 米制深度（无尺度）", "maps"),
            ("② 控制点与仿射拟合", "fit"),
            ("③ 误差图与 N/σ 对照", "error"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(middle)
        left.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.2, 4.3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=52, anchor="nw", justify="left")
        self.stats.pack(side="left", fill="y", padx=(12, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "代理：r = a·(1/Z) + b（a、b 任意，即“无尺度”）｜标定：控制点最小二乘 r ≈ a·(1/Z)+b，"
                "全图 Z_hat = a/(r−b)｜传播：δZ ≈ Z²·|δ(1/Z)|，远处被平方放大"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def _maps_axes(self):
        self.fig.clear()
        return self.fig.subplots(1, 3)

    def _pair_axes(self):
        self.fig.clear()
        return self.fig.subplots(1, 2)

    def draw_maps(self):
        valid = self.data["valid_mask"]
        relative = self.data["relative_map"]
        depth = self.data["depth_map_m"]
        naive = np.full(depth.shape, np.nan)
        naive[valid] = 1.0 / relative[valid]
        ax_r, ax_z, ax_n = self._maps_axes()
        im = ax_r.imshow(relative, cmap="viridis")
        self.fig.colorbar(im, ax=ax_r, shrink=0.85, label="r（任意单位）")
        ax_r.set(title="相对深度 r：只有深浅，没有米", xlabel="u / px", ylabel="v / px")
        im = ax_z.imshow(depth, cmap="magma")
        self.fig.colorbar(im, ax=ax_z, shrink=0.85, label="Z / m")
        ax_z.set(title="米制深度 Z（逐像素真值）", xlabel="u / px", ylabel="v / px")
        im = ax_n.imshow(naive, cmap="magma", vmin=0.0)
        self.fig.colorbar(im, ax=ax_n, shrink=0.85, label="1/r 误读成的“深度” / m")
        ax_n.set(title="无标定：假设 a=1, b=0", xlabel="u / px", ylabel="v / px")

    def draw_fit(self):
        depth = self.data["depth_map_m"]
        pixels = self.data["reference_ctrl_pixels"]
        ax_map, ax_fit = self._pair_axes()
        im = ax_map.imshow(depth, cmap="magma")
        self.fig.colorbar(im, ax=ax_map, shrink=0.85, label="Z / m")
        ax_map.scatter(
            pixels[:, 0], pixels[:, 1], marker="x", s=90, c="#22d3ee", lw=2.0, label="控制点"
        )
        ax_map.legend(loc="lower right", fontsize=8)
        ax_map.set(
            title=f"控制点分布（N={len(pixels)}，须覆盖深度范围）", xlabel="u / px", ylabel="v / px"
        )
        ctrl_u = 1.0 / self.data["reference_ctrl_depth_m"]
        u_line = np.linspace(0.9 * ctrl_u.min(), 1.05 * ctrl_u.max(), 60)
        a_fit, b_fit = self.data["reference_fit_ab"]
        ax_fit.plot(
            u_line,
            A_TRUE * u_line + B_TRUE,
            "--",
            color="#0f172a",
            lw=1.5,
            label=f"真值 a={A_TRUE}, b={B_TRUE}（未知）",
        )
        ax_fit.plot(
            u_line,
            a_fit * u_line + b_fit,
            "-",
            color="#2563eb",
            lw=1.7,
            label=f"拟合 a={a_fit:.3f}, b={b_fit:.3f}",
        )
        ax_fit.scatter(
            ctrl_u,
            self.data["reference_ctrl_relative_noisy"],
            marker="x",
            s=70,
            c="#dc2626",
            lw=1.8,
            label=f"控制点读数（σ={REFERENCE_SIGMA:.0%}）",
        )
        ax_fit.set(xlabel="1/Z / (1/m)", ylabel="r（任意单位）", title="逆深度域的仿射拟合")
        ax_fit.legend(fontsize=8)

    def draw_error(self):
        report = self.data["report"]
        error = np.abs(self.data["reference_error_m"])
        pixels = self.data["reference_ctrl_pixels"]
        ax_map, ax_curve = self._pair_axes()
        im = ax_map.imshow(error, cmap="inferno")
        self.fig.colorbar(im, ax=ax_map, shrink=0.85, label="|Z_hat − Z| / m")
        ax_map.scatter(
            pixels[:, 0], pixels[:, 1], marker="x", s=90, c="#22d3ee", lw=2.0, label="控制点"
        )
        ax_map.legend(loc="lower right", fontsize=8)
        ax_map.set(
            title=f"标定后误差（N={REFERENCE_N}, σ={REFERENCE_SIGMA:.0%}，单种子）",
            xlabel="u / px",
            ylabel="v / px",
        )
        n_values = self.data["n_values"]
        mean_err = self.data["group_mean_abs_error_m"] * 100.0
        for sigma_index, sigma in enumerate(SIGMA_VALUES):
            ax_curve.plot(
                n_values,
                mean_err[:, sigma_index],
                "-o",
                ms=4,
                label=f"σ={sigma:.0%}" if sigma else "σ=0（数值精确）",
            )
        ax_curve.plot(
            n_values,
            self.data["group_near_mean_abs_error_m"][:, -1] * 100.0,
            "--",
            color="#9333ea",
            lw=1.2,
            label="σ=3% 近段（≤15% 分位）",
        )
        ax_curve.plot(
            n_values,
            self.data["group_far_mean_abs_error_m"][:, -1] * 100.0,
            "--",
            color="#ea580c",
            lw=1.2,
            label="σ=3% 远段（≥85% 分位）",
        )
        ax_curve.axhline(
            report["naive_mean_abs_error_m"] * 100.0,
            color="#0f172a",
            ls=":",
            lw=1.2,
            label="无标定（a=1, b=0）",
        )
        ax_curve.set_yscale("log")
        ax_curve.set(
            xlabel="控制点数量 N",
            ylabel="全图平均 |Z_hat − Z| / cm",
            title=f"N–误差曲线（{report['runs_per_group']} 种子均值，对数轴）",
        )
        ax_curve.legend(fontsize=8, loc="center left")

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.data["report"]
        valid = self.data["valid_mask"]
        relative = self.data["relative_map"]
        depth = self.data["depth_map_m"]
        if mode == "maps":
            naive_err = np.abs(1.0 / relative[valid] - depth[valid])
            text = (
                "① 这一幕在说明：单目图像没有米制尺度\n\n"
                f"  相对深度 r 范围 {relative[valid].min():.2f}–{relative[valid].max():.2f}"
                "（单位任意）\n"
                f"  米制深度 Z 范围 {depth[valid].min():.2f}–{depth[valid].max():.2f} m\n"
                "  同样的深浅次序，配上不同的 a、b\n"
                "  就是完全不同的米制世界。\n\n"
                "  无标定对照（把 r 直接当 1/Z 用，a=1, b=0）：\n"
                f"  全图平均误差 {naive_err.mean() * 100:.0f} cm，"
                f"最大 {naive_err.max() * 100:.0f} cm\n"
                f"  完美先验（真值 a、b）：最大误差 "
                f"{report['perfect_prior_max_abs_error_m']:.1e} m\n\n"
                "  意义：尺度必须由外部信息补上——\n"
                "  这正是第二十二课“单目只有射线”的图像版。"
            )
        elif mode == "fit":
            a_fit, b_fit = self.data["reference_fit_ab"]
            text = (
                "② 这一幕在标定：用少量已知深度的点拉回米制\n\n"
                f"  控制点 {REFERENCE_N} 个（模拟稀疏测距/GIS 控制点），\n"
                f"  读数噪声 σ={REFERENCE_SIGMA:.0%}（乘性，加在 r 上）\n"
                "  最小二乘：r ≈ a·(1/Z)+b → Z_hat = a/(r−b)\n\n"
                f"  拟合 a={a_fit:.3f}, b={b_fit:.3f}\n"
                f"  真值 a={A_TRUE:.1f}, b={B_TRUE:.1f}（标定不知道）\n\n"
                f"  约束：控制点的 1/Z 跨度 ≥ {MIN_INV_DEPTH_SPAN}，\n"
                "  两点几乎同深会让拟合病态（斜率可被噪声翻转）。\n"
                "  理想仿射代理下 σ=0 时 2 个控制点即精确复原\n"
                f"  （实验中 σ=0 各组误差 ~1e-13 cm）。"
            )
        else:
            ref = report["reference_group"]
            ratios = np.array(report["mechanism_ratio_median"])
            far_near = np.array(report["group_far_over_near"])[2, 1]
            text = (
                "③ 这一幕在量化：误差在哪、多大、为什么\n\n"
                f"  参考组 N={ref['n']}, σ={ref['sigma']:.0%}（{report['runs_per_group']} 种子均值）：\n"
                f"  全图平均 {ref['mean_abs_error_m'] * 100:.2f} cm；\n"
                f"  近段 {ref['near_mean_abs_error_m'] * 100:.2f} / "
                f"远段 {ref['far_mean_abs_error_m'] * 100:.2f} cm"
                f"（远/近 ≈ {far_near:.1f}）\n\n"
                "  机制：δZ ≈ Z²·|δ(1/Z)|，b 的误差在远处被\n"
                "  Z² 放大——远处更差不是偶然。\n"
                "  公式核对（拟合参数协方差 vs 传播公式）：\n"
                f"  8 组中位数比值 {np.nanmin(ratios):.2f}–{np.nanmax(ratios):.2f}"
                f"（{report['mechanism_realizations']} 次实现）\n\n"
                "  提醒：σ=0 时误差并非零而是 ~1e-13 cm，\n"
                "  说明本课误差全部来自读数噪声的传播。"
            )
        self.stats.configure(text=text)

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "maps":
            self.draw_maps()
        elif mode == "fit":
            self.draw_fit()
        else:
            self.draw_error()
        self.fill_stats(mode)
        status = {
            "maps": "① 这一幕在证明：相对深度的“深浅”不含米——必须由标定补上尺度",
            "fit": "② 这一幕在标定：少数已知深度的控制点把 r 拉回米制，拟合在逆深度域进行",
            "error": "③ 这一幕在量化：标定误差随深度平方放大、随控制点数与噪声变化，公式比值≈1",
        }[mode]
        self.status.configure(text=status + "｜静态图，无动画。按 Esc 退出。")
        self.canvas.draw_idle()


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.geometry("1600x760")
    root.minsize(1380, 660)
    MonocularMetricDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
