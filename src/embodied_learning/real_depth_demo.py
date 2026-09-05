"""Tk teaching demo for lesson 24: real Depth-Anything affine check (static panels).

Three modes over one formal recording (``results/real_depth_affine_2026-09-05``):

1. ``scales``    -- calibrated metric error map vs the lesson-23 ideal proxy (≈0).
2. ``structure`` -- residual map and the U-shaped residual-vs-1/Z curve.
3. ``scan``      -- control-point sweep (N) with 1-sigma band and near/far split.

The loader re-validates the summary contract, the npz digest, and cross-checks
the dense fit against the archive before anything is drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPERIMENT = "real_depth_affine_check"
DEFAULT_RESULTS = Path("results/real_depth_affine_2026-09-05")
N_VALUES = [2, 5, 10, 20]

EXPECTED_NPZ_KEYS = {
    "bin_centers",
    "bin_means",
    "bin_sizes",
    "bin_stds",
    "dense_error_m",
    "dense_fit_ab",
    "n_values",
    "proxy_fit_failures",
    "proxy_group_max_abs_error_m",
    "proxy_group_mean_abs_error_m",
    "proxy_reference_error_m",
    "proxy_reference_fit_ab",
    "real_fit_failures",
    "real_group_far_mean_abs_error_m",
    "real_group_max_abs_error_m",
    "real_group_mean_abs_error_m",
    "real_group_mean_signed_error_m",
    "real_group_median_abs_error_m",
    "real_group_near_mean_abs_error_m",
    "real_group_r2_ctrl",
    "real_group_std_abs_error_m",
    "real_reference_ctrl_depth_m",
    "real_reference_ctrl_pixels",
    "real_reference_ctrl_r",
    "real_reference_error_m",
    "real_reference_fit_ab",
    "real_reference_z_hat_m",
    "residual_map",
    "runs_per_group",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_analysis(directory):
    """Validate a lesson-24 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("canvas_px") != [640, 480]
        or list(report.get("n_values", ())) != N_VALUES
        or report.get("min_inv_depth_span") != 0.1
        or not report.get("model")
    ):
        raise ValueError("Incompatible lesson-24 recording")
    path = directory / "analysis.npz"
    if _digest(path) != report.get("analysis_npz_sha256"):
        raise ValueError("Analysis checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    if list(data["n_values"]) != list(N_VALUES):
        raise ValueError("Archive N scan disagrees with the summary")
    if not np.allclose(
        data["dense_fit_ab"], [report["dense_fit"]["a"], report["dense_fit"]["b"]], atol=1e-9
    ):
        raise ValueError("Archive dense fit disagrees with the summary")
    if not np.isclose(
        float(np.asarray(data["runs_per_group"])), report["runs_per_group"], rtol=0.0, atol=1e-12
    ):
        raise ValueError("Archive runs disagree with the summary")
    return {"report": report, **data}


class RealDepthDemo:
    """Affine-check narrative: error maps, residual structure, N scan."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and labels
        self.root = root
        self.data = data
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第二十四课 · 真实 DA V2 的仿射检验：标定后的米制误差与残差结构",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Depth Anything V2 输出 r_pred 经第 23 课控制点协议做逆深度仿射标定。\n"
                "三个模式共用同一条正式实验记录：① 误差图对照 ② 残差结构 ③ N 扫描；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="scales")
        for label, key in (
            ("① 标定后误差图 vs 理想代理", "scales"),
            ("② 残差结构（U 形 + 空间分布）", "structure"),
            ("③ 控制点数量 N 扫描", "scan"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        self.stats = ttk.Label(outer, text="", justify="left", font=("Microsoft YaHei", 10))
        self.stats.pack(anchor="w", pady=(4, 4))
        self.fig = Figure(figsize=(11.5, 4.6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=outer)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Button(outer, text="退出 (Esc)", command=self.close).pack(anchor="e", pady=(6, 0))
        root.bind("<Escape>", lambda _event: self.close())
        self.redraw()

    # -- mode panels -----------------------------------------------------
    def _panel_scales(self):
        report = self.data["report"]
        ax = self.fig.add_subplot(1, 2, 1)
        err_cm = self.data["real_reference_error_m"] * 100.0
        im = ax.imshow(err_cm, cmap="magma", origin="upper")
        ax.set_title("真实 DA V2：标定后全图误差 (cm)", fontsize=10)
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax2 = self.fig.add_subplot(1, 2, 2)
        err_cm_proxy = self.data["proxy_reference_error_m"] * 100.0
        im2 = ax2.imshow(err_cm_proxy, cmap="magma", origin="upper", vmin=0.0)
        ax2.set_title("第 23 课理想代理：同口径误差 (cm)", fontsize=10)
        self.fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        fit = report["dense_fit"]
        self.stats.config(
            text=(
                f"全图拟合 r=a·(1/Z)+b：a={fit['a']:.3f}, b={fit['b']:.3f}, R²={fit['r_squared']:.4f}；"
                f"残差 std={fit['residual_std']:.4f}（r std 的 {100 * fit['residual_std_over_r_std']:.1f}%）\n"
                f"右图理想代理误差 ~1e-14 cm：真实模型的 3 cm 量级误差全部来自非仿射残差，"
                f"而不是标定协议（本幕在对照：标定协议上界）"
            )
        )

    def _panel_structure(self):
        report = self.data["report"]
        ax = self.fig.add_subplot(1, 2, 1)
        res = self.data["residual_map"]
        vmax = np.nanpercentile(np.abs(res[res != 0]), 99) if np.any(res != 0) else 1.0
        im = ax.imshow(res, cmap="RdBu_r", origin="upper", vmin=-vmax, vmax=vmax)
        ax.set_title("仿射残差空间分布（99 分位截断）", fontsize=10)
        self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax2 = self.fig.add_subplot(1, 2, 2)
        ax2.errorbar(
            self.data["bin_centers"],
            self.data["bin_means"],
            yerr=self.data["bin_stds"],
            fmt="o-",
            color="tab:blue",
            capsize=3,
        )
        ax2.axhline(0.0, color="k", lw=0.8, ls="--")
        ax2.set_xlabel("1/Z（分箱中心）")
        ax2.set_ylabel("残差均值 ±1σ")
        ax2.set_title("残差 vs 1/Z：U 形结构（非白噪声）", fontsize=10)
        valid = self.data["dense_error_m"] > 0
        corr = float("nan")
        if valid.sum() > 2:
            u = np.tile(np.arange(res.shape[1], dtype=float), (res.shape[0], 1))
            corr = float(np.corrcoef(res[valid], u[valid])[0, 1])
        self.stats.config(
            text=(
                f"残差不是白噪声：1/Z 方向 U 形（分箱均值从 + 到 − 再回正），"
                f"与像素横坐标相关 r={corr:+.3f}（左正右负）\n"
                f"机制：残差 std 占 r std 的 {100 * report['dense_fit']['residual_std_over_r_std']:.1f}%——"
                f"仿射是一阶近似，亚厘米精度需要分段/逐区域标定"
            )
        )

    def _panel_scan(self):
        ax = self.fig.add_subplot(1, 2, 1)
        n = self.data["n_values"]
        mean = self.data["real_group_mean_abs_error_m"] * 100.0
        std = self.data["real_group_std_abs_error_m"] * 100.0
        ax.errorbar(
            n,
            mean,
            yerr=std,
            fmt="o-",
            color="tab:orange",
            capsize=3,
            label="真实 DA V2（均值±1σ）",
        )
        proxy = self.data["proxy_group_mean_abs_error_m"] * 100.0
        ax.plot(
            n, np.maximum(proxy, 1e-15), "s--", color="tab:green", label="第 23 课理想代理（≈0）"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(n))
        ax.set_xticklabels([str(v) for v in n])
        ax.set_xlabel("控制点数量 N")
        ax.set_ylabel("全图平均误差 (cm)")
        ax.set_title("N 扫描：N=10 后饱和在 ~3.2 cm", fontsize=10)
        ax.legend(fontsize=8)
        ax2 = self.fig.add_subplot(1, 2, 2)
        width = 0.35
        xs = np.arange(len(n))
        near = self.data["real_group_near_mean_abs_error_m"] * 100.0
        far = self.data["real_group_far_mean_abs_error_m"] * 100.0
        ax2.bar(xs - width / 2, near, width, color="tab:blue", label="近段（15% 分位内）")
        ax2.bar(xs + width / 2, far, width, color="tab:red", label="远段（85% 分位外）")
        ax2.set_xticks(xs)
        ax2.set_xticklabels([str(v) for v in n])
        ax2.set_xlabel("控制点数量 N")
        ax2.set_ylabel("误差均值 (cm)")
        ax2.set_title("远/近分层：δZ≈Z²·δ(1/Z) 的几何放大", fontsize=10)
        ax2.legend(fontsize=8)
        failures = int(self.data["real_fit_failures"].sum())
        self.stats.config(
            text=(
                f"每组 20 种子：N=2→20 全图均值 4.66→3.20 cm，N=10 后收益饱和——"
                f"3 cm 是全局仿射标定的下限（残差结构决定，非控制点数量）\n"
                f"远/近误差比 3.7–5.9 倍（σ=3% 口径见讲义表）；拟合失效 {failures}/160；"
                f"本幕在对照：理想代理各 N 均 ≈1e-14 cm，N 只压噪声、不压结构"
            )
        )

    # -- plumbing --------------------------------------------------------
    def redraw(self):
        self.fig.clear()
        getattr(self, f"_panel_{self.mode.get()}")()
        self.fig.tight_layout()
        self.canvas.draw()

    def close(self):
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    import tkinter as tk

    data = load_analysis(args.results)
    root = tk.Tk()
    root.title("第二十四课 · 真实单目相对深度的仿射检验")
    root.geometry("1240x760")
    demo = RealDepthDemo(root, data)
    root.mainloop()
    del demo


if __name__ == "__main__":
    main()
