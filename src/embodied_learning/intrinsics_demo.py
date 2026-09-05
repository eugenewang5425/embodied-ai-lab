"""Lesson 25 viewer: chessboard poses, reprojection truth, M-convergence.

Three static modes share one lesson-25 recording:
1. the board plane with the camera pose pool and one view's corner
   correspondences;
2. homography-driven reprojection with the naive guessed K vs the estimated K
   (known poses vs decomposed poses - the pose-absorption pitfall);
3. the f/cx/cy error convergence in M with the degenerate pose-set record.
Layout, Esc handling and the meaning panel follow the lesson-22/23 demos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.camera_intrinsics import (
    BOARD_COLS,
    BOARD_ROWS,
    EXPERIMENT,
    FOCAL_PX,
    M_VALUES,
    NAIVE_K,
    REFERENCE_M,
    REFERENCE_SIGMA_PX,
    SIGMA_VALUES_PX,
    SQUARE_M,
    board_corners,
    decompose_extrinsics,
    intrinsic_from_homographies,
    intrinsic_matrix,
    project_points,
)
from embodied_learning.experiments.pinhole_projection import (
    HEIGHT_PX,
    K_INTRINSIC,
    WIDTH_PX,
)

DEFAULT_RESULTS = "results/camera_intrinsics_2026-09-05"
EXPECTED_NPZ_KEYS = {
    "board_corners_m",
    "pool_eye_m",
    "pool_target_m",
    "m_values",
    "sigma_values_px",
    "runs_per_group",
    "k_est_f_mean_px",
    "k_est_f_std_px",
    "k_est_f_err_mean_px",
    "k_est_f_err_median_px",
    "k_est_cx_mean_px",
    "k_est_cx_std_px",
    "k_est_cx_err_mean_px",
    "k_est_cy_mean_px",
    "k_est_cy_std_px",
    "k_est_cy_err_mean_px",
    "reproj_rms_est_px",
    "reproj_rms_naive_px",
    "reproj_rms_true_px",
    "roundtrip_est_m",
    "roundtrip_naive_m",
    "roundtrip_true_m",
    "fit_failures",
    "resample_attempts",
    "mechanism_c_px",
    "ref_pose_index",
    "ref_rotation",
    "ref_translation",
    "ref_pixels_clean",
    "ref_pixels_noisy",
    "ref_homographies",
    "ref_k_est",
    "ref_normal_spread_deg",
    "probe_point_m",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-25 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("canvas_px") != [640, 480]
        or report.get("f_true_px") != FOCAL_PX
        or not np.array_equal(np.array(report.get("k_true")), K_INTRINSIC)
        or not np.array_equal(np.array(report.get("naive_k")), NAIVE_K)
        or report.get("m_values") != list(M_VALUES)
        or report.get("sigma_values_px") != list(SIGMA_VALUES_PX)
        or report.get("board", {}).get("inner_corners") != [BOARD_COLS, BOARD_ROWS]
        or report.get("board", {}).get("square_m") != SQUARE_M
        or report.get("reference_group", {}).get("m") != REFERENCE_M
        or report.get("reference_group", {}).get("sigma_px") != REFERENCE_SIGMA_PX
    ):
        raise ValueError("Incompatible lesson-25 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # Re-derive the scene from the lesson-22 camera constants and compare.
    corners = board_corners()
    if not np.allclose(corners, data["board_corners_m"], atol=1e-12):
        raise ValueError("Stored board corners do not match the constants")
    projected = project_points(
        corners, data["ref_rotation"][0], data["ref_translation"][0], K_INTRINSIC
    )
    if not np.allclose(projected, data["ref_pixels_clean"][0], atol=1e-9):
        raise ValueError("Stored reference pixels do not match the board/camera")
    reference = report["reference_group"]
    if abs(reference["f_px"] - FOCAL_PX) < 1e-9:
        raise ValueError("Stored reference estimate equals truth (bug guard)")
    if reference["reproj_rms_naive_px"] <= reference["reproj_rms_est_px"]:
        raise ValueError("Naive K must be worse than the estimated K")
    return {"report": report, **data}


class IntrinsicsDemo:
    """Chessboard poses, reprojection truth, and M-convergence narrative."""

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
            text="第二十五课 · 合成棋盘格张氏标定（估计相机内参 K）",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "9×6 内角点棋盘格（方格 0.03 m）被同一相机从多个非平行位姿观察；"
                "每幅图解出一个单应 H，\n"
                "把 v 向量约束堆叠解出 B = K⁻ᵀK⁻¹，再分解 K 与每幅外参。"
                "三个模式共用同一条正式实验记录：\n"
                "① 位姿集合与角点对应 ② 重投影：猜的 K vs 估计 K ③ f/cx/cy 随 M 收敛与退化反例；"
                "全部为静态图，无动画"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="poses")
        for label, key in (
            ("① 棋盘格位姿集合与角点对应", "poses"),
            ("② 单应重投影：标定前 vs 标定后", "reproj"),
            ("③ K 估计随 M 收敛与退化反例", "converge"),
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
                "模型：fx=fy=f、skew=0（4 参数版，与第二十二课 K 对齐）｜"
                "要求：≥2 幅图中棋盘平面法线互不平行，否则 v 系统降秩｜"
                "退化反例：纯平移（姿态全同）或固定俯仰环绕都让平面在相机系中平行"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def _pair_axes(self):
        self.fig.clear()
        return self.fig.subplots(1, 2)

    def draw_poses(self):
        corners = self.data["board_corners_m"]
        pool_eye = self.data["pool_eye_m"]
        ref_index = self.data["ref_pose_index"]
        self.fig.clear()
        ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        ax_img = self.fig.add_subplot(1, 2, 2)
        ax3d.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c="#2563eb", s=8)
        ax3d.scatter(
            pool_eye[:, 0], pool_eye[:, 1], pool_eye[:, 2], c="#94a3b8", s=14, label="姿态池"
        )
        ref_eye = pool_eye[ref_index]
        ax3d.scatter(
            ref_eye[:, 0], ref_eye[:, 1], ref_eye[:, 2], c="#dc2626", s=28, label="参考组位姿"
        )
        for eye in ref_eye:
            ax3d.plot(
                [eye[0], 0.0], [eye[1], 0.0], [eye[2], 0.0], color="#dc2626", lw=0.6, alpha=0.5
            )
        ax3d.set(
            xlabel="X_d / m",
            ylabel="Y_d / m",
            zlabel="高 / m",
            title=f"棋盘平面与姿态池（{len(pool_eye)} 个可用位姿）",
        )
        ax3d.view_init(elev=28, azim=-55)
        ax3d.legend(fontsize=8)
        clean = self.data["ref_pixels_clean"][0]
        noisy = self.data["ref_pixels_noisy"][0]
        for row in range(BOARD_ROWS):
            block = slice(row * BOARD_COLS, (row + 1) * BOARD_COLS)
            ax_img.plot(
                clean[block, 0],
                HEIGHT_PX - clean[block, 1],
                "-",
                color="#0f172a",
                lw=0.7,
                alpha=0.5,
            )
        ax_img.scatter(
            noisy[:, 0],
            HEIGHT_PX - noisy[:, 1],
            marker="x",
            s=36,
            c="#dc2626",
            label="检测角点（含噪声）",
        )
        ax_img.set_xlim(-10, WIDTH_PX + 10)
        ax_img.set_ylim(-10, HEIGHT_PX + 10)
        ax_img.set_aspect("equal")
        ax_img.set(
            xlabel="u / px",
            ylabel="v / px（已翻转）",
            title=f"参考组第 1 幅的 54 个角点（σ={REFERENCE_SIGMA_PX} px）",
        )
        ax_img.legend(fontsize=8, loc="lower left")

    def draw_reproj(self):
        from matplotlib.patches import Rectangle

        corners = self.data["board_corners_m"]
        homographies = self.data["ref_homographies"]
        k_est = intrinsic_matrix(*self.data["ref_k_est"])
        noisy = self.data["ref_pixels_noisy"][0]
        ax_img, ax_bar = self._pair_axes()
        estimated = [decompose_extrinsics(homography, k_est) for homography in homographies]
        naive = [decompose_extrinsics(homography, NAIVE_K) for homography in homographies]
        proj_est = project_points(
            corners, estimated[0]["rotation"], estimated[0]["translation"], k_est
        )
        proj_naive = project_points(corners, naive[0]["rotation"], naive[0]["translation"], NAIVE_K)
        ax_img.add_patch(
            Rectangle((0, 0), WIDTH_PX, HEIGHT_PX, fill=False, color="#0f172a", lw=1.0)
        )
        ax_img.scatter(
            noisy[:, 0], HEIGHT_PX - noisy[:, 1], marker="x", s=34, c="#dc2626", label="检测角点"
        )
        ax_img.scatter(
            proj_est[:, 0], HEIGHT_PX - proj_est[:, 1], s=10, c="#2563eb", label="估计 K 重投影"
        )
        ax_img.scatter(
            proj_naive[:, 0],
            HEIGHT_PX - proj_naive[:, 1],
            s=12,
            facecolors="none",
            edgecolors="#ea580c",
            label="猜的 K 重投影（分解位姿）",
        )
        ax_img.set_xlim(-60, WIDTH_PX + 60)
        ax_img.set_ylim(-60, HEIGHT_PX + 60)
        ax_img.set_aspect("equal")
        ax_img.set(
            xlabel="u / px",
            ylabel="v / px（已翻转）",
            title="参考组第 1 幅：位姿由各自 K 从 H 分解",
        )
        ax_img.legend(fontsize=8, loc="lower left")
        report = self.data["report"]
        reference = report["reference_group"]
        labels = ("无标定 K\n（猜的）", "真值 K\n（先验）", "估计 K\n（本课）")
        values = (
            reference["reproj_rms_naive_px"],
            reference["reproj_rms_true_px"],
            reference["reproj_rms_est_px"],
        )
        bars = ax_bar.bar(labels, values, color=["#94a3b8", "#22c55e", "#2563eb"])
        ax_bar.set_yscale("log")
        for bar, value in zip(bars, values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.15,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
        ax_bar.set(
            ylabel="角点重投影 RMS / px",
            title=f"真值位姿下的重投影（参考组 M={REFERENCE_M}）",
        )
        ax_bar.set_ylim(top=max(values) * 5)

    def draw_converge(self):
        report = self.data["report"]
        m_values = self.data["m_values"]
        f_err = self.data["k_est_f_err_mean_px"]
        cx_err = self.data["k_est_cx_err_mean_px"]
        ax_f, ax_degen = self._pair_axes()
        for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
            ax_f.plot(
                m_values,
                f_err[:, sigma_index],
                "-o",
                ms=4,
                label=f"f 误差，σ={sigma} px" if sigma else "f 误差，σ=0（数值精确）",
            )
        ax_f.plot(m_values, cx_err[:, 2], "--", color="#9333ea", lw=1.2, label="cx 误差，σ=1 px")
        ax_f.set_yscale("log")
        ax_f.set(
            xlabel="图像数量 M",
            ylabel=f"估计误差均值 / px（{report['runs_per_group']} 种子）",
            title="K 估计误差随图像数量 M 收敛",
        )
        ax_f.legend(fontsize=8)
        cases = report["degenerate_case"]
        names = ("参考组\n(M=5, 非平行)", "纯平移退化\n（姿态全同）", "固定俯仰环绕\n（法线全同）")
        reference_cond = float(
            intrinsic_from_homographies(list(self.data["ref_homographies"]))["cond"]
        )
        degen_translation = float(cases["parallel_translation"]["cond"])
        degen_orbit = float(cases["parallel_orbit_fixed_tilt"]["cond"])
        bars = ax_degen.bar(
            names,
            [reference_cond, degen_translation, degen_orbit],
            color=["#22c55e", "#94a3b8", "#ea580c"],
        )
        ax_degen.set_yscale("log")
        for bar, cond in zip(bars, (reference_cond, degen_translation, degen_orbit)):
            ax_degen.text(
                bar.get_x() + bar.get_width() / 2,
                cond * 1.4,
                f"{cond:.1e}",
                ha="center",
                fontsize=9,
            )
        ax_degen.set(ylabel="v 系统条件数", title="退化姿态集：条件数爆炸（守卫在 rank<3 触发）")

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.data["report"]
        reference = report["reference_group"]
        pool = report["pose_pool"]
        if mode == "poses":
            text = (
                "① 这一幕在摆数据：同一平面、多个非平行位姿\n\n"
                f"  棋盘 {BOARD_COLS}×{BOARD_ROWS} 内角点，方格 {SQUARE_M} m；\n"
                f"  姿态池 {pool['kept']}/{pool['candidates']} 个位姿可用\n"
                "  （3 个俯仰带 × 8 个方位角，整板须在画面内）。\n\n"
                f"  参考组抽 {REFERENCE_M} 幅，法线最大夹角 "
                f"{self.data['ref_normal_spread_deg'].item():.0f}°\n"
                "  抽样时要求 ≥2 个法线夹角 ≥ "
                f"{pool['min_normal_angle_deg']}°，\n"
                "  否则该组图就是张氏法的退化配置。\n\n"
                "  每幅图：已知平面坐标 ↔ 检测像素（高斯噪声），\n"
                "  归一化 DLT 解单应 H。"
            )
        elif mode == "reproj":
            text = (
                "② 这一幕在对照：猜的 K 与估计 K 差多少\n\n"
                f"  参考组（M={reference['m']}, σ={reference['sigma_px']} px）估计：\n"
                f"  f={reference['f_px']:.1f}（真值 {FOCAL_PX}）, "
                f"cx={reference['cx_px']:.1f}, cy={reference['cy_px']:.1f}\n\n"
                f"  真值位姿下重投影 RMS：\n"
                f"  猜的 K {reference['reproj_rms_naive_px']:.1f} px ≫ "
                f"估计 K {reference['reproj_rms_est_px']:.1f} px\n"
                f"  （真值 K {reference['reproj_rms_true_px']:.2f} px = 噪声地板）\n\n"
                "  陷阱（左图）：若位姿也从同一个 K 反解，\n"
                "  错的 K 会被错的位姿吸收，重投影看起来一样好——\n"
                "  这就是重投影 RMS 不能用来发现 K 错误的原因。"
            )
        else:
            c_table = np.array(report["mechanism_c_px"])
            pooled = float(np.nanmedian(c_table[:, 1:]))
            text = (
                "③ 这一幕在量化：误差怎么随图像数量 M 收敛\n\n"
                f"  σ=0：f/cx/cy 误差 ~1e-12 px（数值精确）；\n"
                f"  M=20, σ=2 px：f 误差均值 "
                f"{np.array(report['k_est_f_err_mean_px'])[-1, 3]:.1f} px。\n"
                f"  机制：误差 ≈ C/√M·σ，本实验 C 中位 ≈ {pooled:.0f} px\n"
                "  （C 吸收棋盘尺寸与视角几何）。\n\n"
                "  退化反例（右图）：纯平移或固定俯仰环绕\n"
                "  让所有平面在相机系中平行 → v 系统降秩，\n"
                "  守卫按 rank<3 触发；强行求解也得不到有效 K。\n\n"
                "  提醒：未做整体非线性精化，上面的误差是\n"
                "  闭式解的诚实水平。"
            )
        self.stats.configure(text=text)

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "poses":
            self.draw_poses()
        elif mode == "reproj":
            self.draw_reproj()
        else:
            self.draw_converge()
        self.fill_stats(mode)
        status = {
            "poses": "① 这一幕在摆数据：一平面多姿态、角点对应、以及“非平行”的抽样守卫",
            "reproj": "② 这一幕在对照：只有把位姿钉在真值上，重投影才暴露 K 的错误；分解链会自洽地吸收它",
            "converge": "③ 这一幕在量化：σ=0 精确、误差 ∝ σ/√M（C≈常数）、退化姿态集条件数爆炸被守卫捕获",
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
    IntrinsicsDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
