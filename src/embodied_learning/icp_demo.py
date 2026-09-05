"""Lesson 26 viewer: two noisy clouds, the ICP iteration, and the symmetry.

Three static modes share one lesson-26 recording:
1. the two raw point clouds (A frame vs B frame) and the ground-truth transform;
2. the reference ICP run step by step: correspondences + current pose, with a
   slider over the stored iterations;
3. the observable-subspace error vs sigma, the convergence-radius matrix and
   the degenerate ground-only counterexample.
Layout, Esc handling and the meaning panel follow the lesson-22/23 demos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.point_cloud_icp import (
    CONVERGED_TRANS_M,
    DEGEN_SIGMA_INDEX,
    EIGEN_CUTOFF,
    EXPERIMENT,
    MAX_CLOUD_POINTS,
    MAX_ITERS,
    POSE_DELTA_M,
    POSE_YAW_DEG,
    RADIUS_SHIFT_M,
    RADIUS_SIGMA_M,
    RADIUS_YAW_DEG,
    SIGMA_VALUES,
    TAU0_M,
    TAU_FLOOR_M,
    TAU_GAMMA,
    VOXEL_SIZE_M,
    apply_transform,
    build_clouds,
    nearest_indices,
    scene_and_poses,
)

DEFAULT_RESULTS = "results/point_cloud_icp_2026-09-05"
EXPECTED_NPZ_KEYS = {
    "gt_rotation",
    "gt_translation",
    "ref_init_rotation",
    "ref_init_translation",
    "ref_fixed_cloud",
    "ref_moving_cloud",
    "ref_fixed_pole",
    "ref_fixed_normals",
    "ref_moving_normals",
    "ref_point_to_point_rotation",
    "ref_point_to_point_translation",
    "ref_point_to_point_mean_dist_m",
    "ref_point_to_point_inlier_ratio",
    "ref_point_to_point_tau_m",
    "ref_point_to_plane_rotation",
    "ref_point_to_plane_translation",
    "ref_point_to_plane_mean_dist_m",
    "ref_point_to_plane_inlier_ratio",
    "ref_point_to_plane_tau_m",
    "degen_init_rotation",
    "degen_init_translation",
    "degen_fixed_cloud",
    "degen_moving_cloud",
    "degen_final_rotation",
    "degen_final_translation",
    "degen_final_moving_cloud",
    "degen_mean_dist_m",
    "degen_tau_m",
    "sigma_values",
    "sweep_trans_mean_m",
    "sweep_trans_std_m",
    "sweep_trans_median_m",
    "sweep_trans_max_m",
    "sweep_rot_mean_deg",
    "sweep_rot_std_deg",
    "sweep_trans_obs_mean_m",
    "sweep_trans_obs_std_m",
    "sweep_rot_obs_mean_deg",
    "sweep_spin_mean_deg",
    "sweep_spin_std_deg",
    "sweep_iters_mean",
    "sweep_iters_median",
    "sweep_iters_max",
    "sweep_converged_count",
    "sweep_mean_pair_dist_m",
    "sweep_vec_mean_m",
    "radius_yaw_values",
    "radius_shift_values",
    "radius_converged_fraction",
    "radius_mean_trans_obs_m",
    "radius_mean_trans_naive_m",
    "radius_mean_spin_deg",
    "radius_mean_iters",
    "identity_converged_fraction",
    "identity_mean_trans_err_m",
    "identity_mean_trans_obs_m",
    "identity_mean_rot_err_deg",
    "truth_mean_trans_err_m",
    "truth_mean_trans_obs_m",
    "degen_trans_mean_m",
    "degen_trans_std_m",
    "degen_rot_mean_deg",
    "degen_converged_count",
    "degen_converged_but_wrong_count",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-26 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("canvas_px") != [640, 480]
        or report.get("pose_b", {}).get("delta_world_m") != list(POSE_DELTA_M)
        or report.get("pose_b", {}).get("yaw_deg") != POSE_YAW_DEG
        or report.get("sigma_values_m") != list(SIGMA_VALUES)
        or report.get("downsample", {}).get("voxel_size_m") != VOXEL_SIZE_M
        or report.get("downsample", {}).get("max_points") != MAX_CLOUD_POINTS
        or report.get("icp", {}).get("max_iters") != MAX_ITERS
        or report.get("icp", {}).get("tau0_m") != TAU0_M
        or report.get("icp", {}).get("tau_gamma") != TAU_GAMMA
        or report.get("icp", {}).get("tau_floor_m") != TAU_FLOOR_M
        or report.get("icp", {}).get("p2plane_eigen_cutoff") != EIGEN_CUTOFF
    ):
        raise ValueError("Incompatible lesson-26 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    # Re-derive the reference clouds from the lesson-22 constants and the seed
    # (scene_and_poses already cross-checks the renderer against lesson 23).
    scene = scene_and_poses()
    clouds = build_clouds(scene, 0.15, DEGEN_SIGMA_INDEX, 0, report.get("base_seed", 0))
    if not np.allclose(clouds["fixed"], data["ref_fixed_cloud"], atol=1e-9):
        raise ValueError("Stored fixed cloud does not match the seeded scene")
    if not np.allclose(clouds["moving"], data["ref_moving_cloud"], atol=1e-9):
        raise ValueError("Stored moving cloud does not match the seeded scene")
    return {"report": report, "scene": scene, **data}


class IcpDemo:
    """Two-cloud registration narrative: raw clouds, iterations, error anatomy."""

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
            text="第二十六课 · 两帧带噪点云的 ICP 配准",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "相机 A 沿用第二十二课位姿；相机 B 平移 [0.5, −0.3, 0.1] m 并绕世界 z 转 −25°，"
                "各自渲染第二十二课场景（地面 + 杆）后加深度噪声。\n"
                "三个模式共用同一条正式实验记录：① 两片原始点云与真值变换 ② ICP 迭代过程（可步进）"
                " ③ 误差、收敛半径与退化反例；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="clouds")
        for label, key in (
            ("① 两片原始点云与真值", "clouds"),
            ("② ICP 迭代过程（对应线 + 位姿）", "iters"),
            ("③ 误差、收敛半径与退化反例", "error"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        self.step = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=1,
            orient="horizontal",
            length=260,
            command=self._on_slider,
        )
        self.slider.pack(side="left", padx=(0, 6))
        self.step_label = ttk.Label(controls, text="", width=24)
        self.step_label.pack(side="left")
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
                "场景对称性：绕杆轴的旋转同时保持地面与杆面不变 → ICP 只能把位姿确定到该对称族；"
                "杆把不可观自由度从纯地面的 3 维压到 1 维"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _on_slider(self, value):
        self.step.set(round(float(value)))
        self.step_label.configure(text=f"第 {self.step.get()} 步 / 共 {self._n_steps() - 1}")
        if self.mode.get() == "iters":
            self.draw_iters()
            self.canvas.draw_idle()

    def _n_steps(self):
        return len(self.data["ref_point_to_plane_mean_dist_m"]) + 1

    # ------------------------------------------------------------------ modes
    def _cloud_axes(self):
        self.fig.clear()
        return self.fig.subplots(1, 2, subplot_kw={"projection": "3d"})

    def draw_clouds(self):
        ax_raw, ax_al = self._cloud_axes()
        fixed = self.data["ref_fixed_cloud"]
        moving = self.data["ref_moving_cloud"]
        gt_r, gt_t = self.data["gt_rotation"], self.data["gt_translation"]
        ax_raw.scatter(*fixed.T, s=3, c="#2563eb", alpha=0.5, label="A 系点云（固定）")
        ax_raw.scatter(*moving.T, s=3, c="#ea580c", alpha=0.5, label="B 系点云（待配准）")
        ax_raw.set(title="配准前：各在自己的相机系（错位）", xlabel="x / m", ylabel="y / m")
        ax_al.scatter(*fixed.T, s=3, c="#2563eb", alpha=0.5, label="A 系点云")
        aligned = apply_transform(gt_r, gt_t, moving)
        ax_al.scatter(*aligned.T, s=3, c="#16a34a", alpha=0.5, label="B 点云按真值变换后")
        ax_al.set(
            title="真值变换下两片点云重合（表面重合，采样不同）", xlabel="x / m", ylabel="y / m"
        )
        for ax in (ax_raw, ax_al):
            ax.view_init(elev=35, azim=-60)
            ax.legend(fontsize=7, loc="upper left")

    def draw_iters(self):
        report = self.data["report"]
        k = min(self.step.get(), self._n_steps() - 1)
        self.fig.clear()
        ax3d = self.fig.add_subplot(1, 2, 1, projection="3d")
        ax_curve = self.fig.add_subplot(1, 2, 2)
        fixed = self.data["ref_fixed_cloud"]
        moving = self.data["ref_moving_cloud"]
        rotations = self.data["ref_point_to_plane_rotation"]
        translations = self.data["ref_point_to_plane_translation"]
        k = min(k, len(rotations) - 1)
        current = apply_transform(rotations[k], translations[k], moving)
        ax3d.scatter(*fixed.T, s=3, c="#2563eb", alpha=0.45, label="A 系点云")
        ax3d.scatter(*current.T, s=3, c="#ea580c", alpha=0.55, label=f"B 点云 @ 第 {k} 步")
        # correspondence lines for a subset of the kept pairs
        idx, dist = nearest_indices(current, fixed)
        keep = np.flatnonzero(dist <= max(0.30, TAU0_M * TAU_GAMMA**k))
        for i in keep[:: max(1, len(keep) // 90)]:
            ax3d.plot(
                [current[i, 0], fixed[idx[i], 0]],
                [current[i, 1], fixed[idx[i], 1]],
                [current[i, 2], fixed[idx[i], 2]],
                color="#94a3b8",
                lw=0.6,
                alpha=0.5,
            )
        ax3d.set(title=f"参考运行第 {k} 步：灰线 = 保留的对应", xlabel="x / m", ylabel="y / m")
        ax3d.view_init(elev=35, azim=-60)
        ax3d.legend(fontsize=7, loc="upper left")
        for key, color, label in (
            ("ref_point_to_point_mean_dist_m", "#ea580c", "点到点"),
            ("ref_point_to_plane_mean_dist_m", "#2563eb", "点对面"),
        ):
            values = self.data[key]
            ax_curve.plot(np.arange(len(values)), values, "-o", ms=3, color=color, label=label)
            taus = self.data[key.replace("mean_dist_m", "tau_m")]
            ax_curve.plot(np.arange(len(taus)), taus, ":", color=color, lw=1.0, alpha=0.7)
        ax_curve.axvline(k, color="#0f172a", lw=0.8, alpha=0.6)
        ax_curve.set_yscale("log")
        ax_curve.set(
            xlabel="迭代步（点线 = 剔除阈值 τ）",
            ylabel="保留对应的平均距离 / m",
            title="对应-求解交替的收敛轨迹",
        )
        ax_curve.legend(fontsize=8)
        iterations = report["reference_run"]["iterations"]
        self.step_label.configure(
            text=(
                f"第 {k} 步 / 共 {self._n_steps() - 1}"
                f"（点对面共 {iterations['point_to_plane']} 轮，"
                f"点到点 {iterations['point_to_point']} 轮）"
            )
        )

    def draw_error(self):
        ax_sigma, ax_degen = self.fig.subplots(1, 2)
        sigma_values = self.data["sigma_values"]
        for oi, (color, label) in enumerate(zip(("#ea580c", "#2563eb"), ("点到点", "点对面"))):
            means = self.data["sweep_trans_obs_mean_m"][oi] * 100.0
            stds = self.data["sweep_trans_obs_std_m"][oi] * 100.0
            ax_sigma.errorbar(
                sigma_values,
                means,
                yerr=stds,
                fmt="-o",
                ms=4,
                capsize=3,
                color=color,
                label=label,
            )
        ax_sigma.axhline(CONVERGED_TRANS_M * 100.0, color="#0f172a", ls="--", lw=1.0)
        ax_sigma.set_xscale("log")
        ax_sigma.set_yscale("log")
        ax_sigma.set(
            xlabel="深度噪声 σ / m",
            ylabel="可观子空间平移误差 / cm（真值初值）",
            title="真值初值下误差随 σ 近似线性放大",
        )
        ax_sigma.legend(fontsize=8)
        fraction = self.data["radius_converged_fraction"][0]
        im = ax_degen.imshow(fraction, cmap="RdYlGn", vmin=0.0, vmax=1.0, origin="lower")
        for yi in range(fraction.shape[0]):
            for xi in range(fraction.shape[1]):
                ax_degen.text(
                    xi, yi, f"{fraction[yi, xi]:.0%}", ha="center", va="center", fontsize=9
                )
        ax_degen.set_xticks(range(len(RADIUS_SHIFT_M)), [f"{v}" for v in RADIUS_SHIFT_M])
        ax_degen.set_yticks(range(len(RADIUS_YAW_DEG)), [f"{v}°" for v in RADIUS_YAW_DEG])
        ax_degen.set(
            xlabel="平移初值扰动 / m（世界系 x）",
            ylabel="旋转初值扰动（绕世界 z）",
            title=f"收敛半径矩阵（点对面，σ={RADIUS_SIGMA_M} m，<2 cm 占比）",
        )
        fig_colorbar = self.fig.colorbar(im, ax=ax_degen, shrink=0.8)
        fig_colorbar.set_label("收敛占比")

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.data["report"]
        reference = report["reference_run"]
        radius = report["radius_study"]
        degen = report["degenerate"]
        if mode == "clouds":
            pixels = report["pixels"]
            down = report["downsample"]
            gt = report["pose_b"]
            text = (
                "① 这一幕在摆数据：两台相机、两片带噪点云\n\n"
                f"  相机 B = A 平移 {gt['delta_world_m']} m、绕 z 转 {gt['yaw_deg']}°\n"
                f"  真值变换：旋转 {gt['gt_rotation_deg']:.1f}°（绕世界 z）\n"
                f"  平移 {gt['gt_translation_m']}（模长 {gt['gt_translation_norm_m']:.3f} m）\n\n"
                f"  渲染像素：A {pixels['valid_a']}（杆 {pixels['pole_a']}），\n"
                f"  B {pixels['valid_b']}（杆 {pixels['pole_b']}）\n"
                f"  降采样：{down['voxel_size_m']} m 体素各取 1 点 →\n"
                f"  {down['ref_fixed_points']} / {down['ref_moving_points']} 点"
                f"（杆 {down['ref_fixed_pole_points']} / {down['ref_moving_pole_points']}）\n\n"
                f"  右图：真值变换下两片点云表面重合——\n"
                "  ICP 要找回的就是这个变换。"
            )
        elif mode == "iters":
            q = reference
            text = (
                "② 这一幕在跑算法：对应-求解交替\n\n"
                f"  参考运行：σ={reference['sigma_m']} m、0.1 m 平移初值误差\n"
                f"  点对面：{reference['iterations']['point_to_plane']} 轮收敛"
                f"（末秩 {reference.get('p2plane_rank_last', '—')}/6）\n"
                f"  点到点：{reference['iterations']['point_to_point']} 轮未收敛\n\n"
                f"  点对面平移误差 {q['final_trans_err_m']['point_to_plane'] * 100:.2f} cm\n"
                f"  （对称族内 {q['final_trans_obs_m']['point_to_plane'] * 100:.2f} cm，"
                f"谷坐标 {q['final_spin_deg']['point_to_plane']:+.1f}°）\n\n"
                "  场景有一个精确对称族：绕杆轴的旋转\n"
                "  同时保持地面与杆不变——旋转自由度\n"
                "  结构性不可观，ICP 只能确定到该族。\n"
                "  拖动滑条逐步观察对应与位姿。"
            )
        else:
            truth = radius["truth_init"]["point_to_plane"]
            identity = radius["identity_init"]["point_to_plane"]
            pole = degen["pole_scene"]["point_to_plane"]
            ground = degen["ground_only"]["point_to_plane"]
            text = (
                "③ 这一幕在量化：误差、半径与退化\n\n"
                f"  真值初值：可观误差 {truth['mean_trans_obs_m'] * 100:.2f} cm（σ=0.02 m）\n"
                f"  单位阵初值：朴素 {identity['mean_trans_err_m'] * 100:.0f} cm、"
                f"谷上 {identity['mean_trans_obs_m'] * 100:.0f} cm —— 无先验不可用\n\n"
                f"  收敛半径（可观 <2 cm 判据）：平移 ~0.1-0.2 m、\n"
                f"  旋转 ~0°（切向残差按 ~3 m/rad 留在谷里）\n\n"
                f"  退化反例（0.1 m 平移初值，σ=0.15 m）：\n"
                f"  含杆 {pole['trans_mean_m'] * 100:.1f} cm → "
                f"仅地面 {ground['trans_mean_m'] * 100:.1f} cm\n"
                f"  （{ground['converged_but_wrong_count']}/20 “收敛但错”）\n\n"
                "  杆 = 打破平面对称的非对称几何。"
            )
        self.stats.configure(text=text)

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "clouds":
            self.draw_clouds()
        elif mode == "iters":
            self.slider.configure(from_=0, to=self._n_steps() - 1)
            self.draw_iters()
        else:
            self.draw_error()
        self.fill_stats(mode)
        status = {
            "clouds": "① 这一幕在摆数据：两个传感器系里的两片带噪点云，真值变换已知",
            "iters": "② 这一幕在跑算法：对应-求解交替、阈值收缩、以及绕杆轴对称族上的漂移",
            "error": "③ 这一幕在量化：σ 线性地板、收敛半径、以及去掉杆后的平面滑动",
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
    IcpDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
