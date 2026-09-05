"""Lesson 27 viewer: masks, identities and what confusion does to the fix.

Three static modes share one lesson-27 recording (bench npz + summary):
1. the rendered view with truth/model mask overlays and identity labels
   (a slider steps through the four poses);
2. the localization error comparison: lesson-18 axis baseline vs mask-centroid
   groups vs the forced identity mismatch, per group and per pose;
3. the mismatch explosion and the delta-phi = delta-px / f propagation check.
Layout, Esc handling and the meaning panel follow the lesson-22/26 demos:
every mode calls fig.clear() first, all quoted numbers come from summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.visual_grounding import (
    BEARING_STD_RAD,
    EXPERIMENT,
    GROUP_NAMES,
    RANGE_STD_M,
    SHIFT_PX,
    mask_boundary,
)

DEFAULT_RESULTS = "results/visual_grounding_2026-09-05"
EXPECTED_NPZ_KEYS = {
    "rgb",
    "depth_m",
    "landmark_label",
    "valid",
    "pose_rotation",
    "pose_translation",
    "landmark_xy",
    "chosen_mask",
    "chosen_area",
    "chosen_iou",
    "chosen_axis_dist_m",
    "chosen_margin_m",
    "identity_ok",
    "candidate_count",
    "group_names",
    "obs_range_m",
    "obs_bearing_rad",
    "truth_poses",
    "err_vec_m",
    "err_norm_m",
    "heading_err_rad",
    "range_bias_m",
    "axis_range_m",
    "shift_px",
    "shift_bearing_rad",
    "shift_pred_ratio",
    "chain_shift_heading_rad",
    "chain_shift_pos_m",
    "meta_json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-27 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("scene", {}).get("canvas_px") != [640, 480]
        or report.get("noise", {}).get("range_std_m") != RANGE_STD_M
        or report.get("noise", {}).get("bearing_std_rad") != BEARING_STD_RAD
        or report.get("scene", {}).get("focal_px") != 600.0
        or list(report.get("mechanism", {}).get("shift_px", ())) != list(SHIFT_PX)
        or list(report.get("groups", {}).keys()) != list(GROUP_NAMES)
    ):
        raise ValueError("Incompatible lesson-27 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != EXPECTED_NPZ_KEYS:
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    if data["group_names"].tolist() != list(GROUP_NAMES):
        raise ValueError("Archive group names disagree with the summary")
    if not np.allclose(data["shift_px"], np.asarray(SHIFT_PX, dtype=float)):
        raise ValueError("Archive shift scan disagrees with the summary")
    if not np.allclose(data["landmark_xy"], report["scene"]["landmark_xy"], atol=1e-12):
        raise ValueError("Archive landmark axes disagree with the summary")
    if int(data["identity_ok"].sum()) != report["identity"]["correct"]:
        raise ValueError("Archive identity outcome disagrees with the summary")
    if (
        int(data["identity_ok"].size) != report["identity"]["total"]
        or not np.isclose(
            report["identity"]["accuracy"],
            report["identity"]["correct"] / max(report["identity"]["total"], 1),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError("Identity accuracy disagrees with the archive outcome")
    return {"report": report, **data}


class GroundingDemo:
    """Mask-to-identity narrative: overlays, error comparison, mechanism."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for titles and labels
        self.root = root
        self.data = data
        self.n_poses = int(data["rgb"].shape[0])
        self.n_land = int(data["landmark_xy"].shape[0])
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第二十七课 · 视觉基础模型给地标“发身份”：掩码 → 最近邻身份 → 定位链",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "MobileSAM 在渲染图上自动出掩码；掩码质心经渲染深度反投影，按最近地标轴分配身份，\n"
                "再走第十八课的 Procrustes 解算。三个模式共用同一条正式实验记录："
                "① 掩码叠加与身份 ② 误差对照 ③ 错配爆炸与传播；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="overlays")
        for label, key in (
            ("① 掩码叠加与身份标签", "overlays"),
            ("② 身份与掩码质量的定位误差对照", "groups"),
            ("③ 错配爆炸与 δφ=δpx/f 传播", "mechanism"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        self.step = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            controls,
            from_=0,
            to=max(0, self.n_poses - 1),
            orient="horizontal",
            length=200,
            command=self._on_slider,
        )
        self.slider.pack(side="left", padx=(0, 6))
        self.step_label = ttk.Label(controls, text="", width=22)
        self.step_label.pack(side="left")
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(middle)
        left.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(10.2, 4.3), dpi=100, layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=50, anchor="nw", justify="left")
        self.stats.pack(side="left", fill="y", padx=(12, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "身份来自几何（质心反投影 + 最近地标轴），模型只负责“哪里是一个物体”；"
                "真值标签只用于评估，不进入选择、分配或解算"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _on_slider(self, value):
        self.step.set(round(float(value)))
        self.step_label.configure(text=f"位姿 {self.step.get()} / {self.n_poses - 1}")
        if self.mode.get() == "overlays":
            self.draw_overlays()
            self.canvas.draw_idle()

    # ------------------------------------------------------------------ modes
    def draw_overlays(self):
        """① rendered view + truth/model mask edges + identity labels."""
        self.fig.clear()
        report = self.data["report"]
        ax_img, ax_iou = self.fig.subplots(1, 2)
        pose = min(self.step.get(), self.n_poses - 1)
        rgb = self.data["rgb"][pose]
        ax_img.imshow(rgb)
        for i in range(self.n_land):
            edge = mask_boundary(self.data["landmark_label"][pose] == (i + 1))
            overlay = np.zeros((*edge.shape, 4))
            overlay[edge] = (0.1, 0.7, 0.2, 0.9)
            ax_img.imshow(overlay)
            edge = mask_boundary(self.data["chosen_mask"][pose, i])
            overlay = np.zeros((*edge.shape, 4))
            overlay[edge] = (0.9, 0.2, 0.15, 0.9)
            ax_img.imshow(overlay)
            rows, cols = np.nonzero(self.data["chosen_mask"][pose, i])
            ok = bool(self.data["identity_ok"][pose, i])
            ax_img.text(
                cols.mean(),
                max(rows.mean() - 16.0, 8.0),
                f"L{i + 1} {'身份对' if ok else '身份错'}",
                color="white",
                fontsize=9,
                ha="center",
                bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
            )
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        n_correct = int(self.data["identity_ok"][pose].sum())
        ax_img.set(
            title=f"位姿 {pose}：真值边缘（绿）vs 模型掩码（红），\n"
            f"身份 {n_correct}/{self.n_land} 对",
        )
        iou = np.asarray(self.data["chosen_iou"], dtype=float).ravel()
        bars = ax_iou.bar(np.arange(iou.size), iou, color="#2563eb")
        for index, (bar, value) in enumerate(zip(bars, iou)):
            pose_i, land_i = divmod(index, self.n_land)
            bar.set_color("#dc2626" if not self.data["identity_ok"][pose_i, land_i] else "#2563eb")
            ax_iou.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                fontsize=7,
            )
        mean_iou = report["iou"]["mean"]
        ax_iou.axhline(mean_iou, color="#0f172a", ls="--", lw=1.0)
        ax_iou.set_ylim(0.0, 1.12)
        ax_iou.set(
            xlabel="位姿 × 地标（按位姿分块）",
            ylabel="IoU（模型掩码 vs 真值区域）",
            title=f"所选掩码 IoU：均值 {mean_iou:.3f}，最小 {report['iou']['min']:.3f}",
        )
        ax_iou.grid(alpha=0.2)

    def draw_groups(self):
        """② localization error: baseline vs mask groups vs mismatch."""
        self.fig.clear()
        report = self.data["report"]
        groups = report["groups"]
        ax_bar, ax_pose = self.fig.subplots(1, 2)
        names = list(GROUP_NAMES)
        means = [groups[name]["pos_mean_m"] * 100.0 for name in names]
        stds = [groups[name]["pos_std_m"] * 100.0 for name in names]
        colors = [
            "#64748b",
            "#2563eb",
            "#93c5fd",
            "#60a5fa",
            "#3b82f6",
            "#bfdbfe",
            "#ea580c",
            "#f97316",
            "#dc2626",
        ]
        bars = ax_bar.bar(names, means, yerr=stds, capsize=3, color=colors)
        for bar, mean in zip(bars, means):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                mean * 1.35,
                f"{mean:.1f}",
                ha="center",
                fontsize=7,
            )
        ax_bar.set_yscale("log")
        ax_bar.set_ylim(0.1, max(means) * 6.0)
        ax_bar.tick_params(axis="x", rotation=38, labelsize=7)
        ax_bar.set(
            ylabel="定位位置误差 / cm（4 位姿 × 20 种子，均值±1σ）",
            title="第十八课基线 vs 掩码质心组 vs 强制错配（对数轴）",
        )
        ax_bar.grid(alpha=0.2)
        for name, color in (
            ("baseline", "#64748b"),
            ("truth_mask", "#2563eb"),
            ("model_identity", "#f97316"),
            ("mismatch", "#dc2626"),
        ):
            per_pose = np.asarray(groups[name]["per_pose_mean_m"]) * 100.0
            ax_pose.plot(
                np.arange(len(per_pose)),
                per_pose,
                "-o",
                ms=4,
                color=color,
                label=f"{name}（均值 {per_pose.mean():.2f} cm）",
            )
        ax_pose.set_yscale("log")
        ax_pose.set_ylim(1.0, 2000.0)
        ax_pose.set(
            xlabel="相机位姿编号（0 = 第 22 课锚点位姿）",
            ylabel="每位姿平均位置误差 / cm",
            title="错配把每个位姿都炸到米级；其余组位姿间差异来自距离",
        )
        ax_pose.legend(fontsize=7, loc="upper left")
        ax_pose.grid(alpha=0.2)

    def draw_mechanism(self):
        """③ mismatch explosion + delta-phi = delta-px / f propagation."""
        self.fig.clear()
        report = self.data["report"]
        groups = report["groups"]
        ax_boom, ax_law = self.fig.subplots(1, 2)
        n_poses = report["n_poses"]
        poses = np.arange(n_poses)
        baseline = np.asarray(groups["baseline"]["per_pose_mean_m"]) * 100.0
        mismatch = np.asarray(groups["mismatch"]["per_pose_mean_m"]) * 100.0
        width = 0.38
        ax_boom.bar(poses - width / 2, baseline, width, color="#64748b", label="正确身份（基线）")
        ax_boom.bar(poses + width / 2, mismatch, width, color="#dc2626", label="强制循环错配")
        for x, value in zip(poses - width / 2, baseline):
            ax_boom.text(x, value * 1.15, f"{value:.1f}", ha="center", fontsize=7)
        for x, value in zip(poses + width / 2, mismatch):
            ax_boom.text(x, value * 1.15, f"{value:.0f}", ha="center", fontsize=7)
        ax_boom.set_yscale("log")
        ax_boom.set_ylim(1.0, 4000.0)
        ax_boom.set_xticks(poses)
        ax_boom.set(
            xlabel="相机位姿编号",
            ylabel="平均位置误差 / cm（对数轴）",
            title=f"同读数、同噪声，只换身份：\n均值 {groups['baseline']['pos_mean_m'] * 100:.1f} cm → "
            f"{groups['mismatch']['pos_mean_m'] * 100:.0f} cm，朝向偏 {groups['mismatch']['heading_mean_deg']:.1f}°",
        )
        ax_boom.legend(fontsize=7, loc="upper left")
        ax_boom.grid(alpha=0.2)
        shift_px = np.asarray(self.data["shift_px"], dtype=float)
        measured = np.degrees(np.nanmean(-self.data["shift_bearing_rad"], axis=(0, 1))) * 1000.0
        focal = report["scene"]["focal_px"]
        factor = float(np.nanmean(self.data["shift_pred_ratio"]))
        ax_law.plot(shift_px, measured, "-o", ms=4, color="#dc2626", label="实测 |Δβ|（精确射线差分）")
        ax_law.plot(shift_px, shift_px / focal * 1000.0, "--", color="#0f172a", label="δpx / f（水平相机）")
        ax_law.plot(
            shift_px,
            shift_px / focal * 1000.0 * factor,
            ":",
            color="#2563eb",
            label=f"δpx/f × 俯仰因子 {factor:.2f}",
        )
        chain = np.degrees(self.data["chain_shift_heading_rad"]) * 1000.0
        ax_law.plot(
            shift_px,
            chain,
            "-s",
            ms=4,
            color="#9333ea",
            label="整链朝向偏差（含 0.42° 噪声地板）",
        )
        ax_law.set(
            xlabel="掩码质心横移 δ / px（图内向右，世界方位角减小）",
            ylabel="方位角偏移 / mrad",
            title="|Δβ| ≈ δpx/f：质心偏差如何进入观测；\n共同横移被朝向吸收，位置误差几乎不动",
        )
        ax_law.legend(fontsize=7, loc="upper left")
        ax_law.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.data["report"]
        identity = report["identity"]
        masks = report["masks"]
        groups = report["groups"]
        bias = report["surface_range_bias_m"]["mean"] * 100.0
        if mode == "overlays":
            pose = min(self.step.get(), self.n_poses - 1)
            iou_pose = np.asarray(self.data["chosen_iou"], dtype=float)[pose]
            dist_pose = np.asarray(self.data["chosen_axis_dist_m"], dtype=float)[pose]
            iou_text = "  ".join(f"L{i + 1} {value:.2f}" for i, value in enumerate(iou_pose))
            dist_text = "  ".join(f"L{i + 1} {value * 100:.1f}" for i, value in enumerate(dist_pose))
            text = (
                "① 这一幕在摆数据：模型出了掩码，身份由几何决定\n\n"
                f"  候选掩码（本位姿 / 全部）：\n"
                f"  {int(self.data['candidate_count'][pose])} / {masks['total']}"
                f"（漏检地标 {masks['missed_landmarks']} 个）\n\n"
                f"  所选掩码 IoU：{iou_text}\n"
                f"  质心离地标轴 / cm：{dist_text}\n\n"
                f"  身份正确 {identity['correct']}/{identity['total']}"
                f"（准确率 {identity['accuracy']:.0%}），\n"
                f"  最近轴距离均值 {identity['mean_axis_dist_m'] * 100:.2f} cm\n"
                f"  ≈ 圆柱半径（质心落在近侧柱面上），\n"
                f"  最小身份裕度 {identity['min_margin_m']:.2f} m（远大于 6 cm）\n\n"
                "  绿 = 真值区域，红 = 模型掩码；拖动滑条换位姿。"
            )
        elif mode == "groups":
            text = (
                "② 这一幕在对照：把“身份已知”换成模型给身份\n\n"
                f"  基线（第 18 课轴观测，真值身份）：\n"
                f"  {groups['baseline']['pos_mean_m'] * 100:.2f} ± {groups['baseline']['pos_std_m'] * 100:.2f} cm\n\n"
                f"  掩码质心观测（真值掩码 / 模型掩码）：\n"
                f"  {groups['truth_mask']['pos_mean_m'] * 100:.2f} / "
                f"{groups['model_mask']['pos_mean_m'] * 100:.2f} cm\n"
                f"  腐蚀膨胀 ±4/8 px：{groups['erode4']['pos_mean_m'] * 100:.2f} / "
                f"{groups['erode8']['pos_mean_m'] * 100:.2f} / "
                f"{groups['dilate4']['pos_mean_m'] * 100:.2f} / "
                f"{groups['dilate8']['pos_mean_m'] * 100:.2f} cm\n\n"
                f"  模型身份（完整管线）：{groups['model_identity']['pos_mean_m'] * 100:.2f} cm\n"
                f"  表面测距系统偏置：{bias:+.2f} cm ≈ −半径 6 cm\n\n"
                "  身份全对时，定位链只多付“质心在柱面上”这笔账；\n"
                "  掩码质量（腐蚀/膨胀）几乎不影响质心一阶矩。"
            )
        else:
            text = (
                "③ 这一幕在量化：身份混淆与质心偏差的传播\n\n"
                f"  强制循环错配（同读数同噪声）：\n"
                f"  {groups['baseline']['pos_mean_m'] * 100:.1f} cm → "
                f"{groups['mismatch']['pos_mean_m'] * 100:.0f} cm\n"
                f"  （中位 {groups['mismatch']['pos_median_m'] * 100:.0f} cm，"
                f"朝向 {groups['mismatch']['heading_mean_deg']:.1f}° ≈ 三点循环 signature）\n\n"
                f"  传播定律：|Δβ| = δpx/f（f={report['scene']['focal_px']:.0f} px）\n"
                f"  实测/定律 比值 {report['mechanism']['shift_ratio_to_px_over_f'][-1]:.3f}"
                f"（一阶公式，偏轴项未建模）\n\n"
                f"  整链：共同横移几乎全部进朝向\n"
                f"  （{np.degrees(self.data['chain_shift_heading_rad'])[-1] * 1000:.1f} mrad @ 8 px），\n"
                f"  位置 {self.data['chain_shift_pos_m'][-1] * 100:.1f} cm 几乎不动\n\n"
                "  错配是系统性错误：平均不掉，只能靠正确的关联。"
            )
        self.stats.configure(text=text)

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        self.slider.configure(from_=0, to=max(0, self.n_poses - 1))
        if mode == "overlays":
            self.draw_overlays()
            self.step_label.configure(text=f"位姿 {min(self.step.get(), self.n_poses - 1)} / {self.n_poses - 1}")
        elif mode == "groups":
            self.draw_groups()
        else:
            self.draw_mechanism()
        self.fill_stats(mode)
        status = {
            "overlays": "① 这一幕在摆数据：MobileSAM 掩码 + 深度反投影 + 最近邻身份",
            "groups": "② 这一幕在对照：真值身份基线 vs 模型身份 vs 强制错配（同噪声流）",
            "mechanism": "③ 这一幕在量化：错配爆炸形态与 δφ=δpx/f 的传播核对",
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
    root.title("第二十七课 · 视觉基础模型给地标“发身份”")
    root.geometry("1600x760")
    root.minsize(1380, 660)
    GroundingDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
