"""Lesson 22 read-only viewer: world -> pixels, exact depth, ray, noisy depth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pinhole_projection import (
    CX_PX,
    CY_PX,
    DEPTH_NOISE_STD_M,
    EXPERIMENT,
    EYE,
    FOCAL_PX,
    HEIGHT_PX,
    K_INTRINSIC,
    NEAR_PLANE_M,
    TARGET,
    WIDTH_PX,
    look_at,
    project_with_depth,
    unproject,
)

DEFAULT_RESULTS = "results/mobile_pinhole_2026-09-03"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replays(directory):
    """Validate a lesson-22 recording; returns the archive dict (read-only)."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != EXPERIMENT
        or report.get("schema_version") != 1
        or report.get("camera_model") != "pinhole {u = K [R|t] X / z}"
        or report.get("canvas_px") != [WIDTH_PX, HEIGHT_PX]
        or report.get("focal_px") != FOCAL_PX
        or report.get("depth_noise_std_m") != DEPTH_NOISE_STD_M
        or report.get("eye_m") != list(EYE)
        or report.get("target_m") != list(TARGET)
        or not np.array_equal(report.get("intrinsic"), K_INTRINSIC)
    ):
        raise ValueError("Incompatible lesson-22 recording")
    path = directory / "trajectories.npz"
    if digest(path) != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != {
            "world_points",
            "projected_pixels",
            "camera_points",
            "roundtrip_error_m",
            "ray_points",
            "ray_depths",
            "noisy_cloud_errors_m",
            "noisy_cloud_reprojection_px",
            "cloud_depths",
            "cloud_world",
            "mean_error_by_depth",
            "depth_bins",
            "pose_error_m",
            "ray_norm",
            "depth_noise_estimates_m",
        }:
            raise ValueError("Unexpected archive arrays")
        points = npz["world_points"].copy()
        pixels = npz["projected_pixels"].copy()
    # Independently recompute round trip from the stored scene.
    rotation, translation = look_at(EYE, TARGET)
    fresh_pixels, fresh_depths, fresh_world = project_with_depth(points, rotation, translation)
    if not np.array_equal(fresh_pixels, pixels):
        raise ValueError("Stored pixels do not match the scene/camera")
    clouds = unproject(np.asarray(pixels), np.asarray(fresh_depths), rotation, translation)
    if not np.allclose(np.linalg.norm(clouds - fresh_world, axis=1), 0, atol=1e-12):
        raise ValueError("Round trip inconsistent")
    return {
        "report": report,
        "points": points,
        "pixels": pixels,
        "rotation": rotation,
        "translation": translation,
    }


class PinholeDemo:
    """Three modes: exact depth / no-depth ray / noisy depth (same pixels)."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.data = data
        outer = ttk.Frame(root if parent is None else parent, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="第二十二课 · 针孔相机：世界 ↔ 像素 ↔ 点云",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "相机内参 fx=600 px、主点 (320,240)、画布 640×480；外参由 look-at 给出\n"
                "黄色十字=主点；三种查看模式只改解释，不改变同一组投影"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="exact")
        for label, key, note in (
            ("① 精确深度：往返", "exact", "投影→反投影误差 1e-15 m 级"),
            ("② 无深度：只有一条射线", "ray", "同一像素三个深度→相同像素、共线世界点"),
            ("③ 深度噪声：点云误差", "noisy", "20 次统计：误差÷射线倍率 ≈ 4 cm"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        ttk.Label(controls, text="（" + note + "）").pack(side="left")
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        self.canvas = tk.Canvas(
            middle, background="#ffffff", highlightthickness=0, width=720, height=430
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=48, anchor="nw", justify="left")
        self.stats.pack(side="right", fill="y", padx=(12, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "投影：u = fx·x/z + cx ；反投影：X = d · K⁻¹[u,v,1]（d 为深度）｜"
                "无深度时同一像素只约束一条射线（单目尺度不可恢复的一点）｜"
                "点云误差 = 深度噪声 × 射线长度 |K⁻¹[u,v,1]|，图像边缘放大"
            ),
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def redraw(self):
        if not hasattr(self, "canvas"):
            return
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 60:
            return
        pixels = np.asarray(self.data["pixels"])
        scale = min((w - 80) / WIDTH_PX, (h - 90) / HEIGHT_PX)
        ox, oy = (w - WIDTH_PX * scale) / 2, (h - HEIGHT_PX * scale) / 2

        def xy(pixel):
            return ox + pixel[0] * scale, oy + (HEIGHT_PX - pixel[1]) * scale

        # Camera frame + principal point.
        c.create_rectangle(ox, oy, ox + WIDTH_PX * scale, oy + HEIGHT_PX * scale, outline="#cbd5e1")
        px, py = xy([CX_PX, CY_PX])
        c.create_line(px - 10, py, px + 10, py, fill="#dc2626", width=2)
        c.create_line(px, py - 10, px, py + 10, fill="#dc2626", width=2)
        c.create_text(
            ox + 8,
            oy + 14,
            text="图像平面 640×480 px（红叉=主点）",
            anchor="nw",
            font=("Microsoft YaHei", 9),
            fill="#64748b",
        )
        if self.mode.get() == "exact":
            for u, v in pixels:
                x, y = xy([u, v])
                c.create_oval(x - 2.2, y - 2.2, x + 2.2, y + 2.2, fill="#2563eb", outline="")
            report = self.data["report"]
            self.stats.configure(
                text=(
                    "① 精确深度（模拟理想深度传感器）\n\n"
                    f"方法：pixel + depth → K⁻¹ 射线 × depth → 世界系\n"
                    f"往返最大误差：{report['roundtrip_max_error_m']:.2e} m\n"
                    f"可见点数：{report['visible_points']}（近裁剪面 {NEAR_PLANE_M:.1f} m 之外）\n\n"
                    "深度是米制且精确时，三维点可以被完全恢复。\n"
                    "这就是 RGB-D / 激光雷达所能提供的：每像素一个距离。"
                )
            )
        elif self.mode.get() == "ray":
            report = self.data["report"]
            for u, v in pixels:
                x, y = xy([u, v])
                c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill="#cbd5e1", outline="")
            payload = report["ray_payload"]
            pixel = payload["pixel"]
            x, y = xy(pixel)
            c.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#0891b2", width=3)
            text = "无深度：三个候选深度（米）\n"
            for depth, point in zip(payload["depths_m"], payload["points_m"]):
                text += (
                    f"  {depth:.1f} m → 世界 ({point[0]:+.2f}, {point[1]:+.2f}, {point[2]:+.2f})\n"
                )
            text += (
                "\n三个点在同一像素（重投影一致），\n世界坐标却完全不同——\n"
                "像素本身不含深度信息，只有一条射线。"
            )
            self.stats.configure(text="② " + text)
        else:
            report = self.data["report"]
            for u, v in pixels:
                x, y = xy([u, v])
                c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill="#94a3b8", outline="")
            self.stats.configure(
                text=(
                    "③ 深度噪声（σ=5 cm，20 个种子）\n\n"
                    f"点云位置误差均值 {report['noisy_mean_error_m'] * 100:.2f} cm\n"
                    f"最大值 {report['noisy_max_error_m'] * 100:.2f} cm\n"
                    f"误差 ÷ 射线倍率 ≈ {report['noise_estimate_mean_m'] * 100:.2f} cm"
                    f"（近 {report['noise_estimate_at_near_m'] * 100:.2f} / "
                    f"远 {report['noise_estimate_at_far_m'] * 100:.2f}）\n\n"
                    "同一个 5 cm 深度噪声，在图像边缘（近处物体）放得更大。\n"
                    "误差 ÷ |K⁻¹[u,v,1]| 恒定 ≈ 4 cm（= σ·√(2/π)），\n"
                    "证明机制：点云误差 = 深度噪声 × 射线长度。"
                )
            )
        self.status.configure(
            text="三种模式共用同一组像素；切模式不改数据。窗口右上为数值面板。按 Esc 或关闭退出。"
        )


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.geometry("1180x640")
    root.minsize(1000, 580)
    PinholeDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
