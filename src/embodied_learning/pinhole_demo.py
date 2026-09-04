"""Lesson 22 viewer: rotatable 3D scene view + pixel plane + explanation panel.

The 3D view is itself a pinhole observation: a second synthetic camera looks at
the scene from the south-east so you can turn and zoom the whole geometry.
"""

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
    """Validate a lesson-22 recording; returns the data dict."""
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
        expected = {
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
        }
        if set(npz.files) != expected:
            raise ValueError("Unexpected archive arrays")
        points = npz["world_points"].copy()
        pixels = npz["projected_pixels"].copy()
        noisy_world = npz["cloud_world"].copy()
        noisy_errors = npz["noisy_cloud_errors_m"].copy()
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
        "visible_world": fresh_world,
        "noisy_world": noisy_world,
        "noisy_errors": noisy_errors,
    }


class PinholeDemo:
    """Rotatable 3D scene (mpl) + pixel plane (tk canvas) + numeric panel."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for the 3D axis labels/titles
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
                "左边 3D 视图可拖拽旋转/滚轮缩放：地面网格 + 竖直杆 + 相机金字塔与观测射线\n"
                "右边像素平面（红叉=主点）与数字面板；三种模式共用同一组像素，只换解释"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="exact")
        for label, key in (
            ("① 精确深度：往返", "exact"),
            ("② 无深度：只有一条射线", "ray"),
            ("③ 深度噪声：点云误差", "noisy"),
        ):
            ttk.Radiobutton(
                controls, text=label, variable=self.mode, value=key, command=self.redraw
            ).pack(side="left", padx=(0, 10))
        middle = ttk.Frame(outer)
        middle.pack(fill="both", expand=True, pady=(8, 0))
        # --- 3D panel (matplotlib embedded; itself a pinhole view) ---
        three = ttk.Frame(middle)
        three.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(6.4, 4.6), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.canvas3d = FigureCanvasTkAgg(self.fig, master=three)
        self.canvas3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas3d, three).pack(side="top")
        # --- 2D pixel plane ---
        two = ttk.Frame(middle)
        two.pack(side="left", fill="both", padx=(10, 0))
        self.canvas = tk.Canvas(
            two, background="#ffffff", highlightthickness=0, width=460, height=430
        )
        self.canvas.pack(side="top", fill="both", expand=True)
        # --- numeric panel ---
        self.stats = ttk.Label(middle, width=42, anchor="nw", justify="left")
        self.stats.pack(side="left", fill="y", padx=(12, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "投影：u = fx·x/z + cx ；反投影：X = d · K⁻¹[u,v,1]（d 为深度）｜无深度时同一像素只约束一条射线｜"
                "点云误差 = 深度噪声 × 射线长度 |K⁻¹[u,v,1]|，图像边缘放大｜3D 视图=另一台针孔相机"
            ),
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    def _scene_base(self):
        """Ground grid lines, pole line, camera pyramid — shared by all modes."""
        ax = self.ax3d
        xs = np.linspace(0.75, 5.25, 10)
        for value in xs:
            ax.plot([value, value], [xs[0], xs[-1]], [0, 0], color="#cbd5e1", lw=0.6)
            ax.plot([xs[0], xs[-1]], [value, value], [0, 0], color="#cbd5e1", lw=0.6)
        ax.plot([4.2, 4.2], [3.6, 3.6], [0, 2.0], color="#0f172a", lw=2.5)
        # Camera pyramid: rays through the four image corners at 2 m.
        inverse = np.linalg.inv(K_INTRINSIC)
        corners_px = np.array([[0, 0], [WIDTH_PX, 0], [WIDTH_PX, HEIGHT_PX], [0, HEIGHT_PX]])
        rays = np.column_stack([corners_px, np.ones(4)]) @ inverse.T
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        camera_center = np.array(EYE)
        far = camera_center + rays * 2.0
        for j in range(4):
            ax.plot(
                [camera_center[0], far[j, 0]],
                [camera_center[1], far[j, 1]],
                [camera_center[2], far[j, 2]],
                color="#dc2626",
                lw=0.9,
            )
            k = (j + 1) % 4
            ax.plot(
                [far[j, 0], far[k, 0]],
                [far[j, 1], far[k, 1]],
                [far[j, 2], far[k, 2]],
                color="#dc2626",
                lw=0.9,
            )
        ax.scatter(*camera_center, color="#dc2626", s=50, marker="o")
        ax.scatter(*TARGET, color="#0f172a", s=40, marker="*")

    def draw_3d(self, mode):
        ax = self.ax3d
        ax.clear()
        self._scene_base()
        data = self.data
        visible = data["visible_world"]
        if mode == "exact":
            ax.scatter(
                visible[:, 0], visible[:, 1], visible[:, 2], color="#2563eb", s=12, depthshade=False
            )
            ax.set_title("3D 场景：可见世界点（可拖拽旋转）")
        elif mode == "ray":
            ax.scatter(
                visible[:, 0], visible[:, 1], visible[:, 2], color="#cbd5e1", s=8, depthshade=False
            )
            ray = data["report"]["ray_payload"]
            candidates = np.array(ray["points_m"])
            chosen_pixel = ray["pixel"]
            # Ray from the camera through the chosen pixel at three depths.
            direction = np.linalg.inv(K_INTRINSIC) @ np.array(
                [chosen_pixel[0], chosen_pixel[1], 1.0]
            )
            camera_ray = direction / np.linalg.norm(direction)
            start = np.array(EYE)
            end = start + camera_ray * 6.5
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color="#0891b2",
                lw=2.0,
            )
            ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                candidates[:, 2],
                color="#0891b2",
                s=70,
                marker="o",
                depthshade=False,
            )
            ax.set_title("无深度：一条射线上的三个候选（同像素）")
        else:
            ax.scatter(
                visible[:, 0],
                visible[:, 1],
                visible[:, 2],
                color="#2563eb",
                s=10,
                depthshade=False,
                label="真值点",
            )
            noisy = data["noisy_world"]
            ax.scatter(
                noisy[:, 0],
                noisy[:, 1],
                noisy[:, 2],
                color="#ea580c",
                s=8,
                depthshade=False,
                label="种子 #0 噪声点云（σ=5 cm）",
            )
            # Error lines for a sample of points.
            step = max(1, len(visible) // 14)
            for i in range(0, len(visible), step):
                ax.plot(
                    [visible[i, 0], noisy[i, 0]],
                    [visible[i, 1], noisy[i, 1]],
                    [visible[i, 2], noisy[i, 2]],
                    color="#9333ea",
                    lw=0.8,
                    alpha=0.7,
                )
            ax.legend(fontsize=8, loc="upper right")
            ax.set_title("深度噪声：真值（蓝）与还原点云（橙）及误差连线")
        ax.set_xlabel("东 / m")
        ax.set_ylabel("北 / m")
        ax.set_zlabel("高 / m")
        ax.set_box_aspect((10, 10, 4))
        ax.view_init(elev=28, azim=-58)
        self.canvas3d.draw_idle()

    def redraw(self):
        if not hasattr(self, "canvas"):
            return
        self.draw_3d(self.mode.get())
        self.draw_pixels(self.mode.get())
        self.fill_stats(self.mode.get())
        text = {
            "exact": "① 精确深度：往返一致",
            "ray": "② 无深度：同一像素只给一条射线",
            "noisy": "③ 深度噪声：误差 ÷ 射线倍率 ≈ 4 cm",
        }[self.mode.get()]
        self.status.configure(
            text=text
            + "｜3D 视图可拖拽旋转（本身也是一台针孔相机）；右侧画布=同一组投影像素。按 Esc 退出。"
        )

    def draw_pixels(self, mode):
        c = self.canvas
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 60:
            return
        pixels = np.asarray(self.data["pixels"])
        scale = min((w - 60) / WIDTH_PX, (h - 70) / HEIGHT_PX)
        ox, oy = (w - WIDTH_PX * scale) / 2, (h - HEIGHT_PX * scale) / 2

        def xy(pixel):
            return ox + pixel[0] * scale, oy + (HEIGHT_PX - pixel[1]) * scale

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
        for u, v in pixels:
            x, y = xy([u, v])
            c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill="#cbd5e1", outline="")
        if mode == "ray":
            payload = self.data["report"]["ray_payload"]
            x, y = xy(payload["pixel"])
            c.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#0891b2", width=3)
        else:
            for u, v in pixels:
                x, y = xy([u, v])
                c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill="#2563eb", outline="")

    def fill_stats(self, mode):
        report = self.data["report"]
        if mode == "exact":
            text = (
                "① 精确深度（模拟理想深度传感器）\n\n"
                "方法：pixel + depth → K⁻¹ 射线 × depth → 世界系\n"
                f"往返最大误差：{report['roundtrip_max_error_m']:.2e} m\n"
                f"可见点数：{report['visible_points']}（近裁剪面 {NEAR_PLANE_M:.1f} m 之外）\n\n"
                "深度是米制且精确时，三维点可以被完全恢复。\n"
                "这就是 RGB-D / 激光雷达所提供的：每像素一个距离。"
            )
        elif mode == "ray":
            payload = report["ray_payload"]
            lines = []
            for depth, point in zip(payload["depths_m"], payload["points_m"]):
                lines.append(
                    f"  {depth:.1f} m → 世界 ({point[0]:+.2f}, {point[1]:+.2f}, {point[2]:+.2f})"
                )
            text = (
                "② 无深度\n\n"
                "高亮像素的 3D 视图中一条青色射线：\n" + "\n".join(lines) + "\n\n"
                "三个点在同一像素（重投影一致），世界坐标不同——\n"
                "像素本身不含深度信息，只有一条射线。\n"
                "单目深度模型只能输出相对深度，正是少了这一步。"
            )
        else:
            text = (
                "③ 深度噪声（σ=5 cm，20 个种子）\n\n"
                f"点云位置误差均值 {report['noisy_mean_error_m'] * 100:.2f} cm\n"
                f"最大值 {report['noisy_max_error_m'] * 100:.2f} cm\n"
                f"误差 ÷ 射线倍率 ≈ {report['noise_estimate_mean_m'] * 100:.2f} cm"
                f"（近 {report['noise_estimate_at_near_m'] * 100:.2f} / "
                f"远 {report['noise_estimate_at_far_m'] * 100:.2f}）\n\n"
                "3D 视图：蓝=真值，橙=种子 #0 噪声点云，紫线=逐点误差。\n"
                "误差 ÷ |K⁻¹[u,v,1]| 恒定 ≈ 4 cm（= σ·√(2/π)），\n"
                "证明机制：点云误差 = 深度噪声 × 射线长度。"
            )
        self.stats.configure(text=text)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.geometry("1560x680")
    root.minsize(1280, 620)
    PinholeDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
