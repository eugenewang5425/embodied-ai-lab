"""Lesson 22 viewer: pinhole-narrated 3D view + pixel plane + meaning panel.

The 3D view shows the classic pinhole visualization: optical centre, principal
axis, image plane at focal distance, and world points whose rays cross the
image plane to become pixels. Its own perspective is also a pinhole camera.
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
# Visual zoom for the depth-noise error vectors (real mean ~12 cm now that
# the synthetic depth sensor is deliberately coarse: sigma = 15 cm).
ERROR_ZOOM = 6
IMAGE_PLANE_F_M = 1.5  # drawn image-plane distance for the 3D frustum


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
    rotation, translation = look_at(EYE, TARGET)
    fresh_pixels, fresh_depths, fresh_world = project_with_depth(points, rotation, translation)
    if not np.array_equal(fresh_pixels, pixels):
        raise ValueError("Stored pixels do not match the scene/camera")
    clouds = unproject(np.asarray(pixels), np.asarray(fresh_depths), rotation, translation)
    if not np.allclose(np.linalg.norm(clouds - fresh_world, axis=1), 0, atol=1e-12):
        raise ValueError("Round trip inconsistent")
    pole_mask = (np.abs(fresh_world[:, 0] - 4.2) < 1e-9) & (np.abs(fresh_world[:, 1] - 3.6) < 1e-9)
    return {
        "report": report,
        "points": points,
        "pixels": pixels,
        "rotation": rotation,
        "translation": translation,
        "visible_world": fresh_world,
        "noisy_world": noisy_world,
        "pole_mask": pole_mask,
    }


# Representative world points whose observer rays get drawn (index into visible).
REPRESENTATIVE = (0, 12, 24, 36, 48, 60, 72, 84, 96)


class PinholeDemo:
    """3D pinhole narrative + 2D pixel plane + meaning panel."""

    def __init__(self, root, data, parent=None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        from embodied_learning.plotting import configure_plot_font

        configure_plot_font()  # CJK-capable font for the 3D labels and titles
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
                "3D 视图：红点=相机光心，红色虚线=光轴（指向目标），青色矩形=距光心 1.5 m 处的图像平面，"
                "从光心到世界点的细线穿过图像平面变成像素\n"
                "本视图可拖拽旋转/滚轮缩放（它的透视本身又是一台针孔相机）；右侧像素平面与数字面板"
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
        three = ttk.Frame(middle)
        three.pack(side="left", fill="both", expand=True)
        self.fig = Figure(figsize=(6.4, 4.6), dpi=100)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.canvas3d = FigureCanvasTkAgg(self.fig, master=three)
        self.canvas3d.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas3d, three).pack(side="top")
        two = ttk.Frame(middle)
        two.pack(side="left", fill="both", padx=(10, 0))
        self.canvas = tk.Canvas(
            two, background="#ffffff", highlightthickness=0, width=460, height=430
        )
        self.canvas.pack(side="top", fill="both", expand=True)
        self.stats = ttk.Label(middle, width=46, anchor="nw", justify="left")
        self.stats.pack(side="left", fill="y", padx=(12, 0))
        self.status = ttk.Label(outer, text="")
        self.status.pack(anchor="w", pady=(6, 0))
        ttk.Label(
            outer,
            text=(
                "投影：u = fx·x/z + cx ；反投影：X = d · K⁻¹[u,v,1]（d 为深度）｜无深度时同一像素只约束一条射线｜"
                "点云误差 = 深度噪声 × 射线长度 |K⁻¹[u,v,1]|，图像边缘放大"
            ),
        ).pack(anchor="w")
        self.canvas.bind("<Configure>", lambda _: self.redraw())
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    @property
    def _eye_world(self):
        return np.array(EYE)

    def _image_plane_corners(self):
        """Image-plane rectangle corners in WORLD coordinates at distance f."""
        f = IMAGE_PLANE_F_M
        half_u = (WIDTH_PX / 2) * f / FOCAL_PX
        half_v = (HEIGHT_PX / 2) * f / FOCAL_PX
        corners_cam = np.array(
            [[-half_u, -half_v, f], [half_u, -half_v, f], [half_u, half_v, f], [-half_u, half_v, f]]
        )
        rotation = self.data["rotation"]
        corners_world = (rotation.T @ corners_cam.T).T
        eye = self._eye_world
        return corners_world + eye, eye

    def _frame_camera(self):
        """Optical centre, principal axis and image-plane frame in world space."""
        corners_world, eye = self._image_plane_corners()
        ax = self.ax3d
        # Image plane rectangle (pale blue) with its outline.
        xs = np.r_[corners_world[:, 0], corners_world[0, 0]]
        ys = np.r_[corners_world[:, 1], corners_world[0, 1]]
        zs = np.r_[corners_world[:, 2], corners_world[0, 2]]
        ax.plot(xs, ys, zs, color="#0ea5e9", lw=1.6)
        # Frustum edges: optical centre -> each image-plane corner.
        for corner in corners_world:
            ax.plot(
                [eye[0], corner[0]],
                [eye[1], corner[1]],
                [eye[2], corner[2]],
                color="#dc2626",
                lw=1.0,
                alpha=0.8,
            )
        ax.scatter(*eye, color="#dc2626", s=60, zorder=10)
        # Principal axis: optical centre -> target (dashed dark).
        ax.plot(
            [eye[0], TARGET[0]],
            [eye[1], TARGET[1]],
            [eye[2], TARGET[2]],
            color="#0f172a",
            ls="--",
            lw=1.6,
        )
        ax.text(
            *(np.array(eye) + np.array(TARGET)) / 2 + [0, 0, 0.15],
            "光轴（视线）",
            fontsize=9,
            color="#0f172a",
        )
        ax.text(
            *(corners_world[0] + [0, 0, 0.12]),
            "图像平面 640×480 (1.5 m)",
            fontsize=8,
            color="#0ea5e9",
        )
        ax.scatter(*TARGET, color="#0f172a", s=50, marker="*")
        return corners_world, eye

    def draw_3d(self, mode):
        ax = self.ax3d
        ax.clear()
        visible = self.data["visible_world"]
        # Ground grid + pole (scene).
        xs = np.linspace(0.75, 5.25, 10)
        for value in xs:
            ax.plot([value, value], [xs[0], xs[-1]], [0, 0], color="#cbd5e1", lw=0.6)
            ax.plot([xs[0], xs[-1]], [value, value], [0, 0], color="#cbd5e1", lw=0.6)
        ax.plot([4.2, 4.2], [3.6, 3.6], [0, 2.0], color="#0f172a", lw=2.5)
        corners_world, eye = self._frame_camera()
        if mode == "exact":
            ax.set_title("世界点（蓝）的观测射线穿过图像平面 → 变成像素")
            ax.scatter(
                visible[:, 0],
                visible[:, 1],
                visible[:, 2],
                color="#2563eb",
                s=12,
                depthshade=False,
                label="可见世界点（99）",
            )
            self._draw_rays(visible, corners_world, eye)
            ax.legend(fontsize=8, loc="upper right")
        elif mode == "ray":
            ax.set_title("无深度：同一像素在图像平面上，深度候选点全在一条射线上")
            ax.scatter(
                visible[:, 0], visible[:, 1], visible[:, 2], color="#cbd5e1", s=8, depthshade=False
            )
            ray = self.data["report"]["ray_payload"]
            candidates = np.array(ray["points_m"])
            chosen_pixel = ray["pixel"]
            # Ray from the pinhole through the chosen pixel, rotated to WORLD axes.
            direction_cam = np.linalg.inv(K_INTRINSIC) @ np.array(
                [chosen_pixel[0], chosen_pixel[1], 1.0]
            )
            direction_world = self.data["rotation"].T @ direction_cam
            direction_world = direction_world / np.linalg.norm(direction_world)
            end = eye + direction_world * 7.0
            ax.plot(
                [eye[0], end[0]],
                [eye[1], end[1]],
                [eye[2], end[2]],
                color="#0891b2",
                lw=2.4,
                label="该像素对应的射线",
            )
            # The same pixel marked on the image plane.
            f = IMAGE_PLANE_F_M
            hit_cam = np.array(
                [
                    (chosen_pixel[0] - CX_PX) * f / FOCAL_PX,
                    (chosen_pixel[1] - CY_PX) * f / FOCAL_PX,
                    f,
                ]
            )
            hit_world = (self.data["rotation"].T @ hit_cam) + eye
            ax.scatter(
                *hit_world,
                color="#0891b2",
                s=90,
                marker="X",
                zorder=11,
                label="该像素在图像平面上的位置",
            )
            ax.scatter(
                candidates[:, 0],
                candidates[:, 1],
                candidates[:, 2],
                color="#0891b2",
                s=70,
                depthshade=False,
                label="深度 2/4/6 m 的三个候选（共线）",
            )
            ax.legend(fontsize=8, loc="upper left")
        else:
            ax.set_title(
                f"深度噪声：误差箭头放大 ×{ERROR_ZOOM}"
                f"（真实均值 {self.data['report']['noisy_mean_error_m'] * 100:.1f} cm）"
            )
            noisy = self.data["noisy_world"]
            error_vectors = (noisy - visible) * ERROR_ZOOM
            drawn = visible + error_vectors
            ax.scatter(
                drawn[:, 0],
                drawn[:, 1],
                drawn[:, 2],
                color="#ea580c",
                s=10,
                depthshade=False,
                label=f"还原点云（误差放大 ×{ERROR_ZOOM} 后的位置）",
            )
            ax.quiver(
                visible[:, 0],
                visible[:, 1],
                visible[:, 2],
                error_vectors[:, 0],
                error_vectors[:, 1],
                error_vectors[:, 2],
                color="#9333ea",
                linewidth=2.2,
                arrow_length_ratio=0.18,
                label="误差向量（方向=该点观测射线）",
            )
            ax.scatter(
                visible[:, 0],
                visible[:, 1],
                visible[:, 2],
                color="#2563eb",
                s=7,
                depthshade=False,
                label="真值点（误差起点）",
            )
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("东 / m")
        ax.set_ylabel("北 / m")
        ax.set_zlabel("高 / m")
        ax.set_xlim(0, 6.5)
        ax.set_ylim(0, 6.5)
        ax.set_zlim(-0.5, 3.5)
        ax.set_box_aspect((6.5, 6.5, 3.8))
        ax.view_init(elev=26, azim=-58)
        self.canvas3d.draw_idle()

    def _draw_rays(self, visible, corners_world, eye):
        """Thin lines optical centre -> world point, stopping at the image plane."""
        f = IMAGE_PLANE_F_M
        rotation = self.data["rotation"]
        count = 0
        for index in REPRESENTATIVE:
            if index >= len(visible):
                continue
            point = visible[index]
            camera_point = rotation @ (point - eye)
            if camera_point[2] <= f:
                continue
            hit_cam = camera_point * (f / camera_point[2])
            hit_world = (rotation.T @ hit_cam) + eye
            ax = self.ax3d
            ax.plot(
                [eye[0], point[0]],
                [eye[1], point[1]],
                [eye[2], point[2]],
                color="#94a3b8",
                lw=0.7,
                alpha=0.65,
            )
            ax.scatter(*hit_world, color="#f59e0b", s=26, zorder=11, depthshade=False)
            count += 1
            if count >= 9:
                break
        # One highlighted example arrow: eye -> image-plane pixel (amber dashed).
        sample = visible[min(30, len(visible) - 1)]
        camera_point = rotation @ (sample - eye)
        hit_cam = camera_point * (f / camera_point[2])
        hit_world = (rotation.T @ hit_cam) + eye
        ax.plot(
            [eye[0], hit_world[0]],
            [eye[1], hit_world[1]],
            [eye[2], hit_world[2]],
            color="#f59e0b",
            ls="--",
            lw=1.8,
        )
        ax.text(*(hit_world + [0, 0, 0.1]), "此点变成的像素", fontsize=8, color="#f59e0b")

    def redraw(self):
        if not hasattr(self, "canvas"):
            return
        self.draw_3d(self.mode.get())
        self.draw_pixels(self.mode.get())
        self.fill_stats(self.mode.get())
        text = {
            "exact": "① 这一幕在证明：像素 + 米制深度 = 三维点（图像平面上的每个点都能还原成世界点）",
            "ray": "② 这一幕在说明：单目像素没有尺度——同一像素的任何深度都在同一条射线上",
            "noisy": "③ 这一幕在量化：深度噪声如何传播成点云误差，以及它随图像边缘放大的规律",
        }[self.mode.get()]
        self.status.configure(
            text=text + "｜3D 视图可拖拽旋转（本身也是一台针孔相机）。按 Esc 退出。"
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
        pole_mask = self.data["pole_mask"]
        for index, (u, v) in enumerate(pixels):
            x, y = xy([u, v])
            fill = "#ea580c" if pole_mask[index] else "#cbd5e1"
            c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill=fill, outline="")
        if pole_mask.any():
            c.create_text(
                ox + 8,
                oy + 32,
                text="橙色=竖直杆：底部 2 点在画面内；杆顶超出图像上边界（视场）",
                anchor="nw",
                font=("Microsoft YaHei", 8),
                fill="#ea580c",
            )
        if mode == "ray":
            payload = self.data["report"]["ray_payload"]
            x, y = xy(payload["pixel"])
            c.create_oval(x - 6, y - 6, x + 6, y + 6, outline="#0891b2", width=3)
        else:
            for index, (u, v) in enumerate(pixels):
                x, y = xy([u, v])
                fill = "#ea580c" if pole_mask[index] else "#2563eb"
                c.create_oval(x - 2.0, y - 2.0, x + 2.0, y + 2.0, fill=fill, outline="")

    def fill_stats(self, mode):
        report = self.data["report"]
        if mode == "exact":
            text = (
                "① 这一幕（目的）\n"
                "三维世界点 → 穿过图像平面 → 像素；\n"
                "像素 + 米制深度 → 反投影回同一三维点。\n\n"
                "统计特征与意义：\n"
                f"  往返最大误差 {report['roundtrip_max_error_m']:.2e} m\n"
                f"  （99 个可见点，近裁剪面 {NEAR_PLANE_M:.1f} m）\n"
                "  1e-15 m 量级 = 浮点舍入，不是“定位误差”——\n"
                "  它证明几何关系可逆：像素+深度=三维点。\n\n"
                "应用意义：RGB-D/激光雷达给每像素一个距离，\n"
                "这就是它们能直接产出点云的原理。"
            )
        elif mode == "ray":
            payload = report["ray_payload"]
            lines = []
            for depth, point in zip(payload["depths_m"], payload["points_m"]):
                lines.append(
                    f"  {depth:.1f} m → 世界 ({point[0]:+.2f}, {point[1]:+.2f}, {point[2]:+.2f})"
                )
            text = (
                "② 这一幕（目的）\n"
                "同一像素对应一条射线；三个深度候选都在这条射线上，\n"
                "而且重投影回图像是同一点（像素不随深度变）。\n\n"
                "统计特征与意义：\n" + "\n".join(lines) + "\n\n"
                "特征：三个候选的世界坐标不同但共线；\n"
                "意义：单目图像不含距离——所以单目深度模型\n"
                "（如 Depth Anything）只能输出相对深度，\n"
                "米制尺度必须由几何/标定/先验补上。"
            )
        else:
            text = (
                "③ 这一幕（目的）\n"
                "给深度加 σ=5 cm 噪声，看它如何变成点云误差。\n\n"
                "统计特征与意义：\n"
                f"  点云误差均值 {report['noisy_mean_error_m'] * 100:.2f} cm，"
                f"最大 {report['noisy_max_error_m'] * 100:.2f} cm\n"
                f"  误差 ÷ 射线倍率 ≈ {report['noise_estimate_mean_m'] * 100:.2f} cm"
                f"（近 {report['noise_estimate_at_near_m'] * 100:.2f} / "
                f"远 {report['noise_estimate_at_far_m'] * 100:.2f}）\n\n"
                "特征：误差 ∝ |K⁻¹[u,v,1]|（图像上离主点越远越大），\n"
                "除回倍率后 ≈ σ·√(2/π)=3.99 cm——机制自洽。\n"
                "意义：同样的传感器噪声，在图像边缘（近处大视角）\n"
                "放得更大；精度不由标称值单独决定。"
            )
        self.stats.configure(text=text)


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(DEFAULT_RESULTS))
    args = parser.parse_args()
    data = load_replays(args.results)
    root = tk.Tk()
    root.geometry("1580x700")
    root.minsize(1300, 640)
    PinholeDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
