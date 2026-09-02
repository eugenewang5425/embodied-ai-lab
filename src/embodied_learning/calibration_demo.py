"""Lesson 16: inspect measurements, fit a scale, then replay held-out routes."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_learning.encoder_calibration import correct_right_encoder, fit_right_correction
from embodied_learning.experiments.encoder_calibration import (
    EXPERIMENT,
    METHODS,
    digest,
    fit_methods,
)
from embodied_learning.experiments.mobile_frames import GEOMETRY
from embodied_learning.odometry import estimate_poses
from embodied_learning.odometry_demo import OdometryDemo
from embodied_learning.odometry_demo import load_replays as load_odometry_replays

TEACHING_LABELS = ("不修正（旧 +2%）", "用准确尺子标定", "标定尺子偏大1%")
WINDOW_TITLE = "第十六课 · 看测量 → 求系数 → 换路线验证"


def measurement_columns(rows, reference_multiplier):
    """Display the same inputs used by the fitter, not held-out pose errors."""
    if reference_multiplier not in (1.0, 1.01):
        raise ValueError("Expected the recorded exact or +1% reference instrument")
    angles = np.array([row["measured_right_angle_rad"] for row in rows])
    distances = np.array([row["external_distance_m"] for row in rows]) * reference_multiplier
    return np.column_stack([angles, GEOMETRY.radius_m * angles, distances])


def load_replays(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    if digest(directory / "calibration.json") != report.get("calibration_sha256"):
        raise ValueError("Calibration checksum mismatch")
    if (
        report.get("input_right_scale") != 1.02
        or calibration.get("schema_version") != 1
        or calibration.get("reference_bias_variant_multiplier") != 1.01
    ):
        raise ValueError("Incompatible calibration protocol")
    factors = fit_methods(calibration["runs"])
    if calibration.get("fitted_corrections") != dict(zip((k for k, _, _ in METHODS), factors)):
        raise ValueError("Fitted corrections do not match calibration measurements")
    variants = tuple(
        (key, label, report["input_right_scale"] * factor, color)
        for (key, label, color), factor in zip(METHODS, factors)
    )
    replays = load_odometry_replays(directory, variants=variants, experiment=EXPERIMENT)
    for replay in replays:
        arrays = replay.arrays
        for (key, _, _), factor, metadata in zip(METHODS, factors, replay.metadata["estimates"]):
            expected = correct_right_encoder(arrays["raw_encoder_angles_rad"], factor)
            if metadata.get("correction_factor") != factor or not np.array_equal(
                arrays[f"{key}_encoder_angles_rad"], expected
            ):
                raise ValueError("Corrected readings do not match fitted factor")
            if not np.array_equal(arrays[f"{key}_poses"], estimate_poses(expected)):
                raise ValueError("Saved pose does not match corrected encoder integration")
    return replays, variants


class CalibrationDemo:
    """One Tk root, three teaching pages; numerical experiment files stay read-only."""

    def __init__(self, root, directory, speed=0.25):
        import tkinter as tk
        from tkinter import ttk

        replays, variants = load_replays(directory)
        self.rows = json.loads((Path(directory) / "calibration.json").read_text(encoding="utf-8"))[
            "runs"
        ]
        self.root, self.fitted = root, None
        self.reference_multiplier = tk.DoubleVar(master=root, value=1.0)
        self.measurement_detail = tk.StringVar(master=root)
        self.formula = tk.StringVar(master=root)
        self.fit_status = tk.StringVar(master=root)
        root.title(WINDOW_TITLE)
        root.geometry("1100x780")
        root.minsize(1040, 760)
        root.option_add("*Font", "{Microsoft YaHei} 10")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)
        self.pages = [ttk.Frame(self.notebook) for _ in range(3)]
        for page, label in zip(self.pages, ("① 对照测量", "② 求修正系数", "③ 换路线验证")):
            self.notebook.add(page, text=label)

        first = ttk.Frame(self.pages[0], padding=22)
        first.pack(fill="both", expand=True)
        ttk.Label(
            first, text="同一段路，两种测量为什么不一致？", font=("Microsoft YaHei", 19, "bold")
        ).pack(anchor="w")
        ttk.Label(
            first,
            text="这是四段独立的标定记录，还不是后面的 12 秒直行或 24 秒方形验证。\n"
            "尺子测车走了多远；右轮编码器报转角，用已知半径 0.05 m 换算成距离。",
            justify="left",
        ).pack(anchor="w", pady=(12, 16))
        instruments = ttk.LabelFrame(
            first, text="先选标定用的外部尺子（不改变原始编码器数据）", padding=12
        )
        instruments.pack(fill="x")
        for label, multiplier in (("准确尺子", 1.0), ("标定用的尺子偏大 1%", 1.01)):
            ttk.Radiobutton(
                instruments,
                text=label,
                variable=self.reference_multiplier,
                value=multiplier,
                command=self.change_instrument,
            ).pack(side="left", padx=(0, 30))
        self.table = ttk.Treeview(
            first,
            columns=("run", "angle", "encoder", "reference", "difference"),
            show="headings",
            height=4,
            selectmode="browse",
        )
        for key, label, width in (
            ("run", "独立标定段", 130),
            ("angle", "右轮报告 / rad", 170),
            ("encoder", "编码器换算 / m", 180),
            ("reference", "外部尺子读数 / m", 180),
            ("difference", "两者之差 / mm", 175),
        ):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, anchor="center", stretch=True)
        self.table.pack(fill="x", pady=18)
        self.table.bind("<<TreeviewSelect>>", self.describe_measurement)
        ttk.Label(
            first,
            textvariable=self.measurement_detail,
            font=("Microsoft YaHei", 15),
            justify="left",
        ).pack(anchor="w", pady=12)
        ttk.Label(
            first,
            text="负数表示后退，不是错误。右轮原始读数保持不变；换尺子只改参考距离。\n"
            "本课外部测距由仿真生成，不是真实硬件测量；暂不加入噪声或打滑。",
            justify="left",
        ).pack(anchor="w", pady=12)
        ttk.Button(
            first, text="下一步：怎样把读数校准？ →", command=lambda: self.notebook.select(1)
        ).pack(anchor="w", pady=16)

        second = ttk.Frame(self.pages[1], padding=22)
        second.pack(fill="both", expand=True)
        ttk.Label(
            second, text="用测量求系数，不直接填入偏差答案", font=("Microsoft YaHei", 19, "bold")
        ).pack(anchor="w")
        ttk.Label(
            second,
            text="目标：让 c × 编码器换算距离，尽量接近外部尺子读数。\n"
            "只有一个固定比例 c；后面的验证路线不参与求解。",
            justify="left",
        ).pack(anchor="w", pady=(12, 20))
        ttk.Button(second, text="用这四段数据计算 c", command=self.calculate).pack(
            anchor="w", pady=8
        )
        ttk.Label(
            second, textvariable=self.formula, font=("Microsoft YaHei", 14), justify="left"
        ).pack(anchor="w", pady=18)
        ttk.Label(second, textvariable=self.fit_status, justify="left").pack(anchor="w", pady=12)
        self.validation_button = ttk.Button(
            second, text="带着这个系数，换路线验证 →", command=self.begin_validation
        )
        self.validation_button.pack(anchor="w", pady=12)
        ttk.Button(second, text="← 返回换一把尺子", command=lambda: self.notebook.select(0)).pack(
            anchor="w"
        )

        teaching_variants = tuple(
            (key, label, scale, color)
            for (key, _, scale, color), label in zip(variants, TEACHING_LABELS)
        )
        self.replay = OdometryDemo(
            root, replays, speed, variants=teaching_variants, calibration=True, parent=self.pages[2]
        )
        self.notebook.bind("<<NotebookTabChanged>>", self.page_changed)
        root.bind("<space>", self.toggle_playback)
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.change_instrument()

    def change_instrument(self):
        self.fitted = None
        self.columns = measurement_columns(self.rows, self.reference_multiplier.get())
        self.table.delete(*self.table.get_children())
        for i, (angle, encoder, reference) in enumerate(self.columns):
            self.table.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    f"第{i + 1}段 · {'前进' if angle > 0 else '后退'}",
                    f"{angle:+.4f}",
                    f"{encoder:+.4f}",
                    f"{reference:+.4f}",
                    f"{(encoder - reference) * 1000:+.2f}",
                ),
            )
        self.table.selection_set("0")
        self.describe_measurement()
        self.formula.set("尚未计算。\n先观察第①页两列距离，再点击上面的计算按钮。")
        self.fit_status.set("换尺子后必须重新计算；不会继续使用上一把尺子的系数。")
        self.validation_button.state(["disabled"])
        self.notebook.select(0)
        self.notebook.tab(2, state="disabled")
        self.replay.clock.seek(self.replay.clock.index)
        self.replay.redraw()

    def describe_measurement(self, _=None):
        selected = self.table.selection()
        if not selected:
            return
        i = int(selected[0])
        angle, encoder, reference = self.columns[i]
        self.measurement_detail.set(
            f"第{i + 1}段：尺子说 {reference:+.4f} m，编码器算出 {encoder:+.4f} m\n\n"
            f"编码器换算：0.05 m × {angle:+.4f} rad = {encoder:+.4f} m\n"
            f"两者之差：{(encoder - reference) * 1000:+.2f} mm"
        )

    def calculate(self):
        angles, x, d = self.columns.T
        factor = fit_right_correction(angles, d, GEOMETRY.radius_m)
        method_index = 1 if self.reference_multiplier.get() == 1.0 else 2
        expected = self.replay.replays[0].metadata["estimates"][method_index]["correction_factor"]
        if not np.isclose(factor, expected, rtol=0, atol=1e-14):
            raise ValueError("Visible calibration inputs do not match the frozen validation result")
        self.fitted = factor
        self.replay.variant.set(TEACHING_LABELS[method_index])
        self.replay.select_variant()
        self.formula.set(
            f"先看第1段：c = {d[0]:.4f} ÷ {x[0]:.4f} ≈ {d[0] / x[0]:.6f}\n\n"
            f"四段一起求：c = Σ(x × d) ÷ Σ(x²)\n"
            f"                  = {x @ d:.8f} ÷ {x @ x:.8f} = {factor:.9f}\n\n"
            "x 是编码器换算距离，d 是尺子读数；这就是一元最小二乘。\n"
            f"以第1段检查：{factor:.6f} × {x[0]:.4f} ≈ {factor * x[0]:.4f} m"
        )
        self.fit_status.set(
            "准确基准：得到的 c 将用于另一条路线，不再按验证结果调参。\n"
            "改变的是送入里程计的右轮增量，不是原始记录或电机动作。"
            if method_index == 1
            else "注意：这个 c 对准的是偏大的尺子，不是真实距离。\n"
            "多用几段相同比例偏大的数据，不能自动消除尺子的系统误差。"
        )
        self.notebook.tab(2, state="normal")
        self.validation_button.state(["!disabled"])

    def begin_validation(self):
        if self.fitted is None:
            return
        method_index = 1 if self.reference_multiplier.get() == 1.0 else 2
        self.replay.variant.set(TEACHING_LABELS[method_index])
        self.replay.select_variant()  # Retain the timestamp when comparing two instruments.
        self.notebook.select(2)

    def page_changed(self, _=None):
        if self.notebook.index(self.notebook.select()) != 2:
            self.replay.clock.seek(self.replay.clock.index)
            self.replay.redraw()

    def toggle_playback(self, _=None):
        if self.notebook.index(self.notebook.select()) == 2:
            self.replay.toggle()

    def close(self):
        self.replay.close()


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/encoder_calibration_2026-09-03_v2")
    )
    parser.add_argument("--speed", type=float, choices=[0.1, 0.25, 0.5, 1.0], default=0.25)
    args = parser.parse_args()
    root = tk.Tk()
    CalibrationDemo(root, args.results, args.speed)
    root.mainloop()


if __name__ == "__main__":
    main()
