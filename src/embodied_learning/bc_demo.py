"""Lesson 28 viewer: what a cloned (state-only) policy can and cannot do.

Three static modes share one lesson-28 recording (npz + summary):
1. end-effector paths, expert (feedforward+PD) versus the BC torque driven
   directly, for the best BC episode and a failure episode;
2. the data-scaling curves: success rate on the training paths and on fresh
   same-distribution paths, per data seed and averaged;
3. open-loop MSE versus closed-loop success, plus the failure trajectory.
Layout, Esc handling and the meaning panel follow the lesson-27 demo:
every mode calls fig.clear() first, all quoted numbers come from summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embodied_learning.arm_path import segment_distance
from embodied_learning.experiments.bc_imitation import EXPERIMENT

DEFAULT_RESULTS = "results/bc_imitation_2026-09-05"
BASE_NPZ_KEYS = {
    "expert_train_x",
    "expert_train_y",
    "expert_gen_x",
    "expert_gen_y",
    "sample_sizes",
    "success_train",
    "success_gen",
    "max_cross_train_mm",
    "max_cross_gen_mm",
    "final_train_mse",
    "full_train_mse",
    "full_gen_mse",
    "loss_curves",
}
ROLE_ARRAYS = (
    "states",
    "points",
    "desired_points",
    "torques_nm",
    "requested_torques_nm",
)


def expected_npz_keys(featured_count):
    keys = set(BASE_NPZ_KEYS)
    for index in range(featured_count):
        for role in ("bc", "expert"):
            keys.update(f"case{index}_{role}_{name}" for name in ROLE_ARRAYS)
    return keys


def featured_max_cross_mm(points, desired_points, target, dt):
    tip = points[:, -1]
    times = np.arange(len(points)) * dt
    movement = times <= 4.0 + 1e-10  # the 4 s movement window of the 4s+3s protocol
    distance = segment_distance(tip, desired_points[0], target)
    return float(distance[movement].max() * 1000.0)


def load_replays(directory):
    """Validate a lesson-28 recording; returns the data dict for the demo."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT or report.get("schema_version") != 1:
        raise ValueError("Incompatible lesson-28 recording")
    network = report.get("network") or report.get("expert_verification") and {} or {}
    network = report.get("network", {})
    hidden = network.get("hidden")
    if hidden != [64, 64] or network.get("activation") != "relu" or network.get("output") != 2:
        raise ValueError("Unexpected network contract")
    sizes = report["scaling"]["sample_sizes_executed"]
    parameter_count = network.get("parameter_count")
    expected_parameters = sum(
        (4 if i == 0 else hidden[i - 1]) * units + units for i, units in enumerate([*hidden, 2])
    )
    if parameter_count != expected_parameters:
        raise ValueError("Network parameter count disagrees with the hidden sizes")
    path = directory / "trajectories.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != report.get("trajectories_sha256"):
        raise ValueError("Trajectory checksum mismatch")
    with np.load(path, allow_pickle=False) as npz:
        if set(npz.files) != expected_npz_keys(len(report["featured_cases"])):
            raise ValueError("Unexpected archive arrays")
        data = {key: npz[key].copy() for key in npz.files}
    executed = [
        size if size else report["steps_per_path"] for size in data["sample_sizes"].tolist()
    ]
    if executed != sizes:
        raise ValueError("Archive sample sizes disagree with the summary")
    success_train, success_gen = data["success_train"], data["success_gen"]
    if success_train.shape[0] != len(sizes) or success_train.shape != success_gen.shape:
        raise ValueError("Archive success arrays disagree with the scan size")
    episodes = report["expert_verification"]["train"]["executed"]
    if success_train.shape[2] != episodes:
        raise ValueError("Archive episode count disagrees with the expert verification")
    seeds = success_train.shape[1]
    if len(report["mse_vs_success"]) != len(sizes) * seeds:
        raise ValueError("MSE table size disagrees with the archive")
    if not np.allclose(
        success_train.mean(axis=(1, 2)), report["scaling"]["train_success_mean"], rtol=0, atol=1e-12
    ):
        raise ValueError("Train success mean disagrees with the archive")
    if not np.allclose(
        success_gen.mean(axis=(1, 2)), report["scaling"]["gen_success_mean"], rtol=0, atol=1e-12
    ):
        raise ValueError("Generalization success mean disagrees with the archive")
    for name, array in (
        ("full_train_mse_mean", data["full_train_mse"].mean(axis=1)),
        ("full_gen_mse_mean", data["full_gen_mse"].mean(axis=1)),
        ("final_train_mse_mean", data["final_train_mse"].mean(axis=1)),
    ):
        if not np.allclose(array, report["scaling"][name], rtol=0, atol=1e-12):
            raise ValueError(f"{name} disagrees with the archive")
    records = {
        "train": {r["id"]: r for r in report["train_records"] if r["status"] == "executed"},
        "gen": {r["id"]: r for r in report["generalization_records"] if r["status"] == "executed"},
    }
    for index, case in enumerate(report["featured_cases"]):
        bc_points = data[f"case{index}_bc_points"]
        bc_desired = data[f"case{index}_bc_desired_points"]
        if bc_points.ndim != 3 or bc_points.shape[1:] != (3, 2) or bc_desired.shape[1] != 2:
            raise ValueError(f"Featured case {index} array shapes disagree with the contract")
        target = np.asarray(case["target_m"])
        recomputed = featured_max_cross_mm(bc_points, bc_desired, target, report["dt_s"])
        stored = case["bc_max_cross_track_mm"]
        consistent = np.isfinite(recomputed) if stored is None else abs(recomputed - stored) <= 1e-6
        if not consistent:
            raise ValueError(f"Featured case {index} max cross-track disagrees with its arrays")
        expert_points = data[f"case{index}_expert_points"]
        expert_desired = data[f"case{index}_expert_desired_points"]
        expert_cross = featured_max_cross_mm(expert_points, expert_desired, target, report["dt_s"])
        record = records[case["set"]].get(case["id"])
        if record is None or abs(expert_cross - record["max_cross_track_mm"]) > 1e-6:
            raise ValueError(f"Featured case {index} expert arrays disagree with the records")
    return {"report": report, **data}


class BcDemo:
    """Expert-versus-clone narrative: paths, data scaling, MSE vs success."""

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
            text="第二十八课 · 行为克隆：第十三课专家的 (状态, 力矩) 数据能替代动力学模型吗",
            font=("Microsoft YaHei", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "纯 numpy MLP（2×64 ReLU）模仿“前馈+PD”专家：输入只有 [q1,q2,dq1,dq2]（无知觉版本，\n"
                "不看参考），输出 2 个力矩直接驱动电机（无 PD 兜底）。三个模式共用同一条正式记录："
                "① 专家 vs BC 末端路径 ② 数据量–成功率 ③ MSE–成功率与失败轨迹；全部为静态图"
            ),
        ).pack(anchor="w", pady=(2, 6))
        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        self.mode = tk.StringVar(value="paths")
        for label, key in (
            ("① 专家 vs BC 末端路径（成功对照 + 失败案例）", "paths"),
            ("② 数据量–成功率（训练内 / 泛化）", "scaling"),
            ("③ MSE–成功率散点 + 失败案例轨迹", "mechanism"),
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
                "专家读规划参考（前馈+PD 都看参考），BC 只看当前状态——这一口径差如实呈现；"
                "成功率是 25 条路径上的有限样本计数，不是总体概率"
            ),
        ).pack(anchor="w")
        root.bind("<Escape>", lambda _: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.redraw()

    def close(self):
        self.root.destroy()

    # ------------------------------------------------------------------ modes
    def draw_paths(self):
        """① expert vs BC end-effector paths: best episode and a failure."""
        self.fig.clear()
        ax_best, ax_fail = self.fig.subplots(1, 2)
        cases = self.report["featured_cases"]
        for ax, index, title in ((ax_best, 0, "最佳 BC 回合"), (ax_fail, 1, "失败案例")):
            if index >= len(cases):
                ax.text(0.5, 0.5, "无失败案例", ha="center", va="center", transform=ax.transAxes)
                ax.set(title=title)
                continue
            case = cases[index]
            bc = self.data[f"case{index}_bc_points"][:, -1] * 100.0  # tip site of (N, 3, 2)
            expert = self.data[f"case{index}_expert_points"][:, -1] * 100.0
            desired = self.data[f"case{index}_bc_desired_points"] * 100.0
            ax.plot(desired[[0, -1], 0], desired[[0, -1], 1], "k:", linewidth=1, label="规定直线")
            ax.plot(expert[:, 0], expert[:, 1], "--", color="gray", label="专家（前馈+PD）")
            color = "#0f766e" if case["kind"] == "best" else "#b91c1c"
            outcome = "通过" if case["bc_path_success"] else "未通过"
            ax.plot(
                bc[:, 0],
                bc[:, 1],
                color=color,
                label=f"BC（{outcome}，{case['sample_size']} 步/路径）",
            )
            ax.set(
                xlabel="X（cm）",
                ylabel="Y（cm）",
                aspect="equal",
                title=f"{title}：{case['id']}（{'训练内' if case['set'] == 'train' else '泛化'}路径）",
            )
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.2)

    def draw_scaling(self):
        """② data-scaling success curves, in-sample and generalization."""
        self.fig.clear()
        ax = self.fig.subplots(1, 1)
        sizes = self.report["scaling"]["sample_sizes_executed"]
        positions = np.arange(len(sizes))
        success_train = self.data["success_train"]
        success_gen = self.data["success_gen"]
        for success, label, color in (
            (success_train, "训练内（同批路径）", "#0f766e"),
            (success_gen, "泛化（新种子路径）", "#b91c1c"),
        ):
            for seed_index in range(success.shape[1]):
                ax.scatter(
                    positions,
                    success[:, seed_index].mean(axis=1) * 100,
                    s=18,
                    color=color,
                    alpha=0.45,
                )
            mean = success.mean(axis=(1, 2)) * 100
            ax.plot(positions, mean, "o-", color=color, label=label)
            for x, value in zip(positions, mean, strict=True):
                ax.annotate(
                    f"{value:.0f}",
                    (x, value),
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
        ax.set(
            xticks=positions,
            xticklabels=[str(size) for size in sizes],
            xlabel="每条路径采样的 (状态, 力矩) 步数",
            ylabel="路径验收通过率（%）",
            ylim=(-3, 108),
            title="数据量–成功率：小点是单个数据种子（均值标注）",
        )
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

    def draw_mechanism(self):
        """③ open-loop MSE vs closed-loop success, plus a failure trajectory."""
        self.fig.clear()
        ax_scatter, ax_fail = self.fig.subplots(1, 2)
        rows = self.report["mse_vs_success"]
        sizes = self.report["scaling"]["sample_sizes_executed"]
        for field, success_field, label, color in (
            ("full_train_mse", "train_success_fraction", "训练内", "#0f766e"),
            ("full_gen_mse", "gen_success_fraction", "泛化", "#b91c1c"),
        ):
            mse = [row[field] for row in rows]
            success = [row[success_field] * 100 for row in rows]
            ax_scatter.scatter(mse, success, color=color, s=26, label=label)
        means = self.data["full_train_mse"].mean(axis=1)
        train_mean = self.report["scaling"]["train_success_mean"]
        for x, y, size in zip(means, train_mean, sizes, strict=True):
            ax_scatter.annotate(
                str(size), (x, y * 100), fontsize=8, xytext=(4, 3), textcoords="offset points"
            )
        ax_scatter.set(
            xlabel="专家状态分布上的全量 MSE（开环）",
            ylabel="闭环成功率（%）",
            title="开环 MSE 不等于闭环成功率（复合误差）",
            xscale="log",
            ylim=(-3, 105),
        )
        ax_scatter.legend(fontsize=8, loc="upper left")
        ax_scatter.grid(alpha=0.2)
        failures = [case for case in self.report["featured_cases"] if case["kind"] != "best"]
        if not failures:
            ax_fail.text(
                0.5, 0.5, "无失败案例", ha="center", va="center", transform=ax_fail.transAxes
            )
            ax_fail.set(title="失败案例轨迹")
            return
        index = self.report["featured_cases"].index(failures[0])
        case = failures[0]
        bc_points = self.data[f"case{index}_bc_points"]
        bc_desired = self.data[f"case{index}_bc_desired_points"]
        expert_points = self.data[f"case{index}_expert_points"]
        expert_desired = self.data[f"case{index}_expert_desired_points"]
        dt = self.report["dt_s"]
        bc_times = np.arange(len(bc_points)) * dt
        expert_times = np.arange(len(expert_points)) * dt
        ax_fail.plot(
            bc_times,
            np.linalg.norm(bc_points[:, -1] - bc_desired, axis=1) * 1000,
            color="#b91c1c",
            label="BC（失败案例）",
        )
        ax_fail.plot(
            expert_times,
            np.linalg.norm(expert_points[:, -1] - expert_desired, axis=1) * 1000,
            "--",
            color="gray",
            label="专家（同一路径）",
        )
        ax_fail.axhline(2.0, color="gray", linestyle=":")
        ax_fail.set(
            xlabel="时间（s）",
            ylabel="距同时刻规定点（mm）",
            title=f"失败案例 {case['id']}（{'训练内' if case['set'] == 'train' else '泛化'}）："
            "小误差被闭环放大",
        )
        ax_fail.legend(fontsize=8, loc="upper left")
        ax_fail.grid(alpha=0.2)

    # ------------------------------------------------------------------ panel
    def fill_stats(self, mode):
        report = self.report
        scaling = report["scaling"]
        sizes = scaling["sample_sizes_executed"]
        episodes = report["expert_verification"]["train"]["executed"]
        gen_episodes = report["expert_verification"]["generalization"]["executed"]
        seeds = self.data["success_train"].shape[1]
        total = episodes * seeds
        if mode == "paths":
            lines = ["① 这一幕在对比：同一条路径，专家 vs 克隆\n"]
            for index, case in enumerate(report["featured_cases"]):
                lines.append(
                    f"  [{case['kind']}] {case['id']}"
                    f"（{'训练内' if case['set'] == 'train' else '泛化'}，"
                    f"{case['sample_size']} 步/路径，种子 {case['data_seed_index']}）"
                )
                bc_cross = case["bc_max_cross_track_mm"]
                cross_text = f"{bc_cross:.1f} mm" if bc_cross is not None else "发散"
                lines.append(
                    f"    BC 移动期最大偏离 {cross_text}，"
                    f"{'通过' if case['bc_path_success'] else '未通过'}验收"
                )
                record = next(
                    r
                    for r in (
                        report["train_records"]
                        if case["set"] == "train"
                        else report["generalization_records"]
                    )
                    if r["id"] == case["id"] and r["status"] == "executed"
                )
                lines.append(
                    f"    专家同一画面：最大偏离 {record['max_cross_track_mm']:.3f} mm，"
                    f"{'通过' if record['path_success'] else '未通过'}"
                )
            lines.append("\n  BC 只看 [q1,q2,dq1,dq2]，不看参考与目标；")
            lines.append("  专家读完整规划参考。口径差如实呈现。")
        elif mode == "scaling":
            lines = ["② 这一幕在扫描：数据量换成成功率\n"]
            lines.append(
                f"  专家复验：训练内 {episodes}/{episodes}，"
                f"泛化 {gen_episodes}/{gen_episodes}（第 13 课口径）"
            )
            lines.append(f"  每档 = {seeds} 个数据种子 × {episodes} 条路径")
            for index, size in enumerate(sizes):
                train_count = int(self.data["success_train"][index].sum())
                gen_count = int(self.data["success_gen"][index].sum())
                lines.append(
                    f"  {size:>4} 步/路径：训练内 {train_count}/{total}"
                    f"（{scaling['train_success_mean'][index] * 100:.1f}%），"
                    f"泛化 {gen_count}/{total}"
                    f"（{scaling['gen_success_mean'][index] * 100:.1f}%）"
                )
            lines.append("\n  通过 = 第 13 课验收：2 mm 路径门限 + 末尾 0.5 s 停稳；")
            lines.append("  纯 BC 力矩直接驱动，无 PD 兜底。")
        else:
            lines = ["③ 这一幕在量化：开环误差与闭环成败\n"]
            ratio = [
                g / max(t, 1e-12)
                for t, g in zip(scaling["full_train_mse_mean"], scaling["full_gen_mse_mean"])
            ]
            for index, size in enumerate(sizes):
                lines.append(
                    f"  {size:>4} 步/路径：训练 MSE "
                    f"{scaling['full_train_mse_mean'][index]:.2e}，"
                    f"泛化 MSE {scaling['full_gen_mse_mean'][index]:.2e}"
                )
            lines.append(f"  泛化/训练 MSE 比值：{min(ratio):.2f}–{max(ratio):.2f}")
            lines.append("  （专家未见过的状态分布上误差更大：分布移）")
            failures = [c for c in report["featured_cases"] if c["kind"] != "best"]
            if failures:
                case = failures[0]
                reason = case.get("failure_reason", "acceptance_criteria")
                cross = case["bc_max_cross_track_mm"]
                cross_text = f"{cross:.0f} mm" if cross is not None else "发散"
                lines.append(f"  失败案例 {case['id']}：最大偏离 {cross_text}（{reason}）")
            lines.append("\n  MSE 降了不等于成功率升：闭环误差会复合。")
        self.stats.configure(text="\n".join(lines))

    def redraw(self):
        if not hasattr(self, "fig"):
            return
        mode = self.mode.get()
        if mode == "paths":
            self.draw_paths()
        elif mode == "scaling":
            self.draw_scaling()
        else:
            self.draw_mechanism()
        self.fill_stats(mode)
        status = {
            "paths": "① 这一幕在对比：专家（模型在环）与 BC（数据在环）的同路径末端轨迹",
            "scaling": "② 这一幕在扫描：每条路径 10/50/200/全量步 × 3 个数据种子",
            "mechanism": "③ 这一幕在量化：开环 MSE、分布移与闭环失败形态",
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
    root.title("第二十八课 · 行为克隆：用数据替代模型")
    root.geometry("1600x760")
    root.minsize(1380, 660)
    BcDemo(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()
