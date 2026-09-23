"""Generate readable evidence figures for lessons 43-56 and the navigation pilot.

Each panel gets a full README-width row. All measurements come from the
archived summary.json/trajectories.npz records under results/. Rerun with:

    uv run python docs/img/make_readme_figures.py
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.plotting import configure_plot_font

RESULTS = Path("results")
OUT = Path("docs/img")
DPI = 140

C_EST = "#b91c1c"
C_PF = "#0f766e"
C_SELF = "#9ca3af"
C_BAD = "#b91c1c"
C_GOOD = "#2563eb"


def style_axes(ax, title):
    ax.set_title(title, fontsize=15, loc="left", pad=13)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", color="#e5e7eb", lw=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def num(v):
    return float("nan") if v is None else v


def load(experiment_dir):
    d = RESULTS / experiment_dir
    report = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    with np.load(d / "trajectories.npz", allow_pickle=False) as npz:
        data = {k: npz[k].copy() for k in npz.files}
    return report, data


def panels(n):
    fig, axes = plt.subplots(n, 1, figsize=(9, n * 3.5), dpi=DPI, layout="constrained")
    return fig, axes


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.has_data():
            raise ValueError(f"{name}: empty panel")
        title = ax._left_title.get_window_extent(renderer)
        if title.x0 < 0 or title.x1 > fig.bbox.x1:
            raise ValueError(f"{name}: panel title leaves image bounds")
        labels = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible()]
        if any(a.overlaps(b) for a, b in pairwise(labels)):
            raise ValueError(f"{name}: x-axis tick labels overlap")
        legend = ax.get_legend()
        if legend is not None:
            box = legend.get_window_extent(renderer)
            if box.x0 < 0 or box.x1 > fig.bbox.x1:
                raise ValueError(f"{name}: legend leaves image bounds")
    fig.savefig(OUT / name, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def traj_panel(ax, chains, title):
    """Top-down trajectory panel: chains = [(label, xy, color, lw)]."""
    for label, xy, color, lw in chains:
        ax.plot(xy[:, 0], xy[:, 1], "-", color=color, lw=lw, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    style_axes(ax, title)
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.legend(
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(chains),
        framealpha=0.95,
    )


def save_trajectory(chains, title, name):
    """Give a metric XY route its own image so README scaling keeps it legible."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI, layout="constrained")
    traj_panel(ax, chains, title)
    save(fig, name)


def bar_values(ax, bars, labels, offset=0.02):
    """Annotate bars, including genuine zeroes; never turn missing data into zero."""
    for bar, label in zip(bars, labels, strict=True):
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=11,
            )


# ------------------------------------------------------------ lesson 43
def lesson43():
    report, data = load("grid_nav_2026-09-06_v2")
    agg = report["aggregates"]
    fig, axes = panels(2)
    names = ["A 盲飞", "B 全栈"]
    rates = [
        agg["A"]["obstacle"]["arrival_rate"],
        agg["B"]["obstacle"]["arrival_rate"],
    ]
    coll = [
        agg["A"]["obstacle"]["collision_events_total"],
        agg["B"]["obstacle"]["collision_events_total"],
    ]
    bars = axes[0].bar(names, rates, color=[C_BAD, C_PF], width=0.55)
    bar_values(axes[0], bars, [f"{int(v * 15)}/15" for v in rates])
    style_axes(axes[0], "有障碍场景到达率")
    axes[0].set_ylabel("到达率", fontsize=12)
    axes[0].set_ylim(0, 1.15)
    bars = axes[1].bar(names, coll, color=[C_BAD, C_PF], width=0.55)
    bar_values(axes[1], bars, [str(v) for v in coll], offset=20)
    style_axes(axes[1], "碰撞事件总数")
    axes[1].set_ylabel("碰撞事件", fontsize=12)
    save(fig, "lesson-43-charts.png")
    save_trajectory(
        [
            ("B 全栈", data["truth_B_1_0"], C_PF, 1.8),
            ("A 盲飞", data["truth_A_1_0"], C_BAD, 1.3),
        ],
        "场景 1 · 真实运动轨迹（俯瞰）",
        "lesson-43-trajectory.png",
    )


# ------------------------------------------------------------ lesson 44
def lesson44():
    report, data = load("nav_pose_error_2026-09-07")
    agg = report["aggregates"]
    names = ["T 真值", "O1 里程计1%", "O2 里程计2%", "F 融合观测"]
    rates = [agg[g]["obstacle"]["arrival_rate"] for g in ("T", "O1", "O2", "F")]
    fig, axes = panels(2)
    bars = axes[0].bar(names, rates, color=[C_GOOD, C_BAD, C_BAD, C_PF], width=0.6)
    bar_values(axes[0], bars, [f"{int(v * 15)}/15" for v in rates])
    style_axes(axes[0], "有障碍场景到达率")
    axes[0].set_ylabel("到达率", fontsize=12)
    axes[0].set_ylim(0, 1.15)
    axes[0].tick_params(axis="x", labelsize=10)
    error_groups = ("T", "O1", "O2", "F")
    errors = [agg[g]["obstacle"]["mean_position_error_m"] for g in error_groups]
    bars = axes[1].bar(
        ["T 真值", "O1 1%", "O2 2%", "F 融合"],
        errors,
        color=[C_GOOD, C_BAD, C_BAD, C_PF],
        width=0.6,
    )
    bar_values(axes[1], bars, [f"{v:.3f}" for v in errors], offset=0.04)
    style_axes(axes[1], "平均定位误差（真值组为 0）")
    axes[1].set_ylabel("平均定位误差 (m)", fontsize=12)
    save(fig, "lesson-44-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_O2_1_0"], "k", 1.7),
            ("O2 估计位姿", data["estimates_O2_1_0"], C_BAD, 1.4),
        ],
        "O2 组 · 真实轨迹与估计位姿（场景 1）",
        "lesson-44-trajectory.png",
    )


# ------------------------------------------------------------ lesson 45
def lesson45():
    report, data = load("scan_slam_2026-09-07")
    agg = report["aggregates"]
    names = ["T 真值", "E 里程计", "S +扫描匹配"]
    rates = [agg[g]["obstacle"]["arrival_rate"] for g in ("T", "E", "S")]
    fig, axes = panels(2)
    bars = axes[0].bar(names, rates, color=[C_GOOD, C_BAD, C_BAD], width=0.55)
    bar_values(axes[0], bars, [f"{int(v * 15)}/15" for v in rates])
    style_axes(axes[0], "有障碍场景到达率")
    axes[0].set_ylabel("到达率", fontsize=12)
    axes[0].set_ylim(0, 1.15)
    errs = []
    for g in ("T", "E", "S"):
        o = agg[g]["obstacle"]
        v = o.get("mean_position_error_m")
        errs.append(num(v))
    bars = axes[1].bar(names, errs, color=[C_GOOD, C_BAD, C_BAD], width=0.55)
    bar_values(axes[1], bars, [f"{v:.2f}" for v in errs], offset=0.04)
    style_axes(axes[1], "平均位置误差：E 与 S 同量级")
    axes[1].set_ylabel("平均误差 (m)", fontsize=12)
    save(fig, "lesson-45-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_S_1_0"], "k", 1.7),
            ("S 估计位姿", data["estimates_S_1_0"], C_BAD, 1.4),
        ],
        "S 组 · 真实轨迹与估计位姿（场景 1）",
        "lesson-45-trajectory.png",
    )


# ------------------------------------------------------------ lesson 46
def lesson46():
    report, data = load("loop_closure_2026-09-07")
    agg = report["aggregates"]
    groups = ("L", "LT")
    before = [agg[g]["final_position_error_m"] for g in groups]
    after = [agg[g]["final_position_error_relaxed_m"] for g in groups]
    fig, axes = panels(2)
    x = np.arange(2)
    axes[0].bar(x - 0.18, before, 0.36, color=C_BAD, label="校正前")
    bars = axes[0].bar(x + 0.18, after, 0.36, color=C_PF, label="校正后")
    bar_values(axes[0], bars, [f"{v:.3f} m" for v in after], offset=0.1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["L 匹配回环", "LT 黄金边"])
    style_axes(axes[0], "回环校正前后的末端误差")
    axes[0].set_ylabel("末端误差 (m)", fontsize=12)
    axes[0].legend(fontsize=10)
    bars = axes[1].bar(["L 匹配回环"], [agg["L"]["loop_ok_rate"]], color=C_PF, width=0.5)
    bar_values(axes[1], bars, [f"{agg['L']['loop_ok_rate']:.2f}"])
    axes[1].set_ylim(0, 1.05)
    style_axes(axes[1], "回环检测成功率（12 次重放）")
    axes[1].set_ylabel("成功率", fontsize=12)
    save(fig, "lesson-46-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_0_0"], "k", 1.7),
            ("L 漂移位姿", data["est_L_0_0"], C_BAD, 1.4),
            ("L 校正节点", data["relaxed_chain"], C_PF, 1.8),
        ],
        "一次巡逻 · 真实轨迹、漂移与回环校正",
        "lesson-46-trajectory.png",
    )


# ------------------------------------------------------------ lesson 47
def lesson47():
    report, data = load("pose_graph_2026-09-07")
    agg = report["aggregates"]
    names = ["N", "ARC 弧长", "FG 因子图", "FGT 黄金锚"]
    keys = ["N", "ARC", "FG", "FGT"]
    means = [num(agg[g]["mean_position_error_m"]) for g in keys]
    finals = [agg[g]["final_position_error_m"] for g in ("ARC", "FG", "FGT")]
    fig, axes = panels(2)
    bars = axes[0].bar(names, means, color=[C_SELF, C_GOOD, C_PF, C_GOOD], width=0.6)
    bar_values(axes[0], bars, [f"{v:.2f}" for v in means], offset=0.05)
    style_axes(axes[0], "全程平均位置误差（形状）")
    axes[0].set_ylabel("平均误差 (m)", fontsize=12)
    bars = axes[1].bar(["ARC", "FG", "FGT"], finals, color=[C_GOOD, C_PF, C_GOOD], width=0.6)
    bar_values(axes[1], bars, [f"{v:.3f}" for v in finals], offset=0.03)
    style_axes(axes[1], "末端闭合误差（N 组未做闭合）")
    axes[1].set_ylabel("末端误差 (m)", fontsize=12)
    save(fig, "lesson-47-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_showcase"], "k", 1.5),
            ("ARC 校正", data["arc_showcase"], C_GOOD, 1.7),
            ("FG 优化", data["fg_showcase"], C_PF, 1.7),
        ],
        "一次巡逻 · 真值与两种后端位姿",
        "lesson-47-trajectory.png",
    )


# ------------------------------------------------------------ lesson 48
def lesson48():
    report, data = load("robust_graph_2026-09-07")
    groups = report["protocol"]["groups"]
    names = list(groups)
    means = [num(report["aggregates"][g]["mean_position_error_m"]) for g in names]
    finals = [num(report["aggregates"][g]["final_position_error_m"]) for g in names]
    fig, axes = panels(2)
    axes[0].bar(names, means, color=[C_SELF, C_PF, C_EST, C_GOOD, C_GOOD])
    style_axes(axes[0], "全程平均位置误差")
    axes[0].set_ylabel("平均误差 (m)", fontsize=12)
    final_groups = names[1:]
    bars = axes[1].bar(final_groups, finals[1:], color=[C_PF, C_EST, C_GOOD, C_GOOD])
    bar_values(axes[1], bars, [f"{v:.2f}" for v in finals[1:]], offset=0.12)
    style_axes(axes[1], "末端误差（N 组无回环后端，不计为 0）")
    axes[1].set_ylabel("末端误差 (m)", fontsize=12)
    save(fig, "lesson-48-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_showcase"], "k", 1.5),
            ("LS 好边", data["chain_LS-G"], C_PF, 1.7),
            ("LS 毒化边", data["chain_LS-B"], C_BAD, 1.7),
        ],
        "一次巡逻 · 好边与毒化边的位姿形状",
        "lesson-48-trajectory.png",
    )


# ------------------------------------------------------------ lesson 49
def lesson49():
    report, _data = load("matcher_engineering_2026-09-08")
    groups = ("B", "P", "R", "K")
    acc = [report["aggregates"][g]["acceptance"] for g in groups]
    fig, axes = panels(2)
    axes[0].bar(groups, acc, color=[C_SELF, C_GOOD, C_GOOD, C_PF])
    style_axes(axes[0], "回环候选接受率：B → P → R → K")
    axes[0].set_ylabel("接受率", fontsize=12)
    legacy = report["aggregates"]["K"]["direction_correctness"]
    bars = axes[1].bar(["K 组方向正确率\n（旧判据，第 54 课勘误）"], [legacy], color=C_BAD)
    bar_values(axes[1], bars, [f"{legacy:.1%}"])
    axes[1].axhline(0.80, color="k", ls="--", lw=1)
    style_axes(axes[1], "方向正确率：旧判据（第 54 课已勘误）")
    axes[1].set_ylim(0, 1.05)
    save(fig, "lesson-49-charts.png")


# ------------------------------------------------------------ lesson 50
def lesson50():
    report, _data = load("feature_loops_2026-09-08")
    r = report["hypothesis"]["results"]
    fig, axes = panels(2)
    names = ("排序直方图", "环移角剖面", "建图一致相关峰")
    overlaps = (
        r["overlap_sorted_histogram"],
        r["overlap_cyclic_profile"],
        r["overlap_map_agreement"],
    )
    axes[0].bar(names, overlaps, color=[C_SELF, C_SELF, C_BAD])
    axes[0].axhline(0.3, color="k", ls="--", lw=1)
    style_axes(axes[0], "三种特征的评分分布重叠度")
    axes[0].set_ylabel("重叠度", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    old_rates = [r["direction_correctness"], 0.049]
    bars = axes[1].bar(
        ["无检索管线", "oracle 基线"],
        old_rates,
        color=[C_BAD, C_SELF],
    )
    bar_values(axes[1], bars, [f"{v:.1%}" for v in old_rates])
    style_axes(axes[1], "旧判据方向正确率（第 54 课已勘误）")
    axes[1].axhline(0.80, color="k", ls="--", lw=1)
    axes[1].set_ylabel("旧判据正确率", fontsize=12)
    axes[1].set_ylim(0, 1.05)
    save(fig, "lesson-50-charts.png")


# ------------------------------------------------------------ lesson 51
def lesson51():
    report, _data = load("switchable_graph_2026-09-08")
    r = report["hypothesis"]["results"]
    curve = r["lambda_curve"]
    lams = sorted(float(k) for k in curve)
    g_final = [curve[str(l)]["sc_g_final"] for l in lams]
    b_mean = [curve[str(l)]["sc_b_mean"] for l in lams]
    fig, axes = panels(2)
    axes[0].plot(lams, g_final, "o-", color=C_PF, label="SC-G 末端（治疗线 1.5 m）")
    axes[0].plot(lams, b_mean, "s-", color=C_BAD, label="SC-B 均值（预防线 = N+0.5）")
    axes[0].axhline(1.5, color=C_PF, ls="--", lw=1)
    est = report["aggregates"]["N"]["mean_position_error_m"]
    axes[0].axhline(est + 0.5, color=C_BAD, ls="--", lw=1)
    axes[0].set_xscale("log")
    style_axes(axes[0], "λ 扫描：好边闭合与毒化抑制无法兼得")
    axes[0].set_xlabel("λ（log）", fontsize=12)
    axes[0].set_ylabel("误差 (m)", fontsize=12)
    axes[0].legend(fontsize=10)
    finals = report["hypothesis"]["results"]["finals"]
    axes[1].bar(
        ["LS-G", "HUB-G", "SC λ=50", "MM-G"],
        [num(finals["LS-G"]), num(finals["HUB-G"]), num(finals["SC-G-lam50"]), num(finals["MM-G"])],
        color=[C_GOOD, C_BAD, C_BAD, C_SELF],
    )
    axes[1].axhline(1.5, color="k", ls="--", lw=1)
    style_axes(axes[1], "好边末端闭合（虚线 = 1.5 m 判据）")
    axes[1].set_ylabel("末端误差 (m)", fontsize=12)
    save(fig, "lesson-51-charts.png")


# ------------------------------------------------------------ lesson 52
def lesson52():
    report, data = load("relative_loops_2026-09-09")
    groups = ("REL-LS-G", "REL-LS-B", "SC-G", "SC-B", "MM-G", "MM-B")
    means = [num(report["aggregates"][g]["mean_position_error_m"]) for g in groups]
    est_mean = report["aggregates"]["N"]["mean_position_error_m"]
    fig, axes = panels(2)
    colors = [C_GOOD, C_BAD, C_PF, C_BAD, C_SELF, C_BAD]
    axes[0].bar(groups, means, color=colors)
    axes[0].axhline(est_mean, color="k", ls="--", lw=1)
    style_axes(axes[0], "相对回环边：各后端平均误差")
    axes[0].set_ylabel("平均误差 (m)", fontsize=12)
    switches = [num(report["aggregates"][g]["switch_final"]) for g in ("SC-G", "SC-B")]
    axes[1].bar(["SC-G", "SC-B"], switches, color=[C_PF, C_BAD])
    axes[1].axhline(0.3, color="k", ls="--", lw=1)
    style_axes(axes[1], "开关终值（虚线 = 0.3 拒绝线）")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("开关 s", fontsize=12)
    save(fig, "lesson-52-charts.png")
    save_trajectory(
        [
            ("LS 好边", data["chain_REL-LS-G"], C_GOOD, 1.6),
            ("LS 毒化边", data["chain_REL-LS-B"], C_BAD, 1.6),
            ("SC 毒化边", data["chain_SC-B"], C_PF, 1.7),
        ],
        "同一里程计链 · 相对边后端的位姿形状",
        "lesson-52-trajectory.png",
    )


# ------------------------------------------------------------ lesson 53
def lesson53():
    report, data = load("production_solver_2026-09-09")
    sweep = report["sweep"]
    labels = [f"{s['loss']}\n{s['scale']}" for s in sweep]
    means = [num(s["mean"]) for s in sweep]
    est_mean = report["aggregates"]["N"]["mean_position_error_m"]
    fig, axes = panels(2)
    axes[0].bar(labels, means, color=[C_BAD if m > est_mean + 0.5 else C_PF for m in means])
    axes[0].axhline(est_mean + 0.5, color="k", ls="--", lw=1)
    style_axes(axes[0], "毒化边：损失函数 × 尺度扫描")
    axes[0].set_ylabel("SCIPY-HUB-B 平均误差 (m)", fontsize=11)
    axes[0].set_ylim(0, max(means) * 1.2)
    groups = ("N", "SCIPY-LS", "SCIPY-HUB-B")
    avg = [report["aggregates"][g]["mean_position_error_m"] for g in groups]
    bars = axes[1].bar(
        ["N 基线", "scipy LS", "scipy Huber-B"], avg, color=[C_SELF, C_GOOD, C_BAD], width=0.6
    )
    bar_values(axes[1], bars, [f"{v:.2f}" for v in avg], offset=0.06)
    style_axes(axes[1], "同口径对照：全程平均位置误差")
    axes[1].set_ylabel("平均误差 (m)", fontsize=12)
    save(fig, "lesson-53-charts.png")
    save_trajectory(
        [
            ("scipy LS", data["chain_SCIPY-LS"], C_GOOD, 1.6),
            ("scipy Huber-B", data["chain_SCIPY-HUB-B"], C_BAD, 1.7),
            ("自实现 Huber-B", data["chain_SELF-HUB-B"], C_SELF, 1.4),
        ],
        "同一里程计链 · 生产求解器后端位姿",
        "lesson-53-trajectory.png",
    )


# ------------------------------------------------------------ lesson 54
def lesson54():
    report, _data = load("aniso_env_2026-09-09")
    agg = report["aggregates"]
    r = report["hypothesis"]["results"]
    fig, axes = panels(2)
    arms = ("ISO", "ANISO")
    legacy = [agg[a]["K"]["legacy_direction_correctness"] for a in arms]
    corrected = [agg[a]["K"]["direction_correctness"] for a in arms]
    x = np.arange(2)
    axes[0].bar(x - 0.18, legacy, 0.36, color=C_SELF, label="旧判据（伪影）")
    axes[0].bar(x + 0.18, corrected, 0.36, color=C_PF, label="修正判据")
    for i, (lv, co) in enumerate(zip(legacy, corrected, strict=True)):
        axes[0].text(i - 0.18, lv + 0.02, f"{lv:.2f}", ha="center", fontsize=11)
        axes[0].text(i + 0.18, co + 0.02, f"{co:.2f}", ha="center", fontsize=11)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["ISO 方墙", "ANISO 多边形+杂物"], fontsize=12)
    axes[0].axhline(0.80, color="k", ls="--", lw=1)
    style_axes(axes[0], "同一匹配器：旧判据与修正判据")
    axes[0].set_ylabel("K 组方向正确率", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=10, loc="upper left")
    axes[1].bar(
        ["ISO 修正", "ANISO 修正"],
        [r["iso_corrected"], r["aniso_corrected"]],
        color=[C_GOOD, C_BAD],
    )
    axes[1].axhline(0.80, color="k", ls="--", lw=1)
    style_axes(axes[1], "修正判据：各向异性环境未改善")
    axes[1].set_ylim(0, 1.05)
    save(fig, "lesson-54-charts.png")


# ------------------------------------------------------------ lesson 55
def lesson55():
    report, data = load("rgbd_loops_2026-09-23_v2")
    agg = report["aggregates"]
    fig, axes = panels(2)
    groups = ("GEO-ALL", "GEO-ORACLE", "RGBD-TOPK")
    names = ("无检索基线", "真值候选参考", "标记检索 top-12")
    correct = [agg[g]["direction_correctness"] for g in groups]
    axes[0].bar(names, correct, color=[C_SELF, C_GOOD, C_PF])
    axes[0].axhline(0.80, color="k", ls="--", lw=1)
    for i, v in enumerate(correct):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=11)
    style_axes(axes[0], "方向正确率：检索前后对照")
    axes[0].set_ylabel("方向正确率", fontsize=12)
    axes[0].set_ylim(0, 1.08)
    frames = data["probe_frames_all"]
    sims = data["probe_sims"]
    true_loop = data["probe_near_start"].astype(bool)
    selected = np.isin(frames, data["topk_frames"])
    axes[1].scatter(
        frames[~true_loop], sims[~true_loop], color=C_SELF, s=17, label="其他帧", zorder=2
    )
    axes[1].scatter(
        frames[true_loop], sims[true_loop], color=C_BAD, s=24, label="真回环帧", zorder=3
    )
    axes[1].scatter(
        frames[selected],
        sims[selected],
        facecolors="none",
        edgecolors="black",
        s=115,
        linewidths=1.2,
        label="top-12",
        zorder=4,
    )
    style_axes(axes[1], "外观相似度：候选帧与 top-12")
    axes[1].set_xlabel("探帧", fontsize=12)
    axes[1].set_ylabel("外观相似度", fontsize=12)
    axes[1].set_ylim(0.8, 1.01)
    axes[1].legend(fontsize=10, loc="lower left", ncol=3)
    save(fig, "lesson-55-charts.png")


# ------------------------------------------------------------ lesson 56
def lesson56():
    report, data = load("map_localization_2026-09-24_v5")
    fig, axes = panels(2)
    axes[0].plot(data["est_err_curve"], color=C_EST, lw=1.2, label="EST 无图")
    axes[0].plot(data["pf_self_err"], color="#7c3aed", lw=1.0, label="PF 自建图")
    axes[0].plot(data["pf_sensor_truth_err"], color=C_GOOD, lw=1.0, label="PF 真值位姿投影图")
    axes[0].plot(data["pf_true_err"], color=C_PF, lw=1.5, label="PF 理想占据图")
    axes[0].axhline(0.6, color=C_PF, ls="--", lw=1)
    style_axes(axes[0], "位置误差随帧变化（虚线 = 0.6 m 目标）")
    axes[0].set_xlabel("帧", fontsize=12)
    axes[0].set_ylabel("位置误差 (m)", fontsize=12)
    axes[0].legend(fontsize=10, loc="upper left", ncol=2, framealpha=0.95)
    trials = report["hypothesis"]["results"]["kidnap_trials"]
    labels = [f"{t['fraction']:.2f}" for t in trials]
    ends = [t.get("end_err_m", float("nan")) for t in trials]
    bars = axes[1].bar(labels, ends, color=[C_GOOD if t.get("success") else C_BAD for t in trials])
    bar_values(axes[1], bars, [f"{v:.1f}" for v in ends], offset=0.15)
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    style_axes(axes[1], "理想图绑架重定位：5 次中 4 次达到恢复线")
    axes[1].set_xlabel("绑架时点（巡游进度）", fontsize=12)
    axes[1].set_ylabel("恢复窗末误差 (m)", fontsize=12)
    save(fig, "lesson-56-charts.png")
    save_trajectory(
        [
            ("真实运动", data["truth_chain"], "k", 1.6),
            ("EST 无图", data["est_chain"], C_EST, 1.5),
            ("PF 理想占据图", data["pf_true_chain"], C_PF, 1.7),
        ],
        "同一巡游 · 真实轨迹与两条估计链",
        "lesson-56-trajectory.png",
    )


def navigation_pilot():
    """Small-sample pilot: show localization and same-scan map alignment separately."""
    path = Path("docs/benchmarks/navigation-paired-pilot-v4.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    fig, axes = panels(2)
    arms = ("ISO", "ANISO")
    x = np.arange(len(arms))
    width = 0.23
    for offset, key, label, color in (
        (-width, "est_mean_m", "EST 无图", C_EST),
        (0, "pf_true_mean_m", "PF 理想图", C_PF),
        (width, "pf_self_mean_m", "PF 自建图", "#7c3aed"),
    ):
        values = [report["summary"]["by_arm"][arm]["mean_errors_m"][key] for arm in arms]
        bars = axes[0].bar(x + offset, values, width, color=color, label=label)
        bar_values(axes[0], bars, [f"{v:.2f}" for v in values], offset=0.05)
    axes[0].set_xticks(x, arms)
    axes[0].set_ylim(0, 3.7)
    axes[0].set_ylabel("平均位置误差 (m)", fontsize=12)
    axes[0].legend(fontsize=10, loc="upper right", ncol=3)
    style_axes(axes[0], "跨场景试跑：每组 3 个有效场景 × 2 个传感器种子")

    rows = [row for row in report["rows"] if row["status"] == "ok" and row["sensor_seed"] == 0]
    labels = [f"{row['arm']} 场景{row['world_seed']}" for row in rows]
    xx = np.arange(len(rows))
    axes[1].bar(
        xx - 0.18,
        [row["map_alignment_precision"] for row in rows],
        0.36,
        label="精确率",
        color=C_GOOD,
    )
    axes[1].bar(
        xx + 0.18,
        [row["map_alignment_recall"] for row in rows],
        0.36,
        label="召回率",
        color=C_PF,
    )
    axes[1].axhline(0.7, color="black", ls="--", lw=1)
    axes[1].set_xticks(xx, labels)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("同帧地图对齐", fontsize=12)
    axes[1].legend(fontsize=10, loc="upper right", ncol=2)
    style_axes(axes[1], "地图表面对齐：虚线为精确率与召回率的 0.70 门槛")
    save(fig, "navigation-pilot-charts.png")


def main():
    configure_plot_font()
    lesson43()
    lesson44()
    lesson45()
    lesson46()
    lesson47()
    lesson48()
    lesson49()
    lesson50()
    lesson51()
    lesson52()
    lesson53()
    lesson54()
    lesson55()
    lesson56()
    navigation_pilot()


if __name__ == "__main__":
    main()
