"""Generate the README chart figures for lessons 48-56 from the records.

One PNG per lesson (one or two panels, unified style), read from the
official records under results/.  Rerun after any record refresh:

    uv run python docs/img/make_readme_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.plotting import configure_plot_font

RESULTS = Path("results")
OUT = Path("docs/img")
DPI = 150

C_EST = "#b91c1c"
C_PF = "#0f766e"
C_SELF = "#9ca3af"
C_BAD = "#b91c1c"
C_GOOD = "#2563eb"


def style_axes(ax, title):
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def num(v):
    return v if v is not None else 0.0


def load(experiment_dir):
    d = RESULTS / experiment_dir
    report = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    with np.load(d / "trajectories.npz", allow_pickle=False) as npz:
        data = {k: npz[k].copy() for k in npz.files}
    return report, data


def two_panel(figsize=(10.0, 4.2)):
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=DPI)
    return fig, axes


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------ lesson 48
def lesson48():
    report, data = load("robust_graph_2026-09-07")
    groups = report["protocol"]["groups"]
    names = list(groups)
    means = [num(report["aggregates"][g]["mean_position_error_m"]) for g in names]
    finals = [num(report["aggregates"][g]["final_position_error_m"]) for g in names]
    fig, axes = two_panel()
    axes[0].bar(names, means, color=[C_EST, C_PF, C_EST, C_PF, C_GOOD])
    style_axes(axes[0], "平均位置误差（m）：毒化边比无回环更糟")
    axes[0].set_ylabel("平均误差 (m)", fontsize=12)
    axes[1].bar(names, finals, color=[C_EST, C_PF, C_EST, C_PF, C_GOOD])
    style_axes(axes[1], "末端误差（m）：Huber 下好回环也无法闭合")
    save(fig, "lesson-48-charts.png")


# ------------------------------------------------------------ lesson 49
def lesson49():
    report, data = load("matcher_engineering_2026-09-08")
    groups = ("B", "P", "R", "K")
    acc = [report["aggregates"][g]["acceptance"] for g in groups]
    fig, axes = two_panel()
    axes[0].bar(groups, acc, color=[C_SELF, C_PF, C_GOOD, C_BAD])
    style_axes(axes[0], "接受率阶梯：0.32 → 0.43 → 0.52 → 0.76")
    axes[0].set_ylabel("接受率", fontsize=12)
    legacy = report["aggregates"]["K"]["direction_correctness"]
    axes[1].bar(["K 组方向正确率\n（旧判据，第 54 课勘误）"], [legacy], color=C_BAD)
    axes[1].axhline(0.80, color="k", ls="--", lw=1)
    style_axes(axes[1], "方向正确率 ≤5.4%（旧判据；修正后见第 54 课）")
    axes[1].set_ylim(0, 1.05)
    save(fig, "lesson-49-charts.png")


# ------------------------------------------------------------ lesson 50
def lesson50():
    report, _data = load("feature_loops_2026-09-08")
    r = report["hypothesis"]["results"]
    fig, axes = two_panel()
    names = ("排序直方图", "环移角剖面", "建图一致相关峰")
    overlaps = (
        r["overlap_sorted_histogram"],
        r["overlap_cyclic_profile"],
        r["overlap_map_agreement"],
    )
    axes[0].bar(names, overlaps, color=[C_SELF, C_SELF, C_BAD])
    axes[0].axhline(0.3, color="k", ls="--", lw=1)
    style_axes(axes[0], "判别分布重叠度（虚线=0.3 可分离线）")
    axes[0].set_ylabel("重叠度", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(
        ["无检索管线", "oracle 基线"],
        [r["direction_correctness"], 0.049],
        color=[C_BAD, C_SELF],
    )
    style_axes(axes[1], "方向正确率：2.6% vs 4.9%（均远低于 80% 线）")
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
    fig, axes = two_panel()
    axes[0].plot(lams, g_final, "o-", color=C_PF, label="SC-G 末端（治疗线 1.5 m）")
    axes[0].plot(lams, b_mean, "s-", color=C_BAD, label="SC-B 均值（预防线 = N+0.5）")
    axes[0].axhline(1.5, color=C_PF, ls="--", lw=1)
    est = report["aggregates"]["N"]["mean_position_error_m"]
    axes[0].axhline(est + 0.5, color=C_BAD, ls="--", lw=1)
    axes[0].set_xscale("log")
    style_axes(axes[0], "λ 敏感性：两失效带反相重叠，可行域为空")
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
    style_axes(axes[1], "各后端末端闭合（虚线=1.5 m 治疗线）")
    axes[1].set_ylabel("末端误差 (m)", fontsize=12)
    save(fig, "lesson-51-charts.png")


# ------------------------------------------------------------ lesson 52
def lesson52():
    report, data = load("relative_loops_2026-09-09")
    groups = ("REL-LS-G", "REL-LS-B", "SC-G", "SC-B", "MM-G", "MM-B")
    means = [num(report["aggregates"][g]["mean_position_error_m"]) for g in groups]
    est_mean = report["aggregates"]["N"]["mean_position_error_m"]
    fig, axes = two_panel()
    colors = [C_GOOD, C_BAD, C_PF, C_BAD, C_SELF, C_BAD]
    axes[0].bar(groups, means, color=colors)
    axes[0].axhline(est_mean, color="k", ls="--", lw=1)
    axes[0].text(5.4, est_mean + 0.1, "N 基线", fontsize=10)
    style_axes(axes[0], "各后端平均误差（m）：毒化相对边同样毒（脆弱性跨边型）")
    axes[0].set_ylabel("平均误差 (m)", fontsize=12)
    switches = [num(report["aggregates"][g]["switch_final"]) for g in ("SC-G", "SC-B")]
    axes[1].bar(["SC-G", "SC-B"], switches, color=[C_PF, C_BAD])
    axes[1].axhline(0.3, color="k", ls="--", lw=1)
    axes[1].text(1.05, 0.32, "拒绝线 0.3", fontsize=10)
    style_axes(axes[1], "开关终值：SC-B 折中 s=0.61 半闭合而非拒绝")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("开关 s", fontsize=12)
    save(fig, "lesson-52-charts.png")


# ------------------------------------------------------------ lesson 53
def lesson53():
    report, _data = load("production_solver_2026-09-09")
    sweep = report["sweep"]
    labels = [f"{s['loss']}\n{s['scale']}" for s in sweep]
    means = [num(s["mean"]) for s in sweep]
    est_mean = report["aggregates"]["N"]["mean_position_error_m"]
    fig, axes = two_panel()
    axes[0].bar(labels, means, color=[C_BAD if m > est_mean + 0.5 else C_PF for m in means])
    axes[0].axhline(est_mean + 0.5, color="k", ls="--", lw=1)
    style_axes(axes[0], "损失×尺度扫描：无组合把毒化边压回 N+0.5 内")
    axes[0].set_ylabel("SCIPY-HUB-B 平均误差 (m)", fontsize=11)
    axes[0].set_ylim(0, max(means) * 1.2)
    finals = report["hypothesis"]["results"]["finals"]
    axes[1].bar(
        ["EST", "SCIPY-LS", "SCIPY-HUB-B"],
        [
            report["aggregates"]["N"]["mean_position_error_m"],
            finals["SCIPY-LS"],
            finals["SCIPY-HUB-B"],
        ],
        color=[C_SELF, C_GOOD, C_BAD],
    )
    style_axes(axes[1], "对照：好边闭合正常，毒化边全灭")
    axes[1].set_ylabel("误差 (m)", fontsize=12)
    save(fig, "lesson-53-charts.png")


# ------------------------------------------------------------ lesson 54
def lesson54():
    report, _data = load("aniso_env_2026-09-09")
    agg = report["aggregates"]
    r = report["hypothesis"]["results"]
    fig, axes = two_panel()
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
    style_axes(axes[0], "度量勘误：同一匹配器，换判据后 0.0 → 0.58")
    axes[0].set_ylabel("K 组方向正确率", fontsize=12)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=10, loc="upper left")
    axes[1].bar(
        ["ISO 修正", "ANISO 修正"],
        [r["iso_corrected"], r["aniso_corrected"]],
        color=[C_GOOD, C_BAD],
    )
    axes[1].axhline(0.80, color="k", ls="--", lw=1)
    style_axes(axes[1], "环境假设证伪：ANISO 反而略低")
    axes[1].set_ylim(0, 1.05)
    save(fig, "lesson-54-charts.png")


# ------------------------------------------------------------ lesson 55
def lesson55():
    report, data = load("rgbd_loops_2026-09-09")
    agg = report["aggregates"]
    fig, axes = two_panel()
    groups = ("GEO-ALL", "GEO-ORACLE", "RGBD-TOPK")
    names = ("无检索基线", "真值候选参考", "RGB-D top-12")
    correct = [agg[g]["direction_correctness"] for g in groups]
    axes[0].bar(names, correct, color=[C_SELF, C_GOOD, C_BAD])
    axes[0].axhline(0.80, color="k", ls="--", lw=1)
    for i, v in enumerate(correct):
        axes[0].text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=11)
    style_axes(axes[0], "方向正确率：检索后 0.302 → 1.000")
    axes[0].set_ylabel("方向正确率", fontsize=12)
    axes[0].set_ylim(0, 1.08)
    frames = data["probe_frames_all"]
    sims = data["probe_sims"]
    near = data["probe_near_start"]
    topk = set(data["topk_frames"].tolist())
    for i, f in enumerate(frames):
        ax = axes[1]
        color = C_BAD if near[i] else C_SELF
        ax.scatter(f, sims[i], c=color, s=18 if f in topk else 8,
                   zorder=3 if f in topk else 2)
    style_axes(axes[1], "检索迹线：红=真回环帧，大点=top-12（全红=精确率 1.0）")
    axes[1].set_xlabel("探帧", fontsize=12)
    axes[1].set_ylabel("外观相似度", fontsize=12)
    axes[1].set_ylim(0.8, 1.01)
    save(fig, "lesson-55-charts.png")


# ------------------------------------------------------------ lesson 56
def lesson56():
    report, data = load("map_localization_2026-09-09")
    fig, axes = two_panel(figsize=(10.0, 4.4))
    axes[0].plot(data["est_err_curve"], color=C_EST, lw=1.1, label="EST 无图漂移链")
    axes[0].plot(data["pf_self_err"], color=C_SELF, lw=0.9, label="PF-SELFBUILT 自建图")
    axes[0].plot(data["pf_true_err"], color=C_PF, lw=1.3, label="PF-TRUEMAP 优质图")
    axes[0].axhline(0.6, color=C_PF, ls="--", lw=1)
    style_axes(axes[0], "误差曲线：EST 累积发散 vs PF 有界振荡（世界模型的价值）")
    axes[0].set_xlabel("帧", fontsize=12)
    axes[0].set_ylabel("位置误差 (m)", fontsize=12)
    axes[0].legend(fontsize=10)
    trials = report["hypothesis"]["results"]["kidnap_trials"]
    labels = [f"{t['fraction']:.2f}" for t in trials]
    ends = [t.get("end_err_m", float("nan")) for t in trials]
    axes[1].bar(labels, ends, color=[C_GOOD if t.get("success") else C_BAD for t in trials])
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    style_axes(axes[1], "绑架重定位 0/5（虚线=1.0 m 恢复线，阴性入册）")
    axes[1].set_xlabel("绑架时点（巡游进度）", fontsize=12)
    axes[1].set_ylabel("恢复窗末误差 (m)", fontsize=12)
    save(fig, "lesson-56-charts.png")


def main():
    configure_plot_font()
    lesson48()
    lesson49()
    lesson50()
    lesson51()
    lesson52()
    lesson53()
    lesson54()
    lesson55()
    lesson56()


if __name__ == "__main__":
    main()
