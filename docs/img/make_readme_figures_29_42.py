"""Render standalone README results from the official lesson 29-42 records.

Run from the repository root: uv run python docs/img/make_readme_figures_29_42.py
The old demo mosaics remain available as expandable interface references.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from make_readme_figures import C_BAD, C_GOOD, C_PF, C_SELF, panels, save, style_axes

from embodied_learning.plotting import configure_plot_font

ROOT = Path("results")
RECORDS = {
    29: "ppo_swingup_2026-09-06",
    30: "residual_swingup_2026-09-06",
    31: "pbrs_swingup_2026-09-06",
    32: "dapg_swingup_2026-09-06",
    33: "goexplore_swingup_2026-09-06",
    34: "twophase_swingup_2026-09-06",
    35: "sac_swingup_2026-09-06",
    36: "dagger_swingup_2026-09-06",
    37: "act_swingup_2026-09-06",
    38: "combo_swingup_2026-09-06",
    39: "rl_goal_reaching_2026-09-06",
    40: "rl_arm_reaching_2026-09-06",
    41: "staged_verification_2026-09-06",
    42: "act_torch_reaching_2026-09-06",
}


def record(lesson):
    folder = ROOT / RECORDS[lesson]
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with np.load(folder / "trajectories.npz", allow_pickle=False) as npz:
        data = {k: npz[k].copy() for k in npz.files}
    return report, data


def curve(ax, values, title, ylabel, *, color=C_PF):
    values = np.asarray(values)
    stride = max(1, len(values) // 1000)
    ax.plot(np.arange(len(values))[::stride], values[::stride], color=color, lw=1.5)
    style_axes(ax, title)
    ax.set_xlabel("记录点", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)


def phase(ax, baseline, sample, title, *, sample_label="本课样本", sample_color=C_BAD):
    """A cart-pole motion trace in its native angle/angular-speed state space."""
    ax.plot(baseline[:, 1], baseline[:, 3], color=C_SELF, lw=1.3, label="第 7 课基线")
    ax.plot(sample[:, 1], sample[:, 3], color=sample_color, lw=1.5, label=sample_label)
    style_axes(ax, title)
    ax.set_xlabel("杆角 θ (rad)", fontsize=12)
    ax.set_ylabel("角速度 (rad/s)", fontsize=12)
    ax.legend(fontsize=11, loc="upper right")


def pendulum_lessons():
    recipes = {
        29: ("reward_curve_0", "训练奖励 · 种子 0", "平均奖励", "eval_det_states", "PPO 均值动作"),
        30: (
            "reward_curve_0_0",
            "残差训练奖励 · 档 0 / 种子 0",
            "平均奖励",
            "det_states_0_0",
            "残差策略",
        ),
        31: (
            "reward_curve_1_1",
            "PBRS 训练奖励 · 高势权 / 种子 1",
            "平均奖励",
            "det_states_1_1",
            "PBRS 策略",
        ),
        32: (
            "reward_curve_0_0",
            "DAPG 训练奖励 · 档 0 / 种子 0",
            "平均奖励",
            "det_states_0_0",
            "DAPG 策略",
        ),
        33: (
            "coverage_curve",
            "Go-Explore 档案覆盖格数",
            "已访问格数",
            "capture0_full_states",
            "捕获轨迹",
        ),
        34: (
            "reward_curve_0",
            "两阶段奖励训练曲线 · 种子 0",
            "平均奖励",
            "det_states_0",
            "两阶段策略",
        ),
        35: (
            "alpha_curve_auto_0",
            "SAC 自动温度 α · 种子 0",
            "α",
            "det_states_auto_0",
            "SAC 自动温度",
        ),
        38: (
            "reward_curve_0_0",
            "组合策略训练奖励 · 档 0 / 种子 0",
            "平均奖励",
            "det_states_0_0",
            "组合策略",
        ),
    }
    for lesson, (metric, title, ylabel, state_key, label) in recipes.items():
        _report, data = record(lesson)
        fig, axes = panels(2)
        curve(axes[0], data[metric], title, ylabel)
        axes[0].set_xlabel("档案探索段" if lesson == 33 else "优化更新序号", fontsize=12)
        sample = data[state_key]
        if sample.ndim == 3:
            sample = sample[0]
        phase(
            axes[1],
            data["baseline_states"],
            sample,
            "杆角–角速度运动轨迹（单次样本）",
            sample_label=label,
            sample_color=C_PF if lesson == 33 else C_BAD,
        )
        save(fig, f"lesson-{lesson:02d}-results.png")


def lesson36():
    _report, data = record(36)
    fig, axes = panels(2)
    counts = np.isfinite(data["eval_arrival_s_0_0"]).sum(axis=1)
    axes[0].plot(np.arange(len(counts)), counts, "o-", color=C_PF, lw=2)
    style_axes(axes[0], "DAgger 档 0 / 种子 0：每轮评估到达数")
    axes[0].set_xlabel("纠错轮次", fontsize=12)
    axes[0].set_ylabel("到达回合 / 20", fontsize=12)
    axes[0].set_ylim(bottom=0)
    phase(
        axes[1],
        data["baseline_states"],
        data["det_states_0_0_r5"],
        "末轮均值动作：杆角–角速度运动轨迹",
        sample_label="DAgger 末轮",
    )
    save(fig, "lesson-36-results.png")


def lesson37():
    _report, data = record(37)
    fig, axes = panels(2)
    names = ["单步 MSE", "块 K1", "块 K2", "块 K4", "扩展档"]
    reached = [np.isfinite(data[f"ckpt_arrival_s_{i}_0"]).sum() for i in range(5)]
    bars = axes[0].bar(names, reached, color=[C_SELF, C_GOOD, C_GOOD, C_PF, C_SELF])
    for bar, value in zip(bars, reached, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, value + 0.15, f"{value}/12", ha="center", fontsize=11
        )
    style_axes(axes[0], "种子 0：12 个检查点中均值动作进入直立区的次数")
    axes[0].set_ylabel("有到达的检查点", fontsize=12)
    axes[0].set_ylim(0, 13)
    phase(
        axes[1],
        data["baseline_states"],
        data["det_mixture_states_3_0"],
        "K4 块策略：杆角–角速度运动轨迹",
        sample_label="K4 块策略",
        sample_color=C_PF,
    )
    save(fig, "lesson-37-results.png")


def planar_path(ax, baseline, policy, goal, title, *, arm=False):
    if arm:
        baseline, policy = baseline[:, 2, :], policy[:, 2, :]
    else:
        baseline, policy = baseline[:, :2], policy[:, :2]
    ax.plot(baseline[:, 0], baseline[:, 1], color=C_GOOD, lw=1.6, label="解析/手工基线")
    ax.plot(policy[:, 0], policy[:, 1], color=C_BAD, lw=1.5, label="学习策略")
    ax.scatter([goal[0]], [goal[1]], color=C_PF, marker="*", s=160, label="目标")
    style_axes(ax, title)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=11, loc="upper right")


def reaching_lesson(lesson, *, arm=False, act=False):
    _report, data = record(lesson)
    fig, axes = panels(2)
    for seed, color in enumerate((C_BAD, C_GOOD, C_PF)):
        y = data[f"eval_curve_successes_{seed}"]
        x = data[f"eval_curve_epochs_{seed}"] if act else data[f"eval_curve_steps_{seed}"]
        axes[0].plot(x, y, "o-", color=color, label=f"种子 {seed}")
    style_axes(axes[0], "训练中定期评估：成功回合数")
    axes[0].set_xlabel("epoch" if act else "训练步", fontsize=12)
    axes[0].set_ylabel("成功回合 / 20", fontsize=12)
    axes[0].set_ylim(0, 21)
    axes[0].legend(fontsize=11, loc="upper right")
    planar_path(
        axes[1],
        data["baseline_truth_0"],
        data["eval_truth_0_0"],
        data["eval_goals"][0],
        "目标 0：机械臂末端轨迹" if arm else "目标 0：差速车真实运动轨迹",
        arm=arm,
    )
    save(fig, f"lesson-{lesson:02d}-results.png")


def lesson41():
    report, data = record(41)
    fig, axes = panels(2)
    rows = report["comparison_table"][:2]
    bars = axes[0].bar(
        ["基线", "基线+线性残差"], [r["median_settled_at_s"] for r in rows], color=[C_SELF, C_GOOD]
    )
    for bar, row in zip(bars, rows, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{row['successes']}/20 · {row['median_settled_at_s']:.2f}s",
            ha="center",
            fontsize=11,
        )
    style_axes(axes[0], "底座与线性修正：成功率相同，稳定时间相同")
    axes[0].set_ylabel("稳定时间中位数 (s)", fontsize=12)
    axes[0].set_ylim(0, 5.8)
    phase(
        axes[1],
        data["baseline_states"],
        data["lin_eval_states"],
        "杆角–角速度运动轨迹",
        sample_label="线性修正",
        sample_color=C_GOOD,
    )
    save(fig, "lesson-41-results.png")


def main():
    configure_plot_font()
    pendulum_lessons()
    lesson36()
    lesson37()
    reaching_lesson(39)
    reaching_lesson(40, arm=True)
    lesson41()
    reaching_lesson(42, arm=True, act=True)


if __name__ == "__main__":
    main()
