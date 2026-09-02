"""Lesson 9: reuse the arm teaching window for path replay and singularity probes."""

import argparse
import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from embodied_learning.arm_demo import ArmDemo, ArmReplay, draw_arm
from embodied_learning.arm_dynamics import FEEDFORWARD_COLORS, FEEDFORWARD_METHODS
from embodied_learning.arm_path import (
    IK_COMPARISON_METHODS,
    METHOD_COLORS,
    METHODS,
    TIMINGS,
    damped_velocity,
    jacobian,
    segment_distance,
    terminal_window_diagnostics,
)
from embodied_learning.planar_arm import LENGTHS

WINDOW_TITLE = "第九课 · Jacobian：沿直线运动与伸直的限制"


@dataclass
class PathReplay(ArmReplay):
    desired: np.ndarray
    q_reference: np.ndarray
    dq_reference: np.ndarray
    requested_torques: np.ndarray | None = None
    feedforward_torques: np.ndarray | None = None
    feedback_torques: np.ndarray | None = None

    @cached_property
    def terminal_check(self):
        return terminal_window_diagnostics(
            self.states,
            self.points,
            self.metadata["target_m"],
            self.metadata["goal_q_rad"],
            self.dt,
            move_seconds=self.metadata.get("movement_s", 8.0),
        )


def load_replays(directory):
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    families = {
        "planar_2r_path": METHODS,
        "planar_2r_ik_path": IK_COMPARISON_METHODS,
        "planar_2r_timing": tuple((key, seconds) for key, seconds, _ in TIMINGS),
        "planar_2r_feedforward": FEEDFORWARD_METHODS,
    }
    expected = families.get(report.get("experiment"))
    if expected is None or report.get("schema_version") != 1:
        raise ValueError("Expected arm path, IK, timing or feedforward results")
    if not np.allclose(report["lengths_m"], LENGTHS, atol=1e-12, rtol=0):
        raise ValueError("Saved arm geometry differs from the probe")
    if [case["key"] for case in report["cases"]] != [m for m, _ in expected]:
        raise ValueError("Expected the ordered paired comparison methods")
    replays = []
    with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
        for case in report["cases"]:
            timing = report["experiment"] == "planar_2r_timing"
            feedforward = report["experiment"] == "planar_2r_feedforward"
            if timing or feedforward:
                seconds = dict(expected)[case["key"]] if timing else 4.0
                if case.get("movement_s") != seconds or case.get("hold_s") != 3:
                    raise ValueError("Invalid timing protocol")
                if timing and case.get("status") == "planning_rejected":
                    peak = case.get("peak_reference_speed_rad_s", float("nan"))
                    if not math.isfinite(peak) or peak <= 1 or not case.get("reason"):
                        raise ValueError("Invalid planning rejection")
                    if any(k.startswith(case["key"] + "_") for k in archive.files):
                        raise ValueError("Rejected plans must not contain fabricated trajectories")
                    continue
                if case.get("status") != "executed":
                    raise ValueError("Invalid execution status")
            key, dt = case["key"], float(case["dt_s"])
            arrays = [
                archive[f"{key}_{name}"].copy()
                for name in (
                    "states",
                    "points",
                    "torques_nm",
                    "desired_points",
                    "q_reference",
                    "dq_reference",
                )
            ]
            states, points, torques, desired, qref, dqref = arrays
            n = len(torques)
            shapes = [(n + 1, 4), (n + 1, 3, 2), (n, 2), (n + 1, 2), (n + 1, 2), (n, 2)]
            if (
                n < 1
                or not math.isfinite(dt)
                or dt <= 0
                or case["steps"] != n
                or any(
                    value.shape != shape or not np.isfinite(value).all()
                    for value, shape in zip(arrays, shapes, strict=True)
                )
            ):
                raise ValueError("Invalid path replay or timestamp alignment")
            if not np.isclose(case["duration_s"], n * dt):
                raise ValueError("Invalid duration")
            requested = None
            if timing or feedforward:
                if case.get("completed") and not np.isclose(n * dt, seconds + 3):
                    raise ValueError("Invalid timing horizon")
                requested = archive[f"{key}_requested_torques_nm"].copy()
                if requested.shape != (n, 2) or not np.isfinite(requested).all():
                    raise ValueError("Invalid requested torque array")
                if not np.allclose(torques, np.clip(requested, -0.25, 0.25), rtol=0, atol=1e-12):
                    raise ValueError("Applied torque does not match bounded PD request")
                requested.flags.writeable = False
            ff, fb = None, None
            if feedforward:
                ff, fb = [
                    archive[f"{key}_{name}_torques_nm"].copy()
                    for name in ("feedforward", "feedback")
                ]
                if any(v.shape != (n, 2) or not np.isfinite(v).all() for v in (ff, fb)):
                    raise ValueError("Invalid feedforward/feedback arrays")
                if not np.allclose(requested, ff + fb, rtol=0, atol=1e-12):
                    raise ValueError("Torque components do not add to the total request")
                if key == "pd" and np.any(ff != 0):
                    raise ValueError("The PD-only baseline must not contain feedforward")
                ff.flags.writeable = fb.flags.writeable = False
            for value in arrays:
                value.flags.writeable = False
            replays.append(
                PathReplay(
                    case, states, points, torques, dt, desired, qref, dqref, requested, ff, fb
                )
            )
    if not replays:
        raise ValueError("No executed trajectories to replay")
    return replays


class PathDemo(ArmDemo):
    def __init__(self, root, replays, speed=0.25, rejected_cases=()):
        self.is_timing = replays[0].metadata.get("lesson") == 12
        self.is_feedforward = replays[0].metadata.get("lesson") == 13
        self.rejected_cases = rejected_cases
        # Open on the new method, while retaining all three case choices and old controls.
        preferred = (
            "waypoint_ik"
            if any(r.metadata["key"] == "waypoint_ik" for r in replays)
            else "jacobian_path"
        )
        if self.is_feedforward:
            preferred = "feedforward_pd"
        colors = {
            **METHOD_COLORS,
            **{key: color for key, _, color in TIMINGS},
            **FEEDFORWARD_COLORS,
        }
        self.method_styles = [
            (r.metadata["key"], r.metadata["label"], colors[r.metadata["key"]]) for r in replays
        ]
        ordered = sorted(replays, key=lambda r: r.metadata["key"] != preferred)
        super().__init__(root, ordered, speed)
        root.title(WINDOW_TITLE)
        self.heading_label.configure(text="到达终点 ≠ 沿直线到达：Jacobian 把两种速度联系起来")
        trial = self.replay.metadata.get("trial")
        if self.is_feedforward:
            name = trial["id"]
            root.title(f"第十三课 · 模型前馈与 PD 修正：{name}")
            self.heading_label.configure(text=f"第十三课 · {name}：提前出力，再修正误差")
        elif self.is_timing:
            name = trial["id"]
            root.title(f"第十二课 · 动作时间与电机限制：{name}")
            self.heading_label.configure(text=f"第十二课 · {name}：同一路径，8 / 4 / 2 秒")
        elif self.replay.metadata.get("lesson") == 11:
            name = trial["id"] if trial else "fixed_line"
            root.title(f"第十一课 · 逐点解析 IK 与局部 Jacobian：{name}")
            self.heading_label.configure(text=f"第十一课 · {name}：换参考算法，不换电机")
        elif trial:
            root.title(f"第十课 · 多路径检验：{trial['id']}（seed={trial['seed']}）")
            self.heading_label.configure(text=f"第十课 · {trial['id']}：沿用同一机械臂与控制器")
        self.tabs.tab(0, text="① Jacobian 探针：伸直时缺少哪个方向？")
        self.tabs.tab(1, text="② 三种方案：真实动力学慢放")
        if self.is_timing:
            self.tabs.tab(1, text="② 不同时长：真实动力学慢放")
        if self.is_feedforward:
            self.tabs.tab(1, text="② 原 PD / 前馈 + PD：4秒路径对照")
        self.geometry_hint.configure(
            text="静态速度预测：滑块直接设姿态，不驱动电机。红箭头为希望向世界 +X 移动，绿箭头为 J·dq 的预测。"
        )
        self.geometry_footer.configure(
            text="先点 0°, 0°，再点 90°, 0°：都伸直，却缺少不同方向。阻尼最小二乘不会补回缺失的自由运动方向。"
        )
        self.replay_footer.configure(
            text="灰虚线=规定路径；橙圆=此刻规定点；橙叉=终点；紫线=实际已走路径。8 s 移动 + 3 s 停留；无碰撞/噪声。"
        )
        self.set_angles((0, 0))
        self.tabs.select(1)
        self.refresh()

    def update_geometry(self, _=None):
        q = np.deg2rad([var.get() for var in self.q_vars])
        self.probe.reset(q)
        j = jacobian(q)
        sigma = np.linalg.svd(j, compute_uv=False)[-1]
        requested = np.array([0.02, 0])
        dq, _ = damped_velocity(q, requested)
        predicted = j @ dq
        self.geometry_stats.set(
            f"J（m/rad）：\n[{j[0, 0]:+.3f}，{j[0, 1]:+.3f}]\n[{j[1, 0]:+.3f}，{j[1, 1]:+.3f}]\n最小奇异值：{sigma:.4f}\n\n希望速度 X,Y（cm/s）\n+2.00，0.00\nDLS 关节速度（rad/s）\n{dq[0]:+.3f}，{dq[1]:+.3f}\n预测末端速度（cm/s）\n{predicted[0] * 100:+.3f}，{predicted[1] * 100:+.3f}\n速度误差 {np.linalg.norm(predicted - requested) * 100:.3f} cm/s"
        )
        points = self.probe.points()
        draw_arm(self.geometry_canvas, points, q)
        c = self.geometry_canvas
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 100:
            return
        scale = (min(w, h) - 70) / 1.6
        origin = np.array([w / 2, h / 2]) + points[-1] * [scale, -scale]
        for velocity, color, width in ((requested, "#dc2626", 5), (predicted, "#16a34a", 2)):
            end = origin + 5 * velocity * [scale, -scale]
            c.create_line(*origin, *end, fill=color, width=width, arrow="last")
        c.create_text(
            12,
            84,
            anchor="nw",
            text="红：希望速度　绿：预测速度（箭头按 5 秒位移示意）",
            fill="#475569",
        )

    def refresh(self):
        super().refresh()
        self.status.set(self.status.get() + "　｜下图：偏离线段（虚线全程 / 实线已播）")
        r, i = self.replay, self.clock.index
        tip, desired = r.points[i, -1], r.desired[i]
        cross = float(segment_distance(tip, r.metadata["start_m"], r.metadata["target_m"])) * 1000
        timed = np.linalg.norm(tip - desired) * 1000
        command = (
            "已结束，无下一步力矩"
            if i == self.clock.steps
            else f"下一步力矩（N·m）\n{r.torques[i, 0]:+.4f}，{r.torques[i, 1]:+.4f}"
        )
        check = self.replay_result_text(r)
        self.stats.set(
            f"当前末端 X,Y（cm）\n{tip[0] * 100:+.2f}，{tip[1] * 100:+.2f}\n同时刻规定 X,Y（cm）\n{desired[0] * 100:+.2f}，{desired[1] * 100:+.2f}\n\n偏离线段：{cross:.3f} mm\n时间跟踪误差：{timed:.3f} mm\n全程最大偏离：{r.metadata['max_cross_track_mm']:.3f} mm\n{check}\n\n{command}"
        )
        draw_arm(
            self.scene,
            r.points[i],
            r.states[i, :2],
            r.metadata["target_m"],
            r.points[: i + 1, -1],
            r.desired[[0, -1]],
            desired,
        )
        if self.is_timing:
            movement = r.metadata["movement_s"]
            self.status.set(self.status.get() + f"　｜ 动作 {movement:g}s + 停留 3s")
            rejection = "；".join(
                f"{c['movement_s']:g}s 规划拒绝：{c['peak_reference_speed_rad_s']:.3f}>1 rad/s，未执行"
                for c in self.rejected_cases
            )
            if hasattr(self, "replay_footer"):
                self.replay_footer.configure(
                    text=(rejection or "三个时间方案均获准执行")
                    + "。曲线按物理秒对齐；0.25×只改变回放速度。"
                )
            command = "已结束，无下一步力矩"
            if i < self.clock.steps:
                requested = r.requested_torques[i]
                clipped = np.any(np.abs(requested) > 0.25 + 1e-12)
                command = (
                    f"下一步力矩（N·m）\nPD请求 {requested[0]:+.3f}，{requested[1]:+.3f}\n"
                    f"实际施加 {r.torques[i, 0]:+.3f}，{r.torques[i, 1]:+.3f}\n"
                    f"限幅：{'正在截断' if clipped else '未截断'}（±0.25）"
                )
            self.stats.set(
                f"动作时间 {movement:g}s，停留 3s\n"
                f"当前末端（cm）\n{tip[0] * 100:+.2f}，{tip[1] * 100:+.2f}\n"
                f"偏离线段 {cross:.3f} mm\n时间误差 {timed:.3f} mm\n"
                f"移动期最大偏离\n{r.metadata['max_cross_track_mm']:.3f} mm\n{check}\n\n"
                f"{command}\n全段截断 {r.metadata['clipped_steps']} 步"
            )

        if self.is_feedforward:
            self.status.set(self.status.get() + "　｜ 两方案均 4s 动作 + 3s 停留")
            if hasattr(self, "replay_footer"):
                self.replay_footer.configure(
                    text="前馈仅用预定参考和已知模型；PD读取实际状态。合计后统一限幅 ±0.25 N·m；回放倍率不改变物理。"
                )
            command = "已结束，无下一步力矩"
            if i < self.clock.steps:
                ff, fb, requested = (
                    r.feedforward_torques[i],
                    r.feedback_torques[i],
                    r.requested_torques[i],
                )
                clipped = np.any(np.abs(requested) > 0.25 + 1e-12)
                command = (
                    f"下一步力矩（肩 / 肘，N·m）\n"
                    f"模型前馈 {ff[0]:+.3f}，{ff[1]:+.3f}\n"
                    f"PD修正 {fb[0]:+.3f}，{fb[1]:+.3f}\n"
                    f"合计请求 {requested[0]:+.3f}，{requested[1]:+.3f}\n"
                    f"实际施加 {r.torques[i, 0]:+.3f}，{r.torques[i, 1]:+.3f}\n"
                    f"限幅：{'正在截断' if clipped else '未截断'}"
                )
            self.stats.set(
                f"{r.metadata['label']}\n同一 4s 动作 + 3s 停留\n"
                f"偏离线段 {cross:.3f} mm\n时间误差 {timed:.3f} mm\n"
                f"移动期最大偏离 {r.metadata['max_cross_track_mm']:.3f} mm\n{check}\n\n"
                f"{command}\n全段截断 {r.metadata['clipped_steps']} 步"
            )

    @staticmethod
    def replay_result_text(replay):
        case = replay.metadata
        if case["failure_reason"]:
            return f"整段结果：物理失败\n{case['failure_reason']}"
        if case["path_success"]:
            return "整段结果：路径与停稳通过"
        names = {
            "tip_position": "末端位置",
            "joint_position": "关节角",
            "joint_speed": "关节速度",
            "insufficient_post_movement_window": "时长不足",
        }
        if not case["endpoint_success"]:
            missed = "、".join(names[v] for v in replay.terminal_check["violations"])
            return f"整段结果：未通过\n停稳未过：{missed or '持续时间不足'}"
        return "整段结果：到达，但偏离路径"

    def draw_chart(self):
        c, selected = self.chart, self.replay
        w, h = c.winfo_width(), c.winfo_height()
        if min(w, h) < 80:
            return
        c.delete("all")
        errors = {
            r.metadata["key"]: segment_distance(
                r.points[:, -1], r.metadata["start_m"], r.metadata["target_m"]
            )
            * 1000
            for r in self.replays
        }
        peak = max(2, max(float(e.max()) for e in errors.values())) * 1.1
        duration = max((len(r.states) - 1) * r.dt for r in self.replays)
        left, right, top, bottom = 65, w - 18, 27, h - 27
        now = self.clock.index * selected.dt
        for number, (key, label, color) in enumerate(self.method_styles):
            r = next(r for r in self.replays if r.metadata["key"] == key)
            error = errors[key]
            coords = np.column_stack(
                [
                    left + (right - left) * np.arange(len(error)) * r.dt / duration,
                    bottom - (bottom - top) * error / peak,
                ]
            )
            c.create_text(
                left + number * (right - left) / len(self.method_styles),
                12,
                text=label,
                anchor="w",
                fill=color,
            )
            c.create_line(*coords.ravel(), fill=color, dash=(2, 5), width=1)
            count = min(round(now / r.dt), len(error) - 1)
            if count:
                c.create_line(
                    *coords[: count + 1].ravel(),
                    fill=color,
                    width=3 if key == selected.metadata["key"] else 2,
                )
        cursor = left + (right - left) * now / duration
        c.create_line(cursor, top, cursor, bottom, fill="#64748b", dash=(3, 3))
        c.create_line(left, top, left, bottom, right, bottom, fill="#64748b")
        c.create_text(8, 12, text="mm", anchor="w")
        for value in (0, peak / 2, peak):
            c.create_text(
                left - 8, bottom - (bottom - top) * value / peak, text=f"{value:.1f}", anchor="e"
            )
        for t in sorted({0, min(4, duration), min(8, duration), duration}):
            c.create_text(left + (right - left) * t / duration, bottom + 15, text=f"{t:g}s")
        if self.is_timing or self.is_feedforward:
            bound = bottom - (bottom - top) * 2 / peak
            c.create_line(left, bound, right, bound, fill="#94a3b8", dash=(3, 3))
            c.create_text(right, bound - 8, text="2 mm 门限", anchor="e", fill="#64748b")


def main():
    import tkinter as tk

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--speed", type=float, default=0.25)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or not 0 < args.speed <= 1:
        parser.error("speed must be in (0, 1]")
    try:
        replays = load_replays(args.results)
        report = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
        rejected = [c for c in report["cases"] if c.get("status") == "planning_rejected"]
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    root = tk.Tk()
    demo = PathDemo(root, replays, args.speed, rejected_cases=rejected)
    root.mainloop()
    del demo


if __name__ == "__main__":
    main()
