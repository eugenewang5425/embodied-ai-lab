"""Lessons 72/73: geometry audit and paired physical pick-and-place records."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.plotting import configure_plot_font
from embodied_learning.so101 import (
    ASSETS,
    CONTROL_DT,
    DT,
    HOME,
    TABLE_Z,
    SceneConfig,
    SO101Simulation,
    camera_geometry,
    grasp_rotation,
    project_point,
    verify_assets,
)

METHODS = {
    "clamp": "正常夹取：位置已知、夹爪闭合",
    "open": "空夹对照：夹爪始终张开",
    "offset": "偏差对照：抓取位置偏5.5cm",
}
COLORS = {"clamp": "#0f766e", "open": "#64748b", "offset": "#c2410c"}
PHASES = (
    "静置",
    "接近物体",
    "下降对准",
    "夹紧",
    "抬升",
    "保持",
    "搬运",
    "下放",
    "松开",
    "撤离",
    "放置检查",
)
FORMAL_POSITIONS = tuple((x, y) for x in (0.19, 0.22, 0.24) for y in (-0.04, 0.015, 0.055))


def consecutive_duration(mask, dt=CONTROL_DT):
    best = current = 0
    for value in np.asarray(mask, bool):
        current = current + 1 if value else 0
        best = max(best, current)
    # Samples at n timestamps span n-1 intervals; do not count the first as an interval.
    return max(0, best - 1) * dt


def calibration_audit():
    sim = SO101Simulation()
    rng = np.random.default_rng(72)
    maximum = 0.0
    cameras = []
    for _ in range(32):
        q = rng.uniform(sim.ranges[:, 0] * 0.7, sim.ranges[:, 1] * 0.7)
        sim.reset(np.r_[q, 1.1])
        analytic = np.zeros((3, sim.model.nv))
        mujoco.mj_jacSite(sim.model, sim.data, analytic, None, sim.site)
        numerical = np.empty((3, 5))
        for j in range(5):
            delta = np.eye(5)[j] * 1e-6
            numerical[:, j] = (sim.fk(q + delta)[0] - sim.fk(q - delta)[0]) / 2e-6
        maximum = max(maximum, float(np.max(np.abs(numerical - analytic[:, :5]))))
        for name in ("overhead", "wrist_cam"):
            intrinsic, rotation, origin = camera_geometry(sim, name)
            camera_point = np.array([0.02, -0.01, 0.25])
            point = origin + rotation @ camera_point
            uv, depth = project_point(sim, name, point)
            back = origin + rotation @ (np.linalg.solve(intrinsic, np.r_[uv, 1.0]) * depth)
            cameras.append(float(np.linalg.norm(back - point)))
    sim.reset()
    cases = []
    for key, xyz, tilt in (
        ("table_pick", [0.22, 0.025, TABLE_Z + 0.023], 0.0),
        ("vertical_lift", [0.22, 0.025, TABLE_Z + 0.143], 0.0),
        ("tilted_lift", [0.22, 0.025, TABLE_Z + 0.143], -0.45),
        ("outside_workspace", [0.8, 0.025, TABLE_Z + 0.023], 0.0),
    ):
        result = sim.ik(xyz, grasp_rotation(xyz, tilt), HOME[:5])
        cases.append(
            {
                "key": key,
                "target_xyz_m": xyz,
                "tilt_rad": tilt,
                **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()},
            }
        )
    return {
        "jacobian_poses": 32,
        "max_jacobian_error_m_per_rad": maximum,
        "camera_roundtrips": len(cameras),
        "max_camera_roundtrip_error_m": max(cameras),
        "camera_note": "模型内坐标/投影一致性检查；不是实机标定或视觉识别精度",
        "ik_cases": cases,
    }


def plan_episode(sim, method):
    if method not in METHODS:
        raise ValueError("Unknown comparison method")
    config = sim.config
    grasp = np.array([*config.object_xyz[:2], TABLE_Z + config.object_half_size + 0.003])
    if method == "offset":
        grasp[1] += 0.055
    drop = np.array([*config.target_xy, grasp[2]])
    closed = 1.1 if method == "open" else 0.0
    targets = (
        (1, grasp + [0, 0, 0.12], 1.1, 2.0, -0.45),
        (2, grasp, 1.1, 2.0, 0.0),
        (3, grasp, closed, 3.0, 0.0),
        (4, grasp + [0, 0, 0.12], closed, 2.0, -0.45),
        (5, grasp + [0, 0, 0.12], closed, 1.2, -0.45),
        (6, drop + [0, 0, 0.12], closed, 2.0, -0.45),
        (7, drop, closed, 2.0, 0.0),
        (8, drop, 1.1, 3.0, 0.0),
        (9, drop + [0, 0, 0.12], 1.1, 2.0, -0.45),
        (10, drop + [0, 0, 0.12], 1.1, 1.0, -0.45),
    )
    q, result = HOME.copy(), []
    for phase, xyz, opening, duration, tilt in targets:
        ik = sim.ik(xyz, grasp_rotation(xyz, tilt), q[:5])
        row = {
            "phase": phase,
            "xyz": np.asarray(xyz),
            "duration": duration,
            "target": np.r_[ik["q"], opening],
            "ik": ik,
        }
        result.append(row)
        if not ik["reachable"]:
            return result, f"{PHASES[phase]}的目标姿态不可达"
        q = row["target"]
    return result, ""


def score_episode(arrays, config, rejected=""):
    phase = arrays["phase"]
    settled = np.flatnonzero(phase == 0)
    baseline = float(arrays["object_xyz"][settled[-1], 2])
    lift = arrays["object_xyz"][:, 2] - baseline
    forces = arrays["jaw_normal_n"]
    lifted = (
        (lift >= 0.1) & (phase == 5) & np.all(forces[:, :2] > 0.02, axis=1) & (forces[:, 2] < 0.02)
    )
    hold = consecutive_duration(lifted)
    xy_error = np.linalg.norm(arrays["object_xyz"][:, :2] - config.target_xy, axis=1)
    speed = np.linalg.norm(arrays["qvel"][:, 6:9], axis=1)
    spin = np.linalg.norm(arrays["qvel"][:, 9:12], axis=1)
    release = (
        (phase == 10)
        & (xy_error <= 0.025)
        & (np.abs(arrays["object_xyz"][:, 2] - TABLE_Z - config.object_half_size) <= 0.003)
        & (speed < 0.02)
        & (spin < 0.2)
        & np.all(forces[:, :2] < 0.02, axis=1)
        & (arrays["qpos"][:, 5] > 0.9)
    )
    placement = consecutive_duration(release)
    lifted_ok, placed_ok = hold >= 1.0 - 1e-9, placement >= 0.5 - 1e-9
    return {
        "success": bool(not rejected and lifted_ok and placed_ok),
        "stable_lift": bool(lifted_ok),
        "stable_placement": bool(placed_ok),
        "lift_hold_s": hold,
        "placement_hold_s": placement,
        "max_lift_m": float(max(0.0, np.max(lift[phase > 0]))) if np.any(phase > 0) else 0.0,
        "final_xy_error_m": float(xy_error[-1]),
        "peak_jaw_normal_n": arrays["physics_jaw_n"][:, :2].max(axis=0).tolist(),
        "peak_other_arm_contact_n": float(arrays["physics_other_contact_n"].max()),
        "max_penetration_m": float(arrays["physics_penetration_m"].max()),
        "sim_duration_s": float(arrays["timestamps"][-1]),
        "plan_rejected": rejected,
    }


def run_episode(config=None, method="clamp"):
    config = SceneConfig() if config is None else config
    started = time.perf_counter()
    sim = SO101Simulation(config)
    plan, rejected = plan_episode(sim, method)
    frames, commands, waypoints, phase_ids, raw = [], [], [], [], []

    def append(command, phase, waypoint):
        obs = sim.observe()
        frames.append(obs)
        commands.append(command.copy())
        waypoints.append(np.asarray(waypoint).copy())
        phase_ids.append(phase)

    q = HOME.copy()
    append(q, 0, sim.fk(q[:5])[0])
    # Physical settling is part of the archived input, not a qpos teleport.
    for _ in range(round(1.0 / CONTROL_DT)):
        sim.step(q)
        raw.extend(sim.last_physics)
        append(q, 0, sim.fk(q[:5])[0])
    if not rejected:
        for row in plan:
            start, end = q.copy(), row["target"]
            n = round(row["duration"] / CONTROL_DT)
            for j in range(1, n + 1):
                t = j / n
                smooth = t**3 * (10 - 15 * t + 6 * t * t)
                q = start + (end - start) * smooth
                sim.step(q)
                raw.extend(sim.last_physics)
                append(q, row["phase"], row["xyz"])
    arrays = {
        "timestamps": np.array([v["time"] for v in frames]),
        "qpos": np.array([v["qpos"] for v in frames]),
        "qvel": np.array([v["qvel"] for v in frames]),
        "tcp_xyz": np.array([v["tcp"] for v in frames]),
        "object_xyz": np.array([v["object"] for v in frames]),
        "jaw_normal_n": np.array([v["jaw_normal_n"] for v in frames]),
        "commands": np.array(commands),
        "waypoint_xyz": np.array(waypoints),
        "phase": np.array(phase_ids, dtype=np.int16),
        "physics_time": np.array([v[0] for v in raw]),
        "physics_qpos": np.array([v[1] for v in raw]),
        "physics_qvel": np.array([v[2] for v in raw]),
        "physics_jaw_n": np.array([v[3] for v in raw]),
        "physics_other_contact_n": np.array([v[4] for v in raw]),
        "physics_penetration_m": np.array([v[5] for v in raw]),
    }
    report = {
        "method": method,
        "label": METHODS[method],
        "config": asdict(config),
        **score_episode(arrays, config, rejected),
        "wall_s": time.perf_counter() - started,
        "plan": [
            {
                "phase": PHASES[v["phase"]],
                "xyz_m": v["xyz"].tolist(),
                "q_target_rad": v["target"].tolist(),
                "position_error_m": v["ik"]["position_error_m"],
                "orientation_error_rad": v["ik"]["orientation_error_rad"],
            }
            for v in plan
        ],
    }
    return report, arrays


def make_figures(report, arrays, destination):
    configure_plot_font()
    destination.mkdir(parents=True, exist_ok=True)
    audit = report["calibration"]
    labels = ["桌面抓取", "竖直抬升", "倾转抬升", "远处目标"]
    cases = audit["ik_cases"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric, scale, title, unit in (
        (axes[0], "position_error_m", 1000.0, "目标位置与逆解末端的差", "位置残差 / mm"),
        (axes[1], "orientation_error_rad", 180 / np.pi, "要求方向与逆解方向的差", "方向残差 / °"),
    ):
        values = [v[metric] * scale for v in cases]
        bars = axis.bar(
            labels, values, color=["#0f766e" if v["reachable"] else "#c2410c" for v in cases]
        )
        axis.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.2))
        for bar, value in zip(bars, values, strict=True):
            text = "近零" if value < 0.001 else f"{value:.2f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2, value + max(values) * 0.025, text, ha="center"
            )
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination / "lesson72-geometry-audit.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for axis, metric, heading in zip(
        axes,
        ("stable_lift", "stable_placement", "success"),
        ("抬高≥10cm并保持1s", "松开后稳定放置", "两项同时成立"),
        strict=True,
    ):
        counts = [
            sum(row[metric] for row in report["episodes"] if row["method"] == method)
            for method in METHODS
        ]
        totals = [sum(row["method"] == method for row in report["episodes"]) for method in METHODS]
        bars = axis.bar(
            ["正常夹取", "夹爪张开", "位置偏差"], counts, color=[COLORS[k] for k in METHODS]
        )
        axis.set_ylim(0, max(totals) * 1.3)
        axis.set_title(heading)
        axis.set_ylabel("成功回合数")
        for bar, count, total in zip(bars, counts, totals, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2, count + 0.1, f"{count}/{total}", ha="center"
            )
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination / "lesson73-grasp-results.png", dpi=160)
    plt.close(figure)
    # Fixed representative: middle position, all arms, no outcome-based selection.
    position = "p04" if len(report["positions_xy_m"]) > 1 else "p00"
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
    for method in METHODS:
        key = f"{position}_{method}"
        a = arrays[key]
        baseline = a["object_xyz"][np.flatnonzero(a["phase"] == 0)[-1], 2]
        axes[0].plot(
            a["timestamps"],
            (a["object_xyz"][:, 2] - baseline) * 100,
            label=METHODS[method].split("：")[0],
            color=COLORS[method],
        )
        axes[1].plot(a["object_xyz"][:, 0], a["object_xyz"][:, 1], color=COLORS[method])
    a = arrays[f"{position}_clamp"]
    axes[2].plot(a["timestamps"], a["jaw_normal_n"][:, 0], label="固定指接触力", color="#2563eb")
    axes[2].plot(a["timestamps"], a["jaw_normal_n"][:, 1], label="活动指接触力", color="#c2410c")
    axes[0].axhline(10, color="#475569", ls="--", label="抬升门槛10cm")
    axes[0].set(xlabel="仿真时间 / s", ylabel="物体抬升 / cm", title="抬升和放下是不同阶段")
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8, frameon=False
    )
    goal = report["episodes"][0]["config"]["target_xy"]
    axes[1].scatter(*goal, marker="x", color="#111827", s=80, label="放置目标")
    axes[1].set(xlabel="世界X / m", ylabel="世界Y / m", title="物体实际轨迹（俯视）")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=8, frameon=False)
    axes[2].set(xlabel="仿真时间 / s", ylabel="每侧法向力之和 / N", title="正常夹取：两侧力与释放")
    axes[2].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=8, frameon=False
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(destination / "lesson73-grasp-traces.png", dpi=160)
    plt.close(figure)


def collect(output: Path, smoke=False, figures: Path | None = None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    manifest = verify_assets()
    audit = calibration_audit()
    positions = ((0.22, 0.025),) if smoke else FORMAL_POSITIONS
    report = {
        "experiment": "so101_physical_grasping",
        "schema_version": 1,
        "date": "2026-10-01",
        "model_commit": manifest["commit"],
        "model_manifest_sha256": hashlib.sha256(
            (ASSETS / "provenance.json").read_bytes()
        ).hexdigest(),
        "physics_dt_s": DT,
        "control_dt_s": CONTROL_DT,
        "phases": PHASES,
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "positions_xy_m": positions,
        "calibration": audit,
        "episodes": [],
        "sensor_note": "相机按实际存档状态重渲染；控制使用已知物体位置，没有视觉识别",
        "scoring_note": "最大抬升仅统计初始化静置结束后；初始落到桌面不算抬升。稳定抬升和放置按专门阶段评分。",
        "force_timing": "physics_time=t表示mj_step结束状态；对应力在该次[t-dt,t]求解中产生；控制帧力为mj_forward瞬时诊断",
        "simulation_note": "固定底座、理想关节读数、简化碰撞体、假设摩擦和夹爪限矩；没有实机标定",
        "protocol": "docs/72-73-manipulation-plan.md",
    }
    combined, by_key = {}, {}
    started = time.perf_counter()
    for i, (x, y) in enumerate(positions):
        config = SceneConfig(object_xyz=(x, y, TABLE_Z + 0.025))
        for method in METHODS:
            row, values = run_episode(config, method)
            key = f"p{i:02}_{method}"
            row["key"], row["position_key"] = key, f"p{i:02}"
            report["episodes"].append(row)
            by_key[key] = values
            combined.update({f"{key}__{name}": v for name, v in values.items()})
            print(
                f"{key}: lift={row['stable_lift']} placed={row['stable_placement']} "
                f"success={row['success']} ({row['wall_s']:.2f}s)",
                flush=True,
            )
    report["wall_s"] = time.perf_counter() - started
    output.mkdir(parents=True)
    np.savez_compressed(output / "trajectories.npz", **combined)
    report["archive_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    source = output / "source"
    source.mkdir()
    report["source_sha256"] = {}
    for path in (Path(__file__), Path(__file__).parents[1] / "so101.py"):
        data = path.read_bytes()
        (source / path.name).write_bytes(data)
        report["source_sha256"][path.name] = hashlib.sha256(data).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_figures(report, by_key, figures or output)
    print(f"Saved {len(report['episodes'])} episodes: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    collect(args.output, args.smoke, args.figures)


if __name__ == "__main__":
    main()
