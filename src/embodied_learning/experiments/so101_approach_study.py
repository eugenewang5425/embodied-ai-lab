"""Lesson 73 round two: paired approach paths and finger-midpoint calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from embodied_learning.experiments.so101_grasping import PHASES, plan_episode, score_episode
from embodied_learning.gripper_calibration import FingerMidpointCalibration
from embodied_learning.so101 import (
    ASSETS,
    CONTROL_DT,
    DT,
    HOME,
    TABLE_Z,
    SceneConfig,
    SO101Simulation,
    grasp_rotation,
    verify_assets,
)

METHODS = {
    "joint": "原策略：关节插值、TCP对准",
    "cartesian": "空间路径：位置与倾角插值、TCP对准",
    "center_joint": "关节位置路径＋实际开口中心",
    "center_cartesian": "空间直线＋实际开口中心",
    "open": "空夹对照：校准后仍不闭合",
    "offset": "偏差对照：校准后仍偏5.5cm",
}
COLORS = {
    "joint": "#2563eb",
    "cartesian": "#a855f7",
    "center_joint": "#d97706",
    "center_cartesian": "#0f766e",
    "open": "#64748b",
    "offset": "#c2410c",
}
FORMAL_POSITIONS = tuple((x, y) for x in (0.205, 0.225, 0.235) for y in (-0.025, 0.035, 0.07))
FORMAL_YAWS = (-15.0, 0.0, 15.0)
ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    Path(__file__),
    ROOT / "src/embodied_learning/gripper_calibration.py",
    ROOT / "src/embodied_learning/experiments/so101_grasping.py",
    ROOT / "src/embodied_learning/so101.py",
)


def run_episode(config, method, yaw_deg=0.0):
    started = time.perf_counter()
    sim = SO101Simulation(config)
    yaw = np.deg2rad(yaw_deg)
    sim.data.qpos[sim.object_qadr + 3 : sim.object_qadr + 7] = [
        np.cos(yaw / 2),
        0,
        0,
        np.sin(yaw / 2),
    ]
    mujoco.mj_forward(sim.model, sim.data)
    base = method if method in ("open", "offset") else "clamp"
    plan, rejected = plan_episode(sim, base)
    calibration = FingerMidpointCalibration(sim)
    centered = method.startswith("center") or method in ("open", "offset")
    cartesian = method in ("cartesian", "center_cartesian", "open", "offset")
    frames, commands, waypoints, phases, raw = [], [], [], [], []

    def append(command, phase, waypoint):
        frames.append(sim.observe())
        commands.append(command.copy())
        waypoints.append(np.asarray(waypoint).copy())
        phases.append(phase)

    q = HOME.copy()
    append(q, 0, sim.fk(q[:5])[0])
    for _ in range(round(1.0 / CONTROL_DT)):
        sim.step(q)
        raw.extend(sim.last_physics)
        append(q, 0, sim.fk(q[:5])[0])
    nominal_q = HOME.copy()
    previous_xyz = sim.fk(HOME[:5])[0]
    previous_tilt = -0.45
    last_arm = HOME[:5].copy()
    if not rejected:
        for row in plan:
            start, end = nominal_q.copy(), row["target"]
            tilt = 0.0 if row["phase"] in (2, 3, 7, 8) else -0.45
            n = round(row["duration"] / CONTROL_DT)
            for j in range(1, n + 1):
                t = j / n
                smooth = t**3 * (10 - 15 * t + 6 * t * t)
                nominal_q = start + (end - start) * smooth
                q = nominal_q.copy()
                # Approach is still the original joint path. Remaining phases
                # separate Cartesian path geometry from midpoint alignment.
                if row["phase"] > 1 and (cartesian or centered):
                    if cartesian:
                        xyz = previous_xyz + (row["xyz"] - previous_xyz) * smooth
                    else:
                        xyz = sim.fk(nominal_q[:5])[0]
                    pitch = previous_tilt + (tilt - previous_tilt) * smooth
                    rotation = grasp_rotation(xyz, pitch)
                    if centered:
                        xyz, rotation = calibration.tcp_target(xyz, sim.data.qpos[5], pitch)
                    ik = sim.ik(xyz, rotation, last_arm)
                    if not ik["reachable"]:
                        rejected = f"{PHASES[row['phase']]}在线目标不可达"
                        break
                    q[:5] = ik["q"]
                sim.step(q)
                raw.extend(sim.last_physics)
                append(q, row["phase"], row["xyz"])
                last_arm = q[:5].copy()
            if rejected:
                break
            previous_xyz = row["xyz"].copy()
            previous_tilt = tilt
    arrays = {
        "timestamps": np.array([v["time"] for v in frames]),
        "qpos": np.array([v["qpos"] for v in frames]),
        "qvel": np.array([v["qvel"] for v in frames]),
        "tcp_xyz": np.array([v["tcp"] for v in frames]),
        "object_xyz": np.array([v["object"] for v in frames]),
        "jaw_normal_n": np.array([v["jaw_normal_n"] for v in frames]),
        "commands": np.array(commands),
        "waypoint_xyz": np.array(waypoints),
        "phase": np.array(phases, dtype=np.int16),
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
        "object_yaw_deg": yaw_deg,
        **score_episode(arrays, config, rejected),
        "wall_s": time.perf_counter() - started,
    }
    return report, arrays


def roster(smoke):
    if smoke:
        return [
            (f"p{i:02}", xy, 0.0, method)
            for i, xy in enumerate(((0.22, 0.015), (0.22, 0.055)))
            for method in tuple(METHODS)[:4]
        ]
    rows = []
    for i, xy in enumerate(FORMAL_POSITIONS):
        for k, yaw in enumerate(FORMAL_YAWS):
            methods = ["joint", "center_cartesian"]
            if i in (3, 4, 5):
                methods += ["open", "offset"]
            rows.extend((f"p{i:02}a{k}", xy, yaw, method) for method in methods)
    return rows


def _job(case):
    position, xy, yaw, method = case
    config = SceneConfig(object_xyz=(*xy, TABLE_Z + 0.025))
    row, arrays = run_episode(config, method, yaw)
    row.update(key=f"{position}_{method}", position_key=position)
    return row, arrays


def collect(output, smoke=False, workers=3):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    cases = roster(smoke)
    manifest = verify_assets()
    report = {
        "experiment": "so101_physical_grasping",
        "schema_version": 1,
        "study_round": 2,
        "development": smoke,
        "date": "2026-10-01",
        "model_commit": manifest["commit"],
        "model_manifest_sha256": hashlib.sha256(
            (ASSETS / "provenance.json").read_bytes()
        ).hexdigest(),
        "physics_dt_s": DT,
        "control_dt_s": CONTROL_DT,
        "phases": PHASES,
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "positions_xy_m": FORMAL_POSITIONS if not smoke else ((0.22, 0.015), (0.22, 0.055)),
        "method_labels": METHODS,
        "method_colors": COLORS,
        "episodes": [],
        "sensor_note": "已知初始物体位置；关节开度反馈来自当前帧，RGB未参与控制",
        "simulation_note": "物体仅初始化设置朝向；之后只由mj_step产生运动。原限矩、质量、摩擦、模型与评分不变。",
        "force_timing": "physics_time=t对应[t-dt,t]求解力；控制帧为mj_forward瞬时诊断",
        "protocol": "docs/73-approach-round2-plan.md",
        "cases": cases,
    }
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    report["source_sha256"] = {}
    for path in (*SOURCES, ROOT / report["protocol"]):
        data = path.read_bytes()
        (source / path.name).write_bytes(data)
        report["source_sha256"][path.name] = hashlib.sha256(data).hexdigest()
    (output / "registration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    combined = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, case) for case in cases]
        for future in as_completed(futures):
            row, arrays = future.result()
            report["episodes"].append(row)
            combined.update({f"{row['key']}__{k}": v for k, v in arrays.items()})
            np.savez_compressed(output / f"{row['key']}.npz", **arrays)
            print(
                f"{len(report['episodes'])}/{len(cases)} {row['key']}: success={row['success']} lift={row['max_lift_m']:.3f} place={row['final_xy_error_m']:.3f} other={row['peak_other_arm_contact_n']:.2f} {row['plan_rejected']}",
                flush=True,
            )
    report["episodes"].sort(key=lambda row: row["key"])
    report["wall_s"] = time.perf_counter() - started
    np.savez_compressed(output / "trajectories.npz", **combined)
    report["archive_sha256"] = hashlib.sha256(
        (output / "trajectories.npz").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    collect(args.output, args.smoke, args.workers)


if __name__ == "__main__":
    main()
