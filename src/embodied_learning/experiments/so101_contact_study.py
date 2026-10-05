"""Lesson 73 round three: surface alignment x causal bilateral-contact gate."""

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

from embodied_learning.experiments.so101_approach_study import run_episode as baseline_episode
from embodied_learning.experiments.so101_grasping import PHASES, plan_episode, score_episode
from embodied_learning.gripper_calibration import FingerMidpointCalibration
from embodied_learning.gripper_contact import BilateralContactGate, SurfaceBandCalibration
from embodied_learning.so101 import (
    ASSETS,
    CONTROL_DT,
    DT,
    HOME,
    TABLE_Z,
    SceneConfig,
    SO101Simulation,
    verify_assets,
)

METHODS = {
    "baseline": "A 指尖校准：第二轮基线",
    "surface": "B 指面校准：只换接触参考点",
    "gate": "C 接触门控：基线＋抬升前双侧力检查",
    "surface_gate": "D 指面＋门控：两项同时使用",
    "open": "空夹：指面校准但不闭合",
    "offset": "偏位：指面校准但偏5.5cm",
}
COLORS = dict(
    zip(METHODS, ("#2563eb", "#0f766e", "#a855f7", "#d97706", "#64748b", "#c2410c"), strict=True)
)
FORMAL_POSITIONS = tuple((x, y) for x in (0.21, 0.23, 0.245) for y in (-0.035, 0.025, 0.065))
FORMAL_YAWS = (-20.0, 5.0, 20.0)
DEVELOPMENT = (
    ("p00a2", (0.205, -0.025), 15),
    ("p02a0", (0.205, 0.070), -15),
    ("p03a2", (0.225, -0.025), 15),
    ("p05a0", (0.225, 0.070), -15),
    ("p06a2", (0.235, -0.025), 15),
    ("p08a0", (0.235, 0.070), -15),
    ("p04a1", (0.225, 0.035), 0),
    ("p00a1", (0.205, -0.025), 0),
)
ROOT = Path(__file__).resolve().parents[3]
SOURCES = (
    Path(__file__),
    ROOT / "src/embodied_learning/gripper_contact.py",
    ROOT / "src/embodied_learning/gripper_calibration.py",
    ROOT / "src/embodied_learning/experiments/so101_approach_study.py",
    ROOT / "src/embodied_learning/experiments/so101_grasping.py",
    ROOT / "src/embodied_learning/so101.py",
)


def run_episode(config, method, yaw_deg=0.0, surface_parameters=None):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if method == "baseline":
        report, arrays = baseline_episode(config, "center_cartesian", yaw_deg)
        report.update(method=method, label=METHODS[method], contact_refused=False)
        return report, arrays
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
    plan, rejected = plan_episode(sim, method if method in ("open", "offset") else "clamp")
    calibration = (
        FingerMidpointCalibration(sim)
        if method == "gate"
        else SurfaceBandCalibration(sim, **(surface_parameters or {}))
    )
    gate = BilateralContactGate()
    use_gate = method in ("gate", "surface_gate")
    refused = False
    frames, commands, waypoints, phases, raw, gate_elapsed, solved_min = [], [], [], [], [], [], []

    def append(command, phase, waypoint):
        frames.append(sim.observe())
        commands.append(command.copy())
        waypoints.append(np.asarray(waypoint).copy())
        phases.append(phase)
        gate_elapsed.append(gate.elapsed)
        solved_min.append(
            np.min([v[3][:2] for v in sim.last_physics], axis=0)
            if sim.last_physics
            else np.zeros(2)
        )

    q = HOME.copy()
    append(q, 0, sim.fk(q[:5])[0])
    for _ in range(round(1.0 / CONTROL_DT)):
        sim.step(q)
        raw.extend(sim.last_physics)
        append(q, 0, sim.fk(q[:5])[0])
    nominal_q = HOME.copy()
    previous_xyz, previous_tilt, last_arm = sim.fk(HOME[:5])[0], -0.45, HOME[:5].copy()
    if not rejected:
        for row in plan:
            if row["phase"] == 4 and use_gate and not gate.ready:
                rejected, refused = "抬升前未形成持续0.2s的双侧接触", True
                break
            start, end = nominal_q.copy(), row["target"]
            tilt = 0.0 if row["phase"] in (2, 3, 7, 8) else -0.45
            n = round(row["duration"] / CONTROL_DT)
            for j in range(1, n + 1):
                t = j / n
                smooth = t**3 * (10 - 15 * t + 6 * t * t)
                nominal_q = start + (end - start) * smooth
                q = nominal_q.copy()
                if row["phase"] > 1:
                    xyz = previous_xyz + (row["xyz"] - previous_xyz) * smooth
                    pitch = previous_tilt + (tilt - previous_tilt) * smooth
                    xyz, rotation = calibration.tcp_target(xyz, sim.data.qpos[5], pitch)
                    ik = sim.ik(xyz, rotation, last_arm)
                    if not ik["reachable"]:
                        rejected = f"{PHASES[row['phase']]}在线目标不可达"
                        break
                    q[:5] = ik["q"]
                sim.step(q)
                raw.extend(sim.last_physics)
                if row["phase"] == 3:
                    gate.update(np.array([v[3][:2] for v in sim.last_physics]), CONTROL_DT)
                append(q, row["phase"], row["xyz"])
                last_arm = q[:5].copy()
            if rejected:
                break
            previous_xyz, previous_tilt = row["xyz"].copy(), tilt
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
        "gate_elapsed_s": np.array(gate_elapsed),
        "solved_jaw_min_n": np.array(solved_min),
    }
    report = {
        "method": method,
        "label": METHODS[method],
        "config": asdict(config),
        "object_yaw_deg": yaw_deg,
        "contact_refused": refused,
        "surface_parameters": surface_parameters or {"band_x": 0.0, "preparation_opening": 0.45},
        **score_episode(arrays, config, rejected),
        "wall_s": time.perf_counter() - started,
    }
    return report, arrays


def roster(development=False):
    if development:
        return [(p, xy, yaw, m) for p, xy, yaw in DEVELOPMENT for m in tuple(METHODS)[:4]]
    return [
        (f"p{i:02}a{k}", xy, yaw, method)
        for i, xy in enumerate(FORMAL_POSITIONS)
        for k, yaw in enumerate(FORMAL_YAWS)
        for method in (tuple(METHODS) if i in (3, 4, 5) else tuple(METHODS)[:4])
    ]


def _job(case):
    position, xy, yaw, method = case
    row, arrays = run_episode(SceneConfig(object_xyz=(*xy, TABLE_Z + 0.025)), method, yaw)
    row.update(key=f"{position}_{method}", position_key=position)
    return row, arrays


def collect(output, development=False, workers=3):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    cases = roster(development)
    manifest = verify_assets()
    report = {
        "experiment": "so101_physical_grasping",
        "schema_version": 1,
        "study_round": 3,
        "development": development,
        "date": "2026-10-05",
        "model_commit": manifest["commit"],
        "model_manifest_sha256": hashlib.sha256(
            (ASSETS / "provenance.json").read_bytes()
        ).hexdigest(),
        "physics_dt_s": DT,
        "control_dt_s": CONTROL_DT,
        "phases": PHASES,
        "runtime": {"mujoco": mujoco.__version__, "numpy": np.__version__},
        "positions_xy_m": sorted({xy for _, xy, _, _ in cases}),
        "method_labels": METHODS,
        "method_colors": COLORS,
        "episodes": [],
        "cases": cases,
        "preferred_method": "surface",
        "sensor_note": "已知初始位置；仅实时自身开度与过去子步接触力参与候选控制，RGB未参与控制",
        "simulation_note": "只在初始设置物体朝向；原模型、限矩、质量、摩擦、阶段时长和评分不变",
        "force_timing": "physics_time=t对应[t-dt,t]求解力；控制帧为mj_forward瞬时诊断",
        "protocol": "docs/73-contact-round3-plan.md",
        "calibration": {"band_x_m": 0.0, "preparation_opening_rad": 0.45, "samples": 131},
        "contact_gate": {
            "threshold_n": 0.05,
            "duration_s": 0.2,
            "phase": 3,
            "decision_before_phase": 4,
        },
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
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
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
                f"{len(report['episodes'])}/{len(cases)} {row['key']}: success={row['success']} lift={row['max_lift_m']:.3f} {row['plan_rejected']}",
                flush=True,
            )
    report["episodes"].sort(key=lambda r: r["key"])
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
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    collect(args.output, args.development, args.workers)


if __name__ == "__main__":
    main()
