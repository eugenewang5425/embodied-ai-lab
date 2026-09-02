"""Lesson 12: retime the frozen lesson11 paths; no controller or motor changes."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.arm_path import (
    HOLD_SECONDS,
    REFERENCE_SPEED_LIMIT,
    TIMINGS,
    ReferenceSpeedError,
)
from embodied_learning.experiments.arm_ik_comparison import load_source
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.planar_arm import KD, KP, LENGTHS, MODEL_PATH, TORQUE_LIMIT
from embodied_learning.plotting import configure_plot_font


def timing_diagnostics(arrays, case):
    dt, movement = case["dt_s"], case["movement_s"]
    requested, applied = arrays["requested_torques_nm"], arrays["torques_nm"]
    clipped = np.any(np.abs(requested) > TORQUE_LIMIT + 1e-12, axis=1)
    active = np.arange(len(applied)) * dt < movement - 1e-10
    end_index = round(movement / dt)
    return {
        "peak_reference_speed_rad_s": float(np.max(np.abs(arrays["dq_reference"]))),
        "peak_requested_pd_torque_nm": np.max(np.abs(requested), axis=0).tolist(),
        "clipped_steps": int(np.count_nonzero(clipped)),
        "clipped_movement_steps": int(np.count_nonzero(clipped & active)),
        "clipped_movement_fraction": float(np.mean(clipped[active])),
        "clipped_duration_s": float(np.count_nonzero(clipped) * dt),
        "max_command_clipping_nm": float(np.max(np.abs(requested - applied))),
        "tip_error_at_movement_end_mm": (
            float(np.linalg.norm(arrays["points"][end_index, -1] - case["target_m"]) * 1000)
            if end_index < len(arrays["states"])
            else None
        ),
    }


def verify_eight_second_baseline(source, arrays):
    with np.load(source / "trajectories.npz", allow_pickle=False) as old:
        for name, value in arrays.items():
            if not np.array_equal(value, old[f"waypoint_ik_{name}"]):
                raise ValueError(f"8s lesson11 baseline changed: {source.name}/{name}")


def run_trial(specification, source, output):
    cases, archive = [], {}
    for key, movement, color in TIMINGS:
        common = {
            "key": key,
            "label": f"{movement:g} s 完成动作",
            "plot_color": color,
            "method": "waypoint_ik",
            "movement_s": movement,
            "hold_s": HOLD_SECONDS,
            "lesson": 12,
            "trial": {"id": specification["id"], "group": specification["group"]},
        }
        try:
            arrays, case = run_path(
                "waypoint_ik",
                initial_q=specification["initial_q_rad"],
                target=specification["target_m"],
                move_seconds=movement,
            )
        except ReferenceSpeedError as exc:
            cases.append(
                {
                    **common,
                    "status": "planning_rejected",
                    "reason": str(exc),
                    "peak_reference_speed_rad_s": exc.peak_rad_s,
                }
            )
            continue
        if movement == 8:
            verify_eight_second_baseline(source, arrays)
        case.update(timing_diagnostics(arrays, case))
        case.update(common)
        case["status"] = "executed"
        case["baseline_identical"] = True if movement == 8 else None
        cases.append(case)
        archive.update({f"{key}_{name}": value for name, value in arrays.items()})
    report = {
        "experiment": "planar_2r_timing",
        "schema_version": 1,
        "lengths_m": LENGTHS.tolist(),
        "cases": cases,
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "hold_s": HOLD_SECONDS,
        "reference_speed_limit_rad_s": REFERENCE_SPEED_LIMIT,
        "protocol": "Only move duration changes: 8,4,2s. Same waypoint IK, dt=0.02s, PD, motors and 3s hold. Path<=2mm over [0,T], final continuous 0.5s after T with original tip/joint/speed limits. End-of-movement error separately reported; path_success is not an on-time arrival guarantee.",
        "rejection": "Plans above 1rad/s are recorded without executing physics or fabricating replay arrays. Not counted as simulated motor failures.",
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return cases


def aggregate(trials):
    groups = {}
    for group in ("interior", "near_extension", "diagnostic", "all"):
        selected = [t for t in trials if group == "all" or t["group"] == group]
        groups[group] = {}
        for key, _, _ in TIMINGS:
            cases = [next(c for c in t["cases"] if c["key"] == key) for t in selected]
            executed = [c for c in cases if c["status"] == "executed"]
            groups[group][key] = {
                "planned": len(cases),
                "planning_rejected": len(cases) - len(executed),
                "executed": len(executed),
                "completed": sum(c["completed"] for c in executed),
                "path_successes": sum(c["path_success"] for c in executed),
                "endpoint_successes": sum(c["endpoint_success"] for c in executed),
                "physical_failures": sum(bool(c["failure_reason"]) for c in executed),
                "episodes_with_clipping": sum(c["clipped_steps"] > 0 for c in executed),
            }
    return groups


def save_overview(path, trials, groups):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    passed, rejected, cross, torque = axes.ravel()
    group_names = (("interior", "内部"), ("near_extension", "近伸直"), ("diagnostic", "固定伸直"))
    x = np.arange(3)
    for i, (key, movement, color) in enumerate(TIMINGS):
        stats = [groups[group][key] for group, _ in group_names]
        for ax, field in ((passed, "path_successes"), (rejected, "planning_rejected")):
            bars = ax.bar(
                x + (i - 1) * 0.25,
                [s[field] / s["planned"] * 100 for s in stats],
                width=0.23,
                color=color,
                label=f"{movement:g} s",
            )
            ax.bar_label(bars, labels=[f"{s[field]}/{s['planned']}" for s in stats], fontsize=8)
            ax.set(xticks=x, xticklabels=[name for _, name in group_names], ylim=(0, 120))
        for trial in trials:
            case = next(c for c in trial["cases"] if c["key"] == key)
            if case["status"] == "executed":
                cross.scatter(movement, case["max_cross_track_mm"], color=color, s=20, alpha=0.7)
    passed.set(title="路径与停稳通过 / 全部计划", ylabel="%（已知路径集）")
    rejected.set(title="规划速度超限：没有执行物理", ylabel="拒绝比例 %")
    passed.legend()
    cross.axhline(2, color="gray", linestyle=":", label="2 mm 验收线")
    cross.set(
        title="每个点是一条已执行路径",
        xlabel="动作时长（s）",
        ylabel="移动期最大偏离（mm）",
        xticks=[2, 4, 8],
    )
    cross.legend()
    # Fixed first interior path, chosen independently of outcomes.
    first = next(t for t in trials if t["group"] == "interior")
    executed = [c for c in first["cases"] if c["status"] == "executed"]
    fastest = min(executed, key=lambda c: c["movement_s"])
    with np.load(
        path.parent / first["relative_results"] / "trajectories.npz", allow_pickle=False
    ) as data:
        key = fastest["key"]
        times = np.arange(fastest["steps"] + 1) * fastest["dt_s"]
        for j, color in enumerate(("#2563eb", "#ea580c")):
            torque.stairs(
                data[f"{key}_requested_torques_nm"][:, j],
                times,
                color=color,
                linestyle="--",
                label=f"关节 {j + 1} PD 请求",
            )
            torque.stairs(
                data[f"{key}_torques_nm"][:, j], times, color=color, label=f"关节 {j + 1} 实际施加"
            )
    for bound in (-TORQUE_LIMIT, TORQUE_LIMIT):
        torque.axhline(bound, color="gray", linestyle=":")
    torque.set(
        title=f"{first['id']} / {fastest['movement_s']:g}s：请求不等于施加",
        xlabel="时间（s）",
        ylabel="力矩（N·m）",
    )
    torque.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    manifest, raw, digest = load_source(source, experiment="planar_2r_waypoint_ik_comparison")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_manifest.json").write_bytes(raw)
    protocol = {
        "source_results": str(source.resolve()),
        "source_manifest_sha256": digest,
        "movement_seconds": [t for _, t, _ in TIMINGS],
        "hold_seconds": HOLD_SECONDS,
        "selection": "All source trials, unchanged order, no outcome-based filtering",
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "reference_speed_limit_rad_s": REFERENCE_SPEED_LIMIT,
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    trials = []
    for spec in manifest["trials"]:
        relative = f"trials/{spec['id']}"
        cases = run_trial(spec, source / relative, output / relative)
        trials.append({**spec, "relative_results": relative, "cases": cases})
        print(spec["id"], {c["key"]: c.get("path_success", c["status"]) for c in cases}, flush=True)
    groups = aggregate(trials)
    report = {
        "experiment": "planar_2r_timing_batch",
        "schema_version": 1,
        **protocol,
        "groups": groups,
        "trials": trials,
        "limitations": "Known-path paired regression, not unseen robustness. No payload, contact or noise. Requested PD torque is not inverse-dynamics required torque; clipping and error alone do not establish sole causation. Same 3s hold; no relaxed success criteria.",
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_overview(output / "overview.png", trials, groups)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run_experiment(args.source_results, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
