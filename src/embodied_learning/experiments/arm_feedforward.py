"""Lesson 13: nominal inverse-dynamics feedforward plus unchanged PD, all paths at 4s."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.arm_dynamics import (
    FEEDFORWARD_COLORS,
    FEEDFORWARD_METHODS,
    audit_inverse_dynamics,
    reference_acceleration,
)
from embodied_learning.arm_path import REFERENCE_SPEED_LIMIT
from embodied_learning.experiments.arm_ik_comparison import load_source
from embodied_learning.experiments.arm_path import run_path
from embodied_learning.experiments.arm_timing import timing_diagnostics
from embodied_learning.planar_arm import KD, KP, LENGTHS, MODEL_PATH, TORQUE_LIMIT
from embodied_learning.plotting import configure_plot_font

MOVEMENT_S, HOLD_S = 4.0, 3.0


def load_timing_source(source):
    manifest, raw, digest = load_source(
        source,
        experiment="planar_2r_timing_batch",
        manifest_name="source_manifest.json",
        hash_key="source_manifest_sha256",
    )
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    for key, expected in (("kp", KP), ("kd", KD)):
        if not np.array_equal(summary[key], expected):
            raise ValueError(f"Source {key} differs from the current controller")
    if (
        summary["torque_limit_nm"] != TORQUE_LIMIT
        or summary["hold_seconds"] != HOLD_S
        or summary["reference_speed_limit_rad_s"] != REFERENCE_SPEED_LIMIT
        or MOVEMENT_S not in summary["movement_seconds"]
    ):
        raise ValueError("Source timing or limits differ")
    return manifest, raw, digest


def verify_pd_baseline(source, arrays):
    with np.load(source / "trajectories.npz", allow_pickle=False) as old:
        for name, array in arrays.items():
            if not np.array_equal(array, old[f"move_4s_{name}"]):
                raise ValueError(f"4s PD baseline changed: {source.name}/{name}")


def run_trial(specification, source, output):
    archive, cases, baseline_reference = {}, [], None
    for controller, label in FEEDFORWARD_METHODS:
        arrays, case = run_path(
            "waypoint_ik",
            initial_q=specification["initial_q_rad"],
            target=specification["target_m"],
            move_seconds=MOVEMENT_S,
            controller=controller,
        )
        if controller == "pd":
            verify_pd_baseline(source, arrays)
            baseline_reference = {
                name: arrays[name].copy()
                for name in ("q_reference", "dq_reference", "desired_points")
            }
            arrays.update(
                feedback_torques_nm=arrays["requested_torques_nm"].copy(),
                feedforward_torques_nm=np.zeros_like(arrays["torques_nm"]),
                ddq_reference=reference_acceleration(arrays["dq_reference"], case["dt_s"]),
            )
        else:
            for name, expected in baseline_reference.items():
                # A genuine early physical failure may truncate the saved reference.
                if not np.array_equal(arrays[name], expected[: len(arrays[name])]):
                    raise ValueError("Paired geometric references differ")
        diagnostics = timing_diagnostics(arrays, case)
        diagnostics["peak_total_requested_torque_nm"] = diagnostics.pop(
            "peak_requested_pd_torque_nm"
        )
        active = np.arange(case["steps"]) * case["dt_s"] < MOVEMENT_S - 1e-10
        case.update(diagnostics)
        case.update(
            key=controller,
            label=label,
            lesson=13,
            status="executed",
            controller=controller,
            trial={"id": specification["id"], "group": specification["group"]},
            rms_feedback_torque_nm=np.sqrt(
                np.mean(arrays["feedback_torques_nm"][active] ** 2, axis=0)
            ).tolist(),
            peak_feedforward_torque_nm=np.max(
                np.abs(arrays["feedforward_torques_nm"]), axis=0
            ).tolist(),
            peak_reference_acceleration_rad_s2=float(np.max(np.abs(arrays["ddq_reference"]))),
            baseline_identical=True if controller == "pd" else None,
        )
        archive.update({f"{controller}_{name}": array for name, array in arrays.items()})
        cases.append(case)
    report = {
        "experiment": "planar_2r_feedforward",
        "schema_version": 1,
        "lengths_m": LENGTHS.tolist(),
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "movement_s": MOVEMENT_S,
        "hold_s": HOLD_S,
        "cases": cases,
        "force_alignment": "Feedforward, feedback, total requested and actual torque[k] act over [k*dt,(k+1)*dt). Total=requested FF+PD, actual=clip(total,+/-0.25Nm). No fabricated final-frame force.",
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
        for key, _ in FEEDFORWARD_METHODS:
            cases = [next(c for c in t["cases"] if c["key"] == key) for t in selected]
            complete = [c for c in cases if c["completed"]]
            groups[group][key] = {
                "episodes": len(cases),
                "completed": len(complete),
                "path_successes": sum(c["path_success"] for c in cases),
                "endpoint_successes": sum(c["endpoint_success"] for c in cases),
                "physical_failures": sum(bool(c["failure_reason"]) for c in cases),
                "episodes_with_clipping": sum(c["clipped_steps"] > 0 for c in cases),
                "median_max_cross_mm": float(np.median([c["max_cross_track_mm"] for c in complete]))
                if complete
                else None,
                "median_timed_rms_mm": float(
                    np.median([c["rms_timed_tracking_mm"] for c in complete])
                )
                if complete
                else None,
            }
    return groups


def save_overview(path, trials, groups):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    counts, paired, errors, torque = axes.ravel()
    names = (("interior", "内部"), ("near_extension", "近伸直"), ("diagnostic", "固定伸直"))
    for i, (key, label) in enumerate(FEEDFORWARD_METHODS):
        stats = [groups[group][key] for group, _ in names]
        bars = counts.bar(
            np.arange(3) + (i - 0.5) * 0.3,
            [s["path_successes"] / s["episodes"] * 100 for s in stats],
            width=0.28,
            color=FEEDFORWARD_COLORS[key],
            label=label,
        )
        counts.bar_label(
            bars, labels=[f"{s['path_successes']}/{s['episodes']}" for s in stats], fontsize=9
        )
    counts.set(
        xticks=np.arange(3),
        xticklabels=[n for _, n in names],
        ylim=(0, 120),
        ylabel="路径通过率（%）",
        title="同一批路径，全部 4s + 3s",
    )
    counts.legend()
    pairs = np.array([[c["max_cross_track_mm"] for c in t["cases"]] for t in trials])
    paired.scatter(pairs[:, 0], pairs[:, 1], color="#0f766e", label="每点一条配对路径")
    limit = max(2.0, float(pairs.max())) * 1.1
    paired.plot([0, limit], [0, limit], "k--", linewidth=1, label="相等线")
    paired.set(
        xlabel="原 PD 最大偏离（mm）",
        ylabel="前馈 + PD 最大偏离（mm）",
        title="相等线下方：本指标减小",
        xlim=(0, limit),
        ylim=(0, limit),
        aspect="equal",
    )
    paired.legend(fontsize=8)
    fixed = next(t for t in trials if t["id"] == "singular_inward")
    with np.load(
        path.parent / fixed["relative_results"] / "trajectories.npz", allow_pickle=False
    ) as data:
        for key, label in FEEDFORWARD_METHODS:
            tip = data[f"{key}_points"][:, -1]
            desired = data[f"{key}_desired_points"]
            times = np.arange(len(tip)) * 0.02
            errors.plot(
                times,
                np.linalg.norm(tip - desired, axis=1) * 1000,
                color=FEEDFORWARD_COLORS[key],
                label=label,
            )
        for name, label, color in (
            ("feedforward_torques_nm", "模型前馈", "#2563eb"),
            ("feedback_torques_nm", "PD 修正", "#ea580c"),
            ("torques_nm", "实际施加", "#0f766e"),
        ):
            torque.stairs(data[f"feedforward_pd_{name}"][:, 0], times, label=label, color=color)
        torque.stairs(
            data["feedforward_pd_requested_torques_nm"][:, 0],
            times,
            label="合计请求",
            color="#7c3aed",
            linestyle="--",
        )
    errors.set(title="固定伸直反例：距同时刻规定点", xlabel="时间（s）", ylabel="误差（mm）")
    errors.legend(fontsize=8)
    torque.set(title="固定伸直反例：肩部力矩分解", xlabel="时间（s）", ylabel="力矩（N·m）")
    for bound in (-TORQUE_LIMIT, TORQUE_LIMIT):
        torque.axhline(bound, color="gray", linestyle=":")
    torque.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    manifest, raw, digest = load_timing_source(source)
    audit = audit_inverse_dynamics()
    protocol = {
        "source_results": str(source.resolve()),
        "source_manifest_sha256": digest,
        "source_summary_sha256": hashlib.sha256((source / "summary.json").read_bytes()).hexdigest(),
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "movement_s": MOVEMENT_S,
        "hold_s": HOLD_S,
        "dt_s": 0.02,
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limit_nm": TORQUE_LIMIT,
        "reference_speed_limit_rad_s": REFERENCE_SPEED_LIMIT,
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "selection": "All source paths, same order, fixed 4s movement; no outcome filtering",
        "feedforward": "Offline mj_inverse on reference q,dq,ddq in separate MjData with the nominal physical model; qfrc_inverse includes passive damping compensation. ddq[k]=(dq[k+1]-dq[k])/dt, terminal dq=0. This finite-difference plan is not an exact continuous-time derivative, nor the future measured state.",
        "controller": "actual ctrl = clip(tau_ff + Kp*wrapped(qref-q) + Kd*(dqref-dq), +/-0.25Nm); PD gains and reference arrays unchanged. FF is computed on reference state, not computed-torque feedback linearization on actual state.",
        "acceptance": "Same 2mm maximum finite-segment distance during [0,4s], complete 7s, and terminal continuous >=0.5s after 4s: tip<=2mm, joints<=.01rad, speeds<=.02rad/s. Not a strict arrival-by-4s criterion; error at 4s reported separately.",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_manifest.json").write_bytes(raw)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    trials = []
    for spec in manifest["trials"]:
        relative = f"trials/{spec['id']}"
        cases = run_trial(spec, source / relative, output / relative)
        trials.append({**spec, "relative_results": relative, "cases": cases})
        print(spec["id"], {c["key"]: round(c["max_cross_track_mm"], 3) for c in cases}, flush=True)
    groups = aggregate(trials)
    report = {
        "experiment": "planar_2r_feedforward_batch",
        "schema_version": 1,
        **protocol,
        "inverse_dynamics_audit": audit,
        "groups": groups,
        "trials": trials,
        "limitations": "Known-path, exact-model ideal simulation. No noise, payload uncertainty or contact. Feedforward does not increase motor capacity or eliminate numerical/feedback errors; nominal-model improvement is not real-hardware robustness.",
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
    print(json.dumps(report["groups"], indent=2))


if __name__ == "__main__":
    main()
