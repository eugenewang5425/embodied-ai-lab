"""Lesson 11: frozen lesson10 paths, a waypoint IK reference, and unchanged joint PD."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.arm_path import IK_COMPARISON_METHODS, METHOD_COLORS, validate_line
from embodied_learning.experiments.arm_path import run_experiment as run_trial
from embodied_learning.experiments.arm_path_batch import GROUPS, aggregate
from embodied_learning.planar_arm import MODEL_PATH, forward_kinematics, vector2
from embodied_learning.plotting import configure_plot_font


def load_source(source, *, experiment="planar_2r_path_batch"):
    source = Path(source)
    raw = (source / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary.get("experiment") != experiment or manifest.get("schema_version") != 1:
        raise ValueError(f"Expected {experiment} source results")
    count = manifest.get("per_group")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
        raise ValueError("Invalid source group size")
    digest = hashlib.sha256(raw).hexdigest()
    if summary.get("manifest_sha256") != digest:
        raise ValueError("Source manifest byte hash does not match; use the verified lesson10 _v2")
    if summary.get("model_xml_sha256") != hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest():
        raise ValueError("Source model differs; this would not be a single-factor comparison")
    seen = set()
    for trial in manifest["trials"]:
        key = trial["id"]
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key) or key in seen:
            raise ValueError("Invalid or duplicate trial id")
        seen.add(key)
        if trial["group"] not in ("interior", "near_extension", "diagnostic"):
            raise ValueError("Unknown source group")
        start = forward_kinematics(vector2(trial["initial_q_rad"]))
        if not np.allclose(start, vector2(trial["start_m"]), rtol=0, atol=1e-12):
            raise ValueError("Source initial angles and tip coordinates disagree")
        validate_line(start, trial["target_m"])
    for group in ("interior", "near_extension"):
        if sum(t["group"] == group for t in manifest["trials"]) != manifest["per_group"]:
            raise ValueError("Source group count disagrees with its protocol")
    diagnostic = [t for t in manifest["trials"] if t["group"] == "diagnostic"]
    if len(diagnostic) != 1 or diagnostic[0]["id"] != "singular_inward":
        raise ValueError("Expected the one fixed singular-start diagnostic")
    return manifest, raw, digest


def verify_retained_trajectories(source, destination):
    """No fallback reruns or numerical tolerance hiding changes to the baseline."""
    with (
        np.load(source / "trajectories.npz", allow_pickle=False) as old,
        np.load(destination / "trajectories.npz", allow_pickle=False) as new,
    ):
        for method in ("joint_interpolation", "jacobian_path"):
            for array in (
                "states",
                "points",
                "torques_nm",
                "desired_points",
                "q_reference",
                "dq_reference",
            ):
                key = f"{method}_{array}"
                if not np.array_equal(old[key], new[key]):
                    raise ValueError(f"Retained controller trajectory changed: {source.name}/{key}")


def save_overview(path, trials, groups):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    xy, timed, counts, tracking = axes.ravel()
    diagnostic = next(t for t in trials if t["id"] == "singular_inward")
    with np.load(
        path.parent / diagnostic["relative_results"] / "trajectories.npz", allow_pickle=False
    ) as archive:
        for method, label in IK_COMPARISON_METHODS:
            color = METHOD_COLORS[method]
            tip = archive[f"{method}_points"][:, -1]
            desired = archive[f"{method}_desired_points"]
            dt = next(c["dt_s"] for c in diagnostic["cases"] if c["key"] == method)
            times = np.arange(len(tip)) * dt
            xy.plot(tip[:, 0] * 100, tip[:, 1] * 100, color=color, label=label)
            xy.plot(*tip[-1] * 100, marker="o", color=color)
            timed.plot(
                times, np.linalg.norm(tip - desired, axis=1) * 1000, color=color, label=label
            )
            if method == "waypoint_ik":
                ref_tip = np.array(
                    [forward_kinematics(q) for q in archive[f"{method}_q_reference"]]
                )
                tracking.plot(
                    times,
                    np.linalg.norm(ref_tip - desired, axis=1) * 1000,
                    label="几何参考误差",
                    linestyle="--",
                    color="#475569",
                )
                tracking.plot(
                    times,
                    np.linalg.norm(tip - desired, axis=1) * 1000,
                    label="真实末端误差",
                    color=color,
                )
        xy.plot(
            desired[[0, -1], 0] * 100,
            desired[[0, -1], 1] * 100,
            "k--",
            linewidth=1,
            label="规定路径",
        )
    xy.set(
        xlabel="X（cm）",
        ylabel="Y（cm，纵向放大看误差）",
        title="完全伸直向内收：停住、走弯路、跟随直线",
    )
    timed.set(
        xlabel="时间（s）", ylabel="距同时刻规定点（mm）", title="同一条路径、同一台机械臂、同一 PD"
    )
    tracking.set(xlabel="时间（s）", ylabel="误差（mm）", title="逐点 IK：几何准确不等于实际零误差")
    group_defs = (*GROUPS, ("diagnostic", "固定奇异初态"))
    x = np.arange(len(group_defs))
    for i, (method, label) in enumerate(IK_COMPARISON_METHODS):
        stats = [groups[group][method] for group, _ in group_defs]
        bars = counts.bar(
            x + (i - 1) * 0.25,
            [100 * s["path_successes"] / s["episodes"] for s in stats],
            width=0.23,
            color=METHOD_COLORS[method],
            label=label,
        )
        counts.bar_label(
            bars, labels=[f"{s['path_successes']}/{s['episodes']}" for s in stats], fontsize=8
        )
    counts.set(
        xticks=x,
        xticklabels=[label for _, label in group_defs],
        ylabel="本批路径通过率（%）",
        ylim=(0, 123),
        title="原清单原样复用；固定反例单列",
    )
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    xy.legend(fontsize=8)
    timed.legend(fontsize=8)
    tracking.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_comparison(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    manifest, raw, digest = load_source(source)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_bytes(raw)
    trials = []
    for index, specification in enumerate(manifest["trials"]):
        relative = f"trials/{specification['id']}"
        report = run_trial(
            output / relative,
            initial_q=specification["initial_q_rad"],
            target=specification["target_m"],
            methods=IK_COMPARISON_METHODS,
            make_plot=False,
            trial={
                "id": specification["id"],
                "group": specification["group"],
                "seed": manifest["seed"],
            },
        )
        verify_retained_trajectories(source / relative, output / relative)
        trials.append(
            {
                **specification,
                "relative_results": relative,
                "retained_trajectories_identical": True,
                "cases": report["cases"],
            }
        )
        ik = next(c for c in report["cases"] if c["key"] == "waypoint_ik")
        print(
            f"[{index + 1}/{len(manifest['trials'])}] {specification['id']}: IK_path={ik['path_success']}, cross={ik['max_cross_track_mm']:.3f}mm",
            flush=True,
        )
    groups = aggregate(trials, IK_COMPARISON_METHODS)
    summary = {
        "experiment": "planar_2r_waypoint_ik_comparison",
        "schema_version": 1,
        "source_results": str(source.resolve()),
        "source_manifest_sha256": digest,
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "seed": manifest["seed"],
        "per_group": manifest["per_group"],
        "controller_episode_count": 3 * len(trials),
        "groups": groups,
        "trials": trials,
        "retained_controller_validation": "All states, points, torques, desired points, joint references and reference velocities of joint interpolation and Jacobian exactly equal the saved lesson10 arrays. Manifest bytes unchanged.",
        "protocol": "Only reference generation differs: smooth joint interpolation, local DLS Jacobian, or waypoint analytic IK. Positive elbow branch, same desired line/time, 8+3s, same PD gains and torque bounds, original 2mm path and 0.5s terminal criteria.",
        "waypoint_velocity": "Forward finite differences of the unwrapped offline joint reference over 0.02s; velocity plans above 1rad/s rejected, not silently clipped. This does not make a singular instantaneous Jacobian invertible.",
        "limitations": "Re-evaluation of a known, small lesson10 manifest, not an independent unseen test set. 2R analytic geometry, no contact/noise/payload. Not a general singularity solution for arbitrary robots or timings.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_overview(output / "overview.png", trials, groups)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_comparison(args.source_results, args.output)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    for group, methods in report["groups"].items():
        print(
            group,
            {method: f"{s['path_successes']}/{s['episodes']}" for method, s in methods.items()},
        )


if __name__ == "__main__":
    main()
