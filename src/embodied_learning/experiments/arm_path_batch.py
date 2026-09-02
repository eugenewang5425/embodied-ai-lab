"""Lesson 10: seeded paired path coverage, with rejected geometry and a singular start."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from embodied_learning.arm_path import COLORS, METHODS, segment_distance, validate_line
from embodied_learning.experiments.arm_path import run_experiment as run_trial
from embodied_learning.experiments.arm_reaching import INITIAL_Q
from embodied_learning.planar_arm import LENGTHS, MODEL_PATH, forward_kinematics, inverse_kinematics
from embodied_learning.plotting import configure_plot_font

GROUPS = (("interior", "内部路径"), ("near_extension", "接近伸直"))
DEFAULT_SEED = 400
DEFAULT_PER_GROUP = 12


def polar(radius, angle):
    return radius * np.array([np.cos(angle), np.sin(angle)])


def generate_manifest(seed=DEFAULT_SEED, per_group=DEFAULT_PER_GROUP):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not isinstance(per_group, int) or isinstance(per_group, bool) or not 1 <= per_group <= 100:
        raise ValueError("per_group must be an integer in [1, 100]")
    accepted, rejected = [], []
    for group_index, (group, _) in enumerate(GROUPS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, group_index]))
        found = 0
        for candidate_index in range(10000):
            if group == "interior":
                initial = np.deg2rad([rng.uniform(-90, 90), rng.uniform(40, 140)])
                target = forward_kinematics(
                    np.deg2rad([rng.uniform(-90, 90), rng.uniform(40, 140)])
                )
            else:
                phi = np.deg2rad(rng.uniform(-60, 60))
                start = polar(rng.uniform(0.40, 0.58), phi)
                initial = inverse_kinematics(start)[0]
                target = polar(rng.uniform(0.6995, 0.69999), phi + np.deg2rad(rng.uniform(-35, 35)))
            start = forward_kinematics(initial)
            length = float(np.linalg.norm(target - start))
            reason = ""
            try:
                validate_line(start, target)
            except ValueError:
                reason = "unreachable_line"
            if not reason and not 0.12 <= length <= 0.65:
                reason = "length_outside_sampling_band"
            if (
                not reason
                and group == "interior"
                and segment_distance([0, 0], start, target) < 0.18
            ):
                reason = "interior_clearance_below_0.18m"
            candidate = {
                "group": group,
                "candidate_index": candidate_index,
                "initial_q_rad": initial.tolist(),
                "start_m": start.tolist(),
                "target_m": target.tolist(),
                "length_m": length,
            }
            if reason:
                rejected.append({**candidate, "reason": reason})
                continue
            accepted.append({**candidate, "id": f"{group}_{found:02d}", "kind": "seeded"})
            found += 1
            if found == per_group:
                break
        if found != per_group:
            raise ValueError(f"Sampling budget exhausted for {group}")
    accepted.append(
        {
            "id": "singular_inward",
            "group": "diagnostic",
            "kind": "fixed_counterexample",
            "candidate_index": None,
            "initial_q_rad": [0.0, 0.0],
            "start_m": [0.7, 0.0],
            "target_m": [0.5, 0.0],
            "length_m": 0.2,
        }
    )
    # These are geometry tests, never counted as failed controller episodes.
    geometry_checks = []
    for key, initial, target in (
        ("outer_unreachable", INITIAL_Q, [0.75, 0]),
        ("inner_unreachable", INITIAL_Q, [0, 0]),
        ("line_crosses_hole", inverse_kinematics([0.35, 0])[0], [-0.35, 0]),
    ):
        start = forward_kinematics(initial)
        try:
            validate_line(start, target)
        except ValueError as exc:
            geometry_checks.append(
                {
                    "id": key,
                    "start_m": start.tolist(),
                    "target_m": target,
                    "rejected": True,
                    "reason": str(exc),
                }
            )
        else:
            raise ValueError(f"Expected the geometry preflight to reject {key}")
    return {
        "schema_version": 1,
        "seed": seed,
        "per_group": per_group,
        "generator": "numpy default_rng(SeedSequence([seed, group_index])); first accepted candidates in order; fixed positive-elbow IK branch",
        "sampling_protocol": {
            "interior": "q1 start/goal uniform [-90,90]deg; q2 uniform [40,140]deg; segment min radius>=0.18m",
            "near_extension": "start radius uniform [0.40,0.58]m, start polar angle [-60,60]deg; target radius uniform [0.6995,0.69999]m, target angle=start angle+uniform[-35,35]deg",
            "both": "entire segment reachable; length [0.12,0.65]m; zero initial velocities; same 8s movement and 3s hold; no outcome-based rejection",
        },
        "trials": accepted,
        "sampling_rejections": rejected,
        "geometry_checks": geometry_checks,
    }


def aggregate(trials, methods=METHODS):
    groups = {}
    for group, _ in (*GROUPS, ("diagnostic", "固定反例")):
        groups[group] = {}
        selected = [trial for trial in trials if trial["group"] == group]
        for method, _ in methods:
            cases = [
                next(case for case in trial["cases"] if case["key"] == method) for trial in selected
            ]
            completed = [case for case in cases if case["completed"]]
            groups[group][method] = {
                "episodes": len(cases),
                "completed": len(completed),
                "endpoint_successes": sum(case["endpoint_success"] for case in cases),
                "path_successes": sum(case["path_success"] for case in cases),
                "physical_failures": sum(bool(case["failure_reason"]) for case in cases),
                "episodes_with_torque_saturation": sum(
                    case["torque_saturated_steps"] > 0 for case in cases
                ),
                "episodes_with_reference_speed_limit": sum(
                    case["reference_speed_limited_steps"] > 0 for case in cases
                ),
                "median_max_cross_mm_completed_only": float(
                    np.median([case["max_cross_track_mm"] for case in completed])
                )
                if completed
                else None,
                "median_timed_rms_mm_completed_only": float(
                    np.median([case["rms_timed_tracking_mm"] for case in completed])
                )
                if completed
                else None,
                "max_final_tip_mm_completed_only": max(
                    (case["final_tip_error_mm"] for case in completed), default=None
                ),
            }
    return groups


def save_overview(path, manifest, trials, groups):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    workspace, rates, errors, failure = axes.ravel()
    group_colors = ("#7c3aed", "#dc2626")
    for (group, label), color in zip(GROUPS, group_colors, strict=True):
        pairs = [t for t in manifest["trials"] if t["group"] == group]
        for index, trial in enumerate(pairs):
            p = np.array([trial["start_m"], trial["target_m"]])
            workspace.plot(
                p[:, 0], p[:, 1], color=color, alpha=0.45, label=label if index == 0 else None
            )
            workspace.plot(*p[1], ".", color=color)
    for radius in (abs(LENGTHS[0] - LENGTHS[1]), sum(LENGTHS)):
        workspace.add_patch(plt.Circle((0, 0), radius, fill=False, color="gray", linestyle=":"))
    workspace.set(
        xlabel="世界 X（m）",
        ylabel="世界 Y（m）",
        aspect="equal",
        title="抽样几何：点是终点，不按仿真结果挑路径",
    )
    workspace.legend(fontsize=8)
    x = np.arange(2)
    for method_index, ((method, label), color) in enumerate(zip(METHODS, COLORS, strict=True)):
        counts = [groups[group][method]["path_successes"] for group, _ in GROUPS]
        totals = [groups[group][method]["episodes"] for group, _ in GROUPS]
        fractions = [100 * count / total for count, total in zip(counts, totals, strict=True)]
        bars = rates.bar(
            x + (method_index - 1) * 0.25, fractions, width=0.23, color=color, label=label
        )
        rates.bar_label(
            bars,
            labels=[f"{count}/{total}" for count, total in zip(counts, totals, strict=True)],
            fontsize=8,
        )
    rates.set(
        xticks=x,
        xticklabels=[label for _, label in GROUPS],
        ylabel="本批路径通过率（%）",
        ylim=(0, 122),
        title="包含终点停稳要求；不是普遍成功概率",
    )
    rates.legend(fontsize=8, loc="upper right")
    for (group, label), color in zip(GROUPS, group_colors, strict=True):
        cases = [
            next(c for c in t["cases"] if c["key"] == "jacobian_path")
            for t in trials
            if t["group"] == group
        ]
        for passed, marker in ((True, "o"), (False, "x")):
            subset = [c for c in cases if c["path_success"] == passed]
            errors.scatter(
                [c["min_actual_sigma_m_per_rad"] for c in subset],
                [c["rms_timed_tracking_mm"] for c in subset],
                color=color,
                marker=marker,
                label=f"{label} / {'通过' if passed else '未通过'}",
            )
    errors.set(
        xlabel="实际轨迹最小奇异值（m/rad）",
        ylabel="时间跟踪 RMS（mm）",
        title="仅 Jacobian：接近奇异与误差（相关性）",
    )
    errors.legend(fontsize=8)
    diagnostic = next(t for t in trials if t["id"] == "singular_inward")
    with np.load(
        path.parent / diagnostic["relative_results"] / "trajectories.npz", allow_pickle=False
    ) as archive:
        for (method, label), color in zip(METHODS, COLORS, strict=True):
            tip = archive[f"{method}_points"][:, -1]
            desired = archive[f"{method}_desired_points"]
            times = np.arange(len(tip)) * 0.02
            failure.plot(
                times, np.linalg.norm(tip - desired, axis=1) * 1000, color=color, label=label
            )
    failure.set(
        xlabel="时间（s）",
        ylabel="时间跟踪误差（mm）",
        title="固定反例：完全伸直向内收，Jacobian 卡住",
    )
    failure.legend(fontsize=8)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_batch(output, *, seed=DEFAULT_SEED, per_group=DEFAULT_PER_GROUP):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    manifest = generate_manifest(seed, per_group)
    output.mkdir(parents=True, exist_ok=False)
    # Commit the sampling plan before any controller runs; failed episodes remain included.
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output / "manifest.json").write_text(manifest_text, encoding="utf-8", newline="\n")
    trials = []
    for index, specification in enumerate(manifest["trials"]):
        relative = f"trials/{specification['id']}"
        report = run_trial(
            output / relative,
            initial_q=specification["initial_q_rad"],
            target=specification["target_m"],
            make_plot=False,
            trial={"id": specification["id"], "group": specification["group"], "seed": seed},
        )
        trial = {**specification, "relative_results": relative, "cases": report["cases"]}
        trials.append(trial)
        jac = next(c for c in trial["cases"] if c["key"] == "jacobian_path")
        print(
            f"[{index + 1}/{len(manifest['trials'])}] {specification['id']}: J_path={jac['path_success']}, cross={jac['max_cross_track_mm']:.3f}mm, final={jac['final_tip_error_mm']:.3f}mm",
            flush=True,
        )
    groups = aggregate(trials)
    representatives = {
        "interior_first": "trials/interior_00",
        "near_extension_first": "trials/near_extension_00",
        "singular_start": "trials/singular_inward",
    }
    summary = {
        "experiment": "planar_2r_path_batch",
        "schema_version": 1,
        "seed": seed,
        "per_group": per_group,
        "random_path_count": 2 * per_group,
        "fixed_diagnostic_count": 1,
        "controller_episode_count": 3 * len(trials),
        "sampling_rejection_count": len(manifest["sampling_rejections"]),
        "geometry_rejections": manifest["geometry_checks"],
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "groups": groups,
        "replays": representatives,
        "trials": trials,
        "protocol": "Frozen lesson9 controllers/model/gains/limits, 8s movement +3s hold, three methods paired per path; same path_success and endpoint_success as lesson9. Sampling/rejected geometry/fixed diagnostic counted separately. No outcome-based filtering.",
        "diagnosis_limits": "Saturation, minimum singular values and reference/execution errors are logged, not automatically treated as proven root causes. Raw torque requests are not actual applied torques. Reference and execution RMS do not add arithmetically.",
        "limitations": "Small, deliberately stratified positive-elbow sample, not uniform workspace sampling or a population success probability. No noise/contact/payload change. No automatic singularity escape, retuning or path replanning.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_overview(output / "overview.png", manifest, trials, groups)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--per-group", type=int, default=DEFAULT_PER_GROUP)
    args = parser.parse_args()
    try:
        report = run_batch(args.output, seed=args.seed, per_group=args.per_group)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for group, methods in report["groups"].items():
        print(
            group,
            {method: f"{c['path_successes']}/{c['episodes']}" for method, c in methods.items()},
        )


if __name__ == "__main__":
    main()
