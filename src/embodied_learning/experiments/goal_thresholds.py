"""Lesson 21 supplement: change stopping tolerance; keep true acceptance fixed."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from embodied_learning.experiments.goal_reaching import (
    CASES,
    DT,
    METHODS,
    digest,
    load_recording,
    run_experiment,
)
from embodied_learning.goal_control import DEFAULT_CONFIG

VARIANTS = (
    ("cm2", 0.02, "2 cm", "#2563eb"),
    ("cm1", 0.01, "1 cm", "#ea580c"),
    ("cm05", 0.005, "0.5 cm", "#9333ea"),
)
DEFAULT_RESULTS = "results/goal_thresholds_2026-09-03"


def stopping_metrics(arrays):
    """Count failed settling attempts, not steering changes or landmark updates.

    The terminal zero is not a movement interval or a new settling attempt.
    """
    modes = arrays["modes"][:-1]
    settling = modes == 2
    first = np.flatnonzero(settling)
    return {
        "first_stop_attempt_s": float(first[0] * DT) if len(first) else None,
        "settling_attempts": int(np.count_nonzero(settling & ~np.r_[False, settling[:-1]])),
        "restart_count": int(np.count_nonzero((modes[:-1] == 2) & np.isin(modes[1:], [0, 1]))),
        "moving_duration_s": float(
            np.count_nonzero(np.any(arrays["commands"][:-1] != 0, axis=1)) * DT
        ),
    }


def summarize(records, runs):
    rows, pairs = [], []
    for case, _, _ in CASES:
        for method, _, _ in METHODS:
            baseline = [records[(VARIANTS[0][0], case, run, method)][1] for run in range(runs)]
            for variant, _, label, _ in VARIANTS:
                selected = [records[(variant, case, run, method)] for run in range(runs)]
                values = [trial for _, trial in selected]
                stops = [stopping_metrics(arrays) for arrays, _ in selected]
                rows.append(
                    {
                        "case": case,
                        "method": method,
                        "variant": variant,
                        "label": label,
                        "true_success_count": sum(t["true_success"] for t in values),
                        "false_arrival_count": sum(t["false_arrival"] for t in values),
                        "timeout_count": sum(not t["controller_arrived"] for t in values),
                        "mean_true_final_distance_m": float(
                            np.mean([t["true_final_distance_m"] for t in values])
                        ),
                        "mean_estimated_final_distance_m": float(
                            np.mean([t["estimated_final_distance_m"] for t in values])
                        ),
                        "mean_duration_s": float(np.mean([t["duration_s"] for t in values])),
                        "max_duration_s": max(t["duration_s"] for t in values),
                        "mean_restart_count": float(np.mean([s["restart_count"] for s in stops])),
                        "max_restart_count": max(s["restart_count"] for s in stops),
                    }
                )
                if variant == VARIANTS[0][0]:
                    continue
                for run, (trial, old, stop) in enumerate(zip(values, baseline, stops)):
                    pairs.append(
                        {
                            "case": case,
                            "method": method,
                            "variant": variant,
                            "run": run,
                            "true_error_change_m": trial["true_final_distance_m"]
                            - old["true_final_distance_m"],
                            "duration_change_s": trial["duration_s"] - old["duration_s"],
                            "restart_count": stop["restart_count"],
                            "success_gained": bool(
                                trial["true_success"] and not old["true_success"]
                            ),
                            "success_lost": bool(old["true_success"] and not trial["true_success"]),
                        }
                    )
    return rows, pairs


def compare_baseline(records, baseline):
    old_report, old_records = load_recording(baseline)
    selected = {key[1:]: value for key, value in records.items() if key[0] == "cm2"}
    if selected.keys() != old_records.keys():
        raise ValueError("Baseline trial identities differ")
    for key, (arrays, trial) in selected.items():
        old_arrays, old_trial = old_records[key]
        if trial != old_trial:
            raise ValueError("Baseline trial metrics differ")
        for name in arrays:
            if not np.array_equal(arrays[name], old_arrays[name]):
                raise ValueError(f"Baseline trajectory differs: {key}/{name}")
    return {
        "directory": str(Path(baseline).resolve()),
        "summary_sha256": digest(Path(baseline) / "summary.json"),
        "trajectories_sha256": old_report["trajectories_sha256"],
        "exact_array_match_trials": len(selected),
    }


def run_thresholds(output, *, runs=20, seed=0, baseline=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(runs) is not int or runs < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Need integer runs >= 2 and seed >= 0")
    output.mkdir(parents=True)
    records, manifests = {}, []
    for key, radius, label, color in VARIANTS:
        config = replace(DEFAULT_CONFIG, estimated_stop_radius_m=radius)
        run_experiment(output / key, runs=runs, seed=seed, config=config)
        _, loaded = load_recording(output / key)
        records.update({(key, *identity): value for identity, value in loaded.items()})
        manifests.append(
            {
                "key": key,
                "radius_m": radius,
                "label": label,
                "color": color,
                "summary_sha256": digest(output / key / "summary.json"),
            }
        )
    rows, pairs = summarize(records, runs)
    report = {
        "experiment": "estimated_stopping_tolerance_comparison",
        "schema_version": 1,
        "runs": runs,
        "seed": seed,
        "variants": manifests,
        "rows": rows,
        "pairs": pairs,
        "fixed_controller": asdict(DEFAULT_CONFIG),
        "baseline_verification": compare_baseline(records, baseline) if baseline else None,
        "analysis_source_sha256": digest(__file__),
        "metric_definitions": {
            "restart_count": "Settling (zero wheels) -> driving/turning, excluding the terminal frame",
            "mean_duration_s": "All trials including timeout duration, not a successful-arrival-only mean",
            "true_error_change_m": "Variant final true distance minus paired 2 cm baseline; negative is better",
        },
        "limits": [
            "Only estimated_stop_radius_m changes; physical acceptance remains 3 cm and dwell 0.4 s",
            "Paired noise at common timestamps, not identical sensor values after trajectories diverge",
            "Ideal velocity kinematics; no braking dynamics, slip, obstacles or live ROS",
            "Finite paired samples, not hardware precision or population success rates",
            "Each trial ends at arrival or 40 s; stopped cars freeze without further observations",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report, records


def load_thresholds(directory):
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (
        report.get("experiment") != "estimated_stopping_tolerance_comparison"
        or report.get("schema_version") != 1
    ):
        raise ValueError("Not a stopping-tolerance recording")
    if report.get("fixed_controller") != asdict(DEFAULT_CONFIG):
        raise ValueError("Fixed controller mismatch")
    if (
        type(report.get("runs")) is not int
        or report["runs"] < 2
        or type(report.get("seed")) is not int
        or report["seed"] < 0
    ):
        raise ValueError("Invalid runs or seed")
    if len(report["variants"]) != len(VARIANTS):
        raise ValueError("Missing tolerance variant")
    records, reference = {}, None
    for entry, (key, radius, label, color) in zip(report["variants"], VARIANTS):
        if [entry[k] for k in ("key", "radius_m", "label", "color")] != [key, radius, label, color]:
            raise ValueError("Tolerance identity mismatch")
        if digest(directory / key / "summary.json") != entry["summary_sha256"]:
            raise ValueError("Child summary checksum mismatch")
        child, loaded = load_recording(directory / key)
        if child["controller"] != asdict(replace(DEFAULT_CONFIG, estimated_stop_radius_m=radius)):
            raise ValueError("More than stopping tolerance changed")
        # Child manifests must agree on every non-result, non-tolerance parameter.
        settings = {
            k: v
            for k, v in child.items()
            if k not in {"controller", "comparisons", "trials", "trajectories_sha256"}
        }
        if reference is None:
            reference = settings
        elif settings != reference:
            raise ValueError("Unpaired environment or sensor configuration")
        if child["runs"] != report["runs"] or child["seed"] != report["seed"]:
            raise ValueError("Unpaired seed or run count")
        records.update({(key, *identity): value for identity, value in loaded.items()})
    rows, pairs = summarize(records, report["runs"])
    if rows != report["rows"] or pairs != report["pairs"]:
        raise ValueError("Comparison metrics mismatch")
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline", type=Path, help="Optional lesson-21 recording for exact 2 cm comparison"
    )
    args = parser.parse_args()
    report, _ = run_thresholds(args.output, runs=args.runs, seed=args.seed, baseline=args.baseline)
    print(json.dumps(report["rows"], ensure_ascii=False, indent=2))
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
