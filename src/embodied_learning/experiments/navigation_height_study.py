"""Round fourteen reuses the frozen physical episode in a private namespace."""

import argparse
import importlib.util
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from embodied_learning.clearance_navigation import ClearanceNavigationController
from embodied_learning.experiments import navigation_clearance_study as baseline
from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.height_aware_planning import HeightAwareNavigationController
from embodied_learning.navigation_robustness import CONDITIONS

ROOT = Path(__file__).resolve().parents[3]
METHODS = ("corridor_depth", "height_depth")
PROFILES = baseline.PROFILES
LAYOUT_BOXES = {
    "street": (5.6, 6.4, 5.4, 6.6),
    "street_west": (5.2, 6.2, 5.3, 6.5),
    "street_north": (5.9, 6.6, 5.55, 6.75),
}


@lru_cache(maxsize=1)
def private_engine():
    # Preserve baseline module globals: each worker owns an isolated engine.
    spec = importlib.util.spec_from_file_location("_height_episode_engine", baseline.__file__)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    engine.LAYOUT_BOXES = LAYOUT_BOXES
    return engine


def episode(layout, obstacle, seed, condition, method, profile, max_steps=900):
    if method not in METHODS:
        raise ValueError(method)
    engine = private_engine()
    engine.ClearanceNavigationController = (
        HeightAwareNavigationController if method == "height_depth" else ClearanceNavigationController
    )
    # Both execute the same registered force/queue/arrival lifecycle; the private
    # factory selects only the planning field. No actual boxes enter its API.
    row, data = engine.episode(layout, obstacle, seed, condition, "corridor_depth", profile, max_steps)
    row["method"] = method
    return row, data


def registered_cases(smoke=False, seed_filter=None):
    def case(layout, profile, seed, method, cohort):
        ob = "low" if profile == "bypass_low" else "beam" if profile in (
            "bypass_beam", "clear_overhead") else "crate"
        return layout, ob, seed, cohort, "delay_2", method, profile

    if smoke:
        cases = [case("street", p, 0, m, "development") for p in PROFILES for m in METHODS]
        cases += [case("street", "bypass_beam", 33, m, "development") for m in METHODS]
        return [c for c in cases if seed_filter is None or c[2] == seed_filter]
    final = [case(layout, p, seed, "height_depth", "stability_holdout")
             for layout in LAYOUT_BOXES for p in PROFILES for seed in range(37, 42)]
    reference = [case("street", p, seed, "corridor_depth", "paired_holdout")
                 for p in PROFILES for seed in (37, 38, 39)]
    negative = [case("blocked", "bypass_crate", seed, "height_depth", "impossible_negative")
                for seed in (37, 38, 39)]
    return final + reference + negative


def execute_case(case, out):
    layout, ob, seed, cohort, delay, method, profile = case
    started = time.perf_counter()
    row, data = episode(layout, ob, seed, CONDITIONS[delay], method, profile)
    file = f"{layout}_{ob}_s{seed}_{delay}_{method}_{profile}.npz"
    np.savez_compressed(Path(out) / file, **data)
    row.update(file=file, sha256=digest(Path(out) / file), cohort=cohort,
               wall_s=time.perf_counter() - started)
    return row


def run(output, smoke=False, profile_filter=None, method_filter=None, seed_filter=None,
        workers=1, resume=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    parent = ROOT / "results/navigation_reinforcement_v13/protocol.json"
    frozen = json.loads(parent.read_text(encoding="utf8"))
    for name, expected in frozen["source_sha256"].items():
        assert digest(ROOT / name) == expected, f"frozen round-13 source changed: {name}"
    sources = [ROOT / name for name in frozen["source_sha256"]]
    sources += [Path(__file__), ROOT / "src/embodied_learning/height_aware_planning.py"]
    for file in sources:
        target = out / "source" / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            assert target.read_bytes() == file.read_bytes(), f"resume source changed: {file}"
        else:
            target.write_bytes(file.read_bytes())
    cases = registered_cases(smoke, seed_filter)
    if profile_filter or method_filter or seed_filter is not None:
        assert smoke, "filters and old seeds are development-only"
        cases = [c for c in cases if (not profile_filter or c[6] == profile_filter)
                 and (not method_filter or c[5] == method_filter)]
    protocol = {
        **{k: v for k, v in frozen.items() if k not in (
            "registration_sha256", "source_sha256", "cases", "methods", "feasibility_witnesses")},
        "round": 14, "smoke": smoke, "cases": cases, "methods": list(METHODS),
        "workers": workers, "mujoco_version": mujoco.__version__,
        "depth_used_by": list(METHODS),
        "depth_dimensions_by_method": {m: [320, 240] for m in METHODS},
        "stability_required": {"method": "height_depth", "positive_episodes": 60 if not smoke else 5,
                               "layouts": list(LAYOUT_BOXES), "seeds": list(range(37, 42))},
        "controller": "private frozen round-13 lifecycle; only keep measured heights in planning feasibility",
        "parent_protocol_sha256": digest(parent),
        "registration_sha256": digest(ROOT / "docs/69-height-planning-round14-plan.md"),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "feasibility_witnesses": {
            f"{layout}/{p}": private_engine().feasibility_witness(p, layout)
            for layout in LAYOUT_BOXES for p in PROFILES
        },
    }
    if resume:
        summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
        assert summary["protocol"] == json.loads(json.dumps(protocol)), "resume protocol changed"
        assert digest(out / "protocol.json") == summary["protocol_sha256"]
        assert digest(out / "registration.md") == protocol["registration_sha256"]
        done = set()
        for row in summary["rows"]:
            key = (row["layout"], row["obstacle"], row["seed"], row["cohort"],
                   row["condition_key"], row["method"], row["profile"])
            assert key in cases and key not in done, "invalid/duplicate resumed case"
            assert digest(out / row["file"]) == row["sha256"]
            done.add(key)
        cases = [c for c in cases if c not in done]
    else:
        (out / "registration.md").write_bytes(
            (ROOT / "docs/69-height-planning-round14-plan.md").read_bytes())
        save_json(out / "protocol.json", protocol)
        summary = {"protocol": protocol, "protocol_sha256": digest(out / "protocol.json"), "rows": []}
    previous_wall = summary.get("wall_s", 0.0)
    started = time.perf_counter()
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    def persist(row):
        summary["rows"].append(row)
        summary["wall_s"] = previous_wall + time.perf_counter() - started
        temp = out / "summary.json.tmp"
        save_json(temp, summary)
        temp.replace(out / "summary.json")
        print(f"{len(summary['rows'])}/{len(protocol['cases'])} {row['layout']}/{row['profile']}/s{row['seed']} "
              f"{row['method']}: {row['status']} passed={row['passed']} F={row['contact_peak_force_n']:.2f}", flush=True)

    if workers == 1:
        for case in cases:
            persist(execute_case(case, out))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job in as_completed([pool.submit(execute_case, c, out) for c in cases]):
                persist(job.result())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/navigation_reinforcement_v14")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=(0, 33))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, args.smoke, args.profile, args.method, args.seed, args.workers, args.resume)


if __name__ == "__main__":
    main()
