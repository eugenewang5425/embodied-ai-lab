"""Retest every old reachable fusion failure separately from fresh holdouts."""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from embodied_learning.experiments import navigation_height_study as study
from embodied_learning.experiments.footprint_tracking import digest, save_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/navigation_height_known_regressions_v14")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old_paths = [ROOT / "results/navigation_reinforcement_v11/summary.json",
                 ROOT / "results/navigation_reinforcement_v13/summary.json",
                 ROOT / "results/navigation_bypass_development_v11_metric/summary.json"]
    identities = set()
    failures = []
    for path in old_paths:
        summary = json.loads(path.read_text(encoding="utf8"))
        for row in summary["rows"]:
            if row["method"] not in ("lidar_depth", "corridor_depth") or row["passed"]:
                continue
            if row["cohort"] == "impossible_negative":
                continue
            identity = (row["layout"], row["obstacle"], row["seed"], row["profile"])
            if identity not in identities:
                identities.add(identity)
                failures.append({"summary": str(path.relative_to(ROOT)), "file": row["file"],
                                 "sha256": row["sha256"], "old_status": row["status"], "case": identity})
    assert len(failures) == 5
    formal = json.loads((ROOT / "results/navigation_reinforcement_v14/protocol.json").read_text(encoding="utf8"))
    for name, sha in formal["source_sha256"].items():
        assert digest(ROOT / name) == sha, name
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    cases = [(layout, ob, seed, "known_regression", "delay_2", "height_depth", profile)
             for layout, ob, seed, profile in sorted(identities)]
    protocol = {"known_failures": failures, "cases": cases,
                "parent_protocol_sha256": digest(ROOT / "results/navigation_reinforcement_v14/protocol.json"),
                "source_sha256": formal["source_sha256"], "runner_sha256": digest(Path(__file__)),
                "meaning": "previously seen failure retests; excluded from 60 fresh holdouts"}
    save_json(args.output / "protocol.json", protocol)
    summary = {"protocol": protocol, "protocol_sha256": digest(args.output / "protocol.json"), "rows": []}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as pool:
        for job in as_completed([pool.submit(study.execute_case, case, args.output) for case in cases]):
            row = job.result()
            summary["rows"].append(row)
            summary["wall_s"] = time.perf_counter() - started
            save_json(args.output / "summary.json", summary)
            print(f"{len(summary['rows'])}/5 {row['profile']} seed{row['seed']}: {row['status']} F={row['contact_peak_force_n']:.2f}", flush=True)
    assert all(row["passed"] for row in summary["rows"]), "a known failure remains unsolved"


if __name__ == "__main__":
    main()
