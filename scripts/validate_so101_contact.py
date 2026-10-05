"""Independently replay physics, then audit causal gates and historical bridges."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from validate_so101_approach import validate as physical_validate

from embodied_learning.gripper_contact import BilateralContactGate
from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.so101 import CONTROL_DT


def audit_gate(record):
    checked = refused = 0
    cfg = record.summary["contact_gate"]
    for key, row in record.rows.items():
        if row["method"] == "baseline":
            continue
        a = record.arrays[key]
        gate = BilateralContactGate(cfg["threshold_n"], cfg["duration_s"])
        n = len(a["timestamps"])
        assert a["gate_elapsed_s"].shape == (n,)
        assert a["solved_jaw_min_n"].shape == (n, 2)
        assert a["gate_elapsed_s"][0] == 0
        for i in range(1, n):
            past = a["physics_jaw_n"][(i - 1) * 10 : i * 10, :2]
            np.testing.assert_allclose(a["solved_jaw_min_n"][i], past.min(axis=0), rtol=0, atol=0)
            if a["phase"][i] == 3:
                gate.update(past, CONTROL_DT)
            np.testing.assert_allclose(a["gate_elapsed_s"][i], gate.elapsed, rtol=0, atol=1e-12)
        if row["method"] in ("gate", "surface_gate"):
            if row["contact_refused"]:
                assert not gate.ready and a["phase"][-1] == 3
                assert not np.any(a["phase"] >= 4) and not row["success"]
                refused += 1
            if np.any(a["phase"] == 4):
                assert gate.ready and not row["contact_refused"]
        else:
            assert not row["contact_refused"]
        checked += n - 1
    return {"past_intervals_checked": checked, "contact_refusals": refused}


def bridge(development, previous):
    new = ManipulationRecord(development)
    old = ManipulationRecord(previous)
    matches = []
    for p in new.positions:
        _, a = new.select(p, "baseline")
        _, b = old.select(p, "center_cartesian")
        assert set(a) == set(b)
        assert all(np.array_equal(a[k], b[k]) for k in a), p
        matches.append({"position": p, "arrays": len(a), "bitwise_identical": True})
    return matches


def gate_control_bridge(record):
    matches = []
    for position in record.positions:
        for reference, gated in (("baseline", "gate"), ("surface", "surface_gate")):
            _, a = record.select(position, reference)
            row, b = record.select(position, gated)
            n = len(b["timestamps"])
            for name in ("commands", "qpos", "qvel", "object_xyz"):
                np.testing.assert_array_equal(a[name][:n], b[name])
            matches.append(
                {
                    "position": position,
                    "pair": [reference, gated],
                    "frames": n,
                    "refused": row["contact_refused"],
                    "bitwise_identical_until_end": True,
                }
            )
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--development", type=Path, default=Path("results/so101_contact_dev_v2"))
    parser.add_argument("--previous", type=Path, default=Path("results/so101_approach_v3b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = physical_validate(args.directory)
    result["causal_gate_audit"] = audit_gate(ManipulationRecord(args.directory))
    result["development_gate_audit"] = audit_gate(ManipulationRecord(args.development))
    result["baseline_bridge"] = bridge(args.development, args.previous)
    result["gate_control_bridge"] = gate_control_bridge(ManipulationRecord(args.directory))
    result["contact_validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["wall_s"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
