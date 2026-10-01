"""Recompute the round-13 qualification from signed archives, not a passed flag."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def physical_replay(xml, truth, states, forces, contacts):
    """Integrate recorded applied forces; remeasure every body/world contact."""
    model = mujoco.MjModel.from_xml_string(str(xml.item()))
    data = mujoco.MjData(model)
    data.qpos[:] = truth[0]
    mujoco.mj_forward(model, data)
    state_error = contact_error = continuous = 0.0
    steps = 0
    for state, force, expected in zip(states.reshape(-1, 6), forces.reshape(-1, 3),
                                     contacts.reshape(-1, 5), strict=True):
        assert np.linalg.norm(force[:2]) <= 6.4 + 1e-10
        assert abs(force[2]) <= 0.59532 + 1e-10
        data.qfrc_applied[:] = force
        mujoco.mj_step(model, data)
        state_error = max(state_error, float(np.max(np.abs(np.r_[data.qpos, data.qvel] - state))))
        normal, penetration = [], []
        for j in range(data.ncon):
            c = data.contact[j]
            if (model.geom_bodyid[c.geom1] == 0) == (model.geom_bodyid[c.geom2] == 0):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, j, wrench)
            if wrench[0] > 1e-6:
                normal.append(float(wrench[0]))
                penetration.append(max(0.0, -float(c.dist)))
        continuous = continuous + 0.002 if normal else 0.0
        measured = np.array([len(normal), max(normal, default=0), sum(normal),
                             max(penetration, default=0), continuous])
        contact_error = max(contact_error, float(np.max(np.abs(measured - expected))))
        steps += 1
    assert state_error < 1e-9 and contact_error < 1e-7
    return steps, state_error, contact_error


def validate(directory):
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text(encoding="utf8"))
    protocol = summary["protocol"]
    assert sha256(out / "protocol.json") == summary["protocol_sha256"]
    assert json.loads((out / "protocol.json").read_text(encoding="utf8")) == protocol
    assert sha256(out / "registration.md") == protocol["registration_sha256"]
    for name, expected in protocol["source_sha256"].items():
        assert sha256(out / "source" / name) == expected, name
        assert sha256(ROOT / name) == expected, f"executed source changed: {name}"
    cases = {tuple(c) for c in protocol["cases"]}
    rows = summary["rows"]
    identities = {
        (r["layout"], r["obstacle"], r["seed"], r["cohort"],
         r["condition_key"], r["method"], r["profile"]) for r in rows
    }
    expected_count = {13: 87, 14: 75}[protocol["round"]]
    assert len(cases) == len(rows) == len(identities) == expected_count and cases == identities
    checked = []
    total_substeps = 0
    max_state_error = max_contact_error = 0.0
    for row in rows:
        file = out / row["file"]
        assert sha256(file) == row["sha256"], file
        # Decode physical evidence only; images are already covered by file hash.
        with np.load(file, allow_pickle=False) as data:
            truth, goal, velocity = data["truth"], data["goal"], data["world_velocity"]
            contact = data["substep_contact"].reshape(-1, 5)
            states = data["substep_state"]
            steps, state_error, contact_error = physical_replay(
                data["model_xml"], truth, states, data["substep_force"], data["substep_contact"]
            )
            total_substeps += steps
            max_state_error = max(max_state_error, state_error)
            max_contact_error = max(max_contact_error, contact_error)
            commands = data["commands"]
            assert len(truth) == row["frames"] == len(velocity) == len(commands)
            assert len(states) == len(truth) - 1
            if len(states):
                positions = states[:, -1, :3].copy()
                positions[:, 2] = np.arctan2(np.sin(positions[:, 2]), np.cos(positions[:, 2]))
                np.testing.assert_allclose(positions, truth[1:], atol=1e-12, rtol=0)
                np.testing.assert_allclose(states[:, -1, 3:], velocity[1:], atol=1e-12, rtol=0)
            distance = float(np.linalg.norm(truth[-1, :2] - goal[-1]))
            stopped = bool(np.linalg.norm(velocity[-1, :2]) < 0.01 and abs(velocity[-1, 2]) < 0.01)
            force = float(contact[:, 1].max(initial=0))
            penetration = float(contact[:, 3].max(initial=0))
            longest = float(contact[:, 4].max(initial=0))
            limits = protocol["contact_limits"]
            assert limits == {"peak_force_n": 40.0, "penetration_m": 0.005, "continuous_contact_s": 0.5}
            mild = force <= 40 and penetration <= 0.005 and longest <= 0.5 + 1e-12
            arrived = distance < 0.25 and stopped
            assert np.isclose(distance, row["final_goal_distance_m"], atol=1e-12)
            assert stopped == row["physically_stopped"]
            assert force == row["contact_peak_force_n"]
            assert mild == (not row["severe_contact"])
            assert bool(row["reached"]) == bool(row["announced"] and arrived)
            assert (bool(row["announced"]) and arrived and mild) == bool(row["passed"])
            assert bool(row["announced"] and not arrived) == bool(row["false_arrival"])
        checked.append({"file": row["file"], "sha256": row["sha256"], "method": row["method"],
                            "layout": row["layout"], "profile": row["profile"], "cohort": row["cohort"],
                            "seed": row["seed"], "actual_arrived": arrived, "passed": row["passed"],
                            "false_arrival": row["false_arrival"], "peak_force_n": force,
                            "final_distance_m": distance, "physically_stopped": stopped})
        print(f"checked {len(checked)}/{expected_count} {row['file']}", flush=True)
    final_method = protocol["stability_required"]["method"]
    positive = [r for r in checked if r["method"] == final_method
                and r["cohort"] != "impossible_negative"]
    negative = [r for r in checked if r["cohort"] == "impossible_negative"]
    assert len(positive) == 60 and len(negative) == 3
    qualified = all(r["passed"] and not r["false_arrival"] for r in positive)
    qualified &= all(not r["actual_arrived"] and not r["false_arrival"]
                     and r["physically_stopped"] and r["peak_force_n"] == 0 for r in negative)
    return {"qualified": qualified, "positive_passed": sum(r["passed"] for r in positive),
                "physical_substeps": total_substeps,
                "max_physical_state_error": max_state_error,
                "max_physical_contact_error": max_contact_error,
                "positive_total": 60, "negative_correct": sum(not r["actual_arrived"] for r in negative),
                "input_summary_sha256": sha256(out / "summary.json"),
                "validator_sha256": sha256(Path(__file__)),
                "rows": checked, "layouts": dict(Counter(r["layout"] for r in positive))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/navigation_reinforcement_v13")
    parser.add_argument("--output", default="docs/benchmarks/navigation-clearance-validation-v13.json")
    args = parser.parse_args()
    result = validate(args.input)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(f"qualified={result['qualified']}; {result['positive_passed']}/60, negative={result['negative_correct']}/3")
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
