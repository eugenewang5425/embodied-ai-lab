"""Audit the new yaw-initialized records without changing historical validators."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_learning.experiments.so101_grasping import score_episode
from embodied_learning.manipulation_record import ManipulationRecord
from embodied_learning.so101 import SceneConfig, SO101Simulation


def validate(directory):
    record = ManipulationRecord(directory)
    if record.summary["runtime"]["mujoco"] != mujoco.__version__:
        raise ValueError("Use the archived MuJoCo version")
    state_error = force_error = geometry_error = 0.0
    steps = 0
    for key, row in record.rows.items():
        config = SceneConfig(**row["config"])
        sim = SO101Simulation(config)
        a = record.arrays[key]
        # Initial pose is part of the input, including object yaw. Never copy
        # future states while replaying the actual requests.
        sim.data.qpos[:] = a["qpos"][0]
        sim.data.qvel[:] = a["qvel"][0]
        sim.data.ctrl[:] = a["commands"][0]
        mujoco.mj_forward(sim.model, sim.data)
        score = score_episode(a, config, row["plan_rejected"])
        for metric, value in score.items():
            if isinstance(value, (str, bool)):
                assert row[metric] == value, (key, metric)
            else:
                np.testing.assert_allclose(row[metric], value, rtol=0, atol=1e-10)
        offset = 0
        for i, command in enumerate(a["commands"][1:], 1):
            observation = sim.step(command)
            geometry_error = max(
                geometry_error,
                float(np.max(np.abs(observation["tcp"] - a["tcp_xyz"][i]))),
                float(np.max(np.abs(observation["object"] - a["object_xyz"][i]))),
            )
            for t, q, v, forces, other, penetration in sim.last_physics:
                state_error = max(
                    state_error,
                    float(np.max(np.abs(q - a["physics_qpos"][offset]))),
                    float(np.max(np.abs(v - a["physics_qvel"][offset]))),
                )
                force_error = max(
                    force_error,
                    float(np.max(np.abs(forces - a["physics_jaw_n"][offset]))),
                    abs(other - a["physics_other_contact_n"][offset]),
                )
                assert abs(t - a["physics_time"][offset]) < 1e-10
                assert abs(penetration - a["physics_penetration_m"][offset]) < 1e-10
                offset += 1
        steps += offset
        print(f"{key}: score/physical replay verified", flush=True)
    assert state_error < 1e-9 and force_error < 1e-7 and geometry_error < 1e-9
    counts = {
        m: {
            "success": sum(r["success"] for r in record.rows.values() if r["method"] == m),
            "total": sum(r["method"] == m for r in record.rows.values()),
        }
        for m in record.methods
    }
    return {
        "episodes": len(record.rows),
        "physical_steps": steps,
        "max_state_error": state_error,
        "max_force_error_n": force_error,
        "max_geometry_error_m": geometry_error,
        "counts": counts,
        "archive_sha256": record.summary["archive_sha256"],
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "原评分、真实请求和逐2ms状态/力核验；不证明视觉或实机有效",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
