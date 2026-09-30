"""Recompute scoring and replay every mj_step input, state and applied contact force."""

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
        raise ValueError("Use the archived MuJoCo version for physical replay")
    maximum_state = maximum_force = maximum_geometry = 0.0
    steps = 0
    for key, row in record.rows.items():
        config = SceneConfig(**row["config"])
        sim = SO101Simulation(config)
        values = record.arrays[key]
        score = score_episode(values, config, row["plan_rejected"])
        for metric, value in score.items():
            if isinstance(value, (str, bool)):
                assert row[metric] == value, (key, metric)
            else:
                np.testing.assert_allclose(row[metric], value, rtol=0, atol=1e-10)
        offset = 0
        for i, command in enumerate(values["commands"][1:], start=1):
            observation = sim.step(command)
            maximum_geometry = max(
                maximum_geometry,
                float(np.max(np.abs(observation["tcp"] - values["tcp_xyz"][i]))),
                float(np.max(np.abs(observation["object"] - values["object_xyz"][i]))),
            )
            for time_s, qpos, qvel, forces, other, penetration in sim.last_physics:
                maximum_state = max(
                    maximum_state,
                    float(np.max(np.abs(qpos - values["physics_qpos"][offset]))),
                    float(np.max(np.abs(qvel - values["physics_qvel"][offset]))),
                )
                maximum_force = max(
                    maximum_force,
                    float(np.max(np.abs(forces - values["physics_jaw_n"][offset]))),
                    abs(other - values["physics_other_contact_n"][offset]),
                )
                assert abs(time_s - values["physics_time"][offset]) < 1e-10
                assert abs(penetration - values["physics_penetration_m"][offset]) < 1e-10
                offset += 1
        steps += offset
        print(f"{key}: score/state/force replay verified", flush=True)
    assert maximum_state < 1e-9 and maximum_force < 1e-7 and maximum_geometry < 1e-9
    return {
        "episodes": len(record.rows),
        "physical_steps": steps,
        "max_state_error": maximum_state,
        "max_force_error_n": maximum_force,
        "max_geometry_error_m": maximum_geometry,
        "archive_sha256": record.summary["archive_sha256"],
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_note": "输入重放验证状态/接触和评分；不证明真实机器人参数或视觉估计有效",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.directory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
