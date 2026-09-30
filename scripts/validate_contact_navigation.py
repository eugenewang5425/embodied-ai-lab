"""Independent direct MuJoCo replay of every recorded physical substep."""

import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/contact_navigation_v6"


def main():
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    p = summary["protocol"]
    assert len(summary["rows"]) == len(p["cases"]) == 39
    assert digest(DATA / "protocol.json") == summary["protocol_sha256"]
    assert digest(ROOT / "docs/69-contact-round6-plan.md") == p["registration_sha256"]
    for name, sha in p["source_sha256"].items():
        assert digest(ROOT / name) == digest(DATA / "source" / name) == sha
    substeps = frames = 0
    state_error = contact_error = 0.0
    for row in summary["rows"]:
        a = read_checked(DATA / row["file"], row["sha256"])
        m = mujoco.MjModel.from_xml_string(str(a["model_xml"].item()))
        d = mujoco.MjData(m)
        d.qpos[:] = a["truth"][0]
        mujoco.mj_forward(m, d)
        assert a["substep_state"].shape == (len(a["truth"]) - 1, 50, 6)
        actual = []
        continuous = 0.0
        for states, forces, metrics in zip(
            a["substep_state"], a["substep_force"], a["substep_contact"], strict=True
        ):
            assert np.linalg.norm(forces[:, :2], axis=1).max() <= 6.4 + 1e-10
            assert np.abs(forces[:, 2]).max() <= 0.59532 + 1e-10
            for state, force, expected in zip(states, forces, metrics, strict=True):
                d.qfrc_applied[:] = force
                mujoco.mj_step(m, d)
                error = float(np.max(np.abs(np.r_[d.qpos, d.qvel] - state)))
                state_error = max(state_error, error)
                assert error < 1e-9
                normal = []
                penetration = []
                for j in range(d.ncon):
                    c = d.contact[j]
                    if (m.geom_bodyid[c.geom1] == 0) == (m.geom_bodyid[c.geom2] == 0):
                        continue
                    w = np.zeros(6)
                    mujoco.mj_contactForce(m, d, j, w)
                    if w[0] > 1e-6:
                        normal.append(float(w[0]))
                        penetration.append(max(0.0, -float(c.dist)))
                continuous = continuous + 0.002 if normal else 0.0
                measured = np.array(
                    [
                        len(normal),
                        max(normal, default=0),
                        sum(normal),
                        max(penetration, default=0),
                        continuous,
                    ]
                )
                contact_error = max(contact_error, float(np.max(abs(measured - expected))))
                np.testing.assert_allclose(measured, expected, atol=1e-7, rtol=0)
                actual.append(measured)
                substeps += 1
        q = a["substep_state"]
        if len(q):
            poses = q[:, -1, :3].copy()
            poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))
            np.testing.assert_allclose(poses, a["truth"][1:], atol=1e-10)
            np.testing.assert_allclose(q[:, -1, 3:], a["world_velocity"][1:], atol=1e-10)
            c = np.array(actual)
            active = c[:, 0] > 0
            events = int(np.sum(active & ~np.r_[False, active[:-1]]))
            peak, depth, duration = c[:, 1].max(), c[:, 3].max(), c[:, 4].max()
            np.testing.assert_allclose(
                a["contact_interval"][1:, 1], a["substep_contact"][:, :, 1].max(axis=1)
            )
            assert row["contact_events"] == events
            assert abs(row["contact_impulse_ns"] - c[:, 2].sum() * 0.002) < 1e-7
            assert abs(row["contact_total_s"] - active.sum() * 0.002) < 1e-8
        else:
            peak = depth = duration = events = 0
        mild = peak <= 40 and depth <= 0.005 and duration <= 0.5 + 1e-12
        assert row["severe_contact"] == (not mild)
        stopped = np.linalg.norm(d.qvel[:2]) < 0.01 and abs(d.qvel[2]) < 0.01
        distance = np.linalg.norm(d.qpos[:2] - a["goal"][-1])
        assert row["physically_stopped"] == stopped
        assert abs(row["final_goal_distance_m"] - distance) < 1e-10
        assert row["passed"] == bool(
            row["reached"]
            and stopped
            and distance < 0.25
            and mild
            and (row["contact_allowed"] or events == 0)
        )
        assert not row["false_arrival"]
        cap = np.minimum(0.6, 0.8 * np.linalg.norm(a["estimate"][:, :2] - a["goal"], axis=1))
        assert (abs(a["requested_command"][:, 0]) <= cap + 1e-10).all()
        frames += len(a["truth"])
    paired = 0
    for row in summary["rows"]:
        if row["method"] != "zero_strict":
            continue
        other = next(
            r
            for r in summary["rows"]
            if r["method"] == "zero_light"
            and all(
                r[k] == row[k] for k in ("layout", "obstacle", "seed", "condition_key", "profile")
            )
        )
        first = read_checked(DATA / row["file"], row["sha256"])
        second = read_checked(DATA / other["file"], other["sha256"])
        for key in first:
            if key != "control_wall_s":
                assert np.array_equal(
                    first[key], second[key], equal_nan=first[key].dtype.kind not in "US"
                ), key
        paired += 1
    result = {
        "episodes": 39,
        "identical_zero_margin_pairs": paired,
        "frames": frames,
        "physics_steps": substeps,
        "max_state_replay_error": state_error,
        "max_contact_replay_error": contact_error,
        "summary_sha256": digest(DATA / "summary.json"),
        "checks": [
            "direct 2ms mj_step from stored applied force",
            "contacts independently extracted in contact frame",
            "force budgets",
            "physical stop including lateral velocity",
            "true arrival and contact policy",
            "frame/substep anchors",
            "approach cap",
            "all archive and frozen source hashes",
        ],
    }
    save_json(ROOT / "docs/benchmarks/contact-navigation-v6-validation.json", result)
    save_json(ROOT / "docs/benchmarks/contact-navigation-v6.json", summary)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
