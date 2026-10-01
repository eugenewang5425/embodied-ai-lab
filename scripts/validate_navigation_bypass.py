"""Independent direct MuJoCo replay of every recorded physical substep."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.experiments.navigation_bypass_study import feasibility_witness, map_scene
from embodied_learning.navigation_geometry import NavigationRig
from embodied_learning.navigation_map_update import EvidenceMap
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]


def validate(round_number):
    DATA = ROOT / f"results/navigation_reinforcement_v{round_number}"
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    p = summary["protocol"]
    assert len(summary["rows"]) == len(p["cases"]) == 24
    assert digest(DATA / "protocol.json") == summary["protocol_sha256"]
    assert digest(ROOT / "docs/navigation-bypass-round11-plan.md") == p["registration_sha256"]
    for name, sha in p["source_sha256"].items():
        assert digest(ROOT / name) == digest(DATA / "source" / name) == sha
    for profile, witness in p["feasibility_witnesses"].items():
        assert witness == feasibility_witness(profile)
    for profile in p["feasibility_witnesses"]:
        for seed in (29, 30, 31):
            pair = [r for r in summary["rows"] if r["profile"] == profile and r["seed"] == seed]
            a, b = [read_checked(DATA / r["file"], r["sha256"]) for r in pair]
            for key in ("truth", "ranges", "hits", "depth", "depth_valid"):
                np.testing.assert_array_equal(a[key][:5], b[key][:5])
    substeps = frames = 0
    state_error = contact_error = 0.0
    dev_directory = ROOT / "results/navigation_bypass_development_v11_metric"
    dev = json.loads((dev_directory / "summary.json").read_text(encoding="utf8"))
    assert digest(dev_directory / "protocol.json") == dev["protocol_sha256"]
    for name, sha in dev["protocol"]["source_sha256"].items():
        assert digest(dev_directory / "source" / name) == sha == p["source_sha256"][name]
    counterexample = next(
        r
        for r in dev["rows"]
        if r["profile"] == "bypass_crate" and r["method"] == "lidar_depth" and r["seed"] == 0
    )
    physical_cases = [(DATA, row) for row in summary["rows"]] + [(dev_directory, counterexample)]
    for directory, row in physical_cases:
        a = read_checked(directory / row["file"], row["sha256"])
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
        cap = np.minimum(
            row["fixed_speed_cap_m_s"],
            0.8 * np.linalg.norm(a["estimate"][:, :2] - a["goal"], axis=1),
        )
        assert (abs(a["requested_command"][:, 0]) <= cap + 1e-10).all()
        frames += len(a["truth"])
    map_frames = cleared_checks = 0
    if round_number == 11:
        for row in summary["rows"]:
            a = read_checked(DATA / row["file"], row["sha256"])
            rig = NavigationRig(
                **{**p["rig"], "width": row["depth_width"], "height": row["depth_height"]}
            )
            prior, _ = map_scene(row["profile"])
            memory = EvidenceMap(prior, rig, True, True)
            for frame in range(len(a["truth"])):
                before = memory.occupied.copy()
                counts = {"added": 0, "removed": 0, "free_evidence": 0}
                depth_input, valid_input = a["depth"][frame], a["depth_valid"][frame]
                if row["method"] == "lidar_only":
                    depth_input = np.full_like(depth_input, np.nan)
                    valid_input = np.zeros_like(valid_input)
                if a["reason"][frame] != "abort_braking":
                    counts = memory.update(
                        a["estimate"][frame],
                        a["ranges"][frame],
                        a["hits"][frame],
                        depth_input,
                        valid_input,
                    )
                np.testing.assert_array_equal(memory.grid, a["map_frames"][frame])
                np.testing.assert_array_equal(
                    memory.grid & a["prior_map"], a["planning_prior_frames"][frame]
                )
                np.testing.assert_array_equal(list(counts.values()), a["map_counts"][frame])
                removed = before & ~memory.occupied
                centers = memory.centers[removed]
                if row["method"] == "lidar_only":
                    assert not len(centers), "camera clearing leaked into LiDAR-only group"
                if len(centers):
                    assert frame >= 2
                    # Independent camera projection and 9-pixel depth minimum.
                    # Test all removed voxels against each of three original scans.
                    for history in range(frame - 2, frame + 1):
                        pose = a["estimate"][history]
                        c, sn = np.cos(pose[2]), np.sin(pose[2])
                        delta = centers[:, :2] - pose[:2]
                        forward = delta[:, 0] * c + delta[:, 1] * sn - rig.camera_x_m
                        left = -delta[:, 0] * sn + delta[:, 1] * c
                        u = np.rint(rig.width / 2 - 0.5 - rig.focal * left / forward).astype(int)
                        v = np.rint(
                            rig.height / 2
                            - 0.5
                            - rig.focal * (centers[:, 2] - rig.camera_z_m) / forward
                        ).astype(int)
                        assert (
                            (forward > 0).all()
                            and (u >= 1).all()
                            and (u < rig.width - 1).all()
                            and (v >= 1).all()
                            and (v < rig.height - 1).all()
                        )
                        for du in (-1, 0, 1):
                            for dv in (-1, 0, 1):
                                measurement = a["depth"][history, v + dv, u + du]
                                valid = a["depth_valid"][history, v + dv, u + du]
                                usable = valid | np.isclose(measurement, 8.0, atol=1e-10, rtol=0)
                                assert usable.all() and np.isfinite(measurement).all()
                                assert (measurement > forward + 0.12).all()
                    cleared_checks += len(centers)
                map_frames += 1
    result = {
        "episodes": len(physical_cases),
        "formal_episodes": len(summary["rows"]),
        "development_counterexample": counterexample,
        "development_summary_sha256": digest(dev_directory / "summary.json"),
        "round": round_number,
        "map_rebuilt_frames": map_frames,
        "independently_checked_cleared_voxels": cleared_checks,
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
            "sampled feasible bypass, not supplied to controllers",
            "paired raw sensors identical during stationary warmup",
            "LiDAR-only map reconstructed without camera input",
        ],
    }
    save_json(
        ROOT / f"docs/benchmarks/navigation-reinforcement-v{round_number}-validation.json", result
    )
    save_json(ROOT / f"docs/benchmarks/navigation-reinforcement-v{round_number}.json", summary)
    save_json(ROOT / "docs/benchmarks/navigation-bypass-development-v11-metric.json", dev)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(11,), required=True)
    validate(parser.parse_args().round)
