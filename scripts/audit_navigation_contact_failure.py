"""Diagnose the frozen round-eleven development contact, without changing control.

Stored forces reproduce the factual episode. Stop trials are causal offline
counterfactuals from saved qpos/qvel and already pending requests, not new scores.
Actual boxes are used only by physics and independent geometry evaluation.
"""

import json
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from embodied_learning.braking_control import brake_step, reachable
from embodied_learning.contact_motion import ContactPlant
from embodied_learning.execution_safety import queued_stop_path
from embodied_learning.experiments.footprint_tracking import digest, save_json
from embodied_learning.experiments.navigation_bypass_study import map_scene
from embodied_learning.navigation_geometry import (
    NavigationRig,
    body_points,
    collision_clearance,
    depth_points,
    lidar_obstacle_points,
    point_clearance,
    rotation,
    world_points,
)
from embodied_learning.navigation_map_update import EvidenceMap
from embodied_learning.plotting import configure_plot_font
from embodied_learning.replay_viewer.campus import read_checked

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/navigation_bypass_development_v11_metric"
OUT = ROOT / "docs/benchmarks/navigation-contact-diagnosis-v11.json"
FIGURE = ROOT / "docs/img/lesson69-round11-contact-diagnosis.png"


def pending_requests(a, frame, delay=2):
    """Recover intents issued before this frame, without reading future controls."""
    return [
        (a["requested_command"][k].copy(), bool(a["braking_active"][k]), k)
        for k in range(frame - delay, frame)
    ]


def absolute_path(relative, pose):
    out = relative.copy()
    out[:, :2] = out[:, :2] @ rotation(pose[2]).T + pose[:2]
    out[:, 2] += pose[2]
    return out


def contact_wrenches(model, data):
    result = []
    for j in range(data.ncon):
        c = data.contact[j]
        if (model.geom_bodyid[c.geom1] == 0) == (model.geom_bodyid[c.geom2] == 0):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, j, wrench)
        if wrench[0] > 1e-6:
            result.append(
                {
                    "geom_names": [
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
                        for g in (c.geom1, c.geom2)
                    ],
                    "position_m": c.pos.tolist(),
                    "normal": c.frame[:3].tolist(),
                    "normal_force_n": float(wrench[0]),
                    "penetration_m": max(0.0, -float(c.dist)),
                }
            )
    return result


def replay_contact(a):
    model = mujoco.MjModel.from_xml_string(str(a["model_xml"].item()))
    data = mujoco.MjData(model)
    data.qpos[:] = a["truth"][0]
    mujoco.mj_forward(model, data)
    error = peak_error = 0.0
    first = None
    for frame, (states, forces, expected) in enumerate(
        zip(a["substep_state"], a["substep_force"], a["substep_contact"], strict=True)
    ):
        for substep, (state, force, contact) in enumerate(
            zip(states, forces, expected, strict=True)
        ):
            before = np.r_[data.qpos, data.qvel].copy()
            data.qfrc_applied[:] = force
            mujoco.mj_step(model, data)
            error = max(error, float(np.max(np.abs(np.r_[data.qpos, data.qvel] - state))))
            pairs = contact_wrenches(model, data)
            peak = max((c["normal_force_n"] for c in pairs), default=0.0)
            peak_error = max(peak_error, abs(peak - contact[1]))
            assert len(pairs) == contact[0]
            if pairs and first is None:
                first = {
                    "interval_frame": frame,
                    "substep": substep,
                    "reported_sample_time_s": frame * 0.1 + (substep + 1) * 0.002,
                    "pre_step_state": before.tolist(),
                    "post_step_state": state.tolist(),
                    "contacts": pairs,
                }
    assert first is not None and error < 1e-9 and peak_error < 1e-8
    return first, error, peak_error


def physical_stop(a, boxes, rig, frame, immediate=False, disable_new_contact=False):
    """Only timing of the zero/brake request changes; gains/force limits stay fixed."""
    plant = ContactPlant(boxes, rig, a["truth"][frame])
    plant.data.qvel[:] = a["world_velocity"][frame]
    if disable_new_contact:
        # Diagnostic rollout of the unimpeded motor response, not a passable world.
        g = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_{len(boxes) - 1}")
        plant.model.geom_contype[g] = plant.model.geom_conaffinity[g] = 0
    mujoco.mj_forward(plant.model, plant.data)
    queue = [] if immediate else pending_requests(a, frame)
    states = [np.r_[plant.data.qpos, plant.data.qvel].copy()]
    contacts = []
    for tick in range(50):
        target, brake, _ = queue[tick] if tick < len(queue) else (np.zeros(2), True, frame)
        command = (
            brake_step(plant.velocity, rig, True)
            if brake
            else reachable(plant.velocity, target, rig)
        )
        state, _, contact = plant.step(command)
        states.extend(state)
        contacts.extend(contact)
        if plant.stopped and tick >= len(queue):
            break
    else:
        raise AssertionError("physical stop did not settle within five seconds")
    states, contacts = np.asarray(states), np.asarray(contacts)
    result = {
        "frame": frame,
        "time_s": frame * 0.1,
        "immediate_local_brake": immediate,
        "new_obstacle_contact_disabled_for_model_audit": disable_new_contact,
        "pending_request_frames": [q[2] for q in queue],
        "elapsed_s": len(contacts) * 0.002,
        "travel_until_stop_m": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()),
        "end_displacement_m": (states[-1, :2] - states[0, :2]).tolist(),
        "sampled_geometry_separation_m": min(
            collision_clearance(p, boxes, rig) for p in states[:, :3]
        ),
        "contact_peak_force_n": float(contacts[:, 1].max()),
        "physically_stopped": bool(plant.stopped),
    }
    return result, states


def polygon(pose, rig):
    return world_points(
        [
            [-rig.rear_m, -rig.half_width_m, 0],
            [rig.front_m, -rig.half_width_m, 0],
            [rig.front_m, rig.half_width_m, 0],
            [-rig.rear_m, rig.half_width_m, 0],
        ],
        pose,
    )[:, :2]


def plot(a, lidar, rig, snapshots, first, trials, prediction):
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    obstacle = a["actual_boxes"][-1]
    impact_pose = np.array(first["pre_step_state"][:3])
    plan = a["plan"][195]
    plan = plan[np.isfinite(plan[:, 0])]
    ax = axes[0, 0]
    ax.add_patch(Rectangle((obstacle[0], obstacle[2]), 0.8, 1.2, fc="#efd5af", ec="#af7823"))
    world = snapshots[200]["world"]
    near = (abs(world[:, 0] - 5.6) < 0.1) & (abs(world[:, 1] - 5.4) < 0.18)
    ax.scatter(*world[near, :2].T, s=13, c="#475569", alpha=0.5, label="地图中保留的观测点")
    ax.plot(*plan[:2].T, "--", c="#c78a00", lw=2, label="融合组当前规划")
    ax.plot(*a["truth"][190:, :2].T, c="#db4b54", lw=2, label="融合组实际轨迹")
    ax.plot(*lidar["truth"][190:220, :2].T, c="#007e9e", lw=2, label="仅雷达实际轨迹")
    for pose, color in ((a["truth"][200], "#238b63"), (impact_pose, "#db4b54")):
        ax.add_patch(Polygon(polygon(pose, rig), fill=False, ec=color, lw=2))
    ax.set(
        title="A  箱体已被看见，融合组路线更贴边",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(4.94, 5.76),
        ylim=(4.98, 5.55),
        aspect="equal",
    )
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[0, 1]
    frames = np.arange(194, len(a["truth"]))
    t = frames * 0.1
    ax.step(
        t, a["requested_command"][frames, 0], where="post", color="#d99b00", label="规划端前进请求"
    )
    ax.step(t, a["commands"][frames, 0], where="post", color="#64748b", label="电机速度目标")
    states = a["substep_state"][194:].reshape(-1, 6)
    measured = states[:, 3] * np.cos(states[:, 2]) + states[:, 4] * np.sin(states[:, 2])
    measured_time = 19.4 + (np.arange(len(states)) + 1) * 0.002
    ax.plot(measured_time, measured, color="#2076ba", lw=2, label="车身实测前进速度（500Hz）")
    for time, color in ((20.0, "#d99b00"), (20.2, "#64748b"), (20.418, "#db4b54")):
        ax.axvline(time, color=color, ls=":", alpha=0.8)
    ax.set(
        title="B  20.0s请求刹车，20.2s执行，20.418s接触",
        xlabel="时间 (s)",
        ylabel="前进速度 (m/s)",
        ylim=(-0.015, 0.225),
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ideal_distance = np.linalg.norm(np.diff(prediction[:, :2], axis=0), axis=1).sum()
    values = [
        ideal_distance,
        trials["queued_unimpeded"]["travel_until_stop_m"],
        trials["immediate"]["travel_until_stop_m"],
    ]
    labels = ["原预测：排队后停车", "物理诊断：排队后停车", "物理对照：立即本地停车"]
    ax.barh(labels, np.array(values) * 1000, color=["#d99b00", "#db4b54", "#238b63"], height=0.5)
    for i, value in enumerate(values):
        ax.text(value * 1000 + 1, i, f"{value * 1000:.1f} mm", va="center", fontsize=11)
    ax.invert_yaxis()
    ax.set(
        title="C  从同一20.0s状态，实际停车行程更长",
        xlabel="行走到停稳的距离 (mm)\n物理诊断项关闭新箱体接触响应，仅用于测量停车模型",
        xlim=(0, 96),
    )
    ax.xaxis.label.set_fontsize(10)
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1, 1]
    ax.add_patch(Rectangle((5.6, 5.4), 0.8, 1.2, fc="#efd5af", ec="#af7823"))
    aligned = impact_pose.copy()
    aligned[2] = np.arctan2(*(plan[1] - plan[0])[::-1])
    for pose, color, name in (
        (aligned, "#238b63", "同位置按规划朝向摆放（几何诊断）"),
        (impact_pose, "#db4b54", "接触前真实底盘朝向"),
    ):
        ax.add_patch(Polygon(polygon(pose, rig), fill=False, ec=color, lw=2, label=name))
        corner = world_points([[rig.front_m, rig.half_width_m, 0]], pose)[0]
        ax.plot(*corner[:2], "o", color=color, ms=6)
    contact = first["contacts"][0]["position_m"]
    ax.plot(*contact[:2], "x", color="black", ms=9, label="前左角接触点")
    ax.set(
        title="D  前左角先撞上，约2.4°角差吃掉余量",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(5.55, 5.65),
        ylim=(5.36, 5.455),
        aspect="equal",
    )
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.legend(loc="upper left", fontsize=8)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)


def main():
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf8"))
    assert digest(DATA / "protocol.json") == summary["protocol_sha256"]
    for name, sha in summary["protocol"]["source_sha256"].items():
        assert digest(DATA / "source" / name) == digest(ROOT / name) == sha
    row = next(
        r
        for r in summary["rows"]
        if r["profile"] == "bypass_crate" and r["method"] == "lidar_depth" and r["seed"] == 0
    )
    comparison = next(
        r
        for r in summary["rows"]
        if r["profile"] == "bypass_crate" and r["method"] == "lidar_only" and r["seed"] == 0
    )
    a = read_checked(DATA / row["file"], row["sha256"])
    lidar = read_checked(DATA / comparison["file"], comparison["sha256"])
    for key in ("truth", "ranges", "hits", "depth", "depth_valid"):
        np.testing.assert_array_equal(a[key][:5], lidar[key][:5])
    rig = NavigationRig(
        **{**summary["protocol"]["rig"], "width": row["depth_width"], "height": row["depth_height"]}
    )
    prior, boxes = map_scene(row["profile"])
    np.testing.assert_array_equal(boxes, a["actual_boxes"])
    first, state_error, peak_error = replay_contact(a)

    memory = EvidenceMap(prior, rig, True, True)
    snapshots, predictions, map_checks = {}, [], 0
    recent_removed = 0
    for frame in range(first["interval_frame"] + 1):
        before, old_points = memory.occupied.copy(), memory.points.copy()
        memory.update(
            a["estimate"][frame],
            a["ranges"][frame],
            a["hits"][frame],
            a["depth"][frame],
            a["depth_valid"][frame],
        )
        np.testing.assert_array_equal(memory.grid, a["map_frames"][frame])
        np.testing.assert_array_equal(
            memory.grid & a["prior_map"], a["planning_prior_frames"][frame]
        )
        np.testing.assert_array_equal(list(memory.last.values()), a["map_counts"][frame])
        map_checks += 1
        removed = old_points[before & ~memory.occupied]
        near = (abs(removed[:, 0] - 6) < 0.7) & (abs(removed[:, 1] - 6) < 0.9)
        if frame >= first["interval_frame"] - 10:
            recent_removed += int(near.sum())
        if frame not in (195, 198, 199, 200, 201, 202, 203, 204):
            continue
        world = memory.points[memory.occupied].copy()
        local = body_points(world, a["estimate"][frame])
        local = local[np.linalg.norm(local[:, :2], axis=1) < 2.5]
        raw = np.vstack(
            (
                lidar_obstacle_points(a["ranges"][frame], a["hits"][frame], rig),
                depth_points(a["depth"][frame], a["depth_valid"][frame], rig),
            )
        )
        relative, _ = queued_stop_path(
            a["incoming_velocity"][frame], pending_requests(a, frame), np.zeros(2), True, rig
        )
        absolute = absolute_path(relative, a["truth"][frame])
        predictions.append(
            {
                "frame": frame,
                "time_s": frame * 0.1,
                "stored_points_forecast_clearance_m": point_clearance(local, relative, rig),
                "raw_points_forecast_clearance_m": point_clearance(raw, relative, rig),
                "oracle_forecast_separation_m": min(
                    collision_clearance(p, boxes, rig) for p in absolute
                ),
                "predicted_stop_distance_m": float(
                    np.linalg.norm(np.diff(relative[:, :2], axis=0), axis=1).sum()
                ),
                "local_guard_overrode": bool(a["local_override"][frame]),
            }
        )
        snapshots[frame] = {"world": world, "relative": relative, "absolute": absolute}
    assert recent_removed == 0

    trials, trial_states = {}, {}
    for name, frame, immediate, disable in (
        ("queued", 200, False, False),
        ("queued_unimpeded", 200, False, True),
        ("immediate", 200, True, False),
        ("one_tick_earlier", 199, False, False),
    ):
        trials[name], trial_states[name] = physical_stop(a, boxes, rig, frame, immediate, disable)
    assert abs(trials["queued"]["contact_peak_force_n"] - row["contact_peak_force_n"]) < 1e-8
    factual_tail = np.vstack(
        (np.r_[a["truth"][200], a["world_velocity"][200]], a["substep_state"][200:].reshape(-1, 6))
    )
    np.testing.assert_allclose(trial_states["queued"], factual_tail, rtol=0, atol=1e-9)
    assert (
        trials["immediate"]["contact_peak_force_n"]
        == trials["one_tick_earlier"]["contact_peak_force_n"]
        == 0
    )
    assert trials["queued_unimpeded"]["sampled_geometry_separation_m"] < 0
    for name in ("immediate", "one_tick_earlier"):
        assert (
            trials[name]["physically_stopped"] and trials[name]["sampled_geometry_separation_m"] > 0
        )
    plan = a["plan"][195]
    plan = plan[np.isfinite(plan[:, 0])]
    edge = plan[1] - plan[0]
    yaw = np.arctan2(edge[1], edge[0])
    poses = np.column_stack((plan[0] + np.linspace(0, 1, 4001)[:, None] * edge, np.full(4001, yaw)))
    plan_min = min(collision_clearance(p, boxes, rig) for p in poses)
    impact = np.array(first["pre_step_state"][:3])
    aligned = impact.copy()
    aligned[2] = yaw
    brake = int(np.flatnonzero(a["reason"] == "queue_body_brake")[0])
    assert brake == 200 and a["executed_request_frame"][202] == 200
    assert np.isclose(a["commands"][202, 0], a["incoming_velocity"][202, 0] - 0.08)
    report = {
        "scope": "frozen development seed0 crate fusion diagnosis, not a new formal benchmark",
        "selected_by": "the only fusion heavy_contact_abort in the frozen development batch",
        "script_sha256": digest(Path(__file__)),
        "summary_sha256": digest(DATA / "summary.json"),
        "archive": row["file"],
        "archive_sha256": row["sha256"],
        "comparison_archive": comparison["file"],
        "comparison_archive_sha256": comparison["sha256"],
        "source_sha256": summary["protocol"]["source_sha256"],
        "physics_steps_replayed": int(np.prod(a["substep_force"].shape[:2])),
        "map_frames_rebuilt": map_checks,
        "state_replay_max_error": state_error,
        "peak_force_replay_max_error": peak_error,
        "first_contact": first,
        "recorded_peak_normal_force_n": row["contact_peak_force_n"],
        "recorded_max_penetration_m": row["contact_penetration_m"],
        "recorded_longest_contact_s": row["contact_longest_s"],
        "peak_force_abort_threshold_n": 40,
        "first_brake_request_frame": brake,
        "first_brake_execution_frame": 202,
        "first_local_guard_override_frame": int(np.flatnonzero(a["local_override"])[0]),
        "prediction_checks": predictions,
        "counterfactual_stop_trials": trials,
        "box_neighborhood_points_removed_during_last_1s": recent_removed,
        "last_plan_first_edge_sampled_separation_m": plan_min,
        "last_plan_first_edge_yaw_rad": float(yaw),
        "first_contact_heading_minus_plan_deg": float(np.rad2deg(impact[2] - yaw)),
        "first_contact_pose_separation_m": collision_clearance(impact, boxes, rig),
        "same_xy_aligned_to_plan_separation_m": collision_clearance(aligned, boxes, rig),
        "brake_first_interval_requested_speed_m_s": float(a["commands"][202, 0]),
        "brake_first_interval_measured_end_speed_m_s": float(a["incoming_velocity"][203, 0]),
        "brake_first_interval_average_deceleration_m_s2": float(
            (a["incoming_velocity"][202, 0] - a["incoming_velocity"][203, 0]) / 0.1
        ),
        "counterfactual_limits": [
            "same saved qpos and all three qvel components, same force servo and contact settings",
            "pending intents reconstructed solely from requests issued before the trial frame",
            "immediate trial changes braking timing; it stops before the goal and is not task success",
            "contact-disabled trial measures motor response, with original geometry retained for scoring",
            "aligned body pose is a geometry diagnostic, not a simulated new controller",
            "raw/voxel point comparisons and oracle evaluation do not identify every upstream map effect",
        ],
    }
    plot(a, lidar, rig, snapshots, first, trials, snapshots[200]["absolute"])
    report["figure_sha256"] = digest(FIGURE)
    save_json(OUT, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "physics_steps_replayed",
                    "map_frames_rebuilt",
                    "state_replay_max_error",
                    "peak_force_replay_max_error",
                    "last_plan_first_edge_sampled_separation_m",
                    "first_contact_heading_minus_plan_deg",
                    "same_xy_aligned_to_plan_separation_m",
                    "brake_first_interval_average_deceleration_m_s2",
                    "counterfactual_stop_trials",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
