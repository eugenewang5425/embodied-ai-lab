"""Full-rotation recovery: down-start, below-horizontal starts, and active-force knockdowns."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.experiments.lqr_comparison import SETTLED_TOLERANCES
from embodied_learning.plotting import configure_plot_font
from embodied_learning.swingup import (
    MODEL_PATH,
    SAFE_CART_POSITION,
    HybridSwingupController,
    SwingupParameters,
    design_swingup_lqr,
    make_swingup_environment,
    wrap_angle,
)


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    angle_deg: float = -180.0
    force_n: float = 0.0
    push_start_s: float = 5.0
    push_duration_s: float = 0.4


SCENARIOS = (
    Scenario("down", "正下方 -180°"),
    Scenario("below_left", "左下方 -120°", -120),
    Scenario("below_right", "右下方 +120°", 120),
    Scenario("push_right", "向右强推 +400 N", 0, 400),
    Scenario("push_left", "向左强推 -400 N", 0, -400),
    Scenario("overload", "超限失败 +600 N", 0, 600),
)


def run_scenario(scenario: Scenario, design=None, horizon: int = 750):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not np.isfinite(
        [scenario.angle_deg, scenario.force_n, scenario.push_start_s, scenario.push_duration_s]
    ).all():
        raise ValueError("scenario values must be finite")
    if scenario.push_start_s < 0 or scenario.push_duration_s <= 0:
        raise ValueError("invalid push timing")
    design = design or design_swingup_lqr()
    dt = design.dt
    start, duration = round(scenario.push_start_s / dt), round(scenario.push_duration_s / dt)
    if not np.allclose(
        [start * dt, duration * dt],
        [scenario.push_start_s, scenario.push_duration_s],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("push timing must align with the control period")
    schedule = np.zeros(horizon)
    schedule[start : start + duration] = scenario.force_n
    env = make_swingup_environment(max_episode_steps=horizon)
    try:
        env.reset(seed=0)
        state = design.controller.reference.copy()
        state[1] += np.deg2rad(scenario.angle_deg)
        env.unwrapped.set_state(state[:2], state[2:])
        controller = HybridSwingupController(env.unwrapped.model, design)
        states, controls, forces, modes, energies = [state.copy()], [], [], [], []
        for force in schedule:
            # Controller remains active throughout; receives only s[k], not force[k].
            action = controller.action(state)
            env.unwrapped.data.qfrc_applied[0] = force
            state, _, terminated, truncated, info = env.step(action)
            controls.append(float(action[0]))
            forces.append(force)
            modes.append(controller.mode)
            energies.append(controller.last_energy)
            states.append(state.copy())
            if terminated or truncated:
                break
        arrays = {
            "states": np.asarray(states),
            "controls": np.asarray(controls),
            "applied_force_n": np.asarray(forces),
            "scheduled_force_n": schedule,
            "modes": np.asarray(modes),
            "energy_j": np.asarray(energies),
            "end_flags": np.array([terminated, truncated]),
        }
        metadata = {
            **asdict(scenario),
            "failure_reason": info["failure_reason"],
            "target_energy_j": controller.gravity_energy,
            "hinge_inertia_kg_m2": controller.hinge_inertia,
        }
        return arrays, metadata
    finally:
        env.close()


def recovery_metrics(arrays, metadata, reference, dt):
    states, controls, forces = (arrays[k] for k in ("states", "controls", "applied_force_n"))
    errors = states - reference
    errors[:, 1] = wrap_angle(errors[:, 1])
    times = np.arange(len(states)) * dt
    below = np.cos(errors[:, 1]) < 0
    affected = np.flatnonzero(arrays["scheduled_force_n"])
    after = (affected[-1] + 1) * dt if len(affected) else 0.0
    in_tolerance = np.all(np.abs(errors) <= SETTLED_TOLERANCES, axis=1)
    final_tail = np.logical_and.accumulate(in_tolerance[::-1])[::-1]
    candidates = np.flatnonzero(
        final_tail & (times >= after - 1e-10) & (times[-1] - times >= 2.0 - 1e-10)
    )
    settled = (
        float(times[candidates[0]]) if len(candidates) and not arrays["end_flags"][0] else None
    )
    below_times = times[below & (times >= (affected[0] * dt if len(affected) else 0))]
    # States at the end of each nonzero-force interval demonstrate the actual response.
    during_below = np.flatnonzero((forces != 0) & below[1:])
    old_cross = np.flatnonzero(np.abs(states[:, 1]) > 0.2)
    return {
        "steps": len(controls),
        "duration_s": float(times[-1]),
        "terminated": bool(arrays["end_flags"][0]),
        "truncated": bool(arrays["end_flags"][1]),
        "recovered": settled is not None,
        "settled_at_s": settled,
        "recovery_after_push_end_s": settled - after
        if settled is not None and len(affected)
        else None,
        "first_below_horizontal_s": float(below_times[0]) if len(below_times) else None,
        "below_horizontal_during_push": bool(len(during_below)),
        "first_old_angle_limit_crossing_s": float(times[old_cross[0]]) if len(old_cross) else None,
        "peak_absolute_angle_deg": float(np.rad2deg(np.max(np.abs(errors[:, 1])))),
        "max_abs_cart_position_m": float(np.max(np.abs(states[:, 0]))),
        "peak_abs_motor_force_n": float(100 * np.max(np.abs(controls))),
        "opposing_motor_steps_during_push": int(np.sum(forces * controls < -1e-6)),
        "applied_push_steps": int(np.count_nonzero(forces)),
        "mode_transitions": [
            {"time_s": i * dt, "mode": str(mode)}
            for i, mode in enumerate(arrays["modes"])
            if i == 0 or mode != arrays["modes"][i - 1]
        ],
        "final_wrapped_state_error": errors[-1].tolist(),
        "failure_reason": metadata["failure_reason"],
    }


def save_plot(path, archive, report):
    configure_plot_font()
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), layout="constrained", sharex="col")
    ref, dt = np.asarray(report["reference"]), report["dt_s"]
    for col, key, title in [
        (0, "down", "正下方静止 → 摆起 → 扶稳"),
        (1, "push_right", "控制持续开启：+400 N 强推 → 重新摆起"),
    ]:
        s, u, f = (archive[f"{key}_{k}"] for k in ("states", "controls", "applied_force_n"))
        ts, edges = np.arange(len(s)) * dt, np.arange(len(u) + 1) * dt
        axes[0, col].plot(ts, np.cos(s[:, 1] - ref[1]), color="#0f766e")
        axes[0, col].axhspan(-1, 0, alpha=0.1, color="orange")
        axes[0, col].set(title=title, ylabel="杆端相对高度 / 杆长", ylim=(-1.1, 1.1))
        axes[1, col].plot(ts, s[:, 0], color="#2563eb")
        axes[1, col].set(ylabel="小车位置（m）")
        axes[1, col].axhline(SAFE_CART_POSITION, ls="--", color="gray")
        axes[1, col].axhline(-SAFE_CART_POSITION, ls="--", color="gray")
        axes[2, col].stairs(u * 100, edges, label="电机力", color="#2563eb")
        axes[2, col].stairs(f, edges, label="外部推力", color="#ea580c")
        axes[2, col].set(ylabel="水平力（N）")
        axes[2, col].legend()
        modes = archive[f"{key}_modes"]
        axes[3, col].stairs(np.where(modes == "balance", 2, np.where(modes == "kick", 0, 1)), edges)
        axes[3, col].set(
            yticks=[0, 1, 2],
            yticklabels=["启动", "摆起", "LQR"],
            ylabel="当前控制模式",
            xlabel="仿真时间（s）",
        )
        for axis in axes[:, col]:
            axis.grid(alpha=0.2)
            axis.set_xlim(0, min(20, ts[-1]))
            if key == "push_right":
                axis.axvspan(5, 5.4, alpha=0.15, color="orange")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(output: Path, horizon=750):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if horizon < 250:
        raise ValueError("need at least 10 seconds for this protocol")
    design = design_swingup_lqr()
    archive, cases = {}, []
    for scenario in SCENARIOS:
        arrays, metadata = run_scenario(scenario, design, horizon)
        metrics = recovery_metrics(arrays, metadata, design.controller.reference, design.dt)
        cases.append({**metadata, **metrics})
        archive.update({f"{scenario.key}_{key}": value for key, value in arrays.items()})
    report = {
        "experiment": "full_rotation_swingup",
        "schema_version": 1,
        "dt_s": design.dt,
        "horizon_steps": horizon,
        "R": 1.0,
        "reference": design.controller.reference.tolist(),
        "Q": design.q.tolist(),
        "K": design.controller.gain.tolist(),
        "actuator_gear": design.actuator_gear,
        "control_limit": design.controller.control_limit,
        "parameters": asdict(SwingupParameters()),
        "cart_failure_boundary_m": SAFE_CART_POSITION,
        "cart_joint_limits_m": [-2.5, 2.5],
        "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "numpy_version": np.__version__,
        "tolerance_by_state": SETTLED_TOLERANCES.tolist(),
        "minimum_settled_tail_s": 2.0,
        "archive_alignment": "s[0..N]; u/force/mode/energy[k] belong to s[k] and interval [k*dt,(k+1)*dt); scheduled_force may extend past failure",
        "angle_convention": "alpha=joint_angle-reference_angle; 0 up, +/-pi down; physics and archive continuous, feedback/errors wrapped",
        "success_protocol": "all four wrapped state errors within tolerances for final continuous tail >=2s; push recovery measured after scheduled pulse ends; no physical failure",
        "limitations": [
            "Deterministic calibrated demonstrations, not a global recovery guarantee",
            "Full state feedback; no sensor noise, delay or model mismatch in this stage",
            "Failure boundaries checked after each 0.04s control step, not hard collision avoidance",
            "One unactuated full-rotation hinge; no floor or pole-cart collision",
        ],
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_plot(output / "comparison.png", archive, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_experiment(args.output)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for case in report["cases"]:
        print(
            f"{case['key']}: recovered={case['recovered']}, settled={case['settled_at_s']}s, "
            f"max_x={case['max_abs_cart_position_m']:.3f}m, below_during_push={case['below_horizontal_during_push']}, failure={case['failure_reason'] or 'none'}"
        )


if __name__ == "__main__":
    main()
