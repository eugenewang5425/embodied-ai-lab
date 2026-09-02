"""Long-horizon PD/LQR comparison with raw traces and explicit success criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy

from embodied_learning.controllers import PDController
from embodied_learning.controllers.lqr import LQRDesign, design_lqr
from embodied_learning.environments import ENVIRONMENT_ID, make_inverted_pendulum_environment
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode, summarize
from embodied_learning.plotting import configure_plot_font

plt.switch_backend("Agg")

# State order: cart position, joint angle, cart velocity, angular velocity.
SETTLED_TOLERANCES = np.array([0.02, 0.01, 0.02, 0.02])
MIN_SETTLED_SECONDS = 2.0


def episode_metrics(trace: EpisodeTrace, reference: np.ndarray, horizon: int) -> dict:
    states = np.asarray(trace.observations)
    errors = states - reference
    survived = trace.length == horizon and not trace.terminated
    within = np.all(np.abs(errors) <= SETTLED_TOLERANCES, axis=1)
    stays_within = np.logical_and.accumulate(within[::-1])[::-1]
    required_tail = int(np.ceil(MIN_SETTLED_SECONDS / trace.dt))
    candidates = np.flatnonzero(stays_within)
    candidates = candidates[candidates <= trace.length - required_tail]
    settled = bool(survived and len(candidates))
    commands = np.asarray(trace.actions)
    return {
        "seed": trace.seed,
        "steps": trace.length,
        "duration_s": trace.length * trace.dt,
        "terminated": trace.terminated,
        "truncated": trace.truncated,
        "survived_horizon": survived,
        "settled_at_end": settled,
        "settling_time_s": float((candidates[0] + 1) * trace.dt) if settled else None,
        "final_state": states[-1].tolist(),
        "final_error": errors[-1].tolist(),
        "max_abs_position_m": float(np.max(np.abs(states[:, 0]))),
        "max_abs_upright_error_rad": float(np.max(np.abs(errors[:, 1]))),
        "control_rms": float(np.sqrt(np.mean(commands**2))),
        "saturated_steps": int(np.sum(np.abs(commands) >= 3.0 - 1e-6)),
    }


def evaluate(traces: list[EpisodeTrace], design: LQRDesign, horizon: int) -> dict:
    rows = [episode_metrics(t, design.controller.reference, horizon) for t in traces]
    return {
        **summarize(traces, horizon),
        "settled_episodes": sum(row["settled_at_end"] for row in rows),
        "max_final_abs_position_m": max(abs(row["final_state"][0]) for row in rows),
        "episodes_detail": rows,
    }


def _save_plot(
    groups: dict[str, list[EpisodeTrace]],
    displaced: dict[str, EpisodeTrace],
    design: LQRDesign,
    output: Path,
) -> None:
    configure_plot_font()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout="constrained")
    labels = {"random": "随机", "pd": "角度 PD", "lqr": "全状态 LQR"}
    colors = {"random": "#64748b", "pd": "#dc2626", "lqr": "#2563eb"}
    styles = {"pd": "--", "lqr": "-"}
    # A common displaced initial state makes recentering visible. Never join resets.
    for policy, trace in displaced.items():
        values = np.vstack([trace.initial_observation, *trace.observations])
        times = np.arange(len(values)) * trace.dt
        for axis, column, scale in [(axes[0, 0], 0, 1), (axes[0, 1], 1, 180 / np.pi)]:
            series = (values[:, column] - design.controller.reference[column]) * scale
            axis.plot(times, series, styles[policy], color=colors[policy], label=labels[policy])
            if trace.terminated:
                axis.plot(times[-1], series[-1], "x", color=colors[policy], markersize=8)
        # Zoom control commands to the first six seconds to make initial correction visible.
        count = min(trace.length, int(np.ceil(6 / trace.dt)))
        axes[1, 0].plot(
            np.arange(count) * trace.dt,
            trace.actions[:count],
            styles[policy],
            color=colors[policy],
            label=labels[policy],
        )
    axes[0, 0].axhline(0, color="gray", linewidth=0.7)
    axes[0, 0].axhline(1, color="gray", linewidth=0.7, linestyle=":", label="导轨上限")
    axes[0, 0].axhline(-1, color="gray", linewidth=0.7, linestyle=":")
    axes[0, 0].set(title="车能否回到中心？（同一初态偏移 0.20 m）", ylabel="小车位置（m）")
    axes[0, 1].axhline(0, color="gray", linewidth=0.7)
    axes[0, 1].set(title="杆是否接近真实竖直方向？", ylabel="相对竖直的角度（度）")
    axes[1, 0].set(title="前 6 秒的控制输入（100u 对应水平力/N）", ylabel="控制输入 u")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xlabel("仿真时间（s）")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    policies = list(groups)
    means = [np.mean([t.length * t.dt for t in groups[p]]) for p in policies]
    bars = axes[1, 1].bar([labels[p] for p in policies], means, color=[colors[p] for p in policies])
    axes[1, 1].bar_label(bars, labels=[f"{v:.2f} s" for v in means], padding=4)
    ceiling = max(t.length * t.dt for ts in groups.values() for t in ts)
    axes[1, 1].set_ylim(0, ceiling * 1.2)
    axes[1, 1].set(title=f"{len(groups['pd'])} 个默认初态：平均未失败时长", ylabel="时长（s）")
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("从扶住摆杆到小车居中｜× 表示失败，不代表完成任务")
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


def run_comparison(
    output: Path, episodes: int = 20, horizon: int = 1000, control_weight: float = 0.1
) -> dict:
    if episodes < 1 or horizon < 50:
        raise ValueError("episodes must be positive; horizon must be at least 50 (2 seconds)")
    # Avoid silently replacing an earlier experiment or a user result directory.
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory; already exists: {output}")
    design = design_lqr(control_weight=control_weight)
    controllers = {"random": None, "pd": PDController(40, 1), "lqr": design.controller}
    groups = {
        name: [run_episode(name, seed, horizon, controller) for seed in range(episodes)]
        for name, controller in controllers.items()
    }
    # Independent reset seeds, with the same design weights and no tuning on these rollouts.
    heldout = [
        run_episode("lqr", seed, horizon, design.controller)
        for seed in range(episodes, 2 * episodes)
    ]
    initial = design.controller.reference + np.array([0.2, 0.05, 0.0, 0.0])
    displaced = {
        name: run_episode(name, 0, horizon, controllers[name], initial_state=initial)
        for name in ("pd", "lqr")
    }
    env = make_inverted_pendulum_environment()
    try:
        model_hash = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
    finally:
        env.close()
    closed_loop = design.a - design.b @ design.controller.gain
    summary = {
        "schema_version": 1,
        "environment": ENVIRONMENT_ID,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "model_xml_sha256": model_hash,
        "horizon_steps": horizon,
        "dt_s": design.dt,
        "frame_skip": design.frame_skip,
        "seeds": list(range(episodes)),
        "holdout_seeds": list(range(episodes, 2 * episodes)),
        "success_definition": "Reached horizon without termination, including the last step",
        "settled_definition": {
            "absolute_state_error_tolerances": SETTLED_TOLERANCES.tolist(),
            "minimum_final_tail_s": MIN_SETTLED_SECONDS,
        },
        "design": {
            "state_order": ["x_m", "joint_theta_rad", "velocity_m_s", "angular_velocity_rad_s"],
            "reference": design.controller.reference.tolist(),
            "formula": "u = clip(-K @ (state - reference), -3, 3)",
            "K": design.controller.gain.tolist(),
            "A": design.a.tolist(),
            "B": design.b.tolist(),
            "Q": design.q.tolist(),
            "R": design.r.tolist(),
            "actuator_gear": design.actuator_gear,
            "force_note": "For this model only: cart actuator force in N = 100 * u",
            "closed_loop_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(closed_loop)))),
        },
        "policies": {name: evaluate(traces, design, horizon) for name, traces in groups.items()},
        "lqr_holdout": evaluate(heldout, design, horizon),
        "displaced_initial_state": initial.tolist(),
        "displaced": {
            name: episode_metrics(t, design.controller.reference, horizon)
            for name, t in displaced.items()
        },
        "limitations": [
            "Local upright linearization; not swing-up or global stability",
            "Ideal simulator full-state feedback; no sensor noise or delays",
            "Clipping enforces the command bound but does not preserve unconstrained LQR optimality",
            "Per-step RMS metrics cover different durations when policies fail early",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    arrays = {}
    archive_groups = {
        **groups,
        "lqr_holdout": heldout,
        **{f"displaced_{name}": [t] for name, t in displaced.items()},
    }
    for name, traces in archive_groups.items():
        for trace in traces:
            key = f"{name}_seed{trace.seed}"
            arrays[f"{key}_states"] = np.vstack([trace.initial_observation, *trace.observations])
            arrays[f"{key}_controls"] = np.asarray(trace.actions)
            arrays[f"{key}_rewards"] = np.asarray(trace.rewards)
            arrays[f"{key}_end_flags"] = np.array([trace.terminated, trace.truncated])
    np.savez_compressed(output / "trajectories.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _save_plot(groups, displaced, design, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New result directory; never overwrites"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--r", type=float, default=0.1, help="Control cost weight, positive")
    args = parser.parse_args()
    if args.episodes < 1 or args.horizon < 50 or not np.isfinite(args.r) or args.r <= 0:
        parser.error("need episodes >= 1, horizon >= 50, and finite r > 0")
    try:
        summary = run_comparison(args.output, args.episodes, args.horizon, args.r)
    except FileExistsError as exc:
        parser.error(str(exc))
    for name, metrics in {**summary["policies"], "lqr_holdout": summary["lqr_holdout"]}.items():
        print(
            f"{name}: survived={metrics['successful_episodes']}/{metrics['episodes']}, "
            f"mean_steps={metrics['mean_episode_length']}, settled={metrics['settled_episodes']}"
        )
    print(f"Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
