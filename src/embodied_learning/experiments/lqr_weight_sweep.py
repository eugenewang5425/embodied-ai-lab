"""One-factor LQR experiment: hold Q/model/initial states fixed and vary R."""

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

from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.environments import make_inverted_pendulum_environment
from embodied_learning.experiments.lqr_comparison import (
    MIN_SETTLED_SECONDS,
    SETTLED_TOLERANCES,
    episode_metrics,
    evaluate,
)
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode
from embodied_learning.plotting import configure_plot_font

plt.switch_backend("Agg")


def input_metrics(trace: EpisodeTrace) -> dict[str, float]:
    commands = np.asarray(trace.actions)
    peak = float(np.max(np.abs(commands)))
    return {
        "peak_absolute_control": peak,
        "peak_absolute_actuator_force_n": peak * trace.actuator_gear,
        # This is an effort proxy, not mechanical work or electrical consumption.
        "squared_input_integral": float(np.sum(commands**2) * trace.dt),
    }


def save_plot(rows: list[dict], demos: list[EpisodeTrace], output: Path) -> None:
    configure_plot_font()
    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), layout="constrained")
    colors = ("#2563eb", "#ea580c", "#15803d", "#9333ea")
    styles = ("-", "--", "-.", ":")
    for i, (row, trace) in enumerate(zip(rows, demos, strict=True)):
        states = np.vstack([trace.initial_observation, *trace.observations])
        times = np.arange(len(states)) * trace.dt
        label = f"R = {row['R']:g}"
        kwargs = {
            "label": label,
            "color": colors[i % len(colors)],
            "linestyle": styles[i % len(styles)],
            "linewidth": 1.8,
        }
        mask = times <= 6.0
        axes[0].plot(times[mask], states[mask, 0] * 100, **kwargs)
        times_u = np.arange(trace.length) * trace.dt
        mask_u = times_u <= 2.0
        axes[1].step(times_u[mask_u], np.asarray(trace.actions)[mask_u], where="post", **kwargs)
        if trace.terminated and times[-1] <= 6.0:
            axes[0].plot(times[-1], states[-1, 0] * 100, "x", color=kwargs["color"])
    axes[0].axhspan(-2, 2, color="gray", alpha=0.12, label="位置容差 ±2 cm（仅一项）")
    axes[0].axhline(0, color="gray", linewidth=0.7)
    axes[0].set(title="小车怎样回到中心？（展示前 6 秒）", ylabel="小车位置（cm）")
    axes[1].set(
        title="起步时用了多大的输入？（展示前 2 秒）", ylabel="控制输入 u（水平力 = 100u N）"
    )
    for axis in axes:
        axis.set_xlabel("仿真时间（s）")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=10)
    figure.suptitle("只改变 R｜同一模型、Q、初始位置 20 cm 与初始倾角")
    figure.savefig(output / "comparison.png", dpi=150)
    plt.close(figure)


def run_sweep(
    output: Path,
    r_values: tuple[float, ...] = (0.1, 1.0, 10.0),
    episodes: int = 20,
    horizon: int = 1000,
) -> dict:
    values = np.asarray(r_values, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Provide at least two distinct, finite, positive R values")
    if episodes < 1 or horizon < 50:
        raise ValueError("Need episodes >= 1 and horizon >= 50 (2 seconds)")
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")

    designs = [design_lqr(control_weight=float(r)) for r in values]
    base = designs[0]
    initial = base.controller.reference + np.array([0.2, 0.05, 0, 0])
    rows, demos, archive = [], [], {}
    for i, (r, design) in enumerate(zip(values, designs, strict=True)):
        for common, current in (
            (base.a, design.a),
            (base.b, design.b),
            (base.q, design.q),
            (base.controller.reference, design.controller.reference),
        ):
            np.testing.assert_array_equal(common, current)
        traces = [run_episode("lqr", seed, horizon, design.controller) for seed in range(episodes)]
        demo = run_episode("lqr", 0, horizon, design.controller, initial_state=initial)
        demos.append(demo)
        rows.append(
            {
                "R": float(r),
                "K": design.controller.gain.tolist(),
                "default_initial_states": evaluate(traces, design, horizon),
                "displaced": {
                    **episode_metrics(demo, design.controller.reference, horizon),
                    **input_metrics(demo),
                },
            }
        )
        for name, group in (("seed", traces), ("displaced", [demo])):
            for trace in group:
                key = f"case{i}_{name}{trace.seed}"
                archive[f"{key}_states"] = np.vstack(
                    [trace.initial_observation, *trace.observations]
                )
                archive[f"{key}_controls"] = np.asarray(trace.actions)
                archive[f"{key}_end_flags"] = np.array([trace.terminated, trace.truncated])
    env = make_inverted_pendulum_environment()
    try:
        model_hash = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
    finally:
        env.close()
    summary = {
        "schema_version": 1,
        "varied_parameter": "R only; K recomputed for each R",
        "Q": base.q.tolist(),
        "A": base.a.tolist(),
        "B": base.b.tolist(),
        "reference": base.controller.reference.tolist(),
        "dt_s": base.dt,
        "actuator_gear": base.actuator_gear,
        "control_limit": base.controller.control_limit,
        "horizon_steps": horizon,
        "seeds": list(range(episodes)),
        "displaced_initial_state": initial.tolist(),
        "model_xml_sha256": model_hash,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "settled_definition": {
            "state_error_tolerances": SETTLED_TOLERANCES.tolist(),
            "minimum_final_tail_s": MIN_SETTLED_SECONDS,
        },
        "conditions": rows,
        "limitations": [
            "Local ideal-simulator experiment; not a universal monotonicity claim",
            "squared_input_integral = sum(u**2)*dt; not joules or battery usage",
            "Different R means different objectives; do not rank controllers by their different weighted total costs",
            "Compare effort over equal completed horizons; failures must not be interpreted as efficiency",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_plot(rows, demos, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r-values", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    args = parser.parse_args()
    try:
        report = run_sweep(args.output, tuple(args.r_values), args.episodes, args.horizon)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for row in report["conditions"]:
        demo = row["displaced"]
        print(
            f"R={row['R']:g}: peak_u={demo['peak_absolute_control']:.6f}, "
            f"settling_s={demo['settling_time_s']}, "
            f"survived={row['default_initial_states']['successful_episodes']}/{args.episodes}"
        )


if __name__ == "__main__":
    main()
