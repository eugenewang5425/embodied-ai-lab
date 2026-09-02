"""Paired random cart-push experiment: fixed plant and LQR gains, no new algorithm."""

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
)
from embodied_learning.experiments.pd_comparison import EpisodeTrace, run_episode
from embodied_learning.plotting import configure_plot_font

R_VALUES = (0.1, 1.0, 10.0)


def save_push_plot(output: Path, archive, report: dict) -> None:
    """Show the first predeclared seed, not a selected best-looking episode."""
    configure_plot_font()
    seed, dt = report["seeds"][0], report["dt_s"]
    forces = archive[f"seed{seed}_scheduled_force_n"]
    affected = np.flatnonzero(forces)
    start, end = affected[0] * dt, (affected[-1] + 1) * dt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    for r, color in zip(R_VALUES, ("#2563eb", "#ea580c", "#15803d"), strict=True):
        states = archive[f"r{r:g}_seed{seed}_states"]
        controls = archive[f"r{r:g}_seed{seed}_controls"]
        axes[0].plot(
            np.arange(len(states)) * dt,
            np.rad2deg(states[:, 1] - report["reference"][1]),
            label=f"R={r:g}",
            color=color,
        )
        axes[1].step(
            np.arange(len(controls)) * dt, controls, where="post", label=f"R={r:g}", color=color
        )
        if archive[f"r{r:g}_seed{seed}_end_flags"][0]:
            axes[0].plot(
                (len(states) - 1) * dt,
                np.rad2deg(states[-1, 1] - report["reference"][1]),
                "x",
                color=color,
            )
    for axis in axes:
        axis.axvspan(start, end, color="#f59e0b", alpha=0.2, label="外部推力持续区间")
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
        axis.set_xlabel("仿真时间（s）")
    axes[0].set_ylabel("真实倾角（°，不放大）")
    axes[1].set_ylabel("控制输入 u（执行器水平力=100u N）")
    figure.suptitle(
        f"同一外部推力下的恢复｜seed={seed}，{forces[affected[0]]:+.1f} N，持续 {end - start:.2f} s"
    )
    figure.savefig(output, dpi=150)
    plt.close(figure)


def random_push(seed: int, horizon: int, dt: float) -> np.ndarray:
    """One 0.2 s pulse, signed magnitude U[30,80] N, onset on [2,3] s grid."""
    if seed < 0 or not np.isfinite(dt) or dt <= 0 or horizon * dt < 6:
        raise ValueError("Need seed >= 0, dt > 0, and at least six simulation seconds")
    rng = np.random.default_rng(seed)
    start = int(rng.integers(int(np.ceil(2 / dt)), int(np.floor(3 / dt)) + 1))
    count = max(1, round(0.2 / dt))
    force = float(rng.choice([-1, 1]) * rng.uniform(30, 80))
    forces = np.zeros(horizon)
    forces[start : start + count] = force
    return forces


def recovery_metrics(
    trace: EpisodeTrace, reference: np.ndarray, horizon: int, forces: np.ndarray
) -> dict:
    row = episode_metrics(trace, reference, horizon)
    affected = np.flatnonzero(forces)
    if not len(affected):
        raise ValueError("A nonzero push is required to measure recovery")
    start, end = int(affected[0]), int(affected[-1] + 1)
    states = np.vstack([trace.initial_observation, *trace.observations])
    within = np.all(np.abs(states - reference) <= SETTLED_TOLERANCES, axis=1)
    remains = np.logical_and.accumulate(within[::-1])[::-1]
    candidates = np.flatnonzero(remains)
    # Tail duration is measured between timestamps, including the state at re-entry.
    candidates = candidates[
        (candidates >= end)
        & (candidates <= trace.length - int(np.ceil(MIN_SETTLED_SECONDS / trace.dt)))
    ]
    recovered = row["survived_horizon"] and bool(len(candidates))
    return {
        **row,
        "push_start_s": start * trace.dt,
        "push_end_s": end * trace.dt,
        "push_force_n": float(forces[start]),
        "recovered_after_push": recovered,
        "recovery_after_push_end_s": float((candidates[0] - end) * trace.dt) if recovered else None,
        "peak_absolute_control": float(np.max(np.abs(trace.actions))),
    }


def run_disturbance(
    output: Path, episodes: int = 20, first_seed: int = 100, horizon: int = 250
) -> dict:
    if episodes < 1 or first_seed < 0 or horizon < 150:
        raise ValueError("Need episodes >= 1, first_seed >= 0 and horizon >= 150")
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    designs = [design_lqr(control_weight=r) for r in R_VALUES]
    base = designs[0]
    seeds = list(range(first_seed, first_seed + episodes))
    pushes = {seed: random_push(seed, horizon, base.dt) for seed in seeds}
    archive, rows = {}, []
    for r, design in zip(R_VALUES, designs, strict=True):
        for common, current in ((base.a, design.a), (base.b, design.b), (base.q, design.q)):
            np.testing.assert_array_equal(common, current)
        baseline = run_episode(
            "lqr", first_seed, horizon, design.controller, initial_state=base.controller.reference
        )
        group = []
        for seed in seeds:
            trace = run_episode(
                "lqr",
                seed,
                horizon,
                design.controller,
                initial_state=base.controller.reference,
                external_forces_n=pushes[seed],
            )
            group.append(
                recovery_metrics(trace, design.controller.reference, horizon, pushes[seed])
            )
            key = f"r{r:g}_seed{seed}"
            archive[f"{key}_states"] = np.vstack([trace.initial_observation, *trace.observations])
            archive[f"{key}_controls"] = np.asarray(trace.actions)
            archive[f"{key}_applied_force_n"] = np.asarray(trace.external_forces_n)
            archive[f"{key}_end_flags"] = np.array([trace.terminated, trace.truncated])
        archive[f"r{r:g}_baseline_states"] = np.vstack(
            [baseline.initial_observation, *baseline.observations]
        )
        archive[f"r{r:g}_baseline_controls"] = np.asarray(baseline.actions)
        recovered = [v["recovery_after_push_end_s"] for v in group if v["recovered_after_push"]]
        rows.append(
            {
                "R": r,
                "K": design.controller.gain.tolist(),
                "baseline": episode_metrics(baseline, design.controller.reference, horizon),
                "survived": sum(v["survived_horizon"] for v in group),
                "recovered": len(recovered),
                "median_recovery_s_among_recovered": float(np.median(recovered))
                if recovered
                else None,
                "maximum_angle_error_deg": float(
                    np.rad2deg(max(v["max_abs_upright_error_rad"] for v in group))
                ),
                "episodes": group,
            }
        )
    for seed, forces in pushes.items():
        archive[f"seed{seed}_scheduled_force_n"] = forces
    env = make_inverted_pendulum_environment()
    try:
        model_hash = hashlib.sha256(Path(env.unwrapped.fullpath).read_bytes()).hexdigest()
    finally:
        env.close()
    report = {
        "schema_version": 1,
        "experiment": "paired_random_cart_push",
        "dt_s": base.dt,
        "horizon_steps": horizon,
        "seeds": seeds,
        "initial_state": base.controller.reference.tolist(),
        "reference": base.controller.reference.tolist(),
        "A": base.a.tolist(),
        "B": base.b.tolist(),
        "Q": base.q.tolist(),
        "actuator_gear": base.actuator_gear,
        "control_limit": base.controller.control_limit,
        "model_xml_sha256": model_hash,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "push_protocol": "One 0.20s cart force; onset uniform control-step grid [2,3]s; random sign; magnitude U[30,80]N; common schedule for all R; controller sees state only",
        "settled_tolerances": SETTLED_TOLERANCES.tolist(),
        "minimum_final_tail_s": MIN_SETTLED_SECONDS,
        "conditions": rows,
        "limitations": [
            "Synthetic push distribution, not measured hardware statistics",
            "Ideal state feedback; no sensor noise, delay or model mismatch",
            "Fixed nominal K; no retraining or advance knowledge of push",
            "Recovery timed from push end, failures retained and never ranked as low effort",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_push_plot(output / "comparison.png", archive, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--first-seed", type=int, default=100)
    args = parser.parse_args()
    try:
        report = run_disturbance(args.output, args.episodes, args.first_seed)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    for row in report["conditions"]:
        print(
            f"R={row['R']:g}: survived={row['survived']}/{args.episodes}, recovered={row['recovered']}/{args.episodes}, median_recovery={row['median_recovery_s_among_recovered']}s"
        )


if __name__ == "__main__":
    main()
