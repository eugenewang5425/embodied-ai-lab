"""Compare seeded random actions with a PD controller on the same task."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from embodied_learning.controllers import PDController
from embodied_learning.environments import ENVIRONMENT_ID, make_inverted_pendulum_environment
from embodied_learning.plotting import configure_plot_font

plt.switch_backend("Agg")

PolicyName = Literal["random", "pd", "lqr"]


class Controller(Protocol):
    def action(self, observation: NDArray[np.floating]) -> NDArray[np.float32]: ...


@dataclass
class EpisodeTrace:
    seed: int
    observations: list[NDArray[np.float64]]
    actions: list[float]
    rewards: list[float]
    terminated: bool
    frames: list[NDArray[np.uint8]]
    truncated: bool = False
    dt: float = 0.04
    actuator_gear: float = 100.0
    initial_observation: NDArray[np.float64] | None = None
    external_forces_n: list[float] | None = None

    @property
    def length(self) -> int:
        return len(self.rewards)


def run_episode(
    policy: PolicyName,
    seed: int,
    horizon: int,
    controller: Controller | None = None,
    capture_frames: bool = False,
    initial_state: NDArray[np.float64] | None = None,
    external_forces_n: NDArray[np.float64] | None = None,
) -> EpisodeTrace:
    """Run one episode with a reproducible initial state and action source."""
    if policy not in ("random", "pd", "lqr"):
        raise ValueError("unknown policy")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if policy != "random" and controller is None:
        raise ValueError("a controller is required for feedback policies")
    if external_forces_n is not None:
        external_forces_n = np.asarray(external_forces_n, dtype=np.float64)
        if external_forces_n.shape != (horizon,) or not np.isfinite(external_forces_n).all():
            raise ValueError("external_forces_n must contain one finite force per step")
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=np.float64)
        if initial_state.shape != (4,) or not np.isfinite(initial_state).all():
            raise ValueError("initial_state must contain four finite values")

    render_mode = "rgb_array" if capture_frames else None
    render_kwargs = {"width": 640, "height": 480} if capture_frames else {}
    env = make_inverted_pendulum_environment(
        render_mode=render_mode,
        max_episode_steps=horizon,
        **render_kwargs,
    )
    capture_steps = {0, horizon // 2, horizon - 1}
    observations: list[NDArray[np.float64]] = []
    actions: list[float] = []
    rewards: list[float] = []
    frames: list[NDArray[np.uint8]] = []
    episode_terminated = False
    episode_truncated = False

    try:
        env.action_space.seed(seed)
        observation, _ = env.reset(seed=seed)
        if initial_state is not None:
            env.unwrapped.set_state(initial_state[:2], initial_state[2:])
            observation = initial_state.copy()
        initial_observation = np.asarray(observation, dtype=np.float64).copy()
        dt = env.unwrapped.dt
        gear = float(env.unwrapped.model.actuator_gear[0, 0])
        for step_index in range(horizon):
            if policy == "random":
                action = env.action_space.sample()
            else:
                assert controller is not None
                action = controller.action(np.asarray(observation))

            # Extra cart force in N, separate from the motor command and its gear.
            # Assign every step, including zero after a pulse: MuJoCo keeps applied forces.
            env.unwrapped.data.qfrc_applied[0] = (
                0.0 if external_forces_n is None else external_forces_n[step_index]
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            observations.append(np.asarray(observation, dtype=np.float64).copy())
            actions.append(float(action[0]))
            rewards.append(float(reward))

            if capture_frames and step_index in capture_steps:
                frames.append(np.asarray(env.render(), dtype=np.uint8).copy())

            if terminated or truncated:
                episode_terminated = bool(terminated)
                episode_truncated = bool(truncated)
                break
    finally:
        env.close()

    return EpisodeTrace(
        seed=seed,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=episode_terminated,
        frames=frames,
        truncated=episode_truncated,
        dt=dt,
        actuator_gear=gear,
        initial_observation=initial_observation,
        external_forces_n=(
            None if external_forces_n is None else external_forces_n[: len(actions)].tolist()
        ),
    )


def summarize(traces: list[EpisodeTrace], horizon: int) -> dict[str, object]:
    if not traces or horizon < 1:
        raise ValueError("traces must be nonempty and horizon positive")
    lengths = np.asarray([trace.length for trace in traces], dtype=np.float64)
    successes = [trace.length == horizon and not trace.terminated for trace in traces]
    angles = np.concatenate(
        [np.asarray([observation[1] for observation in trace.observations]) for trace in traces]
    )
    positions = np.concatenate(
        [np.asarray([observation[0] for observation in trace.observations]) for trace in traces]
    )
    actions = np.concatenate([np.asarray(trace.actions) for trace in traces])
    forces = np.concatenate([np.asarray(trace.actions) * trace.actuator_gear for trace in traces])

    return {
        "episodes": len(traces),
        "successful_episodes": int(np.sum(successes)),
        "success_rate": round(float(np.mean(successes)), 6),
        "mean_episode_length": round(float(np.mean(lengths)), 6),
        "minimum_episode_length": int(np.min(lengths)),
        "mean_absolute_pole_angle_rad": round(float(np.mean(np.abs(angles))), 8),
        "maximum_absolute_pole_angle_rad": round(float(np.max(np.abs(angles))), 8),
        "maximum_absolute_cart_position_m": round(float(np.max(np.abs(positions))), 8),
        "root_mean_square_control": round(float(np.sqrt(np.mean(np.square(actions)))), 8),
        "root_mean_square_force_n": round(float(np.sqrt(np.mean(np.square(forces)))), 8),
    }


def run_comparison(
    output_dir: Path,
    seeds: range,
    horizon: int,
    proportional_gain: float,
    derivative_gain: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = PDController(proportional_gain, derivative_gain)

    random_traces = [run_episode("random", seed, horizon) for seed in seeds]
    pd_traces = [run_episode("pd", seed, horizon, controller) for seed in seeds]

    representative_seed = seeds.start
    random_example = run_episode("random", representative_seed, horizon)
    pd_example = run_episode(
        "pd",
        representative_seed,
        horizon,
        controller,
        capture_frames=True,
    )

    random_summary = summarize(random_traces, horizon)
    pd_summary = summarize(pd_traces, horizon)
    summary: dict[str, object] = {
        "schema_version": 2,
        "environment": ENVIRONMENT_ID,
        "horizon_steps": horizon,
        "seconds_per_step": pd_example.dt,
        "evaluated_seeds": list(seeds),
        "controller": {
            "formula": "control = Kp * pole_angle + Kd * pole_angular_velocity",
            "proportional_gain": proportional_gain,
            "derivative_gain": derivative_gain,
            "control_limit": 3.0,
            "actuator_gear": pd_example.actuator_gear,
        },
        "random": random_summary,
        "pd": pd_summary,
    }

    configure_plot_font()
    _save_comparison_plot(random_example, pd_example, random_traces, pd_traces, output_dir)
    _save_frame_montage(pd_example.frames, output_dir / "pd_simulation_frames.png")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _save_comparison_plot(
    random_example: EpisodeTrace,
    pd_example: EpisodeTrace,
    random_traces: list[EpisodeTrace],
    pd_traces: list[EpisodeTrace],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    time_random = (np.arange(random_example.length) + 1) * random_example.dt
    time_pd = (np.arange(pd_example.length) + 1) * pd_example.dt
    random_angles = np.rad2deg([value[1] for value in random_example.observations])
    pd_angles = np.rad2deg([value[1] for value in pd_example.observations])

    axes[0, 0].plot(time_random, random_angles, label="随机动作", color="#dc2626")
    axes[0, 0].plot(time_pd, pd_angles, label="PD 控制", color="#2563eb")
    axes[0, 0].axhline(np.rad2deg(0.2), color="black", linestyle="--", alpha=0.5)
    axes[0, 0].axhline(-np.rad2deg(0.2), color="black", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("同一初始状态下的摆杆角度")
    axes[0, 0].set_xlabel("时间（秒）")
    axes[0, 0].set_ylabel("关节角（度）")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(
        time_random - random_example.dt, random_example.actions, label="随机动作", color="#dc2626"
    )
    axes[0, 1].plot(time_pd - pd_example.dt, pd_example.actions, label="PD 控制", color="#2563eb")
    axes[0, 1].set_title("控制输入（实际水平力 = 100 × u N）")
    axes[0, 1].set_xlabel("时间（秒）")
    axes[0, 1].set_ylabel("控制输入 u")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    seed_axis = [trace.seed for trace in random_traces]
    axes[1, 0].plot(
        seed_axis,
        [trace.length for trace in random_traces],
        marker="o",
        label="随机动作",
        color="#dc2626",
    )
    axes[1, 0].plot(
        seed_axis,
        [trace.length for trace in pd_traces],
        marker="o",
        label="PD 控制",
        color="#2563eb",
    )
    axes[1, 0].set_title(f"{len(random_traces)} 个初始状态下的回合长度")
    axes[1, 0].set_xlabel("随机种子")
    axes[1, 0].set_ylabel("完成步数")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].bar(
        ["随机动作", "PD 控制"],
        [
            np.mean([trace.length for trace in random_traces]),
            np.mean([trace.length for trace in pd_traces]),
        ],
        color=["#dc2626", "#2563eb"],
    )
    axes[1, 1].set_title("平均回合长度")
    axes[1, 1].set_ylabel("完成步数")
    axes[1, 1].grid(axis="y", alpha=0.25)

    figure.suptitle("随机动作与 PD 反馈控制对比")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(figure)


def _save_frame_montage(frames: list[NDArray[np.uint8]], output_path: Path) -> None:
    if len(frames) != 3:
        raise RuntimeError(f"expected 3 PD frames, got {len(frames)}")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for axis, frame, label in zip(axes, frames, ("开始", "中间", "结束"), strict=True):
        axis.imshow(frame)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle("PD 控制下的倒立摆")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/pd_comparison"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_comparison(
        output_dir=args.output,
        seeds=range(args.episodes),
        horizon=args.horizon,
        proportional_gain=args.kp,
        derivative_gain=args.kd,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
