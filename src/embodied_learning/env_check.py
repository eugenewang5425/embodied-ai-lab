"""Run a deterministic MuJoCo smoke experiment and save visible evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from embodied_learning.environments import (
    ENVIRONMENT_ID,
    make_inverted_pendulum_environment,
)
from embodied_learning.plotting import configure_plot_font

plt.switch_backend("Agg")


def run_environment_check(output_dir: Path, steps: int, seed: int) -> dict[str, object]:
    """Step a MuJoCo inverted pendulum with seeded random actions."""
    if steps < 3:
        raise ValueError("steps must be at least 3 so the visual summary has three samples")

    configure_plot_font()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_steps = {0, steps // 2, steps - 1}
    frames: list[np.ndarray] = []
    rewards: list[float] = []
    pole_angles: list[float] = []
    episode_lengths: list[int] = []
    episode_rewards: list[float] = []
    current_length = 0
    current_reward = 0.0

    env = make_inverted_pendulum_environment(
        render_mode="rgb_array",
        width=640,
        height=480,
    )
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    observation_shape = list(np.asarray(observation).shape)
    action_shape = list(env.action_space.shape)

    try:
        for step_index in range(steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, _ = env.step(action)

            reward_value = float(reward)
            rewards.append(reward_value)
            pole_angles.append(float(np.asarray(observation)[1]))
            current_length += 1
            current_reward += reward_value

            if step_index in capture_steps:
                frames.append(np.asarray(env.render()).copy())

            if terminated or truncated:
                episode_lengths.append(current_length)
                episode_rewards.append(current_reward)
                current_length = 0
                current_reward = 0.0
                observation, _ = env.reset(seed=seed + len(episode_lengths))

        if current_length:
            episode_lengths.append(current_length)
            episode_rewards.append(current_reward)
    finally:
        env.close()

    _save_frame_montage(frames, output_dir / "simulation_frames.png")
    _save_metric_plot(rewards, pole_angles, output_dir / "metrics.png")

    summary: dict[str, object] = {
        "environment": ENVIRONMENT_ID,
        "policy": "seeded_random_actions",
        "seed": seed,
        "steps": steps,
        "episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "episode_rewards": [round(value, 6) for value in episode_rewards],
        "mean_reward_per_step": round(float(np.mean(rewards)), 6),
        "max_absolute_pole_angle_rad": round(float(np.max(np.abs(pole_angles))), 6),
        "observation_shape": observation_shape,
        "action_shape": action_shape,
        "captured_frame_shape": list(frames[0].shape),
        "gymnasium_version": gym.__version__,
        "mujoco_version": mujoco.__version__,
        "unicode_path_workaround": True,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _save_frame_montage(frames: list[np.ndarray], output_path: Path) -> None:
    if len(frames) != 3:
        raise RuntimeError(f"expected 3 captured frames, got {len(frames)}")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    labels = ("开始", "中间", "结束")
    for axis, frame, label in zip(axes, frames, labels, strict=True):
        axis.imshow(frame)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle("MuJoCo 倒立摆：随机动作环境自检")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_metric_plot(rewards: list[float], pole_angles: list[float], output_path: Path) -> None:
    step_axis = np.arange(len(rewards))
    figure, (reward_axis, angle_axis) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    reward_axis.plot(step_axis, rewards, color="#2563eb", linewidth=1.2)
    reward_axis.set_ylabel("每步奖励")
    reward_axis.grid(alpha=0.25)

    angle_axis.plot(step_axis, np.rad2deg(pole_angles), color="#dc2626", linewidth=1.2)
    angle_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    angle_axis.set_xlabel("仿真步")
    angle_axis.set_ylabel("摆杆角度（度）")
    angle_axis.grid(alpha=0.25)

    figure.suptitle("随机动作基线：奖励与摆杆角度")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/env_check"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_environment_check(args.output, args.steps, args.seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
