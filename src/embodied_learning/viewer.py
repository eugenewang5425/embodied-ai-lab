"""Open a short interactive MuJoCo window for visual verification."""

from __future__ import annotations

import argparse
import time

from embodied_learning.controllers import PDController
from embodied_learning.controllers.lqr import design_lqr
from embodied_learning.environments import make_inverted_pendulum_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("random", "pd", "lqr"), default="random")
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.1)
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("seconds must be positive")
    controller = (
        design_lqr(control_weight=args.r).controller
        if args.policy == "lqr"
        else PDController(args.kp, args.kd)
    )
    env = make_inverted_pendulum_environment(render_mode="human")
    env.action_space.seed(args.seed)
    observation, _ = env.reset(seed=args.seed)
    deadline = time.perf_counter() + args.seconds

    try:
        while time.perf_counter() < deadline:
            if args.policy != "random":
                action = controller.action(observation)
            else:
                action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
