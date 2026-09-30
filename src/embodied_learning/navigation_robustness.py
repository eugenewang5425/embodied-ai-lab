"""Controlled pose errors and a causal, acceleration-limited command transport."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from embodied_learning.braking_control import DT, brake_step, reachable


@dataclass(frozen=True)
class Perturbation:
    key: str = "zero"
    dy_m: float = 0.0
    yaw_deg: float = 0.0
    delay_steps: int = 0

    def to_dict(self):
        return asdict(self)


CONDITIONS = {"zero": Perturbation()}
for factor in ("y", "yaw"):
    for sign, direction in ((-1, "minus"), (1, "plus")):
        for value, tag in ((0.5, "05"), (1, "1"), (2, "2"), (4, "4")):
            key = f"{factor}_{direction}_{tag}"
            CONDITIONS[key] = Perturbation(
                key,
                dy_m=sign * value / 100 if factor == "y" else 0,
                yaw_deg=sign * value if factor == "yaw" else 0,
            )
for steps in range(1, 5):
    key = f"delay_{steps}"
    CONDITIONS[key] = Perturbation(key, delay_steps=steps)

GROUPS = {
    **{
        f"{factor}_{direction}": ["zero"]
        + [f"{factor}_{direction}_{tag}" for tag in ("05", "1", "2", "4")]
        for factor in ("y", "yaw")
        for direction in ("plus", "minus")
    },
    "delay": ["zero"] + [f"delay_{k}" for k in range(1, 5)],
}


def biased_pose(truth, condition):
    estimate = np.array(truth, float, copy=True)
    estimate[1] += condition.dy_m
    if condition.yaw_deg:
        angle = estimate[2] + np.deg2rad(condition.yaw_deg)
        estimate[2] = np.arctan2(np.sin(angle), np.cos(angle))
    return estimate


class IntentQueue:
    """Delay intentions, then apply physical limits using current executed speed.

    A command issued at k becomes eligible at k+delay_steps. Safety requests use
    the same queue. Initial slots mean stationary hold, not future commands.
    """

    def __init__(self, delay_steps, rig):
        if isinstance(delay_steps, bool) or int(delay_steps) != delay_steps or delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
        self.rig = rig
        self.queue = deque((np.zeros(2), True, -1) for _ in range(int(delay_steps)))

    def step(self, desired, stopping, frame, incoming):
        self.queue.append((np.array(desired, float, copy=True), bool(stopping), int(frame)))
        target, brake, source = self.queue.popleft()
        executed = (
            brake_step(incoming, self.rig, True) if brake else reachable(incoming, target, self.rig)
        )
        return executed, target, brake, source

    @property
    def pending_motion(self):
        return sum(not brake and np.max(np.abs(target)) > 1e-12 for target, brake, _ in self.queue)

    @property
    def delay_s(self):
        return len(self.queue) * DT
