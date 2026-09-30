"""Queue-aware forecasts and an observation-only actuator safety prototype."""

import math
from collections import deque

import numpy as np

from embodied_learning.braking_control import DT, brake_step, braking_path, reachable
from embodied_learning.navigation_geometry import advance, point_clearance
from embodied_learning.navigation_robustness import IntentQueue

METHODS = {
    "baseline_exec": (False, False),
    "queue_forecast": (True, False),
    "local_guard": (False, True),
    "queue_guard": (True, True),
}
FRESHNESS_TICKS = 2


def queued_stop_path(velocity, pending, desired, stopping, rig):
    """All outstanding intentions, this request, then coupled braking to rest.

    Predict using current acceleration limits, not cached future speeds. Pending
    is a read-only snapshot; no truth geometry, pose or future sensor is needed.
    """
    pose, poses, commands = np.zeros(3), [np.zeros(3)], []
    current = np.asarray(velocity, float).copy()
    sequence = [(target, brake) for target, brake, _ in pending]
    sequence.append((np.asarray(desired), stopping))
    for target, brake in sequence:
        current = brake_step(current, rig, True) if brake else reachable(current, target, rig)
        steps = max(1, math.ceil((abs(current[0]) + rig.radius * abs(current[1])) * DT / 0.01))
        poses.extend(advance(pose, *current, t) for t in np.linspace(0, DT, steps + 1)[1:])
        pose = advance(pose, *current, DT)
        commands.append(current.copy())
    tail, braking = braking_path(current, rig, True)
    # Tail is in the endpoint body frame, including its starting pose.
    c, s = np.cos(pose[2]), np.sin(pose[2])
    tail[:, :2] = tail[:, :2] @ np.array([[c, s], [-s, c]]) + pose[:2]
    tail[:, 2] += pose[2]
    poses.extend(tail[1:])
    commands.extend(braking)
    return np.asarray(poses), np.asarray(commands).reshape(-1, 2)


def command_stop_path(candidate, rig):
    # Candidate has already been acceleration-limited using actual incoming speed.
    first = advance(np.zeros(3), *candidate, DT)
    n = max(1, math.ceil((abs(candidate[0]) + rig.radius * abs(candidate[1])) * DT / 0.01))
    head = np.array([advance(np.zeros(3), *candidate, t) for t in np.linspace(0, DT, n + 1)])
    tail, _ = braking_path(candidate, rig, True)
    c, s = np.cos(first[2]), np.sin(first[2])
    tail[:, :2] = tail[:, :2] @ np.array([[c, s], [-s, c]]) + first[:2]
    tail[:, 2] += first[2]
    return np.vstack((head, tail[1:]))


class GuardedTransport(IntentQueue):
    """Local guard acts AFTER transport; expiry latches a physical stop.

    This API must run independently of planning in a deployed system. Calling
    both in one blocked Python thread does not provide a real-time watchdog.
    """

    def __init__(self, delay_steps, rig, enabled):
        super().__init__(delay_steps, rig)
        self.enabled = enabled
        self.last_fresh = 0
        self.latched = False

    def execute(self, desired, stopping, frame, incoming, points, fresh=True):
        if fresh:
            self.last_fresh = frame
            u, target, brake, source = self.step(desired, stopping, frame, incoming)
        elif self.queue:
            # Missing request: consume previously issued intentions causally.
            self.queue.append((np.asarray(incoming).copy(), False, -2))
            target, brake, source = self.queue.popleft()
            u = (
                brake_step(incoming, self.rig, True)
                if brake
                else reachable(incoming, target, self.rig)
            )
        else:
            u, target, brake, source = (
                np.asarray(incoming).copy(),
                np.asarray(incoming).copy(),
                False,
                -2,
            )
        age = frame - self.last_fresh
        gap = point_clearance(points, command_stop_path(u, self.rig), self.rig, self.rig.margin_m)
        expiry = self.enabled and age >= FRESHNESS_TICKS
        self.latched |= expiry
        obstacle = self.enabled and gap <= 0
        override = self.enabled and (obstacle or self.latched)
        if override:
            u = brake_step(incoming, self.rig, True)
            # Purge stale motion, retaining transport length and stationary slots.
            self.queue = deque((np.zeros(2), True, -1) for _ in self.queue)
            brake = True
        return u, target, brake, source, override, expiry, gap, age
