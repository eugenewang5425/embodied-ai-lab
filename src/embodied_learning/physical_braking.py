"""Observation-only stopping forecasts matched to the planar force servo.

The predictor contains the robot and actuator model, with no world obstacles.
It accepts measured body-frame vx/vy/yaw-rate and already issued intentions.
Historical kinematic controllers and archived experiments are unchanged.
"""

from collections import deque

import mujoco
import numpy as np

from embodied_learning.braking_control import DT, brake_step, reachable
from embodied_learning.contact_motion import (
    FORCE_LIMIT_N,
    PHYSICS_DT,
    TORQUE_LIMIT_NM,
    build_xml,
)
from embodied_learning.execution_safety import FRESHNESS_TICKS
from embodied_learning.navigation_geometry import rotation
from embodied_learning.navigation_robustness import IntentQueue
from embodied_learning.navigation_sensor_ablation import SensorMapController


def measured_body_velocity(world_velocity, estimated_pose):
    velocity = np.asarray(world_velocity, float).copy()
    velocity[:2] = velocity[:2] @ rotation(estimated_pose[2])
    return velocity


def swept_point_clearance(points, poses, rig, margin=0.0):
    """Same signed boxes as point_clearance; batch poses to reduce Python cost."""
    points, poses = np.asarray(points), np.asarray(poses)
    if not len(points):
        return float("inf")
    best = float("inf")
    for chunk in np.array_split(poses, max(1, int(np.ceil(len(poses) / 32)))):
        dx = points[None, :, 0] - chunk[:, None, 0]
        dy = points[None, :, 1] - chunk[:, None, 1]
        c, s = np.cos(chunk[:, 2, None]), np.sin(chunk[:, 2, None])
        x, y = dx * c + dy * s, -dx * s + dy * c
        for part in rig.parts:
            lo, hi = part[[0, 2, 4]], part[[1, 3, 5]]
            center, half = (lo + hi) / 2, (hi - lo) / 2
            qx = np.abs(x - center[0]) - half[0] - margin
            qy = np.abs(y - center[1]) - half[1] - margin
            qz = np.abs(points[None, :, 2] - center[2]) - half[2]
            outside = np.sqrt(
                np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2
            )
            inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0)
            best = min(best, float((outside + inside).min()))
    return best


class PhysicalStopPredictor:
    def __init__(self, rig):
        self.rig = rig
        self.model = mujoco.MjModel.from_xml_string(build_xml(np.empty((0, 6)), rig))
        self.data = mujoco.MjData(self.model)

    def predict(
        self,
        velocity,
        pending=(),
        desired=None,
        stopping=True,
        first_command=None,
        record_states=True,
    ):
        velocity = np.asarray(velocity, float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("full finite body vx/vy/yaw-rate is required")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = velocity
        mujoco.mj_forward(self.model, self.data)
        sequence = [(np.asarray(u), bool(brake)) for u, brake, _ in pending]
        if desired is not None:
            sequence.append((np.asarray(desired), bool(stopping)))
        poses, states, commands = [np.zeros(3)], [np.r_[self.data.qpos, self.data.qvel]], []
        distance = 0.0
        last = self.data.qpos.copy()
        for tick in range(50):
            yaw = self.data.qpos[2]
            incoming = np.array(
                [
                    self.data.qvel[0] * np.cos(yaw) + self.data.qvel[1] * np.sin(yaw),
                    self.data.qvel[2],
                ]
            )
            target, brake = sequence[tick] if tick < len(sequence) else (np.zeros(2), True)
            command = (
                np.asarray(first_command, float).copy()
                if tick == 0 and first_command is not None
                else brake_step(incoming, self.rig, True)
                if brake
                else reachable(incoming, target, self.rig)
            )
            commands.append(command.copy())
            for _ in range(round(DT / PHYSICS_DT)):
                yaw = self.data.qpos[2]
                desired_xy = command[0] * np.array([np.cos(yaw), np.sin(yaw)])
                force = 80 * (desired_xy - self.data.qvel[:2])
                force *= min(1.0, FORCE_LIMIT_N / max(np.linalg.norm(force), 1e-12))
                torque = np.clip(
                    8 * (command[1] - self.data.qvel[2]), -TORQUE_LIMIT_NM, TORQUE_LIMIT_NM
                )
                self.data.qfrc_applied[:] = [*force, torque]
                mujoco.mj_step(self.model, self.data)
                if record_states:
                    states.append(np.r_[self.data.qpos, self.data.qvel])
                delta = self.data.qpos - last
                distance += np.linalg.norm(delta[:2]) + self.rig.radius * abs(delta[2])
                last = self.data.qpos.copy()
                if distance >= 0.001:
                    poses.append(last.copy())
                    distance = 0.0
            # Include the slow motor tail and lateral velocity, beyond the task's
            # 0.01 stop threshold. A target of zero never teleports qvel to zero.
            if tick >= len(sequence) and np.max(np.abs(self.data.qvel)) < 1e-4:
                break
        else:
            raise RuntimeError("physical stop forecast exceeded five seconds")
        poses.append(self.data.qpos.copy())
        return np.asarray(poses), np.asarray(commands), np.asarray(states)


class PhysicalBrakingController(SensorMapController):
    def __init__(self, prior_boxes, rig):
        super().__init__(prior_boxes, rig, "lidar_depth")
        self.predictor = PhysicalStopPredictor(rig)
        self.body_velocity = np.zeros(3)

    def step(self, estimate, goal, ranges, hits, depth, valid, frame, velocity):
        self.latest = {
            "speed_cap": np.nan,
            "preview_curvature": np.nan,
            "nominal": np.zeros(2),
            "stop_clearance": np.nan,
        }
        points = self.observe(estimate, ranges, hits, depth, valid)
        if self.path is None or frame - self.last_plan >= 10:
            self.path = self.make_plan(estimate, goal)
            self.wp, self.last_plan = 0, frame
        if self.path is None:
            return np.zeros(2), "no_observed_route", float("inf")
        desired = self.tracking_command(estimate, velocity)
        candidate = reachable(velocity, desired, self.rig)
        path, _, _ = self.predictor.predict(
            self.body_velocity, first_command=candidate, record_states=False
        )
        stop, _, _ = self.predictor.predict(self.body_velocity, record_states=False)
        gap = swept_point_clearance(points, path, self.rig, self.rig.margin_m)
        self.latest["stop_clearance"] = swept_point_clearance(
            points, stop, self.rig, self.rig.margin_m
        )
        if gap <= 0:
            return np.zeros(2), "swept_body_brake", gap
        return desired, "following", gap


class PhysicalGuardedTransport(IntentQueue):
    def __init__(self, delay_steps, rig, predictor):
        super().__init__(delay_steps, rig)
        self.predictor = predictor
        self.last_fresh, self.latched = 0, False
        self.body_velocity = np.zeros(3)

    def execute(self, desired, stopping, frame, incoming, points, fresh=True):
        if fresh:
            self.last_fresh = frame
            u, target, brake, source = self.step(desired, stopping, frame, incoming)
        elif self.queue:
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
        path, _, _ = self.predictor.predict(
            self.body_velocity, first_command=u, record_states=False
        )
        gap = swept_point_clearance(points, path, self.rig, self.rig.margin_m)
        expiry = age >= FRESHNESS_TICKS
        self.latched |= expiry
        override = gap <= 0 or self.latched
        if override:
            u = brake_step(incoming, self.rig, True)
            self.queue = deque((np.zeros(2), True, -1) for _ in self.queue)
            brake = True
        return u, target, brake, source, override, expiry, gap, age
