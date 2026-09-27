"""Causal arrival decisions and scan-constrained bounded recovery (lessons 70/71)."""

import numpy as np

from embodied_learning.experiments.campus_patrol import RADIUS, RES, advance, plan


def particle_radius(particles, weights, estimate, quantile=0.95):
    """Internal weighted concentration; explicitly NOT a guaranteed error bound."""
    distance = np.linalg.norm(particles[:, :2] - estimate[:2], axis=1)
    order = np.argsort(distance)
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    return float(distance[order[min(len(order) - 1, np.searchsorted(cumulative, quantile))]])


def scan_supports_motion(ranges, local_poses, radius=RADIUS + 0.02):
    """Require the whole sampled disc to stay inside current measured free rays.

    Neighboring rays use their smaller range and a 3 cm reserve. This is a
    conservative finite-ray diagnostic, not proof about unseen small obstacles.
    """
    angles = np.arange(32) * (2 * np.pi / 32)
    rim = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    points = np.asarray(local_poses)[:, None, :2] + rim
    distance = np.linalg.norm(points, axis=2)
    bins = (np.arctan2(points[..., 1], points[..., 0]) % (2 * np.pi)) * len(ranges) / (2 * np.pi)
    left = np.floor(bins).astype(int) % len(ranges)
    limit = np.minimum(ranges[left], ranges[(left + 1) % len(ranges)]) - 0.03
    return bool(np.all(distance < limit))


class NavigationDecision:
    def __init__(self, mode, occupied, system_margin=0.10, recovery_frames=40):
        if mode not in ("baseline", "uncertainty", "joint", "recovery"):
            raise ValueError(mode)
        self.mode, self.occupied = mode, occupied
        self.system_margin, self.recovery_frames = system_margin, recovery_frames
        self.telemetry = []
        self.recovery_events = []
        self.recover_start = None
        self.info = {}

    def arrival(self, estimate, pf, goal, previous_command, config):
        dist = float(np.linalg.norm(estimate[:2] - goal))
        r95 = particle_radius(pf.p, pf.w, estimate) if pf is not None else 0.0
        budget = r95 + self.system_margin
        limit = config.estimate_arrival_m
        ready = dist < limit
        if self.mode in ("uncertainty", "joint"):
            limit = max(0.04, min(limit, config.true_arrival_m - budget))
            ready = dist < limit and dist + budget <= config.true_arrival_m
        stopped = abs(previous_command[0]) <= 0.02 and abs(previous_command[1]) <= 0.05
        if self.mode == "joint":
            ready = ready and stopped
        self.info = {
            "arrival_distance": dist,
            "arrival_r95": r95,
            "arrival_budget": budget,
            "arrival_limit": limit,
            "arrival_ready": ready,
            "observed_stopped": stopped,
            "reason": "following",
            "recovery_active": False,
        }
        return ready, limit

    def recover(self, estimate, goal, ranges, hits, safe, frame):
        if self.mode != "recovery":
            return None
        if self.info.get("arrival_r95", 0.0) > 0.25:
            self.info["reason"] = "localization_ambiguous"
            return None
        cell = np.floor(estimate[:2][::-1] / RES).astype(int)
        inside = np.all(cell >= 0) and np.all(cell < safe.shape)
        if not inside or safe[tuple(cell)]:
            self.info["reason"] = "outside_map" if not inside else "global_route_blocked"
            return None
        if self.recover_start is None:
            if len(self.recovery_events) >= 3:
                self.info["reason"] = "recovery_attempts_exhausted"
                return None
            self.recover_start = frame
            self.recovery_events.append({"start_frame": frame, "outcome": "trying"})
        if frame - self.recover_start >= self.recovery_frames:
            self.info["reason"] = "recovery_budget_exhausted"
            self.recovery_events[-1].update(end_frame=frame, outcome=self.info["reason"])
            return None
        candidates = []
        for velocity in (-0.15, 0.15):
            for omega in (0.0, -0.20, 0.20):
                local = np.array(
                    [advance(np.zeros(3), velocity * t, omega * t) for t in np.linspace(0, 3.0, 46)]
                )
                if not scan_supports_motion(ranges, local):
                    continue
                c, s = np.cos(estimate[2]), np.sin(estimate[2])
                xy = local[:, :2] @ np.array([[c, s], [-s, c]]) + estimate[:2]
                cells = np.floor(xy[:, ::-1] / RES).astype(int)
                if np.any(cells < 0) or np.any(cells >= np.array(safe.shape)):
                    continue
                # No connector may cross a raw occupied cell, even if its far
                # endpoint is free. No ground-truth state enters this decision.
                if self.occupied[cells[:, 0], cells[:, 1]].any() or not safe[tuple(cells[-1])]:
                    continue
                path = plan(safe, xy[-1], goal)
                if path is None:
                    continue
                cost = np.linalg.norm(path[0] - goal) + 0.2 * abs(omega)
                candidates.append((cost, velocity, omega))
        if not candidates:
            self.info["reason"] = "no_observed_safe_escape"
            self.recovery_events[-1].update(end_frame=frame, outcome=self.info["reason"])
            return None
        _, v, w = min(candidates)
        self.info.update(reason="observed_low_speed_escape", recovery_active=True)
        return v, w

    def path_available(self, frame):
        if self.recover_start is not None:
            self.recovery_events[-1].update(end_frame=frame, outcome="replanned_after_real_motion")
            self.recover_start = None

    def record(self):
        self.telemetry.append(self.info.copy())
