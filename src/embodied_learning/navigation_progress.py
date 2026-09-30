"""Sequential route progress; never chooses a distant segment at a crossing."""

import numpy as np

from embodied_learning.navigation_map_update import MapController
from embodied_learning.path_tracking import wrap

PROJECTION_TOLERANCE_M = 0.05


def advance_prefix(path, pose, waypoint):
    """Pass a retained start only when the robot lies on its outgoing segment.

    A replan can retain a start behind the robot. The original radius rule
    cannot pass that point. Check the first segment in route order instead of
    finding the globally nearest segment, which could jump across a loop.
    Subsequent waypoints keep the frozen radius rule and steering controller.
    """
    result = {"waypoint": int(waypoint), "prefix_passed": False, "fraction": np.nan}
    if waypoint != 0 or path is None or len(path) < 2:
        return result
    edge = np.asarray(path[1]) - path[0]
    length_sq = float(edge @ edge)
    if length_sq <= 1e-16:
        return result
    fraction = float((pose[:2] - path[0]) @ edge / length_sq)
    projection = path[0] + np.clip(fraction, 0, 1) * edge
    near = np.linalg.norm(pose[:2] - projection) <= PROJECTION_TOLERANCE_M
    # Only the interior of this first segment proves the start was passed.
    # A pose beyond a turn or near a later crossing does not establish it.
    passed = near and 1e-8 < fraction <= 1
    result.update(waypoint=1 if passed else 0, prefix_passed=bool(passed), fraction=fraction)
    return result


class ProgressMapController(MapController):
    def tracking_command(self, estimate, velocity):
        info = advance_prefix(self.path, estimate, self.wp)
        self.wp = info["waypoint"]
        desired = super().tracking_command(estimate, velocity)
        tracking = self.tracking_log[-1]
        # The radius rule may advance a corner before the body reaches it.
        # A distant waypoint bearing then disagrees with tangent steering.
        # Use the same path/cross-track heading for both steering and speed.
        heading = tracking["heading_rad"] - np.arctan2(2 * tracking["cross_m"], 0.6)
        gate = float(wrap(heading - estimate[2]))
        forward = self.fixed_cap if abs(gate) < 0.3 else 0.0
        raw = forward * tracking["curvature_m_inv"] + 2 * wrap(heading - estimate[2])
        desired[:] = forward, np.clip(raw, -1, 1)
        tracking["angular_raw"] = float(raw)
        self.latest.update(
            nominal=desired.copy(),
            waypoint_index=self.wp,
            prefix_passed=info["prefix_passed"],
            prefix_fraction=info["fraction"],
            progress_heading_error_rad=gate,
        )
        return desired
