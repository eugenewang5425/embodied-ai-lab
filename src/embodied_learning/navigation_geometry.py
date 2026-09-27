"""Metric sensor extrinsics, body volumes and swept stopping envelopes.

Robot axes: x forward, y left, z up. Camera optical coordinates: z forward,
x image-right, y image-down. Depth is optical-axis distance, not ray length.
All dimensions describe a controlled simulation fixture, not a measured robot.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class NavigationRig:
    front_m: float = 0.30
    rear_m: float = 0.25
    half_width_m: float = 0.22
    top_m: float = 0.58
    lidar_x_m: float = 0.08
    lidar_z_m: float = 0.32
    camera_x_m: float = 0.20
    camera_z_m: float = 0.55
    vfov_deg: float = 65.0
    width: int = 80
    height: int = 60
    lidar_rays: int = 180
    max_range_m: float = 8.0
    latency_s: float = 0.20
    brake_m_s2: float = 0.8
    angular_brake_rad_s2: float = 1.8
    margin_m: float = 0.06

    @property
    def parts(self):
        # Conservative full rectangular base, mast and camera housing.
        return np.array(
            [
                [-self.rear_m, self.front_m, -self.half_width_m, self.half_width_m, 0.0, 0.30],
                [0.14, 0.19, -0.025, 0.025, 0.30, 0.56],
                [0.15, 0.20, -0.03, 0.03, 0.52, self.top_m],
            ]
        )

    @property
    def radius(self):
        return float(np.hypot(max(self.front_m, self.rear_m), self.half_width_m))

    @property
    def focal(self):
        return self.height / (2 * np.tan(np.deg2rad(self.vfov_deg) / 2))

    def to_dict(self):
        return asdict(self)


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def world_points(points, pose):
    out = np.array(points, dtype=float, copy=True)
    out[..., :2] = out[..., :2] @ rotation(pose[2]).T + pose[:2]
    return out


def body_points(points, pose):
    out = np.array(points, dtype=float, copy=True)
    out[..., :2] = (out[..., :2] - pose[:2]) @ rotation(pose[2])
    return out


def ray_boxes(origin, directions, boxes, max_range=8.0):
    """Slab intersection; parameter t retains optical depth for camera rays."""
    directions = np.asarray(directions, float)
    boxes = np.asarray(boxes, float).reshape(-1, 6)
    if not len(boxes):
        return np.full(len(directions), max_range), np.full(len(directions), -1, int)
    lo, hi = boxes[:, [0, 2, 4]], boxes[:, [1, 3, 5]]
    d = directions[:, None, :]
    parallel = np.abs(d) < 1e-12
    denom = np.where(parallel, 1.0, d)
    a, b = (lo - origin) / denom, (hi - origin) / denom
    entry = np.where(parallel, -np.inf, np.minimum(a, b))
    leave = np.where(parallel, np.inf, np.maximum(a, b))
    invalid = np.any(parallel & ((origin < lo) | (origin > hi)), axis=-1)
    near, far = entry.max(axis=-1), leave.min(axis=-1)
    dist = np.where((far >= np.maximum(near, 0)) & ~invalid, np.maximum(near, 0), np.inf)
    ids = np.argmin(dist, axis=1)
    ranges = dist[np.arange(len(ids)), ids]
    return np.minimum(ranges, max_range), np.where(ranges < max_range, ids, -1)


def camera_directions(rig):
    u, v = np.meshgrid(np.arange(rig.width), np.arange(rig.height))
    # Pixel centers; same convention as the MuJoCo perspective renderer.
    return np.column_stack(
        (
            np.ones(u.size),
            -(u.ravel() + 0.5 - rig.width / 2) / rig.focal,
            -(v.ravel() + 0.5 - rig.height / 2) / rig.focal,
        )
    )


def sense(boxes, pose, rig, rng=None):
    """Generate co-timed LiDAR and depth from the same 3D obstacles."""
    angles = np.arange(rig.lidar_rays) * (2 * np.pi / rig.lidar_rays)
    directions = np.column_stack(
        (np.cos(angles + pose[2]), np.sin(angles + pose[2]), np.zeros(len(angles)))
    )
    origin = world_points([[rig.lidar_x_m, 0, rig.lidar_z_m]], pose)[0]
    ranges, ids = ray_boxes(origin, directions, boxes, rig.max_range_m)
    hit = ids >= 0
    if rng is not None:
        ranges = np.where(
            hit, np.clip(ranges + rng.normal(0, 0.005, len(ranges)), 0.01, rig.max_range_m), ranges
        )
    dirs = camera_directions(rig)
    dirs[:, :2] = dirs[:, :2] @ rotation(pose[2]).T
    camera = world_points([[rig.camera_x_m, 0, rig.camera_z_m]], pose)[0]
    ground = [-100, 100, -100, 100, -0.05, 0]
    depth, did = ray_boxes(camera, dirs, np.vstack((boxes, ground)), rig.max_range_m)
    valid = did >= 0
    if rng is not None:
        depth = np.where(
            valid, np.clip(depth + rng.normal(0, 0.003, len(depth)), 0.01, rig.max_range_m), depth
        )
    return ranges, hit, depth.reshape(rig.height, rig.width), valid.reshape(rig.height, rig.width)


def lidar_points(ranges, hit, rig):
    angles = np.arange(len(ranges)) * (2 * np.pi / len(ranges))
    return np.column_stack(
        (
            rig.lidar_x_m + ranges[hit] * np.cos(angles[hit]),
            ranges[hit] * np.sin(angles[hit]),
            np.full(np.sum(hit), rig.lidar_z_m),
        )
    )


def depth_points(depth, valid, rig):
    points = camera_directions(rig) * np.asarray(depth).ravel()[:, None]
    points += [rig.camera_x_m, 0, rig.camera_z_m]
    # Ground is not an obstacle; near-floor uncertainty remains a stated limit.
    keep = np.asarray(valid).ravel() & (points[:, 2] > 0.025) & (points[:, 2] < rig.top_m + 0.05)
    return points[keep]


def lidar_obstacle_points(ranges, hit, rig):
    """A 2D return has no height extent: conservatively treat it as a column."""
    points = lidar_points(ranges, hit, rig)
    return np.vstack(
        [np.column_stack((points[:, :2], np.full(len(points), z))) for z in (0.15, 0.4, 0.55)]
    )


def project(points, rig):
    p = np.asarray(points) - [rig.camera_x_m, 0, rig.camera_z_m]
    z = p[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack(
            (
                rig.width / 2 - 0.5 - rig.focal * p[:, 1] / z,
                rig.height / 2 - 0.5 - rig.focal * p[:, 2] / z,
            )
        )
    return uv, z > 0.01


def advance(pose, v, w, dt):
    theta = pose[2]
    if abs(w) < 1e-9:
        delta = np.array([v * dt * np.cos(theta), v * dt * np.sin(theta)])
    else:
        delta = (
            v
            / w
            * np.array(
                [np.sin(theta + w * dt) - np.sin(theta), -np.cos(theta + w * dt) + np.cos(theta)]
            )
        )
    return np.r_[pose[:2] + delta, np.arctan2(np.sin(theta + w * dt), np.cos(theta + w * dt))]


def stop_path(v, w, rig, dt=0.04):
    """Command held during latency, then linear/angular braking to rest."""
    duration = rig.latency_s + max(abs(v) / rig.brake_m_s2, abs(w) / rig.angular_brake_rad_s2)
    p, poses = np.zeros(3), [np.zeros(3)]
    for t in np.arange(0, duration + dt / 2, dt):
        after = max(0.0, t - rig.latency_s)
        vt = np.sign(v) * max(0.0, abs(v) - rig.brake_m_s2 * after)
        wt = np.sign(w) * max(0.0, abs(w) - rig.angular_brake_rad_s2 * after)
        p = advance(p, vt, wt, dt)
        poses.append(p)
    return np.array(poses)


def point_clearance(points, poses, rig, margin=0.0):
    """Minimum signed point-to-body-box distance along a proposed trajectory."""
    if not len(points):
        return float("inf")
    best = float("inf")
    for pose in poses:
        local = body_points(points, pose)
        for b in rig.parts:
            lo, hi = b[[0, 2, 4]], b[[1, 3, 5]]
            # Expand xy only: ground clearance is a distinct geometry question.
            pad = np.array([margin, margin, 0.0])
            q = np.abs(local - (lo + hi) / 2) - ((hi - lo) / 2 + pad)
            dist = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            best = min(best, float(dist.min()))
    return best


def collision_clearance(pose, boxes, rig):
    """Independent box/box separating-axis oracle; never given to a controller.

    Returns a signed separating distance (a conservative clearance lower bound).
    Nonpositive means overlap. Includes the actual height of every body part.
    """
    axes = np.vstack((np.eye(2), rotation(pose[2]).T))
    result = float("inf")
    for part in rig.parts:
        center = np.array([(part[0] + part[1]) / 2, (part[2] + part[3]) / 2])
        half = np.array([(part[1] - part[0]) / 2, (part[3] - part[2]) / 2])
        center = center @ rotation(pose[2]).T + pose[:2]
        rp = np.abs(axes @ rotation(pose[2])) @ half
        for obstacle in boxes:
            if part[5] <= obstacle[4] or part[4] >= obstacle[5]:
                continue
            c = np.array([(obstacle[0] + obstacle[1]) / 2, (obstacle[2] + obstacle[3]) / 2])
            h = np.array([(obstacle[1] - obstacle[0]) / 2, (obstacle[3] - obstacle[2]) / 2])
            sep = np.abs(axes @ (c - center)) - rp - np.abs(axes) @ h
            result = min(result, float(sep.max()))
    return result


def swept_clearance(pose, v, w, dt, boxes, rig):
    # <=1 cm translation or corner travel between samples; report this resolution.
    steps = max(1, int(np.ceil((abs(v) + rig.radius * abs(w)) * dt / 0.01)))
    return min(
        collision_clearance(advance(pose, v, w, t), boxes, rig)
        for t in np.linspace(0, dt, steps + 1)
    )
