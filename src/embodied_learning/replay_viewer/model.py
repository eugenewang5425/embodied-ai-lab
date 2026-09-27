"""Data boundary between an experiment adapter and presentation panels."""

from dataclasses import dataclass, field, replace
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ReplayEntry:
    case: str
    variant: str
    label: str


@dataclass(frozen=True)
class Track:
    key: str
    label: str
    color: str
    poses: np.ndarray
    reference_key: str = "truth"
    role: str = "estimate"


@dataclass(frozen=True)
class CameraSpec:
    """Presentation camera: robot +x forward, +y left, world +z up."""

    height_m: float = 0.30
    forward_m: float = 0.18
    vertical_fov_deg: float = 65.0
    near_clip_m: float = 0.01

    def __post_init__(self):
        if (
            not np.isfinite(
                [self.height_m, self.forward_m, self.vertical_fov_deg, self.near_clip_m]
            ).all()
            or self.height_m <= 0
            or self.near_clip_m <= 0
            or not 1 < self.vertical_fov_deg < 179
        ):
            raise ValueError("invalid presentation camera geometry")


@dataclass
class ReplayRecord:
    title: str
    timestamps: np.ndarray
    tracks: tuple[Track, ...]
    maps: dict[str, np.ndarray]
    resolution_m: float
    primitives: tuple = ()
    wall_count: int = 0
    origin: tuple[float, float] = (0.0, 0.0)
    metadata: dict = field(default_factory=dict)
    camera: CameraSpec = field(default_factory=CameraSpec)
    lidar: dict = field(default_factory=dict)
    goals: dict = field(default_factory=dict)
    missions: dict = field(default_factory=dict)
    plans: dict = field(default_factory=dict)
    lengths: dict = field(default_factory=dict)

    def __post_init__(self):
        self.timestamps = np.array(self.timestamps, dtype=float, copy=True)
        if (
            self.timestamps.ndim != 1
            or len(self.timestamps) < 2
            or not np.isfinite(self.timestamps).all()
            or not (np.diff(self.timestamps) > 0).all()
        ):
            raise ValueError("timestamps must contain at least two finite increasing times")
        if not np.isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("map resolution must be finite and positive")
        if not self.maps or not self.tracks or self.tracks[0].key != "truth":
            raise ValueError("replay needs maps and a first track named truth")
        if len({t.key for t in self.tracks}) != len(self.tracks):
            raise ValueError("duplicate track key")
        owned = []
        for track in self.tracks:
            poses = np.array(track.poses, dtype=float, copy=True)
            if poses.shape != (len(self.timestamps), 3) or not np.isfinite(poses).all():
                raise ValueError(f"{track.key}: expected finite (N,3) poses")
            poses.setflags(write=False)
            owned.append(replace(track, poses=poses))
        self.tracks = tuple(owned)
        self.track_by_key = {track.key: track for track in self.tracks}
        self.maps = {k: np.array(v, dtype=bool, copy=True) for k, v in self.maps.items()}
        shapes = {a.shape for a in self.maps.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or min(next(iter(shapes))) < 2:
            raise ValueError("maps must be equally sized nonempty two-dimensional grids")
        self.timestamps.setflags(write=False)
        for array in self.maps.values():
            array.setflags(write=False)
        self.errors = {}
        self.yaw_errors = {}
        self.statistics = {}
        for track in self.tracks[1:]:
            if track.role == "truth":
                continue
            truth = self.track_by_key[track.reference_key].poses
            error = np.linalg.norm(track.poses[:, :2] - truth[:, :2], axis=1)
            delta = track.poses[:, 2] - truth[:, 2]
            yaw = np.degrees(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
            error.setflags(write=False)
            yaw.setflags(write=False)
            self.errors[track.key] = error
            self.yaw_errors[track.key] = yaw
            length = self.lengths.get(track.key, len(error))
            if not 1 <= length <= len(error):
                raise ValueError("invalid unpadded recording length")
            measured = error[:length]
            self.statistics[track.key] = {
                "mean_m": float(measured.mean()),
                "p95_m": float(np.percentile(measured, 95)),
                "end_m": float(measured[-1]),
            }
        for key, scan in self.lidar.items():
            if key not in self.track_by_key:
                raise ValueError("scan must belong to a track")
            ranges, hits = (
                np.array(scan["ranges"], copy=True),
                np.array(scan["hits"], dtype=bool, copy=True),
            )
            if (
                ranges.ndim != 2
                or ranges.shape[1] == 0
                or ranges.shape[0] != len(self.timestamps)
                or hits.shape != ranges.shape
            ):
                raise ValueError("scan arrays must align with timestamps")
            if not np.isfinite(ranges).all() or (ranges < 0).any():
                raise ValueError("invalid scan ranges")
            ranges.setflags(write=False)
            hits.setflags(write=False)
            self.lidar[key] = {**scan, "ranges": ranges, "hits": hits}
        for collection, shape in (
            (self.goals, (len(self.timestamps), 2)),
            (self.missions, (len(self.timestamps),)),
        ):
            for key, values in collection.items():
                array = np.array(values, copy=True)
                if (
                    key not in self.track_by_key
                    or array.shape != shape
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("goals/missions must align with track timestamps")
                array.setflags(write=False)
                collection[key] = array
        for key, values in self.plans.items():
            array = np.array(values, dtype=float, copy=True)
            if (
                key not in self.track_by_key
                or array.ndim != 3
                or array.shape[0] != len(self.timestamps)
                or array.shape[2] != 2
            ):
                raise ValueError("plans must contain timestamp-aligned (K,2) waypoints")
            if (
                np.isinf(array).any()
                or not (np.isfinite(array).all(axis=2) | np.isnan(array).all(axis=2)).all()
            ):
                raise ValueError("plan padding must be complete NaN coordinate pairs")
            array.setflags(write=False)
            self.plans[key] = array

    @property
    def estimates(self):
        return tuple(t for t in self.tracks if t.key in self.errors)

    @property
    def method_keys(self):
        return tuple(
            self.metadata.get(
                "methods", {t.key: {} for t in self.tracks if t.role != "truth" or t.key == "truth"}
            )
        )

    def method_info(self, key):
        return self.metadata.get("methods", {}).get(
            key,
            {"label": self.track_by_key[key].label, "description": "位置估计与实际运动的对照。"},
        )

    def true_key(self, key):
        return self.track_by_key[key].reference_key if key != "truth" else "truth"

    def method_tracks(self, key):
        return {key, self.true_key(key)}

    def goal_info(self, frame, key):
        if key not in self.goals:
            return None
        mission = int(self.missions[key][frame])
        names = self.metadata.get("task_names", [])
        name = names[mission] if 0 <= mission < len(names) else f"巡检点 {mission + 1}"
        return name, self.goals[key][frame]

    def scan_world(self, frame, key, estimated=True):
        """Only hits become points; max-range returns never draw fake obstacles."""
        pose_key = key if estimated else self.true_key(key)
        pose = self.track_by_key[pose_key].poses[frame]
        scan = self.lidar.get(key, self.lidar.get("truth"))
        if scan is not None:
            ranges, hits = scan["ranges"][frame], scan["hits"][frame]
        elif self.metadata.get("lidar_mode") == "geometry_recast" and self.primitives:
            from embodied_learning.experiments.grid_nav import cast_rays

            actual = self.track_by_key[self.true_key(key)].poses[frame]
            offsets = np.arange(64) * (2 * np.pi / 64)
            ranges, hits = cast_rays(*actual[:2], actual[2] + offsets, self.primitives, (), 4.0)
        else:
            return np.empty((0, 2))
        angles = pose[2] + np.arange(len(ranges)) * (2 * np.pi / len(ranges))
        return (
            pose[:2]
            + np.column_stack((np.cos(angles[hits]), np.sin(angles[hits]))) * ranges[hits, None]
        )

    @property
    def extent(self):
        rows, cols = next(iter(self.maps.values())).shape
        x, y = self.origin
        return (x, x + cols * self.resolution_m, y, y + rows * self.resolution_m)


class ReplaySource(Protocol):
    """Implement this protocol to connect a different experiment."""

    title: str
    entries: tuple[ReplayEntry, ...]

    def load(self, case: str, variant: str) -> ReplayRecord: ...
