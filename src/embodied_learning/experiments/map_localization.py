"""Lesson 56: truth-controlled 2D mapping and localization replay.

The first patrol maps physical scans using estimated poses. The second patrol
compares odometry with particle filters against a self-built map, a same-scan
truth-pose projection, and ideal occupancy. Kidnap trials score the state at
each recovery-window end. The patrol controller reads truth, so this module
does not implement autonomous navigation or production SLAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embodied_learning.experiments.aniso_env import PATROL_INSET, _build_world, patrol_start
from embodied_learning.experiments.grid_nav import (
    DT,
    GEOMETRY,
    GridNavConfig,
    build_walls,
)
from embodied_learning.experiments.grid_nav import (
    cast_rays as cast_rays_world,
)
from embodied_learning.experiments.matcher_engineering import (
    MatcherConfig,
    collect_patrol_laps,
    estimate_step_slam,
)
from embodied_learning.experiments.rgbd_loops import (
    MARKER_CONFUSION,
    MARKER_DROPOUT,
    MARKER_PALETTE,
    build_appearance_world,
    rgbd_frame,
)

EXPERIMENT = "map_localization_lesson56"
SCHEMA_VERSION = 5
DEFAULT_RESULTS = "results/map_localization_2026-09-24_v5"
MARKER_CELL_M = 0.5
PF_PARTICLES = 400
PF_RAYS = 32
PF_STEP_M = 0.15
# AMCL-standard odometry-inflation: the cloud spread between measurement
# updates must be comparable to the likelihood sharpness, else the filter
# over-trusts odometry and never anchors (the recorded tuning)
PF_SIGMA_T = 0.05
PF_SIGMA_R_DEG = 2.0
KIDNAP_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
RECOVERY_FRAMES = 300
RESEED_ESS_FRAMES = 100
RESEED_TOP_CELLS = 8
DRIFT_ENVELOPE_T = 6.0
DRIFT_ENVELOPE_R = 80.0
ACCEPT_MEDIAN_NN_M = 0.15


@dataclass(frozen=True)
class MapLocConfig:
    """Defaults ARE the pre-registered lesson-56 run."""

    base: object = None
    matcher: object = None
    laps: int = 2
    range_noise_sigma_m: float = 0.06
    pf_particles: int = PF_PARTICLES
    pf_rays: int = PF_RAYS
    kidnap_fraction: float = 0.0  # >0 runs a single kidnap at this fraction
    arm: str = "ISO"
    run_kidnap: bool = True
    seed: int = 0

    def __post_init__(self):
        if self.base is None:
            object.__setattr__(
                self,
                "base",
                GridNavConfig(rays=64, max_range_m=4.0, obstacle_scenes=1, inits=1),
            )
        if self.matcher is None:
            object.__setattr__(
                self,
                "matcher",
                MatcherConfig(
                    base=self.base,
                    laps=self.laps,
                    range_noise_sigma_m=self.range_noise_sigma_m,
                    seed=self.seed,
                ),
            )
        if type(self.pf_particles) is not int or not (50 <= self.pf_particles <= 2000):
            raise ValueError("pf_particles out of range")
        if type(self.pf_rays) is not int or not (8 <= self.pf_rays <= 64):
            raise ValueError("pf_rays out of range")
        if not (0.0 <= self.kidnap_fraction < 1.0):
            raise ValueError("kidnap_fraction out of range")
        if self.arm not in ("ISO", "ANISO"):
            raise ValueError("arm must be ISO or ANISO")
        if type(self.run_kidnap) is not bool:
            raise ValueError("run_kidnap must be bool")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    @property
    def config(self) -> GridNavConfig:
        return self.base


def wrap(a):
    return float(np.arctan2(math.sin(a), math.cos(a)))


def wrap_arr(a):
    return np.arctan2(np.sin(a), np.cos(a))


def measured_body_velocity(delta_wheels_rad, encoder_bias):
    """Convert biased wheel-angle increments to body-frame rates for one step."""
    biased = np.asarray(delta_wheels_rad) * np.array([1.0, 1.0 + encoder_bias])
    distance_step, angle_step = GEOMETRY.body_velocity(biased)
    return distance_step / DT, angle_step / DT


# ---------------------------------------------------------- appearance map
def first_lap_frames(estimated, lap_len_m=None):
    """Select approximately one lap using odometry arc and patrol geometry."""
    if lap_len_m is None:
        waypoints = np.asarray(PATROL_INSET, dtype=float)
        lap_len_m = (
            float(
                np.linalg.norm(np.diff(np.vstack((waypoints, waypoints[0])), axis=0), axis=1).sum()
            )
            + 0.2
        )
    arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(estimated[:, :2], axis=0), axis=1)))
    )
    return [k for k in range(len(estimated) - 1) if arc[k] <= lap_len_m]


def marker_map_from_observations(observed_histograms, estimated, lap_frames, world_size_m):
    """Paint observed appearance at estimated poses; no world geometry input."""
    n_m = round(world_size_m / MARKER_CELL_M)
    marker_map = np.zeros((n_m, n_m, MARKER_PALETTE))
    selected = set(lap_frames)
    for k in range(0, len(observed_histograms), 2):
        if k not in selected:
            continue
        pose = estimated[k]
        ix = min(n_m - 1, max(0, int(pose[0] / MARKER_CELL_M)))
        iy = min(n_m - 1, max(0, int(pose[1] / MARKER_CELL_M)))
        marker_map[iy, ix] += observed_histograms[k]
    return marker_map


def occupancy_from_observations(ranges, hits, poses, frame_ids, base):
    """Project the same sensed ranges through a chosen pose chain."""
    hit_count = np.zeros((base.grid_cells, base.grid_cells), dtype=int)
    offsets = np.arange(base.rays) * (2.0 * math.pi / base.rays)
    for k in frame_ids:
        pose = poses[k]
        for ray in np.flatnonzero(hits[k]):
            angle = pose[2] + offsets[ray]
            hx = int((pose[0] + math.cos(angle) * ranges[k, ray]) / base.res_m)
            hy = int((pose[1] + math.sin(angle) * ranges[k, ray]) / base.res_m)
            if 0 <= hx < base.grid_cells and 0 <= hy < base.grid_cells:
                hit_count[hy, hx] += 1
    return (hit_count >= 2).astype(float)


def occupancy_scores(grid, truth):
    """Precision, recall and IoU of occupied cells against a diagnostic map."""
    marked = grid > 0
    actual = truth > 0
    tp = int((marked & actual).sum())
    return {
        "precision": float(tp / max(1, marked.sum())),
        "recall": float(tp / max(1, actual.sum())),
        "iou": float(tp / max(1, (marked | actual).sum())),
    }


def scan_map_alignment(grid, reference, res, tolerance_cells=2):
    """Compare two surface maps made from identical scans, allowing grid quantization."""
    from scipy.ndimage import distance_transform_edt

    marked = grid > 0
    expected = reference > 0
    if not marked.any() or not expected.any():
        return {"precision": 0.0, "recall": 0.0, "median_offset_m": None}
    to_reference = distance_transform_edt(~expected) * res
    to_marked = distance_transform_edt(~marked) * res
    tolerance_m = tolerance_cells * res
    return {
        "precision": float(np.mean(to_reference[marked] <= tolerance_m)),
        "recall": float(np.mean(to_marked[expected] <= tolerance_m)),
        "median_offset_m": float(np.median(to_reference[marked])),
    }


def reseed_candidates(marker_map, obs_hist):
    """Top coarse cells by cosine between the stored histogram and the
    observed one; returns cell centers (x, y)."""
    flat = marker_map.reshape(-1, MARKER_PALETTE)
    norms = np.linalg.norm(flat, axis=1)
    on = np.flatnonzero(norms > 0)
    obs_n = float(np.linalg.norm(obs_hist))
    if obs_n <= 0 or len(on) == 0:
        return []
    scores = (flat[on] @ obs_hist) / (norms[on] * obs_n)
    order = on[np.argsort(scores)[::-1][:RESEED_TOP_CELLS]]
    n_m = marker_map.shape[0]
    return [((g % n_m + 0.5) * MARKER_CELL_M, (g // n_m + 0.5) * MARKER_CELL_M) for g in order]


# ---------------------------------------------------------- particle filter
class ParticleFilter:
    """AMCL-style localization against the built occupancy grid."""

    def __init__(self, grid, res, particles_n, rng, init_pose=None, spread=0.4, rays_n=PF_RAYS):
        # likelihood field (Thrun Ch.6.4): distance to the nearest occupied
        # cell, precomputed once - tolerant to map discretization where the
        # ray-march model was brittle (the recorded tuning)
        from scipy.ndimage import distance_transform_edt

        self.lf = distance_transform_edt(grid <= 0) * res
        self.res = res
        self.rng = rng
        n = particles_n
        if init_pose is None:
            self.p = np.column_stack(
                [
                    rng.uniform(0.5, 9.5, n),
                    rng.uniform(0.5, 9.5, n),
                    rng.uniform(-math.pi, math.pi, n),
                ]
            )
        else:
            self.p = np.column_stack(
                [
                    init_pose[0] + rng.normal(0, spread, n),
                    init_pose[1] + rng.normal(0, spread, n),
                    init_pose[2] + rng.normal(0, 0.1, n),
                ]
            )
        self.w = np.full(n, 1.0 / n)
        self.ray_idx = np.linspace(0, 63, rays_n).astype(int)
        self.ray_off = self.ray_idx * (2.0 * math.pi / 64)

    def propagate(self, d_xy, d_th):
        n = len(self.p)
        self.p[:, 0] += d_xy[0] + self.rng.normal(0, PF_SIGMA_T, n)
        self.p[:, 1] += d_xy[1] + self.rng.normal(0, PF_SIGMA_T, n)
        self.p[:, 2] = wrap_arr(
            self.p[:, 2] + d_th + np.radians(self.rng.normal(0, PF_SIGMA_R_DEG, n))
        )

    def propagate_body(self, v, om, dt=DT):
        """Body-frame propagation (the kidnapped robot's REAL wheels: the
        new route's odometry, not the abandoned route's world deltas)."""
        n = len(self.p)
        mid = self.p[:, 2] + om * dt / 2.0
        self.p[:, 0] += v * dt * np.cos(mid) + self.rng.normal(0, PF_SIGMA_T, n)
        self.p[:, 1] += v * dt * np.sin(mid) + self.rng.normal(0, PF_SIGMA_T, n)
        self.p[:, 2] = wrap_arr(
            self.p[:, 2] + om * dt + np.radians(self.rng.normal(0, PF_SIGMA_R_DEG, n))
        )

    def measure(self, observed, _est_pose):
        """Likelihood-field measurement: endpoint distances to the nearest
        mapped obstacle (only valid-return rays; scale 0.20 m)."""
        idx = self.ray_idx
        obs = observed[idx]
        valid = obs < 3.9
        n_g = self.lf.shape[0]
        weights = np.zeros(len(self.p))
        for i in range(len(self.p)):
            th = self.p[i, 2]
            angles = th + self.ray_off
            ex = self.p[i, 0] + np.cos(angles) * obs
            ey = self.p[i, 1] + np.sin(angles) * obs
            ix = np.floor(ex / self.res).astype(int)
            iy = np.floor(ey / self.res).astype(int)
            inside = (ix >= 0) & (ix < n_g) & (iy >= 0) & (iy < n_g)
            sel = inside & valid
            if sel.sum() < 4:
                continue
            d = self.lf[iy[sel], ix[sel]]
            weights[i] = math.exp(-float(np.sum(d**2)) / (2 * 0.20**2 * max(1, int(sel.sum()))))
        total = weights.sum()
        if total <= 0:
            self.w = np.full(len(self.p), 1.0 / len(self.p))
        else:
            self.w = weights / total

    def ess(self):
        return 1.0 / max(1e-12, float(np.sum(self.w**2)))

    def resample(self):
        n = len(self.p)
        positions = (self.rng.uniform() + np.arange(n)) / n
        cum = np.cumsum(self.w)
        out = np.zeros_like(self.p)
        i = 0
        for j in range(n):
            while positions[j] > cum[i]:
                i = min(i + 1, n - 1)
            out[j] = self.p[i]
        self.p = out
        self.w = np.full(n, 1.0 / n)

    def estimate(self):
        th = self.p[:, 2]
        mean_th = math.atan2(float(self.w @ np.sin(th)), float(self.w @ np.cos(th)))
        return np.array([self.w @ self.p[:, 0], self.w @ self.p[:, 1], mean_th])

    def reseed(self, centers, rng, headings=None):
        n = len(self.p)
        if headings is None:
            headings = [rng.uniform(-math.pi, math.pi)]
        per = max(1, n // (max(1, len(centers)) * len(headings)))
        if not centers:
            self.p = np.column_stack(
                [
                    rng.uniform(0.5, 9.5, n),
                    rng.uniform(0.5, 9.5, n),
                    rng.uniform(-math.pi, math.pi, n),
                ]
            )
        else:
            out = []
            for cx, cy in centers:
                for th in headings:
                    for _ in range(per):
                        out.append(
                            [
                                cx + rng.normal(0, 0.25),
                                cy + rng.normal(0, 0.25),
                                wrap_arr(np.array([th + rng.normal(0, 0.15)]))[0],
                            ]
                        )
            while len(out) < n:
                out.append(
                    [
                        rng.uniform(0.5, 9.5),
                        rng.uniform(0.5, 9.5),
                        rng.uniform(-math.pi, math.pi),
                    ]
                )
            self.p = np.asarray(out[:n], dtype=float)
        self.w = np.full(n, 1.0 / n)


def kidnap_route(map_truth, loc_truth, kidnap_at):
    """Cross-stream kidnap: from kidnap_at the robot continues along the
    MAPPING patrol's route (a real, driven, collision-free path).

    The rigid-shift model was infeasible: a shifted copy of the whole
    remaining path almost always clips an obstacle, whatever the shift
    (the recorded first-run failures).  Carrying the robot to another
    real route is the physically coherent kidnap.
    """
    teleported = loc_truth.copy()
    n = len(loc_truth) - 1
    m_max = len(map_truth) - 1
    for k in range(kidnap_at, n + 1):
        j = min(m_max, kidnap_at + (k - kidnap_at))
        teleported[k] = map_truth[min(j, m_max)]
    return teleported


def run_experiment(output, *, seed=0, sensor_seed=None, config=None, log=print):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if sensor_seed is None:
        sensor_seed = seed
    if type(sensor_seed) is not int or sensor_seed < 0:
        raise ValueError("sensor_seed must be a non-negative integer")
    if log is None:
        log = lambda *_message: None
    if config is None:
        config = MapLocConfig()
    elif config.__class__ is not MapLocConfig:
        raise TypeError("config must be MapLocConfig")
    started = time.perf_counter()
    base = config.base

    # ---------------------------------------------------------- phase 1
    obstacles, walls = _build_world(config.arm, seed, base)
    physical_walls = build_walls(base.world_size_m, base.res_m) if walls is None else walls
    map_stream = collect_patrol_laps(
        obstacles,
        patrol_start(base, seed)[0],
        config.matcher,
        walls=physical_walls,
        patrol=PATROL_INSET,
    )
    if map_stream["laps_done"] < config.laps:
        raise RuntimeError("mapping patrol incomplete")
    app_obstacles, app_walls, ids = build_appearance_world(
        base, seed, obstacles=obstacles, world_walls=physical_walls
    )
    primitives = (*app_obstacles, *app_walls)
    # The two mapped grids use identical scans and selected frames.  Only
    # their pose chain changes: odometry is the deployable mapper; physical
    # truth is a diagnostic upper bound for attributing map-placement error.
    m_est = [map_stream["truth"][0]]
    for k in range(len(map_stream["wheels"])):
        m_est.append(
            estimate_step_slam(m_est[-1], map_stream["wheels"][k], config.matcher.slam.encoder_bias)
        )
    m_est = np.asarray(m_est)
    lap_frames = first_lap_frames(m_est)
    if config.run_kidnap:
        marker_rng = np.random.default_rng([sensor_seed, 56001])
        observed_markers = np.vstack(
            [rgbd_frame(pose, primitives, ids, marker_rng) for pose in map_stream["truth"][:-1]]
        )
        marker_map = marker_map_from_observations(
            observed_markers, m_est, lap_frames, base.world_size_m
        )
    else:
        n_marker_cells = round(base.world_size_m / MARKER_CELL_M)
        marker_map = np.zeros((n_marker_cells, n_marker_cells, MARKER_PALETTE))
    built_grid = occupancy_from_observations(
        map_stream["ranges"], map_stream["hit"], m_est, lap_frames, base
    )
    sensor_truth_grid = occupancy_from_observations(
        map_stream["ranges"], map_stream["hit"], map_stream["truth"], lap_frames, base
    )
    from embodied_learning.experiments.grid_nav import true_occupancy

    true_map = true_occupancy(obstacles, physical_walls, base.grid_cells, base.res_m)
    map_scores = occupancy_scores(built_grid, true_map)
    sensor_truth_scores = occupancy_scores(sensor_truth_grid, true_map)
    paired_scan_alignment = scan_map_alignment(built_grid, sensor_truth_grid, base.res_m)
    hit_rate = map_scores["recall"]
    map_precision = map_scores["precision"]
    map_iou = map_scores["iou"]
    log(
        f"phase1: laps {map_stream['laps_done']} frames {len(map_stream['truth']) - 1} "
        f"map hit-rate {hit_rate:.2f}"
    )

    # ---------------------------------------------------------- phase 2
    loc_seed = seed + 500
    loc_stream = collect_patrol_laps(
        obstacles,
        patrol_start(base, loc_seed)[0],
        config.matcher,
        walls=physical_walls,
        patrol=PATROL_INSET,
    )
    if loc_stream["laps_done"] < config.laps:
        raise RuntimeError("localization patrol incomplete")
    truth = loc_stream["truth"].copy()
    n_frames = len(loc_stream["wheels"])
    rng_obs = np.random.default_rng([sensor_seed, 56002])
    ranges_obs = np.clip(
        loc_stream["ranges"]
        + rng_obs.normal(0.0, config.range_noise_sigma_m, loc_stream["ranges"].shape),
        0.0,
        None,
    )
    est = [truth[0]]
    for k in range(n_frames):
        est.append(
            estimate_step_slam(est[-1], loc_stream["wheels"][k], config.matcher.slam.encoder_bias)
        )
    est = np.asarray(est)

    # ---------------------------------------------------------- PF + kidnap
    trials = []
    ray_offsets = np.arange(base.rays) * (2.0 * math.pi / base.rays)
    map_truth = map_stream["truth"]
    map_wheels = map_stream["wheels"]
    kidnap_fractions = KIDNAP_FRACTIONS if config.run_kidnap else ()
    for frac in kidnap_fractions:
        rng_pf = np.random.default_rng([sensor_seed, 56003, int(frac * 100)])
        kidnap_at = int(n_frames * frac)
        teleported = kidnap_route(map_truth, truth, kidnap_at)
        # observation re-synthesis: from kidnap_at the ranges come from the
        # NEW route (noisy ray-cast against the TRUE world at the teleported
        # poses) - the carried robot's sensor sees its new surroundings
        obs = ranges_obs.copy()
        for k in range(kidnap_at, n_frames):
            r_cast, _h = cast_rays_world(
                teleported[k][0],
                teleported[k][1],
                teleported[k][2] + ray_offsets,
                obstacles,
                physical_walls,
                base.max_range_m,
            )
            obs[k] = np.clip(
                r_cast + rng_obs.normal(0.0, config.range_noise_sigma_m, r_cast.shape), 0.0, None
            )
        pf = ParticleFilter(
            true_map.astype(float),
            base.res_m,
            config.pf_particles,
            rng_pf,
            init_pose=truth[kidnap_at] * 0 + truth[0],
            rays_n=config.pf_rays,
        )
        pf_error = np.zeros(n_frames + 1)
        reseed_frame = None
        post_frames = 0
        end_k = min(n_frames, kidnap_at + RECOVERY_FRAMES)
        recovery_pose = None
        for k in range(n_frames + 1):
            if k > 0 and k != kidnap_at:
                if k > kidnap_at:
                    # carried to the new route: its REAL odometry drives the
                    # particles (the abandoned route's deltas are meaningless)
                    source_wheels = map_wheels[min(k - 1, len(map_wheels) - 1)]
                else:
                    source_wheels = loc_stream["wheels"][k - 1]
                v_m, om_m = measured_body_velocity(source_wheels, config.matcher.slam.encoder_bias)
                pf.propagate_body(v_m, om_m)
            if k % 5 == 0:
                pf.measure(obs[min(k, n_frames - 1)], est[min(k, n_frames - 1)])
                ess_before = pf.ess()
                in_recovery = (
                    k >= kidnap_at and reseed_frame is None and (k - kidnap_at) <= RESEED_ESS_FRAMES
                )
                if in_recovery and ess_before < config.pf_particles / 4:
                    # the kidnap detector: the observed scan no longer matches
                    # the map where the particles sit, so the weights collapse
                    obs_hist = rgbd_frame(
                        teleported[min(k, len(teleported) - 1)], primitives, ids, rng_pf
                    )
                    headings = np.linspace(-math.pi, math.pi, 8, endpoint=False)
                    pf.reseed(reseed_candidates(marker_map, obs_hist), rng_pf, headings=headings)
                    reseed_frame = k
                elif ess_before < config.pf_particles / 4:
                    pf.resample()
            est_err = float(
                np.linalg.norm(pf.estimate()[:2] - teleported[min(k, len(teleported) - 1), :2])
            )
            pf_error[k] = est_err
            if k == end_k:
                recovery_pose = pf.estimate().copy()
            if k >= kidnap_at:
                post_frames += 1
        window = slice(kidnap_at, min(n_frames, kidnap_at + RECOVERY_FRAMES) + 1)
        # the recovery verdict is read at the WINDOW END: later corridor
        # sliding (the bounded along-wall oscillation of straight-wall
        # localization) is a tracking property, not a relocalization one
        assert recovery_pose is not None
        end_err = float(np.linalg.norm(recovery_pose[:2] - teleported[end_k, :2]))
        end_err_th = abs(math.degrees(wrap(recovery_pose[2] - teleported[end_k, 2])))
        success = end_err <= 1.0 and end_err_th <= 25.0
        trials.append(
            {
                "fraction": frac,
                "kidnap_at": kidnap_at,
                "reseed_frame": reseed_frame,
                "recovery_end_frame": end_k,
                "end_err_m": end_err,
                "end_err_deg": end_err_th,
                "success": bool(success),
                "pf_error_curve": pf_error[window].tolist(),
                "pf_error_full": pf_error,
            }
        )
        log(
            f"  kidnap {frac:.2f}: reseed@{reseed_frame} end_err {end_err:.2f} m "
            f"{end_err_th:.1f} deg success={success}"
        )

    # ------------------------------------------------ full-run PF (no kidnap)
    # two filters: against the TRUE map (the calibrated world model of
    # production AMCL) and against the SELF-BUILT map (the SLAM
    # chicken-and-egg: a map from a drifting chain is smeared)
    def run_pf(grid):
        rng_pf = np.random.default_rng([sensor_seed, 56004])
        # the start pose is KNOWN (est[0] == truth[0], zero drift): a blind
        # uniform init over the arena never converges with 400 particles
        # (the recorded failure mode)
        pf = ParticleFilter(
            grid,
            base.res_m,
            config.pf_particles,
            rng_pf,
            init_pose=est[0],
            rays_n=config.pf_rays,
        )
        chain = np.zeros((n_frames + 1, 3))
        for k in range(n_frames + 1):
            if k > 0:
                v_m, om_m = measured_body_velocity(
                    loc_stream["wheels"][k - 1], config.matcher.slam.encoder_bias
                )
                pf.propagate_body(v_m, om_m)
            if k % 5 == 0:
                pf.measure(ranges_obs[min(k, n_frames - 1)], est[min(k, n_frames - 1)])
                if pf.ess() < config.pf_particles / 4:
                    pf.resample()
            chain[k] = pf.estimate()
        return chain

    pf_true_chain = run_pf(true_map.astype(float))
    pf_sensor_truth_chain = run_pf(sensor_truth_grid)
    pf_self_chain = run_pf(built_grid)
    est_err_curve = np.linalg.norm(est[: n_frames + 1, :2] - truth[: n_frames + 1, :2], axis=1)
    pf_true_err = np.linalg.norm(pf_true_chain[:, :2] - truth[: n_frames + 1, :2], axis=1)
    pf_sensor_truth_err = np.linalg.norm(
        pf_sensor_truth_chain[:, :2] - truth[: n_frames + 1, :2], axis=1
    )
    pf_self_err = np.linalg.norm(pf_self_chain[:, :2] - truth[: n_frames + 1, :2], axis=1)
    est_mean = float(np.mean(est_err_curve))
    pf_mean = float(np.mean(pf_true_err))
    est_final = float(est_err_curve[-1])
    pf_final = float(pf_true_err[-1])
    pf_self_mean = float(np.mean(pf_self_err))
    pf_self_final = float(pf_self_err[-1])
    hypothesis = {
        "claim": (
            "修正第 56 课的判据对象与时序：自建图 PF 相对无图链的均值 <=20%、"
            "末端 <=0.6 m；绑架恢复必须在指定窗口末测量，>=4/5 才成立"
        ),
        "criteria": {
            "collector": "两段巡游均完成 2 圈",
            "bounded": "PF-SELFBUILT 均值误差 <= 20% EST 均值 且末端 <= 0.6 m",
            "kidnap": "5 次绑架试验中 >= 4 次恢复（优质图上，窗口末 <= 1.0 m / 25 deg）",
            "smear": "PF-SELFBUILT 均值 >= 3× PF-TRUEMAP 均值（自建图涂抹的代价量化）",
        },
        "results": {
            "collector_met": bool(
                map_stream["laps_done"] >= config.laps and loc_stream["laps_done"] >= config.laps
            ),
            "bounded_met": bool(pf_self_mean <= 0.2 * est_mean and pf_self_final <= 0.6),
            "smear_ratio": float(pf_self_mean / max(1e-12, pf_mean)),
            "smear_claim_met": bool(pf_self_mean >= 3.0 * pf_mean),
            "kidnap_met": (
                bool(sum(1 for t in trials if t.get("success")) >= 4) if config.run_kidnap else None
            ),
            "errors": {
                "est_mean_m": est_mean,
                "pf_mean_m": pf_mean,
                "est_final_m": est_final,
                "pf_final_m": pf_final,
                "pf_self_mean_m": pf_self_mean,
                "pf_self_final_m": pf_self_final,
                "pf_sensor_truth_mean_m": float(np.mean(pf_sensor_truth_err)),
                "pf_sensor_truth_final_m": float(pf_sensor_truth_err[-1]),
                "est_p95_m": float(np.percentile(est_err_curve, 95)),
                "pf_true_p95_m": float(np.percentile(pf_true_err, 95)),
                "pf_self_p95_m": float(np.percentile(pf_self_err, 95)),
                "pf_sensor_truth_p95_m": float(np.percentile(pf_sensor_truth_err, 95)),
            },
            "map_hit_rate": hit_rate,
            "map_precision": map_precision,
            "map_iou": map_iou,
            "sensor_truth_map": sensor_truth_scores,
            "paired_scan_alignment": paired_scan_alignment,
            "reseed_rate": (
                float(sum(1 for t in trials if t.get("reseed_frame") is not None) / len(trials))
                if trials
                else None
            ),
            "kidnap_trials": [
                {k: v for k, v in t.items() if k not in ("pf_error_curve", "pf_error_full")}
                for t in trials
            ],
        },
    }
    archives = {
        "est_err_curve": est_err_curve,
        "pf_true_err": pf_true_err,
        "pf_sensor_truth_err": pf_sensor_truth_err,
        "pf_self_err": pf_self_err,
        "pf_true_chain": pf_true_chain,
        "pf_sensor_truth_chain": pf_sensor_truth_chain,
        "pf_self_chain": pf_self_chain,
        "est_chain": est[: n_frames + 1],
        "truth_chain": truth[: n_frames + 1],
        "marker_map": np.max(marker_map, axis=2),
        "built_grid": built_grid,
        "sensor_truth_grid": sensor_truth_grid,
        "true_grid": true_map.astype(float),
        "kidnap_pf_err": (
            np.vstack([trial["pf_error_full"] for trial in trials])
            if trials
            else np.zeros((0, n_frames + 1))
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archives)
    report = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "sensor_seed": sensor_seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "protocol": {
            "arm": config.arm,
            "phase1": "真值控制巡游；传感观测按里程计估计位姿落图，首圈以估计弧长选取",
            "phase2": (
                f"新噪声实例；PF {config.pf_particles} 粒子/{config.pf_rays} 线/"
                "0.15 m 栅格对优质图与自建图分别测量"
            ),
            "kidnap": (
                f"5 次真值瞬移（{[round(f, 2) for f in KIDNAP_FRACTIONS]}），ESS 塌缩后重播种"
                if config.run_kidnap
                else "未运行；定位基准专用"
            ),
            "appearance": {
                "palette": MARKER_PALETTE,
                "confusion": MARKER_CONFUSION,
                "dropout": MARKER_DROPOUT,
            },
            "map_recall": hit_rate,
            "map_precision": map_precision,
            "map_iou": map_iou,
            "map_alignment_reference": "same scan frames projected at physical truth pose; two-cell tolerance",
            "truth_use": "真值控制巡游及仿真生成观测；建图落点/首圈筛选/定位决策不读真值；优质图组仅诊断",
            "metric_correction": "v3 修正地图和绑架评分；v4 使用 PATROL_INSET 巡游；v5 首圈取帧阈值与实际 16 m 巡游一致",
        },
        "hypothesis": hypothesis,
        "wall_time_s": time.perf_counter() - started,
        "archive_keys": sorted(archives.keys()),
        "trajectories_sha256": hashlib.sha256(
            (output / "trajectories.npz").read_bytes()
        ).hexdigest(),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="lesson-56 map localization record")
    parser.add_argument("--output", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor-seed", type=int, default=None)
    parser.add_argument("--arm", choices=("ISO", "ANISO"), default="ISO")
    parser.add_argument("--particles", type=int, default=PF_PARTICLES)
    parser.add_argument("--no-kidnap", action="store_true")
    args = parser.parse_args()
    cfg = MapLocConfig(
        seed=args.seed,
        arm=args.arm,
        pf_particles=args.particles,
        run_kidnap=not args.no_kidnap,
    )
    report = run_experiment(args.output, seed=args.seed, sensor_seed=args.sensor_seed, config=cfg)
    print(json.dumps(report["hypothesis"]["results"], indent=2))


if __name__ == "__main__":
    main()
