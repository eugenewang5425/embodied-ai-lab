"""Lesson 27: foundation-model masks assign landmark identities for localization.

Lessons 18-20 assumed landmark identity: every (range, bearing) reading came
with the ground-truth label of which landmark it belonged to (docs/20 declared
assumption). This experiment replaces that assumption with a vision foundation
model: the monocular-depth bench (grounding_inference.py) renders the lesson-22
scene extended to THREE landmark cylinders, runs the real MobileSAM automatic
mask generator on the rendered RGB, and archives every candidate mask together
with the per-pixel depth and the truth label map. THIS module (main repo, no
torch) consumes that npz and answers:

  1. identity assignment - a mask centroid is back-projected through the
     rendered depth into the world frame and matched to the nearest known
     landmark axis (nearest-neighbour identity). Evaluated against the truth
     label map (dominant label under the mask).
  2. localization chain - mask pixel centroid + depth -> world surface point ->
     (range, bearing) observation -> the UNCHANGED lesson-18 Procrustes solver
     (landmark_localization.solve_pose) -> camera ground pose. Compared under
     identical noise (sigma_r = 1 cm, sigma_beta = 0.57 deg, the lesson-18
     values, common random numbers across groups) against the lesson-18 style
     baseline that observes the landmark AXIS directly with truth identity.
  3. ablation (one variable at a time) - mask quality: truth segmentation vs
     the model mask vs truth mask eroded/dilated by +-k px; identity: truth vs
     model nearest-neighbour vs a FORCED cyclic mismatch (explosion shape);
     M camera poses x 20 seeds.
  4. mechanism - a mask-centroid shift of delta px changes the bearing by
     delta_px / f for a LEVEL camera; the pitched camera scales it by
     1/(cos(pitch) + (v-cy)/f * sin(pitch)). Verified against the exact ray
     bearing, and propagated through the full chain (a COMMON pixel shift is
     absorbed by the heading, not the position, of the rigid fit).

The solve is reused verbatim: lesson-18 ``solve_pose`` returns the VEHICLE
pose (sensor pose composed with the inverse installation SENSOR_IN_BODY); the
camera plays the role of the sensor, so the camera pose is recovered exactly
as ``compose(solve_pose(readings, landmarks), SENSOR_IN_BODY)``.

CLI (main repo venv, torch stays out of the test env):

  uv run python -m embodied_learning.experiments.visual_grounding \
      --input monocular-depth/outputs/<bench dir>/grounding_masks.npz \
      --output results/visual_grounding_my_run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.differential_drive import SENSOR_IN_BODY, compose
from embodied_learning.experiments.pinhole_projection import (
    K_INTRINSIC,
    look_at,
    unproject,
)
from embodied_learning.landmark_localization import (
    bearing_reading,
    solve_pose,
)

EXPERIMENT = "visual_grounding_identity"
SCHEMA_VERSION = 1
# Lesson-18 observation noise, reused verbatim so the comparison is same-condition.
RANGE_STD_M = 0.01
BEARING_STD_RAD = 0.01
DEFAULT_RUNS = 20
DEFAULT_SEED = 0
# Landmark-candidate rule (scene-structure prior, no truth labels involved):
# landmarks stand on the ground (world z = 0) and rise to 1.2 m; a mask is a
# candidate when enough of its pixels have valid depth and the highest
# back-projected pixel reaches at least HEIGHT_GATE_M above the ground.
# Ground-only masks never exceed z = 0; the smallest visible cylinder extent
# (anchor pose, top clipped by the frame) measured ~0.56 m, so 0.5 m sits
# between the two populations.
HEIGHT_GATE_M = 0.5
MIN_VALID_PX = 30
# Mask-quality ablation on the truth segmentation (square structuring element).
ERODE_DILATE_PX = (4, 8)
# Mechanism scan: horizontal mask-centroid shift in pixels.
SHIFT_PX = (1, 2, 4, 8)
# Groups (single-variable factorial):
#   baseline      - lesson-18 sensor: direct axis observation, truth identity
#   truth_mask    - pixel-exact truth segmentation, centroid + depth, truth identity
#   erode{k}/dilate{k} - truth segmentation morphed by k px, truth identity
#   model_mask    - chosen model mask, truth identity (mask quality isolated)
#   model_identity - chosen model mask paired by model NN identity (full pipeline)
#   mismatch      - baseline observations with the identity cyclically shifted
GROUP_NAMES = (
    "baseline",
    "truth_mask",
    "erode4",
    "erode8",
    "dilate4",
    "dilate8",
    "model_mask",
    "model_identity",
    "mismatch",
)
MASK_SOURCE_GROUPS = ("truth_mask", "erode4", "erode8", "dilate4", "dilate8", "model_mask")
REQUIRED_INPUT_KEYS = {
    "rgb",
    "depth_m",
    "valid",
    "landmark_label",
    "model_mask",
    "mask_pose",
    "pose_rotation",
    "pose_translation",
    "landmark_xy",
    "meta_json",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap_angle(angle):
    """Wrap to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def camera_ground_pose(rot, trans):
    """The camera as a lesson-18 style 2D sensor: ground position + heading.

    heading is the world yaw of the camera forward axis (R row 2) projected to
    the horizontal plane; the camera has zero roll by look-at construction.
    """
    rot = np.asarray(rot, dtype=float)
    trans = np.asarray(trans, dtype=float)
    eye = -rot.T @ trans
    forward = rot[2]
    return np.array([eye[0], eye[1], float(np.arctan2(forward[1], forward[0]))])


def pixel_bearing(u, v, rot, heading):
    """Horizontal bearing of the ray through pixel (u, v), relative to heading.

    Exact geometry (no small-angle shortcut): the ray direction in the world is
    R^T K^-1 [u, v, 1]; its horizontal bearing minus the camera heading. For a
    LEVEL camera at the principal point this is atan((u-cx)/f) ~= (u-cx)/f,
    i.e. the delta_phi = delta_px / f law; pitch rescales it by
    1/(cos(pitch) + (v-cy)/f * sin(pitch)). SIGN: bearings follow the world
    atan2(dy, dx) convention (counterclockwise positive), while u grows along
    the camera right axis, so a RIGHTWARD centroid shift DECREASES the bearing:
    d beta / du = -1 / (f * (cos p + (v-cy)/f * sin p)).
    """
    ray_cam = np.linalg.inv(K_INTRINSIC) @ np.array([u, v, 1.0])
    ray_world = np.asarray(rot).T @ ray_cam
    return wrap_angle(float(np.arctan2(ray_world[1], ray_world[0])) - heading)


def pitch_factor(v, rot):
    """d(beta)/du scaled by pitch: 1 / (f * (cos p + (v-cy)/f * sin p)) / (1/f)."""
    forward = np.asarray(rot)[2]
    cos_p = float(np.hypot(forward[0], forward[1]))
    sin_p = float(forward[2])
    v_tilde = (v - K_INTRINSIC[1, 2]) / K_INTRINSIC[0, 0]
    return 1.0 / (cos_p + v_tilde * sin_p)


def mask_observation(u_c, v_c, depth_value, rot, trans):
    """(range, bearing) of the back-projected surface point, camera as sensor."""
    world = unproject(np.array([[u_c, v_c]]), np.array([depth_value]), rot, trans)[0]
    sensor = camera_ground_pose(rot, trans)
    dx, dy = world[0] - sensor[0], world[1] - sensor[1]
    return float(np.hypot(dx, dy)), float(wrap_angle(np.arctan2(dy, dx) - sensor[2])), world


# ------------------------------------------------------------------ mask ops
def shift_mask(mask, dpx):
    """Shift a boolean mask horizontally by dpx pixels (right for dpx > 0)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    width = mask.shape[1]
    if dpx == 0:
        return mask.copy()
    src0, dst0 = max(0, -dpx), max(0, dpx)
    cols = min(width - src0, width - dst0)
    if cols > 0:
        out[:, dst0 : dst0 + cols] = mask[:, src0 : src0 + cols]
    return out


def _shift2d(mask, dr, dc):
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    height, width = mask.shape
    src_r, dst_r = max(0, -dr), max(0, dr)
    src_c, dst_c = max(0, -dc), max(0, dc)
    rows = min(height - src_r, height - dst_r)
    cols = min(width - src_c, width - dst_c)
    if rows > 0 and cols > 0:
        out[dst_r : dst_r + rows, dst_c : dst_c + cols] = mask[
            src_r : src_r + rows, src_c : src_c + cols
        ]
    return out


def _morph_square(mask, k, erode):
    """Erode/dilate with a square (2k+1)^2 element, separable in rows/cols.

    Outside the image counts as background: zero-filled shifts erode border
    pixels and stop dilation at the frame (masks clipped by the image border,
    like the anchor-pose cylinder tops, behave accordingly).
    """
    mask = np.asarray(mask, dtype=bool)
    filled = np.zeros_like(mask)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return filled
    r0 = max(0, rows[0] - k)
    r1 = min(mask.shape[0], rows[-1] + 1 + k)
    c0 = max(0, cols[0] - k)
    c1 = min(mask.shape[1], cols[-1] + 1 + k)
    crop = mask[r0:r1, c0:c1]
    offsets = range(-k, k + 1)
    if erode:
        work = crop.copy()
        for dc in offsets:  # intersection of shifted originals: horizontal segment
            work &= _shift2d(crop, 0, dc)
        snapshot = work.copy()
        for dr in offsets:  # intersection against the SNAPSHOT: vertical segment
            work &= _shift2d(snapshot, dr, 0)
    else:
        work = np.zeros_like(crop)
        for dc in offsets:  # dilate by the horizontal segment first
            work |= _shift2d(crop, 0, dc)
        grown = np.zeros_like(crop)
        for dr in offsets:  # then by the vertical segment
            grown |= _shift2d(work, dr, 0)
        work = grown
    filled[r0:r1, c0:c1] = work
    return filled


def erode_mask(mask, k):
    return _morph_square(mask, k, erode=True)


def dilate_mask(mask, k):
    return _morph_square(mask, k, erode=False)


def mask_boundary(mask):
    """Boundary pixels of a boolean mask (4-neighbourhood), for overlays."""
    mask = np.asarray(mask, dtype=bool)
    inner = (
        _shift2d(mask, 1, 0)
        & _shift2d(mask, -1, 0)
        & _shift2d(mask, 0, 1)
        & _shift2d(mask, 0, -1)
        & mask
    )
    return mask & ~inner


def mask_centroid(mask, valid, depth):
    """Centroid of the mask's valid-depth pixels + depth at the centroid.

    Returns (u_c, v_c, z_c, valid_count). Background pixels have no depth and
    cannot be back-projected, so the centroid runs over valid pixels only; the
    depth sample falls back to the mean of a 3x3 window when the rounded
    centroid pixel itself is invalid.
    """
    rows, cols = np.nonzero(np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        return None
    sel = np.asarray(valid, dtype=bool)[rows, cols]
    rows, cols = rows[sel], cols[sel]
    if len(rows) < MIN_VALID_PX:
        return None
    u_c, v_c = float(cols.mean()), float(rows.mean())
    ui, vi = round(u_c), round(v_c)
    z_c = float(np.asarray(depth, dtype=float)[vi, ui])
    if not np.isfinite(z_c):
        window = np.asarray(depth, dtype=float)[max(0, vi - 1) : vi + 2, max(0, ui - 1) : ui + 2]
        finite = window[np.isfinite(window)]
        if len(finite) == 0:
            return None
        z_c = float(finite.mean())
    return u_c, v_c, z_c, len(rows)


# ------------------------------------------------------- candidate analysis
def analyze_pose(data, pose_index):
    """Candidate masks of one pose: geometry only, no truth labels.

    A candidate needs >= MIN_VALID_PX valid-depth pixels and a back-projected
    height above ground of at least HEIGHT_GATE_M (landmarks stand on the
    ground; ground-only masks never leave z = 0). Identity = nearest landmark
    axis to the back-projected centroid; margin = distance to the second
    nearest axis. IoU against the truth label map is EVALUATION ONLY.
    """
    rot = data["pose_rotation"][pose_index]
    trans = data["pose_translation"][pose_index]
    depth = data["depth_m"][pose_index]
    valid = data["valid"][pose_index]
    label = data["landmark_label"][pose_index]
    landmarks = data["landmark_xy"]
    mask_indices = np.flatnonzero(data["mask_pose"] == pose_index)
    candidates = []
    for k in mask_indices:
        mask = data["model_mask"][k]
        centroid = mask_centroid(mask, valid, depth)
        if centroid is None:
            continue
        u_c, v_c, z_c, valid_px = centroid
        rows, cols = np.nonzero(mask)
        height_sel = valid[rows, cols]
        if not height_sel.any():
            continue
        pixels = np.column_stack(
            [cols[height_sel], rows[height_sel], np.ones(int(height_sel.sum()))]
        ).astype(float)
        rays = (np.linalg.inv(K_INTRINSIC) @ pixels.T).T
        depths = depth[rows[height_sel], cols[height_sel]]
        world = (rot.T @ ((rays * depths[:, None]) - trans).T).T
        h_max = float(world[:, 2].max())
        if h_max < HEIGHT_GATE_M:
            continue
        _, _, world_c = mask_observation(u_c, v_c, z_c, rot, trans)
        dist = np.linalg.norm(world_c[None, :2] - landmarks, axis=1)
        order = np.argsort(dist)
        nn_identity = int(order[0])
        margin = float(dist[order[1]] - dist[order[0]])
        labels = label[rows, cols]
        dominant = int(np.bincount(labels).argmax())
        iou = 0.0
        if dominant > 0:
            inter = int((labels == dominant).sum())
            union = int(len(rows) + (label == dominant).sum() - inter)
            iou = inter / union if union else 0.0
        candidates.append(
            {
                "mask_index": int(k),
                "u_c": u_c,
                "v_c": v_c,
                "z_c": z_c,
                "valid_px": valid_px,
                "world_xy": world_c[:2].copy(),
                "axis_dist_m": dist.copy(),
                "nn_identity": nn_identity,
                "min_dist_m": float(dist[nn_identity]),
                "margin_m": margin,
                "truth_label": dominant,
                "iou": iou,
            }
        )
    chosen = []
    for i in range(len(landmarks)):
        pool = [c for c in candidates if c["nn_identity"] == i]
        if pool:
            chosen.append(min(pool, key=lambda c: c["min_dist_m"]))
        else:
            chosen.append(None)
    return candidates, chosen


# ----------------------------------------------------- group observations
def build_group_observations(data, pose_index, pose_analysis):
    """Deterministic (range, bearing) per group and landmark, before noise."""
    rot = data["pose_rotation"][pose_index]
    trans = data["pose_translation"][pose_index]
    label = data["landmark_label"][pose_index]
    landmarks = data["landmark_xy"]
    sensor = camera_ground_pose(rot, trans)
    chosen = pose_analysis[1]
    n_land = len(landmarks)
    obs = {group: np.full((n_land, 2), np.nan) for group in GROUP_NAMES}
    range_bias = np.full(n_land, np.nan)
    axis_range = np.full(n_land, np.nan)
    for i in range(n_land):
        axis_range[i], obs["baseline"][i, 1] = bearing_reading(landmarks[i], sensor)
        obs["baseline"][i, 0] = axis_range[i]
        truth_mask = label == (i + 1)
        centroid = mask_centroid(truth_mask, data["valid"][pose_index], data["depth_m"][pose_index])
        if centroid is not None:
            u_c, v_c, z_c, _ = centroid
            r, beta, _ = mask_observation(u_c, v_c, z_c, rot, trans)
            obs["truth_mask"][i] = (r, beta)
            range_bias[i] = r - axis_range[i]
        for k_px in ERODE_DILATE_PX:
            eroded = mask_centroid(
                erode_mask(truth_mask, k_px), data["valid"][pose_index], data["depth_m"][pose_index]
            )
            if eroded is not None:
                obs[f"erode{k_px}"][i] = mask_observation(
                    eroded[0], eroded[1], eroded[2], rot, trans
                )[:2]
            dilated = mask_centroid(
                dilate_mask(truth_mask, k_px),
                data["valid"][pose_index],
                data["depth_m"][pose_index],
            )
            if dilated is not None:
                obs[f"dilate{k_px}"][i] = mask_observation(
                    dilated[0], dilated[1], dilated[2], rot, trans
                )[:2]
        pick = chosen[i]
        if pick is not None:
            r, beta, _ = mask_observation(pick["u_c"], pick["v_c"], pick["z_c"], rot, trans)
            obs["model_mask"][i] = (r, beta)
    # model_identity: the observation the full pipeline offers for landmark i
    # is the chosen mask whose model NN identity is i (by the greedy pick this
    # is chosen[i]; the indirection keeps the pairing model-driven).
    by_identity = {}
    for i, pick in enumerate(chosen):
        if pick is not None:
            by_identity.setdefault(pick["nn_identity"], obs["model_mask"][i])
    for i in range(n_land):
        if i in by_identity:
            obs["model_identity"][i] = by_identity[i]
    obs["mismatch"] = obs["baseline"].copy()  # permutation happens at solve time
    return obs, range_bias, axis_range


def solve_camera_pose(readings, landmarks):
    """Lesson-18 Procrustes (reused verbatim) -> camera ground pose.

    solve_pose returns the vehicle pose = sensor o inverse(SENSOR_IN_BODY);
    composing the installation back recovers the sensor (camera) pose exactly.
    """
    vehicle = solve_pose(readings, landmarks)
    return compose(vehicle, SENSOR_IN_BODY)


def solve_with_available(readings, landmarks):
    """Solve on the finite readings only; returns (pose, n_used) or (None, 0)."""
    readings = np.asarray(readings, dtype=float)
    landmarks = np.asarray(landmarks, dtype=float)
    keep = np.isfinite(readings).all(axis=1) & (readings[:, 0] > 0)
    if keep.sum() < 2:
        return None, int(keep.sum())
    pose = solve_camera_pose(readings[keep], landmarks[keep])
    return pose, int(keep.sum())


# ---------------------------------------------------------------- experiment
def load_input(npz_path):
    """Load and validate the bench archive (contract + pose cross-check)."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"input npz missing: {path}")
    with np.load(path, allow_pickle=False) as npz:
        missing = REQUIRED_INPUT_KEYS - set(npz.files)
        if missing:
            raise ValueError(f"input npz missing keys: {sorted(missing)}")
        data = {key: npz[key].copy() for key in npz.files}
    meta = json.loads(data["meta_json"].item())
    if not np.allclose(np.asarray(meta["intrinsic"], dtype=float), K_INTRINSIC, atol=1e-12):
        raise ValueError("input intrinsic differs from the lesson-22 K")
    n_poses = data["pose_rotation"].shape[0]
    for key, other in (("pose_eye_m", "pose_target_m"),):
        if len(meta.get(key, ())) != n_poses:
            raise ValueError(f"meta {key} does not match {n_poses} poses")
    for p in range(n_poses):
        rot, trans = look_at(
            np.asarray(meta["pose_eye_m"][p], dtype=float),
            np.asarray(meta["pose_target_m"][p], dtype=float),
        )
        if not np.allclose(rot, data["pose_rotation"][p], atol=1e-9):
            raise ValueError(f"pose {p} rotation disagrees with look_at(eye, target)")
        if not np.allclose(trans, data["pose_translation"][p], atol=1e-9):
            raise ValueError(f"pose {p} translation disagrees with look_at(eye, target)")
    if not np.allclose(
        data["landmark_xy"], np.asarray(meta["landmark_xy"], dtype=float), atol=1e-12
    ):
        raise ValueError("landmark_xy disagrees with meta")
    if data["model_mask"].size == 0:
        raise ValueError("bench archive contains no model masks")
    return data, meta


def run_experiment(input_npz, output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two noise repetitions")
    data, bench_meta = load_input(input_npz)
    landmarks = data["landmark_xy"]
    n_poses = data["pose_rotation"].shape[0]
    n_land = len(landmarks)
    truth_poses = np.array(
        [
            camera_ground_pose(data["pose_rotation"][p], data["pose_translation"][p])
            for p in range(n_poses)
        ]
    )

    analyses = [analyze_pose(data, p) for p in range(n_poses)]
    group_obs = []
    range_bias = np.full((n_poses, n_land), np.nan)
    axis_range = np.full((n_poses, n_land), np.nan)
    for p in range(n_poses):
        obs, bias, axis_r = build_group_observations(data, p, analyses[p])
        group_obs.append(obs)
        range_bias[p] = bias
        axis_range[p] = axis_r
    obs_range = np.stack([group_obs[p][name][:, 0] for p in range(n_poses) for name in GROUP_NAMES])
    obs_range = obs_range.reshape(n_poses, len(GROUP_NAMES), n_land).transpose(1, 0, 2)
    obs_bearing = np.stack(
        [group_obs[p][name][:, 1] for p in range(n_poses) for name in GROUP_NAMES]
    )
    obs_bearing = obs_bearing.reshape(n_poses, len(GROUP_NAMES), n_land).transpose(1, 0, 2)

    # ---- localization over seeds (common random numbers across groups)
    err_vec = np.full((len(GROUP_NAMES), n_poses, runs, 2), np.nan)
    err_norm = np.full((len(GROUP_NAMES), n_poses, runs), np.nan)
    heading_err = np.full((len(GROUP_NAMES), n_poses, runs), np.nan)
    reduced_solves = np.zeros((len(GROUP_NAMES), n_poses), dtype=int)
    for p in range(n_poses):
        for s in range(runs):
            rng = np.random.default_rng([seed, s, p])
            noise_r = rng.normal(0.0, RANGE_STD_M, n_land)
            noise_b = rng.normal(0.0, BEARING_STD_RAD, n_land)
            for gi, group in enumerate(GROUP_NAMES):
                readings = group_obs[p][group] + np.column_stack([noise_r, noise_b])
                if group == "mismatch":
                    readings = np.roll(readings, 1, axis=0)
                pose, used = solve_with_available(readings, landmarks)
                if pose is None:
                    continue  # fewer than two usable readings: no solve this seed
                if used < n_land:
                    reduced_solves[gi, p] += 1
                err_vec[gi, p, s] = pose[:2] - truth_poses[p][:2]
                err_norm[gi, p, s] = float(np.linalg.norm(err_vec[gi, p, s]))
                heading_err[gi, p, s] = float(wrap_angle(pose[2] - truth_poses[p][2]))

    # ---- mechanism: pixel shift -> bearing (exact ray) and through the chain
    shift_px = np.asarray(SHIFT_PX, dtype=float)
    shift_bearing = np.full((n_poses, n_land, len(SHIFT_PX)), np.nan)
    shift_ratio_pred = np.full((n_poses, n_land), np.nan)
    for p in range(n_poses):
        rot = data["pose_rotation"][p]
        heading = truth_poses[p][2]
        for i in range(n_land):
            pick = analyses[p][1][i]
            if pick is None:
                continue
            shift_ratio_pred[p, i] = pitch_factor(pick["v_c"], rot)
            for si, dpx in enumerate(SHIFT_PX):
                after = pixel_bearing(pick["u_c"] + dpx, pick["v_c"], rot, heading)
                before = pixel_bearing(pick["u_c"], pick["v_c"], rot, heading)
                shift_bearing[p, i, si] = float(after - before)
    # full-chain shift: the truth masks move by +dpx (depth resampled at the
    # shifted centroid) and the chain runs with truth identity and the same
    # noise streams; a COMMON bearing bias is absorbed by the fitted heading.
    chain_heading = np.zeros(len(SHIFT_PX))
    chain_pos = np.zeros(len(SHIFT_PX))
    for si, dpx in enumerate(SHIFT_PX):
        errs, heads = [], []
        for p in range(n_poses):
            label = data["landmark_label"][p]
            shifted_obs = np.full((n_land, 2), np.nan)
            for i in range(n_land):
                centroid = mask_centroid(
                    shift_mask(label == (i + 1), int(dpx)),
                    data["valid"][p],
                    data["depth_m"][p],
                )
                if centroid is None:
                    continue
                r_shift, beta_shift, _ = mask_observation(
                    centroid[0],
                    centroid[1],
                    centroid[2],
                    data["pose_rotation"][p],
                    data["pose_translation"][p],
                )
                shifted_obs[i] = (r_shift, beta_shift)
            for s in range(runs):
                rng = np.random.default_rng([seed, s, p])
                noise = np.column_stack(
                    [rng.normal(0.0, RANGE_STD_M, n_land), rng.normal(0.0, BEARING_STD_RAD, n_land)]
                )
                pose, _ = solve_with_available(shifted_obs + noise, landmarks)
                if pose is None:
                    continue
                heads.append(float(wrap_angle(pose[2] - truth_poses[p][2])))
                errs.append(float(np.linalg.norm(pose[:2] - truth_poses[p][:2])))
        chain_heading[si] = float(np.mean(heads)) if heads else np.nan
        chain_pos[si] = float(np.mean(errs)) if errs else np.nan

    # ---- identity evaluation (truth labels, evaluation only)
    identity_ok = np.zeros((n_poses, n_land), dtype=bool)
    chosen_dmin = np.full((n_poses, n_land), np.nan)
    chosen_margin = np.full((n_poses, n_land), np.nan)
    chosen_iou = np.full((n_poses, n_land), np.nan)
    chosen_area = np.zeros((n_poses, n_land), dtype=int)
    chosen_mask = np.zeros((n_poses, n_land, *data["depth_m"].shape[1:]), dtype=bool)
    candidate_count = np.zeros(n_poses, dtype=int)
    for p in range(n_poses):
        candidates, chosen = analyses[p]
        candidate_count[p] = len(candidates)
        for i, pick in enumerate(chosen):
            if pick is None:
                continue
            identity_ok[p, i] = pick["nn_identity"] == i and pick["truth_label"] == i + 1
            chosen_dmin[p, i] = pick["min_dist_m"]
            chosen_margin[p, i] = pick["margin_m"]
            chosen_iou[p, i] = pick["iou"]
            chosen_area[p, i] = int(data["model_mask"][pick["mask_index"]].sum())
            chosen_mask[p, i] = data["model_mask"][pick["mask_index"]]

    def group_stats(name):
        gi = GROUP_NAMES.index(name)
        values = err_norm[gi].ravel()
        vectors = err_vec[gi].reshape(-1, 2)
        heads = heading_err[gi].ravel()
        return {
            "pos_mean_m": float(np.nanmean(values)),
            "pos_std_m": float(np.nanstd(values, ddof=1)),
            "pos_median_m": float(np.nanmedian(values)),
            "pos_max_m": float(np.nanmax(values)),
            "signed_mean_m": [float(np.nanmean(vectors[:, 0])), float(np.nanmean(vectors[:, 1]))],
            "heading_mean_deg": float(np.degrees(np.nanmean(heads))),
            "heading_std_deg": float(np.degrees(np.nanstd(heads, ddof=1))),
            "per_pose_mean_m": [float(np.nanmean(err_norm[gi, p])) for p in range(n_poses)],
            "reduced_solves": int(reduced_solves[gi].sum()),
        }

    groups = {name: group_stats(name) for name in GROUP_NAMES}
    finite_bias = range_bias[np.isfinite(range_bias)]
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "input_npz": str(Path(input_npz)),
        "input_npz_sha256": digest(input_npz),
        "model": {
            "name": bench_meta.get("model"),
            "package_source": bench_meta.get("model_package_source"),
            "weights_source": bench_meta.get("model_weights_source"),
            "checkpoint": bench_meta.get("checkpoint"),
            "checkpoint_sha256": bench_meta.get("checkpoint_sha256"),
            "amg_params": bench_meta.get("amg_params"),
            "device": bench_meta.get("device"),
            "inference_runtime_s_per_pose": bench_meta.get("inference_runtime_s_per_pose"),
        },
        "scene": {
            "canvas_px": bench_meta.get("canvas_px"),
            "intrinsic": K_INTRINSIC.tolist(),
            "focal_px": float(K_INTRINSIC[0, 0]),
            "landmark_xy": landmarks.tolist(),
            "landmark_radius_m": float(data["landmark_radius_m"]),
            "landmark_height_m": float(data["landmark_height_m"]),
            "pose_eye_m": bench_meta.get("pose_eye_m"),
            "pose_target_m": bench_meta.get("pose_target_m"),
            "fog_tau_m": bench_meta.get("fog_tau_m"),
        },
        "noise": {"range_std_m": RANGE_STD_M, "bearing_std_rad": BEARING_STD_RAD},
        "runs_per_group": runs,
        "base_seed": seed,
        "n_poses": n_poses,
        "selection_rule": {
            "height_gate_m": HEIGHT_GATE_M,
            "min_valid_px": MIN_VALID_PX,
            "identity": "nearest landmark axis to the back-projected mask centroid",
            "pick": "per landmark, the candidate with the smallest axis distance",
            "note": (
                "scene-structure prior (landmarks stand on the ground); no truth "
                "labels enter selection, assignment or the solve - truth labels "
                "are used only to evaluate identity accuracy and IoU"
            ),
        },
        "masks": {
            "total": int(data["mask_pose"].size),
            "candidates_per_pose": candidate_count.tolist(),
            "chosen_per_pose": [
                [
                    int(analyses[p][1][i]["mask_index"]) if analyses[p][1][i] is not None else -1
                    for i in range(n_land)
                ]
                for p in range(n_poses)
            ],
            "missed_landmarks": int(sum(c is None for row in analyses for c in row[1])),
        },
        "identity": {
            "correct": int(identity_ok.sum()),
            "total": int(identity_ok.size),
            "accuracy": float(identity_ok.mean()),
            "mean_axis_dist_m": float(np.nanmean(chosen_dmin)),
            "max_axis_dist_m": float(np.nanmax(chosen_dmin)),
            "mean_margin_m": float(np.nanmean(chosen_margin)),
            "min_margin_m": float(np.nanmin(chosen_margin)),
        },
        "iou": {
            "per_pose_landmark": chosen_iou.tolist(),
            "mean": float(np.nanmean(chosen_iou)),
            "min": float(np.nanmin(chosen_iou)),
        },
        "groups": groups,
        "surface_range_bias_m": {
            "per_pose_landmark": range_bias.tolist(),
            "mean": float(finite_bias.mean()) if finite_bias.size else np.nan,
            "note": "mask centroid observes the near cylinder SURFACE: range is ~ -radius vs the axis",
        },
        "mechanism": {
            "shift_px": SHIFT_PX,
            "shift_bearing_rad": shift_bearing.tolist(),
            "shift_pred_ratio": shift_ratio_pred.tolist(),
            "shift_ratio_to_px_over_f": [
                float(
                    np.nanmean(
                        -shift_bearing[:, :, si]
                        / (shift_px[si] / K_INTRINSIC[0, 0])
                        / shift_ratio_pred
                    )
                )
                for si in range(len(SHIFT_PX))
            ],
            "shift_sign_note": (
                "bearings are world atan2(dy, dx), u grows along camera right: "
                "a rightward centroid shift DECREASES the bearing; ratios quoted "
                "as -d_beta / (d_px/f * pitch factor), |ratio| ~ 1 verifies the law"
            ),
            "chain_shift": {
                "heading_mean_rad": chain_heading.tolist(),
                "pos_mean_m": chain_pos.tolist(),
            },
        },
        "trajectories_sha256": "",
        "source_sha256": {"experiments/visual_grounding.py": digest(Path(__file__).resolve())},
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            (
                "Identity comes from geometry (back-projected centroid + nearest axis); "
                "the model contributes masks only - no learned classification is evaluated"
            ),
            (
                "Candidate selection uses the scene-structure prior that landmarks stand "
                "on the ground; class-level detection ('which mask is a landmark at all') "
                "remains assumed"
            ),
            (
                "Depth is the rendered analytic scene depth, not a learned depth map; "
                "identity errors induced by depth noise are out of scope"
            ),
            (
                "The mask centroid observes the near cylinder surface, giving a "
                "systematic -radius range bias vs the lesson-18 axis observation; "
                "no surface-vs-axis correction is applied"
            ),
            (
                "Single scene, three cylinders, four poses; forced mismatch bounds the "
                "confusion shape but real-scene confusion rates are not measured"
            ),
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    archive = {
        "rgb": data["rgb"],
        "depth_m": data["depth_m"].astype(np.float32),
        "landmark_label": data["landmark_label"],
        "valid": data["valid"],
        "pose_rotation": data["pose_rotation"],
        "pose_translation": data["pose_translation"],
        "landmark_xy": landmarks,
        "chosen_mask": chosen_mask,
        "chosen_area": chosen_area,
        "chosen_iou": chosen_iou,
        "chosen_axis_dist_m": chosen_dmin,
        "chosen_margin_m": chosen_margin,
        "identity_ok": identity_ok,
        "candidate_count": candidate_count,
        "group_names": np.array(GROUP_NAMES),
        "obs_range_m": obs_range,
        "obs_bearing_rad": obs_bearing,
        "truth_poses": truth_poses,
        "err_vec_m": err_vec,
        "err_norm_m": err_norm,
        "heading_err_rad": heading_err,
        "range_bias_m": range_bias,
        "axis_range_m": axis_range,
        "shift_px": shift_px,
        "shift_bearing_rad": shift_bearing,
        "shift_pred_ratio": shift_ratio_pred,
        "chain_shift_heading_rad": chain_heading,
        "chain_shift_pos_m": chain_pos,
        "meta_json": data["meta_json"],
    }
    np.savez_compressed(output / "trajectories.npz", **archive)
    summary["trajectories_sha256"] = digest(output / "trajectories.npz")
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(archive, summary, output)
    return summary


def make_plot(archive, summary, output):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    groups = summary["groups"]
    n_poses = summary["n_poses"]
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    fig.suptitle("第二十七课 视觉基础模型给地标“发身份”：掩码 → 最近邻身份 → Procrustes 定位链")
    ax_img = fig.add_subplot(2, 3, 1)
    ax_iou = fig.add_subplot(2, 3, 2)
    ax_pos = fig.add_subplot(2, 3, 3)
    ax_head = fig.add_subplot(2, 3, 4)
    ax_law = fig.add_subplot(2, 3, 5)
    ax_chain = fig.add_subplot(2, 3, 6)

    pose = 1 if n_poses > 1 else 0
    rgb = archive["rgb"][pose]
    ax_img.imshow(rgb)
    for i in range(len(archive["landmark_xy"])):
        edge = mask_boundary(archive["landmark_label"][pose] == (i + 1))
        overlay = np.zeros((*edge.shape, 4))
        overlay[edge] = (0.1, 0.7, 0.2, 0.9)
        ax_img.imshow(overlay)
        edge = mask_boundary(archive["chosen_mask"][pose, i])
        overlay = np.zeros((*edge.shape, 4))
        overlay[edge] = (0.9, 0.2, 0.15, 0.9)
        ax_img.imshow(overlay)
        rows, cols = np.nonzero(archive["chosen_mask"][pose, i])
        ok = archive["identity_ok"][pose, i]
        ax_img.text(
            cols.mean(),
            rows.mean() - 14,
            f"L{i + 1}{'对' if ok else '错'}",
            color="white",
            fontsize=9,
            ha="center",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5},
        )
    ax_img.set(
        xlabel="u / px",
        ylabel="v / px",
        title=f"位姿 {pose}：真值边缘（绿）vs 模型掩码（红），身份 {'全对' if summary['identity']['accuracy'] == 1.0 else '含错'}",
    )
    ax_img.set_xticks([]), ax_img.set_yticks([])

    iou = np.asarray(summary["iou"]["per_pose_landmark"])
    xs = np.arange(iou.size)
    ax_iou.bar(xs, iou.ravel(), color="#2563eb")
    ax_iou.set_ylim(0, 1.05)
    ax_iou.axhline(summary["iou"]["mean"], color="#0f172a", ls="--", lw=1.0)
    ax_iou.set(
        xlabel="位姿 × 地标（按位姿分块）",
        ylabel="IoU（模型掩码 vs 真值区域）",
        title=f"掩码 IoU：均值 {summary['iou']['mean']:.2f}，最小 {summary['iou']['min']:.2f}",
    )

    names = list(GROUP_NAMES)
    means = [groups[name]["pos_mean_m"] * 100 for name in names]
    stds = [groups[name]["pos_std_m"] * 100 for name in names]
    colors = [
        "#64748b",
        "#2563eb",
        "#60a5fa",
        "#3b82f6",
        "#93c5fd",
        "#bfdbfe",
        "#ea580c",
        "#f97316",
        "#dc2626",
    ]
    bars = ax_pos.bar(names, means, yerr=stds, capsize=3, color=colors)
    for bar, mean in zip(bars, means):
        ax_pos.text(
            bar.get_x() + bar.get_width() / 2,
            mean * 1.35,
            f"{mean:.1f}",
            ha="center",
            fontsize=7,
        )
    ax_pos.set_yscale("log")
    ax_pos.set_ylim(0.1, max(means) * 6)
    ax_pos.tick_params(axis="x", rotation=40, labelsize=7)
    ax_pos.set(
        ylabel="定位位置误差 / cm（4 位姿 × 20 种子，均值±1σ）",
        title="身份 vs 掩码质量 vs 强制错配（对数轴）",
    )

    head_means = [abs(groups[name]["heading_mean_deg"]) for name in names]
    bars = ax_head.bar(names, head_means, color=colors)
    for bar, value in zip(bars, head_means):
        ax_head.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.2,
            f"{value:.2f}",
            ha="center",
            fontsize=7,
        )
    ax_head.set_yscale("log")
    ax_head.set_ylim(1e-3, max(head_means) * 5)
    ax_head.tick_params(axis="x", rotation=40, labelsize=7)
    ax_head.set(
        ylabel="|朝向误差均值| / °（20 种子）",
        title="共同质心偏置进入朝向；错配爆炸在朝向上同样显现",
    )

    shift_px = archive["shift_px"]
    focal = summary["scene"]["focal_px"]
    bearing = archive["shift_bearing_rad"]
    measured = np.degrees(np.nanmean(-bearing, axis=(0, 1))) * 1000  # mrad, |Δβ|
    ax_law.plot(shift_px, measured, "-o", ms=4, color="#dc2626", label="实测 |Δβ|（精确射线几何）")
    ax_law.plot(
        shift_px, shift_px / focal * 1000, "--", color="#0f172a", label="δpx / f（水平相机定律）"
    )
    pred = np.nanmean(archive["shift_pred_ratio"])
    ax_law.plot(
        shift_px,
        shift_px / focal * 1000 * pred,
        ":",
        color="#2563eb",
        label=f"δpx/f × 俯仰因子 {pred:.2f}",
    )
    ax_law.set(
        xlabel="掩码质心横移 δ / px（图内向右；世界方位角相应减小）",
        ylabel="方位角偏移 |Δβ| / mrad",
        title="机制核对：|Δβ| = δpx/f（俯仰相机按因子放大）",
    )
    ax_law.legend(fontsize=8)

    ax_chain.plot(
        shift_px,
        np.degrees(archive["chain_shift_heading_rad"]) * 1000,
        "-o",
        ms=4,
        color="#9333ea",
        label="整链朝向偏差",
    )
    ax_chain.plot(
        shift_px,
        archive["chain_shift_pos_m"] * 100,
        "-s",
        ms=4,
        color="#ea580c",
        label="整链位置偏差",
    )
    ax_chain.plot(
        shift_px,
        shift_px / focal * 1000 * pred,
        "--",
        color="#0f172a",
        lw=1.0,
        label="δpx/f × 俯仰因子（参考）",
    )
    ax_chain.set(
        xlabel="真值掩码整体横移 δ / px（深度按新质心重采样）",
        ylabel="整链误差（朝向 mrad / 位置 cm）",
        title="共同横移几乎全部被朝向吸收，位置几乎不动",
    )
    ax_chain.legend(fontsize=8)
    for ax in (ax_iou, ax_pos, ax_head, ax_law, ax_chain):
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="bench grounding_masks.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for noise statistics")
    report = run_experiment(args.input, args.output, runs=args.runs, seed=args.seed)
    identity = report["identity"]
    groups = report["groups"]
    print(
        f"identity {identity['correct']}/{identity['total']} correct "
        f"(mean axis dist {identity['mean_axis_dist_m'] * 100:.1f} cm, "
        f"min margin {identity['min_margin_m'] * 100:.0f} cm); "
        f"mask IoU mean {report['iou']['mean']:.2f}"
    )
    for name in ("baseline", "truth_mask", "model_mask", "model_identity", "mismatch"):
        stat = groups[name]
        unit = "m" if stat["pos_mean_m"] >= 1 else "cm"
        value = stat["pos_mean_m"] if unit == "m" else stat["pos_mean_m"] * 100
        print(
            f"{name:>14}: pos {value:8.2f} {unit} (std {stat['pos_std_m'] * (1 if unit == 'm' else 100):.2f}), "
            f"heading {stat['heading_mean_deg']:+.3f} deg"
        )
    print(
        f"surface range bias {report['surface_range_bias_m']['mean'] * 100:.1f} cm; "
        "shift ratio to px/f: "
        f"{[format(r, '.3f') for r in report['mechanism']['shift_ratio_to_px_over_f']]}"
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
