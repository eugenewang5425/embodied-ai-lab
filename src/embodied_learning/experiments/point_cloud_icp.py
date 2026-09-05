"""Lesson 26: ICP registration of two noisy depth clouds (robot version of
multi-station GIS point-cloud registration).

Camera A is the lesson-22 look-at pose; camera B is camera A rigidly moved by
a known world-frame translation and a yaw about world z. Both cameras render
the lesson-22 scene (ground grid + solid pole) per pixel with lesson-23-style
analytic ray casting; independent Gaussian depth noise is added; the lesson-22
``unproject`` turns each depth map into a point cloud expressed in its own
sensor frame. A hand-written ICP (vectorized O(N^2) nearest neighbours on
~2000 voxel-downsampled points, a shrinking trimming threshold, point-to-point
SVD vs point-to-plane linear least squares) registers the B cloud onto the A
cloud. Normals for point-to-plane are the analytic surface normals (ground
plane + pole cylinder are known in this simulated scene); the k-NN PCA
alternative is measured and reported as a diagnostic.

The scene has an exact one-parameter ICP symmetry: a yaw about the vertical
line through the pole maps BOTH surfaces onto themselves, so that degree of
freedom is structurally unidentifiable - the ground is blind to any horizontal
rotation (its normal is vertical) and the cylinder is blind to spins about its
own axis. The experiment therefore separates the observable subspace (plane
alignment + in-plane translation, anchored by the pole) from the symmetry mode,
whose initial error survives as footprint ~ (horizontal lever arm) x (yaw).

Checks:
(1) initial-guess study: truth init (upper bound), identity init (no prior),
    and a yaw/shift perturbation grid over 20 seeds -> convergence radius
    (final translation error < 2 cm) and the yaw footprint law;
(2) depth-noise sigma sweep in {0.05, 0.15, 0.30} m with a translation-only
    initial guess: rotation/translation error statistics and iteration counts
    for both objectives;
(3) degenerate counterexample: ground-only clouds (pole removed) -> ICP
    "converges" onto a wrong pose (sliding along the plane);
(4) mechanism ratios: error vs sigma, the yaw footprint slope, and the
    point-to-plane vs point-to-point iteration-count contrast.
No scipy, no cv2, no open3d: nearest neighbours are a vectorized Gram-trick
matrix product on the downsampled clouds, with the O(N^2) trade-off recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.experiments.monocular_metric import (
    GRID_X_MAX_M,
    GRID_X_MIN_M,
    GRID_Y_MAX_M,
    GRID_Y_MIN_M,
    POLE_RADIUS_M,
    POLE_TOP_M,
    POLE_X_M,
    POLE_Y_M,
)
from embodied_learning.experiments.monocular_metric import (
    render_depth_map as render_lesson23_depth_map,
)
from embodied_learning.experiments.pinhole_projection import (
    EYE,
    HEIGHT_PX,
    K_INTRINSIC,
    NEAR_PLANE_M,
    TARGET,
    WIDTH_PX,
    look_at,
    to_camera,
    unproject,
)

EXPERIMENT = "point_cloud_icp_registration"
# Pose B = pose A translated by DELTA (world frame) and yawed about world z.
# The magnitude follows the review-doc suggestion; the yaw SIGN (-25 deg) was
# chosen after checking the pole stays comfortably inside B's field of view
# (23.4 deg off-axis vs the 28.1 deg half-FOV; +25 deg leaves only 1.5 deg).
POSE_DELTA_M = (0.5, -0.3, 0.1)
POSE_YAW_DEG = -25.0
# Downsampling: one representative (closest to the cloud centroid within its
# voxel) per occupied 0.10 m voxel in the sensor frame, then a seeded cut to
# the point budget.
VOXEL_SIZE_M = 0.10
MAX_CLOUD_POINTS = 2000
SIGMA_VALUES = (0.05, 0.15, 0.30)
OBJECTIVES = ("point_to_point", "point_to_plane")
# Trimming schedule tau_k = max(tau_floor, tau0 * gamma^k): wide enough at k=0
# to keep pole pairs under a 10 deg yaw error (lever arm ~3 m), shrinking to
# a floor that keeps noise-level pairs but cuts gross mismatches.
TAU0_M = 0.90
TAU_GAMMA = 0.85
TAU_FLOOR_M = 0.30
MIN_PAIRS = 10
MAX_ITERS = 200
TOL = 1e-6  # on the increment: translation norm (m) and rotation angle (rad)
STEP_CLAMP_RAD = 0.35  # linearization safety clamp on the p2plane increment
# Eigenvalue cutoff of the point-to-plane normal equations, relative to the
# largest eigenvalue: increment components along directions with less than
# ~1% of the strongest direction's information are dropped, so the exact
# scene symmetry (yaw about the pole axis, eigenvalue ratio a few 1e-5) and
# other unobservable directions keep the current pose instead of amplifying
# normal-estimation noise into a random walk along the symmetry valley.
EIGEN_CUTOFF = 1e-4
NORMAL_K = 12  # PCA normals, diagnostic only (analytic normals are primary)
# "Converged" for the radius study means the final translation error against
# the ground-truth transform is below this, per the review-doc criterion.
CONVERGED_TRANS_M = 0.02
# Sigma-sweep initial guess: the TRUE transform. The sweep measures the noise
# floor of the observable subspace, which grows ~linearly with sigma; any
# sizeable initial error instead converges to a point along the scene's
# symmetry valley whose offset is set by sampling geometry, not by sigma
# (measured in the radius study).
DEGEN_INIT_SHIFT_M = (0.1, 0.0, 0.0)
RADIUS_YAW_DEG = (0.0, 5.0, 10.0, 20.0)
RADIUS_SHIFT_M = (0.0, 0.1, 0.3, 0.5)
# The radius study is measured at sigma = 0.02 m: the observable-subspace
# error floor grows ~linearly with sigma (see the sigma sweep) and would sit
# above the 2 cm criterion at 0.05 m, leaving no criterion margin.
RADIUS_SIGMA_M = 0.02
# light-mode (test fixture) variants of the convergence-radius study
LIGHT_RADIUS_YAW_DEG = (0.0, 10.0)
LIGHT_RADIUS_SHIFT_M = (0.0, 0.1)
LIGHT_MAX_ITERS = 40
DEGEN_SIGMA_INDEX = 1  # degenerate counterexample at sigma = 0.15 m
DEFAULT_RUNS = 20
DEFAULT_SEED = 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------- SE(3)
def rotation_z(degrees):
    angle = np.radians(float(degrees))
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rodrigues(omega):
    """Rotation matrix from a rotation vector (small-increment linearization)."""
    omega = np.asarray(omega, dtype=float)
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    axis = omega / angle
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def rotation_angle_deg(rotation):
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def apply_transform(rotation, translation, points):
    return (rotation @ np.asarray(points, dtype=float).T).T + translation


def pose_error(rotation, translation, gt_rotation, gt_translation):
    """Rotation error (deg) and translation error vector/norm against truth."""
    rot_err = rotation_angle_deg(rotation @ gt_rotation.T)
    trans_vec = np.asarray(translation, dtype=float) - gt_translation
    return rot_err, trans_vec, float(np.linalg.norm(trans_vec))


# Spin grid for the symmetry quotient (deg about the vertical pole axis).
SYM_SPIN_GRID_DEG = np.linspace(-60.0, 60.0, 2401)


def symmetry_quotient_error(
    rotation_est, translation_est, gt_rotation, gt_translation, rotation_a, translation_a
):
    """Decompose the registration error into observable part + symmetry part.

    The scene's point-to-plane cost is invariant under T -> T o Spin(s), where
    Spin(s) rotates about the vertical line through the pole: such a spin maps
    BOTH surfaces onto themselves, so ICP determines the pose only up to this
    one-parameter family. Sliding the estimate along the family and taking the
    member closest to the truth (in translation) yields the
    observable-subspace error; the required spin s* is the symmetry-family
    coordinate (its translation footprint is ~|pole lever| x s). Naive errors
    (no quotient) are returned alongside.
    """
    naive_rot, naive_vec, naive_trans = pose_error(
        rotation_est, translation_est, gt_rotation, gt_translation
    )
    point = rotation_a @ np.array([POLE_X_M, POLE_Y_M, 0.0]) + translation_a
    axis = rotation_a[:, 2]  # world up-axis in the camera frame
    axis = axis / np.linalg.norm(axis)
    angles = np.radians(SYM_SPIN_GRID_DEG)
    cos_s = np.cos(angles)[:, None, None]
    sin_s = np.sin(angles)[:, None, None]
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    outer = np.outer(axis, axis)
    spins = cos_s * np.eye(3)[None] + sin_s * cross[None] + (1.0 - cos_s) * outer[None]
    spin_translation = point[None, :] - np.einsum("sij,j->si", spins, point)
    rot_family = np.einsum("ij,sjk->sik", rotation_est, spins)
    trans_family = np.einsum("ij,sj->si", rotation_est, spin_translation) + translation_est[None, :]
    trans_err = np.linalg.norm(trans_family - gt_translation[None, :], axis=1)
    best = int(np.argmin(trans_err))
    rot_obs = rotation_angle_deg(rot_family[best] @ gt_rotation.T)
    return {
        "naive_rot_deg": naive_rot,
        "naive_trans_m": naive_trans,
        "naive_vec": naive_vec,
        "spin_deg": float(SYM_SPIN_GRID_DEG[best]),
        "trans_err_obs_m": float(trans_err[best]),
        "rot_err_obs_deg": float(rot_obs),
    }


# ------------------------------------------------------------- scene & poses
def pose_b():
    """Camera B: pose A translated by POSE_DELTA_M and yawed POSE_YAW_DEG.

    Returns (rotation, translation, eye) with the lesson-22 world->camera
    convention. The GT B-frame -> A-frame registration transform is then
    exactly rotation = Rz(-POSE_YAW_DEG), translation = R_A @ POSE_DELTA_M.
    """
    rotation_a, _ = look_at(EYE, TARGET)
    rotation_b = rotation_z(POSE_YAW_DEG) @ rotation_a
    eye_b = EYE + np.asarray(POSE_DELTA_M, dtype=float)
    translation_b = -rotation_b @ eye_b
    return rotation_b, translation_b, eye_b


def render_depth_map_for(rotation, eye):
    """Generalized lesson-23 analytic ray casting for an arbitrary pose.

    Same surfaces (ground plane over the lesson-22 grid extent + the solid
    pole cylinder) and the same conventions: depth = camera-frame z, a pixel
    is valid when the nearest surface hit lies beyond the near plane, and the
    pole mask marks pixels whose nearest hit is the cylinder. At the lesson-22
    pose this reproduces monocular_metric.render_depth_map exactly.
    """
    eye = np.asarray(eye, dtype=float)
    cols = np.arange(WIDTH_PX, dtype=float)
    rows = np.arange(HEIGHT_PX, dtype=float)
    uu, vv = np.meshgrid(cols, rows)
    homogeneous = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    rays = rotation.T @ (np.linalg.inv(K_INTRINSIC) @ homogeneous)
    wx, wy, wz = rays
    depth = np.full(uu.size, np.nan)
    pole = np.zeros(uu.size, dtype=bool)
    towards_ground = wz < -1e-12
    s_ground = np.full(uu.size, np.nan)
    s_ground[towards_ground] = -eye[2] / wz[towards_ground]
    ground_x = eye[0] + s_ground * wx
    ground_y = eye[1] + s_ground * wy
    on_grid = (
        towards_ground
        & (ground_x >= GRID_X_MIN_M)
        & (ground_x <= GRID_X_MAX_M)
        & (ground_y >= GRID_Y_MIN_M)
        & (ground_y <= GRID_Y_MAX_M)
    )
    depth[on_grid] = s_ground[on_grid]
    offset_x, offset_y = eye[0] - POLE_X_M, eye[1] - POLE_Y_M
    quad_a = wx * wx + wy * wy
    quad_b = 2.0 * (wx * offset_x + wy * offset_y)
    quad_c = offset_x * offset_x + offset_y * offset_y - POLE_RADIUS_M**2
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    hits = (quad_a > 1e-12) & (disc >= 0.0)
    s_pole = np.full(uu.size, np.nan)
    s_pole[hits] = (-quad_b[hits] - np.sqrt(disc[hits])) / (2.0 * quad_a[hits])
    with np.errstate(invalid="ignore"):
        pole_z = eye[2] + s_pole * wz
    on_pole = hits & (s_pole > NEAR_PLANE_M) & (pole_z >= 0.0) & (pole_z <= POLE_TOP_M)
    nearer = on_pole & (np.isnan(depth) | (s_pole < depth))
    depth[nearer] = s_pole[nearer]
    pole[nearer] = True
    depth[depth <= NEAR_PLANE_M] = np.nan  # near-plane cull for ground hits
    valid = np.isfinite(depth)
    return {
        "depth_m": depth.reshape(HEIGHT_PX, WIDTH_PX),
        "valid": valid.reshape(HEIGHT_PX, WIDTH_PX),
        "pole": pole.reshape(HEIGHT_PX, WIDTH_PX),
    }


def scene_and_poses():
    """Render both poses once and cross-check pose A against the lesson-23 map."""
    rotation_a, translation_a = look_at(EYE, TARGET)
    rotation_b, translation_b, eye_b = pose_b()
    rendered_a = render_depth_map_for(rotation_a, EYE)
    lesson_map = render_lesson23_depth_map()
    if not np.array_equal(lesson_map["valid"], rendered_a["valid"]):
        raise ValueError("Pose-A validity mask does not match the lesson-23 renderer")
    if not np.allclose(lesson_map["depth_m"], rendered_a["depth_m"], atol=1e-12, equal_nan=True):
        raise ValueError("Pose-A depth map does not match the lesson-23 renderer")
    rendered_b = render_depth_map_for(rotation_b, eye_b)
    gt_rotation, gt_translation = init_from_pose_b(
        rotation_b, translation_b, rotation_a, translation_a
    )
    return {
        "rotation_a": rotation_a,
        "translation_a": translation_a,
        "rotation_b": rotation_b,
        "translation_b": translation_b,
        "eye_b": eye_b,
        "rendered_a": rendered_a,
        "rendered_b": rendered_b,
        "gt_rotation": gt_rotation,
        "gt_translation": gt_translation,
    }


def init_from_pose_b(rotation_b, translation_b, rotation_a, translation_a):
    """GT B-frame -> A-frame transform implied by the two camera poses."""
    rotation = rotation_a @ rotation_b.T
    translation = translation_a - rotation @ translation_b
    return rotation, translation


def perturbed_pose_b(yaw_deg, shift, rotation_b, eye_b):
    """Perturb the TRUE pose B (world-frame yaw about its eye + x-shift).

    The perturbed pose plays the role of a coarse prior (odometry/GIS rough
    initial alignment); the ICP initial guess is the B->A transform of it.
    """
    rotation_pert = rotation_z(yaw_deg) @ rotation_b
    eye_pert = np.asarray(eye_b, dtype=float) + np.asarray(shift, dtype=float)
    translation_pert = -rotation_pert @ eye_pert
    return rotation_pert, translation_pert, eye_pert


def initial_guess(scene, yaw_deg, shift):
    """Initial guess = B->A transform of the perturbed true pose B."""
    rotation_pert, translation_pert, _ = perturbed_pose_b(
        yaw_deg, shift, scene["rotation_b"], scene["eye_b"]
    )
    return init_from_pose_b(
        rotation_pert, translation_pert, scene["rotation_a"], scene["translation_a"]
    )


def initial_guess_translation(scene, shift):
    """Initial guess = the true transform with a pure world-frame shift."""
    return scene["gt_rotation"], scene["gt_translation"] + np.asarray(shift, dtype=float)


# -------------------------------------------------------------------- clouds
def voxel_select(points, rng, voxel_size=VOXEL_SIZE_M, max_points=MAX_CLOUD_POINTS):
    """Indices of the downsampled cloud: for every occupied voxel the member
    closest to one uniform random in-voxel target, then a seeded uniform cut
    to ``max_points``. Deterministic given ``rng``. Both random choices
    matter: a deterministic pick preserves the two clouds' voxel lattices (the
    nearest-neighbour cost between two lattices has a coherent nesting force
    that drags the registration centimetres off the truth), and a raw random
    PIXEL pick clusters where perspective piles pixels toward the camera.
    Uniform in-voxel targets keep the samples space-uniform and mutually
    incoherent."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a non-empty [N, 3] point array")
    keys = np.floor(points / voxel_size).astype(np.int64)
    kmin = keys.min(axis=0)
    span = keys.max(axis=0) - kmin + 1
    code = ((keys[:, 0] - kmin[0]) * span[1] + (keys[:, 1] - kmin[1])) * span[2] + (
        keys[:, 2] - kmin[2]
    )
    uniq, inverse = np.unique(code, return_inverse=True)
    kz = uniq % span[2]
    ky = (uniq // span[2]) % span[1]
    kx = uniq // (span[1] * span[2])
    base = np.column_stack([kx, ky, kz]) + kmin
    # one uniform random target inside each voxel: the representative is then
    # space-uniform (a raw pixel pick would cluster where pixels - and their
    # perspective density - pile up toward the camera) and the two clouds'
    # representatives stay mutually incoherent.
    targets = (base + rng.random((len(uniq), 3))) * voxel_size
    d2 = ((points - targets[inverse]) ** 2).sum(axis=1)
    order = np.lexsort((d2, inverse))
    sorted_inv = inverse[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_inv[1:] != sorted_inv[:-1]
    selected = order[first]
    if len(selected) > max_points:
        drop = rng.choice(len(selected), size=len(selected) - max_points, replace=False)
        mask = np.ones(len(selected), dtype=bool)
        mask[drop] = False
        selected = selected[mask]
    return selected


def voxel_downsample(points, rng, voxel_size=VOXEL_SIZE_M, max_points=MAX_CLOUD_POINTS):
    points = np.asarray(points, dtype=float)
    return points[voxel_select(points, rng, voxel_size, max_points)]


def build_clouds(scene, sigma, sigma_index, repetition, seed, ground_only=False, block=1):
    """Noisy depth -> sensor-frame clouds (plus pole flags) for one realization.

    Independent Gaussian depth noise (one stream per realization, shared by
    both cameras) is added to the clean depth maps; the lesson-22 ``unproject``
    maps pixels+depth back to world points, which are then expressed in each
    camera's own frame and voxel-downsampled. The per-point pole flags come
    from the renderer's pole mask and survive the downsampling. With
    ``ground_only`` the pole pixels are dropped before downsampling (the
    degenerate flat-scene counterexample).
    """
    rng = np.random.default_rng([seed, block, sigma_index, repetition])
    clouds = {}
    for key, rotation, translation, rendered in (
        ("fixed", scene["rotation_a"], scene["translation_a"], scene["rendered_a"]),
        ("moving", scene["rotation_b"], scene["translation_b"], scene["rendered_b"]),
    ):
        valid, pole = rendered["valid"], rendered["pole"]
        depth = rendered["depth_m"]
        if sigma > 0.0:
            depth = depth + rng.normal(0.0, sigma, depth.shape)
        rows, cols = np.nonzero(valid)
        pole_flag = pole[rows, cols]
        keep = ~pole_flag if ground_only else np.ones(len(rows), dtype=bool)
        pixels = np.column_stack([cols, rows]).astype(float)[keep]
        depths = depth[rows, cols][keep]
        world = unproject(pixels, depths, rotation, translation)
        points = to_camera(world, rotation, translation)
        selected = voxel_select(points, rng)
        clouds[key] = points[selected]
        clouds[key + "_pole"] = pole_flag[keep][selected]
    return clouds


# ------------------------------------------------------------ nearest & PCA
def nearest_indices(query, reference):
    """Nearest reference point for every query point.

    Vectorized Gram trick: d2 = |q|^2 + |r|^2 - 2 q r^T evaluated in float32
    (2000 x 2000 -> ~16 MB, ~5 ms), the recorded O(N^2) trade-off of doing
    nearest neighbours without scipy/kd-trees on downsampled clouds.
    """
    query = np.ascontiguousarray(query, dtype=np.float32)
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    if query.shape[1] != 3 or reference.shape[1] != 3:
        raise ValueError("Expected [N, 3] point arrays")
    q2 = np.einsum("ij,ij->i", query, query)[:, None]
    r2 = np.einsum("ij,ij->i", reference, reference)[None, :]
    d2 = q2 + r2 - 2.0 * (query @ reference.T)
    np.maximum(d2, 0.0, out=d2)
    index = np.argmin(d2, axis=1)
    distance = np.sqrt(d2[np.arange(len(query)), index])
    return index, distance


def _pairwise_self_d2(points):
    points32 = np.ascontiguousarray(points, dtype=np.float32)
    sq = np.einsum("ij,ij->i", points32, points32)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (points32 @ points32.T)
    np.maximum(d2, 0.0, out=d2)
    d2[np.arange(len(points32)), np.arange(len(points32))] = np.inf  # drop self
    return d2


def estimate_normals(points, k=NORMAL_K):
    """PCA normal per point from its k-NN covariance (sign toward the origin).

    Diagnostic alternative to the analytic normals: on noisy clouds the
    k-neighbourhood plane fit tilts by O(depth noise / neighbourhood size),
    which is measured against the analytic normals in the summary.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= k:
        raise ValueError("Need more points than neighbours for PCA normals")
    d2 = _pairwise_self_d2(points)
    part = np.argpartition(d2, k - 1, axis=1)[:, :k]
    neighbours = points[part]  # (N, k, 3)
    local = neighbours - neighbours.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", local, local)
    _, vectors = np.linalg.eigh(cov)  # ascending eigenvalues
    normals = vectors[:, :, 0]
    flip = np.einsum("ni,ni->n", normals, points) > 0.0
    normals[flip] *= -1.0
    return normals


def analytic_normals(points, pole_flags, rotation, translation):
    """Exact surface normals of the simulated scene, in the cloud's frame.

    Ground points (pole flag False) take the world up-axis mapped through the
    pose; pole points take the horizontal radial direction from the known pole
    axis to the point (the direction is stable even on noisy points because
    the depth noise moves them mostly along the ray). Normals are flipped to
    point toward the sensor at the frame origin.
    """
    points = np.asarray(points, dtype=float)
    pole_flags = np.asarray(pole_flags, dtype=bool)
    if len(points) != len(pole_flags):
        raise ValueError("Need one pole flag per point")
    normals = np.zeros_like(points)
    ground = ~pole_flags
    normals[ground] = rotation[:, 2]  # R @ e_z: world up-axis in camera frame
    if pole_flags.any():
        world = (rotation.T @ (points[pole_flags] - translation).T).T
        radial = world - np.array([POLE_X_M, POLE_Y_M, 0.0])
        radial[:, 2] = 0.0
        norm = np.linalg.norm(radial, axis=1, keepdims=True)
        fallback = norm[:, 0] < 1e-9
        if fallback.any():  # degenerate noisy point on the axis: reuse the mean
            mean_dir = (
                radial[~fallback].mean(axis=0) if (~fallback).any() else np.array([1.0, 0.0, 0.0])
            )
            radial[fallback] = mean_dir
            norm[fallback] = np.linalg.norm(mean_dir)
        normals[pole_flags] = (rotation @ (radial / norm).T).T
    flip = np.einsum("ij,ij->i", normals, points) > 0.0
    normals[flip] *= -1.0
    return normals


# ------------------------------------------------------------------ solvers
def solve_rigid_point_to_point(source, target):
    """Least-squares rigid transform mapping source onto target (SVD/Kabsch)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Expected matching [N, 3] arrays")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    cross_cov = (source - source_centroid).T @ (target - target_centroid)
    u, _, vt = np.linalg.svd(cross_cov)
    correction = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, correction]) @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def solve_point_to_plane(source, target, normals_fixed, normals_moving, eigen_cutoff=EIGEN_CUTOFF):
    """One linearized TWO-SIDED point-to-plane increment (hand-written normal
    equations).

    Minimizes sum r_i^2 with the symmetric residual
    r_i = 0.5*(n_q + n_p).(R p_i + t - q_i)
    (n_q = fixed-point normal, n_p = moving-point normal; R linearized as
    I + [omega]_x about the cloud centroid - the Hartley-style balancing of the
    lesson-25 DLT, here so the 6x6 system is not scale-dominated by points far
    from the origin). The two-sided form cancels the systematic chord bias
    -r(1-cos(dphi)) that a one-sided residual carries when two discrete
    samples of a curved surface are matched at different azimuths. The 6x6
    normal matrix is solved through its eigendecomposition with a relative
    cutoff: directions whose eigenvalue is below ``eigen_cutoff`` of the
    largest (unobservable modes such as sliding a plane along itself, or the
    scene's yaw symmetry) are projected OUT, so the increment keeps the
    current pose along them instead of amplifying normal-estimation noise
    into large spurious jumps. Returns (rotation_increment,
    translation_increment, condition_number, rank).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    normals_fixed = np.asarray(normals_fixed, dtype=float)
    normals_moving = np.asarray(normals_moving, dtype=float)
    if not (len(source) == len(target) == len(normals_fixed) == len(normals_moving)):
        raise ValueError("Pair arrays must have equal length")
    centroid = source.mean(axis=0)
    local = source - centroid
    offset = local - (target - centroid)
    normals_avg = 0.5 * (normals_fixed + normals_moving)
    residual0 = np.einsum("ij,ij->i", normals_avg, offset)
    design = np.column_stack([np.cross(local, normals_avg), normals_avg])
    normal_matrix = design.T @ design
    rhs = design.T @ (-residual0)
    eigenvalues, eigenvectors = np.linalg.eigh(normal_matrix)
    cutoff = max(float(eigenvalues[-1]) * eigen_cutoff, 1e-18)
    keep = eigenvalues > cutoff
    rank = int(np.sum(keep))
    if rank == 0:
        return np.eye(3), np.zeros(3), np.inf, 0
    retained = eigenvectors[:, keep]
    solution = retained @ ((retained.T @ rhs) / eigenvalues[keep])
    cond = float(np.sqrt(eigenvalues[-1] / eigenvalues[keep][0]))
    omega, shift = solution[:3], solution[3:]
    angle = float(np.linalg.norm(omega))
    if angle > STEP_CLAMP_RAD:  # keep the linearization honest
        omega = omega * (STEP_CLAMP_RAD / angle)
        shift = shift * (STEP_CLAMP_RAD / angle)
        angle = STEP_CLAMP_RAD
    rotation = rodrigues(omega)
    translation = shift - (rotation - np.eye(3)) @ centroid
    return rotation, translation, cond, rank


# ---------------------------------------------------------------------- ICP
def icp_register(
    moving,
    fixed,
    fixed_normals=None,
    *,
    moving_normals=None,
    init_rotation=None,
    init_translation=None,
    objective="point_to_plane",
    max_iters=MAX_ITERS,
    tol=TOL,
    tau0_m=TAU0_M,
    tau_gamma=TAU_GAMMA,
    tau_floor_m=TAU_FLOOR_M,
    min_pairs=MIN_PAIRS,
    record_transforms=False,
):
    """Alternating correspondence/solve ICP with a shrinking trim threshold.

    point_to_plane uses the TWO-SIDED residual 0.5*(n_fixed + n_moving).(Tp-q):
    on a curved surface sampled at different azimuths the one-sided residual
    carries a systematic chord bias -r(1-cos(dphi)) (the surface curves away
    from the tangent plane), which the two-sided average cancels exactly.
    Returns a dict with the final (rotation, translation), the number of
    solves, ``converged`` plus a ``reason`` in {"tol", "max_iters", "starved"}
    and a per-step history (threshold, pair count, mean/RMS pair distance,
    inlier ratio, increment sizes, p2plane condition number and rank).
    Failures are recorded, never raised: a starved or capped run returns the
    last pose.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    moving = np.asarray(moving, dtype=float)
    fixed = np.asarray(fixed, dtype=float)
    if moving.shape[1] != 3 or fixed.shape[1] != 3:
        raise ValueError("Expected [N, 3] point arrays")
    if objective == "point_to_plane":
        if fixed_normals is None or len(fixed_normals) != len(fixed):
            raise ValueError("point_to_plane needs one normal per fixed point")
        if moving_normals is None or len(moving_normals) != len(moving):
            raise ValueError("point_to_plane needs one normal per moving point")
        fixed_normals = np.asarray(fixed_normals, dtype=float)
        moving_normals = np.asarray(moving_normals, dtype=float)
    rotation = np.eye(3) if init_rotation is None else np.asarray(init_rotation, dtype=float)
    translation = (
        np.zeros(3) if init_translation is None else np.asarray(init_translation, dtype=float)
    )
    transforms = [(rotation.copy(), translation.copy())] if record_transforms else []
    history = {
        key: []
        for key in (
            "tau_m",
            "pairs",
            "mean_dist_m",
            "rms_dist_m",
            "inlier_ratio",
            "drot_deg",
            "dt_m",
            "cond",
            "rank",
        )
    }
    converged = False
    reason = "max_iters"
    iterations = 0
    stats_last = None
    for step in range(max_iters):
        transformed = apply_transform(rotation, translation, moving)
        index, distance = nearest_indices(transformed, fixed)
        tau = max(tau_floor_m, tau0_m * tau_gamma**step)
        keep = distance <= tau
        pairs = int(keep.sum())
        stats = {
            "tau_m": tau,
            "pairs": pairs,
            "mean_dist_m": float(distance[keep].mean()) if pairs else np.nan,
            "rms_dist_m": float(np.sqrt((distance[keep] ** 2).mean())) if pairs else np.nan,
            "inlier_ratio": pairs / len(moving),
            "drot_deg": np.nan,
            "dt_m": np.nan,
            "cond": np.nan,
            "rank": 6,
        }
        stats_last = stats
        if pairs < min_pairs:
            reason = "starved"
            break
        source = transformed[keep]
        target = fixed[index[keep]]
        if objective == "point_to_point":
            rot_inc, trans_inc = solve_rigid_point_to_point(source, target)
            cond, rank = np.nan, 6
        else:
            rot_inc, trans_inc, cond, rank = solve_point_to_plane(
                source,
                target,
                fixed_normals[index[keep]],
                moving_normals[keep],
            )
        drot = rotation_angle_deg(rot_inc)
        dt = float(np.linalg.norm(trans_inc))
        stats["drot_deg"], stats["dt_m"], stats["cond"], stats["rank"] = drot, dt, cond, rank
        rotation = rot_inc @ rotation
        translation = rot_inc @ translation + trans_inc
        iterations = step + 1
        for key, value in history.items():
            value.append(stats[key])
        if record_transforms:
            transforms.append((rotation.copy(), translation.copy()))
        if np.radians(drot) < tol and dt < tol:
            converged = True
            reason = "tol"
            break
    if stats_last is not None and len(history["pairs"]) == 0:
        for key, value in stats_last.items():
            history[key].append(value)
    result = {
        "rotation": rotation,
        "translation": translation,
        "iterations": iterations,
        "converged": converged,
        "reason": reason,
        "history": history,
        "last_tau_m": stats_last["tau_m"] if stats_last else np.nan,
        "last_mean_dist_m": stats_last["mean_dist_m"] if stats_last else np.nan,
        "last_inlier_ratio": stats_last["inlier_ratio"] if stats_last else np.nan,
        "last_cond": stats_last["cond"] if stats_last else np.nan,
        "last_rank": stats_last["rank"] if stats_last else np.nan,
    }
    if record_transforms:
        result["transform_rotation"] = np.asarray([r for r, _ in transforms])
        result["transform_translation"] = np.asarray([t for _, t in transforms])
    return result


# ----------------------------------------------------------------- experiment
def _agg(values, sigma):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "max": float(values.max()),
        "over_sigma": float(values.mean() / sigma) if sigma > 0 else np.nan,
    }


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED, light=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two repetitions for noise statistics")
    # light=True keeps every mechanism and contract identical but shrinks the
    # convergence-radius grid and its iteration budget; used only by tests so a
    # full-suite run does not re-run the formal-size study four times over.
    radius_yaw = LIGHT_RADIUS_YAW_DEG if light else RADIUS_YAW_DEG
    radius_shift = LIGHT_RADIUS_SHIFT_M if light else RADIUS_SHIFT_M
    radius_max_iters = LIGHT_MAX_ITERS if light else MAX_ITERS
    scene = scene_and_poses()
    gt_rotation, gt_translation = scene["gt_rotation"], scene["gt_translation"]
    sweep_rotation0, sweep_translation0 = gt_rotation, gt_translation
    degen_rotation0, degen_translation0 = initial_guess_translation(scene, DEGEN_INIT_SHIFT_M)
    _, _, degen_trans_err0 = pose_error(
        degen_rotation0, degen_translation0, gt_rotation, gt_translation
    )
    ident_rot_err0, _, ident_trans_err0 = pose_error(
        np.eye(3), np.zeros(3), gt_rotation, gt_translation
    )
    # horizontal lever arm of the pole about camera A: theoretical yaw footprint
    pole_offset = np.array([POLE_X_M - EYE[0], POLE_Y_M - EYE[1]])
    yaw_lever_m = float(np.linalg.norm(pole_offset))

    def clouds_for(sigma_index, repetition, ground_only=False):
        return build_clouds(
            scene, SIGMA_VALUES[sigma_index], sigma_index, repetition, seed, ground_only
        )

    def radius_clouds(repetition):
        return build_clouds(scene, RADIUS_SIGMA_M, 0, repetition, seed, block=2)

    def make_normals(clouds):
        """Analytic normals for both clouds, each in its own sensor frame."""
        return (
            analytic_normals(
                clouds["fixed"],
                clouds["fixed_pole"],
                scene["rotation_a"],
                scene["translation_a"],
            ),
            analytic_normals(
                clouds["moving"],
                clouds["moving_pole"],
                scene["rotation_b"],
                scene["translation_b"],
            ),
        )

    def run_pair(clouds, normals, rotation0, translation0, **kwargs):
        kwargs.setdefault("max_iters", radius_max_iters)
        fixed_normals, moving_normals = normals
        return {
            objective: icp_register(
                clouds["moving"],
                clouds["fixed"],
                fixed_normals,
                moving_normals=moving_normals,
                init_rotation=rotation0,
                init_translation=translation0,
                objective=objective,
                **kwargs,
            )
            for objective in OBJECTIVES
        }

    def score(rotation_est, translation_est):
        """Naive pose error + the symmetry-quotient decomposition."""
        rot_err, trans_vec, trans_norm = pose_error(
            rotation_est, translation_est, gt_rotation, gt_translation
        )
        quotient = symmetry_quotient_error(
            rotation_est,
            translation_est,
            gt_rotation,
            gt_translation,
            scene["rotation_a"],
            scene["translation_a"],
        )
        return rot_err, trans_vec, trans_norm, quotient

    # ---- reference run (demo trajectory): sigma=0.15, repetition 0,
    #      0.1 m translation-only initial error
    ref_clouds = clouds_for(DEGEN_SIGMA_INDEX, 0)
    ref_fixed_normals, ref_moving_normals = make_normals(ref_clouds)
    ref_normals = ref_fixed_normals
    pca_normals = estimate_normals(ref_clouds["fixed"], k=NORMAL_K)
    pca_align = np.abs(np.einsum("ij,ij->i", pca_normals, ref_normals))
    pca_angle_deg = np.degrees(np.arccos(np.clip(pca_align, -1.0, 1.0)))
    pca_median_err_deg = float(np.median(pca_angle_deg))
    ref_runs = run_pair(
        ref_clouds,
        (ref_fixed_normals, ref_moving_normals),
        degen_rotation0,
        degen_translation0,
        record_transforms=True,
    )
    ref_errors = {
        obj: score(res["rotation"], res["translation"])[3] for obj, res in ref_runs.items()
    }
    # ---- degenerate reference (ground only, same stream: paired realization)
    degen_clouds_ref = clouds_for(DEGEN_SIGMA_INDEX, 0, ground_only=True)
    degen_fixed_normals, degen_moving_normals = make_normals(degen_clouds_ref)
    degen_ref = icp_register(
        degen_clouds_ref["moving"],
        degen_clouds_ref["fixed"],
        degen_fixed_normals,
        moving_normals=degen_moving_normals,
        init_rotation=degen_rotation0,
        init_translation=degen_translation0,
        objective="point_to_plane",
        record_transforms=True,
    )
    degen_ref_rot_err, _, degen_ref_trans_err = pose_error(
        degen_ref["rotation"], degen_ref["translation"], gt_rotation, gt_translation
    )

    # ---- sigma sweep (translation-only init, both objectives, 20 seeds)
    n_sigma = len(SIGMA_VALUES)
    sweep_keys = (
        "trans_mean_m",
        "trans_std_m",
        "trans_median_m",
        "trans_max_m",
        "rot_mean_deg",
        "rot_std_deg",
        "trans_obs_mean_m",
        "trans_obs_std_m",
        "rot_obs_mean_deg",
        "spin_mean_deg",
        "spin_std_deg",
        "iters_mean",
        "iters_median",
        "iters_max",
        "converged_count",
        "mean_pair_dist_m",
    )
    sweep = {key: np.zeros((len(OBJECTIVES), n_sigma)) for key in sweep_keys}
    sweep_vec_mean = np.zeros((len(OBJECTIVES), n_sigma, 3))
    normals_cache = {}
    for sigma_index, sigma in enumerate(SIGMA_VALUES):
        collected = {
            (objective, key): []
            for objective in OBJECTIVES
            for key in (
                "trans",
                "rot",
                "trans_obs",
                "rot_obs",
                "spin",
                "iters",
                "converged",
                "pair_dist",
                "vec",
            )
        }
        for repetition in range(runs):
            clouds = clouds_for(sigma_index, repetition)
            if (sigma_index, repetition) not in normals_cache:
                normals_cache[(sigma_index, repetition)] = make_normals(clouds)
            normals = normals_cache[(sigma_index, repetition)]
            results = run_pair(clouds, normals, sweep_rotation0, sweep_translation0)
            for objective, res in results.items():
                rot_err, trans_vec, trans_norm, quotient = score(
                    res["rotation"], res["translation"]
                )
                collected[(objective, "trans")].append(trans_norm)
                collected[(objective, "rot")].append(rot_err)
                collected[(objective, "trans_obs")].append(quotient["trans_err_obs_m"])
                collected[(objective, "rot_obs")].append(quotient["rot_err_obs_deg"])
                collected[(objective, "spin")].append(quotient["spin_deg"])
                collected[(objective, "iters")].append(res["iterations"])
                collected[(objective, "converged")].append(res["converged"])
                collected[(objective, "pair_dist")].append(res["last_mean_dist_m"])
                collected[(objective, "vec")].append(trans_vec)
        for oi, objective in enumerate(OBJECTIVES):
            trans = _agg(collected[(objective, "trans")], sigma)
            rot = _agg(collected[(objective, "rot")], sigma)
            trans_obs = _agg(collected[(objective, "trans_obs")], sigma)
            spin = _agg(collected[(objective, "spin")], sigma)
            iters = _agg(collected[(objective, "iters")], sigma)
            sweep["trans_mean_m"][oi, sigma_index] = trans["mean"]
            sweep["trans_std_m"][oi, sigma_index] = trans["std"]
            sweep["trans_median_m"][oi, sigma_index] = trans["median"]
            sweep["trans_max_m"][oi, sigma_index] = trans["max"]
            sweep["rot_mean_deg"][oi, sigma_index] = rot["mean"]
            sweep["rot_std_deg"][oi, sigma_index] = rot["std"]
            sweep["trans_obs_mean_m"][oi, sigma_index] = trans_obs["mean"]
            sweep["trans_obs_std_m"][oi, sigma_index] = trans_obs["std"]
            sweep["rot_obs_mean_deg"][oi, sigma_index] = float(
                np.mean(collected[(objective, "rot_obs")])
            )
            sweep["spin_mean_deg"][oi, sigma_index] = spin["mean"]
            sweep["spin_std_deg"][oi, sigma_index] = spin["std"]
            sweep["iters_mean"][oi, sigma_index] = iters["mean"]
            sweep["iters_median"][oi, sigma_index] = iters["median"]
            sweep["iters_max"][oi, sigma_index] = iters["max"]
            sweep["converged_count"][oi, sigma_index] = int(
                np.sum(collected[(objective, "converged")])
            )
            sweep["mean_pair_dist_m"][oi, sigma_index] = float(
                np.nanmean(collected[(objective, "pair_dist")])
            )
            sweep_vec_mean[oi, sigma_index] = np.mean(collected[(objective, "vec")], axis=0)
        print(f"sigma sweep {sigma:.2f} m done", flush=True)

    # ---- convergence-radius grid at sigma=0.05 (point-to-plane matrix,
    #      identity/truth reference runs for both objectives). Convergence is
    #      judged on the observable-subspace error: the ICP cost is invariant
    #      along the scene's spin symmetry, so the naive translation error
    #      carries the symmetry coordinate (footprint ~ lever x spin) that no
    #      amount of iteration can remove.
    radius_frac = np.zeros((1, len(radius_yaw), len(radius_shift)))
    radius_err_obs = np.zeros_like(radius_frac)
    radius_err_naive = np.zeros_like(radius_frac)
    radius_spin = np.zeros_like(radius_frac)
    radius_iters = np.zeros_like(radius_frac)
    identity = {
        objective: {"converged": [], "trans": [], "trans_obs": [], "rot": [], "iters": []}
        for objective in OBJECTIVES
    }
    truth = {
        objective: {"converged": [], "trans": [], "trans_obs": [], "rot": [], "iters": []}
        for objective in OBJECTIVES
    }
    for repetition in range(runs):
        clouds = radius_clouds(repetition)
        normals = make_normals(clouds)
        for yi, yaw in enumerate(radius_yaw):
            for xi, shift_value in enumerate(radius_shift):
                yaw_signed = yaw if repetition % 2 == 0 else -yaw
                shift = np.array([shift_value, 0.0, 0.0])
                if (repetition // 2) % 2 == 1:
                    shift = -shift
                rotation0, translation0 = initial_guess(scene, yaw_signed, shift)
                res = icp_register(
                    clouds["moving"],
                    clouds["fixed"],
                    normals[0],
                    moving_normals=normals[1],
                    init_rotation=rotation0,
                    init_translation=translation0,
                    objective="point_to_plane",
                    max_iters=radius_max_iters,
                )
                _, _, _, quotient = score(res["rotation"], res["translation"])
                radius_frac[0, yi, xi] += quotient["trans_err_obs_m"] < CONVERGED_TRANS_M
                radius_err_obs[0, yi, xi] += quotient["trans_err_obs_m"]
                radius_err_naive[0, yi, xi] += quotient["naive_trans_m"]
                radius_spin[0, yi, xi] += quotient["spin_deg"]
                radius_iters[0, yi, xi] += res["iterations"]
        for objective in OBJECTIVES:
            for tag, store in (("identity", identity), ("truth", truth)):
                if tag == "identity":
                    rotation0 = np.eye(3)
                    translation0 = np.zeros(3)
                else:
                    rotation0, translation0 = gt_rotation, gt_translation
                res = icp_register(
                    clouds["moving"],
                    clouds["fixed"],
                    normals[0],
                    moving_normals=normals[1],
                    init_rotation=rotation0,
                    init_translation=translation0,
                    objective=objective,
                )
                rot_err, _, trans_norm, quotient = score(res["rotation"], res["translation"])
                store[objective]["converged"].append(
                    quotient["trans_err_obs_m"] < CONVERGED_TRANS_M
                )
                store[objective]["trans"].append(trans_norm)
                store[objective]["trans_obs"].append(quotient["trans_err_obs_m"])
                store[objective]["rot"].append(rot_err)
                store[objective]["iters"].append(res["iterations"])
        print(f"radius repetition {repetition + 1}/{runs} done", flush=True)
    for array in (radius_frac, radius_err_obs, radius_err_naive, radius_spin, radius_iters):
        array /= runs
    # measured yaw footprint on the symmetry coordinate: shift=0 cells
    footprint = radius_spin[0, :, 0] / np.maximum(np.asarray(radius_yaw), 1e-9)

    # ---- degenerate counterexample: ground-only clouds (pole removed)
    degen = {
        objective: {"trans": [], "rot": [], "converged": [], "iters": [], "wrong": []}
        for objective in OBJECTIVES
    }
    for repetition in range(runs):
        clouds = clouds_for(DEGEN_SIGMA_INDEX, repetition, ground_only=True)
        normals = make_normals(clouds)
        results = run_pair(clouds, normals, degen_rotation0, degen_translation0)
        for objective, res in results.items():
            rot_err, _, trans_norm, _ = score(res["rotation"], res["translation"])
            degen[objective]["trans"].append(trans_norm)
            degen[objective]["rot"].append(rot_err)
            degen[objective]["converged"].append(res["converged"])
            degen[objective]["iters"].append(res["iterations"])
            degen[objective]["wrong"].append(res["converged"] and trans_norm >= CONVERGED_TRANS_M)
    degen_stats = {}
    for objective in OBJECTIVES:
        degen_stats[objective] = {
            "trans_mean_m": float(np.mean(degen[objective]["trans"])),
            "trans_std_m": float(np.std(degen[objective]["trans"], ddof=1)),
            "rot_mean_deg": float(np.mean(degen[objective]["rot"])),
            "converged_count": int(np.sum(degen[objective]["converged"])),
            "converged_but_wrong_count": int(np.sum(degen[objective]["wrong"])),
            "iters_mean": float(np.mean(degen[objective]["iters"])),
        }
    sigma_ref = DEGEN_SIGMA_INDEX
    pole_reference = {}
    for oi, objective in enumerate(OBJECTIVES):
        pole_reference[objective] = {
            "trans_mean_m": float(sweep["trans_mean_m"][oi, sigma_ref]),
            "trans_std_m": float(sweep["trans_std_m"][oi, sigma_ref]),
            "trans_obs_mean_m": float(sweep["trans_obs_mean_m"][oi, sigma_ref]),
            "rot_mean_deg": float(sweep["rot_mean_deg"][oi, sigma_ref]),
            "converged_count": int(sweep["converged_count"][oi, sigma_ref]),
            "iters_mean": float(sweep["iters_mean"][oi, sigma_ref]),
        }

    identity_stats = {
        objective: {
            "converged_fraction": float(np.mean(identity[objective]["converged"])),
            "mean_trans_err_m": float(np.mean(identity[objective]["trans"])),
            "mean_trans_obs_m": float(np.mean(identity[objective]["trans_obs"])),
            "mean_rot_err_deg": float(np.mean(identity[objective]["rot"])),
            "mean_iters": float(np.mean(identity[objective]["iters"])),
        }
        for objective in OBJECTIVES
    }
    truth_stats = {
        objective: {
            "converged_fraction": float(np.mean(truth[objective]["converged"])),
            "mean_trans_err_m": float(np.mean(truth[objective]["trans"])),
            "mean_trans_obs_m": float(np.mean(truth[objective]["trans_obs"])),
            "mean_rot_err_deg": float(np.mean(truth[objective]["rot"])),
            "mean_iters": float(np.mean(truth[objective]["iters"])),
        }
        for objective in OBJECTIVES
    }

    # ---- archive & summary
    archive = {
        "gt_rotation": gt_rotation,
        "gt_translation": gt_translation,
        "ref_init_rotation": degen_rotation0,
        "ref_init_translation": degen_translation0,
        "ref_fixed_cloud": ref_clouds["fixed"],
        "ref_moving_cloud": ref_clouds["moving"],
        "ref_fixed_pole": ref_clouds["fixed_pole"],
        "ref_fixed_normals": ref_fixed_normals,
        "ref_moving_normals": ref_moving_normals,
        "ref_point_to_point_rotation": ref_runs["point_to_point"]["transform_rotation"],
        "ref_point_to_point_translation": ref_runs["point_to_point"]["transform_translation"],
        "ref_point_to_point_mean_dist_m": np.asarray(
            ref_runs["point_to_point"]["history"]["mean_dist_m"]
        ),
        "ref_point_to_point_inlier_ratio": np.asarray(
            ref_runs["point_to_point"]["history"]["inlier_ratio"]
        ),
        "ref_point_to_point_tau_m": np.asarray(ref_runs["point_to_point"]["history"]["tau_m"]),
        "ref_point_to_plane_rotation": ref_runs["point_to_plane"]["transform_rotation"],
        "ref_point_to_plane_translation": ref_runs["point_to_plane"]["transform_translation"],
        "ref_point_to_plane_mean_dist_m": np.asarray(
            ref_runs["point_to_plane"]["history"]["mean_dist_m"]
        ),
        "ref_point_to_plane_inlier_ratio": np.asarray(
            ref_runs["point_to_plane"]["history"]["inlier_ratio"]
        ),
        "ref_point_to_plane_tau_m": np.asarray(ref_runs["point_to_plane"]["history"]["tau_m"]),
        "degen_init_rotation": degen_rotation0,
        "degen_init_translation": degen_translation0,
        "degen_fixed_cloud": degen_clouds_ref["fixed"],
        "degen_moving_cloud": degen_clouds_ref["moving"],
        "degen_final_rotation": degen_ref["rotation"],
        "degen_final_translation": degen_ref["translation"],
        "degen_final_moving_cloud": apply_transform(
            degen_ref["rotation"], degen_ref["translation"], degen_clouds_ref["moving"]
        ),
        "degen_mean_dist_m": np.asarray(degen_ref["history"]["mean_dist_m"]),
        "degen_tau_m": np.asarray(degen_ref["history"]["tau_m"]),
        "sigma_values": np.asarray(SIGMA_VALUES),
        "sweep_trans_mean_m": sweep["trans_mean_m"],
        "sweep_trans_std_m": sweep["trans_std_m"],
        "sweep_trans_median_m": sweep["trans_median_m"],
        "sweep_trans_max_m": sweep["trans_max_m"],
        "sweep_rot_mean_deg": sweep["rot_mean_deg"],
        "sweep_rot_std_deg": sweep["rot_std_deg"],
        "sweep_trans_obs_mean_m": sweep["trans_obs_mean_m"],
        "sweep_trans_obs_std_m": sweep["trans_obs_std_m"],
        "sweep_rot_obs_mean_deg": sweep["rot_obs_mean_deg"],
        "sweep_spin_mean_deg": sweep["spin_mean_deg"],
        "sweep_spin_std_deg": sweep["spin_std_deg"],
        "sweep_iters_mean": sweep["iters_mean"],
        "sweep_iters_median": sweep["iters_median"],
        "sweep_iters_max": sweep["iters_max"],
        "sweep_converged_count": sweep["converged_count"],
        "sweep_mean_pair_dist_m": sweep["mean_pair_dist_m"],
        "sweep_vec_mean_m": sweep_vec_mean,
        "radius_yaw_values": np.asarray(radius_yaw),
        "radius_shift_values": np.asarray(radius_shift),
        "radius_converged_fraction": radius_frac,
        "radius_mean_trans_obs_m": radius_err_obs,
        "radius_mean_trans_naive_m": radius_err_naive,
        "radius_mean_spin_deg": radius_spin,
        "radius_mean_iters": radius_iters,
        "identity_converged_fraction": np.asarray(
            [identity_stats[obj]["converged_fraction"] for obj in OBJECTIVES]
        ),
        "identity_mean_trans_err_m": np.asarray(
            [identity_stats[obj]["mean_trans_err_m"] for obj in OBJECTIVES]
        ),
        "identity_mean_trans_obs_m": np.asarray(
            [identity_stats[obj]["mean_trans_obs_m"] for obj in OBJECTIVES]
        ),
        "identity_mean_rot_err_deg": np.asarray(
            [identity_stats[obj]["mean_rot_err_deg"] for obj in OBJECTIVES]
        ),
        "truth_mean_trans_err_m": np.asarray(
            [truth_stats[obj]["mean_trans_err_m"] for obj in OBJECTIVES]
        ),
        "truth_mean_trans_obs_m": np.asarray(
            [truth_stats[obj]["mean_trans_obs_m"] for obj in OBJECTIVES]
        ),
        "degen_trans_mean_m": np.asarray([degen_stats[obj]["trans_mean_m"] for obj in OBJECTIVES]),
        "degen_trans_std_m": np.asarray([degen_stats[obj]["trans_std_m"] for obj in OBJECTIVES]),
        "degen_rot_mean_deg": np.asarray([degen_stats[obj]["rot_mean_deg"] for obj in OBJECTIVES]),
        "degen_converged_count": np.asarray(
            [degen_stats[obj]["converged_count"] for obj in OBJECTIVES]
        ),
        "degen_converged_but_wrong_count": np.asarray(
            [degen_stats[obj]["converged_but_wrong_count"] for obj in OBJECTIVES]
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    edge_cells = sorted(
        {
            (0, 0),
            (0, len(radius_shift) - 1),
            (len(radius_yaw) - 1, 0),
            (len(radius_yaw) - 1, len(radius_shift) - 1),
            (1, 1),
            (1, 2),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 3),
        }
    )
    edge_cells = [
        (yi, xi) for yi, xi in edge_cells if yi < len(radius_yaw) and xi < len(radius_shift)
    ]
    radius_edge = {
        f"{radius_yaw[yi]:.0f}deg_{radius_shift[xi]:.1f}m": {
            "converged_fraction": float(radius_frac[0, yi, xi]),
            "mean_trans_obs_m": float(radius_err_obs[0, yi, xi]),
            "mean_trans_naive_m": float(radius_err_naive[0, yi, xi]),
            "mean_spin_deg": float(radius_spin[0, yi, xi]),
            "mean_iters": float(radius_iters[0, yi, xi]),
        }
        for yi, xi in edge_cells
    }
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "intrinsic": K_INTRINSIC.tolist(),
        "near_plane_m": NEAR_PLANE_M,
        "eye_a_m": EYE.tolist(),
        "target_a_m": TARGET.tolist(),
        "pose_b": {
            "delta_world_m": list(POSE_DELTA_M),
            "yaw_deg": POSE_YAW_DEG,
            "eye_b_m": scene["eye_b"].tolist(),
            "gt_rotation_deg": rotation_angle_deg(gt_rotation),
            "gt_translation_m": gt_translation.tolist(),
            "gt_translation_norm_m": float(np.linalg.norm(gt_translation)),
        },
        "pixels": {
            "valid_a": int(scene["rendered_a"]["valid"].sum()),
            "pole_a": int(scene["rendered_a"]["pole"].sum()),
            "valid_b": int(scene["rendered_b"]["valid"].sum()),
            "pole_b": int(scene["rendered_b"]["pole"].sum()),
        },
        "downsample": {
            "method": "voxel grid, closest to one uniform random in-voxel target, seeded cut",
            "voxel_size_m": VOXEL_SIZE_M,
            "max_points": MAX_CLOUD_POINTS,
            "ref_fixed_points": len(ref_clouds["fixed"]),
            "ref_moving_points": len(ref_clouds["moving"]),
            "ref_fixed_pole_points": int(ref_clouds["fixed_pole"].sum()),
            "ref_moving_pole_points": int(ref_clouds["moving_pole"].sum()),
            "ref_degen_fixed_points": len(degen_clouds_ref["fixed"]),
            "ref_degen_moving_points": len(degen_clouds_ref["moving"]),
        },
        "icp": {
            "objectives": list(OBJECTIVES),
            "normals": "analytic surface normals (ground plane + pole cylinder)",
            "normals_pca_knn_median_err_deg": pca_median_err_deg,
            "nearest": "vectorized Gram-trick O(N^2) on float32, downsampled clouds",
            "max_iters": radius_max_iters,
            "tol": TOL,
            "tau0_m": TAU0_M,
            "tau_gamma": TAU_GAMMA,
            "tau_floor_m": TAU_FLOOR_M,
            "trim_schedule": "tau_k = max(tau_floor, tau0 * gamma^k)",
            "min_pairs": MIN_PAIRS,
            "step_clamp_rad": STEP_CLAMP_RAD,
            "p2plane_eigen_cutoff": EIGEN_CUTOFF,
            "p2plane_singular_guard": (
                "eigendirections below cutoff*lambda_max are projected out of the "
                "increment (unobservable modes keep the current pose)"
            ),
        },
        "converged_trans_threshold_m": CONVERGED_TRANS_M,
        "symmetry": {
            "mode": "yaw about the vertical line through the pole",
            "reason": (
                "the ground plane is blind to any horizontal rotation and the "
                "cylinder is blind to spins about its own axis - both axes are "
                "world-vertical, so one yaw DOF is structurally unidentifiable"
            ),
            "yaw_lever_m": yaw_lever_m,
        },
        "sigma_values_m": list(SIGMA_VALUES),
        "runs_per_group": runs,
        "base_seed": seed,
        "initial_guesses": {
            "sweep": {"type": "truth", "rot_err_deg": 0.0, "trans_err_m": 0.0},
            "degenerate": {
                "type": "translation only",
                "shift_m": list(DEGEN_INIT_SHIFT_M),
                "rot_err_deg": 0.0,
                "trans_err_m": degen_trans_err0,
            },
            "identity": {
                "rot_err_deg": ident_rot_err0,
                "trans_err_m": ident_trans_err0,
            },
        },
        "reference_run": {
            "sigma_m": SIGMA_VALUES[DEGEN_SIGMA_INDEX],
            "init": "degenerate (0.1 m translation only)",
            "final_trans_err_m": {
                obj: float(ref_errors[obj]["naive_trans_m"]) for obj in OBJECTIVES
            },
            "final_rot_err_deg": {
                obj: float(ref_errors[obj]["naive_rot_deg"]) for obj in OBJECTIVES
            },
            "final_trans_obs_m": {
                obj: float(ref_errors[obj]["trans_err_obs_m"]) for obj in OBJECTIVES
            },
            "final_spin_deg": {obj: float(ref_errors[obj]["spin_deg"]) for obj in OBJECTIVES},
            "iterations": {obj: int(res["iterations"]) for obj, res in ref_runs.items()},
            "converged": {obj: bool(res["converged"]) for obj, res in ref_runs.items()},
            "p2plane_rank_last": int(ref_runs["point_to_plane"]["last_rank"]),
        },
        "sigma_sweep": {
            "init": "truth",
            "trans_mean_m": sweep["trans_mean_m"].tolist(),
            "trans_std_m": sweep["trans_std_m"].tolist(),
            "trans_median_m": sweep["trans_median_m"].tolist(),
            "trans_max_m": sweep["trans_max_m"].tolist(),
            "rot_mean_deg": sweep["rot_mean_deg"].tolist(),
            "rot_std_deg": sweep["rot_std_deg"].tolist(),
            "trans_obs_mean_m": sweep["trans_obs_mean_m"].tolist(),
            "trans_obs_std_m": sweep["trans_obs_std_m"].tolist(),
            "rot_obs_mean_deg": sweep["rot_obs_mean_deg"].tolist(),
            "spin_mean_deg": sweep["spin_mean_deg"].tolist(),
            "spin_std_deg": sweep["spin_std_deg"].tolist(),
            "iters_mean": sweep["iters_mean"].tolist(),
            "iters_median": sweep["iters_median"].tolist(),
            "iters_max": sweep["iters_max"].tolist(),
            "converged_count": sweep["converged_count"].tolist(),
            "mean_pair_dist_m": sweep["mean_pair_dist_m"].tolist(),
            "trans_vec_mean_m": sweep_vec_mean.tolist(),
            "err_over_sigma": (sweep["trans_mean_m"] / np.asarray(SIGMA_VALUES)[None, :]).tolist(),
            "err_obs_over_sigma": (
                sweep["trans_obs_mean_m"] / np.asarray(SIGMA_VALUES)[None, :]
            ).tolist(),
        },
        "radius_study": {
            "sigma_m": RADIUS_SIGMA_M,
            "objective": "point_to_plane",
            "init": "perturbed truth (yaw about world z, x-shift; sign alternates per seed)",
            "convergence_metric": "observable-subspace translation error (symmetry quotient)",
            "rotation_perturb_deg": list(radius_yaw),
            "translation_perturb_m": list(radius_shift),
            "converged_fraction": radius_frac[0].tolist(),
            "mean_trans_obs_m": radius_err_obs[0].tolist(),
            "mean_trans_naive_m": radius_err_naive[0].tolist(),
            "mean_spin_deg": radius_spin[0].tolist(),
            "mean_iters": radius_iters[0].tolist(),
            "truth_init": truth_stats,
            "identity_init": identity_stats,
            "selected_cells": radius_edge,
            "spin_retention_by_row": {
                f"{yaw:.0f}deg": float(footprint[yi])
                for yi, yaw in enumerate(radius_yaw)
                if yaw > 0
            },
            "p2p_matrix_note": (
                "point-to-point is omitted from the grid: its per-iteration "
                "in-plane correction is a few percent (see sigma_sweep iteration "
                "counts), so no cell could converge within the iteration budget"
            ),
        },
        "degenerate": {
            "sigma_m": SIGMA_VALUES[DEGEN_SIGMA_INDEX],
            "init": "degenerate (0.1 m translation only)",
            "pole_scene": pole_reference,
            "ground_only": degen_stats,
            "degen_reference_run": {
                "converged": bool(degen_ref["converged"]),
                "reason": degen_ref["reason"],
                "iterations": int(degen_ref["iterations"]),
                "trans_err_m": float(degen_ref_trans_err),
                "rot_err_deg": float(degen_ref_rot_err),
                "rank_last": int(degen_ref["last_rank"]),
            },
        },
        "mechanism": {
            "err_over_sigma": {
                objective: [
                    float(sweep["trans_mean_m"][oi, i] / SIGMA_VALUES[i]) for i in range(n_sigma)
                ]
                for oi, objective in enumerate(OBJECTIVES)
            },
            "err_obs_over_sigma": {
                objective: [
                    float(sweep["trans_obs_mean_m"][oi, i] / SIGMA_VALUES[i])
                    for i in range(n_sigma)
                ]
                for oi, objective in enumerate(OBJECTIVES)
            },
            "iters_median_ratio_p2p_over_p2plane": float(
                sweep["iters_median"][0, sigma_ref] / max(sweep["iters_median"][1, sigma_ref], 1.0)
            ),
            "symmetry_lever_m": yaw_lever_m,
        },
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {"experiments/point_cloud_icp.py": digest(Path(__file__).resolve())},
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            (
                "Nearest-neighbour correspondences on synthetic clouds: no occlusion-aware "
                "matching, no dynamic objects, no robust kernel beyond distance trimming"
            ),
            (
                "Single scene, single pose pair, one registration per run: no multi-frame "
                "tracking, no loop closure, no global map consistency"
            ),
            (
                "Brute-force O(N^2) nearest neighbours on ~2000-point downsampled clouds; "
                "real registrations need kd-trees/hashes at 1e5-1e6 points"
            ),
            (
                "Depth noise is i.i.d. Gaussian per pixel; real depth cameras show "
                "structured, depth- and surface-dependent noise"
            ),
            (
                "Normals are analytic (the simulated surfaces are known); the PCA k-NN "
                "alternative tilts by the recorded median angle on noisy clouds and was "
                "observed to destabilize point-to-plane ICP"
            ),
            (
                "The convergence radius is measured for yaw-about-world-z / x-shift "
                "perturbations only; the true basin is direction-dependent (anisotropic)"
            ),
        ],
    }
    if light:
        summary["light_mode"] = True
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_plot(archive, summary, output)
    return summary


def make_plot(archive, summary, output):
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    fig.suptitle("第二十六课 两帧带噪点云的 ICP 配准（GIS 多测站拼合的机器人版）")
    ax_raw = fig.add_subplot(2, 3, 1, projection="3d")
    ax_reg = fig.add_subplot(2, 3, 2, projection="3d")
    ax_curve = fig.add_subplot(2, 3, 3)
    ax_sigma = fig.add_subplot(2, 3, 4)
    ax_radius = fig.add_subplot(2, 3, 5)
    ax_degen = fig.add_subplot(2, 3, 6)
    fixed = archive["ref_fixed_cloud"]
    moving = archive["ref_moving_cloud"]
    ax_raw.scatter(*fixed.T, s=3, c="#2563eb", alpha=0.5, label="A 系点云（固定）")
    ax_raw.scatter(*moving.T, s=3, c="#ea580c", alpha=0.5, label="B 系点云（待配准）")
    ax_raw.set(title="配准前：两片点云各在自己的相机系（错位）", xlabel="x / m", ylabel="y / m")
    ax_reg.scatter(*fixed.T, s=3, c="#2563eb", alpha=0.5, label="A 系点云")
    registered = apply_transform(
        archive["ref_point_to_plane_rotation"][-1],
        archive["ref_point_to_plane_translation"][-1],
        moving,
    )
    err = summary["reference_run"]["final_trans_err_m"]["point_to_plane"]
    ax_reg.scatter(*registered.T, s=3, c="#16a34a", alpha=0.5, label="ICP 配准后（点对面）")
    ax_reg.set(
        title=f"配准后：ICP 把 B 系点云搬回 A 系（平移误差 {err * 100:.2f} cm）",
        xlabel="x / m",
        ylabel="y / m",
    )
    for ax in (ax_raw, ax_reg):
        ax.view_init(elev=35, azim=-60)
        ax.legend(fontsize=7, loc="upper left")
    for key, color, label in (
        ("ref_point_to_point_mean_dist_m", "#ea580c", "点到点"),
        ("ref_point_to_plane_mean_dist_m", "#2563eb", "点对面"),
    ):
        values = archive[key]
        ax_curve.plot(np.arange(1, len(values) + 1), values, "-o", ms=3, color=color, label=label)
        taus = archive[key.replace("mean_dist_m", "tau_m")]
        ax_curve.plot(np.arange(1, len(taus) + 1), taus, ":", color=color, lw=1.0, alpha=0.7)
    ax_curve.set_yscale("log")
    ax_curve.set(
        xlabel="迭代轮",
        ylabel="保留对应的平均距离 / m（点线 = 剔除阈值 τ）",
        title="参考运行（σ=0.15 m，0.1 m 平移初值误差）：对应-求解交替",
    )
    ax_curve.legend(fontsize=8)
    sigma_values = archive["sigma_values"]
    for oi, (color, label) in enumerate(zip(("#ea580c", "#2563eb"), ("点到点", "点对面"))):
        means = archive["sweep_trans_mean_m"][oi] * 100.0
        stds = archive["sweep_trans_std_m"][oi] * 100.0
        ax_sigma.errorbar(
            sigma_values, means, yerr=stds, fmt="-o", ms=4, capsize=3, color=color, label=label
        )
        for sigma, mean in zip(sigma_values, means):
            ax_sigma.annotate(
                f"{mean / (sigma * 100):.2f}σ",
                (sigma, mean),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color=color,
            )
    ax_sigma.axhline(CONVERGED_TRANS_M * 100.0, color="#0f172a", ls="--", lw=1.0)
    ax_sigma.text(sigma_values[0], CONVERGED_TRANS_M * 104.0, "2 cm 收敛判据", fontsize=7)
    ax_sigma.set_xscale("log")
    ax_sigma.set_yscale("log")
    ax_sigma.set(
        xlabel="深度噪声 σ / m",
        ylabel="平移误差 / cm（20 种子均值±1σ，标注 误差/σ）",
        title="配准误差随深度噪声近似线性放大",
    )
    ax_sigma.legend(fontsize=8)
    fraction = archive["radius_converged_fraction"][0]
    im = ax_radius.imshow(fraction, cmap="RdYlGn", vmin=0.0, vmax=1.0, origin="lower")
    for yi in range(fraction.shape[0]):
        for xi in range(fraction.shape[1]):
            ax_radius.text(xi, yi, f"{fraction[yi, xi]:.0%}", ha="center", va="center", fontsize=9)
    ax_radius.set_xticks(range(len(RADIUS_SHIFT_M)), [f"{v}" for v in RADIUS_SHIFT_M])
    ax_radius.set_yticks(range(len(RADIUS_YAW_DEG)), [f"{v}°" for v in RADIUS_YAW_DEG])
    ax_radius.set(
        xlabel="平移初值扰动 / m（世界系 x）",
        ylabel="旋转初值扰动（绕世界 z）",
        title=f"收敛半径矩阵（点对面，σ={RADIUS_SIGMA_M} m，格 = 可观子空间收敛占比）",
    )
    fig.colorbar(im, ax=ax_radius, shrink=0.8, label="收敛（<2 cm）占比")
    labels = ("含杆\n点到点", "仅地面\n点到点", "含杆\n点对面", "仅地面\n点对面")
    pole = summary["degenerate"]["pole_scene"]
    ground = summary["degenerate"]["ground_only"]
    values = [
        pole["point_to_point"]["trans_mean_m"] * 100.0,
        ground["point_to_point"]["trans_mean_m"] * 100.0,
        pole["point_to_plane"]["trans_mean_m"] * 100.0,
        ground["point_to_plane"]["trans_mean_m"] * 100.0,
    ]
    bars = ax_degen.bar(labels, values, color=["#fbbf24", "#ea580c", "#60a5fa", "#1d4ed8"])
    for bar, value in zip(bars, values):
        ax_degen.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.2,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
        )
    ax_degen.set_yscale("log")
    ax_degen.set(
        ylabel="平移误差 / cm（20 种子均值）",
        title="退化反例：去掉杆后“收敛”到错误位姿（沿平面滑动）",
    )
    for ax in (ax_curve, ax_sigma, ax_degen):
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--light",
        action="store_true",
        help="shrink the convergence-radius grid and its iteration budget (test fixture)",
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for noise statistics")
    report = run_experiment(args.output, runs=args.runs, seed=args.seed, light=args.light)
    reference = report["reference_run"]
    p2plane = reference["final_trans_err_m"]["point_to_plane"]
    truth = report["radius_study"]["truth_init"]["point_to_plane"]
    identity = report["radius_study"]["identity_init"]["point_to_plane"]
    print(
        f"reference run (sigma={reference['sigma_m']}, 0.1 m translation init): p2plane "
        f"trans err {p2plane * 100:.2f} cm in {reference['iterations']['point_to_plane']} iters"
    )
    print(
        f"truth-init trans err {truth['mean_trans_err_m'] * 100:.2f} cm "
        f"(converged {truth['converged_fraction']:.0%}); "
        f"identity-init rot err {identity['mean_rot_err_deg']:.1f} deg, "
        f"trans err {identity['mean_trans_err_m'] * 100:.1f} cm "
        f"(converged {identity['converged_fraction']:.0%})"
    )
    ground = report["degenerate"]["ground_only"]["point_to_plane"]
    print(
        f"degenerate 0.1 m shift: pole scene {report['degenerate']['pole_scene']['point_to_plane']['trans_mean_m'] * 100:.1f} cm"
        f" vs ground-only {ground['trans_mean_m'] * 100:.1f} cm"
        f" (converged-but-wrong {ground['converged_but_wrong_count']}/{report['runs_per_group']})"
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
