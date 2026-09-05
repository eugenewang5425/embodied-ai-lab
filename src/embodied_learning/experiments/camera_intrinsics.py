"""Lesson 25: camera intrinsics K from a synthetic chessboard (minimal Zhang).

Lessons 22-23 always *assumed* the intrinsics K = [600, 600, 320, 240]. This
lesson estimates K itself: one 9x6-corner planar chessboard (square 0.03 m) is
observed by the lesson-22 look-at camera from M different poses (each pose
sees the plane at a different orientation - Zhang's method needs at least two
NON-parallel planes). For every view the known board-plane coordinates are put
in correspondence with the "detected" corner pixels (true projection + Gaussian
noise sigma), one homography H per view is fitted by normalized DLT, the
v-vector constraints v12 and v11 - v22 are stacked over views, and the matrix
B = K^-T K^-1 (reduced model: fx = fy = f, skew = 0) is solved in closed form.
Per-view extrinsics follow from H and K-hat.

Checks:
(1) three-level comparison: naive guessed K vs true K vs estimated K, via the
    reprojection RMS of the board corners and the project->unproject round trip
    of one known 3D probe point;
(2) sweep M in {2, 3, 5, 10, 20} images x sigma in {0, 0.5, 1, 2} px x seeds;
(3) mechanism: |f_hat - f| ~ C/sqrt(M) * sigma_px, the constant C is reported;
(4) degenerate counter-examples: (a) identical camera orientation (pure
    translation => all board planes parallel) and (b) a fixed-tilt orbit
    around the board centre (found while designing the pool: the plane normal
    is then identical in every camera frame, so this is the same parallel-
    plane degeneracy). Both make the v-system rank deficient; the guard must
    catch them and the forced solutions are recorded.

No cv2, no scipy least squares, no calibration library: numpy SVD/lstsq only.
Overall nonlinear refinement is deliberately NOT done (recorded honestly).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pinhole_projection import (
    CX_PX,
    CY_PX,
    FOCAL_PX,
    HEIGHT_PX,
    K_INTRINSIC,
    WIDTH_PX,
    look_at,
    to_camera,
)

EXPERIMENT = "camera_intrinsics_zhang"
K_TRUE = K_INTRINSIC  # the lesson-22 intrinsics stop being "given" in this lesson
# A plausible but WRONG default guess (uncalibrated baseline): right cx, wrong
# focal AND wrong principal row (the true cy is 240, not 320).
NAIVE_FOCAL_PX = 500.0
NAIVE_CX_PX = 320.0
NAIVE_CY_PX = 320.0
NAIVE_K = np.array(
    [
        [NAIVE_FOCAL_PX, 0.0, NAIVE_CX_PX],
        [0.0, NAIVE_FOCAL_PX, NAIVE_CY_PX],
        [0.0, 0.0, 1.0],
    ]
)
# Board: 9x6 inner corners, square 0.03 m, centred on its own plane frame
# (X_d, Y_d, 0); the board plane IS the world z = 0 plane of this lesson.
BOARD_COLS, BOARD_ROWS = 9, 6
SQUARE_M = 0.03
M_VALUES = (2, 3, 5, 10, 20)
SIGMA_VALUES_PX = (0.0, 0.5, 1.0, 2.0)
REFERENCE_M = 5
REFERENCE_SIGMA_PX = 1.0
DEFAULT_RUNS = 20
DEFAULT_SEED = 0
# A calibration image is only usable when the WHOLE board is well inside it.
IMAGE_MARGIN_PX = 8.0
MIN_CORNER_DEPTH_M = 0.25
# Round-trip probe: one known 3D point 10 cm above the board plane.
PROBE_POINT_M = np.array([0.06, 0.04, 0.10])
# Pose pool: 8 azimuths x 3 well-separated tilts, one working distance each;
# filtered by board visibility. Orbits at a FIXED tilt are themselves a
# degeneracy (the plane normal is then identical in every camera frame, i.e.
# all planes parallel - the pilot run hit rank 2 exactly there), so the three
# tilt bands are 15 degrees apart and every draw must mix them (see
# MIN_NORMAL_ANGLE_DEG).
POOL_TILTS_DEG = (12.0, 27.0, 42.0)
POOL_AZIMUTHS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
POOL_DISTANCES_M = (0.55, 0.67, 0.79)
# A draw of views is usable only when two board-plane normals (in the camera
# frame) differ by at least this angle; parallel planes are Zhang's known
# degenerate configuration and make the v-system rank deficient.
MIN_NORMAL_ANGLE_DEG = 10.0
# Fixed noiseless sanity views: indices span all three tilt bands.
SIGMA_ZERO_VIEWS = (0, 8, 16, 1, 9)
# Degenerate counter-example A: identical orientation, eyes translated only.
DEGENERATE_FORWARD = np.array([0.12, 0.06, -1.0])
DEGENERATE_EYES_M = (
    (0.0, 0.0, 0.62),
    (0.05, 0.0, 0.62),
    (-0.05, 0.0, 0.62),
    (0.0, 0.045, 0.62),
    (0.0, -0.045, 0.62),
    (0.04, 0.03, 0.62),
    (-0.04, -0.03, 0.62),
    (0.05, -0.04, 0.62),
)
RANK_TOL = 1e-10  # singular values below s_max * RANK_TOL count as zero


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def board_corners(cols=BOARD_COLS, rows=BOARD_ROWS, square=SQUARE_M):
    """Inner-corner plane coordinates (X_d, Y_d, 0), centred on the board."""
    xs = (np.arange(cols) - (cols - 1) / 2.0) * square
    ys = (np.arange(rows) - (rows - 1) / 2.0) * square
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)])


def project_points(points, rotation, translation, intrinsic):
    """Vectorized pinhole projection of [N, 3] points (no culling here)."""
    camera = to_camera(points, rotation, translation)
    if np.any(camera[:, 2] <= 0.0):
        raise ValueError("Point behind the camera")
    homogeneous = intrinsic @ camera.T
    return (homogeneous[:2] / homogeneous[2]).T


def _normalise_2d(points):
    """Hartley similarity: centroid to origin, mean distance to sqrt(2)."""
    centroid = points.mean(axis=0)
    shift = np.linalg.norm(points - centroid, axis=1).mean()
    if shift < 1e-12:
        raise ValueError("Degenerate point configuration")
    scale = np.sqrt(2.0) / shift
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = (transform @ homogeneous.T).T
    return transformed[:, :2], transform


def homography_dlt(plane_xy, pixels):
    """Planar homography from >=4 correspondences by normalized DLT.

    Hartley normalization is what keeps the least-squares well conditioned:
    without it the pixel columns (~600) and the plane columns (~0.1) differ by
    three orders of magnitude and the SVD is dominated by scale, not geometry.
    """
    plane_xy = np.asarray(plane_xy, dtype=float)
    pixels = np.asarray(pixels, dtype=float)
    if plane_xy.shape != pixels.shape or plane_xy.ndim != 2 or plane_xy.shape[1] != 2:
        raise ValueError("Need matching [N, 2] plane and pixel coordinates")
    if len(plane_xy) < 4:
        raise ValueError("A homography needs at least four correspondences")
    plane_n, transform_p = _normalise_2d(plane_xy)
    pixel_n, transform_i = _normalise_2d(pixels)
    rows = []
    for (x, y), (u, v) in zip(plane_n, pixel_n):
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    homography_n = vt[-1].reshape(3, 3)
    homography = np.linalg.inv(transform_i) @ homography_n @ transform_p
    if abs(homography[2, 2]) < 1e-15:
        raise ValueError("Homography normalization failed (H[2,2] ~ 0)")
    return homography / homography[2, 2]


def _zhang_rows(homography):
    """Two constraint rows of the reduced (fx=fy, skew=0) model for one view.

    With B = K^-T K^-1 = [[b11, 0, b13], [0, b11, b23], [b13, b23, b33]] the
    rotation orthonormality r1 . r2 = 0 and |r1| = |r2| give
    h1' B h2 = 0 and h1' B h1 - h2' B h2 = 0 for h_i = column i of H.
    """
    h1, h2 = homography[:, 0], homography[:, 1]
    row_orth = np.array(
        [
            h1[0] * h2[0] + h1[1] * h2[1],
            h1[0] * h2[2] + h1[2] * h2[0],
            h1[1] * h2[2] + h1[2] * h2[1],
            h1[2] * h2[2],
        ]
    )
    row_norm = np.array(
        [
            h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2,
            2.0 * (h1[0] * h1[2] - h2[0] * h2[2]),
            2.0 * (h1[1] * h1[2] - h2[1] * h2[2]),
            h1[2] ** 2 - h2[2] ** 2,
        ]
    )
    return np.vstack([row_orth, row_norm])


def intrinsic_from_homographies(homographies, *, check_rank=True):
    """Closed-form (f, cx, cy) from >= 2 homographies via B = K^-T K^-1.

    b is the SVD null vector of the stacked v-system (4 unknowns, defined up
    to scale => rank 3 needed, i.e. at least two non-parallel planes). The
    scale is fixed by the bottom-right block of B. When `check_rank` is on, a
    rank-deficient system (parallel planes, single view) raises a ValueError
    carrying `.rank`, `.cond` and `.singular_values` for honest reporting.
    """
    homographies = list(homographies)
    if len(homographies) < 2:
        raise ValueError("Zhang calibration needs at least two views")
    system = np.vstack([_zhang_rows(h) for h in homographies])
    _, singular, vt = np.linalg.svd(system)
    b = vt[-1]
    if b[0] < 0:
        b = -b
    cond = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    rank = int(np.sum(singular > singular[0] * RANK_TOL))

    def fail(message):
        error = ValueError(message)
        error.rank = rank
        error.cond = cond
        error.singular_values = singular
        raise error

    if check_rank and rank < 3:
        fail(
            f"Degenerate view set: v-system rank {rank} < 3 "
            f"(cond {cond:.3e}); planes parallel or orientations repeated"
        )
    b11, b13, b23, b33 = b
    if b11 <= 0.0:
        fail("Ill-conditioned B: negative b11 (focal length would be imaginary)")
    denominator = b33 - (b13**2 + b23**2) / b11
    if denominator <= 0.0:
        fail("Ill-conditioned B: non-positive scale denominator")
    scale = 1.0 / denominator  # B_true = scale * B_solved
    focal = float(1.0 / np.sqrt(scale * b11))
    cx = float(-b13 / b11)
    cy = float(-b23 / b11)
    return {"f": focal, "cx": cx, "cy": cy, "cond": cond, "rank": rank, "b": b}


def intrinsic_matrix(f, cx, cy):
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def decompose_extrinsics(homography, intrinsic):
    """Per-view pose [R | t] from H and a known K (H = s K [r1 r2 t]).

    r1, r2 are the COLUMNS of R (board x/y axes in camera frame), so R is
    assembled by column stacking; the third column is r1 x r2 (right-handed).
    """
    inverse = np.linalg.inv(intrinsic)
    h1, h2, h3 = homography[:, 0], homography[:, 1], homography[:, 2]
    lambda1 = float(np.linalg.norm(inverse @ h1))
    lambda2 = float(np.linalg.norm(inverse @ h2))
    r1 = (inverse @ h1) / lambda1
    r2 = (inverse @ h2) / lambda2
    rotation = np.column_stack([r1, r2, np.cross(r1, r2)])
    translation = (inverse @ h3) / lambda1
    orthogonality = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    scale_gap = float(abs(lambda1 - lambda2) / lambda1)
    return {
        "rotation": rotation,
        "translation": translation,
        "orthogonality_error": orthogonality,
        "scale_gap": scale_gap,
    }


def reprojection_rms(plane_points, extrinsics, intrinsic, detected):
    """RMS pixel error of corners re-projected through (K, per-view poses)."""
    squared = []
    for extrinsic, pixels in zip(extrinsics, detected):
        projected = project_points(
            plane_points, extrinsic["rotation"], extrinsic["translation"], intrinsic
        )
        squared.append(np.sum((projected - pixels) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def reprojection_rms_known_pose(plane_points, true_poses, intrinsic, detected):
    """RMS pixel error with the TRUE poses, isolating the quality of K.

    Reprojecting through the *decomposed* poses is self-consistent for ANY K
    (a wrong K is absorbed by wrong poses, and only the homography residual
    remains), so it cannot detect a wrong K. Holding the poses at the true
    values removes that freedom: a wrong K shows up in full.
    """
    squared = []
    for (rotation, translation), pixels in zip(true_poses, detected):
        projected = project_points(plane_points, rotation, translation, intrinsic)
        squared.append(np.sum((projected - pixels) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(squared))))


def roundtrip_error(probe, true_poses, estimated_poses, intrinsic):
    """Pixel -> ray -> world with estimated K and pose for one known 3D point.

    The probe pixel and its camera depth come from the TRUE camera (like a
    depth sensor); the unprojection uses the ESTIMATED K and pose, so the
    world error mixes intrinsic and extrinsic estimation error.
    """
    probe = np.asarray(probe, dtype=float)
    errors = []
    for (rotation, translation), estimated in zip(true_poses, estimated_poses):
        pixel = project_points(probe[None], rotation, translation, K_TRUE)[0]
        depth = to_camera(probe[None], rotation, translation)[0, 2]
        ray = np.linalg.inv(intrinsic) @ np.array([pixel[0], pixel[1], 1.0])
        camera_hat = depth * ray
        world_hat = estimated["rotation"].T @ (camera_hat - estimated["translation"])
        errors.append(float(np.linalg.norm(world_hat - probe)))
    return float(np.mean(errors))


def build_pose_pool(plane_points):
    """Deterministic pose pool; keep poses seeing the whole board with margin."""
    eyes, targets = [], []
    index = 0
    for tilt_deg in POOL_TILTS_DEG:
        for azimuth_deg in POOL_AZIMUTHS_DEG:
            tilt = np.deg2rad(tilt_deg)
            azimuth = np.deg2rad(azimuth_deg)
            distance = POOL_DISTANCES_M[index % len(POOL_DISTANCES_M)]
            direction = np.array(
                [
                    np.sin(tilt) * np.cos(azimuth),
                    np.sin(tilt) * np.sin(azimuth),
                    np.cos(tilt),
                ]
            )
            eyes.append(distance * direction)
            targets.append(np.zeros(3))
            index += 1
    keep = []
    for eye in eyes:
        rotation, translation = look_at(eye, np.zeros(3))
        camera = to_camera(plane_points, rotation, translation)
        pixels = project_points(plane_points, rotation, translation, K_TRUE)
        inside = (
            (pixels[:, 0] >= IMAGE_MARGIN_PX)
            & (pixels[:, 0] <= WIDTH_PX - IMAGE_MARGIN_PX)
            & (pixels[:, 1] >= IMAGE_MARGIN_PX)
            & (pixels[:, 1] <= HEIGHT_PX - IMAGE_MARGIN_PX)
        )
        keep.append(bool(inside.all()) and float(camera[:, 2].min()) > MIN_CORNER_DEPTH_M)
    pool_eye = np.asarray(eyes)[keep]
    pool_target = np.asarray(targets)[keep]
    return pool_eye, pool_target, len(eyes)


def pool_poses(pool_eye, pool_target, indices):
    poses = [look_at(pool_eye[i], pool_target[i]) for i in indices]
    return [(rotation, translation) for rotation, translation in poses]


def plane_normal_camera(rotation):
    """Board-plane normal (world z) expressed in the camera frame."""
    return rotation[:, 2]


def _max_normal_angle_deg(normals):
    unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine).max()))


def sample_pose_indices(pool_normals, m_count, rng):
    """Draw M distinct pool poses with non-parallel planes, Zhang's requirement.

    Resampling mirrors lesson 23's control-point span rule: a draw whose
    board-plane normals (camera frame) are all parallel - e.g. two views from
    the same tilt band of the pool, or any set of pure translations - is
    exactly the degenerate configuration and is rejected before fitting.
    Returns (indices, tries).
    """
    pool_size = len(pool_normals)
    if not 2 <= m_count <= pool_size:
        raise ValueError("Need 2 <= M <= pool size")
    for tries in range(1, 1001):
        indices = np.sort(rng.choice(pool_size, size=m_count, replace=False))
        if _max_normal_angle_deg(pool_normals[indices]) >= MIN_NORMAL_ANGLE_DEG:
            return indices, tries
    raise RuntimeError("No pose draw with sufficiently non-parallel planes")


def build_degenerate_views(plane_points):
    """Same orientation for every view (pure translation) => parallel planes."""
    forward = DEGENERATE_FORWARD / np.linalg.norm(DEGENERATE_FORWARD)
    poses = []
    for eye in np.asarray(DEGENERATE_EYES_M, dtype=float):
        rotation, translation = look_at(eye, eye + forward)
        poses.append((rotation, translation))
    pixels = [
        project_points(plane_points, rotation, translation, K_TRUE)
        for rotation, translation in poses
    ]
    return poses, pixels


def build_orbit_views(plane_points):
    """Fixed-tilt orbit around the board centre: normals identical => parallel.

    This is the degeneracy found while designing the pool: cameras circling
    the board at one elevation see the plane under the SAME normal
    (0, -sin(tilt), -cos(tilt)) in every camera frame, so the v-system stays
    rank 2 no matter how many views are added.
    """
    tilt = np.deg2rad(27.0)
    poses = []
    for azimuth_deg, distance in zip(
        (0.0, 45.0, 90.0, 135.0, 180.0), POOL_DISTANCES_M + (0.55, 0.67)
    ):
        azimuth = np.deg2rad(azimuth_deg)
        eye = distance * np.array(
            [np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)]
        )
        poses.append(look_at(eye, np.zeros(3)))
    pixels = [
        project_points(plane_points, rotation, translation, K_TRUE)
        for rotation, translation in poses
    ]
    return poses, pixels


def _mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def _sweep_cell(
    plane_points, pool_pixels, pool_normals, pool_eye, pool_target, runs, seed, m_index, sigma_index
):
    """One (M, sigma) group: pose draw + corner noise redrawn per repetition."""
    m_count = M_VALUES[m_index]
    sigma = SIGMA_VALUES_PX[sigma_index]
    f_hat, cx_hat, cy_hat = [], [], []
    reproj_est, reproj_naive, reproj_true = [], [], []
    trip_est, trip_naive, trip_true = [], [], []
    failures = 0
    resample_total = 0
    for repetition in range(runs):
        rng = np.random.default_rng([seed, m_index, sigma_index, repetition])
        indices, tries = sample_pose_indices(pool_normals, m_count, rng)
        resample_total += tries - 1
        poses = pool_poses(pool_eye, pool_target, indices)
        detected = pool_pixels[indices] + rng.normal(0.0, sigma, pool_pixels[indices].shape)
        homographies = [homography_dlt(plane_points[:, :2], pixels) for pixels in detected]
        try:
            estimate = intrinsic_from_homographies(homographies)
        except ValueError:
            failures += 1
            continue
        k_est = intrinsic_matrix(estimate["f"], estimate["cx"], estimate["cy"])
        extrinsics = [decompose_extrinsics(h, k_est) for h in homographies]
        naive = [decompose_extrinsics(h, NAIVE_K) for h in homographies]
        oracle = [decompose_extrinsics(h, K_TRUE) for h in homographies]
        f_hat.append(estimate["f"])
        cx_hat.append(estimate["cx"])
        cy_hat.append(estimate["cy"])
        reproj_est.append(reprojection_rms_known_pose(plane_points, poses, k_est, detected))
        reproj_naive.append(reprojection_rms_known_pose(plane_points, poses, NAIVE_K, detected))
        reproj_true.append(reprojection_rms_known_pose(plane_points, poses, K_TRUE, detected))
        trip_est.append(roundtrip_error(PROBE_POINT_M, poses, extrinsics, k_est))
        trip_naive.append(roundtrip_error(PROBE_POINT_M, poses, naive, NAIVE_K))
        trip_true.append(roundtrip_error(PROBE_POINT_M, poses, oracle, K_TRUE))

    f_errors = [abs(value - FOCAL_PX) for value in f_hat]
    return {
        "f_mean": _mean_or_nan(f_hat),
        "f_std": float(np.std(f_hat, ddof=1)) if len(f_hat) > 1 else 0.0,
        "f_err_mean": _mean_or_nan(f_errors),
        "f_err_median": float(np.median(f_errors)) if f_errors else float("nan"),
        "cx_mean": _mean_or_nan(cx_hat),
        "cx_std": float(np.std(cx_hat, ddof=1)) if len(cx_hat) > 1 else 0.0,
        "cx_err_mean": _mean_or_nan([abs(value - CX_PX) for value in cx_hat]),
        "cy_mean": _mean_or_nan(cy_hat),
        "cy_std": float(np.std(cy_hat, ddof=1)) if len(cy_hat) > 1 else 0.0,
        "cy_err_mean": _mean_or_nan([abs(value - CY_PX) for value in cy_hat]),
        "reproj_est": _mean_or_nan(reproj_est),
        "reproj_naive": _mean_or_nan(reproj_naive),
        "reproj_true": _mean_or_nan(reproj_true),
        "trip_est": _mean_or_nan(trip_est),
        "trip_naive": _mean_or_nan(trip_naive),
        "trip_true": _mean_or_nan(trip_true),
        "failures": failures,
        "resamples": resample_total,
    }


def _reference_bundle(plane_points, pool_pixels, pool_normals, pool_eye, pool_target, seed):
    """One stored draw (reference M, sigma, run 0) for the demo and the doc."""
    rng = np.random.default_rng(
        [seed, 9001, REFERENCE_M, SIGMA_VALUES_PX.index(REFERENCE_SIGMA_PX)]
    )
    indices, _tries = sample_pose_indices(pool_normals, REFERENCE_M, rng)
    poses = pool_poses(pool_eye, pool_target, indices)
    clean = pool_pixels[indices]
    noisy = clean + rng.normal(0.0, REFERENCE_SIGMA_PX, clean.shape)
    homographies = [homography_dlt(plane_points[:, :2], pixels) for pixels in noisy]
    estimate = intrinsic_from_homographies(homographies)
    k_est = intrinsic_matrix(estimate["f"], estimate["cx"], estimate["cy"])
    extrinsics = [decompose_extrinsics(h, k_est) for h in homographies]
    naive = [decompose_extrinsics(h, NAIVE_K) for h in homographies]
    oracle = [decompose_extrinsics(h, K_TRUE) for h in homographies]
    return {
        "pose_index": indices,
        "poses": poses,
        "normal_spread_deg": _max_normal_angle_deg(pool_normals[indices]),
        "pixels_clean": clean,
        "pixels_noisy": noisy,
        "homographies": homographies,
        "k_est": k_est,
        "f": estimate["f"],
        "cx": estimate["cx"],
        "cy": estimate["cy"],
        "reproj_est": reprojection_rms_known_pose(plane_points, poses, k_est, noisy),
        "reproj_naive": reprojection_rms_known_pose(plane_points, poses, NAIVE_K, noisy),
        "reproj_true": reprojection_rms_known_pose(plane_points, poses, K_TRUE, noisy),
        "reproj_decomposed_est": reprojection_rms(plane_points, extrinsics, k_est, noisy),
        "reproj_decomposed_naive": reprojection_rms(plane_points, naive, NAIVE_K, noisy),
        "reproj_decomposed_true": reprojection_rms(plane_points, oracle, K_TRUE, noisy),
        "trip_est": roundtrip_error(PROBE_POINT_M, poses, extrinsics, k_est),
        "trip_naive": roundtrip_error(PROBE_POINT_M, poses, naive, NAIVE_K),
        "trip_true": roundtrip_error(PROBE_POINT_M, poses, oracle, K_TRUE),
        "orthogonality_max": float(
            max(extrinsic["orthogonality_error"] for extrinsic in extrinsics)
        ),
        "scale_gap_max": float(max(extrinsic["scale_gap"] for extrinsic in extrinsics)),
    }


def _degenerate_record(plane_points):
    """Parallel-plane counter-examples: guard must fire; forced solve recorded."""
    cases = {}
    for name, builder in (
        ("parallel_translation", build_degenerate_views),
        ("parallel_orbit_fixed_tilt", build_orbit_views),
    ):
        _, pixels = builder(plane_points)
        homographies = [homography_dlt(plane_points[:, :2], view) for view in pixels]
        guarded_error = None
        try:
            intrinsic_from_homographies(homographies)
        except ValueError as error:
            guarded_error = error
        forced = None
        forced_error = None
        try:
            forced = intrinsic_from_homographies(homographies, check_rank=False)
        except ValueError as error:
            forced_error = str(error)
        cases[name] = {
            "views": len(homographies),
            "sigma_px": 0.0,
            "rank": int(guarded_error.rank)
            if guarded_error
            else (forced["rank"] if forced else -1),
            "cond": float(guarded_error.cond)
            if guarded_error
            else (float(forced["cond"]) if forced else float("inf")),
            "guard_triggered": guarded_error is not None,
            "guard_message": str(guarded_error) if guarded_error else "",
            "forced_f": None if forced is None else forced["f"],
            "forced_f_error": None if forced is None else abs(forced["f"] - FOCAL_PX),
            "forced_note": ""
            if forced is not None
            else f"min-norm b yields no valid K: {forced_error}",
        }
    return cases


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two repetitions for noise statistics")
    plane_points = board_corners()
    pool_eye, pool_target, pool_total = build_pose_pool(plane_points)
    if len(pool_eye) < max(M_VALUES):
        raise RuntimeError(f"Pose pool too small after visibility filter: {len(pool_eye)}")
    pool_poses_all = pool_poses(pool_eye, pool_target, range(len(pool_eye)))
    pool_pixels = np.asarray(
        [
            project_points(plane_points, rotation, translation, K_TRUE)
            for rotation, translation in pool_poses_all
        ]
    )
    pool_normals = np.asarray([plane_normal_camera(rotation) for rotation, _ in pool_poses_all])
    # Noiseless sanity: fixed views spanning all three tilt bands recover K.
    exact_homographies = [
        homography_dlt(plane_points[:, :2], pool_pixels[i]) for i in SIGMA_ZERO_VIEWS
    ]
    exact = intrinsic_from_homographies(exact_homographies)
    sigma_zero_f_err = abs(exact["f"] - FOCAL_PX)
    sigma_zero_cx_err = abs(exact["cx"] - CX_PX)
    sigma_zero_cy_err = abs(exact["cy"] - CY_PX)

    m_count, sigma_count = len(M_VALUES), len(SIGMA_VALUES_PX)
    keys = (
        "f_mean",
        "f_std",
        "f_err_mean",
        "f_err_median",
        "cx_mean",
        "cx_std",
        "cx_err_mean",
        "cy_mean",
        "cy_std",
        "cy_err_mean",
        "reproj_est",
        "reproj_naive",
        "reproj_true",
        "trip_est",
        "trip_naive",
        "trip_true",
    )
    table = {key: np.zeros((m_count, sigma_count)) for key in keys}
    failures = np.zeros((m_count, sigma_count), dtype=int)
    resamples = np.zeros((m_count, sigma_count), dtype=int)
    for m_index in range(m_count):
        for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
            stats = _sweep_cell(
                plane_points,
                pool_pixels,
                pool_normals,
                pool_eye,
                pool_target,
                runs,
                seed,
                m_index,
                sigma_index,
            )
            for key in keys:
                table[key][m_index, sigma_index] = stats[key]
            failures[m_index, sigma_index] = stats["failures"]
            resamples[m_index, sigma_index] = stats["resamples"]
    # Mechanism: |f_hat - f| ~ C / sqrt(M) * sigma => C = err * sqrt(M) / sigma
    # should be roughly the same constant across M at fixed sigma.
    mechanism_c = np.full((m_count, sigma_count), np.nan)
    for m_index, m_value in enumerate(M_VALUES):
        for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
            if sigma > 0.0:
                mechanism_c[m_index, sigma_index] = (
                    table["f_err_mean"][m_index, sigma_index] * np.sqrt(m_value) / sigma
                )
    reference = _reference_bundle(
        plane_points, pool_pixels, pool_normals, pool_eye, pool_target, seed
    )
    degenerate = _degenerate_record(plane_points)

    archive = {
        "board_corners_m": plane_points,
        "pool_eye_m": pool_eye,
        "pool_target_m": pool_target,
        "m_values": np.array(M_VALUES),
        "sigma_values_px": np.array(SIGMA_VALUES_PX),
        "runs_per_group": np.array(runs),
        "k_est_f_mean_px": table["f_mean"],
        "k_est_f_std_px": table["f_std"],
        "k_est_f_err_mean_px": table["f_err_mean"],
        "k_est_f_err_median_px": table["f_err_median"],
        "k_est_cx_mean_px": table["cx_mean"],
        "k_est_cx_std_px": table["cx_std"],
        "k_est_cx_err_mean_px": table["cx_err_mean"],
        "k_est_cy_mean_px": table["cy_mean"],
        "k_est_cy_std_px": table["cy_std"],
        "k_est_cy_err_mean_px": table["cy_err_mean"],
        "reproj_rms_est_px": table["reproj_est"],
        "reproj_rms_naive_px": table["reproj_naive"],
        "reproj_rms_true_px": table["reproj_true"],
        "roundtrip_est_m": table["trip_est"],
        "roundtrip_naive_m": table["trip_naive"],
        "roundtrip_true_m": table["trip_true"],
        "fit_failures": failures,
        "resample_attempts": resamples,
        "mechanism_c_px": mechanism_c,
        "ref_pose_index": reference["pose_index"],
        "ref_rotation": np.asarray([pose[0] for pose in reference["poses"]]),
        "ref_translation": np.asarray([pose[1] for pose in reference["poses"]]),
        "ref_pixels_clean": reference["pixels_clean"],
        "ref_pixels_noisy": reference["pixels_noisy"],
        "ref_homographies": np.asarray(reference["homographies"]),
        "ref_k_est": np.array([reference["f"], reference["cx"], reference["cy"]]),
        "ref_normal_spread_deg": np.array(reference["normal_spread_deg"]),
        "probe_point_m": PROBE_POINT_M,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "model": "Zhang planar calibration, reduced intrinsics (fx = fy = f, skew = 0)",
        "pipeline": "normalized DLT homography -> v-vectors -> B = K^-T K^-1 -> K, per-view [R|t]",
        "nonlinear_refinement_done": False,
        "camera_model": "pinhole {u = K [R|t] X / z} (lesson-22 look-at camera)",
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "k_true": K_TRUE.tolist(),
        "f_true_px": FOCAL_PX,
        "principal_point_true_px": [CX_PX, CY_PX],
        "naive_k": NAIVE_K.tolist(),
        "naive_focal_px": NAIVE_FOCAL_PX,
        "naive_principal_point_px": [NAIVE_CX_PX, NAIVE_CY_PX],
        "board": {
            "inner_corners": [BOARD_COLS, BOARD_ROWS],
            "square_m": SQUARE_M,
            "corners_total": len(plane_points),
        },
        "probe_point_m": PROBE_POINT_M.tolist(),
        "pose_pool": {
            "candidates": pool_total,
            "kept": len(pool_eye),
            "tilts_deg": list(POOL_TILTS_DEG),
            "azimuths_deg": list(POOL_AZIMUTHS_DEG),
            "distances_m": list(POOL_DISTANCES_M),
            "image_margin_px": IMAGE_MARGIN_PX,
            "min_corner_depth_m": MIN_CORNER_DEPTH_M,
            "min_normal_angle_deg": MIN_NORMAL_ANGLE_DEG,
            "sigma_zero_views": list(SIGMA_ZERO_VIEWS),
        },
        "m_values": list(M_VALUES),
        "sigma_values_px": list(SIGMA_VALUES_PX),
        "reference_group": {
            "m": REFERENCE_M,
            "sigma_px": REFERENCE_SIGMA_PX,
            "f_px": reference["f"],
            "cx_px": reference["cx"],
            "cy_px": reference["cy"],
            "f_error_px": abs(reference["f"] - FOCAL_PX),
            "cx_error_px": abs(reference["cx"] - CX_PX),
            "cy_error_px": abs(reference["cy"] - CY_PX),
            "normal_spread_deg": reference["normal_spread_deg"],
            "reproj_rms_est_px": reference["reproj_est"],
            "reproj_rms_naive_px": reference["reproj_naive"],
            "reproj_rms_true_px": reference["reproj_true"],
            "reproj_rms_decomposed_est_px": reference["reproj_decomposed_est"],
            "reproj_rms_decomposed_naive_px": reference["reproj_decomposed_naive"],
            "reproj_rms_decomposed_true_px": reference["reproj_decomposed_true"],
            "roundtrip_est_m": reference["trip_est"],
            "roundtrip_naive_m": reference["trip_naive"],
            "roundtrip_true_m": reference["trip_true"],
            "extrinsic_orthogonality_max": reference["orthogonality_max"],
            "extrinsic_scale_gap_max": reference["scale_gap_max"],
        },
        "sigma_zero": {
            "f_error_px": sigma_zero_f_err,
            "cx_error_px": sigma_zero_cx_err,
            "cy_error_px": sigma_zero_cy_err,
            "views": len(SIGMA_ZERO_VIEWS),
        },
        "k_est_f_mean_px": table["f_mean"].tolist(),
        "k_est_f_std_px": table["f_std"].tolist(),
        "k_est_f_err_mean_px": table["f_err_mean"].tolist(),
        "k_est_f_err_median_px": table["f_err_median"].tolist(),
        "k_est_cx_mean_px": table["cx_mean"].tolist(),
        "k_est_cx_std_px": table["cx_std"].tolist(),
        "k_est_cx_err_mean_px": table["cx_err_mean"].tolist(),
        "k_est_cy_mean_px": table["cy_mean"].tolist(),
        "k_est_cy_std_px": table["cy_std"].tolist(),
        "k_est_cy_err_mean_px": table["cy_err_mean"].tolist(),
        "reproj_rms_est_px": table["reproj_est"].tolist(),
        "reproj_rms_naive_px": table["reproj_naive"].tolist(),
        "reproj_rms_true_px": table["reproj_true"].tolist(),
        "roundtrip_est_m": table["trip_est"].tolist(),
        "roundtrip_naive_m": table["trip_naive"].tolist(),
        "roundtrip_true_m": table["trip_true"].tolist(),
        "fit_failures": failures.tolist(),
        "mechanism_c_px": mechanism_c.tolist(),
        "degenerate_case": degenerate,
        "resample_attempts": resamples.tolist(),
        "runs_per_group": runs,
        "base_seed": seed,
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            "experiments/camera_intrinsics.py": digest(Path(__file__).resolve()),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            (
                "Synthetic corners: no real corner detector, no lens distortion, "
                "no distortion coefficients estimated, no real images"
            ),
            (
                "Reduced intrinsics only (fx = fy = f, skew = 0); the full 5-parameter "
                "model and overall nonlinear refinement are deliberately not done"
            ),
            (
                "Every view sees the whole board with margin; partial views, "
                "outliers and robust losses are out of scope"
            ),
            (
                "The C/sqrt(M)*sigma mechanism check is an empirical scaling "
                "verified over 20 seeds, not an asymptotic theorem"
            ),
        ],
    }
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
    fig.suptitle("第二十五课 合成棋盘格张氏标定：从“给定的 K”到“估计出的 K”")
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax_view = fig.add_subplot(2, 3, 2)
    ax_rms = fig.add_subplot(2, 3, 3)
    ax_trip = fig.add_subplot(2, 3, 4)
    ax_m = fig.add_subplot(2, 3, 5)
    ax_mech = fig.add_subplot(2, 3, 6)

    corners = archive["board_corners_m"]
    pool_eye = archive["pool_eye_m"]
    ref_idx = archive["ref_pose_index"]
    ax3d.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c="#2563eb", s=8, label="棋盘角点")
    ax3d.scatter(pool_eye[:, 0], pool_eye[:, 1], pool_eye[:, 2], c="#94a3b8", s=14, label="姿态池")
    ref_eye = pool_eye[ref_idx]
    ax3d.scatter(ref_eye[:, 0], ref_eye[:, 1], ref_eye[:, 2], c="#dc2626", s=26, label="参考组位姿")
    for eye in ref_eye:
        ax3d.plot([eye[0], 0.0], [eye[1], 0.0], [eye[2], 0.0], color="#dc2626", lw=0.6, alpha=0.5)
    ax3d.set(
        xlabel="X_d / m",
        ylabel="Y_d / m",
        zlabel="高 / m",
        title=f"棋盘平面与相机姿态池（{len(pool_eye)} 个可用位姿）",
    )
    ax3d.view_init(elev=28, azim=-55)
    ax3d.legend(fontsize=8)

    corners_noisy = archive["ref_pixels_noisy"][0]
    k_est = intrinsic_matrix(*archive["ref_k_est"])
    rotation = archive["ref_rotation"][0]
    translation = archive["ref_translation"][0]
    projected = project_points(corners, rotation, translation, k_est)
    ax_view.add_patch(
        plt.Rectangle((0, 0), WIDTH_PX, HEIGHT_PX, fill=False, color="#0f172a", lw=1.0)
    )
    ax_view.scatter(
        corners_noisy[:, 0],
        HEIGHT_PX - corners_noisy[:, 1],
        marker="x",
        s=34,
        c="#dc2626",
        label="检测角点（含噪声）",
    )
    ax_view.scatter(
        projected[:, 0],
        HEIGHT_PX - projected[:, 1],
        s=10,
        c="#2563eb",
        label="估计 K 重投影",
    )
    ax_view.set_xlim(-10, WIDTH_PX + 10)
    ax_view.set_ylim(-10, HEIGHT_PX + 10)
    ax_view.set_aspect("equal")
    ax_view.set(
        xlabel="u / px",
        ylabel="v / px（已翻转）",
        title=f"参考组第 1 幅：M={REFERENCE_M}, σ={REFERENCE_SIGMA_PX} px",
    )
    ax_view.legend(fontsize=8, loc="lower left")

    ref = summary["reference_group"]
    labels = ("无标定 K（猜的）", "真值 K（先验）", "估计 K（本课）")
    values = (
        ref["reproj_rms_naive_px"],
        ref["reproj_rms_true_px"],
        ref["reproj_rms_est_px"],
    )
    bars = ax_rms.bar(labels, values, color=["#94a3b8", "#22c55e", "#2563eb"])
    ax_rms.set_yscale("log")
    for bar, value in zip(bars, values):
        ax_rms.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:.2f}",
            ha="center",
            fontsize=9,
        )
    ax_rms.set(ylabel="角点重投影 RMS / px", title="重投影误差：三水平对照（参考组）")

    trip = (ref["roundtrip_naive_m"], ref["roundtrip_true_m"], ref["roundtrip_est_m"])
    bars = ax_trip.bar(labels, trip, color=["#94a3b8", "#22c55e", "#2563eb"])
    ax_trip.set_yscale("log")
    for bar, value in zip(bars, trip):
        ax_trip.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.3,
            f"{value:.1e}",
            ha="center",
            fontsize=9,
        )
    probe = archive["probe_point_m"]
    ax_trip.set(
        ylabel="往返误差 / m",
        title=f"往返误差：已知 3D 探针点（z=+{probe[2]:.2f} m）",
    )

    m_values = archive["m_values"]
    f_err = archive["k_est_f_err_mean_px"]
    for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
        ax_m.plot(
            m_values,
            f_err[:, sigma_index],
            "-o",
            ms=4,
            label=f"σ={sigma} px" if sigma else "σ=0（数值精确）",
        )
    ax_m.set_yscale("log")
    ax_m.set(
        xlabel="图像数量 M",
        ylabel="焦距误差 |f_est − f| 均值 / px",
        title=f"f 估计误差随 M 收敛（{summary['runs_per_group']} 种子均值）",
    )
    ax_m.legend(fontsize=8)

    c_table = archive["mechanism_c_px"]
    for sigma_index, sigma in enumerate(SIGMA_VALUES_PX):
        if sigma == 0.0:
            continue
        ax_mech.plot(m_values, c_table[:, sigma_index], "-o", ms=4, label=f"σ={sigma} px")
    pooled = float(np.nanmedian(c_table[:, 1:]))
    ax_mech.axhline(pooled, color="#0f172a", ls="--", lw=1.2, label=f"中位 C≈{pooled:.0f} px")
    ax_mech.set(
        xlabel="图像数量 M",
        ylabel="C = 误差·√M / σ / px",
        title="机制核对：C 近似常数（误差 ∝ σ/√M）",
    )
    ax_mech.legend(fontsize=8)
    for ax in (ax_view, ax_rms, ax_trip, ax_m, ax_mech):
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for noise statistics")
    report = run_experiment(args.output, runs=args.runs, seed=args.seed)
    zero = report["sigma_zero"]
    ref = report["reference_group"]
    translation_case = report["degenerate_case"]["parallel_translation"]
    print(
        f"sigma=0 ({zero['views']} views): f err {zero['f_error_px']:.2e} px, "
        f"cx err {zero['cx_error_px']:.2e} px, cy err {zero['cy_error_px']:.2e} px"
    )
    print(
        f"reference M={ref['m']} sigma={ref['sigma_px']} px: f={ref['f_px']:.2f} "
        f"cx={ref['cx_px']:.2f} cy={ref['cy_px']:.2f}; reproj RMS "
        f"naive {ref['reproj_rms_naive_px']:.2f} / true {ref['reproj_rms_true_px']:.2f} / "
        f"est {ref['reproj_rms_est_px']:.2f} px"
    )
    print(
        f"degenerate parallel views: rank {translation_case['rank']}, "
        f"cond {translation_case['cond']:.2e}, "
        f"guard {'triggered' if translation_case['guard_triggered'] else 'NOT triggered'}, "
        f"forced f {translation_case['forced_f'] if translation_case['forced_f'] is not None else translation_case['forced_note']}"
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
