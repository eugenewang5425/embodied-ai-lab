"""Lesson 23: calibrating monocular relative depth to metric scale.

Lesson 22 showed that one view gives only rays: a monocular depth model can
at best output *relative* depth without metric scale. This lesson models that
output with the classic inverse-depth affine ambiguity

    r = a * (1/Z) + b          (a, b unknown and arbitrary),

then recovers metric depth Z_hat = a / (r - b) by ordinary least squares on
N control points whose true metric depth is known (sparse laser ranging /
GIS-style control points, drawn from the rendered depth map).

Scene: the same ground patch and vertical pole as lesson 22, rendered per
pixel with the same pinhole camera (K, look-at) via analytic ray casting -
no renderer, no learned model, no new dependency.

Checks:
(1) uncalibrated baseline (assuming a=1, b=0) vs perfect prior (true a, b)
    vs calibrated with N in {2, 3, 5, 10} control points;
(2) multiplicative relative-depth reading noise sigma in {0, 1%, 3%} on the
    control points only (the proxy field itself stays ideal), repeated seeds;
(3) errors stratified by depth (near/far at the 15%/85% depth quantiles):
    delta_Z ~ Z^2 * delta(1/Z) makes far pixels worse;
(4) mechanism check: with a fixed control set the sampling covariance of the
    fitted (a, b) matches the exact sandwich-formula prediction (verified with
    100x repetitions), and both covariances propagate through
    dZ ~ Z^2 * d(1/Z) into per-pixel std maps whose median ratio is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.experiments.pinhole_projection import (
    EYE,
    HEIGHT_PX,
    K_INTRINSIC,
    NEAR_PLANE_M,
    TARGET,
    WIDTH_PX,
    look_at,
    scene_points,
)

EXPERIMENT = "monocular_metric_calibration"
# Relative-depth proxy: r = A_TRUE*(1/Z) + B_TRUE. Any (a, b) pair describes
# the same geometry - that arbitrariness IS the "no metric scale" property.
A_TRUE = 5.0
B_TRUE = 0.2
# Uncalibrated baseline: a user who assumes the raw relative map is 1/Z in m.
NAIVE_A = 1.0
NAIVE_B = 0.0
# Dense surfaces reuse the lesson-22 sparse scene: the grid extent and the
# pole axis/height come from scene_points(); the sparse pole line becomes a
# solid cylinder (radius below) so it can occlude the ground behind it.
_SCENE_POINTS, _GROUND_POINTS, _POLE_POINTS = scene_points()
GRID_X_MIN_M = float(_GROUND_POINTS[:, 0].min())
GRID_X_MAX_M = float(_GROUND_POINTS[:, 0].max())
GRID_Y_MIN_M = float(_GROUND_POINTS[:, 1].min())
GRID_Y_MAX_M = float(_GROUND_POINTS[:, 1].max())
POLE_X_M = float(_POLE_POINTS[0, 0])
POLE_Y_M = float(_POLE_POINTS[0, 1])
POLE_TOP_M = float(_POLE_POINTS[:, 2].max())
POLE_RADIUS_M = 0.06
NEAR_QUANTILE = 0.15
FAR_QUANTILE = 0.85
N_VALUES = (2, 3, 5, 10)
SIGMA_VALUES = (0.0, 0.01, 0.03)  # relative noise on the control-point r reading
# Control points must span the inverse-depth range (like GIS control points
# must cover the area): otherwise two almost-equally-deep points make the
# 2x2 fit ill-conditioned and the noise can flip the slope sign.
MIN_INV_DEPTH_SPAN = 0.1
REFERENCE_N = 5
REFERENCE_SIGMA = 0.01
DEFAULT_RUNS = 20
DEFAULT_SEED = 0


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_depth_map():
    """Per-pixel metric depth (camera-frame z) of the lesson-22 scene.

    Rays are cast through every pixel centre with the lesson-22 pinhole model:
    X_cam = s * K^-1[u, v, 1], so the ray parameter s IS the camera depth.
    Surfaces: the ground plane z=0 over the grid extent, plus the pole as a
    vertical cylinder. A pixel is valid when a surface hit lies in front of
    the near plane (0.5 m); occlusion = the nearer hit wins.
    """
    rotation, _ = look_at(EYE, TARGET)
    cols = np.arange(WIDTH_PX, dtype=float)
    rows = np.arange(HEIGHT_PX, dtype=float)
    uu, vv = np.meshgrid(cols, rows)
    homogeneous = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
    rays_world = rotation.T @ (np.linalg.inv(K_INTRINSIC) @ homogeneous)
    wx, wy, wz = rays_world
    depth = np.full(uu.size, np.nan)
    pole = np.zeros(uu.size, dtype=bool)
    # Ground plane z = 0: eye_z + s * w_z = 0 needs w_z < 0 (looking down).
    towards_ground = wz < -1e-12
    s_ground = np.full(uu.size, np.nan)
    s_ground[towards_ground] = -EYE[2] / wz[towards_ground]
    ground_x = EYE[0] + s_ground * wx
    ground_y = EYE[1] + s_ground * wy
    on_grid = (
        towards_ground
        & (ground_x >= GRID_X_MIN_M)
        & (ground_x <= GRID_X_MAX_M)
        & (ground_y >= GRID_Y_MIN_M)
        & (ground_y <= GRID_Y_MAX_M)
    )
    depth[on_grid] = s_ground[on_grid]
    # Pole cylinder: (x - px)^2 + (y - py)^2 = radius^2 along the ray.
    offset_x, offset_y = EYE[0] - POLE_X_M, EYE[1] - POLE_Y_M
    quad_a = wx * wx + wy * wy
    quad_b = 2.0 * (wx * offset_x + wy * offset_y)
    quad_c = offset_x * offset_x + offset_y * offset_y - POLE_RADIUS_M**2
    disc = quad_b * quad_b - 4.0 * quad_a * quad_c
    hits = (quad_a > 1e-12) & (disc >= 0.0)
    s_pole = np.full(uu.size, np.nan)
    s_pole[hits] = (-quad_b[hits] - np.sqrt(disc[hits])) / (2.0 * quad_a[hits])
    with np.errstate(invalid="ignore"):
        pole_z = EYE[2] + s_pole * wz
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


def relative_from_depth(depth_map, valid):
    """The proxy monocular output: r = A_TRUE/Z + B_TRUE, arbitrary units."""
    relative = np.full(depth_map.shape, np.nan)
    relative[valid] = A_TRUE / depth_map[valid] + B_TRUE
    return relative


def metric_from_relative(relative, a, b, valid):
    """Invert the affine map: Z_hat = a / (r - b) on valid pixels."""
    offset = relative[valid] - b
    if np.any(offset <= 0.0):
        raise ValueError("Relative depth does not admit this offset b")
    metric = np.full(relative.shape, np.nan)
    metric[valid] = a / offset
    return metric


def sample_control_pixels(depth_map, valid, count, rng):
    """Draw count distinct valid pixels covering the inverse-depth range.

    Resampling until 1/Z spans at least MIN_INV_DEPTH_SPAN mirrors practice:
    control points must be spread over the scene, not bunched at one depth.
    """
    flat = np.flatnonzero(valid.ravel())
    if count < 2 or count > flat.size:
        raise ValueError("Need 2 <= count <= number of valid pixels")
    for _ in range(1000):
        chosen = rng.choice(flat, size=count, replace=False)
        depths = depth_map.ravel()[chosen]
        if np.ptp(1.0 / depths) >= MIN_INV_DEPTH_SPAN:
            rows, cols = np.unravel_index(chosen, depth_map.shape)
            return np.column_stack([cols, rows]).astype(float), depths
    raise RuntimeError("No well-spread control-point draw; relax MIN_INV_DEPTH_SPAN")


def fit_inverse_affine(inv_depth, relative):
    """Ordinary least squares r ~ a*(1/Z) + b; returns (a, b)."""
    u = np.asarray(inv_depth, dtype=float)
    r = np.asarray(relative, dtype=float)
    if u.ndim != 1 or u.size < 2:
        raise ValueError("Need at least two control points")
    if np.ptp(u) < 1e-12:
        raise ValueError("Control points must span the inverse-depth range")
    design = np.column_stack([u, np.ones_like(u)])
    coeffs, *_ = np.linalg.lstsq(design, r, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def affine_covariance(inv_depth, relative, sigma):
    """Exact covariance of (a, b) for linear OLS with noise std sigma*|r_i|.

    theta_hat - theta = (X'X)^-1 X' eps, so the sandwich formula is exact for
    this linear model (no Gaussian large-sample approximation needed).
    """
    u = np.asarray(inv_depth, dtype=float)
    design = np.column_stack([u, np.ones_like(u)])
    noise_var = (sigma * np.asarray(relative, dtype=float)) ** 2
    inverse_precision = np.linalg.inv(design.T @ design)
    middle = design.T @ (design * noise_var[:, None])
    return inverse_precision @ middle @ inverse_precision


def predicted_error_std(depth_map, valid, a_fit, covariance):
    """Linear propagation: dZ ~ Z^2 * |d(1/Z)|, d(1/Z) = -(db + u da)/a."""
    z = depth_map[valid]
    u = 1.0 / z
    var_a, var_b, cov_ab = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    var_u = np.maximum(var_b + u * u * var_a + 2.0 * u * cov_ab, 0.0) / a_fit**2
    predicted = np.full(depth_map.shape, np.nan)
    predicted[valid] = z * z * np.sqrt(var_u)
    return predicted


def _sweep_group(
    depth, valid, near_mask, far_mask, relative, runs, seed, n_points, n_index, sigma, sigma_index
):
    """One (N, sigma) group: resample control points and noise per repetition."""
    per_mean, per_max, per_near, per_far, per_signed = [], [], [], [], []
    failures = 0
    for repetition in range(runs):
        rng = np.random.default_rng([seed, n_index, sigma_index, repetition])
        _, ctrl_depth = sample_control_pixels(depth, valid, n_points, rng)
        ctrl_relative = A_TRUE / ctrl_depth + B_TRUE
        noisy = ctrl_relative * (1.0 + rng.normal(0.0, sigma, n_points))
        a_fit, b_fit = fit_inverse_affine(1.0 / ctrl_depth, noisy)
        if a_fit <= 0.0:
            failures += 1
            continue
        try:
            z_hat = metric_from_relative(relative, a_fit, b_fit, valid)
        except ValueError:
            failures += 1  # b_hat beyond the r range: no valid metric map
            continue
        error = np.abs(z_hat[valid] - depth[valid])
        signed_error = z_hat[valid] - depth[valid]
        per_mean.append(error.mean())
        per_max.append(error.max())
        per_near.append(error[near_mask[valid]].mean())
        per_far.append(error[far_mask[valid]].mean())
        per_signed.append(signed_error.mean())
    return {
        "mean": float(np.mean(per_mean)),
        "median": float(np.median(per_mean)),
        "std": float(np.std(per_mean, ddof=1)) if len(per_mean) > 1 else 0.0,
        "max": float(np.max(per_max)),
        "near": float(np.mean(per_near)),
        "far": float(np.mean(per_far)),
        "signed": float(np.mean(per_signed)),
        "failures": failures,
    }


def run_experiment(output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two repetitions for noise statistics")
    rendered = render_depth_map()
    depth, valid, pole = rendered["depth_m"], rendered["valid"], rendered["pole"]
    relative = relative_from_depth(depth, valid)
    q_near = float(np.quantile(depth[valid], NEAR_QUANTILE))
    q_far = float(np.quantile(depth[valid], FAR_QUANTILE))
    near_mask = valid & (depth <= q_near)
    far_mask = valid & (depth >= q_far)
    # Baseline 1: uncalibrated user (assumed a=1, b=0). Baseline 2: oracle.
    naive_map = metric_from_relative(relative, NAIVE_A, NAIVE_B, valid)
    naive_error = np.abs(naive_map[valid] - depth[valid])
    perfect_map = metric_from_relative(relative, A_TRUE, B_TRUE, valid)
    perfect_max = float(np.max(np.abs(perfect_map[valid] - depth[valid])))
    # Main sweep: control points resampled and noise redrawn per repetition.
    n_count, sigma_count = len(N_VALUES), len(SIGMA_VALUES)
    group = {
        key: np.zeros((n_count, sigma_count))
        for key in ("mean", "median", "std", "max", "near", "far", "signed")
    }
    failures = np.zeros((n_count, sigma_count), dtype=int)
    for n_index, n_points in enumerate(N_VALUES):
        for sigma_index, sigma in enumerate(SIGMA_VALUES):
            stats = _sweep_group(
                depth,
                valid,
                near_mask,
                far_mask,
                relative,
                runs,
                seed,
                n_points,
                n_index,
                sigma,
                sigma_index,
            )
            for key, values in group.items():
                values[n_index, sigma_index] = stats[key]
            failures[n_index, sigma_index] = stats["failures"]
    # Mechanism check with a FIXED control set per N: the sampling covariance
    # of the fitted (a, b) must match the exact sandwich prediction. Both
    # covariances are propagated through dZ ~ Z^2 * (db + u da)/a into
    # per-pixel std maps; their median ratio verifies the whole chain.
    # The check uses 100x the sweep repetitions: each realization is a cheap
    # 2-parameter fit, while a covariance estimate from only `runs` samples
    # carries ~sqrt(2/(runs-1)) relative error and cannot verify a formula.
    mech_runs = runs * 100
    ratio_median = np.full((n_count, sigma_count), np.nan)
    var_a_ratio = np.full((n_count, sigma_count), np.nan)
    var_b_ratio = np.full((n_count, sigma_count), np.nan)
    reference_z_hat = None
    for n_index, n_points in enumerate(N_VALUES):
        rng_set = np.random.default_rng([seed, 4102, n_index])
        ctrl_pixels, ctrl_depth = sample_control_pixels(depth, valid, n_points, rng_set)
        ctrl_u = 1.0 / ctrl_depth
        ctrl_relative = A_TRUE / ctrl_depth + B_TRUE
        for sigma_index, sigma in enumerate(SIGMA_VALUES):
            if sigma == 0.0:
                continue  # no noise -> no spread to compare
            fits = []
            for repetition in range(mech_runs):
                rng_noise = np.random.default_rng([seed, 4103, n_index, sigma_index, repetition])
                noisy = ctrl_relative * (1.0 + rng_noise.normal(0.0, sigma, n_points))
                fits.append(fit_inverse_affine(ctrl_u, noisy))
                if n_points == REFERENCE_N and sigma == REFERENCE_SIGMA and reference_z_hat is None:
                    try:
                        candidate = metric_from_relative(relative, *fits[-1], valid)
                    except ValueError:
                        candidate = None  # degenerate draw; try the next one
                    if candidate is not None:
                        reference_z_hat = candidate
                        reference_error = candidate - depth
                        reference_fit = fits[-1]
                        reference_noisy = noisy.copy()
                        reference_ctrl = (ctrl_pixels, ctrl_depth, ctrl_relative)
            samples = np.asarray(fits)
            empirical_cov = np.cov(samples.T)
            sandwich = affine_covariance(ctrl_u, ctrl_relative, sigma)
            a_mean = float(samples[:, 0].mean())
            empirical_std = predicted_error_std(depth, valid, a_mean, empirical_cov)
            predicted_std = predicted_error_std(depth, valid, a_mean, sandwich)
            comparable = predicted_std[valid] > 0.0
            ratio_median[n_index, sigma_index] = float(
                np.median(empirical_std[valid][comparable] / predicted_std[valid][comparable])
            )
            var_a_ratio[n_index, sigma_index] = empirical_cov[0, 0] / sandwich[0, 0]
            var_b_ratio[n_index, sigma_index] = empirical_cov[1, 1] / sandwich[1, 1]
            if n_points == REFERENCE_N and sigma == REFERENCE_SIGMA:
                mechanism_empirical_std = empirical_std
                mechanism_predicted_std = predicted_std
    archive = {
        "depth_map_m": depth,
        "valid_mask": valid,
        "pole_mask": pole,
        "relative_map": relative,
        "near_mask": near_mask,
        "far_mask": far_mask,
        "n_values": np.array(N_VALUES),
        "sigma_values": np.array(SIGMA_VALUES),
        "runs_per_group": np.array(runs),
        "group_mean_abs_error_m": group["mean"],
        "group_median_abs_error_m": group["median"],
        "group_std_abs_error_m": group["std"],
        "group_max_abs_error_m": group["max"],
        "group_near_mean_abs_error_m": group["near"],
        "group_far_mean_abs_error_m": group["far"],
        "group_mean_signed_error_m": group["signed"],
        "fit_failures": failures,
        "mechanism_ratio_median": ratio_median,
        "mechanism_var_a_ratio": var_a_ratio,
        "mechanism_var_b_ratio": var_b_ratio,
        "mechanism_empirical_std_m": mechanism_empirical_std,
        "mechanism_predicted_std_m": mechanism_predicted_std,
        "reference_z_hat_m": reference_z_hat,
        "reference_error_m": reference_error,
        "reference_fit_ab": np.array(reference_fit),
        "reference_ctrl_pixels": reference_ctrl[0],
        "reference_ctrl_depth_m": reference_ctrl[1],
        "reference_ctrl_relative": reference_ctrl[2],
        "reference_ctrl_relative_noisy": reference_noisy,
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "trajectories.npz", **archive)
    far_over_near = np.where(group["near"] > 0, group["far"] / group["near"], np.nan)
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": 1,
        "relative_proxy": "r = a*(1/Z) + b (inverse-depth affine ambiguity, arbitrary units)",
        "calibration": "ordinary least squares r ~ a*(1/Z) + b on N control points",
        "camera_model": "pinhole {u = K [R|t] X / z} (lesson-22 camera, unchanged)",
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "intrinsic": K_INTRINSIC.tolist(),
        "eye_m": EYE.tolist(),
        "target_m": TARGET.tolist(),
        "near_plane_m": NEAR_PLANE_M,
        "pole_radius_m": POLE_RADIUS_M,
        "a_true": A_TRUE,
        "b_true": B_TRUE,
        "naive_assumed_a": NAIVE_A,
        "naive_assumed_b": NAIVE_B,
        "n_values": list(N_VALUES),
        "sigma_values_relative": list(SIGMA_VALUES),
        "min_inv_depth_span": MIN_INV_DEPTH_SPAN,
        "reference_group": {
            "n": REFERENCE_N,
            "sigma": REFERENCE_SIGMA,
            "fit_a": float(reference_fit[0]),
            "fit_b": float(reference_fit[1]),
            "mean_abs_error_m": float(group["mean"][2, 1]),
            "near_mean_abs_error_m": float(group["near"][2, 1]),
            "far_mean_abs_error_m": float(group["far"][2, 1]),
        },
        "valid_pixels": int(valid.sum()),
        "pole_pixels": int(pole.sum()),
        "depth_min_m": float(depth[valid].min()),
        "depth_max_m": float(depth[valid].max()),
        "near_quantile_depth_m": q_near,
        "far_quantile_depth_m": q_far,
        "runs_per_group": runs,
        "mechanism_realizations": mech_runs,
        "base_seed": seed,
        "naive_mean_abs_error_m": float(naive_error.mean()),
        "naive_max_abs_error_m": float(naive_error.max()),
        "perfect_prior_max_abs_error_m": perfect_max,
        "group_mean_abs_error_m": group["mean"].tolist(),
        "group_median_abs_error_m": group["median"].tolist(),
        "group_std_abs_error_m": group["std"].tolist(),
        "group_max_abs_error_m": group["max"].tolist(),
        "group_near_mean_abs_error_m": group["near"].tolist(),
        "group_far_mean_abs_error_m": group["far"].tolist(),
        "group_far_over_near": far_over_near.tolist(),
        "group_mean_signed_error_m": group["signed"].tolist(),
        "fit_failures": failures.tolist(),
        "mechanism_ratio_median": ratio_median.tolist(),
        "mechanism_var_a_ratio": var_a_ratio.tolist(),
        "mechanism_var_b_ratio": var_b_ratio.tolist(),
        "trajectories_sha256": digest(output / "trajectories.npz"),
        "source_sha256": {
            "experiments/monocular_metric.py": digest(Path(__file__).resolve()),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            (
                "Relative depth is an ideal affine proxy r = a/Z + b; real models add "
                "non-affine distortion that this calibration cannot remove"
            ),
            (
                "Noise is multiplicative on the control-point r readings only; the full "
                "relative field stays noiseless by design"
            ),
            (
                "Control points are drawn uniformly over valid pixels with a minimum "
                "inverse-depth span; real control-point placement is rarely uniform"
            ),
            (
                "Single view, static scene, no occlusion boundaries beyond the pole, "
                "no image rendering and no learned depth network"
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
    fig.suptitle("第二十三课 单目相对深度 → 米制尺度标定（N 控制点 × σ 读数噪声）")
    ax_r = fig.add_subplot(2, 3, 1)
    ax_truth = fig.add_subplot(2, 3, 2)
    ax_naive = fig.add_subplot(2, 3, 3)
    ax_error = fig.add_subplot(2, 3, 4)
    ax_fit = fig.add_subplot(2, 3, 5)
    ax_curve = fig.add_subplot(2, 3, 6)
    valid = archive["valid_mask"]
    relative = archive["relative_map"]
    depth = archive["depth_map_m"]
    im = ax_r.imshow(relative, cmap="viridis")
    fig.colorbar(im, ax=ax_r, shrink=0.85, label="r（任意单位）")
    ax_r.set(title="相对深度 r = a/Z + b：没有米制尺度", xlabel="u / px", ylabel="v / px")
    im = ax_truth.imshow(depth, cmap="magma")
    fig.colorbar(im, ax=ax_truth, shrink=0.85, label="Z / m")
    ax_truth.set(title="米制深度 Z（逐像素真值）", xlabel="u / px", ylabel="v / px")
    naive = np.full(depth.shape, np.nan)
    naive[valid] = 1.0 / relative[valid]
    im = ax_naive.imshow(naive, cmap="magma", vmin=0.0)
    fig.colorbar(im, ax=ax_naive, shrink=0.85, label="1/r 解读成的“深度” / m")
    ax_naive.set(
        title=f"无标定：假设 a=1, b=0（均值误差 {summary['naive_mean_abs_error_m']:.2f} m）",
        xlabel="u / px",
        ylabel="v / px",
    )
    error = np.abs(archive["reference_error_m"])
    im = ax_error.imshow(error, cmap="inferno")
    fig.colorbar(im, ax=ax_error, shrink=0.85, label="|Z_hat − Z| / m")
    pixels = archive["reference_ctrl_pixels"]
    ax_error.scatter(
        pixels[:, 0], pixels[:, 1], marker="x", s=70, c="#22d3ee", lw=1.6, label="控制点"
    )
    ax_error.legend(loc="lower right", fontsize=8)
    ref = summary["reference_group"]
    ax_error.set(
        title=f"标定后误差（N={ref['n']}, σ={ref['sigma']:.0%}，单种子）",
        xlabel="u / px",
        ylabel="v / px",
    )
    ctrl_u = 1.0 / archive["reference_ctrl_depth_m"]
    u_line = np.linspace(ctrl_u.min() * 0.9, ctrl_u.max() * 1.05, 50)
    ax_fit.plot(
        u_line, A_TRUE * u_line + B_TRUE, "--", color="#0f172a", lw=1.4, label="真值 a, b（未知）"
    )
    a_fit, b_fit = archive["reference_fit_ab"]
    ax_fit.plot(
        u_line,
        a_fit * u_line + b_fit,
        "-",
        color="#2563eb",
        lw=1.6,
        label=f"拟合 a={a_fit:.3f}, b={b_fit:.3f}",
    )
    ax_fit.scatter(
        ctrl_u,
        archive["reference_ctrl_relative_noisy"],
        marker="x",
        s=60,
        c="#dc2626",
        lw=1.6,
        label="控制点读数（含噪声）",
    )
    ax_fit.set(xlabel="1/Z / (1/m)", ylabel="r（任意单位）", title="控制点上的逆深度仿射拟合")
    ax_fit.legend(fontsize=8)
    n_values = archive["n_values"]
    mean_err = archive["group_mean_abs_error_m"] * 100.0
    for sigma_index, sigma in enumerate(SIGMA_VALUES):
        ax_curve.plot(
            n_values,
            mean_err[:, sigma_index],
            "-o",
            ms=4,
            label=f"σ={sigma:.0%}" if sigma else "σ=0（数值精确）",
        )
    ax_curve.plot(
        n_values,
        archive["group_near_mean_abs_error_m"][:, -1] * 100.0,
        "--",
        color="#9333ea",
        lw=1.2,
        label="σ=3% 近段（≤15% 分位）",
    )
    ax_curve.plot(
        n_values,
        archive["group_far_mean_abs_error_m"][:, -1] * 100.0,
        "--",
        color="#ea580c",
        lw=1.2,
        label="σ=3% 远段（≥85% 分位）",
    )
    ax_curve.axhline(
        summary["naive_mean_abs_error_m"] * 100.0,
        color="#0f172a",
        ls=":",
        lw=1.2,
        label="无标定（a=1, b=0）",
    )
    ax_curve.set_yscale("log")
    ax_curve.set(
        xlabel="控制点数量 N",
        ylabel="全图平均 |Z_hat − Z| / cm",
        title="N–误差曲线（20 种子均值，对数轴）",
    )
    ax_curve.legend(fontsize=8)
    for ax in (ax_r, ax_truth, ax_naive, ax_error, ax_fit, ax_curve):
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
    ref = report["reference_group"]
    print(
        f"naive (a=1,b=0) mean |err| {report['naive_mean_abs_error_m'] * 100:.1f} cm; "
        f"perfect prior max {report['perfect_prior_max_abs_error_m']:.2e} m"
    )
    print(
        f"reference N={ref['n']} sigma={ref['sigma']:.0%}: mean |err| "
        f"{ref['mean_abs_error_m'] * 100:.2f} cm "
        f"(near {ref['near_mean_abs_error_m'] * 100:.2f} / far {ref['far_mean_abs_error_m'] * 100:.2f} cm)"
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
