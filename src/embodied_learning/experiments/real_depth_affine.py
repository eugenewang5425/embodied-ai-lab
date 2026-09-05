"""Lesson 24: affine check of a REAL monocular relative-depth model.

Lesson 23 calibrated an ideal affine proxy r = a*(1/Z) + b with control points
and proved the mechanism on noise. This lesson feeds the real Depth Anything
V2 Small output (archived by monocular-depth/bench/affine_check_pinhole.py on
the SAME lesson-22 scene: 640x480, K=[600,600,320,240], the same eye/target,
near plane 0.5 m, valid=280687 pixels) through the SAME inverse-depth affine
machinery and quantifies how far the real model deviates from a single global
affine law:

(1) control-point calibration exactly as in lesson 23 (N in {2, 5, 10, 20},
    1/Z span >= 0.1, 20 seeds, no synthetic noise - the model supplies its own)
    with the noiseless ideal proxy r = 5/Z + 0.2 scanned on the SAME paired
    control-point draws as the upper bound;
(2) a dense ordinary-least-squares fit over ALL valid pixels (the other
    calibration reading), its R^2 and residual structure: mean/std, residual
    vs 1/Z binned curve, correlation with image position, pole-vs-ground split;
(3) calibrated metric error after inversion (full image, near/far strata at
    the 15%/85% depth quantiles), real model vs ideal proxy side by side.

This module imports the lesson-23 public functions unchanged (fit, inversion,
control sampling, constants). torch and the monocular-depth venv stay out of
the main repo: the input is a plain .npz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from embodied_learning.experiments.monocular_metric import (
    A_TRUE,
    B_TRUE,
    FAR_QUANTILE,
    MIN_INV_DEPTH_SPAN,
    NEAR_QUANTILE,
    fit_inverse_affine,
    metric_from_relative,
    relative_from_depth,
    sample_control_pixels,
)
from embodied_learning.experiments.pinhole_projection import HEIGHT_PX, WIDTH_PX

EXPERIMENT = "real_depth_affine_check"
SCHEMA_VERSION = 1
N_VALUES = (2, 5, 10, 20)
REFERENCE_N = 5
DEFAULT_RUNS = 20
DEFAULT_SEED = 0
N_BINS = 12
# Both scans (real model and ideal proxy) share ONE random stream per
# (N, repetition) so each pair uses identical control points: the real-minus-
# ideal gap is then paired, not two independent draws.
RNG_SALT = 9201
REQUIRED_INPUT_KEYS = ("depth_m", "valid", "pole", "rgb", "r_pred", "meta_json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bench_npz(path):
    """Load and validate the bench archive produced by affine_check_pinhole.py."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bench npz not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_INPUT_KEYS if key not in data]
        if missing:
            raise ValueError(f"Bench npz missing keys: {missing}")
        depth = np.asarray(data["depth_m"], dtype=float)
        valid = np.asarray(data["valid"], dtype=bool)
        pole = np.asarray(data["pole"], dtype=bool)
        rgb = np.asarray(data["rgb"])
        r_pred = np.asarray(data["r_pred"], dtype=float)
        meta = json.loads(str(data["meta_json"]))
    shape = (HEIGHT_PX, WIDTH_PX)
    for name, array in (("depth_m", depth), ("r_pred", r_pred), ("valid", valid), ("pole", pole)):
        if array.shape != shape:
            raise ValueError(f"{name} shape {array.shape} does not match lesson-22 canvas {shape}")
    if rgb.shape != (HEIGHT_PX, WIDTH_PX, 3):
        raise ValueError(f"rgb shape {rgb.shape} does not match {(HEIGHT_PX, WIDTH_PX, 3)}")
    if not np.isfinite(r_pred[valid]).all():
        raise ValueError("r_pred has non-finite values on valid pixels")
    if "model" not in meta:
        raise ValueError("Bench npz metadata lacks a model name")
    return {
        "depth": depth,
        "valid": valid,
        "pole": pole,
        "rgb": rgb,
        "r_pred": r_pred,
        "meta": meta,
    }


def r_squared(inv_depth, relative, a, b):
    """Coefficient of determination of r ~ a*(1/Z) + b on the fitted samples."""
    r = np.asarray(relative, dtype=float)
    prediction = a * np.asarray(inv_depth, dtype=float) + b
    ss_residual = float(((r - prediction) ** 2).sum())
    ss_total = float(((r - r.mean()) ** 2).sum())
    if ss_total <= 0.0:
        return float("nan")
    return 1.0 - ss_residual / ss_total


def pearson(x, y):
    """Pearson correlation, or None when either side is (numerically) constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.ptp(x) < 1e-15 or np.ptp(y) < 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bin_residuals(inv_depth, residual, count):
    """Equal-width bins over the 1/Z range: mean/std residual per nonempty bin."""
    inv_depth = np.asarray(inv_depth, dtype=float)
    residual = np.asarray(residual, dtype=float)
    edges = np.linspace(inv_depth.min(), inv_depth.max(), count + 1)
    index = np.clip(np.digitize(inv_depth, edges) - 1, 0, count - 1)
    centers, means, stds, sizes = [], [], [], []
    for b in range(count):
        selected = index == b
        size = int(selected.sum())
        if size == 0:
            continue
        centers.append(float((edges[b] + edges[b + 1]) / 2.0))
        means.append(float(residual[selected].mean()))
        stds.append(float(residual[selected].std(ddof=1)) if size > 1 else 0.0)
        sizes.append(size)
    return centers, means, stds, sizes


def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def _float_array(values):
    """np.array with None mapped to NaN so the archive stays pure float."""
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _sweep_group(depth, valid, near_mask, far_mask, r_field, n_points, n_index, runs, seed):
    """One N group over `runs` seeds; control points are shared with the proxy."""
    per_mean, per_max, per_near, per_far, per_signed, per_r2 = [], [], [], [], [], []
    failures = 0
    for repetition in range(runs):
        rng = np.random.default_rng([seed, RNG_SALT, n_index, repetition])
        pixels, ctrl_depth = sample_control_pixels(depth, valid, n_points, rng)
        rows = pixels[:, 1].astype(int)
        cols = pixels[:, 0].astype(int)
        ctrl_r = r_field[rows, cols]
        a_fit, b_fit = fit_inverse_affine(1.0 / ctrl_depth, ctrl_r)
        if a_fit <= 0.0:
            failures += 1
            continue
        try:
            z_hat = metric_from_relative(r_field, a_fit, b_fit, valid)
        except ValueError:
            failures += 1  # b_hat beyond the r range: no valid metric map
            continue
        error = np.abs(z_hat[valid] - depth[valid])
        per_mean.append(error.mean())
        per_max.append(error.max())
        per_near.append(error[near_mask[valid]].mean())
        per_far.append(error[far_mask[valid]].mean())
        per_signed.append((z_hat[valid] - depth[valid]).mean())
        per_r2.append(r_squared(1.0 / ctrl_depth, ctrl_r, a_fit, b_fit))
    return {
        "mean": _mean_or_none(per_mean),
        "median": float(np.median(per_mean)) if per_mean else None,
        "std": float(np.std(per_mean, ddof=1)) if len(per_mean) > 1 else 0.0,
        "max": float(np.max(per_max)) if per_max else None,
        "near": _mean_or_none(per_near),
        "far": _mean_or_none(per_far),
        "signed": _mean_or_none(per_signed),
        "r2_ctrl": _mean_or_none(per_r2),
        "failures": failures,
    }


def reference_fit(depth, valid, r_field, seed):
    """The deterministic N=5, repetition-0 fit used by the plots."""
    rng = np.random.default_rng([seed, RNG_SALT, N_VALUES.index(REFERENCE_N), 0])
    pixels, ctrl_depth = sample_control_pixels(depth, valid, REFERENCE_N, rng)
    ctrl_r = r_field[pixels[:, 1].astype(int), pixels[:, 0].astype(int)]
    a_fit, b_fit = fit_inverse_affine(1.0 / ctrl_depth, ctrl_r)
    try:
        z_hat = metric_from_relative(r_field, a_fit, b_fit, valid)
        error = z_hat - depth
    except ValueError:
        z_hat, error = None, None
    return {
        "pixels": pixels,
        "ctrl_depth": ctrl_depth,
        "ctrl_r": ctrl_r,
        "fit_ab": np.array([a_fit, b_fit]),
        "z_hat": z_hat,
        "error": error,
    }


def run_experiment(input_npz, output, *, runs=DEFAULT_RUNS, seed=DEFAULT_SEED):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if runs < 2:
        raise ValueError("Need at least two repetitions for seed statistics")
    scene = load_bench_npz(input_npz)
    depth, valid, pole, r_pred = scene["depth"], scene["valid"], scene["pole"], scene["r_pred"]
    # Ideal affine proxy on the same depth map: the paired upper bound.
    r_proxy = relative_from_depth(depth, valid)
    q_near = float(np.quantile(depth[valid], NEAR_QUANTILE))
    q_far = float(np.quantile(depth[valid], FAR_QUANTILE))
    near_mask = valid & (depth <= q_near)
    far_mask = valid & (depth >= q_far)

    scans = {}
    for label, field in (("real", r_pred), ("proxy", r_proxy)):
        scans[label] = [
            _sweep_group(depth, valid, near_mask, far_mask, field, n, n_index, runs, seed)
            for n_index, n in enumerate(N_VALUES)
        ]

    # Dense (all-pixel) ordinary-least-squares reading of the same affine law.
    u_all = 1.0 / depth[valid]
    r_all = r_pred[valid]
    a_dense, b_dense = fit_inverse_affine(u_all, r_all)
    r2_dense = r_squared(u_all, r_all, a_dense, b_dense)
    residual_map = np.full(depth.shape, np.nan)
    residual_map[valid] = r_all - (a_dense * u_all + b_dense)
    residual = residual_map[valid]
    rows_all, cols_all = np.nonzero(valid)
    half_u = cols_all < WIDTH_PX / 2
    half_v = rows_all < HEIGHT_PX / 2
    dense_metric = None
    try:
        z_hat_dense = metric_from_relative(r_pred, a_dense, b_dense, valid)
        dense_error = np.abs(z_hat_dense[valid] - depth[valid])
        dense_metric = {
            "mean_abs_error_m": float(dense_error.mean()),
            "max_abs_error_m": float(dense_error.max()),
            "near_mean_abs_error_m": float(dense_error[near_mask[valid]].mean()),
            "far_mean_abs_error_m": float(dense_error[far_mask[valid]].mean()),
        }
        dense_error_map = z_hat_dense - depth
    except ValueError:
        dense_error_map = None  # b_dense beyond the r range: inversion impossible
    # Sanity: the dense fit of the IDEAL proxy must recover (A_TRUE, B_TRUE).
    a_proxy_dense, b_proxy_dense = fit_inverse_affine(u_all, r_proxy[valid])

    ref_real = reference_fit(depth, valid, r_pred, seed)
    ref_proxy = reference_fit(depth, valid, r_proxy, seed)
    bin_centers, bin_means, bin_stds, bin_sizes = bin_residuals(u_all, residual, N_BINS)

    nan_map = np.full(depth.shape, np.nan)
    archive = {
        "residual_map": residual_map,
        "dense_fit_ab": np.array([a_dense, b_dense]),
        "dense_error_m": nan_map if dense_error_map is None else dense_error_map,
        "bin_centers": np.array(bin_centers),
        "bin_means": np.array(bin_means),
        "bin_stds": np.array(bin_stds),
        "bin_sizes": np.array(bin_sizes, dtype=int),
        "real_reference_error_m": nan_map if ref_real["error"] is None else ref_real["error"],
        "real_reference_z_hat_m": nan_map if ref_real["z_hat"] is None else ref_real["z_hat"],
        "real_reference_ctrl_pixels": ref_real["pixels"],
        "real_reference_ctrl_depth_m": ref_real["ctrl_depth"],
        "real_reference_ctrl_r": ref_real["ctrl_r"],
        "real_reference_fit_ab": ref_real["fit_ab"],
        "proxy_reference_error_m": nan_map if ref_proxy["error"] is None else ref_proxy["error"],
        "proxy_reference_fit_ab": ref_proxy["fit_ab"],
        "n_values": np.array(N_VALUES),
        "runs_per_group": np.array(runs),
        "real_group_mean_abs_error_m": _float_array([g["mean"] for g in scans["real"]]),
        "real_group_median_abs_error_m": _float_array([g["median"] for g in scans["real"]]),
        "real_group_std_abs_error_m": _float_array([g["std"] for g in scans["real"]]),
        "real_group_max_abs_error_m": _float_array([g["max"] for g in scans["real"]]),
        "real_group_near_mean_abs_error_m": _float_array([g["near"] for g in scans["real"]]),
        "real_group_far_mean_abs_error_m": _float_array([g["far"] for g in scans["real"]]),
        "real_group_mean_signed_error_m": _float_array([g["signed"] for g in scans["real"]]),
        "real_group_r2_ctrl": _float_array([g["r2_ctrl"] for g in scans["real"]]),
        "real_fit_failures": np.array([g["failures"] for g in scans["real"]], dtype=int),
        "proxy_group_mean_abs_error_m": _float_array([g["mean"] for g in scans["proxy"]]),
        "proxy_group_max_abs_error_m": _float_array([g["max"] for g in scans["proxy"]]),
        "proxy_fit_failures": np.array([g["failures"] for g in scans["proxy"]], dtype=int),
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "analysis.npz", **archive)

    real_scan, proxy_scan = scans["real"], scans["proxy"]
    summary = {
        "experiment": EXPERIMENT,
        "schema_version": SCHEMA_VERSION,
        "input_path": str(input_npz),
        "input_sha256": digest(input_npz),
        "analysis_npz_sha256": digest(output / "analysis.npz"),
        "model": scene["meta"].get("model"),
        "model_input_size": scene["meta"].get("input_size"),
        "model_device": scene["meta"].get("device"),
        "bench_timestamp_utc": scene["meta"].get("timestamp_utc"),
        "fog_tau_m": scene["meta"].get("fog_tau_m"),
        "camera_model": "pinhole {u = K [R|t] X / z} (lesson-22 camera, unchanged)",
        "canvas_px": [WIDTH_PX, HEIGHT_PX],
        "calibration": (
            "ordinary least squares r ~ a*(1/Z) + b; control-point protocol of "
            f"lesson 23 (uniform draws over valid pixels, 1/Z span >= {MIN_INV_DEPTH_SPAN:.2f}) "
            "plus a dense all-pixel fit"
        ),
        "proxy_upper_bound": f"r = {A_TRUE}/Z + {B_TRUE} without noise (lesson-23 proxy)",
        "min_inv_depth_span": MIN_INV_DEPTH_SPAN,
        "n_values": list(N_VALUES),
        "reference_n": REFERENCE_N,
        "runs_per_group": runs,
        "base_seed": seed,
        "rng_salt": RNG_SALT,
        "valid_pixels": int(valid.sum()),
        "pole_pixels": int(pole.sum()),
        "depth_min_m": float(depth[valid].min()),
        "depth_max_m": float(depth[valid].max()),
        "near_quantile_depth_m": q_near,
        "far_quantile_depth_m": q_far,
        "r_pred_min": float(r_pred[valid].min()),
        "r_pred_max": float(r_pred[valid].max()),
        "dense_fit": {
            "a": float(a_dense),
            "b": float(b_dense),
            "r_squared": None if np.isnan(r2_dense) else float(r2_dense),
            "residual_mean": float(residual.mean()),
            "residual_std": float(residual.std(ddof=1)),
            "residual_max_abs": float(np.abs(residual).max()),
            "residual_std_over_r_std": float(residual.std(ddof=1) / r_all.std(ddof=1)),
            "bin_residual_range": float(max(bin_means) - min(bin_means)),
            "residual_correlation_u_px": pearson(residual, cols_all),
            "residual_correlation_v_px": pearson(residual, rows_all),
            "residual_mean_left_half": float(residual[half_u].mean()),
            "residual_mean_right_half": float(residual[~half_u].mean()),
            "residual_mean_top_half": float(residual[half_v].mean()),
            "residual_mean_bottom_half": float(residual[~half_v].mean()),
            "pole_mean_abs_residual": float(np.abs(residual[pole[valid]]).mean()),
            "ground_mean_abs_residual": float(np.abs(residual[~pole[valid]]).mean()),
        },
        "dense_fit_proxy_sanity": {"a": float(a_proxy_dense), "b": float(b_proxy_dense)},
        "dense_metric_error": dense_metric,
        "reference_group": {
            "n": REFERENCE_N,
            "fit_a": float(ref_real["fit_ab"][0]),
            "fit_b": float(ref_real["fit_ab"][1]),
            "ctrl_r": ref_real["ctrl_r"].tolist(),
            "ctrl_u": (1.0 / ref_real["ctrl_depth"]).tolist(),
        },
        "real_group_mean_abs_error_m": [g["mean"] for g in real_scan],
        "real_group_median_abs_error_m": [g["median"] for g in real_scan],
        "real_group_std_abs_error_m": [g["std"] for g in real_scan],
        "real_group_max_abs_error_m": [g["max"] for g in real_scan],
        "real_group_near_mean_abs_error_m": [g["near"] for g in real_scan],
        "real_group_far_mean_abs_error_m": [g["far"] for g in real_scan],
        "real_group_mean_signed_error_m": [g["signed"] for g in real_scan],
        "real_group_r2_ctrl": [g["r2_ctrl"] for g in real_scan],
        "real_fit_failures": [g["failures"] for g in real_scan],
        "proxy_group_mean_abs_error_m": [g["mean"] for g in proxy_scan],
        "proxy_group_max_abs_error_m": [g["max"] for g in proxy_scan],
        "proxy_fit_failures": [g["failures"] for g in proxy_scan],
        "paired_gap_mean_minus_proxy_m": [
            None if (g["mean"] is None or p["mean"] is None) else g["mean"] - p["mean"]
            for g, p in zip(real_scan, proxy_scan)
        ],
        "trajectories_sha256": digest(output / "analysis.npz"),
        "source_sha256": {
            "experiments/real_depth_affine.py": digest(Path(__file__).resolve()),
        },
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "limitations": [
            (
                "The domain is a clean Lambertian synthetic render (checkerboard "
                "ground, one pole, optional fog); the verdict applies to THIS "
                "domain, not to real photographs"
            ),
            (
                "Control-point depths are exact pixels of the rendered Z map; a "
                "real deployment would add ranging error on top of model error"
            ),
            (
                "One scene, one viewpoint, one checkpoint (DA v2 Small); other "
                "scenes or encoders may deviate differently"
            ),
            (
                "The affine law r = a/Z + b is fitted, not derived: a high R^2 "
                "supports it only within the residual size reported here"
            ),
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    make_plot(archive, summary, output, scene)
    return summary


def make_plot(archive, summary, output, scene):
    import matplotlib

    matplotlib.use("Agg")  # headless analysis plots; never open a Tk window here
    import matplotlib.pyplot as plt

    from embodied_learning.plotting import configure_plot_font

    configure_plot_font()
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    fig.suptitle("第二十四课 真实单目相对深度（Depth Anything V2 Small）的逆深度仿射检验")
    ax_rgb = fig.add_subplot(2, 3, 1)
    ax_r = fig.add_subplot(2, 3, 2)
    ax_res = fig.add_subplot(2, 3, 3)
    ax_curve = fig.add_subplot(2, 3, 4)
    ax_scan = fig.add_subplot(2, 3, 5)
    ax_error = fig.add_subplot(2, 3, 6)
    ax_rgb.imshow(scene["rgb"])
    ax_rgb.set(title="输入：合成场景朗伯渲染（bench 生成）", xlabel="u / px", ylabel="v / px")
    im = ax_r.imshow(scene["r_pred"], cmap="viridis")
    fig.colorbar(im, ax=ax_r, shrink=0.85, label="r_pred（任意单位）")
    ax_r.set(title="真实模型输出 r_pred", xlabel="u / px", ylabel="v / px")
    residual = archive["residual_map"]
    residual_std = summary["dense_fit"]["residual_std"]
    im = ax_res.imshow(residual, cmap="coolwarm", vmin=-3 * residual_std, vmax=3 * residual_std)
    fig.colorbar(im, ax=ax_res, shrink=0.85, label="残差 r − (a·(1/Z)+b)")
    dense = summary["dense_fit"]
    r2_text = "N/A" if dense["r_squared"] is None else f"{dense['r_squared']:.4f}"
    ax_res.set(
        title=f"全图拟合残差（R²={r2_text}，σ={residual_std:.3f}）",
        xlabel="u / px",
        ylabel="v / px",
    )
    ax_curve.errorbar(
        archive["bin_centers"],
        archive["bin_means"],
        yerr=archive["bin_stds"],
        fmt="-o",
        ms=4,
        color="#2563eb",
        ecolor="#93c5fd",
        capsize=2,
        label="分箱均值 ± 1σ",
    )
    ax_curve.axhline(0.0, color="#0f172a", ls="--", lw=1.2)
    ax_curve.set(
        xlabel="1/Z / (1/m)",
        ylabel="残差（任意单位）",
        title=f"残差 vs 1/Z（分箱均值域 {dense['bin_residual_range']:.3f}）",
    )
    ax_curve.legend(fontsize=8)
    n_values = archive["n_values"]
    ax_scan.plot(
        n_values,
        archive["proxy_group_mean_abs_error_m"] * 100.0,
        "-o",
        ms=4,
        label="理想代理 r=5/Z+0.2（对照上界）",
    )
    ax_scan.plot(
        n_values,
        archive["real_group_mean_abs_error_m"] * 100.0,
        "-o",
        ms=4,
        label="真实 DA V2 输出",
    )
    ax_scan.plot(
        n_values,
        archive["real_group_near_mean_abs_error_m"] * 100.0,
        "--",
        color="#9333ea",
        lw=1.2,
        label="真实·近段（≤15% 分位）",
    )
    ax_scan.plot(
        n_values,
        archive["real_group_far_mean_abs_error_m"] * 100.0,
        "--",
        color="#ea580c",
        lw=1.2,
        label="真实·远段（≥85% 分位）",
    )
    ax_scan.set_yscale("log")
    ax_scan.set(
        xlabel="控制点数量 N",
        ylabel="全图平均 |Z_hat − Z| / cm",
        title=f"N–误差曲线（{summary['runs_per_group']} 种子均值，对数轴）",
    )
    ax_scan.legend(fontsize=8)
    error = archive["real_reference_error_m"]
    if error is not None:
        im = ax_error.imshow(error, cmap="inferno")
        fig.colorbar(im, ax=ax_error, shrink=0.85, label="Z_hat − Z / m")
        pixels = archive["real_reference_ctrl_pixels"]
        ax_error.scatter(
            pixels[:, 0], pixels[:, 1], marker="x", s=70, c="#22d3ee", lw=1.6, label="控制点"
        )
        ax_error.legend(loc="lower right", fontsize=8)
    ref = summary["reference_group"]
    ax_error.set(
        title=f"标定后误差（N={ref['n']}，单种子）",
        xlabel="u / px",
        ylabel="v / px",
    )
    for ax in (ax_rgb, ax_r, ax_res, ax_curve, ax_scan, ax_error):
        ax.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="bench npz from monocular-depth/bench/affine_check_pinhole.py",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 for seed statistics")
    report = run_experiment(args.input, args.output, runs=args.runs, seed=args.seed)
    dense = report["dense_fit"]
    dense_metric = report["dense_metric_error"]
    real = report["real_group_mean_abs_error_m"]
    proxy = report["proxy_group_mean_abs_error_m"]
    r2 = dense["r_squared"]
    print(
        f"dense fit a={dense['a']:.4f} b={dense['b']:.4f} "
        f"R^2={'N/A' if r2 is None else f'{r2:.4f}'}; "
        f"residual std {dense['residual_std']:.4f} "
        f"({dense['residual_std_over_r_std']:.1%} of r std)"
    )
    if dense_metric is not None:
        print(
            f"dense calibrated error: mean {dense_metric['mean_abs_error_m'] * 100:.1f} cm "
            f"(near {dense_metric['near_mean_abs_error_m'] * 100:.1f} / "
            f"far {dense_metric['far_mean_abs_error_m'] * 100:.1f} cm)"
        )
    for n, r_value, p_value in zip(report["n_values"], real, proxy):
        r_text = "failed" if r_value is None else f"{r_value * 100:.2f} cm"
        print(f"N={n:>2}: real mean |err| {r_text} vs proxy {p_value * 100:.4f} cm")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
