"""Fit one fixed right-encoder factor from independent signed straight distances."""

import math

import numpy as np

from embodied_learning.odometry import encoder_array


def fit_right_correction(right_angle_rad, reference_distance_m, radius_m):
    """Origin-constrained least squares: reference distance ~= c * r * reading.

    Inputs are total signed increments on separate, externally verified straight
    runs. No simulator bias, true wheel rotations or evaluation poses enter here.
    Radius is assumed known; radius error and encoder scale are not identifiable
    separately with these measurements. This is not a track-width calibration.
    """
    angle = np.asarray(right_angle_rad, dtype=float)
    distance = np.asarray(reference_distance_m, dtype=float)
    if (
        not math.isfinite(radius_m)
        or radius_m <= 0
        or angle.ndim != 1
        or angle.shape != distance.shape
        or not angle.size
        or not np.isfinite(angle).all()
        or not np.isfinite(distance).all()
        or np.any(angle == 0)
        or np.any(distance == 0)
        or np.any(np.sign(angle) != np.sign(distance))
    ):
        raise ValueError("Need finite nonzero straight increments with matching signed distances")
    # Normalize each vector to avoid squaring very large or tiny inputs.
    amax, dmax = float(np.abs(angle).max()), float(np.abs(distance).max())
    a, d = angle / amax, distance / dmax
    factor = float((a @ d) / (a @ a) * (dmax / amax) / radius_m)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Correction must be finite and positive")
    return factor


def correct_right_encoder(readings, factor):
    """Correct signed increments, preserving both absolute starting counters."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Correction must be finite and positive")
    result = encoder_array(readings).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, 1] = result[0, 1] + factor * (result[:, 1] - result[0, 1])
    if not np.isfinite(result).all():
        raise ValueError("Corrected readings overflow")
    return result
