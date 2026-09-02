"""Deterministic backward view selection with explicit geometry constraints."""

from __future__ import annotations

import cv2
import numpy as np

from .calibration import solve_observations


def geometry_features(objects, pixels, image_size, fit):
    matrix = np.asarray(fit["camera_matrix"])
    distortion = np.asarray(fit["distortion_coefficients"])
    angles, tilted, regions = [], [], []
    for obj, img in zip(objects, pixels, strict=True):
        ok, rotation, _ = cv2.solvePnP(obj, img, matrix, distortion)
        if not ok:
            raise ValueError("Cannot estimate board pose")
        normal = cv2.Rodrigues(rotation)[0][:, 2]
        normal *= 1 if normal[2] >= 0 else -1
        angles.append(np.degrees(np.arctan2(normal[:2], normal[2])))
        tilted.append(np.degrees(np.arccos(np.clip(normal[2], -1, 1))) >= 15)
        center = img.reshape(-1, 2).mean(axis=0) / image_size
        col, row = np.minimum((center * 3).astype(int), 2)
        regions.append((int(row), int(col)))
    return np.asarray(angles), np.asarray(tilted), regions


def coverage(pixels, indices, image_size):
    xy = np.concatenate([pixels[i].reshape(-1, 2) for i in indices]).astype(np.float32)
    return float(cv2.contourArea(cv2.convexHull(xy)) / np.prod(image_size))


def select_views(objects, pixels, image_size, *, min_views=15, progress=None):
    """Return a greedy frontier; held-out observations must not enter this call."""
    if len(objects) != len(pixels) or not 12 <= min_views <= len(objects):
        raise ValueError("Need matching observations and at least min_views >= 12")
    active = list(range(len(objects)))
    initial = solve_observations(objects, pixels, image_size)
    angles, tilted, regions = geometry_features(objects, pixels, image_size, initial)
    initial_hull = coverage(pixels, active, image_size)
    corners = {region for region in regions if region[0] != 1 and region[1] != 1}
    span_floor = np.minimum(np.ptp(angles, axis=0), 15.0)
    tilted_floor = min(4, int(tilted.sum()))

    def geometry_ok(indices):
        return (
            coverage(pixels, indices, image_size) >= 0.9 * initial_hull
            and corners.issubset({regions[i] for i in indices})
            and int(tilted[indices].sum()) >= tilted_floor
            and np.all(np.ptp(angles[indices], axis=0) >= span_floor - 1e-9)
        )

    frontier = [{"indices": active.copy(), "fit": initial, "removed_index": None}]
    while len(active) > min_views:
        choices = []
        for removed in active:
            indices = [i for i in active if i != removed]
            if not geometry_ok(indices):
                continue
            try:
                fit = solve_observations(
                    [objects[i] for i in indices], [pixels[i] for i in indices], image_size
                )
            except (ValueError, cv2.error):
                continue
            if not np.isfinite(fit["rms_px"]):
                continue
            choices.append((fit["rms_px"], max(fit["per_view_rms_px"]), removed, indices, fit))
        if not choices:
            break
        _, _, removed, active, fit = min(choices, key=lambda row: row[:3])
        frontier.append({"indices": active.copy(), "fit": fit, "removed_index": removed})
        if progress:
            progress(len(active), fit["rms_px"])
    passing = [point for point in frontier if point["fit"]["quality_pass"]]
    # First passing point has the most retained images on this deterministic path.
    chosen = passing[0] if passing else frontier[-1]
    return (
        chosen,
        frontier,
        {
            "method": "greedy backward elimination; evaluate every eligible one-view removal",
            "global_optimum_proven": False,
            "min_views": min_views,
            "minimum_hull_fraction_of_initial": 0.9,
            "initial_hull_image_fraction": initial_hull,
            "selected_hull_image_fraction": coverage(pixels, chosen["indices"], image_size),
            "preserved_corner_regions": sorted(corners),
            "minimum_tilted_views": tilted_floor,
            "minimum_xy_normal_angle_span_deg": span_floor.tolist(),
            "pose_note": "Geometry constraints use initial-fit pose estimates, not independent ground truth.",
        },
    )


def heldout_errors(objects, pixels, indices, fit):
    matrix, distortion = (
        np.asarray(fit["camera_matrix"]),
        np.asarray(fit["distortion_coefficients"]),
    )
    errors, squared_all = [], []
    for i in indices:
        ok, rotation, translation = cv2.solvePnP(objects[i], pixels[i], matrix, distortion)
        if not ok:
            raise ValueError("Held-out pose failed")
        projected, _ = cv2.projectPoints(objects[i], rotation, translation, matrix, distortion)
        squared = np.sum((pixels[i].reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)
        errors.append({"index": i, "rms_px": float(np.sqrt(squared.mean()))})
        squared_all.extend(squared.tolist())
    return errors, squared_all
