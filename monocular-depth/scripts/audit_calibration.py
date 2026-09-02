"""Audit one calibration session without promoting it or changing original photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from monocular_depth.calibration import detect_board, make_board, solve_observations
from monocular_depth.focus import validate_frame_focus
from monocular_depth.io import read_image, write_image
from monocular_depth.records import read_json, write_json


def pose_error(obj, img, matrix, distortion):
    ok, rotation, translation = cv2.solvePnP(obj, img, matrix, distortion)
    if not ok:
        raise ValueError("Pose estimation failed")
    projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
    squared = np.sum((img.reshape(-1, 2) - projected.reshape(-1, 2)) ** 2, axis=1)
    normal = cv2.Rodrigues(rotation)[0][:, 2]
    tilt = np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1)))
    return float(np.sqrt(squared.mean())), float(tilt), squared


def audit(images: Path, candidate: Path, output: Path, physical_size_confirmed: bool = False):
    output.mkdir(parents=True, exist_ok=False)
    record = read_json(candidate)
    board = make_board(record["board"])
    matrix = np.asarray(record["camera_matrix"], dtype=float)
    distortion = np.asarray(record["distortion_coefficients"], dtype=float)
    width, height = record["image_size"]
    parameters = cv2.aruco.CharucoParameters()
    parameters.cameraMatrix = matrix
    parameters.distCoeffs = distortion
    refined_detector = cv2.aruco.CharucoDetector(board, parameters)
    objects, pixels, refined_objects, refined_pixels, views = [], [], [], [], []
    thumbnails, focus_targets = [], []
    session = record.get("capture_session") or {}
    require_focus = bool(session.get("focus_lock_required"))
    worst_indices = set(np.argsort(record["per_view_rms_px"])[-4:].tolist())
    for index, filename in enumerate(record["accepted_images"]):
        image_path = Path(record.get("source_image_paths", {}).get(filename, str(images / filename)))
        frame = read_image(image_path)
        if frame is None or frame.shape[:2] != (height, width):
            raise ValueError(f"Unreadable or mismatched image: {filename}")
        detection = detect_board(frame, board)
        if detection is None:
            raise ValueError(f"Previously accepted detection missing: {filename}")
        obj, img, _corners, ids = detection
        objects.append(obj)
        pixels.append(img)
        xy = img.reshape(-1, 2)
        minimum, maximum = xy.min(axis=0), xy.max(axis=0)
        if require_focus:
            trace = read_json(image_path.with_suffix(".json"))
            target = validate_frame_focus(trace["focus_before_frame"], trace["focus_after_frame"])
            if trace["image_file"] != image_path.name or trace["image_size"] != [width, height]:
                raise ValueError("Focus trace does not match image")
            if focus_targets and target != focus_targets[0]:
                raise ValueError("Mixed focus targets in audit")
            focus_targets.append(target)
        rms, tilt, _ = pose_error(obj, img, matrix, distortion)
        undistorted = cv2.undistortPoints(img, matrix, distortion, P=matrix)
        homography, _ = cv2.findHomography(obj.reshape(-1, 3)[:, :2], undistorted, method=0)
        plane_pixels = cv2.perspectiveTransform(obj.reshape(-1, 1, 3)[:, :, :2], homography)
        plane_rms = float(np.sqrt(np.mean(np.sum((plane_pixels - undistorted) ** 2, axis=2))))
        refined_corners, refined_ids, _, _ = refined_detector.detectBoard(frame)
        if refined_ids is None or not np.array_equal(refined_ids, ids):
            raise ValueError("Camera-aware refinement changed corner IDs; comparison refused")
        ref_obj, ref_img = board.matchImagePoints(refined_corners, refined_ids)
        refined_objects.append(ref_obj)
        refined_pixels.append(ref_img)
        views.append(
            {
                "file": filename,
                "corners": len(ids),
                "rms_px": rms,
                "tilt_deg_from_candidate": tilt,
                "undistorted_planar_homography_rms_px": plane_rms,
                "corner_center_fraction": (xy.mean(axis=0) / [width, height]).tolist(),
                "corner_bbox_fraction": (
                    np.concatenate([minimum, maximum]) / [width, height, width, height]
                ).tolist(),
                "refinement_mean_corner_shift_px": float(
                    np.linalg.norm(ref_img - img, axis=2).mean()
                ),
            }
        )
        # Diagnostic crops only; never resample or modify the source observations.
        x0, y0 = np.maximum(np.floor(minimum - 100).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(maximum + 100).astype(int), [width, height])
        crop = frame[y0:y1, x0:x1]
        if index in worst_indices or record["per_view_rms_px"][index] > 1.5:
            write_image(output / f"board_crop_{Path(filename).stem}.png", crop)
        tile = np.full((260, 420, 3), 245, dtype=np.uint8)
        scale = min(400 / crop.shape[1], 220 / crop.shape[0])
        preview = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile[30:30 + preview.shape[0], 10:10 + preview.shape[1]] = preview
        cv2.putText(
            tile, f"{Path(filename).stem}  RMS {rms:.3f}  tilt {tilt:.0f}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
        )
        thumbnails.append(tile)

    for start in range(0, len(thumbnails), 16):
        page = thumbnails[start:start + 16]
        page += [np.full_like(thumbnails[0], 245)] * (16 - len(page))
        sheet = np.vstack([np.hstack(page[row:row + 4]) for row in range(0, 16, 4)])
        write_image(output / f"board_contact_{start // 16 + 1}.jpg", sheet)

    folds, held_squared = [], []
    for fold in range(5):
        train = [i for i in range(len(objects)) if i % 5 != fold]
        held = [i for i in range(len(objects)) if i % 5 == fold]
        fit = solve_observations(
            [objects[i] for i in train], [pixels[i] for i in train], (width, height)
        )
        k = np.asarray(fit["camera_matrix"])
        d = np.asarray(fit["distortion_coefficients"])
        results = []
        for i in held:
            rms, _, squared = pose_error(objects[i], pixels[i], k, d)
            held_squared.extend(squared.tolist())
            results.append({"file": views[i]["file"], "rms_px": rms})
        folds.append(
            {
                "fold": fold,
                "train_views": len(train),
                "train_rms_px": fit["rms_px"],
                "camera_matrix": k.tolist(),
                "heldout": results,
            }
        )

    flagged = [i for i, v in enumerate(views) if v["rms_px"] > 1.5]
    kept = [i for i in range(len(objects)) if i not in flagged]
    sensitivity = None
    if len(kept) >= 12:
        sensitivity = solve_observations(
            [objects[i] for i in kept], [pixels[i] for i in kept], (width, height)
        )
    refinement = solve_observations(refined_objects, refined_pixels, (width, height))
    all_xy = np.concatenate(pixels).reshape(-1, 2)
    histogram, _, _ = np.histogram2d(
        all_xy[:, 1], all_xy[:, 0], bins=(3, 4), range=[[0, height], [0, width]]
    )
    hull_fraction = cv2.contourArea(cv2.convexHull(all_xy)) / (width * height)
    fx_fy = np.array([np.asarray(f["camera_matrix"])[[0, 1], [0, 1]] for f in folds])
    summary = {
        "source_session": str(images.resolve()),
        "source_candidate": str(candidate.resolve()),
        "image_size": [width, height],
        "views": len(views),
        "baseline_rms_px": record["rms_px"],
        "baseline_quality_pass": record["quality_pass"],
        "high_error_files": [views[i]["file"] for i in flagged],
        "cross_validation_5fold_rms_px": float(np.sqrt(np.mean(held_squared))),
        "cross_validation_note": "Intrinsics exclude each held-out view; its pose is fitted on its corners. Same-session consistency only, not independent metric validation.",
        "fold_focal_range_px": {
            "min": fx_fy.min(axis=0).tolist(),
            "max": fx_fy.max(axis=0).tolist(),
        },
        "corner_coverage_3rows_4cols": histogram.astype(int).tolist(),
        "corner_convex_hull_image_fraction": hull_fraction,
        "tilt_range_deg": [
            min(v["tilt_deg_from_candidate"] for v in views),
            max(v["tilt_deg_from_candidate"] for v in views),
        ],
        "diagnostic_excluding_high_error_views": {
            "kept_views": len(kept),
            "computed": sensitivity is not None,
            "rms_px": sensitivity["rms_px"] if sensitivity else None,
            "max_view_rms_px": max(sensitivity["per_view_rms_px"]) if sensitivity else None,
            "quality_pass": sensitivity["quality_pass"] if sensitivity else None,
            "camera_matrix": sensitivity["camera_matrix"] if sensitivity else None,
            "skip_reason": None if sensitivity else "Fewer than 12 views remain; no refit performed",
            "note": "Residual-selected sensitivity analysis only, NOT a promoted calibration or an independent validation.",
        },
        "diagnostic_camera_aware_corner_refinement": {
            "rms_px": refinement["rms_px"],
            "quality_pass": refinement["quality_pass"],
            "camera_matrix": refinement["camera_matrix"],
            "note": "Uses the baseline candidate during corner interpolation; diagnostic only.",
        },
        "autofocus_reported_at_session_start": session.get("autofocus_reported"),
        "focus_trace_available": bool(focus_targets),
        "focus_verified_views": len(focus_targets),
        "fixed_focus_driver_units": focus_targets[0] if focus_targets else None,
        "promoted_to_default": False,
        "physical_board_dimensions_confirmed_by_user": physical_size_confirmed,
    }
    write_json(
        output / "audit.json", {**summary, "per_view": views, "cross_validation_folds": folds}
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-size-confirmed", action="store_true")
    args = parser.parse_args()
    audit(args.images, args.candidate, args.output, args.physical_size_confirmed)
