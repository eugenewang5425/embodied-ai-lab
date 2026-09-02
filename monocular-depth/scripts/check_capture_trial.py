"""Inspect a small capture trial without fitting or promoting camera intrinsics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from monocular_depth.calibration import detect_board, make_board
from monocular_depth.focus import validate_frame_focus
from monocular_depth.io import read_image, write_image
from monocular_depth.records import read_json, write_json


def check_trial(images: Path, output: Path):
    session = read_json(images / "session.json")
    if not session.get("focus_lock_required"):
        raise ValueError("This trial checker requires per-frame focus records")
    board = make_board(session["board"])
    files = sorted(images.glob("view_*.png"))
    if not files:
        raise ValueError("No saved photos in this session")
    output.mkdir(parents=True, exist_ok=False)
    rows, tiles, signatures = [], [], []
    target = None
    for path in files:
        frame = read_image(path)
        if frame is None:
            raise ValueError(f"Unreadable photo: {path.name}")
        height, width = frame.shape[:2]
        record = read_json(path.with_suffix(".json"))
        current = validate_frame_focus(record["focus_before_frame"], record["focus_after_frame"])
        if target is not None and current != target:
            raise ValueError("Mixed focus targets")
        target = current
        if record["image_file"] != path.name or record["image_size"] != [width, height]:
            raise ValueError("Photo/record mismatch")
        if session["actual_image_size"] != [width, height]:
            raise ValueError("Photo/session resolution mismatch")
        detection = detect_board(frame, board)
        if detection is None:
            raise ValueError(f"Insufficient corners: {path.name}")
        _, img, _, ids = detection
        xy = img.reshape(-1, 2)
        duplicate = next(
            (
                old_name for old_name, old_ids, old_xy in signatures
                if np.array_equal(old_ids, ids.ravel())
                and np.linalg.norm(old_xy - xy, axis=1).mean() < 5.0
            ), None,
        )
        signatures.append((path.name, ids.ravel(), xy))
        low, high = xy.min(axis=0), xy.max(axis=0)
        padding = max(float(np.max(high - low)) * 0.3, 100)
        x0, y0 = np.maximum(np.floor(low - padding).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(high + padding).astype(int), [width, height])
        crop = frame[y0:y1, x0:x1]
        write_image(output / f"crop_{path.stem}.png", crop)
        tile = np.full((360, 480, 3), 245, dtype=np.uint8)
        scale = min(460 / crop.shape[1], 310 / crop.shape[0])
        preview = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tile[40:40 + preview.shape[0], 10:10 + preview.shape[1]] = preview
        cv2.putText(
            tile, f"{path.stem} | {len(ids)} corners | focus {target:g}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA,
        )
        tiles.append(tile)
        rows.append({
            "file": path.name,
            "image_size": [width, height],
            "corners": len(ids),
            "fixed_focus_driver_units": target,
            "near_duplicate_of": duplicate,
            "inner_corner_center_fraction": (xy.mean(axis=0) / [width, height]).tolist(),
            "inner_corner_bbox_fraction": (
                np.concatenate([low, high]) / [width, height, width, height]
            ).tolist(),
        })
    for start in range(0, len(tiles), 9):
        page = tiles[start:start + 9]
        page += [np.full_like(tiles[0], 245)] * (9 - len(page))
        sheet = np.vstack([np.hstack(page[row:row + 3]) for row in range(0, 9, 3)])
        write_image(output / f"contact_{start // 9 + 1}.jpg", sheet)
    result = {
        "source_session": str(images.resolve()),
        "photos": len(rows),
        "focus_verified_photos": len(rows),
        "fixed_focus_driver_units": target,
        "calibration_computed": False,
        "note": "Capture precheck only. No optical sharpness guarantee or calibration pass.",
        "per_view": rows,
    }
    write_json(output / "trial.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_trial(args.images, args.output)
