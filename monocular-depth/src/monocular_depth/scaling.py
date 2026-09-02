"""Per-frame global scale correction from a known-size planar target.

The benchmark on our board-ground-truth set showed that model depth error is a
per-image global scale: the ratio board/object within one frame is 1.00 ±1%
even though the absolute scale wanders ±10% frame to frame. So the correction
is: detect the ChArUco board in the frame, recover the board plane from the
calibrated intrinsics, take the median ratio true/predicted on the board
interior, and multiply the whole depth map by that scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibration import detect_board, make_board
from .distance import board_pose, plane_depth_map, region_mask
from .records import timestamp, write_json


def estimate_scale(depth, rgb, matrix, board_spec, margin_fraction=0.12, min_pixels=2000):
    """Median true/predicted ratio on the board interior; None if no board."""
    matrix = np.asarray(matrix, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float32)
    detection = detect_board(rgb, make_board(board_spec))
    if detection is None:
        return None
    _, image_points, _, ids = detection
    if len(ids) < 10:
        return None
    pose = board_pose(detection, matrix)
    true_z = plane_depth_map(pose, matrix, (rgb.shape[0], rgb.shape[1]))
    mask = region_mask(image_points, (rgb.shape[0], rgb.shape[1]), margin_fraction) > 0
    finite = np.isfinite(true_z) & mask & np.isfinite(depth) & (depth > 0)
    if int(finite.sum()) < min_pixels:
        return None
    ratio = true_z[finite] / depth[finite]
    scale = float(np.median(ratio))
    return {
        "scale": scale,
        "mad": float(np.median(np.abs(ratio - scale))),
        "interior_pixels": int(finite.sum()),
        "corner_count": len(ids),
        "board_tilt_deg": pose["normal_angle_to_optical_axis_deg"],
        "plane_distance_m": pose["plane_distance_m"],
        "pose_rms_px": pose["reprojection_rms_px"],
    }


def apply_scale(depth, scale) -> np.ndarray:
    return (np.asarray(depth, dtype=np.float32) * float(scale)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-frame global scale from the board; writes scaled depth"
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", choices=("da2-file", "unidepth", "depthpro"), default="da2-file")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from monocular_depth.geometry import load_metric_capture

    record, rgb, depth, matrix = load_metric_capture(args.metadata)
    if args.model == "unidepth":
        from .unidepth_model import infer_unidepth, load_unidepth

        model, _ = load_unidepth()
        depth = infer_unidepth(model, rgb, matrix)
    elif args.model == "depthpro":
        from .depthpro_model import infer_depthpro, load_depthpro

        model, transform = load_depthpro()
        depth = infer_depthpro(model, transform, rgb, matrix)
    info = estimate_scale(depth, rgb, matrix, record["calibration"]["board"])
    if info is None:
        parser.exit(2, "Board not detected in the frame; refused to calibrate scale\n")
    corrected = apply_scale(depth, info["scale"])
    out_dir = args.out_dir or args.metadata.parent
    stem = args.metadata.stem.replace("_metadata", "") + f"_scaled_{args.model}"
    depth_path = out_dir / f"{stem}_depth.npy"
    np.save(depth_path, corrected)
    report = {
        "schema_version": 1,
        "timestamp_utc": timestamp(),
        "source_metadata": str(args.metadata.resolve()),
        "model": args.model,
        "frame_scale": info,
        "scaled_depth_file": depth_path.name,
        "note": "Scale applies to the whole frame; object/board transfer validated at ±1% (80 cm coplanar set).",
        "not_for_safety": True,
    }
    write_json(out_dir / f"{stem}_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
