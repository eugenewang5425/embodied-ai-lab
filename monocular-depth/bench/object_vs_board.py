"""Compare model depth on a real object and the calibration board in one frame.

Setup required: the ChArUco board and a real flat-faced object are both placed
flat against the SAME plane (e.g., leaning on the same wall), so the board pose
recovered from the frame gives the true depth on that whole plane. The object
region (given by the user or picked visually) must lie on the same plane.

For every saved capture, we score board region and object region separately for
the DA-V2 depth already stored in the capture and for any model depth array we
are handed (e.g., UniDepth output from the benchmark path). The question: does
the scale measured on the board transfer to the real object?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--object-xywh", nargs=4, type=int, required=True,
                        help="object bbox in the undistorted image: x y w h")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(PROJECT / "src"))
    from monocular_depth.calibration import detect_board, make_board
    from monocular_depth.distance import board_pose, plane_depth_map, region_mask
    from monocular_depth.geometry import load_metric_capture
    x, y, w, h = args.object_xywh
    rows = []
    for meta in sorted(args.capture_dir.glob("*_metadata.json")):
        try:
            record, rgb, depth, matrix = load_metric_capture(meta)
        except Exception as error:  # noqa: BLE001
            print(f"skip {meta.name[:28]} ({str(error)[:50]})")
            continue
        spec = record["calibration"]["board"]
        detection = detect_board(rgb, make_board(spec))
        if detection is None:
            print(f"skip {meta.name[:28]} (no board)")
            continue
        pose = board_pose(detection, matrix)
        true_z = plane_depth_map(pose, matrix, (rgb.shape[0], rgb.shape[1]))
        finite = np.isfinite(true_z) & (rgb > 0).all(axis=2)
        board_mask = region_mask(detection[1], (rgb.shape[0], rgb.shape[1]), 0.12) > 0
        object_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
        object_mask[y : y + h, x : x + w] = True
        object_mask &= finite

        def stats(pred_depth, mask=object_mask, true=true_z):
            valid = mask & np.isfinite(pred_depth) & (pred_depth > 0)
            if valid.sum() < 500:
                return None
            ratio = true[valid] / pred_depth[valid]
            return {
                "pixels": int(valid.sum()),
                "true_median_m": float(np.median(true[valid])),
                "pred_median_m": float(np.median(pred_depth[valid])),
                "scale": float(np.median(ratio)),
                "mad": float(np.median(np.abs(ratio - np.median(ratio)))),
            }

        entry = {
            "capture": meta.name,
            "board_pose_plane_m": pose["plane_distance_m"],
            "board_tilt_deg": pose["normal_angle_to_optical_axis_deg"],
            "board": stats(depth),
            "board_scale": float(
                np.nanmedian(
                    true_z[board_mask & np.isfinite(depth) & (depth > 0)]
                    / depth[board_mask & np.isfinite(depth) & (depth > 0)]
                )
            ),
            "object_da2": stats(depth),
        }
        rows.append(entry)
        print(
            f"{entry['capture'][:28]} plane={entry['board_pose_plane_m']:.3f} "
            f"board_scale={entry['board_scale']:.4f} obj_scale="
            f"{None if entry['object_da2'] is None else round(entry['object_da2']['scale'], 4)}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
