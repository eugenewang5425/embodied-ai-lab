"""Coplanar board vs real-object transfer analysis across models.

Regions (raw undistorted coords) must lie on the same plane as the ChArUco
board (leaning on the monitor box). The phone itself is a tilted object and its
numbers are reported with a caveat, not as a coplanar reference.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from monocular_depth.calibration import detect_board, make_board
from monocular_depth.distance import board_pose, plane_depth_map, region_mask
from monocular_depth.geometry import load_metric_capture

REGIONS = {
    "boxface": (2660, 300, 190, 520),
    "phone": (2323, 252, 330, 650),
    "base": (2305, 1315, 610, 130),
}


def region_stats(true_z, pred, rect, rgb):
    x, y, w, h = rect
    mask = np.zeros(true_z.shape, dtype=bool)
    mask[y : y + h, x : x + w] = True
    valid = mask & np.isfinite(true_z) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 500:
        return None
    ratio = true_z[valid] / pred[valid]
    return {
        "pixels": int(valid.sum()),
        "true_median_m": float(np.median(true_z[valid])),
        "pred_median_m": float(np.median(pred[valid])),
        "scale": float(np.median(ratio)),
        "mad": float(np.median(np.abs(ratio - np.median(ratio)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--models", nargs="+", default=["da2", "unidepth", "depthpro"])
    args = parser.parse_args()

    unidepth = None
    depthpro = (None, None)

    rows = []
    for meta in sorted(args.capture_dir.glob("*_metadata.json")):
        try:
            record, rgb, depth, matrix = load_metric_capture(meta)
        except Exception as error:  # noqa: BLE001
            print("skip", meta.name[:28], str(error)[:50])
            continue
        detection = detect_board(rgb, make_board(record["calibration"]["board"]))
        if detection is None:
            print("skip", meta.name[:28], "no board")
            continue
        pose = board_pose(detection, matrix)
        true_z = plane_depth_map(pose, matrix, (rgb.shape[0], rgb.shape[1]))
        board_mask = region_mask(detection[1], (rgb.shape[0], rgb.shape[1]), 0.12) > 0

        preds = {"da2": depth}
        if "unidepth" in args.models:
            if unidepth is None:
                from monocular_depth.unidepth_model import infer_unidepth, load_unidepth

                model, _ = load_unidepth()
                unidepth = model
            import torch

            preds["unidepth"] = infer_unidepth(unidepth, rgb, matrix)
            torch.cuda.synchronize()
        if "depthpro" in args.models:
            if depthpro[0] is None:
                import torch
                from depth_pro import create_model_and_transforms
                from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

                config = copy.copy(DEFAULT_MONODEPTH_CONFIG_DICT)
                config.checkpoint_uri = str(PROJECT / "bench" / "weights" / "depth_pro.pt")
                model, transform = create_model_and_transforms(
                    config=config, device="cuda", precision=torch.float16
                )
                model.eval()
                depthpro = (model, transform)
            import cv2
            import torch

            model, transform = depthpro
            image = transform(rgb[:, :, ::-1].copy())
            with torch.no_grad():
                pred = model.infer(image, matrix[0, 0])
            torch.cuda.synchronize()
            p = np.asarray(pred["depth"].cpu().numpy()).squeeze()
            if p.ndim != 2 or p.shape != (rgb.shape[0], rgb.shape[1]):
                p = cv2.resize(
                    p.astype(np.float32),
                    (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            preds["depthpro"] = p

        entry = {"capture": meta.name, "tilt_deg": pose["normal_angle_to_optical_axis_deg"],
                 "regions": {}}
        for key, rect in REGIONS.items():
            entry["regions"][key] = {}
            for model, pred in preds.items():
                entry["regions"][key][model] = region_stats(true_z, pred, rect, rgb)
        bstats = {}
        for model, pred in preds.items():
            valid = board_mask & np.isfinite(true_z) & np.isfinite(pred) & (pred > 0)
            ratio = true_z[valid] / pred[valid]
            bstats[model] = float(np.median(ratio))
        entry["board_scale"] = bstats
        rows.append(entry)
        board = entry["board_scale"]
        print(
            f"{meta.name[17:29]} tilt={entry['tilt_deg']:5.1f} | board "
            + " ".join(f"{m}={v:.4f}" for m, v in board.items())
            + " | boxface/phone/base "
            + " ".join(
                f"{reg}:{entry['regions'][reg][m]['scale']:.4f}" if entry['regions'][reg].get(m)
                else f"{reg}:n/a"
                for m in board for reg in ("boxface", "phone", "base")
            )
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
