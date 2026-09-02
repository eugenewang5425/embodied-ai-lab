"""Benchmark metric depth models on collected board-pose ground-truth frames.

Every saved calibrated metric capture contains the undistorted RGB, the
rectified intrinsics K', and (for our Depth Anything V2 baseline) the model
depth array. The ChArUco board pose recovered from the frame gives per-pixel
true optical-axis depth on the board interior, so each model's depth map can
be scored on the same pixels: median required scale, in-plane MAD, raw error.

Usage:
  python bench/eval_models.py --root outputs --out bench/outputs/benchmark.json
  python bench/eval_models.py --root outputs --probe-one <metadata.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from monocular_depth.calibration import (
    detect_board,
    make_board,
)
from monocular_depth.distance import board_pose, plane_depth_map, region_mask
from monocular_depth.geometry import load_metric_capture
from monocular_depth.records import read_json


def distance_band(true_m: float) -> str:
    if true_m < 0.55:
        return "0.4m"
    if true_m < 0.66:
        return "0.6m"
    if true_m < 0.82:
        return "0.7m"
    if true_m < 1.06:
        return "0.9m"
    return "1.2m"


def frame_truth(record, rgb, matrix):
    """Board pose ground truth on the board interior; None if detection fails."""
    spec = record["calibration"]["board"]
    detection = detect_board(rgb, make_board(spec))
    if detection is None:
        return None
    _, image_points, _, ids = detection
    if len(ids) < 10:
        return None
    pose = board_pose(detection, matrix)
    true_z = plane_depth_map(pose, matrix, (rgb.shape[0], rgb.shape[1]))
    mask = region_mask(image_points, (rgb.shape[0], rgb.shape[1]), 0.12)
    finite = np.isfinite(true_z) & (mask > 0)
    return true_z, finite, pose


def score(pred_depth, true_z, finite):
    z_true = true_z[finite].astype(np.float64)
    z_pred = pred_depth[finite].astype(np.float64)
    valid = np.isfinite(z_pred) & (z_pred > 0)
    if valid.sum() < 1000:
        return None
    ratio = z_true[valid] / z_pred[valid]
    scale = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - scale)))
    raw_rel = float((np.median(z_pred[valid]) - np.median(z_true[valid])) / np.median(z_true[valid]))
    return {
        "interior_pixels": int(valid.sum()),
        "true_median_m": float(np.median(z_true[valid])),
        "pred_median_m": float(np.median(z_pred[valid])),
        "scale": scale,
        "scale_mad": mad,
        "raw_relative_error": raw_rel,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="scan *_metadata.json below this dir")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--models", nargs="+", default=["da2", "metric3d", "unidepth", "depthpro"])
    parser.add_argument("--probe-one", type=Path)
    args = parser.parse_args()

    if args.probe_one:
        probe(args.probe_one)
        return

    frames = []
    for meta in sorted(args.root.rglob("*_metadata.json")):
        record = read_json(meta)
        if record.get("variant") != "metric":
            continue
        frames.append(meta)
    if not frames:
        parser.error("No metric capture metadata found under --root")
    print(f"Frames: {len(frames)}")

    results = []
    for meta in frames:
        try:
            record, rgb, da2_depth, matrix = load_metric_capture(meta)
        except Exception as error:  # noqa: BLE001
            print(f"skip {meta.name[:28]} ({str(error)[:60]})")
            continue
        truth = frame_truth(record, rgb, matrix)
        if truth is None:
            print(f"skip {meta.name[:28]} (no board)")
            continue
        true_z, finite, pose = truth
        entry = {
            "capture": meta.name,
            "source": str(meta.parent),
            "distance_band": distance_band(float(np.median(true_z[finite]))),
            "pose_tilt_deg": pose["normal_angle_to_optical_axis_deg"],
            "pose_rms_px": pose["reprojection_rms_px"],
            "scores": {},
        }
        if "da2" in args.models:
            entry["scores"]["da2"] = score(da2_depth, true_z, finite)
        if "metric3d" in args.models:
            entry["scores"]["metric3d"] = score_metric3d(rgb, true_z, finite)
        if "unidepth" in args.models:
            entry["scores"]["unidepth"] = score_unidepth(rgb, matrix, true_z, finite)
        if "depthpro" in args.models:
            entry["scores"]["depthpro"] = score_depthpro(rgb, matrix, true_z, finite)
        results.append(entry)
        print(f"{entry['capture'][:28]} {entry['distance_band']:>5} "
              + " ".join(f"{k}={v['scale']:.4f}/{v['scale_mad']:.4f}" for k, v in entry["scores"].items() if v))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {args.out.resolve()}")


def probe(meta_path):
    _, rgb, _, matrix = load_metric_capture(meta_path)
    print("rgb:", rgb.shape, "K':", np.round(np.asarray(matrix), 1).tolist())
    from onnxruntime import InferenceSession

    sess = InferenceSession(
        str(PROJECT / "bench" / "weights" / "metric3d_vits_fp16.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    for inp in sess.get_inputs():
        print("onnx input:", inp.name, inp.shape, inp.type)
    for outp in sess.get_outputs():
        print("onnx output:", outp.name, outp.shape, outp.type)


def score_metric3d(rgb, true_z, finite):
    from onnxruntime import InferenceSession, SessionOptions

    options = SessionOptions()
    options.log_severity_level = 3
    sess = InferenceSession(
        str(PROJECT / "bench" / "weights" / "metric3d_vits_fp16.onnx"),
        sess_options=options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    inp = sess.get_inputs()[0]
    resized = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_LINEAR)
    image = resized[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)
    if "float16" in inp.type:
        image = image.astype(np.float16)
    input_tensor = image
    started = time.perf_counter()
    outputs = sess.run(None, {inp.name: input_tensor})
    runtime = time.perf_counter() - started
    out = np.asarray(outputs[0])
    if out.ndim == 4:
        out = out[0, 0]
    if out.ndim == 3:
        out = out[0]
    pred = cv2.resize(out.astype(np.float32), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    res = score(pred, true_z, finite)
    if res is not None:
        res["runtime_s"] = runtime
    return res


_DEPTHPRO_LOADED = {"ok": False, "model": None, "transform": None, "error": None}


def score_depthpro(rgb, matrix, true_z, finite):
    import torch

    if not _DEPTHPRO_LOADED["ok"]:
        import copy

        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT

        try:
            config = copy.copy(DEFAULT_MONODEPTH_CONFIG_DICT)
            config.checkpoint_uri = str(PROJECT / "bench" / "weights" / "depth_pro.pt")
            model, transform = create_model_and_transforms(
                config=config, device="cuda", precision=torch.float16
            )
            model.eval()
            _DEPTHPRO_LOADED.update(ok=True, model=model, transform=transform)
        except Exception as error:  # noqa: BLE001
            _DEPTHPRO_LOADED["error"] = repr(error)
            return None
    if _DEPTHPRO_LOADED["error"]:
        return None

    model = _DEPTHPRO_LOADED["model"]
    transform = _DEPTHPRO_LOADED["transform"]
    try:
        image = transform(rgb[:, :, ::-1].copy())
        started = time.perf_counter()
        with torch.no_grad():
            pred = model.infer(image, matrix[0, 0])
        torch.cuda.synchronize()
        runtime = time.perf_counter() - started
    except Exception as error:  # noqa: BLE001
        _DEPTHPRO_LOADED["error"] = repr(error)
        return None
    depth_out = pred["depth"]
    if hasattr(depth_out, "squeeze"):
        depth_out = depth_out.squeeze()
    depth_out = np.asarray(
        depth_out.cpu().numpy() if hasattr(depth_out, "cpu") else depth_out,
        dtype=np.float32,
    )
    depth = depth_out.squeeze() if depth_out.ndim != 2 else depth_out
    if depth.ndim != 2 or depth.shape != (rgb.shape[0], rgb.shape[1]):
        import cv2

        depth = cv2.resize(
            depth.astype(np.float32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    res = score(depth, true_z, finite)
    if res is not None:
        res["runtime_s"] = runtime
        res["focal_px"] = float(pred["focallength_px"])
    return res


_UNIDEPTH_LOADED = {"ok": False, "model": None, "error": None}


def score_unidepth(rgb, matrix, true_z, finite):
    if not _UNIDEPTH_LOADED["ok"]:
        import types

        sys.modules.setdefault("wandb", types.ModuleType("wandb"))

        import torch

        repo = PROJECT / "bench" / "third_party" / "UniDepth"
        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "unidepth"))
        from hubconf import UniDepth
        from safetensors.torch import load_file

        try:
            model = UniDepth(version="v2", backbone="vits14", pretrained=False)
            state = load_file(str(PROJECT / "bench" / "weights" / "unidepth_v2_vits14.bin"))
            result = model.load_state_dict(state, strict=False)
            print("UniDepth state load | missing:", len(result.missing_keys), "unexpected:", len(result.unexpected_keys))
            for key in result.missing_keys[:8]:
                print("  missing:", key)
            for key in result.unexpected_keys[:8]:
                print("  unexpected:", key)
            if result.missing_keys or result.unexpected_keys:
                raise ValueError("UniDepth checkpoint keys do not match the architecture")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            _UNIDEPTH_LOADED.update(ok=True, model=model)
        except Exception as error:  # noqa: BLE001
            _UNIDEPTH_LOADED["error"] = repr(error)
            return None
    if _UNIDEPTH_LOADED["error"]:
        return None
    import torch
    from unidepth.utils.camera import Pinhole

    model = _UNIDEPTH_LOADED["model"]
    device = next(model.parameters()).device
    image = torch.from_numpy(rgb[:, :, ::-1].copy()).permute(2, 0, 1).contiguous().to(device)
    camera = Pinhole(K=torch.tensor(np.asarray(matrix, dtype=np.float32), device=device))
    started = time.perf_counter()
    with torch.no_grad():
        preds = model.infer(image, camera)
    runtime = time.perf_counter() - started
    depth = preds["depth"]
    if torch.is_tensor(depth):
        depth = depth.cpu().numpy()
    depth = np.asarray(depth).squeeze()
    if depth.ndim != 2:
        depth = depth[0]
    if depth.shape != (rgb.shape[0], rgb.shape[1]):
        depth = cv2.resize(
            depth.astype(np.float32),
            (rgb.shape[1], rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    res = score(depth, true_z, finite)
    if res is not None:
        res["runtime_s"] = runtime
        res["depth_definition"] = "infer_raw"
    return res


if __name__ == "__main__":
    main()
