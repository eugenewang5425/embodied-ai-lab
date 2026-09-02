from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import cv2
import torch

from .capture import VerifiedCapture
from .config import (
    DEFAULT_CAMERA_FOURCC,
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    PROJECT_ROOT,
    VARIANTS,
)
from .exposure import ExposureLock
from .io import save_result
from .model import comparison_view, load_model
from .pipeline import DepthPipeline


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    if width <= 0 or height <= 0:
        raise ValueError("Camera width and height must be positive")
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {index}; close other camera applications")
        # This device exposes 4K through MJPEG, not uncompressed YUY2.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*DEFAULT_CAMERA_FOURCC))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, DEFAULT_CAMERA_FPS)
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError(
                "Camera returned no frame; close other camera applications and retry"
            )
        actual = (frame.shape[1], frame.shape[0])
        if actual != (width, height):
            raise RuntimeError(
                f"Camera resolution mismatch: requested {width}x{height}, "
                f"received {actual[0]}x{actual[1]}; refusing silent fallback"
            )
        print(
            f"Camera {index}: verified {actual[0]}x{actual[1]} frame; "
            f"requested {DEFAULT_CAMERA_FOURCC} at {DEFAULT_CAMERA_FPS} FPS "
            "(capture frame rate is not benchmarked)"
        )
        return camera
    except Exception:
        camera.release()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 Small on a webcam")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--variant", choices=(*VARIANTS, "unidepth", "depthpro"), default="relative")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--gain", type=float, help="lock camera gain (driver units)")
    parser.add_argument("--wb-temp", type=float, help="lock color-temperature (Kelvin)")
    parser.add_argument(
        "--frame-scale",
        action="store_true",
        help="per-frame global scale from the ChArUco board (validated on coplanar targets)",
    )
    parser.add_argument("--headless", action="store_true", help="run without an OpenCV window")
    parser.add_argument(
        "--max-frames", type=int, default=0, help="stop after N frames; 0 is unlimited"
    )
    parser.add_argument("--save-last", action="store_true", help="save the last processed frame")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "captures")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be zero or positive")
    if args.headless and args.max_frames == 0:
        parser.error("--headless requires a positive --max-frames value")

    if args.variant == "unidepth":
        from .unidepth_model import UniDepthPipeline, load_unidepth

        model, device = load_unidepth()
        pipeline = UniDepthPipeline(model, args.calibration)
    elif args.variant == "depthpro":
        from .depthpro_model import DepthProPipeline, load_depthpro

        model, transform = load_depthpro()
        pipeline = DepthProPipeline(model, transform, args.calibration)
        device = next(model.parameters()).device
    else:
        model, device = load_model(variant=args.variant)
        pipeline = DepthPipeline(model, args.variant, args.input_size, args.calibration)
    if args.frame_scale and not args.calibration:
        parser.error("--frame-scale requires --calibration")
    if pipeline.calibration and pipeline.calibration["image_size"] != [args.width, args.height]:
        parser.error("Requested camera resolution differs from calibration")
    camera = open_camera(args.camera, args.width, args.height)
    capture = None
    exposure = None
    frame_scale_state = {"scale": None, "info": None}
    smoothed_fps = 0.0
    frame_count = 0
    last_image = None
    last_depth = None
    last_raw = None
    last_metadata = None

    def save_current():
        last_metadata["focus_before_save"] = capture.verify_current(last_metadata)
        if exposure is not None:
            exposure.verify_current()
        record_variant = "metric" if args.variant in ("unidepth", "depthpro") else args.variant
        stem = datetime.now(tz=UTC).strftime("capture_%Y%m%d_%H%M%S_%f_utc") + f"_{args.variant}"
        paths = save_result(
            args.output_dir,
            stem,
            last_image,
            last_depth,
            variant=record_variant,
            metadata=last_metadata,
            raw_image=last_raw if args.calibration else None,
        )
        print(f"Saved {paths[-1].resolve()}")

    print(f"Device: {device}; camera: {args.camera}; press S to save, Q/Esc to quit")
    try:
        capture = VerifiedCapture(camera, pipeline.calibration, args.camera)
        locked = capture.wait_for_lock()
        if locked:
            print(
                f"Calibrated focus LOCKED: {locked['focus']}; autofocus={locked['autofocus']}",
                flush=True,
            )
        if args.gain is not None or args.wb_temp is not None:
            exposure = ExposureLock(camera, gain=args.gain, wb_temperature=args.wb_temp)
            if not exposure.request_lock():
                raise ValueError(f"Exposure control lock refused: {exposure.message}")
            timeout = 5.0
            started = perf_counter()
            while exposure.state != "LOCKED":
                if exposure.state == "FAILED":
                    raise ValueError(f"Exposure control lock failed: {exposure.message}")
                if perf_counter() - started > timeout:
                    raise RuntimeError("Timed out waiting for exposure control lock")
                exposure.observe()
            print(
                f"Exposure controls locked: {exposure.targets}; "
                "exposure itself stays driver-managed (auto exposure)",
                flush=True,
            )
        if not args.headless:
            cv2.namedWindow("Monocular Depth Lab", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Monocular Depth Lab", 1440, 540)
        while True:
            raw_image, capture_metadata = capture.read()
            if raw_image.shape[:2] != (args.height, args.width):
                raise RuntimeError(
                    "Camera resolution changed during capture; restart with a new session"
                )

            started = perf_counter()
            image, depth, metadata = pipeline.process(raw_image)
            metadata.update(capture_metadata)
            metadata["focus_after_inference"] = capture.verify_current(metadata)
            if exposure is not None:
                metadata["exposure_snapshot"] = exposure.observe()
            if args.frame_scale:
                from .scaling import apply_scale, estimate_scale

                info = estimate_scale(
                    depth,
                    image,
                    metadata["camera_matrix"],
                    pipeline.calibration["board"],
                )
                if info is None:
                    info = None if frame_scale_state["scale"] is None else {
                        **frame_scale_state["info"],
                        "stale": True,
                    }
                else:
                    frame_scale_state.update(scale=info["scale"], info=info)
                if info is not None and frame_scale_state["scale"] is not None:
                    depth = apply_scale(depth, frame_scale_state["scale"])
                    info["applied_scale"] = frame_scale_state["scale"]
                    metadata["frame_scale"] = info
            if device.type == "cuda":
                torch.cuda.synchronize()
            frame_time = perf_counter() - started
            current_fps = 1.0 / max(frame_time, 1e-9)
            smoothed_fps = (
                current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps
            )

            view = comparison_view(image, depth)
            overlay_scale = max(1.0, image.shape[1] / 1280)
            cv2.putText(
                view,
                f"{getattr(pipeline, 'provenance', {}).get('model', 'Depth model')} | model+prep {smoothed_fps:.1f} FPS | {args.variant} (unvalidated)",
                (int(20 * overlay_scale), int(35 * overlay_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * overlay_scale,
                (255, 255, 255),
                max(2, int(2 * overlay_scale)),
                cv2.LINE_AA,
            )
            if capture.focus:
                cv2.putText(
                    view,
                    f"Focus LOCKED {capture.focus.target:g} | {args.width}x{args.height} | S save | Q quit",
                    (int(20 * overlay_scale), int(70 * overlay_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * overlay_scale,
                    (0, 255, 0),
                    max(2, int(2 * overlay_scale)),
                    cv2.LINE_AA,
                )
            if args.frame_scale and frame_scale_state["scale"] is not None:
                stale = (metadata.get("frame_scale") or {}).get("stale", False)
                cv2.putText(
                    view,
                    f"frame-scale s={frame_scale_state['scale']:.4f}{' (stale)' if stale else ''}",
                    (int(20 * overlay_scale), int(105 * overlay_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * overlay_scale,
                    (255, 255, 0),
                    max(2, int(2 * overlay_scale)),
                    cv2.LINE_AA,
                )
            last_image, last_depth = image, depth
            last_raw, last_metadata = raw_image, metadata
            frame_count += 1

            if args.headless:
                key = 255
            else:
                cv2.imshow("Monocular Depth Lab", view)
                key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s") and last_image is not None and last_depth is not None:
                save_current()
            if args.max_frames and frame_count >= args.max_frames:
                break
        if args.save_last and last_image is not None and last_depth is not None:
            save_current()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(
        f"Processed {frame_count} frames; model+prep EMA: {smoothed_fps:.1f} FPS (not end-to-end)"
    )
