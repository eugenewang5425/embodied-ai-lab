"""ChArUco intrinsics workflow. No automatic claim of metric-depth accuracy."""

from __future__ import annotations

import argparse
import base64
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH, PROJECT_ROOT
from .focus import FocusLock, validate_frame_focus
from .io import read_image, write_image
from .records import read_json, timestamp, write_json

DEFAULT_BOARD = {
    "squares_x": 7,
    "squares_y": 5,
    "square_length_m": 0.025,
    "marker_length_m": 0.018,
    "dictionary": "DICT_4X4_50",
    "legacy_pattern": False,
}
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DEFAULT_BOARD_PATH = CALIBRATION_DIR / "board" / "board.json"


def make_board(spec: dict) -> cv2.aruco.CharucoBoard:
    sx, sy = int(spec["squares_x"]), int(spec["squares_y"])
    square, marker = float(spec["square_length_m"]), float(spec["marker_length_m"])
    if sx < 3 or sy < 3 or not 0 < marker < square:
        raise ValueError("Invalid board dimensions")
    if not spec["dictionary"].startswith("DICT_"):
        raise ValueError("Invalid ArUco dictionary")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    board = cv2.aruco.CharucoBoard((sx, sy), square, marker, dictionary)
    board.setLegacyPattern(bool(spec.get("legacy_pattern", False)))
    return board


def generate_board(output_dir: Path) -> Path:
    if any((output_dir / name).exists() for name in ("board.json", "board.png", "print.html")):
        raise FileExistsError("Board files already exist; choose a new --output-dir")
    board = make_board(DEFAULT_BOARD)
    pixels = board.generateImage((1400, 1000), marginSize=0, borderBits=1)
    write_image(output_dir / "board.png", pixels)
    write_json(output_dir / "board.json", DEFAULT_BOARD)
    encoded = base64.b64encode(cv2.imencode(".png", pixels)[1]).decode("ascii")
    html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>ChArUco A4 - print at 100%</title><style>
@page {{ size: A4 portrait; margin: 15mm; }}
body {{ margin:0; font-family:sans-serif; }}
img {{ width:175mm; height:125mm; display:block; margin-top:15mm; }}
.ruler {{ border-top:1mm solid black; width:100mm; margin-top:12mm; }}
</style><h2>ChArUco 相机标定板</h2>
<p>A4 纵向，100% / 实际大小。关闭“适应页面”、页眉页脚。</p>
<p>打印后尺量：每格 25 mm；整板 175 × 125 mm。粘在平整硬板上。</p>
<img alt="7 by 5 ChArUco board" src="data:image/png;base64,{encoded}">
<div class="ruler"></div><p>上方黑线长度应为 100 mm。</p>
<p>DICT_4X4_50 · marker 18 mm · legacyPattern=false</p></html>"""
    with (output_dir / "print.html").open("x", encoding="utf-8") as stream:
        stream.write(html)
    return output_dir / "print.html"


def detect_board(image: np.ndarray, board: cv2.aruco.CharucoBoard):
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if ids is None or len(ids) < 10 or board.checkCharucoCornersCollinear(ids):
        return None
    object_points, image_points = board.matchImagePoints(corners, ids)
    return object_points, image_points, corners, ids


def solve_observations(objects: list, pixels: list, image_size: tuple[int, int]) -> dict:
    if len(objects) < 12 or len(objects) != len(pixels):
        raise ValueError("Need at least 12 valid, varied views (20-40 recommended)")
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        objects, pixels, image_size, None, None
    )
    errors = []
    for obj, img, rvec, tvec in zip(objects, pixels, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, distortion)
        error = np.asarray(img).reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(error**2, axis=1)))))
    return {
        "schema_version": 1,
        "timestamp_utc": timestamp(),
        "image_size": list(image_size),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "rms_px": float(rms),
        "per_view_rms_px": errors,
        "valid_views": len(objects),
        "quality_pass": bool(np.isfinite(rms) and rms <= 0.8 and max(errors) <= 1.5),
        "quality_note": "Reprojection gate only; does not validate metric depth or world pose.",
    }


def solve_directory(images: Path, board_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {output}")
    spec = read_json(board_path)
    board = make_board(spec)
    session = read_json(images / "session.json") if (images / "session.json").exists() else None
    require_focus = bool(session and session.get("focus_lock_required"))
    session_focus = None
    objects, pixels, accepted, rejected, signatures = [], [], [], [], []
    image_size = None
    files = sorted(p for p in images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    for path in files:
        frame = read_image(path)
        if frame is None:
            rejected.append({"file": path.name, "reason": "unreadable"})
            continue
        if require_focus:
            try:
                frame_record = read_json(path.with_suffix(".json"))
                target = validate_frame_focus(
                    frame_record["focus_before_frame"], frame_record["focus_after_frame"]
                )
                if session_focus is not None and target != session_focus:
                    raise ValueError("Mixed focus targets in a single session")
                if frame_record["image_file"] != path.name or frame_record["image_size"] != [
                    frame.shape[1],
                    frame.shape[0],
                ]:
                    raise ValueError("Focus record does not match image")
                session_focus = target
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejected.append({"file": path.name, "reason": f"focus provenance: {error}"})
                continue
        size = (frame.shape[1], frame.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError("Calibration images have mixed resolutions; use a single camera mode")
        image_size = size
        detection = detect_board(frame, board)
        if detection is None:
            rejected.append({"file": path.name, "reason": "insufficient/noncollinear corners"})
            continue
        obj, img, _, ids = detection
        # Reject near-identical captures of the same observed corners. Do not let
        # 30 duplicate images masquerade as 30 different calibration views.
        signature = (ids.ravel(), img.reshape(-1, 2))
        duplicate = any(
            np.array_equal(old_ids, signature[0])
            and np.linalg.norm(old_img - signature[1], axis=1).mean() < 5.0
            for old_ids, old_img in signatures
        )
        if duplicate:
            rejected.append({"file": path.name, "reason": "near-duplicate viewpoint"})
            continue
        signatures.append(signature)
        objects.append(obj)
        pixels.append(img)
        accepted.append(path.name)
    if image_size is None:
        raise ValueError("No readable calibration images")
    result = solve_observations(objects, pixels, image_size)
    result.update(
        {
            "source": "charuco_images",
            "board": spec,
            "accepted_images": accepted,
            "rejected_images": rejected,
            "capture_session": session,
            "focus_lock_verified_for_accepted_frames": require_focus,
            "fixed_focus_driver_units": session_focus,
        }
    )
    write_json(output, result)
    return result


def load_calibration(path: Path) -> dict:
    record = read_json(path)
    validate_calibration(record)
    return record


def validate_calibration(record: dict) -> None:
    matrix = np.asarray(record["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(record["distortion_coefficients"], dtype=np.float64)
    size = record["image_size"]
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or matrix[0, 0] <= 0
        or matrix[1, 1] <= 0
        or not np.allclose(matrix[2], [0, 0, 1])
        or abs(matrix[0, 1]) > 1e-9
        or abs(matrix[1, 0]) > 1e-9
        or distortion.ndim != 1
        or distortion.size not in (4, 5, 8, 12, 14)
        or not np.isfinite(distortion).all()
        or len(size) != 2
        or any(not isinstance(x, int) or x <= 0 for x in size)
    ):
        raise ValueError("Invalid camera calibration arrays")
    if record.get("source") != "charuco_images" or record.get("quality_pass") is not True:
        raise ValueError(
            "Calibration must come from accepted ChArUco images and pass the quality gate"
        )
    errors = np.asarray(record.get("per_view_rms_px", []), dtype=float)
    rms = record.get("rms_px", float("inf"))
    if (
        errors.ndim != 1
        or len(errors) < 12
        or not np.isfinite(errors).all()
        or np.any(errors < 0)
        or np.any(errors > 1.5)
        or record.get("valid_views") != len(errors)
        or not np.isfinite(rms)
        or not 0 <= rms <= 0.8
    ):
        raise ValueError("Calibration quality statistics are missing or failed")


def prepare_image(image: np.ndarray, calibration: dict | None):
    if calibration is None:
        return (
            image,
            np.ones(image.shape[:2], dtype=bool),
            {
                "image_space": "raw",
                "camera_matrix": None,
                "calibrated": False,
            },
        )
    height, width = image.shape[:2]
    if [width, height] != calibration["image_size"]:
        raise ValueError("Image resolution does not match calibration; reselect the camera mode")
    matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion_coefficients"], dtype=np.float64)
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, (width, height), 0)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, None, new_matrix, (width, height), cv2.CV_32FC1
    )
    valid = (map_x >= 0) & (map_y >= 0) & (map_x < width - 1) & (map_y < height - 1)
    corrected = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return (
        corrected,
        valid,
        {
            "image_space": "undistorted",
            "camera_matrix": new_matrix.tolist(),
            "calibrated": True,
            "calibration_rms_px": calibration["rms_px"],
        },
    )


def save_calibration_frame(output_dir, count, frame, before, after, received_at, corner_count):
    target = validate_frame_focus(before, after)
    image_path = output_dir / f"view_{count:03d}.png"
    record_path = image_path.with_suffix(".json")
    if image_path.exists() or record_path.exists():
        raise FileExistsError("Refusing to overwrite a captured frame")
    write_image(image_path, frame)
    write_json(
        record_path,
        {
            "schema_version": 1,
            "image_file": image_path.name,
            "image_size": [frame.shape[1], frame.shape[0]],
            "frame_received_at_utc": received_at,
            "saved_at_utc": timestamp(),
            "timestamp_note": "Host receive time; not a hardware exposure timestamp",
            "detected_charuco_corners": corner_count,
            "fixed_focus_driver_units": target,
            "focus_before_frame": before,
            "focus_after_frame": after,
        },
    )


def collect_frames(
    board_path: Path,
    output_dir: Path,
    camera_id: int,
    width: int,
    height: int,
    fixed_focus: float | None = None,
):
    from .webcam_demo import open_camera

    if fixed_focus is not None and (not np.isfinite(fixed_focus) or fixed_focus < 0):
        raise ValueError("Fixed focus must be finite and nonnegative (driver units)")
    board = make_board(read_json(board_path))
    output_dir.mkdir(parents=True, exist_ok=False)
    camera = open_camera(camera_id, width, height)
    count = 0
    try:
        focus = FocusLock(camera)
        if fixed_focus is None:
            focus.enable_auto()
            print(
                "Place the board and wait until sharp. F locks focus, A retries autofocus, S saves, Q exits.",
                flush=True,
            )
        else:
            if not focus.request_lock(fixed_focus):
                raise ValueError(f"Cannot restore fixed focus {fixed_focus}: {focus.message}")
            print(
                f"Fixed focus {fixed_focus:g} requested; wait for green LOCKED. "
                "S saves, Q exits. A/F are disabled.",
                flush=True,
            )
        print("After the first saved photo, focus cannot be changed in this session.", flush=True)
        cv2.namedWindow("ChArUco capture", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("ChArUco capture", 1280, 720)
        previous_state = None
        while True:
            before = focus.observe(new_frame=False)
            ok, frame = camera.read()
            received_at = timestamp()
            after = focus.observe(new_frame=True)
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Camera resolution changed; start a new calibration session")
            if focus.state != previous_state:
                print(
                    f"Focus: {focus.state} | target={focus.target} | "
                    f"value={after['focus']} | autofocus={after['autofocus']} | "
                    f"stable_frames={after['stable_samples']}",
                    flush=True,
                )
                previous_state = focus.state
                if focus.state == "LOCKED" and not (output_dir / "focus-lock.json").exists():
                    write_json(
                        output_dir / "focus-lock.json", {"timestamp_utc": timestamp(), **after}
                    )
            if count == 0 and not (output_dir / "session.json").exists():
                write_json(
                    output_dir / "session.json",
                    {
                        "timestamp_utc": timestamp(),
                        "schema_version": 2,
                        "focus_lock_required": True,
                        "focus_policy": "F requests lock; 8 stable frames and 0.75 s required; no refocus after first save",
                        "fixed_focus_requested": fixed_focus,
                        "focus_controls_disabled": fixed_focus is not None,
                        "camera_index": camera_id,
                        "actual_image_size": [frame.shape[1], frame.shape[0]],
                        "requested_image_size": [width, height],
                        "fps_reported": camera.get(cv2.CAP_PROP_FPS),
                        "fourcc_reported": int(camera.get(cv2.CAP_PROP_FOURCC)),
                        "autofocus_reported": camera.get(cv2.CAP_PROP_AUTOFOCUS),
                        "focus_reported": camera.get(cv2.CAP_PROP_FOCUS),
                        "camera_settings_note": "Driver-reported values may be unsupported. Keep focus fixed.",
                        "board": read_json(board_path),
                    },
                )
            detection = detect_board(frame, board)
            view = frame.copy()
            if detection is not None:
                cv2.aruco.drawDetectedCornersCharuco(view, detection[2], detection[3])
            overlay_scale = max(1.0, width / 1280)
            cv2.putText(
                view,
                f"Saved {count} | {width}x{height} | S save (change angle) | Q quit",
                (int(15 * overlay_scale), int(35 * overlay_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7 * overlay_scale,
                (0, 220, 0),
                max(2, int(2 * overlay_scale)),
            )
            color = (0, 220, 0) if focus.state == "LOCKED" else (0, 180, 255)
            if focus.state == "FAILED":
                color = (0, 0, 255)
            for row, message in enumerate(
                (
                    f"Focus: {focus.state} | value={after['focus']} | "
                    + (
                        "FIXED - A/F disabled"
                        if fixed_focus is not None
                        else "A autofocus | F lock"
                    ),
                    focus.message,
                )
            ):
                cv2.putText(
                    view,
                    message,
                    (int(15 * overlay_scale), int((70 + row * 30) * overlay_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65 * overlay_scale,
                    color,
                    max(2, int(2 * overlay_scale)),
                    cv2.LINE_AA,
                )
            cv2.imshow("ChArUco capture", view)
            key = cv2.waitKey(10) & 0xFF
            if (
                key in (ord("q"), 27)
                or cv2.getWindowProperty("ChArUco capture", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            if key in (ord("f"), ord("a")):
                if count or fixed_focus is not None:
                    print(
                        "Focus is frozen for this session. Q to exit and start a new session before refocusing.",
                        flush=True,
                    )
                else:
                    focus.request_lock() if key == ord("f") else focus.enable_auto()
                    print(f"Focus: {focus.state} - {focus.message}", flush=True)
            if key == ord("s"):
                latest = focus.observe(new_frame=False)
                try:
                    validate_frame_focus(before, after)
                    validate_frame_focus(after, latest)
                except ValueError:
                    print(
                        "Save blocked: fixed focus not verified; wait for LOCKED or restart if FAILED."
                        if fixed_focus is not None
                        else "Save blocked: press F and wait for green LOCKED before taking the next frame.",
                        flush=True,
                    )
                    continue
                if detection is None:
                    print("Need at least 10 non-collinear detected ChArUco corners")
                else:
                    save_calibration_frame(
                        output_dir, count, frame, before, after, received_at, len(detection[3])
                    )
                    count += 1
                    print(f"Saved {count} photos; locked focus={focus.target}", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {count} views: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChArUco board, capture and intrinsics calibration"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    board_cmd = sub.add_parser("board", help="generate printable A4 board at 100 percent scale")
    board_cmd.add_argument("--output-dir", type=Path, default=CALIBRATION_DIR / "board")
    capture = sub.add_parser("collect", help="manual S-to-save capture; requires a physical board")
    capture.add_argument("--board-json", type=Path, default=DEFAULT_BOARD_PATH)
    capture.add_argument(
        "--output-dir",
        type=Path,
        default=CALIBRATION_DIR
        / "sessions"
        / datetime.now(UTC).strftime("session_%Y%m%d_%H%M%S_%f"),
    )
    capture.add_argument("--camera", type=int, default=0)
    capture.add_argument("--width", type=int, default=DEFAULT_CAMERA_WIDTH)
    capture.add_argument("--height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    capture.add_argument(
        "--focus", type=float, help="Restore a fixed focus in driver units; disables A/F refocusing"
    )
    solve = sub.add_parser("solve", help="solve from collected views; refuses insufficient data")
    solve.add_argument("--images", type=Path, required=True)
    solve.add_argument("--board-json", type=Path, default=DEFAULT_BOARD_PATH)
    solve.add_argument("--output", type=Path, default=CALIBRATION_DIR / "camera.json")
    args = parser.parse_args()
    try:
        if args.command == "board":
            print(generate_board(args.output_dir))
        elif args.command == "collect":
            collect_frames(
                args.board_json, args.output_dir, args.camera, args.width, args.height, args.focus
            )
        else:
            result = solve_directory(args.images, args.board_json, args.output)
            print(f"RMS: {result['rms_px']:.3f} px; quality_pass={result['quality_pass']}")
            print(args.output.resolve())
            if not result["quality_pass"]:
                raise SystemExit(2)
    except (ValueError, OSError, KeyError, cv2.error) as error:
        parser.exit(2, f"Calibration refused: {error}\n")
