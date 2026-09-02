from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .model import comparison_view, normalize_depth
from .records import timestamp, write_json


def read_image(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix or ".png"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise OSError(f"Could not encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(path)
    except OSError as error:
        raise OSError(f"Could not write {path}") from error


def save_result(
    output_dir: Path,
    stem: str,
    image: np.ndarray,
    depth: np.ndarray,
    *,
    variant: str = "relative",
    metadata: dict | None = None,
    raw_image: np.ndarray | None = None,
) -> list[Path]:
    if variant not in ("relative", "metric"):
        raise ValueError("Unknown depth variant")
    if image.shape[:2] != depth.shape or depth.ndim != 2:
        raise ValueError("RGB and depth resolutions must match")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"{stem}_rgb.png"
    depth_array_path = output_dir / f"{stem}_depth.npy"
    depth_image_path = output_dir / f"{stem}_depth.png"
    comparison_path = output_dir / f"{stem}_comparison.png"
    record_path = output_dir / f"{stem}_metadata.json"
    raw_path = output_dir / f"{stem}_raw.png" if raw_image is not None else None
    paths = [rgb_path, depth_array_path, depth_image_path, comparison_path]
    if raw_path is not None:
        paths.append(raw_path)
    paths.append(record_path)
    if any(path.exists() for path in paths):
        raise FileExistsError("Output already exists; choose a new --output-dir or filename")
    record = {
        "image_space": "raw",
        "calibrated": False,
        "camera_matrix": None,
        **(metadata or {}),
        "schema_version": 1,
        "timestamp_utc": timestamp(),
        "variant": variant,
        "unit": "m" if variant == "metric" else "relative",
        "image_size": [image.shape[1], image.shape[0]],
        "rgb_file": rgb_path.name,
        "depth_file": depth_array_path.name,
        "raw_file": raw_path.name if raw_path is not None else rgb_path.name,
        "depth_definition": "predicted optical-axis Z" if variant == "metric" else "relative score",
        "metric_accuracy_validated": False,
        "world_pose": None,
        "confidence_available": False,
        "visualization": "per-frame min/max; PNG colors are not distance measurements",
    }

    write_image(rgb_path, image)
    np.save(depth_array_path, depth)
    write_image(depth_image_path, normalize_depth(depth))
    write_image(comparison_path, comparison_view(image, depth))
    if raw_path is not None:
        write_image(raw_path, raw_image)
    write_json(record_path, record)
    return paths
