from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .config import METRIC_SHA256, MODEL_CONFIG, UPSTREAM_ROOT, checkpoint_for


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    checkpoint: Path | None = None,
    upstream_root: Path = UPSTREAM_ROOT,
    variant: str = "relative",
) -> tuple[Any, torch.device]:
    checkpoint = checkpoint or checkpoint_for(variant)
    if not upstream_root.is_dir():
        raise FileNotFoundError(
            f"Depth Anything V2 source not found: {upstream_root}\n"
            "Clone the official repository as described in README.md."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}\n"
            f"Run: uv run --no-editable depth-download --variant {variant}"
        )
    if variant == "metric":
        with checkpoint.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != METRIC_SHA256:
                raise ValueError("Metric checkpoint SHA256 mismatch; refusing to load")

    model_class = upstream_model_class(upstream_root, variant)
    options = {**MODEL_CONFIG, **({"max_depth": 20.0} if variant == "metric" else {})}
    model = model_class(**options)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    device = select_device()
    model = model.to(device).eval()
    model.depth_variant = variant
    return model, device


def upstream_model_class(upstream_root: Path, variant: str) -> Any:
    checkpoint_for(variant)  # Validate even when a custom checkpoint is supplied.
    code_root = upstream_root / ("metric_depth" if variant == "metric" else "")
    package_path = (code_root / "depth_anything_v2").resolve()
    if not (package_path / "dpt.py").is_file():
        raise FileNotFoundError(f"Missing upstream model code: {package_path}")
    # The official relative and metric packages have the same name. Isolate them
    # so loading both in one process cannot silently select the wrong head.
    suffix = hashlib.sha256(str(package_path).encode()).hexdigest()[:12]
    name = f"_da2_{variant}_{suffix}"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        sys.modules[name] = package
    return importlib.import_module(f"{name}.dpt").DepthAnythingV2


def infer(model: Any, bgr_image: np.ndarray, input_size: int = 518) -> np.ndarray:
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Input image is empty")
    if input_size < 28:
        raise ValueError("Input size must be at least 28 pixels")
    return model.infer_image(bgr_image, input_size).astype(np.float32, copy=False)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    if not finite.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    minimum = float(depth[finite].min())
    maximum = float(depth[finite].max())
    if maximum - minimum < 1e-8:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = (depth - minimum) / (maximum - minimum)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(normalize_depth(depth), cv2.COLORMAP_INFERNO)


def comparison_view(bgr_image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    colored = colorize_depth(depth)
    if colored.shape[:2] != bgr_image.shape[:2]:
        colored = cv2.resize(colored, (bgr_image.shape[1], bgr_image.shape[0]))
    separator = np.full((bgr_image.shape[0], 12, 3), 255, dtype=np.uint8)
    return cv2.hconcat([bgr_image, separator, colored])
