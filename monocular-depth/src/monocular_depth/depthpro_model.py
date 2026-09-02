"""Depth Pro (Apple) integration: metric depth as a pipeline variant.

Depth Pro is the quality upper bound from our benchmark (raw relative error
median -2.8% over 0.4-0.9 m, fp16@1536, 0.76 s/frame, 3.97 GB peak on the
RTX 5070 Laptop). License: Apple Machine Learning Research Model (research
only, not for commercial use).
"""

from __future__ import annotations

import copy
import hashlib

import cv2
import numpy as np

from .config import PROJECT_ROOT

MODEL_WEIGHTS = PROJECT_ROOT / "bench" / "weights" / "depth_pro.pt"
WEIGHT_SHA256 = "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"


def load_depthpro():
    """Load Depth Pro; returns (model, transform). Requires weights and deps."""
    import torch

    if not MODEL_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"Depth Pro weights not found: {MODEL_WEIGHTS}\n"
            "Download from https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt"
        )
    with MODEL_WEIGHTS.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError("Depth Pro weight SHA256 mismatch; refusing to load")
    try:
        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Depth Pro needs the 'depth-pro' package in the running environment"
        ) from error
    config = copy.copy(DEFAULT_MONODEPTH_CONFIG_DICT)
    config.checkpoint_uri = str(MODEL_WEIGHTS)
    model, transform = create_model_and_transforms(
        config=config, device="cuda" if torch.cuda.is_available() else "cpu",
        precision=torch.half,
    )
    model.eval()
    return model, transform


def infer_depthpro(model, transform, bgr_image, matrix) -> np.ndarray:
    """Metric depth for one BGR frame using the calibrated focal length."""
    import torch

    image = transform(bgr_image[:, :, ::-1].copy())
    with torch.no_grad():
        pred = model.infer(image, np.asarray(matrix, dtype=np.float64)[0, 0])
    torch.cuda.synchronize()
    depth = np.asarray(pred["depth"].cpu().numpy()).squeeze()
    if depth.ndim != 2 or depth.shape != bgr_image.shape[:2]:
        depth = cv2.resize(
            depth.astype(np.float32),
            (bgr_image.shape[1], bgr_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return depth.astype(np.float32, copy=False)


class DepthProPipeline:
    """DepthPipeline-compatible wrapper: calibration + Depth Pro inference."""

    def __init__(self, model, transform, calibration):
        from .calibration import load_calibration

        self.model, self.transform = model, transform
        self.calibration = load_calibration(calibration) if calibration else None
        with MODEL_WEIGHTS.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.provenance = {
            "model": "Depth Pro (ViT-L16, fp16@1536)",
            "checkpoint_name": MODEL_WEIGHTS.name,
            "checkpoint_sha256": digest,
            "input_size": 1536,
            "calibration": self.calibration,
            "calibration_file": str(calibration.resolve()) if calibration else None,
            "depth_origin": "depth_pro",
        }

    def process(self, raw_image):
        from .calibration import prepare_image

        image, valid, camera_info = prepare_image(raw_image, self.calibration)
        depth = infer_depthpro(self.model, self.transform, image, camera_info["camera_matrix"])
        if camera_info.get("camera_matrix") is not None:
            depth[~valid] = np.nan
        return image, depth, {**self.provenance, **camera_info}
