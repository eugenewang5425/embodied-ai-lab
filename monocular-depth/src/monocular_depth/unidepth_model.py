"""UniDepth V2 integration: metric depth as a first-class pipeline variant.

UniDepth V2 (lpiccinelli-eth/UniDepth, CC-BY-NC-4.0) takes the undistorted
RGB plus the rectified intrinsics and returns per-pixel metric depth. It is
the recommended metric model for this lab after the DA-V2 benchmark (raw
relative error +67% -> +12% median on our board ground truth set).

License note: research/non-commercial only. See docs/depth-model-benchmark.
"""

from __future__ import annotations

import hashlib
import sys
import types

import cv2
import numpy as np

from .config import PROJECT_ROOT

UNIDEPTH_REPO = PROJECT_ROOT / "bench" / "third_party" / "UniDepth"
UNIDEPTH_WEIGHTS = PROJECT_ROOT / "bench" / "weights" / "unidepth_v2_vits14.bin"
WEIGHT_SHA256 = "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"


def load_unidepth(device=None):
    """Load the UniDepth V2 ViT-S14 model; returns (model, device)."""
    import torch

    if sys.modules.get("wandb") is None:
        # UniDepth's visualization helpers import wandb; only needed for training logs.
        sys.modules["wandb"] = types.ModuleType("wandb")
    repo = UNIDEPTH_REPO.resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"UniDepth source not found: {repo}")
    if not UNIDEPTH_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"UniDepth weights not found: {UNIDEPTH_WEIGHTS}\n"
            "Download from https://huggingface.co/lpiccinelli/unidepth-v2-vits14"
        )
    with UNIDEPTH_WEIGHTS.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != WEIGHT_SHA256:
        raise ValueError("UniDepth weight SHA256 mismatch; refusing to load")

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "unidepth"))
    try:
        from hubconf import UniDepth
        from safetensors.torch import load_file
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "UniDepth needs timm, einops, safetensors; install them in the "
            "environment that runs this lab"
        ) from error
    model = UniDepth(version="v2", backbone="vits14", pretrained=False)
    state = load_file(str(UNIDEPTH_WEIGHTS))
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"UniDepth weight load incomplete: {len(result.missing_keys)} missing, "
            f"{len(result.unexpected_keys)} unexpected"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, device


def infer_unidepth(model, bgr_image, matrix) -> np.ndarray:
    """Run UniDepth V2 on one BGR frame with the given intrinsics (meters)."""
    import torch
    from unidepth.utils.camera import Pinhole

    device = next(model.parameters()).device
    image = torch.from_numpy(bgr_image[:, :, ::-1].copy()).permute(2, 0, 1).contiguous().to(device)
    camera = Pinhole(K=torch.tensor(np.asarray(matrix, dtype=np.float32), device=device))
    with torch.no_grad():
        preds = model.infer(image, camera)
    depth = preds["depth"]
    if torch.is_tensor(depth):
        depth = depth.cpu().numpy()
    depth = np.asarray(depth).squeeze()
    if depth.ndim != 2:
        depth = depth[0]
    if depth.shape != bgr_image.shape[:2]:
        depth = cv2.resize(
            depth.astype(np.float32),
            (bgr_image.shape[1], bgr_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return depth.astype(np.float32, copy=False)


class UniDepthPipeline:
    """DepthPipeline-compatible wrapper: calibration + UniDepth inference."""

    def __init__(self, model, calibration):
        from .calibration import load_calibration

        self.model = model
        self.calibration = load_calibration(calibration) if calibration else None
        with UNIDEPTH_WEIGHTS.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.provenance = {
            "model": "UniDepth V2 (ViT-S14)",
            "checkpoint_name": UNIDEPTH_WEIGHTS.name,
            "checkpoint_sha256": digest,
            "input_size": None,
            "calibration": self.calibration,
            "calibration_file": str(calibration.resolve()) if calibration else None,
            "depth_origin": "unidepth_v2",
        }

    def process(self, raw_image):
        from .calibration import prepare_image

        image, valid, camera_info = prepare_image(raw_image, self.calibration)
        depth = infer_unidepth(self.model, image, camera_info["camera_matrix"])
        if camera_info.get("camera_matrix") is not None:
            depth[~valid] = np.nan
        return image, depth, {**self.provenance, **camera_info}
