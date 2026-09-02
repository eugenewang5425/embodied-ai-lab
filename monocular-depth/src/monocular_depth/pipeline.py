"""Shared preprocessing and provenance for the image and camera entry points."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .calibration import load_calibration, prepare_image
from .config import METRIC_REVISION, UPSTREAM_COMMIT, checkpoint_for
from .model import infer


class DepthPipeline:
    def __init__(self, model, variant: str, input_size: int, calibration_path: Path | None):
        self.model, self.variant, self.input_size = model, variant, input_size
        self.calibration = load_calibration(calibration_path) if calibration_path else None
        checkpoint = checkpoint_for(variant)
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.provenance = {
            "model": "Depth Anything V2 Small"
            + (" Metric Hypersim" if variant == "metric" else ""),
            "checkpoint_name": checkpoint.name,
            "checkpoint_sha256": digest,
            "checkpoint_revision": METRIC_REVISION if variant == "metric" else None,
            "upstream_expected_commit": UPSTREAM_COMMIT,
            "input_size": input_size,
            "calibration": self.calibration,
            "calibration_file": str(calibration_path.resolve()) if calibration_path else None,
        }

    def process(self, raw_image):
        image, valid, camera_info = prepare_image(raw_image, self.calibration)
        depth = infer(self.model, image, self.input_size)
        depth[~valid] = np.nan
        return image, depth, {**self.provenance, **camera_info}
