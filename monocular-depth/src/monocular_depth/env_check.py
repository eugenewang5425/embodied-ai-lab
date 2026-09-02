from __future__ import annotations

import argparse
import sys

import cv2
import torch

from .config import UPSTREAM_ROOT, VARIANTS, checkpoint_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="relative")
    args = parser.parse_args()
    checkpoint = checkpoint_for(args.variant)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GiB")
    print(f"Upstream source: {UPSTREAM_ROOT} ({'ok' if UPSTREAM_ROOT.is_dir() else 'missing'})")
    print(f"Checkpoint: {checkpoint} ({'ok' if checkpoint.is_file() else 'missing'})")

    if not torch.cuda.is_available() or not UPSTREAM_ROOT.is_dir() or not checkpoint.is_file():
        raise SystemExit(1)
